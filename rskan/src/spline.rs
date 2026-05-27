//! B-spline math: `extend_grid`, `b_batch`, `coef2curve`, `curve2coef`.

use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

use crate::linalg::cholesky_solve;

/// Extend a `[I, G+1]` uniform-linspace grid with `k_extend` ghost knots on each
/// side, spaced by the per-row interval `h_i = (grid[i, -1] - grid[i, 0]) / G`.
///
/// Equivalent to pykan's `spline.extend_grid` (iterative `torch.cat` loop in
/// `spline.py`) — computed analytically here. Output shape `[I, G + 1 + 2·k_extend]`.
pub fn extend_grid(grid: ArrayView2<f32>, k_extend: usize) -> Array2<f32> {
    assert!(grid.shape()[1] >= 2, "grid must have at least 2 knots per row");
    let (rows, n) = (grid.shape()[0], grid.shape()[1]);
    let out_n = n + 2 * k_extend;
    let mut out = Array2::<f32>::zeros((rows, out_n));

    for i in 0..rows {
        let lo = grid[[i, 0]];
        let hi = grid[[i, n - 1]];
        let h = (hi - lo) / (n as f32 - 1.0);

        for j in 0..k_extend {
            out[[i, j]] = lo - ((k_extend - j) as f32) * h;
        }
        for j in 0..n {
            out[[i, k_extend + j]] = grid[[i, j]];
        }
        for j in 0..k_extend {
            out[[i, k_extend + n + j]] = hi + ((j + 1) as f32) * h;
        }
    }
    out
}

/// Tikhonov ridge for the normal-equations substitute of `torch.linalg.lstsq`.
/// Pykan's commented-out fallback (`spline.py`) uses 1e-8; matches our parity
/// floor for non-degenerate B-spline bases.
const CURVE2COEF_RIDGE: f32 = 1e-8;

/// CPU B-spline basis evaluator (the same algorithm as `b_batch`, but on
/// `ndarray` and without autograd). Used by `curve2coef` at init time and by
/// the round-trip test. Output shape `[batch, in_dim, n_basis]` where
/// `n_basis = grid.shape[1] - k - 1`.
///
/// Implements pykan's `spline.B_batch` Cox-de Boor recursion iteratively.
pub fn b_batch_nd(x: ArrayView2<f32>, grid: ArrayView2<f32>, k: usize) -> Array3<f32> {
    let (batch, in_dim) = (x.shape()[0], x.shape()[1]);
    assert_eq!(grid.shape()[0], in_dim, "grid in_dim must match x in_dim");
    let knots = grid.shape()[1];
    assert!(knots > k + 1, "grid must have > k+1 knots; got {knots} with k={k}");

    // Base case (k_curr = 0): indicator [grid[j], grid[j+1])
    let mut v = Array3::<f32>::zeros((batch, in_dim, knots - 1));
    for b in 0..batch {
        for i in 0..in_dim {
            for j in 0..(knots - 1) {
                let xv = x[[b, i]];
                if xv >= grid[[i, j]] && xv < grid[[i, j + 1]] {
                    v[[b, i, j]] = 1.0;
                }
            }
        }
    }

    // Iterative Cox-de Boor: k_curr in 1..=k.
    for k_curr in 1..=k {
        let len_prev = knots - k_curr;
        let mut v_new = Array3::<f32>::zeros((batch, in_dim, len_prev - 1));
        for b in 0..batch {
            for i in 0..in_dim {
                for j in 0..(len_prev - 1) {
                    let g_a = grid[[i, j]];
                    let g_b = grid[[i, j + k_curr]];
                    let g_c = grid[[i, j + k_curr + 1]];
                    let g_d = grid[[i, j + 1]];

                    let lf_num = x[[b, i]] - g_a;
                    let lf_den = g_b - g_a;
                    let rf_num = g_c - x[[b, i]];
                    let rf_den = g_c - g_d;

                    let lf = if lf_den.abs() > 0.0 { lf_num / lf_den } else { 0.0 };
                    let rf = if rf_den.abs() > 0.0 { rf_num / rf_den } else { 0.0 };

                    let val = lf * v[[b, i, j]] + rf * v[[b, i, j + 1]];
                    v_new[[b, i, j]] = if val.is_nan() { 0.0 } else { val };
                }
            }
        }
        v = v_new;
    }
    v
}

/// Solve for B-spline coefficients that fit `y_eval` at `x_eval` knot positions.
///
/// Pykan uses `torch.linalg.lstsq` (SVD-based gelsy). We substitute ridge-
/// regularized normal equations + Cholesky (`λ = 1e-8`), matching to ≤ 1e-6
/// for well-conditioned B-spline bases. Init-only — no autograd.
///
/// - `x_eval` shape `[batch, in_dim]` (typically `grid_inner.T`, `batch = G+1`)
/// - `y_eval` shape `[batch, in_dim, out_dim]` (the `noises` from init)
/// - `grid`   shape `[in_dim, knots = G+1+2k]`
/// - Output   shape `[in_dim, out_dim, n_basis = G+k]`
pub fn curve2coef(
    x_eval: ArrayView2<f32>,
    y_eval: ArrayView3<f32>,
    grid: ArrayView2<f32>,
    k: usize,
) -> Array3<f32> {
    let (batch, in_dim) = (x_eval.shape()[0], x_eval.shape()[1]);
    let out_dim = y_eval.shape()[2];
    let n_basis = grid.shape()[1] - k - 1;

    assert_eq!(y_eval.shape(), &[batch, in_dim, out_dim]);
    assert_eq!(grid.shape()[0], in_dim);

    let mat = b_batch_nd(x_eval, grid, k);                  // [batch, in_dim, n_basis]
    let mut coef = Array3::<f32>::zeros((in_dim, out_dim, n_basis));

    // Per-in_dim solve: same M for all O right-hand-sides.
    for i in 0..in_dim {
        // M = mat[:, i, :]   shape [batch, n_basis]
        let mut m = Array2::<f32>::zeros((batch, n_basis));
        for b in 0..batch {
            for c in 0..n_basis {
                m[[b, c]] = mat[[b, i, c]];
            }
        }
        // MtM = M^T M + λ I       shape [n_basis, n_basis]
        let mut mtm = m.t().dot(&m);
        for d in 0..n_basis {
            mtm[[d, d]] += CURVE2COEF_RIDGE;
        }
        // Y_i = y_eval[:, i, :]   shape [batch, out_dim]
        let mut y_i = Array2::<f32>::zeros((batch, out_dim));
        for b in 0..batch {
            for o in 0..out_dim {
                y_i[[b, o]] = y_eval[[b, i, o]];
            }
        }
        // MtY = M^T Y_i           shape [n_basis, out_dim]
        let mty = m.t().dot(&y_i);
        // Solve MtM · C = MtY     C shape [n_basis, out_dim]
        let c_sol = cholesky_solve(mtm.view(), mty.view());

        for o in 0..out_dim {
            for cidx in 0..n_basis {
                coef[[i, o, cidx]] = c_sol[[cidx, o]];
            }
        }
    }
    coef
}
