//! Tiny Cholesky solver used by `curve2coef` at init time.
//!
//! Matrices here are small (typically `n_basis × n_basis`, e.g. 8×8 for DDR
//! scale) and symmetric positive (semi)definite by construction (Gram matrices
//! of B-spline basis evaluations, plus a Tikhonov ridge). No LAPACK linkage.
//!
//! For larger `grid + k` configurations the basis matrix `M` may be
//! rank-deficient (`batch = grid + 1 < n_basis = grid + k` when `k >= 1`).
//! The base ridge of 1e-8 then becomes insufficient in f32 accumulation, so
//! `cholesky_solve_robust` retries with progressively larger ridges before
//! giving up.

use ndarray::{Array2, ArrayView2};

/// In-place Cholesky factorization: `A = L · L^T` where `L` is lower-triangular.
/// On success, overwrites `a` so its lower triangle holds `L` (upper triangle
/// is zeroed). On failure, returns `Err((j, diag))` where `j` is the index
/// where the diagonal went non-positive and `diag` is the Schur complement
/// that caused the failure.
fn try_cholesky_in_place(a: &mut Array2<f32>) -> Result<(), (usize, f32)> {
    let n = a.shape()[0];
    assert_eq!(a.shape()[1], n, "cholesky requires a square matrix");

    for j in 0..n {
        let mut diag = a[[j, j]];
        for k in 0..j {
            diag -= a[[j, k]] * a[[j, k]];
        }
        if diag <= 0.0 {
            return Err((j, diag));
        }
        let ljj = diag.sqrt();
        a[[j, j]] = ljj;
        for i in (j + 1)..n {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= a[[i, k]] * a[[j, k]];
            }
            a[[i, j]] = s / ljj;
        }
    }
    // zero the strict upper triangle (cosmetic; back-substitution ignores it)
    for i in 0..n {
        for j in (i + 1)..n {
            a[[i, j]] = 0.0;
        }
    }
    Ok(())
}

/// Solve `L · y = b` (forward substitution). `L` lower-triangular `[n, n]`,
/// `b` shape `[n, m]`. Overwrites `b` in place with `y`.
fn forward_sub(l: &Array2<f32>, b: &mut Array2<f32>) {
    let n = l.shape()[0];
    let m = b.shape()[1];
    for col in 0..m {
        for i in 0..n {
            let mut s = b[[i, col]];
            for k in 0..i {
                s -= l[[i, k]] * b[[k, col]];
            }
            b[[i, col]] = s / l[[i, i]];
        }
    }
}

/// Solve `L^T · x = y` (back substitution). `L` lower-triangular `[n, n]`,
/// `y` shape `[n, m]`. Overwrites `y` in place with `x`.
fn back_sub(l: &Array2<f32>, y: &mut Array2<f32>) {
    let n = l.shape()[0];
    let m = y.shape()[1];
    for col in 0..m {
        for i in (0..n).rev() {
            let mut s = y[[i, col]];
            for k in (i + 1)..n {
                s -= l[[k, i]] * y[[k, col]];
            }
            y[[i, col]] = s / l[[i, i]];
        }
    }
}

/// Solve `A · x = b` for symmetric positive (semi)definite `A` via Cholesky.
///
/// `a` shape `[n, n]`, `b` shape `[n, m]`. Returns `x` shape `[n, m]`.
/// Panics if `A` is not SPD (caller must add a ridge if needed). Kept for
/// backwards compatibility with `curve2coef`'s previous code path; the
/// robust path is `cholesky_solve_robust`.
pub fn cholesky_solve(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    let n = a.shape()[0];
    assert_eq!(a.shape()[1], n, "A must be square");
    assert_eq!(b.shape()[0], n, "b row count must equal A's size");

    let mut l = a.to_owned();
    match try_cholesky_in_place(&mut l) {
        Ok(()) => {}
        Err((j, diag)) => panic!(
            "cholesky: non-SPD pivot at index {j} (Schur complement = {diag}); add a ridge"
        ),
    }

    let mut x = b.to_owned();
    forward_sub(&l, &mut x);
    back_sub(&l, &mut x);
    x
}

/// Solve `A · x = b` with **adaptive Tikhonov ridging**.
///
/// Starts with `base_ridge` added to the diagonal of `A`; if the Cholesky
/// factorisation fails (non-positive pivot in f32), retries with
/// progressively larger ridges (`1×`, `100×`, `10_000×`, `1_000_000×`
/// `base_ridge`) before giving up. The progression covers the
/// rank-deficiency cases that arise from B-spline `curve2coef` when
/// `batch < n_basis`.
///
/// For well-conditioned matrices, only the first attempt is made — output
/// is bit-equivalent to the previous `cholesky_solve` behaviour at
/// `ridge = base_ridge`.
pub fn cholesky_solve_robust(
    a: ArrayView2<f32>,
    b: ArrayView2<f32>,
    base_ridge: f32,
) -> Array2<f32> {
    let n = a.shape()[0];
    assert_eq!(a.shape()[1], n, "A must be square");
    assert_eq!(b.shape()[0], n, "b row count must equal A's size");

    const RIDGE_MULTIPLIERS: [f32; 4] = [1.0, 100.0, 10_000.0, 1_000_000.0];
    for (attempt, &mult) in RIDGE_MULTIPLIERS.iter().enumerate() {
        let ridge = base_ridge * mult;
        let mut l = a.to_owned();
        for i in 0..n {
            l[[i, i]] += ridge;
        }
        if try_cholesky_in_place(&mut l).is_ok() {
            let mut x = b.to_owned();
            forward_sub(&l, &mut x);
            back_sub(&l, &mut x);
            if attempt > 0 {
                // Surface so downstream tests can verify the well-conditioned
                // code path is hit on small grids. eprintln is intentional:
                // init is one-shot, this is informational, not a hot loop.
                eprintln!(
                    "rskan: cholesky_solve_robust escalated to ridge {ridge:e} \
                     (attempt {})",
                    attempt + 1,
                );
            }
            return x;
        }
    }
    panic!(
        "cholesky_solve_robust: matrix non-SPD even at ridge = {} (base × 1e6)",
        base_ridge * RIDGE_MULTIPLIERS[RIDGE_MULTIPLIERS.len() - 1]
    );
}
