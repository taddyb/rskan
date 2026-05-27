//! B-spline math: `extend_grid`, `b_batch`, `coef2curve`, `curve2coef`.

use ndarray::{Array2, ArrayView2};

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
