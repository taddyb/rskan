//! Tiny Cholesky solver used by `curve2coef` at init time.
//!
//! Matrices here are small (typically `n_basis × n_basis`, e.g. 8×8 for DDR
//! scale) and symmetric positive definite by construction (Gram matrices of
//! B-spline basis evaluations, plus a Tikhonov ridge). No LAPACK linkage.

use ndarray::{Array2, ArrayView2};

/// In-place Cholesky factorization: `A = L · L^T` where `L` is lower-triangular.
/// Overwrites `a` so its lower triangle holds `L`. Panics if `A` is not SPD.
fn cholesky_in_place(a: &mut Array2<f32>) {
    let n = a.shape()[0];
    assert_eq!(a.shape()[1], n, "cholesky requires a square matrix");

    for j in 0..n {
        let mut diag = a[[j, j]];
        for k in 0..j {
            diag -= a[[j, k]] * a[[j, k]];
        }
        assert!(
            diag > 0.0,
            "cholesky: non-SPD pivot at index {j} ({diag}); add a ridge"
        );
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

/// Solve `A · x = b` for symmetric positive definite `A` via Cholesky.
///
/// `a` shape `[n, n]`, `b` shape `[n, m]`. Returns `x` shape `[n, m]`.
/// Panics if `A` is not SPD (caller must add a ridge if needed).
pub fn cholesky_solve(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    let n = a.shape()[0];
    assert_eq!(a.shape()[1], n, "A must be square");
    assert_eq!(b.shape()[0], n, "b row count must equal A's size");

    let mut l = a.to_owned();
    cholesky_in_place(&mut l);

    let mut x = b.to_owned();
    forward_sub(&l, &mut x);
    back_sub(&l, &mut x);
    x
}
