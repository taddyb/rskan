//! Cholesky solver unit tests.

use approx::assert_abs_diff_eq;
use ndarray::{array, Array1, Array2};
use rskan::linalg;

#[test]
fn cholesky_solve_2x2_spd() {
    // [[4, 2], [2, 5]] x = [10, 13]
    // Exact solution: x = [2, 1.8] (verify by hand: 4·2 + 2·1.8 = 11.6 — no, redo)
    // Set up: A x = b where x = [1, 2].
    // A x = [[4, 2], [2, 5]] [1, 2]^T = [4+4, 2+10] = [8, 12].
    let a: Array2<f32> = array![[4.0, 2.0], [2.0, 5.0]];
    let b: Array2<f32> = array![[8.0], [12.0]];

    let x = linalg::cholesky_solve(a.view(), b.view());

    assert_eq!(x.shape(), &[2, 1]);
    assert_abs_diff_eq!(x[[0, 0]], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(x[[1, 0]], 2.0, epsilon = 1e-5);
}

#[test]
fn cholesky_solve_multiple_rhs() {
    // Same A, two right-hand sides.
    let a: Array2<f32> = array![[4.0, 2.0], [2.0, 5.0]];
    // A · [[1, 0], [2, 1]] = [[4+4, 0+2], [2+10, 0+5]] = [[8, 2], [12, 5]]
    let b: Array2<f32> = array![[8.0, 2.0], [12.0, 5.0]];

    let x = linalg::cholesky_solve(a.view(), b.view());

    assert_eq!(x.shape(), &[2, 2]);
    assert_abs_diff_eq!(x[[0, 0]], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(x[[1, 0]], 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(x[[0, 1]], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(x[[1, 1]], 1.0, epsilon = 1e-5);
}

#[test]
fn cholesky_solve_3x3_identity_rhs() {
    // A = diag(1, 4, 9); x = A^{-1} b.
    // b = [1, 4, 9] → x = [1, 1, 1]
    let a: Array2<f32> = array![
        [1.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 9.0],
    ];
    let b: Array2<f32> = array![[1.0], [4.0], [9.0]];

    let x = linalg::cholesky_solve(a.view(), b.view());

    for i in 0..3 {
        assert_abs_diff_eq!(x[[i, 0]], 1.0, epsilon = 1e-5);
    }
}
