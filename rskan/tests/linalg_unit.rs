//! Cholesky solver unit tests.

use approx::assert_abs_diff_eq;
use ndarray::{array, Array2};
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

/// `cholesky_solve_robust` rescues rank-deficient matrices by escalating
/// the Tikhonov ridge. A rank-1 2×2 matrix is the smallest example that
/// reliably exercises the escalation path in f32.
#[test]
fn cholesky_solve_robust_handles_rank_deficient_matrix() {
    use ndarray::array;

    // Symmetric rank-1: eigenvalues {2, 0}.
    let a = array![[1.0_f32, 1.0], [1.0, 1.0]];
    let b = array![[1.0_f32], [1.0]];

    // base_ridge = 1e-8: attempt 1 may or may not succeed in f32 (the
    // ridge floor is close to the rounding floor). The robust solver
    // escalates if needed and must produce a finite result.
    let x = linalg::cholesky_solve_robust(a.view(), b.view(), 1e-8);

    assert_eq!(x.shape(), &[2, 1]);
    for &v in x.iter() {
        assert!(v.is_finite(), "robust solve produced non-finite element: {v}");
    }

    // (A + λI) x ≈ b for the actual ridge used. We don't know which ridge
    // attempt won, but at any ridge ≤ 1e-2, the residual should be small.
    // Use a 1e-1 tolerance to cover the worst escalation case.
    let lhs = a.dot(&x);
    let residual: f32 = (&lhs - &b).iter().map(|v| v.abs()).sum();
    assert!(
        residual < 1e-1,
        "(A + λI) x ≈ b residual too large: {residual}"
    );
}

/// `cholesky_solve_robust` is bit-equivalent to `cholesky_solve` on
/// well-conditioned matrices — i.e. attempt 1 wins, no escalation.
#[test]
fn cholesky_solve_robust_matches_plain_on_well_conditioned() {
    use ndarray::array;
    // Symmetric positive-definite 3×3, well-conditioned enough that
    // base_ridge = 1e-8 succeeds on first attempt.
    let a = array![
        [4.0_f32, 1.0, 0.0],
        [1.0,     3.0, 1.0],
        [0.0,     1.0, 2.0],
    ];
    let b = array![[1.0_f32], [2.0], [3.0]];

    // Apply the same 1e-8 ridge that the robust solver uses on attempt 1.
    let mut a_ridge = a.clone();
    for i in 0..3 { a_ridge[[i, i]] += 1e-8; }
    let plain = linalg::cholesky_solve(a_ridge.view(), b.view());

    let robust = linalg::cholesky_solve_robust(a.view(), b.view(), 1e-8);

    for (p, r) in plain.iter().zip(robust.iter()) {
        let diff = (p - r).abs();
        assert!(diff < 1e-6, "robust drifted from plain at well-conditioned input: |{p} - {r}| = {diff}");
    }
}
