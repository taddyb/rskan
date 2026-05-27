//! Analytical unit tests for spline primitives (no pykan fixtures).

use approx::assert_abs_diff_eq;
use ndarray::{array, Array2};
use rskan::spline;

#[test]
fn extend_grid_two_inputs_k2() {
    // I=2, G=4 → input is [2, 5]; k=2 → output is [2, 9].
    // For each input row, knots at [-1, -0.5, 0, 0.5, 1], h = 0.5.
    // After extension: [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2].
    let g: Array2<f32> = array![
        [-1.0, -0.5, 0.0, 0.5, 1.0],
        [-1.0, -0.5, 0.0, 0.5, 1.0],
    ];

    let extended = spline::extend_grid(g.view(), 2);

    assert_eq!(extended.shape(), &[2, 9]);
    let expected: Array2<f32> = array![
        [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
    ];
    for ((i, j), &v) in extended.indexed_iter() {
        assert_abs_diff_eq!(v, expected[[i, j]], epsilon = 1e-7);
    }
}

#[test]
fn extend_grid_k_zero_is_identity() {
    let g: Array2<f32> = array![[0.0, 1.0, 2.0]];
    let out = spline::extend_grid(g.view(), 0);
    assert_eq!(out.shape(), &[1, 3]);
    for (i, &v) in out.iter().enumerate() {
        assert_abs_diff_eq!(v, g[[0, i]], epsilon = 1e-7);
    }
}

#[test]
fn extend_grid_differing_h_per_row() {
    // Two inputs with different ranges → different h per row.
    let g: Array2<f32> = array![
        [0.0, 1.0, 2.0],      // h = 1.0
        [0.0, 0.5, 1.0],      // h = 0.5
    ];
    let out = spline::extend_grid(g.view(), 1);
    assert_eq!(out.shape(), &[2, 5]);
    assert_abs_diff_eq!(out[[0, 0]], -1.0, epsilon = 1e-7);
    assert_abs_diff_eq!(out[[0, 4]],  3.0, epsilon = 1e-7);
    assert_abs_diff_eq!(out[[1, 0]], -0.5, epsilon = 1e-7);
    assert_abs_diff_eq!(out[[1, 4]],  1.5, epsilon = 1e-7);
}
