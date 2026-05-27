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

#[test]
fn curve2coef_roundtrip_recovers_targets() {
    // Build a grid, sample target values on its inner knots, fit coefficients,
    // and confirm coef2curve at those same knots recovers the targets to ~1e-5.
    use ndarray::Array3;
    let g_in: Array2<f32> = array![[-1.0, -0.5, 0.0, 0.5, 1.0]];      // I=1, G+1=5
    let k: usize = 3;
    let grid_full = spline::extend_grid(g_in.view(), k);              // [1, 11]

    let inner_t = g_in.t().to_owned();                                // [5, 1]
    // targets shape [batch=5, I=1, O=2]: smooth-ish y(x) = [x, x^2]
    let mut targets = Array3::<f32>::zeros((5, 1, 2));
    for (j, &x) in g_in.row(0).iter().enumerate() {
        targets[[j, 0, 0]] = x;
        targets[[j, 0, 1]] = x * x;
    }

    let coef = spline::curve2coef(inner_t.view(), targets.view(), grid_full.view(), k);
    assert_eq!(coef.shape(), &[1, 2, g_in.shape()[1] + k - 1]);       // [I, O, G+k]

    // Reconstruct y at the same inner positions via the CPU b_batch helper.
    let basis = spline::b_batch_nd(inner_t.view(), grid_full.view(), k); // [5, 1, G+k]
    for j in 0..5 {
        let mut y0 = 0.0_f32;
        let mut y1 = 0.0_f32;
        for c in 0..basis.shape()[2] {
            y0 += basis[[j, 0, c]] * coef[[0, 0, c]];
            y1 += basis[[j, 0, c]] * coef[[0, 1, c]];
        }
        assert_abs_diff_eq!(y0, targets[[j, 0, 0]], epsilon = 1e-4);
        assert_abs_diff_eq!(y1, targets[[j, 0, 1]], epsilon = 1e-4);
    }
}

#[test]
fn b_batch_burn_matches_cpu_for_uniform_grid() {
    use burn::backend::NdArray;
    use burn::tensor::{Tensor, TensorData};
    use ndarray::{Array1, Array2 as NdArray2};

    type B = NdArray<f32>;
    let device = Default::default();
    let (batch, in_dim, k) = (8usize, 2usize, 3usize);
    let g_intervals = 4usize;
    let knots = g_intervals + 1 + 2 * k;                          // = 11

    // Build a uniform extended grid on [-1, 1].
    let row = Array1::linspace(-1.0_f32, 1.0_f32, g_intervals + 1);
    let g_pre: NdArray2<f32> =
        NdArray2::from_shape_fn((in_dim, g_intervals + 1), |(_, j)| row[j]);
    let grid_full = spline::extend_grid(g_pre.view(), k);          // [I, knots]

    // x sampled in [-0.8, 0.8] so all values are in-domain.
    let mut x_arr = NdArray2::<f32>::zeros((batch, in_dim));
    for b in 0..batch {
        for i in 0..in_dim {
            x_arr[[b, i]] = -0.8 + 1.6 * (b as f32 + i as f32 * 0.13) / batch as f32;
        }
    }

    // CPU reference.
    let expected = spline::b_batch_nd(x_arr.view(), grid_full.view(), k);
    assert_eq!(expected.shape(), &[batch, in_dim, knots - k - 1]);

    // Burn computation. `TensorData::new(vec, shape)` is the canonical
    // constructor in Burn 0.21; the plan's chained `.convert().reshape()` form
    // does not exist on `TensorData` (only on `Tensor`).
    let x_t: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(x_arr.as_slice().unwrap().to_vec(), [batch, in_dim]),
        &device,
    );
    let grid_t: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(grid_full.as_slice().unwrap().to_vec(), [in_dim, knots]),
        &device,
    );

    let actual_t: Tensor<B, 3> = spline::b_batch(x_t, grid_t, k);
    let actual_data = actual_t.into_data().convert::<f32>();
    let actual = actual_data.as_slice::<f32>().unwrap();

    for b in 0..batch {
        for i in 0..in_dim {
            for c in 0..(knots - k - 1) {
                let flat = (b * in_dim + i) * (knots - k - 1) + c;
                assert_abs_diff_eq!(
                    actual[flat],
                    expected[[b, i, c]],
                    epsilon = 1e-6
                );
            }
        }
    }
}
