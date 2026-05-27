//! Init smoke tests — shapes, require_grad flags, reproducibility.
//! Reproducibility tests are added in Task 9; this file starts with
//! `init_from_parts` shape-check coverage only.

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use rskan::{KanLayer, KanLayerConfig};

type B = NdArray<f32>;

const SEED: u64 = 42;

fn ones_2d(rows: usize, cols: usize) -> Tensor<B, 2> {
    let device = Default::default();
    Tensor::from_data(
        TensorData::new(vec![1.0_f32; rows * cols], [rows, cols]),
        &device,
    )
}

fn zeros_3d(d0: usize, d1: usize, d2: usize) -> Tensor<B, 3> {
    let device = Default::default();
    Tensor::from_data(
        TensorData::new(vec![0.0_f32; d0 * d1 * d2], [d0, d1, d2]),
        &device,
    )
}

#[test]
fn init_from_parts_accepts_correct_shapes() {
    let device = Default::default();
    let cfg = KanLayerConfig::new(3, 5, SEED).with_num(4).with_k(3);

    let layer: KanLayer<B> = cfg.init_from_parts(
        &device,
        ones_2d(3, 4 + 1 + 2 * 3),                     // grid [I, K]
        zeros_3d(3, 5, 4 + 3),                          // coef [I, O, n_basis]
        ones_2d(3, 5),                                  // scale_base
        ones_2d(3, 5),                                  // scale_sp
        ones_2d(3, 5),                                  // mask
    );

    assert_eq!(layer.k, 3);
    assert_eq!(layer.grid.val().dims(),       [3, 11]);
    assert_eq!(layer.coef.val().dims(),       [3, 5, 7]);
    assert_eq!(layer.scale_base.val().dims(), [3, 5]);
    assert_eq!(layer.scale_sp.val().dims(),   [3, 5]);
    assert_eq!(layer.mask.val().dims(),       [3, 5]);
}

#[test]
#[should_panic(expected = "grid shape")]
fn init_from_parts_rejects_bad_grid_shape() {
    let device = Default::default();
    let cfg = KanLayerConfig::new(3, 5, SEED).with_num(4).with_k(3);
    let _ = cfg.init_from_parts::<B>(
        &device,
        ones_2d(3, 10),                                 // wrong: should be [3, 11]
        zeros_3d(3, 5, 7),
        ones_2d(3, 5),
        ones_2d(3, 5),
        ones_2d(3, 5),
    );
}

#[test]
fn forward_zero_coef_equals_silu_base_branch() {
    use burn::tensor::activation::silu;

    let device = Default::default();
    let (i, o, num, k) = (4usize, 3usize, 5usize, 3usize);
    let knots = num + 1 + 2 * k;
    let n_basis = num + k;
    let cfg = KanLayerConfig::new(i, o, SEED).with_num(num).with_k(k);

    // Build a generic extended grid on [-1, 1].
    use ndarray::{Array1, Array2 as NdArray2};
    let row = Array1::linspace(-1.0_f32, 1.0_f32, num + 1);
    let g_pre: NdArray2<f32> =
        NdArray2::from_shape_fn((i, num + 1), |(_, j)| row[j]);
    let grid_full = rskan::spline::extend_grid(g_pre.view(), k);
    let grid_t: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(grid_full.as_slice().unwrap().to_vec(), [i, knots]),
        &device,
    );

    let layer = cfg.init_from_parts::<B>(
        &device,
        grid_t,
        zeros_3d(i, o, n_basis),                        // coef = 0 → spline branch = 0
        ones_2d(i, o),                                  // scale_base = 1
        ones_2d(i, o),                                  // scale_sp = 1 (irrelevant)
        ones_2d(i, o),                                  // mask = 1
    );

    // Random-ish x inside the grid.
    let batch = 8usize;
    let x_data: Vec<f32> = (0..batch * i)
        .map(|n| -0.5 + (n as f32) * 0.05)
        .collect();
    let x_t: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(x_data, [batch, i]),
        &device,
    );

    let y = layer.forward(x_t.clone());
    assert_eq!(y.dims(), [batch, o]);

    // Expected: y[b, o] = Σ_i SiLU(x[b, i]). (scale_base=1, mask=1, spline=0.)
    let silu_x = silu(x_t);                                   // [B, I]
    let expected_per_batch = silu_x.sum_dim(1);               // [B, 1]
    // Broadcast across the O outputs:
    let expected_broadcast = expected_per_batch.expand([batch, o]);
    // Equal within 1e-5.
    let diff = (y - expected_broadcast).abs().max();
    let max = diff.into_scalar();
    assert!(max < 1e-5, "max diff = {max}");
}

#[test]
fn forward_zero_scale_base_keeps_only_spline_branch() {
    let device = Default::default();
    let (i, o, num, k) = (2usize, 2usize, 4usize, 3usize);
    let knots = num + 1 + 2 * k;
    let n_basis = num + k;
    let cfg = KanLayerConfig::new(i, o, SEED).with_num(num).with_k(k);

    use ndarray::{Array1, Array2 as NdArray2, Array3};
    let row = Array1::linspace(-1.0_f32, 1.0_f32, num + 1);
    let g_pre: NdArray2<f32> =
        NdArray2::from_shape_fn((i, num + 1), |(_, j)| row[j]);
    let grid_full = rskan::spline::extend_grid(g_pre.view(), k);
    let grid_t: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(grid_full.as_slice().unwrap().to_vec(), [i, knots]),
        &device,
    );

    // coef: arbitrary small values.
    let coef_arr: Array3<f32> = Array3::from_shape_fn((i, o, n_basis), |(a, b, c)| {
        0.1 * (a as f32 + b as f32 * 0.5 + c as f32 * 0.25)
    });
    let coef_t: Tensor<B, 3> = Tensor::from_data(
        TensorData::new(coef_arr.as_slice().unwrap().to_vec(), [i, o, n_basis]),
        &device,
    );

    // scale_base = 0, so the SiLU branch contributes nothing.
    let zeros_2d_fn = |r: usize, c: usize| -> Tensor<B, 2> {
        Tensor::from_data(
            TensorData::new(vec![0.0_f32; r * c], [r, c]),
            &device,
        )
    };

    let layer = cfg.init_from_parts::<B>(
        &device,
        grid_t.clone(),
        coef_t.clone(),
        zeros_2d_fn(i, o),                            // scale_base = 0
        ones_2d(i, o),                                // scale_sp = 1
        ones_2d(i, o),                                // mask = 1
    );

    let batch = 5usize;
    let x_data: Vec<f32> = (0..batch * i).map(|n| -0.3 + (n as f32) * 0.07).collect();
    let x_t: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(x_data, [batch, i]),
        &device,
    );
    let y = layer.forward(x_t.clone());

    // Expected: y[b, o] = Σ_i spline_{i, o}(x[b, i])
    let spline = rskan::spline::coef2curve(x_t, grid_t, coef_t, k); // [B, I, O]
    let expected_per_batch = spline.sum_dim(1).squeeze_dim::<2>(1); // [B, O]
    let diff = (y - expected_per_batch).abs().max();
    let max = diff.into_scalar();
    assert!(max < 1e-5, "max diff = {max}");
}
