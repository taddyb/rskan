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

#[test]
fn init_same_seed_produces_identical_weights() {
    let device = Default::default();
    let cfg = KanLayerConfig::new(4, 3, SEED).with_num(5).with_k(3);

    let a: KanLayer<B> = cfg.init(&device);
    let b: KanLayer<B> = cfg.init(&device);

    let max_diff = |t1: Tensor<B, 2>, t2: Tensor<B, 2>| -> f32 {
        (t1 - t2).abs().max().into_scalar()
    };
    let max_diff3 = |t1: Tensor<B, 3>, t2: Tensor<B, 3>| -> f32 {
        (t1 - t2).abs().max().into_scalar()
    };

    assert_eq!(max_diff(a.grid.val(), b.grid.val()),             0.0);
    assert_eq!(max_diff3(a.coef.val(), b.coef.val()),            0.0);
    assert_eq!(max_diff(a.scale_base.val(), b.scale_base.val()), 0.0);
    assert_eq!(max_diff(a.scale_sp.val(), b.scale_sp.val()),     0.0);
    assert_eq!(max_diff(a.mask.val(), b.mask.val()),             0.0);
}

#[test]
fn init_different_seeds_produce_different_coefs() {
    let device = Default::default();
    let a = KanLayerConfig::new(4, 3, 1).with_num(5).with_k(3).init::<B>(&device);
    let b = KanLayerConfig::new(4, 3, 2).with_num(5).with_k(3).init::<B>(&device);

    let diff = (a.coef.val() - b.coef.val()).abs().max().into_scalar();
    assert!(diff > 1e-4, "different seeds should produce different coef");
}

#[test]
fn init_shapes_are_correct() {
    let device = Default::default();
    let (i, o, num, k) = (7usize, 5usize, 6usize, 3usize);
    let layer = KanLayerConfig::new(i, o, SEED).with_num(num).with_k(k).init::<B>(&device);

    assert_eq!(layer.grid.val().dims(),       [i, num + 1 + 2 * k]);
    assert_eq!(layer.coef.val().dims(),       [i, o, num + k]);
    assert_eq!(layer.scale_base.val().dims(), [i, o]);
    assert_eq!(layer.scale_sp.val().dims(),   [i, o]);
    assert_eq!(layer.mask.val().dims(),       [i, o]);

    // Mask is all ones.
    let mask_max = layer.mask.val().clone().max().into_scalar();
    let mask_min = layer.mask.val().min().into_scalar();
    assert_eq!(mask_max, 1.0);
    assert_eq!(mask_min, 1.0);
}

#[test]
fn init_respects_trainability_flags() {
    // `is_require_grad` is a no-op on non-autodiff backends (always returns
    // false), so the trainability flags can only be observed under Autodiff.
    use burn::backend::Autodiff;
    type AD = Autodiff<NdArray<f32>>;

    let device = Default::default();
    let frozen_sb = KanLayerConfig::new(3, 3, SEED)
        .with_sb_trainable(false)
        .init::<AD>(&device);
    assert!(!frozen_sb.scale_base.is_require_grad());
    assert!( frozen_sb.coef.is_require_grad());
    assert!( frozen_sb.scale_sp.is_require_grad());
    assert!(!frozen_sb.grid.is_require_grad());
    assert!(!frozen_sb.mask.is_require_grad());

    let frozen_sp = KanLayerConfig::new(3, 3, SEED)
        .with_sp_trainable(false)
        .init::<AD>(&device);
    assert!(!frozen_sp.scale_sp.is_require_grad());
}

#[test]
fn init_from_parts_respects_trainability_flags() {
    // Same coverage as init_respects_trainability_flags but for the fixture-load path.
    use burn::backend::{Autodiff, NdArray};
    type AD = Autodiff<NdArray<f32>>;

    let device = Default::default();
    let (i, o, num, k) = (3usize, 3usize, 5usize, 3usize);
    let knots = num + 1 + 2 * k;
    let n_basis = num + k;

    let ones_2d_ad = |r: usize, c: usize| -> Tensor<AD, 2> {
        Tensor::from_data(
            TensorData::new(vec![1.0_f32; r * c], [r, c]),
            &device,
        )
    };
    let zeros_3d_ad = |d0: usize, d1: usize, d2: usize| -> Tensor<AD, 3> {
        Tensor::from_data(
            TensorData::new(vec![0.0_f32; d0 * d1 * d2], [d0, d1, d2]),
            &device,
        )
    };

    let cfg = KanLayerConfig::new(i, o, SEED)
        .with_num(num).with_k(k)
        .with_sb_trainable(false);

    let layer: KanLayer<AD> = cfg.init_from_parts(
        &device,
        ones_2d_ad(i, knots),
        zeros_3d_ad(i, o, n_basis),
        ones_2d_ad(i, o),
        ones_2d_ad(i, o),
        ones_2d_ad(i, o),
    );

    assert!( layer.coef.is_require_grad(),       "coef should be trainable");
    assert!(!layer.scale_base.is_require_grad(), "scale_base should be frozen (sb_trainable=false)");
    assert!( layer.scale_sp.is_require_grad(),   "scale_sp should be trainable");
    assert!(!layer.grid.is_require_grad(),       "grid should be frozen");
    assert!(!layer.mask.is_require_grad(),       "mask should be frozen");
}

#[test]
fn kan_forward_equals_manual_layer_composition() {
    use rskan::{Kan, KanConfig, KanLayer, KanLayerConfig};

    let device = Default::default();
    let cfg = KanConfig::new(vec![3, 4, 2], SEED)
        .with_grid(5).with_k(3).with_noise_scale(0.3);

    let model: Kan<B> = cfg.init(&device);
    assert_eq!(model.layers.len(), 2);

    // Build the same two layers manually with sub-seeds = SEED + l.
    let layer0: KanLayer<B> = KanLayerConfig::new(3, 4, SEED.wrapping_add(0))
        .with_num(5).with_k(3).with_noise_scale(0.3).init(&device);
    let layer1: KanLayer<B> = KanLayerConfig::new(4, 2, SEED.wrapping_add(1))
        .with_num(5).with_k(3).with_noise_scale(0.3).init(&device);

    // Inputs.
    let batch = 6usize;
    let x_data: Vec<f32> = (0..batch * 3).map(|n| -0.4 + (n as f32) * 0.03).collect();
    let x_t: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(x_data, [batch, 3]),
        &device,
    );

    let y_model = model.forward(x_t.clone());
    let y_manual = layer1.forward(layer0.forward(x_t));

    let diff = (y_model - y_manual).abs().max().into_scalar();
    assert!(diff < 1e-6, "Kan forward should equal manual composition; max diff = {diff}");
}

#[test]
fn kan_requires_at_least_two_widths() {
    let device = Default::default();
    let result = std::panic::catch_unwind(|| {
        rskan::KanConfig::new(vec![5], SEED).init::<B>(&device)
    });
    assert!(result.is_err());
}

/// Confirm KanLayer can initialise across the full grid/k matrix DDR uses
/// in production. Was added after DDR's grid=50/k=2 config broke the
/// original cholesky_solve at the 1e-8 ridge.
#[test]
fn kan_layer_inits_at_ddr_production_grid_50_k_2() {
    let device = Default::default();
    let layer = KanLayerConfig::new(21, 21, 42)
        .with_num(50)
        .with_k(2)
        .with_noise_scale(0.3)
        .init::<B>(&device);

    let coef: Vec<f32> = layer.coef.val().into_data().to_vec().unwrap();
    assert!(
        coef.iter().all(|v| v.is_finite()),
        "grid=50 k=2: coef contains non-finite value"
    );
}

#[test]
fn kan_layer_inits_across_grid_k_matrix() {
    let device = Default::default();
    for (grid, k) in [(5, 3), (10, 2), (10, 3), (20, 2), (30, 2), (50, 2), (50, 3), (100, 2)] {
        let layer = KanLayerConfig::new(8, 8, 0)
            .with_num(grid)
            .with_k(k)
            .with_noise_scale(0.3)
            .init::<B>(&device);

        let coef: Vec<f32> = layer.coef.val().into_data().to_vec().unwrap();
        assert!(
            coef.iter().all(|v| v.is_finite()),
            "grid={grid} k={k}: coef contains non-finite value"
        );
    }
}
