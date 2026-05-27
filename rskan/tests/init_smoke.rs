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
