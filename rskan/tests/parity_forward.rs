//! Forward fixture sweep against pykan exports.

mod common;

use burn::backend::NdArray;
use common::assert_close::{assert_close_flat, tensor_to_vec};
use common::fixture::{load_layer_fixture, load_manifest, LayerFixture};
use common::tolerances::{FORWARD_ATOL, FORWARD_RTOL};
use rskan::KanLayerConfig;

type B = NdArray<f32>;

fn run_case(case: &str) {
    let device = Default::default();
    let f: LayerFixture<B> = load_layer_fixture(case, &device);

    let cfg = KanLayerConfig::new(f.params.in_dim, f.params.out_dim, f.params.weight_seed)
        .with_num(f.params.num)
        .with_k(f.params.k)
        .with_noise_scale(f.params.noise_scale)
        .with_scale_base_mu(f.params.scale_base_mu)
        .with_scale_base_sigma(f.params.scale_base_sigma)
        .with_scale_sp(f.params.scale_sp)
        .with_grid_range(f.params.grid_range)
        .with_sp_trainable(f.params.sp_trainable)
        .with_sb_trainable(f.params.sb_trainable);

    let layer = cfg.init_from_parts::<B>(
        &device, f.grid, f.coef, f.scale_base, f.scale_sp, f.mask,
    );

    let y_actual = layer.forward(f.x);
    let actual = tensor_to_vec(y_actual);
    let expected = tensor_to_vec(f.y);

    assert_close_flat(&actual, &expected, FORWARD_ATOL, FORWARD_RTOL, case);
}

#[test]
fn fixture_sweep_forward() {
    for entry in load_manifest().cases {
        if entry.kind == "layer" {
            run_case(&entry.name);
        }
    }
}

/// **Regression case** — pinned in CLAUDE.md as the rskan-side analogue of
/// ddrs's `compare_ddr_sandbox` ABSOLUTE MATCH gate. This test must never
/// regress without a deliberate spec change.
#[test]
fn ddr_scale_must_match_pykan() {
    run_case("kanlayer_i21_o21_k3_g5_s1");
}
