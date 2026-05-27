//! Multi-layer `Kan` parity, including per-layer trajectory checks.
//!
//! Catches any hidden pykan `MultKAN.forward` divergence under
//! affine_trainable=False (e.g. subnode transforms that aren't truly identity).

mod common;

use burn::backend::NdArray;
use common::assert_close::{assert_close_flat, tensor_to_vec};
use common::fixture::{load_kan_fixture, load_manifest, KanFixture};
use common::tolerances::{FORWARD_ATOL, FORWARD_RTOL};
use rskan::{Kan, KanLayerConfig};

type B = NdArray<f32>;

fn run_case(case: &str) {
    let device = Default::default();
    let f: KanFixture<B> = load_kan_fixture(case, &device);

    // Build each KanLayer via init_from_parts; collect into a Kan manually.
    let mut layers = Vec::with_capacity(f.layers.len());
    for (l, slice) in f.layers.iter().enumerate() {
        let in_dim = f.params.widths[l];
        let out_dim = f.params.widths[l + 1];

        let cfg = KanLayerConfig::new(in_dim, out_dim, f.params.weight_seed)
            .with_num(f.params.grid)
            .with_k(f.params.k)
            .with_noise_scale(f.params.noise_scale)
            .with_scale_base_mu(f.params.scale_base_mu)
            .with_scale_base_sigma(f.params.scale_base_sigma)
            .with_scale_sp(f.params.scale_sp)
            .with_grid_range(f.params.grid_range);

        layers.push(cfg.init_from_parts::<B>(
            &device,
            slice.grid.clone(),
            slice.coef.clone(),
            slice.scale_base.clone(),
            slice.scale_sp.clone(),
            slice.mask.clone(),
        ));
    }
    let kan: Kan<B> = Kan { layers };

    // Verify each layer's output matches the hooked trajectory.
    let mut x = f.x;
    for (l, layer) in kan.layers.iter().enumerate() {
        x = layer.forward(x);
        let actual = tensor_to_vec(x.clone());
        let expected = tensor_to_vec(f.trajectory[l].clone());
        assert_close_flat(
            &actual, &expected, FORWARD_ATOL, FORWARD_RTOL,
            &format!("{case}: layer {l}"),
        );
    }

    // Final output match.
    let actual = tensor_to_vec(x);
    let expected = tensor_to_vec(f.y);
    assert_close_flat(
        &actual, &expected, FORWARD_ATOL, FORWARD_RTOL,
        &format!("{case}: final y"),
    );
}

#[test]
fn fixture_sweep_kan_stack() {
    for entry in load_manifest().cases {
        if entry.kind == "kan" {
            run_case(&entry.name);
        }
    }
}
