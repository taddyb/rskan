//! Backward fixture sweep against pykan exports.

mod common;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use common::assert_close::{assert_close_flat, tensor_to_vec};
use common::fixture::{load_layer_fixture, load_manifest, LayerFixture};
use common::tolerances::{BACKWARD_ATOL, BACKWARD_RTOL};
use rskan::KanLayerConfig;

type B = Autodiff<NdArray<f32>>;

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
        &device,
        f.grid.clone(),
        f.coef.clone(),
        f.scale_base.clone(),
        f.scale_sp.clone(),
        f.mask.clone(),
    );

    // Mark x as a leaf with grad.
    let x: Tensor<B, 2> = f.x.clone().require_grad();
    let y = layer.forward(x.clone());
    let loss = y.sum();
    let grads = loss.backward();

    // Extract gradients. `Tensor::grad` already materializes Tensor<InnerBackend, D>.
    let gx = x.grad(&grads).expect("x.grad missing");
    let gcoef = layer.coef.val().grad(&grads).expect("coef.grad missing");
    let gsb = layer
        .scale_base
        .val()
        .grad(&grads)
        .expect("scale_base.grad missing");
    let gsp = layer
        .scale_sp
        .val()
        .grad(&grads)
        .expect("scale_sp.grad missing");

    assert_close_flat(
        &tensor_to_vec(gx),
        &tensor_to_vec(f.grad_x),
        BACKWARD_ATOL,
        BACKWARD_RTOL,
        &format!("{case}: grad_x"),
    );
    assert_close_flat(
        &tensor_to_vec(gcoef),
        &tensor_to_vec(f.grad_coef),
        BACKWARD_ATOL,
        BACKWARD_RTOL,
        &format!("{case}: grad_coef"),
    );
    assert_close_flat(
        &tensor_to_vec(gsb),
        &tensor_to_vec(f.grad_scale_base),
        BACKWARD_ATOL,
        BACKWARD_RTOL,
        &format!("{case}: grad_scale_base"),
    );
    assert_close_flat(
        &tensor_to_vec(gsp),
        &tensor_to_vec(f.grad_scale_sp),
        BACKWARD_ATOL,
        BACKWARD_RTOL,
        &format!("{case}: grad_scale_sp"),
    );
}

#[test]
fn fixture_sweep_backward() {
    for entry in load_manifest().cases {
        if entry.kind == "layer" {
            run_case(&entry.name);
        }
    }
}
