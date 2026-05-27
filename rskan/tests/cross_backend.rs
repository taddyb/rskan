//! Cross-backend agreement: NdArray ↔ Cuda.
//! Only compiled with `--features cuda`. Skipped if a CUDA driver is absent.

#![cfg(feature = "cuda")]

mod common;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::{backend::Backend, Tensor, TensorData};
use burn_cuda::Cuda;
use common::assert_close::{assert_close_flat, tensor_to_vec};
use common::fixture::{load_layer_fixture, LayerFixture, LayerParams};
use common::tolerances::{
    CROSS_BACKWARD_ATOL, CROSS_BACKWARD_RTOL, CROSS_FORWARD_ATOL, CROSS_FORWARD_RTOL,
};
use rskan::KanLayerConfig;

fn move_layer_to<BSrc: Backend, BDst: Backend>(
    src: &LayerFixture<BSrc>,
    dst_dev: &BDst::Device,
) -> LayerFixture<BDst> {
    let move_2 = |t: Tensor<BSrc, 2>| -> Tensor<BDst, 2> {
        let dims = t.dims();
        let data = t.into_data().convert::<f32>();
        Tensor::from_data(
            TensorData::new(data.as_slice::<f32>().unwrap().to_vec(), [dims[0], dims[1]]),
            dst_dev,
        )
    };
    let move_3 = |t: Tensor<BSrc, 3>| -> Tensor<BDst, 3> {
        let dims = t.dims();
        let data = t.into_data().convert::<f32>();
        Tensor::from_data(
            TensorData::new(
                data.as_slice::<f32>().unwrap().to_vec(),
                [dims[0], dims[1], dims[2]],
            ),
            dst_dev,
        )
    };
    LayerFixture {
        params: LayerParams {
            name: src.params.name.clone(),
            in_dim: src.params.in_dim,
            out_dim: src.params.out_dim,
            num: src.params.num,
            k: src.params.k,
            noise_scale: src.params.noise_scale,
            scale_base_mu: src.params.scale_base_mu,
            scale_base_sigma: src.params.scale_base_sigma,
            scale_sp: src.params.scale_sp,
            grid_range: src.params.grid_range,
            sp_trainable: src.params.sp_trainable,
            sb_trainable: src.params.sb_trainable,
            weight_seed: src.params.weight_seed,
            x_seed: src.params.x_seed,
            batch: src.params.batch,
        },
        grid: move_2(src.grid.clone()),
        coef: move_3(src.coef.clone()),
        scale_base: move_2(src.scale_base.clone()),
        scale_sp: move_2(src.scale_sp.clone()),
        mask: move_2(src.mask.clone()),
        x: move_2(src.x.clone()),
        y: move_2(src.y.clone()),
        grad_x: move_2(src.grad_x.clone()),
        grad_coef: move_3(src.grad_coef.clone()),
        grad_scale_base: move_2(src.grad_scale_base.clone()),
        grad_scale_sp: move_2(src.grad_scale_sp.clone()),
    }
}

#[test]
fn ndarray_vs_cuda_forward_matches() {
    type BCpu = NdArray<f32>;
    type BCuda = Cuda<f32>;

    let cpu_dev = Default::default();
    let cuda_dev = burn_cuda::CudaDevice::default();

    let case = "kanlayer_i21_o21_k3_g5_s1";
    let f_cpu: LayerFixture<BCpu> = load_layer_fixture(case, &cpu_dev);
    let f_cuda: LayerFixture<BCuda> = move_layer_to::<BCpu, BCuda>(&f_cpu, &cuda_dev);

    let cfg = KanLayerConfig::new(
        f_cpu.params.in_dim,
        f_cpu.params.out_dim,
        f_cpu.params.weight_seed,
    )
    .with_num(f_cpu.params.num)
    .with_k(f_cpu.params.k);

    let layer_cpu = cfg.init_from_parts::<BCpu>(
        &cpu_dev,
        f_cpu.grid.clone(),
        f_cpu.coef.clone(),
        f_cpu.scale_base.clone(),
        f_cpu.scale_sp.clone(),
        f_cpu.mask.clone(),
    );
    let layer_cuda = cfg.init_from_parts::<BCuda>(
        &cuda_dev,
        f_cuda.grid,
        f_cuda.coef,
        f_cuda.scale_base,
        f_cuda.scale_sp,
        f_cuda.mask,
    );

    let y_cpu = layer_cpu.forward(f_cpu.x);
    let y_cuda = layer_cuda.forward(f_cuda.x);

    assert_close_flat(
        &tensor_to_vec(y_cuda),
        &tensor_to_vec(y_cpu),
        CROSS_FORWARD_ATOL,
        CROSS_FORWARD_RTOL,
        "ndarray vs cuda forward",
    );
}

#[test]
fn ndarray_vs_cuda_backward_matches() {
    type BCpu = Autodiff<NdArray<f32>>;
    type BCuda = Autodiff<Cuda<f32>>;

    let cpu_dev = Default::default();
    let cuda_dev = burn_cuda::CudaDevice::default();

    let case = "kanlayer_i8_o8_k3_g5_s1";
    let f_cpu: LayerFixture<BCpu> = load_layer_fixture(case, &cpu_dev);
    let f_cuda: LayerFixture<BCuda> = move_layer_to::<BCpu, BCuda>(&f_cpu, &cuda_dev);

    let cfg = KanLayerConfig::new(
        f_cpu.params.in_dim,
        f_cpu.params.out_dim,
        f_cpu.params.weight_seed,
    )
    .with_num(f_cpu.params.num)
    .with_k(f_cpu.params.k);

    let layer_cpu = cfg.init_from_parts::<BCpu>(
        &cpu_dev,
        f_cpu.grid.clone(),
        f_cpu.coef.clone(),
        f_cpu.scale_base.clone(),
        f_cpu.scale_sp.clone(),
        f_cpu.mask.clone(),
    );
    let layer_cuda = cfg.init_from_parts::<BCuda>(
        &cuda_dev,
        f_cuda.grid,
        f_cuda.coef,
        f_cuda.scale_base,
        f_cuda.scale_sp,
        f_cuda.mask,
    );

    let x_cpu = f_cpu.x.clone().require_grad();
    let x_cuda = f_cuda.x.clone().require_grad();
    let y_cpu = layer_cpu.forward(x_cpu.clone());
    let y_cuda = layer_cuda.forward(x_cuda.clone());

    let g_cpu = y_cpu.sum().backward();
    let g_cuda = y_cuda.sum().backward();

    let gx_cpu = x_cpu.grad(&g_cpu).expect("x.grad cpu");
    let gx_cuda = x_cuda.grad(&g_cuda).expect("x.grad cuda");

    assert_close_flat(
        &tensor_to_vec(gx_cuda),
        &tensor_to_vec(gx_cpu),
        CROSS_BACKWARD_ATOL,
        CROSS_BACKWARD_RTOL,
        "ndarray vs cuda grad_x",
    );
}
