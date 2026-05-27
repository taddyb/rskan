//! Criterion bench: KanLayer::forward across batch sizes (NdArray + Cuda).

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rskan::{KanLayer, KanLayerConfig};

const SEED: u64 = 0xC0FFEE;

fn build_layer<B: burn::tensor::backend::Backend>(device: &B::Device) -> KanLayer<B> {
    KanLayerConfig::new(21, 21, SEED).with_num(5).with_k(3).init(device)
}

fn build_input<B: burn::tensor::backend::Backend>(batch: usize, device: &B::Device) -> Tensor<B, 2> {
    let data: Vec<f32> = (0..(batch * 21))
        .map(|n| -0.8 + 1.6 * (n as f32 / (batch * 21) as f32))
        .collect();
    Tensor::from_data(TensorData::new(data, [batch, 21]), device)
}

fn bench_ndarray(c: &mut Criterion) {
    type B = NdArray<f32>;
    let device = Default::default();
    let layer = build_layer::<B>(&device);

    let mut group = c.benchmark_group("kanlayer_forward/ndarray");
    for &batch in &[16usize, 128, 1024, 5000] {
        let x = build_input::<B>(batch, &device);
        group.bench_with_input(BenchmarkId::from_parameter(batch), &x, |b, x| {
            b.iter(|| black_box(layer.forward(black_box(x.clone()))));
        });
    }
    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_cuda(c: &mut Criterion) {
    type B = burn_cuda::Cuda<f32>;
    let device = burn_cuda::CudaDevice::default();
    let layer = build_layer::<B>(&device);

    let mut group = c.benchmark_group("kanlayer_forward/cuda");
    for &batch in &[16usize, 128, 1024, 5000] {
        let x = build_input::<B>(batch, &device);
        group.bench_with_input(BenchmarkId::from_parameter(batch), &x, |b, x| {
            b.iter(|| black_box(layer.forward(black_box(x.clone()))));
        });
    }
    group.finish();
}

#[cfg(feature = "cuda")]
criterion_group!(benches, bench_ndarray, bench_cuda);
#[cfg(not(feature = "cuda"))]
criterion_group!(benches, bench_ndarray);
criterion_main!(benches);
