//! Seeded init helpers for `KanLayer` (pykan-recipe).
//!
//! All RNG sampling happens on CPU via `rand::StdRng`; tensors are then
//! materialized on the target device via `Tensor::from_data`. This sidesteps
//! any Burn-internal RNG state as a source of non-determinism.

use burn::module::Param;
use burn::tensor::{backend::Backend, Tensor, TensorData};
use ndarray::{Array1, Array2, Array3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::spline::extend_grid;

/// Build a uniform linspace grid `[in_dim, num + 1]` and extend it by `k` ghost
/// knots on each side, producing the frozen `grid` buffer of shape
/// `[in_dim, num + 1 + 2k]`.
pub(crate) fn build_grid(
    in_dim: usize,
    num: usize,
    k: usize,
    grid_range: [f64; 2],
) -> Array2<f32> {
    let lo = grid_range[0] as f32;
    let hi = grid_range[1] as f32;
    let row = Array1::linspace(lo, hi, num + 1);
    let g_pre: Array2<f32> =
        Array2::from_shape_fn((in_dim, num + 1), |(_, j)| row[j]);
    extend_grid(g_pre.view(), k)
}

/// Sample the noise targets used to fit the initial spline coefficients.
/// Distribution: `U(-1/2, 1/2) * (noise_scale / num)`. Shape `[num+1, in_dim, out_dim]`.
pub(crate) fn sample_noises(
    rng: &mut StdRng,
    in_dim: usize,
    out_dim: usize,
    num: usize,
    noise_scale: f64,
) -> Array3<f32> {
    let amp = (noise_scale / num as f64) as f32;
    Array3::from_shape_fn((num + 1, in_dim, out_dim), |_| {
        (rng.gen::<f32>() - 0.5) * amp
    })
}

/// Sample `scale_base` per pykan KANLayer.py:110 — `Uniform[-1, 1]` despite
/// the docstring claiming Normal. Code wins.
pub(crate) fn sample_scale_base(
    rng: &mut StdRng,
    in_dim: usize,
    out_dim: usize,
    mu: f64,
    sigma: f64,
) -> Array2<f32> {
    let inv_sqrt = 1.0 / (in_dim as f32).sqrt();
    let mu = mu as f32;
    let sigma = sigma as f32;
    Array2::from_shape_fn((in_dim, out_dim), |_| {
        (mu + sigma * (rng.gen::<f32>() * 2.0 - 1.0)) * inv_sqrt
    })
}

/// Build the `scale_sp` initial value: constant `scale_sp / sqrt(in_dim)`
/// (mask is ones in v1, so it's omitted from the product).
pub(crate) fn build_scale_sp(in_dim: usize, out_dim: usize, scale_sp: f64) -> Array2<f32> {
    let inv_sqrt = 1.0 / (in_dim as f32).sqrt();
    Array2::from_elem((in_dim, out_dim), (scale_sp as f32) * inv_sqrt)
}

/// Materialize an `ndarray::Array2<f32>` into a Burn `Param<Tensor<B, 2>>`.
pub(crate) fn to_param_2<B: Backend>(arr: Array2<f32>, device: &B::Device) -> Param<Tensor<B, 2>> {
    let (r, c) = (arr.shape()[0], arr.shape()[1]);
    let data = TensorData::new(arr.as_slice().unwrap().to_vec(), [r, c]);
    Param::from_tensor(Tensor::from_data(data, device))
}

/// Materialize an `ndarray::Array3<f32>` into a Burn `Param<Tensor<B, 3>>`.
pub(crate) fn to_param_3<B: Backend>(arr: Array3<f32>, device: &B::Device) -> Param<Tensor<B, 3>> {
    let (d0, d1, d2) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let data = TensorData::new(arr.as_slice().unwrap().to_vec(), [d0, d1, d2]);
    Param::from_tensor(Tensor::from_data(data, device))
}

/// Build the initial RNG from a u64 seed.
pub(crate) fn rng_from_seed(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}
