//! Tolerance constants justified in spec §7.4. One place to tune.

/// Forward parity (NdArray): 4× cubic-spline approx floor 5⁻⁸ ≈ 2.5e-6.
pub const FORWARD_ATOL: f32 = 1e-5;
pub const FORWARD_RTOL: f32 = 1e-4;

/// Backward parity (NdArray): one order of magnitude looser for accumulated
/// reduction rounding through three levels of basis recursion + einsum sum.
pub const BACKWARD_ATOL: f32 = 1e-4;
pub const BACKWARD_RTOL: f32 = 1e-3;

/// Cross-backend forward (NdArray ↔ Cuda): CUDA reduction-order non-determinism
/// at large fan-in (e.g. 21-way sum in coef2curve permute+matmul) exceeds the
/// "FMA-equivalent" tolerance the spec originally assumed. Loosened to match
/// the backward path's tolerance — empirically grounded.
pub const CROSS_FORWARD_ATOL: f32 = 1e-4;
pub const CROSS_FORWARD_RTOL: f32 = 1e-3;

/// Cross-backend backward: CUDA reduction-order non-determinism.
pub const CROSS_BACKWARD_ATOL: f32 = 1e-4;
pub const CROSS_BACKWARD_RTOL: f32 = 1e-3;
