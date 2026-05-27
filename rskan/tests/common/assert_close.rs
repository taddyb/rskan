//! Tensor / ndarray closeness assertions with descriptive failure messages.

use burn::tensor::{backend::Backend, Tensor};
use ndarray::ArrayD;

/// Maximum absolute and relative differences between two flat slices of f32.
/// Returns `(max_abs, max_rel, worst_idx)`.
pub fn worst_diff(actual: &[f32], expected: &[f32]) -> (f32, f32, usize) {
    assert_eq!(actual.len(), expected.len(), "len mismatch");
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut worst = 0usize;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let abs = (a - e).abs();
        let rel = abs / (e.abs().max(1e-12));
        if abs > max_abs {
            max_abs = abs;
            worst = i;
        }
        if rel > max_rel {
            max_rel = rel;
        }
    }
    (max_abs, max_rel, worst)
}

/// Assert that two flat f32 slices agree within `atol` *or* `rtol`.
///
/// Panics with a descriptive message including the worst element's index,
/// actual, expected, and both abs/rel diffs.
pub fn assert_close_flat(
    actual: &[f32],
    expected: &[f32],
    atol: f32,
    rtol: f32,
    name: &str,
) {
    let (max_abs, max_rel, worst) = worst_diff(actual, expected);
    if max_abs > atol && max_rel > rtol {
        panic!(
            "{name}: parity violation\n  worst idx={worst}\n  actual={}, expected={}\n  max_abs={max_abs:.3e} (atol={atol:.0e})\n  max_rel={max_rel:.3e} (rtol={rtol:.0e})",
            actual[worst], expected[worst]
        );
    }
}

/// Convenience: tensor → flat Vec<f32>.
pub fn tensor_to_vec<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    let data = t.into_data().convert::<f32>();
    data.as_slice::<f32>().unwrap().to_vec()
}

/// ndarray (dynamic-rank) → flat Vec<f32> (row-major).
pub fn nd_to_vec(arr: &ArrayD<f32>) -> Vec<f32> {
    arr.iter().copied().collect()
}
