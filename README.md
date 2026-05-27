# rskan

Burn-based Kolmogorov-Arnold Network (KAN) layers in Rust, with PyO3 bindings.

Drop-in KAN head for [ddrs](https://github.com/taddyb/ddrs) — forward + backward
numerical parity vs [pykan](https://github.com/KindXiaoming/pykan) 0.2.x.

## Status

**v1 — done.** All Rust + Python parity tests green; pykan-exported fixtures
cover bare-layer + multi-layer cases at DDR scale. v1.1 (CubeCL custom backward)
and `torch.autograd.Function` integration are separate specs.

## Layout

- `rskan/` — Rust library (`KanLayer`, `Kan`, B-spline math, init).
- `rskan-py/` — Python bindings (`forward`, `forward_with_grad`).
- `fixtures/` — pykan-exported `.npy` ground truth (committed).
- `scripts/` — fixture export script (run under DDR's uv venv).
- `docs/superpowers/specs/` — design spec.
- `docs/superpowers/plans/` — implementation plan.
- `docs/REGRESSION.md` — pinned "must never regress" test list.

## Quick start

### Rust

```rust
use burn::backend::{Autodiff, NdArray};
use rskan::{KanLayerConfig};

type B = Autodiff<NdArray<f32>>;
let device = Default::default();
let layer = KanLayerConfig::new(/*in_dim=*/ 21, /*out_dim=*/ 21, /*seed=*/ 1)
    .with_num(5).with_k(3)
    .init::<B>(&device);
let y = layer.forward(x);   // x: Tensor<B, 2>
```

### Python

```python
import numpy as np
import rskan

layer = rskan.KanLayer(in_dim=21, out_dim=21, num=5, k=3, seed=1, device="cpu")
x = np.random.uniform(-1, 1, (256, 21)).astype(np.float32)
y, grads = layer.forward_with_grad(x)
# grads = {"x": ..., "coef": ..., "scale_base": ..., "scale_sp": ...}
```

## Build & test

```bash
# Rust
cargo test  --release -p rskan                   # full Rust parity sweep
cargo test  --release -p rskan --features cuda   # adds cross-backend NdArray↔Cuda
cargo bench --bench kanlayer_forward             # Criterion

# Python (requires DDR's uv venv with pykan)
cd rskan-py && maturin develop --release
cd ~/projects/ddr && uv run pytest ~/projects/rskan/rskan-py/tests/
```

## Regenerating fixtures

```bash
cd ~/projects/ddr && uv run python ~/projects/rskan/scripts/export_pykan_fixtures.py
```

See `docs/REGRESSION.md` for the must-never-regress test list.
