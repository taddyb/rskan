# rskan

Burn-based Kolmogorov-Arnold Network (KAN) layers in Rust, with PyO3 bindings.

Built to be the GPU-capable KAN head for [ddrs](https://github.com/taddyb/ddrs).
Achieves forward+backward numerical parity against [pykan](https://github.com/KindXiaoming/pykan) 0.2.x.

## Status

v1 in development. See `docs/superpowers/plans/2026-05-26-rskan-v1-kanlayer.md`.

## Layout

- `rskan/` — Rust library.
- `rskan-py/` — Python bindings (PyO3 cdylib).
- `fixtures/` — pykan-exported parity fixtures (committed `.npy`).
- `scripts/` — fixture export script (run under DDR's uv venv).

## Build

```bash
cargo test --release -p rskan
maturin develop --release   # for Python bindings
```
