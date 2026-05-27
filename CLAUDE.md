# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

`rskan` is a Burn-based Rust port of `pykan`'s KAN layer. It is built to be the
GPU-capable KAN head for `~/projects/ddrs` (which currently has an MLP
placeholder in `src/nn/mlp.rs`). The Python reference lives in
`~/projects/ddr/.venv/lib/python3.13/site-packages/kan/` (pykan 0.2.x).

Design spec: `docs/superpowers/specs/2026-05-26-rskan-v1-kanlayer-design.md`.
Implementation plan: `docs/superpowers/plans/2026-05-26-rskan-v1-kanlayer.md`.

## Critical invariants

1. **f32 throughout the library.** Never introduce f64 or bf16. ddrs's
   gradient-exact invariant lives at the f32 precision floor.
2. **`KanLayerConfig::new(in_dim, out_dim, seed)` — `seed` is required.**
   No global RNG, no default seed anywhere in the library.
3. **`init` is structural-parity only against pykan**, not bit-exact RNG
   parity. For bit-exact pykan reproduction, use `init_from_parts` to load
   pykan's exported weights from `fixtures/`.
4. **Regression test that must never go red:**
   `cargo test --release -p rskan --test parity_forward -- ddr_scale_must_match_pykan`
5. **Fixtures are committed bytes.** Never edit `.npy` files by hand. Regen via:
   ```bash
   cd ~/projects/ddr && uv run python ~/projects/rskan/scripts/export_pykan_fixtures.py
   ```
6. **No `[patch.crates-io]` block in `rskan/Cargo.toml`.** When ddrs consumes
   rskan, ddrs's workspace-root patch block governs burn/cubecl unification.

## Commands

```bash
cargo build --release                                   # debug=remove --release
cargo test  --release -p rskan                          # full Rust sweep
cargo test  --release -p rskan --features cuda          # adds CUDA cross-backend test
cargo bench --bench kanlayer_forward                    # Criterion
maturin develop --release                               # build rskan-py into a Python env
uv run pytest rskan-py/tests/                           # Python-side parity
```
