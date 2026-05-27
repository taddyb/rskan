# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

`rskan` is a Burn-based Rust port of `pykan`'s KAN layer. It is built to be the
GPU-capable KAN head for `~/projects/ddrs` (which currently has an MLP
placeholder in `src/nn/mlp.rs`). The Python reference lives in
`~/projects/ddr/.venv/lib/python3.13/site-packages/kan/` (pykan 0.2.x).

Design spec: `docs/superpowers/specs/2026-05-26-rskan-v1-kanlayer-design.md`.
Implementation plan: `docs/superpowers/plans/2026-05-26-rskan-v1-kanlayer.md`.
Regression list:   `docs/REGRESSION.md`.

## Critical invariants — break these and the port is meaningless

1. **f32 throughout the library.** Never introduce f64 or bf16. ddrs's
   gradient-exact invariant lives at the f32 precision floor.
2. **`seed` is required everywhere.** `KanLayerConfig::new(in_dim, out_dim, seed)`
   takes `seed` as a positional argument with no default. Same on `KanConfig::new`.
   Python bindings require `seed=` as a kwarg. No global RNG anywhere.
3. **`KanLayerConfig::init` is structural-parity only against pykan**, not
   bit-exact RNG parity. For bit-equivalent pykan reproduction, use
   `init_from_parts` to load pykan's exported weights from `fixtures/`.
4. **Don't replace the autodiff forward path with custom kernels in v1.** The
   CubeCL fused-kernel backward is a v1.1 milestone with its own spec; gated
   on a numerical-parity test vs the v1 autodiff path.
5. **Regression test that must never go red:**
   `cargo test --release -p rskan --test parity_forward -- ddr_scale_must_match_pykan`
   See `docs/REGRESSION.md` for the full pinned list.
6. **Fixtures are committed bytes.** Never edit `.npy` files by hand. Regen via:
   ```bash
   cd ~/projects/ddr && uv run python ~/projects/rskan/scripts/export_pykan_fixtures.py
   ```
7. **No `[patch.crates-io]` block in `rskan/Cargo.toml`.** When ddrs consumes
   rskan, ddrs's workspace-root patch block governs burn/cubecl unification.

## Architecture in one screen

```
rskan/
├── rskan/src/
│   ├── spline.rs           B-spline math: b_batch (Burn + ndarray),
│   │                       coef2curve (Burn), curve2coef (CPU), extend_grid
│   ├── linalg.rs           Hand-rolled Cholesky solver (init-only)
│   ├── init.rs             Seeded init helpers (StdRng → ndarray → Param)
│   ├── layer.rs            KanLayerConfig, KanLayer<B>
│   └── kan.rs              KanConfig, Kan<B> (pure-KAN reduction)
├── rskan-py/               PyO3 cdylib: PyKanLayer, PyKan + forward_with_grad
├── fixtures/               pykan-exported .npy ground truth (committed)
└── scripts/                Python exporter (run under DDR's uv venv)
```

Pure-KAN reduction: v1 implements MultKAN with all multiplication subnodes
absent and `affine_trainable=False`. The paper (§2 of arXiv:2408.10205) proves
this collapses to sequential `KanLayer`s with identity affine wrappers; we
structurally omit those Params.

## Commands

```bash
cargo build  --release
cargo test   --release -p rskan                  # full parity sweep
cargo test   --release -p rskan --features cuda  # adds NdArray↔Cuda
cargo bench  --bench kanlayer_forward            # Criterion
cargo run    --release -p rskan --example tiny_regression
maturin develop --release                        # build rskan-py
cd ~/projects/ddr && uv run pytest ~/projects/rskan/rskan-py/tests/
```

## ddrs integration (forward-looking)

When ddrs consumes rskan:

1. Drop the inter-block ReLU in `ddrs/src/nn/mlp.rs` (matches DDR-Python's
   `kan.py:53` direct chaining). This is a behavioral change — re-fixture any
   ddrs head-output tests.
2. Use the **same seed for every inner `KanLayer`** in `KanHead::init`
   (DDR-Python's `kan.py:24-34` quirk: same `seed=seed` to every `KAN([H,H])`).
   This OVERRIDES rskan's own `KanConfig::init` sub-seed derivation.
3. Regenerate `compare_ddr_sandbox.py` against DDR's native KAN head (a brief
   end-to-end training run), then verify ABSOLUTE MATCH (< 1e-3 m³/s) on the
   5-reach RAPID sandbox.

See spec §8 for the full migration sequence.
