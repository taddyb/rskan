# rskan v1 — KanLayer Drop-In Design

**Status:** Draft — pending user review
**Date:** 2026-05-26
**Author:** brainstormed with Claude
**Target consumer:** `~/projects/ddrs` (replaces `ddrs/src/nn/mlp.rs` as the KAN head)
**Parity oracle:** `~/projects/ddr/.venv/lib/python3.13/site-packages/kan` (pykan 0.2.x, the version DDR-Python imports)
**Reference paper:** Liu, Ma, Wang, Matusik, Tegmark — *KAN 2.0: Kolmogorov-Arnold Networks Meet Science* (arXiv:2408.10205, Aug 2024)

---

## 0. Why this exists

DDR (`~/projects/ddr`) is the Python/PyTorch reference Muskingum-Cunge routing solver. Its parameter head is a **KAN** (`pykan.KAN`) — learnable B-spline activations on edges instead of fixed activations on nodes. ddrs (`~/projects/ddrs`) is its gradient-exact BURN/Rust port, but the head was substituted by an MLP (`ddrs/src/nn/mlp.rs`) as a temporary placeholder. **rskan v1** ships the real KAN in Rust+Burn so ddrs can be truly gradient-exact against DDR end-to-end.

### Scope (locked through brainstorm)

| In scope (v1)                                                                 | Out of scope (deferred)                                              |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `KanLayer` (B-spline edge activations) + multi-layer `Kan` stack              | `update_grid_from_samples` (#3 grid refinement)                      |
| Burn 0.21 autodiff path; CPU (NdArray) + CUDA backends                        | `prune` (#4 structural sparsification)                               |
| pykan-parity init: noise→`curve2coef`→coef, seeded; `Uniform[-1,1]` scale_base | `fix_symbolic` / `auto_symbolic` (#5 symbolic regression)            |
| f32 throughout (matches ddrs's gradient-exact invariant)                      | Visualization (use pykan's own plots / DDR's offline analysis)       |
| Fixture-based numerical parity vs pykan (forward + backward)                  | LBFGS (Adam is sufficient; ddrs's training loop owns the optimizer)  |
| ddrs drop-in: rename `Mlp` → `KanHead`, swap `Linear+ReLU` blocks for `KanLayer` | Multiplication subnodes (MultKAN's `n_l^m > 0`)                   |
| Python bindings (`rskan-py`): `forward_with_grad(x, grad_y)` returns numpy + grad dict | `torch.autograd.Function` wrapper (v1.1; user-side 20-line adapter)  |
| Workspace layout: `rskan/` (Rust lib) + `rskan-py/` (PyO3 cdylib)              | dlpack zero-copy from Python (v1.1)                                  |
| Local `maturin develop`; no PyPI distribution                                  | CubeCL fused-kernel custom backward (v1.1 perf milestone)            |

Performance is **not** a v1 success criterion. Correctness against pykan is. The v1.1 milestone (a separate spec, gated on v1 parity passing) replaces the inner autodiff path with a CubeCL fused-kernel custom backward modeled on `ddrs/src/sparse.rs::CsrSolveOp`.

---

## 1. Reproducibility (cross-cutting invariant)

Every randomness source is enumerated and traced to an explicit, user-controlled seed. **There is no default seed and no global RNG.**

| Randomness source                                                | Seed origin                                                                                                  | Reproducibility                                                              |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| `KanLayerConfig::seed` (`noises` + `scale_base` sampling)        | Required field, no default. Positional arg to `KanLayerConfig::new`.                                          | Same seed → bit-exact tensors; deterministic via `StdRng::seed_from_u64`.    |
| `KanConfig::seed` (multi-layer stack)                            | Required field, no default. Per-layer sub-seed = `seed.wrapping_add(l as u64)`.                              | Deterministic across layers.                                                 |
| `KanHeadConfig::seed` (ddrs side)                                | Sourced from ddrs's `random_seed`. **Same seed passed to every inner `KanLayer` to match DDR-Python quirk.** | Matches DDR's `kan.py:24-34`.                                                |
| Pykan fixture export script (per-case)                           | `torch.manual_seed(case.weight_seed)` once at the start of each case.                                         | Re-running exporter against same pykan version → byte-identical `.npy`.     |
| Pykan fixture `x` input sampling                                 | Separate `torch.manual_seed(case.x_seed)` where `x_seed = weight_seed ^ 0xDEADBEEF`. Declared in `params.json`. | Decouples weight init from input draws.                                      |
| Examples (`tiny_regression.rs`, etc.)                            | Top-of-file `const SEED: u64 = …` constant.                                                                  | Deterministic loss curves; CI can fingerprint.                               |
| `init_smoke.rs` reproducibility test                             | Hardcoded `seed=42`; calls `init` twice with same config.                                                    | Asserts exact tensor equality across runs.                                   |
| Burn backend RNG                                                 | Not used. All sampling flows through `StdRng` then materializes on device via `Tensor::from_data`.            | Eliminates Burn-internal RNG state as a divergence source.                   |
| CUDA reduction non-determinism (gradient sums)                   | Not seedable (hardware-level FP non-associativity).                                                          | Cross-backend tests bound to `atol=1e-4` on backward; documented exception.  |

**API-level rules baked into the design:**

1. `seed: u64` is a required positional argument on `KanLayerConfig::new(in_dim, out_dim, seed)`. Cannot be forgotten.
2. `init_from_parts` carries `seed` from config but never uses it (stored for future "re-init" paths).
3. Every test file declares a `const SEED: u64` at module scope. Never `thread_rng()`.
4. Every fixture records its `(weight_seed, x_seed)` pair in `params.json`.

---

## 2. Math foundations

From KAN 2.0 paper Eq. 1-3:

```
KART (Eq. 1):    f(x) = Σ_q Φ_q( Σ_p φ_{q,p}(x_p) )
KAN layer (2):   Φ_l(x_l)_j = Σ_i φ_{l,i,j}(x_{l,i})
KAN net (3):     KAN(x) = Φ_{L-1} ∘ … ∘ Φ_0 (x)
Edge activation: φ(x) = scale_base · SiLU(x) + scale_sp · Σ_n c_n · B_n(x)
                 (c_n trainable; B_n is the n-th cubic B-spline basis on the extended grid)
```

**Pure-KAN reduction.** v1 implements MultKAN with `n_l^m = 0 ∀l` and `affine_trainable = False`. The paper (§2) proves this collapses to a sequential `KANLayer` composition with identity subnode/node affine transforms. We *structurally omit* the affine wrappers (rather than carry frozen identity Params) — see §6.D for justification.

**Subnode/node terminology.** Pykan distinguishes "subnodes" (output of a `KANLayer`) from "nodes" (input to the next, after affine transforms and any multiplication). With `affine_trainable=False` and zero multiplication subnodes, subnode = node. The rskan API does not surface this distinction.

**Tolerance justification.** Forward-parity `atol=1e-5` sits at ~4× the cubic-spline theoretical approximation floor `5⁻⁸ ≈ 2.5e-6` (paper §3.3 footnote on kanpiler), accommodating accumulated f32 rounding through three levels of Cox–de Boor recursion (~5–10 ULP per level) without being slack.

---

## 3. Architecture & crate layout

```
~/projects/rskan/                            # workspace root
├── Cargo.toml                               # [workspace] members = ["rskan", "rskan-py"]
├── README.md   CLAUDE.md                    # invariants + regen instructions
├── rskan/                                   # Rust library crate
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs                           # re-exports public API
│   │   ├── spline.rs                        # b_batch, coef2curve, curve2coef, extend_grid
│   │   ├── layer.rs                         # KanLayerConfig, KanLayer<B>
│   │   ├── kan.rs                           # KanConfig, Kan<B>
│   │   ├── init.rs                          # seeded init helpers (CPU → device materialization)
│   │   └── linalg.rs                        # tiny Cholesky solve for curve2coef
│   ├── tests/
│   │   ├── common/                          # fixture loader, tolerance helper
│   │   ├── spline_unit.rs
│   │   ├── parity_forward.rs                # fixture sweep, forward only
│   │   ├── parity_backward.rs               # fixture sweep, backward only
│   │   ├── kan_stack.rs                     # multi-layer + per-layer trajectory check
│   │   ├── init_smoke.rs                    # init reproducibility + require_grad flags
│   │   └── cross_backend.rs                 # NdArray ↔ Cuda agreement [feature="cuda"]
│   ├── benches/
│   │   └── kanlayer_forward.rs              # Criterion, NdArray + Cuda
│   └── examples/
│       └── tiny_regression.rs               # seeded "fit sin(x)" smoke
├── rskan-py/                                # PyO3 cdylib (Python bridge)
│   ├── Cargo.toml                           # crate-type = ["cdylib"]
│   ├── pyproject.toml                       # maturin config; package name "rskan"
│   ├── src/lib.rs                           # #[pymodule], PyKanLayer, PyKan
│   └── python/rskan/
│       ├── __init__.py
│       ├── _torch.py                        # stub in v1; helper example in v1.1
│       └── py.typed
├── fixtures/                                # exported by scripts/export_pykan_fixtures.py
│   ├── manifest.json                        # machine-readable list of cases
│   └── <case_name>/                         # see §7.2 for the full case schema
├── scripts/
│   └── export_pykan_fixtures.py             # run under DDR's uv venv
└── docs/superpowers/specs/                  # this file
```

### Dependencies (`rskan/Cargo.toml`)

```toml
[dependencies]
burn       = { version = "0.21", default-features = false, features = ["std", "ndarray", "autodiff"] }
burn-cuda  = { version = "0.21", default-features = false, optional = true }
thiserror  = "1"

[features]
default = []
cuda    = ["burn-cuda"]

[dev-dependencies]
burn         = { version = "0.21", default-features = false, features = ["std", "ndarray", "autodiff", "train"] }
ndarray      = "0.16"
ndarray-npy  = "0.9"
approx       = "0.5"
serde        = { version = "1", features = ["derive"] }
serde_json   = "1"
criterion    = "0.5"
rand         = "0.8"

[[bench]]
name    = "kanlayer_forward"
harness = false
```

### `[patch.crates-io]` strategy

rskan's `Cargo.toml` declares its own dev-time `[patch.crates-io]` block (mirroring ddrs's vendored cubecl + burn paths) **but gated behind a `dev-patches` feature so it's empty when consumed as a dependency.** When ddrs adds `rskan = { path = "../rskan" }`, ddrs's workspace-root `[patch.crates-io]` block is the single source of truth for the burn/cubecl unification. This is the one ecosystem risk and the one thing we verify first (`cargo tree -p rskan` from ddrs after the swap).

### Backend & precision strategy

- All public types are `<B: Backend>` generic.
- Tests run primarily under `burn::backend::Autodiff<NdArray<f32>>` (deterministic, fast to compile, easy to debug).
- Cross-backend tests (`feature = "cuda"`) verify `Autodiff<Cuda<f32>>` agrees with NdArray within tolerance.
- **f32 throughout. No mixed precision.** ddrs's gradient-exact invariant lives at the f32 floor; supporting f64/bf16 would be dead weight here and complicate `curve2coef`'s least-squares solve.

---

## 4. Public API (Rust)

### `KanLayer<B>` — `rskan/src/layer.rs`

```rust
#[derive(Config, Debug)]
pub struct KanLayerConfig {
    pub in_dim: usize,
    pub out_dim: usize,

    /// pykan `num`: grid intervals. Extended grid has `num + 1 + 2k` knots.
    #[config(default = 5)]                   pub num: usize,
    /// Spline order. pykan default 3 (cubic).
    #[config(default = 3)]                   pub k: usize,

    /// pykan KANLayer default = 0.5 (raw layer). KanConfig propagates 0.3 (MultKAN default).
    #[config(default = 0.5)]                 pub noise_scale: f64,
    #[config(default = 0.0)]                 pub scale_base_mu: f64,
    #[config(default = 1.0)]                 pub scale_base_sigma: f64,
    #[config(default = 1.0)]                 pub scale_sp: f64,
    #[config(default = "[-1.0, 1.0]")]       pub grid_range: [f64; 2],
    #[config(default = true)]                pub sp_trainable: bool,
    #[config(default = true)]                pub sb_trainable: bool,

    /// REQUIRED. No default. Positional in `KanLayerConfig::new`.
    pub seed: u64,
}

impl KanLayerConfig {
    pub fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self;
    pub fn init<B: Backend>(&self, device: &B::Device) -> KanLayer<B>;
    pub fn init_from_parts<B: Backend>(
        &self, device: &B::Device,
        grid: Tensor<B, 2>,        // [in_dim, num + 1 + 2k]
        coef: Tensor<B, 3>,        // [in_dim, out_dim, num + k]
        scale_base: Tensor<B, 2>,  // [in_dim, out_dim]
        scale_sp: Tensor<B, 2>,    // [in_dim, out_dim]
        mask: Tensor<B, 2>,        // [in_dim, out_dim]
    ) -> KanLayer<B>;
}

#[derive(Module, Debug)]
pub struct KanLayer<B: Backend> {
    pub(crate) grid:       Param<Tensor<B, 2>>,  // require_grad=false
    pub(crate) coef:       Param<Tensor<B, 3>>,  // trainable
    pub(crate) scale_base: Param<Tensor<B, 2>>,  // require_grad=sb_trainable
    pub(crate) scale_sp:   Param<Tensor<B, 2>>,  // require_grad=sp_trainable
    pub(crate) mask:       Param<Tensor<B, 2>>,  // require_grad=false
    pub k: usize,
}

impl<B: Backend> KanLayer<B> {
    /// Forward pass. `[batch, in_dim]` → `[batch, out_dim]`.
    /// Equivalent to pykan's `KANLayer.forward(x)[0]` (we drop the `(preacts, postacts,
    /// postspline)` tuple returns; they were caching/visualization only, both descoped).
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2>;
}
```

### `Kan<B>` — multi-layer stack — `rskan/src/kan.rs`

```rust
#[derive(Config, Debug)]
pub struct KanConfig {
    /// Widths from input to output. `widths=[H, H]` → one KanLayer(H→H). `widths=[H; N+1]`
    /// → N stacked KanLayers, matching DDR-Python's `num_hidden_layers=N` pattern.
    pub widths: Vec<usize>,

    /// pykan MultKAN default = 3.  (KANLayer default for `num` is 5.)
    #[config(default = 3)]                   pub grid: usize,
    #[config(default = 3)]                   pub k: usize,
    /// pykan MultKAN default = 0.3 (not 0.5 like raw KANLayer).
    #[config(default = 0.3)]                 pub noise_scale: f64,
    #[config(default = 0.0)]                 pub scale_base_mu: f64,
    #[config(default = 1.0)]                 pub scale_base_sigma: f64,
    /// **Non-pykan API surface**: pykan's MultKAN hardcodes 1.0 for inner KANLayers.
    /// Leave at default for pykan-equivalent behavior; field exists for ablations.
    #[config(default = 1.0)]                 pub scale_sp: f64,
    #[config(default = "[-1.0, 1.0]")]       pub grid_range: [f64; 2],
    #[config(default = true)]                pub sp_trainable: bool,
    #[config(default = true)]                pub sb_trainable: bool,

    pub seed: u64,  // REQUIRED
}

impl KanConfig {
    pub fn new(widths: Vec<usize>, seed: u64) -> Self;
    pub fn init<B: Backend>(&self, device: &B::Device) -> Kan<B>;
}

#[derive(Module, Debug)]
pub struct Kan<B: Backend> {
    pub(crate) layers: Vec<KanLayer<B>>,
    // Note: subnode_scale/bias, node_scale/bias structurally omitted (pure-KAN reduction).
}

impl<B: Backend> Kan<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2>;
}
```

### What is intentionally *not* in the Rust public API

| pykan API                                                  | Status     | Reason                                                |
| ---------------------------------------------------------- | ---------- | ----------------------------------------------------- |
| `update_grid_from_samples`                                 | Omitted    | Grid refinement descoped (#3).                        |
| `prune`                                                    | Omitted    | Pruning descoped (#4).                                |
| `fix_symbolic` / `auto_symbolic`                           | Omitted    | Symbolic regression descoped (#5).                    |
| `forward()` returning `(y, preacts, postacts, postspline)` | Omitted    | Tuple existed for caching/viz; return just `y`.       |
| `affine_trainable=True`                                    | Omitted    | DDR uses False; affine wrappers structurally omitted. |
| `sparse_init=True`                                         | Omitted    | DDR uses False.                                       |
| `singularity_avoiding` / `y_th`                            | Omitted    | Symbolic-only feature.                                |
| `save_act`, `cache_data`                                   | Omitted    | Visualization-only.                                   |
| `to(device)`                                               | Omitted    | Burn handles via `Module::to_device`.                 |

Errors: invalid configs panic via `assert!` at construction with descriptive messages (matches `burn::nn::LinearConfig`'s posture). No `Result` in the public surface.

---

## 5. Internal math (`rskan/src/spline.rs`)

Notation: `B = batch`, `I = in_dim`, `O = out_dim`, `G = num` (grid intervals), `k` = spline order, `K = G + 1 + 2k` extended-knot count, `n_basis = G + k`.

### 5.1 `extend_grid` — init-only, CPU

```rust
fn extend_grid(grid: ArrayView2<f32>, k_extend: usize) -> Array2<f32>
// Input:  shape [I, G+1] uniform linspace
// Output: shape [I, G+1+2k] with k ghost knots each side, spaced by h = (grid[:,-1] - grid[:,0]) / G.
```

Implemented analytically (no per-step `cat`): prepend `[grid[:,0] - k·h, …, grid[:,0] - h]`, append `[grid[:,-1] + h, …, grid[:,-1] + k·h]`. Bitwise-equivalent to pykan's `spline.py:extend_grid` iterative `cat` loop, just clearer.

### 5.2 `b_batch` — runtime, autograd

Cox–de Boor B-spline basis recursion, rewritten as an **iterative loop** (vs pykan's Python recursion) to keep the autodiff tape linear in `k`.

```rust
pub(crate) fn b_batch<B: Backend>(
    x: Tensor<B, 2>,       // [B, I]
    grid: Tensor<B, 2>,    // [I, K]
    k: usize,
) -> Tensor<B, 3>          // [B, I, n_basis]
```

Algorithm (slicing/broadcast pseudo-code; real Burn 0.21 ops: `Tensor::slice`, `unsqueeze_dim`, `greater_equal`, `lower`, `mask_fill`):

```rust
let x3    = x.unsqueeze_dim::<3>(2);                          // [B, I, 1]
let grid3 = grid.unsqueeze_dim::<3>(0);                       // [1, I, K]

// k=0 base case: indicator (x ∈ [grid[i], grid[i+1]))
let lo = grid3.clone().slice([.., .., 0..K-1]);
let hi = grid3.clone().slice([.., .., 1..K]);
let mut v = x3.clone().greater_equal(lo).float()
          * x3.clone().lower(hi).float();                     // [B, I, K-1]

// Iterative Cox-de Boor: k_curr in 1..=k
for k_curr in 1..=k {
    let len_prev = K - k_curr;
    let g_a = grid3.clone().slice([.., .., 0..(K - k_curr - 1)]);
    let g_b = grid3.clone().slice([.., .., k_curr..(K - 1)]);
    let g_c = grid3.clone().slice([.., .., (k_curr + 1)..K]);
    let g_d = grid3.clone().slice([.., .., 1..(K - k_curr)]);

    let lf = (x3.clone() - g_a.clone()) / (g_b - g_a);
    let rf = (g_c.clone() - x3.clone()) / (g_c - g_d);

    let v_l = v.clone().slice([.., .., 0..(len_prev - 1)]);
    let v_r = v.slice([.., .., 1..len_prev]);

    v = lf * v_l + rf * v_r;                                  // [B, I, len_prev - 1]
}

// pykan parity: nan_to_num (only fires on degenerate grids; no-op for ours).
let nan_mask = v.clone().is_nan();
v.mask_fill(nan_mask, 0.0)
```

Tape size: O(k) levels of `Tensor<B, 3>` intermediates, each ~`[B, I, n_basis ± O(1)]`. For DDR's `H=21, k=3, G=5, batch≈5000`: ~50 KB f32 × 3 levels × N_layers — trivial on GPU.

### 5.3 `coef2curve` — runtime, autograd

Pykan's `einsum('ijk,jlk->ijl', b, coef)` rewritten as a batched matmul (Burn 0.21 has no general einsum):

```rust
pub(crate) fn coef2curve<B: Backend>(
    x_eval: Tensor<B, 2>,      // [B, I]
    grid:   Tensor<B, 2>,      // [I, K]
    coef:   Tensor<B, 3>,      // [I, O, n_basis]
    k:      usize,
) -> Tensor<B, 3>              // [B, I, O]
{
    let b   = b_batch(x_eval, grid, k);         // [B, I, n_basis]
    let b_p = b.permute([1, 0, 2]);             // [I, B, n_basis]
    let c_p = coef.permute([0, 2, 1]);          // [I, n_basis, O]
    b_p.matmul(c_p).permute([1, 0, 2])          // [I, B, O] → [B, I, O]
}
```

Verified: Burn 0.21's `Tensor::matmul` (`burn-tensor/src/tensor/api/numeric.rs:915`) broadcasts batched matmul over leading dims when `D ≥ 3`.

### 5.4 `curve2coef` — init-only, CPU, detached

Pykan uses `torch.linalg.lstsq` (Burn lacks it). Since this runs once at init with no gradient required, we solve on `ndarray` and convert to a device tensor at the end.

Strategy: solve `(MᵀM + λI) C = Mᵀ Y` (Tikhonov-regularized normal equations, `λ = 1e-8`) via a hand-rolled Cholesky factorization in `rskan/src/linalg.rs`. Matrices are tiny (`n_basis × n_basis` = e.g. 8×8 for DDR scale) — no LAPACK linkage needed.

```rust
fn curve2coef(
    x_eval: ArrayView2<f32>,    // [batch, I]
    y_eval: ArrayView3<f32>,    // [batch, I, O]  (the `noises` from init)
    grid:   ArrayView2<f32>,    // [I, K]
    k:      usize,
) -> Array3<f32>                // [I, O, n_basis]
```

Per-input-dim solve: factorize `MᵀM + λI` once per `i ∈ 0..I`, then apply to all `O` right-hand-sides at once. Total work for DDR scale: `21 × (8³ + 8² × 21) ≈ 40k flops`. Microseconds.

**Numerical floor**: ridge-normal-equations on well-conditioned B-spline bases matches `lstsq` to ≤ 1e-6 in f32. Fallback if a parity test ever fails on `coef`: implement Householder QR (~50 LOC, same matrix sizes). Documented but not built in v1.

Doc note: *"Recommended `num ≤ 20`; larger grids may show init drift vs pykan beyond the forward-parity tolerance."*

### 5.5 `is_nan` / `mask_fill` availability

Confirmed: Burn 0.21 has `Tensor::is_nan` (`api/float.rs:559`) and `Tensor::mask_fill` (`api/base.rs:1745`). Both used in ddrs (`mmc_op.rs`).

---

## 6. Init recipe (`rskan/src/init.rs`)

**Structural-only RNG parity.** Our `KanLayerConfig::init` produces statistically-equivalent weights to pykan's `KANLayer(seed=…)` but not bit-identical ones (PyTorch's Mersenne-Twister vs Rust's `StdRng`). For bit-equivalence, use `init_from_parts` to load pykan's exported tensors. Documented in `CLAUDE.md`.

### A. The five-step `KanLayer` init

For a `KanLayerConfig { in_dim, out_dim, num, k, noise_scale, scale_base_mu, scale_base_sigma, scale_sp, grid_range, sp_trainable, sb_trainable, seed }`:

```rust
pub fn init<B: Backend>(&self, device: &B::Device) -> KanLayer<B> {
    let (I, O, G, k_) = (self.in_dim, self.out_dim, self.num, self.k);
    let mut rng = StdRng::seed_from_u64(self.seed);
    let inv_sqrt_in = 1.0 / (I as f32).sqrt();

    // (1) grid: uniform linspace + k ghost knots each side. Shape [I, G+1+2k]. Frozen.
    let lo = self.grid_range[0] as f32;
    let hi = self.grid_range[1] as f32;
    let row = Array1::linspace(lo, hi, G + 1);
    let grid_pre: Array2<f32> = Array2::from_shape_fn((I, G + 1), |(_, j)| row[j]);
    let grid_full: Array2<f32> = extend_grid(grid_pre.view(), k_);

    // (2) noise targets ~ U(-1/2, 1/2) * (noise_scale / G). Shape [G+1, I, O].
    let noise_amp = (self.noise_scale / G as f64) as f32;
    let noises: Array3<f32> = Array3::from_shape_fn(
        (G + 1, I, O), |_| (rng.gen::<f32>() - 0.5) * noise_amp);

    // (3) coef = curve2coef(grid_inner.T, noises, grid_full, k). Shape [I, O, G+k]. Trainable.
    let grid_inner_t = grid_full.slice(s![.., k_..k_ + G + 1]).t().to_owned();
    let coef = curve2coef(grid_inner_t.view(), noises.view(), grid_full.view(), k_);

    // (4) mask = ones[I, O]. Frozen. (sparse_init=False in v1.)
    let mask = Array2::<f32>::ones((I, O));

    // (5a) scale_base ~ (mu + sigma * U[-1, 1]) / sqrt(in_dim). U[-1,1] not Normal (pykan
    //      code at KANLayer.py:110 uses Uniform despite the docstring claiming Normal).
    let mu = self.scale_base_mu as f32;
    let sigma = self.scale_base_sigma as f32;
    let scale_base = Array2::from_shape_fn(
        (I, O), |_| (mu + sigma * (rng.gen::<f32>() * 2.0 - 1.0)) * inv_sqrt_in);

    // (5b) scale_sp = scale_sp_arg / sqrt(in_dim) (mask=ones, omitted from product).
    let scale_sp = Array2::from_elem((I, O), (self.scale_sp as f32) * inv_sqrt_in);

    KanLayer {
        grid:       to_param_2(grid_full,  device).set_require_grad(false),
        coef:       to_param_3(coef,       device),
        scale_base: to_param_2(scale_base, device).set_require_grad(self.sb_trainable),
        scale_sp:   to_param_2(scale_sp,   device).set_require_grad(self.sp_trainable),
        mask:       to_param_2(mask,       device).set_require_grad(false),
        k: k_,
    }
}
```

`to_param_N(arr, device) = Param::from_tensor(Tensor::from_data(TensorData::from(arr), device))`. `Param::set_require_grad(bool)` is confirmed in Burn 0.21 (`burn-core/src/module/param/base.rs:415`).

### B. `init_from_parts`

Shape-checks all five tensors against `(in_dim, out_dim, num, k)`, panics on mismatch, wraps each in `Param` with the trainability flags from `self`. No RNG consumed. Used by fixture parity tests and by any future "load pretrained" path.

### C. `KanConfig::init` — multi-layer

```rust
pub fn init<B: Backend>(&self, device: &B::Device) -> Kan<B> {
    assert!(self.widths.len() >= 2);
    assert!(self.k >= 1 && self.grid >= 1);

    let layers: Vec<KanLayer<B>> = (0..self.widths.len() - 1).map(|l| {
        KanLayerConfig::new(self.widths[l], self.widths[l + 1],
                            self.seed.wrapping_add(l as u64))
            .with_num(self.grid)
            .with_k(self.k)
            .with_noise_scale(self.noise_scale)     // 0.3 from MultKAN default
            .with_scale_base_mu(self.scale_base_mu)
            .with_scale_base_sigma(self.scale_base_sigma)
            .with_scale_sp(self.scale_sp)           // 1.0; matches MultKAN's hardcode
            .with_grid_range(self.grid_range)
            .with_sp_trainable(self.sp_trainable)
            .with_sb_trainable(self.sb_trainable)
            .init(device)
    }).collect();
    Kan { layers }
}
```

(Note: ddrs's `KanHead::init` deliberately *overrides* this sub-seed derivation to use the same seed for every inner KanLayer — see §8 for the DDR-quirk rationale.)

### D. Affine wrappers omitted (paper-validated)

KAN 2.0 §2: when `n_l^m = 0 ∀l` and `affine_trainable = False`, MultKAN reduces to pure sequential `KANLayer` composition. We omit `node_scale/bias` and `subnode_scale/bias` Params entirely (rather than carry frozen identities) because:

1. DDR's call pattern (`KAN([H, H], …)` with default `affine_trainable=False`) hits exactly this reduction.
2. Carrying identity Params would inflate model size and create dead-weight in saved checkpoints.
3. The `Module` derive on `Kan<B>` stays minimal.

If `affine_trainable=True` is ever needed (it isn't, for ddrs), that becomes a v2 spec — non-additive API change, structurally adds new fields.

---

## 7. Verification harness

The harness is the single source of truth for "rskan v1 ≡ pykan." Every parity claim is backed by a committed fixture file.

### 7.1 Export script — `scripts/export_pykan_fixtures.py`

Run under DDR's uv venv:

```bash
cd ~/projects/ddr && uv run python ~/projects/rskan/scripts/export_pykan_fixtures.py
```

Per case: `torch.manual_seed(weight_seed)` → build pykan module → `torch.manual_seed(x_seed)` → sample `x` → `y = model(x)` → `y.sum().backward()` → dump tensors + grads as `.npy` + `params.json`.

For multi-layer cases the script registers forward hooks on each `KANLayer` to capture the trajectory `x → x_l0 → x_l1 → … → y` so per-layer parity can be verified.

### 7.2 Fixture cases (committed bytes)

```
fixtures/
├── manifest.json
├── README.md                                  # regen instructions
├── kanlayer_i3_o5_k2_g3_s1/                   # small, k=2
├── kanlayer_i8_o8_k3_g5_s1/                   # cubic, standard
├── kanlayer_i1_o1_k3_g5_s1/                   # single-edge degenerate
├── kanlayer_i21_o21_k3_g5_s1/                 # DDR-scale  ← REGRESSION CASE
├── kanlayer_i3_o3_k3_g5_s1_ood/               # OOD x values (boundary + outside grid_range)
├── kanlayer_i3_o3_k3_g1_s1/                   # num=1 degenerate single-interval grid
├── kan_w[21,21,21]_k3_g5_s1/                  # 2-layer stack (DDR num_hidden_layers=2)
└── kan_w[8,8,8,8]_k3_g5_s1/                   # 3-layer stack
```

Total disk: ~100 KB. Each case's `params.json` includes `weight_seed`, `x_seed`, all hyperparameters, and a `pykan_version` field for traceability.

### 7.3 Rust-side tests

```
rskan/tests/
├── common/                                    # fixture loader, assert_close helper
├── spline_unit.rs                             # b_batch / extend_grid / curve2coef-coef2curve roundtrip
├── parity_forward.rs                          # fixture sweep, NdArray, forward only
├── parity_backward.rs                         # fixture sweep, Autodiff<NdArray>, backward only
├── kan_stack.rs                               # multi-layer + per-layer trajectory check
├── init_smoke.rs                              # seed reproducibility + require_grad flags + save/load roundtrip
└── cross_backend.rs                           # NdArray ↔ Cuda agreement [#[cfg(feature="cuda")]]
```

The `kanlayer_i21_o21_k3_g5_s1` case is elevated to **regression status** via `#[test] fn ddr_scale_must_match_pykan` in `parity_forward.rs`. Pinned in `CLAUDE.md`. Mirrors ddrs's `compare_ddr_sandbox` "ABSOLUTE MATCH" pattern.

### 7.4 Tolerance policy

| Path                                          | atol  | rtol  | Justification                                                              |
| --------------------------------------------- | ----- | ----- | -------------------------------------------------------------------------- |
| Forward (NdArray)                             | 1e-5  | 1e-4  | 4× cubic-spline approx floor `5⁻⁸ ≈ 2.5e-6` + 5–10 ULP Cox–de Boor.        |
| Backward (NdArray)                            | 1e-4  | 1e-3  | One order of magnitude looser for accumulated reduction rounding.          |
| Cross-backend forward (NdArray ↔ Cuda)        | 1e-5  | 1e-4  | Should match within FMA-equivalent rounding.                               |
| Cross-backend backward (NdArray ↔ Cuda)       | 1e-4  | 1e-3  | CUDA reduction-order non-determinism.                                      |
| Init reproducibility (same seed, twice)       | exact | exact | StdRng is deterministic; drift = bug.                                      |

All tolerances are `const` in `rskan/tests/common/tolerances.rs` — one place to tune.

### 7.5 CI / reproducibility

- `cargo test -p rskan --release` — full Rust sweep, no Python. ~5 s.
- `cargo test -p rskan --release --features cuda` — adds cross-backend (skipped without CUDA driver).
- `uv run pytest tests/python/` — Python-side parity against pykan (requires `pykan` in venv).
- Fixtures committed to git; CI never invokes Python or pykan.

### 7.6 Edge cases covered

- OOD x values triggering `nan_to_num` (boundary + outside grid_range).
- `num=1` degenerate single-interval grid.
- Single-edge `in_dim=out_dim=1`.
- `k ∈ {2, 3}` (production cases).
- Multi-layer trajectory match (catches any hidden pykan `MultKAN.forward` path divergence under `affine_trainable=False`).

---

## 8. ddrs integration (`KanHead`)

### 8.1 What rskan replaces

`ddrs/src/nn/mlp.rs`'s `Mlp<B>` becomes `kan_head.rs`'s `KanHead<B>`:

```text
KanHead<B>:
  input  : Linear<B>                          // F → H,  kaiming_normal_(relu) init
  hidden : Vec<rskan::KanLayer<B>>            // N × KanLayer(H, H, k, grid)
  output : Linear<B>                          // H → P,  xavier_normal_(gain=0.1) init
  learnable_parameters: Vec<String>
```

Forward path:

```text
x [N, F]
  └─ input (Linear+bias)                      // NO ReLU after — see §8.2
  └─ hidden[0]  KanLayer(H, H)                // internal SiLU + spline
  └─ hidden[1]  KanLayer(H, H)
  └─ … hidden[num_hidden_layers - 1]
  └─ output (Linear+bias)
       └─ Sigmoid
       └─ per-parameter HashMap<String, Tensor<B, 1>>
```

### 8.2 Decision: drop the inter-block ReLU

DDR-Python's `kan.py:53` writes `_x = self.input(_x)` then jumps directly into KAN layers — **no ReLU between the input Linear and the first KAN block**. ddrs's MLP era kept a ReLU there only because the head was structurally an MLP. We **remove the ReLU** so the head matches DDR-Python and the gradient-exact invariant holds.

This is a behavioral change. The pre-rskan `Mlp` outputs and the post-rskan `KanHead` outputs will differ; any ddrs test asserting specific head outputs needs re-fixturing. Audit before commit. The `compare_ddr_sandbox` fixture must be regenerated from DDR-Python with its native KAN head — this requires DDR to do a (brief) end-to-end training run on the sandbox, not just a parameter dump. Documented in ddrs's `CLAUDE.md` post-swap.

### 8.3 Decision: same seed for all inner KanLayers

DDR-Python's `kan.py:24-34` constructs every inner `KAN([H, H])` with `seed=seed` — the same seed for all `num_hidden_layers` blocks. With pykan's `manual_seed`-then-construct pattern, this means each inner KAN module's RNG state starts identically.

**`KanHead::init` overrides `rskan::KanConfig`'s `wrapping_add(l)` sub-seed derivation** to pass the same `seed` to every inner `KanLayerConfig::new(...)`. Diverges from rskan's own `KanConfig` convention but matches DDR-Python's actual quirk. Trade-off: independent layer init (better statistics) vs. parity (the v1 goal). We choose parity.

```rust
let hidden: Vec<KanLayer<B>> = (0..self.num_hidden_layers).map(|_l| {
    KanLayerConfig::new(self.hidden_size, self.hidden_size, self.seed)  // SAME seed every layer
        .with_num(self.grid)
        .with_k(self.k)
        .with_noise_scale(0.3)        // MultKAN default — matches DDR
        .init(device)
}).collect();
```

If DDR ever fixes this upstream (unique seeds per layer), ddrs's `KanHead::init` changes in lockstep — single-file update.

### 8.4 Config plumbing

`KanHeadConfig` gains `grid: usize`, `k: usize`, `seed: u64` (required). Sourced from `merit_training.yaml` (DDR's config already has `grid: 5, k: 3`); ddrs adds the same keys. Defaults `grid=5, k=3` match DDR production.

### 8.5 Cargo wiring

```toml
# ddrs/Cargo.toml
[dependencies]
rskan = { path = "../rskan" }
# No new [patch.crates-io] entries — rskan inherits ddrs's vendored cubecl + burn
# via cargo's "patches apply at workspace root" rule.
```

Pre-flight check: `cargo tree -p rskan` from ddrs must show `burn-std`, `cubecl-cuda` resolving exactly once. If duplicate trait-object errors appear, rskan's `[patch.crates-io]` block needs to be gated behind a `dev-patches` feature.

### 8.6 ddrs regression gates

1. **`cargo run --release --example compare_ddr_sandbox`** must still report ABSOLUTE MATCH (`max abs diff < 1e-3 m³/s`). The sandbox fixture is regenerated against DDR's native KAN head.
2. **New `cargo test --test rskan_head_parity`** — per-edge gradient parity between `KanHead` (built from fixtures) and DDR-Python's `kan(...)` on the same inputs. Strictly stronger than the routing-level gate (head drift can be masked by Manning's-n smoothing).

### 8.7 Migration commit sequence

1. Ship rskan v1 standalone (all tests green).
2. ddrs: rename `mlp.rs` → `kan_head.rs`, swap `Vec<Linear<B>>` for `Vec<KanLayer<B>>`. Get it compiling with the old ReLU still in place.
3. Regenerate `compare_ddr_sandbox.py` fixture under DDR uv venv with DDR's KAN head.
4. Run `compare_ddr_sandbox` — expect a degradation from the residual ReLU.
5. Drop the inter-block ReLU. Rerun — ABSOLUTE MATCH should hold.
6. Add `tests/rskan_head_parity.rs` in ddrs.
7. Update `ddrs/CLAUDE.md` invariants.

### 8.8 What we explicitly *don't* touch in ddrs

`routing/`, `sparse.rs`, `geometry.rs`, `config.rs`, `data/`, the existing `[patch.crates-io]` block, the `burn_custom_backward.md` skill. The diff is surgical: one rename, one new dep, two new YAML keys, one new test.

---

## 9. Python bindings (`rskan-py`)

### 9.1 Scope: `forward_with_grad` returns numpy + grad dict

Python passes inputs in and gets back **both** `y` and the gradient bundle. No `torch.autograd.Function` integration in v1 — that becomes a 20-line user-side adapter on top of our API (or v1.1, if we ship it as a helper).

### 9.2 Public Python API

```python
import numpy as np
import rskan

# Construction (pykan-parity defaults; seed REQUIRED — no kwarg default)
layer = rskan.KanLayer(
    in_dim=21, out_dim=21, num=5, k=3,
    noise_scale=0.5, scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0,
    grid_range=(-1.0, 1.0), sp_trainable=True, sb_trainable=True,
    seed=1, device="cpu",                        # "cuda:0" if wheel built with feature="cuda"
)

# Multi-layer
model = rskan.Kan(
    widths=[21, 21, 21], grid=3, k=3, noise_scale=0.3, scale_sp=1.0,
    seed=1, device="cpu",
)

# Pure forward (no autodiff allocation)
y = layer.forward(x)                             # → np.ndarray (B, O), float32

# Forward + gradient extraction
y, grads = layer.forward_with_grad(x)            # implicit grad_y = ones
y, grads = layer.forward_with_grad(x, grad_y=g)  # explicit upstream gradient
# grads is a dict:
#   grads["x"]          shape (B, I)
#   grads["coef"]       shape (I, O, num+k)
#   grads["scale_base"] shape (I, O)
#   grads["scale_sp"]   shape (I, O)
# grid and mask: frozen, no key.

# Parameter access (copies)
layer.grid(), layer.coef(), layer.scale_base(), layer.scale_sp(), layer.mask()

# Fixture-style construction — bypass init
layer = rskan.KanLayer.from_parts(
    grid=np_grid, coef=np_coef,
    scale_base=np_sb, scale_sp=np_ss, mask=np_mask,
    k=3, device="cpu",
)

# Parameter update (rebuild; no in-place writes in v1)
layer = rskan.KanLayer.from_parts(
    grid=layer.grid(), coef=layer.coef() - lr * grads["coef"],
    scale_base=layer.scale_base() - lr * grads["scale_base"],
    scale_sp=layer.scale_sp() - lr * grads["scale_sp"],
    mask=layer.mask(), k=3, device="cpu",
)
```

### 9.3 Implementation sketch — `rskan-py/src/lib.rs`

```rust
#[pymethods]
impl PyKanLayer {
    fn forward<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f32>)
        -> PyResult<Py<PyArray2<f32>>> { /* no autodiff path */ }

    fn forward_with_grad<'py>(
        &self, py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        grad_y: Option<PyReadonlyArray2<'py, f32>>,
    ) -> PyResult<(Py<PyArray2<f32>>, PyObject /* dict */)> {
        // 1. Move x to Burn Autodiff<NdArray<f32>> backend with require_grad.
        // 2. y = self.inner.forward(x_tape.clone());
        // 3. loss = match grad_y { None => y.clone().sum(),
        //                          Some(g) => (y.clone() * Tensor::from(g)).sum() };
        // 4. grads = loss.backward();
        // 5. Extract grad_x, grad_coef, grad_scale_base, grad_scale_sp via grads.get(&param.id()).
        // 6. Convert each to numpy, build dict, return (y_numpy, grads_dict).
    }
}
```

Two type-erasure dispatch variants at the FFI boundary:

```rust
enum KanLayerImpl {
    Cpu(rskan::KanLayer<burn::backend::Autodiff<burn::backend::NdArray<f32>>>),
    #[cfg(feature = "cuda")]
    Cuda(rskan::KanLayer<burn::backend::Autodiff<burn_cuda::Cuda<f32>>>),
}
```

Burn-side cost per `forward_with_grad`: one forward + one backward. ~10 ms on NdArray at DDR scale; fine for verification and inference, hot enough that v1.1 CubeCL custom-backward would help.

### 9.4 Python-side verification

`tests/python/test_parity_pykan.py` covers both forward and backward against pykan:

```python
def test_kanlayer_forward_backward_parity(case):
    fix = load_fixture(case)
    rskan_layer = rskan.KanLayer.from_parts(**fix.params_dict, **fix.tensors)
    pykan_layer = build_pykan_layer(**fix.params_dict)
    pykan_layer.load_state_dict(fix.pykan_state_dict)

    y_rskan, grads_rskan = rskan_layer.forward_with_grad(fix.x)

    x_t = torch.tensor(fix.x, requires_grad=True)
    y_t, *_ = pykan_layer(x_t)
    y_t.sum().backward()

    np.testing.assert_allclose(y_rskan, y_t.detach().numpy(), atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(grads_rskan["x"], x_t.grad.numpy(), atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(grads_rskan["coef"],
                               pykan_layer.coef.grad.numpy(), atol=1e-4, rtol=1e-3)
    # ... scale_base, scale_sp
```

Catches FFI marshaling bugs (wrong stride, wrong dtype, transposition errors at the numpy ↔ Burn boundary) that pure Rust tests can't see.

### 9.5 Deferred to v1.1

- `rskan._torch.autograd_function` — `torch.autograd.Function` adapter over `forward_with_grad`.
- dlpack zero-copy.
- In-place parameter updates from Python.
- Tape caching for repeated backward calls.

---

## 10. Out-of-scope catalog + future work

### 10.1 Out of scope (v1)

| Feature                                            | Spec where it would live              | Trigger for adding                                |
| -------------------------------------------------- | ------------------------------------- | ------------------------------------------------- |
| Grid refinement (`update_grid_from_samples`)        | `rskan v1.x — grid refinement`        | When DDR/ddrs needs adaptive grids between epochs |
| Pruning (`prune`)                                  | `rskan v1.x — pruning`                 | When sparse routing-param nets are desired        |
| Symbolic regression (`fix_symbolic`/`auto_symbolic`) | `rskan v2 — symbolic`                  | If KAN interpretability becomes a research goal   |
| Multiplication subnodes (`n_l^m > 0`)              | `rskan v2 — MultKAN`                   | If multiplicative inductive bias proves useful    |
| `affine_trainable=True`                            | `rskan v2 — affine wrappers`           | Currently unneeded; non-additive API change      |
| Sparse init (`sparse_init=True`)                   | `rskan v2 — pruning`                   | Same trigger as pruning                           |
| LBFGS optimizer                                    | `n/a`                                  | ddrs's Adam-based training loop owns this        |
| Visualization                                      | `n/a`                                  | Use pykan or DDR's offline notebooks             |

### 10.2 v1.1 — perf milestone

CubeCL fused-kernel custom backward for `b_batch` + `coef2curve`, modeled on `ddrs/src/sparse.rs::CsrSolveOp`. Replaces the inner forward+autodiff path while keeping the public API identical. Gated on a numerical-parity test vs v1's pure-autodiff path. Separate spec.

### 10.3 v1.2 — Python integration polish

- `rskan._torch` autograd wrapper (~20 LOC).
- dlpack zero-copy bridge.
- Tape caching for repeated `forward_with_grad` calls on identical inputs.

Each is a separate spec with its own brainstorm → plan → implement cycle.

---

## 11. Risk register

| Risk                                                                 | Probability | Mitigation                                                                                       |
| -------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------ |
| `[patch.crates-io]` unification between rskan and ddrs fails         | Medium      | Gate rskan's patch block behind `dev-patches` feature; verify via `cargo tree -p rskan` early.   |
| `curve2coef` ridge-normal-equations drifts beyond f32 tolerance vs `lstsq` for larger `num` | Low         | Householder QR fallback documented in §5.4. Caps `num ≤ 20` in docs.                              |
| Multi-layer pykan trajectory diverges from naive `Kan` composition under `affine_trainable=False` | Low         | `kan_stack.rs` trajectory test written FIRST during implementation; surfaces divergence early.   |
| GPU backward non-determinism breaks cross-backend tolerance          | Low–Medium  | Cross-backend backward at `atol=1e-4`. CUDA-vs-pykan parity not tested directly (NdArray-only).  |
| Pykan publishes 0.3.x with API changes                               | Medium      | Pin `pykan` to DDR's `uv.lock` version. Fixtures are committed bytes; CI doesn't need pykan.    |
| `compare_ddr_sandbox` fixture regeneration requires DDR training run | Certain     | Documented in ddrs `CLAUDE.md`. One-time cost. Audited test list before swap.                    |
| FFI marshaling bug at numpy ↔ Burn boundary                          | Medium      | Python-side parity tests are the explicit guard. Strict shape/dtype asserts in PyO3 layer.       |

---

## 12. Glossary

| Term                | Meaning                                                                              |
| ------------------- | ------------------------------------------------------------------------------------ |
| `B`                 | batch size                                                                           |
| `I`, `O`            | input dimension, output dimension of a KanLayer                                      |
| `G` (`num`)         | number of grid intervals                                                             |
| `k`                 | spline order (default 3 = cubic)                                                     |
| `K`                 | extended-knot count, `G + 1 + 2k`                                                    |
| `n_basis`           | basis function count, `G + k`                                                        |
| subnode             | output of a `KANLayer` (pykan terminology; pre-multiplication)                       |
| node                | input to a `KANLayer` (post-affine-transform; with `affine_trainable=False`, = subnode) |
| KART                | Kolmogorov–Arnold Representation Theorem (paper Eq. 1)                               |
| pure-KAN reduction  | MultKAN with `n_l^m = 0 ∀l` and `affine_trainable = False` → sequential KANLayers   |
| structural parity   | matching shapes, distributions, and gradients — but not bit-exact RNG byte sequences |
| `init_from_parts`   | non-RNG constructor that loads pre-existing weights (e.g., from pykan fixtures)      |

---

## 13. References

1. Liu, Ma, Wang, Matusik, Tegmark. *KAN 2.0: Kolmogorov-Arnold Networks Meet Science*. arXiv:2408.10205 (Aug 2024). `~/Downloads/2408.10205v1.pdf`.
2. Liu et al. (original KAN). Cited as [57] in KAN 2.0. Defines edge activation `φ(x) = b(x) + spline(x)` with `b = SiLU`.
3. pykan source at `~/projects/ddr/.venv/lib/python3.13/site-packages/kan/` — `KANLayer.py`, `MultKAN.py`, `spline.py`.
4. DDR reference: `~/projects/ddr/src/ddr/nn/kan.py`, `~/projects/ddr/src/ddr/geometry/predictor.py`, `~/projects/ddr/CLAUDE.md`.
5. ddrs (consumer): `~/projects/ddrs/src/nn/mlp.rs`, `~/projects/ddrs/CLAUDE.md`, `~/projects/ddrs/.claude/skills/burn_custom_backward.md`.
6. Burn 0.21 source (vendored): `~/projects/burn/crates/burn-tensor/`, `~/projects/burn/crates/burn-core/`.
