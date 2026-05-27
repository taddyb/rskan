//! `KanLayer` — B-spline edge activations on `[in_dim, out_dim]` edges.
//!
//! Forward pass: `y[b, o] = Σ_i mask[i, o] · (scale_base[i, o] · SiLU(x[b, i])
//!                                          + scale_sp[i, o] · spline_{i,o}(x[b, i]))`
//! where `spline_{i,o}(x) = Σ_n coef[i, o, n] · B_n(x)` with `B_n` the n-th
//! cubic B-spline basis on the extended grid.

use burn::config::Config;
use burn::module::{Module, Param};
use burn::tensor::activation::silu;
use burn::tensor::{backend::Backend, Tensor};

use crate::spline::coef2curve;

/// pykan parity: the constructor mirrors `KANLayer.__init__`.
///
/// **Field order is load-bearing.** Burn's `Config` derive turns fields
/// *without* `#[config(default = …)]` into positional args of `::new(…)` in
/// declaration order. We want `KanLayerConfig::new(in_dim, out_dim, seed)`,
/// so those three come first; everything else has a default.
#[derive(Config, Debug)]
pub struct KanLayerConfig {
    /// Input dimension (`in_dim` in pykan).
    pub in_dim: usize,
    /// Output dimension (`out_dim` in pykan).
    pub out_dim: usize,
    /// Seed for the noise + scale_base sampling at init. REQUIRED — no default.
    pub seed: u64,

    /// Grid intervals (`num` in pykan). Extended grid has `num + 1 + 2k` knots.
    #[config(default = 5)]                pub num: usize,
    /// Spline order. pykan default 3 (cubic).
    #[config(default = 3)]                pub k: usize,

    /// pykan KANLayer default = 0.5. KanConfig propagates 0.3 (MultKAN default).
    #[config(default = 0.5)]              pub noise_scale: f64,
    #[config(default = 0.0)]              pub scale_base_mu: f64,
    #[config(default = 1.0)]              pub scale_base_sigma: f64,
    #[config(default = 1.0)]              pub scale_sp: f64,
    #[config(default = "[-1.0, 1.0]")]    pub grid_range: [f64; 2],
    #[config(default = true)]             pub sp_trainable: bool,
    #[config(default = true)]             pub sb_trainable: bool,
}

impl KanLayerConfig {
    /// Build a `KanLayer` from pre-existing parameter tensors (fixture /
    /// pretrained-loading path). All shapes are checked against
    /// `(in_dim, out_dim, num, k)` and panic on mismatch.
    pub fn init_from_parts<B: Backend>(
        &self,
        _device: &B::Device,
        grid: Tensor<B, 2>,
        coef: Tensor<B, 3>,
        scale_base: Tensor<B, 2>,
        scale_sp: Tensor<B, 2>,
        mask: Tensor<B, 2>,
    ) -> KanLayer<B> {
        let i = self.in_dim;
        let o = self.out_dim;
        let knots = self.num + 1 + 2 * self.k;
        let n_basis = self.num + self.k;

        assert_eq!(
            grid.dims(),
            [i, knots],
            "grid shape mismatch: expected [{i}, {knots}], got {:?}",
            grid.dims()
        );
        assert_eq!(
            coef.dims(),
            [i, o, n_basis],
            "coef shape mismatch: expected [{i}, {o}, {n_basis}], got {:?}",
            coef.dims()
        );
        assert_eq!(
            scale_base.dims(),
            [i, o],
            "scale_base shape mismatch: expected [{i}, {o}], got {:?}",
            scale_base.dims()
        );
        assert_eq!(
            scale_sp.dims(),
            [i, o],
            "scale_sp shape mismatch: expected [{i}, {o}], got {:?}",
            scale_sp.dims()
        );
        assert_eq!(
            mask.dims(),
            [i, o],
            "mask shape mismatch: expected [{i}, {o}], got {:?}",
            mask.dims()
        );

        KanLayer {
            grid:       Param::from_tensor(grid).set_require_grad(false),
            coef:       Param::from_tensor(coef),
            scale_base: Param::from_tensor(scale_base).set_require_grad(self.sb_trainable),
            scale_sp:   Param::from_tensor(scale_sp).set_require_grad(self.sp_trainable),
            mask:       Param::from_tensor(mask).set_require_grad(false),
            k: self.k,
        }
    }
}

/// A KAN edge-activation layer. Equivalent to pykan's `KANLayer` under
/// `sparse_init=False`, `base_fun=SiLU` (descoped: `update_grid_from_samples`,
/// `prune`, symbolic branch, caching).
#[derive(Module, Debug)]
pub struct KanLayer<B: Backend> {
    pub grid:       Param<Tensor<B, 2>>,    // [I, K] — frozen
    pub coef:       Param<Tensor<B, 3>>,    // [I, O, n_basis] — trainable
    pub scale_base: Param<Tensor<B, 2>>,    // [I, O] — trainable iff sb_trainable
    pub scale_sp:   Param<Tensor<B, 2>>,    // [I, O] — trainable iff sp_trainable
    pub mask:       Param<Tensor<B, 2>>,    // [I, O] — frozen (ones in v1)
    pub k: usize,
}

impl<B: Backend> KanLayer<B> {
    /// Forward pass.
    ///
    /// Input `x` shape `[B, in_dim]`. Output shape `[B, out_dim]`.
    ///
    /// `y[b, o] = Σ_i mask[i, o] · (scale_base[i, o] · SiLU(x[b, i])
    ///                            + scale_sp[i, o]   · spline_{i,o}(x[b, i]))`
    ///
    /// Equivalent to pykan's `KANLayer.forward(x)[0]` — we drop the
    /// `(preacts, postacts, postspline)` tuple returns.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let base = silu(x.clone());                                  // [B, I]
        let spline = coef2curve(
            x,
            self.grid.val(),
            self.coef.val(),
            self.k,
        );                                                            // [B, I, O]

        let sb = self.scale_base.val().unsqueeze_dim::<3>(0);         // [1, I, O]
        let sp = self.scale_sp.val().unsqueeze_dim::<3>(0);           // [1, I, O]
        let m  = self.mask.val().unsqueeze_dim::<3>(0);               // [1, I, O]
        let base3 = base.unsqueeze_dim::<3>(2);                       // [B, I, 1]

        let y = sb * base3 + sp * spline;                              // [B, I, O]
        let y = y * m;                                                 // [B, I, O]
        // Sum over I, drop the singleton dim → [B, O]
        y.sum_dim(1).squeeze_dim::<2>(1)
    }
}
