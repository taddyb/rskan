//! `Kan` — sequential stack of `KanLayer`s with identity affine wrappers
//! (the pure-KAN reduction of MultKAN; see paper §2 and design spec §6.D).

use burn::config::Config;
use burn::module::Module;
use burn::tensor::{backend::Backend, Tensor};

use crate::layer::{KanLayer, KanLayerConfig};

/// Multi-layer KAN configuration.
///
/// Field order is load-bearing: `widths` and `seed` (no defaults) come first
/// for the generated `::new(widths, seed)`.
#[derive(Config, Debug)]
pub struct KanConfig {
    /// Widths from input to output. `widths=[H, H]` → one `KanLayer(H→H)`;
    /// `widths=[H, H, H]` → two layers. ddrs's `num_hidden_layers=N`
    /// corresponds to `widths=vec![H; N+1]`.
    pub widths: Vec<usize>,
    /// REQUIRED. No default.
    pub seed: u64,

    /// pykan MultKAN default = 3 (KANLayer's `num` default is 5).
    #[config(default = 3)]               pub grid: usize,
    #[config(default = 3)]               pub k: usize,
    /// pykan MultKAN default = 0.3 (not 0.5 like raw KANLayer).
    #[config(default = 0.3)]             pub noise_scale: f64,
    #[config(default = 0.0)]             pub scale_base_mu: f64,
    #[config(default = 1.0)]             pub scale_base_sigma: f64,
    /// Non-pykan API surface: MultKAN hardcodes 1.0 internally. Default
    /// matches; field exists for ablations.
    #[config(default = 1.0)]             pub scale_sp: f64,
    #[config(default = "[-1.0, 1.0]")]   pub grid_range: [f64; 2],
    #[config(default = true)]            pub sp_trainable: bool,
    #[config(default = true)]            pub sb_trainable: bool,
}

impl KanConfig {
    /// Build a `Kan` with per-layer sub-seeds derived as `seed.wrapping_add(l)`.
    /// (ddrs's `KanHead::init` deliberately overrides this to match DDR's
    /// "same seed all inner layers" quirk; rskan's own `Kan` uses the cleaner
    /// per-layer derivation.)
    pub fn init<B: Backend>(&self, device: &B::Device) -> Kan<B> {
        assert!(
            self.widths.len() >= 2,
            "Kan needs at least 2 widths (got {})",
            self.widths.len()
        );
        assert!(self.k    >= 1, "k must be >= 1");
        assert!(self.grid >= 1, "grid must be >= 1");

        let layers: Vec<KanLayer<B>> = (0..self.widths.len() - 1)
            .map(|l| {
                KanLayerConfig::new(
                    self.widths[l],
                    self.widths[l + 1],
                    self.seed.wrapping_add(l as u64),
                )
                .with_num(self.grid)
                .with_k(self.k)
                .with_noise_scale(self.noise_scale)
                .with_scale_base_mu(self.scale_base_mu)
                .with_scale_base_sigma(self.scale_base_sigma)
                .with_scale_sp(self.scale_sp)
                .with_grid_range(self.grid_range)
                .with_sp_trainable(self.sp_trainable)
                .with_sb_trainable(self.sb_trainable)
                .init(device)
            })
            .collect();

        Kan { layers }
    }
}

/// A stack of `KanLayer`s applied sequentially. No subnode/node affine
/// wrappers — pure-KAN reduction (paper §2, spec §6.D).
#[derive(Module, Debug)]
pub struct Kan<B: Backend> {
    pub layers: Vec<KanLayer<B>>,
}

impl<B: Backend> Kan<B> {
    /// Forward pass: sequential application of `KanLayer::forward`.
    pub fn forward(&self, mut x: Tensor<B, 2>) -> Tensor<B, 2> {
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }
}
