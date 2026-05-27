//! rskan — Burn-based KAN (Kolmogorov-Arnold Network) layers.
//!
//! See `docs/superpowers/specs/2026-05-26-rskan-v1-kanlayer-design.md` for the
//! full design. Public surface is intentionally small: `KanLayer`/`KanLayerConfig`
//! and `Kan`/`KanConfig`.

#![forbid(unsafe_code)]

pub mod spline;
pub mod linalg;
pub mod init;
pub mod layer;
pub mod kan;

// Public re-exports — added in later tasks as types land:
pub use layer::{KanLayer, KanLayerConfig};
// pub use kan::{Kan, KanConfig};  // Uncommented in Task 10
