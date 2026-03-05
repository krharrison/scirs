//! Nonlinear dimensionality reduction methods.
//!
//! ## Modules
//!
//! - [`autoencoder_embed`]: Autoencoder-based dimensionality reduction with
//!   mini-batch SGD training.

pub mod autoencoder_embed;

pub use autoencoder_embed::{AEEmbedder, AELayer};
