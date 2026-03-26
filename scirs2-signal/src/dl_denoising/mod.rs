//! Deep Learning Denoising module for signal processing.
//!
//! Provides neural-network-inspired denoising methods including:
//! - Denoising autoencoders with skip connections (U-Net style)
//! - Residual learning (DnCNN-inspired)
//! - Diffusion-based denoising (DDPM-lite)
//! - Audio-specific DDPM denoiser ([`AudioDiffusionDenoiser`])

pub mod audio_diffusion;
mod autoencoder;
mod diffusion;
mod residual;
mod types;

pub use audio_diffusion::{AudioDiffusionConfig, AudioDiffusionDenoiser};
pub use autoencoder::DenoisingAutoencoder;
pub use diffusion::DiffusionDenoiser;
pub use residual::{BatchNorm1D, ResidualDenoiser};
pub use types::{Conv1DParams, DLDenoiseConfig, DenoisingMethod, DenoisingResult, TrainingConfig};
