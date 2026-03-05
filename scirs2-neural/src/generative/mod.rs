//! Generative Models
//!
//! This module provides implementations of modern deep generative models:
//!
//! - [`ddpm`] — Denoising Diffusion Probabilistic Models (Ho et al. 2020) with
//!   a minimal SimpleUNet backbone, forward/reverse processes, and training loss.
//! - [`score_matching`] — Score-based density estimation via Explicit, Denoising,
//!   and Sliced Score Matching objectives.
//! - [`flow_matching`] — Conditional Flow Matching (Lipman et al. 2022) with Euler
//!   ODE integration for generation.
//! - [`vq_vae`] — Vector Quantized VAE (van den Oord et al. 2017) with EMA codebook
//!   updates and straight-through gradient estimator.
//!
//! ## Design Notes
//!
//! All implementations use hand-written `Vec<f32>` / `Vec<f64>` arithmetic
//! for maximum portability. No external random crates are used; internal LCG
//! and Box-Muller primitives supply randomness.
//!
//! # References
//! - Ho, J., Jain, A. & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models.
//!   <https://arxiv.org/abs/2006.11239>
//! - Song, Y. & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the
//!   Data Distribution. <https://arxiv.org/abs/1907.05600>
//! - Lipman, Y. et al. (2022). Flow Matching for Generative Modeling.
//!   <https://arxiv.org/abs/2210.02747>
//! - van den Oord, A. et al. (2017). Neural Discrete Representation Learning (VQ-VAE).
//!   <https://arxiv.org/abs/1711.00937>

pub mod ddpm;
pub mod flow_matching;
pub mod score_matching;
pub mod vq_vae;
