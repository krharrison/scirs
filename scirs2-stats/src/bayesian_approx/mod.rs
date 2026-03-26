//! Bayesian Neural Network posterior approximations.
//!
//! This module provides **post-hoc** Bayesian approximations for neural
//! network weights — enabling uncertainty quantification without modifying
//! the base training procedure.
//!
//! Two families of methods are provided:
//!
//! ## Laplace Approximation
//!
//! Fits a Gaussian posterior `p(θ|D) ≈ N(θ*, H⁻¹)` centred at the MAP
//! estimate θ* using the curvature (Hessian) of the loss surface.
//!
//! Practical diagonal GGN (Fisher approximation):
//! ```text
//!   H_ii ≈ Σₙ (∂lossₙ/∂θᵢ)²   (squared gradients)
//!   σ²ᵢ = 1 / (H_ii + λ)
//! ```
//!
//! See [`laplace::fit_laplace`] for the high-level API.
//!
//! ## SWAG (Stochastic Weight Averaging Gaussian)
//!
//! Collects SGD weight snapshots and fits a Gaussian with diagonal +
//! low-rank covariance (Maddox et al. 2019):
//! ```text
//!   Σ ≈ diag(σ²_diag) / 2 + D̂ D̂ᵀ / (2(C−1))
//! ```
//!
//! See [`swag::SwagCollector`] and [`swag::sample_weights`] for the API.
//!
//! # Example
//!
//! ```rust
//! use scirs2_stats::bayesian_approx::{
//!     laplace::fit_laplace, swag::{SwagCollector, sample_weights},
//!     types::{LaplaceConfig, SwagConfig},
//! };
//!
//! // --- Laplace ---
//! let map_weights = vec![1.0f64, 0.5];
//! let loss_fn = |w: &[f64]| -> Vec<f64> {
//!     vec![(w[0] - 1.0).powi(2), (w[1] - 0.5).powi(2)]
//! };
//! let config = LaplaceConfig::default();
//! let lap = fit_laplace(&map_weights, &loss_fn, &config).expect("laplace");
//! println!("Laplace uncertainty: {:?}", lap.uncertainty);
//!
//! // --- SWAG ---
//! let mut collector = SwagCollector::new(2, 5);
//! for t in 0..20usize {
//!     collector.update(&[1.0 + t as f64 * 0.01, 0.5 - t as f64 * 0.005]);
//! }
//! let state = collector.finalize().expect("finalize");
//! let samples = sample_weights(&state, 10, 42).expect("sample");
//! println!("SWAG samples: {}", samples.len());
//! ```

pub mod laplace;
pub mod swag;
pub mod types;

pub use laplace::{
    diagonal_ggn, fd_per_sample_gradients, fit_laplace, posterior_variance_from_ggn,
    predict_mean_linear, predict_variance_linear,
};
pub use swag::{fit_swag, predict_ensemble, sample_weights, SwagCollector, SwagState};
pub use types::{BnnApproxResult, HessianMethod, LaplaceConfig, SwagConfig};
