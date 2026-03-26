//! Automatic Differentiation Variational Inference (ADVI).
//!
//! This module provides a self-contained ADVI implementation (Kucukelbir et al. 2017)
//! with:
//!
//! - **Mean-field** variational family: q(θ) = Π N(μᵢ, σᵢ²)
//! - **Bijective parameter transforms**: log (positive), logit (bounded), softmax (simplex)
//! - **ELBO optimization**: reparameterization trick + Adam optimizer
//! - **Posterior sampling** from the fitted Gaussian approximation
//!
//! # Quick start
//!
//! ```rust
//! use scirs2_stats::advi::{AdviConfig, AdviOptimizer, sample_posterior};
//!
//! // Log-joint: N(3 | θ, 1) · N(θ | 0, 1)  =>  posterior N(1.5, 0.5)
//! let log_joint = |theta: &[f64]| {
//!     let t = theta[0];
//!     -0.5 * (3.0 - t) * (3.0 - t) - 0.5 * t * t
//! };
//!
//! let config = AdviConfig { n_iter: 300, n_samples: 5, lr: 0.05, ..AdviConfig::default() };
//! let result = AdviOptimizer::new(config).fit(&log_joint, 1).expect("fit");
//! let samples = sample_posterior(&result, 100, 42).expect("sample");
//! println!("variational mean: {:.3}", result.mu[0]);
//! ```

pub mod advi;
pub mod transforms;
pub mod types;

pub use advi::{
    make_linear_regression_log_joint, mean_field_entropy, sample_posterior, AdviOptimizer,
};
pub use transforms::{
    log_jacobian_bounded, log_jacobian_positive, log_transform, logit_transform, softmax_transform,
    TransformSpec,
};
pub use types::{AdviConfig, AdviResult, ConstraintType, VariationalFamily};
