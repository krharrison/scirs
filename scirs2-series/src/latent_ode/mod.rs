//! Latent ODE / ODE-RNN for irregularly-sampled time series.
//!
//! This module provides a self-contained implementation of the **Latent Neural
//! ODE** architecture (Rubanova et al., NeurIPS 2019) in pure Rust.
//!
//! ## Architecture
//!
//! ```text
//!   observations  →  RecognitionRnn (reverse-time GRU)
//!                              ↓
//!                     q(z₀) = N(μ, σ²)   (reparameterisation)
//!                              ↓ sample
//!                             z₀  →  OdeFunc  →  z(t)  →  Decoder  →  x̂(t)
//! ```
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use scirs2_series::latent_ode::{model::LatentOde, types::LatentOdeConfig};
//!
//! let mut config = LatentOdeConfig::default();
//! config.latent_dim = 8;
//! config.hidden_dim = 16;
//! config.n_layers = 1;
//! let mut model = LatentOde::new(config, 1).expect("model");
//! let obs = vec![(0.0_f64, vec![1.0_f64]), (0.5, vec![0.7]), (1.0, vec![0.4])];
//! let result = model.fit(&obs, 10).expect("fit");
//! let preds = model.predict(&[0.25, 0.75]).expect("predict");
//! ```

pub mod model;
pub mod ode_func;
pub mod recognition_rnn;
pub mod types;

pub use model::LatentOde;
pub use types::{LatentOdeConfig, LatentOdeResult};
