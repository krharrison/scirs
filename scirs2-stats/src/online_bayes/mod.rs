//! Online Bayesian inference methods for scirs2-stats.
//!
//! This module provides:
//!
//! - **Online Variational Bayes** ([`OnlineVb`]): stochastic natural-gradient
//!   updates for conjugate exponential-family models, with an LDA-style topic
//!   model specialisation (Hoffman et al. 2010).
//! - **Streaming Gaussian Process** ([`StreamingGp`]): rank-1 Kalman-style
//!   updates with a sparse inducing-point budget (Csató & Opper 2002 style).

pub mod online_vb;
pub mod streaming_gp;

pub use online_vb::{OnlineVb, OnlineVbConfig};
pub use streaming_gp::{StreamingGp, StreamingGpConfig};
