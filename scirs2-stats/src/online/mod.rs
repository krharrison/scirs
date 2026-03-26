//! Online learning algorithms for Bayesian statistics
//!
//! This module provides streaming/online algorithms that process data in mini-batches
//! without requiring the entire dataset in memory. Key methods include:
//!
//! - **Online Variational Bayes**: Stochastic Variational Inference (SVI) for conjugate
//!   exponential family models, following Hoffman et al. (2013). Uses natural gradient
//!   updates on variational parameters with a decaying learning rate schedule.
//!
//! - **Streaming Gaussian Processes**: Sparse variational GP (SVGP) that maintains a
//!   compact inducing point representation and performs Kalman-like rank-1 updates as
//!   new data arrives, enabling O(M²N) rather than O(N³) complexity.
//!
//! # References
//!
//! - Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013).
//!   Stochastic variational inference. *JMLR*, 14(1), 1303–1347.
//! - Hensman, J., Fusi, N., & Lawrence, N. D. (2013).
//!   Gaussian processes for big data. *UAI*.
//! - Bui, T. D., Nguyen, C. V., & Turner, R. E. (2017).
//!   Streaming sparse GP approximations. *NIPS*.

pub mod streaming_gp;
pub mod variational_bayes;

pub use streaming_gp::*;
pub use variational_bayes::*;
