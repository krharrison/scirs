//! Online Meta-Learning for rapid adaptation to new time series tasks.
//!
//! This module implements gradient-based meta-learning algorithms that learn an
//! initialisation which can be quickly adapted to new tasks with a few gradient steps.
//!
//! ## Algorithms
//!
//! | Module | Algorithm | Reference |
//! |--------|-----------|-----------|
//! | [`maml`] | FOMAML (first-order MAML) | Finn et al. (2017) |
//! | [`reptile`] | Reptile | Nichol et al. (2018) |
//! | [`online_maml`] | Online MAML (streaming) | — |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use scirs2_series::online_meta::{MamlOptimizer, MetaLearnerConfig, Task};
//!
//! let config = MetaLearnerConfig {
//!     feature_dim: 2,
//!     inner_lr: 0.05,
//!     outer_lr: 0.01,
//!     n_inner_steps: 5,
//!     ..Default::default()
//! };
//! let mut optimizer = MamlOptimizer::new(config);
//!
//! let task = Task {
//!     support_x: vec![vec![0.0, 1.0], vec![1.0, 0.0]],
//!     support_y: vec![1.0, 2.0],
//!     query_x: vec![vec![0.5, 0.5]],
//!     query_y: vec![1.5],
//! };
//! let loss = optimizer.meta_train_step(&[task]);
//! assert!(loss.is_finite());
//! ```

pub mod maml;
pub mod online_maml;
pub mod reptile;
pub mod types;

pub use maml::{MamlOptimizer, MetaLinearModel};
pub use online_maml::OnlineMetaLearner;
pub use reptile::ReptileOptimizer;
pub use types::{
    linear_predict, mse, predict_all, AdaptationMetrics, MetaLearnerConfig, MetaLearnerResult, Task,
};
