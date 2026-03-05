//! # ReparamGradConfig - Trait Implementations
//!
//! This module contains trait implementations for `ReparamGradConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::validation::*;

use super::types::ReparamGradConfig;

impl Default for ReparamGradConfig {
    fn default() -> Self {
        Self {
            n_samples: 1,
            rao_blackwell: true,
            control_variates: false,
            baseline_decay: 0.99,
        }
    }
}
