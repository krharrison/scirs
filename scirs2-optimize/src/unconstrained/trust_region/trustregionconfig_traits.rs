//! # TrustRegionConfig - Trait Implementations
//!
//! This module contains trait implementations for `TrustRegionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::TrustRegionConfig;

impl Default for TrustRegionConfig {
    fn default() -> Self {
        Self {
            initial_radius: 1.0,
            max_radius: 100.0,
            eta1: 0.25,
            eta2: 0.75,
            gamma1: 0.25,
            gamma2: 2.0,
            max_iter: 1000,
            tolerance: 1e-6,
            ftol: 1e-12,
            eps: 1.4901161193847656e-8,
            min_radius: 1e-14,
        }
    }
}
