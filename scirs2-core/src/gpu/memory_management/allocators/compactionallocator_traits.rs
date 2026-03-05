//! # CompactionAllocator - Trait Implementations
//!
//! This module contains trait implementations for `CompactionAllocator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::CompactionAllocator;

impl Default for CompactionAllocator {
    fn default() -> Self {
        Self::new(1024 * 1024 * 1024, 0.3)
    }
}
