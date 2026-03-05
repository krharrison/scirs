//! # SlabAllocator - Trait Implementations
//!
//! This module contains trait implementations for `SlabAllocator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::SlabAllocator;

impl Default for SlabAllocator {
    fn default() -> Self {
        Self::new()
    }
}
