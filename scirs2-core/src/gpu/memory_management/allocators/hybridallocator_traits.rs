//! # HybridAllocator - Trait Implementations
//!
//! This module contains trait implementations for `HybridAllocator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::HybridAllocator;

impl Default for HybridAllocator {
    fn default() -> Self {
        Self::new().expect("Failed to create default hybrid allocator")
    }
}
