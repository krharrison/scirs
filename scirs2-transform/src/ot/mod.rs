//! Optimal transport methods.
//!
//! This module provides:
//! - [`unbalanced`]: Unbalanced optimal transport (UOT) where source and target
//!   distributions may have different total mass.

pub mod unbalanced;

pub use unbalanced::{
    unbalanced_sinkhorn, UnbalancedOtConfig, UnbalancedOtResult, UnbalancedRegularization,
};
