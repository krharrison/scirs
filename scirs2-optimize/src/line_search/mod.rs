//! Enhanced line search algorithms for second-order optimization methods
//!
//! This module provides production-quality line search implementations:
//! - Strong Wolfe conditions (zoom algorithm)
//! - Hager–Zhang (CG_DESCENT approximate Wolfe)
//! - Safeguarded Powell cubic interpolation
//! - Backtracking Armijo

pub mod enhanced;

pub use enhanced::{
    BacktrackingArmijo, HagerZhang, HagerZhangConfig, LineSearchResult, SafeguardedPowell,
    StrongWolfe, StrongWolfeConfig,
};
