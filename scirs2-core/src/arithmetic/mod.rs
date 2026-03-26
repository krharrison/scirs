//! Extended precision arithmetic for scientific computing.
//!
//! This module provides double-double arithmetic (~31 decimal digits of precision)
//! using error-free transformations (EFTs) based on Dekker/Knuth algorithms.
//!
//! # Overview
//!
//! | Type | Description |
//! |------|-------------|
//! | [`DoubleDouble`] | Double-double floating-point number (hi + lo) |
//!
//! # Functions
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`dot_dd`] | Accumulated dot product in double-double precision |
//! | [`sum_dd`] | Compensated summation (Ogita-Rump-Oishi) |
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::arithmetic::{DoubleDouble, sum_dd};
//!
//! // Catastrophic cancellation handled correctly
//! let values = [1.0_f64, 1e100, 1.0, -1e100];
//! let result = sum_dd(&values);
//! assert!((result.to_f64() - 2.0).abs() < 1e-10);
//! ```

pub mod double_double;

pub use double_double::{dot_dd, sum_dd, DoubleDouble};
