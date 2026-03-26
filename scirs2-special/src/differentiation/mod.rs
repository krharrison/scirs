//! Derivative rules and numerical differentiation for special functions.
//!
//! ## Modules
//!
//! - [`symbolic_rules`] – Closed-form derivatives (Gamma, Bessel, ₁F₁, K) and
//!   a generic high-accuracy finite-difference differentiator.

pub mod symbolic_rules;

pub use symbolic_rules::{DerivativeRule, SpecialFunctionDerivatives};
