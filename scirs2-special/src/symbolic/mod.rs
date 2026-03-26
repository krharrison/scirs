//! Symbolic representation and manipulation of special functions.
//!
//! ## Modules
//!
//! - [`types`] – The [`Expr`] symbolic expression tree with evaluation,
//!   differentiation, and simplification.
//! - [`series`] – [`PowerSeries`] struct and Taylor-expansion factories for
//!   Gamma, Erf, Bessel J, and ₁F₁.
//! - [`asymptotic`] – [`AsymptoticExpansion`] struct and asymptotic series for
//!   large-argument approximations (Stirling, Bessel, erfc, ₁F₁).

pub mod asymptotic;
pub mod series;
pub mod types;

pub use asymptotic::{
    asymptotic_1f1, asymptotic_bessel_j, asymptotic_erf, asymptotic_gamma, eval_1f1_asymptotic,
    eval_bessel_j_asymptotic, eval_erfc_asymptotic, eval_stirling_gamma, eval_stirling_lngamma,
    AsymptoticExpansion, AsymptoticTerm,
};
pub use series::{taylor_1f1, taylor_bessel_j, taylor_erf, taylor_gamma, PowerSeries};
pub use types::Expr;
