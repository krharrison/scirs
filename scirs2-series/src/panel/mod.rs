//! Panel data econometric models
//!
//! Panel (longitudinal) data combines cross-sectional and time-series dimensions.
//! This module implements the standard fixed-effects (within) estimator, random-effects
//! GLS estimator, the Hausman specification test, and difference-in-differences (DiD).
//!
//! # Structure
//!
//! Data are organised as `N` individuals observed for `T_i` time periods each.
//! In balanced panels `T_i = T` for all `i`.
//!
//! ```text
//! y_{it} = α_i + X_{it}' β + ε_{it},    i=1..N, t=1..T_i
//! ```
//!
//! # Models
//!
//! - [`FixedEffectsModel`] — Within estimator; eliminates unobserved heterogeneity `α_i`
//! - [`RandomEffectsModel`] — GLS estimator; treats `α_i` as random
//! - [`HausmanTest`] — Tests H₀: RE is consistent (RE preferred for efficiency)
//! - [`DifferenceInDifferences`] — Causal estimation from quasi-experimental design
//!
//! # References
//! - Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*.
//!   MIT Press.

pub mod fixed_effects;
pub mod random_effects;
pub mod did;
pub mod hausman;

pub use did::DifferenceInDifferences;
pub use fixed_effects::FixedEffectsModel;
pub use hausman::HausmanTest;
pub use random_effects::RandomEffectsModel;
