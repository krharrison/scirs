//! Matrix nearness problems and perturbation theory.
//!
//! This module collects algorithms for finding the closest matrix of a given
//! structure and for quantifying how matrix properties (eigenvalues, invariant
//! subspaces, linear-system solutions) change under perturbation.
//!
//! ## Sub-modules
//!
//! | Sub-module | Description |
//! |---|---|
//! | [`nearest`] | Nearest matrix problems (SPD, orthogonal, correlation, …) |
//! | [`perturbation`] | Classical perturbation bounds (Weyl, Davis-Kahan, Bauer-Fike, …) |
//! | [`pseudospectrum`] | ε-pseudospectrum, Kreiss constant, pseudospectral abscissa |
//!
//! ## Quick reference
//!
//! ### Nearest structured matrices
//!
//! ```rust,ignore
//! use scirs2_linalg::nearness::nearest::{
//!     nearest_positive_definite,
//!     nearest_orthogonal,
//!     nearest_symmetric,
//!     nearest_correlation,
//!     nearest_doubly_stochastic,
//!     nearest_low_rank,
//! };
//! ```
//!
//! ### Perturbation theory
//!
//! ```rust,ignore
//! use scirs2_linalg::nearness::perturbation::{
//!     weyl_bounds,
//!     davis_kahan_bound,
//!     bauer_fike_bound,
//!     relative_perturbation_bound,
//!     condition_number_sensitivity,
//! };
//! ```
//!
//! ### Pseudospectrum
//!
//! ```rust,ignore
//! use scirs2_linalg::nearness::pseudospectrum::{
//!     epsilon_pseudospectrum,
//!     kreiss_constant,
//!     pseudospectral_abscissa,
//!     transient_bound,
//! };
//! ```
//!
//! ## References
//!
//! - Higham, N. J. (1988). Linear Algebra Appl. 103: 103–118.
//! - Higham, N. J. (2002). IMA J. Numer. Anal. 22(3): 329–343.
//! - Davis, C.; Kahan, W. M. (1970). SIAM J. Numer. Anal. 7(1): 1–46.
//! - Weyl, H. (1912). Math. Ann. 71: 441–479.
//! - Trefethen, L. N.; Embree, M. (2005). *Spectra and Pseudospectra*.
//!   Princeton University Press.

pub mod nearest;
pub mod perturbation;
pub mod pseudospectrum;

// ---------------------------------------------------------------------------
// Convenient flat re-exports
// ---------------------------------------------------------------------------

pub use nearest::{
    nearest_correlation, nearest_doubly_stochastic, nearest_low_rank, nearest_orthogonal,
    nearest_positive_definite, nearest_symmetric, NearestCorrelationResult,
    NearestDoublyStochasticResult, NearestLowRankResult, NearestOrthogonalResult, NearestPdResult,
};

pub use perturbation::{
    bauer_fike_bound, condition_number_sensitivity, davis_kahan_bound,
    relative_perturbation_bound, weyl_bounds, BauerFikeResult, ConditionSensitivityResult,
    DavisKahanResult, RelativePerturbationResult, WeylBoundsResult,
};

pub use pseudospectrum::{
    epsilon_pseudospectrum, kreiss_constant, pseudospectral_abscissa, transient_bound,
    KreissResult, PseudospectrumGrid, PseudospectralAbscissaResult, TransientBoundResult,
};
