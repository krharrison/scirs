//! Causal Inference Methods
//!
//! This module provides a comprehensive suite of causal inference estimators:
//!
//! ## Sub-modules
//!
//! | Module | Methods |
//! |--------|---------|
//! | [`instrumental_variables`] | 2SLS, LIML, Hausman test, weak-instrument diagnostics |
//! | [`difference_in_differences`] | DiD (TWFE), synthetic control, event study, staggered DiD |
//! | [`regression_discontinuity`] | Sharp RDD, fuzzy RDD, bandwidth selection, RD plots |
//! | [`propensity_score`] | Logistic PS model, IPW, nearest-neighbour / kernel matching |
//!
//! ## Quick start
//!
//! ```rust
//! use scirs2_stats::causal::instrumental_variables::{IVEstimator, WeakInstrumentTest};
//! use scirs2_stats::causal::propensity_score::{
//!     PropensityScoreModel, IPW, PSMatching, MatchingMethod,
//! };
//! ```
//!
//! ## References
//!
//! - Angrist, J.D. & Pischke, J.-S. (2009). Mostly Harmless Econometrics.
//! - Callaway, B. & Sant'Anna, P.H.C. (2021). Difference-in-Differences with
//!   Multiple Time Periods.
//! - Imbens, G.W. & Kalyanaraman, K. (2012). Optimal Bandwidth Choice for
//!   the Regression Discontinuity Estimator.
//! - Rosenbaum, P.R. & Rubin, D.B. (1983). The Central Role of the Propensity
//!   Score in Observational Studies for Causal Effects.

pub mod difference_in_differences;
pub mod instrumental_variables;
pub mod propensity_score;
pub mod regression_discontinuity;

// ---------------------------------------------------------------------------
// Re-exports — instrumental variables
// ---------------------------------------------------------------------------

pub use instrumental_variables::{
    HausmanResult, HausmanTest, IVEstimator, IVResult, WeakInstrumentResult, WeakInstrumentTest,
    LIML,
};

// ---------------------------------------------------------------------------
// Re-exports — difference-in-differences
// ---------------------------------------------------------------------------

pub use difference_in_differences::{
    AttGt, DiD, DiDResult, EventCoefficient, EventStudy, EventStudyResult, StaggeredDiD,
    StaggeredDiDResult, SyntheticControl,
};

// ---------------------------------------------------------------------------
// Re-exports — regression discontinuity
// ---------------------------------------------------------------------------

pub use regression_discontinuity::{
    BandwidthMethod, BandwidthSelector, FuzzyRDD, RDDPlot, RDDResult, RDD,
};

// ---------------------------------------------------------------------------
// Re-exports — propensity score
// ---------------------------------------------------------------------------

pub use propensity_score::{
    MatchingMethod, MatchingResult, OverlapCheck, OverlapResult, PSMatching, PSResult,
    PropensityScoreModel, TrimMethod, IPW,
};

/// Convenience function: fit a propensity score model and estimate ATE/ATT/ATC via IPW.
pub use propensity_score::ps_estimate;

// ---------------------------------------------------------------------------
// Structural Equation Models
// ---------------------------------------------------------------------------

pub mod sem;

pub use sem::{satisfies_backdoor, IdentificationResult, LinearEquation, SEMWithIntercepts, SEM};

// ---------------------------------------------------------------------------
// Linear SEM with ndarray interface
// ---------------------------------------------------------------------------

pub mod hedge;
pub mod id_algorithm;
pub mod linear_sem;
pub mod semi_markov_graph;
pub mod symbolic_prob;

pub use linear_sem::{LinearSEM, LinearSEMWithIntercepts};
