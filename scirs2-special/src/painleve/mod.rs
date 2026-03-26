//! Painleve Transcendents
//!
//! This module provides numerical solutions for the six Painleve equations (P-I
//! through P-VI), a family of second-order nonlinear ODEs whose solutions define
//! new transcendental functions that cannot be expressed in terms of previously
//! known functions.
//!
//! ## Background
//!
//! The Painleve equations were classified by Paul Painleve and Bertrand Gambier
//! (1900-1910) as the complete list of second-order ODEs of the form
//! `y'' = R(t, y, y')` (R rational in y and y') whose movable singularities are
//! at most poles. This *Painleve property* makes them the nonlinear analogues of
//! the classical special functions.
//!
//! ## Applications
//!
//! Painleve transcendents appear in:
//!
//! - **Random matrix theory**: Tracy-Widom distribution (P-II)
//! - **Integrable systems**: KdV, NLS, sine-Gordon equations
//! - **Statistical mechanics**: Ising model correlations (P-III, P-V)
//! - **Quantum gravity**: 2D quantum gravity models (P-I)
//! - **Nonlinear waves**: Self-similar solutions of dispersive PDEs
//!
//! ## Module Organisation
//!
//! - [`types`] -- Equation variants, solver configuration, solution containers
//! - [`equations`] -- Right-hand sides of P-I through P-VI
//! - [`solver`] -- Adaptive Dormand-Prince RK45 integrator with pole detection
//! - [`connection`] -- Distinguished asymptotic solutions (Hastings-McLeod,
//!   tritronquee, Ablowitz-Segur)
//!
//! ## References
//!
//! - DLMF Chapter 32: <https://dlmf.nist.gov/32>
//! - Clarkson, P.A. (2006), "Painleve Equations -- Nonlinear Special Functions",
//!   *Journal of Computational and Applied Mathematics*, 153, 127-140.
//! - Fornberg, B. & Weideman, J.A.C. (2011), "A numerical methodology for the
//!   Painleve equations", *Journal of Computational Physics*, 230, 5957-5973.

pub mod connection;
pub mod equations;
pub mod solver;
pub mod types;

// Re-export key types and functions
pub use connection::{
    ablowitz_segur, hastings_mcleod, painleve_i_tritronquee, painleve_iv_special,
};
pub use equations::{painleve_rhs, painleve_system};
pub use solver::solve_painleve;
pub use types::{PainleveConfig, PainleveEquation, PainleveSolution};
