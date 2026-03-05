//! Advanced copula models for dependence structure modeling.
//!
//! This module provides a comprehensive set of bivariate and multivariate copula
//! families suitable for statistical modeling of complex dependence structures.
//!
//! # Module Structure
//! - [`archimedean`]: Archimedean copulas (Clayton, Gumbel, Frank, Joe, BB1)
//! - [`elliptical`]: Elliptical copulas (Gaussian, Student-t)
//! - [`vine`]: Vine copulas (D-vine, C-vine) for multivariate modeling
//!
//! # Quick Start
//! ```no_run
//! use scirs2_stats::copula::archimedean::{ClaytonCopula, SimpleLcg};
//! use scirs2_stats::copula::elliptical::GaussianCopula;
//! use scirs2_stats::copula::vine::{DVine, PairCopula};
//!
//! // Fit a Clayton copula
//! let u = vec![0.2, 0.5, 0.7, 0.9];
//! let v = vec![0.3, 0.4, 0.8, 0.95];
//! let clayton = ClaytonCopula::fit(&u, &v).unwrap();
//!
//! // Sample from a 3D D-vine Gaussian
//! let dvine = DVine::gaussian(3, 0.6).unwrap();
//! let mut rng = SimpleLcg::new(42);
//! let samples = dvine.sample(100, &mut rng);
//! ```
//!
//! # Mathematical Background
//! Copulas separate the marginal distributions from the dependence structure.
//! By Sklar's theorem, any joint CDF F(x,y) with marginals F_X, F_Y can be
//! written as F(x,y) = C(F_X(x), F_Y(y)) for a unique copula C when the
//! marginals are continuous.
//!
//! # References
//! - Nelsen, R.B. (2006). *An Introduction to Copulas* (2nd ed.). Springer.
//! - Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.
//! - Aas, K. et al. (2009). Pair-copula constructions of multiple dependence structures.

pub mod archimedean;
pub mod elliptical;
pub mod vine;

pub use archimedean::{
    BB1Copula, ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula,
    LcgRng, SimpleLcg, compute_kendall_tau,
};
pub use elliptical::{GaussianCopula, StudentCopula, norm_cdf, norm_ppf, student_t_cdf};
pub use vine::{CVine, DVine, PairCopula, VineTree};
