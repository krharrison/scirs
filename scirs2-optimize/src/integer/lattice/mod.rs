//! Lattice reduction preprocessing for integer programs.
//!
//! This module provides LLL (Lenstra-Lenstra-Lovász) and BKZ (Block Korkine-Zolotarev)
//! lattice basis reduction algorithms for preprocessing mixed-integer programs.
//! Lattice reduction can significantly improve the performance of branch-and-bound
//! solvers by transforming the ILP constraint matrix to a more favorable basis.
//!
//! # References
//! - Lenstra, H.W., Lenstra, A.K., Lovász, L. (1982). "Factoring polynomials with rational
//!   number coefficients." Mathematische Annalen, 261(4), 515–534.
//! - Schnorr, C.P., Euchner, M. (1994). "Lattice basis reduction: Improved practical
//!   algorithms and solving subset sum problems." Mathematical programming, 66(1-3), 181–199.

pub mod gram_schmidt;
pub mod lll;
pub mod svp;
pub mod bkz;
pub mod mip_preprocess;

pub use lll::{LLLConfig, LLLReducer, LLLResult};
pub use bkz::{BKZConfig, BKZReducer, BKZResult};
pub use mip_preprocess::{
    LatticePreprocessor, LatticePreprocessorConfig, LatticePreprocessorResult, ReductionMethod,
};
