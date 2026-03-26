//! Time-Dependent Hartree-Fock (TDHF) module.
//!
//! This module provides:
//! - Two-electron repulsion integrals (ERI) with Obara-Saika / direct s-type approach
//! - Ground-state restricted Hartree-Fock SCF solver with optional DIIS acceleration
//! - Real-time TDHF propagation (Magnus2 / Euler) with external field support
//! - Linear-response Casida equations (Tamm-Dancoff approximation)
//!
//! # Physical units
//! All quantities are in atomic units (Hartree, bohr, ℏ = 1, m_e = 1).
//!
//! # Quick start
//! ```rust,no_run
//! use scirs2_integrate::specialized::quantum::tdhf::{
//!     HartreeFockSCF, ScfConfig, RealTimeTDHF, TdhfConfig,
//! };
//! use scirs2_integrate::specialized::quantum::gaussian_integrals::{
//!     normalized_s_gto, sto3g_basis,
//! };
//! ```

pub mod eri;
pub mod propagation;
pub mod response;
pub mod scf;

pub use eri::{build_eri_tensor, compute_eri_ssss, get_eri, schwarz_screening};
pub use propagation::{Propagator, RealTimeTDHF, TdhfConfig, TdhfResult, TdhfState};
pub use response::{CasidaConfig, CasidaResult, CasidaSolver};
pub use scf::{HartreeFockSCF, ScfConfig, ScfConverger, ScfResult};
