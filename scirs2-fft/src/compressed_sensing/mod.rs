//! Compressed sensing recovery via FFT-based measurements.
//!
//! Provides OMP, ISTA, and FISTA solvers for recovering sparse signals
//! from partial Discrete Fourier Transform (DFT) observations.

mod ista;
mod omp;
pub mod types;

pub use ista::{soft_threshold, IstaSolver};
pub use omp::OmpSolver;
pub use types::{CsConfig, CsResult, Measurement, RecoveryMethod};
