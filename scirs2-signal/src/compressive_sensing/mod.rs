//! Compressive sensing algorithms.
//!
//! Provides:
//! - [`measurements`]: Measurement matrix generators (Gaussian, Bernoulli, partial DCT, sparse JL) and RIP checking
//! - [`recovery`]: Signal recovery via Basis Pursuit, BPDN, and Dantzig Selector
//! - [`cs_algorithms`]: CoSaMP, IHT, and threshold utilities

pub mod cs_algorithms;
pub mod measurements;
pub mod recovery;

// Convenience re-exports
pub use cs_algorithms::{hard_threshold, soft_threshold, CoSaMP, IHT};
pub use measurements::{is_rip, BernoulliMatrix, GaussianMatrix, PartialDCT, Rng, SparseJL};
pub use recovery::{basis_pursuit, basis_pursuit_denoising, dantzig_selector};
