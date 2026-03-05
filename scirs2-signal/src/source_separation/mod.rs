//! Source separation algorithms.
//!
//! Provides:
//! - [`bss`]: Blind Source Separation (JADE, InfoMax, SOBI, PCA whitening)
//! - [`nmf`]: Non-negative Matrix Factorization for audio/spectral signals
//! - [`sparse_coding`]: OMP, ISTA-Lasso, K-SVD dictionary learning

pub mod bss;
pub mod nmf;
pub mod sparse_coding;

// Convenience re-exports
pub use bss::{jade_ica, infomax_ica, pca_whitening, sobi};
pub use nmf::{als_nmf, multiplicative_updates_frobenius, multiplicative_updates_kl, NMF, NMFConfig};
pub use sparse_coding::{hard_threshold, soft_threshold, DictionaryLearning, Lasso, OMP};
