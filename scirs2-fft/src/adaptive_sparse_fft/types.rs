//! Types for the Adaptive Sparse FFT algorithm.
//!
//! This module defines configuration and result types used by the adaptive
//! parameter-free sparse FFT implementation.

use scirs2_core::numeric::Complex64;

/// Configuration for the Adaptive Sparse FFT algorithm.
///
/// The algorithm automatically estimates signal sparsity without requiring the
/// user to specify the number of non-zero frequency components in advance.
///
/// # Examples
///
/// ```
/// use scirs2_fft::adaptive_sparse_fft::AdaptiveSfftConfig;
///
/// let config = AdaptiveSfftConfig::default();
/// assert_eq!(config.max_sparsity, 64);
/// assert!((config.confidence - 0.95).abs() < 1e-10);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct AdaptiveSfftConfig {
    /// Maximum sparsity to consider during estimation (default: 64).
    pub max_sparsity: usize,
    /// Confidence level for sparsity estimation (default: 0.95).
    ///
    /// Higher values require more energy to be captured before declaring
    /// the sparsity estimate complete.
    pub confidence: f64,
    /// Width of the flat filter in frequency bins (default: 8).
    ///
    /// Larger values provide better frequency localization at the cost of
    /// spectral leakage suppression.
    pub filter_width: usize,
    /// Maximum number of refinement iterations (default: 5).
    ///
    /// Each iteration refines the sparsity estimate by analysing the residual
    /// energy from the previous round.
    pub max_iterations: usize,
    /// Energy threshold for considering a component significant (default: 1e-6).
    ///
    /// Components whose relative energy falls below this threshold are ignored.
    pub energy_threshold: f64,
}

impl Default for AdaptiveSfftConfig {
    fn default() -> Self {
        Self {
            max_sparsity: 64,
            confidence: 0.95,
            filter_width: 8,
            max_iterations: 5,
            energy_threshold: 1e-6,
        }
    }
}

/// Result of an Adaptive Sparse FFT computation.
///
/// Contains the identified sparse frequency components along with metadata
/// about the estimation process.
#[derive(Debug, Clone)]
pub struct AdaptiveSfftResult {
    /// Frequency bin indices of the identified non-zero components.
    pub frequencies: Vec<usize>,
    /// Complex spectral coefficients at the identified frequency bins.
    pub coefficients: Vec<Complex64>,
    /// Estimated number of non-zero components (sparsity level `k`).
    pub estimated_sparsity: usize,
    /// Number of refinement iterations actually performed.
    pub iterations: usize,
    /// Total signal energy (for normalisation).
    pub total_energy: f64,
    /// Fraction of energy captured by the recovered components.
    pub captured_energy_fraction: f64,
}

impl AdaptiveSfftResult {
    /// Create a new empty result.
    pub fn empty() -> Self {
        Self {
            frequencies: Vec::new(),
            coefficients: Vec::new(),
            estimated_sparsity: 0,
            iterations: 0,
            total_energy: 0.0,
            captured_energy_fraction: 0.0,
        }
    }
}
