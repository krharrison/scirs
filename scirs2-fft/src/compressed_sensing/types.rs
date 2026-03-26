//! Types for compressed sensing via FFT-based measurements.

/// Configuration for compressed sensing recovery algorithms.
#[derive(Debug, Clone)]
pub struct CsConfig {
    /// Target sparsity (number of non-zero coefficients).
    pub sparsity: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
}

impl Default for CsConfig {
    fn default() -> Self {
        Self {
            sparsity: 5,
            max_iter: 100,
            tol: 1e-6,
        }
    }
}

/// Sparse frequency-domain measurements (partial DFT observations).
///
/// Represents y = A·x where A is a partial DFT matrix built from `indices`.
#[derive(Debug, Clone)]
pub struct Measurement {
    /// Which frequency bins were observed (row indices of the DFT matrix).
    pub indices: Vec<usize>,
    /// Observed values: real and imaginary parts interleaved as `[re0, im0, re1, im1, …]`.
    pub values: Vec<f64>,
}

/// Result of a compressed sensing recovery.
#[derive(Debug, Clone)]
pub struct CsResult {
    /// Recovered time-domain signal of length N.
    pub recovered: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final residual ‖y − A·x‖₂.
    pub residual: f64,
}

/// Recovery algorithm to use.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryMethod {
    /// Orthogonal Matching Pursuit.
    Omp,
    /// Iterative Shrinkage / Thresholding (ISTA / FISTA).
    Ista,
    /// Alternating Direction Method of Multipliers.
    Admm,
    /// Compressive Sampling Matching Pursuit.
    CoSaMP,
}
