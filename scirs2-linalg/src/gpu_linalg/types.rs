//! Result types for GPU-accelerated linear algebra operations.
//!
//! These types mirror the standard decomposition result formats but are returned
//! by the GPU-accelerated implementations.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;

/// Result of an LU factorization: P * A = L * U
///
/// The convention is: `P * A = L * U`, equivalently `A = P^T * L * U`.
///
/// Contains the permutation matrix P, lower-triangular matrix L (with unit diagonal),
/// and upper-triangular matrix U.
#[derive(Debug, Clone)]
pub struct LuFactors<F: Float> {
    /// Permutation matrix P
    pub p: Array2<F>,
    /// Lower-triangular matrix L (unit diagonal)
    pub l: Array2<F>,
    /// Upper-triangular matrix U
    pub u: Array2<F>,
}

/// Result of a QR factorization: A = QR
///
/// Contains the orthogonal matrix Q and upper-triangular matrix R.
#[derive(Debug, Clone)]
pub struct QrFactors<F: Float> {
    /// Orthogonal matrix Q
    pub q: Array2<F>,
    /// Upper-triangular matrix R
    pub r: Array2<F>,
}

/// Result of a Cholesky factorization: A = LL^T
///
/// Contains the lower-triangular Cholesky factor L.
#[derive(Debug, Clone)]
pub struct CholeskyFactors<F: Float> {
    /// Lower-triangular Cholesky factor L
    pub l: Array2<F>,
}

/// Result of a singular value decomposition: A = USV^T
///
/// Contains the left singular vectors U, singular values S (sorted descending),
/// and right singular vectors V^T.
#[derive(Debug, Clone)]
pub struct SvdFactors<F: Float> {
    /// Left singular vectors U
    pub u: Array2<F>,
    /// Singular values (sorted descending)
    pub s: Array1<F>,
    /// Right singular vectors V^T (transposed)
    pub vt: Array2<F>,
}

/// Result of a batched matrix multiplication.
///
/// Contains the resulting matrices from multiplying corresponding pairs
/// in the input batches.
#[derive(Debug, Clone)]
pub struct BatchMatmulResult<F: Float> {
    /// Result matrices, one for each pair in the batch
    pub results: Vec<Array2<F>>,
}

/// Execution backend information for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionBackend {
    /// Operation was executed on GPU
    Gpu,
    /// Operation fell back to CPU
    CpuFallback,
}

impl std::fmt::Display for ExecutionBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionBackend::Gpu => write!(f, "GPU"),
            ExecutionBackend::CpuFallback => write!(f, "CPU (fallback)"),
        }
    }
}
