//! Types for low-rank factorization updates.
//!
//! This module defines the configuration, result, and classification types
//! used by the LU, QR, and sketched update algorithms.

use std::fmt;

/// Configuration for low-rank update operations.
///
/// Controls numerical tolerances and verification behavior.
///
/// # Example
///
/// ```
/// use scirs2_sparse::low_rank_update::LowRankUpdateConfig;
///
/// let config = LowRankUpdateConfig {
///     tolerance: 1e-10,
///     verify_result: true,
///     max_iterations: 200,
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct LowRankUpdateConfig {
    /// Numerical tolerance for zero-checks and stability decisions.
    ///
    /// Elements below this threshold are treated as numerically zero.
    pub tolerance: f64,

    /// Whether to verify the result after factorization update.
    ///
    /// When `true`, the algorithm performs additional checks to ensure
    /// the updated factorization is correct (at extra computational cost).
    pub verify_result: bool,

    /// Maximum number of iterations for iterative refinement steps.
    pub max_iterations: usize,
}

impl Default for LowRankUpdateConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            verify_result: false,
            max_iterations: 100,
        }
    }
}

/// Result of an LU factorization update.
///
/// Contains the updated L, U factors and permutation vector along with
/// diagnostic information.
#[derive(Debug, Clone)]
pub struct LUUpdateResult {
    /// The updated lower-triangular factor L'.
    pub l: Vec<Vec<f64>>,

    /// The updated upper-triangular factor U'.
    pub u: Vec<Vec<f64>>,

    /// The permutation vector p (row i of PA corresponds to row `p[i]` of A).
    pub p: Vec<usize>,

    /// Whether the update completed successfully.
    pub success: bool,

    /// Rough estimate of the condition number of the updated factorization.
    pub condition_estimate: f64,
}

impl fmt::Display for LUUpdateResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LUUpdateResult {{ n={}, success={}, cond_est={:.6e} }}",
            self.l.len(),
            self.success,
            self.condition_estimate,
        )
    }
}

/// Result of a QR factorization update.
///
/// Contains the updated Q and R factors along with diagnostic information.
#[derive(Debug, Clone)]
pub struct QRUpdateResult {
    /// The updated orthogonal factor Q'.
    pub q: Vec<Vec<f64>>,

    /// The updated upper-triangular factor R'.
    pub r: Vec<Vec<f64>>,

    /// Whether the update completed successfully.
    pub success: bool,
}

impl fmt::Display for QRUpdateResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QRUpdateResult {{ m={}, n={}, success={} }}",
            self.q.len(),
            self.r.first().map_or(0, |row| row.len()),
            self.success,
        )
    }
}

/// Classification of factorization types for low-rank updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FactorizationType {
    /// LU factorization with partial pivoting (PA = LU).
    LU,
    /// QR factorization (A = QR).
    QR,
    /// Cholesky factorization (A = LL^T).
    Cholesky,
}

impl fmt::Display for FactorizationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FactorizationType::LU => write!(f, "LU"),
            FactorizationType::QR => write!(f, "QR"),
            FactorizationType::Cholesky => write!(f, "Cholesky"),
            #[allow(unreachable_patterns)]
            _ => write!(f, "Unknown"),
        }
    }
}

/// Configuration for sketched (randomized) low-rank updates.
///
/// Controls the sketch dimension, random seed, and error tolerance
/// for approximate factorization updates.
///
/// # Example
///
/// ```
/// use scirs2_sparse::low_rank_update::SketchConfig;
///
/// let config = SketchConfig {
///     sketch_dimension: 10,
///     seed: 12345,
///     error_tolerance: 1e-8,
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SketchConfig {
    /// Dimension of the random sketch (0 = automatic selection).
    pub sketch_dimension: usize,

    /// Seed for the random number generator.
    pub seed: u64,

    /// Acceptable error tolerance for the approximation.
    pub error_tolerance: f64,
}

impl Default for SketchConfig {
    fn default() -> Self {
        Self {
            sketch_dimension: 0,
            seed: 42,
            error_tolerance: 1e-6,
        }
    }
}

/// Estimate the condition number of a square matrix from its diagonal.
///
/// Returns `max(|diag|) / min(|diag|)`. If any diagonal element is zero
/// (or below `tol`), returns `f64::INFINITY`.
pub(crate) fn estimate_condition(mat: &[Vec<f64>], tol: f64) -> f64 {
    let n = mat.len();
    if n == 0 {
        return 1.0;
    }

    let mut min_diag = f64::INFINITY;
    let mut max_diag = 0.0_f64;

    for i in 0..n {
        if i >= mat[i].len() {
            return f64::INFINITY;
        }
        let d = mat[i][i].abs();
        if d < tol {
            return f64::INFINITY;
        }
        if d < min_diag {
            min_diag = d;
        }
        if d > max_diag {
            max_diag = d;
        }
    }

    if min_diag <= 0.0 {
        f64::INFINITY
    } else {
        max_diag / min_diag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = LowRankUpdateConfig::default();
        assert!((cfg.tolerance - 1e-12).abs() < 1e-15);
        assert!(!cfg.verify_result);
        assert_eq!(cfg.max_iterations, 100);
    }

    #[test]
    fn test_sketch_config_defaults() {
        let cfg = SketchConfig::default();
        assert_eq!(cfg.sketch_dimension, 0);
        assert_eq!(cfg.seed, 42);
        assert!((cfg.error_tolerance - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_factorization_type_display() {
        assert_eq!(format!("{}", FactorizationType::LU), "LU");
        assert_eq!(format!("{}", FactorizationType::QR), "QR");
        assert_eq!(format!("{}", FactorizationType::Cholesky), "Cholesky");
    }

    #[test]
    fn test_lu_result_display() {
        let res = LUUpdateResult {
            l: vec![vec![1.0]],
            u: vec![vec![2.0]],
            p: vec![0],
            success: true,
            condition_estimate: 1.0,
        };
        let s = format!("{}", res);
        assert!(s.contains("success=true"));
    }

    #[test]
    fn test_qr_result_display() {
        let res = QRUpdateResult {
            q: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            r: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            success: true,
        };
        let s = format!("{}", res);
        assert!(s.contains("success=true"));
    }

    #[test]
    fn test_estimate_condition_identity() {
        let m = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let cond = estimate_condition(&m, 1e-14);
        assert!((cond - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_estimate_condition_singular() {
        let m = vec![vec![1.0, 0.0], vec![0.5, 0.0]];
        let cond = estimate_condition(&m, 1e-14);
        assert!(cond.is_infinite());
    }
}
