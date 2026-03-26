//! Types for sparse Cholesky modifications (rank-1 updates/downdates).
//!
//! This module defines the configuration, result, and classification types
//! used by the Cholesky update and downdate algorithms.

use std::fmt;

/// Configuration for Cholesky update/downdate operations.
///
/// Controls numerical checks and tolerances used during the modification
/// of a Cholesky factorization.
///
/// # Example
///
/// ```
/// use scirs2_sparse::cholesky_update::CholUpdateConfig;
///
/// let config = CholUpdateConfig {
///     check_positive_definite: true,
///     tolerance: 1e-12,
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CholUpdateConfig {
    /// Whether to verify positive definiteness of the result.
    ///
    /// When `true`, the algorithm checks that all diagonal elements of the
    /// updated factor remain positive, returning an error if the result
    /// would be indefinite (particularly relevant for downdates).
    pub check_positive_definite: bool,

    /// Numerical tolerance for zero-checks and stability decisions.
    ///
    /// Diagonal elements below this threshold are considered numerically
    /// zero, triggering a positive-definiteness failure when checking is
    /// enabled.
    pub tolerance: f64,
}

impl Default for CholUpdateConfig {
    fn default() -> Self {
        Self {
            check_positive_definite: true,
            tolerance: 1e-14,
        }
    }
}

/// Result of a Cholesky update or downdate operation.
///
/// Contains the modified Cholesky factor along with diagnostic information
/// about the success and numerical conditioning of the operation.
///
/// # Representation
///
/// The factor is stored as a dense lower-triangular matrix in row-major
/// order: `factor[i][j]` is the element at row `i`, column `j`, with
/// `factor[i][j] == 0` for `j > i`.
#[derive(Debug, Clone)]
pub struct CholUpdateResult {
    /// The modified lower-triangular Cholesky factor L'.
    ///
    /// Satisfies L' L'^T = A' where A' is the modified matrix.
    pub factor: Vec<Vec<f64>>,

    /// Whether the update completed successfully.
    ///
    /// For updates this is almost always `true`; for downdates it may be
    /// `false` if the result would be indefinite (only when
    /// `check_positive_definite` is enabled in the config).
    pub successful: bool,

    /// Rough estimate of the condition number of the modified factor.
    ///
    /// Computed as `max(diag) / min(diag)` of the factor, giving a cheap
    /// (but often useful) lower bound on the true condition number.
    pub condition_estimate: f64,
}

impl fmt::Display for CholUpdateResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CholUpdateResult {{ n={}, successful={}, cond_est={:.6e} }}",
            self.factor.len(),
            self.successful,
            self.condition_estimate,
        )
    }
}

/// Classification of Cholesky modification types.
///
/// Describes the kind of rank-structured perturbation applied to the
/// original matrix A = L L^T.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UpdateType {
    /// Rank-1 update: A' = A + α u u^T  (α > 0).
    RankOneUpdate,
    /// Rank-1 downdate: A' = A − α u u^T  (α > 0).
    RankOneDowndate,
    /// Low-rank update: A' = A + W D W^T where D is diagonal.
    LowRankUpdate,
}

impl fmt::Display for UpdateType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UpdateType::RankOneUpdate => write!(f, "Rank-1 Update"),
            UpdateType::RankOneDowndate => write!(f, "Rank-1 Downdate"),
            UpdateType::LowRankUpdate => write!(f, "Low-Rank Update"),
            #[allow(unreachable_patterns)]
            _ => write!(f, "Unknown Update Type"),
        }
    }
}

/// Estimate the condition number of a lower-triangular matrix from its diagonal.
///
/// Returns `max(|diag|) / min(|diag|)`.  If any diagonal element is zero
/// (or below `tol`), returns `f64::INFINITY`.
pub(crate) fn estimate_condition(l: &[Vec<f64>], tol: f64) -> f64 {
    let n = l.len();
    if n == 0 {
        return 1.0;
    }

    let mut min_diag = f64::INFINITY;
    let mut max_diag = 0.0_f64;

    for i in 0..n {
        let d = l[i][i].abs();
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
        let cfg = CholUpdateConfig::default();
        assert!(cfg.check_positive_definite);
        assert!(cfg.tolerance > 0.0);
        assert!(cfg.tolerance < 1e-10);
    }

    #[test]
    fn test_update_type_display() {
        assert_eq!(format!("{}", UpdateType::RankOneUpdate), "Rank-1 Update");
        assert_eq!(
            format!("{}", UpdateType::RankOneDowndate),
            "Rank-1 Downdate"
        );
        assert_eq!(format!("{}", UpdateType::LowRankUpdate), "Low-Rank Update");
    }

    #[test]
    fn test_estimate_condition_identity() {
        let l = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let cond = estimate_condition(&l, 1e-14);
        assert!((cond - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_estimate_condition_singular() {
        let l = vec![vec![1.0, 0.0], vec![0.5, 0.0]];
        let cond = estimate_condition(&l, 1e-14);
        assert!(cond.is_infinite());
    }

    #[test]
    fn test_result_display() {
        let res = CholUpdateResult {
            factor: vec![vec![1.0]],
            successful: true,
            condition_estimate: 1.0,
        };
        let s = format!("{}", res);
        assert!(s.contains("successful=true"));
    }
}
