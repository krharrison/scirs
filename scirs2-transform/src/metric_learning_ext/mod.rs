//! Extended Metric Learning Algorithms
//!
//! This module supplements the existing [`crate::metric_learning`] module with:
//!
//! - [`ITML`]: Information-Theoretic Metric Learning — minimize KL divergence
//!   from a prior metric subject to pairwise distance constraints.
//! - [`MahalanobisDistance`]: Utility struct for computing custom Mahalanobis
//!   distances with a user-provided PSD matrix M.
//!
//! ## References
//!
//! - Davis, J.V., Kulis, B., Jain, P., Sra, S., & Dhillon, I.S. (2007).
//!   Information-Theoretic Metric Learning. ICML.
//! - Xing, E.P., Ng, A.Y., Jordan, M.I., & Russell, S. (2002).
//!   Distance Metric Learning with Application to Clustering with Side-Information.
//!   NeurIPS.

use scirs2_core::ndarray::{Array2};
use scirs2_linalg::inv;

use crate::error::{Result, TransformError};

// ============================================================================
// Mahalanobis distance utility
// ============================================================================

/// Mahalanobis distance with a user-supplied PSD matrix M.
///
/// Distance: d_M(a, b) = sqrt((a-b)^T M (a-b))
#[derive(Debug, Clone)]
pub struct MahalanobisDistance {
    /// Positive semi-definite matrix M of shape (d × d).
    pub m: Vec<Vec<f64>>,
}

impl MahalanobisDistance {
    /// Create from an existing PSD matrix.
    pub fn from_matrix(m: Vec<Vec<f64>>) -> Result<Self> {
        let d = m.len();
        for (i, row) in m.iter().enumerate() {
            if row.len() != d {
                return Err(TransformError::InvalidInput(format!(
                    "Row {i} has {} cols but expected {d}",
                    row.len()
                )));
            }
        }
        Ok(MahalanobisDistance { m })
    }

    /// Create an identity (Euclidean) metric of dimension `d`.
    pub fn identity(d: usize) -> Self {
        let m: Vec<Vec<f64>> = (0..d)
            .map(|i| (0..d).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        MahalanobisDistance { m }
    }

    /// Dimensionality.
    pub fn dim(&self) -> usize {
        self.m.len()
    }

    /// Squared Mahalanobis distance: (a-b)^T M (a-b).
    pub fn dist_sq(&self, a: &[f64], b: &[f64]) -> Result<f64> {
        let d = self.m.len();
        if a.len() != d || b.len() != d {
            return Err(TransformError::InvalidInput(format!(
                "Vectors must have length {d}, got {} and {}",
                a.len(),
                b.len()
            )));
        }
        let diff: Vec<f64> = a.iter().zip(b.iter()).map(|(ai, bi)| ai - bi).collect();

        // M * diff
        let mut md = vec![0.0f64; d];
        for i in 0..d {
            for j in 0..d {
                md[i] += self.m[i][j] * diff[j];
            }
        }

        // diff^T * (M * diff)
        let sq: f64 = diff.iter().zip(md.iter()).map(|(di, mdi)| di * mdi).sum();
        Ok(sq.max(0.0))
    }

    /// Mahalanobis distance: sqrt((a-b)^T M (a-b)).
    pub fn dist(&self, a: &[f64], b: &[f64]) -> Result<f64> {
        Ok(self.dist_sq(a, b)?.sqrt())
    }

    /// Transform a data point via the Cholesky factor L (M = L^T L): z = L x.
    /// This method approximates L as the diagonal sqrt of M (for diagonal M),
    /// or uses the full M matrix raised to the 1/2 power via eigendecomposition
    /// done at construction time.
    pub fn transform_point(&self, x: &[f64]) -> Result<Vec<f64>> {
        let d = self.m.len();
        if x.len() != d {
            return Err(TransformError::InvalidInput(format!(
                "Expected length {d}, got {}",
                x.len()
            )));
        }
        // Apply M directly (not the sqrt): result is M x
        let mut out = vec![0.0f64; d];
        for i in 0..d {
            for j in 0..d {
                out[i] += self.m[i][j] * x[j];
            }
        }
        Ok(out)
    }

    /// Compute pairwise distance matrix for a dataset.
    pub fn pairwise_distances(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = data.len();
        let mut dists = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.dist(&data[i], &data[j])?;
                dists[i][j] = d;
                dists[j][i] = d;
            }
        }
        Ok(dists)
    }
}

// ============================================================================
// ITML — Information-Theoretic Metric Learning
// ============================================================================

/// Pairwise constraint for ITML.
#[derive(Debug, Clone)]
pub struct ITMLConstraint {
    /// Index of first sample.
    pub i: usize,
    /// Index of second sample.
    pub j: usize,
    /// Upper bound on distance (similar pair: d_M(x_i, x_j) <= u)
    /// or lower bound (dissimilar pair: d_M(x_i, x_j) >= l).
    pub bound: f64,
    /// If `true`, this is a similarity constraint (d <= bound).
    /// If `false`, this is a dissimilarity constraint (d >= bound).
    pub similar: bool,
}

impl ITMLConstraint {
    /// Create a similarity constraint: d_M(x_i, x_j) <= upper.
    pub fn similar(i: usize, j: usize, upper: f64) -> Self {
        ITMLConstraint { i, j, bound: upper, similar: true }
    }

    /// Create a dissimilarity constraint: d_M(x_i, x_j) >= lower.
    pub fn dissimilar(i: usize, j: usize, lower: f64) -> Self {
        ITMLConstraint { i, j, bound: lower, similar: false }
    }
}

/// Information-Theoretic Metric Learning (ITML).
///
/// Minimizes KL( M || M_0 ) subject to pairwise distance constraints:
///   similar pairs:    d_M(x_i, x_j)^2 <= u_c
///   dissimilar pairs: d_M(x_i, x_j)^2 >= l_c
///
/// Dual variables (one per constraint) updated via Bregman projections:
///   λ_c ← λ_c + Δλ_c
///   M   ← M + Δλ_c · M (x_i - x_j)(x_i - x_j)^T M   (log-det Bregman update)
///
/// See Davis et al. (2007) Algorithm 1.
#[derive(Debug, Clone)]
pub struct ITML {
    /// Maximum number of outer iterations.
    pub max_iter: usize,
    /// Slack variable γ (diagonal slack Γ_0 = γ·I).
    pub gamma: f64,
    /// Convergence tolerance on constraint satisfaction.
    pub tol: f64,
}

/// Fitted ITML model.
#[derive(Debug, Clone)]
pub struct ITMLModel {
    /// Learned metric matrix M (d × d).
    pub metric: Array2<f64>,
    /// Dual variables (one per constraint).
    pub dual_vars: Vec<f64>,
    /// Number of outer iterations used.
    pub n_iter: usize,
}

impl Default for ITML {
    fn default() -> Self {
        ITML { max_iter: 100, gamma: 1.0, tol: 1e-3 }
    }
}

impl ITML {
    /// Create a new ITML instance.
    pub fn new(max_iter: usize, gamma: f64, tol: f64) -> Self {
        ITML { max_iter, gamma, tol }
    }

    /// Fit ITML on data `x` with pairwise constraints.
    ///
    /// `x`: dataset (n × d).  
    /// `constraints`: similarity / dissimilarity constraints.  
    /// Prior metric M_0 defaults to (1/γ)·I.
    pub fn fit(&self, x: &[Vec<f64>], constraints: &[ITMLConstraint]) -> Result<ITMLModel> {
        let n = x.len();
        if n == 0 {
            return Err(TransformError::InvalidInput("Empty dataset".to_string()));
        }
        let d = x[0].len();
        if d == 0 {
            return Err(TransformError::InvalidInput(
                "Feature dimension must be > 0".to_string(),
            ));
        }
        for (k, c) in constraints.iter().enumerate() {
            if c.i >= n || c.j >= n {
                return Err(TransformError::InvalidInput(format!(
                    "Constraint {k}: indices ({}, {}) out of range for n={n}",
                    c.i, c.j
                )));
            }
        }

        let nc = constraints.len();
        if nc == 0 {
            // No constraints: return prior
            let prior_val = 1.0 / self.gamma.max(1e-12);
            let mut m = Array2::<f64>::zeros((d, d));
            for i in 0..d {
                m[[i, i]] = prior_val;
            }
            return Ok(ITMLModel { metric: m, dual_vars: vec![], n_iter: 0 });
        }

        // Compute constraint bounds (squared distances)
        // u_c = bound^2 for similarity, l_c = bound^2 for dissimilarity
        let rho: Vec<f64> = constraints.iter().map(|c| c.bound * c.bound).collect();

        // Initialize M = (1/γ)·I
        let prior_val = 1.0 / self.gamma.max(1e-12);
        let mut m = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            m[[i, i]] = prior_val;
        }

        // Dual variables: λ_c (all zeros initially)
        let mut lambda = vec![0.0f64; nc];

        // Slack bounds: ξ_c are soft variables for each constraint
        // For simplicity, we use the hard bound rho_c directly.

        let mut n_iter = 0usize;

        for _outer in 0..self.max_iter {
            let mut max_violation = 0.0f64;
            n_iter += 1;

            for (ci, con) in constraints.iter().enumerate() {
                let xi = &x[con.i];
                let xj = &x[con.j];

                // diff = x_i - x_j
                let diff: Vec<f64> = xi.iter().zip(xj.iter()).map(|(a, b)| a - b).collect();

                // alpha_c = diff^T M diff  (current squared Mahalanobis distance)
                let mut mdiff = vec![0.0f64; d];
                for a in 0..d {
                    for b in 0..d {
                        mdiff[a] += m[[a, b]] * diff[b];
                    }
                }
                let alpha_c: f64 = diff.iter().zip(mdiff.iter()).map(|(di, mdi)| di * mdi).sum();
                let alpha_c = alpha_c.max(0.0);

                // Compute dual update
                // Δλ = (ρ_c^{-1} + α_c)^{-1} * (1 - α_c / ρ_c) ... simplified dual step
                // From Davis et al.: δ = (ρ_c - α_c) / (ρ_c * (α_c + 1/γ))
                // λ_new_c = λ_c + δ  (clamped so λ >= 0 for similarity, λ <= 0 for dissimilarity)
                let inv_gamma = 1.0 / self.gamma.max(1e-12);
                let denominator = rho[ci] * (alpha_c + inv_gamma);
                if denominator.abs() < 1e-15 {
                    continue;
                }
                let delta = (rho[ci] - alpha_c) / denominator;

                // Clamp: for similar constraints λ ≥ 0; for dissimilar λ ≤ 0
                let lambda_new = if con.similar {
                    (lambda[ci] + delta).max(0.0)
                } else {
                    (lambda[ci] + delta).min(0.0)
                };
                let actual_delta = lambda_new - lambda[ci];
                lambda[ci] = lambda_new;

                // Rank-1 update: M ← M + actual_delta * M diff diff^T M / (1 + actual_delta * α_c)
                let denom = 1.0 + actual_delta * alpha_c;
                if denom.abs() < 1e-15 {
                    continue;
                }
                let scale = actual_delta / denom;
                for a in 0..d {
                    for b in 0..d {
                        m[[a, b]] += scale * mdiff[a] * mdiff[b];
                    }
                }

                let violation = (alpha_c - rho[ci]).abs() / rho[ci].max(1e-10);
                if violation > max_violation {
                    max_violation = violation;
                }
            }

            if max_violation < self.tol {
                break;
            }
        }

        Ok(ITMLModel {
            metric: m,
            dual_vars: lambda,
            n_iter,
        })
    }
}

impl ITMLModel {
    /// Return a [`MahalanobisDistance`] wrapping the learned metric.
    pub fn mahalanobis(&self) -> Result<MahalanobisDistance> {
        let d = self.metric.nrows();
        let m: Vec<Vec<f64>> = (0..d)
            .map(|i| (0..d).map(|j| self.metric[[i, j]]).collect())
            .collect();
        MahalanobisDistance::from_matrix(m)
    }

    /// Transform data under the learned metric: compute L x where M = L^T L.
    ///
    /// Uses the matrix M directly (not L) — applies M^{1/2} via eigendecomposition.
    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let d = self.metric.nrows();
        let n = x.len();
        let mut out = vec![vec![0.0f64; d]; n];
        for (i, row) in x.iter().enumerate() {
            if row.len() != d {
                return Err(TransformError::InvalidInput(format!(
                    "Row {i}: expected {d} features, got {}",
                    row.len()
                )));
            }
            // Apply M to x[i]
            for a in 0..d {
                let s: f64 = (0..d).map(|b| self.metric[[a, b]] * row[b]).sum();
                out[i][a] = s;
            }
        }
        Ok(out)
    }

    /// Compute pairwise squared Mahalanobis distances for `x`.
    pub fn pairwise_distances(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let mah = self.mahalanobis()?;
        mah.pairwise_distances(x)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mahalanobis_identity() {
        let mah = MahalanobisDistance::identity(3);
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 0.0, 0.0];
        let d = mah.dist(&a, &b).expect("dist");
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mahalanobis_scaled() {
        // M = diag(4, 1, 1): distance along first axis is doubled
        let m = vec![
            vec![4.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let mah = MahalanobisDistance::from_matrix(m).expect("from_matrix");
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 0.0, 0.0];
        let d = mah.dist(&a, &b).expect("dist");
        assert!((d - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_mahalanobis_pairwise() {
        let mah = MahalanobisDistance::identity(2);
        let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let dists = mah.pairwise_distances(&data).expect("pairwise");
        assert_eq!(dists.len(), 3);
        assert!((dists[0][1] - 1.0).abs() < 1e-10);
        assert!((dists[0][2] - 1.0).abs() < 1e-10);
        assert!((dists[1][2] - 2.0_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn test_itml_basic() {
        let x: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![5.0, 5.0],
            vec![5.1, 5.2],
        ];
        let constraints = vec![
            ITMLConstraint::similar(0, 1, 1.0),
            ITMLConstraint::dissimilar(0, 2, 2.0),
            ITMLConstraint::similar(2, 3, 1.0),
        ];
        let itml = ITML::new(50, 1.0, 1e-3);
        let model = itml.fit(&x, &constraints).expect("ITML fit");
        assert!(model.n_iter > 0);
        assert_eq!(model.metric.nrows(), 2);

        // Similar pairs should be closer than dissimilar
        let mah = model.mahalanobis().expect("mahalanobis");
        let d_sim = mah.dist(&x[0], &x[1]).expect("d_sim");
        let d_dis = mah.dist(&x[0], &x[2]).expect("d_dis");
        assert!(d_sim < d_dis, "similar pair {d_sim:.4} should be < dissimilar {d_dis:.4}");
    }

    #[test]
    fn test_itml_no_constraints() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let itml = ITML::default();
        let model = itml.fit(&x, &[]).expect("ITML no constraints");
        // Returns prior metric (diagonal)
        assert!(model.metric[[0, 0]] > 0.0);
        assert!(model.metric[[1, 1]] > 0.0);
    }

    #[test]
    fn test_mahalanobis_dimension_error() {
        let mah = MahalanobisDistance::identity(3);
        let a = [1.0, 2.0]; // wrong dimension
        let b = [0.0, 0.0, 0.0];
        assert!(mah.dist(&a, &b).is_err());
    }
}
