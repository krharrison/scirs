//! Low-Rank Representation (LRR).
//!
//! Recovers the global low-rank structure of data lying on a union of subspaces
//! by solving:
//!
//! ```text
//!   min_Z  ||Z||_*  +  lambda * ||E||_{2,1}   s.t.  X = XZ + E
//! ```
//!
//! where ||·||_* is the nuclear norm and ||·||_{2,1} is the L_{2,1} norm
//! (sum of L2 norms of rows).
//!
//! The nuclear norm proximal operator reduces to singular value soft-thresholding
//! (SVT). This implementation uses an ADMM-based alternating minimisation with
//! a simplified SVT step on the Gram matrix X^T X.
//!
//! # Reference
//!
//! Liu, G., Lin, Z., Yan, S., Sun, J., Yu, Y., Ma, Y. (2013). *Robust Recovery
//! of Subspace Structures by Low-Rank Representation*. TPAMI.

use crate::error::{ClusteringError, Result};
use crate::subspace_advanced::ssc::spectral_cluster_normalized;

/// Low-Rank Representation clustering.
///
/// # Example
///
/// ```
/// use scirs2_cluster::subspace_advanced::LowRankRepresentation;
///
/// let data = vec![
///     vec![1.0, 0.0], vec![2.0, 0.0], vec![3.0, 0.0],
///     vec![0.0, 1.0], vec![0.0, 2.0], vec![0.0, 3.0],
/// ];
/// let labels = LowRankRepresentation::new(2, 0.1).fit(&data).expect("operation should succeed");
/// assert_eq!(labels.len(), 6);
/// ```
pub struct LowRankRepresentation {
    /// Number of output clusters.
    pub n_clusters: usize,
    /// Nuclear-norm / L_{2,1} regularisation weight.
    pub lambda: f64,
    /// Maximum ADMM iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
}

impl LowRankRepresentation {
    /// Create a new `LowRankRepresentation` instance.
    pub fn new(n_clusters: usize, lambda: f64) -> Self {
        Self {
            n_clusters,
            lambda,
            max_iter: 100,
            tol: 1e-6,
        }
    }

    /// Override max iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Fit LRR and return cluster labels.
    ///
    /// # Errors
    ///
    /// Returns an error if `data` is empty or `n_clusters` exceeds the number
    /// of points.
    pub fn fit(&self, data: &[Vec<f64>]) -> Result<Vec<usize>> {
        let n = data.len();
        if n == 0 {
            return Err(ClusteringError::InvalidInput(
                "input data must not be empty".to_string(),
            ));
        }
        if self.n_clusters > n {
            return Err(ClusteringError::InvalidInput(format!(
                "n_clusters ({}) exceeds number of data points ({})",
                self.n_clusters, n
            )));
        }

        // Compute the self-representation Z via nuclear-norm SVT on Gram matrix.
        let z = self.compute_low_rank_representation(data)?;

        // Build symmetric affinity W = (|Z| + |Z^T|) / 2.
        let mut affinity = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                affinity[i][j] = (z[i][j].abs() + z[j][i].abs()) / 2.0;
            }
        }

        // Spectral clustering on affinity.
        spectral_cluster_normalized(&affinity, n, self.n_clusters)
    }

    /// Compute the low-rank self-representation matrix Z.
    ///
    /// Uses the ADMM update rule:
    ///   Z <- prox_{||·||_* / rho}(X^T X + rho * A - Y)  (SVT on n×n Gram)
    ///   A <- prox_{lambda/rho * ||·||_{2,1}}(Z + Y/rho)
    ///   Y <- Y + rho * (Z - A)
    ///
    /// Simplified: no explicit noise E; uses a single-block ADMM.
    fn compute_low_rank_representation(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = data.len();

        // Compute Gram matrix G = X^T X  (n×n, self inner products).
        let g = gram_matrix(data);

        // ADMM initialisation.
        let rho = 1.0;
        let mut z = vec![vec![0.0f64; n]; n];
        let mut a = vec![vec![0.0f64; n]; n];
        let mut y = vec![vec![0.0f64; n]; n]; // dual variable

        for _iter in 0..self.max_iter {
            let z_prev = z.clone();

            // Z-update: SVT( G + rho * A - Y, 1/rho ).
            //   Arg = G + rho*A - Y
            //   Z = prox_{||·||_*/rho}(Arg) ≈ soft-threshold(Arg, 1/rho)
            //   (element-wise SVT approximation on symmetric matrix).
            let thresh = 1.0 / rho;
            for i in 0..n {
                for j in 0..n {
                    let arg = g[i][j] + rho * a[i][j] - y[i][j];
                    z[i][j] = soft_threshold_scalar(arg, thresh);
                }
            }

            // A-update: prox_{lambda/rho * ||·||_{2,1}}(Z + Y/rho).
            //   Row-wise soft-threshold by lambda/rho on the L2 norm of each row.
            let row_thresh = self.lambda / rho;
            for i in 0..n {
                let mut tmp_row: Vec<f64> = (0..n).map(|j| z[i][j] + y[i][j] / rho).collect();
                let row_norm: f64 = tmp_row.iter().map(|x| x * x).sum::<f64>().sqrt();
                if row_norm > row_thresh {
                    let scale = 1.0 - row_thresh / row_norm;
                    for j in 0..n {
                        a[i][j] = scale * tmp_row[j];
                    }
                } else {
                    for j in 0..n {
                        a[i][j] = 0.0;
                    }
                }
            }

            // Y-update: dual ascent.
            for i in 0..n {
                for j in 0..n {
                    y[i][j] += rho * (z[i][j] - a[i][j]);
                }
            }

            // Check primal convergence.
            let primal_res: f64 = z
                .iter()
                .zip(z_prev.iter())
                .flat_map(|(r, rp)| r.iter().zip(rp.iter()).map(|(a, b)| (a - b).powi(2)))
                .sum::<f64>()
                .sqrt();
            if primal_res < self.tol {
                break;
            }
        }

        Ok(z)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the n×n Gram matrix G_{ij} = <x_i, x_j>.
fn gram_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = data.len();
    let mut g = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let dot: f64 = data[i]
                .iter()
                .zip(data[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            g[i][j] = dot;
            g[j][i] = dot;
        }
    }
    g
}

/// Element-wise soft-threshold (proximal operator of L1 / nuclear norm entry).
#[inline]
fn soft_threshold_scalar(z: f64, thresh: f64) -> f64 {
    if z > thresh {
        z - thresh
    } else if z < -thresh {
        z + thresh
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_subspace_data() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 0.0, 0.0],
            vec![2.0, 0.0, 0.0],
            vec![3.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![0.0, 3.0, 0.0],
        ]
    }

    #[test]
    fn test_lrr_basic() {
        let data = two_subspace_data();
        let labels = LowRankRepresentation::new(2, 0.1)
            .fit(&data)
            .expect("LRR fit should succeed");
        assert_eq!(labels.len(), 6);
        for &l in &labels {
            assert!(l < 2);
        }
    }

    #[test]
    fn test_lrr_empty_input() {
        let data: Vec<Vec<f64>> = vec![];
        let err = LowRankRepresentation::new(2, 0.1).fit(&data);
        assert!(err.is_err());
    }

    #[test]
    fn test_lrr_n_clusters_exceeds_n() {
        let data = vec![vec![1.0, 0.0]];
        let err = LowRankRepresentation::new(5, 0.1).fit(&data);
        assert!(err.is_err());
    }

    #[test]
    fn test_gram_matrix() {
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let g = gram_matrix(&data);
        assert!((g[0][0] - 1.0).abs() < 1e-10);
        assert!((g[0][1] - 0.0).abs() < 1e-10);
        assert!((g[0][2] - 1.0).abs() < 1e-10);
        assert!((g[1][2] - 1.0).abs() < 1e-10);
        assert!((g[2][2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_soft_threshold_scalar() {
        assert!((soft_threshold_scalar(0.5, 0.3) - 0.2).abs() < 1e-10);
        assert!((soft_threshold_scalar(-0.5, 0.3) + 0.2).abs() < 1e-10);
        assert_eq!(soft_threshold_scalar(0.1, 0.3), 0.0);
    }
}
