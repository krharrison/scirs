//! Sparse Subspace Clustering (SSC).
//!
//! Each data point is expressed as a sparse linear combination of *other* data
//! points in the dataset (Elhamifar & Vidal 2013). The self-expression matrix C
//! is found by solving:
//!
//! ```text
//!   min_C  ||C||_1   s.t.  X = XC,  diag(C) = 0
//! ```
//!
//! approximated via a proximal gradient (ISTA) descent.  The symmetric affinity
//! W = (|C| + |C^T|) / 2 is then used as input to normalized spectral clustering.

use crate::error::{ClusteringError, Result};

/// Sparse Subspace Clustering.
///
/// # Example
///
/// ```
/// use scirs2_cluster::subspace_advanced::SparseSubspaceClustering;
///
/// // Three points on span{e1} and three on span{e2}
/// let data = vec![
///     vec![1.0, 0.0, 0.0],
///     vec![2.0, 0.0, 0.0],
///     vec![3.0, 0.0, 0.0],
///     vec![0.0, 1.0, 0.0],
///     vec![0.0, 2.0, 0.0],
///     vec![0.0, 3.0, 0.0],
/// ];
/// let labels = SparseSubspaceClustering::new(2, 0.1).fit(&data).expect("operation should succeed");
/// assert_eq!(labels.len(), 6);
/// ```
pub struct SparseSubspaceClustering {
    /// Number of clusters for spectral step.
    pub n_clusters: usize,
    /// L1 sparsity regularisation weight.
    pub lambda: f64,
    /// Maximum ISTA iterations per data point.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// ISTA step size (learning rate).
    pub step_size: f64,
}

impl SparseSubspaceClustering {
    /// Create a new `SparseSubspaceClustering` with given cluster count and lambda.
    pub fn new(n_clusters: usize, lambda: f64) -> Self {
        Self {
            n_clusters,
            lambda,
            max_iter: 500,
            tol: 1e-6,
            step_size: 0.01,
        }
    }

    /// Override the maximum ISTA iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Override the ISTA step size.
    pub fn with_step_size(mut self, step: f64) -> Self {
        self.step_size = step;
        self
    }

    /// Fit SSC and return cluster label for each data point.
    ///
    /// # Errors
    ///
    /// Returns an error if `data` is empty or if `n_clusters` exceeds the
    /// number of points.
    pub fn fit(&self, data: &[Vec<f64>]) -> Result<Vec<usize>> {
        let n = data.len();
        if n == 0 {
            return Err(ClusteringError::InvalidInput(
                "input data must not be empty".to_string(),
            ));
        }
        if self.n_clusters > n {
            return Err(ClusteringError::InvalidInput(format!(
                "n_clusters ({}) exceeds number of points ({})",
                self.n_clusters, n
            )));
        }

        // Step 1: solve sparse self-expression for each point.
        let c_matrix = self.solve_sparse_representation(data)?;

        // Step 2: build symmetric affinity W = (|C| + |C^T|) / 2.
        let mut affinity = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                affinity[i][j] = (c_matrix[i][j].abs() + c_matrix[j][i].abs()) / 2.0;
            }
        }

        // Step 3: normalized spectral clustering on W.
        spectral_cluster_normalized(&affinity, n, self.n_clusters)
    }

    /// Solve the LASSO self-expression problem for every data point via
    /// proximal gradient descent (ISTA).
    fn solve_sparse_representation(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = data.len();
        let dim = data[0].len();
        let mut c = vec![vec![0.0f64; n]; n];

        for i in 0..n {
            let xi = &data[i];
            let mut ci = vec![0.0f64; n];
            let mut prev_ci = ci.clone();

            for _iter in 0..self.max_iter {
                // Compute residual: xi - X * ci  (exclude j==i via ci[i]=0)
                let mut residual = xi.clone();
                for j in 0..n {
                    if j == i || ci[j] == 0.0 {
                        continue;
                    }
                    for d in 0..dim {
                        residual[d] -= ci[j] * data[j][d];
                    }
                }

                // Gradient step + soft-threshold for each j ≠ i.
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    // Gradient of 0.5 * ||residual||^2 w.r.t. ci[j] = -<data[j], residual>
                    let grad: f64 = -data[j]
                        .iter()
                        .zip(residual.iter())
                        .map(|(a, r)| a * r)
                        .sum::<f64>();

                    let z = ci[j] - self.step_size * grad;
                    let thresh = self.step_size * self.lambda;
                    ci[j] = soft_threshold(z, thresh);
                }

                // Convergence check.
                let change: f64 = ci
                    .iter()
                    .zip(prev_ci.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if change < self.tol {
                    break;
                }
                prev_ci.clone_from(&ci);
            }
            c[i] = ci;
        }
        Ok(c)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Soft-threshold (proximal operator of L1 norm).
#[inline]
fn soft_threshold(z: f64, thresh: f64) -> f64 {
    if z > thresh {
        z - thresh
    } else if z < -thresh {
        z + thresh
    } else {
        0.0
    }
}

/// Normalized spectral clustering using the random-walk Laplacian L = D^{-1} W.
///
/// Uses power iteration to approximate the top-k eigenvectors, then k-means on
/// the resulting n×k embedding.
pub(crate) fn spectral_cluster_normalized(
    affinity: &[Vec<f64>],
    n: usize,
    k: usize,
) -> Result<Vec<usize>> {
    if k == 0 {
        return Err(ClusteringError::InvalidInput(
            "n_clusters must be at least 1".to_string(),
        ));
    }

    // Degree vector.
    let d: Vec<f64> = affinity
        .iter()
        .map(|row| row.iter().sum::<f64>().max(1e-12))
        .collect();

    // Build normalized row-stochastic matrix P = D^{-1} W (random-walk Laplacian).
    let p: Vec<Vec<f64>> = affinity
        .iter()
        .zip(d.iter())
        .map(|(row, &di)| row.iter().map(|&w| w / di).collect())
        .collect();

    // Power iteration to find the top-k eigenvectors of P.
    // Initialise with identity-like vectors to spread across the spectrum.
    let mut vecs: Vec<Vec<f64>> = (0..k)
        .map(|j| {
            (0..n)
                .map(|i| if i == (j % n) { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();

    for _iter in 0..100 {
        for v in &mut vecs {
            // v <- P * v
            let new_v: Vec<f64> = (0..n)
                .map(|i| {
                    p[i].iter()
                        .zip(v.iter())
                        .map(|(pij, vj)| pij * vj)
                        .sum::<f64>()
                })
                .collect();
            // L2 normalize.
            let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            *v = new_v.into_iter().map(|x| x / norm).collect();
        }
    }

    // Build embedding matrix: each row i has k features vecs[0][i], ..., vecs[k-1][i].
    let embedding: Vec<Vec<f64>> = (0..n)
        .map(|i| vecs.iter().map(|v| v[i]).collect())
        .collect();

    // K-means on the embedding.
    kmeans_pp(&embedding, k)
}

/// K-means++ initialization + Lloyd iterations.
fn kmeans_pp(data: &[Vec<f64>], k: usize) -> Result<Vec<usize>> {
    let n = data.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let dim = data[0].len();
    let k = k.min(n);

    // K-means++ initialization: pick first center uniformly, subsequent centers
    // with probability proportional to squared distance.
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    centroids.push(data[0].to_vec());

    for _ in 1..k {
        let mut dists: Vec<f64> = data
            .iter()
            .map(|p| {
                centroids
                    .iter()
                    .map(|c| {
                        c.iter()
                            .zip(p.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        let total: f64 = dists.iter().sum();
        if total < 1e-15 {
            // All remaining points are essentially at existing centroids.
            centroids.push(data[centroids.len() % n].to_vec());
            continue;
        }

        // Normalise to probabilities and pick via cumulative sum.
        let target = total * 0.5; // deterministic selection at median for reproducibility
        let mut cumsum = 0.0;
        let mut chosen = n - 1;
        for (idx, &d) in dists.iter().enumerate() {
            cumsum += d;
            if cumsum >= target {
                chosen = idx;
                break;
            }
        }
        centroids.push(data[chosen].to_vec());
    }

    let mut labels = vec![0usize; n];

    // Lloyd iterations.
    for _iter in 0..200 {
        let mut changed = false;

        // Assignment step.
        for (i, point) in data.iter().enumerate() {
            let closest = centroids
                .iter()
                .enumerate()
                .map(|(ci, c)| {
                    let d: f64 = c
                        .iter()
                        .zip(point.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (ci, d)
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(ci, _)| ci)
                .unwrap_or(0);

            if labels[i] != closest {
                changed = true;
                labels[i] = closest;
            }
        }

        if !changed {
            break;
        }

        // Update step.
        let mut sums = vec![vec![0.0f64; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, &l) in labels.iter().enumerate() {
            for d in 0..dim {
                sums[l][d] += data[i][d];
            }
            counts[l] += 1;
        }
        for (l, (s, &c)) in sums.iter().zip(counts.iter()).enumerate() {
            if c > 0 {
                centroids[l] = s.iter().map(|x| x / c as f64).collect();
            }
        }
    }

    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subspace_data() -> Vec<Vec<f64>> {
        // Three points on span{e1} and three on span{e2}
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
    fn test_ssc_basic() {
        let data = subspace_data();
        let labels = SparseSubspaceClustering::new(2, 0.1)
            .fit(&data)
            .expect("SSC fit should succeed");
        assert_eq!(labels.len(), 6);
        // Each label must be in [0, 2)
        for &l in &labels {
            assert!(l < 2);
        }
    }

    #[test]
    fn test_ssc_empty_input() {
        let data: Vec<Vec<f64>> = vec![];
        let err = SparseSubspaceClustering::new(2, 0.1).fit(&data);
        assert!(err.is_err());
    }

    #[test]
    fn test_ssc_n_clusters_exceeds_n() {
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let err = SparseSubspaceClustering::new(5, 0.1).fit(&data);
        assert!(err.is_err());
    }

    #[test]
    fn test_soft_threshold() {
        assert!((soft_threshold(0.5, 0.3) - 0.2).abs() < 1e-10);
        assert!((soft_threshold(-0.5, 0.3) + 0.2).abs() < 1e-10);
        assert_eq!(soft_threshold(0.1, 0.3), 0.0);
    }
}
