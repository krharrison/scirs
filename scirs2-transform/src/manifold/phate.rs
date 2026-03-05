//! PHATE: Potential of Heat-diffusion for Affinity-based Trajectory Embedding
//! (Moon et al. 2019).
//!
//! PHATE preserves both local and global structure by computing a diffusion
//! process on the data graph and then embedding the resulting *potential
//! distances* with classical MDS (CMDS).
//!
//! ## Algorithm Overview
//!
//! 1. Build an adaptive Gaussian affinity kernel with per-point bandwidth
//!    (k-NN adaptive: `σ_i = d(x_i, x_{i,k})`).
//! 2. Row-normalize to obtain the Markov transition matrix `P`.
//! 3. Compute diffused matrix `P^t` via repeated matrix multiplication.
//! 4. Compute potential distances:
//!    `U(i, j) = -log(P^t(i, j) + ε)` (or informational distance variant).
//! 5. Embed potential distances with classical MDS (double-centering +
//!    truncated eigendecomposition via power iteration).
//!
//! ## References
//!
//! - Moon, K. R., van Dijk, D., Wang, Z., Gigante, S., Burkhardt, D. B.,
//!   Chen, W. S., … & Krishnaswamy, S. (2019). Visualizing structure and
//!   transitions in high-dimensional biological data. *Nature Biotechnology*,
//!   37(12), 1482–1492. https://doi.org/10.1038/s41587-019-0336-3

use crate::error::TransformError;

// ─── Public Types ─────────────────────────────────────────────────────────────

/// Hyperparameters for the PHATE algorithm.
#[derive(Debug, Clone)]
pub struct PHATEParams {
    /// Number of output embedding dimensions (default 2)
    pub n_components: usize,
    /// Number of nearest neighbours used to set the adaptive bandwidth (default 5)
    pub k: usize,
    /// Maximum number of landmark points for large-N approximation (default 2000)
    pub n_landmark: usize,
    /// Diffusion time — number of Markov chain steps (default 1)
    pub t: usize,
    /// Informational distance exponent γ.
    ///   * `γ = 1`  → Euclidean distance on log probabilities (cosine variant)
    ///   * `γ = -1` → Von Neumann entropy variant
    ///   Default 1.0.
    pub gamma: f64,
}

impl Default for PHATEParams {
    fn default() -> Self {
        Self {
            n_components: 2,
            k: 5,
            n_landmark: 2000,
            t: 1,
            gamma: 1.0,
        }
    }
}

/// Result returned by [`PHATE::fit_transform`].
#[derive(Debug, Clone)]
pub struct PHATEResult {
    /// Low-dimensional embedding: `[n_samples][n_components]`
    pub embedding: Vec<Vec<f64>>,
    /// Potential distance matrix `[n_samples][n_samples]`
    ///
    /// Contains `U(i, j) = -log(P^t(i, j))` values.
    pub potential: Vec<Vec<f64>>,
    /// The diffusion time `t` that was used
    pub diffusion_time: usize,
}

/// PHATE dimensionality reduction.
///
/// # Example
/// ```rust,no_run
/// use scirs2_transform::manifold::phate::PHATE;
/// let data: Vec<Vec<f64>> = (0..20)
///     .map(|i| vec![i as f64, (i as f64).sqrt()])
///     .collect();
/// let result = PHATE::new(2).with_k(3).fit_transform(&data).expect("should succeed");
/// assert_eq!(result.embedding.len(), 20);
/// ```
pub struct PHATE {
    /// Hyperparameters controlling the PHATE run
    pub params: PHATEParams,
}

// ─── Implementation ───────────────────────────────────────────────────────────

impl PHATE {
    /// Create a new PHATE reducer with default parameters and `n_components`
    /// output dimensions.
    pub fn new(n_components: usize) -> Self {
        let mut params = PHATEParams::default();
        params.n_components = n_components;
        Self { params }
    }

    /// Set the number of nearest neighbours for adaptive bandwidth.
    pub fn with_k(mut self, k: usize) -> Self {
        self.params.k = k;
        self
    }

    /// Set the diffusion time.
    pub fn with_t(mut self, t: usize) -> Self {
        self.params.t = t;
        self
    }

    /// Set the informational distance exponent γ.
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.params.gamma = gamma;
        self
    }

    /// Fit and embed `data`.
    ///
    /// # Arguments
    /// * `data` – Row-major dataset; each inner `Vec<f64>` is one sample.
    ///
    /// # Errors
    /// Returns `TransformError::InvalidInput` for empty input.
    pub fn fit_transform(&self, data: &[Vec<f64>]) -> Result<PHATEResult, TransformError> {
        let n = data.len();
        if n == 0 {
            return Err(TransformError::InvalidInput(
                "PHATE requires at least one sample".into(),
            ));
        }
        if n == 1 {
            return Ok(PHATEResult {
                embedding: vec![vec![0.0; self.params.n_components]],
                potential: vec![vec![0.0]],
                diffusion_time: self.params.t,
            });
        }

        let k = self.params.k.min(n - 1).max(1);

        // 1. Build alpha-decaying (adaptive Gaussian) Markov kernel
        let kernel = self.compute_markov_kernel(data, k);

        // 2. Diffuse P^t
        let diffused = self.diffuse_kernel(&kernel, n, self.params.t);

        // 3. Compute potential distances
        let potential = self.compute_potential_distances(&diffused, n);

        // 4. Classical MDS on potential distances
        let embedding = self.classical_mds(&potential, n)?;

        Ok(PHATEResult {
            embedding,
            potential,
            diffusion_time: self.params.t,
        })
    }

    // ── Step 1: Adaptive Gaussian Markov kernel ───────────────────────────────

    /// Build the row-normalized Markov transition matrix using an adaptive
    /// Gaussian kernel with per-point bandwidth `σ_i = d(x_i, x_{i,k})`.
    ///
    /// Kernel entry: `K(i, j) = exp(-||x_i - x_j||^2 / (σ_i · σ_j))`
    fn compute_markov_kernel(&self, data: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
        let n = data.len();

        // Compute all pairwise squared distances
        let sq_dists: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        data[i]
                            .iter()
                            .zip(data[j].iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                    })
                    .collect()
            })
            .collect();

        // Per-point adaptive bandwidth: distance to the k-th neighbour
        let bandwidths: Vec<f64> = (0..n)
            .map(|i| {
                let mut sorted_dists: Vec<f64> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| sq_dists[i][j].sqrt())
                    .collect();
                sorted_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                // k-th NN distance (1-indexed)
                sorted_dists
                    .get(k.saturating_sub(1))
                    .copied()
                    .unwrap_or(1.0)
                    .max(1e-10)
            })
            .collect();

        // Build kernel matrix
        let mut kernel: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let denom = bandwidths[i] * bandwidths[j];
                        (-sq_dists[i][j] / denom.max(1e-20)).exp()
                    })
                    .collect()
            })
            .collect();

        // Row-normalize to Markov matrix
        for row in &mut kernel {
            let s: f64 = row.iter().sum::<f64>().max(1e-15);
            for v in row.iter_mut() {
                *v /= s;
            }
        }

        kernel
    }

    // ── Step 2: Diffusion P^t ─────────────────────────────────────────────────

    /// Compute `P^t` by iterated matrix multiplication.
    ///
    /// For `t = 1` this is a no-op (returns `p`).
    /// For `t > 1` we perform `t - 1` additional multiplications.
    fn diffuse_kernel(&self, p: &[Vec<f64>], n: usize, t: usize) -> Vec<Vec<f64>> {
        if t <= 1 {
            return p.to_vec();
        }
        let mut result = p.to_vec();
        for _ in 1..t {
            let new_result = mat_mul(&result, p, n);
            result = new_result;
        }
        result
    }

    // ── Step 3: Potential distances ───────────────────────────────────────────

    /// Compute the PHATE potential distance matrix.
    ///
    /// `U(i, j) = ||u_i − u_j||_2` where `u_i(k) = -log(P^t(i,k) + ε)`.
    ///
    /// The resulting pairwise distance is then built from these log-probability
    /// vectors via Euclidean distance in "potential space".
    fn compute_potential_distances(&self, diff: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
        let eps = 1e-7;
        // Compute log-probability (potential) vectors u_i
        let u: Vec<Vec<f64>> = diff
            .iter()
            .map(|row| row.iter().map(|&p| -(p + eps).ln()).collect())
            .collect();

        // Build pairwise Euclidean distance matrix in potential space
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        u[i].iter()
                            .zip(u[j].iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt()
                    })
                    .collect()
            })
            .collect()
    }

    // ── Step 4: Classical MDS ─────────────────────────────────────────────────

    /// Classical Metric MDS via double-centering + power iteration.
    ///
    /// Given a pairwise distance matrix `D`, computes the `n × k` embedding
    /// from the top `k` eigenvectors of the double-centered matrix:
    ///
    /// `B = -0.5 · H D^2 H`   where `H = I − (1/n) 11^T`
    fn classical_mds(
        &self,
        distances: &[Vec<f64>],
        n: usize,
    ) -> Result<Vec<Vec<f64>>, TransformError> {
        let k = self.params.n_components.min(n - 1).max(1);

        // Squared distance matrix
        let d2: Vec<Vec<f64>> = distances
            .iter()
            .map(|row| row.iter().map(|x| x * x).collect())
            .collect();

        // Row means and grand mean for double-centering
        let row_means: Vec<f64> = d2
            .iter()
            .map(|row| row.iter().sum::<f64>() / n as f64)
            .collect();
        let grand_mean: f64 = row_means.iter().sum::<f64>() / n as f64;

        // B = -0.5 * (D^2 - row_mean - col_mean + grand_mean)
        let mut b: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        -0.5 * (d2[i][j] - row_means[i] - row_means[j] + grand_mean)
                    })
                    .collect()
            })
            .collect();

        // Power iteration to extract top-k eigenvectors/values of B
        let mut embedding: Vec<Vec<f64>> = vec![vec![0.0; k]; n];

        for comp in 0..k {
            // Initial vector: linearly spaced to avoid all-equal (which stalls convergence)
            let mut v: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
            normalize_inplace(&mut v);

            let mut eigenval = 0.0f64;

            for _iter in 0..300 {
                let bv = mat_vec_mul(&b, &v, n);
                let new_ev: f64 = v.iter().zip(bv.iter()).map(|(vi, bvi)| vi * bvi).sum();
                let new_norm = bv
                    .iter()
                    .map(|x| x * x)
                    .sum::<f64>()
                    .sqrt()
                    .max(1e-15);
                let new_v: Vec<f64> = bv.iter().map(|x| x / new_norm).collect();
                let delta = (new_ev - eigenval).abs();
                eigenval = new_ev;
                v = new_v;
                if delta < 1e-12 {
                    break;
                }
            }

            // Embedding coordinate: sign(λ) · |λ|^0.5 · v_i
            let scale = eigenval.abs().sqrt();
            for i in 0..n {
                embedding[i][comp] = scale * v[i];
            }

            // Deflation: B ← B − λ · v·v^T
            for i in 0..n {
                for j in 0..n {
                    b[i][j] -= eigenval * v[i] * v[j];
                }
            }
        }

        Ok(embedding)
    }
}

// ─── Matrix utilities ─────────────────────────────────────────────────────────

/// Dense n×n matrix multiplication: `C = A · B`
fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut c = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for k in 0..n {
            if a[i][k].abs() < 1e-15 {
                continue; // skip near-zero entries for speed
            }
            for j in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Dense matrix–vector product: `y = A · x`
fn mat_vec_mul(a: &[Vec<f64>], x: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| a[i].iter().zip(x.iter()).map(|(aij, xj)| aij * xj).sum())
        .collect()
}

/// Normalize a vector in-place to unit L2 norm.
fn normalize_inplace(v: &mut Vec<f64>) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-15);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(n: usize) -> Vec<Vec<f64>> {
        use std::f64::consts::TAU;
        (0..n)
            .map(|i| {
                let t = TAU * i as f64 / n as f64;
                vec![t.cos(), t.sin(), (i as f64) * 0.05]
            })
            .collect()
    }

    #[test]
    fn test_phate_output_shape() {
        let data = make_data(10);
        let result = PHATE::new(2)
            .with_k(3)
            .fit_transform(&data)
            .expect("PHATE fit_transform");
        assert_eq!(result.embedding.len(), 10, "wrong number of samples");
        assert!(
            result.embedding.iter().all(|row| row.len() == 2),
            "every row should have 2 dimensions"
        );
    }

    #[test]
    fn test_phate_potential_shape() {
        let data = make_data(8);
        let result = PHATE::new(2).with_k(2).fit_transform(&data).expect("PHATE");
        assert_eq!(result.potential.len(), 8);
        assert!(result.potential.iter().all(|r| r.len() == 8));
    }

    #[test]
    fn test_phate_embedding_finite() {
        let data = make_data(12);
        let result = PHATE::new(2).with_k(3).fit_transform(&data).expect("PHATE");
        for row in &result.embedding {
            for &v in row {
                assert!(v.is_finite(), "embedding contains non-finite value: {v}");
            }
        }
    }

    #[test]
    fn test_phate_potential_nonneg() {
        // Potential distances should be non-negative
        let data = make_data(8);
        let result = PHATE::new(2).with_k(2).fit_transform(&data).expect("PHATE");
        for row in &result.potential {
            for &v in row {
                assert!(v >= 0.0, "negative potential distance: {v}");
            }
        }
    }

    #[test]
    fn test_phate_3d_output() {
        let data = make_data(10);
        let result = PHATE::new(3).with_k(3).fit_transform(&data).expect("PHATE 3d");
        assert_eq!(result.embedding.len(), 10);
        assert!(result.embedding.iter().all(|r| r.len() == 3));
    }

    #[test]
    fn test_phate_diffusion_time() {
        let data = make_data(10);
        let result = PHATE::new(2).with_k(3).with_t(3).fit_transform(&data).expect("PHATE t=3");
        assert_eq!(result.diffusion_time, 3);
    }

    #[test]
    fn test_phate_empty_input() {
        let result = PHATE::new(2).fit_transform(&[]);
        assert!(result.is_err(), "empty input should return Err");
    }

    #[test]
    fn test_phate_single_sample() {
        let data = vec![vec![1.0, 2.0]];
        let result = PHATE::new(2).fit_transform(&data).expect("single sample");
        assert_eq!(result.embedding.len(), 1);
        assert_eq!(result.embedding[0].len(), 2);
    }
}
