//! Low-rank matrix completion benchmark generator
//!
//! Generates synthetic low-rank matrices with partial observations for
//! testing matrix completion algorithms such as nuclear-norm minimization,
//! ALS, and SVT.

/// Configuration for the low-rank matrix completion benchmark
#[derive(Debug, Clone)]
pub struct LowRankConfig {
    /// Number of rows
    pub n_rows: usize,
    /// Number of columns
    pub n_cols: usize,
    /// True rank of the matrix
    pub rank: usize,
    /// Standard deviation of Gaussian noise added to observations
    pub noise_std: f64,
    /// Fraction of entries that are observed (0 < fraction <= 1)
    pub observation_fraction: f64,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for LowRankConfig {
    fn default() -> Self {
        Self {
            n_rows: 100,
            n_cols: 100,
            rank: 5,
            noise_std: 0.1,
            observation_fraction: 0.5,
            seed: 42,
        }
    }
}

/// Low-rank matrix completion benchmark dataset
#[derive(Debug, Clone)]
pub struct LowRankDataset {
    /// True low-rank matrix (n_rows × n_cols)
    pub matrix: Vec<Vec<f64>>,
    /// Boolean mask indicating which entries are observed
    pub observed_mask: Vec<Vec<bool>>,
    /// Noisy observations (None for unobserved entries)
    pub noisy_observations: Vec<Vec<Option<f64>>>,
    /// True singular values of the matrix (approximate via power iteration)
    pub singular_values: Vec<f64>,
}

/// Simple seeded LCG PRNG for deterministic generation
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-10);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Generate a low-rank matrix completion benchmark dataset.
///
/// The true matrix is M = U * V^T where U ∈ R^{n_rows × rank} and
/// V ∈ R^{n_cols × rank} are drawn from standard normal distributions.
/// Observations are sampled uniformly at random with the given fraction,
/// and Gaussian noise is added to each observed entry.
///
/// # Arguments
///
/// * `config` - Configuration specifying dimensions, rank, noise, and observation fraction
///
/// # Returns
///
/// A [`LowRankDataset`] containing the true matrix, observation mask,
/// noisy observations, and approximate singular values.
pub fn make_low_rank(config: &LowRankConfig) -> LowRankDataset {
    let mut rng = Lcg::new(config.seed);

    // Generate factor matrices U (n_rows × rank) and V (n_cols × rank)
    let u_mat: Vec<Vec<f64>> = (0..config.n_rows)
        .map(|_| (0..config.rank).map(|_| rng.next_normal()).collect())
        .collect();
    let v_mat: Vec<Vec<f64>> = (0..config.n_cols)
        .map(|_| (0..config.rank).map(|_| rng.next_normal()).collect())
        .collect();

    // Compute true matrix M = U * V^T
    let matrix: Vec<Vec<f64>> = (0..config.n_rows)
        .map(|i| {
            (0..config.n_cols)
                .map(|j| {
                    (0..config.rank)
                        .map(|k| u_mat[i][k] * v_mat[j][k])
                        .sum::<f64>()
                })
                .collect()
        })
        .collect();

    // Generate observation mask
    let observed_mask: Vec<Vec<bool>> = (0..config.n_rows)
        .map(|_| {
            (0..config.n_cols)
                .map(|_| rng.next_f64() < config.observation_fraction)
                .collect()
        })
        .collect();

    // Add Gaussian noise to observed entries
    let noisy_observations: Vec<Vec<Option<f64>>> = (0..config.n_rows)
        .map(|i| {
            (0..config.n_cols)
                .map(|j| {
                    if observed_mask[i][j] {
                        let noise = rng.next_normal() * config.noise_std;
                        Some(matrix[i][j] + noise)
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect();

    // Approximate singular values via power iteration on M^T * M
    let singular_values = approximate_singular_values(&matrix, config.rank, config.seed);

    LowRankDataset {
        matrix,
        observed_mask,
        noisy_observations,
        singular_values,
    }
}

/// Approximate the top-k singular values of a matrix using the power method.
///
/// Uses the power iteration method on M^T * M to find eigenvalues, then
/// computes singular values as their square roots.
fn approximate_singular_values(matrix: &[Vec<f64>], k: usize, seed: u64) -> Vec<f64> {
    let n_rows = matrix.len();
    if n_rows == 0 {
        return Vec::new();
    }
    let n_cols = matrix[0].len();
    if n_cols == 0 {
        return Vec::new();
    }

    let k = k.min(n_cols).min(n_rows);
    let mut rng = Lcg::new(seed.wrapping_add(9_999_999));
    let mut singular_values = Vec::with_capacity(k);

    // Work on a deflated copy of M^T * M
    // We iteratively find leading singular vectors via power iteration
    let mut residual: Vec<Vec<f64>> = matrix.to_vec();

    for _ in 0..k {
        // Initialize random vector in column space
        let mut v: Vec<f64> = (0..n_cols).map(|_| rng.next_normal()).collect();
        normalize_vec(&mut v);

        // Power iteration: v <- M^T * (M * v) / ||...||
        for _ in 0..50 {
            // u = M * v
            let u: Vec<f64> = (0..n_rows)
                .map(|i| (0..n_cols).map(|j| residual[i][j] * v[j]).sum::<f64>())
                .collect();
            // v_new = M^T * u
            let mut v_new: Vec<f64> = (0..n_cols)
                .map(|j| (0..n_rows).map(|i| residual[i][j] * u[i]).sum::<f64>())
                .collect();
            normalize_vec(&mut v_new);
            v = v_new;
        }

        // Compute corresponding left singular vector u = M * v
        let mut u: Vec<f64> = (0..n_rows)
            .map(|i| (0..n_cols).map(|j| residual[i][j] * v[j]).sum::<f64>())
            .collect();
        let sigma = norm_vec(&u);
        if sigma < 1e-12 {
            singular_values.push(0.0);
            continue;
        }
        singular_values.push(sigma);
        for x in &mut u {
            *x /= sigma;
        }

        // Deflate: residual <- residual - sigma * u * v^T
        for i in 0..n_rows {
            for j in 0..n_cols {
                residual[i][j] -= sigma * u[i] * v[j];
            }
        }
    }

    singular_values
}

fn norm_vec(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn normalize_vec(v: &mut [f64]) {
    let n = norm_vec(v);
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Compute the RMSE between a true matrix and an estimated matrix over all entries.
///
/// # Arguments
///
/// * `true_matrix` - Ground truth matrix (n_rows × n_cols)
/// * `estimated` - Estimated matrix of same dimensions
///
/// # Returns
///
/// Root mean squared error across all entries.
pub fn reconstruction_error(true_matrix: &[Vec<f64>], estimated: &[Vec<f64>]) -> f64 {
    let n_rows = true_matrix.len();
    if n_rows == 0 {
        return 0.0;
    }
    let n_cols = true_matrix[0].len();
    let total = (n_rows * n_cols) as f64;
    if total == 0.0 {
        return 0.0;
    }
    let sse: f64 = (0..n_rows)
        .map(|i| {
            (0..n_cols)
                .map(|j| {
                    let diff = true_matrix[i][j] - estimated[i][j];
                    diff * diff
                })
                .sum::<f64>()
        })
        .sum();
    (sse / total).sqrt()
}

/// Compute the RMSE between the dataset's true matrix and an estimated matrix,
/// evaluated only on observed entries.
///
/// # Arguments
///
/// * `dataset` - Low-rank dataset containing the true matrix and observation mask
/// * `estimated` - Estimated matrix to evaluate
///
/// # Returns
///
/// Root mean squared error on observed entries only.
pub fn observed_rmse(dataset: &LowRankDataset, estimated: &[Vec<f64>]) -> f64 {
    let n_rows = dataset.matrix.len();
    if n_rows == 0 {
        return 0.0;
    }
    let n_cols = dataset.matrix[0].len();
    let mut sse = 0.0f64;
    let mut count = 0usize;
    for ((true_row, est_row), mask_row) in dataset
        .matrix
        .iter()
        .zip(estimated.iter())
        .zip(dataset.observed_mask.iter())
    {
        for ((&t, &e), &observed) in true_row.iter().zip(est_row.iter()).zip(mask_row.iter()) {
            if observed {
                let diff = t - e;
                sse += diff * diff;
                count += 1;
            }
        }
    }
    if count == 0 {
        return 0.0;
    }
    (sse / count as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_rank_singular_values() {
        let config = LowRankConfig {
            n_rows: 20,
            n_cols: 20,
            rank: 2,
            noise_std: 0.01,
            observation_fraction: 0.8,
            seed: 123,
        };
        let ds = make_low_rank(&config);
        assert_eq!(ds.singular_values.len(), 2);
        // The two singular values should be significantly larger than zero
        assert!(
            ds.singular_values[0] > 1.0,
            "First singular value should be notable"
        );
        // Matrix dimensions
        assert_eq!(ds.matrix.len(), 20);
        assert_eq!(ds.matrix[0].len(), 20);
    }

    #[test]
    fn test_observation_count() {
        let config = LowRankConfig {
            n_rows: 10,
            n_cols: 10,
            rank: 2,
            noise_std: 0.0,
            observation_fraction: 0.5,
            seed: 42,
        };
        let ds = make_low_rank(&config);
        let observed_count: usize = ds
            .observed_mask
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&v| v)
            .count();
        // Expect roughly 50 ± 20 observations
        assert!(
            observed_count > 20,
            "Too few observations: {observed_count}"
        );
        assert!(
            observed_count <= 100,
            "Too many observations: {observed_count}"
        );
    }

    #[test]
    fn test_noisy_observations_match_mask() {
        let config = LowRankConfig::default();
        let ds = make_low_rank(&config);
        for i in 0..ds.matrix.len() {
            for j in 0..ds.matrix[0].len() {
                match ds.noisy_observations[i][j] {
                    Some(_) => assert!(ds.observed_mask[i][j]),
                    None => assert!(!ds.observed_mask[i][j]),
                }
            }
        }
    }

    #[test]
    fn test_reconstruction_error_zero_on_exact() {
        let config = LowRankConfig {
            n_rows: 5,
            n_cols: 5,
            rank: 2,
            noise_std: 0.0,
            observation_fraction: 1.0,
            seed: 1,
        };
        let ds = make_low_rank(&config);
        let err = reconstruction_error(&ds.matrix, &ds.matrix);
        assert!(err < 1e-12, "Self-reconstruction error should be zero");
    }

    #[test]
    fn test_observed_rmse() {
        let config = LowRankConfig {
            n_rows: 10,
            n_cols: 10,
            rank: 2,
            noise_std: 0.0,
            observation_fraction: 0.8,
            seed: 7,
        };
        let ds = make_low_rank(&config);
        // Perfect estimate => observed RMSE = 0
        let err = observed_rmse(&ds, &ds.matrix);
        assert!(err < 1e-12, "Observed RMSE against truth should be zero");
    }
}
