//! Bayesian Factor Analysis Model.
//!
//! The model is:
//! ```text
//! x_i = Λ f_i + ε_i,   ε_i ~ N(0, Ψ)
//! f_i ~ N(0, I_k)
//! Λ_{jk} ~ N(0, 1),    Ψ_{jj} ~ InvGamma(a₀, b₀)
//! ```
//! where Λ is the p×k factor loading matrix, f_i is the k-dimensional
//! latent factor vector, and Ψ is a diagonal residual covariance matrix.
//!
//! Estimation uses the EM algorithm (the factor analysis E/M steps).

use crate::error::{StatsError, StatsResult as Result};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Structure
// ---------------------------------------------------------------------------

/// Bayesian Factor Analysis Model.
///
/// After fitting, `loadings` holds the p×k factor loading matrix (as a
/// flat row-major vector), `uniquenesses` holds the diagonal residual
/// variances Ψ_{jj}, and `log_likelihood` is the final EM log-likelihood.
#[derive(Debug, Clone)]
pub struct FactorAnalysisModel {
    /// Number of observed variables.
    pub n_vars: usize,
    /// Number of latent factors.
    pub n_factors: usize,
    /// Factor loading matrix Λ, shape p×k (row-major, p rows, k cols).
    pub loadings: Vec<f64>,
    /// Diagonal residual variances Ψ (length p).
    pub uniquenesses: Vec<f64>,
    /// Factor score estimates for each observation (N × k, row-major).
    pub factor_scores: Vec<f64>,
    /// Number of observations used in the last fit.
    pub n_obs: usize,
    /// Log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Whether EM converged.
    pub converged: bool,
    /// Number of EM iterations run.
    pub n_iter: usize,
    /// Cumulative proportion of variance explained by each factor.
    pub cumulative_variance_explained: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl FactorAnalysisModel {
    /// Construct a new (unfitted) `FactorAnalysisModel`.
    ///
    /// # Errors
    /// Returns an error when `n_vars == 0` or `n_factors == 0` or
    /// `n_factors >= n_vars`.
    pub fn new(n_vars: usize, n_factors: usize) -> Result<Self> {
        if n_vars == 0 {
            return Err(StatsError::InvalidArgument(
                "n_vars must be >= 1".into(),
            ));
        }
        if n_factors == 0 {
            return Err(StatsError::InvalidArgument(
                "n_factors must be >= 1".into(),
            ));
        }
        if n_factors >= n_vars {
            return Err(StatsError::InvalidArgument(format!(
                "n_factors ({n_factors}) must be < n_vars ({n_vars})"
            )));
        }
        Ok(Self {
            n_vars,
            n_factors,
            loadings: vec![0.0; n_vars * n_factors],
            uniquenesses: vec![1.0; n_vars],
            factor_scores: Vec::new(),
            n_obs: 0,
            log_likelihood: f64::NEG_INFINITY,
            converged: false,
            n_iter: 0,
            cumulative_variance_explained: Vec::new(),
        })
    }

    /// Fit the factor model using the EM algorithm.
    ///
    /// # Parameters
    /// - `data`: Observations (N × p), each row is one observation.
    /// - `max_iter`: Maximum number of EM iterations.
    /// - `tol`: Convergence tolerance on the change in log-likelihood.
    ///
    /// # Errors
    /// Returns an error on dimension mismatches or insufficient data.
    pub fn fit_em(&mut self, data: &[Vec<f64>], max_iter: usize, tol: f64) -> Result<()> {
        let n = data.len();
        if n == 0 {
            return Err(StatsError::InsufficientData(
                "data must be non-empty".into(),
            ));
        }
        let p = self.n_vars;
        let k = self.n_factors;

        for (i, row) in data.iter().enumerate() {
            if row.len() != p {
                return Err(StatsError::DimensionMismatch(format!(
                    "data[{i}] has {} cols, expected {p}",
                    row.len()
                )));
            }
        }

        // Compute sample mean and center
        let mean = compute_mean(data, p);
        let centered: Vec<Vec<f64>> = data
            .iter()
            .map(|row| {
                row.iter()
                    .zip(mean.iter())
                    .map(|(&xi, &mi)| xi - mi)
                    .collect()
            })
            .collect();

        // Initialize loadings via first-k principal components of sample covariance
        let s_cov = sample_covariance(&centered, p);
        self.loadings = init_loadings_pca(&s_cov, p, k);
        // Initialize uniquenesses from residual variance
        self.uniquenesses = (0..p)
            .map(|j| {
                let lambda_sq: f64 = (0..k).map(|l| self.loadings[j * k + l].powi(2)).sum();
                (s_cov[j][j] - lambda_sq).max(1e-4)
            })
            .collect();

        let mut prev_ll = f64::NEG_INFINITY;

        for iter in 0..max_iter {
            self.n_iter = iter + 1;

            // ---- E-step ----
            // Compute β = Λ^T Ψ⁻¹ (k×p)
            // Factor scores: E[f|x] = β x_centered
            // E[f f^T | x] = I - β Λ + β x x^T β^T
            let beta = compute_beta(&self.loadings, &self.uniquenesses, p, k);

            let mut sum_eff = vec![vec![0.0_f64; k]; k]; // Σ E[ff^T|x]
            let mut sum_efx = vec![vec![0.0_f64; p]; k]; // Σ E[f|x] x^T
            let mut f_scores = vec![vec![0.0_f64; k]; n];

            for (i, xi) in centered.iter().enumerate() {
                // E[f_i | x_i] = β x_i
                let ef: Vec<f64> = (0..k)
                    .map(|l| (0..p).map(|j| beta[l][j] * xi[j]).sum::<f64>())
                    .collect();

                // E[f_i f_i^T | x_i] = (I - β Λ) + ef ef^T
                let i_minus_beta_lambda = compute_i_minus_beta_lambda(&beta, &self.loadings, p, k);
                for l1 in 0..k {
                    for l2 in 0..k {
                        sum_eff[l1][l2] +=
                            i_minus_beta_lambda[l1][l2] + ef[l1] * ef[l2];
                    }
                }
                for l in 0..k {
                    for j in 0..p {
                        sum_efx[l][j] += ef[l] * xi[j];
                    }
                }
                f_scores[i] = ef;
            }

            // ---- M-step ----
            // Update Λ: Λ_new = (Σ E[f|x] x^T)^T * (Σ E[ff^T|x])^{-1}
            // i.e., Λ_new[j, :] = (Σ_x x_j E[f|x]) * (Σ E[ff^T|x])^{-1}
            let sum_eff_inv = invert_sym_k(&sum_eff, k)?;

            let mut new_loadings = vec![0.0_f64; p * k];
            for j in 0..p {
                // row j of Λ: sum_efx[:, j] * sum_eff_inv
                for l2 in 0..k {
                    for l1 in 0..k {
                        new_loadings[j * k + l2] += sum_efx[l1][j] * sum_eff_inv[l1][l2];
                    }
                }
            }
            self.loadings = new_loadings;

            // Update Ψ: Ψ_jj = (1/n)(S_jj - Λ_j · (Σ E[f|x] x_j^T / n))
            // S_jj is the sample variance of variable j
            for j in 0..p {
                let s_jj = s_cov[j][j];
                // Λ_j · mean E[f|x] x_j contribution
                let lambda_term: f64 = (0..k)
                    .map(|l| self.loadings[j * k + l] * sum_efx[l][j] / n as f64)
                    .sum();
                self.uniquenesses[j] = (s_jj - lambda_term).max(1e-6);
            }

            // Compute log-likelihood
            let ll = factor_log_likelihood(&centered, &self.loadings, &self.uniquenesses, p, k, n);
            if (ll - prev_ll).abs() < tol && iter > 2 {
                self.converged = true;
                self.log_likelihood = ll;
                break;
            }
            prev_ll = ll;
            self.log_likelihood = ll;
        }

        // Store factor scores
        let beta = compute_beta(&self.loadings, &self.uniquenesses, p, k);
        self.factor_scores = Vec::with_capacity(n * k);
        for xi in &centered {
            for l in 0..k {
                let fs_l: f64 = (0..p).map(|j| beta[l][j] * xi[j]).sum();
                self.factor_scores.push(fs_l);
            }
        }

        self.n_obs = n;

        // Compute cumulative variance explained
        self.cumulative_variance_explained = compute_cum_var_explained(&self.loadings, &self.uniquenesses, p, k);

        Ok(())
    }

    /// Get the loading matrix as a 2-D vec (p × k).
    pub fn loading_matrix(&self) -> Vec<Vec<f64>> {
        let p = self.n_vars;
        let k = self.n_factors;
        (0..p)
            .map(|j| (0..k).map(|l| self.loadings[j * k + l]).collect())
            .collect()
    }

    /// Get factor scores for observation `i`.
    pub fn factor_scores_for(&self, i: usize) -> Result<Vec<f64>> {
        if self.factor_scores.is_empty() {
            return Err(StatsError::InvalidInput(
                "Model has not been fitted yet".into(),
            ));
        }
        let k = self.n_factors;
        let start = i * k;
        if start + k > self.factor_scores.len() {
            return Err(StatsError::InvalidArgument(format!(
                "observation index {i} out of range"
            )));
        }
        Ok(self.factor_scores[start..start + k].to_vec())
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn compute_mean(data: &[Vec<f64>], p: usize) -> Vec<f64> {
    let n = data.len() as f64;
    let mut mean = vec![0.0_f64; p];
    for row in data {
        for (j, &v) in row.iter().enumerate() {
            mean[j] += v;
        }
    }
    mean.iter_mut().for_each(|m| *m /= n);
    mean
}

fn sample_covariance(centered: &[Vec<f64>], p: usize) -> Vec<Vec<f64>> {
    let n = centered.len() as f64;
    let mut cov = vec![vec![0.0_f64; p]; p];
    for row in centered {
        for j in 0..p {
            for l in 0..p {
                cov[j][l] += row[j] * row[l];
            }
        }
    }
    for j in 0..p {
        for l in 0..p {
            cov[j][l] /= (n - 1.0).max(1.0);
        }
    }
    cov
}

/// Initialize loadings via the first-k eigenvectors of the sample covariance.
/// Uses power iteration for simplicity.
fn init_loadings_pca(s_cov: &[Vec<f64>], p: usize, k: usize) -> Vec<f64> {
    let mut loadings = vec![0.0_f64; p * k];

    // Use a deflated power iteration approach
    let mut s_deflated = s_cov.to_vec();

    for factor in 0..k {
        // Power iteration to find dominant eigenvector
        let mut v = vec![1.0_f64 / (p as f64).sqrt(); p];
        for _ in 0..50 {
            let mut av = vec![0.0_f64; p];
            for i in 0..p {
                for j in 0..p {
                    av[i] += s_deflated[i][j] * v[j];
                }
            }
            let norm: f64 = av.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm < 1e-12 {
                break;
            }
            v = av.iter().map(|&x| x / norm).collect();
        }
        // eigenvalue
        let lambda: f64 = (0..p)
            .map(|i| (0..p).map(|j| v[i] * s_deflated[i][j] * v[j]).sum::<f64>())
            .sum();
        let sqrt_lambda = lambda.sqrt().max(1e-8);
        for j in 0..p {
            loadings[j * k + factor] = v[j] * sqrt_lambda;
        }
        // Deflate
        for i in 0..p {
            for j in 0..p {
                s_deflated[i][j] -= lambda * v[i] * v[j];
            }
        }
    }
    loadings
}

/// Compute β = (Λ^T Ψ⁻¹ Λ + I)^{-1} Λ^T Ψ⁻¹   (k×p)
fn compute_beta(loadings: &[f64], psi: &[f64], p: usize, k: usize) -> Vec<Vec<f64>> {
    // M = Λ^T Ψ⁻¹ Λ + I_k   (k×k)
    let mut m = vec![vec![0.0_f64; k]; k];
    for l1 in 0..k {
        for l2 in 0..k {
            let val: f64 = (0..p)
                .map(|j| loadings[j * k + l1] * loadings[j * k + l2] / psi[j])
                .sum();
            m[l1][l2] = val + if l1 == l2 { 1.0 } else { 0.0 };
        }
    }
    let m_inv = invert_sym_k(&m, k).unwrap_or_else(|_| eye_k(k));

    // β = M_inv * Λ^T Ψ⁻¹   (k×p)
    let mut beta = vec![vec![0.0_f64; p]; k];
    for l in 0..k {
        for j in 0..p {
            let lambda_psi: f64 = loadings[j * k..j * k + k]
                .iter()
                .enumerate()
                .map(|(l2, &lam)| m_inv[l][l2] * lam)
                .sum();
            beta[l][j] = lambda_psi / psi[j];
        }
    }
    beta
}

/// Compute (I_k - β Λ)  (k×k)
fn compute_i_minus_beta_lambda(beta: &[Vec<f64>], loadings: &[f64], p: usize, k: usize) -> Vec<Vec<f64>> {
    let mut result = eye_k(k);
    for l1 in 0..k {
        for l2 in 0..k {
            let bl: f64 = (0..p)
                .map(|j| beta[l1][j] * loadings[j * k + l2])
                .sum();
            result[l1][l2] -= bl;
        }
    }
    result
}

fn eye_k(k: usize) -> Vec<Vec<f64>> {
    (0..k).map(|i| (0..k).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect()
}

/// Invert a small k×k symmetric matrix using Gaussian elimination.
fn invert_sym_k(m: &[Vec<f64>], k: usize) -> Result<Vec<Vec<f64>>> {
    let mut aug: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            let mut row = m[i].clone();
            row.extend((0..k).map(|j| if i == j { 1.0 } else { 0.0 }));
            row
        })
        .collect();

    for col in 0..k {
        let pivot = (col..k)
            .max_by(|&i, &j| aug[i][col].abs().partial_cmp(&aug[j][col].abs()).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(p) = pivot {
            aug.swap(col, p);
        }
        let pv = aug[col][col];
        if pv.abs() < 1e-14 {
            aug[col][col] += 1e-10;
        }
        let pv = aug[col][col];
        for j in 0..2 * k {
            let v = aug[col][j];
            aug[col][j] = v / pv;
        }
        for row in 0..k {
            if row != col {
                let factor = aug[row][col];
                for j in 0..2 * k {
                    let v = aug[col][j];
                    aug[row][j] -= factor * v;
                }
            }
        }
    }

    Ok((0..k).map(|i| aug[i][k..].to_vec()).collect())
}

/// Factor model log-likelihood.
fn factor_log_likelihood(
    centered: &[Vec<f64>],
    loadings: &[f64],
    psi: &[f64],
    p: usize,
    k: usize,
    n: usize,
) -> f64 {
    // Σ = Λ Λ^T + Ψ
    let mut sigma = vec![vec![0.0_f64; p]; p];
    for j1 in 0..p {
        for j2 in 0..p {
            sigma[j1][j2] = (0..k)
                .map(|l| loadings[j1 * k + l] * loadings[j2 * k + l])
                .sum::<f64>()
                + if j1 == j2 { psi[j1] } else { 0.0 };
        }
    }

    // log|Σ| via Cholesky
    let log_det = log_det_chol_slice(&sigma, p).unwrap_or(0.0);

    // Σ⁻¹ via Cholesky
    let sigma_inv = invert_sym_via_chol(&sigma, p).unwrap_or_else(|_| eye_p(p));

    let mut ll = 0.0_f64;
    for xi in centered {
        let mut quad = 0.0_f64;
        for j1 in 0..p {
            for j2 in 0..p {
                quad += xi[j1] * sigma_inv[j1][j2] * xi[j2];
            }
        }
        ll -= 0.5 * (p as f64 * (2.0 * PI).ln() + log_det + quad);
    }
    ll / n as f64
}

fn eye_p(p: usize) -> Vec<Vec<f64>> {
    (0..p).map(|i| (0..p).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect()
}

fn log_det_chol_slice(m: &[Vec<f64>], n: usize) -> Result<f64> {
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = m[i][j];
            for kk in 0..j {
                s -= l[i][kk] * l[j][kk];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(StatsError::ComputationError(
                        "Non-positive definite matrix".into(),
                    ));
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let log_det: f64 = (0..n).map(|i| l[i][i].ln()).sum::<f64>() * 2.0;
    Ok(log_det)
}

fn invert_sym_via_chol(m: &[Vec<f64>], n: usize) -> Result<Vec<Vec<f64>>> {
    // For small p, just use Gaussian elimination
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = m[i].clone();
            row.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            row
        })
        .collect();

    for col in 0..n {
        let pivot_row = (col..n)
            .max_by(|&i, &j| aug[i][col].abs().partial_cmp(&aug[j][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal));
        if let Some(p) = pivot_row {
            aug.swap(col, p);
        }
        if aug[col][col].abs() < 1e-14 {
            aug[col][col] += 1e-10;
        }
        let pv = aug[col][col];
        for j in 0..2 * n {
            let v = aug[col][j];
            aug[col][j] = v / pv;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..2 * n {
                    let v = aug[col][j];
                    aug[row][j] -= factor * v;
                }
            }
        }
    }
    Ok((0..n).map(|i| aug[i][n..].to_vec()).collect())
}

fn compute_cum_var_explained(loadings: &[f64], uniquenesses: &[f64], p: usize, k: usize) -> Vec<f64> {
    let total_var: f64 = uniquenesses.iter().sum::<f64>()
        + (0..p).map(|j| (0..k).map(|l| loadings[j * k + l].powi(2)).sum::<f64>()).sum::<f64>();

    let mut cumulative = Vec::with_capacity(k);
    let mut cum = 0.0;
    for l in 0..k {
        let factor_var: f64 = (0..p).map(|j| loadings[j * k + l].powi(2)).sum();
        cum += factor_var / total_var;
        cumulative.push(cum);
    }
    cumulative
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_factor_data(n: usize, seed: u64) -> Vec<Vec<f64>> {
        // 4 vars, 2 true factors
        // True Λ = [[1,0],[1,0],[0,1],[0,1]], Ψ = [0.5, 0.5, 0.5, 0.5]
        let mut rng_state = seed;
        let lcg = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*s >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0
        };
        (0..n)
            .map(|_| {
                let f1 = lcg(&mut rng_state);
                let f2 = lcg(&mut rng_state);
                vec![
                    f1 + 0.5 * lcg(&mut rng_state),
                    f1 + 0.5 * lcg(&mut rng_state),
                    f2 + 0.5 * lcg(&mut rng_state),
                    f2 + 0.5 * lcg(&mut rng_state),
                ]
            })
            .collect()
    }

    #[test]
    fn test_construction() {
        assert!(FactorAnalysisModel::new(4, 2).is_ok());
        assert!(FactorAnalysisModel::new(0, 2).is_err());
        assert!(FactorAnalysisModel::new(4, 0).is_err());
        assert!(FactorAnalysisModel::new(4, 4).is_err());
    }

    #[test]
    fn test_fit_em_basic() {
        let data = make_factor_data(100, 42);
        let mut model = FactorAnalysisModel::new(4, 2).unwrap();
        model.fit_em(&data, 200, 1e-6).unwrap();

        assert_eq!(model.n_obs, 100);
        assert!(model.log_likelihood.is_finite());
        assert_eq!(model.loadings.len(), 8); // 4 vars × 2 factors
        assert_eq!(model.uniquenesses.len(), 4);
        // All uniquenesses should be positive
        assert!(model.uniquenesses.iter().all(|&u| u > 0.0));
    }

    #[test]
    fn test_loading_matrix_shape() {
        let data = make_factor_data(50, 1);
        let mut model = FactorAnalysisModel::new(4, 2).unwrap();
        model.fit_em(&data, 100, 1e-5).unwrap();
        let lm = model.loading_matrix();
        assert_eq!(lm.len(), 4);
        assert!(lm.iter().all(|row| row.len() == 2));
    }

    #[test]
    fn test_factor_scores() {
        let data = make_factor_data(30, 7);
        let mut model = FactorAnalysisModel::new(4, 2).unwrap();
        model.fit_em(&data, 100, 1e-5).unwrap();
        let fs = model.factor_scores_for(0).unwrap();
        assert_eq!(fs.len(), 2);
        assert!(fs.iter().all(|&f| f.is_finite()));
        assert!(model.factor_scores_for(100).is_err());
    }

    #[test]
    fn test_variance_explained() {
        let data = make_factor_data(80, 3);
        let mut model = FactorAnalysisModel::new(4, 2).unwrap();
        model.fit_em(&data, 100, 1e-5).unwrap();
        assert_eq!(model.cumulative_variance_explained.len(), 2);
        // Cumulative should be monotone and between 0 and 1
        let cve = &model.cumulative_variance_explained;
        assert!(cve[0] >= 0.0 && cve[0] <= 1.0);
        assert!(cve[1] >= cve[0]);
        assert!(cve[1] <= 1.0 + 1e-10);
    }

    #[test]
    fn test_insufficient_data() {
        let mut model = FactorAnalysisModel::new(4, 2).unwrap();
        assert!(model.fit_em(&[], 100, 1e-6).is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let data = vec![vec![1.0, 2.0, 3.0]]; // 3 vars, but model expects 4
        let mut model = FactorAnalysisModel::new(4, 2).unwrap();
        assert!(model.fit_em(&data, 100, 1e-6).is_err());
    }
}
