//! CLIME (Constrained L1 Minimization for Inverse Matrix Estimation)
//! (Cai, Liu & Luo, 2011)
//!
//! Estimates a sparse precision matrix by solving:
//!   min ||Omega||_1  subject to  ||S * Omega - I||_inf <= lambda
//!
//! where S is the sample covariance matrix, Omega is the precision matrix,
//! and lambda controls the constraint tolerance (larger lambda = sparser).
//!
//! The problem decomposes into p independent column-wise subproblems,
//! each solved via a Dantzig-selector-like coordinate descent approach.

use crate::error::{StatsError, StatsResult};

/// Configuration for the CLIME estimator
#[derive(Debug, Clone)]
pub struct CLIMEConfig {
    /// Constraint tolerance parameter (must be > 0)
    /// Controls ||S * Omega - I||_inf <= lambda
    pub lambda: f64,
    /// Maximum number of iterations for the coordinate descent solver
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to symmetrize the result via min(|Omega_ij|, |Omega_ji|)
    pub symmetrize: bool,
}

impl Default for CLIMEConfig {
    fn default() -> Self {
        CLIMEConfig {
            lambda: 0.1,
            max_iter: 500,
            tolerance: 1e-6,
            symmetrize: true,
        }
    }
}

impl CLIMEConfig {
    /// Create a new CLIME configuration with the given lambda
    pub fn new(lambda: f64) -> StatsResult<Self> {
        if lambda <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "lambda must be positive".to_string(),
            ));
        }
        Ok(CLIMEConfig {
            lambda,
            ..Default::default()
        })
    }

    /// Set the maximum number of iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set whether to symmetrize the result
    pub fn with_symmetrize(mut self, symmetrize: bool) -> Self {
        self.symmetrize = symmetrize;
        self
    }
}

/// Result of the CLIME estimation
#[derive(Debug, Clone)]
pub struct CLIMEResult {
    /// Estimated precision matrix (p x p)
    pub precision_matrix: Vec<Vec<f64>>,
    /// Whether the algorithm converged for all columns
    pub converged: bool,
    /// Total iterations across all column solves
    pub iterations: usize,
    /// Number of non-zero off-diagonal entries (edges)
    pub n_edges: usize,
    /// Dimension of the matrix
    pub dimension: usize,
}

impl CLIMEResult {
    /// Return the density (fraction of non-zero off-diagonal entries)
    pub fn density(&self) -> f64 {
        let p = self.dimension;
        let max_edges = p * (p - 1) / 2;
        if max_edges == 0 {
            return 0.0;
        }
        self.n_edges as f64 / max_edges as f64
    }

    /// Return the adjacency (sparsity) pattern
    pub fn adjacency(&self, threshold: f64) -> Vec<Vec<bool>> {
        let p = self.dimension;
        let mut adj = vec![vec![false; p]; p];
        for i in 0..p {
            for j in 0..p {
                if i != j && self.precision_matrix[i][j].abs() > threshold {
                    adj[i][j] = true;
                }
            }
        }
        adj
    }
}

/// Solve the CLIME problem for estimating a sparse precision matrix.
///
/// Given a sample covariance matrix S (p x p), solves p independent
/// column-wise optimization problems:
///
///   For each column j: min ||omega_j||_1  subject to  ||S * omega_j - e_j||_inf <= lambda
///
/// where e_j is the j-th standard basis vector.
///
/// # Arguments
/// * `sample_covariance` - p x p sample covariance matrix (must be symmetric)
/// * `config` - CLIME configuration
///
/// # Returns
/// A `CLIMEResult` containing the estimated precision matrix
pub fn clime(
    sample_covariance: &[Vec<f64>],
    config: &CLIMEConfig,
) -> StatsResult<CLIMEResult> {
    let p = sample_covariance.len();
    if p == 0 {
        return Err(StatsError::InvalidArgument(
            "Covariance matrix must be non-empty".to_string(),
        ));
    }

    // Validate square and symmetric
    for i in 0..p {
        if sample_covariance[i].len() != p {
            return Err(StatsError::DimensionMismatch(format!(
                "Row {} has length {} but expected {}",
                i,
                sample_covariance[i].len(),
                p
            )));
        }
    }

    for i in 0..p {
        for j in (i + 1)..p {
            if (sample_covariance[i][j] - sample_covariance[j][i]).abs() > 1e-10 {
                return Err(StatsError::InvalidArgument(
                    "Covariance matrix must be symmetric".to_string(),
                ));
            }
        }
    }

    // Check positive diagonal
    for i in 0..p {
        if sample_covariance[i][i] <= 0.0 {
            return Err(StatsError::InvalidArgument(format!(
                "Diagonal entry S[{},{}] must be positive, got {}",
                i, i, sample_covariance[i][i]
            )));
        }
    }

    // Special case: p == 1
    if p == 1 {
        let omega_val = 1.0 / sample_covariance[0][0];
        return Ok(CLIMEResult {
            precision_matrix: vec![vec![omega_val]],
            converged: true,
            iterations: 0,
            n_edges: 0,
            dimension: 1,
        });
    }

    // Solve p independent column problems
    let mut omega = vec![vec![0.0; p]; p];
    let mut all_converged = true;
    let mut total_iterations = 0;

    for j in 0..p {
        let (col_omega, col_converged, col_iters) =
            solve_clime_column(sample_covariance, j, config)?;

        for i in 0..p {
            omega[i][j] = col_omega[i];
        }
        if !col_converged {
            all_converged = false;
        }
        total_iterations += col_iters;
    }

    // Symmetrize if requested: Omega_sym[i][j] = sign * min(|Omega[i][j]|, |Omega[j][i]|)
    if config.symmetrize {
        let omega_unsym = omega.clone();
        for i in 0..p {
            for j in (i + 1)..p {
                let val_ij = omega_unsym[i][j];
                let val_ji = omega_unsym[j][i];
                // Pick the entry with smaller absolute value, preserving sign
                let sym_val = if val_ij.abs() <= val_ji.abs() {
                    val_ij
                } else {
                    val_ji
                };
                omega[i][j] = sym_val;
                omega[j][i] = sym_val;
            }
        }
    }

    // Count edges
    let mut n_edges = 0;
    for i in 0..p {
        for j in (i + 1)..p {
            if omega[i][j].abs() > 1e-10 {
                n_edges += 1;
            }
        }
    }

    Ok(CLIMEResult {
        precision_matrix: omega,
        converged: all_converged,
        iterations: total_iterations,
        n_edges,
        dimension: p,
    })
}

/// Solve one column of the CLIME problem using Dantzig selector coordinate descent.
///
/// min ||omega||_1  subject to  ||S * omega - e_j||_inf <= lambda
///
/// We reformulate this as an unconstrained penalized problem using the
/// augmented Lagrangian / penalty approach:
///   min ||omega||_1 + (mu/2) * sum_k max(0, |S_k*omega - e_jk| - lambda)^2
///
/// and solve via coordinate descent, increasing mu to enforce constraints.
fn solve_clime_column(
    s: &[Vec<f64>],
    j: usize,
    config: &CLIMEConfig,
) -> StatsResult<(Vec<f64>, bool, usize)> {
    let p = s.len();
    let lambda = config.lambda;

    // Initialize omega_j = e_j / S[j][j]
    let mut omega = vec![0.0; p];
    omega[j] = 1.0 / s[j][j];

    let mut converged = false;
    let mut total_iters = 0;

    // Use a penalty method: iterate with increasing mu
    let mut mu = 10.0;
    let mu_factor = 2.0;
    let max_outer = 10;

    for _outer in 0..max_outer {
        // Inner coordinate descent loop
        let inner_max = config.max_iter / max_outer;
        for _inner in 0..inner_max {
            let mut max_change: f64 = 0.0;
            total_iters += 1;

            for k in 0..p {
                let s_kk = s[k][k];
                if s_kk.abs() < 1e-15 {
                    continue;
                }

                let old_val = omega[k];

                // Compute (S * omega)_k
                let s_omega_k = dot_row(s, k, &omega);
                let target = if k == j { 1.0 } else { 0.0 };
                let residual = s_omega_k - target;

                // Gradient of penalty: mu * s_kk * clamp(residual, -lambda, lambda) adjustment
                // The update tries to: (1) minimize |omega_k| and (2) push residual into [-lambda, lambda]
                let penalty_grad = if residual > lambda {
                    mu * s_kk * (residual - lambda)
                } else if residual < -lambda {
                    mu * s_kk * (residual + lambda)
                } else {
                    0.0
                };

                // Denominator for the update
                let denom = 1.0 + mu * s_kk * s_kk;

                // Numerator: old_val * denom - penalty_grad, then soft-threshold
                let numerator = old_val * denom - penalty_grad;
                let new_val = soft_threshold(numerator, 1.0) / denom;

                omega[k] = new_val;

                let change = (new_val - old_val).abs();
                if change > max_change {
                    max_change = change;
                }
            }

            if max_change < config.tolerance {
                // Check if constraints are satisfied
                let max_violation = compute_max_violation(s, &omega, j, lambda);
                if max_violation <= lambda * 0.1 + config.tolerance {
                    converged = true;
                    return Ok((omega, converged, total_iters));
                }
                break; // inner converged but constraints not met; increase mu
            }
        }

        // Check overall convergence
        let max_violation = compute_max_violation(s, &omega, j, lambda);
        if max_violation <= config.tolerance {
            converged = true;
            return Ok((omega, converged, total_iters));
        }

        mu *= mu_factor;
    }

    Ok((omega, converged, total_iters))
}

/// Compute the maximum constraint violation: max_k ||(S*omega)_k - e_jk| - lambda||_+
fn compute_max_violation(s: &[Vec<f64>], omega: &[f64], j: usize, lambda: f64) -> f64 {
    let p = s.len();
    let mut max_viol: f64 = 0.0;
    for k in 0..p {
        let s_omega_k = dot_row(s, k, omega);
        let target = if k == j { 1.0 } else { 0.0 };
        let viol = (s_omega_k - target).abs() - lambda;
        if viol > max_viol {
            max_viol = viol;
        }
    }
    max_viol.max(0.0)
}

/// Compute dot product of row i of matrix S with vector v
#[inline]
fn dot_row(s: &[Vec<f64>], i: usize, v: &[f64]) -> f64 {
    let mut result = 0.0;
    for (k, val) in v.iter().enumerate() {
        result += s[i][k] * val;
    }
    result
}

/// Soft-thresholding operator: sign(x) * max(|x| - t, 0)
#[inline]
fn soft_threshold(x: f64, t: f64) -> f64 {
    if x > t {
        x - t
    } else if x < -t {
        x + t
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create identity covariance matrix
    fn identity_covariance(p: usize) -> Vec<Vec<f64>> {
        let mut s = vec![vec![0.0; p]; p];
        for i in 0..p {
            s[i][i] = 1.0;
        }
        s
    }

    /// Helper: create a known covariance with tridiagonal precision
    fn tridiagonal_covariance(p: usize, rho: f64) -> Vec<Vec<f64>> {
        // AR(1) covariance: S[i][j] = rho^|i-j|
        let mut s = vec![vec![0.0; p]; p];
        for i in 0..p {
            for j in 0..p {
                let diff = if i > j { i - j } else { j - i };
                s[i][j] = rho.powi(diff as i32);
            }
        }
        s
    }

    #[test]
    fn test_clime_identity_covariance() {
        let s = identity_covariance(3);
        let config = CLIMEConfig::new(0.1)
            .expect("config creation failed")
            .with_max_iter(2000);
        let result = clime(&s, &config).expect("CLIME failed");

        assert_eq!(result.dimension, 3);

        // Precision should be close to identity
        for i in 0..3 {
            assert!(
                (result.precision_matrix[i][i] - 1.0).abs() < 0.3,
                "Diagonal should be near 1.0, got {}",
                result.precision_matrix[i][i]
            );
            for j in 0..3 {
                if i != j {
                    assert!(
                        result.precision_matrix[i][j].abs() < 0.3,
                        "Off-diagonal should be near 0, got {}",
                        result.precision_matrix[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn test_clime_single_variable() {
        let s = vec![vec![4.0]];
        let config = CLIMEConfig::new(0.1).expect("config creation failed");
        let result = clime(&s, &config).expect("CLIME failed");

        assert!(result.converged);
        assert!((result.precision_matrix[0][0] - 0.25).abs() < 1e-10);
        assert_eq!(result.n_edges, 0);
    }

    #[test]
    fn test_clime_symmetry() {
        let s = tridiagonal_covariance(4, 0.5);
        let config = CLIMEConfig::new(0.2)
            .expect("config creation failed")
            .with_symmetrize(true);
        let result = clime(&s, &config).expect("CLIME failed");

        let p = result.dimension;
        for i in 0..p {
            for j in 0..p {
                assert!(
                    (result.precision_matrix[i][j] - result.precision_matrix[j][i]).abs() < 1e-10,
                    "Precision matrix should be symmetric: [{},{}]={} vs [{},{}]={}",
                    i, j, result.precision_matrix[i][j],
                    j, i, result.precision_matrix[j][i]
                );
            }
        }
    }

    #[test]
    fn test_clime_unsymmetrized() {
        let s = tridiagonal_covariance(3, 0.5);
        let config = CLIMEConfig::new(0.1)
            .expect("config creation failed")
            .with_symmetrize(false);
        let result = clime(&s, &config).expect("CLIME failed");

        // Just verify it runs; unsymmetrized result may not be symmetric
        assert_eq!(result.dimension, 3);
    }

    #[test]
    fn test_clime_sparsity_increases_with_lambda() {
        let s = tridiagonal_covariance(4, 0.5);

        let config_small = CLIMEConfig::new(0.05).expect("config creation failed");
        let result_small = clime(&s, &config_small).expect("CLIME small lambda failed");

        let config_large = CLIMEConfig::new(0.5).expect("config creation failed");
        let result_large = clime(&s, &config_large).expect("CLIME large lambda failed");

        // Larger lambda should give sparser result
        assert!(
            result_large.n_edges <= result_small.n_edges,
            "Larger lambda should give sparser result: {} vs {}",
            result_large.n_edges,
            result_small.n_edges
        );
    }

    #[test]
    fn test_clime_density() {
        let s = identity_covariance(4);
        let config = CLIMEConfig::new(0.5).expect("config creation failed");
        let result = clime(&s, &config).expect("CLIME failed");

        let density = result.density();
        assert!(density >= 0.0 && density <= 1.0);
    }

    #[test]
    fn test_clime_adjacency() {
        let s = tridiagonal_covariance(3, 0.5);
        let config = CLIMEConfig::new(0.1).expect("config creation failed");
        let result = clime(&s, &config).expect("CLIME failed");

        let adj = result.adjacency(0.01);
        assert_eq!(adj.len(), 3);

        // Adjacency should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(adj[i][j], adj[j][i]);
            }
        }
    }

    #[test]
    fn test_clime_error_non_square() {
        let s = vec![vec![1.0, 0.5], vec![0.5]]; // not square
        let config = CLIMEConfig::new(0.1).expect("config creation failed");
        let result = clime(&s, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_clime_error_asymmetric() {
        let s = vec![vec![1.0, 0.5], vec![0.6, 1.0]]; // asymmetric
        let config = CLIMEConfig::new(0.1).expect("config creation failed");
        let result = clime(&s, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_clime_error_non_positive_diagonal() {
        let s = vec![vec![0.0, 0.0], vec![0.0, 1.0]]; // zero diagonal
        let config = CLIMEConfig::new(0.1).expect("config creation failed");
        let result = clime(&s, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_clime_error_empty() {
        let s: Vec<Vec<f64>> = vec![];
        let config = CLIMEConfig::new(0.1).expect("config creation failed");
        let result = clime(&s, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_clime_error_invalid_lambda() {
        let result = CLIMEConfig::new(-0.1);
        assert!(result.is_err());

        let result = CLIMEConfig::new(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_clime_config_builder() {
        let config = CLIMEConfig::new(0.2)
            .expect("config creation failed")
            .with_max_iter(1000)
            .with_tolerance(1e-8)
            .with_symmetrize(false);

        assert!((config.lambda - 0.2).abs() < 1e-15);
        assert_eq!(config.max_iter, 1000);
        assert!((config.tolerance - 1e-8).abs() < 1e-15);
        assert!(!config.symmetrize);
    }
}
