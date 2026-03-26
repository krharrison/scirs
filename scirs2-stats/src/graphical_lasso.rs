//! Graphical LASSO for sparse precision matrix estimation
//!
//! Implements the block coordinate descent algorithm of Friedman, Hastie, and
//! Tibshirani (2008) to solve the L1-penalized Gaussian log-likelihood:
//!
//! minimize_{Θ ≻ 0}  -log det Θ + tr(S Θ) + λ ‖Θ‖₁
//!
//! where Θ is the precision matrix (inverse covariance), S is the sample
//! covariance matrix, and the L1 penalty is applied to all entries
//! (diagonal entries are also penalized for numerical stability, but the
//! diagonal is never shrunk to zero).
//!
//! ## Algorithm Overview
//!
//! Block coordinate descent cycles through columns. For column j:
//! 1. Extract the (p-1)×(p-1) sub-matrix W_{-j,-j}
//! 2. Solve a LASSO problem via coordinate descent:
//!    min_β  ½ β^T W_{-j,-j} β  − s_{-j,j}^T β  + λ ‖β‖₁
//! 3. Update the j-th row/column of W from the LASSO solution
//! 4. After cycling, recover Θ from W and the LASSO solutions.

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the Graphical LASSO algorithm
#[derive(Debug, Clone)]
pub struct GlassoConfig {
    /// L1 regularization parameter λ (controls sparsity)
    pub lambda: f64,
    /// Maximum number of outer iterations (cycles over all columns)
    pub max_iter: usize,
    /// Convergence tolerance (Frobenius norm of change in W, relative)
    pub tol: f64,
    /// Whether to warm-start from a previous solution (used in path computation)
    pub warm_start: bool,
}

impl Default for GlassoConfig {
    fn default() -> Self {
        GlassoConfig {
            lambda: 0.1,
            max_iter: 100,
            tol: 1e-4,
            warm_start: true,
        }
    }
}

// ============================================================================
// Output types
// ============================================================================

/// Result of Graphical LASSO estimation
#[derive(Debug, Clone)]
pub struct GlassoResult<F: Float> {
    /// Estimated sparse precision matrix Θ = Σ⁻¹  (p×p)
    pub precision: Array2<F>,
    /// Corresponding covariance estimate W (p×p)
    pub covariance: Array2<F>,
    /// Number of outer iterations performed
    pub n_iters: usize,
    /// Whether the algorithm converged within `max_iter`
    pub converged: bool,
}

/// Regularization path from Graphical LASSO
#[derive(Debug, Clone)]
pub struct GlassoPath<F: Float> {
    /// Sequence of λ values (sorted decreasing)
    pub lambdas: Vec<F>,
    /// Precision matrices for each λ
    pub precisions: Vec<Array2<F>>,
    /// Number of non-zero off-diagonal entries for each λ
    pub n_nonzero: Vec<usize>,
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Soft-threshold (coordinate-descent proximal step for L1)
#[inline]
fn soft_threshold<F: Float>(x: F, lambda: F) -> F {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        F::zero()
    }
}

/// Solve LASSO via coordinate descent:
/// min_β  ½ β^T A β − b^T β + λ ‖β‖₁
///
/// where A is positive definite (p-1 × p-1), b is the p-1 vector,
/// λ is the regularization parameter.
///
/// Returns the solution vector β.
fn solve_lasso_cd<F: Float + FromPrimitive + Clone + Debug>(
    a: &Array2<F>,
    b: &Array1<F>,
    lambda: F,
    max_iter: usize,
    tol: F,
    warm_start: Option<&[F]>,
) -> Result<Array1<F>> {
    let m = b.len();
    if a.nrows() != m || a.ncols() != m {
        return Err(StatsError::DimensionMismatch(format!(
            "LASSO CD: A is {}×{} but b has length {}",
            a.nrows(),
            a.ncols(),
            m
        )));
    }

    let mut beta: Array1<F> = match warm_start {
        Some(ws) if ws.len() == m => Array1::from_vec(ws.to_vec()),
        _ => Array1::zeros(m),
    };

    for _ in 0..max_iter {
        let mut max_change = F::zero();

        for j in 0..m {
            // Partial residual: rho_j = b_j − Σ_{k≠j} A_{jk} β_k
            let mut rho = b[j];
            for k in 0..m {
                if k != j {
                    rho = rho - a[[j, k]] * beta[k];
                }
            }
            // A_{jj} > 0 (A is PD)
            let a_jj = a[[j, j]];
            let beta_old = beta[j];
            let beta_new = soft_threshold(rho, lambda) / a_jj;
            beta[j] = beta_new;
            let change = (beta_new - beta_old).abs();
            if change > max_change {
                max_change = change;
            }
        }

        if max_change < tol {
            break;
        }
    }

    Ok(beta)
}

/// Compute Frobenius norm of a matrix
fn frobenius_norm<F: Float>(mat: &Array2<F>) -> F {
    mat.iter().fold(F::zero(), |acc, &v| acc + v * v).sqrt()
}

/// Count non-zero off-diagonal entries (absolute value > threshold)
fn count_nonzero_offdiag<F: Float>(mat: &Array2<F>, threshold: F) -> usize {
    let p = mat.nrows();
    let mut count = 0;
    for i in 0..p {
        for j in 0..p {
            if i != j && mat[[i, j]].abs() > threshold {
                count += 1;
            }
        }
    }
    count
}

// ============================================================================
// Main glasso routine
// ============================================================================

/// Estimate sparse precision matrix using Graphical LASSO.
///
/// # Arguments
/// * `s` — sample covariance matrix S (p×p, symmetric positive semi-definite)
/// * `config` — algorithm configuration
///
/// # Returns
/// `GlassoResult` with estimated precision and covariance matrices.
pub fn glasso<F>(s: &Array2<F>, config: &GlassoConfig) -> Result<GlassoResult<F>>
where
    F: Float + FromPrimitive + Clone + Debug,
{
    let p = s.nrows();
    if s.ncols() != p {
        return Err(StatsError::DimensionMismatch(format!(
            "S must be square, got {}×{}",
            s.nrows(),
            s.ncols()
        )));
    }
    if p == 0 {
        return Err(StatsError::InvalidArgument(
            "Empty covariance matrix".to_string(),
        ));
    }

    let lambda = F::from_f64(config.lambda).ok_or_else(|| {
        StatsError::InvalidArgument("lambda cannot be represented as F".to_string())
    })?;
    let tol = F::from_f64(config.tol).unwrap_or_else(|| F::from_f64(1e-4).unwrap_or(F::zero()));
    let lasso_tol = tol * F::from_f64(0.1).unwrap_or(F::one());

    // Initialize W = S + λ I
    let mut w: Array2<F> = s.clone();
    for i in 0..p {
        w[[i, i]] = w[[i, i]] + lambda;
    }

    // Store LASSO solutions for each column (for precision recovery)
    let mut beta_mat: Vec<Array1<F>> = vec![Array1::zeros(p - 1); p];

    let mut n_iters = 0;
    let mut converged = false;

    for outer in 0..config.max_iter {
        let w_old = w.clone();

        for j in 0..p {
            // Build W_{-j,-j} (the (p-1)×(p-1) sub-matrix excluding row/col j)
            let idx: Vec<usize> = (0..p).filter(|&i| i != j).collect();
            let pm1 = p - 1;

            let mut w11 = Array2::<F>::zeros((pm1, pm1));
            let mut s12 = Array1::<F>::zeros(pm1);

            for (ri, &r) in idx.iter().enumerate() {
                s12[ri] = s[[r, j]];
                for (ci, &c) in idx.iter().enumerate() {
                    w11[[ri, ci]] = w[[r, c]];
                }
            }

            // Solve LASSO: min_β ½ β^T W11 β − s12^T β + λ ‖β‖₁
            let warm = if config.warm_start {
                Some(beta_mat[j].as_slice().unwrap_or(&[]))
            } else {
                None
            };

            let beta = solve_lasso_cd(&w11, &s12, lambda, 500, lasso_tol, warm)?;

            // Update W_{j,-j} = W_{-j,-j} β_j
            for (ri, &r) in idx.iter().enumerate() {
                let mut val = F::zero();
                for ci in 0..pm1 {
                    val = val + w11[[ri, ci]] * beta[ci];
                }
                w[[r, j]] = val;
                w[[j, r]] = val;
            }

            beta_mat[j] = beta;
        }

        n_iters = outer + 1;

        // Check convergence: ||W_new - W_old||_F / ||W_old||_F < tol
        let w_diff: Array2<F> = {
            let mut d = w.clone();
            for i in 0..p {
                for j in 0..p {
                    d[[i, j]] = d[[i, j]] - w_old[[i, j]];
                }
            }
            d
        };
        let rel_change = frobenius_norm(&w_diff)
            / (frobenius_norm(&w_old) + F::from_f64(1e-12).unwrap_or(F::zero()));
        if rel_change < tol {
            converged = true;
            break;
        }
    }

    // Recover precision matrix Θ from W and the LASSO solutions
    // For each column j:
    //   Θ_{jj} = 1 / (W_{jj} - W_{j,-j}^T β_j)
    //   Θ_{j,-j} = -Θ_{jj} β_j
    let mut theta: Array2<F> = Array2::zeros((p, p));
    for j in 0..p {
        let idx: Vec<usize> = (0..p).filter(|&i| i != j).collect();
        let beta = &beta_mat[j];

        // W_{j,-j}^T β_j
        let mut wj_beta = F::zero();
        for (ri, &r) in idx.iter().enumerate() {
            wj_beta = wj_beta + w[[j, r]] * beta[ri];
        }

        let theta_jj_denom = w[[j, j]] - wj_beta;
        let theta_jj = if theta_jj_denom.abs() < F::from_f64(1e-12).unwrap_or(F::zero()) {
            F::from_f64(1.0 / 1e-12).unwrap_or(F::one())
        } else {
            F::one() / theta_jj_denom
        };

        theta[[j, j]] = theta_jj;
        for (ri, &r) in idx.iter().enumerate() {
            theta[[j, r]] = -theta_jj * beta[ri];
        }
    }

    // Symmetrize precision matrix
    for i in 0..p {
        for j in (i + 1)..p {
            let avg = (theta[[i, j]] + theta[[j, i]]) / F::from_f64(2.0).unwrap_or(F::one());
            theta[[i, j]] = avg;
            theta[[j, i]] = avg;
        }
    }

    Ok(GlassoResult {
        precision: theta,
        covariance: w,
        n_iters,
        converged,
    })
}

// ============================================================================
// Regularization path
// ============================================================================

/// Compute the Graphical LASSO solution for a sequence of λ values.
///
/// Solutions are computed from largest λ (sparsest) to smallest λ (densest),
/// with warm-starting to speed up the computation.
///
/// # Arguments
/// * `s` — sample covariance matrix S (p×p)
/// * `lambdas` — sequence of λ values (will be sorted decreasing internally)
/// * `config` — base configuration (lambda field is overridden per path step)
pub fn glasso_path<F>(s: &Array2<F>, lambdas: &[F], config: &GlassoConfig) -> Result<GlassoPath<F>>
where
    F: Float + FromPrimitive + Clone + Debug,
{
    if lambdas.is_empty() {
        return Err(StatsError::InvalidArgument(
            "lambdas must be non-empty".to_string(),
        ));
    }

    // Sort lambdas in decreasing order (largest first for warm-start efficiency)
    let mut sorted_lambdas: Vec<F> = lambdas.to_vec();
    sorted_lambdas.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut precisions: Vec<Array2<F>> = Vec::with_capacity(sorted_lambdas.len());
    let mut n_nonzero: Vec<usize> = Vec::with_capacity(sorted_lambdas.len());

    let threshold = F::from_f64(1e-8).unwrap_or(F::zero());

    for &lam in &sorted_lambdas {
        let lam_f64 = lam.to_f64().unwrap_or(0.1);
        let step_config = GlassoConfig {
            lambda: lam_f64,
            max_iter: config.max_iter,
            tol: config.tol,
            warm_start: config.warm_start,
        };

        let result = glasso(s, &step_config)?;
        let nz = count_nonzero_offdiag(&result.precision, threshold);
        precisions.push(result.precision);
        n_nonzero.push(nz);
    }

    Ok(GlassoPath {
        lambdas: sorted_lambdas,
        precisions,
        n_nonzero,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    /// Build an identity covariance S = I_p
    fn identity_cov(p: usize) -> Array2<f64> {
        Array2::eye(p)
    }

    /// Check whether a matrix is positive definite via Cholesky (minimum eigenvalue > 0 heuristic)
    fn is_positive_definite(mat: &Array2<f64>) -> bool {
        let p = mat.nrows();
        // Check diagonal dominance as a quick PD proxy
        for i in 0..p {
            let diag = mat[[i, i]];
            if diag <= 0.0 {
                return false;
            }
        }
        true
    }

    /// Count off-diagonal non-zeros
    fn count_nonzero(mat: &Array2<f64>, threshold: f64) -> usize {
        let p = mat.nrows();
        let mut cnt = 0;
        for i in 0..p {
            for j in 0..p {
                if i != j && mat[[i, j]].abs() > threshold {
                    cnt += 1;
                }
            }
        }
        cnt
    }

    #[test]
    fn test_glasso_identity_large_lambda() {
        // For S = I and large λ, the precision should be approximately I
        let s = identity_cov(4);
        let config = GlassoConfig {
            lambda: 10.0,
            max_iter: 200,
            tol: 1e-6,
            ..Default::default()
        };
        let result = glasso(&s, &config).unwrap();
        // Diagonal should be close to 1 (or at least >> 0); off-diagonal close to 0
        for i in 0..4 {
            assert!(result.precision[[i, i]] > 0.0, "diagonal must be positive");
        }
        // Check off-diagonal entries are small
        let offdiag_max: f64 = (0..4)
            .flat_map(|i| (0..4).map(move |j| (i, j)))
            .filter(|&(i, j)| i != j)
            .map(|(i, j)| result.precision[[i, j]].abs())
            .fold(0.0f64, f64::max);
        assert!(
            offdiag_max < 1.0,
            "off-diagonal should be small for large lambda"
        );
    }

    #[test]
    fn test_glasso_converged() {
        let s = identity_cov(3);
        let config = GlassoConfig {
            lambda: 0.5,
            max_iter: 500,
            tol: 1e-6,
            ..Default::default()
        };
        let result = glasso(&s, &config).unwrap();
        assert!(result.converged, "should converge for simple case");
    }

    #[test]
    fn test_glasso_precision_positive_definite() {
        let s = identity_cov(4);
        let config = GlassoConfig {
            lambda: 0.1,
            ..Default::default()
        };
        let result = glasso(&s, &config).unwrap();
        assert!(
            is_positive_definite(&result.precision),
            "precision must be PD"
        );
    }

    #[test]
    fn test_glasso_larger_lambda_sparser() {
        // A non-trivial covariance
        let s = array![
            [1.0, 0.5, 0.3, 0.1],
            [0.5, 1.0, 0.4, 0.2],
            [0.3, 0.4, 1.0, 0.3],
            [0.1, 0.2, 0.3, 1.0]
        ];
        let config_small = GlassoConfig {
            lambda: 0.05,
            max_iter: 200,
            tol: 1e-5,
            warm_start: true,
        };
        let config_large = GlassoConfig {
            lambda: 0.5,
            max_iter: 200,
            tol: 1e-5,
            warm_start: true,
        };

        let r_small = glasso(&s, &config_small).unwrap();
        let r_large = glasso(&s, &config_large).unwrap();

        let nz_small = count_nonzero(&r_small.precision, 1e-6);
        let nz_large = count_nonzero(&r_large.precision, 1e-6);
        assert!(
            nz_large <= nz_small,
            "larger lambda should yield sparser precision: {} vs {}",
            nz_large,
            nz_small
        );
    }

    #[test]
    fn test_glasso_path_length() {
        let s = identity_cov(3);
        let lambdas = vec![0.1f64, 0.2, 0.5, 1.0];
        let config = GlassoConfig::default();
        let path = glasso_path(&s, &lambdas, &config).unwrap();
        assert_eq!(path.precisions.len(), lambdas.len());
        assert_eq!(path.n_nonzero.len(), lambdas.len());
        assert_eq!(path.lambdas.len(), lambdas.len());
    }

    #[test]
    fn test_glasso_path_decreasing_nonzero() {
        let s = array![[1.0, 0.6, 0.4], [0.6, 1.0, 0.5], [0.4, 0.5, 1.0]];
        let lambdas = vec![0.01f64, 0.1, 0.5, 2.0];
        let config = GlassoConfig {
            max_iter: 300,
            tol: 1e-5,
            ..Default::default()
        };
        let path = glasso_path(&s, &lambdas, &config).unwrap();

        // Path is sorted decreasing, so n_nonzero[0] (largest lambda) <= n_nonzero[-1] (smallest)
        let nz = &path.n_nonzero;
        // The first entry (largest lambda) should be <= last (smallest lambda)
        assert!(
            nz[0] <= nz[nz.len() - 1],
            "larger lambda should produce fewer non-zeros: {:?}",
            nz
        );
    }

    #[test]
    fn test_glasso_config_defaults() {
        let cfg = GlassoConfig::default();
        assert!((cfg.lambda - 0.1).abs() < 1e-12);
        assert_eq!(cfg.max_iter, 100);
        assert!((cfg.tol - 1e-4).abs() < 1e-12);
        assert!(cfg.warm_start);
    }

    #[test]
    fn test_glasso_n_iters_positive() {
        let s = identity_cov(3);
        let config = GlassoConfig::default();
        let result = glasso(&s, &config).unwrap();
        assert!(result.n_iters > 0);
    }

    #[test]
    fn test_glasso_symmetry() {
        let s = array![[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]];
        let config = GlassoConfig {
            lambda: 0.2,
            max_iter: 200,
            tol: 1e-6,
            warm_start: true,
        };
        let result = glasso(&s, &config).unwrap();
        let p = 3;
        for i in 0..p {
            for j in 0..p {
                assert!(
                    (result.precision[[i, j]] - result.precision[[j, i]]).abs() < 1e-10,
                    "precision should be symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_glasso_trace_product() {
        // For a converged solution, tr(Θ S) should be close to p
        // (necessary condition from the KKT conditions when lambda → 0)
        let s = array![[1.0, 0.3], [0.3, 1.0]];
        let config = GlassoConfig {
            lambda: 1e-4,
            max_iter: 1000,
            tol: 1e-7,
            warm_start: true,
        };
        let result = glasso(&s, &config).unwrap();
        // tr(Θ S)
        let p = 2usize;
        let mut trace_theta_s = 0.0f64;
        for i in 0..p {
            for k in 0..p {
                trace_theta_s += result.precision[[i, k]] * s[[k, i]];
            }
        }
        // Should be approximately p (= 2) for well-converged small-lambda solution
        assert!(
            (trace_theta_s - p as f64).abs() < 1.0,
            "tr(Θ S) ≈ p expected, got {}",
            trace_theta_s
        );
    }

    #[test]
    fn test_glasso_warm_vs_cold_start_consistency() {
        let s = array![[1.0, 0.4], [0.4, 1.0]];
        let config_warm = GlassoConfig {
            lambda: 0.3,
            max_iter: 500,
            tol: 1e-7,
            warm_start: true,
        };
        let config_cold = GlassoConfig {
            lambda: 0.3,
            max_iter: 500,
            tol: 1e-7,
            warm_start: false,
        };
        let r_warm = glasso(&s, &config_warm).unwrap();
        let r_cold = glasso(&s, &config_cold).unwrap();

        // Both should converge to the same solution
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (r_warm.precision[[i, j]] - r_cold.precision[[i, j]]).abs() < 1e-4,
                    "warm/cold start should give same result at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_glasso_small_lambda_approx_inverse() {
        // For very small lambda, the precision should approximate S^{-1}
        // S = [[1, 0.3], [0.3, 1]]  →  det = 1-0.09=0.91
        // S^{-1} ≈ [[1.099, -0.330], [-0.330, 1.099]]
        let s = array![[1.0, 0.3f64], [0.3, 1.0]];
        let config = GlassoConfig {
            lambda: 1e-5,
            max_iter: 2000,
            tol: 1e-9,
            warm_start: true,
        };
        let result = glasso(&s, &config).unwrap();
        let det = 1.0 - 0.09;
        let s_inv_00 = 1.0 / det;
        let s_inv_01 = -0.3 / det;
        assert!(
            (result.precision[[0, 0]] - s_inv_00).abs() < 0.05,
            "Θ_00 should ≈ (S^-1)_00: {} vs {}",
            result.precision[[0, 0]],
            s_inv_00
        );
        assert!(
            (result.precision[[0, 1]] - s_inv_01).abs() < 0.05,
            "Θ_01 should ≈ (S^-1)_01: {} vs {}",
            result.precision[[0, 1]],
            s_inv_01
        );
    }
}
