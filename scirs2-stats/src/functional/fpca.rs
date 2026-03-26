//! Functional Principal Component Analysis (fPCA).
//!
//! Performs PCA on functional data by:
//! 1. Smoothing each curve to basis coefficients
//! 2. Estimating the covariance function
//! 3. Eigendecomposition to obtain eigenfunctions and scores
//!
//! This is the functional analogue of multivariate PCA, where each observation
//! is an entire curve rather than a finite-dimensional vector.

use scirs2_core::ndarray::{Array1, Array2, Axis};

use super::basis::{compute_basis_coefficients, evaluate_basis, gcv_select_lambda};
use super::types::{FPCAResult, FunctionalConfig, FunctionalData};
use crate::error::{StatsError, StatsResult};

/// Perform functional PCA on a set of curves.
///
/// The algorithm:
/// 1. Evaluate basis functions on the data grid
/// 2. Smooth each curve to obtain basis coefficients (using penalized LS)
/// 3. Compute the covariance matrix of the smoothed curves on the grid
/// 4. Perform eigendecomposition of the discretized covariance operator
/// 5. Extract eigenfunctions, eigenvalues, and scores
///
/// # Arguments
/// * `data` - Functional data (n curves on a common grid)
/// * `config` - Configuration: basis type, smoothing parameter, number of components
///
/// # Returns
/// An `FPCAResult` with eigenvalues, eigenfunctions, scores, and variance explained.
///
/// # Errors
/// Returns an error if basis evaluation, smoothing, or eigendecomposition fails.
pub fn functional_pca(data: &FunctionalData, config: &FunctionalConfig) -> StatsResult<FPCAResult> {
    let n_curves = data.n_curves();
    let n_grid = data.n_grid();
    let n_components = config.n_components.min(n_curves).min(n_grid);

    if n_components == 0 {
        return Err(StatsError::InvalidArgument(
            "n_components must be at least 1".to_string(),
        ));
    }

    // Step 1: Evaluate basis
    let phi = evaluate_basis(&data.grid, &config.basis)?;

    // Step 2: Determine smoothing parameter
    let lambda = match config.smoothing_param {
        Some(lam) => lam,
        None => {
            // Use GCV on the first curve as a representative
            gcv_select_lambda(&data.observations[0], &data.grid, &config.basis)?
        }
    };

    // Step 3: Smooth each curve and get smoothed values on the grid
    let mut smoothed = Array2::<f64>::zeros((n_curves, n_grid));
    for (i, obs) in data.observations.iter().enumerate() {
        let coeffs = compute_basis_coefficients(obs, &phi, lambda)?;
        let fitted = phi.dot(&coeffs);
        for j in 0..n_grid {
            smoothed[[i, j]] = fitted[j];
        }
    }

    // Step 4: Compute the mean function
    let mean_func = smoothed.mean_axis(Axis(0)).ok_or_else(|| {
        StatsError::ComputationError("Failed to compute mean function".to_string())
    })?;

    // Step 5: Center the curves
    let mut centered = Array2::<f64>::zeros((n_curves, n_grid));
    for i in 0..n_curves {
        for j in 0..n_grid {
            centered[[i, j]] = smoothed[[i, j]] - mean_func[j];
        }
    }

    // Step 6: Compute the discretized covariance matrix (grid x grid)
    // C(s_j, s_k) = (1/(n-1)) sum_i (x_i(s_j) - mean(s_j))(x_i(s_k) - mean(s_k))
    // For numerical integration, weight by grid spacing
    let dt = compute_grid_spacing(&data.grid);

    // Covariance matrix on the grid
    let n_minus_1 = if n_curves > 1 {
        (n_curves - 1) as f64
    } else {
        1.0
    };
    let cov_matrix = centered.t().dot(&centered) / n_minus_1;

    // Weight by sqrt(dt) for proper L2 inner product
    // The eigenvalue problem is: integral C(s,t) phi(t) dt = lambda phi(s)
    // Discretized: C * diag(dt) * phi = lambda * phi
    // Equivalent symmetric problem: diag(sqrt(dt)) * C * diag(sqrt(dt)) * psi = lambda * psi
    let sqrt_dt: Array1<f64> = dt.iter().map(|d| d.sqrt()).collect::<Vec<_>>().into();
    let mut weighted_cov = Array2::<f64>::zeros((n_grid, n_grid));
    for i in 0..n_grid {
        for j in 0..n_grid {
            weighted_cov[[i, j]] = sqrt_dt[i] * cov_matrix[[i, j]] * sqrt_dt[j];
        }
    }

    // Step 7: Eigendecomposition using power iteration with deflation
    let (eigenvalues, eigenvectors) = symmetric_eigen_decomposition(&weighted_cov, n_components)?;

    // Transform eigenvectors back: phi = diag(1/sqrt(dt)) * psi
    let mut eigenfunctions = Array2::<f64>::zeros((n_components, n_grid));
    for k in 0..n_components {
        for j in 0..n_grid {
            let inv_sqrt = if sqrt_dt[j].abs() > 1e-14 {
                1.0 / sqrt_dt[j]
            } else {
                0.0
            };
            eigenfunctions[[k, j]] = eigenvectors[[j, k]] * inv_sqrt;
        }
        // Normalize the eigenfunction in L2(grid)
        let norm_sq: f64 = (0..n_grid)
            .map(|j| eigenfunctions[[k, j]].powi(2) * dt[j])
            .sum();
        let norm = norm_sq.sqrt();
        if norm > 1e-14 {
            for j in 0..n_grid {
                eigenfunctions[[k, j]] /= norm;
            }
        }
    }

    // Step 8: Compute scores by projecting centered curves onto eigenfunctions
    // score_{ik} = integral (x_i(t) - mean(t)) * phi_k(t) dt
    let mut scores = Array2::<f64>::zeros((n_curves, n_components));
    for i in 0..n_curves {
        for k in 0..n_components {
            let mut score = 0.0;
            for j in 0..n_grid {
                score += centered[[i, j]] * eigenfunctions[[k, j]] * dt[j];
            }
            scores[[i, k]] = score;
        }
    }

    // Step 9: Compute variance explained
    let total_variance: f64 = eigenvalues.iter().sum::<f64>();
    let variance_explained = if total_variance > 1e-14 {
        &eigenvalues / total_variance
    } else {
        Array1::<f64>::zeros(n_components)
    };

    Ok(FPCAResult {
        eigenvalues,
        eigenfunctions,
        scores,
        variance_explained,
        grid: data.grid.clone(),
    })
}

/// Reconstruct curves from fPCA scores using a truncated number of components.
///
/// Reconstruction: x_hat_i(t) = mean(t) + sum_{k=1}^{K} score_{ik} * phi_k(t)
///
/// Note: the mean function is not stored in `FPCAResult`; this reconstruction
/// is relative to the mean (i.e., the centered curves). To get the full
/// reconstruction, the caller should add the mean function back.
///
/// # Arguments
/// * `result` - fPCA result containing eigenfunctions and scores
/// * `n_components` - Number of components to use (capped at available components)
///
/// # Returns
/// Reconstructed curves: one `Vec<f64>` per observation.
pub fn reconstruct_from_scores(result: &FPCAResult, n_components: usize) -> Vec<Vec<f64>> {
    let n_curves = result.scores.nrows();
    let n_grid = result.grid.len();
    let k = n_components
        .min(result.eigenvalues.len())
        .min(result.eigenfunctions.nrows());

    let mut reconstructed = Vec::with_capacity(n_curves);
    for i in 0..n_curves {
        let mut curve = vec![0.0; n_grid];
        for comp in 0..k {
            let score = result.scores[[i, comp]];
            for j in 0..n_grid {
                curve[j] += score * result.eigenfunctions[[comp, j]];
            }
        }
        reconstructed.push(curve);
    }
    reconstructed
}

/// Compute trapezoidal-rule grid spacing weights.
fn compute_grid_spacing(grid: &[f64]) -> Vec<f64> {
    let n = grid.len();
    if n <= 1 {
        return vec![1.0; n];
    }
    let mut dt = vec![0.0; n];
    // Trapezoidal weights
    dt[0] = (grid[1] - grid[0]) / 2.0;
    for i in 1..n - 1 {
        dt[i] = (grid[i + 1] - grid[i - 1]) / 2.0;
    }
    dt[n - 1] = (grid[n - 1] - grid[n - 2]) / 2.0;
    dt
}

/// Symmetric eigendecomposition via power iteration with deflation.
///
/// Returns the top `k` eigenvalues (descending) and corresponding eigenvectors
/// (columns of the returned matrix).
fn symmetric_eigen_decomposition(
    a: &Array2<f64>,
    k: usize,
) -> StatsResult<(Array1<f64>, Array2<f64>)> {
    let n = a.nrows();
    let k = k.min(n);
    let max_iter = 1000;
    let tol = 1e-12;

    let mut eigenvalues = Array1::<f64>::zeros(k);
    let mut eigenvectors = Array2::<f64>::zeros((n, k));
    let mut deflated = a.clone();

    for comp in 0..k {
        // Power iteration for the dominant eigenvector of the deflated matrix
        let mut v = Array1::<f64>::zeros(n);
        // Initialize with a non-zero vector (use alternating pattern for diversity)
        for i in 0..n {
            v[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-14 {
            v /= norm;
        }

        let mut eigenvalue = 0.0;
        for _iter in 0..max_iter {
            let w = deflated.dot(&v);
            let new_eigenvalue: f64 = v.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
            let w_norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if w_norm < 1e-14 {
                // Zero eigenvalue
                eigenvalue = 0.0;
                break;
            }
            let new_v = &w / w_norm;

            if (new_eigenvalue - eigenvalue).abs() < tol * (1.0 + eigenvalue.abs()) {
                eigenvalue = new_eigenvalue;
                v = new_v;
                break;
            }
            eigenvalue = new_eigenvalue;
            v = new_v;
        }

        eigenvalues[comp] = eigenvalue.max(0.0); // eigenvalues of covariance are non-negative
        for i in 0..n {
            eigenvectors[[i, comp]] = v[i];
        }

        // Deflate: A <- A - lambda * v * v^T
        for i in 0..n {
            for j in 0..n {
                deflated[[i, j]] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    Ok((eigenvalues, eigenvectors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functional::types::BasisType;

    #[test]
    fn test_fpca_sine_cosine_mixture() {
        // Generate functional data: mixture of sin and cos
        let n_grid = 100;
        let grid: Vec<f64> = (0..n_grid).map(|i| i as f64 / n_grid as f64).collect();
        let n_curves = 50;

        let mut observations = Vec::with_capacity(n_curves);
        for i in 0..n_curves {
            let a = (i as f64 * 0.37).sin() * 2.0; // Score for component 1
            let b = (i as f64 * 0.73).cos() * 1.0; // Score for component 2
            let curve: Vec<f64> = grid
                .iter()
                .map(|&t| {
                    a * (2.0 * std::f64::consts::PI * t).sin()
                        + b * (2.0 * std::f64::consts::PI * t).cos()
                        + 0.01 * (i as f64 * t * 17.3).sin() // tiny noise
                })
                .collect();
            observations.push(curve);
        }

        let data = FunctionalData::new(grid, observations).expect("Data creation should succeed");
        let config = FunctionalConfig {
            basis: BasisType::BSpline {
                n_basis: 20,
                degree: 3,
            },
            smoothing_param: Some(1e-6),
            n_components: 4,
        };

        let result = functional_pca(&data, &config).expect("fPCA should succeed");

        // The first two components should explain most of the variance
        let top2_var: f64 = result.variance_explained[0] + result.variance_explained[1];
        assert!(
            top2_var > 0.95,
            "Top 2 components should explain >95% variance, got {}",
            top2_var
        );

        // Eigenvalues should be in descending order
        for i in 1..result.eigenvalues.len() {
            assert!(
                result.eigenvalues[i] <= result.eigenvalues[i - 1] + 1e-10,
                "Eigenvalues should be descending"
            );
        }
    }

    #[test]
    fn test_fpca_variance_explained_sums_to_one() {
        let n_grid = 50;
        let grid: Vec<f64> = (0..n_grid).map(|i| i as f64 / n_grid as f64).collect();
        let n_curves = 30;

        let mut observations = Vec::with_capacity(n_curves);
        for i in 0..n_curves {
            let curve: Vec<f64> = grid
                .iter()
                .map(|&t| (i as f64 * 0.5).sin() * t + (i as f64 * 0.3).cos() * t * t)
                .collect();
            observations.push(curve);
        }

        let data = FunctionalData::new(grid, observations).expect("Data creation should succeed");
        let config = FunctionalConfig {
            basis: BasisType::Fourier { n_basis: 11 },
            smoothing_param: Some(1e-4),
            n_components: 5,
        };

        let result = functional_pca(&data, &config).expect("fPCA should succeed");

        // All variance_explained values should be non-negative
        for &ve in result.variance_explained.iter() {
            assert!(ve >= -1e-10, "Variance explained should be non-negative");
        }

        // Sum should be <= 1.0 (could be less if we don't have all components)
        let total: f64 = result.variance_explained.iter().sum();
        assert!(
            total <= 1.0 + 1e-6,
            "Sum of variance explained should be <= 1.0, got {}",
            total
        );
    }

    #[test]
    fn test_reconstruct_from_scores() {
        let n_grid = 50;
        let grid: Vec<f64> = (0..n_grid).map(|i| i as f64 / n_grid as f64).collect();
        let n_curves = 20;

        let mut observations = Vec::with_capacity(n_curves);
        for i in 0..n_curves {
            let curve: Vec<f64> = grid
                .iter()
                .map(|&t| {
                    (i as f64 * 0.4).sin() * (2.0 * std::f64::consts::PI * t).sin()
                        + (i as f64 * 0.7).cos() * t
                })
                .collect();
            observations.push(curve);
        }

        let data = FunctionalData::new(grid, observations).expect("Data creation should succeed");
        let config = FunctionalConfig {
            basis: BasisType::BSpline {
                n_basis: 15,
                degree: 3,
            },
            smoothing_param: Some(1e-5),
            n_components: 3,
        };

        let result = functional_pca(&data, &config).expect("fPCA should succeed");

        let reconstructed = reconstruct_from_scores(&result, 3);
        assert_eq!(reconstructed.len(), n_curves);
        assert_eq!(reconstructed[0].len(), n_grid);

        // Reconstruction with all components should be close to the centered data
        // (we'd need the mean to compare to original, so just check dimensions)
        let recon_1 = reconstruct_from_scores(&result, 1);
        assert_eq!(recon_1.len(), n_curves);
    }

    #[test]
    fn test_fpca_eigenfunctions_orthogonal() {
        let n_grid = 60;
        let grid: Vec<f64> = (0..n_grid).map(|i| i as f64 / n_grid as f64).collect();
        let n_curves = 30;

        let mut observations = Vec::with_capacity(n_curves);
        for i in 0..n_curves {
            let curve: Vec<f64> = grid
                .iter()
                .map(|&t| {
                    (i as f64 * 0.5).sin() * (2.0 * std::f64::consts::PI * t).sin()
                        + (i as f64 * 0.3).cos() * (4.0 * std::f64::consts::PI * t).cos()
                        + (i as f64 * 0.7).sin() * t
                })
                .collect();
            observations.push(curve);
        }

        let data =
            FunctionalData::new(grid.clone(), observations).expect("Data creation should succeed");
        let config = FunctionalConfig {
            basis: BasisType::BSpline {
                n_basis: 15,
                degree: 3,
            },
            smoothing_param: Some(1e-5),
            n_components: 3,
        };

        let result = functional_pca(&data, &config).expect("fPCA should succeed");

        // Eigenfunctions should be approximately orthogonal in L2
        let dt: Vec<f64> = {
            let n = grid.len();
            let mut d = vec![0.0; n];
            d[0] = (grid[1] - grid[0]) / 2.0;
            for i in 1..n - 1 {
                d[i] = (grid[i + 1] - grid[i - 1]) / 2.0;
            }
            d[n - 1] = (grid[n - 1] - grid[n - 2]) / 2.0;
            d
        };

        for k1 in 0..result.eigenfunctions.nrows() {
            for k2 in (k1 + 1)..result.eigenfunctions.nrows() {
                let inner: f64 = (0..n_grid)
                    .map(|j| {
                        result.eigenfunctions[[k1, j]] * result.eigenfunctions[[k2, j]] * dt[j]
                    })
                    .sum();
                assert!(
                    inner.abs() < 0.1,
                    "Eigenfunctions {} and {} should be orthogonal, inner product = {}",
                    k1,
                    k2,
                    inner
                );
            }
        }
    }
}
