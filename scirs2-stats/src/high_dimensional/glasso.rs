//! Graphical Lasso (GLASSO) for sparse precision matrix estimation
//!
//! Implements the block coordinate descent algorithm of Friedman, Hastie, and
//! Tibshirani (2008) for solving the L1-penalized Gaussian log-likelihood:
//!
//! minimize_{Theta > 0}  -log det(Theta) + tr(S * Theta) + lambda * ||Theta||_1
//!
//! where Theta is the precision matrix (inverse covariance), S is the sample
//! covariance matrix, and the L1 penalty is applied only to off-diagonal elements.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};

/// A precision matrix wrapper providing convenient access to the estimated
/// sparse inverse covariance structure.
#[derive(Debug, Clone)]
pub struct PrecisionMatrix {
    /// The estimated precision matrix Theta (p x p)
    pub matrix: Array2<f64>,
    /// Number of non-zero off-diagonal entries (edges in the graph)
    pub num_edges: usize,
    /// The regularization parameter used
    pub lambda: f64,
}

impl PrecisionMatrix {
    /// Create a new PrecisionMatrix from a raw matrix and lambda
    pub fn new(matrix: Array2<f64>, lambda: f64) -> Self {
        let p = matrix.nrows();
        let mut num_edges = 0;
        for i in 0..p {
            for j in (i + 1)..p {
                if matrix[[i, j]].abs() > 1e-10 {
                    num_edges += 1;
                }
            }
        }
        PrecisionMatrix {
            matrix,
            num_edges,
            lambda,
        }
    }

    /// Return the adjacency (sparsity) pattern: true where |Theta_ij| > threshold
    pub fn adjacency(&self, threshold: f64) -> Array2<bool> {
        let p = self.matrix.nrows();
        let mut adj = Array2::from_elem((p, p), false);
        for i in 0..p {
            for j in 0..p {
                if i != j && self.matrix[[i, j]].abs() > threshold {
                    adj[[i, j]] = true;
                }
            }
        }
        adj
    }

    /// Return the density (fraction of non-zero off-diagonal entries)
    pub fn density(&self) -> f64 {
        let p = self.matrix.nrows();
        let max_edges = p * (p - 1) / 2;
        if max_edges == 0 {
            return 0.0;
        }
        self.num_edges as f64 / max_edges as f64
    }
}

/// Configuration for the Graphical Lasso algorithm
#[derive(Debug, Clone)]
pub struct GraphicalLassoConfig {
    /// Regularization parameter (must be > 0)
    pub lambda: f64,
    /// Maximum number of block coordinate descent iterations
    pub max_iter: usize,
    /// Convergence tolerance: stop when max |Delta Theta| < tol
    pub tolerance: f64,
    /// Whether to use warm start from a provided initial estimate
    pub warm_start: Option<Array2<f64>>,
}

impl Default for GraphicalLassoConfig {
    fn default() -> Self {
        GraphicalLassoConfig {
            lambda: 0.1,
            max_iter: 100,
            tolerance: 1e-4,
            warm_start: None,
        }
    }
}

impl GraphicalLassoConfig {
    /// Create a new config with the given lambda
    pub fn new(lambda: f64) -> StatsResult<Self> {
        if lambda <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "lambda must be positive".to_string(),
            ));
        }
        Ok(GraphicalLassoConfig {
            lambda,
            ..Default::default()
        })
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set warm start initial estimate
    pub fn with_warm_start(mut self, initial: Array2<f64>) -> Self {
        self.warm_start = Some(initial);
        self
    }
}

/// Result of the Graphical Lasso estimation
#[derive(Debug, Clone)]
pub struct GraphicalLassoResult {
    /// Estimated precision matrix Theta
    pub precision: PrecisionMatrix,
    /// Estimated covariance matrix Sigma = Theta^{-1}
    pub covariance: Array2<f64>,
    /// Number of iterations until convergence
    pub n_iter: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Final objective function value
    pub objective: f64,
}

/// Main Graphical Lasso estimator
#[derive(Debug, Clone)]
pub struct GraphicalLasso {
    config: GraphicalLassoConfig,
}

impl GraphicalLasso {
    /// Create a new GraphicalLasso estimator with the given configuration
    pub fn new(config: GraphicalLassoConfig) -> Self {
        GraphicalLasso { config }
    }

    /// Create a GraphicalLasso with just a lambda value (using defaults for the rest)
    pub fn with_lambda(lambda: f64) -> StatsResult<Self> {
        Ok(GraphicalLasso {
            config: GraphicalLassoConfig::new(lambda)?,
        })
    }

    /// Fit the Graphical Lasso model given the sample covariance matrix S.
    ///
    /// # Arguments
    /// * `s` - Sample covariance matrix (p x p, symmetric positive semi-definite)
    ///
    /// # Returns
    /// A `GraphicalLassoResult` containing the estimated precision and covariance matrices.
    pub fn fit(&self, s: &ArrayView2<f64>) -> StatsResult<GraphicalLassoResult> {
        let p = s.nrows();
        if s.ncols() != p {
            return Err(StatsError::DimensionMismatch(
                "Sample covariance matrix must be square".to_string(),
            ));
        }
        if p == 0 {
            return Err(StatsError::InvalidArgument(
                "Sample covariance matrix must be non-empty".to_string(),
            ));
        }

        // Check symmetry
        for i in 0..p {
            for j in (i + 1)..p {
                if (s[[i, j]] - s[[j, i]]).abs() > 1e-10 {
                    return Err(StatsError::InvalidArgument(
                        "Sample covariance matrix must be symmetric".to_string(),
                    ));
                }
            }
        }

        let lambda = self.config.lambda;

        // Special case: p == 1
        if p == 1 {
            let s_val = s[[0, 0]];
            if s_val <= 0.0 {
                return Err(StatsError::InvalidArgument(
                    "Diagonal of covariance must be positive".to_string(),
                ));
            }
            let theta_val = 1.0 / s_val;
            let precision_mat = Array2::from_elem((1, 1), theta_val);
            let cov_mat = Array2::from_elem((1, 1), s_val);
            let obj = -theta_val.ln() + 1.0; // -log det + tr(S*Theta) = -ln(theta) + s*theta = -ln(1/s) + 1
            return Ok(GraphicalLassoResult {
                precision: PrecisionMatrix::new(precision_mat, lambda),
                covariance: cov_mat,
                n_iter: 0,
                converged: true,
                objective: obj,
            });
        }

        // Initialize W (covariance estimate) and Theta (precision)
        let mut w = if let Some(ref warm) = self.config.warm_start {
            if warm.nrows() != p || warm.ncols() != p {
                return Err(StatsError::DimensionMismatch(
                    "Warm start matrix dimension mismatch".to_string(),
                ));
            }
            // Warm start: invert the provided precision to get covariance
            invert_symmetric(warm)?
        } else {
            // Initialize W = S + lambda * I
            let mut w_init = s.to_owned();
            for i in 0..p {
                w_init[[i, i]] += lambda;
            }
            w_init
        };

        let mut converged = false;
        let mut n_iter = 0;

        // Block coordinate descent
        for iter in 0..self.config.max_iter {
            let w_old = w.clone();

            for j in 0..p {
                // Partition: W_{11} is W without row/col j, w_{12} is column j (without diagonal)
                // s_{12} is the j-th column of S (without diagonal entry)
                let (w11, s12) = partition_matrix(&w, s, j);

                // Solve the Lasso subproblem for beta:
                //   minimize  beta^T W_{11} beta  - 2 * s_{12}^T beta + lambda * ||beta||_1
                // Using coordinate descent on the Lasso
                let beta = solve_lasso_subproblem(&w11, &s12, lambda, 100, 1e-6)?;

                // Update W: w_{12} = W_{11} * beta
                let w12 = w11.dot(&beta);

                // Write back into W
                let mut idx = 0;
                for i in 0..p {
                    if i != j {
                        w[[i, j]] = w12[idx];
                        w[[j, i]] = w12[idx];
                        idx += 1;
                    }
                }
            }

            // Check convergence: max absolute change in W
            let mut max_change: f64 = 0.0;
            for i in 0..p {
                for k in 0..p {
                    let diff = (w[[i, k]] - w_old[[i, k]]).abs();
                    if diff > max_change {
                        max_change = diff;
                    }
                }
            }

            n_iter = iter + 1;
            if max_change < self.config.tolerance {
                converged = true;
                break;
            }
        }

        // Compute precision matrix Theta = W^{-1}
        let theta = invert_symmetric(&w)?;

        // Compute objective: -log det(Theta) + tr(S * Theta) + lambda * ||Theta||_1 (off-diag)
        let objective = compute_objective(&theta, s, lambda)?;

        Ok(GraphicalLassoResult {
            precision: PrecisionMatrix::new(theta, lambda),
            covariance: w,
            n_iter,
            converged,
            objective,
        })
    }
}

/// Compute the GLASSO objective function value
pub(crate) fn compute_objective(
    theta: &Array2<f64>,
    s: &ArrayView2<f64>,
    lambda: f64,
) -> StatsResult<f64> {
    let p = theta.nrows();

    // -log det(Theta)
    let log_det = log_determinant(theta)?;
    let neg_log_det = -log_det;

    // tr(S * Theta)
    let mut trace_st = 0.0;
    for i in 0..p {
        for j in 0..p {
            trace_st += s[[i, j]] * theta[[i, j]];
        }
    }

    // lambda * ||Theta||_1 (off-diagonal only)
    let mut l1_off_diag = 0.0;
    for i in 0..p {
        for j in 0..p {
            if i != j {
                l1_off_diag += theta[[i, j]].abs();
            }
        }
    }

    Ok(neg_log_det + trace_st + lambda * l1_off_diag)
}

/// Compute log determinant of a symmetric positive definite matrix via Cholesky
fn log_determinant(a: &Array2<f64>) -> StatsResult<f64> {
    let p = a.nrows();
    // Cholesky decomposition: A = L * L^T
    let l = cholesky_decomp(a)?;
    let mut log_det = 0.0;
    for i in 0..p {
        let diag = l[[i, i]];
        if diag <= 0.0 {
            return Err(StatsError::ComputationError(
                "Matrix is not positive definite (Cholesky failed)".to_string(),
            ));
        }
        log_det += diag.ln();
    }
    Ok(2.0 * log_det)
}

/// Cholesky decomposition: returns lower triangular L such that A = L * L^T
fn cholesky_decomp(a: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let p = a.nrows();
    let mut l = Array2::<f64>::zeros((p, p));

    for i in 0..p {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }
            if i == j {
                let val = a[[i, i]] - sum;
                if val <= 0.0 {
                    return Err(StatsError::ComputationError(format!(
                        "Matrix is not positive definite at index {}",
                        i
                    )));
                }
                l[[i, j]] = val.sqrt();
            } else {
                if l[[j, j]].abs() < 1e-15 {
                    return Err(StatsError::ComputationError(
                        "Near-zero diagonal in Cholesky decomposition".to_string(),
                    ));
                }
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }
    Ok(l)
}

/// Invert a symmetric positive definite matrix via Cholesky decomposition
fn invert_symmetric(a: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let p = a.nrows();
    let l = cholesky_decomp(a)?;

    // Solve L * Z = I for Z (forward substitution)
    let mut z = Array2::<f64>::zeros((p, p));
    for col in 0..p {
        for i in 0..p {
            let mut sum = if i == col { 1.0 } else { 0.0 };
            for k in 0..i {
                sum -= l[[i, k]] * z[[k, col]];
            }
            z[[i, col]] = sum / l[[i, i]];
        }
    }

    // Solve L^T * X = Z for X (backward substitution)
    let mut inv = Array2::<f64>::zeros((p, p));
    for col in 0..p {
        for i in (0..p).rev() {
            let mut sum = z[[i, col]];
            for k in (i + 1)..p {
                sum -= l[[k, i]] * inv[[k, col]];
            }
            inv[[i, col]] = sum / l[[i, i]];
        }
    }

    Ok(inv)
}

/// Extract the (p-1) x (p-1) submatrix W_{11} and the (p-1)-vector s_{12}
/// by removing row/column j.
fn partition_matrix(w: &Array2<f64>, s: &ArrayView2<f64>, j: usize) -> (Array2<f64>, Array1<f64>) {
    let p = w.nrows();
    let pm1 = p - 1;

    let mut w11 = Array2::<f64>::zeros((pm1, pm1));
    let mut s12 = Array1::<f64>::zeros(pm1);

    let mut ri = 0;
    for i in 0..p {
        if i == j {
            continue;
        }
        s12[ri] = s[[i, j]];
        let mut ci = 0;
        for k in 0..p {
            if k == j {
                continue;
            }
            w11[[ri, ci]] = w[[i, k]];
            ci += 1;
        }
        ri += 1;
    }

    (w11, s12)
}

/// Solve the Lasso regression subproblem via coordinate descent:
///   minimize  0.5 * beta^T W11 beta - s12^T beta + lambda * ||beta||_1
///
/// This is the inner loop of the GLASSO block coordinate descent.
fn solve_lasso_subproblem(
    w11: &Array2<f64>,
    s12: &Array1<f64>,
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> StatsResult<Array1<f64>> {
    let pm1 = s12.len();
    let mut beta = Array1::<f64>::zeros(pm1);

    for _iter in 0..max_iter {
        let mut max_change: f64 = 0.0;

        for k in 0..pm1 {
            // Compute the partial residual
            let mut residual = s12[k];
            for m in 0..pm1 {
                if m != k {
                    residual -= w11[[k, m]] * beta[m];
                }
            }

            // Soft-thresholding
            let w_kk = w11[[k, k]];
            if w_kk.abs() < 1e-15 {
                continue;
            }
            let new_val = soft_threshold(residual, lambda) / w_kk;

            let change = (new_val - beta[k]).abs();
            if change > max_change {
                max_change = change;
            }
            beta[k] = new_val;
        }

        if max_change < tol {
            break;
        }
    }

    Ok(beta)
}

/// Soft-thresholding operator: sign(x) * max(|x| - lambda, 0)
#[inline]
fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Test GLASSO on identity covariance: should recover near-identity precision
    #[test]
    fn test_glasso_identity_covariance() {
        let s = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let config = GraphicalLassoConfig::new(0.01).expect("config creation failed");
        let glasso = GraphicalLasso::new(config);
        let result = glasso.fit(&s.view()).expect("GLASSO fit failed");

        assert!(result.converged, "GLASSO should converge");

        // Precision matrix should be close to identity
        let theta = &result.precision.matrix;
        for i in 0..3 {
            assert!(
                (theta[[i, i]] - 1.0).abs() < 0.1,
                "Diagonal should be near 1.0, got {}",
                theta[[i, i]]
            );
            for j in 0..3 {
                if i != j {
                    assert!(
                        theta[[i, j]].abs() < 0.1,
                        "Off-diagonal should be near 0.0, got {}",
                        theta[[i, j]]
                    );
                }
            }
        }
    }

    /// Test GLASSO recovers known sparse structure (chain graph)
    #[test]
    fn test_glasso_sparse_structure_recovery() {
        // True precision: tridiagonal (chain graph)
        // Theta = [[1, -0.5, 0], [-0.5, 1.25, -0.5], [0, -0.5, 1]]
        // Sigma = inv(Theta)
        let true_theta = array![[1.0, -0.5, 0.0], [-0.5, 1.25, -0.5], [0.0, -0.5, 1.0]];
        // Invert to get covariance
        let sigma = invert_symmetric(&true_theta).expect("inversion failed");

        // Use small lambda to recover structure
        let config = GraphicalLassoConfig::new(0.05).expect("config creation failed");
        let glasso = GraphicalLasso::new(config);
        let result = glasso.fit(&sigma.view()).expect("GLASSO fit failed");

        let theta = &result.precision.matrix;
        // The (0,2) and (2,0) entries should be near zero (no direct edge)
        assert!(
            theta[[0, 2]].abs() < 0.2,
            "Expected near-zero for (0,2), got {}",
            theta[[0, 2]]
        );
        assert!(
            theta[[2, 0]].abs() < 0.2,
            "Expected near-zero for (2,0), got {}",
            theta[[2, 0]]
        );
        // The (0,1) entry should be non-zero
        assert!(
            theta[[0, 1]].abs() > 0.1,
            "Expected non-zero for (0,1), got {}",
            theta[[0, 1]]
        );
    }

    /// Test GLASSO with different lambda values: larger lambda => sparser
    #[test]
    fn test_glasso_lambda_sparsity() {
        let s = array![[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]];

        let config_small = GraphicalLassoConfig::new(0.01).expect("config creation failed");
        let result_small = GraphicalLasso::new(config_small)
            .fit(&s.view())
            .expect("GLASSO fit failed");

        let config_large = GraphicalLassoConfig::new(0.5).expect("config creation failed");
        let result_large = GraphicalLasso::new(config_large)
            .fit(&s.view())
            .expect("GLASSO fit failed");

        // Larger lambda should yield fewer edges (sparser)
        assert!(
            result_large.precision.num_edges <= result_small.precision.num_edges,
            "Larger lambda should give sparser result: {} edges (large) vs {} edges (small)",
            result_large.precision.num_edges,
            result_small.precision.num_edges
        );
    }

    /// Test convergence behavior
    #[test]
    fn test_glasso_convergence() {
        let s = array![
            [2.0, 0.8, 0.3, 0.1],
            [0.8, 2.0, 0.5, 0.2],
            [0.3, 0.5, 2.0, 0.6],
            [0.1, 0.2, 0.6, 2.0]
        ];

        let config = GraphicalLassoConfig::new(0.1)
            .expect("config creation failed")
            .with_max_iter(200)
            .with_tolerance(1e-6);
        let result = GraphicalLasso::new(config)
            .fit(&s.view())
            .expect("GLASSO fit failed");

        assert!(
            result.converged,
            "Should converge with sufficient iterations"
        );
        assert!(result.n_iter > 0, "Should take at least one iteration");
    }

    /// Test warm start produces same (or similar) result
    #[test]
    fn test_glasso_warm_start() {
        let s = array![[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]];

        // First run without warm start
        let config1 = GraphicalLassoConfig::new(0.1).expect("config creation failed");
        let result1 = GraphicalLasso::new(config1)
            .fit(&s.view())
            .expect("GLASSO fit failed");

        // Second run with warm start from first result
        let config2 = GraphicalLassoConfig::new(0.1)
            .expect("config creation failed")
            .with_warm_start(result1.precision.matrix.clone());
        let result2 = GraphicalLasso::new(config2)
            .fit(&s.view())
            .expect("GLASSO fit failed");

        // Results should be close
        let p = result1.precision.matrix.nrows();
        for i in 0..p {
            for j in 0..p {
                assert!(
                    (result1.precision.matrix[[i, j]] - result2.precision.matrix[[i, j]]).abs()
                        < 0.05,
                    "Warm start result should be close to cold start"
                );
            }
        }

        // Warm start should converge in fewer (or equal) iterations
        assert!(
            result2.n_iter <= result1.n_iter + 1,
            "Warm start should not take significantly more iterations"
        );
    }

    /// Test single-variable (p=1) case
    #[test]
    fn test_glasso_single_variable() {
        let s = array![[4.0]];
        let config = GraphicalLassoConfig::new(0.1).expect("config creation failed");
        let result = GraphicalLasso::new(config)
            .fit(&s.view())
            .expect("GLASSO fit failed");

        assert!(
            (result.precision.matrix[[0, 0]] - 0.25).abs() < 1e-10,
            "Precision should be 1/variance"
        );
        assert!(result.converged);
    }

    /// Test error on non-square matrix
    #[test]
    fn test_glasso_non_square_error() {
        // We can't easily create a non-square ArrayView2 from array!, so test with asymmetric
        let s = array![[1.0, 0.5], [0.6, 1.0]]; // asymmetric
        let config = GraphicalLassoConfig::new(0.1).expect("config creation failed");
        let result = GraphicalLasso::new(config).fit(&s.view());
        assert!(result.is_err(), "Should error on asymmetric matrix");
    }

    /// Test invalid lambda
    #[test]
    fn test_glasso_invalid_lambda() {
        let result = GraphicalLassoConfig::new(-0.1);
        assert!(result.is_err(), "Should error on negative lambda");

        let result = GraphicalLassoConfig::new(0.0);
        assert!(result.is_err(), "Should error on zero lambda");
    }

    /// Test PrecisionMatrix helper methods
    #[test]
    fn test_precision_matrix_helpers() {
        let mat = array![[1.0, -0.3, 0.0], [-0.3, 1.0, -0.2], [0.0, -0.2, 1.0]];
        let pm = PrecisionMatrix::new(mat, 0.1);

        assert_eq!(pm.num_edges, 2, "Should have 2 edges (0-1 and 1-2)");

        let adj = pm.adjacency(0.1);
        assert!(adj[[0, 1]]);
        assert!(adj[[1, 0]]);
        assert!(adj[[1, 2]]);
        assert!(adj[[2, 1]]);
        assert!(!adj[[0, 2]]);
        assert!(!adj[[2, 0]]);

        let density = pm.density();
        assert!((density - 2.0 / 3.0).abs() < 1e-10);
    }
}
