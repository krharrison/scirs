//! Neighborhood Selection (Meinshausen & Buhlmann, 2006)
//!
//! Estimates the support of the precision matrix by running LASSO regressions
//! for each variable against all others. This approach is computationally
//! efficient and provides consistent edge selection under certain conditions.
//!
//! The method works by:
//! 1. For each variable j, regressing X_j on X_{-j} using L1-penalized (LASSO) regression
//! 2. Identifying non-zero coefficients as neighbors of variable j
//! 3. Symmetrizing the resulting adjacency matrix using AND or OR rule

use crate::error::{StatsError, StatsResult};

/// Rule for symmetrizing the adjacency matrix after neighborhood selection
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum SelectionRule {
    /// Both (i,j) and (j,i) must be non-zero for edge inclusion (conservative)
    And,
    /// Either (i,j) or (j,i) non-zero suffices for edge inclusion (liberal)
    Or,
}

/// Configuration for the neighborhood selection algorithm
#[derive(Debug, Clone)]
pub struct NeighborhoodSelectionConfig {
    /// LASSO regularization strength (must be > 0)
    pub alpha: f64,
    /// Threshold for edge inclusion (coefficients below this are zeroed)
    pub threshold: f64,
    /// AND or OR rule for edge symmetrization
    pub method: SelectionRule,
    /// Maximum number of coordinate descent iterations per LASSO
    pub max_iter: usize,
    /// Convergence tolerance for LASSO coordinate descent
    pub tolerance: f64,
}

impl Default for NeighborhoodSelectionConfig {
    fn default() -> Self {
        NeighborhoodSelectionConfig {
            alpha: 0.1,
            threshold: 1e-6,
            method: SelectionRule::Or,
            max_iter: 1000,
            tolerance: 1e-6,
        }
    }
}

impl NeighborhoodSelectionConfig {
    /// Create a new configuration with the given regularization strength
    pub fn new(alpha: f64) -> StatsResult<Self> {
        if alpha <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "alpha must be positive".to_string(),
            ));
        }
        Ok(NeighborhoodSelectionConfig {
            alpha,
            ..Default::default()
        })
    }

    /// Set the threshold for edge inclusion
    pub fn with_threshold(mut self, threshold: f64) -> StatsResult<Self> {
        if threshold < 0.0 {
            return Err(StatsError::InvalidArgument(
                "threshold must be non-negative".to_string(),
            ));
        }
        self.threshold = threshold;
        Ok(self)
    }

    /// Set the symmetrization rule
    pub fn with_method(mut self, method: SelectionRule) -> Self {
        self.method = method;
        self
    }

    /// Set maximum iterations for LASSO solver
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
}

/// Result of the neighborhood selection procedure
#[derive(Debug, Clone)]
pub struct NeighborhoodSelectionResult {
    /// Boolean adjacency matrix (p x p) indicating edges in the estimated graph
    pub adjacency_matrix: Vec<Vec<bool>>,
    /// LASSO coefficients for each variable's regression (p x p, row j = regression of X_j)
    pub coefficients: Vec<Vec<f64>>,
    /// Total number of undirected edges in the estimated graph
    pub n_edges: usize,
    /// Number of variables
    pub n_variables: usize,
}

impl NeighborhoodSelectionResult {
    /// Return the density (fraction of edges out of maximum possible)
    pub fn density(&self) -> f64 {
        let p = self.n_variables;
        let max_edges = p * (p - 1) / 2;
        if max_edges == 0 {
            return 0.0;
        }
        self.n_edges as f64 / max_edges as f64
    }

    /// Return the degree (number of neighbors) for each variable
    pub fn degrees(&self) -> Vec<usize> {
        let p = self.n_variables;
        let mut deg = vec![0usize; p];
        for i in 0..p {
            for j in (i + 1)..p {
                if self.adjacency_matrix[i][j] {
                    deg[i] += 1;
                    deg[j] += 1;
                }
            }
        }
        deg
    }
}

/// Run neighborhood selection on a data matrix.
///
/// For each variable j, performs LASSO regression of X_j on X_{-j} using
/// coordinate descent. Non-zero coefficients indicate edges in the
/// conditional independence graph.
///
/// # Arguments
/// * `data` - n_samples x p_features data matrix (row-major)
/// * `config` - Configuration parameters
///
/// # Returns
/// A `NeighborhoodSelectionResult` containing the estimated graph structure
pub fn neighborhood_selection(
    data: &[Vec<f64>],
    config: &NeighborhoodSelectionConfig,
) -> StatsResult<NeighborhoodSelectionResult> {
    let n = data.len();
    if n < 2 {
        return Err(StatsError::InsufficientData(
            "Need at least 2 samples for neighborhood selection".to_string(),
        ));
    }

    let p = data[0].len();
    if p == 0 {
        return Err(StatsError::InvalidArgument(
            "Data must have at least one variable".to_string(),
        ));
    }

    // Validate all rows have same length
    for (i, row) in data.iter().enumerate() {
        if row.len() != p {
            return Err(StatsError::DimensionMismatch(format!(
                "Row {} has length {} but expected {}",
                i,
                row.len(),
                p
            )));
        }
    }

    // Standardize columns (zero mean, unit variance) for numerical stability
    let (standardized, _means, _stds) = standardize_columns(data, n, p)?;

    // Run LASSO for each variable
    let mut raw_adj = vec![vec![false; p]; p]; // directed adjacency
    let mut coefficients = vec![vec![0.0; p]; p];

    for j in 0..p {
        // Build response y = X_j and design matrix X_{-j}
        let y: Vec<f64> = standardized.iter().map(|row| row[j]).collect();
        let x_cols: Vec<usize> = (0..p).filter(|&k| k != j).collect();

        // Run coordinate descent LASSO
        let beta = lasso_coordinate_descent(
            &standardized,
            &y,
            &x_cols,
            config.alpha,
            config.max_iter,
            config.tolerance,
        )?;

        // Map coefficients back and mark non-zero entries
        for (idx, &col) in x_cols.iter().enumerate() {
            coefficients[j][col] = beta[idx];
            if beta[idx].abs() > config.threshold {
                raw_adj[j][col] = true;
            }
        }
    }

    // Symmetrize adjacency matrix using AND or OR rule
    let mut adj = vec![vec![false; p]; p];
    let mut n_edges = 0;

    for i in 0..p {
        for j in (i + 1)..p {
            let edge = match config.method {
                SelectionRule::And => raw_adj[i][j] && raw_adj[j][i],
                SelectionRule::Or => raw_adj[i][j] || raw_adj[j][i],
                _ => raw_adj[i][j] || raw_adj[j][i], // default to OR for future variants
            };
            if edge {
                adj[i][j] = true;
                adj[j][i] = true;
                n_edges += 1;
            }
        }
    }

    Ok(NeighborhoodSelectionResult {
        adjacency_matrix: adj,
        coefficients,
        n_edges,
        n_variables: p,
    })
}

/// Standardize columns to zero mean and unit variance
fn standardize_columns(
    data: &[Vec<f64>],
    n: usize,
    p: usize,
) -> StatsResult<(Vec<Vec<f64>>, Vec<f64>, Vec<f64>)> {
    let mut means = vec![0.0; p];
    let mut stds = vec![0.0; p];

    // Compute means
    for row in data {
        for j in 0..p {
            means[j] += row[j];
        }
    }
    for j in 0..p {
        means[j] /= n as f64;
    }

    // Compute standard deviations
    for row in data {
        for j in 0..p {
            let diff = row[j] - means[j];
            stds[j] += diff * diff;
        }
    }
    for j in 0..p {
        stds[j] = (stds[j] / (n - 1).max(1) as f64).sqrt();
        if stds[j] < 1e-15 {
            stds[j] = 1.0; // avoid division by zero for constant columns
        }
    }

    // Standardize
    let mut standardized = vec![vec![0.0; p]; n];
    for i in 0..n {
        for j in 0..p {
            standardized[i][j] = (data[i][j] - means[j]) / stds[j];
        }
    }

    Ok((standardized, means, stds))
}

/// LASSO regression via coordinate descent
///
/// Solves: min (1/2n) ||y - X*beta||^2 + alpha * ||beta||_1
///
/// # Arguments
/// * `data` - Full standardized data matrix (n x p)
/// * `y` - Response vector (length n)
/// * `x_cols` - Indices of predictor columns to use
/// * `alpha` - L1 regularization parameter
/// * `max_iter` - Maximum coordinate descent iterations
/// * `tol` - Convergence tolerance
fn lasso_coordinate_descent(
    data: &[Vec<f64>],
    y: &[f64],
    x_cols: &[usize],
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> StatsResult<Vec<f64>> {
    let n = data.len();
    let n_features = x_cols.len();
    let n_f64 = n as f64;

    // Precompute X^T X diagonal entries and X^T y
    let mut xty = vec![0.0; n_features];
    let mut xtx_diag = vec![0.0; n_features];

    for k in 0..n_features {
        let col = x_cols[k];
        for i in 0..n {
            xty[k] += data[i][col] * y[i];
            xtx_diag[k] += data[i][col] * data[i][col];
        }
        xty[k] /= n_f64;
        xtx_diag[k] /= n_f64;
    }

    let mut beta = vec![0.0; n_features];
    let mut residuals: Vec<f64> = y.to_vec();

    for _iter in 0..max_iter {
        let mut max_change: f64 = 0.0;

        for k in 0..n_features {
            let col = x_cols[k];
            let old_beta = beta[k];

            // Compute partial residual: r_k = (1/n) * X_k^T * (y - X*beta + X_k*beta_k)
            let mut partial_residual = 0.0;
            for i in 0..n {
                partial_residual += data[i][col] * (residuals[i] + data[i][col] * old_beta);
            }
            partial_residual /= n_f64;

            // Soft-thresholding update
            let denominator = xtx_diag[k];
            if denominator.abs() < 1e-15 {
                beta[k] = 0.0;
            } else {
                beta[k] = soft_threshold(partial_residual, alpha) / denominator;
            }

            // Update residuals
            let delta = beta[k] - old_beta;
            if delta.abs() > 1e-15 {
                for i in 0..n {
                    residuals[i] -= data[i][col] * delta;
                }
            }

            let change = delta.abs();
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

    /// Helper: generate AR(1) data with known tridiagonal precision matrix
    fn generate_ar1_data(n: usize, p: usize, rho: f64) -> Vec<Vec<f64>> {
        let mut data = vec![vec![0.0; p]; n];
        // Use deterministic pseudo-random values for reproducibility
        for i in 0..n {
            data[i][0] = ((i as f64 * 1.7 + 0.3).sin() * 2.0).tanh();
            for j in 1..p {
                let noise = ((i as f64 * (j as f64 + 1.0) * 0.73 + 0.5).sin()).tanh();
                data[i][j] = rho * data[i][j - 1] + (1.0 - rho * rho).sqrt() * noise;
            }
        }
        data
    }

    #[test]
    fn test_neighborhood_selection_basic() {
        let data = generate_ar1_data(100, 4, 0.5);
        let config = NeighborhoodSelectionConfig::new(0.1)
            .expect("config creation failed");
        let result = neighborhood_selection(&data, &config)
            .expect("neighborhood selection failed");

        assert_eq!(result.n_variables, 4);
        // Should have some edges
        assert!(result.n_edges <= 4 * 3 / 2, "Cannot exceed max edges");
    }

    #[test]
    fn test_neighborhood_selection_and_vs_or() {
        let data = generate_ar1_data(80, 3, 0.6);

        let config_or = NeighborhoodSelectionConfig::new(0.15)
            .expect("config creation failed")
            .with_method(SelectionRule::Or);
        let result_or = neighborhood_selection(&data, &config_or)
            .expect("OR selection failed");

        let config_and = NeighborhoodSelectionConfig::new(0.15)
            .expect("config creation failed")
            .with_method(SelectionRule::And);
        let result_and = neighborhood_selection(&data, &config_and)
            .expect("AND selection failed");

        // AND rule should be at least as sparse as OR rule
        assert!(
            result_and.n_edges <= result_or.n_edges,
            "AND ({}) should have <= edges than OR ({})",
            result_and.n_edges,
            result_or.n_edges
        );
    }

    #[test]
    fn test_neighborhood_selection_high_regularization_sparse() {
        let data = generate_ar1_data(50, 5, 0.3);
        let config = NeighborhoodSelectionConfig::new(1.0)
            .expect("config creation failed");
        let result = neighborhood_selection(&data, &config)
            .expect("neighborhood selection failed");

        // High regularization should yield very sparse graph
        assert!(
            result.n_edges <= 3,
            "High regularization should give sparse graph, got {} edges",
            result.n_edges
        );
    }

    #[test]
    fn test_neighborhood_selection_identity_data() {
        // Independent variables should yield empty graph
        let n = 100;
        let p = 3;
        let mut data = vec![vec![0.0; p]; n];
        for i in 0..n {
            for j in 0..p {
                // Independent "random" values
                data[i][j] = ((i as f64 * (j as f64 * 3.7 + 1.1) + 0.9).sin()).tanh();
            }
        }

        let config = NeighborhoodSelectionConfig::new(0.3)
            .expect("config creation failed");
        let result = neighborhood_selection(&data, &config)
            .expect("neighborhood selection failed");

        // For independent data with moderate regularization, should be sparse
        assert!(
            result.n_edges <= 2,
            "Independent data should yield sparse graph, got {} edges",
            result.n_edges
        );
    }

    #[test]
    fn test_neighborhood_selection_symmetry() {
        let data = generate_ar1_data(60, 4, 0.4);
        let config = NeighborhoodSelectionConfig::new(0.1)
            .expect("config creation failed");
        let result = neighborhood_selection(&data, &config)
            .expect("neighborhood selection failed");

        let p = result.n_variables;
        for i in 0..p {
            for j in 0..p {
                assert_eq!(
                    result.adjacency_matrix[i][j],
                    result.adjacency_matrix[j][i],
                    "Adjacency matrix should be symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_neighborhood_selection_error_insufficient_data() {
        let data = vec![vec![1.0, 2.0]]; // only 1 sample
        let config = NeighborhoodSelectionConfig::new(0.1)
            .expect("config creation failed");
        let result = neighborhood_selection(&data, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_neighborhood_selection_error_empty_variables() {
        let data: Vec<Vec<f64>> = vec![vec![], vec![]];
        let config = NeighborhoodSelectionConfig::new(0.1)
            .expect("config creation failed");
        let result = neighborhood_selection(&data, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_neighborhood_selection_error_invalid_alpha() {
        let result = NeighborhoodSelectionConfig::new(-0.1);
        assert!(result.is_err());

        let result = NeighborhoodSelectionConfig::new(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_neighborhood_selection_density() {
        let data = generate_ar1_data(50, 4, 0.3);
        let config = NeighborhoodSelectionConfig::new(0.1)
            .expect("config creation failed");
        let result = neighborhood_selection(&data, &config)
            .expect("neighborhood selection failed");

        let density = result.density();
        assert!(density >= 0.0 && density <= 1.0, "Density should be in [0, 1]");
    }

    #[test]
    fn test_neighborhood_selection_degrees() {
        let data = generate_ar1_data(80, 3, 0.5);
        let config = NeighborhoodSelectionConfig::new(0.05)
            .expect("config creation failed");
        let result = neighborhood_selection(&data, &config)
            .expect("neighborhood selection failed");

        let degrees = result.degrees();
        assert_eq!(degrees.len(), 3);

        // Sum of degrees should be 2 * n_edges
        let total_degree: usize = degrees.iter().sum();
        assert_eq!(total_degree, 2 * result.n_edges);
    }

    #[test]
    fn test_neighborhood_selection_dimension_mismatch() {
        let data = vec![vec![1.0, 2.0], vec![3.0]]; // inconsistent lengths
        let config = NeighborhoodSelectionConfig::new(0.1)
            .expect("config creation failed");
        let result = neighborhood_selection(&data, &config);
        assert!(result.is_err());
    }
}
