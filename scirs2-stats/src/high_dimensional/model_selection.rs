//! Model selection methods for Graphical Lasso
//!
//! Provides criteria for choosing the optimal regularization parameter lambda:
//! - BIC (Bayesian Information Criterion)
//! - EBIC (Extended BIC with tunable gamma)
//! - K-fold cross-validation
//! - StARS (Stability Approach to Regularization Selection)
//! - Lambda path generation
//! - Partial correlation extraction from precision matrices

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};

use super::glasso::{compute_objective, GraphicalLasso, GraphicalLassoConfig};

/// Wrapper for a covariance matrix estimate, with metadata about the sample
#[derive(Debug, Clone)]
pub struct CovarianceEstimate {
    /// The sample covariance matrix
    pub matrix: Array2<f64>,
    /// Number of samples used to compute it
    pub n_samples: usize,
    /// Number of variables (p)
    pub n_variables: usize,
}

impl CovarianceEstimate {
    /// Create a new covariance estimate
    pub fn new(matrix: Array2<f64>, n_samples: usize) -> StatsResult<Self> {
        let p = matrix.nrows();
        if matrix.ncols() != p {
            return Err(StatsError::DimensionMismatch(
                "Covariance matrix must be square".to_string(),
            ));
        }
        if n_samples == 0 {
            return Err(StatsError::InvalidArgument(
                "n_samples must be positive".to_string(),
            ));
        }
        Ok(CovarianceEstimate {
            matrix,
            n_samples,
            n_variables: p,
        })
    }
}

/// Which criterion to use for model selection
#[derive(Debug, Clone, Copy)]
pub enum ModelSelectionCriterion {
    /// Standard BIC
    Bic,
    /// Extended BIC with gamma parameter
    Ebic(f64),
    /// K-fold cross-validation
    CrossValidation(usize),
    /// StARS with beta threshold
    Stars(f64),
}

/// Result of EBIC computation for a single lambda
#[derive(Debug, Clone)]
pub struct EbicResult {
    /// The lambda value
    pub lambda: f64,
    /// The EBIC score (lower is better)
    pub ebic: f64,
    /// Number of non-zero off-diagonal entries in the precision matrix
    pub num_edges: usize,
}

/// Result of cross-validation for a single lambda
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// The lambda value
    pub lambda: f64,
    /// Mean negative log-likelihood across folds
    pub mean_nll: f64,
    /// Standard error of the negative log-likelihood across folds
    pub se_nll: f64,
}

/// Result of StARS for a single lambda
#[derive(Debug, Clone)]
pub struct StarsResult {
    /// The lambda value
    pub lambda: f64,
    /// Average instability (D-bar) across all edge pairs
    pub instability: f64,
}

/// Result of lambda selection
#[derive(Debug, Clone)]
pub struct LambdaSelectionResult {
    /// The optimal lambda
    pub best_lambda: f64,
    /// The criterion value at the optimal lambda
    pub best_score: f64,
    /// All lambdas evaluated
    pub lambdas: Vec<f64>,
    /// Scores for each lambda
    pub scores: Vec<f64>,
}

/// Matrix of partial correlations extracted from a precision matrix
#[derive(Debug, Clone)]
pub struct PartialCorrelationMatrix {
    /// The partial correlation matrix (p x p)
    pub matrix: Array2<f64>,
}

/// Generate a log-spaced path of lambda values from lambda_max down to lambda_min.
///
/// lambda_max is computed as the maximum absolute off-diagonal entry of S.
/// lambda_min = lambda_max * ratio.
///
/// # Arguments
/// * `s` - Sample covariance matrix
/// * `n_lambdas` - Number of lambda values to generate
/// * `ratio` - Ratio of lambda_min / lambda_max (e.g., 0.01)
pub fn generate_lambda_path(
    s: &ArrayView2<f64>,
    n_lambdas: usize,
    ratio: f64,
) -> StatsResult<Vec<f64>> {
    let p = s.nrows();
    if s.ncols() != p {
        return Err(StatsError::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }
    if n_lambdas == 0 {
        return Err(StatsError::InvalidArgument(
            "n_lambdas must be positive".to_string(),
        ));
    }
    if ratio <= 0.0 || ratio >= 1.0 {
        return Err(StatsError::InvalidArgument(
            "ratio must be in (0, 1)".to_string(),
        ));
    }

    // lambda_max = max absolute off-diagonal of S
    let mut lambda_max: f64 = 0.0;
    for i in 0..p {
        for j in 0..p {
            if i != j {
                let v = s[[i, j]].abs();
                if v > lambda_max {
                    lambda_max = v;
                }
            }
        }
    }

    if lambda_max <= 0.0 {
        // Diagonal matrix: all off-diag are zero, use a small default
        lambda_max = 1.0;
    }

    let lambda_min = lambda_max * ratio;
    let log_max = lambda_max.ln();
    let log_min = lambda_min.ln();

    let mut path = Vec::with_capacity(n_lambdas);
    if n_lambdas == 1 {
        path.push(lambda_max);
    } else {
        for i in 0..n_lambdas {
            let t = i as f64 / (n_lambdas - 1) as f64;
            let log_val = log_max + t * (log_min - log_max);
            path.push(log_val.exp());
        }
    }

    Ok(path)
}

/// Compute BIC for a given precision matrix estimate.
///
/// BIC = -2 * log-likelihood + k * ln(n)
///
/// where the Gaussian log-likelihood (up to constant) is:
///   (n/2) * [log det(Theta) - tr(S * Theta)]
///
/// and k is the number of free parameters (non-zero entries in upper triangle of Theta).
pub fn compute_bic(
    theta: &ArrayView2<f64>,
    s: &ArrayView2<f64>,
    n_samples: usize,
) -> StatsResult<f64> {
    let p = theta.nrows();
    if theta.ncols() != p || s.nrows() != p || s.ncols() != p {
        return Err(StatsError::DimensionMismatch(
            "Matrices must be square and same size".to_string(),
        ));
    }
    if n_samples == 0 {
        return Err(StatsError::InvalidArgument(
            "n_samples must be positive".to_string(),
        ));
    }

    let n = n_samples as f64;

    // Log-likelihood (up to constant): (n/2) * [log det(Theta) - tr(S*Theta)]
    let log_det = log_det_via_cholesky(theta)?;
    let trace_st = trace_product(s, theta);

    let log_lik = (n / 2.0) * (log_det - trace_st);

    // Count free parameters: non-zero entries in upper triangle (including diagonal)
    let mut k = 0usize;
    for i in 0..p {
        for j in i..p {
            if theta[[i, j]].abs() > 1e-10 {
                k += 1;
            }
        }
    }

    let bic = -2.0 * log_lik + (k as f64) * n.ln();
    Ok(bic)
}

/// Compute Extended BIC (EBIC) for a given precision matrix estimate.
///
/// EBIC = BIC + 4 * gamma * k * ln(p)
///
/// where gamma in [0,1] controls penalization of model complexity.
/// gamma=0 gives standard BIC; gamma=0.5 is recommended for high-dimensional settings.
pub fn compute_ebic(
    theta: &ArrayView2<f64>,
    s: &ArrayView2<f64>,
    n_samples: usize,
    gamma: f64,
) -> StatsResult<EbicResult> {
    if !(0.0..=1.0).contains(&gamma) {
        return Err(StatsError::InvalidArgument(
            "gamma must be in [0, 1]".to_string(),
        ));
    }

    let p = theta.nrows();
    let bic = compute_bic(theta, s, n_samples)?;

    // Count edges (non-zero off-diagonal upper triangle entries)
    let mut num_edges = 0usize;
    for i in 0..p {
        for j in (i + 1)..p {
            if theta[[i, j]].abs() > 1e-10 {
                num_edges += 1;
            }
        }
    }

    let ebic = bic + 4.0 * gamma * (num_edges as f64) * (p as f64).ln();

    Ok(EbicResult {
        lambda: 0.0, // to be set by caller
        ebic,
        num_edges,
    })
}

/// K-fold cross-validation for selecting lambda.
///
/// Splits the data into K folds, fits GLASSO on the training folds,
/// and evaluates the negative log-likelihood on the held-out fold.
///
/// # Arguments
/// * `data` - n x p data matrix (rows are samples)
/// * `lambdas` - lambda values to evaluate
/// * `k_folds` - number of cross-validation folds
pub fn cross_validate_glasso(
    data: &ArrayView2<f64>,
    lambdas: &[f64],
    k_folds: usize,
) -> StatsResult<LambdaSelectionResult> {
    let n = data.nrows();
    let p = data.ncols();

    if n < k_folds {
        return Err(StatsError::InsufficientData(format!(
            "Need at least {} samples for {}-fold CV, got {}",
            k_folds, k_folds, n
        )));
    }
    if k_folds < 2 {
        return Err(StatsError::InvalidArgument(
            "k_folds must be at least 2".to_string(),
        ));
    }
    if lambdas.is_empty() {
        return Err(StatsError::InvalidArgument(
            "lambdas must be non-empty".to_string(),
        ));
    }

    // Compute fold assignments (simple sequential split)
    let fold_size = n / k_folds;
    let mut fold_assignments = vec![0usize; n];
    for i in 0..n {
        fold_assignments[i] = (i / fold_size).min(k_folds - 1);
    }

    let mut best_lambda = lambdas[0];
    let mut best_score = f64::MAX;
    let mut all_scores = Vec::with_capacity(lambdas.len());

    for &lambda in lambdas {
        let mut fold_nlls = Vec::with_capacity(k_folds);

        for fold in 0..k_folds {
            // Split into train and test
            let mut train_rows = Vec::new();
            let mut test_rows = Vec::new();
            for i in 0..n {
                if fold_assignments[i] == fold {
                    test_rows.push(i);
                } else {
                    train_rows.push(i);
                }
            }

            if train_rows.is_empty() || test_rows.is_empty() {
                continue;
            }

            // Compute training covariance
            let train_cov = compute_sample_covariance(data, &train_rows)?;
            // Compute test covariance
            let test_cov = compute_sample_covariance(data, &test_rows)?;

            // Fit GLASSO on training data
            let config = GraphicalLassoConfig::new(lambda)?;
            let glasso = GraphicalLasso::new(config);
            let result = glasso.fit(&train_cov.view())?;

            // Evaluate negative log-likelihood on test data
            let theta = &result.precision.matrix;
            let log_det = log_det_via_cholesky(&theta.view());
            match log_det {
                Ok(ld) => {
                    let trace_st = trace_product(&test_cov.view(), &theta.view());
                    let nll = -ld + trace_st;
                    fold_nlls.push(nll);
                }
                Err(_) => {
                    // If precision is not positive definite, penalize heavily
                    fold_nlls.push(f64::MAX / (k_folds as f64));
                }
            }
        }

        if fold_nlls.is_empty() {
            all_scores.push(f64::MAX);
            continue;
        }

        let mean_nll: f64 = fold_nlls.iter().sum::<f64>() / fold_nlls.len() as f64;
        all_scores.push(mean_nll);

        if mean_nll < best_score {
            best_score = mean_nll;
            best_lambda = lambda;
        }
    }

    Ok(LambdaSelectionResult {
        best_lambda,
        best_score,
        lambdas: lambdas.to_vec(),
        scores: all_scores,
    })
}

/// StARS (Stability Approach to Regularization Selection)
///
/// Measures edge instability across subsamples. Selects the smallest lambda
/// such that instability D-bar < beta (typically beta = 0.05 or 0.1).
///
/// # Arguments
/// * `data` - n x p data matrix
/// * `lambdas` - lambda values (should be sorted descending)
/// * `n_subsamples` - number of subsamples to draw
/// * `subsample_ratio` - fraction of data to use per subsample (e.g. 0.8)
/// * `beta` - instability threshold
pub fn select_lambda_stars(
    data: &ArrayView2<f64>,
    lambdas: &[f64],
    n_subsamples: usize,
    subsample_ratio: f64,
    beta: f64,
) -> StatsResult<LambdaSelectionResult> {
    let n = data.nrows();
    let p = data.ncols();

    if n < 4 {
        return Err(StatsError::InsufficientData(
            "Need at least 4 samples for StARS".to_string(),
        ));
    }
    if lambdas.is_empty() {
        return Err(StatsError::InvalidArgument(
            "lambdas must be non-empty".to_string(),
        ));
    }
    if !(0.0..1.0).contains(&subsample_ratio) || subsample_ratio <= 0.0 {
        return Err(StatsError::InvalidArgument(
            "subsample_ratio must be in (0, 1)".to_string(),
        ));
    }
    if beta <= 0.0 || beta >= 0.5 {
        return Err(StatsError::InvalidArgument(
            "beta must be in (0, 0.5)".to_string(),
        ));
    }

    let b = (n as f64 * subsample_ratio).floor() as usize;
    if b < 2 {
        return Err(StatsError::InsufficientData(
            "Subsample size too small".to_string(),
        ));
    }

    let n_edges = p * (p - 1) / 2;
    let mut best_lambda = lambdas[0]; // most regularized
    let mut best_score = f64::MAX;
    let mut all_scores = Vec::with_capacity(lambdas.len());

    // Use a simple deterministic subsampling strategy for reproducibility
    for &lambda in lambdas {
        // Track edge selection frequencies across subsamples
        let mut edge_freq = Array2::<f64>::zeros((p, p));

        for sub in 0..n_subsamples {
            // Deterministic subsample selection (round-robin with offset)
            let offset = (sub * 7) % n; // simple hash-like offset
            let mut indices: Vec<usize> = (0..b).map(|i| (offset + i) % n).collect();
            indices.sort();
            indices.dedup();
            // Ensure we have enough samples
            while indices.len() < b {
                let next = (indices.last().copied().unwrap_or(0) + 1) % n;
                if !indices.contains(&next) {
                    indices.push(next);
                } else {
                    // Just add sequentially
                    for candidate in 0..n {
                        if !indices.contains(&candidate) {
                            indices.push(candidate);
                            break;
                        }
                    }
                }
            }
            indices.truncate(b);

            let sub_cov = compute_sample_covariance(data, &indices)?;

            let config = GraphicalLassoConfig::new(lambda)?;
            let glasso = GraphicalLasso::new(config);
            match glasso.fit(&sub_cov.view()) {
                Ok(result) => {
                    let theta = &result.precision.matrix;
                    for i in 0..p {
                        for j in (i + 1)..p {
                            if theta[[i, j]].abs() > 1e-10 {
                                edge_freq[[i, j]] += 1.0;
                            }
                        }
                    }
                }
                Err(_) => {
                    // Skip failed fits
                    continue;
                }
            }
        }

        // Normalize frequencies
        let ns = n_subsamples as f64;
        edge_freq.mapv_inplace(|v| v / ns);

        // Compute instability: D_bar = (2/choose(p,2)) * sum_{i<j} freq*(1-freq)
        let mut instability = 0.0;
        for i in 0..p {
            for j in (i + 1)..p {
                let f = edge_freq[[i, j]];
                instability += f * (1.0 - f);
            }
        }
        if n_edges > 0 {
            instability *= 2.0 / n_edges as f64;
        }

        all_scores.push(instability);

        // Track the smallest lambda where instability < beta
        // (lambdas assumed sorted largest to smallest)
        if instability < beta && instability < best_score {
            best_score = instability;
            best_lambda = lambda;
        }
    }

    // If no lambda had instability < beta, pick the largest lambda (most stable)
    if best_score == f64::MAX {
        best_lambda = lambdas[0];
        best_score = all_scores.first().copied().unwrap_or(f64::MAX);
    }

    Ok(LambdaSelectionResult {
        best_lambda,
        best_score,
        lambdas: lambdas.to_vec(),
        scores: all_scores,
    })
}

/// Extract partial correlations from a precision matrix.
///
/// The partial correlation between variables i and j (controlling for all others) is:
///   rho_{ij|rest} = -Theta_{ij} / sqrt(Theta_{ii} * Theta_{jj})
pub fn extract_partial_correlations(
    theta: &ArrayView2<f64>,
) -> StatsResult<PartialCorrelationMatrix> {
    let p = theta.nrows();
    if theta.ncols() != p {
        return Err(StatsError::DimensionMismatch(
            "Precision matrix must be square".to_string(),
        ));
    }

    let mut pcor = Array2::<f64>::zeros((p, p));

    for i in 0..p {
        pcor[[i, i]] = 1.0; // partial correlation of variable with itself
        for j in (i + 1)..p {
            let denom = theta[[i, i]] * theta[[j, j]];
            if denom <= 0.0 {
                // Cannot compute partial correlation if diagonal is non-positive
                pcor[[i, j]] = 0.0;
                pcor[[j, i]] = 0.0;
            } else {
                let val = -theta[[i, j]] / denom.sqrt();
                // Clamp to [-1, 1]
                let clamped = val.max(-1.0).min(1.0);
                pcor[[i, j]] = clamped;
                pcor[[j, i]] = clamped;
            }
        }
    }

    Ok(PartialCorrelationMatrix { matrix: pcor })
}

// ---- Internal helpers ----

/// Compute log determinant via Cholesky decomposition
fn log_det_via_cholesky(a: &ArrayView2<f64>) -> StatsResult<f64> {
    let p = a.nrows();
    let l = cholesky_lower(a)?;
    let mut log_det = 0.0;
    for i in 0..p {
        let d = l[[i, i]];
        if d <= 0.0 {
            return Err(StatsError::ComputationError(
                "Matrix not positive definite".to_string(),
            ));
        }
        log_det += d.ln();
    }
    Ok(2.0 * log_det)
}

/// Lower-triangular Cholesky factor
fn cholesky_lower(a: &ArrayView2<f64>) -> StatsResult<Array2<f64>> {
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
                        "Matrix not positive definite at index {}",
                        i
                    )));
                }
                l[[i, j]] = val.sqrt();
            } else {
                if l[[j, j]].abs() < 1e-15 {
                    return Err(StatsError::ComputationError(
                        "Near-zero diagonal in Cholesky".to_string(),
                    ));
                }
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }
    Ok(l)
}

/// Compute tr(A * B) without forming the full product
fn trace_product(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> f64 {
    let p = a.nrows();
    let mut trace = 0.0;
    for i in 0..p {
        for k in 0..p {
            trace += a[[i, k]] * b[[k, i]];
        }
    }
    trace
}

/// Compute sample covariance from a subset of rows of a data matrix
fn compute_sample_covariance(
    data: &ArrayView2<f64>,
    row_indices: &[usize],
) -> StatsResult<Array2<f64>> {
    let n = row_indices.len();
    let p = data.ncols();

    if n < 2 {
        return Err(StatsError::InsufficientData(
            "Need at least 2 samples for covariance".to_string(),
        ));
    }

    // Compute column means
    let mut means = Array1::<f64>::zeros(p);
    for &idx in row_indices {
        for j in 0..p {
            means[j] += data[[idx, j]];
        }
    }
    means.mapv_inplace(|v| v / n as f64);

    // Compute covariance
    let mut cov = Array2::<f64>::zeros((p, p));
    for &idx in row_indices {
        for i in 0..p {
            let di = data[[idx, i]] - means[i];
            for j in i..p {
                let dj = data[[idx, j]] - means[j];
                cov[[i, j]] += di * dj;
            }
        }
    }

    let denom = (n - 1) as f64;
    for i in 0..p {
        for j in i..p {
            cov[[i, j]] /= denom;
            if j > i {
                cov[[j, i]] = cov[[i, j]];
            }
        }
    }

    Ok(cov)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Test lambda path generation
    #[test]
    fn test_lambda_path_generation() {
        let s = array![[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]];

        let path = generate_lambda_path(&s.view(), 10, 0.01).expect("path generation failed");
        assert_eq!(path.len(), 10);

        // Should be in decreasing order
        for i in 1..path.len() {
            assert!(
                path[i] < path[i - 1],
                "Lambda path should be decreasing: {} >= {}",
                path[i],
                path[i - 1]
            );
        }

        // First should be close to max off-diagonal of S (0.5)
        assert!(
            (path[0] - 0.5).abs() < 1e-10,
            "First lambda should be lambda_max=0.5, got {}",
            path[0]
        );

        // Last should be close to lambda_max * ratio
        assert!(
            (path[9] - 0.005).abs() < 1e-3,
            "Last lambda should be ~0.005, got {}",
            path[9]
        );
    }

    /// Test BIC computation
    #[test]
    fn test_bic_computation() {
        // Identity precision, identity covariance, n=100
        let theta = array![[1.0, 0.0], [0.0, 1.0]];
        let s = array![[1.0, 0.0], [0.0, 1.0]];

        let bic = compute_bic(&theta.view(), &s.view(), 100).expect("BIC computation failed");

        // log det(I) = 0, tr(I*I) = 2
        // log_lik = (100/2)*(0 - 2) = -100
        // k = 2 (two non-zero diagonal entries)
        // BIC = -2*(-100) + 2*ln(100) = 200 + 2*4.605... = ~209.2
        assert!(
            (bic - (200.0 + 2.0 * 100.0_f64.ln())).abs() < 1e-6,
            "BIC value incorrect: {}",
            bic
        );
    }

    /// Test EBIC computation with gamma = 0 should equal BIC
    #[test]
    fn test_ebic_equals_bic_at_gamma_zero() {
        let theta = array![[1.0, -0.3, 0.0], [-0.3, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let s = array![[1.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let bic = compute_bic(&theta.view(), &s.view(), 50).expect("BIC failed");
        let ebic_result = compute_ebic(&theta.view(), &s.view(), 50, 0.0).expect("EBIC failed");

        assert!(
            (ebic_result.ebic - bic).abs() < 1e-10,
            "EBIC with gamma=0 should equal BIC"
        );
    }

    /// Test EBIC with gamma > 0 penalizes more
    #[test]
    fn test_ebic_gamma_penalization() {
        let theta = array![[1.0, -0.3, 0.0], [-0.3, 1.0, -0.2], [0.0, -0.2, 1.0]];
        let s = array![[1.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 1.0]];

        let ebic0 = compute_ebic(&theta.view(), &s.view(), 50, 0.0).expect("EBIC gamma=0 failed");
        let ebic05 =
            compute_ebic(&theta.view(), &s.view(), 50, 0.5).expect("EBIC gamma=0.5 failed");

        assert!(
            ebic05.ebic > ebic0.ebic,
            "EBIC with gamma=0.5 should be larger than gamma=0: {} vs {}",
            ebic05.ebic,
            ebic0.ebic
        );
    }

    /// Test partial correlation extraction from a known precision matrix
    #[test]
    fn test_partial_correlations_known_graph() {
        // Precision matrix for a simple chain: 1 -- 2 -- 3 (no direct 1--3 edge)
        let theta = array![[1.0, -0.5, 0.0], [-0.5, 1.25, -0.5], [0.0, -0.5, 1.0]];

        let pcor = extract_partial_correlations(&theta.view()).expect("partial correlation failed");

        // rho_{12} = -(-0.5) / sqrt(1.0*1.25) = 0.5/1.118... ~ 0.4472
        let expected_12 = 0.5 / (1.0 * 1.25_f64).sqrt();
        assert!(
            (pcor.matrix[[0, 1]] - expected_12).abs() < 1e-4,
            "Partial corr (0,1) should be ~{}, got {}",
            expected_12,
            pcor.matrix[[0, 1]]
        );

        // rho_{13} should be 0 (no direct edge)
        assert!(
            pcor.matrix[[0, 2]].abs() < 1e-10,
            "Partial corr (0,2) should be 0, got {}",
            pcor.matrix[[0, 2]]
        );

        // Diagonal should be 1
        for i in 0..3 {
            assert!(
                (pcor.matrix[[i, i]] - 1.0).abs() < 1e-10,
                "Diagonal partial corr should be 1"
            );
        }

        // Symmetry
        assert!(
            (pcor.matrix[[0, 1]] - pcor.matrix[[1, 0]]).abs() < 1e-10,
            "Partial correlations should be symmetric"
        );
    }

    /// Test cross-validation runs without error on small data
    #[test]
    fn test_cross_validation_basic() {
        // Generate a small dataset: 20 samples, 3 variables
        // Use a simple deterministic pattern
        let n = 20;
        let p = 3;
        let mut data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let t = i as f64 / n as f64;
            data[[i, 0]] = t;
            data[[i, 1]] = t * 0.8 + 0.1 * (i as f64 * 1.3).sin();
            data[[i, 2]] = 0.5 - t + 0.05 * (i as f64 * 2.7).cos();
        }

        let lambdas = vec![0.5, 0.3, 0.1];
        let result =
            cross_validate_glasso(&data.view(), &lambdas, 4).expect("cross-validation failed");

        assert_eq!(result.lambdas.len(), 3);
        assert_eq!(result.scores.len(), 3);
        assert!(result.best_lambda > 0.0);
    }

    /// Test BIC selects reasonable lambda
    #[test]
    fn test_bic_selects_reasonable_lambda() {
        // Sample covariance from near-identity structure
        let s = array![[1.0, 0.05, 0.02], [0.05, 1.0, 0.03], [0.02, 0.03, 1.0]];

        let lambdas = generate_lambda_path(&s.view(), 5, 0.01).expect("path generation failed");

        let mut best_lambda = lambdas[0];
        let mut best_bic = f64::MAX;
        let n_samples = 100;

        for &lambda in &lambdas {
            let config = GraphicalLassoConfig::new(lambda).expect("config failed");
            let glasso = GraphicalLasso::new(config);
            let result = glasso.fit(&s.view()).expect("fit failed");

            let bic = compute_bic(&result.precision.matrix.view(), &s.view(), n_samples)
                .expect("BIC failed");

            if bic < best_bic {
                best_bic = bic;
                best_lambda = lambda;
            }
        }

        // The selected lambda should be somewhere in the range
        assert!(best_lambda > 0.0, "BIC should select a positive lambda");
    }

    /// Test StARS basic functionality
    #[test]
    fn test_stars_basic() {
        let n = 30;
        let p = 3;
        let mut data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let t = i as f64 / n as f64;
            data[[i, 0]] = t + 0.1 * (i as f64 * 0.7).sin();
            data[[i, 1]] = t * 0.5 + 0.1 * (i as f64 * 1.1).cos();
            data[[i, 2]] = 1.0 - t + 0.1 * (i as f64 * 0.3).sin();
        }

        let lambdas = vec![0.5, 0.3, 0.1, 0.05];
        let result =
            select_lambda_stars(&data.view(), &lambdas, 5, 0.8, 0.1).expect("StARS failed");

        assert_eq!(result.lambdas.len(), 4);
        assert_eq!(result.scores.len(), 4);
        assert!(result.best_lambda > 0.0);

        // Instability scores should be in [0, 0.5]
        for &score in &result.scores {
            assert!(
                score >= 0.0 && score <= 0.5,
                "Instability should be in [0, 0.5], got {}",
                score
            );
        }
    }

    /// Test sample covariance computation
    #[test]
    fn test_sample_covariance() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let indices = vec![0, 1, 2, 3];
        let cov = compute_sample_covariance(&data.view(), &indices)
            .expect("covariance computation failed");

        // For this data, variance of each column should be computed correctly
        // Column 0: [1,3,5,7], mean=4, var = (9+1+1+9)/3 = 20/3
        // Column 1: [2,4,6,8], mean=5, var = (9+1+1+9)/3 = 20/3
        // Covariance = (1-4)(2-5)+(3-4)(4-5)+(5-4)(6-5)+(7-4)(8-5) / 3
        //            = 9+1+1+9 / 3 = 20/3
        let expected_var = 20.0 / 3.0;
        assert!(
            (cov[[0, 0]] - expected_var).abs() < 1e-10,
            "Variance of col 0 should be {}, got {}",
            expected_var,
            cov[[0, 0]]
        );
        assert!(
            (cov[[0, 1]] - expected_var).abs() < 1e-10,
            "Covariance should be {}, got {}",
            expected_var,
            cov[[0, 1]]
        );
    }

    /// Test lambda path generation edge cases
    #[test]
    fn test_lambda_path_single() {
        let s = array![[1.0, 0.5], [0.5, 1.0]];
        let path = generate_lambda_path(&s.view(), 1, 0.01).expect("path generation failed");
        assert_eq!(path.len(), 1);
        assert!((path[0] - 0.5).abs() < 1e-10);
    }

    /// Test error handling in model selection
    #[test]
    fn test_model_selection_errors() {
        let s = array![[1.0, 0.5], [0.5, 1.0]];

        // Invalid ratio
        let result = generate_lambda_path(&s.view(), 10, 0.0);
        assert!(result.is_err());

        let result = generate_lambda_path(&s.view(), 10, 1.0);
        assert!(result.is_err());

        // Invalid gamma
        let theta = array![[1.0, 0.0], [0.0, 1.0]];
        let result = compute_ebic(&theta.view(), &s.view(), 50, 1.5);
        assert!(result.is_err());
    }
}
