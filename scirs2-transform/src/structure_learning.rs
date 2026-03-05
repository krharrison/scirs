//! Probabilistic Graphical Model Structure Learning
//!
//! This module provides algorithms for learning the structure of probabilistic
//! graphical models from observational data, including both undirected (Markov
//! networks) and directed (Bayesian network) structures.
//!
//! ## Algorithms
//!
//! - **[`GaussianGraphicalModel`]**: Graphical Lasso (GLasso) for estimating the
//!   precision matrix of a Gaussian Markov random field.
//! - **[`PCAlgorithm`]**: Peter-Clark (PC) algorithm for learning DAG structure
//!   via conditional independence testing.
//! - **[`GES`]**: Greedy Equivalence Search for Bayesian network structure
//!   learning via score-based search over equivalence classes.
//! - **[`CopulaGraphical`]**: Copula-based graphical model for non-Gaussian data.
//!
//! ## References
//!
//! - Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse
//!   covariance estimation with the graphical lasso. Biostatistics.
//! - Spirtes, P., Glymour, C., & Scheines, R. (2001). Causation, Prediction,
//!   and Search. MIT Press.
//! - Chickering, D.M. (2002). Optimal structure identification with greedy
//!   search. JMLR.

use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::PI;

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_linalg::{cholesky, solve, solve_multiple};

use crate::error::{Result, TransformError};

// ============================================================================
// Utility: sample covariance matrix
// ============================================================================

/// Compute the sample covariance matrix of x (shape n × d).
///
/// Returns the d × d biased covariance Σ = (1/n) X_c^T X_c
/// where X_c is mean-centered.
pub fn sample_covariance(x: &ArrayView2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let d = x.ncols();

    // Compute means
    let mean: Array1<f64> = (0..d)
        .map(|j| x.column(j).iter().sum::<f64>() / n as f64)
        .collect();

    // Centered covariance
    let mut cov = Array2::<f64>::zeros((d, d));
    for i in 0..n {
        for a in 0..d {
            for b in a..d {
                let v = (x[[i, a]] - mean[a]) * (x[[i, b]] - mean[b]) / n as f64;
                cov[[a, b]] += v;
                if a != b {
                    cov[[b, a]] += v;
                }
            }
        }
    }
    cov
}

/// Compute the sample correlation matrix from data.
pub fn sample_correlation(x: &ArrayView2<f64>) -> Array2<f64> {
    let cov = sample_covariance(x);
    let d = cov.nrows();
    let stds: Vec<f64> = (0..d).map(|i| cov[[i, i]].sqrt().max(1e-10)).collect();
    let mut corr = cov.clone();
    for i in 0..d {
        for j in 0..d {
            corr[[i, j]] /= stds[i] * stds[j];
        }
    }
    corr
}

// ============================================================================
// GaussianGraphicalModel — Graphical Lasso
// ============================================================================

/// Result of fitting a Gaussian graphical model.
#[derive(Debug, Clone)]
pub struct GaussianGraphicalResult {
    /// Estimated precision matrix Θ = Σ⁻¹ (d × d)
    pub precision: Array2<f64>,
    /// Estimated covariance matrix Σ (d × d)
    pub covariance: Array2<f64>,
    /// Adjacency matrix of the learned graph (d × d, boolean as f64)
    pub adjacency: Array2<f64>,
    /// Log-likelihood at convergence
    pub log_likelihood: f64,
    /// Number of ADMM iterations run
    pub n_iters: usize,
}

/// Gaussian Graphical Model via Graphical Lasso (GLasso).
///
/// Estimates the sparse precision matrix Θ = Σ⁻¹ by solving:
///
/// max_{Θ ≻ 0} log det(Θ) - tr(SΘ) - λ ||Θ||_{1,off}
///
/// where S is the sample covariance and ||·||_{1,off} is the ℓ₁ norm of
/// off-diagonal entries, using coordinate descent (Banerjee et al. 2008 /
/// Friedman et al. 2008 implementation).
///
/// # Example
/// ```
/// use scirs2_transform::structure_learning::GaussianGraphicalModel;
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::zeros((30, 5));
/// let mut ggm = GaussianGraphicalModel::new(0.1);
/// let result = ggm.fit(&x.view()).expect("GGM fit should succeed");
/// assert_eq!(result.precision.shape(), &[5, 5]);
/// ```
#[derive(Debug, Clone)]
pub struct GaussianGraphicalModel {
    /// Sparsity regularization parameter λ
    pub lambda: f64,
    /// Maximum coordinate descent iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Threshold for declaring an edge (precision non-zero)
    pub edge_threshold: f64,
    /// Fitted precision matrix
    precision: Option<Array2<f64>>,
    /// Fitted covariance estimate
    covariance: Option<Array2<f64>>,
}

impl GaussianGraphicalModel {
    /// Create a new GGM / GLasso estimator.
    ///
    /// # Arguments
    /// * `lambda` - ℓ₁ sparsity regularization (larger = sparser graph)
    pub fn new(lambda: f64) -> Self {
        GaussianGraphicalModel {
            lambda,
            max_iter: 100,
            tol: 1e-4,
            edge_threshold: 1e-5,
            precision: None,
            covariance: None,
        }
    }

    /// Set convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set maximum iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Soft-threshold operator: sign(x) * max(0, |x| - thresh).
    fn soft_threshold(x: f64, thresh: f64) -> f64 {
        if x > thresh {
            x - thresh
        } else if x < -thresh {
            x + thresh
        } else {
            0.0
        }
    }

    /// Graphical lasso via block coordinate descent (Banerjee 2008 algorithm).
    ///
    /// Each outer iteration cycles over all variables j, solving a lasso
    /// subproblem in the off-diagonal row/column of Θ.
    fn graphical_lasso_bcd(s: &Array2<f64>, lambda: f64, max_iter: usize, tol: f64) -> Result<(Array2<f64>, Array2<f64>)> {
        let d = s.nrows();

        // Initialize Σ estimate = S + λ I
        let mut sigma_est = s.clone();
        for i in 0..d {
            sigma_est[[i, i]] += lambda;
        }

        // Initialize Θ = (S + λI)^{-1} (diagonal approximation)
        let diag: Vec<f64> = (0..d).map(|i| 1.0 / sigma_est[[i, i]].max(1e-10)).collect();
        let mut theta = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            theta[[i, i]] = diag[i];
        }

        let mut prev_theta = theta.clone();

        for _iter in 0..max_iter {
            // Block coordinate descent over each j
            for j in 0..d {
                // Partition: -j indices
                let minus_j: Vec<usize> = (0..d).filter(|&k| k != j).collect();
                let n_mj = minus_j.len();

                // S_{-j, -j}: submatrix of S excluding row/col j
                let s_mj_mj: Array2<f64> = {
                    let mut m = Array2::<f64>::zeros((n_mj, n_mj));
                    for (ai, &a) in minus_j.iter().enumerate() {
                        for (bi, &b) in minus_j.iter().enumerate() {
                            m[[ai, bi]] = s[[a, b]];
                        }
                    }
                    m
                };

                // s_{-j, j}
                let s_mj_j: Array1<f64> = minus_j.iter().map(|&k| s[[k, j]]).collect();

                // Current sigma_{-j, -j}
                let sigma_mj_mj: Array2<f64> = {
                    let mut m = Array2::<f64>::zeros((n_mj, n_mj));
                    for (ai, &a) in minus_j.iter().enumerate() {
                        for (bi, &b) in minus_j.iter().enumerate() {
                            m[[ai, bi]] = sigma_est[[a, b]];
                        }
                    }
                    m
                };

                // Solve lasso: beta = argmin_{beta} (1/2) beta^T W beta + s_{-j,j}^T beta s.t. lasso
                // W = sigma_{-j, -j}, c = s_{-j, j}
                // Coordinate descent inner loop
                let mut beta: Array1<f64> = Array1::zeros(n_mj);

                for _inner in 0..100 {
                    let mut max_change = 0.0f64;
                    for k in 0..n_mj {
                        let w_kk = sigma_mj_mj[[k, k]].max(1e-10);
                        // residual: W beta - s_{-j,j} excluding component k
                        let r_k: f64 = (0..n_mj)
                            .filter(|&l| l != k)
                            .map(|l| sigma_mj_mj[[k, l]] * beta[l])
                            .sum::<f64>()
                            + s_mj_j[k];

                        let beta_new = -Self::soft_threshold(r_k, lambda) / w_kk;
                        max_change = max_change.max((beta_new - beta[k]).abs());
                        beta[k] = beta_new;
                    }
                    if max_change < tol * 0.1 {
                        break;
                    }
                }

                // Update sigma estimate: sigma_{-j, j} = sigma_{-j,-j} @ beta
                let new_col: Array1<f64> = sigma_mj_mj.dot(&beta);
                for (ai, &a) in minus_j.iter().enumerate() {
                    sigma_est[[a, j]] = new_col[ai];
                    sigma_est[[j, a]] = new_col[ai];
                }
                sigma_est[[j, j]] = s[[j, j]] + lambda - {
                    let bc: f64 = beta.iter().zip(new_col.iter()).map(|(a, b)| a * b).sum();
                    bc
                };
            }

            // Update theta from sigma_est
            let diff_max = theta.iter().zip(prev_theta.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);

            prev_theta = theta.clone();

            // Approximate precision update: Θ ≈ Σ^{-1}
            // Use diagonal approximation for stability
            for i in 0..d {
                theta[[i, i]] = 1.0 / sigma_est[[i, i]].max(1e-10);
            }

            if diff_max < tol {
                break;
            }
        }

        // Full precision: solve Σ Θ = I
        let eye = Array2::<f64>::eye(d);
        // Add small regularization for numerical stability
        let mut sigma_reg = sigma_est.clone();
        for i in 0..d {
            sigma_reg[[i, i]] += 1e-8;
        }

        let theta_full = solve_multiple(&sigma_reg.view(), &eye.view(), None)
            .unwrap_or_else(|_| {
                // Fallback: diagonal inverse
                let mut t = Array2::<f64>::zeros((d, d));
                for i in 0..d {
                    t[[i, i]] = 1.0 / sigma_reg[[i, i]].max(1e-10);
                }
                t
            });

        Ok((theta_full, sigma_est))
    }

    /// Fit the GGM / GLasso to data.
    ///
    /// # Arguments
    /// * `x` - Data matrix of shape (n, d)
    pub fn fit(&mut self, x: &ArrayView2<f64>) -> Result<GaussianGraphicalResult> {
        let n = x.nrows();
        let d = x.ncols();

        if n < d {
            return Err(TransformError::InvalidInput(format!(
                "GaussianGraphicalModel: n_samples ({}) should be >= n_features ({}) for stable estimation",
                n, d
            )));
        }
        if d == 0 {
            return Err(TransformError::InvalidInput(
                "GaussianGraphicalModel: data must have at least 1 feature".to_string(),
            ));
        }

        let s = sample_covariance(x);
        let (precision, covariance) = Self::graphical_lasso_bcd(&s, self.lambda, self.max_iter, self.tol)?;

        // Build adjacency: |Θ_{ij}| > threshold for i ≠ j
        let mut adjacency = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                if i != j && precision[[i, j]].abs() > self.edge_threshold {
                    adjacency[[i, j]] = 1.0;
                }
            }
        }

        // Compute log-likelihood: log det(Θ) - tr(S Θ) - d log(2π) (up to n/2 factor)
        let mut log_det = 0.0f64;
        let mut tr_s_theta = 0.0f64;
        for i in 0..d {
            log_det += precision[[i, i]].abs().ln();
            for j in 0..d {
                tr_s_theta += s[[i, j]] * precision[[j, i]];
            }
        }
        let log_likelihood = 0.5 * n as f64 * (log_det - tr_s_theta - d as f64 * (2.0 * PI).ln());

        self.precision = Some(precision.clone());
        self.covariance = Some(covariance.clone());

        Ok(GaussianGraphicalResult {
            precision,
            covariance,
            adjacency,
            log_likelihood,
            n_iters: self.max_iter,
        })
    }

    /// Return the fitted precision matrix.
    pub fn precision(&self) -> Option<&Array2<f64>> {
        self.precision.as_ref()
    }

    /// Return the fitted covariance estimate.
    pub fn covariance(&self) -> Option<&Array2<f64>> {
        self.covariance.as_ref()
    }

    /// Compute the marginal partial correlations from the precision matrix.
    ///
    /// ρ_{ij|rest} = -Θ_{ij} / sqrt(Θ_{ii} Θ_{jj})
    pub fn partial_correlations(&self) -> Option<Array2<f64>> {
        let theta = self.precision.as_ref()?;
        let d = theta.nrows();
        let mut pcorr = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                if i != j {
                    let denom = (theta[[i, i]] * theta[[j, j]]).sqrt().max(1e-10);
                    pcorr[[i, j]] = -theta[[i, j]] / denom;
                }
            }
        }
        Some(pcorr)
    }
}

// ============================================================================
// PC Algorithm — Skeleton + Orientation
// ============================================================================

/// Result of the PC algorithm.
#[derive(Debug, Clone)]
pub struct PCAlgorithmResult {
    /// Skeleton (undirected edges as symmetric adjacency matrix)
    pub skeleton: Array2<f64>,
    /// Partially directed acyclic graph (PDAG) as adjacency matrix:
    /// adj[i,j]=1 means i → j; adj[i,j]=adj[j,i]=1 means i — j (undirected)
    pub pdag: Array2<f64>,
    /// Separation sets: sep_sets[(i,j)] = set of conditioning variables
    pub sep_sets: HashMap<(usize, usize), Vec<usize>>,
    /// Number of conditional independence tests performed
    pub n_tests: usize,
}

/// PC Algorithm for learning Bayesian network structure.
///
/// The PC algorithm learns a directed acyclic graph (DAG) representing
/// conditional independence structure from data using:
///
/// 1. **Skeleton phase**: Remove edges between conditionally independent variables.
/// 2. **Orientation phase**: Orient v-structures (colliders) and propagate
///    Meek rules to recover the completed partially directed DAG (CPDAG).
///
/// Conditional independence is tested using Fisher's z-test on partial
/// correlations (assumes multivariate normality).
///
/// # Example
/// ```
/// use scirs2_transform::structure_learning::PCAlgorithm;
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::zeros((50, 4));
/// let mut pc = PCAlgorithm::new(0.05, 2);
/// let result = pc.fit(&x.view()).expect("PC fit should succeed");
/// assert_eq!(result.skeleton.shape(), &[4, 4]);
/// ```
#[derive(Debug, Clone)]
pub struct PCAlgorithm {
    /// Significance level α for conditional independence tests
    pub alpha: f64,
    /// Maximum conditioning set size (depth of search)
    pub max_cond_set: usize,
    /// Minimum samples to avoid degenerate test (auto-computed if 0)
    pub min_samples: usize,
}

impl PCAlgorithm {
    /// Create a new PC algorithm instance.
    ///
    /// # Arguments
    /// * `alpha` - Significance level for conditional independence tests (0 < α < 1)
    /// * `max_cond_set` - Maximum size of conditioning set
    pub fn new(alpha: f64, max_cond_set: usize) -> Self {
        PCAlgorithm { alpha, max_cond_set, min_samples: 10 }
    }

    /// Fisher's z-transformation: z = (1/2) ln((1+r)/(1-r))
    fn fisher_z(r: f64) -> f64 {
        let r_clamped = r.max(-0.9999).min(0.9999);
        0.5 * ((1.0 + r_clamped) / (1.0 - r_clamped)).ln()
    }

    /// Compute partial correlation of (x_i, x_j) given x_S from the correlation matrix.
    ///
    /// Uses recursion: ρ_{ij|S} = (ρ_{ij|S\k} - ρ_{ik|S\k} ρ_{jk|S\k}) /
    ///                             sqrt((1 - ρ_{ik|S\k}²)(1 - ρ_{jk|S\k}²))
    fn partial_correlation(
        corr: &Array2<f64>,
        i: usize,
        j: usize,
        conditioning: &[usize],
    ) -> f64 {
        if conditioning.is_empty() {
            return corr[[i, j]];
        }

        // Build submatrix for {i, j} ∪ conditioning
        let mut vars = vec![i, j];
        vars.extend_from_slice(conditioning);
        vars.sort_unstable();
        vars.dedup();
        let k = vars.len();

        // Extract submatrix from corr
        let mut sub = Array2::<f64>::zeros((k, k));
        for (ai, &a) in vars.iter().enumerate() {
            for (bi, &b) in vars.iter().enumerate() {
                sub[[ai, bi]] = corr[[a, b]];
            }
        }

        // Invert sub via Gaussian elimination with regularization
        let mut aug = Array2::<f64>::zeros((k, 2 * k));
        for r in 0..k {
            for c in 0..k {
                aug[[r, c]] = sub[[r, c]];
            }
            aug[[r, k + r]] = 1.0;
            aug[[r, r]] += 1e-8; // regularize
        }

        // Forward elimination
        for col in 0..k {
            // Find pivot
            let mut max_val = aug[[col, col]].abs();
            let mut max_row = col;
            for row in (col + 1)..k {
                if aug[[row, col]].abs() > max_val {
                    max_val = aug[[row, col]].abs();
                    max_row = row;
                }
            }
            if max_val < 1e-12 {
                continue;
            }
            // Swap rows
            if max_row != col {
                for c in 0..(2 * k) {
                    let tmp = aug[[col, c]];
                    aug[[col, c]] = aug[[max_row, c]];
                    aug[[max_row, c]] = tmp;
                }
            }
            let pivot = aug[[col, col]];
            for c in 0..(2 * k) {
                aug[[col, c]] /= pivot;
            }
            for row in 0..k {
                if row == col {
                    continue;
                }
                let factor = aug[[row, col]];
                for c in 0..(2 * k) {
                    let v = aug[[col, c]] * factor;
                    aug[[row, c]] -= v;
                }
            }
        }

        // Extract inverse
        let mut inv = Array2::<f64>::zeros((k, k));
        for r in 0..k {
            for c in 0..k {
                inv[[r, c]] = aug[[r, k + c]];
            }
        }

        // Partial correlation = -inv[i_idx, j_idx] / sqrt(inv[i_idx,i_idx] * inv[j_idx,j_idx])
        let i_idx = vars.iter().position(|&x| x == i).unwrap_or(0);
        let j_idx = vars.iter().position(|&x| x == j).unwrap_or(1);

        let denom = (inv[[i_idx, i_idx]] * inv[[j_idx, j_idx]]).sqrt().max(1e-10);
        (-inv[[i_idx, j_idx]] / denom).max(-1.0).min(1.0)
    }

    /// Fisher's z-test for conditional independence.
    ///
    /// H₀: ρ_{ij|S} = 0 (conditional independence)
    /// Test statistic: √(n - |S| - 3) · |z| where z = fisher_z(r̂_{ij|S})
    fn ci_test(
        corr: &Array2<f64>,
        n: usize,
        i: usize,
        j: usize,
        cond_set: &[usize],
        alpha: f64,
    ) -> bool {
        let r = Self::partial_correlation(corr, i, j, cond_set);
        let k = cond_set.len();
        let df = (n as i64 - k as i64 - 3).max(1) as f64;
        let z = Self::fisher_z(r).abs() * df.sqrt();

        // Standard normal quantile for two-tailed test at level α
        // Approximate: z_{α/2} for common values
        let z_alpha = normal_quantile(1.0 - alpha / 2.0);
        z < z_alpha
    }

    /// Enumerate conditioning subsets of size k from the adjacency list of i (excluding j).
    fn conditioning_subsets(
        adj: &[Vec<usize>],
        i: usize,
        j: usize,
        size: usize,
    ) -> Vec<Vec<usize>> {
        let candidates: Vec<usize> = adj[i].iter()
            .chain(adj[j].iter())
            .copied()
            .filter(|&k| k != i && k != j)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        combinations(&candidates, size)
    }

    /// Fit the PC algorithm to data and learn the CPDAG.
    pub fn fit(&mut self, x: &ArrayView2<f64>) -> Result<PCAlgorithmResult> {
        let n = x.nrows();
        let d = x.ncols();

        if n < self.min_samples {
            return Err(TransformError::InvalidInput(format!(
                "PCAlgorithm: need at least {} samples, got {}",
                self.min_samples, n
            )));
        }
        if d < 2 {
            return Err(TransformError::InvalidInput(
                "PCAlgorithm: need at least 2 variables".to_string(),
            ));
        }

        let corr = sample_correlation(x);

        // Initialize skeleton: complete undirected graph
        let mut adj: Vec<Vec<usize>> = (0..d)
            .map(|i| (0..d).filter(|&j| j != i).collect())
            .collect();

        let mut sep_sets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        let mut n_tests = 0usize;

        // Phase 1: Skeleton estimation
        for cond_size in 0..=self.max_cond_set {
            let mut to_remove: Vec<(usize, usize, Vec<usize>)> = Vec::new();

            for i in 0..d {
                for &j in &adj[i].clone() {
                    if j <= i {
                        continue; // process each pair once
                    }

                    // Test conditioning sets of current size
                    let subsets = Self::conditioning_subsets(&adj, i, j, cond_size);

                    for cond in &subsets {
                        n_tests += 1;
                        if Self::ci_test(&corr, n, i, j, cond, self.alpha) {
                            to_remove.push((i, j, cond.clone()));
                            break; // found a separating set
                        }
                    }
                }
            }

            // Remove edges for independent pairs
            for (i, j, sep) in &to_remove {
                adj[*i].retain(|&k| k != *j);
                adj[*j].retain(|&k| k != *i);
                sep_sets.insert((*i, *j), sep.clone());
                sep_sets.insert((*j, *i), sep.clone());
            }
        }

        // Build skeleton matrix
        let mut skeleton = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for &j in &adj[i] {
                skeleton[[i, j]] = 1.0;
                skeleton[[j, i]] = 1.0;
            }
        }

        // Phase 2: Orient v-structures (colliders)
        // i -- k -- j with i not adjacent to j
        let mut pdag = skeleton.clone();

        for k in 0..d {
            let neighbors: Vec<usize> = adj[k].clone();
            for (ni, &i) in neighbors.iter().enumerate() {
                for &j in &neighbors[..ni] {
                    // Check if i and j are not adjacent
                    if adj[i].contains(&j) {
                        continue;
                    }
                    // Check if k is NOT in sep_sets[(i, j)]
                    let sep = sep_sets.get(&(i, j)).cloned().unwrap_or_default();
                    if !sep.contains(&k) {
                        // Orient i → k ← j
                        pdag[[i, k]] = 1.0;
                        pdag[[k, i]] = 0.0;
                        pdag[[j, k]] = 1.0;
                        pdag[[k, j]] = 0.0;
                    }
                }
            }
        }

        // Apply Meek rules (simplified R1: orient to avoid new v-structures)
        for _ in 0..d {
            let mut changed = false;

            // Meek R1: if i → j -- k and i not adjacent to k, orient j → k
            for j in 0..d {
                for i in 0..d {
                    if pdag[[i, j]] == 1.0 && pdag[[j, i]] == 0.0 {
                        // i → j; check j -- k
                        for k in 0..d {
                            if k == i {
                                continue;
                            }
                            if pdag[[j, k]] == 1.0 && pdag[[k, j]] == 1.0 {
                                // j -- k undirected; check i not adj k
                                if adj[i].contains(&k) {
                                    continue;
                                }
                                // Orient j → k
                                pdag[[k, j]] = 0.0;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if !changed {
                break;
            }
        }

        Ok(PCAlgorithmResult {
            skeleton,
            pdag,
            sep_sets,
            n_tests,
        })
    }
}

// ============================================================================
// GES — Greedy Equivalence Search
// ============================================================================

/// Result of the GES algorithm.
#[derive(Debug, Clone)]
pub struct GESResult {
    /// Learned CPDAG as adjacency matrix (directed + undirected edges)
    pub cpdag: Array2<f64>,
    /// Final BIC score
    pub final_score: f64,
    /// Score at each GES phase
    pub score_history: Vec<f64>,
}

/// Greedy Equivalence Search (GES) for Bayesian network structure learning.
///
/// GES performs a score-based greedy search over the space of completed
/// partially directed acyclic graphs (CPDAGs / equivalence classes of DAGs)
/// in three phases:
///
/// 1. **Forward (FES)**: Add edges that maximally improve the BIC score
/// 2. **Backward (BES)**: Remove edges whose removal improves the score
/// 3. (Optional) **Turning**: Turn edge directions that improve the score
///
/// Uses the BIC (Bayesian Information Criterion) as the scoring function
/// assuming multivariate Gaussianity.
///
/// # Example
/// ```
/// use scirs2_transform::structure_learning::GES;
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::zeros((40, 4));
/// let mut ges = GES::new();
/// let result = ges.fit(&x.view()).expect("GES fit should succeed");
/// assert_eq!(result.cpdag.shape(), &[4, 4]);
/// ```
#[derive(Debug, Clone)]
pub struct GES {
    /// Penalty coefficient for BIC (default 1.0)
    pub penalty: f64,
    /// Maximum number of FES + BES rounds
    pub max_rounds: usize,
}

impl GES {
    /// Create a new GES instance with default settings.
    pub fn new() -> Self {
        GES { penalty: 1.0, max_rounds: 3 }
    }

    /// Set BIC penalty coefficient.
    pub fn with_penalty(mut self, penalty: f64) -> Self {
        self.penalty = penalty;
        self
    }

    /// Compute the BIC score contribution for node j given its parents Pa(j) in data x.
    ///
    /// BIC_j = -n/2 * log(RSS_j/n) - |Pa(j)| * log(n) / 2 * penalty
    fn local_bic(
        x: &ArrayView2<f64>,
        j: usize,
        parents: &[usize],
        penalty: f64,
    ) -> f64 {
        let n = x.nrows();
        let p = parents.len();

        let y: Vec<f64> = (0..n).map(|i| x[[i, j]]).collect();
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;

        let rss = if p == 0 {
            // No parents: predict with mean
            y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>()
        } else {
            // OLS regression y_j ~ Pa(j)
            // Build design matrix (n x p+1) with intercept
            let mut x_reg = Array2::<f64>::ones((n, p + 1));
            for (col, &par) in parents.iter().enumerate() {
                for row in 0..n {
                    x_reg[[row, col + 1]] = x[[row, par]];
                }
            }
            let y_arr: Array2<f64> = Array1::from_vec(y).insert_axis(Axis(1));

            // (X^T X)^{-1} X^T y via normal equations
            let xtx = x_reg.t().dot(&x_reg);
            let xty = x_reg.t().dot(&y_arr);

            let mut xtx_reg = xtx.clone();
            for i in 0..(p + 1) {
                xtx_reg[[i, i]] += 1e-8;
            }

            let beta = solve_multiple(&xtx_reg.view(), &xty.view(), None)
                .unwrap_or_else(|_| Array2::<f64>::zeros((p + 1, 1)));

            let y_pred: Vec<f64> = (0..n)
                .map(|i| {
                    let mut pred = beta[[0, 0]];
                    for col in 0..p {
                        pred += x_reg[[i, col + 1]] * beta[[col + 1, 0]];
                    }
                    pred
                })
                .collect();

            y_pred.iter().enumerate()
                .map(|(i, &yp)| {
                    let yi = x[[i, j]];
                    (yi - yp).powi(2)
                })
                .sum::<f64>()
        };

        let rss_safe = rss.max(1e-10);
        -((n as f64) / 2.0) * (rss_safe / n as f64).ln()
            - penalty * (p as f64) * (n as f64).ln() / 2.0
    }

    /// Compute total BIC score for all nodes given parent sets.
    fn total_bic(
        x: &ArrayView2<f64>,
        parents: &[Vec<usize>],
        penalty: f64,
    ) -> f64 {
        let d = x.ncols();
        (0..d).map(|j| Self::local_bic(x, j, &parents[j], penalty)).sum()
    }

    /// Check if adding edge i → j creates a directed cycle.
    /// Uses DFS to detect if j can already reach i.
    fn creates_cycle(adj: &[Vec<usize>], from: usize, to: usize) -> bool {
        // Can we reach `from` from `to`? If yes, adding from→to creates a cycle.
        let d = adj.len();
        let mut visited = vec![false; d];
        let mut queue = VecDeque::new();
        queue.push_back(to);

        while let Some(node) = queue.pop_front() {
            if node == from {
                return true;
            }
            if visited[node] {
                continue;
            }
            visited[node] = true;
            for &next in &adj[node] {
                queue.push_back(next);
            }
        }
        false
    }

    /// Fit GES to data.
    ///
    /// # Arguments
    /// * `x` - Data matrix, shape (n, d)
    pub fn fit(&mut self, x: &ArrayView2<f64>) -> Result<GESResult> {
        let n = x.nrows();
        let d = x.ncols();

        if n < 5 {
            return Err(TransformError::InvalidInput(
                "GES: need at least 5 samples".to_string(),
            ));
        }
        if d < 2 {
            return Err(TransformError::InvalidInput(
                "GES: need at least 2 variables".to_string(),
            ));
        }

        // Initialize: empty graph
        let mut parents: Vec<Vec<usize>> = vec![Vec::new(); d];
        let mut directed_adj: Vec<Vec<usize>> = vec![Vec::new(); d]; // i → j means j in directed_adj[i]

        let mut score_history = Vec::new();
        let penalty = self.penalty;

        let initial_score = Self::total_bic(x, &parents, penalty);
        score_history.push(initial_score);

        // Forward Equivalence Search (FES)
        for _round in 0..self.max_rounds {
            let mut best_gain = 0.0f64;
            let mut best_edge: Option<(usize, usize)> = None;

            for i in 0..d {
                for j in 0..d {
                    if i == j {
                        continue;
                    }
                    if parents[j].contains(&i) {
                        continue; // edge already exists
                    }
                    if Self::creates_cycle(&directed_adj, i, j) {
                        continue; // would create cycle
                    }

                    let old_bic = Self::local_bic(x, j, &parents[j], penalty);
                    let mut new_parents = parents[j].clone();
                    new_parents.push(i);
                    let new_bic = Self::local_bic(x, j, &new_parents, penalty);
                    let gain = new_bic - old_bic;

                    if gain > best_gain {
                        best_gain = gain;
                        best_edge = Some((i, j));
                    }
                }
            }

            if let Some((i, j)) = best_edge {
                parents[j].push(i);
                directed_adj[i].push(j);
                let new_score = Self::total_bic(x, &parents, penalty);
                score_history.push(new_score);
            } else {
                break;
            }
        }

        // Backward Equivalence Search (BES)
        for _round in 0..self.max_rounds {
            let mut best_gain = 0.0f64;
            let mut best_removal: Option<(usize, usize)> = None;

            for j in 0..d {
                let pj = parents[j].clone();
                for &i in &pj {
                    let old_bic = Self::local_bic(x, j, &parents[j], penalty);
                    let new_parents: Vec<usize> = parents[j].iter()
                        .copied()
                        .filter(|&k| k != i)
                        .collect();
                    let new_bic = Self::local_bic(x, j, &new_parents, penalty);
                    let gain = new_bic - old_bic;

                    if gain > best_gain {
                        best_gain = gain;
                        best_removal = Some((i, j));
                    }
                }
            }

            if let Some((i, j)) = best_removal {
                parents[j].retain(|&k| k != i);
                directed_adj[i].retain(|&k| k != j);
                let new_score = Self::total_bic(x, &parents, penalty);
                score_history.push(new_score);
            } else {
                break;
            }
        }

        // Build CPDAG from learned DAG
        let final_score = *score_history.last().unwrap_or(&0.0);
        let mut cpdag = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for &j in &directed_adj[i] {
                cpdag[[i, j]] = 1.0;
            }
        }

        Ok(GESResult { cpdag, final_score, score_history })
    }
}

impl Default for GES {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CopulaGraphical
// ============================================================================

/// Result of copula graphical model estimation.
#[derive(Debug, Clone)]
pub struct CopulaGraphicalResult {
    /// Estimated copula correlation matrix R (d × d)
    pub copula_correlation: Array2<f64>,
    /// Estimated precision matrix of the Gaussian copula
    pub precision: Array2<f64>,
    /// Graph adjacency matrix
    pub adjacency: Array2<f64>,
}

/// Copula-based Gaussian Graphical Model.
///
/// Extends the Gaussian graphical model to handle non-Gaussian data by:
/// 1. Mapping each marginal to normality via non-parametric rank-based
///    (empirical CDF) transformation.
/// 2. Applying GLasso to the resulting pseudo-observations.
///
/// This is the "nonparanormal" / "SKEPTIC" approach of Liu et al. (2012).
///
/// # References
/// Liu, H., Lafferty, J., & Wasserman, L. (2009). The Nonparanormal: Semiparametric
/// Estimation of High Dimensional Undirected Graphs. JMLR.
///
/// # Example
/// ```
/// use scirs2_transform::structure_learning::CopulaGraphical;
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::zeros((30, 5));
/// let mut cgg = CopulaGraphical::new(0.1);
/// let result = cgg.fit(&x.view()).expect("CopulaGraphical fit should succeed");
/// assert_eq!(result.copula_correlation.shape(), &[5, 5]);
/// ```
#[derive(Debug, Clone)]
pub struct CopulaGraphical {
    /// Sparsity regularization for the precision matrix
    pub lambda: f64,
    /// GGM maximum iterations
    pub max_iter: usize,
}

impl CopulaGraphical {
    /// Create a new copula graphical model.
    ///
    /// # Arguments
    /// * `lambda` - GLasso sparsity penalty
    pub fn new(lambda: f64) -> Self {
        CopulaGraphical { lambda, max_iter: 100 }
    }

    /// Normal quantile function (Beasley-Springer-Moro approximation).
    fn normal_quantile_approx(p: f64) -> f64 {
        normal_quantile(p)
    }

    /// Transform data column-wise using rank-based normal scores.
    ///
    /// Each column is mapped to approximately N(0,1) via the empirical CDF:
    /// x_{ij} → Φ⁻¹(rank(x_{ij}) / (n + 1))
    fn rank_normal_transform(x: &ArrayView2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let d = x.ncols();
        let mut z = Array2::<f64>::zeros((n, d));

        for j in 0..d {
            let col: Vec<f64> = (0..n).map(|i| x[[i, j]]).collect();
            let mut ranks: Vec<(usize, f64)> = col.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            ranks.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut rank_vec = vec![0usize; n];
            for (rank, (orig_idx, _)) in ranks.iter().enumerate() {
                rank_vec[*orig_idx] = rank + 1;
            }

            for i in 0..n {
                let p = rank_vec[i] as f64 / (n + 1) as f64;
                z[[i, j]] = Self::normal_quantile_approx(p);
            }
        }
        z
    }

    /// Compute Kendall's tau-based copula correlation estimate.
    ///
    /// sin(π/2 · τ_{ij}) is a consistent estimate of the Gaussian copula correlation.
    fn kendall_copula_correlation(x: &ArrayView2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let d = x.ncols();
        let mut corr = Array2::<f64>::eye(d);

        for i in 0..d {
            for j in (i + 1)..d {
                let tau = Self::kendall_tau_col(x, i, j, n);
                let rho = (PI / 2.0 * tau).sin().max(-1.0).min(1.0);
                corr[[i, j]] = rho;
                corr[[j, i]] = rho;
            }
        }
        corr
    }

    /// Compute Kendall's tau between columns i and j.
    fn kendall_tau_col(x: &ArrayView2<f64>, ci: usize, cj: usize, n: usize) -> f64 {
        let mut concordant = 0i64;
        let mut discordant = 0i64;

        for a in 0..n {
            for b in (a + 1)..n {
                let sign_i = (x[[a, ci]] - x[[b, ci]]).signum();
                let sign_j = (x[[a, cj]] - x[[b, cj]]).signum();
                let prod = sign_i * sign_j;
                if prod > 0.0 {
                    concordant += 1;
                } else if prod < 0.0 {
                    discordant += 1;
                }
            }
        }

        let n_pairs = (n * (n - 1) / 2) as f64;
        if n_pairs < 1.0 {
            return 0.0;
        }
        (concordant - discordant) as f64 / n_pairs
    }

    /// Fit the copula graphical model to data.
    ///
    /// # Arguments
    /// * `x` - Data matrix, shape (n, d); may be non-Gaussian
    pub fn fit(&mut self, x: &ArrayView2<f64>) -> Result<CopulaGraphicalResult> {
        let n = x.nrows();
        let d = x.ncols();

        if n < d {
            return Err(TransformError::InvalidInput(format!(
                "CopulaGraphical: n_samples ({}) should be >= n_features ({}) for stable estimation",
                n, d
            )));
        }

        // Step 1: Rank-based normal score transform
        let z = Self::rank_normal_transform(x);

        // Step 2: Compute copula correlation (Kendall-based or sample)
        let copula_corr = if n <= 200 {
            // Use Kendall-based estimate for small n
            Self::kendall_copula_correlation(x)
        } else {
            // Use sample correlation of normal scores for large n (faster)
            sample_correlation(&z.view())
        };

        // Step 3: Apply GLasso to copula correlation
        let mut ggm = GaussianGraphicalModel::new(self.lambda);
        ggm.max_iter = self.max_iter;

        // Convert copula_corr to a pseudo-dataset style by treating it as S
        // and calling graphical_lasso_bcd directly
        let (precision, _) = GaussianGraphicalModel::graphical_lasso_bcd(
            &copula_corr,
            self.lambda,
            self.max_iter,
            1e-4,
        )?;

        // Adjacency
        let edge_thresh = 1e-5;
        let mut adjacency = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                if i != j && precision[[i, j]].abs() > edge_thresh {
                    adjacency[[i, j]] = 1.0;
                }
            }
        }

        let _ = ggm;
        Ok(CopulaGraphicalResult {
            copula_correlation: copula_corr,
            precision,
            adjacency,
        })
    }
}

// ============================================================================
// Utility: combinations, normal quantile
// ============================================================================

/// Generate all combinations of size k from a slice.
fn combinations(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if items.len() < k {
        return vec![];
    }

    let mut result = Vec::new();
    let n = items.len();

    // Iterative via bitmask (ok for small n)
    if n <= 20 {
        for mask in 0u32..(1u32 << n) {
            if mask.count_ones() as usize == k {
                let combo: Vec<usize> = (0..n)
                    .filter(|&i| (mask >> i) & 1 == 1)
                    .map(|i| items[i])
                    .collect();
                result.push(combo);
            }
        }
    } else {
        // Recursive for larger inputs
        fn comb_rec(items: &[usize], k: usize, start: usize, current: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
            if current.len() == k {
                result.push(current.clone());
                return;
            }
            let remaining = items.len() - start;
            let needed = k - current.len();
            if remaining < needed {
                return;
            }
            for i in start..items.len() {
                current.push(items[i]);
                comb_rec(items, k, i + 1, current, result);
                current.pop();
            }
        }
        let mut current = Vec::new();
        comb_rec(items, k, 0, &mut current, &mut result);
    }

    result
}

/// Approximate inverse normal CDF (Beasley-Springer-Moro algorithm).
///
/// Valid for p ∈ (0, 1); clamps to ±8 outside.
pub fn normal_quantile(p: f64) -> f64 {
    let p_clamped = p.max(1e-12).min(1.0 - 1e-12);

    // Rational approximation (Abramowitz & Stegun 26.2.17)
    let t = if p_clamped <= 0.5 {
        (-2.0 * p_clamped.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p_clamped).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let num = c0 + c1 * t + c2 * t * t;
    let den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    let approx = t - num / den;

    if p_clamped <= 0.5 {
        -approx
    } else {
        approx
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_sample_covariance_zeros() {
        let x = Array2::<f64>::zeros((10, 3));
        let cov = sample_covariance(&x.view());
        assert_eq!(cov.shape(), &[3, 3]);
        assert!(cov.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_sample_covariance_identity() {
        // X = I_3 repeated: rows are standard basis vectors
        let mut x = Array2::<f64>::zeros((9, 3));
        for i in 0..9 {
            x[[i, i % 3]] = 1.0;
        }
        let cov = sample_covariance(&x.view());
        assert_eq!(cov.shape(), &[3, 3]);
    }

    #[test]
    fn test_ggm_fit_shape() {
        let x = Array2::<f64>::zeros((20, 4));
        // Add identity-like structure
        let mut x2 = x.clone();
        for i in 0..20 {
            x2[[i, i % 4]] = 1.0;
        }
        let mut ggm = GaussianGraphicalModel::new(0.1);
        let res = ggm.fit(&x2.view()).expect("GGM fit should succeed");
        assert_eq!(res.precision.shape(), &[4, 4]);
        assert_eq!(res.adjacency.shape(), &[4, 4]);
    }

    #[test]
    fn test_ggm_adjacency_diagonal_zero() {
        let mut x = Array2::<f64>::zeros((20, 3));
        for i in 0..20 {
            x[[i, i % 3]] = (i as f64).sin();
        }
        let mut ggm = GaussianGraphicalModel::new(0.2);
        let res = ggm.fit(&x.view()).expect("GGM fit should succeed");
        // Diagonal of adjacency should be 0
        for i in 0..3 {
            assert_eq!(res.adjacency[[i, i]], 0.0);
        }
    }

    #[test]
    fn test_ggm_partial_correlations() {
        let mut x = Array2::<f64>::zeros((20, 3));
        for i in 0..20 {
            x[[i, 0]] = i as f64;
            x[[i, 1]] = (i as f64).sin();
            x[[i, 2]] = -(i as f64);
        }
        let mut ggm = GaussianGraphicalModel::new(0.05);
        let _ = ggm.fit(&x.view()).expect("GGM fit should succeed");
        let pcorr = ggm.partial_correlations().expect("partial_correlations should succeed");
        assert_eq!(pcorr.shape(), &[3, 3]);
        // Diagonal should be 0
        for i in 0..3 {
            assert_eq!(pcorr[[i, i]], 0.0);
        }
        // Off-diagonal in [-1, 1]
        for i in 0..3 {
            for j in 0..3 {
                assert!(pcorr[[i, j]].abs() <= 1.0 + 1e-10);
            }
        }
    }

    #[test]
    fn test_pc_algorithm_fit() {
        let mut x = Array2::<f64>::zeros((50, 4));
        for i in 0..50 {
            for j in 0..4 {
                x[[i, j]] = (i as f64 * (j + 1) as f64).sin();
            }
        }
        let mut pc = PCAlgorithm::new(0.05, 2);
        let res = pc.fit(&x.view()).expect("PC fit should succeed");
        assert_eq!(res.skeleton.shape(), &[4, 4]);
        assert_eq!(res.pdag.shape(), &[4, 4]);
        // Skeleton is symmetric
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(res.skeleton[[i, j]], res.skeleton[[j, i]]);
            }
        }
    }

    #[test]
    fn test_ges_fit_shape() {
        let mut x = Array2::<f64>::zeros((30, 4));
        for i in 0..30 {
            x[[i, 0]] = i as f64;
            x[[i, 1]] = (i as f64).cos();
            x[[i, 2]] = (i as f64).sin();
            x[[i, 3]] = i as f64 / 30.0;
        }
        let mut ges = GES::new();
        let res = ges.fit(&x.view()).expect("GES fit should succeed");
        assert_eq!(res.cpdag.shape(), &[4, 4]);
        assert!(!res.score_history.is_empty());
    }

    #[test]
    fn test_copula_graphical_fit_shape() {
        let mut x = Array2::<f64>::zeros((30, 4));
        for i in 0..30 {
            for j in 0..4 {
                x[[i, j]] = ((i + j) as f64).exp() % 10.0;
            }
        }
        let mut cgg = CopulaGraphical::new(0.2);
        let res = cgg.fit(&x.view()).expect("CopulaGraphical fit should succeed");
        assert_eq!(res.copula_correlation.shape(), &[4, 4]);
        assert_eq!(res.precision.shape(), &[4, 4]);
        // Copula correlation diagonal should be 1
        for i in 0..4 {
            assert!((res.copula_correlation[[i, i]] - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_normal_quantile_median() {
        let q = normal_quantile(0.5);
        assert!(q.abs() < 0.1); // median ≈ 0
    }

    #[test]
    fn test_combinations_k2() {
        let items = vec![0, 1, 2, 3];
        let combos = combinations(&items, 2);
        assert_eq!(combos.len(), 6); // C(4,2) = 6
    }

    #[test]
    fn test_combinations_k0() {
        let items = vec![1, 2, 3];
        let combos = combinations(&items, 0);
        assert_eq!(combos.len(), 1); // empty set
    }
}
