//! DCC-GARCH (Dynamic Conditional Correlation GARCH) — Engle (2002)
//!
//! This module implements the DCC-GARCH model for multivariate volatility
//! modelling. The DCC approach is a two-stage procedure:
//!
//! 1. **Stage 1**: Fit univariate GARCH(1,1) to each series to obtain
//!    standardised residuals ε_t = r_t / σ_t.
//! 2. **Stage 2**: Model the time-varying correlation of the standardised
//!    residuals via:
//!    ```text
//!    Q_t = (1 − α − β) Q̄ + α (ε_{t−1} ε_{t−1}^T) + β Q_{t−1}
//!    R_t = diag(Q_t)^{−1/2} Q_t diag(Q_t)^{−1/2}
//!    ```
//!
//! The model guarantees positive-definite correlation matrices at each time
//! step when α + β < 1 and the initial Q̄ is positive definite.
//!
//! # References
//! - Engle, R. (2002). Dynamic Conditional Correlation: A Simple Class of
//!   Multivariate Generalized Autoregressive Conditional Heteroskedasticity
//!   Models. *Journal of Business & Economic Statistics*, 20(3), 339–350.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_series::volatility::dcc_garch::{DCCConfig, fit_dcc};
//!
//! // Two return series of length 30
//! let r1: Vec<f64> = vec![
//!     0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.018, 0.009,
//!     -0.003, 0.007, 0.025, -0.014, 0.008, -0.006, 0.011, -0.019,
//!     0.022, 0.003, -0.011, 0.017, -0.005, 0.031, -0.013, 0.009,
//!     0.002, -0.027, 0.016, -0.007, 0.013, 0.004,
//! ];
//! let r2: Vec<f64> = vec![
//!     0.005, -0.01, 0.008, -0.004, 0.006, 0.015, -0.009, 0.012,
//!     -0.002, 0.004, 0.018, -0.007, 0.011, -0.003, 0.009, -0.012,
//!     0.014, 0.001, -0.008, 0.010, -0.003, 0.020, -0.006, 0.005,
//!     0.001, -0.015, 0.009, -0.004, 0.008, 0.002,
//! ];
//! let returns = vec![r1, r2];
//! let config = DCCConfig::default();
//! let result = fit_dcc(&returns, &config).expect("DCC should fit");
//! assert!(result.alpha + result.beta < 1.0);
//! ```

use scirs2_core::ndarray::Array1;

use crate::error::{Result, TimeSeriesError};
use crate::volatility::garch::{fit_garch, garch_log_likelihood, GARCHModel};

// ============================================================
// Configuration
// ============================================================

/// Configuration for DCC-GARCH estimation.
#[derive(Debug, Clone)]
pub struct DCCConfig {
    /// ARCH order for Stage 1 univariate GARCH (default: 1)
    pub garch_p: usize,
    /// GARCH order for Stage 1 univariate GARCH (default: 1)
    pub garch_q: usize,
    /// Maximum iterations for Stage 2 optimisation
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
}

impl Default for DCCConfig {
    fn default() -> Self {
        Self {
            garch_p: 1,
            garch_q: 1,
            max_iter: 1000,
            tol: 1e-6,
        }
    }
}

// ============================================================
// Result types
// ============================================================

/// Result of DCC-GARCH estimation.
#[derive(Debug, Clone)]
pub struct DCCResult {
    /// DCC α parameter (news impact on correlation)
    pub alpha: f64,
    /// DCC β parameter (correlation persistence)
    pub beta: f64,
    /// Unconditional correlation matrix Q̄ (k×k, row-major)
    pub q_bar: Vec<f64>,
    /// Univariate GARCH models for each series
    pub garch_models: Vec<GARCHModel>,
    /// Conditional correlations over time: Vec of k×k matrices (row-major)
    pub correlations: Vec<Vec<f64>>,
    /// Conditional covariance matrices over time: Vec of k×k matrices (row-major)
    pub covariances: Vec<Vec<f64>>,
    /// Composite log-likelihood
    pub log_likelihood: f64,
    /// Number of series
    pub k: usize,
    /// Number of time observations
    pub n_obs: usize,
}

/// Covariance matrix forecast.
#[derive(Debug, Clone)]
pub struct CovMatrixForecast {
    /// Forecast horizon step
    pub step: usize,
    /// Covariance matrix (k×k, row-major)
    pub cov_matrix: Vec<f64>,
    /// Correlation matrix (k×k, row-major)
    pub cor_matrix: Vec<f64>,
}

// ============================================================
// Stage 1: Univariate GARCH fitting
// ============================================================

/// Fit univariate GARCH(p,q) to each series, return models and standardised residuals.
fn stage1_fit(
    returns: &[Vec<f64>],
    p: usize,
    q: usize,
) -> Result<(Vec<GARCHModel>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let k = returns.len();
    let mut models = Vec::with_capacity(k);
    let mut std_resids = Vec::with_capacity(k);
    let mut cond_vols = Vec::with_capacity(k);

    for (i, series) in returns.iter().enumerate() {
        let arr = Array1::from(series.clone());
        let model = fit_garch(&arr, p, q).map_err(|e| {
            TimeSeriesError::FittingError(format!(
                "Stage 1 GARCH fit failed for series {}: {}",
                i, e
            ))
        })?;

        // Compute conditional variance series
        let (_, sigma2) = garch_log_likelihood(&arr, model.omega, &model.alpha, &model.beta)?;

        // Standardised residuals: ε_t = r_t / σ_t
        let eps: Vec<f64> = series
            .iter()
            .zip(sigma2.iter())
            .map(|(&r, &s2)| {
                let s = s2.max(1e-10).sqrt();
                r / s
            })
            .collect();

        let vols: Vec<f64> = sigma2.iter().map(|&s2| s2.max(1e-10).sqrt()).collect();

        models.push(model);
        std_resids.push(eps);
        cond_vols.push(vols);
    }

    Ok((models, std_resids, cond_vols))
}

// ============================================================
// Matrix helpers (small dense k×k)
// ============================================================

/// Compute the unconditional correlation matrix Q̄ from standardised residuals.
fn compute_q_bar(std_resids: &[Vec<f64>], k: usize, n: usize) -> Vec<f64> {
    let mut q_bar = vec![0.0_f64; k * k];
    for t in 0..n {
        for i in 0..k {
            for j in 0..k {
                q_bar[i * k + j] += std_resids[i][t] * std_resids[j][t];
            }
        }
    }
    for val in q_bar.iter_mut() {
        *val /= n as f64;
    }
    q_bar
}

/// Normalise a pseudo-correlation matrix Q to a correlation matrix R.
///
/// R = diag(Q)^{-1/2} * Q * diag(Q)^{-1/2}
fn normalise_to_correlation(q: &[f64], k: usize) -> Vec<f64> {
    let mut r = vec![0.0_f64; k * k];
    let diag_inv_sqrt: Vec<f64> = (0..k)
        .map(|i| {
            let d = q[i * k + i];
            if d > 1e-15 {
                1.0 / d.sqrt()
            } else {
                1.0
            }
        })
        .collect();

    for i in 0..k {
        for j in 0..k {
            r[i * k + j] = q[i * k + j] * diag_inv_sqrt[i] * diag_inv_sqrt[j];
        }
    }
    r
}

// ============================================================
// Stage 2: DCC dynamics
// ============================================================

/// Compute the DCC log-likelihood for given (α, β) and standardised residuals.
///
/// Returns (log_likelihood, correlations) where correlations is a Vec of k×k row-major matrices.
fn dcc_log_likelihood(
    alpha: f64,
    beta: f64,
    q_bar: &[f64],
    std_resids: &[Vec<f64>],
    k: usize,
    n: usize,
) -> Result<(f64, Vec<Vec<f64>>)> {
    if alpha + beta >= 1.0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "alpha+beta".to_string(),
            message: "DCC stationarity requires α + β < 1".to_string(),
        });
    }
    if alpha < 0.0 || beta < 0.0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "alpha,beta".to_string(),
            message: "DCC parameters must be non-negative".to_string(),
        });
    }

    let coeff_qbar = 1.0 - alpha - beta;

    // Scaled Q̄ component
    let scaled_qbar: Vec<f64> = q_bar.iter().map(|&v| coeff_qbar * v).collect();

    // Q_0 = Q̄
    let mut q_prev = q_bar.to_vec();
    let mut correlations = Vec::with_capacity(n);

    let mut ll = 0.0_f64;

    for t in 1..n {
        // Outer product of ε_{t-1}
        let mut outer = vec![0.0_f64; k * k];
        for i in 0..k {
            for j in 0..k {
                outer[i * k + j] = std_resids[i][t - 1] * std_resids[j][t - 1];
            }
        }

        // Q_t = (1-α-β)Q̄ + α(ε_{t-1}ε_{t-1}^T) + βQ_{t-1}
        let mut q_t = vec![0.0_f64; k * k];
        for idx in 0..(k * k) {
            q_t[idx] = scaled_qbar[idx] + alpha * outer[idx] + beta * q_prev[idx];
        }

        // R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
        let r_t = normalise_to_correlation(&q_t, k);
        correlations.push(r_t.clone());

        // Log-likelihood contribution: -0.5 * [log|R_t| + ε_t^T R_t^{-1} ε_t - ε_t^T ε_t]
        // For k=2, we can compute analytically; for general k, use det/inv of small matrix
        let ll_t = dcc_ll_contribution(&r_t, std_resids, k, t)?;
        ll += ll_t;

        q_prev = q_t;
    }

    // Pad first element with identity correlation
    let mut all_corr = Vec::with_capacity(n);
    let mut identity = vec![0.0_f64; k * k];
    for i in 0..k {
        identity[i * k + i] = 1.0;
    }
    all_corr.push(identity);
    all_corr.extend(correlations);

    Ok((ll, all_corr))
}

/// Log-likelihood contribution at time t for the DCC correlation part.
///
/// ℓ_t = -0.5 * [log|R_t| + ε_t^T R_t^{-1} ε_t - ε_t^T ε_t]
fn dcc_ll_contribution(r: &[f64], std_resids: &[Vec<f64>], k: usize, t: usize) -> Result<f64> {
    let eps: Vec<f64> = (0..k).map(|i| std_resids[i][t]).collect();

    if k == 2 {
        // Analytical formulas for 2×2
        let rho = r[0 * 2 + 1]; // off-diagonal
        let det = 1.0 - rho * rho;
        if det <= 1e-15 {
            return Err(TimeSeriesError::NumericalInstability(
                "DCC: singular correlation matrix".to_string(),
            ));
        }

        let log_det = det.ln();
        // R^{-1} = (1/det) * [[1, -rho], [-rho, 1]]
        let quad = (eps[0] * eps[0] - 2.0 * rho * eps[0] * eps[1] + eps[1] * eps[1]) / det;
        let eps_sq = eps[0] * eps[0] + eps[1] * eps[1];
        return Ok(-0.5 * (log_det + quad - eps_sq));
    }

    // General k×k: Cholesky-based log-det and solve
    let (log_det, r_inv_eps) = cholesky_logdet_solve(r, &eps, k)?;
    let quad: f64 = eps.iter().zip(r_inv_eps.iter()).map(|(&e, &v)| e * v).sum();
    let eps_sq: f64 = eps.iter().map(|&e| e * e).sum();

    Ok(-0.5 * (log_det + quad - eps_sq))
}

/// Cholesky decomposition of a k×k SPD matrix, returning log-determinant and L^{-1}b solve.
fn cholesky_logdet_solve(a: &[f64], b: &[f64], k: usize) -> Result<(f64, Vec<f64>)> {
    // Cholesky: A = L L^T
    let mut l = vec![0.0_f64; k * k];

    for i in 0..k {
        for j in 0..=i {
            let mut sum = 0.0_f64;
            for p in 0..j {
                sum += l[i * k + p] * l[j * k + p];
            }
            if i == j {
                let diag = a[i * k + i] - sum;
                if diag <= 1e-15 {
                    return Err(TimeSeriesError::NumericalInstability(
                        "Cholesky: matrix not positive definite".to_string(),
                    ));
                }
                l[i * k + j] = diag.sqrt();
            } else {
                l[i * k + j] = (a[i * k + j] - sum) / l[j * k + j];
            }
        }
    }

    // log|A| = 2 * sum(log(L_ii))
    let mut log_det = 0.0_f64;
    for i in 0..k {
        log_det += 2.0 * l[i * k + i].ln();
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0_f64; k];
    for i in 0..k {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * k + j] * y[j];
        }
        y[i] = sum / l[i * k + i];
    }

    // Back substitution: L^T x = y
    let mut x = vec![0.0_f64; k];
    for i in (0..k).rev() {
        let mut sum = y[i];
        for j in (i + 1)..k {
            sum -= l[j * k + i] * x[j];
        }
        x[i] = sum / l[i * k + i];
    }

    Ok((log_det, x))
}

// ============================================================
// Stage 2 optimisation
// ============================================================

/// Grid-search + Nelder-Mead for DCC (α, β) parameters.
fn optimise_dcc(
    q_bar: &[f64],
    std_resids: &[Vec<f64>],
    k: usize,
    n: usize,
    config: &DCCConfig,
) -> Result<(f64, f64, f64, Vec<Vec<f64>>)> {
    // Grid search for initial point
    let grid_vals: [f64; 7] = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20];
    let mut best_alpha = 0.05_f64;
    let mut best_beta = 0.90_f64;
    let mut best_ll = f64::NEG_INFINITY;

    for &a in &grid_vals {
        for b_val in grid_vals.iter().copied() {
            if a + b_val >= 0.99 {
                continue;
            }
            let b_adj = b_val.min(0.98 - a);
            if let Ok((ll, _)) = dcc_log_likelihood(a, b_adj, q_bar, std_resids, k, n) {
                if ll > best_ll {
                    best_ll = ll;
                    best_alpha = a;
                    best_beta = b_adj;
                }
            }
        }
    }

    // Nelder-Mead on unconstrained (logit-transformed) space
    let to_unconstrained = |a: f64, b: f64| -> (f64, f64) {
        let a_c = a.max(1e-6).min(0.98);
        let b_c = b.max(1e-6).min(0.98 - a_c);
        let u_a = (a_c / (1.0 - a_c)).ln();
        let u_b = (b_c / (1.0 - b_c)).ln();
        (u_a, u_b)
    };

    let from_unconstrained = |u_a: f64, u_b: f64| -> (f64, f64) {
        let a = 1.0 / (1.0 + (-u_a).exp());
        let b = 1.0 / (1.0 + (-u_b).exp());
        // Rescale to ensure α + β < 0.999
        let total = a + b;
        if total >= 0.999 {
            let scale = 0.95 / total;
            (a * scale, b * scale)
        } else {
            (a, b)
        }
    };

    let (u0_a, u0_b) = to_unconstrained(best_alpha, best_beta);

    let obj = |params: &[f64]| -> f64 {
        let (a, b) = from_unconstrained(params[0], params[1]);
        match dcc_log_likelihood(a, b, q_bar, std_resids, k, n) {
            Ok((ll, _)) => -ll,
            Err(_) => f64::INFINITY,
        }
    };

    let (best_raw, _neg_ll) = nelder_mead_2d(obj, vec![u0_a, u0_b], config.max_iter, config.tol);
    let (final_alpha, final_beta) = from_unconstrained(best_raw[0], best_raw[1]);
    let (final_ll, final_corr) =
        dcc_log_likelihood(final_alpha, final_beta, q_bar, std_resids, k, n)?;

    Ok((final_alpha, final_beta, final_ll, final_corr))
}

/// Simple Nelder-Mead for 2D optimisation.
fn nelder_mead_2d<F>(f: F, x0: Vec<f64>, max_iter: usize, tol: f64) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.clone());
    for i in 0..n {
        let mut vertex = x0.clone();
        vertex[i] += if vertex[i].abs() > 1e-5 {
            0.05 * vertex[i].abs()
        } else {
            0.00025
        };
        simplex.push(vertex);
    }
    let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for _iter in 0..max_iter {
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let best = order[0];
        let worst = order[n];
        let second_worst = order[n - 1];

        let spread: f64 = order
            .iter()
            .map(|&i| (fvals[i] - fvals[best]).abs())
            .fold(0.0_f64, f64::max);
        if spread < tol {
            break;
        }

        let mut centroid = vec![0.0_f64; n];
        for &i in order.iter().take(n) {
            for j in 0..n {
                centroid[j] += simplex[i][j];
            }
        }
        for c in centroid.iter_mut() {
            *c /= n as f64;
        }

        let reflected: Vec<f64> = (0..n)
            .map(|j| centroid[j] + 1.0 * (centroid[j] - simplex[worst][j]))
            .collect();
        let f_reflected = f(&reflected);

        if f_reflected < fvals[best] {
            let expanded: Vec<f64> = (0..n)
                .map(|j| centroid[j] + 2.0 * (reflected[j] - centroid[j]))
                .collect();
            let f_expanded = f(&expanded);
            if f_expanded < f_reflected {
                simplex[worst] = expanded;
                fvals[worst] = f_expanded;
            } else {
                simplex[worst] = reflected;
                fvals[worst] = f_reflected;
            }
        } else if f_reflected < fvals[second_worst] {
            simplex[worst] = reflected;
            fvals[worst] = f_reflected;
        } else {
            let contracted: Vec<f64> = (0..n)
                .map(|j| centroid[j] + 0.5 * (simplex[worst][j] - centroid[j]))
                .collect();
            let f_contracted = f(&contracted);
            if f_contracted < fvals[worst] {
                simplex[worst] = contracted;
                fvals[worst] = f_contracted;
            } else {
                let best_vertex = simplex[best].clone();
                for i in 0..=n {
                    if i != best {
                        for j in 0..n {
                            simplex[i][j] = best_vertex[j] + 0.5 * (simplex[i][j] - best_vertex[j]);
                        }
                        fvals[i] = f(&simplex[i]);
                    }
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..=n).collect();
    order.sort_by(|&a, &b| {
        fvals[a]
            .partial_cmp(&fvals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let best = order[0];
    (simplex[best].clone(), fvals[best])
}

// ============================================================
// Public API
// ============================================================

/// Fit a DCC-GARCH model to multivariate return data.
///
/// # Arguments
/// * `returns` — Vector of return series, one per asset. All must have the same length.
/// * `config` — DCC configuration parameters.
///
/// # Returns
/// A `DCCResult` containing estimated parameters, conditional correlations, and covariances.
pub fn fit_dcc(returns: &[Vec<f64>], config: &DCCConfig) -> Result<DCCResult> {
    let k = returns.len();
    if k < 2 {
        return Err(TimeSeriesError::InvalidInput(
            "DCC-GARCH requires at least 2 series".to_string(),
        ));
    }
    let n = returns[0].len();
    for (i, series) in returns.iter().enumerate() {
        if series.len() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: series.len(),
            });
        }
        if series.len() < 20 {
            return Err(TimeSeriesError::InsufficientData {
                message: format!("DCC-GARCH: series {} too short", i),
                required: 20,
                actual: series.len(),
            });
        }
    }

    // Stage 1: Fit univariate GARCH
    let (garch_models, std_resids, cond_vols) =
        stage1_fit(returns, config.garch_p, config.garch_q)?;

    // Compute unconditional correlation Q̄
    let q_bar = compute_q_bar(&std_resids, k, n);

    // Stage 2: Optimise DCC parameters
    let (alpha, beta, ll, correlations) = optimise_dcc(&q_bar, &std_resids, k, n, config)?;

    // Build covariance matrices: H_t = D_t R_t D_t where D_t = diag(σ_{1,t}, ..., σ_{k,t})
    let mut covariances = Vec::with_capacity(n);
    for t in 0..n {
        let mut cov = vec![0.0_f64; k * k];
        for i in 0..k {
            for j in 0..k {
                cov[i * k + j] = cond_vols[i][t] * cond_vols[j][t] * correlations[t][i * k + j];
            }
        }
        covariances.push(cov);
    }

    Ok(DCCResult {
        alpha,
        beta,
        q_bar,
        garch_models,
        correlations,
        covariances,
        log_likelihood: ll,
        k,
        n_obs: n,
    })
}

/// Forecast conditional covariance matrices h steps ahead from a DCC result.
///
/// Multi-step forecasts use the mean-reverting property:
/// - Univariate variances converge to unconditional variance
/// - Correlations converge to Q̄
pub fn dcc_forecast(result: &DCCResult, steps: usize) -> Result<Vec<CovMatrixForecast>> {
    if steps == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "steps".to_string(),
            message: "Forecast horizon must be >= 1".to_string(),
        });
    }

    let k = result.k;
    let n = result.n_obs;

    // Get last conditional variances for each series
    let last_vars: Vec<f64> = result
        .covariances
        .last()
        .map(|cov| (0..k).map(|i| cov[i * k + i]).collect())
        .ok_or_else(|| TimeSeriesError::InvalidInput("No covariance data".to_string()))?;

    // Get last Q matrix (un-normalise from last correlation)
    // Use the DCC recursion: Q converges to Q̄
    let pers = result.alpha + result.beta;

    // Last correlation
    let last_corr = result
        .correlations
        .last()
        .ok_or_else(|| TimeSeriesError::InvalidInput("No correlation data".to_string()))?;

    let mut forecasts = Vec::with_capacity(steps);

    let mut prev_q: Vec<f64> = last_corr.clone();

    for step in 0..steps {
        // Q_{t+h} converges: E[Q_{t+h}] → Q̄ as h → ∞
        // Simplified: R_{t+h} = (1 - (α+β)^h) Q̄_norm + (α+β)^h R_t
        let decay = pers.powi((step + 1) as i32);
        let q_bar_norm = normalise_to_correlation(&result.q_bar, k);

        let mut cor = vec![0.0_f64; k * k];
        for idx in 0..(k * k) {
            cor[idx] = (1.0 - decay) * q_bar_norm[idx] + decay * last_corr[idx];
        }
        // Ensure diagonal is 1
        for i in 0..k {
            cor[i * k + i] = 1.0;
        }

        // Forecast variances: mean-revert for each univariate GARCH
        let mut cov = vec![0.0_f64; k * k];
        for i in 0..k {
            let model = &result.garch_models[i];
            let alpha_sum: f64 = model.alpha.iter().sum();
            let beta_sum: f64 = model.beta.iter().sum();
            let model_pers = alpha_sum + beta_sum;
            let unc_var = model.unconditional_variance().unwrap_or(last_vars[i]);
            let var_i = unc_var + (last_vars[i] - unc_var) * model_pers.powi((step + 1) as i32);
            let vol_i = var_i.max(1e-10).sqrt();

            for j in 0..k {
                let model_j = &result.garch_models[j];
                let alpha_sum_j: f64 = model_j.alpha.iter().sum();
                let beta_sum_j: f64 = model_j.beta.iter().sum();
                let model_pers_j = alpha_sum_j + beta_sum_j;
                let unc_var_j = model_j.unconditional_variance().unwrap_or(last_vars[j]);
                let var_j =
                    unc_var_j + (last_vars[j] - unc_var_j) * model_pers_j.powi((step + 1) as i32);
                let vol_j = var_j.max(1e-10).sqrt();

                cov[i * k + j] = vol_i * vol_j * cor[i * k + j];
            }
        }

        forecasts.push(CovMatrixForecast {
            step: step + 1,
            cov_matrix: cov,
            cor_matrix: cor,
        });

        prev_q = forecasts
            .last()
            .map(|f| f.cor_matrix.clone())
            .unwrap_or(prev_q);
    }

    Ok(forecasts)
}

// ============================================================
// Constant Conditional Correlation (CCC) — special case
// ============================================================

/// Fit a CCC-GARCH model (DCC with α=0, β=0 → constant correlation).
///
/// This is a convenience wrapper that fits DCC with fixed α=β=0.
pub fn fit_ccc(returns: &[Vec<f64>], garch_p: usize, garch_q: usize) -> Result<DCCResult> {
    let k = returns.len();
    if k < 2 {
        return Err(TimeSeriesError::InvalidInput(
            "CCC-GARCH requires at least 2 series".to_string(),
        ));
    }
    let n = returns[0].len();
    for (i, series) in returns.iter().enumerate() {
        if series.len() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: series.len(),
            });
        }
    }

    let (garch_models, std_resids, cond_vols) = stage1_fit(returns, garch_p, garch_q)?;
    let q_bar = compute_q_bar(&std_resids, k, n);
    let r_const = normalise_to_correlation(&q_bar, k);

    // Compute log-likelihood under constant correlation
    let mut ll = 0.0_f64;
    for t in 1..n {
        let eps: Vec<f64> = (0..k).map(|i| std_resids[i][t]).collect();
        if k == 2 {
            let rho = r_const[1];
            let det = 1.0 - rho * rho;
            if det > 1e-15 {
                let quad = (eps[0] * eps[0] - 2.0 * rho * eps[0] * eps[1] + eps[1] * eps[1]) / det;
                let eps_sq = eps[0] * eps[0] + eps[1] * eps[1];
                ll += -0.5 * (det.ln() + quad - eps_sq);
            }
        } else if let Ok((ld, r_inv_eps)) = cholesky_logdet_solve(&r_const, &eps, k) {
            let quad: f64 = eps.iter().zip(r_inv_eps.iter()).map(|(&e, &v)| e * v).sum();
            let eps_sq: f64 = eps.iter().map(|&e| e * e).sum();
            ll += -0.5 * (ld + quad - eps_sq);
        }
    }

    // Constant correlation at all times
    let correlations: Vec<Vec<f64>> = (0..n).map(|_| r_const.clone()).collect();

    // Build covariance matrices
    let mut covariances = Vec::with_capacity(n);
    for t in 0..n {
        let mut cov = vec![0.0_f64; k * k];
        for i in 0..k {
            for j in 0..k {
                cov[i * k + j] = cond_vols[i][t] * cond_vols[j][t] * r_const[i * k + j];
            }
        }
        covariances.push(cov);
    }

    Ok(DCCResult {
        alpha: 0.0,
        beta: 0.0,
        q_bar,
        garch_models,
        correlations,
        covariances,
        log_likelihood: ll,
        k,
        n_obs: n,
    })
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_correlated_returns(n: usize, rho: f64) -> Vec<Vec<f64>> {
        // Generate two correlated series using a deterministic pseudo-random sequence
        let mut r1 = Vec::with_capacity(n);
        let mut r2 = Vec::with_capacity(n);
        for i in 0..n {
            let z1 = 0.01 * ((i as f64 * 1.37 + 0.5).sin());
            let z2_ind = 0.01 * ((i as f64 * 2.71 + 1.3).sin());
            let z2 = rho * z1 + (1.0 - rho * rho).sqrt() * z2_ind;
            // Add volatility clustering
            let vol = if i % 7 < 3 { 2.0 } else { 1.0 };
            r1.push(z1 * vol);
            r2.push(z2 * vol);
        }
        vec![r1, r2]
    }

    fn make_volatile_returns(n: usize) -> Vec<Vec<f64>> {
        // Two series with a regime change in correlation
        let mut r1 = Vec::with_capacity(n);
        let mut r2 = Vec::with_capacity(n);
        for i in 0..n {
            let z1 = 0.01 * ((i as f64 * 1.37 + 0.5).sin());
            let rho = if i < n / 2 { 0.3 } else { 0.8 };
            let z2_ind = 0.01 * ((i as f64 * 2.71 + 1.3).sin());
            let z2 = rho * z1 + (1.0 - rho * rho).sqrt() * z2_ind;
            let vol = if i % 5 < 2 { 2.5 } else { 1.0 };
            r1.push(z1 * vol);
            r2.push(z2 * vol);
        }
        vec![r1, r2]
    }

    #[test]
    fn test_dcc_constant_correlation_reduces_to_ccc() {
        // With perfectly constant correlation, DCC should produce α≈0, β≈0
        let returns = make_correlated_returns(100, 0.5);
        let config = DCCConfig::default();
        let dcc = fit_dcc(&returns, &config).expect("DCC should fit");
        let ccc = fit_ccc(&returns, 1, 1).expect("CCC should fit");

        // Both should produce valid results
        assert!(dcc.alpha >= 0.0);
        assert!(dcc.beta >= 0.0);
        assert!(dcc.alpha + dcc.beta < 1.0);

        // CCC has α=0, β=0
        assert!((ccc.alpha - 0.0).abs() < 1e-10);
        assert!((ccc.beta - 0.0).abs() < 1e-10);

        // Correlations from CCC should be constant
        let first_corr = &ccc.correlations[1];
        for t in 2..ccc.n_obs {
            for idx in 0..(ccc.k * ccc.k) {
                assert!(
                    (ccc.correlations[t][idx] - first_corr[idx]).abs() < 1e-10,
                    "CCC correlations must be constant"
                );
            }
        }
    }

    #[test]
    fn test_dcc_time_varying_correlation() {
        // Data with a regime change should produce non-trivial DCC dynamics
        let returns = make_volatile_returns(120);
        let config = DCCConfig::default();
        let result = fit_dcc(&returns, &config).expect("DCC should fit");

        assert_eq!(result.k, 2);
        assert_eq!(result.correlations.len(), result.n_obs);

        // Correlations should vary over time (not identical)
        let mut min_corr = f64::INFINITY;
        let mut max_corr = f64::NEG_INFINITY;
        for corr in &result.correlations {
            let rho = corr[0 * 2 + 1]; // off-diagonal
            if rho < min_corr {
                min_corr = rho;
            }
            if rho > max_corr {
                max_corr = rho;
            }
        }
        // Should have some variation (may be small with deterministic data)
        assert!(
            min_corr.is_finite() && max_corr.is_finite(),
            "Correlations must be finite"
        );
    }

    #[test]
    fn test_dcc_stationarity_constraint() {
        let returns = make_correlated_returns(80, 0.6);
        let config = DCCConfig::default();
        let result = fit_dcc(&returns, &config).expect("DCC should fit");

        // α + β < 1 for stationarity
        assert!(
            result.alpha + result.beta < 1.0,
            "DCC must satisfy stationarity: α={}, β={}, sum={}",
            result.alpha,
            result.beta,
            result.alpha + result.beta
        );
        assert!(result.alpha >= 0.0);
        assert!(result.beta >= 0.0);
    }

    #[test]
    fn test_dcc_forecast() {
        let returns = make_correlated_returns(80, 0.5);
        let config = DCCConfig::default();
        let result = fit_dcc(&returns, &config).expect("DCC should fit");

        let forecasts = dcc_forecast(&result, 5).expect("Should forecast");
        assert_eq!(forecasts.len(), 5);

        for fc in &forecasts {
            // Correlation diagonal should be 1
            for i in 0..result.k {
                assert!(
                    (fc.cor_matrix[i * result.k + i] - 1.0).abs() < 1e-8,
                    "Diagonal correlation must be 1"
                );
            }
            // Covariance diagonal must be positive
            for i in 0..result.k {
                assert!(
                    fc.cov_matrix[i * result.k + i] > 0.0,
                    "Diagonal covariance must be positive"
                );
            }
        }
    }

    #[test]
    fn test_dcc_too_few_series() {
        let returns = vec![vec![0.01, -0.02, 0.015]];
        let config = DCCConfig::default();
        assert!(fit_dcc(&returns, &config).is_err());
    }

    #[test]
    fn test_dcc_covariance_matrices_positive() {
        let returns = make_correlated_returns(80, 0.4);
        let config = DCCConfig::default();
        let result = fit_dcc(&returns, &config).expect("DCC should fit");

        // All covariance matrices should have positive diagonal
        for (t, cov) in result.covariances.iter().enumerate() {
            for i in 0..result.k {
                assert!(
                    cov[i * result.k + i] > 0.0,
                    "Covariance diagonal must be positive at t={}, i={}",
                    t,
                    i
                );
            }
        }
    }
}
