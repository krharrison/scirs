//! ARCH (AutoRegressive Conditional Heteroskedasticity) models
//!
//! This module implements the ARCH(p) model introduced by Engle (1982).
//! ARCH models capture the phenomenon of volatility clustering in financial
//! time series, where large changes tend to be followed by large changes.
//!
//! # Model Specification
//!
//! The ARCH(p) model specifies the conditional variance as:
//!
//! ```text
//! σ²ₜ = ω + α₁ε²ₜ₋₁ + α₂ε²ₜ₋₂ + ... + αₚε²ₜ₋ₚ
//! ```
//!
//! where:
//! - `σ²ₜ` is the conditional variance at time t
//! - `ω > 0` is the long-run average variance (intercept)
//! - `αᵢ ≥ 0` are the ARCH coefficients
//! - `εₜ` are the residuals (innovations)
//!
//! For covariance stationarity: Σαᵢ < 1
//!
//! # Examples
//!
//! ```rust
//! use scirs2_series::volatility::arch::{ARCHModel, fit_arch, arch_volatility, engle_test};
//! use scirs2_core::ndarray::array;
//!
//! // Fit ARCH(1) model
//! let returns = array![0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.018, 0.009,
//!                       -0.003, 0.007, 0.025, -0.014, 0.008, -0.006, 0.011];
//! let model = fit_arch(&returns, 1).expect("should succeed");
//! let vol = arch_volatility(&returns, &model).expect("should succeed");
//! ```

use scirs2_core::ndarray::Array1;

use crate::error::{Result, TimeSeriesError};

/// ARCH(p) model parameters
///
/// Represents the estimated parameters of an AutoRegressive Conditional
/// Heteroskedasticity model of order p.
#[derive(Debug, Clone)]
pub struct ARCHModel {
    /// Order of the ARCH model (number of lagged squared residuals)
    pub p: usize,
    /// Intercept (long-run variance component); must be positive
    pub omega: f64,
    /// ARCH coefficients α₁, ..., αₚ; each must be non-negative
    pub alpha: Vec<f64>,
    /// Log-likelihood at optimum (populated after fitting)
    pub log_likelihood: f64,
    /// Number of observations used in estimation
    pub n_obs: usize,
}

impl ARCHModel {
    /// Create a new ARCH model with given parameters (no validation)
    pub fn new(p: usize, omega: f64, alpha: Vec<f64>) -> Result<Self> {
        if alpha.len() != p {
            return Err(TimeSeriesError::InvalidModel(format!(
                "alpha length {} does not match order p={}",
                alpha.len(),
                p
            )));
        }
        if omega <= 0.0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "omega".to_string(),
                message: "omega must be strictly positive".to_string(),
            });
        }
        for (i, &a) in alpha.iter().enumerate() {
            if a < 0.0 {
                return Err(TimeSeriesError::InvalidParameter {
                    name: format!("alpha[{}]", i),
                    message: "ARCH coefficients must be non-negative".to_string(),
                });
            }
        }
        Ok(Self {
            p,
            omega,
            alpha,
            log_likelihood: f64::NEG_INFINITY,
            n_obs: 0,
        })
    }

    /// Compute the persistence of the ARCH model (sum of alpha coefficients)
    ///
    /// For covariance stationarity the persistence must be < 1.
    pub fn persistence(&self) -> f64 {
        self.alpha.iter().sum()
    }

    /// Unconditional (long-run) variance implied by the model
    ///
    /// Only valid when persistence() < 1.
    pub fn unconditional_variance(&self) -> Result<f64> {
        let pers = self.persistence();
        if pers >= 1.0 {
            return Err(TimeSeriesError::InvalidModel(
                "Model is non-stationary: persistence >= 1".to_string(),
            ));
        }
        Ok(self.omega / (1.0 - pers))
    }

    /// Number of free parameters in the model (omega + p alpha coefficients)
    pub fn n_params(&self) -> usize {
        1 + self.p
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the Gaussian ARCH log-likelihood for given parameters.
///
/// Returns (log_likelihood, conditional_variances).
fn arch_log_likelihood(
    residuals: &Array1<f64>,
    omega: f64,
    alpha: &[f64],
    p: usize,
) -> (f64, Vec<f64>) {
    let n = residuals.len();
    let mut sigma2 = vec![0.0_f64; n];

    // Unconditional variance for initialisation (use sample variance as proxy)
    let sample_var: f64 = {
        let mean = residuals.iter().sum::<f64>() / n as f64;
        residuals.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64
    };
    let init_var = sample_var.max(1e-8);

    // Fill initial p observations with unconditional estimate
    for i in 0..p.min(n) {
        sigma2[i] = init_var;
    }

    for t in p..n {
        let mut var_t = omega;
        for lag in 1..=p {
            let idx = t - lag;
            let eps_sq = residuals[idx] * residuals[idx];
            var_t += alpha[lag - 1] * eps_sq;
        }
        sigma2[t] = var_t.max(1e-10);
    }

    // Gaussian log-likelihood: -0.5 * Σ [log(2π) + log(σ²ₜ) + ε²ₜ/σ²ₜ]
    let log2pi = (2.0 * std::f64::consts::PI).ln();
    let mut ll = 0.0_f64;
    for t in p..n {
        ll -= 0.5 * (log2pi + sigma2[t].ln() + residuals[t] * residuals[t] / sigma2[t]);
    }

    (ll, sigma2)
}

/// Nelder-Mead simplex optimiser (pure Rust, no external dependency).
///
/// Minimises `f(x)`.  Returns `(best_x, best_value)`.
fn nelder_mead<F>(
    f: F,
    x0: Vec<f64>,
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    // Build initial simplex
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.clone());
    for i in 0..n {
        let mut vertex = x0.clone();
        vertex[i] += if vertex[i].abs() > 1e-5 { 0.05 * vertex[i].abs() } else { 0.00025 };
        simplex.push(vertex);
    }

    let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for _iter in 0..max_iter {
        // Sort
        let mut order: Vec<usize> = (0..n + 1).collect();
        order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap_or(std::cmp::Ordering::Equal));
        let best = order[0];
        let worst = order[n];
        let second_worst = order[n - 1];

        // Convergence check
        let spread: f64 = order
            .iter()
            .map(|&i| (fvals[i] - fvals[best]).abs())
            .fold(0.0_f64, f64::max);
        if spread < tol {
            break;
        }

        // Centroid of all but worst
        let mut centroid = vec![0.0_f64; n];
        for &i in order.iter().take(n) {
            for j in 0..n {
                centroid[j] += simplex[i][j];
            }
        }
        for c in centroid.iter_mut() {
            *c /= n as f64;
        }

        // Reflection
        let reflected: Vec<f64> = (0..n)
            .map(|j| centroid[j] + 1.0 * (centroid[j] - simplex[worst][j]))
            .collect();
        let f_reflected = f(&reflected);

        if f_reflected < fvals[best] {
            // Expansion
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
            // Contraction
            let contracted: Vec<f64> = (0..n)
                .map(|j| centroid[j] + 0.5 * (simplex[worst][j] - centroid[j]))
                .collect();
            let f_contracted = f(&contracted);
            if f_contracted < fvals[worst] {
                simplex[worst] = contracted;
                fvals[worst] = f_contracted;
            } else {
                // Shrink
                let best_vertex = simplex[best].clone();
                for i in 0..n + 1 {
                    if i != best {
                        for j in 0..n {
                            simplex[i][j] =
                                best_vertex[j] + 0.5 * (simplex[i][j] - best_vertex[j]);
                        }
                        fvals[i] = f(&simplex[i]);
                    }
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..n + 1).collect();
    order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap_or(std::cmp::Ordering::Equal));
    let best = order[0];
    (simplex[best].clone(), fvals[best])
}

/// Project unconstrained parameters to the ARCH feasible region.
///
/// Uses softplus for positivity and normalises alpha if persistence is too large.
fn project_params_arch(raw: &[f64], p: usize) -> (f64, Vec<f64>) {
    // softplus: log(1 + exp(x)) ≈ x for large x, smooth and always positive
    let softplus = |x: f64| (1.0 + x.exp()).ln().max(1e-8);

    let omega = softplus(raw[0]);
    let mut alpha: Vec<f64> = (0..p).map(|i| softplus(raw[1 + i])).collect();

    // Enforce stationarity: Σα < 0.999
    let alpha_sum: f64 = alpha.iter().sum();
    if alpha_sum >= 0.999 {
        let scale = 0.95 / alpha_sum;
        for a in alpha.iter_mut() {
            *a *= scale;
        }
    }

    (omega, alpha)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Fit an ARCH(p) model to a returns or residuals series via MLE.
///
/// Uses the Nelder-Mead simplex algorithm to maximise the Gaussian
/// log-likelihood.  The parameters are reparameterised to ensure positivity
/// (via softplus) and stationarity constraints.
///
/// # Arguments
///
/// * `residuals` - Time series of residuals (demeaned returns)
/// * `p`         - ARCH order (number of lagged squared residuals)
///
/// # Returns
///
/// A fitted [`ARCHModel`] with MLE parameter estimates.
///
/// # Examples
///
/// ```rust
/// use scirs2_series::volatility::arch::fit_arch;
/// use scirs2_core::ndarray::Array1;
///
/// let returns = Array1::from(vec![
///     0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.018, 0.009,
///     -0.003, 0.007, 0.025, -0.014, 0.008, -0.006, 0.011, -0.019,
///     0.022, 0.003, -0.011, 0.017,
/// ]);
/// let model = fit_arch(&returns, 1).expect("should succeed");
/// assert!(model.omega > 0.0);
/// ```
pub fn fit_arch(residuals: &Array1<f64>, p: usize) -> Result<ARCHModel> {
    let n = residuals.len();
    if n < p + 10 {
        return Err(TimeSeriesError::InsufficientData {
            message: "ARCH fitting".to_string(),
            required: p + 10,
            actual: n,
        });
    }
    if p == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "p".to_string(),
            message: "ARCH order must be at least 1".to_string(),
        });
    }

    // Sample variance as starting point for omega
    let mean = residuals.iter().sum::<f64>() / n as f64;
    let sample_var = residuals.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64;

    // Initial raw parameters (will be passed through softplus)
    // omega_raw ≈ log(exp(sample_var * 0.1) - 1) ≈ sample_var * 0.1 for small values
    let omega_init = (sample_var * 0.1).max(1e-6);
    // alpha starting at 0.1 per lag
    let mut x0 = vec![omega_init.ln(); 1 + p];
    for i in 0..p {
        x0[1 + i] = 0.1_f64.ln();
    }

    let resid_clone: Vec<f64> = residuals.iter().cloned().collect();
    let n_params = 1 + p;

    let obj = move |raw: &[f64]| {
        let (omega, alpha) = project_params_arch(raw, p);
        let (ll, _) = arch_log_likelihood(
            &Array1::from(resid_clone.clone()),
            omega,
            &alpha,
            p,
        );
        -ll // minimise negative log-likelihood
    };

    let (best_raw, neg_ll) = nelder_mead(obj, x0, 2000 * n_params, 1e-8);
    let (omega, alpha) = project_params_arch(&best_raw, p);

    let mut model = ARCHModel::new(p, omega, alpha)?;
    model.log_likelihood = -neg_ll;
    model.n_obs = n;
    Ok(model)
}

/// Compute the conditional variance (volatility²) series implied by an ARCH model.
///
/// Returns a vector of length n where the first p entries are initialised
/// using the unconditional variance (sample variance), and subsequent entries
/// use the ARCH recursion.
///
/// # Arguments
///
/// * `residuals` - Residual series used for variance recursion
/// * `model`     - Fitted (or manually specified) ARCH model
///
/// # Returns
///
/// Vector of conditional variances `σ²ₜ` for t = 0, ..., n-1.
///
/// # Examples
///
/// ```rust
/// use scirs2_series::volatility::arch::{fit_arch, arch_volatility};
/// use scirs2_core::ndarray::Array1;
///
/// let returns = Array1::from(vec![
///     0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.018, 0.009,
///     -0.003, 0.007, 0.025, -0.014, 0.008, -0.006, 0.011, -0.019,
/// ]);
/// let model = fit_arch(&returns, 1).expect("should succeed");
/// let cond_var = arch_volatility(&returns, &model).expect("should succeed");
/// assert_eq!(cond_var.len(), returns.len());
/// ```
pub fn arch_volatility(residuals: &Array1<f64>, model: &ARCHModel) -> Result<Vec<f64>> {
    let n = residuals.len();
    if n < model.p {
        return Err(TimeSeriesError::InsufficientData {
            message: "arch_volatility".to_string(),
            required: model.p,
            actual: n,
        });
    }
    let (_, sigma2) = arch_log_likelihood(residuals, model.omega, &model.alpha, model.p);
    Ok(sigma2)
}

/// Engle's ARCH-LM (Lagrange Multiplier) test for heteroskedasticity.
///
/// Tests the null hypothesis that a series exhibits no ARCH effects up to
/// lag `lags`.  Under H₀ the test statistic T·R² is asymptotically χ²(lags).
///
/// # Arguments
///
/// * `residuals` - Residual series (demeaned returns)
/// * `lags`      - Number of lags to include in the auxiliary regression
///
/// # Returns
///
/// [`EngleLMResult`] containing the test statistic and p-value.
///
/// # Examples
///
/// ```rust
/// use scirs2_series::volatility::arch::engle_test;
/// use scirs2_core::ndarray::Array1;
///
/// let returns = Array1::from(vec![
///     0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.018, 0.009,
///     -0.003, 0.007, 0.025, -0.014, 0.008, -0.006, 0.011, -0.019,
///     0.022, 0.003, -0.011, 0.017,
/// ]);
/// let result = engle_test(&returns, 1).expect("should succeed");
/// // p < 0.05 suggests significant ARCH effects
/// println!("stat={:.4}, p={:.4}", result.statistic, result.p_value);
/// ```
pub fn engle_test(residuals: &Array1<f64>, lags: usize) -> Result<EngleLMResult> {
    let n = residuals.len();
    if n < lags + 10 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Engle ARCH-LM test".to_string(),
            required: lags + 10,
            actual: n,
        });
    }

    // Squared residuals
    let sq: Vec<f64> = residuals.iter().map(|&r| r * r).collect();

    // Demean squared residuals
    let sq_mean = sq.iter().sum::<f64>() / n as f64;
    let sq_demeaned: Vec<f64> = sq.iter().map(|&s| s - sq_mean).collect();

    // Total sum of squares of sq_demeaned (for R² computation)
    let tss: f64 = sq_demeaned[lags..].iter().map(|&s| s * s).sum();

    // OLS of sq on its lags (intercept included).
    // We use the efficient closed-form approach: compute R² via the explained
    // sum of squares from the auxiliary regression.
    // Design matrix: [1, sq_{t-1}, ..., sq_{t-lags}]  for t = lags..n
    let t = n - lags; // number of rows in auxiliary regression
    let k = lags + 1; // number of columns (including intercept)

    // Build X (t × k) and y (t) in flat Vec for simple normal-equations
    let mut x_mat = vec![0.0_f64; t * k];
    let mut y_vec = vec![0.0_f64; t];

    for i in 0..t {
        let row = lags + i;
        x_mat[i * k] = 1.0; // intercept
        for lag in 1..=lags {
            x_mat[i * k + lag] = sq[row - lag];
        }
        y_vec[i] = sq[row];
    }

    // Normal equations X'X β = X'y via Cholesky
    let xtx = mat_mul_transpose(&x_mat, &x_mat, t, k);
    let xty = mat_vec_mul_transpose(&x_mat, &y_vec, t, k);
    let beta = solve_symmetric_pos_def(&xtx, &xty, k)?;

    // Fitted values and RSS
    let mut rss = 0.0_f64;
    for i in 0..t {
        let mut fitted = 0.0_f64;
        for j in 0..k {
            fitted += x_mat[i * k + j] * beta[j];
        }
        let resid = y_vec[i] - fitted;
        rss += resid * resid;
    }

    // ESS = TSS - RSS
    let ess = tss - rss;
    let r_squared = if tss > 1e-15 { ess / tss } else { 0.0 };

    // LM statistic ~ T * R²
    let lm_stat = t as f64 * r_squared;

    // Chi-squared CDF approximation (degrees of freedom = lags)
    let p_value = chi2_survival(lm_stat, lags as f64);

    Ok(EngleLMResult {
        statistic: lm_stat,
        p_value,
        lags,
        n_obs: n,
        r_squared,
    })
}

/// Result of Engle's ARCH-LM test
#[derive(Debug, Clone)]
pub struct EngleLMResult {
    /// LM test statistic (T·R²)
    pub statistic: f64,
    /// Asymptotic p-value (χ²(lags))
    pub p_value: f64,
    /// Number of lags tested
    pub lags: usize,
    /// Number of observations
    pub n_obs: usize,
    /// R² from auxiliary regression
    pub r_squared: f64,
}

impl EngleLMResult {
    /// Whether the test rejects H₀ (no ARCH effects) at the given significance level
    pub fn reject_null(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

// ---------------------------------------------------------------------------
// Linear algebra helpers (no external crate)
// ---------------------------------------------------------------------------

/// Compute A'A where A is (m × n) stored row-major, result is (n × n)
fn mat_mul_transpose(a: &[f64], _b: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0_f64;
            for k in 0..m {
                s += a[k * n + i] * a[k * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// Compute A'y where A is (m × n) row-major, y is m-vector; result is n-vector
fn mat_vec_mul_transpose(a: &[f64], y: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; n];
    for j in 0..n {
        let mut s = 0.0_f64;
        for i in 0..m {
            s += a[i * n + j] * y[i];
        }
        c[j] = s;
    }
    c
}

/// Solve a symmetric positive-definite system via Cholesky decomposition.
/// `a` is (n×n) stored row-major, `b` is n-vector.
fn solve_symmetric_pos_def(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>> {
    // Cholesky: A = L L'
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(TimeSeriesError::NumericalInstability(
                        "Cholesky decomposition failed: matrix not positive definite".to_string(),
                    ));
                }
                l[i * n + j] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * y[j];
        }
        y[i] = s / l[i * n + i];
    }

    // Backward substitution: L' x = y
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in i + 1..n {
            s -= l[j * n + i] * x[j];
        }
        x[i] = s / l[i * n + i];
    }

    Ok(x)
}

/// Survival function of the chi-squared distribution (upper tail probability).
///
/// Uses the regularised incomplete gamma function via a continued-fraction
/// approximation (Numerical Recipes / DLMF 8.9).
pub(crate) fn chi2_survival(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    regularised_upper_gamma(df / 2.0, x / 2.0)
}

/// Regularised upper incomplete gamma: Q(a, x) = Γ(a,x)/Γ(a)
pub(crate) fn regularised_upper_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - regularised_lower_gamma_series(a, x)
    } else {
        regularised_upper_gamma_cf(a, x)
    }
}

/// Series expansion for regularised lower incomplete gamma P(a, x)
fn regularised_lower_gamma_series(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 3e-7_f64;

    if x <= 0.0 {
        return 0.0;
    }

    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..max_iter {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * eps {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Continued-fraction expansion for Q(a, x)
fn regularised_upper_gamma_cf(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 3e-7_f64;
    let fpmin = 1e-300_f64;

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..=max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = (an * d + b).abs().max(fpmin).copysign(an * d + b);
        c = (b + an / c).abs().max(fpmin).copysign(b + an / c);
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < eps {
            break;
        }
    }

    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Lanczos approximation of the natural log of the gamma function
pub(crate) fn ln_gamma(x: f64) -> f64 {
    // Lanczos coefficients (g=7, n=9)
    let coeffs = [
        0.999_999_999_999_809_3_f64,
        676.520_368_121_885_1,
        -1_259.139_216_722_403,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_9,
        -0.138_571_095_266_549_55,
        9.984_369_578_019_571_9e-6,
        1.505_632_735_149_31e-7,
    ];

    if x < 0.5 {
        std::f64::consts::PI.ln()
            - ((std::f64::consts::PI * x).sin()).ln()
            - ln_gamma(1.0 - x)
    } else {
        let z = x - 1.0;
        let mut sum = coeffs[0];
        for (i, &c) in coeffs.iter().enumerate().skip(1) {
            sum += c / (z + i as f64);
        }
        let t = z + 7.5; // g + 0.5
        (2.0 * std::f64::consts::PI).sqrt().ln()
            + (t.ln() * (z + 0.5))
            - t
            + sum.ln()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn sample_returns() -> Array1<f64> {
        Array1::from(vec![
            0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.018, 0.009, -0.003, 0.007, 0.025,
            -0.014, 0.008, -0.006, 0.011, -0.019, 0.022, 0.003, -0.011, 0.017, -0.005, 0.031,
            -0.013, 0.009, 0.002, -0.027, 0.016, -0.007, 0.013, 0.004,
        ])
    }

    #[test]
    fn test_arch_model_new() {
        let model = ARCHModel::new(1, 0.001, vec![0.1]).expect("Should create");
        assert_eq!(model.p, 1);
        assert!((model.omega - 0.001).abs() < 1e-12);
        assert!((model.alpha[0] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_arch_model_invalid() {
        assert!(ARCHModel::new(1, -0.001, vec![0.1]).is_err());
        assert!(ARCHModel::new(1, 0.001, vec![0.1, 0.2]).is_err());
        assert!(ARCHModel::new(1, 0.001, vec![-0.1]).is_err());
    }

    #[test]
    fn test_arch_persistence() {
        let model = ARCHModel::new(2, 0.001, vec![0.2, 0.3]).expect("Should create");
        assert!((model.persistence() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_arch_unconditional_variance() {
        let model = ARCHModel::new(1, 0.0001, vec![0.1]).expect("Should create");
        let unc_var = model.unconditional_variance().expect("Should compute");
        let expected = 0.0001 / (1.0 - 0.1);
        assert!((unc_var - expected).abs() < 1e-12);
    }

    #[test]
    fn test_fit_arch() {
        let returns = sample_returns();
        let model = fit_arch(&returns, 1).expect("fit should succeed");
        assert!(model.omega > 0.0);
        assert!(model.alpha.len() == 1);
        assert!(model.alpha[0] >= 0.0);
        assert!(model.log_likelihood.is_finite());
    }

    #[test]
    fn test_fit_arch_order2() {
        let returns = sample_returns();
        let model = fit_arch(&returns, 2).expect("fit should succeed");
        assert_eq!(model.p, 2);
        assert!(model.persistence() < 1.0);
    }

    #[test]
    fn test_arch_volatility() {
        let returns = sample_returns();
        let model = fit_arch(&returns, 1).expect("fit should succeed");
        let vol = arch_volatility(&returns, &model).expect("Should compute");
        assert_eq!(vol.len(), returns.len());
        for &v in &vol {
            assert!(v > 0.0, "Conditional variances must be positive");
        }
    }

    #[test]
    fn test_engle_test() {
        let returns = sample_returns();
        let result = engle_test(&returns, 1).expect("Test should run");
        assert!(result.statistic >= 0.0);
        assert!((0.0..=1.0).contains(&result.p_value));
        assert_eq!(result.lags, 1);
    }

    #[test]
    fn test_engle_test_insufficient_data() {
        let tiny = Array1::from(vec![0.01, -0.02, 0.015]);
        let result = engle_test(&tiny, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_chi2_survival() {
        // chi2_survival(0.0, 1.0) should be 1.0
        let p = chi2_survival(0.0, 1.0);
        assert!((p - 1.0).abs() < 1e-6);
        // chi2_survival(3.84, 1) should be ~ 0.05
        let p = chi2_survival(3.84, 1.0);
        assert!((p - 0.05).abs() < 0.01);
    }
}
