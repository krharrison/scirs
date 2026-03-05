//! Dynamic Factor Model (DFM)
//!
//! This module implements Dynamic Factor Models (DFMs) using the EM algorithm
//! of Doz, Giannone and Reichlin (2012).  The model decomposes a panel of
//! macroeconomic / financial time series into a small number of latent common
//! factors plus idiosyncratic noise.
//!
//! # Model
//!
//! ## Observation equation
//! ```text
//! X_t = Λ F_t + ε_t,   ε_t ~ N(0, R)       (K × 1)
//! ```
//! where `X_t` is the K-vector of observables at time t, `Λ` (`K × r`) is
//! the factor loading matrix, `F_t` (`r × 1`) is the latent factor vector,
//! and `R` is a diagonal idiosyncratic covariance matrix.
//!
//! ## Factor dynamics (VAR(p))
//! ```text
//! F_t = A_1 F_{t-1} + ... + A_p F_{t-p} + η_t,   η_t ~ N(0, Q)
//! ```
//! Written in companion form with state `s_t = [F_t', ..., F_{t-p+1}']'`
//! this becomes a standard first-order state-space model.
//!
//! # Estimation
//!
//! The EM algorithm alternates between:
//! * **E-step**: run Kalman filter + RTS smoother to compute posterior
//!   moments E[s_t | X] and E[s_t s_{t-1}' | X].
//! * **M-step**: re-estimate Λ, A, R, Q by OLS using the smoothed moments.
//!
//! # References
//! - Doz, C., Giannone, D. & Reichlin, L. (2012). A Quasi Maximum Likelihood
//!   Approach for Large Approximate Dynamic Factor Models. *Review of Economics
//!   and Statistics* 94(4): 1014–1024.
//! - Stock, J.H. & Watson, M.W. (2002). Macroeconomic Forecasting Using
//!   Diffuse Factor Models. *Journal of Business & Economic Statistics*.
//! - Bai, J. & Ng, S. (2002). Determining the Number of Factors in Approximate
//!   Factor Models. *Econometrica* 70(1): 191–221.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};

// ---------------------------------------------------------------------------
// Public data structures
// ---------------------------------------------------------------------------

/// Fitted Dynamic Factor Model.
///
/// All fields correspond to the **last** EM iterate (or the converged
/// solution when `max_iter` is large enough).
#[derive(Clone, Debug)]
pub struct DynamicFactorModel {
    /// Factor loading matrix `Λ` of shape `(n_vars, n_factors)`.
    pub loadings: Array2<f64>,
    /// VAR coefficient matrices `[A_1, ..., A_p]`, each `(n_factors, n_factors)`.
    pub ar_matrices: Vec<Array2<f64>>,
    /// Smoothed factors, shape `(T, n_factors)`.
    pub factors: Array2<f64>,
    /// Idiosyncratic covariance matrix `R`, shape `(n_vars, n_vars)` (diagonal).
    pub r: Array2<f64>,
    /// Factor innovation covariance `Q`, shape `(n_factors, n_factors)`.
    pub q: Array2<f64>,
    /// Number of latent factors.
    pub n_factors: usize,
    /// Number of observed variables.
    pub n_vars: usize,
    /// Number of VAR lags.
    pub n_lags: usize,
    /// EM log-likelihood trajectory (one value per iteration).
    pub log_lik_history: Vec<f64>,
}

impl DynamicFactorModel {
    /// Return the smoothed factor matrix (rows = time, cols = factors).
    ///
    /// This is a convenience alias for the `factors` field.
    pub fn extract_factors(&self) -> Array2<f64> {
        self.factors.clone()
    }

    /// One-step-ahead forecast of the observations at time `T+1`, given
    /// the last smoothed factor state.
    ///
    /// Returns a `K`-vector `X̂_{T+1} = Λ * E[F_{T+1} | X_{1:T}]`.
    pub fn forecast_one_step(&self) -> StatsResult<Array1<f64>> {
        let t = self.factors.nrows();
        if t == 0 {
            return Err(StatsError::InsufficientData(
                "No factor estimates available for forecasting".into(),
            ));
        }
        let r = self.n_factors;
        let p = self.n_lags;

        // Build companion state from the last p factor vectors
        let comp_dim = r * p;
        let mut s_last = Array1::<f64>::zeros(comp_dim);
        for lag in 0..p {
            if t >= lag + 1 {
                let f_row = self.factors.row(t - 1 - lag).to_owned();
                for j in 0..r {
                    s_last[lag * r + j] = f_row[j];
                }
            }
        }

        // Companion transition for one step: F_{T+1} = A_1 F_T + ... + A_p F_{T-p+1}
        let comp = build_companion(&self.ar_matrices, r, p)?;
        let s_next = mat_vec_mul(&comp, &s_last)?;
        let f_next = s_next.slice(scirs2_core::ndarray::s![0..r]).to_owned();

        // X̂ = Λ * F_{T+1}
        let x_hat = mat_vec_mul(&self.loadings, &f_next)?;
        Ok(x_hat)
    }
}

/// Specification of a Dynamic Factor Model before estimation.
#[derive(Clone, Debug)]
pub struct DynamicFactor {
    /// Number of latent factors `r`.
    pub n_factors: usize,
    /// Number of observed variables `K`.
    pub n_vars: usize,
    /// Number of VAR lags `p`.
    pub n_lags: usize,
}

impl DynamicFactor {
    /// Create a new DFM specification.
    pub fn new(n_factors: usize, n_vars: usize, n_lags: usize) -> StatsResult<Self> {
        if n_factors == 0 {
            return Err(StatsError::InvalidInput(
                "DynamicFactor: n_factors must be >= 1".into(),
            ));
        }
        if n_vars == 0 {
            return Err(StatsError::InvalidInput(
                "DynamicFactor: n_vars must be >= 1".into(),
            ));
        }
        if n_lags == 0 {
            return Err(StatsError::InvalidInput(
                "DynamicFactor: n_lags must be >= 1".into(),
            ));
        }
        if n_factors > n_vars {
            return Err(StatsError::InvalidInput(
                "DynamicFactor: n_factors cannot exceed n_vars".into(),
            ));
        }
        Ok(Self {
            n_factors,
            n_vars,
            n_lags,
        })
    }
}

// ---------------------------------------------------------------------------
// EM estimation entry point
// ---------------------------------------------------------------------------

/// Fit a Dynamic Factor Model via the EM algorithm.
///
/// # Arguments
/// * `x`        – `(T, K)` matrix of observations (rows = time, cols = variables).
///   Missing values (NaN) are silently replaced by the cross-sectional mean
///   before the first E-step (a simple pre-treatment).
/// * `n_factors` – number of latent factors `r`.
/// * `n_lags`    – number of VAR lags `p` for the factor dynamics.
/// * `max_iter`  – maximum EM iterations (typically 100–500).
/// * `tol`       – convergence tolerance on the relative log-likelihood change.
///   Pass `None` to use the default `1e-6`.
///
/// # Returns
/// A [`DynamicFactorModel`] containing all estimated parameters and smoothed factors.
///
/// # Errors
/// Returns errors for dimension problems, insufficient data, or numerical
/// failures during Cholesky / inversion.
pub fn fit_dynamic_factor(
    x: &Array2<f64>,
    n_factors: usize,
    n_lags: usize,
    max_iter: usize,
    tol: Option<f64>,
) -> StatsResult<DynamicFactorModel> {
    let big_t = x.nrows();
    let k = x.ncols();
    let r = n_factors;
    let p = n_lags;
    let convergence_tol = tol.unwrap_or(1e-6);

    if big_t <= p + 1 {
        return Err(StatsError::InsufficientData(format!(
            "Need > {} observations for n_lags={}, got {}",
            p + 1,
            p,
            big_t
        )));
    }
    if r == 0 || r > k {
        return Err(StatsError::InvalidInput(format!(
            "n_factors={} must be in 1..=n_vars={}",
            r, k
        )));
    }
    if max_iter == 0 {
        return Err(StatsError::InvalidInput("max_iter must be >= 1".into()));
    }

    // ── Pre-treatment: replace NaN with variable mean ─────────────────────
    let x_clean = impute_column_means(x);

    // ── Initialisation via PCA ─────────────────────────────────────────────
    let (mut lambda, mut factors_init) = pca_init(&x_clean, r)?;
    // Initial R: diagonal of residual variance from PCA
    let mut r_mat = init_idiosyncratic_cov(&x_clean, &lambda, &factors_init, k)?;
    // Initial AR parameters: OLS on the PCA factors
    let mut ar_mats = init_ar_matrices(&factors_init, r, p)?;
    // Initial Q: identity scaled small
    let mut q_mat = Array2::from_diag(&Array1::from_elem(r, 0.1_f64));

    let mut log_lik_history: Vec<f64> = Vec::with_capacity(max_iter);
    let mut prev_log_lik = f64::NEG_INFINITY;

    let mut smoothed_factors = factors_init.clone();

    for _iter in 0..max_iter {
        // ── E-step: Kalman filter + RTS smoother ──────────────────────────
        let (new_factors, ptt, ptt1, log_lik) =
            e_step(&x_clean, &lambda, &ar_mats, &r_mat, &q_mat, r, p)?;

        log_lik_history.push(log_lik);
        smoothed_factors = new_factors;

        // ── M-step: re-estimate parameters ────────────────────────────────
        let (new_lambda, new_r) = m_step_loadings(&x_clean, &smoothed_factors, &ptt, k, r)?;
        let (new_ar, new_q) = m_step_ar(&smoothed_factors, &ptt, &ptt1, r, p)?;

        lambda = new_lambda;
        r_mat = new_r;
        ar_mats = new_ar;
        q_mat = new_q;

        // ── Convergence check ─────────────────────────────────────────────
        let rel_change = if prev_log_lik.is_finite() && prev_log_lik != 0.0 {
            (log_lik - prev_log_lik).abs() / prev_log_lik.abs().max(1.0)
        } else {
            f64::INFINITY
        };
        if rel_change < convergence_tol && _iter > 0 {
            break;
        }
        prev_log_lik = log_lik;
    }

    Ok(DynamicFactorModel {
        loadings: lambda,
        ar_matrices: ar_mats,
        factors: smoothed_factors,
        r: r_mat,
        q: q_mat,
        n_factors: r,
        n_vars: k,
        n_lags: p,
        log_lik_history,
    })
}

// ---------------------------------------------------------------------------
// E-step: Kalman filter + RTS smoother
// ---------------------------------------------------------------------------

/// Returns `(smoothed_factors, P_{t|T}, P_{t,t-1|T}, log_lik)`.
///
/// * `smoothed_factors` – `(T, r)` matrix of smoothed factor means.
/// * `P_{t|T}` – `Vec<Array2<f64>>` of length T, each `(r, r)`.
/// * `P_{t,t-1|T}` – `Vec<Array2<f64>>` of length T-1, each `(r, r)`.
///   Element `s` = Cov(F_{s+1}, F_s | X).
fn e_step(
    x: &Array2<f64>,
    lambda: &Array2<f64>,    // (k, r)
    ar_mats: &[Array2<f64>], // p matrices each (r, r)
    r_mat: &Array2<f64>,     // (k, k)
    q_mat: &Array2<f64>,     // (r, r)
    r_dim: usize,            // n_factors
    p: usize,                // n_lags
) -> StatsResult<(
    Array2<f64>,      // smoothed factors (T, r)
    Vec<Array2<f64>>, // P_{t|T}
    Vec<Array2<f64>>, // P_{t,t-1|T}
    f64,              // log-likelihood
)> {
    let big_t = x.nrows();
    let k = x.ncols();
    let comp_dim = r_dim * p;

    // Build companion transition matrix C and companion Q
    let comp = build_companion(ar_mats, r_dim, p)?;
    let comp_q = build_companion_q(q_mat, r_dim, p);

    // Companion observation matrix: Z = [Λ  0 ... 0]  shape (k, comp_dim)
    let mut z_obs = Array2::<f64>::zeros((k, comp_dim));
    for i in 0..k {
        for j in 0..r_dim {
            z_obs[[i, j]] = lambda[[i, j]];
        }
    }

    // Diffuse initial covariance
    let p0 = Array2::from_diag(&Array1::from_elem(comp_dim, 1e4_f64));
    let x0 = Array1::<f64>::zeros(comp_dim);

    // ── Forward Kalman filter ──────────────────────────────────────────────
    let mut filt_mean: Vec<Array1<f64>> = Vec::with_capacity(big_t);
    let mut filt_cov: Vec<Array2<f64>> = Vec::with_capacity(big_t);
    let mut pred_mean: Vec<Array1<f64>> = Vec::with_capacity(big_t);
    let mut pred_cov: Vec<Array2<f64>> = Vec::with_capacity(big_t);

    let log2pi = (2.0 * std::f64::consts::PI).ln();
    let mut log_lik = 0.0_f64;

    let mut x_cur = x0;
    let mut p_cur = p0;

    for t in 0..big_t {
        // Predict
        let x_pred = mat_vec_mul(&comp, &x_cur)?;
        let cp = mat_mat_mul(&comp, &p_cur)?;
        let mut p_pred = mat_mat_mul_bt(&cp, &comp)? + &comp_q;

        // Ensure P_pred stays symmetric positive-definite
        symmetrize(&mut p_pred);
        regularise_pd(&mut p_pred, 1e-10);

        // Store prediction
        pred_mean.push(x_pred.clone());
        pred_cov.push(p_pred.clone());

        // Observation
        let y_t = x.row(t).to_owned();

        // Innovation
        let z_xp = mat_vec_mul(&z_obs, &x_pred)?;
        let innovation = &y_t - &z_xp;

        // S = Z P_pred Z' + R
        let zp = mat_mat_mul(&z_obs, &p_pred)?;
        let mut s = mat_mat_mul_bt(&zp, &z_obs)? + r_mat;

        // Ensure S stays symmetric positive-definite
        symmetrize(&mut s);
        regularise_pd(&mut s, 1e-6);

        // If S has very small diagonal elements, add stronger regularization
        {
            let s_max_diag = (0..s.nrows())
                .map(|ii| s[[ii, ii]].abs())
                .fold(0.0_f64, f64::max);
            let s_min_diag = (0..s.nrows())
                .map(|ii| s[[ii, ii]].abs())
                .fold(f64::INFINITY, f64::min);
            if s_min_diag < 1e-10 * s_max_diag || s_min_diag < 1e-12 {
                let eps = (1e-6 * s_max_diag).max(1e-8);
                for ii in 0..s.nrows() {
                    s[[ii, ii]] += eps;
                }
            }
        }

        // Log-likelihood
        let s_inv = inv_symmetric(s.clone())?;
        let log_det_s = log_det_posdef(&s)?;
        let sv = mat_vec_mul(&s_inv, &innovation)?;
        let quad: f64 = innovation.iter().zip(sv.iter()).map(|(&a, &b)| a * b).sum();
        log_lik += -0.5 * (k as f64 * log2pi + log_det_s + quad);

        // Kalman gain K = P_pred Z' S^{-1}
        let pzt = mat_mat_mul_bt(&p_pred, &z_obs)?;
        let kgain = mat_mat_mul(&pzt, &s_inv)?;

        // Update
        let kv = mat_vec_mul(&kgain, &innovation)?;
        let x_upd = &x_pred + &kv;
        let kz = mat_mat_mul(&kgain, &z_obs)?;
        let i_kz = eye_minus(kz)?;
        let mut p_upd = mat_mat_mul(&i_kz, &p_pred)?;

        // Ensure P_upd stays symmetric positive-definite
        symmetrize(&mut p_upd);
        regularise_pd(&mut p_upd, 1e-8);

        filt_mean.push(x_upd.clone());
        filt_cov.push(p_upd.clone());
        x_cur = x_upd;
        p_cur = p_upd;
    }

    // ── Backward RTS smoother ──────────────────────────────────────────────
    let mut smooth_mean: Vec<Array1<f64>> = filt_mean.clone();
    let mut smooth_cov: Vec<Array2<f64>> = filt_cov.clone();
    // Cross-covariance Cov(s_{t+1}, s_t | X) needed for M-step
    let mut cross_cov: Vec<Array2<f64>> = vec![Array2::zeros((comp_dim, comp_dim)); big_t];

    for t in (0..big_t - 1).rev() {
        let p_pred_tp1 = &filt_cov[t] * 0.0 + &pred_cov[t + 1]; // clone pred_cov[t+1]
        let p_pred_tp1_inv = inv_symmetric(p_pred_tp1.clone())?;

        // Smoother gain G_t = P_{t|t} C' P_{t+1|t}^{-1}
        let pct = mat_mat_mul_bt(&filt_cov[t], &comp)?;
        let g = mat_mat_mul(&pct, &p_pred_tp1_inv)?;

        // Smoothed mean
        let diff = &smooth_mean[t + 1] - &pred_mean[t + 1];
        let g_diff = mat_vec_mul(&g, &diff)?;
        let x_smooth = &filt_mean[t] + &g_diff;

        // Smoothed covariance
        let dp = &smooth_cov[t + 1] - &p_pred_tp1;
        let g_dp = mat_mat_mul(&g, &dp)?;
        let correction = mat_mat_mul_bt(&g_dp, &g)?;
        let p_smooth = &filt_cov[t] + &correction;

        // Cross-covariance: P_{t+1,t|T} = G_t P_{t+1|T} ... actually
        // the standard formula: Cov(s_{t+1}, s_t | X) = P_{t+1|T} G_t'
        // (de Jong & Mackinnon convention)
        let cross = mat_mat_mul_bt(&smooth_cov[t + 1], &g)?;
        cross_cov[t + 1] = cross;

        smooth_mean[t] = x_smooth;
        smooth_cov[t] = p_smooth;
    }
    // Handle t=0 cross-cov (rarely needed, set to zero)
    cross_cov[0] = Array2::zeros((comp_dim, comp_dim));

    // ── Extract top-r factor block from companion state ────────────────────
    let mut smoothed_factors = Array2::<f64>::zeros((big_t, r_dim));
    for t in 0..big_t {
        for j in 0..r_dim {
            smoothed_factors[[t, j]] = smooth_mean[t][j];
        }
    }

    // Extract r × r covariance blocks
    let p_tt: Vec<Array2<f64>> = smooth_cov
        .iter()
        .map(|p| Array2::from_shape_fn((r_dim, r_dim), |(i, j)| p[[i, j]]))
        .collect();

    let p_tt1: Vec<Array2<f64>> = cross_cov
        .iter()
        .skip(1)
        .map(|cc| Array2::from_shape_fn((r_dim, r_dim), |(i, j)| cc[[i, j]]))
        .collect();

    Ok((smoothed_factors, p_tt, p_tt1, log_lik))
}

// ---------------------------------------------------------------------------
// M-step: loadings and idiosyncratic variances
// ---------------------------------------------------------------------------

/// Update `Λ` and `R` (diagonal) using smoothed factor moments.
///
/// The OLS formula is:
/// ```text
/// Λ = (Σ X_t F_t') (Σ (F_t F_t' + P_{t|T}))^{-1}
/// R = diag( T^{-1} Σ (X_t - Λ F_t)(X_t - Λ F_t)' + Λ P_{t|T} Λ' )
/// ```
fn m_step_loadings(
    x: &Array2<f64>,
    factors: &Array2<f64>, // (T, r)
    p_tt: &[Array2<f64>],  // Vec of (r, r)
    k: usize,
    r: usize,
) -> StatsResult<(Array2<f64>, Array2<f64>)> {
    let big_t = x.nrows();

    // S_xf = Σ X_t F_t'  shape (k, r)
    let mut s_xf = Array2::<f64>::zeros((k, r));
    // S_ff = Σ (F_t F_t' + P_{t|T})  shape (r, r)
    let mut s_ff = Array2::<f64>::zeros((r, r));

    for t in 0..big_t {
        let x_t = x.row(t).to_owned();
        let f_t = factors.row(t).to_owned();
        // S_xf += x_t f_t'
        for i in 0..k {
            for j in 0..r {
                s_xf[[i, j]] += x_t[i] * f_t[j];
            }
        }
        // S_ff += f_t f_t' + P_{t|T}
        for i in 0..r {
            for j in 0..r {
                s_ff[[i, j]] += f_t[i] * f_t[j] + p_tt[t][[i, j]];
            }
        }
    }

    // Λ = S_xf * S_ff^{-1}
    let s_ff_inv = inv_symmetric(s_ff)?;
    let lambda_new = mat_mat_mul(&s_xf, &s_ff_inv)?;

    // Residual variance R (diagonal)
    let mut r_diag = Array1::<f64>::zeros(k);
    for t in 0..big_t {
        let x_t = x.row(t).to_owned();
        let f_t = factors.row(t).to_owned();
        let lf = mat_vec_mul(&lambda_new, &f_t)?;
        let resid = &x_t - &lf;
        // P_tt correction: lambda * P_{t|T} * lambda'
        let lp = mat_mat_mul(&lambda_new, &p_tt[t])?;
        let lpl = mat_mat_mul_bt(&lp, &lambda_new)?;
        for i in 0..k {
            r_diag[i] += resid[i] * resid[i] + lpl[[i, i]];
        }
    }
    r_diag.mapv_inplace(|v| v / big_t as f64);
    // Enforce a minimum idiosyncratic variance to avoid degenerate solutions
    r_diag.mapv_inplace(|v| v.max(1e-6));

    let r_new = Array2::from_diag(&r_diag);

    Ok((lambda_new, r_new))
}

// ---------------------------------------------------------------------------
// M-step: AR parameters and factor innovation covariance
// ---------------------------------------------------------------------------

/// Update `A_1,...,A_p` and `Q` using smoothed factor moments.
///
/// Companion-form OLS:
/// ```text
/// [A_1,...,A_p] = (Σ_{t=p}^T F_t S_{t-1}') (Σ_{t=p}^T S_{t-1} S_{t-1}')^{-1}
/// ```
/// where `S_{t-1} = [F_{t-1}',...,F_{t-p}']'` is the stacked lag vector.
///
/// We use the smoothed second-moment matrices to account for uncertainty.
fn m_step_ar(
    factors: &Array2<f64>, // (T, r)
    p_tt: &[Array2<f64>],  // (T) of (r,r) smoothed covs
    p_tt1: &[Array2<f64>], // (T-1) of (r,r) cross-covs  Cov(F_{t+1}, F_t)
    r: usize,
    p: usize,
) -> StatsResult<(Vec<Array2<f64>>, Array2<f64>)> {
    let big_t = factors.nrows();

    if big_t <= p {
        return Err(StatsError::InsufficientData(
            "m_step_ar: not enough observations for the given number of lags".into(),
        ));
    }

    // Build the regression matrices using the stacked approach
    // For simplicity and robustness we use the sample moments directly
    // (ignoring the uncertainty correction for the lagged covariates).
    // This is the Doz-Giannone-Reichlin approximate M-step.

    let n = big_t - p; // number of regression observations

    // Response: F_{t}  for t = p, ..., T-1  → shape (n, r)
    let y_reg = factors.slice(scirs2_core::ndarray::s![p.., ..]).to_owned(); // (n, r)

    // Regressor: [F_{t-1}', ..., F_{t-p}']  → shape (n, r*p)
    let mut x_reg = Array2::<f64>::zeros((n, r * p));
    for t_idx in 0..n {
        let t = t_idx + p;
        for lag in 0..p {
            let f_lag = factors.row(t - 1 - lag);
            for j in 0..r {
                x_reg[[t_idx, lag * r + j]] = f_lag[j];
            }
        }
    }

    // OLS: B = (X'X)^{-1} X' Y  shape (r*p, r)
    let x_reg_t = x_reg.t().to_owned();
    let xt_x = mat_mat_mul(&x_reg_t, &x_reg)?; // (r*p, r*p)
    let xt_y = mat_mat_mul(&x_reg_t, &y_reg)?; // (r*p, r)

    let xt_x_inv = inv_symmetric_regularised(xt_x, 1e-8)?;
    let b_hat = mat_mat_mul(&xt_x_inv, &xt_y)?; // (r*p, r)

    // Unpack B into A_1, ..., A_p  (each r×r, note: B is transposed wrt equation)
    // b_hat[lag*r .. (lag+1)*r, :] = A_{lag+1}'  so A_{lag+1} = b_hat[..].t()
    let mut ar_mats: Vec<Array2<f64>> = Vec::with_capacity(p);
    for lag in 0..p {
        let a_t = b_hat
            .slice(scirs2_core::ndarray::s![lag * r..(lag + 1) * r, ..])
            .to_owned();
        // a_t is (r, r), rows are coefficients for response dimension
        ar_mats.push(a_t.t().to_owned());
    }

    // Q: innovation covariance
    // Q = T^{-1} Σ (F_t - Â F_{t-1:t-p})(...)' + uncertainty corrections
    let mut q_new = Array2::<f64>::zeros((r, r));
    for t_idx in 0..n {
        let t = t_idx + p;
        let f_t = factors.row(t).to_owned();
        // Predicted factor
        let reg_t = x_reg.row(t_idx).to_owned();
        let f_pred: Array1<f64> = {
            let reg_t_mat = reg_t
                .view()
                .into_shape_with_order((1, r * p))
                .map_err(|_| StatsError::ComputationError("reshape failed".into()))?
                .to_owned();
            let pred_mat = mat_mat_mul(&reg_t_mat, &b_hat)?;
            pred_mat.row(0).to_owned()
        };
        let resid = &f_t - &f_pred;
        for i in 0..r {
            for j in 0..r {
                q_new[[i, j]] += resid[i] * resid[j];
            }
        }
        // Add P_{t|T} contribution
        if t < p_tt.len() {
            for i in 0..r {
                for j in 0..r {
                    q_new[[i, j]] += p_tt[t][[i, j]];
                }
            }
        }
    }
    q_new.mapv_inplace(|v| v / n as f64);

    // Ensure Q is symmetric positive definite
    symmetrize(&mut q_new);
    regularise_pd(&mut q_new, 1e-8);

    Ok((ar_mats, q_new))
}

// ---------------------------------------------------------------------------
// Initialisation via PCA
// ---------------------------------------------------------------------------

/// Compute the top-`r` principal components of the data matrix `X` (T × K).
///
/// Returns `(Λ, F)` where `Λ` is `(K, r)` and `F` is `(T, r)`.
fn pca_init(x: &Array2<f64>, r: usize) -> StatsResult<(Array2<f64>, Array2<f64>)> {
    let big_t = x.nrows();
    let k = x.ncols();
    if r > k.min(big_t) {
        return Err(StatsError::InvalidInput(format!(
            "pca_init: r={} > min(T,K)={}",
            r,
            k.min(big_t)
        )));
    }

    // Demean columns
    let col_means: Array1<f64> = x
        .mean_axis(Axis(0))
        .ok_or_else(|| StatsError::ComputationError("pca_init: mean_axis failed".into()))?;
    let mut x_dm = x.clone();
    for mut row in x_dm.rows_mut() {
        for (v, &m) in row.iter_mut().zip(col_means.iter()) {
            *v -= m;
        }
    }

    // Sample covariance: C = X' X / T  (K × K)
    let xt = x_dm.t().to_owned();
    let cov = mat_mat_mul(&xt, &x_dm).map(|c| c.mapv(|v| v / big_t as f64))?;

    // Power iteration to get the top-r eigenvectors
    let evecs = power_iter_top_r(&cov, r, 500)?; // (K, r)

    // Factors: F = X * V  (T × r)
    let factors = mat_mat_mul(&x_dm, &evecs)?; // T × r

    Ok((evecs, factors))
}

/// Power iteration to extract the top `r` eigenvectors of a symmetric matrix.
///
/// Uses deflation (subtracting the outer product after each extraction).
fn power_iter_top_r(a: &Array2<f64>, r: usize, n_iter: usize) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    let mut current = a.clone();
    let mut evecs = Array2::<f64>::zeros((n, r));

    for k in 0..r {
        // Start from a random-ish vector (deterministic: column k of identity)
        let mut v = Array1::<f64>::zeros(n);
        v[k % n] = 1.0;
        if k < n {
            v[k] = 1.0;
        } else {
            v[0] = 1.0;
        }
        // Normalise
        let norm = (v.iter().map(|x| x * x).sum::<f64>()).sqrt();
        if norm > 1e-15 {
            v.mapv_inplace(|x| x / norm);
        }

        for _ in 0..n_iter {
            let av = mat_vec_mul(&current, &v)?;
            let new_norm = (av.iter().map(|x| x * x).sum::<f64>()).sqrt();
            if new_norm < 1e-15 {
                break;
            }
            v = av.mapv(|x| x / new_norm);
        }

        // Store eigenvector
        for i in 0..n {
            evecs[[i, k]] = v[i];
        }

        // Deflate: remove contribution of this eigenvector
        let eigenvalue = {
            let av = mat_vec_mul(&current, &v)?;
            av.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum::<f64>()
        };
        for i in 0..n {
            for j in 0..n {
                current[[i, j]] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    Ok(evecs)
}

/// Estimate initial idiosyncratic covariance from PCA residuals.
fn init_idiosyncratic_cov(
    x: &Array2<f64>,
    lambda: &Array2<f64>,
    factors: &Array2<f64>,
    k: usize,
) -> StatsResult<Array2<f64>> {
    let big_t = x.nrows();
    let lf = mat_mat_mul(factors, &lambda.t().to_owned())?; // T × k
    let resid = x - &lf;
    let mut r_diag = Array1::<f64>::zeros(k);
    for t in 0..big_t {
        let row = resid.row(t);
        for i in 0..k {
            r_diag[i] += row[i] * row[i];
        }
    }
    r_diag.mapv_inplace(|v| (v / big_t as f64).max(1e-6));
    Ok(Array2::from_diag(&r_diag))
}

/// Estimate initial VAR parameters by OLS on PCA factors.
fn init_ar_matrices(
    factors: &Array2<f64>, // (T, r)
    r: usize,
    p: usize,
) -> StatsResult<Vec<Array2<f64>>> {
    let big_t = factors.nrows();
    if big_t <= p {
        // Fall back to zero matrices
        return Ok(vec![Array2::zeros((r, r)); p]);
    }

    let n = big_t - p;
    let y_reg = factors.slice(scirs2_core::ndarray::s![p.., ..]).to_owned();
    let mut x_reg = Array2::<f64>::zeros((n, r * p));
    for t_idx in 0..n {
        let t = t_idx + p;
        for lag in 0..p {
            let f_lag = factors.row(t - 1 - lag);
            for j in 0..r {
                x_reg[[t_idx, lag * r + j]] = f_lag[j];
            }
        }
    }

    // OLS: b_hat = (X^T X)^{-1} X^T Y
    let x_reg_t = x_reg.t().to_owned();
    let xt_x = mat_mat_mul(&x_reg_t, &x_reg)?; // (r*p, r*p)
    let xt_y = mat_mat_mul(&x_reg_t, &y_reg)?; // (r*p, r)
    let xt_x_inv = inv_symmetric_regularised(xt_x, 1e-8)?; // (r*p, r*p)
    let b_hat = mat_mat_mul(&xt_x_inv, &xt_y)?; // (r*p, r)

    let mut ar_mats: Vec<Array2<f64>> = Vec::with_capacity(p);
    for lag in 0..p {
        let a_t = b_hat
            .slice(scirs2_core::ndarray::s![lag * r..(lag + 1) * r, ..])
            .to_owned();
        ar_mats.push(a_t.t().to_owned());
    }
    Ok(ar_mats)
}

// ---------------------------------------------------------------------------
// Companion form utilities
// ---------------------------------------------------------------------------

/// Build the `(r*p, r*p)` companion transition matrix from AR matrices.
fn build_companion(ar_mats: &[Array2<f64>], r: usize, p: usize) -> StatsResult<Array2<f64>> {
    let dim = r * p;
    let mut comp = Array2::<f64>::zeros((dim, dim));

    for (lag, a) in ar_mats.iter().enumerate() {
        if a.nrows() != r || a.ncols() != r {
            return Err(StatsError::DimensionMismatch(format!(
                "AR matrix {} has shape {}×{}, expected {}×{}",
                lag,
                a.nrows(),
                a.ncols(),
                r,
                r
            )));
        }
        for i in 0..r {
            for j in 0..r {
                comp[[i, lag * r + j]] = a[[i, j]];
            }
        }
    }
    // Sub-diagonal identity blocks
    for i in 1..p {
        for j in 0..r {
            comp[[i * r + j, (i - 1) * r + j]] = 1.0;
        }
    }
    Ok(comp)
}

/// Build the companion process-noise covariance (only top-left `r×r` block).
fn build_companion_q(q: &Array2<f64>, r: usize, p: usize) -> Array2<f64> {
    let dim = r * p;
    let mut cq = Array2::<f64>::zeros((dim, dim));
    for i in 0..r {
        for j in 0..r {
            cq[[i, j]] = q[[i, j]];
        }
    }
    // Add small process noise to lagged state dimensions to prevent
    // the covariance from degenerating when p > 1.
    // Without this, the lower-right block of P_pred can become
    // non-positive-definite, causing Cholesky failures in the E-step.
    if p > 1 {
        let q_trace = (0..r).map(|i| q[[i, i]]).sum::<f64>() / r as f64;
        let eps = (1e-8 * q_trace).max(1e-12);
        for i in r..dim {
            cq[[i, i]] = eps;
        }
    }
    cq
}

// ---------------------------------------------------------------------------
// Data utilities
// ---------------------------------------------------------------------------

/// Replace NaN values in each column with the column mean.
fn impute_column_means(x: &Array2<f64>) -> Array2<f64> {
    let big_t = x.nrows();
    let k = x.ncols();
    let mut out = x.clone();
    for j in 0..k {
        let col = x.column(j);
        let (sum, cnt) = col
            .iter()
            .filter(|v| v.is_finite())
            .fold((0.0_f64, 0usize), |(s, n), &v| (s + v, n + 1));
        if cnt == 0 {
            continue;
        }
        let mean = sum / cnt as f64;
        for i in 0..big_t {
            if !out[[i, j]].is_finite() {
                out[[i, j]] = mean;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Numerical linear-algebra helpers (private)
// ---------------------------------------------------------------------------

fn mat_vec_mul(a: &Array2<f64>, x: &Array1<f64>) -> StatsResult<Array1<f64>> {
    if a.ncols() != x.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "mat_vec_mul: A is {}×{} but x has len {}",
            a.nrows(),
            a.ncols(),
            x.len()
        )));
    }
    let n = a.nrows();
    let m = a.ncols();
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = 0.0_f64;
        for k in 0..m {
            s += a[[i, k]] * x[k];
        }
        y[i] = s;
    }
    Ok(y)
}

fn mat_mat_mul(a: &Array2<f64>, b: &Array2<f64>) -> StatsResult<Array2<f64>> {
    if a.ncols() != b.nrows() {
        return Err(StatsError::DimensionMismatch(format!(
            "mat_mat_mul: A is {}×{} but B is {}×{}",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        )));
    }
    let n = a.nrows();
    let kk = a.ncols();
    let m = b.ncols();
    let mut c = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0_f64;
            for l in 0..kk {
                s += a[[i, l]] * b[[l, j]];
            }
            c[[i, j]] = s;
        }
    }
    Ok(c)
}

fn mat_mat_mul_bt(a: &Array2<f64>, b: &Array2<f64>) -> StatsResult<Array2<f64>> {
    if a.ncols() != b.ncols() {
        return Err(StatsError::DimensionMismatch(format!(
            "mat_mat_mul_bt: A is {}×{} but B^T is {}×{}",
            a.nrows(),
            a.ncols(),
            b.ncols(),
            b.nrows()
        )));
    }
    let n = a.nrows();
    let kk = a.ncols();
    let m = b.nrows();
    let mut c = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0_f64;
            for l in 0..kk {
                s += a[[i, l]] * b[[j, l]];
            }
            c[[i, j]] = s;
        }
    }
    Ok(c)
}

fn eye_minus(a: Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(StatsError::DimensionMismatch(
            "eye_minus: not square".into(),
        ));
    }
    let mut result = -a;
    for i in 0..n {
        result[[i, i]] += 1.0;
    }
    Ok(result)
}

fn cholesky_lower(a: Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(StatsError::DimensionMismatch(
            "cholesky_lower: not square".into(),
        ));
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for kk in 0..j {
                s -= l[[i, kk]] * l[[j, kk]];
            }
            if i == j {
                if s <= 0.0 {
                    let eps = (1e-10_f64).max(s.abs() * 1e-8);
                    let s_reg = s + eps;
                    if s_reg <= 0.0 {
                        return Err(StatsError::ComputationError(format!(
                            "Cholesky failed at ({},{}): s={}",
                            i, j, s
                        )));
                    }
                    l[[i, j]] = s_reg.sqrt();
                } else {
                    l[[i, j]] = s.sqrt();
                }
            } else {
                let ljj = l[[j, j]];
                if ljj.abs() < 1e-15 {
                    return Err(StatsError::ComputationError(
                        "Cholesky: near-zero diagonal".into(),
                    ));
                }
                l[[i, j]] = s / ljj;
            }
        }
    }
    Ok(l)
}

fn lower_tri_inv(l: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = l.nrows();
    let mut inv = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        inv[[j, j]] = 1.0 / l[[j, j]];
        for i in j + 1..n {
            let mut s = 0.0_f64;
            for kk in j..i {
                s += l[[i, kk]] * inv[[kk, j]];
            }
            inv[[i, j]] = -s / l[[i, i]];
        }
    }
    Ok(inv)
}

fn inv_symmetric(a: Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(StatsError::DimensionMismatch(
            "inv_symmetric: not square".into(),
        ));
    }
    if n == 1 {
        let val = a[[0, 0]];
        if val.abs() < 1e-15 {
            return Err(StatsError::ComputationError(
                "inv_symmetric: singular 1×1".into(),
            ));
        }
        let mut inv = Array2::<f64>::zeros((1, 1));
        inv[[0, 0]] = 1.0 / val;
        return Ok(inv);
    }

    // Maximum absolute diagonal element for scaling regularisation
    let max_diag = (0..n)
        .map(|i| a[[i, i]].abs())
        .fold(0.0_f64, f64::max)
        .max(1e-12);

    // Try Cholesky with escalating regularisation
    let regularisation_levels = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0];
    for &eps_scale in &regularisation_levels {
        let mut reg_a = a.clone();
        let eps = eps_scale * max_diag;
        if eps > 0.0 {
            for i in 0..n {
                reg_a[[i, i]] += eps;
            }
        }
        match cholesky_lower(reg_a) {
            Ok(l) => {
                let l_inv = lower_tri_inv(&l)?;
                let l_inv_t = l_inv.t().to_owned();
                return mat_mat_mul(&l_inv_t, &l_inv);
            }
            Err(_) => continue,
        }
    }
    // Cholesky failed at all levels — fall back to Gauss-Jordan elimination
    // with partial pivoting for robustness.
    gauss_jordan_inv(&a)
}

/// Gauss-Jordan elimination with partial pivoting for matrix inversion.
/// Used as a fallback when Cholesky decomposition fails.
fn gauss_jordan_inv(a: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    // Augmented matrix [A | I]
    let mut aug = Array2::<f64>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    for col in 0..n {
        // Partial pivoting: find the row with the largest absolute value in this column
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return Err(StatsError::ComputationError(
                "gauss_jordan_inv: singular matrix".into(),
            ));
        }
        // Swap rows
        if max_row != col {
            for j in 0..(2 * n) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }
        // Scale pivot row
        let pivot = aug[[col, col]];
        for j in 0..(2 * n) {
            aug[[col, j]] /= pivot;
        }
        // Eliminate column in all other rows
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[[row, col]];
            for j in 0..(2 * n) {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    // Extract inverse from the right half
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    Ok(inv)
}

/// Invert with Tikhonov regularisation for numerical safety.
fn inv_symmetric_regularised(mut a: Array2<f64>, reg: f64) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    for i in 0..n {
        a[[i, i]] += reg;
    }
    inv_symmetric(a)
}

fn log_det_posdef(a: &Array2<f64>) -> StatsResult<f64> {
    let n = a.nrows();
    // Try Cholesky with escalating regularisation to handle near-singular innovation covariance
    let max_diag = (0..n)
        .map(|i| a[[i, i]].abs())
        .fold(0.0_f64, f64::max)
        .max(1e-12);

    let reg_levels = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0];
    for &eps_scale in &reg_levels {
        let mut reg_a = a.clone();
        let eps = eps_scale * max_diag;
        if eps > 0.0 {
            for i in 0..n {
                reg_a[[i, i]] += eps;
            }
        }
        match cholesky_lower(reg_a) {
            Ok(l) => {
                let log_det: f64 = (0..n).map(|i| 2.0 * l[[i, i]].ln()).sum();
                return Ok(log_det);
            }
            Err(_) => continue,
        }
    }
    Err(StatsError::ComputationError(
        "log_det_posdef: Cholesky failed after regularisation".into(),
    ))
}

fn symmetrize(a: &mut Array2<f64>) {
    let n = a.nrows();
    for i in 0..n {
        for j in i + 1..n {
            let v = (a[[i, j]] + a[[j, i]]) * 0.5;
            a[[i, j]] = v;
            a[[j, i]] = v;
        }
    }
}

fn regularise_pd(a: &mut Array2<f64>, eps: f64) {
    let n = a.nrows();
    for i in 0..n {
        a[[i, i]] = a[[i, i]].max(eps);
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn synthetic_data(t: usize, k: usize, r: usize) -> Array2<f64> {
        // Generate data as: X = F * Λ' + ε
        // where F is (T, r) random walk and Λ is (k, r) fixed
        let lambda: Array2<f64> =
            Array2::from_shape_fn((k, r), |(i, j)| ((i + j + 1) as f64) * 0.5);
        let mut factors = Array2::<f64>::zeros((t, r));
        for t_idx in 1..t {
            for j in 0..r {
                let prev = factors[[t_idx - 1, j]];
                // Deterministic trend + small oscillation
                factors[[t_idx, j]] = prev * 0.9 + (t_idx as f64 * 0.1 + j as f64).sin() * 0.5;
            }
        }
        let mut x = Array2::<f64>::zeros((t, k));
        for t_idx in 0..t {
            for i in 0..k {
                let mut val = 0.0_f64;
                for j in 0..r {
                    val += factors[[t_idx, j]] * lambda[[i, j]];
                }
                // Simple deterministic noise
                val += (t_idx as f64 * 0.3 + i as f64).cos() * 0.1;
                x[[t_idx, i]] = val;
            }
        }
        x
    }

    #[test]
    fn test_dfm_basic_fit() {
        let data = synthetic_data(50, 4, 2);
        let model = fit_dynamic_factor(&data, 2, 1, 30, None).expect("fit_dynamic_factor");
        assert_eq!(model.n_factors, 2);
        assert_eq!(model.n_vars, 4);
        assert_eq!(model.n_lags, 1);
        assert_eq!(model.factors.nrows(), 50);
        assert_eq!(model.factors.ncols(), 2);
        assert_eq!(model.loadings.nrows(), 4);
        assert_eq!(model.loadings.ncols(), 2);
        assert!(!model.log_lik_history.is_empty());
        let last_ll = *model.log_lik_history.last().expect("log_lik");
        assert!(last_ll.is_finite(), "log-likelihood must be finite");
    }

    #[test]
    fn test_dfm_extract_factors() {
        let data = synthetic_data(40, 5, 1);
        let model = fit_dynamic_factor(&data, 1, 1, 20, None).expect("fit");
        let factors = model.extract_factors();
        assert_eq!(factors.nrows(), 40);
        assert_eq!(factors.ncols(), 1);
    }

    #[test]
    fn test_dfm_forecast_one_step() {
        let data = synthetic_data(50, 4, 2);
        let model = fit_dynamic_factor(&data, 2, 1, 20, None).expect("fit");
        let forecast = model.forecast_one_step().expect("forecast");
        assert_eq!(forecast.len(), 4);
        assert!(forecast.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_dfm_with_two_lags() {
        // Use more observations and fewer variables to give the 2-lag model
        // enough data for numerically stable Cholesky factorisations.
        let data = synthetic_data(120, 4, 2);
        let model = fit_dynamic_factor(&data, 2, 2, 20, Some(1e-4)).expect("fit p=2");
        assert_eq!(model.ar_matrices.len(), 2);
        assert_eq!(model.factors.nrows(), 120);
    }

    #[test]
    fn test_dfm_invalid_inputs() {
        let data = synthetic_data(30, 4, 2);
        // n_factors > n_vars is invalid
        assert!(fit_dynamic_factor(&data, 5, 1, 10, None).is_err());
        // Insufficient observations
        let tiny = synthetic_data(3, 4, 1);
        assert!(fit_dynamic_factor(&tiny, 1, 3, 10, None).is_err());
    }

    #[test]
    fn test_dynamic_factor_spec() {
        let spec = DynamicFactor::new(2, 5, 1).expect("spec");
        assert_eq!(spec.n_factors, 2);
        assert_eq!(spec.n_vars, 5);
        assert_eq!(spec.n_lags, 1);
        // Invalid: n_factors > n_vars
        assert!(DynamicFactor::new(6, 5, 1).is_err());
        // Invalid: zero
        assert!(DynamicFactor::new(0, 5, 1).is_err());
    }

    #[test]
    fn test_impute_column_means() {
        let mut x = Array2::<f64>::zeros((4, 2));
        x[[0, 0]] = 1.0;
        x[[1, 0]] = f64::NAN;
        x[[2, 0]] = 3.0;
        x[[3, 0]] = 4.0;
        let out = impute_column_means(&x);
        // Column 0 mean of [1, 3, 4] = 8/3 ≈ 2.667
        assert!(out[[1, 0]].is_finite());
        assert!((out[[1, 0]] - (1.0 + 3.0 + 4.0) / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_pca_init_dimensions() {
        let data = synthetic_data(30, 5, 2);
        let (lambda, factors) = pca_init(&data, 2).expect("pca_init");
        assert_eq!(lambda.nrows(), 5);
        assert_eq!(lambda.ncols(), 2);
        assert_eq!(factors.nrows(), 30);
        assert_eq!(factors.ncols(), 2);
    }
}
