//! Random-Effects (GLS) Panel Estimator — Balestra & Nerlove (1966)
//!
//! The random-effects estimator treats the individual effects `α_i` as random
//! variables drawn from a distribution with mean 0 and variance `σ²_u`.
//!
//! # GLS Transformation
//!
//! Under the random-effects assumption:
//! ```text
//! Ω = σ²_e I_{NT} + σ²_u (I_N ⊗ ι_T ι_T')
//! ```
//! The quasi-demeaning transformation subtracts `θ = 1 − σ_e/sqrt(T σ²_u + σ²_e)`
//! of the individual mean:
//! ```text
//! ỹ_{it} = y_{it} − θ ȳ_i
//! ```
//! The RE estimator is then OLS on the quasi-demeaned data.
//!
//! # Variance Components Estimation
//!
//! We use the Swamy-Arora method:
//! - `σ̂²_e = RSS_FE / (N*(T-1) - K)` (from within residuals)
//! - `σ̂²_u = max(0, σ̂²_be − σ̂²_e/T)` (from between residuals)

use crate::error::{Result, TimeSeriesError};
use crate::panel::fixed_effects::{invert_sym, mat_vec_mul, PanelData};

// ============================================================
// Random-Effects model struct
// ============================================================

/// Random-Effects (GLS) estimator for balanced panel data.
#[derive(Debug, Clone)]
pub struct RandomEffectsModel {
    /// GLS slope coefficients β̂_RE (length K)
    pub coefficients: Vec<f64>,
    /// Standard errors of β̂_RE
    pub std_errors: Vec<f64>,
    /// t-statistics for H₀: β_k = 0
    pub t_stats: Vec<f64>,
    /// Idiosyncratic error variance σ̂²_e
    pub sigma_e: f64,
    /// Random-effect variance σ̂²_u
    pub sigma_u: f64,
    /// Quasi-demeaning parameter θ
    pub theta: f64,
    /// Within R² (after quasi-demeaning)
    pub r_squared: f64,
    /// Number of individuals N
    pub n_individuals: usize,
    /// Number of time periods T
    pub n_periods: usize,
    /// Number of regressors K
    pub n_regressors: usize,
    /// (X̃'Ω⁻¹X̃)⁻¹ — used in Hausman test
    pub(crate) gls_cov: Vec<f64>,
}

impl RandomEffectsModel {
    /// Fit the random-effects estimator to balanced panel data.
    ///
    /// Uses the Swamy-Arora two-step approach:
    /// 1. Run FE to get `σ̂²_e`
    /// 2. Run between regression to get `σ̂²_u`
    /// 3. Construct `θ` and quasi-demean
    /// 4. OLS on quasi-demeaned data
    pub fn fit(data: &PanelData) -> Result<Self> {
        let n = data.n_individuals;
        let t = data.n_periods;
        let k = data.n_regressors;
        let n_obs = n * t;

        if n_obs < k + 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "RE: too few observations".into(),
                required: k + 2,
                actual: n_obs,
            });
        }

        // ---- Step 1: Within (FE) residuals → σ̂²_e ----
        let fe = crate::panel::fixed_effects::FixedEffectsModel::fit(data)?;
        let sigma2_e = fe.sigma2.max(1e-15);

        // ---- Step 2: Between regression → σ̂²_be ----
        // Between means: ȳ_i, X̄_i
        let mut y_bar = vec![0.0_f64; n];
        let mut x_bar = vec![vec![0.0_f64; k]; n];

        for i in 0..n {
            y_bar[i] = data.y[i].iter().sum::<f64>() / t as f64;
            for j in 0..k {
                x_bar[i][j] = data.x[i].iter().map(|xt| xt[j]).sum::<f64>() / t as f64;
            }
        }

        // OLS on between data (including intercept)
        let k_be = k + 1;
        let mut btb = vec![0.0_f64; k_be * k_be];
        let mut bty = vec![0.0_f64; k_be];

        for i in 0..n {
            let row_be: Vec<f64> = std::iter::once(1.0_f64)
                .chain(x_bar[i].iter().copied())
                .collect();
            for (a, &ra) in row_be.iter().enumerate() {
                bty[a] += ra * y_bar[i];
                for (b, &rb) in row_be.iter().enumerate() {
                    btb[a * k_be + b] += ra * rb;
                }
            }
        }

        let beta_be = match invert_sym(k_be, &btb) {
            Ok(inv) => mat_vec_mul(k_be, &inv, &bty),
            Err(_) => {
                // If between matrix is singular (e.g. no between variation), default to zeros
                vec![0.0_f64; k_be]
            }
        };

        // Between residuals → σ̂²_be
        let mut ss_be = 0.0_f64;
        for i in 0..n {
            let xb: f64 = beta_be[0]
                + (0..k).map(|j| x_bar[i][j] * beta_be[j + 1]).sum::<f64>();
            ss_be += (y_bar[i] - xb).powi(2);
        }

        let df_be = (n as f64 - k_be as f64).max(1.0);
        let sigma2_be = ss_be / df_be;

        // ---- Step 3: Variance components ----
        let sigma2_u = (sigma2_be - sigma2_e / t as f64).max(0.0);

        // θ = 1 - σ_e / sqrt(T * σ²_u + σ²_e)
        let denom_theta = (t as f64 * sigma2_u + sigma2_e).sqrt().max(1e-15);
        let theta = 1.0 - sigma2_e.sqrt() / denom_theta;
        let theta = theta.clamp(0.0, 1.0);

        // ---- Step 4: Quasi-demean and OLS ----
        let mut y_qd = Vec::with_capacity(n_obs);
        let mut x_qd = Vec::with_capacity(n_obs * k);

        for i in 0..n {
            for tt in 0..t {
                y_qd.push(data.y[i][tt] - theta * y_bar[i]);
                for j in 0..k {
                    x_qd.push(data.x[i][tt][j] - theta * x_bar[i][j]);
                }
            }
        }

        let mut xtx = vec![0.0_f64; k * k];
        let mut xty = vec![0.0_f64; k];

        for obs in 0..n_obs {
            for a in 0..k {
                xty[a] += x_qd[obs * k + a] * y_qd[obs];
                for b in 0..k {
                    xtx[a * k + b] += x_qd[obs * k + a] * x_qd[obs * k + b];
                }
            }
        }

        let gls_cov = invert_sym(k, &xtx)?;
        let beta = mat_vec_mul(k, &gls_cov, &xty);

        // Residuals and R²
        let y_qd_mean = y_qd.iter().sum::<f64>() / n_obs as f64;
        let mut ss_res = 0.0_f64;
        let mut ss_tot = 0.0_f64;
        for obs in 0..n_obs {
            let xb: f64 = (0..k).map(|j| x_qd[obs * k + j] * beta[j]).sum();
            ss_res += (y_qd[obs] - xb).powi(2);
            ss_tot += (y_qd[obs] - y_qd_mean).powi(2);
        }

        let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
        let sigma2_resid = ss_res / (n_obs as f64 - k as f64).max(1.0);

        // Scale GLS covariance by residual variance
        let gls_cov_scaled: Vec<f64> = gls_cov.iter().map(|&v| v * sigma2_resid).collect();

        let std_errors: Vec<f64> = (0..k)
            .map(|j| (gls_cov_scaled[j * k + j]).sqrt().max(0.0))
            .collect();

        let t_stats: Vec<f64> = beta
            .iter()
            .zip(std_errors.iter())
            .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
            .collect();

        Ok(Self {
            coefficients: beta,
            std_errors,
            t_stats,
            sigma_e: sigma2_e.sqrt(),
            sigma_u: sigma2_u.sqrt(),
            theta,
            r_squared,
            n_individuals: n,
            n_periods: t,
            n_regressors: k,
            gls_cov: gls_cov_scaled,
        })
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_panel(n: usize, t: usize, true_beta: f64) -> PanelData {
        // DGP: y_it = alpha_i + true_beta * x_it + eps
        // alpha_i ~ U[0, 0.5*i] (random effects)
        let mut y = vec![vec![0.0; t]; n];
        let mut x = vec![vec![vec![0.0; 1]; t]; n];

        for i in 0..n {
            let alpha_i = (i as f64) * 0.3;
            for tt in 0..t {
                let xit = (i as f64 * 0.7 + tt as f64 * 0.4).sin() * 2.0 + 1.0;
                x[i][tt][0] = xit;
                y[i][tt] = alpha_i + true_beta * xit;
            }
        }

        PanelData::new(y, x).expect("PanelData")
    }

    #[test]
    fn test_re_fit_basic() {
        let data = make_panel(8, 6, 1.5);
        let model = RandomEffectsModel::fit(&data).expect("RE fit");
        // With exact DGP, RE should recover β ≈ 1.5
        assert!(
            (model.coefficients[0] - 1.5).abs() < 0.1,
            "β̂_RE = {:.4}, expected ≈ 1.5",
            model.coefficients[0]
        );
    }

    #[test]
    fn test_re_theta_in_range() {
        let data = make_panel(8, 6, 1.5);
        let model = RandomEffectsModel::fit(&data).expect("RE fit");
        assert!(
            (0.0..=1.0).contains(&model.theta),
            "θ must be in [0,1], got {}",
            model.theta
        );
    }

    #[test]
    fn test_re_sigma_positive() {
        let data = make_panel(8, 6, 1.5);
        let model = RandomEffectsModel::fit(&data).expect("RE fit");
        assert!(model.sigma_e >= 0.0);
        assert!(model.sigma_u >= 0.0);
    }
}
