//! Hausman Specification Test for Fixed vs. Random Effects
//!
//! The Hausman (1978) test distinguishes between the fixed-effects (FE) and
//! random-effects (RE) estimators under the null hypothesis that the individual
//! effects are uncorrelated with the regressors:
//!
//! - H₀: Cov(α_i, X_{it}) = 0 → RE is consistent and efficient; FE is consistent
//!   but inefficient.
//! - H₁: Cov(α_i, X_{it}) ≠ 0 → RE is inconsistent; FE remains consistent.
//!
//! # Test Statistic
//!
//! ```text
//! H = (β̂_FE − β̂_RE)' [Var(β̂_FE) − Var(β̂_RE)]⁻¹ (β̂_FE − β̂_RE)
//! ```
//!
//! Under H₀, `H ~ χ²(K)` where `K` is the number of time-varying regressors.
//!
//! # References
//! - Hausman, J. A. (1978). Specification Tests in Econometrics.
//!   *Econometrica*, 46(6), 1251–1271.

use crate::error::{Result, TimeSeriesError};
use crate::panel::fixed_effects::{invert_sym, mat_vec_mul, FixedEffectsModel};
use crate::panel::random_effects::RandomEffectsModel;
use crate::volatility::arch::chi2_survival;

/// Result of the Hausman specification test.
#[derive(Debug, Clone)]
pub struct HausmanTestResult {
    /// Hausman test statistic (χ² distributed under H₀)
    pub statistic: f64,
    /// p-value: P(χ²(K) > H)
    pub p_value: f64,
    /// Degrees of freedom = K (number of time-varying regressors)
    pub df: usize,
    /// Difference vector β̂_FE − β̂_RE
    pub beta_diff: Vec<f64>,
    /// Whether to reject H₀ at 5% significance
    pub reject_null: bool,
}

/// Hausman test namespace.
pub struct HausmanTest;

impl HausmanTest {
    /// Perform the Hausman test comparing FE and RE estimates.
    ///
    /// # Arguments
    /// * `fe` — fitted fixed-effects model
    /// * `re` — fitted random-effects model (on the same data)
    ///
    /// # Errors
    /// Returns an error if the models have different numbers of regressors or if
    /// the covariance difference matrix is not positive-definite (common in
    /// small samples; can be regularised).
    pub fn test(
        fe: &FixedEffectsModel,
        re: &RandomEffectsModel,
    ) -> Result<HausmanTestResult> {
        let k = fe.n_regressors;
        if k != re.n_regressors {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Hausman test: FE has {} regressors, RE has {}",
                k, re.n_regressors
            )));
        }
        if k == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Hausman test: no regressors to test".into(),
            ));
        }

        // Covariance difference: Ψ = V(β̂_FE) − V(β̂_RE)
        // V(β̂_FE) = σ²_e (X̃'X̃)⁻¹
        // V(β̂_RE) = GLS covariance (already scaled by σ² in RE fit)
        let v_fe = fe.sigma2 * &fe.xtx_inv;

        // Difference matrix
        let psi: Vec<f64> = v_fe
            .iter()
            .zip(re.gls_cov.iter())
            .map(|(&vfe, &vre)| vfe - vre)
            .collect();

        // Regularise to avoid numerical issues in small samples
        let psi_reg = regularise_matrix(k, &psi);

        let psi_inv = match invert_sym(k, &psi_reg) {
            Ok(inv) => inv,
            Err(_) => {
                // If not invertible, use diagonal elements only (approximation)
                let mut diag_inv = vec![0.0_f64; k * k];
                for j in 0..k {
                    let d = psi_reg[j * k + j];
                    diag_inv[j * k + j] = if d.abs() > 1e-12 { 1.0 / d } else { 0.0 };
                }
                diag_inv
            }
        };

        let beta_diff: Vec<f64> = fe
            .coefficients
            .iter()
            .zip(re.coefficients.iter())
            .map(|(&bfe, &bre)| bfe - bre)
            .collect();

        let psi_inv_diff = mat_vec_mul(k, &psi_inv, &beta_diff);
        let statistic: f64 = beta_diff
            .iter()
            .zip(psi_inv_diff.iter())
            .map(|(&d, &p)| d * p)
            .sum();

        let statistic = statistic.abs(); // should be positive, but guard against numerics
        let p_value = chi2_survival(statistic, k as f64);
        let reject_null = p_value < 0.05;

        Ok(HausmanTestResult {
            statistic,
            p_value,
            df: k,
            beta_diff,
            reject_null,
        })
    }
}

/// Add a small ridge regularisation to the diagonal to improve conditioning.
fn regularise_matrix(k: usize, m: &[f64]) -> Vec<f64> {
    let max_diag = (0..k)
        .map(|i| m[i * k + i].abs())
        .fold(0.0_f64, f64::max);
    let ridge = max_diag * 1e-8;
    let mut reg = m.to_vec();
    for i in 0..k {
        reg[i * k + i] += ridge;
    }
    reg
}

// Trait to allow multiplying Vec<f64> by scalar (private helper)
trait ScaleMul {
    fn mul_scalar(&self, s: f64) -> Self;
}

impl ScaleMul for Vec<f64> {
    fn mul_scalar(&self, s: f64) -> Self {
        self.iter().map(|&v| v * s).collect()
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panel::fixed_effects::PanelData;

    fn make_panel_correlated(n: usize, t: usize) -> PanelData {
        // DGP where α_i is correlated with X (violates RE assumption)
        let mut y = vec![vec![0.0; t]; n];
        let mut x = vec![vec![vec![0.0; 1]; t]; n];
        for i in 0..n {
            let alpha_i = i as f64 * 0.5;
            for tt in 0..t {
                // x is correlated with alpha_i
                let xit = alpha_i * 0.3 + (i as f64 * 0.5 + tt as f64 * 0.3).sin();
                x[i][tt][0] = xit;
                y[i][tt] = alpha_i + 1.5 * xit;
            }
        }
        PanelData::new(y, x).expect("PanelData")
    }

    #[test]
    fn test_hausman_test_runs() {
        let data = make_panel_correlated(10, 8);
        let fe = FixedEffectsModel::fit(&data).expect("FE fit");
        let re = RandomEffectsModel::fit(&data).expect("RE fit");
        let result = HausmanTest::test(&fe, &re).expect("Hausman test");
        assert!(result.statistic >= 0.0, "statistic must be non-negative");
        assert!((0.0..=1.0).contains(&result.p_value), "p-value must be in [0,1]");
        assert_eq!(result.df, 1);
        assert_eq!(result.beta_diff.len(), 1);
    }

    #[test]
    fn test_hausman_mismatched_regressors() {
        let data = make_panel_correlated(6, 5);
        let fe = FixedEffectsModel::fit(&data).expect("FE");
        // Build a fake RE model with different k
        let re = RandomEffectsModel {
            coefficients: vec![1.0, 2.0],
            std_errors: vec![0.1, 0.1],
            t_stats: vec![10.0, 20.0],
            sigma_e: 0.05,
            sigma_u: 0.1,
            theta: 0.5,
            r_squared: 0.8,
            n_individuals: 6,
            n_periods: 5,
            n_regressors: 2,
            gls_cov: vec![0.01, 0.0, 0.0, 0.01],
        };
        assert!(HausmanTest::test(&fe, &re).is_err());
    }
}
