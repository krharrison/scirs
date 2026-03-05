//! Fixed-Effects (Within) Panel Estimator
//!
//! The fixed-effects estimator controls for all time-invariant unobserved
//! individual heterogeneity by demeaning ("within-transformation") each
//! individual's observations before running OLS.
//!
//! # Within Transformation
//!
//! For individual `i` with `T_i` observations:
//! ```text
//! ỹ_{it} = y_{it} − ȳ_i
//! X̃_{it} = X_{it} − X̄_i
//! ```
//! Then OLS on `ỹ = X̃ β + ε̃` yields the within estimator β̂_FE.
//!
//! Fixed effects `α̂_i = ȳ_i − X̄_i β̂_FE`.
//!
//! # Inference
//!
//! Variance: `V(β̂_FE) = σ² (X̃'X̃)⁻¹` where
//! `σ² = RSS / (N*T - N - K)` (degrees of freedom: subtract fixed effects).

use crate::error::{Result, TimeSeriesError};

// ============================================================
// Data input structure
// ============================================================

/// Balanced panel data container.
///
/// Organises observations as `N` individuals × `T` time periods × `K` regressors.
/// All individuals must have the same number of time periods (balanced panel).
#[derive(Debug, Clone)]
pub struct PanelData {
    /// `y[i][t]` — dependent variable for individual i at time t
    pub y: Vec<Vec<f64>>,
    /// `x[i][t][k]` — k-th regressor for individual i at time t
    pub x: Vec<Vec<Vec<f64>>>,
    /// Number of individuals N
    pub n_individuals: usize,
    /// Number of time periods T (balanced)
    pub n_periods: usize,
    /// Number of regressors K (excluding fixed effects intercepts)
    pub n_regressors: usize,
}

impl PanelData {
    /// Construct a balanced panel dataset and validate dimensions.
    pub fn new(y: Vec<Vec<f64>>, x: Vec<Vec<Vec<f64>>>) -> Result<Self> {
        let n = y.len();
        if n == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "PanelData: no individuals (empty y)".into(),
            ));
        }
        if x.len() != n {
            return Err(TimeSeriesError::InvalidInput(format!(
                "PanelData: x has {} individuals but y has {}",
                x.len(),
                n
            )));
        }

        let t = y[0].len();
        if t == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "PanelData: zero time periods".into(),
            ));
        }

        for (i, (yi, xi)) in y.iter().zip(x.iter()).enumerate() {
            if yi.len() != t {
                return Err(TimeSeriesError::InvalidInput(format!(
                    "PanelData: individual {} has {} periods, expected {}",
                    i,
                    yi.len(),
                    t
                )));
            }
            if xi.len() != t {
                return Err(TimeSeriesError::InvalidInput(format!(
                    "PanelData: individual {} has x with {} periods, expected {}",
                    i,
                    xi.len(),
                    t
                )));
            }
        }

        let k = if x[0].is_empty() || x[0][0].is_empty() {
            0
        } else {
            x[0][0].len()
        };

        for (i, xi) in x.iter().enumerate() {
            for (t_idx, xt) in xi.iter().enumerate() {
                if xt.len() != k {
                    return Err(TimeSeriesError::InvalidInput(format!(
                        "PanelData: individual {i}, period {t_idx}: {} regressors, expected {k}",
                        xt.len()
                    )));
                }
            }
        }

        Ok(Self {
            n_individuals: n,
            n_periods: t,
            n_regressors: k,
            y,
            x,
        })
    }
}

// ============================================================
// Fixed-Effects model
// ============================================================

/// Fixed-Effects (Within) estimator for balanced panel data.
#[derive(Debug, Clone)]
pub struct FixedEffectsModel {
    /// OLS slope coefficients β̂_FE (length K)
    pub coefficients: Vec<f64>,
    /// Standard errors of β̂_FE
    pub std_errors: Vec<f64>,
    /// t-statistics for H₀: β_k = 0
    pub t_stats: Vec<f64>,
    /// Individual fixed effects α̂_i (length N)
    pub fixed_effects: Vec<f64>,
    /// Residual variance σ̂²
    pub sigma2: f64,
    /// R² (within-group)
    pub r_squared: f64,
    /// Number of individuals N
    pub n_individuals: usize,
    /// Number of time periods T
    pub n_periods: usize,
    /// Number of regressors K
    pub n_regressors: usize,
    /// Score matrix (N*T × K) of demeaned regressors — used in Hausman test
    pub(crate) xtx_inv: Vec<f64>,
}

impl FixedEffectsModel {
    /// Fit a fixed-effects (within) estimator to balanced panel data.
    pub fn fit(data: &PanelData) -> Result<Self> {
        let n = data.n_individuals;
        let t = data.n_periods;
        let k = data.n_regressors;

        if n * t < k + n + 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "FE: too few observations for model identification".into(),
                required: k + n + 2,
                actual: n * t,
            });
        }

        // --- Within transformation: demean by individual ---
        let mut y_mean = vec![0.0_f64; n];
        let mut x_mean = vec![vec![0.0_f64; k]; n];

        for i in 0..n {
            y_mean[i] = data.y[i].iter().sum::<f64>() / t as f64;
            for j in 0..k {
                x_mean[i][j] =
                    data.x[i].iter().map(|xt| xt[j]).sum::<f64>() / t as f64;
            }
        }

        // Build demeaned y and X
        let n_obs = n * t;
        let mut y_tilde = Vec::with_capacity(n_obs);
        let mut x_tilde = Vec::with_capacity(n_obs * k);

        for i in 0..n {
            for tt in 0..t {
                y_tilde.push(data.y[i][tt] - y_mean[i]);
                for j in 0..k {
                    x_tilde.push(data.x[i][tt][j] - x_mean[i][j]);
                }
            }
        }

        // OLS: β = (X̃'X̃)⁻¹ X̃'ỹ
        let mut xtx = vec![0.0_f64; k * k];
        let mut xty = vec![0.0_f64; k];

        for obs in 0..n_obs {
            for a in 0..k {
                xty[a] += x_tilde[obs * k + a] * y_tilde[obs];
                for b in 0..k {
                    xtx[a * k + b] += x_tilde[obs * k + a] * x_tilde[obs * k + b];
                }
            }
        }

        let xtx_inv = invert_sym(k, &xtx)?;
        let beta = mat_vec_mul(k, &xtx_inv, &xty);

        // Fixed effects: α̂_i = ȳ_i − X̄_i β
        let mut fixed_effects = Vec::with_capacity(n);
        for i in 0..n {
            let xb: f64 = (0..k).map(|j| x_mean[i][j] * beta[j]).sum();
            fixed_effects.push(y_mean[i] - xb);
        }

        // Residuals and variance
        // df = N*T - N - K (subtract N fixed effects + K slopes)
        let df = n_obs as f64 - n as f64 - k as f64;
        let df = df.max(1.0);

        let mut ss_res = 0.0_f64;
        let mut ss_tot = 0.0_f64;
        let grand_mean_y: f64 = y_tilde.iter().sum::<f64>(); // 0 by construction

        for obs in 0..n_obs {
            let xb: f64 = (0..k).map(|j| x_tilde[obs * k + j] * beta[j]).sum();
            let resid = y_tilde[obs] - xb;
            ss_res += resid * resid;
            ss_tot += (y_tilde[obs] - grand_mean_y).powi(2);
        }

        let sigma2 = ss_res / df;
        let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };

        // Standard errors: sqrt(sigma2 * diag(XtX_inv))
        let std_errors: Vec<f64> = (0..k)
            .map(|j| (sigma2 * xtx_inv[j * k + j]).sqrt())
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
            fixed_effects,
            sigma2,
            r_squared,
            n_individuals: n,
            n_periods: t,
            n_regressors: k,
            xtx_inv,
        })
    }

    /// Predict fitted values for in-sample observations.
    pub fn predict(&self, data: &PanelData) -> Result<Vec<Vec<f64>>> {
        let n = data.n_individuals;
        let t = data.n_periods;
        let k = data.n_regressors;

        if k != self.n_regressors {
            return Err(TimeSeriesError::InvalidInput(format!(
                "predict: expected {} regressors, got {}",
                self.n_regressors, k
            )));
        }
        if n != self.n_individuals {
            return Err(TimeSeriesError::InvalidInput(format!(
                "predict: expected {} individuals, got {}",
                self.n_individuals, n
            )));
        }

        let mut fitted = vec![vec![0.0_f64; t]; n];
        for i in 0..n {
            let alpha_i = if i < self.fixed_effects.len() {
                self.fixed_effects[i]
            } else {
                0.0
            };
            for tt in 0..t {
                let xb: f64 = (0..k).map(|j| data.x[i][tt][j] * self.coefficients[j]).sum();
                fitted[i][tt] = alpha_i + xb;
            }
        }

        Ok(fitted)
    }
}

// ============================================================
// Linear algebra utilities (small matrices)
// ============================================================

/// Invert a symmetric k×k matrix via Cholesky decomposition.
pub(crate) fn invert_sym(k: usize, a: &[f64]) -> Result<Vec<f64>> {
    if k == 0 {
        return Ok(Vec::new());
    }

    // Cholesky: L such that A = L L'
    let mut l = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum: f64 = a[i * k + j];
            for p in 0..j {
                sum -= l[i * k + p] * l[j * k + p];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(TimeSeriesError::NumericalError(
                        "FE OLS: Cholesky failure — X̃'X̃ is not positive-definite (possible multicollinearity)".into(),
                    ));
                }
                l[i * k + i] = sum.sqrt();
            } else {
                l[i * k + j] = sum / l[j * k + j];
            }
        }
    }

    // Solve L * Z = I to get L⁻¹
    let mut l_inv = vec![0.0_f64; k * k];
    for col in 0..k {
        l_inv[col * k + col] = 1.0 / l[col * k + col];
        for row in (col + 1)..k {
            let mut sum = 0.0_f64;
            for p in col..row {
                sum += l[row * k + p] * l_inv[p * k + col];
            }
            l_inv[row * k + col] = -sum / l[row * k + row];
        }
    }

    // A⁻¹ = (L⁻¹)' L⁻¹
    let mut a_inv = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut val = 0.0_f64;
            for p in i.max(j)..k {
                val += l_inv[p * k + i] * l_inv[p * k + j];
            }
            a_inv[i * k + j] = val;
        }
    }

    Ok(a_inv)
}

/// Matrix-vector product for a k×k matrix and k-vector.
pub(crate) fn mat_vec_mul(k: usize, m: &[f64], v: &[f64]) -> Vec<f64> {
    (0..k)
        .map(|i| (0..k).map(|j| m[i * k + j] * v[j]).sum())
        .collect()
}

/// Matrix-matrix product for k×k matrices.
pub(crate) fn mat_mul(k: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            for p in 0..k {
                c[i * k + j] += a[i * k + p] * b[p * k + j];
            }
        }
    }
    c
}

/// Matrix subtraction.
pub(crate) fn mat_sub(k: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai - bi).collect()
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a balanced panel dataset with known DGP:
    ///   y_{it} = α_i + 2.0 * x_{it} + eps
    fn make_panel(n: usize, t: usize) -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
        let alpha: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let mut y = vec![vec![0.0; t]; n];
        let mut x = vec![vec![vec![0.0; 1]; t]; n];
        for i in 0..n {
            for tt in 0..t {
                let xit = (i as f64 + tt as f64 * 0.3).sin() + 1.0;
                x[i][tt][0] = xit;
                // eps = 0 (clean DGP)
                y[i][tt] = alpha[i] + 2.0 * xit;
            }
        }
        (y, x)
    }

    #[test]
    fn test_panel_data_new_valid() {
        let (y, x) = make_panel(4, 5);
        let data = PanelData::new(y, x).expect("Should create PanelData");
        assert_eq!(data.n_individuals, 4);
        assert_eq!(data.n_periods, 5);
        assert_eq!(data.n_regressors, 1);
    }

    #[test]
    fn test_panel_data_mismatched_lengths() {
        let y = vec![vec![1.0, 2.0], vec![3.0]]; // individual 1 has 1 period, not 2
        let x = vec![vec![vec![1.0]; 2], vec![vec![1.0]; 2]];
        assert!(PanelData::new(y, x).is_err());
    }

    #[test]
    fn test_fe_fit_recovers_true_beta() {
        let (y, x) = make_panel(10, 8);
        let data = PanelData::new(y, x).expect("PanelData");
        let model = FixedEffectsModel::fit(&data).expect("FE fit");
        // With exact DGP (no noise), β̂ should be very close to 2.0
        assert!(
            (model.coefficients[0] - 2.0).abs() < 1e-6,
            "β̂ = {:.6}, expected 2.0",
            model.coefficients[0]
        );
    }

    #[test]
    fn test_fe_fixed_effects_consistency() {
        let (y, x) = make_panel(5, 6);
        let data = PanelData::new(y, x).expect("PanelData");
        let model = FixedEffectsModel::fit(&data).expect("FE fit");
        assert_eq!(model.fixed_effects.len(), 5);
        // FEs should be close to true α_i = i * 0.5
        for (i, &fe) in model.fixed_effects.iter().enumerate() {
            let true_alpha = i as f64 * 0.5;
            assert!(
                (fe - true_alpha).abs() < 1e-5,
                "FE[{i}] = {fe:.4}, expected {true_alpha:.4}"
            );
        }
    }

    #[test]
    fn test_fe_predict_shape() {
        let (y, x) = make_panel(4, 5);
        let data = PanelData::new(y, x).expect("PanelData");
        let model = FixedEffectsModel::fit(&data).expect("FE fit");
        let fitted = model.predict(&data).expect("predict");
        assert_eq!(fitted.len(), 4);
        assert_eq!(fitted[0].len(), 5);
    }
}
