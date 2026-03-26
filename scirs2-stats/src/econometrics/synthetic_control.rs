//! Synthetic Control Method
//!
//! Implementation of the Abadie, Diamond, and Hainmueller (2010) synthetic
//! control method for comparative case studies.
//!
//! The method finds a weighted combination of control (donor) units that
//! best approximates the treated unit's pre-treatment characteristics.
//! Treatment effects are estimated as the gap between the treated unit
//! and its synthetic counterpart in the post-treatment period.
//!
//! # Key Features
//!
//! - **Simplex-constrained optimization**: Weights are non-negative and sum to 1
//! - **Predictor importance matrix V**: Diagonal weighting for predictor variables
//! - **Placebo tests**: Permutation inference by applying the method to each control
//! - **RMSPE ratio**: Post/pre RMSPE for significance assessment
//!
//! # References
//!
//! - Abadie, A., Diamond, A. & Hainmueller, J. (2010). Synthetic Control
//!   Methods for Comparative Case Studies. JASA, 105(490), 493-505.
//! - Abadie, A., Diamond, A. & Hainmueller, J. (2015). Comparative Politics
//!   and the Synthetic Control Method. AJPS, 59(2), 495-510.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a synthetic control estimation.
#[derive(Debug, Clone)]
pub struct SyntheticControlResult {
    /// Optimal weights for each donor unit (sum to 1, all >= 0).
    pub weights: Array1<f64>,

    /// Gap series: treated - synthetic for each post-treatment period.
    pub gap_series: Array1<f64>,

    /// Pre-treatment RMSPE (root mean squared prediction error).
    pub pre_rmspe: f64,

    /// Post-treatment RMSPE.
    pub post_rmspe: f64,

    /// RMSPE ratio (post/pre). Large values indicate a treatment effect.
    pub rmspe_ratio: f64,

    /// Synthetic control values for all periods (pre + post).
    pub synthetic_series: Array1<f64>,

    /// Placebo p-value (if placebo tests were run).
    /// Fraction of placebo units with RMSPE ratio >= treated unit's ratio.
    pub placebo_p_value: Option<f64>,

    /// Placebo results for each donor unit (if computed).
    pub placebo_results: Option<Vec<PlaceboResult>>,

    /// Diagonal elements of the V matrix (predictor importance weights).
    pub v_weights: Option<Array1<f64>>,
}

/// Result of a placebo test for a single donor unit.
#[derive(Debug, Clone)]
pub struct PlaceboResult {
    /// Index of the donor unit used as placebo-treated.
    pub donor_index: usize,

    /// Pre-treatment RMSPE for this placebo.
    pub pre_rmspe: f64,

    /// Post-treatment RMSPE for this placebo.
    pub post_rmspe: f64,

    /// RMSPE ratio for this placebo.
    pub rmspe_ratio: f64,

    /// Gap series for this placebo.
    pub gap_series: Array1<f64>,
}

// ---------------------------------------------------------------------------
// Synthetic Control Estimator
// ---------------------------------------------------------------------------

/// Synthetic Control Method estimator.
///
/// Finds optimal weights W for donor units such that:
///   synthetic = sum_j w_j * donor_j  ~=  treated  (pre-treatment)
///
/// The weights minimize:
///   ||X1 - X0 * W||^2_V
///
/// subject to w_j >= 0, sum w_j = 1, where V is a diagonal importance matrix.
pub struct SyntheticControlEstimator {
    /// Maximum iterations for weight optimization.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Whether to run placebo tests.
    pub run_placebo: bool,
    /// Optional diagonal V matrix weights (predictor importance).
    /// If None, uses identity (equal weights).
    pub v_weights: Option<Vec<f64>>,
}

impl SyntheticControlEstimator {
    /// Create a new estimator with default settings.
    pub fn new() -> Self {
        Self {
            max_iter: 5000,
            tol: 1e-8,
            run_placebo: false,
            v_weights: None,
        }
    }

    /// Create an estimator that also runs placebo tests.
    pub fn with_placebo() -> Self {
        Self {
            max_iter: 5000,
            tol: 1e-8,
            run_placebo: true,
            v_weights: None,
        }
    }

    /// Set the predictor importance weights (diagonal V matrix).
    pub fn set_v_weights(mut self, v: Vec<f64>) -> Self {
        self.v_weights = Some(v);
        self
    }

    /// Set optimization parameters.
    pub fn set_optimization(mut self, max_iter: usize, tol: f64) -> Self {
        self.max_iter = max_iter;
        self.tol = tol;
        self
    }

    /// Fit the synthetic control and estimate treatment effects.
    ///
    /// # Arguments
    /// * `y_treated_pre`  - pre-treatment outcomes for treated unit (T_pre,)
    /// * `y_treated_post` - post-treatment outcomes for treated unit (T_post,)
    /// * `y_donors_pre`   - pre-treatment outcomes for donors (T_pre x n_donors)
    /// * `y_donors_post`  - post-treatment outcomes for donors (T_post x n_donors)
    ///
    /// # Returns
    /// [`SyntheticControlResult`] with weights, gaps, and optionally placebo p-values.
    pub fn fit(
        &self,
        y_treated_pre: &ArrayView1<f64>,
        y_treated_post: &ArrayView1<f64>,
        y_donors_pre: &ArrayView2<f64>,
        y_donors_post: &ArrayView2<f64>,
    ) -> StatsResult<SyntheticControlResult> {
        let t_pre = y_treated_pre.len();
        let t_post = y_treated_post.len();
        let n_donors = y_donors_pre.ncols();

        // Validate
        if t_pre == 0 {
            return Err(StatsError::InsufficientData(
                "Need at least one pre-treatment period".into(),
            ));
        }
        if t_post == 0 {
            return Err(StatsError::InsufficientData(
                "Need at least one post-treatment period".into(),
            ));
        }
        if n_donors == 0 {
            return Err(StatsError::InvalidArgument(
                "Need at least one donor unit".into(),
            ));
        }
        if y_donors_pre.nrows() != t_pre {
            return Err(StatsError::DimensionMismatch(
                "y_donors_pre rows must equal y_treated_pre length".into(),
            ));
        }
        if y_donors_post.nrows() != t_post {
            return Err(StatsError::DimensionMismatch(
                "y_donors_post rows must equal y_treated_post length".into(),
            ));
        }
        if y_donors_post.ncols() != n_donors {
            return Err(StatsError::DimensionMismatch(
                "y_donors_post must have same number of donors as y_donors_pre".into(),
            ));
        }

        // Build V matrix
        let v_diag = self.get_v_diag(t_pre)?;

        // Fit weights on pre-treatment data
        let weights = self.optimize_weights(y_treated_pre, y_donors_pre, &v_diag)?;

        // Compute synthetic series for all periods
        let synthetic_pre = y_donors_pre.dot(&weights);
        let synthetic_post = y_donors_post.dot(&weights);

        // Gap series (post-treatment)
        let gap_series = y_treated_post.to_owned() - &synthetic_post;

        // Pre-treatment RMSPE
        let pre_gap = y_treated_pre.to_owned() - &synthetic_pre;
        let pre_rmspe = rmspe(&pre_gap);

        // Post-treatment RMSPE
        let post_rmspe = rmspe(&gap_series);

        let rmspe_ratio = if pre_rmspe > 1e-15 {
            post_rmspe / pre_rmspe
        } else {
            f64::INFINITY
        };

        // Full synthetic series
        let mut synthetic_series = Array1::<f64>::zeros(t_pre + t_post);
        for t in 0..t_pre {
            synthetic_series[t] = synthetic_pre[t];
        }
        for t in 0..t_post {
            synthetic_series[t_pre + t] = synthetic_post[t];
        }

        // Placebo tests
        let (placebo_p, placebo_results) = if self.run_placebo {
            let (p, results) = self.run_placebo_tests(
                y_treated_pre,
                y_treated_post,
                y_donors_pre,
                y_donors_post,
                rmspe_ratio,
                &v_diag,
            )?;
            (Some(p), Some(results))
        } else {
            (None, None)
        };

        Ok(SyntheticControlResult {
            weights,
            gap_series,
            pre_rmspe,
            post_rmspe,
            rmspe_ratio,
            synthetic_series,
            placebo_p_value: placebo_p,
            placebo_results,
            v_weights: Some(v_diag),
        })
    }

    /// Fit weights using predictor variables (covariates) in addition to outcomes.
    ///
    /// # Arguments
    /// * `predictors_treated` - predictor values for treated unit (n_predictors,)
    /// * `predictors_donors`  - predictor values for donors (n_predictors x n_donors)
    /// * `y_treated_pre`      - pre-treatment outcomes for treated unit
    /// * `y_donors_pre`       - pre-treatment outcomes for donors
    /// * `y_treated_post`     - post-treatment outcomes for treated unit
    /// * `y_donors_post`      - post-treatment outcomes for donors
    pub fn fit_with_predictors(
        &self,
        predictors_treated: &ArrayView1<f64>,
        predictors_donors: &ArrayView2<f64>,
        y_treated_pre: &ArrayView1<f64>,
        y_donors_pre: &ArrayView2<f64>,
        y_treated_post: &ArrayView1<f64>,
        y_donors_post: &ArrayView2<f64>,
    ) -> StatsResult<SyntheticControlResult> {
        let n_pred = predictors_treated.len();
        let t_pre = y_treated_pre.len();
        let n_donors = y_donors_pre.ncols();

        if predictors_donors.nrows() != n_pred {
            return Err(StatsError::DimensionMismatch(
                "predictors_donors rows must match predictors_treated length".into(),
            ));
        }
        if predictors_donors.ncols() != n_donors {
            return Err(StatsError::DimensionMismatch(
                "predictors_donors columns must match number of donors".into(),
            ));
        }

        // Stack predictors and outcomes for fitting
        let total_rows = n_pred + t_pre;
        let mut x_treated = Array1::<f64>::zeros(total_rows);
        let mut x_donors = Array2::<f64>::zeros((total_rows, n_donors));

        for p in 0..n_pred {
            x_treated[p] = predictors_treated[p];
            for j in 0..n_donors {
                x_donors[[p, j]] = predictors_donors[[p, j]];
            }
        }
        for t in 0..t_pre {
            x_treated[n_pred + t] = y_treated_pre[t];
            for j in 0..n_donors {
                x_donors[[n_pred + t, j]] = y_donors_pre[[t, j]];
            }
        }

        let v_diag = self.get_v_diag(total_rows)?;
        let weights = self.optimize_weights(&x_treated.view(), &x_donors.view(), &v_diag)?;

        // Compute results using outcome data
        let synthetic_pre = y_donors_pre.dot(&weights);
        let synthetic_post = y_donors_post.dot(&weights);
        let gap_series = y_treated_post.to_owned() - &synthetic_post;

        let pre_gap = y_treated_pre.to_owned() - &synthetic_pre;
        let pre_rmspe = rmspe(&pre_gap);
        let post_rmspe = rmspe(&gap_series);
        let rmspe_ratio = if pre_rmspe > 1e-15 {
            post_rmspe / pre_rmspe
        } else {
            f64::INFINITY
        };

        let t_post = y_treated_post.len();
        let mut synthetic_series = Array1::<f64>::zeros(t_pre + t_post);
        for t in 0..t_pre {
            synthetic_series[t] = synthetic_pre[t];
        }
        for t in 0..t_post {
            synthetic_series[t_pre + t] = synthetic_post[t];
        }

        Ok(SyntheticControlResult {
            weights,
            gap_series,
            pre_rmspe,
            post_rmspe,
            rmspe_ratio,
            synthetic_series,
            placebo_p_value: None,
            placebo_results: None,
            v_weights: Some(v_diag),
        })
    }

    /// Get the V diagonal vector (predictor importance weights).
    fn get_v_diag(&self, n_rows: usize) -> StatsResult<Array1<f64>> {
        match &self.v_weights {
            Some(v) => {
                if v.len() != n_rows {
                    return Err(StatsError::DimensionMismatch(format!(
                        "V weights length {} != number of predictors/periods {n_rows}",
                        v.len()
                    )));
                }
                Ok(Array1::from_vec(v.clone()))
            }
            None => Ok(Array1::ones(n_rows)),
        }
    }

    /// Optimize weights using projected gradient descent with V-weighted objective.
    ///
    /// Minimize ||V^{1/2} (x1 - X0 w)||^2  s.t.  w >= 0, sum(w) = 1
    fn optimize_weights(
        &self,
        x_treated: &ArrayView1<f64>,
        x_donors: &ArrayView2<f64>,
        v_diag: &Array1<f64>,
    ) -> StatsResult<Array1<f64>> {
        let n_donors = x_donors.ncols();
        let n_rows = x_donors.nrows();

        if n_donors == 0 {
            return Err(StatsError::InvalidArgument(
                "Need at least one donor".into(),
            ));
        }

        // V-weight the data: scale rows by sqrt(v_diag)
        let v_sqrt: Array1<f64> = v_diag.mapv(|v| v.max(0.0).sqrt());
        let mut x0_v = Array2::<f64>::zeros((n_rows, n_donors));
        let mut x1_v = Array1::<f64>::zeros(n_rows);
        for t in 0..n_rows {
            x1_v[t] = x_treated[t] * v_sqrt[t];
            for j in 0..n_donors {
                x0_v[[t, j]] = x_donors[[t, j]] * v_sqrt[t];
            }
        }

        // Pre-compute X0'X0 and X0'x1 for gradient
        let x0t_x0 = x0_v.t().dot(&x0_v);
        let x0t_x1 = x0_v.t().dot(&x1_v);

        // Step size: 1 / (max row sum of |X0'X0|)
        let max_row_sum: f64 = x0t_x0
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v.abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let lr = if max_row_sum > 0.0 {
            0.5 / max_row_sum
        } else {
            1e-3
        };

        // Initialize uniform weights
        let mut w = Array1::from_elem(n_donors, 1.0 / n_donors as f64);

        for _ in 0..self.max_iter {
            // Gradient: 2 * (X0'X0 w - X0'x1)
            let grad = x0t_x0.dot(&w) - &x0t_x1;
            let w_new_raw = &w - &grad.mapv(|g| g * lr);
            let w_new = project_simplex(&w_new_raw.view());

            let diff: f64 = (&w_new - &w).iter().map(|&d| d * d).sum::<f64>().sqrt();
            w = w_new;
            if diff < self.tol {
                break;
            }
        }

        Ok(w)
    }

    /// Run placebo tests: apply synthetic control to each donor as if it were treated.
    fn run_placebo_tests(
        &self,
        y_treated_pre: &ArrayView1<f64>,
        y_treated_post: &ArrayView1<f64>,
        y_donors_pre: &ArrayView2<f64>,
        y_donors_post: &ArrayView2<f64>,
        treated_rmspe_ratio: f64,
        v_diag: &Array1<f64>,
    ) -> StatsResult<(f64, Vec<PlaceboResult>)> {
        let t_pre = y_treated_pre.len();
        let t_post = y_treated_post.len();
        let n_donors = y_donors_pre.ncols();

        let mut placebo_results = Vec::with_capacity(n_donors);
        let mut n_extreme = 0_usize;

        for d in 0..n_donors {
            // Treat donor d as the "treated" unit
            let placebo_treated_pre = y_donors_pre.column(d).to_owned();
            let placebo_treated_post = y_donors_post.column(d).to_owned();

            // Build donor pool: remaining donors + the actual treated unit
            let n_placebo_donors = n_donors; // -1 for excluded + 1 for treated
            let mut placebo_donors_pre = Array2::<f64>::zeros((t_pre, n_placebo_donors));
            let mut placebo_donors_post = Array2::<f64>::zeros((t_post, n_placebo_donors));

            let mut col = 0;
            for j in 0..n_donors {
                if j == d {
                    // Replace with actual treated unit
                    for t in 0..t_pre {
                        placebo_donors_pre[[t, col]] = y_treated_pre[t];
                    }
                    for t in 0..t_post {
                        placebo_donors_post[[t, col]] = y_treated_post[t];
                    }
                } else {
                    for t in 0..t_pre {
                        placebo_donors_pre[[t, col]] = y_donors_pre[[t, j]];
                    }
                    for t in 0..t_post {
                        placebo_donors_post[[t, col]] = y_donors_post[[t, j]];
                    }
                }
                col += 1;
            }

            // Fit placebo weights
            let placebo_weights = match self.optimize_weights(
                &placebo_treated_pre.view(),
                &placebo_donors_pre.view(),
                v_diag,
            ) {
                Ok(w) => w,
                Err(_) => continue, // Skip if optimization fails
            };

            let synth_pre = placebo_donors_pre.dot(&placebo_weights);
            let synth_post = placebo_donors_post.dot(&placebo_weights);

            let pre_gap = placebo_treated_pre - &synth_pre;
            let post_gap = placebo_treated_post - &synth_post;

            let pre_rmspe_p = rmspe(&pre_gap);
            let post_rmspe_p = rmspe(&post_gap);
            let ratio = if pre_rmspe_p > 1e-15 {
                post_rmspe_p / pre_rmspe_p
            } else {
                f64::INFINITY
            };

            if ratio >= treated_rmspe_ratio {
                n_extreme += 1;
            }

            placebo_results.push(PlaceboResult {
                donor_index: d,
                pre_rmspe: pre_rmspe_p,
                post_rmspe: post_rmspe_p,
                rmspe_ratio: ratio,
                gap_series: post_gap,
            });
        }

        // p-value: (number of placebos with ratio >= treated ratio + 1) / (n_donors + 1)
        let p_value = (n_extreme as f64 + 1.0) / (n_donors as f64 + 1.0);

        Ok((p_value, placebo_results))
    }
}

impl Default for SyntheticControlEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Root mean squared prediction error.
fn rmspe(gaps: &Array1<f64>) -> f64 {
    if gaps.is_empty() {
        return 0.0;
    }
    let mse: f64 = gaps.iter().map(|&g| g * g).sum::<f64>() / gaps.len() as f64;
    mse.sqrt()
}

/// Project a vector onto the probability simplex: w >= 0, sum(w) = 1.
///
/// Uses the algorithm of Duchi et al. (2008).
fn project_simplex(v: &ArrayView1<f64>) -> Array1<f64> {
    let n = v.len();
    let mut u: Vec<f64> = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut rho = 0_usize;
    let mut cum = 0.0_f64;
    for (j, &uj) in u.iter().enumerate() {
        cum += uj;
        if uj - (cum - 1.0) / (j as f64 + 1.0) > 0.0 {
            rho = j;
        }
    }
    let cum_rho: f64 = u[..=rho].iter().sum();
    let theta = (cum_rho - 1.0) / (rho as f64 + 1.0);
    v.mapv(|vi| (vi - theta).max(0.0))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_synthetic_control_perfect_fit() {
        // Donor 0 perfectly matches treated; others are very different (non-linear)
        let t_pre = 20;
        let t_post = 10;
        let n_donors = 3;

        let y_treated_pre: Array1<f64> = (0..t_pre).map(|t| t as f64).collect();
        let y_treated_post: Array1<f64> = (t_pre..t_pre + t_post).map(|t| t as f64 + 5.0).collect();

        let mut y_donors_pre = Array2::<f64>::zeros((t_pre, n_donors));
        let mut y_donors_post = Array2::<f64>::zeros((t_post, n_donors));
        for t in 0..t_pre {
            y_donors_pre[[t, 0]] = t as f64; // perfect match
            y_donors_pre[[t, 1]] = 100.0 + t as f64 * 3.0; // very different level
            y_donors_pre[[t, 2]] = -50.0 + (t as f64).powi(2); // quadratic, very different
        }
        for t in 0..t_post {
            let tt = (t_pre + t) as f64;
            y_donors_post[[t, 0]] = tt;
            y_donors_post[[t, 1]] = 100.0 + tt * 3.0;
            y_donors_post[[t, 2]] = -50.0 + tt.powi(2);
        }

        let sc = SyntheticControlEstimator::new().set_optimization(10000, 1e-10);
        let res = sc
            .fit(
                &y_treated_pre.view(),
                &y_treated_post.view(),
                &y_donors_pre.view(),
                &y_donors_post.view(),
            )
            .expect("SC should succeed");

        // Weights should sum to 1
        let wsum: f64 = res.weights.iter().sum();
        assert!(
            (wsum - 1.0).abs() < 1e-6,
            "Weights should sum to 1, got {}",
            wsum
        );
        // All non-negative
        assert!(res.weights.iter().all(|&w| w >= -1e-10));

        // Pre-treatment RMSPE should be small (good fit achievable)
        assert!(
            res.pre_rmspe < 2.0,
            "Pre-RMSPE should be small, got {}",
            res.pre_rmspe
        );

        // Donor 0 should get the largest weight
        let max_w = res
            .weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (res.weights[0] - max_w).abs() < 1e-6,
            "Donor 0 should get largest weight (w0={}, max={})",
            res.weights[0],
            max_w
        );
    }

    #[test]
    fn test_synthetic_control_treatment_effect() {
        // Two identical donors pre-treatment; treated unit jumps by 10 post-treatment
        let t_pre = 15;
        let t_post = 10;
        let te = 10.0_f64;

        let y_treated_pre: Array1<f64> = (0..t_pre).map(|t| t as f64).collect();
        let y_treated_post: Array1<f64> = (t_pre..t_pre + t_post).map(|t| t as f64 + te).collect();

        let mut y_donors_pre = Array2::<f64>::zeros((t_pre, 2));
        let mut y_donors_post = Array2::<f64>::zeros((t_post, 2));
        for t in 0..t_pre {
            y_donors_pre[[t, 0]] = t as f64;
            y_donors_pre[[t, 1]] = t as f64 + 0.1;
        }
        for t in 0..t_post {
            let tt = (t_pre + t) as f64;
            y_donors_post[[t, 0]] = tt;
            y_donors_post[[t, 1]] = tt + 0.1;
        }

        let sc = SyntheticControlEstimator::new();
        let res = sc
            .fit(
                &y_treated_pre.view(),
                &y_treated_post.view(),
                &y_donors_pre.view(),
                &y_donors_post.view(),
            )
            .expect("SC should succeed");

        // Gap should be approximately te
        let avg_gap: f64 = res.gap_series.iter().sum::<f64>() / res.gap_series.len() as f64;
        assert!(
            (avg_gap - te).abs() < 2.0,
            "Average gap should be ~{te}, got {}",
            avg_gap
        );
    }

    #[test]
    fn test_synthetic_control_placebo() {
        let t_pre = 10;
        let t_post = 5;
        let n_donors = 3;

        // All units follow the same trend; no actual treatment effect
        let y_treated_pre: Array1<f64> = (0..t_pre).map(|t| t as f64).collect();
        let y_treated_post: Array1<f64> = (t_pre..t_pre + t_post).map(|t| t as f64).collect();

        let mut y_donors_pre = Array2::<f64>::zeros((t_pre, n_donors));
        let mut y_donors_post = Array2::<f64>::zeros((t_post, n_donors));
        for t in 0..t_pre {
            for j in 0..n_donors {
                y_donors_pre[[t, j]] = t as f64 + (j as f64) * 0.01;
            }
        }
        for t in 0..t_post {
            let tt = (t_pre + t) as f64;
            for j in 0..n_donors {
                y_donors_post[[t, j]] = tt + (j as f64) * 0.01;
            }
        }

        let sc = SyntheticControlEstimator::with_placebo();
        let res = sc
            .fit(
                &y_treated_pre.view(),
                &y_treated_post.view(),
                &y_donors_pre.view(),
                &y_donors_post.view(),
            )
            .expect("SC with placebo should succeed");

        // With no treatment effect, p-value should not be very small
        assert!(
            res.placebo_p_value.is_some(),
            "Placebo p-value should be computed"
        );
        let p = res.placebo_p_value.expect("checked above");
        // p should be > 0 (with small donor pool, p = (k+1)/(n+1) for some k)
        assert!(p > 0.0 && p <= 1.0, "p-value should be in (0, 1], got {p}");
    }

    #[test]
    fn test_synthetic_control_with_predictors() {
        let t_pre = 10;
        let t_post = 5;
        let n_donors = 3;
        let n_pred = 2;

        // Predictors
        let pred_treated = Array1::from_vec(vec![10.0, 20.0]);
        let mut pred_donors = Array2::<f64>::zeros((n_pred, n_donors));
        pred_donors[[0, 0]] = 10.0;
        pred_donors[[1, 0]] = 20.0; // matches
        pred_donors[[0, 1]] = 5.0;
        pred_donors[[1, 1]] = 25.0;
        pred_donors[[0, 2]] = 15.0;
        pred_donors[[1, 2]] = 15.0;

        let y_treated_pre: Array1<f64> = (0..t_pre).map(|t| t as f64).collect();
        let y_treated_post: Array1<f64> = (t_pre..t_pre + t_post).map(|t| t as f64 + 3.0).collect();

        let mut y_donors_pre = Array2::<f64>::zeros((t_pre, n_donors));
        let mut y_donors_post = Array2::<f64>::zeros((t_post, n_donors));
        for t in 0..t_pre {
            y_donors_pre[[t, 0]] = t as f64;
            y_donors_pre[[t, 1]] = t as f64 * 0.5;
            y_donors_pre[[t, 2]] = t as f64 * 1.5;
        }
        for t in 0..t_post {
            let tt = (t_pre + t) as f64;
            y_donors_post[[t, 0]] = tt;
            y_donors_post[[t, 1]] = tt * 0.5;
            y_donors_post[[t, 2]] = tt * 1.5;
        }

        let sc = SyntheticControlEstimator::new();
        let res = sc
            .fit_with_predictors(
                &pred_treated.view(),
                &pred_donors.view(),
                &y_treated_pre.view(),
                &y_donors_pre.view(),
                &y_treated_post.view(),
                &y_donors_post.view(),
            )
            .expect("SC with predictors should succeed");

        let wsum: f64 = res.weights.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-6);
        assert!(res.weights.iter().all(|&w| w >= -1e-10));
    }

    #[test]
    fn test_synthetic_control_v_weights() {
        let t_pre = 10;
        let t_post = 5;

        let y_treated_pre: Array1<f64> = (0..t_pre).map(|t| t as f64).collect();
        let y_treated_post: Array1<f64> = (t_pre..t_pre + t_post).map(|t| t as f64).collect();

        let mut y_donors_pre = Array2::<f64>::zeros((t_pre, 2));
        let mut y_donors_post = Array2::<f64>::zeros((t_post, 2));
        for t in 0..t_pre {
            y_donors_pre[[t, 0]] = t as f64 + 0.1;
            y_donors_pre[[t, 1]] = t as f64 * 2.0;
        }
        for t in 0..t_post {
            let tt = (t_pre + t) as f64;
            y_donors_post[[t, 0]] = tt + 0.1;
            y_donors_post[[t, 1]] = tt * 2.0;
        }

        // Weight later periods more heavily
        let v = (0..t_pre)
            .map(|t| (t as f64 + 1.0) / t_pre as f64)
            .collect();
        let sc = SyntheticControlEstimator::new().set_v_weights(v);
        let res = sc
            .fit(
                &y_treated_pre.view(),
                &y_treated_post.view(),
                &y_donors_pre.view(),
                &y_donors_post.view(),
            )
            .expect("SC with V weights should succeed");

        let wsum: f64 = res.weights.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-6);
        assert!(res.v_weights.is_some());
    }
}
