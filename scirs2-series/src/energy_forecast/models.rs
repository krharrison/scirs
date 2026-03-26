//! Forecasting models for energy load prediction
//!
//! Implements quantile regression (IRLS), gradient-boosted decision stumps,
//! and a combined energy forecaster.

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};

use super::features::{FeatureConfig, FeatureMatrix};
use super::types::{EnergyForecastConfig, EnergyForecastResult, LoadProfile, QuantileForecast};

// ---------------------------------------------------------------------------
// Quantile Regression via IRLS
// ---------------------------------------------------------------------------

/// Quantile regression using iteratively reweighted least squares (IRLS).
///
/// Minimises the pinball (check) loss for a given quantile τ by transforming
/// it into a sequence of weighted least-squares problems.
#[derive(Debug, Clone)]
pub struct QuantileRegressor {
    /// Fitted coefficients (including intercept as last element)
    coefficients: Vec<f64>,
    /// The quantile level
    quantile: f64,
}

impl QuantileRegressor {
    /// Fit a quantile regression model.
    ///
    /// Uses IRLS with pinball-loss-derived weights.
    /// `x` is (n, p), `y` is length n, `quantile` in (0, 1).
    pub fn fit(x: &Array2<f64>, y: &[f64], quantile: f64) -> Result<Self> {
        let n = x.nrows();
        let p = x.ncols();

        if n == 0 || y.len() != n {
            return Err(TimeSeriesError::InvalidInput(
                "x and y must have the same number of rows and be non-empty".to_string(),
            ));
        }
        if !(0.0 < quantile && quantile < 1.0) {
            return Err(TimeSeriesError::InvalidParameter {
                name: "quantile".to_string(),
                message: format!("must be in (0,1), got {}", quantile),
            });
        }

        // Augment x with intercept column
        let p_aug = p + 1;
        let mut x_aug = Array2::<f64>::ones((n, p_aug));
        x_aug.slice_mut(s![.., ..p]).assign(x);

        // Initialise coefficients to zero
        let mut beta = vec![0.0; p_aug];

        let max_iter = 50;
        let eps = 1e-8;

        for _ in 0..max_iter {
            // Compute residuals
            let mut residuals = Vec::with_capacity(n);
            for i in 0..n {
                let mut pred = 0.0;
                for j in 0..p_aug {
                    pred += x_aug[[i, j]] * beta[j];
                }
                residuals.push(y[i] - pred);
            }

            // Compute weights from pinball loss derivative
            let mut weights = Vec::with_capacity(n);
            for &r in &residuals {
                let abs_r = r.abs().max(eps);
                let w = if r >= 0.0 {
                    quantile / abs_r
                } else {
                    (1.0 - quantile) / abs_r
                };
                weights.push(w);
            }

            // Weighted least squares: (X^T W X) beta = X^T W y
            let mut xtw_x = vec![0.0; p_aug * p_aug];
            let mut xtw_y = vec![0.0; p_aug];

            for i in 0..n {
                let w = weights[i];
                for j in 0..p_aug {
                    xtw_y[j] += x_aug[[i, j]] * w * y[i];
                    for k in 0..p_aug {
                        xtw_x[j * p_aug + k] += x_aug[[i, j]] * w * x_aug[[i, k]];
                    }
                }
            }

            // Add small regularisation for numerical stability
            for j in 0..p_aug {
                xtw_x[j * p_aug + j] += 1e-10;
            }

            // Solve via Cholesky-like approach (simple Gaussian elimination)
            let new_beta = solve_linear_system(&xtw_x, &xtw_y, p_aug)?;

            // Check convergence
            let delta: f64 = new_beta
                .iter()
                .zip(beta.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            beta = new_beta;
            if delta < eps {
                break;
            }
        }

        Ok(Self {
            coefficients: beta,
            quantile,
        })
    }

    /// Predict using the fitted model.
    pub fn predict(&self, x: &Array2<f64>) -> Vec<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let p_aug = self.coefficients.len();
        let mut preds = Vec::with_capacity(n);

        for i in 0..n {
            let mut val = 0.0;
            for j in 0..p.min(p_aug - 1) {
                val += x[[i, j]] * self.coefficients[j];
            }
            // intercept
            if p_aug > 0 {
                val += self.coefficients[p_aug - 1];
            }
            preds.push(val);
        }
        preds
    }
}

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
fn solve_linear_system(a_flat: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>> {
    // Build augmented matrix
    let mut aug = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a_flat[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col * (n + 1) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[row * (n + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            // Nearly singular; return zeros for this column
            continue;
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[col * (n + 1) + j];
                aug[col * (n + 1) + j] = aug[max_row * (n + 1) + j];
                aug[max_row * (n + 1) + j] = tmp;
            }
        }

        // Eliminate below
        let pivot = aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let diag = aug[i * (n + 1) + i];
        if diag.abs() < 1e-15 {
            x[i] = 0.0;
            continue;
        }
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / diag;
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Gradient-Boosted Decision Stumps
// ---------------------------------------------------------------------------

/// A single decision stump: splits on one feature at a threshold.
#[derive(Debug, Clone)]
struct DecisionStump {
    /// Feature index to split on
    feature_idx: usize,
    /// Split threshold
    threshold: f64,
    /// Prediction for samples where feature <= threshold
    left_value: f64,
    /// Prediction for samples where feature > threshold
    right_value: f64,
}

/// Gradient-boosted decision stumps for regression.
///
/// An ensemble of single-split decision trees trained with gradient descent
/// on squared-error loss.
#[derive(Debug, Clone)]
pub struct GradientBoostedStumps {
    /// Sequence of fitted stumps
    stumps: Vec<DecisionStump>,
    /// Learning rate applied to each stump
    learning_rate: f64,
    /// Initial prediction (mean of training targets)
    initial_prediction: f64,
}

impl GradientBoostedStumps {
    /// Fit gradient-boosted stumps.
    ///
    /// `x` shape (n, p), `y` length n, `n_estimators` stumps, `learning_rate` shrinkage.
    pub fn fit(
        x: &Array2<f64>,
        y: &[f64],
        n_estimators: usize,
        learning_rate: f64,
    ) -> Result<Self> {
        let n = x.nrows();
        let p = x.ncols();

        if n == 0 || y.len() != n {
            return Err(TimeSeriesError::InvalidInput(
                "x and y must match in length and be non-empty".to_string(),
            ));
        }

        // Initial prediction: mean of y
        let mean_y: f64 = y.iter().sum::<f64>() / n as f64;
        let mut predictions = vec![mean_y; n];
        let mut stumps = Vec::with_capacity(n_estimators);

        for _ in 0..n_estimators {
            // Compute residuals (negative gradient of squared loss)
            let residuals: Vec<f64> = y
                .iter()
                .zip(predictions.iter())
                .map(|(&yi, &pi)| yi - pi)
                .collect();

            // Fit a stump to the residuals
            let stump = fit_best_stump(x, &residuals, n, p);
            // Update predictions
            for i in 0..n {
                let pred = if x[[i, stump.feature_idx]] <= stump.threshold {
                    stump.left_value
                } else {
                    stump.right_value
                };
                predictions[i] += learning_rate * pred;
            }

            stumps.push(stump);
        }

        Ok(Self {
            stumps,
            learning_rate,
            initial_prediction: mean_y,
        })
    }

    /// Predict using the ensemble of stumps.
    pub fn predict(&self, x: &Array2<f64>) -> Vec<f64> {
        let n = x.nrows();
        let mut preds = vec![self.initial_prediction; n];

        for stump in &self.stumps {
            for i in 0..n {
                let val = if x[[i, stump.feature_idx]] <= stump.threshold {
                    stump.left_value
                } else {
                    stump.right_value
                };
                preds[i] += self.learning_rate * val;
            }
        }
        preds
    }

    /// Compute mean squared error of the ensemble on given data.
    pub fn mse(&self, x: &Array2<f64>, y: &[f64]) -> f64 {
        let preds = self.predict(x);
        let n = y.len();
        if n == 0 {
            return 0.0;
        }
        preds
            .iter()
            .zip(y.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / n as f64
    }
}

/// Fit the best single decision stump to a target vector.
fn fit_best_stump(x: &Array2<f64>, target: &[f64], n: usize, p: usize) -> DecisionStump {
    let mut best_loss = f64::MAX;
    let mut best = DecisionStump {
        feature_idx: 0,
        threshold: 0.0,
        left_value: 0.0,
        right_value: 0.0,
    };

    // Sample up to 20 thresholds per feature for efficiency
    let max_thresholds = 20;

    for feat in 0..p {
        // Collect feature values
        let col: Vec<f64> = (0..n).map(|i| x[[i, feat]]).collect();
        let mut sorted = col.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted.dedup();

        // Pick evenly-spaced thresholds
        let step = if sorted.len() <= max_thresholds {
            1
        } else {
            sorted.len() / max_thresholds
        };

        let mut idx = 0;
        while idx < sorted.len() {
            let threshold = sorted[idx];

            // Compute left/right means
            let mut left_sum = 0.0;
            let mut left_count = 0usize;
            let mut right_sum = 0.0;
            let mut right_count = 0usize;

            for i in 0..n {
                if col[i] <= threshold {
                    left_sum += target[i];
                    left_count += 1;
                } else {
                    right_sum += target[i];
                    right_count += 1;
                }
            }

            if left_count == 0 || right_count == 0 {
                idx += step;
                continue;
            }

            let left_val = left_sum / left_count as f64;
            let right_val = right_sum / right_count as f64;

            // Compute loss
            let mut loss = 0.0;
            for i in 0..n {
                let pred = if col[i] <= threshold {
                    left_val
                } else {
                    right_val
                };
                loss += (target[i] - pred).powi(2);
            }

            if loss < best_loss {
                best_loss = loss;
                best = DecisionStump {
                    feature_idx: feat,
                    threshold,
                    left_value: left_val,
                    right_value: right_val,
                };
            }

            idx += step;
        }
    }

    best
}

// ---------------------------------------------------------------------------
// Combined Energy Forecaster
// ---------------------------------------------------------------------------

/// Combined energy forecaster using quantile regression and boosted stumps.
///
/// Fits multiple quantile regressors plus a boosted-stump point forecaster,
/// then ensures quantile non-crossing via isotonic sort.
#[derive(Debug, Clone)]
pub struct EnergyForecaster {
    /// Quantile regressors, one per quantile level
    quantile_models: Vec<QuantileRegressor>,
    /// Boosted stumps for point forecast
    point_model: GradientBoostedStumps,
    /// Feature configuration used during fitting
    feature_config: FeatureConfig,
    /// Forecast configuration
    config: EnergyForecastConfig,
    /// Training feature matrix (needed for future feature alignment)
    last_profile: LoadProfile,
}

impl EnergyForecaster {
    /// Fit the combined forecaster on historical load data.
    pub fn fit(profile: &LoadProfile, config: &EnergyForecastConfig) -> Result<Self> {
        let feat_config = FeatureConfig::default();
        let x = FeatureMatrix::build(profile, &feat_config)?;
        let y = &profile.load_values;

        // Fit point model
        let point_model =
            GradientBoostedStumps::fit(&x, y, config.n_estimators, config.learning_rate)?;

        // Fit quantile models
        let mut quantile_models = Vec::with_capacity(config.quantile_levels.len());
        for &tau in &config.quantile_levels {
            let qr = QuantileRegressor::fit(&x, y, tau)?;
            quantile_models.push(qr);
        }

        Ok(Self {
            quantile_models,
            point_model,
            feature_config: feat_config,
            config: config.clone(),
            last_profile: profile.clone(),
        })
    }

    /// Generate probabilistic forecasts for `horizon` steps ahead.
    ///
    /// Uses the last portion of training data as features for the forecast
    /// horizon. Quantile non-crossing is enforced by sorting quantile
    /// predictions at each time step.
    pub fn predict(&self, horizon: usize) -> Result<EnergyForecastResult> {
        let n = self.last_profile.load_values.len();
        if horizon == 0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "horizon".to_string(),
                message: "must be > 0".to_string(),
            });
        }

        // Build forecast features step-by-step, extending the profile
        let mut extended_ts = self.last_profile.timestamps.clone();
        let mut extended_load = self.last_profile.load_values.clone();
        let mut extended_temps = self.last_profile.temperatures.clone();

        let last_t = extended_ts.last().copied().unwrap_or(0.0);

        // Pre-extend all vectors to the full length so they stay aligned
        for h in 1..=horizon {
            extended_ts.push(last_t + h as f64);
            // Placeholder load values (will be filled iteratively)
            extended_load.push(0.0);
            if let Some(ref mut t) = extended_temps {
                let last_temp = t.last().copied().unwrap_or(65.0);
                t.push(last_temp);
            }
        }

        // Iteratively fill forecast load values using point model
        for h in 0..horizon {
            let row_idx = n + h;
            let profile_ext = LoadProfile {
                timestamps: extended_ts.clone(),
                load_values: extended_load.clone(),
                temperatures: extended_temps.clone(),
                holiday_mask: None,
            };

            let x_ext = FeatureMatrix::build(&profile_ext, &self.feature_config)?;
            if row_idx >= x_ext.nrows() {
                break;
            }
            let row = x_ext.row(row_idx);
            let row_2d = row.insert_axis(Axis(0)).to_owned();

            let pt = self.point_model.predict(&row_2d);
            extended_load[row_idx] = pt[0];
        }

        // Build final feature matrix for the forecast portion
        let full_profile = LoadProfile {
            timestamps: extended_ts,
            load_values: extended_load,
            temperatures: extended_temps,
            holiday_mask: None,
        };
        let x_full = FeatureMatrix::build(&full_profile, &self.feature_config)?;

        let forecast_start = n;
        let forecast_end = (n + horizon).min(x_full.nrows());
        let x_forecast = x_full
            .slice(s![forecast_start..forecast_end, ..])
            .to_owned();

        // Point forecasts from boosted stumps
        let point_forecasts = self.point_model.predict(&x_forecast);

        // Quantile forecasts
        let mut quantile_forecasts: Vec<QuantileForecast> = self
            .quantile_models
            .iter()
            .zip(self.config.quantile_levels.iter())
            .map(|(model, &tau)| {
                let values = model.predict(&x_forecast);
                QuantileForecast {
                    quantile: tau,
                    values,
                }
            })
            .collect();

        // Enforce quantile non-crossing via isotonic sort at each time step
        enforce_non_crossing(&mut quantile_forecasts);

        Ok(EnergyForecastResult {
            point_forecasts,
            quantile_forecasts,
            metrics: None,
        })
    }
}

/// Sort quantile forecasts at each time step to prevent crossing.
fn enforce_non_crossing(forecasts: &mut [QuantileForecast]) {
    if forecasts.is_empty() {
        return;
    }
    let horizon = forecasts[0].values.len();
    let n_q = forecasts.len();

    // Sort by quantile level
    forecasts.sort_by(|a, b| {
        a.quantile
            .partial_cmp(&b.quantile)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // At each time step, sort the values
    for t in 0..horizon {
        let mut vals: Vec<f64> = forecasts.iter().map(|q| q.values[t]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (q_idx, val) in vals.into_iter().enumerate() {
            if q_idx < n_q {
                forecasts[q_idx].values[t] = val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_quantile_regression_linear() {
        // y = 2*x + 1 with no noise
        let n = 50;
        let mut x_data = Vec::with_capacity(n);
        let mut y_data = Vec::with_capacity(n);
        for i in 0..n {
            let xi = i as f64 / n as f64;
            x_data.push(xi);
            y_data.push(2.0 * xi + 1.0);
        }
        let x = Array2::from_shape_vec((n, 1), x_data).expect("shape ok");
        let qr = QuantileRegressor::fit(&x, &y_data, 0.5).expect("fit ok");
        let preds = qr.predict(&x);
        // Should approximately recover y
        let max_err: f64 = preds
            .iter()
            .zip(y_data.iter())
            .map(|(p, y)| (p - y).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 0.5, "max error {} too large", max_err);
    }

    #[test]
    fn test_quantile_non_crossing() {
        let n = 20;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y_data: Vec<f64> = (0..n).map(|i| (i as f64).sin() * 10.0 + 50.0).collect();
        let x = Array2::from_shape_vec((n, 1), x_data).expect("shape ok");

        let quantiles = [0.1, 0.5, 0.9];
        let mut qfs: Vec<QuantileForecast> = quantiles
            .iter()
            .map(|&tau| {
                let qr = QuantileRegressor::fit(&x, &y_data, tau).expect("fit ok");
                QuantileForecast {
                    quantile: tau,
                    values: qr.predict(&x),
                }
            })
            .collect();

        enforce_non_crossing(&mut qfs);

        // Check non-crossing: q10 <= q50 <= q90 at each step
        for t in 0..n {
            assert!(
                qfs[0].values[t] <= qfs[1].values[t] + 1e-10,
                "crossing at t={}",
                t
            );
            assert!(
                qfs[1].values[t] <= qfs[2].values[t] + 1e-10,
                "crossing at t={}",
                t
            );
        }
    }

    #[test]
    fn test_boosted_stumps_training_loss_decreases() {
        let n = 30;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y_data: Vec<f64> = (0..n).map(|i| 3.0 * (i as f64) + 5.0).collect();
        let x = Array2::from_shape_vec((n, 1), x_data).expect("shape ok");

        let model_few = GradientBoostedStumps::fit(&x, &y_data, 5, 0.1).expect("fit ok");
        let model_many = GradientBoostedStumps::fit(&x, &y_data, 50, 0.1).expect("fit ok");

        let mse_few = model_few.mse(&x, &y_data);
        let mse_many = model_many.mse(&x, &y_data);

        assert!(
            mse_many <= mse_few + 1e-10,
            "more estimators should not increase training loss: {} vs {}",
            mse_many,
            mse_few
        );
    }

    #[test]
    fn test_energy_forecaster_fit_predict() {
        let n = 72; // 3 days of hourly data
        let profile = LoadProfile {
            timestamps: (0..n).map(|i| i as f64).collect(),
            load_values: (0..n)
                .map(|i| 100.0 + 20.0 * (2.0 * std::f64::consts::PI * i as f64 / 24.0).sin())
                .collect(),
            temperatures: Some(
                (0..n)
                    .map(|i| 60.0 + 10.0 * (i as f64 / 24.0).sin())
                    .collect(),
            ),
            holiday_mask: None,
        };

        let config = EnergyForecastConfig {
            horizon: 6,
            quantile_levels: vec![0.1, 0.5, 0.9],
            n_estimators: 20,
            learning_rate: 0.1,
            ..EnergyForecastConfig::default()
        };

        let forecaster = EnergyForecaster::fit(&profile, &config).expect("fit ok");
        let result = forecaster.predict(6).expect("predict ok");

        assert_eq!(result.point_forecasts.len(), 6);
        assert_eq!(result.quantile_forecasts.len(), 3);
        for qf in &result.quantile_forecasts {
            assert_eq!(qf.values.len(), 6);
        }
    }
}
