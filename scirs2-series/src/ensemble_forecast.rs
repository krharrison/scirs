//! Ensemble Forecasting Methods
//!
//! Combines multiple forecasts to produce more accurate and robust predictions.
//! Implements a variety of ensemble strategies from the forecasting literature:
//!
//! - **Simple average**: Equal-weight combination of all forecasts
//! - **Weighted average**: Inverse-error weighting from holdout performance
//! - **Stacking**: Train a meta-learner (ridge regression) on holdout predictions
//! - **Median ensemble**: Robust against outlier forecasts
//! - **Trimmed mean**: Remove extreme forecasts before averaging
//! - **Bagging**: Bootstrap aggregating for time series models
//! - **Dynamic selection**: Choose best model per forecast origin based on recent performance
//!
//! # References
//!
//! - Timmermann, A. (2006) "Forecast Combinations"
//! - Bates, J.M. & Granger, C.W.J. (1969) "The Combination of Forecasts"
//! - Breiman, L. (1996) "Bagging Predictors"

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive, ToPrimitive};
use std::fmt::{Debug, Display};

use crate::error::{Result, TimeSeriesError};
use crate::evaluation;

// ---------------------------------------------------------------------------
// Ensemble methods (functional API)
// ---------------------------------------------------------------------------

/// Combine forecasts using simple averaging
///
/// f_ensemble(t) = (1/K) * sum_{k=1}^{K} f_k(t)
///
/// # Arguments
/// * `forecasts` - Matrix of forecasts, shape (n_models, n_steps)
///
/// # Returns
/// Combined forecast of length n_steps
pub fn simple_average<F>(forecasts: &Array2<F>) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let (n_models, n_steps) = forecasts.dim();
    validate_forecast_matrix(n_models, n_steps)?;

    let n_models_f = F::from_usize(n_models).ok_or_else(|| {
        TimeSeriesError::ComputationError("Failed to convert model count".to_string())
    })?;

    let mut result = Array1::zeros(n_steps);
    for j in 0..n_steps {
        let mut sum = F::zero();
        for i in 0..n_models {
            sum = sum + forecasts[[i, j]];
        }
        result[j] = sum / n_models_f;
    }

    Ok(result)
}

/// Combine forecasts using inverse-error weighted averaging
///
/// Weights are proportional to 1/error on the holdout set.
/// Models with lower error get higher weight.
///
/// w_k = (1/e_k) / sum(1/e_j)
///
/// # Arguments
/// * `forecasts` - Matrix of forecasts, shape (n_models, n_steps)
/// * `holdout_actual` - Actual values on the holdout set
/// * `holdout_forecasts` - Forecasts on the holdout set, shape (n_models, n_holdout)
///
/// # Returns
/// Combined forecast of length n_steps
pub fn weighted_average<F>(
    forecasts: &Array2<F>,
    holdout_actual: &Array1<F>,
    holdout_forecasts: &Array2<F>,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display,
{
    let (n_models, n_steps) = forecasts.dim();
    validate_forecast_matrix(n_models, n_steps)?;

    let weights = compute_inverse_error_weights(holdout_actual, holdout_forecasts)?;
    weighted_combine(forecasts, &weights)
}

/// Combine forecasts using a stacking meta-learner (ridge regression)
///
/// Trains a ridge regression model on holdout predictions to learn
/// optimal combination weights. The combiner is trained to minimize
/// squared prediction error on holdout data.
///
/// # Arguments
/// * `forecasts` - Matrix of forecasts, shape (n_models, n_steps)
/// * `holdout_actual` - Actual values on the holdout set
/// * `holdout_forecasts` - Forecasts on the holdout set, shape (n_models, n_holdout)
/// * `lambda` - Ridge regularization parameter (>= 0)
///
/// # Returns
/// Combined forecast of length n_steps
pub fn stacking_ensemble<F>(
    forecasts: &Array2<F>,
    holdout_actual: &Array1<F>,
    holdout_forecasts: &Array2<F>,
    lambda: F,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display,
{
    let (n_models, n_steps) = forecasts.dim();
    validate_forecast_matrix(n_models, n_steps)?;

    if lambda < F::zero() {
        return Err(TimeSeriesError::InvalidParameter {
            name: "lambda".to_string(),
            message: format!("Ridge parameter must be non-negative, got {lambda}"),
        });
    }

    let (n_hm, n_holdout) = holdout_forecasts.dim();
    if n_hm != n_models {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n_models,
            actual: n_hm,
        });
    }
    if holdout_actual.len() != n_holdout {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n_holdout,
            actual: holdout_actual.len(),
        });
    }

    // Solve ridge regression: w = (X^T X + lambda I)^{-1} X^T y
    // where X is (n_holdout, n_models) and y is (n_holdout,)
    let weights = ridge_regression_weights(holdout_forecasts, holdout_actual, lambda)?;

    weighted_combine(forecasts, &weights)
}

/// Combine forecasts using the median
///
/// At each time step, takes the median across all model forecasts.
/// Robust to outlier forecasts from individual models.
///
/// # Arguments
/// * `forecasts` - Matrix of forecasts, shape (n_models, n_steps)
///
/// # Returns
/// Combined forecast of length n_steps
pub fn median_ensemble<F>(forecasts: &Array2<F>) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let (n_models, n_steps) = forecasts.dim();
    validate_forecast_matrix(n_models, n_steps)?;

    let mut result = Array1::zeros(n_steps);

    for j in 0..n_steps {
        let mut values: Vec<F> = (0..n_models).map(|i| forecasts[[i, j]]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        result[j] = compute_median(&values);
    }

    Ok(result)
}

/// Combine forecasts using trimmed mean
///
/// Removes the `trim_fraction` most extreme forecasts from each end
/// and averages the remaining. Provides robustness between simple average
/// and median.
///
/// # Arguments
/// * `forecasts` - Matrix of forecasts, shape (n_models, n_steps)
/// * `trim_fraction` - Fraction of models to trim from each end (0.0 to 0.5 exclusive)
///
/// # Returns
/// Combined forecast of length n_steps
pub fn trimmed_mean<F>(forecasts: &Array2<F>, trim_fraction: F) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display,
{
    let (n_models, n_steps) = forecasts.dim();
    validate_forecast_matrix(n_models, n_steps)?;

    let half = F::from_f64(0.5).ok_or_else(|| {
        TimeSeriesError::ComputationError("Failed to convert constant".to_string())
    })?;

    if trim_fraction < F::zero() || trim_fraction >= half {
        return Err(TimeSeriesError::InvalidParameter {
            name: "trim_fraction".to_string(),
            message: format!("Must be in [0, 0.5), got {trim_fraction}"),
        });
    }

    let n_trim = (F::from_usize(n_models).ok_or_else(|| {
        TimeSeriesError::ComputationError("Failed to convert model count".to_string())
    })? * trim_fraction)
        .floor();
    let n_trim_usize: usize = n_trim.to_usize().unwrap_or(0);

    let remaining = n_models - 2 * n_trim_usize;
    if remaining == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "trim_fraction".to_string(),
            message: "Trim fraction too large: no models remaining after trimming".to_string(),
        });
    }

    let remaining_f = F::from_usize(remaining).ok_or_else(|| {
        TimeSeriesError::ComputationError("Failed to convert remaining count".to_string())
    })?;

    let mut result = Array1::zeros(n_steps);

    for j in 0..n_steps {
        let mut values: Vec<F> = (0..n_models).map(|i| forecasts[[i, j]]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let sum: F = values[n_trim_usize..n_models - n_trim_usize]
            .iter()
            .copied()
            .fold(F::zero(), |acc, x| acc + x);

        result[j] = sum / remaining_f;
    }

    Ok(result)
}

/// Bootstrap aggregating (bagging) for time series
///
/// Generates B bootstrap samples of the training data using a block
/// bootstrap (to preserve temporal dependence), trains a model on each,
/// and averages the forecasts.
///
/// Since we operate on pre-computed forecasts, this function accepts
/// the original data and a user-provided fitting function.
///
/// # Arguments
/// * `data` - Original time series data
/// * `n_bootstrap` - Number of bootstrap samples
/// * `block_size` - Block length for block bootstrap
/// * `forecast_steps` - Number of steps to forecast
/// * `fit_and_forecast` - Function: (bootstrap_sample) -> forecast
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Bagged (averaged) forecast of length forecast_steps
pub fn bagging_forecast<F, Func>(
    data: &Array1<F>,
    n_bootstrap: usize,
    block_size: usize,
    forecast_steps: usize,
    fit_and_forecast: Func,
    seed: u64,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
    Func: Fn(&Array1<F>, usize) -> Result<Array1<F>>,
{
    let n = data.len();

    if n < block_size {
        return Err(TimeSeriesError::InsufficientData {
            message: "for bagging".to_string(),
            required: block_size,
            actual: n,
        });
    }

    if block_size == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "block_size".to_string(),
            message: "Block size must be positive".to_string(),
        });
    }

    if n_bootstrap == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "n_bootstrap".to_string(),
            message: "Number of bootstrap samples must be positive".to_string(),
        });
    }

    if forecast_steps == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "forecast_steps".to_string(),
            message: "Number of forecast steps must be positive".to_string(),
        });
    }

    let mut accumulated = Array1::zeros(forecast_steps);
    let mut success_count = 0usize;

    // Simple LCG for block bootstrap index generation (pure Rust, no external deps)
    let mut rng_state = seed;

    for _b in 0..n_bootstrap {
        // Generate block bootstrap sample
        let bootstrap_sample = block_bootstrap(data, block_size, &mut rng_state)?;

        // Fit model and forecast
        match fit_and_forecast(&bootstrap_sample, forecast_steps) {
            Ok(forecast) => {
                if forecast.len() == forecast_steps {
                    for j in 0..forecast_steps {
                        accumulated[j] = accumulated[j] + forecast[j];
                    }
                    success_count += 1;
                }
            }
            Err(_) => {
                // Skip failed bootstrap samples
                continue;
            }
        }
    }

    if success_count == 0 {
        return Err(TimeSeriesError::FittingError(
            "All bootstrap model fits failed".to_string(),
        ));
    }

    let count_f = F::from_usize(success_count)
        .ok_or_else(|| TimeSeriesError::ComputationError("Failed to convert count".to_string()))?;

    for j in 0..forecast_steps {
        accumulated[j] = accumulated[j] / count_f;
    }

    Ok(accumulated)
}

/// Dynamic ensemble selection
///
/// At each forecast origin, selects or weights models based on their
/// performance over a recent window of observations. Models that performed
/// well recently get higher weight.
///
/// # Arguments
/// * `forecasts` - Matrix of forecasts, shape (n_models, n_steps)
/// * `recent_actuals` - Recent actual values for performance evaluation
/// * `recent_forecasts` - Recent forecasts for each model, shape (n_models, n_recent)
/// * `top_k` - Number of top models to include (0 = use all with dynamic weights)
///
/// # Returns
/// Combined forecast of length n_steps
pub fn dynamic_ensemble_selection<F>(
    forecasts: &Array2<F>,
    recent_actuals: &Array1<F>,
    recent_forecasts: &Array2<F>,
    top_k: usize,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display,
{
    let (n_models, n_steps) = forecasts.dim();
    validate_forecast_matrix(n_models, n_steps)?;

    let (n_rm, n_recent) = recent_forecasts.dim();
    if n_rm != n_models {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n_models,
            actual: n_rm,
        });
    }
    if recent_actuals.len() != n_recent {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n_recent,
            actual: recent_actuals.len(),
        });
    }

    // Compute recent MAE for each model
    let mut model_errors: Vec<(usize, F)> = Vec::with_capacity(n_models);
    for i in 0..n_models {
        let model_forecast = recent_forecasts.row(i).to_owned();
        let mae_val = evaluation::mae(recent_actuals, &model_forecast).unwrap_or(F::infinity());
        model_errors.push((i, mae_val));
    }

    // Sort by error (ascending)
    model_errors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Select top-k or all
    let k = if top_k == 0 || top_k >= n_models {
        n_models
    } else {
        top_k
    };

    let selected = &model_errors[..k];

    // Compute inverse-error weights for selected models
    let mut weights = Array1::zeros(n_models);
    let mut weight_sum = F::zero();

    for &(idx, err) in selected {
        if err > F::epsilon() {
            let w = F::one() / err;
            weights[idx] = w;
            weight_sum = weight_sum + w;
        }
    }

    // If all selected models have zero error, use equal weights
    if weight_sum <= F::epsilon() {
        let equal_w = F::one()
            / F::from_usize(k).ok_or_else(|| {
                TimeSeriesError::ComputationError("Failed to convert k".to_string())
            })?;
        for &(idx, _) in selected {
            weights[idx] = equal_w;
        }
    } else {
        // Normalize
        for w in weights.iter_mut() {
            *w = *w / weight_sum;
        }
    }

    weighted_combine(forecasts, &weights)
}

// ---------------------------------------------------------------------------
// EnsembleForecaster builder pattern
// ---------------------------------------------------------------------------

/// Strategy for ensemble combination
#[derive(Debug, Clone)]
pub enum CombineStrategy {
    /// Simple average of all forecasts
    SimpleAverage,
    /// Inverse-error weighted average
    InverseErrorWeighted,
    /// Ridge-regression stacking
    Stacking {
        /// Regularization parameter
        lambda: f64,
    },
    /// Median ensemble
    Median,
    /// Trimmed mean
    TrimmedMean {
        /// Fraction to trim from each side
        trim_fraction: f64,
    },
    /// Dynamic selection of top-k
    DynamicSelection {
        /// Number of best models to use (0 = all)
        top_k: usize,
    },
}

/// Builder for ensemble forecast combination
///
/// Holds multiple model forecasts and combines them using the chosen strategy.
#[derive(Debug)]
pub struct EnsembleForecaster<F: Float> {
    /// Individual model forecasts, each of length forecast_steps
    model_forecasts: Vec<Array1<F>>,
    /// Model names (optional, for identification)
    model_names: Vec<String>,
    /// Holdout actual values (needed for weighted/stacking/dynamic methods)
    holdout_actual: Option<Array1<F>>,
    /// Holdout forecasts per model (needed for weighted/stacking/dynamic methods)
    holdout_forecasts: Vec<Array1<F>>,
    /// Combination strategy
    strategy: CombineStrategy,
}

impl<F: Float + FromPrimitive + Debug + Display> EnsembleForecaster<F> {
    /// Create a new ensemble forecaster
    pub fn new(strategy: CombineStrategy) -> Self {
        Self {
            model_forecasts: Vec::new(),
            model_names: Vec::new(),
            holdout_actual: None,
            holdout_forecasts: Vec::new(),
            strategy,
        }
    }

    /// Add a model forecast to the ensemble
    pub fn add_forecast(&mut self, name: &str, forecast: Array1<F>) -> &mut Self {
        self.model_names.push(name.to_string());
        self.model_forecasts.push(forecast);
        self
    }

    /// Set holdout data for performance-based weighting
    pub fn set_holdout(&mut self, actual: Array1<F>, forecasts: Vec<Array1<F>>) -> &mut Self {
        self.holdout_actual = Some(actual);
        self.holdout_forecasts = forecasts;
        self
    }

    /// Combine the forecasts using the configured strategy
    pub fn combine(&self) -> Result<Array1<F>> {
        if self.model_forecasts.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "No forecasts to combine".to_string(),
            ));
        }

        let forecast_matrix = self.build_forecast_matrix()?;

        match &self.strategy {
            CombineStrategy::SimpleAverage => simple_average(&forecast_matrix),
            CombineStrategy::InverseErrorWeighted => {
                let (actual, holdout_mat) = self.require_holdout()?;
                weighted_average(&forecast_matrix, &actual, &holdout_mat)
            }
            CombineStrategy::Stacking { lambda } => {
                let lambda_f = F::from_f64(*lambda).ok_or_else(|| {
                    TimeSeriesError::ComputationError("Failed to convert lambda".to_string())
                })?;
                let (actual, holdout_mat) = self.require_holdout()?;
                stacking_ensemble(&forecast_matrix, &actual, &holdout_mat, lambda_f)
            }
            CombineStrategy::Median => median_ensemble(&forecast_matrix),
            CombineStrategy::TrimmedMean { trim_fraction } => {
                let tf = F::from_f64(*trim_fraction).ok_or_else(|| {
                    TimeSeriesError::ComputationError("Failed to convert trim_fraction".to_string())
                })?;
                trimmed_mean(&forecast_matrix, tf)
            }
            CombineStrategy::DynamicSelection { top_k } => {
                let (actual, holdout_mat) = self.require_holdout()?;
                dynamic_ensemble_selection(&forecast_matrix, &actual, &holdout_mat, *top_k)
            }
        }
    }

    /// Get model weights (only meaningful after combination with weighted methods)
    pub fn compute_weights(&self) -> Result<Vec<(String, F)>> {
        if self.model_forecasts.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "No forecasts added".to_string(),
            ));
        }

        let weights = match &self.strategy {
            CombineStrategy::SimpleAverage => {
                let n = self.model_forecasts.len();
                let w = F::one()
                    / F::from_usize(n).ok_or_else(|| {
                        TimeSeriesError::ComputationError("Failed to convert".to_string())
                    })?;
                vec![w; n]
            }
            CombineStrategy::InverseErrorWeighted => {
                let (actual, holdout_mat) = self.require_holdout()?;
                compute_inverse_error_weights(&actual, &holdout_mat)?.to_vec()
            }
            CombineStrategy::Stacking { lambda } => {
                let lambda_f = F::from_f64(*lambda).ok_or_else(|| {
                    TimeSeriesError::ComputationError("Failed to convert".to_string())
                })?;
                let (actual, holdout_mat) = self.require_holdout()?;
                ridge_regression_weights(&holdout_mat, &actual, lambda_f)?.to_vec()
            }
            _ => {
                let n = self.model_forecasts.len();
                let w = F::one()
                    / F::from_usize(n).ok_or_else(|| {
                        TimeSeriesError::ComputationError("Failed to convert".to_string())
                    })?;
                vec![w; n]
            }
        };

        Ok(self
            .model_names
            .iter()
            .zip(weights.iter())
            .map(|(name, &w)| (name.clone(), w))
            .collect())
    }

    // --- internal helpers ---

    fn build_forecast_matrix(&self) -> Result<Array2<F>> {
        let n_models = self.model_forecasts.len();
        let n_steps = self.model_forecasts[0].len();

        for (i, f) in self.model_forecasts.iter().enumerate() {
            if f.len() != n_steps {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: n_steps,
                    actual: f.len(),
                });
            }
            let _ = i;
        }

        let mut matrix = Array2::zeros((n_models, n_steps));
        for (i, f) in self.model_forecasts.iter().enumerate() {
            for j in 0..n_steps {
                matrix[[i, j]] = f[j];
            }
        }

        Ok(matrix)
    }

    fn require_holdout(&self) -> Result<(Array1<F>, Array2<F>)> {
        let actual = self
            .holdout_actual
            .as_ref()
            .ok_or_else(|| {
                TimeSeriesError::InvalidInput(
                    "Holdout data required for this strategy. Call set_holdout() first."
                        .to_string(),
                )
            })?
            .clone();

        if self.holdout_forecasts.len() != self.model_forecasts.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.model_forecasts.len(),
                actual: self.holdout_forecasts.len(),
            });
        }

        let n_holdout = actual.len();
        let n_models = self.holdout_forecasts.len();
        let mut holdout_mat = Array2::zeros((n_models, n_holdout));
        for (i, hf) in self.holdout_forecasts.iter().enumerate() {
            if hf.len() != n_holdout {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: n_holdout,
                    actual: hf.len(),
                });
            }
            for j in 0..n_holdout {
                holdout_mat[[i, j]] = hf[j];
            }
        }

        Ok((actual, holdout_mat))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn validate_forecast_matrix(n_models: usize, n_steps: usize) -> Result<()> {
    if n_models == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "No models in forecast matrix".to_string(),
        ));
    }
    if n_steps == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Forecast length is zero".to_string(),
        ));
    }
    Ok(())
}

/// Compute inverse-error weights from holdout performance
fn compute_inverse_error_weights<F>(
    holdout_actual: &Array1<F>,
    holdout_forecasts: &Array2<F>,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Display,
{
    let n_models = holdout_forecasts.nrows();
    let mut weights = Array1::zeros(n_models);
    let mut weight_sum = F::zero();

    for i in 0..n_models {
        let model_forecast = holdout_forecasts.row(i).to_owned();
        let mae_val = evaluation::mae(holdout_actual, &model_forecast)?;

        if mae_val > F::epsilon() {
            let w = F::one() / mae_val;
            weights[i] = w;
            weight_sum = weight_sum + w;
        }
    }

    if weight_sum <= F::epsilon() {
        // All models have zero error => equal weights
        let equal_w = F::one()
            / F::from_usize(n_models).ok_or_else(|| {
                TimeSeriesError::ComputationError("Failed to convert".to_string())
            })?;
        for w in weights.iter_mut() {
            *w = equal_w;
        }
    } else {
        for w in weights.iter_mut() {
            *w = *w / weight_sum;
        }
    }

    Ok(weights)
}

/// Weighted combination of forecasts
fn weighted_combine<F>(forecasts: &Array2<F>, weights: &Array1<F>) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let (n_models, n_steps) = forecasts.dim();
    if weights.len() != n_models {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n_models,
            actual: weights.len(),
        });
    }

    let mut result = Array1::zeros(n_steps);
    for j in 0..n_steps {
        let mut sum = F::zero();
        for i in 0..n_models {
            sum = sum + weights[i] * forecasts[[i, j]];
        }
        result[j] = sum;
    }

    Ok(result)
}

/// Ridge regression to solve for stacking weights
///
/// Solves: w = (X^T X + lambda I)^{-1} X^T y
/// where X is (n_holdout, n_models) transposed from holdout_forecasts
fn ridge_regression_weights<F>(
    holdout_forecasts: &Array2<F>,
    holdout_actual: &Array1<F>,
    lambda: F,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n_models = holdout_forecasts.nrows();
    let n_holdout = holdout_forecasts.ncols();

    if holdout_actual.len() != n_holdout {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n_holdout,
            actual: holdout_actual.len(),
        });
    }

    // X is (n_holdout, n_models) -- transpose of holdout_forecasts
    // X^T X is (n_models, n_models)
    let mut xtx = Array2::<F>::zeros((n_models, n_models));
    for i in 0..n_models {
        for j in 0..n_models {
            let mut sum = F::zero();
            for t in 0..n_holdout {
                sum = sum + holdout_forecasts[[i, t]] * holdout_forecasts[[j, t]];
            }
            xtx[[i, j]] = sum;
        }
    }

    // Add ridge penalty
    for i in 0..n_models {
        xtx[[i, i]] = xtx[[i, i]] + lambda;
    }

    // X^T y is (n_models,)
    let mut xty = Array1::<F>::zeros(n_models);
    for i in 0..n_models {
        let mut sum = F::zero();
        for t in 0..n_holdout {
            sum = sum + holdout_forecasts[[i, t]] * holdout_actual[t];
        }
        xty[i] = sum;
    }

    // Solve via Gaussian elimination with partial pivoting
    let weights = solve_linear_system(&xtx, &xty)?;

    Ok(weights)
}

/// Solve Ax = b via Gaussian elimination with partial pivoting
fn solve_linear_system<F>(a: &Array2<F>, b: &Array1<F>) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: b.len(),
        });
    }

    // Augmented matrix [A | b]
    let mut aug = Array2::<F>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < F::epsilon() {
            return Err(TimeSeriesError::NumericalInstability(
                "Singular matrix in linear solve".to_string(),
            ));
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Eliminate below
        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..=n {
                let above = aug[[col, j]];
                aug[[row, j]] = aug[[row, j]] - factor * above;
            }
        }
    }

    // Back substitution
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum = sum - aug[[i, j]] * x[j];
        }
        let diag = aug[[i, i]];
        if diag.abs() < F::epsilon() {
            return Err(TimeSeriesError::NumericalInstability(
                "Zero diagonal in back substitution".to_string(),
            ));
        }
        x[i] = sum / diag;
    }

    Ok(x)
}

/// Compute median of a sorted slice
fn compute_median<F: Float + FromPrimitive>(sorted: &[F]) -> F {
    let n = sorted.len();
    if n == 0 {
        return F::zero();
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        let two = F::from_f64(2.0).unwrap_or(F::one() + F::one());
        (sorted[n / 2 - 1] + sorted[n / 2]) / two
    }
}

/// Block bootstrap for time series
///
/// Creates a bootstrap sample by randomly selecting contiguous blocks
/// of the original series.
fn block_bootstrap<F: Float + FromPrimitive>(
    data: &Array1<F>,
    block_size: usize,
    rng_state: &mut u64,
) -> Result<Array1<F>> {
    let n = data.len();
    let n_blocks = (n + block_size - 1) / block_size;
    let max_start = if n > block_size { n - block_size } else { 0 };

    let mut result = Vec::with_capacity(n_blocks * block_size);

    for _ in 0..n_blocks {
        // LCG random number
        *rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let start = if max_start > 0 {
            (*rng_state >> 33) as usize % (max_start + 1)
        } else {
            0
        };

        let end = (start + block_size).min(n);
        for i in start..end {
            result.push(data[i]);
        }
    }

    // Truncate to original length
    result.truncate(n);

    Ok(Array1::from(result))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    const TOL: f64 = 1e-10;

    fn make_forecast_matrix() -> Array2<f64> {
        // 3 models, 4 time steps
        Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, // Model 1
                1.2, 1.8, 3.2, 3.8, // Model 2
                0.8, 2.2, 2.8, 4.2, // Model 3
            ],
        )
        .expect("Failed to create forecast matrix")
    }

    #[test]
    fn test_simple_average() {
        let forecasts = make_forecast_matrix();
        let result = simple_average(&forecasts).expect("Simple average should succeed");
        assert_eq!(result.len(), 4);
        // Step 0: (1.0 + 1.2 + 0.8) / 3 = 1.0
        assert!((result[0] - 1.0).abs() < TOL);
        // Step 1: (2.0 + 1.8 + 2.2) / 3 = 2.0
        assert!((result[1] - 2.0).abs() < TOL);
    }

    #[test]
    fn test_simple_average_single_model() {
        let forecasts =
            Array2::from_shape_vec((1, 3), vec![5.0, 6.0, 7.0]).expect("Failed to create matrix");
        let result = simple_average(&forecasts).expect("Should succeed");
        assert!((result[0] - 5.0).abs() < TOL);
        assert!((result[1] - 6.0).abs() < TOL);
        assert!((result[2] - 7.0).abs() < TOL);
    }

    #[test]
    fn test_weighted_average() {
        let forecasts = make_forecast_matrix();
        let holdout_actual = array![10.0, 20.0, 30.0];
        let holdout_forecasts = Array2::from_shape_vec(
            (3, 3),
            vec![
                10.1, 20.1, 30.1, // Model 1: MAE = 0.1
                11.0, 21.0, 31.0, // Model 2: MAE = 1.0
                12.0, 22.0, 32.0, // Model 3: MAE = 2.0
            ],
        )
        .expect("Failed to create holdout matrix");

        let result = weighted_average(&forecasts, &holdout_actual, &holdout_forecasts)
            .expect("Weighted average should succeed");
        assert_eq!(result.len(), 4);
        // Model 1 should dominate (lowest error)
    }

    #[test]
    fn test_median_ensemble() {
        let forecasts = make_forecast_matrix();
        let result = median_ensemble(&forecasts).expect("Median should succeed");
        assert_eq!(result.len(), 4);
        // Step 0: sorted [0.8, 1.0, 1.2] => median = 1.0
        assert!((result[0] - 1.0).abs() < TOL);
        // Step 1: sorted [1.8, 2.0, 2.2] => median = 2.0
        assert!((result[1] - 2.0).abs() < TOL);
    }

    #[test]
    fn test_median_even_models() {
        let forecasts = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 10.0, // Model 1
                2.0, 20.0, // Model 2
                3.0, 30.0, // Model 3
                4.0, 40.0, // Model 4
            ],
        )
        .expect("Failed to create matrix");
        let result = median_ensemble(&forecasts).expect("Median should succeed");
        // Step 0: sorted [1,2,3,4] => median = (2+3)/2 = 2.5
        assert!((result[0] - 2.5).abs() < TOL);
    }

    #[test]
    fn test_trimmed_mean_no_trim() {
        let forecasts = make_forecast_matrix();
        let result_tm = trimmed_mean(&forecasts, 0.0).expect("Trimmed mean should succeed");
        let result_sa = simple_average(&forecasts).expect("Simple average should succeed");
        // With 0 trim, should equal simple average
        for i in 0..4 {
            assert!((result_tm[i] - result_sa[i]).abs() < TOL);
        }
    }

    #[test]
    fn test_trimmed_mean_with_outlier() {
        // 5 models, step 0: [1.0, 1.1, 1.0, 0.9, 100.0]
        let forecasts = Array2::from_shape_vec((5, 1), vec![1.0, 1.1, 1.0, 0.9, 100.0])
            .expect("Failed to create matrix");

        // Trim 20% from each end => remove 1 from each => use middle 3
        let result = trimmed_mean(&forecasts, 0.2).expect("Trimmed mean should succeed");
        // Sorted: [0.9, 1.0, 1.0, 1.1, 100.0] => trim 1 from each => [1.0, 1.0, 1.1] => mean = 1.0333...
        assert!((result[0] - (1.0 + 1.0 + 1.1) / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_trimmed_mean_invalid_fraction() {
        let forecasts = make_forecast_matrix();
        assert!(trimmed_mean(&forecasts, 0.5).is_err());
        assert!(trimmed_mean(&forecasts, -0.1).is_err());
    }

    #[test]
    fn test_stacking_ensemble() {
        let forecasts = make_forecast_matrix();
        let holdout_actual = array![10.0, 20.0, 30.0, 40.0, 50.0];
        let holdout_forecasts = Array2::from_shape_vec(
            (3, 5),
            vec![
                10.1, 20.1, 30.1, 40.1, 50.1, 10.5, 20.5, 30.5, 40.5, 50.5, 9.5, 19.5, 29.5, 39.5,
                49.5,
            ],
        )
        .expect("Failed to create holdout matrix");

        let result = stacking_ensemble(&forecasts, &holdout_actual, &holdout_forecasts, 0.01)
            .expect("Stacking should succeed");
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_stacking_negative_lambda() {
        let forecasts = make_forecast_matrix();
        let holdout_actual = array![1.0, 2.0];
        let holdout_forecasts =
            Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]).expect("Failed");
        assert!(stacking_ensemble(&forecasts, &holdout_actual, &holdout_forecasts, -1.0).is_err());
    }

    #[test]
    fn test_dynamic_ensemble_selection() {
        let forecasts = make_forecast_matrix();
        let recent_actuals = array![10.0, 20.0, 30.0];
        let recent_forecasts = Array2::from_shape_vec(
            (3, 3),
            vec![
                10.1, 20.1, 30.1, // Model 1: best
                11.0, 21.0, 31.0, // Model 2: ok
                15.0, 25.0, 35.0, // Model 3: worst
            ],
        )
        .expect("Failed to create matrix");

        // Select top 1 model
        let result = dynamic_ensemble_selection(&forecasts, &recent_actuals, &recent_forecasts, 1)
            .expect("Dynamic selection should succeed");
        assert_eq!(result.len(), 4);
        // Should be close to model 1's forecast
        assert!((result[0] - 1.0).abs() < TOL);
    }

    #[test]
    fn test_dynamic_all_models() {
        let forecasts = make_forecast_matrix();
        let recent_actuals = array![1.0, 2.0, 3.0];
        let recent_forecasts =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
                .expect("Failed");
        let result = dynamic_ensemble_selection(&forecasts, &recent_actuals, &recent_forecasts, 0)
            .expect("Should succeed");
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_bagging_forecast() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // Simple model: forecast = last value repeated
        let fit_fn = |sample: &Array1<f64>, steps: usize| -> Result<Array1<f64>> {
            let last = sample[sample.len() - 1];
            Ok(Array1::from_elem(steps, last))
        };

        let result = bagging_forecast(&data, 5, 3, 4, fit_fn, 42).expect("Bagging should succeed");
        assert_eq!(result.len(), 4);
        // All values should be reasonable (from the data range)
        for &v in result.iter() {
            assert!(v >= 0.0 && v <= 15.0);
        }
    }

    #[test]
    fn test_bagging_invalid_params() {
        let data = array![1.0, 2.0, 3.0];
        let fit_fn = |_: &Array1<f64>, _: usize| -> Result<Array1<f64>> { Ok(array![1.0]) };

        // Block size too large
        assert!(bagging_forecast(&data, 5, 10, 1, &fit_fn, 42).is_err());
        // Zero bootstrap
        assert!(bagging_forecast(&data, 0, 1, 1, &fit_fn, 42).is_err());
        // Zero block size
        assert!(bagging_forecast(&data, 5, 0, 1, &fit_fn, 42).is_err());
        // Zero forecast steps
        assert!(bagging_forecast(&data, 5, 1, 0, &fit_fn, 42).is_err());
    }

    #[test]
    fn test_empty_forecast_matrix() {
        let forecasts = Array2::<f64>::zeros((0, 5));
        assert!(simple_average(&forecasts).is_err());
    }

    #[test]
    fn test_ensemble_forecaster_builder() {
        let mut ef = EnsembleForecaster::new(CombineStrategy::SimpleAverage);
        ef.add_forecast("model_a", array![1.0, 2.0, 3.0]);
        ef.add_forecast("model_b", array![1.2, 1.8, 3.2]);
        ef.add_forecast("model_c", array![0.8, 2.2, 2.8]);

        let result = ef.combine().expect("Combine should succeed");
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < TOL);
    }

    #[test]
    fn test_ensemble_forecaster_weighted() {
        let mut ef = EnsembleForecaster::new(CombineStrategy::InverseErrorWeighted);
        ef.add_forecast("good", array![10.0, 20.0]);
        ef.add_forecast("bad", array![15.0, 25.0]);

        let holdout_actual = array![1.0, 2.0, 3.0];
        let holdout_f = vec![
            array![1.1, 2.1, 3.1], // good model
            array![2.0, 3.0, 4.0], // bad model
        ];
        ef.set_holdout(holdout_actual, holdout_f);

        let result = ef.combine().expect("Weighted combine should succeed");
        assert_eq!(result.len(), 2);

        let weights = ef.compute_weights().expect("Weights should succeed");
        assert_eq!(weights.len(), 2);
        // Good model should have higher weight
        assert!(weights[0].1 > weights[1].1);
    }

    #[test]
    fn test_ensemble_forecaster_no_forecasts() {
        let ef = EnsembleForecaster::<f64>::new(CombineStrategy::SimpleAverage);
        assert!(ef.combine().is_err());
    }

    #[test]
    fn test_ensemble_forecaster_missing_holdout() {
        let mut ef = EnsembleForecaster::new(CombineStrategy::InverseErrorWeighted);
        ef.add_forecast("model", array![1.0, 2.0]);
        assert!(ef.combine().is_err());
    }

    #[test]
    fn test_solve_linear_system_2x2() {
        // Simple test: 2x + y = 5, x + 3y = 7
        // Solution: x = 1.6, y = 1.8
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 3.0])
            .expect("Failed to create matrix");
        let b = array![5.0, 7.0];
        let x = solve_linear_system(&a, &b).expect("Solve should succeed");
        assert!((x[0] - 1.6).abs() < 1e-10);
        assert!((x[1] - 1.8).abs() < 1e-10);
    }

    #[test]
    fn test_block_bootstrap_preserves_length() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut rng = 42u64;
        let sample = block_bootstrap(&data, 3, &mut rng).expect("Bootstrap should succeed");
        assert_eq!(sample.len(), data.len());
    }

    #[test]
    fn test_compute_median_odd() {
        assert!((compute_median(&[1.0, 2.0, 3.0]) - 2.0).abs() < TOL);
    }

    #[test]
    fn test_compute_median_even() {
        assert!((compute_median(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < TOL);
    }

    #[test]
    fn test_weighted_combine_dimension_mismatch() {
        let forecasts = make_forecast_matrix();
        let weights = array![0.5, 0.5]; // Wrong length
        assert!(weighted_combine(&forecasts, &weights).is_err());
    }

    #[test]
    fn test_f32_simple_average() {
        let forecasts =
            Array2::from_shape_vec((2, 3), vec![1.0f32, 2.0, 3.0, 1.5, 2.5, 3.5]).expect("Failed");
        let result = simple_average(&forecasts);
        assert!(result.is_ok());
    }
}
