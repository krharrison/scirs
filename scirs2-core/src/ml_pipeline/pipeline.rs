//! High-level ML pipeline with regression and cross-validation support.
//!
//! This module extends the base [`Pipeline`] builder (from [`super::builder`])
//! with:
//!
//! - [`RegressionPipeline`] — chains a preprocessing [`Pipeline`] with any
//!   regression predictor into a single end-to-end object.
//! - [`cross_validate_regression`] — k-fold cross-validation producing R² per fold.
//!
//! These types complement the existing [`crate::ml_pipeline::Pipeline`] builder
//! and [`crate::ml_pipeline::MLPipelineGeneric`].

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::builder::{Pipeline, PipelineError};
use super::evaluator::r2_score;
use super::predictor::{LinearRegressor, KnnRegressor};
use super::scalers::StandardScaler;
use super::traits::FeatureTransformer;

// ─────────────────────────────────────────────────────────────────────────────
// RegressionPredictor trait (local, f64-specialised)
// ─────────────────────────────────────────────────────────────────────────────

/// A trait for regression predictors that work within a [`RegressionPipeline`].
///
/// This is a specialised, object-safe version of `ModelPredictor` for
/// `f64` regression tasks. Implementors must be `Send + Sync`.
pub trait RegressionPredictor: Send + Sync {
    /// Fit the model to training data.
    fn fit_reg(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<(), PipelineError>;

    /// Predict target values for test data.
    fn predict_reg(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, PipelineError>;

    /// Human-readable model name.
    fn model_name(&self) -> &str;
}

// Blanket implementation for LinearRegressor
impl RegressionPredictor for LinearRegressor {
    fn fit_reg(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<(), PipelineError> {
        self.fit(x, y)
    }

    fn predict_reg(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, PipelineError> {
        self.predict(x)
    }

    fn model_name(&self) -> &str {
        "LinearRegressor"
    }
}

// Blanket implementation for KnnRegressor
impl RegressionPredictor for KnnRegressor {
    fn fit_reg(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<(), PipelineError> {
        self.fit(x, y)
    }

    fn predict_reg(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, PipelineError> {
        self.predict(x)
    }

    fn model_name(&self) -> &str {
        "KnnRegressor"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RegressionPipeline
// ─────────────────────────────────────────────────────────────────────────────

/// A complete regression pipeline: preprocessing transformers + a regression model.
///
/// Combines a [`Pipeline`] of feature transformers with any [`RegressionPredictor`]
/// into a unified structure with the standard `fit`/`predict`/`fit_predict` API.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::pipeline::RegressionPipeline;
/// use scirs2_core::ml_pipeline::predictor::LinearRegressor;
/// use scirs2_core::ml_pipeline::scalers::StandardScaler;
/// use scirs2_core::ml_pipeline::builder::Pipeline;
/// use ndarray::{Array1, Array2};
///
/// let preprocessing = Pipeline::<f64>::new()
///     .add_transformer(Box::new(StandardScaler::new()));
///
/// let predictor = LinearRegressor::new();
/// let mut reg_pipeline = RegressionPipeline::new(preprocessing, predictor);
///
/// // y = 2*x + 1
/// let x = Array2::from_shape_vec((4, 1), vec![1.0f64, 2.0, 3.0, 4.0]).expect("ok");
/// let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0]);
///
/// reg_pipeline.fit(x.view(), y.view()).expect("fit ok");
/// let preds = reg_pipeline.predict(x.view()).expect("predict ok");
/// assert_eq!(preds.len(), 4);
/// # }
/// ```
pub struct RegressionPipeline {
    preprocessing: Pipeline<f64>,
    predictor: Box<dyn RegressionPredictor>,
}

impl RegressionPipeline {
    /// Create a new `RegressionPipeline`.
    pub fn new(
        preprocessing: Pipeline<f64>,
        predictor: impl RegressionPredictor + 'static,
    ) -> Self {
        Self {
            preprocessing,
            predictor: Box::new(predictor),
        }
    }

    /// Fit all preprocessing steps and the predictor to training data.
    pub fn fit(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<(), PipelineError> {
        if x.nrows() == 0 {
            return Err(PipelineError::EmptyInput("RegressionPipeline.fit".to_string()));
        }
        let x_transformed = self.preprocessing.fit_transform(&x.to_owned())?;
        self.predictor.fit_reg(x_transformed.view(), y)
    }

    /// Predict using the fitted preprocessing steps and predictor.
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, PipelineError> {
        if x.nrows() == 0 {
            return Err(PipelineError::EmptyInput("RegressionPipeline.predict".to_string()));
        }
        let x_transformed = self.preprocessing.transform(&x.to_owned())?;
        self.predictor.predict_reg(x_transformed.view())
    }

    /// Fit and immediately predict on the same data.
    pub fn fit_predict(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<Array1<f64>, PipelineError> {
        self.fit(x, y)?;
        self.predict(x)
    }

    /// Human-readable name combining pipeline step names and predictor name.
    pub fn name(&self) -> String {
        let steps = self.preprocessing.step_names().join(" → ");
        let pred = self.predictor.model_name();
        if steps.is_empty() {
            pred.to_string()
        } else {
            format!("{steps} → {pred}")
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-validation
// ─────────────────────────────────────────────────────────────────────────────

/// Perform k-fold cross-validation on a regression pipeline.
///
/// Splits the data into `n_folds` folds, then for each fold:
/// 1. Creates a fresh pipeline using `pipeline_factory()`.
/// 2. Fits on the `n_folds - 1` non-held-out folds.
/// 3. Evaluates (R²) on the held-out fold.
///
/// Returns a vector of R² scores, one per fold.
///
/// # Arguments
///
/// * `pipeline_factory` — closure that creates a fresh, unfitted `RegressionPipeline`
/// * `x` — feature matrix, shape `(n_samples, n_features)`
/// * `y` — target vector, length `n_samples`
/// * `n_folds` — number of CV folds (must be ≥ 2)
///
/// # Errors
///
/// Returns `PipelineError::StepError` if `n_folds < 2` or `n_folds > n_samples`.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::pipeline::{RegressionPipeline, cross_validate_regression};
/// use scirs2_core::ml_pipeline::predictor::LinearRegressor;
/// use scirs2_core::ml_pipeline::builder::Pipeline;
/// use ndarray::{Array1, Array2};
///
/// // y = x (perfectly linear)
/// let x = Array2::from_shape_vec(
///     (10, 1),
///     (1..=10).map(|i| i as f64).collect::<Vec<_>>(),
/// ).expect("ok");
/// let y = Array1::from_vec((1..=10).map(|i| i as f64).collect::<Vec<_>>());
///
/// let factory = || RegressionPipeline::new(Pipeline::new(), LinearRegressor::new());
/// let scores = cross_validate_regression(factory, x.view(), y.view(), 5).expect("cv ok");
/// assert_eq!(scores.len(), 5);
/// for &score in &scores {
///     assert!(score > 0.5, "R² per fold should be high for linear data: {score}");
/// }
/// # }
/// ```
pub fn cross_validate_regression(
    pipeline_factory: impl Fn() -> RegressionPipeline,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    n_folds: usize,
) -> Result<Vec<f64>, PipelineError> {
    let n = x.nrows();
    if n_folds < 2 {
        return Err(PipelineError::StepError {
            step: "cross_validate_regression".to_string(),
            message: format!("n_folds must be >= 2, got {n_folds}"),
        });
    }
    if n_folds > n {
        return Err(PipelineError::StepError {
            step: "cross_validate_regression".to_string(),
            message: format!("n_folds ({n_folds}) must be <= n_samples ({n})"),
        });
    }

    let fold_size = n / n_folds;
    let mut scores = Vec::with_capacity(n_folds);

    for fold in 0..n_folds {
        // Determine test indices for this fold
        let test_start = fold * fold_size;
        let test_end = if fold == n_folds - 1 {
            n // last fold absorbs remainder
        } else {
            (fold + 1) * fold_size
        };

        // Collect train and test indices
        let train_indices: Vec<usize> = (0..n)
            .filter(|&i| i < test_start || i >= test_end)
            .collect();
        let test_indices: Vec<usize> = (test_start..test_end).collect();

        if train_indices.is_empty() || test_indices.is_empty() {
            continue;
        }

        // Build train set
        let n_train = train_indices.len();
        let n_test = test_indices.len();
        let ncols = x.ncols();

        let mut x_train_data = Vec::with_capacity(n_train * ncols);
        let mut y_train_data = Vec::with_capacity(n_train);
        for &i in &train_indices {
            for j in 0..ncols {
                x_train_data.push(x[[i, j]]);
            }
            y_train_data.push(y[i]);
        }
        let x_train = Array2::from_shape_vec((n_train, ncols), x_train_data).map_err(|e| {
            PipelineError::StepError {
                step: "cross_validate_regression".to_string(),
                message: format!("train array shape error: {e}"),
            }
        })?;
        let y_train = Array1::from_vec(y_train_data);

        // Build test set
        let mut x_test_data = Vec::with_capacity(n_test * ncols);
        let mut y_test_data = Vec::with_capacity(n_test);
        for &i in &test_indices {
            for j in 0..ncols {
                x_test_data.push(x[[i, j]]);
            }
            y_test_data.push(y[i]);
        }
        let x_test = Array2::from_shape_vec((n_test, ncols), x_test_data).map_err(|e| {
            PipelineError::StepError {
                step: "cross_validate_regression".to_string(),
                message: format!("test array shape error: {e}"),
            }
        })?;
        let y_test = Array1::from_vec(y_test_data);

        // Fit and evaluate
        let mut pipeline = pipeline_factory();
        pipeline.fit(x_train.view(), y_train.view())?;
        let preds = pipeline.predict(x_test.view())?;
        let score = r2_score(&y_test, &preds);
        scores.push(score);
    }

    Ok(scores)
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience constructors
// ─────────────────────────────────────────────────────────────────────────────

/// Build a `RegressionPipeline` with `StandardScaler` + `LinearRegressor`.
///
/// This is the most common configuration for a simple baseline regression model.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::pipeline::linear_regression_pipeline;
/// use ndarray::{Array1, Array2};
///
/// let mut p = linear_regression_pipeline();
/// let x = Array2::from_shape_vec((3, 1), vec![1.0f64, 2.0, 3.0]).expect("ok");
/// let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
/// p.fit(x.view(), y.view()).expect("fit ok");
/// # }
/// ```
pub fn linear_regression_pipeline() -> RegressionPipeline {
    RegressionPipeline::new(
        Pipeline::<f64>::new().add_transformer(Box::new(StandardScaler::new())),
        LinearRegressor::new(),
    )
}

/// Build a `RegressionPipeline` with `StandardScaler` + `KnnRegressor`.
///
/// # Arguments
///
/// * `k` — number of nearest neighbours
pub fn knn_regression_pipeline(k: usize) -> RegressionPipeline {
    RegressionPipeline::new(
        Pipeline::<f64>::new().add_transformer(Box::new(StandardScaler::new())),
        KnnRegressor::new(k),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn make_linear_data(n: usize) -> (Array2<f64>, Array1<f64>) {
        // y = 3*x + 2
        let x: Array2<f64> = Array2::from_shape_vec(
            (n, 1),
            (1..=n).map(|i| i as f64).collect::<Vec<_>>(),
        )
        .expect("shape ok");
        let y: Array1<f64> = Array1::from_vec(
            (1..=n).map(|i| 3.0 * i as f64 + 2.0).collect::<Vec<_>>(),
        );
        (x, y)
    }

    // ── RegressionPipeline ────────────────────────────────────────────────────

    #[test]
    fn test_regression_pipeline_fit_predict() {
        let (x, y) = make_linear_data(6);
        let mut pipeline = linear_regression_pipeline();
        pipeline.fit(x.view(), y.view()).expect("fit ok");
        let preds = pipeline.predict(x.view()).expect("predict ok");
        assert_eq!(preds.len(), 6);
        // Predictions should be close to true values
        for (i, (&yt, &yp)) in y.iter().zip(preds.iter()).enumerate() {
            assert!((yt - yp).abs() < 1.0, "row {i}: true={yt}, pred={yp}");
        }
    }

    #[test]
    fn test_regression_pipeline_knn() {
        let (x, y) = make_linear_data(8);
        let mut pipeline = knn_regression_pipeline(2);
        pipeline.fit(x.view(), y.view()).expect("fit ok");
        let preds = pipeline.predict(x.view()).expect("predict ok");
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_regression_pipeline_predict_before_fit_fails() {
        let pipeline = RegressionPipeline::new(Pipeline::new(), LinearRegressor::new());
        let x = Array2::from_shape_vec((2, 1), vec![1.0f64, 2.0]).expect("ok");
        // transform before fit should fail
        let result = pipeline.predict(x.view());
        assert!(result.is_err(), "predict before fit should fail");
    }

    #[test]
    fn test_regression_pipeline_fit_predict_method() {
        let (x, y) = make_linear_data(4);
        let mut pipeline = linear_regression_pipeline();
        let preds = pipeline.fit_predict(x.view(), y.view()).expect("fit_predict ok");
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_regression_pipeline_empty_input() {
        let mut pipeline = linear_regression_pipeline();
        let x: Array2<f64> = Array2::zeros((0, 1));
        let y: Array1<f64> = Array1::zeros(0);
        let result = pipeline.fit(x.view(), y.view());
        assert!(matches!(result, Err(PipelineError::EmptyInput(_))));
    }

    #[test]
    fn test_regression_pipeline_name() {
        let pipeline = linear_regression_pipeline();
        let name = pipeline.name();
        assert!(name.contains("StandardScaler"), "name: {name}");
        assert!(name.contains("LinearRegressor"), "name: {name}");
    }

    // ── cross_validate_regression ─────────────────────────────────────────────

    #[test]
    fn test_cross_validate_regression_linear_data() {
        // For perfectly linear data, all fold R² scores should be high
        let (x, y) = make_linear_data(20);
        let factory = || linear_regression_pipeline();
        let scores = cross_validate_regression(factory, x.view(), y.view(), 5).expect("cv ok");
        assert_eq!(scores.len(), 5);
        for &s in &scores {
            assert!(s > 0.5, "fold R² should be high for linear data: {s}");
        }
    }

    #[test]
    fn test_cross_validate_regression_n_folds_too_small() {
        let (x, y) = make_linear_data(10);
        let factory = || linear_regression_pipeline();
        let result = cross_validate_regression(factory, x.view(), y.view(), 1);
        assert!(matches!(result, Err(PipelineError::StepError { .. })));
    }

    #[test]
    fn test_cross_validate_regression_n_folds_too_large() {
        let (x, y) = make_linear_data(3);
        let factory = || linear_regression_pipeline();
        let result = cross_validate_regression(factory, x.view(), y.view(), 5);
        assert!(matches!(result, Err(PipelineError::StepError { .. })));
    }

    #[test]
    fn test_cross_validate_regression_2_folds() {
        let (x, y) = make_linear_data(10);
        let factory = || linear_regression_pipeline();
        let scores = cross_validate_regression(factory, x.view(), y.view(), 2).expect("cv ok");
        assert_eq!(scores.len(), 2);
    }

    #[test]
    fn test_cross_validate_regression_knn() {
        let (x, y) = make_linear_data(12);
        let factory = || knn_regression_pipeline(2);
        let scores = cross_validate_regression(factory, x.view(), y.view(), 3).expect("cv ok");
        assert_eq!(scores.len(), 3);
    }
}
