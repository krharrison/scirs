//! Pipeline builder and associated error types.
//!
//! [`Pipeline`] provides a fluent builder API for composing
//! [`FeatureTransformer`] steps into an ordered sequence. It supports
//! `fit`, `transform`, `fit_transform`, and `predict` operations.

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::fmt;
use thiserror::Error;

use super::traits::{FeatureTransformer, ModelPredictor};

/// Errors that can occur within the ML pipeline.
#[derive(Error, Debug, Clone)]
pub enum PipelineError {
    /// Attempted to transform before the pipeline (or a step) was fitted.
    #[error("Pipeline step '{0}' has not been fitted. Call fit() before transform().")]
    NotFitted(String),

    /// The input data is empty (zero rows or zero columns).
    #[error("Empty input data in step '{0}': data must have at least one row and one column.")]
    EmptyInput(String),

    /// The number of features in the input does not match what was seen during fitting.
    #[error("Feature count mismatch in step '{step}': expected {expected} columns, got {actual}.")]
    FeatureCountMismatch {
        /// Name of the transformer/step that detected the mismatch.
        step: String,
        /// Number of features the step was fitted on.
        expected: usize,
        /// Number of features in the new input.
        actual: usize,
    },

    /// A numeric overflow or underflow was detected.
    #[error("Numeric error in step '{0}': {1}")]
    NumericError(String, String),

    /// No predictor has been set on the pipeline.
    #[error("No ModelPredictor is configured. Call set_predictor() before predict().")]
    NoPredictorSet,

    /// An implementation-specific error from a transformer or predictor.
    #[error("Step '{step}' error: {message}")]
    StepError {
        /// Name of the step.
        step: String,
        /// Error description.
        message: String,
    },

    /// Invalid configuration of the pipeline or a step.
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

/// A single step in the pipeline.
///
/// Each step wraps a [`FeatureTransformer`] implementation.
pub struct PipelineStep<T: Clone + fmt::Debug + Float + FromPrimitive + 'static> {
    transformer: Box<dyn FeatureTransformer<T>>,
}

impl<T: Clone + fmt::Debug + Float + FromPrimitive + 'static> PipelineStep<T> {
    /// Create a new step from a boxed transformer.
    pub fn new(transformer: Box<dyn FeatureTransformer<T>>) -> Self {
        Self { transformer }
    }

    /// Fit the step to data.
    pub fn fit(&mut self, data: &Array2<T>) -> Result<(), PipelineError> {
        self.transformer.fit(data)
    }

    /// Transform data using the fitted step.
    pub fn transform(&self, data: &Array2<T>) -> Result<Array2<T>, PipelineError> {
        self.transformer.transform(data)
    }

    /// Fit and transform in a single call.
    pub fn fit_transform(&mut self, data: &Array2<T>) -> Result<Array2<T>, PipelineError> {
        self.transformer.fit_transform(data)
    }

    /// The name of the underlying transformer.
    pub fn name(&self) -> &str {
        self.transformer.name()
    }

    /// Whether the underlying transformer has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.transformer.is_fitted()
    }
}

/// Composable ML pipeline builder.
///
/// `Pipeline<T>` holds an ordered list of [`FeatureTransformer`] steps.
/// On `fit`, each step is fitted sequentially, with the output of step *i*
/// used as input to step *i+1* (chain-fitting). On `transform`, data passes
/// through all steps in order.
///
/// An optional [`ModelPredictor`] can be attached with [`add_predictor`]; if
/// set, calling [`predict`] will transform data and then run the predictor.
///
/// # Generic Parameter
///
/// - `T`: The element type (e.g., `f32`, `f64`). Must implement [`Float`] and
///   [`FromPrimitive`] from `num-traits`.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::{Pipeline, StandardScaler, MinMaxScaler};
/// use ndarray::Array2;
///
/// let data = Array2::from_shape_vec(
///     (4, 2),
///     vec![1.0f64, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
/// ).expect("should succeed");
///
/// let mut pipeline = Pipeline::new()
///     .add_transformer(Box::new(StandardScaler::new()))
///     .add_transformer(Box::new(MinMaxScaler::new()));
///
/// let result = pipeline.fit_transform(&data).expect("should succeed");
/// assert_eq!(result.shape(), &[4, 2]);
/// # }
/// ```
///
/// [`add_predictor`]: Pipeline::add_predictor
/// [`predict`]: Pipeline::predict
pub struct Pipeline<T: Clone + fmt::Debug + Float + FromPrimitive + 'static> {
    steps: Vec<PipelineStep<T>>,
    predictor: Option<Box<dyn ModelPredictor<T>>>,
    is_fitted: bool,
}

impl<T: Clone + fmt::Debug + Float + FromPrimitive + 'static> Pipeline<T> {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            predictor: None,
            is_fitted: false,
        }
    }

    /// Append a transformer step to the end of the pipeline.
    ///
    /// Steps are applied in the order they are added.
    ///
    /// Returns `self` so calls can be chained (builder pattern).
    pub fn add_transformer(mut self, transformer: Box<dyn FeatureTransformer<T>>) -> Self {
        self.steps.push(PipelineStep::new(transformer));
        self
    }

    /// Attach a model predictor that runs after all transformer steps.
    pub fn add_predictor(mut self, predictor: Box<dyn ModelPredictor<T>>) -> Self {
        self.predictor = Some(predictor);
        self
    }

    /// Fit all transformer steps sequentially to the training data.
    ///
    /// Each step is fitted on the output of the previous step, enabling
    /// the pipeline to correctly propagate learned statistics end-to-end.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered during fitting.
    pub fn fit(&mut self, data: &Array2<T>) -> Result<(), PipelineError> {
        if self.steps.is_empty() {
            self.is_fitted = true;
            return Ok(());
        }

        let mut current_data = data.clone();
        for step in &mut self.steps {
            current_data = step.fit_transform(&current_data)?;
        }
        self.is_fitted = true;
        Ok(())
    }

    /// Transform data through all fitted steps.
    ///
    /// # Errors
    ///
    /// - [`PipelineError::NotFitted`] if [`fit`] has not been called.
    /// - Any error from the individual transformer steps.
    ///
    /// [`fit`]: Pipeline::fit
    pub fn transform(&self, data: &Array2<T>) -> Result<Array2<T>, PipelineError> {
        if self.steps.is_empty() {
            return Ok(data.clone());
        }
        if !self.is_fitted {
            return Err(PipelineError::NotFitted("Pipeline".to_string()));
        }
        let mut current_data = data.clone();
        for step in &self.steps {
            current_data = step.transform(&current_data)?;
        }
        Ok(current_data)
    }

    /// Fit all steps and transform in a single call.
    ///
    /// Equivalent to calling [`fit`] followed by [`transform`] on the same data.
    ///
    /// [`fit`]: Pipeline::fit
    /// [`transform`]: Pipeline::transform
    pub fn fit_transform(&mut self, data: &Array2<T>) -> Result<Array2<T>, PipelineError> {
        if self.steps.is_empty() {
            self.is_fitted = true;
            return Ok(data.clone());
        }
        let mut current_data = data.clone();
        for step in &mut self.steps {
            current_data = step.fit_transform(&current_data)?;
        }
        self.is_fitted = true;
        Ok(current_data)
    }

    /// Transform data and run the predictor.
    ///
    /// # Errors
    ///
    /// - [`PipelineError::NoPredictorSet`] if no predictor has been configured.
    /// - Any error from the transformer steps or the predictor.
    pub fn predict(&self, data: &Array2<T>) -> Result<Array1<T>, PipelineError> {
        let transformed = self.transform(data)?;
        match &self.predictor {
            Some(pred) => pred.predict(&transformed),
            None => Err(PipelineError::NoPredictorSet),
        }
    }

    /// Fit all steps, transform data, and run the predictor in a single call.
    pub fn fit_predict(&mut self, data: &Array2<T>) -> Result<Array1<T>, PipelineError> {
        let transformed = self.fit_transform(data)?;
        match &self.predictor {
            Some(pred) => pred.predict(&transformed),
            None => Err(PipelineError::NoPredictorSet),
        }
    }

    /// Whether the pipeline has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Number of transformer steps in the pipeline.
    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    /// Names of all transformer steps in order.
    pub fn step_names(&self) -> Vec<&str> {
        self.steps.iter().map(|s| s.name()).collect()
    }

    /// Whether a predictor has been attached.
    pub fn has_predictor(&self) -> bool {
        self.predictor.is_some()
    }
}

/// A default empty pipeline. Equivalent to `Pipeline::new()`.
impl<T: Clone + fmt::Debug + Float + FromPrimitive + 'static> Default for Pipeline<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod builder_tests {
    use super::*;
    use crate::ml_pipeline::scalers::{MinMaxScaler, NormalizerTransform, StandardScaler};
    use ndarray::Array2;

    fn sample_data() -> Array2<f64> {
        Array2::from_shape_vec((4, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0])
            .expect("shape is valid")
    }

    #[test]
    fn test_pipeline_default_is_empty() {
        let p: Pipeline<f64> = Pipeline::default();
        assert_eq!(p.n_steps(), 0);
        assert!(!p.is_fitted());
        assert!(!p.has_predictor());
    }

    #[test]
    fn test_pipeline_empty_is_identity() {
        let mut p: Pipeline<f64> = Pipeline::new();
        let data = sample_data();
        let result = p.fit_transform(&data).expect("empty pipeline is identity");
        assert_eq!(result, data);
    }

    #[test]
    fn test_pipeline_step_names() {
        let p = Pipeline::<f64>::new()
            .add_transformer(Box::new(StandardScaler::new()))
            .add_transformer(Box::new(MinMaxScaler::new()))
            .add_transformer(Box::new(NormalizerTransform::l2()));
        assert_eq!(
            p.step_names(),
            vec!["StandardScaler", "MinMaxScaler", "NormalizerL2"]
        );
    }

    #[test]
    fn test_pipeline_transform_before_fit_returns_error() {
        let p = Pipeline::<f64>::new().add_transformer(Box::new(StandardScaler::new()));
        let data = sample_data();
        let result = p.transform(&data);
        assert!(matches!(result, Err(PipelineError::NotFitted(_))));
    }

    #[test]
    fn test_pipeline_predict_without_predictor_returns_error() {
        let mut p = Pipeline::<f64>::new().add_transformer(Box::new(StandardScaler::new()));
        let data = sample_data();
        p.fit(&data).expect("fit should succeed");
        let result = p.predict(&data);
        assert!(matches!(result, Err(PipelineError::NoPredictorSet)));
    }

    #[test]
    fn test_pipeline_multiple_steps_shape_preserved() {
        let data = sample_data();
        let mut p = Pipeline::<f64>::new()
            .add_transformer(Box::new(StandardScaler::new()))
            .add_transformer(Box::new(MinMaxScaler::new()));
        let result = p
            .fit_transform(&data)
            .expect("multi-step pipeline should work");
        assert_eq!(result.shape(), data.shape());
    }

    #[test]
    fn test_pipeline_fit_then_transform_new_data() {
        let train = sample_data();
        let test = Array2::from_shape_vec((2, 2), vec![100.0_f64, 200.0, -5.0, -50.0])
            .expect("shape is valid");

        let mut p = Pipeline::<f64>::new().add_transformer(Box::new(StandardScaler::new()));
        p.fit(&train).expect("fit should succeed");
        let result = p
            .transform(&test)
            .expect("transform new data should succeed");
        assert_eq!(result.shape(), test.shape());
    }

    #[test]
    fn test_pipeline_is_fitted_transitions() {
        let data = sample_data();
        let mut p = Pipeline::<f64>::new().add_transformer(Box::new(StandardScaler::new()));
        assert!(!p.is_fitted());
        p.fit(&data).expect("fit should succeed");
        assert!(p.is_fitted());
    }
}
