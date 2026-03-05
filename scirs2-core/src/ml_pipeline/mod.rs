//! High-Level ML Pipeline API for SciRS2
//!
//! This module provides a composable, type-safe machine learning pipeline
//! framework for SciRS2. It offers:
//!
//! - [`MLPipelineGeneric`] — generic composable pipeline that applies transforms sequentially
//! - [`ModelPredictor`] trait — defines the prediction interface
//! - [`FeatureTransformer`] trait — defines the fit/transform interface
//! - [`Pipeline`] builder — fluent API for constructing pipelines
//! - Concrete transformers: [`StandardScaler`], [`MinMaxScaler`], [`NormalizerTransform`]
//!
//! The existing streaming/DAG pipeline API is available via the [`pipeline_core`] submodule,
//! which exposes [`MLPipeline`], [`FeatureTransformerNode`], [`ModelPredictorNode`], etc.
//!
//! # Architecture
//!
//! The high-level API is built on two core traits:
//!
//! - **[`FeatureTransformer`]**: Stateful transformations that can be fitted to
//!   training data and then applied to new data. Implementations maintain internal
//!   state (e.g., learned mean/std for [`StandardScaler`]).
//!
//! - **[`ModelPredictor`]**: Encapsulates ML model inference. Implementations
//!   receive a feature matrix and return predictions.
//!
//! # Example
//!
//! ```rust
//! # #[cfg(feature = "ml_pipeline")]
//! # {
//! use scirs2_core::ml_pipeline::{Pipeline, StandardScaler, MinMaxScaler};
//! use ndarray::Array2;
//!
//! // Build a pipeline: StandardScaler -> MinMaxScaler
//! let mut pipeline = Pipeline::new()
//!     .add_transformer(Box::new(StandardScaler::new()))
//!     .add_transformer(Box::new(MinMaxScaler::new()));
//!
//! let data = Array2::from_shape_vec(
//!     (4, 2),
//!     vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
//! ).expect("should succeed");
//!
//! let transformed = pipeline.fit_transform(&data).expect("should succeed");
//! assert_eq!(transformed.shape(), &[4, 2]);
//! # }
//! ```

// Sub-modules
pub mod builder;
pub mod pipeline_core;
pub mod scalers;
pub mod traits;

// Re-export from pipeline_core (existing streaming/DAG pipeline)
pub use pipeline_core::{
    DataBatch, DataSample, DataType, ErrorStrategy, FeatureConstraint, FeatureSchema, FeatureValue,
    MLPipeline, MLPipelineError, ModelType, MonitoringConfig, PipelineConfig, PipelineMetrics,
    PipelineNode, TransformType,
};

// Re-export FeatureTransformer struct from pipeline_core (existing, renamed to avoid clash)
pub use pipeline_core::FeatureTransformer as FeatureTransformerNode;
// Re-export ModelPredictor struct from pipeline_core (existing, renamed to avoid clash)
pub use pipeline_core::ModelPredictor as ModelPredictorNode;

// Re-export high-level traits
pub use traits::{FeatureTransformer, ModelPredictor};

// Re-export concrete transformers
pub use scalers::{MinMaxScaler, NormType, NormalizerTransform, StandardScaler};

// Re-export the Pipeline builder and error types
pub use builder::{Pipeline, PipelineError, PipelineStep};

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::fmt;

/// A generic, composable ML pipeline that applies a sequence of transformer steps.
///
/// `MLPipelineGeneric<T>` wraps a [`Pipeline`] and an optional [`ModelPredictor`]
/// implementation into a unified structure that can be used for end-to-end
/// inference (transform then predict).
///
/// # Type Parameters
///
/// - `T`: The numeric type used for computations (typically `f32` or `f64`).
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::{MLPipelineGeneric, StandardScaler};
/// use ndarray::Array2;
///
/// let data = Array2::<f64>::from_shape_vec(
///     (3, 2),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
/// ).expect("should succeed");
///
/// let mut ml_pipeline = MLPipelineGeneric::<f64>::new();
/// ml_pipeline.add_transformer(Box::new(StandardScaler::new()));
/// ml_pipeline.fit(&data).expect("should succeed");
/// let transformed = ml_pipeline.transform(&data).expect("should succeed");
/// assert_eq!(transformed.shape(), &[3, 2]);
/// # }
/// ```
pub struct MLPipelineGeneric<T: Clone + fmt::Debug + Float + FromPrimitive + 'static> {
    steps: Vec<Box<dyn FeatureTransformer<T>>>,
    predictor: Option<Box<dyn ModelPredictor<T>>>,
    is_fitted: bool,
}

impl<T: Clone + fmt::Debug + Float + FromPrimitive + Send + Sync + 'static> MLPipelineGeneric<T> {
    /// Create a new, empty generic ML pipeline.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            predictor: None,
            is_fitted: false,
        }
    }

    /// Add a feature transformer step to the pipeline.
    ///
    /// Steps are applied in the order they are added during `transform()`.
    pub fn add_transformer(&mut self, transformer: Box<dyn FeatureTransformer<T>>) {
        self.steps.push(transformer);
    }

    /// Set the model predictor (the final prediction step).
    pub fn set_predictor(&mut self, predictor: Box<dyn ModelPredictor<T>>) {
        self.predictor = Some(predictor);
    }

    /// Fit the pipeline (all transformers) to the given data.
    ///
    /// This iterates through all transformer steps and calls `fit_transform()` on each
    /// in sequence, using the output of the previous transformer as input to
    /// the next (chain-fitting).
    pub fn fit(&mut self, data: &Array2<T>) -> Result<(), PipelineError> {
        if self.steps.is_empty() {
            self.is_fitted = true;
            return Ok(());
        }
        let mut current = data.clone();
        for step in &mut self.steps {
            current = step.fit_transform(&current)?;
        }
        self.is_fitted = true;
        Ok(())
    }

    /// Transform data through all fitted pipeline steps.
    pub fn transform(&self, data: &Array2<T>) -> Result<Array2<T>, PipelineError> {
        if self.steps.is_empty() {
            return Ok(data.clone());
        }
        if !self.is_fitted {
            return Err(PipelineError::NotFitted("MLPipelineGeneric".to_string()));
        }
        let mut current = data.clone();
        for step in &self.steps {
            current = step.transform(&current)?;
        }
        Ok(current)
    }

    /// Fit all transformers and then transform in one call.
    pub fn fit_transform(&mut self, data: &Array2<T>) -> Result<Array2<T>, PipelineError> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Fit all transformers to training data, transform it, then run the predictor.
    ///
    /// Returns predictions as a 1-D array of length equal to the number of samples.
    pub fn fit_predict(&mut self, data: &Array2<T>) -> Result<Array1<T>, PipelineError> {
        let transformed = self.fit_transform(data)?;
        match &self.predictor {
            Some(pred) => pred.predict(&transformed),
            None => Err(PipelineError::NoPredictorSet),
        }
    }

    /// Transform data and then predict (requires prior `fit`).
    pub fn predict(&self, data: &Array2<T>) -> Result<Array1<T>, PipelineError> {
        let transformed = self.transform(data)?;
        match &self.predictor {
            Some(pred) => pred.predict(&transformed),
            None => Err(PipelineError::NoPredictorSet),
        }
    }

    /// Predict on a batch of rows (semantic alias for [`Self::predict`]).
    pub fn predict_batch(&self, data: &Array2<T>) -> Result<Array1<T>, PipelineError> {
        self.predict(data)
    }

    /// Whether the pipeline has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Number of transformer steps.
    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    /// Whether a predictor has been attached.
    pub fn has_predictor(&self) -> bool {
        self.predictor.is_some()
    }
}

impl<T: Clone + fmt::Debug + Float + FromPrimitive + Send + Sync + 'static> Default
    for MLPipelineGeneric<T>
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],
        )
        .expect("shape is valid")
    }

    // ----- FeatureTransformer trait tests -----

    #[test]
    fn test_standard_scaler_fit_transform() {
        let data = make_data();
        let mut scaler = StandardScaler::new();
        scaler.fit(&data).expect("fit should succeed");
        let result = scaler.transform(&data).expect("transform should succeed");
        assert_eq!(result.shape(), data.shape());

        // After standard scaling the mean of each column should be ~0
        for col in 0..result.ncols() {
            let col_mean: f64 = result.column(col).sum() / result.nrows() as f64;
            assert!(
                col_mean.abs() < 1e-10,
                "column {col} mean not ~0: {col_mean}"
            );
        }
    }

    #[test]
    fn test_min_max_scaler_fit_transform() {
        let data = make_data();
        let mut scaler = MinMaxScaler::new();
        let result = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed");
        assert_eq!(result.shape(), data.shape());

        // All values should be in [0, 1]
        for &v in result.iter() {
            assert!(v >= 0.0 - 1e-10 && v <= 1.0 + 1e-10, "out of [0,1]: {v}");
        }
    }

    #[test]
    fn test_normalizer_transform_l2() {
        let data = Array2::from_shape_vec((2, 3), vec![3.0f64, 4.0, 0.0, 1.0, 0.0, 0.0])
            .expect("shape is valid");
        let mut norm = NormalizerTransform::l2();
        let result = norm
            .fit_transform(&data)
            .expect("fit_transform should succeed");
        // Row 0: [3,4,0] / 5 = [0.6, 0.8, 0]
        assert!((result[[0, 0]] - 0.6).abs() < 1e-10);
        assert!((result[[0, 1]] - 0.8).abs() < 1e-10);
        assert!(result[[0, 2]].abs() < 1e-10);
        // Row 1: [1,0,0] / 1 = [1, 0, 0]
        assert!((result[[1, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalizer_transform_l1() {
        let data = Array2::from_shape_vec((1, 3), vec![1.0f64, 2.0, 3.0]).expect("shape is valid");
        let mut norm = NormalizerTransform::l1();
        let result = norm
            .fit_transform(&data)
            .expect("fit_transform should succeed");
        // sum = 6, each divided by 6
        assert!((result[[0, 0]] - 1.0 / 6.0).abs() < 1e-10);
        assert!((result[[0, 1]] - 2.0 / 6.0).abs() < 1e-10);
        assert!((result[[0, 2]] - 3.0 / 6.0).abs() < 1e-10);
    }

    // ----- Pipeline builder tests -----

    #[test]
    fn test_pipeline_builder_chain() {
        let data = make_data();
        let mut pipeline = Pipeline::<f64>::new()
            .add_transformer(Box::new(StandardScaler::new()))
            .add_transformer(Box::new(MinMaxScaler::new()));

        let result = pipeline
            .fit_transform(&data)
            .expect("fit_transform should succeed");
        assert_eq!(result.shape(), data.shape());

        // After StandardScaler + MinMaxScaler all values in [0, 1]
        for &v in result.iter() {
            assert!(v >= 0.0 - 1e-10 && v <= 1.0 + 1e-10, "out of [0,1]: {v}");
        }
    }

    #[test]
    fn test_pipeline_fit_then_transform_independent() {
        let train_data = make_data();
        let test_data =
            Array2::from_shape_vec((2, 3), vec![100.0, 200.0, 300.0, -10.0, -20.0, -30.0])
                .expect("shape is valid");

        let mut pipeline = Pipeline::<f64>::new().add_transformer(Box::new(StandardScaler::new()));

        pipeline.fit(&train_data).expect("fit should succeed");
        let result = pipeline
            .transform(&test_data)
            .expect("transform should succeed");
        assert_eq!(result.shape(), test_data.shape());
    }

    #[test]
    fn test_pipeline_transform_without_fit_fails() {
        let data = make_data();
        let pipeline = Pipeline::<f64>::new().add_transformer(Box::new(StandardScaler::new()));

        let result = pipeline.transform(&data);
        assert!(result.is_err(), "transform before fit should fail");
    }

    #[test]
    fn test_empty_pipeline_is_identity() {
        let data = make_data();
        let mut pipeline = Pipeline::<f64>::new();
        let result = pipeline
            .fit_transform(&data)
            .expect("empty pipeline should succeed");
        assert_eq!(result, data);
    }

    // ----- MLPipelineGeneric tests -----

    #[test]
    fn test_ml_pipeline_generic_fit_transform() {
        let data = make_data();
        let mut mlp = MLPipelineGeneric::<f64>::new();
        mlp.add_transformer(Box::new(StandardScaler::new()));
        mlp.add_transformer(Box::new(MinMaxScaler::new()));

        let result = mlp
            .fit_transform(&data)
            .expect("fit_transform should succeed");
        assert_eq!(result.shape(), data.shape());
    }

    #[test]
    fn test_ml_pipeline_predict_without_predictor_fails() {
        let data = make_data();
        let mut mlp = MLPipelineGeneric::<f64>::new();
        mlp.add_transformer(Box::new(StandardScaler::new()));
        mlp.fit(&data).expect("fit should succeed");

        let result = mlp.predict(&data);
        assert!(
            result.is_err(),
            "predict without predictor should return error"
        );
        assert!(matches!(result, Err(PipelineError::NoPredictorSet)));
    }

    #[test]
    fn test_ml_pipeline_generic_default() {
        let mlp = MLPipelineGeneric::<f64>::default();
        assert_eq!(mlp.n_steps(), 0);
        assert!(!mlp.is_fitted());
        assert!(!mlp.has_predictor());
    }

    // ----- Error handling tests -----

    #[test]
    fn test_standard_scaler_zero_variance_column() {
        // A column with all equal values should have std = 0; transform should map to 0
        let data = Array2::from_shape_vec((3, 2), vec![5.0f64, 1.0, 5.0, 2.0, 5.0, 3.0])
            .expect("shape is valid");
        let mut scaler = StandardScaler::new();
        let result = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed");
        // Column 0 has zero variance → scaled to 0
        assert_eq!(result[[0, 0]], 0.0);
        assert_eq!(result[[1, 0]], 0.0);
        assert_eq!(result[[2, 0]], 0.0);
    }

    #[test]
    fn test_min_max_scaler_uniform_column() {
        // A column with all equal values: range = 0 → should output 0.0 (out_min)
        let data = Array2::from_shape_vec((3, 1), vec![7.0f64, 7.0, 7.0]).expect("shape is valid");
        let mut scaler = MinMaxScaler::new();
        let result = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed");
        for &v in result.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_transform_empty_matrix_fails() {
        let data: Array2<f64> = Array2::zeros((0, 3));
        let mut scaler = StandardScaler::new();
        let result = scaler.fit(&data);
        assert!(result.is_err(), "fitting empty data should fail");
    }

    // ----- Existing MLPipeline (streaming/DAG) still accessible -----

    #[test]
    fn test_existing_pipeline_api_still_works() {
        use pipeline_core::{DataBatch, PipelineConfig};

        let pipeline = MLPipeline::new("test".to_string(), PipelineConfig::default());
        let batch = DataBatch::new();
        // Empty pipeline with empty batch: batch size 0 should be within max_batchsize
        let result = pipeline.execute(batch);
        assert!(
            result.is_ok(),
            "empty pipeline on empty batch should succeed"
        );
    }
}
