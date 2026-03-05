//! Core traits for the high-level ML Pipeline API.
//!
//! This module defines the two foundational traits used throughout the
//! SciRS2 ML pipeline framework:
//!
//! - [`FeatureTransformer`]: stateful transformations (fit + transform)
//! - [`ModelPredictor`]: inference interface (predict / predict_batch)
//!
//! Downstream crates implement these traits to plug into [`super::Pipeline`]
//! and [`super::MLPipelineGeneric`].

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::fmt;

use super::builder::PipelineError;

/// Trait for stateful feature transformations.
///
/// A `FeatureTransformer` learns statistics from training data during [`fit`]
/// and applies those statistics during [`transform`]. Implementations must
/// handle zero-variance columns, empty inputs, and other degenerate cases
/// gracefully by returning [`PipelineError`] instead of panicking.
///
/// [`fit`]: FeatureTransformer::fit
/// [`transform`]: FeatureTransformer::transform
///
/// # Implementation Contract
///
/// 1. `fit` **must** be called before `transform`.
/// 2. `fit_transform` is equivalent to calling `fit` followed by `transform`
///    on the same data; the default implementation enforces this.
/// 3. After `fit`, subsequent `transform` calls with different data of the
///    same column count must succeed.
/// 4. No method may use `unwrap()` or `expect()` in production paths.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::{FeatureTransformer, StandardScaler, PipelineError};
/// use ndarray::Array2;
///
/// let data = Array2::from_shape_vec((3, 2), vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("should succeed");
/// let mut scaler = StandardScaler::new();
/// scaler.fit(&data).expect("should succeed");
/// let transformed = scaler.transform(&data).expect("should succeed");
/// assert_eq!(transformed.shape(), &[3, 2]);
/// # }
/// ```
pub trait FeatureTransformer<T: Clone + fmt::Debug + Float + FromPrimitive + 'static>:
    Send + Sync
{
    /// Fit the transformer to training data.
    ///
    /// Learns any statistics required for later calls to [`transform`].
    /// Returns an error if the data is invalid (empty, NaN, etc.).
    ///
    /// [`transform`]: FeatureTransformer::transform
    fn fit(&mut self, data: &Array2<T>) -> Result<(), PipelineError>;

    /// Apply the fitted transformation to data.
    ///
    /// Must be called after [`fit`]. Returns an error if the transformer has
    /// not been fitted or if the input shape is incompatible.
    ///
    /// [`fit`]: FeatureTransformer::fit
    fn transform(&self, data: &Array2<T>) -> Result<Array2<T>, PipelineError>;

    /// Fit and transform in a single call.
    ///
    /// The default implementation calls `fit` then `transform`. Override for
    /// efficiency when fitting and transforming can be done in one pass.
    fn fit_transform(&mut self, data: &Array2<T>) -> Result<Array2<T>, PipelineError> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Human-readable name for this transformer (used in error messages).
    fn name(&self) -> &str;

    /// Whether the transformer has been fitted to data.
    fn is_fitted(&self) -> bool;

    /// Number of features expected by this transformer after fitting.
    ///
    /// Returns `None` if not yet fitted.
    fn n_features_in(&self) -> Option<usize>;
}

/// Trait for ML model predictors.
///
/// A `ModelPredictor` encapsulates the inference step of an ML pipeline.
/// It receives a feature matrix (already transformed by the upstream
/// [`FeatureTransformer`] steps) and returns a 1-D prediction array.
///
/// # Implementation Contract
///
/// 1. `predict` must return a 1-D array of length `data.nrows()`.
/// 2. `predict_batch` is semantically identical to `predict`; it exists as a
///    separate method to allow implementations to apply different optimisations
///    for batched inference (e.g., running inference in chunks).
/// 3. No panics — all errors must be returned as [`PipelineError`].
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::{ModelPredictor, PipelineError};
/// use ndarray::{Array1, Array2};
///
/// /// A trivial predictor that returns the mean of each row.
/// struct RowMeanPredictor;
///
/// impl ModelPredictor<f64> for RowMeanPredictor {
///     fn predict(&self, data: &Array2<f64>) -> Result<Array1<f64>, PipelineError> {
///         let n = data.nrows();
///         if n == 0 {
///             return Err(PipelineError::EmptyInput("predict".to_string()));
///         }
///         let preds: Vec<f64> = (0..n)
///             .map(|i| data.row(i).sum() / data.ncols() as f64)
///             .collect();
///         Ok(Array1::from_vec(preds))
///     }
///
///     fn predict_batch(&self, data: &Array2<f64>) -> Result<Array1<f64>, PipelineError> {
///         self.predict(data)
///     }
///
///     fn name(&self) -> &str { "RowMeanPredictor" }
/// }
///
/// let data = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("should succeed");
/// let predictor = RowMeanPredictor;
/// let preds = predictor.predict(&data).expect("should succeed");
/// assert_eq!(preds.len(), 2);
/// assert!((preds[0] - 2.0).abs() < 1e-10);
/// assert!((preds[1] - 5.0).abs() < 1e-10);
/// # }
/// ```
pub trait ModelPredictor<T: Clone + fmt::Debug + Float + FromPrimitive + 'static>:
    Send + Sync
{
    /// Generate predictions for all rows in `data`.
    ///
    /// Returns a 1-D array of length `data.nrows()`.
    fn predict(&self, data: &Array2<T>) -> Result<Array1<T>, PipelineError>;

    /// Generate predictions for a batch, potentially using batch-optimised logic.
    ///
    /// Default implementation delegates to [`predict`].
    ///
    /// [`predict`]: ModelPredictor::predict
    fn predict_batch(&self, data: &Array2<T>) -> Result<Array1<T>, PipelineError> {
        self.predict(data)
    }

    /// Human-readable name for this predictor.
    fn name(&self) -> &str;
}
