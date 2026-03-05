//! Variance threshold feature selection
//!
//! Removes features with variance below a given threshold. Features that are
//! mostly constant carry little discriminative information and can be safely removed.

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};

use crate::error::{Result, TransformError};

/// VarianceThreshold for removing low-variance features
///
/// Features with variance below the threshold are removed. This is useful for
/// removing features that are mostly constant and don't provide much information.
///
/// # Examples
///
/// ```
/// use scirs2_transform::feature_selection::VarianceThreshold;
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::from_shape_vec(
///     (3, 4),
///     vec![1.0, 1.0, 5.0, 1.0,
///          1.0, 2.0, 5.0, 3.0,
///          1.0, 3.0, 5.0, 5.0],
/// ).expect("should succeed");
///
/// let mut selector = VarianceThreshold::new(0.0).expect("should succeed");
/// let result = selector.fit_transform(&data).expect("should succeed");
/// // Constant features (columns 0 and 2) are removed
/// assert_eq!(result.shape(), &[3, 2]);
/// ```
#[derive(Debug, Clone)]
pub struct VarianceThreshold {
    /// Variance threshold for feature selection
    threshold: f64,
    /// Variances computed for each feature (learned during fit)
    variances_: Option<Array1<f64>>,
    /// Indices of selected features
    selected_features_: Option<Vec<usize>>,
    /// Number of features seen during fit
    n_features_in_: Option<usize>,
}

impl VarianceThreshold {
    /// Creates a new VarianceThreshold selector
    ///
    /// # Arguments
    /// * `threshold` - Features with variance below this threshold are removed
    ///
    /// # Errors
    /// Returns error if threshold is negative
    pub fn new(threshold: f64) -> Result<Self> {
        if threshold < 0.0 {
            return Err(TransformError::InvalidInput(
                "Threshold must be non-negative".to_string(),
            ));
        }

        Ok(VarianceThreshold {
            threshold,
            variances_: None,
            selected_features_: None,
            n_features_in_: None,
        })
    }

    /// Creates a VarianceThreshold with default threshold (0.0)
    ///
    /// This will only remove features that are completely constant.
    pub fn with_defaults() -> Self {
        VarianceThreshold {
            threshold: 0.0,
            variances_: None,
            selected_features_: None,
            n_features_in_: None,
        }
    }

    /// Returns the variance threshold
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Fits the VarianceThreshold to the input data
    ///
    /// Computes the variance for each feature and determines which features
    /// exceed the threshold.
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        if n_samples < 2 {
            return Err(TransformError::InvalidInput(
                "At least 2 samples required to compute variance".to_string(),
            ));
        }

        // Compute variance for each feature using Welford's online algorithm
        // for numerical stability
        let mut variances = Array1::zeros(n_features);
        let mut selected_features = Vec::new();

        for j in 0..n_features {
            let mut mean = 0.0_f64;
            let mut m2 = 0.0_f64;

            for i in 0..n_samples {
                let val: f64 = NumCast::from(x[[i, j]]).unwrap_or(0.0);
                let delta = val - mean;
                mean += delta / (i as f64 + 1.0);
                let delta2 = val - mean;
                m2 += delta * delta2;
            }

            // Population variance (consistent with sklearn)
            let variance = m2 / n_samples as f64;
            variances[j] = variance;

            if variance > self.threshold {
                selected_features.push(j);
            }
        }

        self.variances_ = Some(variances);
        self.selected_features_ = Some(selected_features);
        self.n_features_in_ = Some(n_features);

        Ok(())
    }

    /// Transforms the input data by removing low-variance features
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        let selected_features = self.selected_features_.as_ref().ok_or_else(|| {
            TransformError::NotFitted("VarianceThreshold has not been fitted".to_string())
        })?;

        let n_features_in = self.n_features_in_.unwrap_or(0);
        if n_features != n_features_in {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but VarianceThreshold was fitted with {} features",
                n_features, n_features_in
            )));
        }

        let n_selected = selected_features.len();
        let mut transformed = Array2::zeros((n_samples, n_selected));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            for i in 0..n_samples {
                transformed[[i, new_idx]] = NumCast::from(x[[i, old_idx]]).unwrap_or(0.0);
            }
        }

        Ok(transformed)
    }

    /// Fits and transforms in one step
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the variances computed for each feature
    pub fn variances(&self) -> Option<&Array1<f64>> {
        self.variances_.as_ref()
    }

    /// Returns the indices of selected features
    pub fn get_support(&self) -> Option<&Vec<usize>> {
        self.selected_features_.as_ref()
    }

    /// Returns a boolean mask indicating which features are selected
    pub fn get_support_mask(&self) -> Option<Array1<bool>> {
        let n_features_in = self.n_features_in_?;
        let selected = self.selected_features_.as_ref()?;
        let mut mask = Array1::from_elem(n_features_in, false);

        for &idx in selected {
            mask[idx] = true;
        }

        Some(mask)
    }

    /// Returns the number of selected features
    pub fn n_features_selected(&self) -> Option<usize> {
        self.selected_features_.as_ref().map(|s| s.len())
    }

    /// Inverse transform is not applicable for feature selection
    pub fn inverse_transform(&self, _x: &Array2<f64>) -> Result<Array2<f64>> {
        Err(TransformError::TransformationError(
            "inverse_transform is not supported for feature selection".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_variance_threshold_basic() {
        let data = Array::from_shape_vec(
            (3, 4),
            vec![1.0, 1.0, 5.0, 1.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 5.0, 5.0],
        )
        .expect("test data");

        let mut selector = VarianceThreshold::with_defaults();
        let transformed = selector.fit_transform(&data).expect("fit_transform");

        assert_eq!(transformed.shape(), &[3, 2]);
        let selected = selector.get_support().expect("get_support");
        assert_eq!(selected, &[1, 3]);

        assert_abs_diff_eq!(transformed[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[1, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 0]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_variance_threshold_custom() {
        let data = Array::from_shape_vec(
            (4, 3),
            vec![1.0, 1.0, 1.0, 2.0, 1.1, 2.0, 3.0, 1.0, 3.0, 4.0, 1.1, 4.0],
        )
        .expect("test data");

        let mut selector = VarianceThreshold::new(0.1).expect("new");
        let transformed = selector.fit_transform(&data).expect("fit_transform");

        assert_eq!(transformed.shape(), &[4, 2]);
        let selected = selector.get_support().expect("get_support");
        assert_eq!(selected, &[0, 2]);
    }

    #[test]
    fn test_variance_threshold_support_mask() {
        let data = Array::from_shape_vec(
            (3, 4),
            vec![1.0, 1.0, 5.0, 1.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 5.0, 5.0],
        )
        .expect("test data");

        let mut selector = VarianceThreshold::with_defaults();
        selector.fit(&data).expect("fit");

        let mask = selector.get_support_mask().expect("mask");
        assert_eq!(mask.len(), 4);
        assert!(!mask[0]);
        assert!(mask[1]);
        assert!(!mask[2]);
        assert!(mask[3]);
        assert_eq!(selector.n_features_selected().expect("n_selected"), 2);
    }

    #[test]
    fn test_variance_threshold_all_constant() {
        let data = Array::from_shape_vec((3, 2), vec![5.0, 10.0, 5.0, 10.0, 5.0, 10.0])
            .expect("test data");

        let mut selector = VarianceThreshold::with_defaults();
        let transformed = selector.fit_transform(&data).expect("fit_transform");

        assert_eq!(transformed.shape(), &[3, 0]);
        assert_eq!(selector.n_features_selected().expect("n_selected"), 0);
    }

    #[test]
    fn test_variance_threshold_errors() {
        assert!(VarianceThreshold::new(-0.1).is_err());

        let small_data = Array::from_shape_vec((1, 2), vec![1.0, 2.0]).expect("test data");
        let mut selector = VarianceThreshold::with_defaults();
        assert!(selector.fit(&small_data).is_err());

        let data =
            Array::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test data");
        let selector_unfitted = VarianceThreshold::with_defaults();
        assert!(selector_unfitted.transform(&data).is_err());
    }

    #[test]
    fn test_variance_calculation_welford() {
        let data = Array::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).expect("test data");

        let mut selector = VarianceThreshold::with_defaults();
        selector.fit(&data).expect("fit");

        let variances = selector.variances().expect("variances");
        let expected_variance = 2.0 / 3.0;
        assert_abs_diff_eq!(variances[0], expected_variance, epsilon = 1e-10);
    }

    #[test]
    fn test_feature_mismatch() {
        let train =
            Array::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .expect("test data");
        let test = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("test data");

        let mut selector = VarianceThreshold::with_defaults();
        selector.fit(&train).expect("fit");
        assert!(selector.transform(&test).is_err());
    }
}
