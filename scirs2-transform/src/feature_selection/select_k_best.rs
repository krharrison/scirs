//! Select K best features by score
//!
//! A generic feature selector that accepts any scoring function and selects
//! the top K features by score. Works with chi-squared, F-test, mutual
//! information, or custom scoring functions.

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{Result, TransformError};

/// Scoring function type alias
///
/// Takes (X, y) and returns (scores, p_values).
/// Both arrays have length n_features.
pub type ScoreFunc = fn(&Array2<f64>, &Array1<f64>) -> Result<(Array1<f64>, Array1<f64>)>;

/// Select K best features using a scoring function
///
/// A generic feature selector that works with any scoring function that
/// returns (scores, p_values) for each feature.
///
/// # Examples
///
/// ```
/// use scirs2_transform::feature_selection::SelectKBest;
/// use scirs2_transform::feature_selection::f_test::f_classif;
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let x = Array2::from_shape_vec(
///     (6, 3),
///     vec![1.0, 5.0, 0.5, 2.0, 5.1, 0.6, 1.5, 5.0, 0.4,
///          8.0, 5.0, 0.5, 9.0, 5.1, 0.6, 8.5, 5.0, 0.4],
/// ).expect("should succeed");
/// let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
///
/// let mut selector = SelectKBest::new(f_classif, 1);
/// selector.fit(&x, &y).expect("should succeed");
/// let result = selector.transform(&x).expect("should succeed");
/// assert_eq!(result.shape()[1], 1);
/// ```
#[derive(Debug, Clone)]
pub struct SelectKBest {
    /// Scoring function
    score_func: ScoreFunc,
    /// Number of features to select
    k: usize,
    /// Selected feature indices (sorted)
    selected_features_: Option<Vec<usize>>,
    /// Scores for all features
    scores_: Option<Array1<f64>>,
    /// P-values for all features
    p_values_: Option<Array1<f64>>,
    /// Number of features at fit time
    n_features_in_: Option<usize>,
}

impl SelectKBest {
    /// Create a new SelectKBest
    ///
    /// # Arguments
    /// * `score_func` - Function computing (scores, p_values) per feature
    /// * `k` - Number of top features to select
    pub fn new(score_func: ScoreFunc, k: usize) -> Self {
        SelectKBest {
            score_func,
            k,
            selected_features_: None,
            scores_: None,
            p_values_: None,
            n_features_in_: None,
        }
    }

    /// Fit the selector
    ///
    /// # Arguments
    /// * `x` - Feature matrix, shape (n_samples, n_features)
    /// * `y` - Target vector, shape (n_samples,)
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_features = x.shape()[1];

        if self.k > n_features {
            return Err(TransformError::InvalidInput(format!(
                "k={} must be <= n_features={}",
                self.k, n_features
            )));
        }

        if self.k == 0 {
            return Err(TransformError::InvalidInput("k must be > 0".to_string()));
        }

        let (scores, p_values) = (self.score_func)(x, y)?;

        if scores.len() != n_features {
            return Err(TransformError::InvalidInput(format!(
                "Score function returned {} scores but expected {}",
                scores.len(),
                n_features
            )));
        }

        // Select top k features by score (descending)
        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut selected: Vec<usize> = indices.into_iter().take(self.k).collect();
        selected.sort(); // Sort by original index for consistent output

        self.scores_ = Some(scores);
        self.p_values_ = Some(p_values);
        self.selected_features_ = Some(selected);
        self.n_features_in_ = Some(n_features);

        Ok(())
    }

    /// Transform data by selecting features
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let selected = self.selected_features_.as_ref().ok_or_else(|| {
            TransformError::NotFitted("SelectKBest has not been fitted".to_string())
        })?;

        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let n_features_in = self.n_features_in_.unwrap_or(0);

        if n_features != n_features_in {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, expected {}",
                n_features, n_features_in
            )));
        }

        let mut transformed = Array2::zeros((n_samples, selected.len()));
        for (new_idx, &old_idx) in selected.iter().enumerate() {
            for i in 0..n_samples {
                transformed[[i, new_idx]] = x[[i, old_idx]];
            }
        }

        Ok(transformed)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array2<f64>> {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Get selected feature indices
    pub fn get_support(&self) -> Option<&Vec<usize>> {
        self.selected_features_.as_ref()
    }

    /// Get feature scores
    pub fn scores(&self) -> Option<&Array1<f64>> {
        self.scores_.as_ref()
    }

    /// Get p-values
    pub fn p_values(&self) -> Option<&Array1<f64>> {
        self.p_values_.as_ref()
    }

    /// Get a boolean support mask
    pub fn get_support_mask(&self) -> Option<Array1<bool>> {
        let n_features_in = self.n_features_in_?;
        let selected = self.selected_features_.as_ref()?;
        let mut mask = Array1::from_elem(n_features_in, false);
        for &idx in selected {
            mask[idx] = true;
        }
        Some(mask)
    }

    /// Get the number of selected features
    pub fn n_features_selected(&self) -> Option<usize> {
        self.selected_features_.as_ref().map(|s| s.len())
    }
}

/// Select features by p-value threshold
///
/// Instead of selecting a fixed number of features, select all features
/// whose p-value is below a given threshold.
#[derive(Debug, Clone)]
pub struct SelectByPValue {
    /// Scoring function
    score_func: ScoreFunc,
    /// P-value threshold
    alpha: f64,
    /// Selected feature indices
    selected_features_: Option<Vec<usize>>,
    /// Scores
    scores_: Option<Array1<f64>>,
    /// P-values
    p_values_: Option<Array1<f64>>,
    /// Number of features at fit time
    n_features_in_: Option<usize>,
}

impl SelectByPValue {
    /// Create a new SelectByPValue
    ///
    /// # Arguments
    /// * `score_func` - Scoring function returning (scores, p_values)
    /// * `alpha` - P-value threshold (features with p < alpha are selected)
    pub fn new(score_func: ScoreFunc, alpha: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(TransformError::InvalidInput(
                "alpha must be in (0, 1]".to_string(),
            ));
        }

        Ok(SelectByPValue {
            score_func,
            alpha,
            selected_features_: None,
            scores_: None,
            p_values_: None,
            n_features_in_: None,
        })
    }

    /// Fit the selector
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_features = x.shape()[1];
        let (scores, p_values) = (self.score_func)(x, y)?;

        let mut selected = Vec::new();
        for j in 0..n_features {
            if p_values[j] < self.alpha {
                selected.push(j);
            }
        }

        self.scores_ = Some(scores);
        self.p_values_ = Some(p_values);
        self.selected_features_ = Some(selected);
        self.n_features_in_ = Some(n_features);

        Ok(())
    }

    /// Transform data by selecting features
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let selected = self.selected_features_.as_ref().ok_or_else(|| {
            TransformError::NotFitted("SelectByPValue has not been fitted".to_string())
        })?;

        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let n_features_in = self.n_features_in_.unwrap_or(0);

        if n_features != n_features_in {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, expected {}",
                n_features, n_features_in
            )));
        }

        let mut transformed = Array2::zeros((n_samples, selected.len()));
        for (new_idx, &old_idx) in selected.iter().enumerate() {
            for i in 0..n_samples {
                transformed[[i, new_idx]] = x[[i, old_idx]];
            }
        }

        Ok(transformed)
    }

    /// Fit and transform
    pub fn fit_transform(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array2<f64>> {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Get selected feature indices
    pub fn get_support(&self) -> Option<&Vec<usize>> {
        self.selected_features_.as_ref()
    }

    /// Get scores
    pub fn scores(&self) -> Option<&Array1<f64>> {
        self.scores_.as_ref()
    }

    /// Get p-values
    pub fn p_values(&self) -> Option<&Array1<f64>> {
        self.p_values_.as_ref()
    }
}

/// Select features by percentile of scores
///
/// Selects features whose score is in the top `percentile` percent.
#[derive(Debug, Clone)]
pub struct SelectPercentile {
    /// Scoring function
    score_func: ScoreFunc,
    /// Percentile threshold (0-100)
    percentile: f64,
    /// Selected feature indices
    selected_features_: Option<Vec<usize>>,
    /// Scores
    scores_: Option<Array1<f64>>,
    /// P-values
    p_values_: Option<Array1<f64>>,
    /// Number of features at fit time
    n_features_in_: Option<usize>,
}

impl SelectPercentile {
    /// Create a new SelectPercentile
    ///
    /// # Arguments
    /// * `score_func` - Scoring function
    /// * `percentile` - Percentile threshold (0-100)
    pub fn new(score_func: ScoreFunc, percentile: f64) -> Result<Self> {
        if !(0.0..=100.0).contains(&percentile) {
            return Err(TransformError::InvalidInput(
                "percentile must be in [0, 100]".to_string(),
            ));
        }

        Ok(SelectPercentile {
            score_func,
            percentile,
            selected_features_: None,
            scores_: None,
            p_values_: None,
            n_features_in_: None,
        })
    }

    /// Fit the selector
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_features = x.shape()[1];
        let (scores, p_values) = (self.score_func)(x, y)?;

        // Sort scores to find threshold
        let mut sorted_scores: Vec<f64> = scores.iter().copied().collect();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let threshold_idx = ((1.0 - self.percentile / 100.0) * n_features as f64).floor() as usize;
        let threshold_idx = threshold_idx.min(n_features.saturating_sub(1));
        let threshold = sorted_scores[threshold_idx];

        let mut selected = Vec::new();
        for j in 0..n_features {
            if scores[j] >= threshold {
                selected.push(j);
            }
        }

        self.scores_ = Some(scores);
        self.p_values_ = Some(p_values);
        self.selected_features_ = Some(selected);
        self.n_features_in_ = Some(n_features);

        Ok(())
    }

    /// Transform data by selecting features
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let selected = self.selected_features_.as_ref().ok_or_else(|| {
            TransformError::NotFitted("SelectPercentile has not been fitted".to_string())
        })?;

        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let n_features_in = self.n_features_in_.unwrap_or(0);

        if n_features != n_features_in {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, expected {}",
                n_features, n_features_in
            )));
        }

        let mut transformed = Array2::zeros((n_samples, selected.len()));
        for (new_idx, &old_idx) in selected.iter().enumerate() {
            for i in 0..n_samples {
                transformed[[i, new_idx]] = x[[i, old_idx]];
            }
        }

        Ok(transformed)
    }

    /// Fit and transform
    pub fn fit_transform(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array2<f64>> {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Get selected feature indices
    pub fn get_support(&self) -> Option<&Vec<usize>> {
        self.selected_features_.as_ref()
    }

    /// Get scores
    pub fn scores(&self) -> Option<&Array1<f64>> {
        self.scores_.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_selection::f_test::f_classif;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_select_k_best_f_classif() {
        let x = Array::from_shape_vec(
            (6, 3),
            vec![
                1.0, 5.0, 0.5, 2.0, 5.1, 0.6, 1.5, 5.0, 0.4, 8.0, 5.0, 0.5, 9.0, 5.1, 0.6, 8.5,
                5.0, 0.4,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut selector = SelectKBest::new(f_classif, 1);
        let transformed = selector.fit_transform(&x, &y).expect("fit_transform");
        assert_eq!(transformed.shape(), &[6, 1]);

        let selected = selector.get_support().expect("support");
        assert_eq!(selected, &[0]); // Feature 0 most discriminative
    }

    #[test]
    fn test_select_k_best_support_mask() {
        let x = Array::from_shape_vec(
            (6, 3),
            vec![
                1.0, 5.0, 0.5, 2.0, 5.1, 0.6, 1.5, 5.0, 0.4, 8.0, 5.0, 0.5, 9.0, 5.1, 0.6, 8.5,
                5.0, 0.4,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut selector = SelectKBest::new(f_classif, 2);
        selector.fit(&x, &y).expect("fit");

        let mask = selector.get_support_mask().expect("mask");
        assert_eq!(mask.len(), 3);
        let n_true = mask.iter().filter(|&&v| v).count();
        assert_eq!(n_true, 2);
        assert_eq!(selector.n_features_selected().expect("n"), 2);
    }

    #[test]
    fn test_select_k_best_errors() {
        let x = Array::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("test data");
        let y = Array::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        // k > n_features
        let mut selector = SelectKBest::new(f_classif, 5);
        assert!(selector.fit(&x, &y).is_err());

        // k = 0
        let mut selector = SelectKBest::new(f_classif, 0);
        assert!(selector.fit(&x, &y).is_err());
    }

    #[test]
    fn test_select_k_best_not_fitted() {
        let x = Array::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("test data");
        let selector = SelectKBest::new(f_classif, 1);
        assert!(selector.transform(&x).is_err());
    }

    #[test]
    fn test_select_by_p_value() {
        let x = Array::from_shape_vec(
            (6, 3),
            vec![
                1.0, 5.0, 0.5, 2.0, 5.1, 0.6, 1.5, 5.0, 0.4, 8.0, 5.0, 0.5, 9.0, 5.1, 0.6, 8.5,
                5.0, 0.4,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut selector = SelectByPValue::new(f_classif, 0.05).expect("new");
        selector.fit(&x, &y).expect("fit");

        let selected = selector.get_support().expect("support");
        // Feature 0 should pass the significance test
        assert!(selected.contains(&0));
    }

    #[test]
    fn test_select_by_p_value_errors() {
        assert!(SelectByPValue::new(f_classif, 0.0).is_err());
        assert!(SelectByPValue::new(f_classif, 1.5).is_err());
    }

    #[test]
    fn test_select_percentile() {
        let x = Array::from_shape_vec(
            (6, 4),
            vec![
                1.0, 5.0, 0.5, 0.3, 2.0, 5.1, 0.6, 0.2, 1.5, 5.0, 0.4, 0.4, 8.0, 5.0, 0.5, 0.3,
                9.0, 5.1, 0.6, 0.2, 8.5, 5.0, 0.4, 0.4,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut selector = SelectPercentile::new(f_classif, 50.0).expect("new");
        let transformed = selector.fit_transform(&x, &y).expect("fit_transform");

        // Should select about half the features
        assert!(transformed.shape()[1] >= 1);
        assert!(transformed.shape()[1] <= 4);
    }

    #[test]
    fn test_select_percentile_errors() {
        assert!(SelectPercentile::new(f_classif, -1.0).is_err());
        assert!(SelectPercentile::new(f_classif, 101.0).is_err());
    }
}
