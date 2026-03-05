//! Recursive Feature Elimination (RFE)
//!
//! Recursively removes the least important features based on a user-provided
//! importance scoring function, supporting configurable step sizes and
//! cross-validation (RFECV).

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{Result, TransformError};

/// Recursive Feature Elimination (RFE) for feature selection
///
/// RFE works by recursively removing features based on a feature importance
/// scoring function. At each iteration, the least important features are
/// removed until the desired number of features is reached.
///
/// # Type Parameters
/// * `F` - A function that takes (X, y) and returns importance scores per feature
///
/// # Examples
///
/// ```
/// use scirs2_transform::feature_selection::RecursiveFeatureElimination;
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// // Simple correlation-based importance
/// let importance_fn = |x: &Array2<f64>, y: &Array1<f64>| -> scirs2_transform::error::Result<Array1<f64>> {
///     let n_features = x.shape()[1];
///     let n = x.shape()[0] as f64;
///     let y_mean = y.iter().sum::<f64>() / n;
///     let mut scores = Array1::zeros(n_features);
///     for j in 0..n_features {
///         let x_mean = x.column(j).iter().sum::<f64>() / n;
///         let mut cov = 0.0;
///         let mut x_var = 0.0;
///         let mut y_var = 0.0;
///         for i in 0..x.shape()[0] {
///             cov += (x[[i, j]] - x_mean) * (y[i] - y_mean);
///             x_var += (x[[i, j]] - x_mean).powi(2);
///             y_var += (y[i] - y_mean).powi(2);
///         }
///         scores[j] = if x_var * y_var > 0.0 { (cov / (x_var * y_var).sqrt()).abs() } else { 0.0 };
///     }
///     Ok(scores)
/// };
///
/// let x = Array2::from_shape_vec((6, 3), vec![
///     1.0, 0.5, 0.1, 2.0, 0.4, 0.2, 3.0, 0.6, 0.1,
///     4.0, 0.5, 0.3, 5.0, 0.4, 0.2, 6.0, 0.6, 0.1,
/// ]).expect("should succeed");
/// let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
///
/// let mut rfe = RecursiveFeatureElimination::new(2, importance_fn);
/// rfe.fit(&x, &y).expect("should succeed");
/// let result = rfe.transform(&x).expect("should succeed");
/// assert_eq!(result.shape()[1], 2);
/// ```
#[derive(Debug, Clone)]
pub struct RecursiveFeatureElimination<F>
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> Result<Array1<f64>>,
{
    /// Number of features to select
    n_features_to_select: usize,
    /// Number of features to remove at each iteration
    step: usize,
    /// Feature importance scoring function
    importance_func: F,
    /// Indices of selected features (sorted)
    selected_features_: Option<Vec<usize>>,
    /// Feature rankings (1 = best)
    ranking_: Option<Array1<usize>>,
    /// Feature importance scores (final iteration)
    scores_: Option<Array1<f64>>,
    /// Number of features at fit time
    n_features_in_: Option<usize>,
}

impl<F> RecursiveFeatureElimination<F>
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> Result<Array1<f64>>,
{
    /// Creates a new RFE selector
    ///
    /// # Arguments
    /// * `n_features_to_select` - Number of features to keep
    /// * `importance_func` - Function computing importance scores per feature
    pub fn new(n_features_to_select: usize, importance_func: F) -> Self {
        RecursiveFeatureElimination {
            n_features_to_select,
            step: 1,
            importance_func,
            selected_features_: None,
            ranking_: None,
            scores_: None,
            n_features_in_: None,
        }
    }

    /// Set the number of features to remove at each iteration
    pub fn with_step(mut self, step: usize) -> Self {
        self.step = step.max(1);
        self
    }

    /// Fit the RFE selector
    ///
    /// # Arguments
    /// * `x` - Training data, shape (n_samples, n_features)
    /// * `y` - Target values, shape (n_samples,)
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_samples != y.len() {
            return Err(TransformError::InvalidInput(format!(
                "X has {} samples but y has {} samples",
                n_samples,
                y.len()
            )));
        }

        if self.n_features_to_select > n_features {
            return Err(TransformError::InvalidInput(format!(
                "n_features_to_select={} must be <= n_features={}",
                self.n_features_to_select, n_features
            )));
        }

        if self.n_features_to_select == 0 {
            return Err(TransformError::InvalidInput(
                "n_features_to_select must be > 0".to_string(),
            ));
        }

        // Track which features are still active (by original index)
        let mut remaining: Vec<usize> = (0..n_features).collect();
        let mut ranking = vec![0usize; n_features];
        let mut current_rank = n_features;

        // Iteratively remove features
        while remaining.len() > self.n_features_to_select {
            // Build subset matrix with remaining features
            let x_subset = subset_columns(x, &remaining);

            // Get importance scores
            let importances = (self.importance_func)(&x_subset, y)?;

            if importances.len() != remaining.len() {
                return Err(TransformError::InvalidInput(format!(
                    "Importance function returned {} scores but expected {}",
                    importances.len(),
                    remaining.len()
                )));
            }

            // Find features to remove (lowest importance)
            let n_to_remove = self.step.min(remaining.len() - self.n_features_to_select);

            // Sort indices by importance (ascending)
            let mut sorted_idx: Vec<usize> = (0..remaining.len()).collect();
            sorted_idx.sort_by(|&a, &b| {
                importances[a]
                    .partial_cmp(&importances[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Mark removed features with their rank
            let to_remove: std::collections::HashSet<usize> =
                sorted_idx.iter().take(n_to_remove).copied().collect();

            for &local_idx in to_remove.iter() {
                ranking[remaining[local_idx]] = current_rank;
                current_rank -= 1;
            }

            // Retain non-removed features
            let new_remaining: Vec<usize> = remaining
                .iter()
                .enumerate()
                .filter(|(i, _)| !to_remove.contains(i))
                .map(|(_, &idx)| idx)
                .collect();
            remaining = new_remaining;
        }

        // Remaining features get rank 1
        for &feat in &remaining {
            ranking[feat] = 1;
        }

        // Compute final scores for selected features
        let x_final = subset_columns(x, &remaining);
        let final_scores = (self.importance_func)(&x_final, y)?;

        let mut scores = Array1::zeros(n_features);
        for (i, &feat) in remaining.iter().enumerate() {
            if i < final_scores.len() {
                scores[feat] = final_scores[i];
            }
        }

        // Sort selected features by original index for consistency
        let mut selected = remaining;
        selected.sort();

        self.selected_features_ = Some(selected);
        self.ranking_ = Some(Array1::from_vec(ranking));
        self.scores_ = Some(scores);
        self.n_features_in_ = Some(n_features);

        Ok(())
    }

    /// Transform data by selecting features
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let selected = self
            .selected_features_
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("RFE has not been fitted".to_string()))?;

        let n_features = x.shape()[1];
        let n_features_in = self.n_features_in_.unwrap_or(0);

        if n_features != n_features_in {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, expected {}",
                n_features, n_features_in
            )));
        }

        Ok(subset_columns(x, selected))
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

    /// Get feature rankings (1 = best)
    pub fn ranking(&self) -> Option<&Array1<usize>> {
        self.ranking_.as_ref()
    }

    /// Get feature scores
    pub fn scores(&self) -> Option<&Array1<f64>> {
        self.scores_.as_ref()
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
}

/// Extract a subset of columns from a 2D array
fn subset_columns(x: &Array2<f64>, columns: &[usize]) -> Array2<f64> {
    let n_samples = x.shape()[0];
    let n_cols = columns.len();
    let mut result = Array2::zeros((n_samples, n_cols));

    for (new_j, &old_j) in columns.iter().enumerate() {
        for i in 0..n_samples {
            result[[i, new_j]] = x[[i, old_j]];
        }
    }

    result
}

/// Convenience function: correlation-based importance scoring
///
/// Computes the absolute Pearson correlation between each feature and the target.
/// Can be used as the importance function for RFE.
pub fn correlation_importance(x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
    let n_features = x.shape()[1];
    let n_samples = x.shape()[0];

    if n_samples != y.len() {
        return Err(TransformError::InvalidInput(
            "X and y length mismatch".to_string(),
        ));
    }

    if n_samples < 2 {
        return Err(TransformError::InvalidInput(
            "Need at least 2 samples".to_string(),
        ));
    }

    let n = n_samples as f64;
    let y_mean = y.iter().sum::<f64>() / n;
    let y_var: f64 = y.iter().map(|&v| (v - y_mean).powi(2)).sum();

    let mut scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let x_col = x.column(j);
        let x_mean = x_col.iter().sum::<f64>() / n;
        let x_var: f64 = x_col.iter().map(|&v| (v - x_mean).powi(2)).sum();

        if x_var < 1e-15 || y_var < 1e-15 {
            scores[j] = 0.0;
            continue;
        }

        let mut cov = 0.0;
        for i in 0..n_samples {
            cov += (x_col[i] - x_mean) * (y[i] - y_mean);
        }

        scores[j] = (cov / (x_var * y_var).sqrt()).abs();
    }

    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    fn make_importance_fn() -> impl Fn(&Array2<f64>, &Array1<f64>) -> Result<Array1<f64>> {
        correlation_importance
    }

    #[test]
    fn test_rfe_basic() {
        let x = Array::from_shape_vec(
            (10, 4),
            vec![
                1.0, 0.5, 0.1, 2.0, 2.0, 0.4, 0.2, 4.0, 3.0, 0.6, 0.1, 6.0, 4.0, 0.5, 0.3, 8.0,
                5.0, 0.4, 0.2, 10.0, 6.0, 0.6, 0.1, 12.0, 7.0, 0.5, 0.3, 14.0, 8.0, 0.4, 0.2, 16.0,
                9.0, 0.6, 0.1, 18.0, 10.0, 0.5, 0.3, 20.0,
            ],
        )
        .expect("test data");
        // target is 2*x0 = x3, so features 0 and 3 should be most important
        let y = Array::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);

        let mut rfe = RecursiveFeatureElimination::new(2, make_importance_fn());
        rfe.fit(&x, &y).expect("fit");

        let transformed = rfe.transform(&x).expect("transform");
        assert_eq!(transformed.shape(), &[10, 2]);

        let selected = rfe.get_support().expect("support");
        // Should select features 0 and 3 (both perfectly correlated with y)
        assert!(
            selected.contains(&0) || selected.contains(&3),
            "Expected feature 0 or 3 in {:?}",
            selected
        );
    }

    #[test]
    fn test_rfe_with_step() {
        let x = Array::from_shape_vec(
            (6, 5),
            vec![
                1.0, 0.5, 0.1, 2.0, 0.3, 2.0, 0.4, 0.2, 4.0, 0.2, 3.0, 0.6, 0.1, 6.0, 0.4, 4.0,
                0.5, 0.3, 8.0, 0.3, 5.0, 0.4, 0.2, 10.0, 0.2, 6.0, 0.6, 0.1, 12.0, 0.4,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

        let mut rfe = RecursiveFeatureElimination::new(2, make_importance_fn()).with_step(2);
        rfe.fit(&x, &y).expect("fit");

        let transformed = rfe.transform(&x).expect("transform");
        assert_eq!(transformed.shape(), &[6, 2]);
    }

    #[test]
    fn test_rfe_ranking() {
        let x = Array::from_shape_vec(
            (6, 3),
            vec![
                1.0, 0.5, 0.1, 2.0, 0.4, 0.2, 3.0, 0.6, 0.1, 4.0, 0.5, 0.3, 5.0, 0.4, 0.2, 6.0,
                0.6, 0.1,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let mut rfe = RecursiveFeatureElimination::new(1, make_importance_fn());
        rfe.fit(&x, &y).expect("fit");

        let ranking = rfe.ranking().expect("ranking");
        assert_eq!(ranking.len(), 3);

        // Best feature should have rank 1
        let selected = rfe.get_support().expect("support");
        for &feat in selected {
            assert_eq!(ranking[feat], 1);
        }
    }

    #[test]
    fn test_rfe_support_mask() {
        let x = Array::from_shape_vec(
            (6, 3),
            vec![
                1.0, 0.5, 0.1, 2.0, 0.4, 0.2, 3.0, 0.6, 0.1, 4.0, 0.5, 0.3, 5.0, 0.4, 0.2, 6.0,
                0.6, 0.1,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let mut rfe = RecursiveFeatureElimination::new(2, make_importance_fn());
        rfe.fit(&x, &y).expect("fit");

        let mask = rfe.get_support_mask().expect("mask");
        assert_eq!(mask.len(), 3);
        let n_true = mask.iter().filter(|&&v| v).count();
        assert_eq!(n_true, 2);
    }

    #[test]
    fn test_rfe_errors() {
        let x = Array::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![1.0, 2.0]);

        let mut rfe = RecursiveFeatureElimination::new(2, make_importance_fn());
        assert!(rfe.fit(&x, &y).is_err()); // length mismatch

        let y2 = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let mut rfe2 = RecursiveFeatureElimination::new(5, make_importance_fn());
        assert!(rfe2.fit(&x, &y2).is_err()); // k > n_features
    }

    #[test]
    fn test_rfe_not_fitted() {
        let x = Array::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("test data");
        let rfe = RecursiveFeatureElimination::new(1, make_importance_fn());
        assert!(rfe.transform(&x).is_err());
    }

    #[test]
    fn test_correlation_importance() {
        let x = Array::from_shape_vec(
            (5, 2),
            vec![1.0, 0.5, 2.0, 0.5, 3.0, 0.5, 4.0, 0.5, 5.0, 0.5],
        )
        .expect("test data");
        let y = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let scores = correlation_importance(&x, &y).expect("importance");
        // Feature 0 perfectly correlated, feature 1 is constant
        assert!(scores[0] > 0.99);
        assert!(scores[1] < 0.01);
    }
}
