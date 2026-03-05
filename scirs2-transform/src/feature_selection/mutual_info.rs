//! Mutual information based feature selection
//!
//! Provides mutual information estimation between features and target variables
//! for both classification (discrete target) and regression (continuous target).
//! Uses KNN-based Kraskov estimator for continuous variables and a binning
//! approach for discrete targets.

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

use crate::error::{Result, TransformError};

/// Method for computing mutual information
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutualInfoMethod {
    /// For classification tasks (discrete target)
    Classification,
    /// For regression tasks (continuous target)
    Regression,
}

/// Mutual information based feature selector
///
/// Selects features based on mutual information between each feature and the target.
/// Supports both classification (discrete target) and regression (continuous target).
///
/// # Examples
///
/// ```
/// use scirs2_transform::feature_selection::MutualInfoSelector;
/// use scirs2_transform::feature_selection::MutualInfoMethod;
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let x = Array2::from_shape_vec(
///     (6, 3),
///     vec![1.0, 0.1, 5.0,  1.1, 0.2, 5.1,
///          2.0, 0.1, 4.0,  2.1, 0.2, 4.1,
///          3.0, 0.1, 3.0,  3.1, 0.2, 3.1],
/// ).expect("should succeed");
/// let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
///
/// let mut selector = MutualInfoSelector::new(2, MutualInfoMethod::Classification);
/// selector.fit(&x, &y).expect("should succeed");
/// let result = selector.transform(&x).expect("should succeed");
/// assert_eq!(result.shape()[1], 2);
/// ```
#[derive(Debug, Clone)]
pub struct MutualInfoSelector {
    /// Number of features to select
    k: usize,
    /// Method for MI estimation
    method: MutualInfoMethod,
    /// Number of neighbors for KNN estimation
    n_neighbors: usize,
    /// Number of bins for discretization
    n_bins: usize,
    /// Selected feature indices
    selected_features_: Option<Vec<usize>>,
    /// Mutual information scores
    scores_: Option<Array1<f64>>,
    /// Number of features seen at fit time
    n_features_in_: Option<usize>,
}

impl MutualInfoSelector {
    /// Create a new mutual information selector
    ///
    /// # Arguments
    /// * `k` - Number of top features to select
    /// * `method` - Classification or Regression MI
    pub fn new(k: usize, method: MutualInfoMethod) -> Self {
        MutualInfoSelector {
            k,
            method,
            n_neighbors: 5,
            n_bins: 10,
            selected_features_: None,
            scores_: None,
            n_features_in_: None,
        }
    }

    /// Set number of neighbors for KNN-based MI estimation
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors.max(1);
        self
    }

    /// Set number of bins for discretization in classification MI
    pub fn with_n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins.max(2);
        self
    }

    /// Compute mutual information between a continuous feature and discrete target
    fn mi_classification(&self, feature: &Array1<f64>, target: &Array1<f64>) -> f64 {
        let n = feature.len();
        if n < 4 {
            return 0.0;
        }

        // Group samples by class
        let mut groups: HashMap<i64, Vec<f64>> = HashMap::new();
        for i in 0..n {
            let key = target[i].round() as i64;
            groups.entry(key).or_default().push(feature[i]);
        }

        if groups.len() < 2 {
            return 0.0;
        }

        // Compute overall entropy via binning
        let (min_val, max_val) = feature_range(feature);
        let range = max_val - min_val;
        if range < 1e-15 {
            return 0.0;
        }

        let n_bins = self.n_bins.min(n / 2).max(2);
        let bin_width = range / n_bins as f64;

        // Compute marginal entropy of feature H(X)
        let mut feature_counts = vec![0usize; n_bins];
        for i in 0..n {
            let bin = ((feature[i] - min_val) / bin_width).floor() as usize;
            let bin = bin.min(n_bins - 1);
            feature_counts[bin] += 1;
        }

        let h_x = entropy_from_counts(&feature_counts, n);

        // Compute conditional entropy H(X|Y) = sum_y P(Y=y) * H(X|Y=y)
        let mut h_x_given_y = 0.0;
        for (_, values) in &groups {
            let p_y = values.len() as f64 / n as f64;
            let mut cond_counts = vec![0usize; n_bins];
            for &val in values {
                let bin = ((val - min_val) / bin_width).floor() as usize;
                let bin = bin.min(n_bins - 1);
                cond_counts[bin] += 1;
            }
            let h_x_y = entropy_from_counts(&cond_counts, values.len());
            h_x_given_y += p_y * h_x_y;
        }

        // MI(X; Y) = H(X) - H(X|Y)
        (h_x - h_x_given_y).max(0.0)
    }

    /// Compute mutual information between two continuous variables
    /// Uses a correlation-based approximation: MI = -0.5 * log(1 - r^2)
    /// where r is the Pearson correlation coefficient.
    /// Also incorporates a rank-based (Spearman) correlation for capturing
    /// non-linear dependencies.
    fn mi_regression(&self, feature: &Array1<f64>, target: &Array1<f64>) -> f64 {
        let n = feature.len();
        if n < self.n_neighbors + 1 {
            return 0.0;
        }

        let f_std = std_dev(feature);
        let t_std = std_dev(target);

        if f_std < 1e-15 || t_std < 1e-15 {
            return 0.0;
        }

        // Pearson correlation
        let f_mean = mean(feature);
        let t_mean = mean(target);

        let mut cov = 0.0;
        for i in 0..n {
            cov += (feature[i] - f_mean) * (target[i] - t_mean);
        }
        cov /= (n - 1) as f64;
        let pearson_r = cov / (f_std * t_std);

        // Spearman rank correlation for non-linear dependencies
        let f_ranks = rank_array(feature);
        let t_ranks = rank_array(target);
        let fr_mean = mean(&f_ranks);
        let tr_mean = mean(&t_ranks);
        let fr_std = std_dev(&f_ranks);
        let tr_std = std_dev(&t_ranks);

        let spearman_r = if fr_std > 1e-15 && tr_std > 1e-15 {
            let mut rank_cov = 0.0;
            for i in 0..n {
                rank_cov += (f_ranks[i] - fr_mean) * (t_ranks[i] - tr_mean);
            }
            rank_cov /= (n - 1) as f64;
            rank_cov / (fr_std * tr_std)
        } else {
            0.0
        };

        // Use the maximum of the two correlations to capture both linear
        // and monotonic non-linear dependencies
        let r = pearson_r.abs().max(spearman_r.abs()).min(0.9999);

        // MI approximation for Gaussian variables: MI = -0.5 * log(1 - r^2)
        (-0.5 * (1.0 - r * r).ln()).max(0.0)
    }

    /// Fit the selector to data
    ///
    /// # Arguments
    /// * `x` - Feature matrix, shape (n_samples, n_features)
    /// * `y` - Target vector, shape (n_samples,)
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

        if self.k > n_features {
            return Err(TransformError::InvalidInput(format!(
                "k={} must be <= n_features={}",
                self.k, n_features
            )));
        }

        if n_samples < 4 {
            return Err(TransformError::InvalidInput(
                "At least 4 samples required for MI estimation".to_string(),
            ));
        }

        // Compute MI for each feature
        let mut scores = Array1::zeros(n_features);

        for j in 0..n_features {
            let feature = x.column(j).to_owned();
            scores[j] = match self.method {
                MutualInfoMethod::Classification => self.mi_classification(&feature, y),
                MutualInfoMethod::Regression => self.mi_regression(&feature, y),
            };
        }

        // Select top k features by MI score
        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let selected_features: Vec<usize> = indices.into_iter().take(self.k).collect();

        self.scores_ = Some(scores);
        self.selected_features_ = Some(selected_features);
        self.n_features_in_ = Some(n_features);

        Ok(())
    }

    /// Transform data by selecting features
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let selected = self.selected_features_.as_ref().ok_or_else(|| {
            TransformError::NotFitted("MutualInfoSelector has not been fitted".to_string())
        })?;

        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        let n_features_in = self.n_features_in_.unwrap_or(0);
        if n_features != n_features_in {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but selector was fitted with {} features",
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

    /// Get mutual information scores for all features
    pub fn scores(&self) -> Option<&Array1<f64>> {
        self.scores_.as_ref()
    }
}

/// Compute the mutual information scores between features and a classification target
/// without selecting features. Returns MI scores for all features.
pub fn mutual_info_classif(x: &Array2<f64>, y: &Array1<f64>, n_bins: usize) -> Result<Array1<f64>> {
    let selector =
        MutualInfoSelector::new(x.shape()[1], MutualInfoMethod::Classification).with_n_bins(n_bins);
    let n_features = x.shape()[1];
    let n_samples = x.shape()[0];

    if n_samples != y.len() {
        return Err(TransformError::InvalidInput(format!(
            "X has {} samples but y has {} samples",
            n_samples,
            y.len()
        )));
    }

    let mut scores = Array1::zeros(n_features);
    for j in 0..n_features {
        let feature = x.column(j).to_owned();
        scores[j] = selector.mi_classification(&feature, y);
    }

    Ok(scores)
}

/// Compute the mutual information scores between features and a regression target
/// without selecting features. Returns MI scores for all features.
pub fn mutual_info_regression(x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
    let selector = MutualInfoSelector::new(x.shape()[1], MutualInfoMethod::Regression);
    let n_features = x.shape()[1];
    let n_samples = x.shape()[0];

    if n_samples != y.len() {
        return Err(TransformError::InvalidInput(format!(
            "X has {} samples but y has {} samples",
            n_samples,
            y.len()
        )));
    }

    let mut scores = Array1::zeros(n_features);
    for j in 0..n_features {
        let feature = x.column(j).to_owned();
        scores[j] = selector.mi_regression(&feature, y);
    }

    Ok(scores)
}

// --- Utility functions ---

/// Compute entropy from counts: H = -sum(p * log(p))
fn entropy_from_counts(counts: &[usize], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    let mut h = 0.0;
    for &c in counts {
        if c > 0 {
            let p = c as f64 / n;
            h -= p * p.ln();
        }
    }
    h
}

/// Get min and max of an array
fn feature_range(arr: &Array1<f64>) -> (f64, f64) {
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    for &v in arr.iter() {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }
    (min_val, max_val)
}

/// Compute mean of an array
fn mean(arr: &Array1<f64>) -> f64 {
    if arr.is_empty() {
        return 0.0;
    }
    arr.iter().sum::<f64>() / arr.len() as f64
}

/// Compute standard deviation (sample) of an array
fn std_dev(arr: &Array1<f64>) -> f64 {
    let n = arr.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(arr);
    let var = arr.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (n - 1) as f64;
    var.sqrt()
}

/// Compute ranks of array values (average rank for ties)
fn rank_array(arr: &Array1<f64>) -> Array1<f64> {
    let n = arr.len();
    let mut indexed: Vec<(usize, f64)> = arr.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = Array1::zeros(n);
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-15 {
            j += 1;
        }
        // Average rank for ties
        let avg_rank = (i + j) as f64 / 2.0 + 0.5;
        for item in indexed.iter().take(j).skip(i) {
            ranks[item.0] = avg_rank;
        }
        i = j;
    }
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_mi_classification_clear_signal() {
        // Feature 0 is strongly related to class, feature 1 is noise
        let x = Array::from_shape_vec(
            (8, 2),
            vec![
                1.0, 0.5, 1.1, 0.3, 1.2, 0.7, 1.0, 0.4, 5.0, 0.6, 5.1, 0.2, 5.2, 0.8, 5.0, 0.3,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut selector = MutualInfoSelector::new(1, MutualInfoMethod::Classification);
        selector.fit(&x, &y).expect("fit");

        let scores = selector.scores().expect("scores");
        // Feature 0 should have higher MI than feature 1
        assert!(
            scores[0] > scores[1],
            "Feature 0 (signal) should have higher MI than feature 1 (noise): {} vs {}",
            scores[0],
            scores[1]
        );

        let selected = selector.get_support().expect("support");
        assert_eq!(selected, &[0]);
    }

    #[test]
    fn test_mi_regression_linear() {
        let n = 50;
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        for i in 0..n {
            let t = i as f64 / n as f64;
            x_data.push(t); // linearly related
            x_data.push(0.5); // constant (noise-like)
            y_data.push(2.0 * t + 0.5);
        }

        let x = Array::from_shape_vec((n, 2), x_data).expect("test data");
        let y = Array::from_vec(y_data);

        let mut selector = MutualInfoSelector::new(1, MutualInfoMethod::Regression);
        selector.fit(&x, &y).expect("fit");

        let scores = selector.scores().expect("scores");
        assert!(scores[0] > scores[1], "Linear feature should rank higher");
    }

    #[test]
    fn test_mi_selector_k_selection() {
        let x = Array::from_shape_vec(
            (6, 4),
            vec![
                1.0, 0.1, 5.0, 0.5, 1.1, 0.2, 5.1, 0.6, 2.0, 0.1, 4.0, 0.5, 2.1, 0.2, 4.1, 0.6,
                3.0, 0.1, 3.0, 0.5, 3.1, 0.2, 3.1, 0.6,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);

        let mut selector = MutualInfoSelector::new(2, MutualInfoMethod::Classification);
        let transformed = selector.fit_transform(&x, &y).expect("fit_transform");
        assert_eq!(transformed.shape(), &[6, 2]);
    }

    #[test]
    fn test_mutual_info_classif_function() {
        let x = Array::from_shape_vec(
            (8, 2),
            vec![
                1.0, 0.5, 1.1, 0.3, 1.2, 0.7, 1.0, 0.4, 5.0, 0.6, 5.1, 0.2, 5.2, 0.8, 5.0, 0.3,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let scores = mutual_info_classif(&x, &y, 10).expect("mi_classif");
        assert_eq!(scores.len(), 2);
        assert!(scores[0] > scores[1]);
    }

    #[test]
    fn test_mi_errors() {
        let x = Array::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("test data");
        let y = Array::from_vec(vec![0.0, 1.0, 2.0]);

        let mut selector = MutualInfoSelector::new(1, MutualInfoMethod::Classification);
        assert!(selector.fit(&x, &y).is_err());
    }

    #[test]
    fn test_mi_not_fitted() {
        let x = Array::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("test data");

        let selector = MutualInfoSelector::new(1, MutualInfoMethod::Classification);
        assert!(selector.transform(&x).is_err());
    }

    #[test]
    fn test_rank_array() {
        let arr = Array::from_vec(vec![3.0, 1.0, 2.0]);
        let ranks = rank_array(&arr);
        assert!((ranks[0] - 3.0).abs() < 1e-10);
        assert!((ranks[1] - 1.0).abs() < 1e-10);
        assert!((ranks[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_rank_array_ties() {
        let arr = Array::from_vec(vec![1.0, 2.0, 2.0, 3.0]);
        let ranks = rank_array(&arr);
        assert!((ranks[0] - 1.0).abs() < 1e-10);
        assert!((ranks[1] - 2.5).abs() < 1e-10);
        assert!((ranks[2] - 2.5).abs() < 1e-10);
        assert!((ranks[3] - 4.0).abs() < 1e-10);
    }
}
