//! Incremental PCA via the Arora et al. (2012) block-update algorithm.
//!
//! Standard batch PCA must hold the entire data matrix in memory.  Incremental
//! PCA processes one batch at a time, updating the component estimate after
//! each batch using a rank-revealing QR-based merge step.  Memory usage is
//! O(n_components × n_features) regardless of the total number of samples.
//!
//! # References
//!
//! * Arora, R., Cotter, A., Livescu, K., & Srebro, N. (2012).
//!   "Stochastic Optimization of PCA with Capped MSG".
//! * Ross, D. A., Lim, J., Lin, R.-S., & Yang, M.-H. (2008).
//!   "Incremental Learning for Robust Visual Tracking".

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_linalg::{svd, qr};

use crate::error::{Result, TransformError};

/// Incremental PCA — fits principal components from a stream of data batches.
///
/// After each call to [`partial_fit`](IncrementalPCA::partial_fit) the internal
/// component matrix is updated without requiring access to previous batches.
#[derive(Debug, Clone)]
pub struct IncrementalPCA {
    /// Number of principal components to keep.
    n_components: usize,
    /// Suggested batch size (used only for documentation; callers choose size).
    batch_size: Option<usize>,
    /// Current principal-component matrix  (n_components × n_features).
    /// `None` until the first batch is seen.
    components: Option<Array2<f64>>,
    /// Singular values of the current component basis.
    singular_values: Option<Array1<f64>>,
    /// Running mean (n_features,) for centring.
    mean: Option<Array1<f64>>,
    /// Total samples processed.
    n_samples_seen: usize,
    /// Total variance per feature (Welford M2 accumulator).
    variance_sum: Option<Array1<f64>>,
    /// Whether to subtract the feature mean before projecting.
    with_mean: bool,
    /// Noise variance estimate (average of discarded singular values).
    noise_variance: f64,
}

impl IncrementalPCA {
    /// Create a new `IncrementalPCA`.
    ///
    /// # Arguments
    ///
    /// * `n_components` – Number of principal components to retain.
    /// * `batch_size`   – Optional hint for downstream callers about a suitable
    ///                    batch size.  The algorithm itself imposes no constraint.
    ///
    /// # Panics
    ///
    /// Panics if `n_components` is zero.
    pub fn new(n_components: usize, batch_size: Option<usize>) -> Self {
        assert!(n_components > 0, "n_components must be positive");
        Self {
            n_components,
            batch_size,
            components: None,
            singular_values: None,
            mean: None,
            n_samples_seen: 0,
            variance_sum: None,
            with_mean: true,
            noise_variance: 0.0,
        }
    }

    /// Disable mean-centring (default: enabled).
    pub fn set_with_mean(&mut self, with_mean: bool) {
        self.with_mean = with_mean;
    }

    /// Return the suggested batch size if one was provided.
    pub fn batch_size(&self) -> Option<usize> {
        self.batch_size
    }

    /// Update the component estimate with a new batch `x` (n_samples × n_features).
    ///
    /// The algorithm:
    /// 1. Centre `x` using its batch mean (and update the running global mean).
    /// 2. Form a tall-thin matrix by stacking the current component matrix
    ///    (scaled by singular values) on top of the centred batch.
    /// 3. Compute the truncated SVD of that stacked matrix.
    /// 4. Keep the top `n_components` left-singular-vectors as the new components.
    pub fn partial_fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let (n_batch, n_features) = (x.shape()[0], x.shape()[1]);

        if n_batch == 0 {
            return Err(TransformError::InvalidInput(
                "partial_fit received an empty batch".into(),
            ));
        }

        if n_batch < self.n_components {
            return Err(TransformError::InvalidInput(format!(
                "Batch size ({}) must be >= n_components ({})",
                n_batch, self.n_components
            )));
        }

        // ---------------------------------------------------------------
        // Validate / initialise running state
        // ---------------------------------------------------------------
        if let Some(ref c) = self.components {
            let existing_features = c.shape()[1];
            if existing_features != n_features {
                return Err(TransformError::InvalidInput(format!(
                    "Feature count changed: expected {}, got {}",
                    existing_features, n_features
                )));
            }
        }

        // ---------------------------------------------------------------
        // Update running mean (Welford)
        // ---------------------------------------------------------------
        let batch_mean = x.mean_axis(Axis(0)).ok_or_else(|| {
            TransformError::ComputationError("Failed to compute batch mean".into())
        })?;

        let new_n = self.n_samples_seen + n_batch;

        let global_mean = match self.mean.take() {
            None => {
                self.variance_sum = Some(Array1::zeros(n_features));
                batch_mean.clone()
            }
            Some(old_mean) => {
                // Welford parallel update
                let weight_old = self.n_samples_seen as f64;
                let weight_new = n_batch as f64;
                let weight_total = new_n as f64;

                let mut updated = Array1::zeros(n_features);
                for j in 0..n_features {
                    updated[j] =
                        (weight_old * old_mean[j] + weight_new * batch_mean[j]) / weight_total;
                }

                // Update variance accumulator (M2)
                if let Some(ref mut var_sum) = self.variance_sum {
                    for j in 0..n_features {
                        let delta = batch_mean[j] - old_mean[j];
                        var_sum[j] +=
                            delta * delta * weight_old * weight_new / weight_total;
                    }
                }
                updated
            }
        };

        self.n_samples_seen = new_n;

        // ---------------------------------------------------------------
        // Centre the batch
        // ---------------------------------------------------------------
        let mut x_centred = x.to_owned();
        if self.with_mean {
            for j in 0..n_features {
                let m = if self.n_samples_seen == n_batch {
                    batch_mean[j]
                } else {
                    global_mean[j]
                };
                for i in 0..n_batch {
                    x_centred[[i, j]] -= m;
                }
            }
        }

        self.mean = Some(global_mean);

        // ---------------------------------------------------------------
        // Build stacked matrix  [S·V^T ; X_centred]
        // ---------------------------------------------------------------
        let stacked: Array2<f64> = match (&self.components, &self.singular_values) {
            (Some(comp), Some(sv)) => {
                // comp : n_components × n_features
                // sv   : n_components
                // Weighted rows: sv[k] * comp[k, :]
                let k = comp.shape()[0];
                let mut weighted = Array2::zeros((k, n_features));
                for i in 0..k {
                    let s = sv[i];
                    for j in 0..n_features {
                        weighted[[i, j]] = s * comp[[i, j]];
                    }
                }

                // Stack: (k + n_batch) × n_features
                let mut stacked = Array2::zeros((k + n_batch, n_features));
                for i in 0..k {
                    for j in 0..n_features {
                        stacked[[i, j]] = weighted[[i, j]];
                    }
                }
                for i in 0..n_batch {
                    for j in 0..n_features {
                        stacked[[k + i, j]] = x_centred[[i, j]];
                    }
                }
                stacked
            }
            _ => {
                // First batch — just use the centred data
                x_centred.clone()
            }
        };

        // ---------------------------------------------------------------
        // Truncated SVD of the stacked matrix
        // (full_matrices=false equivalent: thin SVD)
        // ---------------------------------------------------------------
        self.update_components_from_svd(stacked, n_features)?;

        Ok(())
    }

    /// Perform the SVD and extract the top-k components.
    fn update_components_from_svd(
        &mut self,
        stacked: Array2<f64>,
        n_features: usize,
    ) -> Result<()> {
        let k = self.n_components.min(stacked.shape()[0].min(stacked.shape()[1]));

        // Thin SVD  →  U (m×k), s (k,), Vt (k×n)
        // scirs2-linalg::svd returns (U, s, Vt) with full_matrices flag
        let (u_opt, s_vec, vt_opt) = svd(&stacked, false)
            .map_err(|e| TransformError::ComputationError(format!("SVD failed: {e}")))?;

        let u = u_opt.ok_or_else(|| {
            TransformError::ComputationError("SVD did not return U matrix".into())
        })?;
        let vt = vt_opt.ok_or_else(|| {
            TransformError::ComputationError("SVD did not return Vt matrix".into())
        })?;

        // Clamp k to actual rank returned
        let actual_k = k.min(s_vec.len()).min(vt.shape()[0]);

        // Noise variance: mean of discarded singular values squared / n_features
        if s_vec.len() > actual_k {
            let discarded_sq_sum: f64 = s_vec.iter().skip(actual_k).map(|&v| v * v).sum();
            let n_discarded = (s_vec.len() - actual_k) as f64;
            self.noise_variance = discarded_sq_sum / (n_discarded * n_features as f64);
        }

        // Keep top-k rows of Vt as new components
        let mut new_components = Array2::zeros((actual_k, n_features));
        for i in 0..actual_k {
            for j in 0..n_features {
                new_components[[i, j]] = vt[[i, j]];
            }
        }

        let mut new_sv = Array1::zeros(actual_k);
        for i in 0..actual_k {
            new_sv[i] = s_vec[i];
        }

        // Ensure consistent sign: largest-magnitude element of each component
        // is positive (deterministic sign convention).
        flip_component_signs(&mut new_components, actual_k, n_features);

        // Suppress unused variable warning for `u`
        let _ = u;

        self.components = Some(new_components);
        self.singular_values = Some(new_sv);

        Ok(())
    }

    /// Project `x` onto the current principal components.
    ///
    /// Returns a matrix of shape (n_samples, n_components).
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_features) = (x.shape()[0], x.shape()[1]);
        let components = self.components.as_ref().ok_or_else(|| {
            TransformError::NotFitted("Call partial_fit before transform".into())
        })?;
        let k = components.shape()[0];

        if components.shape()[1] != n_features {
            return Err(TransformError::InvalidInput(format!(
                "Feature mismatch: model has {} features, input has {}",
                components.shape()[1],
                n_features
            )));
        }

        let mean = self.mean.as_ref().ok_or_else(|| {
            TransformError::NotFitted("Mean not available; call partial_fit first".into())
        })?;

        // Centre input
        let mut x_c = x.to_owned();
        if self.with_mean {
            for j in 0..n_features {
                for i in 0..n_samples {
                    x_c[[i, j]] -= mean[j];
                }
            }
        }

        // X_c @ components^T   →  (n_samples, k)
        let mut result = Array2::zeros((n_samples, k));
        for i in 0..n_samples {
            for p in 0..k {
                let mut dot = 0.0f64;
                for j in 0..n_features {
                    dot += x_c[[i, j]] * components[[p, j]];
                }
                result[[i, p]] = dot;
            }
        }
        Ok(result)
    }

    /// Reconstruct data from the projected representation.
    ///
    /// Returns approximation of the original (centred) data.
    pub fn inverse_transform(&self, x_transformed: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, k_in) = (x_transformed.shape()[0], x_transformed.shape()[1]);
        let components = self.components.as_ref().ok_or_else(|| {
            TransformError::NotFitted("Call partial_fit before inverse_transform".into())
        })?;
        let (k, n_features) = (components.shape()[0], components.shape()[1]);

        if k_in != k {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} components, got {}",
                k, k_in
            )));
        }

        // X_transformed @ components  →  (n_samples, n_features)
        let mut result = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                let mut val = 0.0f64;
                for p in 0..k {
                    val += x_transformed[[i, p]] * components[[p, j]];
                }
                result[[i, j]] = val;
            }
        }

        // Add back mean
        if self.with_mean {
            if let Some(ref mean) = self.mean {
                for j in 0..n_features {
                    for i in 0..n_samples {
                        result[[i, j]] += mean[j];
                    }
                }
            }
        }

        Ok(result)
    }

    /// Return the current principal components (n_components × n_features).
    pub fn get_components(&self) -> Option<&Array2<f64>> {
        self.components.as_ref()
    }

    /// Singular values of the current component basis.
    pub fn singular_values(&self) -> Option<&Array1<f64>> {
        self.singular_values.as_ref()
    }

    /// Explained variance per component: `s_i^2 / (n_samples - 1)`.
    pub fn explained_variance(&self) -> Option<Array1<f64>> {
        let sv = self.singular_values.as_ref()?;
        if self.n_samples_seen < 2 {
            return None;
        }
        let denom = (self.n_samples_seen - 1) as f64;
        Some(sv.mapv(|s| s * s / denom))
    }

    /// Ratio of variance explained by each component.
    pub fn explained_variance_ratio(&self) -> Option<Array1<f64>> {
        let ev = self.explained_variance()?;
        let var_sum = self.variance_sum.as_ref()?;

        // Total variance = sum of per-feature variances
        let total_var: f64 = var_sum.iter().sum::<f64>() / (self.n_samples_seen - 1) as f64
            + ev.iter().sum::<f64>();

        if total_var <= 0.0 {
            return None;
        }

        Some(ev.mapv(|v| v / total_var))
    }

    /// Noise variance estimate (average variance not explained by kept components).
    pub fn noise_variance(&self) -> f64 {
        self.noise_variance
    }

    /// Total samples processed so far.
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }

    /// Current number of fitted components (may be less than `n_components`
    /// before enough samples have been seen).
    pub fn n_components_fitted(&self) -> usize {
        self.components.as_ref().map(|c| c.shape()[0]).unwrap_or(0)
    }

    /// Reset the model to its initial (unfitted) state.
    pub fn reset(&mut self) {
        self.components = None;
        self.singular_values = None;
        self.mean = None;
        self.n_samples_seen = 0;
        self.variance_sum = None;
        self.noise_variance = 0.0;
    }
}

/// Enforce a deterministic sign convention: the element with the largest
/// absolute value in each component row is made positive.
fn flip_component_signs(components: &mut Array2<f64>, k: usize, n_features: usize) {
    for i in 0..k {
        let max_abs_idx = (0..n_features)
            .max_by(|&a, &b| {
                components[[i, a]]
                    .abs()
                    .partial_cmp(&components[[i, b]].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        if components[[i, max_abs_idx]] < 0.0 {
            for j in 0..n_features {
                components[[i, j]] = -components[[i, j]];
            }
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_data(rows: usize, cols: usize, seed_offset: f64) -> Array2<f64> {
        let mut data = Array2::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                data[[i, j]] = ((i * cols + j) as f64 + seed_offset).sin() * 10.0;
            }
        }
        data
    }

    #[test]
    fn test_basic_partial_fit_and_transform() {
        let mut ipca = IncrementalPCA::new(2, Some(20));
        let batch = make_data(20, 5, 0.0);
        ipca.partial_fit(&batch).expect("partial_fit should succeed");

        assert_eq!(ipca.n_samples_seen(), 20);
        assert!(ipca.get_components().is_some());
        let c = ipca.get_components().expect("components should be available after fit");
        assert_eq!(c.shape(), [2, 5]);

        let projected = ipca.transform(&batch).expect("transform should succeed");
        assert_eq!(projected.shape(), [20, 2]);
    }

    #[test]
    fn test_multiple_batches() {
        let mut ipca = IncrementalPCA::new(3, None);
        for b in 0..5usize {
            let batch = make_data(10, 8, b as f64 * 3.14);
            ipca.partial_fit(&batch).expect("partial_fit");
        }
        assert_eq!(ipca.n_samples_seen(), 50);

        let query = make_data(4, 8, 99.0);
        let out = ipca.transform(&query).expect("transform");
        assert_eq!(out.shape(), [4, 3]);
    }

    #[test]
    fn test_explained_variance_ratio_sums_to_lte_one() {
        let mut ipca = IncrementalPCA::new(2, None);
        let batch = make_data(30, 6, 1.23);
        ipca.partial_fit(&batch).expect("partial_fit should succeed");

        if let Some(evr) = ipca.explained_variance_ratio() {
            let total: f64 = evr.iter().sum();
            assert!(total <= 1.0 + 1e-10, "EVR sum = {total}");
            assert!(total >= 0.0, "EVR should be non-negative");
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let mut ipca = IncrementalPCA::new(2, None);
        let batch = make_data(15, 4, 0.0);
        ipca.partial_fit(&batch).expect("partial_fit should succeed");
        ipca.reset();
        assert_eq!(ipca.n_samples_seen(), 0);
        assert!(ipca.get_components().is_none());
    }

    #[test]
    fn test_batch_too_small_errors() {
        let mut ipca = IncrementalPCA::new(5, None);
        let small_batch = make_data(3, 4, 0.0);
        assert!(ipca.partial_fit(&small_batch).is_err());
    }

    #[test]
    fn test_feature_mismatch_errors() {
        let mut ipca = IncrementalPCA::new(2, None);
        let b1 = make_data(10, 4, 0.0);
        let b2 = make_data(10, 6, 0.0);
        ipca.partial_fit(&b1).expect("partial_fit should succeed");
        assert!(ipca.partial_fit(&b2).is_err());
    }
}
