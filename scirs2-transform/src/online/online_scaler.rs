//! Online (streaming) normalization scalers.
//!
//! All three scalers support the same `partial_fit / transform / inverse_transform` API
//! and process data one batch at a time without retaining the raw data.
//!
//! | Struct | Statistic tracked | Reference |
//! |--------|------------------|-----------|
//! | [`OnlineStandardScaler`] | Mean + variance (Welford) | Welford 1962 |
//! | [`OnlineMinMaxScaler`]   | Running min / max | — |
//! | [`OnlineRobustScaler`]   | Median + IQR (GK quantile sketch) | Greenwald & Khanna 2001 |

use std::collections::BinaryHeap;

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{Result, TransformError};

// ════════════════════════════════════════════════════════════════════════════
// OnlineStandardScaler
// ════════════════════════════════════════════════════════════════════════════

/// Streaming standard (z-score) scaler.
///
/// Maintains running mean and variance via Welford's online algorithm.
/// Produces zero-mean, unit-variance output as new batches arrive.
#[derive(Debug, Clone)]
pub struct OnlineStandardScaler {
    /// Feature-wise running mean.
    mean: Array1<f64>,
    /// Feature-wise Welford M2 accumulator (sum of squared deviations).
    m2: Array1<f64>,
    /// Total samples seen.
    n_samples: usize,
    /// Whether to subtract the mean.
    with_mean: bool,
    /// Whether to divide by the standard deviation.
    with_std: bool,
    /// Minimum denominator to avoid division by zero.
    epsilon: f64,
}

impl OnlineStandardScaler {
    /// Create a new scaler for `n_features` features.
    pub fn new(n_features: usize) -> Self {
        Self {
            mean: Array1::zeros(n_features),
            m2: Array1::zeros(n_features),
            n_samples: 0,
            with_mean: true,
            with_std: true,
            epsilon: 1e-8,
        }
    }

    /// Configure whether to centre (default: true).
    pub fn set_with_mean(mut self, v: bool) -> Self {
        self.with_mean = v;
        self
    }

    /// Configure whether to scale (default: true).
    pub fn set_with_std(mut self, v: bool) -> Self {
        self.with_std = v;
        self
    }

    fn n_features(&self) -> usize {
        self.mean.len()
    }

    /// Update running statistics using Welford's online algorithm.
    pub fn partial_fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let (n_batch, n_feat) = (x.shape()[0], x.shape()[1]);
        if n_feat != self.n_features() {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features(),
                n_feat
            )));
        }
        for i in 0..n_batch {
            self.n_samples += 1;
            let n = self.n_samples as f64;
            for j in 0..n_feat {
                let val = x[[i, j]];
                let delta = val - self.mean[j];
                self.mean[j] += delta / n;
                let delta2 = val - self.mean[j];
                self.m2[j] += delta * delta2;
            }
        }
        Ok(())
    }

    /// Transform `x` using current statistics.
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_feat) = (x.shape()[0], x.shape()[1]);
        if n_feat != self.n_features() {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features(),
                n_feat
            )));
        }
        if self.n_samples == 0 {
            return Err(TransformError::NotFitted(
                "Call partial_fit before transform".into(),
            ));
        }

        let std = self.current_std();

        let mut result = x.to_owned();
        for j in 0..n_feat {
            for i in 0..n_samples {
                if self.with_mean {
                    result[[i, j]] -= self.mean[j];
                }
                if self.with_std {
                    result[[i, j]] /= std[j];
                }
            }
        }
        Ok(result)
    }

    /// Invert the transformation.
    pub fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_feat) = (x.shape()[0], x.shape()[1]);
        if n_feat != self.n_features() {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features(),
                n_feat
            )));
        }
        if self.n_samples == 0 {
            return Err(TransformError::NotFitted(
                "Call partial_fit before inverse_transform".into(),
            ));
        }

        let std = self.current_std();
        let mut result = x.to_owned();
        for j in 0..n_feat {
            for i in 0..n_samples {
                if self.with_std {
                    result[[i, j]] *= std[j];
                }
                if self.with_mean {
                    result[[i, j]] += self.mean[j];
                }
            }
        }
        Ok(result)
    }

    /// Return the current per-feature standard deviation.
    fn current_std(&self) -> Array1<f64> {
        if self.n_samples <= 1 {
            return Array1::ones(self.n_features());
        }
        let denom = (self.n_samples - 1) as f64;
        self.m2.mapv(|m2| (m2 / denom).sqrt().max(self.epsilon))
    }

    /// Return a copy of the running mean.
    pub fn mean(&self) -> &Array1<f64> {
        &self.mean
    }

    /// Return the current variance estimate (sample variance).
    pub fn variance(&self) -> Array1<f64> {
        if self.n_samples <= 1 {
            return Array1::zeros(self.n_features());
        }
        let denom = (self.n_samples - 1) as f64;
        self.m2.mapv(|m2| m2 / denom)
    }

    /// Total samples seen.
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.mean.fill(0.0);
        self.m2.fill(0.0);
        self.n_samples = 0;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// OnlineMinMaxScaler
// ════════════════════════════════════════════════════════════════════════════

/// Streaming min-max scaler.
///
/// Scales features to the `[feature_range.0, feature_range.1]` interval based
/// on the running minimum and maximum.
#[derive(Debug, Clone)]
pub struct OnlineMinMaxScaler {
    /// Feature-wise running minimum.
    min: Array1<f64>,
    /// Feature-wise running maximum.
    max: Array1<f64>,
    /// Number of samples seen.
    n_samples: usize,
    /// Output range `(lo, hi)`.
    feature_range: (f64, f64),
}

impl OnlineMinMaxScaler {
    /// Create a new scaler, mapping to `[0, 1]` by default.
    pub fn new(n_features: usize) -> Self {
        Self::with_range(n_features, (0.0, 1.0))
    }

    /// Create a new scaler with a custom output range.
    pub fn with_range(n_features: usize, feature_range: (f64, f64)) -> Self {
        Self {
            min: Array1::from_elem(n_features, f64::INFINITY),
            max: Array1::from_elem(n_features, f64::NEG_INFINITY),
            n_samples: 0,
            feature_range,
        }
    }

    fn n_features(&self) -> usize {
        self.min.len()
    }

    /// Update running min and max.
    pub fn partial_fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let (n_batch, n_feat) = (x.shape()[0], x.shape()[1]);
        if n_feat != self.n_features() {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features(),
                n_feat
            )));
        }
        for i in 0..n_batch {
            for j in 0..n_feat {
                let v = x[[i, j]];
                if v < self.min[j] {
                    self.min[j] = v;
                }
                if v > self.max[j] {
                    self.max[j] = v;
                }
            }
        }
        self.n_samples += n_batch;
        Ok(())
    }

    /// Scale `x` to the configured feature range.
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_feat) = (x.shape()[0], x.shape()[1]);
        if n_feat != self.n_features() {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features(),
                n_feat
            )));
        }
        if self.n_samples == 0 {
            return Err(TransformError::NotFitted(
                "Call partial_fit before transform".into(),
            ));
        }

        let (lo, hi) = self.feature_range;
        let range_out = hi - lo;
        let mut result = x.to_owned();

        for j in 0..n_feat {
            let range_in = self.max[j] - self.min[j];
            for i in 0..n_samples {
                if range_in <= 0.0 {
                    result[[i, j]] = lo;
                } else {
                    result[[i, j]] =
                        lo + (x[[i, j]] - self.min[j]) / range_in * range_out;
                }
            }
        }
        Ok(result)
    }

    /// Invert the min-max scaling.
    pub fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_feat) = (x.shape()[0], x.shape()[1]);
        if n_feat != self.n_features() {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features(),
                n_feat
            )));
        }
        if self.n_samples == 0 {
            return Err(TransformError::NotFitted(
                "Call partial_fit before inverse_transform".into(),
            ));
        }

        let (lo, hi) = self.feature_range;
        let range_out = hi - lo;
        let mut result = x.to_owned();

        for j in 0..n_feat {
            let range_in = self.max[j] - self.min[j];
            for i in 0..n_samples {
                if range_out <= 0.0 {
                    result[[i, j]] = self.min[j];
                } else {
                    result[[i, j]] =
                        self.min[j] + (x[[i, j]] - lo) / range_out * range_in;
                }
            }
        }
        Ok(result)
    }

    /// Running minimum per feature.
    pub fn data_min(&self) -> &Array1<f64> {
        &self.min
    }

    /// Running maximum per feature.
    pub fn data_max(&self) -> &Array1<f64> {
        &self.max
    }

    /// Total samples seen.
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.min.fill(f64::INFINITY);
        self.max.fill(f64::NEG_INFINITY);
        self.n_samples = 0;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// OnlineRobustScaler  (Greenwald-Khanna quantile sketch)
// ════════════════════════════════════════════════════════════════════════════

/// A single GK sketch entry: (value, g, delta).
///
/// `g` = `rank_min(v_i) - rank_min(v_{i-1})`
/// `delta` = `rank_max(v_i) - rank_min(v_i)`
#[derive(Debug, Clone)]
struct GKEntry {
    v: f64,
    g: usize,
    delta: usize,
}

/// Greenwald-Khanna ε-approximate quantile summary for a single feature.
///
/// Memory: O(1/ε · log(ε·n)).  For ε = 0.01 the error is ≤ 1 % rank error.
#[derive(Debug, Clone)]
struct GKSketch {
    summary: Vec<GKEntry>,
    n: usize,
    epsilon: f64,
}

impl GKSketch {
    fn new(epsilon: f64) -> Self {
        Self {
            summary: Vec::new(),
            n: 0,
            epsilon,
        }
    }

    fn insert(&mut self, v: f64) {
        self.n += 1;
        let capacity = (2.0 * self.epsilon * self.n as f64).floor() as usize;

        // Find insertion position
        let pos = self.summary.partition_point(|e| e.v <= v);

        let delta = if pos == 0 || pos == self.summary.len() {
            0
        } else {
            capacity.saturating_sub(1)
        };

        self.summary.insert(pos, GKEntry { v, g: 1, delta });

        // Compress: merge tuples where g_i + g_{i+1} + delta_{i+1} <= 2*epsilon*n
        if self.n % (1.max((1.0 / (2.0 * self.epsilon)) as usize)) == 0 {
            self.compress(capacity);
        }
    }

    fn compress(&mut self, capacity: usize) {
        let mut i = 0;
        while i + 1 < self.summary.len() {
            let combined = self.summary[i].g
                + self.summary[i + 1].g
                + self.summary[i + 1].delta;
            if combined <= capacity + 1 {
                // Merge i+1 into i
                self.summary[i].g += self.summary[i + 1].g;
                self.summary.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    /// Query for the `phi`-quantile (φ ∈ [0, 1]).
    fn query(&self, phi: f64) -> Option<f64> {
        if self.summary.is_empty() {
            return None;
        }
        if self.n == 0 {
            return None;
        }

        let target_rank = (phi * self.n as f64).ceil() as usize;
        let allowed_error = (self.epsilon * self.n as f64).ceil() as usize;

        let mut rank_min = 0usize;
        for entry in &self.summary {
            rank_min += entry.g;
            let rank_max = rank_min + entry.delta;
            if rank_min >= target_rank.saturating_sub(allowed_error)
                && rank_max <= target_rank + allowed_error
            {
                return Some(entry.v);
            }
        }

        // Fallback: return the value closest to the target rank
        let mut cumulative = 0usize;
        let mut best = self.summary[0].v;
        let mut best_dist = usize::MAX;
        for entry in &self.summary {
            cumulative += entry.g;
            let dist = if cumulative >= target_rank {
                cumulative - target_rank
            } else {
                target_rank - cumulative
            };
            if dist < best_dist {
                best_dist = dist;
                best = entry.v;
            }
        }
        Some(best)
    }
}

/// Streaming robust scaler using the Greenwald-Khanna ε-approximate quantile sketch.
///
/// Subtracts the approximate median and divides by the approximate IQR
/// (interquartile range = Q75 − Q25).
#[derive(Debug, Clone)]
pub struct OnlineRobustScaler {
    sketches: Vec<GKSketch>,
    n_samples: usize,
    /// Quantile range to compute the spread (default: (0.25, 0.75)).
    quantile_range: (f64, f64),
    /// GK sketch error parameter (default: 0.01 → 1% rank error).
    epsilon: f64,
    /// Minimum denominator.
    min_scale: f64,
}

impl OnlineRobustScaler {
    /// Create a new robust scaler for `n_features` features with default settings.
    pub fn new(n_features: usize) -> Self {
        Self::with_params(n_features, (0.25, 0.75), 0.01)
    }

    /// Create with custom quantile range and GK error parameter.
    ///
    /// * `quantile_range` – `(q_low, q_high)` in `[0, 1]`.
    /// * `epsilon`        – GK error bound (smaller = more accurate, more memory).
    pub fn with_params(
        n_features: usize,
        quantile_range: (f64, f64),
        epsilon: f64,
    ) -> Self {
        let sketches = (0..n_features).map(|_| GKSketch::new(epsilon)).collect();
        Self {
            sketches,
            n_samples: 0,
            quantile_range,
            epsilon,
            min_scale: 1e-8,
        }
    }

    fn n_features(&self) -> usize {
        self.sketches.len()
    }

    /// Update the quantile sketches with new data.
    pub fn partial_fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let (n_batch, n_feat) = (x.shape()[0], x.shape()[1]);
        if n_feat != self.n_features() {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features(),
                n_feat
            )));
        }
        for i in 0..n_batch {
            for j in 0..n_feat {
                self.sketches[j].insert(x[[i, j]]);
            }
        }
        self.n_samples += n_batch;
        Ok(())
    }

    /// Transform `x` using the running median and IQR.
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_feat) = (x.shape()[0], x.shape()[1]);
        if n_feat != self.n_features() {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features(),
                n_feat
            )));
        }
        if self.n_samples == 0 {
            return Err(TransformError::NotFitted(
                "Call partial_fit before transform".into(),
            ));
        }

        let (q_lo, q_hi) = self.quantile_range;
        let mut result = x.to_owned();

        for j in 0..n_feat {
            let median = self.sketches[j]
                .query(0.5)
                .unwrap_or(0.0);
            let lo = self.sketches[j].query(q_lo).unwrap_or(0.0);
            let hi = self.sketches[j].query(q_hi).unwrap_or(0.0);
            let iqr = (hi - lo).max(self.min_scale);

            for i in 0..n_samples {
                result[[i, j]] = (x[[i, j]] - median) / iqr;
            }
        }
        Ok(result)
    }

    /// Invert the robust scaling.
    pub fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_feat) = (x.shape()[0], x.shape()[1]);
        if n_feat != self.n_features() {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features(),
                n_feat
            )));
        }
        if self.n_samples == 0 {
            return Err(TransformError::NotFitted(
                "Call partial_fit before inverse_transform".into(),
            ));
        }

        let (q_lo, q_hi) = self.quantile_range;
        let mut result = x.to_owned();

        for j in 0..n_feat {
            let median = self.sketches[j].query(0.5).unwrap_or(0.0);
            let lo = self.sketches[j].query(q_lo).unwrap_or(0.0);
            let hi = self.sketches[j].query(q_hi).unwrap_or(0.0);
            let iqr = (hi - lo).max(self.min_scale);

            for i in 0..n_samples {
                result[[i, j]] = x[[i, j]] * iqr + median;
            }
        }
        Ok(result)
    }

    /// Total samples seen.
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples
    }

    /// Return the approximate median for each feature.
    pub fn center(&self) -> Vec<Option<f64>> {
        self.sketches.iter().map(|s| s.query(0.5)).collect()
    }

    /// Return the approximate IQR for each feature.
    pub fn scale(&self) -> Vec<Option<f64>> {
        let (q_lo, q_hi) = self.quantile_range;
        self.sketches
            .iter()
            .map(|s| {
                let lo = s.query(q_lo)?;
                let hi = s.query(q_hi)?;
                Some(hi - lo)
            })
            .collect()
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        for sketch in &mut self.sketches {
            *sketch = GKSketch::new(self.epsilon);
        }
        self.n_samples = 0;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn range_data(n: usize, features: usize, lo: f64, hi: f64) -> Array2<f64> {
        let mut d = Array2::zeros((n, features));
        for i in 0..n {
            for j in 0..features {
                // evenly spaced values
                d[[i, j]] = lo + (hi - lo) * (i * features + j) as f64
                    / (n * features) as f64;
            }
        }
        d
    }

    #[test]
    fn test_standard_scaler_zero_mean() {
        let mut sc = OnlineStandardScaler::new(3);
        let data = range_data(100, 3, 0.0, 10.0);
        sc.partial_fit(&data).expect("partial_fit should succeed");
        let out = sc.transform(&data).expect("transform should succeed");

        // Mean of transformed output should be ≈ 0
        for j in 0..3 {
            let col_mean: f64 = (0..100).map(|i| out[[i, j]]).sum::<f64>() / 100.0;
            assert!(
                col_mean.abs() < 1e-9,
                "col {j} mean = {col_mean}"
            );
        }
    }

    #[test]
    fn test_standard_scaler_round_trip() {
        let mut sc = OnlineStandardScaler::new(2);
        let data = range_data(50, 2, -5.0, 5.0);
        sc.partial_fit(&data).expect("partial_fit should succeed");
        let transformed = sc.transform(&data).expect("transform should succeed");
        let recovered = sc.inverse_transform(&transformed).expect("inverse_transform should succeed");
        for i in 0..50 {
            for j in 0..2 {
                assert!(
                    (data[[i, j]] - recovered[[i, j]]).abs() < 1e-9,
                    "round-trip error at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn test_minmax_range_clamp() {
        let mut sc = OnlineMinMaxScaler::new(2);
        let data = range_data(100, 2, 0.0, 1.0);
        sc.partial_fit(&data).expect("partial_fit should succeed");
        let out = sc.transform(&data).expect("transform should succeed");
        for i in 0..100 {
            for j in 0..2 {
                assert!(out[[i, j]] >= -1e-9 && out[[i, j]] <= 1.0 + 1e-9);
            }
        }
    }

    #[test]
    fn test_minmax_round_trip() {
        let mut sc = OnlineMinMaxScaler::with_range(2, (-1.0, 1.0));
        let data = range_data(40, 2, -3.0, 7.0);
        sc.partial_fit(&data).expect("partial_fit should succeed");
        let t = sc.transform(&data).expect("transform should succeed");
        let r = sc.inverse_transform(&t).expect("inverse_transform should succeed");
        for i in 0..40 {
            for j in 0..2 {
                assert!((data[[i, j]] - r[[i, j]]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_robust_scaler_median_approx() {
        let mut sc = OnlineRobustScaler::new(1);
        // Insert 1000 values from 1 to 1000
        let mut data = Array2::zeros((1000, 1));
        for i in 0..1000usize {
            data[[i, 0]] = (i + 1) as f64;
        }
        sc.partial_fit(&data).expect("partial_fit should succeed");

        let medians = sc.center();
        if let Some(Some(m)) = medians.first() {
            // true median is 500.5 — GK with ε=0.01 allows ±10 rank error
            assert!((*m - 500.5).abs() < 15.0, "median ≈ {m}");
        }
    }

    #[test]
    fn test_robust_scaler_transform_shape() {
        let mut sc = OnlineRobustScaler::new(4);
        let data = range_data(200, 4, -10.0, 10.0);
        sc.partial_fit(&data).expect("partial_fit should succeed");
        let out = sc.transform(&data).expect("transform should succeed");
        assert_eq!(out.shape(), data.shape());
    }
}
