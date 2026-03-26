//! Online/Streaming Interpolation with Incremental Updates (WS227)
//!
//! This module provides streaming interpolation that supports:
//! - Incremental data ingestion via `add_point` / `add_batch`
//! - Rank-1 (Sherman-Morrison) updates for RBF kernel matrices
//! - Moving-window mode that retains only the last N points
//! - Configurable update strategies
//!
//! # Example
//!
//! ```rust
//! use scirs2_interpolate::streaming_online::{
//!     OnlineRbfInterpolator, OnlineConfig, UpdateStrategy,
//! };
//!
//! let config = OnlineConfig {
//!     max_points: 100,
//!     window_mode: true,
//!     update_strategy: UpdateStrategy::ShermanMorrison,
//! };
//! let mut interp = OnlineRbfInterpolator::new(config, 1.0);
//! interp.add_point(0.0, 0.0).expect("add");
//! interp.add_point(1.0, 1.0).expect("add");
//! interp.add_point(2.0, 4.0).expect("add");
//! let val = interp.predict(1.5).expect("predict");
//! assert!(val > 1.0 && val < 4.0);
//! ```

use crate::error::{InterpolateError, InterpolateResult};

// ---------------------------------------------------------------------------
// UpdateStrategy
// ---------------------------------------------------------------------------

/// Strategy for updating the interpolation model when new data arrives.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateStrategy {
    /// Rank-1 Sherman-Morrison update of the kernel-matrix inverse.
    /// O(n^2) per point instead of O(n^3) full recompute.
    ShermanMorrison,
    /// Full recomputation of the kernel-matrix inverse.
    FullRecompute,
    /// Partial update: accumulate new points and recompute only when a
    /// threshold number of points has been added since the last recompute.
    PartialUpdate,
}

impl Default for UpdateStrategy {
    fn default() -> Self {
        Self::ShermanMorrison
    }
}

// ---------------------------------------------------------------------------
// OnlineConfig
// ---------------------------------------------------------------------------

/// Configuration for the online streaming interpolator.
#[derive(Debug, Clone)]
pub struct OnlineConfig {
    /// Maximum number of data points retained.  When exceeded the oldest
    /// points are evicted (FIFO).
    pub max_points: usize,
    /// If `true` the interpolator operates in moving-window mode and only
    /// keeps the last `max_points` data points.
    pub window_mode: bool,
    /// How the internal model is updated on ingestion.
    pub update_strategy: UpdateStrategy,
}

impl Default for OnlineConfig {
    fn default() -> Self {
        Self {
            max_points: 1000,
            window_mode: false,
            update_strategy: UpdateStrategy::ShermanMorrison,
        }
    }
}

// ---------------------------------------------------------------------------
// OnlineRbfInterpolator
// ---------------------------------------------------------------------------

/// Streaming/online RBF interpolator with incremental (rank-1) updates.
///
/// Internally maintains:
/// - Vectors of x- and y-values (in insertion order).
/// - The inverse of the kernel matrix (for Sherman-Morrison updates).
/// - RBF weights (alpha) such that `K * alpha = y`.
///
/// Gaussian kernel: `phi(r) = exp(-r^2 / (2 * sigma^2))`
pub struct OnlineRbfInterpolator {
    config: OnlineConfig,
    /// x-coordinates of stored data points.
    xs: Vec<f64>,
    /// y-coordinates of stored data points.
    ys: Vec<f64>,
    /// Inverse of the (regularised) kernel matrix  K_inv  (n x n, row-major).
    k_inv: Vec<f64>,
    /// RBF weights  alpha = K_inv * y .
    alpha: Vec<f64>,
    /// Kernel bandwidth (sigma).
    sigma: f64,
    /// Regularisation parameter added to the diagonal.
    regularisation: f64,
    /// Number of points added since the last full recompute (for PartialUpdate).
    points_since_recompute: usize,
    /// Threshold for PartialUpdate strategy.
    partial_update_threshold: usize,
}

impl OnlineRbfInterpolator {
    /// Create a new streaming RBF interpolator.
    ///
    /// * `config` – streaming configuration.
    /// * `sigma` – Gaussian kernel bandwidth.
    pub fn new(config: OnlineConfig, sigma: f64) -> Self {
        Self {
            config,
            xs: Vec::new(),
            ys: Vec::new(),
            k_inv: Vec::new(),
            alpha: Vec::new(),
            sigma: if sigma.abs() < 1e-30 { 1.0 } else { sigma },
            regularisation: 1e-10,
            points_since_recompute: 0,
            partial_update_threshold: 20,
        }
    }

    /// Number of data points currently stored.
    pub fn len(&self) -> usize {
        self.xs.len()
    }

    /// Whether the interpolator is empty.
    pub fn is_empty(&self) -> bool {
        self.xs.is_empty()
    }

    /// Return the current configuration.
    pub fn config(&self) -> &OnlineConfig {
        &self.config
    }

    // -- Kernel helpers -----------------------------------------------------

    /// Gaussian RBF kernel evaluation.
    #[inline]
    fn kernel(&self, xi: f64, xj: f64) -> f64 {
        let r = xi - xj;
        (-r * r / (2.0 * self.sigma * self.sigma)).exp()
    }

    // -- Full recompute -----------------------------------------------------

    /// Full O(n^3) recomputation of K_inv and alpha.
    fn full_recompute(&mut self) -> InterpolateResult<()> {
        let n = self.xs.len();
        if n == 0 {
            self.k_inv.clear();
            self.alpha.clear();
            return Ok(());
        }

        // Build kernel matrix K  (n x n, row-major).
        let mut k = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                k[i * n + j] = self.kernel(self.xs[i], self.xs[j]);
            }
            k[i * n + i] += self.regularisation;
        }

        // Invert K via Gauss-Jordan.
        self.k_inv = self.invert_matrix(&k, n)?;

        // alpha = K_inv * y
        self.alpha = self.mat_vec_mul(&self.k_inv, &self.ys, n);
        self.points_since_recompute = 0;
        Ok(())
    }

    /// Gauss-Jordan matrix inversion (row-major, n x n).
    fn invert_matrix(&self, a: &[f64], n: usize) -> InterpolateResult<Vec<f64>> {
        // Augmented matrix [A | I]
        let mut aug = vec![0.0f64; n * 2 * n];
        for i in 0..n {
            for j in 0..n {
                aug[i * 2 * n + j] = a[i * n + j];
            }
            aug[i * 2 * n + n + i] = 1.0;
        }

        for col in 0..n {
            // Partial pivoting
            let mut max_row = col;
            let mut max_val = aug[col * 2 * n + col].abs();
            for row in (col + 1)..n {
                let v = aug[row * 2 * n + col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_val < 1e-14 {
                return Err(InterpolateError::ComputationError(
                    "Singular kernel matrix during inversion".to_string(),
                ));
            }
            if max_row != col {
                for j in 0..(2 * n) {
                    let tmp = aug[col * 2 * n + j];
                    aug[col * 2 * n + j] = aug[max_row * 2 * n + j];
                    aug[max_row * 2 * n + j] = tmp;
                }
            }

            let pivot = aug[col * 2 * n + col];
            for j in 0..(2 * n) {
                aug[col * 2 * n + j] /= pivot;
            }

            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug[row * 2 * n + col];
                for j in 0..(2 * n) {
                    aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
                }
            }
        }

        // Extract inverse from augmented matrix.
        let mut inv = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                inv[i * n + j] = aug[i * 2 * n + n + j];
            }
        }
        Ok(inv)
    }

    /// Matrix-vector multiplication (row-major).
    fn mat_vec_mul(&self, m: &[f64], v: &[f64], n: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += m[i * n + j] * v[j];
            }
            out[i] = s;
        }
        out
    }

    // -- Sherman-Morrison rank-1 update -------------------------------------

    /// Add one point using Sherman-Morrison formula.
    ///
    /// Given existing K_inv (n x n) and a new point, build the (n+1) x (n+1)
    /// inverse incrementally.
    fn sherman_morrison_add(&mut self, x_new: f64, y_new: f64) -> InterpolateResult<()> {
        let n = self.xs.len(); // old size (before push)

        if n == 0 {
            // Bootstrap: single-point case.
            let k_val = self.kernel(x_new, x_new) + self.regularisation;
            if k_val.abs() < 1e-30 {
                return Err(InterpolateError::ComputationError(
                    "Degenerate single-point kernel".to_string(),
                ));
            }
            self.xs.push(x_new);
            self.ys.push(y_new);
            self.k_inv = vec![1.0 / k_val];
            self.alpha = vec![y_new / k_val];
            return Ok(());
        }

        // k_new = [K(x_new, x_1), …, K(x_new, x_n)]
        let mut k_new = Vec::with_capacity(n);
        for i in 0..n {
            k_new.push(self.kernel(x_new, self.xs[i]));
        }
        let k_nn = self.kernel(x_new, x_new) + self.regularisation;

        // u = K_inv * k_new  (n-vector)
        let u = self.mat_vec_mul(&self.k_inv, &k_new, n);

        // schur = k_nn - k_new^T * u
        let mut dot = 0.0;
        for i in 0..n {
            dot += k_new[i] * u[i];
        }
        let schur = k_nn - dot;
        if schur.abs() < 1e-30 {
            // Degenerate – fall back to full recompute.
            self.xs.push(x_new);
            self.ys.push(y_new);
            return self.full_recompute();
        }
        let inv_schur = 1.0 / schur;

        // Build new K_inv of size (n+1) x (n+1).
        let n1 = n + 1;
        let mut new_k_inv = vec![0.0f64; n1 * n1];

        // Top-left block: K_inv + inv_schur * u * u^T
        for i in 0..n {
            for j in 0..n {
                new_k_inv[i * n1 + j] = self.k_inv[i * n + j] + inv_schur * u[i] * u[j];
            }
        }
        // Right column and bottom row
        for i in 0..n {
            new_k_inv[i * n1 + n] = -inv_schur * u[i];
            new_k_inv[n * n1 + i] = -inv_schur * u[i];
        }
        // Bottom-right corner
        new_k_inv[n * n1 + n] = inv_schur;

        // Push the new data
        self.xs.push(x_new);
        self.ys.push(y_new);
        self.k_inv = new_k_inv;

        // Recompute alpha = K_inv * y
        self.alpha = self.mat_vec_mul(&self.k_inv, &self.ys, n1);
        Ok(())
    }

    // -- Public API ---------------------------------------------------------

    /// Add a single data point.
    pub fn add_point(&mut self, x: f64, y: f64) -> InterpolateResult<()> {
        if !x.is_finite() || !y.is_finite() {
            return Err(InterpolateError::InvalidInput {
                message: "Non-finite value in add_point".to_string(),
            });
        }

        #[allow(unreachable_patterns)]
        match self.config.update_strategy {
            UpdateStrategy::ShermanMorrison => {
                self.sherman_morrison_add(x, y)?;
            }
            UpdateStrategy::FullRecompute => {
                self.xs.push(x);
                self.ys.push(y);
                self.full_recompute()?;
            }
            UpdateStrategy::PartialUpdate => {
                self.xs.push(x);
                self.ys.push(y);
                self.points_since_recompute += 1;
                if self.points_since_recompute >= self.partial_update_threshold
                    || self.xs.len() <= 3
                {
                    self.full_recompute()?;
                }
            }
            _ => {
                // Fallback for future variants
                self.xs.push(x);
                self.ys.push(y);
                self.full_recompute()?;
            }
        }

        // Enforce window/max_points
        if self.config.window_mode || self.xs.len() > self.config.max_points {
            while self.xs.len() > self.config.max_points {
                self.remove_oldest(1)?;
            }
        }
        Ok(())
    }

    /// Add a batch of data points.
    pub fn add_batch(&mut self, xs: &[f64], ys: &[f64]) -> InterpolateResult<()> {
        if xs.len() != ys.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "xs len {} != ys len {}",
                xs.len(),
                ys.len()
            )));
        }
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            self.add_point(x, y)?;
        }
        Ok(())
    }

    /// Remove the oldest `n` data points and recompute the model.
    pub fn remove_oldest(&mut self, n: usize) -> InterpolateResult<()> {
        let to_remove = n.min(self.xs.len());
        if to_remove == 0 {
            return Ok(());
        }
        // Drain from the front
        self.xs.drain(..to_remove);
        self.ys.drain(..to_remove);
        // Must recompute after removal
        self.full_recompute()
    }

    /// Predict the interpolated value at `x`.
    pub fn predict(&self, x: f64) -> InterpolateResult<f64> {
        if self.xs.is_empty() {
            return Err(InterpolateError::InsufficientData(
                "No data points in online interpolator".to_string(),
            ));
        }
        // alpha may be shorter than xs when using PartialUpdate and model
        // has not yet been recomputed.  Use the valid prefix of alpha if
        // available, otherwise fall back to IDW.
        let n_alpha = self.alpha.len();
        if n_alpha == self.xs.len() && n_alpha > 0 {
            let mut result = 0.0;
            for i in 0..n_alpha {
                result += self.alpha[i] * self.kernel(x, self.xs[i]);
            }
            Ok(result)
        } else if n_alpha > 0 {
            // Use stale model (first n_alpha points) as approximation.
            let mut result = 0.0;
            for i in 0..n_alpha {
                result += self.alpha[i] * self.kernel(x, self.xs[i]);
            }
            Ok(result)
        } else {
            // No model at all – simple IDW fallback.
            let mut numer = 0.0_f64;
            let mut denom = 0.0_f64;
            for i in 0..self.xs.len() {
                let d2 = (x - self.xs[i]).powi(2) + 1e-20;
                let w = 1.0 / d2;
                numer += w * self.ys[i];
                denom += w;
            }
            if denom > 1e-300 {
                Ok(numer / denom)
            } else {
                Ok(0.0)
            }
        }
    }

    /// Predict values at multiple query points.
    pub fn predict_batch(&self, xs: &[f64]) -> InterpolateResult<Vec<f64>> {
        let mut out = Vec::with_capacity(xs.len());
        for &x in xs {
            out.push(self.predict(x)?);
        }
        Ok(out)
    }

    /// Return a snapshot of the stored x-coordinates.
    pub fn x_data(&self) -> &[f64] {
        &self.xs
    }

    /// Return a snapshot of the stored y-coordinates.
    pub fn y_data(&self) -> &[f64] {
        &self.ys
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> OnlineConfig {
        OnlineConfig {
            max_points: 100,
            window_mode: false,
            update_strategy: UpdateStrategy::ShermanMorrison,
        }
    }

    #[test]
    fn test_add_points_incrementally_predict_matches_batch() {
        // Build an interpolator incrementally, compare with a full-recompute one.
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [0.0, 1.0, 4.0, 9.0, 16.0]; // y = x^2

        let mut incr = OnlineRbfInterpolator::new(default_config(), 1.5);
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            incr.add_point(x, y).expect("add_point");
        }

        let mut batch = OnlineRbfInterpolator::new(
            OnlineConfig {
                update_strategy: UpdateStrategy::FullRecompute,
                ..default_config()
            },
            1.5,
        );
        batch.add_batch(&xs, &ys).expect("add_batch");

        // Predictions should be close.
        for &xq in &[0.5, 1.5, 2.5, 3.5] {
            let v_incr = incr.predict(xq).expect("predict incr");
            let v_batch = batch.predict(xq).expect("predict batch");
            assert!(
                (v_incr - v_batch).abs() < 0.5,
                "Mismatch at x={xq}: incr={v_incr}, batch={v_batch}"
            );
        }
    }

    #[test]
    fn test_window_mode_drops_old_points() {
        let mut interp = OnlineRbfInterpolator::new(
            OnlineConfig {
                max_points: 5,
                window_mode: true,
                update_strategy: UpdateStrategy::FullRecompute,
            },
            1.0,
        );
        for i in 0..10 {
            interp.add_point(i as f64, (i * i) as f64).expect("add");
        }
        assert!(interp.len() <= 5);
        // The oldest points should have been dropped.
        assert!(
            interp.x_data()[0] >= 5.0,
            "Oldest x should be >= 5, got {}",
            interp.x_data()[0]
        );
    }

    #[test]
    fn test_sherman_morrison_matches_full_recompute() {
        let xs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let ys = [1.0, 1.25, 2.0, 3.25, 5.0, 7.25, 10.0];

        let mut sm = OnlineRbfInterpolator::new(
            OnlineConfig {
                update_strategy: UpdateStrategy::ShermanMorrison,
                ..default_config()
            },
            1.0,
        );
        let mut fc = OnlineRbfInterpolator::new(
            OnlineConfig {
                update_strategy: UpdateStrategy::FullRecompute,
                ..default_config()
            },
            1.0,
        );

        for (&x, &y) in xs.iter().zip(ys.iter()) {
            sm.add_point(x, y).expect("sm add");
            fc.add_point(x, y).expect("fc add");
        }

        for &xq in &[0.25, 0.75, 1.25, 1.75, 2.25, 2.75] {
            let v_sm = sm.predict(xq).expect("sm predict");
            let v_fc = fc.predict(xq).expect("fc predict");
            assert!(
                (v_sm - v_fc).abs() < 1e-6,
                "SM vs FC mismatch at x={xq}: sm={v_sm}, fc={v_fc}"
            );
        }
    }

    #[test]
    fn test_remove_oldest() {
        let mut interp = OnlineRbfInterpolator::new(default_config(), 1.0);
        for i in 0..8 {
            interp.add_point(i as f64, (i * i) as f64).expect("add");
        }
        assert_eq!(interp.len(), 8);
        interp.remove_oldest(3).expect("remove");
        assert_eq!(interp.len(), 5);
        assert!(
            (interp.x_data()[0] - 3.0).abs() < 1e-12,
            "After removing 3, first x should be 3.0, got {}",
            interp.x_data()[0]
        );
    }

    #[test]
    fn test_add_batch_dimension_mismatch() {
        let mut interp = OnlineRbfInterpolator::new(default_config(), 1.0);
        let result = interp.add_batch(&[1.0, 2.0], &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_empty() {
        let interp = OnlineRbfInterpolator::new(default_config(), 1.0);
        let result = interp.predict(0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_non_finite_input_rejected() {
        let mut interp = OnlineRbfInterpolator::new(default_config(), 1.0);
        assert!(interp.add_point(f64::NAN, 1.0).is_err());
        assert!(interp.add_point(1.0, f64::INFINITY).is_err());
    }

    #[test]
    fn test_single_point_predict() {
        let mut interp = OnlineRbfInterpolator::new(default_config(), 1.0);
        interp.add_point(2.0, 5.0).expect("add");
        let val = interp.predict(2.0).expect("predict");
        assert!(
            (val - 5.0).abs() < 1e-6,
            "Single-point predict at x=2 should be ~5, got {val}"
        );
    }

    #[test]
    fn test_partial_update_strategy() {
        let mut interp = OnlineRbfInterpolator::new(
            OnlineConfig {
                max_points: 100,
                window_mode: false,
                update_strategy: UpdateStrategy::PartialUpdate,
            },
            1.0,
        );
        for i in 0..25 {
            interp
                .add_point(i as f64 * 0.2, (i as f64 * 0.2).sin())
                .expect("add");
        }
        let val = interp.predict(1.0).expect("predict");
        let expected = 1.0_f64.sin();
        assert!(
            (val - expected).abs() < 0.5,
            "Partial update predict at x=1.0: got {val}, expected ~{expected}"
        );
    }

    #[test]
    fn test_predict_batch() {
        let mut interp = OnlineRbfInterpolator::new(default_config(), 1.0);
        interp
            .add_batch(&[0.0, 1.0, 2.0], &[0.0, 1.0, 4.0])
            .expect("add_batch");
        let vals = interp.predict_batch(&[0.5, 1.5]).expect("predict_batch");
        assert_eq!(vals.len(), 2);
        assert!(vals[0] > 0.0 && vals[0] < 1.5);
        assert!(vals[1] > 1.0 && vals[1] < 4.0);
    }

    #[test]
    fn test_default_config() {
        let config = OnlineConfig::default();
        assert_eq!(config.max_points, 1000);
        assert!(!config.window_mode);
        assert_eq!(config.update_strategy, UpdateStrategy::ShermanMorrison);
    }

    #[test]
    fn test_update_strategy_default() {
        let strategy = UpdateStrategy::default();
        assert_eq!(strategy, UpdateStrategy::ShermanMorrison);
    }
}
