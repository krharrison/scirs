//! Calibration Observers for Quantization
//!
//! Provides observer patterns for collecting tensor statistics during
//! calibration passes. These statistics determine optimal quantization
//! parameters (scale and zero_point).
//!
//! Observers:
//! - **MinMaxObserver**: Tracks global min/max values
//! - **PercentileObserver**: Uses percentile clipping (e.g., 99.99th)
//! - **MovingAverageObserver**: Exponential moving average of min/max

use crate::error::{Error, Result};
use scirs2_core::ndarray::{ArrayD, ArrayView, IxDyn};

/// Statistics collected by an observer
#[derive(Debug, Clone)]
pub struct ObserverStats {
    /// Observed minimum value
    pub min_val: f64,
    /// Observed maximum value
    pub max_val: f64,
    /// Number of samples observed
    pub n_samples: usize,
    /// Running mean (for diagnostics)
    pub mean: f64,
}

impl Default for ObserverStats {
    fn default() -> Self {
        Self {
            min_val: f64::INFINITY,
            max_val: f64::NEG_INFINITY,
            n_samples: 0,
            mean: 0.0,
        }
    }
}

/// Trait for quantization calibration observers
pub trait Observer {
    /// Feed a tensor to the observer
    fn observe(&mut self, tensor: &ArrayD<f64>);

    /// Get the current statistics
    fn stats(&self) -> &ObserverStats;

    /// Compute scale and zero_point for symmetric quantization at the given bit-width
    fn compute_symmetric_params(&self, bits: u8) -> (f64, i32) {
        let s = self.stats();
        let abs_max = s.max_val.abs().max(s.min_val.abs());
        let qmax = (1i64 << (bits as i64 - 1)) - 1;
        let scale = if qmax > 0 && abs_max > 0.0 {
            abs_max / qmax as f64
        } else {
            1.0
        };
        (scale, 0)
    }

    /// Compute scale and zero_point for asymmetric quantization at the given bit-width
    fn compute_asymmetric_params(&self, bits: u8) -> (f64, i32) {
        let s = self.stats();
        let qmax = (1i64 << bits as i64) - 1;
        let range = s.max_val - s.min_val;
        let scale = if qmax > 0 && range > 0.0 {
            range / qmax as f64
        } else {
            1.0
        };
        let zero_point = if scale > 0.0 {
            (-s.min_val / scale).round() as i32
        } else {
            0
        };
        (scale, zero_point.clamp(0, qmax as i32))
    }

    /// Reset the observer state
    fn reset(&mut self);
}

// ---------------------------------------------------------------------------
// MinMaxObserver
// ---------------------------------------------------------------------------

/// Tracks the global minimum and maximum values seen across all observations.
#[derive(Debug, Clone)]
pub struct MinMaxObserver {
    stats: ObserverStats,
}

impl MinMaxObserver {
    /// Create a new MinMax observer
    pub fn new() -> Self {
        Self {
            stats: ObserverStats::default(),
        }
    }
}

impl Default for MinMaxObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl Observer for MinMaxObserver {
    fn observe(&mut self, tensor: &ArrayD<f64>) {
        if tensor.is_empty() {
            return;
        }
        let min_val = tensor.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = tensor.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        self.stats.min_val = self.stats.min_val.min(min_val);
        self.stats.max_val = self.stats.max_val.max(max_val);
        self.stats.n_samples += 1;

        // Update running mean
        let tensor_mean = tensor.iter().sum::<f64>() / tensor.len() as f64;
        let n = self.stats.n_samples as f64;
        self.stats.mean = self.stats.mean * ((n - 1.0) / n) + tensor_mean / n;
    }

    fn stats(&self) -> &ObserverStats {
        &self.stats
    }

    fn reset(&mut self) {
        self.stats = ObserverStats::default();
    }
}

// ---------------------------------------------------------------------------
// PercentileObserver
// ---------------------------------------------------------------------------

/// Uses percentile-based clipping to determine the calibration range.
///
/// Instead of using the raw min/max (which can be dominated by outliers),
/// this observer clips at a configurable percentile (e.g., 99.99th).
#[derive(Debug, Clone)]
pub struct PercentileObserver {
    stats: ObserverStats,
    /// Percentile for clipping (e.g., 99.99)
    percentile: f64,
    /// All observed values (sorted lazily on stats query)
    all_values: Vec<f64>,
    /// Whether the computed stats are stale
    dirty: bool,
}

impl PercentileObserver {
    /// Create a new percentile observer
    ///
    /// # Arguments
    /// * `percentile` - The percentile to use for clipping (0.0 to 100.0).
    ///   For example, 99.99 means clip at the 0.01th and 99.99th percentiles.
    pub fn new(percentile: f64) -> Result<Self> {
        if !(0.0..=100.0).contains(&percentile) {
            return Err(Error::InvalidArgument(format!(
                "Percentile must be in [0, 100], got {}",
                percentile
            )));
        }
        Ok(Self {
            stats: ObserverStats::default(),
            percentile,
            all_values: Vec::new(),
            dirty: false,
        })
    }

    /// Recompute percentile-based min/max from collected values
    fn recompute(&mut self) {
        if self.all_values.is_empty() {
            return;
        }
        self.all_values
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = self.all_values.len();
        let lower_pct = (100.0 - self.percentile) / 100.0;
        let upper_pct = self.percentile / 100.0;

        let lower_idx = ((n as f64 * lower_pct).floor() as usize).min(n - 1);
        let upper_idx = ((n as f64 * upper_pct).ceil() as usize).min(n - 1);

        self.stats.min_val = self.all_values[lower_idx];
        self.stats.max_val = self.all_values[upper_idx];

        let sum: f64 = self.all_values.iter().sum();
        self.stats.mean = sum / n as f64;

        self.dirty = false;
    }
}

impl Observer for PercentileObserver {
    fn observe(&mut self, tensor: &ArrayD<f64>) {
        if tensor.is_empty() {
            return;
        }
        self.all_values.extend(tensor.iter().cloned());
        self.stats.n_samples += 1;
        self.dirty = true;
    }

    fn stats(&self) -> &ObserverStats {
        if self.dirty {
            // Need mutable access; use interior trick via clone
            // In practice the caller should call recompute(), but
            // to satisfy the trait we return current state
            &self.stats
        } else {
            &self.stats
        }
    }

    fn reset(&mut self) {
        self.stats = ObserverStats::default();
        self.all_values.clear();
        self.dirty = false;
    }
}

impl PercentileObserver {
    /// Finalize and return updated stats (call this after all observations)
    pub fn finalize(&mut self) -> &ObserverStats {
        if self.dirty {
            self.recompute();
        }
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// MovingAverageObserver
// ---------------------------------------------------------------------------

/// Uses exponential moving average of min/max values.
///
/// min_ema = alpha * current_min + (1 - alpha) * prev_min_ema
/// max_ema = alpha * current_max + (1 - alpha) * prev_max_ema
#[derive(Debug, Clone)]
pub struct MovingAverageObserver {
    stats: ObserverStats,
    /// Smoothing factor (0 < alpha <= 1)
    alpha: f64,
}

impl MovingAverageObserver {
    /// Create a new moving average observer
    ///
    /// # Arguments
    /// * `alpha` - Smoothing factor in (0, 1]. Higher values give more weight
    ///   to recent observations. Common values: 0.01 to 0.1.
    pub fn new(alpha: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(Error::InvalidArgument(format!(
                "Alpha must be in (0, 1], got {}",
                alpha
            )));
        }
        Ok(Self {
            stats: ObserverStats::default(),
            alpha,
        })
    }
}

impl Observer for MovingAverageObserver {
    fn observe(&mut self, tensor: &ArrayD<f64>) {
        if tensor.is_empty() {
            return;
        }
        let current_min = tensor.iter().cloned().fold(f64::INFINITY, f64::min);
        let current_max = tensor.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if self.stats.n_samples == 0 {
            // First observation: initialize directly
            self.stats.min_val = current_min;
            self.stats.max_val = current_max;
        } else {
            self.stats.min_val = self.alpha * current_min + (1.0 - self.alpha) * self.stats.min_val;
            self.stats.max_val = self.alpha * current_max + (1.0 - self.alpha) * self.stats.max_val;
        }
        self.stats.n_samples += 1;

        // Update running mean
        let tensor_mean = tensor.iter().sum::<f64>() / tensor.len() as f64;
        let n = self.stats.n_samples as f64;
        self.stats.mean = self.stats.mean * ((n - 1.0) / n) + tensor_mean / n;
    }

    fn stats(&self) -> &ObserverStats {
        &self.stats
    }

    fn reset(&mut self) {
        self.stats = ObserverStats::default();
    }
}

// ---------------------------------------------------------------------------
// CalibrationRunner: convenience struct that applies an observer to many tensors
// ---------------------------------------------------------------------------

/// Collects calibration statistics for multiple named tensors
pub struct CalibrationRunner<O: Observer> {
    observers: std::collections::HashMap<String, O>,
    /// Factory function stored as a trait is complex; instead we store config
    observer_factory: fn() -> O,
}

/// Create a calibration runner for MinMax observers
pub fn minmax_calibration_runner() -> CalibrationRunner<MinMaxObserver> {
    CalibrationRunner {
        observers: std::collections::HashMap::new(),
        observer_factory: MinMaxObserver::new,
    }
}

impl<O: Observer> CalibrationRunner<O> {
    /// Create a new calibration runner with a custom observer factory
    pub fn new(factory: fn() -> O) -> Self {
        Self {
            observers: std::collections::HashMap::new(),
            observer_factory: factory,
        }
    }

    /// Feed a named tensor observation
    pub fn observe(&mut self, name: &str, tensor: &ArrayD<f64>) {
        let factory = self.observer_factory;
        let observer = self
            .observers
            .entry(name.to_string())
            .or_insert_with(factory);
        observer.observe(tensor);
    }

    /// Get all observer statistics
    pub fn all_stats(&self) -> std::collections::HashMap<String, ObserverStats> {
        self.observers
            .iter()
            .map(|(name, obs)| (name.clone(), obs.stats().clone()))
            .collect()
    }

    /// Get stats for a specific tensor
    pub fn get_stats(&self, name: &str) -> Option<&ObserverStats> {
        self.observers.get(name).map(|o| o.stats())
    }

    /// Get the observer for a specific tensor
    pub fn get_observer(&self, name: &str) -> Option<&O> {
        self.observers.get(name)
    }

    /// Reset all observers
    pub fn reset_all(&mut self) {
        for obs in self.observers.values_mut() {
            obs.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_minmax_observer_basic() {
        let mut obs = MinMaxObserver::new();
        let t1 = array![1.0_f64, -2.0, 3.0, 0.5].into_dyn();
        obs.observe(&t1);
        let s = obs.stats();
        assert_eq!(s.n_samples, 1);
        assert!((s.min_val - (-2.0)).abs() < 1e-10);
        assert!((s.max_val - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_minmax_observer_multiple() {
        let mut obs = MinMaxObserver::new();
        obs.observe(&array![1.0_f64, 2.0].into_dyn());
        obs.observe(&array![-5.0_f64, 0.0].into_dyn());
        obs.observe(&array![0.0_f64, 10.0].into_dyn());
        let s = obs.stats();
        assert_eq!(s.n_samples, 3);
        assert!((s.min_val - (-5.0)).abs() < 1e-10);
        assert!((s.max_val - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_minmax_symmetric_params() {
        let mut obs = MinMaxObserver::new();
        obs.observe(&array![-4.0_f64, 2.0, 0.0, 4.0].into_dyn());
        let (scale, zp) = obs.compute_symmetric_params(8);
        assert_eq!(zp, 0);
        // abs_max = 4.0, qmax = 127 => scale ~ 0.0315
        assert!((scale - 4.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn test_minmax_asymmetric_params() {
        let mut obs = MinMaxObserver::new();
        obs.observe(&array![0.0_f64, 1.0, 2.0, 3.0].into_dyn());
        let (scale, zp) = obs.compute_asymmetric_params(8);
        // range = 3.0, qmax = 255 => scale ~ 0.01176
        assert!(scale > 0.0);
        assert!(zp >= 0);
    }

    #[test]
    fn test_percentile_observer() {
        let mut obs = PercentileObserver::new(99.0).expect("test: create observer");
        // Add 1000 values: 0..999, plus one outlier at 10000
        let mut data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        data.push(10000.0);
        let tensor = ArrayD::from_shape_vec(IxDyn(&[1001]), data).expect("test: create tensor");
        obs.observe(&tensor);
        let stats = obs.finalize();
        // 99th percentile should clip the outlier
        assert!(
            stats.max_val < 10000.0,
            "max should be clipped: {}",
            stats.max_val
        );
        assert!(stats.min_val >= 0.0);
    }

    #[test]
    fn test_percentile_observer_invalid() {
        let result = PercentileObserver::new(101.0);
        assert!(result.is_err());
        let result = PercentileObserver::new(-1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_moving_average_observer() {
        let mut obs = MovingAverageObserver::new(0.1).expect("test: create observer");
        // First observation sets values directly
        obs.observe(&array![0.0_f64, 10.0].into_dyn());
        assert!((obs.stats().min_val - 0.0).abs() < 1e-10);
        assert!((obs.stats().max_val - 10.0).abs() < 1e-10);

        // Second observation: EMA
        obs.observe(&array![2.0_f64, 8.0].into_dyn());
        // min_ema = 0.1 * 2.0 + 0.9 * 0.0 = 0.2
        // max_ema = 0.1 * 8.0 + 0.9 * 10.0 = 9.8
        assert!((obs.stats().min_val - 0.2).abs() < 1e-10);
        assert!((obs.stats().max_val - 9.8).abs() < 1e-10);
    }

    #[test]
    fn test_moving_average_invalid_alpha() {
        assert!(MovingAverageObserver::new(0.0).is_err());
        assert!(MovingAverageObserver::new(1.5).is_err());
        assert!(MovingAverageObserver::new(-0.1).is_err());
    }

    #[test]
    fn test_observer_reset() {
        let mut obs = MinMaxObserver::new();
        obs.observe(&array![1.0_f64, 5.0].into_dyn());
        assert_eq!(obs.stats().n_samples, 1);
        obs.reset();
        assert_eq!(obs.stats().n_samples, 0);
        assert_eq!(obs.stats().min_val, f64::INFINITY);
    }

    #[test]
    fn test_calibration_runner() {
        let mut runner = minmax_calibration_runner();
        runner.observe("layer1", &array![1.0_f64, 2.0, 3.0].into_dyn());
        runner.observe("layer2", &array![-1.0_f64, 0.0, 1.0].into_dyn());
        runner.observe("layer1", &array![0.0_f64, 5.0].into_dyn());

        let s1 = runner.get_stats("layer1");
        assert!(s1.is_some());
        let s1 = s1.expect("test: stats");
        assert_eq!(s1.n_samples, 2);
        assert!((s1.max_val - 5.0).abs() < 1e-10);

        let s2 = runner.get_stats("layer2");
        assert!(s2.is_some());
        assert_eq!(s2.expect("test: stats").n_samples, 1);

        let all = runner.all_stats();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_calibration_runner_reset() {
        let mut runner = minmax_calibration_runner();
        runner.observe("l1", &array![1.0_f64].into_dyn());
        runner.reset_all();
        let s = runner.get_stats("l1");
        assert!(s.is_some());
        assert_eq!(s.expect("test: stats").n_samples, 0);
    }

    #[test]
    fn test_empty_tensor_observation() {
        let mut obs = MinMaxObserver::new();
        let empty = ArrayD::<f64>::zeros(IxDyn(&[0]));
        obs.observe(&empty);
        assert_eq!(obs.stats().n_samples, 0);
    }

    #[test]
    fn test_percentile_multiple_observations() {
        let mut obs = PercentileObserver::new(95.0).expect("test: create observer");
        for i in 0..10 {
            let data: Vec<f64> = (0..100).map(|j| (i * 100 + j) as f64).collect();
            let tensor = ArrayD::from_shape_vec(IxDyn(&[100]), data).expect("test: create tensor");
            obs.observe(&tensor);
        }
        let stats = obs.finalize();
        assert_eq!(stats.n_samples, 10);
        // 95th percentile of 0..999 should be around 949
        assert!(stats.max_val < 999.0);
        assert!(stats.max_val > 900.0);
    }
}
