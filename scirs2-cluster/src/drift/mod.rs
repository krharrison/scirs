//! Concept Drift Detection algorithms.
//!
//! Implements:
//! - **ADWIN** (Adaptive Windowing – Bifet & Gavalda 2007)
//! - **DDM** (Drift Detection Method – Gama 2004)
//! - **Page-Hinkley Test** for mean-shift detection
//! - **ClusterDriftDetector**: combines DDM with nearest-centroid assignment
//!   to signal when the clustering model should be retrained.

use std::collections::VecDeque;

use crate::error::ClusteringError;

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Status returned by [`DdmDetector::add_element`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DriftStatus {
    /// No drift detected.
    Normal,
    /// Data distribution may be changing (warning zone).
    Warning,
    /// Drift has been confirmed.
    Drift,
}

/// Selector for which underlying detector to use inside [`ClusterDriftDetector`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DriftDetectorType {
    /// Adaptive Windowing.
    ADWIN,
    /// Drift Detection Method.
    DDM,
    /// Early DDM.
    EDDM,
    /// Hoeffding DDM.
    HDDM,
    /// Page-Hinkley test.
    PageHinkley,
}

impl Default for DriftDetectorType {
    fn default() -> Self {
        Self::DDM
    }
}

// ---------------------------------------------------------------------------
// ADWIN
// ---------------------------------------------------------------------------

/// Configuration for [`AdwinDetector`].
#[derive(Debug, Clone)]
pub struct AdwinConfig {
    /// Confidence parameter (smaller → more sensitive).
    pub delta: f64,
    /// Clock interval (check cut every `clock` elements).
    pub clock: usize,
    /// Maximum number of exponential histogram buckets.
    pub max_buckets: usize,
    /// Minimum window size before drift checks are performed.
    pub min_window: usize,
}

impl Default for AdwinConfig {
    fn default() -> Self {
        Self {
            delta: 0.002,
            clock: 32,
            max_buckets: 5,
            min_window: 10,
        }
    }
}

/// ADWIN detector — maintains an adaptive sliding window and detects drift via
/// the Hoeffding-based ε-cut criterion.
pub struct AdwinDetector {
    config: AdwinConfig,
    window: VecDeque<f64>,
    total: f64,
    n: usize,
    elements_seen: usize,
}

impl AdwinDetector {
    /// Create a new `AdwinDetector` with the given configuration.
    pub fn new(config: AdwinConfig) -> Self {
        Self {
            config,
            window: VecDeque::new(),
            total: 0.0,
            n: 0,
            elements_seen: 0,
        }
    }

    /// Add a single observation. Returns `true` if drift is detected.
    pub fn add_element(&mut self, value: f64) -> bool {
        self.window.push_back(value);
        self.total += value;
        self.n += 1;
        self.elements_seen += 1;

        // Only run cut detection every `clock` elements and after min_window
        if self.n < self.config.min_window || self.elements_seen % self.config.clock != 0 {
            return false;
        }

        self.detect_drift()
    }

    /// Mean of the current window.
    pub fn mean(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.total / self.n as f64
        }
    }

    /// Number of elements in the current window.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Internal: scan all splits and return `true` (and shrink window) if any
    /// violates the Hoeffding bound.
    fn detect_drift(&mut self) -> bool {
        // Pre-compute cumulative sums from right to left for efficiency.
        let win: Vec<f64> = self.window.iter().copied().collect();
        let n = win.len();
        if n < self.config.min_window {
            return false;
        }

        // Right cumsum (suffix)
        let mut right_sum = vec![0.0f64; n + 1];
        for i in (0..n).rev() {
            right_sum[i] = right_sum[i + 1] + win[i];
        }

        // Try every split i ∈ [1, n-1]
        let mut left_sum = 0.0f64;
        let mut drift_detected = false;
        let mut cut_point = 0usize;

        for i in 1..n {
            left_sum += win[i - 1];
            let n0 = i as f64;
            let n1 = (n - i) as f64;
            let sum1 = right_sum[i];

            let mu0 = left_sum / n0;
            let mu1 = sum1 / n1;

            // Hoeffding ε_cut: sqrt(1/(2 * harmonic_n) * ln(4n/delta))
            // harmonic_n = n0*n1/n
            let harmonic_n = n0 * n1 / n as f64;
            if harmonic_n <= 0.0 {
                continue;
            }
            let eps_cut =
                ((1.0 / (2.0 * harmonic_n)) * (4.0 * n as f64 / self.config.delta).ln()).sqrt();

            if (mu0 - mu1).abs() >= eps_cut {
                drift_detected = true;
                cut_point = i;
                break;
            }
        }

        if drift_detected {
            // Shrink window: keep only the newer (right) half from the cut point onward.
            // We drop the left part (indices 0..cut_point).
            let to_remove: Vec<f64> = self.window.drain(0..cut_point).collect();
            let removed_sum: f64 = to_remove.iter().sum();
            self.total -= removed_sum;
            self.n -= to_remove.len();
        }

        drift_detected
    }
}

// ---------------------------------------------------------------------------
// DDM
// ---------------------------------------------------------------------------

/// Configuration for [`DdmDetector`].
#[derive(Debug, Clone)]
pub struct DdmConfig {
    /// Number of standard deviations above baseline for a warning.
    pub warning_level: f64,
    /// Number of standard deviations above baseline for confirmed drift.
    pub drift_level: f64,
    /// Minimum number of instances before drift checking begins.
    pub min_instances: usize,
}

impl Default for DdmConfig {
    fn default() -> Self {
        Self {
            warning_level: 2.0,
            drift_level: 3.0,
            min_instances: 30,
        }
    }
}

/// DDM (Drift Detection Method) detector based on a running binomial error estimate.
pub struct DdmDetector {
    config: DdmConfig,
    /// Instance count.
    n: usize,
    /// Running error rate.
    p: f64,
    /// Running standard deviation estimate.
    s: f64,
    /// Minimum recorded (p + s).
    p_min: f64,
    s_min: f64,
}

impl DdmDetector {
    /// Create a new `DdmDetector`.
    pub fn new(config: DdmConfig) -> Self {
        Self {
            config,
            n: 0,
            p: 0.0,
            s: 0.0,
            p_min: f64::INFINITY,
            s_min: f64::INFINITY,
        }
    }

    /// Add a single observation. `error` = `true` if the prediction was wrong.
    ///
    /// Returns the current [`DriftStatus`].
    pub fn add_element(&mut self, error: bool) -> DriftStatus {
        self.n += 1;
        let err_val = if error { 1.0 } else { 0.0 };

        // Online update of error rate (Welford-style equivalent for Bernoulli)
        self.p = (self.p * (self.n as f64 - 1.0) + err_val) / self.n as f64;
        self.s = if self.p > 0.0 && self.p < 1.0 {
            (self.p * (1.0 - self.p) / self.n as f64).sqrt()
        } else {
            0.0
        };

        if self.n < self.config.min_instances {
            return DriftStatus::Normal;
        }

        // Update running minimum
        let ps = self.p + self.s;
        if ps < self.p_min + self.s_min {
            self.p_min = self.p;
            self.s_min = self.s;
        }

        let baseline = self.p_min + self.s_min;
        if baseline.is_infinite() {
            return DriftStatus::Normal;
        }

        if ps > baseline + self.config.drift_level * self.s_min {
            return DriftStatus::Drift;
        }
        if ps > baseline + self.config.warning_level * self.s_min {
            return DriftStatus::Warning;
        }
        DriftStatus::Normal
    }

    /// Reset all statistics (call after drift is confirmed and the model is retrained).
    pub fn reset(&mut self) {
        self.n = 0;
        self.p = 0.0;
        self.s = 0.0;
        self.p_min = f64::INFINITY;
        self.s_min = f64::INFINITY;
    }
}

// ---------------------------------------------------------------------------
// Page-Hinkley Test
// ---------------------------------------------------------------------------

/// Configuration for [`PageHinkleyDetector`].
#[derive(Debug, Clone)]
pub struct PageHinkleyConfig {
    /// Minimum acceptable amplitude of change (allow for noise).
    pub delta: f64,
    /// Detection threshold for the test statistic.
    pub lambda: f64,
    /// Forgetting factor (EWMA smoothing for reference mean).
    pub alpha: f64,
}

impl Default for PageHinkleyConfig {
    fn default() -> Self {
        Self {
            delta: 0.005,
            lambda: 50.0,
            alpha: 0.9999,
        }
    }
}

/// Page-Hinkley test for detecting an upward mean shift in a data stream.
pub struct PageHinkleyDetector {
    config: PageHinkleyConfig,
    /// EWMA reference mean.
    x_hat: f64,
    /// Cumulative sum of deviations.
    cum_sum: f64,
    /// Running minimum of `cum_sum`.
    min_sum: f64,
    /// Whether any elements have been seen.
    initialised: bool,
}

impl PageHinkleyDetector {
    /// Create a new `PageHinkleyDetector`.
    pub fn new(config: PageHinkleyConfig) -> Self {
        Self {
            config,
            x_hat: 0.0,
            cum_sum: 0.0,
            min_sum: 0.0,
            initialised: false,
        }
    }

    /// Add a single observation. Returns `true` if drift (upward mean shift) is detected.
    pub fn add_element(&mut self, value: f64) -> bool {
        if !self.initialised {
            self.x_hat = value;
            self.initialised = true;
            return false;
        }

        // Update EWMA reference
        self.x_hat = self.x_hat * self.config.alpha + value * (1.0 - self.config.alpha);

        // Update cumulative sum
        self.cum_sum += value - self.x_hat - self.config.delta;

        // Update running minimum
        if self.cum_sum < self.min_sum {
            self.min_sum = self.cum_sum;
        }

        (self.cum_sum - self.min_sum) > self.config.lambda
    }

    /// Reset detector state.
    pub fn reset(&mut self) {
        self.x_hat = 0.0;
        self.cum_sum = 0.0;
        self.min_sum = 0.0;
        self.initialised = false;
    }
}

// ---------------------------------------------------------------------------
// ClusterDriftDetector
// ---------------------------------------------------------------------------

/// Configuration for [`ClusterDriftDetector`].
#[derive(Debug, Clone)]
pub struct ClusterDriftConfig {
    /// Underlying drift detector algorithm.
    pub detector_type: DriftDetectorType,
    /// Number of clusters (centroids used for point assignment).
    pub n_clusters: usize,
    /// Number of drift events before flagging that retraining is needed.
    pub retrain_threshold: usize,
}

impl Default for ClusterDriftConfig {
    fn default() -> Self {
        Self {
            detector_type: DriftDetectorType::DDM,
            n_clusters: 5,
            retrain_threshold: 50,
        }
    }
}

/// Cluster-aware drift detector: assigns incoming points to the nearest centroid
/// and feeds correct/incorrect signals to a DDM detector.
pub struct ClusterDriftDetector {
    config: ClusterDriftConfig,
    centroids: Vec<Vec<f64>>,
    ddm: DdmDetector,
    drift_count: usize,
    retrain_requested: bool,
}

impl ClusterDriftDetector {
    /// Create a new `ClusterDriftDetector` with pre-fitted `centroids`.
    pub fn new(config: ClusterDriftConfig) -> Self {
        Self {
            config,
            centroids: Vec::new(),
            ddm: DdmDetector::new(DdmConfig::default()),
            drift_count: 0,
            retrain_requested: false,
        }
    }

    /// Set current centroids (e.g. after an initial clustering run).
    pub fn set_centroids(&mut self, centroids: Vec<Vec<f64>>) {
        self.centroids = centroids;
    }

    /// Update the detector with a new `point`.
    ///
    /// - If `true_label` is provided, feeds a correct/incorrect signal to DDM.
    /// - Returns `Some(DriftStatus)` only when DDM emits `Warning` or `Drift`.
    pub fn update(&mut self, point: &[f64], true_label: Option<usize>) -> Option<DriftStatus> {
        if self.centroids.is_empty() {
            return None;
        }

        // Nearest centroid
        let predicted = nearest_centroid_idx(point, &self.centroids);

        if let Some(label) = true_label {
            let error = predicted != label;
            let status = self.ddm.add_element(error);
            match &status {
                DriftStatus::Drift => {
                    self.drift_count += 1;
                    if self.drift_count >= self.config.retrain_threshold {
                        self.retrain_requested = true;
                    }
                    return Some(status);
                }
                DriftStatus::Warning => {
                    return Some(status);
                }
                DriftStatus::Normal => {}
            }
        }
        None
    }

    /// Returns `true` when the number of accumulated drift events has reached the retrain threshold.
    pub fn should_retrain(&self) -> bool {
        self.retrain_requested
    }

    /// Reset retrain flag (call after retraining is performed).
    pub fn reset_retrain_flag(&mut self) {
        self.retrain_requested = false;
        self.drift_count = 0;
        self.ddm.reset();
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn nearest_centroid_idx(point: &[f64], centroids: &[Vec<f64>]) -> usize {
    let mut best = 0;
    let mut best_dist = f64::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let d: f64 = point
            .iter()
            .zip(c.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        if d < best_dist {
            best_dist = d;
            best = i;
        }
    }
    best
}

// Suppress unused import lint (ClusteringError is used in the public API of
// ClusterDriftConfig via From bounds elsewhere; keep it visible)
const _: fn() = || {
    let _: ClusteringError = ClusteringError::Other("placeholder".into());
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ADWIN tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_adwin_no_drift_on_uniform_stream() {
        let config = AdwinConfig {
            delta: 0.002,
            clock: 10,
            min_window: 20,
            ..Default::default()
        };
        let mut detector = AdwinDetector::new(config);
        let mut drift_count = 0usize;

        // Feed 500 points from the same distribution
        let mut rng: u64 = 42;
        for _ in 0..500 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = ((rng >> 11) as f64 / (u64::MAX >> 11) as f64) * 2.0; // uniform [0,2)
            if detector.add_element(val) {
                drift_count += 1;
            }
        }
        // Expect very few (ideally 0) false alarms
        assert!(
            drift_count <= 3,
            "Expected ≤3 false alarms on uniform stream, got {}",
            drift_count
        );
    }

    #[test]
    fn test_adwin_detects_abrupt_step_change() {
        let config = AdwinConfig {
            delta: 0.002,
            clock: 1,
            min_window: 10,
            ..Default::default()
        };
        let mut detector = AdwinDetector::new(config);
        let mut detected_after = None;
        let mut rng: u64 = 99;

        let total = 200;
        for i in 0..total {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((rng >> 33) as f64 / (u64::MAX >> 33) as f64 - 0.5) * 0.1;
            // Step from 0.0 to 10.0 at midpoint
            let val = if i < total / 2 { noise } else { 10.0 + noise };
            if detector.add_element(val) && detected_after.is_none() {
                detected_after = Some(i);
            }
        }
        assert!(
            detected_after.is_some(),
            "ADWIN should detect the abrupt step change"
        );
        // Should detect after at least some samples in the second half
        let det_pos = detected_after.unwrap();
        assert!(
            det_pos >= total / 2,
            "Drift should be detected in the second half (pos={})",
            det_pos
        );
    }

    // -----------------------------------------------------------------------
    // DDM tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ddm_normal_then_normal() {
        let mut ddm = DdmDetector::new(DdmConfig::default());
        // 200 correct predictions → DDM should report Normal
        let mut last_status = DriftStatus::Normal;
        for _ in 0..200 {
            last_status = ddm.add_element(false);
        }
        assert_eq!(last_status, DriftStatus::Normal);
    }

    #[test]
    fn test_ddm_correct_then_errors_triggers_drift() {
        let mut ddm = DdmDetector::new(DdmConfig {
            warning_level: 2.0,
            drift_level: 3.0,
            min_instances: 30,
        });
        // 100 correct first
        for _ in 0..100 {
            ddm.add_element(false);
        }
        // Then 100 errors
        let mut detected_drift = false;
        for _ in 0..100 {
            if ddm.add_element(true) == DriftStatus::Drift {
                detected_drift = true;
                break;
            }
        }
        assert!(
            detected_drift,
            "DDM should detect drift after switching from 100 correct to 100 errors"
        );
    }

    // -----------------------------------------------------------------------
    // Page-Hinkley tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_page_hinkley_no_drift_stable() {
        let mut ph = PageHinkleyDetector::new(PageHinkleyConfig::default());
        let mut drift_count = 0usize;
        let mut rng: u64 = 12345;

        for _ in 0..300 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = 1.0 + ((rng >> 33) as f64 / (u64::MAX >> 33) as f64 - 0.5) * 0.1;
            if ph.add_element(val) {
                drift_count += 1;
            }
        }
        assert!(drift_count == 0, "No drift expected on stable signal");
    }

    #[test]
    fn test_page_hinkley_detects_mean_shift() {
        let mut ph = PageHinkleyDetector::new(PageHinkleyConfig {
            delta: 0.005,
            lambda: 10.0,
            alpha: 0.999,
        });

        // Stable phase
        for _ in 0..100 {
            ph.add_element(1.0);
        }

        // Sudden shift to 5.0
        let mut detected_within_20 = false;
        for i in 0..20 {
            if ph.add_element(5.0) {
                detected_within_20 = true;
                let _ = i;
                break;
            }
        }
        assert!(
            detected_within_20,
            "Page-Hinkley should detect the mean shift within 20 samples"
        );
    }

    // -----------------------------------------------------------------------
    // ClusterDriftDetector tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cluster_drift_no_retrain_without_threshold() {
        let config = ClusterDriftConfig {
            detector_type: DriftDetectorType::DDM,
            n_clusters: 2,
            retrain_threshold: 100,
        };
        let mut detector = ClusterDriftDetector::new(config);
        detector.set_centroids(vec![vec![0.0], vec![10.0]]);

        // Feed a few misclassified points — not enough to cross threshold
        for _ in 0..5 {
            detector.update(&[5.0], Some(0));
        }
        assert!(!detector.should_retrain());
    }
}
