//! Distribution drift detection algorithms
//!
//! Provides a [`DriftDetector`] trait and four concrete implementations:
//!
//! | Detector | Method | Multivariate? |
//! |----------|--------|---------------|
//! | [`KolmogorovSmirnovDetector`] | Two-sample KS test | No (1D) |
//! | [`PopulationStabilityIndexDetector`] | PSI via binning | No (1D) |
//! | [`WassersteinDetector`] | Earth-mover / Wasserstein-1 | No (1D) |
//! | [`MaximumMeanDiscrepancyDetector`] | Kernel MMD² | Yes |
//!
//! Each detector compares a *reference* window against a *test* window and
//! produces a [`DriftResult`] indicating whether drift was detected.

use crate::error::{Result, TransformError};

/// Result of a drift detection test.
#[derive(Debug, Clone)]
pub struct DriftResult {
    /// Whether drift was detected at the configured significance level.
    pub detected: bool,
    /// The test statistic.
    pub statistic: f64,
    /// P-value if available (not all methods provide one).
    pub p_value: Option<f64>,
    /// The threshold used for the decision.
    pub threshold: f64,
}

/// Trait for drift detectors that compare two 1-D sample arrays.
pub trait DriftDetector: Send + Sync {
    /// Compare a reference distribution against a test distribution.
    fn detect(&self, reference: &[f64], test: &[f64]) -> Result<DriftResult>;
}

// ---------------------------------------------------------------------------
// Kolmogorov-Smirnov detector
// ---------------------------------------------------------------------------

/// Two-sample Kolmogorov-Smirnov test for distribution shift.
///
/// The KS statistic is the maximum absolute difference between the empirical
/// CDFs of the two samples. The p-value is approximated via the asymptotic
/// distribution.
#[derive(Debug, Clone)]
pub struct KolmogorovSmirnovDetector {
    /// Significance level (default 0.05).
    significance_level: f64,
}

impl KolmogorovSmirnovDetector {
    /// Create a new KS detector with the given significance level.
    pub fn new(significance_level: f64) -> Result<Self> {
        if significance_level <= 0.0 || significance_level >= 1.0 {
            return Err(TransformError::InvalidInput(
                "significance_level must be in (0, 1)".to_string(),
            ));
        }
        Ok(Self { significance_level })
    }

    /// Create with default significance level 0.05.
    pub fn default_config() -> Self {
        Self {
            significance_level: 0.05,
        }
    }
}

impl DriftDetector for KolmogorovSmirnovDetector {
    fn detect(&self, reference: &[f64], test: &[f64]) -> Result<DriftResult> {
        if reference.is_empty() || test.is_empty() {
            return Err(TransformError::InvalidInput(
                "Reference and test samples must be non-empty".to_string(),
            ));
        }

        let mut ref_sorted: Vec<f64> = reference.to_vec();
        ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut test_sorted: Vec<f64> = test.to_vec();
        test_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n1 = reference.len() as f64;
        let n2 = test.len() as f64;

        // Merge both sorted arrays and compute max CDF difference
        let mut i = 0usize;
        let mut j = 0usize;
        let mut d_max: f64 = 0.0;

        while i < ref_sorted.len() || j < test_sorted.len() {
            let ref_val = if i < ref_sorted.len() {
                ref_sorted[i]
            } else {
                f64::INFINITY
            };
            let test_val = if j < test_sorted.len() {
                test_sorted[j]
            } else {
                f64::INFINITY
            };

            if ref_val <= test_val {
                i += 1;
            }
            if test_val <= ref_val {
                j += 1;
            }

            let cdf_ref = (i as f64) / n1;
            let cdf_test = (j as f64) / n2;
            let diff = (cdf_ref - cdf_test).abs();
            if diff > d_max {
                d_max = diff;
            }
        }

        // Asymptotic p-value approximation (Kolmogorov distribution)
        let en = (n1 * n2 / (n1 + n2)).sqrt();
        let lambda = (en + 0.12 + 0.11 / en) * d_max;
        let p_value = ks_p_value(lambda);

        let threshold = ks_critical_value(n1 as usize, n2 as usize, self.significance_level);
        let detected = d_max > threshold;

        Ok(DriftResult {
            detected,
            statistic: d_max,
            p_value: Some(p_value),
            threshold,
        })
    }
}

/// Approximate KS critical value using the asymptotic formula.
fn ks_critical_value(n1: usize, n2: usize, alpha: f64) -> f64 {
    let n = ((n1 * n2) as f64 / (n1 + n2) as f64).sqrt();
    // c(alpha) = sqrt(-0.5 * ln(alpha/2))
    let c = (-0.5 * (alpha / 2.0).ln()).sqrt();
    c / n
}

/// Kolmogorov distribution survival function approximation.
///
/// Uses the series expansion: P(K > x) = 2 * sum_{k=1}^{inf} (-1)^{k+1} * exp(-2*k^2*x^2)
fn ks_p_value(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }
    if lambda > 4.0 {
        return 0.0; // Effectively zero
    }

    let mut p = 0.0;
    for k in 1..=100 {
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        let term = sign * (-2.0 * (k as f64).powi(2) * lambda * lambda).exp();
        p += term;
        if term.abs() < 1e-15 {
            break;
        }
    }
    (2.0 * p).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Population Stability Index (PSI)
// ---------------------------------------------------------------------------

/// Population Stability Index (PSI) for measuring distribution shift.
///
/// PSI is computed by binning both distributions and comparing the bin
/// proportions:
///
/// ```text
/// PSI = sum_i (p_i - q_i) * ln(p_i / q_i)
/// ```
///
/// Interpretation: PSI < 0.1 → negligible, 0.1-0.25 → moderate, > 0.25 → significant.
#[derive(Debug, Clone)]
pub struct PopulationStabilityIndexDetector {
    /// Number of bins for the histogram.
    n_bins: usize,
    /// PSI threshold for drift detection.
    threshold: f64,
}

impl PopulationStabilityIndexDetector {
    /// Create a new PSI detector.
    ///
    /// * `n_bins` – number of equal-width bins (default: 10)
    /// * `threshold` – PSI value above which drift is declared (default: 0.25)
    pub fn new(n_bins: usize, threshold: f64) -> Result<Self> {
        if n_bins < 2 {
            return Err(TransformError::InvalidInput(
                "n_bins must be >= 2".to_string(),
            ));
        }
        if threshold <= 0.0 {
            return Err(TransformError::InvalidInput(
                "threshold must be positive".to_string(),
            ));
        }
        Ok(Self { n_bins, threshold })
    }

    /// Create with default settings (10 bins, threshold 0.25).
    pub fn default_config() -> Self {
        Self {
            n_bins: 10,
            threshold: 0.25,
        }
    }
}

impl DriftDetector for PopulationStabilityIndexDetector {
    fn detect(&self, reference: &[f64], test: &[f64]) -> Result<DriftResult> {
        if reference.is_empty() || test.is_empty() {
            return Err(TransformError::InvalidInput(
                "Reference and test samples must be non-empty".to_string(),
            ));
        }

        // Find global min/max for binning
        let mut global_min = f64::INFINITY;
        let mut global_max = f64::NEG_INFINITY;
        for &v in reference.iter().chain(test.iter()) {
            if v < global_min {
                global_min = v;
            }
            if v > global_max {
                global_max = v;
            }
        }

        if (global_max - global_min).abs() < 1e-15 {
            // All values identical → no drift
            return Ok(DriftResult {
                detected: false,
                statistic: 0.0,
                p_value: None,
                threshold: self.threshold,
            });
        }

        let bin_width = (global_max - global_min) / self.n_bins as f64;
        let eps = 1e-10; // Avoid log(0)

        // Count bins
        let ref_counts = bin_counts(reference, global_min, bin_width, self.n_bins);
        let test_counts = bin_counts(test, global_min, bin_width, self.n_bins);

        let n_ref = reference.len() as f64;
        let n_test = test.len() as f64;

        let mut psi = 0.0;
        for i in 0..self.n_bins {
            let p = (ref_counts[i] as f64 / n_ref) + eps;
            let q = (test_counts[i] as f64 / n_test) + eps;
            psi += (p - q) * (p / q).ln();
        }

        Ok(DriftResult {
            detected: psi > self.threshold,
            statistic: psi,
            p_value: None,
            threshold: self.threshold,
        })
    }
}

/// Bin data into `n_bins` equal-width buckets starting at `min_val`.
fn bin_counts(data: &[f64], min_val: f64, bin_width: f64, n_bins: usize) -> Vec<usize> {
    let mut counts = vec![0usize; n_bins];
    for &v in data {
        let idx = ((v - min_val) / bin_width).floor() as usize;
        let idx = idx.min(n_bins - 1);
        counts[idx] += 1;
    }
    counts
}

// ---------------------------------------------------------------------------
// Wasserstein distance (1-D)
// ---------------------------------------------------------------------------

/// Wasserstein-1 (earth mover's) distance for 1-D distribution shift detection.
///
/// The Wasserstein-1 distance between two 1-D distributions equals the area
/// between their empirical CDFs.
#[derive(Debug, Clone)]
pub struct WassersteinDetector {
    /// Distance threshold above which drift is detected.
    threshold: f64,
}

impl WassersteinDetector {
    /// Create a new Wasserstein detector with the given threshold.
    pub fn new(threshold: f64) -> Result<Self> {
        if threshold <= 0.0 {
            return Err(TransformError::InvalidInput(
                "threshold must be positive".to_string(),
            ));
        }
        Ok(Self { threshold })
    }
}

impl DriftDetector for WassersteinDetector {
    fn detect(&self, reference: &[f64], test: &[f64]) -> Result<DriftResult> {
        if reference.is_empty() || test.is_empty() {
            return Err(TransformError::InvalidInput(
                "Reference and test samples must be non-empty".to_string(),
            ));
        }

        let mut ref_sorted: Vec<f64> = reference.to_vec();
        ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut test_sorted: Vec<f64> = test.to_vec();
        test_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n1 = reference.len() as f64;
        let n2 = test.len() as f64;

        // Merge sorted values and integrate |CDF_ref - CDF_test|
        let mut all_vals: Vec<f64> = Vec::with_capacity(reference.len() + test.len());
        all_vals.extend_from_slice(&ref_sorted);
        all_vals.extend_from_slice(&test_sorted);
        all_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        all_vals.dedup();

        let mut distance = 0.0;
        let mut prev_val = all_vals[0];

        for &val in all_vals.iter().skip(1) {
            // CDF values at prev_val
            let cdf_ref = count_le(&ref_sorted, prev_val) as f64 / n1;
            let cdf_test = count_le(&test_sorted, prev_val) as f64 / n2;
            distance += (cdf_ref - cdf_test).abs() * (val - prev_val);
            prev_val = val;
        }

        Ok(DriftResult {
            detected: distance > self.threshold,
            statistic: distance,
            p_value: None,
            threshold: self.threshold,
        })
    }
}

/// Count elements in a sorted array that are <= val.
fn count_le(sorted: &[f64], val: f64) -> usize {
    match sorted.binary_search_by(|x| x.partial_cmp(&val).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(pos) => {
            // Find rightmost occurrence
            let mut p = pos;
            while p + 1 < sorted.len()
                && sorted[p + 1]
                    .partial_cmp(&val)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    == std::cmp::Ordering::Equal
            {
                p += 1;
            }
            p + 1
        }
        Err(pos) => pos,
    }
}

// ---------------------------------------------------------------------------
// Maximum Mean Discrepancy (MMD)
// ---------------------------------------------------------------------------

/// Maximum Mean Discrepancy (MMD) for multivariate drift detection.
///
/// Uses a Gaussian (RBF) kernel with bandwidth `sigma`. The biased MMD²
/// estimator is:
///
/// ```text
/// MMD²(P,Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
/// ```
///
/// where x,x' ~ P and y,y' ~ Q.
///
/// For multivariate data, each sample is a slice of length `dim`.
#[derive(Debug, Clone)]
pub struct MaximumMeanDiscrepancyDetector {
    /// RBF kernel bandwidth.
    sigma: f64,
    /// Dimensionality of each sample.
    dim: usize,
    /// MMD² threshold for drift detection.
    threshold: f64,
}

impl MaximumMeanDiscrepancyDetector {
    /// Create a new MMD detector.
    ///
    /// * `dim` – dimensionality of each sample
    /// * `sigma` – RBF kernel bandwidth
    /// * `threshold` – MMD² value above which drift is declared
    pub fn new(dim: usize, sigma: f64, threshold: f64) -> Result<Self> {
        if dim == 0 {
            return Err(TransformError::InvalidInput(
                "dim must be positive".to_string(),
            ));
        }
        if sigma <= 0.0 {
            return Err(TransformError::InvalidInput(
                "sigma must be positive".to_string(),
            ));
        }
        if threshold <= 0.0 {
            return Err(TransformError::InvalidInput(
                "threshold must be positive".to_string(),
            ));
        }
        Ok(Self {
            sigma,
            dim,
            threshold,
        })
    }

    /// Detect drift on multivariate data.
    ///
    /// `reference` and `test` are flattened arrays where every `dim` consecutive
    /// elements form one sample.
    pub fn detect_multivariate(&self, reference: &[f64], test: &[f64]) -> Result<DriftResult> {
        if reference.len() % self.dim != 0 || test.len() % self.dim != 0 {
            return Err(TransformError::InvalidInput(format!(
                "Data length must be a multiple of dim ({})",
                self.dim
            )));
        }

        let n_ref = reference.len() / self.dim;
        let n_test = test.len() / self.dim;

        if n_ref < 2 || n_test < 2 {
            return Err(TransformError::InvalidInput(
                "Need at least 2 samples in each set".to_string(),
            ));
        }

        let gamma = 1.0 / (2.0 * self.sigma * self.sigma);

        // E[k(x,x')]
        let mut kxx = 0.0;
        for i in 0..n_ref {
            for j in (i + 1)..n_ref {
                kxx += rbf_kernel(
                    &reference[i * self.dim..(i + 1) * self.dim],
                    &reference[j * self.dim..(j + 1) * self.dim],
                    gamma,
                );
            }
        }
        kxx *= 2.0 / (n_ref * (n_ref - 1)) as f64;

        // E[k(y,y')]
        let mut kyy = 0.0;
        for i in 0..n_test {
            for j in (i + 1)..n_test {
                kyy += rbf_kernel(
                    &test[i * self.dim..(i + 1) * self.dim],
                    &test[j * self.dim..(j + 1) * self.dim],
                    gamma,
                );
            }
        }
        kyy *= 2.0 / (n_test * (n_test - 1)) as f64;

        // E[k(x,y)]
        let mut kxy = 0.0;
        for i in 0..n_ref {
            for j in 0..n_test {
                kxy += rbf_kernel(
                    &reference[i * self.dim..(i + 1) * self.dim],
                    &test[j * self.dim..(j + 1) * self.dim],
                    gamma,
                );
            }
        }
        kxy /= (n_ref * n_test) as f64;

        let mmd2 = kxx - 2.0 * kxy + kyy;
        let mmd2 = mmd2.max(0.0); // Numerical safety

        Ok(DriftResult {
            detected: mmd2 > self.threshold,
            statistic: mmd2,
            p_value: None,
            threshold: self.threshold,
        })
    }
}

impl DriftDetector for MaximumMeanDiscrepancyDetector {
    /// For the 1-D DriftDetector trait, each element is treated as a 1-D sample.
    fn detect(&self, reference: &[f64], test: &[f64]) -> Result<DriftResult> {
        if self.dim != 1 {
            return Err(TransformError::InvalidInput(
                "Use detect_multivariate() for dim > 1".to_string(),
            ));
        }
        self.detect_multivariate(reference, test)
    }
}

/// RBF kernel: k(x, y) = exp(-gamma * ||x - y||^2)
fn rbf_kernel(x: &[f64], y: &[f64], gamma: f64) -> f64 {
    let sq_dist: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    (-gamma * sq_dist).exp()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ks_no_drift_same_distribution() {
        let reference: Vec<f64> = (0..200).map(|i| (i as f64) * 0.01).collect();
        let test: Vec<f64> = (0..200).map(|i| (i as f64) * 0.01 + 0.001).collect();

        let ks = KolmogorovSmirnovDetector::default_config();
        let result = ks.detect(&reference, &test).expect("detect");

        assert!(
            !result.detected,
            "Should NOT detect drift on nearly identical distributions: stat={}",
            result.statistic
        );
        assert!(result.p_value.is_some());
    }

    #[test]
    fn test_ks_detect_mean_shift() {
        let reference: Vec<f64> = (0..300).map(|i| (i as f64) * 0.01).collect();
        let test: Vec<f64> = (0..300).map(|i| (i as f64) * 0.01 + 5.0).collect();

        let ks = KolmogorovSmirnovDetector::default_config();
        let result = ks.detect(&reference, &test).expect("detect");

        assert!(
            result.detected,
            "Should detect drift after mean shift of 5.0: stat={}",
            result.statistic
        );
    }

    #[test]
    fn test_ks_empty_input() {
        let ks = KolmogorovSmirnovDetector::default_config();
        assert!(ks.detect(&[], &[1.0]).is_err());
        assert!(ks.detect(&[1.0], &[]).is_err());
    }

    #[test]
    fn test_ks_invalid_significance() {
        assert!(KolmogorovSmirnovDetector::new(0.0).is_err());
        assert!(KolmogorovSmirnovDetector::new(1.0).is_err());
        assert!(KolmogorovSmirnovDetector::new(-0.1).is_err());
    }

    #[test]
    fn test_psi_identical_distributions() {
        let data: Vec<f64> = (0..500).map(|i| (i as f64) * 0.01).collect();

        let psi = PopulationStabilityIndexDetector::default_config();
        let result = psi.detect(&data, &data).expect("detect");

        assert!(
            result.statistic < 0.01,
            "PSI for identical distributions should be ~0, got {}",
            result.statistic
        );
        assert!(!result.detected);
    }

    #[test]
    fn test_psi_detect_shift() {
        let reference: Vec<f64> = (0..500).map(|i| (i as f64) * 0.01).collect();
        let test: Vec<f64> = (0..500).map(|i| (i as f64) * 0.01 + 10.0).collect();

        let psi = PopulationStabilityIndexDetector::default_config();
        let result = psi.detect(&reference, &test).expect("detect");

        assert!(
            result.detected,
            "PSI should detect large distribution shift: psi={}",
            result.statistic
        );
    }

    #[test]
    fn test_psi_constant_values() {
        let data = vec![1.0; 100];
        let psi = PopulationStabilityIndexDetector::default_config();
        let result = psi.detect(&data, &data).expect("detect");
        assert!(!result.detected);
        assert!(result.statistic.abs() < 1e-10);
    }

    #[test]
    fn test_wasserstein_no_drift() {
        let reference: Vec<f64> = (0..200).map(|i| (i as f64) * 0.01).collect();
        let test: Vec<f64> = (0..200).map(|i| (i as f64) * 0.01 + 0.001).collect();

        let w = WassersteinDetector::new(1.0).expect("create");
        let result = w.detect(&reference, &test).expect("detect");

        assert!(
            !result.detected,
            "Should not detect drift: distance={}",
            result.statistic
        );
    }

    #[test]
    fn test_wasserstein_detect_shift() {
        let reference: Vec<f64> = (0..200).map(|i| (i as f64) * 0.01).collect();
        let test: Vec<f64> = (0..200).map(|i| (i as f64) * 0.01 + 10.0).collect();

        let w = WassersteinDetector::new(1.0).expect("create");
        let result = w.detect(&reference, &test).expect("detect");

        assert!(
            result.detected,
            "Should detect shift of 10.0: distance={}",
            result.statistic
        );
    }

    #[test]
    fn test_mmd_no_drift() {
        // Same distribution
        let reference: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1).collect();
        let test: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1 + 0.01).collect();

        let mmd = MaximumMeanDiscrepancyDetector::new(1, 1.0, 0.1).expect("create");
        let result = mmd.detect(&reference, &test).expect("detect");

        assert!(
            !result.detected,
            "Should not detect drift on similar distributions: mmd2={}",
            result.statistic
        );
    }

    #[test]
    fn test_mmd_detect_shift() {
        let reference: Vec<f64> = (0..50).map(|i| (i as f64) * 0.1).collect();
        let test: Vec<f64> = (0..50).map(|i| (i as f64) * 0.1 + 100.0).collect();

        let mmd = MaximumMeanDiscrepancyDetector::new(1, 1.0, 0.01).expect("create");
        let result = mmd.detect(&reference, &test).expect("detect");

        assert!(
            result.detected,
            "Should detect large shift: mmd2={}",
            result.statistic
        );
    }

    #[test]
    fn test_mmd_multivariate() {
        let dim = 3;
        // 20 samples of dim 3, tightly clustered around 0
        let reference: Vec<f64> = (0..60).map(|i| (i as f64) * 0.01).collect();
        // 20 samples of dim 3, shifted by 50
        let test: Vec<f64> = (0..60).map(|i| (i as f64) * 0.01 + 50.0).collect();

        let mmd = MaximumMeanDiscrepancyDetector::new(dim, 1.0, 0.01).expect("create");
        let result = mmd.detect_multivariate(&reference, &test).expect("detect");

        assert!(
            result.detected,
            "Should detect multivariate drift: mmd2={}",
            result.statistic
        );
    }

    #[test]
    fn test_mmd_error_wrong_dim() {
        let mmd = MaximumMeanDiscrepancyDetector::new(3, 1.0, 0.1).expect("create");
        // 5 elements not divisible by dim=3
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(mmd.detect_multivariate(&data, &data).is_err());
    }
}
