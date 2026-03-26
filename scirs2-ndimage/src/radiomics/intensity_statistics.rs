//! First-order intensity statistics for radiomics analysis.
//!
//! Computes PyRadiomics-compatible first-order features from the intensity
//! distribution of voxels inside a binary mask.

/// First-order intensity statistics computed from the foreground intensity distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct IntensityFeatures {
    /// Arithmetic mean of intensity values
    pub mean: f64,
    /// Variance (second central moment)
    pub variance: f64,
    /// Skewness (third standardised central moment)
    pub skewness: f64,
    /// Excess kurtosis (fourth standardised central moment − 3)
    pub kurtosis: f64,
    /// Shannon entropy computed from a 64-bin histogram: −Σ p·log₂(p)
    pub entropy: f64,
    /// Uniformity = Σ p² (sum of squared bin probabilities)
    pub uniformity: f64,
    /// Energy = Σ x² (sum of squared intensity values)
    pub energy: f64,
    /// Median intensity
    pub median: f64,
    /// 10th percentile
    pub p10: f64,
    /// 25th percentile
    pub p25: f64,
    /// 75th percentile
    pub p75: f64,
    /// 90th percentile
    pub p90: f64,
    /// Interquartile range = p75 − p25
    pub iqr: f64,
    /// Range = max − min
    pub range: f64,
    /// Mean absolute deviation from the mean
    pub mean_abs_deviation: f64,
    /// Robust mean absolute deviation = median absolute deviation from the median
    pub robust_mean_abs_deviation: f64,
    /// Coefficient of variation = σ / |μ|  (NaN if mean ≈ 0)
    pub coefficient_variation: f64,
    /// Quartile coefficient of dispersion = (p75 − p25) / (p75 + p25)
    pub quartile_coef_dispersion: f64,
}

/// Compute first-order intensity statistics for voxels selected by a boolean mask.
///
/// * `image` – flat array of intensity values (same length as `mask`)
/// * `mask`  – boolean flat array; only positions where `mask[i]` is `true` are included
///
/// Returns `None` if no voxel is selected (empty mask).
pub fn compute_intensity_features(image: &[f64], mask: &[bool]) -> Option<IntensityFeatures> {
    if image.len() != mask.len() {
        return None;
    }

    // Collect foreground values
    let values: Vec<f64> = image
        .iter()
        .zip(mask.iter())
        .filter_map(|(&v, &m)| if m { Some(v) } else { None })
        .collect();

    let n = values.len();
    if n == 0 {
        return None;
    }

    // Basic statistics
    let sum: f64 = values.iter().sum();
    let mean = sum / n as f64;

    let variance: f64 = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std = variance.sqrt();

    // Skewness = E[(X-μ)³] / σ³
    let skewness = if std > 0.0 {
        let m3 = values.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n as f64;
        m3 / (std * std * std)
    } else {
        0.0
    };

    // Excess kurtosis = E[(X-μ)⁴] / σ⁴ − 3
    let kurtosis = if std > 0.0 {
        let m4 = values.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n as f64;
        m4 / (std * std * std * std) - 3.0
    } else {
        -3.0
    };

    // Energy = Σ x²
    let energy: f64 = values.iter().map(|&x| x * x).sum();

    // --- Histogram-based features (64 bins) ---
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    const N_BINS: usize = 64;
    let bin_width = if (max_val - min_val).abs() < 1e-15 {
        1.0 // all values equal — put everything in one bin
    } else {
        (max_val - min_val) / N_BINS as f64
    };

    let mut bins = vec![0usize; N_BINS];
    for &v in &values {
        let idx = ((v - min_val) / bin_width).floor() as usize;
        let idx = idx.min(N_BINS - 1);
        bins[idx] += 1;
    }

    let nf = n as f64;
    let probs: Vec<f64> = bins.iter().map(|&c| c as f64 / nf).collect();

    // Entropy: −Σ p·log₂(p)
    let epsilon = 1e-15;
    let entropy: f64 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * (p + epsilon).log2())
        .sum();

    // Uniformity = Σ p²
    let uniformity: f64 = probs.iter().map(|&p| p * p).sum();

    // --- Order statistics ---
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let percentile = |frac: f64| -> f64 {
        let pos = frac * (n - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let frac_part = pos - lo as f64;
        sorted[lo] * (1.0 - frac_part) + sorted[hi] * frac_part
    };

    let median = percentile(0.5);
    let p10 = percentile(0.10);
    let p25 = percentile(0.25);
    let p75 = percentile(0.75);
    let p90 = percentile(0.90);
    let iqr = p75 - p25;
    let range = max_val - min_val;

    // Mean absolute deviation from the mean
    let mean_abs_deviation: f64 = values.iter().map(|&x| (x - mean).abs()).sum::<f64>() / nf;

    // Robust MAD = median absolute deviation from median
    let mut abs_dev_from_median: Vec<f64> = values.iter().map(|&x| (x - median).abs()).collect();
    abs_dev_from_median.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let robust_mean_abs_deviation = {
        let m = abs_dev_from_median.len();
        if m == 0 {
            0.0
        } else {
            let pos = 0.5 * (m - 1) as f64;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(m - 1);
            let fp = pos - lo as f64;
            abs_dev_from_median[lo] * (1.0 - fp) + abs_dev_from_median[hi] * fp
        }
    };

    // Coefficient of variation = σ / |μ|
    let coefficient_variation = if mean.abs() > 1e-15 {
        std / mean.abs()
    } else {
        f64::NAN
    };

    // Quartile coefficient of dispersion = (p75 − p25) / (p75 + p25)
    let quartile_coef_dispersion = {
        let denom = p75 + p25;
        if denom.abs() > 1e-15 {
            iqr / denom
        } else {
            f64::NAN
        }
    };

    Some(IntensityFeatures {
        mean,
        variance,
        skewness,
        kurtosis,
        entropy,
        uniformity,
        energy,
        median,
        p10,
        p25,
        p75,
        p90,
        iqr,
        range,
        mean_abs_deviation,
        robust_mean_abs_deviation,
        coefficient_variation,
        quartile_coef_dispersion,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_basic_stats_known_array() {
        let image = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mask = vec![true; 5];
        let f = compute_intensity_features(&image, &mask).expect("should return features");
        // mean = 3
        assert!(approx_eq(f.mean, 3.0, 1e-10), "mean = {}", f.mean);
        // variance = 2  (population variance)
        assert!(
            approx_eq(f.variance, 2.0, 1e-10),
            "variance = {}",
            f.variance
        );
        // range = 4
        assert!(approx_eq(f.range, 4.0, 1e-10), "range = {}", f.range);
        // median = 3
        assert!(approx_eq(f.median, 3.0, 1e-10), "median = {}", f.median);
        // iqr = p75 - p25 = 4 - 2 = 2
        assert!(approx_eq(f.iqr, 2.0, 0.01), "iqr = {}", f.iqr);
    }

    #[test]
    fn test_kurtosis_uniform_approx() {
        // For a perfectly uniform distribution kurtosis is negative; for our
        // small array just verify it returns a finite number.
        let image = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mask = vec![true; 5];
        let f = compute_intensity_features(&image, &mask).expect("features");
        assert!(f.kurtosis.is_finite(), "kurtosis must be finite");
    }

    #[test]
    fn test_entropy_positive() {
        let image = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mask = vec![true; 5];
        let f = compute_intensity_features(&image, &mask).expect("features");
        assert!(
            f.entropy > 0.0,
            "entropy should be positive for varied values"
        );
    }

    #[test]
    fn test_energy() {
        let image = vec![1.0f64, 2.0, 3.0];
        let mask = vec![true; 3];
        let f = compute_intensity_features(&image, &mask).expect("features");
        // energy = 1 + 4 + 9 = 14
        assert!(approx_eq(f.energy, 14.0, 1e-10), "energy = {}", f.energy);
    }

    #[test]
    fn test_mask_selects_subset() {
        let image = vec![10.0f64, 1.0, 2.0, 3.0, 20.0];
        let mask = vec![false, true, true, true, false];
        let f = compute_intensity_features(&image, &mask).expect("features");
        assert!(approx_eq(f.mean, 2.0, 1e-10), "mean = {}", f.mean);
    }

    #[test]
    fn test_empty_mask_returns_none() {
        let image = vec![1.0f64, 2.0];
        let mask = vec![false, false];
        assert!(compute_intensity_features(&image, &mask).is_none());
    }

    #[test]
    fn test_constant_signal() {
        let image = vec![5.0f64; 10];
        let mask = vec![true; 10];
        let f = compute_intensity_features(&image, &mask).expect("features");
        assert!(approx_eq(f.mean, 5.0, 1e-10));
        assert!(approx_eq(f.variance, 0.0, 1e-10));
        assert!(approx_eq(f.skewness, 0.0, 1e-10));
    }
}
