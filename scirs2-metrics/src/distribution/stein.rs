//! Kernel Stein Discrepancy (KSD) for Goodness-of-Fit Testing
//!
//! This module provides advanced KSD implementations with:
//! - RBF and IMQ kernel support
//! - U-statistic (unbiased) and V-statistic (biased) estimators
//! - Bootstrap hypothesis testing for p-value computation
//! - Automatic bandwidth selection via median heuristic

use super::types::{KernelType, KsdConfig, KsdResult};
use crate::error::{MetricsError, Result};

// ────────────────────────────────────────────────────────────────────────────
// Kernel Stein Discrepancy struct
// ────────────────────────────────────────────────────────────────────────────

/// Kernel Stein Discrepancy calculator for goodness-of-fit testing.
///
/// KSD measures how well a set of samples fits a target distribution P
/// given access to the score function `s_P(x) = d/dx log p(x)`.
///
/// The statistic is based on the Stein kernel:
/// ```text
/// u_P(x, y) = s_P(x) * s_P(y) * k(x,y)
///           + s_P(x) * dk/dy(x,y)
///           + s_P(y) * dk/dx(x,y)
///           + d²k/dxdy(x,y)
/// ```
///
/// # Example
/// ```ignore
/// use scirs2_metrics::distribution::stein::KernelSteinDiscrepancy;
/// use scirs2_metrics::distribution::types::{KsdConfig, KernelType};
///
/// let samples = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
/// let score_fn = |x: f64| -x; // score of N(0,1)
/// let config = KsdConfig::default();
///
/// let ksd = KernelSteinDiscrepancy::new(&config);
/// let result = ksd.u_statistic(&samples, &score_fn).expect("should succeed");
/// ```
pub struct KernelSteinDiscrepancy {
    /// Kernel type
    kernel: KernelType,
    /// Bandwidth (None => median heuristic)
    bandwidth: Option<f64>,
    /// Number of bootstrap resamples
    n_bootstrap: usize,
}

impl KernelSteinDiscrepancy {
    /// Create a new KSD calculator from configuration.
    pub fn new(config: &KsdConfig) -> Self {
        Self {
            kernel: config.kernel,
            bandwidth: config.bandwidth,
            n_bootstrap: config.n_bootstrap,
        }
    }

    /// Compute the unbiased U-statistic KSD estimator.
    ///
    /// ```text
    /// KSD²_U = 1/(n(n-1)) * Σ_{i≠j} u_P(x_i, x_j)
    /// ```
    ///
    /// # Arguments
    /// * `samples` - observed samples (at least 2)
    /// * `score_fn` - score function `s(x) = d/dx log p(x)` of the target
    ///
    /// # Returns
    /// KSD value (square root of the U-statistic).
    pub fn u_statistic(&self, samples: &[f64], score_fn: &dyn Fn(f64) -> f64) -> Result<f64> {
        if samples.len() < 2 {
            return Err(MetricsError::InvalidInput(
                "at least 2 samples required for KSD U-statistic".to_string(),
            ));
        }

        let n = samples.len();
        let h2 = self.resolve_bandwidth(samples)?;
        let scores: Vec<f64> = samples.iter().map(|&x| score_fn(x)).collect();

        let mut ksd_sq = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let u = self.stein_kernel(samples[i], samples[j], scores[i], scores[j], h2);
                ksd_sq += 2.0 * u; // symmetry: u(x_i, x_j) = u(x_j, x_i)
            }
        }

        let normalizer = (n * (n - 1)) as f64;
        Ok((ksd_sq / normalizer).max(0.0).sqrt())
    }

    /// Compute the V-statistic KSD estimator (biased but lower variance).
    ///
    /// ```text
    /// KSD²_V = 1/n² * Σ_{i,j} u_P(x_i, x_j)
    /// ```
    ///
    /// # Arguments
    /// * `samples` - observed samples (at least 2)
    /// * `score_fn` - score function of the target
    ///
    /// # Returns
    /// KSD value (square root of the V-statistic).
    pub fn v_statistic(&self, samples: &[f64], score_fn: &dyn Fn(f64) -> f64) -> Result<f64> {
        if samples.len() < 2 {
            return Err(MetricsError::InvalidInput(
                "at least 2 samples required for KSD V-statistic".to_string(),
            ));
        }

        let n = samples.len();
        let h2 = self.resolve_bandwidth(samples)?;
        let scores: Vec<f64> = samples.iter().map(|&x| score_fn(x)).collect();

        let mut ksd_sq = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let u = self.stein_kernel(samples[i], samples[j], scores[i], scores[j], h2);
                ksd_sq += u;
            }
        }

        let normalizer = (n * n) as f64;
        Ok((ksd_sq / normalizer).max(0.0).sqrt())
    }

    /// Bootstrap test for KSD goodness-of-fit.
    ///
    /// Tests H₀: samples come from the target distribution (KSD = 0).
    ///
    /// The bootstrap procedure:
    /// 1. Compute the observed KSD U-statistic
    /// 2. For each bootstrap replicate, resample with replacement and compute
    ///    a centered Stein kernel matrix statistic
    /// 3. p-value = fraction of bootstrap stats ≥ observed stat
    ///
    /// # Arguments
    /// * `samples` - observed samples
    /// * `score_fn` - score function of the target
    /// * `significance` - significance level (e.g. 0.05)
    ///
    /// # Returns
    /// A `KsdResult` with the statistic, p-value, and rejection decision.
    pub fn bootstrap_test(
        &self,
        samples: &[f64],
        score_fn: &dyn Fn(f64) -> f64,
        significance: f64,
    ) -> Result<KsdResult> {
        if samples.len() < 2 {
            return Err(MetricsError::InvalidInput(
                "at least 2 samples required for bootstrap test".to_string(),
            ));
        }
        if significance <= 0.0 || significance >= 1.0 {
            return Err(MetricsError::InvalidInput(
                "significance must be in (0, 1)".to_string(),
            ));
        }

        let n = samples.len();
        let h2 = self.resolve_bandwidth(samples)?;
        let scores: Vec<f64> = samples.iter().map(|&x| score_fn(x)).collect();

        // Compute full Stein kernel matrix
        let mut kernel_matrix = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in i..n {
                let u = self.stein_kernel(samples[i], samples[j], scores[i], scores[j], h2);
                kernel_matrix[i][j] = u;
                kernel_matrix[j][i] = u;
            }
        }

        // Observed U-statistic (n * (n-1) * KSD²_U)
        let mut observed = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    observed += kernel_matrix[i][j];
                }
            }
        }

        // Wild bootstrap: resample signs using deterministic sequence
        let golden = 0.618_033_988_749_895_f64;
        let mut count_exceed = 0usize;

        for b in 0..self.n_bootstrap {
            // Generate Rademacher weights (+1 or -1)
            let mut boot_stat = 0.0f64;
            // Use a simple deterministic hash for reproducibility
            let mut weights = Vec::with_capacity(n);
            for i in 0..n {
                let hash =
                    ((b as f64 + 1.0) * golden + (i as f64 + 1.0) * std::f64::consts::PI).fract();
                let w = if hash < 0.5 { 1.0 } else { -1.0 };
                weights.push(w);
            }

            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        boot_stat += weights[i] * weights[j] * kernel_matrix[i][j];
                    }
                }
            }

            if boot_stat >= observed {
                count_exceed += 1;
            }
        }

        let p_value = count_exceed as f64 / self.n_bootstrap as f64;
        let statistic = (observed / (n * (n - 1)) as f64).max(0.0).sqrt();

        Ok(KsdResult {
            statistic,
            p_value: Some(p_value),
            rejected: Some(p_value < significance),
            kernel: self.kernel,
            bandwidth: h2.sqrt(),
        })
    }

    /// Resolve bandwidth: use provided value or compute via median heuristic.
    fn resolve_bandwidth(&self, samples: &[f64]) -> Result<f64> {
        match self.bandwidth {
            Some(bw) => {
                if bw <= 0.0 {
                    return Err(MetricsError::InvalidInput(
                        "bandwidth must be positive".to_string(),
                    ));
                }
                Ok(2.0 * bw * bw)
            }
            None => {
                let n = samples.len();
                let mut dists: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
                for i in 0..n {
                    for j in (i + 1)..n {
                        dists.push((samples[i] - samples[j]).powi(2));
                    }
                }
                dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let med = if dists.is_empty() {
                    1.0
                } else {
                    dists[dists.len() / 2]
                };
                let h2 = med / (n as f64).ln().max(1.0);
                if h2 <= 0.0 {
                    return Err(MetricsError::CalculationError(
                        "computed bandwidth is non-positive; samples may be constant".to_string(),
                    ));
                }
                Ok(h2)
            }
        }
    }

    /// Compute the Stein kernel u_P(x, y) for the configured kernel type.
    ///
    /// For 1D:
    /// ```text
    /// u_P(x, y) = s(x)*s(y)*k(x,y) + s(x)*dk/dy + s(y)*dk/dx + d²k/dxdy
    /// ```
    fn stein_kernel(&self, x: f64, y: f64, sx: f64, sy: f64, h2: f64) -> f64 {
        let diff = x - y;
        let diff_sq = diff * diff;

        match self.kernel {
            KernelType::Rbf => {
                let k = (-diff_sq / h2).exp();
                let dk_dy = k * 2.0 * diff / h2;
                let dk_dx = -dk_dy;
                let d2k = k * (2.0 / h2) * (1.0 - 2.0 * diff_sq / h2);
                sx * sy * k + sx * dk_dy + sy * dk_dx + d2k
            }
            KernelType::Imq { c, beta } => {
                // k(x,y) = (c² + ||x-y||²)^β
                let base = c * c + diff_sq;
                let k = base.powf(beta);

                // dk/dy = β * (c² + (x-y)²)^(β-1) * (-2(x-y))
                let dk_dy = beta * base.powf(beta - 1.0) * (-2.0 * diff);
                let dk_dx = beta * base.powf(beta - 1.0) * (2.0 * diff);

                // d²k/dxdy = β*(β-1)*(base)^(β-2) * (2*diff) * (-2*diff)
                //          + β*(base)^(β-1) * (-2)
                let d2k = beta * (beta - 1.0) * base.powf(beta - 2.0) * (-4.0 * diff_sq)
                    + beta * base.powf(beta - 1.0) * (-2.0);

                sx * sy * k + sx * dk_dy + sy * dk_dx + d2k
            }
            KernelType::Polynomial { alpha, c, degree } => {
                // k(x,y) = (alpha * x*y + c)^degree
                let inner = alpha * x * y + c;
                let k = inner.powi(degree as i32);

                // dk/dy = degree * (alpha*x*y + c)^(d-1) * alpha * x
                let dk_dy = if degree >= 1 {
                    degree as f64 * inner.powi(degree as i32 - 1) * alpha * x
                } else {
                    0.0
                };
                let dk_dx = if degree >= 1 {
                    degree as f64 * inner.powi(degree as i32 - 1) * alpha * y
                } else {
                    0.0
                };

                // d²k/dxdy = d*(d-1)*(inner)^(d-2) * alpha^2 * x * y
                //           + d*(inner)^(d-1) * alpha
                let d2k = if degree >= 2 {
                    degree as f64
                        * (degree as f64 - 1.0)
                        * inner.powi(degree as i32 - 2)
                        * alpha
                        * alpha
                        * x
                        * y
                        + degree as f64 * inner.powi(degree as i32 - 1) * alpha
                } else if degree == 1 {
                    alpha
                } else {
                    0.0
                };

                sx * sy * k + sx * dk_dy + sy * dk_dx + d2k
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Convenience functions
// ────────────────────────────────────────────────────────────────────────────

/// Compute the KSD U-statistic with default configuration.
///
/// Shorthand for creating a `KernelSteinDiscrepancy` with default settings
/// and computing the U-statistic.
pub fn ksd_u_statistic(samples: &[f64], score_fn: &dyn Fn(f64) -> f64) -> Result<f64> {
    let config = KsdConfig::default();
    let ksd = KernelSteinDiscrepancy::new(&config);
    ksd.u_statistic(samples, score_fn)
}

/// Compute the KSD V-statistic with default configuration.
pub fn ksd_v_statistic(samples: &[f64], score_fn: &dyn Fn(f64) -> f64) -> Result<f64> {
    let config = KsdConfig::default();
    let ksd = KernelSteinDiscrepancy::new(&config);
    ksd.v_statistic(samples, score_fn)
}

/// Run a bootstrap KSD test with default configuration.
///
/// # Arguments
/// * `samples` - observed data
/// * `score_fn` - score of the null distribution
/// * `significance` - significance level (e.g. 0.05)
///
/// # Returns
/// A `KsdResult` with statistic, p-value, and rejection decision.
pub fn ksd_bootstrap_test(
    samples: &[f64],
    score_fn: &dyn Fn(f64) -> f64,
    significance: f64,
) -> Result<KsdResult> {
    let config = KsdConfig::default();
    let ksd = KernelSteinDiscrepancy::new(&config);
    ksd.bootstrap_test(samples, score_fn, significance)
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn normal_score(x: f64) -> f64 {
        -x // d/dx log N(0,1) = -x
    }

    fn shifted_normal_score(x: f64) -> f64 {
        -(x - 5.0) // d/dx log N(5,1) = -(x-5)
    }

    #[test]
    fn test_ksd_u_statistic_nonnegative() {
        let samples = vec![-1.5, -0.5, 0.0, 0.5, 1.5, -1.0, 1.0, 0.3, -0.3, 0.8];
        let ksd = ksd_u_statistic(&samples, &normal_score).expect("should succeed");
        assert!(ksd >= 0.0, "KSD must be non-negative, got {ksd}");
    }

    #[test]
    fn test_ksd_v_statistic_nonnegative() {
        let samples = vec![-1.0, 0.0, 1.0, 0.5, -0.5];
        let ksd = ksd_v_statistic(&samples, &normal_score).expect("should succeed");
        assert!(ksd >= 0.0, "KSD must be non-negative, got {ksd}");
    }

    #[test]
    fn test_ksd_detects_mismatch() {
        // Samples from N(0,1) tested against N(5,1) should have large KSD
        let samples_from_n01 = vec![-1.0, -0.5, 0.0, 0.3, 0.5, 1.0, -0.8, 0.7, -0.2, 0.1];
        let ksd_match = ksd_u_statistic(&samples_from_n01, &normal_score).expect("should succeed");
        let ksd_mismatch =
            ksd_u_statistic(&samples_from_n01, &shifted_normal_score).expect("should succeed");

        assert!(
            ksd_mismatch > ksd_match,
            "KSD should be larger for mismatched distribution: match={ksd_match}, mismatch={ksd_mismatch}"
        );
    }

    #[test]
    fn test_ksd_small_for_matching_distribution() {
        // Samples roughly from N(0,1) tested against N(0,1) should be small
        let samples = vec![-1.5, -0.5, 0.0, 0.5, 1.5, -1.0, 1.0, 0.3, -0.3, 0.8];
        let ksd = ksd_u_statistic(&samples, &normal_score).expect("should succeed");
        // Not zero due to finite samples, but should be reasonable
        assert!(
            ksd < 2.0,
            "KSD for matching dist should be moderate, got {ksd}"
        );
    }

    #[test]
    fn test_ksd_with_imq_kernel() {
        let config = KsdConfig {
            kernel: KernelType::Imq { c: 1.0, beta: -0.5 },
            bandwidth: Some(1.0),
            n_bootstrap: 100,
        };
        let ksd_calc = KernelSteinDiscrepancy::new(&config);
        let samples = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let result = ksd_calc
            .u_statistic(&samples, &normal_score)
            .expect("should succeed");
        assert!(result >= 0.0, "IMQ KSD must be non-negative, got {result}");
    }

    #[test]
    fn test_ksd_with_polynomial_kernel() {
        let config = KsdConfig {
            kernel: KernelType::Polynomial {
                alpha: 1.0,
                c: 1.0,
                degree: 3,
            },
            bandwidth: Some(1.0),
            n_bootstrap: 100,
        };
        let ksd_calc = KernelSteinDiscrepancy::new(&config);
        let samples = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let result = ksd_calc
            .u_statistic(&samples, &normal_score)
            .expect("should succeed");
        assert!(result >= 0.0, "Poly KSD must be non-negative, got {result}");
    }

    #[test]
    fn test_ksd_bootstrap_returns_p_value() {
        let samples = vec![-1.0, -0.5, 0.0, 0.5, 1.0, -0.8, 0.3, 0.7, -0.2, 0.1];
        let config = KsdConfig {
            n_bootstrap: 200,
            ..Default::default()
        };
        let ksd_calc = KernelSteinDiscrepancy::new(&config);
        let result = ksd_calc
            .bootstrap_test(&samples, &normal_score, 0.05)
            .expect("should succeed");

        assert!(result.statistic >= 0.0);
        assert!(result.p_value.is_some());
        let p = result.p_value.expect("p_value should be Some");
        assert!(
            (0.0..=1.0).contains(&p),
            "p-value must be in [0,1], got {p}"
        );
        assert!(result.rejected.is_some());
    }

    #[test]
    fn test_ksd_too_few_samples() {
        let samples = vec![1.0];
        assert!(ksd_u_statistic(&samples, &normal_score).is_err());
        assert!(ksd_v_statistic(&samples, &normal_score).is_err());
    }

    #[test]
    fn test_ksd_bootstrap_invalid_significance() {
        let samples = vec![0.0, 1.0, 2.0];
        assert!(ksd_bootstrap_test(&samples, &normal_score, 0.0).is_err());
        assert!(ksd_bootstrap_test(&samples, &normal_score, 1.0).is_err());
    }
}
