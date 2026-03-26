//! Multi-source depth fusion.
//!
//! Combines depth maps from multiple sensors using Kalman filtering or
//! confidence-weighted averaging.

use scirs2_core::ndarray::Array2;

use crate::error::{Result, VisionError};

use super::types::{CompletionMethod, CompletionResult, DepthSource, FusionConfig};

// ─────────────────────────────────────────────────────────────────────────────
// KalmanDepthFusion
// ─────────────────────────────────────────────────────────────────────────────

/// Per-pixel Kalman filter for fusing multiple depth sources.
///
/// Each pixel maintains a state `(depth_estimate, variance)`.  Sources are
/// incorporated sequentially via the standard Kalman measurement-update step.
#[derive(Debug, Clone)]
pub struct KalmanDepthFusion {
    /// Per-pixel depth estimate.
    state_mean: Array2<f64>,
    /// Per-pixel variance (uncertainty).
    state_var: Array2<f64>,
    /// Fusion configuration.
    config: FusionConfig,
    /// Number of sources incorporated so far.
    source_count: usize,
}

impl KalmanDepthFusion {
    /// Create a new Kalman depth fusion state.
    pub fn new(height: usize, width: usize, config: FusionConfig) -> Self {
        Self {
            state_mean: Array2::zeros((height, width)),
            state_var: Array2::from_elem((height, width), 1e6), // large initial variance
            config,
            source_count: 0,
        }
    }

    /// Incorporate a new depth source via Kalman measurement update.
    ///
    /// For each pixel with a valid measurement (`depth > 0`), the update is:
    /// ```text
    /// K = P / (P + R)
    /// x = x + K * (z - x)
    /// P = (1 - K) * P
    /// ```
    ///
    /// Outliers are rejected using a Mahalanobis-distance test.
    pub fn update(&mut self, source: &DepthSource) -> Result<()> {
        let (h, w) = self.state_mean.dim();
        let (sh, sw) = source.depth_map.dim();
        if sh != h || sw != w {
            return Err(VisionError::DimensionMismatch(format!(
                "source '{}' is {sh}x{sw} but fusion state is {h}x{w}",
                source.source_id
            )));
        }

        let r = source.noise_variance;

        // Prediction step: add process noise.
        if self.source_count > 0 {
            for row in 0..h {
                for col in 0..w {
                    self.state_var[[row, col]] += self.config.process_noise;
                }
            }
        }

        // Measurement update.
        for row in 0..h {
            for col in 0..w {
                let z = source.depth_map[[row, col]];
                if z <= 0.0 || !z.is_finite() {
                    continue; // no valid measurement
                }

                let p = self.state_var[[row, col]];
                let x = self.state_mean[[row, col]];

                // Outlier rejection (skip for first source).
                if self.source_count > 0 && self.is_outlier(z, x, p, r) {
                    continue;
                }

                // Kalman gain.
                let k = p / (p + r);

                // State update.
                self.state_mean[[row, col]] = x + k * (z - x);
                self.state_var[[row, col]] = (1.0 - k) * p;
            }
        }

        self.source_count += 1;
        Ok(())
    }

    /// Mahalanobis-distance outlier test.
    ///
    /// Returns `true` if the measurement is an outlier:
    /// `|z - x| / sqrt(P + R) > threshold`.
    fn is_outlier(&self, z: f64, x: f64, p: f64, r: f64) -> bool {
        let innovation_var = p + r;
        if innovation_var <= 0.0 {
            return false;
        }
        let mahalanobis = (z - x).abs() / innovation_var.sqrt();
        mahalanobis > self.config.outlier_threshold
    }

    /// Extract the current fused depth map and confidence.
    pub fn result(&self) -> CompletionResult {
        let (h, w) = self.state_mean.dim();
        let mut conf = Array2::zeros((h, w));

        // Confidence is inversely proportional to variance.
        let max_var = self
            .state_var
            .iter()
            .filter(|v| v.is_finite())
            .fold(1.0_f64, |a, &b| a.max(b));

        for row in 0..h {
            for col in 0..w {
                let v = self.state_var[[row, col]];
                conf[[row, col]] = if max_var > 1e-12 {
                    1.0 - (v / max_var).min(1.0)
                } else {
                    1.0
                };
            }
        }

        CompletionResult {
            dense_depth: self.state_mean.clone(),
            confidence_map: conf,
            method_used: CompletionMethod::KalmanFusion,
            iterations: self.source_count,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// fuse_sources
// ─────────────────────────────────────────────────────────────────────────────

/// Fuse multiple depth sources using sequential Kalman updates.
///
/// # Errors
/// Returns an error if no sources are provided or if sources have inconsistent
/// dimensions.
pub fn fuse_sources(sources: &[DepthSource], config: &FusionConfig) -> Result<CompletionResult> {
    if sources.is_empty() {
        return Err(VisionError::InvalidParameter(
            "no depth sources provided".to_string(),
        ));
    }

    let (h, w) = sources[0].depth_map.dim();
    let mut fusion = KalmanDepthFusion::new(h, w, config.clone());

    for source in sources {
        fusion.update(source)?;
    }

    Ok(fusion.result())
}

// ─────────────────────────────────────────────────────────────────────────────
// confidence_weighted_average
// ─────────────────────────────────────────────────────────────────────────────

/// Simple confidence-weighted average of multiple depth sources.
///
/// For each pixel: `d = sum(conf_i * d_i) / sum(conf_i)`.
///
/// # Errors
/// Returns an error if no sources are provided or dimensions are inconsistent.
pub fn confidence_weighted_average(sources: &[DepthSource]) -> Result<CompletionResult> {
    if sources.is_empty() {
        return Err(VisionError::InvalidParameter(
            "no depth sources provided".to_string(),
        ));
    }

    let (h, w) = sources[0].depth_map.dim();
    let mut depth_sum: Array2<f64> = Array2::zeros((h, w));
    let mut weight_sum: Array2<f64> = Array2::zeros((h, w));

    for source in sources {
        let (sh, sw) = source.depth_map.dim();
        if sh != h || sw != w {
            return Err(VisionError::DimensionMismatch(format!(
                "source '{}' is {sh}x{sw} but expected {h}x{w}",
                source.source_id
            )));
        }

        for row in 0..h {
            for col in 0..w {
                let d = source.depth_map[[row, col]];
                if d <= 0.0 || !d.is_finite() {
                    continue;
                }
                let c = source.confidence[[row, col]].max(0.0);
                depth_sum[[row, col]] += c * d;
                weight_sum[[row, col]] += c;
            }
        }
    }

    let mut dense = Array2::zeros((h, w));
    let mut conf = Array2::zeros((h, w));

    for row in 0..h {
        for col in 0..w {
            let ws: f64 = weight_sum[[row, col]];
            if ws > 1e-12 {
                dense[[row, col]] = depth_sum[[row, col]] / ws;
                conf[[row, col]] = ws.min(1.0);
            }
        }
    }

    Ok(CompletionResult {
        dense_depth: dense,
        confidence_map: conf,
        method_used: CompletionMethod::ConfidenceWeightedAverage,
        iterations: 0,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_source(id: &str, depth: f64, confidence: f64, noise: f64) -> DepthSource {
        DepthSource {
            source_id: id.to_string(),
            depth_map: Array2::from_elem((4, 4), depth),
            confidence: Array2::from_elem((4, 4), confidence),
            noise_variance: noise,
        }
    }

    #[test]
    fn kalman_two_identical_sources_reduces_variance() {
        let config = FusionConfig::default();
        let mut fusion = KalmanDepthFusion::new(4, 4, config);

        let source = make_source("s1", 5.0, 1.0, 0.1);
        fusion.update(&source).expect("ok");
        let var_after_one = fusion.state_var[[0, 0]];

        fusion.update(&source).expect("ok");
        let var_after_two = fusion.state_var[[0, 0]];

        assert!(
            var_after_two < var_after_one,
            "variance should decrease: {var_after_two} >= {var_after_one}"
        );
    }

    #[test]
    fn kalman_weights_by_inverse_noise() {
        let config = FusionConfig {
            outlier_threshold: 100.0, // no rejection
            process_noise: 0.0,
            measurement_noise: 0.1,
        };

        // Source A: depth=2, low noise
        let source_a = DepthSource {
            source_id: "a".to_string(),
            depth_map: Array2::from_elem((2, 2), 2.0),
            confidence: Array2::from_elem((2, 2), 1.0),
            noise_variance: 0.01,
        };
        // Source B: depth=8, high noise
        let source_b = DepthSource {
            source_id: "b".to_string(),
            depth_map: Array2::from_elem((2, 2), 8.0),
            confidence: Array2::from_elem((2, 2), 1.0),
            noise_variance: 10.0,
        };

        let result = fuse_sources(&[source_a, source_b], &config).expect("ok");
        let fused = result.dense_depth[[0, 0]];
        // Should be much closer to 2.0 (low noise) than 8.0 (high noise).
        assert!(
            fused < 5.0,
            "fused depth should favour low-noise source, got {fused}"
        );
    }

    #[test]
    fn outlier_rejected_at_5_sigma() {
        let config = FusionConfig {
            outlier_threshold: 3.0,
            process_noise: 0.0,
            measurement_noise: 0.1,
        };

        let mut fusion = KalmanDepthFusion::new(4, 4, config);

        // First source: depth=5
        let s1 = make_source("s1", 5.0, 1.0, 0.1);
        fusion.update(&s1).expect("ok");
        let after_s1 = fusion.state_mean[[0, 0]];

        // Second source: extreme outlier depth=500
        let s2 = make_source("s2", 500.0, 1.0, 0.1);
        fusion.update(&s2).expect("ok");
        let after_s2 = fusion.state_mean[[0, 0]];

        // The outlier should be rejected, so depth should stay near 5.0.
        assert!(
            (after_s2 - after_s1).abs() < 1.0,
            "outlier should be rejected: before={after_s1}, after={after_s2}"
        );
    }

    #[test]
    fn confidence_weighted_correct_mean() {
        let s1 = DepthSource {
            source_id: "s1".to_string(),
            depth_map: Array2::from_elem((2, 2), 4.0),
            confidence: Array2::from_elem((2, 2), 0.75),
            noise_variance: 0.1,
        };
        let s2 = DepthSource {
            source_id: "s2".to_string(),
            depth_map: Array2::from_elem((2, 2), 8.0),
            confidence: Array2::from_elem((2, 2), 0.25),
            noise_variance: 0.1,
        };

        let result = confidence_weighted_average(&[s1, s2]).expect("ok");
        // Expected: (0.75*4 + 0.25*8) / (0.75+0.25) = (3+2)/1 = 5.0
        let val = result.dense_depth[[0, 0]];
        assert!(
            (val - 5.0).abs() < 1e-9,
            "weighted mean should be 5.0, got {val}"
        );
    }

    #[test]
    fn single_source_passthrough() {
        let source = make_source("s", 7.0, 1.0, 0.1);
        let config = FusionConfig::default();
        let result = fuse_sources(&[source], &config).expect("ok");
        let val = result.dense_depth[[0, 0]];
        assert!(
            (val - 7.0).abs() < 0.01,
            "single source should pass through, got {val}"
        );
    }

    #[test]
    fn empty_sources_error() {
        let config = FusionConfig::default();
        assert!(fuse_sources(&[], &config).is_err());
        assert!(confidence_weighted_average(&[]).is_err());
    }

    #[test]
    fn config_defaults_reasonable() {
        let config = FusionConfig::default();
        assert!(config.outlier_threshold > 0.0);
        assert!(config.process_noise >= 0.0);
        assert!(config.measurement_noise > 0.0);

        let dc = super::super::types::DepthCompletionConfig::default();
        assert!(dc.sigma_spatial > 0.0);
        assert!(dc.sigma_intensity > 0.0);
        assert!(dc.tv_lambda > 0.0);
        assert!(dc.max_iterations > 0);
        assert!(dc.convergence_tol > 0.0);
    }

    #[test]
    fn dimension_mismatch_errors() {
        let s1 = make_source("s1", 5.0, 1.0, 0.1);
        let s2 = DepthSource {
            source_id: "s2".to_string(),
            depth_map: Array2::from_elem((3, 3), 5.0),
            confidence: Array2::from_elem((3, 3), 1.0),
            noise_variance: 0.1,
        };
        let config = FusionConfig::default();
        assert!(fuse_sources(&[s1, s2], &config).is_err());
    }
}
