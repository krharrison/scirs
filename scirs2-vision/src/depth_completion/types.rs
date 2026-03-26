//! Types for depth completion.

use scirs2_core::ndarray::Array2;

use crate::error::{Result, VisionError};

// ─────────────────────────────────────────────────────────────────────────────
// CompletionMethod
// ─────────────────────────────────────────────────────────────────────────────

/// Method used for depth completion.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CompletionMethod {
    /// Nearest-neighbour propagation.
    NearestNeighbor,
    /// Bilateral guided upsampling using an RGB guide image.
    BilateralGuided,
    /// Total-variation regularised completion.
    TotalVariation,
    /// Guided upsampling (reserved for future use).
    GuidedUpsampling,
    /// Inverse-distance-weighted interpolation.
    InverseDistanceWeighted,
    /// Confidence-weighted average fusion.
    ConfidenceWeightedAverage,
    /// Kalman fusion of multiple depth sources.
    KalmanFusion,
}

// ─────────────────────────────────────────────────────────────────────────────
// DepthCompletionConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for depth completion algorithms.
#[derive(Debug, Clone)]
pub struct DepthCompletionConfig {
    /// Completion method to use.
    pub method: CompletionMethod,
    /// Spatial sigma for bilateral filtering (pixels).
    pub sigma_spatial: f64,
    /// Intensity sigma for bilateral filtering.
    pub sigma_intensity: f64,
    /// Regularisation weight for total-variation completion.
    pub tv_lambda: f64,
    /// Maximum number of iterations (TV / iterative methods).
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub convergence_tol: f64,
}

impl Default for DepthCompletionConfig {
    fn default() -> Self {
        Self {
            method: CompletionMethod::NearestNeighbor,
            sigma_spatial: 5.0,
            sigma_intensity: 0.1,
            tv_lambda: 0.1,
            max_iterations: 100,
            convergence_tol: 1e-4,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SparseMeasurement / SparseDepthMap
// ─────────────────────────────────────────────────────────────────────────────

/// A single sparse depth measurement.
#[derive(Debug, Clone)]
pub struct SparseMeasurement {
    /// Row index in the image.
    pub row: usize,
    /// Column index in the image.
    pub col: usize,
    /// Depth value.
    pub depth: f64,
    /// Confidence in `[0, 1]`.
    pub confidence: f64,
}

/// A sparse depth map: image dimensions plus a set of sparse measurements.
#[derive(Debug, Clone)]
pub struct SparseDepthMap {
    /// Image height.
    pub height: usize,
    /// Image width.
    pub width: usize,
    /// Sparse depth measurements.
    pub measurements: Vec<SparseMeasurement>,
}

impl SparseDepthMap {
    /// Create a new sparse depth map, validating that all measurements are in bounds.
    pub fn new(height: usize, width: usize, measurements: Vec<SparseMeasurement>) -> Result<Self> {
        for m in &measurements {
            if m.row >= height || m.col >= width {
                return Err(VisionError::InvalidParameter(format!(
                    "measurement at ({}, {}) is out of bounds for {}x{} image",
                    m.row, m.col, height, width
                )));
            }
        }
        Ok(Self {
            height,
            width,
            measurements,
        })
    }

    /// Validate that the sparse depth map is non-empty.
    pub fn validate_non_empty(&self) -> Result<()> {
        if self.measurements.is_empty() {
            return Err(VisionError::InvalidParameter(
                "sparse depth map has no measurements".to_string(),
            ));
        }
        Ok(())
    }

    /// Rasterise measurements into a dense array (0.0 where no measurement exists).
    pub fn to_dense(&self) -> Array2<f64> {
        let mut dense = Array2::zeros((self.height, self.width));
        for m in &self.measurements {
            dense[[m.row, m.col]] = m.depth;
        }
        dense
    }

    /// Rasterise confidence values.
    pub fn confidence_map(&self) -> Array2<f64> {
        let mut conf = Array2::zeros((self.height, self.width));
        for m in &self.measurements {
            conf[[m.row, m.col]] = m.confidence;
        }
        conf
    }

    /// Create a binary mask of observed pixels.
    pub fn observation_mask(&self) -> Array2<bool> {
        let mut mask = Array2::from_elem((self.height, self.width), false);
        for m in &self.measurements {
            mask[[m.row, m.col]] = true;
        }
        mask
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CompletionResult
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a depth completion operation.
#[derive(Debug, Clone)]
pub struct CompletionResult {
    /// Dense depth map.
    pub dense_depth: Array2<f64>,
    /// Per-pixel confidence in `[0, 1]`.
    pub confidence_map: Array2<f64>,
    /// Method used to produce this result.
    pub method_used: CompletionMethod,
    /// Number of iterations (0 for non-iterative methods).
    pub iterations: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// FusionConfig / DepthSource
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for multi-source depth fusion.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Mahalanobis-distance threshold for outlier rejection.
    pub outlier_threshold: f64,
    /// Process noise variance for Kalman prediction step.
    pub process_noise: f64,
    /// Default measurement noise variance.
    pub measurement_noise: f64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            outlier_threshold: 3.0,
            process_noise: 0.01,
            measurement_noise: 0.1,
        }
    }
}

/// A single depth source for multi-sensor fusion.
#[derive(Debug, Clone)]
pub struct DepthSource {
    /// Identifier for this source.
    pub source_id: String,
    /// Dense depth map from this source (0.0 = invalid).
    pub depth_map: Array2<f64>,
    /// Per-pixel confidence.
    pub confidence: Array2<f64>,
    /// Noise variance of this sensor.
    pub noise_variance: f64,
}
