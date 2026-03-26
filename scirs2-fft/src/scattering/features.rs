//! Scattering feature extraction and normalization
//!
//! Provides utilities for extracting usable feature vectors from scattering
//! transform results, including:
//! - Log-scattering normalization: log(1 + |Sx|)
//! - L2 normalization per order
//! - Standardization (zero mean, unit variance)
//! - Feature concatenation and flattening
//! - Joint time-frequency features for 2D signals

use crate::error::{FFTError, FFTResult};

use super::scattering::{ScatteringConfig, ScatteringResult, ScatteringTransform};

/// Normalization methods for scattering features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureNormalization {
    /// No normalization (raw coefficients)
    None,
    /// Log-scattering: log(1 + |Sx|)
    Log,
    /// L2 normalization per coefficient path
    L2,
    /// Standardization: (x - mean) / std
    Standardize,
    /// Log followed by standardization
    LogStandardize,
}

/// Controls how time dimension is handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeFrequencyMode {
    /// Average over time dimension (produces a single feature vector per signal)
    TimeAveraged,
    /// Keep time dimension (produces a time-frequency matrix)
    TimeSeries,
}

/// Scattering feature extractor.
///
/// Wraps a `ScatteringTransform` and provides normalized feature extraction.
#[derive(Debug, Clone)]
pub struct ScatteringFeatureExtractor {
    transform: ScatteringTransform,
    normalization: FeatureNormalization,
    mode: TimeFrequencyMode,
}

impl ScatteringFeatureExtractor {
    /// Create a new feature extractor.
    ///
    /// # Arguments
    /// * `config` - Scattering configuration
    /// * `signal_length` - Expected input signal length
    /// * `normalization` - Normalization method
    /// * `mode` - Time-frequency handling mode
    pub fn new(
        config: ScatteringConfig,
        signal_length: usize,
        normalization: FeatureNormalization,
        mode: TimeFrequencyMode,
    ) -> FFTResult<Self> {
        let transform = ScatteringTransform::new(config, signal_length)?;
        Ok(Self {
            transform,
            normalization,
            mode,
        })
    }

    /// Extract features from a signal.
    ///
    /// Returns a `ScatteringFeatures` containing the normalized feature representation.
    pub fn extract(&self, signal: &[f64]) -> FFTResult<ScatteringFeatures> {
        let result = self.transform.transform(signal)?;
        let features = self.normalize_result(&result)?;
        Ok(features)
    }

    /// Normalize a scattering result into features.
    fn normalize_result(&self, result: &ScatteringResult) -> FFTResult<ScatteringFeatures> {
        let num_paths = result.coefficients.len();
        let output_length = result.output_length;

        // Collect all coefficient paths as rows
        let mut matrix: Vec<Vec<f64>> = result
            .coefficients
            .iter()
            .map(|c| c.values.clone())
            .collect();

        // Apply normalization
        match self.normalization {
            FeatureNormalization::None => {}
            FeatureNormalization::Log => {
                apply_log_normalization(&mut matrix);
            }
            FeatureNormalization::L2 => {
                apply_l2_normalization(&mut matrix);
            }
            FeatureNormalization::Standardize => {
                apply_standardization(&mut matrix);
            }
            FeatureNormalization::LogStandardize => {
                apply_log_normalization(&mut matrix);
                apply_standardization(&mut matrix);
            }
        }

        // Reduce time dimension if requested
        let feature_vector = match self.mode {
            TimeFrequencyMode::TimeAveraged => {
                // Average each path over time
                matrix
                    .iter()
                    .map(|row| {
                        if row.is_empty() {
                            0.0
                        } else {
                            row.iter().sum::<f64>() / row.len() as f64
                        }
                    })
                    .collect()
            }
            TimeFrequencyMode::TimeSeries => {
                // Flatten: concatenate all paths
                matrix.iter().flat_map(|row| row.iter().copied()).collect()
            }
        };

        Ok(ScatteringFeatures {
            feature_vector,
            num_paths,
            output_length,
            num_zeroth: result.num_zeroth,
            num_first: result.num_first,
            num_second: result.num_second,
            normalization: self.normalization,
            mode: self.mode,
        })
    }
}

/// Normalized scattering features ready for downstream use.
#[derive(Debug, Clone)]
pub struct ScatteringFeatures {
    /// The feature vector (flattened or time-averaged)
    pub feature_vector: Vec<f64>,
    /// Number of scattering paths
    pub num_paths: usize,
    /// Output length per path (before time-averaging)
    pub output_length: usize,
    /// Number of zeroth-order paths
    pub num_zeroth: usize,
    /// Number of first-order paths
    pub num_first: usize,
    /// Number of second-order paths
    pub num_second: usize,
    /// Normalization applied
    pub normalization: FeatureNormalization,
    /// Time-frequency mode used
    pub mode: TimeFrequencyMode,
}

impl ScatteringFeatures {
    /// Dimensionality of the feature vector.
    pub fn dim(&self) -> usize {
        self.feature_vector.len()
    }

    /// L2 norm of the feature vector.
    pub fn norm(&self) -> f64 {
        self.feature_vector
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
    }
}

/// Joint time-frequency scattering features for 2D signals (basic version).
///
/// Applies 1D scattering along rows and columns independently, then combines.
#[derive(Debug, Clone)]
pub struct JointScatteringFeatures {
    /// Features from row-wise scattering
    pub row_features: Vec<ScatteringFeatures>,
    /// Features from column-wise scattering
    pub col_features: Vec<ScatteringFeatures>,
}

impl JointScatteringFeatures {
    /// Compute joint scattering features for a 2D signal (row-major layout).
    ///
    /// # Arguments
    /// * `data` - 2D signal in row-major order
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// * `config` - Scattering configuration
    /// * `normalization` - Normalization method
    pub fn compute(
        data: &[f64],
        rows: usize,
        cols: usize,
        config: ScatteringConfig,
        normalization: FeatureNormalization,
    ) -> FFTResult<Self> {
        if data.len() != rows * cols {
            return Err(FFTError::DimensionError(format!(
                "data length {} does not match rows={} * cols={}",
                data.len(),
                rows,
                cols
            )));
        }

        // Row-wise scattering
        let row_extractor = ScatteringFeatureExtractor::new(
            config.clone(),
            cols,
            normalization,
            TimeFrequencyMode::TimeAveraged,
        )?;

        let mut row_features = Vec::with_capacity(rows);
        for r in 0..rows {
            let row_data = &data[r * cols..(r + 1) * cols];
            let features = row_extractor.extract(row_data)?;
            row_features.push(features);
        }

        // Column-wise scattering
        let col_extractor = ScatteringFeatureExtractor::new(
            config,
            rows,
            normalization,
            TimeFrequencyMode::TimeAveraged,
        )?;

        let mut col_features = Vec::with_capacity(cols);
        for c in 0..cols {
            let col_data: Vec<f64> = (0..rows).map(|r| data[r * cols + c]).collect();
            let features = col_extractor.extract(&col_data)?;
            col_features.push(features);
        }

        Ok(Self {
            row_features,
            col_features,
        })
    }

    /// Flatten into a single feature vector by concatenating row and column features.
    pub fn flatten(&self) -> Vec<f64> {
        let mut result = Vec::new();
        for f in &self.row_features {
            result.extend_from_slice(&f.feature_vector);
        }
        for f in &self.col_features {
            result.extend_from_slice(&f.feature_vector);
        }
        result
    }
}

/// Apply log-scattering normalization: x -> log(1 + |x|)
fn apply_log_normalization(matrix: &mut [Vec<f64>]) {
    for row in matrix.iter_mut() {
        for v in row.iter_mut() {
            *v = (1.0 + v.abs()).ln();
        }
    }
}

/// Apply L2 normalization to each row independently.
fn apply_l2_normalization(matrix: &mut [Vec<f64>]) {
    for row in matrix.iter_mut() {
        let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for v in row.iter_mut() {
                *v /= norm;
            }
        }
    }
}

/// Apply standardization (zero mean, unit variance) to each row.
fn apply_standardization(matrix: &mut [Vec<f64>]) {
    for row in matrix.iter_mut() {
        if row.is_empty() {
            continue;
        }
        let n = row.len() as f64;
        let mean: f64 = row.iter().sum::<f64>() / n;
        let variance: f64 = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev > 1e-15 {
            for v in row.iter_mut() {
                *v = (*v - mean) / std_dev;
            }
        } else {
            // Constant row: set to zero
            for v in row.iter_mut() {
                *v = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_test_signal(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin() + 0.3 * (2.0 * PI * 20.0 * t).cos()
            })
            .collect()
    }

    #[test]
    fn test_log_normalization_handles_zeros() {
        let config = ScatteringConfig::new(2, vec![2]).with_max_order(1);
        let extractor = ScatteringFeatureExtractor::new(
            config,
            128,
            FeatureNormalization::Log,
            TimeFrequencyMode::TimeAveraged,
        )
        .expect("extractor creation should succeed");

        // Zero signal should produce finite features
        let signal = vec![0.0; 128];
        let features = extractor.extract(&signal).expect("extract should succeed");

        for v in &features.feature_vector {
            assert!(v.is_finite(), "log-scattering should handle zeros: got {v}");
        }
    }

    #[test]
    fn test_feature_extraction_time_averaged() {
        let config = ScatteringConfig::new(3, vec![4, 1]);
        let n = 256;
        let extractor = ScatteringFeatureExtractor::new(
            config,
            n,
            FeatureNormalization::None,
            TimeFrequencyMode::TimeAveraged,
        )
        .expect("extractor creation should succeed");

        let signal = make_test_signal(n);
        let features = extractor.extract(&signal).expect("extract should succeed");

        // Time-averaged: one value per path
        assert_eq!(features.dim(), features.num_paths);
        assert!(features.norm() > 0.0, "features should be non-trivial");
    }

    #[test]
    fn test_feature_extraction_time_series() {
        let config = ScatteringConfig::new(2, vec![2]).with_max_order(1);
        let n = 128;
        let extractor = ScatteringFeatureExtractor::new(
            config,
            n,
            FeatureNormalization::None,
            TimeFrequencyMode::TimeSeries,
        )
        .expect("extractor creation should succeed");

        let signal = make_test_signal(n);
        let features = extractor.extract(&signal).expect("extract should succeed");

        // Time-series: num_paths * output_length values
        assert_eq!(features.dim(), features.num_paths * features.output_length);
    }

    #[test]
    fn test_l2_normalization() {
        let mut matrix = vec![vec![3.0, 4.0], vec![0.0, 0.0], vec![1.0, 0.0]];
        apply_l2_normalization(&mut matrix);

        // [3,4] -> [0.6, 0.8] (norm=5)
        assert!((matrix[0][0] - 0.6).abs() < 1e-10);
        assert!((matrix[0][1] - 0.8).abs() < 1e-10);

        // [0,0] -> stays [0,0] (zero norm)
        assert!((matrix[1][0]).abs() < 1e-10);
        assert!((matrix[1][1]).abs() < 1e-10);

        // [1,0] -> [1,0] (norm=1)
        assert!((matrix[2][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_standardization() {
        let mut matrix = vec![vec![2.0, 4.0, 6.0]];
        apply_standardization(&mut matrix);

        // mean=4, std=sqrt(8/3)
        let mean: f64 = matrix[0].iter().sum::<f64>() / 3.0;
        assert!(
            mean.abs() < 1e-10,
            "standardized mean should be ~0, got {mean}"
        );

        let var: f64 = matrix[0].iter().map(|v| v * v).sum::<f64>() / 3.0;
        assert!(
            (var - 1.0).abs() < 1e-10,
            "standardized variance should be ~1, got {var}"
        );
    }

    #[test]
    fn test_log_standardize_normalization() {
        let config = ScatteringConfig::new(2, vec![2]).with_max_order(1);
        let extractor = ScatteringFeatureExtractor::new(
            config,
            128,
            FeatureNormalization::LogStandardize,
            TimeFrequencyMode::TimeAveraged,
        )
        .expect("extractor creation should succeed");

        let signal = make_test_signal(128);
        let features = extractor.extract(&signal).expect("extract should succeed");

        // Should produce finite, non-trivial features
        for v in &features.feature_vector {
            assert!(v.is_finite(), "LogStandardize should produce finite values");
        }
    }

    #[test]
    fn test_joint_scattering_features() {
        let rows = 16;
        let cols = 32;
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| {
                let r = (i / cols) as f64;
                let c = (i % cols) as f64;
                (2.0 * PI * r / rows as f64).sin() + (2.0 * PI * c / cols as f64).cos()
            })
            .collect();

        let config = ScatteringConfig::new(2, vec![2]).with_max_order(1);
        let joint =
            JointScatteringFeatures::compute(&data, rows, cols, config, FeatureNormalization::Log)
                .expect("joint scattering should succeed");

        assert_eq!(joint.row_features.len(), rows);
        assert_eq!(joint.col_features.len(), cols);

        let flat = joint.flatten();
        assert!(!flat.is_empty(), "joint features should not be empty");
        for v in &flat {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let config = ScatteringConfig::new(2, vec![2]).with_max_order(1);
        let result = JointScatteringFeatures::compute(
            &[1.0, 2.0, 3.0],
            2,
            3,
            config,
            FeatureNormalization::None,
        );
        assert!(result.is_err());
    }
}
