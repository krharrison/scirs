//! Feature engineering for energy load forecasting
//!
//! Extracts calendar, temperature, lag, and Fourier features from load profiles.

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::Array2;

use super::types::LoadProfile;

/// Configuration for feature extraction
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Lag indices to include (e.g., 24 = same hour yesterday)
    pub lags: Vec<usize>,
    /// Number of Fourier harmonics
    pub fourier_terms: usize,
    /// Base temperature for HDD/CDD calculation (degrees F)
    pub base_temperature: f64,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            lags: vec![24, 48, 168],
            fourier_terms: 3,
            base_temperature: 65.0,
        }
    }
}

/// Calendar-based feature extraction
pub struct CalendarFeatures;

impl CalendarFeatures {
    /// Extract calendar features from timestamps.
    ///
    /// Each timestamp is treated as an hour index from an arbitrary epoch.
    /// Returns vectors for: hour_of_day, day_of_week, month, is_weekend.
    pub fn extract(timestamps: &[f64]) -> Vec<Vec<f64>> {
        let n = timestamps.len();
        let mut hour_of_day = Vec::with_capacity(n);
        let mut day_of_week = Vec::with_capacity(n);
        let mut month = Vec::with_capacity(n);
        let mut is_weekend = Vec::with_capacity(n);

        for &t in timestamps {
            let hour_idx = t.floor() as i64;
            // hour of day: 0..23
            let hod = ((hour_idx % 24) + 24) % 24;
            hour_of_day.push(hod as f64);

            // day of week: 0..6 (each day = 24 hours)
            let day = ((hour_idx / 24) % 7 + 7) % 7;
            day_of_week.push(day as f64);

            // month approximation: 1..12 (each month ≈ 730 hours)
            let m = ((hour_idx / 730) % 12 + 12) % 12 + 1;
            month.push(m as f64);

            // weekend: days 5 and 6
            let weekend = if day >= 5 { 1.0 } else { 0.0 };
            is_weekend.push(weekend);
        }

        vec![hour_of_day, day_of_week, month, is_weekend]
    }
}

/// Temperature-based feature extraction
pub struct TemperatureFeatures;

impl TemperatureFeatures {
    /// Extract temperature features: HDD, CDD, T².
    ///
    /// - HDD = max(0, base_temp - T)
    /// - CDD = max(0, T - base_temp)
    /// - T² = temperature squared
    pub fn extract(temps: &[f64], base_temp: f64) -> Vec<Vec<f64>> {
        let n = temps.len();
        let mut hdd = Vec::with_capacity(n);
        let mut cdd = Vec::with_capacity(n);
        let mut t_sq = Vec::with_capacity(n);

        for &t in temps {
            hdd.push((base_temp - t).max(0.0));
            cdd.push((t - base_temp).max(0.0));
            t_sq.push(t * t);
        }

        vec![hdd, cdd, t_sq]
    }
}

/// Lag-based feature extraction
pub struct LagFeatures;

impl LagFeatures {
    /// Extract lagged values of a series.
    ///
    /// For each lag in `lags`, produces a vector where entry i = series\[i - lag\]
    /// or 0.0 if i < lag.
    pub fn extract(series: &[f64], lags: &[usize]) -> Vec<Vec<f64>> {
        let n = series.len();
        let mut result = Vec::with_capacity(lags.len());

        for &lag in lags {
            let mut lagged = Vec::with_capacity(n);
            for i in 0..n {
                if i >= lag {
                    lagged.push(series[i - lag]);
                } else {
                    lagged.push(0.0);
                }
            }
            result.push(lagged);
        }

        result
    }
}

/// Fourier harmonic feature extraction
pub struct FourierFeatures;

impl FourierFeatures {
    /// Generate Fourier basis features.
    ///
    /// For k = 1..=n_terms, produces sin(2πk·i/period) and cos(2πk·i/period)
    /// for i = 0..n.
    pub fn extract(n: usize, period: f64, n_terms: usize) -> Vec<Vec<f64>> {
        let mut result = Vec::with_capacity(2 * n_terms);

        for k in 1..=n_terms {
            let mut sin_vals = Vec::with_capacity(n);
            let mut cos_vals = Vec::with_capacity(n);
            let freq = 2.0 * std::f64::consts::PI * (k as f64) / period;

            for i in 0..n {
                sin_vals.push((freq * i as f64).sin());
                cos_vals.push((freq * i as f64).cos());
            }

            result.push(sin_vals);
            result.push(cos_vals);
        }

        result
    }
}

/// Feature matrix builder
pub struct FeatureMatrix;

impl FeatureMatrix {
    /// Build a combined feature matrix from a load profile and configuration.
    ///
    /// Concatenates calendar, temperature (if available), lag, and Fourier
    /// features column-wise into an Array2.
    pub fn build(profile: &LoadProfile, config: &FeatureConfig) -> Result<Array2<f64>> {
        let n = profile.load_values.len();
        if n == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Load profile must have at least one observation".to_string(),
            ));
        }
        if profile.timestamps.len() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: profile.timestamps.len(),
            });
        }

        let mut columns: Vec<Vec<f64>> = Vec::new();

        // Calendar features
        let cal = CalendarFeatures::extract(&profile.timestamps);
        columns.extend(cal);

        // Temperature features (if available)
        if let Some(ref temps) = profile.temperatures {
            if temps.len() != n {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: n,
                    actual: temps.len(),
                });
            }
            let temp_feats = TemperatureFeatures::extract(temps, config.base_temperature);
            columns.extend(temp_feats);
        }

        // Lag features
        let lag_feats = LagFeatures::extract(&profile.load_values, &config.lags);
        columns.extend(lag_feats);

        // Fourier features (period = 24 for hourly data)
        let fourier_feats = FourierFeatures::extract(n, 24.0, config.fourier_terms);
        columns.extend(fourier_feats);

        // Build Array2 from columns
        let n_cols = columns.len();
        let mut data = Vec::with_capacity(n * n_cols);
        for row in 0..n {
            for col in &columns {
                data.push(col[row]);
            }
        }

        Array2::from_shape_vec((n, n_cols), data).map_err(|e| {
            TimeSeriesError::ComputationError(format!("Failed to build feature matrix: {}", e))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calendar_features_known_timestamp() {
        // hour 50: hour_of_day = 50 % 24 = 2, day_of_week = 50/24 = 2
        let feats = CalendarFeatures::extract(&[50.0]);
        assert!((feats[0][0] - 2.0).abs() < 1e-12, "hour_of_day");
        assert!((feats[1][0] - 2.0).abs() < 1e-12, "day_of_week");
    }

    #[test]
    fn test_hdd_cdd() {
        let feats = TemperatureFeatures::extract(&[50.0], 65.0);
        assert!((feats[0][0] - 15.0).abs() < 1e-12, "HDD should be 15");
        assert!((feats[1][0]).abs() < 1e-12, "CDD should be 0");
        assert!((feats[2][0] - 2500.0).abs() < 1e-12, "T^2 should be 2500");
    }

    #[test]
    fn test_hdd_cdd_hot() {
        let feats = TemperatureFeatures::extract(&[80.0], 65.0);
        assert!((feats[0][0]).abs() < 1e-12, "HDD should be 0");
        assert!((feats[1][0] - 15.0).abs() < 1e-12, "CDD should be 15");
    }

    #[test]
    fn test_lag_features() {
        let series = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let lags = vec![1, 2];
        let feats = LagFeatures::extract(&series, &lags);
        // lag=1: [0, 10, 20, 30, 40]
        assert!((feats[0][0]).abs() < 1e-12, "lag1[0] should be 0 (padding)");
        assert!((feats[0][1] - 10.0).abs() < 1e-12);
        // lag=2: [0, 0, 10, 20, 30]
        assert!((feats[1][0]).abs() < 1e-12);
        assert!((feats[1][1]).abs() < 1e-12);
        assert!((feats[1][2] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_fourier_orthogonality() {
        // Sum of sin(k*x)*cos(k*x) over full period should be near 0
        let n = 24;
        let feats = FourierFeatures::extract(n, 24.0, 1);
        let sin_vals = &feats[0];
        let cos_vals = &feats[1];
        let dot: f64 = sin_vals.iter().zip(cos_vals).map(|(s, c)| s * c).sum();
        assert!(
            dot.abs() < 1e-10,
            "sin and cos should be orthogonal, got {}",
            dot
        );
    }

    #[test]
    fn test_feature_matrix_build() {
        let profile = LoadProfile {
            timestamps: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            load_values: vec![100.0, 110.0, 105.0, 120.0, 115.0],
            temperatures: Some(vec![60.0, 62.0, 58.0, 70.0, 65.0]),
            holiday_mask: None,
        };
        let config = FeatureConfig {
            lags: vec![1, 2],
            fourier_terms: 1,
            base_temperature: 65.0,
        };
        let mat = FeatureMatrix::build(&profile, &config).expect("should build");
        // 4 calendar + 3 temp + 2 lag + 2 fourier = 11 columns
        assert_eq!(mat.shape(), &[5, 11]);
    }

    #[test]
    fn test_feature_config_default() {
        let cfg = FeatureConfig::default();
        assert_eq!(cfg.lags, vec![24, 48, 168]);
        assert_eq!(cfg.fourier_terms, 3);
        assert!((cfg.base_temperature - 65.0).abs() < 1e-12);
    }
}
