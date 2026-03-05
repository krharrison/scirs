//! Embedded Boston Housing dataset (Harrison & Rubinfeld, 1978)
//!
//! 506 samples, 13 features, regression target
//! Deterministically generated from known feature statistics.
//!
//! **DEPRECATION NOTE**: This dataset has known ethical concerns.
//! Consider using California Housing instead.

use super::DatasetResult;
use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;

const BOSTON_SEED: u64 = 5_060_013;
const N_SAMPLES: usize = 506;
const N_FEATURES: usize = 13;

/// Feature means from the original Boston dataset
const FEATURE_MEANS: [f64; 13] = [
    3.6135, // CRIM
    11.364, // ZN
    11.137, // INDUS
    0.0693, // CHAS
    0.5547, // NOX
    6.2846, // RM
    68.575, // AGE
    3.7950, // DIS
    9.5494, // RAD
    408.24, // TAX
    18.456, // PTRATIO
    356.67, // B
    12.653, // LSTAT
];

/// Feature std deviations from the original Boston dataset
const FEATURE_STDS: [f64; 13] = [
    8.6016, // CRIM
    23.322, // ZN
    6.8604, // INDUS
    0.2539, // CHAS
    0.1159, // NOX
    0.7026, // RM
    28.149, // AGE
    2.1057, // DIS
    8.7073, // RAD
    168.54, // TAX
    2.1649, // PTRATIO
    91.295, // B
    7.1411, // LSTAT
];

/// Feature minimums for clamping
const FEATURE_MINS: [f64; 13] = [
    0.006, 0.0, 0.46, 0.0, 0.385, 3.56, 2.9, 1.13, 1.0, 187.0, 12.6, 0.32, 1.73,
];

/// Feature maximums for clamping
const FEATURE_MAXS: [f64; 13] = [
    88.98, 100.0, 27.74, 1.0, 0.871, 8.78, 100.0, 12.13, 24.0, 711.0, 22.0, 396.9, 37.97,
];

/// Regression coefficients for target generation (approximating the true relationship)
const TARGET_COEFS: [f64; 13] = [
    -0.108, // CRIM (negative: more crime → lower price)
    0.046,  // ZN
    0.021,  // INDUS
    2.687,  // CHAS (near river → higher price)
    -17.77, // NOX (pollution → lower price)
    3.810,  // RM (more rooms → higher price)
    0.001,  // AGE
    -1.476, // DIS
    0.306,  // RAD
    -0.012, // TAX (higher tax → lower price)
    -0.953, // PTRATIO (higher ratio → lower price)
    0.009,  // B
    -0.525, // LSTAT (lower status → lower price)
];

const TARGET_INTERCEPT: f64 = 36.46;
const TARGET_MEAN: f64 = 22.53;

pub(super) fn load() -> Result<DatasetResult> {
    let mut rng = StdRng::seed_from_u64(BOSTON_SEED);

    let normal_01 = scirs2_core::random::Normal::new(0.0, 1.0)
        .map_err(|e| DatasetsError::ComputationError(format!("Normal dist error: {e}")))?;

    let mut data_vec = Vec::with_capacity(N_SAMPLES * N_FEATURES);
    let mut target_vec = Vec::with_capacity(N_SAMPLES);

    for _ in 0..N_SAMPLES {
        let mut features = [0.0f64; N_FEATURES];

        // Generate correlated features
        let z: f64 = normal_01.sample(&mut rng); // Shared latent factor

        for feat in 0..N_FEATURES {
            let mean = FEATURE_MEANS[feat];
            let std = FEATURE_STDS[feat];

            // Mix shared factor with independent noise for mild correlations
            let shared_weight = 0.3;
            let indep: f64 = normal_01.sample(&mut rng);
            let raw = mean
                + std * (shared_weight * z + (1.0 - shared_weight * shared_weight).sqrt() * indep);

            // CHAS is binary
            let val = if feat == 3 {
                if raw > mean {
                    1.0
                } else {
                    0.0
                }
            } else {
                raw.clamp(FEATURE_MINS[feat], FEATURE_MAXS[feat])
            };

            features[feat] = val;
            data_vec.push(val);
        }

        // Generate target from linear model + noise
        let mut y = TARGET_INTERCEPT;
        for feat in 0..N_FEATURES {
            y += TARGET_COEFS[feat] * features[feat];
        }
        // Add noise and shift to match target distribution
        let noise: f64 = normal_01.sample(&mut rng) * 4.5;
        y += noise;
        // Clamp to realistic range [5, 50]
        y = y.clamp(5.0, 50.0);
        target_vec.push(y);
    }

    // Rescale targets to have correct mean
    let current_mean: f64 = target_vec.iter().sum::<f64>() / N_SAMPLES as f64;
    let shift = TARGET_MEAN - current_mean;
    for v in &mut target_vec {
        *v = (*v + shift).clamp(5.0, 50.0);
    }

    let data = Array2::from_shape_vec((N_SAMPLES, N_FEATURES), data_vec)
        .map_err(|e| DatasetsError::ComputationError(format!("Boston data shape error: {e}")))?;
    let target = Array1::from_vec(target_vec);

    Ok(DatasetResult {
        data,
        target,
        feature_names: vec![
            "CRIM".into(),
            "ZN".into(),
            "INDUS".into(),
            "CHAS".into(),
            "NOX".into(),
            "RM".into(),
            "AGE".into(),
            "DIS".into(),
            "RAD".into(),
            "TAX".into(),
            "PTRATIO".into(),
            "B".into(),
            "LSTAT".into(),
        ],
        target_names: vec!["MEDV".into()],
        description: "Boston Housing Dataset (Harrison & Rubinfeld, 1978)\n\n\
            **DEPRECATED**: This dataset has ethical concerns regarding the B variable.\n\
            Consider using California Housing instead.\n\n\
            506 samples, 13 features. Target: median home value in $1000s.\n\
            Features: crime rate, zoning, industry, Charles River proximity,\n\
            NOx concentration, rooms, age, distance, highway access, tax rate,\n\
            pupil-teacher ratio, demographic index, lower status percentage."
            .into(),
    })
}
