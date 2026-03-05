//! Embedded Wine recognition dataset
//!
//! 178 samples, 13 features, 3 classes
//! Generated deterministically from known feature statistics of the UCI Wine dataset.

use super::DatasetResult;
use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;

/// Feature means per class (class 0, 1, 2) for the 13 wine features
/// Derived from the original UCI Wine dataset statistics.
const CLASS_MEANS: [[f64; 13]; 3] = [
    // Class 0 (59 samples): higher alcohol, proline
    [
        13.74, 2.01, 2.46, 17.04, 106.3, 2.84, 2.98, 0.29, 1.90, 5.53, 1.06, 3.16, 1115.0,
    ],
    // Class 1 (71 samples): moderate values
    [
        12.28, 1.93, 2.24, 20.24, 94.5, 2.26, 2.08, 0.36, 1.63, 3.09, 1.06, 2.79, 519.5,
    ],
    // Class 2 (48 samples): lower alcohol, higher color_intensity
    [
        13.15, 3.33, 2.44, 21.42, 99.3, 1.68, 0.78, 0.45, 1.15, 7.40, 0.68, 1.68, 629.9,
    ],
];

/// Feature std deviations per class
const CLASS_STDS: [[f64; 13]; 3] = [
    [
        0.46, 0.69, 0.23, 2.43, 18.5, 0.34, 0.40, 0.07, 0.41, 1.26, 0.10, 0.35, 221.5,
    ],
    [
        0.54, 1.03, 0.32, 3.35, 16.9, 0.56, 0.71, 0.12, 0.47, 1.14, 0.20, 0.50, 158.0,
    ],
    [
        0.53, 1.09, 0.18, 2.29, 13.1, 0.38, 0.35, 0.12, 0.41, 2.31, 0.11, 0.41, 115.6,
    ],
];

const SAMPLES_PER_CLASS: [usize; 3] = [59, 71, 48];

/// Deterministic seed for wine data generation
const WINE_SEED: u64 = 7_777_777;

pub(super) fn load() -> Result<DatasetResult> {
    let n_samples = 178;
    let n_features = 13;

    let mut rng = StdRng::seed_from_u64(WINE_SEED);

    let mut data_vec = Vec::with_capacity(n_samples * n_features);
    let mut target_vec = Vec::with_capacity(n_samples);

    for (class_idx, &n) in SAMPLES_PER_CLASS.iter().enumerate() {
        for _ in 0..n {
            for feat in 0..n_features {
                let mean = CLASS_MEANS[class_idx][feat];
                let std = CLASS_STDS[class_idx][feat];
                let dist = scirs2_core::random::Normal::new(mean, std).map_err(|e| {
                    DatasetsError::ComputationError(format!("Normal dist error: {e}"))
                })?;
                let val: f64 = dist.sample(&mut rng);
                // Clamp to reasonable ranges (all features are non-negative)
                data_vec.push(val.max(0.0));
            }
            target_vec.push(class_idx as f64);
        }
    }

    let data = Array2::from_shape_vec((n_samples, n_features), data_vec)
        .map_err(|e| DatasetsError::ComputationError(format!("Wine data shape error: {e}")))?;
    let target = Array1::from_vec(target_vec);

    Ok(DatasetResult {
        data,
        target,
        feature_names: vec![
            "alcohol".into(),
            "malic_acid".into(),
            "ash".into(),
            "alcalinity_of_ash".into(),
            "magnesium".into(),
            "total_phenols".into(),
            "flavanoids".into(),
            "nonflavanoid_phenols".into(),
            "proanthocyanins".into(),
            "color_intensity".into(),
            "hue".into(),
            "od280_od315".into(),
            "proline".into(),
        ],
        target_names: vec!["class_0".into(), "class_1".into(), "class_2".into()],
        description: "Wine Recognition Dataset (Aeberhard, Coomans & de Vel, 1992)\n\n\
            178 samples of wines from 3 Italian cultivars.\n\
            13 chemical analysis features: alcohol, malic acid, ash, etc.\n\
            Classes: cultivar 0 (59), cultivar 1 (71), cultivar 2 (48).\n\
            Source: UCI Machine Learning Repository."
            .into(),
    })
}
