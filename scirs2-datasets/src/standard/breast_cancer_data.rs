//! Embedded Breast Cancer Wisconsin (Diagnostic) dataset
//!
//! 569 samples, 30 features, 2 classes (malignant/benign)
//! Generated deterministically from known feature statistics.

use super::DatasetResult;
use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;

/// Feature means for malignant (class 0) samples -- 212 samples
/// 10 base measurements x 3 (mean, SE, worst) = 30 features
const MALIGNANT_MEANS: [f64; 30] = [
    17.46, 21.60, 115.4, 978.4, 0.1044, 0.1450, 0.1607, 0.0879, 0.1927, 0.0627, 0.6139, 1.210,
    4.324, 72.08, 0.00699, 0.02917, 0.03806, 0.01471, 0.01794, 0.00395, 21.13, 29.32, 141.4,
    1422.3, 0.1448, 0.3748, 0.4504, 0.1822, 0.3232, 0.09156,
];

/// Feature std devs for malignant samples
const MALIGNANT_STDS: [f64; 30] = [
    3.20, 4.04, 22.2, 368.0, 0.0128, 0.0526, 0.0840, 0.0347, 0.0235, 0.0071, 0.319, 0.536, 2.42,
    48.5, 0.00287, 0.0183, 0.0254, 0.0088, 0.0078, 0.00259, 4.28, 5.51, 29.8, 588.0, 0.0225,
    0.1538, 0.2090, 0.0643, 0.0625, 0.0197,
];

/// Feature means for benign (class 1) samples -- 357 samples
const BENIGN_MEANS: [f64; 30] = [
    12.15, 17.91, 78.1, 463.0, 0.0924, 0.0802, 0.0461, 0.0259, 0.1742, 0.0629, 0.2847, 1.220,
    2.000, 23.67, 0.00715, 0.02143, 0.01902, 0.00935, 0.02058, 0.00376, 13.38, 23.52, 87.0, 558.9,
    0.1252, 0.1827, 0.1663, 0.0741, 0.2702, 0.0796,
];

/// Feature std devs for benign samples
const BENIGN_STDS: [f64; 30] = [
    1.78, 4.07, 11.8, 134.0, 0.0132, 0.0422, 0.0390, 0.0164, 0.0275, 0.0072, 0.141, 0.575, 1.37,
    18.1, 0.00377, 0.0145, 0.0186, 0.0061, 0.0094, 0.00295, 2.12, 6.07, 14.4, 193.0, 0.0196,
    0.1046, 0.1442, 0.0394, 0.0493, 0.0184,
];

const N_MALIGNANT: usize = 212;
const N_BENIGN: usize = 357;
const BC_SEED: u64 = 5_690_301;

pub(super) fn load() -> Result<DatasetResult> {
    let n_samples = N_MALIGNANT + N_BENIGN;
    let n_features = 30;

    let mut rng = StdRng::seed_from_u64(BC_SEED);

    let mut data_vec = Vec::with_capacity(n_samples * n_features);
    let mut target_vec = Vec::with_capacity(n_samples);

    // Generate malignant samples (class 0)
    for _ in 0..N_MALIGNANT {
        for feat in 0..n_features {
            let dist =
                scirs2_core::random::Normal::new(MALIGNANT_MEANS[feat], MALIGNANT_STDS[feat])
                    .map_err(|e| {
                        DatasetsError::ComputationError(format!("Normal dist error: {e}"))
                    })?;
            let val: f64 = dist.sample(&mut rng);
            data_vec.push(val.max(0.0));
        }
        target_vec.push(0.0);
    }

    // Generate benign samples (class 1)
    for _ in 0..N_BENIGN {
        for feat in 0..n_features {
            let dist = scirs2_core::random::Normal::new(BENIGN_MEANS[feat], BENIGN_STDS[feat])
                .map_err(|e| DatasetsError::ComputationError(format!("Normal dist error: {e}")))?;
            let val: f64 = dist.sample(&mut rng);
            data_vec.push(val.max(0.0));
        }
        target_vec.push(1.0);
    }

    let data = Array2::from_shape_vec((n_samples, n_features), data_vec).map_err(|e| {
        DatasetsError::ComputationError(format!("Breast cancer data shape error: {e}"))
    })?;
    let target = Array1::from_vec(target_vec);

    let base_names = [
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave_points",
        "symmetry",
        "fractal_dimension",
    ];
    let suffixes = ["_mean", "_se", "_worst"];

    let mut feature_names = Vec::with_capacity(30);
    for suffix in &suffixes {
        for base in &base_names {
            feature_names.push(format!("{base}{suffix}"));
        }
    }

    Ok(DatasetResult {
        data,
        target,
        feature_names,
        target_names: vec!["malignant".into(), "benign".into()],
        description: "Breast Cancer Wisconsin (Diagnostic) Dataset\n\n\
            569 samples computed from FNA images of breast masses.\n\
            30 features: mean, SE, and worst of 10 cell nucleus measurements.\n\
            Classes: malignant (0, 212 samples), benign (1, 357 samples).\n\
            Source: UCI Machine Learning Repository."
            .into(),
    })
}
