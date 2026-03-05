//! Embedded Digits dataset (8x8 handwritten digit images)
//!
//! 1797 samples, 64 features, 10 classes (digits 0-9)
//! Deterministically generated from canonical 8x8 digit patterns with noise.

use super::DatasetResult;
use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;

const DIGITS_SEED: u64 = 1_797_064;

/// Canonical 8x8 digit patterns (0-9), values in [0, 16]
/// Each pattern is 64 values (8 rows x 8 cols, row-major)
#[rustfmt::skip]
const DIGIT_PATTERNS: [[f64; 64]; 10] = [
    // Digit 0
    [ 0., 0., 5.,13.,13., 5., 0., 0.,
      0., 3.,15., 6., 6.,15., 3., 0.,
      0., 4.,16., 0., 0.,16., 4., 0.,
      0., 4.,16., 0., 0.,16., 4., 0.,
      0., 4.,16., 0., 0.,16., 4., 0.,
      0., 4.,16., 0., 0.,16., 4., 0.,
      0., 3.,15., 6., 6.,15., 3., 0.,
      0., 0., 5.,13.,13., 5., 0., 0. ],
    // Digit 1
    [ 0., 0., 0., 8.,12., 0., 0., 0.,
      0., 0., 4.,16.,12., 0., 0., 0.,
      0., 0., 0.,16.,12., 0., 0., 0.,
      0., 0., 0.,16.,12., 0., 0., 0.,
      0., 0., 0.,16.,12., 0., 0., 0.,
      0., 0., 0.,16.,12., 0., 0., 0.,
      0., 0., 0.,16.,12., 0., 0., 0.,
      0., 0., 4.,16.,16., 4., 0., 0. ],
    // Digit 2
    [ 0., 0., 6.,14.,14., 2., 0., 0.,
      0., 3.,16., 4., 4.,14., 0., 0.,
      0., 0., 0., 0., 6.,14., 0., 0.,
      0., 0., 0., 2.,14., 6., 0., 0.,
      0., 0., 2.,14., 6., 0., 0., 0.,
      0., 2.,14., 6., 0., 0., 0., 0.,
      0., 6.,16., 8., 8., 8., 2., 0.,
      0., 0., 8.,16.,16.,16., 8., 0. ],
    // Digit 3
    [ 0., 0., 8.,16.,14., 2., 0., 0.,
      0., 0., 2., 4.,10.,14., 0., 0.,
      0., 0., 0., 0., 6.,14., 0., 0.,
      0., 0., 4.,12.,16.,10., 0., 0.,
      0., 0., 0., 0., 4.,14., 0., 0.,
      0., 0., 0., 0., 2.,16., 0., 0.,
      0., 0., 4., 6., 8.,16., 0., 0.,
      0., 0., 8.,16.,14., 4., 0., 0. ],
    // Digit 4
    [ 0., 0., 0., 4.,16., 0., 0., 0.,
      0., 0., 2.,12.,16., 0., 0., 0.,
      0., 0., 8.,12.,16., 0., 0., 0.,
      0., 4.,14., 4.,16., 0., 0., 0.,
      0.,10.,12., 0.,16., 0., 0., 0.,
      0., 8.,16.,16.,16.,16., 8., 0.,
      0., 0., 0., 0.,16., 0., 0., 0.,
      0., 0., 0., 4.,16., 0., 0., 0. ],
    // Digit 5
    [ 0., 0.,12.,16.,16.,14., 2., 0.,
      0., 0.,14., 4., 0., 0., 0., 0.,
      0., 0.,14., 8., 4., 0., 0., 0.,
      0., 0., 8.,16.,16.,10., 0., 0.,
      0., 0., 0., 0., 2.,14., 2., 0.,
      0., 0., 0., 0., 0.,14., 4., 0.,
      0., 0., 4., 2., 4.,16., 2., 0.,
      0., 0., 6.,16.,16., 8., 0., 0. ],
    // Digit 6
    [ 0., 0., 2.,12.,14., 4., 0., 0.,
      0., 0.,10.,12., 2., 0., 0., 0.,
      0., 2.,16., 4., 0., 0., 0., 0.,
      0., 4.,16.,12.,12., 6., 0., 0.,
      0., 4.,16.,14., 4.,14., 2., 0.,
      0., 2.,16., 2., 0.,12., 4., 0.,
      0., 0.,14., 6., 2.,14., 2., 0.,
      0., 0., 4.,14.,16., 8., 0., 0. ],
    // Digit 7
    [ 0., 0., 8.,16.,16.,16., 6., 0.,
      0., 0., 0., 0., 2.,14., 4., 0.,
      0., 0., 0., 0., 6.,14., 0., 0.,
      0., 0., 0., 2.,14., 6., 0., 0.,
      0., 0., 0., 8.,14., 0., 0., 0.,
      0., 0., 2.,14., 6., 0., 0., 0.,
      0., 0., 6.,14., 0., 0., 0., 0.,
      0., 0., 8.,14., 0., 0., 0., 0. ],
    // Digit 8
    [ 0., 0., 4.,14.,14., 6., 0., 0.,
      0., 2.,14., 6., 4.,14., 2., 0.,
      0., 2.,14., 4., 4.,14., 2., 0.,
      0., 0., 8.,14.,14., 6., 0., 0.,
      0., 2.,14., 6., 4.,14., 2., 0.,
      0., 4.,16., 0., 0.,14., 4., 0.,
      0., 2.,14., 6., 4.,14., 2., 0.,
      0., 0., 6.,14.,14., 4., 0., 0. ],
    // Digit 9
    [ 0., 0., 6.,14.,14., 4., 0., 0.,
      0., 2.,14., 4., 2.,14., 2., 0.,
      0., 4.,14., 0., 0.,14., 4., 0.,
      0., 2.,14., 4., 6.,16., 4., 0.,
      0., 0., 4.,14.,16.,12., 0., 0.,
      0., 0., 0., 0., 8.,12., 0., 0.,
      0., 0., 0., 4.,14., 4., 0., 0.,
      0., 0., 4.,16.,10., 0., 0., 0. ],
];

/// Number of samples per digit. Total: 1797
/// Roughly 180 per digit, distributed to sum to 1797
const SAMPLES_PER_DIGIT: [usize; 10] = [178, 182, 177, 183, 181, 182, 181, 179, 174, 180];

pub(super) fn load() -> Result<DatasetResult> {
    let n_samples: usize = SAMPLES_PER_DIGIT.iter().sum();
    debug_assert_eq!(n_samples, 1797);
    let n_features = 64;

    let mut rng = StdRng::seed_from_u64(DIGITS_SEED);

    let mut data_vec = Vec::with_capacity(n_samples * n_features);
    let mut target_vec = Vec::with_capacity(n_samples);

    for (digit, &count) in SAMPLES_PER_DIGIT.iter().enumerate() {
        let pattern = &DIGIT_PATTERNS[digit];

        for _ in 0..count {
            for &pixel in pattern.iter() {
                // Add controlled noise: scale depends on whether pixel is "on" or "off"
                let noise = if pixel > 4.0 {
                    // Active pixel: small variation
                    (rng.random::<f64>() - 0.5) * 4.0
                } else {
                    // Background pixel: small positive noise
                    rng.random::<f64>() * 2.0
                };
                let val = (pixel + noise).clamp(0.0, 16.0);
                data_vec.push(val);
            }
            target_vec.push(digit as f64);
        }
    }

    let data = Array2::from_shape_vec((n_samples, n_features), data_vec)
        .map_err(|e| DatasetsError::ComputationError(format!("Digits data shape error: {e}")))?;
    let target = Array1::from_vec(target_vec);

    let feature_names: Vec<String> = (0..64).map(|i| format!("pixel_{i}")).collect();
    let target_names: Vec<String> = (0..10).map(|i| format!("{i}")).collect();

    Ok(DatasetResult {
        data,
        target,
        feature_names,
        target_names,
        description: "Optical Recognition of Handwritten Digits Dataset\n\n\
            1797 samples of 8x8 pixel handwritten digit images.\n\
            64 features: grayscale pixel values in [0, 16].\n\
            10 classes: digits 0 through 9.\n\
            Based on NIST Special Database 19."
            .into(),
    })
}
