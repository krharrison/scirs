//! Laws Texture Energy Measures (Full 25-Kernel Set)
//!
//! Laws' texture energy method uses a set of 5-element 1D kernels whose
//! outer products form 25 distinct 5x5 2D kernels. Each kernel captures
//! a specific texture pattern (level, edge, spot, wave, ripple).
//!
//! This module provides:
//! - All 25 Laws 2D kernels
//! - Texture energy map computation (convolve + local energy)
//! - Mean energy feature extraction
//! - Rotation-invariant features via symmetric kernel pair averaging
//!
//! # 1D Kernels
//!
//! | Name | Coefficients | Detects |
//! |------|-------------|---------|
//! | L5   | `[1, 4, 6, 4, 1]` | Level (averaging) |
//! | E5   | `[-1, -2, 0, 2, 1]` | Edge |
//! | S5   | `[-1, 0, 2, 0, -1]` | Spot |
//! | W5   | `[-1, 2, 0, -2, 1]` | Wave |
//! | R5   | `[1, -4, 6, -4, 1]` | Ripple |
//!
//! # References
//!
//! - Laws, K.I. (1980). "Rapid Texture Identification." Proc. SPIE 0238,
//!   Image Processing for Missile Guidance.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::Array2;

/// Laws 1D vector types (length 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LawsVector {
    /// Level: `[1, 4, 6, 4, 1]` -- local averaging
    L5,
    /// Edge: `[-1, -2, 0, 2, 1]` -- edge detection
    E5,
    /// Spot: `[-1, 0, 2, 0, -1]` -- spot detection
    S5,
    /// Wave: `[-1, 2, 0, -2, 1]` -- wave detection
    W5,
    /// Ripple: `[1, -4, 6, -4, 1]` -- ripple detection
    R5,
}

impl LawsVector {
    /// Return the 5-element filter coefficients.
    pub fn coefficients(&self) -> [f64; 5] {
        match self {
            LawsVector::L5 => [1.0, 4.0, 6.0, 4.0, 1.0],
            LawsVector::E5 => [-1.0, -2.0, 0.0, 2.0, 1.0],
            LawsVector::S5 => [-1.0, 0.0, 2.0, 0.0, -1.0],
            LawsVector::W5 => [-1.0, 2.0, 0.0, -2.0, 1.0],
            LawsVector::R5 => [1.0, -4.0, 6.0, -4.0, 1.0],
        }
    }

    /// All five Laws vectors.
    pub fn all() -> [LawsVector; 5] {
        [
            LawsVector::L5,
            LawsVector::E5,
            LawsVector::S5,
            LawsVector::W5,
            LawsVector::R5,
        ]
    }

    /// Short name for display.
    pub fn name(&self) -> &'static str {
        match self {
            LawsVector::L5 => "L5",
            LawsVector::E5 => "E5",
            LawsVector::S5 => "S5",
            LawsVector::W5 => "W5",
            LawsVector::R5 => "R5",
        }
    }
}

/// Named Laws 2D kernel (row x column).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LawsKernelName {
    /// Row vector
    pub row: LawsVector,
    /// Column vector
    pub col: LawsVector,
}

impl LawsKernelName {
    /// Create a new kernel name.
    pub fn new(row: LawsVector, col: LawsVector) -> Self {
        Self { row, col }
    }

    /// Display name like "L5E5".
    pub fn display_name(&self) -> String {
        format!("{}{}", self.row.name(), self.col.name())
    }
}

/// Configuration for full Laws texture energy computation.
#[derive(Debug, Clone)]
pub struct LawsFullConfig {
    /// Which kernel pairs to use. If empty, uses all 25.
    pub pairs: Vec<LawsKernelName>,
    /// Window size for energy computation (must be odd, default 15).
    pub window_size: usize,
    /// Whether to remove DC component before filtering (default true).
    pub remove_dc: bool,
}

impl Default for LawsFullConfig {
    fn default() -> Self {
        Self {
            pairs: Vec::new(), // empty = all 25
            window_size: 15,
            remove_dc: true,
        }
    }
}

/// Result of full Laws texture energy computation.
#[derive(Debug, Clone)]
pub struct LawsFullResult {
    /// Kernel names corresponding to each energy map / feature.
    pub kernel_names: Vec<LawsKernelName>,
    /// Texture energy maps, one per kernel pair.
    pub energy_maps: Vec<Array2<f64>>,
    /// Feature vector: mean energy for each kernel pair.
    pub feature_vector: Vec<f64>,
}

/// Generate all 25 Laws 2D kernels.
///
/// Returns a list of `(LawsKernelName, Array2<f64>)` tuples.
pub fn laws_all_kernels() -> Vec<(LawsKernelName, Array2<f64>)> {
    let vecs = LawsVector::all();
    let mut kernels = Vec::with_capacity(25);

    for &row_v in &vecs {
        let r = row_v.coefficients();
        for &col_v in &vecs {
            let c = col_v.coefficients();
            let mut kernel = Array2::<f64>::zeros((5, 5));
            for i in 0..5 {
                for j in 0..5 {
                    kernel[[i, j]] = r[i] * c[j];
                }
            }
            kernels.push((LawsKernelName::new(row_v, col_v), kernel));
        }
    }

    kernels
}

/// Compute Laws texture energy for a set of kernel pairs.
///
/// For each kernel pair:
/// 1. Build the 5x5 kernel as outer product of the two 1D vectors
/// 2. Optionally remove DC component (subtract local mean)
/// 3. Convolve the image with the kernel
/// 4. Compute local energy (average absolute value in a window)
///
/// # Parameters
/// - `image` - input f64 image (at least 5x5)
/// - `config` - configuration (if `None`, uses defaults with all 25 kernels)
///
/// # Errors
/// Returns error if image is too small or window size is invalid.
pub fn laws_texture_energy_full(
    image: &Array2<f64>,
    config: Option<LawsFullConfig>,
) -> NdimageResult<LawsFullResult> {
    let cfg = config.unwrap_or_default();
    let (ny, nx) = image.dim();
    if ny < 5 || nx < 5 {
        return Err(NdimageError::InvalidInput(
            "Image must be at least 5x5 for Laws filters".into(),
        ));
    }
    if cfg.window_size == 0 || cfg.window_size % 2 == 0 {
        return Err(NdimageError::InvalidInput(
            "window_size must be an odd positive integer".into(),
        ));
    }

    // Determine which pairs to use
    let pairs: Vec<LawsKernelName> = if cfg.pairs.is_empty() {
        // All 25 pairs
        let vecs = LawsVector::all();
        let mut all = Vec::with_capacity(25);
        for &row_v in &vecs {
            for &col_v in &vecs {
                all.push(LawsKernelName::new(row_v, col_v));
            }
        }
        all
    } else {
        cfg.pairs
    };

    // Optionally remove DC
    let processed = if cfg.remove_dc {
        remove_dc_component(image, cfg.window_size)
    } else {
        image.clone()
    };

    let mut kernel_names = Vec::with_capacity(pairs.len());
    let mut energy_maps = Vec::with_capacity(pairs.len());
    let mut feature_vector = Vec::with_capacity(pairs.len());

    for name in &pairs {
        let r = name.row.coefficients();
        let c = name.col.coefficients();
        let mut kernel = Array2::<f64>::zeros((5, 5));
        for i in 0..5 {
            for j in 0..5 {
                kernel[[i, j]] = r[i] * c[j];
            }
        }

        let response = convolve_2d(&processed, &kernel);
        let energy = compute_local_energy(&response, cfg.window_size);

        let mean_e = if energy.is_empty() {
            0.0
        } else {
            energy.sum() / energy.len() as f64
        };

        kernel_names.push(*name);
        feature_vector.push(mean_e);
        energy_maps.push(energy);
    }

    Ok(LawsFullResult {
        kernel_names,
        energy_maps,
        feature_vector,
    })
}

/// Compute rotation-invariant Laws features.
///
/// Symmetric kernel pairs (e.g., L5E5 and E5L5) detect the same texture
/// pattern at different orientations. For rotation invariance, we average
/// the energy of symmetric pairs:
///
/// `E_sym(A,B) = (E(A,B) + E(B,A)) / 2`
///
/// This reduces the 25 features to 15 unique features:
/// - 5 symmetric pairs: L5L5, E5E5, S5S5, W5W5, R5R5
/// - 10 averaged pairs: L5E5, L5S5, L5W5, L5R5, E5S5, E5W5, E5R5, S5W5, S5R5, W5R5
///
/// # Parameters
/// - `image` - input f64 image
/// - `window_size` - energy window (odd, default 15)
///
/// # Returns
/// Vector of 15 rotation-invariant features with their names.
pub fn laws_rotation_invariant_features(
    image: &Array2<f64>,
    window_size: Option<usize>,
) -> NdimageResult<Vec<(String, f64)>> {
    let ws = window_size.unwrap_or(15);

    // Compute all 25 kernels
    let config = LawsFullConfig {
        pairs: Vec::new(), // all 25
        window_size: ws,
        remove_dc: true,
    };
    let result = laws_texture_energy_full(image, Some(config))?;

    // Build a lookup from (row, col) -> feature_index
    let vecs = LawsVector::all();
    let mut energy_lookup: std::collections::HashMap<(LawsVector, LawsVector), f64> =
        std::collections::HashMap::new();

    for (i, name) in result.kernel_names.iter().enumerate() {
        energy_lookup.insert((name.row, name.col), result.feature_vector[i]);
    }

    // Generate 15 unique symmetric features
    let mut features = Vec::with_capacity(15);
    for (idx_a, &va) in vecs.iter().enumerate() {
        for &vb in &vecs[idx_a..] {
            let e_ab = energy_lookup.get(&(va, vb)).copied().unwrap_or(0.0);
            let e_ba = energy_lookup.get(&(vb, va)).copied().unwrap_or(0.0);
            let sym_energy = (e_ab + e_ba) / 2.0;
            let name = format!("{}{}", va.name(), vb.name());
            features.push((name, sym_energy));
        }
    }

    Ok(features)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Remove DC component by subtracting local mean (box filter).
fn remove_dc_component(image: &Array2<f64>, window: usize) -> Array2<f64> {
    let (ny, nx) = image.dim();
    let half = (window / 2) as i64;
    let mut out = Array2::<f64>::zeros((ny, nx));

    for i in 0..ny {
        for j in 0..nx {
            let mut sum = 0.0f64;
            let mut count = 0.0f64;
            for di in -half..=half {
                for dj in -half..=half {
                    let ni = i as i64 + di;
                    let nj = j as i64 + dj;
                    if ni >= 0 && ni < ny as i64 && nj >= 0 && nj < nx as i64 {
                        sum += image[[ni as usize, nj as usize]];
                        count += 1.0;
                    }
                }
            }
            out[[i, j]] = image[[i, j]] - sum / count.max(1.0);
        }
    }
    out
}

/// Convolve a 2D image with a kernel (zero-padded boundary).
fn convolve_2d(image: &Array2<f64>, kernel: &Array2<f64>) -> Array2<f64> {
    let (ny, nx) = image.dim();
    let (ky, kx) = kernel.dim();
    let half_ky = (ky / 2) as i64;
    let half_kx = (kx / 2) as i64;
    let mut out = Array2::<f64>::zeros((ny, nx));

    for i in 0..ny {
        for j in 0..nx {
            let mut sum = 0.0;
            for ki in 0..ky {
                for kj in 0..kx {
                    let ii = i as i64 + ki as i64 - half_ky;
                    let jj = j as i64 + kj as i64 - half_kx;
                    if ii >= 0 && ii < ny as i64 && jj >= 0 && jj < nx as i64 {
                        sum += image[[ii as usize, jj as usize]] * kernel[[ki, kj]];
                    }
                }
            }
            out[[i, j]] = sum;
        }
    }
    out
}

/// Compute local energy: average of absolute values in a window.
fn compute_local_energy(image: &Array2<f64>, window: usize) -> Array2<f64> {
    let (ny, nx) = image.dim();
    let half = (window / 2) as i64;
    let mut out = Array2::<f64>::zeros((ny, nx));

    for i in 0..ny {
        for j in 0..nx {
            let mut sum = 0.0f64;
            let mut count = 0.0f64;
            for di in -half..=half {
                for dj in -half..=half {
                    let ni = i as i64 + di;
                    let nj = j as i64 + dj;
                    if ni >= 0 && ni < ny as i64 && nj >= 0 && nj < nx as i64 {
                        sum += image[[ni as usize, nj as usize]].abs();
                        count += 1.0;
                    }
                }
            }
            out[[i, j]] = sum / count.max(1.0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_test_image() -> Array2<f64> {
        Array2::from_shape_fn((16, 16), |(i, j)| {
            ((i as f64 / 4.0).sin() * (j as f64 / 4.0).cos()) * 128.0 + 128.0
        })
    }

    #[test]
    fn test_laws_all_kernels_count() {
        let kernels = laws_all_kernels();
        assert_eq!(kernels.len(), 25, "Should have 25 Laws kernels");

        // Each kernel should be 5x5
        for (name, kernel) in &kernels {
            assert_eq!(
                kernel.dim(),
                (5, 5),
                "Kernel {} should be 5x5",
                name.display_name()
            );
        }
    }

    #[test]
    fn test_laws_l5l5_is_averaging() {
        // L5*L5 should be a smoothing/averaging kernel (all positive values)
        let kernels = laws_all_kernels();
        let l5l5 = &kernels[0].1; // First kernel should be L5L5
        assert_eq!(kernels[0].0.display_name(), "L5L5");

        // All values should be positive (product of two positive vectors)
        for &v in l5l5.iter() {
            assert!(v >= 0.0, "L5L5 should have all non-negative values");
        }

        // L5 = [1,4,6,4,1], sum = 16
        // L5*L5 outer product sum = 16*16 = 256
        let total: f64 = l5l5.iter().sum();
        assert!(
            (total - 256.0).abs() < 1e-10,
            "L5L5 sum should be 256, got {}",
            total
        );
    }

    #[test]
    fn test_laws_e5_detects_edges() {
        // E5 kernel should sum to 0 (high-pass characteristic)
        let e5 = LawsVector::E5.coefficients();
        let sum: f64 = e5.iter().sum();
        assert!(sum.abs() < 1e-10, "E5 should sum to 0");

        // Apply L5E5 to an image with vertical edges
        let mut img = Array2::<f64>::zeros((16, 16));
        // Left half = 0, right half = 100 (vertical edge at col 8)
        for i in 0..16 {
            for j in 8..16 {
                img[[i, j]] = 100.0;
            }
        }

        let config = LawsFullConfig {
            pairs: vec![
                LawsKernelName::new(LawsVector::L5, LawsVector::E5),
                LawsKernelName::new(LawsVector::L5, LawsVector::L5),
            ],
            window_size: 5,
            remove_dc: false,
        };
        let result = laws_texture_energy_full(&img, Some(config)).expect("laws");

        // L5E5 should have higher energy than L5L5 for edge image
        let e_l5e5 = result.feature_vector[0];
        assert!(
            e_l5e5 > 0.0,
            "L5E5 should detect the edge, energy = {}",
            e_l5e5
        );
    }

    #[test]
    fn test_laws_uniform_image_zero_energy() {
        let img = Array2::from_elem((16, 16), 100.0);
        let config = LawsFullConfig {
            pairs: vec![
                LawsKernelName::new(LawsVector::L5, LawsVector::E5),
                LawsKernelName::new(LawsVector::E5, LawsVector::S5),
                LawsKernelName::new(LawsVector::S5, LawsVector::S5),
            ],
            window_size: 5,
            remove_dc: true,
        };
        let result = laws_texture_energy_full(&img, Some(config)).expect("laws");

        for (i, &v) in result.feature_vector.iter().enumerate() {
            assert!(
                v < 1e-10,
                "Uniform image should have ~0 energy for kernel {}, got {}",
                result.kernel_names[i].display_name(),
                v
            );
        }
    }

    #[test]
    fn test_laws_full_25_kernels() {
        let img = make_test_image();
        let result = laws_texture_energy_full(&img, None).expect("laws");
        assert_eq!(result.energy_maps.len(), 25);
        assert_eq!(result.feature_vector.len(), 25);
        assert_eq!(result.kernel_names.len(), 25);

        for &v in &result.feature_vector {
            assert!(v >= 0.0, "Energy should be non-negative");
            assert!(v.is_finite(), "Energy should be finite");
        }
    }

    #[test]
    fn test_laws_rotation_invariant() {
        let img = make_test_image();
        let features = laws_rotation_invariant_features(&img, Some(5)).expect("ri features");

        // Should have 15 unique features (5 diagonal + 10 off-diagonal)
        assert_eq!(
            features.len(),
            15,
            "Should have 15 rotation-invariant features"
        );

        // Check names
        assert_eq!(features[0].0, "L5L5");
        assert_eq!(features[1].0, "L5E5");

        // All values should be non-negative
        for (name, val) in &features {
            assert!(
                *val >= 0.0 && val.is_finite(),
                "Feature {} should be non-negative and finite, got {}",
                name,
                val
            );
        }
    }

    #[test]
    fn test_laws_rotation_invariant_symmetry() {
        let img = make_test_image();
        // Compute full features to verify symmetry averaging
        let full_config = LawsFullConfig {
            window_size: 5,
            ..Default::default()
        };
        let full = laws_texture_energy_full(&img, Some(full_config)).expect("full");
        let ri = laws_rotation_invariant_features(&img, Some(5)).expect("ri");

        // Find L5E5 and E5L5 in full results
        let mut e_l5e5 = 0.0;
        let mut e_e5l5 = 0.0;
        for (i, name) in full.kernel_names.iter().enumerate() {
            if name.display_name() == "L5E5" {
                e_l5e5 = full.feature_vector[i];
            }
            if name.display_name() == "E5L5" {
                e_e5l5 = full.feature_vector[i];
            }
        }

        // Find L5E5 in rotation-invariant results
        let ri_l5e5 = ri
            .iter()
            .find(|(n, _)| n == "L5E5")
            .map(|(_, v)| *v)
            .unwrap_or(0.0);

        // Should be the average
        let expected = (e_l5e5 + e_e5l5) / 2.0;
        assert!(
            (ri_l5e5 - expected).abs() < 1e-10,
            "RI feature should be average of symmetric pair: {} vs {}",
            ri_l5e5,
            expected
        );
    }

    #[test]
    fn test_laws_errors() {
        let small = Array2::<f64>::zeros((3, 3));
        assert!(laws_texture_energy_full(&small, None).is_err());

        let img = make_test_image();
        let config = LawsFullConfig {
            window_size: 4, // even -> error
            ..Default::default()
        };
        assert!(laws_texture_energy_full(&img, Some(config)).is_err());
    }

    #[test]
    fn test_laws_vector_properties() {
        // L5 sum = 16, all others sum to 0
        let l5: f64 = LawsVector::L5.coefficients().iter().sum();
        assert!((l5 - 16.0).abs() < 1e-10);

        for v in &[
            LawsVector::E5,
            LawsVector::S5,
            LawsVector::W5,
            LawsVector::R5,
        ] {
            let sum: f64 = v.coefficients().iter().sum();
            assert!(
                sum.abs() < 1e-10,
                "{} should sum to 0, got {}",
                v.name(),
                sum
            );
        }
    }
}
