//! Texture Analysis Module
//!
//! Provides advanced texture descriptors for image analysis:
//!
//! - **RLM** (Run-Length Matrix) features: SRE, LRE, GLN, RLN, RP, LGRE, HGRE
//! - **GLSZM** (Gray-Level Size Zone Matrix) computation and features
//! - **NGTDM** (Neighborhood Gray-Tone Difference Matrix) features
//! - **Laws** texture energy measures (full 25-kernel set with rotation invariance)
//!
//! ## Common Types
//!
//! [`QuantizationConfig`] controls how continuous-valued images are quantized
//! to discrete gray levels before texture matrix computation.
//!
//! [`TextureFeatures`] is a unified container for features from any texture method.

pub mod glszm;
pub mod laws;
pub mod legacy;
pub mod ngtdm;
pub mod rlm;

// Re-exports
pub use glszm::{compute_glszm, glszm_features, GlszmConnectivity, GlszmFeatures, GlszmResult};
pub use laws::{
    laws_all_kernels, laws_rotation_invariant_features, laws_texture_energy_full, LawsFullConfig,
    LawsFullResult, LawsKernelName, LawsVector as LawsVectorFull,
};
pub use ngtdm::{compute_ngtdm, ngtdm_features, NgtdmFeatures, NgtdmResult};
pub use rlm::{compute_rlm, rlm_features, RlmDirection, RlmFeatures, RlmResult};

// Re-export legacy texture types (GLCM, LBP, Gabor, original Laws)
pub use legacy::{
    compute_glcm, gabor_filter_bank as legacy_gabor_filter_bank, glcm_features,
    glcm_features_from_image, laws_texture_energy as legacy_laws_texture_energy, lbp_basic,
    lbp_histogram, lbp_rotation_invariant, lbp_uniform, quantize_image, GaborBankConfig,
    GaborBankResult, GlcmFeatures, GlcmOffset, LawsConfig, LawsResult,
    LawsVector as LawsVectorLegacy,
};

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::Array2;

/// Configuration for quantizing a continuous image to discrete gray levels.
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Number of discrete gray levels (must be >= 2).
    pub n_levels: usize,
    /// Optional fixed minimum value for scaling. If `None`, uses image minimum.
    pub min_val: Option<f64>,
    /// Optional fixed maximum value for scaling. If `None`, uses image maximum.
    pub max_val: Option<f64>,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            n_levels: 8,
            min_val: None,
            max_val: None,
        }
    }
}

/// Quantize a floating-point image to discrete gray levels in `[0, n_levels-1]`.
///
/// Returns a `u8` array. Pixel values are linearly mapped from
/// `[min_val, max_val]` to `[0, n_levels - 1]`.
///
/// # Errors
/// Returns `NdimageError::InvalidInput` if `n_levels < 2` or image is empty.
pub fn quantize_image_u8(
    image: &Array2<f64>,
    config: &QuantizationConfig,
) -> NdimageResult<Array2<u8>> {
    if config.n_levels < 2 {
        return Err(NdimageError::InvalidInput(
            "n_levels must be at least 2".into(),
        ));
    }
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    if config.n_levels > 256 {
        return Err(NdimageError::InvalidInput(
            "n_levels must be <= 256 for u8 quantization".into(),
        ));
    }

    let i_min = config
        .min_val
        .unwrap_or_else(|| image.iter().copied().fold(f64::INFINITY, f64::min));
    let i_max = config
        .max_val
        .unwrap_or_else(|| image.iter().copied().fold(f64::NEG_INFINITY, f64::max));

    let range = i_max - i_min;
    if range < 1e-15 {
        return Ok(Array2::zeros(image.dim()));
    }

    let scale = (config.n_levels as f64 - 1.0) / range;
    let max_level = (config.n_levels - 1) as f64;

    let result = image.mapv(|v| {
        let scaled = ((v - i_min) * scale).round().clamp(0.0, max_level);
        scaled as u8
    });

    Ok(result)
}

/// Unified container for texture features from any method.
///
/// Each field is `Option` so that only the computed features are populated.
#[derive(Debug, Clone, Default)]
pub struct TextureFeatures {
    /// RLM features (if computed)
    pub rlm: Option<RlmFeatures>,
    /// GLSZM features (if computed)
    pub glszm: Option<GlszmFeatures>,
    /// NGTDM features (if computed)
    pub ngtdm: Option<NgtdmFeatures>,
    /// Laws texture energy feature vector (if computed)
    pub laws_features: Option<Vec<f64>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_quantize_image_u8_basic() {
        let img = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f64);
        let config = QuantizationConfig {
            n_levels: 4,
            ..Default::default()
        };
        let q = quantize_image_u8(&img, &config).expect("quantize failed");
        assert!(q.iter().all(|&v| v < 4));
        // Min pixel should map to 0
        assert_eq!(q[[0, 0]], 0);
        // Max pixel should map to 3
        assert_eq!(q[[3, 3]], 3);
    }

    #[test]
    fn test_quantize_image_u8_uniform() {
        let img = Array2::from_elem((4, 4), 42.0);
        let config = QuantizationConfig::default();
        let q = quantize_image_u8(&img, &config).expect("quantize failed");
        // Uniform image -> all zeros
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_quantize_image_u8_custom_range() {
        let img = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f64);
        let config = QuantizationConfig {
            n_levels: 16,
            min_val: Some(0.0),
            max_val: Some(15.0),
        };
        let q = quantize_image_u8(&img, &config).expect("quantize failed");
        // Should map 0..15 to 0..15
        assert_eq!(q[[0, 0]], 0);
        assert_eq!(q[[3, 3]], 15);
    }

    #[test]
    fn test_quantize_image_u8_errors() {
        let img = Array2::from_elem((4, 4), 0.0);
        let config = QuantizationConfig {
            n_levels: 1,
            ..Default::default()
        };
        assert!(quantize_image_u8(&img, &config).is_err());

        let empty = Array2::<f64>::zeros((0, 0));
        let config2 = QuantizationConfig::default();
        assert!(quantize_image_u8(&empty, &config2).is_err());
    }

    #[test]
    fn test_texture_features_default() {
        let tf = TextureFeatures::default();
        assert!(tf.rlm.is_none());
        assert!(tf.glszm.is_none());
        assert!(tf.ngtdm.is_none());
        assert!(tf.laws_features.is_none());
    }
}
