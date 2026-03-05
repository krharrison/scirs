//! Texture Analysis Module
//!
//! Provides classical texture descriptors for image analysis:
//!
//! - **GLCM** (Gray-Level Co-occurrence Matrix) computation and features
//! - **LBP** (Local Binary Patterns) -- basic and rotation-invariant
//! - **Gabor filter bank** for multi-scale, multi-orientation texture features
//! - **Laws' texture energy measures**

use scirs2_core::ndarray::{Array1, Array2};
use std::f64::consts::PI;

use crate::error::{NdimageError, NdimageResult};

// ===========================================================================
// GLCM - Gray-Level Co-occurrence Matrix
// ===========================================================================

/// Direction / offset for GLCM computation
#[derive(Debug, Clone, Copy)]
pub struct GlcmOffset {
    /// Row offset (dy)
    pub dy: i32,
    /// Column offset (dx)
    pub dx: i32,
}

impl GlcmOffset {
    /// Horizontal offset (0 degrees)
    pub fn horizontal() -> Self {
        Self { dy: 0, dx: 1 }
    }
    /// Vertical offset (90 degrees)
    pub fn vertical() -> Self {
        Self { dy: 1, dx: 0 }
    }
    /// Diagonal 45 degrees
    pub fn diagonal_45() -> Self {
        Self { dy: -1, dx: 1 }
    }
    /// Diagonal 135 degrees
    pub fn diagonal_135() -> Self {
        Self { dy: 1, dx: 1 }
    }
}

/// Features extracted from a GLCM
#[derive(Debug, Clone)]
pub struct GlcmFeatures {
    /// Contrast: sum of squared differences weighted by co-occurrence probability
    pub contrast: f64,
    /// Dissimilarity: sum of absolute differences weighted by probability
    pub dissimilarity: f64,
    /// Homogeneity (Inverse Difference Moment): concentration near diagonal
    pub homogeneity: f64,
    /// Energy (Angular Second Moment): uniformity / orderliness
    pub energy: f64,
    /// Correlation: linear dependency of gray levels
    pub correlation: f64,
    /// ASM (Angular Second Moment) = energy^2
    pub asm: f64,
    /// Entropy: randomness of the GLCM
    pub entropy: f64,
}

/// Compute a GLCM from a quantized gray-level image.
///
/// `image` must contain non-negative integer-valued gray levels in [0, levels-1].
/// `levels` is the number of distinct gray levels.
/// `offset` specifies the spatial relationship.
///
/// The returned matrix is normalized (probabilities sum to 1).
/// Both (i,j) and (j,i) transitions are counted (symmetric GLCM).
pub fn compute_glcm(
    image: &Array2<f64>,
    levels: usize,
    offset: &GlcmOffset,
) -> NdimageResult<Array2<f64>> {
    if levels == 0 {
        return Err(NdimageError::InvalidInput(
            "Number of gray levels must be > 0".into(),
        ));
    }
    let (ny, nx) = image.dim();
    if ny == 0 || nx == 0 {
        return Err(NdimageError::InvalidInput("Image must be non-empty".into()));
    }

    let mut glcm = Array2::<f64>::zeros((levels, levels));
    let mut count = 0.0;

    for i in 0..ny {
        for j in 0..nx {
            let ni = i as i32 + offset.dy;
            let nj = j as i32 + offset.dx;

            if ni >= 0 && ni < ny as i32 && nj >= 0 && nj < nx as i32 {
                let g1 = image[[i, j]].round() as usize;
                let g2 = image[[ni as usize, nj as usize]].round() as usize;

                if g1 < levels && g2 < levels {
                    glcm[[g1, g2]] += 1.0;
                    glcm[[g2, g1]] += 1.0; // symmetric
                    count += 2.0;
                }
            }
        }
    }

    // Normalize
    if count > 0.0 {
        glcm.mapv_inplace(|v| v / count);
    }

    Ok(glcm)
}

/// Quantize an image to a fixed number of gray levels.
///
/// Maps the image range [min, max] to [0, levels-1].
pub fn quantize_image(image: &Array2<f64>, levels: usize) -> NdimageResult<Array2<f64>> {
    if levels == 0 {
        return Err(NdimageError::InvalidInput(
            "Number of levels must be > 0".into(),
        ));
    }

    let mut i_min = f64::INFINITY;
    let mut i_max = f64::NEG_INFINITY;
    for &v in image.iter() {
        if v < i_min {
            i_min = v;
        }
        if v > i_max {
            i_max = v;
        }
    }

    let range = i_max - i_min;
    if range < 1e-15 {
        return Ok(Array2::zeros(image.dim()));
    }

    let scale = (levels as f64 - 1.0) / range;
    Ok(image.mapv(|v| ((v - i_min) * scale).round().min((levels - 1) as f64)))
}

/// Extract all standard GLCM features from a pre-computed, normalized GLCM.
pub fn glcm_features(glcm: &Array2<f64>) -> NdimageResult<GlcmFeatures> {
    let (rows, cols) = glcm.dim();
    if rows != cols || rows == 0 {
        return Err(NdimageError::InvalidInput(
            "GLCM must be a non-empty square matrix".into(),
        ));
    }
    let n = rows;

    // Marginal means and standard deviations
    let mut mu_i = 0.0;
    let mut mu_j = 0.0;
    for i in 0..n {
        for j in 0..n {
            let p = glcm[[i, j]];
            mu_i += (i as f64) * p;
            mu_j += (j as f64) * p;
        }
    }

    let mut sigma_i_sq = 0.0;
    let mut sigma_j_sq = 0.0;
    for i in 0..n {
        for j in 0..n {
            let p = glcm[[i, j]];
            sigma_i_sq += (i as f64 - mu_i).powi(2) * p;
            sigma_j_sq += (j as f64 - mu_j).powi(2) * p;
        }
    }

    let sigma_i = sigma_i_sq.sqrt();
    let sigma_j = sigma_j_sq.sqrt();

    let mut contrast = 0.0;
    let mut dissimilarity = 0.0;
    let mut homogeneity = 0.0;
    let mut energy = 0.0;
    let mut correlation = 0.0;
    let mut entropy = 0.0;

    for i in 0..n {
        for j in 0..n {
            let p = glcm[[i, j]];
            let diff = (i as f64 - j as f64).abs();

            contrast += diff * diff * p;
            dissimilarity += diff * p;
            homogeneity += p / (1.0 + diff * diff);
            energy += p * p;

            if p > 1e-15 {
                entropy -= p * p.ln();
            }

            if sigma_i > 1e-15 && sigma_j > 1e-15 {
                correlation += (i as f64 - mu_i) * (j as f64 - mu_j) * p;
            }
        }
    }

    if sigma_i > 1e-15 && sigma_j > 1e-15 {
        correlation /= sigma_i * sigma_j;
    } else {
        correlation = 0.0;
    }

    Ok(GlcmFeatures {
        contrast,
        dissimilarity,
        homogeneity,
        energy: energy.sqrt(),
        correlation,
        asm: energy,
        entropy,
    })
}

/// Convenience: compute GLCM and extract features in one call.
pub fn glcm_features_from_image(
    image: &Array2<f64>,
    levels: usize,
    offset: &GlcmOffset,
) -> NdimageResult<GlcmFeatures> {
    let quantized = quantize_image(image, levels)?;
    let glcm = compute_glcm(&quantized, levels, offset)?;
    glcm_features(&glcm)
}

// ===========================================================================
// LBP - Local Binary Patterns
// ===========================================================================

/// Compute the basic LBP image.
///
/// For each pixel, the 8 neighbors in a 3x3 window are compared to the center.
/// Each comparison yields a bit (1 if neighbor >= center, else 0).
/// The bits are packed into an integer in [0, 255].
///
/// Border pixels are set to 0.
pub fn lbp_basic(image: &Array2<f64>) -> NdimageResult<Array2<u8>> {
    let (ny, nx) = image.dim();
    if ny < 3 || nx < 3 {
        return Err(NdimageError::InvalidInput(
            "Image must be at least 3x3 for LBP".into(),
        ));
    }

    let mut lbp = Array2::<u8>::zeros((ny, nx));

    // 8-neighbor offsets in clockwise order starting from top-left
    let offsets: [(i32, i32); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ];

    for i in 1..(ny - 1) {
        for j in 1..(nx - 1) {
            let center = image[[i, j]];
            let mut code: u8 = 0;
            for (bit, &(di, dj)) in offsets.iter().enumerate() {
                let ni = (i as i32 + di) as usize;
                let nj = (j as i32 + dj) as usize;
                if image[[ni, nj]] >= center {
                    code |= 1 << bit;
                }
            }
            lbp[[i, j]] = code;
        }
    }

    Ok(lbp)
}

/// Compute the rotation-invariant LBP.
///
/// Each LBP code is replaced by the minimum value obtainable by bit-rotating
/// the 8-bit pattern.  This yields at most 36 distinct patterns.
pub fn lbp_rotation_invariant(image: &Array2<f64>) -> NdimageResult<Array2<u8>> {
    let basic = lbp_basic(image)?;
    let (ny, nx) = basic.dim();
    let mut ri_lbp = Array2::<u8>::zeros((ny, nx));

    for i in 0..ny {
        for j in 0..nx {
            ri_lbp[[i, j]] = min_rotation_8bit(basic[[i, j]]);
        }
    }

    Ok(ri_lbp)
}

/// Compute the rotation-invariant uniform LBP.
///
/// Uniform patterns have at most 2 bit transitions (0->1 or 1->0) when
/// the bit string is viewed circularly.  These are mapped to distinct
/// labels 0..P; all non-uniform patterns get a single "misc" label (P+1).
pub fn lbp_uniform(image: &Array2<f64>) -> NdimageResult<Array2<u8>> {
    let basic = lbp_basic(image)?;
    let (ny, nx) = basic.dim();
    let mut uni_lbp = Array2::<u8>::zeros((ny, nx));

    // Build lookup table for uniform LBP
    let lut = build_uniform_lut();

    for i in 0..ny {
        for j in 0..nx {
            uni_lbp[[i, j]] = lut[basic[[i, j]] as usize];
        }
    }

    Ok(uni_lbp)
}

/// Build a lookup table mapping each 8-bit LBP code to its uniform-LBP label.
fn build_uniform_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    let mut label = 0u8;

    // First, find all uniform patterns and assign labels
    for code in 0..=255u16 {
        let transitions = count_transitions(code as u8);
        if transitions <= 2 {
            lut[code as usize] = label;
            // label for uniform: we want rotation-invariant grouping
            // So we group by number-of-1-bits for uniform patterns
            label = label.wrapping_add(1);
        }
    }

    // Non-uniform patterns get the same final label
    let non_uniform_label = label;
    for code in 0..=255u16 {
        let transitions = count_transitions(code as u8);
        if transitions > 2 {
            lut[code as usize] = non_uniform_label;
        }
    }

    lut
}

/// Count the number of 0->1 and 1->0 transitions in a circular 8-bit pattern.
fn count_transitions(code: u8) -> u32 {
    let mut t = 0u32;
    for bit in 0..8 {
        let b0 = (code >> bit) & 1;
        let b1 = (code >> ((bit + 1) % 8)) & 1;
        if b0 != b1 {
            t += 1;
        }
    }
    t
}

/// Minimum rotation of an 8-bit pattern (circular left-shifts).
fn min_rotation_8bit(code: u8) -> u8 {
    let mut min_val = code;
    let mut rotated = code;
    for _ in 1..8 {
        rotated = (rotated << 1) | (rotated >> 7); // 8-bit rotate left
        if rotated < min_val {
            min_val = rotated;
        }
    }
    min_val
}

/// Compute the histogram of an LBP image.
///
/// `n_bins` is the number of histogram bins (256 for basic LBP, or the
/// number of distinct labels for uniform LBP).
/// The histogram is L1-normalized (sums to 1).
pub fn lbp_histogram(lbp_image: &Array2<u8>, n_bins: usize) -> NdimageResult<Array1<f64>> {
    if n_bins == 0 {
        return Err(NdimageError::InvalidInput(
            "Number of bins must be > 0".into(),
        ));
    }

    let mut hist = Array1::<f64>::zeros(n_bins);
    let mut count = 0.0;

    for &v in lbp_image.iter() {
        let bin = (v as usize).min(n_bins - 1);
        hist[bin] += 1.0;
        count += 1.0;
    }

    if count > 0.0 {
        hist.mapv_inplace(|v| v / count);
    }

    Ok(hist)
}

// ===========================================================================
// Gabor filter bank
// ===========================================================================

/// Configuration for a Gabor filter bank
#[derive(Debug, Clone)]
pub struct GaborBankConfig {
    /// Orientations in radians.  Defaults to [0, pi/4, pi/2, 3pi/4].
    pub orientations: Vec<f64>,
    /// Wavelengths (lambda) in pixels.  Defaults to [4, 8, 16].
    pub wavelengths: Vec<f64>,
    /// Bandwidth (sigma / lambda ratio).  Default 0.56.
    pub bandwidth: f64,
    /// Spatial aspect ratio (gamma).  Default 0.5.
    pub gamma: f64,
}

impl Default for GaborBankConfig {
    fn default() -> Self {
        Self {
            orientations: vec![0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0],
            wavelengths: vec![4.0, 8.0, 16.0],
            bandwidth: 0.56,
            gamma: 0.5,
        }
    }
}

/// Result from applying a Gabor filter bank
#[derive(Debug, Clone)]
pub struct GaborBankResult {
    /// Response magnitudes for each (orientation, wavelength) pair
    /// Outer index: orientation, inner index: wavelength
    pub responses: Vec<Vec<Array2<f64>>>,
    /// Feature vector: mean energy for each (orientation, wavelength)
    pub feature_vector: Vec<f64>,
}

/// Apply a bank of Gabor filters and compute the feature vector.
///
/// For each (orientation, wavelength) pair, a Gabor kernel is constructed
/// and convolved with the image.  The response magnitude is computed, and
/// its mean serves as one element of the feature vector.
pub fn gabor_filter_bank(
    image: &Array2<f64>,
    config: Option<GaborBankConfig>,
) -> NdimageResult<GaborBankResult> {
    let cfg = config.unwrap_or_default();
    let (ny, nx) = image.dim();
    if ny < 3 || nx < 3 {
        return Err(NdimageError::InvalidInput(
            "Image must be at least 3x3 for Gabor filtering".into(),
        ));
    }

    let mut responses = Vec::with_capacity(cfg.orientations.len());
    let mut feature_vector = Vec::new();

    for &theta in &cfg.orientations {
        let mut orient_responses = Vec::with_capacity(cfg.wavelengths.len());
        for &lambda in &cfg.wavelengths {
            let sigma = cfg.bandwidth * lambda;
            let kernel = make_gabor_kernel(sigma, theta, lambda, cfg.gamma);
            let response = convolve_2d(image, &kernel);
            let magnitude = response.mapv(|v| v.abs());

            let mean_energy = magnitude.sum() / magnitude.len() as f64;
            feature_vector.push(mean_energy);

            orient_responses.push(magnitude);
        }
        responses.push(orient_responses);
    }

    Ok(GaborBankResult {
        responses,
        feature_vector,
    })
}

/// Create a real-part Gabor kernel.
fn make_gabor_kernel(sigma: f64, theta: f64, lambda: f64, gamma: f64) -> Array2<f64> {
    let half = (3.0 * sigma).ceil() as i32;
    let size = (2 * half + 1) as usize;
    let mut kernel = Array2::<f64>::zeros((size, size));

    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let sigma_sq = sigma * sigma;
    let gamma_sq = gamma * gamma;
    let freq = 2.0 * PI / lambda;

    for i in 0..size {
        for j in 0..size {
            let y = (i as i32 - half) as f64;
            let x = (j as i32 - half) as f64;
            let x_rot = x * cos_t + y * sin_t;
            let y_rot = -x * sin_t + y * cos_t;
            let gauss = (-(x_rot * x_rot + gamma_sq * y_rot * y_rot) / (2.0 * sigma_sq)).exp();
            kernel[[i, j]] = gauss * (freq * x_rot).cos();
        }
    }

    kernel
}

/// Simple spatial convolution for 2D f64 arrays (zero-padded boundary).
fn convolve_2d(image: &Array2<f64>, kernel: &Array2<f64>) -> Array2<f64> {
    let (ny, nx) = image.dim();
    let (ky, kx) = kernel.dim();
    let half_ky = (ky / 2) as i32;
    let half_kx = (kx / 2) as i32;
    let mut out = Array2::<f64>::zeros((ny, nx));

    for i in 0..ny {
        for j in 0..nx {
            let mut sum = 0.0;
            for ki in 0..ky {
                for kj in 0..kx {
                    let ii = i as i32 + ki as i32 - half_ky;
                    let jj = j as i32 + kj as i32 - half_kx;
                    if ii >= 0 && ii < ny as i32 && jj >= 0 && jj < nx as i32 {
                        sum += image[[ii as usize, jj as usize]] * kernel[[ki, kj]];
                    }
                }
            }
            out[[i, j]] = sum;
        }
    }
    out
}

// ===========================================================================
// Laws' Texture Energy Measures
// ===========================================================================

/// Laws' 1-D vectors (length 5).
/// L5 = Level, E5 = Edge, S5 = Spot, W5 = Wave, R5 = Ripple
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LawsVector {
    L5,
    E5,
    S5,
    W5,
    R5,
}

impl LawsVector {
    /// Return the 5-element filter vector
    fn coefficients(&self) -> [f64; 5] {
        match self {
            LawsVector::L5 => [1.0, 4.0, 6.0, 4.0, 1.0],
            LawsVector::E5 => [-1.0, -2.0, 0.0, 2.0, 1.0],
            LawsVector::S5 => [-1.0, 0.0, 2.0, 0.0, -1.0],
            LawsVector::W5 => [-1.0, 2.0, 0.0, -2.0, 1.0],
            LawsVector::R5 => [1.0, -4.0, 6.0, -4.0, 1.0],
        }
    }
}

/// Configuration for Laws' texture energy
#[derive(Debug, Clone)]
pub struct LawsConfig {
    /// Filter pair: (row vector, column vector).
    /// Common pairs: (L5,E5), (L5,S5), (E5,S5), etc.
    pub pairs: Vec<(LawsVector, LawsVector)>,
    /// Window size for energy computation (should be odd).
    pub window_size: usize,
}

impl Default for LawsConfig {
    fn default() -> Self {
        Self {
            pairs: vec![
                (LawsVector::L5, LawsVector::E5),
                (LawsVector::L5, LawsVector::S5),
                (LawsVector::E5, LawsVector::S5),
                (LawsVector::E5, LawsVector::E5),
                (LawsVector::S5, LawsVector::S5),
            ],
            window_size: 15,
        }
    }
}

/// Result of Laws' texture energy computation
#[derive(Debug, Clone)]
pub struct LawsResult {
    /// Texture energy maps, one per filter pair
    pub energy_maps: Vec<Array2<f64>>,
    /// Feature vector: mean energy for each filter pair
    pub feature_vector: Vec<f64>,
}

/// Compute Laws' texture energy measures.
///
/// For each specified pair (V_row, V_col):
///   1. Build the 5x5 kernel as the outer product of the two vectors
///   2. Convolve the image with the kernel
///   3. Compute the local energy (sum of absolute values in a window)
pub fn laws_texture_energy(
    image: &Array2<f64>,
    config: Option<LawsConfig>,
) -> NdimageResult<LawsResult> {
    let cfg = config.unwrap_or_default();
    let (ny, nx) = image.dim();
    if ny < 5 || nx < 5 {
        return Err(NdimageError::InvalidInput(
            "Image must be at least 5x5 for Laws' filters".into(),
        ));
    }
    if cfg.window_size == 0 || cfg.window_size % 2 == 0 {
        return Err(NdimageError::InvalidInput(
            "Laws' window_size must be an odd positive integer".into(),
        ));
    }

    // Remove DC component (subtract local mean) -- optional but standard
    let dc_removed = remove_dc(image, cfg.window_size);

    let mut energy_maps = Vec::with_capacity(cfg.pairs.len());
    let mut feature_vector = Vec::with_capacity(cfg.pairs.len());

    for &(ref row_vec, ref col_vec) in &cfg.pairs {
        // Build 5x5 kernel
        let r = row_vec.coefficients();
        let c = col_vec.coefficients();
        let mut kernel = Array2::<f64>::zeros((5, 5));
        for i in 0..5 {
            for j in 0..5 {
                kernel[[i, j]] = r[i] * c[j];
            }
        }

        // Convolve
        let response = convolve_2d(&dc_removed, &kernel);

        // Local energy: absolute value averaged over window
        let energy = local_energy(&response, cfg.window_size);

        let mean_e = energy.sum() / energy.len().max(1) as f64;
        feature_vector.push(mean_e);
        energy_maps.push(energy);
    }

    Ok(LawsResult {
        energy_maps,
        feature_vector,
    })
}

/// Remove DC by subtracting a local mean (box filter).
fn remove_dc(image: &Array2<f64>, window: usize) -> Array2<f64> {
    let (ny, nx) = image.dim();
    let half = (window / 2) as i32;
    let mut out = Array2::<f64>::zeros((ny, nx));

    for i in 0..ny {
        for j in 0..nx {
            let mut sum: f64 = 0.0;
            let mut count: f64 = 0.0;
            for di in -half..=half {
                for dj in -half..=half {
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;
                    if ni >= 0 && ni < ny as i32 && nj >= 0 && nj < nx as i32 {
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

/// Compute local energy: sum of |values| in a window.
fn local_energy(image: &Array2<f64>, window: usize) -> Array2<f64> {
    let (ny, nx) = image.dim();
    let half = (window / 2) as i32;
    let mut out = Array2::<f64>::zeros((ny, nx));

    for i in 0..ny {
        for j in 0..nx {
            let mut sum: f64 = 0.0;
            let mut count: f64 = 0.0;
            for di in -half..=half {
                for dj in -half..=half {
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;
                    if ni >= 0 && ni < ny as i32 && nj >= 0 && nj < nx as i32 {
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

// ===========================================================================
// Tests
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_test_image() -> Array2<f64> {
        Array2::from_shape_fn((16, 16), |(i, j)| {
            ((i as f64 / 4.0).sin() * (j as f64 / 4.0).cos()) * 128.0 + 128.0
        })
    }

    // -- GLCM tests --

    #[test]
    fn test_quantize_image() {
        let img = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f64);
        let q = quantize_image(&img, 4).expect("quantize failed");
        // Min should be 0, max should be 3
        assert!(q.iter().all(|&v| v >= 0.0 && v <= 3.0));
    }

    #[test]
    fn test_compute_glcm_uniform() {
        // Uniform image -> all weight on diagonal
        let img = Array2::<f64>::ones((8, 8)) * 2.0;
        let q = quantize_image(&img, 4).expect("quantize");
        let glcm = compute_glcm(&q, 4, &GlcmOffset::horizontal()).expect("glcm");
        // All mass should be on one diagonal element
        let total: f64 = glcm.iter().sum();
        assert!((total - 1.0).abs() < 1e-10, "GLCM should be normalized");
    }

    #[test]
    fn test_glcm_features_uniform() {
        // Uniform GLCM (all same value) -> energy, contrast properties
        let mut glcm = Array2::<f64>::zeros((4, 4));
        glcm[[2, 2]] = 1.0; // All mass at one cell
        let feats = glcm_features(&glcm).expect("features");
        assert!(
            (feats.asm - 1.0).abs() < 1e-10,
            "ASM should be 1 for single-cell"
        );
        assert!(
            feats.contrast < 1e-10,
            "Contrast should be 0 for single-cell"
        );
    }

    #[test]
    fn test_glcm_features_from_image() {
        let img = make_test_image();
        let feats = glcm_features_from_image(&img, 8, &GlcmOffset::horizontal()).expect("features");
        assert!(feats.contrast >= 0.0);
        assert!(feats.energy >= 0.0);
        assert!(feats.homogeneity >= 0.0);
        assert!(feats.entropy >= 0.0);
    }

    #[test]
    fn test_glcm_correlation_identical_neighbors() {
        // If all neighbors are the same level, correlation should be high
        let img = Array2::from_shape_fn((8, 8), |(i, _j)| i as f64);
        // Horizontal neighbors in the same row have the same value
        let q = quantize_image(&img, 8).expect("quantize");
        let glcm = compute_glcm(&q, 8, &GlcmOffset::horizontal()).expect("glcm");
        let feats = glcm_features(&glcm).expect("features");
        // High correlation expected
        assert!(
            feats.correlation > 0.5,
            "Correlation should be high, got {}",
            feats.correlation
        );
    }

    // -- LBP tests --

    #[test]
    fn test_lbp_basic_shape() {
        let img = make_test_image();
        let lbp = lbp_basic(&img).expect("lbp_basic");
        assert_eq!(lbp.dim(), img.dim());
    }

    #[test]
    fn test_lbp_basic_uniform_image() {
        // All same value -> center == neighbors -> all bits set -> code 255
        let img = Array2::<f64>::ones((8, 8)) * 50.0;
        let lbp = lbp_basic(&img).expect("lbp_basic");
        // Interior pixels should be 255
        assert_eq!(lbp[[3, 3]], 255);
    }

    #[test]
    fn test_lbp_rotation_invariant() {
        let img = make_test_image();
        let ri = lbp_rotation_invariant(&img).expect("lbp_ri");
        assert_eq!(ri.dim(), img.dim());
        // All codes should be <= basic codes (minimum rotation)
        let basic = lbp_basic(&img).expect("basic");
        for (r, b) in ri.iter().zip(basic.iter()) {
            assert!(*r <= *b);
        }
    }

    #[test]
    fn test_lbp_uniform() {
        let img = make_test_image();
        let uni = lbp_uniform(&img).expect("lbp_uniform");
        assert_eq!(uni.dim(), img.dim());
    }

    #[test]
    fn test_lbp_histogram() {
        let img = make_test_image();
        let lbp = lbp_basic(&img).expect("lbp");
        let hist = lbp_histogram(&lbp, 256).expect("histogram");
        assert_eq!(hist.len(), 256);
        let total: f64 = hist.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Histogram should be normalized"
        );
    }

    #[test]
    fn test_lbp_too_small() {
        let img = Array2::<f64>::zeros((2, 2));
        assert!(lbp_basic(&img).is_err());
    }

    // -- Gabor tests --

    #[test]
    fn test_gabor_filter_bank() {
        let img = make_test_image();
        let result = gabor_filter_bank(&img, None).expect("gabor bank");
        // 4 orientations * 3 wavelengths = 12 features
        assert_eq!(result.feature_vector.len(), 12);
        assert_eq!(result.responses.len(), 4);
        assert_eq!(result.responses[0].len(), 3);
        for &v in &result.feature_vector {
            assert!(v >= 0.0, "Gabor energy should be non-negative");
        }
    }

    #[test]
    fn test_gabor_filter_bank_custom() {
        let img = make_test_image();
        let cfg = GaborBankConfig {
            orientations: vec![0.0, PI / 2.0],
            wavelengths: vec![6.0],
            bandwidth: 0.5,
            gamma: 0.5,
        };
        let result = gabor_filter_bank(&img, Some(cfg)).expect("gabor custom");
        assert_eq!(result.feature_vector.len(), 2);
    }

    // -- Laws' tests --

    #[test]
    fn test_laws_texture_energy() {
        let img = make_test_image();
        let result = laws_texture_energy(&img, None).expect("laws");
        // Default has 5 pairs
        assert_eq!(result.energy_maps.len(), 5);
        assert_eq!(result.feature_vector.len(), 5);
        for &v in &result.feature_vector {
            assert!(v >= 0.0, "Laws energy should be non-negative");
        }
    }

    #[test]
    fn test_laws_uniform_image() {
        // Uniform image should have low texture energy
        let img = Array2::<f64>::ones((16, 16)) * 100.0;
        let result = laws_texture_energy(&img, None).expect("laws");
        for &v in &result.feature_vector {
            assert!(
                v < 1e-10,
                "Uniform image should have ~0 Laws energy, got {}",
                v
            );
        }
    }

    #[test]
    fn test_laws_vectors() {
        // L5 coefficients should sum to 16
        let l5 = LawsVector::L5.coefficients();
        let sum: f64 = l5.iter().sum();
        assert!((sum - 16.0).abs() < 1e-10);

        // E5 should sum to 0
        let e5 = LawsVector::E5.coefficients();
        let e5_sum: f64 = e5.iter().sum();
        assert!(e5_sum.abs() < 1e-10);
    }

    #[test]
    fn test_laws_invalid_window() {
        let img = make_test_image();
        let cfg = LawsConfig {
            pairs: vec![(LawsVector::L5, LawsVector::E5)],
            window_size: 4, // even -> error
        };
        assert!(laws_texture_energy(&img, Some(cfg)).is_err());
    }

    #[test]
    fn test_count_transitions() {
        // 0b00000000 -> 0 transitions
        assert_eq!(count_transitions(0), 0);
        // 0b11111111 -> 0 transitions
        assert_eq!(count_transitions(255), 0);
        // 0b00000001 -> 2 transitions (0->1 at bit0, 1->0 at bit1)
        assert_eq!(count_transitions(1), 2);
        // 0b00001111 -> 2 transitions
        assert_eq!(count_transitions(0x0F), 2);
    }

    #[test]
    fn test_min_rotation() {
        // 0b00000001 and 0b10000000 should map to same minimum
        assert_eq!(min_rotation_8bit(1), min_rotation_8bit(128));
        // 0b00000000 -> 0
        assert_eq!(min_rotation_8bit(0), 0);
        // 0b11111111 -> 255
        assert_eq!(min_rotation_8bit(255), 255);
    }

    #[test]
    fn test_glcm_empty_image() {
        let img = Array2::<f64>::zeros((0, 0));
        assert!(compute_glcm(&img, 4, &GlcmOffset::horizontal()).is_err());
    }

    #[test]
    fn test_gabor_too_small() {
        let img = Array2::<f64>::zeros((2, 2));
        assert!(gabor_filter_bank(&img, None).is_err());
    }

    #[test]
    fn test_laws_too_small() {
        let img = Array2::<f64>::zeros((3, 3));
        assert!(laws_texture_energy(&img, None).is_err());
    }
}
