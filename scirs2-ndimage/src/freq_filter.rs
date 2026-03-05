//! Frequency Domain Filtering Module
//!
//! Provides classical frequency-domain filters applied via FFT:
//!
//! - **Ideal** low-pass, high-pass, band-pass filters
//! - **Butterworth** low-pass and high-pass filters
//! - **Gaussian** low-pass and high-pass filters
//! - **Homomorphic** filtering (illumination / reflectance separation)
//! - **Notch** filter (periodic noise removal)
//! - **Wiener** deconvolution (noise-aware inverse filtering)

use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::Complex64;
use scirs2_fft::{fft2, fftfreq, ifft2};
use std::f64::consts::PI;

use crate::error::{NdimageError, NdimageResult};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Type of frequency filter
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
}

/// Configuration for a Butterworth filter
#[derive(Debug, Clone)]
pub struct ButterworthConfig {
    /// Cutoff frequency (normalized, 0..0.5)
    pub cutoff: f64,
    /// Filter order (higher = sharper transition)
    pub order: u32,
}

/// Configuration for a band-pass filter
#[derive(Debug, Clone)]
pub struct BandPassConfig {
    /// Lower cutoff frequency (normalized, 0..0.5)
    pub low_cutoff: f64,
    /// Upper cutoff frequency (normalized, 0..0.5)
    pub high_cutoff: f64,
}

/// Configuration for a notch (band-reject) filter
#[derive(Debug, Clone)]
pub struct NotchConfig {
    /// Center frequency along rows (normalized)
    pub freq_y: f64,
    /// Center frequency along columns (normalized)
    pub freq_x: f64,
    /// Radius of the notch (in normalized frequency units)
    pub radius: f64,
}

/// Configuration for homomorphic filtering
#[derive(Debug, Clone)]
pub struct HomomorphicConfig {
    /// Low-frequency gain (gamma_L), typically < 1
    pub gamma_low: f64,
    /// High-frequency gain (gamma_H), typically > 1
    pub gamma_high: f64,
    /// Cutoff frequency (normalized)
    pub cutoff: f64,
    /// Sharpness parameter (like Butterworth order)
    pub sharpness: f64,
}

impl Default for HomomorphicConfig {
    fn default() -> Self {
        Self {
            gamma_low: 0.3,
            gamma_high: 2.0,
            cutoff: 0.1,
            sharpness: 2.0,
        }
    }
}

/// Configuration for Wiener deconvolution
#[derive(Debug, Clone)]
pub struct WienerConfig {
    /// Estimated noise-to-signal power ratio (K).
    /// Larger values produce more regularization (less noise amplification).
    pub noise_power_ratio: f64,
}

impl Default for WienerConfig {
    fn default() -> Self {
        Self {
            noise_power_ratio: 0.01,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: build frequency-distance grid
// ---------------------------------------------------------------------------

/// For an (ny, nx) image, compute the normalized distance from DC for every
/// frequency bin.  The distances are in [0, 0.5*sqrt(2)] roughly.
fn freq_distance_grid(ny: usize, nx: usize) -> NdimageResult<Array2<f64>> {
    let freqs_y =
        fftfreq(ny, 1.0).map_err(|e| NdimageError::ComputationError(format!("fftfreq: {}", e)))?;
    let freqs_x =
        fftfreq(nx, 1.0).map_err(|e| NdimageError::ComputationError(format!("fftfreq: {}", e)))?;

    let mut dist = Array2::<f64>::zeros((ny, nx));
    for (i, &fy) in freqs_y.iter().enumerate() {
        for (j, &fx) in freqs_x.iter().enumerate() {
            dist[[i, j]] = (fy * fy + fx * fx).sqrt();
        }
    }
    Ok(dist)
}

/// Convert image to frequency domain, apply a real-valued mask, convert back.
fn apply_freq_mask(image: &Array2<f64>, mask: &Array2<f64>) -> NdimageResult<Array2<f64>> {
    let (ny, nx) = image.dim();

    let spectrum = fft2(image, None, None, None)
        .map_err(|e| NdimageError::ComputationError(format!("FFT: {}", e)))?;

    let mut filtered = Array2::<Complex64>::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            filtered[[i, j]] = spectrum[[i, j]] * mask[[i, j]];
        }
    }

    let result = ifft2(&filtered, None, None, None)
        .map_err(|e| NdimageError::ComputationError(format!("IFFT: {}", e)))?;

    let mut out = Array2::<f64>::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            out[[i, j]] = result[[i, j]].re;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Ideal filters
// ---------------------------------------------------------------------------

/// Ideal low-pass filter.
///
/// Passes all frequencies within `cutoff` distance from DC and blocks
/// everything else.  The hard cutoff produces ringing (Gibbs phenomenon).
///
/// `cutoff` is a normalized frequency in (0, 0.5].
pub fn ideal_lowpass(image: &Array2<f64>, cutoff: f64) -> NdimageResult<Array2<f64>> {
    validate_cutoff(cutoff)?;
    let (ny, nx) = image.dim();
    let dist = freq_distance_grid(ny, nx)?;
    let mask = dist.mapv(|d| if d <= cutoff { 1.0 } else { 0.0 });
    apply_freq_mask(image, &mask)
}

/// Ideal high-pass filter.
///
/// Blocks all frequencies within `cutoff` distance from DC.
pub fn ideal_highpass(image: &Array2<f64>, cutoff: f64) -> NdimageResult<Array2<f64>> {
    validate_cutoff(cutoff)?;
    let (ny, nx) = image.dim();
    let dist = freq_distance_grid(ny, nx)?;
    let mask = dist.mapv(|d| if d > cutoff { 1.0 } else { 0.0 });
    apply_freq_mask(image, &mask)
}

/// Ideal band-pass filter.
///
/// Passes frequencies in [low_cutoff, high_cutoff].
pub fn ideal_bandpass(image: &Array2<f64>, config: &BandPassConfig) -> NdimageResult<Array2<f64>> {
    if config.low_cutoff >= config.high_cutoff {
        return Err(NdimageError::InvalidInput(
            "low_cutoff must be less than high_cutoff".into(),
        ));
    }
    validate_cutoff(config.low_cutoff)?;
    validate_cutoff(config.high_cutoff)?;

    let (ny, nx) = image.dim();
    let dist = freq_distance_grid(ny, nx)?;
    let mask = dist.mapv(|d| {
        if d >= config.low_cutoff && d <= config.high_cutoff {
            1.0
        } else {
            0.0
        }
    });
    apply_freq_mask(image, &mask)
}

// ---------------------------------------------------------------------------
// Butterworth filters
// ---------------------------------------------------------------------------

/// Butterworth low-pass filter.
///
/// Transfer function: H(d) = 1 / (1 + (d / cutoff)^(2*order))
///
/// Smoother roll-off than ideal filters; no ringing for low orders.
pub fn butterworth_lowpass(
    image: &Array2<f64>,
    config: &ButterworthConfig,
) -> NdimageResult<Array2<f64>> {
    validate_cutoff(config.cutoff)?;
    if config.order == 0 {
        return Err(NdimageError::InvalidInput(
            "Butterworth order must be >= 1".into(),
        ));
    }
    let (ny, nx) = image.dim();
    let dist = freq_distance_grid(ny, nx)?;
    let order2 = 2 * config.order;
    let cutoff = config.cutoff;
    let mask = dist.mapv(|d| {
        if cutoff < 1e-15 {
            0.0
        } else {
            1.0 / (1.0 + (d / cutoff).powi(order2 as i32))
        }
    });
    apply_freq_mask(image, &mask)
}

/// Butterworth high-pass filter.
///
/// Transfer function: H(d) = 1 / (1 + (cutoff / d)^(2*order))
pub fn butterworth_highpass(
    image: &Array2<f64>,
    config: &ButterworthConfig,
) -> NdimageResult<Array2<f64>> {
    validate_cutoff(config.cutoff)?;
    if config.order == 0 {
        return Err(NdimageError::InvalidInput(
            "Butterworth order must be >= 1".into(),
        ));
    }
    let (ny, nx) = image.dim();
    let dist = freq_distance_grid(ny, nx)?;
    let order2 = 2 * config.order;
    let cutoff = config.cutoff;
    let mask = dist.mapv(|d| {
        if d < 1e-15 {
            0.0
        } else {
            1.0 / (1.0 + (cutoff / d).powi(order2 as i32))
        }
    });
    apply_freq_mask(image, &mask)
}

// ---------------------------------------------------------------------------
// Gaussian frequency filters
// ---------------------------------------------------------------------------

/// Gaussian low-pass filter.
///
/// H(d) = exp(-d^2 / (2 * cutoff^2))
///
/// No ringing; the smoothest possible transition.
pub fn gaussian_lowpass(image: &Array2<f64>, cutoff: f64) -> NdimageResult<Array2<f64>> {
    validate_cutoff(cutoff)?;
    let (ny, nx) = image.dim();
    let dist = freq_distance_grid(ny, nx)?;
    let two_c2 = 2.0 * cutoff * cutoff;
    let mask = dist.mapv(|d| (-d * d / two_c2).exp());
    apply_freq_mask(image, &mask)
}

/// Gaussian high-pass filter.
///
/// H(d) = 1 - exp(-d^2 / (2 * cutoff^2))
pub fn gaussian_highpass(image: &Array2<f64>, cutoff: f64) -> NdimageResult<Array2<f64>> {
    validate_cutoff(cutoff)?;
    let (ny, nx) = image.dim();
    let dist = freq_distance_grid(ny, nx)?;
    let two_c2 = 2.0 * cutoff * cutoff;
    let mask = dist.mapv(|d| 1.0 - (-d * d / two_c2).exp());
    apply_freq_mask(image, &mask)
}

// ---------------------------------------------------------------------------
// Homomorphic filtering
// ---------------------------------------------------------------------------

/// Homomorphic filtering for illumination / reflectance separation.
///
/// The image model is:  I(x,y) = L(x,y) * R(x,y)
/// where L is slowly-varying illumination and R is the reflectance.
///
/// Steps:
///   1. Take log of the image
///   2. Apply a filter that attenuates low frequencies (illumination) and
///      amplifies high frequencies (reflectance)
///   3. Exponentiate the result
///
/// This effectively compresses the dynamic range and enhances contrast.
pub fn homomorphic_filter(
    image: &Array2<f64>,
    config: Option<HomomorphicConfig>,
) -> NdimageResult<Array2<f64>> {
    let cfg = config.unwrap_or_default();
    let (ny, nx) = image.dim();

    // Step 1: Take log (shift to avoid log(0))
    let epsilon = 1e-10;
    let log_image = image.mapv(|v| (v.abs() + epsilon).ln());

    // Step 2: Build the homomorphic transfer function
    // H(u,v) = (gamma_H - gamma_L) * (1 - exp(-c * D^2 / D0^2)) + gamma_L
    let dist = freq_distance_grid(ny, nx)?;
    let d0_sq = cfg.cutoff * cfg.cutoff;
    let range = cfg.gamma_high - cfg.gamma_low;

    let mask = dist.mapv(|d| {
        let d_sq = d * d;
        let hp = 1.0 - (-cfg.sharpness * d_sq / d0_sq.max(1e-15)).exp();
        range * hp + cfg.gamma_low
    });

    // Apply filter in frequency domain
    let filtered_log = apply_freq_mask(&log_image, &mask)?;

    // Step 3: Exponentiate
    let result = filtered_log.mapv(|v| v.exp());
    Ok(result)
}

// ---------------------------------------------------------------------------
// Notch filter (remove periodic noise)
// ---------------------------------------------------------------------------

/// Notch (band-reject) filter to remove periodic noise at specific frequencies.
///
/// Creates a pair of notches (at the given frequency and its conjugate) in the
/// frequency domain, each with the specified radius.  Multiple notches can be
/// applied by calling this function repeatedly or providing multiple configs.
pub fn notch_filter(image: &Array2<f64>, notches: &[NotchConfig]) -> NdimageResult<Array2<f64>> {
    if notches.is_empty() {
        return Ok(image.clone());
    }
    let (ny, nx) = image.dim();

    let freqs_y =
        fftfreq(ny, 1.0).map_err(|e| NdimageError::ComputationError(format!("fftfreq: {}", e)))?;
    let freqs_x =
        fftfreq(nx, 1.0).map_err(|e| NdimageError::ComputationError(format!("fftfreq: {}", e)))?;

    // Start with all-pass mask
    let mut mask = Array2::<f64>::ones((ny, nx));

    for notch in notches {
        if notch.radius <= 0.0 {
            return Err(NdimageError::InvalidInput(
                "Notch radius must be positive".into(),
            ));
        }
        let r2 = notch.radius * notch.radius;

        for (i, &fy) in freqs_y.iter().enumerate() {
            for (j, &fx) in freqs_x.iter().enumerate() {
                // Distance to notch center
                let dy = fy - notch.freq_y;
                let dx = fx - notch.freq_x;
                let d1 = dy * dy + dx * dx;

                // Distance to conjugate
                let dy2 = fy + notch.freq_y;
                let dx2 = fx + notch.freq_x;
                let d2 = dy2 * dy2 + dx2 * dx2;

                // Reject if inside either notch
                if d1 < r2 || d2 < r2 {
                    mask[[i, j]] = 0.0;
                }
            }
        }
    }

    apply_freq_mask(image, &mask)
}

// ---------------------------------------------------------------------------
// Wiener deconvolution
// ---------------------------------------------------------------------------

/// Wiener deconvolution: restore an image blurred by a known PSF.
///
/// Given:
///   G(u,v) = H(u,v) * F(u,v) + N(u,v)
/// where G is the observed (blurred+noisy) image, H is the PSF, F is the
/// original, and N is noise.
///
/// The Wiener estimate is:
///   F_hat = (H* / (|H|^2 + K)) * G
///
/// where K is the noise-to-signal power ratio.
///
/// `psf` must have the same shape as `blurred`.  If the PSF is smaller,
/// zero-pad it to image size beforehand.
pub fn wiener_deconvolution(
    blurred: &Array2<f64>,
    psf: &Array2<f64>,
    config: Option<WienerConfig>,
) -> NdimageResult<Array2<f64>> {
    let cfg = config.unwrap_or_default();
    let (ny, nx) = blurred.dim();

    if psf.dim() != (ny, nx) {
        return Err(NdimageError::DimensionError(
            "PSF must have the same shape as the blurred image (zero-pad if needed)".into(),
        ));
    }
    if cfg.noise_power_ratio < 0.0 {
        return Err(NdimageError::InvalidInput(
            "noise_power_ratio must be non-negative".into(),
        ));
    }

    // FFT of blurred image and PSF
    let g_spec = fft2(blurred, None, None, None)
        .map_err(|e| NdimageError::ComputationError(format!("FFT blurred: {}", e)))?;
    let h_spec = fft2(psf, None, None, None)
        .map_err(|e| NdimageError::ComputationError(format!("FFT PSF: {}", e)))?;

    let k = cfg.noise_power_ratio;

    // F_hat = (H* / (|H|^2 + K)) * G
    let mut f_hat_spec = Array2::<Complex64>::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            let h = h_spec[[i, j]];
            let h_conj = h.conj();
            let h_mag_sq = h.norm_sqr();
            let wiener_factor = h_conj / (h_mag_sq + k);
            f_hat_spec[[i, j]] = wiener_factor * g_spec[[i, j]];
        }
    }

    let result = ifft2(&f_hat_spec, None, None, None)
        .map_err(|e| NdimageError::ComputationError(format!("IFFT: {}", e)))?;

    let mut out = Array2::<f64>::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            out[[i, j]] = result[[i, j]].re;
        }
    }
    Ok(out)
}

/// Create a zero-padded PSF array matching a given image shape.
///
/// The PSF kernel (typically much smaller) is placed at the top-left corner
/// of a zero array of size (ny, nx).  This is the convention expected by
/// FFT-based deconvolution.
pub fn pad_psf_to_image_size(
    psf: &Array2<f64>,
    ny: usize,
    nx: usize,
) -> NdimageResult<Array2<f64>> {
    let (ky, kx) = psf.dim();
    if ky > ny || kx > nx {
        return Err(NdimageError::InvalidInput(
            "PSF is larger than the target image size".into(),
        ));
    }
    let mut padded = Array2::<f64>::zeros((ny, nx));
    // Place kernel centered at origin (top-left, with wrap for negative indices)
    let half_ky = ky / 2;
    let half_kx = kx / 2;
    for ki in 0..ky {
        for kj in 0..kx {
            let yi = (ki + ny - half_ky) % ny;
            let xi = (kj + nx - half_kx) % nx;
            padded[[yi, xi]] = psf[[ki, kj]];
        }
    }
    Ok(padded)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_cutoff(cutoff: f64) -> NdimageResult<()> {
    if cutoff <= 0.0 || cutoff > 0.5 {
        return Err(NdimageError::InvalidInput(
            "Cutoff frequency must be in (0, 0.5]".into(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_test_image(ny: usize, nx: usize) -> Array2<f64> {
        // Low-freq component + high-freq component
        Array2::from_shape_fn((ny, nx), |(i, j)| {
            let low = ((i as f64 / ny as f64) * 2.0 * PI).sin()
                + ((j as f64 / nx as f64) * 2.0 * PI).cos();
            let high = if (i + j) % 2 == 0 { 0.3 } else { -0.3 };
            low + high
        })
    }

    #[test]
    fn test_ideal_lowpass() {
        let img = make_test_image(32, 32);
        let filtered = ideal_lowpass(&img, 0.2).expect("ideal_lowpass failed");
        assert_eq!(filtered.dim(), img.dim());

        // High-freq checkerboard should be reduced
        let orig_var = neighbor_variance(&img);
        let filt_var = neighbor_variance(&filtered);
        assert!(
            filt_var < orig_var,
            "Low-pass should reduce high-freq variance"
        );
    }

    #[test]
    fn test_ideal_highpass() {
        let img = make_test_image(32, 32);
        let filtered = ideal_highpass(&img, 0.2).expect("ideal_highpass failed");
        assert_eq!(filtered.dim(), img.dim());
        // DC should be near zero (mean removed)
        let mean = filtered.sum() / filtered.len() as f64;
        assert!(mean.abs() < 1.0, "High-pass should remove DC component");
    }

    #[test]
    fn test_ideal_bandpass() {
        let img = make_test_image(32, 32);
        let cfg = BandPassConfig {
            low_cutoff: 0.05,
            high_cutoff: 0.2,
        };
        let filtered = ideal_bandpass(&img, &cfg).expect("ideal_bandpass failed");
        assert_eq!(filtered.dim(), img.dim());
    }

    #[test]
    fn test_butterworth_lowpass() {
        let img = make_test_image(32, 32);
        let cfg = ButterworthConfig {
            cutoff: 0.15,
            order: 2,
        };
        let filtered = butterworth_lowpass(&img, &cfg).expect("butterworth_lowpass failed");
        assert_eq!(filtered.dim(), img.dim());
        let orig_var = neighbor_variance(&img);
        let filt_var = neighbor_variance(&filtered);
        assert!(filt_var < orig_var, "Butterworth LP should smooth");
    }

    #[test]
    fn test_butterworth_highpass() {
        let img = make_test_image(32, 32);
        let cfg = ButterworthConfig {
            cutoff: 0.15,
            order: 2,
        };
        let filtered = butterworth_highpass(&img, &cfg).expect("butterworth_highpass failed");
        assert_eq!(filtered.dim(), img.dim());
    }

    #[test]
    fn test_gaussian_lowpass() {
        let img = make_test_image(32, 32);
        let filtered = gaussian_lowpass(&img, 0.1).expect("gaussian_lowpass failed");
        assert_eq!(filtered.dim(), img.dim());
        let orig_var = neighbor_variance(&img);
        let filt_var = neighbor_variance(&filtered);
        assert!(filt_var < orig_var);
    }

    #[test]
    fn test_gaussian_highpass() {
        let img = make_test_image(32, 32);
        let filtered = gaussian_highpass(&img, 0.1).expect("gaussian_highpass failed");
        assert_eq!(filtered.dim(), img.dim());
    }

    #[test]
    fn test_homomorphic_filter() {
        // Image with large dynamic range
        let img = Array2::from_shape_fn((32, 32), |(i, j)| {
            let illumination = 50.0 + 40.0 * ((i as f64 / 32.0) * PI).cos();
            let reflectance = 0.5 + 0.3 * ((j as f64 / 4.0) * 2.0 * PI).sin();
            illumination * reflectance
        });

        let filtered = homomorphic_filter(&img, None).expect("homomorphic failed");
        assert_eq!(filtered.dim(), img.dim());
        // All values should be positive (exponential output)
        assert!(filtered.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_notch_filter() {
        let img = make_test_image(32, 32);
        let notches = vec![NotchConfig {
            freq_y: 0.0,
            freq_x: 0.25,
            radius: 0.05,
        }];
        let filtered = notch_filter(&img, &notches).expect("notch_filter failed");
        assert_eq!(filtered.dim(), img.dim());
    }

    #[test]
    fn test_notch_filter_empty() {
        let img = make_test_image(16, 16);
        let filtered = notch_filter(&img, &[]).expect("empty notch should pass through");
        for (a, b) in img.iter().zip(filtered.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn test_wiener_deconvolution_identity_psf() {
        let img = make_test_image(32, 32);
        // PSF = delta -> should recover input (modulo numerical noise)
        let mut psf = Array2::<f64>::zeros((32, 32));
        psf[[0, 0]] = 1.0;

        let restored = wiener_deconvolution(
            &img,
            &psf,
            Some(WienerConfig {
                noise_power_ratio: 0.0,
            }),
        )
        .expect("wiener failed");

        let max_err = img
            .iter()
            .zip(restored.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-8,
            "Identity PSF should recover input, max_err={}",
            max_err
        );
    }

    #[test]
    fn test_wiener_deconvolution_blurred() {
        let img = Array2::from_shape_fn((32, 32), |(i, j)| {
            ((i as f64 / 8.0).sin() * (j as f64 / 8.0).cos()) * 100.0
        });

        // Simple box-blur PSF (3x3), padded to image size
        let mut psf_small = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                psf_small[[i, j]] = 1.0 / 9.0;
            }
        }
        let psf = pad_psf_to_image_size(&psf_small, 32, 32).expect("pad_psf failed");

        // Blur the image
        let blurred_spec = fft2(&img, None, None, None).expect("fft2 failed");
        let psf_spec = fft2(&psf, None, None, None).expect("fft2 failed");
        let mut blurred_freq = Array2::<Complex64>::zeros((32, 32));
        for i in 0..32 {
            for j in 0..32 {
                blurred_freq[[i, j]] = blurred_spec[[i, j]] * psf_spec[[i, j]];
            }
        }
        let blurred_complex = ifft2(&blurred_freq, None, None, None).expect("ifft2 failed");
        let blurred = blurred_complex.mapv(|c| c.re);

        // Deconvolve
        let restored = wiener_deconvolution(
            &blurred,
            &psf,
            Some(WienerConfig {
                noise_power_ratio: 0.001,
            }),
        )
        .expect("wiener failed");

        // Restored should be closer to original than blurred is
        let err_blurred: f64 = img
            .iter()
            .zip(blurred.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();
        let err_restored: f64 = img
            .iter()
            .zip(restored.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();
        assert!(
            err_restored < err_blurred,
            "Wiener should reduce error: blurred={}, restored={}",
            err_blurred,
            err_restored
        );
    }

    #[test]
    fn test_pad_psf() {
        let psf = Array2::from_shape_fn((3, 3), |(i, j)| (i * 3 + j + 1) as f64);
        let padded = pad_psf_to_image_size(&psf, 8, 8).expect("pad failed");
        assert_eq!(padded.dim(), (8, 8));
        // Sum should be preserved
        let orig_sum: f64 = psf.iter().sum();
        let pad_sum: f64 = padded.iter().sum();
        assert!((orig_sum - pad_sum).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_cutoff() {
        let img = Array2::zeros((8, 8));
        assert!(ideal_lowpass(&img, 0.0).is_err());
        assert!(ideal_lowpass(&img, 0.6).is_err());
        assert!(ideal_lowpass(&img, -0.1).is_err());
    }

    #[test]
    fn test_butterworth_invalid_order() {
        let img = Array2::zeros((8, 8));
        let cfg = ButterworthConfig {
            cutoff: 0.1,
            order: 0,
        };
        assert!(butterworth_lowpass(&img, &cfg).is_err());
    }

    #[test]
    fn test_bandpass_invalid_range() {
        let img = Array2::zeros((8, 8));
        let cfg = BandPassConfig {
            low_cutoff: 0.3,
            high_cutoff: 0.1,
        };
        assert!(ideal_bandpass(&img, &cfg).is_err());
    }

    #[test]
    fn test_wiener_shape_mismatch() {
        let img = Array2::zeros((8, 8));
        let psf = Array2::zeros((4, 4));
        assert!(wiener_deconvolution(&img, &psf, None).is_err());
    }

    // Helper: average squared difference between adjacent pixels (rough HF measure)
    fn neighbor_variance(img: &Array2<f64>) -> f64 {
        let (ny, nx) = img.dim();
        let mut sum = 0.0;
        let mut count = 0.0;
        for i in 0..ny {
            for j in 1..nx {
                let d = img[[i, j]] - img[[i, j - 1]];
                sum += d * d;
                count += 1.0;
            }
        }
        if count > 0.0 {
            sum / count
        } else {
            0.0
        }
    }
}
