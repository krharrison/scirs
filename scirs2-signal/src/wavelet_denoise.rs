//! Wavelet-based signal denoising
//!
//! This module provides a clean, self-contained API for wavelet denoising that
//! builds on the lower-level DWT machinery in [`crate::dwt`] and [`crate::swt`].
//!
//! # Features
//!
//! - **Discrete Wavelet Transform** (DWT): `dwt` / `idwt` with a
//!   `DwtResult` container that keeps approximation and per-level detail
//!   coefficients together.
//! - **Stationary Wavelet Transform** (SWT / undecimated): `swt` / `iswt` and
//!   `cycle_spinning_denoise` for translation-invariant denoising.
//! - **Thresholding**: hard, soft, and Garrote operators.
//! - **Threshold selection**: universal (VisuShrink), SURE, BayesShrink,
//!   and MAD-based noise estimation.
//! - **End-to-end pipeline**: `denoise_signal` applies the full
//!   decompose → threshold → reconstruct pipeline in one call.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_signal::wavelet_denoise::{
//!     denoise_signal, WaveletType, ThresholdMethod,
//! };
//!
//! let signal: Vec<f64> = (0..128)
//!     .map(|i| (i as f64 * 0.1).sin() + 0.05 * (i as f64 * 1.7).cos())
//!     .collect();
//!
//! let denoised = denoise_signal(&signal, WaveletType::Daubechies4, 3,
//!     ThresholdMethod::VisushrinkSoft).expect("denoising failed");
//! assert_eq!(denoised.len(), signal.len());
//! ```

use crate::dwt::Wavelet;
use crate::error::{SignalError, SignalResult};

// ============================================================================
// Public types
// ============================================================================

/// Supported wavelet families for the wavelet-denoising API.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaveletType {
    /// Haar wavelet (db1)
    Haar,
    /// Daubechies 4-tap wavelet (db2)
    Daubechies4,
    /// Daubechies 8-tap wavelet (db4)
    Daubechies8,
    /// Symlet order 4 (sym4)
    Symlet4,
    /// Coiflet order 1 (coif1)
    Coiflet1,
}

impl WaveletType {
    /// Convert to the underlying [`Wavelet`] enum used by the DWT engine.
    pub fn to_wavelet(self) -> Wavelet {
        match self {
            WaveletType::Haar => Wavelet::Haar,
            WaveletType::Daubechies4 => Wavelet::DB(2),
            WaveletType::Daubechies8 => Wavelet::DB(4),
            WaveletType::Symlet4 => Wavelet::Sym(4),
            WaveletType::Coiflet1 => Wavelet::Coif(1),
        }
    }
}

/// Result of a multi-level Discrete Wavelet Transform.
///
/// The approximation coefficients live in `approximation`; the detail
/// coefficients at each level occupy `details[0]` (finest / level-1) through
/// `details[level-1]` (coarsest / level-`level`).
#[derive(Debug, Clone)]
pub struct DwtResult {
    /// Approximation coefficients at the coarsest level.
    pub approximation: Vec<f64>,
    /// Detail coefficients: `details[0]` = finest level, `details.last()` =
    /// coarsest level.
    pub details: Vec<Vec<f64>>,
    /// The wavelet used for this decomposition (needed for reconstruction).
    pub wavelet: WaveletType,
    /// Original signal length (needed for exact reconstruction).
    pub original_length: usize,
}

/// Result of a Stationary Wavelet Transform (undecimated / à trous).
///
/// Both approximation and detail arrays have the same length as the original
/// signal.
#[derive(Debug, Clone)]
pub struct SwtResult {
    /// Approximation coefficients at the coarsest level (same length as input).
    pub approximation: Vec<f64>,
    /// Detail coefficients per level; all same length as input.
    pub details: Vec<Vec<f64>>,
    /// The wavelet used.
    pub wavelet: WaveletType,
}

/// Threshold application method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThresholdMethod {
    /// Universal threshold with *soft* shrinkage (VisuShrink soft).
    VisushrinkSoft,
    /// Universal threshold with *hard* zeroing (VisuShrink hard).
    VisushrinkHard,
    /// Stein's Unbiased Risk Estimate (SURE / SureShrink).
    Sure,
    /// BayesShrink: per-level adaptive threshold via a Bayesian estimator.
    BayesShrink,
    /// Universal threshold (alias for `VisushrinkSoft`).
    Universal,
}

// ============================================================================
// Low-level thresholding operators
// ============================================================================

/// Hard thresholding: set |c| < threshold to 0, leave others unchanged.
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::wavelet_denoise::hard_threshold;
/// let c = vec![0.5, 1.5, -0.3, 2.0];
/// let t = hard_threshold(&c, 1.0);
/// assert_eq!(t, vec![0.0, 1.5, 0.0, 2.0]);
/// ```
pub fn hard_threshold(coeffs: &[f64], threshold: f64) -> Vec<f64> {
    coeffs
        .iter()
        .map(|&c| if c.abs() < threshold { 0.0 } else { c })
        .collect()
}

/// Soft thresholding (shrinkage): shrink towards zero by `threshold`.
///
/// `sign(c) * max(|c| - threshold, 0)`
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::wavelet_denoise::soft_threshold;
/// let c = vec![0.5, 1.5, -2.0];
/// let t = soft_threshold(&c, 1.0);
/// assert!((t[0]).abs() < 1e-10);
/// assert!((t[1] - 0.5).abs() < 1e-10);
/// assert!((t[2] + 1.0).abs() < 1e-10);
/// ```
pub fn soft_threshold(coeffs: &[f64], threshold: f64) -> Vec<f64> {
    coeffs
        .iter()
        .map(|&c| {
            let abs_c = c.abs();
            if abs_c <= threshold {
                0.0
            } else {
                c.signum() * (abs_c - threshold)
            }
        })
        .collect()
}

/// Garrote (non-negative garrote) thresholding.
///
/// `c - threshold^2 / c` for |c| > threshold, else 0.
///
/// This is a compromise between hard and soft thresholding that has lower
/// bias than soft thresholding and better continuity than hard thresholding.
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::wavelet_denoise::garrote_threshold;
/// let c = vec![2.0, 0.5, -3.0];
/// let t = garrote_threshold(&c, 1.0);
/// // |2.0| > 1.0 → 2.0 - 1/2.0 = 1.5
/// assert!((t[0] - 1.5).abs() < 1e-10);
/// // |0.5| ≤ 1.0 → 0
/// assert_eq!(t[1], 0.0);
/// ```
pub fn garrote_threshold(coeffs: &[f64], threshold: f64) -> Vec<f64> {
    let t2 = threshold * threshold;
    coeffs
        .iter()
        .map(|&c| {
            if c.abs() <= threshold {
                0.0
            } else {
                c - t2 / c
            }
        })
        .collect()
}

// ============================================================================
// Threshold selection methods
// ============================================================================

/// Universal (VisuShrink) threshold: σ √(2 ln n).
///
/// # Arguments
///
/// * `n`     - Signal length
/// * `sigma` - Noise standard deviation estimate
pub fn universal_threshold(n: usize, sigma: f64) -> f64 {
    sigma * (2.0 * (n as f64).ln()).sqrt()
}

/// Estimate noise standard deviation from finest-level detail coefficients
/// using the Median Absolute Deviation (MAD) estimator.
///
/// σ̂ = median(|d|) / 0.6745
///
/// This is the standard robust noise estimator proposed by Donoho & Johnstone.
pub fn estimate_noise_mad(detail_coeffs: &[f64]) -> f64 {
    if detail_coeffs.is_empty() {
        return 0.0;
    }
    // Compute median of the coefficients
    let mut sorted: Vec<f64> = detail_coeffs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_val = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    // MAD = median(|x_i - median(x)|)
    let mut abs_devs: Vec<f64> = detail_coeffs
        .iter()
        .map(|c| (c - median_val).abs())
        .collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = if abs_devs.len() % 2 == 0 {
        (abs_devs[abs_devs.len() / 2 - 1] + abs_devs[abs_devs.len() / 2]) / 2.0
    } else {
        abs_devs[abs_devs.len() / 2]
    };
    // sigma_hat = MAD / 0.6745
    mad / 0.6745
}

/// SURE (Stein's Unbiased Risk Estimate) threshold selector.
///
/// Minimises the SURE risk over all possible soft-threshold values drawn from
/// the sorted absolute values of the coefficients.
///
/// Returns the threshold value that minimises the estimated risk.
pub fn sure_threshold(coeffs: &[f64]) -> f64 {
    let n = coeffs.len();
    if n == 0 {
        return 0.0;
    }

    // Candidate thresholds: sorted |c|
    let mut cands: Vec<f64> = coeffs.iter().map(|c| c.abs()).collect();
    cands.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let sum_sq: f64 = coeffs.iter().map(|c| c * c).sum();

    let mut best_risk = f64::INFINITY;
    let mut best_t = 0.0_f64;

    for &t in &cands {
        // SURE risk = n - 2 * #{|c| <= t} + sum(min(c^2, t^2))
        let k = coeffs.iter().filter(|&&c| c.abs() <= t).count() as f64;
        let clamped: f64 = coeffs.iter().map(|&c| c * c.abs().min(t) / c.abs().max(1e-30)).map(|v| v * v.abs()).sum::<f64>();
        // Simplified SURE:  sum min(c², t²) - n + 2*(n - k)
        let sq_sum_clamped: f64 = coeffs
            .iter()
            .map(|&c| if c.abs() <= t { c * c } else { t * t })
            .sum();
        let risk = (sum_sq - sq_sum_clamped) + sq_sum_clamped - 2.0 * k + n as f64;
        let _ = clamped; // used only for documentation clarity
        if risk < best_risk {
            best_risk = risk;
            best_t = t;
        }
    }

    best_t
}

// ============================================================================
// Multi-level DWT
// ============================================================================

/// Perform a multi-level Discrete Wavelet Transform.
///
/// Decomposes `signal` into `level` levels using the given wavelet.  Returns
/// a [`DwtResult`] containing approximation coefficients and per-level detail
/// coefficients.
///
/// # Arguments
///
/// * `signal`  - Input signal (any length; padding is handled internally)
/// * `wavelet` - Wavelet family to use
/// * `level`   - Number of decomposition levels (clamped to the maximum
///               achievable given the signal length)
///
/// # Errors
///
/// Returns [`SignalError::ValueError`] if the signal is empty or `level` is 0.
pub fn dwt(signal: &[f64], wavelet: WaveletType, level: usize) -> SignalResult<DwtResult> {
    if signal.is_empty() {
        return Err(SignalError::ValueError(
            "dwt: signal must not be empty".to_string(),
        ));
    }
    if level == 0 {
        return Err(SignalError::ValueError(
            "dwt: level must be at least 1".to_string(),
        ));
    }

    let original_length = signal.len();
    let wav = wavelet.to_wavelet();

    // Use existing wavedec — returns [approx_L, detail_L, detail_{L-1}, ..., detail_1]
    let coeffs = crate::dwt::wavedec(signal, wav, Some(level), None)?;

    // wavedec returns at least 1 element (the approximation only if level=0,
    // which we disallow, so we always have ≥ 2 elements).
    if coeffs.len() < 2 {
        return Err(SignalError::ComputationError(
            "dwt: decomposition produced no detail coefficients".to_string(),
        ));
    }

    let approximation = coeffs[0].clone();
    // details[0] = finest (level-1), details.last() = coarsest (level-L)
    // wavedec stores [approx, detail_L, detail_{L-1}, ..., detail_1]
    // so we reverse to get finest-first
    let details: Vec<Vec<f64>> = coeffs[1..].iter().rev().cloned().collect();

    Ok(DwtResult {
        approximation,
        details,
        wavelet,
        original_length,
    })
}

/// Reconstruct a signal from a [`DwtResult`].
///
/// The inverse DWT calls `waverec` on the coefficient arrays stored in
/// `result`, truncating or padding to `result.original_length`.
pub fn idwt(result: &DwtResult) -> SignalResult<Vec<f64>> {
    if result.details.is_empty() {
        return Err(SignalError::ValueError(
            "idwt: DwtResult contains no detail coefficients".to_string(),
        ));
    }

    let wav = result.wavelet.to_wavelet();

    // waverec expects [approx, detail_L, detail_{L-1}, ..., detail_1]
    // our details are finest-first, so we need to reverse them
    let mut coeffs = vec![result.approximation.clone()];
    for d in result.details.iter().rev() {
        coeffs.push(d.clone());
    }

    let reconstructed = crate::dwt::waverec(&coeffs, wav)?;

    // Trim or zero-pad to original length
    let mut out = reconstructed;
    out.resize(result.original_length, 0.0);
    Ok(out)
}

// ============================================================================
// Stationary Wavelet Transform (undecimated)
// ============================================================================

/// Perform a multi-level Stationary (Undecimated) Wavelet Transform.
///
/// Unlike the standard DWT the SWT does not downsample, so all coefficient
/// arrays have the same length as the input signal.  This translation
/// invariance makes SWT-based denoising superior to DWT-based denoising when
/// the exact location of features in the signal matters.
///
/// # Arguments
///
/// * `signal`  - Input signal
/// * `wavelet` - Wavelet family
/// * `level`   - Number of decomposition levels
///
/// # Errors
///
/// Returns [`SignalError::ValueError`] for empty signals or zero levels.
pub fn swt(signal: &[f64], wavelet: WaveletType, level: usize) -> SignalResult<SwtResult> {
    if signal.is_empty() {
        return Err(SignalError::ValueError(
            "swt: signal must not be empty".to_string(),
        ));
    }
    if level == 0 {
        return Err(SignalError::ValueError(
            "swt: level must be at least 1".to_string(),
        ));
    }

    let wav = wavelet.to_wavelet();
    // crate::swt::swt returns (details_vec, final_approximation)
    // where details_vec[0] = level-1 (finest), details_vec[last] = level-L (coarsest)
    let (details_vec, approximation) = crate::swt::swt(signal, wav, level, None)?;

    // Store details finest-first (they come out finest-first from crate::swt::swt)
    let details: Vec<Vec<f64>> = details_vec;

    Ok(SwtResult {
        approximation,
        details,
        wavelet,
    })
}

/// Reconstruct a signal from a [`SwtResult`] using the inverse SWT.
pub fn iswt(result: &SwtResult) -> SignalResult<Vec<f64>> {
    if result.details.is_empty() {
        return Err(SignalError::ValueError(
            "iswt: SwtResult contains no detail coefficients".to_string(),
        ));
    }

    let wav = result.wavelet.to_wavelet();
    // crate::swt::iswt(details, approx, wavelet)
    // details should be in the same order as returned by crate::swt::swt
    crate::swt::iswt(&result.details, &result.approximation, wav)
}

// ============================================================================
// End-to-end denoising pipelines
// ============================================================================

/// Denoise a signal using multi-level DWT thresholding.
///
/// The pipeline is:
/// 1. Decompose with `dwt(signal, wavelet, level)`.
/// 2. Estimate noise σ from the finest-level detail coefficients via MAD.
/// 3. Compute a threshold according to `method`.
/// 4. Apply the threshold to *all* detail levels (approximation is left
///    untouched).
/// 5. Reconstruct with `idwt`.
///
/// # Arguments
///
/// * `signal`  - Noisy input signal
/// * `wavelet` - Wavelet family
/// * `level`   - Decomposition depth
/// * `method`  - Threshold selection and application strategy
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::wavelet_denoise::{denoise_signal, WaveletType, ThresholdMethod};
///
/// let n = 256usize;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (i as f64 * std::f64::consts::PI / 32.0).sin())
///     .collect();
/// let denoised = denoise_signal(&signal, WaveletType::Haar, 2,
///     ThresholdMethod::VisushrinkSoft).expect("failed");
/// assert_eq!(denoised.len(), n);
/// ```
pub fn denoise_signal(
    signal: &[f64],
    wavelet: WaveletType,
    level: usize,
    method: ThresholdMethod,
) -> SignalResult<Vec<f64>> {
    let mut result = dwt(signal, wavelet, level)?;

    // Estimate noise from finest-level detail
    let sigma = estimate_noise_mad(
        result
            .details
            .first()
            .ok_or_else(|| SignalError::ComputationError("no detail coefficients".to_string()))?,
    );

    let n = signal.len();

    for detail in result.details.iter_mut() {
        let threshold = match method {
            ThresholdMethod::VisushrinkSoft | ThresholdMethod::Universal => {
                universal_threshold(n, sigma)
            }
            ThresholdMethod::VisushrinkHard => universal_threshold(n, sigma),
            ThresholdMethod::Sure => sure_threshold(detail),
            ThresholdMethod::BayesShrink => {
                // BayesShrink per-level threshold: sigma_n^2 / sigma_x
                // where sigma_x = sqrt(max(var(d) - sigma_n^2, 0))
                let var_d = variance(detail);
                let sigma_n2 = sigma * sigma;
                let sigma_x = (var_d - sigma_n2).max(0.0).sqrt();
                if sigma_x < 1e-10 {
                    universal_threshold(n, sigma)
                } else {
                    sigma_n2 / sigma_x
                }
            }
        };

        *detail = match method {
            ThresholdMethod::VisushrinkHard => hard_threshold(detail, threshold),
            ThresholdMethod::VisushrinkSoft
            | ThresholdMethod::Universal
            | ThresholdMethod::Sure
            | ThresholdMethod::BayesShrink => soft_threshold(detail, threshold),
        };
    }

    idwt(&result)
}

/// Cycle-spinning denoising using the Stationary Wavelet Transform.
///
/// Cycle spinning averages over all cyclic shifts of the signal to eliminate
/// the translation-dependent artefacts that arise with the standard DWT.  For
/// a signal of length *n* this performs *n* DWT denoisings (one per shift) and
/// returns the average reconstruction.
///
/// For large signals this can be expensive; consider limiting the number of
/// shifts by passing a sub-sampled grid in custom code, or use
/// `swt`-based denoising directly.
///
/// # Arguments
///
/// * `signal`  - Noisy input signal
/// * `wavelet` - Wavelet family
/// * `level`   - Decomposition depth
///
/// # Notes
///
/// This implementation uses the SWT (which is equivalent to averaging over all
/// shifts) rather than explicitly iterating over shifts, making it O(n log n)
/// rather than O(n² log n).
pub fn cycle_spinning_denoise(
    signal: &[f64],
    wavelet: WaveletType,
    level: usize,
) -> SignalResult<Vec<f64>> {
    let mut swt_result = swt(signal, wavelet, level)?;

    // Estimate noise from finest-level detail via MAD
    let sigma = estimate_noise_mad(
        swt_result
            .details
            .first()
            .ok_or_else(|| SignalError::ComputationError("no detail coefficients".to_string()))?,
    );

    let n = signal.len();
    let threshold = universal_threshold(n, sigma);

    for detail in swt_result.details.iter_mut() {
        *detail = soft_threshold(detail, threshold);
    }

    iswt(&swt_result)
}

// ============================================================================
// Internal helpers
// ============================================================================

fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Thresholding operators
    // -----------------------------------------------------------------------

    #[test]
    fn test_hard_threshold() {
        let c = vec![0.5, 1.5, -0.3, 2.0];
        let t = hard_threshold(&c, 1.0);
        assert_eq!(t, vec![0.0, 1.5, 0.0, 2.0]);
    }

    #[test]
    fn test_soft_threshold() {
        let c = vec![2.0, -2.0, 0.5, -0.5];
        let t = soft_threshold(&c, 1.0);
        assert!((t[0] - 1.0).abs() < 1e-10, "expected 1.0, got {}", t[0]);
        assert!((t[1] + 1.0).abs() < 1e-10, "expected -1.0, got {}", t[1]);
        assert_eq!(t[2], 0.0);
        assert_eq!(t[3], 0.0);
    }

    #[test]
    fn test_garrote_threshold() {
        let c = vec![2.0, 0.5, -3.0];
        let t = garrote_threshold(&c, 1.0);
        // 2.0 - 1/2.0 = 1.5
        assert!((t[0] - 1.5).abs() < 1e-10, "expected 1.5, got {}", t[0]);
        assert_eq!(t[1], 0.0);
        // -3.0 - 1/(-3.0) = -3.0 + 1/3 ≈ -2.667
        assert!((t[2] + 3.0 - 1.0 / 3.0).abs() < 1e-10, "garrote[-3] wrong");
    }

    // -----------------------------------------------------------------------
    // Threshold selectors
    // -----------------------------------------------------------------------

    #[test]
    fn test_universal_threshold() {
        let t = universal_threshold(1024, 1.0);
        let expected = (2.0 * (1024_f64).ln()).sqrt();
        assert!((t - expected).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_noise_mad_constant() {
        // All detail coefficients equal → MAD = 0 → sigma = 0
        let c = vec![1.0_f64; 64];
        assert!(estimate_noise_mad(&c) < 1e-12);
    }

    #[test]
    fn test_estimate_noise_mad_gaussian() {
        // For a standard normal the MAD/0.6745 estimator should be close to 1
        // Use a deterministic pseudo-random sequence
        let c: Vec<f64> = (0..1024)
            .map(|i| {
                // simple deterministic "noise" via a sine mix
                (i as f64 * 0.7).sin() + (i as f64 * 1.3).cos()
            })
            .collect();
        let sigma = estimate_noise_mad(&c);
        assert!(sigma > 0.0, "sigma should be positive");
    }

    #[test]
    fn test_sure_threshold_returns_nonnegative() {
        let c: Vec<f64> = (0..64).map(|i| (i as f64 * 0.3).sin()).collect();
        let t = sure_threshold(&c);
        assert!(t >= 0.0, "SURE threshold must be non-negative");
    }

    // -----------------------------------------------------------------------
    // DWT round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_dwt_idwt_roundtrip_haar() {
        let signal: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let result = dwt(&signal, WaveletType::Haar, 3).expect("dwt failed");
        let reconstructed = idwt(&result).expect("idwt failed");
        assert_eq!(reconstructed.len(), signal.len());
        for (a, b) in signal.iter().zip(reconstructed.iter()) {
            assert!(
                (a - b).abs() < 1e-8,
                "round-trip error too large: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_dwt_idwt_roundtrip_db4() {
        let signal: Vec<f64> = (0..128)
            .map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.3).cos())
            .collect();
        let result = dwt(&signal, WaveletType::Daubechies8, 4).expect("dwt failed");
        let reconstructed = idwt(&result).expect("idwt failed");
        assert_eq!(reconstructed.len(), signal.len());
        let mse: f64 = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / signal.len() as f64;
        assert!(mse < 1e-10, "DWT round-trip MSE too large: {mse}");
    }

    #[test]
    fn test_dwt_empty_signal_error() {
        let result = dwt(&[], WaveletType::Haar, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_dwt_zero_level_error() {
        let result = dwt(&[1.0, 2.0, 3.0, 4.0], WaveletType::Haar, 0);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // SWT round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_swt_iswt_roundtrip() {
        let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.2).sin()).collect();
        let result = swt(&signal, WaveletType::Haar, 2).expect("swt failed");
        // All coefficient arrays should have the same length as the signal
        assert_eq!(result.approximation.len(), signal.len());
        for d in &result.details {
            assert_eq!(d.len(), signal.len());
        }
        let reconstructed = iswt(&result).expect("iswt failed");
        assert_eq!(reconstructed.len(), signal.len());
        let mse: f64 = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / signal.len() as f64;
        assert!(mse < 1e-8, "SWT round-trip MSE too large: {mse}");
    }

    // -----------------------------------------------------------------------
    // Denoising pipelines
    // -----------------------------------------------------------------------

    #[test]
    fn test_denoise_signal_reduces_noise() {
        use std::f64::consts::PI;
        let n = 256usize;
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / n as f64).sin())
            .collect();
        // Add deterministic noise
        let noisy: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &c)| c + 0.3 * (i as f64 * 17.0).sin())
            .collect();

        let denoised = denoise_signal(&noisy, WaveletType::Daubechies8, 4,
            ThresholdMethod::VisushrinkSoft)
            .expect("denoise_signal failed");

        assert_eq!(denoised.len(), n);

        // Denoised signal should have lower MSE against clean than noisy does
        let mse_noisy: f64 = clean.iter().zip(noisy.iter())
            .map(|(c, n)| (c - n).powi(2)).sum::<f64>() / n as f64;
        let mse_denoised: f64 = clean.iter().zip(denoised.iter())
            .map(|(c, d)| (c - d).powi(2)).sum::<f64>() / n as f64;
        assert!(
            mse_denoised < mse_noisy,
            "denoised MSE ({mse_denoised:.6}) should be < noisy MSE ({mse_noisy:.6})"
        );
    }

    #[test]
    fn test_denoise_signal_all_methods() {
        let signal: Vec<f64> = (0..128)
            .map(|i| (i as f64 * 0.15).sin() + 0.1 * (i as f64 * 2.7).cos())
            .collect();

        for method in [
            ThresholdMethod::VisushrinkSoft,
            ThresholdMethod::VisushrinkHard,
            ThresholdMethod::Sure,
            ThresholdMethod::BayesShrink,
            ThresholdMethod::Universal,
        ] {
            let out = denoise_signal(&signal, WaveletType::Daubechies4, 3, method)
                .expect("denoise_signal failed");
            assert_eq!(out.len(), signal.len(), "length mismatch for {method:?}");
        }
    }

    #[test]
    fn test_cycle_spinning_denoise() {
        let signal: Vec<f64> = (0..64)
            .map(|i| (i as f64 * 0.3).sin())
            .collect();
        let denoised = cycle_spinning_denoise(&signal, WaveletType::Haar, 2)
            .expect("cycle_spinning_denoise failed");
        assert_eq!(denoised.len(), signal.len());
    }

    // -----------------------------------------------------------------------
    // Wavelet type conversion
    // -----------------------------------------------------------------------

    #[test]
    fn test_wavelet_type_conversion() {
        let types = [
            WaveletType::Haar,
            WaveletType::Daubechies4,
            WaveletType::Daubechies8,
            WaveletType::Symlet4,
            WaveletType::Coiflet1,
        ];
        for wt in types {
            let _ = wt.to_wavelet(); // must not panic
        }
    }
}
