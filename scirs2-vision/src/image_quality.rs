//! Image quality assessment metrics (ndarray API)
//!
//! This module provides no-reference and full-reference image quality metrics
//! operating on `ndarray::Array2<f64>` and `Array3<f64>` (C×H×W) images
//! with values normalised to `[0, 1]`:
//!
//! | Function | Type | Description |
//! |----------|------|-------------|
//! | [`ssim`] | Full-ref | Structural Similarity Index |
//! | [`psnr`] | Full-ref | Peak Signal-to-Noise Ratio |
//! | [`mse_image`] | Full-ref | Mean Squared Error |
//! | [`niqe_score`] | No-ref | Simplified NIQE quality estimate |
//! | [`brisque_features`] | No-ref | BRISQUE MSCN coefficient features |
//!
//! # Note on normalisation
//!
//! All functions expect pixel values in `[0, 1]` unless otherwise stated.
//! For `psnr` the `max_value` parameter should be set to `1.0` when the input
//! is in `[0, 1]`, or `255.0` when the input is in `[0, 255]`.
//!
//! # Example
//!
//! ```rust
//! use scirs2_vision::image_quality::{ssim, psnr, mse_image};
//! use scirs2_core::ndarray::Array2;
//!
//! let a: Array2<f64> = Array2::from_elem((64, 64), 0.5);
//! let b: Array2<f64> = Array2::from_elem((64, 64), 0.5);
//!
//! assert_eq!(mse_image(&a, &b).unwrap(), 0.0);
//! assert!(psnr(&a, &b, 1.0).unwrap().is_infinite());
//! assert!((ssim(&a, &b).unwrap() - 1.0).abs() < 1e-6);
//! ```

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::{s, Array1, Array2};

// ─────────────────────────────────────────────────────────────────────────────
// MSE
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the mean squared error between two images.
///
/// `MSE = Σ(a - b)² / n`
///
/// Both images must have identical shapes.  Returns `0.0` for identical
/// inputs.
///
/// # Arguments
///
/// * `img1` – Reference image, shape `(H, W)`.
/// * `img2` – Distorted image, shape `(H, W)`.
///
/// # Errors
///
/// Returns [`VisionError::DimensionMismatch`] when shapes differ.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::image_quality::mse_image;
/// use scirs2_core::ndarray::Array2;
///
/// let a: Array2<f64> = Array2::zeros((32, 32));
/// let b: Array2<f64> = Array2::from_elem((32, 32), 0.1);
/// let m = mse_image(&a, &b).unwrap();
/// assert!((m - 0.01).abs() < 1e-9);
/// ```
pub fn mse_image(img1: &Array2<f64>, img2: &Array2<f64>) -> Result<f64> {
    check_same_shape(img1, img2)?;
    let n = img1.len();
    if n == 0 {
        return Ok(0.0);
    }
    let sum_sq: f64 = img1
        .iter()
        .zip(img2.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum();
    Ok(sum_sq / n as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// PSNR
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Peak Signal-to-Noise Ratio between two images.
///
/// `PSNR = 20 · log₁₀(max_value / √MSE)`
///
/// Returns `+∞` when `MSE = 0` (identical images).
///
/// # Arguments
///
/// * `original`   – Reference image, shape `(H, W)`.
/// * `distorted`  – Distorted image, shape `(H, W)`.
/// * `max_value`  – Dynamic range (e.g. `1.0` or `255.0`).
///
/// # Errors
///
/// Returns [`VisionError::DimensionMismatch`] when shapes differ.
/// Returns [`VisionError::InvalidParameter`] when `max_value ≤ 0`.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::image_quality::psnr;
/// use scirs2_core::ndarray::Array2;
///
/// let a: Array2<f64> = Array2::from_elem((32, 32), 0.5);
/// let b: Array2<f64> = Array2::from_elem((32, 32), 0.4);
/// let p = psnr(&a, &b, 1.0).unwrap();
/// assert!(p > 0.0 && p < f64::INFINITY);
/// ```
pub fn psnr(original: &Array2<f64>, distorted: &Array2<f64>, max_value: f64) -> Result<f64> {
    if max_value <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "max_value must be > 0".to_string(),
        ));
    }
    let m = mse_image(original, distorted)?;
    if m == 0.0 {
        return Ok(f64::INFINITY);
    }
    Ok(20.0 * (max_value / m.sqrt()).log10())
}

// ─────────────────────────────────────────────────────────────────────────────
// SSIM
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Structural Similarity Index (SSIM) between two images.
///
/// The implementation follows Wang et al. (2004) with a sliding 11×11
/// Gaussian window (`σ = 1.5`):
///
/// `SSIM(x, y) = (2 μ_x μ_y + C1)(2 σ_{xy} + C2) /
///               ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))`
///
/// with `C1 = (0.01 · L)²`, `C2 = (0.03 · L)²`, `L = 1.0`.
///
/// The return value is the *mean* SSIM over all valid (interior) windows.
///
/// # Arguments
///
/// * `img1` – Reference image, shape `(H, W)`.
/// * `img2` – Distorted image, shape `(H, W)`.
///
/// # Errors
///
/// Returns [`VisionError::DimensionMismatch`] when shapes differ.
/// Returns [`VisionError::InvalidParameter`] when the image is too small
/// for the 11×11 window.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::image_quality::ssim;
/// use scirs2_core::ndarray::Array2;
///
/// let a: Array2<f64> = Array2::from_elem((64, 64), 0.5);
/// assert!((ssim(&a, &a).unwrap() - 1.0).abs() < 1e-9);
/// ```
pub fn ssim(img1: &Array2<f64>, img2: &Array2<f64>) -> Result<f64> {
    check_same_shape(img1, img2)?;

    let window_size = 11usize;
    let sigma = 1.5_f64;
    let (h, w) = img1.dim();

    if h < window_size || w < window_size {
        return Err(VisionError::InvalidParameter(format!(
            "Image too small ({h}×{w}) for 11×11 SSIM window"
        )));
    }

    let half = window_size / 2;
    let window = gaussian_kernel_2d(window_size, sigma);
    let win_sum: f64 = window.iter().sum();

    // SSIM constants for dynamic range L = 1.0
    let c1 = (0.01_f64).powi(2);
    let c2 = (0.03_f64).powi(2);

    let mut ssim_sum = 0.0_f64;
    let mut count = 0usize;

    for y in half..h - half {
        for x in half..w - half {
            let patch1 = img1.slice(s![y - half..=y + half, x - half..=x + half]);
            let patch2 = img2.slice(s![y - half..=y + half, x - half..=x + half]);

            // Weighted statistics
            let mut mu1 = 0.0_f64;
            let mut mu2 = 0.0_f64;
            for ((wy, wx), &w_val) in window.indexed_iter() {
                mu1 += patch1[[wy, wx]] * w_val;
                mu2 += patch2[[wy, wx]] * w_val;
            }
            mu1 /= win_sum;
            mu2 /= win_sum;

            let mut var1 = 0.0_f64;
            let mut var2 = 0.0_f64;
            let mut cov = 0.0_f64;
            for ((wy, wx), &w_val) in window.indexed_iter() {
                let d1 = patch1[[wy, wx]] - mu1;
                let d2 = patch2[[wy, wx]] - mu2;
                var1 += w_val * d1 * d1;
                var2 += w_val * d2 * d2;
                cov += w_val * d1 * d2;
            }
            var1 /= win_sum;
            var2 /= win_sum;
            cov /= win_sum;

            let num = (2.0 * mu1 * mu2 + c1) * (2.0 * cov + c2);
            let den = (mu1 * mu1 + mu2 * mu2 + c1) * (var1 + var2 + c2);

            ssim_sum += num / den;
            count += 1;
        }
    }

    Ok(if count > 0 {
        ssim_sum / count as f64
    } else {
        1.0
    })
}

/// Build a (normalised) 2D Gaussian kernel.
fn gaussian_kernel_2d(size: usize, sigma: f64) -> Array2<f64> {
    let half = size as i64 / 2;
    let mut k = Array2::zeros((size, size));
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0_f64;

    for y in 0..size {
        for x in 0..size {
            let dy = y as i64 - half;
            let dx = x as i64 - half;
            let v = (-(dy * dy + dx * dx) as f64 / two_sigma_sq).exp();
            k[[y, x]] = v;
            sum += v;
        }
    }
    if sum > 0.0 {
        k.mapv_inplace(|v| v / sum);
    }
    k
}

// ─────────────────────────────────────────────────────────────────────────────
// NIQE – no-reference quality estimate
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a simplified no-reference quality score inspired by NIQE.
///
/// The Natural Image Quality Evaluator (NIQE, Mittal et al., 2013) fits a
/// multivariate Gaussian model to MSCN (Mean-Subtracted Contrast-Normalised)
/// coefficients of a *pristine* corpus and measures how far a test image
/// departs from that model.
///
/// This implementation uses a small fixed "corpus prior" (mean=0, std=1) and
/// computes a scalar distance score via:
///
/// 1. Extract patch MSCN statistics (mean, variance, skewness, kurtosis).
/// 2. Average over non-overlapping `patch_size × patch_size` patches.
/// 3. Return the Mahalanobis-like distance from the natural image prior.
///
/// **Interpretation**: lower scores indicate better quality (closer to natural
/// image statistics).  The scale is relative and useful only for comparison
/// between images.
///
/// # Arguments
///
/// * `image` – Grayscale image, shape `(H, W)`, values in `[0, 1]`.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when the image is too small.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::image_quality::niqe_score;
/// use scirs2_core::ndarray::Array2;
///
/// let natural_like: Array2<f64> = Array2::from_shape_fn((64, 64), |(y, x)| {
///     (y as f64 * 0.1 + x as f64 * 0.05).sin() * 0.5 + 0.5
/// });
/// let score = niqe_score(&natural_like).unwrap();
/// assert!(score >= 0.0);
/// ```
pub fn niqe_score(image: &Array2<f64>) -> Result<f64> {
    let (h, w) = image.dim();
    let patch_size = 16usize;

    if h < patch_size || w < patch_size {
        return Err(VisionError::InvalidParameter(format!(
            "Image too small ({h}×{w}) for NIQE (need at least {patch_size}×{patch_size})"
        )));
    }

    // Compute MSCN image
    let mscn = compute_mscn(image, 7, 7.0 / 6.0);

    // Collect patch-level statistics
    let mut patch_stats: Vec<[f64; 4]> = Vec::new();
    let mut py = 0;
    while py + patch_size <= h {
        let mut px = 0;
        while px + patch_size <= w {
            let patch = mscn.slice(s![py..py + patch_size, px..px + patch_size]);
            let stats = mscn_patch_stats(patch.to_owned().view());
            patch_stats.push(stats);
            px += patch_size;
        }
        py += patch_size;
    }

    if patch_stats.is_empty() {
        return Ok(0.0);
    }

    // Average statistics across patches
    let n = patch_stats.len() as f64;
    let mut mu = [0.0_f64; 4];
    for s in &patch_stats {
        for (k, mu_k) in mu.iter_mut().enumerate() {
            *mu_k += s[k];
        }
    }
    for mu_k in mu.iter_mut() {
        *mu_k /= n;
    }

    // Natural image prior: mean=0, std=1, skewness≈0, kurtosis≈3
    let prior = [0.0_f64, 1.0_f64, 0.0_f64, 3.0_f64];

    // Simplified Mahalanobis: Σ (μ_k - prior_k)² / prior_variance_k
    let prior_var = [1.0_f64, 1.0_f64, 0.5_f64, 4.0_f64];
    let dist: f64 = (0..4)
        .map(|k| (mu[k] - prior[k]).powi(2) / prior_var[k])
        .sum();

    Ok(dist.sqrt())
}

/// Compute MSCN (Mean-Subtracted Contrast-Normalised) image.
///
/// `MSCN(y,x) = (I(y,x) − μ(y,x)) / (σ(y,x) + ε)`
///
/// where `μ` and `σ` are local mean and standard deviation computed with a
/// Gaussian window.
fn compute_mscn(image: &Array2<f64>, window_size: usize, sigma: f64) -> Array2<f64> {
    let (h, w) = image.dim();
    let kernel = gaussian_kernel_2d(window_size, sigma);
    let half = window_size / 2;
    let win_sum: f64 = kernel.iter().sum();
    const EPS: f64 = 1e-7;

    let mut local_mean = Array2::zeros((h, w));
    let mut local_var = Array2::zeros((h, w));

    // Local mean
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0_f64;
            let mut wsum = 0.0_f64;
            for ky in 0..window_size {
                for kx in 0..window_size {
                    let iy = y as i64 + ky as i64 - half as i64;
                    let ix = x as i64 + kx as i64 - half as i64;
                    if iy >= 0 && iy < h as i64 && ix >= 0 && ix < w as i64 {
                        let wv = kernel[[ky, kx]];
                        sum += image[[iy as usize, ix as usize]] * wv;
                        wsum += wv;
                    }
                }
            }
            local_mean[[y, x]] = if wsum > 0.0 { sum / wsum } else { 0.0 };
        }
    }

    // Local variance
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0_f64;
            let mut wsum = 0.0_f64;
            let mean = local_mean[[y, x]];
            for ky in 0..window_size {
                for kx in 0..window_size {
                    let iy = y as i64 + ky as i64 - half as i64;
                    let ix = x as i64 + kx as i64 - half as i64;
                    if iy >= 0 && iy < h as i64 && ix >= 0 && ix < w as i64 {
                        let wv = kernel[[ky, kx]];
                        let d = image[[iy as usize, ix as usize]] - mean;
                        sum += wv * d * d;
                        wsum += wv;
                    }
                }
            }
            local_var[[y, x]] = if wsum > 0.0 { sum / wsum } else { 0.0 };
        }
    }

    // Normalise
    let mut mscn = Array2::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let sigma_local = local_var[[y, x]].sqrt() + EPS;
            mscn[[y, x]] = (image[[y, x]] - local_mean[[y, x]]) / sigma_local;
        }
    }
    mscn
}

/// Compute mean, variance, skewness, and kurtosis of an MSCN patch.
fn mscn_patch_stats(patch: scirs2_core::ndarray::ArrayView2<f64>) -> [f64; 4] {
    let n = patch.len();
    if n == 0 {
        return [0.0; 4];
    }
    let nf = n as f64;

    let mean: f64 = patch.iter().sum::<f64>() / nf;
    let var: f64 = patch.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / nf;
    let std = var.sqrt();

    let (skew, kurt) = if std > 1e-10 {
        let skew = patch
            .iter()
            .map(|&v| ((v - mean) / std).powi(3))
            .sum::<f64>()
            / nf;
        let kurt = patch
            .iter()
            .map(|&v| ((v - mean) / std).powi(4))
            .sum::<f64>()
            / nf;
        (skew, kurt)
    } else {
        (0.0, 3.0)
    };

    [mean, std, skew, kurt]
}

// ─────────────────────────────────────────────────────────────────────────────
// BRISQUE features
// ─────────────────────────────────────────────────────────────────────────────

/// Compute BRISQUE feature vector from MSCN coefficients.
///
/// BRISQUE (Mittal et al., 2012) characterises local normalised luminance
/// distributions using fitted generalised Gaussian distribution (GGD)
/// parameters and pairwise products of adjacent MSCN coefficients.
///
/// This implementation extracts a 36-element feature vector at two scales
/// (original and 2× downsampled):
///
/// * 4 values per direction × 2 directions (horizontal + vertical) = 8 values
///   from MSCN pairwise products at each scale (α, σ² of fitted GGD).
/// * Plus 2 values from the MSCN image itself (α, σ²) per scale.
/// * Total: `(2 + 8) × 2 scales = 20` → padded/extended to 36 for full BRISQUE
///   by including diagonal products and second-scale mean/std.
///
/// The returned vector can be fed directly into a pre-trained SVM or other
/// regressor for absolute quality prediction.
///
/// # Arguments
///
/// * `image` – Grayscale image, shape `(H, W)`, values in `[0, 1]`.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when the image is smaller than
/// `16 × 16`.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::image_quality::brisque_features;
/// use scirs2_core::ndarray::Array2;
///
/// let img: Array2<f64> = Array2::from_shape_fn((64, 64), |(y, x)| {
///     (y as f64 * 0.1 + x as f64 * 0.05).sin() * 0.5 + 0.5
/// });
/// let feat = brisque_features(&img).unwrap();
/// assert_eq!(feat.len(), 36);
/// ```
pub fn brisque_features(image: &Array2<f64>) -> Result<Array1<f64>> {
    let (h, w) = image.dim();
    if h < 16 || w < 16 {
        return Err(VisionError::InvalidParameter(format!(
            "Image too small ({h}×{w}) for BRISQUE (need >= 16×16)"
        )));
    }

    let mut features: Vec<f64> = Vec::with_capacity(36);

    // Process at two scales
    let mut current = image.to_owned();
    for _scale in 0..2 {
        let (ch, cw) = current.dim();

        // Compute MSCN
        let mscn = compute_mscn(&current, 7, 7.0 / 6.0);

        // Fit GGD to MSCN coefficients
        let (alpha, sigma_sq) = fit_ggd(&mscn);
        features.push(alpha);
        features.push(sigma_sq);

        // Pairwise product statistics for 4 directions (H, V, D1, D2)
        let directions: [(i64, i64); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
        for &(dy, dx) in &directions {
            let products = compute_pairwise_products(&mscn, dy, dx);
            let (lp, rp, mu, sig) = pairwise_ggd_params(&products);
            features.push(lp);
            features.push(rp);
            features.push(mu);
            features.push(sig);
        }

        // Downsample by 2 for next scale
        let new_h = ch / 2;
        let new_w = cw / 2;
        if new_h < 16 || new_w < 16 {
            // Fill remaining features with zeros if image too small to downsample
            let remaining = 36 - features.len();
            features.extend(std::iter::repeat_n(0.0, remaining));
            break;
        }
        let mut downsampled = Array2::zeros((new_h, new_w));
        for y in 0..new_h {
            for x in 0..new_w {
                // Average 2×2 block
                let sum = current[[2 * y, 2 * x]]
                    + current[[2 * y, (2 * x + 1).min(cw - 1)]]
                    + current[[(2 * y + 1).min(ch - 1), 2 * x]]
                    + current[[(2 * y + 1).min(ch - 1), (2 * x + 1).min(cw - 1)]];
                downsampled[[y, x]] = sum / 4.0;
            }
        }
        current = downsampled;
    }

    // Ensure exactly 36 features
    features.truncate(36);
    while features.len() < 36 {
        features.push(0.0);
    }

    Ok(Array1::from(features))
}

/// Fit a Generalised Gaussian Distribution (GGD) to an MSCN image.
///
/// Uses the moment-matching estimator: `α ≈ (E[|x|²] / E[|x|])²`.
/// Returns `(α, σ²)` where `σ²` is the variance.
fn fit_ggd(mscn: &Array2<f64>) -> (f64, f64) {
    let n = mscn.len();
    if n == 0 {
        return (1.0, 1.0);
    }
    let nf = n as f64;

    let mean: f64 = mscn.iter().sum::<f64>() / nf;
    let var: f64 = mscn.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / nf;
    let abs_mean: f64 = mscn.iter().map(|&v| v.abs()).sum::<f64>() / nf;

    let alpha = if abs_mean > 1e-10 {
        (var / (abs_mean * abs_mean)).sqrt()
    } else {
        1.0
    };

    (alpha.clamp(0.1, 10.0), var)
}

/// Compute pairwise products of adjacent MSCN coefficients in direction `(dy, dx)`.
fn compute_pairwise_products(mscn: &Array2<f64>, dy: i64, dx: i64) -> Array2<f64> {
    let (h, w) = mscn.dim();
    let mut prod = Array2::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let ny = y as i64 + dy;
            let nx = x as i64 + dx;
            if ny >= 0 && ny < h as i64 && nx >= 0 && nx < w as i64 {
                prod[[y, x]] = mscn[[y, x]] * mscn[[ny as usize, nx as usize]];
            }
        }
    }
    prod
}

/// Compute asymmetric GGD parameters from pairwise product distribution.
///
/// Returns `(left_param, right_param, mean, sigma)` representing the
/// asymmetric GGD fit (positive and negative half distributions).
fn pairwise_ggd_params(products: &Array2<f64>) -> (f64, f64, f64, f64) {
    let n = products.len();
    if n == 0 {
        return (1.0, 1.0, 0.0, 1.0);
    }
    let nf = n as f64;

    let mean: f64 = products.iter().sum::<f64>() / nf;
    let var: f64 = products.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / nf;

    // Left and right half distributions
    let mut left: Vec<f64> = Vec::new();
    let mut right: Vec<f64> = Vec::new();
    for &v in products.iter() {
        if v < mean {
            left.push((v - mean).abs());
        } else {
            right.push((v - mean).abs());
        }
    }

    let left_var = if !left.is_empty() {
        left.iter().map(|v| v * v).sum::<f64>() / left.len() as f64
    } else {
        1.0
    };
    let right_var = if !right.is_empty() {
        right.iter().map(|v| v * v).sum::<f64>() / right.len() as f64
    } else {
        1.0
    };

    let left_param = left_var.sqrt().max(1e-10);
    let right_param = right_var.sqrt().max(1e-10);

    (left_param, right_param, mean, var.sqrt().max(1e-10))
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Check that two 2D arrays have the same shape.
fn check_same_shape(a: &Array2<f64>, b: &Array2<f64>) -> Result<()> {
    if a.dim() != b.dim() {
        Err(VisionError::DimensionMismatch(format!(
            "Images have different shapes: {:?} vs {:?}",
            a.dim(),
            b.dim()
        )))
    } else {
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn ramp_image(h: usize, w: usize) -> Array2<f64> {
        Array2::from_shape_fn((h, w), |(y, x)| (y * w + x) as f64 / (h * w) as f64)
    }

    fn noise_image(h: usize, w: usize, seed: u64) -> Array2<f64> {
        let mut s = seed;
        Array2::from_shape_fn((h, w), |_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f64) / (u32::MAX as f64)
        })
    }

    // ── MSE ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_mse_identical() {
        let a = ramp_image(32, 32);
        assert_eq!(mse_image(&a, &a).expect("MSE failed"), 0.0);
    }

    #[test]
    fn test_mse_known_value() {
        let a: Array2<f64> = Array2::zeros((32, 32));
        let b: Array2<f64> = Array2::from_elem((32, 32), 0.1);
        let m = mse_image(&a, &b).expect("MSE failed");
        assert!((m - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_mse_dimension_mismatch() {
        let a: Array2<f64> = Array2::zeros((32, 32));
        let b: Array2<f64> = Array2::zeros((16, 16));
        assert!(mse_image(&a, &b).is_err());
    }

    #[test]
    fn test_mse_non_negative() {
        let a = noise_image(32, 32, 1);
        let b = noise_image(32, 32, 2);
        assert!(mse_image(&a, &b).expect("MSE failed") >= 0.0);
    }

    // ── PSNR ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_psnr_identical() {
        let a: Array2<f64> = Array2::from_elem((32, 32), 0.5);
        assert!(psnr(&a, &a, 1.0).expect("PSNR failed").is_infinite());
    }

    #[test]
    fn test_psnr_positive() {
        let a: Array2<f64> = Array2::from_elem((32, 32), 0.5);
        let b: Array2<f64> = Array2::from_elem((32, 32), 0.4);
        let p = psnr(&a, &b, 1.0).expect("PSNR failed");
        assert!(p > 0.0 && p.is_finite());
    }

    #[test]
    fn test_psnr_invalid_max_value() {
        let a: Array2<f64> = Array2::zeros((32, 32));
        assert!(psnr(&a, &a, 0.0).is_err());
        assert!(psnr(&a, &a, -1.0).is_err());
    }

    #[test]
    fn test_psnr_decreases_with_more_noise() {
        let a = ramp_image(32, 32);
        let b_low_noise = a.mapv(|v| (v + 0.01).clamp(0.0, 1.0));
        let b_high_noise = a.mapv(|v| (v + 0.1).clamp(0.0, 1.0));
        let p_low = psnr(&a, &b_low_noise, 1.0).expect("PSNR low noise");
        let p_high = psnr(&a, &b_high_noise, 1.0).expect("PSNR high noise");
        assert!(p_low > p_high, "PSNR should decrease with more noise");
    }

    // ── SSIM ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_ssim_identical_uniform() {
        let a: Array2<f64> = Array2::from_elem((64, 64), 0.5);
        let s = ssim(&a, &a).expect("SSIM failed");
        assert!(
            (s - 1.0).abs() < 1e-6,
            "SSIM of identical images should be 1, got {s}"
        );
    }

    #[test]
    fn test_ssim_identical_ramp() {
        let a = ramp_image(64, 64);
        let s = ssim(&a, &a).expect("SSIM failed");
        assert!(
            (s - 1.0).abs() < 1e-6,
            "SSIM of identical ramp should be 1, got {s}"
        );
    }

    #[test]
    fn test_ssim_range() {
        let a = ramp_image(64, 64);
        let b = noise_image(64, 64, 42);
        let s = ssim(&a, &b).expect("SSIM failed");
        assert!(s <= 1.0, "SSIM must be <= 1, got {s}");
        // Note: SSIM can theoretically be negative for very dissimilar images
        assert!(s.is_finite());
    }

    #[test]
    fn test_ssim_dimension_mismatch() {
        let a: Array2<f64> = Array2::zeros((64, 64));
        let b: Array2<f64> = Array2::zeros((32, 32));
        assert!(ssim(&a, &b).is_err());
    }

    #[test]
    fn test_ssim_small_image() {
        let a: Array2<f64> = Array2::zeros((8, 8));
        assert!(ssim(&a, &a).is_err()); // Too small for 11×11 window
    }

    // ── NIQE ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_niqe_natural_image() {
        let img = Array2::from_shape_fn((64, 64), |(y, x)| {
            (y as f64 * 0.1 + x as f64 * 0.05).sin() * 0.5 + 0.5
        });
        let score = niqe_score(&img).expect("NIQE failed");
        assert!(score >= 0.0 && score.is_finite());
    }

    #[test]
    fn test_niqe_small_image() {
        let img: Array2<f64> = Array2::zeros((8, 8));
        assert!(niqe_score(&img).is_err());
    }

    #[test]
    fn test_niqe_uniform_vs_natural() {
        // Uniform image deviates from natural stats (zero variance)
        let uniform = Array2::from_elem((64, 64), 0.5_f64);
        let natural = Array2::from_shape_fn((64, 64), |(y, x)| {
            let v = (y as f64 * 0.2).sin() * (x as f64 * 0.15).cos();
            v * 0.3 + 0.5
        });
        let score_u = niqe_score(&uniform).expect("NIQE uniform");
        let score_n = niqe_score(&natural).expect("NIQE natural");
        // Both should be finite non-negative
        assert!(score_u >= 0.0 && score_u.is_finite());
        assert!(score_n >= 0.0 && score_n.is_finite());
    }

    // ── BRISQUE ───────────────────────────────────────────────────────────────

    #[test]
    fn test_brisque_feature_length() {
        let img = ramp_image(64, 64);
        let feat = brisque_features(&img).expect("BRISQUE failed");
        assert_eq!(feat.len(), 36);
    }

    #[test]
    fn test_brisque_finite_values() {
        let img = noise_image(64, 64, 99);
        let feat = brisque_features(&img).expect("BRISQUE failed");
        for (i, &v) in feat.iter().enumerate() {
            assert!(v.is_finite(), "Feature {i} is not finite: {v}");
        }
    }

    #[test]
    fn test_brisque_small_image() {
        let img: Array2<f64> = Array2::zeros((8, 8));
        assert!(brisque_features(&img).is_err());
    }

    #[test]
    fn test_brisque_deterministic() {
        let img = noise_image(64, 64, 7);
        let f1 = brisque_features(&img).expect("BRISQUE 1");
        let f2 = brisque_features(&img).expect("BRISQUE 2");
        for (a, b) in f1.iter().zip(f2.iter()) {
            assert_eq!(a, b, "BRISQUE should be deterministic");
        }
    }
}
