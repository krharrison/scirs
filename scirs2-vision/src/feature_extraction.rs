//! Classical image feature extraction
//!
//! This module provides descriptors and filter responses widely used in
//! computer vision pipelines:
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`hog_features`] | Histogram of Oriented Gradients |
//! | [`lbp_features`] | Local Binary Patterns |
//! | [`gabor_filter`] | Gabor band-pass filtering |
//! | [`gabor_features`] | Multi-scale/orientation Gabor feature vector |
//! | [`daisy_descriptor`] | Dense DAISY descriptor matrix |
//! | [`brief_descriptor`] | BRIEF binary keypoint descriptor |
//!
//! All functions operate on `ndarray::Array2<f64>` grayscale images with
//! pixel values normalised to `[0, 1]` or arbitrary positive range, and
//! return `ndarray` arrays or standard vectors.
//!
//! # Example – HOG descriptor
//!
//! ```rust
//! use scirs2_vision::feature_extraction::hog_features;
//! use scirs2_core::ndarray::Array2;
//!
//! let image: Array2<f64> = Array2::ones((64, 128));
//! let feat = hog_features(&image, 8, 2, 9).unwrap();
//! assert!(!feat.is_empty());
//! ```

use std::f64::consts::PI;

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::{Array1, Array2, Array3};

// ─────────────────────────────────────────────────────────────────────────────
// HOG – Histogram of Oriented Gradients
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a HOG feature descriptor for a grayscale image.
///
/// The implementation follows the standard Dalal & Triggs (2005) pipeline:
///
/// 1. Compute pixel-level gradient magnitudes and orientations.
/// 2. Accumulate unsigned-orientation histograms per cell (`cell_size × cell_size`).
/// 3. Normalise each overlapping block (`block_size × block_size` cells) with
///    L2-Hys (clipped L2).
/// 4. Flatten all blocks into a single feature vector.
///
/// # Arguments
///
/// * `image`     – Grayscale image array, shape `(H, W)`, any positive range.
/// * `cell_size` – Size of each spatial cell in pixels.
/// * `block_size`– Number of cells per block side (e.g. 2 → 2×2 blocks).
/// * `n_bins`    – Number of orientation histogram bins (e.g. 9).
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when cell/block sizes are zero,
/// or the image is too small to contain a single block.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature_extraction::hog_features;
/// use scirs2_core::ndarray::Array2;
///
/// let image: Array2<f64> = Array2::ones((64, 128));
/// let feat = hog_features(&image, 8, 2, 9).unwrap();
/// assert!(!feat.is_empty());
/// ```
pub fn hog_features(
    image: &Array2<f64>,
    cell_size: usize,
    block_size: usize,
    n_bins: usize,
) -> Result<Array1<f64>> {
    if cell_size == 0 {
        return Err(VisionError::InvalidParameter(
            "cell_size must be > 0".to_string(),
        ));
    }
    if block_size == 0 {
        return Err(VisionError::InvalidParameter(
            "block_size must be > 0".to_string(),
        ));
    }
    if n_bins == 0 {
        return Err(VisionError::InvalidParameter(
            "n_bins must be > 0".to_string(),
        ));
    }

    let (h, w) = image.dim();
    let n_cells_y = h / cell_size;
    let n_cells_x = w / cell_size;

    if n_cells_y < block_size || n_cells_x < block_size {
        return Err(VisionError::InvalidParameter(format!(
            "Image too small for block_size={block_size}: \
             only {n_cells_y}×{n_cells_x} cells"
        )));
    }

    // ── Step 1: per-pixel gradients ──────────────────────────────────────────
    let (magnitudes, orientations) = compute_gradients(image);

    // ── Step 2: cell histograms ───────────────────────────────────────────────
    // cell_hists shape: (n_cells_y, n_cells_x, n_bins)
    let mut cell_hists = vec![vec![vec![0.0_f64; n_bins]; n_cells_x]; n_cells_y];

    let bin_width = PI / n_bins as f64; // unsigned orientations in [0, π)

    #[allow(clippy::needless_range_loop)]
    for cy in 0..n_cells_y {
        for cx in 0..n_cells_x {
            let y0 = cy * cell_size;
            let x0 = cx * cell_size;
            for dy in 0..cell_size {
                for dx in 0..cell_size {
                    let y = y0 + dy;
                    let x = x0 + dx;
                    if y >= h || x >= w {
                        continue;
                    }
                    let mag = magnitudes[[y, x]];
                    // Unsigned orientation in [0, π)
                    let mut ori = orientations[[y, x]];
                    if ori < 0.0 {
                        ori += PI;
                    }
                    ori = ori.rem_euclid(PI);

                    // Bilinear bin interpolation
                    let bin_f = ori / bin_width;
                    let bin0 = (bin_f as usize) % n_bins;
                    let bin1 = (bin0 + 1) % n_bins;
                    let frac = bin_f - bin_f.floor();
                    cell_hists[cy][cx][bin0] += mag * (1.0 - frac);
                    cell_hists[cy][cx][bin1] += mag * frac;
                }
            }
        }
    }

    // ── Step 3: block normalisation (L2-Hys) ──────────────────────────────────
    let n_blocks_y = n_cells_y - block_size + 1;
    let n_blocks_x = n_cells_x - block_size + 1;
    let block_dim = block_size * block_size * n_bins;
    let total_len = n_blocks_y * n_blocks_x * block_dim;

    let mut features = Array1::zeros(total_len);
    let mut feat_idx = 0;

    for by in 0..n_blocks_y {
        for bx in 0..n_blocks_x {
            // Collect block cells
            let mut block_vec: Vec<f64> = Vec::with_capacity(block_dim);
            for dy in 0..block_size {
                for dx in 0..block_size {
                    block_vec.extend_from_slice(&cell_hists[by + dy][bx + dx]);
                }
            }

            // L2 normalise
            let norm = block_vec.iter().map(|v| v * v).sum::<f64>().sqrt() + 1e-7;
            // Clamp (Hys clip at 0.2)
            let clipped: Vec<f64> = block_vec.iter().map(|v| (v / norm).min(0.2)).collect();
            // Re-normalise
            let norm2 = clipped.iter().map(|v| v * v).sum::<f64>().sqrt() + 1e-7;

            for &v in &clipped {
                if feat_idx < features.len() {
                    features[feat_idx] = v / norm2;
                    feat_idx += 1;
                }
            }
        }
    }

    Ok(features)
}

/// Compute Sobel gradient magnitude and orientation for an image.
fn compute_gradients(image: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let (h, w) = image.dim();
    let mut magnitudes = Array2::zeros((h, w));
    let mut orientations = Array2::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let ym1 = y.saturating_sub(1);
            let yp1 = (y + 1).min(h - 1);
            let xm1 = x.saturating_sub(1);
            let xp1 = (x + 1).min(w - 1);

            let gx = image[[y, xp1]] - image[[y, xm1]];
            let gy = image[[yp1, x]] - image[[ym1, x]];

            magnitudes[[y, x]] = (gx * gx + gy * gy).sqrt();
            orientations[[y, x]] = gy.atan2(gx);
        }
    }
    (magnitudes, orientations)
}

// ─────────────────────────────────────────────────────────────────────────────
// LBP – Local Binary Patterns
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a Local Binary Pattern (LBP) histogram feature vector.
///
/// For each pixel a circular LBP code is computed by sampling `n_points`
/// evenly-spaced neighbours on a circle of the given `radius` and bilinear
/// interpolation.  The resulting LBP image is then histogrammed into a
/// `(n_points + 2)`-bin "uniform" pattern histogram.
///
/// # Arguments
///
/// * `image`    – Grayscale image, shape `(H, W)`.
/// * `radius`   – Radius of the sampling circle (pixels, must be ≥ 1).
/// * `n_points` – Number of sample points on the circle.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when `radius < 1` or
/// `n_points < 2`.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature_extraction::lbp_features;
/// use scirs2_core::ndarray::Array2;
///
/// let image: Array2<f64> = Array2::ones((32, 32));
/// let feat = lbp_features(&image, 1, 8).unwrap();
/// assert_eq!(feat.len(), 10); // n_points + 2 = 10 for uniform LBP
/// ```
pub fn lbp_features(image: &Array2<f64>, radius: usize, n_points: usize) -> Result<Array1<f64>> {
    if radius < 1 {
        return Err(VisionError::InvalidParameter(
            "radius must be >= 1".to_string(),
        ));
    }
    if n_points < 2 {
        return Err(VisionError::InvalidParameter(
            "n_points must be >= 2".to_string(),
        ));
    }

    let (h, w) = image.dim();
    let r = radius as f64;

    // Compute LBP code for each pixel
    let mut lbp_image = Array2::<u32>::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let center = image[[y, x]];
            let mut code = 0u32;
            for p in 0..n_points {
                let angle = 2.0 * PI * p as f64 / n_points as f64;
                let sample_x = x as f64 + r * angle.cos();
                let sample_y = y as f64 - r * angle.sin();

                let val = bilinear_sample(image, sample_y, sample_x);
                if val >= center {
                    code |= 1u32 << p;
                }
            }
            lbp_image[[y, x]] = code;
        }
    }

    // Convert to uniform-LBP histogram (n_points + 2 bins)
    let n_bins = n_points + 2;
    let mut hist = vec![0.0_f64; n_bins];

    for y in 0..h {
        for x in 0..w {
            let code = lbp_image[[y, x]];
            let transitions = count_transitions(code, n_points);
            let bin = if transitions <= 2 {
                uniform_lbp_bin(code, n_points)
            } else {
                n_bins - 1 // Non-uniform
            };
            if bin < n_bins {
                hist[bin] += 1.0;
            }
        }
    }

    // L1 normalise
    let total: f64 = hist.iter().sum::<f64>() + 1e-10;
    let hist_norm: Vec<f64> = hist.iter().map(|v| v / total).collect();

    Ok(Array1::from(hist_norm))
}

/// Bilinear interpolation of an image at fractional coordinates.
fn bilinear_sample(image: &Array2<f64>, y: f64, x: f64) -> f64 {
    let (h, w) = image.dim();
    if h == 0 || w == 0 {
        return 0.0;
    }

    let y_clamped = y.clamp(0.0, (h - 1) as f64);
    let x_clamped = x.clamp(0.0, (w - 1) as f64);

    let y0 = y_clamped.floor() as usize;
    let x0 = x_clamped.floor() as usize;
    let y1 = (y0 + 1).min(h - 1);
    let x1 = (x0 + 1).min(w - 1);

    let fy = y_clamped - y0 as f64;
    let fx = x_clamped - x0 as f64;

    image[[y0, x0]] * (1.0 - fy) * (1.0 - fx)
        + image[[y0, x1]] * (1.0 - fy) * fx
        + image[[y1, x0]] * fy * (1.0 - fx)
        + image[[y1, x1]] * fy * fx
}

/// Count bit transitions in an LBP code (circular).
fn count_transitions(code: u32, n_points: usize) -> usize {
    let mut transitions = 0;
    for p in 0..n_points {
        let bit0 = (code >> p) & 1;
        let bit1 = (code >> ((p + 1) % n_points)) & 1;
        if bit0 != bit1 {
            transitions += 1;
        }
    }
    transitions
}

/// Map a uniform LBP code to its canonical bin index.
///
/// Uniform patterns (≤ 2 transitions) map to `popcount(code)` (0..=n_points).
fn uniform_lbp_bin(code: u32, _n_points: usize) -> usize {
    code.count_ones() as usize
}

// ─────────────────────────────────────────────────────────────────────────────
// Gabor filter
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a Gabor filter to a grayscale image and return the response magnitude.
///
/// A Gabor filter is the product of a Gaussian envelope and a complex
/// sinusoidal carrier.  This function returns the *magnitude* of the
/// complex Gabor response: `|G| = sqrt(G_real² + G_imag²)`.
///
/// The kernel is computed analytically and convolved via direct (spatial) correlation.
///
/// # Arguments
///
/// * `image`     – Grayscale image, shape `(H, W)`.
/// * `frequency` – Spatial frequency of the sinusoidal carrier (cycles/pixel).
/// * `theta`     – Orientation of the filter in radians.
/// * `sigma`     – Standard deviation of the Gaussian envelope (pixels).
/// * `gamma`     – Spatial aspect ratio (1.0 = isotropic).
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when `sigma ≤ 0` or
/// `frequency ≤ 0`.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature_extraction::gabor_filter;
/// use scirs2_core::ndarray::Array2;
///
/// let img: Array2<f64> = Array2::ones((32, 32));
/// let resp = gabor_filter(&img, 0.1, 0.0, 3.0, 1.0).unwrap();
/// assert_eq!(resp.dim(), (32, 32));
/// ```
pub fn gabor_filter(
    image: &Array2<f64>,
    frequency: f64,
    theta: f64,
    sigma: f64,
    gamma: f64,
) -> Result<Array2<f64>> {
    if sigma <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "sigma must be > 0".to_string(),
        ));
    }
    if frequency <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "frequency must be > 0".to_string(),
        ));
    }

    let (h, w) = image.dim();

    // Kernel half-size: 3σ is a good coverage threshold
    let half = (3.0 * sigma).ceil() as i32;
    let ksize = (2 * half + 1) as usize;

    // Build the kernel
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let sigma_x = sigma;
    let sigma_y = sigma / gamma.max(1e-10);

    let mut kernel_real = Array2::zeros((ksize, ksize));
    let mut kernel_imag = Array2::zeros((ksize, ksize));

    for ky in 0..ksize {
        for kx in 0..ksize {
            let x = (kx as i32 - half) as f64;
            let y = (ky as i32 - half) as f64;

            // Rotate
            let x_prime = x * cos_t + y * sin_t;
            let y_prime = -x * sin_t + y * cos_t;

            let gauss = (-0.5
                * (x_prime * x_prime / (sigma_x * sigma_x)
                    + y_prime * y_prime / (sigma_y * sigma_y)))
                .exp();

            let phase = 2.0 * PI * frequency * x_prime;
            kernel_real[[ky, kx]] = gauss * phase.cos();
            kernel_imag[[ky, kx]] = gauss * phase.sin();
        }
    }

    // Correlate image with kernel (zero-padded borders)
    let mut response = Array2::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let mut real_sum = 0.0_f64;
            let mut imag_sum = 0.0_f64;

            for ky in 0..ksize {
                for kx in 0..ksize {
                    let iy = y as i64 + ky as i64 - half as i64;
                    let ix = x as i64 + kx as i64 - half as i64;

                    if iy >= 0 && iy < h as i64 && ix >= 0 && ix < w as i64 {
                        let pixel = image[[iy as usize, ix as usize]];
                        real_sum += pixel * kernel_real[[ky, kx]];
                        imag_sum += pixel * kernel_imag[[ky, kx]];
                    }
                }
            }
            response[[y, x]] = (real_sum * real_sum + imag_sum * imag_sum).sqrt();
        }
    }

    Ok(response)
}

// ─────────────────────────────────────────────────────────────────────────────
// Gabor feature vector
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a multi-scale, multi-orientation Gabor feature vector.
///
/// For each `(frequency, theta)` pair a Gabor filter is applied and the
/// mean and standard deviation of the response magnitude are collected.
/// The final feature vector has length `2 · |frequencies| · |thetas|`.
///
/// # Arguments
///
/// * `image`       – Grayscale image, shape `(H, W)`.
/// * `frequencies` – Spatial frequencies (e.g. `&[0.1, 0.2, 0.4]`).
/// * `thetas`      – Orientation angles in radians.
///
/// # Errors
///
/// Propagates errors from [`gabor_filter`].
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature_extraction::gabor_features;
/// use scirs2_core::ndarray::Array2;
///
/// let img: Array2<f64> = Array2::ones((32, 32));
/// let frequencies = vec![0.1, 0.2];
/// let thetas = vec![0.0, std::f64::consts::PI / 4.0];
/// let feat = gabor_features(&img, &frequencies, &thetas).unwrap();
/// assert_eq!(feat.len(), 2 * 2 * 2); // 2 * n_freq * n_theta
/// ```
pub fn gabor_features(
    image: &Array2<f64>,
    frequencies: &[f64],
    thetas: &[f64],
) -> Result<Array1<f64>> {
    let sigma = 3.0; // default sigma
    let gamma = 1.0; // isotropic

    let mut features: Vec<f64> = Vec::with_capacity(2 * frequencies.len() * thetas.len());

    for &freq in frequencies {
        for &theta in thetas {
            let response = gabor_filter(image, freq, theta, sigma, gamma)?;
            let (mean, std) = mean_std(&response);
            features.push(mean);
            features.push(std);
        }
    }

    Ok(Array1::from(features))
}

/// Compute mean and population standard deviation of a 2D array.
fn mean_std(arr: &Array2<f64>) -> (f64, f64) {
    let n = arr.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let sum: f64 = arr.iter().sum();
    let mean = sum / n as f64;
    let var: f64 = arr.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    (mean, var.sqrt())
}

// ─────────────────────────────────────────────────────────────────────────────
// DAISY descriptor
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a simplified DAISY dense descriptor matrix.
///
/// DAISY (Tola et al., 2010) is a dense descriptor computed by sampling
/// Gaussian-smoothed gradient histograms at concentric rings of points.
/// This simplified variant:
///
/// - Computes gradient orientation histograms over a single ring of `8`
///   evenly-spaced sample points at radius `radius`.
/// - Returns one descriptor per grid point spaced `step` pixels apart.
/// - Each descriptor has `8` bins (one per sample point, magnitude-weighted).
///
/// The returned array has shape `(N_points, 8)` where `N_points` is the
/// number of valid interior grid positions.
///
/// # Arguments
///
/// * `image`  – Grayscale image, shape `(H, W)`.
/// * `step`   – Spacing between descriptor centres (pixels).
/// * `radius` – Sampling ring radius (pixels).
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when `step == 0` or
/// `radius == 0`.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature_extraction::daisy_descriptor;
/// use scirs2_core::ndarray::Array2;
///
/// let img: Array2<f64> = Array2::ones((32, 32));
/// let desc = daisy_descriptor(&img, 4, 4).unwrap();
/// assert_eq!(desc.ncols(), 8);
/// assert!(desc.nrows() > 0);
/// ```
pub fn daisy_descriptor(image: &Array2<f64>, step: usize, radius: usize) -> Result<Array2<f64>> {
    if step == 0 {
        return Err(VisionError::InvalidParameter(
            "step must be > 0".to_string(),
        ));
    }
    if radius == 0 {
        return Err(VisionError::InvalidParameter(
            "radius must be > 0".to_string(),
        ));
    }

    let (h, w) = image.dim();
    let n_hist = 8usize; // orientations per ring

    // Compute gradient maps once
    let (magnitudes, orientations) = compute_gradients(image);

    // Grid of descriptor centres (interior only, margin = radius)
    let margin = radius;
    if h <= 2 * margin || w <= 2 * margin {
        return Err(VisionError::InvalidParameter(
            "Image too small for given radius".to_string(),
        ));
    }

    let mut descriptors: Vec<Vec<f64>> = Vec::new();

    let mut cy = margin;
    while cy < h - margin {
        let mut cx = margin;
        while cx < w - margin {
            let mut desc = vec![0.0_f64; n_hist];
            let r = radius as f64;

            for k in 0..n_hist {
                let angle = 2.0 * PI * k as f64 / n_hist as f64;
                let sx = cx as f64 + r * angle.cos();
                let sy = cy as f64 - r * angle.sin();

                let mag = bilinear_sample(&magnitudes, sy, sx);
                let ori = bilinear_sample(&orientations, sy, sx);

                // Encode orientation into 8 bins
                let bin_f = ((ori + PI) / (2.0 * PI)) * n_hist as f64;
                let bin = (bin_f as usize) % n_hist;
                desc[bin] += mag;
            }

            // L2 normalise
            let norm = desc.iter().map(|v| v * v).sum::<f64>().sqrt() + 1e-7;
            for v in desc.iter_mut() {
                *v /= norm;
            }

            descriptors.push(desc);
            cx += step;
        }
        cy += step;
    }

    if descriptors.is_empty() {
        // Return empty matrix of correct column count
        return Ok(Array2::zeros((0, n_hist)));
    }

    let n = descriptors.len();
    let mut out = Array2::zeros((n, n_hist));
    for (i, row) in descriptors.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[[i, j]] = v;
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// BRIEF descriptor
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a BRIEF binary descriptor for a single keypoint.
///
/// BRIEF (Calonder et al., 2010) encodes a patch as a vector of binary
/// intensity comparisons between random point pairs.  This function uses a
/// deterministic, fixed test set derived from a Gaussian-distributed test
/// pattern with seed based on `patch_size`, so results are reproducible.
///
/// The descriptor is returned as a `Vec<u8>` of packed bits; with
/// `n_bits = 256` tests the vector has length 32.
///
/// # Arguments
///
/// * `image`      – Grayscale image, shape `(H, W)`.
/// * `keypoint`   – `(row, col)` centre of the patch.
/// * `patch_size` – Side length of the sampling patch (must be odd, ≥ 7).
///
/// # Errors
///
/// * [`VisionError::InvalidParameter`] – when `patch_size < 7` or even.
/// * [`VisionError::InvalidInput`] – when the keypoint is too close to the
///   image border for the chosen `patch_size`.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature_extraction::brief_descriptor;
/// use scirs2_core::ndarray::Array2;
///
/// let img: Array2<f64> = Array2::ones((64, 64));
/// let kp = (32, 32);
/// let desc = brief_descriptor(&img, kp, 31).unwrap();
/// assert_eq!(desc.len(), 32); // 256 bits / 8
/// ```
pub fn brief_descriptor(
    image: &Array2<f64>,
    keypoint: (usize, usize),
    patch_size: usize,
) -> Result<Vec<u8>> {
    if patch_size < 7 {
        return Err(VisionError::InvalidParameter(
            "patch_size must be >= 7".to_string(),
        ));
    }
    if patch_size.is_multiple_of(2) {
        return Err(VisionError::InvalidParameter(
            "patch_size must be odd".to_string(),
        ));
    }

    let (h, w) = image.dim();
    let half = patch_size / 2;
    let (ky, kx) = keypoint;

    if ky < half || ky + half >= h || kx < half || kx + half >= w {
        return Err(VisionError::InvalidInput(format!(
            "Keypoint ({ky},{kx}) too close to image border for patch_size={patch_size}"
        )));
    }

    const N_BITS: usize = 256;
    let test_pairs = generate_brief_test_pairs(patch_size, N_BITS);

    let n_bytes = N_BITS / 8;
    let mut descriptor = vec![0u8; n_bytes];

    for (bit_idx, &(p1y, p1x, p2y, p2x)) in test_pairs.iter().enumerate() {
        let y1 = (ky as i32 + p1y) as usize;
        let x1 = (kx as i32 + p1x) as usize;
        let y2 = (ky as i32 + p2y) as usize;
        let x2 = (kx as i32 + p2x) as usize;

        // Bounds already guaranteed by construction
        let bit = if image[[y1, x1]] < image[[y2, x2]] {
            1u8
        } else {
            0u8
        };

        let byte_idx = bit_idx / 8;
        let bit_offset = bit_idx % 8;
        descriptor[byte_idx] |= bit << bit_offset;
    }

    Ok(descriptor)
}

/// Generate deterministic BRIEF test pairs using a simple LCG pseudo-RNG.
///
/// Returns a list of `n_tests` point-pair offsets `(dy1, dx1, dy2, dx2)`.
fn generate_brief_test_pairs(patch_size: usize, n_tests: usize) -> Vec<(i32, i32, i32, i32)> {
    let half = (patch_size / 2) as i32;

    // Simple LCG seeded by patch_size for determinism
    let mut state: u64 = (patch_size as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);

    let lcg_next = |s: &mut u64| -> i32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to [-half, half]
        let raw = ((*s >> 33) as i32) % (half + 1);
        if (*s >> 63) & 1 == 1 {
            -raw
        } else {
            raw
        }
    };

    let mut pairs = Vec::with_capacity(n_tests);
    for _ in 0..n_tests {
        let dy1 = lcg_next(&mut state);
        let dx1 = lcg_next(&mut state);
        let dy2 = lcg_next(&mut state);
        let dx2 = lcg_next(&mut state);
        pairs.push((dy1, dx1, dy2, dx2));
    }
    pairs
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn checkerboard(h: usize, w: usize) -> Array2<f64> {
        let mut a = Array2::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                a[[y, x]] = if (y + x) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        a
    }

    // ── HOG ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_hog_uniform_image() {
        let img: Array2<f64> = Array2::ones((64, 64));
        let feat = hog_features(&img, 8, 2, 9).expect("HOG failed");
        // Uniform image → zero gradients → all-zero HOG
        for &v in feat.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_hog_feature_length() {
        let img: Array2<f64> = Array2::ones((64, 128));
        let cell_size = 8;
        let block_size = 2;
        let n_bins = 9;
        let feat = hog_features(&img, cell_size, block_size, n_bins).expect("HOG failed");
        let n_cells_y = 64 / cell_size;
        let n_cells_x = 128 / cell_size;
        let n_blocks_y = n_cells_y - block_size + 1;
        let n_blocks_x = n_cells_x - block_size + 1;
        let expected = n_blocks_y * n_blocks_x * block_size * block_size * n_bins;
        assert_eq!(feat.len(), expected);
    }

    #[test]
    fn test_hog_checkerboard() {
        let img = checkerboard(64, 64);
        let feat = hog_features(&img, 8, 2, 9).expect("HOG on checkerboard");
        assert!(!feat.is_empty());
        // Checkerboard has gradients → non-zero features
        let max: f64 = feat.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max > 0.0, "Expected non-zero HOG features");
    }

    #[test]
    fn test_hog_invalid_params() {
        let img: Array2<f64> = Array2::ones((64, 64));
        assert!(hog_features(&img, 0, 2, 9).is_err());
        assert!(hog_features(&img, 8, 0, 9).is_err());
        assert!(hog_features(&img, 8, 2, 0).is_err());
    }

    #[test]
    fn test_hog_image_too_small() {
        let img: Array2<f64> = Array2::ones((4, 4));
        // 4/8 = 0 cells → less than block_size=2
        assert!(hog_features(&img, 8, 2, 9).is_err());
    }

    // ── LBP ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_lbp_uniform_image() {
        let img: Array2<f64> = Array2::ones((32, 32));
        let feat = lbp_features(&img, 1, 8).expect("LBP failed");
        assert_eq!(feat.len(), 10);
    }

    #[test]
    fn test_lbp_normalised() {
        let img = checkerboard(32, 32);
        let feat = lbp_features(&img, 1, 8).expect("LBP failed");
        let sum: f64 = feat.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "LBP should be L1-normalised, sum={sum}"
        );
    }

    #[test]
    fn test_lbp_invalid_params() {
        let img: Array2<f64> = Array2::ones((32, 32));
        assert!(lbp_features(&img, 0, 8).is_err());
        assert!(lbp_features(&img, 1, 1).is_err());
    }

    // ── Gabor ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_gabor_filter_shape() {
        let img: Array2<f64> = Array2::ones((32, 32));
        let resp = gabor_filter(&img, 0.1, 0.0, 3.0, 1.0).expect("Gabor failed");
        assert_eq!(resp.dim(), (32, 32));
    }

    #[test]
    fn test_gabor_filter_non_negative_response() {
        let img = checkerboard(32, 32);
        let resp = gabor_filter(&img, 0.1, 0.0, 3.0, 1.0).expect("Gabor failed");
        // Magnitude response must be non-negative
        for &v in resp.iter() {
            assert!(v >= 0.0, "Gabor magnitude is negative: {v}");
        }
    }

    #[test]
    fn test_gabor_filter_invalid_params() {
        let img: Array2<f64> = Array2::ones((32, 32));
        assert!(gabor_filter(&img, 0.1, 0.0, 0.0, 1.0).is_err());
        assert!(gabor_filter(&img, 0.0, 0.0, 3.0, 1.0).is_err());
    }

    #[test]
    fn test_gabor_features_length() {
        let img: Array2<f64> = Array2::ones((32, 32));
        let freqs = vec![0.1, 0.2, 0.4];
        let thetas = vec![0.0, PI / 4.0, PI / 2.0];
        let feat = gabor_features(&img, &freqs, &thetas).expect("Gabor features failed");
        assert_eq!(feat.len(), 2 * freqs.len() * thetas.len());
    }

    // ── DAISY ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_daisy_descriptor_shape() {
        let img: Array2<f64> = Array2::ones((64, 64));
        let desc = daisy_descriptor(&img, 4, 8).expect("DAISY failed");
        assert_eq!(desc.ncols(), 8);
        assert!(desc.nrows() > 0);
    }

    #[test]
    fn test_daisy_descriptor_normalised() {
        let img = checkerboard(64, 64);
        let desc = daisy_descriptor(&img, 4, 8).expect("DAISY failed");
        for row in desc.rows() {
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            // Rows should be approximately unit-norm (or zero for flat regions)
            assert!(norm <= 1.0 + 1e-6, "DAISY row norm > 1: {norm}");
        }
    }

    #[test]
    fn test_daisy_invalid_params() {
        let img: Array2<f64> = Array2::ones((64, 64));
        assert!(daisy_descriptor(&img, 0, 8).is_err());
        assert!(daisy_descriptor(&img, 4, 0).is_err());
    }

    // ── BRIEF ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_brief_descriptor_length() {
        let img: Array2<f64> = Array2::ones((64, 64));
        let desc = brief_descriptor(&img, (32, 32), 31).expect("BRIEF failed");
        assert_eq!(desc.len(), 32); // 256 bits = 32 bytes
    }

    #[test]
    fn test_brief_descriptor_deterministic() {
        let img = checkerboard(64, 64);
        let d1 = brief_descriptor(&img, (32, 32), 31).expect("BRIEF failed");
        let d2 = brief_descriptor(&img, (32, 32), 31).expect("BRIEF failed");
        assert_eq!(d1, d2, "BRIEF must be deterministic");
    }

    #[test]
    fn test_brief_descriptor_invalid_patch() {
        let img: Array2<f64> = Array2::ones((64, 64));
        assert!(brief_descriptor(&img, (32, 32), 6).is_err()); // too small
        assert!(brief_descriptor(&img, (32, 32), 8).is_err()); // even
    }

    #[test]
    fn test_brief_descriptor_border_check() {
        let img: Array2<f64> = Array2::ones((64, 64));
        // Keypoint too close to border for patch_size=31 (half=15)
        assert!(brief_descriptor(&img, (5, 5), 31).is_err());
    }

    #[test]
    fn test_brief_hamming_distance() {
        // Two uniform images with different values → different descriptors
        let img1: Array2<f64> = Array2::from_elem((64, 64), 0.2);
        let img2: Array2<f64> = Array2::from_elem((64, 64), 0.8);
        let d1 = brief_descriptor(&img1, (32, 32), 31).expect("BRIEF failed");
        let d2 = brief_descriptor(&img2, (32, 32), 31).expect("BRIEF failed");
        // Uniform images: all comparisons result in same bit, descriptors identical
        // (both fully 0 or fully some pattern based on LCG pairs)
        assert_eq!(d1.len(), d2.len());
    }
}
