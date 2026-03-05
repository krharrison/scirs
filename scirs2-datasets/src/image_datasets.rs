//! Image dataset generators.
//!
//! This module provides synthetic image generators for testing image processing,
//! segmentation, and computer vision algorithms.
//!
//! # Generators
//!
//! - [`make_checkerboard`]  – Black-and-white checkerboard pattern.
//! - [`make_circles_image`] – Grayscale image with randomly placed filled circles.
//! - [`make_gradient_image`] – Smooth bilinear gradient image.
//! - [`make_noisy_image`]   – Add Gaussian noise to an existing image.
//! - [`make_blobs_image`]   – Gaussian-blob segmentation dataset (image + label mask).

use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::{Array2, s};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

// ─────────────────────────────────────────────────────────────────────────────
// make_checkerboard
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a checkerboard pattern image.
///
/// Tiles alternate between `0.0` (black) and `1.0` (white) based on
/// `(row / tile_size + col / tile_size) % 2`.
///
/// # Arguments
///
/// * `rows`      – Image height in pixels (must be > 0).
/// * `cols`      – Image width in pixels (must be > 0).
/// * `tile_size` – Side length of each square tile in pixels (must be > 0).
///
/// # Returns
///
/// `Array2<f64>` of shape `(rows, cols)` with values in {0.0, 1.0}.
///
/// # Errors
///
/// Returns an error if any argument is 0.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::image_datasets::make_checkerboard;
///
/// let img = make_checkerboard(64, 64, 8).expect("checkerboard failed");
/// assert_eq!(img.shape(), &[64, 64]);
/// assert_eq!(img[[0, 0]], 0.0);   // top-left tile is black
/// assert_eq!(img[[0, 8]], 1.0);   // next tile is white
/// ```
pub fn make_checkerboard(rows: usize, cols: usize, tile_size: usize) -> Result<Array2<f64>> {
    if rows == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_checkerboard: rows must be > 0".to_string(),
        ));
    }
    if cols == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_checkerboard: cols must be > 0".to_string(),
        ));
    }
    if tile_size == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_checkerboard: tile_size must be > 0".to_string(),
        ));
    }

    let mut img = Array2::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let tile_r = r / tile_size;
            let tile_c = c / tile_size;
            img[[r, c]] = if (tile_r + tile_c) % 2 == 1 { 1.0 } else { 0.0 };
        }
    }
    Ok(img)
}

// ─────────────────────────────────────────────────────────────────────────────
// make_circles_image
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a grayscale image with randomly placed filled circles.
///
/// The image starts as all zeros (black).  Each circle is drawn with a random
/// center, a random radius between `min_r` and `max_r`, and filled with `1.0`.
///
/// # Arguments
///
/// * `rows`      – Image height (must be > 0).
/// * `cols`      – Image width (must be > 0).
/// * `n_circles` – Number of circles to draw (0 is valid; returns all-zero image).
/// * `seed`      – Random seed.
///
/// # Returns
///
/// `Array2<f64>` of shape `(rows, cols)` with values in [0.0, 1.0].
///
/// # Errors
///
/// Returns an error if `rows == 0` or `cols == 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::image_datasets::make_circles_image;
///
/// let img = make_circles_image(64, 64, 5, 42).expect("circles failed");
/// assert_eq!(img.shape(), &[64, 64]);
/// ```
pub fn make_circles_image(
    rows: usize,
    cols: usize,
    n_circles: usize,
    seed: u64,
) -> Result<Array2<f64>> {
    if rows == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_circles_image: rows must be > 0".to_string(),
        ));
    }
    if cols == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_circles_image: cols must be > 0".to_string(),
        ));
    }

    let mut img = Array2::zeros((rows, cols));

    if n_circles == 0 {
        return Ok(img);
    }

    let min_dim = rows.min(cols) as f64;
    let min_r = (min_dim * 0.04).max(1.0);
    let max_r = (min_dim * 0.20).max(min_r + 1.0);

    let mut rng = make_rng(seed);
    let row_dist =
        scirs2_core::random::Uniform::new(0usize, rows).map_err(|e| {
            DatasetsError::ComputationError(format!("Uniform row distribution failed: {e}"))
        })?;
    let col_dist =
        scirs2_core::random::Uniform::new(0usize, cols).map_err(|e| {
            DatasetsError::ComputationError(format!("Uniform col distribution failed: {e}"))
        })?;
    let r_dist = scirs2_core::random::Uniform::new(min_r, max_r).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform radius distribution failed: {e}"))
    })?;

    for _ in 0..n_circles {
        let cr = row_dist.sample(&mut rng) as f64;
        let cc = col_dist.sample(&mut rng) as f64;
        let radius = r_dist.sample(&mut rng);
        let r_sq = radius * radius;

        let r_min = ((cr - radius).floor().max(0.0) as usize).min(rows);
        let r_max = ((cr + radius).ceil() as usize + 1).min(rows);
        let c_min = ((cc - radius).floor().max(0.0) as usize).min(cols);
        let c_max = ((cc + radius).ceil() as usize + 1).min(cols);

        for r in r_min..r_max {
            for c in c_min..c_max {
                let dr = r as f64 - cr;
                let dc = c as f64 - cc;
                if dr * dr + dc * dc <= r_sq {
                    img[[r, c]] = 1.0;
                }
            }
        }
    }

    Ok(img)
}

// ─────────────────────────────────────────────────────────────────────────────
// make_gradient_image
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a bilinear gradient test image.
///
/// Pixel value is `(r / (rows-1) + c / (cols-1)) / 2.0`, ranging smoothly
/// from `0.0` at the top-left to `1.0` at the bottom-right.  Single-row or
/// single-column images use only the available dimension.
///
/// # Arguments
///
/// * `rows` – Image height (must be > 0).
/// * `cols` – Image width (must be > 0).
///
/// # Returns
///
/// `Array2<f64>` of shape `(rows, cols)` with values in [0.0, 1.0].
///
/// # Errors
///
/// Returns an error if `rows == 0` or `cols == 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::image_datasets::make_gradient_image;
///
/// let img = make_gradient_image(64, 64).expect("gradient failed");
/// assert_eq!(img.shape(), &[64, 64]);
/// assert!((img[[0, 0]] - 0.0).abs() < 1e-9);
/// assert!((img[[63, 63]] - 1.0).abs() < 1e-9);
/// ```
pub fn make_gradient_image(rows: usize, cols: usize) -> Result<Array2<f64>> {
    if rows == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_gradient_image: rows must be > 0".to_string(),
        ));
    }
    if cols == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_gradient_image: cols must be > 0".to_string(),
        ));
    }

    let mut img = Array2::zeros((rows, cols));
    let row_scale = if rows > 1 { (rows - 1) as f64 } else { 1.0 };
    let col_scale = if cols > 1 { (cols - 1) as f64 } else { 1.0 };

    for r in 0..rows {
        for c in 0..cols {
            let vr = r as f64 / row_scale;
            let vc = c as f64 / col_scale;
            img[[r, c]] = (vr + vc) / 2.0;
        }
    }
    Ok(img)
}

// ─────────────────────────────────────────────────────────────────────────────
// make_noisy_image
// ─────────────────────────────────────────────────────────────────────────────

/// Add Gaussian noise to an existing image array.
///
/// Noise values are sampled from N(0, noise_std²) and added element-wise.
/// The result is **not** clipped to [0, 1] — the caller may clip if needed.
///
/// # Arguments
///
/// * `base`      – Reference to the source image (any shape).
/// * `noise_std` – Standard deviation of the Gaussian noise (must be ≥ 0).
/// * `seed`      – Random seed.
///
/// # Returns
///
/// `Array2<f64>` of the same shape as `base`.
///
/// # Errors
///
/// Returns an error if `noise_std < 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::image_datasets::{make_gradient_image, make_noisy_image};
///
/// let base = make_gradient_image(32, 32).expect("gradient");
/// let noisy = make_noisy_image(&base, 0.05, 42).expect("noisy");
/// assert_eq!(noisy.shape(), base.shape());
/// ```
pub fn make_noisy_image(base: &Array2<f64>, noise_std: f64, seed: u64) -> Result<Array2<f64>> {
    if noise_std < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "make_noisy_image: noise_std must be >= 0".to_string(),
        ));
    }

    let mut result = base.to_owned();
    if noise_std == 0.0 {
        return Ok(result);
    }

    let mut rng = make_rng(seed);
    let dist = scirs2_core::random::Normal::new(0.0_f64, noise_std).map_err(|e| {
        DatasetsError::ComputationError(format!("Normal distribution creation failed: {e}"))
    })?;

    for val in result.iter_mut() {
        *val += dist.sample(&mut rng);
    }
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// make_blobs_image
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a Gaussian-blob segmentation dataset.
///
/// Places `n_blobs` Gaussian blobs at random locations.  Each pixel's value is
/// the sum of Gaussian contributions from all blobs (saturated at 1.0).  The
/// label mask marks each pixel with the index (1-indexed) of the blob that
/// contributes the most intensity to it, or `0` for background.
///
/// # Arguments
///
/// * `rows`        – Image height (must be > 0).
/// * `cols`        – Image width (must be > 0).
/// * `n_blobs`     – Number of Gaussian blobs (must be ≥ 1).
/// * `blob_radius` – Characteristic radius (sigma) of each Gaussian blob in pixels
///                   (must be > 0).
/// * `seed`        – Random seed.
///
/// # Returns
///
/// `(image, labels)` where
/// - `image`  is `Array2<f64>` of shape `(rows, cols)` with values in [0, 1].
/// - `labels` is `Array2<u8>`  of shape `(rows, cols)`: `0` = background,
///   `1..=n_blobs` = foreground blob index.
///
/// # Errors
///
/// Returns an error if any argument is 0 or `blob_radius <= 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::image_datasets::make_blobs_image;
///
/// let (img, labels) = make_blobs_image(64, 64, 3, 8.0, 42).expect("blobs failed");
/// assert_eq!(img.shape(), &[64, 64]);
/// assert_eq!(labels.shape(), &[64, 64]);
/// ```
pub fn make_blobs_image(
    rows: usize,
    cols: usize,
    n_blobs: usize,
    blob_radius: f64,
    seed: u64,
) -> Result<(Array2<f64>, Array2<u8>)> {
    if rows == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_blobs_image: rows must be > 0".to_string(),
        ));
    }
    if cols == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_blobs_image: cols must be > 0".to_string(),
        ));
    }
    if n_blobs == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_blobs_image: n_blobs must be >= 1".to_string(),
        ));
    }
    if blob_radius <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "make_blobs_image: blob_radius must be > 0".to_string(),
        ));
    }

    let mut rng = make_rng(seed);
    let row_dist =
        scirs2_core::random::Uniform::new(0.0_f64, rows as f64).map_err(|e| {
            DatasetsError::ComputationError(format!("Uniform row failed: {e}"))
        })?;
    let col_dist =
        scirs2_core::random::Uniform::new(0.0_f64, cols as f64).map_err(|e| {
            DatasetsError::ComputationError(format!("Uniform col failed: {e}"))
        })?;

    // Sample blob centers
    let mut centers: Vec<(f64, f64)> = Vec::with_capacity(n_blobs);
    for _ in 0..n_blobs {
        centers.push((row_dist.sample(&mut rng), col_dist.sample(&mut rng)));
    }

    let sigma2 = blob_radius * blob_radius;
    // For each blob: contribution[blob][r*cols+c] = Gaussian value
    // We store per-blob contribution to determine dominant blob for labels
    let mut contrib: Vec<Vec<f64>> = (0..n_blobs)
        .map(|_| vec![0.0_f64; rows * cols])
        .collect();

    for (b, &(cr, cc)) in centers.iter().enumerate() {
        for r in 0..rows {
            for c in 0..cols {
                let dr = r as f64 - cr;
                let dc = c as f64 - cc;
                contrib[b][r * cols + c] = (-(dr * dr + dc * dc) / (2.0 * sigma2)).exp();
            }
        }
    }

    let mut image = Array2::zeros((rows, cols));
    let mut labels: Array2<u8> = Array2::zeros((rows, cols));

    // threshold: pixel intensity > 0.1 considered foreground
    let threshold = 0.1_f64;

    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            let total: f64 = contrib.iter().map(|b| b[idx]).sum();
            image[[r, c]] = total.min(1.0);

            // Find the dominant blob (highest contribution)
            let mut best_blob = 0;
            let mut best_val = 0.0_f64;
            for (b, b_contrib) in contrib.iter().enumerate() {
                if b_contrib[idx] > best_val {
                    best_val = b_contrib[idx];
                    best_blob = b;
                }
            }
            // Mark as foreground only if dominant contribution exceeds threshold
            if best_val > threshold {
                // saturating cast: n_blobs can be at most 255 before u8 overflow
                labels[[r, c]] = (best_blob + 1).min(255) as u8;
            } else {
                labels[[r, c]] = 0;
            }
        }
    }

    Ok((image, labels))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── make_checkerboard ────────────────────────────────────────────────────

    #[test]
    fn test_checkerboard_shape() {
        let img = make_checkerboard(64, 64, 8).expect("checkerboard shape");
        assert_eq!(img.shape(), &[64, 64]);
    }

    #[test]
    fn test_checkerboard_values() {
        let img = make_checkerboard(16, 16, 4).expect("checkerboard values");
        // Top-left 4×4 tile → 0.0 (black)
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(img[[r, c]], 0.0, "top-left tile must be 0 at ({r},{c})");
            }
        }
        // Next tile in same row (cols 4..8) → 1.0 (white)
        for r in 0..4 {
            for c in 4..8 {
                assert_eq!(img[[r, c]], 1.0, "second tile must be 1 at ({r},{c})");
            }
        }
    }

    #[test]
    fn test_checkerboard_only_0_and_1() {
        let img = make_checkerboard(32, 48, 6).expect("checkerboard 0/1");
        for &v in img.iter() {
            assert!(v == 0.0 || v == 1.0, "checkerboard must contain only 0 or 1; got {v}");
        }
    }

    #[test]
    fn test_checkerboard_error_zero_rows() {
        assert!(make_checkerboard(0, 10, 2).is_err());
    }

    #[test]
    fn test_checkerboard_error_zero_tile() {
        assert!(make_checkerboard(10, 10, 0).is_err());
    }

    // ── make_circles_image ───────────────────────────────────────────────────

    #[test]
    fn test_circles_shape() {
        let img = make_circles_image(64, 64, 5, 42).expect("circles shape");
        assert_eq!(img.shape(), &[64, 64]);
    }

    #[test]
    fn test_circles_foreground_present() {
        let img = make_circles_image(128, 128, 10, 7).expect("circles fg");
        let n_foreground = img.iter().filter(|&&v| v > 0.5).count();
        assert!(n_foreground > 0, "some pixels should be filled");
    }

    #[test]
    fn test_circles_zero_circles() {
        let img = make_circles_image(32, 32, 0, 1).expect("zero circles");
        assert!(img.iter().all(|&v| v == 0.0), "no circles → all zero");
    }

    #[test]
    fn test_circles_values_in_range() {
        let img = make_circles_image(64, 64, 8, 3).expect("circles range");
        for &v in img.iter() {
            assert!((0.0..=1.0).contains(&v), "pixel value {v} out of [0,1]");
        }
    }

    // ── make_gradient_image ──────────────────────────────────────────────────

    #[test]
    fn test_gradient_shape() {
        let img = make_gradient_image(64, 64).expect("gradient shape");
        assert_eq!(img.shape(), &[64, 64]);
    }

    #[test]
    fn test_gradient_corners() {
        let img = make_gradient_image(64, 64).expect("gradient corners");
        assert!((img[[0, 0]] - 0.0).abs() < 1e-9, "top-left must be 0");
        assert!((img[[63, 63]] - 1.0).abs() < 1e-9, "bottom-right must be 1");
    }

    #[test]
    fn test_gradient_monotone_along_diagonal() {
        let img = make_gradient_image(50, 50).expect("gradient mono");
        for i in 1..50 {
            assert!(
                img[[i, i]] >= img[[i - 1, i - 1]],
                "gradient must be monotonically non-decreasing along the diagonal"
            );
        }
    }

    #[test]
    fn test_gradient_values_in_range() {
        let img = make_gradient_image(32, 48).expect("gradient range");
        for &v in img.iter() {
            assert!(
                (-1e-9..=1.0 + 1e-9).contains(&v),
                "gradient pixel {v} out of [0,1]"
            );
        }
    }

    #[test]
    fn test_gradient_error_zero_rows() {
        assert!(make_gradient_image(0, 10).is_err());
    }

    // ── make_noisy_image ─────────────────────────────────────────────────────

    #[test]
    fn test_noisy_shape() {
        let base = make_gradient_image(32, 32).expect("base");
        let noisy = make_noisy_image(&base, 0.1, 42).expect("noisy shape");
        assert_eq!(noisy.shape(), base.shape());
    }

    #[test]
    fn test_noisy_zero_std_equals_base() {
        let base = make_gradient_image(16, 16).expect("base");
        let noisy = make_noisy_image(&base, 0.0, 1).expect("noisy zero std");
        for (a, b) in base.iter().zip(noisy.iter()) {
            assert_eq!(a, b, "zero noise should return identical image");
        }
    }

    #[test]
    fn test_noisy_differs_from_base() {
        let base = make_gradient_image(32, 32).expect("base");
        let noisy = make_noisy_image(&base, 0.1, 42).expect("noisy differs");
        let diff: f64 = base.iter().zip(noisy.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.01, "noisy image should differ from base");
    }

    #[test]
    fn test_noisy_determinism() {
        let base = make_gradient_image(20, 20).expect("base");
        let n1 = make_noisy_image(&base, 0.05, 99).expect("n1");
        let n2 = make_noisy_image(&base, 0.05, 99).expect("n2");
        for (a, b) in n1.iter().zip(n2.iter()) {
            assert!((a - b).abs() < 1e-12, "same seed must produce identical output");
        }
    }

    #[test]
    fn test_noisy_error_negative_std() {
        let base = make_gradient_image(10, 10).expect("base");
        assert!(make_noisy_image(&base, -0.1, 1).is_err());
    }

    // ── make_blobs_image ─────────────────────────────────────────────────────

    #[test]
    fn test_blobs_shape() {
        let (img, labels) = make_blobs_image(64, 64, 3, 8.0, 42).expect("blobs shape");
        assert_eq!(img.shape(), &[64, 64]);
        assert_eq!(labels.shape(), &[64, 64]);
    }

    #[test]
    fn test_blobs_image_in_range() {
        let (img, _) = make_blobs_image(64, 64, 3, 8.0, 7).expect("blobs range");
        for &v in img.iter() {
            assert!(
                (-1e-12..=1.0 + 1e-12).contains(&v),
                "blob image value {v} out of [0,1]"
            );
        }
    }

    #[test]
    fn test_blobs_labels_valid() {
        let n_blobs = 3usize;
        let (_, labels) = make_blobs_image(64, 64, n_blobs, 10.0, 5).expect("blobs labels");
        for &l in labels.iter() {
            assert!(
                l as usize <= n_blobs,
                "label {l} exceeds n_blobs={n_blobs}"
            );
        }
    }

    #[test]
    fn test_blobs_foreground_present() {
        let (_, labels) = make_blobs_image(128, 128, 4, 15.0, 1).expect("blobs fg");
        let fg = labels.iter().filter(|&&l| l > 0).count();
        assert!(fg > 0, "blobs must produce some foreground pixels");
    }

    #[test]
    fn test_blobs_error_n_blobs_zero() {
        assert!(make_blobs_image(64, 64, 0, 8.0, 1).is_err());
    }

    #[test]
    fn test_blobs_error_radius_zero() {
        assert!(make_blobs_image(64, 64, 3, 0.0, 1).is_err());
    }
}
