//! Retinal image analysis: vessel segmentation, optic disc detection, and preprocessing.
//!
//! This module provides:
//!
//! - [`detect_optic_disc`] – bright circular region detection for optic disc localisation
//! - [`preprocess_retinal`] – CLAHE-based contrast enhancement
//! - [`frangi_vesselness`] – multi-scale Frangi vessel enhancement filter
//! - [`measure_layer_thickness`] – OCT layer thickness estimation
//! - [`detect_drusen_approximate`] – AMD drusen detection via background subtraction

use crate::error::VisionError;
use crate::medical::cell_analysis::gaussian_blur_2d;
use scirs2_core::ndarray::{Array2, ArrayView2};

// ── Optic disc detection ──────────────────────────────────────────────────────

/// Detect the optic disc as the brightest large circular region in a retinal image.
///
/// The algorithm:
/// 1. Blur the image with a coarse Gaussian to suppress vessels and noise.
/// 2. Find the global maximum as a candidate disc centre.
/// 3. Estimate the radius from the region of pixels above a brightness threshold.
///
/// # Returns
///
/// `Ok(Some((row, col, radius)))` if a disc-like region is found, `Ok(None)` otherwise.
///
/// # Errors
///
/// Returns [`VisionError::OperationError`] for empty images.
pub fn detect_optic_disc(
    retinal_image: ArrayView2<f64>,
) -> Result<Option<(f64, f64, f64)>, VisionError> {
    let (rows, cols) = retinal_image.dim();
    if rows == 0 || cols == 0 {
        return Err(VisionError::OperationError("Empty image".to_string()));
    }

    // Smooth to suppress fine detail
    let sigma = (rows.min(cols) as f64 * 0.02).max(2.0);
    let smoothed = gaussian_blur_2d(retinal_image, sigma);

    // Find the brightest pixel
    let mut max_val = f64::NEG_INFINITY;
    let mut max_r = 0usize;
    let mut max_c = 0usize;
    for r in 0..rows {
        for c in 0..cols {
            if smoothed[[r, c]] > max_val {
                max_val = smoothed[[r, c]];
                max_r = r;
                max_c = c;
            }
        }
    }

    if max_val <= 0.0 {
        return Ok(None);
    }

    // Disc threshold: 70% of the maximum value
    let threshold = max_val * 0.70;
    let disc_pixels: usize = smoothed.iter().filter(|&&v| v >= threshold).count();
    if disc_pixels == 0 {
        return Ok(None);
    }

    // Approximate radius from area
    let radius = (disc_pixels as f64 / std::f64::consts::PI).sqrt();
    // Clamp to a sensible fraction of image size
    let max_radius = rows.min(cols) as f64 * 0.25;
    let radius = radius.min(max_radius).max(1.0);

    Ok(Some((max_r as f64, max_c as f64, radius)))
}

// ── Retinal preprocessing (CLAHE) ────────────────────────────────────────────

/// Preprocess a retinal image using Contrast-Limited Adaptive Histogram Equalisation (CLAHE).
///
/// CLAHE divides the image into non-overlapping tiles of size `tile_size` and applies
/// histogram equalisation within each tile, clipping the histogram at `clip_limit`
/// to prevent over-amplification of noise.  Bilinear interpolation blends tile borders.
///
/// # Arguments
///
/// * `image`     – input image in \[0, 1\] (values are clamped before processing)
/// * `clip_limit` – histogram clip limit (1.0 = no clipping; typical range 2–4)
/// * `tile_size` – `(tile_rows, tile_cols)` in pixels
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] for invalid tile sizes or an empty image.
pub fn preprocess_retinal(
    image: ArrayView2<f64>,
    clip_limit: f64,
    tile_size: (usize, usize),
) -> Result<Array2<f64>, VisionError> {
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(VisionError::InvalidParameter("Empty image".to_string()));
    }
    let (th, tw) = tile_size;
    if th == 0 || tw == 0 {
        return Err(VisionError::InvalidParameter(
            "tile_size dimensions must be > 0".to_string(),
        ));
    }
    if clip_limit < 1.0 {
        return Err(VisionError::InvalidParameter(
            "clip_limit must be >= 1.0".to_string(),
        ));
    }

    const N_BINS: usize = 256;

    // Number of tiles in each dimension
    let n_tiles_r = rows.div_ceil(th);
    let n_tiles_c = cols.div_ceil(tw);

    // Build a CLimited equalised CDF for every tile
    // tile_cdfs[tile_r][tile_c] = [N_BINS] mapping bin → [0,1]
    let mut tile_cdfs: Vec<Vec<[f64; N_BINS]>> =
        vec![vec![[0.0_f64; N_BINS]; n_tiles_c]; n_tiles_r];

    #[allow(clippy::needless_range_loop)]
    for tr in 0..n_tiles_r {
        for tc in 0..n_tiles_c {
            let r0 = tr * th;
            let c0 = tc * tw;
            let r1 = (r0 + th).min(rows);
            let c1 = (c0 + tw).min(cols);
            let tile_n = (r1 - r0) * (c1 - c0);

            // Build histogram
            let mut hist = [0usize; N_BINS];
            for r in r0..r1 {
                for c in c0..c1 {
                    let v = image[[r, c]].clamp(0.0, 1.0);
                    let bin = ((v * (N_BINS - 1) as f64).round() as usize).min(N_BINS - 1);
                    hist[bin] += 1;
                }
            }

            // Clip and redistribute
            let clip = ((clip_limit * tile_n as f64 / N_BINS as f64).ceil() as usize).max(1);
            let mut excess = 0usize;
            for bin_val in hist.iter_mut().take(N_BINS) {
                if *bin_val > clip {
                    excess += *bin_val - clip;
                    *bin_val = clip;
                }
            }
            let redistrib = excess / N_BINS;
            for bin_val in hist.iter_mut().take(N_BINS) {
                *bin_val += redistrib;
            }

            // Build CDF
            let mut cdf = [0.0_f64; N_BINS];
            let mut running = 0usize;
            for b in 0..N_BINS {
                running += hist[b];
                cdf[b] = running as f64 / tile_n as f64;
            }

            tile_cdfs[tr][tc] = cdf;
        }
    }

    // Apply bilinear interpolation of CDFs
    let mut out = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let v = image[[r, c]].clamp(0.0, 1.0);
            let bin = ((v * (N_BINS - 1) as f64).round() as usize).min(N_BINS - 1);

            // Tile coordinates (fractional)
            let tf_r = (r as f64 + 0.5) / th as f64 - 0.5;
            let tf_c = (c as f64 + 0.5) / tw as f64 - 0.5;
            let tr0 = (tf_r.floor() as isize).clamp(0, n_tiles_r as isize - 1) as usize;
            let tc0 = (tf_c.floor() as isize).clamp(0, n_tiles_c as isize - 1) as usize;
            let tr1 = (tr0 + 1).min(n_tiles_r - 1);
            let tc1 = (tc0 + 1).min(n_tiles_c - 1);
            let wr = (tf_r - tr0 as f64).clamp(0.0, 1.0);
            let wc = (tf_c - tc0 as f64).clamp(0.0, 1.0);

            let v00 = tile_cdfs[tr0][tc0][bin];
            let v01 = tile_cdfs[tr0][tc1][bin];
            let v10 = tile_cdfs[tr1][tc0][bin];
            let v11 = tile_cdfs[tr1][tc1][bin];

            let interp =
                (1.0 - wr) * ((1.0 - wc) * v00 + wc * v01) + wr * ((1.0 - wc) * v10 + wc * v11);
            out[[r, c]] = interp.clamp(0.0, 1.0);
        }
    }

    Ok(out)
}

// ── Frangi vesselness filter ──────────────────────────────────────────────────

/// Apply the Frangi multi-scale vessel enhancement filter.
///
/// For each scale σ in `sigmas`:
/// 1. Compute the 2-D Hessian matrix `H` at each pixel via second-order
///    Gaussian derivatives.
/// 2. Derive eigenvalues λ₁ ≤ λ₂ of `H`.
/// 3. Compute the vesselness response:
///    - `R_B = λ₁² / λ₂²` (blobness)
///    - `S = √(λ₁² + λ₂²)` (structure measure)
///    - `V = exp(-R_B/(2β²)) × (1 − exp(-S²/(2c²)))` if `λ₂ < 0`, else 0
/// 4. Take the maximum response over all scales.
///
/// # Arguments
///
/// * `image`  – input image (should be normalised to a consistent intensity range)
/// * `sigmas` – set of Gaussian smoothing scales (σ values in pixels)
/// * `beta`   – Frangi β parameter controlling blobness sensitivity (typical 0.5)
/// * `c`      – Frangi c parameter controlling background suppression
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] if `sigmas` is empty or contains non-positive values.
pub fn frangi_vesselness(
    image: ArrayView2<f64>,
    sigmas: &[f64],
    beta: f64,
    c: f64,
) -> Result<Array2<f64>, VisionError> {
    if sigmas.is_empty() {
        return Err(VisionError::InvalidParameter(
            "sigmas must not be empty".to_string(),
        ));
    }
    for &s in sigmas {
        if s <= 0.0 {
            return Err(VisionError::InvalidParameter(format!(
                "all sigmas must be positive; found {s}"
            )));
        }
    }
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Ok(Array2::zeros((rows, cols)));
    }

    let mut max_response = Array2::<f64>::zeros((rows, cols));

    for &sigma in sigmas {
        let (lam1, lam2) = hessian_eigenvalues(image, sigma);

        for r in 0..rows {
            for col in 0..cols {
                let l1 = lam1[[r, col]];
                let l2 = lam2[[r, col]];
                // For bright-on-dark tubular structures: λ₂ < 0
                if l2 >= 0.0 {
                    continue;
                }
                let rb = if l2.abs() > 1e-12 {
                    (l1 / l2).powi(2)
                } else {
                    0.0
                };
                let s_sq = l1 * l1 + l2 * l2;
                let two_beta_sq = 2.0 * beta * beta;
                let two_c_sq = 2.0 * c * c;
                let v = (-rb / two_beta_sq).exp() * (1.0 - (-s_sq / two_c_sq).exp());
                if v > max_response[[r, col]] {
                    max_response[[r, col]] = v;
                }
            }
        }
    }

    Ok(max_response)
}

// ── OCT layer thickness measurement ──────────────────────────────────────────

/// Estimate retinal layer thickness from an OCT-like image.
///
/// For each column, the function finds the first and last row above `threshold`
/// and returns `last_row − first_row` as the thickness.  Returns 0 for columns
/// with no supra-threshold pixels.
///
/// # Errors
///
/// Returns [`VisionError::OperationError`] for empty images.
pub fn measure_layer_thickness(
    image: ArrayView2<f64>,
    threshold: f64,
) -> Result<Array2<f64>, VisionError> {
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(VisionError::OperationError("Empty image".to_string()));
    }

    // Output: one row, `cols` columns — thickness per A-scan column
    let mut thickness = Array2::<f64>::zeros((1, cols));
    for c in 0..cols {
        let mut first = None;
        let mut last = None;
        for r in 0..rows {
            if image[[r, c]] > threshold {
                if first.is_none() {
                    first = Some(r);
                }
                last = Some(r);
            }
        }
        if let (Some(f), Some(l)) = (first, last) {
            thickness[[0, c]] = (l - f) as f64;
        }
    }
    Ok(thickness)
}

// ── Drusen detection ──────────────────────────────────────────────────────────

/// Detect drusen (bright deposits associated with AMD) by background subtraction.
///
/// The algorithm:
/// 1. Estimate a smooth background by Gaussian blurring with `background_sigma`.
/// 2. Subtract the background to isolate local bright spots.
/// 3. Threshold the residual to produce a binary drusen mask.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] for non-positive `background_sigma` or empty image.
pub fn detect_drusen_approximate(
    retinal_image: ArrayView2<f64>,
    background_sigma: f64,
) -> Result<Array2<bool>, VisionError> {
    let (rows, cols) = retinal_image.dim();
    if rows == 0 || cols == 0 {
        return Err(VisionError::InvalidParameter("Empty image".to_string()));
    }
    if background_sigma <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "background_sigma must be positive".to_string(),
        ));
    }

    let background = gaussian_blur_2d(retinal_image, background_sigma);
    // Residual = image − background (drusen appear as local positive peaks)
    let residual = retinal_image.to_owned() - &background;

    // Adaptive threshold: mean + 1.5 std of positive residuals
    let pos_residuals: Vec<f64> = residual.iter().copied().filter(|&v| v > 0.0).collect();
    if pos_residuals.is_empty() {
        return Ok(Array2::from_elem((rows, cols), false));
    }
    let mean_res = pos_residuals.iter().sum::<f64>() / pos_residuals.len() as f64;
    let std_res = (pos_residuals
        .iter()
        .map(|v| (v - mean_res).powi(2))
        .sum::<f64>()
        / pos_residuals.len() as f64)
        .sqrt();
    let drusen_thresh = mean_res + 1.5 * std_res;

    Ok(residual.mapv(|v| v > drusen_thresh))
}

// ── Hessian eigenvalue computation ───────────────────────────────────────────

/// Compute the two eigenvalues of the 2-D Hessian at every pixel for a given scale σ.
///
/// Returns `(lambda1, lambda2)` where `|lambda1| ≤ |lambda2|` (ordered by magnitude).
fn hessian_eigenvalues(image: ArrayView2<f64>, sigma: f64) -> (Array2<f64>, Array2<f64>) {
    let (rows, cols) = image.dim();

    // Second-order Gaussian derivative kernels (1-D)
    let radius = (3.0 * sigma).ceil() as usize;
    let gauss: Vec<f64> = {
        let mut k: Vec<f64> = (-(radius as i64)..=radius as i64)
            .map(|x| (-(x as f64).powi(2) / (2.0 * sigma * sigma)).exp())
            .collect();
        let s: f64 = k.iter().sum();
        k.iter_mut().for_each(|v| *v /= s);
        k
    };
    let gauss2: Vec<f64> = {
        let mut k: Vec<f64> = (-(radius as i64)..=radius as i64)
            .map(|x| {
                let xf = x as f64;
                ((xf * xf / (sigma * sigma) - 1.0) * (-(xf * xf) / (2.0 * sigma * sigma)).exp())
                    / (sigma * sigma)
            })
            .collect();
        // Normalise to zero DC (second derivative integrates to 0 over ℝ)
        k
    };
    let gauss1: Vec<f64> = {
        (-(radius as i64)..=radius as i64)
            .map(|x| {
                let xf = x as f64;
                -xf / (sigma * sigma * sigma * (2.0 * std::f64::consts::PI).sqrt())
                    * (-(xf * xf) / (2.0 * sigma * sigma)).exp()
            })
            .collect()
    };

    // Scale-normalised: σ² × D²G
    let scale = sigma * sigma;

    let conv_row = |kern: &[f64], row: usize| -> Vec<f64> {
        (0..cols)
            .map(|c| {
                kern.iter()
                    .enumerate()
                    .map(|(ki, &kv)| {
                        let offset = ki as i64 - radius as i64;
                        let nc = (c as i64 + offset).clamp(0, cols as i64 - 1) as usize;
                        image[[row, nc]] * kv
                    })
                    .sum::<f64>()
            })
            .collect()
    };

    let conv_col_from = |input: &Array2<f64>, kern: &[f64], col: usize| -> Vec<f64> {
        (0..rows)
            .map(|r| {
                kern.iter()
                    .enumerate()
                    .map(|(ki, &kv)| {
                        let offset = ki as i64 - radius as i64;
                        let nr = (r as i64 + offset).clamp(0, rows as i64 - 1) as usize;
                        input[[nr, col]] * kv
                    })
                    .sum::<f64>()
            })
            .collect()
    };

    // Hxx: D2_r ⊗ G_c
    let mut hxx_inter = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        let row_d2 = conv_row(&gauss2, r);
        for c in 0..cols {
            hxx_inter[[r, c]] = row_d2[c];
        }
    }
    let mut hxx = Array2::<f64>::zeros((rows, cols));
    for c in 0..cols {
        let col_g = conv_col_from(&hxx_inter, &gauss, c);
        for r in 0..rows {
            hxx[[r, c]] = col_g[r] * scale;
        }
    }

    // Hyy: G_r ⊗ D2_c
    let mut hyy_inter = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        let row_g = conv_row(&gauss, r);
        for c in 0..cols {
            hyy_inter[[r, c]] = row_g[c];
        }
    }
    let mut hyy = Array2::<f64>::zeros((rows, cols));
    for c in 0..cols {
        let col_d2 = conv_col_from(&hyy_inter, &gauss2, c);
        for r in 0..rows {
            hyy[[r, c]] = col_d2[r] * scale;
        }
    }

    // Hxy: D1_r ⊗ D1_c
    let mut hxy_inter = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        let row_d1 = conv_row(&gauss1, r);
        for c in 0..cols {
            hxy_inter[[r, c]] = row_d1[c];
        }
    }
    let mut hxy = Array2::<f64>::zeros((rows, cols));
    for c in 0..cols {
        let col_d1 = conv_col_from(&hxy_inter, &gauss1, c);
        for r in 0..rows {
            hxy[[r, c]] = col_d1[r] * scale;
        }
    }

    // Compute eigenvalues of 2×2 symmetric matrix [Hxx Hxy; Hxy Hyy]
    // λ = ((Hxx+Hyy) ± √((Hxx-Hyy)²+4Hxy²)) / 2
    let mut lam1 = Array2::<f64>::zeros((rows, cols));
    let mut lam2 = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let a = hxx[[r, c]];
            let d = hyy[[r, c]];
            let b = hxy[[r, c]];
            let trace = a + d;
            let disc = ((a - d).powi(2) + 4.0 * b * b).sqrt();
            let ev1 = (trace - disc) / 2.0;
            let ev2 = (trace + disc) / 2.0;
            // Order by magnitude
            if ev1.abs() <= ev2.abs() {
                lam1[[r, c]] = ev1;
                lam2[[r, c]] = ev2;
            } else {
                lam1[[r, c]] = ev2;
                lam2[[r, c]] = ev1;
            }
        }
    }
    (lam1, lam2)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    // ── detect_optic_disc ────────────────────────────────────────────────────

    #[test]
    fn test_detect_optic_disc_empty_error() {
        let img: Array2<f64> = Array2::zeros((0, 0));
        assert!(detect_optic_disc(img.view()).is_err());
    }

    #[test]
    fn test_detect_optic_disc_dark_image_no_disc() {
        let img = Array2::from_elem((20, 20), 0.0_f64);
        let result = detect_optic_disc(img.view()).expect("Should succeed");
        assert!(result.is_none(), "Uniform dark image → no disc");
    }

    #[test]
    fn test_detect_optic_disc_bright_region() {
        // Large bright region centred at (15, 15) in a 30×30 image
        let mut img = Array2::<f64>::zeros((30, 30));
        for r in 10..20 {
            for c in 10..20 {
                img[[r, c]] = 1.0;
            }
        }
        let result = detect_optic_disc(img.view()).expect("Should succeed");
        assert!(result.is_some());
        let (row, col, radius) = result.expect("Has disc");
        assert!(radius > 0.0, "radius must be positive");
        // Centre should be roughly in the bright region
        assert!((5.0..25.0).contains(&row), "row={row}");
        assert!((5.0..25.0).contains(&col), "col={col}");
    }

    // ── preprocess_retinal ───────────────────────────────────────────────────

    #[test]
    fn test_preprocess_retinal_empty_error() {
        let img: Array2<f64> = Array2::zeros((0, 0));
        assert!(preprocess_retinal(img.view(), 2.0, (8, 8)).is_err());
    }

    #[test]
    fn test_preprocess_retinal_zero_tile_error() {
        let img = Array2::from_elem((10, 10), 0.5_f64);
        assert!(preprocess_retinal(img.view(), 2.0, (0, 8)).is_err());
        assert!(preprocess_retinal(img.view(), 2.0, (8, 0)).is_err());
    }

    #[test]
    fn test_preprocess_retinal_invalid_clip_limit() {
        let img = Array2::from_elem((10, 10), 0.5_f64);
        assert!(preprocess_retinal(img.view(), 0.5, (4, 4)).is_err());
    }

    #[test]
    fn test_preprocess_retinal_output_range() {
        let mut img = Array2::<f64>::zeros((16, 16));
        // Gradient image
        for r in 0..16 {
            for c in 0..16 {
                img[[r, c]] = (r * 16 + c) as f64 / 256.0;
            }
        }
        let out = preprocess_retinal(img.view(), 2.0, (4, 4)).expect("Should succeed");
        assert_eq!(out.dim(), (16, 16));
        for &v in out.iter() {
            assert!((0.0..=1.0).contains(&v), "output out of range: {v}");
        }
    }

    #[test]
    fn test_preprocess_retinal_uniform_image() {
        // Uniform image should map all pixels to a single value
        let img = Array2::from_elem((8, 8), 0.5_f64);
        let out = preprocess_retinal(img.view(), 2.0, (4, 4)).expect("Should succeed");
        let first = out[[0, 0]];
        for &v in out.iter() {
            assert!((v - first).abs() < 1e-6, "uniform input → uniform output");
        }
    }

    // ── frangi_vesselness ────────────────────────────────────────────────────

    #[test]
    fn test_frangi_vesselness_empty_sigmas_error() {
        let img = Array2::from_elem((10, 10), 0.5_f64);
        assert!(frangi_vesselness(img.view(), &[], 0.5, 0.15).is_err());
    }

    #[test]
    fn test_frangi_vesselness_non_positive_sigma_error() {
        let img = Array2::from_elem((10, 10), 0.5_f64);
        assert!(frangi_vesselness(img.view(), &[1.0, 0.0], 0.5, 0.15).is_err());
        assert!(frangi_vesselness(img.view(), &[-1.0], 0.5, 0.15).is_err());
    }

    #[test]
    fn test_frangi_vesselness_output_shape() {
        let img = Array2::from_elem((20, 20), 0.5_f64);
        let out = frangi_vesselness(img.view(), &[1.0, 2.0], 0.5, 0.15).expect("Should succeed");
        assert_eq!(out.dim(), (20, 20));
    }

    #[test]
    fn test_frangi_vesselness_non_negative_output() {
        let img = Array2::from_elem((20, 20), 0.3_f64);
        let out = frangi_vesselness(img.view(), &[1.0], 0.5, 0.15).expect("Should succeed");
        for &v in out.iter() {
            assert!(v >= 0.0, "vesselness must be non-negative, got {v}");
        }
    }

    #[test]
    fn test_frangi_vesselness_with_line() {
        // Horizontal bright line on dark background → vessel response
        let mut img = Array2::<f64>::zeros((30, 30));
        for c in 0..30 {
            img[[15, c]] = 1.0;
        }
        let out =
            frangi_vesselness(img.view(), &[0.5, 1.0, 2.0], 0.5, 0.15).expect("Should succeed");
        // Vesselness along the line should be higher than at a background pixel
        let line_val = out[[15, 15]];
        let bg_val = out[[2, 2]];
        assert!(
            line_val >= bg_val,
            "line vesselness {line_val} should be >= background {bg_val}"
        );
    }

    // ── measure_layer_thickness ───────────────────────────────────────────────

    #[test]
    fn test_measure_layer_thickness_empty_error() {
        let img: Array2<f64> = Array2::zeros((0, 0));
        assert!(measure_layer_thickness(img.view(), 0.5).is_err());
    }

    #[test]
    fn test_measure_layer_thickness_no_layer() {
        let img = Array2::from_elem((10, 10), 0.0_f64);
        let thick = measure_layer_thickness(img.view(), 0.5).expect("Should succeed");
        assert_eq!(thick.dim(), (1, 10));
        assert!(thick.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_measure_layer_thickness_single_layer() {
        let mut img = Array2::<f64>::zeros((20, 5));
        // Layer from row 5 to row 14 (inclusive) in all columns
        for r in 5..15 {
            for c in 0..5 {
                img[[r, c]] = 1.0;
            }
        }
        let thick = measure_layer_thickness(img.view(), 0.5).expect("Should succeed");
        // Thickness should be 14 - 5 = 9
        for &t in thick.iter() {
            assert!((t - 9.0).abs() < 1e-10, "expected thickness 9, got {t}");
        }
    }

    // ── detect_drusen_approximate ─────────────────────────────────────────────

    #[test]
    fn test_detect_drusen_empty_error() {
        let img: Array2<f64> = Array2::zeros((0, 0));
        assert!(detect_drusen_approximate(img.view(), 5.0).is_err());
    }

    #[test]
    fn test_detect_drusen_non_positive_sigma_error() {
        let img = Array2::from_elem((10, 10), 0.5_f64);
        assert!(detect_drusen_approximate(img.view(), 0.0).is_err());
        assert!(detect_drusen_approximate(img.view(), -1.0).is_err());
    }

    #[test]
    fn test_detect_drusen_output_shape() {
        let img = Array2::from_elem((20, 20), 0.3_f64);
        let mask = detect_drusen_approximate(img.view(), 5.0).expect("Should succeed");
        assert_eq!(mask.dim(), (20, 20));
    }
}
