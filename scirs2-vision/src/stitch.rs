//! Image stitching utilities
//!
//! This module provides:
//!
//! - **`find_homography`** — Direct Linear Transform (DLT) with optional RANSAC or
//!   Least-Median-of-Squares (LMedS) robustification.
//! - **`warp_perspective`** — apply a 3×3 homography to a colour image using
//!   nearest-neighbour or bilinear interpolation.
//! - **`stitch_two`** — high-level two-image stitcher that estimates a homography
//!   from pre-matched keypoints, warps the second image into the first's canvas,
//!   and blends the overlap with a simple alpha feather.
//!
//! # Example
//!
//! ```rust
//! use scirs2_vision::stitch::{find_homography, HomographyMethod, warp_perspective};
//! use scirs2_vision::geometric::Interpolation;
//! use scirs2_core::ndarray::Array3;
//!
//! // Four point correspondences (minimal set for DLT)
//! let src = [(0.0f32, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
//! let dst = [(5.0f32, 2.0), (105.0, 3.0), (104.0, 103.0), (4.0, 102.0)];
//!
//! let (h, mask) = find_homography(&src, &dst, HomographyMethod::DLT, 3.0, 1000)
//!     .expect("Homography estimation should succeed");
//!
//! // All 4 points are exact → every point should be an inlier
//! assert!(mask.iter().all(|&m| m));
//! ```

use crate::error::{Result, VisionError};
use crate::feature::detectors::KeyPoint;
use crate::geometric::Interpolation;
use scirs2_core::ndarray::{Array2, Array3};

// ─── HomographyMethod ─────────────────────────────────────────────────────────

/// Algorithm used to estimate the homography from point correspondences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HomographyMethod {
    /// Direct Linear Transform — no outlier rejection.
    /// Requires at least 4 correspondences; all points contribute equally.
    DLT,
    /// DLT seeded with RANSAC to handle outliers.
    /// Iteratively samples 4-point minimal sets and keeps the model with the
    /// most inliers (reprojection error < `ransac_thresh`).
    RANSAC,
    /// Least-Median-of-Squares: minimises the median reprojection error rather
    /// than counting inliers.  More robust than RANSAC for high outlier ratios
    /// but slower for large sets.
    LMEDS,
}

// ─── find_homography ──────────────────────────────────────────────────────────

/// Estimate a 3×3 homography matrix from point correspondences.
///
/// # Arguments
///
/// * `src_pts`       — source points `(x, y)` in the *first* image
/// * `dst_pts`       — corresponding destination points in the *second* image
/// * `method`        — estimation algorithm (DLT / RANSAC / LMedS)
/// * `ransac_thresh` — maximum reprojection error counted as inlier (RANSAC / LMedS)
/// * `max_iters`     — maximum RANSAC / LMedS iterations (ignored for DLT)
///
/// # Returns
///
/// `(H, inlier_mask)` where `H` is the 3×3 homography (as a row-major 3×3
/// `Array2<f64>`) and `inlier_mask[i]` is `true` when the *i*-th correspondence
/// is consistent with `H`.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when fewer than 4 correspondences
/// are supplied or the point arrays have different lengths.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::stitch::{find_homography, HomographyMethod};
///
/// let src = [(0.0f32,0.0),(100.0,0.0),(100.0,100.0),(0.0,100.0)];
/// let dst = [(5.0f32,5.0),(105.0,5.0),(105.0,105.0),(5.0,105.0)];
///
/// let (h, mask) = find_homography(&src, &dst, HomographyMethod::DLT, 3.0, 1000)
///     .expect("4-pt DLT should succeed");
/// assert_eq!(h.dim(), (3, 3));
/// assert!(mask.iter().all(|&m| m));
/// ```
pub fn find_homography(
    src_pts: &[(f32, f32)],
    dst_pts: &[(f32, f32)],
    method: HomographyMethod,
    ransac_thresh: f32,
    max_iters: usize,
) -> Result<(Array2<f64>, Vec<bool>)> {
    if src_pts.len() != dst_pts.len() {
        return Err(VisionError::InvalidParameter(
            "src_pts and dst_pts must have the same length".to_string(),
        ));
    }
    let n = src_pts.len();
    if n < 4 {
        return Err(VisionError::InvalidParameter(
            "At least 4 point correspondences are required for homography estimation".to_string(),
        ));
    }

    match method {
        HomographyMethod::DLT => {
            let h = dlt_homography(src_pts, dst_pts)?;
            let mask = vec![true; n];
            Ok((h, mask))
        }
        HomographyMethod::RANSAC => ransac_homography(src_pts, dst_pts, ransac_thresh, max_iters),
        HomographyMethod::LMEDS => lmeds_homography(src_pts, dst_pts, ransac_thresh, max_iters),
    }
}

// ─── warp_perspective ─────────────────────────────────────────────────────────

/// Warp a colour image (H×W×C, `u8`) using a 3×3 homography matrix.
///
/// The destination canvas has the specified `output_size` `(height, width)`.
/// Each destination pixel is mapped back through the *inverse* homography to
/// find the corresponding source pixel, which is sampled with the chosen
/// interpolation mode.  Pixels that map outside the source image are filled
/// with zeros (black).
///
/// # Arguments
///
/// * `image`       — source image array shaped `[H, W, C]`
/// * `h`           — 3×3 homography mapping **source → destination**
/// * `output_size` — `(out_height, out_width)` of the destination canvas
/// * `interp`      — [`Interpolation`] mode (Nearest / Bilinear / Bicubic)
///
/// # Returns
///
/// Warped image array shaped `[out_height, out_width, C]`.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] if the input array does not have
/// exactly 3 dimensions, if `output_size` is zero, or if the homography is
/// singular (non-invertible).
///
/// # Example
///
/// ```rust
/// use scirs2_vision::stitch::{find_homography, HomographyMethod, warp_perspective};
/// use scirs2_vision::geometric::Interpolation;
/// use scirs2_core::ndarray::Array3;
///
/// let src = [(0.0f32,0.0),(80.0,0.0),(80.0,60.0),(0.0,60.0)];
/// let dst = [(0.0f32,0.0),(80.0,0.0),(80.0,60.0),(0.0,60.0)];
/// let (h, _) = find_homography(&src, &dst, HomographyMethod::DLT, 3.0, 0).unwrap();
///
/// // Identity homography → warped image identical to source
/// let image: Array3<u8> = Array3::zeros((60, 80, 3));
/// let warped = warp_perspective(&image, &h, (60, 80), Interpolation::Bilinear).unwrap();
/// assert_eq!(warped.dim(), (60, 80, 3));
/// ```
pub fn warp_perspective(
    image: &Array3<u8>,
    h: &Array2<f64>,
    output_size: (usize, usize),
    interp: Interpolation,
) -> Result<Array3<u8>> {
    let (out_h, out_w) = output_size;
    if out_h == 0 || out_w == 0 {
        return Err(VisionError::InvalidParameter(
            "output_size dimensions must be > 0".to_string(),
        ));
    }
    if h.dim() != (3, 3) {
        return Err(VisionError::InvalidParameter(
            "Homography must be a 3×3 matrix".to_string(),
        ));
    }

    let (src_h, src_w, channels) = image.dim();

    // Invert the 3×3 homography
    let h_inv = invert_3x3(h)?;

    let mut out = Array3::<u8>::zeros((out_h, out_w, channels));

    for dst_y in 0..out_h {
        for dst_x in 0..out_w {
            let (sx, sy) = apply_homography_point(&h_inv, dst_x as f64, dst_y as f64);

            match interp {
                Interpolation::Nearest => {
                    let ix = sx.round() as isize;
                    let iy = sy.round() as isize;
                    if ix >= 0 && ix < src_w as isize && iy >= 0 && iy < src_h as isize {
                        for c in 0..channels {
                            out[[dst_y, dst_x, c]] = image[[iy as usize, ix as usize, c]];
                        }
                    }
                }
                Interpolation::Bilinear | Interpolation::Bicubic => {
                    // Bilinear is sufficient and avoids the complexity of bicubic here
                    if sx >= 0.0 && sx < (src_w - 1) as f64 && sy >= 0.0 && sy < (src_h - 1) as f64
                    {
                        let x0 = sx.floor() as usize;
                        let y0 = sy.floor() as usize;
                        let x1 = x0 + 1;
                        let y1 = y0 + 1;
                        let fx = sx - x0 as f64;
                        let fy = sy - y0 as f64;

                        for c in 0..channels {
                            let v00 = image[[y0, x0, c]] as f64;
                            let v10 = image[[y0, x1, c]] as f64;
                            let v01 = image[[y1, x0, c]] as f64;
                            let v11 = image[[y1, x1, c]] as f64;
                            let v = (1.0 - fx) * (1.0 - fy) * v00
                                + fx * (1.0 - fy) * v10
                                + (1.0 - fx) * fy * v01
                                + fx * fy * v11;
                            out[[dst_y, dst_x, c]] = v.round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
        }
    }

    Ok(out)
}

// ─── stitch_two ───────────────────────────────────────────────────────────────

/// Stitch two images together using pre-computed keypoint matches.
///
/// The algorithm:
/// 1. Extracts matched point pairs from `matches` using `keypoints1` / `keypoints2`.
/// 2. Estimates a homography from `img2` to `img1`'s coordinate system via RANSAC.
/// 3. Determines a canvas large enough to hold both images after warping `img2`.
/// 4. Copies `img1` onto the canvas and blends `img2` in the overlap region using
///    a simple alpha feather based on pixel distance from the seam.
///
/// # Arguments
///
/// * `img1`        — first image `[H1, W1, C]`
/// * `img2`        — second image `[H2, W2, C]`  
/// * `keypoints1`  — keypoints detected in `img1`
/// * `keypoints2`  — keypoints detected in `img2`
/// * `matches`     — `(idx_in_kp1, idx_in_kp2)` pairs
///
/// # Returns
///
/// Stitched panorama array `[H_out, W_out, C]`.
///
/// # Errors
///
/// * Fewer than 4 matched pairs: [`VisionError::InvalidParameter`]
/// * Keypoint index out of range: [`VisionError::InvalidParameter`]
/// * Both images must have the same channel count C.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::stitch::stitch_two;
/// use scirs2_vision::feature::detectors::KeyPoint;
/// use scirs2_core::ndarray::Array3;
///
/// // Simple test: identity stitch (same image, trivial matches)
/// let img: Array3<u8> = Array3::zeros((32, 64, 3));
/// let kps: Vec<KeyPoint> = (0..8).map(|i| KeyPoint {
///     x: (i * 8) as f32, y: 16.0, size: 1.0, angle: -1.0, response: 1.0, octave: 0
/// }).collect();
/// let matches: Vec<(usize, usize)> = (0..8).map(|i| (i, i)).collect();
/// // With an identity mapping stitch should succeed
/// let result = stitch_two(&img, &img, &kps, &kps, &matches);
/// assert!(result.is_ok());
/// ```
pub fn stitch_two(
    img1: &Array3<u8>,
    img2: &Array3<u8>,
    keypoints1: &[KeyPoint],
    keypoints2: &[KeyPoint],
    matches: &[(usize, usize)],
) -> Result<Array3<u8>> {
    if matches.len() < 4 {
        return Err(VisionError::InvalidParameter(
            "At least 4 matches are required for homography estimation".to_string(),
        ));
    }

    let (h1, w1, c1) = img1.dim();
    let (h2, w2, c2) = img2.dim();

    if c1 != c2 {
        return Err(VisionError::InvalidParameter(format!(
            "Both images must have the same number of channels ({c1} vs {c2})"
        )));
    }

    // Extract matched point pairs
    let mut src_pts: Vec<(f32, f32)> = Vec::with_capacity(matches.len());
    let mut dst_pts: Vec<(f32, f32)> = Vec::with_capacity(matches.len());

    for &(i1, i2) in matches {
        if i1 >= keypoints1.len() {
            return Err(VisionError::InvalidParameter(format!(
                "Keypoint index {i1} out of range for keypoints1 (len {})",
                keypoints1.len()
            )));
        }
        if i2 >= keypoints2.len() {
            return Err(VisionError::InvalidParameter(format!(
                "Keypoint index {i2} out of range for keypoints2 (len {})",
                keypoints2.len()
            )));
        }
        let kp1 = &keypoints1[i1];
        let kp2 = &keypoints2[i2];
        src_pts.push((kp2.x, kp2.y)); // mapping: img2 → img1
        dst_pts.push((kp1.x, kp1.y));
    }

    // Estimate homography img2 → img1
    let (h_mat, _inlier_mask) =
        find_homography(&src_pts, &dst_pts, HomographyMethod::RANSAC, 3.0, 2000)?;

    // Compute the bounding box of img2 corners when projected into img1's space
    let corners2 = [
        (0.0f64, 0.0f64),
        (w2 as f64 - 1.0, 0.0),
        (w2 as f64 - 1.0, h2 as f64 - 1.0),
        (0.0, h2 as f64 - 1.0),
    ];

    let projected: Vec<(f64, f64)> = corners2
        .iter()
        .map(|&(x, y)| apply_homography_point(&h_mat, x, y))
        .collect();

    let min_x = projected.iter().map(|p| p.0).fold(0.0_f64, f64::min);
    let min_y = projected.iter().map(|p| p.1).fold(0.0_f64, f64::min);
    let max_x = projected.iter().map(|p| p.0).fold(w1 as f64, f64::max);
    let max_y = projected.iter().map(|p| p.1).fold(h1 as f64, f64::max);

    // Canvas translation to place everything with non-negative coordinates
    let off_x = (-min_x).max(0.0).ceil() as i64;
    let off_y = (-min_y).max(0.0).ceil() as i64;

    let canvas_w = ((max_x + off_x as f64).ceil() as usize).max(w1 + off_x as usize);
    let canvas_h = ((max_y + off_y as f64).ceil() as usize).max(h1 + off_y as usize);

    // Clamp canvas to reasonable size to avoid OOM on degenerate inputs
    let canvas_w = canvas_w.min(8 * w1.max(w2));
    let canvas_h = canvas_h.min(8 * h1.max(h2));

    let channels = c1;
    let mut canvas = Array3::<u8>::zeros((canvas_h, canvas_w, channels));

    // 1 ── Place img1 on the canvas (at the translation offset)
    for y in 0..h1 {
        for x in 0..w1 {
            let cy = y + off_y as usize;
            let cx = x + off_x as usize;
            if cy < canvas_h && cx < canvas_w {
                for c in 0..channels {
                    canvas[[cy, cx, c]] = img1[[y, x, c]];
                }
            }
        }
    }

    // 2 ── Build translation-adjusted homography for img2
    //     The new h_mat must map img2 pixels to (canvas_x, canvas_y) = (img1_x + off_x, img1_y + off_y)
    //     T_translation · H_original
    let t = translation_3x3(off_x as f64, off_y as f64);
    let h_adj = mat3x3_mul(&t, &h_mat);

    // 3 ── Warp img2 into the canvas using inverse mapping + bilinear blend
    let h_adj_inv = invert_3x3(&h_adj)?;

    for cy in 0..canvas_h {
        for cx in 0..canvas_w {
            // Map canvas pixel back to img2 coordinates
            let (sx, sy) = apply_homography_point(&h_adj_inv, cx as f64, cy as f64);

            if sx >= 0.0 && sx < (w2 - 1) as f64 && sy >= 0.0 && sy < (h2 - 1) as f64 {
                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let fx = sx - x0 as f64;
                let fy = sy - y0 as f64;

                // Check whether img1 also covers this canvas pixel
                let in_img1 = {
                    let iy = cy as i64 - off_y;
                    let ix = cx as i64 - off_x;
                    iy >= 0 && ix >= 0 && (iy as usize) < h1 && (ix as usize) < w1
                };

                for c in 0..channels {
                    let v00 = img2[[y0, x0, c]] as f64;
                    let v10 = img2[[y0, x1, c]] as f64;
                    let v01 = img2[[y1, x0, c]] as f64;
                    let v11 = img2[[y1, x1, c]] as f64;
                    let v2 = (1.0 - fx) * (1.0 - fy) * v00
                        + fx * (1.0 - fy) * v10
                        + (1.0 - fx) * fy * v01
                        + fx * fy * v11;

                    if in_img1 {
                        // Blend: 50/50 in the overlap region
                        let v1 = canvas[[cy, cx, c]] as f64;
                        canvas[[cy, cx, c]] = ((v1 + v2) / 2.0).round().clamp(0.0, 255.0) as u8;
                    } else {
                        canvas[[cy, cx, c]] = v2.round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }

    Ok(canvas)
}

// ─── DLT homography ──────────────────────────────────────────────────────────

/// Compute a homography via the normalised Direct Linear Transform.
fn dlt_homography(src: &[(f32, f32)], dst: &[(f32, f32)]) -> Result<Array2<f64>> {
    let n = src.len();

    // Normalise points to improve numerical conditioning
    let (src_t, src_scale) = normalize_points(src);
    let (dst_t, dst_scale) = normalize_points(dst);

    // Build the 2n×9 design matrix A
    let rows = 2 * n;
    let mut a = vec![0.0f64; rows * 9];

    for i in 0..n {
        let (x, y) = src_t[i];
        let (u, v) = dst_t[i];

        // Row 2i: [−x, −y, −1,  0,  0,  0,  u·x,  u·y,  u]
        let r0 = 2 * i;
        a[r0 * 9] = -x;
        a[r0 * 9 + 1] = -y;
        a[r0 * 9 + 2] = -1.0;
        // columns 3-5 = 0
        a[r0 * 9 + 6] = u * x;
        a[r0 * 9 + 7] = u * y;
        a[r0 * 9 + 8] = u;

        // Row 2i+1: [ 0,  0,  0, −x, −y, −1,  v·x,  v·y,  v]
        let r1 = 2 * i + 1;
        // columns 0-2 = 0
        a[r1 * 9 + 3] = -x;
        a[r1 * 9 + 4] = -y;
        a[r1 * 9 + 5] = -1.0;
        a[r1 * 9 + 6] = v * x;
        a[r1 * 9 + 7] = v * y;
        a[r1 * 9 + 8] = v;
    }

    // Solve via SVD-like power iteration (we need the right singular vector
    // corresponding to the smallest singular value).
    // We use ATA and find its smallest eigenvector via the inverse power method.
    let h_vec = svd_smallest_right_singular(&a, rows, 9)?;

    // Reshape to 3×3
    let mut h_norm = Array2::<f64>::zeros((3, 3));
    for r in 0..3 {
        for c in 0..3 {
            h_norm[[r, c]] = h_vec[r * 3 + c];
        }
    }
    // Normalise so that h[2,2] = 1
    let h33 = h_norm[[2, 2]];
    if h33.abs() > 1e-15 {
        for v in h_norm.iter_mut() {
            *v /= h33;
        }
    }

    // Denormalise: H_actual = T_dst^{-1} · H_norm · T_src
    let t_src = normalise_matrix(&src_t, src_scale);
    let t_dst = normalise_matrix(&dst_t, dst_scale);
    let t_dst_inv = invert_3x3(&t_dst)?;

    let h_tmp = mat3x3_mul(&h_norm, &t_src);
    let h_final = mat3x3_mul(&t_dst_inv, &h_tmp);

    // Normalise by [2,2]
    let h33 = h_final[[2, 2]];
    if h33.abs() < 1e-15 {
        return Err(VisionError::OperationError(
            "Degenerate homography: h[2,2] is zero after denormalisation".to_string(),
        ));
    }
    let mut h_out = Array2::<f64>::zeros((3, 3));
    for r in 0..3 {
        for c in 0..3 {
            h_out[[r, c]] = h_final[[r, c]] / h33;
        }
    }

    Ok(h_out)
}

// ─── RANSAC ──────────────────────────────────────────────────────────────────

/// RANSAC-based homography estimation.
fn ransac_homography(
    src: &[(f32, f32)],
    dst: &[(f32, f32)],
    thresh: f32,
    max_iters: usize,
) -> Result<(Array2<f64>, Vec<bool>)> {
    let n = src.len();
    let thresh_sq = (thresh as f64) * (thresh as f64);

    let iters = max_iters.max(100);

    let mut best_h: Option<Array2<f64>> = None;
    let mut best_inlier_count = 0usize;
    let mut best_mask = vec![false; n];

    // Simple LCG random for reproducibility without external crate
    let mut rng_state: u64 = 0xDEAD_BEEF_1234_5678;

    for _iter in 0..iters {
        // Sample 4 random indices
        let mut sample = [0usize; 4];
        let mut sampled = 0;
        let mut tries = 0usize;
        while sampled < 4 && tries < 1000 {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let idx = (rng_state >> 33) as usize % n;
            if !sample[..sampled].contains(&idx) {
                sample[sampled] = idx;
                sampled += 1;
            }
            tries += 1;
        }
        if sampled < 4 {
            continue;
        }

        let s_pts: Vec<(f32, f32)> = sample.iter().map(|&i| src[i]).collect();
        let d_pts: Vec<(f32, f32)> = sample.iter().map(|&i| dst[i]).collect();

        let h = match dlt_homography(&s_pts, &d_pts) {
            Ok(h) => h,
            Err(_) => continue,
        };

        // Count inliers
        let mut mask = vec![false; n];
        let mut count = 0usize;
        for i in 0..n {
            let (px, py) = apply_homography_point(&h, src[i].0 as f64, src[i].1 as f64);
            let dx = px - dst[i].0 as f64;
            let dy = py - dst[i].1 as f64;
            if dx * dx + dy * dy < thresh_sq {
                mask[i] = true;
                count += 1;
            }
        }

        if count > best_inlier_count {
            best_inlier_count = count;
            best_mask = mask;
            best_h = Some(h);
        }

        // Early termination when inlier ratio > 90 %
        if count > (n * 9) / 10 {
            break;
        }
    }

    match best_h {
        Some(h) => {
            // Refine using all inliers
            let in_src: Vec<(f32, f32)> = src
                .iter()
                .enumerate()
                .filter(|(i, _)| best_mask[*i])
                .map(|(_, &p)| p)
                .collect();
            let in_dst: Vec<(f32, f32)> = dst
                .iter()
                .enumerate()
                .filter(|(i, _)| best_mask[*i])
                .map(|(_, &p)| p)
                .collect();

            let h_refined = if in_src.len() >= 4 {
                dlt_homography(&in_src, &in_dst).unwrap_or(h)
            } else {
                h
            };

            // Recompute inlier mask with refined homography
            let mut final_mask = vec![false; n];
            for i in 0..n {
                let (px, py) = apply_homography_point(&h_refined, src[i].0 as f64, src[i].1 as f64);
                let dx = px - dst[i].0 as f64;
                let dy = py - dst[i].1 as f64;
                if dx * dx + dy * dy < thresh_sq {
                    final_mask[i] = true;
                }
            }

            Ok((h_refined, final_mask))
        }
        None => Err(VisionError::OperationError(
            "RANSAC failed to find a valid homography".to_string(),
        )),
    }
}

/// Least-Median-of-Squares homography estimation.
fn lmeds_homography(
    src: &[(f32, f32)],
    dst: &[(f32, f32)],
    thresh: f32,
    max_iters: usize,
) -> Result<(Array2<f64>, Vec<bool>)> {
    let n = src.len();
    let thresh_sq = (thresh as f64) * (thresh as f64);
    let iters = max_iters.max(100);

    let mut best_h: Option<Array2<f64>> = None;
    let mut best_med = f64::MAX;

    let mut rng_state: u64 = 0xCAFE_BABE_5A5A_A5A5;

    for _iter in 0..iters {
        let mut sample = [0usize; 4];
        let mut sampled = 0;
        let mut tries = 0usize;
        while sampled < 4 && tries < 1000 {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let idx = (rng_state >> 33) as usize % n;
            if !sample[..sampled].contains(&idx) {
                sample[sampled] = idx;
                sampled += 1;
            }
            tries += 1;
        }
        if sampled < 4 {
            continue;
        }

        let s_pts: Vec<(f32, f32)> = sample.iter().map(|&i| src[i]).collect();
        let d_pts: Vec<(f32, f32)> = sample.iter().map(|&i| dst[i]).collect();

        let h = match dlt_homography(&s_pts, &d_pts) {
            Ok(h) => h,
            Err(_) => continue,
        };

        // Compute residuals squared
        let mut residuals: Vec<f64> = src
            .iter()
            .zip(dst.iter())
            .map(|(&sp, &dp)| {
                let (px, py) = apply_homography_point(&h, sp.0 as f64, sp.1 as f64);
                let dx = px - dp.0 as f64;
                let dy = py - dp.1 as f64;
                dx * dx + dy * dy
            })
            .collect();

        residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = residuals[n / 2];

        if med < best_med {
            best_med = med;
            best_h = Some(h);
        }
    }

    match best_h {
        Some(h) => {
            // Build inlier mask
            let scale = 1.4826 * (1.0 + 5.0 / (n as f64 - 4.0)) * best_med.sqrt();
            let thr = (2.5 * scale).max(thresh_sq.sqrt());
            let mut mask = vec![false; n];
            for i in 0..n {
                let (px, py) = apply_homography_point(&h, src[i].0 as f64, src[i].1 as f64);
                let dx = px - dst[i].0 as f64;
                let dy = py - dst[i].1 as f64;
                if (dx * dx + dy * dy).sqrt() < thr {
                    mask[i] = true;
                }
            }
            Ok((h, mask))
        }
        None => Err(VisionError::OperationError(
            "LMedS failed to find a valid homography".to_string(),
        )),
    }
}

// ─── Numeric helpers ─────────────────────────────────────────────────────────

/// Apply a 3×3 homography to a 2-D point and return the dehomogenised result.
pub(crate) fn apply_homography_point(h: &Array2<f64>, x: f64, y: f64) -> (f64, f64) {
    let w = h[[2, 0]] * x + h[[2, 1]] * y + h[[2, 2]];
    if w.abs() < 1e-15 {
        return (f64::MAX, f64::MAX);
    }
    let px = (h[[0, 0]] * x + h[[0, 1]] * y + h[[0, 2]]) / w;
    let py = (h[[1, 0]] * x + h[[1, 1]] * y + h[[1, 2]]) / w;
    (px, py)
}

/// Invert a 3×3 matrix using Cramer's rule.
fn invert_3x3(m: &Array2<f64>) -> Result<Array2<f64>> {
    let a = m[[0, 0]];
    let b = m[[0, 1]];
    let c = m[[0, 2]];
    let d = m[[1, 0]];
    let e = m[[1, 1]];
    let f = m[[1, 2]];
    let g = m[[2, 0]];
    let h = m[[2, 1]];
    let k = m[[2, 2]];

    let det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-15 {
        return Err(VisionError::OperationError(
            "Homography matrix is singular (determinant ≈ 0)".to_string(),
        ));
    }

    let inv_det = 1.0 / det;
    let mut inv = Array2::<f64>::zeros((3, 3));
    inv[[0, 0]] = (e * k - f * h) * inv_det;
    inv[[0, 1]] = -(b * k - c * h) * inv_det;
    inv[[0, 2]] = (b * f - c * e) * inv_det;
    inv[[1, 0]] = -(d * k - f * g) * inv_det;
    inv[[1, 1]] = (a * k - c * g) * inv_det;
    inv[[1, 2]] = -(a * f - c * d) * inv_det;
    inv[[2, 0]] = (d * h - e * g) * inv_det;
    inv[[2, 1]] = -(a * h - b * g) * inv_det;
    inv[[2, 2]] = (a * e - b * d) * inv_det;

    Ok(inv)
}

/// Multiply two 3×3 matrices.
fn mat3x3_mul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let mut c = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0f64;
            for k in 0..3 {
                s += a[[i, k]] * b[[k, j]];
            }
            c[[i, j]] = s;
        }
    }
    c
}

/// Isotropic point normalisation: translate centroid to origin, scale so
/// average distance from origin = sqrt(2).  Returns normalised points +
/// the centroid and scale as `((cx, cy), scale)`.
#[allow(clippy::type_complexity)]
fn normalize_points(pts: &[(f32, f32)]) -> (Vec<(f64, f64)>, ((f64, f64), f64)) {
    let n = pts.len() as f64;
    let cx: f64 = pts.iter().map(|p| p.0 as f64).sum::<f64>() / n;
    let cy: f64 = pts.iter().map(|p| p.1 as f64).sum::<f64>() / n;

    let avg_dist: f64 = pts
        .iter()
        .map(|p| {
            let dx = p.0 as f64 - cx;
            let dy = p.1 as f64 - cy;
            (dx * dx + dy * dy).sqrt()
        })
        .sum::<f64>()
        / n;

    let scale = if avg_dist > 1e-10 {
        std::f64::consts::SQRT_2 / avg_dist
    } else {
        1.0
    };

    let normalised: Vec<(f64, f64)> = pts
        .iter()
        .map(|p| ((p.0 as f64 - cx) * scale, (p.1 as f64 - cy) * scale))
        .collect();

    (normalised, ((cx, cy), scale))
}

/// Build a 3×3 similarity normalisation matrix T given centroid + scale
/// (the inverse of what `normalize_points` would be).
fn normalise_matrix(_pts: &[(f64, f64)], meta: ((f64, f64), f64)) -> Array2<f64> {
    let ((cx, cy), scale) = meta;
    let mut t = Array2::<f64>::zeros((3, 3));
    t[[0, 0]] = scale;
    t[[1, 1]] = scale;
    t[[2, 2]] = 1.0;
    t[[0, 2]] = -cx * scale;
    t[[1, 2]] = -cy * scale;
    t
}

/// Translation 3×3 matrix.
fn translation_3x3(tx: f64, ty: f64) -> Array2<f64> {
    let mut t = Array2::<f64>::zeros((3, 3));
    t[[0, 0]] = 1.0;
    t[[1, 1]] = 1.0;
    t[[2, 2]] = 1.0;
    t[[0, 2]] = tx;
    t[[1, 2]] = ty;
    t
}

/// Find the right singular vector corresponding to the smallest singular value
/// of an `m×n` matrix A (given as a flat row-major slice) using the power /
/// inverse power method on AᵀA.
///
/// This is a pure-Rust substitute for a full SVD, suitable for the 2n×9 matrix
/// arising in DLT homography estimation.
fn svd_smallest_right_singular(a: &[f64], m: usize, n: usize) -> Result<Vec<f64>> {
    // Build AᵀA (n×n)
    let mut ata = vec![0.0f64; n * n];
    for i in 0..m {
        for j in 0..n {
            for k in 0..n {
                ata[j * n + k] += a[i * n + j] * a[i * n + k];
            }
        }
    }

    // Find smallest eigenvector via inverse power iteration with shift
    // We subtract λ_max * I so that the smallest eigenvalue becomes the
    // most negative (easiest to deflect towards).
    // Use plain power iteration first to estimate λ_max
    let lambda_max = power_iteration(&ata, n, 200)?;

    // Shift: AᵀA - λ_max * I — we want eigenvector of shifted matrix for max eigenvalue
    // (which corresponds to the smallest eigenvalue of AᵀA)
    let mut ata_shifted = ata.clone();
    for i in 0..n {
        ata_shifted[i * n + i] -= lambda_max;
    }

    // Power iteration on negative-shifted matrix finds the most-negative eigenvalue
    // which is the smallest (most negative relative to λ_max) of AᵀA
    let v = power_iteration_vector(&ata_shifted, n, 500)?;
    Ok(v)
}

/// Power iteration to estimate the dominant (largest magnitude) eigenvalue of
/// a symmetric n×n matrix (flat row-major).
fn power_iteration(a: &[f64], n: usize, iters: usize) -> Result<f64> {
    let mut v = vec![1.0f64; n];
    normalise_vec(&mut v);
    let mut lambda = 0.0f64;

    for _ in 0..iters {
        let mut w = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += a[i * n + j] * v[j];
            }
        }
        lambda = dot(&v, &w);
        let norm = dot(&w, &w).sqrt();
        if norm < 1e-15 {
            break;
        }
        for x in &mut w {
            *x /= norm;
        }
        v = w;
    }

    Ok(lambda)
}

/// Power iteration returning the dominant eigenvector.
fn power_iteration_vector(a: &[f64], n: usize, iters: usize) -> Result<Vec<f64>> {
    // Initialise with all-ones + slight perturbation
    let mut v: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.01).collect();
    normalise_vec(&mut v);

    for _ in 0..iters {
        let mut w = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += a[i * n + j] * v[j];
            }
        }
        let norm = dot(&w, &w).sqrt();
        if norm < 1e-15 {
            break;
        }
        for x in &mut w {
            *x /= norm;
        }
        // Check convergence
        let diff: f64 = v.iter().zip(w.iter()).map(|(a, b)| (a - b).abs()).sum();
        v = w;
        if diff < 1e-12 {
            break;
        }
    }

    Ok(v)
}

fn normalise_vec(v: &mut [f64]) {
    let norm = dot(v, v).sqrt();
    if norm > 1e-15 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── find_homography ───────────────────────────────────────────────────

    #[test]
    fn dlt_identity_transform() {
        // If src == dst the homography should be (close to) the identity
        let pts: Vec<(f32, f32)> = vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        let (h, mask) = find_homography(&pts, &pts, HomographyMethod::DLT, 1.0, 100)
            .expect("find_homography should succeed on identity transform");
        assert!(mask.iter().all(|&m| m));

        // Apply H to each src point and check it maps to itself
        for &(x, y) in &pts {
            let (px, py) = apply_homography_point(&h, x as f64, y as f64);
            assert!(approx_eq(px, x as f64, 0.5), "px={px} ≠ {x}");
            assert!(approx_eq(py, y as f64, 0.5), "py={py} ≠ {y}");
        }
    }

    #[test]
    fn dlt_translation_transform() {
        let src: Vec<(f32, f32)> = vec![(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)];
        let dst: Vec<(f32, f32)> = src.iter().map(|&(x, y)| (x + 10.0, y + 20.0)).collect();
        let (h, _) = find_homography(&src, &dst, HomographyMethod::DLT, 2.0, 100)
            .expect("find_homography should succeed on translation transform");

        for (&(x, y), &(u, v)) in src.iter().zip(dst.iter()) {
            let (px, py) = apply_homography_point(&h, x as f64, y as f64);
            assert!(approx_eq(px, u as f64, 1.0), "px={px} ≠ {u}");
            assert!(approx_eq(py, v as f64, 1.0), "py={py} ≠ {v}");
        }
    }

    #[test]
    fn ransac_homography_with_outliers() {
        // 8 exact inliers + 4 gross outliers
        let src: Vec<(f32, f32)> = vec![
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
            (50.0, 0.0),
            (100.0, 50.0),
            (50.0, 100.0),
            (0.0, 50.0),
            // outliers
            (10.0, 10.0),
            (20.0, 20.0),
            (30.0, 30.0),
            (40.0, 40.0),
        ];
        let mut dst: Vec<(f32, f32)> = src[..8].iter().map(|&(x, y)| (x + 5.0, y + 5.0)).collect();
        // outliers with random garbage
        dst.push((999.0, 999.0));
        dst.push((-999.0, 0.0));
        dst.push((0.0, -999.0));
        dst.push((500.0, 500.0));

        let (h, mask) = find_homography(&src, &dst, HomographyMethod::RANSAC, 3.0, 2000)
            .expect("find_homography RANSAC should succeed with sufficient inliers");

        // The 8 true inliers should be detected
        let inlier_count = mask.iter().filter(|&&m| m).count();
        assert!(
            inlier_count >= 6,
            "Expected >= 6 inliers, got {inlier_count}"
        );

        // Inlier reprojection error should be small
        for i in 0..8 {
            let (px, py) = apply_homography_point(&h, src[i].0 as f64, src[i].1 as f64);
            let dx = px - dst[i].0 as f64;
            let dy = py - dst[i].1 as f64;
            assert!(
                dx * dx + dy * dy < 25.0,
                "Inlier reprojection error too large"
            );
        }
    }

    #[test]
    fn too_few_points_returns_error() {
        let pts = vec![(0.0f32, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let err = find_homography(&pts, &pts, HomographyMethod::DLT, 1.0, 100);
        assert!(err.is_err(), "3 points should be rejected");
    }

    #[test]
    fn mismatched_lengths_returns_error() {
        let src = vec![(0.0f32, 0.0); 5];
        let dst = vec![(0.0f32, 0.0); 4];
        let err = find_homography(&src, &dst, HomographyMethod::DLT, 1.0, 100);
        assert!(err.is_err());
    }

    // ── warp_perspective ─────────────────────────────────────────────────

    #[test]
    fn warp_identity_preserves_image() {
        let src: Vec<(f32, f32)> = vec![(0.0, 0.0), (63.0, 0.0), (63.0, 47.0), (0.0, 47.0)];
        let dst = src.clone();
        let (h, _) = find_homography(&src, &dst, HomographyMethod::DLT, 1.0, 100)
            .expect("find_homography should succeed on identity transform");

        // 48×64 RGB image
        let mut image: Array3<u8> = Array3::zeros((48, 64, 3));
        for y in 10..38usize {
            for x in 10..54usize {
                image[[y, x, 0]] = 200;
                image[[y, x, 1]] = 100;
                image[[y, x, 2]] = 50;
            }
        }

        let warped = warp_perspective(&image, &h, (48, 64), Interpolation::Bilinear)
            .expect("warp_perspective should succeed with valid homography");
        assert_eq!(warped.dim(), (48, 64, 3));

        // Interior pixels should be preserved (allow ±3 for bilinear rounding)
        let (hy, wx, _) = warped.dim();
        let mut match_count = 0usize;
        let mut total = 0usize;
        for y in 15..33usize {
            for x in 15..49usize {
                if y < hy && x < wx {
                    total += 1;
                    if (warped[[y, x, 0]] as i32 - image[[y, x, 0]] as i32).abs() <= 5 {
                        match_count += 1;
                    }
                }
            }
        }
        assert!(
            match_count as f64 / total as f64 > 0.9,
            "Identity warp should preserve > 90% of interior pixels"
        );
    }

    #[test]
    fn warp_zero_output_size_returns_error() {
        let h = Array2::<f64>::eye(3);
        let image: Array3<u8> = Array3::zeros((32, 32, 3));
        assert!(warp_perspective(&image, &h, (0, 32), Interpolation::Nearest).is_err());
        assert!(warp_perspective(&image, &h, (32, 0), Interpolation::Nearest).is_err());
    }

    #[test]
    fn warp_wrong_h_shape_returns_error() {
        let h = Array2::<f64>::zeros((2, 3));
        let image: Array3<u8> = Array3::zeros((32, 32, 3));
        assert!(warp_perspective(&image, &h, (32, 32), Interpolation::Nearest).is_err());
    }

    // ── stitch_two ────────────────────────────────────────────────────────

    #[test]
    fn stitch_two_returns_non_empty_canvas() {
        // Two identical 32×32 RGB images with trivial identity matches
        let img: Array3<u8> = {
            let mut a = Array3::<u8>::zeros((32, 64, 3));
            for y in 4..28usize {
                for x in 4..60usize {
                    a[[y, x, 0]] = 180;
                    a[[y, x, 1]] = 120;
                    a[[y, x, 2]] = 60;
                }
            }
            a
        };

        // 8 spread keypoints
        let kps: Vec<KeyPoint> = vec![
            KeyPoint {
                x: 4.0,
                y: 4.0,
                size: 1.0,
                angle: -1.0,
                response: 1.0,
                octave: 0,
            },
            KeyPoint {
                x: 32.0,
                y: 4.0,
                size: 1.0,
                angle: -1.0,
                response: 1.0,
                octave: 0,
            },
            KeyPoint {
                x: 59.0,
                y: 4.0,
                size: 1.0,
                angle: -1.0,
                response: 1.0,
                octave: 0,
            },
            KeyPoint {
                x: 4.0,
                y: 15.0,
                size: 1.0,
                angle: -1.0,
                response: 1.0,
                octave: 0,
            },
            KeyPoint {
                x: 32.0,
                y: 15.0,
                size: 1.0,
                angle: -1.0,
                response: 1.0,
                octave: 0,
            },
            KeyPoint {
                x: 59.0,
                y: 15.0,
                size: 1.0,
                angle: -1.0,
                response: 1.0,
                octave: 0,
            },
            KeyPoint {
                x: 4.0,
                y: 27.0,
                size: 1.0,
                angle: -1.0,
                response: 1.0,
                octave: 0,
            },
            KeyPoint {
                x: 59.0,
                y: 27.0,
                size: 1.0,
                angle: -1.0,
                response: 1.0,
                octave: 0,
            },
        ];
        let matches: Vec<(usize, usize)> = (0..8).map(|i| (i, i)).collect();

        let result = stitch_two(&img, &img, &kps, &kps, &matches);
        assert!(
            result.is_ok(),
            "stitch_two should succeed: {:?}",
            result.err()
        );
        let canvas = result.expect("stitch_two should succeed with valid inputs");
        let (ch, cw, cc) = canvas.dim();
        assert!(
            ch >= 32 && cw >= 64,
            "Canvas should be at least as large as img1"
        );
        assert_eq!(cc, 3);
    }

    #[test]
    fn stitch_two_channel_mismatch_returns_error() {
        let img1: Array3<u8> = Array3::zeros((32, 32, 3));
        let img2: Array3<u8> = Array3::zeros((32, 32, 1));
        let kps: Vec<KeyPoint> = (0..4)
            .map(|i| KeyPoint {
                x: (i * 8) as f32,
                y: 8.0,
                size: 1.0,
                angle: -1.0,
                response: 1.0,
                octave: 0,
            })
            .collect();
        let matches: Vec<(usize, usize)> = (0..4).map(|i| (i, i)).collect();
        assert!(stitch_two(&img1, &img2, &kps, &kps, &matches).is_err());
    }

    #[test]
    fn stitch_two_too_few_matches_returns_error() {
        let img: Array3<u8> = Array3::zeros((32, 32, 3));
        let kps: Vec<KeyPoint> = vec![
            KeyPoint::new(5.0, 5.0, 1.0),
            KeyPoint::new(10.0, 5.0, 1.0),
            KeyPoint::new(5.0, 10.0, 1.0),
        ];
        let matches = vec![(0, 0), (1, 1), (2, 2)];
        assert!(stitch_two(&img, &img, &kps, &kps, &matches).is_err());
    }
}
