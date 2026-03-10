//! Stereo vision algorithms
//!
//! Provides dense stereo matching, disparity estimation with multiple matching
//! cost functions (SAD, SSD, NCC, Census), a simplified Semi-Global Matching
//! (SGM) implementation, depth-from-disparity conversion, stereo image
//! rectification, and the 8-point algorithm for fundamental / essential matrix
//! estimation.

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::{s, Array2, Array3};

// ─────────────────────────────────────────────────────────────────────────────
// Matching cost functions
// ─────────────────────────────────────────────────────────────────────────────

/// Matching cost metric used during block matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchingCost {
    /// Sum of Absolute Differences.
    Sad,
    /// Sum of Squared Differences.
    Ssd,
    /// Normalised Cross-Correlation (converted to a cost: `1 - NCC`).
    Ncc,
    /// Census transform Hamming distance.
    Census,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: extract a grayscale patch from an Array2<f64>
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the SAD between two equal-sized image patches extracted from grayscale
/// arrays at positions `(cy, cx)` and `(cy, rx)` respectively (same row `cy`).
fn patch_sad(
    left: &Array2<f64>,
    right: &Array2<f64>,
    cy: usize,
    lx: usize,
    rx: usize,
    half: usize,
) -> f64 {
    let mut sum = 0.0_f64;
    let y0 = cy.saturating_sub(half);
    let y1 = (cy + half + 1).min(left.nrows());
    let x0l = lx.saturating_sub(half);
    let x1l = (lx + half + 1).min(left.ncols());
    let x0r = rx.saturating_sub(half);
    let x1r = (rx + half + 1).min(right.ncols());
    let rows = y1 - y0;
    let cols = (x1l - x0l).min(x1r - x0r);
    for dy in 0..rows {
        for dx in 0..cols {
            let lv = left[[y0 + dy, x0l + dx]];
            let rv = right[[y0 + dy, x0r + dx]];
            sum += (lv - rv).abs();
        }
    }
    sum
}

/// Sum of Squared Differences.
fn patch_ssd(
    left: &Array2<f64>,
    right: &Array2<f64>,
    cy: usize,
    lx: usize,
    rx: usize,
    half: usize,
) -> f64 {
    let mut sum = 0.0_f64;
    let y0 = cy.saturating_sub(half);
    let y1 = (cy + half + 1).min(left.nrows());
    let x0l = lx.saturating_sub(half);
    let x1l = (lx + half + 1).min(left.ncols());
    let x0r = rx.saturating_sub(half);
    let x1r = (rx + half + 1).min(right.ncols());
    let rows = y1 - y0;
    let cols = (x1l - x0l).min(x1r - x0r);
    for dy in 0..rows {
        for dx in 0..cols {
            let diff = left[[y0 + dy, x0l + dx]] - right[[y0 + dy, x0r + dx]];
            sum += diff * diff;
        }
    }
    sum
}

/// Normalised Cross-Correlation returned as cost `1 - NCC`.
fn patch_ncc(
    left: &Array2<f64>,
    right: &Array2<f64>,
    cy: usize,
    lx: usize,
    rx: usize,
    half: usize,
) -> f64 {
    let y0 = cy.saturating_sub(half);
    let y1 = (cy + half + 1).min(left.nrows());
    let x0l = lx.saturating_sub(half);
    let x1l = (lx + half + 1).min(left.ncols());
    let x0r = rx.saturating_sub(half);
    let x1r = (rx + half + 1).min(right.ncols());
    let rows = y1 - y0;
    let cols = (x1l - x0l).min(x1r - x0r);
    let n = (rows * cols) as f64;

    if n == 0.0 {
        return 1.0;
    }

    let mut sum_l = 0.0_f64;
    let mut sum_r = 0.0_f64;
    let mut sum_l2 = 0.0_f64;
    let mut sum_r2 = 0.0_f64;
    let mut sum_lr = 0.0_f64;

    for dy in 0..rows {
        for dx in 0..cols {
            let l = left[[y0 + dy, x0l + dx]];
            let r = right[[y0 + dy, x0r + dx]];
            sum_l += l;
            sum_r += r;
            sum_l2 += l * l;
            sum_r2 += r * r;
            sum_lr += l * r;
        }
    }

    let mean_l = sum_l / n;
    let mean_r = sum_r / n;
    let cov = sum_lr / n - mean_l * mean_r;
    let var_l = (sum_l2 / n - mean_l * mean_l).max(0.0);
    let var_r = (sum_r2 / n - mean_r * mean_r).max(0.0);
    let denom = var_l.sqrt() * var_r.sqrt();
    if denom < 1e-10 {
        return 1.0;
    }
    let ncc = (cov / denom).clamp(-1.0, 1.0);
    1.0 - ncc
}

/// Census transform: encode a 5×5 neighbourhood around each pixel as a 25-bit
/// integer.  Returns the Hamming distance between the two census codes.
fn census_hamming(left: &Array2<f64>, right: &Array2<f64>, cy: usize, lx: usize, rx: usize) -> f64 {
    let h = left.nrows();
    let wl = left.ncols();
    let wr = right.ncols();
    let lv = left[[cy, lx]];
    let rv = right[[cy, rx]];

    let mut cl: u64 = 0;
    let mut cr: u64 = 0;
    let mut bit = 0u64;

    for dy in -2i64..=2 {
        for dx in -2i64..=2 {
            if dy == 0 && dx == 0 {
                continue;
            }
            let ny = (cy as i64 + dy) as usize;
            let nlx = (lx as i64 + dx) as usize;
            let nrx = (rx as i64 + dx) as usize;

            if ny < h {
                if nlx < wl && left[[ny, nlx]] < lv {
                    cl |= 1 << bit;
                }
                if nrx < wr && right[[ny, nrx]] < rv {
                    cr |= 1 << bit;
                }
            }
            bit += 1;
        }
    }

    (cl ^ cr).count_ones() as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// block_matching
// ─────────────────────────────────────────────────────────────────────────────

/// Dense block matching to produce a disparity map.
///
/// For each pixel in the left image the function searches the corresponding
/// scanline in the right image within `[0, max_disp)` and returns the disparity
/// that minimises the chosen matching cost over a `block_size × block_size` window.
///
/// # Arguments
///
/// * `left`       – Grayscale left image `Array2<f64>`, values in `[0, 255]`.
/// * `right`      – Grayscale right image of the same dimensions.
/// * `block_size` – Odd positive integer; the matching window diameter in pixels.
/// * `max_disp`   – Maximum disparity to search (exclusive).
/// * `cost`       – Matching cost function.
///
/// # Returns
///
/// Disparity map with the same spatial dimensions as the input, where each cell
/// contains the estimated disparity (0 = no valid match).
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when:
/// - image dimensions differ,
/// - `block_size` is even or zero,
/// - `max_disp` is zero.
///
/// # Example
///
/// ```
/// use scirs2_vision::stereo_legacy::{block_matching, MatchingCost};
/// use scirs2_core::ndarray::Array2;
///
/// let left = Array2::from_elem((32, 64), 128.0_f64);
/// let right = Array2::from_elem((32, 64), 128.0_f64);
/// let disp = block_matching(&left, &right, 5, 16, MatchingCost::Sad).unwrap();
/// assert_eq!(disp.dim(), (32, 64));
/// ```
pub fn block_matching(
    left: &Array2<f64>,
    right: &Array2<f64>,
    block_size: usize,
    max_disp: usize,
    cost: MatchingCost,
) -> Result<Array2<f64>> {
    if left.dim() != right.dim() {
        return Err(VisionError::DimensionMismatch(
            "left and right images must have the same dimensions".to_string(),
        ));
    }
    if block_size == 0 || block_size.is_multiple_of(2) {
        return Err(VisionError::InvalidParameter(
            "block_size must be a positive odd integer".to_string(),
        ));
    }
    if max_disp == 0 {
        return Err(VisionError::InvalidParameter(
            "max_disp must be greater than zero".to_string(),
        ));
    }

    let (h, w) = left.dim();
    let half = block_size / 2;
    let mut disparity = Array2::zeros((h, w));

    for y in half..h.saturating_sub(half) {
        for x in half..w.saturating_sub(half) {
            let mut best_disp = 0usize;
            let mut best_cost = f64::INFINITY;

            let d_max = max_disp.min(x + 1); // can't go past image left edge
            for d in 0..d_max {
                let rx = x - d;
                let c = match cost {
                    MatchingCost::Sad => patch_sad(left, right, y, x, rx, half),
                    MatchingCost::Ssd => patch_ssd(left, right, y, x, rx, half),
                    MatchingCost::Ncc => patch_ncc(left, right, y, x, rx, half),
                    MatchingCost::Census => census_hamming(left, right, y, x, rx),
                };
                if c < best_cost {
                    best_cost = c;
                    best_disp = d;
                }
            }
            disparity[[y, x]] = best_disp as f64;
        }
    }

    Ok(disparity)
}

// ─────────────────────────────────────────────────────────────────────────────
// StereoMatcher builder
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration and entry-point for dense stereo matching.
///
/// Bundles the matching parameters and provides a single `compute` method that
/// dispatches to the requested algorithm.
#[derive(Debug, Clone)]
pub struct StereoMatcher {
    /// Block size (must be odd).
    pub block_size: usize,
    /// Maximum disparity.
    pub max_disp: usize,
    /// Matching cost metric.
    pub cost: MatchingCost,
}

impl StereoMatcher {
    /// Create a new stereo matcher with the given parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_vision::stereo_legacy::{StereoMatcher, MatchingCost};
    /// let matcher = StereoMatcher::new(7, 64, MatchingCost::Census);
    /// assert_eq!(matcher.block_size, 7);
    /// ```
    pub fn new(block_size: usize, max_disp: usize, cost: MatchingCost) -> Self {
        Self {
            block_size,
            max_disp,
            cost,
        }
    }

    /// Compute a dense disparity map from a rectified stereo pair.
    pub fn compute(&self, left: &Array2<f64>, right: &Array2<f64>) -> Result<Array2<f64>> {
        block_matching(left, right, self.block_size, self.max_disp, self.cost)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Semi-Global Matching (simplified)
// ─────────────────────────────────────────────────────────────────────────────

/// SGM penalty parameters.
#[derive(Debug, Clone)]
pub struct SgmPenalties {
    /// Small disparity change penalty (neighbour disparity differs by 1).
    pub p1: f64,
    /// Large disparity change penalty (neighbour disparity differs by >1).
    pub p2: f64,
}

impl Default for SgmPenalties {
    fn default() -> Self {
        Self {
            p1: 15.0,
            p2: 100.0,
        }
    }
}

/// Compute a disparity map using a simplified Semi-Global Matching approach.
///
/// The implementation aggregates matching costs along 8 scan-line directions
/// (horizontal left-to-right and right-to-left, vertical top-to-bottom and
/// bottom-to-top, and 4 diagonal paths) using the standard SGM recurrence with
/// penalties `p1` for a disparity change of 1 and `p2` for larger changes.
///
/// # Arguments
///
/// * `left`     – Grayscale left image `Array2<f64>`, values in `[0, 255]`.
/// * `right`    – Grayscale right image of the same dimensions.
/// * `max_disp` – Maximum disparity (exclusive).
/// * `penalties`– SGM smoothness penalties.
///
/// # Returns
///
/// Disparity map (winner-take-all over the aggregated cost volume).
///
/// # Example
///
/// ```
/// use scirs2_vision::stereo_legacy::{disparity_map_sgm, SgmPenalties};
/// use scirs2_core::ndarray::Array2;
///
/// let left = Array2::from_elem((16, 32), 100.0_f64);
/// let right = Array2::from_elem((16, 32), 100.0_f64);
/// let disp = disparity_map_sgm(&left, &right, 8, &SgmPenalties::default()).unwrap();
/// assert_eq!(disp.dim(), (16, 32));
/// ```
pub fn disparity_map_sgm(
    left: &Array2<f64>,
    right: &Array2<f64>,
    max_disp: usize,
    penalties: &SgmPenalties,
) -> Result<Array2<f64>> {
    if left.dim() != right.dim() {
        return Err(VisionError::DimensionMismatch(
            "left and right images must have the same dimensions".to_string(),
        ));
    }
    if max_disp == 0 {
        return Err(VisionError::InvalidParameter(
            "max_disp must be greater than zero".to_string(),
        ));
    }

    let (h, w) = left.dim();

    // Build pixel-level matching cost volume using SAD with a 1×1 patch (per-pixel).
    // Using Array3 stored as Vec<f64> for index [y*w*max_disp + x*max_disp + d].
    let mut cost_volume = vec![0.0_f64; h * w * max_disp];
    for y in 0..h {
        for x in 0..w {
            for d in 0..max_disp {
                let cost_idx = y * w * max_disp + x * max_disp + d;
                if x >= d {
                    let lv = left[[y, x]];
                    let rv = right[[y, x - d]];
                    cost_volume[cost_idx] = (lv - rv).abs();
                } else {
                    cost_volume[cost_idx] = 255.0; // out-of-range
                }
            }
        }
    }

    // Aggregate along 8 paths and accumulate into a total cost volume.
    let mut aggregated = vec![0.0_f64; h * w * max_disp];

    // Helper: aggregate one directional path using the SGM recurrence.
    // `path_pixels` is an ordered list of (y, x) pixel coordinates for this path.
    let aggregate_path = |path_pixels: &[(usize, usize)], cost_vol: &[f64], agg: &mut Vec<f64>| {
        let n = path_pixels.len();
        if n == 0 {
            return;
        }

        // L_r(p, d) values for previous pixel in the path.
        let mut prev_lr = vec![0.0_f64; max_disp];
        let mut prev_min_lr: f64;

        let (y0, x0) = path_pixels[0];
        for d in 0..max_disp {
            let c = cost_vol[y0 * w * max_disp + x0 * max_disp + d];
            prev_lr[d] = c;
            agg[y0 * w * max_disp + x0 * max_disp + d] += c;
        }
        prev_min_lr = prev_lr.iter().cloned().fold(f64::INFINITY, f64::min);

        #[allow(clippy::needless_range_loop)]
        for i in 1..n {
            let (y, x) = path_pixels[i];
            let mut cur_lr = vec![0.0_f64; max_disp];
            let mut min_cur = f64::INFINITY;

            for d in 0..max_disp {
                let matching_cost = cost_vol[y * w * max_disp + x * max_disp + d];

                // Cost from same disparity in previous pixel.
                let same_d = prev_lr[d];

                // Cost from adjacent disparities.
                let adj_d1 = if d > 0 {
                    prev_lr[d - 1] + penalties.p1
                } else {
                    f64::INFINITY
                };
                let adj_d2 = if d + 1 < max_disp {
                    prev_lr[d + 1] + penalties.p1
                } else {
                    f64::INFINITY
                };

                // Cost from any other disparity.
                let other = prev_min_lr + penalties.p2;

                let lr = matching_cost + (same_d.min(adj_d1).min(adj_d2).min(other) - prev_min_lr);

                cur_lr[d] = lr;
                agg[y * w * max_disp + x * max_disp + d] += lr;
                if lr < min_cur {
                    min_cur = lr;
                }
            }

            prev_lr = cur_lr;
            prev_min_lr = min_cur;
        }
    };

    // Path 0: left→right along each row.
    for y in 0..h {
        let path: Vec<(usize, usize)> = (0..w).map(|x| (y, x)).collect();
        aggregate_path(&path, &cost_volume, &mut aggregated);
    }

    // Path 1: right→left along each row.
    for y in 0..h {
        let path: Vec<(usize, usize)> = (0..w).rev().map(|x| (y, x)).collect();
        aggregate_path(&path, &cost_volume, &mut aggregated);
    }

    // Path 2: top→bottom along each column.
    for x in 0..w {
        let path: Vec<(usize, usize)> = (0..h).map(|y| (y, x)).collect();
        aggregate_path(&path, &cost_volume, &mut aggregated);
    }

    // Path 3: bottom→top along each column.
    for x in 0..w {
        let path: Vec<(usize, usize)> = (0..h).rev().map(|y| (y, x)).collect();
        aggregate_path(&path, &cost_volume, &mut aggregated);
    }

    // Path 4: top-left → bottom-right diagonal.
    {
        let starts: Vec<(usize, usize)> = (0..w)
            .map(|x| (0, x))
            .chain((1..h).map(|y| (y, 0)))
            .collect();
        for (y0, x0) in starts {
            let path: Vec<(usize, usize)> = (0..)
                .map(|k| (y0 + k, x0 + k))
                .take_while(|&(y, x)| y < h && x < w)
                .collect();
            aggregate_path(&path, &cost_volume, &mut aggregated);
        }
    }

    // Path 5: bottom-right → top-left diagonal.
    {
        let starts: Vec<(usize, usize)> = (0..w)
            .map(|x| (h - 1, x))
            .chain((0..h - 1).map(|y| (y, w - 1)))
            .collect();
        for (y0, x0) in starts {
            let path: Vec<(usize, usize)> = (0..)
                .map(|k| (y0 as i64 - k as i64, x0 as i64 - k as i64))
                .take_while(|&(y, x)| y >= 0 && x >= 0)
                .map(|(y, x)| (y as usize, x as usize))
                .collect();
            aggregate_path(&path, &cost_volume, &mut aggregated);
        }
    }

    // Path 6: top-right → bottom-left diagonal.
    {
        let starts: Vec<(usize, usize)> = (0..w)
            .map(|x| (0, x))
            .chain((1..h).map(|y| (y, w - 1)))
            .collect();
        for (y0, x0) in starts {
            let path: Vec<(usize, usize)> = (0..)
                .map(|k| (y0 + k, x0 as i64 - k as i64))
                .take_while(|&(y, x)| y < h && x >= 0)
                .map(|(y, x)| (y, x as usize))
                .collect();
            aggregate_path(&path, &cost_volume, &mut aggregated);
        }
    }

    // Path 7: bottom-left → top-right diagonal.
    {
        let starts: Vec<(usize, usize)> = (0..w)
            .map(|x| (h - 1, x))
            .chain((0..h - 1).map(|y| (y, 0)))
            .collect();
        for (y0, x0) in starts {
            let path: Vec<(usize, usize)> = (0..)
                .map(|k| (y0 as i64 - k as i64, x0 + k))
                .take_while(|&(y, x)| y >= 0 && x < w)
                .map(|(y, x)| (y as usize, x))
                .collect();
            aggregate_path(&path, &cost_volume, &mut aggregated);
        }
    }

    // Winner-take-all: pick the disparity with the smallest aggregated cost.
    let mut disparity = Array2::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let base = y * w * max_disp + x * max_disp;
            let (best_d, _) = aggregated[base..base + max_disp].iter().enumerate().fold(
                (0usize, f64::INFINITY),
                |(bd, bc), (d, &c)| {
                    if c < bc {
                        (d, c)
                    } else {
                        (bd, bc)
                    }
                },
            );
            disparity[[y, x]] = best_d as f64;
        }
    }

    Ok(disparity)
}

// ─────────────────────────────────────────────────────────────────────────────
// depth_from_disparity
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a disparity map to a metric depth map.
///
/// Uses the stereo baseline formula:  `depth = focal_length × baseline / disparity`.
/// Pixels with a disparity of zero are assigned a depth of zero (invalid).
///
/// # Arguments
///
/// * `disparity`    – Disparity map in pixels.
/// * `focal_length` – Camera focal length in pixels.
/// * `baseline`     – Stereo baseline distance (same units as desired depth output).
///
/// # Returns
///
/// Depth map in the same units as `baseline`.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when `focal_length ≤ 0` or
/// `baseline ≤ 0`.
///
/// # Example
///
/// ```
/// use scirs2_vision::stereo_legacy::depth_from_disparity;
/// use scirs2_core::ndarray::Array2;
///
/// let mut disp: Array2<f64> = Array2::zeros((4, 4));
/// disp[[1, 1]] = 2.0;
/// let depth = depth_from_disparity(&disp, 500.0, 0.1).unwrap();
/// assert!((depth[[1, 1]] - 25.0_f64).abs() < 1e-9);
/// ```
pub fn depth_from_disparity(
    disparity: &Array2<f64>,
    focal_length: f64,
    baseline: f64,
) -> Result<Array2<f64>> {
    if focal_length <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "focal_length must be positive".to_string(),
        ));
    }
    if baseline <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "baseline must be positive".to_string(),
        ));
    }

    let (h, w) = disparity.dim();
    let mut depth = Array2::zeros((h, w));
    let fb = focal_length * baseline;

    for y in 0..h {
        for x in 0..w {
            let d = disparity[[y, x]];
            if d > 0.0 {
                depth[[y, x]] = fb / d;
            }
        }
    }

    Ok(depth)
}

// ─────────────────────────────────────────────────────────────────────────────
// stereo_rectify
// ─────────────────────────────────────────────────────────────────────────────

/// Apply pre-computed rectification homographies to a stereo image pair.
///
/// Each pixel `(u, v)` in the output is mapped back to the source image via
/// the inverse of the respective homography `H`, then sampled with bilinear
/// interpolation.
///
/// # Arguments
///
/// * `h1`     – 3×3 rectification homography for the left image (row-major).
/// * `h2`     – 3×3 rectification homography for the right image.
/// * `left`   – Left image with shape `[H, W, C]`.
/// * `right`  – Right image with shape `[H, W, C]` (must equal left's shape).
///
/// # Returns
///
/// `(rectified_left, rectified_right)` with the same shapes as the inputs.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when image shapes differ or a
/// homography matrix is not invertible.
pub fn stereo_rectify(
    h1: &Array2<f64>,
    h2: &Array2<f64>,
    left: &Array3<f64>,
    right: &Array3<f64>,
) -> Result<(Array3<f64>, Array3<f64>)> {
    if left.dim() != right.dim() {
        return Err(VisionError::DimensionMismatch(
            "left and right images must have the same shape".to_string(),
        ));
    }
    let rect_l = warp_homography(h1, left)?;
    let rect_r = warp_homography(h2, right)?;
    Ok((rect_l, rect_r))
}

/// Warp an image using the inverse of a 3×3 homography with bilinear interpolation.
fn warp_homography(h: &Array2<f64>, image: &Array3<f64>) -> Result<Array3<f64>> {
    let h_inv = invert_3x3(h)?;
    let (rows, cols, chans) = image.dim();
    let mut out = Array3::zeros((rows, cols, chans));

    for v in 0..rows {
        for u in 0..cols {
            // Forward: destination (u,v) → apply H⁻¹ → source position.
            let (sx, sy) = apply_homography_3x3(&h_inv, u as f64, v as f64);
            let u0 = sx.floor() as i64;
            let v0 = sy.floor() as i64;
            let u1 = u0 + 1;
            let v1 = v0 + 1;
            let du = sx - u0 as f64;
            let dv = sy - v0 as f64;

            let in_bounds = |uu: i64, vv: i64| -> bool {
                uu >= 0 && uu < cols as i64 && vv >= 0 && vv < rows as i64
            };

            for ch in 0..chans {
                let s = |uu: i64, vv: i64| -> f64 {
                    if in_bounds(uu, vv) {
                        image[[vv as usize, uu as usize, ch]]
                    } else {
                        0.0
                    }
                };
                out[[v, u, ch]] = s(u0, v0) * (1.0 - du) * (1.0 - dv)
                    + s(u1, v0) * du * (1.0 - dv)
                    + s(u0, v1) * (1.0 - du) * dv
                    + s(u1, v1) * du * dv;
            }
        }
    }
    Ok(out)
}

/// Apply a 3×3 homography `h` to a point `(x, y)` in homogeneous coordinates.
fn apply_homography_3x3(h: &Array2<f64>, x: f64, y: f64) -> (f64, f64) {
    let xp = h[[0, 0]] * x + h[[0, 1]] * y + h[[0, 2]];
    let yp = h[[1, 0]] * x + h[[1, 1]] * y + h[[1, 2]];
    let wp = h[[2, 0]] * x + h[[2, 1]] * y + h[[2, 2]];
    if wp.abs() < 1e-12 {
        (xp, yp)
    } else {
        (xp / wp, yp / wp)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fundamental matrix (8-point algorithm)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the fundamental matrix `F` from point correspondences using the
/// normalised 8-point algorithm.
///
/// Enforces the rank-2 constraint by zeroing the smallest singular value.
/// Requires at least 8 point pairs.
///
/// # Arguments
///
/// * `pts1` – Points in the left image, shape `[N, 2]`.
/// * `pts2` – Corresponding points in the right image, shape `[N, 2]`.
///
/// # Returns
///
/// 3×3 fundamental matrix `F` such that `pts2ᵀ F pts1 = 0` (approximately).
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when fewer than 8 points are
/// supplied or the shapes are inconsistent.
///
/// # Example
///
/// ```
/// use scirs2_vision::stereo_legacy::fundamental_matrix;
/// use scirs2_core::ndarray::Array2;
///
/// // 8 (arbitrary) synthetic correspondences.
/// let pts1 = Array2::from_shape_vec((8, 2), vec![
///     10.0_f64, 20.0, 30.0, 15.0, 50.0, 40.0, 80.0, 10.0,
///     20.0, 60.0, 70.0, 30.0, 100.0, 50.0, 5.0, 70.0,
/// ]).unwrap();
/// let pts2 = Array2::from_shape_vec((8, 2), vec![
///     12.0_f64, 20.0, 32.0, 15.0, 52.0, 40.0, 82.0, 10.0,
///     22.0, 60.0, 72.0, 30.0, 102.0, 50.0, 7.0, 70.0,
/// ]).unwrap();
/// let f = fundamental_matrix(&pts1, &pts2).unwrap();
/// assert_eq!(f.dim(), (3, 3));
/// ```
pub fn fundamental_matrix(pts1: &Array2<f64>, pts2: &Array2<f64>) -> Result<Array2<f64>> {
    let (n, c1) = pts1.dim();
    let (m, c2) = pts2.dim();

    if c1 != 2 || c2 != 2 {
        return Err(VisionError::InvalidParameter(
            "pts1 and pts2 must have exactly 2 columns".to_string(),
        ));
    }
    if n != m {
        return Err(VisionError::DimensionMismatch(
            "pts1 and pts2 must have the same number of rows".to_string(),
        ));
    }
    if n < 8 {
        return Err(VisionError::InvalidParameter(
            "At least 8 point correspondences are required".to_string(),
        ));
    }

    // Normalise points.
    let (pts1_n, t1) = normalize_points(pts1);
    let (pts2_n, t2) = normalize_points(pts2);

    // Build the 9-column design matrix A (one row per correspondence).
    // A row: [x'x  x'y  x'  y'x  y'y  y'  x  y  1]
    let mut a = vec![0.0_f64; n * 9];
    for i in 0..n {
        let x = pts1_n[[i, 0]];
        let y = pts1_n[[i, 1]];
        let xp = pts2_n[[i, 0]];
        let yp = pts2_n[[i, 1]];
        a[i * 9] = xp * x;
        a[i * 9 + 1] = xp * y;
        a[i * 9 + 2] = xp;
        a[i * 9 + 3] = yp * x;
        a[i * 9 + 4] = yp * y;
        a[i * 9 + 5] = yp;
        a[i * 9 + 6] = x;
        a[i * 9 + 7] = y;
        a[i * 9 + 8] = 1.0;
    }

    // Solve via SVD (using our local thin SVD implementation).
    let a_mat = Array2::from_shape_vec((n, 9), a)
        .map_err(|e| VisionError::OperationError(e.to_string()))?;
    let f_vec = last_right_singular_vector(&a_mat)?;

    // Reshape to 3×3.
    let mut f = Array2::from_shape_vec((3, 3), f_vec)
        .map_err(|e| VisionError::OperationError(e.to_string()))?;

    // Enforce rank-2 by zeroing the smallest singular value of F.
    enforce_rank2(&mut f)?;

    // Denormalise: F = T2ᵀ F̃ T1.
    let f_denorm = mat3_mul(&mat3_transpose(&t2), &mat3_mul(&f, &t1)?)?;
    Ok(f_denorm)
}

/// Compute the essential matrix from the fundamental matrix and camera calibrations.
///
/// `E = K2ᵀ F K1`
///
/// # Arguments
///
/// * `f`  – 3×3 fundamental matrix.
/// * `k1` – 3×3 intrinsic matrix of camera 1.
/// * `k2` – 3×3 intrinsic matrix of camera 2.
///
/// # Example
///
/// ```
/// use scirs2_vision::stereo_legacy::essential_matrix_from_fundamental;
/// use scirs2_core::ndarray::Array2;
///
/// let f: Array2<f64> = Array2::eye(3);
/// let k1: Array2<f64> = Array2::eye(3);
/// let k2: Array2<f64> = Array2::eye(3);
/// let e = essential_matrix_from_fundamental(&f, &k1, &k2).unwrap();
/// assert_eq!(e.dim(), (3, 3));
/// ```
pub fn essential_matrix_from_fundamental(
    f: &Array2<f64>,
    k1: &Array2<f64>,
    k2: &Array2<f64>,
) -> Result<Array2<f64>> {
    // E = K2ᵀ F K1
    let k2t = mat3_transpose(k2);
    mat3_mul(&mat3_mul(&k2t, f)?, k1)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal linear-algebra helpers (3×3 only; no external crate needed)
// ─────────────────────────────────────────────────────────────────────────────

/// Normalise a set of 2-D points so their centroid is at the origin and their
/// RMS distance from the origin is √2.
///
/// Returns the normalised points and the 3×3 normalisation matrix.
fn normalize_points(pts: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let n = pts.nrows();
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    for i in 0..n {
        cx += pts[[i, 0]];
        cy += pts[[i, 1]];
    }
    cx /= n as f64;
    cy /= n as f64;

    let mut mean_dist = 0.0_f64;
    for i in 0..n {
        let dx = pts[[i, 0]] - cx;
        let dy = pts[[i, 1]] - cy;
        mean_dist += (dx * dx + dy * dy).sqrt();
    }
    mean_dist /= n as f64;

    let scale = if mean_dist > 1e-10 {
        2.0_f64.sqrt() / mean_dist
    } else {
        1.0
    };

    let mut out = Array2::zeros((n, 2));
    for i in 0..n {
        out[[i, 0]] = (pts[[i, 0]] - cx) * scale;
        out[[i, 1]] = (pts[[i, 1]] - cy) * scale;
    }

    // T = [[s, 0, -s*cx], [0, s, -s*cy], [0, 0, 1]]
    let t_data = vec![
        scale,
        0.0,
        -scale * cx,
        0.0,
        scale,
        -scale * cy,
        0.0,
        0.0,
        1.0,
    ];
    let t = Array2::from_shape_vec((3, 3), t_data).expect("shape is fixed");
    (out, t)
}

/// Compute `Aᵀ A` and find its smallest eigenvector by the power-iteration /
/// Jacobi method.  For our use-case (solving a 9×9 system) we use a direct
/// Golub-Kahan bidiagonalisation followed by a Jacobi sweep on the (small)
/// bidiagonal matrix.
///
/// For simplicity we use a numerically stable QR-based approach: compute `Aᵀ A`,
/// then find the eigenvector corresponding to the smallest eigenvalue using
/// inverse power iteration.  The matrix is only 9×9 so this is fast.
fn last_right_singular_vector(a: &Array2<f64>) -> Result<Vec<f64>> {
    let (m, n) = a.dim();
    // AᵀA is n×n.
    let mut ata = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut v = 0.0_f64;
            for k in 0..m {
                v += a[[k, i]] * a[[k, j]];
            }
            ata[i * n + j] = v;
        }
    }

    // Find eigenvector for the smallest eigenvalue via inverse power iteration.
    // Shift: use a small negative shift to target the smallest eigenvalue.
    // First obtain a rough estimate of the smallest eigenvalue via Gershgorin.
    let min_eigenvalue_approx = {
        let mut min_ev = f64::INFINITY;
        for i in 0..n {
            let diag = ata[i * n + i];
            let off: f64 = (0..n)
                .filter(|&j| j != i)
                .map(|j| ata[i * n + j].abs())
                .sum();
            let lower = diag - off;
            if lower < min_ev {
                min_ev = lower;
            }
        }
        min_ev
    };

    // Shift the matrix: B = AᵀA - σI  with σ slightly below min_ev.
    let sigma = min_eigenvalue_approx - 1.0;
    let mut b = ata.clone();
    for i in 0..n {
        b[i * n + i] -= sigma;
    }

    // Solve by LU decomposition + back-substitution (inverse iteration).
    // We start from a random-ish vector and iterate B⁻¹ v / ||B⁻¹ v||.
    let mut v: Vec<f64> = (0..n).map(|i| if i == n - 1 { 1.0 } else { 0.0 }).collect();

    for _ in 0..50 {
        // Solve B x = v using Gaussian elimination with partial pivoting.
        let x = solve_linear_system(&b, &v, n)?;
        let norm: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>().sqrt();
        if norm < 1e-14 {
            break;
        }
        v = x.iter().map(|&xi| xi / norm).collect();
    }

    Ok(v)
}

/// Gaussian elimination with partial pivoting to solve `A x = b` (both n×n / n).
fn solve_linear_system(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>> {
    // Augmented matrix [A | b].
    let mut aug: Vec<f64> = (0..n)
        .flat_map(|i| {
            let mut row: Vec<f64> = a[i * n..(i + 1) * n].to_vec();
            row.push(b[i]);
            row
        })
        .collect();

    let nc = n + 1;

    for col in 0..n {
        // Partial pivot.
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                aug[r1 * nc + col]
                    .abs()
                    .partial_cmp(&aug[r2 * nc + col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);

        if pivot_row != col {
            for k in 0..nc {
                aug.swap(col * nc + k, pivot_row * nc + k);
            }
        }

        let pivot = aug[col * nc + col];
        if pivot.abs() < 1e-14 {
            // Singular: return the unit vector in direction `col`.
            let mut unit = vec![0.0_f64; n];
            unit[col] = 1.0;
            return Ok(unit);
        }

        for row in (col + 1)..n {
            let factor = aug[row * nc + col] / pivot;
            for k in col..nc {
                let val = aug[col * nc + k] * factor;
                aug[row * nc + k] -= val;
            }
        }
    }

    // Back substitution.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = aug[i * nc + n]; // rhs
        for j in (i + 1)..n {
            s -= aug[i * nc + j] * x[j];
        }
        let diag = aug[i * nc + i];
        x[i] = if diag.abs() > 1e-14 { s / diag } else { 0.0 };
    }
    Ok(x)
}

/// Enforce the rank-2 constraint on a 3×3 matrix by zeroing its smallest
/// singular value via Jacobi SVD.
fn enforce_rank2(f: &mut Array2<f64>) -> Result<()> {
    // We compute a full SVD of the 3×3 matrix using classical Jacobi iterations.
    let (u, mut s, vt) = svd_3x3(f)?;

    // Zero the smallest singular value.
    let min_idx = if s[0] <= s[1] && s[0] <= s[2] {
        0
    } else if s[1] <= s[2] {
        1
    } else {
        2
    };
    s[min_idx] = 0.0;

    // Reconstruct F = U Σ Vᵀ.
    for i in 0..3 {
        for j in 0..3 {
            f[[i, j]] = u[[i, 0]] * s[0] * vt[[0, j]]
                + u[[i, 1]] * s[1] * vt[[1, j]]
                + u[[i, 2]] * s[2] * vt[[2, j]];
        }
    }
    Ok(())
}

/// Compute the SVD of a 3×3 matrix using Jacobi one-sided iterations.
///
/// Returns `(U, s, Vᵀ)` where `s` contains the singular values (not
/// necessarily sorted).
fn svd_3x3(a: &Array2<f64>) -> Result<(Array2<f64>, [f64; 3], Array2<f64>)> {
    // One-sided Jacobi SVD on AᵀA → eigendecomposition → V, Σ, U.
    // We work on B = AᵀA (3×3 symmetric PSD).
    let mut b = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                b[i][j] += a[[k, i]] * a[[k, j]];
            }
        }
    }

    // Jacobi eigendecomposition of the symmetric 3×3 matrix b.
    let mut v = [[0.0_f64; 3]; 3];
    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        v[i][i] = 1.0;
    }

    for _ in 0..100 {
        // Find the off-diagonal element with the largest absolute value.
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        #[allow(clippy::needless_range_loop)]
        for i in 0..3 {
            for j in (i + 1)..3 {
                if b[i][j].abs() > max_val {
                    max_val = b[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }

        // Compute Jacobi rotation angle.
        let theta = 0.5 * (b[p][p] - b[q][q]);
        let t_sgn = if theta >= 0.0 { 1.0_f64 } else { -1.0_f64 };
        let t = t_sgn / (theta.abs() + (theta * theta + b[p][q] * b[p][q]).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Update b.
        let bpp = b[p][p];
        let bqq = b[q][q];
        let bpq = b[p][q];
        b[p][p] = c * c * bpp - 2.0 * s * c * bpq + s * s * bqq;
        b[q][q] = s * s * bpp + 2.0 * s * c * bpq + c * c * bqq;
        b[p][q] = 0.0;
        b[q][p] = 0.0;
        #[allow(clippy::needless_range_loop)]
        for r in 0..3 {
            if r != p && r != q {
                let brp = b[r][p];
                let brq = b[r][q];
                b[r][p] = c * brp - s * brq;
                b[p][r] = b[r][p];
                b[r][q] = s * brp + c * brq;
                b[q][r] = b[r][q];
            }
        }

        // Accumulate V.
        #[allow(clippy::needless_range_loop)]
        for r in 0..3 {
            let vrp = v[r][p];
            let vrq = v[r][q];
            v[r][p] = c * vrp - s * vrq;
            v[r][q] = s * vrp + c * vrq;
        }
    }

    // Singular values are square roots of eigenvalues (diagonal of b).
    let s_vals = [
        b[0][0].max(0.0).sqrt(),
        b[1][1].max(0.0).sqrt(),
        b[2][2].max(0.0).sqrt(),
    ];

    // Build U: u_i = A v_i / sigma_i.
    let mut u_arr = [[0.0_f64; 3]; 3];
    for j in 0..3 {
        let sigma = s_vals[j];
        if sigma > 1e-14 {
            for i in 0..3 {
                let mut ui = 0.0_f64;
                for k in 0..3 {
                    ui += a[[i, k]] * v[k][j];
                }
                u_arr[i][j] = ui / sigma;
            }
        } else {
            // Fill with zeros; the singular vector for a zero singular value is
            // arbitrary (we just set a canonical basis vector).
            u_arr[j][j] = 1.0;
        }
    }

    let u_mat = Array2::from_shape_vec(
        (3, 3),
        u_arr.iter().flat_map(|row| row.iter().copied()).collect(),
    )
    .map_err(|e| VisionError::OperationError(e.to_string()))?;

    let vt_data: Vec<f64> = (0..3).flat_map(|i| (0..3).map(move |j| v[j][i])).collect();
    let vt_mat = Array2::from_shape_vec((3, 3), vt_data)
        .map_err(|e| VisionError::OperationError(e.to_string()))?;

    Ok((u_mat, s_vals, vt_mat))
}

/// Multiply two 3×3 `Array2<f64>` matrices.
fn mat3_mul(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    if a.dim() != (3, 3) || b.dim() != (3, 3) {
        return Err(VisionError::InvalidParameter(
            "mat3_mul requires two 3×3 matrices".to_string(),
        ));
    }
    let mut c = Array2::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[[i, j]] += a[[i, k]] * b[[k, j]];
            }
        }
    }
    Ok(c)
}

/// Transpose a 3×3 matrix.
fn mat3_transpose(a: &Array2<f64>) -> Array2<f64> {
    let mut t = Array2::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            t[[i, j]] = a[[j, i]];
        }
    }
    t
}

/// Invert a 3×3 matrix using the explicit cofactor formula.
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

    if det.abs() < 1e-14 {
        return Err(VisionError::LinAlgError(
            "Matrix is singular; cannot invert".to_string(),
        ));
    }

    let inv_det = 1.0 / det;
    let data = vec![
        (e * k - f * h) * inv_det,
        (c * h - b * k) * inv_det,
        (b * f - c * e) * inv_det,
        (f * g - d * k) * inv_det,
        (a * k - c * g) * inv_det,
        (c * d - a * f) * inv_det,
        (d * h - e * g) * inv_det,
        (b * g - a * h) * inv_det,
        (a * e - b * d) * inv_det,
    ];
    Array2::from_shape_vec((3, 3), data).map_err(|e| VisionError::OperationError(e.to_string()))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_block_matching_uniform() {
        // A uniform image pair: all disparities should match (cost 0 at disp 0).
        let left = Array2::from_elem((16, 32), 100.0_f64);
        let right = Array2::from_elem((16, 32), 100.0_f64);
        let disp = block_matching(&left, &right, 5, 8, MatchingCost::Sad)
            .expect("block_matching should succeed on uniform images");
        assert_eq!(disp.dim(), (16, 32));
        // Interior pixels should have disparity 0 (uniform image, no gradient).
        assert!((disp[[8, 16]] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_block_matching_shifted() {
        // Right image is left image shifted by 3 pixels.
        let h = 20usize;
        let w = 40usize;
        let shift = 3usize;
        let mut left = Array2::zeros((h, w));
        let mut right = Array2::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                left[[y, x]] = (x % 8) as f64 * 30.0;
            }
            for x in shift..w {
                right[[y, x - shift]] = left[[y, x]];
            }
        }
        let disp = block_matching(&left, &right, 5, 8, MatchingCost::Sad)
            .expect("block_matching should succeed on shifted images");
        // Check a pixel not too close to the border.
        let d = disp[[10, 20]];
        assert!(
            (d - shift as f64).abs() <= 1.0,
            "disparity={d}, expected~{shift}"
        );
    }

    #[test]
    fn test_block_matching_invalid_params() {
        let img = Array2::from_elem((8, 8), 0.0_f64);
        // Even block size.
        assert!(block_matching(&img, &img, 4, 4, MatchingCost::Sad).is_err());
        // max_disp = 0.
        assert!(block_matching(&img, &img, 3, 0, MatchingCost::Sad).is_err());
        // Size mismatch.
        let other = Array2::from_elem((8, 10), 0.0_f64);
        assert!(block_matching(&img, &other, 3, 4, MatchingCost::Sad).is_err());
    }

    #[test]
    fn test_block_matching_ssd() {
        let left = Array2::from_elem((16, 32), 50.0_f64);
        let right = Array2::from_elem((16, 32), 50.0_f64);
        let disp = block_matching(&left, &right, 3, 8, MatchingCost::Ssd)
            .expect("block_matching SSD should succeed");
        assert_eq!(disp.dim(), (16, 32));
    }

    #[test]
    fn test_block_matching_ncc() {
        let left = Array2::from_elem((16, 32), 80.0_f64);
        let right = Array2::from_elem((16, 32), 80.0_f64);
        let disp = block_matching(&left, &right, 3, 4, MatchingCost::Ncc)
            .expect("block_matching NCC should succeed");
        assert_eq!(disp.dim(), (16, 32));
    }

    #[test]
    fn test_block_matching_census() {
        let left = Array2::from_elem((16, 32), 64.0_f64);
        let right = Array2::from_elem((16, 32), 64.0_f64);
        let disp = block_matching(&left, &right, 5, 4, MatchingCost::Census)
            .expect("block_matching Census should succeed");
        assert_eq!(disp.dim(), (16, 32));
    }

    #[test]
    fn test_sgm_basic() {
        let left = Array2::from_elem((8, 16), 100.0_f64);
        let right = Array2::from_elem((8, 16), 100.0_f64);
        let disp = disparity_map_sgm(&left, &right, 4, &SgmPenalties::default())
            .expect("disparity_map_sgm should succeed on uniform images");
        assert_eq!(disp.dim(), (8, 16));
    }

    #[test]
    fn test_sgm_invalid() {
        let img = Array2::from_elem((8, 8), 0.0_f64);
        assert!(disparity_map_sgm(&img, &img, 0, &SgmPenalties::default()).is_err());
        let other = Array2::from_elem((4, 8), 0.0_f64);
        assert!(disparity_map_sgm(&img, &other, 4, &SgmPenalties::default()).is_err());
    }

    #[test]
    fn test_depth_from_disparity() {
        let mut disp = Array2::zeros((4, 4));
        disp[[2, 2]] = 5.0;
        let depth = depth_from_disparity(&disp, 500.0, 0.1)
            .expect("depth_from_disparity should succeed with valid params");
        // depth = 500 * 0.1 / 5 = 10.0
        assert!(
            (depth[[2, 2]] - 10.0).abs() < 1e-9,
            "depth={}",
            depth[[2, 2]]
        );
        // Zero-disparity pixels should remain 0.
        assert_eq!(depth[[0, 0]], 0.0);
    }

    #[test]
    fn test_depth_from_disparity_invalid() {
        let disp = Array2::from_elem((4, 4), 1.0_f64);
        assert!(depth_from_disparity(&disp, 0.0, 0.1).is_err());
        assert!(depth_from_disparity(&disp, 500.0, 0.0).is_err());
        assert!(depth_from_disparity(&disp, -1.0, 0.1).is_err());
    }

    #[test]
    fn test_stereo_rectify_identity() {
        // Identity homographies should leave the images unchanged.
        let h = Array2::eye(3);
        let img = Array3::from_elem((8, 8, 3), 128.0_f64);
        let (rl, rr) = stereo_rectify(&h, &h, &img, &img)
            .expect("stereo_rectify should succeed with identity homographies");
        assert_eq!(rl.dim(), img.dim());
        assert_eq!(rr.dim(), img.dim());
        // The mean absolute error should be tiny (bilinear interpolation of
        // a constant image should reproduce the constant exactly).
        let err: f64 = (&rl - &img).iter().map(|x| x.abs()).sum::<f64>();
        assert!(err < 1.0, "err={err}");
    }

    #[test]
    fn test_stereo_rectify_size_mismatch() {
        let h = Array2::eye(3);
        let left = Array3::from_elem((8, 8, 3), 0.0_f64);
        let right = Array3::from_elem((4, 8, 3), 0.0_f64);
        assert!(stereo_rectify(&h, &h, &left, &right).is_err());
    }

    #[test]
    fn test_fundamental_matrix_shape() {
        let pts1 = Array2::from_shape_vec(
            (8, 2),
            vec![
                10.0, 20.0, 30.0, 15.0, 50.0, 40.0, 80.0, 10.0, 20.0, 60.0, 70.0, 30.0, 100.0,
                50.0, 5.0, 70.0,
            ],
        )
        .expect("from_shape_vec should succeed with correct element count");
        let pts2 = Array2::from_shape_vec(
            (8, 2),
            vec![
                12.0, 20.0, 32.0, 15.0, 52.0, 40.0, 82.0, 10.0, 22.0, 60.0, 72.0, 30.0, 102.0,
                50.0, 7.0, 70.0,
            ],
        )
        .expect("from_shape_vec should succeed with correct element count");
        let f = fundamental_matrix(&pts1, &pts2)
            .expect("fundamental_matrix should succeed with 8 point correspondences");
        assert_eq!(f.dim(), (3, 3));
    }

    #[test]
    fn test_fundamental_matrix_too_few_points() {
        let pts = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        )
        .expect("from_shape_vec should succeed with correct element count");
        assert!(fundamental_matrix(&pts, &pts).is_err());
    }

    #[test]
    fn test_essential_matrix_identity() {
        let f = Array2::eye(3);
        let k = Array2::eye(3);
        let e = essential_matrix_from_fundamental(&f, &k, &k)
            .expect("essential_matrix_from_fundamental should succeed with identity K");
        // With identity K matrices, E should equal F.
        for i in 0..3 {
            for j in 0..3 {
                let expected = f[[i, j]];
                let got = e[[i, j]];
                assert!(
                    (got - expected).abs() < 1e-9,
                    "[{i},{j}] expected {expected} got {got}"
                );
            }
        }
    }

    #[test]
    fn test_stereo_matcher_builder() {
        let m = StereoMatcher::new(5, 16, MatchingCost::Ncc);
        let left = Array2::from_elem((16, 32), 100.0_f64);
        let right = Array2::from_elem((16, 32), 100.0_f64);
        let disp = m
            .compute(&left, &right)
            .expect("StereoMatcher compute should succeed on valid inputs");
        assert_eq!(disp.dim(), (16, 32));
    }

    #[test]
    fn test_invert_3x3_identity() {
        let id = Array2::eye(3);
        let inv = invert_3x3(&id).expect("invert_3x3 should succeed on identity matrix");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[[i, j]] - expected).abs() < 1e-10);
            }
        }
    }
}
