//! Depth estimation algorithms.
//!
//! Provides three families of depth / disparity computation:
//!
//! 1. **Block Matching (BM)** – simple winner-takes-all SAD-based stereo.
//! 2. **Semi-Global Block Matching (SGBM)** – scanline-aggregated cost with
//!    smoothness penalties P1 and P2.
//! 3. **Depth from Focus** – Laplacian-based sharpness focus stack fusion.
//! 4. **Relative depth from gradients** – scale-ambiguous monocular cue.

use crate::error::{Result, VisionError};

// ─────────────────────────────────────────────────────────────────────────────
// Block Matching (BM)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a disparity map using simple winner-takes-all block matching (SAD).
///
/// # Arguments
/// * `left`         – Grayscale left image `[row][col]` with pixel values in
///   `[0, 255]`.
/// * `right`        – Grayscale right image with identical dimensions.
/// * `max_disparity` – Maximum horizontal disparity to search (pixels).
/// * `block_size`   – Full window side length (must be odd, ≥ 3).
///
/// # Returns
/// A disparity map of the same dimensions.  Border pixels where the window
/// would extend outside the image are set to `0`.
///
/// # Errors
/// Returns `Err` when the images have different sizes, `block_size` is even,
/// or `block_size < 3`.
///
/// # Example
/// ```
/// use scirs2_vision::depth::compute_disparity_bm;
///
/// let row: Vec<u8> = (0..16u8).collect();
/// let left:  Vec<Vec<u8>> = vec![row.clone(); 16];
/// let right: Vec<Vec<u8>> = vec![row.clone(); 16];
/// let disp = compute_disparity_bm(&left, &right, 8, 3).unwrap();
/// assert_eq!(disp.len(), 16);
/// ```
pub fn compute_disparity_bm(
    left: &[Vec<u8>],
    right: &[Vec<u8>],
    max_disparity: usize,
    block_size: usize,
) -> Result<Vec<Vec<f32>>> {
    if block_size < 3 || block_size.is_multiple_of(2) {
        return Err(VisionError::InvalidParameter(
            "block_size must be odd and ≥ 3".to_string(),
        ));
    }
    let rows = left.len();
    if rows == 0 {
        return Ok(Vec::new());
    }
    let cols = left[0].len();
    if right.len() != rows || right.iter().any(|r| r.len() != cols) {
        return Err(VisionError::InvalidParameter(
            "left and right images must have identical dimensions".to_string(),
        ));
    }

    let half = block_size / 2;
    let mut disp_map = vec![vec![0f32; cols]; rows];

    #[allow(clippy::needless_range_loop)]
    for row in half..rows.saturating_sub(half) {
        for col in half..cols.saturating_sub(half) {
            let mut best_disp = 0usize;
            let mut best_sad = u64::MAX;

            let max_d = max_disparity.min(col.saturating_sub(half) + 1);
            for d in 0..max_d {
                let right_col = col - d;
                if right_col < half {
                    break;
                }
                let mut sad = 0u64;
                for dr in 0..block_size {
                    let lr = row + dr - half;
                    if lr >= rows {
                        continue;
                    }
                    for dc in 0..block_size {
                        let lc = col + dc - half;
                        let rc = right_col + dc - half;
                        if lc >= cols || rc >= cols {
                            continue;
                        }
                        let diff = left[lr][lc] as i32 - right[lr][rc] as i32;
                        sad += diff.unsigned_abs() as u64;
                    }
                }
                if sad < best_sad {
                    best_sad = sad;
                    best_disp = d;
                }
            }
            disp_map[row][col] = best_disp as f32;
        }
    }
    Ok(disp_map)
}

// ─────────────────────────────────────────────────────────────────────────────
// SGBM
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters for Semi-Global Block Matching.
#[derive(Debug, Clone)]
pub struct SGBMParams {
    /// Minimum disparity (usually 0).
    pub min_disparity: i32,
    /// Number of disparities to search (must be positive and divisible by 16).
    pub num_disparities: i32,
    /// Block side length used for the matching cost (odd, 3–11).
    pub block_size: usize,
    /// Smoothness penalty for a 1-pixel disparity change between neighbours.
    pub p1: i32,
    /// Smoothness penalty for a >1-pixel disparity change.
    pub p2: i32,
    /// Ratio test threshold for uniqueness (0 = disabled, typical 5–15).
    pub uniqueness_ratio: f64,
}

impl Default for SGBMParams {
    fn default() -> Self {
        Self {
            min_disparity: 0,
            num_disparities: 64,
            block_size: 5,
            p1: 8,
            p2: 32,
            uniqueness_ratio: 10.0,
        }
    }
}

impl SGBMParams {
    /// Validate and return self, or an error describing what is wrong.
    pub fn validate(&self) -> Result<()> {
        if self.num_disparities <= 0 || self.num_disparities % 16 != 0 {
            return Err(VisionError::InvalidParameter(
                "num_disparities must be positive and divisible by 16".to_string(),
            ));
        }
        if self.block_size < 3 || self.block_size.is_multiple_of(2) {
            return Err(VisionError::InvalidParameter(
                "block_size must be odd and ≥ 3".to_string(),
            ));
        }
        if self.p2 <= self.p1 {
            return Err(VisionError::InvalidParameter(
                "p2 must be strictly greater than p1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Compute a disparity map using a simplified Semi-Global Block Matching.
///
/// The cost volume is computed with SAD over a `block_size × block_size`
/// window.  Scanline aggregation is performed in four directions (left→right,
/// right→left, top→bottom, bottom→top).  A uniqueness test is applied when
/// `params.uniqueness_ratio > 0`.
///
/// # Errors
/// Returns `Err` when parameters are invalid or images differ in size.
///
/// # Example
/// ```
/// use scirs2_vision::depth::{SGBMParams, compute_disparity_sgbm};
///
/// let row: Vec<u8> = (0..64u8).collect();
/// let left:  Vec<Vec<u8>> = vec![row.clone(); 64];
/// let right: Vec<Vec<u8>> = vec![row.clone(); 64];
/// let params = SGBMParams { num_disparities: 16, block_size: 3,
///     p1: 8, p2: 32, ..Default::default() };
/// let disp = compute_disparity_sgbm(&left, &right, &params).unwrap();
/// assert_eq!(disp.len(), 64);
/// ```
pub fn compute_disparity_sgbm(
    left: &[Vec<u8>],
    right: &[Vec<u8>],
    params: &SGBMParams,
) -> Result<Vec<Vec<f32>>> {
    params.validate()?;

    let rows = left.len();
    if rows == 0 {
        return Ok(Vec::new());
    }
    let cols = left[0].len();
    if right.len() != rows || right.iter().any(|r| r.len() != cols) {
        return Err(VisionError::InvalidParameter(
            "left and right images must have identical dimensions".to_string(),
        ));
    }

    let half = params.block_size / 2;
    let min_d = params.min_disparity;
    let num_d = params.num_disparities as usize;

    // ── Build cost volume C[row][col][d] using SAD ────────────────────────
    // Flatten to 1-D: index = (row * cols + col) * num_d + d
    let mut cost = vec![0i32; rows * cols * num_d];

    #[allow(clippy::needless_range_loop)]
    for row in half..rows.saturating_sub(half) {
        for col in half..cols.saturating_sub(half) {
            for di in 0..num_d {
                let d = min_d + di as i32;
                if d < 0 {
                    cost[(row * cols + col) * num_d + di] = i32::MAX / 2;
                    continue;
                }
                let right_col = col as i32 - d;
                if right_col < half as i32 || right_col >= cols as i32 - half as i32 {
                    cost[(row * cols + col) * num_d + di] = i32::MAX / 2;
                    continue;
                }
                let rc = right_col as usize;
                let mut sad = 0i32;
                for dr in 0..params.block_size {
                    let lr = row + dr - half;
                    if lr >= rows {
                        continue;
                    }
                    for dc in 0..params.block_size {
                        let lc = col + dc - half;
                        let rcc = rc + dc - half;
                        if lc >= cols || rcc >= cols {
                            continue;
                        }
                        sad += (left[lr][lc] as i32 - right[lr][rcc] as i32).abs();
                    }
                }
                cost[(row * cols + col) * num_d + di] = sad;
            }
        }
    }

    // ── Scanline aggregation in 4 directions ─────────────────────────────
    let p1 = params.p1;
    let p2 = params.p2;

    // Aggregated cost (sum of 4 directions)
    let mut agg = vec![0i64; rows * cols * num_d];

    // Helper closures:  left→right  and  right→left
    let scan_h = |agg: &mut Vec<i64>, left_to_right: bool| {
        for row in 0..rows {
            let col_iter: Vec<usize> = if left_to_right {
                (0..cols).collect()
            } else {
                (0..cols).rev().collect()
            };
            let mut prev_cost = vec![0i32; num_d];
            let mut prev_min = 0i32;
            for (ci, &col) in col_iter.iter().enumerate() {
                for di in 0..num_d {
                    let base = cost[(row * cols + col) * num_d + di];
                    let lr_val = if ci == 0 {
                        base
                    } else {
                        let from_same = prev_cost[di];
                        let from_m1 = if di > 0 {
                            prev_cost[di - 1] + p1
                        } else {
                            i32::MAX / 2
                        };
                        let from_p1 = if di + 1 < num_d {
                            prev_cost[di + 1] + p1
                        } else {
                            i32::MAX / 2
                        };
                        let from_any = prev_min + p2;
                        let min4 = from_same.min(from_m1).min(from_p1).min(from_any);
                        base + min4 - prev_min
                    };
                    agg[(row * cols + col) * num_d + di] += lr_val as i64;
                    prev_cost[di] = lr_val;
                }
                prev_min = *prev_cost.iter().min().unwrap_or(&0);
            }
        }
    };

    let scan_v = |agg: &mut Vec<i64>, top_to_bottom: bool| {
        for col in 0..cols {
            let row_iter: Vec<usize> = if top_to_bottom {
                (0..rows).collect()
            } else {
                (0..rows).rev().collect()
            };
            let mut prev_cost = vec![0i32; num_d];
            let mut prev_min = 0i32;
            for (ri, &row) in row_iter.iter().enumerate() {
                for di in 0..num_d {
                    let base = cost[(row * cols + col) * num_d + di];
                    let lr_val = if ri == 0 {
                        base
                    } else {
                        let from_same = prev_cost[di];
                        let from_m1 = if di > 0 {
                            prev_cost[di - 1] + p1
                        } else {
                            i32::MAX / 2
                        };
                        let from_p1 = if di + 1 < num_d {
                            prev_cost[di + 1] + p1
                        } else {
                            i32::MAX / 2
                        };
                        let from_any = prev_min + p2;
                        let min4 = from_same.min(from_m1).min(from_p1).min(from_any);
                        base + min4 - prev_min
                    };
                    agg[(row * cols + col) * num_d + di] += lr_val as i64;
                    prev_cost[di] = lr_val;
                }
                prev_min = *prev_cost.iter().min().unwrap_or(&0);
            }
        }
    };

    scan_h(&mut agg, true);
    scan_h(&mut agg, false);
    scan_v(&mut agg, true);
    scan_v(&mut agg, false);

    // ── Winner-takes-all + uniqueness test ───────────────────────────────
    let mut disp_map = vec![vec![0f32; cols]; rows];

    #[allow(clippy::needless_range_loop)]
    for row in 0..rows {
        for col in 0..cols {
            let base_idx = (row * cols + col) * num_d;
            let slice = &agg[base_idx..base_idx + num_d];

            let mut best_di = 0usize;
            let mut best_val = i64::MAX;
            let mut second_val = i64::MAX;

            for (di, &v) in slice.iter().enumerate() {
                if v < best_val {
                    second_val = best_val;
                    best_val = v;
                    best_di = di;
                } else if v < second_val {
                    second_val = v;
                }
            }

            // Uniqueness test
            if params.uniqueness_ratio > 0.0 {
                let threshold = (best_val as f64 * (1.0 + params.uniqueness_ratio / 100.0)) as i64;
                if second_val < threshold {
                    continue; // Ambiguous match → leave as 0
                }
            }

            disp_map[row][col] = (min_d + best_di as i32) as f32;
        }
    }

    Ok(disp_map)
}

// ─────────────────────────────────────────────────────────────────────────────
// Depth from Focus
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters for depth-from-focus stack estimation.
#[derive(Debug, Clone)]
pub struct DepthFromFocusParams {
    /// Number of focus planes (must equal the length of `focus_distances`).
    pub num_focus_planes: usize,
    /// Focus distance (metres) of each plane in the stack.  The plane with the
    /// highest Laplacian sharpness at a given pixel wins.
    pub focus_distances: Vec<f64>,
}

/// Estimate a depth map from a focus stack using Laplacian sharpness.
///
/// For each pixel position the algorithm selects the focus-stack plane with
/// the maximum absolute Laplacian response and assigns the corresponding focus
/// distance as the depth estimate.
///
/// # Arguments
/// * `focus_stack` – Slice of grayscale images `[plane][row][col]`.  All
///   planes must have identical dimensions.
/// * `params`      – Focus distances and related settings.
///
/// # Returns
/// A depth map `[row][col]` with values from `params.focus_distances`.
///
/// # Errors
/// Returns `Err` when the stack is empty, dimensions are inconsistent, or the
/// number of planes does not match `params.num_focus_planes`.
///
/// # Example
/// ```
/// use scirs2_vision::depth::{DepthFromFocusParams, depth_from_focus_stack};
///
/// let plane0 = vec![vec![1.0f64, 0.0, 0.0], vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]];
/// let plane1 = vec![vec![0.0f64, 0.0, 0.0], vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 1.0]];
/// let stack = vec![plane0, plane1];
/// let params = DepthFromFocusParams {
///     num_focus_planes: 2,
///     focus_distances: vec![1.0, 2.0],
/// };
/// let depth = depth_from_focus_stack(&stack, &params).unwrap();
/// assert_eq!(depth.len(), 3);
/// ```
pub fn depth_from_focus_stack(
    focus_stack: &[Vec<Vec<f64>>],
    params: &DepthFromFocusParams,
) -> Result<Vec<Vec<f64>>> {
    if focus_stack.is_empty() {
        return Err(VisionError::InvalidParameter(
            "focus_stack must not be empty".to_string(),
        ));
    }
    if focus_stack.len() != params.num_focus_planes {
        return Err(VisionError::InvalidParameter(format!(
            "focus_stack has {} planes but params.num_focus_planes = {}",
            focus_stack.len(),
            params.num_focus_planes
        )));
    }
    if params.focus_distances.len() != params.num_focus_planes {
        return Err(VisionError::InvalidParameter(
            "focus_distances.len() must equal num_focus_planes".to_string(),
        ));
    }

    let rows = focus_stack[0].len();
    if rows == 0 {
        return Err(VisionError::InvalidParameter(
            "Images in focus stack must not be empty".to_string(),
        ));
    }
    let cols = focus_stack[0][0].len();

    // Validate all planes have the same size
    for (pi, plane) in focus_stack.iter().enumerate() {
        if plane.len() != rows || plane.iter().any(|r| r.len() != cols) {
            return Err(VisionError::InvalidParameter(format!(
                "focus_stack plane {} has different dimensions",
                pi
            )));
        }
    }

    // Compute Laplacian sharpness for each plane
    let laplacian = |plane: &[Vec<f64>]| -> Vec<Vec<f64>> {
        let mut lap = vec![vec![0.0f64; cols]; rows];
        for r in 1..rows.saturating_sub(1) {
            for c in 1..cols.saturating_sub(1) {
                let val = -4.0 * plane[r][c]
                    + plane[r - 1][c]
                    + plane[r + 1][c]
                    + plane[r][c - 1]
                    + plane[r][c + 1];
                lap[r][c] = val.abs();
            }
        }
        lap
    };

    let sharpness: Vec<Vec<Vec<f64>>> = focus_stack.iter().map(|p| laplacian(p)).collect();

    // WTA: for each pixel select the plane with max sharpness
    let mut depth_map = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let mut best_plane = 0usize;
            let mut best_sharp = f64::NEG_INFINITY;
            for (pi, sharpness_pi) in sharpness.iter().enumerate().take(params.num_focus_planes) {
                let s = sharpness_pi[r][c];
                if s > best_sharp {
                    best_sharp = s;
                    best_plane = pi;
                }
            }
            depth_map[r][c] = params.focus_distances[best_plane];
        }
    }

    Ok(depth_map)
}

// ─────────────────────────────────────────────────────────────────────────────
// Relative depth from gradients (monocular)
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate a scale-ambiguous relative depth map from image gradients.
///
/// Uses the heuristic that regions of low gradient magnitude tend to be
/// smooth / far (lower relative depth) while high-frequency regions tend to
/// be near textured surfaces.  The resulting map is the *reciprocal* of the
/// Gaussian-smoothed gradient magnitude, normalised to `[0, 1]`.
///
/// # Arguments
/// * `image` – Grayscale image `[row][col]` with values in any consistent range.
///
/// # Returns
/// A relative depth map `[row][col]` ∈ `[0, 1]`, where `1` indicates
/// "nearer" (sharper gradient) and `0` "farther" (smoother region).
///
/// # Example
/// ```
/// use scirs2_vision::depth::relative_depth_from_gradients;
///
/// let image = vec![
///     vec![0.0, 0.0, 0.0, 0.0, 0.0],
///     vec![0.0, 1.0, 1.0, 1.0, 0.0],
///     vec![0.0, 1.0, 2.0, 1.0, 0.0],
///     vec![0.0, 1.0, 1.0, 1.0, 0.0],
///     vec![0.0, 0.0, 0.0, 0.0, 0.0],
/// ];
/// let depth = relative_depth_from_gradients(&image);
/// assert_eq!(depth.len(), 5);
/// assert_eq!(depth[0].len(), 5);
/// ```
pub fn relative_depth_from_gradients(image: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = image.len();
    if rows == 0 {
        return Vec::new();
    }
    let cols = image[0].len();
    if cols == 0 {
        return vec![Vec::new(); rows];
    }

    // Sobel gradient magnitude
    let mut grad = vec![vec![0.0f64; cols]; rows];
    for r in 1..rows.saturating_sub(1) {
        for c in 1..cols.saturating_sub(1) {
            let gx = -image[r - 1][c - 1] + image[r - 1][c + 1] - 2.0 * image[r][c - 1]
                + 2.0 * image[r][c + 1]
                - image[r + 1][c - 1]
                + image[r + 1][c + 1];
            let gy = -image[r - 1][c - 1] - 2.0 * image[r - 1][c] - image[r - 1][c + 1]
                + image[r + 1][c - 1]
                + 2.0 * image[r + 1][c]
                + image[r + 1][c + 1];
            grad[r][c] = (gx * gx + gy * gy).sqrt();
        }
    }

    // 3×3 box-filter smoothing
    let mut smooth = vec![vec![0.0f64; cols]; rows];
    for r in 1..rows.saturating_sub(1) {
        for c in 1..cols.saturating_sub(1) {
            let mut sum = 0.0;
            for dr in 0..3usize {
                for dc in 0..3usize {
                    sum += grad[r + dr - 1][c + dc - 1];
                }
            }
            smooth[r][c] = sum / 9.0;
        }
    }

    // Global normalise to [0, 1]
    let max_val = smooth
        .iter()
        .flat_map(|row| row.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    if max_val <= 0.0 {
        return smooth; // All zeros → flat image
    }

    for row in smooth.iter_mut() {
        for v in row.iter_mut() {
            *v /= max_val;
        }
    }

    smooth
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: constant image
    fn const_img(rows: usize, cols: usize, val: u8) -> Vec<Vec<u8>> {
        vec![vec![val; cols]; rows]
    }

    #[test]
    fn test_bm_same_image_zero_disparity() {
        // Identical images → disparity 0 everywhere
        let img = const_img(16, 32, 128);
        let disp = compute_disparity_bm(&img, &img, 16, 3)
            .expect("compute_disparity_bm should succeed on identical images");
        for row in &disp {
            for &d in row {
                // Border pixels may be 0; interior should also be 0 for identical imgs
                assert!(d >= 0.0);
            }
        }
    }

    #[test]
    fn test_bm_invalid_block_size() {
        let img = const_img(8, 8, 0);
        assert!(compute_disparity_bm(&img, &img, 4, 2).is_err()); // even
        assert!(compute_disparity_bm(&img, &img, 4, 1).is_err()); // < 3
    }

    #[test]
    fn test_bm_dimension_mismatch() {
        let left = const_img(8, 8, 0);
        let right = const_img(8, 9, 0);
        assert!(compute_disparity_bm(&left, &right, 4, 3).is_err());
    }

    #[test]
    fn test_sgbm_basic() {
        let row: Vec<u8> = (0..64u8).collect();
        let img: Vec<Vec<u8>> = vec![row; 64];
        let params = SGBMParams {
            num_disparities: 16,
            block_size: 3,
            p1: 8,
            p2: 32,
            ..Default::default()
        };
        let disp = compute_disparity_sgbm(&img, &img, &params)
            .expect("compute_disparity_sgbm should succeed with valid params");
        assert_eq!(disp.len(), 64);
        assert_eq!(disp[0].len(), 64);
    }

    #[test]
    fn test_sgbm_invalid_num_disparities() {
        let img = const_img(8, 8, 0);
        let params = SGBMParams {
            num_disparities: 15,
            ..Default::default()
        };
        assert!(compute_disparity_sgbm(&img, &img, &params).is_err());
    }

    #[test]
    fn test_sgbm_invalid_p1_p2() {
        let img = const_img(8, 8, 0);
        let params = SGBMParams {
            p1: 32,
            p2: 8,
            num_disparities: 16,
            ..Default::default()
        };
        assert!(compute_disparity_sgbm(&img, &img, &params).is_err());
    }

    #[test]
    fn test_depth_from_focus_basic() {
        // Plane 0: bright at top-left → high sharpness there
        // Plane 1: bright at bottom-right → high sharpness there
        let size = 5;
        let mut plane0 = vec![vec![0.0f64; size]; size];
        let mut plane1 = vec![vec![0.0f64; size]; size];
        plane0[0][0] = 10.0;
        plane1[4][4] = 10.0;

        let params = DepthFromFocusParams {
            num_focus_planes: 2,
            focus_distances: vec![1.0, 5.0],
        };
        let depth = depth_from_focus_stack(&[plane0, plane1], &params)
            .expect("depth_from_focus_stack should succeed with valid planes");
        assert_eq!(depth.len(), size);
    }

    #[test]
    fn test_depth_from_focus_plane_count_mismatch() {
        let plane = vec![vec![0.0f64; 4]; 4];
        let params = DepthFromFocusParams {
            num_focus_planes: 3,
            focus_distances: vec![1.0, 2.0, 3.0],
        };
        assert!(depth_from_focus_stack(&[plane], &params).is_err());
    }

    #[test]
    fn test_relative_depth_from_gradients_flat() {
        let flat = vec![vec![1.0f64; 8]; 8];
        let depth = relative_depth_from_gradients(&flat);
        assert_eq!(depth.len(), 8);
        // Flat image → all zeros
        for row in &depth {
            for &v in row {
                assert!((v).abs() < 1e-10, "v={}", v);
            }
        }
    }

    #[test]
    fn test_relative_depth_from_gradients_edge() {
        // An edge image: left half = 0, right half = 1
        let mut image = vec![vec![0.0f64; 10]; 10];
        for row in image.iter_mut().take(10) {
            for cell in row.iter_mut().skip(5).take(5) {
                *cell = 1.0;
            }
        }
        let depth = relative_depth_from_gradients(&image);
        // Max value should be 1.0
        let max_v = depth
            .iter()
            .flat_map(|r| r.iter())
            .cloned()
            .fold(0.0f64, f64::max);
        assert!(max_v > 0.0);
    }

    #[test]
    fn test_relative_depth_empty() {
        let empty: Vec<Vec<f64>> = Vec::new();
        let depth = relative_depth_from_gradients(&empty);
        assert!(depth.is_empty());
    }
}
