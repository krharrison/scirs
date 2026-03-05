//! Disparity map computation algorithms.
//!
//! Implements Block Matching (BM) and Semi-Global Matching (SGM/SGM2) for
//! dense stereo correspondence from rectified image pairs.

use crate::error::VisionError;

// ─────────────────────────────────────────────────────────────────────────────
// DisparityMap
// ─────────────────────────────────────────────────────────────────────────────

/// Per-pixel horizontal displacement (disparity) between a rectified stereo pair.
///
/// Positive disparity means the left-image feature appears to the right of the
/// corresponding right-image feature (the standard convention for a left-camera
/// coordinate frame).
#[derive(Debug, Clone)]
pub struct DisparityMap {
    /// Row-major disparity values — `NaN` / 0.0 marks invalid pixels.
    pub data: Vec<f32>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Inclusive minimum disparity searched during matching.
    pub min_disp: i32,
    /// Inclusive maximum disparity searched during matching.
    pub max_disp: i32,
}

impl DisparityMap {
    /// Create a zero-filled disparity map.
    pub fn new(width: usize, height: usize, min_disp: i32, max_disp: i32) -> Self {
        Self {
            data: vec![0.0_f32; width * height],
            width,
            height,
            min_disp,
            max_disp,
        }
    }

    /// Read disparity at `(row, col)`.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.width + col]
    }

    /// Write disparity at `(row, col)`.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        self.data[row * self.width + col] = val;
    }

    /// Convert disparity values to metric depth.
    ///
    /// Uses the thin-lens relation:
    /// ```text
    /// depth = baseline × focal_length / disparity
    /// ```
    /// Pixels with disparity ≤ 0 are mapped to 0.
    pub fn to_depth(&self, baseline: f32, focal_length: f32) -> Vec<f32> {
        self.data
            .iter()
            .map(|&d| {
                if d > 0.0 {
                    baseline * focal_length / d
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Number of pixels with valid (> 0) disparity.
    pub fn valid_count(&self) -> usize {
        self.data.iter().filter(|&&d| d > 0.0).count()
    }

    /// Sub-pixel refinement via parabolic fit around the best-matching disparity.
    ///
    /// Modifies the disparity map in-place.  Requires a pre-computed
    /// per-pixel cost slice (already aggregated); the refined offset is
    /// clamped to ±0.5.
    pub fn apply_subpixel_refinement(&mut self, cost_volume: &[f32], n_disp: usize) {
        for row in 0..self.height {
            for col in 0..self.width {
                let d = self.get(row, col) as i32 - self.min_disp;
                if d < 1 || d as usize + 1 >= n_disp {
                    continue;
                }
                let base = (row * self.width + col) * n_disp;
                let c_m = cost_volume[base + d as usize - 1];
                let c_0 = cost_volume[base + d as usize];
                let c_p = cost_volume[base + d as usize + 1];
                let denom = c_m - 2.0 * c_0 + c_p;
                if denom.abs() > 1e-6 {
                    let delta = 0.5 * (c_m - c_p) / denom;
                    let refined = (d as f32 + delta + self.min_disp as f32)
                        .clamp(self.min_disp as f32, self.max_disp as f32);
                    self.set(row, col, refined);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BlockMatching
// ─────────────────────────────────────────────────────────────────────────────

/// Block Matching (BM) stereo algorithm using Sum of Absolute Differences (SAD).
///
/// This is the classical baseline stereo matcher: efficient but prone to errors
/// on textureless regions and near depth discontinuities.
///
/// ## Example
/// ```
/// use scirs2_vision::stereo::disparity::{BlockMatching, DisparityMap};
///
/// let bm = BlockMatching::new(5, 0, 32);
/// let left  = vec![128u8; 64 * 64];
/// let right = vec![128u8; 64 * 64];
/// let disp = bm.compute(&left, &right, 64, 64).unwrap();
/// assert_eq!(disp.width, 64);
/// ```
pub struct BlockMatching {
    /// Block half-width. Full block is `(2*half+1)×(2*half+1)`. Must be ≥ 1.
    pub block_size: usize,
    /// Minimum disparity to search (inclusive).
    pub min_disparity: i32,
    /// Maximum disparity to search (inclusive).
    pub max_disparity: i32,
    /// Ratio threshold for the uniqueness (ratio) test.  0 disables the test.
    pub uniqueness_ratio: f32,
    /// Enable left-right consistency check (filters occluded pixels).
    pub lr_check: bool,
    /// Threshold for left-right check (in pixels).
    pub lr_threshold: i32,
}

impl BlockMatching {
    /// Create a new BlockMatching with default uniqueness ratio 0.15.
    pub fn new(block_size: usize, min_disp: i32, max_disp: i32) -> Self {
        Self {
            block_size,
            min_disparity: min_disp,
            max_disparity: max_disp,
            uniqueness_ratio: 0.15,
            lr_check: false,
            lr_threshold: 1,
        }
    }

    /// Compute the disparity map from a rectified grayscale stereo pair.
    ///
    /// `left` and `right` are row-major `[H × W]` grayscale byte buffers.
    pub fn compute(
        &self,
        left: &[u8],
        right: &[u8],
        width: usize,
        height: usize,
    ) -> Result<DisparityMap, VisionError> {
        self.validate(width, height)?;

        let half = self.block_size / 2;
        let mut disp_map = DisparityMap::new(width, height, self.min_disparity, self.max_disparity);

        // Build integral image for the right image (to speed up SAD via sliding window).
        // For simplicity we use direct SAD here; an optimised implementation would use
        // running-sum decomposition.
        let compute_for = |img_l: &[u8], img_r: &[u8]| -> Vec<f32> {
            let mut out = vec![0.0f32; width * height];
            for row in half..height.saturating_sub(half) {
                for col in (half as i32 + self.max_disparity) as usize..width.saturating_sub(half) {
                    let mut best_cost: u64 = u64::MAX;
                    let mut best_d = self.min_disparity;
                    let mut second_best: u64 = u64::MAX;

                    for d in self.min_disparity..=self.max_disparity {
                        let right_col = col as i32 - d;
                        if right_col < half as i32 || right_col >= width as i32 - half as i32 {
                            continue;
                        }
                        let rc = right_col as usize;

                        let mut cost: u64 = 0;
                        for br in -(half as i32)..=half as i32 {
                            let lr = (row as i32 + br) as usize;
                            for bc in -(half as i32)..=half as i32 {
                                let lc = (col as i32 + bc) as usize;
                                let rcc = (rc as i32 + bc) as usize;
                                let lv = img_l[lr * width + lc] as i64;
                                let rv = img_r[lr * width + rcc] as i64;
                                cost += (lv - rv).unsigned_abs();
                            }
                        }

                        if cost < best_cost {
                            second_best = best_cost;
                            best_cost = cost;
                            best_d = d;
                        } else if cost < second_best {
                            second_best = cost;
                        }
                    }

                    // Uniqueness / ratio test.
                    let pass_unique = if second_best > 0 && self.uniqueness_ratio > 0.0 {
                        (best_cost as f32 / second_best as f32) < 1.0 - self.uniqueness_ratio
                    } else {
                        true
                    };

                    if pass_unique {
                        out[row * width + col] = best_d as f32;
                    }
                }
            }
            out
        };

        let left_disp = compute_for(left, right);

        if self.lr_check {
            // Compute right-to-left disparity for L-R consistency check.
            let right_disp = compute_for(right, left);
            for row in 0..height {
                for col in 0..width {
                    let dl = left_disp[row * width + col];
                    let matched_col = col as i32 - dl as i32;
                    if matched_col >= 0 && matched_col < width as i32 {
                        let dr = right_disp[row * width + matched_col as usize];
                        if (dl + dr).abs() <= self.lr_threshold as f32 {
                            disp_map.set(row, col, dl);
                        }
                    }
                }
            }
        } else {
            disp_map.data = left_disp;
        }

        Ok(disp_map)
    }

    fn validate(&self, width: usize, height: usize) -> Result<(), VisionError> {
        if self.block_size == 0 {
            return Err(VisionError::InvalidParameter(
                "block_size must be at least 1".into(),
            ));
        }
        if self.min_disparity > self.max_disparity {
            return Err(VisionError::InvalidParameter(
                "min_disparity must be ≤ max_disparity".into(),
            ));
        }
        let half = self.block_size / 2;
        if height < 2 * half + 1 || width < 2 * half + 1 {
            return Err(VisionError::InvalidParameter(
                "Image is smaller than the matching block".into(),
            ));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SemiGlobalMatching
// ─────────────────────────────────────────────────────────────────────────────

/// Semi-Global Matching (SGM) — Hirschmüller 2008.
///
/// Uses Census transform as the matching cost and aggregates along 8 scan-line
/// paths with smooth disparity-transition penalties P1 and P2.
///
/// ## Example
/// ```
/// use scirs2_vision::stereo::disparity::SemiGlobalMatching;
///
/// let sgm = SemiGlobalMatching::new(0, 16);
/// let left  = vec![128u8; 64 * 64];
/// let right = vec![128u8; 64 * 64];
/// let disp = sgm.compute(&left, &right, 64, 64).unwrap();
/// assert_eq!(disp.height, 64);
/// ```
pub struct SemiGlobalMatching {
    /// Census window half-size (full window = `2*block_size+1`).
    pub block_size: usize,
    /// Minimum disparity to search (inclusive).
    pub min_disparity: i32,
    /// Maximum disparity to search (inclusive).
    pub max_disparity: i32,
    /// Penalty for a ±1 disparity change between adjacent path pixels.
    pub p1: u32,
    /// Penalty for a >1 disparity change between adjacent path pixels.
    pub p2: u32,
    /// Enable sub-pixel refinement via parabolic interpolation.
    pub subpixel: bool,
}

impl SemiGlobalMatching {
    /// Default constructor: `block_size=3`, `p1=8`, `p2=32`.
    pub fn new(min_disp: i32, max_disp: i32) -> Self {
        Self {
            block_size: 3,
            min_disparity: min_disp,
            max_disparity: max_disp,
            p1: 8,
            p2: 32,
            subpixel: false,
        }
    }

    /// Compute the SGM disparity map for a rectified stereo pair.
    pub fn compute(
        &self,
        left: &[u8],
        right: &[u8],
        width: usize,
        height: usize,
    ) -> Result<DisparityMap, VisionError> {
        self.validate(width, height)?;

        let n_disp = (self.max_disparity - self.min_disparity + 1) as usize;

        // ── Census transform ──────────────────────────────────────────────────
        let census_left = self.census_transform(left, width, height);
        let census_right = self.census_transform(right, width, height);

        // ── Matching cost volume [H * W * N_DISP] ────────────────────────────
        let mut base_cost = vec![0u16; height * width * n_disp];
        self.build_cost_volume(
            &census_left,
            &census_right,
            &mut base_cost,
            width,
            height,
            n_disp,
        );

        // ── Aggregate along 8 directions ─────────────────────────────────────
        let mut agg_cost = vec![0u32; height * width * n_disp];

        let directions: [(i32, i32); 8] = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ];

        for (dr, dc) in &directions {
            self.aggregate_along_path(&base_cost, &mut agg_cost, width, height, n_disp, *dr, *dc);
        }

        // ── Winner-take-all ───────────────────────────────────────────────────
        let mut disp_map = DisparityMap::new(width, height, self.min_disparity, self.max_disparity);

        for row in 0..height {
            for col in 0..width {
                let base = (row * width + col) * n_disp;
                let best_d = (0..n_disp).min_by_key(|&d| agg_cost[base + d]).unwrap_or(0);
                disp_map.set(row, col, (best_d as i32 + self.min_disparity) as f32);
            }
        }

        if self.subpixel {
            // Convert agg_cost to f32 for refinement.
            let agg_f32: Vec<f32> = agg_cost.iter().map(|&v| v as f32).collect();
            disp_map.apply_subpixel_refinement(&agg_f32, n_disp);
        }

        Ok(disp_map)
    }

    // ── Census transform ─────────────────────────────────────────────────────

    /// Compute Census transform bit-strings for an image.
    ///
    /// Each pixel is encoded as a 64-bit integer where bit `i` is set if the
    /// `i`-th neighbour (raster order, centre excluded) is strictly less than
    /// the centre pixel.
    fn census_transform(&self, img: &[u8], width: usize, height: usize) -> Vec<u64> {
        let mut census = vec![0u64; width * height];
        let h = self.block_size;

        for row in h..height.saturating_sub(h) {
            for col in h..width.saturating_sub(h) {
                let center = img[row * width + col];
                let mut bits = 0u64;
                let mut pos = 0u32;

                'outer: for br in -(h as i32)..=h as i32 {
                    for bc in -(h as i32)..=h as i32 {
                        if br == 0 && bc == 0 {
                            continue;
                        }
                        let nr = (row as i32 + br) as usize;
                        let nc = (col as i32 + bc) as usize;
                        if img[nr * width + nc] < center {
                            bits |= 1u64 << pos;
                        }
                        pos += 1;
                        if pos >= 64 {
                            break 'outer;
                        }
                    }
                }
                census[row * width + col] = bits;
            }
        }
        census
    }

    // ── Matching cost volume ─────────────────────────────────────────────────

    fn build_cost_volume(
        &self,
        census_l: &[u64],
        census_r: &[u64],
        costs: &mut [u16],
        width: usize,
        height: usize,
        n_disp: usize,
    ) {
        for row in 0..height {
            for col in 0..width {
                let base = (row * width + col) * n_disp;
                for d in 0..n_disp {
                    let d_i = d as i32 + self.min_disparity;
                    let right_col = col as i32 - d_i;
                    costs[base + d] = if right_col >= 0 && right_col < width as i32 {
                        let hamming = (census_l[row * width + col]
                            ^ census_r[row * width + right_col as usize])
                            .count_ones() as u16;
                        hamming.min(255)
                    } else {
                        255
                    };
                }
            }
        }
    }

    // ── Path aggregation ─────────────────────────────────────────────────────

    fn aggregate_along_path(
        &self,
        base_cost: &[u16],
        agg: &mut [u32],
        width: usize,
        height: usize,
        n_disp: usize,
        dr: i32,
        dc: i32,
    ) {
        // Enumerate pixels in path-traversal order: start from the border
        // opposite to the direction (dr, dc).
        let rows: Box<dyn Iterator<Item = usize>> = if dr >= 0 {
            Box::new(0..height)
        } else {
            Box::new((0..height).rev())
        };

        // We process row-by-row; column ordering is handled inside.
        for row in rows {
            let cols: Box<dyn Iterator<Item = usize>> = if dc >= 0 {
                Box::new(0..width)
            } else {
                Box::new((0..width).rev())
            };

            for col in cols {
                let prev_r = row as i32 - dr;
                let prev_c = col as i32 - dc;
                let has_prev =
                    prev_r >= 0 && prev_r < height as i32 && prev_c >= 0 && prev_c < width as i32;

                let cur_base = (row * width + col) * n_disp;

                if !has_prev {
                    // Border pixel: path cost equals base matching cost.
                    for d in 0..n_disp {
                        agg[cur_base + d] =
                            agg[cur_base + d].saturating_add(base_cost[cur_base + d] as u32);
                    }
                    continue;
                }

                let pr = prev_r as usize;
                let pc = prev_c as usize;
                let prev_base = (pr * width + pc) * n_disp;

                // Minimum cost at previous pixel along path.
                let min_prev = (0..n_disp)
                    .map(|d| agg[prev_base + d])
                    .min()
                    .unwrap_or(u32::MAX);

                for d in 0..n_disp {
                    let matching = base_cost[cur_base + d] as u32;

                    // SGM path cost Lr(p, d) =
                    //   C(p,d) + min(Lr(p-r,d), Lr(p-r,d±1)+P1, min_k(Lr(p-r,k))+P2) - min_k(Lr(p-r,k))
                    let same_d = agg[prev_base + d];
                    let d_plus = if d + 1 < n_disp {
                        agg[prev_base + d + 1].saturating_add(self.p1)
                    } else {
                        u32::MAX
                    };
                    let d_minus = if d > 0 {
                        agg[prev_base + d - 1].saturating_add(self.p1)
                    } else {
                        u32::MAX
                    };
                    let other = min_prev.saturating_add(self.p2);

                    let path_cost =
                        matching + same_d.min(d_plus).min(d_minus).min(other) - min_prev;

                    agg[cur_base + d] = agg[cur_base + d].saturating_add(path_cost);
                }
            }
        }
    }

    fn validate(&self, width: usize, height: usize) -> Result<(), VisionError> {
        if self.min_disparity > self.max_disparity {
            return Err(VisionError::InvalidParameter(
                "min_disparity must be ≤ max_disparity".into(),
            ));
        }
        if self.block_size == 0 {
            return Err(VisionError::InvalidParameter(
                "block_size must be at least 1".into(),
            ));
        }
        if height < 2 * self.block_size + 1 || width < 2 * self.block_size + 1 {
            return Err(VisionError::InvalidParameter(
                "Image is smaller than the census window".into(),
            ));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_stereo(width: usize, height: usize, true_disp: i32) -> (Vec<u8>, Vec<u8>) {
        // Left image: horizontal gradient texture.
        let left: Vec<u8> = (0..height * width)
            .map(|i| ((i % width) * 255 / width) as u8)
            .collect();
        // Right image: shifted left image by `true_disp` pixels.
        let mut right = vec![128u8; height * width];
        for row in 0..height {
            for col in 0..width {
                let src = col as i32 + true_disp;
                if src >= 0 && src < width as i32 {
                    right[row * width + col] = left[row * width + src as usize];
                }
            }
        }
        (left, right)
    }

    #[test]
    fn test_disparity_map_new_and_accessors() {
        let mut dm = DisparityMap::new(10, 8, 0, 16);
        assert_eq!(dm.width, 10);
        assert_eq!(dm.height, 8);
        dm.set(3, 5, 7.0);
        assert!((dm.get(3, 5) - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_to_depth() {
        let mut dm = DisparityMap::new(4, 4, 0, 16);
        dm.set(1, 1, 4.0);
        let depth = dm.to_depth(0.1, 500.0);
        // depth = 0.1 * 500 / 4 = 12.5
        assert!((depth[1 * 4 + 1] - 12.5).abs() < 1e-4, "depth={}", depth[5]);
        // Zero-disparity pixel → depth 0.
        assert_eq!(depth[0], 0.0);
    }

    #[test]
    fn test_block_matching_uniform_image() {
        // Uniform images → all costs equal, BM may produce disparity 0.
        let bm = BlockMatching::new(3, 0, 8);
        let img = vec![200u8; 32 * 32];
        let disp = bm.compute(&img, &img, 32, 32).expect("BM failed");
        assert_eq!(disp.width, 32);
        assert_eq!(disp.height, 32);
    }

    #[test]
    fn test_block_matching_known_disparity() {
        let width = 64;
        let height = 32;
        let true_disp = 4i32;
        let (left, right) = synthetic_stereo(width, height, true_disp);

        let bm = BlockMatching {
            block_size: 5,
            min_disparity: 0,
            max_disparity: 16,
            uniqueness_ratio: 0.10,
            lr_check: false,
            lr_threshold: 1,
        };

        let disp = bm.compute(&left, &right, width, height).expect("BM failed");

        // Check the central region (avoid border artefacts).
        let mut correct = 0;
        let mut total = 0;
        let margin = 8usize;
        for row in margin..height - margin {
            for col in margin..width - margin {
                let d = disp.get(row, col);
                if d > 0.0 {
                    if (d - true_disp as f32).abs() <= 1.0 {
                        correct += 1;
                    }
                    total += 1;
                }
            }
        }
        // At least 50% of valid pixels should have correct disparity.
        if total > 0 {
            assert!(
                correct * 100 / total >= 50,
                "BM accuracy too low: {correct}/{total}"
            );
        }
    }

    #[test]
    fn test_sgm_runs_without_error() {
        let sgm = SemiGlobalMatching::new(0, 8);
        let img = vec![100u8; 32 * 32];
        let disp = sgm.compute(&img, &img, 32, 32).expect("SGM failed");
        assert_eq!(disp.width, 32);
    }

    #[test]
    fn test_block_matching_invalid_params() {
        let bm = BlockMatching::new(3, 5, 2); // min > max
        let res = bm.compute(&[0u8; 64], &[0u8; 64], 8, 8);
        assert!(res.is_err());
    }
}
