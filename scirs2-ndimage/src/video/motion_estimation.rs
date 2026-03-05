//! Motion estimation algorithms for video sequences.
//!
//! Provides block-matching motion estimation and frame-difference methods:
//!
//! * [`block_matching`] – exhaustive block-matching (EBMA) with configurable search window.
//! * [`three_step_search`] – three-step block-matching search (TSS / 3SS).
//! * [`diamond_search`] – diamond search pattern for block matching.
//! * [`motion_compensated_frame`] – reconstruct a frame from motion vectors.
//! * [`temporal_difference`] – simple frame-to-frame difference.
//! * [`accumulate_difference`] – Jain's accumulated difference image (ADI).

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::Array2;

// ─── MotionVector ─────────────────────────────────────────────────────────────

/// A 2-D motion vector (dy, dx) in pixel units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionVector {
    /// Vertical displacement (positive = down).
    pub dy: i32,
    /// Horizontal displacement (positive = right).
    pub dx: i32,
}

impl MotionVector {
    /// Create a new motion vector.
    pub fn new(dy: i32, dx: i32) -> Self {
        Self { dy, dx }
    }

    /// L2 magnitude of the vector.
    pub fn magnitude(&self) -> f64 {
        ((self.dy as f64).powi(2) + (self.dx as f64).powi(2)).sqrt()
    }

    /// Zero motion vector.
    pub fn zero() -> Self {
        Self { dy: 0, dx: 0 }
    }
}

// ─── BlockMatchResult ─────────────────────────────────────────────────────────

/// Result of block-matching for a single macroblock.
#[derive(Debug, Clone)]
pub struct BlockMatchResult {
    /// Row of the macroblock origin (top-left corner) in the target frame.
    pub block_row: usize,
    /// Column of the macroblock origin in the target frame.
    pub block_col: usize,
    /// Best-match motion vector.
    pub motion_vector: MotionVector,
    /// Minimum matching cost (SAD or MSE) at the best match.
    pub cost: f64,
}

// ─── BlockMatchConfig ─────────────────────────────────────────────────────────

/// Configuration for block-matching algorithms.
#[derive(Debug, Clone)]
pub struct BlockMatchConfig {
    /// Size of each macroblock (square; must be ≥ 1).
    pub block_size: usize,
    /// Search window radius in pixels (full window = 2*radius+1 × 2*radius+1).
    pub search_radius: usize,
    /// Cost function: `false` = SAD (sum of absolute differences),
    ///                `true`  = MSE (mean squared error).
    pub use_mse: bool,
}

impl Default for BlockMatchConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            search_radius: 8,
            use_mse: false,
        }
    }
}

// ─── Cost functions ───────────────────────────────────────────────────────────

/// Sum of absolute differences between two equal-sized regions.
/// Returns f64::MAX if either region is out of bounds.
fn sad_region(
    reference: &Array2<f64>,
    target: &Array2<f64>,
    ref_row: i32,
    ref_col: i32,
    tgt_row: i32,
    tgt_col: i32,
    block_size: usize,
) -> f64 {
    let rows = reference.nrows() as i32;
    let cols = reference.ncols() as i32;
    let bs = block_size as i32;

    // Bounds check
    if ref_row < 0
        || ref_col < 0
        || ref_row + bs > rows
        || ref_col + bs > cols
        || tgt_row < 0
        || tgt_col < 0
        || tgt_row + bs > rows
        || tgt_col + bs > cols
    {
        return f64::MAX;
    }

    let mut sad = 0.0_f64;
    for dr in 0..block_size {
        for dc in 0..block_size {
            let r_r = (ref_row + dr as i32) as usize;
            let r_c = (ref_col + dc as i32) as usize;
            let t_r = (tgt_row + dr as i32) as usize;
            let t_c = (tgt_col + dc as i32) as usize;
            sad += (reference[[r_r, r_c]] - target[[t_r, t_c]]).abs();
        }
    }
    sad
}

/// Mean squared error between two equal-sized regions.
fn mse_region(
    reference: &Array2<f64>,
    target: &Array2<f64>,
    ref_row: i32,
    ref_col: i32,
    tgt_row: i32,
    tgt_col: i32,
    block_size: usize,
) -> f64 {
    let rows = reference.nrows() as i32;
    let cols = reference.ncols() as i32;
    let bs = block_size as i32;
    if ref_row < 0
        || ref_col < 0
        || ref_row + bs > rows
        || ref_col + bs > cols
        || tgt_row < 0
        || tgt_col < 0
        || tgt_row + bs > rows
        || tgt_col + bs > cols
    {
        return f64::MAX;
    }

    let n = (block_size * block_size) as f64;
    let mut mse = 0.0_f64;
    for dr in 0..block_size {
        for dc in 0..block_size {
            let r_r = (ref_row + dr as i32) as usize;
            let r_c = (ref_col + dc as i32) as usize;
            let t_r = (tgt_row + dr as i32) as usize;
            let t_c = (tgt_col + dc as i32) as usize;
            let d = reference[[r_r, r_c]] - target[[t_r, t_c]];
            mse += d * d;
        }
    }
    mse / n
}

fn compute_cost(
    reference: &Array2<f64>,
    target: &Array2<f64>,
    ref_row: i32,
    ref_col: i32,
    tgt_row: i32,
    tgt_col: i32,
    block_size: usize,
    use_mse: bool,
) -> f64 {
    if use_mse {
        mse_region(reference, target, ref_row, ref_col, tgt_row, tgt_col, block_size)
    } else {
        sad_region(reference, target, ref_row, ref_col, tgt_row, tgt_col, block_size)
    }
}

// ─── block_matching ───────────────────────────────────────────────────────────

/// Exhaustive Block Matching Algorithm (EBMA).
///
/// Partitions `reference` into non-overlapping macroblocks of `config.block_size`,
/// and for each block searches `target` within ±`config.search_radius` pixels
/// to find the best motion vector (minimum cost).
///
/// # Returns
/// A `Vec<BlockMatchResult>`, one entry per macroblock.
pub fn block_matching(
    reference: &Array2<f64>,
    target: &Array2<f64>,
    config: &BlockMatchConfig,
) -> NdimageResult<Vec<BlockMatchResult>> {
    if reference.shape() != target.shape() {
        return Err(NdimageError::DimensionError(
            "reference and target must have the same shape".into(),
        ));
    }
    let bs = config.block_size;
    if bs == 0 {
        return Err(NdimageError::InvalidInput("block_size must be >= 1".into()));
    }
    let rows = reference.nrows();
    let cols = reference.ncols();
    let radius = config.search_radius as i32;
    let mut results = Vec::new();

    let n_rows = rows / bs;
    let n_cols = cols / bs;

    for br in 0..n_rows {
        for bc in 0..n_cols {
            let ref_row = (br * bs) as i32;
            let ref_col = (bc * bs) as i32;
            let mut best_cost = f64::MAX;
            let mut best_dy = 0_i32;
            let mut best_dx = 0_i32;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let tgt_row = ref_row + dy;
                    let tgt_col = ref_col + dx;
                    let cost = compute_cost(
                        reference,
                        target,
                        ref_row,
                        ref_col,
                        tgt_row,
                        tgt_col,
                        bs,
                        config.use_mse,
                    );
                    if cost < best_cost {
                        best_cost = cost;
                        best_dy = dy;
                        best_dx = dx;
                    }
                }
            }

            results.push(BlockMatchResult {
                block_row: br * bs,
                block_col: bc * bs,
                motion_vector: MotionVector::new(best_dy, best_dx),
                cost: best_cost,
            });
        }
    }
    Ok(results)
}

// ─── three_step_search ────────────────────────────────────────────────────────

/// Three-step search (TSS) for a single macroblock.
///
/// Searches for the motion vector that minimises cost for the block at
/// `(ref_row, ref_col)` in `reference` relative to `target`,
/// using the 3-step search pattern with maximum displacement `max_disp`.
fn three_step_search_block(
    reference: &Array2<f64>,
    target: &Array2<f64>,
    ref_row: i32,
    ref_col: i32,
    block_size: usize,
    max_disp: i32,
    use_mse: bool,
) -> (i32, i32, f64) {
    let mut center_y = ref_row;
    let mut center_x = ref_col;
    let mut step = max_disp / 2;
    let mut best_cost = f64::MAX;
    let mut best_dy = 0_i32;
    let mut best_dx = 0_i32;

    while step >= 1 {
        let mut local_best_cost = best_cost;
        let mut local_dy = 0_i32;
        let mut local_dx = 0_i32;

        for &dy_off in &[-step, 0, step] {
            for &dx_off in &[-step, 0, step] {
                let tgt_row = center_y + dy_off;
                let tgt_col = center_x + dx_off;
                let cost = compute_cost(
                    reference,
                    target,
                    ref_row,
                    ref_col,
                    tgt_row,
                    tgt_col,
                    block_size,
                    use_mse,
                );
                if cost < local_best_cost {
                    local_best_cost = cost;
                    local_dy = tgt_row - ref_row;
                    local_dx = tgt_col - ref_col;
                }
            }
        }
        if local_best_cost < best_cost {
            best_cost = local_best_cost;
            best_dy = local_dy;
            best_dx = local_dx;
        }
        center_y = ref_row + best_dy;
        center_x = ref_col + best_dx;
        step /= 2;
    }
    (best_dy, best_dx, best_cost)
}

/// Three-step search block-matching for all macroblocks.
///
/// Faster than EBMA at the cost of potentially missing the global minimum.
/// `max_disp` is the maximum allowed displacement in any direction.
pub fn three_step_search(
    reference: &Array2<f64>,
    target: &Array2<f64>,
    config: &BlockMatchConfig,
) -> NdimageResult<Vec<BlockMatchResult>> {
    if reference.shape() != target.shape() {
        return Err(NdimageError::DimensionError(
            "reference and target must have the same shape".into(),
        ));
    }
    let bs = config.block_size;
    if bs == 0 {
        return Err(NdimageError::InvalidInput("block_size must be >= 1".into()));
    }
    let rows = reference.nrows();
    let cols = reference.ncols();
    let max_disp = config.search_radius as i32;
    if max_disp < 1 {
        return Err(NdimageError::InvalidInput(
            "search_radius must be >= 1 for three_step_search".into(),
        ));
    }
    let mut results = Vec::new();

    for br in 0..(rows / bs) {
        for bc in 0..(cols / bs) {
            let ref_row = (br * bs) as i32;
            let ref_col = (bc * bs) as i32;
            let (dy, dx, cost) = three_step_search_block(
                reference,
                target,
                ref_row,
                ref_col,
                bs,
                max_disp,
                config.use_mse,
            );
            results.push(BlockMatchResult {
                block_row: br * bs,
                block_col: bc * bs,
                motion_vector: MotionVector::new(dy, dx),
                cost,
            });
        }
    }
    Ok(results)
}

// ─── diamond_search ───────────────────────────────────────────────────────────

/// Diamond Search (DS) pattern for a single macroblock.
///
/// Uses the Large Diamond Search Pattern (LDSP) until convergence, then
/// Small Diamond Search Pattern (SDSP) for the final step.
fn diamond_search_block(
    reference: &Array2<f64>,
    target: &Array2<f64>,
    ref_row: i32,
    ref_col: i32,
    block_size: usize,
    max_disp: i32,
    use_mse: bool,
) -> (i32, i32, f64) {
    // Large diamond: 5 points
    let ldsp: [(i32, i32); 9] = [
        (-2, 0), (-1, -1), (-1, 1), (0, -2), (0, 0), (0, 2), (1, -1), (1, 1), (2, 0),
    ];
    // Small diamond: 5 points
    let sdsp: [(i32, i32); 5] = [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)];

    let clamp = |v: i32| v.max(-max_disp).min(max_disp);

    let mut center_dy = 0_i32;
    let mut center_dx = 0_i32;

    loop {
        let mut best_cost = f64::MAX;
        let mut best_off_dy = 0_i32;
        let mut best_off_dx = 0_i32;

        for &(dy_off, dx_off) in &ldsp {
            let new_dy = clamp(center_dy + dy_off);
            let new_dx = clamp(center_dx + dx_off);
            let tgt_row = ref_row + new_dy;
            let tgt_col = ref_col + new_dx;
            let cost = compute_cost(
                reference,
                target,
                ref_row,
                ref_col,
                tgt_row,
                tgt_col,
                block_size,
                use_mse,
            );
            if cost < best_cost {
                best_cost = cost;
                best_off_dy = new_dy;
                best_off_dx = new_dx;
            }
        }

        if best_off_dy == center_dy && best_off_dx == center_dx {
            // Centre was the best — switch to small diamond
            break;
        }
        center_dy = best_off_dy;
        center_dx = best_off_dx;
    }

    // SDSP refinement
    let mut best_cost = f64::MAX;
    let mut best_dy = center_dy;
    let mut best_dx = center_dx;
    for &(dy_off, dx_off) in &sdsp {
        let new_dy = clamp(center_dy + dy_off);
        let new_dx = clamp(center_dx + dx_off);
        let tgt_row = ref_row + new_dy;
        let tgt_col = ref_col + new_dx;
        let cost = compute_cost(
            reference,
            target,
            ref_row,
            ref_col,
            tgt_row,
            tgt_col,
            block_size,
            use_mse,
        );
        if cost < best_cost {
            best_cost = cost;
            best_dy = new_dy;
            best_dx = new_dx;
        }
    }
    (best_dy, best_dx, best_cost)
}

/// Diamond search block-matching for all macroblocks.
pub fn diamond_search(
    reference: &Array2<f64>,
    target: &Array2<f64>,
    config: &BlockMatchConfig,
) -> NdimageResult<Vec<BlockMatchResult>> {
    if reference.shape() != target.shape() {
        return Err(NdimageError::DimensionError(
            "reference and target must have the same shape".into(),
        ));
    }
    let bs = config.block_size;
    if bs == 0 {
        return Err(NdimageError::InvalidInput("block_size must be >= 1".into()));
    }
    let rows = reference.nrows();
    let cols = reference.ncols();
    let max_disp = config.search_radius as i32;
    let mut results = Vec::new();

    for br in 0..(rows / bs) {
        for bc in 0..(cols / bs) {
            let ref_row = (br * bs) as i32;
            let ref_col = (bc * bs) as i32;
            let (dy, dx, cost) = diamond_search_block(
                reference,
                target,
                ref_row,
                ref_col,
                bs,
                max_disp,
                config.use_mse,
            );
            results.push(BlockMatchResult {
                block_row: br * bs,
                block_col: bc * bs,
                motion_vector: MotionVector::new(dy, dx),
                cost,
            });
        }
    }
    Ok(results)
}

// ─── motion_compensated_frame ─────────────────────────────────────────────────

/// Reconstruct an approximation of `target` by motion-compensating `reference`.
///
/// For each macroblock in the block-match results, the corresponding block
/// is copied from `reference` offset by the motion vector into the output.
/// Pixels not covered by any macroblock are filled from `reference` directly.
pub fn motion_compensated_frame(
    reference: &Array2<f64>,
    motion_vectors: &[BlockMatchResult],
    block_size: usize,
) -> NdimageResult<Array2<f64>> {
    if block_size == 0 {
        return Err(NdimageError::InvalidInput("block_size must be >= 1".into()));
    }
    let rows = reference.nrows();
    let cols = reference.ncols();
    let mut output = reference.clone();

    for bmr in motion_vectors {
        let dst_row = bmr.block_row as i32;
        let dst_col = bmr.block_col as i32;
        let src_row = dst_row + bmr.motion_vector.dy;
        let src_col = dst_col + bmr.motion_vector.dx;
        let bs = block_size as i32;

        // Bounds check for source block
        if src_row < 0
            || src_col < 0
            || src_row + bs > rows as i32
            || src_col + bs > cols as i32
        {
            continue; // Skip out-of-bounds blocks
        }

        for dr in 0..block_size {
            for dc in 0..block_size {
                let sr = (src_row + dr as i32) as usize;
                let sc = (src_col + dc as i32) as usize;
                let dr2 = (dst_row + dr as i32) as usize;
                let dc2 = (dst_col + dc as i32) as usize;
                if dr2 < rows && dc2 < cols {
                    output[[dr2, dc2]] = reference[[sr, sc]];
                }
            }
        }
    }
    Ok(output)
}

// ─── temporal_difference ─────────────────────────────────────────────────────

/// Frame-difference image: `|next - prev|`.
///
/// Values are clipped to [0, max_val] if `max_val > 0`, otherwise returned raw.
pub fn temporal_difference(
    prev: &Array2<f64>,
    next: &Array2<f64>,
    threshold: Option<f64>,
) -> NdimageResult<Array2<f64>> {
    if prev.shape() != next.shape() {
        return Err(NdimageError::DimensionError(
            "prev and next frames must have the same shape".into(),
        ));
    }
    let rows = prev.nrows();
    let cols = prev.ncols();
    let thr = threshold.unwrap_or(0.0);
    Ok(Array2::from_shape_fn((rows, cols), |(r, c)| {
        let d = (next[[r, c]] - prev[[r, c]]).abs();
        if thr > 0.0 && d < thr {
            0.0
        } else {
            d
        }
    }))
}

// ─── accumulate_difference ────────────────────────────────────────────────────

/// Jain's Accumulated Difference Image (ADI).
///
/// Maintains two accumulators:
/// - `adi_plus[r,c]` : count of frames where `frame[r,c] > ref[r,c]`
/// - `adi_minus[r,c]`: count of frames where `frame[r,c] < ref[r,c]`
///
/// A "signed" ADI is returned as `adi_plus - adi_minus` (float counts).
/// Moving regions produce large absolute values; static regions near zero.
///
/// # Arguments
/// * `reference`   – background/reference frame.
/// * `frames`      – sequence of frames to accumulate.
/// * `threshold`   – minimum |diff| to count as a change (avoids noise).
///
/// # Returns
/// `(adi_plus, adi_minus, adi_signed)` each of shape (rows, cols).
pub fn accumulate_difference(
    reference: &Array2<f64>,
    frames: &[Array2<f64>],
    threshold: f64,
) -> NdimageResult<(Array2<f64>, Array2<f64>, Array2<f64>)> {
    if frames.is_empty() {
        return Err(NdimageError::InvalidInput(
            "frames slice must not be empty".into(),
        ));
    }
    let rows = reference.nrows();
    let cols = reference.ncols();
    for (i, f) in frames.iter().enumerate() {
        if f.shape() != reference.shape() {
            return Err(NdimageError::DimensionError(format!(
                "frame {i} has shape {:?}, expected {:?}",
                f.shape(),
                reference.shape()
            )));
        }
    }

    let mut adi_plus = Array2::<f64>::zeros((rows, cols));
    let mut adi_minus = Array2::<f64>::zeros((rows, cols));

    for frame in frames {
        for r in 0..rows {
            for c in 0..cols {
                let diff = frame[[r, c]] - reference[[r, c]];
                if diff > threshold {
                    adi_plus[[r, c]] += 1.0;
                } else if diff < -threshold {
                    adi_minus[[r, c]] += 1.0;
                }
            }
        }
    }

    let adi_signed = &adi_plus - &adi_minus;
    Ok((adi_plus, adi_minus, adi_signed))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_frame(rows: usize, cols: usize, val: f64) -> Array2<f64> {
        Array2::from_elem((rows, cols), val)
    }

    #[test]
    fn test_temporal_difference_zero() {
        let f = make_frame(8, 8, 100.0);
        let diff = temporal_difference(&f, &f, None).expect("ok");
        assert!(diff.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_temporal_difference_constant_shift() {
        let prev = make_frame(8, 8, 50.0);
        let next = make_frame(8, 8, 80.0);
        let diff = temporal_difference(&prev, &next, None).expect("ok");
        assert!(diff.iter().all(|&v| (v - 30.0).abs() < 1e-9));
    }

    #[test]
    fn test_block_matching_zero_motion() {
        let frame = Array2::from_shape_fn((32, 32), |(r, c)| (r + c) as f64);
        let cfg = BlockMatchConfig {
            block_size: 8,
            search_radius: 4,
            use_mse: false,
        };
        let results = block_matching(&frame, &frame, &cfg).expect("ok");
        for r in &results {
            assert_eq!(r.motion_vector, MotionVector::zero(), "expected zero MV");
        }
    }

    #[test]
    fn test_three_step_search_zero_motion() {
        let frame = Array2::from_shape_fn((32, 32), |(r, c)| (r * 2 + c) as f64);
        let cfg = BlockMatchConfig {
            block_size: 8,
            search_radius: 4,
            use_mse: false,
        };
        let results = three_step_search(&frame, &frame, &cfg).expect("ok");
        assert!(!results.is_empty());
        for r in &results {
            assert_eq!(r.motion_vector, MotionVector::zero());
        }
    }

    #[test]
    fn test_diamond_search_smoke() {
        let frame = Array2::from_shape_fn((32, 32), |(r, c)| (r + c * 2) as f64);
        let cfg = BlockMatchConfig {
            block_size: 8,
            search_radius: 4,
            use_mse: false,
        };
        let results = diamond_search(&frame, &frame, &cfg).expect("ok");
        assert_eq!(results.len(), 4); // 32/8 × 32/8 = 16 blocks → 4×4
    }

    #[test]
    fn test_motion_compensated_frame_identity() {
        let frame = Array2::from_shape_fn((16, 16), |(r, c)| (r + c) as f64);
        let cfg = BlockMatchConfig {
            block_size: 8,
            search_radius: 2,
            use_mse: false,
        };
        let mvs = block_matching(&frame, &frame, &cfg).expect("ok");
        let reconstructed = motion_compensated_frame(&frame, &mvs, 8).expect("ok");
        // With zero motion, reconstruction should equal the original
        for r in 0..16 {
            for c in 0..16 {
                assert!((frame[[r, c]] - reconstructed[[r, c]]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_accumulate_difference() {
        let reference = make_frame(4, 4, 100.0);
        // Alternating high/low frames
        let frames: Vec<_> = (0..6)
            .map(|i| make_frame(4, 4, if i % 2 == 0 { 120.0 } else { 80.0 }))
            .collect();
        let (plus, minus, signed) = accumulate_difference(&reference, &frames, 5.0).expect("ok");
        // 3 frames above, 3 below
        assert!((plus[[0, 0]] - 3.0).abs() < 1e-9);
        assert!((minus[[0, 0]] - 3.0).abs() < 1e-9);
        assert!(signed[[0, 0]].abs() < 1e-9);
    }
}
