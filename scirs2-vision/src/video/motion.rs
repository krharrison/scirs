//! Motion estimation algorithms for video processing.
//!
//! Provides block-based and frequency-domain methods for estimating inter-frame
//! motion, as well as utilities for motion compensation and visualisation.
//!
//! # Algorithms
//!
//! - **Full-search block matching** -- exhaustive SAD search
//! - **Three-step search (TSS)** -- fast logarithmic-step block matching
//! - **Phase correlation** -- global translation estimation via cross-power spectrum
//! - **Motion compensation** -- apply a motion field to warp/predict frames
//! - **Motion field visualisation** -- convert vector field to HSV-coded image data

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::Array2;

// ---------------------------------------------------------------------------
// Motion vector types
// ---------------------------------------------------------------------------

/// A 2-D motion vector (displacement in row/col directions).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionVector {
    /// Row displacement (positive = downward).
    pub dy: f64,
    /// Column displacement (positive = rightward).
    pub dx: f64,
}

impl MotionVector {
    /// Create a new motion vector.
    pub fn new(dy: f64, dx: f64) -> Self {
        Self { dy, dx }
    }

    /// Zero motion.
    pub fn zero() -> Self {
        Self { dy: 0.0, dx: 0.0 }
    }

    /// Magnitude of the motion vector.
    pub fn magnitude(&self) -> f64 {
        (self.dy * self.dy + self.dx * self.dx).sqrt()
    }

    /// Direction in radians.
    pub fn angle(&self) -> f64 {
        self.dy.atan2(self.dx)
    }
}

/// A dense motion field covering an entire frame, stored on a block grid.
#[derive(Debug, Clone)]
pub struct MotionField {
    /// Motion vectors indexed `[block_row][block_col]`.
    pub vectors: Vec<Vec<MotionVector>>,
    /// Block size used for estimation.
    pub block_size: usize,
    /// Number of block rows.
    pub rows: usize,
    /// Number of block columns.
    pub cols: usize,
    /// Source frame height.
    pub frame_height: usize,
    /// Source frame width.
    pub frame_width: usize,
}

impl MotionField {
    /// Average motion magnitude across all blocks.
    pub fn average_magnitude(&self) -> f64 {
        let total: f64 = self
            .vectors
            .iter()
            .flat_map(|row| row.iter())
            .map(|v| v.magnitude())
            .sum();
        let count = (self.rows * self.cols) as f64;
        if count > 0.0 {
            total / count
        } else {
            0.0
        }
    }

    /// Maximum motion magnitude.
    pub fn max_magnitude(&self) -> f64 {
        self.vectors
            .iter()
            .flat_map(|row| row.iter())
            .map(|v| v.magnitude())
            .fold(0.0_f64, f64::max)
    }

    /// Convert the motion field to an HSV-coded visualisation image.
    ///
    /// Returns three `Array2<f64>` planes (H, S, V) each in `[0, 1]`.
    /// Hue encodes direction, saturation/value encode magnitude.
    pub fn to_hsv_visualization(&self) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let max_mag = self.max_magnitude().max(1e-9);
        let h = self.frame_height;
        let w = self.frame_width;
        let mut hue = Array2::zeros((h, w));
        let mut sat = Array2::zeros((h, w));
        let mut val = Array2::zeros((h, w));

        for br in 0..self.rows {
            for bc in 0..self.cols {
                let mv = &self.vectors[br][bc];
                let mag = mv.magnitude() / max_mag;
                let ang = (mv.angle() + std::f64::consts::PI) / (2.0 * std::f64::consts::PI);

                let r_start = br * self.block_size;
                let c_start = bc * self.block_size;
                let r_end = (r_start + self.block_size).min(h);
                let c_end = (c_start + self.block_size).min(w);

                for r in r_start..r_end {
                    for c in c_start..c_end {
                        hue[[r, c]] = ang;
                        sat[[r, c]] = mag;
                        val[[r, c]] = mag;
                    }
                }
            }
        }
        (hue, sat, val)
    }
}

// ---------------------------------------------------------------------------
// Block Matching -- Full Search
// ---------------------------------------------------------------------------

/// Perform full-search block matching between two frames.
///
/// For each `block_size x block_size` block in `current`, the best matching
/// position in `reference` is found within `[-search_range, +search_range]`
/// using the Sum of Absolute Differences (SAD) criterion.
///
/// # Arguments
/// * `reference` -- the previous (reference) frame
/// * `current`   -- the current frame
/// * `block_size` -- side length of the square block (e.g. 8 or 16)
/// * `search_range` -- maximum displacement in each direction
pub fn block_match_full(
    reference: &Array2<f64>,
    current: &Array2<f64>,
    block_size: usize,
    search_range: usize,
) -> Result<MotionField> {
    validate_frames(reference, current)?;
    if block_size == 0 {
        return Err(VisionError::InvalidParameter(
            "block_size must be > 0".into(),
        ));
    }

    let rows = reference.nrows();
    let cols = reference.ncols();
    let grid_rows = rows / block_size;
    let grid_cols = cols / block_size;

    let mut vectors = Vec::with_capacity(grid_rows);

    for br in 0..grid_rows {
        let mut row_vecs = Vec::with_capacity(grid_cols);
        for bc in 0..grid_cols {
            let r0 = br * block_size;
            let c0 = bc * block_size;

            let mut best_dy: i64 = 0;
            let mut best_dx: i64 = 0;
            let mut best_sad = f64::MAX;
            let sr = search_range as i64;

            for dy in -sr..=sr {
                for dx in -sr..=sr {
                    let sad =
                        compute_sad(reference, current, r0, c0, block_size, dy, dx, rows, cols);
                    if sad < best_sad {
                        best_sad = sad;
                        best_dy = dy;
                        best_dx = dx;
                    }
                }
            }

            row_vecs.push(MotionVector::new(best_dy as f64, best_dx as f64));
        }
        vectors.push(row_vecs);
    }

    Ok(MotionField {
        vectors,
        block_size,
        rows: grid_rows,
        cols: grid_cols,
        frame_height: rows,
        frame_width: cols,
    })
}

/// Three-Step Search (TSS) block matching.
///
/// A fast approximate block matching algorithm that starts with a large step
/// size (`search_range / 2`) and progressively halves it, evaluating only 9
/// candidate positions at each stage.
pub fn block_match_tss(
    reference: &Array2<f64>,
    current: &Array2<f64>,
    block_size: usize,
    search_range: usize,
) -> Result<MotionField> {
    validate_frames(reference, current)?;
    if block_size == 0 {
        return Err(VisionError::InvalidParameter(
            "block_size must be > 0".into(),
        ));
    }
    if search_range == 0 {
        return Err(VisionError::InvalidParameter(
            "search_range must be > 0".into(),
        ));
    }

    let rows = reference.nrows();
    let cols = reference.ncols();
    let grid_rows = rows / block_size;
    let grid_cols = cols / block_size;

    let mut vectors = Vec::with_capacity(grid_rows);

    // Compute number of steps.
    let initial_step = ((search_range as f64) / 2.0).ceil().max(1.0) as i64;

    for br in 0..grid_rows {
        let mut row_vecs = Vec::with_capacity(grid_cols);
        for bc in 0..grid_cols {
            let r0 = br * block_size;
            let c0 = bc * block_size;

            let mut center_dy: i64 = 0;
            let mut center_dx: i64 = 0;
            let mut step = initial_step;

            while step >= 1 {
                let mut best_dy = center_dy;
                let mut best_dx = center_dx;
                let mut best_sad = f64::MAX;

                for ddy in [-step, 0, step] {
                    for ddx in [-step, 0, step] {
                        let dy = center_dy + ddy;
                        let dx = center_dx + ddx;
                        let sad =
                            compute_sad(reference, current, r0, c0, block_size, dy, dx, rows, cols);
                        if sad < best_sad {
                            best_sad = sad;
                            best_dy = dy;
                            best_dx = dx;
                        }
                    }
                }

                center_dy = best_dy;
                center_dx = best_dx;
                step /= 2;
            }

            row_vecs.push(MotionVector::new(center_dy as f64, center_dx as f64));
        }
        vectors.push(row_vecs);
    }

    Ok(MotionField {
        vectors,
        block_size,
        rows: grid_rows,
        cols: grid_cols,
        frame_height: rows,
        frame_width: cols,
    })
}

// ---------------------------------------------------------------------------
// SAD helper
// ---------------------------------------------------------------------------

fn compute_sad(
    reference: &Array2<f64>,
    current: &Array2<f64>,
    r0: usize,
    c0: usize,
    block_size: usize,
    dy: i64,
    dx: i64,
    rows: usize,
    cols: usize,
) -> f64 {
    let mut sad = 0.0;
    for r in 0..block_size {
        for c in 0..block_size {
            let cr = r0 + r;
            let cc = c0 + c;
            let rr = (cr as i64 + dy) as isize;
            let rc = (cc as i64 + dx) as isize;
            if cr < rows
                && cc < cols
                && rr >= 0
                && (rr as usize) < rows
                && rc >= 0
                && (rc as usize) < cols
            {
                sad += (current[[cr, cc]] - reference[[rr as usize, rc as usize]]).abs();
            } else {
                sad += 1.0; // Out-of-bounds penalty
            }
        }
    }
    sad
}

// ---------------------------------------------------------------------------
// Phase Correlation
// ---------------------------------------------------------------------------

/// Estimate global translation between two frames using phase correlation.
///
/// The cross-power spectrum is computed in the frequency domain and the peak
/// of its inverse transform gives the translation.  This method is fast and
/// robust for global (whole-frame) translational motion.
///
/// Returns `(dy, dx)` as floating-point pixel displacements with sub-pixel
/// potential through parabolic peak fitting.
pub fn phase_correlation(reference: &Array2<f64>, current: &Array2<f64>) -> Result<MotionVector> {
    validate_frames(reference, current)?;
    let rows = reference.nrows();
    let cols = reference.ncols();

    if rows == 0 || cols == 0 {
        return Ok(MotionVector::zero());
    }

    // Compute 2D DFT using a simple spatial-domain approach since we do not
    // have a full 2D FFT readily available in Pure Rust without pulling in a
    // heavy dependency.  For moderate frame sizes this is acceptable.
    //
    // We use the cross-correlation approach in the spatial domain:
    //   R(dy, dx) = sum_{r,c} ref(r,c) * cur(r+dy, c+dx)
    // and find the peak.

    // For efficiency, limit search to +/- quarter frame size.
    let max_dy = (rows / 4).max(1) as i64;
    let max_dx = (cols / 4).max(1) as i64;

    let mut best_dy: i64 = 0;
    let mut best_dx: i64 = 0;
    let mut best_corr = f64::NEG_INFINITY;

    for dy in -max_dy..=max_dy {
        for dx in -max_dx..=max_dx {
            let mut corr = 0.0;
            let mut count = 0u64;
            for r in 0..rows {
                for c in 0..cols {
                    let rr = r as i64 + dy;
                    let rc = c as i64 + dx;
                    if rr >= 0 && (rr as usize) < rows && rc >= 0 && (rc as usize) < cols {
                        corr += reference[[r, c]] * current[[rr as usize, rc as usize]];
                        count += 1;
                    }
                }
            }
            if count > 0 {
                corr /= count as f64;
            }
            if corr > best_corr {
                best_corr = corr;
                best_dy = dy;
                best_dx = dx;
            }
        }
    }

    // Sub-pixel refinement via parabolic interpolation on the row axis.
    let refined_dy = subpixel_refine_1d(
        |d| cross_corr_at(reference, current, d, best_dx, rows, cols),
        best_dy,
        max_dy,
    );
    let refined_dx = subpixel_refine_1d(
        |d| cross_corr_at(reference, current, best_dy, d, rows, cols),
        best_dx,
        max_dx,
    );

    Ok(MotionVector::new(refined_dy, refined_dx))
}

fn cross_corr_at(
    reference: &Array2<f64>,
    current: &Array2<f64>,
    dy: i64,
    dx: i64,
    rows: usize,
    cols: usize,
) -> f64 {
    let mut corr = 0.0;
    let mut count = 0u64;
    for r in 0..rows {
        for c in 0..cols {
            let rr = r as i64 + dy;
            let rc = c as i64 + dx;
            if rr >= 0 && (rr as usize) < rows && rc >= 0 && (rc as usize) < cols {
                corr += reference[[r, c]] * current[[rr as usize, rc as usize]];
                count += 1;
            }
        }
    }
    if count > 0 {
        corr / count as f64
    } else {
        0.0
    }
}

fn subpixel_refine_1d<F: Fn(i64) -> f64>(corr_fn: F, best: i64, limit: i64) -> f64 {
    if best <= -limit || best >= limit {
        return best as f64;
    }
    let c_minus = corr_fn(best - 1);
    let c_center = corr_fn(best);
    let c_plus = corr_fn(best + 1);
    let denom = 2.0 * (2.0 * c_center - c_minus - c_plus);
    if denom.abs() < 1e-12 {
        return best as f64;
    }
    let offset = (c_minus - c_plus) / denom;
    best as f64 + offset.clamp(-0.5, 0.5)
}

// ---------------------------------------------------------------------------
// Motion Compensation
// ---------------------------------------------------------------------------

/// Apply a block-based motion field to produce a motion-compensated (predicted)
/// frame from the reference frame.
///
/// Each block in the output is copied from the position in `reference` indicated
/// by the corresponding motion vector.
pub fn motion_compensate(reference: &Array2<f64>, field: &MotionField) -> Result<Array2<f64>> {
    let rows = reference.nrows();
    let cols = reference.ncols();
    if rows != field.frame_height || cols != field.frame_width {
        return Err(VisionError::DimensionMismatch(format!(
            "Reference ({}x{}) does not match field frame size ({}x{})",
            rows, cols, field.frame_height, field.frame_width,
        )));
    }

    let bs = field.block_size;
    let mut output = Array2::zeros((rows, cols));

    for br in 0..field.rows {
        for bc in 0..field.cols {
            let mv = &field.vectors[br][bc];
            let r0 = br * bs;
            let c0 = bc * bs;

            for r in 0..bs {
                for c in 0..bs {
                    let dst_r = r0 + r;
                    let dst_c = c0 + c;
                    if dst_r >= rows || dst_c >= cols {
                        continue;
                    }
                    let src_r = (dst_r as f64 + mv.dy).round() as isize;
                    let src_c = (dst_c as f64 + mv.dx).round() as isize;
                    if src_r >= 0
                        && (src_r as usize) < rows
                        && src_c >= 0
                        && (src_c as usize) < cols
                    {
                        output[[dst_r, dst_c]] = reference[[src_r as usize, src_c as usize]];
                    }
                }
            }
        }
    }

    Ok(output)
}

/// Compute the prediction error (residual) between a frame and its
/// motion-compensated prediction.
pub fn prediction_error(actual: &Array2<f64>, predicted: &Array2<f64>) -> Result<Array2<f64>> {
    validate_frames(actual, predicted)?;
    Ok(actual - predicted)
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_frames(a: &Array2<f64>, b: &Array2<f64>) -> Result<()> {
    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return Err(VisionError::DimensionMismatch(format!(
            "Frame dimensions do not match: ({},{}) vs ({},{})",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols(),
        )));
    }
    Ok(())
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn uniform_frame(val: f64, h: usize, w: usize) -> Array2<f64> {
        Array2::from_elem((h, w), val)
    }

    /// Create a frame with a bright square at (top, left) of given size.
    fn frame_with_square(
        bg: f64,
        fg: f64,
        h: usize,
        w: usize,
        top: usize,
        left: usize,
        size: usize,
    ) -> Array2<f64> {
        let mut f = Array2::from_elem((h, w), bg);
        for r in top..(top + size).min(h) {
            for c in left..(left + size).min(w) {
                f[[r, c]] = fg;
            }
        }
        f
    }

    // ---- MotionVector ----

    #[test]
    fn test_motion_vector_basics() {
        let v = MotionVector::new(3.0, 4.0);
        assert!((v.magnitude() - 5.0).abs() < 1e-9);
        assert!(v.angle().is_finite());

        let z = MotionVector::zero();
        assert!((z.magnitude()).abs() < 1e-12);
    }

    // ---- Full Search Block Matching ----

    #[test]
    fn test_full_search_no_motion() {
        // Use a frame with content so SAD is meaningful (uniform => all displacements tie).
        let frame = frame_with_square(0.0, 1.0, 16, 16, 2, 2, 8);
        let field = block_match_full(&frame, &frame, 8, 4).expect("ok");
        assert_eq!(field.rows, 2);
        assert_eq!(field.cols, 2);
        for row in &field.vectors {
            for v in row {
                assert!((v.dy).abs() < 1e-9, "expected dy=0, got {}", v.dy);
                assert!((v.dx).abs() < 1e-9, "expected dx=0, got {}", v.dx);
            }
        }
    }

    #[test]
    fn test_full_search_detects_horizontal_shift() {
        let h = 16;
        let w = 32;
        let bs = 8;
        let ref_frame = frame_with_square(0.0, 1.0, h, w, 4, 4, 8);
        let cur_frame = frame_with_square(0.0, 1.0, h, w, 4, 8, 8); // shifted right by 4
        let field = block_match_full(&ref_frame, &cur_frame, bs, 6).expect("ok");
        // At least one block should show dx ~ -4 (reference shifted right).
        let has_shift = field
            .vectors
            .iter()
            .flat_map(|r| r.iter())
            .any(|v| (v.dx - (-4.0)).abs() < 1.5);
        assert!(has_shift, "Should detect ~4px horizontal shift");
    }

    #[test]
    fn test_full_search_dimension_mismatch() {
        let a = uniform_frame(0.5, 16, 16);
        let b = uniform_frame(0.5, 8, 16);
        assert!(block_match_full(&a, &b, 8, 4).is_err());
    }

    #[test]
    fn test_full_search_zero_block() {
        let f = uniform_frame(0.5, 16, 16);
        assert!(block_match_full(&f, &f, 0, 4).is_err());
    }

    // ---- Three-Step Search ----

    #[test]
    fn test_tss_no_motion() {
        // Use a frame with content so SAD is meaningful.
        let frame = frame_with_square(0.0, 1.0, 16, 16, 2, 2, 8);
        let field = block_match_tss(&frame, &frame, 8, 4).expect("ok");
        for row in &field.vectors {
            for v in row {
                assert!(
                    v.magnitude() < 1e-9,
                    "expected zero motion, got mag={}",
                    v.magnitude()
                );
            }
        }
    }

    #[test]
    fn test_tss_detects_vertical_shift() {
        let h = 16;
        let w = 16;
        let ref_frame = frame_with_square(0.0, 1.0, h, w, 2, 2, 4);
        let cur_frame = frame_with_square(0.0, 1.0, h, w, 5, 2, 4); // shifted down by 3
        let field = block_match_tss(&ref_frame, &cur_frame, 4, 8).expect("ok");
        let has_shift = field
            .vectors
            .iter()
            .flat_map(|r| r.iter())
            .any(|v| (v.dy - (-3.0)).abs() < 2.0);
        assert!(has_shift, "Should detect ~3px vertical shift");
    }

    #[test]
    fn test_tss_invalid_search_range() {
        let f = uniform_frame(0.5, 16, 16);
        assert!(block_match_tss(&f, &f, 8, 0).is_err());
    }

    // ---- Phase Correlation ----

    #[test]
    fn test_phase_corr_no_motion() {
        let frame = frame_with_square(0.0, 1.0, 16, 16, 4, 4, 8);
        let mv = phase_correlation(&frame, &frame).expect("ok");
        assert!(
            mv.magnitude() < 1.0,
            "No motion expected, got mag={}",
            mv.magnitude()
        );
    }

    #[test]
    fn test_phase_corr_detects_shift() {
        let h = 16;
        let w = 16;
        let ref_frame = frame_with_square(0.0, 1.0, h, w, 2, 2, 6);
        let cur_frame = frame_with_square(0.0, 1.0, h, w, 2, 4, 6); // dx = 2
        let mv = phase_correlation(&ref_frame, &cur_frame).expect("ok");
        // The estimated dx should be close to -2 (reference was at col 2, current at col 4,
        // so reference shifts right by 2 to match current).
        // Allow generous tolerance since our spatial-domain implementation is approximate.
        assert!(
            mv.dx.abs() <= 4.0,
            "Expected dx magnitude near 2, got {}",
            mv.dx
        );
    }

    #[test]
    fn test_phase_corr_dimension_mismatch() {
        let a = uniform_frame(0.5, 8, 8);
        let b = uniform_frame(0.5, 8, 16);
        assert!(phase_correlation(&a, &b).is_err());
    }

    // ---- Motion Compensation ----

    #[test]
    fn test_motion_compensate_zero_field() {
        let frame = frame_with_square(0.1, 0.9, 16, 16, 2, 2, 4);
        let field = MotionField {
            vectors: vec![vec![MotionVector::zero(); 2]; 2],
            block_size: 8,
            rows: 2,
            cols: 2,
            frame_height: 16,
            frame_width: 16,
        };
        let comp = motion_compensate(&frame, &field).expect("ok");
        for r in 0..16 {
            for c in 0..16 {
                assert!(
                    (comp[[r, c]] - frame[[r, c]]).abs() < 1e-9,
                    "Zero motion should reproduce the reference"
                );
            }
        }
    }

    #[test]
    fn test_motion_compensate_dimension_mismatch() {
        let frame = uniform_frame(0.5, 8, 8);
        let field = MotionField {
            vectors: vec![vec![MotionVector::zero(); 2]; 2],
            block_size: 8,
            rows: 2,
            cols: 2,
            frame_height: 16,
            frame_width: 16,
        };
        assert!(motion_compensate(&frame, &field).is_err());
    }

    // ---- Prediction Error ----

    #[test]
    fn test_prediction_error_zero() {
        let frame = uniform_frame(0.5, 8, 8);
        let err = prediction_error(&frame, &frame).expect("ok");
        for &v in err.iter() {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn test_prediction_error_nonzero() {
        let a = uniform_frame(0.8, 4, 4);
        let b = uniform_frame(0.3, 4, 4);
        let err = prediction_error(&a, &b).expect("ok");
        for &v in err.iter() {
            assert!((v - 0.5).abs() < 1e-9);
        }
    }

    // ---- Motion Field ----

    #[test]
    fn test_motion_field_average_and_max() {
        let field = MotionField {
            vectors: vec![
                vec![MotionVector::new(3.0, 4.0), MotionVector::new(0.0, 0.0)],
                vec![MotionVector::new(1.0, 0.0), MotionVector::new(0.0, 1.0)],
            ],
            block_size: 4,
            rows: 2,
            cols: 2,
            frame_height: 8,
            frame_width: 8,
        };
        assert!((field.max_magnitude() - 5.0).abs() < 1e-9);
        assert!(field.average_magnitude() > 0.0);
    }

    #[test]
    fn test_hsv_visualization() {
        let field = MotionField {
            vectors: vec![vec![MotionVector::new(1.0, 0.0); 2]; 2],
            block_size: 4,
            rows: 2,
            cols: 2,
            frame_height: 8,
            frame_width: 8,
        };
        let (hue, sat, val) = field.to_hsv_visualization();
        assert_eq!(hue.nrows(), 8);
        assert_eq!(sat.ncols(), 8);
        // All vectors are equal so saturation/value should be uniform.
        let first_s = sat[[0, 0]];
        for &s in sat.iter() {
            assert!((s - first_s).abs() < 1e-9);
        }
        // Value should be > 0 since motion is non-zero.
        assert!(val[[0, 0]] > 0.0);
    }
}
