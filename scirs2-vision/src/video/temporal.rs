//! Temporal filtering and video stabilisation algorithms.
//!
//! Operates on sequences of single-channel `Array2<f64>` frames where pixel
//! values are expected in `[0, 1]`.
//!
//! # Algorithms
//!
//! - **Frame differencing** -- single and double (three-frame) differencing
//! - **Temporal median filtering** -- per-pixel median over a sliding window
//! - **Temporal Gaussian smoothing** -- per-pixel Gaussian-weighted average
//! - **Video stabilisation** -- smooth a motion trajectory to reduce jitter
//! - **Frame interpolation** -- linear and motion-compensated interpolation

use crate::error::{Result, VisionError};
use crate::video::motion::{block_match_full, motion_compensate, MotionField, MotionVector};
use scirs2_core::ndarray::Array2;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Frame Differencing
// ---------------------------------------------------------------------------

/// Compute the absolute difference between two consecutive frames (single
/// frame differencing).
///
/// The output is the per-pixel absolute difference, suitable for simple
/// change detection.
pub fn frame_difference(prev: &Array2<f64>, curr: &Array2<f64>) -> Result<Array2<f64>> {
    validate_pair(prev, curr)?;
    Ok((curr - prev).mapv(f64::abs))
}

/// Double (three-frame) differencing.
///
/// Computes the pixel-wise AND of `|curr - prev|` and `|next - curr|`,
/// which suppresses spurious detections caused by gradual illumination
/// changes.  The output is the element-wise minimum of the two differences.
pub fn double_frame_difference(
    prev: &Array2<f64>,
    curr: &Array2<f64>,
    next: &Array2<f64>,
) -> Result<Array2<f64>> {
    validate_pair(prev, curr)?;
    validate_pair(curr, next)?;
    let d1 = (curr - prev).mapv(f64::abs);
    let d2 = (next - curr).mapv(f64::abs);
    // Element-wise minimum.
    let rows = d1.nrows();
    let cols = d1.ncols();
    let mut result = Array2::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            result[[r, c]] = d1[[r, c]].min(d2[[r, c]]);
        }
    }
    Ok(result)
}

/// Threshold a difference image to produce a binary mask.
///
/// Pixels above `threshold` are set to 1.0, others to 0.0.
pub fn threshold_difference(diff: &Array2<f64>, threshold: f64) -> Array2<f64> {
    diff.mapv(|v| if v > threshold { 1.0 } else { 0.0 })
}

// ---------------------------------------------------------------------------
// Temporal Median Filter
// ---------------------------------------------------------------------------

/// Temporal median filter over a sliding window of frames.
///
/// Maintains an internal buffer of the last `window_size` frames and outputs
/// the per-pixel median.  This is effective at removing transient noise and
/// short-duration foreground objects.
#[derive(Debug, Clone)]
pub struct TemporalMedianFilter {
    /// Internal frame buffer.
    buffer: VecDeque<Array2<f64>>,
    /// Maximum window size.
    window_size: usize,
}

impl TemporalMedianFilter {
    /// Create a new temporal median filter.
    pub fn new(window_size: usize) -> Result<Self> {
        if window_size == 0 {
            return Err(VisionError::InvalidParameter(
                "window_size must be > 0".into(),
            ));
        }
        Ok(Self {
            buffer: VecDeque::with_capacity(window_size),
            window_size,
        })
    }

    /// Add a frame and return the current per-pixel median.
    pub fn apply(&mut self, frame: &Array2<f64>) -> Result<Array2<f64>> {
        // Validate dimensions against existing buffer.
        if let Some(first) = self.buffer.front() {
            if first.nrows() != frame.nrows() || first.ncols() != frame.ncols() {
                return Err(VisionError::DimensionMismatch(format!(
                    "Frame ({},{}) does not match buffer ({},{})",
                    frame.nrows(),
                    frame.ncols(),
                    first.nrows(),
                    first.ncols(),
                )));
            }
        }

        self.buffer.push_back(frame.clone());
        if self.buffer.len() > self.window_size {
            self.buffer.pop_front();
        }

        let n = self.buffer.len();
        let rows = frame.nrows();
        let cols = frame.ncols();
        let mut result = Array2::zeros((rows, cols));
        let mut pixel_buf = Vec::with_capacity(n);

        for r in 0..rows {
            for c in 0..cols {
                pixel_buf.clear();
                for f in &self.buffer {
                    pixel_buf.push(f[[r, c]]);
                }
                pixel_buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                result[[r, c]] = if n % 2 == 1 {
                    pixel_buf[n / 2]
                } else {
                    (pixel_buf[n / 2 - 1] + pixel_buf[n / 2]) / 2.0
                };
            }
        }

        Ok(result)
    }

    /// Number of frames currently in the buffer.
    pub fn buffered_frames(&self) -> usize {
        self.buffer.len()
    }

    /// Clear the internal buffer.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

// ---------------------------------------------------------------------------
// Temporal Gaussian Smoothing
// ---------------------------------------------------------------------------

/// Temporal Gaussian smoothing filter.
///
/// Each pixel in the output is the Gaussian-weighted average of the same
/// pixel across the buffered frames, with the most recent frame at the
/// centre of the kernel.
#[derive(Debug, Clone)]
pub struct TemporalGaussianFilter {
    /// Internal frame buffer.
    buffer: VecDeque<Array2<f64>>,
    /// Maximum window size (must be odd).
    window_size: usize,
    /// Pre-computed Gaussian weights.
    weights: Vec<f64>,
}

impl TemporalGaussianFilter {
    /// Create a new temporal Gaussian filter.
    ///
    /// `sigma` controls the temporal spread.  `window_size` must be odd and
    /// >= 1.
    pub fn new(window_size: usize, sigma: f64) -> Result<Self> {
        if window_size == 0 {
            return Err(VisionError::InvalidParameter(
                "window_size must be > 0".into(),
            ));
        }
        if sigma <= 0.0 {
            return Err(VisionError::InvalidParameter(
                "sigma must be positive".into(),
            ));
        }
        // Ensure odd.
        let ws = if window_size.is_multiple_of(2) {
            window_size + 1
        } else {
            window_size
        };
        let half = (ws / 2) as f64;
        let mut weights = Vec::with_capacity(ws);
        let mut sum = 0.0;
        for i in 0..ws {
            let x = i as f64 - half;
            let w = (-0.5 * (x / sigma).powi(2)).exp();
            weights.push(w);
            sum += w;
        }
        // Normalise.
        if sum > 0.0 {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        }

        Ok(Self {
            buffer: VecDeque::with_capacity(ws),
            window_size: ws,
            weights,
        })
    }

    /// Add a frame and return the temporally smoothed result.
    pub fn apply(&mut self, frame: &Array2<f64>) -> Result<Array2<f64>> {
        if let Some(first) = self.buffer.front() {
            if first.nrows() != frame.nrows() || first.ncols() != frame.ncols() {
                return Err(VisionError::DimensionMismatch(
                    "Frame dimensions do not match buffer".into(),
                ));
            }
        }

        self.buffer.push_back(frame.clone());
        if self.buffer.len() > self.window_size {
            self.buffer.pop_front();
        }

        let n = self.buffer.len();
        let rows = frame.nrows();
        let cols = frame.ncols();
        let mut result = Array2::zeros((rows, cols));

        // Use only the last `n` weights, right-aligned (most recent frame
        // maps to the last weight).
        let offset = self.window_size - n;

        for r in 0..rows {
            for c in 0..cols {
                let mut val = 0.0;
                let mut wsum = 0.0;
                for (i, f) in self.buffer.iter().enumerate() {
                    let w = self.weights[offset + i];
                    val += f[[r, c]] * w;
                    wsum += w;
                }
                result[[r, c]] = if wsum > 0.0 { val / wsum } else { 0.0 };
            }
        }

        Ok(result)
    }

    /// Number of buffered frames.
    pub fn buffered_frames(&self) -> usize {
        self.buffer.len()
    }

    /// Clear the buffer.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

// ---------------------------------------------------------------------------
// Video Stabilisation
// ---------------------------------------------------------------------------

/// Smooth a trajectory of 2-D motion estimates to reduce camera jitter.
///
/// Given a sequence of cumulative translations `(dx_i, dy_i)`, applies a
/// moving-average filter of the given `window` size and returns the smoothed
/// trajectory.  The caller can then compute the correction as the difference
/// between the original and smoothed trajectories.
///
/// Returns a vector of `(smoothed_dx, smoothed_dy)`.
pub fn smooth_trajectory(trajectory: &[(f64, f64)], window: usize) -> Result<Vec<(f64, f64)>> {
    if window == 0 {
        return Err(VisionError::InvalidParameter("window must be > 0".into()));
    }
    let n = trajectory.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let half = window / 2;
    let mut smoothed = Vec::with_capacity(n);

    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let count = (hi - lo) as f64;
        let mut sx = 0.0;
        let mut sy = 0.0;
        for &(tx, ty) in trajectory.iter().take(hi).skip(lo) {
            sx += tx;
            sy += ty;
        }
        smoothed.push((sx / count, sy / count));
    }

    Ok(smoothed)
}

/// Compute stabilisation correction vectors from an original and smoothed
/// trajectory.
///
/// Returns `correction[i] = (smooth[i].0 - orig[i].0, smooth[i].1 - orig[i].1)`.
pub fn stabilisation_corrections(
    original: &[(f64, f64)],
    smoothed: &[(f64, f64)],
) -> Result<Vec<(f64, f64)>> {
    if original.len() != smoothed.len() {
        return Err(VisionError::DimensionMismatch(
            "original and smoothed trajectory lengths differ".into(),
        ));
    }
    Ok(original
        .iter()
        .zip(smoothed.iter())
        .map(|((ox, oy), (sx, sy))| (sx - ox, sy - oy))
        .collect())
}

/// Apply a translational correction to a frame (simple shift + crop).
///
/// Pixels that fall outside the frame boundary after shifting are set to 0.
pub fn apply_translation(frame: &Array2<f64>, dy: f64, dx: f64) -> Array2<f64> {
    let rows = frame.nrows();
    let cols = frame.ncols();
    let mut out = Array2::zeros((rows, cols));
    let idy = dy.round() as isize;
    let idx = dx.round() as isize;

    for r in 0..rows {
        for c in 0..cols {
            let sr = r as isize - idy;
            let sc = c as isize - idx;
            if sr >= 0 && (sr as usize) < rows && sc >= 0 && (sc as usize) < cols {
                out[[r, c]] = frame[[sr as usize, sc as usize]];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Frame Interpolation
// ---------------------------------------------------------------------------

/// Linear (blend) frame interpolation.
///
/// Returns `(1 - t) * frame_a + t * frame_b` for `t` in `[0, 1]`.
pub fn interpolate_linear(
    frame_a: &Array2<f64>,
    frame_b: &Array2<f64>,
    t: f64,
) -> Result<Array2<f64>> {
    validate_pair(frame_a, frame_b)?;
    if !(0.0..=1.0).contains(&t) {
        return Err(VisionError::InvalidParameter("t must be in [0, 1]".into()));
    }
    Ok(frame_a * (1.0 - t) + frame_b * t)
}

/// Motion-compensated frame interpolation.
///
/// Estimates a motion field between `frame_a` and `frame_b` using block
/// matching, then warps both frames to the intermediate time `t` and blends
/// them.
///
/// # Arguments
/// * `frame_a` -- the first frame
/// * `frame_b` -- the second frame
/// * `t` -- interpolation factor in `[0, 1]`
/// * `block_size` -- block size for motion estimation (e.g. 8)
/// * `search_range` -- search range for block matching (e.g. 8)
pub fn interpolate_motion_compensated(
    frame_a: &Array2<f64>,
    frame_b: &Array2<f64>,
    t: f64,
    block_size: usize,
    search_range: usize,
) -> Result<Array2<f64>> {
    validate_pair(frame_a, frame_b)?;
    if !(0.0..=1.0).contains(&t) {
        return Err(VisionError::InvalidParameter("t must be in [0, 1]".into()));
    }
    if block_size == 0 {
        return Err(VisionError::InvalidParameter(
            "block_size must be > 0".into(),
        ));
    }

    // Estimate forward motion field from A to B.
    let field_ab = block_match_full(frame_a, frame_b, block_size, search_range)?;

    // Scale motion vectors by t and (1-t) for the two warps.
    let rows = field_ab.rows;
    let cols = field_ab.cols;

    // Warp A forward by t * motion.
    let mut field_a_to_t = field_ab.clone();
    for br in 0..rows {
        for bc in 0..cols {
            let v = &field_ab.vectors[br][bc];
            field_a_to_t.vectors[br][bc] = MotionVector::new(v.dy * t, v.dx * t);
        }
    }
    let warped_a = motion_compensate(frame_a, &field_a_to_t)?;

    // Warp B backward by (1-t) * motion.
    let mut field_b_to_t = field_ab.clone();
    for br in 0..rows {
        for bc in 0..cols {
            let v = &field_ab.vectors[br][bc];
            field_b_to_t.vectors[br][bc] = MotionVector::new(-v.dy * (1.0 - t), -v.dx * (1.0 - t));
        }
    }
    let warped_b = motion_compensate(frame_b, &field_b_to_t)?;

    // Blend.
    Ok(&warped_a * (1.0 - t) + &warped_b * t)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_pair(a: &Array2<f64>, b: &Array2<f64>) -> Result<()> {
    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return Err(VisionError::DimensionMismatch(format!(
            "Frame dimensions ({},{}) vs ({},{})",
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

    fn uniform(val: f64, h: usize, w: usize) -> Array2<f64> {
        Array2::from_elem((h, w), val)
    }

    fn with_square(
        bg: f64,
        fg: f64,
        h: usize,
        w: usize,
        t: usize,
        l: usize,
        s: usize,
    ) -> Array2<f64> {
        let mut f = Array2::from_elem((h, w), bg);
        for r in t..(t + s).min(h) {
            for c in l..(l + s).min(w) {
                f[[r, c]] = fg;
            }
        }
        f
    }

    // ---- Frame Differencing ----

    #[test]
    fn test_frame_diff_identical() {
        let f = uniform(0.5, 8, 8);
        let diff = frame_difference(&f, &f).expect("ok");
        for &v in diff.iter() {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn test_frame_diff_detects_change() {
        let a = uniform(0.0, 8, 8);
        let b = with_square(0.0, 1.0, 8, 8, 2, 2, 3);
        let diff = frame_difference(&a, &b).expect("ok");
        // Changed pixels should be 1.0.
        assert!((diff[[2, 2]] - 1.0).abs() < 1e-9);
        // Unchanged pixels should be 0.
        assert!(diff[[0, 0]].abs() < 1e-12);
    }

    #[test]
    fn test_frame_diff_dimension_mismatch() {
        let a = uniform(0.5, 8, 8);
        let b = uniform(0.5, 4, 8);
        assert!(frame_difference(&a, &b).is_err());
    }

    #[test]
    fn test_double_frame_diff() {
        let prev = uniform(0.0, 8, 8);
        let curr = with_square(0.0, 1.0, 8, 8, 2, 2, 3);
        let next = with_square(0.0, 1.0, 8, 8, 2, 2, 3);
        let d = double_frame_difference(&prev, &curr, &next).expect("ok");
        // The object is in both curr and next, so d2 = |next-curr| = 0 for object pixels
        // => double diff = min(1, 0) = 0 for those pixels.
        assert!(
            d[[2, 2]].abs() < 1e-9,
            "Stationary object should be suppressed"
        );
    }

    #[test]
    fn test_double_frame_diff_moving() {
        let prev = uniform(0.0, 8, 8);
        let curr = with_square(0.0, 1.0, 8, 8, 2, 2, 2);
        let next = with_square(0.0, 1.0, 8, 8, 4, 4, 2); // moved
        let d = double_frame_difference(&prev, &curr, &next).expect("ok");
        // At pixel (2,2): d1=1, d2=1 => min=1.
        assert!(d[[2, 2]] > 0.5, "Moving object should be detected");
    }

    #[test]
    fn test_threshold_difference() {
        let mut diff = uniform(0.0, 4, 4);
        diff[[0, 0]] = 0.3;
        diff[[1, 1]] = 0.6;
        let mask = threshold_difference(&diff, 0.5);
        assert!(mask[[0, 0]].abs() < 1e-12, "0.3 < 0.5 => 0");
        assert!((mask[[1, 1]] - 1.0).abs() < 1e-12, "0.6 > 0.5 => 1");
    }

    // ---- Temporal Median Filter ----

    #[test]
    fn test_temporal_median_constant() {
        let mut filter = TemporalMedianFilter::new(5).expect("ok");
        let f = uniform(0.5, 4, 4);
        for _ in 0..5 {
            let result = filter.apply(&f).expect("ok");
            for &v in result.iter() {
                assert!((v - 0.5).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_temporal_median_removes_outlier() {
        let mut filter = TemporalMedianFilter::new(5).expect("ok");
        let normal = uniform(0.3, 4, 4);
        let outlier = uniform(0.9, 4, 4);
        filter.apply(&normal).expect("ok");
        filter.apply(&normal).expect("ok");
        filter.apply(&outlier).expect("ok"); // one outlier
        filter.apply(&normal).expect("ok");
        let result = filter.apply(&normal).expect("ok");
        // Median of [0.3, 0.3, 0.9, 0.3, 0.3] = 0.3
        for &v in result.iter() {
            assert!(
                (v - 0.3).abs() < 1e-9,
                "Outlier should be filtered out, got {v}"
            );
        }
    }

    #[test]
    fn test_temporal_median_invalid_window() {
        assert!(TemporalMedianFilter::new(0).is_err());
    }

    #[test]
    fn test_temporal_median_dimension_mismatch() {
        let mut filter = TemporalMedianFilter::new(3).expect("ok");
        filter.apply(&uniform(0.5, 4, 4)).expect("ok");
        assert!(filter.apply(&uniform(0.5, 8, 8)).is_err());
    }

    #[test]
    fn test_temporal_median_reset() {
        let mut filter = TemporalMedianFilter::new(3).expect("ok");
        filter.apply(&uniform(0.5, 4, 4)).expect("ok");
        assert_eq!(filter.buffered_frames(), 1);
        filter.reset();
        assert_eq!(filter.buffered_frames(), 0);
    }

    // ---- Temporal Gaussian Smoothing ----

    #[test]
    fn test_temporal_gaussian_constant() {
        let mut filter = TemporalGaussianFilter::new(5, 1.0).expect("ok");
        let f = uniform(0.7, 4, 4);
        for _ in 0..5 {
            let result = filter.apply(&f).expect("ok");
            for &v in result.iter() {
                assert!(
                    (v - 0.7).abs() < 0.01,
                    "Constant input should give ~0.7, got {v}"
                );
            }
        }
    }

    #[test]
    fn test_temporal_gaussian_smooths() {
        let mut filter = TemporalGaussianFilter::new(5, 1.5).expect("ok");
        let low = uniform(0.2, 4, 4);
        let high = uniform(0.8, 4, 4);
        filter.apply(&low).expect("ok");
        filter.apply(&low).expect("ok");
        let result = filter.apply(&high).expect("ok");
        // Should be between low and high.
        for &v in result.iter() {
            assert!(v > 0.15 && v < 0.85, "Should be smoothed, got {v}");
        }
    }

    #[test]
    fn test_temporal_gaussian_invalid() {
        assert!(TemporalGaussianFilter::new(0, 1.0).is_err());
        assert!(TemporalGaussianFilter::new(5, 0.0).is_err());
        assert!(TemporalGaussianFilter::new(5, -1.0).is_err());
    }

    // ---- Video Stabilisation ----

    #[test]
    fn test_smooth_trajectory_constant() {
        let traj: Vec<(f64, f64)> = (0..10).map(|i| (i as f64, 0.0)).collect();
        let smoothed = smooth_trajectory(&traj, 3).expect("ok");
        assert_eq!(smoothed.len(), 10);
        // Middle elements should be close to original for a linear trajectory.
        for i in 2..8 {
            assert!(
                (smoothed[i].0 - traj[i].0).abs() < 1.5,
                "Smoothed should be close to original linear trajectory"
            );
        }
    }

    #[test]
    fn test_smooth_trajectory_reduces_jitter() {
        // Trajectory with jitter: 0, 2, 0, 2, 0, 2
        let traj: Vec<(f64, f64)> = (0..6)
            .map(|i| (if i % 2 == 0 { 0.0 } else { 2.0 }, 0.0))
            .collect();
        let smoothed = smooth_trajectory(&traj, 3).expect("ok");
        // Smoothed values should have less variance.
        let orig_var: f64 = traj.iter().map(|(x, _)| (x - 1.0).powi(2)).sum::<f64>() / 6.0;
        let smooth_var: f64 = smoothed.iter().map(|(x, _)| (x - 1.0).powi(2)).sum::<f64>() / 6.0;
        assert!(
            smooth_var < orig_var,
            "Smoothed variance ({smooth_var}) should be less than original ({orig_var})"
        );
    }

    #[test]
    fn test_smooth_trajectory_empty() {
        let smoothed = smooth_trajectory(&[], 3).expect("ok");
        assert!(smoothed.is_empty());
    }

    #[test]
    fn test_smooth_trajectory_invalid_window() {
        assert!(smooth_trajectory(&[(0.0, 0.0)], 0).is_err());
    }

    #[test]
    fn test_stabilisation_corrections() {
        let orig = vec![(1.0, 2.0), (3.0, 4.0)];
        let smooth = vec![(1.5, 2.5), (2.5, 3.5)];
        let corr = stabilisation_corrections(&orig, &smooth).expect("ok");
        assert!((corr[0].0 - 0.5).abs() < 1e-9);
        assert!((corr[0].1 - 0.5).abs() < 1e-9);
        assert!((corr[1].0 - (-0.5)).abs() < 1e-9);
    }

    #[test]
    fn test_stabilisation_corrections_mismatch() {
        assert!(stabilisation_corrections(&[(0.0, 0.0)], &[]).is_err());
    }

    #[test]
    fn test_apply_translation_zero() {
        let f = with_square(0.0, 1.0, 8, 8, 2, 2, 3);
        let shifted = apply_translation(&f, 0.0, 0.0);
        for r in 0..8 {
            for c in 0..8 {
                assert!((shifted[[r, c]] - f[[r, c]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_apply_translation_shift() {
        let f = with_square(0.0, 1.0, 8, 8, 2, 2, 3);
        let shifted = apply_translation(&f, 1.0, 0.0); // shift down by 1
                                                       // Original bright pixel at (2,2) should now appear at (3,2).
        assert!((shifted[[3, 2]] - 1.0).abs() < 1e-9);
        // (2,2) in shifted should come from (1,2) in original => 0.
        assert!(shifted[[2, 2]].abs() < 1e-9);
    }

    // ---- Frame Interpolation ----

    #[test]
    fn test_interpolate_linear_endpoints() {
        let a = uniform(0.0, 8, 8);
        let b = uniform(1.0, 8, 8);
        let at0 = interpolate_linear(&a, &b, 0.0).expect("ok");
        let at1 = interpolate_linear(&a, &b, 1.0).expect("ok");
        for &v in at0.iter() {
            assert!(v.abs() < 1e-12, "t=0 should give frame_a");
        }
        for &v in at1.iter() {
            assert!((v - 1.0).abs() < 1e-12, "t=1 should give frame_b");
        }
    }

    #[test]
    fn test_interpolate_linear_midpoint() {
        let a = uniform(0.0, 4, 4);
        let b = uniform(1.0, 4, 4);
        let mid = interpolate_linear(&a, &b, 0.5).expect("ok");
        for &v in mid.iter() {
            assert!((v - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn test_interpolate_linear_invalid_t() {
        let a = uniform(0.0, 4, 4);
        let b = uniform(1.0, 4, 4);
        assert!(interpolate_linear(&a, &b, -0.1).is_err());
        assert!(interpolate_linear(&a, &b, 1.1).is_err());
    }

    #[test]
    fn test_interpolate_linear_mismatch() {
        let a = uniform(0.0, 4, 4);
        let b = uniform(1.0, 8, 8);
        assert!(interpolate_linear(&a, &b, 0.5).is_err());
    }

    #[test]
    fn test_interpolate_motion_compensated() {
        let h = 16;
        let w = 16;
        let a = with_square(0.0, 1.0, h, w, 2, 2, 4);
        let b = with_square(0.0, 1.0, h, w, 2, 6, 4); // shifted right
        let mid = interpolate_motion_compensated(&a, &b, 0.5, 4, 8).expect("ok");
        // The result should have some non-zero content.
        let sum: f64 = mid.iter().sum();
        assert!(sum > 0.0, "Interpolated frame should have content");
    }

    #[test]
    fn test_interpolate_motion_compensated_invalid() {
        let a = uniform(0.0, 16, 16);
        let b = uniform(1.0, 16, 16);
        assert!(interpolate_motion_compensated(&a, &b, 0.5, 0, 4).is_err());
        assert!(interpolate_motion_compensated(&a, &b, -0.1, 4, 4).is_err());
    }
}
