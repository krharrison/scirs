//! Video stabilisation algorithms.
//!
//! This module estimates and smooths inter-frame camera motion to reduce the
//! appearance of hand-shake or platform vibration in a video sequence.
//!
//! # Pipeline
//!
//! 1. **`estimate_global_motion`** – estimate an affine transform between
//!    consecutive frames using a combination of phase correlation (translation)
//!    and a patch-based affine refinement.
//! 2. **`smooth_trajectory`** – apply a causal moving-average window to the
//!    cumulative transform trajectory, producing a smoothed path.
//! 3. **`stabilize_sequence`** – warp each frame with the correction transform
//!    (smoothed − raw) to cancel unwanted motion.
//! 4. **`crop_stabilized`** – crop a border to remove black regions introduced
//!    by the warping.
//!
//! # Utility
//!
//! - [`affine_warp`] – apply a 2×3 [`Affine2D`] matrix to a grayscale image.

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::{Array2, Array3};

// ---------------------------------------------------------------------------
// Affine2D transform
// ---------------------------------------------------------------------------

/// A 2×3 affine transformation matrix (row-major).
///
/// Stored as:
/// ```text
/// [ a  b  tx ]
/// [ c  d  ty ]
/// ```
/// Applying the transform to a point `(x, y)`:
/// ```text
/// x' = a·x + b·y + tx
/// y' = c·x + d·y + ty
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Affine2D {
    /// 2×3 coefficient matrix stored row-major: [a, b, tx, c, d, ty].
    pub m: [f64; 6],
}

impl Affine2D {
    /// Create from raw coefficients `[a, b, tx, c, d, ty]`.
    pub fn new(a: f64, b: f64, tx: f64, c: f64, d: f64, ty: f64) -> Self {
        Self {
            m: [a, b, tx, c, d, ty],
        }
    }

    /// Identity transform.
    pub fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    }

    /// Pure translation.
    pub fn translation(tx: f64, ty: f64) -> Self {
        Self::new(1.0, 0.0, tx, 0.0, 1.0, ty)
    }

    /// Apply this transform to a 2D point `(x, y)` → `(x', y')`.
    pub fn apply_point(&self, x: f64, y: f64) -> (f64, f64) {
        let xp = self.m[0] * x + self.m[1] * y + self.m[2];
        let yp = self.m[3] * x + self.m[4] * y + self.m[5];
        (xp, yp)
    }

    /// Compose `self` after `other`: result(p) = self(other(p)).
    pub fn compose(&self, other: &Affine2D) -> Affine2D {
        // Equivalent to 3×3 homogeneous matrix multiply (last row [0,0,1]).
        let a = self.m[0] * other.m[0] + self.m[1] * other.m[3];
        let b = self.m[0] * other.m[1] + self.m[1] * other.m[4];
        let tx = self.m[0] * other.m[2] + self.m[1] * other.m[5] + self.m[2];
        let c = self.m[3] * other.m[0] + self.m[4] * other.m[3];
        let d = self.m[3] * other.m[1] + self.m[4] * other.m[4];
        let ty = self.m[3] * other.m[2] + self.m[4] * other.m[5] + self.m[5];
        Affine2D::new(a, b, tx, c, d, ty)
    }

    /// Approximate inverse via 2×2 adjugate.
    pub fn inverse(&self) -> Result<Affine2D> {
        let det = self.m[0] * self.m[4] - self.m[1] * self.m[3];
        if det.abs() < 1e-12 {
            return Err(VisionError::OperationError(
                "Affine2D::inverse: matrix is singular".into(),
            ));
        }
        let inv_det = 1.0 / det;
        let a = self.m[4] * inv_det;
        let b = -self.m[1] * inv_det;
        let c = -self.m[3] * inv_det;
        let d = self.m[0] * inv_det;
        let tx = -(a * self.m[2] + b * self.m[5]);
        let ty = -(c * self.m[2] + d * self.m[5]);
        Ok(Affine2D::new(a, b, tx, c, d, ty))
    }

    /// Element-wise weighted average of two transforms (used for smoothing).
    pub fn lerp(&self, other: &Affine2D, t: f64) -> Affine2D {
        let mut m = [0.0f64; 6];
        #[allow(clippy::needless_range_loop)]
        for i in 0..6 {
            m[i] = self.m[i] * (1.0 - t) + other.m[i] * t;
        }
        Affine2D { m }
    }
}

// ---------------------------------------------------------------------------
// Global motion estimation
// ---------------------------------------------------------------------------

/// Estimate the affine global motion between two single-channel frames.
///
/// Uses phase correlation to estimate the dominant translation, then refines
/// with a simple least-squares patch-based affine fit using a sparse grid
/// of patches.
///
/// Returns an [`Affine2D`] that maps points in `frame1` to their approximate
/// corresponding locations in `frame2`.
pub fn estimate_global_motion(frame1: &Array2<f64>, frame2: &Array2<f64>) -> Result<Affine2D> {
    let shape = frame1.dim();
    if shape != frame2.dim() {
        return Err(VisionError::DimensionMismatch(
            "estimate_global_motion: frames must have identical shapes".into(),
        ));
    }
    let (rows, cols) = shape;

    // --- Phase correlation for translation component ---
    let (tx, ty) = phase_correlation_translation(frame1, frame2);

    // --- Patch-based affine refinement ---
    // Sample a 4×4 grid of small patches and find their best-match offsets.
    let patch_size = 8usize;
    let search = 4usize; // search radius
    let grid_rows = 4usize;
    let grid_cols = 4usize;

    let step_r = rows / (grid_rows + 1);
    let step_c = cols / (grid_cols + 1);

    // Points in frame1 (source) and their matched locations in frame2 (dest).
    let mut src_pts: Vec<[f64; 2]> = Vec::new();
    let mut dst_pts: Vec<[f64; 2]> = Vec::new();

    for gr in 1..=grid_rows {
        for gc in 1..=grid_cols {
            let r0 = gr * step_r;
            let c0 = gc * step_c;

            if r0 + patch_size >= rows || c0 + patch_size >= cols {
                continue;
            }

            let (dr, dc) = find_patch_match(frame1, frame2, r0, c0, patch_size, search);

            src_pts.push([
                c0 as f64 + patch_size as f64 / 2.0,
                r0 as f64 + patch_size as f64 / 2.0,
            ]);
            dst_pts.push([
                c0 as f64 + patch_size as f64 / 2.0 + dc as f64,
                r0 as f64 + patch_size as f64 / 2.0 + dr as f64,
            ]);
        }
    }

    // If we have enough points fit a least-squares affine; otherwise fall back
    // to the translation-only estimate.
    if src_pts.len() >= 3 {
        if let Ok(aff) = fit_affine_ls(&src_pts, &dst_pts) {
            return Ok(aff);
        }
    }

    // Translation fallback.
    Ok(Affine2D::translation(tx, ty))
}

/// Simple patch matching via sum-of-absolute-differences.
///
/// Returns `(best_dr, best_dc)` in pixels.
fn find_patch_match(
    frame1: &Array2<f64>,
    frame2: &Array2<f64>,
    r0: usize,
    c0: usize,
    patch_size: usize,
    search: usize,
) -> (i32, i32) {
    let (rows, cols) = frame1.dim();
    let mut best_sad = f64::MAX;
    let mut best_dr = 0i32;
    let mut best_dc = 0i32;

    let s = search as i32;
    for dr in -s..=s {
        for dc in -s..=s {
            let r1 = r0 as i32 + dr;
            let c1 = c0 as i32 + dc;
            if r1 < 0
                || c1 < 0
                || (r1 as usize) + patch_size > rows
                || (c1 as usize) + patch_size > cols
            {
                continue;
            }
            let mut sad = 0.0_f64;
            for pr in 0..patch_size {
                for pc in 0..patch_size {
                    let v1 = frame1[[r0 + pr, c0 + pc]];
                    let v2 = frame2[[(r1 as usize) + pr, (c1 as usize) + pc]];
                    sad += (v1 - v2).abs();
                }
            }
            // Prefer smaller displacement when SAD values are tied (e.g.
            // uniform images where all patches match equally well).
            let cur_disp = (best_dr.abs() + best_dc.abs()) as f64;
            let new_disp = (dr.abs() + dc.abs()) as f64;
            if sad < best_sad || (sad == best_sad && new_disp < cur_disp) {
                best_sad = sad;
                best_dr = dr;
                best_dc = dc;
            }
        }
    }
    (best_dr, best_dc)
}

/// Least-squares affine fit from corresponding 2D point pairs.
///
/// Solves `A · m = b` where `m = [a, b, tx]ᵀ` and `[c, d, ty]ᵀ` separately.
fn fit_affine_ls(src: &[[f64; 2]], dst: &[[f64; 2]]) -> Result<Affine2D> {
    let n = src.len();
    if n < 3 {
        return Err(VisionError::InvalidParameter(
            "fit_affine_ls: need at least 3 point pairs".into(),
        ));
    }

    // Build matrix A (n×3) and right-hand sides bx, by.
    let mut ata = [[0.0f64; 3]; 3];
    let mut atbx = [0.0f64; 3];
    let mut atby = [0.0f64; 3];

    for i in 0..n {
        let row = [src[i][0], src[i][1], 1.0];
        for j in 0..3 {
            for k in 0..3 {
                ata[j][k] += row[j] * row[k];
            }
            atbx[j] += row[j] * dst[i][0];
            atby[j] += row[j] * dst[i][1];
        }
    }

    let mx = solve_3x3(&ata, &atbx)?;
    let my = solve_3x3(&ata, &atby)?;

    Ok(Affine2D::new(mx[0], mx[1], mx[2], my[0], my[1], my[2]))
}

/// Solve a 3×3 linear system via Cramer's rule.
fn solve_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Result<[f64; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    if det.abs() < 1e-12 {
        return Err(VisionError::LinAlgError(
            "solve_3x3: singular matrix".into(),
        ));
    }

    // Augment A with b and compute each solution component via Cramer.
    let mut result = [0.0f64; 3];
    for col in 0..3 {
        let mut am = *a;
        for row in 0..3 {
            am[row][col] = b[row];
        }
        let det_i = am[0][0] * (am[1][1] * am[2][2] - am[1][2] * am[2][1])
            - am[0][1] * (am[1][0] * am[2][2] - am[1][2] * am[2][0])
            + am[0][2] * (am[1][0] * am[2][1] - am[1][1] * am[2][0]);
        result[col] = det_i / det;
    }
    Ok(result)
}

/// Estimate dominant translation via real-space cross-correlation (no FFT dependency).
///
/// Searches an integer displacement window of ±`max_shift` pixels using
/// normalized cross-correlation (NCC).
fn phase_correlation_translation(frame1: &Array2<f64>, frame2: &Array2<f64>) -> (f64, f64) {
    let (rows, cols) = frame1.dim();
    let max_shift = 16usize;

    // Down-sample to at most 64×64 for speed.
    let scale = 1usize.max(rows.max(cols) / 64);
    let sr = (rows / scale).max(1);
    let sc = (cols / scale).max(1);

    let ds1 = downsample(frame1, sr, sc);
    let ds2 = downsample(frame2, sr, sc);

    let shift_r = max_shift / scale;
    let shift_c = max_shift / scale;

    let mut best_ncc = -1.0_f64;
    let mut best_dy = 0i32;
    let mut best_dx = 0i32;

    let sr_i = shift_r as i32;
    let sc_i = shift_c as i32;

    for dy in -sr_i..=sr_i {
        for dx in -sc_i..=sc_i {
            let ncc = compute_ncc(&ds1, &ds2, dy, dx);
            let cur_disp = (best_dy.abs() + best_dx.abs()) as f64;
            let new_disp = (dy.abs() + dx.abs()) as f64;
            if ncc > best_ncc || (ncc == best_ncc && new_disp < cur_disp) {
                best_ncc = ncc;
                best_dy = dy;
                best_dx = dx;
            }
        }
    }

    (best_dx as f64 * scale as f64, best_dy as f64 * scale as f64)
}

/// Simple average downsampling.
fn downsample(src: &Array2<f64>, target_rows: usize, target_cols: usize) -> Array2<f64> {
    let (sr, sc) = src.dim();
    let mut dst = Array2::<f64>::zeros((target_rows, target_cols));
    let block_r = sr / target_rows;
    let block_c = sc / target_cols;
    for r in 0..target_rows {
        for c in 0..target_cols {
            let mut sum = 0.0;
            let mut cnt = 0usize;
            for br in 0..block_r {
                for bc in 0..block_c {
                    let sr_idx = (r * block_r + br).min(sr - 1);
                    let sc_idx = (c * block_c + bc).min(sc - 1);
                    sum += src[[sr_idx, sc_idx]];
                    cnt += 1;
                }
            }
            dst[[r, c]] = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
        }
    }
    dst
}

/// Normalised cross-correlation between two images at integer shift (dy, dx).
fn compute_ncc(a: &Array2<f64>, b: &Array2<f64>, dy: i32, dx: i32) -> f64 {
    let (rows, cols) = a.dim();
    let r0 = (-dy).max(0) as usize;
    let r1 = rows.min(rows.saturating_sub(dy.max(0) as usize));
    let c0 = (-dx).max(0) as usize;
    let c1 = cols.min(cols.saturating_sub(dx.max(0) as usize));

    if r0 >= r1 || c0 >= c1 {
        return -1.0;
    }

    let mut sum_a = 0.0_f64;
    let mut sum_b = 0.0_f64;
    let mut count = 0usize;

    for r in r0..r1 {
        for c in c0..c1 {
            let br = (r as i32 + dy) as usize;
            let bc = (c as i32 + dx) as usize;
            if br < rows && bc < cols {
                sum_a += a[[r, c]];
                sum_b += b[[br, bc]];
                count += 1;
            }
        }
    }

    if count == 0 {
        return -1.0;
    }

    let mean_a = sum_a / count as f64;
    let mean_b = sum_b / count as f64;

    let mut num = 0.0_f64;
    let mut den_a = 0.0_f64;
    let mut den_b = 0.0_f64;

    for r in r0..r1 {
        for c in c0..c1 {
            let br = (r as i32 + dy) as usize;
            let bc = (c as i32 + dx) as usize;
            if br < rows && bc < cols {
                let da = a[[r, c]] - mean_a;
                let db = b[[br, bc]] - mean_b;
                num += da * db;
                den_a += da * da;
                den_b += db * db;
            }
        }
    }

    let denom = (den_a * den_b).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        num / denom
    }
}

// ---------------------------------------------------------------------------
// Trajectory smoothing
// ---------------------------------------------------------------------------

/// Smooth a sequence of cumulative transforms using a causal moving average.
///
/// Each element of the output is the average of the previous `window`
/// transforms (including the current one).  A larger window yields stronger
/// stabilisation at the cost of a larger temporal delay.
///
/// # Arguments
///
/// * `transforms` – per-frame global motion estimates (from `estimate_global_motion`)
/// * `window`     – smoothing window size in frames (≥ 1)
pub fn smooth_trajectory(transforms: &[Affine2D], window: usize) -> Result<Vec<Affine2D>> {
    if transforms.is_empty() {
        return Ok(Vec::new());
    }
    let window = window.max(1);

    // Accumulate the cumulative (integrated) path.
    let n = transforms.len();
    let mut cumulative: Vec<Affine2D> = Vec::with_capacity(n);
    let mut cum = Affine2D::identity();
    for t in transforms {
        cum = cum.compose(t);
        cumulative.push(cum.clone());
    }

    // Apply causal moving average to each coefficient.
    let mut smoothed: Vec<Affine2D> = Vec::with_capacity(n);
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let count = (i - start + 1) as f64;
        let mut avg = [0.0f64; 6];
        #[allow(clippy::needless_range_loop)]
        for j in start..=i {
            #[allow(clippy::needless_range_loop)]
            for k in 0..6 {
                avg[k] += cumulative[j].m[k];
            }
        }
        for avg_k in avg.iter_mut() {
            *avg_k /= count;
        }
        smoothed.push(Affine2D { m: avg });
    }

    Ok(smoothed)
}

// ---------------------------------------------------------------------------
// Stabilize a sequence
// ---------------------------------------------------------------------------

/// Stabilise a sequence of RGB frames.
///
/// For each frame the correction transform `smoothed[i] · raw[i]⁻¹` is
/// applied so that the camera appears to follow the smoothed trajectory.
///
/// # Arguments
///
/// * `frames`         – sequence of `[H, W, 3]` frames
/// * `transforms`     – per-frame motion estimates (length == `frames.len() - 1`)
/// * `smooth_window`  – smoothing window for `smooth_trajectory`
pub fn stabilize_sequence(
    frames: &[Array3<f64>],
    transforms: &[Affine2D],
    smooth_window: usize,
) -> Result<Vec<Array3<f64>>> {
    if frames.is_empty() {
        return Ok(Vec::new());
    }
    if !transforms.is_empty() && transforms.len() != frames.len() - 1 {
        return Err(VisionError::InvalidParameter(format!(
            "stabilize_sequence: expected {} transforms for {} frames, got {}",
            frames.len() - 1,
            frames.len(),
            transforms.len()
        )));
    }

    // Build per-frame cumulative raw transforms (identity for frame 0).
    let n = frames.len();
    let mut raw_cum: Vec<Affine2D> = Vec::with_capacity(n);
    raw_cum.push(Affine2D::identity());
    let mut cum = Affine2D::identity();
    for t in transforms {
        cum = cum.compose(t);
        raw_cum.push(cum.clone());
    }

    // Smooth the trajectory.
    let smoothed = smooth_trajectory(&raw_cum, smooth_window)?;

    // For each frame, correction = smoothed[i] · raw_cum[i]⁻¹.
    let mut output: Vec<Array3<f64>> = Vec::with_capacity(n);
    for (i, frame) in frames.iter().enumerate() {
        let raw_inv = raw_cum[i].inverse()?;
        let correction = smoothed[i].compose(&raw_inv);
        output.push(affine_warp_rgb(frame, &correction)?);
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Affine warp (grayscale + colour helpers)
// ---------------------------------------------------------------------------

/// Apply an affine transform to a single-channel image using bilinear interpolation.
///
/// For each output pixel `(x, y)` the source sample is pulled from
/// `transform.inverse()(x, y)` (inverse mapping).
pub fn affine_warp(image: &Array2<f64>, transform: &Affine2D) -> Result<Array2<f64>> {
    let (rows, cols) = image.dim();
    let inv = transform.inverse()?;
    let mut output = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let (src_x, src_y) = inv.apply_point(c as f64, r as f64);
            output[[r, c]] = bilinear_sample(image, src_y, src_x, rows, cols);
        }
    }
    Ok(output)
}

/// Apply an affine transform to an RGB `[H, W, 3]` frame.
fn affine_warp_rgb(frame: &Array3<f64>, transform: &Affine2D) -> Result<Array3<f64>> {
    let (rows, cols, channels) = frame.dim();
    let inv = transform.inverse()?;
    let mut output = Array3::<f64>::zeros((rows, cols, channels));
    for r in 0..rows {
        for c in 0..cols {
            let (src_x, src_y) = inv.apply_point(c as f64, r as f64);
            for ch in 0..channels {
                // Per-channel bilinear sample.
                let fr = src_y.clamp(0.0, (rows - 1) as f64);
                let fc = src_x.clamp(0.0, (cols - 1) as f64);
                let r0 = fr.floor() as usize;
                let c0 = fc.floor() as usize;
                let r1 = (r0 + 1).min(rows - 1);
                let c1 = (c0 + 1).min(cols - 1);
                let dr = fr - r0 as f64;
                let dc = fc - c0 as f64;
                let top = frame[[r0, c0, ch]] * (1.0 - dc) + frame[[r0, c1, ch]] * dc;
                let bot = frame[[r1, c0, ch]] * (1.0 - dc) + frame[[r1, c1, ch]] * dc;
                output[[r, c, ch]] = top * (1.0 - dr) + bot * dr;
            }
        }
    }
    Ok(output)
}

/// Bilinear sample from a 2D array with border clamping.
fn bilinear_sample(image: &Array2<f64>, fr: f64, fc: f64, rows: usize, cols: usize) -> f64 {
    let fr = fr.clamp(0.0, (rows - 1) as f64);
    let fc = fc.clamp(0.0, (cols - 1) as f64);
    let r0 = fr.floor() as usize;
    let c0 = fc.floor() as usize;
    let r1 = (r0 + 1).min(rows - 1);
    let c1 = (c0 + 1).min(cols - 1);
    let dr = fr - r0 as f64;
    let dc = fc - c0 as f64;
    let top = image[[r0, c0]] * (1.0 - dc) + image[[r0, c1]] * dc;
    let bot = image[[r1, c0]] * (1.0 - dc) + image[[r1, c1]] * dc;
    top * (1.0 - dr) + bot * dr
}

// ---------------------------------------------------------------------------
// Border cropping
// ---------------------------------------------------------------------------

/// Crop the borders of all stabilised frames by `crop_ratio` (e.g. 0.05 = 5%).
///
/// This removes the black border artefacts introduced by the stabilising warps.
pub fn crop_stabilized(
    stabilized_frames: &[Array3<f64>],
    crop_ratio: f64,
) -> Result<Vec<Array3<f64>>> {
    if stabilized_frames.is_empty() {
        return Ok(Vec::new());
    }
    if !(0.0..0.5).contains(&crop_ratio) {
        return Err(VisionError::InvalidParameter(
            "crop_stabilized: crop_ratio must be in [0, 0.5)".into(),
        ));
    }

    let (rows, cols, _channels) = stabilized_frames[0].dim();
    let r_crop = (rows as f64 * crop_ratio).round() as usize;
    let c_crop = (cols as f64 * crop_ratio).round() as usize;

    let r_start = r_crop;
    let r_end = rows.saturating_sub(r_crop);
    let c_start = c_crop;
    let c_end = cols.saturating_sub(c_crop);

    if r_start >= r_end || c_start >= c_end {
        return Err(VisionError::InvalidParameter(
            "crop_stabilized: crop_ratio too large, resulting image has zero area".into(),
        ));
    }

    let mut cropped: Vec<Array3<f64>> = Vec::with_capacity(stabilized_frames.len());
    for frame in stabilized_frames {
        let (fr, fc, fch) = frame.dim();
        if fr != rows || fc != cols {
            return Err(VisionError::DimensionMismatch(
                "crop_stabilized: all frames must have the same shape".into(),
            ));
        }
        let sliced = frame
            .slice(scirs2_core::ndarray::s![r_start..r_end, c_start..c_end, ..])
            .to_owned();
        cropped.push(sliced);
    }

    Ok(cropped)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array2, Array3};

    fn gray_frame(h: usize, w: usize, val: f64) -> Array2<f64> {
        Array2::from_elem((h, w), val)
    }

    fn rgb_frame(h: usize, w: usize, val: f64) -> Array3<f64> {
        Array3::from_elem((h, w, 3), val)
    }

    #[test]
    fn affine2d_identity_apply() {
        let id = Affine2D::identity();
        let (x, y) = id.apply_point(3.0, 7.0);
        assert!((x - 3.0).abs() < 1e-10);
        assert!((y - 7.0).abs() < 1e-10);
    }

    #[test]
    fn affine2d_compose_identity() {
        let t = Affine2D::translation(5.0, -3.0);
        let id = Affine2D::identity();
        let composed = t.compose(&id);
        assert!((composed.m[2] - 5.0).abs() < 1e-10);
        assert!((composed.m[5] - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn affine2d_inverse_roundtrip() {
        let t = Affine2D::translation(4.0, -2.0);
        let inv = t.inverse().expect("inverse failed");
        let composed = t.compose(&inv);
        // Should be close to identity.
        assert!((composed.m[0] - 1.0).abs() < 1e-9);
        assert!((composed.m[4] - 1.0).abs() < 1e-9);
        assert!(composed.m[2].abs() < 1e-9);
        assert!(composed.m[5].abs() < 1e-9);
    }

    #[test]
    fn affine_warp_identity_no_change() {
        let mut img = Array2::<f64>::zeros((8, 8));
        for r in 0..8usize {
            for c in 0..8usize {
                img[[r, c]] = (r * 8 + c) as f64 / 64.0;
            }
        }
        let id = Affine2D::identity();
        let warped = affine_warp(&img, &id).expect("affine_warp failed");
        for (a, b) in img.iter().zip(warped.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn estimate_global_motion_identical_frames() {
        let f = gray_frame(32, 32, 0.5);
        let aff = estimate_global_motion(&f, &f).expect("estimate_global_motion failed");
        // Should be close to identity / zero translation.
        assert!((aff.m[2]).abs() < 2.0); // allow up to 2px tolerance for NCC discretisation
        assert!((aff.m[5]).abs() < 2.0);
    }

    #[test]
    fn smooth_trajectory_single_transform() {
        let transforms = vec![Affine2D::translation(1.0, 2.0)];
        let smoothed = smooth_trajectory(&transforms, 3).expect("smooth_trajectory failed");
        assert_eq!(smoothed.len(), 1);
    }

    #[test]
    fn stabilize_sequence_identity_transforms() {
        let frames: Vec<Array3<f64>> = (0..3).map(|_| rgb_frame(8, 8, 0.5)).collect();
        let transforms: Vec<Affine2D> = (0..2).map(|_| Affine2D::identity()).collect();
        let stabilized =
            stabilize_sequence(&frames, &transforms, 3).expect("stabilize_sequence failed");
        assert_eq!(stabilized.len(), 3);
        for f in &stabilized {
            assert_eq!(f.dim(), (8, 8, 3));
        }
    }

    #[test]
    fn crop_stabilized_basic() {
        let frames: Vec<Array3<f64>> = (0..2).map(|_| rgb_frame(20, 20, 0.5)).collect();
        let cropped = crop_stabilized(&frames, 0.1).expect("crop_stabilized failed");
        assert_eq!(cropped.len(), 2);
        // 10% crop on each side: 20 * 0.1 = 2 → output is 16×16.
        let (h, w, c) = cropped[0].dim();
        assert_eq!(c, 3);
        assert!(h < 20 && w < 20);
    }

    #[test]
    fn crop_stabilized_empty_input() {
        let result = crop_stabilized(&[], 0.05).expect("empty crop failed");
        assert!(result.is_empty());
    }
}
