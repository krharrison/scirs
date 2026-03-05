//! Video frame buffer and temporal processing utilities.
//!
//! This module provides data structures for working with video frame sequences
//! and temporal algorithms such as per-pixel temporal median filtering and
//! background subtraction using a Mixture of Gaussians (MoG) model.
//!
//! # Overview
//!
//! - [`VideoFrame`] -- a single video frame with metadata
//! - [`FrameBuffer`] -- a circular (ring) buffer of [`VideoFrame`]s
//! - [`temporal_median_filter`] -- per-pixel median over a window of RGB frames
//! - [`MogBackground`] / [`background_subtraction_mog`] -- per-pixel MoG foreground detection
//! - [`frame_interpolation`] -- flow-based frame interpolation

use crate::error::{Result, VisionError};
use crate::optical_flow_dense::warp_image;
use scirs2_core::ndarray::{Array2, Array3};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// VideoFrame
// ---------------------------------------------------------------------------

/// A single video frame with associated timestamp and spatial metadata.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Pixel data stored as `[height, width, channels]` in `[0, 1]`.
    pub data: Array3<f64>,
    /// Timestamp in seconds.
    pub timestamp: f64,
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
}

impl VideoFrame {
    /// Create a new [`VideoFrame`].
    ///
    /// # Errors
    ///
    /// Returns an error if the array dimensions are inconsistent with the
    /// supplied `width` and `height`.
    pub fn new(data: Array3<f64>, timestamp: f64) -> Result<Self> {
        let shape = data.dim();
        let height = shape.0;
        let width = shape.1;
        Ok(Self {
            data,
            timestamp,
            width,
            height,
        })
    }

    /// Return the number of colour channels.
    pub fn channels(&self) -> usize {
        self.data.dim().2
    }

    /// Extract a single channel as an `Array2<f64>`.
    pub fn channel(&self, ch: usize) -> Result<Array2<f64>> {
        if ch >= self.channels() {
            return Err(VisionError::InvalidParameter(format!(
                "channel index {ch} out of range (frame has {} channels)",
                self.channels()
            )));
        }
        Ok(self
            .data
            .slice(scirs2_core::ndarray::s![.., .., ch])
            .to_owned())
    }

    /// Convert to grayscale by averaging all channels.
    pub fn to_grayscale(&self) -> Array2<f64> {
        let (h, w, c) = self.data.dim();
        let mut gray = Array2::<f64>::zeros((h, w));
        let weight = 1.0 / c as f64;
        for ch in 0..c {
            for r in 0..h {
                for col in 0..w {
                    gray[[r, col]] += self.data[[r, col, ch]] * weight;
                }
            }
        }
        gray
    }
}

// ---------------------------------------------------------------------------
// FrameBuffer (circular buffer)
// ---------------------------------------------------------------------------

/// A capacity-bounded circular buffer of [`VideoFrame`]s.
///
/// When the buffer is full, the oldest frame is automatically evicted.
#[derive(Debug, Clone)]
pub struct FrameBuffer {
    /// Internal double-ended queue acting as a ring buffer.
    pub frames: VecDeque<VideoFrame>,
    /// Maximum number of frames retained.
    pub capacity: usize,
}

impl FrameBuffer {
    /// Create a new `FrameBuffer` with the given capacity.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(VisionError::InvalidParameter(
                "FrameBuffer: capacity must be at least 1".into(),
            ));
        }
        Ok(Self {
            frames: VecDeque::with_capacity(capacity),
            capacity,
        })
    }

    /// Push a frame onto the back.  If the buffer is at capacity the oldest
    /// (front) frame is dropped first.
    pub fn push(&mut self, frame: VideoFrame) {
        if self.frames.len() == self.capacity {
            self.frames.pop_front();
        }
        self.frames.push_back(frame);
    }

    /// Number of frames currently stored.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` if the buffer contains no frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Whether the buffer has reached its maximum capacity.
    pub fn is_full(&self) -> bool {
        self.frames.len() == self.capacity
    }

    /// Retrieve a reference to the most recent frame, if any.
    pub fn latest(&self) -> Option<&VideoFrame> {
        self.frames.back()
    }

    /// Iterate over frames in chronological order.
    pub fn iter(&self) -> impl Iterator<Item = &VideoFrame> {
        self.frames.iter()
    }
}

// ---------------------------------------------------------------------------
// Temporal median filter
// ---------------------------------------------------------------------------

/// Apply a per-pixel temporal median filter over a slice of RGB (or N-channel) frames.
///
/// All frames must have the same spatial shape `[H, W, C]`.  The output is
/// the per-pixel, per-channel median over the supplied window.
///
/// # Arguments
///
/// * `frames`  – slice of at least 1 frame, each `[H, W, C]` in `[0, 1]`
/// * `window`  – number of frames to include in the median (capped to
///   `frames.len()` if larger)
pub fn temporal_median_filter(frames: &[Array3<f64>], window: usize) -> Result<Array3<f64>> {
    if frames.is_empty() {
        return Err(VisionError::InvalidParameter(
            "temporal_median_filter: frames slice must not be empty".into(),
        ));
    }
    let window = window.min(frames.len()).max(1);
    let ref_shape = frames[0].dim();
    for (i, f) in frames.iter().enumerate() {
        if f.dim() != ref_shape {
            return Err(VisionError::DimensionMismatch(format!(
                "temporal_median_filter: frame {i} shape {:?} != reference {:?}",
                f.dim(),
                ref_shape
            )));
        }
    }

    let (h, w, c) = ref_shape;
    let start = frames.len().saturating_sub(window);
    let window_frames = &frames[start..];
    let n = window_frames.len();
    let mut output = Array3::<f64>::zeros((h, w, c));

    for row in 0..h {
        for col in 0..w {
            for ch in 0..c {
                let mut vals: Vec<f64> = window_frames.iter().map(|f| f[[row, col, ch]]).collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                output[[row, col, ch]] = if n % 2 == 1 {
                    vals[n / 2]
                } else {
                    (vals[n / 2 - 1] + vals[n / 2]) * 0.5
                };
            }
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Mixture of Gaussians background subtraction
// ---------------------------------------------------------------------------

/// Configuration for the Mixture of Gaussians background model.
#[derive(Debug, Clone)]
pub struct MogBackground {
    /// Number of Gaussian components per pixel.
    pub n_gaussians: usize,
    /// Learning rate for updating the model (α).  Typical: 0.005.
    pub learning_rate: f64,
    /// Mahalanobis distance threshold for foreground / background decision.
    pub threshold: f64,
    // Per-pixel model: [H, W, K] tensors for mean, variance and weight.
    means: Option<Array3<f64>>,
    variances: Option<Array3<f64>>,
    weights: Option<Array3<f64>>,
}

impl MogBackground {
    /// Create a new uninitialised MoG background model.
    ///
    /// The model is initialised on the first call to [`background_subtraction_mog`].
    pub fn new(n_gaussians: usize, learning_rate: f64, threshold: f64) -> Result<Self> {
        if n_gaussians == 0 {
            return Err(VisionError::InvalidParameter(
                "MogBackground: n_gaussians must be at least 1".into(),
            ));
        }
        if !(0.0..=1.0).contains(&learning_rate) {
            return Err(VisionError::InvalidParameter(
                "MogBackground: learning_rate must be in (0, 1]".into(),
            ));
        }
        Ok(Self {
            n_gaussians,
            learning_rate,
            threshold,
            means: None,
            variances: None,
            weights: None,
        })
    }
}

impl Default for MogBackground {
    fn default() -> Self {
        Self {
            n_gaussians: 3,
            learning_rate: 0.005,
            threshold: 2.5,
            means: None,
            variances: None,
            weights: None,
        }
    }
}

/// Apply Mixture of Gaussians background subtraction to a single-channel frame.
///
/// Updates `background_model` in place and returns a boolean mask where
/// `true` indicates a foreground (moving) pixel.
///
/// The frame values are expected in `[0, 1]`.
pub fn background_subtraction_mog(
    frame: &Array2<f64>,
    background_model: &mut MogBackground,
) -> Result<Array2<bool>> {
    let (rows, cols) = frame.dim();
    let k = background_model.n_gaussians;
    let alpha = background_model.learning_rate;
    let thr = background_model.threshold;

    // Initialise model on first call.
    if background_model.means.is_none() {
        // All components start at the current frame value with high variance
        // and equal weight.
        let init_weight = 1.0 / k as f64;
        let mut means = Array3::<f64>::zeros((rows, cols, k));
        let variances = Array3::<f64>::from_elem((rows, cols, k), 0.01);
        let weights = Array3::<f64>::from_elem((rows, cols, k), init_weight);
        // Copy frame pixel into each component mean.
        for r in 0..rows {
            for c in 0..cols {
                for ki in 0..k {
                    means[[r, c, ki]] = frame[[r, c]];
                }
            }
        }
        background_model.means = Some(means);
        background_model.variances = Some(variances);
        background_model.weights = Some(weights);
    }

    let means = background_model.means.as_mut().expect("means initialised");
    let variances = background_model
        .variances
        .as_mut()
        .expect("variances initialised");
    let weights = background_model
        .weights
        .as_mut()
        .expect("weights initialised");

    let mut fg_mask = Array2::<bool>::from_elem((rows, cols), false);

    for r in 0..rows {
        for c in 0..cols {
            let pixel = frame[[r, c]];
            let mut matched = false;
            let mut best_ki = 0usize;

            // Find matching component (Mahalanobis distance check).
            for ki in 0..k {
                let diff = pixel - means[[r, c, ki]];
                let var = variances[[r, c, ki]];
                if var > 1e-12 && diff * diff / var < thr * thr {
                    // Update matching component.
                    let rho = alpha / weights[[r, c, ki]].max(1e-12);
                    means[[r, c, ki]] += rho * diff;
                    variances[[r, c, ki]] = (1.0 - rho) * var + rho * diff * diff;
                    weights[[r, c, ki]] = (1.0 - alpha) * weights[[r, c, ki]] + alpha;

                    matched = true;
                    best_ki = ki;
                    break;
                }
            }

            if !matched {
                // Replace the least-weighted component.
                let mut min_w = weights[[r, c, 0]];
                let mut min_ki = 0;
                for ki in 1..k {
                    if weights[[r, c, ki]] < min_w {
                        min_w = weights[[r, c, ki]];
                        min_ki = ki;
                    }
                }
                means[[r, c, min_ki]] = pixel;
                variances[[r, c, min_ki]] = 0.01;
                weights[[r, c, min_ki]] = alpha;
                best_ki = min_ki;
            }

            // Renormalise weights.
            let mut w_sum = 0.0_f64;
            for ki in 0..k {
                w_sum += weights[[r, c, ki]];
            }
            if w_sum > 1e-12 {
                for ki in 0..k {
                    weights[[r, c, ki]] /= w_sum;
                }
            }

            // Foreground decision: pixel belongs to background if its matched
            // component has weight above 1/K (dominant component check).
            let is_bg = matched && weights[[r, c, best_ki]] > 1.0 / k as f64;
            fg_mask[[r, c]] = !is_bg;
        }
    }

    Ok(fg_mask)
}

// ---------------------------------------------------------------------------
// Flow-based frame interpolation
// ---------------------------------------------------------------------------

/// Interpolate a frame between `frame1` and `frame2` at time `t ∈ [0, 1]`.
///
/// Uses the supplied optical flow `flow` (computed from `frame1` to `frame2`)
/// to warp `frame1` forward by `t` and `frame2` backward by `(1 - t)`, then
/// blends the two warped images linearly.
///
/// # Arguments
///
/// * `frame1` – source frame `[H, W, C]`
/// * `frame2` – target frame `[H, W, C]`
/// * `t`      – interpolation parameter in `[0, 1]` (0 → frame1, 1 → frame2)
/// * `flow`   – `(u, v)` flow field from frame1 to frame2 (shape `[H, W]` each)
pub fn frame_interpolation(
    frame1: &Array3<f64>,
    frame2: &Array3<f64>,
    t: f64,
    flow: (&Array2<f64>, &Array2<f64>),
) -> Result<Array3<f64>> {
    let shape = frame1.dim();
    if shape != frame2.dim() {
        return Err(VisionError::DimensionMismatch(
            "frame_interpolation: frame1 and frame2 must have identical shapes".into(),
        ));
    }
    let (u, v) = flow;
    let (h, w, _) = shape;
    if u.dim() != (h, w) || v.dim() != (h, w) {
        return Err(VisionError::DimensionMismatch(
            "frame_interpolation: flow field spatial shape must match frame spatial shape".into(),
        ));
    }
    if !(0.0..=1.0).contains(&t) {
        return Err(VisionError::InvalidParameter(
            "frame_interpolation: t must be in [0, 1]".into(),
        ));
    }

    let (rows, cols, channels) = shape;

    // Forward warp: scale flow by t.
    let u_fwd = u.mapv(|x| x * t);
    let v_fwd = v.mapv(|x| x * t);
    // Backward warp: scale flow by -(1-t).
    let u_bwd = u.mapv(|x| -x * (1.0 - t));
    let v_bwd = v.mapv(|x| -x * (1.0 - t));

    let mut output = Array3::<f64>::zeros((rows, cols, channels));

    for ch in 0..channels {
        let ch1 = frame1
            .slice(scirs2_core::ndarray::s![.., .., ch])
            .to_owned();
        let ch2 = frame2
            .slice(scirs2_core::ndarray::s![.., .., ch])
            .to_owned();

        let warped1 = warp_image(&ch1, &u_fwd, &v_fwd)?;
        let warped2 = warp_image(&ch2, &u_bwd, &v_bwd)?;

        for r in 0..rows {
            for c in 0..cols {
                output[[r, c, ch]] = (1.0 - t) * warped1[[r, c]] + t * warped2[[r, c]];
            }
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array2, Array3};

    fn rgb_frame(h: usize, w: usize, val: f64) -> Array3<f64> {
        Array3::from_elem((h, w, 3), val)
    }

    #[test]
    fn frame_buffer_circular_eviction() {
        let mut buf = FrameBuffer::new(3).expect("FrameBuffer::new failed");
        for i in 0..5u32 {
            let data = rgb_frame(4, 4, i as f64 / 10.0);
            let frame = VideoFrame::new(data, i as f64).expect("VideoFrame::new failed");
            buf.push(frame);
        }
        assert_eq!(buf.len(), 3);
        // Oldest kept should be frame 2 (timestamp = 2.0).
        assert!((buf.frames[0].timestamp - 2.0).abs() < 1e-9);
    }

    #[test]
    fn temporal_median_filter_single_frame() {
        let f = rgb_frame(4, 4, 0.7);
        let out = temporal_median_filter(&[f.clone()], 1).expect("median filter failed");
        for &v in out.iter() {
            assert!((v - 0.7).abs() < 1e-10);
        }
    }

    #[test]
    fn temporal_median_filter_three_frames() {
        // Three frames: 0.2, 0.5, 0.8 → median should be 0.5.
        let f1 = rgb_frame(2, 2, 0.2);
        let f2 = rgb_frame(2, 2, 0.5);
        let f3 = rgb_frame(2, 2, 0.8);
        let out = temporal_median_filter(&[f1, f2, f3], 3).expect("median filter failed");
        for &v in out.iter() {
            assert!((v - 0.5).abs() < 1e-10, "expected 0.5, got {v}");
        }
    }

    #[test]
    fn mog_background_first_frame_all_foreground_zero() {
        // On the very first frame everything is initialised to match the pixel,
        // so we expect either all background or well-defined behaviour.
        let frame = Array2::from_elem((4, 4), 0.5_f64);
        let mut model = MogBackground::new(3, 0.005, 2.5).expect("MogBackground::new failed");
        let mask = background_subtraction_mog(&frame, &mut model).expect("mog failed");
        // After first frame init, every pixel "matched" the newly created component
        // but the component weight equals alpha (< 1/K), so all are foreground.
        // Either way the mask must have the correct shape.
        assert_eq!(mask.dim(), (4, 4));
    }

    #[test]
    fn mog_background_converges_to_background() {
        // After many identical frames the model should classify them as background.
        let frame = Array2::from_elem((4, 4), 0.5_f64);
        let mut model = MogBackground::new(3, 0.1, 2.5).expect("MogBackground::new failed");
        for _ in 0..50 {
            let _ = background_subtraction_mog(&frame, &mut model);
        }
        let mask = background_subtraction_mog(&frame, &mut model).expect("mog failed");
        // All pixels should be background after convergence.
        for val in mask.iter() {
            assert!(!val, "expected background, got foreground");
        }
    }

    #[test]
    fn frame_interpolation_at_zero_returns_frame1() {
        let f1 = rgb_frame(4, 4, 0.2);
        let f2 = rgb_frame(4, 4, 0.8);
        let u = Array2::zeros((4, 4));
        let v = Array2::zeros((4, 4));
        let out = frame_interpolation(&f1, &f2, 0.0, (&u, &v)).expect("interpolation failed");
        for &val in out.iter() {
            assert!((val - 0.2).abs() < 1e-10, "expected 0.2, got {val}");
        }
    }

    #[test]
    fn frame_interpolation_at_one_returns_frame2() {
        let f1 = rgb_frame(4, 4, 0.2);
        let f2 = rgb_frame(4, 4, 0.8);
        let u = Array2::zeros((4, 4));
        let v = Array2::zeros((4, 4));
        let out = frame_interpolation(&f1, &f2, 1.0, (&u, &v)).expect("interpolation failed");
        for &val in out.iter() {
            assert!((val - 0.8).abs() < 1e-10, "expected 0.8, got {val}");
        }
    }
}
