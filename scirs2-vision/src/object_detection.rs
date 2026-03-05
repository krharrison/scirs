//! # Generic Object Detection Utilities
//!
//! This module provides fundamental building blocks for object detection pipelines,
//! including bounding box representations, Non-Maximum Suppression (NMS) algorithms,
//! sliding window generation, anchor box creation, and IoU computation.
//!
//! ## Features
//!
//! - **`BoundingBox`**: float-coordinate box with score and class_id
//! - **`nms()`**: Standard greedy Non-Maximum Suppression
//! - **`soft_nms()`**: Soft-NMS with Gaussian or linear score decay
//! - **`sliding_window()`**: sliding window iterator with stride and scale
//! - **`anchor_boxes()`**: grid-based anchor box generation for SSD/YOLO style
//! - **`compute_iou()`**: Intersection over Union between two boxes
//!
//! ## Example
//!
//! ```rust
//! use scirs2_vision::object_detection::{BoundingBox, nms, compute_iou};
//!
//! let b1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0, 0.9, 0);
//! let b2 = BoundingBox::new(1.0, 1.0, 11.0, 11.0, 0.7, 0);
//! let b3 = BoundingBox::new(50.0, 50.0, 60.0, 60.0, 0.85, 1);
//!
//! let kept = nms(&[b1, b2, b3], 0.5);
//! assert_eq!(kept.len(), 2);
//! ```

use crate::error::{Result, VisionError};

// ---------------------------------------------------------------------------
// BoundingBox
// ---------------------------------------------------------------------------

/// Floating-point bounding box for detection pipelines.
///
/// Coordinates use (x1, y1) = top-left corner and (x2, y2) = bottom-right
/// corner in pixel space.  `score` is the detection confidence ∈ [0, 1]
/// and `class_id` is a zero-indexed class label.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundingBox {
    /// Top-left x coordinate (inclusive)
    pub x1: f64,
    /// Top-left y coordinate (inclusive)
    pub y1: f64,
    /// Bottom-right x coordinate (exclusive)
    pub x2: f64,
    /// Bottom-right y coordinate (exclusive)
    pub y2: f64,
    /// Detection confidence score ∈ [0, 1]
    pub score: f64,
    /// Class identifier (0-indexed)
    pub class_id: usize,
}

impl BoundingBox {
    /// Create a new `BoundingBox`, normalising corners so x1 ≤ x2 and y1 ≤ y2.
    ///
    /// # Arguments
    /// * `x1`, `y1` – top-left pixel coordinate
    /// * `x2`, `y2` – bottom-right pixel coordinate
    /// * `score`    – confidence score
    /// * `class_id` – class index
    pub fn new(x1: f64, y1: f64, x2: f64, y2: f64, score: f64, class_id: usize) -> Self {
        Self {
            x1: x1.min(x2),
            y1: y1.min(y2),
            x2: x1.max(x2),
            y2: y1.max(y2),
            score,
            class_id,
        }
    }

    /// Create a box from centre coordinates and width/height.
    pub fn from_center(cx: f64, cy: f64, w: f64, h: f64, score: f64, class_id: usize) -> Self {
        let hw = w.abs() * 0.5;
        let hh = h.abs() * 0.5;
        Self::new(cx - hw, cy - hh, cx + hw, cy + hh, score, class_id)
    }

    /// Width in pixels.
    #[inline]
    pub fn width(&self) -> f64 {
        (self.x2 - self.x1).max(0.0)
    }

    /// Height in pixels.
    #[inline]
    pub fn height(&self) -> f64 {
        (self.y2 - self.y1).max(0.0)
    }

    /// Area in pixels².
    #[inline]
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// Centre coordinates (cx, cy).
    #[inline]
    pub fn center(&self) -> (f64, f64) {
        ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)
    }

    /// Expand the box by a factor around its centre.
    ///
    /// A factor of 1.0 returns the original box; 1.2 expands 20% in each direction.
    pub fn scale(&self, factor: f64) -> Self {
        let (cx, cy) = self.center();
        let hw = self.width() * 0.5 * factor;
        let hh = self.height() * 0.5 * factor;
        Self::new(
            cx - hw,
            cy - hh,
            cx + hw,
            cy + hh,
            self.score,
            self.class_id,
        )
    }

    /// Clip the box to an image boundary `(width, height)`.
    pub fn clip(&self, img_w: f64, img_h: f64) -> Self {
        Self::new(
            self.x1.max(0.0),
            self.y1.max(0.0),
            self.x2.min(img_w),
            self.y2.min(img_h),
            self.score,
            self.class_id,
        )
    }
}

// ---------------------------------------------------------------------------
// IoU
// ---------------------------------------------------------------------------

/// Compute Intersection over Union (IoU) between two bounding boxes.
///
/// Returns 0.0 when the union is zero (e.g. both boxes have zero area).
///
/// # Example
/// ```rust
/// use scirs2_vision::object_detection::{BoundingBox, compute_iou};
/// let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0, 1.0, 0);
/// let b = BoundingBox::new(5.0, 5.0, 15.0, 15.0, 1.0, 0);
/// let iou = compute_iou(&a, &b);
/// assert!((iou - 25.0 / 175.0).abs() < 1e-10);
/// ```
pub fn compute_iou(a: &BoundingBox, b: &BoundingBox) -> f64 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);

    let inter_w = (ix2 - ix1).max(0.0);
    let inter_h = (iy2 - iy1).max(0.0);
    let inter = inter_w * inter_h;

    let union = a.area() + b.area() - inter;
    if union < 1e-12 {
        0.0
    } else {
        inter / union
    }
}

// ---------------------------------------------------------------------------
// NMS
// ---------------------------------------------------------------------------

/// Standard greedy Non-Maximum Suppression.
///
/// Boxes are sorted by descending score. A box is suppressed if its IoU with
/// any previously kept box of the same class exceeds `iou_threshold`.
///
/// # Arguments
/// * `boxes`         – candidate boxes (will be cloned and sorted)
/// * `iou_threshold` – suppress if IoU > this value (typical: 0.4 – 0.5)
///
/// # Returns
/// Kept boxes in descending score order.
pub fn nms(boxes: &[BoundingBox], iou_threshold: f64) -> Vec<BoundingBox> {
    if boxes.is_empty() {
        return Vec::new();
    }

    // Sort by descending score
    let mut sorted: Vec<&BoundingBox> = boxes.iter().collect();
    sorted.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n = sorted.len();
    let mut suppressed = vec![false; n];
    let mut kept = Vec::new();

    for i in 0..n {
        if suppressed[i] {
            continue;
        }
        kept.push(sorted[i].clone());
        for j in (i + 1)..n {
            if suppressed[j] {
                continue;
            }
            // Only suppress same-class boxes
            if sorted[i].class_id == sorted[j].class_id
                && compute_iou(sorted[i], sorted[j]) > iou_threshold
            {
                suppressed[j] = true;
            }
        }
    }
    kept
}

// ---------------------------------------------------------------------------
// Soft-NMS
// ---------------------------------------------------------------------------

/// Score decay mode for [`soft_nms`].
#[derive(Clone, Debug, Copy, PartialEq)]
pub enum SoftNmsMethod {
    /// Linear decay: score ← score × (1 − IoU)
    Linear,
    /// Gaussian decay: score ← score × exp(−IoU² / σ²)
    Gaussian {
        /// Gaussian bandwidth (typical: 0.5)
        sigma: f64,
    },
}

/// Soft Non-Maximum Suppression with score decay rather than hard elimination.
///
/// Unlike standard NMS, overlapping detections are not removed but their scores
/// are reduced proportionally to the overlap. Boxes with a final score below
/// `score_threshold` are discarded.
///
/// # Arguments
/// * `boxes`           – candidate boxes (cloned internally)
/// * `iou_threshold`   – IoU at which score decay kicks in
/// * `score_threshold` – minimum final score to keep a box
/// * `method`          – decay function ([`SoftNmsMethod::Linear`] or [`SoftNmsMethod::Gaussian`])
///
/// # Returns
/// Kept boxes sorted by final (decayed) score descending.
pub fn soft_nms(
    boxes: &[BoundingBox],
    iou_threshold: f64,
    score_threshold: f64,
    method: SoftNmsMethod,
) -> Vec<BoundingBox> {
    if boxes.is_empty() {
        return Vec::new();
    }

    let mut candidates: Vec<BoundingBox> = boxes.to_vec();
    let mut kept: Vec<BoundingBox> = Vec::with_capacity(candidates.len());

    while !candidates.is_empty() {
        // Find the candidate with the highest score
        let best_idx = candidates
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let best = candidates.swap_remove(best_idx);

        // Decay scores of remaining boxes
        for candidate in candidates.iter_mut() {
            let iou = compute_iou(&best, candidate);
            if iou > iou_threshold {
                match method {
                    SoftNmsMethod::Linear => {
                        candidate.score *= 1.0 - iou;
                    }
                    SoftNmsMethod::Gaussian { sigma } => {
                        candidate.score *= (-iou * iou / (sigma * sigma)).exp();
                    }
                }
            }
        }

        kept.push(best);
        // Remove boxes with decayed score below threshold
        candidates.retain(|b| b.score >= score_threshold);
    }

    // Sort result by descending score
    kept.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    kept
}

// ---------------------------------------------------------------------------
// Sliding window
// ---------------------------------------------------------------------------

/// A single sliding window entry: pixel position and size.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowSpec {
    /// Top-left x pixel
    pub x: usize,
    /// Top-left y pixel
    pub y: usize,
    /// Window width in pixels
    pub width: usize,
    /// Window height in pixels
    pub height: usize,
    /// Scale factor relative to the base window size
    pub scale: f64,
}

/// Generate sliding window positions over an image at multiple scales.
///
/// Returns a list of [`WindowSpec`] entries covering the image from top-left
/// to bottom-right, stepping by `stride` pixels at each scale level.  Each
/// successive scale multiplies the window by `scale_factor`.
///
/// # Arguments
/// * `img_width`    – image width in pixels
/// * `img_height`   – image height in pixels
/// * `win_width`    – base window width
/// * `win_height`   – base window height
/// * `stride`       – step size (pixels at the *base* scale)
/// * `scale_factor` – multiplicative scale step (e.g. 1.25 → 25% larger each step)
/// * `min_size`     – minimum window size; stop scaling down past this
/// * `num_scales`   – maximum number of scale levels to generate
///
/// # Errors
/// Returns [`VisionError::InvalidInput`] if `win_width` or `win_height` is 0.
pub fn sliding_window(
    img_width: usize,
    img_height: usize,
    win_width: usize,
    win_height: usize,
    stride: usize,
    scale_factor: f64,
    num_scales: usize,
) -> Result<Vec<WindowSpec>> {
    if win_width == 0 || win_height == 0 {
        return Err(VisionError::InvalidInput(
            "sliding_window: window dimensions must be > 0".to_string(),
        ));
    }
    if stride == 0 {
        return Err(VisionError::InvalidInput(
            "sliding_window: stride must be > 0".to_string(),
        ));
    }
    if scale_factor <= 0.0 {
        return Err(VisionError::InvalidInput(
            "sliding_window: scale_factor must be positive".to_string(),
        ));
    }

    let mut windows = Vec::new();

    for scale_idx in 0..num_scales {
        let scale = scale_factor.powi(scale_idx as i32);
        let w = ((win_width as f64) * scale).round() as usize;
        let h = ((win_height as f64) * scale).round() as usize;

        if w == 0 || h == 0 || w > img_width || h > img_height {
            // Skip degenerate or oversized windows
            continue;
        }

        let step = ((stride as f64) * scale).round().max(1.0) as usize;

        let mut y = 0usize;
        while y + h <= img_height {
            let mut x = 0usize;
            while x + w <= img_width {
                windows.push(WindowSpec {
                    x,
                    y,
                    width: w,
                    height: h,
                    scale,
                });
                x += step;
            }
            y += step;
        }
    }

    Ok(windows)
}

// ---------------------------------------------------------------------------
// Anchor boxes
// ---------------------------------------------------------------------------

/// Configuration for grid-based anchor box generation.
#[derive(Clone, Debug)]
pub struct AnchorConfig {
    /// Base anchor sizes (in pixels at scale 1.0)
    pub base_sizes: Vec<f64>,
    /// Aspect ratios width/height (e.g. [0.5, 1.0, 2.0])
    pub aspect_ratios: Vec<f64>,
    /// Additional scale multipliers applied to each base size
    pub scales: Vec<f64>,
    /// Image width
    pub img_width: usize,
    /// Image height
    pub img_height: usize,
    /// Feature map width (grid columns)
    pub feat_width: usize,
    /// Feature map height (grid rows)
    pub feat_height: usize,
}

impl Default for AnchorConfig {
    fn default() -> Self {
        Self {
            base_sizes: vec![32.0, 64.0, 128.0, 256.0, 512.0],
            aspect_ratios: vec![0.5, 1.0, 2.0],
            scales: vec![1.0, 2.0f64.sqrt()],
            img_width: 512,
            img_height: 512,
            feat_width: 16,
            feat_height: 16,
        }
    }
}

/// Generate a grid of anchor boxes from an [`AnchorConfig`].
///
/// For each cell in the `feat_width × feat_height` grid, one anchor is created
/// per `(base_size, aspect_ratio, scale)` combination, centred at the projected
/// pixel coordinates of that cell.
///
/// # Errors
/// Returns [`VisionError::InvalidInput`] if any dimension is zero.
pub fn anchor_boxes(config: &AnchorConfig) -> Result<Vec<BoundingBox>> {
    if config.feat_width == 0 || config.feat_height == 0 {
        return Err(VisionError::InvalidInput(
            "anchor_boxes: feature map dimensions must be > 0".to_string(),
        ));
    }
    if config.img_width == 0 || config.img_height == 0 {
        return Err(VisionError::InvalidInput(
            "anchor_boxes: image dimensions must be > 0".to_string(),
        ));
    }
    if config.base_sizes.is_empty() || config.aspect_ratios.is_empty() || config.scales.is_empty() {
        return Err(VisionError::InvalidInput(
            "anchor_boxes: base_sizes, aspect_ratios, and scales must be non-empty".to_string(),
        ));
    }

    let stride_x = config.img_width as f64 / config.feat_width as f64;
    let stride_y = config.img_height as f64 / config.feat_height as f64;

    let mut anchors = Vec::new();

    for row in 0..config.feat_height {
        let cy = (row as f64 + 0.5) * stride_y;
        for col in 0..config.feat_width {
            let cx = (col as f64 + 0.5) * stride_x;

            for &base_size in &config.base_sizes {
                for &ratio in &config.aspect_ratios {
                    for &scale in &config.scales {
                        // area = (base_size * scale)^2
                        let area = (base_size * scale).powi(2);
                        // ratio = width / height  →  width = sqrt(area * ratio)
                        let w = (area * ratio).sqrt();
                        let h = area / w;

                        anchors.push(BoundingBox::new(
                            cx - w * 0.5,
                            cy - h * 0.5,
                            cx + w * 0.5,
                            cy + h * 0.5,
                            1.0,
                            0,
                        ));
                    }
                }
            }
        }
    }

    Ok(anchors)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_geometry() {
        let b = BoundingBox::new(10.0, 20.0, 50.0, 80.0, 0.9, 1);
        assert!((b.width() - 40.0).abs() < 1e-10);
        assert!((b.height() - 60.0).abs() < 1e-10);
        assert!((b.area() - 2400.0).abs() < 1e-10);
        let (cx, cy) = b.center();
        assert!((cx - 30.0).abs() < 1e-10);
        assert!((cy - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_iou_identical() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0, 1.0, 0);
        assert!((compute_iou(&a, &a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_iou_disjoint() {
        let a = BoundingBox::new(0.0, 0.0, 5.0, 5.0, 1.0, 0);
        let b = BoundingBox::new(10.0, 10.0, 15.0, 15.0, 1.0, 0);
        assert!((compute_iou(&a, &b)).abs() < 1e-10);
    }

    #[test]
    fn test_compute_iou_partial() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0, 1.0, 0);
        let b = BoundingBox::new(5.0, 5.0, 15.0, 15.0, 1.0, 0);
        let iou = compute_iou(&a, &b);
        // intersection = 5×5 = 25, union = 100+100-25 = 175
        assert!((iou - 25.0 / 175.0).abs() < 1e-10);
    }

    #[test]
    fn test_nms_removes_overlapping() {
        let boxes = vec![
            BoundingBox::new(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            BoundingBox::new(1.0, 1.0, 11.0, 11.0, 0.7, 0), // heavily overlaps first
            BoundingBox::new(50.0, 50.0, 60.0, 60.0, 0.8, 1), // different class, disjoint
        ];
        let kept = nms(&boxes, 0.5);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].score - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_nms_different_classes_kept() {
        // Same position, different class — NMS should keep both
        let boxes = vec![
            BoundingBox::new(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            BoundingBox::new(0.0, 0.0, 10.0, 10.0, 0.8, 1),
        ];
        let kept = nms(&boxes, 0.5);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_soft_nms_linear() {
        let boxes = vec![
            BoundingBox::new(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            BoundingBox::new(1.0, 1.0, 11.0, 11.0, 0.8, 0),
            BoundingBox::new(50.0, 50.0, 60.0, 60.0, 0.7, 0),
        ];
        let kept = soft_nms(&boxes, 0.3, 0.3, SoftNmsMethod::Linear);
        // Disjoint box should survive; heavily overlapping box may be suppressed
        assert!(!kept.is_empty());
        // All remaining scores should be >= threshold
        for b in &kept {
            assert!(b.score >= 0.3);
        }
    }

    #[test]
    fn test_soft_nms_gaussian() {
        let boxes = vec![
            BoundingBox::new(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            BoundingBox::new(0.5, 0.5, 10.5, 10.5, 0.8, 0),
        ];
        let kept = soft_nms(&boxes, 0.3, 0.1, SoftNmsMethod::Gaussian { sigma: 0.5 });
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_sliding_window_basic() {
        let windows =
            sliding_window(100, 100, 20, 20, 10, 1.0, 1).expect("sliding_window should succeed");
        // Without scaling: (100-20)/10 + 1 = 9 positions per axis → 9×9 = 81
        assert_eq!(windows.len(), 81);
        for w in &windows {
            assert!(w.x + w.width <= 100);
            assert!(w.y + w.height <= 100);
        }
    }

    #[test]
    fn test_sliding_window_error_zero_dims() {
        assert!(sliding_window(100, 100, 0, 20, 10, 1.0, 1).is_err());
    }

    #[test]
    fn test_anchor_boxes_count() {
        let config = AnchorConfig {
            base_sizes: vec![32.0],
            aspect_ratios: vec![1.0],
            scales: vec![1.0],
            img_width: 256,
            img_height: 256,
            feat_width: 4,
            feat_height: 4,
        };
        let anchors = anchor_boxes(&config).expect("anchor_boxes should succeed");
        // 4×4 grid × 1 base × 1 ratio × 1 scale = 16
        assert_eq!(anchors.len(), 16);
    }

    #[test]
    fn test_anchor_boxes_multi() {
        let config = AnchorConfig {
            base_sizes: vec![32.0, 64.0],
            aspect_ratios: vec![0.5, 1.0, 2.0],
            scales: vec![1.0, 2.0f64.sqrt()],
            img_width: 512,
            img_height: 512,
            feat_width: 8,
            feat_height: 8,
        };
        let anchors = anchor_boxes(&config).expect("anchor_boxes should succeed");
        // 8×8 × 2 × 3 × 2 = 768
        assert_eq!(anchors.len(), 768);
    }

    #[test]
    fn test_from_center() {
        let b = BoundingBox::from_center(10.0, 10.0, 4.0, 6.0, 0.9, 0);
        assert!((b.x1 - 8.0).abs() < 1e-10);
        assert!((b.y1 - 7.0).abs() < 1e-10);
        assert!((b.x2 - 12.0).abs() < 1e-10);
        assert!((b.y2 - 13.0).abs() < 1e-10);
    }
}
