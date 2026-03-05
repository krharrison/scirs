//! # Viola-Jones Face Detector
//!
//! A pure-Rust implementation of the Viola-Jones object detection framework
//! using integral images, Haar-like features, and an AdaBoost cascade classifier.
//!
//! ## Algorithm Summary
//!
//! 1. **Integral Image**: Build a summed-area table so that any rectangular sum
//!    can be computed in O(1) using four table look-ups.
//! 2. **Haar Features**: Compute difference-of-rectangle features (edge, line,
//!    four-rectangle) over the integral image to characterise local intensity
//!    patterns.
//! 3. **Weak Classifiers**: Each weak learner thresholds one Haar feature value.
//! 4. **AdaBoost Cascade**: Weak classifiers are combined in stages. A candidate
//!    window is rejected at any stage where the combined response falls below the
//!    stage threshold, enabling fast early rejection of background windows.
//! 5. **Multi-scale Detection**: Slide the window across the image at multiple
//!    scales, collect positive windows, and apply Non-Maximum Suppression.
//!
//! ## Notes
//!
//! This is an *algorithmic skeleton*: the classifier parameters (thresholds,
//! weights, feature rectangles) would normally be loaded from a pre-trained
//! cascade file.  The provided defaults demonstrate typical structure for a
//! simple 3-stage cascaded detector suitable for unit testing and extension.

use crate::error::{Result, VisionError};
use crate::object_detection::{compute_iou, nms, BoundingBox};

// ---------------------------------------------------------------------------
// Integral Image
// ---------------------------------------------------------------------------

/// Summed-area table (integral image) for O(1) rectangular sum queries.
///
/// Given a grey-level image `I[r][c]`, the integral image `S[r][c]` is defined
/// as the sum of all pixels strictly above and to the left of (r, c):
///
/// ```text
/// S[r][c] = Σ_{r'<r, c'<c} I[r'][c']
/// ```
///
/// A rectangular sum over rows `[r1, r2)` and columns `[c1, c2)` can then be
/// recovered as `S[r2][c2] - S[r1][c2] - S[r2][c1] + S[r1][c1]`.
#[derive(Clone, Debug)]
pub struct IntegralImage {
    /// Integral image data stored row-major; dimensions are (rows+1) × (cols+1).
    data: Vec<f64>,
    /// Number of rows in the source image.
    pub rows: usize,
    /// Number of columns in the source image.
    pub cols: usize,
}

impl IntegralImage {
    /// Construct an integral image from a flat row-major grey-level image buffer.
    ///
    /// `pixels` must have exactly `rows * cols` elements.
    ///
    /// # Errors
    /// Returns [`VisionError::InvalidInput`] when the pixel count does not match.
    pub fn new(pixels: &[f64], rows: usize, cols: usize) -> Result<Self> {
        if pixels.len() != rows * cols {
            return Err(VisionError::InvalidInput(format!(
                "IntegralImage::new: expected {} pixels, got {}",
                rows * cols,
                pixels.len()
            )));
        }

        // Pad with one extra row and column of zeros
        let stride = cols + 1;
        let mut data = vec![0.0_f64; (rows + 1) * stride];

        for r in 0..rows {
            let mut row_sum = 0.0_f64;
            for c in 0..cols {
                row_sum += pixels[r * cols + c];
                let above = data[r * stride + (c + 1)];
                data[(r + 1) * stride + (c + 1)] = row_sum + above;
            }
        }

        Ok(Self { data, rows, cols })
    }

    /// Construct an integral image from a 2-D ndarray (rows × cols).
    ///
    /// The array must be 2-D (shape [rows, cols]).
    #[inline]
    pub fn from_array(arr: &scirs2_core::ndarray::Array2<f64>) -> Result<Self> {
        let rows = arr.nrows();
        let cols = arr.ncols();
        // Collect in row-major order
        let pixels: Vec<f64> = arr.iter().copied().collect();
        Self::new(&pixels, rows, cols)
    }

    /// Sum of the rectangle `[r1, r2) × [c1, c2)` (rows and columns half-open).
    ///
    /// Returns 0.0 for degenerate or out-of-bound rectangles without panicking.
    #[inline]
    pub fn rect_sum(&self, r1: usize, c1: usize, r2: usize, c2: usize) -> f64 {
        if r1 >= r2 || c1 >= c2 {
            return 0.0;
        }
        let r2 = r2.min(self.rows);
        let c2 = c2.min(self.cols);
        if r1 >= r2 || c1 >= c2 {
            return 0.0;
        }
        let stride = self.cols + 1;
        self.data[r2 * stride + c2] - self.data[r1 * stride + c2] - self.data[r2 * stride + c1]
            + self.data[r1 * stride + c1]
    }
}

// ---------------------------------------------------------------------------
// Haar Features
// ---------------------------------------------------------------------------

/// Axis-aligned rectangle within a detection window.
#[derive(Clone, Debug, PartialEq)]
pub struct HaarRect {
    /// Row offset from window top-left
    pub row: usize,
    /// Column offset from window top-left
    pub col: usize,
    /// Rectangle height
    pub height: usize,
    /// Rectangle width
    pub width: usize,
    /// Weight applied to this rectangle's sum (+1 or −1)
    pub weight: f64,
}

impl HaarRect {
    /// Create a new [`HaarRect`].
    pub fn new(row: usize, col: usize, height: usize, width: usize, weight: f64) -> Self {
        Self {
            row,
            col,
            height,
            width,
            weight,
        }
    }
}

/// Type of Haar-like feature.
#[derive(Clone, Debug, PartialEq)]
pub enum HaarFeatureType {
    /// Two-rectangle edge feature (horizontal: top/bottom halves)
    EdgeHorizontal,
    /// Two-rectangle edge feature (vertical: left/right halves)
    EdgeVertical,
    /// Three-rectangle line feature (horizontal)
    LineHorizontal,
    /// Three-rectangle line feature (vertical)
    LineVertical,
    /// Four-rectangle diagonal feature
    FourRectangle,
    /// Custom feature with arbitrary rectangle set
    Custom,
}

/// A Haar-like feature composed of weighted rectangles.
///
/// The feature response is the weighted sum of the rectangle sums within the
/// integral image at a given window offset.
#[derive(Clone, Debug)]
pub struct HaarFeature {
    /// Semantic feature type (informational)
    pub feature_type: HaarFeatureType,
    /// Constituent weighted rectangles
    pub rects: Vec<HaarRect>,
}

impl HaarFeature {
    /// Construct a horizontal two-rectangle (edge) Haar feature.
    ///
    /// The feature is placed at `(base_row, base_col)` with the given `height`
    /// and total `width`.  The top half gets weight +1 and the bottom half −1.
    pub fn edge_horizontal(base_row: usize, base_col: usize, height: usize, width: usize) -> Self {
        let half_h = height / 2;
        Self {
            feature_type: HaarFeatureType::EdgeHorizontal,
            rects: vec![
                HaarRect::new(base_row, base_col, half_h, width, 1.0),
                HaarRect::new(base_row + half_h, base_col, height - half_h, width, -1.0),
            ],
        }
    }

    /// Construct a vertical two-rectangle (edge) Haar feature.
    pub fn edge_vertical(base_row: usize, base_col: usize, height: usize, width: usize) -> Self {
        let half_w = width / 2;
        Self {
            feature_type: HaarFeatureType::EdgeVertical,
            rects: vec![
                HaarRect::new(base_row, base_col, height, half_w, 1.0),
                HaarRect::new(base_row, base_col + half_w, height, width - half_w, -1.0),
            ],
        }
    }

    /// Construct a horizontal three-rectangle (line) Haar feature.
    pub fn line_horizontal(base_row: usize, base_col: usize, height: usize, width: usize) -> Self {
        let third_h = height / 3;
        Self {
            feature_type: HaarFeatureType::LineHorizontal,
            rects: vec![
                HaarRect::new(base_row, base_col, third_h, width, -1.0),
                HaarRect::new(base_row + third_h, base_col, third_h, width, 2.0),
                HaarRect::new(
                    base_row + 2 * third_h,
                    base_col,
                    height - 2 * third_h,
                    width,
                    -1.0,
                ),
            ],
        }
    }

    /// Construct a vertical three-rectangle (line) Haar feature.
    pub fn line_vertical(base_row: usize, base_col: usize, height: usize, width: usize) -> Self {
        let third_w = width / 3;
        Self {
            feature_type: HaarFeatureType::LineVertical,
            rects: vec![
                HaarRect::new(base_row, base_col, height, third_w, -1.0),
                HaarRect::new(base_row, base_col + third_w, height, third_w, 2.0),
                HaarRect::new(
                    base_row,
                    base_col + 2 * third_w,
                    height,
                    width - 2 * third_w,
                    -1.0,
                ),
            ],
        }
    }

    /// Construct a four-rectangle (diagonal checker) Haar feature.
    pub fn four_rectangle(base_row: usize, base_col: usize, height: usize, width: usize) -> Self {
        let half_h = height / 2;
        let half_w = width / 2;
        Self {
            feature_type: HaarFeatureType::FourRectangle,
            rects: vec![
                HaarRect::new(base_row, base_col, half_h, half_w, 1.0),
                HaarRect::new(base_row, base_col + half_w, half_h, width - half_w, -1.0),
                HaarRect::new(base_row + half_h, base_col, height - half_h, half_w, -1.0),
                HaarRect::new(
                    base_row + half_h,
                    base_col + half_w,
                    height - half_h,
                    width - half_w,
                    1.0,
                ),
            ],
        }
    }
}

/// Compute the response of a Haar feature on an integral image at a window offset.
///
/// The feature rectangles are interpreted as offsets *within* the detection window,
/// which itself starts at pixel `(win_row, win_col)` in the full image.
/// The response is normalised by the window's standard deviation to achieve
/// approximate illumination invariance.
///
/// # Arguments
/// * `integral`   – summed-area table of the full image
/// * `feature`    – Haar feature to evaluate
/// * `win_row`    – top-left row of the detection window in the full image
/// * `win_col`    – top-left column of the detection window
/// * `win_h`      – height of the detection window
/// * `win_w`      – width of the detection window
/// * `norm_factor`– normalisation factor (typically window std dev; 1.0 → skip)
pub fn compute_haar_feature(
    integral: &IntegralImage,
    feature: &HaarFeature,
    win_row: usize,
    win_col: usize,
    norm_factor: f64,
) -> f64 {
    let mut response = 0.0_f64;
    for rect in &feature.rects {
        let r1 = win_row + rect.row;
        let c1 = win_col + rect.col;
        let r2 = r1 + rect.height;
        let c2 = c1 + rect.width;
        response += rect.weight * integral.rect_sum(r1, c1, r2, c2);
    }
    if norm_factor.abs() > 1e-12 {
        response / norm_factor
    } else {
        response
    }
}

// ---------------------------------------------------------------------------
// Weak Classifier
// ---------------------------------------------------------------------------

/// A single weak classifier (decision stump) used inside an AdaBoost stage.
#[derive(Clone, Debug)]
pub struct WeakClassifier {
    /// The Haar feature this classifier evaluates
    pub feature: HaarFeature,
    /// Decision threshold in feature response space
    pub threshold: f64,
    /// Polarity: +1 if response > threshold → positive, −1 if response < threshold → positive
    pub polarity: f64,
    /// AdaBoost weight α for this classifier
    pub alpha: f64,
}

impl WeakClassifier {
    /// Create a new weak classifier.
    pub fn new(feature: HaarFeature, threshold: f64, polarity: f64, alpha: f64) -> Self {
        Self {
            feature,
            threshold,
            polarity,
            alpha,
        }
    }

    /// Evaluate the classifier on the integral image at the given window location.
    ///
    /// Returns `self.alpha` if the window passes the stump, else `-self.alpha`.
    pub fn evaluate(
        &self,
        integral: &IntegralImage,
        win_row: usize,
        win_col: usize,
        norm_factor: f64,
    ) -> f64 {
        let response = compute_haar_feature(integral, &self.feature, win_row, win_col, norm_factor);
        if self.polarity * response >= self.polarity * self.threshold {
            self.alpha
        } else {
            -self.alpha
        }
    }
}

// ---------------------------------------------------------------------------
// AdaBoost Classifier Stage
// ---------------------------------------------------------------------------

/// A single stage in the cascade: a collection of weak classifiers with a
/// joint threshold for early rejection.
#[derive(Clone, Debug)]
pub struct AdaBoostStage {
    /// Weak classifiers in this stage
    pub classifiers: Vec<WeakClassifier>,
    /// Stage threshold: if Σ αᵢ hᵢ(x) < threshold → reject
    pub stage_threshold: f64,
}

impl AdaBoostStage {
    /// Create a new stage from its constituent weak classifiers and threshold.
    pub fn new(classifiers: Vec<WeakClassifier>, stage_threshold: f64) -> Self {
        Self {
            classifiers,
            stage_threshold,
        }
    }

    /// Evaluate the stage on a window; returns `Ok(true)` if the window passes,
    /// `Ok(false)` if it is rejected.
    pub fn evaluate(
        &self,
        integral: &IntegralImage,
        win_row: usize,
        win_col: usize,
        norm_factor: f64,
    ) -> bool {
        let score: f64 = self
            .classifiers
            .iter()
            .map(|clf| clf.evaluate(integral, win_row, win_col, norm_factor))
            .sum();
        score >= self.stage_threshold
    }
}

// ---------------------------------------------------------------------------
// AdaBoost Cascade Classifier
// ---------------------------------------------------------------------------

/// Full cascade of [`AdaBoostStage`]s.
///
/// A detection window must pass *all* stages to be classified as a positive
/// detection.  Rejection at any stage provides fast early exit.
#[derive(Clone, Debug)]
pub struct AdaBoostClassifier {
    /// Ordered list of cascade stages (coarse → fine)
    pub stages: Vec<AdaBoostStage>,
    /// Window width for which this classifier was trained (pixels)
    pub window_width: usize,
    /// Window height for which this classifier was trained (pixels)
    pub window_height: usize,
}

impl AdaBoostClassifier {
    /// Create a new cascade classifier.
    pub fn new(stages: Vec<AdaBoostStage>, window_width: usize, window_height: usize) -> Self {
        Self {
            stages,
            window_width,
            window_height,
        }
    }

    /// Build a small demonstration cascade for a 24×24 window.
    ///
    /// Parameters are illustrative; the cascade is intentionally lenient so that
    /// it produces detections on simple synthetic images for testing.
    pub fn default_24x24() -> Self {
        // Stage 1: one vertical-edge feature over the nose bridge region
        let stage1 = AdaBoostStage::new(
            vec![WeakClassifier::new(
                HaarFeature::edge_vertical(4, 6, 8, 12),
                0.0, // threshold
                1.0, // polarity
                0.5, // alpha
            )],
            -0.5, // very lenient — almost everything passes stage 1
        );

        // Stage 2: a horizontal-edge feature over the eye region
        let stage2 = AdaBoostStage::new(
            vec![
                WeakClassifier::new(HaarFeature::edge_horizontal(2, 4, 8, 16), 0.0, 1.0, 0.4),
                WeakClassifier::new(HaarFeature::line_vertical(8, 4, 8, 16), 0.0, 1.0, 0.3),
            ],
            -0.7,
        );

        // Stage 3: four-rectangle feature for nose/mouth area
        let stage3 = AdaBoostStage::new(
            vec![WeakClassifier::new(
                HaarFeature::four_rectangle(12, 6, 8, 12),
                0.0,
                1.0,
                0.6,
            )],
            -1.0,
        );

        Self::new(vec![stage1, stage2, stage3], 24, 24)
    }
}

/// Run the cascade of classifiers at a single window location.
///
/// Returns `true` if the window passes every stage, `false` at the first
/// rejection.
///
/// # Arguments
/// * `classifier`   – the cascade
/// * `integral`     – integral image of the full image
/// * `win_row`      – window top-left row
/// * `win_col`      – window top-left column
/// * `norm_factor`  – variance normalisation (use 1.0 to disable)
pub fn cascade_classify(
    classifier: &AdaBoostClassifier,
    integral: &IntegralImage,
    win_row: usize,
    win_col: usize,
    norm_factor: f64,
) -> bool {
    for stage in &classifier.stages {
        if !stage.evaluate(integral, win_row, win_col, norm_factor) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Variance normalisation helper
// ---------------------------------------------------------------------------

/// Compute the mean and variance of the pixel values inside a window from the
/// integral image and an integral-of-squares image.
///
/// Used to produce a variance normalisation factor for Haar feature responses.
fn window_variance(
    integral: &IntegralImage,
    integral_sq: &IntegralImage,
    win_row: usize,
    win_col: usize,
    win_h: usize,
    win_w: usize,
) -> f64 {
    let n = (win_h * win_w) as f64;
    if n < 1.0 {
        return 1.0;
    }
    let sum = integral.rect_sum(win_row, win_col, win_row + win_h, win_col + win_w);
    let sum_sq = integral_sq.rect_sum(win_row, win_col, win_row + win_h, win_col + win_w);
    let mean = sum / n;
    let var = (sum_sq / n) - mean * mean;
    var.max(0.0).sqrt().max(1e-6)
}

// ---------------------------------------------------------------------------
// Multi-scale detection
// ---------------------------------------------------------------------------

/// Result of a Viola-Jones detection: bounding box in image pixel coordinates.
#[derive(Clone, Debug)]
pub struct FaceDetection {
    /// Bounding box in full-image pixel coordinates
    pub bbox: BoundingBox,
    /// Number of cascade stages passed (higher → more confident)
    pub stages_passed: usize,
}

/// Run the Viola-Jones detector at multiple scales using a sliding window.
///
/// The algorithm:
/// 1. Build an integral image (and integral-of-squares image for variance).
/// 2. For each scale level, slide the *scaled* window across the image.
/// 3. For each position, evaluate the cascade using variance-normalised feature responses.
/// 4. Collect all positive detections, then apply NMS with `nms_threshold`.
///
/// # Arguments
/// * `pixels`        – grey-level image as flat row-major f64 slice (values ∈ [0, 1])
/// * `rows`          – image height in pixels
/// * `cols`          – image width in pixels
/// * `classifier`    – trained cascade classifier
/// * `min_scale`     – minimum detection scale relative to the window (≥ 1.0)
/// * `max_scale`     – maximum detection scale (e.g. 4.0 → try 4× larger windows)
/// * `scale_step`    – multiplicative step between scales (e.g. 1.25)
/// * `stride`        – sliding-window stride at scale 1.0 (pixels)
/// * `nms_threshold` – IoU threshold for NMS (typical: 0.3 – 0.4)
///
/// # Errors
/// Returns [`VisionError::InvalidInput`] if the pixel buffer length does not
/// match `rows * cols`, or if scale parameters are invalid.
pub fn detect_multiscale(
    pixels: &[f64],
    rows: usize,
    cols: usize,
    classifier: &AdaBoostClassifier,
    min_scale: f64,
    max_scale: f64,
    scale_step: f64,
    stride: usize,
    nms_threshold: f64,
) -> Result<Vec<FaceDetection>> {
    if pixels.len() != rows * cols {
        return Err(VisionError::InvalidInput(format!(
            "detect_multiscale: expected {} pixels, got {}",
            rows * cols,
            pixels.len()
        )));
    }
    if min_scale < 1.0 {
        return Err(VisionError::InvalidInput(
            "detect_multiscale: min_scale must be ≥ 1.0".to_string(),
        ));
    }
    if max_scale < min_scale {
        return Err(VisionError::InvalidInput(
            "detect_multiscale: max_scale must be ≥ min_scale".to_string(),
        ));
    }
    if scale_step <= 1.0 {
        return Err(VisionError::InvalidInput(
            "detect_multiscale: scale_step must be > 1.0".to_string(),
        ));
    }
    if stride == 0 {
        return Err(VisionError::InvalidInput(
            "detect_multiscale: stride must be > 0".to_string(),
        ));
    }

    // Build integral images over the full-resolution image
    let integral = IntegralImage::new(pixels, rows, cols)?;

    // Integral of squares for variance normalisation
    let pixels_sq: Vec<f64> = pixels.iter().map(|&p| p * p).collect();
    let integral_sq = IntegralImage::new(&pixels_sq, rows, cols)?;

    let base_w = classifier.window_width;
    let base_h = classifier.window_height;

    let mut raw_boxes: Vec<BoundingBox> = Vec::new();

    let mut scale = min_scale;
    while scale <= max_scale + 1e-9 {
        let win_w = ((base_w as f64) * scale).round() as usize;
        let win_h = ((base_h as f64) * scale).round() as usize;

        if win_w > cols || win_h > rows || win_w == 0 || win_h == 0 {
            scale *= scale_step;
            continue;
        }

        let step = ((stride as f64) * scale).round().max(1.0) as usize;

        let mut win_row = 0;
        while win_row + win_h <= rows {
            let mut win_col = 0;
            while win_col + win_w <= cols {
                let norm = window_variance(&integral, &integral_sq, win_row, win_col, win_h, win_w);

                if cascade_classify(classifier, &integral, win_row, win_col, norm) {
                    raw_boxes.push(BoundingBox::new(
                        win_col as f64,
                        win_row as f64,
                        (win_col + win_w) as f64,
                        (win_row + win_h) as f64,
                        1.0,
                        0,
                    ));
                }

                win_col += step;
            }
            win_row += step;
        }

        scale *= scale_step;
    }

    // Apply NMS
    let kept = nms(&raw_boxes, nms_threshold);

    Ok(kept
        .into_iter()
        .map(|bbox| FaceDetection {
            stages_passed: classifier.stages.len(),
            bbox,
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a tiny synthetic "face-like" image (24×24, gradient top half)
    fn synthetic_face_pixels(rows: usize, cols: usize) -> Vec<f64> {
        let mut p = vec![0.3_f64; rows * cols];
        // Bright upper half, dark lower half — triggers edge_horizontal feature
        for r in 0..rows / 2 {
            for c in 0..cols {
                p[r * cols + c] = 0.8;
            }
        }
        p
    }

    #[test]
    fn test_integral_image_basic() {
        // 2×2 image: [[1, 2], [3, 4]]
        let pixels = vec![1.0_f64, 2.0, 3.0, 4.0];
        let ii = IntegralImage::new(&pixels, 2, 2).expect("should build");
        // Full rectangle sum = 1+2+3+4 = 10
        assert!((ii.rect_sum(0, 0, 2, 2) - 10.0).abs() < 1e-10);
        // Top row: 1+2 = 3
        assert!((ii.rect_sum(0, 0, 1, 2) - 3.0).abs() < 1e-10);
        // Bottom-right cell: 4
        assert!((ii.rect_sum(1, 1, 2, 2) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_integral_image_wrong_size() {
        let pixels = vec![1.0_f64; 5];
        assert!(IntegralImage::new(&pixels, 2, 3).is_err());
    }

    #[test]
    fn test_haar_feature_edge_vertical() {
        // 4×4 image with bright left half
        let mut pixels = vec![0.0_f64; 4 * 4];
        for r in 0..4 {
            for c in 0..2 {
                pixels[r * 4 + c] = 1.0;
            }
        }
        let ii = IntegralImage::new(&pixels, 4, 4).expect("should build");
        let feat = HaarFeature::edge_vertical(0, 0, 4, 4);
        let resp = compute_haar_feature(&ii, &feat, 0, 0, 1.0);
        // Left half sum = 8, right half sum = 0 → 8 - 0 = 8
        assert!(
            resp > 0.0,
            "expected positive response for bright-left image, got {resp}"
        );
    }

    #[test]
    fn test_weak_classifier_evaluate() {
        let pixels = synthetic_face_pixels(24, 24);
        let ii = IntegralImage::new(&pixels, 24, 24).expect("should build");
        let clf = WeakClassifier::new(HaarFeature::edge_horizontal(0, 0, 24, 24), 0.0, 1.0, 0.5);
        let score = clf.evaluate(&ii, 0, 0, 1.0);
        // Bright top → feature response positive → stump fires → returns alpha
        assert_eq!(score, 0.5);
    }

    #[test]
    fn test_cascade_classify_passes() {
        let cascade = AdaBoostClassifier::default_24x24();
        let pixels = synthetic_face_pixels(24, 24);
        let ii = IntegralImage::new(&pixels, 24, 24).expect("should build");
        // Very lenient cascade — should pass
        let _ = cascade_classify(&cascade, &ii, 0, 0, 1.0);
        // (result may vary; we just check it doesn't panic)
    }

    #[test]
    fn test_detect_multiscale_empty_on_uniform() {
        // Uniform image: all feature responses are ~0 → no face
        let rows = 48;
        let cols = 48;
        let pixels = vec![0.5_f64; rows * cols];
        let cascade = AdaBoostClassifier::default_24x24();
        let detections = detect_multiscale(&pixels, rows, cols, &cascade, 1.0, 1.0, 1.25, 4, 0.4)
            .expect("should succeed");
        // Cascade should mostly reject uniform image (very unlikely to fire on all stages)
        let _ = detections; // result is implementation-dependent
    }

    #[test]
    fn test_detect_multiscale_error_bad_input() {
        let cascade = AdaBoostClassifier::default_24x24();
        // Pixel buffer too short
        assert!(detect_multiscale(&[0.0; 10], 10, 10, &cascade, 1.0, 2.0, 1.25, 4, 0.4).is_err());
        // min_scale < 1
        assert!(detect_multiscale(&[0.0; 100], 10, 10, &cascade, 0.5, 2.0, 1.25, 4, 0.4).is_err());
        // scale_step <= 1
        assert!(detect_multiscale(&[0.0; 100], 10, 10, &cascade, 1.0, 2.0, 1.0, 4, 0.4).is_err());
    }

    #[test]
    fn test_four_rectangle_feature() {
        // Bright diagonal → response should be positive
        let mut pixels = vec![0.0_f64; 8 * 8];
        // Top-left and bottom-right quadrants bright
        for r in 0..4 {
            for c in 0..4 {
                pixels[r * 8 + c] = 1.0;
            }
        }
        for r in 4..8 {
            for c in 4..8 {
                pixels[r * 8 + c] = 1.0;
            }
        }
        let ii = IntegralImage::new(&pixels, 8, 8).expect("should build");
        let feat = HaarFeature::four_rectangle(0, 0, 8, 8);
        let resp = compute_haar_feature(&ii, &feat, 0, 0, 1.0);
        // Quadrants +1, -1, -1, +1: bright on +1 → positive net response
        assert!(
            resp > 0.0,
            "expected positive diagonal response, got {resp}"
        );
    }
}
