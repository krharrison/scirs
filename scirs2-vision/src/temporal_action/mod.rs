//! Temporal action recognition for video understanding.
//!
//! Provides anchor-free temporal action detection with sliding-window feature
//! extraction, softmax-based class scoring, 1-D temporal NMS, and mean
//! Average Precision evaluation.

use crate::error::{Result, VisionError};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for temporal action recognition.
#[derive(Debug, Clone)]
pub struct TemporalConfig {
    /// Number of frames per sliding window.
    pub window_size: usize,
    /// Stride between consecutive windows.
    pub stride: usize,
    /// Number of action classes.
    pub n_classes: usize,
    /// Dimensionality of per-frame feature vectors.
    pub feature_dim: usize,
    /// IoU threshold for temporal NMS.
    pub iou_threshold: f64,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            window_size: 16,
            stride: 8,
            n_classes: 10,
            feature_dim: 512,
            iou_threshold: 0.5,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core data structures
// ─────────────────────────────────────────────────────────────────────────────

/// A single detected action proposal in temporal space.
#[derive(Debug, Clone)]
pub struct ActionProposal {
    /// Start frame index (inclusive).
    pub start_frame: usize,
    /// End frame index (exclusive).
    pub end_frame: usize,
    /// Predicted class index.
    pub class_id: usize,
    /// Confidence score in `[0, 1]`.
    pub confidence: f64,
}

/// Dense feature representation of a video clip.
#[derive(Debug, Clone)]
pub struct VideoFeatures {
    /// Total number of temporal positions (one per sliding window).
    pub n_frames: usize,
    /// Dimensionality of each feature vector.
    pub feature_dim: usize,
    /// Feature matrix: `n_frames × feature_dim`.
    pub features: Vec<Vec<f64>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Pooling helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the temporal average pool of a slice of feature vectors.
///
/// Returns a vector of length equal to the feature dimension.
/// Returns an empty vector if `features` is empty.
pub fn temporal_average_pool(features: &[Vec<f64>]) -> Vec<f64> {
    if features.is_empty() {
        return Vec::new();
    }
    let dim = features[0].len();
    let n = features.len() as f64;
    let mut out = vec![0.0_f64; dim];
    for fv in features {
        for (d, &v) in fv.iter().enumerate() {
            if d < dim {
                out[d] += v;
            }
        }
    }
    for v in &mut out {
        *v /= n;
    }
    out
}

/// Compute the temporal max pool of a slice of feature vectors.
///
/// Returns a vector of length equal to the feature dimension.
/// Returns an empty vector if `features` is empty.
pub fn temporal_max_pool(features: &[Vec<f64>]) -> Vec<f64> {
    if features.is_empty() {
        return Vec::new();
    }
    let dim = features[0].len();
    let mut out = vec![f64::NEG_INFINITY; dim];
    for fv in features {
        for (d, &v) in fv.iter().enumerate() {
            if d < dim && v > out[d] {
                out[d] = v;
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic pseudo-random weight initialisation
// ─────────────────────────────────────────────────────────────────────────────

/// Simple LCG PRNG for reproducible weight initialisation (no external crates).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Return the next value in `[-1, 1]`.
    fn next_f64(&mut self) -> f64 {
        // Numerical Recipes LCG constants
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.state >> 33) as f64;
        bits / (u32::MAX as f64) * 2.0 - 1.0
    }
}

/// Build a random linear projection matrix `out_dim × in_dim`.
fn random_projection(in_dim: usize, out_dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Lcg::new(seed);
    let scale = (1.0 / in_dim as f64).sqrt();
    (0..out_dim)
        .map(|_| (0..in_dim).map(|_| rng.next_f64() * scale).collect())
        .collect()
}

/// Apply a linear projection: `out = W · x`, where `W` is `out_dim × in_dim`.
fn linear_project(weights: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    weights
        .iter()
        .map(|row| row.iter().zip(x.iter()).map(|(w, v)| w * v).sum::<f64>())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature extraction
// ─────────────────────────────────────────────────────────────────────────────

/// Extract temporal features from a sequence of per-frame feature vectors.
///
/// A sliding window of width `window_size` (step `window_size / 2`) is applied.
/// For each window the average pool and max pool are concatenated into a
/// `2 × feature_dim` descriptor, then projected down to `feature_dim` via a
/// random linear map.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] if `frame_features` is empty or
/// any inner vector is empty.
pub fn extract_temporal_features(
    frame_features: &[Vec<f64>],
    window_size: usize,
) -> Result<VideoFeatures> {
    if frame_features.is_empty() {
        return Err(VisionError::InvalidParameter(
            "frame_features must not be empty".into(),
        ));
    }
    let feature_dim = frame_features[0].len();
    if feature_dim == 0 {
        return Err(VisionError::InvalidParameter(
            "feature_dim must be > 0".into(),
        ));
    }
    if window_size == 0 {
        return Err(VisionError::InvalidParameter(
            "window_size must be > 0".into(),
        ));
    }

    let n_total = frame_features.len();
    let stride = (window_size / 2).max(1);
    let proj_dim = feature_dim; // project 2*feature_dim → feature_dim
    let proj = random_projection(2 * feature_dim, proj_dim, 42);

    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut pos = 0usize;
    loop {
        let start = pos;
        let end = (pos + window_size).min(n_total);
        if start >= n_total {
            break;
        }
        let window = &frame_features[start..end];

        let avg = temporal_average_pool(window);
        let mx = temporal_max_pool(window);

        // Concatenate avg ++ max → 2*feature_dim
        let mut concat = Vec::with_capacity(2 * feature_dim);
        concat.extend_from_slice(&avg);
        concat.extend_from_slice(&mx);

        // Project to feature_dim
        let projected = linear_project(&proj, &concat);
        features.push(projected);

        pos += stride;
        if end == n_total {
            break;
        }
    }

    let n_windows = features.len();
    Ok(VideoFeatures {
        n_frames: n_windows,
        feature_dim: proj_dim,
        features,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Softmax
// ─────────────────────────────────────────────────────────────────────────────

fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / logits.len() as f64; logits.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Action detector
// ─────────────────────────────────────────────────────────────────────────────

/// Anchor-free temporal action detector.
///
/// Each sliding-window position is scored against `n_classes` class prototypes.
pub struct ActionDetector {
    /// Runtime configuration.
    pub config: TemporalConfig,
    /// Class weight matrix: `n_classes × feature_dim`.
    pub class_weights: Vec<Vec<f64>>,
}

impl ActionDetector {
    /// Construct a new detector with random class weights.
    pub fn new(config: TemporalConfig) -> Self {
        let class_weights = random_projection(config.feature_dim, config.n_classes, 1337);
        Self {
            config,
            class_weights,
        }
    }

    /// Score every window position in `features` and produce raw proposals.
    ///
    /// The temporal interval of each proposal is derived from the sliding-window
    /// position (stride = `window_size / 2`, min 1).
    pub fn score_proposals(&self, features: &VideoFeatures) -> Vec<ActionProposal> {
        let stride = (self.config.window_size / 2).max(1);
        let mut proposals = Vec::with_capacity(features.n_frames);

        for (i, feat) in features.features.iter().enumerate() {
            // Compute raw class scores = class_weights · feature
            let logits: Vec<f64> = self
                .class_weights
                .iter()
                .map(|row| row.iter().zip(feat.iter()).map(|(w, v)| w * v).sum::<f64>())
                .collect();

            let probs = softmax(&logits);

            // Top-1 class
            let (class_id, &confidence) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));

            let start_frame = i * stride;
            let end_frame = start_frame + self.config.window_size;

            proposals.push(ActionProposal {
                start_frame,
                end_frame,
                class_id,
                confidence,
            });
        }
        proposals
    }

    /// Run non-maximum suppression and return final proposals.
    pub fn detect(&self, features: &VideoFeatures) -> Vec<ActionProposal> {
        let proposals = self.score_proposals(features);
        nms_temporal(proposals, self.config.iou_threshold)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Temporal IoU and NMS
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the 1-D temporal Intersection over Union between two proposals.
///
/// Intervals are `[start, end)`.  Returns `0.0` if they do not overlap.
pub fn temporal_iou(a: &ActionProposal, b: &ActionProposal) -> f64 {
    let inter_start = a.start_frame.max(b.start_frame);
    let inter_end = a.end_frame.min(b.end_frame);
    if inter_end <= inter_start {
        return 0.0;
    }
    let intersection = (inter_end - inter_start) as f64;
    let union =
        (a.end_frame - a.start_frame) as f64 + (b.end_frame - b.start_frame) as f64 - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// 1-D temporal Non-Maximum Suppression.
///
/// Proposals are sorted by confidence (descending); any proposal that overlaps
/// the current highest-confidence proposal by more than `iou_threshold` is
/// suppressed.
pub fn nms_temporal(mut proposals: Vec<ActionProposal>, iou_threshold: f64) -> Vec<ActionProposal> {
    // Sort by confidence descending
    proposals.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<ActionProposal> = Vec::new();
    let mut suppressed = vec![false; proposals.len()];

    for i in 0..proposals.len() {
        if suppressed[i] {
            continue;
        }
        kept.push(proposals[i].clone());
        for j in (i + 1)..proposals.len() {
            if suppressed[j] {
                continue;
            }
            if temporal_iou(&proposals[i], &proposals[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    kept
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluation metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the temporal mean Average Precision (mAP) across all classes.
///
/// For each class the predictions are sorted by confidence, and the area under
/// the Precision-Recall curve (via trapezoidal integration) is computed.  The
/// per-class APs are then averaged.
///
/// # Parameters
///
/// * `predictions`   – detector output (may cover multiple classes).
/// * `ground_truth`  – ground-truth annotations (may cover multiple classes).
/// * `iou_threshold` – IoU threshold above which a prediction counts as a TP.
///
/// Returns `0.0` if there are no ground-truth annotations.
pub fn mean_average_precision_temporal(
    predictions: &[ActionProposal],
    ground_truth: &[ActionProposal],
    iou_threshold: f64,
) -> f64 {
    if ground_truth.is_empty() {
        return 0.0;
    }

    // Collect unique class IDs from ground truth
    let mut classes: Vec<usize> = ground_truth.iter().map(|p| p.class_id).collect();
    classes.sort_unstable();
    classes.dedup();

    let mut ap_sum = 0.0_f64;
    for &cls in &classes {
        let gt_cls: Vec<&ActionProposal> =
            ground_truth.iter().filter(|p| p.class_id == cls).collect();
        let mut preds_cls: Vec<&ActionProposal> =
            predictions.iter().filter(|p| p.class_id == cls).collect();

        if gt_cls.is_empty() {
            continue;
        }

        // Sort predictions by confidence descending
        preds_cls.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n_gt = gt_cls.len();
        let mut matched = vec![false; n_gt];
        let mut tp = 0usize;
        let mut fp = 0usize;

        // Precision-Recall pairs
        let mut pr_pairs: Vec<(f64, f64)> = Vec::new();

        for pred in &preds_cls {
            // Find best-matching unmatched GT
            let best = gt_cls
                .iter()
                .enumerate()
                .filter(|(idx, _)| !matched[*idx])
                .map(|(idx, gt)| (idx, temporal_iou(pred, gt)))
                .filter(|(_, iou)| *iou >= iou_threshold)
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((idx, _)) = best {
                tp += 1;
                matched[idx] = true;
            } else {
                fp += 1;
            }

            let precision = tp as f64 / (tp + fp) as f64;
            let recall = tp as f64 / n_gt as f64;
            pr_pairs.push((recall, precision));
        }

        // Area under PR curve via trapezoidal rule
        let ap = area_under_pr_curve(&pr_pairs);
        ap_sum += ap;
    }

    ap_sum / classes.len() as f64
}

/// Compute area under a Precision-Recall curve using the trapezoidal rule.
fn area_under_pr_curve(pr_pairs: &[(f64, f64)]) -> f64 {
    if pr_pairs.is_empty() {
        return 0.0;
    }
    // Sort by recall ascending
    let mut sorted = pr_pairs.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Prepend (0, 1) for completeness
    let mut points = vec![(0.0_f64, 1.0_f64)];
    points.extend_from_slice(&sorted);

    let mut area = 0.0_f64;
    for w in points.windows(2) {
        let dr = w[1].0 - w[0].0;
        let avg_p = (w[0].1 + w[1].1) / 2.0;
        area += dr * avg_p;
    }
    area.max(0.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame_features(n_frames: usize, dim: usize) -> Vec<Vec<f64>> {
        (0..n_frames)
            .map(|i| (0..dim).map(|d| (i + d) as f64 * 0.01).collect())
            .collect()
    }

    #[test]
    fn test_extract_temporal_features_dimensions() {
        let frames = make_frame_features(32, 64);
        let vf = extract_temporal_features(&frames, 8).expect("extraction failed");
        // feature_dim should equal input dim (projection maps 2*64 → 64)
        assert_eq!(vf.feature_dim, 64);
        assert!(vf.n_frames > 0);
        assert_eq!(vf.features.len(), vf.n_frames);
        for fv in &vf.features {
            assert_eq!(fv.len(), 64);
        }
    }

    #[test]
    fn test_extract_temporal_features_single_window() {
        // Fewer frames than window_size → exactly 1 window
        let frames = make_frame_features(4, 16);
        let vf = extract_temporal_features(&frames, 8).expect("extraction failed");
        assert_eq!(vf.n_frames, 1);
    }

    #[test]
    fn test_extract_temporal_features_error_on_empty() {
        let result = extract_temporal_features(&[], 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_temporal_average_pool() {
        let feats = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let avg = temporal_average_pool(&feats);
        assert!((avg[0] - 2.0).abs() < 1e-9);
        assert!((avg[1] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_temporal_max_pool() {
        let feats = vec![vec![1.0, 5.0], vec![3.0, 2.0]];
        let mx = temporal_max_pool(&feats);
        assert!((mx[0] - 3.0).abs() < 1e-9);
        assert!((mx[1] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_temporal_iou_non_overlapping() {
        let a = ActionProposal {
            start_frame: 0,
            end_frame: 5,
            class_id: 0,
            confidence: 0.9,
        };
        let b = ActionProposal {
            start_frame: 5,
            end_frame: 10,
            class_id: 0,
            confidence: 0.8,
        };
        assert_eq!(temporal_iou(&a, &b), 0.0);
    }

    #[test]
    fn test_temporal_iou_full_overlap() {
        let a = ActionProposal {
            start_frame: 0,
            end_frame: 10,
            class_id: 0,
            confidence: 0.9,
        };
        let b = ActionProposal {
            start_frame: 0,
            end_frame: 10,
            class_id: 0,
            confidence: 0.8,
        };
        assert!((temporal_iou(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_temporal_iou_partial() {
        let a = ActionProposal {
            start_frame: 0,
            end_frame: 10,
            class_id: 0,
            confidence: 0.9,
        };
        let b = ActionProposal {
            start_frame: 5,
            end_frame: 15,
            class_id: 0,
            confidence: 0.8,
        };
        // intersection = 5, union = 15
        let expected = 5.0 / 15.0;
        assert!((temporal_iou(&a, &b) - expected).abs() < 1e-9);
    }

    #[test]
    fn test_nms_temporal_removes_overlap() {
        let proposals = vec![
            ActionProposal {
                start_frame: 0,
                end_frame: 10,
                class_id: 0,
                confidence: 0.9,
            },
            ActionProposal {
                start_frame: 2,
                end_frame: 12,
                class_id: 0,
                confidence: 0.7,
            },
            ActionProposal {
                start_frame: 20,
                end_frame: 30,
                class_id: 0,
                confidence: 0.6,
            },
        ];
        let kept = nms_temporal(proposals, 0.3);
        // The second proposal overlaps with the first heavily; should be suppressed
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_nms_temporal_no_overlap() {
        let proposals = vec![
            ActionProposal {
                start_frame: 0,
                end_frame: 5,
                class_id: 0,
                confidence: 0.9,
            },
            ActionProposal {
                start_frame: 10,
                end_frame: 15,
                class_id: 0,
                confidence: 0.8,
            },
            ActionProposal {
                start_frame: 20,
                end_frame: 25,
                class_id: 0,
                confidence: 0.7,
            },
        ];
        let kept = nms_temporal(proposals, 0.5);
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn test_action_detector_smoke() {
        let config = TemporalConfig {
            window_size: 4,
            stride: 2,
            n_classes: 3,
            feature_dim: 8,
            iou_threshold: 0.5,
        };
        let detector = ActionDetector::new(config);
        let frames = make_frame_features(16, 8);
        let vf = extract_temporal_features(&frames, 4).expect("extraction failed");
        let proposals = detector.score_proposals(&vf);
        assert!(!proposals.is_empty());
        for p in &proposals {
            assert!(p.confidence >= 0.0 && p.confidence <= 1.0);
        }
    }

    #[test]
    fn test_mean_average_precision_perfect() {
        // One class, one ground-truth annotation, one matching prediction
        let gt = vec![ActionProposal {
            start_frame: 0,
            end_frame: 10,
            class_id: 0,
            confidence: 1.0,
        }];
        let pred = vec![ActionProposal {
            start_frame: 0,
            end_frame: 10,
            class_id: 0,
            confidence: 0.95,
        }];
        let map = mean_average_precision_temporal(&pred, &gt, 0.5);
        // Perfect match → AP = 1.0 → mAP = 1.0
        assert!((map - 1.0).abs() < 1e-9, "mAP = {map}");
    }

    #[test]
    fn test_mean_average_precision_empty_gt() {
        let map = mean_average_precision_temporal(&[], &[], 0.5);
        assert_eq!(map, 0.0);
    }
}
