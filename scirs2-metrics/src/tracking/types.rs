//! Types for multi-object tracking metrics.

use std::collections::HashMap;

/// A 2D bounding box associated with a specific frame and track.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TrackingBox {
    /// Frame index (0-based)
    pub frame_id: usize,
    /// Track identifier
    pub track_id: u64,
    /// Left coordinate
    pub x1: f64,
    /// Top coordinate
    pub y1: f64,
    /// Right coordinate
    pub x2: f64,
    /// Bottom coordinate
    pub y2: f64,
    /// Detection confidence score
    pub confidence: f64,
}

impl TrackingBox {
    /// Construct a new TrackingBox.
    pub fn new(
        frame_id: usize,
        track_id: u64,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        confidence: f64,
    ) -> Self {
        Self {
            frame_id,
            track_id,
            x1,
            y1,
            x2,
            y2,
            confidence,
        }
    }

    /// Area of this bounding box (zero if degenerate).
    pub fn area(&self) -> f64 {
        let w = (self.x2 - self.x1).max(0.0);
        let h = (self.y2 - self.y1).max(0.0);
        w * h
    }

    /// Intersection over Union with another box.
    pub fn iou(&self, other: &TrackingBox) -> f64 {
        let ix1 = self.x1.max(other.x1);
        let iy1 = self.y1.max(other.y1);
        let ix2 = self.x2.min(other.x2);
        let iy2 = self.y2.min(other.y2);

        let iw = (ix2 - ix1).max(0.0);
        let ih = (iy2 - iy1).max(0.0);
        let intersection = iw * ih;

        if intersection <= 0.0 {
            return 0.0;
        }

        let union = self.area() + other.area() - intersection;
        if union <= 0.0 {
            0.0
        } else {
            intersection / union
        }
    }
}

/// Algorithm to use for frame-level assignment.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchAlg {
    /// Hungarian (Munkres) optimal assignment
    Hungarian,
    /// Greedy best-first matching
    Greedy,
}

/// Configuration for tracking metric evaluation.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TrackingMetricsConfig {
    /// Minimum IoU overlap to count as a match (default: 0.5)
    pub iou_threshold: f64,
    /// Assignment algorithm (default: Hungarian)
    pub match_alg: MatchAlg,
}

impl Default for TrackingMetricsConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.5,
            match_alg: MatchAlg::Hungarian,
        }
    }
}

/// Summary of tracking performance over all frames.
#[derive(Debug, Clone)]
pub struct TrackingMetrics {
    /// Multi-Object Tracking Accuracy: 1 - (FP+FN+IDSW)/GT
    pub mota: f64,
    /// Multi-Object Tracking Precision: mean IoU of matched pairs
    pub motp: f64,
    /// ID F1 score
    pub idf1: f64,
    /// Number of identity switches
    pub id_switches: usize,
    /// Number of false positives
    pub false_positives: usize,
    /// Number of false negatives (missed detections)
    pub false_negatives: usize,
    /// Tracks with >80% of GT frames covered
    pub mostly_tracked: usize,
    /// Tracks with <20% of GT frames covered
    pub mostly_lost: usize,
    /// True positive count
    pub tp: usize,
    /// Total GT detection count
    pub n_gt: usize,
}

/// Per-GT-track statistics used for MT/ML computation.
#[derive(Debug, Default)]
pub struct GtTrackStats {
    /// Total frames this GT track appears in
    pub total_frames: usize,
    /// Frames where this GT track was successfully matched
    pub matched_frames: usize,
}

/// Frame-level intermediate matching result.
#[derive(Debug, Default)]
pub struct FrameMatchResult {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub id_switches: usize,
    pub total_iou: f64,
    /// Map from gt_id -> matched pred_id (for this frame)
    pub gt_to_pred: HashMap<u64, u64>,
}
