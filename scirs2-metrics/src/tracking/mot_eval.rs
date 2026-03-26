//! CLEAR MOT evaluation implementation.
//!
//! Implements MOTA, MOTP, IDF1 as described in:
//! Bernardin & Stiefelhagen (2008) "Evaluating Multiple Object Tracking Performance:
//! The CLEAR MOT Metrics"

use std::collections::HashMap;

use super::types::{
    FrameMatchResult, GtTrackStats, MatchAlg, TrackingBox, TrackingMetrics, TrackingMetricsConfig,
};

/// Evaluator for multi-object tracking metrics.
pub struct MotEvaluator {
    config: TrackingMetricsConfig,
}

impl MotEvaluator {
    /// Create a new evaluator with the given configuration.
    pub fn new(config: TrackingMetricsConfig) -> Self {
        Self { config }
    }

    /// Evaluate tracking performance over all frames.
    ///
    /// `gt` and `pred` are flat slices of boxes (any frame order).
    /// Frames with no detections are treated as empty.
    pub fn evaluate(&self, gt: &[TrackingBox], pred: &[TrackingBox]) -> TrackingMetrics {
        // Group by frame
        let mut gt_by_frame: HashMap<usize, Vec<&TrackingBox>> = HashMap::new();
        let mut pred_by_frame: HashMap<usize, Vec<&TrackingBox>> = HashMap::new();
        let mut max_frame = 0usize;

        for b in gt {
            gt_by_frame.entry(b.frame_id).or_default().push(b);
            max_frame = max_frame.max(b.frame_id);
        }
        for b in pred {
            pred_by_frame.entry(b.frame_id).or_default().push(b);
            max_frame = max_frame.max(b.frame_id);
        }

        let n_frames = max_frame + 1;

        // Per-GT-track statistics for MT/ML
        let mut gt_track_stats: HashMap<u64, GtTrackStats> = HashMap::new();
        for b in gt {
            let entry = gt_track_stats.entry(b.track_id).or_default();
            entry.total_frames += 1;
        }

        // Accumulated totals
        let mut total_tp = 0usize;
        let mut total_fp = 0usize;
        let mut total_fn = 0usize;
        let mut total_idsw = 0usize;
        let mut total_iou = 0.0f64;
        let mut n_gt = 0usize;

        // Previous frame's gt->pred assignment (for ID switch detection)
        let mut prev_gt_to_pred: HashMap<u64, u64> = HashMap::new();

        // For IDF1: count per-gt-track true positives / false positives / false negatives
        let mut id_tp_by_gt: HashMap<u64, usize> = HashMap::new();
        let mut id_fp_total = 0usize;
        let mut id_fn_total = 0usize;

        for frame_id in 0..n_frames {
            let empty_gt: Vec<&TrackingBox> = vec![];
            let empty_pred: Vec<&TrackingBox> = vec![];
            let gt_frame = gt_by_frame.get(&frame_id).unwrap_or(&empty_gt);
            let pred_frame = pred_by_frame.get(&frame_id).unwrap_or(&empty_pred);

            n_gt += gt_frame.len();

            let result = self.match_frame(gt_frame, pred_frame, &mut prev_gt_to_pred);

            total_tp += result.tp;
            total_fp += result.fp;
            total_fn += result.fn_;
            total_idsw += result.id_switches;
            total_iou += result.total_iou;

            // Update GT-track match stats for MT/ML
            for gt_id in result.gt_to_pred.keys() {
                let entry = gt_track_stats.entry(*gt_id).or_default();
                entry.matched_frames += 1;
                *id_tp_by_gt.entry(*gt_id).or_default() += 1;
            }
            // FP: predicted boxes that were not matched
            id_fp_total += result.fp;
            // FN: GT boxes that were not matched
            id_fn_total += result.fn_;

            prev_gt_to_pred = result.gt_to_pred;
        }

        // MOTA = 1 - (FP + FN + IDSW) / GT
        let mota = if n_gt == 0 {
            0.0
        } else {
            1.0 - (total_fp + total_fn + total_idsw) as f64 / n_gt as f64
        };

        // MOTP = mean IoU over matched pairs
        let motp = if total_tp == 0 {
            0.0
        } else {
            total_iou / total_tp as f64
        };

        // IDF1: 2*IDTP / (2*IDTP + IDFP + IDFN)
        // IDTP = total matched frames across all gt tracks
        let id_tp_total: usize = id_tp_by_gt.values().sum();
        let idf1 = {
            let denom = 2 * id_tp_total + id_fp_total + id_fn_total;
            if denom == 0 {
                0.0
            } else {
                2.0 * id_tp_total as f64 / denom as f64
            }
        };

        // MT / ML
        let mostly_tracked = gt_track_stats
            .values()
            .filter(|s| {
                s.total_frames > 0 && (s.matched_frames as f64 / s.total_frames as f64) > 0.8
            })
            .count();
        let mostly_lost = gt_track_stats
            .values()
            .filter(|s| {
                s.total_frames > 0 && (s.matched_frames as f64 / s.total_frames as f64) < 0.2
            })
            .count();

        TrackingMetrics {
            mota,
            motp,
            idf1,
            id_switches: total_idsw,
            false_positives: total_fp,
            false_negatives: total_fn,
            mostly_tracked,
            mostly_lost,
            tp: total_tp,
            n_gt,
        }
    }

    /// Match detections in a single frame using Hungarian or Greedy assignment.
    ///
    /// Returns per-frame TP/FP/FN/IDSW/IoU totals and the current assignment map.
    fn match_frame(
        &self,
        gt_frame: &[&TrackingBox],
        pred_frame: &[&TrackingBox],
        prev_gt_to_pred: &mut HashMap<u64, u64>,
    ) -> FrameMatchResult {
        let n_gt = gt_frame.len();
        let n_pred = pred_frame.len();

        if n_gt == 0 && n_pred == 0 {
            return FrameMatchResult::default();
        }
        if n_gt == 0 {
            return FrameMatchResult {
                fp: n_pred,
                ..Default::default()
            };
        }
        if n_pred == 0 {
            return FrameMatchResult {
                fn_: n_gt,
                ..Default::default()
            };
        }

        // Build IoU cost matrix
        let mut iou_matrix = vec![vec![0.0f64; n_pred]; n_gt];
        for (i, g) in gt_frame.iter().enumerate() {
            for (j, p) in pred_frame.iter().enumerate() {
                iou_matrix[i][j] = g.iou(p);
            }
        }

        // Compute assignment
        let assignment = match self.config.match_alg {
            MatchAlg::Hungarian => hungarian_assignment(&iou_matrix, n_gt, n_pred),
            MatchAlg::Greedy => greedy_assignment(&iou_matrix, n_gt, n_pred),
        };

        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_ = 0usize;
        let mut id_switches = 0usize;
        let mut total_iou = 0.0f64;
        let mut gt_to_pred: HashMap<u64, u64> = HashMap::new();
        let mut matched_pred = vec![false; n_pred];

        for (gt_idx, pred_idx_opt) in assignment.iter().enumerate() {
            let g = gt_frame[gt_idx];
            match pred_idx_opt {
                Some(pred_idx) => {
                    let iou_val = iou_matrix[gt_idx][*pred_idx];
                    if iou_val >= self.config.iou_threshold {
                        tp += 1;
                        total_iou += iou_val;
                        let p = pred_frame[*pred_idx];
                        gt_to_pred.insert(g.track_id, p.track_id);
                        matched_pred[*pred_idx] = true;

                        // Check for ID switch
                        if let Some(&prev_pred_id) = prev_gt_to_pred.get(&g.track_id) {
                            if prev_pred_id != p.track_id {
                                id_switches += 1;
                            }
                        }
                    } else {
                        fn_ += 1;
                    }
                }
                None => {
                    fn_ += 1;
                }
            }
        }

        // Count unmatched predictions as FP
        fp += matched_pred.iter().filter(|&&m| !m).count();

        FrameMatchResult {
            tp,
            fp,
            fn_,
            id_switches,
            total_iou,
            gt_to_pred,
        }
    }
}

/// Hungarian (Munkres) algorithm for maximum-weight bipartite matching.
/// Returns Vec of length n_gt, each entry is Some(pred_idx) or None.
fn hungarian_assignment(iou_matrix: &[Vec<f64>], n_gt: usize, n_pred: usize) -> Vec<Option<usize>> {
    // Convert to cost matrix (we want max IoU, so negate)
    let n = n_gt.max(n_pred);
    let inf = 1e18_f64;

    // Padded cost matrix (n x n)
    let mut cost = vec![vec![inf; n]; n];
    for i in 0..n_gt {
        for j in 0..n_pred {
            cost[i][j] = 1.0 - iou_matrix[i][j];
        }
    }

    // Standard O(n^3) Hungarian
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1]; // p[j] = row matched to column j (1-indexed)
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;

            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    // Extract assignment: for each gt row i, find column j where p[j] == i+1
    let mut ans = vec![None; n_gt];
    for j in 1..=n {
        if p[j] > 0 && p[j] <= n_gt {
            let row = p[j] - 1; // 0-indexed gt
            let col = j - 1; // 0-indexed pred
            if col < n_pred {
                ans[row] = Some(col);
            }
        }
    }
    ans
}

/// Greedy assignment: iteratively take highest-IoU pairs.
fn greedy_assignment(iou_matrix: &[Vec<f64>], n_gt: usize, n_pred: usize) -> Vec<Option<usize>> {
    let mut ans = vec![None; n_gt];
    let mut used_pred = vec![false; n_pred];

    // Flatten all pairs and sort by iou descending
    let mut pairs: Vec<(f64, usize, usize)> = Vec::with_capacity(n_gt * n_pred);
    for i in 0..n_gt {
        for j in 0..n_pred {
            pairs.push((iou_matrix[i][j], i, j));
        }
    }
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut used_gt = vec![false; n_gt];
    for (_, i, j) in pairs {
        if !used_gt[i] && !used_pred[j] {
            ans[i] = Some(j);
            used_gt[i] = true;
            used_pred[j] = true;
        }
    }
    ans
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::types::{MatchAlg, TrackingMetricsConfig};

    fn make_box(frame: usize, track: u64, x1: f64, y1: f64, x2: f64, y2: f64) -> TrackingBox {
        TrackingBox::new(frame, track, x1, y1, x2, y2, 1.0)
    }

    #[test]
    fn test_perfect_tracking() {
        // Single object, single frame, perfect match
        let gt = vec![make_box(0, 1, 0.0, 0.0, 10.0, 10.0)];
        let pred = vec![make_box(0, 1, 0.0, 0.0, 10.0, 10.0)];
        let config = TrackingMetricsConfig::default();
        let evaluator = MotEvaluator::new(config);
        let m = evaluator.evaluate(&gt, &pred);
        assert!(
            (m.mota - 1.0).abs() < 1e-9,
            "MOTA should be 1.0: {}",
            m.mota
        );
        assert!(
            (m.motp - 1.0).abs() < 1e-9,
            "MOTP should be 1.0: {}",
            m.motp
        );
        assert_eq!(m.id_switches, 0);
        assert_eq!(m.false_positives, 0);
        assert_eq!(m.false_negatives, 0);
    }

    #[test]
    fn test_all_false_positives() {
        // No GT, all preds are FP
        let gt: Vec<TrackingBox> = vec![];
        let pred = vec![make_box(0, 1, 0.0, 0.0, 10.0, 10.0)];
        let config = TrackingMetricsConfig::default();
        let evaluator = MotEvaluator::new(config);
        let m = evaluator.evaluate(&gt, &pred);
        // No GT means n_gt=0, MOTA=0 by convention (no denominator)
        assert_eq!(m.false_positives, 1);
        assert_eq!(m.n_gt, 0);
    }

    #[test]
    fn test_iou_nonoverlapping() {
        let b1 = make_box(0, 1, 0.0, 0.0, 1.0, 1.0);
        let b2 = make_box(0, 2, 5.0, 5.0, 6.0, 6.0);
        assert_eq!(b1.iou(&b2), 0.0);
    }

    #[test]
    fn test_iou_identical() {
        let b1 = make_box(0, 1, 0.0, 0.0, 10.0, 10.0);
        let b2 = make_box(0, 1, 0.0, 0.0, 10.0, 10.0);
        assert!((b1.iou(&b2) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_id_switch_counted() {
        // Two frames, GT track 1, pred switches from track 10 to track 20
        let gt = vec![
            make_box(0, 1, 0.0, 0.0, 10.0, 10.0),
            make_box(1, 1, 0.0, 0.0, 10.0, 10.0),
        ];
        let pred = vec![
            make_box(0, 10, 0.0, 0.0, 10.0, 10.0),
            make_box(1, 20, 0.0, 0.0, 10.0, 10.0),
        ];
        let config = TrackingMetricsConfig::default();
        let evaluator = MotEvaluator::new(config);
        let m = evaluator.evaluate(&gt, &pred);
        assert_eq!(m.id_switches, 1, "Should detect 1 ID switch");
    }

    #[test]
    fn test_config_default() {
        let c = TrackingMetricsConfig::default();
        assert!((c.iou_threshold - 0.5).abs() < 1e-9);
        assert_eq!(c.match_alg, MatchAlg::Hungarian);
    }

    #[test]
    fn test_greedy_alg() {
        let gt = vec![make_box(0, 1, 0.0, 0.0, 10.0, 10.0)];
        let pred = vec![make_box(0, 1, 0.0, 0.0, 10.0, 10.0)];
        let config = TrackingMetricsConfig {
            match_alg: MatchAlg::Greedy,
            ..Default::default()
        };
        let evaluator = MotEvaluator::new(config);
        let m = evaluator.evaluate(&gt, &pred);
        assert!((m.mota - 1.0).abs() < 1e-9);
    }
}
