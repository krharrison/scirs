//! Panoptic segmentation utilities
//!
//! Panoptic segmentation (Kirillov et al. 2019) unifies semantic and instance
//! segmentation under one framework that assigns every pixel a `(category, instance_id)`
//! pair.  This module provides:
//!
//! - [`PanopticSegment`] – lightweight descriptor for one panoptic segment.
//! - [`panoptic_quality`] – Panoptic Quality (PQ), Segmentation Quality (SQ),
//!   and Recognition Quality (RQ) metrics.
//! - [`merge_semantic_instance`] – combine a semantic label map with an instance
//!   mask map into a panoptic encoding.
//! - [`instance_to_panoptic`] – convert per-instance binary masks into a
//!   panoptic label map.
//!
//! # Array layout
//!
//! Label maps are 2-D integer arrays (`Array2<i32>`, shape `[H, W]`).

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::{Array2, Array3};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

/// A single panoptic segment descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct PanopticSegment {
    /// Unique segment identifier (globally unique within one image).
    pub id: u64,
    /// Semantic category index.
    pub category: usize,
    /// `true` if this is a "thing" class (countable objects like cars, people);
    /// `false` if it is a "stuff" class (amorphous regions like sky, grass).
    pub is_thing: bool,
    /// Pixel area (number of pixels belonging to this segment).
    pub area: usize,
    /// Optional RGB visualization colour.
    pub color: Option<(u8, u8, u8)>,
}

impl PanopticSegment {
    /// Construct a new `PanopticSegment`.
    pub fn new(id: u64, category: usize, is_thing: bool, area: usize) -> Self {
        Self { id, category, is_thing, area, color: None }
    }

    /// Attach a visualization colour.
    pub fn with_color(mut self, color: (u8, u8, u8)) -> Self {
        self.color = Some(color);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Panoptic Quality metric
// ─────────────────────────────────────────────────────────────────────────────

/// Matching between one predicted and one ground-truth segment.
#[derive(Debug, Clone)]
struct MatchedPair {
    iou: f64,
}

/// Compute Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition
/// Quality (RQ) from predicted and ground-truth panoptic label maps.
///
/// The three quantities are defined per the original paper:
///
/// - **SQ** = sum of IoU for matched pairs / number of matches
/// - **RQ** = TP / (TP + 0.5 FP + 0.5 FN)
/// - **PQ** = SQ × RQ = sum of IoU / (TP + 0.5 FP + 0.5 FN)
///
/// A predicted–GT pair is considered a match when their IoU ≥ 0.5 (the fixed
/// panoptic matching threshold).
///
/// Segments with `id == 0` are treated as the "void" / "crowd" label and
/// excluded from the metric.
///
/// # Arguments
///
/// * `pred_map`   – Predicted panoptic label map `[H, W]` (integer segment IDs).
/// * `gt_map`     – Ground-truth panoptic label map `[H, W]`.
/// * `pred_segs`  – Metadata for predicted segments (used to look up IDs).
/// * `gt_segs`    – Metadata for ground-truth segments.
///
/// # Returns
///
/// `(pq, sq, rq)` – all values in `[0, 1]`.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when `pred_map` and `gt_map` have
/// different shapes.
pub fn panoptic_quality(
    pred_map: &Array2<i64>,
    gt_map: &Array2<i64>,
    pred_segs: &[PanopticSegment],
    gt_segs: &[PanopticSegment],
) -> Result<(f64, f64, f64)> {
    let (ph, pw) = pred_map.dim();
    let (gh, gw) = gt_map.dim();
    if ph != gh || pw != gw {
        return Err(VisionError::InvalidParameter(format!(
            "panoptic_quality: pred_map shape {}×{} != gt_map shape {}×{}",
            ph, pw, gh, gw
        )));
    }
    let h = ph;
    let w = pw;

    // Build sets of valid segment IDs.
    let pred_ids: std::collections::HashSet<i64> = pred_segs.iter().map(|s| s.id as i64).collect();
    let gt_ids: std::collections::HashSet<i64> = gt_segs.iter().map(|s| s.id as i64).collect();

    // Intersection matrix: (pred_id, gt_id) → overlap pixel count.
    let mut intersect: HashMap<(i64, i64), u64> = HashMap::new();
    // Per-segment pixel counts.
    let mut pred_counts: HashMap<i64, u64> = HashMap::new();
    let mut gt_counts: HashMap<i64, u64> = HashMap::new();

    for y in 0..h {
        for x in 0..w {
            let p = pred_map[[y, x]];
            let g = gt_map[[y, x]];
            if p == 0 || g == 0 {
                continue; // void
            }
            *pred_counts.entry(p).or_insert(0) += 1;
            *gt_counts.entry(g).or_insert(0) += 1;
            *intersect.entry((p, g)).or_insert(0) += 1;
        }
    }

    const MATCH_THRESHOLD: f64 = 0.5;
    let mut matches: Vec<MatchedPair> = Vec::new();
    let mut matched_pred: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut matched_gt: std::collections::HashSet<i64> = std::collections::HashSet::new();

    // Build category look-ups for pred and GT segments.
    let pred_cat: HashMap<i64, usize> = pred_segs.iter().map(|s| (s.id as i64, s.category)).collect();
    let gt_cat: HashMap<i64, usize> = gt_segs.iter().map(|s| (s.id as i64, s.category)).collect();

    // Greedy matching by IoU descending. Only consider pairs where both
    // segments belong to the same category AND share the same segment ID.
    let mut candidate_pairs: Vec<((i64, i64), f64)> = intersect
        .iter()
        .filter_map(|(&(p, g), &inter)| {
            if !pred_ids.contains(&p) || !gt_ids.contains(&g) {
                return None;
            }
            // Only match segments with the same ID.
            if p != g {
                return None;
            }
            let pc = *pred_counts.get(&p).unwrap_or(&0) as f64;
            let gc = *gt_counts.get(&g).unwrap_or(&0) as f64;
            let union = pc + gc - inter as f64;
            if union <= 0.0 {
                return None;
            }
            Some(((p, g), inter as f64 / union))
        })
        .collect();
    candidate_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for ((p, g), iou) in candidate_pairs {
        if iou < MATCH_THRESHOLD {
            break;
        }
        if matched_pred.contains(&p) || matched_gt.contains(&g) {
            continue;
        }
        matched_pred.insert(p);
        matched_gt.insert(g);
        matches.push(MatchedPair { iou });
    }
    // Suppress unused variable warnings
    let _ = (&pred_cat, &gt_cat);

    let tp = matches.len() as f64;
    let fp = pred_ids.iter().filter(|id| **id != 0 && !matched_pred.contains(id)).count() as f64;
    let fn_ = gt_ids.iter().filter(|id| **id != 0 && !matched_gt.contains(id)).count() as f64;

    let iou_sum: f64 = matches.iter().map(|m| m.iou).sum();

    let denom = tp + 0.5 * fp + 0.5 * fn_;
    let pq = if denom > 0.0 { iou_sum / denom } else { 1.0 };
    let sq = if tp > 0.0 { iou_sum / tp } else { 1.0 };
    let rq = if denom > 0.0 { tp / denom } else { 1.0 };

    Ok((pq, sq, rq))
}

// ─────────────────────────────────────────────────────────────────────────────
// Merge semantic + instance into panoptic
// ─────────────────────────────────────────────────────────────────────────────

/// Merge a semantic label map and an instance ID map into a panoptic label map.
///
/// Panoptic encoding assigns each pixel a single integer `panoptic_id` that
/// encodes both the semantic category and (for "thing" classes) the instance ID.
///
/// The encoding used here is:
///
/// ```text
/// panoptic_id = category_id * INSTANCE_MULTIPLIER + instance_id
/// ```
///
/// where `INSTANCE_MULTIPLIER = 1000` (consistent with the COCO panoptic format).
///
/// For "stuff" classes all pixels of the same category share the same
/// `panoptic_id` (i.e. `instance_id = 0`).
///
/// # Arguments
///
/// * `semantic_map`  – Semantic label map `[H, W]`, integer category indices.
/// * `instance_map`  – Instance ID map `[H, W]`.  Pixels with `instance_id == 0`
///                     are treated as background / stuff.
/// * `thing_classes` – Set of category IDs that are "thing" classes.
///
/// # Returns
///
/// Panoptic label map `[H, W]` and a `Vec<PanopticSegment>` describing each
/// unique segment found.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when shapes differ.
pub fn merge_semantic_instance(
    semantic_map: &Array2<i32>,
    instance_map: &Array2<i32>,
    thing_classes: &[usize],
) -> Result<(Array2<i64>, Vec<PanopticSegment>)> {
    let (sh, sw) = semantic_map.dim();
    let (ih, iw) = instance_map.dim();
    if sh != ih || sw != iw {
        return Err(VisionError::InvalidParameter(format!(
            "merge_semantic_instance: semantic_map shape {}×{} != instance_map shape {}×{}",
            sh, sw, ih, iw
        )));
    }

    const INSTANCE_MULTIPLIER: i64 = 1000;
    let thing_set: std::collections::HashSet<usize> = thing_classes.iter().cloned().collect();

    let mut panoptic_map = Array2::<i64>::zeros((sh, sw));
    let mut segment_areas: HashMap<i64, usize> = HashMap::new();
    let mut segment_meta: HashMap<i64, (usize, bool)> = HashMap::new();

    for y in 0..sh {
        for x in 0..sw {
            let cat = semantic_map[[y, x]] as usize;
            let inst = instance_map[[y, x]] as i64;
            let is_thing = thing_set.contains(&cat);

            let pan_id = if is_thing && inst > 0 {
                cat as i64 * INSTANCE_MULTIPLIER + inst
            } else {
                // stuff: encode only by category
                cat as i64 * INSTANCE_MULTIPLIER
            };

            panoptic_map[[y, x]] = pan_id;
            *segment_areas.entry(pan_id).or_insert(0) += 1;
            segment_meta.entry(pan_id).or_insert((cat, is_thing));
        }
    }

    let mut segments: Vec<PanopticSegment> = segment_areas
        .iter()
        .map(|(&pan_id, &area)| {
            let (cat, is_thing) = segment_meta.get(&pan_id).cloned().unwrap_or((0, false));
            PanopticSegment::new(pan_id as u64, cat, is_thing, area)
        })
        .collect();
    segments.sort_by_key(|s| s.id);

    Ok((panoptic_map, segments))
}

// ─────────────────────────────────────────────────────────────────────────────
// Instance masks to panoptic label map
// ─────────────────────────────────────────────────────────────────────────────

/// One instance prediction: a binary mask + category + confidence score.
#[derive(Debug, Clone)]
pub struct InstancePrediction {
    /// Binary mask `[H, W]` (non-zero = instance pixel).
    pub mask: Array2<u8>,
    /// Semantic category index.
    pub category: usize,
    /// Confidence / detection score in `[0, 1]`.
    pub score: f32,
    /// `true` if this is a "thing" instance.
    pub is_thing: bool,
}

impl InstancePrediction {
    /// Construct a new `InstancePrediction`.
    pub fn new(mask: Array2<u8>, category: usize, score: f32, is_thing: bool) -> Self {
        Self { mask, category, score, is_thing }
    }
}

/// Convert a list of per-instance binary masks into a panoptic label map.
///
/// The algorithm processes instances in descending score order and uses a
/// "paint" strategy: each instance's pixels that are not yet claimed are
/// assigned to that instance.  The resulting panoptic ID is:
///
/// ```text
/// panoptic_id = category * 1000 + instance_rank
/// ```
///
/// where `instance_rank` is a 1-based counter per category.
///
/// # Arguments
///
/// * `instances` – Instance predictions, may be in any order; sorted by score internally.
/// * `height`    – Output map height.
/// * `width`     – Output map width.
///
/// # Returns
///
/// Panoptic label map `[H, W]` (pixels with no instance stay 0) and a
/// `Vec<PanopticSegment>` describing each placed segment.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when any mask has dimensions
/// different from `(height, width)`.
pub fn instance_to_panoptic(
    instances: &[InstancePrediction],
    height: usize,
    width: usize,
) -> Result<(Array2<i64>, Vec<PanopticSegment>)> {
    // Validate all masks up-front.
    for (i, inst) in instances.iter().enumerate() {
        let (mh, mw) = inst.mask.dim();
        if mh != height || mw != width {
            return Err(VisionError::InvalidParameter(format!(
                "instance_to_panoptic: instance {} mask shape {}×{} != expected {}×{}",
                i, mh, mw, height, width
            )));
        }
    }

    const INSTANCE_MULTIPLIER: i64 = 1000;

    // Sort by descending score (highest confidence first).
    let mut order: Vec<usize> = (0..instances.len()).collect();
    order.sort_by(|&a, &b| {
        instances[b]
            .score
            .partial_cmp(&instances[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut panoptic_map = Array2::<i64>::zeros((height, width));
    let mut segments: Vec<PanopticSegment> = Vec::new();
    let mut cat_counters: HashMap<usize, i64> = HashMap::new();

    for idx in order {
        let inst = &instances[idx];
        let rank = {
            let counter = cat_counters.entry(inst.category).or_insert(0);
            *counter += 1;
            *counter
        };
        let pan_id = inst.category as i64 * INSTANCE_MULTIPLIER + rank;

        let mut area = 0usize;
        for y in 0..height {
            for x in 0..width {
                if inst.mask[[y, x]] != 0 && panoptic_map[[y, x]] == 0 {
                    panoptic_map[[y, x]] = pan_id;
                    area += 1;
                }
            }
        }

        if area > 0 {
            segments.push(PanopticSegment::new(
                pan_id as u64,
                inst.category,
                inst.is_thing,
                area,
            ));
        }
    }

    Ok((panoptic_map, segments))
}

// ─────────────────────────────────────────────────────────────────────────────
// Visualization helper
// ─────────────────────────────────────────────────────────────────────────────

/// Render a panoptic label map into a colour image for visualization.
///
/// Each unique segment ID is assigned a deterministic colour derived from the
/// segment ID itself.  The background (id == 0) is rendered as black.
///
/// # Arguments
///
/// * `panoptic_map` – Panoptic label map `[H, W]`.
///
/// # Returns
///
/// RGB image as `Array3<u8>` with shape `[H, W, 3]`.
pub fn colorize_panoptic(panoptic_map: &Array2<i64>) -> Array3<u8> {
    let (h, w) = panoptic_map.dim();
    let mut out = Array3::<u8>::zeros((h, w, 3));

    for y in 0..h {
        for x in 0..w {
            let id = panoptic_map[[y, x]];
            if id == 0 {
                continue; // background stays black
            }
            // Hash the id into RGB using a simple LCG mix.
            let hashed = id.wrapping_mul(6364136223846793005_i64).wrapping_add(1442695040888963407);
            out[[y, x, 0]] = ((hashed >> 16) & 0xFF) as u8;
            out[[y, x, 1]] = ((hashed >> 8) & 0xFF) as u8;
            out[[y, x, 2]] = (hashed & 0xFF) as u8;
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_map(h: usize, w: usize, val: i64) -> Array2<i64> {
        Array2::from_elem((h, w), val)
    }

    // ── PanopticSegment ───────────────────────────────────────────────────

    #[test]
    fn test_panoptic_segment_construction() {
        let seg = PanopticSegment::new(42, 3, true, 100).with_color((255, 0, 0));
        assert_eq!(seg.id, 42);
        assert_eq!(seg.category, 3);
        assert!(seg.is_thing);
        assert_eq!(seg.area, 100);
        assert_eq!(seg.color, Some((255, 0, 0)));
    }

    // ── panoptic_quality ──────────────────────────────────────────────────

    #[test]
    fn test_pq_perfect_match() {
        // Predicted == GT → PQ = SQ = RQ = 1.
        let map = Array2::from_shape_fn((4, 4), |(y, x)| {
            if x < 2 { 1i64 } else { 2i64 }
        });
        let segs = vec![
            PanopticSegment::new(1, 0, true, 8),
            PanopticSegment::new(2, 1, true, 8),
        ];
        let (pq, sq, rq) = panoptic_quality(&map, &map, &segs, &segs).expect("pq failed");
        assert!((pq - 1.0).abs() < 1e-9, "PQ={}", pq);
        assert!((sq - 1.0).abs() < 1e-9, "SQ={}", sq);
        assert!((rq - 1.0).abs() < 1e-9, "RQ={}", rq);
    }

    #[test]
    fn test_pq_no_overlap() {
        // Pred and GT have completely different IDs → TP=0.
        let pred_map = make_map(4, 4, 1);
        let gt_map = make_map(4, 4, 2);
        let pred_segs = vec![PanopticSegment::new(1, 0, true, 16)];
        let gt_segs = vec![PanopticSegment::new(2, 0, true, 16)];
        let (pq, _sq, rq) = panoptic_quality(&pred_map, &gt_map, &pred_segs, &gt_segs)
            .expect("pq failed");
        assert!((pq).abs() < 1e-9, "PQ should be 0, got {}", pq);
        assert!((rq).abs() < 1e-9, "RQ should be 0, got {}", rq);
    }

    #[test]
    fn test_pq_shape_mismatch() {
        let pred = make_map(4, 4, 1);
        let gt = make_map(5, 4, 1);
        let res = panoptic_quality(&pred, &gt, &[], &[]);
        assert!(res.is_err());
    }

    #[test]
    fn test_pq_empty_segs() {
        // No segments → vacuous PQ=1.
        let pred = Array2::<i64>::zeros((4, 4));
        let gt = Array2::<i64>::zeros((4, 4));
        let (pq, sq, rq) = panoptic_quality(&pred, &gt, &[], &[]).expect("pq failed");
        assert!((pq - 1.0).abs() < 1e-9, "PQ={}", pq);
        let _ = (sq, rq);
    }

    // ── merge_semantic_instance ───────────────────────────────────────────

    #[test]
    fn test_merge_basic() {
        let semantic = Array2::<i32>::from_elem((4, 4), 1); // all category 1
        let instance = Array2::<i32>::from_shape_fn((4, 4), |(_, x)| if x < 2 { 1 } else { 2 });
        let (pan_map, segs) =
            merge_semantic_instance(&semantic, &instance, &[1]).expect("merge failed");
        assert_eq!(pan_map.dim(), (4, 4));
        // Thing class: pan_id for instance 1 = 1*1000 + 1 = 1001
        assert_eq!(pan_map[[0, 0]], 1001);
        assert_eq!(pan_map[[0, 2]], 1002);
        assert!(!segs.is_empty());
    }

    #[test]
    fn test_merge_stuff_class() {
        // Stuff: all pixels same category share same panoptic id regardless of instance.
        let semantic = Array2::<i32>::from_elem((3, 3), 5); // category 5 = stuff
        let instance = Array2::<i32>::from_shape_fn((3, 3), |(y, _)| y as i32 + 1);
        let (pan_map, segs) =
            merge_semantic_instance(&semantic, &instance, &[]).expect("merge stuff failed");
        // All pixels same pan_id = 5*1000 + 0 = 5000.
        for v in pan_map.iter() {
            assert_eq!(*v, 5000, "unexpected stuff pan_id {}", v);
        }
        assert_eq!(segs.len(), 1);
    }

    #[test]
    fn test_merge_shape_mismatch() {
        let semantic = Array2::<i32>::zeros((3, 3));
        let instance = Array2::<i32>::zeros((4, 3));
        let res = merge_semantic_instance(&semantic, &instance, &[]);
        assert!(res.is_err());
    }

    // ── instance_to_panoptic ──────────────────────────────────────────────

    #[test]
    fn test_instance_to_panoptic_basic() {
        let mut m1 = Array2::<u8>::zeros((4, 4));
        m1[[0, 0]] = 1; m1[[0, 1]] = 1;
        let mut m2 = Array2::<u8>::zeros((4, 4));
        m2[[3, 3]] = 1; m2[[3, 2]] = 1;

        let instances = vec![
            InstancePrediction::new(m1, 2, 0.9, true),
            InstancePrediction::new(m2, 2, 0.8, true),
        ];
        let (pan_map, segs) =
            instance_to_panoptic(&instances, 4, 4).expect("i2p failed");
        assert_eq!(pan_map.dim(), (4, 4));
        assert!(pan_map[[0, 0]] != 0);
        assert!(pan_map[[3, 3]] != 0);
        assert!(pan_map[[0, 0]] != pan_map[[3, 3]]);
        assert_eq!(segs.len(), 2);
    }

    #[test]
    fn test_instance_to_panoptic_overlap_resolved() {
        // Two overlapping masks – higher-score one wins overlapping pixels.
        let full_mask = Array2::<u8>::from_elem((4, 4), 1u8);
        let instances = vec![
            InstancePrediction::new(full_mask.clone(), 0, 0.9, true), // higher score
            InstancePrediction::new(full_mask, 0, 0.5, true),          // lower score
        ];
        let (pan_map, segs) = instance_to_panoptic(&instances, 4, 4).expect("i2p failed");
        // Second instance gets 0 area, should not appear in segs.
        assert_eq!(segs.len(), 1, "second instance should have area 0");
        // All pixels claimed by first.
        for v in pan_map.iter() {
            assert_ne!(*v, 0);
        }
    }

    #[test]
    fn test_instance_to_panoptic_mask_size_mismatch() {
        let bad_mask = Array2::<u8>::zeros((3, 3));
        let instances = vec![InstancePrediction::new(bad_mask, 0, 0.9, true)];
        let res = instance_to_panoptic(&instances, 4, 4);
        assert!(res.is_err());
    }

    // ── colorize_panoptic ─────────────────────────────────────────────────

    #[test]
    fn test_colorize_shape() {
        let pan = make_map(6, 8, 1001);
        let color = colorize_panoptic(&pan);
        assert_eq!(color.dim(), (6, 8, 3));
    }

    #[test]
    fn test_colorize_background_black() {
        let pan = Array2::<i64>::zeros((4, 4));
        let color = colorize_panoptic(&pan);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(color[[y, x, 0]], 0);
                assert_eq!(color[[y, x, 1]], 0);
                assert_eq!(color[[y, x, 2]], 0);
            }
        }
    }
}
