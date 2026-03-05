//! Instance segmentation algorithms
//!
//! This module provides instance-level segmentation functionality including:
//! - Watershed-based instance separation
//! - Mask IoU and Non-Maximum Suppression
//! - Panoptic Quality metric
//! - Instance overlap utilities

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::Array2;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

// ---------------------------------------------------------------------------
// InstanceMask
// ---------------------------------------------------------------------------

/// A single instance produced by an instance segmentation model.
#[derive(Debug, Clone)]
pub struct InstanceMask {
    /// Class identifier (0-indexed)
    pub class_id: usize,
    /// Detection confidence / quality score in [0, 1]
    pub score: f64,
    /// Binary pixel mask (`true` = object foreground)
    pub mask: Array2<bool>,
    /// Tight axis-aligned bounding box `[y_min, x_min, y_max, x_max]`
    pub bbox: [usize; 4],
}

impl InstanceMask {
    /// Construct a new [`InstanceMask`] with an automatically computed bounding box.
    ///
    /// If the mask is all-false the bbox is set to `[0, 0, 0, 0]`.
    pub fn new(class_id: usize, score: f64, mask: Array2<bool>) -> Self {
        let bbox = compute_bbox(&mask);
        Self {
            class_id,
            score,
            mask,
            bbox,
        }
    }

    /// Return the number of foreground pixels.
    pub fn area(&self) -> usize {
        self.mask.iter().filter(|&&v| v).count()
    }
}

/// Compute a tight bounding box from a binary mask.
///
/// Returns `[y_min, x_min, y_max, x_max]`.  Returns `[0,0,0,0]` for empty masks.
fn compute_bbox(mask: &Array2<bool>) -> [usize; 4] {
    let (height, width) = mask.dim();
    let mut y_min = height;
    let mut y_max = 0usize;
    let mut x_min = width;
    let mut x_max = 0usize;
    let mut found = false;

    for y in 0..height {
        for x in 0..width {
            if mask[[y, x]] {
                found = true;
                if y < y_min {
                    y_min = y;
                }
                if y > y_max {
                    y_max = y;
                }
                if x < x_min {
                    x_min = x;
                }
                if x > x_max {
                    x_max = x;
                }
            }
        }
    }

    if found {
        [y_min, x_min, y_max, x_max]
    } else {
        [0, 0, 0, 0]
    }
}

// ---------------------------------------------------------------------------
// Watershed instance segmentation
// ---------------------------------------------------------------------------

/// Priority queue entry for the watershed flooding (min-heap by gradient value).
#[derive(PartialEq)]
struct WatershedEntry {
    /// Negated gradient so that `BinaryHeap` (max-heap) acts as a min-heap.
    neg_gradient: ordered_float::NotNan<f64>,
    y: usize,
    x: usize,
}

impl Eq for WatershedEntry {}

impl PartialOrd for WatershedEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WatershedEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // max-heap on neg_gradient → min-heap on actual gradient
        self.neg_gradient.cmp(&other.neg_gradient)
    }
}

/// Marker-controlled watershed segmentation.
///
/// Floods a gradient image starting from pre-placed markers.  Each marker
/// expands into its catchment basin.  Pixels labelled `0` in `markers` are
/// uninitialized and will be flooded; non-zero pixels are seeds.
/// The returned label map uses the same non-zero label values as `markers`;
/// pixels where no basin reached them retain label `0`.
///
/// # Arguments
/// * `gradient` - Gradient magnitude image `[height, width]` (higher = boundary)
/// * `markers`  - Initial marker map; `0` = unlabelled, positive = seed label
pub fn watershed_instance(gradient: &Array2<f64>, markers: &Array2<i32>) -> Result<Array2<i32>> {
    let (height, width) = gradient.dim();
    let (mh, mw) = markers.dim();
    if height != mh || width != mw {
        return Err(VisionError::DimensionMismatch(format!(
            "gradient ({height}×{width}) and markers ({mh}×{mw}) must have the same shape"
        )));
    }
    if height == 0 || width == 0 {
        return Err(VisionError::InvalidParameter(
            "gradient must be non-empty".to_string(),
        ));
    }

    let mut output = markers.to_owned();
    let mut in_queue = Array2::<bool>::from_elem((height, width), false);

    let mut heap: BinaryHeap<WatershedEntry> = BinaryHeap::new();

    // Seed the heap with all pixels adjacent to markers
    for y in 0..height {
        for x in 0..width {
            if markers[[y, x]] == 0 {
                continue;
            }
            let neighbours: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
            for (dy, dx) in neighbours {
                let ny = y as i64 + dy;
                let nx = x as i64 + dx;
                if ny < 0 || ny >= height as i64 || nx < 0 || nx >= width as i64 {
                    continue;
                }
                let ny = ny as usize;
                let nx = nx as usize;
                if output[[ny, nx]] == 0 && !in_queue[[ny, nx]] {
                    in_queue[[ny, nx]] = true;
                    let neg = ordered_float::NotNan::new(-gradient[[ny, nx]])
                        .unwrap_or_else(|_| ordered_float::NotNan::default());
                    heap.push(WatershedEntry {
                        neg_gradient: neg,
                        y: ny,
                        x: nx,
                    });
                }
            }
        }
    }

    // Flood filling
    while let Some(entry) = heap.pop() {
        let y = entry.y;
        let x = entry.x;

        // Assign label from the neighbouring marker with the lowest boundary cost
        let mut best_label = 0i32;
        let mut best_grad = f64::INFINITY;

        let neighbours: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        for (dy, dx) in neighbours {
            let ny = y as i64 + dy;
            let nx = x as i64 + dx;
            if ny < 0 || ny >= height as i64 || nx < 0 || nx >= width as i64 {
                continue;
            }
            let ny = ny as usize;
            let nx = nx as usize;
            let nb_label = output[[ny, nx]];
            if nb_label != 0 {
                // Use the neighbour's gradient as the "cost" of propagating through it
                let cost = gradient[[ny, nx]];
                if cost < best_grad {
                    best_grad = cost;
                    best_label = nb_label;
                }
            }
        }

        if best_label != 0 {
            output[[y, x]] = best_label;

            // Expand into unlabelled neighbours
            for (dy, dx) in neighbours {
                let ny = y as i64 + dy;
                let nx = x as i64 + dx;
                if ny < 0 || ny >= height as i64 || nx < 0 || nx >= width as i64 {
                    continue;
                }
                let ny = ny as usize;
                let nx = nx as usize;
                if output[[ny, nx]] == 0 && !in_queue[[ny, nx]] {
                    in_queue[[ny, nx]] = true;
                    let neg = ordered_float::NotNan::new(-gradient[[ny, nx]])
                        .unwrap_or_else(|_| ordered_float::NotNan::default());
                    heap.push(WatershedEntry {
                        neg_gradient: neg,
                        y: ny,
                        x: nx,
                    });
                }
            }
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Mask IoU
// ---------------------------------------------------------------------------

/// Compute intersection-over-union between two binary masks.
///
/// Returns 0.0 when both masks are empty (zero union).
pub fn mask_iou(mask1: &Array2<bool>, mask2: &Array2<bool>) -> Result<f64> {
    let (h1, w1) = mask1.dim();
    let (h2, w2) = mask2.dim();
    if h1 != h2 || w1 != w2 {
        return Err(VisionError::DimensionMismatch(format!(
            "mask1 ({h1}×{w1}) and mask2 ({h2}×{w2}) must have the same shape"
        )));
    }

    let mut intersection = 0usize;
    let mut union_ = 0usize;

    for y in 0..h1 {
        for x in 0..w1 {
            let a = mask1[[y, x]];
            let b = mask2[[y, x]];
            if a && b {
                intersection += 1;
            }
            if a || b {
                union_ += 1;
            }
        }
    }

    if union_ == 0 {
        Ok(0.0)
    } else {
        Ok(intersection as f64 / union_ as f64)
    }
}

// ---------------------------------------------------------------------------
// Mask NMS
// ---------------------------------------------------------------------------

/// Non-maximum suppression on instance masks using mask IoU.
///
/// Instances are sorted by descending score; any instance whose mask IoU with
/// an already-selected instance exceeds `iou_threshold` is suppressed.
///
/// # Arguments
/// * `instances`     - Candidate instance masks
/// * `iou_threshold` - IoU threshold above which the lower-scored instance is suppressed
pub fn mask_nms(instances: &[InstanceMask], iou_threshold: f64) -> Result<Vec<InstanceMask>> {
    if instances.is_empty() {
        return Ok(Vec::new());
    }

    // Sort indices by descending score
    let mut indices: Vec<usize> = (0..instances.len()).collect();
    indices.sort_by(|&a, &b| {
        instances[b]
            .score
            .partial_cmp(&instances[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<InstanceMask> = Vec::new();

    'outer: for &idx in &indices {
        let candidate = &instances[idx];
        for already_kept in &kept {
            // Only compare within the same class (optional convention, matches most frameworks)
            if already_kept.class_id != candidate.class_id {
                continue;
            }
            let iou = mask_iou(&candidate.mask, &already_kept.mask)?;
            if iou > iou_threshold {
                continue 'outer;
            }
        }
        kept.push(candidate.clone());
    }

    Ok(kept)
}

// ---------------------------------------------------------------------------
// Panoptic Quality
// ---------------------------------------------------------------------------

/// Compute Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition
/// Quality (RQ) for a single semantic class.
///
/// The formulae are from Kirillov et al., "Panoptic Segmentation", CVPR 2019:
///
/// ```text
/// PQ = SQ × RQ
/// SQ = Σ_{(p,g)∈TP} IoU(p,g) / |TP|
/// RQ = |TP| / (|TP| + ½|FP| + ½|FN|)
/// ```
///
/// A predicted instance `p` is matched to ground-truth `g` if their mask IoU
/// exceeds 0.5 (the standard threshold).
///
/// # Arguments
/// * `predicted`    - Predicted instance masks (all same class)
/// * `ground_truth` - Ground-truth instance masks (all same class)
///
/// # Returns
/// `(pq, sq, rq)` tuple.
pub fn panoptic_quality(
    predicted: &[InstanceMask],
    ground_truth: &[InstanceMask],
) -> Result<(f64, f64, f64)> {
    let iou_threshold = 0.5f64;

    // Build a cost matrix: IoU between every predicted × gt pair
    let n_pred = predicted.len();
    let n_gt = ground_truth.len();

    if n_pred == 0 && n_gt == 0 {
        // Nothing to evaluate; PQ = 1 by convention (perfect vacuous case)
        return Ok((1.0, 1.0, 1.0));
    }

    // Greedy matching: sort pairs by descending IoU, greedily assign
    let mut iou_pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (pi, pred_inst) in predicted.iter().enumerate().take(n_pred) {
        for (gi, gt_inst) in ground_truth.iter().enumerate().take(n_gt) {
            // Only compare spatially compatible pairs (same dimensions)
            let (ph, pw) = pred_inst.mask.dim();
            let (gh, gw) = gt_inst.mask.dim();
            if ph != gh || pw != gw {
                continue;
            }
            let iou = mask_iou(&pred_inst.mask, &gt_inst.mask)?;
            if iou > iou_threshold {
                iou_pairs.push((iou, pi, gi));
            }
        }
    }

    // Sort descending by IoU
    iou_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut matched_pred: HashSet<usize> = HashSet::new();
    let mut matched_gt: HashSet<usize> = HashSet::new();
    let mut tp_iou_sum = 0.0f64;
    let mut tp_count = 0usize;

    for (iou, pi, gi) in &iou_pairs {
        if matched_pred.contains(pi) || matched_gt.contains(gi) {
            continue;
        }
        matched_pred.insert(*pi);
        matched_gt.insert(*gi);
        tp_iou_sum += iou;
        tp_count += 1;
    }

    let fp = n_pred - matched_pred.len();
    let fn_ = n_gt - matched_gt.len();

    let tp_f = tp_count as f64;
    let fp_f = fp as f64;
    let fn_f = fn_ as f64;

    let sq = if tp_count > 0 { tp_iou_sum / tp_f } else { 0.0 };

    let denom = tp_f + 0.5 * fp_f + 0.5 * fn_f;
    let rq = if denom > 0.0 { tp_f / denom } else { 0.0 };
    let pq = sq * rq;

    Ok((pq, sq, rq))
}

// ---------------------------------------------------------------------------
// Instance overlap
// ---------------------------------------------------------------------------

/// Check whether two instance masks overlap (share at least one foreground pixel).
///
/// Returns an error if the masks have different spatial dimensions.
pub fn instance_overlap(inst1: &InstanceMask, inst2: &InstanceMask) -> Result<bool> {
    let (h1, w1) = inst1.mask.dim();
    let (h2, w2) = inst2.mask.dim();
    if h1 != h2 || w1 != w2 {
        return Err(VisionError::DimensionMismatch(format!(
            "inst1 mask ({h1}×{w1}) and inst2 mask ({h2}×{w2}) must have the same shape"
        )));
    }
    for y in 0..h1 {
        for x in 0..w1 {
            if inst1.mask[[y, x]] && inst2.mask[[y, x]] {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

// ---------------------------------------------------------------------------
// Utility: build InstanceMask from a label map
// ---------------------------------------------------------------------------

/// Convert a dense label map (e.g. watershed output) to a vector of [`InstanceMask`]s.
///
/// Label `0` is treated as background and ignored.
/// All instances are assigned `class_id = 0` and `score = 1.0` (modify as needed).
pub fn label_map_to_instances(label_map: &Array2<i32>) -> Result<Vec<InstanceMask>> {
    let (height, width) = label_map.dim();
    let mut label_set: HashMap<i32, Vec<(usize, usize)>> = HashMap::new();

    for y in 0..height {
        for x in 0..width {
            let lbl = label_map[[y, x]];
            if lbl == 0 {
                continue;
            }
            label_set.entry(lbl).or_default().push((y, x));
        }
    }

    let mut instances: Vec<InstanceMask> = Vec::new();
    for (_, pixels) in label_set {
        let mut mask = Array2::<bool>::from_elem((height, width), false);
        for (y, x) in pixels {
            mask[[y, x]] = true;
        }
        instances.push(InstanceMask::new(0, 1.0, mask));
    }

    // Sort by area descending for deterministic ordering
    instances.sort_by_key(|inst| Reverse(inst.area()));

    Ok(instances)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array2, Array3};

    fn make_mask(height: usize, width: usize, pixels: &[(usize, usize)]) -> Array2<bool> {
        let mut m = Array2::<bool>::from_elem((height, width), false);
        for &(y, x) in pixels {
            m[[y, x]] = true;
        }
        m
    }

    #[test]
    fn test_mask_iou_identical() {
        let m = make_mask(4, 4, &[(0, 0), (0, 1), (1, 0)]);
        let iou = mask_iou(&m, &m).expect("mask_iou should succeed");
        assert!((iou - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mask_iou_disjoint() {
        let m1 = make_mask(4, 4, &[(0, 0)]);
        let m2 = make_mask(4, 4, &[(3, 3)]);
        let iou = mask_iou(&m1, &m2).expect("mask_iou should succeed");
        assert!((iou - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_mask_nms_removes_overlap() {
        let m1 = make_mask(4, 4, &[(0, 0), (0, 1), (1, 0), (1, 1)]);
        let m2 = make_mask(4, 4, &[(0, 0), (0, 1), (1, 0)]);
        let instances = vec![
            InstanceMask::new(0, 0.9, m1.clone()),
            InstanceMask::new(0, 0.7, m2.clone()),
        ];
        let kept = mask_nms(&instances, 0.5).expect("mask_nms should succeed");
        // IoU(m1, m2) = 3/4 = 0.75 > 0.5, so lower-score m2 should be suppressed
        assert_eq!(kept.len(), 1);
        assert!((kept[0].score - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_mask_nms_keeps_disjoint() {
        let m1 = make_mask(4, 4, &[(0, 0)]);
        let m2 = make_mask(4, 4, &[(3, 3)]);
        let instances = vec![InstanceMask::new(0, 0.9, m1), InstanceMask::new(0, 0.8, m2)];
        let kept = mask_nms(&instances, 0.5).expect("mask_nms should succeed");
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_panoptic_quality_perfect() {
        let m = make_mask(4, 4, &[(0, 0), (0, 1)]);
        let pred = vec![InstanceMask::new(0, 1.0, m.clone())];
        let gt = vec![InstanceMask::new(0, 1.0, m)];
        let (pq, sq, rq) = panoptic_quality(&pred, &gt).expect("panoptic_quality should succeed");
        assert!((pq - 1.0).abs() < 1e-10);
        assert!((sq - 1.0).abs() < 1e-10);
        assert!((rq - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_panoptic_quality_empty() {
        let (pq, sq, rq) = panoptic_quality(&[], &[]).expect("panoptic_quality should succeed");
        // Vacuous perfect case
        assert!((pq - 1.0).abs() < 1e-10);
        let _ = (sq, rq);
    }

    #[test]
    fn test_instance_overlap_true() {
        let m1 = make_mask(4, 4, &[(1, 1), (2, 2)]);
        let m2 = make_mask(4, 4, &[(2, 2), (3, 3)]);
        let i1 = InstanceMask::new(0, 1.0, m1);
        let i2 = InstanceMask::new(0, 1.0, m2);
        assert!(instance_overlap(&i1, &i2).expect("should succeed"));
    }

    #[test]
    fn test_instance_overlap_false() {
        let m1 = make_mask(4, 4, &[(0, 0)]);
        let m2 = make_mask(4, 4, &[(3, 3)]);
        let i1 = InstanceMask::new(0, 1.0, m1);
        let i2 = InstanceMask::new(0, 1.0, m2);
        assert!(!instance_overlap(&i1, &i2).expect("should succeed"));
    }

    #[test]
    fn test_watershed_instance_basic() {
        let mut gradient = Array2::<f64>::zeros((5, 5));
        // High-gradient boundary down the middle
        for y in 0..5 {
            gradient[[y, 2]] = 10.0;
        }
        let mut markers = Array2::<i32>::zeros((5, 5));
        markers[[2, 0]] = 1;
        markers[[2, 4]] = 2;
        let labels = watershed_instance(&gradient, &markers).expect("watershed should succeed");
        assert_eq!(labels.dim(), (5, 5));
        // Left seed should dominate left side
        assert_eq!(labels[[2, 0]], 1);
        // Right seed should dominate right side
        assert_eq!(labels[[2, 4]], 2);
    }

    #[test]
    fn test_label_map_to_instances() {
        let mut lmap = Array2::<i32>::zeros((4, 4));
        lmap[[0, 0]] = 1;
        lmap[[0, 1]] = 1;
        lmap[[3, 3]] = 2;
        let instances =
            label_map_to_instances(&lmap).expect("label_map_to_instances should succeed");
        assert_eq!(instances.len(), 2);
    }
}
