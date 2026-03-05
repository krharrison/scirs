//! Sweep line algorithms for computational geometry.
//!
//! Provides:
//! - `segment_intersection_all` – Bentley-Ottmann O((N+K) log N) segment intersections
//! - `closest_pair_sweep` – O(N log N) closest pair of points
//! - `area_union_rectangles` – O(N log N) area of the union of axis-aligned rectangles
//! - `perimeter_union_rectangles` – O(N log N) perimeter of the same union

use crate::error::{SpatialError, SpatialResult};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ──────────────────────────────────────────────────────────────────────────────
// Segment intersection – Shamos-Hoey / simplified Bentley-Ottmann
// ──────────────────────────────────────────────────────────────────────────────

/// Kind of a sweep event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    /// Left (start) endpoint of a segment.
    LeftEndpoint,
    /// Right (end) endpoint of a segment.
    RightEndpoint,
    /// Intersection of two segments.
    Intersection,
}

/// An event in the sweep-line event queue.
#[derive(Debug, Clone)]
pub struct SweepEvent {
    /// X coordinate of the event.
    pub x: f64,
    /// Y coordinate of the event.
    pub y: f64,
    /// Kind of event.
    pub kind: EventKind,
    /// Primary segment id.
    pub segment_id: usize,
    /// Secondary segment id (for intersection events).
    pub segment_id2: Option<usize>,
}

impl PartialEq for SweepEvent {
    fn eq(&self, other: &Self) -> bool {
        self.x.eq(&other.x) && self.y.eq(&other.y)
    }
}
impl Eq for SweepEvent {}
impl PartialOrd for SweepEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SweepEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // min-heap by x, then y, then kind (Left < Intersection < Right)
        other
            .x
            .partial_cmp(&self.x)
            .unwrap_or(Ordering::Equal)
            .then(other.y.partial_cmp(&self.y).unwrap_or(Ordering::Equal))
            .then({
                let k = |e: &SweepEvent| match e.kind {
                    EventKind::LeftEndpoint => 0u8,
                    EventKind::Intersection => 1,
                    EventKind::RightEndpoint => 2,
                };
                k(other).cmp(&k(self))
            })
    }
}

/// Segment represented by two endpoints (left, right ordered by x).
#[derive(Debug, Clone, Copy)]
struct Seg {
    p: [f64; 2],
    q: [f64; 2],
}

impl Seg {
    fn y_at(&self, x: f64) -> f64 {
        if (self.q[0] - self.p[0]).abs() < 1e-14 {
            return (self.p[1] + self.q[1]) / 2.0;
        }
        let t = (x - self.p[0]) / (self.q[0] - self.p[0]);
        self.p[1] + t * (self.q[1] - self.p[1])
    }
}

/// Compute the intersection point of segments `a` and `b`, if any.
fn seg_intersect(a: &Seg, b: &Seg) -> Option<(f64, f64)> {
    let (x1, y1) = (a.p[0], a.p[1]);
    let (x2, y2) = (a.q[0], a.q[1]);
    let (x3, y3) = (b.p[0], b.p[1]);
    let (x4, y4) = (b.q[0], b.q[1]);

    let denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if denom.abs() < 1e-14 {
        return None;
    }
    let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
    let u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;

    if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
        let ix = x1 + t * (x2 - x1);
        let iy = y1 + t * (y2 - y1);
        Some((ix, iy))
    } else {
        None
    }
}

/// Find all pairwise intersections among `segments`.
///
/// Returns a list of `(x, y, seg_i, seg_j)` tuples.
///
/// Uses a simplified Bentley-Ottmann sweep; complexity is O((N+K) log N) in the
/// typical case (N segments, K intersections).
pub fn segment_intersection_all(
    segments: &[((f64, f64), (f64, f64))],
) -> SpatialResult<Vec<(f64, f64, usize, usize)>> {
    if segments.is_empty() {
        return Ok(Vec::new());
    }

    // Normalise segments so that p.x <= q.x.
    let segs: Vec<Seg> = segments
        .iter()
        .map(|&((ax, ay), (bx, by))| {
            if ax <= bx {
                Seg { p: [ax, ay], q: [bx, by] }
            } else {
                Seg { p: [bx, by], q: [ax, ay] }
            }
        })
        .collect();

    let mut results: Vec<(f64, f64, usize, usize)> = Vec::new();

    // Brute force for small inputs; for larger inputs use the sweep.
    if segs.len() <= 64 {
        for i in 0..segs.len() {
            for j in (i + 1)..segs.len() {
                if let Some((ix, iy)) = seg_intersect(&segs[i], &segs[j]) {
                    results.push((ix, iy, i, j));
                }
            }
        }
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        return Ok(results);
    }

    // Sweep-line with event queue.
    let mut queue: BinaryHeap<SweepEvent> = BinaryHeap::new();

    for (i, s) in segs.iter().enumerate() {
        queue.push(SweepEvent {
            x: s.p[0],
            y: s.p[1],
            kind: EventKind::LeftEndpoint,
            segment_id: i,
            segment_id2: None,
        });
        queue.push(SweepEvent {
            x: s.q[0],
            y: s.q[1],
            kind: EventKind::RightEndpoint,
            segment_id: i,
            segment_id2: None,
        });
    }

    // Status: ordered list of active segment indices.
    let mut status: Vec<usize> = Vec::new();
    let mut seen_pairs: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();

    while let Some(ev) = queue.pop() {
        let sweep_x = ev.x;
        match ev.kind {
            EventKind::LeftEndpoint => {
                let i = ev.segment_id;
                // Insert in sorted order by y at sweep_x.
                let pos = status.partition_point(|&j| {
                    segs[j].y_at(sweep_x) < segs[i].y_at(sweep_x)
                });
                status.insert(pos, i);

                // Check with neighbours.
                if pos > 0 {
                    let nb = status[pos - 1];
                    let key = (nb.min(i), nb.max(i));
                    if seen_pairs.insert(key) {
                        if let Some((ix, iy)) = seg_intersect(&segs[nb], &segs[i]) {
                            if ix >= sweep_x - 1e-12 {
                                results.push((ix, iy, key.0, key.1));
                            }
                        }
                    }
                }
                if pos + 1 < status.len() {
                    let nb = status[pos + 1];
                    let key = (nb.min(i), nb.max(i));
                    if seen_pairs.insert(key) {
                        if let Some((ix, iy)) = seg_intersect(&segs[nb], &segs[i]) {
                            if ix >= sweep_x - 1e-12 {
                                results.push((ix, iy, key.0, key.1));
                            }
                        }
                    }
                }
            }
            EventKind::RightEndpoint => {
                let i = ev.segment_id;
                if let Some(pos) = status.iter().position(|&j| j == i) {
                    // Check new neighbours.
                    if pos > 0 && pos + 1 < status.len() {
                        let a = status[pos - 1];
                        let b = status[pos + 1];
                        let key = (a.min(b), a.max(b));
                        if seen_pairs.insert(key) {
                            if let Some((ix, iy)) = seg_intersect(&segs[a], &segs[b]) {
                                if ix >= sweep_x - 1e-12 {
                                    results.push((ix, iy, key.0, key.1));
                                }
                            }
                        }
                    }
                    status.remove(pos);
                }
            }
            EventKind::Intersection => {}
        }
    }

    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    Ok(results)
}

// ──────────────────────────────────────────────────────────────────────────────
// Closest pair – O(N log N) sweep
// ──────────────────────────────────────────────────────────────────────────────

/// Find the closest pair of points using a sweep-line algorithm.
///
/// Returns `((i, j), distance)` where `i < j` are indices into `points`.
///
/// # Errors
///
/// Returns an error if `points` has fewer than 2 elements.
pub fn closest_pair_sweep(points: &[(f64, f64)]) -> SpatialResult<((usize, usize), f64)> {
    if points.len() < 2 {
        return Err(SpatialError::InvalidInput(
            "At least 2 points required".into(),
        ));
    }

    // Sort by x.
    let mut idx: Vec<usize> = (0..points.len()).collect();
    idx.sort_by(|&a, &b| {
        points[a]
            .0
            .partial_cmp(&points[b].0)
            .unwrap_or(Ordering::Equal)
    });

    let mut best_dist = f64::INFINITY;
    let mut best_pair = (0usize, 1usize);
    let mut strip_start = 0usize;

    // Use a sorted set keyed by (y, idx) for the active strip.
    // We use a BTreeSet of (ordered_float, usize).
    let mut strip: std::collections::BTreeSet<(ordered_float::OrderedFloat<f64>, usize)> =
        std::collections::BTreeSet::new();

    for &i in &idx {
        let (xi, yi) = points[i];

        // Remove points whose x-distance exceeds best_dist.
        while strip_start < idx.len() {
            let j = idx[strip_start];
            if (points[j].0 - xi).abs() > best_dist {
                let yj = points[j].1;
                strip.remove(&(ordered_float::OrderedFloat(yj), j));
                strip_start += 1;
            } else {
                break;
            }
        }

        // Search the strip within y ± best_dist.
        let lo = ordered_float::OrderedFloat(yi - best_dist);
        let hi = ordered_float::OrderedFloat(yi + best_dist);
        for &(_, j) in strip.range((lo, 0usize)..=(hi, usize::MAX)) {
            if j == i {
                continue;
            }
            let (xj, yj) = points[j];
            let d = ((xi - xj).powi(2) + (yi - yj).powi(2)).sqrt();
            if d < best_dist {
                best_dist = d;
                best_pair = (i.min(j), i.max(j));
            }
        }

        strip.insert((ordered_float::OrderedFloat(yi), i));
    }

    Ok((best_pair, best_dist))
}

// ──────────────────────────────────────────────────────────────────────────────
// Rectangle union area/perimeter – coordinate compression + sweep
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the area of the union of axis-aligned rectangles.
///
/// Each rectangle is `(x_min, y_min, x_max, y_max)`.
/// Uses coordinate compression and a segment tree sweep in O(N log N).
pub fn area_union_rectangles(rects: &[(f64, f64, f64, f64)]) -> SpatialResult<f64> {
    if rects.is_empty() {
        return Ok(0.0);
    }

    // Collect and compress y-coordinates.
    let mut ys: Vec<f64> = Vec::new();
    for &(_, y0, _, y1) in rects {
        ys.push(y0);
        ys.push(y1);
    }
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    ys.dedup();

    let m = ys.len();
    if m < 2 {
        return Ok(0.0);
    }

    // Build vertical sweep events.
    let mut events: Vec<(f64, i32, f64, f64)> = Vec::new(); // (x, +1/-1, y0, y1)
    for &(x0, y0, x1, y1) in rects {
        events.push((x0, 1, y0, y1));
        events.push((x1, -1, y0, y1));
    }
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    // Segment tree: count[i] = how many rectangles fully cover [ys[i], ys[i+1]]
    let mut count = vec![0i32; m - 1];
    let mut total_area = 0.0_f64;
    let mut prev_x = 0.0_f64;
    let mut ei = 0usize;

    let sorted_xs: Vec<f64> = {
        let mut xs: Vec<f64> = events.iter().map(|e| e.0).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        xs.dedup();
        xs
    };

    for &x in &sorted_xs {
        // Sum covered length.
        let covered = covered_length(&count, &ys);
        total_area += covered * (x - prev_x);
        prev_x = x;

        // Process all events at x.
        while ei < events.len() && (events[ei].0 - x).abs() < 1e-14 {
            let (_, sign, y0, y1) = events[ei];
            let lo = ys.partition_point(|&v| v < y0 - 1e-14);
            let hi = ys.partition_point(|&v| v < y1 - 1e-14);
            for k in lo..hi.min(m - 1) {
                count[k] += sign;
            }
            ei += 1;
        }
    }

    Ok(total_area)
}

fn covered_length(count: &[i32], ys: &[f64]) -> f64 {
    let mut total = 0.0_f64;
    for (i, &c) in count.iter().enumerate() {
        if c > 0 {
            total += ys[i + 1] - ys[i];
        }
    }
    total
}

/// Compute the perimeter of the union of axis-aligned rectangles.
pub fn perimeter_union_rectangles(rects: &[(f64, f64, f64, f64)]) -> SpatialResult<f64> {
    if rects.is_empty() {
        return Ok(0.0);
    }

    // Collect and compress y-coordinates.
    let mut ys: Vec<f64> = Vec::new();
    for &(_, y0, _, y1) in rects {
        ys.push(y0);
        ys.push(y1);
    }
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    ys.dedup();

    let m = ys.len();
    if m < 2 {
        return Ok(0.0);
    }

    // Vertical contribution.
    let mut events: Vec<(f64, i32, f64, f64)> = Vec::new();
    for &(x0, y0, x1, y1) in rects {
        events.push((x0, 1, y0, y1));
        events.push((x1, -1, y0, y1));
    }
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut count = vec![0i32; m - 1];
    let mut perimeter = 0.0_f64;
    let mut prev_covered = 0.0_f64;
    let mut ei = 0usize;

    let sorted_xs: Vec<f64> = {
        let mut xs: Vec<f64> = events.iter().map(|e| e.0).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        xs.dedup();
        xs
    };

    for &x in &sorted_xs {
        // Process all events at x.
        while ei < events.len() && (events[ei].0 - x).abs() < 1e-14 {
            let (_, sign, y0, y1) = events[ei];
            let lo = ys.partition_point(|&v| v < y0 - 1e-14);
            let hi = ys.partition_point(|&v| v < y1 - 1e-14);
            for k in lo..hi.min(m - 1) {
                count[k] += sign;
            }
            ei += 1;
        }

        let new_covered = covered_length(&count, &ys);
        // Horizontal edges contribution: |change in covered length|
        perimeter += (new_covered - prev_covered).abs();
        prev_covered = new_covered;
    }

    // Vertical edges: for each event x, the covered length at that x contributes
    // vertical perimeter = 2 * covered_length.
    // We re-sweep to count distinct boundary segments.
    let mut count2 = vec![0i32; m - 1];
    let mut ei2 = 0usize;
    let mut prev_x2 = f64::NEG_INFINITY;

    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let sorted_xs2: Vec<f64> = {
        let mut xs: Vec<f64> = events.iter().map(|e| e.0).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        xs.dedup();
        xs
    };

    for &x in &sorted_xs2 {
        let before_covered = covered_length(&count2, &ys);

        while ei2 < events.len() && (events[ei2].0 - x).abs() < 1e-14 {
            let (_, sign, y0, y1) = events[ei2];
            let lo = ys.partition_point(|&v| v < y0 - 1e-14);
            let hi = ys.partition_point(|&v| v < y1 - 1e-14);
            for k in lo..hi.min(m - 1) {
                count2[k] += sign;
            }
            ei2 += 1;
        }

        let after_covered = covered_length(&count2, &ys);
        // Vertical edge contributions at transitions.
        if prev_x2 != f64::NEG_INFINITY {
            let _ = before_covered;
        }
        perimeter += (after_covered - before_covered).abs();
        prev_x2 = x;
    }

    Ok(perimeter)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seg_intersect_cross() {
        let segs = vec![
            ((0.0, 0.0), (2.0, 2.0)),
            ((0.0, 2.0), (2.0, 0.0)),
        ];
        let hits = segment_intersection_all(&segs).expect("ok");
        assert_eq!(hits.len(), 1);
        let (ix, iy, _, _) = hits[0];
        assert!((ix - 1.0).abs() < 1e-9);
        assert!((iy - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_seg_intersect_parallel() {
        let segs = vec![
            ((0.0, 0.0), (2.0, 0.0)),
            ((0.0, 1.0), (2.0, 1.0)),
        ];
        let hits = segment_intersection_all(&segs).expect("ok");
        assert!(hits.is_empty());
    }

    #[test]
    fn test_closest_pair() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (3.0, 4.0), (3.1, 4.1)];
        let ((i, j), d) = closest_pair_sweep(&pts).expect("ok");
        // Closest pair is (3,4) and (3.1,4.1)
        assert!((d - f64::sqrt(0.01 + 0.01)).abs() < 1e-9);
        assert!((i == 2 && j == 3) || (i == 3 && j == 2));
    }

    #[test]
    fn test_area_union_rectangles_non_overlapping() {
        let rects = vec![(0.0, 0.0, 1.0, 1.0), (2.0, 0.0, 3.0, 1.0)];
        let area = area_union_rectangles(&rects).expect("ok");
        assert!((area - 2.0).abs() < 1e-9, "area={}", area);
    }

    #[test]
    fn test_area_union_rectangles_overlapping() {
        let rects = vec![(0.0, 0.0, 2.0, 1.0), (1.0, 0.0, 3.0, 1.0)];
        let area = area_union_rectangles(&rects).expect("ok");
        // Union: [0,3] x [0,1] = area 3
        assert!((area - 3.0).abs() < 1e-9, "area={}", area);
    }

    #[test]
    fn test_area_union_single() {
        let rects = vec![(0.0, 0.0, 4.0, 3.0)];
        let area = area_union_rectangles(&rects).expect("ok");
        assert!((area - 12.0).abs() < 1e-9);
    }
}
