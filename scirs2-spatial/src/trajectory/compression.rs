//! Trajectory compression and simplification algorithms
//!
//! - **Douglas–Peucker** (Ramer, 1972 / Douglas & Peucker, 1973): recursive
//!   perpendicular-distance simplification.
//! - **Visvalingam–Whyatt** (1993): iterative area-based vertex elimination.
//! - **Dead Reckoning**: forward-prediction compression with a configurable
//!   maximum error threshold.
//! - **Online Douglas–Peucker**: streaming variant suitable for live GPS feeds.
//! - Utility: [`compression_ratio`] to measure simplification quality.

use crate::error::{SpatialError, SpatialResult};
use crate::trajectory::similarity::Point2D;
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Perpendicular distance from point `p` to the line through `a` and `b`.
///
/// Returns 0.0 when `a == b`.
fn perp_dist(p: &Point2D, a: &Point2D, b: &Point2D) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len_sq = dx * dx + dy * dy;
    if len_sq < f64::EPSILON {
        // Degenerate segment; return point-to-point distance.
        let ex = p[0] - a[0];
        let ey = p[1] - a[1];
        return (ex * ex + ey * ey).sqrt();
    }
    // |cross product| / |segment|
    let cross = (p[0] - a[0]) * dy - (p[1] - a[1]) * dx;
    cross.abs() / len_sq.sqrt()
}

/// Triangle area formed by three consecutive points (absolute value).
fn triangle_area(a: &Point2D, b: &Point2D, c: &Point2D) -> f64 {
    let area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
    area.abs() / 2.0
}

// ---------------------------------------------------------------------------
// Ramer–Douglas–Peucker
// ---------------------------------------------------------------------------

/// Simplify a trajectory using the Ramer–Douglas–Peucker algorithm.
///
/// Points whose perpendicular distance to the current chord is less than
/// `tolerance` are discarded.  Endpoints are always preserved.
///
/// # Errors
///
/// Returns [`SpatialError::ValueError`] if `tolerance` is negative.
pub fn douglas_peucker(points: &[Point2D], tolerance: f64) -> SpatialResult<Vec<Point2D>> {
    if points.len() <= 2 {
        return Ok(points.to_vec());
    }
    if tolerance < 0.0 {
        return Err(SpatialError::ValueError(
            "Douglas–Peucker: tolerance must be non-negative".to_string(),
        ));
    }

    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;

    rdp_recursive(points, 0, points.len() - 1, tolerance, &mut keep);

    Ok(points
        .iter()
        .zip(keep.iter())
        .filter_map(|(p, &k)| if k { Some(*p) } else { None })
        .collect())
}

fn rdp_recursive(
    points: &[Point2D],
    start: usize,
    end: usize,
    tolerance: f64,
    keep: &mut Vec<bool>,
) {
    if end <= start + 1 {
        return;
    }

    let mut max_dist = 0.0_f64;
    let mut max_idx = start;

    for i in (start + 1)..end {
        let d = perp_dist(&points[i], &points[start], &points[end]);
        if d > max_dist {
            max_dist = d;
            max_idx = i;
        }
    }

    if max_dist > tolerance {
        keep[max_idx] = true;
        rdp_recursive(points, start, max_idx, tolerance, keep);
        rdp_recursive(points, max_idx, end, tolerance, keep);
    }
}

// ---------------------------------------------------------------------------
// Visvalingam–Whyatt
// ---------------------------------------------------------------------------

/// Ordered entry for the min-heap: (area, index).
/// We negate the area so that `BinaryHeap` (a max-heap) gives the minimum.
#[derive(PartialEq)]
struct VwEntry {
    neg_area: ordered_float::OrderedFloat<f64>,
    idx: usize,
}

impl Eq for VwEntry {}

impl PartialOrd for VwEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VwEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.neg_area.cmp(&other.neg_area)
    }
}

// We manually implement the heap using indices to track live/dead status.

/// Simplify a trajectory using the Visvalingam–Whyatt algorithm.
///
/// Repeatedly removes the point that forms the triangle with the smallest
/// effective area with its two neighbours, until no triangle area is below
/// `min_area` or the target point count is reached.
///
/// # Arguments
///
/// * `points`    – Input trajectory.
/// * `min_area`  – Triangles with area ≤ `min_area` are eligible for removal.
/// * `max_points`– Optional upper limit on the number of output points.
///
/// # Errors
///
/// Returns [`SpatialError::ValueError`] if `min_area` is negative.
pub fn visvalingam_whyatt(
    points: &[Point2D],
    min_area: f64,
    max_points: Option<usize>,
) -> SpatialResult<Vec<Point2D>> {
    if points.len() <= 2 {
        return Ok(points.to_vec());
    }
    if min_area < 0.0 {
        return Err(SpatialError::ValueError(
            "Visvalingam–Whyatt: min_area must be non-negative".to_string(),
        ));
    }

    let n = points.len();
    let target = max_points.unwrap_or(2).max(2);

    // Doubly-linked list over indices to support O(1) removal.
    let mut prev: Vec<usize> = (0..n).map(|i| if i == 0 { 0 } else { i - 1 }).collect();
    let mut next: Vec<usize> = (0..n)
        .map(|i| if i == n - 1 { n - 1 } else { i + 1 })
        .collect();
    let mut alive = vec![true; n];

    // Pre-compute areas.
    let mut areas = vec![f64::INFINITY; n];
    for i in 1..n - 1 {
        areas[i] = triangle_area(&points[prev[i]], &points[i], &points[next[i]]);
    }

    // Simple O(n log n) simulation using a heap.
    // The heap may contain stale entries; we skip them when their area differs.
    let mut heap = BinaryHeap::new();
    for i in 1..n - 1 {
        heap.push(VwEntry {
            neg_area: ordered_float::OrderedFloat(-areas[i]),
            idx: i,
        });
    }

    let mut live_count = n;

    while live_count > target {
        // Pop the entry with the smallest area.
        let entry = match heap.pop() {
            Some(e) => e,
            None => break,
        };
        let i = entry.idx;
        // Stale check: the stored area no longer matches the current area.
        let stored_area = -entry.neg_area.into_inner();
        if !alive[i] || (stored_area - areas[i]).abs() > f64::EPSILON * 1e3 {
            continue;
        }
        // Only remove if area ≤ min_area.
        if areas[i] > min_area {
            break;
        }

        // Remove point i.
        alive[i] = false;
        live_count -= 1;
        let p = prev[i];
        let nx = next[i];
        next[p] = nx;
        prev[nx] = p;

        // Recompute area for the neighbours.
        if p > 0 {
            let pp = prev[p];
            let np = next[p];
            let new_area = triangle_area(&points[pp], &points[p], &points[np]);
            // Enforce monotone increase (Visvalingam's "max" rule).
            areas[p] = new_area.max(areas[i]);
            heap.push(VwEntry {
                neg_area: ordered_float::OrderedFloat(-areas[p]),
                idx: p,
            });
        }
        if nx < n - 1 {
            let pn = prev[nx];
            let nn = next[nx];
            let new_area = triangle_area(&points[pn], &points[nx], &points[nn]);
            areas[nx] = new_area.max(areas[i]);
            heap.push(VwEntry {
                neg_area: ordered_float::OrderedFloat(-areas[nx]),
                idx: nx,
            });
        }
    }

    Ok(points
        .iter()
        .enumerate()
        .filter_map(|(i, p)| if alive[i] { Some(*p) } else { None })
        .collect())
}

// ---------------------------------------------------------------------------
// Dead Reckoning
// ---------------------------------------------------------------------------

/// Compress a trajectory using the Dead Reckoning algorithm.
///
/// Each new point is predicted by linear extrapolation from the last two
/// *recorded* points.  A point is recorded only when the true position
/// deviates from the prediction by more than `max_error`.
///
/// The first two points are always kept as the algorithm needs an initial
/// direction vector.
///
/// # Errors
///
/// Returns [`SpatialError::ValueError`] if `max_error` is negative.
pub fn dead_reckoning(points: &[Point2D], max_error: f64) -> SpatialResult<Vec<Point2D>> {
    if points.len() <= 2 {
        return Ok(points.to_vec());
    }
    if max_error < 0.0 {
        return Err(SpatialError::ValueError(
            "Dead Reckoning: max_error must be non-negative".to_string(),
        ));
    }

    let mut result = vec![points[0], points[1]];
    // Last two *recorded* points used for linear prediction.
    let mut p_prev = points[0];
    let mut p_last = points[1];

    for &p in &points[2..] {
        // Predict: extrapolate from p_prev → p_last by the same step.
        let predicted = [
            2.0 * p_last[0] - p_prev[0],
            2.0 * p_last[1] - p_prev[1],
        ];
        let dx = p[0] - predicted[0];
        let dy = p[1] - predicted[1];
        let err = (dx * dx + dy * dy).sqrt();

        if err > max_error {
            result.push(p);
        }
        // Always update the reference points to track current trajectory
        p_prev = p_last;
        p_last = p;
    }

    // Always include the last point.
    if let Some(&last) = points.last() {
        if let Some(&stored_last) = result.last() {
            if (stored_last[0] - last[0]).abs() > f64::EPSILON
                || (stored_last[1] - last[1]).abs() > f64::EPSILON
            {
                result.push(last);
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Online (streaming) Douglas–Peucker
// ---------------------------------------------------------------------------

/// A streaming Douglas–Peucker compressor.
///
/// Points are fed one at a time via [`push`][`OnlineDouglasPeucker::push`]; the
/// internal buffer is simplified and emitted whenever the lookahead buffer
/// exceeds `buffer_size`.  Call [`finish`][`OnlineDouglasPeucker::finish`] to
/// flush any remaining points.
pub struct OnlineDouglasPeucker {
    tolerance: f64,
    buffer_size: usize,
    buffer: Vec<Point2D>,
    /// Points that have been finalised and emitted.
    output: Vec<Point2D>,
}

impl OnlineDouglasPeucker {
    /// Create a new online compressor.
    ///
    /// # Arguments
    ///
    /// * `tolerance`   – RDP perpendicular-distance tolerance.
    /// * `buffer_size` – How many raw points to accumulate before simplifying
    ///   a chunk.  Larger values give better compression at the cost of more
    ///   latency.
    pub fn new(tolerance: f64, buffer_size: usize) -> Self {
        Self {
            tolerance,
            buffer_size: buffer_size.max(4),
            buffer: Vec::new(),
            output: Vec::new(),
        }
    }

    /// Feed one point into the compressor.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`douglas_peucker`].
    pub fn push(&mut self, p: Point2D) -> SpatialResult<()> {
        self.buffer.push(p);
        if self.buffer.len() >= self.buffer_size {
            self.flush()?;
        }
        Ok(())
    }

    /// Flush the internal buffer and return all emitted output so far.
    ///
    /// Retains the last point in the buffer as the anchor for the next chunk.
    fn flush(&mut self) -> SpatialResult<()> {
        if self.buffer.len() < 2 {
            return Ok(());
        }
        let mut simplified = douglas_peucker(&self.buffer, self.tolerance)?;

        // Keep the last point in the buffer as anchor for the next chunk.
        let anchor = *self.buffer.last().expect("buffer non-empty after size check");

        // Do not re-emit the first point if it was already emitted as the
        // last anchor of the previous flush.
        if let Some(&prev_last) = self.output.last() {
            if (simplified[0][0] - prev_last[0]).abs() < f64::EPSILON
                && (simplified[0][1] - prev_last[1]).abs() < f64::EPSILON
            {
                simplified.remove(0);
            }
        }

        // Move all but the last point to output.
        if !simplified.is_empty() {
            let keep_end = simplified.len().saturating_sub(1);
            self.output.extend_from_slice(&simplified[..keep_end]);
        }

        self.buffer.clear();
        self.buffer.push(anchor);
        Ok(())
    }

    /// Finish the stream and return the fully simplified trajectory.
    ///
    /// # Errors
    ///
    /// Propagates any error from the final [`douglas_peucker`] call.
    pub fn finish(mut self) -> SpatialResult<Vec<Point2D>> {
        if self.buffer.len() >= 2 {
            let simplified = douglas_peucker(&self.buffer, self.tolerance)?;
            if let Some(&prev_last) = self.output.last() {
                let mut start = 0;
                if let Some(&first) = simplified.first() {
                    if (first[0] - prev_last[0]).abs() < f64::EPSILON
                        && (first[1] - prev_last[1]).abs() < f64::EPSILON
                    {
                        start = 1;
                    }
                }
                self.output.extend_from_slice(&simplified[start..]);
            } else {
                self.output.extend_from_slice(&simplified);
            }
        } else {
            // Single remaining point: just append it if distinct from output end.
            if let Some(&p) = self.buffer.first() {
                if let Some(&prev) = self.output.last() {
                    if (p[0] - prev[0]).abs() > f64::EPSILON
                        || (p[1] - prev[1]).abs() > f64::EPSILON
                    {
                        self.output.push(p);
                    }
                } else {
                    self.output.push(p);
                }
            }
        }
        Ok(self.output)
    }
}

// ---------------------------------------------------------------------------
// Compression ratio
// ---------------------------------------------------------------------------

/// Compute the compression ratio of a simplified trajectory.
///
/// Returns `original.len() / simplified.len()`.  A ratio > 1 means compression
/// was achieved.  Returns 1.0 if either slice is empty.
pub fn compression_ratio(original: &[Point2D], simplified: &[Point2D]) -> f64 {
    if simplified.is_empty() || original.is_empty() {
        return 1.0;
    }
    original.len() as f64 / simplified.len() as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn collinear_points(n: usize) -> Vec<Point2D> {
        (0..n).map(|i| [i as f64, 0.0]).collect()
    }

    #[test]
    fn test_dp_collinear_reduces_to_endpoints() {
        let pts = collinear_points(100);
        let simplified = douglas_peucker(&pts, 0.01).expect("dp collinear");
        assert_eq!(simplified.len(), 2, "collinear points should reduce to 2");
    }

    #[test]
    fn test_dp_preserves_endpoints() {
        let pts = collinear_points(10);
        let simplified = douglas_peucker(&pts, 0.5).expect("dp endpoints");
        assert_eq!(simplified.first(), Some(&pts[0]));
        assert_eq!(simplified.last(), Some(pts.last().expect("last exists")));
    }

    #[test]
    fn test_dp_high_tolerance() {
        let pts: Vec<Point2D> = vec![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [3.0, 1.0],
            [4.0, 0.0],
        ];
        // Very high tolerance → only keep endpoints.
        let simplified = douglas_peucker(&pts, 10.0).expect("dp high");
        assert_eq!(simplified.len(), 2);
    }

    #[test]
    fn test_vw_reduces_collinear() {
        let pts = collinear_points(20);
        // Triangle area for collinear points = 0 → all interior points removed.
        let simplified =
            visvalingam_whyatt(&pts, 0.001, None).expect("vw collinear");
        assert!(simplified.len() <= 3, "got {}", simplified.len());
    }

    #[test]
    fn test_dead_reckoning_straight_line() {
        // On a perfectly straight line, dead reckoning never deviates.
        let pts = collinear_points(20);
        let compressed = dead_reckoning(&pts, 0.01).expect("dr straight");
        // Should keep only start, anchor, and end.
        assert!(compressed.len() <= 3, "got {}", compressed.len());
    }

    #[test]
    fn test_dead_reckoning_zigzag() {
        // Sharp zigzag: every step exceeds the threshold.
        let pts: Vec<Point2D> = (0..10)
            .map(|i| [i as f64, if i % 2 == 0 { 0.0 } else { 10.0 }])
            .collect();
        let compressed = dead_reckoning(&pts, 0.1).expect("dr zigzag");
        // Most points should be kept.
        assert!(compressed.len() >= 5);
    }

    #[test]
    fn test_online_dp_basic() {
        let pts = collinear_points(50);
        let mut compressor = OnlineDouglasPeucker::new(0.01, 10);
        for &p in &pts {
            compressor.push(p).expect("push ok");
        }
        let result = compressor.finish().expect("finish ok");
        assert!(result.len() <= pts.len());
        assert!(result.len() >= 2);
    }

    #[test]
    fn test_compression_ratio() {
        let original = collinear_points(100);
        let simplified = collinear_points(10);
        let ratio = compression_ratio(&original, &simplified);
        assert!((ratio - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_compression_ratio_empty() {
        let r = compression_ratio(&[], &[]);
        assert!((r - 1.0).abs() < 1e-10);
    }
}
