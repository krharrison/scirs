//! Closest pair of points — O(n log n) divide-and-conquer algorithm.
//!
//! Also provides a brute-force O(n²) reference implementation used for small
//! cases and testing.
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::geom::closest_pair::closest_pair;
//!
//! let pts = [[0.0_f64, 0.0], [3.0, 4.0], [1.0, 1.0]];
//! let (i, j, d) = closest_pair(&pts).expect("non-empty");
//! // Closest pair is (0,0) and (1,1), distance = sqrt(2)
//! assert!((d - 2_f64.sqrt()).abs() < 1e-9);
//! ```

use crate::error::{SpatialError, SpatialResult};

// ── Public API ─────────────────────────────────────────────────────────────────

/// Find the closest pair of points in 2D in O(n log n).
///
/// Returns `Some((i, j, distance))` where `i` and `j` are indices into
/// `points` with `i < j`, or `None` if `points` has fewer than 2 entries.
pub fn closest_pair(points: &[[f64; 2]]) -> Option<(usize, usize, f64)> {
    let n = points.len();
    if n < 2 {
        return None;
    }
    if n == 2 {
        return Some((0, 1, euclidean(points[0], points[1])));
    }

    // Build sorted index array (sort by x, break ties by y)
    let mut sorted_by_x: Vec<usize> = (0..n).collect();
    sorted_by_x.sort_by(|&a, &b| {
        points[a][0]
            .partial_cmp(&points[b][0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                points[a][1]
                    .partial_cmp(&points[b][1])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let (i, j, d) = rec_closest(points, &sorted_by_x);
    if i < j { Some((i, j, d)) } else { Some((j, i, d)) }
}

/// Brute-force O(n²) closest pair.  Used internally for small sub-problems.
pub fn closest_pair_brute(points: &[[f64; 2]]) -> Option<(usize, usize, f64)> {
    let n = points.len();
    if n < 2 { return None; }
    let mut best_i = 0;
    let mut best_j = 1;
    let mut best_d = euclidean(points[0], points[1]);
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean(points[i], points[j]);
            if d < best_d {
                best_d = d;
                best_i = i;
                best_j = j;
            }
        }
    }
    Some((best_i, best_j, best_d))
}

// ── Divide-and-conquer kernel ─────────────────────────────────────────────────

/// Recursive divide-and-conquer on a sub-slice of indices sorted by x.
/// Returns `(orig_idx_a, orig_idx_b, distance)`.
fn rec_closest(pts: &[[f64; 2]], sorted_x: &[usize]) -> (usize, usize, f64) {
    let n = sorted_x.len();

    // Base cases: brute-force for small n
    if n <= 8 {
        return brute_on_indices(pts, sorted_x);
    }

    let mid = n / 2;
    let mid_x = pts[sorted_x[mid]][0];

    let (li, lj, ld) = rec_closest(pts, &sorted_x[..mid]);
    let (ri, rj, rd) = rec_closest(pts, &sorted_x[mid..]);

    let (mut best_i, mut best_j, mut best_d) = if ld <= rd {
        (li, lj, ld)
    } else {
        (ri, rj, rd)
    };

    // Build strip of points within best_d of the dividing line
    let mut strip: Vec<usize> = sorted_x
        .iter()
        .copied()
        .filter(|&idx| (pts[idx][0] - mid_x).abs() < best_d)
        .collect();

    // Sort strip by y
    strip.sort_by(|&a, &b| {
        pts[a][1]
            .partial_cmp(&pts[b][1])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Check pairs in strip (inner loop bounded by O(1) comparisons)
    let sn = strip.len();
    for i in 0..sn {
        let mut j = i + 1;
        while j < sn && (pts[strip[j]][1] - pts[strip[i]][1]) < best_d {
            let d = euclidean(pts[strip[i]], pts[strip[j]]);
            if d < best_d {
                best_d = d;
                best_i = strip[i];
                best_j = strip[j];
            }
            j += 1;
        }
    }

    (best_i, best_j, best_d)
}

fn brute_on_indices(pts: &[[f64; 2]], indices: &[usize]) -> (usize, usize, f64) {
    let mut best_i = indices[0];
    let mut best_j = indices[1];
    let mut best_d = euclidean(pts[indices[0]], pts[indices[1]]);
    for a in 0..indices.len() {
        for b in (a + 1)..indices.len() {
            let d = euclidean(pts[indices[a]], pts[indices[b]]);
            if d < best_d {
                best_d = d;
                best_i = indices[a];
                best_j = indices[b];
            }
        }
    }
    (best_i, best_j, best_d)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline]
fn euclidean(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

// ── All-pairs distances (convenience) ─────────────────────────────────────────

/// Compute the full pairwise distance matrix for a point set.
///
/// Returns an `n × n` flat vector (row-major) of Euclidean distances.
pub fn pairwise_distances(points: &[[f64; 2]]) -> Vec<f64> {
    let n = points.len();
    let mut mat = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean(points[i], points[j]);
            mat[i * n + j] = d;
            mat[j * n + i] = d;
        }
    }
    mat
}

/// Farthest pair of points in 2D, O(n²).
///
/// Returns `Some((i, j, distance))` or `None` if fewer than 2 points.
pub fn farthest_pair(points: &[[f64; 2]]) -> Option<(usize, usize, f64)> {
    let n = points.len();
    if n < 2 { return None; }
    let mut best_i = 0;
    let mut best_j = 1;
    let mut best_d = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean(points[i], points[j]);
            if d > best_d {
                best_d = d;
                best_i = i;
                best_j = j;
            }
        }
    }
    Some((best_i, best_j, best_d))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_closest_pair_basic() {
        let pts = [[0.0_f64, 0.0], [3.0, 4.0], [1.0, 1.0]];
        let (i, j, d) = closest_pair(&pts).expect("non-empty");
        // (0,0)↔(1,1) = sqrt(2)
        let expected = 2.0_f64.sqrt();
        assert!(
            (d - expected).abs() < 1e-9,
            "Expected {expected}, got {d} for pair ({i},{j})"
        );
    }

    #[test]
    fn test_closest_pair_none_for_one_point() {
        assert!(closest_pair(&[[0.0, 0.0]]).is_none());
        assert!(closest_pair(&[]).is_none());
    }

    #[test]
    fn test_closest_pair_two_points() {
        let pts = [[0.0_f64, 0.0], [3.0, 4.0]];
        let (i, j, d) = closest_pair(&pts).expect("non-empty");
        assert!((d - 5.0).abs() < 1e-9);
        assert_eq!((i, j), (0, 1));
    }

    #[test]
    fn test_closest_pair_matches_brute() {
        let pts = [
            [1.0_f64, 2.0], [4.0, 6.0], [0.5, 0.5], [3.0, 1.0],
            [7.0, 8.0], [2.0, 3.0], [5.0, 5.0], [0.0, 4.0],
            [6.0, 2.0], [1.5, 1.5],
        ];
        let (_, _, d_fast) = closest_pair(&pts).expect("non-empty");
        let (_, _, d_brute) = closest_pair_brute(&pts).expect("non-empty");
        assert!(
            (d_fast - d_brute).abs() < 1e-9,
            "fast={d_fast} brute={d_brute}"
        );
    }

    #[test]
    fn test_closest_pair_large() {
        // 100 points on a grid; each cell is 1×1 so min dist = 1
        let pts: Vec<[f64; 2]> = (0..100)
            .map(|i| [(i % 10) as f64, (i / 10) as f64])
            .collect();
        let (_, _, d) = closest_pair(&pts).expect("non-empty");
        assert!((d - 1.0).abs() < 1e-9, "Expected 1.0, got {d}");
    }

    #[test]
    fn test_pairwise_distances_triangle() {
        let pts = [[0.0_f64, 0.0], [3.0, 0.0], [0.0, 4.0]];
        let mat = pairwise_distances(&pts);
        assert!((mat[0 * 3 + 1] - 3.0).abs() < 1e-12);
        assert!((mat[0 * 3 + 2] - 4.0).abs() < 1e-12);
        assert!((mat[1 * 3 + 2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_farthest_pair() {
        let pts = [[0.0_f64, 0.0], [3.0, 0.0], [0.0, 4.0]];
        let (i, j, d) = farthest_pair(&pts).expect("non-empty");
        assert!((d - 5.0).abs() < 1e-12, "Expected 5.0 (hypotenuse), got {d}");
    }
}
