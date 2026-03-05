//! Convex hull algorithms for 2D point sets.
//!
//! Implements three well-known algorithms:
//! - Andrew's **monotone chain** (`convex_hull_2d`) — O(n log n), returns CCW hull.
//! - **Graham scan** via `GrahamScan::compute` — delegates to monotone chain.
//! - **Jarvis march** (gift wrapping) via `JarvisMarch::compute` — O(nh), exact.
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::geom::convex_hull::{convex_hull_2d, GrahamScan, JarvisMarch};
//!
//! let pts = vec![
//!     [0.0_f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5],
//! ];
//! let hull = convex_hull_2d(pts.clone());
//! assert_eq!(hull.len(), 4);
//! ```

// ── Cross-product helper ───────────────────────────────────────────────────────

/// Signed cross product of vectors OA and OB.
#[inline]
fn cross(o: &[f64; 2], a: &[f64; 2], b: &[f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

// ── Andrew's monotone chain ────────────────────────────────────────────────────

/// Compute the **convex hull** of a 2D point set using Andrew's monotone chain.
///
/// Returns the hull vertices in counter-clockwise order.  Collinear points on
/// the boundary are excluded.
///
/// Time complexity: O(n log n).  Returns the input unchanged if `n < 3`.
pub fn convex_hull_2d(mut points: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    let n = points.len();
    if n < 3 {
        return points;
    }

    // Lexicographic sort: by x, then y
    points.sort_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut lower: Vec<[f64; 2]> = Vec::with_capacity(n);
    for p in &points {
        while lower.len() >= 2 && cross(&lower[lower.len() - 2], &lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(*p);
    }

    let mut upper: Vec<[f64; 2]> = Vec::with_capacity(n);
    for p in points.iter().rev() {
        while upper.len() >= 2 && cross(&upper[upper.len() - 2], &upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(*p);
    }

    // Remove last points of each half (they are the first points of the other half)
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

// ── Graham scan ───────────────────────────────────────────────────────────────

/// Graham scan convex hull algorithm.
///
/// In this implementation the Graham scan is equivalent to Andrew's monotone
/// chain — both are Θ(n log n) and produce identical output.
pub struct GrahamScan;

impl GrahamScan {
    /// Compute the convex hull (CCW) of `points`.
    pub fn compute(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
        convex_hull_2d(points.to_vec())
    }
}

// ── Jarvis march (gift wrapping) ──────────────────────────────────────────────

/// Jarvis march (gift wrapping) convex hull algorithm.
///
/// Time complexity O(n·h) where h is the number of hull vertices.
/// Preferred over monotone chain when h ≪ n.
pub struct JarvisMarch;

impl JarvisMarch {
    /// Compute the convex hull (CCW) of `points`.
    ///
    /// Returns the input unchanged for fewer than 3 points.
    pub fn compute(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let n = points.len();
        if n < 3 {
            return points.to_vec();
        }

        // Find leftmost (and bottommost among ties) point as start
        let start = points
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a[0].partial_cmp(&b[0])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal))
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut hull: Vec<[f64; 2]> = Vec::new();
        let mut current = start;

        loop {
            hull.push(points[current]);
            // Find the most counter-clockwise point
            let mut next = (current + 1) % n;

            for i in 0..n {
                let c = cross(&points[current], &points[next], &points[i]);
                // Strictly counter-clockwise: update
                if c < 0.0 {
                    next = i;
                } else if c == 0.0 {
                    // Collinear: prefer the farther point to avoid missing hull vertices
                    let d_next = dist2(points[current], points[next]);
                    let d_i = dist2(points[current], points[i]);
                    if d_i > d_next {
                        next = i;
                    }
                }
            }

            current = next;
            if current == start {
                break;
            }
            // Safety: hull cannot exceed n vertices
            if hull.len() >= n {
                break;
            }
        }

        hull
    }
}

#[inline]
fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

// ── Diameter (farthest pair) ──────────────────────────────────────────────────

/// Compute the diameter of a point set (maximum pairwise distance) using
/// rotating calipers on the convex hull.
///
/// Returns `(index_a, index_b, distance)` in the *original* `points` slice,
/// or `None` if fewer than 2 points.
pub fn point_set_diameter(points: &[[f64; 2]]) -> Option<(usize, usize, f64)> {
    let n = points.len();
    if n < 2 {
        return None;
    }
    if n == 2 {
        let d = ((points[0][0]-points[1][0]).powi(2) + (points[0][1]-points[1][1]).powi(2)).sqrt();
        return Some((0, 1, d));
    }

    let hull = convex_hull_2d(points.to_vec());
    let h = hull.len();
    if h < 2 {
        // Degenerate
        let d = ((points[0][0]-points[1][0]).powi(2) + (points[0][1]-points[1][1]).powi(2)).sqrt();
        return Some((0, 1, d));
    }

    let mut max_d2 = 0.0_f64;
    let mut pi = 0usize;
    let mut pj = 0usize;

    // Rotating calipers
    let mut j = 1usize;
    for i in 0..h {
        let ni = (i + 1) % h;
        loop {
            let nj = (j + 1) % h;
            // Cross product of hull edge i→ni with vector i→j vs i→nj
            let ex = hull[ni][0] - hull[i][0];
            let ey = hull[ni][1] - hull[i][1];
            let cross_j  = ex * (hull[j][1]  - hull[i][1]) - ey * (hull[j][0]  - hull[i][0]);
            let cross_nj = ex * (hull[nj][1] - hull[i][1]) - ey * (hull[nj][0] - hull[i][0]);
            if cross_nj > cross_j {
                j = nj;
            } else {
                break;
            }
        }
        let d2 = (hull[i][0]-hull[j][0]).powi(2) + (hull[i][1]-hull[j][1]).powi(2);
        if d2 > max_d2 {
            max_d2 = d2;
            // Map back to original indices
            pi = find_idx(points, hull[i]);
            pj = find_idx(points, hull[j]);
        }
    }

    Some((pi, pj, max_d2.sqrt()))
}

fn find_idx(points: &[[f64; 2]], target: [f64; 2]) -> usize {
    points.iter().enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (a[0]-target[0]).powi(2) + (a[1]-target[1]).powi(2);
            let db = (b[0]-target[0]).powi(2) + (b[1]-target[1]).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convex_hull_square() {
        let pts = vec![
            [0.0_f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5],
        ];
        let hull = convex_hull_2d(pts);
        assert_eq!(hull.len(), 4, "Square interior point should be excluded");
    }

    #[test]
    fn test_convex_hull_triangle() {
        let pts = vec![[0.0_f64, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let hull = convex_hull_2d(pts);
        assert_eq!(hull.len(), 3);
    }

    #[test]
    fn test_convex_hull_collinear() {
        // All on a line
        let pts: Vec<[f64; 2]> = (0..5).map(|i| [i as f64, 0.0]).collect();
        let hull = convex_hull_2d(pts);
        // Monotone chain removes collinear points → 2 endpoints
        assert!(hull.len() <= 2, "Got {} hull points", hull.len());
    }

    #[test]
    fn test_graham_scan_square() {
        let pts = vec![
            [0.0_f64, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0], [2.0, 1.5],
        ];
        let hull = GrahamScan::compute(&pts);
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn test_jarvis_march_square() {
        let pts = vec![
            [0.0_f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5],
        ];
        let hull = JarvisMarch::compute(&pts);
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn test_convex_hull_too_few_points() {
        assert_eq!(convex_hull_2d(vec![]).len(), 0);
        assert_eq!(convex_hull_2d(vec![[0.0, 0.0]]).len(), 1);
        assert_eq!(convex_hull_2d(vec![[0.0, 0.0], [1.0, 1.0]]).len(), 2);
    }

    #[test]
    fn test_point_set_diameter() {
        let pts = vec![[0.0_f64, 0.0], [3.0, 0.0], [0.0, 4.0]];
        let (_, _, d) = point_set_diameter(&pts).expect("Should return diameter");
        // Diameter is the longest edge: hypot(3,4)=5
        assert!((d - 5.0).abs() < 1e-9, "Expected 5.0, got {}", d);
    }
}
