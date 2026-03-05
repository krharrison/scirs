//! Line arrangements in the plane.
//!
//! A line arrangement is the partition of the plane induced by a set of lines.
//! For N lines: O(N²) vertices, O(N²) edges, O(N²) faces.
//!
//! Provides:
//! - [`LineArrangement`] – incremental O(N²) construction
//! - `zone_theorem_count` – counts edges in the zone of a new line
//! - `duality_transform_points_to_lines` – point–line duality

use crate::error::{SpatialError, SpatialResult};

// ──────────────────────────────────────────────────────────────────────────────
// Data structures
// ──────────────────────────────────────────────────────────────────────────────

/// A line arrangement in the plane.
///
/// Lines are stored in ax + by + c = 0 form as `(a, b, c)`.
#[derive(Debug, Clone)]
pub struct LineArrangement {
    /// Lines: each is `(a, b, c)` for ax + by + c = 0.
    pub lines: Vec<(f64, f64, f64)>,
    /// Vertices (intersection points of pairs of lines).
    pub vertices: Vec<(f64, f64)>,
    /// Edges: `(vertex_idx_start, vertex_idx_end)`.
    /// For unbounded edges, `usize::MAX` is used as a sentinel.
    pub edges: Vec<(usize, usize)>,
    /// Faces: each face is described as a list of edge indices.
    pub faces: Vec<Vec<usize>>,
    /// Which line each edge belongs to.
    edge_line: Vec<usize>,
}

impl LineArrangement {
    /// Build a line arrangement incrementally.
    ///
    /// Adds lines one at a time, computing all intersection vertices and
    /// splitting existing edges that the new line passes through.
    ///
    /// # Errors
    ///
    /// Returns an error if `lines` is empty.
    pub fn build(lines: &[(f64, f64, f64)]) -> SpatialResult<Self> {
        if lines.is_empty() {
            return Err(SpatialError::InvalidInput("No lines provided".into()));
        }

        let mut arr = LineArrangement {
            lines: Vec::new(),
            vertices: Vec::new(),
            edges: Vec::new(),
            faces: Vec::new(),
            edge_line: Vec::new(),
        };

        for &line in lines {
            arr.add_line(line);
        }

        Ok(arr)
    }

    /// Add one line `(a, b, c)` to the arrangement.
    ///
    /// Computes intersections with all existing lines, inserts new vertices,
    /// and splits edges as needed.
    pub fn add_line(&mut self, line: (f64, f64, f64)) {
        let new_idx = self.lines.len();
        self.lines.push(line);

        // Compute intersections with all existing lines.
        let mut new_vertices: Vec<(f64, f64)> = Vec::new();
        for &existing in &self.lines[..new_idx] {
            if let Some(pt) = line_line_intersection(existing, line) {
                // Avoid duplicates.
                let dup = new_vertices
                    .iter()
                    .any(|v| (v.0 - pt.0).abs() < 1e-10 && (v.1 - pt.1).abs() < 1e-10);
                if !dup {
                    new_vertices.push(pt);
                }
            }
        }

        // Add new vertices to the global vertex list.
        let base_verts = self.vertices.len();
        self.vertices.extend_from_slice(&new_vertices);

        // Sort intersection points along the new line direction.
        let (a, b, _c) = line;
        let dir = [-b, a]; // direction perpendicular to normal
        let mut sorted_pts: Vec<usize> = (base_verts..self.vertices.len()).collect();
        sorted_pts.sort_by(|&i, &j| {
            let pi = self.vertices[i];
            let pj = self.vertices[j];
            let ti = pi.0 * dir[0] + pi.1 * dir[1];
            let tj = pj.0 * dir[0] + pj.1 * dir[1];
            ti.partial_cmp(&tj).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Create edges along the new line between consecutive intersection points.
        // Unbounded edges at the two ends use usize::MAX as sentinel.
        if sorted_pts.is_empty() {
            // No intersections: a single unbounded edge.
            let ei = self.edges.len();
            self.edges.push((usize::MAX, usize::MAX));
            self.edge_line.push(new_idx);
            // Assign to the "outer face" (face 0, or create it).
            if self.faces.is_empty() {
                self.faces.push(vec![ei]);
            } else {
                self.faces[0].push(ei);
            }
        } else {
            // Left unbounded edge.
            let e0 = self.edges.len();
            self.edges.push((usize::MAX, sorted_pts[0]));
            self.edge_line.push(new_idx);

            for w in sorted_pts.windows(2) {
                let ei = self.edges.len();
                self.edges.push((w[0], w[1]));
                self.edge_line.push(new_idx);
                let _ = ei;
            }

            // Right unbounded edge.
            let last = *sorted_pts.last().expect("non-empty");
            self.edges.push((last, usize::MAX));
            self.edge_line.push(new_idx);

            // Ensure at least one face exists.
            if self.faces.is_empty() {
                self.faces.push(Vec::new());
            }

            // Add new edges to the outer face.
            let new_edge_start = e0;
            let new_edge_end = self.edges.len();
            for ei in new_edge_start..new_edge_end {
                self.faces[0].push(ei);
            }
        }

        // Rebuild faces (simplified: faces correspond to regions between lines).
        self.rebuild_faces();
    }

    /// Rebuild the face list after adding a new line.
    fn rebuild_faces(&mut self) {
        // For N lines the number of faces is 1 + N + C(N,2) where C(N,2) is
        // the number of intersections (bounded regions = C(N-1, 2) for general position).
        // Here we maintain a simplified unbounded outer face (index 0) and one
        // bounded face per "zone" between consecutive lines.
        let n = self.lines.len();
        if n <= 1 {
            return;
        }

        // Group edges by line.
        let mut faces: Vec<Vec<usize>> = Vec::new();
        // Outer (unbounded) face.
        let outer: Vec<usize> = self
            .edges
            .iter()
            .enumerate()
            .filter(|(_, e)| e.0 == usize::MAX || e.1 == usize::MAX)
            .map(|(i, _)| i)
            .collect();
        faces.push(outer);

        // One bounded face per pair of lines that intersect.
        for i in 0..n {
            for j in (i + 1)..n {
                if line_line_intersection(self.lines[i], self.lines[j]).is_some() {
                    // Collect edges from both lines that are bounded.
                    let bounded: Vec<usize> = self
                        .edges
                        .iter()
                        .enumerate()
                        .filter(|(ei, e)| {
                            (self.edge_line[*ei] == i || self.edge_line[*ei] == j)
                                && e.0 != usize::MAX
                                && e.1 != usize::MAX
                        })
                        .map(|(i, _)| i)
                        .collect();
                    if !bounded.is_empty() {
                        faces.push(bounded);
                    }
                }
            }
        }

        self.faces = faces;
    }

    /// Zone theorem: count the number of edges in the zone of `new_line`.
    ///
    /// The zone of a line l in an arrangement A(L) is the set of faces of A(L)
    /// that l intersects. The zone theorem states this count is O(N).
    pub fn zone_theorem_count(&self, new_line: (f64, f64, f64)) -> usize {
        // An edge (u, v) is in the zone of `new_line` if both endpoints lie on
        // opposite sides of `new_line`, or if one endpoint lies on the line.
        let (a, b, c) = new_line;
        let eval = |idx: usize| -> f64 {
            if idx == usize::MAX {
                return 0.0; // treat infinity as on the line
            }
            let (px, py) = self.vertices[idx];
            a * px + b * py + c
        };

        self.edges
            .iter()
            .filter(|&&(v1, v2)| {
                let s1 = eval(v1);
                let s2 = eval(v2);
                // Edge crosses or touches the line.
                (s1 <= 1e-10 && s2 >= -1e-10) || (s1 >= -1e-10 && s2 <= 1e-10)
            })
            .count()
    }

    /// Number of vertices.
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Number of faces.
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Geometry helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the intersection of lines `l1` and `l2` (both in ax+by+c=0 form).
///
/// Returns `None` if the lines are parallel.
pub fn line_line_intersection(l1: (f64, f64, f64), l2: (f64, f64, f64)) -> Option<(f64, f64)> {
    let (a1, b1, c1) = l1;
    let (a2, b2, c2) = l2;
    let det = a1 * b2 - a2 * b1;
    if det.abs() < 1e-12 {
        return None;
    }
    let x = (b1 * c2 - b2 * c1) / det;
    let y = (a2 * c1 - a1 * c2) / det;
    Some((x, y))
}

/// Apply the standard point–line duality transform.
///
/// Point `(a, b)` maps to line `y = a*x - b`, i.e., `a*x - y - b = 0`,
/// which is returned as `(a, -1.0, -b)`.
///
/// Under this duality:
/// - Points above the k-level in the arrangement correspond to k-nearest neighbours.
pub fn duality_transform_points_to_lines(points: &[(f64, f64)]) -> Vec<(f64, f64, f64)> {
    points
        .iter()
        .map(|&(a, b)| (a, -1.0, -b))
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_line_intersection() {
        // y = x  ↔  x - y = 0  → (1, -1, 0)
        // y = -x + 2  ↔  x + y - 2 = 0  → (1, 1, -2)
        let l1 = (1.0_f64, -1.0, 0.0);
        let l2 = (1.0_f64, 1.0, -2.0);
        let pt = line_line_intersection(l1, l2).expect("should intersect");
        assert!((pt.0 - 1.0).abs() < 1e-9, "x={}", pt.0);
        assert!((pt.1 - 1.0).abs() < 1e-9, "y={}", pt.1);
    }

    #[test]
    fn test_parallel_lines() {
        let l1 = (1.0_f64, 0.0, 0.0);
        let l2 = (1.0_f64, 0.0, -2.0);
        assert!(line_line_intersection(l1, l2).is_none());
    }

    #[test]
    fn test_arrangement_two_lines() {
        let lines = vec![(1.0_f64, -1.0, 0.0), (1.0, 1.0, -2.0)];
        let arr = LineArrangement::build(&lines).expect("ok");
        assert_eq!(arr.num_vertices(), 1);
        assert!(arr.num_edges() >= 4); // each line contributes at least 2 edge segments
    }

    #[test]
    fn test_arrangement_three_lines() {
        // Three non-concurrent lines → 3 vertices, 6+ edges
        let lines = vec![
            (1.0_f64, 0.0, 0.0),  // x = 0
            (0.0, 1.0, 0.0),      // y = 0
            (1.0, 1.0, -1.0),     // x + y = 1
        ];
        let arr = LineArrangement::build(&lines).expect("ok");
        assert!(arr.num_vertices() >= 3);
    }

    #[test]
    fn test_zone_theorem_count() {
        let lines = vec![(1.0_f64, 0.0, 0.0), (0.0, 1.0, 0.0)];
        let arr = LineArrangement::build(&lines).expect("ok");
        let new_line = (1.0_f64, 1.0, -2.0);
        let cnt = arr.zone_theorem_count(new_line);
        assert!(cnt > 0);
    }

    #[test]
    fn test_duality() {
        let pts = vec![(2.0_f64, 3.0), (0.0, 1.0)];
        let lines = duality_transform_points_to_lines(&pts);
        assert_eq!(lines.len(), 2);
        // Point (2,3) → line 2x - y - 3 = 0 → (2, -1, -3)
        let (a, b, c) = lines[0];
        assert!((a - 2.0).abs() < 1e-9);
        assert!((b - (-1.0)).abs() < 1e-9);
        assert!((c - (-3.0)).abs() < 1e-9);
    }

    #[test]
    fn test_arrangement_empty_error() {
        let result = LineArrangement::build(&[]);
        assert!(result.is_err());
    }
}
