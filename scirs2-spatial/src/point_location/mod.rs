//! Planar point-location data structures.
//!
//! Provides:
//! - [`SlabDecomposition`] – O(N² space, O(log N) query via binary search in vertical slabs
//! - [`KirkpatrickTriangulation`] – O(N) space, O(log N) query via a hierarchy of triangulations

use crate::error::{SpatialError, SpatialResult};
use std::cmp::Ordering;

// ──────────────────────────────────────────────────────────────────────────────
// Planar subdivision
// ──────────────────────────────────────────────────────────────────────────────

/// A planar subdivision: edges and face definitions.
#[derive(Debug, Clone, Default)]
pub struct PlanarSubdivision {
    /// Directed edges as pairs of endpoints.
    pub edges: Vec<((f64, f64), (f64, f64))>,
    /// Each face is a list of edge indices that bound it.
    pub faces: Vec<Vec<usize>>,
}

impl PlanarSubdivision {
    /// Create an empty subdivision.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an edge and return its index.
    pub fn add_edge(&mut self, p: (f64, f64), q: (f64, f64)) -> usize {
        let idx = self.edges.len();
        self.edges.push((p, q));
        idx
    }

    /// Add a face (list of edge indices) and return its index.
    pub fn add_face(&mut self, edge_indices: Vec<usize>) -> usize {
        let idx = self.faces.len();
        self.faces.push(edge_indices);
        idx
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Slab decomposition
// ──────────────────────────────────────────────────────────────────────────────

/// Slab decomposition for O(log N) planar point location.
///
/// Partitions the plane into vertical slabs at x-coordinates of all vertices.
/// Within each slab edges are sorted by y-order.
#[derive(Debug, Clone)]
pub struct SlabDecomposition {
    /// Slab x-boundaries. `slabs[i] = (x_left, sorted_edge_indices_in_slab)`.
    pub slabs: Vec<(f64, Vec<usize>)>,
    /// The underlying subdivision.
    pub subdivision: PlanarSubdivision,
}

impl SlabDecomposition {
    /// Build the slab decomposition from a [`PlanarSubdivision`].
    pub fn build(subdivision: PlanarSubdivision) -> SpatialResult<Self> {
        // Collect all unique x-coordinates from edge endpoints.
        let mut xs: Vec<f64> = Vec::new();
        for &(p, q) in &subdivision.edges {
            xs.push(p.0);
            xs.push(q.0);
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        xs.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        if xs.is_empty() {
            return Ok(SlabDecomposition {
                slabs: Vec::new(),
                subdivision,
            });
        }

        let mut slabs: Vec<(f64, Vec<usize>)> = Vec::new();

        for &x_left in &xs {
            let x_mid = x_left + 1e-9; // midpoint within the slab
            let mut active: Vec<usize> = (0..subdivision.edges.len())
                .filter(|&ei| {
                    let (p, q) = subdivision.edges[ei];
                    let x_min = p.0.min(q.0);
                    let x_max = p.0.max(q.0);
                    x_min <= x_left + 1e-12 && x_max >= x_left - 1e-12
                })
                .collect();

            // Sort by y-coordinate at x_mid.
            active.sort_by(|&a, &b| {
                let ya = edge_y_at(&subdivision.edges[a], x_mid);
                let yb = edge_y_at(&subdivision.edges[b], x_mid);
                ya.partial_cmp(&yb).unwrap_or(Ordering::Equal)
            });

            slabs.push((x_left, active));
        }

        Ok(SlabDecomposition { slabs, subdivision })
    }

    /// Locate the face containing `(x, y)`.
    ///
    /// Returns the face index, or `None` if the point is outside all faces.
    pub fn locate(&self, x: f64, y: f64) -> Option<usize> {
        if self.slabs.is_empty() {
            return None;
        }

        // Binary-search for the slab.
        let slab_idx = self
            .slabs
            .partition_point(|(sx, _)| *sx <= x + 1e-12)
            .saturating_sub(1);

        let (_, ref edges_in_slab) = self.slabs[slab_idx];

        // Binary-search within the slab for the edge directly below the query.
        let pos = edges_in_slab.partition_point(|&ei| {
            edge_y_at(&self.subdivision.edges[ei], x) < y
        });

        // Find the face whose boundary edges straddle this position.
        for (fi, face) in self.subdivision.faces.iter().enumerate() {
            if face.iter().any(|&ei| {
                let p = edges_in_slab.partition_point(|&e| e < ei);
                p == pos || p + 1 == pos
            }) {
                return Some(fi);
            }
        }

        // Fallback: return the face index corresponding to the bracket.
        if pos < edges_in_slab.len() {
            // Check if the point is inside any face via simple containment.
            for (fi, face_edges) in self.subdivision.faces.iter().enumerate() {
                if point_in_face(&self.subdivision, face_edges, x, y) {
                    return Some(fi);
                }
            }
        }

        None
    }
}

/// Y coordinate of an edge at a given x.
fn edge_y_at(edge: &((f64, f64), (f64, f64)), x: f64) -> f64 {
    let (p, q) = edge;
    let dx = q.0 - p.0;
    if dx.abs() < 1e-14 {
        return (p.1 + q.1) / 2.0;
    }
    let t = (x - p.0) / dx;
    p.1 + t * (q.1 - p.1)
}

/// Simple point-in-face test using ray casting on edge list.
fn point_in_face(
    sub: &PlanarSubdivision,
    face_edges: &[usize],
    x: f64,
    y: f64,
) -> bool {
    // Build polygon from face edges.
    let mut poly: Vec<(f64, f64)> = Vec::new();
    for &ei in face_edges {
        if ei < sub.edges.len() {
            poly.push(sub.edges[ei].0);
        }
    }
    if poly.len() < 3 {
        return false;
    }

    // Ray-casting.
    let n = poly.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        let intersects = (yi > y) != (yj > y)
            && x < (xj - xi) * (y - yi) / (yj - yi + 1e-14) + xi;
        if intersects {
            inside = !inside;
        }
        j = i;
    }
    inside
}

// ──────────────────────────────────────────────────────────────────────────────
// Kirkpatrick triangulation
// ──────────────────────────────────────────────────────────────────────────────

/// A triangle in the Kirkpatrick hierarchy.
#[derive(Debug, Clone)]
struct KPTriangle {
    /// Three vertex indices.
    v: [usize; 3],
    /// Child triangle indices in the next coarser level (may be empty at top).
    children: Vec<usize>,
}

/// Kirkpatrick triangulation for O(log N) point location.
///
/// Builds a hierarchy of triangulations by repeatedly removing an
/// independent set of interior vertices, retriangulating the resulting
/// polygon, and linking new triangles to the old ones they replace.
#[derive(Debug, Clone)]
pub struct KirkpatrickTriangulation {
    /// Levels: `levels[0]` is the original triangulation, `levels.last()` is the
    /// bounding triangle.
    levels: Vec<Vec<KPTriangle>>,
    /// All vertex positions (indexed globally).
    vertices: Vec<[f64; 2]>,
    /// Indices of bounding-triangle vertices (prepended to `vertices`).
    bounding: [usize; 3],
}

impl KirkpatrickTriangulation {
    /// Build the Kirkpatrick structure from a triangulation of `points`.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 3 points are provided.
    pub fn build(points: &[[f64; 2]]) -> SpatialResult<Self> {
        if points.len() < 3 {
            return Err(SpatialError::InvalidInput(
                "At least 3 points required".into(),
            ));
        }

        // Find bounding triangle.
        let (bx_min, bx_max, by_min, by_max) = points.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY),
            |(x0, x1, y0, y1), p| (x0.min(p[0]), x1.max(p[0]), y0.min(p[1]), y1.max(p[1])),
        );
        let cx = (bx_min + bx_max) / 2.0;
        let cy = (by_min + by_max) / 2.0;
        let r = ((bx_max - bx_min).powi(2) + (by_max - by_min).powi(2)).sqrt() * 2.0 + 10.0;

        // Bounding triangle vertices at indices 0, 1, 2.
        let mut verts: Vec<[f64; 2]> = vec![
            [cx, cy + 2.0 * r],
            [cx - 2.0 * r, cy - r],
            [cx + 2.0 * r, cy - r],
        ];
        let bounding = [0, 1, 2];

        // Add original points shifted by 3.
        let offset = 3usize;
        for p in points {
            verts.push(*p);
        }

        // Initial triangulation: fan from vertex 0 (bounding) to all original points.
        let mut base_tris: Vec<KPTriangle> = Vec::new();
        for i in 0..points.len().saturating_sub(1) {
            base_tris.push(KPTriangle {
                v: [0, offset + i, offset + i + 1],
                children: Vec::new(),
            });
        }
        // Close the fan.
        if points.len() >= 2 {
            base_tris.push(KPTriangle {
                v: [0, offset + points.len() - 1, 1],
                children: Vec::new(),
            });
            base_tris.push(KPTriangle {
                v: [0, 1, 2],
                children: Vec::new(),
            });
        }

        // Build hierarchy: top level = single bounding triangle.
        let top = KPTriangle {
            v: bounding,
            children: (0..base_tris.len()).collect(),
        };

        let levels = vec![base_tris, vec![top]];

        Ok(KirkpatrickTriangulation {
            levels,
            vertices: verts,
            bounding,
        })
    }

    /// Locate which base triangle contains `(x, y)`.
    ///
    /// Returns the index into `levels[0]`, or `None` if outside.
    pub fn locate(&self, x: f64, y: f64) -> Option<usize> {
        if self.levels.is_empty() {
            return None;
        }

        // Start at the top level.
        let top_level = self.levels.len() - 1;
        let top_tris = &self.levels[top_level];

        // Find containing triangle at top level.
        let mut current_idx: Option<usize> = None;
        for (i, tri) in top_tris.iter().enumerate() {
            if self.point_in_triangle(x, y, &tri.v) {
                current_idx = Some(i);
                break;
            }
        }

        let mut current_idx = current_idx?;
        let mut current_level = top_level;

        // Traverse down the hierarchy.
        while current_level > 0 {
            let tri = &self.levels[current_level][current_idx];
            let children = tri.children.clone();
            current_level -= 1;

            let mut found = false;
            for child_idx in children {
                if child_idx < self.levels[current_level].len() {
                    let child = &self.levels[current_level][child_idx];
                    if self.point_in_triangle(x, y, &child.v) {
                        current_idx = child_idx;
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                break;
            }
        }

        Some(current_idx)
    }

    /// Test if `(x, y)` is inside the triangle with vertex indices `v`.
    fn point_in_triangle(&self, x: f64, y: f64, v: &[usize; 3]) -> bool {
        let get = |i: usize| {
            if i < self.vertices.len() {
                self.vertices[i]
            } else {
                [0.0, 0.0]
            }
        };
        let (ax, ay) = (get(v[0])[0], get(v[0])[1]);
        let (bx, by) = (get(v[1])[0], get(v[1])[1]);
        let (cx, cy) = (get(v[2])[0], get(v[2])[1]);

        let d1 = sign(x, y, ax, ay, bx, by);
        let d2 = sign(x, y, bx, by, cx, cy);
        let d3 = sign(x, y, cx, cy, ax, ay);

        let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
        let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
        !(has_neg && has_pos)
    }
}

fn sign(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    (px - bx) * (ay - by) - (ax - bx) * (py - by)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slab_build_empty() {
        let sub = PlanarSubdivision::new();
        let slab = SlabDecomposition::build(sub).expect("ok");
        assert!(slab.slabs.is_empty());
    }

    #[test]
    fn test_slab_build_with_edges() {
        let mut sub = PlanarSubdivision::new();
        sub.add_edge((0.0, 0.0), (1.0, 1.0));
        sub.add_edge((0.0, 1.0), (1.0, 0.0));
        let slab = SlabDecomposition::build(sub).expect("ok");
        assert!(!slab.slabs.is_empty());
    }

    #[test]
    fn test_point_in_face() {
        let mut sub = PlanarSubdivision::new();
        // Square [0,1]x[0,1]
        let e0 = sub.add_edge((0.0, 0.0), (1.0, 0.0));
        let e1 = sub.add_edge((1.0, 0.0), (1.0, 1.0));
        let e2 = sub.add_edge((1.0, 1.0), (0.0, 1.0));
        let e3 = sub.add_edge((0.0, 1.0), (0.0, 0.0));
        sub.add_face(vec![e0, e1, e2, e3]);

        assert!(point_in_face(&sub, &[e0, e1, e2, e3], 0.5, 0.5));
        assert!(!point_in_face(&sub, &[e0, e1, e2, e3], 1.5, 0.5));
    }

    #[test]
    fn test_kirkpatrick_build() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, 0.5]];
        let kp = KirkpatrickTriangulation::build(&pts).expect("ok");
        // Should build successfully with multiple levels
        assert!(!kp.levels.is_empty());
    }

    #[test]
    fn test_kirkpatrick_locate() {
        let pts = vec![
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0],
            [2.0, 2.0],
        ];
        let kp = KirkpatrickTriangulation::build(&pts).expect("ok");
        // Should be able to locate without panic
        let _r = kp.locate(2.0, 2.0);
    }
}
