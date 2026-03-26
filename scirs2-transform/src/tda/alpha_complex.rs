//! Alpha complex filtration for 2D point clouds
//!
//! Implements the Bowyer-Watson incremental Delaunay triangulation and uses
//! circumradii as filtration values to build the alpha complex. Persistent
//! homology is then computed via standard boundary-matrix column reduction.
//!
//! ## Algorithm
//!
//! 1. Build Delaunay triangulation with Bowyer-Watson.
//! 2. Assign each simplex a filtration value = circumradius of its smallest
//!    enclosing circle:
//!    - 0-simplex (vertex):  0.0
//!    - 1-simplex (edge):    circumradius of the two-point "circle" = half edge length
//!    - 2-simplex (triangle): circumradius of the circumscribed circle
//! 3. Sort all simplices by filtration value.
//! 4. Reduce the boundary matrix to extract persistence pairs.

use crate::error::{Result, TransformError};
use std::collections::{HashMap, HashSet};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for alpha complex construction.
#[derive(Debug, Clone)]
pub struct AlphaConfig {
    /// Maximum alpha value — simplices with filtration value > max_alpha are ignored.
    pub max_alpha: f64,
}

impl Default for AlphaConfig {
    fn default() -> Self {
        Self {
            max_alpha: f64::INFINITY,
        }
    }
}

// ─── Simplex ──────────────────────────────────────────────────────────────────

/// A simplex with an associated filtration value.
#[derive(Debug, Clone, PartialEq)]
pub struct Simplex {
    /// Sorted vertex indices forming the simplex.
    pub vertices: Vec<usize>,
    /// Filtration value (circumradius for alpha complex).
    pub filtration_value: f64,
}

impl Simplex {
    /// Dimension of the simplex (number of vertices minus one).
    pub fn dimension(&self) -> usize {
        self.vertices.len().saturating_sub(1)
    }

    /// Return the boundary faces as Simplices with the same filtration value.
    ///
    /// The boundary of a k-simplex consists of all (k-1)-faces obtained by
    /// removing one vertex at a time.
    pub fn boundary_faces(&self) -> Vec<Vec<usize>> {
        if self.vertices.len() <= 1 {
            return Vec::new();
        }
        (0..self.vertices.len())
            .map(|i| {
                self.vertices
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, &v)| v)
                    .collect::<Vec<usize>>()
            })
            .collect()
    }
}

// ─── Geometry helpers ─────────────────────────────────────────────────────────

/// Compute the circumradius of the triangle formed by points a, b, c in 2D.
///
/// Returns `f64::INFINITY` if the points are collinear (degenerate triangle).
pub fn circumradius_2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    let ax = a[0] - c[0];
    let ay = a[1] - c[1];
    let bx = b[0] - c[0];
    let by = b[1] - c[1];

    let d = 2.0 * (ax * by - ay * bx);
    if d.abs() < 1e-14 {
        return f64::INFINITY; // collinear
    }

    let ux = (by * (ax * ax + ay * ay) - ay * (bx * bx + by * by)) / d;
    let uy = (ax * (bx * bx + by * by) - bx * (ax * ax + ay * ay)) / d;

    (ux * ux + uy * uy).sqrt()
}

/// Euclidean distance between two 2D points.
#[inline]
fn dist2d(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

// ─── Bowyer-Watson Delaunay triangulation ─────────────────────────────────────

/// A triangle stored by vertex indices (sorted).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Triangle {
    v: [usize; 3],
}

impl Triangle {
    fn new(a: usize, b: usize, c: usize) -> Self {
        let mut v = [a, b, c];
        v.sort_unstable();
        Self { v }
    }
}

/// Check whether point p lies inside the circumcircle of triangle (a, b, c).
fn in_circumcircle(p: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    // Use the standard determinant test.
    let ax = a[0] - p[0];
    let ay = a[1] - p[1];
    let bx = b[0] - p[0];
    let by = b[1] - p[1];
    let cx = c[0] - p[0];
    let cy = c[1] - p[1];

    let det = ax * (by * (cx * cx + cy * cy) - cy * (bx * bx + by * by))
        - ay * (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by))
        + (ax * ax + ay * ay) * (bx * cy - by * cx);

    // Positive det means p is inside (assuming CCW orientation of a,b,c).
    // We check both orientations.
    let orientation = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if orientation > 0.0 {
        det > 0.0
    } else {
        det < 0.0
    }
}

/// Run Bowyer-Watson incremental Delaunay triangulation on a set of 2D points.
///
/// Returns a list of triangles (each a sorted triple of vertex indices).
/// The super-triangle indices (n, n+1, n+2) are stripped from the output.
fn bowyer_watson(points: &[[f64; 2]]) -> Vec<Triangle> {
    let n = points.len();
    if n < 3 {
        return Vec::new();
    }

    // Bounding box → super-triangle
    let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in points {
        min_x = min_x.min(p[0]);
        min_y = min_y.min(p[1]);
        max_x = max_x.max(p[0]);
        max_y = max_y.max(p[1]);
    }
    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let delta_max = dx.max(dy).max(1.0);
    let mid_x = (min_x + max_x) / 2.0;
    let mid_y = (min_y + max_y) / 2.0;

    // Super-triangle vertices stored after the real points
    let st0 = [mid_x - 20.0 * delta_max, mid_y - delta_max];
    let st1 = [mid_x, mid_y + 20.0 * delta_max];
    let st2 = [mid_x + 20.0 * delta_max, mid_y - delta_max];
    let super_indices = [n, n + 1, n + 2];

    // Extended point list (original + super-triangle)
    let mut pts: Vec<[f64; 2]> = points.to_vec();
    pts.push(st0);
    pts.push(st1);
    pts.push(st2);

    let mut triangles: HashSet<Triangle> = HashSet::from([Triangle::new(
        super_indices[0],
        super_indices[1],
        super_indices[2],
    )]);

    for (idx, &p) in points.iter().enumerate() {
        // Find all triangles whose circumcircle contains p
        let bad: Vec<Triangle> = triangles
            .iter()
            .filter(|t| in_circumcircle(p, pts[t.v[0]], pts[t.v[1]], pts[t.v[2]]))
            .cloned()
            .collect();

        // Find the boundary polygon of the hole (edges NOT shared by two bad triangles)
        let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
        for t in &bad {
            let edges = [(t.v[0], t.v[1]), (t.v[0], t.v[2]), (t.v[1], t.v[2])];
            for (a, b) in edges {
                let key = (a.min(b), a.max(b));
                *edge_count.entry(key).or_insert(0) += 1;
            }
        }
        let boundary: Vec<(usize, usize)> = edge_count
            .into_iter()
            .filter(|(_, cnt)| *cnt == 1)
            .map(|(e, _)| e)
            .collect();

        // Remove bad triangles
        for t in &bad {
            triangles.remove(t);
        }

        // Re-triangulate with new point
        for (a, b) in boundary {
            triangles.insert(Triangle::new(idx, a, b));
        }
    }

    // Remove triangles that share vertices with the super-triangle
    triangles
        .into_iter()
        .filter(|t| !t.v.iter().any(|&v| v >= n))
        .collect()
}

// ─── AlphaComplex ─────────────────────────────────────────────────────────────

/// Alpha complex built from a 2D point cloud via Delaunay triangulation.
#[derive(Debug, Clone)]
pub struct AlphaComplex {
    /// The input 2D points.
    pub points: Vec<[f64; 2]>,
    /// All simplices sorted by filtration value.
    pub simplices: Vec<Simplex>,
}

impl AlphaComplex {
    /// Construct an alpha complex from a set of 2D points.
    ///
    /// Uses the Bowyer-Watson algorithm to build the Delaunay triangulation.
    /// Filtration values are circumradii (half edge-length for 1-simplices,
    /// circumradius for 2-simplices).
    pub fn new(points: &[[f64; 2]]) -> Self {
        let n = points.len();
        let mut simplex_map: HashMap<Vec<usize>, f64> = HashMap::new();

        // 0-simplices: filtration 0
        for i in 0..n {
            simplex_map.insert(vec![i], 0.0);
        }

        if n >= 2 {
            let triangles = bowyer_watson(points);

            // Build edge set from Delaunay triangulation
            let mut edges: HashSet<(usize, usize)> = HashSet::new();
            for t in &triangles {
                edges.insert((t.v[0].min(t.v[1]), t.v[0].max(t.v[1])));
                edges.insert((t.v[0].min(t.v[2]), t.v[0].max(t.v[2])));
                edges.insert((t.v[1].min(t.v[2]), t.v[1].max(t.v[2])));
            }

            // 1-simplices: half edge length as filtration value
            for (a, b) in &edges {
                let d = dist2d(points[*a], points[*b]) / 2.0;
                let key = vec![*a, *b];
                let entry = simplex_map.entry(key).or_insert(f64::INFINITY);
                if d < *entry {
                    *entry = d;
                }
            }

            // 2-simplices: circumradius of triangle
            for t in &triangles {
                let a = t.v[0];
                let b = t.v[1];
                let c = t.v[2];
                let r = circumradius_2d(points[a], points[b], points[c]);
                // Filtration value = max(circumradius, max edge filtration)
                let e_ab = simplex_map
                    .get(&vec![a.min(b), a.max(b)])
                    .copied()
                    .unwrap_or(0.0);
                let e_ac = simplex_map
                    .get(&vec![a.min(c), a.max(c)])
                    .copied()
                    .unwrap_or(0.0);
                let e_bc = simplex_map
                    .get(&vec![b.min(c), b.max(c)])
                    .copied()
                    .unwrap_or(0.0);
                let fv = r.max(e_ab).max(e_ac).max(e_bc);
                simplex_map.insert(vec![a, b, c], fv);
            }
        }

        // Convert to Vec<Simplex> and sort
        let mut simplices: Vec<Simplex> = simplex_map
            .into_iter()
            .map(|(vertices, fv)| Simplex {
                vertices,
                filtration_value: fv,
            })
            .collect();

        // Sort: primary by filtration value, secondary by dimension (ascending), then by vertices
        simplices.sort_by(|a, b| {
            a.filtration_value
                .partial_cmp(&b.filtration_value)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.vertices.len().cmp(&b.vertices.len()))
                .then(a.vertices.cmp(&b.vertices))
        });

        Self {
            points: points.to_vec(),
            simplices,
        }
    }

    /// Return all simplices with filtration value ≤ alpha.
    pub fn filtration(&self, alpha: f64) -> Vec<Simplex> {
        self.simplices
            .iter()
            .filter(|s| s.filtration_value <= alpha)
            .cloned()
            .collect()
    }

    /// Compute persistence pairs (birth, death, dimension) via boundary matrix reduction.
    ///
    /// Returns pairs sorted by birth value.
    pub fn persistence_pairs(&self) -> Vec<(f64, f64, usize)> {
        compute_persistence_from_simplices(&self.simplices)
    }
}

// ─── Boundary matrix reduction ────────────────────────────────────────────────

/// Compute persistence pairs from a filtered simplicial complex.
///
/// Uses the standard reduction algorithm (Edelsbrunner et al. 2002):
/// column j is reduced by subtracting the column whose pivot equals pivot(j).
///
/// Returns `(birth, death, dimension)` triples.
pub fn compute_persistence_from_simplices(simplices: &[Simplex]) -> Vec<(f64, f64, usize)> {
    let n = simplices.len();

    // Map each simplex to its index in the filtration
    let mut simplex_index: HashMap<Vec<usize>, usize> = HashMap::new();
    for (idx, s) in simplices.iter().enumerate() {
        simplex_index.insert(s.vertices.clone(), idx);
    }

    // Build boundary matrix as columns of sorted row indices (mod-2 representation)
    let mut columns: Vec<Vec<usize>> = simplices
        .iter()
        .map(|s| {
            let mut col: Vec<usize> = s
                .boundary_faces()
                .iter()
                .filter_map(|face| simplex_index.get(face).copied())
                .collect();
            col.sort_unstable();
            col
        })
        .collect();

    // Standard column reduction
    let mut pivot_col: HashMap<usize, usize> = HashMap::new(); // pivot row -> column index
    let mut pairs: Vec<(f64, f64, usize)> = Vec::new();
    let mut paired: HashSet<usize> = HashSet::new();

    for j in 0..n {
        // Reduce column j
        while let Some(&pivot) = columns[j].last() {
            if let Some(&k) = pivot_col.get(&pivot) {
                // XOR columns (mod-2 addition)
                let col_k = columns[k].clone();
                sym_diff_sorted(&mut columns[j], &col_k);
            } else {
                break;
            }
        }

        if let Some(&pivot) = columns[j].last() {
            // Column j has pivot at row `pivot`
            pivot_col.insert(pivot, j);
            let birth_idx = pivot;
            let death_idx = j;
            let dim = simplices[birth_idx].dimension();
            pairs.push((
                simplices[birth_idx].filtration_value,
                simplices[death_idx].filtration_value,
                dim,
            ));
            paired.insert(birth_idx);
            paired.insert(death_idx);
        }
    }

    // Essential features (unpaired simplices) get death = infinity
    // We skip them as the task requests only finite pairs from the matrix reduction

    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    pairs
}

/// Symmetric difference of two sorted vectors (mod-2 addition of columns).
pub fn sym_diff_sorted(a: &mut Vec<usize>, b: &[usize]) {
    let mut result: Vec<usize> = Vec::new();
    let mut ai = 0;
    let mut bi = 0;
    while ai < a.len() && bi < b.len() {
        match a[ai].cmp(&b[bi]) {
            std::cmp::Ordering::Less => {
                result.push(a[ai]);
                ai += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[bi]);
                bi += 1;
            }
            std::cmp::Ordering::Equal => {
                // Same element: cancel in mod-2
                ai += 1;
                bi += 1;
            }
        }
    }
    result.extend_from_slice(&a[ai..]);
    result.extend_from_slice(&b[bi..]);
    *a = result;
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Equilateral triangle with side length 1.0.
    fn equilateral_pts() -> Vec<[f64; 2]> {
        vec![[0.0, 0.0], [1.0, 0.0], [0.5, 3_f64.sqrt() / 2.0]]
    }

    #[test]
    fn test_circumradius_equilateral() {
        // Equilateral triangle with side 1: circumradius = 1/sqrt(3)
        let pts = equilateral_pts();
        let r = circumradius_2d(pts[0], pts[1], pts[2]);
        let expected = 1.0 / 3_f64.sqrt();
        assert!(
            (r - expected).abs() < 1e-10,
            "circumradius={r}, expected={expected}"
        );
    }

    #[test]
    fn test_circumradius_right_triangle() {
        // Right triangle with legs 3,4 — hypotenuse 5 — circumradius = hypotenuse/2 = 2.5
        let a = [0.0, 0.0];
        let b = [3.0, 0.0];
        let c = [0.0, 4.0];
        let r = circumradius_2d(a, b, c);
        assert!((r - 2.5).abs() < 1e-10, "circumradius={r}");
    }

    #[test]
    fn test_filtration_threshold() {
        // 5 points: a square + center
        let pts: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]];
        let ac = AlphaComplex::new(&pts);

        // At alpha=0 only vertices should be present
        let f0 = ac.filtration(0.0);
        assert!(f0.iter().all(|s| s.dimension() == 0));
        assert_eq!(f0.len(), 5);

        // At alpha=infinity all simplices returned
        let f_inf = ac.filtration(f64::INFINITY);
        assert!(!f_inf.is_empty());
        // All 5 vertices present
        assert!(f_inf.iter().filter(|s| s.dimension() == 0).count() >= 5);
    }

    #[test]
    fn test_persistence_pairs_triangle() {
        let pts = equilateral_pts();
        let ac = AlphaComplex::new(&pts);
        let pairs = ac.persistence_pairs();
        // For a filled triangle, we expect some pairs (the 2-simplex kills a 1-cycle)
        // Birth should be <= death for all pairs
        for (birth, death, _dim) in &pairs {
            assert!(birth <= death, "Invalid pair: birth={birth}, death={death}");
        }
    }

    #[test]
    fn test_persistence_pairs_not_empty_for_triangle() {
        let pts = equilateral_pts();
        let ac = AlphaComplex::new(&pts);
        let pairs = ac.persistence_pairs();
        // There must be at least one pair (connecting the 3 vertices)
        assert!(!pairs.is_empty(), "Expected at least one persistence pair");
    }

    #[test]
    fn test_five_point_cloud() {
        let pts: Vec<[f64; 2]> = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0]];
        let ac = AlphaComplex::new(&pts);
        // Should have 5 vertices
        assert_eq!(
            ac.simplices.iter().filter(|s| s.dimension() == 0).count(),
            5
        );
        // Edges exist
        assert!(
            ac.simplices.iter().any(|s| s.dimension() == 1),
            "Expected edges"
        );
    }
}
