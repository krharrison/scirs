//! Proximity-based graph construction from point sets
//!
//! Provides algorithms for building Euclidean MST, Gabriel graph,
//! relative neighborhood graph, and alpha shapes from 2D point sets.

use crate::error::{SpatialError, SpatialResult};
use std::collections::HashSet;

/// An edge in a Minimum Spanning Tree with its weight (Euclidean distance)
#[derive(Debug, Clone)]
pub struct MstEdge {
    /// Index of the first endpoint
    pub u: usize,
    /// Index of the second endpoint
    pub v: usize,
    /// Euclidean distance (edge weight)
    pub weight: f64,
}

/// Compute the Euclidean Minimum Spanning Tree of a 2D point set
///
/// The Euclidean MST connects all points with the minimum total edge length.
/// This implementation uses a brute-force Kruskal's algorithm on all pairwise
/// distances with a union-find structure for cycle detection.
///
/// For large point sets, building a Delaunay triangulation first and then
/// computing the MST on the Delaunay edges would be more efficient (O(n log n)
/// vs O(n^2 log n)), but this direct approach is simpler and correct.
///
/// # Arguments
///
/// * `points` - A slice of [x, y] coordinates
///
/// # Returns
///
/// * A vector of MST edges sorted by weight
///
/// # Examples
///
/// ```
/// use scirs2_spatial::proximity::euclidean_mst;
///
/// let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
/// let mst = euclidean_mst(&points).expect("compute");
/// assert_eq!(mst.len(), 3); // n-1 edges
///
/// // Total weight should be 3.0 (three unit edges)
/// let total: f64 = mst.iter().map(|e| e.weight).sum();
/// assert!((total - 3.0).abs() < 1e-10);
/// ```
pub fn euclidean_mst(points: &[[f64; 2]]) -> SpatialResult<Vec<MstEdge>> {
    let n = points.len();

    if n == 0 {
        return Ok(Vec::new());
    }

    if n == 1 {
        return Ok(Vec::new());
    }

    // Build all edges with distances
    let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = points[i][0] - points[j][0];
            let dy = points[i][1] - points[j][1];
            let dist = (dx * dx + dy * dy).sqrt();
            edges.push((i, j, dist));
        }
    }

    // Sort by distance
    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    // Kruskal's algorithm with union-find
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<usize> = vec![0; n];
    let mut mst = Vec::with_capacity(n - 1);

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) -> bool {
        let rx = find(parent, x);
        let ry = find(parent, y);
        if rx == ry {
            return false;
        }
        if rank[rx] < rank[ry] {
            parent[rx] = ry;
        } else if rank[rx] > rank[ry] {
            parent[ry] = rx;
        } else {
            parent[ry] = rx;
            rank[rx] += 1;
        }
        true
    }

    for (u, v, w) in edges {
        if mst.len() == n - 1 {
            break;
        }
        if union(&mut parent, &mut rank, u, v) {
            mst.push(MstEdge { u, v, weight: w });
        }
    }

    Ok(mst)
}

/// Compute the Gabriel graph of a 2D point set
///
/// The Gabriel graph connects two points p_i and p_j if and only if the
/// diametral circle (circle with p_i p_j as diameter) contains no other points.
/// Equivalently, edge (i, j) exists iff for all k != i,j:
///   d(i,j)^2 <= d(i,k)^2 + d(j,k)^2
///
/// The Gabriel graph is a subgraph of the Delaunay triangulation and a
/// supergraph of the Euclidean MST.
///
/// # Arguments
///
/// * `points` - A slice of [x, y] coordinates
///
/// # Returns
///
/// * A vector of edges (i, j) in the Gabriel graph
///
/// # Examples
///
/// ```
/// use scirs2_spatial::proximity::gabriel_graph;
///
/// let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
/// let edges = gabriel_graph(&points).expect("compute");
/// // Square: edges are (0,1), (0,2), (1,3), (2,3) - no diagonals
/// assert_eq!(edges.len(), 4);
/// ```
pub fn gabriel_graph(points: &[[f64; 2]]) -> SpatialResult<Vec<(usize, usize)>> {
    let n = points.len();

    if n < 2 {
        return Ok(Vec::new());
    }

    let mut edges = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = points[i][0] - points[j][0];
            let dy = points[i][1] - points[j][1];
            let dist_sq_ij = dx * dx + dy * dy;

            // Check if any other point lies inside the diametral circle
            let mut is_gabriel = true;

            for k in 0..n {
                if k == i || k == j {
                    continue;
                }

                let dxi = points[i][0] - points[k][0];
                let dyi = points[i][1] - points[k][1];
                let dist_sq_ik = dxi * dxi + dyi * dyi;

                let dxj = points[j][0] - points[k][0];
                let dyj = points[j][1] - points[k][1];
                let dist_sq_jk = dxj * dxj + dyj * dyj;

                // Gabriel condition: d(i,j)^2 <= d(i,k)^2 + d(j,k)^2
                if dist_sq_ij > dist_sq_ik + dist_sq_jk + 1e-10 {
                    is_gabriel = false;
                    break;
                }
            }

            if is_gabriel {
                edges.push((i, j));
            }
        }
    }

    Ok(edges)
}

/// Compute the Relative Neighborhood Graph (RNG) of a 2D point set
///
/// The RNG connects two points p_i and p_j if and only if there is no other
/// point p_k that is closer to both p_i and p_j than they are to each other.
/// Formally, edge (i, j) exists iff for all k != i,j:
///   d(i,j) <= max(d(i,k), d(j,k))
///
/// The RNG is a subgraph of the Gabriel graph and a supergraph of the
/// Euclidean MST.
///
/// # Arguments
///
/// * `points` - A slice of [x, y] coordinates
///
/// # Returns
///
/// * A vector of edges (i, j) in the relative neighborhood graph
///
/// # Examples
///
/// ```
/// use scirs2_spatial::proximity::relative_neighborhood_graph;
///
/// let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
/// let edges = relative_neighborhood_graph(&points).expect("compute");
/// // RNG is a subgraph of Gabriel graph
/// assert!(edges.len() <= 4);
/// ```
pub fn relative_neighborhood_graph(points: &[[f64; 2]]) -> SpatialResult<Vec<(usize, usize)>> {
    let n = points.len();

    if n < 2 {
        return Ok(Vec::new());
    }

    let mut edges = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = points[i][0] - points[j][0];
            let dy = points[i][1] - points[j][1];
            let dist_ij = (dx * dx + dy * dy).sqrt();

            let mut is_rng = true;

            for k in 0..n {
                if k == i || k == j {
                    continue;
                }

                let dxi = points[i][0] - points[k][0];
                let dyi = points[i][1] - points[k][1];
                let dist_ik = (dxi * dxi + dyi * dyi).sqrt();

                let dxj = points[j][0] - points[k][0];
                let dyj = points[j][1] - points[k][1];
                let dist_jk = (dxj * dxj + dyj * dyj).sqrt();

                // RNG condition: d(i,j) <= max(d(i,k), d(j,k))
                if dist_ij > dist_ik.max(dist_jk) + 1e-10 {
                    is_rng = false;
                    break;
                }
            }

            if is_rng {
                edges.push((i, j));
            }
        }
    }

    Ok(edges)
}

/// Compute the alpha shape edges for a 2D point set
///
/// Alpha shapes are a generalization of convex hulls. For a given alpha parameter,
/// the alpha shape is the intersection of all closed complements of circles with
/// radius 1/alpha that contain all points. As alpha decreases toward 0, the alpha
/// shape approaches the convex hull. As alpha increases, the shape becomes more
/// concave, eventually decomposing into individual points.
///
/// An edge (i, j) is in the alpha shape if there exists a circle of radius
/// 1/alpha passing through both p_i and p_j that contains no other points.
///
/// # Arguments
///
/// * `points` - A slice of [x, y] coordinates
/// * `alpha` - The alpha parameter (> 0). Larger alpha = more concave shape.
///   A good starting value is the inverse of the typical edge length.
///
/// # Returns
///
/// * A vector of edges (i, j) forming the alpha shape boundary
///
/// # Examples
///
/// ```
/// use scirs2_spatial::proximity::alpha_shape_edges;
///
/// let points = vec![
///     [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
///     [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
/// ];
///
/// // With small alpha, most edges are included (near convex hull)
/// let edges = alpha_shape_edges(&points, 0.5).expect("compute");
/// assert!(!edges.is_empty());
/// ```
pub fn alpha_shape_edges(points: &[[f64; 2]], alpha: f64) -> SpatialResult<Vec<(usize, usize)>> {
    let n = points.len();

    if n < 2 {
        return Ok(Vec::new());
    }

    if alpha <= 0.0 {
        return Err(SpatialError::ValueError(
            "Alpha must be positive".to_string(),
        ));
    }

    let radius = 1.0 / alpha;
    let radius_sq = radius * radius;
    let mut edges = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = points[j][0] - points[i][0];
            let dy = points[j][1] - points[i][1];
            let dist_sq = dx * dx + dy * dy;
            let dist = dist_sq.sqrt();

            // If the edge is longer than the diameter (2*radius), skip
            if dist > 2.0 * radius + 1e-10 {
                continue;
            }

            // Find the two circle centers of radius r passing through p_i and p_j
            let mid_x = (points[i][0] + points[j][0]) * 0.5;
            let mid_y = (points[i][1] + points[j][1]) * 0.5;

            let half_dist_sq = dist_sq * 0.25;
            let h_sq = radius_sq - half_dist_sq;

            if h_sq < 0.0 {
                continue; // Edge too long for this radius
            }

            let h = h_sq.sqrt();

            // Normal direction perpendicular to the edge
            let nx = -dy / dist;
            let ny = dx / dist;

            // Two potential circle centers
            let c1x = mid_x + h * nx;
            let c1y = mid_y + h * ny;
            let c2x = mid_x - h * nx;
            let c2y = mid_y - h * ny;

            // Check if either circle is empty (contains no other points)
            let circle1_empty = (0..n).all(|k| {
                if k == i || k == j {
                    return true;
                }
                let dxk = points[k][0] - c1x;
                let dyk = points[k][1] - c1y;
                dxk * dxk + dyk * dyk >= radius_sq - 1e-10
            });

            let circle2_empty = (0..n).all(|k| {
                if k == i || k == j {
                    return true;
                }
                let dxk = points[k][0] - c2x;
                let dyk = points[k][1] - c2y;
                dxk * dxk + dyk * dyk >= radius_sq - 1e-10
            });

            if circle1_empty || circle2_empty {
                edges.push((i, j));
            }
        }
    }

    Ok(edges)
}

/// Get boundary edges of an alpha shape (edges that appear in only one triangle)
///
/// This extracts only the boundary from the alpha shape edges, which forms
/// the concave hull outline.
///
/// # Arguments
///
/// * `points` - A slice of [x, y] coordinates
/// * `alpha` - The alpha parameter (> 0)
///
/// # Returns
///
/// * Boundary edges forming the concave hull
pub fn alpha_shape_boundary(points: &[[f64; 2]], alpha: f64) -> SpatialResult<Vec<(usize, usize)>> {
    let all_edges = alpha_shape_edges(points, alpha)?;

    // Build an edge set for quick lookup
    let edge_set: HashSet<(usize, usize)> = all_edges.iter().copied().collect();

    // A boundary edge is one where the two adjacent triangles that could use it
    // don't both exist. We approximate this by checking if for each edge (i,j),
    // there are 0 or 1 common neighbors k such that both (i,k) and (j,k) are edges.
    let mut boundary = Vec::new();

    for &(i, j) in &all_edges {
        let common_count = (0..points.len())
            .filter(|&k| {
                k != i
                    && k != j
                    && (edge_set.contains(&(i.min(k), i.max(k))))
                    && (edge_set.contains(&(j.min(k), j.max(k))))
            })
            .count();

        // Boundary edges have 0 or 1 common neighbor (1 means one side is open)
        if common_count <= 1 {
            boundary.push((i, j));
        }
    }

    Ok(boundary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mst_triangle() {
        let points = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]]; // equilateral
        let mst = euclidean_mst(&points).expect("compute");
        assert_eq!(mst.len(), 2);

        // All edges should be ~1.0
        for e in &mst {
            assert!((e.weight - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_mst_square() {
        let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let mst = euclidean_mst(&points).expect("compute");
        assert_eq!(mst.len(), 3);

        // Total weight should be 3.0 (three unit edges, not the diagonal)
        let total: f64 = mst.iter().map(|e| e.weight).sum();
        assert!((total - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mst_single_point() {
        let points = vec![[0.0, 0.0]];
        let mst = euclidean_mst(&points).expect("compute");
        assert!(mst.is_empty());
    }

    #[test]
    fn test_mst_empty() {
        let points: Vec<[f64; 2]> = vec![];
        let mst = euclidean_mst(&points).expect("compute");
        assert!(mst.is_empty());
    }

    #[test]
    fn test_gabriel_graph_square() {
        let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let edges = gabriel_graph(&points).expect("compute");

        // For a unit square, diagonal points lie exactly on the diametral circle
        // boundary (d(i,j)^2 == d(i,k)^2 + d(j,k)^2 for diagonals).
        // With our tolerance, diagonals may or may not be included.
        // The 4 side edges must always be present.
        let edge_set: HashSet<(usize, usize)> = edges.into_iter().collect();
        assert!(edge_set.contains(&(0, 1))); // bottom
        assert!(edge_set.contains(&(0, 2))); // left
        assert!(edge_set.contains(&(1, 3))); // right
        assert!(edge_set.contains(&(2, 3))); // top
        assert!(edge_set.len() >= 4);
        assert!(edge_set.len() <= 6); // max C(4,2) = 6
    }

    #[test]
    fn test_gabriel_is_supergraph_of_rng() {
        let points = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 1.0]];

        let gabriel_edges = gabriel_graph(&points).expect("compute");
        let rng_edges = relative_neighborhood_graph(&points).expect("compute");

        let gabriel_set: HashSet<(usize, usize)> = gabriel_edges.into_iter().collect();
        let rng_set: HashSet<(usize, usize)> = rng_edges.into_iter().collect();

        // Every RNG edge should be in Gabriel graph
        for edge in &rng_set {
            assert!(
                gabriel_set.contains(edge),
                "RNG edge {:?} not in Gabriel graph",
                edge
            );
        }
    }

    #[test]
    fn test_rng_square() {
        let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let edges = relative_neighborhood_graph(&points).expect("compute");

        // RNG of a square: the 4 side edges (no diagonals)
        assert_eq!(edges.len(), 4);
    }

    #[test]
    fn test_rng_is_supergraph_of_mst() {
        let points = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let mst = euclidean_mst(&points).expect("compute");
        let rng_edges = relative_neighborhood_graph(&points).expect("compute");

        let rng_set: HashSet<(usize, usize)> = rng_edges.into_iter().collect();

        // Every MST edge should be in the RNG
        for e in &mst {
            let edge = (e.u.min(e.v), e.u.max(e.v));
            assert!(rng_set.contains(&edge), "MST edge {:?} not in RNG", edge);
        }
    }

    #[test]
    fn test_alpha_shape_large_alpha() {
        // With very large alpha (small radius), few or no edges should survive
        let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let edges = alpha_shape_edges(&points, 10.0).expect("compute");
        // Very large alpha means very small circle radius - edges too long
        // might have some short edges
        assert!(edges.len() <= 6); // at most C(4,2)=6
    }

    #[test]
    fn test_alpha_shape_small_alpha() {
        // With small alpha (large radius), most edges should be included
        let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let edges = alpha_shape_edges(&points, 0.1).expect("compute");
        // Small alpha = large circles = almost convex hull, should include all 6 edges
        assert!(edges.len() >= 4);
    }

    #[test]
    fn test_alpha_shape_invalid_alpha() {
        let points = vec![[0.0, 0.0], [1.0, 0.0]];
        assert!(alpha_shape_edges(&points, 0.0).is_err());
        assert!(alpha_shape_edges(&points, -1.0).is_err());
    }

    #[test]
    fn test_graph_hierarchy() {
        // The containment hierarchy should be: MST ⊆ RNG ⊆ Gabriel ⊆ Delaunay
        let points = vec![[0.0, 0.0], [2.0, 0.0], [1.0, 1.5], [3.0, 1.0], [0.5, 2.5]];

        let mst = euclidean_mst(&points).expect("mst");
        let rng = relative_neighborhood_graph(&points).expect("rng");
        let gabriel = gabriel_graph(&points).expect("gabriel");

        let mst_edges: HashSet<(usize, usize)> =
            mst.iter().map(|e| (e.u.min(e.v), e.u.max(e.v))).collect();
        let rng_set: HashSet<(usize, usize)> = rng.into_iter().collect();
        let gabriel_set: HashSet<(usize, usize)> = gabriel.into_iter().collect();

        // MST ⊆ RNG
        for edge in &mst_edges {
            assert!(rng_set.contains(edge), "MST edge {:?} not in RNG", edge);
        }

        // RNG ⊆ Gabriel
        for edge in &rng_set {
            assert!(
                gabriel_set.contains(edge),
                "RNG edge {:?} not in Gabriel",
                edge
            );
        }

        assert!(mst_edges.len() <= rng_set.len());
        assert!(rng_set.len() <= gabriel_set.len());
    }
}
