//! Cluster tree for hierarchical matrix (H-matrix) construction.
//!
//! This module implements the cluster tree data structure used to organize
//! row and column indices into a binary tree of clusters for H-matrix assembly.
//! Each cluster node is associated with a bounding box in a geometric coordinate
//! space.  The admissibility criterion (η-admissibility) determines whether a
//! pair of clusters can be approximated with a low-rank block.
//!
//! # References
//!
//! - Hackbusch, W. (2015): *Hierarchical Matrices: Algorithms and Analysis*,
//!   Springer Series in Computational Mathematics, Vol. 49.
//! - Bebendorf, M. (2008): *Hierarchical Matrices*, Springer Lecture Notes
//!   in Computational Science and Engineering, Vol. 63.

use crate::error::{SparseError, SparseResult};

// ---------------------------------------------------------------------------
// BoundingBox
// ---------------------------------------------------------------------------

/// An axis-aligned bounding box in d-dimensional space.
///
/// Used to represent the geometric extent of an index cluster.
#[derive(Debug, Clone)]
pub struct BoundingBox {
    /// Minimum coordinates (length = spatial dimension d).
    pub min: Vec<f64>,
    /// Maximum coordinates (length = spatial dimension d).
    pub max: Vec<f64>,
}

impl BoundingBox {
    /// Construct a new bounding box.
    ///
    /// # Errors
    /// Returns an error if `min` and `max` have different lengths or if any
    /// `min[i] > max[i]`.
    pub fn new(min: Vec<f64>, max: Vec<f64>) -> SparseResult<Self> {
        if min.len() != max.len() {
            return Err(SparseError::ValueError(format!(
                "BoundingBox: min length {} != max length {}",
                min.len(),
                max.len()
            )));
        }
        for (i, (&lo, &hi)) in min.iter().zip(max.iter()).enumerate() {
            if lo > hi {
                return Err(SparseError::ValueError(format!(
                    "BoundingBox: min[{}]={} > max[{}]={}",
                    i, lo, i, hi
                )));
            }
        }
        Ok(Self { min, max })
    }

    /// Return the spatial dimension of this bounding box.
    pub fn dim(&self) -> usize {
        self.min.len()
    }

    /// Diameter (longest edge length / Euclidean diagonal).
    ///
    /// diam = sqrt( sum_i (max_i - min_i)^2 )
    pub fn diameter(&self) -> f64 {
        self.min
            .iter()
            .zip(self.max.iter())
            .map(|(&lo, &hi)| {
                let d = hi - lo;
                d * d
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Distance from this bounding box to another.
    ///
    /// dist(τ, σ) = sqrt( sum_i max(0, max(min_τ[i]-max_σ[i], min_σ[i]-max_τ[i]))^2 )
    pub fn distance_to(&self, other: &BoundingBox) -> f64 {
        self.min
            .iter()
            .zip(self.max.iter())
            .zip(other.min.iter().zip(other.max.iter()))
            .map(|((&lo_s, &hi_s), (&lo_o, &hi_o))| {
                let gap = f64::max(0.0, f64::max(lo_s - hi_o, lo_o - hi_s));
                gap * gap
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Compute the bounding box of two bounding boxes (union hull).
    pub fn union(&self, other: &BoundingBox) -> SparseResult<BoundingBox> {
        if self.dim() != other.dim() {
            return Err(SparseError::ValueError(format!(
                "BoundingBox::union: dimension mismatch {} vs {}",
                self.dim(),
                other.dim()
            )));
        }
        let min = self
            .min
            .iter()
            .zip(other.min.iter())
            .map(|(&a, &b)| f64::min(a, b))
            .collect();
        let max = self
            .max
            .iter()
            .zip(other.max.iter())
            .map(|(&a, &b)| f64::max(a, b))
            .collect();
        BoundingBox::new(min, max)
    }

    /// Center of the bounding box.
    pub fn center(&self) -> Vec<f64> {
        self.min
            .iter()
            .zip(self.max.iter())
            .map(|(&lo, &hi)| 0.5 * (lo + hi))
            .collect()
    }

    /// Index of the widest dimension (longest edge).
    pub fn widest_dim(&self) -> usize {
        self.min
            .iter()
            .zip(self.max.iter())
            .enumerate()
            .max_by(|(_, (lo1, hi1)), (_, (lo2, hi2))| {
                let d1 = *hi1 - *lo1;
                let d2 = *hi2 - *lo2;
                d1.partial_cmp(&d2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// ClusterNode
// ---------------------------------------------------------------------------

/// A node in the cluster tree.
///
/// Leaf nodes hold a contiguous slice of (sorted) indices.
/// Internal nodes hold two children and carry the union of their bounding
/// boxes.
#[derive(Debug, Clone)]
pub struct ClusterNode {
    /// Global row/column indices belonging to this cluster.
    pub indices: Vec<usize>,
    /// Bounding box enclosing all point coordinates of indices in this cluster.
    pub bbox: BoundingBox,
    /// Index into the `nodes` arena of the left child, or `None` for a leaf.
    pub left: Option<usize>,
    /// Index into the `nodes` arena of the right child, or `None` for a leaf.
    pub right: Option<usize>,
    /// Depth of this node in the tree (root = 0).
    pub depth: usize,
}

impl ClusterNode {
    /// Returns `true` if this node is a leaf (no children).
    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}

// ---------------------------------------------------------------------------
// ClusterTree
// ---------------------------------------------------------------------------

/// Binary cluster tree over a set of d-dimensional points.
///
/// The tree is stored as an arena (`Vec<ClusterNode>`).  The root is always
/// at index 0.
///
/// # Construction
///
/// ```rust
/// use scirs2_sparse::hierarchical::cluster_tree::{ClusterTree, build_cluster_tree};
///
/// let coords: Vec<[f64; 2]> = (0..8)
///     .map(|i| [i as f64, 0.0_f64])
///     .collect();
/// let flat: Vec<f64> = coords.iter().flat_map(|p| p.iter().copied()).collect();
/// let tree = build_cluster_tree(&flat, 2, 2).expect("build failed");
/// assert!(tree.root_idx() == 0);
/// ```
#[derive(Debug, Clone)]
pub struct ClusterTree {
    /// Arena of cluster nodes.
    pub nodes: Vec<ClusterNode>,
    /// Spatial dimension of the point set.
    pub dim: usize,
    /// Minimum number of indices in a leaf node.
    pub leaf_size: usize,
}

impl ClusterTree {
    /// Return the index of the root node (always 0).
    pub fn root_idx(&self) -> usize {
        0
    }

    /// Return a reference to the root node.
    pub fn root(&self) -> &ClusterNode {
        &self.nodes[0]
    }

    /// Total number of nodes in the tree (leaves + internals).
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Collect all leaf-node indices.
    pub fn leaf_indices(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.is_leaf())
            .map(|(i, _)| i)
            .collect()
    }

    /// Depth of the tree (maximum depth over all nodes).
    pub fn depth(&self) -> usize {
        self.nodes.iter().map(|n| n.depth).max().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// build_cluster_tree  (public entry point)
// ---------------------------------------------------------------------------

/// Build a binary cluster tree by recursive geometric bisection.
///
/// The algorithm splits the current index set along the widest spatial
/// dimension at the median coordinate value.  Splitting continues until a
/// node contains ≤ `leaf_size` indices.
///
/// # Parameters
/// - `coords`: flat array of length `n * dim` with point coordinates in
///   row-major order (point 0 = coords[0..dim], point 1 = coords[dim..2*dim], …).
/// - `dim`: spatial dimension (must be ≥ 1).
/// - `leaf_size`: maximum number of indices in a leaf (must be ≥ 1).
///
/// # Errors
/// - If `coords.len() % dim != 0`.
/// - If `leaf_size == 0`.
/// - If `dim == 0`.
///
/// # Example
/// ```rust
/// use scirs2_sparse::hierarchical::cluster_tree::build_cluster_tree;
/// let coords: Vec<f64> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
/// let tree = build_cluster_tree(&coords, 2, 2).expect("ok");
/// assert_eq!(tree.num_nodes(), 3); // root + 2 leaves
/// ```
pub fn build_cluster_tree(
    coords: &[f64],
    dim: usize,
    leaf_size: usize,
) -> SparseResult<ClusterTree> {
    if dim == 0 {
        return Err(SparseError::ValueError(
            "build_cluster_tree: dim must be >= 1".to_string(),
        ));
    }
    if leaf_size == 0 {
        return Err(SparseError::ValueError(
            "build_cluster_tree: leaf_size must be >= 1".to_string(),
        ));
    }
    if coords.len() % dim != 0 {
        return Err(SparseError::ValueError(format!(
            "build_cluster_tree: coords length {} is not divisible by dim {}",
            coords.len(),
            dim
        )));
    }
    let n = coords.len() / dim;
    if n == 0 {
        return Err(SparseError::ValueError(
            "build_cluster_tree: empty coordinate set".to_string(),
        ));
    }

    // Pre-compute per-point coordinate slices as owned vecs for convenience.
    let points: Vec<Vec<f64>> = (0..n)
        .map(|i| coords[i * dim..(i + 1) * dim].to_vec())
        .collect();

    // Initial index set
    let all_indices: Vec<usize> = (0..n).collect();

    // Pre-compute global bounding box
    let root_bbox = compute_bbox(&points, &all_indices)?;

    let mut tree = ClusterTree {
        nodes: Vec::new(),
        dim,
        leaf_size,
    };

    // Reserve slot 0 for root (will be filled by recursive build).
    // We use a stack-based approach to avoid deep recursion.
    build_recursive(&mut tree, &points, all_indices, root_bbox, 0)?;

    Ok(tree)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Recursively build the cluster tree, appending nodes into `tree.nodes`.
/// Returns the arena index of the newly inserted node.
fn build_recursive(
    tree: &mut ClusterTree,
    points: &[Vec<f64>],
    indices: Vec<usize>,
    bbox: BoundingBox,
    depth: usize,
) -> SparseResult<usize> {
    let node_idx = tree.nodes.len();

    // Push a placeholder so that node_idx is stable.
    tree.nodes.push(ClusterNode {
        indices: indices.clone(),
        bbox: bbox.clone(),
        left: None,
        right: None,
        depth,
    });

    if indices.len() <= tree.leaf_size {
        // Leaf: nothing more to do.
        return Ok(node_idx);
    }

    // Split along the widest dimension at the median.
    let split_dim = bbox.widest_dim();
    let (left_indices, right_indices) = split_indices(points, &indices, split_dim);

    if left_indices.is_empty() || right_indices.is_empty() {
        // Degenerate split (all points coincide in that dimension): make leaf.
        return Ok(node_idx);
    }

    let left_bbox = compute_bbox(points, &left_indices)?;
    let right_bbox = compute_bbox(points, &right_indices)?;

    let left_child = build_recursive(tree, points, left_indices, left_bbox, depth + 1)?;
    let right_child = build_recursive(tree, points, right_indices, right_bbox, depth + 1)?;

    // Update the previously inserted placeholder node.
    tree.nodes[node_idx].left = Some(left_child);
    tree.nodes[node_idx].right = Some(right_child);

    Ok(node_idx)
}

/// Compute the bounding box of a set of `points` indexed by `indices`.
fn compute_bbox(points: &[Vec<f64>], indices: &[usize]) -> SparseResult<BoundingBox> {
    if indices.is_empty() {
        return Err(SparseError::ValueError(
            "compute_bbox: empty index set".to_string(),
        ));
    }
    let dim = points[indices[0]].len();
    let mut min_coords = vec![f64::INFINITY; dim];
    let mut max_coords = vec![f64::NEG_INFINITY; dim];
    for &idx in indices {
        for (d, &coord) in points[idx].iter().enumerate() {
            if coord < min_coords[d] {
                min_coords[d] = coord;
            }
            if coord > max_coords[d] {
                max_coords[d] = coord;
            }
        }
    }
    // Collapse degenerate dimensions (single point => min == max is valid).
    BoundingBox::new(min_coords, max_coords)
}

/// Split an index set into two halves by the median coordinate in `split_dim`.
///
/// Indices with coordinate ≤ median go left; the rest go right.
/// If all coordinates are equal, the split produces an empty right set
/// (caller should treat this as a leaf).
fn split_indices(
    points: &[Vec<f64>],
    indices: &[usize],
    split_dim: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut coords: Vec<(f64, usize)> = indices
        .iter()
        .map(|&i| (points[i][split_dim], i))
        .collect();
    // Sort by coordinate value.
    coords.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mid = coords.len() / 2;
    let left: Vec<usize> = coords[..mid].iter().map(|(_, i)| *i).collect();
    let right: Vec<usize> = coords[mid..].iter().map(|(_, i)| *i).collect();
    (left, right)
}

// ---------------------------------------------------------------------------
// admissibility_check
// ---------------------------------------------------------------------------

/// η-admissibility condition for a pair of cluster bounding boxes.
///
/// A cluster pair (τ, σ) is *admissible* if
///
/// ```text
/// min(diam(τ), diam(σ))  ≤  η · dist(τ, σ)
/// ```
///
/// Admissible pairs can be approximated by a low-rank block; non-admissible
/// pairs must be subdivided further or stored as dense blocks.
///
/// # Parameters
/// - `tau`: bounding box of the row cluster τ.
/// - `sigma`: bounding box of the column cluster σ.
/// - `eta`: admissibility parameter (typical values 0.5–2.0; larger η → more
///   admissible pairs but potentially less accurate low-rank approximations).
///
/// # Returns
/// `true` if the pair is admissible, `false` otherwise.
///
/// # Example
/// ```rust
/// use scirs2_sparse::hierarchical::cluster_tree::{BoundingBox, admissibility_check};
///
/// let tau   = BoundingBox::new(vec![0.0], vec![1.0]).expect("valid input");
/// let sigma = BoundingBox::new(vec![5.0], vec![6.0]).expect("valid input");
/// assert!(admissibility_check(&tau, &sigma, 1.0));  // min(1, 1) = 1 ≤ 1 * 4 = 4
/// ```
pub fn admissibility_check(tau: &BoundingBox, sigma: &BoundingBox, eta: f64) -> bool {
    let diam_tau = tau.diameter();
    let diam_sigma = sigma.diameter();
    let dist = tau.distance_to(sigma);

    // Guard: if clusters overlap (dist == 0), they are never admissible.
    if dist <= 0.0 {
        return false;
    }

    let min_diam = f64::min(diam_tau, diam_sigma);
    min_diam <= eta * dist
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_1d_tree(n: usize, leaf_size: usize) -> ClusterTree {
        let coords: Vec<f64> = (0..n).map(|i| i as f64).collect();
        build_cluster_tree(&coords, 1, leaf_size).expect("build failed")
    }

    #[test]
    fn test_bounding_box_diameter() {
        let bb = BoundingBox::new(vec![0.0, 0.0], vec![3.0, 4.0]).expect("ok");
        let d = bb.diameter();
        assert!((d - 5.0).abs() < 1e-12, "diameter = {}", d);
    }

    #[test]
    fn test_bounding_box_distance() {
        let a = BoundingBox::new(vec![0.0], vec![1.0]).expect("ok");
        let b = BoundingBox::new(vec![3.0], vec![4.0]).expect("ok");
        let d = a.distance_to(&b);
        assert!((d - 2.0).abs() < 1e-12, "distance = {}", d);
    }

    #[test]
    fn test_bounding_box_overlapping_distance() {
        let a = BoundingBox::new(vec![0.0], vec![2.0]).expect("ok");
        let b = BoundingBox::new(vec![1.0], vec![3.0]).expect("ok");
        let d = a.distance_to(&b);
        assert_eq!(d, 0.0, "overlapping boxes should have distance 0");
    }

    #[test]
    fn test_build_cluster_tree_leaf() {
        // Exactly leaf_size points → root should be a leaf.
        let tree = make_1d_tree(2, 2);
        assert_eq!(tree.num_nodes(), 1);
        assert!(tree.root().is_leaf());
    }

    #[test]
    fn test_build_cluster_tree_split() {
        // 4 points, leaf_size=2 → root + 2 leaves = 3 nodes.
        let tree = make_1d_tree(4, 2);
        assert_eq!(tree.num_nodes(), 3, "nodes={}", tree.num_nodes());
        assert!(!tree.root().is_leaf());
    }

    #[test]
    fn test_build_cluster_tree_large() {
        let tree = make_1d_tree(16, 2);
        // All indices should be covered exactly once across leaves.
        let leaves = tree.leaf_indices();
        let mut all: Vec<usize> = leaves
            .iter()
            .flat_map(|&li| tree.nodes[li].indices.iter().copied())
            .collect();
        all.sort_unstable();
        let expected: Vec<usize> = (0..16).collect();
        assert_eq!(all, expected, "indices not covered: {:?}", all);
    }

    #[test]
    fn test_admissibility_check() {
        // Well-separated clusters in 1D.
        let tau = BoundingBox::new(vec![0.0], vec![1.0]).expect("ok");
        let sigma = BoundingBox::new(vec![5.0], vec![6.0]).expect("ok");
        // min(1, 1) = 1 ≤ 1.0 * 4 = 4 → admissible
        assert!(admissibility_check(&tau, &sigma, 1.0));
    }

    #[test]
    fn test_admissibility_check_not_admissible() {
        // Adjacent clusters.
        let tau = BoundingBox::new(vec![0.0], vec![1.0]).expect("ok");
        let sigma = BoundingBox::new(vec![1.5], vec![2.5]).expect("ok");
        // diam=1, dist=0.5 → min(1,1)=1 > 0.5*0.5=0.25  → NOT admissible for η=0.5
        assert!(!admissibility_check(&tau, &sigma, 0.5));
    }

    #[test]
    fn test_admissibility_overlapping() {
        // Overlapping clusters are never admissible.
        let tau = BoundingBox::new(vec![0.0], vec![2.0]).expect("ok");
        let sigma = BoundingBox::new(vec![1.0], vec![3.0]).expect("ok");
        assert!(!admissibility_check(&tau, &sigma, 10.0));
    }

    #[test]
    fn test_2d_cluster_tree() {
        // 2D grid of points.
        let mut coords = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                coords.push(i as f64);
                coords.push(j as f64);
            }
        }
        let tree = build_cluster_tree(&coords, 2, 2).expect("build failed");
        let leaves = tree.leaf_indices();
        let mut all: Vec<usize> = leaves
            .iter()
            .flat_map(|&li| tree.nodes[li].indices.iter().copied())
            .collect();
        all.sort_unstable();
        let expected: Vec<usize> = (0..16).collect();
        assert_eq!(all, expected);
    }

    #[test]
    fn test_error_dim_zero() {
        let coords = vec![1.0, 2.0];
        let result = build_cluster_tree(&coords, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_leaf_size_zero() {
        let coords = vec![1.0, 2.0];
        let result = build_cluster_tree(&coords, 1, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_coords_not_divisible() {
        let coords = vec![1.0, 2.0, 3.0];
        let result = build_cluster_tree(&coords, 2, 1);
        assert!(result.is_err());
    }
}
