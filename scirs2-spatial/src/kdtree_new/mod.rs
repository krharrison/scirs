//! Advanced k-d tree with k-NN, radius search, and approximate nearest-neighbour.
//!
//! This module provides a clean, general-purpose k-d tree implementation that
//! supports:
//! - Exact k-nearest-neighbours (k-NN)
//! - Radius search (all points within distance r)
//! - (1+ε)-approximate nearest neighbour via defeatist search

use crate::error::{SpatialError, SpatialResult};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ──────────────────────────────────────────────────────────────────────────────
// Node
// ──────────────────────────────────────────────────────────────────────────────

/// A node in the k-d tree.
#[derive(Debug, Clone)]
pub struct KDNode {
    /// The point stored at this node.
    pub point: Vec<f64>,
    /// Original index into the input point array.
    pub orig_idx: usize,
    /// Left subtree.
    pub left: Option<Box<KDNode>>,
    /// Right subtree.
    pub right: Option<Box<KDNode>>,
    /// Splitting axis.
    pub axis: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// KD-Tree
// ──────────────────────────────────────────────────────────────────────────────

/// A k-d tree supporting k-NN, radius, and approximate NN queries.
#[derive(Debug, Clone)]
pub struct KDTree {
    pub root: Option<Box<KDNode>>,
    /// Dimensionality.
    pub dim: usize,
}

impl KDTree {
    /// Build a k-d tree from `points` using median-of-medians splitting.
    ///
    /// # Errors
    ///
    /// Returns an error if `points` is empty or points have inconsistent dimension.
    pub fn build(points: &[Vec<f64>]) -> SpatialResult<Self> {
        if points.is_empty() {
            return Err(SpatialError::InvalidInput("No points provided".into()));
        }
        let dim = points[0].len();
        if dim == 0 {
            return Err(SpatialError::InvalidInput("Points must be non-zero dimensional".into()));
        }
        for p in points.iter() {
            if p.len() != dim {
                return Err(SpatialError::InvalidInput(
                    "All points must have the same dimension".into(),
                ));
            }
        }

        let mut indexed: Vec<(usize, Vec<f64>)> =
            points.iter().cloned().enumerate().collect();

        let root = build_recursive(&mut indexed, dim, 0);
        Ok(KDTree { root, dim })
    }

    /// Find the k nearest neighbours of `query`.
    ///
    /// Returns a list of `(original_index, squared_distance)` sorted by distance (ascending).
    ///
    /// # Errors
    ///
    /// Returns an error if `query.len() != self.dim`.
    pub fn nearest_k(&self, query: &[f64], k: usize) -> SpatialResult<Vec<(usize, f64)>> {
        if query.len() != self.dim {
            return Err(SpatialError::InvalidInput(
                "Query dimension mismatch".into(),
            ));
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        // Max-heap of (dist², idx) – we keep the k smallest.
        let mut heap: BinaryHeap<(ordered_float::OrderedFloat<f64>, usize)> = BinaryHeap::new();
        knn_search(self.root.as_deref(), query, k, &mut heap);

        let mut results: Vec<(usize, f64)> = heap
            .into_iter()
            .map(|(d2, idx)| (idx, d2.into_inner()))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        Ok(results)
    }

    /// Find all points within Euclidean distance `r` of `query`.
    ///
    /// Returns a list of `original_index` values (unsorted).
    ///
    /// # Errors
    ///
    /// Returns an error if `query.len() != self.dim`.
    pub fn radius_search(&self, query: &[f64], r: f64) -> SpatialResult<Vec<usize>> {
        if query.len() != self.dim {
            return Err(SpatialError::InvalidInput(
                "Query dimension mismatch".into(),
            ));
        }
        let r2 = r * r;
        let mut results = Vec::new();
        radius_search_recursive(self.root.as_deref(), query, r2, &mut results);
        Ok(results)
    }

    /// Find the (1+ε)-approximate nearest neighbour of `query`.
    ///
    /// Uses defeatist search: do not backtrack if the candidate is within
    /// `(1+epsilon)` of the lower bound on the unexplored subtree.
    ///
    /// Returns `(original_index, distance)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `query.len() != self.dim` or the tree is empty.
    pub fn approx_nearest(&self, query: &[f64], epsilon: f64) -> SpatialResult<(usize, f64)> {
        if query.len() != self.dim {
            return Err(SpatialError::InvalidInput(
                "Query dimension mismatch".into(),
            ));
        }
        let root = self
            .root
            .as_deref()
            .ok_or_else(|| SpatialError::InvalidInput("Tree is empty".into()))?;

        let mut best_dist2 = f64::INFINITY;
        let mut best_idx = 0usize;
        approx_nn_search(root, query, epsilon, &mut best_dist2, &mut best_idx);
        Ok((best_idx, best_dist2.sqrt()))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Construction
// ──────────────────────────────────────────────────────────────────────────────

fn build_recursive(
    points: &mut [(usize, Vec<f64>)],
    dim: usize,
    depth: usize,
) -> Option<Box<KDNode>> {
    if points.is_empty() {
        return None;
    }
    let axis = depth % dim;

    // Median split.
    let median = points.len() / 2;
    points.select_nth_unstable_by(median, |a, b| {
        a.1[axis]
            .partial_cmp(&b.1[axis])
            .unwrap_or(Ordering::Equal)
    });

    let (orig_idx, point) = (points[median].0, points[median].1.clone());

    let left = build_recursive(&mut points[..median], dim, depth + 1);
    let right = build_recursive(&mut points[median + 1..], dim, depth + 1);

    Some(Box::new(KDNode {
        point,
        orig_idx,
        left,
        right,
        axis,
    }))
}

// ──────────────────────────────────────────────────────────────────────────────
// k-NN search
// ──────────────────────────────────────────────────────────────────────────────

fn knn_search(
    node: Option<&KDNode>,
    query: &[f64],
    k: usize,
    heap: &mut BinaryHeap<(ordered_float::OrderedFloat<f64>, usize)>,
) {
    let node = match node {
        Some(n) => n,
        None => return,
    };

    let d2 = sq_dist(query, &node.point);
    let od2 = ordered_float::OrderedFloat(d2);

    // Update heap.
    if heap.len() < k {
        heap.push((od2, node.orig_idx));
    } else if let Some(&(top, _)) = heap.peek() {
        if od2 < top {
            heap.pop();
            heap.push((od2, node.orig_idx));
        }
    }

    let axis = node.axis;
    let diff = query[axis] - node.point[axis];
    let (first, second) = if diff <= 0.0 {
        (node.left.as_deref(), node.right.as_deref())
    } else {
        (node.right.as_deref(), node.left.as_deref())
    };

    knn_search(first, query, k, heap);

    // Check if we need to explore the other side.
    let worst_dist2 = heap
        .peek()
        .map(|&(d, _)| d.into_inner())
        .unwrap_or(f64::INFINITY);
    if diff * diff < worst_dist2 || heap.len() < k {
        knn_search(second, query, k, heap);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Radius search
// ──────────────────────────────────────────────────────────────────────────────

fn radius_search_recursive(
    node: Option<&KDNode>,
    query: &[f64],
    r2: f64,
    results: &mut Vec<usize>,
) {
    let node = match node {
        Some(n) => n,
        None => return,
    };

    let d2 = sq_dist(query, &node.point);
    if d2 <= r2 {
        results.push(node.orig_idx);
    }

    let axis = node.axis;
    let diff = query[axis] - node.point[axis];

    // Always explore the near side.
    let (near, far) = if diff <= 0.0 {
        (node.left.as_deref(), node.right.as_deref())
    } else {
        (node.right.as_deref(), node.left.as_deref())
    };

    radius_search_recursive(near, query, r2, results);

    // Only explore far side if the slab distance is ≤ r².
    if diff * diff <= r2 {
        radius_search_recursive(far, query, r2, results);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Approximate NN (defeatist search)
// ──────────────────────────────────────────────────────────────────────────────

fn approx_nn_search(
    node: &KDNode,
    query: &[f64],
    epsilon: f64,
    best_dist2: &mut f64,
    best_idx: &mut usize,
) {
    let d2 = sq_dist(query, &node.point);
    if d2 < *best_dist2 {
        *best_dist2 = d2;
        *best_idx = node.orig_idx;
    }

    let axis = node.axis;
    let diff = query[axis] - node.point[axis];
    let (near, far) = if diff <= 0.0 {
        (node.left.as_deref(), node.right.as_deref())
    } else {
        (node.right.as_deref(), node.left.as_deref())
    };

    if let Some(near_node) = near {
        approx_nn_search(near_node, query, epsilon, best_dist2, best_idx);
    }

    // Defeatist: only explore far side if the plane distance is within (1+ε) of best.
    let plane_dist2 = diff * diff;
    let threshold = *best_dist2 * (1.0 + epsilon).powi(2);
    if plane_dist2 < threshold {
        if let Some(far_node) = far {
            approx_nn_search(far_node, query, epsilon, best_dist2, best_idx);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Utility
// ──────────────────────────────────────────────────────────────────────────────

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_2d_points() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![5.0, 5.0],
        ]
    }

    #[test]
    fn test_build() {
        let pts = make_2d_points();
        let tree = KDTree::build(&pts).expect("build ok");
        assert_eq!(tree.dim, 2);
        assert!(tree.root.is_some());
    }

    #[test]
    fn test_nearest_1() {
        let pts = make_2d_points();
        let tree = KDTree::build(&pts).expect("build ok");
        let result = tree.nearest_k(&[0.1, 0.1], 1).expect("ok");
        assert_eq!(result.len(), 1);
        // Nearest should be index 0 (0,0)
        assert_eq!(result[0].0, 0);
    }

    #[test]
    fn test_nearest_k() {
        let pts = make_2d_points();
        let tree = KDTree::build(&pts).expect("build ok");
        let result = tree.nearest_k(&[0.5, 0.5], 4).expect("ok");
        assert_eq!(result.len(), 4);
        // The 4 corner points should be returned (not the far point at (5,5))
        let idxs: Vec<usize> = result.iter().map(|&(i, _)| i).collect();
        assert!(idxs.iter().all(|&i| i < 4));
    }

    #[test]
    fn test_radius_search() {
        let pts = make_2d_points();
        let tree = KDTree::build(&pts).expect("build ok");
        let result = tree.radius_search(&[0.5, 0.5], 1.0).expect("ok");
        // All four corner points are within distance 1 of (0.5, 0.5)
        // (distance = sqrt(0.5²+0.5²) ≈ 0.707)
        assert_eq!(result.len(), 4, "found {:?}", result);
    }

    #[test]
    fn test_radius_search_excludes_far() {
        let pts = make_2d_points();
        let tree = KDTree::build(&pts).expect("build ok");
        let result = tree.radius_search(&[0.5, 0.5], 0.5).expect("ok");
        assert!(result.is_empty(), "should find nothing within r=0.5 of centre");
    }

    #[test]
    fn test_approx_nearest() {
        let pts = make_2d_points();
        let tree = KDTree::build(&pts).expect("build ok");
        let (idx, dist) = tree.approx_nearest(&[0.0, 0.0], 0.1).expect("ok");
        assert_eq!(idx, 0);
        assert!(dist < 1e-9);
    }

    #[test]
    fn test_empty_build_error() {
        let result = KDTree::build(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dim_mismatch_error() {
        let pts = make_2d_points();
        let tree = KDTree::build(&pts).expect("build ok");
        assert!(tree.nearest_k(&[0.0, 0.0, 0.0], 1).is_err());
        assert!(tree.radius_search(&[0.0], 1.0).is_err());
        assert!(tree.approx_nearest(&[0.0, 0.0, 0.0], 0.1).is_err());
    }

    #[test]
    fn test_3d_kdtree() {
        let pts: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![10.0, 10.0, 10.0],
        ];
        let tree = KDTree::build(&pts).expect("build ok");
        let result = tree.nearest_k(&[0.1, 0.1, 0.1], 2).expect("ok");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 0); // origin is closest
    }
}
