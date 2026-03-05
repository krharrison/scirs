//! Enhanced Ball Tree for efficient spatial queries
//!
//! This module provides a high-performance ball tree implementation with:
//! - Hierarchical ball partitioning using furthest-point splitting
//! - Single nearest neighbor and k-NN queries
//! - Range queries (find all points within radius)
//! - Dual-tree algorithms for batch nearest neighbor queries
//! - Configurable leaf size and distance metrics
//!
//! # References
//!
//! * Omohundro, S.M. (1989) "Five Balltree Construction Algorithms"
//! * Liu, T. et al. (2006) "An Investigation of Practical Approximate Nearest Neighbor Algorithms"
//! * Gray & Moore (2003) "Rapid Evaluation of Multiple Density Models"

use crate::distance::{Distance, EuclideanDistance};
use crate::error::{SpatialError, SpatialResult};
use crate::safe_conversions::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::Float;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// BallNode
// ---------------------------------------------------------------------------

/// A node in the ball tree hierarchy
#[derive(Clone, Debug)]
struct BallNode<T: Float> {
    /// Center of the bounding ball
    center: Vec<T>,
    /// Radius of the bounding ball
    radius: T,
    /// Range of point indices owned by this node [start, end)
    start: usize,
    /// Exclusive end index
    end: usize,
    /// Left child index (None for leaf)
    left: Option<usize>,
    /// Right child index (None for leaf)
    right: Option<usize>,
}

impl<T: Float> BallNode<T> {
    fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    fn count(&self) -> usize {
        self.end - self.start
    }
}

// ---------------------------------------------------------------------------
// Neighbor heap item
// ---------------------------------------------------------------------------

/// A candidate neighbor ordered by distance (max-heap so we can pop the farthest)
#[derive(Clone, Debug)]
struct NeighborCandidate<T: Float> {
    index: usize,
    distance: T,
}

impl<T: Float> PartialEq for NeighborCandidate<T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<T: Float> Eq for NeighborCandidate<T> {}

impl<T: Float> PartialOrd for NeighborCandidate<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Float> Ord for NeighborCandidate<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // max-heap: larger distance is "greater"
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for building and querying the enhanced ball tree
#[derive(Clone, Debug)]
pub struct BallTreeConfig {
    /// Maximum number of points in a leaf node (default: 40)
    pub leaf_size: usize,
}

impl Default for BallTreeConfig {
    fn default() -> Self {
        Self { leaf_size: 40 }
    }
}

impl BallTreeConfig {
    /// Create a new configuration with the given leaf size
    pub fn with_leaf_size(mut self, leaf_size: usize) -> Self {
        self.leaf_size = leaf_size;
        self
    }
}

// ---------------------------------------------------------------------------
// EnhancedBallTree
// ---------------------------------------------------------------------------

/// Enhanced Ball Tree for efficient spatial queries in arbitrary dimensions
///
/// The ball tree partitions data points into a hierarchy of nested
/// hyperspheres ("balls"). Compared to KD-trees, ball trees often perform
/// better in higher dimensions because the bounding balls can capture the
/// actual geometry of the point distribution more faithfully.
///
/// # Type Parameters
///
/// * `T` - Floating-point type (f32 or f64)
/// * `D` - Distance metric implementing the [`Distance`] trait
#[derive(Clone, Debug)]
pub struct EnhancedBallTree<T: Float + Send + Sync, D: Distance<T>> {
    /// Copy of data points (rows reordered during construction)
    data: Array2<T>,
    /// Mapping from internal index to original index
    indices: Vec<usize>,
    /// Flat array of tree nodes
    nodes: Vec<BallNode<T>>,
    /// Number of samples
    n_samples: usize,
    /// Dimensionality
    n_features: usize,
    /// Distance metric
    dist: D,
    /// Configuration
    config: BallTreeConfig,
    /// Phantom
    _phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync + 'static> EnhancedBallTree<T, EuclideanDistance<T>> {
    /// Convenience constructor using Euclidean distance
    pub fn with_euclidean(data: &ArrayView2<T>, config: BallTreeConfig) -> SpatialResult<Self> {
        Self::new(data, EuclideanDistance::new(), config)
    }
}

impl<T: Float + Send + Sync + 'static, D: Distance<T> + Send + Sync + 'static>
    EnhancedBallTree<T, D>
{
    /// Create a new enhanced ball tree from a set of data points.
    ///
    /// # Arguments
    ///
    /// * `data`   - An (n_samples x n_features) array of points
    /// * `dist`   - Distance metric
    /// * `config` - Build configuration
    pub fn new(data: &ArrayView2<T>, dist: D, config: BallTreeConfig) -> SpatialResult<Self> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if n_samples == 0 {
            return Err(SpatialError::ValueError(
                "Cannot build ball tree from empty data".to_string(),
            ));
        }
        if n_features == 0 {
            return Err(SpatialError::ValueError(
                "Points must have at least one dimension".to_string(),
            ));
        }

        let data_owned = if data.is_standard_layout() {
            data.to_owned()
        } else {
            data.as_standard_layout().to_owned()
        };

        let indices: Vec<usize> = (0..n_samples).collect();
        let leaf_size = config.leaf_size.max(1);
        let config = BallTreeConfig {
            leaf_size,
            ..config
        };

        let mut tree = EnhancedBallTree {
            data: data_owned,
            indices,
            nodes: Vec::with_capacity(2 * n_samples / config.leaf_size + 1),
            n_samples,
            n_features,
            dist,
            config,
            _phantom: PhantomData,
        };

        tree.build(0, n_samples)?;
        Ok(tree)
    }

    // ------------------------------------------------------------------
    // Construction helpers
    // ------------------------------------------------------------------

    /// Recursively build the tree for the index range [start, end).
    /// Returns the node index of the root of this subtree.
    fn build(&mut self, start: usize, end: usize) -> SpatialResult<usize> {
        let count = end - start;
        let center = self.compute_centroid(start, end)?;
        let radius = self.compute_radius(&center, start, end);

        let node_idx = self.nodes.len();

        if count <= self.config.leaf_size {
            // Leaf node
            self.nodes.push(BallNode {
                center,
                radius,
                start,
                end,
                left: None,
                right: None,
            });
            return Ok(node_idx);
        }

        // Placeholder node (we fill children after recursing)
        self.nodes.push(BallNode {
            center: center.clone(),
            radius,
            start,
            end,
            left: None,
            right: None,
        });

        // Split using furthest-point heuristic
        let mid = self.split_points(start, end, &center)?;

        // Guard: if the split didn't separate anything, make it a leaf
        if mid == start || mid == end {
            self.nodes[node_idx].left = None;
            self.nodes[node_idx].right = None;
            return Ok(node_idx);
        }

        let left_idx = self.build(start, mid)?;
        let right_idx = self.build(mid, end)?;

        self.nodes[node_idx].left = Some(left_idx);
        self.nodes[node_idx].right = Some(right_idx);

        Ok(node_idx)
    }

    /// Compute centroid of points in [start, end)
    fn compute_centroid(&self, start: usize, end: usize) -> SpatialResult<Vec<T>> {
        let count = end - start;
        let mut center = vec![T::zero(); self.n_features];
        for i in start..end {
            let row = self.data.row(self.indices[i]);
            for (j, val) in row.iter().enumerate() {
                center[j] = center[j] + *val;
            }
        }
        let n = safe_from_usize::<T>(count, "ball_tree centroid")?;
        for c in &mut center {
            *c = *c / n;
        }
        Ok(center)
    }

    /// Compute the radius of a ball centered at `center` that encloses
    /// all points in [start, end).
    fn compute_radius(&self, center: &[T], start: usize, end: usize) -> T {
        let mut max_dist = T::zero();
        for i in start..end {
            let row = self.data.row(self.indices[i]);
            let d = self.dist.distance(center, row.as_slice().unwrap_or(&[]));
            if d > max_dist {
                max_dist = d;
            }
        }
        max_dist
    }

    /// Split the index range [start, end) into two halves using the
    /// furthest-point heuristic and return the midpoint.
    fn split_points(&mut self, start: usize, end: usize, _center: &[T]) -> SpatialResult<usize> {
        // 1. Find the spread dimension (largest variance)
        let spread_dim = self.find_spread_dimension(start, end)?;

        // 2. Compute median along that dimension and partition
        let mid = (start + end) / 2;
        self.nth_element(start, end, mid, spread_dim);
        Ok(mid)
    }

    /// Find the dimension with the largest spread (max - min) among
    /// points in [start, end).
    fn find_spread_dimension(&self, start: usize, end: usize) -> SpatialResult<usize> {
        let mut best_dim = 0;
        let mut best_spread = T::neg_infinity();
        for d in 0..self.n_features {
            let mut lo = T::infinity();
            let mut hi = T::neg_infinity();
            for i in start..end {
                let v = self.data[[self.indices[i], d]];
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
            let spread = hi - lo;
            if spread > best_spread {
                best_spread = spread;
                best_dim = d;
            }
        }
        Ok(best_dim)
    }

    /// Partial sort so that `indices[mid]` has the correct value and
    /// everything < mid is less-or-equal along `dim`.
    fn nth_element(&mut self, start: usize, end: usize, mid: usize, dim: usize) {
        if start >= end {
            return;
        }
        // Simple in-place selection using Lomuto partition
        self.quick_select(start, end - 1, mid, dim);
    }

    fn quick_select(&mut self, lo: usize, hi: usize, k: usize, dim: usize) {
        if lo >= hi {
            return;
        }
        let pivot = self.partition(lo, hi, dim);
        match k.cmp(&pivot) {
            Ordering::Equal => {}
            Ordering::Less => {
                if pivot > 0 {
                    self.quick_select(lo, pivot - 1, k, dim);
                }
            }
            Ordering::Greater => {
                self.quick_select(pivot + 1, hi, k, dim);
            }
        }
    }

    fn partition(&mut self, lo: usize, hi: usize, dim: usize) -> usize {
        let pivot_val = self.data[[self.indices[hi], dim]];
        let mut store = lo;
        for i in lo..hi {
            if self.data[[self.indices[i], dim]] <= pivot_val {
                self.indices.swap(store, i);
                store += 1;
            }
        }
        self.indices.swap(store, hi);
        store
    }

    /// Retrieve the point slice for an internal index
    fn point(&self, internal_idx: usize) -> Vec<T> {
        let orig = self.indices[internal_idx];
        self.data.row(orig).to_vec()
    }

    // ------------------------------------------------------------------
    // Public query API
    // ------------------------------------------------------------------

    /// Return the number of points stored in the tree.
    pub fn len(&self) -> usize {
        self.n_samples
    }

    /// Return `true` if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.n_samples == 0
    }

    /// Return the dimensionality of the points.
    pub fn dim(&self) -> usize {
        self.n_features
    }

    /// Find the single nearest neighbor of `query`.
    ///
    /// Returns `(original_index, distance)`.
    pub fn nearest(&self, query: &[T]) -> SpatialResult<(usize, T)> {
        if query.len() != self.n_features {
            return Err(SpatialError::DimensionError(format!(
                "Query has {} dims but tree has {}",
                query.len(),
                self.n_features
            )));
        }
        let mut best_dist = T::infinity();
        let mut best_idx = 0usize;
        self.nearest_recurse(0, query, &mut best_idx, &mut best_dist);
        Ok((self.indices[best_idx], best_dist))
    }

    fn nearest_recurse(
        &self,
        node_idx: usize,
        query: &[T],
        best_idx: &mut usize,
        best_dist: &mut T,
    ) {
        let node = &self.nodes[node_idx];

        // Prune: if the closest possible point in this ball is farther
        // than the current best, skip.
        let dist_to_center = self.dist.distance(query, &node.center);
        let lower_bound = if dist_to_center > node.radius {
            dist_to_center - node.radius
        } else {
            T::zero()
        };
        if lower_bound >= *best_dist {
            return;
        }

        if node.is_leaf() {
            for i in node.start..node.end {
                let p = self.point(i);
                let d = self.dist.distance(query, &p);
                if d < *best_dist {
                    *best_dist = d;
                    *best_idx = i;
                }
            }
            return;
        }

        // Visit the child whose center is closer first
        let (first, second) = self.order_children(node, query);
        if let Some(f) = first {
            self.nearest_recurse(f, query, best_idx, best_dist);
        }
        if let Some(s) = second {
            self.nearest_recurse(s, query, best_idx, best_dist);
        }
    }

    /// Find the `k` nearest neighbors of `query`.
    ///
    /// Returns `(indices, distances)` where both vectors are sorted by
    /// ascending distance.
    pub fn query_knn(&self, query: &[T], k: usize) -> SpatialResult<(Vec<usize>, Vec<T>)> {
        if query.len() != self.n_features {
            return Err(SpatialError::DimensionError(format!(
                "Query has {} dims but tree has {}",
                query.len(),
                self.n_features
            )));
        }
        if k == 0 {
            return Ok((vec![], vec![]));
        }
        let k = k.min(self.n_samples);

        let mut heap: BinaryHeap<NeighborCandidate<T>> = BinaryHeap::with_capacity(k + 1);
        self.knn_recurse(0, query, k, &mut heap);

        let mut results: Vec<(usize, T)> = heap
            .into_iter()
            .map(|nc| (self.indices[nc.index], nc.distance))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let (idx, dist): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        Ok((idx, dist))
    }

    fn knn_recurse(
        &self,
        node_idx: usize,
        query: &[T],
        k: usize,
        heap: &mut BinaryHeap<NeighborCandidate<T>>,
    ) {
        let node = &self.nodes[node_idx];

        let dist_to_center = self.dist.distance(query, &node.center);
        let lower_bound = if dist_to_center > node.radius {
            dist_to_center - node.radius
        } else {
            T::zero()
        };

        // Prune if we already have k neighbors and the closest possible
        // point in this ball is farther than the current k-th best.
        if heap.len() >= k {
            if let Some(top) = heap.peek() {
                if lower_bound >= top.distance {
                    return;
                }
            }
        }

        if node.is_leaf() {
            for i in node.start..node.end {
                let p = self.point(i);
                let d = self.dist.distance(query, &p);
                if heap.len() < k {
                    heap.push(NeighborCandidate {
                        index: i,
                        distance: d,
                    });
                } else if let Some(top) = heap.peek() {
                    if d < top.distance {
                        heap.pop();
                        heap.push(NeighborCandidate {
                            index: i,
                            distance: d,
                        });
                    }
                }
            }
            return;
        }

        let (first, second) = self.order_children(node, query);
        if let Some(f) = first {
            self.knn_recurse(f, query, k, heap);
        }
        if let Some(s) = second {
            self.knn_recurse(s, query, k, heap);
        }
    }

    /// Find all points within `radius` of `query`.
    ///
    /// Returns `(indices, distances)` sorted by ascending distance.
    pub fn query_radius(&self, query: &[T], radius: T) -> SpatialResult<(Vec<usize>, Vec<T>)> {
        if query.len() != self.n_features {
            return Err(SpatialError::DimensionError(format!(
                "Query has {} dims but tree has {}",
                query.len(),
                self.n_features
            )));
        }
        if radius < T::zero() {
            return Err(SpatialError::ValueError(
                "Radius must be non-negative".to_string(),
            ));
        }

        let mut results: Vec<(usize, T)> = Vec::new();
        self.range_recurse(0, query, radius, &mut results);

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let (idx, dist): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        Ok((idx, dist))
    }

    fn range_recurse(
        &self,
        node_idx: usize,
        query: &[T],
        radius: T,
        results: &mut Vec<(usize, T)>,
    ) {
        let node = &self.nodes[node_idx];

        let dist_to_center = self.dist.distance(query, &node.center);
        // If the ball is entirely outside the search radius, prune.
        if dist_to_center - node.radius > radius {
            return;
        }

        // If the ball is entirely inside the search radius, add all points.
        if dist_to_center + node.radius <= radius {
            for i in node.start..node.end {
                let p = self.point(i);
                let d = self.dist.distance(query, &p);
                results.push((self.indices[i], d));
            }
            return;
        }

        if node.is_leaf() {
            for i in node.start..node.end {
                let p = self.point(i);
                let d = self.dist.distance(query, &p);
                if d <= radius {
                    results.push((self.indices[i], d));
                }
            }
            return;
        }

        if let Some(l) = node.left {
            self.range_recurse(l, query, radius, results);
        }
        if let Some(r) = node.right {
            self.range_recurse(r, query, radius, results);
        }
    }

    /// Count points within `radius` of `query` without returning them.
    pub fn count_radius(&self, query: &[T], radius: T) -> SpatialResult<usize> {
        if query.len() != self.n_features {
            return Err(SpatialError::DimensionError(format!(
                "Query has {} dims but tree has {}",
                query.len(),
                self.n_features
            )));
        }
        if radius < T::zero() {
            return Err(SpatialError::ValueError(
                "Radius must be non-negative".to_string(),
            ));
        }
        let mut count = 0usize;
        self.count_radius_recurse(0, query, radius, &mut count);
        Ok(count)
    }

    fn count_radius_recurse(&self, node_idx: usize, query: &[T], radius: T, count: &mut usize) {
        let node = &self.nodes[node_idx];
        let dist_to_center = self.dist.distance(query, &node.center);

        if dist_to_center - node.radius > radius {
            return;
        }
        if dist_to_center + node.radius <= radius {
            *count += node.count();
            return;
        }
        if node.is_leaf() {
            for i in node.start..node.end {
                let p = self.point(i);
                let d = self.dist.distance(query, &p);
                if d <= radius {
                    *count += 1;
                }
            }
            return;
        }
        if let Some(l) = node.left {
            self.count_radius_recurse(l, query, radius, count);
        }
        if let Some(r) = node.right {
            self.count_radius_recurse(r, query, radius, count);
        }
    }

    // ------------------------------------------------------------------
    // Dual-tree algorithms
    // ------------------------------------------------------------------

    /// Dual-tree all-nearest-neighbors: for every point in `other`, find its
    /// nearest neighbor in `self`.
    ///
    /// Returns `(indices_in_self, distances)` with one entry per point in
    /// `other`, sorted by the order of points in `other`.
    pub fn dual_tree_nearest(
        &self,
        other: &EnhancedBallTree<T, D>,
    ) -> SpatialResult<(Vec<usize>, Vec<T>)> {
        if self.n_features != other.n_features {
            return Err(SpatialError::DimensionError(format!(
                "Dimension mismatch: self has {} dims, other has {}",
                self.n_features, other.n_features
            )));
        }

        let m = other.n_samples;
        let mut best_idx = vec![0usize; m];
        let mut best_dist = vec![T::infinity(); m];

        if self.nodes.is_empty() || other.nodes.is_empty() {
            return Ok((best_idx, best_dist));
        }

        self.dual_tree_recurse(0, other, 0, &mut best_idx, &mut best_dist);

        // Map internal indices back to original indices
        for bi in &mut best_idx {
            *bi = self.indices[*bi];
        }

        Ok((best_idx, best_dist))
    }

    fn dual_tree_recurse(
        &self,
        self_node: usize,
        other: &EnhancedBallTree<T, D>,
        other_node: usize,
        best_idx: &mut [usize],
        best_dist: &mut [T],
    ) {
        let sn = &self.nodes[self_node];
        let on = &other.nodes[other_node];

        // Lower bound on the distance between any pair of points
        let center_dist = self.dist.distance(&sn.center, &on.center);
        let lower_bound = if center_dist > sn.radius + on.radius {
            center_dist - sn.radius - on.radius
        } else {
            T::zero()
        };

        // Check if we can prune: if the lower bound is >= the maximum
        // of all current best distances for points in `other_node`, skip
        let mut max_best = T::zero();
        for i in on.start..on.end {
            let oi = other.indices[i];
            if best_dist[oi] > max_best {
                max_best = best_dist[oi];
            }
        }
        if lower_bound >= max_best {
            return;
        }

        // Base case: both are leaves
        if sn.is_leaf() && on.is_leaf() {
            for oi in on.start..on.end {
                let op = other.point(oi);
                let orig_oi = other.indices[oi];
                for si in sn.start..sn.end {
                    let sp = self.point(si);
                    let d = self.dist.distance(&op, &sp);
                    if d < best_dist[orig_oi] {
                        best_dist[orig_oi] = d;
                        best_idx[orig_oi] = si;
                    }
                }
            }
            return;
        }

        // Recurse: split the larger node
        if sn.is_leaf() {
            // Split other node
            if let (Some(ol), Some(or)) = (on.left, on.right) {
                self.dual_tree_recurse(self_node, other, ol, best_idx, best_dist);
                self.dual_tree_recurse(self_node, other, or, best_idx, best_dist);
            }
        } else if on.is_leaf() {
            // Split self node
            if let (Some(sl), Some(sr)) = (sn.left, sn.right) {
                // Visit closer child first
                let dl = self.dist.distance(&self.nodes[sl].center, &on.center);
                let dr = self.dist.distance(&self.nodes[sr].center, &on.center);
                if dl <= dr {
                    self.dual_tree_recurse(sl, other, other_node, best_idx, best_dist);
                    self.dual_tree_recurse(sr, other, other_node, best_idx, best_dist);
                } else {
                    self.dual_tree_recurse(sr, other, other_node, best_idx, best_dist);
                    self.dual_tree_recurse(sl, other, other_node, best_idx, best_dist);
                }
            }
        } else {
            // Both are internal -- split both
            let children_s: Vec<usize> = [sn.left, sn.right].iter().filter_map(|x| *x).collect();
            let children_o: Vec<usize> = [on.left, on.right].iter().filter_map(|x| *x).collect();

            // Build pairs sorted by lower bound
            let mut pairs: Vec<(usize, usize, T)> = Vec::new();
            for &sc in &children_s {
                for &oc in &children_o {
                    let cd = self
                        .dist
                        .distance(&self.nodes[sc].center, &other.nodes[oc].center);
                    let lb = if cd > self.nodes[sc].radius + other.nodes[oc].radius {
                        cd - self.nodes[sc].radius - other.nodes[oc].radius
                    } else {
                        T::zero()
                    };
                    pairs.push((sc, oc, lb));
                }
            }
            pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

            for (sc, oc, _) in pairs {
                self.dual_tree_recurse(sc, other, oc, best_idx, best_dist);
            }
        }
    }

    /// Batch k-NN query using dual-tree traversal.
    ///
    /// For each point in `queries`, find the `k` nearest neighbors in this
    /// tree. Returns `(indices, distances)` each of shape `queries.nrows() x k`.
    pub fn batch_knn(
        &self,
        queries: &ArrayView2<T>,
        k: usize,
    ) -> SpatialResult<(Array2<usize>, Array2<T>)> {
        if queries.ncols() != self.n_features {
            return Err(SpatialError::DimensionError(format!(
                "Query has {} dims but tree has {}",
                queries.ncols(),
                self.n_features
            )));
        }
        let m = queries.nrows();
        let k_actual = k.min(self.n_samples);

        let mut idx_out = Array2::zeros((m, k_actual));
        let mut dist_out = Array2::from_elem((m, k_actual), T::infinity());

        for qi in 0..m {
            let q = queries.row(qi).to_vec();
            let (ids, ds) = self.query_knn(&q, k_actual)?;
            for j in 0..ids.len() {
                idx_out[[qi, j]] = ids[j];
                dist_out[[qi, j]] = ds[j];
            }
        }

        Ok((idx_out, dist_out))
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// Order children so the one whose center is closer to query is visited first.
    fn order_children(&self, node: &BallNode<T>, query: &[T]) -> (Option<usize>, Option<usize>) {
        match (node.left, node.right) {
            (Some(l), Some(r)) => {
                let dl = self.dist.distance(query, &self.nodes[l].center);
                let dr = self.dist.distance(query, &self.nodes[r].center);
                if dl <= dr {
                    (Some(l), Some(r))
                } else {
                    (Some(r), Some(l))
                }
            }
            (Some(l), None) => (Some(l), None),
            (None, Some(r)) => (Some(r), None),
            (None, None) => (None, None),
        }
    }

    /// Return the depth of the tree.
    pub fn depth(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        self.node_depth(0)
    }

    fn node_depth(&self, idx: usize) -> usize {
        let node = &self.nodes[idx];
        if node.is_leaf() {
            return 1;
        }
        let ld = node.left.map_or(0, |l| self.node_depth(l));
        let rd = node.right.map_or(0, |r| self.node_depth(r));
        1 + ld.max(rd)
    }

    /// Return the total number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of leaf nodes in the tree.
    pub fn leaf_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    fn make_grid_4() -> Array2<f64> {
        array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    }

    #[test]
    fn test_construction_basic() {
        let pts = make_grid_4();
        let tree = EnhancedBallTree::with_euclidean(&pts.view(), BallTreeConfig::default());
        assert!(tree.is_ok());
        let tree = tree.expect("construction should succeed");
        assert_eq!(tree.len(), 4);
        assert_eq!(tree.dim(), 2);
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_construction_empty() {
        let pts = Array2::<f64>::zeros((0, 2));
        let result = EnhancedBallTree::with_euclidean(&pts.view(), BallTreeConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_nearest_basic() {
        let pts = make_grid_4();
        let tree = EnhancedBallTree::with_euclidean(
            &pts.view(),
            BallTreeConfig::default().with_leaf_size(2),
        )
        .expect("build");
        let (idx, dist) = tree.nearest(&[0.1, 0.1]).expect("nearest");
        assert_eq!(idx, 0); // closest to (0,0)
        assert_relative_eq!(dist, (0.1f64 * 0.1 + 0.1 * 0.1).sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_nearest_exact_match() {
        let pts = make_grid_4();
        let tree = EnhancedBallTree::with_euclidean(&pts.view(), BallTreeConfig::default())
            .expect("build");
        let (idx, dist) = tree.nearest(&[1.0, 1.0]).expect("nearest");
        assert_eq!(idx, 3);
        assert_relative_eq!(dist, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_knn_basic() {
        let pts = make_grid_4();
        let tree = EnhancedBallTree::with_euclidean(
            &pts.view(),
            BallTreeConfig::default().with_leaf_size(1),
        )
        .expect("build");
        let (idx, dist) = tree.query_knn(&[0.5, 0.5], 4).expect("knn");
        assert_eq!(idx.len(), 4);
        // All 4 grid points are equidistant from (0.5,0.5): dist = sqrt(0.5)
        let expected = (0.5f64).sqrt();
        for d in &dist {
            assert_relative_eq!(*d, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_knn_k_exceeds_n() {
        let pts = make_grid_4();
        let tree = EnhancedBallTree::with_euclidean(&pts.view(), BallTreeConfig::default())
            .expect("build");
        let (idx, _dist) = tree.query_knn(&[0.0, 0.0], 10).expect("knn");
        assert_eq!(idx.len(), 4); // capped at n
    }

    #[test]
    fn test_knn_zero() {
        let pts = make_grid_4();
        let tree = EnhancedBallTree::with_euclidean(&pts.view(), BallTreeConfig::default())
            .expect("build");
        let (idx, dist) = tree.query_knn(&[0.0, 0.0], 0).expect("knn");
        assert!(idx.is_empty());
        assert!(dist.is_empty());
    }

    #[test]
    fn test_range_query() {
        let pts = make_grid_4();
        let tree = EnhancedBallTree::with_euclidean(
            &pts.view(),
            BallTreeConfig::default().with_leaf_size(1),
        )
        .expect("build");

        // Radius 0.5: only the origin should be found
        let (idx, _dist) = tree.query_radius(&[0.0, 0.0], 0.5).expect("range");
        assert_eq!(idx.len(), 1);
        assert_eq!(idx[0], 0);

        // Radius 1.5: all 4 points (max dist from origin is sqrt(2) ~ 1.414)
        let (idx, _dist) = tree.query_radius(&[0.0, 0.0], 1.5).expect("range");
        assert_eq!(idx.len(), 4);
    }

    #[test]
    fn test_count_radius() {
        let pts = make_grid_4();
        let tree = EnhancedBallTree::with_euclidean(&pts.view(), BallTreeConfig::default())
            .expect("build");
        let c = tree.count_radius(&[0.5, 0.5], 0.8).expect("count");
        assert_eq!(c, 4); // sqrt(0.5) ~ 0.707 < 0.8
    }

    #[test]
    fn test_range_negative_radius() {
        let pts = make_grid_4();
        let tree = EnhancedBallTree::with_euclidean(&pts.view(), BallTreeConfig::default())
            .expect("build");
        let result = tree.query_radius(&[0.0, 0.0], -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let pts = make_grid_4();
        let tree = EnhancedBallTree::with_euclidean(&pts.view(), BallTreeConfig::default())
            .expect("build");
        let result = tree.nearest(&[0.0, 0.0, 0.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_larger_dataset() {
        let n = 200;
        let dim = 5;
        let mut data = Array2::zeros((n, dim));
        for i in 0..n {
            for d in 0..dim {
                data[[i, d]] = ((i * 7 + d * 13) % 100) as f64 / 100.0;
            }
        }

        let tree = EnhancedBallTree::with_euclidean(
            &data.view(),
            BallTreeConfig::default().with_leaf_size(10),
        )
        .expect("build");

        // Brute-force check for a query
        let query = vec![0.5; dim];
        let (bt_idx, bt_dist) = tree.nearest(&query).expect("nearest");

        let mut bf_best_dist = f64::INFINITY;
        let mut bf_best_idx = 0;
        for i in 0..n {
            let mut d2 = 0.0;
            for dd in 0..dim {
                let diff = data[[i, dd]] - query[dd];
                d2 += diff * diff;
            }
            let d = d2.sqrt();
            if d < bf_best_dist {
                bf_best_dist = d;
                bf_best_idx = i;
            }
        }

        // Both bt_idx and bf_best_idx are valid answers when tie-breaking at equal distance.
        // Assert the ball tree found the correct distance, and that its returned index
        // actually achieves that minimum distance (handles ties correctly).
        assert_relative_eq!(bt_dist, bf_best_dist, epsilon = 1e-10);
        // Verify bt_idx has the same distance as the brute-force best
        let mut bt_verify_dist = 0.0_f64;
        for dd in 0..dim {
            let diff = data[[bt_idx, dd]] - query[dd];
            bt_verify_dist += diff * diff;
        }
        assert_relative_eq!(bt_verify_dist.sqrt(), bf_best_dist, epsilon = 1e-10);
    }

    #[test]
    fn test_knn_ordering() {
        let pts = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0],];
        let tree = EnhancedBallTree::with_euclidean(
            &pts.view(),
            BallTreeConfig::default().with_leaf_size(1),
        )
        .expect("build");

        let (idx, dist) = tree.query_knn(&[0.5, 0.0], 3).expect("knn");
        assert_eq!(idx.len(), 3);
        // Closest should be indices 0 (d=0.5), 1 (d=0.5), 2 (d=1.5)
        assert!(dist[0] <= dist[1]);
        assert!(dist[1] <= dist[2]);
    }

    #[test]
    fn test_dual_tree_nearest() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];
        let queries = array![[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],];

        let tree_data = EnhancedBallTree::with_euclidean(
            &data.view(),
            BallTreeConfig::default().with_leaf_size(1),
        )
        .expect("build data");
        let tree_query = EnhancedBallTree::with_euclidean(
            &queries.view(),
            BallTreeConfig::default().with_leaf_size(1),
        )
        .expect("build queries");

        let (idx, _dist) = tree_data.dual_tree_nearest(&tree_query).expect("dual tree");
        assert_eq!(idx.len(), 4);
        assert_eq!(idx[0], 0); // (0.1,0.1) -> (0,0)
        assert_eq!(idx[1], 1); // (0.9,0.1) -> (1,0)
        assert_eq!(idx[2], 2); // (0.1,0.9) -> (0,1)
        assert_eq!(idx[3], 3); // (0.9,0.9) -> (1,1)
    }

    #[test]
    fn test_batch_knn() {
        let data = make_grid_4();
        let tree = EnhancedBallTree::with_euclidean(
            &data.view(),
            BallTreeConfig::default().with_leaf_size(1),
        )
        .expect("build");

        let queries = array![[0.0, 0.0], [1.0, 1.0]];
        let (idx, dist) = tree.batch_knn(&queries.view(), 2).expect("batch knn");

        assert_eq!(idx.shape(), &[2, 2]);
        assert_eq!(dist.shape(), &[2, 2]);

        // First query (0,0): nearest are idx 0 (d=0) and then 1 or 2 (d=1)
        assert_eq!(idx[[0, 0]], 0);
        assert_relative_eq!(dist[[0, 0]], 0.0, epsilon = 1e-12);

        // Second query (1,1): nearest is idx 3 (d=0)
        assert_eq!(idx[[1, 0]], 3);
        assert_relative_eq!(dist[[1, 0]], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_tree_statistics() {
        let pts = make_grid_4();
        let tree = EnhancedBallTree::with_euclidean(
            &pts.view(),
            BallTreeConfig::default().with_leaf_size(1),
        )
        .expect("build");

        assert!(tree.depth() >= 1);
        assert!(tree.node_count() >= 1);
        assert!(tree.leaf_count() >= 1);
    }

    #[test]
    fn test_single_point() {
        let pts = array![[42.0, 17.0]];
        let tree = EnhancedBallTree::with_euclidean(&pts.view(), BallTreeConfig::default())
            .expect("build");
        let (idx, dist) = tree.nearest(&[0.0, 0.0]).expect("nearest");
        assert_eq!(idx, 0);
        assert_relative_eq!(dist, (42.0f64 * 42.0 + 17.0 * 17.0).sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_high_dimensional() {
        let dim = 20;
        let n = 50;
        let mut data = Array2::zeros((n, dim));
        for i in 0..n {
            for d in 0..dim {
                data[[i, d]] = ((i * 3 + d * 7) % 50) as f64;
            }
        }
        let tree = EnhancedBallTree::with_euclidean(
            &data.view(),
            BallTreeConfig::default().with_leaf_size(5),
        )
        .expect("build");

        let query = vec![25.0; dim];
        let (idx, _dist) = tree.query_knn(&query, 5).expect("knn");
        assert_eq!(idx.len(), 5);
    }

    #[test]
    fn test_all_identical_points() {
        let pts = array![[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]];
        let tree = EnhancedBallTree::with_euclidean(
            &pts.view(),
            BallTreeConfig::default().with_leaf_size(1),
        )
        .expect("build");

        let (idx, dist) = tree.query_knn(&[1.0, 2.0], 2).expect("knn");
        assert_eq!(idx.len(), 2);
        for d in &dist {
            assert_relative_eq!(*d, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_range_empty_result() {
        let pts = make_grid_4();
        let tree = EnhancedBallTree::with_euclidean(&pts.view(), BallTreeConfig::default())
            .expect("build");
        let (idx, dist) = tree.query_radius(&[10.0, 10.0], 0.01).expect("range");
        assert!(idx.is_empty());
        assert!(dist.is_empty());
    }

    #[test]
    fn test_with_manhattan_distance() {
        use crate::distance::ManhattanDistance;
        let pts = make_grid_4();
        let tree = EnhancedBallTree::new(
            &pts.view(),
            ManhattanDistance::new(),
            BallTreeConfig::default(),
        )
        .expect("build");

        let (idx, dist) = tree.nearest(&[0.0, 0.0]).expect("nearest");
        assert_eq!(idx, 0);
        assert_relative_eq!(dist, 0.0, epsilon = 1e-12);

        // Manhattan distance from (0.5, 0.5) to any corner = 1.0
        let (_idx, dist) = tree.query_knn(&[0.5, 0.5], 1).expect("knn");
        assert_relative_eq!(dist[0], 1.0, epsilon = 1e-10);
    }
}
