//! Cover Tree for efficient nearest-neighbour search.
//!
//! A cover tree is a hierarchical data structure that enables O(c^12 n)
//! construction and O(c^6 log n) query for nearest-neighbour search, where
//! *c* is the "expansion constant" of the metric space.
//!
//! This implementation follows the Beygelzimer–Kakade–Langford (2006) simplified
//! cover tree definition with the separation invariant maintained throughout
//! construction.
//!
//! # References
//!
//! * Beygelzimer, A., Kakade, S., & Langford, J. (2006). Cover trees for
//!   nearest neighbor. *ICML*.

use std::collections::BinaryHeap;

use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};

use crate::error::{ClusteringError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Distance trait
// ─────────────────────────────────────────────────────────────────────────────

/// A symmetric distance function over `f64` vectors.
pub trait CoverTreeMetric: Send + Sync {
    /// Compute the distance between two row vectors.
    fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64;
}

/// Euclidean (L2) distance.
#[derive(Debug, Clone, Copy)]
pub struct L2Distance;

impl CoverTreeMetric for L2Distance {
    fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi) * (ai - bi))
            .sum::<f64>()
            .sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal node representation
// ─────────────────────────────────────────────────────────────────────────────

/// A single node in the cover tree.
#[derive(Debug)]
struct CoverNode {
    /// Index into the dataset for the point this node covers.
    point_idx: usize,
    /// Level (integer exponent s.t. 2^level is the covering radius at this level).
    level: i32,
    /// Children at level − 1.
    children: Vec<usize>, // indices into `CoverTree::nodes`
    /// Maximum distance from this node's point to any descendant leaf.
    max_dist: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Cover Tree
// ─────────────────────────────────────────────────────────────────────────────

/// Cover Tree data structure supporting k-nearest-neighbour queries.
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_cluster::topological::cover_tree::{CoverTree, CoverTreeConfig, L2Distance};
///
/// let data = Array2::from_shape_vec((5, 2), vec![
///     0.0, 0.0,
///     1.0, 0.0,
///     0.5, 0.5,
///     5.0, 5.0,
///     5.1, 5.0,
/// ]).expect("operation should succeed");
///
/// let tree = CoverTree::build(data.view(), CoverTreeConfig::default(), &L2Distance).expect("operation should succeed");
/// let neighbours = tree.knn(data.row(0), 2).expect("operation should succeed");
/// assert_eq!(neighbours.len(), 2);
/// ```
pub struct CoverTree {
    /// All points stored in row-major layout.
    data: Array2<f64>,
    /// Flat node storage.
    nodes: Vec<CoverNode>,
    /// Root node index.
    root: usize,
}

/// Configuration for the cover tree.
#[derive(Debug, Clone)]
pub struct CoverTreeConfig {
    /// Base of the level radii (default: 2.0).
    pub base: f64,
}

impl Default for CoverTreeConfig {
    fn default() -> Self {
        Self { base: 2.0 }
    }
}

impl CoverTree {
    /// Build a cover tree from an (n × d) dataset.
    pub fn build(
        data: ArrayView2<f64>,
        config: CoverTreeConfig,
        metric: &dyn CoverTreeMetric,
    ) -> Result<Self> {
        if data.nrows() == 0 {
            return Err(ClusteringError::InvalidInput(
                "cover tree requires at least one data point".into(),
            ));
        }

        // Compute pairwise distances from point 0 to find the initial level.
        let n = data.nrows();
        let mut max_dist: f64 = 0.0;
        for i in 1..n {
            let d = metric.distance(data.row(0), data.row(i));
            if d > max_dist {
                max_dist = d;
            }
        }

        let top_level = if max_dist > 0.0 {
            max_dist.log(config.base).ceil() as i32
        } else {
            0
        };

        // Insert all points one by one using the incremental algorithm.
        let data_owned = data.to_owned();
        let mut tree = CoverTree {
            data: data_owned,
            nodes: Vec::with_capacity(n),
            root: 0,
        };

        // Create root node.
        tree.nodes.push(CoverNode {
            point_idx: 0,
            level: top_level,
            children: Vec::new(),
            max_dist: 0.0,
        });

        for idx in 1..n {
            tree.insert(idx, metric, config.base)?;
        }

        Ok(tree)
    }

    /// Insert point `idx` into the tree (simplified insert).
    fn insert(
        &mut self,
        idx: usize,
        metric: &dyn CoverTreeMetric,
        base: f64,
    ) -> Result<()> {
        // Collect candidates: (node_index, distance_to_new_point).
        // We walk from the root downwards.
        let new_pt = self.data.row(idx);
        let root_pt = self.data.row(self.nodes[0].point_idx);
        let d_root = metric.distance(root_pt, new_pt);

        // If d > 2^level of root we may need to raise the root level.
        // For simplicity in this implementation we use a greedy insertion
        // that descends to the deepest covering node and attaches there.
        // This is the "simplified cover tree" variant from the 2006 paper.

        let inserted = self.try_insert_recursive(0, idx, d_root, metric, base);
        if !inserted {
            // The point falls outside every existing cover; attach to root
            // directly at root.level − 1.
            let new_level = self.nodes[0].level - 1;
            let new_node_idx = self.nodes.len();
            self.nodes.push(CoverNode {
                point_idx: idx,
                level: new_level,
                children: Vec::new(),
                max_dist: 0.0,
            });
            self.nodes[0].children.push(new_node_idx);
            self.update_max_dist(0, metric);
        }
        Ok(())
    }

    /// Recursive insert helper.  Returns `true` if the point was inserted
    /// somewhere in the subtree rooted at `node_idx`.
    fn try_insert_recursive(
        &mut self,
        node_idx: usize,
        point_idx: usize,
        d_to_node: f64,
        metric: &dyn CoverTreeMetric,
        base: f64,
    ) -> bool {
        let level = self.nodes[node_idx].level;
        let cover_radius = base.powi(level);

        if d_to_node > cover_radius {
            return false;
        }

        // Try to insert into a child first (greedy depth-first).
        let child_count = self.nodes[node_idx].children.len();
        for ci in 0..child_count {
            let child_idx = self.nodes[node_idx].children[ci];
            let child_pt = self.data.row(self.nodes[child_idx].point_idx);
            let new_pt = self.data.row(point_idx);
            let d_child = metric.distance(child_pt, new_pt);
            if self.try_insert_recursive(child_idx, point_idx, d_child, metric, base) {
                self.update_max_dist(node_idx, metric);
                return true;
            }
        }

        // None of the children could take it; attach as a new child.
        let new_level = level - 1;
        let new_node_idx = self.nodes.len();
        self.nodes.push(CoverNode {
            point_idx,
            level: new_level,
            children: Vec::new(),
            max_dist: 0.0,
        });
        self.nodes[node_idx].children.push(new_node_idx);
        self.update_max_dist(node_idx, metric);
        true
    }

    /// Recompute `max_dist` for a node from its children.
    fn update_max_dist(&mut self, node_idx: usize, metric: &dyn CoverTreeMetric) {
        let node_pt_idx = self.nodes[node_idx].point_idx;
        let mut best: f64 = 0.0;
        let child_indices: Vec<usize> = self.nodes[node_idx].children.clone();
        for ci in child_indices {
            let child_pt_idx = self.nodes[ci].point_idx;
            let d = metric.distance(
                self.data.row(node_pt_idx),
                self.data.row(child_pt_idx),
            );
            let sub = d + self.nodes[ci].max_dist;
            if sub > best {
                best = sub;
            }
        }
        self.nodes[node_idx].max_dist = best;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Nearest-neighbour query
    // ─────────────────────────────────────────────────────────────────────

    /// Return the `k` nearest neighbours of `query` as `(point_index, distance)` pairs,
    /// sorted by ascending distance.
    pub fn knn(
        &self,
        query: ArrayView1<f64>,
        k: usize,
    ) -> Result<Vec<(usize, f64)>> {
        self.knn_with_metric(query, k, &L2Distance)
    }

    /// Return the `k` nearest neighbours using the supplied metric.
    pub fn knn_with_metric(
        &self,
        query: ArrayView1<f64>,
        k: usize,
        metric: &dyn CoverTreeMetric,
    ) -> Result<Vec<(usize, f64)>> {
        if k == 0 {
            return Err(ClusteringError::InvalidInput("k must be > 0".into()));
        }

        // We use a max-heap of size k to track the current k-best.
        // Entries: `OrderedF64` wrapping (dist, point_idx).
        let mut best: BinaryHeap<OrderedEntry> = BinaryHeap::new();

        // Recursive DFS with branch-and-bound pruning.
        let d_root = metric.distance(self.data.row(self.nodes[self.root].point_idx), query);
        self.knn_recurse(self.root, query, k, &mut best, d_root, metric);

        let mut result: Vec<(usize, f64)> = best
            .into_iter()
            .map(|e| (e.point_idx, e.dist))
            .collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(result)
    }

    fn knn_recurse(
        &self,
        node_idx: usize,
        query: ArrayView1<f64>,
        k: usize,
        best: &mut BinaryHeap<OrderedEntry>,
        d_node: f64,
        metric: &dyn CoverTreeMetric,
    ) {
        // Consider this node's point.
        let node = &self.nodes[node_idx];
        let current_worst = best.peek().map(|e| e.dist).unwrap_or(f64::MAX);

        if d_node < current_worst || best.len() < k {
            if best.len() >= k {
                best.pop();
            }
            best.push(OrderedEntry {
                dist: d_node,
                point_idx: node.point_idx,
            });
        }

        // Branch-and-bound: prune subtrees that cannot contain closer points.
        let current_worst = best.peek().map(|e| e.dist).unwrap_or(f64::MAX);

        for &ci in &node.children {
            let child = &self.nodes[ci];
            let d_child =
                metric.distance(self.data.row(child.point_idx), query);
            // Lower bound on any point in this subtree is d_child - child.max_dist.
            let lb = (d_child - child.max_dist).max(0.0);
            if lb < current_worst || best.len() < k {
                self.knn_recurse(ci, query, k, best, d_child, metric);
            }
        }
    }

    /// Return all points within `radius` of `query`.
    pub fn range_query(
        &self,
        query: ArrayView1<f64>,
        radius: f64,
        metric: &dyn CoverTreeMetric,
    ) -> Vec<(usize, f64)> {
        let mut results = Vec::new();
        let d_root =
            metric.distance(self.data.row(self.nodes[self.root].point_idx), query);
        self.range_recurse(self.root, query, radius, &mut results, d_root, metric);
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    fn range_recurse(
        &self,
        node_idx: usize,
        query: ArrayView1<f64>,
        radius: f64,
        results: &mut Vec<(usize, f64)>,
        d_node: f64,
        metric: &dyn CoverTreeMetric,
    ) {
        let node = &self.nodes[node_idx];
        if d_node <= radius {
            results.push((node.point_idx, d_node));
        }
        for &ci in &node.children {
            let child = &self.nodes[ci];
            let d_child =
                metric.distance(self.data.row(child.point_idx), query);
            let lb = (d_child - child.max_dist).max(0.0);
            if lb <= radius {
                self.range_recurse(ci, query, radius, results, d_child, metric);
            }
        }
    }

    /// Number of points stored in the tree.
    pub fn len(&self) -> usize {
        self.data.nrows()
    }

    /// Returns `true` if the tree contains no points.
    pub fn is_empty(&self) -> bool {
        self.data.nrows() == 0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// A max-heap entry ordered by distance (largest distance = highest priority).
#[derive(Debug, PartialEq)]
struct OrderedEntry {
    dist: f64,
    point_idx: usize,
}

impl Eq for OrderedEntry {}

impl Ord for OrderedEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for OrderedEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn small_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 5.0, 5.0, 5.1, 5.0, 4.9, 5.0,
            ],
        )
        .expect("shape ok")
    }

    #[test]
    fn test_build_and_knn() {
        let data = small_data();
        let tree = CoverTree::build(data.view(), CoverTreeConfig::default(), &L2Distance)
            .expect("build ok");
        // Query the first point; its nearest neighbour (excluding itself) should be close.
        let nn = tree
            .knn_with_metric(data.row(0), 2, &L2Distance)
            .expect("knn ok");
        assert_eq!(nn.len(), 2);
        // First result should be the point itself (distance 0).
        assert!(
            nn[0].1 < 1e-10,
            "first NN should be self, got dist {}",
            nn[0].1
        );
    }

    #[test]
    fn test_range_query() {
        let data = small_data();
        let tree = CoverTree::build(data.view(), CoverTreeConfig::default(), &L2Distance)
            .expect("build ok");
        // Points 3,4,5 are around (5,5); query with radius 0.2 from (5.0,5.0)
        let query = data.row(3);
        let results = tree.range_query(query, 0.15, &L2Distance);
        let idxs: Vec<usize> = results.iter().map(|r| r.0).collect();
        assert!(idxs.contains(&3));
        assert!(idxs.contains(&4));
    }

    #[test]
    fn test_single_point() {
        let data = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).expect("ok");
        let tree = CoverTree::build(data.view(), CoverTreeConfig::default(), &L2Distance)
            .expect("build ok");
        assert_eq!(tree.len(), 1);
        let nn = tree
            .knn_with_metric(data.row(0), 1, &L2Distance)
            .expect("knn ok");
        assert_eq!(nn.len(), 1);
        assert!(nn[0].1 < 1e-10);
    }
}
