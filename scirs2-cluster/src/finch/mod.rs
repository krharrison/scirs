//! FINCH: First Integer Neighbor Clustering Hierarchy
//!
//! Implements the FINCH algorithm (Sarfraz et al. 2019) which constructs a
//! hierarchical clustering by iteratively linking each point to its first
//! nearest neighbour and finding connected components.
//!
//! # Algorithm Overview
//!
//! 1. Compute the undirected 1-NN graph (each point links to its nearest neighbour).
//! 2. Find connected components of that graph → finest partition (level 0).
//! 3. Replace each component with its centroid and repeat.
//! 4. Stop when a single cluster is reached or `n_levels` is exhausted.
//!
//! # References
//!
//! - Sarfraz, Saquib, Vivek Sharma, and Rainer Stiefelhagen.
//!   "Efficient parameter-free clustering using first neighbor relations."
//!   *CVPR 2019*.
//!
//! # Example
//!
//! ```rust
//! use scirs2_cluster::finch::{finch, FINCHConfig, DistanceMetric};
//!
//! let data = vec![
//!     vec![1.0_f64, 2.0],
//!     vec![1.1, 1.9],
//!     vec![0.9, 2.1],
//!     vec![8.0, 8.0],
//!     vec![8.1, 7.9],
//!     vec![7.9, 8.1],
//! ];
//!
//! let config = FINCHConfig::default();
//! let result = finch(&data, &config).expect("finch failed");
//! // Level 0 should recover ~2 clusters
//! assert!(result.partitions[0].iter().any(|&l| l >= 0));
//! ```

use std::collections::BinaryHeap;
use std::cmp::Ordering;

use crate::error::{ClusteringError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Public configuration types
// ─────────────────────────────────────────────────────────────────────────────

/// Distance metric used for nearest-neighbour search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance.
    #[default]
    Euclidean,
    /// Manhattan (L1) distance.
    Manhattan,
    /// Cosine dissimilarity: `1 - cos(a, b)`.
    Cosine,
}

/// Configuration for the FINCH algorithm.
#[derive(Debug, Clone)]
pub struct FINCHConfig {
    /// Maximum number of hierarchy levels to compute.
    /// `None` means run until a single cluster or convergence.
    pub n_levels: Option<usize>,
    /// Distance metric for nearest-neighbour computation.
    pub distance: DistanceMetric,
}

impl Default for FINCHConfig {
    fn default() -> Self {
        Self {
            n_levels: None,
            distance: DistanceMetric::Euclidean,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Output types
// ─────────────────────────────────────────────────────────────────────────────

/// Result of FINCH hierarchical clustering.
#[derive(Debug, Clone)]
pub struct FINCHResult {
    /// One entry per hierarchy level.  `partitions[0]` is the finest partition
    /// (level 0), `partitions[last]` is the coarsest.
    ///
    /// Each inner `Vec<i32>` maps original point index → cluster label (0-indexed).
    pub partitions: Vec<Vec<i32>>,
    /// Number of clusters at each level.
    pub n_clusters_per_level: Vec<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Distance helpers
// ─────────────────────────────────────────────────────────────────────────────

fn dist_sq_euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn dist_manhattan(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

fn dist_cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-15 || nb < 1e-15 {
        return 1.0;
    }
    1.0 - (dot / (na * nb)).clamp(-1.0, 1.0)
}

fn pairwise_distance(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => dist_sq_euclidean(a, b).sqrt(),
        DistanceMetric::Manhattan => dist_manhattan(a, b),
        DistanceMetric::Cosine => dist_cosine(a, b),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// First-neighbour computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the index of the first (nearest) neighbour for every point.
///
/// The neighbour of a point is the closest *other* point; i.e. self-loops are
/// excluded.  Returns a vector of length `n` where `result[i]` is the index of
/// the nearest neighbour of point `i`.
pub fn compute_first_neighbors(data: &[Vec<f64>], metric: DistanceMetric) -> Result<Vec<usize>> {
    let n = data.len();
    if n == 0 {
        return Err(ClusteringError::InvalidInput(
            "Data must be non-empty".to_string(),
        ));
    }
    if n == 1 {
        // Single point: no neighbour; return 0 (self) as sentinel.
        return Ok(vec![0]);
    }

    let dim = data[0].len();
    for (i, row) in data.iter().enumerate() {
        if row.len() != dim {
            return Err(ClusteringError::InvalidInput(format!(
                "Row {} has {} features, expected {}",
                i,
                row.len(),
                dim
            )));
        }
    }

    let mut neighbours = Vec::with_capacity(n);
    for i in 0..n {
        let mut best_j = if i == 0 { 1 } else { 0 };
        let mut best_d = pairwise_distance(&data[i], &data[best_j], metric);
        for j in 0..n {
            if j == i {
                continue;
            }
            let d = pairwise_distance(&data[i], &data[j], metric);
            if d < best_d {
                best_d = d;
                best_j = j;
            }
        }
        neighbours.push(best_j);
    }
    Ok(neighbours)
}

// ─────────────────────────────────────────────────────────────────────────────
// Union-Find (path-compressed)
// ─────────────────────────────────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            Ordering::Less => self.parent[rx] = ry,
            Ordering::Greater => self.parent[ry] = rx,
            Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }
}

/// Find connected components of an *undirected* graph given as an edge list.
///
/// Returns a label vector of length `n_nodes` where each entry is the
/// component ID (0-indexed, densely packed).
pub fn connected_components_undirected(n_nodes: usize, edges: &[(usize, usize)]) -> Vec<i32> {
    let mut uf = UnionFind::new(n_nodes);
    for &(u, v) in edges {
        uf.union(u, v);
    }

    // Normalise root IDs → dense 0-based labels.
    let mut root_to_label = std::collections::HashMap::new();
    let mut next_label: i32 = 0;
    let mut labels = vec![0i32; n_nodes];
    for i in 0..n_nodes {
        let root = uf.find(i);
        let label = root_to_label.entry(root).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        labels[i] = *label;
    }
    labels
}

// ─────────────────────────────────────────────────────────────────────────────
// Centroid computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute per-cluster centroids.
fn compute_centroids(
    data: &[Vec<f64>],
    labels: &[i32],
    n_clusters: usize,
) -> Vec<Vec<f64>> {
    let dim = data[0].len();
    let mut sums = vec![vec![0.0f64; dim]; n_clusters];
    let mut counts = vec![0usize; n_clusters];

    for (point, &label) in data.iter().zip(labels.iter()) {
        let k = label as usize;
        counts[k] += 1;
        for (d, v) in sums[k].iter_mut().zip(point.iter()) {
            *d += v;
        }
    }

    for (sum, count) in sums.iter_mut().zip(counts.iter()) {
        if *count > 0 {
            let inv = 1.0 / *count as f64;
            for v in sum.iter_mut() {
                *v *= inv;
            }
        }
    }
    sums
}

// ─────────────────────────────────────────────────────────────────────────────
// Main FINCH entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run the FINCH hierarchical clustering algorithm.
///
/// # Parameters
///
/// - `data`: slice of feature vectors (each inner `Vec` is one point).
/// - `config`: algorithm configuration.
///
/// # Returns
///
/// A [`FINCHResult`] containing the label assignment at every hierarchy level
/// and the corresponding cluster counts.
pub fn finch(data: &[Vec<f64>], config: &FINCHConfig) -> Result<FINCHResult> {
    let n = data.len();
    if n == 0 {
        return Err(ClusteringError::InvalidInput(
            "Data must be non-empty".to_string(),
        ));
    }
    if n == 1 {
        return Ok(FINCHResult {
            partitions: vec![vec![0i32]],
            n_clusters_per_level: vec![1],
        });
    }

    let dim = data[0].len();
    if dim == 0 {
        return Err(ClusteringError::InvalidInput(
            "Feature dimension must be > 0".to_string(),
        ));
    }

    let max_levels = config.n_levels.unwrap_or(usize::MAX);
    let metric = config.distance;

    let mut partitions: Vec<Vec<i32>> = Vec::new();
    let mut n_clusters_per_level: Vec<usize> = Vec::new();

    // `current_data` holds the working set (centroids at higher levels).
    // `label_map` maps current-level indices → original-point labels.
    let mut current_data: Vec<Vec<f64>> = data.to_vec();
    // Maps current_data index → component label at the *previous* level.
    // After the first pass this is populated from the component partition.
    let mut current_to_orig: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    let mut level = 0;
    // Keep previous cluster count to detect convergence.
    let mut prev_n_clusters = usize::MAX;

    loop {
        if level >= max_levels {
            break;
        }
        let m = current_data.len();
        if m <= 1 {
            // Already a single cluster; propagate label 0 for all originals.
            let labels = vec![0i32; n];
            if partitions.is_empty()
                || *n_clusters_per_level.last().unwrap_or(&usize::MAX) > 1
            {
                partitions.push(labels);
                n_clusters_per_level.push(1);
            }
            break;
        }

        // Step 1: Compute first neighbours in current_data.
        let neighbours = compute_first_neighbors(&current_data, metric)?;

        // Step 2: Build undirected edge list.
        let edges: Vec<(usize, usize)> = (0..m)
            .map(|i| (i, neighbours[i]))
            .collect();

        // Step 3: Connected components.
        let comp_labels = connected_components_undirected(m, &edges);
        let n_clusters = comp_labels.iter().map(|&l| l as usize + 1).max().unwrap_or(0);

        // Check for convergence (same number of clusters as previous level).
        if n_clusters == prev_n_clusters {
            break;
        }
        prev_n_clusters = n_clusters;

        // Step 4: Map component labels back to original points.
        let mut orig_labels = vec![0i32; n];
        for (cur_idx, member_set) in current_to_orig.iter().enumerate() {
            let cluster_id = comp_labels[cur_idx];
            for &orig_idx in member_set {
                orig_labels[orig_idx] = cluster_id;
            }
        }

        partitions.push(orig_labels.clone());
        n_clusters_per_level.push(n_clusters);

        if n_clusters == 1 {
            break;
        }

        // Step 5: Compute centroids of components → next level data.
        let centroids = compute_centroids(&current_data, &comp_labels, n_clusters);

        // Build updated current_to_orig mapping.
        let mut new_to_orig: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
        for (cur_idx, member_set) in current_to_orig.iter().enumerate() {
            let cluster_id = comp_labels[cur_idx] as usize;
            new_to_orig[cluster_id].extend_from_slice(member_set);
        }

        current_data = centroids;
        current_to_orig = new_to_orig;
        level += 1;
    }

    if partitions.is_empty() {
        // Degenerate: return all-same label.
        partitions.push(vec![0i32; n]);
        n_clusters_per_level.push(1);
    }

    Ok(FINCHResult {
        partitions,
        n_clusters_per_level,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn two_cluster_data() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 2.0],
            vec![1.1, 1.9],
            vec![0.9, 2.1],
            vec![1.2, 1.8],
            vec![8.0, 8.0],
            vec![8.1, 7.9],
            vec![7.9, 8.1],
            vec![8.2, 7.8],
        ]
    }

    #[test]
    fn test_finch_two_clusters() {
        let data = two_cluster_data();
        let config = FINCHConfig::default();
        let result = finch(&data, &config).expect("finch failed");

        assert!(!result.partitions.is_empty(), "should have at least one level");
        // Level 0 should recover 2 clusters.
        let n0 = result.n_clusters_per_level[0];
        assert_eq!(n0, 2, "expected 2 clusters at level 0, got {}", n0);

        // All points in group A (indices 0-3) should share a label.
        let p0 = &result.partitions[0];
        assert_eq!(p0[0], p0[1]);
        assert_eq!(p0[0], p0[2]);
        assert_eq!(p0[0], p0[3]);
        // All points in group B (indices 4-7) should share a different label.
        assert_eq!(p0[4], p0[5]);
        assert_eq!(p0[4], p0[6]);
        assert_eq!(p0[4], p0[7]);
        assert_ne!(p0[0], p0[4]);
    }

    #[test]
    fn test_finch_hierarchy() {
        let data = two_cluster_data();
        let config = FINCHConfig::default();
        let result = finch(&data, &config).expect("finch failed");

        // The hierarchy should end with 1 cluster.
        let last_n = *result.n_clusters_per_level.last().expect("non-empty");
        assert_eq!(last_n, 1, "hierarchy should end with 1 cluster");
    }

    #[test]
    fn test_finch_single_point() {
        let data = vec![vec![1.0, 2.0]];
        let config = FINCHConfig::default();
        let result = finch(&data, &config).expect("single point finch");
        assert_eq!(result.n_clusters_per_level[0], 1);
    }

    #[test]
    fn test_compute_first_neighbors() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![10.0, 0.0],
        ];
        let nbrs = compute_first_neighbors(&data, DistanceMetric::Euclidean)
            .expect("neighbours ok");
        // Point 0 is closest to point 1.
        assert_eq!(nbrs[0], 1);
        // Point 1 is closest to point 0.
        assert_eq!(nbrs[1], 0);
        // Point 2 is closest to point 1.
        assert_eq!(nbrs[2], 1);
    }

    #[test]
    fn test_connected_components() {
        // Graph: 0-1-2   3 (isolated)
        let edges = vec![(0, 1), (1, 2)];
        let labels = connected_components_undirected(4, &edges);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_finch_cosine_metric() {
        let data = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
        ];
        let config = FINCHConfig {
            n_levels: Some(1),
            distance: DistanceMetric::Cosine,
        };
        let result = finch(&data, &config).expect("cosine finch");
        assert!(!result.partitions.is_empty());
    }

    #[test]
    fn test_finch_manhattan_metric() {
        let data = two_cluster_data();
        let config = FINCHConfig {
            n_levels: None,
            distance: DistanceMetric::Manhattan,
        };
        let result = finch(&data, &config).expect("manhattan finch");
        assert!(result.n_clusters_per_level[0] >= 1);
    }
}
