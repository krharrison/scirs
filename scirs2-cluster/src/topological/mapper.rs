//! Mapper algorithm (Singh, Mémoli & Carlsson 2007).
//!
//! The Mapper algorithm builds a simplicial-complex–like graph that captures
//! the topology of high-dimensional data through a three-step pipeline:
//!
//! 1. **Filter**: apply a "lens" function f : X → ℝ (see [`filtrations`]).
//! 2. **Cover**: partition the image of f into overlapping intervals.
//! 3. **Cluster + nerve**: cluster each pre-image patch; connect clusters that
//!    share at least one data point.
//!
//! # Usage
//!
//! ```rust
//! use scirs2_core::ndarray::Array2;
//! use scirs2_cluster::topological::mapper::{Mapper, MapperConfig};
//! use scirs2_cluster::topological::filtrations::{EccentricityFiltration, Filtration};
//!
//! let data = Array2::from_shape_vec((8, 2), vec![
//!     0.0, 0.0,  0.2, 0.1,  0.1, 0.2,  0.15, 0.05,
//!     5.0, 5.0,  5.2, 4.9,  4.9, 5.1,  5.1, 5.0,
//! ]).expect("operation should succeed");
//!
//! let config = MapperConfig {
//!     n_intervals: 5,
//!     overlap: 0.4,
//!     min_cluster_size: 1,
//! };
//! let filt = EccentricityFiltration::default();
//! let graph = Mapper::fit(data.view(), &filt, &config).expect("operation should succeed");
//! println!("{} nodes, {} edges", graph.nodes.len(), graph.edges.len());
//! ```
//!
//! # References
//!
//! * Singh, G., Mémoli, F., & Carlsson, G. (2007). Topological methods for
//!   the analysis of high dimensional data sets and 3D object recognition.
//!   *SPBG*, 91–100.

use std::collections::{HashMap, HashSet};

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

use crate::error::{ClusteringError, Result};
use super::filtrations::Filtration;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A single node in the Mapper graph.
///
/// Each node corresponds to a cluster within an interval's pre-image.
#[derive(Debug, Clone)]
pub struct MapperNode {
    /// Global indices of data points belonging to this node.
    pub members: Vec<usize>,
    /// The interval this node's cluster belongs to (0-indexed).
    pub interval_idx: usize,
    /// Mean filter value of member points (useful for visualisation).
    pub filter_mean: f64,
}

/// The output graph of the Mapper algorithm.
#[derive(Debug, Clone)]
pub struct MapperGraph {
    /// One node per cluster found across all intervals.
    pub nodes: Vec<MapperNode>,
    /// Undirected edges `(i, j)` with `i < j`.  An edge means the two nodes
    /// share at least one data point.
    pub edges: Vec<(usize, usize)>,
}

impl MapperGraph {
    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Returns the set of point indices that appear in *any* node
    /// (should be all n points for a well-formed mapper).
    pub fn all_points(&self) -> HashSet<usize> {
        self.nodes.iter().flat_map(|nd| nd.members.iter().copied()).collect()
    }

    /// Adjacency list representation of the graph.
    pub fn adjacency(&self) -> Vec<Vec<usize>> {
        let n = self.nodes.len();
        let mut adj = vec![Vec::new(); n];
        for &(u, v) in &self.edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        adj
    }
}

/// Configuration for the Mapper algorithm.
#[derive(Debug, Clone)]
pub struct MapperConfig {
    /// Number of intervals to divide the filter range into.
    pub n_intervals: usize,
    /// Fractional overlap between consecutive intervals (0.0 = no overlap,
    /// 1.0 = complete overlap).  Typical values: 0.3–0.5.
    pub overlap: f64,
    /// Minimum number of points required to form a cluster node.
    pub min_cluster_size: usize,
}

impl Default for MapperConfig {
    fn default() -> Self {
        Self {
            n_intervals: 10,
            overlap: 0.5,
            min_cluster_size: 1,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mapper
// ─────────────────────────────────────────────────────────────────────────────

/// The Mapper algorithm.
pub struct Mapper;

impl Mapper {
    /// Run the Mapper algorithm on `data` with the supplied filtration and
    /// configuration.
    ///
    /// # Arguments
    ///
    /// * `data`   – n × d data matrix.
    /// * `filt`   – Filter function (lens).
    /// * `config` – Mapper parameters.
    ///
    /// # Returns
    ///
    /// A [`MapperGraph`] whose nodes are clusters and edges connect nodes that
    /// share data points.
    pub fn fit(
        data: ArrayView2<f64>,
        filt: &dyn Filtration,
        config: &MapperConfig,
    ) -> Result<MapperGraph> {
        let n = data.nrows();
        if n == 0 {
            return Err(ClusteringError::InvalidInput(
                "mapper: data must be non-empty".into(),
            ));
        }
        if config.n_intervals == 0 {
            return Err(ClusteringError::InvalidInput(
                "mapper: n_intervals must be > 0".into(),
            ));
        }
        if !(0.0..1.0).contains(&config.overlap) {
            return Err(ClusteringError::InvalidInput(
                "mapper: overlap must be in [0, 1)".into(),
            ));
        }

        // ── Step 1: apply filter ──────────────────────────────────────────
        let filter_vals: Array1<f64> = filt.apply(data)?;
        debug_assert_eq!(filter_vals.len(), n);

        // ── Step 2: build overlapping cover ───────────────────────────────
        let intervals = uniform_cover(
            filter_vals.view(),
            config.n_intervals,
            config.overlap,
        )?;

        // ── Step 3: cluster each pre-image and collect nodes ──────────────
        let mut nodes: Vec<MapperNode> = Vec::new();

        for (interval_idx, (lo, hi)) in intervals.iter().enumerate() {
            // Points whose filter value lies in [lo, hi].
            let members: Vec<usize> = (0..n)
                .filter(|&i| filter_vals[i] >= *lo && filter_vals[i] <= *hi)
                .collect();

            if members.len() < config.min_cluster_size {
                continue;
            }

            // Extract sub-matrix for these members.
            let sub_data = extract_rows(data, &members);

            // Cluster within this patch using single-linkage (by default).
            // We pick a connectivity threshold as half the interval width.
            let interval_width = hi - lo;
            let epsilon = (interval_width * 0.6).max(1e-10);

            let labels = single_linkage_cluster(sub_data.view(), epsilon);
            let max_label = labels.iter().copied().max().unwrap_or(0);

            for label in 0..=max_label {
                let cluster_pts: Vec<usize> = members
                    .iter()
                    .zip(labels.iter())
                    .filter(|(_, &l)| l == label)
                    .map(|(&pt, _)| pt)
                    .collect();

                if cluster_pts.len() < config.min_cluster_size {
                    continue;
                }

                let mean_fv: f64 = cluster_pts
                    .iter()
                    .map(|&i| filter_vals[i])
                    .sum::<f64>()
                    / cluster_pts.len() as f64;

                nodes.push(MapperNode {
                    members: cluster_pts,
                    interval_idx,
                    filter_mean: mean_fv,
                });
            }
        }

        // ── Step 4: build edges (nerve construction) ──────────────────────
        let edges = build_nerve(&nodes);

        Ok(MapperGraph { nodes, edges })
    }

    /// Run Mapper with an explicit precomputed filter vector instead of a
    /// [`Filtration`] trait object.
    ///
    /// Useful when you have computed the filter values yourself.
    pub fn fit_with_filter(
        data: ArrayView2<f64>,
        filter_vals: &Array1<f64>,
        config: &MapperConfig,
    ) -> Result<MapperGraph> {
        let n = data.nrows();
        if filter_vals.len() != n {
            return Err(ClusteringError::InvalidInput(format!(
                "mapper: filter_vals length {} does not match data rows {}",
                filter_vals.len(),
                n
            )));
        }
        if config.n_intervals == 0 {
            return Err(ClusteringError::InvalidInput(
                "mapper: n_intervals must be > 0".into(),
            ));
        }

        let intervals = uniform_cover(filter_vals.view(), config.n_intervals, config.overlap)?;

        let mut nodes: Vec<MapperNode> = Vec::new();

        for (interval_idx, (lo, hi)) in intervals.iter().enumerate() {
            let members: Vec<usize> = (0..n)
                .filter(|&i| filter_vals[i] >= *lo && filter_vals[i] <= *hi)
                .collect();

            if members.len() < config.min_cluster_size {
                continue;
            }

            let sub_data = extract_rows(data, &members);
            let interval_width = hi - lo;
            let epsilon = (interval_width * 0.6).max(1e-10);
            let labels = single_linkage_cluster(sub_data.view(), epsilon);
            let max_label = labels.iter().copied().max().unwrap_or(0);

            for label in 0..=max_label {
                let cluster_pts: Vec<usize> = members
                    .iter()
                    .zip(labels.iter())
                    .filter(|(_, &l)| l == label)
                    .map(|(&pt, _)| pt)
                    .collect();

                if cluster_pts.len() < config.min_cluster_size {
                    continue;
                }

                let mean_fv: f64 = cluster_pts
                    .iter()
                    .map(|&i| filter_vals[i])
                    .sum::<f64>()
                    / cluster_pts.len() as f64;

                nodes.push(MapperNode {
                    members: cluster_pts,
                    interval_idx,
                    filter_mean: mean_fv,
                });
            }
        }

        let edges = build_nerve(&nodes);
        Ok(MapperGraph { nodes, edges })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cover construction
// ─────────────────────────────────────────────────────────────────────────────

/// Build a uniform overlapping cover of the range of `filter_vals`.
///
/// Returns a list of `(low, high)` pairs for each interval.
pub fn uniform_cover(
    filter_vals: scirs2_core::ndarray::ArrayView1<f64>,
    n_intervals: usize,
    overlap: f64,
) -> Result<Vec<(f64, f64)>> {
    let min_val = filter_vals
        .iter()
        .copied()
        .fold(f64::MAX, f64::min);
    let max_val = filter_vals
        .iter()
        .copied()
        .fold(f64::MIN, f64::max);

    if (max_val - min_val).abs() < 1e-12 {
        // All values are the same; single interval.
        return Ok(vec![(min_val - 0.5, max_val + 0.5)]);
    }

    let range = max_val - min_val;
    // Step size between interval centres.
    let step = range / n_intervals as f64;
    // Half-width of each interval including overlap.
    let half = step * (0.5 + overlap / 2.0);

    let intervals: Vec<(f64, f64)> = (0..n_intervals)
        .map(|k| {
            let centre = min_val + step * (k as f64 + 0.5);
            (centre - half, centre + half)
        })
        .collect();

    Ok(intervals)
}

// ─────────────────────────────────────────────────────────────────────────────
// Local single-linkage clustering
// ─────────────────────────────────────────────────────────────────────────────

/// Simple single-linkage clustering within a patch.
///
/// Two points are in the same cluster if their distance is ≤ `epsilon`.
/// Uses union-find for O(α n²) complexity.
fn single_linkage_cluster(data: ArrayView2<f64>, epsilon: f64) -> Vec<usize> {
    let n = data.nrows();
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<usize> = vec![0; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let d: f64 = data
                .row(i)
                .iter()
                .zip(data.row(j).iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            if d <= epsilon {
                union_find_union(&mut parent, &mut rank, i, j);
            }
        }
    }

    // Assign contiguous labels.
    let mut root_to_label: HashMap<usize, usize> = HashMap::new();
    let mut next_label = 0usize;
    let mut labels = vec![0usize; n];
    for i in 0..n {
        let root = union_find_find(&mut parent, i);
        let label = root_to_label.entry(root).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        labels[i] = *label;
    }
    labels
}

fn union_find_find(parent: &mut Vec<usize>, x: usize) -> usize {
    if parent[x] != x {
        parent[x] = union_find_find(parent, parent[x]);
    }
    parent[x]
}

fn union_find_union(parent: &mut Vec<usize>, rank: &mut Vec<usize>, x: usize, y: usize) {
    let rx = union_find_find(parent, x);
    let ry = union_find_find(parent, y);
    if rx == ry {
        return;
    }
    if rank[rx] < rank[ry] {
        parent[rx] = ry;
    } else if rank[rx] > rank[ry] {
        parent[ry] = rx;
    } else {
        parent[ry] = rx;
        rank[rx] += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Nerve construction
// ─────────────────────────────────────────────────────────────────────────────

/// Build the nerve of a cover: add an edge between any two nodes that share
/// at least one data point.
fn build_nerve(nodes: &[MapperNode]) -> Vec<(usize, usize)> {
    let n = nodes.len();
    let mut edges = Vec::new();

    // Build a point → node index map for fast intersection testing.
    let mut pt_to_nodes: HashMap<usize, Vec<usize>> = HashMap::new();
    for (node_idx, node) in nodes.iter().enumerate() {
        for &pt in &node.members {
            pt_to_nodes.entry(pt).or_default().push(node_idx);
        }
    }

    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    for node_lists in pt_to_nodes.values() {
        let k = node_lists.len();
        for i in 0..k {
            for j in (i + 1)..k {
                let u = node_lists[i].min(node_lists[j]);
                let v = node_lists[i].max(node_lists[j]);
                if u != v && seen.insert((u, v)) {
                    edges.push((u, v));
                }
            }
        }
    }
    edges.sort_unstable();
    let _ = n; // suppress unused warning
    edges
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Extract specified rows from an array view.
fn extract_rows(data: ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
    let d = data.ncols();
    let m = indices.len();
    let mut out = Array2::zeros((m, d));
    for (new_i, &old_i) in indices.iter().enumerate() {
        out.row_mut(new_i).assign(&data.row(old_i));
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use crate::topological::filtrations::EccentricityFiltration;

    fn two_blobs() -> Array2<f64> {
        Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.2, 0.1, 0.1, 0.2, 0.15, 0.05, 5.0, 5.0, 5.2, 4.9, 4.9, 5.1, 5.1,
                5.0,
            ],
        )
        .expect("shape ok")
    }

    #[test]
    fn test_mapper_two_blobs() {
        let data = two_blobs();
        let filt = EccentricityFiltration::default();
        let config = MapperConfig {
            n_intervals: 5,
            overlap: 0.4,
            min_cluster_size: 1,
        };
        let graph = Mapper::fit(data.view(), &filt, &config).expect("mapper ok");
        // Should produce at least 2 nodes (one per blob).
        assert!(graph.n_nodes() >= 2, "expected ≥ 2 nodes, got {}", graph.n_nodes());
    }

    #[test]
    fn test_mapper_with_precomputed_filter() {
        let data = two_blobs();
        let n = data.nrows();
        // Manual filter: x-coordinate.
        let filter_vals: Array1<f64> = (0..n).map(|i| data[[i, 0]]).collect();
        let config = MapperConfig::default();
        let graph = Mapper::fit_with_filter(data.view(), &filter_vals, &config).expect("ok");
        assert!(graph.n_nodes() >= 1);
    }

    #[test]
    fn test_uniform_cover_range() {
        use scirs2_core::ndarray::Array1;
        let vals: Array1<f64> = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let intervals = uniform_cover(vals.view(), 4, 0.3).expect("ok");
        assert_eq!(intervals.len(), 4);
        // First interval should start before 0.0 (with overlap).
        assert!(intervals[0].0 <= 0.0);
        // Last interval should end after 4.0 (with overlap).
        assert!(intervals[3].1 >= 4.0);
    }

    #[test]
    fn test_nerve_shared_points() {
        let nodes = vec![
            MapperNode { members: vec![0, 1, 2], interval_idx: 0, filter_mean: 0.1 },
            MapperNode { members: vec![2, 3, 4], interval_idx: 1, filter_mean: 0.5 },
            MapperNode { members: vec![5, 6, 7], interval_idx: 2, filter_mean: 0.9 },
        ];
        let edges = build_nerve(&nodes);
        // Nodes 0 and 1 share point 2 → edge (0,1).
        assert!(edges.contains(&(0, 1)));
        // Nodes 0 and 2 share no points → no edge.
        assert!(!edges.contains(&(0, 2)));
        // Nodes 1 and 2 share no points → no edge.
        assert!(!edges.contains(&(1, 2)));
    }
}
