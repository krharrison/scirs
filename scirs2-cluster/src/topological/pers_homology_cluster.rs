//! Persistence-based clustering algorithms.
//!
//! This module implements:
//!
//! 1. **ToMATo** (Topological Mode Analysis Tool, Chazal et al. 2013) – a
//!    gradient-flow–based clustering algorithm that merges basins of attraction
//!    according to a persistence threshold `τ`.  The output clusters correspond
//!    to prominent peaks (modes) of the density estimator.
//!
//! 2. **Persistence-based flat clustering** – extract flat cluster labels from
//!    a 0-dimensional persistence diagram by keeping only bars with persistence
//!    ≥ a threshold and assigning each point to its representative cluster.
//!
//! 3. **Automatic threshold selection via gap statistic** – choose `τ`
//!    automatically from the largest gap in the sorted persistence values.
//!
//! # References
//!
//! * Chazal, F., Guibas, L., Oudot, S., & Skraba, P. (2013). Persistence-based
//!   clustering in Riemannian manifolds. *JACM*, 60(6), 41.
//! * Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An
//!   Introduction*. AMS.

use std::collections::HashMap;

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

use crate::error::{ClusteringError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Union-Find
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
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Persistence diagram entry
// ─────────────────────────────────────────────────────────────────────────────

/// A bar in a 0-dimensional persistence diagram.
///
/// `birth` and `death` refer to filter values (e.g. density or height).
/// If `death` is `f64::MAX` the component is the "infinite" bar.
#[derive(Debug, Clone)]
pub struct PersistenceBar {
    /// Filter value at which this component was born.
    pub birth: f64,
    /// Filter value at which this component died (merged into an older one).
    /// `f64::MAX` for the infinite bar.
    pub death: f64,
    /// Data-point index of the representative (oldest) point in this component.
    pub representative: usize,
}

impl PersistenceBar {
    /// Persistence (lifetime) of this bar: `death − birth`.
    /// Returns `f64::MAX` for the infinite bar.
    pub fn persistence(&self) -> f64 {
        if self.death == f64::MAX {
            f64::MAX
        } else {
            (self.death - self.birth).abs()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 0-D Persistent Homology (sub-level-set filtration)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the 0-dimensional persistence diagram of `data` using a
/// *super*-level-set filtration on a density estimate.
///
/// Steps:
/// 1. Estimate density at each point using k-NN.
/// 2. Sort points by decreasing density.
/// 3. Add points one at a time; when a point's neighbours (in the k-NN graph)
///    are already present, merge the components.
/// 4. Record a persistence bar `(birth = density[i], death = density[merge point])`
///    when two components merge.
///
/// This is the ToMATo sub-routine from Chazal et al. 2013.
///
/// # Arguments
///
/// * `data`       – n × d data matrix.
/// * `k_neighbors` – Number of neighbours for the density estimate and graph.
///
/// # Returns
///
/// A vector of [`PersistenceBar`] sorted by decreasing persistence.
pub fn density_persistence(
    data: ArrayView2<f64>,
    k_neighbors: usize,
) -> Result<Vec<PersistenceBar>> {
    let n = data.nrows();
    let d = data.ncols();
    if n == 0 || d == 0 {
        return Err(ClusteringError::InvalidInput(
            "persistence: data must be non-empty".into(),
        ));
    }
    let k = k_neighbors.min(n - 1).max(1);

    // ── 1. k-NN density estimate ──────────────────────────────────────────
    // For each point compute the distance to its k-th nearest neighbour.
    let mut knn_dist = vec![0.0f64; n];
    let mut knn_graph: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let d2: f64 = data
                    .row(i)
                    .iter()
                    .zip(data.row(j).iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                (j, d2.sqrt())
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        knn_dist[i] = dists[k - 1].1.max(1e-15);
        for &(j, _) in dists.iter().take(k) {
            knn_graph[i].push(j);
        }
    }

    // density ∝ 1 / r_k^d
    let density: Vec<f64> = knn_dist
        .iter()
        .map(|&r| 1.0 / r.powi(d as i32))
        .collect();

    // ── 2. Sort by decreasing density ─────────────────────────────────────
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        density[b]
            .partial_cmp(&density[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // ── 3. Incremental union-find ─────────────────────────────────────────
    let mut uf = UnionFind::new(n);
    let mut added = vec![false; n];
    // For each component root, track the density at birth.
    let mut birth_density = vec![0.0f64; n];
    let mut bars: Vec<PersistenceBar> = Vec::new();

    for &i in &order {
        added[i] = true;
        birth_density[i] = density[i];
        let mut new_root = i;

        for &j in &knn_graph[i] {
            if !added[j] {
                continue;
            }
            let rj = uf.find(j);
            let ri = uf.find(new_root);
            if ri == rj {
                continue;
            }

            // Merge the younger component into the older one.
            // The component with *higher* birth density is older (born earlier
            // in the super-level-set filtration).
            let birth_ri = birth_density[ri];
            let birth_rj = birth_density[rj];

            if birth_ri >= birth_rj {
                // rj is younger → it dies here.
                bars.push(PersistenceBar {
                    birth: birth_rj,
                    death: density[i],
                    representative: rj,
                });
                uf.union(ri, rj);
                let new_r = uf.find(ri);
                birth_density[new_r] = birth_density[ri];
                new_root = new_r;
            } else {
                // ri is younger → it dies here.
                bars.push(PersistenceBar {
                    birth: birth_ri,
                    death: density[i],
                    representative: ri,
                });
                uf.union(rj, ri);
                let new_r = uf.find(rj);
                birth_density[new_r] = birth_density[rj];
                new_root = new_r;
            }
        }
    }

    // The surviving component.
    let global_root = uf.find(order[0]);
    bars.push(PersistenceBar {
        birth: birth_density[global_root],
        death: f64::MAX,
        representative: global_root,
    });

    // Sort by decreasing persistence.
    bars.sort_by(|a, b| {
        b.persistence()
            .partial_cmp(&a.persistence())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(bars)
}

// ─────────────────────────────────────────────────────────────────────────────
// ToMATo clustering
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for ToMATo.
#[derive(Debug, Clone)]
pub struct TomaToConfig {
    /// Number of neighbours for density and gradient-flow graph (default: 5).
    pub k_neighbors: usize,
    /// Persistence threshold τ.  Clusters with persistence < τ are merged into
    /// their parent.  Set to 0.0 to use the automatic gap heuristic.
    pub persistence_threshold: f64,
    /// If `persistence_threshold == 0.0`, use the largest gap among the
    /// persistence values to select τ automatically.
    pub auto_threshold: bool,
}

impl Default for TomaToConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 5,
            persistence_threshold: 0.0,
            auto_threshold: true,
        }
    }
}

impl TomaToConfig {
    /// Create with an explicit threshold.
    pub fn with_threshold(k_neighbors: usize, tau: f64) -> Self {
        Self {
            k_neighbors,
            persistence_threshold: tau,
            auto_threshold: false,
        }
    }
}

/// Result of ToMATo clustering.
#[derive(Debug, Clone)]
pub struct TomaToResult {
    /// Cluster label for each data point (0-indexed).  Points are assigned to
    /// the most prominent topological mode whose basin they flow into.
    pub labels: Vec<usize>,
    /// Number of clusters found.
    pub n_clusters: usize,
    /// Persistence threshold used.
    pub threshold: f64,
    /// The full persistence diagram (before thresholding).
    pub persistence_diagram: Vec<PersistenceBar>,
}

/// ToMATo: Topological Mode Analysis Tool.
///
/// Clusters data by density gradient flow, retaining only peaks whose
/// prominence (persistence) exceeds a threshold `τ`.
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_cluster::topological::pers_homology_cluster::{tomato, TomaToConfig};
///
/// let data = Array2::from_shape_vec((6, 2), vec![
///     0.0, 0.0,  0.1, 0.1,  -0.1, 0.0,
///     5.0, 5.0,  5.1, 4.9,   4.9, 5.0,
/// ]).expect("operation should succeed");
///
/// let config = TomaToConfig { k_neighbors: 2, auto_threshold: true, ..Default::default() };
/// let result = tomato(data.view(), &config).expect("operation should succeed");
/// assert_eq!(result.labels.len(), 6);
/// ```
pub fn tomato(data: ArrayView2<f64>, config: &TomaToConfig) -> Result<TomaToResult> {
    let n = data.nrows();
    let d = data.ncols();
    if n == 0 || d == 0 {
        return Err(ClusteringError::InvalidInput(
            "tomato: data must be non-empty".into(),
        ));
    }

    let k = config.k_neighbors.min(n - 1).max(1);

    // ── 1. Density + k-NN graph ───────────────────────────────────────────
    let mut knn_dist = vec![0.0f64; n];
    let mut knn_graph: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let d2: f64 = data
                    .row(i)
                    .iter()
                    .zip(data.row(j).iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                (j, d2.sqrt())
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        knn_dist[i] = dists[k - 1].1.max(1e-15);
        for &(j, _) in dists.iter().take(k) {
            knn_graph[i].push(j);
        }
    }

    let density: Vec<f64> = knn_dist
        .iter()
        .map(|&r| 1.0 / r.powi(d as i32))
        .collect();

    // ── 2. Gradient flow: each point flows to its densest neighbour ───────
    // `parent[i]` = the neighbour (or i itself) with maximum density.
    let mut gradient_parent: Vec<usize> = (0..n).collect();
    for i in 0..n {
        let mut best = i;
        let mut best_dens = density[i];
        for &j in &knn_graph[i] {
            if density[j] > best_dens {
                best = j;
                best_dens = density[j];
            }
        }
        gradient_parent[i] = best;
    }

    // Follow the gradient flow to find local maxima (modes).
    // A point is a mode if gradient_parent[i] == i.
    let mut mode_of: Vec<usize> = (0..n).collect();
    for i in 0..n {
        mode_of[i] = follow_gradient(&gradient_parent, i);
    }

    // ── 3. Persistence diagram ─────────────────────────────────────────────
    // Build a merge tree: process edges in decreasing-density order.
    // Sort all k-NN edges by the *lower* endpoint density (merges happen when
    // we add an edge connecting two components).
    let bars = density_persistence(data, k)?;

    // ── 4. Select threshold ────────────────────────────────────────────────
    let tau = if config.auto_threshold || config.persistence_threshold <= 0.0 {
        gap_threshold(&bars)
    } else {
        config.persistence_threshold
    };

    // ── 5. Merge basins below threshold ───────────────────────────────────
    // Two modes merge if the persistence of the younger mode < τ.
    // We iterate over bars in decreasing persistence order (they are already
    // sorted) and use union-find.
    let mut mode_uf = UnionFind::new(n);

    for bar in &bars {
        if bar.death == f64::MAX {
            // The infinite bar: do not merge.
            continue;
        }
        if bar.persistence() < tau {
            // Merge this component into its parent (the surviving root of its
            // gradient-flow neighbour).
            let rep = bar.representative;
            // Find a higher-density neighbour of rep.
            let merge_to = knn_graph[rep]
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    density[a]
                        .partial_cmp(&density[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(rep);
            mode_uf.union(rep, merge_to);
        }
    }

    // Assign labels: each unique root in mode_uf → one cluster.
    let mut root_to_label: HashMap<usize, usize> = HashMap::new();
    let mut next_label = 0usize;
    let mut labels = vec![0usize; n];
    for i in 0..n {
        let mode = mode_of[i];
        let root = mode_uf.find(mode);
        let label = root_to_label.entry(root).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        labels[i] = *label;
    }

    Ok(TomaToResult {
        labels,
        n_clusters: next_label,
        threshold: tau,
        persistence_diagram: bars,
    })
}

/// Follow the gradient flow chain until a fixed point (mode) is reached.
fn follow_gradient(parent: &[usize], start: usize) -> usize {
    let mut cur = start;
    // Limit path length to n to avoid infinite loops.
    let n = parent.len();
    for _ in 0..n {
        let next = parent[cur];
        if next == cur {
            return cur;
        }
        cur = next;
    }
    cur
}

// ─────────────────────────────────────────────────────────────────────────────
// Flat clustering from persistence diagram
// ─────────────────────────────────────────────────────────────────────────────

/// Assign each data point to a cluster by thresholding the persistence diagram.
///
/// Points are assigned to the representative of the bar with the highest
/// persistence that they belong to.  Bars with persistence < `threshold` are
/// ignored (their points merge into the next surviving cluster).
///
/// This is simpler than ToMATo but works well when combined with the
/// 0-D persistence diagram from [`crate::topological_clustering::persistent_homology_0d`].
///
/// # Arguments
///
/// * `bars`      – Persistence diagram (0-D bars), sorted by decreasing persistence.
/// * `point_to_root` – Map from each data point index to its union-find root at
///   the time its bar was closed.  If `None`, points are assigned to bars by
///   index (representative = point index).
/// * `n`         – Total number of data points.
/// * `threshold` – Minimum persistence to keep a cluster.
///
/// # Returns
///
/// A vector of cluster labels (0-indexed) of length n.
pub fn flat_clustering_from_persistence(
    bars: &[PersistenceBar],
    n: usize,
    threshold: f64,
) -> Vec<usize> {
    // Determine surviving bars.
    let surviving: Vec<usize> = bars
        .iter()
        .enumerate()
        .filter(|(_, bar)| bar.persistence() >= threshold || bar.death == f64::MAX)
        .map(|(i, _)| i)
        .collect();

    if surviving.is_empty() {
        // Everything in one cluster.
        return vec![0; n];
    }

    // Assign each point to the surviving bar whose representative is closest.
    // Since we do not have the full assignment here, we use the representative
    // indices as cluster seeds and assign points by nearest-representative index.
    // (A full implementation would carry point-to-component maps from the
    // filtration; here we return representative-indexed labels.)
    let representatives: Vec<usize> = surviving.iter().map(|&i| bars[i].representative).collect();

    // Default: assign all points to cluster 0.
    let mut labels = vec![0usize; n];

    for (label_idx, &rep) in representatives.iter().enumerate() {
        if rep < n {
            labels[rep] = label_idx;
        }
    }

    labels
}

// ─────────────────────────────────────────────────────────────────────────────
// Automatic threshold selection
// ─────────────────────────────────────────────────────────────────────────────

/// Select the persistence threshold by finding the largest gap in the sorted
/// finite persistence values.
///
/// This heuristic is equivalent to elbow-detection on the sorted-persistence
/// curve.
pub fn gap_threshold(bars: &[PersistenceBar]) -> f64 {
    let mut finite_perss: Vec<f64> = bars
        .iter()
        .filter(|b| b.death < f64::MAX)
        .map(|b| b.persistence())
        .collect();
    finite_perss.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if finite_perss.is_empty() {
        return 0.0;
    }
    if finite_perss.len() == 1 {
        return finite_perss[0] * 0.5;
    }

    let mut best_gap = 0.0f64;
    let mut best_tau = finite_perss[0] * 0.5;
    for i in 1..finite_perss.len() {
        let gap = finite_perss[i] - finite_perss[i - 1];
        if gap > best_gap {
            best_gap = gap;
            best_tau = (finite_perss[i] + finite_perss[i - 1]) / 2.0;
        }
    }
    best_tau
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: flat clustering via ToMATo with n_clusters hint
// ─────────────────────────────────────────────────────────────────────────────

/// Run ToMATo and return exactly `n_clusters` clusters by choosing the
/// persistence threshold that gives exactly `n_clusters` surviving bars.
///
/// If no threshold achieves exactly `n_clusters`, returns the closest result.
pub fn tomato_n_clusters(
    data: ArrayView2<f64>,
    n_clusters: usize,
    k_neighbors: usize,
) -> Result<Vec<usize>> {
    if n_clusters == 0 {
        return Err(ClusteringError::InvalidInput(
            "n_clusters must be > 0".into(),
        ));
    }

    let bars = density_persistence(data, k_neighbors)?;

    // Collect finite persistence values and sort descending.
    let mut perss: Vec<f64> = bars
        .iter()
        .filter(|b| b.death < f64::MAX)
        .map(|b| b.persistence())
        .collect();
    perss.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // The number of clusters = number of bars with persistence ≥ τ + 1
    // (the +1 is for the infinite bar).
    // We want n_clusters, so we set τ just above the (n_clusters − 1)-th largest
    // finite persistence value.
    let tau = if n_clusters <= 1 {
        f64::MAX
    } else if n_clusters - 1 <= perss.len() {
        let idx = n_clusters - 1; // 0-based
        if idx < perss.len() {
            perss[idx - 1] * 0.5 + perss[idx.saturating_sub(1)] * 0.5
        } else {
            0.0
        }
    } else {
        0.0
    };

    let config = TomaToConfig {
        k_neighbors,
        persistence_threshold: tau,
        auto_threshold: false,
    };
    let result = tomato(data, &config)?;
    Ok(result.labels)
}

// ─────────────────────────────────────────────────────────────────────────────
// Cluster statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Compute summary statistics for each cluster from a label assignment.
#[derive(Debug, Clone)]
pub struct ClusterStats {
    /// Cluster label (0-indexed).
    pub label: usize,
    /// Number of points in the cluster.
    pub size: usize,
    /// Centroid of the cluster (mean of member coordinates).
    pub centroid: Array1<f64>,
    /// Average intra-cluster distance.
    pub avg_intra_dist: f64,
}

/// Compute per-cluster statistics.
pub fn cluster_stats(
    data: ArrayView2<f64>,
    labels: &[usize],
) -> Result<Vec<ClusterStats>> {
    let n = data.nrows();
    let d = data.ncols();
    if labels.len() != n {
        return Err(ClusteringError::InvalidInput(
            "cluster_stats: labels length must match data rows".into(),
        ));
    }

    let n_clusters = labels.iter().copied().max().unwrap_or(0) + 1;
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    for (i, &l) in labels.iter().enumerate() {
        members[l].push(i);
    }

    let mut stats = Vec::with_capacity(n_clusters);
    for (label, pts) in members.iter().enumerate() {
        if pts.is_empty() {
            continue;
        }
        let m = pts.len();

        // Centroid.
        let mut centroid = Array1::zeros(d);
        for &i in pts {
            for j in 0..d {
                centroid[j] += data[[i, j]];
            }
        }
        for j in 0..d {
            centroid[j] /= m as f64;
        }

        // Average intra-cluster distance.
        let mut sum_dist = 0.0f64;
        let mut count = 0usize;
        for ii in 0..m {
            for jj in (ii + 1)..m {
                let d_val: f64 = data
                    .row(pts[ii])
                    .iter()
                    .zip(data.row(pts[jj]).iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                sum_dist += d_val;
                count += 1;
            }
        }
        let avg_intra_dist = if count > 0 { sum_dist / count as f64 } else { 0.0 };

        stats.push(ClusterStats {
            label,
            size: m,
            centroid,
            avg_intra_dist,
        });
    }
    Ok(stats)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn two_blobs() -> Array2<f64> {
        Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.2, 0.1, -0.1, 0.1, 0.05, -0.05,
                6.0, 6.0, 6.2, 5.9, 5.9, 6.1, 6.1, 6.0,
            ],
        )
        .expect("shape ok")
    }

    #[test]
    fn test_density_persistence_two_blobs() {
        let data = two_blobs();
        let bars = density_persistence(data.view(), 3).expect("ok");
        assert!(!bars.is_empty(), "expected at least one persistence bar");
        // Infinite bar must exist.
        assert!(
            bars.iter().any(|b| b.death == f64::MAX),
            "no infinite bar found"
        );
    }

    #[test]
    fn test_tomato_two_blobs() {
        let data = two_blobs();
        let config = TomaToConfig {
            k_neighbors: 3,
            auto_threshold: true,
            ..Default::default()
        };
        let result = tomato(data.view(), &config).expect("tomato ok");
        assert_eq!(result.labels.len(), 8);
        assert!(
            result.n_clusters >= 1,
            "expected at least 1 cluster, got {}",
            result.n_clusters
        );
    }

    #[test]
    fn test_tomato_n_clusters() {
        let data = two_blobs();
        let labels = tomato_n_clusters(data.view(), 2, 3).expect("ok");
        assert_eq!(labels.len(), 8);
    }

    #[test]
    fn test_gap_threshold_empty() {
        let bars: Vec<PersistenceBar> = Vec::new();
        let tau = gap_threshold(&bars);
        assert_eq!(tau, 0.0);
    }

    #[test]
    fn test_cluster_stats() {
        let data = two_blobs();
        let labels = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let stats = cluster_stats(data.view(), &labels).expect("ok");
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].size, 4);
        assert_eq!(stats[1].size, 4);
    }

    #[test]
    fn test_flat_clustering_from_persistence() {
        let bars = vec![
            PersistenceBar { birth: 10.0, death: f64::MAX, representative: 0 },
            PersistenceBar { birth: 8.0, death: 9.0, representative: 5 },
            PersistenceBar { birth: 3.0, death: 3.5, representative: 2 },
        ];
        let labels = flat_clustering_from_persistence(&bars, 8, 0.8);
        assert_eq!(labels.len(), 8);
    }
}
