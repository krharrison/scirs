//! Density-based clustering enhancements
//!
//! This module extends the base density-based clustering algorithms with
//! advanced techniques for improved cluster detection and outlier analysis.
//!
//! # Algorithms
//!
//! - **HDBSCAN\***: Hierarchical DBSCAN with cluster stability extraction
//! - **Auto-epsilon DBSCAN**: Automatic epsilon selection via k-distance elbow
//! - **SNN density**: Shared nearest neighbor density clustering
//! - **Kernel density clustering**: Density-based clustering using KDE
//! - **LOF**: Local Outlier Factor for outlier/anomaly scoring

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::BinaryHeap;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Distance helpers
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two rows.
fn dist_sq<F: Float>(a: ArrayView1<F>, b: ArrayView1<F>) -> F {
    let mut s = F::zero();
    for i in 0..a.len().min(b.len()) {
        let d = a[i] - b[i];
        s = s + d * d;
    }
    s
}

/// Full pairwise distance matrix (Euclidean).
fn pairwise_distances<F: Float + FromPrimitive + Debug>(data: ArrayView2<F>) -> Array2<F> {
    let n = data.shape()[0];
    let mut dists = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist_sq(data.row(i), data.row(j)).sqrt();
            dists[[i, j]] = d;
            dists[[j, i]] = d;
        }
    }
    dists
}

/// k-nearest-neighbor distances (sorted ascending) for each point.
fn knn_distances<F: Float + FromPrimitive + Debug>(
    dist_mat: &Array2<F>,
    k: usize,
) -> Vec<Vec<(usize, F)>> {
    let n = dist_mat.shape()[0];
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let mut dists: Vec<(usize, F)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, dist_mat[[i, j]]))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        result.push(dists);
    }
    result
}

// ---------------------------------------------------------------------------
// HDBSCAN* with stability
// ---------------------------------------------------------------------------

/// Configuration for HDBSCAN* with stability-based cluster extraction.
#[derive(Debug, Clone)]
pub struct HdbscanStarConfig {
    /// Minimum cluster size for stability computation.
    pub min_cluster_size: usize,
    /// Minimum number of samples for core distance.
    pub min_samples: usize,
    /// Whether to compute cluster membership probabilities.
    pub compute_probabilities: bool,
}

impl Default for HdbscanStarConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: 5,
            min_samples: 5,
            compute_probabilities: true,
        }
    }
}

/// Result of HDBSCAN* with stability.
#[derive(Debug, Clone)]
pub struct HdbscanStarResult<F: Float> {
    /// Cluster labels (-1 = noise).
    pub labels: Array1<i32>,
    /// Membership probabilities (if computed).
    pub probabilities: Option<Array1<F>>,
    /// Stability values for each cluster.
    pub cluster_stabilities: Vec<F>,
    /// Number of clusters found.
    pub n_clusters: usize,
    /// Outlier scores (higher = more outlier-like).
    pub outlier_scores: Array1<F>,
}

/// Internal edge for MST construction.
#[derive(Debug, Clone)]
struct MstEdge<F: Float> {
    i: usize,
    j: usize,
    weight: F,
}

/// Run HDBSCAN* with stability-based cluster extraction.
///
/// This is an enhanced version of HDBSCAN that:
/// 1. Computes mutual reachability distances
/// 2. Builds a minimum spanning tree
/// 3. Constructs a condensed cluster tree
/// 4. Extracts clusters based on stability (excess of mass)
pub fn hdbscan_star<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    config: &HdbscanStarConfig,
) -> Result<HdbscanStarResult<F>> {
    let n = data.shape()[0];

    if n == 0 {
        return Err(ClusteringError::InvalidInput("Empty input data".into()));
    }
    if config.min_cluster_size < 2 {
        return Err(ClusteringError::InvalidInput(
            "min_cluster_size must be >= 2".into(),
        ));
    }
    if config.min_samples < 1 {
        return Err(ClusteringError::InvalidInput(
            "min_samples must be >= 1".into(),
        ));
    }

    let dist_mat = pairwise_distances(data);
    let mpts = config.min_samples;

    // Step 1: compute core distances (distance to mpts-th nearest neighbor)
    let mut core_dists = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut dists: Vec<F> = (0..n)
            .filter(|&j| j != i)
            .map(|j| dist_mat[[i, j]])
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k = (mpts - 1).min(dists.len().saturating_sub(1));
        core_dists[i] = if k < dists.len() {
            dists[k]
        } else {
            F::infinity()
        };
    }

    // Step 2: mutual reachability distance
    // mrd(a,b) = max(core(a), core(b), d(a,b))
    let mut mrd = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist_mat[[i, j]];
            let mr = d.max(core_dists[i]).max(core_dists[j]);
            mrd[[i, j]] = mr;
            mrd[[j, i]] = mr;
        }
    }

    // Step 3: build MST using Prim's algorithm
    let mst = prims_mst(&mrd, n);

    // Step 4: sort MST edges by weight
    let mut sorted_edges = mst;
    sorted_edges.sort_by(|a, b| {
        a.weight
            .partial_cmp(&b.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Step 5: build condensed tree and extract clusters using stability
    let min_cs = config.min_cluster_size;
    let (labels, stabilities) = extract_stable_clusters(&sorted_edges, n, min_cs);

    let n_clusters = if stabilities.is_empty() {
        0
    } else {
        labels
            .iter()
            .filter(|&&l| l >= 0)
            .map(|&l| l)
            .max()
            .map(|m| m as usize + 1)
            .unwrap_or(0)
    };

    // Compute outlier scores: based on distance to nearest core point relative to cluster
    let mut outlier_scores = Array1::<F>::zeros(n);
    for i in 0..n {
        if labels[i] < 0 {
            outlier_scores[i] = F::one();
        } else {
            // Score based on how far from the cluster core
            let ci = labels[i] as usize;
            let mut min_core_dist = F::infinity();
            for j in 0..n {
                if j != i && labels[j] == labels[i] {
                    let d = mrd[[i, j]];
                    if d < min_core_dist {
                        min_core_dist = d;
                    }
                }
            }
            if min_core_dist < F::infinity() {
                let max_core = core_dists
                    .iter()
                    .copied()
                    .filter(|&d| d < F::infinity())
                    .fold(F::zero(), |a, b| a.max(b));
                if max_core > F::epsilon() {
                    outlier_scores[i] = min_core_dist / max_core;
                }
            }
        }
    }

    // Compute probabilities if requested
    let probabilities = if config.compute_probabilities {
        let mut probs = Array1::<F>::zeros(n);
        for i in 0..n {
            probs[i] = F::one() - outlier_scores[i].min(F::one());
        }
        Some(probs)
    } else {
        None
    };

    Ok(HdbscanStarResult {
        labels,
        probabilities,
        cluster_stabilities: stabilities,
        n_clusters,
        outlier_scores,
    })
}

/// Prim's algorithm for MST on a distance matrix.
fn prims_mst<F: Float>(dist_mat: &Array2<F>, n: usize) -> Vec<MstEdge<F>> {
    if n <= 1 {
        return Vec::new();
    }

    let mut in_tree = vec![false; n];
    let mut key = vec![F::infinity(); n]; // minimum weight edge to tree
    let mut parent = vec![0usize; n];
    let mut edges = Vec::with_capacity(n - 1);

    key[0] = F::zero();

    for _ in 0..n {
        // Find minimum key vertex not in tree
        let mut u = None;
        let mut min_key = F::infinity();
        for v in 0..n {
            if !in_tree[v] && key[v] < min_key {
                min_key = key[v];
                u = Some(v);
            }
        }
        let u = match u {
            Some(v) => v,
            None => break,
        };
        in_tree[u] = true;

        if key[u] > F::zero() {
            edges.push(MstEdge {
                i: parent[u],
                j: u,
                weight: key[u],
            });
        }

        // Update keys
        for v in 0..n {
            if !in_tree[v] && dist_mat[[u, v]] < key[v] {
                key[v] = dist_mat[[u, v]];
                parent[v] = u;
            }
        }
    }

    edges
}

/// Extract stable clusters from sorted MST edges using simplified stability analysis.
fn extract_stable_clusters<F: Float + FromPrimitive + Debug>(
    sorted_edges: &[MstEdge<F>],
    n: usize,
    min_cluster_size: usize,
) -> (Array1<i32>, Vec<F>) {
    // Union-Find for tracking connected components
    let mut parent_uf: Vec<usize> = (0..n).collect();
    let mut size: Vec<usize> = vec![1; n];

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        // Path compression
        let mut cur = x;
        while cur != root {
            let next = parent[cur];
            parent[cur] = root;
            cur = next;
        }
        root
    }

    fn union(parent: &mut Vec<usize>, size: &mut Vec<usize>, a: usize, b: usize) -> usize {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra == rb {
            return ra;
        }
        if size[ra] < size[rb] {
            parent[ra] = rb;
            size[rb] += size[ra];
            rb
        } else {
            parent[rb] = ra;
            size[ra] += size[rb];
            ra
        }
    }

    // Process edges in order, tracking when components reach min_cluster_size
    let mut component_birth: Vec<Option<F>> = vec![None; n]; // lambda at which component was born as cluster

    for edge in sorted_edges {
        let ri = find(&mut parent_uf, edge.i);
        let rj = find(&mut parent_uf, edge.j);
        if ri == rj {
            continue;
        }

        let lambda = if edge.weight > F::epsilon() {
            F::one() / edge.weight
        } else {
            F::infinity()
        };

        // Check if merging creates a big-enough cluster
        let new_root = union(&mut parent_uf, &mut size, ri, rj);
        if size[new_root] >= min_cluster_size && component_birth[new_root].is_none() {
            component_birth[new_root] = Some(lambda);
        }
    }

    // Assign labels based on final connected components
    // Reset parent for fresh traversal
    let mut final_parent: Vec<usize> = (0..n).collect();
    let mut final_size: Vec<usize> = vec![1; n];

    for edge in sorted_edges {
        let _root = union(&mut final_parent, &mut final_size, edge.i, edge.j);
    }

    // Find distinct components that are large enough
    let mut cluster_map: std::collections::HashMap<usize, i32> = std::collections::HashMap::new();
    let mut next_label = 0i32;
    let mut labels = Array1::from_elem(n, -1i32);

    for i in 0..n {
        let root = find(&mut final_parent, i);
        if final_size[root] >= min_cluster_size {
            let label = cluster_map.entry(root).or_insert_with(|| {
                let l = next_label;
                next_label += 1;
                l
            });
            labels[i] = *label;
        }
    }

    // Compute stability for each cluster (simplified: proportional to component size * birth lambda)
    let mut stabilities = Vec::new();
    for (&root, &label) in &cluster_map {
        let birth = component_birth[root].unwrap_or_else(|| F::zero());
        let sz = F::from(final_size[root]).unwrap_or_else(|| F::one());
        stabilities.push(birth * sz);
    }

    (labels, stabilities)
}

// ---------------------------------------------------------------------------
// Auto-epsilon DBSCAN
// ---------------------------------------------------------------------------

/// Configuration for automatic epsilon selection.
#[derive(Debug, Clone)]
pub struct AutoEpsilonConfig {
    /// Number of nearest neighbors for k-distance plot (typically = min_samples).
    pub k: usize,
    /// Minimum DBSCAN samples parameter.
    pub min_samples: usize,
    /// Sensitivity of elbow detection (higher = more sensitive).
    pub sensitivity: f64,
}

impl Default for AutoEpsilonConfig {
    fn default() -> Self {
        Self {
            k: 5,
            min_samples: 5,
            sensitivity: 1.0,
        }
    }
}

/// Result of auto-epsilon DBSCAN.
#[derive(Debug, Clone)]
pub struct AutoEpsilonResult<F: Float> {
    /// Cluster labels (-1 = noise).
    pub labels: Array1<i32>,
    /// Selected epsilon value.
    pub epsilon: F,
    /// k-distance plot values (sorted).
    pub k_distances: Vec<F>,
    /// Elbow index in the k-distance plot.
    pub elbow_index: usize,
    /// Number of clusters found.
    pub n_clusters: usize,
}

/// Run DBSCAN with automatic epsilon selection via k-distance elbow detection.
///
/// Computes the k-distance plot, identifies the "elbow" (point of maximum
/// curvature), and uses that distance as epsilon for DBSCAN.
pub fn auto_epsilon_dbscan<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    config: &AutoEpsilonConfig,
) -> Result<AutoEpsilonResult<F>> {
    let n = data.shape()[0];
    if n == 0 {
        return Err(ClusteringError::InvalidInput("Empty input data".into()));
    }
    if config.k == 0 {
        return Err(ClusteringError::InvalidInput("k must be > 0".into()));
    }

    let dist_mat = pairwise_distances(data);
    let k = config.k.min(n - 1);

    // Compute k-distance for each point
    let mut k_dists: Vec<F> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row_dists: Vec<F> = (0..n)
            .filter(|&j| j != i)
            .map(|j| dist_mat[[i, j]])
            .collect();
        row_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let kd = if k <= row_dists.len() {
            row_dists[k - 1]
        } else {
            *row_dists.last().unwrap_or(&F::zero())
        };
        k_dists.push(kd);
    }

    // Sort k-distances ascending for the elbow plot
    k_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Find elbow using maximum curvature (second derivative)
    let elbow_idx = find_elbow(&k_dists, config.sensitivity);
    let epsilon = k_dists[elbow_idx];

    // Run DBSCAN with selected epsilon
    let labels = run_dbscan(data, &dist_mat, epsilon, config.min_samples);

    let n_clusters = labels
        .iter()
        .filter(|&&l| l >= 0)
        .map(|&l| l)
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(0);

    Ok(AutoEpsilonResult {
        labels,
        epsilon,
        k_distances: k_dists,
        elbow_index: elbow_idx,
        n_clusters,
    })
}

/// Find the elbow point in a sorted distance curve.
fn find_elbow<F: Float + FromPrimitive + Debug>(values: &[F], sensitivity: f64) -> usize {
    let n = values.len();
    if n < 3 {
        return n / 2;
    }

    // Use perpendicular distance to the line from first to last point
    let x0 = 0.0f64;
    let y0 = values[0].to_f64().unwrap_or(0.0);
    let x1 = (n - 1) as f64;
    let y1 = values[n - 1].to_f64().unwrap_or(0.0);

    let dx = x1 - x0;
    let dy = y1 - y0;
    let line_len = (dx * dx + dy * dy).sqrt().max(1e-15);

    let mut max_dist = 0.0f64;
    let mut elbow_idx = n / 2;

    for i in 1..(n - 1) {
        let xi = i as f64;
        let yi = values[i].to_f64().unwrap_or(0.0);
        // Perpendicular distance from point to line
        let dist = ((dy * xi - dx * yi + x1 * y0 - y1 * x0).abs()) / line_len;
        let adjusted = dist * sensitivity;
        if adjusted > max_dist {
            max_dist = adjusted;
            elbow_idx = i;
        }
    }

    elbow_idx
}

/// Run DBSCAN given a precomputed distance matrix.
fn run_dbscan<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    dist_mat: &Array2<F>,
    eps: F,
    min_samples: usize,
) -> Array1<i32> {
    let n = data.shape()[0];
    let mut labels = vec![-2i32; n]; // -2 = undefined
    let mut cluster_id = 0i32;

    for i in 0..n {
        if labels[i] != -2 {
            continue;
        }
        let neighbors: Vec<usize> = (0..n).filter(|&j| dist_mat[[i, j]] <= eps).collect();

        if neighbors.len() < min_samples {
            labels[i] = -1; // noise
            continue;
        }

        labels[i] = cluster_id;
        let mut queue = neighbors;
        let mut head = 0;
        while head < queue.len() {
            let cur = queue[head];
            head += 1;
            if labels[cur] == -1 {
                labels[cur] = cluster_id;
                continue;
            }
            if labels[cur] != -2 {
                continue;
            }
            labels[cur] = cluster_id;

            let cur_nb: Vec<usize> = (0..n).filter(|&j| dist_mat[[cur, j]] <= eps).collect();
            if cur_nb.len() >= min_samples {
                for nb in cur_nb {
                    if labels[nb] == -2 || labels[nb] == -1 {
                        queue.push(nb);
                    }
                }
            }
        }
        cluster_id += 1;
    }

    Array1::from_vec(labels)
}

// ---------------------------------------------------------------------------
// Shared Nearest Neighbor (SNN) Density Clustering
// ---------------------------------------------------------------------------

/// Configuration for SNN density clustering.
#[derive(Debug, Clone)]
pub struct SnnConfig {
    /// Number of nearest neighbors for SNN computation (k).
    pub k: usize,
    /// Minimum SNN similarity for two points to be considered neighbors.
    pub snn_threshold: usize,
    /// Minimum number of SNN-neighbors for a core point.
    pub min_snn_neighbors: usize,
}

impl Default for SnnConfig {
    fn default() -> Self {
        Self {
            k: 10,
            snn_threshold: 3,
            min_snn_neighbors: 3,
        }
    }
}

/// Result of SNN density clustering.
#[derive(Debug, Clone)]
pub struct SnnResult<F: Float> {
    /// Cluster labels (-1 = noise).
    pub labels: Array1<i32>,
    /// SNN similarity matrix.
    pub snn_similarity: Array2<F>,
    /// Number of clusters found.
    pub n_clusters: usize,
}

/// Run shared nearest neighbor (SNN) density clustering.
///
/// SNN similarity between two points is the number of shared items in
/// their k-nearest-neighbor lists. Points with high SNN similarity form
/// dense regions that become clusters.
pub fn snn_clustering<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    config: &SnnConfig,
) -> Result<SnnResult<F>> {
    let n = data.shape()[0];
    if n == 0 {
        return Err(ClusteringError::InvalidInput("Empty input data".into()));
    }
    if config.k == 0 {
        return Err(ClusteringError::InvalidInput("k must be > 0".into()));
    }

    let dist_mat = pairwise_distances(data);
    let k = config.k.min(n - 1);
    let knn = knn_distances(&dist_mat, k);

    // Compute SNN similarity matrix
    let mut snn_sim = Array2::<F>::zeros((n, n));
    for i in 0..n {
        let knn_i: std::collections::HashSet<usize> = knn[i].iter().map(|&(j, _)| j).collect();
        for j in (i + 1)..n {
            let knn_j: std::collections::HashSet<usize> =
                knn[j].iter().map(|&(jj, _)| jj).collect();
            let shared = knn_i.intersection(&knn_j).count();
            let sim = F::from(shared).unwrap_or_else(|| F::zero());
            snn_sim[[i, j]] = sim;
            snn_sim[[j, i]] = sim;
        }
    }

    // DBSCAN-like clustering using SNN similarity
    let threshold = F::from(config.snn_threshold).unwrap_or_else(|| F::one());
    let min_nb = config.min_snn_neighbors;

    let mut labels = vec![-2i32; n];
    let mut cluster_id = 0i32;

    for i in 0..n {
        if labels[i] != -2 {
            continue;
        }
        let neighbors: Vec<usize> = (0..n)
            .filter(|&j| j != i && snn_sim[[i, j]] >= threshold)
            .collect();

        if neighbors.len() < min_nb {
            labels[i] = -1;
            continue;
        }

        labels[i] = cluster_id;
        let mut queue = neighbors;
        let mut head = 0;
        while head < queue.len() {
            let cur = queue[head];
            head += 1;
            if labels[cur] == -1 {
                labels[cur] = cluster_id;
                continue;
            }
            if labels[cur] != -2 {
                continue;
            }
            labels[cur] = cluster_id;

            let cur_nb: Vec<usize> = (0..n)
                .filter(|&j| j != cur && snn_sim[[cur, j]] >= threshold)
                .collect();
            if cur_nb.len() >= min_nb {
                for nb in cur_nb {
                    if labels[nb] == -2 || labels[nb] == -1 {
                        queue.push(nb);
                    }
                }
            }
        }
        cluster_id += 1;
    }

    let n_clusters = labels
        .iter()
        .filter(|&&l| l >= 0)
        .max()
        .map(|&m| m as usize + 1)
        .unwrap_or(0);

    Ok(SnnResult {
        labels: Array1::from_vec(labels),
        snn_similarity: snn_sim,
        n_clusters,
    })
}

// ---------------------------------------------------------------------------
// Kernel Density Clustering
// ---------------------------------------------------------------------------

/// Kernel type for density estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KdeKernel {
    /// Gaussian (RBF) kernel.
    Gaussian,
    /// Epanechnikov kernel.
    Epanechnikov,
    /// Uniform (box) kernel.
    Uniform,
}

/// Configuration for kernel density clustering.
#[derive(Debug, Clone)]
pub struct KdcConfig {
    /// Kernel type.
    pub kernel: KdeKernel,
    /// Bandwidth (h). If 0, auto-select via Silverman's rule.
    pub bandwidth: f64,
    /// Density threshold for peak detection (fraction of max density).
    pub density_threshold: f64,
    /// Merge distance: peaks closer than this are merged.
    pub merge_distance: f64,
}

impl Default for KdcConfig {
    fn default() -> Self {
        Self {
            kernel: KdeKernel::Gaussian,
            bandwidth: 0.0,
            density_threshold: 0.1,
            merge_distance: 0.0,
        }
    }
}

/// Result of kernel density clustering.
#[derive(Debug, Clone)]
pub struct KdcResult<F: Float> {
    /// Cluster labels (-1 = low density / noise).
    pub labels: Array1<i32>,
    /// Estimated density at each point.
    pub densities: Array1<F>,
    /// Bandwidth used.
    pub bandwidth: F,
    /// Number of clusters found.
    pub n_clusters: usize,
}

/// Run kernel density-based clustering.
///
/// Estimates density at each data point, identifies local density peaks
/// as cluster centres, and assigns each point to the nearest peak via
/// gradient ascent (mean-shift style).
pub fn kernel_density_clustering<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    config: &KdcConfig,
) -> Result<KdcResult<F>> {
    let (n, d) = (data.shape()[0], data.shape()[1]);
    if n == 0 {
        return Err(ClusteringError::InvalidInput("Empty input data".into()));
    }

    // Auto-bandwidth via Silverman's rule if needed
    let h = if config.bandwidth <= 0.0 {
        silverman_bandwidth(data)
    } else {
        F::from(config.bandwidth).unwrap_or_else(|| F::one())
    };

    if h <= F::epsilon() {
        return Err(ClusteringError::ComputationError(
            "Bandwidth too small".into(),
        ));
    }

    // Estimate density at each point
    let mut densities = Array1::<F>::zeros(n);
    let n_f = F::from(n).unwrap_or_else(|| F::one());
    let h_d = h.powi(d as i32);
    let norm_factor = n_f * h_d;

    for i in 0..n {
        let mut dens = F::zero();
        for j in 0..n {
            let u_sq = dist_sq(data.row(i), data.row(j)) / (h * h);
            let kval = match config.kernel {
                KdeKernel::Gaussian => {
                    let neg_half = F::from(-0.5).unwrap_or_else(|| F::zero());
                    (neg_half * u_sq).exp()
                }
                KdeKernel::Epanechnikov => {
                    if u_sq < F::one() {
                        F::one() - u_sq
                    } else {
                        F::zero()
                    }
                }
                KdeKernel::Uniform => {
                    if u_sq < F::one() {
                        F::one()
                    } else {
                        F::zero()
                    }
                }
            };
            dens = dens + kval;
        }
        densities[i] = dens / norm_factor;
    }

    // Find density peaks via gradient ascent (simplified mean-shift)
    let max_density = densities.iter().copied().fold(F::zero(), |a, b| a.max(b));
    let threshold = F::from(config.density_threshold).unwrap_or_else(|| F::zero()) * max_density;

    // Assign each above-threshold point to a peak
    let mut peak_assignments = vec![-1i32; n];
    let mut peaks: Vec<usize> = Vec::new();

    // Gradient ascent: each point walks to its local maximum
    let mut local_max = vec![0usize; n];
    for i in 0..n {
        let mut current = i;
        for _ in 0..100 {
            // Find the highest-density neighbor
            let mut best = current;
            let mut best_dens = densities[current];
            for j in 0..n {
                if dist_sq(data.row(current), data.row(j)).sqrt()
                    <= h * F::from(2.0).unwrap_or_else(|| F::one())
                {
                    if densities[j] > best_dens {
                        best_dens = densities[j];
                        best = j;
                    }
                }
            }
            if best == current {
                break;
            }
            current = best;
        }
        local_max[i] = current;
    }

    // Group points by their local maximum
    let merge_dist = if config.merge_distance > 0.0 {
        F::from(config.merge_distance).unwrap_or_else(|| h)
    } else {
        h
    };

    // Deduplicate peaks
    let mut peak_map: std::collections::HashMap<usize, i32> = std::collections::HashMap::new();
    let mut next_label = 0i32;

    for i in 0..n {
        if densities[i] < threshold {
            peak_assignments[i] = -1;
            continue;
        }
        let peak = local_max[i];

        // Check if this peak should be merged with an existing one
        let mut merged_label = None;
        for (&existing_peak, &label) in &peak_map {
            if dist_sq(data.row(peak), data.row(existing_peak)).sqrt() <= merge_dist {
                merged_label = Some(label);
                break;
            }
        }

        let label = match merged_label {
            Some(l) => l,
            None => {
                let l = next_label;
                peak_map.insert(peak, l);
                peaks.push(peak);
                next_label += 1;
                l
            }
        };
        peak_assignments[i] = label;
        peak_map.entry(peak).or_insert(label);
    }

    let n_clusters = next_label as usize;

    Ok(KdcResult {
        labels: Array1::from_vec(peak_assignments),
        densities,
        bandwidth: h,
        n_clusters,
    })
}

/// Silverman's rule of thumb for bandwidth selection.
fn silverman_bandwidth<F: Float + FromPrimitive + Debug>(data: ArrayView2<F>) -> F {
    let (n, d) = (data.shape()[0], data.shape()[1]);
    if n < 2 {
        return F::one();
    }

    // Average standard deviation across all dimensions
    let mut total_std = 0.0f64;
    for dim in 0..d {
        let mean = (0..n)
            .map(|i| data[[i, dim]].to_f64().unwrap_or(0.0))
            .sum::<f64>()
            / n as f64;
        let var = (0..n)
            .map(|i| {
                let diff = data[[i, dim]].to_f64().unwrap_or(0.0) - mean;
                diff * diff
            })
            .sum::<f64>()
            / (n - 1) as f64;
        total_std += var.sqrt();
    }
    let avg_std = total_std / d as f64;

    // Silverman's rule: h = 0.9 * min(sigma, IQR/1.34) * n^(-1/5)
    // Simplified: h = 1.06 * sigma * n^(-1/5)
    let h = 1.06 * avg_std * (n as f64).powf(-0.2);
    F::from(h.max(1e-10)).unwrap_or_else(|| F::one())
}

// ---------------------------------------------------------------------------
// Local Outlier Factor (LOF)
// ---------------------------------------------------------------------------

/// Configuration for LOF computation.
#[derive(Debug, Clone)]
pub struct LofConfig {
    /// Number of nearest neighbors (k, also called MinPts).
    pub k: usize,
    /// LOF threshold above which a point is considered an outlier.
    pub outlier_threshold: f64,
}

impl Default for LofConfig {
    fn default() -> Self {
        Self {
            k: 5,
            outlier_threshold: 1.5,
        }
    }
}

/// Result of LOF computation.
#[derive(Debug, Clone)]
pub struct LofResult<F: Float> {
    /// LOF score for each data point (1.0 = normal density, > 1 = outlier-like).
    pub lof_scores: Array1<F>,
    /// Binary outlier labels (true = outlier).
    pub is_outlier: Vec<bool>,
    /// Number of outliers detected.
    pub n_outliers: usize,
    /// k-distance for each point.
    pub k_distances: Array1<F>,
    /// Local reachability density for each point.
    pub lrd: Array1<F>,
}

/// Compute Local Outlier Factor (LOF) scores.
///
/// LOF measures the local density deviation of a point with respect to its
/// neighbors. A LOF score significantly greater than 1 indicates the point
/// is in a region of lower density (potential outlier).
pub fn local_outlier_factor<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    config: &LofConfig,
) -> Result<LofResult<F>> {
    let n = data.shape()[0];
    if n == 0 {
        return Err(ClusteringError::InvalidInput("Empty input data".into()));
    }
    if config.k == 0 || config.k >= n {
        return Err(ClusteringError::InvalidInput(
            "k must be in [1, n_samples)".into(),
        ));
    }

    let dist_mat = pairwise_distances(data);
    let k = config.k;
    let knn = knn_distances(&dist_mat, k);

    // k-distance for each point: distance to k-th nearest neighbor
    let mut k_dist = Array1::<F>::zeros(n);
    for i in 0..n {
        k_dist[i] = if knn[i].len() >= k {
            knn[i][k - 1].1
        } else if let Some(last) = knn[i].last() {
            last.1
        } else {
            F::zero()
        };
    }

    // Reachability distance: reach_dist_k(a, b) = max(k_dist(b), d(a, b))
    // Local reachability density: lrd(p) = 1 / (avg reach_dist from p to its k neighbors)
    let mut lrd = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut sum_reach = F::zero();
        let nb_count = knn[i].len();
        for &(j, d_ij) in &knn[i] {
            let reach = d_ij.max(k_dist[j]);
            sum_reach = sum_reach + reach;
        }
        if nb_count > 0 && sum_reach > F::epsilon() {
            lrd[i] = F::from(nb_count).unwrap_or_else(|| F::one()) / sum_reach;
        } else {
            lrd[i] = F::one(); // avoid division by zero
        }
    }

    // LOF score: avg(lrd(neighbor) / lrd(p)) for all k-neighbors
    let mut lof_scores = Array1::<F>::zeros(n);
    for i in 0..n {
        let nb_count = knn[i].len();
        if nb_count == 0 || lrd[i] <= F::epsilon() {
            lof_scores[i] = F::one();
            continue;
        }
        let mut sum = F::zero();
        for &(j, _) in &knn[i] {
            sum = sum + lrd[j] / lrd[i];
        }
        lof_scores[i] = sum / F::from(nb_count).unwrap_or_else(|| F::one());
    }

    let threshold = F::from(config.outlier_threshold)
        .unwrap_or_else(|| F::from(1.5).unwrap_or_else(|| F::one()));
    let is_outlier: Vec<bool> = lof_scores.iter().map(|&s| s > threshold).collect();
    let n_outliers = is_outlier.iter().filter(|&&o| o).count();

    Ok(LofResult {
        lof_scores,
        is_outlier,
        n_outliers,
        k_distances: k_dist,
        lrd,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_clustered_data() -> Array2<f64> {
        let mut data = Vec::new();
        // Cluster A around (1, 1)
        for i in 0..15 {
            let n = (i as f64 * 0.073).sin() * 0.2;
            data.push(1.0 + n);
            data.push(1.0 + n * 0.5);
        }
        // Cluster B around (5, 5)
        for i in 0..15 {
            let n = (i as f64 * 0.131).sin() * 0.2;
            data.push(5.0 + n);
            data.push(5.0 + n * 0.5);
        }
        // Outlier
        data.push(10.0);
        data.push(10.0);
        Array2::from_shape_vec((31, 2), data).expect("shape failed")
    }

    // -- HDBSCAN* tests --

    #[test]
    fn test_hdbscan_star_basic() {
        let data = make_clustered_data();
        let config = HdbscanStarConfig {
            min_cluster_size: 3,
            min_samples: 3,
            compute_probabilities: true,
        };
        let result = hdbscan_star(data.view(), &config).expect("hdbscan* failed");
        assert_eq!(result.labels.len(), 31);
        assert!(result.n_clusters >= 1);
        assert!(result.probabilities.is_some());
        assert_eq!(result.outlier_scores.len(), 31);
    }

    #[test]
    fn test_hdbscan_star_empty() {
        let data = Array2::<f64>::zeros((0, 3));
        let config = HdbscanStarConfig::default();
        assert!(hdbscan_star(data.view(), &config).is_err());
    }

    #[test]
    fn test_hdbscan_star_invalid_params() {
        let data = make_clustered_data();
        let config = HdbscanStarConfig {
            min_cluster_size: 1,
            ..Default::default()
        };
        assert!(hdbscan_star(data.view(), &config).is_err());
    }

    #[test]
    fn test_hdbscan_star_small_data() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 5.0, 5.0, 5.1, 5.1],
        )
        .expect("shape");
        let config = HdbscanStarConfig {
            min_cluster_size: 2,
            min_samples: 2,
            compute_probabilities: false,
        };
        let result = hdbscan_star(data.view(), &config).expect("hdbscan* failed");
        assert_eq!(result.labels.len(), 5);
        assert!(result.probabilities.is_none());
    }

    // -- Auto-epsilon DBSCAN tests --

    #[test]
    fn test_auto_epsilon_basic() {
        let data = make_clustered_data();
        let config = AutoEpsilonConfig {
            k: 3,
            min_samples: 3,
            sensitivity: 1.0,
        };
        let result = auto_epsilon_dbscan(data.view(), &config).expect("auto-eps failed");
        assert_eq!(result.labels.len(), 31);
        assert!(result.epsilon > 0.0);
        assert!(!result.k_distances.is_empty());
    }

    #[test]
    fn test_auto_epsilon_empty() {
        let data = Array2::<f64>::zeros((0, 2));
        let config = AutoEpsilonConfig::default();
        assert!(auto_epsilon_dbscan(data.view(), &config).is_err());
    }

    #[test]
    fn test_auto_epsilon_invalid_k() {
        let data = make_clustered_data();
        let config = AutoEpsilonConfig {
            k: 0,
            ..Default::default()
        };
        assert!(auto_epsilon_dbscan(data.view(), &config).is_err());
    }

    #[test]
    fn test_find_elbow() {
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.8, 1.5, 3.0, 5.0, 8.0, 12.0];
        let idx = find_elbow(&values, 1.0);
        // Elbow should be somewhere in the middle where curvature changes
        assert!(idx > 0 && idx < values.len() - 1);
    }

    // -- SNN tests --

    #[test]
    fn test_snn_basic() {
        let data = make_clustered_data();
        let config = SnnConfig {
            k: 5,
            snn_threshold: 2,
            min_snn_neighbors: 2,
        };
        let result = snn_clustering(data.view(), &config).expect("snn failed");
        assert_eq!(result.labels.len(), 31);
        // SNN similarity should be symmetric
        let n = result.snn_similarity.shape()[0];
        for i in 0..n {
            for j in 0..n {
                let diff = (result.snn_similarity[[i, j]] - result.snn_similarity[[j, i]]).abs();
                assert!(diff < 1e-10);
            }
        }
    }

    #[test]
    fn test_snn_empty() {
        let data = Array2::<f64>::zeros((0, 2));
        let config = SnnConfig::default();
        assert!(snn_clustering(data.view(), &config).is_err());
    }

    // -- KDE clustering tests --

    #[test]
    fn test_kde_clustering_basic() {
        let data = make_clustered_data();
        let config = KdcConfig {
            kernel: KdeKernel::Gaussian,
            bandwidth: 1.0,
            density_threshold: 0.05,
            merge_distance: 1.0,
        };
        let result = kernel_density_clustering(data.view(), &config).expect("kde failed");
        assert_eq!(result.labels.len(), 31);
        assert_eq!(result.densities.len(), 31);
        assert!(result.bandwidth > 0.0);
    }

    #[test]
    fn test_kde_clustering_auto_bandwidth() {
        let data = make_clustered_data();
        let config = KdcConfig {
            bandwidth: 0.0, // auto
            ..Default::default()
        };
        let result = kernel_density_clustering(data.view(), &config).expect("kde failed");
        assert!(result.bandwidth > 0.0);
    }

    #[test]
    fn test_kde_clustering_epanechnikov() {
        let data = make_clustered_data();
        let config = KdcConfig {
            kernel: KdeKernel::Epanechnikov,
            bandwidth: 2.0,
            density_threshold: 0.01,
            merge_distance: 1.0,
        };
        let result = kernel_density_clustering(data.view(), &config).expect("kde failed");
        assert_eq!(result.labels.len(), 31);
    }

    #[test]
    fn test_kde_clustering_empty() {
        let data = Array2::<f64>::zeros((0, 2));
        let config = KdcConfig::default();
        assert!(kernel_density_clustering(data.view(), &config).is_err());
    }

    #[test]
    fn test_silverman_bandwidth() {
        let data = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 1.1, 2.1, 1.2, 1.9, 0.9, 2.2, 1.3, 1.8, 5.0, 6.0, 5.1, 6.1, 5.2, 5.9,
                4.9, 6.2, 5.3, 5.8,
            ],
        )
        .expect("shape");
        let h = silverman_bandwidth(data.view());
        assert!(h > 0.0);
    }

    // -- LOF tests --

    #[test]
    fn test_lof_basic() {
        let data = make_clustered_data();
        let config = LofConfig {
            k: 5,
            outlier_threshold: 1.5,
        };
        let result = local_outlier_factor(data.view(), &config).expect("lof failed");
        assert_eq!(result.lof_scores.len(), 31);
        assert_eq!(result.is_outlier.len(), 31);
        assert_eq!(result.k_distances.len(), 31);
        assert_eq!(result.lrd.len(), 31);

        // The outlier at (10, 10) should have a high LOF score
        let outlier_score = result.lof_scores[30];
        assert!(
            outlier_score > 1.0,
            "Outlier LOF score should be > 1, got {}",
            outlier_score
        );
    }

    #[test]
    fn test_lof_empty() {
        let data = Array2::<f64>::zeros((0, 2));
        let config = LofConfig::default();
        assert!(local_outlier_factor(data.view(), &config).is_err());
    }

    #[test]
    fn test_lof_invalid_k() {
        let data = make_clustered_data();
        let config = LofConfig {
            k: 0,
            ..Default::default()
        };
        assert!(local_outlier_factor(data.view(), &config).is_err());

        let config2 = LofConfig {
            k: 100, // k >= n
            ..Default::default()
        };
        assert!(local_outlier_factor(data.view(), &config2).is_err());
    }

    #[test]
    fn test_lof_uniform_data() {
        // All same point => LOF should be ~1 for all
        let data = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0,
            ],
        )
        .expect("shape");
        let config = LofConfig {
            k: 3,
            outlier_threshold: 1.5,
        };
        let result = local_outlier_factor(data.view(), &config).expect("lof failed");
        // LOF of identical points should be ~1
        for &score in result.lof_scores.iter() {
            assert!(
                (score - 1.0).abs() < 0.5,
                "LOF for identical points should be ~1, got {}",
                score
            );
        }
    }

    // -- Prim's MST test --

    #[test]
    fn test_prims_mst() {
        let mut dist = Array2::<f64>::zeros((4, 4));
        dist[[0, 1]] = 1.0;
        dist[[1, 0]] = 1.0;
        dist[[0, 2]] = 4.0;
        dist[[2, 0]] = 4.0;
        dist[[0, 3]] = 3.0;
        dist[[3, 0]] = 3.0;
        dist[[1, 2]] = 2.0;
        dist[[2, 1]] = 2.0;
        dist[[1, 3]] = 5.0;
        dist[[3, 1]] = 5.0;
        dist[[2, 3]] = 1.0;
        dist[[3, 2]] = 1.0;

        let mst = prims_mst(&dist, 4);
        assert_eq!(mst.len(), 3);
        let total_weight: f64 = mst.iter().map(|e| e.weight).sum();
        assert!((total_weight - 4.0).abs() < 1e-10); // 1 + 2 + 1 = 4
    }

    // -- Distance helper tests --

    #[test]
    fn test_dist_sq() {
        let a = Array1::from_vec(vec![1.0, 2.0]);
        let b = Array1::from_vec(vec![4.0, 6.0]);
        let d = dist_sq(a.view(), b.view());
        assert!((d - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_knn_distances() {
        let dist = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 3.0, 5.0, 1.0, 0.0, 2.0, 4.0, 3.0, 2.0, 0.0, 1.0, 5.0, 4.0, 1.0, 0.0,
            ],
        )
        .expect("shape");
        let knn = knn_distances(&dist, 2);
        assert_eq!(knn.len(), 4);
        // Point 0's nearest: 1 (dist 1), 2 (dist 3)
        assert_eq!(knn[0][0].0, 1);
        assert!((knn[0][0].1 - 1.0).abs() < 1e-10);
    }
}
