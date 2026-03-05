//! Subspace clustering algorithms
//!
//! This module provides implementations of clustering algorithms that operate in
//! subspaces of the full feature space, enabling discovery of clusters that exist
//! only in projections of the data.
//!
//! # Algorithms
//!
//! - **CLIQUE** (CLustering In QUEst): Grid-based algorithm for axis-aligned subspaces
//! - **PROCLUS** (PROjected CLUStering): K-medoid based projected clustering
//! - **SSC** (Sparse Subspace Clustering): Self-expression based subspace clustering
//! - **Feature subspace selection**: Methods for finding relevant feature subsets
//! - **Subspace quality metrics**: Evaluate clustering quality in subspaces

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// CLIQUE algorithm
// ---------------------------------------------------------------------------

/// Configuration for the CLIQUE algorithm.
#[derive(Debug, Clone)]
pub struct CliqueConfig {
    /// Number of equal-width intervals per dimension (xi).
    pub n_intervals: usize,
    /// Minimum fraction of total data points for a cell to be dense (tau).
    pub density_threshold: f64,
    /// Maximum subspace dimensionality to explore (0 = unlimited).
    pub max_subspace_dim: usize,
}

impl Default for CliqueConfig {
    fn default() -> Self {
        Self {
            n_intervals: 10,
            density_threshold: 0.1,
            max_subspace_dim: 0,
        }
    }
}

/// A dense cell in the CLIQUE grid.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DenseCell {
    /// Feature indices that define the subspace.
    pub dimensions: Vec<usize>,
    /// Grid coordinate in each dimension.
    pub coords: Vec<usize>,
}

/// Result of the CLIQUE algorithm.
#[derive(Debug, Clone)]
pub struct CliqueResult {
    /// Cluster labels per data point (-1 = not assigned).
    pub labels: Array1<i32>,
    /// Dense subspaces discovered (dimension sets with dense cells).
    pub subspaces: Vec<Vec<usize>>,
    /// Dense cells found (grouped by subspace dimensionality).
    pub dense_cells: Vec<DenseCell>,
    /// Number of clusters found.
    pub n_clusters: usize,
}

/// Run the CLIQUE algorithm for axis-aligned subspace clustering.
///
/// CLIQUE partitions each dimension into equal-width intervals, identifies
/// dense cells, then joins them bottom-up to discover higher-dimensional
/// dense subspaces. Connected dense cells in each subspace form clusters.
pub fn clique<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    config: &CliqueConfig,
) -> Result<CliqueResult> {
    let (n_samples, n_features) = (data.shape()[0], data.shape()[1]);

    if n_samples == 0 {
        return Err(ClusteringError::InvalidInput("Empty input data".into()));
    }
    if config.n_intervals == 0 {
        return Err(ClusteringError::InvalidInput(
            "n_intervals must be > 0".into(),
        ));
    }
    if config.density_threshold <= 0.0 || config.density_threshold > 1.0 {
        return Err(ClusteringError::InvalidInput(
            "density_threshold must be in (0, 1]".into(),
        ));
    }

    let tau_count = (config.density_threshold * n_samples as f64).ceil() as usize;

    // Compute per-dimension min/max for grid construction
    let mut mins = vec![F::infinity(); n_features];
    let mut maxs = vec![F::neg_infinity(); n_features];
    for i in 0..n_samples {
        for d in 0..n_features {
            let v = data[[i, d]];
            if v < mins[d] {
                mins[d] = v;
            }
            if v > maxs[d] {
                maxs[d] = v;
            }
        }
    }

    let xi = config.n_intervals;
    let eps = F::from(1e-10).unwrap_or_else(|| F::epsilon());

    // Map a value in dimension d to a grid index in [0, xi)
    let grid_index = |val: F, d: usize| -> usize {
        let range = maxs[d] - mins[d] + eps;
        let idx_f = ((val - mins[d]) / range) * F::from(xi).unwrap_or_else(|| F::one());
        let idx = idx_f.to_usize().unwrap_or(0).min(xi - 1);
        idx
    };

    // --- Step 1: find 1-D dense cells ---
    let mut one_d_dense: HashMap<usize, HashSet<usize>> = HashMap::new(); // dim -> set of grid coords
    for d in 0..n_features {
        let mut counts = vec![0usize; xi];
        for i in 0..n_samples {
            let idx = grid_index(data[[i, d]], d);
            counts[idx] += 1;
        }
        let dense_coords: HashSet<usize> = counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c >= tau_count)
            .map(|(idx, _)| idx)
            .collect();
        if !dense_coords.is_empty() {
            one_d_dense.insert(d, dense_coords);
        }
    }

    // Collect all dense cells across all dimensionalities
    let mut all_dense_cells: Vec<DenseCell> = Vec::new();
    for (&d, coords) in &one_d_dense {
        for &c in coords {
            all_dense_cells.push(DenseCell {
                dimensions: vec![d],
                coords: vec![c],
            });
        }
    }

    // --- Step 2: bottom-up join to find higher-dimensional dense subspaces ---
    // Current level dense cells keyed by (sorted dims, coords)
    let mut current_level: HashSet<(Vec<usize>, Vec<usize>)> = all_dense_cells
        .iter()
        .map(|c| (c.dimensions.clone(), c.coords.clone()))
        .collect();

    let max_dim = if config.max_subspace_dim == 0 {
        n_features
    } else {
        config.max_subspace_dim.min(n_features)
    };

    // Precompute per-sample grid indices
    let mut sample_grid: Vec<Vec<usize>> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut row = Vec::with_capacity(n_features);
        for d in 0..n_features {
            row.push(grid_index(data[[i, d]], d));
        }
        sample_grid.push(row);
    }

    let mut dim_level = 1usize;

    while dim_level < max_dim && !current_level.is_empty() {
        let mut candidate_set: HashSet<(Vec<usize>, Vec<usize>)> = HashSet::new();

        let cells_vec: Vec<(Vec<usize>, Vec<usize>)> = current_level.iter().cloned().collect();

        // Apriori-style candidate generation: merge cells sharing (k-1) common dims
        for i in 0..cells_vec.len() {
            for j in (i + 1)..cells_vec.len() {
                let (ref dims_a, ref coords_a) = cells_vec[i];
                let (ref dims_b, ref coords_b) = cells_vec[j];

                if dims_a.len() != dim_level || dims_b.len() != dim_level {
                    continue;
                }

                // Check if they share (k-1) common dimensions
                let mut merged_dims: BTreeSet<usize> = BTreeSet::new();
                for &d in dims_a {
                    merged_dims.insert(d);
                }
                for &d in dims_b {
                    merged_dims.insert(d);
                }

                if merged_dims.len() != dim_level + 1 {
                    continue;
                }

                let merged_dims_vec: Vec<usize> = merged_dims.iter().copied().collect();

                // Build merged coord vector
                let mut merged_coords: Vec<usize> = Vec::with_capacity(dim_level + 1);
                for &md in &merged_dims_vec {
                    // Find coordinate for this dimension
                    if let Some(pos) = dims_a.iter().position(|&x| x == md) {
                        merged_coords.push(coords_a[pos]);
                    } else if let Some(pos) = dims_b.iter().position(|&x| x == md) {
                        merged_coords.push(coords_b[pos]);
                    }
                }

                if merged_coords.len() != dim_level + 1 {
                    continue;
                }

                candidate_set.insert((merged_dims_vec, merged_coords));
            }
        }

        // Filter candidates: keep only those that are actually dense
        let mut next_level: HashSet<(Vec<usize>, Vec<usize>)> = HashSet::new();

        for (dims, coords) in &candidate_set {
            let mut count = 0usize;
            for sg in &sample_grid {
                let mut matches = true;
                for (pos, &d) in dims.iter().enumerate() {
                    if sg[d] != coords[pos] {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    count += 1;
                }
            }

            if count >= tau_count {
                all_dense_cells.push(DenseCell {
                    dimensions: dims.clone(),
                    coords: coords.clone(),
                });
                next_level.insert((dims.clone(), coords.clone()));
            }
        }

        current_level = next_level;
        dim_level += 1;
    }

    // --- Step 3: assign labels via connected-component analysis on dense cells ---
    // Use highest-dimensional dense cells for labeling
    let max_found_dim = all_dense_cells
        .iter()
        .map(|c| c.dimensions.len())
        .max()
        .unwrap_or(0);

    let top_cells: Vec<&DenseCell> = all_dense_cells
        .iter()
        .filter(|c| c.dimensions.len() == max_found_dim)
        .collect();

    // Build adjacency (two cells are adjacent if they differ by 1 in exactly one coord)
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); top_cells.len()];
    for i in 0..top_cells.len() {
        for j in (i + 1)..top_cells.len() {
            if top_cells[i].dimensions == top_cells[j].dimensions {
                let mut diff_count = 0usize;
                for k in 0..top_cells[i].coords.len() {
                    let a = top_cells[i].coords[k] as i64;
                    let b = top_cells[j].coords[k] as i64;
                    if (a - b).unsigned_abs() == 1 {
                        diff_count += 1;
                    } else if a != b {
                        diff_count = 2; // not adjacent
                        break;
                    }
                }
                if diff_count == 1 {
                    adj[i].push(j);
                    adj[j].push(i);
                }
            }
        }
    }

    // Connected components via BFS
    let mut cell_labels = vec![-1i32; top_cells.len()];
    let mut cluster_id = 0i32;
    for start in 0..top_cells.len() {
        if cell_labels[start] >= 0 {
            continue;
        }
        let mut queue = vec![start];
        cell_labels[start] = cluster_id;
        let mut head = 0usize;
        while head < queue.len() {
            let cur = queue[head];
            head += 1;
            for &nb in &adj[cur] {
                if cell_labels[nb] < 0 {
                    cell_labels[nb] = cluster_id;
                    queue.push(nb);
                }
            }
        }
        cluster_id += 1;
    }

    // Assign data points to clusters
    let mut labels = Array1::from_elem(n_samples, -1i32);
    for i in 0..n_samples {
        for (ci, cell) in top_cells.iter().enumerate() {
            let mut in_cell = true;
            for (pos, &d) in cell.dimensions.iter().enumerate() {
                if sample_grid[i][d] != cell.coords[pos] {
                    in_cell = false;
                    break;
                }
            }
            if in_cell {
                labels[i] = cell_labels[ci];
                break;
            }
        }
    }

    // Collect discovered subspaces
    let mut subspace_set: HashSet<Vec<usize>> = HashSet::new();
    for cell in &all_dense_cells {
        if cell.dimensions.len() > 1 {
            subspace_set.insert(cell.dimensions.clone());
        }
    }
    let subspaces: Vec<Vec<usize>> = subspace_set.into_iter().collect();

    Ok(CliqueResult {
        labels,
        subspaces,
        dense_cells: all_dense_cells,
        n_clusters: cluster_id as usize,
    })
}

// ---------------------------------------------------------------------------
// PROCLUS algorithm
// ---------------------------------------------------------------------------

/// Configuration for the PROCLUS algorithm.
#[derive(Debug, Clone)]
pub struct ProclusConfig {
    /// Number of clusters (k).
    pub n_clusters: usize,
    /// Average number of dimensions per cluster (l).
    pub avg_dimensions: usize,
    /// Number of medoid candidates = n_clusters * multiplier.
    pub candidate_multiplier: usize,
    /// Maximum iterations for the iterative phase.
    pub max_iterations: usize,
    /// Minimum improvement ratio for convergence.
    pub min_improvement: f64,
}

impl Default for ProclusConfig {
    fn default() -> Self {
        Self {
            n_clusters: 3,
            avg_dimensions: 2,
            candidate_multiplier: 10,
            max_iterations: 30,
            min_improvement: 1e-4,
        }
    }
}

/// Result of the PROCLUS algorithm.
#[derive(Debug, Clone)]
pub struct ProclusResult<F: Float> {
    /// Cluster labels per data point (-1 = not assigned).
    pub labels: Array1<i32>,
    /// Medoid indices into the original data.
    pub medoids: Vec<usize>,
    /// Dimensions selected for each cluster (cluster_idx -> dim list).
    pub cluster_dimensions: Vec<Vec<usize>>,
    /// Medoid coordinates.
    pub medoid_coords: Array2<F>,
    /// Number of iterations run.
    pub iterations: usize,
}

/// Run the PROCLUS projected clustering algorithm.
///
/// PROCLUS selects k medoids and, for each medoid, identifies a subset of
/// dimensions (the "projected subspace") where the cluster is most compact.
/// Points are assigned to the closest medoid in its projected subspace.
pub fn proclus<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    config: &ProclusConfig,
) -> Result<ProclusResult<F>> {
    let (n_samples, n_features) = (data.shape()[0], data.shape()[1]);

    if n_samples == 0 {
        return Err(ClusteringError::InvalidInput("Empty input data".into()));
    }
    if config.n_clusters == 0 || config.n_clusters > n_samples {
        return Err(ClusteringError::InvalidInput(
            "n_clusters must be in [1, n_samples]".into(),
        ));
    }
    if config.avg_dimensions == 0 || config.avg_dimensions > n_features {
        return Err(ClusteringError::InvalidInput(
            "avg_dimensions must be in [1, n_features]".into(),
        ));
    }

    let k = config.n_clusters;
    let l = config.avg_dimensions;

    // --- Initialization phase: greedy medoid selection ---
    let n_candidates = (k * config.candidate_multiplier).min(n_samples);
    let mut medoids = greedy_initial_medoids(data, n_candidates, k);

    // --- Iterative phase ---
    let mut cluster_dims: Vec<Vec<usize>> = vec![Vec::new(); k];
    let mut labels = Array1::from_elem(n_samples, -1i32);
    let mut prev_objective = f64::MAX;
    let mut iterations = 0usize;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Determine dimensions for each medoid
        cluster_dims = find_cluster_dimensions(data, &medoids, l, n_features);

        // Assign points to nearest medoid in projected subspace
        labels = assign_projected(data, &medoids, &cluster_dims);

        // Compute objective (sum of projected distances)
        let objective = compute_projected_objective(data, &medoids, &cluster_dims, &labels);

        let improvement = if prev_objective < f64::MAX {
            (prev_objective - objective) / prev_objective.abs().max(1e-15)
        } else {
            1.0
        };

        if improvement < config.min_improvement && iter > 0 {
            break;
        }
        prev_objective = objective;

        // Update medoids: for each cluster, pick the sample minimising projected distance
        for ci in 0..k {
            let mut best_idx = medoids[ci];
            let mut best_cost = f64::MAX;
            for i in 0..n_samples {
                if labels[i] != ci as i32 {
                    continue;
                }
                let cost = cluster_projected_cost(data, i, &cluster_dims[ci], &labels, ci as i32);
                if cost < best_cost {
                    best_cost = cost;
                    best_idx = i;
                }
            }
            medoids[ci] = best_idx;
        }
    }

    let mut medoid_coords = Array2::zeros((k, n_features));
    for (ci, &mi) in medoids.iter().enumerate() {
        medoid_coords.row_mut(ci).assign(&data.row(mi));
    }

    Ok(ProclusResult {
        labels,
        medoids,
        cluster_dimensions: cluster_dims,
        medoid_coords,
        iterations,
    })
}

/// Greedy initialization: pick n_candidates spread-out samples, then select k.
fn greedy_initial_medoids<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    n_candidates: usize,
    k: usize,
) -> Vec<usize> {
    let n = data.shape()[0];
    let nc = n_candidates.min(n);

    // Deterministic spread selection: pick every n/nc-th sample
    let step = (n as f64 / nc as f64).max(1.0);
    let candidates: Vec<usize> = (0..nc)
        .map(|i| ((i as f64 * step) as usize).min(n - 1))
        .collect();

    if candidates.len() <= k {
        return candidates;
    }

    // From candidates, greedily pick k that are spread out
    let mut selected = Vec::with_capacity(k);
    selected.push(candidates[0]);
    let mut min_dists = vec![f64::MAX; candidates.len()];

    for _ in 1..k {
        let last = *selected.last().unwrap_or(&0);
        for (ci, &c) in candidates.iter().enumerate() {
            let d = euclidean_sq_f64(data.row(c), data.row(last));
            if d < min_dists[ci] {
                min_dists[ci] = d;
            }
        }
        // Pick the candidate with max min-distance
        let mut best_ci = 0;
        let mut best_d = -1.0f64;
        for (ci, &d) in min_dists.iter().enumerate() {
            if d > best_d && !selected.contains(&candidates[ci]) {
                best_d = d;
                best_ci = ci;
            }
        }
        selected.push(candidates[best_ci]);
    }

    selected
}

/// Find the best l dimensions for each medoid based on average spread.
fn find_cluster_dimensions<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    medoids: &[usize],
    avg_l: usize,
    n_features: usize,
) -> Vec<Vec<usize>> {
    let n_samples = data.shape()[0];
    let k = medoids.len();
    let total_dims = avg_l * k;

    // Assign each point to nearest medoid (full space, for dimension selection)
    let mut assignments = vec![0usize; n_samples];
    for i in 0..n_samples {
        let mut best_m = 0;
        let mut best_d = f64::MAX;
        for (mi, &m) in medoids.iter().enumerate() {
            let d = euclidean_sq_f64(data.row(i), data.row(m));
            if d < best_d {
                best_d = d;
                best_m = mi;
            }
        }
        assignments[i] = best_m;
    }

    // For each medoid and dimension, compute average absolute deviation
    let mut spreads: Vec<Vec<f64>> = vec![vec![0.0; n_features]; k];
    let mut counts = vec![0usize; k];

    for i in 0..n_samples {
        let ci = assignments[i];
        counts[ci] += 1;
        let mi = medoids[ci];
        for d in 0..n_features {
            let diff = (data[[i, d]] - data[[mi, d]]).abs().to_f64().unwrap_or(0.0);
            spreads[ci][d] += diff;
        }
    }

    for ci in 0..k {
        if counts[ci] > 0 {
            for d in 0..n_features {
                spreads[ci][d] /= counts[ci] as f64;
            }
        }
    }

    // Collect (spread, cluster_idx, dim_idx) and sort ascending
    let mut dim_spread: Vec<(f64, usize, usize)> = Vec::new();
    for ci in 0..k {
        for d in 0..n_features {
            dim_spread.push((spreads[ci][d], ci, d));
        }
    }
    dim_spread.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Greedily assign top-total_dims lowest-spread (cluster, dim) pairs
    let mut cluster_dims: Vec<Vec<usize>> = vec![Vec::new(); k];
    let mut assigned = 0usize;
    for &(_, ci, d) in &dim_spread {
        if assigned >= total_dims {
            break;
        }
        if !cluster_dims[ci].contains(&d) {
            cluster_dims[ci].push(d);
            assigned += 1;
        }
    }

    // Ensure every cluster has at least one dimension
    for ci in 0..k {
        if cluster_dims[ci].is_empty() {
            cluster_dims[ci].push(0);
        }
        cluster_dims[ci].sort();
    }

    cluster_dims
}

/// Assign each point to the nearest medoid using projected distance.
fn assign_projected<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    medoids: &[usize],
    cluster_dims: &[Vec<usize>],
) -> Array1<i32> {
    let n = data.shape()[0];
    let k = medoids.len();
    let mut labels = Array1::from_elem(n, -1i32);

    for i in 0..n {
        let mut best_m = 0i32;
        let mut best_d = f64::MAX;
        for ci in 0..k {
            let mi = medoids[ci];
            let dims = &cluster_dims[ci];
            let mut dist = 0.0f64;
            for &d in dims {
                let diff = (data[[i, d]] - data[[mi, d]]).to_f64().unwrap_or(0.0);
                dist += diff * diff;
            }
            if dist < best_d {
                best_d = dist;
                best_m = ci as i32;
            }
        }
        labels[i] = best_m;
    }

    labels
}

/// Compute the total projected objective (sum of squared projected distances).
fn compute_projected_objective<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    medoids: &[usize],
    cluster_dims: &[Vec<usize>],
    labels: &Array1<i32>,
) -> f64 {
    let n = data.shape()[0];
    let mut total = 0.0f64;
    for i in 0..n {
        let ci = labels[i];
        if ci < 0 {
            continue;
        }
        let ci = ci as usize;
        let mi = medoids[ci];
        for &d in &cluster_dims[ci] {
            let diff = (data[[i, d]] - data[[mi, d]]).to_f64().unwrap_or(0.0);
            total += diff * diff;
        }
    }
    total
}

/// Cost of making sample `idx` a medoid for cluster `ci`.
fn cluster_projected_cost<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    idx: usize,
    dims: &[usize],
    labels: &Array1<i32>,
    ci_label: i32,
) -> f64 {
    let n = data.shape()[0];
    let mut cost = 0.0f64;
    for i in 0..n {
        if labels[i] != ci_label {
            continue;
        }
        for &d in dims {
            let diff = (data[[i, d]] - data[[idx, d]]).to_f64().unwrap_or(0.0);
            cost += diff * diff;
        }
    }
    cost
}

// ---------------------------------------------------------------------------
// Sparse Subspace Clustering (SSC)
// ---------------------------------------------------------------------------

/// Configuration for Sparse Subspace Clustering.
#[derive(Debug, Clone)]
pub struct SscConfig {
    /// Regularisation parameter for L1 penalty (lambda).
    pub lambda: f64,
    /// Maximum ADMM iterations for the L1 self-expression solver.
    pub max_iterations: usize,
    /// Convergence tolerance for ADMM.
    pub tolerance: f64,
    /// Number of clusters (for spectral clustering on the affinity).
    pub n_clusters: usize,
    /// ADMM penalty parameter (rho).
    pub rho: f64,
}

impl Default for SscConfig {
    fn default() -> Self {
        Self {
            lambda: 0.1,
            max_iterations: 200,
            tolerance: 1e-6,
            n_clusters: 3,
            rho: 1.0,
        }
    }
}

/// Result of Sparse Subspace Clustering.
#[derive(Debug, Clone)]
pub struct SscResult<F: Float> {
    /// Cluster labels for each data point.
    pub labels: Array1<i32>,
    /// Sparse self-expression coefficient matrix (n_samples x n_samples).
    pub coefficients: Array2<F>,
    /// Affinity matrix used for spectral clustering.
    pub affinity: Array2<F>,
    /// Number of clusters found.
    pub n_clusters: usize,
}

/// Run Sparse Subspace Clustering via self-expression.
///
/// Solves: min ||C||_1 subject to X = X C, diag(C) = 0
/// using ADMM, then builds an affinity matrix |C| + |C^T| and performs
/// spectral clustering to obtain final labels.
pub fn ssc<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    config: &SscConfig,
) -> Result<SscResult<F>> {
    let (n_samples, n_features) = (data.shape()[0], data.shape()[1]);

    if n_samples == 0 {
        return Err(ClusteringError::InvalidInput("Empty input data".into()));
    }
    if config.n_clusters == 0 || config.n_clusters > n_samples {
        return Err(ClusteringError::InvalidInput(
            "n_clusters must be in [1, n_samples]".into(),
        ));
    }

    // Solve the L1-penalised self-expression problem via ADMM
    // min lambda*||C||_1 + 0.5*||X - X*C||_F^2  s.t. diag(C)=0
    let coefficients = ssc_admm(data, config)?;

    // Build symmetric affinity: W = |C| + |C^T|
    let mut affinity = Array2::<F>::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in 0..n_samples {
            affinity[[i, j]] = coefficients[[i, j]].abs() + coefficients[[j, i]].abs();
        }
    }

    // Spectral clustering on the affinity matrix
    let labels = spectral_from_affinity(&affinity, config.n_clusters)?;
    let n_clusters = config.n_clusters;

    Ok(SscResult {
        labels,
        coefficients,
        affinity,
        n_clusters,
    })
}

/// ADMM solver for L1-penalised self-expression.
fn ssc_admm<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    config: &SscConfig,
) -> Result<Array2<F>> {
    let n = data.shape()[0];
    let lambda_f = F::from(config.lambda).unwrap_or_else(|| F::one());
    let rho_f = F::from(config.rho).unwrap_or_else(|| F::one());
    let one = F::one();

    // Compute X^T X
    let mut xtx = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut dot = F::zero();
            for d in 0..data.shape()[1] {
                dot = dot + data[[i, d]] * data[[j, d]];
            }
            xtx[[i, j]] = dot;
        }
    }

    // Precompute (X^T X + rho * I)^{-1} via simple iterative method
    // For moderate n, use direct Cholesky-like decomposition
    // We use a simplified approach: (X^T X + rho*I) * C_col = X^T X_col + rho * Z_col - U_col
    // Solved per-column via conjugate gradient-like iterations

    let mut c_mat = Array2::<F>::zeros((n, n));
    let mut z_mat = Array2::<F>::zeros((n, n));
    let mut u_mat = Array2::<F>::zeros((n, n));

    // Build A = X^T X + rho * I
    let mut a_mat = xtx.clone();
    for i in 0..n {
        a_mat[[i, i]] = a_mat[[i, i]] + rho_f;
    }

    let tol_f = F::from(config.tolerance).unwrap_or_else(|| F::epsilon());

    for _iter in 0..config.max_iterations {
        // C-update: solve A * c_j = xtx_j + rho*(z_j - u_j) for each column j
        for j in 0..n {
            let mut rhs = Array1::<F>::zeros(n);
            for i in 0..n {
                rhs[i] = xtx[[i, j]] + rho_f * (z_mat[[i, j]] - u_mat[[i, j]]);
            }
            // Solve A * x = rhs using Gauss-Seidel (few iterations for ADMM inner)
            let mut x = Array1::<F>::zeros(n);
            for _gs in 0..20 {
                for i in 0..n {
                    let mut sum = rhs[i];
                    for k in 0..n {
                        if k != i {
                            sum = sum - a_mat[[i, k]] * x[k];
                        }
                    }
                    if a_mat[[i, i]].abs() > F::epsilon() {
                        x[i] = sum / a_mat[[i, i]];
                    }
                }
            }
            // Enforce diag(C) = 0
            x[j] = F::zero();
            for i in 0..n {
                c_mat[[i, j]] = x[i];
            }
        }

        // Z-update: soft thresholding
        let thresh = lambda_f / rho_f;
        for i in 0..n {
            for j in 0..n {
                let v = c_mat[[i, j]] + u_mat[[i, j]];
                z_mat[[i, j]] = soft_threshold(v, thresh);
            }
        }

        // U-update: dual variable
        let mut primal_res = F::zero();
        for i in 0..n {
            for j in 0..n {
                let r = c_mat[[i, j]] - z_mat[[i, j]];
                u_mat[[i, j]] = u_mat[[i, j]] + r;
                primal_res = primal_res + r * r;
            }
        }

        if primal_res.sqrt() < tol_f {
            break;
        }
    }

    Ok(c_mat)
}

/// Soft thresholding operator: sign(x) * max(|x| - t, 0).
fn soft_threshold<F: Float>(x: F, t: F) -> F {
    let abs_x = x.abs();
    if abs_x <= t {
        F::zero()
    } else if x > F::zero() {
        abs_x - t
    } else {
        t - abs_x
    }
}

/// Spectral clustering from a precomputed affinity matrix.
fn spectral_from_affinity<F: Float + FromPrimitive + Debug>(
    affinity: &Array2<F>,
    k: usize,
) -> Result<Array1<i32>> {
    let n = affinity.shape()[0];
    if n <= k {
        return Ok(Array1::from_vec((0..n as i32).collect()));
    }

    // Compute degree matrix and normalised Laplacian
    let mut degree = Array1::<F>::zeros(n);
    for i in 0..n {
        for j in 0..n {
            degree[i] = degree[i] + affinity[[i, j]];
        }
    }

    // L_sym = I - D^{-1/2} W D^{-1/2}  (Ng-Jordan-Weiss)
    let mut l_sym = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let di = if degree[i] > F::epsilon() {
                F::one() / degree[i].sqrt()
            } else {
                F::zero()
            };
            let dj = if degree[j] > F::epsilon() {
                F::one() / degree[j].sqrt()
            } else {
                F::zero()
            };
            l_sym[[i, j]] = if i == j {
                F::one() - di * affinity[[i, j]] * dj
            } else {
                F::zero() - di * affinity[[i, j]] * dj
            };
        }
    }

    // Extract k smallest eigenvectors via power iteration on (I - L_sym)
    // which has largest eigenvalues corresponding to smallest eigenvalues of L_sym
    let shifted = {
        let mut m = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                m[[i, j]] = if i == j {
                    F::one() - l_sym[[i, j]]
                } else {
                    F::zero() - l_sym[[i, j]]
                };
            }
        }
        m
    };

    let mut eigvecs = Array2::<F>::zeros((n, k));
    let mut deflated = shifted.clone();

    for kk in 0..k {
        let v = power_iteration(&deflated, 100)?;
        for i in 0..n {
            eigvecs[[i, kk]] = v[i];
        }
        // Deflate: remove component along v
        let eigenval = rayleigh_quotient(&deflated, &v);
        for i in 0..n {
            for j in 0..n {
                deflated[[i, j]] = deflated[[i, j]] - eigenval * v[i] * v[j];
            }
        }
    }

    // Row-normalise eigenvectors
    for i in 0..n {
        let mut norm = F::zero();
        for kk in 0..k {
            norm = norm + eigvecs[[i, kk]] * eigvecs[[i, kk]];
        }
        let norm = norm.sqrt();
        if norm > F::epsilon() {
            for kk in 0..k {
                eigvecs[[i, kk]] = eigvecs[[i, kk]] / norm;
            }
        }
    }

    // K-means on the rows of eigvecs
    let labels = simple_kmeans(eigvecs.view(), k, 50);
    Ok(labels)
}

/// Power iteration to find the dominant eigenvector.
fn power_iteration<F: Float + FromPrimitive + Debug>(
    mat: &Array2<F>,
    max_iter: usize,
) -> Result<Array1<F>> {
    let n = mat.shape()[0];
    let mut v = Array1::<F>::zeros(n);
    // Initialize with alternating signs for better convergence
    for i in 0..n {
        v[i] = if i % 2 == 0 {
            F::one()
        } else {
            F::zero() - F::one()
        };
    }
    // Normalize
    let mut norm = F::zero();
    for i in 0..n {
        norm = norm + v[i] * v[i];
    }
    let norm = norm.sqrt();
    if norm > F::epsilon() {
        for i in 0..n {
            v[i] = v[i] / norm;
        }
    }

    for _ in 0..max_iter {
        // w = M * v
        let mut w = Array1::<F>::zeros(n);
        for i in 0..n {
            for j in 0..n {
                w[i] = w[i] + mat[[i, j]] * v[j];
            }
        }
        // Normalize
        let mut norm = F::zero();
        for i in 0..n {
            norm = norm + w[i] * w[i];
        }
        let norm = norm.sqrt();
        if norm < F::epsilon() {
            break;
        }
        for i in 0..n {
            v[i] = w[i] / norm;
        }
    }
    Ok(v)
}

/// Rayleigh quotient v^T M v / v^T v.
fn rayleigh_quotient<F: Float>(mat: &Array2<F>, v: &Array1<F>) -> F {
    let n = v.len();
    let mut num = F::zero();
    let mut den = F::zero();
    for i in 0..n {
        den = den + v[i] * v[i];
        for j in 0..n {
            num = num + v[i] * mat[[i, j]] * v[j];
        }
    }
    if den > F::epsilon() {
        num / den
    } else {
        F::zero()
    }
}

/// Simple k-means for spectral clustering post-processing.
fn simple_kmeans<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    k: usize,
    max_iter: usize,
) -> Array1<i32> {
    let (n, d) = (data.shape()[0], data.shape()[1]);
    if n == 0 || k == 0 {
        return Array1::from_elem(n, -1i32);
    }

    // Initialize centroids from first k distinct rows
    let mut centroids = Array2::<F>::zeros((k, d));
    let step = (n as f64 / k as f64).max(1.0);
    for ci in 0..k {
        let idx = ((ci as f64 * step) as usize).min(n - 1);
        centroids.row_mut(ci).assign(&data.row(idx));
    }

    let mut labels = Array1::from_elem(n, 0i32);

    for _ in 0..max_iter {
        // Assign
        let mut changed = false;
        for i in 0..n {
            let mut best = 0i32;
            let mut best_d = f64::MAX;
            for ci in 0..k {
                let dd = euclidean_sq_f64(data.row(i), centroids.row(ci));
                if dd < best_d {
                    best_d = dd;
                    best = ci as i32;
                }
            }
            if labels[i] != best {
                labels[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update centroids
        let mut counts = vec![0usize; k];
        let mut sums = Array2::<F>::zeros((k, d));
        for i in 0..n {
            let ci = labels[i] as usize;
            counts[ci] += 1;
            for dd in 0..d {
                sums[[ci, dd]] = sums[[ci, dd]] + data[[i, dd]];
            }
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                let cnt = F::from(counts[ci]).unwrap_or_else(|| F::one());
                for dd in 0..d {
                    centroids[[ci, dd]] = sums[[ci, dd]] / cnt;
                }
            }
        }
    }

    labels
}

/// Squared Euclidean distance as f64.
fn euclidean_sq_f64<F: Float>(a: ArrayView1<F>, b: ArrayView1<F>) -> f64 {
    let mut s = 0.0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).to_f64().unwrap_or(0.0);
        s += d * d;
    }
    s
}

// ---------------------------------------------------------------------------
// Feature subspace selection
// ---------------------------------------------------------------------------

/// Method for feature subspace selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubspaceSelectionMethod {
    /// Variance-based: select top-k highest variance features.
    Variance,
    /// Entropy-based: select top-k highest entropy features (discretised).
    Entropy,
    /// Correlation-based: greedily select features that are mutually uncorrelated.
    CorrelationFiltering,
}

/// Select the best feature subspace.
///
/// Returns sorted indices of the selected features.
pub fn select_subspace<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    k: usize,
    method: SubspaceSelectionMethod,
) -> Result<Vec<usize>> {
    let n_features = data.shape()[1];
    if k == 0 || k > n_features {
        return Err(ClusteringError::InvalidInput(
            "k must be in [1, n_features]".into(),
        ));
    }

    match method {
        SubspaceSelectionMethod::Variance => select_by_variance(data, k),
        SubspaceSelectionMethod::Entropy => select_by_entropy(data, k),
        SubspaceSelectionMethod::CorrelationFiltering => select_by_correlation(data, k),
    }
}

fn select_by_variance<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    k: usize,
) -> Result<Vec<usize>> {
    let (n, d) = (data.shape()[0], data.shape()[1]);
    if n < 2 {
        let mut r: Vec<usize> = (0..k.min(d)).collect();
        return Ok(r);
    }

    let mut variances: Vec<(usize, f64)> = Vec::with_capacity(d);
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
        variances.push((dim, var));
    }
    variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut result: Vec<usize> = variances.iter().take(k).map(|&(d, _)| d).collect();
    result.sort();
    Ok(result)
}

fn select_by_entropy<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    k: usize,
) -> Result<Vec<usize>> {
    let (n, d) = (data.shape()[0], data.shape()[1]);
    let n_bins = 10usize;

    let mut entropies: Vec<(usize, f64)> = Vec::with_capacity(d);
    for dim in 0..d {
        let mut min_v = f64::MAX;
        let mut max_v = f64::MIN;
        for i in 0..n {
            let v = data[[i, dim]].to_f64().unwrap_or(0.0);
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }
        let range = (max_v - min_v).max(1e-15);
        let mut counts = vec![0usize; n_bins];
        for i in 0..n {
            let v = data[[i, dim]].to_f64().unwrap_or(0.0);
            let bin = (((v - min_v) / range) * (n_bins as f64 - 1e-10)) as usize;
            counts[bin.min(n_bins - 1)] += 1;
        }
        let entropy: f64 = counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n as f64;
                -p * p.ln()
            })
            .sum();
        entropies.push((dim, entropy));
    }
    entropies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut result: Vec<usize> = entropies.iter().take(k).map(|&(d, _)| d).collect();
    result.sort();
    Ok(result)
}

fn select_by_correlation<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    k: usize,
) -> Result<Vec<usize>> {
    let (n, d) = (data.shape()[0], data.shape()[1]);

    // Compute means and stddevs
    let mut means = vec![0.0f64; d];
    for i in 0..n {
        for dim in 0..d {
            means[dim] += data[[i, dim]].to_f64().unwrap_or(0.0);
        }
    }
    for dim in 0..d {
        means[dim] /= n as f64;
    }
    let mut stds = vec![0.0f64; d];
    for i in 0..n {
        for dim in 0..d {
            let diff = data[[i, dim]].to_f64().unwrap_or(0.0) - means[dim];
            stds[dim] += diff * diff;
        }
    }
    for dim in 0..d {
        stds[dim] = (stds[dim] / n.max(1) as f64).sqrt().max(1e-15);
    }

    // Variance ranking for initial feature
    let mut variances: Vec<(usize, f64)> =
        stds.iter().enumerate().map(|(d, &s)| (d, s * s)).collect();
    variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected = vec![variances[0].0];

    while selected.len() < k {
        let mut best_dim = None;
        let mut best_score = f64::MAX;
        for &(dim, _) in &variances {
            if selected.contains(&dim) {
                continue;
            }
            // Max absolute correlation with any selected feature
            let mut max_corr = 0.0f64;
            for &s in &selected {
                let mut cov = 0.0f64;
                for i in 0..n {
                    let a = data[[i, dim]].to_f64().unwrap_or(0.0) - means[dim];
                    let b = data[[i, s]].to_f64().unwrap_or(0.0) - means[s];
                    cov += a * b;
                }
                cov /= n.max(1) as f64;
                let corr = (cov / (stds[dim] * stds[s])).abs();
                if corr > max_corr {
                    max_corr = corr;
                }
            }
            if max_corr < best_score {
                best_score = max_corr;
                best_dim = Some(dim);
            }
        }
        match best_dim {
            Some(d) => selected.push(d),
            None => break,
        }
    }

    selected.sort();
    Ok(selected)
}

// ---------------------------------------------------------------------------
// Subspace quality metrics
// ---------------------------------------------------------------------------

/// Subspace clustering quality metrics.
#[derive(Debug, Clone)]
pub struct SubspaceQuality {
    /// Coverage: fraction of points assigned to a cluster.
    pub coverage: f64,
    /// Average cluster density (points per cluster per dimension).
    pub avg_density: f64,
    /// Subspace dimensionality ratio (avg dims used / total dims).
    pub dimensionality_ratio: f64,
    /// Cluster separation in subspace (average inter-cluster distance).
    pub separation: f64,
}

/// Evaluate the quality of a subspace clustering result.
pub fn subspace_quality<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    labels: &Array1<i32>,
    cluster_dims: &[Vec<usize>],
) -> Result<SubspaceQuality> {
    let (n, d) = (data.shape()[0], data.shape()[1]);

    if labels.len() != n {
        return Err(ClusteringError::InvalidInput(
            "labels length must match n_samples".into(),
        ));
    }

    let n_assigned = labels.iter().filter(|&&l| l >= 0).count();
    let coverage = n_assigned as f64 / n.max(1) as f64;

    let k = cluster_dims.len();
    let avg_dims: f64 = if k > 0 {
        cluster_dims.iter().map(|d| d.len() as f64).sum::<f64>() / k as f64
    } else {
        0.0
    };
    let dimensionality_ratio = avg_dims / d.max(1) as f64;

    // Compute per-cluster count and centroids in projected subspace
    let mut counts = vec![0usize; k];
    for &l in labels.iter() {
        if l >= 0 && (l as usize) < k {
            counts[l as usize] += 1;
        }
    }

    let avg_density = if k > 0 && avg_dims > 0.0 {
        let total_assigned: usize = counts.iter().sum();
        total_assigned as f64 / (k as f64 * avg_dims)
    } else {
        0.0
    };

    // Compute centroids for separation
    let mut centroids: Vec<Vec<f64>> = vec![vec![0.0; d]; k];
    for i in 0..n {
        let l = labels[i];
        if l >= 0 && (l as usize) < k {
            let ci = l as usize;
            for dd in 0..d {
                centroids[ci][dd] += data[[i, dd]].to_f64().unwrap_or(0.0);
            }
        }
    }
    for ci in 0..k {
        if counts[ci] > 0 {
            for dd in 0..d {
                centroids[ci][dd] /= counts[ci] as f64;
            }
        }
    }

    let mut separation = 0.0f64;
    let mut sep_count = 0usize;
    for ci in 0..k {
        for cj in (ci + 1)..k {
            let mut dist = 0.0f64;
            // Use union of both cluster dimensions
            let mut union_dims: HashSet<usize> = HashSet::new();
            for &d in &cluster_dims[ci] {
                union_dims.insert(d);
            }
            for &d in &cluster_dims[cj] {
                union_dims.insert(d);
            }
            for &dd in &union_dims {
                let diff = centroids[ci][dd] - centroids[cj][dd];
                dist += diff * diff;
            }
            separation += dist.sqrt();
            sep_count += 1;
        }
    }
    if sep_count > 0 {
        separation /= sep_count as f64;
    }

    Ok(SubspaceQuality {
        coverage,
        avg_density,
        dimensionality_ratio,
        separation,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_two_cluster_data() -> Array2<f64> {
        // Cluster A at (1,1,0,0), cluster B at (5,5,0,0) — clusters in dims 0,1
        let mut data = Vec::new();
        for _ in 0..20 {
            data.extend_from_slice(&[1.0, 1.0, 0.5, 0.5]);
        }
        for _ in 0..20 {
            data.extend_from_slice(&[5.0, 5.0, 0.5, 0.5]);
        }
        // Add some small noise deterministically
        let mut arr = Array2::from_shape_vec((40, 4), data).expect("shape construction failed");
        for i in 0..40 {
            let noise = (i as f64 * 0.037).sin() * 0.2;
            arr[[i, 0]] = arr[[i, 0]] + noise;
            arr[[i, 1]] = arr[[i, 1]] + noise * 0.5;
        }
        arr
    }

    #[test]
    fn test_clique_basic() {
        let data = make_two_cluster_data();
        let config = CliqueConfig {
            n_intervals: 5,
            density_threshold: 0.1,
            max_subspace_dim: 2,
        };
        let result = clique(data.view(), &config).expect("clique failed");
        // Should find at least 1 cluster
        assert!(
            result.n_clusters >= 1,
            "Expected at least 1 cluster, got {}",
            result.n_clusters
        );
        // Most points should be assigned
        let assigned = result.labels.iter().filter(|&&l| l >= 0).count();
        assert!(
            assigned > 20,
            "Expected most points assigned, got {}",
            assigned
        );
    }

    #[test]
    fn test_clique_empty_data() {
        let data = Array2::<f64>::zeros((0, 4));
        let config = CliqueConfig::default();
        assert!(clique(data.view(), &config).is_err());
    }

    #[test]
    fn test_clique_invalid_params() {
        let data = make_two_cluster_data();
        let config = CliqueConfig {
            n_intervals: 0,
            ..Default::default()
        };
        assert!(clique(data.view(), &config).is_err());

        let config2 = CliqueConfig {
            density_threshold: 0.0,
            ..Default::default()
        };
        assert!(clique(data.view(), &config2).is_err());
    }

    #[test]
    fn test_proclus_basic() {
        let data = make_two_cluster_data();
        let config = ProclusConfig {
            n_clusters: 2,
            avg_dimensions: 2,
            max_iterations: 20,
            ..Default::default()
        };
        let result = proclus(data.view(), &config).expect("proclus failed");
        assert_eq!(result.labels.len(), 40);
        assert_eq!(result.medoids.len(), 2);
        assert_eq!(result.cluster_dimensions.len(), 2);
        // Each cluster should have dimensions
        for dims in &result.cluster_dimensions {
            assert!(!dims.is_empty());
        }
    }

    #[test]
    fn test_proclus_empty_data() {
        let data = Array2::<f64>::zeros((0, 3));
        let config = ProclusConfig::default();
        assert!(proclus(data.view(), &config).is_err());
    }

    #[test]
    fn test_proclus_invalid_k() {
        let data = make_two_cluster_data();
        let config = ProclusConfig {
            n_clusters: 0,
            ..Default::default()
        };
        assert!(proclus(data.view(), &config).is_err());
    }

    #[test]
    fn test_ssc_basic() {
        // Small dataset for SSC (it's O(n^2) in memory)
        let data = Array2::from_shape_vec(
            (12, 3),
            vec![
                1.0, 0.0, 0.0, 1.1, 0.1, 0.0, 0.9, -0.1, 0.0, 1.2, 0.05, 0.0, 0.0, 1.0, 0.0, 0.1,
                1.1, 0.0, -0.1, 0.9, 0.0, 0.05, 1.2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 1.1, 0.0, -0.1,
                0.9, 0.0, 0.05, 1.2,
            ],
        )
        .expect("shape failed");

        let config = SscConfig {
            n_clusters: 3,
            lambda: 0.5,
            max_iterations: 50,
            ..Default::default()
        };
        let result = ssc(data.view(), &config).expect("ssc failed");
        assert_eq!(result.labels.len(), 12);
        assert_eq!(result.n_clusters, 3);
        // Affinity should be symmetric
        let n = result.affinity.shape()[0];
        for i in 0..n {
            for j in 0..n {
                let diff = (result.affinity[[i, j]] - result.affinity[[j, i]]).abs();
                assert!(diff < 1e-10);
            }
        }
    }

    #[test]
    fn test_ssc_empty_data() {
        let data = Array2::<f64>::zeros((0, 3));
        let config = SscConfig::default();
        assert!(ssc(data.view(), &config).is_err());
    }

    #[test]
    fn test_select_subspace_variance() {
        let data = make_two_cluster_data();
        let selected = select_subspace(data.view(), 2, SubspaceSelectionMethod::Variance)
            .expect("variance selection failed");
        assert_eq!(selected.len(), 2);
        // Dims 0 and 1 have the most variance (clusters spread there)
        assert!(selected.contains(&0));
        assert!(selected.contains(&1));
    }

    #[test]
    fn test_select_subspace_entropy() {
        let data = make_two_cluster_data();
        let selected = select_subspace(data.view(), 2, SubspaceSelectionMethod::Entropy)
            .expect("entropy selection failed");
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_select_subspace_correlation() {
        let data = make_two_cluster_data();
        let selected = select_subspace(
            data.view(),
            2,
            SubspaceSelectionMethod::CorrelationFiltering,
        )
        .expect("correlation selection failed");
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_select_subspace_invalid_k() {
        let data = make_two_cluster_data();
        assert!(select_subspace(data.view(), 0, SubspaceSelectionMethod::Variance).is_err());
        assert!(select_subspace(data.view(), 100, SubspaceSelectionMethod::Variance).is_err());
    }

    #[test]
    fn test_subspace_quality() {
        let data = make_two_cluster_data();
        let labels = Array1::from_vec((0..20).map(|_| 0i32).chain((0..20).map(|_| 1i32)).collect());
        let cluster_dims = vec![vec![0, 1], vec![0, 1]];
        let quality =
            subspace_quality(data.view(), &labels, &cluster_dims).expect("quality failed");
        assert!((quality.coverage - 1.0).abs() < 1e-10);
        assert!(quality.dimensionality_ratio > 0.0);
        assert!(quality.separation > 0.0);
    }

    #[test]
    fn test_subspace_quality_with_noise() {
        let data = make_two_cluster_data();
        let mut labels_vec: Vec<i32> = (0..20).map(|_| 0).chain((0..20).map(|_| 1)).collect();
        labels_vec[0] = -1; // one noise point
        let labels = Array1::from_vec(labels_vec);
        let cluster_dims = vec![vec![0, 1], vec![0, 1]];
        let quality =
            subspace_quality(data.view(), &labels, &cluster_dims).expect("quality failed");
        assert!(quality.coverage < 1.0);
    }

    #[test]
    fn test_soft_threshold() {
        assert!((soft_threshold(1.5, 1.0) - 0.5).abs() < 1e-10);
        assert!((soft_threshold(-1.5, 1.0) - (-0.5)).abs() < 1e-10);
        assert!((soft_threshold(0.5, 1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_clique_single_point() {
        let data = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).expect("shape failed");
        let config = CliqueConfig {
            n_intervals: 5,
            density_threshold: 0.5,
            ..Default::default()
        };
        let result = clique(data.view(), &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_proclus_single_cluster() {
        let data = Array2::from_shape_vec((10, 3), (0..30).map(|i| (i as f64) * 0.1).collect())
            .expect("shape failed");
        let config = ProclusConfig {
            n_clusters: 1,
            avg_dimensions: 2,
            ..Default::default()
        };
        let result = proclus(data.view(), &config).expect("proclus failed");
        // All should be cluster 0
        for &l in result.labels.iter() {
            assert_eq!(l, 0);
        }
    }

    #[test]
    fn test_euclidean_sq_f64() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);
        let d = euclidean_sq_f64(a.view(), b.view());
        assert!((d - 27.0).abs() < 1e-10);
    }

    #[test]
    fn test_clique_f32() {
        let data = Array2::<f32>::from_shape_vec(
            (20, 2),
            (0..40)
                .map(|i| if i < 20 { 1.0f32 } else { 5.0f32 })
                .collect(),
        )
        .expect("shape failed");
        let config = CliqueConfig {
            n_intervals: 3,
            density_threshold: 0.2,
            ..Default::default()
        };
        let result = clique(data.view(), &config);
        assert!(result.is_ok());
    }
}
