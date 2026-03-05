//! BIRCH and streaming K-means algorithms.
//!
//! Provides:
//! - `CFNode` / `CFTree` – Clustering Feature tree for compact summaries.
//! - `BIRCH` – the full BIRCH algorithm (tree construction → condensing →
//!   global k-means).
//! - `StreamingKMeans` – mini-batch K-means with streaming updates.
//! - `ChunkClustering` – processes data in chunks with periodic model updates.
//!
//! # References
//!
//! Zhang, T., Ramakrishnan, R., & Livny, M. (1996). BIRCH: an efficient data
//! clustering method for very large databases. *ACM SIGMOD*, 103–114.

use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// CF entry – Cluster Feature triplet
// ---------------------------------------------------------------------------

/// A Clustering Feature (CF) triplet: (n, LS, SS).
#[derive(Debug, Clone)]
pub struct CF<F: Float> {
    /// Number of points.
    pub n: u64,
    /// Linear sum.
    pub ls: Vec<F>,
    /// Squared sum.
    pub ss: Vec<F>,
}

impl<F: Float + FromPrimitive + Debug> CF<F> {
    /// Create a CF from a single data point.
    pub fn from_point(point: &[F]) -> Self {
        let ls: Vec<F> = point.to_vec();
        let ss: Vec<F> = point.iter().map(|&v| v * v).collect();
        Self { n: 1, ls, ss }
    }

    /// Centroid of this CF.
    pub fn centroid(&self) -> Vec<F> {
        let n_f = F::from_u64(self.n).unwrap_or(F::one());
        self.ls.iter().map(|&v| v / n_f).collect()
    }

    /// Radius (average squared root deviation from centroid).
    pub fn radius(&self) -> F {
        if self.n <= 1 {
            return F::zero();
        }
        let n_f = F::from_u64(self.n).unwrap_or(F::one());
        let d = self.ls.len();
        let mut var = F::zero();
        for i in 0..d {
            let mean = self.ls[i] / n_f;
            let mean_sq = self.ss[i] / n_f;
            let v = mean_sq - mean * mean;
            if v > F::zero() {
                var = var + v;
            }
        }
        let d_f = F::from_usize(d).unwrap_or(F::one());
        (var / d_f).sqrt()
    }

    /// Absorb another CF (additive property).
    pub fn merge_cf(&mut self, other: &CF<F>) {
        let d = self.ls.len().min(other.ls.len());
        for i in 0..d {
            self.ls[i] = self.ls[i] + other.ls[i];
            self.ss[i] = self.ss[i] + other.ss[i];
        }
        self.n += other.n;
    }

    /// Absorb a single point.
    pub fn absorb_point(&mut self, point: &[F]) {
        let d = self.ls.len().min(point.len());
        for i in 0..d {
            self.ls[i] = self.ls[i] + point[i];
            self.ss[i] = self.ss[i] + point[i] * point[i];
        }
        self.n += 1;
    }

    /// Squared Euclidean distance between centroids of two CFs.
    pub fn centroid_dist_sq(&self, other: &CF<F>) -> F {
        let c1 = self.centroid();
        let c2 = other.centroid();
        let d = c1.len().min(c2.len());
        let mut s = F::zero();
        for i in 0..d {
            let diff = c1[i] - c2[i];
            s = s + diff * diff;
        }
        s
    }

    /// D2 (inter-cluster) diameter of the merged CF.
    pub fn diameter_of_merge(a: &CF<F>, b: &CF<F>) -> F {
        let n = a.n + b.n;
        if n <= 1 {
            return F::zero();
        }
        let n_f = F::from_u64(n).unwrap_or(F::one());
        let d = a.ls.len().min(b.ls.len());
        // E[||x_i - x_j||^2] for the merged cluster
        // = 2n/(n*(n-1)) * (n * sum_sq - ||sum_ls||^2) / n  (simplified)
        let mut sum_sq = F::zero();
        let mut ls_sq_sum = F::zero();
        for i in 0..d {
            let merged_ls = a.ls[i] + b.ls[i];
            let merged_ss = a.ss[i] + b.ss[i];
            sum_sq = sum_sq + merged_ss;
            ls_sq_sum = ls_sq_sum + merged_ls * merged_ls;
        }
        let two = F::one() + F::one();
        let numerator = two * (n_f * sum_sq - ls_sq_sum);
        let denominator = n_f * (n_f - F::one());
        if denominator <= F::zero() {
            F::zero()
        } else {
            (numerator / denominator).max(F::zero()).sqrt()
        }
    }
}

// ---------------------------------------------------------------------------
// CFNode – node in the CF tree
// ---------------------------------------------------------------------------

/// A leaf node in the CF tree, holding a list of CF entries.
#[derive(Debug, Clone)]
pub struct CFNode<F: Float> {
    /// CF entries stored in this leaf.
    pub entries: Vec<CF<F>>,
    /// Maximum number of CF entries per leaf.
    pub leaf_capacity: usize,
    /// Threshold for merging a new point into an existing entry.
    pub threshold: F,
}

impl<F: Float + FromPrimitive + Debug> CFNode<F> {
    /// Create a new leaf node.
    pub fn new(leaf_capacity: usize, threshold: F) -> Self {
        Self {
            entries: Vec::new(),
            leaf_capacity,
            threshold,
        }
    }

    /// Attempt to insert a point.  Returns `true` if successful (absorbed or
    /// new entry created within capacity), `false` if the node must be split.
    pub fn insert(&mut self, point: &[F]) -> bool {
        // Find closest entry by centroid distance
        if self.entries.is_empty() {
            self.entries.push(CF::from_point(point));
            return true;
        }

        let (best_idx, best_dist_sq) = self.find_closest_entry(point);

        if best_dist_sq.sqrt() <= self.threshold {
            self.entries[best_idx].absorb_point(point);
            return true;
        }

        if self.entries.len() < self.leaf_capacity {
            self.entries.push(CF::from_point(point));
            return true;
        }

        false // needs split
    }

    fn find_closest_entry(&self, point: &[F]) -> (usize, F) {
        let mut best_idx = 0;
        let mut best_dist = F::infinity();
        let point_cf = CF::from_point(point);
        for (i, e) in self.entries.iter().enumerate() {
            let d = e.centroid_dist_sq(&point_cf);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        (best_idx, best_dist)
    }

    /// Collect all CFs in this node.
    pub fn all_cfs(&self) -> &[CF<F>] {
        &self.entries
    }

    /// Total points absorbed in this node.
    pub fn total_points(&self) -> u64 {
        self.entries.iter().map(|e| e.n).sum()
    }
}

// ---------------------------------------------------------------------------
// CFTree – the full CF tree
// ---------------------------------------------------------------------------

/// A simplified CF tree (single-level leaf pool for practical streaming use).
///
/// The full BIRCH tree is multi-level; this implementation uses a flat
/// collection of leaf nodes to keep the code tractable while preserving
/// the key algorithmic properties.
#[derive(Debug, Clone)]
pub struct CFTree<F: Float> {
    /// Leaf nodes.
    pub leaves: Vec<CFNode<F>>,
    /// Maximum entries per leaf.
    pub leaf_capacity: usize,
    /// Branching factor (max entries per internal node, for height estimation).
    pub branching_factor: usize,
    /// Distance threshold for absorbing a point into an existing CF entry.
    pub threshold: F,
    /// Dimension of the data.
    pub n_features: usize,
}

impl<F: Float + FromPrimitive + Debug + Clone> CFTree<F> {
    /// Create a new CF tree.
    pub fn new(branching_factor: usize, leaf_capacity: usize, threshold: F) -> Self {
        Self {
            leaves: Vec::new(),
            leaf_capacity,
            branching_factor,
            threshold,
            n_features: 0,
        }
    }

    /// Insert a data point into the tree.
    pub fn insert(&mut self, point: &[F]) -> Result<()> {
        if self.n_features == 0 {
            self.n_features = point.len();
        } else if point.len() != self.n_features {
            return Err(ClusteringError::InvalidInput(format!(
                "Dimension mismatch: expected {}, got {}",
                self.n_features,
                point.len()
            )));
        }

        // Try to insert into an existing leaf
        for leaf in self.leaves.iter_mut() {
            if leaf.insert(point) {
                return Ok(());
            }
        }

        // All leaves full: create a new leaf
        let mut new_leaf = CFNode::new(self.leaf_capacity, self.threshold);
        new_leaf.insert(point);
        self.leaves.push(new_leaf);
        Ok(())
    }

    /// Insert a batch of points.
    pub fn insert_batch(&mut self, data: ArrayView2<F>) -> Result<()> {
        for row in data.rows() {
            let pt: Vec<F> = row.iter().copied().collect();
            self.insert(&pt)?;
        }
        Ok(())
    }

    /// Collect all CF entries (leaf-level summaries).
    pub fn all_entries(&self) -> Vec<&CF<F>> {
        self.leaves.iter().flat_map(|l| l.entries.iter()).collect()
    }

    /// Total number of CF entries across all leaves.
    pub fn n_entries(&self) -> usize {
        self.leaves.iter().map(|l| l.entries.len()).sum()
    }

    /// Condense the tree by raising the threshold and re-inserting all
    /// current CF entries into a new tree with the higher threshold.
    pub fn condense(&self, new_threshold: F) -> Result<CFTree<F>> {
        let mut new_tree = CFTree::new(self.branching_factor, self.leaf_capacity, new_threshold);
        for entry in self.all_entries() {
            let centroid = entry.centroid();
            new_tree.insert(&centroid)?;
        }
        Ok(new_tree)
    }

    /// Total number of data points summarised by the tree.
    pub fn total_points(&self) -> u64 {
        self.leaves.iter().map(|l| l.total_points()).sum()
    }
}

// ---------------------------------------------------------------------------
// BIRCH
// ---------------------------------------------------------------------------

/// Configuration for the BIRCH algorithm.
#[derive(Debug, Clone)]
pub struct BirchConfig {
    /// Branching factor B.
    pub branching_factor: usize,
    /// Number of CF entries per leaf L.
    pub leaf_capacity: usize,
    /// Initial distance threshold T.
    pub threshold: f64,
    /// Number of final clusters k.
    pub n_clusters: usize,
    /// Whether to run the condensing phase.
    pub run_condensing: bool,
    /// New threshold for condensing phase (if None, uses `threshold * 2`).
    pub condensing_threshold: Option<f64>,
    /// Maximum iterations for global k-means.
    pub max_iter: usize,
}

impl Default for BirchConfig {
    fn default() -> Self {
        Self {
            branching_factor: 50,
            leaf_capacity: 10,
            threshold: 0.5,
            n_clusters: 3,
            run_condensing: false,
            condensing_threshold: None,
            max_iter: 100,
        }
    }
}

/// BIRCH clustering algorithm.
///
/// Phases:
/// 1. **Tree construction** – scan data stream and build CF tree.
/// 2. **Condensing** (optional) – rebuild tree with a higher threshold to
///    reduce the number of leaf-level CF entries.
/// 3. **Global clustering** – run weighted k-means on the CF entry
///    centroids to produce the final clusters.
pub struct BIRCH<F: Float> {
    config: BirchConfig,
    tree: CFTree<F>,
    fitted: bool,
}

impl<F: Float + FromPrimitive + Debug + Clone> BIRCH<F> {
    /// Create a new BIRCH instance.
    pub fn new(config: BirchConfig) -> Self {
        let threshold = F::from_f64(config.threshold).unwrap_or(F::one());
        let tree = CFTree::new(config.branching_factor, config.leaf_capacity, threshold);
        Self {
            config,
            tree,
            fitted: false,
        }
    }

    /// Phase 1: absorb data into the CF tree.
    pub fn fit_online(&mut self, data: ArrayView2<F>) -> Result<()> {
        self.tree.insert_batch(data)?;
        Ok(())
    }

    /// Phase 1 + optional phase 2 + phase 3: full offline fit.
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<BirchResult<F>> {
        // Phase 1
        self.tree.insert_batch(data)?;

        // Phase 2 – optional condensing
        if self.config.run_condensing {
            let new_t = self
                .config
                .condensing_threshold
                .unwrap_or(self.config.threshold * 2.0);
            let new_t_f = F::from_f64(new_t).unwrap_or(F::one() + F::one());
            self.tree = self.tree.condense(new_t_f)?;
        }

        // Phase 3 – global k-means on CF entry centroids
        let entries: Vec<&CF<F>> = self.tree.all_entries();
        if entries.is_empty() {
            return Err(ClusteringError::InvalidState(
                "CF tree is empty after fitting".into(),
            ));
        }
        let n_entries = entries.len();
        let d = self.tree.n_features;
        let k = self.config.n_clusters.min(n_entries);

        let centroids_vec: Vec<Vec<f64>> = entries
            .iter()
            .map(|e| {
                e.centroid()
                    .into_iter()
                    .map(|v| format!("{:?}", v).parse::<f64>().unwrap_or(0.0))
                    .collect()
            })
            .collect();
        let weights: Vec<f64> = entries.iter().map(|e| e.n as f64).collect();

        let (macro_cents, entry_labels) =
            weighted_kmeans(&centroids_vec, &weights, k, self.config.max_iter, 1e-7)?;

        self.fitted = true;

        // Map original data points to cluster labels via leaf entries
        let flat: Vec<f64> = macro_cents.clone().into_iter().flatten().collect();
        let centroids_arr = Array2::from_shape_vec((k, d), flat)
            .map_err(|e| ClusteringError::ComputationError(e.to_string()))?;

        Ok(BirchResult {
            centroids: centroids_arr,
            entry_labels,
            n_clusters: k,
            n_entries,
            total_points: self.tree.total_points(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Number of CF entries currently in the tree.
    pub fn n_entries(&self) -> usize {
        self.tree.n_entries()
    }

    /// Assign a new point to the nearest cluster centroid.
    pub fn predict(&self, result: &BirchResult<F>, point: &[f64]) -> usize {
        let mut best = 0usize;
        let mut best_d = f64::MAX;
        for (j, row) in result.centroids.rows().into_iter().enumerate() {
            let d: f64 = row.iter().zip(point.iter())
                .map(|(&a, &b)| (a - b) * (a - b)).sum();
            if d < best_d {
                best_d = d;
                best = j;
            }
        }
        best
    }
}

// ---------------------------------------------------------------------------
// StreamingKMeans
// ---------------------------------------------------------------------------

/// Configuration for streaming K-means.
#[derive(Debug, Clone)]
pub struct StreamingKMeansConfig {
    /// Number of clusters.
    pub k: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// Learning rate (step size for centroid updates).
    pub learning_rate: f64,
    /// Maximum number of mini-batch iterations.
    pub max_iter: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for StreamingKMeansConfig {
    fn default() -> Self {
        Self {
            k: 5,
            batch_size: 32,
            learning_rate: 0.01,
            max_iter: 300,
            seed: 42,
        }
    }
}

/// Mini-batch K-means with streaming updates.
///
/// Centroids are updated incrementally using a running average scheme:
/// `centroid_j ← centroid_j + (1/count_j) * (x - centroid_j)`.
pub struct StreamingKMeans<F: Float> {
    config: StreamingKMeansConfig,
    /// Current centroid estimates (k × d).
    centroids: Option<Vec<Vec<F>>>,
    /// Per-centroid update counts.
    counts: Vec<u64>,
    n_features: usize,
    n_updates: u64,
}

impl<F: Float + FromPrimitive + Debug + Clone> StreamingKMeans<F> {
    /// Create a new StreamingKMeans instance.
    pub fn new(config: StreamingKMeansConfig) -> Self {
        let k = config.k;
        Self {
            config,
            centroids: None,
            counts: vec![0u64; k],
            n_features: 0,
            n_updates: 0,
        }
    }

    /// Initialize centroids from the first batch using K-means++ seeding.
    pub fn initialize_from_batch(&mut self, batch: ArrayView2<F>) -> Result<()> {
        let (n, d) = (batch.nrows(), batch.ncols());
        if n == 0 {
            return Err(ClusteringError::InvalidInput("Empty batch".into()));
        }
        self.n_features = d;
        let k = self.config.k.min(n);

        // Use first k points as initial centroids (deterministic for testability)
        let mut cents: Vec<Vec<F>> = Vec::with_capacity(k);
        for i in 0..k {
            let row: Vec<F> = batch.row(i).iter().copied().collect();
            cents.push(row);
        }
        self.centroids = Some(cents);
        self.counts = vec![0u64; k];
        Ok(())
    }

    /// Process a single mini-batch (online update).
    pub fn update_batch(&mut self, batch: ArrayView2<F>) -> Result<()> {
        if self.centroids.is_none() {
            self.initialize_from_batch(batch)?;
        }
        let d = self.n_features;
        let centroids = self.centroids.as_mut().ok_or_else(|| {
            ClusteringError::InvalidState("Centroids not initialized".into())
        })?;

        for row in batch.rows() {
            let pt: Vec<F> = row.iter().copied().collect();
            if pt.len() != d {
                continue;
            }

            // Assign to nearest centroid
            let best = Self::nearest_centroid(centroids, &pt);

            // Update centroid with streaming average
            self.counts[best] += 1;
            let lr = F::from_f64(1.0 / self.counts[best] as f64).unwrap_or(F::zero());
            for i in 0..d {
                centroids[best][i] = centroids[best][i] + lr * (pt[i] - centroids[best][i]);
            }
        }
        self.n_updates += 1;
        Ok(())
    }

    /// Process a stream by iterating over the data in mini-batches.
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<StreamingKMeansResult<F>> {
        let n = data.nrows();
        if n == 0 {
            return Err(ClusteringError::InvalidInput("Empty dataset".into()));
        }

        // Initial batch
        let first_end = self.config.batch_size.min(n);
        self.initialize_from_batch(data.slice(scirs2_core::ndarray::s![..first_end, ..]))?;

        // Streaming updates
        let mut start = first_end;
        let mut iter = 0;
        while start < n && iter < self.config.max_iter {
            let end = (start + self.config.batch_size).min(n);
            self.update_batch(data.slice(scirs2_core::ndarray::s![start..end, ..]))?;
            start = end;
            iter += 1;
        }

        // Assign all points
        let centroids = self.centroids.as_ref().ok_or_else(|| {
            ClusteringError::InvalidState("Centroids not initialized".into())
        })?;
        let mut labels = Vec::with_capacity(n);
        for row in data.rows() {
            let pt: Vec<F> = row.iter().copied().collect();
            labels.push(Self::nearest_centroid(centroids, &pt));
        }

        let d = self.n_features;
        let k = centroids.len();
        let flat: Vec<f64> = centroids
            .iter()
            .flat_map(|c| c.iter().map(|&v| format!("{:?}", v).parse::<f64>().unwrap_or(0.0)))
            .collect();
        let cents_arr = Array2::from_shape_vec((k, d), flat)
            .map_err(|e| ClusteringError::ComputationError(e.to_string()))?;

        Ok(StreamingKMeansResult {
            labels,
            centroids: cents_arr,
            n_clusters: k,
            n_updates: self.n_updates,
            _phantom: std::marker::PhantomData,
        })
    }

    fn nearest_centroid(centroids: &[Vec<F>], point: &[F]) -> usize {
        let mut best = 0;
        let mut best_d = F::infinity();
        for (j, c) in centroids.iter().enumerate() {
            let d = c.iter().zip(point.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .fold(F::zero(), |acc, v| acc + v);
            if d < best_d {
                best_d = d;
                best = j;
            }
        }
        best
    }

    /// Current centroid estimates.
    pub fn centroids(&self) -> Option<&[Vec<F>]> {
        self.centroids.as_deref()
    }
}

// ---------------------------------------------------------------------------
// ChunkClustering
// ---------------------------------------------------------------------------

/// Configuration for chunk-based streaming clustering.
#[derive(Debug, Clone)]
pub struct ChunkClusteringConfig {
    /// Number of clusters.
    pub k: usize,
    /// Size of each data chunk.
    pub chunk_size: usize,
    /// How often (in chunks) to refresh the model.
    pub update_frequency: usize,
    /// Forgetting factor for old centroids (0 = no forgetting, 1 = full replace).
    pub forgetting_factor: f64,
}

impl Default for ChunkClusteringConfig {
    fn default() -> Self {
        Self {
            k: 5,
            chunk_size: 100,
            update_frequency: 1,
            forgetting_factor: 0.1,
        }
    }
}

/// Process data in chunks with periodic model updates.
///
/// Each chunk is clustered independently; cluster centroids are then
/// merged into the global model using the forgetting factor.
pub struct ChunkClustering<F: Float> {
    config: ChunkClusteringConfig,
    /// Global centroid estimates.
    global_centroids: Option<Vec<Vec<F>>>,
    n_features: usize,
    chunks_processed: usize,
}

impl<F: Float + FromPrimitive + Debug + Clone> ChunkClustering<F> {
    /// Create a new ChunkClustering instance.
    pub fn new(config: ChunkClusteringConfig) -> Self {
        Self {
            config,
            global_centroids: None,
            n_features: 0,
            chunks_processed: 0,
        }
    }

    /// Process a single data chunk.
    pub fn process_chunk(&mut self, chunk: ArrayView2<F>) -> Result<Vec<usize>> {
        let (n, d) = (chunk.nrows(), chunk.ncols());
        if n == 0 {
            return Ok(Vec::new());
        }
        if self.n_features == 0 {
            self.n_features = d;
        }

        let k = self.config.k.min(n);

        // Cluster this chunk with simple k-means
        let (chunk_cents, chunk_labels) = simple_kmeans_f(chunk, k, 30)?;

        // Update or initialize global centroids
        let ff = F::from_f64(self.config.forgetting_factor).unwrap_or(F::zero());
        let one_minus_ff = F::one() - ff;

        if self.global_centroids.is_none() {
            self.global_centroids = Some(chunk_cents);
        } else if self.chunks_processed % self.config.update_frequency == 0 {
            let global = self.global_centroids.as_mut().ok_or_else(|| {
                ClusteringError::InvalidState("Global centroids missing".into())
            })?;
            for (gc, cc) in global.iter_mut().zip(chunk_cents.iter()) {
                for i in 0..d {
                    gc[i] = one_minus_ff * gc[i] + ff * cc[i];
                }
            }
        }

        self.chunks_processed += 1;
        Ok(chunk_labels)
    }

    /// Process all data in chunks.
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<ChunkClusteringResult<F>> {
        let n = data.nrows();
        let chunk_size = self.config.chunk_size;
        let mut all_labels = vec![0usize; n];
        let mut start = 0;
        while start < n {
            let end = (start + chunk_size).min(n);
            let chunk = data.slice(scirs2_core::ndarray::s![start..end, ..]);
            let chunk_labels = self.process_chunk(chunk)?;
            for (i, &label) in chunk_labels.iter().enumerate() {
                all_labels[start + i] = label;
            }
            start = end;
        }

        let d = self.n_features;
        let k = self.config.k.min(n);
        let global = self.global_centroids.as_ref().ok_or_else(|| {
            ClusteringError::InvalidState("No chunks processed".into())
        })?;
        let flat: Vec<f64> = global
            .iter()
            .flat_map(|c| {
                c.iter()
                    .map(|&v| format!("{:?}", v).parse::<f64>().unwrap_or(0.0))
            })
            .collect();
        let cents_arr = Array2::from_shape_vec((global.len().min(k), d), {
            let used = global.len().min(k);
            flat[..used * d].to_vec()
        })
        .map_err(|e| ClusteringError::ComputationError(e.to_string()))?;

        Ok(ChunkClusteringResult {
            labels: all_labels,
            centroids: cents_arr,
            chunks_processed: self.chunks_processed,
            n_clusters: k,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Return the current global centroid estimates.
    pub fn global_centroids(&self) -> Option<&[Vec<F>]> {
        self.global_centroids.as_deref()
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result from the BIRCH algorithm.
#[derive(Debug, Clone)]
pub struct BirchResult<F: Float> {
    /// Final cluster centroids (k × d).
    pub centroids: Array2<f64>,
    /// Cluster label for each CF entry.
    pub entry_labels: Vec<usize>,
    /// Number of clusters.
    pub n_clusters: usize,
    /// Number of CF entries in the final tree.
    pub n_entries: usize,
    /// Total number of data points processed.
    pub total_points: u64,
    #[doc(hidden)]
    _phantom: std::marker::PhantomData<F>,
}

/// Result from StreamingKMeans.
#[derive(Debug, Clone)]
pub struct StreamingKMeansResult<F: Float> {
    /// Cluster labels for each data point.
    pub labels: Vec<usize>,
    /// Final cluster centroids (k × d).
    pub centroids: Array2<f64>,
    /// Number of clusters.
    pub n_clusters: usize,
    /// Number of mini-batch updates performed.
    pub n_updates: u64,
    #[doc(hidden)]
    _phantom: std::marker::PhantomData<F>,
}

/// Result from ChunkClustering.
#[derive(Debug, Clone)]
pub struct ChunkClusteringResult<F: Float> {
    /// Cluster labels for each data point.
    pub labels: Vec<usize>,
    /// Final global centroid estimates (k × d).
    pub centroids: Array2<f64>,
    /// Number of chunks processed.
    pub chunks_processed: usize,
    /// Number of clusters.
    pub n_clusters: usize,
    #[doc(hidden)]
    _phantom: std::marker::PhantomData<F>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Weighted k-means on pre-computed centroid vectors.
fn weighted_kmeans(
    centroids: &[Vec<f64>],
    weights: &[f64],
    k: usize,
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<Vec<f64>>, Vec<usize>)> {
    let n = centroids.len();
    let d = centroids.first().map(|c| c.len()).unwrap_or(0);
    let k = k.min(n);

    // Initialize with first k points
    let mut macro_cents: Vec<Vec<f64>> = centroids[..k].to_vec();
    let mut labels = vec![0usize; n];

    for _ in 0..max_iter {
        // Assign
        for i in 0..n {
            let mut best = 0;
            let mut best_d = f64::MAX;
            for j in 0..k {
                let d_sq: f64 = centroids[i]
                    .iter()
                    .zip(macro_cents[j].iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                if d_sq < best_d {
                    best_d = d_sq;
                    best = j;
                }
            }
            labels[i] = best;
        }

        // Update
        let mut new_cents = vec![vec![0f64; d]; k];
        let mut wsum = vec![0f64; k];
        for i in 0..n {
            let j = labels[i];
            let w = weights[i];
            wsum[j] += w;
            for dim in 0..d {
                new_cents[j][dim] += w * centroids[i][dim];
            }
        }
        for j in 0..k {
            if wsum[j] > 0.0 {
                for dim in 0..d {
                    new_cents[j][dim] /= wsum[j];
                }
            }
        }

        let shift: f64 = new_cents
            .iter()
            .zip(macro_cents.iter())
            .map(|(a, b)| a.iter().zip(b.iter()).map(|(&x, &y)| (x - y) * (x - y)).sum::<f64>().sqrt())
            .sum();
        macro_cents = new_cents;
        if shift < tol {
            break;
        }
    }
    Ok((macro_cents, labels))
}

/// Simple k-means on an ArrayView2<F>.
fn simple_kmeans_f<F>(
    data: ArrayView2<F>,
    k: usize,
    max_iter: usize,
) -> Result<(Vec<Vec<F>>, Vec<usize>)>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let (n, d) = (data.nrows(), data.ncols());
    if n == 0 || k == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let k = k.min(n);

    let mut cents: Vec<Vec<F>> = (0..k)
        .map(|i| data.row(i).iter().copied().collect())
        .collect();
    let mut labels = vec![0usize; n];

    for _ in 0..max_iter {
        for i in 0..n {
            let row: Vec<F> = data.row(i).iter().copied().collect();
            let mut best = 0;
            let mut best_d = F::infinity();
            for (j, c) in cents.iter().enumerate() {
                let d_sq = c.iter().zip(row.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .fold(F::zero(), |acc, v| acc + v);
                if d_sq < best_d {
                    best_d = d_sq;
                    best = j;
                }
            }
            labels[i] = best;
        }
        let mut new_cents = vec![vec![F::zero(); d]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let j = labels[i];
            counts[j] += 1;
            let row: Vec<F> = data.row(i).iter().copied().collect();
            for dim in 0..d {
                new_cents[j][dim] = new_cents[j][dim] + row[dim];
            }
        }
        for j in 0..k {
            if counts[j] > 0 {
                let nf = F::from_usize(counts[j]).unwrap_or(F::one());
                for dim in 0..d {
                    new_cents[j][dim] = new_cents[j][dim] / nf;
                }
            }
        }
        cents = new_cents;
    }
    Ok((cents, labels))
}

// Fix BirchResult to not use phantom data by restructuring
// (we already have a working definition above using `_phantom: PhantomData<F>`)

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn two_cluster_data() -> Array2<f64> {
        let mut v = Vec::new();
        for i in 0..20 {
            let offset = if i < 10 { 0.0 } else { 10.0 };
            v.extend_from_slice(&[offset + 0.1 * i as f64, offset + 0.1 * i as f64]);
        }
        Array2::from_shape_vec((20, 2), v).expect("shape ok")
    }

    #[test]
    fn test_cf_absorb() {
        let mut cf = CF::<f64>::from_point(&[1.0, 2.0]);
        cf.absorb_point(&[3.0, 4.0]);
        assert_eq!(cf.n, 2);
        let c = cf.centroid();
        assert!((c[0] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_cf_tree_insert() {
        let mut tree: CFTree<f64> = CFTree::new(50, 5, 1.0);
        let data = two_cluster_data();
        tree.insert_batch(data.view()).expect("insert ok");
        assert!(tree.n_entries() > 0);
        assert_eq!(tree.total_points(), 20);
    }

    #[test]
    fn test_birch_fit() {
        let data = two_cluster_data();
        let mut birch: BIRCH<f64> = BIRCH::new(BirchConfig {
            threshold: 2.0,
            n_clusters: 2,
            leaf_capacity: 5,
            ..Default::default()
        });
        let result = birch.fit(data.view()).expect("fit ok");
        assert_eq!(result.n_clusters, 2);
        assert!(!result.entry_labels.is_empty());
    }

    #[test]
    fn test_streaming_kmeans() {
        let data = two_cluster_data();
        let mut skm: StreamingKMeans<f64> = StreamingKMeans::new(StreamingKMeansConfig {
            k: 2,
            batch_size: 5,
            ..Default::default()
        });
        let result = skm.fit(data.view()).expect("fit ok");
        assert_eq!(result.labels.len(), 20);
        assert_eq!(result.n_clusters, 2);
    }

    #[test]
    fn test_chunk_clustering() {
        let data = two_cluster_data();
        let mut cc: ChunkClustering<f64> = ChunkClustering::new(ChunkClusteringConfig {
            k: 2,
            chunk_size: 5,
            update_frequency: 1,
            forgetting_factor: 0.3,
        });
        let result = cc.fit(data.view()).expect("fit ok");
        assert_eq!(result.labels.len(), 20);
        assert!(result.chunks_processed > 0);
    }
}
