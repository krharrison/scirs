//! Streaming (online) clustering algorithms
//!
//! This module provides algorithms designed for clustering data streams where
//! points arrive sequentially and must be processed incrementally.
//!
//! # Algorithms
//!
//! - **CluStream**: Micro-cluster based stream clustering (Aggarwal et al. 2003)
//! - **DenStream**: Density-based stream clustering (Cao et al. 2006)
//! - **StreamKM++**: Coreset-based streaming k-means (Ackermann et al. 2012)
//! - **Sliding window clustering**: Fixed-window online clustering
//! - **Online K-means with forgetting factor**: Exponentially weighted online k-means

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::VecDeque;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Micro-cluster (shared primitive for CluStream / DenStream)
// ---------------------------------------------------------------------------

/// A micro-cluster summarising a set of nearby points.
#[derive(Debug, Clone)]
pub struct MicroCluster<F: Float> {
    /// Linear sum of points (LS).
    pub linear_sum: Vec<F>,
    /// Squared sum of points (SS).
    pub squared_sum: Vec<F>,
    /// Number of points absorbed.
    pub n_points: usize,
    /// Creation timestamp.
    pub creation_time: u64,
    /// Last update timestamp.
    pub last_update_time: u64,
    /// Weight (may differ from n_points for fading models).
    pub weight: F,
}

impl<F: Float + FromPrimitive + Debug> MicroCluster<F> {
    /// Create a new micro-cluster from a single point.
    pub fn from_point(point: &[F], timestamp: u64) -> Self {
        let d = point.len();
        let mut ls = vec![F::zero(); d];
        let mut ss = vec![F::zero(); d];
        for i in 0..d {
            ls[i] = point[i];
            ss[i] = point[i] * point[i];
        }
        Self {
            linear_sum: ls,
            squared_sum: ss,
            n_points: 1,
            creation_time: timestamp,
            last_update_time: timestamp,
            weight: F::one(),
        }
    }

    /// Centroid of the micro-cluster.
    pub fn centroid(&self) -> Vec<F> {
        if self.weight <= F::epsilon() {
            return self.linear_sum.clone();
        }
        self.linear_sum.iter().map(|&v| v / self.weight).collect()
    }

    /// Radius (RMS deviation from centroid).
    pub fn radius(&self) -> F {
        if self.weight <= F::one() {
            return F::zero();
        }
        let d = self.linear_sum.len();
        let w = self.weight;
        let mut variance = F::zero();
        for i in 0..d {
            let mean = self.linear_sum[i] / w;
            let mean_sq = self.squared_sum[i] / w;
            let v = mean_sq - mean * mean;
            variance = variance + if v > F::zero() { v } else { F::zero() };
        }
        (variance / F::from(d).unwrap_or_else(|| F::one())).sqrt()
    }

    /// Absorb a single point.
    pub fn absorb(&mut self, point: &[F], timestamp: u64) {
        let d = self.linear_sum.len().min(point.len());
        for i in 0..d {
            self.linear_sum[i] = self.linear_sum[i] + point[i];
            self.squared_sum[i] = self.squared_sum[i] + point[i] * point[i];
        }
        self.n_points += 1;
        self.weight = self.weight + F::one();
        self.last_update_time = timestamp;
    }

    /// Merge another micro-cluster into this one.
    pub fn merge(&mut self, other: &MicroCluster<F>) {
        let d = self.linear_sum.len().min(other.linear_sum.len());
        for i in 0..d {
            self.linear_sum[i] = self.linear_sum[i] + other.linear_sum[i];
            self.squared_sum[i] = self.squared_sum[i] + other.squared_sum[i];
        }
        self.n_points += other.n_points;
        self.weight = self.weight + other.weight;
        if other.last_update_time > self.last_update_time {
            self.last_update_time = other.last_update_time;
        }
    }

    /// Apply exponential fading with factor lambda over elapsed time.
    pub fn apply_fading(&mut self, lambda: F, elapsed: F) {
        let factor = (F::zero() - lambda * elapsed).exp();
        let d = self.linear_sum.len();
        for i in 0..d {
            self.linear_sum[i] = self.linear_sum[i] * factor;
            self.squared_sum[i] = self.squared_sum[i] * factor;
        }
        self.weight = self.weight * factor;
    }

    /// Squared distance from centroid to a point.
    fn distance_sq_to(&self, point: &[F]) -> F {
        let centroid = self.centroid();
        let d = centroid.len().min(point.len());
        let mut s = F::zero();
        for i in 0..d {
            let diff = centroid[i] - point[i];
            s = s + diff * diff;
        }
        s
    }
}

// ---------------------------------------------------------------------------
// CluStream
// ---------------------------------------------------------------------------

/// Configuration for the CluStream algorithm.
#[derive(Debug, Clone)]
pub struct CluStreamConfig {
    /// Maximum number of micro-clusters to maintain.
    pub max_micro_clusters: usize,
    /// Number of macro-clusters for final output.
    pub n_macro_clusters: usize,
    /// Time horizon for snapshot pyramids (T in the paper).
    pub time_horizon: u64,
    /// Maximum radius factor for absorbing into a micro-cluster.
    pub radius_factor: f64,
}

impl Default for CluStreamConfig {
    fn default() -> Self {
        Self {
            max_micro_clusters: 100,
            n_macro_clusters: 5,
            time_horizon: 1000,
            radius_factor: 2.0,
        }
    }
}

/// CluStream online clustering algorithm.
///
/// Maintains a set of micro-clusters that summarise the data stream.
/// Periodically, macro-clustering (e.g. weighted k-means on micro-cluster
/// centroids) produces the final cluster assignments.
pub struct CluStream<F: Float> {
    config: CluStreamConfig,
    micro_clusters: Vec<MicroCluster<F>>,
    current_time: u64,
    n_features: usize,
    initialized: bool,
}

impl<F: Float + FromPrimitive + Debug> CluStream<F> {
    /// Create a new CluStream instance.
    pub fn new(config: CluStreamConfig) -> Self {
        Self {
            config,
            micro_clusters: Vec::new(),
            current_time: 0,
            n_features: 0,
            initialized: false,
        }
    }

    /// Initialize with a batch of points.
    pub fn initialize(&mut self, data: ArrayView2<F>) -> Result<()> {
        let (n, d) = (data.shape()[0], data.shape()[1]);
        if n == 0 {
            return Err(ClusteringError::InvalidInput("Empty data".into()));
        }
        self.n_features = d;

        // Run simple k-means to get initial micro-clusters
        let k = self.config.max_micro_clusters.min(n);
        let labels = simple_kmeans_init(data, k);

        self.micro_clusters.clear();
        for ci in 0..k {
            let mut mc: Option<MicroCluster<F>> = None;
            for i in 0..n {
                if labels[i] == ci as i32 {
                    match mc.as_mut() {
                        Some(m) => m.absorb(data.row(i).as_slice().unwrap_or(&[]), 0),
                        None => {
                            mc = Some(MicroCluster::from_point(
                                data.row(i).as_slice().unwrap_or(&[]),
                                0,
                            ));
                        }
                    }
                }
            }
            if let Some(m) = mc {
                self.micro_clusters.push(m);
            }
        }
        self.initialized = true;
        Ok(())
    }

    /// Process a single new data point from the stream.
    pub fn process_point(&mut self, point: &[F]) -> Result<()> {
        if !self.initialized {
            return Err(ClusteringError::InvalidState(
                "CluStream not initialized".into(),
            ));
        }
        self.current_time += 1;

        // Find nearest micro-cluster
        let (nearest_idx, nearest_dist) = self.find_nearest_mc(point);

        let rf = F::from(self.config.radius_factor)
            .unwrap_or_else(|| F::from(2.0).unwrap_or_else(|| F::one()));

        // Check if point fits within the micro-cluster radius
        let fits = if let Some(mc) = self.micro_clusters.get(nearest_idx) {
            let r = mc.radius();
            nearest_dist.sqrt() <= r * rf + F::epsilon()
        } else {
            false
        };

        if fits {
            if let Some(mc) = self.micro_clusters.get_mut(nearest_idx) {
                mc.absorb(point, self.current_time);
            }
        } else {
            // Create a new micro-cluster
            if self.micro_clusters.len() >= self.config.max_micro_clusters {
                // Merge two closest micro-clusters to make room
                self.merge_closest_pair();
            }
            self.micro_clusters
                .push(MicroCluster::from_point(point, self.current_time));
        }

        Ok(())
    }

    /// Process a batch of points.
    pub fn process_batch(&mut self, data: ArrayView2<F>) -> Result<()> {
        for i in 0..data.shape()[0] {
            let row = data.row(i);
            self.process_point(row.as_slice().unwrap_or(&[]))?;
        }
        Ok(())
    }

    /// Get current macro-cluster labels for the micro-clusters.
    ///
    /// Returns (micro-cluster centroids, macro-cluster labels for each micro-cluster).
    pub fn get_macro_clusters(&self) -> Result<(Array2<F>, Array1<i32>)> {
        if self.micro_clusters.is_empty() {
            return Err(ClusteringError::InvalidState(
                "No micro-clusters available".into(),
            ));
        }

        let n_mc = self.micro_clusters.len();
        let d = self.n_features;
        let k = self.config.n_macro_clusters.min(n_mc);

        // Build matrix of micro-cluster centroids with weights
        let mut centroids = Array2::<F>::zeros((n_mc, d));
        for (i, mc) in self.micro_clusters.iter().enumerate() {
            let c = mc.centroid();
            for j in 0..d.min(c.len()) {
                centroids[[i, j]] = c[j];
            }
        }

        let labels = simple_kmeans_init(centroids.view(), k);
        Ok((centroids, labels))
    }

    /// Number of current micro-clusters.
    pub fn n_micro_clusters(&self) -> usize {
        self.micro_clusters.len()
    }

    /// Get reference to micro-clusters.
    pub fn micro_clusters(&self) -> &[MicroCluster<F>] {
        &self.micro_clusters
    }

    fn find_nearest_mc(&self, point: &[F]) -> (usize, F) {
        let mut best_idx = 0;
        let mut best_dist = F::infinity();
        for (i, mc) in self.micro_clusters.iter().enumerate() {
            let d = mc.distance_sq_to(point);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        (best_idx, best_dist)
    }

    fn merge_closest_pair(&mut self) {
        if self.micro_clusters.len() < 2 {
            return;
        }
        let n = self.micro_clusters.len();
        let mut best_i = 0;
        let mut best_j = 1;
        let mut best_dist = F::infinity();
        for i in 0..n {
            let ci = self.micro_clusters[i].centroid();
            for j in (i + 1)..n {
                let cj = self.micro_clusters[j].centroid();
                let d: F = ci
                    .iter()
                    .zip(cj.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .fold(F::zero(), |acc, v| acc + v);
                if d < best_dist {
                    best_dist = d;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        // Merge j into i, remove j
        let mc_j = self.micro_clusters[best_j].clone();
        self.micro_clusters[best_i].merge(&mc_j);
        self.micro_clusters.remove(best_j);
    }
}

// ---------------------------------------------------------------------------
// DenStream
// ---------------------------------------------------------------------------

/// Configuration for the DenStream algorithm.
#[derive(Debug, Clone)]
pub struct DenStreamConfig {
    /// Epsilon radius for DBSCAN-like macro-clustering.
    pub epsilon: f64,
    /// Minimum weight for a micro-cluster to be considered potential.
    pub min_points: usize,
    /// Fading factor (lambda): higher = faster forgetting.
    pub lambda: f64,
    /// Beta: threshold factor for potential vs outlier micro-clusters.
    pub beta: f64,
    /// Mu: minimum weight factor for potential micro-clusters.
    pub mu: f64,
    /// Time period for outlier cleanup.
    pub cleanup_interval: u64,
}

impl Default for DenStreamConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            min_points: 3,
            lambda: 0.25,
            beta: 0.2,
            mu: 1.0,
            cleanup_interval: 100,
        }
    }
}

/// DenStream: density-based stream clustering.
///
/// Maintains potential micro-clusters (p-micro-clusters) and outlier
/// micro-clusters (o-micro-clusters). Points are absorbed into nearby
/// p-micro-clusters or create outlier micro-clusters that may be promoted.
pub struct DenStream<F: Float> {
    config: DenStreamConfig,
    /// Potential micro-clusters.
    p_micro_clusters: Vec<MicroCluster<F>>,
    /// Outlier micro-clusters.
    o_micro_clusters: Vec<MicroCluster<F>>,
    current_time: u64,
    n_features: usize,
    initialized: bool,
}

impl<F: Float + FromPrimitive + Debug> DenStream<F> {
    /// Create a new DenStream instance.
    pub fn new(config: DenStreamConfig) -> Self {
        Self {
            config,
            p_micro_clusters: Vec::new(),
            o_micro_clusters: Vec::new(),
            current_time: 0,
            n_features: 0,
            initialized: false,
        }
    }

    /// Initialize with a batch of data.
    pub fn initialize(&mut self, data: ArrayView2<F>) -> Result<()> {
        let (n, d) = (data.shape()[0], data.shape()[1]);
        if n == 0 {
            return Err(ClusteringError::InvalidInput("Empty data".into()));
        }
        self.n_features = d;

        // DBSCAN-like initialization to create initial p-micro-clusters
        let eps = F::from(self.config.epsilon).unwrap_or_else(|| F::one());
        let min_pts = self.config.min_points;

        // Simple: group nearby points into micro-clusters
        let mut assigned = vec![false; n];
        for i in 0..n {
            if assigned[i] {
                continue;
            }
            let mut mc = MicroCluster::from_point(data.row(i).as_slice().unwrap_or(&[]), 0);
            assigned[i] = true;
            for j in (i + 1)..n {
                if assigned[j] {
                    continue;
                }
                let dist_sq = row_dist_sq(data.row(i), data.row(j));
                if dist_sq <= eps * eps {
                    mc.absorb(data.row(j).as_slice().unwrap_or(&[]), 0);
                    assigned[j] = true;
                }
            }
            if mc.n_points >= min_pts {
                self.p_micro_clusters.push(mc);
            } else {
                self.o_micro_clusters.push(mc);
            }
        }

        self.initialized = true;
        Ok(())
    }

    /// Process a single new data point.
    pub fn process_point(&mut self, point: &[F]) -> Result<()> {
        if !self.initialized {
            return Err(ClusteringError::InvalidState(
                "DenStream not initialized".into(),
            ));
        }
        self.current_time += 1;
        let lambda_f = F::from(self.config.lambda).unwrap_or_else(|| F::zero());
        let eps = F::from(self.config.epsilon).unwrap_or_else(|| F::one());
        let mu_f = F::from(self.config.mu).unwrap_or_else(|| F::one());

        // Try to absorb into nearest p-micro-cluster
        let (p_idx, p_dist) = nearest_mc_idx(&self.p_micro_clusters, point);
        if !self.p_micro_clusters.is_empty() && p_dist.sqrt() <= eps {
            self.p_micro_clusters[p_idx].absorb(point, self.current_time);
        } else {
            // Try outlier micro-clusters
            let (o_idx, o_dist) = nearest_mc_idx(&self.o_micro_clusters, point);
            if !self.o_micro_clusters.is_empty() && o_dist.sqrt() <= eps {
                self.o_micro_clusters[o_idx].absorb(point, self.current_time);
                // Check if outlier should be promoted
                if self.o_micro_clusters[o_idx].weight >= mu_f {
                    let promoted = self.o_micro_clusters.remove(o_idx);
                    self.p_micro_clusters.push(promoted);
                }
            } else {
                // Create new outlier micro-cluster
                self.o_micro_clusters
                    .push(MicroCluster::from_point(point, self.current_time));
            }
        }

        // Periodic cleanup
        if self.current_time % self.config.cleanup_interval == 0 {
            self.cleanup(lambda_f);
        }

        Ok(())
    }

    /// Process a batch of points.
    pub fn process_batch(&mut self, data: ArrayView2<F>) -> Result<()> {
        for i in 0..data.shape()[0] {
            self.process_point(data.row(i).as_slice().unwrap_or(&[]))?;
        }
        Ok(())
    }

    /// Get current cluster labels by running DBSCAN on p-micro-cluster centroids.
    pub fn get_clusters(&self) -> Result<(Array2<F>, Array1<i32>)> {
        if self.p_micro_clusters.is_empty() {
            return Err(ClusteringError::InvalidState(
                "No potential micro-clusters".into(),
            ));
        }

        let n = self.p_micro_clusters.len();
        let d = self.n_features;
        let mut centroids = Array2::<F>::zeros((n, d));
        for (i, mc) in self.p_micro_clusters.iter().enumerate() {
            let c = mc.centroid();
            for j in 0..d.min(c.len()) {
                centroids[[i, j]] = c[j];
            }
        }

        // Simple DBSCAN on centroids
        let eps = F::from(self.config.epsilon).unwrap_or_else(|| F::one());
        let labels = dbscan_on_centroids(&centroids, eps, self.config.min_points);

        Ok((centroids, labels))
    }

    /// Number of potential micro-clusters.
    pub fn n_potential(&self) -> usize {
        self.p_micro_clusters.len()
    }

    /// Number of outlier micro-clusters.
    pub fn n_outliers(&self) -> usize {
        self.o_micro_clusters.len()
    }

    fn cleanup(&mut self, lambda: F) {
        let one = F::one();
        let beta_f = F::from(self.config.beta).unwrap_or_else(|| F::zero());
        let mu_f = F::from(self.config.mu).unwrap_or_else(|| F::one());

        // Apply fading to all micro-clusters
        for mc in self.p_micro_clusters.iter_mut() {
            mc.apply_fading(lambda, one);
        }
        for mc in self.o_micro_clusters.iter_mut() {
            mc.apply_fading(lambda, one);
        }

        // Remove p-micro-clusters that fell below threshold
        let threshold = beta_f * mu_f;
        self.p_micro_clusters.retain(|mc| mc.weight >= threshold);

        // Remove very weak outlier micro-clusters
        let outlier_threshold = F::from(0.01).unwrap_or_else(|| F::epsilon());
        self.o_micro_clusters
            .retain(|mc| mc.weight >= outlier_threshold);
    }
}

// ---------------------------------------------------------------------------
// StreamKM++ (coreset-based)
// ---------------------------------------------------------------------------

/// Configuration for StreamKM++.
#[derive(Debug, Clone)]
pub struct StreamKMConfig {
    /// Number of final clusters.
    pub n_clusters: usize,
    /// Coreset size (number of weighted representatives to maintain).
    pub coreset_size: usize,
    /// Number of k-means iterations on the final coreset.
    pub kmeans_iterations: usize,
}

impl Default for StreamKMConfig {
    fn default() -> Self {
        Self {
            n_clusters: 5,
            coreset_size: 200,
            kmeans_iterations: 50,
        }
    }
}

/// Coreset point with weight.
#[derive(Debug, Clone)]
pub struct CoresetPoint<F: Float> {
    /// Coordinates.
    pub coords: Vec<F>,
    /// Weight (number of original points represented).
    pub weight: F,
}

/// StreamKM++: coreset-based streaming k-means.
///
/// Maintains a weighted coreset that summarises the stream. When the
/// buffer overflows, a merge-and-reduce step compresses it back down
/// using k-means++ seeding to select coreset representatives.
pub struct StreamKMPlusPlus<F: Float> {
    config: StreamKMConfig,
    coreset: Vec<CoresetPoint<F>>,
    buffer: Vec<Vec<F>>,
    n_features: usize,
    initialized: bool,
}

impl<F: Float + FromPrimitive + Debug> StreamKMPlusPlus<F> {
    /// Create a new StreamKM++ instance.
    pub fn new(config: StreamKMConfig) -> Self {
        Self {
            config,
            coreset: Vec::new(),
            buffer: Vec::new(),
            n_features: 0,
            initialized: false,
        }
    }

    /// Process a single point from the stream.
    pub fn process_point(&mut self, point: &[F]) -> Result<()> {
        if !self.initialized {
            self.n_features = point.len();
            self.initialized = true;
        }
        self.buffer.push(point.to_vec());

        // When buffer is full, merge-and-reduce
        if self.buffer.len() >= self.config.coreset_size {
            self.merge_reduce()?;
        }
        Ok(())
    }

    /// Process a batch of points.
    pub fn process_batch(&mut self, data: ArrayView2<F>) -> Result<()> {
        for i in 0..data.shape()[0] {
            self.process_point(data.row(i).as_slice().unwrap_or(&[]))?;
        }
        Ok(())
    }

    /// Get final cluster centroids and coreset labels.
    pub fn get_clusters(&self) -> Result<(Array2<F>, Array1<i32>)> {
        // Combine coreset and buffer into a weighted point set
        let mut all_points: Vec<(Vec<F>, F)> = Vec::new();
        for cp in &self.coreset {
            all_points.push((cp.coords.clone(), cp.weight));
        }
        for bp in &self.buffer {
            all_points.push((bp.clone(), F::one()));
        }

        if all_points.is_empty() {
            return Err(ClusteringError::InvalidState(
                "No data processed yet".into(),
            ));
        }

        let n = all_points.len();
        let d = self.n_features;
        let k = self.config.n_clusters.min(n);

        // Build matrix
        let mut mat = Array2::<F>::zeros((n, d));
        let mut weights = Array1::<F>::zeros(n);
        for (i, (coords, w)) in all_points.iter().enumerate() {
            for j in 0..d.min(coords.len()) {
                mat[[i, j]] = coords[j];
            }
            weights[i] = *w;
        }

        // Weighted k-means
        let labels = weighted_kmeans(mat.view(), &weights, k, self.config.kmeans_iterations);

        // Compute centroids
        let mut centroids = Array2::<F>::zeros((k, d));
        let mut total_weights = vec![F::zero(); k];
        for i in 0..n {
            let ci = labels[i] as usize;
            if ci < k {
                total_weights[ci] = total_weights[ci] + weights[i];
                for j in 0..d {
                    centroids[[ci, j]] = centroids[[ci, j]] + mat[[i, j]] * weights[i];
                }
            }
        }
        for ci in 0..k {
            if total_weights[ci] > F::epsilon() {
                for j in 0..d {
                    centroids[[ci, j]] = centroids[[ci, j]] / total_weights[ci];
                }
            }
        }

        Ok((centroids, labels))
    }

    /// Current coreset size (number of weighted representatives).
    pub fn coreset_size(&self) -> usize {
        self.coreset.len()
    }

    fn merge_reduce(&mut self) -> Result<()> {
        // Combine coreset + buffer into one set, then k-means++ reduce
        let mut all: Vec<(Vec<F>, F)> = Vec::new();
        for cp in self.coreset.drain(..) {
            all.push((cp.coords, cp.weight));
        }
        for bp in self.buffer.drain(..) {
            all.push((bp, F::one()));
        }

        let target = self.config.coreset_size / 2;
        if all.len() <= target {
            for (coords, w) in all {
                self.coreset.push(CoresetPoint { coords, weight: w });
            }
            return Ok(());
        }

        let n = all.len();
        let d = self.n_features;
        let k = target.min(n);

        // Build matrix for k-means
        let mut mat = Array2::<F>::zeros((n, d));
        let mut weights = Array1::<F>::zeros(n);
        for (i, (coords, w)) in all.iter().enumerate() {
            for j in 0..d.min(coords.len()) {
                mat[[i, j]] = coords[j];
            }
            weights[i] = *w;
        }

        let labels = weighted_kmeans(mat.view(), &weights, k, 10);

        // Build new coreset from cluster centroids with summed weights
        for ci in 0..k {
            let mut sum = vec![F::zero(); d];
            let mut total_w = F::zero();
            for i in 0..n {
                if labels[i] == ci as i32 {
                    total_w = total_w + weights[i];
                    for j in 0..d {
                        sum[j] = sum[j] + mat[[i, j]] * weights[i];
                    }
                }
            }
            if total_w > F::epsilon() {
                for j in 0..d {
                    sum[j] = sum[j] / total_w;
                }
                self.coreset.push(CoresetPoint {
                    coords: sum,
                    weight: total_w,
                });
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Sliding Window Clustering
// ---------------------------------------------------------------------------

/// Configuration for sliding window clustering.
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Window size (number of recent points to keep).
    pub window_size: usize,
    /// Number of clusters.
    pub n_clusters: usize,
    /// K-means iterations per query.
    pub kmeans_iterations: usize,
}

impl Default for SlidingWindowConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            n_clusters: 5,
            kmeans_iterations: 20,
        }
    }
}

/// Sliding window clustering: maintains a fixed-size window of the most
/// recent points and clusters them on demand.
pub struct SlidingWindowClustering<F: Float> {
    config: SlidingWindowConfig,
    window: VecDeque<Vec<F>>,
    n_features: usize,
}

impl<F: Float + FromPrimitive + Debug> SlidingWindowClustering<F> {
    /// Create a new sliding window clustering instance.
    pub fn new(config: SlidingWindowConfig) -> Self {
        Self {
            config,
            window: VecDeque::new(),
            n_features: 0,
        }
    }

    /// Add a single point to the window.
    pub fn add_point(&mut self, point: &[F]) {
        if self.n_features == 0 {
            self.n_features = point.len();
        }
        self.window.push_back(point.to_vec());
        if self.window.len() > self.config.window_size {
            self.window.pop_front();
        }
    }

    /// Add a batch of points.
    pub fn add_batch(&mut self, data: ArrayView2<F>) {
        for i in 0..data.shape()[0] {
            self.add_point(data.row(i).as_slice().unwrap_or(&[]));
        }
    }

    /// Get current clustering of the window contents.
    pub fn get_clusters(&self) -> Result<(Array2<F>, Array1<i32>)> {
        if self.window.is_empty() {
            return Err(ClusteringError::InvalidState("Window is empty".into()));
        }

        let n = self.window.len();
        let d = self.n_features;
        let k = self.config.n_clusters.min(n);

        let mut mat = Array2::<F>::zeros((n, d));
        for (i, pt) in self.window.iter().enumerate() {
            for j in 0..d.min(pt.len()) {
                mat[[i, j]] = pt[j];
            }
        }

        let labels = simple_kmeans_init(mat.view(), k);

        // Compute centroids
        let mut centroids = Array2::<F>::zeros((k, d));
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let ci = labels[i] as usize;
            if ci < k {
                counts[ci] += 1;
                for j in 0..d {
                    centroids[[ci, j]] = centroids[[ci, j]] + mat[[i, j]];
                }
            }
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                let cnt = F::from(counts[ci]).unwrap_or_else(|| F::one());
                for j in 0..d {
                    centroids[[ci, j]] = centroids[[ci, j]] / cnt;
                }
            }
        }

        Ok((centroids, labels))
    }

    /// Current number of points in the window.
    pub fn window_len(&self) -> usize {
        self.window.len()
    }
}

// ---------------------------------------------------------------------------
// Online K-Means with Forgetting Factor
// ---------------------------------------------------------------------------

/// Configuration for online k-means with forgetting.
#[derive(Debug, Clone)]
pub struct OnlineKMeansConfig {
    /// Number of clusters.
    pub n_clusters: usize,
    /// Forgetting factor in (0, 1]. 1.0 = no forgetting (standard online).
    pub forgetting_factor: f64,
    /// Learning rate schedule: if true, use 1/n_i decay; otherwise constant.
    pub adaptive_learning: bool,
}

impl Default for OnlineKMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 5,
            forgetting_factor: 0.99,
            adaptive_learning: true,
        }
    }
}

/// Online K-means with exponential forgetting factor.
///
/// Maintains cluster centroids that are updated incrementally.
/// The forgetting factor downweights older contributions, allowing
/// the algorithm to adapt to concept drift.
pub struct OnlineKMeans<F: Float> {
    config: OnlineKMeansConfig,
    centroids: Option<Array2<F>>,
    cluster_counts: Vec<F>,
    n_features: usize,
    initialized: bool,
    total_points: usize,
}

impl<F: Float + FromPrimitive + Debug> OnlineKMeans<F> {
    /// Create a new online k-means instance.
    pub fn new(config: OnlineKMeansConfig) -> Self {
        Self {
            config,
            centroids: None,
            cluster_counts: Vec::new(),
            n_features: 0,
            initialized: false,
            total_points: 0,
        }
    }

    /// Initialize with a batch of data (used for seeding centroids).
    pub fn initialize(&mut self, data: ArrayView2<F>) -> Result<()> {
        let (n, d) = (data.shape()[0], data.shape()[1]);
        if n == 0 {
            return Err(ClusteringError::InvalidInput("Empty data".into()));
        }
        self.n_features = d;
        let k = self.config.n_clusters.min(n);

        let labels = simple_kmeans_init(data, k);

        let mut centroids = Array2::<F>::zeros((k, d));
        let mut counts = vec![F::zero(); k];
        for i in 0..n {
            let ci = labels[i] as usize;
            if ci < k {
                counts[ci] = counts[ci] + F::one();
                for j in 0..d {
                    centroids[[ci, j]] = centroids[[ci, j]] + data[[i, j]];
                }
            }
        }
        for ci in 0..k {
            if counts[ci] > F::epsilon() {
                for j in 0..d {
                    centroids[[ci, j]] = centroids[[ci, j]] / counts[ci];
                }
            }
        }

        self.centroids = Some(centroids);
        self.cluster_counts = counts;
        self.initialized = true;
        self.total_points = n;
        Ok(())
    }

    /// Process a single new point.
    pub fn process_point(&mut self, point: &[F]) -> Result<i32> {
        if !self.initialized {
            return Err(ClusteringError::InvalidState(
                "OnlineKMeans not initialized".into(),
            ));
        }

        let centroids = self
            .centroids
            .as_ref()
            .ok_or_else(|| ClusteringError::InvalidState("No centroids".into()))?;

        let k = centroids.shape()[0];
        let d = centroids.shape()[1];
        let ff = F::from(self.config.forgetting_factor).unwrap_or_else(|| F::one());

        // Find nearest centroid
        let mut best_ci = 0;
        let mut best_dist = F::infinity();
        for ci in 0..k {
            let mut dist = F::zero();
            for j in 0..d.min(point.len()) {
                let diff = point[j] - centroids[[ci, j]];
                dist = dist + diff * diff;
            }
            if dist < best_dist {
                best_dist = dist;
                best_ci = ci;
            }
        }

        // Apply forgetting factor to all cluster counts
        for ci in 0..k {
            self.cluster_counts[ci] = self.cluster_counts[ci] * ff;
        }

        // Update centroid
        self.cluster_counts[best_ci] = self.cluster_counts[best_ci] + F::one();
        let eta = if self.config.adaptive_learning {
            F::one() / self.cluster_counts[best_ci]
        } else {
            F::from(0.01).unwrap_or_else(|| F::epsilon())
        };

        let centroids_mut = self
            .centroids
            .as_mut()
            .ok_or_else(|| ClusteringError::InvalidState("No centroids".into()))?;
        for j in 0..d.min(point.len()) {
            centroids_mut[[best_ci, j]] =
                centroids_mut[[best_ci, j]] + eta * (point[j] - centroids_mut[[best_ci, j]]);
        }

        self.total_points += 1;
        Ok(best_ci as i32)
    }

    /// Process a batch and return labels.
    pub fn process_batch(&mut self, data: ArrayView2<F>) -> Result<Array1<i32>> {
        let n = data.shape()[0];
        let mut labels = Array1::from_elem(n, -1i32);
        for i in 0..n {
            labels[i] = self.process_point(data.row(i).as_slice().unwrap_or(&[]))?;
        }
        Ok(labels)
    }

    /// Get current centroids.
    pub fn centroids(&self) -> Option<&Array2<F>> {
        self.centroids.as_ref()
    }

    /// Predict cluster for a point without updating.
    pub fn predict(&self, point: &[F]) -> Result<i32> {
        let centroids = self
            .centroids
            .as_ref()
            .ok_or_else(|| ClusteringError::InvalidState("Not initialized".into()))?;
        let k = centroids.shape()[0];
        let d = centroids.shape()[1];
        let mut best_ci = 0i32;
        let mut best_dist = F::infinity();
        for ci in 0..k {
            let mut dist = F::zero();
            for j in 0..d.min(point.len()) {
                let diff = point[j] - centroids[[ci, j]];
                dist = dist + diff * diff;
            }
            if dist < best_dist {
                best_dist = dist;
                best_ci = ci as i32;
            }
        }
        Ok(best_ci)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Squared distance between two array rows.
fn row_dist_sq<F: Float>(a: ArrayView1<F>, b: ArrayView1<F>) -> F {
    let mut s = F::zero();
    for i in 0..a.len().min(b.len()) {
        let diff = a[i] - b[i];
        s = s + diff * diff;
    }
    s
}

/// Find the nearest micro-cluster to a point; returns (index, sq distance).
fn nearest_mc_idx<F: Float + FromPrimitive + Debug>(
    clusters: &[MicroCluster<F>],
    point: &[F],
) -> (usize, F) {
    let mut best = 0;
    let mut best_d = F::infinity();
    for (i, mc) in clusters.iter().enumerate() {
        let d = mc.distance_sq_to(point);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    (best, best_d)
}

/// Simple DBSCAN on a centroid matrix (for DenStream macro-clustering).
fn dbscan_on_centroids<F: Float + FromPrimitive + Debug>(
    centroids: &Array2<F>,
    eps: F,
    min_pts: usize,
) -> Array1<i32> {
    let n = centroids.shape()[0];
    let eps_sq = eps * eps;
    let mut labels = vec![-2i32; n]; // -2 = undefined, -1 = noise
    let mut cluster_id = 0i32;

    for i in 0..n {
        if labels[i] != -2 {
            continue;
        }
        let neighbors: Vec<usize> = (0..n)
            .filter(|&j| {
                let d = row_dist_sq(centroids.row(i), centroids.row(j));
                d <= eps_sq
            })
            .collect();

        if neighbors.len() < min_pts {
            labels[i] = -1;
            continue;
        }

        labels[i] = cluster_id;
        let mut queue = neighbors.clone();
        let mut head = 0usize;
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

            let cur_neighbors: Vec<usize> = (0..n)
                .filter(|&j| {
                    let d = row_dist_sq(centroids.row(cur), centroids.row(j));
                    d <= eps_sq
                })
                .collect();

            if cur_neighbors.len() >= min_pts {
                for nb in cur_neighbors {
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

/// Simple k-means for initialization purposes.
fn simple_kmeans_init<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    k: usize,
) -> Array1<i32> {
    let (n, d) = (data.shape()[0], data.shape()[1]);
    if n == 0 || k == 0 {
        return Array1::from_elem(n, 0i32);
    }
    let k = k.min(n);

    // K-means++ style init
    let mut centroids = Array2::<F>::zeros((k, d));
    centroids.row_mut(0).assign(&data.row(0));

    for ci in 1..k {
        let mut best_idx = 0;
        let mut best_dist = F::zero();
        for i in 0..n {
            let mut min_d = F::infinity();
            for prev in 0..ci {
                let d = row_dist_sq(data.row(i), centroids.row(prev));
                if d < min_d {
                    min_d = d;
                }
            }
            if min_d > best_dist {
                best_dist = min_d;
                best_idx = i;
            }
        }
        centroids.row_mut(ci).assign(&data.row(best_idx));
    }

    let mut labels = Array1::from_elem(n, 0i32);
    for _ in 0..20 {
        // Assign
        let mut changed = false;
        for i in 0..n {
            let mut best_ci = 0i32;
            let mut best_d = F::infinity();
            for ci in 0..k {
                let d = row_dist_sq(data.row(i), centroids.row(ci));
                if d < best_d {
                    best_d = d;
                    best_ci = ci as i32;
                }
            }
            if labels[i] != best_ci {
                labels[i] = best_ci;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        // Update
        let mut counts = vec![0usize; k];
        let mut sums = Array2::<F>::zeros((k, d));
        for i in 0..n {
            let ci = labels[i] as usize;
            counts[ci] += 1;
            for j in 0..d {
                sums[[ci, j]] = sums[[ci, j]] + data[[i, j]];
            }
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                let cnt = F::from(counts[ci]).unwrap_or_else(|| F::one());
                for j in 0..d {
                    centroids[[ci, j]] = sums[[ci, j]] / cnt;
                }
            }
        }
    }
    labels
}

/// Weighted k-means.
fn weighted_kmeans<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    weights: &Array1<F>,
    k: usize,
    max_iter: usize,
) -> Array1<i32> {
    let (n, d) = (data.shape()[0], data.shape()[1]);
    if n == 0 || k == 0 {
        return Array1::from_elem(n, 0i32);
    }
    let k = k.min(n);

    // Init: pick k spread points
    let mut centroids = Array2::<F>::zeros((k, d));
    let step = (n as f64 / k as f64).max(1.0);
    for ci in 0..k {
        let idx = ((ci as f64 * step) as usize).min(n - 1);
        centroids.row_mut(ci).assign(&data.row(idx));
    }

    let mut labels = Array1::from_elem(n, 0i32);
    for _ in 0..max_iter {
        let mut changed = false;
        for i in 0..n {
            let mut best = 0i32;
            let mut best_d = F::infinity();
            for ci in 0..k {
                let dd = row_dist_sq(data.row(i), centroids.row(ci));
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

        let mut sums = Array2::<F>::zeros((k, d));
        let mut total_w = vec![F::zero(); k];
        for i in 0..n {
            let ci = labels[i] as usize;
            total_w[ci] = total_w[ci] + weights[i];
            for j in 0..d {
                sums[[ci, j]] = sums[[ci, j]] + data[[i, j]] * weights[i];
            }
        }
        for ci in 0..k {
            if total_w[ci] > F::epsilon() {
                for j in 0..d {
                    centroids[[ci, j]] = sums[[ci, j]] / total_w[ci];
                }
            }
        }
    }
    labels
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_stream_data() -> Array2<f64> {
        let mut data = Vec::new();
        // Cluster A around (1, 1)
        for i in 0..30 {
            let noise = (i as f64 * 0.073).sin() * 0.3;
            data.push(1.0 + noise);
            data.push(1.0 + noise * 0.7);
        }
        // Cluster B around (5, 5)
        for i in 0..30 {
            let noise = (i as f64 * 0.131).sin() * 0.3;
            data.push(5.0 + noise);
            data.push(5.0 + noise * 0.7);
        }
        Array2::from_shape_vec((60, 2), data).expect("shape failed")
    }

    // -- MicroCluster tests --

    #[test]
    fn test_micro_cluster_from_point() {
        let mc = MicroCluster::<f64>::from_point(&[1.0, 2.0, 3.0], 0);
        assert_eq!(mc.n_points, 1);
        let c = mc.centroid();
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_micro_cluster_absorb() {
        let mut mc = MicroCluster::<f64>::from_point(&[1.0, 2.0], 0);
        mc.absorb(&[3.0, 4.0], 1);
        assert_eq!(mc.n_points, 2);
        let c = mc.centroid();
        assert!((c[0] - 2.0).abs() < 1e-10);
        assert!((c[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_micro_cluster_merge() {
        let mut mc1 = MicroCluster::<f64>::from_point(&[1.0, 1.0], 0);
        let mc2 = MicroCluster::<f64>::from_point(&[3.0, 3.0], 1);
        mc1.merge(&mc2);
        assert_eq!(mc1.n_points, 2);
        let c = mc1.centroid();
        assert!((c[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_micro_cluster_radius() {
        let mut mc = MicroCluster::<f64>::from_point(&[0.0, 0.0], 0);
        mc.absorb(&[2.0, 0.0], 1);
        mc.absorb(&[0.0, 2.0], 2);
        let r = mc.radius();
        // Should be > 0 for spread-out points
        assert!(r > 0.0);
    }

    #[test]
    fn test_micro_cluster_fading() {
        let mut mc = MicroCluster::<f64>::from_point(&[1.0, 2.0], 0);
        mc.absorb(&[1.0, 2.0], 1);
        let w_before = mc.weight;
        mc.apply_fading(0.5, 1.0);
        assert!(mc.weight < w_before);
    }

    // -- CluStream tests --

    #[test]
    fn test_clustream_basic() {
        let data = make_stream_data();
        let config = CluStreamConfig {
            max_micro_clusters: 20,
            n_macro_clusters: 2,
            ..Default::default()
        };
        let mut cs = CluStream::new(config);
        let init_data = data.slice(scirs2_core::ndarray::s![0..20, ..]);
        cs.initialize(init_data).expect("init failed");

        // Process remaining points
        for i in 20..60 {
            cs.process_point(data.row(i).as_slice().unwrap_or(&[]))
                .expect("process failed");
        }

        assert!(cs.n_micro_clusters() > 0);
        let (centroids, labels) = cs.get_macro_clusters().expect("macro failed");
        assert_eq!(labels.len(), cs.n_micro_clusters());
    }

    #[test]
    fn test_clustream_empty_init() {
        let data = Array2::<f64>::zeros((0, 3));
        let config = CluStreamConfig::default();
        let mut cs = CluStream::new(config);
        assert!(cs.initialize(data.view()).is_err());
    }

    #[test]
    fn test_clustream_not_initialized() {
        let cs = CluStream::<f64>::new(CluStreamConfig::default());
        assert!(cs.get_macro_clusters().is_err());
    }

    // -- DenStream tests --

    #[test]
    fn test_denstream_basic() {
        let data = make_stream_data();
        let config = DenStreamConfig {
            epsilon: 2.0,
            min_points: 2,
            lambda: 0.1,
            ..Default::default()
        };
        let mut ds = DenStream::new(config);
        let init_data = data.slice(scirs2_core::ndarray::s![0..30, ..]);
        ds.initialize(init_data).expect("init failed");

        for i in 30..60 {
            ds.process_point(data.row(i).as_slice().unwrap_or(&[]))
                .expect("process failed");
        }

        assert!(ds.n_potential() > 0);
        let result = ds.get_clusters();
        assert!(result.is_ok());
    }

    #[test]
    fn test_denstream_empty_init() {
        let data = Array2::<f64>::zeros((0, 2));
        let config = DenStreamConfig::default();
        let mut ds = DenStream::new(config);
        assert!(ds.initialize(data.view()).is_err());
    }

    // -- StreamKM++ tests --

    #[test]
    fn test_streamkm_basic() {
        let data = make_stream_data();
        let config = StreamKMConfig {
            n_clusters: 2,
            coreset_size: 20,
            kmeans_iterations: 20,
        };
        let mut skm = StreamKMPlusPlus::new(config);
        skm.process_batch(data.view()).expect("batch failed");
        let (centroids, labels) = skm.get_clusters().expect("clusters failed");
        assert_eq!(labels.len(), skm.coreset_size() + skm.buffer.len());
    }

    #[test]
    fn test_streamkm_single_point() {
        let config = StreamKMConfig {
            n_clusters: 1,
            coreset_size: 100,
            ..Default::default()
        };
        let mut skm = StreamKMPlusPlus::<f64>::new(config);
        skm.process_point(&[1.0, 2.0]).expect("failed");
        let (_, labels) = skm.get_clusters().expect("clusters failed");
        assert_eq!(labels.len(), 1);
    }

    // -- Sliding Window tests --

    #[test]
    fn test_sliding_window_basic() {
        let data = make_stream_data();
        let config = SlidingWindowConfig {
            window_size: 50,
            n_clusters: 2,
            kmeans_iterations: 20,
        };
        let mut sw = SlidingWindowClustering::new(config);
        sw.add_batch(data.view());
        assert_eq!(sw.window_len(), 50); // capped at window_size
        let (_, labels) = sw.get_clusters().expect("clusters failed");
        assert_eq!(labels.len(), 50);
    }

    #[test]
    fn test_sliding_window_empty() {
        let sw = SlidingWindowClustering::<f64>::new(SlidingWindowConfig::default());
        assert!(sw.get_clusters().is_err());
    }

    #[test]
    fn test_sliding_window_overflow() {
        let config = SlidingWindowConfig {
            window_size: 5,
            n_clusters: 2,
            ..Default::default()
        };
        let mut sw = SlidingWindowClustering::<f64>::new(config);
        for i in 0..10 {
            sw.add_point(&[i as f64, i as f64 * 2.0]);
        }
        assert_eq!(sw.window_len(), 5);
    }

    // -- Online K-Means tests --

    #[test]
    fn test_online_kmeans_basic() {
        let data = make_stream_data();
        let config = OnlineKMeansConfig {
            n_clusters: 2,
            forgetting_factor: 0.99,
            adaptive_learning: true,
        };
        let mut okm = OnlineKMeans::new(config);
        let init_data = data.slice(scirs2_core::ndarray::s![0..20, ..]);
        okm.initialize(init_data).expect("init failed");

        let labels = okm
            .process_batch(data.slice(scirs2_core::ndarray::s![20..60, ..]))
            .expect("batch failed");
        assert_eq!(labels.len(), 40);

        // Predict should work
        let pred = okm.predict(&[1.0, 1.0]).expect("predict failed");
        assert!(pred >= 0);
    }

    #[test]
    fn test_online_kmeans_not_init() {
        let okm = OnlineKMeans::<f64>::new(OnlineKMeansConfig::default());
        assert!(okm.predict(&[1.0]).is_err());
    }

    #[test]
    fn test_online_kmeans_forgetting() {
        let config = OnlineKMeansConfig {
            n_clusters: 2,
            forgetting_factor: 0.5, // aggressive forgetting
            adaptive_learning: true,
        };
        let mut okm = OnlineKMeans::<f64>::new(config);
        let init = Array2::from_shape_vec((10, 2), (0..20).map(|i| (i as f64) * 0.1).collect())
            .expect("shape failed");
        okm.initialize(init.view()).expect("init failed");

        // Feed many points from a different region
        for _ in 0..50 {
            let _ = okm.process_point(&[10.0, 10.0]);
        }

        // Centroids should have drifted toward (10, 10)
        let centroids = okm.centroids().expect("no centroids");
        let mut any_close = false;
        for ci in 0..centroids.shape()[0] {
            if (centroids[[ci, 0]] - 10.0).abs() < 3.0 {
                any_close = true;
            }
        }
        assert!(any_close, "Expected centroids to drift toward new data");
    }

    // -- Helper function tests --

    #[test]
    fn test_row_dist_sq() {
        let a = Array1::from_vec(vec![1.0, 2.0]);
        let b = Array1::from_vec(vec![4.0, 6.0]);
        let d = row_dist_sq(a.view(), b.view());
        assert!((d - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_dbscan_on_centroids() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2],
        )
        .expect("shape failed");
        let labels = dbscan_on_centroids(&data, 0.5, 2);
        // Points 0-2 should be one cluster, 3-5 another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_simple_kmeans_init() {
        let data = make_stream_data();
        let labels = simple_kmeans_init(data.view(), 2);
        assert_eq!(labels.len(), 60);
        // Should have at least 2 distinct labels
        let unique: std::collections::HashSet<i32> = labels.iter().copied().collect();
        assert!(unique.len() >= 2);
    }

    #[test]
    fn test_clustream_batch() {
        let data = make_stream_data();
        let config = CluStreamConfig {
            max_micro_clusters: 10,
            n_macro_clusters: 2,
            ..Default::default()
        };
        let mut cs = CluStream::new(config);
        cs.initialize(data.slice(scirs2_core::ndarray::s![0..20, ..]))
            .expect("init");
        cs.process_batch(data.slice(scirs2_core::ndarray::s![20..60, ..]))
            .expect("batch");
        assert!(cs.n_micro_clusters() >= 2);
    }

    #[test]
    fn test_streamkm_coreset_reduces() {
        let config = StreamKMConfig {
            n_clusters: 2,
            coreset_size: 10,
            ..Default::default()
        };
        let mut skm = StreamKMPlusPlus::<f64>::new(config);
        // Feed more than coreset_size points
        for i in 0..30 {
            skm.process_point(&[i as f64, (i * 2) as f64])
                .expect("fail");
        }
        // Coreset should have been compressed
        assert!(skm.coreset_size() > 0);
    }
}
