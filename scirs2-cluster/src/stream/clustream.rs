//! CluStream online clustering algorithm.
//!
//! Implements the CluStream framework (Aggarwal et al. 2003) for clustering
//! high-speed data streams.  The algorithm maintains a set of compact
//! *micro-clusters* (CF-vectors augmented with timestamps) that are updated
//! online.  Periodically a *macro-clustering* step (weighted k-means on the
//! micro-cluster centroids) yields the final cluster labels.
//!
//! # References
//!
//! Aggarwal, C. C., Han, J., Wang, J., & Yu, P. S. (2003). A framework for
//! clustering evolving data streams. *VLDB*, 81–92.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// MicroCluster – compact CF summary
// ---------------------------------------------------------------------------

/// Compact CF (Cluster Feature) summary for a micro-cluster in CluStream.
///
/// A CluStream micro-cluster stores:
/// - `n`   – number of data points absorbed
/// - `ls`  – linear sum of data vectors
/// - `ss`  – squared sum of data vectors (element-wise)
/// - `lst` – linear sum of timestamps
/// - `sst` – squared sum of timestamps
#[derive(Debug, Clone)]
pub struct MicroCluster<F: Float> {
    /// Number of points.
    pub n: u64,
    /// Linear sum of points (dimension d).
    pub ls: Vec<F>,
    /// Squared sum of points (dimension d).
    pub ss: Vec<F>,
    /// Linear sum of timestamps.
    pub lst: F,
    /// Squared sum of timestamps.
    pub sst: F,
    /// Cluster identifier.
    pub id: usize,
}

impl<F: Float + FromPrimitive + Debug> MicroCluster<F> {
    /// Create a new micro-cluster seeded from a single point at a given timestamp.
    pub fn from_point(id: usize, point: &[F], timestamp: u64) -> Self {
        let d = point.len();
        let mut ls = vec![F::zero(); d];
        let mut ss = vec![F::zero(); d];
        for i in 0..d {
            ls[i] = point[i];
            ss[i] = point[i] * point[i];
        }
        let t = F::from_u64(timestamp).unwrap_or(F::zero());
        Self {
            n: 1,
            ls,
            ss,
            lst: t,
            sst: t * t,
            id,
        }
    }

    /// Centroid of this micro-cluster.
    pub fn centroid(&self) -> Vec<F> {
        let w = F::from_u64(self.n).unwrap_or(F::one());
        self.ls.iter().map(|&v| v / w).collect()
    }

    /// Spatial radius (root-mean-square deviation from centroid).
    pub fn radius(&self) -> F {
        if self.n <= 1 {
            return F::zero();
        }
        let n_f = F::from_u64(self.n).unwrap_or(F::one());
        let d = self.ls.len();
        let mut variance = F::zero();
        for i in 0..d {
            let mean = self.ls[i] / n_f;
            let mean_sq = self.ss[i] / n_f;
            let v = mean_sq - mean * mean;
            if v > F::zero() {
                variance = variance + v;
            }
        }
        let d_f = F::from_usize(d).unwrap_or(F::one());
        (variance / d_f).sqrt()
    }

    /// Mean timestamp.
    pub fn mean_time(&self) -> F {
        let n_f = F::from_u64(self.n).unwrap_or(F::one());
        self.lst / n_f
    }

    /// Standard deviation of timestamps (temporal spread).
    pub fn time_std(&self) -> F {
        if self.n <= 1 {
            return F::zero();
        }
        let n_f = F::from_u64(self.n).unwrap_or(F::one());
        let mean_t = self.lst / n_f;
        let mean_t2 = self.sst / n_f;
        let v = mean_t2 - mean_t * mean_t;
        if v > F::zero() { v.sqrt() } else { F::zero() }
    }

    /// Absorb a new point into this micro-cluster.
    pub fn absorb(&mut self, point: &[F], timestamp: u64) {
        let d = self.ls.len().min(point.len());
        for i in 0..d {
            self.ls[i] = self.ls[i] + point[i];
            self.ss[i] = self.ss[i] + point[i] * point[i];
        }
        let t = F::from_u64(timestamp).unwrap_or(F::zero());
        self.lst = self.lst + t;
        self.sst = self.sst + t * t;
        self.n += 1;
    }

    /// Squared Euclidean distance from the centroid of this micro-cluster to `point`.
    pub fn distance_sq_to(&self, point: &[F]) -> F {
        let centroid = self.centroid();
        let d = centroid.len().min(point.len());
        let mut s = F::zero();
        for i in 0..d {
            let diff = centroid[i] - point[i];
            s = s + diff * diff;
        }
        s
    }

    /// Merge another micro-cluster into this one (CluStream additive property).
    pub fn merge(&mut self, other: &MicroCluster<F>) {
        let d = self.ls.len().min(other.ls.len());
        for i in 0..d {
            self.ls[i] = self.ls[i] + other.ls[i];
            self.ss[i] = self.ss[i] + other.ss[i];
        }
        self.lst = self.lst + other.lst;
        self.sst = self.sst + other.sst;
        self.n += other.n;
    }
}

// ---------------------------------------------------------------------------
// PyramidTimeWindow – temporal snapshots at multiple granularities
// ---------------------------------------------------------------------------

/// A snapshot of the micro-cluster set at a particular time.
#[derive(Debug, Clone)]
pub struct Snapshot<F: Float> {
    /// Timestamp at which the snapshot was taken.
    pub timestamp: u64,
    /// Copy of the micro-clusters at snapshot time.
    pub micro_clusters: Vec<MicroCluster<F>>,
}

/// Pyramid time window: maintains snapshots at geometrically increasing
/// time intervals (orders 1 … max_order), keeping at most `alpha` snapshots
/// per order level.
///
/// This allows querying the state of the stream at any time horizon within
/// the retention range.
#[derive(Debug, Clone)]
pub struct PyramidTimeWindow<F: Float> {
    /// Base of the geometric progression (default 2).
    alpha: usize,
    /// Maximum number of orders.
    max_order: usize,
    /// Snapshots keyed by order level.
    snapshots: Vec<Vec<Snapshot<F>>>,
}

impl<F: Float + FromPrimitive + Debug + Clone> PyramidTimeWindow<F> {
    /// Create a new pyramid time window.
    ///
    /// * `alpha`     – number of snapshots to keep per order (≥ 2).
    /// * `max_order` – maximum time order (determines retention horizon).
    pub fn new(alpha: usize, max_order: usize) -> Self {
        let alpha = alpha.max(2);
        Self {
            alpha,
            max_order,
            snapshots: vec![Vec::new(); max_order + 1],
        }
    }

    /// Insert a snapshot at the given order level, pruning old entries.
    pub fn insert(&mut self, order: usize, snapshot: Snapshot<F>) {
        let order = order.min(self.max_order);
        self.snapshots[order].push(snapshot);
        // Keep only the most recent `alpha` snapshots for this order
        let alpha = self.alpha;
        if self.snapshots[order].len() > alpha {
            let excess = self.snapshots[order].len() - alpha;
            self.snapshots[order].drain(..excess);
        }
    }

    /// Retrieve the snapshot closest to `horizon` timestamps in the past
    /// from `current_time`.
    pub fn get_snapshot_for_horizon(
        &self,
        current_time: u64,
        horizon: u64,
    ) -> Option<&Snapshot<F>> {
        let target = current_time.saturating_sub(horizon);
        // Search all order levels for the snapshot with timestamp nearest to `target`
        let mut best: Option<&Snapshot<F>> = None;
        let mut best_diff = u64::MAX;
        for level in &self.snapshots {
            for snap in level {
                let diff = if snap.timestamp >= target {
                    snap.timestamp - target
                } else {
                    target - snap.timestamp
                };
                if diff < best_diff {
                    best_diff = diff;
                    best = Some(snap);
                }
            }
        }
        best
    }

    /// Number of snapshots stored across all levels.
    pub fn total_snapshots(&self) -> usize {
        self.snapshots.iter().map(|v| v.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// MacroKMeans – offline macro-clustering on micro-cluster centroids
// ---------------------------------------------------------------------------

/// Run weighted k-means on micro-cluster centroids to produce macro-clusters.
///
/// Weights are the point counts of each micro-cluster.
#[derive(Debug, Clone)]
pub struct MacroKMeans {
    /// Number of macro-clusters.
    pub k: usize,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence tolerance (centroid shift).
    pub tol: f64,
}

impl Default for MacroKMeans {
    fn default() -> Self {
        Self {
            k: 5,
            max_iter: 100,
            tol: 1e-6,
        }
    }
}

impl MacroKMeans {
    /// Create a new MacroKMeans instance.
    pub fn new(k: usize, max_iter: usize, tol: f64) -> Self {
        Self { k, max_iter, tol }
    }

    /// Fit macro-clusters to the micro-cluster centroids.
    ///
    /// Returns `(macro_centroids, micro_to_macro_labels)`.
    pub fn fit<F>(&self, micro_clusters: &[MicroCluster<F>]) -> Result<(Array2<f64>, Vec<usize>)>
    where
        F: Float + FromPrimitive + Debug,
        f64: From<F>,
    {
        if micro_clusters.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "No micro-clusters to macro-cluster".into(),
            ));
        }
        let n = micro_clusters.len();
        let d = micro_clusters[0].ls.len();
        let k = self.k.min(n);

        // Collect centroids and weights
        let centroids_raw: Vec<Vec<f64>> = micro_clusters
            .iter()
            .map(|mc| mc.centroid().into_iter().map(f64::from).collect())
            .collect();
        let weights: Vec<f64> = micro_clusters
            .iter()
            .map(|mc| mc.n as f64)
            .collect();

        // Initialize macro-centroids using first k micro-cluster centroids
        let mut macro_cents: Vec<Vec<f64>> = centroids_raw[..k].to_vec();

        let mut labels = vec![0usize; n];

        for _ in 0..self.max_iter {
            // Assignment step
            for (i, cent) in centroids_raw.iter().enumerate() {
                let mut best = 0usize;
                let mut best_dist = f64::MAX;
                for (j, mc) in macro_cents.iter().enumerate() {
                    let dist: f64 = cent
                        .iter()
                        .zip(mc.iter())
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best = j;
                    }
                }
                labels[i] = best;
            }

            // Update step (weighted)
            let mut new_cents = vec![vec![0f64; d]; k];
            let mut wsum = vec![0f64; k];
            for (i, cent) in centroids_raw.iter().enumerate() {
                let j = labels[i];
                let w = weights[i];
                wsum[j] += w;
                for dim in 0..d {
                    new_cents[j][dim] += w * cent[dim];
                }
            }
            for j in 0..k {
                if wsum[j] > 0.0 {
                    for dim in 0..d {
                        new_cents[j][dim] /= wsum[j];
                    }
                }
            }

            // Check convergence
            let shift: f64 = new_cents
                .iter()
                .zip(macro_cents.iter())
                .map(|(a, b)| {
                    a.iter()
                        .zip(b.iter())
                        .map(|(&x, &y)| (x - y) * (x - y))
                        .sum::<f64>()
                        .sqrt()
                })
                .sum();

            macro_cents = new_cents;
            if shift < self.tol {
                break;
            }
        }

        // Build output Array2
        let flat: Vec<f64> = macro_cents.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((k, d), flat)
            .map_err(|e| ClusteringError::ComputationError(e.to_string()))?;

        Ok((arr, labels))
    }
}

// ---------------------------------------------------------------------------
// CluStreamConfig
// ---------------------------------------------------------------------------

/// Configuration for the CluStream algorithm.
#[derive(Debug, Clone)]
pub struct CluStreamConfig {
    /// Maximum number of micro-clusters to maintain online.
    pub max_micro_clusters: usize,
    /// Number of macro-clusters for the offline phase.
    pub n_macro_clusters: usize,
    /// Radius factor: maximum allowable radius relative to mean radius for absorption.
    pub radius_factor: f64,
    /// Number of snapshots per pyramid order (alpha parameter).
    pub pyramid_alpha: usize,
    /// Maximum pyramid order.
    pub pyramid_max_order: usize,
    /// Snapshot frequency (every N points).
    pub snapshot_frequency: u64,
    /// Maximum number of macro-clustering iterations.
    pub macro_max_iter: usize,
}

impl Default for CluStreamConfig {
    fn default() -> Self {
        Self {
            max_micro_clusters: 100,
            n_macro_clusters: 5,
            radius_factor: 2.0,
            pyramid_alpha: 2,
            pyramid_max_order: 5,
            snapshot_frequency: 100,
            macro_max_iter: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// CluStream algorithm
// ---------------------------------------------------------------------------

/// CluStream online clustering algorithm.
///
/// Maintains a fixed-size pool of micro-clusters that summarise the
/// high-speed data stream.  Each incoming point is either:
///
/// 1. Absorbed into the closest micro-cluster whose radius remains within
///    `radius_factor * mean_radius`.
/// 2. Added as a new micro-cluster (evicting the oldest or merging the
///    two closest existing ones if the pool is full).
///
/// A pyramid of temporal snapshots is maintained.  When `cluster()` is
/// called the offline `MacroKMeans` phase produces the final assignments.
pub struct CluStream<F: Float> {
    config: CluStreamConfig,
    micro_clusters: Vec<MicroCluster<F>>,
    current_time: u64,
    next_id: usize,
    n_features: usize,
    pyramid: PyramidTimeWindow<F>,
}

impl<F: Float + FromPrimitive + Debug + Clone> CluStream<F> {
    /// Create a new CluStream instance.
    pub fn new(config: CluStreamConfig) -> Self {
        let pyramid = PyramidTimeWindow::new(config.pyramid_alpha, config.pyramid_max_order);
        Self {
            config,
            micro_clusters: Vec::new(),
            current_time: 0,
            next_id: 0,
            n_features: 0,
            pyramid,
        }
    }

    /// Number of micro-clusters currently maintained.
    pub fn n_micro_clusters(&self) -> usize {
        self.micro_clusters.len()
    }

    /// Current timestamp.
    pub fn current_time(&self) -> u64 {
        self.current_time
    }

    /// Process a single data point.
    pub fn update(&mut self, point: ArrayView1<F>) -> Result<()> {
        let point_slice: Vec<F> = point.iter().copied().collect();
        let d = point_slice.len();

        // Dimension check / first-point initialisation
        if self.micro_clusters.is_empty() {
            self.n_features = d;
            let mc = MicroCluster::from_point(self.next_id, &point_slice, self.current_time);
            self.next_id += 1;
            self.micro_clusters.push(mc);
            self.current_time += 1;
            self.maybe_snapshot();
            return Ok(());
        }
        if d != self.n_features {
            return Err(ClusteringError::InvalidInput(format!(
                "Dimension mismatch: expected {}, got {}",
                self.n_features, d
            )));
        }

        // Find closest micro-cluster
        let (closest_idx, closest_dist_sq) = self.find_closest(&point_slice);

        // Compute threshold radius
        let threshold = self.compute_threshold(closest_idx);

        if closest_dist_sq.sqrt() <= threshold {
            // Absorb into closest micro-cluster
            self.micro_clusters[closest_idx].absorb(&point_slice, self.current_time);
        } else if self.micro_clusters.len() < self.config.max_micro_clusters {
            // Room to create a new micro-cluster
            let mc =
                MicroCluster::from_point(self.next_id, &point_slice, self.current_time);
            self.next_id += 1;
            self.micro_clusters.push(mc);
        } else {
            // Pool full: merge two closest micro-clusters and replace with new one
            let (i, j) = self.find_closest_pair();
            // Merge j into i, then replace i with the new point
            let other = self.micro_clusters[j].clone();
            self.micro_clusters[i].merge(&other);
            self.micro_clusters.remove(j);
            let mc =
                MicroCluster::from_point(self.next_id, &point_slice, self.current_time);
            self.next_id += 1;
            self.micro_clusters.push(mc);
        }

        self.current_time += 1;
        self.maybe_snapshot();
        Ok(())
    }

    /// Process a batch of data points.
    pub fn update_batch(&mut self, data: ArrayView2<F>) -> Result<()> {
        for row in data.rows() {
            self.update(row)?;
        }
        Ok(())
    }

    /// Run the offline macro-clustering phase and return the result.
    ///
    /// The returned `CluStreamResult` contains macro-cluster labels for each
    /// micro-cluster and the macro-cluster centroids.
    pub fn cluster(&self) -> Result<CluStreamResult<F>> where f64: From<F> {
        if self.micro_clusters.is_empty() {
            return Err(ClusteringError::InvalidState(
                "CluStream has not been initialized with data".into(),
            ));
        }
        let macro_km = MacroKMeans::new(
            self.config.n_macro_clusters,
            self.config.macro_max_iter,
            1e-6,
        );
        let (macro_centroids, micro_labels) = macro_km.fit(&self.micro_clusters)?;
        Ok(CluStreamResult {
            macro_centroids,
            micro_to_macro: micro_labels,
            micro_clusters: self.micro_clusters.clone(),
            n_micro_clusters: self.micro_clusters.len(),
            n_macro_clusters: self.config.n_macro_clusters.min(self.micro_clusters.len()),
            timestamp: self.current_time,
        })
    }

    /// Query the state of the stream within a temporal horizon from now.
    pub fn cluster_in_horizon(
        &self,
        horizon: u64,
    ) -> Result<Option<CluStreamResult<F>>> where f64: From<F> {
        let snap = self
            .pyramid
            .get_snapshot_for_horizon(self.current_time, horizon);
        let snap = match snap {
            Some(s) => s,
            None => return Ok(None),
        };
        if snap.micro_clusters.is_empty() {
            return Ok(None);
        }
        let macro_km = MacroKMeans::new(
            self.config.n_macro_clusters,
            self.config.macro_max_iter,
            1e-6,
        );
        let (macro_centroids, micro_labels) = macro_km.fit(&snap.micro_clusters)?;
        let n_macro = self
            .config
            .n_macro_clusters
            .min(snap.micro_clusters.len());
        Ok(Some(CluStreamResult {
            macro_centroids,
            micro_to_macro: micro_labels,
            n_micro_clusters: snap.micro_clusters.len(),
            n_macro_clusters: n_macro,
            micro_clusters: snap.micro_clusters.clone(),
            timestamp: snap.timestamp,
        }))
    }

    /// Return a reference to the current micro-clusters.
    pub fn micro_clusters(&self) -> &[MicroCluster<F>] {
        &self.micro_clusters
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn find_closest(&self, point: &[F]) -> (usize, F) {
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

    fn compute_threshold(&self, idx: usize) -> F {
        let r = self.micro_clusters[idx].radius();
        // Fall back to mean radius when the local radius is zero or very small
        if r > F::epsilon() {
            let factor = F::from_f64(self.config.radius_factor).unwrap_or(F::one() + F::one());
            return r * factor;
        }
        // Use mean radius across all non-empty micro-clusters
        let non_zero: Vec<F> = self
            .micro_clusters
            .iter()
            .map(|mc| mc.radius())
            .filter(|&r| r > F::epsilon())
            .collect();
        if non_zero.is_empty() {
            return F::from_f64(1e-10).unwrap_or(F::zero());
        }
        let n_f = F::from_usize(non_zero.len()).unwrap_or(F::one());
        let mean_r: F = non_zero.iter().copied().fold(F::zero(), |a, b| a + b) / n_f;
        let factor = F::from_f64(self.config.radius_factor).unwrap_or(F::one() + F::one());
        mean_r * factor
    }

    fn find_closest_pair(&self) -> (usize, usize) {
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
        (best_i, best_j)
    }

    fn maybe_snapshot(&mut self) {
        if self.current_time % self.config.snapshot_frequency == 0 {
            let snap = Snapshot {
                timestamp: self.current_time,
                micro_clusters: self.micro_clusters.clone(),
            };
            // Decide pyramid order based on time
            let order = self.pyramid_order_for(self.current_time);
            self.pyramid.insert(order, snap);
        }
    }

    fn pyramid_order_for(&self, t: u64) -> usize {
        if t == 0 {
            return 0;
        }
        // Order = floor(log_{alpha}(t mod alpha^{max_order+1}))
        let alpha = self.config.pyramid_alpha as u64;
        let mut order = 0usize;
        let mut level = alpha;
        while order < self.config.pyramid_max_order {
            if t % level != 0 {
                break;
            }
            order += 1;
            level = level.saturating_mul(alpha);
        }
        order
    }
}

// ---------------------------------------------------------------------------
// CluStreamResult
// ---------------------------------------------------------------------------

/// Result of the CluStream offline macro-clustering phase.
#[derive(Debug, Clone)]
pub struct CluStreamResult<F: Float> {
    /// Macro-cluster centroids (shape: `[k, d]`).
    pub macro_centroids: Array2<f64>,
    /// Maps each micro-cluster index to a macro-cluster label.
    pub micro_to_macro: Vec<usize>,
    /// Snapshot of the micro-clusters used for macro-clustering.
    pub micro_clusters: Vec<MicroCluster<F>>,
    /// Number of micro-clusters.
    pub n_micro_clusters: usize,
    /// Number of macro-clusters.
    pub n_macro_clusters: usize,
    /// Timestamp at which this result was produced.
    pub timestamp: u64,
}

impl<F: Float + FromPrimitive + Debug + Clone> CluStreamResult<F> {
    /// Assign a new (unseen) point to the nearest macro-cluster.
    pub fn predict(&self, point: &[f64]) -> usize {
        let d = self.macro_centroids.ncols();
        let mut best = 0usize;
        let mut best_dist = f64::MAX;
        for (j, row) in self.macro_centroids.rows().into_iter().enumerate() {
            let dist: f64 = row
                .iter()
                .zip(point.iter())
                .take(d)
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best = j;
            }
        }
        best
    }

    /// Return the labels for all points that were absorbed by a particular
    /// micro-cluster (identified by `micro_idx`).
    pub fn macro_label_for_micro(&self, micro_idx: usize) -> Option<usize> {
        self.micro_to_macro.get(micro_idx).copied()
    }

    /// Weighted inertia: sum over micro-clusters of
    /// `n_i * ||centroid_i - macro_centroid_{label_i}||^2`.
    pub fn inertia(&self) -> f64
    where
        f64: From<F>,
    {
        let mut total = 0f64;
        let d = self.macro_centroids.ncols();
        for (i, mc) in self.micro_clusters.iter().enumerate() {
            let label = self.micro_to_macro[i];
            let macro_cent = self.macro_centroids.row(label);
            let centroid: Vec<f64> = mc.centroid().into_iter().map(f64::from).collect();
            let dist_sq: f64 = centroid
                .iter()
                .zip(macro_cent.iter())
                .take(d)
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            total += mc.n as f64 * dist_sq;
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_stream_data() -> Array2<f64> {
        // Two clusters: one near (0,0), one near (10,10)
        let mut rows = Vec::new();
        for i in 0..20 {
            let offset = (i % 2) as f64 * 10.0;
            rows.extend_from_slice(&[offset + (i as f64 * 0.1), offset + (i as f64 * 0.1)]);
        }
        Array2::from_shape_vec((20, 2), rows).expect("shape ok")
    }

    #[test]
    fn test_micro_cluster_absorb() {
        let mut mc: MicroCluster<f64> = MicroCluster::from_point(0, &[1.0, 2.0], 0);
        mc.absorb(&[3.0, 4.0], 1);
        assert_eq!(mc.n, 2);
        let c = mc.centroid();
        assert!((c[0] - 2.0).abs() < 1e-10);
        assert!((c[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_clustream_update_and_cluster() {
        let mut cs: CluStream<f64> = CluStream::new(CluStreamConfig {
            max_micro_clusters: 10,
            n_macro_clusters: 2,
            ..Default::default()
        });
        let data = make_stream_data();
        cs.update_batch(data.view()).expect("update batch ok");
        assert!(cs.n_micro_clusters() > 0);
        let result = cs.cluster().expect("cluster ok");
        assert_eq!(result.n_macro_clusters, 2);
        assert!(!result.micro_to_macro.is_empty());
    }

    #[test]
    fn test_pyramid_snapshot() {
        let mut cs: CluStream<f64> = CluStream::new(CluStreamConfig {
            max_micro_clusters: 10,
            n_macro_clusters: 2,
            snapshot_frequency: 5,
            ..Default::default()
        });
        let data = make_stream_data();
        cs.update_batch(data.view()).expect("ok");
        // At least some snapshots should have been stored
        assert!(cs.pyramid.total_snapshots() > 0);
    }

    #[test]
    fn test_macro_kmeans_basic() {
        let mut mcs: Vec<MicroCluster<f64>> = Vec::new();
        for i in 0..6 {
            let v = if i < 3 { 0.0 } else { 10.0 };
            let mut mc = MicroCluster::from_point(i, &[v, v], 0);
            for _ in 0..4 {
                mc.absorb(&[v + 0.1, v + 0.1], 1);
            }
            mcs.push(mc);
        }
        let km = MacroKMeans::new(2, 100, 1e-9);
        let (cents, labels) = km.fit(&mcs).expect("fit ok");
        assert_eq!(cents.nrows(), 2);
        assert_eq!(labels.len(), 6);
    }
}
