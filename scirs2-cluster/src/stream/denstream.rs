//! DenStream – density-based stream clustering.
//!
//! Implements the DenStream algorithm (Cao et al. 2006) for discovering
//! clusters of arbitrary shape in high-speed data streams with noise
//! handling via potential / outlier micro-cluster pools and an exponential
//! fading function.
//!
//! # Algorithm Overview
//!
//! 1. **Online phase** – incoming points are merged into potential micro-clusters
//!    (p-micro-clusters) or outlier micro-clusters (o-micro-clusters).
//! 2. **Offline phase** – a DBSCAN-like macro-clustering is run over the
//!    p-micro-cluster centroids weighted by their faded weights.
//!
//! # References
//!
//! Cao, F., Ester, M., Qian, W., & Zhou, A. (2006). Density-based clustering
//! over an evolving data stream with noise. *SDM*, 328–339.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Fading function
// ---------------------------------------------------------------------------

/// Exponential fading function f(t) = 2^{-λt}.
///
/// Used to discount the weight of old data points.  Higher λ → faster forgetting.
#[derive(Debug, Clone, Copy)]
pub struct Fading {
    /// Fading factor λ (> 0).
    pub lambda: f64,
}

impl Fading {
    /// Create a new fading function with parameter λ.
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda: lambda.max(1e-12),
        }
    }

    /// Evaluate f(elapsed) = 2^{-λ * elapsed}.
    pub fn evaluate(&self, elapsed: f64) -> f64 {
        (-self.lambda * elapsed * std::f64::consts::LN_2).exp()
    }

    /// Threshold weight below which a micro-cluster should be pruned.
    ///
    /// Derived as the weight of a cluster that has received zero new points
    /// since its creation time `t_c`, evaluated at the pruning check time:
    ///
    /// w_min = f(t_c) = 2^{-λ * T_p}
    ///
    /// where T_p is the pruning period.
    pub fn pruning_threshold(&self, pruning_period: f64) -> f64 {
        self.evaluate(pruning_period)
    }
}

// ---------------------------------------------------------------------------
// CoreMicroCluster / OutlierMicroCluster
// ---------------------------------------------------------------------------

/// Potential (core) micro-cluster.
///
/// Stores an exponentially faded CF-vector and is promoted from an outlier
/// micro-cluster once its weight exceeds the threshold β·μ.
#[derive(Debug, Clone)]
pub struct CoreMicroCluster<F: Float> {
    /// Faded linear sum (dimension d).
    pub ls: Vec<F>,
    /// Faded squared sum (dimension d).
    pub ss: Vec<F>,
    /// Faded weight (number of points, discounted).
    pub weight: F,
    /// Timestamp of last update.
    pub last_update: u64,
    /// Timestamp of creation.
    pub creation_time: u64,
    /// Cluster identifier.
    pub id: usize,
}

impl<F: Float + FromPrimitive + Debug> CoreMicroCluster<F> {
    /// Create a new core micro-cluster from a single point.
    pub fn from_point(id: usize, point: &[F], timestamp: u64) -> Self {
        let d = point.len();
        let mut ls = vec![F::zero(); d];
        let mut ss = vec![F::zero(); d];
        for i in 0..d {
            ls[i] = point[i];
            ss[i] = point[i] * point[i];
        }
        Self {
            ls,
            ss,
            weight: F::one(),
            last_update: timestamp,
            creation_time: timestamp,
            id,
        }
    }

    /// Centroid of this micro-cluster.
    pub fn centroid(&self) -> Vec<F> {
        if self.weight <= F::epsilon() {
            return self.ls.clone();
        }
        self.ls.iter().map(|&v| v / self.weight).collect()
    }

    /// Radius (root-mean-square deviation from centroid).
    pub fn radius(&self) -> F {
        if self.weight <= F::one() {
            return F::zero();
        }
        let d = self.ls.len();
        let mut variance = F::zero();
        for i in 0..d {
            let mean = self.ls[i] / self.weight;
            let mean_sq = self.ss[i] / self.weight;
            let v = mean_sq - mean * mean;
            if v > F::zero() {
                variance = variance + v;
            }
        }
        let d_f = F::from_usize(d).unwrap_or(F::one());
        (variance / d_f).sqrt()
    }

    /// Apply fading decay: multiply CF components by `factor`.
    pub fn apply_fading(&mut self, factor: F) {
        for v in self.ls.iter_mut() {
            *v = *v * factor;
        }
        for v in self.ss.iter_mut() {
            *v = *v * factor;
        }
        self.weight = self.weight * factor;
    }

    /// Absorb a point (after applying fading first externally).
    pub fn absorb(&mut self, point: &[F], timestamp: u64) {
        let d = self.ls.len().min(point.len());
        for i in 0..d {
            self.ls[i] = self.ls[i] + point[i];
            self.ss[i] = self.ss[i] + point[i] * point[i];
        }
        self.weight = self.weight + F::one();
        self.last_update = timestamp;
    }

    /// Squared distance from centroid to a point.
    pub fn distance_sq_to(&self, point: &[F]) -> F {
        let c = self.centroid();
        let d = c.len().min(point.len());
        let mut s = F::zero();
        for i in 0..d {
            let diff = c[i] - point[i];
            s = s + diff * diff;
        }
        s
    }

    /// Merge another core micro-cluster into this one.
    pub fn merge(&mut self, other: &CoreMicroCluster<F>) {
        let d = self.ls.len().min(other.ls.len());
        for i in 0..d {
            self.ls[i] = self.ls[i] + other.ls[i];
            self.ss[i] = self.ss[i] + other.ss[i];
        }
        self.weight = self.weight + other.weight;
        if other.last_update > self.last_update {
            self.last_update = other.last_update;
        }
    }
}

/// Outlier micro-cluster.
///
/// Points that cannot be merged into any p-micro-cluster form outlier
/// micro-clusters.  They are promoted to p-micro-clusters once their
/// weight grows sufficiently, or pruned when it falls too low.
#[derive(Debug, Clone)]
pub struct OutlierMicroCluster<F: Float> {
    /// Inner CF data (reuses CoreMicroCluster storage).
    pub inner: CoreMicroCluster<F>,
}

impl<F: Float + FromPrimitive + Debug> OutlierMicroCluster<F> {
    /// Create from a single point.
    pub fn from_point(id: usize, point: &[F], timestamp: u64) -> Self {
        Self {
            inner: CoreMicroCluster::from_point(id, point, timestamp),
        }
    }

    /// Delegate centroid.
    pub fn centroid(&self) -> Vec<F> {
        self.inner.centroid()
    }

    /// Delegate radius.
    pub fn radius(&self) -> F {
        self.inner.radius()
    }

    /// Delegate distance.
    pub fn distance_sq_to(&self, point: &[F]) -> F {
        self.inner.distance_sq_to(point)
    }

    /// Delegate absorb.
    pub fn absorb(&mut self, point: &[F], timestamp: u64) {
        self.inner.absorb(point, timestamp);
    }

    /// Delegate fading.
    pub fn apply_fading(&mut self, factor: F) {
        self.inner.apply_fading(factor);
    }

    /// Faded weight.
    pub fn weight(&self) -> F {
        self.inner.weight
    }

    /// Promote to a CoreMicroCluster.
    pub fn into_core(self) -> CoreMicroCluster<F> {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// DenStreamConfig
// ---------------------------------------------------------------------------

/// Configuration for the DenStream algorithm.
#[derive(Debug, Clone)]
pub struct DenStreamConfig {
    /// Fading parameter λ (controls how fast old data decays).
    pub lambda: f64,
    /// Radius threshold ε for merging a point into a micro-cluster.
    pub epsilon: f64,
    /// Minimum weight threshold μ for a p-micro-cluster to be considered core.
    pub mu: f64,
    /// Weight factor β: o-micro-cluster is promoted when weight > β*μ.
    pub beta: f64,
    /// DBSCAN neighbourhood radius for macro-clustering.
    pub dbscan_epsilon: f64,
    /// DBSCAN minimum points (by weight) for macro-clustering.
    pub dbscan_min_weight: f64,
    /// Number of points between pruning checks.
    pub pruning_interval: u64,
}

impl Default for DenStreamConfig {
    fn default() -> Self {
        Self {
            lambda: 0.25,
            epsilon: 0.5,
            mu: 1.0,
            beta: 0.2,
            dbscan_epsilon: 1.0,
            dbscan_min_weight: 2.0,
            pruning_interval: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// DenStream
// ---------------------------------------------------------------------------

/// DenStream density-based streaming clustering.
pub struct DenStream<F: Float> {
    config: DenStreamConfig,
    fading: Fading,
    /// Potential (core) micro-clusters.
    p_clusters: Vec<CoreMicroCluster<F>>,
    /// Outlier micro-clusters.
    o_clusters: Vec<OutlierMicroCluster<F>>,
    current_time: u64,
    next_id: usize,
    n_features: usize,
}

impl<F: Float + FromPrimitive + Debug + Clone> DenStream<F> {
    /// Create a new DenStream instance.
    pub fn new(config: DenStreamConfig) -> Self {
        let fading = Fading::new(config.lambda);
        Self {
            config,
            fading,
            p_clusters: Vec::new(),
            o_clusters: Vec::new(),
            current_time: 0,
            next_id: 0,
            n_features: 0,
        }
    }

    /// Number of potential micro-clusters.
    pub fn n_p_clusters(&self) -> usize {
        self.p_clusters.len()
    }

    /// Number of outlier micro-clusters.
    pub fn n_o_clusters(&self) -> usize {
        self.o_clusters.len()
    }

    /// Current logical timestamp.
    pub fn current_time(&self) -> u64 {
        self.current_time
    }

    /// Process a single incoming point.
    pub fn update(&mut self, point: ArrayView1<F>) -> Result<()> {
        let pt: Vec<F> = point.iter().copied().collect();
        let d = pt.len();

        if self.n_features == 0 {
            self.n_features = d;
        } else if d != self.n_features {
            return Err(ClusteringError::InvalidInput(format!(
                "Dimension mismatch: expected {}, got {}",
                self.n_features, d
            )));
        }

        let epsilon = F::from_f64(self.config.epsilon).unwrap_or(F::one());
        let mu_val = F::from_f64(self.config.mu).unwrap_or(F::one());
        let beta_mu = F::from_f64(self.config.beta * self.config.mu).unwrap_or(F::one());

        // Apply fading to all micro-clusters
        let elapsed = 1.0_f64;
        let factor = F::from_f64(self.fading.evaluate(elapsed)).unwrap_or(F::one());
        for pc in self.p_clusters.iter_mut() {
            pc.apply_fading(factor);
        }
        for oc in self.o_clusters.iter_mut() {
            oc.apply_fading(factor);
        }

        // Try to merge into a p-micro-cluster
        if let Some(idx) = self.find_nearest_within(&self.p_cluster_centroids(), &pt, epsilon) {
            // Check radius constraint after hypothetical absorption
            let new_radius = self.hypothetical_radius_p(idx, &pt);
            if new_radius <= epsilon {
                self.p_clusters[idx].absorb(&pt, self.current_time);
                self.current_time += 1;
                self.maybe_prune();
                return Ok(());
            }
        }

        // Try to merge into an o-micro-cluster
        if let Some(idx) = self.find_nearest_within(&self.o_cluster_centroids(), &pt, epsilon) {
            let new_radius = self.hypothetical_radius_o(idx, &pt);
            if new_radius <= epsilon {
                self.o_clusters[idx].absorb(&pt, self.current_time);
                // Check if this outlier should be promoted to potential
                if self.o_clusters[idx].weight() > beta_mu {
                    let oc = self.o_clusters.remove(idx);
                    self.p_clusters.push(oc.into_core());
                }
                self.current_time += 1;
                self.maybe_prune();
                return Ok(());
            }
        }

        // Create a new outlier micro-cluster
        let oc = OutlierMicroCluster::from_point(self.next_id, &pt, self.current_time);
        self.next_id += 1;
        self.o_clusters.push(oc);

        self.current_time += 1;
        self.maybe_prune();
        Ok(())
    }

    /// Process a batch of data points.
    pub fn update_batch(&mut self, data: ArrayView2<F>) -> Result<()> {
        for row in data.rows() {
            self.update(row)?;
        }
        Ok(())
    }

    /// Offline macro-clustering using a DBSCAN-like procedure on p-micro-cluster
    /// centroids weighted by their faded weights.
    ///
    /// Returns cluster labels for each p-micro-cluster (-1 = noise).
    pub fn cluster(&self) -> Result<DenStreamResult<F>> {
        if self.p_clusters.is_empty() {
            return Err(ClusteringError::InvalidState(
                "No potential micro-clusters available for macro-clustering".into(),
            ));
        }
        let labels = self.dbscan_on_p_clusters();
        let n_clusters = labels.iter().filter(|&&l| l >= 0).max().map(|&v| (v + 1) as usize).unwrap_or(0);
        Ok(DenStreamResult {
            labels,
            p_clusters: self.p_clusters.clone(),
            o_clusters: self.o_clusters.clone(),
            n_clusters,
            timestamp: self.current_time,
        })
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn p_cluster_centroids(&self) -> Vec<Vec<F>> {
        self.p_clusters.iter().map(|mc| mc.centroid()).collect()
    }

    fn o_cluster_centroids(&self) -> Vec<Vec<F>> {
        self.o_clusters.iter().map(|mc| mc.centroid()).collect()
    }

    fn find_nearest_within(
        &self,
        centroids: &[Vec<F>],
        point: &[F],
        radius: F,
    ) -> Option<usize> {
        let mut best_idx = None;
        let mut best_dist = radius * radius + F::epsilon();
        for (i, c) in centroids.iter().enumerate() {
            let d = c
                .iter()
                .zip(point.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .fold(F::zero(), |acc, v| acc + v);
            if d < best_dist {
                best_dist = d;
                best_idx = Some(i);
            }
        }
        best_idx
    }

    fn hypothetical_radius_p(&self, idx: usize, point: &[F]) -> F {
        let mc = &self.p_clusters[idx];
        let d = mc.ls.len().min(point.len());
        let new_weight = mc.weight + F::one();
        if new_weight <= F::one() {
            return F::zero();
        }
        let mut variance = F::zero();
        for i in 0..d {
            let new_ls = mc.ls[i] + point[i];
            let new_ss = mc.ss[i] + point[i] * point[i];
            let mean = new_ls / new_weight;
            let mean_sq = new_ss / new_weight;
            let v = mean_sq - mean * mean;
            if v > F::zero() {
                variance = variance + v;
            }
        }
        let d_f = F::from_usize(d).unwrap_or(F::one());
        (variance / d_f).sqrt()
    }

    fn hypothetical_radius_o(&self, idx: usize, point: &[F]) -> F {
        let mc = &self.o_clusters[idx].inner;
        let d = mc.ls.len().min(point.len());
        let new_weight = mc.weight + F::one();
        if new_weight <= F::one() {
            return F::zero();
        }
        let mut variance = F::zero();
        for i in 0..d {
            let new_ls = mc.ls[i] + point[i];
            let new_ss = mc.ss[i] + point[i] * point[i];
            let mean = new_ls / new_weight;
            let mean_sq = new_ss / new_weight;
            let v = mean_sq - mean * mean;
            if v > F::zero() {
                variance = variance + v;
            }
        }
        let d_f = F::from_usize(d).unwrap_or(F::one());
        (variance / d_f).sqrt()
    }

    fn maybe_prune(&mut self) {
        if self.current_time % self.config.pruning_interval != 0 {
            return;
        }
        let tp = self.config.pruning_interval as f64;
        let threshold = self.fading.pruning_threshold(tp);
        let beta_mu_threshold = self.config.beta * self.config.mu;

        // Prune p-clusters with weight < mu (they drop below core threshold)
        let mu_f = F::from_f64(self.config.mu).unwrap_or(F::one());
        self.p_clusters.retain(|pc| pc.weight >= mu_f);

        // Prune o-clusters that cannot possibly grow to beta*mu within pruning period
        let threshold_f = F::from_f64(threshold * beta_mu_threshold).unwrap_or(F::zero());
        self.o_clusters
            .retain(|oc| oc.weight() >= threshold_f);
    }

    /// DBSCAN-like macro-clustering on p-micro-cluster centroids.
    ///
    /// Returns a label vector where -1 indicates noise.
    fn dbscan_on_p_clusters(&self) -> Vec<i64> {
        let n = self.p_clusters.len();
        let mut labels = vec![-1i64; n];
        let mut cluster_id = 0i64;

        let eps = self.config.dbscan_epsilon;
        let eps_sq = eps * eps;
        let min_w = self.config.dbscan_min_weight;

        let centroids: Vec<Vec<f64>> = self
            .p_clusters
            .iter()
            .map(|mc| {
                mc.centroid()
                    .into_iter()
                    .map(|v| {
                        // Safe conversion: use a lossy cast via Display format is overkill;
                        // use the From<F> cast pattern expected by the type system.
                        let s = format!("{:?}", v);
                        s.parse::<f64>().unwrap_or(0.0)
                    })
                    .collect()
            })
            .collect();

        let weights: Vec<f64> = self.p_clusters.iter().map(|mc| {
            let s = format!("{:?}", mc.weight);
            s.parse::<f64>().unwrap_or(1.0)
        }).collect();

        for i in 0..n {
            if labels[i] != -1 {
                continue;
            }
            // Gather ε-neighbourhood by weight
            let mut neighbors: Vec<usize> = (0..n)
                .filter(|&j| {
                    let d: f64 = centroids[i]
                        .iter()
                        .zip(centroids[j].iter())
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum();
                    d <= eps_sq
                })
                .collect();
            let total_w: f64 = neighbors.iter().map(|&j| weights[j]).sum();
            if total_w < min_w {
                // Mark as noise for now (may be absorbed later)
                labels[i] = -1;
                continue;
            }
            // Expand cluster
            labels[i] = cluster_id;
            let mut seed_set = neighbors.clone();
            let mut si = 0;
            while si < seed_set.len() {
                let q = seed_set[si];
                si += 1;
                if labels[q] == -1 {
                    labels[q] = cluster_id;
                }
                if labels[q] != cluster_id {
                    labels[q] = cluster_id;
                }
                let q_neighbors: Vec<usize> = (0..n)
                    .filter(|&j| {
                        let d: f64 = centroids[q]
                            .iter()
                            .zip(centroids[j].iter())
                            .map(|(&a, &b)| (a - b) * (a - b))
                            .sum();
                        d <= eps_sq
                    })
                    .collect();
                let q_total_w: f64 = q_neighbors.iter().map(|&j| weights[j]).sum();
                if q_total_w >= min_w {
                    for &r in &q_neighbors {
                        if labels[r] == -1 {
                            seed_set.push(r);
                        }
                    }
                }
            }
            cluster_id += 1;
        }
        labels
    }
}

// ---------------------------------------------------------------------------
// DenStreamResult
// ---------------------------------------------------------------------------

/// Result of DenStream macro-clustering.
#[derive(Debug, Clone)]
pub struct DenStreamResult<F: Float> {
    /// Cluster label for each p-micro-cluster (-1 = noise).
    pub labels: Vec<i64>,
    /// Potential micro-clusters used for macro-clustering.
    pub p_clusters: Vec<CoreMicroCluster<F>>,
    /// Outlier micro-clusters remaining after pruning.
    pub o_clusters: Vec<OutlierMicroCluster<F>>,
    /// Number of clusters (excluding noise).
    pub n_clusters: usize,
    /// Timestamp at which the result was produced.
    pub timestamp: u64,
}

impl<F: Float + FromPrimitive + Debug + Clone> DenStreamResult<F> {
    /// Assign a new point to the nearest cluster (by nearest p-micro-cluster).
    ///
    /// Returns -1 if no clusters were found.
    pub fn predict(&self, point: &[f64]) -> i64 {
        if self.p_clusters.is_empty() || self.n_clusters == 0 {
            return -1;
        }
        let mut best_label = -1i64;
        let mut best_dist = f64::MAX;
        for (i, mc) in self.p_clusters.iter().enumerate() {
            if self.labels[i] < 0 {
                continue;
            }
            let c: Vec<f64> = mc.centroid().into_iter().map(|v| {
                let s = format!("{:?}", v);
                s.parse::<f64>().unwrap_or(0.0)
            }).collect();
            let d: f64 = c.iter().zip(point.iter())
                .map(|(&a, &b)| (a - b) * (a - b)).sum();
            if d < best_dist {
                best_dist = d;
                best_label = self.labels[i];
            }
        }
        best_label
    }

    /// Centroids of the discovered macro-clusters (one per cluster id).
    pub fn cluster_centroids(&self) -> Vec<Vec<f64>> {
        if self.n_clusters == 0 {
            return Vec::new();
        }
        let d = self.p_clusters.first().map(|mc| mc.ls.len()).unwrap_or(0);
        let mut weighted_sum = vec![vec![0f64; d]; self.n_clusters];
        let mut weight_sum = vec![0f64; self.n_clusters];
        for (i, mc) in self.p_clusters.iter().enumerate() {
            let label = self.labels[i];
            if label < 0 {
                continue;
            }
            let label = label as usize;
            if label >= self.n_clusters {
                continue;
            }
            let w: f64 = {
                let s = format!("{:?}", mc.weight);
                s.parse::<f64>().unwrap_or(1.0)
            };
            let c: Vec<f64> = mc.centroid().into_iter().map(|v| {
                let s = format!("{:?}", v);
                s.parse::<f64>().unwrap_or(0.0)
            }).collect();
            weight_sum[label] += w;
            for k in 0..d.min(c.len()) {
                weighted_sum[label][k] += w * c[k];
            }
        }
        weighted_sum
            .into_iter()
            .zip(weight_sum.iter())
            .map(|(mut s, &w)| {
                if w > 0.0 {
                    for v in s.iter_mut() {
                        *v /= w;
                    }
                }
                s
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn two_cluster_stream() -> Array2<f64> {
        let mut rows = Vec::new();
        for i in 0..30 {
            let offset = if i % 2 == 0 { 0.0 } else { 5.0 };
            rows.extend_from_slice(&[offset + 0.1 * i as f64, offset + 0.1 * i as f64]);
        }
        Array2::from_shape_vec((30, 2), rows).expect("shape ok")
    }

    #[test]
    fn test_core_mc_absorb() {
        let mut mc: CoreMicroCluster<f64> = CoreMicroCluster::from_point(0, &[1.0, 1.0], 0);
        mc.absorb(&[3.0, 3.0], 1);
        assert_eq!(mc.weight as u64, 2);
        let c = mc.centroid();
        assert!((c[0] - 2.0).abs() < 1e-9, "centroid[0] = {}", c[0]);
    }

    #[test]
    fn test_fading() {
        let f = Fading::new(0.25);
        let v = f.evaluate(1.0);
        assert!(v > 0.0 && v < 1.0);
    }

    #[test]
    fn test_denstream_update() {
        let mut ds: DenStream<f64> = DenStream::new(DenStreamConfig {
            lambda: 0.25,
            epsilon: 2.0,
            mu: 1.0,
            beta: 0.5,
            dbscan_epsilon: 3.0,
            dbscan_min_weight: 1.0,
            pruning_interval: 50,
        });
        let data = two_cluster_stream();
        ds.update_batch(data.view()).expect("update ok");
        // Some potential clusters should have formed
        assert!(ds.n_p_clusters() + ds.n_o_clusters() > 0);
    }

    #[test]
    fn test_denstream_cluster() {
        let mut ds: DenStream<f64> = DenStream::new(DenStreamConfig {
            lambda: 0.1,
            epsilon: 2.0,
            mu: 1.0,
            beta: 0.5,
            dbscan_epsilon: 4.0,
            dbscan_min_weight: 1.0,
            pruning_interval: 100,
        });
        let data = two_cluster_stream();
        ds.update_batch(data.view()).expect("ok");
        if ds.n_p_clusters() > 0 {
            let result = ds.cluster().expect("cluster ok");
            assert_eq!(result.labels.len(), ds.n_p_clusters());
        }
    }
}
