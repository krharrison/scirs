//! Distributed Mini-Batch K-means (Sculley 2010)
//!
//! Implements distributed mini-batch k-means with k-means++ initialization,
//! simulated parallel workers via `std::thread::scope`, and AllReduce aggregation.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::error::ClusteringError;

/// Configuration for distributed mini-batch k-means.
#[derive(Debug, Clone)]
pub struct DistributedKmeansConfig {
    /// Number of cluster centroids.
    pub n_clusters: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance (centroid shift).
    pub tol: f64,
    /// Number of simulated parallel workers.
    pub n_workers: usize,
    /// Mini-batch size drawn from each worker shard per iteration.
    pub mini_batch_size: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for DistributedKmeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            max_iter: 100,
            tol: 1e-4,
            n_workers: 4,
            mini_batch_size: 256,
            seed: 42,
        }
    }
}

/// A data shard assigned to one worker.
#[derive(Debug, Clone)]
pub struct LocalShard {
    /// Rows of data owned by this shard.
    pub data: Vec<Vec<f64>>,
    /// Zero-based worker identifier.
    pub worker_id: usize,
}

/// Aggregated statistics from a single worker pass.
#[derive(Debug, Clone)]
pub struct ClusterUpdate {
    /// Per-cluster sum of assigned point coordinates (n_clusters × n_dims).
    pub centroid_sums: Vec<Vec<f64>>,
    /// Per-cluster count of assigned points.
    pub centroid_counts: Vec<usize>,
}

impl ClusterUpdate {
    fn new(n_clusters: usize, n_dims: usize) -> Self {
        Self {
            centroid_sums: vec![vec![0.0; n_dims]; n_clusters],
            centroid_counts: vec![0; n_clusters],
        }
    }

    /// Merge another update into this one (AllReduce sum).
    fn merge(&mut self, other: &ClusterUpdate) {
        for k in 0..self.centroid_sums.len() {
            for d in 0..self.centroid_sums[k].len() {
                self.centroid_sums[k][d] += other.centroid_sums[k][d];
            }
            self.centroid_counts[k] += other.centroid_counts[k];
        }
    }
}

/// Result returned by [`DistributedKmeans::fit`].
#[derive(Debug, Clone)]
pub struct KmeansResult {
    /// Final centroid positions.
    pub centroids: Vec<Vec<f64>>,
    /// Cluster label for each input point.
    pub labels: Vec<usize>,
    /// Sum of squared distances to assigned centroids.
    pub inertia: f64,
    /// Actual number of iterations executed.
    pub n_iter: usize,
}

/// Distributed mini-batch k-means clusterer.
pub struct DistributedKmeans {
    config: DistributedKmeansConfig,
}

impl DistributedKmeans {
    /// Create a new `DistributedKmeans` with the given configuration.
    pub fn new(config: DistributedKmeansConfig) -> Self {
        Self { config }
    }

    /// Fit the model on `data` (rows are points, columns are features).
    pub fn fit(&mut self, data: &[Vec<f64>]) -> Result<KmeansResult, ClusteringError> {
        let n = data.len();
        if n == 0 {
            return Err(ClusteringError::InvalidInput("Empty dataset".into()));
        }
        let k = self.config.n_clusters;
        let n_dims = data[0].len();
        if n_dims == 0 {
            return Err(ClusteringError::InvalidInput(
                "Zero-dimensional data".into(),
            ));
        }
        if k > n {
            return Err(ClusteringError::InvalidInput(
                "n_clusters > n_samples".into(),
            ));
        }

        // --- k-means++ initialisation ---
        let mut centroids = kmeans_plus_plus_init(data, k, self.config.seed)?;

        // --- Partition data into worker shards ---
        let shards = partition_into_shards(data, self.config.n_workers);

        let mut rng_state = self.config.seed;
        let mut n_iter = 0usize;

        for iter in 0..self.config.max_iter {
            n_iter = iter + 1;

            // --- Parallel map-reduce across shards ---
            let update = self.parallel_map_reduce(&shards, &centroids, &mut rng_state)?;

            // --- Centroid update ---
            let mut new_centroids = centroids.clone();
            for k_idx in 0..k {
                if update.centroid_counts[k_idx] > 0 {
                    for d in 0..n_dims {
                        new_centroids[k_idx][d] =
                            update.centroid_sums[k_idx][d] / update.centroid_counts[k_idx] as f64;
                    }
                }
                // If a cluster is empty keep its old centroid (standard practice)
            }

            // --- Convergence check ---
            let shift = centroid_shift(&centroids, &new_centroids);
            centroids = new_centroids;

            if shift < self.config.tol {
                n_iter = iter + 1;
                break;
            }
        }

        // --- Final full assignment for labels and inertia ---
        let mut labels = vec![0usize; n];
        let mut inertia = 0.0f64;
        for (i, point) in data.iter().enumerate() {
            let (nearest, dist2) = nearest_centroid(point, &centroids);
            labels[i] = nearest;
            inertia += dist2;
        }

        Ok(KmeansResult {
            centroids,
            labels,
            inertia,
            n_iter,
        })
    }

    /// Simulate parallel workers using `std::thread::scope`.
    ///
    /// Each worker samples `mini_batch_size` points from its shard, assigns them
    /// to the nearest centroid, and accumulates `ClusterUpdate`. The results are
    /// then merged (AllReduce sum) on the main thread.
    fn parallel_map_reduce(
        &self,
        shards: &[LocalShard],
        centroids: &[Vec<f64>],
        rng_state: &mut u64,
    ) -> Result<ClusterUpdate, ClusteringError> {
        if shards.is_empty() {
            return Err(ClusteringError::InvalidInput("No shards provided".into()));
        }
        let n_dims = centroids
            .first()
            .map(|c| c.len())
            .ok_or_else(|| ClusteringError::InvalidInput("Empty centroids".into()))?;
        let k = centroids.len();

        // Pre-generate per-shard random seeds deterministically
        let shard_seeds: Vec<u64> = shards
            .iter()
            .map(|s| {
                *rng_state = lcg_next(*rng_state);
                *rng_state ^ (s.worker_id as u64).wrapping_mul(0x9e3779b97f4a7c15)
            })
            .collect();

        let centroids_arc = Arc::new(centroids.to_vec());
        let mini_batch_size = self.config.mini_batch_size;

        // Shared accumulator protected by a Mutex
        let global_update = Arc::new(Mutex::new(ClusterUpdate::new(k, n_dims)));

        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (shard, &seed) in shards.iter().zip(shard_seeds.iter()) {
                let centroids_ref = Arc::clone(&centroids_arc);
                let global_ref = Arc::clone(&global_update);
                let shard_data = shard.data.clone();

                let handle = scope.spawn(move || {
                    let local_update = worker_mini_batch(
                        &shard_data,
                        &centroids_ref,
                        k,
                        n_dims,
                        mini_batch_size,
                        seed,
                    );
                    // Reduce into global
                    if let Ok(mut guard) = global_ref.lock() {
                        guard.merge(&local_update);
                    }
                });
                handles.push(handle);
            }
            // Threads are joined automatically when scope exits
        });

        let update = Arc::try_unwrap(global_update)
            .map_err(|_| ClusteringError::ComputationError("Arc still shared".into()))?
            .into_inner()
            .map_err(|_| ClusteringError::ComputationError("Mutex poisoned".into()))?;

        Ok(update)
    }
}

// ---------------------------------------------------------------------------
// Worker function (runs in each thread)
// ---------------------------------------------------------------------------

fn worker_mini_batch(
    shard_data: &[Vec<f64>],
    centroids: &[Vec<f64>],
    k: usize,
    n_dims: usize,
    mini_batch_size: usize,
    seed: u64,
) -> ClusterUpdate {
    let mut update = ClusterUpdate::new(k, n_dims);
    if shard_data.is_empty() {
        return update;
    }

    let n = shard_data.len();
    let actual_batch = mini_batch_size.min(n);
    let mut rng = seed;

    for _ in 0..actual_batch {
        rng = lcg_next(rng);
        let idx = (rng >> 33) as usize % n;
        let point = &shard_data[idx];
        let (nearest, _) = nearest_centroid(point, centroids);
        for d in 0..n_dims.min(point.len()) {
            update.centroid_sums[nearest][d] += point[d];
        }
        update.centroid_counts[nearest] += 1;
    }
    update
}

// ---------------------------------------------------------------------------
// k-means++ initialisation
// ---------------------------------------------------------------------------

/// D² seeding (k-means++) for centroid initialisation.
pub fn kmeans_plus_plus_init(
    data: &[Vec<f64>],
    k: usize,
    seed: u64,
) -> Result<Vec<Vec<f64>>, ClusteringError> {
    let n = data.len();
    if n == 0 {
        return Err(ClusteringError::InvalidInput("Empty data".into()));
    }
    if k == 0 || k > n {
        return Err(ClusteringError::InvalidInput(
            "k must be in [1, n_samples]".into(),
        ));
    }

    let mut rng = seed;
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);

    // Choose first centroid uniformly at random
    rng = lcg_next(rng);
    let first = (rng >> 33) as usize % n;
    centroids.push(data[first].clone());

    // Choose remaining centroids with probability proportional to D²
    for _ in 1..k {
        let dists: Vec<f64> = data
            .iter()
            .map(|p| {
                centroids
                    .iter()
                    .map(|c| sq_dist(p, c))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        let total: f64 = dists.iter().sum();
        if total <= 0.0 {
            // All remaining points coincide with existing centroids; pick randomly
            rng = lcg_next(rng);
            let idx = (rng >> 33) as usize % n;
            centroids.push(data[idx].clone());
            continue;
        }

        rng = lcg_next(rng);
        let r = ((rng >> 11) as f64 / (u64::MAX >> 11) as f64) * total;
        let mut cumsum = 0.0;
        let mut chosen = n - 1;
        for (i, &d) in dists.iter().enumerate() {
            cumsum += d;
            if cumsum >= r {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen].clone());
    }

    Ok(centroids)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Partition data rows into `n_workers` shards (round-robin).
fn partition_into_shards(data: &[Vec<f64>], n_workers: usize) -> Vec<LocalShard> {
    let workers = n_workers.max(1);
    let mut shards: Vec<LocalShard> = (0..workers)
        .map(|id| LocalShard {
            data: Vec::new(),
            worker_id: id,
        })
        .collect();

    for (i, point) in data.iter().enumerate() {
        shards[i % workers].data.push(point.clone());
    }
    shards
}

/// Returns (index, squared distance) of nearest centroid.
fn nearest_centroid(point: &[f64], centroids: &[Vec<f64>]) -> (usize, f64) {
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let d = sq_dist(point, c);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    (best_idx, best_dist)
}

/// Squared Euclidean distance (handles mismatched lengths gracefully).
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    a.iter()
        .zip(b.iter())
        .take(len)
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}

/// Mean centroid shift (L2) between two centroid sets.
fn centroid_shift(old: &[Vec<f64>], new: &[Vec<f64>]) -> f64 {
    old.iter()
        .zip(new.iter())
        .map(|(o, n)| sq_dist(o, n).sqrt())
        .sum::<f64>()
        / old.len() as f64
}

/// Linear congruential generator step (Knuth MMIX parameters).
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate 3 well-separated 2-D clusters.
    fn make_three_clusters(seed: u64) -> Vec<Vec<f64>> {
        let centers = [(0.0_f64, 0.0_f64), (10.0, 0.0), (5.0, 8.66)];
        let points_per_cluster = 30usize;
        let mut rng = seed;
        let mut data = Vec::with_capacity(points_per_cluster * 3);

        for (cx, cy) in &centers {
            for _ in 0..points_per_cluster {
                rng = lcg_next(rng);
                let dx = ((rng >> 11) as f64 / (u64::MAX >> 11) as f64 - 0.5) * 0.4;
                rng = lcg_next(rng);
                let dy = ((rng >> 11) as f64 / (u64::MAX >> 11) as f64 - 0.5) * 0.4;
                data.push(vec![cx + dx, cy + dy]);
            }
        }
        data
    }

    #[test]
    fn test_fit_three_well_separated_clusters() {
        let data = make_three_clusters(42);
        let config = DistributedKmeansConfig {
            n_clusters: 3,
            max_iter: 50,
            tol: 1e-5,
            n_workers: 3,
            mini_batch_size: 20,
            seed: 7,
        };
        let mut model = DistributedKmeans::new(config);
        let result = model.fit(&data).expect("fit should succeed");

        assert_eq!(result.labels.len(), data.len());
        assert_eq!(result.centroids.len(), 3);
        assert!(result.inertia.is_finite());
        assert!(result.inertia > 0.0);

        // Each group of 30 consecutive points should predominantly share a label
        let labels_c0: Vec<usize> = result.labels[0..30].to_vec();
        let labels_c1: Vec<usize> = result.labels[30..60].to_vec();
        let labels_c2: Vec<usize> = result.labels[60..90].to_vec();

        // Mode of each group should be distinct
        let mode = |v: &[usize]| -> usize {
            let mut counts = std::collections::HashMap::new();
            for &l in v {
                *counts.entry(l).or_insert(0usize) += 1;
            }
            *counts
                .iter()
                .max_by_key(|(_, c)| *c)
                .map(|(l, _)| l)
                .unwrap_or(&0)
        };

        let m0 = mode(&labels_c0);
        let m1 = mode(&labels_c1);
        let m2 = mode(&labels_c2);
        assert_ne!(m0, m1, "cluster 0 and 1 should have different labels");
        assert_ne!(m1, m2, "cluster 1 and 2 should have different labels");
        assert_ne!(m0, m2, "cluster 0 and 2 should have different labels");
    }

    #[test]
    fn test_kmeans_plus_plus_init() {
        let data: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let centroids = kmeans_plus_plus_init(&data, 4, 99).expect("init should succeed");
        assert_eq!(centroids.len(), 4);
    }

    #[test]
    fn test_empty_data_error() {
        let config = DistributedKmeansConfig::default();
        let mut model = DistributedKmeans::new(config);
        assert!(model.fit(&[]).is_err());
    }

    #[test]
    fn test_single_cluster() {
        let data: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, 0.0]).collect();
        let config = DistributedKmeansConfig {
            n_clusters: 1,
            max_iter: 10,
            n_workers: 2,
            mini_batch_size: 5,
            ..Default::default()
        };
        let mut model = DistributedKmeans::new(config);
        let result = model.fit(&data).expect("single cluster fit should succeed");
        assert_eq!(result.centroids.len(), 1);
        assert!(result.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_inertia_decreases_on_warm_start() {
        // A second fit with more iterations should not produce higher inertia
        // (this is probabilistic — test with a fixed seed).
        let data = make_three_clusters(1234);
        let mut model1 = DistributedKmeans::new(DistributedKmeansConfig {
            n_clusters: 3,
            max_iter: 5,
            seed: 1,
            ..Default::default()
        });
        let mut model2 = DistributedKmeans::new(DistributedKmeansConfig {
            n_clusters: 3,
            max_iter: 50,
            seed: 1,
            ..Default::default()
        });
        let r1 = model1.fit(&data).expect("fit1");
        let r2 = model2.fit(&data).expect("fit2");
        // With more iterations the inertia should be <= (or equal if already converged).
        assert!(r2.inertia <= r1.inertia + 1e-6);
    }

    /// Placeholder to ensure VecDeque import is exercised (used in drift module)
    #[allow(dead_code)]
    fn _use_deque() {
        let _d: VecDeque<f64> = VecDeque::new();
    }
}
