//! Consensus and ensemble clustering methods
//!
//! Algorithms for combining multiple base clusterings into a single robust result.
//!
//! # Methods
//!
//! - **Consensus clustering** via co-association matrix + hierarchical agglomeration
//! - **Evidence accumulation clustering (EAC)**
//! - **Bagging-based ensemble** (random subspace + repeated k-means)
//! - **Weighted voting ensemble** (quality-weighted label agreement)

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Co-association matrix (shared by consensus and EAC)
// ---------------------------------------------------------------------------

/// Build a co-association matrix from a set of label vectors.
///
/// `coassoc[i][j]` = fraction of partitions in which i and j share a cluster.
fn build_coassociation_matrix(partitions: &[Vec<usize>], n: usize) -> Result<Array2<f64>> {
    if partitions.is_empty() {
        return Err(ClusteringError::InvalidInput(
            "at least one partition required".into(),
        ));
    }
    let m = partitions.len() as f64;
    let mut coassoc = Array2::<f64>::zeros((n, n));

    for partition in partitions {
        if partition.len() != n {
            return Err(ClusteringError::InvalidInput(
                "partition length must equal n".into(),
            ));
        }
        for i in 0..n {
            for j in i..n {
                if partition[i] == partition[j] {
                    coassoc[[i, j]] += 1.0;
                    if i != j {
                        coassoc[[j, i]] += 1.0;
                    }
                }
            }
        }
    }

    // Normalise
    coassoc.mapv_inplace(|v| v / m);
    Ok(coassoc)
}

// ---------------------------------------------------------------------------
// Simple single-linkage agglomerative from similarity matrix
// ---------------------------------------------------------------------------

/// Single-linkage agglomerative clustering on a similarity matrix,
/// merging until `k` clusters remain.
fn agglomerative_from_similarity(sim: &Array2<f64>, k: usize) -> Result<Vec<usize>> {
    let n = sim.nrows();
    if k == 0 || k > n {
        return Err(ClusteringError::InvalidInput("k must be in [1, n]".into()));
    }

    // Each point starts in its own cluster
    let mut cluster_id: Vec<usize> = (0..n).collect();
    let mut n_clusters = n;

    // Repeatedly merge the two most similar clusters (single-linkage)
    while n_clusters > k {
        let mut best_sim = f64::NEG_INFINITY;
        let mut best_pair = (0, 1);

        for i in 0..n {
            for j in (i + 1)..n {
                if cluster_id[i] != cluster_id[j] && sim[[i, j]] > best_sim {
                    best_sim = sim[[i, j]];
                    best_pair = (i, j);
                }
            }
        }

        // Merge: relabel cluster_id[best_pair.1] -> cluster_id[best_pair.0]
        let target = cluster_id[best_pair.0];
        let source = cluster_id[best_pair.1];
        if target == source {
            // Edge case: all remaining similarities are -inf
            break;
        }
        for idx in 0..n {
            if cluster_id[idx] == source {
                cluster_id[idx] = target;
            }
        }
        n_clusters -= 1;
    }

    // Remap to contiguous labels 0..k-1
    let mut label_map: HashMap<usize, usize> = HashMap::new();
    let mut next_label = 0usize;
    let mut result = vec![0usize; n];
    for i in 0..n {
        let lbl = label_map.entry(cluster_id[i]).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        result[i] = *lbl;
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Average-linkage agglomerative from similarity matrix
// ---------------------------------------------------------------------------

/// Average-linkage agglomerative clustering on a similarity matrix.
fn average_linkage_from_similarity(sim: &Array2<f64>, k: usize) -> Result<Vec<usize>> {
    let n = sim.nrows();
    if k == 0 || k > n {
        return Err(ClusteringError::InvalidInput("k must be in [1, n]".into()));
    }

    // Maintain cluster membership
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    while clusters.len() > k {
        let nc = clusters.len();
        let mut best_sim = f64::NEG_INFINITY;
        let mut best_pair = (0, 1);

        for ci in 0..nc {
            for cj in (ci + 1)..nc {
                // Average similarity between members
                let mut total = 0.0;
                let count = clusters[ci].len() * clusters[cj].len();
                for &a in &clusters[ci] {
                    for &b in &clusters[cj] {
                        total += sim[[a, b]];
                    }
                }
                let avg = if count > 0 {
                    total / count as f64
                } else {
                    f64::NEG_INFINITY
                };
                if avg > best_sim {
                    best_sim = avg;
                    best_pair = (ci, cj);
                }
            }
        }

        // Merge cj into ci
        let cj_members = clusters.remove(best_pair.1);
        clusters[best_pair.0].extend(cj_members);
    }

    // Assign labels
    let mut result = vec![0usize; n];
    for (label, members) in clusters.iter().enumerate() {
        for &idx in members {
            result[idx] = label;
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// 1. Consensus Clustering
// ---------------------------------------------------------------------------

/// Configuration for consensus clustering.
#[derive(Debug, Clone)]
pub struct ConsensusConfig {
    /// Number of bootstrap subsamples.
    pub n_resamples: usize,
    /// Fraction of data to sample each time.
    pub subsample_ratio: f64,
    /// Range of k to try for internal k-means.
    pub k_range: (usize, usize),
    /// Number of output clusters (from agglomerative step).
    pub n_clusters: usize,
    /// Linkage type for the final agglomerative step ("single" or "average").
    pub linkage: String,
    /// Random seed.
    pub seed: Option<u64>,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            n_resamples: 50,
            subsample_ratio: 0.8,
            k_range: (2, 6),
            n_clusters: 2,
            linkage: "average".into(),
            seed: None,
        }
    }
}

/// Result of consensus clustering.
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    /// Final cluster labels.
    pub labels: Vec<usize>,
    /// Co-association matrix.
    pub coassociation: Array2<f64>,
    /// Number of base partitions generated.
    pub n_partitions: usize,
}

/// Consensus clustering via co-association matrix and hierarchical agglomeration.
///
/// 1. Repeatedly subsample the data and cluster with k-means (varying k).
/// 2. Build a co-association matrix counting how often pairs share a cluster.
/// 3. Use hierarchical agglomeration on the co-association matrix to obtain
///    the final partition.
pub fn consensus_clustering<F>(
    data: ArrayView2<F>,
    config: &ConsensusConfig,
) -> Result<ConsensusResult>
where
    F: Float + FromPrimitive + Debug + 'static + Send + Sync + std::iter::Sum + std::fmt::Display,
{
    let n = data.nrows();
    let d = data.ncols();
    if n < 2 {
        return Err(ClusteringError::InvalidInput(
            "need at least 2 samples".into(),
        ));
    }

    let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(config.seed.unwrap_or(42));
    let sample_size = ((n as f64) * config.subsample_ratio).max(2.0) as usize;
    let k_min = config.k_range.0.max(2);
    let k_max = config.k_range.1.max(k_min);

    let mut partitions: Vec<Vec<usize>> = Vec::new();

    for _iter in 0..config.n_resamples {
        // Subsample indices (without replacement)
        let mut indices: Vec<usize> = (0..n).collect();
        shuffle_partial(&mut indices, sample_size, &mut rng);
        let subset_idx = &indices[..sample_size];

        // Extract subsampled data
        let mut sub_data = Array2::<F>::zeros((sample_size, d));
        for (si, &idx) in subset_idx.iter().enumerate() {
            for j in 0..d {
                sub_data[[si, j]] = data[[idx, j]];
            }
        }

        // Choose k
        let k = if k_min == k_max {
            k_min
        } else {
            rng.random_range(k_min..=k_max)
        };
        let k = k.min(sample_size - 1).max(2);

        // Run simple k-means on the subsample
        let sub_labels = simple_kmeans(&sub_data, k, &mut rng)?;

        // Build full partition (points not sampled get a unique singleton label)
        let max_label = sub_labels.iter().copied().max().unwrap_or(0);
        let mut full_partition = vec![0usize; n];
        let mut next_label = max_label + 1;
        let mut in_sample = vec![false; n];
        for (si, &idx) in subset_idx.iter().enumerate() {
            full_partition[idx] = sub_labels[si];
            in_sample[idx] = true;
        }
        for i in 0..n {
            if !in_sample[i] {
                full_partition[i] = next_label;
                next_label += 1;
            }
        }

        partitions.push(full_partition);
    }

    let coassoc = build_coassociation_matrix(&partitions, n)?;

    let labels = if config.linkage == "single" {
        agglomerative_from_similarity(&coassoc, config.n_clusters)?
    } else {
        average_linkage_from_similarity(&coassoc, config.n_clusters)?
    };

    Ok(ConsensusResult {
        labels,
        coassociation: coassoc,
        n_partitions: partitions.len(),
    })
}

// ---------------------------------------------------------------------------
// 2. Evidence Accumulation Clustering (EAC)
// ---------------------------------------------------------------------------

/// Configuration for EAC.
#[derive(Debug, Clone)]
pub struct EacConfig {
    /// Number of base partitions.
    pub n_partitions: usize,
    /// Range of k for base k-means runs.
    pub k_range: (usize, usize),
    /// Number of output clusters.
    pub n_clusters: usize,
    /// Linkage for the agglomerative step ("single" or "average").
    pub linkage: String,
    /// Random seed.
    pub seed: Option<u64>,
}

impl Default for EacConfig {
    fn default() -> Self {
        Self {
            n_partitions: 30,
            k_range: (2, 8),
            n_clusters: 2,
            linkage: "single".into(),
            seed: None,
        }
    }
}

/// Evidence Accumulation Clustering result.
#[derive(Debug, Clone)]
pub struct EacResult {
    /// Final cluster labels.
    pub labels: Vec<usize>,
    /// Co-association matrix.
    pub coassociation: Array2<f64>,
}

/// Evidence Accumulation Clustering.
///
/// Similar to consensus clustering but runs k-means (with varying k) on the
/// **full** dataset repeatedly, then applies single/average linkage
/// on the resulting co-association matrix.
pub fn evidence_accumulation_clustering<F>(
    data: ArrayView2<F>,
    config: &EacConfig,
) -> Result<EacResult>
where
    F: Float + FromPrimitive + Debug + 'static + Send + Sync + std::iter::Sum + std::fmt::Display,
{
    let n = data.nrows();
    if n < 2 {
        return Err(ClusteringError::InvalidInput(
            "need at least 2 samples".into(),
        ));
    }

    let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(config.seed.unwrap_or(42));
    let k_min = config.k_range.0.max(2);
    let k_max = config.k_range.1.max(k_min);

    let data_owned = data.to_owned();
    let mut partitions: Vec<Vec<usize>> = Vec::new();

    for _iter in 0..config.n_partitions {
        let k = if k_min == k_max {
            k_min
        } else {
            rng.random_range(k_min..=k_max)
        };
        let k = k.min(n - 1).max(2);

        let labels = simple_kmeans(&data_owned, k, &mut rng)?;
        partitions.push(labels);
    }

    let coassoc = build_coassociation_matrix(&partitions, n)?;

    let labels = if config.linkage == "single" {
        agglomerative_from_similarity(&coassoc, config.n_clusters)?
    } else {
        average_linkage_from_similarity(&coassoc, config.n_clusters)?
    };

    Ok(EacResult {
        labels,
        coassociation: coassoc,
    })
}

// ---------------------------------------------------------------------------
// 3. Bagging-based Ensemble
// ---------------------------------------------------------------------------

/// Configuration for bagging-based ensemble.
#[derive(Debug, Clone)]
pub struct BaggingEnsembleConfig {
    /// Number of bags.
    pub n_bags: usize,
    /// Bootstrap sample ratio.
    pub sample_ratio: f64,
    /// Feature subspace ratio (1.0 = use all features).
    pub feature_ratio: f64,
    /// k for each bag's k-means.
    pub k: usize,
    /// Number of output clusters.
    pub n_clusters: usize,
    /// Random seed.
    pub seed: Option<u64>,
}

impl Default for BaggingEnsembleConfig {
    fn default() -> Self {
        Self {
            n_bags: 30,
            sample_ratio: 0.8,
            feature_ratio: 1.0,
            k: 3,
            n_clusters: 3,
            seed: None,
        }
    }
}

/// Result of bagging-based ensemble.
#[derive(Debug, Clone)]
pub struct BaggingResult {
    /// Final cluster labels.
    pub labels: Vec<usize>,
    /// Co-association matrix.
    pub coassociation: Array2<f64>,
    /// Number of bags used.
    pub n_bags: usize,
}

/// Bagging-based ensemble clustering.
///
/// Combines bootstrap sampling and random feature subspace selection
/// to produce diverse base clusterings, then merges them via
/// co-association + agglomerative.
pub fn bagging_ensemble<F>(
    data: ArrayView2<F>,
    config: &BaggingEnsembleConfig,
) -> Result<BaggingResult>
where
    F: Float + FromPrimitive + Debug + 'static + Send + Sync + std::iter::Sum + std::fmt::Display,
{
    let n = data.nrows();
    let d = data.ncols();
    if n < 2 {
        return Err(ClusteringError::InvalidInput(
            "need at least 2 samples".into(),
        ));
    }

    let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(config.seed.unwrap_or(42));
    let sample_size = ((n as f64) * config.sample_ratio).max(2.0) as usize;
    let feature_size = ((d as f64) * config.feature_ratio).max(1.0) as usize;

    let mut partitions: Vec<Vec<usize>> = Vec::new();

    for _bag in 0..config.n_bags {
        // Bootstrap sample (with replacement)
        let mut sample_idx: Vec<usize> = Vec::with_capacity(sample_size);
        for _ in 0..sample_size {
            sample_idx.push(rng.random_range(0..n));
        }

        // Feature subspace
        let mut feature_idx: Vec<usize> = (0..d).collect();
        if feature_size < d {
            shuffle_partial(&mut feature_idx, feature_size, &mut rng);
            feature_idx.truncate(feature_size);
        }

        // Extract sub-data
        let mut sub_data = Array2::<F>::zeros((sample_size, feature_size));
        for (si, &idx) in sample_idx.iter().enumerate() {
            for (fi, &feat) in feature_idx.iter().enumerate() {
                sub_data[[si, fi]] = data[[idx, feat]];
            }
        }

        let k = config.k.min(sample_size - 1).max(2);
        let sub_labels = simple_kmeans(&sub_data, k, &mut rng)?;

        // Map back to full size. Use nearest-neighbour assignment for out-of-bag.
        let max_label = sub_labels.iter().copied().max().unwrap_or(0);
        let mut full_partition = vec![0usize; n];

        // Build centroids from subsample
        let mut centroids = vec![vec![F::zero(); feature_size]; k + 1];
        let mut counts = vec![0usize; k + 1];
        for (si, &lbl) in sub_labels.iter().enumerate() {
            counts[lbl] += 1;
            for (fi, _) in feature_idx.iter().enumerate() {
                centroids[lbl][fi] = centroids[lbl][fi] + sub_data[[si, fi]];
            }
        }
        for c in 0..=max_label {
            if counts[c] > 0 {
                let sz = F::from(counts[c]).unwrap_or(F::one());
                for fi in 0..feature_size {
                    centroids[c][fi] = centroids[c][fi] / sz;
                }
            }
        }

        // Assign all n points to nearest centroid
        for i in 0..n {
            let mut best_c = 0;
            let mut best_d = F::infinity();
            for c in 0..=max_label {
                if counts[c] == 0 {
                    continue;
                }
                let mut sq = F::zero();
                for (fi, &feat) in feature_idx.iter().enumerate() {
                    let diff = data[[i, feat]] - centroids[c][fi];
                    sq = sq + diff * diff;
                }
                if sq < best_d {
                    best_d = sq;
                    best_c = c;
                }
            }
            full_partition[i] = best_c;
        }

        partitions.push(full_partition);
    }

    let coassoc = build_coassociation_matrix(&partitions, n)?;
    let labels = average_linkage_from_similarity(&coassoc, config.n_clusters)?;

    Ok(BaggingResult {
        labels,
        coassociation: coassoc,
        n_bags: config.n_bags,
    })
}

// ---------------------------------------------------------------------------
// 4. Weighted Voting Ensemble
// ---------------------------------------------------------------------------

/// Configuration for weighted voting ensemble.
#[derive(Debug, Clone)]
pub struct WeightedVotingConfig {
    /// Number of output clusters.
    pub n_clusters: usize,
}

impl Default for WeightedVotingConfig {
    fn default() -> Self {
        Self { n_clusters: 2 }
    }
}

/// Weighted voting ensemble result.
#[derive(Debug, Clone)]
pub struct WeightedVotingResult {
    /// Final cluster labels.
    pub labels: Vec<usize>,
    /// Weighted co-association matrix.
    pub weighted_coassociation: Array2<f64>,
}

/// Weighted voting ensemble.
///
/// Each base partition has an associated weight (typically the silhouette score
/// or other quality measure). The co-association matrix is built as a
/// weighted average, then hierarchical clustering extracts the final partition.
///
/// # Arguments
///
/// * `partitions` - Vector of (labels, weight) pairs.
/// * `n` - Number of data points.
/// * `config` - Configuration.
pub fn weighted_voting_ensemble(
    partitions: &[(Vec<usize>, f64)],
    n: usize,
    config: &WeightedVotingConfig,
) -> Result<WeightedVotingResult> {
    if partitions.is_empty() {
        return Err(ClusteringError::InvalidInput(
            "at least one partition required".into(),
        ));
    }

    let total_weight: f64 = partitions.iter().map(|(_, w)| w.max(0.0)).sum();
    if total_weight <= 0.0 {
        return Err(ClusteringError::InvalidInput(
            "total weight must be positive".into(),
        ));
    }

    let mut coassoc = Array2::<f64>::zeros((n, n));
    for (labels, weight) in partitions {
        if labels.len() != n {
            return Err(ClusteringError::InvalidInput(
                "partition length must equal n".into(),
            ));
        }
        let w = weight.max(0.0) / total_weight;
        for i in 0..n {
            for j in i..n {
                if labels[i] == labels[j] {
                    coassoc[[i, j]] += w;
                    if i != j {
                        coassoc[[j, i]] += w;
                    }
                }
            }
        }
    }

    let labels = average_linkage_from_similarity(&coassoc, config.n_clusters)?;

    Ok(WeightedVotingResult {
        labels,
        weighted_coassociation: coassoc,
    })
}

// ---------------------------------------------------------------------------
// Internal k-means
// ---------------------------------------------------------------------------

fn simple_kmeans<F>(
    data: &Array2<F>,
    k: usize,
    rng: &mut scirs2_core::random::rngs::StdRng,
) -> Result<Vec<usize>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = data.nrows();
    let d = data.ncols();
    if k >= n {
        return Ok((0..n).collect());
    }

    // K-means++ init
    let mut centroids = Array2::<F>::zeros((k, d));
    let first = rng.random_range(0..n);
    centroids.row_mut(0).assign(&data.row(first));

    for c in 1..k {
        let mut dists = vec![F::infinity(); n];
        for i in 0..n {
            for prev in 0..c {
                let mut sq = F::zero();
                for j in 0..d {
                    let diff = data[[i, j]] - centroids[[prev, j]];
                    sq = sq + diff * diff;
                }
                if sq < dists[i] {
                    dists[i] = sq;
                }
            }
        }
        let total: F = dists.iter().fold(F::zero(), |a, &v| a + v);
        if total <= F::zero() {
            centroids.row_mut(c).assign(&data.row(c.min(n - 1)));
            continue;
        }
        let r = F::from(rng.random::<f64>()).unwrap_or(F::zero()) * total;
        let mut cum = F::zero();
        let mut chosen = 0;
        for i in 0..n {
            cum = cum + dists[i];
            if cum >= r {
                chosen = i;
                break;
            }
        }
        centroids.row_mut(c).assign(&data.row(chosen));
    }

    // Lloyd
    let mut labels = vec![0usize; n];
    for _iter in 0..100 {
        let mut changed = false;
        for i in 0..n {
            let mut best = 0;
            let mut best_d = F::infinity();
            for c in 0..k {
                let mut sq = F::zero();
                for j in 0..d {
                    let diff = data[[i, j]] - centroids[[c, j]];
                    sq = sq + diff * diff;
                }
                if sq < best_d {
                    best_d = sq;
                    best = c;
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

        let mut new_c = Array2::<F>::zeros((k, d));
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            for j in 0..d {
                new_c[[c, j]] = new_c[[c, j]] + data[[i, j]];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let sz = F::from(counts[c]).unwrap_or(F::one());
                for j in 0..d {
                    new_c[[c, j]] = new_c[[c, j]] / sz;
                }
            }
        }
        centroids = new_c;
    }

    Ok(labels)
}

fn shuffle_partial(
    arr: &mut Vec<usize>,
    count: usize,
    rng: &mut scirs2_core::random::rngs::StdRng,
) {
    let n = arr.len();
    let cnt = count.min(n);
    for i in 0..cnt {
        let j = rng.random_range(i..n);
        arr.swap(i, j);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn well_separated_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2,
            ],
        )
        .expect("data")
    }

    #[test]
    fn test_coassociation_matrix() {
        let partitions = vec![vec![0, 0, 1, 1], vec![0, 0, 1, 1], vec![1, 1, 0, 0]];
        let coassoc = build_coassociation_matrix(&partitions, 4).expect("coassoc");
        assert!((coassoc[[0, 1]] - 1.0).abs() < 1e-10, "same cluster always");
        assert!((coassoc[[2, 3]] - 1.0).abs() < 1e-10);
        assert!(
            (coassoc[[0, 2]] - 0.0).abs() < 1e-10,
            "different cluster always"
        );
    }

    #[test]
    fn test_coassociation_error_empty() {
        let partitions: Vec<Vec<usize>> = vec![];
        assert!(build_coassociation_matrix(&partitions, 4).is_err());
    }

    #[test]
    fn test_agglomerative_single_linkage() {
        let sim = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 0.9, 0.1, 0.1, 0.9, 1.0, 0.1, 0.1, 0.1, 0.1, 1.0, 0.9, 0.1, 0.1, 0.9, 1.0,
            ],
        )
        .expect("sim");
        let labels = agglomerative_from_similarity(&sim, 2).expect("agg");
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_consensus_clustering() {
        let data = well_separated_data();
        let config = ConsensusConfig {
            n_resamples: 20,
            subsample_ratio: 0.8,
            k_range: (2, 4),
            n_clusters: 2,
            linkage: "average".into(),
            seed: Some(42),
        };
        let res = consensus_clustering(data.view(), &config).expect("consensus");
        assert_eq!(res.labels.len(), 8);
        // First 4 should be same cluster
        assert_eq!(res.labels[0], res.labels[1]);
        assert_eq!(res.labels[0], res.labels[2]);
        assert_eq!(res.labels[0], res.labels[3]);
        // Last 4 should be same cluster
        assert_eq!(res.labels[4], res.labels[5]);
        assert_eq!(res.labels[4], res.labels[6]);
        assert_eq!(res.labels[4], res.labels[7]);
        // Two groups should differ
        assert_ne!(res.labels[0], res.labels[4]);
    }

    #[test]
    fn test_consensus_single_linkage() {
        let data = well_separated_data();
        let config = ConsensusConfig {
            n_resamples: 15,
            n_clusters: 2,
            linkage: "single".into(),
            seed: Some(99),
            ..Default::default()
        };
        let res = consensus_clustering(data.view(), &config).expect("consensus single");
        assert_eq!(res.labels.len(), 8);
    }

    #[test]
    fn test_evidence_accumulation() {
        let data = well_separated_data();
        let config = EacConfig {
            n_partitions: 20,
            k_range: (2, 5),
            n_clusters: 2,
            linkage: "single".into(),
            seed: Some(42),
        };
        let res = evidence_accumulation_clustering(data.view(), &config).expect("eac");
        assert_eq!(res.labels.len(), 8);
        assert_eq!(res.labels[0], res.labels[1]);
        assert_eq!(res.labels[4], res.labels[5]);
        assert_ne!(res.labels[0], res.labels[4]);
    }

    #[test]
    fn test_eac_average_linkage() {
        let data = well_separated_data();
        let config = EacConfig {
            n_partitions: 15,
            k_range: (2, 4),
            n_clusters: 2,
            linkage: "average".into(),
            seed: Some(77),
        };
        let res = evidence_accumulation_clustering(data.view(), &config).expect("eac avg");
        assert_eq!(res.labels.len(), 8);
    }

    #[test]
    fn test_bagging_ensemble() {
        let data = well_separated_data();
        let config = BaggingEnsembleConfig {
            n_bags: 20,
            sample_ratio: 0.8,
            feature_ratio: 1.0,
            k: 2,
            n_clusters: 2,
            seed: Some(42),
        };
        let res = bagging_ensemble(data.view(), &config).expect("bagging");
        assert_eq!(res.labels.len(), 8);
        assert_eq!(res.labels[0], res.labels[1]);
        assert_eq!(res.labels[4], res.labels[5]);
        assert_ne!(res.labels[0], res.labels[4]);
    }

    #[test]
    fn test_bagging_feature_subspace() {
        let data = well_separated_data();
        let config = BaggingEnsembleConfig {
            n_bags: 15,
            sample_ratio: 0.7,
            feature_ratio: 0.5,
            k: 2,
            n_clusters: 2,
            seed: Some(123),
        };
        // Should not error even with reduced features
        let res = bagging_ensemble(data.view(), &config).expect("bagging subspace");
        assert_eq!(res.labels.len(), 8);
    }

    #[test]
    fn test_weighted_voting() {
        let partitions = vec![
            (vec![0, 0, 0, 0, 1, 1, 1, 1], 0.9),
            (vec![0, 0, 0, 0, 1, 1, 1, 1], 0.8),
            (vec![1, 0, 1, 0, 0, 1, 0, 1], 0.2), // noisy partition, low weight
        ];
        let config = WeightedVotingConfig { n_clusters: 2 };
        let res = weighted_voting_ensemble(&partitions, 8, &config).expect("wv");
        assert_eq!(res.labels.len(), 8);
        // High-weight partitions dominate: first 4 together, last 4 together
        assert_eq!(res.labels[0], res.labels[1]);
        assert_eq!(res.labels[0], res.labels[2]);
        assert_eq!(res.labels[0], res.labels[3]);
        assert_eq!(res.labels[4], res.labels[5]);
        assert_eq!(res.labels[4], res.labels[6]);
        assert_eq!(res.labels[4], res.labels[7]);
        assert_ne!(res.labels[0], res.labels[4]);
    }

    #[test]
    fn test_weighted_voting_error_empty() {
        let config = WeightedVotingConfig { n_clusters: 2 };
        assert!(weighted_voting_ensemble(&[], 4, &config).is_err());
    }

    #[test]
    fn test_weighted_voting_error_zero_weight() {
        let partitions = vec![(vec![0, 0, 1, 1], 0.0)];
        let config = WeightedVotingConfig { n_clusters: 2 };
        assert!(weighted_voting_ensemble(&partitions, 4, &config).is_err());
    }

    #[test]
    fn test_weighted_voting_equal_weights() {
        let partitions = vec![(vec![0, 0, 1, 1], 1.0), (vec![0, 0, 1, 1], 1.0)];
        let config = WeightedVotingConfig { n_clusters: 2 };
        let res = weighted_voting_ensemble(&partitions, 4, &config).expect("wv equal");
        assert_eq!(res.labels[0], res.labels[1]);
        assert_eq!(res.labels[2], res.labels[3]);
        assert_ne!(res.labels[0], res.labels[2]);
    }

    #[test]
    fn test_average_linkage() {
        let sim = Array2::from_shape_vec(
            (6, 6),
            vec![
                1.0, 0.9, 0.8, 0.1, 0.1, 0.1, 0.9, 1.0, 0.85, 0.1, 0.1, 0.1, 0.8, 0.85, 1.0, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.9, 0.85, 0.1, 0.1, 0.1, 0.9, 1.0, 0.8, 0.1, 0.1,
                0.1, 0.85, 0.8, 1.0,
            ],
        )
        .expect("sim");
        let labels = average_linkage_from_similarity(&sim, 2).expect("avg link");
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }
}
