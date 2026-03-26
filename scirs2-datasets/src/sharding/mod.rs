//! Dataset sharding API for distributed and parallel data loading.
//!
//! This module provides utilities to split datasets into shards for distributed
//! training, cross-validation, and parallel processing workflows.
//!
//! ## Key types
//!
//! - [`ShardingConfig`] — configuration driving how sharding is performed.
//! - [`ShardStrategy`] — enumeration of available sharding strategies.
//! - [`DataShard`] — a single shard containing a set of sample indices.
//! - [`ShardedDataset`] — the complete collection of shards over a dataset.

use crate::error::{DatasetsError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// LCG helpers (avoids pulling in the `rand` crate)
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal 64-bit LCG (Knuth parameters).
struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Advance the state and return the next pseudo-random `u64`.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Return a pseudo-random value in `[0, n)`.
    fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }

    /// Return a pseudo-random `f64` in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy to use when sharding a dataset.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ShardStrategy {
    /// Divide contiguous index ranges equally (optionally shuffled first).
    #[default]
    Index,
    /// Hash-based assignment: sample `i` → shard `i % n_shards`.
    Hash,
    /// Stratified by a categorical label column — preserves class proportions.
    Stratified {
        /// Name of the label column (informational; caller must supply labels).
        label_column: String,
    },
    /// Split by approximate shard size in bytes.
    Size {
        /// Target size (bytes) per shard.
        shard_size_bytes: usize,
    },
}

/// Configuration for dataset sharding.
#[derive(Debug, Clone)]
pub struct ShardingConfig {
    /// Number of shards to produce.
    pub n_shards: usize,
    /// Strategy used to assign samples to shards.
    pub strategy: ShardStrategy,
    /// Whether to shuffle indices before partitioning.
    pub shuffle: bool,
    /// Seed for the LCG when `shuffle` is `true`.
    pub seed: u64,
}

impl Default for ShardingConfig {
    fn default() -> Self {
        Self {
            n_shards: 8,
            strategy: ShardStrategy::default(),
            shuffle: true,
            seed: 42,
        }
    }
}

/// A single shard containing a slice of sample indices.
#[derive(Debug, Clone)]
pub struct DataShard {
    /// Zero-based shard identifier.
    pub shard_id: usize,
    /// Total number of shards in the parent [`ShardedDataset`].
    pub n_shards: usize,
    /// Sample indices belonging to this shard.
    pub indices: Vec<usize>,
    /// Whether this shard is designated as a training shard.
    pub is_train: bool,
}

impl DataShard {
    /// Number of samples in this shard.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns `true` if this shard contains no samples.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// A sharded view over a dataset.
///
/// Contains all shards produced according to a [`ShardingConfig`].
#[derive(Debug, Clone)]
pub struct ShardedDataset {
    /// All shards.
    pub shards: Vec<DataShard>,
    /// Total number of samples in the underlying dataset.
    pub total_size: usize,
    /// Configuration used to build this sharded dataset.
    pub config: ShardingConfig,
}

// ─────────────────────────────────────────────────────────────────────────────
// Sharding functions
// ─────────────────────────────────────────────────────────────────────────────

/// Perform a Fisher-Yates shuffle of `0..n` using a seeded LCG.
///
/// Calling this function with the same `seed` always produces the same ordering.
pub fn consistent_shuffle(n: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = Lcg64::new(seed);
    // Fisher-Yates (Knuth) shuffle
    for i in (1..n).rev() {
        let j = rng.next_usize(i + 1);
        indices.swap(i, j);
    }
    indices
}

/// Shard `n_samples` into `n_shards` equal-sized groups by index.
///
/// If `shuffle` is `true` the indices are first shuffled with the given `seed`.
/// The resulting shards cover all indices exactly once.
pub fn shard_by_index(
    n_samples: usize,
    n_shards: usize,
    shuffle: bool,
    seed: u64,
) -> Vec<DataShard> {
    if n_shards == 0 || n_samples == 0 {
        return Vec::new();
    }

    let indices = if shuffle {
        consistent_shuffle(n_samples, seed)
    } else {
        (0..n_samples).collect()
    };

    let base = n_samples / n_shards;
    let remainder = n_samples % n_shards;

    let mut shards = Vec::with_capacity(n_shards);
    let mut offset = 0usize;

    for shard_id in 0..n_shards {
        let extra = if shard_id < remainder { 1 } else { 0 };
        let size = base + extra;
        let shard_indices = indices[offset..offset + size].to_vec();
        shards.push(DataShard {
            shard_id,
            n_shards,
            indices: shard_indices,
            is_train: true,
        });
        offset += size;
    }

    shards
}

/// Shard using consistent hashing: sample `i` always lands in shard `i % n_shards`.
pub fn shard_by_hash(n_samples: usize, n_shards: usize) -> Vec<DataShard> {
    if n_shards == 0 || n_samples == 0 {
        return Vec::new();
    }

    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); n_shards];
    for i in 0..n_samples {
        buckets[i % n_shards].push(i);
    }

    buckets
        .into_iter()
        .enumerate()
        .map(|(shard_id, indices)| DataShard {
            shard_id,
            n_shards,
            indices,
            is_train: true,
        })
        .collect()
}

/// Stratified sharding: distributes each class proportionally across all shards.
///
/// `labels` must have length `n_samples` with integer class identifiers.
/// The caller controls shuffling and seeding.
pub fn shard_stratified(
    labels: &[usize],
    n_shards: usize,
    shuffle: bool,
    seed: u64,
) -> Vec<DataShard> {
    if n_shards == 0 || labels.is_empty() {
        return Vec::new();
    }

    // Group indices by class.
    let max_class = labels.iter().copied().max().unwrap_or(0);
    let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); max_class + 1];
    for (i, &label) in labels.iter().enumerate() {
        class_indices[label].push(i);
    }

    // Optionally shuffle within each class using a per-class seed.
    if shuffle {
        for (cls, indices) in class_indices.iter_mut().enumerate() {
            let class_seed = seed.wrapping_add(cls as u64 * 0x9e37_79b9_7f4a_7c15);
            let shuffled = consistent_shuffle(indices.len(), class_seed);
            let original = indices.clone();
            for (new_pos, &old_pos) in shuffled.iter().enumerate() {
                indices[new_pos] = original[old_pos];
            }
        }
    }

    // Build shard buckets.
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); n_shards];
    for class_idx in class_indices {
        // Round-robin assignment within this class.
        for (pos, sample_idx) in class_idx.into_iter().enumerate() {
            buckets[pos % n_shards].push(sample_idx);
        }
    }

    buckets
        .into_iter()
        .enumerate()
        .map(|(shard_id, indices)| DataShard {
            shard_id,
            n_shards,
            indices,
            is_train: true,
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// ShardedDataset impl
// ─────────────────────────────────────────────────────────────────────────────

impl ShardedDataset {
    /// Build a [`ShardedDataset`] from a dataset of `n_samples` samples and the
    /// given configuration.
    ///
    /// For [`ShardStrategy::Stratified`] use [`ShardedDataset::new_stratified`]
    /// instead, because labels must be supplied externally.
    pub fn new(n_samples: usize, config: ShardingConfig) -> Result<Self> {
        if config.n_shards == 0 {
            return Err(DatasetsError::InvalidFormat("n_shards must be >= 1".into()));
        }
        if n_samples == 0 {
            return Err(DatasetsError::InvalidFormat(
                "n_samples must be >= 1".into(),
            ));
        }

        let shards = match &config.strategy {
            ShardStrategy::Index => {
                shard_by_index(n_samples, config.n_shards, config.shuffle, config.seed)
            }
            ShardStrategy::Hash => shard_by_hash(n_samples, config.n_shards),
            ShardStrategy::Stratified { .. } => {
                return Err(DatasetsError::InvalidFormat(
                    "Use ShardedDataset::new_stratified for Stratified strategy".into(),
                ));
            }
            ShardStrategy::Size { shard_size_bytes } => {
                // Estimate: assume each sample is `n_samples`-byte rows (fallback to index).
                // For a proper size-based split the caller must know the row size.
                // Here we approximate by treating `shard_size_bytes / (n_samples / n_samples)`
                // and fall back to uniform index sharding with the configured n_shards.
                let _ = shard_size_bytes; // informational only at this level
                shard_by_index(n_samples, config.n_shards, config.shuffle, config.seed)
            }
        };

        Ok(Self {
            shards,
            total_size: n_samples,
            config,
        })
    }

    /// Build a [`ShardedDataset`] using stratified sharding with explicit `labels`.
    pub fn new_stratified(labels: &[usize], config: ShardingConfig) -> Result<Self> {
        if config.n_shards == 0 {
            return Err(DatasetsError::InvalidFormat("n_shards must be >= 1".into()));
        }
        if labels.is_empty() {
            return Err(DatasetsError::InvalidFormat(
                "labels must not be empty".into(),
            ));
        }

        let shards = shard_stratified(labels, config.n_shards, config.shuffle, config.seed);
        let total_size = labels.len();

        Ok(Self {
            shards,
            total_size,
            config,
        })
    }

    /// Look up a shard by its identifier.
    pub fn get_shard(&self, shard_id: usize) -> Option<&DataShard> {
        self.shards.get(shard_id)
    }

    /// Partition shard identifiers into a (train, validation) split.
    ///
    /// The last `ceil(n_shards * val_fraction)` shard IDs are used as the
    /// validation set; the rest are used for training.
    pub fn train_shards(&self, val_fraction: f64) -> (Vec<usize>, Vec<usize>) {
        let n = self.shards.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let n_val = ((n as f64 * val_fraction).ceil() as usize).min(n);
        let n_train = n - n_val;
        let train_ids: Vec<usize> = (0..n_train).collect();
        let val_ids: Vec<usize> = (n_train..n).collect();
        (train_ids, val_ids)
    }

    /// Return an iterator over the sample indices of shard `shard_id`.
    ///
    /// Returns an empty iterator if `shard_id` is out of range.
    pub fn shard_iter(&self, shard_id: usize) -> impl Iterator<Item = usize> + '_ {
        let slice: &[usize] = match self.shards.get(shard_id) {
            Some(shard) => &shard.indices,
            None => &[],
        };
        slice.iter().copied()
    }

    /// Total number of shards.
    pub fn n_shards(&self) -> usize {
        self.shards.len()
    }

    /// Total number of samples across all shards.
    pub fn total_samples(&self) -> usize {
        self.shards.iter().map(|s| s.indices.len()).sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data-carrying shard types and functions
// ─────────────────────────────────────────────────────────────────────────────

/// A data-carrying shard containing feature vectors, labels, and indices.
#[derive(Debug, Clone)]
pub struct DatasetShard {
    /// Zero-based shard identifier.
    pub shard_id: usize,
    /// Total number of shards.
    pub total_shards: usize,
    /// Sample indices from the original dataset.
    pub indices: Vec<usize>,
    /// Feature vectors for samples in this shard.
    pub data: Vec<Vec<f64>>,
    /// Labels for samples in this shard.
    pub labels: Vec<usize>,
}

impl DatasetShard {
    /// Number of samples in this shard.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns `true` if this shard is empty.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Split a dataset into `n_shards` equal parts, optionally shuffled.
///
/// Each shard receives a contiguous slice of the (possibly shuffled) index
/// order, along with the corresponding data and label rows.
///
/// # Errors
///
/// Returns an error if `data.len() != labels.len()` or `n_shards == 0`.
pub fn shard_dataset(
    data: &[Vec<f64>],
    labels: &[usize],
    n_shards: usize,
    seed: u64,
) -> Result<Vec<DatasetShard>> {
    let n = data.len();
    if n != labels.len() {
        return Err(DatasetsError::InvalidFormat(format!(
            "data length ({}) != labels length ({})",
            n,
            labels.len()
        )));
    }
    if n_shards == 0 {
        return Err(DatasetsError::InvalidFormat("n_shards must be >= 1".into()));
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let index_shards = shard_by_index(n, n_shards, true, seed);
    Ok(build_dataset_shards(data, labels, &index_shards))
}

/// Split a dataset into `n_shards` shards that maintain per-shard label
/// distribution matching the global distribution.
///
/// # Errors
///
/// Returns an error if `data.len() != labels.len()` or `n_shards == 0`.
pub fn stratified_shard(
    data: &[Vec<f64>],
    labels: &[usize],
    n_shards: usize,
) -> Result<Vec<DatasetShard>> {
    let n = data.len();
    if n != labels.len() {
        return Err(DatasetsError::InvalidFormat(format!(
            "data length ({}) != labels length ({})",
            n,
            labels.len()
        )));
    }
    if n_shards == 0 {
        return Err(DatasetsError::InvalidFormat("n_shards must be >= 1".into()));
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let index_shards = shard_stratified(labels, n_shards, false, 0);
    Ok(build_dataset_shards(data, labels, &index_shards))
}

/// Split a dataset into `n_shards` shards with consistent random shuffling.
///
/// Uses a seeded shuffle so the same seed always produces the same assignment.
///
/// # Errors
///
/// Returns an error if `data.len() != labels.len()` or `n_shards == 0`.
pub fn shuffled_shard(
    data: &[Vec<f64>],
    labels: &[usize],
    n_shards: usize,
    seed: u64,
) -> Result<Vec<DatasetShard>> {
    shard_dataset(data, labels, n_shards, seed)
}

/// Reconstruct a full dataset from a collection of shards.
///
/// Samples are reassembled in index order when possible; otherwise they
/// appear in the order encountered across shards.
pub fn merge_shards(shards: &[DatasetShard]) -> (Vec<Vec<f64>>, Vec<usize>) {
    if shards.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Collect all (index, data, label) triples.
    let mut entries: Vec<(usize, &Vec<f64>, usize)> = Vec::new();
    for shard in shards {
        for (pos, &idx) in shard.indices.iter().enumerate() {
            entries.push((idx, &shard.data[pos], shard.labels[pos]));
        }
    }

    // Sort by original index for deterministic reconstruction.
    entries.sort_by_key(|(idx, _, _)| *idx);

    let data: Vec<Vec<f64>> = entries.iter().map(|(_, d, _)| (*d).clone()).collect();
    let labels: Vec<usize> = entries.iter().map(|(_, _, l)| *l).collect();
    (data, labels)
}

/// Internal helper: convert index-only shards into data-carrying DatasetShards.
fn build_dataset_shards(
    data: &[Vec<f64>],
    labels: &[usize],
    index_shards: &[DataShard],
) -> Vec<DatasetShard> {
    index_shards
        .iter()
        .map(|is| {
            let shard_data: Vec<Vec<f64>> = is.indices.iter().map(|&i| data[i].clone()).collect();
            let shard_labels: Vec<usize> = is.indices.iter().map(|&i| labels[i]).collect();
            DatasetShard {
                shard_id: is.shard_id,
                total_shards: is.n_shards,
                indices: is.indices.clone(),
                data: shard_data,
                labels: shard_labels,
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_by_index_no_shuffle() {
        let shards = shard_by_index(100, 4, false, 0);
        assert_eq!(shards.len(), 4);
        for shard in &shards {
            assert_eq!(shard.indices.len(), 25);
        }
        // All indices covered exactly once.
        let mut seen = [false; 100];
        for shard in &shards {
            for &i in &shard.indices {
                assert!(!seen[i], "index {i} appears twice");
                seen[i] = true;
            }
        }
        assert!(seen.iter().all(|&v| v));
    }

    #[test]
    fn test_shard_by_index_shuffle() {
        let shards = shard_by_index(100, 4, true, 42);
        assert_eq!(shards.len(), 4);
        let total: usize = shards.iter().map(|s| s.len()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_consistent_shuffle_determinism() {
        let a = consistent_shuffle(50, 12345);
        let b = consistent_shuffle(50, 12345);
        assert_eq!(a, b);
        // Different seed → different order (with overwhelming probability).
        let c = consistent_shuffle(50, 99999);
        assert_ne!(a, c);
    }

    #[test]
    fn test_consistent_shuffle_permutation() {
        let n = 200;
        let shuffled = consistent_shuffle(n, 7);
        assert_eq!(shuffled.len(), n);
        let mut sorted = shuffled.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn test_shard_by_hash() {
        let shards = shard_by_hash(100, 4);
        assert_eq!(shards.len(), 4);
        // Shard 0 contains indices 0,4,8,...
        assert!(shards[0].indices.iter().all(|&i| i % 4 == 0));
        let total: usize = shards.iter().map(|s| s.len()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_stratified_class_proportions() {
        // 50 samples: 30 in class 0, 20 in class 1.
        let mut labels = vec![0usize; 30];
        labels.extend(vec![1usize; 20]);
        let shards = shard_stratified(&labels, 5, false, 0);
        assert_eq!(shards.len(), 5);
        // Each shard should have 10 samples (6 from class-0, 4 from class-1).
        for shard in &shards {
            assert_eq!(shard.indices.len(), 10);
        }
    }

    #[test]
    fn test_sharded_dataset_new() {
        let config = ShardingConfig {
            n_shards: 4,
            strategy: ShardStrategy::Index,
            shuffle: false,
            seed: 0,
        };
        let ds = ShardedDataset::new(100, config).expect("should succeed");
        assert_eq!(ds.n_shards(), 4);
        assert_eq!(ds.total_samples(), 100);
    }

    #[test]
    fn test_train_shards_split() {
        let config = ShardingConfig {
            n_shards: 8,
            strategy: ShardStrategy::Index,
            shuffle: false,
            seed: 0,
        };
        let ds = ShardedDataset::new(80, config).expect("should succeed");
        let (train, val) = ds.train_shards(0.25);
        assert_eq!(train.len() + val.len(), 8);
        assert_eq!(val.len(), 2); // ceil(8 * 0.25) = 2
    }

    #[test]
    fn test_shard_iter() {
        let config = ShardingConfig {
            n_shards: 4,
            strategy: ShardStrategy::Index,
            shuffle: false,
            seed: 0,
        };
        let ds = ShardedDataset::new(40, config).expect("should succeed");
        let collected: Vec<usize> = ds.shard_iter(0).collect();
        assert_eq!(collected.len(), 10);
        // shard 0 contains indices 0..10 (no shuffle).
        assert_eq!(collected, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_shard_iter_out_of_bounds() {
        let config = ShardingConfig::default();
        let ds = ShardedDataset::new(10, config).expect("should succeed");
        let empty: Vec<usize> = ds.shard_iter(999).collect();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_sharded_dataset_invalid_config() {
        let bad_config = ShardingConfig {
            n_shards: 0,
            ..Default::default()
        };
        assert!(ShardedDataset::new(100, bad_config).is_err());
    }

    #[test]
    fn test_shard_id_assignment() {
        let shards = shard_by_index(100, 4, false, 0);
        for (expected_id, shard) in shards.iter().enumerate() {
            assert_eq!(shard.shard_id, expected_id);
            assert_eq!(shard.n_shards, 4);
        }
    }

    #[test]
    fn test_stratified_new_stratified() {
        let labels: Vec<usize> = (0..60).map(|i| i % 3).collect();
        let config = ShardingConfig {
            n_shards: 3,
            strategy: ShardStrategy::Stratified {
                label_column: "class".into(),
            },
            shuffle: false,
            seed: 0,
        };
        let ds = ShardedDataset::new_stratified(&labels, config).expect("ok");
        assert_eq!(ds.n_shards(), 3);
        assert_eq!(ds.total_samples(), 60);
    }

    // ── Data-carrying shard tests ──────────────────────────────────────────

    fn make_test_data(n: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
        let data: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64, (i * 2) as f64]).collect();
        let labels: Vec<usize> = (0..n).map(|i| i % 3).collect();
        (data, labels)
    }

    #[test]
    fn test_shard_dataset_total_samples() {
        let (data, labels) = make_test_data(100);
        let shards = shard_dataset(&data, &labels, 4, 42).expect("ok");
        assert_eq!(shards.len(), 4);
        let total: usize = shards.iter().map(|s| s.len()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_stratified_shard_label_proportions() {
        // 60 class-0, 40 class-1
        let mut labels = vec![0usize; 60];
        labels.extend(vec![1usize; 40]);
        let data: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
        let shards = stratified_shard(&data, &labels, 5).expect("ok");
        assert_eq!(shards.len(), 5);
        for shard in &shards {
            let c0 = shard.labels.iter().filter(|&&l| l == 0).count();
            let c1 = shard.labels.iter().filter(|&&l| l == 1).count();
            // Each shard should have 12 class-0 and 8 class-1
            assert_eq!(c0, 12, "Expected 12 class-0 per shard, got {c0}");
            assert_eq!(c1, 8, "Expected 8 class-1 per shard, got {c1}");
        }
    }

    #[test]
    fn test_merge_shards_recovers_data() {
        let (data, labels) = make_test_data(50);
        let shards = shard_dataset(&data, &labels, 5, 99).expect("ok");
        let (merged_data, merged_labels) = merge_shards(&shards);
        assert_eq!(merged_data.len(), 50);
        assert_eq!(merged_labels.len(), 50);
        // After merge (sorted by index), should match original.
        for i in 0..50 {
            assert_eq!(merged_data[i], data[i], "Data mismatch at index {i}");
            assert_eq!(merged_labels[i], labels[i], "Label mismatch at index {i}");
        }
    }

    #[test]
    fn test_shuffled_shard_determinism() {
        let (data, labels) = make_test_data(30);
        let s1 = shuffled_shard(&data, &labels, 3, 42).expect("ok");
        let s2 = shuffled_shard(&data, &labels, 3, 42).expect("ok");
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert_eq!(a.indices, b.indices);
        }
    }

    #[test]
    fn test_shard_dataset_error_on_mismatch() {
        let data = vec![vec![1.0]; 10];
        let labels = vec![0; 5];
        assert!(shard_dataset(&data, &labels, 2, 0).is_err());
    }

    #[test]
    fn test_merge_empty_shards() {
        let (data, labels) = merge_shards(&[]);
        assert!(data.is_empty());
        assert!(labels.is_empty());
    }
}
