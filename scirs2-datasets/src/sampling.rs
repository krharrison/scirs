//! Mini-batch sampler for iterating over datasets in batches.
//!
//! Provides configurable sampling strategies including sequential, random,
//! stratified (proportional label representation per batch), and weighted
//! random sampling.

use crate::error::{DatasetsError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// LCG helper (deterministic PRNG without external rand dependency)
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

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy used by the mini-batch sampler to select samples.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum SamplerStrategy {
    /// Iterate through the dataset in order.
    #[default]
    Sequential,
    /// Randomly shuffle all indices, then iterate sequentially over the shuffled order.
    Random,
    /// Each batch maintains the same label proportions as the full dataset.
    Stratified,
    /// Sample indices according to per-example weights (with replacement).
    WeightedRandom {
        /// Per-sample weight. Length must equal the number of samples.
        weights: Vec<f64>,
    },
}

/// Configuration for the [`MiniBatchSampler`].
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Number of samples per batch.
    pub batch_size: usize,
    /// Whether to shuffle the dataset before creating batches.
    pub shuffle: bool,
    /// If `true`, the last batch is dropped when it has fewer than `batch_size` samples.
    pub drop_last: bool,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Sampling strategy.
    pub strategy: SamplerStrategy,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: true,
            drop_last: false,
            seed: 42,
            strategy: SamplerStrategy::default(),
        }
    }
}

/// A single mini-batch of data and labels.
#[derive(Debug, Clone)]
pub struct MiniBatch {
    /// Feature vectors for this batch.
    pub data: Vec<Vec<f64>>,
    /// Labels for this batch.
    pub labels: Vec<usize>,
    /// Indices into the original dataset for this batch.
    pub indices: Vec<usize>,
}

/// Mini-batch sampler that yields batches from a dataset.
///
/// Construct via [`MiniBatchSampler::new`] and iterate with [`iter_batches`](MiniBatchSampler::iter_batches).
#[derive(Debug, Clone)]
pub struct MiniBatchSampler {
    config: SamplerConfig,
}

impl MiniBatchSampler {
    /// Create a new sampler with the given configuration.
    pub fn new(config: SamplerConfig) -> Self {
        Self { config }
    }

    /// Return a reference to the current configuration.
    pub fn config(&self) -> &SamplerConfig {
        &self.config
    }

    /// Generate all mini-batches for the given data and labels.
    ///
    /// Returns a `Vec<MiniBatch>` where each batch has at most `config.batch_size`
    /// samples. When `drop_last` is `true`, the final partial batch is omitted.
    ///
    /// # Errors
    ///
    /// Returns an error if data and labels have different lengths, or if
    /// `batch_size` is zero.
    pub fn iter_batches(&self, data: &[Vec<f64>], labels: &[usize]) -> Result<Vec<MiniBatch>> {
        iter_batches(data, labels, &self.config)
    }
}

/// Generate mini-batches from a dataset according to the given configuration.
///
/// This is the free-function equivalent of [`MiniBatchSampler::iter_batches`].
///
/// # Errors
///
/// Returns an error when `data.len() != labels.len()` or `config.batch_size == 0`.
pub fn iter_batches(
    data: &[Vec<f64>],
    labels: &[usize],
    config: &SamplerConfig,
) -> Result<Vec<MiniBatch>> {
    let n = data.len();
    if n != labels.len() {
        return Err(DatasetsError::InvalidFormat(format!(
            "data length ({}) != labels length ({})",
            n,
            labels.len()
        )));
    }
    if config.batch_size == 0 {
        return Err(DatasetsError::InvalidFormat(
            "batch_size must be >= 1".into(),
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let indices = build_index_order(n, labels, config);
    let mut batches = Vec::new();
    let mut offset = 0;

    while offset < indices.len() {
        let end = (offset + config.batch_size).min(indices.len());
        let batch_indices: Vec<usize> = indices[offset..end].to_vec();

        if config.drop_last && batch_indices.len() < config.batch_size {
            break;
        }

        let batch_data: Vec<Vec<f64>> = batch_indices.iter().map(|&i| data[i].clone()).collect();
        let batch_labels: Vec<usize> = batch_indices.iter().map(|&i| labels[i]).collect();

        batches.push(MiniBatch {
            data: batch_data,
            labels: batch_labels,
            indices: batch_indices,
        });

        offset = end;
    }

    Ok(batches)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build the ordered index list according to the strategy.
fn build_index_order(n: usize, labels: &[usize], config: &SamplerConfig) -> Vec<usize> {
    match &config.strategy {
        SamplerStrategy::Sequential => {
            let mut indices: Vec<usize> = (0..n).collect();
            if config.shuffle {
                fisher_yates_shuffle(&mut indices, config.seed);
            }
            indices
        }

        SamplerStrategy::Random => {
            let mut indices: Vec<usize> = (0..n).collect();
            fisher_yates_shuffle(&mut indices, config.seed);
            indices
        }

        SamplerStrategy::Stratified => build_stratified_order(n, labels, config),

        SamplerStrategy::WeightedRandom { weights } => build_weighted_order(n, weights, config),
    }
}

/// Fisher-Yates shuffle using seeded LCG.
fn fisher_yates_shuffle(indices: &mut [usize], seed: u64) {
    let n = indices.len();
    if n <= 1 {
        return;
    }
    let mut rng = Lcg64::new(seed);
    for i in (1..n).rev() {
        let j = rng.next_usize(i + 1);
        indices.swap(i, j);
    }
}

/// Build an index order that groups samples so each batch has proportional
/// label representation.
fn build_stratified_order(n: usize, labels: &[usize], config: &SamplerConfig) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }

    // Group indices by class.
    let max_class = labels.iter().copied().max().unwrap_or(0);
    let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); max_class + 1];
    for (i, &label) in labels.iter().enumerate() {
        class_indices[label].push(i);
    }

    // Optionally shuffle within each class.
    if config.shuffle {
        for (cls, indices) in class_indices.iter_mut().enumerate() {
            let class_seed = config.seed.wrapping_add(cls as u64 * 0x9e37_79b9_7f4a_7c15);
            fisher_yates_shuffle(indices, class_seed);
        }
    }

    // Interleave: round-robin across classes to ensure proportional representation
    // within each batch-sized chunk.
    let mut result = Vec::with_capacity(n);
    let mut cursors: Vec<usize> = vec![0; class_indices.len()];
    let mut remaining = n;

    while remaining > 0 {
        let mut added = false;
        for (cls, indices) in class_indices.iter().enumerate() {
            if cursors[cls] < indices.len() {
                result.push(indices[cursors[cls]]);
                cursors[cls] += 1;
                remaining -= 1;
                added = true;
                if remaining == 0 {
                    break;
                }
            }
        }
        if !added {
            break;
        }
    }

    result
}

/// Build an index order using weighted sampling with replacement.
fn build_weighted_order(n: usize, weights: &[f64], config: &SamplerConfig) -> Vec<usize> {
    if n == 0 || weights.is_empty() {
        return Vec::new();
    }

    let mut rng = Lcg64::new(config.seed);
    let actual_weights: Vec<f64> = if weights.len() >= n {
        weights[..n].to_vec()
    } else {
        // Pad with uniform weight 1.0.
        let mut w = weights.to_vec();
        w.resize(n, 1.0);
        w
    };

    // Build cumulative distribution.
    let total: f64 = actual_weights.iter().sum();
    if total <= 0.0 {
        // Fallback to uniform.
        return (0..n).collect();
    }
    let cumulative: Vec<f64> = actual_weights
        .iter()
        .scan(0.0, |acc, &w| {
            *acc += w / total;
            Some(*acc)
        })
        .collect();

    // Sample n indices with replacement.
    (0..n)
        .map(|_| {
            let u = rng.next_f64();
            // Binary search in cumulative distribution.
            match cumulative.binary_search_by(|probe| {
                probe.partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                Ok(idx) => idx.min(n - 1),
                Err(idx) => idx.min(n - 1),
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

    fn make_simple_data(n: usize, n_features: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                (0..n_features)
                    .map(|j| (i * n_features + j) as f64)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_sequential_batches_correct_size() {
        let data = make_simple_data(100, 5);
        let labels: Vec<usize> = (0..100).map(|i| i % 3).collect();
        let config = SamplerConfig {
            batch_size: 32,
            shuffle: false,
            drop_last: false,
            seed: 42,
            strategy: SamplerStrategy::Sequential,
        };
        let batches = iter_batches(&data, &labels, &config).expect("should succeed");
        // 100 / 32 = 3 full + 1 partial = 4 batches
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0].data.len(), 32);
        assert_eq!(batches[3].data.len(), 4); // remainder
    }

    #[test]
    fn test_drop_last() {
        let data = make_simple_data(50, 3);
        let labels: Vec<usize> = vec![0; 50];
        let config = SamplerConfig {
            batch_size: 16,
            shuffle: false,
            drop_last: true,
            seed: 0,
            strategy: SamplerStrategy::Sequential,
        };
        let batches = iter_batches(&data, &labels, &config).expect("should succeed");
        // 50 / 16 = 3 full batches (48), last partial (2) dropped
        assert_eq!(batches.len(), 3);
        for b in &batches {
            assert_eq!(b.data.len(), 16);
        }
    }

    #[test]
    fn test_random_shuffles_indices() {
        let data = make_simple_data(20, 2);
        let labels: Vec<usize> = vec![0; 20];
        let config = SamplerConfig {
            batch_size: 20,
            shuffle: true,
            drop_last: false,
            seed: 99,
            strategy: SamplerStrategy::Random,
        };
        let batches = iter_batches(&data, &labels, &config).expect("should succeed");
        assert_eq!(batches.len(), 1);
        // Indices should be a permutation of 0..20
        let mut sorted = batches[0].indices.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..20).collect::<Vec<_>>());
        // Very unlikely to be in natural order
        assert_ne!(batches[0].indices, (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn test_stratified_label_proportions() {
        // 60 class-0, 40 class-1
        let n = 100;
        let mut labels: Vec<usize> = vec![0; 60];
        labels.extend(vec![1; 40]);
        let data = make_simple_data(n, 2);

        let config = SamplerConfig {
            batch_size: 20,
            shuffle: false,
            drop_last: false,
            seed: 42,
            strategy: SamplerStrategy::Stratified,
        };
        let batches = iter_batches(&data, &labels, &config).expect("should succeed");
        assert_eq!(batches.len(), 5); // 100 / 20

        // Check overall: total class proportions should be preserved
        let total_c0: usize = batches
            .iter()
            .map(|b| b.labels.iter().filter(|&&l| l == 0).count())
            .sum();
        let total_c1: usize = batches
            .iter()
            .map(|b| b.labels.iter().filter(|&&l| l == 1).count())
            .sum();
        assert_eq!(total_c0, 60);
        assert_eq!(total_c1, 40);

        // Each batch should have at least some representation of both classes
        // (round-robin interleaving distributes them across batches)
        let batches_with_both: usize = batches
            .iter()
            .filter(|b| {
                let c0 = b.labels.iter().filter(|&&l| l == 0).count();
                let c1 = b.labels.iter().filter(|&&l| l == 1).count();
                c0 > 0 && c1 > 0
            })
            .count();
        // At least 4 out of 5 batches should have both classes
        assert!(
            batches_with_both >= 4,
            "Expected most batches to have both classes, got {batches_with_both}"
        );
    }

    #[test]
    fn test_weighted_sampling() {
        let n = 50;
        let data = make_simple_data(n, 2);
        let labels: Vec<usize> = vec![0; n];
        // Give all weight to index 0
        let mut weights = vec![0.0; n];
        weights[0] = 1.0;

        let config = SamplerConfig {
            batch_size: 10,
            shuffle: false,
            drop_last: false,
            seed: 42,
            strategy: SamplerStrategy::WeightedRandom { weights },
        };
        let batches = iter_batches(&data, &labels, &config).expect("should succeed");
        // All samples should be index 0
        for batch in &batches {
            for &idx in &batch.indices {
                assert_eq!(idx, 0, "All indices should be 0 with weight=[1,0,0,...]");
            }
        }
    }

    #[test]
    fn test_reproducibility_same_seed() {
        let data = make_simple_data(40, 3);
        let labels: Vec<usize> = (0..40).map(|i| i % 2).collect();
        let config = SamplerConfig {
            batch_size: 10,
            shuffle: true,
            drop_last: false,
            seed: 777,
            strategy: SamplerStrategy::Random,
        };
        let b1 = iter_batches(&data, &labels, &config).expect("ok");
        let b2 = iter_batches(&data, &labels, &config).expect("ok");
        assert_eq!(b1.len(), b2.len());
        for (a, b) in b1.iter().zip(b2.iter()) {
            assert_eq!(a.indices, b.indices);
        }
    }

    #[test]
    fn test_mismatched_lengths_error() {
        let data = make_simple_data(10, 2);
        let labels: Vec<usize> = vec![0; 5];
        let config = SamplerConfig::default();
        assert!(iter_batches(&data, &labels, &config).is_err());
    }

    #[test]
    fn test_zero_batch_size_error() {
        let data = make_simple_data(10, 2);
        let labels: Vec<usize> = vec![0; 10];
        let config = SamplerConfig {
            batch_size: 0,
            ..Default::default()
        };
        assert!(iter_batches(&data, &labels, &config).is_err());
    }

    #[test]
    fn test_empty_dataset() {
        let data: Vec<Vec<f64>> = Vec::new();
        let labels: Vec<usize> = Vec::new();
        let config = SamplerConfig::default();
        let batches = iter_batches(&data, &labels, &config).expect("ok");
        assert!(batches.is_empty());
    }

    #[test]
    fn test_sampler_struct() {
        let data = make_simple_data(20, 2);
        let labels: Vec<usize> = vec![0; 20];
        let sampler = MiniBatchSampler::new(SamplerConfig {
            batch_size: 5,
            shuffle: false,
            drop_last: false,
            seed: 0,
            strategy: SamplerStrategy::Sequential,
        });
        let batches = sampler.iter_batches(&data, &labels).expect("ok");
        assert_eq!(batches.len(), 4);
        assert_eq!(sampler.config().batch_size, 5);
    }

    #[test]
    fn test_all_indices_covered_sequential() {
        let n = 37;
        let data = make_simple_data(n, 2);
        let labels: Vec<usize> = vec![0; n];
        let config = SamplerConfig {
            batch_size: 10,
            shuffle: false,
            drop_last: false,
            seed: 0,
            strategy: SamplerStrategy::Sequential,
        };
        let batches = iter_batches(&data, &labels, &config).expect("ok");
        let mut all_indices: Vec<usize> = batches
            .iter()
            .flat_map(|b| b.indices.iter().copied())
            .collect();
        all_indices.sort_unstable();
        assert_eq!(all_indices, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn test_batch_data_matches_original() {
        let data = make_simple_data(15, 3);
        let labels: Vec<usize> = (0..15).map(|i| i % 2).collect();
        let config = SamplerConfig {
            batch_size: 5,
            shuffle: false,
            drop_last: false,
            seed: 0,
            strategy: SamplerStrategy::Sequential,
        };
        let batches = iter_batches(&data, &labels, &config).expect("ok");
        for batch in &batches {
            for (pos, &idx) in batch.indices.iter().enumerate() {
                assert_eq!(batch.data[pos], data[idx]);
                assert_eq!(batch.labels[pos], labels[idx]);
            }
        }
    }
}
