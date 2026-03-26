//! DataLoader-style batching API for neural network training.
//!
//! Provides a `DataLoader` struct that wraps an in-memory dataset and yields
//! mini-batches according to configurable sampling strategies. Epoch-level
//! shuffling, stratified sampling, and weighted-random sampling are all
//! supported without external dependencies.

use crate::error::DatasetsError;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;

// ---------------------------------------------------------------------------
// SamplingStrategy
// ---------------------------------------------------------------------------

/// How rows are ordered / selected when building batches.
///
/// `#[non_exhaustive]` allows future strategies to be added without a
/// breaking change.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Rows are yielded in their natural (insertion) order.
    Sequential,

    /// Rows are globally shuffled at the start of each epoch.
    RandomShuffle,

    /// Stratified sampling: rows are first grouped by the provided class
    /// indices, then interleaved so every batch contains representatives
    /// of every class present in the dataset.
    ///
    /// The inner `Vec<usize>` maps each dataset row index to its class label
    /// (integer encoded).
    Stratified(Vec<usize>),

    /// Weighted random sampling without replacement (within an epoch).
    ///
    /// The inner `Vec<f64>` provides a non-negative weight for every row.
    /// Rows with higher weights are more likely to appear early in the epoch
    /// ordering.
    WeightedRandom(Vec<f64>),
}

// ---------------------------------------------------------------------------
// DataLoaderConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`DataLoader`].
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    /// Mini-batch size (default: 32).
    pub batch_size: usize,
    /// If `true`, epoch-level shuffling is performed (overrides `sampling`
    /// strategy to `RandomShuffle` when set). Default: `true`.
    pub shuffle: bool,
    /// Drop the last (potentially smaller) batch if it has fewer than
    /// `batch_size` rows. Default: `false`.
    pub drop_last: bool,
    /// RNG seed (default: 42).
    pub seed: u64,
    /// Row-selection strategy.  `shuffle = true` takes precedence over this
    /// field by forcing `RandomShuffle` behaviour.
    pub sampling: SamplingStrategy,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: true,
            drop_last: false,
            seed: 42,
            sampling: SamplingStrategy::RandomShuffle,
        }
    }
}

// ---------------------------------------------------------------------------
// Batch
// ---------------------------------------------------------------------------

/// A single mini-batch produced by [`DataLoader`].
#[derive(Debug, Clone)]
pub struct Batch {
    /// Feature matrix, shape `[actual_batch_size, n_features]`.
    pub features: Array2<f64>,
    /// Optional label vector, length `actual_batch_size`.
    pub labels: Option<Array1<f64>>,
    /// Original dataset indices of the rows in this batch.
    pub indices: Vec<usize>,
}

impl Batch {
    /// Number of rows in this batch.
    pub fn batch_size(&self) -> usize {
        self.features.nrows()
    }

    /// Number of features per row.
    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }
}

// ---------------------------------------------------------------------------
// DataLoader
// ---------------------------------------------------------------------------

/// Mini-batch iterator over an in-memory dataset.
///
/// Construct with [`DataLoader::new`], then iterate with the standard `Iterator`
/// interface.  Call [`DataLoader::reset_epoch`] to start a fresh epoch (with a
/// new shuffle if configured).
pub struct DataLoader {
    features: Array2<f64>,
    labels: Option<Vec<f64>>,
    config: DataLoaderConfig,
    /// Permuted row indices for the current epoch.
    indices: Vec<usize>,
    current_pos: usize,
    epoch: usize,
    rng: StdRng,
}

impl DataLoader {
    /// Create a new `DataLoader`.
    ///
    /// `labels` is optional; when `None`, the yielded [`Batch`]es will have
    /// `labels = None`.
    pub fn new(features: Array2<f64>, labels: Option<Vec<f64>>, config: DataLoaderConfig) -> Self {
        let n_rows = features.nrows();
        let mut rng = StdRng::seed_from_u64(config.seed);
        let indices = Self::build_indices(n_rows, &config, &mut rng);
        Self {
            features,
            labels,
            config,
            indices,
            current_pos: 0,
            epoch: 0,
            rng,
        }
    }

    /// Total number of complete batches in the current epoch.
    ///
    /// If `drop_last` is `false` and the dataset size is not an exact multiple
    /// of `batch_size`, this includes the partial final batch.
    pub fn n_batches(&self) -> usize {
        let n = self.indices.len();
        let bs = self.config.batch_size.max(1);
        if self.config.drop_last {
            n / bs
        } else {
            n.div_ceil(bs)
        }
    }

    /// Number of rows in the dataset.
    pub fn n_rows(&self) -> usize {
        self.features.nrows()
    }

    /// Number of features per row.
    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }

    /// The 0-based epoch counter. Incremented by `reset_epoch`.
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Advance to the next epoch: resets the position and, when shuffling is
    /// enabled, builds a fresh permutation.
    pub fn reset_epoch(&mut self) {
        self.epoch += 1;
        self.current_pos = 0;
        let n_rows = self.features.nrows();
        self.indices = Self::build_indices(n_rows, &self.config, &mut self.rng);
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Build an ordered index array according to the sampling strategy.
    fn build_indices(n_rows: usize, config: &DataLoaderConfig, rng: &mut StdRng) -> Vec<usize> {
        if n_rows == 0 {
            return vec![];
        }

        // `shuffle = true` overrides the sampling strategy
        if config.shuffle {
            return Self::fisher_yates(n_rows, rng);
        }

        match &config.sampling {
            SamplingStrategy::Sequential => (0..n_rows).collect(),

            SamplingStrategy::RandomShuffle => Self::fisher_yates(n_rows, rng),

            SamplingStrategy::Stratified(class_labels) => {
                Self::stratified_indices(n_rows, class_labels, rng)
            }

            SamplingStrategy::WeightedRandom(weights) => {
                Self::weighted_indices(n_rows, weights, rng)
            }
        }
    }

    /// Fisher-Yates full-dataset shuffle.
    fn fisher_yates(n: usize, rng: &mut StdRng) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            idx.swap(i, j);
        }
        idx
    }

    /// Interleave class buckets so every batch sees all classes.
    fn stratified_indices(n_rows: usize, class_labels: &[usize], rng: &mut StdRng) -> Vec<usize> {
        // Build per-class index lists
        let max_class = class_labels.iter().copied().max().unwrap_or(0);
        let mut buckets: Vec<Vec<usize>> = vec![vec![]; max_class + 1];
        for (row, &cls) in class_labels.iter().enumerate().take(n_rows) {
            buckets[cls].push(row);
        }
        // Shuffle within each bucket
        for bucket in &mut buckets {
            for i in (1..bucket.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                bucket.swap(i, j);
            }
        }
        // Round-robin interleave
        let mut result = Vec::with_capacity(n_rows);
        let mut cursors = vec![0usize; buckets.len()];
        let mut any_remaining = true;
        while any_remaining {
            any_remaining = false;
            for (cls, bucket) in buckets.iter().enumerate() {
                if cursors[cls] < bucket.len() {
                    result.push(bucket[cursors[cls]]);
                    cursors[cls] += 1;
                    any_remaining = true;
                }
            }
        }
        result
    }

    /// Weighted sampling without replacement via the alias / rejection method.
    ///
    /// Uses a simple O(n log n) approach: sort by uniform_variate / weight,
    /// which is equivalent to Efraimidis-Spirakis weighted reservoir sampling
    /// with a reservoir of size n (i.e., all rows).
    fn weighted_indices(n_rows: usize, weights: &[f64], rng: &mut StdRng) -> Vec<usize> {
        let mut keyed: Vec<(f64, usize)> = (0..n_rows)
            .map(|i| {
                let w = if i < weights.len() {
                    weights[i].max(0.0)
                } else {
                    1.0
                };
                // key = -ln(u) / w  (minimise → Efraimidis-Spirakis)
                let u = (rng.next_u64() as f64 + 1.0) / (u64::MAX as f64 + 1.0);
                let key = if w > 0.0 { -u.ln() / w } else { f64::INFINITY };
                (key, i)
            })
            .collect();
        keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        keyed.into_iter().map(|(_, idx)| idx).collect()
    }

    /// Extract rows at `row_indices` from the feature / label arrays.
    fn extract_batch(&self, row_indices: &[usize]) -> Batch {
        let nf = self.features.ncols();
        let bs = row_indices.len();
        let mut feat_flat = Vec::with_capacity(bs * nf);
        let mut label_vals = Vec::with_capacity(bs);

        for &ri in row_indices {
            for j in 0..nf {
                feat_flat.push(self.features[[ri, j]]);
            }
            if let Some(lbl_vec) = &self.labels {
                label_vals.push(if ri < lbl_vec.len() { lbl_vec[ri] } else { 0.0 });
            }
        }

        let features = Array2::from_shape_vec((bs, nf), feat_flat)
            .unwrap_or_else(|_| Array2::zeros((bs, nf.max(1))));

        let labels = if self.labels.is_some() {
            Some(Array1::from_vec(label_vals))
        } else {
            None
        };

        Batch {
            features,
            labels,
            indices: row_indices.to_vec(),
        }
    }
}

impl Iterator for DataLoader {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        let remaining = self.indices.len().saturating_sub(self.current_pos);
        if remaining == 0 {
            return None;
        }

        let bs = self.config.batch_size.max(1);
        let batch_rows = remaining.min(bs);

        // Drop incomplete last batch if requested
        if self.config.drop_last && batch_rows < bs {
            return None;
        }

        let start = self.current_pos;
        let end = start + batch_rows;
        let row_indices: Vec<usize> = self.indices[start..end].to_vec();
        self.current_pos = end;

        Some(self.extract_batch(&row_indices))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_loader(n: usize, f: usize, bs: usize, shuffle: bool) -> DataLoader {
        let data: Vec<f64> = (0..n * f).map(|x| x as f64).collect();
        let features = Array2::from_shape_vec((n, f), data).unwrap();
        let labels: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let config = DataLoaderConfig {
            batch_size: bs,
            shuffle,
            drop_last: false,
            seed: 42,
            sampling: if shuffle {
                SamplingStrategy::RandomShuffle
            } else {
                SamplingStrategy::Sequential
            },
        };
        DataLoader::new(features, Some(labels), config)
    }

    #[test]
    fn test_dataloader_basic() {
        // 100 rows, batch 32 → 4 batches (32, 32, 32, 4)
        let loader = make_loader(100, 4, 32, false);
        assert_eq!(loader.n_batches(), 4);
        let batches: Vec<_> = loader.collect();
        assert_eq!(batches.len(), 4);
        let total: usize = batches.iter().map(|b| b.batch_size()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_dataloader_last_batch() {
        // 105 rows, batch 32, drop_last=false → 4 batches (32, 32, 32, 9)
        let data: Vec<f64> = (0..105 * 2).map(|x| x as f64).collect();
        let features = Array2::from_shape_vec((105, 2), data).unwrap();
        let config = DataLoaderConfig {
            batch_size: 32,
            shuffle: false,
            drop_last: false,
            seed: 0,
            sampling: SamplingStrategy::Sequential,
        };
        let loader = DataLoader::new(features, None, config);
        let batches: Vec<_> = loader.collect();
        assert_eq!(batches.len(), 4);
        assert_eq!(batches.last().unwrap().batch_size(), 9);
    }

    #[test]
    fn test_dataloader_drop_last() {
        // 105 rows, batch 32, drop_last=true → 3 batches of 32 each
        let data: Vec<f64> = (0..105 * 2).map(|x| x as f64).collect();
        let features = Array2::from_shape_vec((105, 2), data).unwrap();
        let config = DataLoaderConfig {
            batch_size: 32,
            shuffle: false,
            drop_last: true,
            seed: 0,
            sampling: SamplingStrategy::Sequential,
        };
        let loader = DataLoader::new(features, None, config);
        let batches: Vec<_> = loader.collect();
        assert_eq!(batches.len(), 3);
        for b in &batches {
            assert_eq!(b.batch_size(), 32);
        }
    }

    #[test]
    fn test_dataloader_shuffle() {
        // Two consecutive epochs should produce different orderings (with high probability)
        let data: Vec<f64> = (0..50 * 2).map(|x| x as f64).collect();
        let features = Array2::from_shape_vec((50, 2), data).unwrap();
        let config = DataLoaderConfig {
            batch_size: 50,
            shuffle: true,
            drop_last: false,
            seed: 99,
            sampling: SamplingStrategy::RandomShuffle,
        };
        let mut loader = DataLoader::new(features, None, config);

        let first_batch = loader.next().expect("first epoch batch");
        loader.reset_epoch();
        let second_batch = loader.next().expect("second epoch batch");

        // The index orderings should differ (p(same) = 1/50! ≈ 0)
        assert_ne!(first_batch.indices, second_batch.indices);
    }

    #[test]
    fn test_dataloader_stratified() {
        // 30 rows, 3 classes × 10 each; batch_size=6 → each batch has 2 per class
        let n = 30usize;
        let data: Vec<f64> = (0..n * 2).map(|x| x as f64).collect();
        let features = Array2::from_shape_vec((n, 2), data).unwrap();
        let class_labels: Vec<usize> = (0..n).map(|i| i % 3).collect();
        let label_f64: Vec<f64> = class_labels.iter().map(|&c| c as f64).collect();
        let config = DataLoaderConfig {
            batch_size: 6,
            shuffle: false,
            drop_last: false,
            seed: 1,
            sampling: SamplingStrategy::Stratified(class_labels),
        };
        let loader = DataLoader::new(features, Some(label_f64), config);
        let batches: Vec<_> = loader.collect();
        // 30 / 6 = 5 batches
        assert_eq!(batches.len(), 5);
        // Each batch should contain rows from multiple classes
        for batch in &batches {
            if let Some(lbls) = &batch.labels {
                let unique: std::collections::HashSet<i64> =
                    lbls.iter().map(|&x| x as i64).collect();
                assert!(
                    unique.len() >= 2,
                    "expected multiple classes per batch, got {unique:?}"
                );
            }
        }
    }

    #[test]
    fn test_dataloader_epoch_count() {
        let mut loader = make_loader(20, 2, 5, true);
        assert_eq!(loader.epoch(), 0);
        // drain
        for _ in loader.by_ref() {}
        loader.reset_epoch();
        assert_eq!(loader.epoch(), 1);
        for _ in loader.by_ref() {}
        loader.reset_epoch();
        assert_eq!(loader.epoch(), 2);
    }

    #[test]
    fn test_dataloader_empty() {
        let features = Array2::<f64>::zeros((0, 3));
        let config = DataLoaderConfig::default();
        let loader = DataLoader::new(features, None, config);
        assert_eq!(loader.n_batches(), 0);
        let batches: Vec<_> = loader.collect();
        assert!(batches.is_empty());
    }

    #[test]
    fn test_dataloader_exact_multiple() {
        // 64 rows, batch 32, drop_last = false → exactly 2 full batches
        let loader = make_loader(64, 4, 32, false);
        let batches: Vec<_> = loader.collect();
        assert_eq!(batches.len(), 2);
        for b in &batches {
            assert_eq!(b.batch_size(), 32);
        }
    }

    #[test]
    fn test_dataloader_weighted_random() {
        let n = 40usize;
        let data: Vec<f64> = (0..n * 2).map(|x| x as f64).collect();
        let features = Array2::from_shape_vec((n, 2), data).unwrap();
        // Give first 10 rows very high weight
        let weights: Vec<f64> = (0..n).map(|i| if i < 10 { 100.0 } else { 1.0 }).collect();
        let config = DataLoaderConfig {
            batch_size: n, // one big batch
            shuffle: false,
            drop_last: false,
            seed: 7,
            sampling: SamplingStrategy::WeightedRandom(weights),
        };
        let mut loader = DataLoader::new(features, None, config);
        let batch = loader.next().expect("batch");
        // High-weight rows (0-9) should dominate early positions
        let top10: Vec<usize> = batch.indices[..10].to_vec();
        let heavy_in_top10 = top10.iter().filter(|&&i| i < 10).count();
        // Statistically very likely to see ≥ 7 heavy rows in first 10
        assert!(
            heavy_in_top10 >= 5,
            "expected heavy rows near top, got {heavy_in_top10}"
        );
    }
}
