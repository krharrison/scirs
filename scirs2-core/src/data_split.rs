//! Data splitting utilities for machine learning workflows
//!
//! This module provides tools for splitting datasets into training, validation,
//! and test sets using various strategies:
//!
//! - [`train_test_split`] - Simple random train/test split
//! - [`stratified_train_test_split`] - Stratified split preserving class proportions
//! - [`KFold`] - K-fold cross-validation
//! - [`StratifiedKFold`] - Stratified K-fold cross-validation
//! - [`LeaveOneOut`] - Leave-one-out cross-validation
//! - [`TimeSeriesSplit`] - Time series cross-validation (expanding or sliding window)
//! - [`GroupKFold`] - Group K-fold (keeps groups intact)
//! - [`ShuffleSplit`] - Repeated random train/test splits

use crate::error::{CoreError, CoreResult, ErrorContext};
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::hash::Hash;

/// Indices for a single split (train indices, test indices).
pub type SplitIndices = (Vec<usize>, Vec<usize>);

// ---------------------------------------------------------------------------
// train_test_split
// ---------------------------------------------------------------------------

/// Split data indices into training and test sets.
///
/// # Arguments
///
/// * `n_samples` - Total number of samples
/// * `test_size` - Fraction of data to use for testing (0.0 .. 1.0)
/// * `seed` - Optional random seed for reproducibility
///
/// # Example
///
/// ```
/// use scirs2_core::data_split::train_test_split;
///
/// let (train, test) = train_test_split(100, 0.2, Some(42)).expect("split failed");
/// assert_eq!(train.len() + test.len(), 100);
/// assert_eq!(test.len(), 20);
/// ```
pub fn train_test_split(
    n_samples: usize,
    test_size: f64,
    seed: Option<u64>,
) -> CoreResult<SplitIndices> {
    validate_split_params(n_samples, test_size)?;

    let n_test = (n_samples as f64 * test_size).round() as usize;
    let n_test = n_test.max(1).min(n_samples - 1);

    let mut indices: Vec<usize> = (0..n_samples).collect();
    let mut rng = make_rng(seed);
    indices.shuffle(&mut rng);

    let test_indices = indices[..n_test].to_vec();
    let train_indices = indices[n_test..].to_vec();
    Ok((train_indices, test_indices))
}

/// Stratified train/test split that preserves the proportion of each class.
///
/// # Arguments
///
/// * `labels` - Class labels for each sample
/// * `test_size` - Fraction of data for testing (0.0 .. 1.0)
/// * `seed` - Optional random seed
///
/// # Example
///
/// ```
/// use scirs2_core::data_split::stratified_train_test_split;
///
/// let labels = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2];
/// let (train, test) = stratified_train_test_split(&labels, 0.3, Some(42)).expect("split");
/// assert_eq!(train.len() + test.len(), 10);
/// ```
pub fn stratified_train_test_split<L: Eq + Hash + Clone>(
    labels: &[L],
    test_size: f64,
    seed: Option<u64>,
) -> CoreResult<SplitIndices> {
    let n_samples = labels.len();
    validate_split_params(n_samples, test_size)?;

    let mut class_indices: HashMap<&L, Vec<usize>> = HashMap::new();
    for (i, label) in labels.iter().enumerate() {
        class_indices.entry(label).or_default().push(i);
    }

    let mut rng = make_rng(seed);
    let mut train_indices = Vec::new();
    let mut test_indices = Vec::new();

    for (_label, mut indices) in class_indices {
        indices.shuffle(&mut rng);
        let n_class_test = (indices.len() as f64 * test_size).round() as usize;
        let n_class_test = n_class_test.max(1).min(indices.len().saturating_sub(1));
        test_indices.extend_from_slice(&indices[..n_class_test]);
        train_indices.extend_from_slice(&indices[n_class_test..]);
    }

    Ok((train_indices, test_indices))
}

// ---------------------------------------------------------------------------
// KFold
// ---------------------------------------------------------------------------

/// K-fold cross-validation splitter.
///
/// Splits the data into K consecutive folds. Each fold is used once as
/// validation while the remaining K-1 folds form the training set.
///
/// # Example
///
/// ```
/// use scirs2_core::data_split::KFold;
///
/// let kf = KFold::new(5, true, Some(42)).expect("kfold");
/// let splits: Vec<_> = kf.split(100).collect();
/// assert_eq!(splits.len(), 5);
/// for (train, test) in &splits {
///     assert_eq!(train.len() + test.len(), 100);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct KFold {
    /// Number of folds
    pub n_splits: usize,
    /// Whether to shuffle before splitting
    pub shuffle: bool,
    /// Random seed
    pub seed: Option<u64>,
}

impl KFold {
    /// Create a new KFold splitter.
    pub fn new(n_splits: usize, shuffle: bool, seed: Option<u64>) -> CoreResult<Self> {
        if n_splits < 2 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "n_splits must be >= 2 for KFold",
            )));
        }
        Ok(Self {
            n_splits,
            shuffle,
            seed,
        })
    }

    /// Generate splits for `n_samples` data points.
    pub fn split(&self, n_samples: usize) -> impl Iterator<Item = SplitIndices> {
        let mut indices: Vec<usize> = (0..n_samples).collect();
        if self.shuffle {
            let mut rng = make_rng(self.seed);
            indices.shuffle(&mut rng);
        }

        let n_splits = self.n_splits;
        let fold_sizes = compute_fold_sizes(n_samples, n_splits);
        let mut folds: Vec<Vec<usize>> = Vec::with_capacity(n_splits);
        let mut offset = 0;
        for &size in &fold_sizes {
            folds.push(indices[offset..offset + size].to_vec());
            offset += size;
        }

        (0..n_splits).map(move |k| {
            let test = folds[k].clone();
            let train: Vec<usize> = folds
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != k)
                .flat_map(|(_, f)| f.iter().copied())
                .collect();
            (train, test)
        })
    }
}

// ---------------------------------------------------------------------------
// StratifiedKFold
// ---------------------------------------------------------------------------

/// Stratified K-fold cross-validation.
///
/// Each fold preserves the percentage of samples for each class.
///
/// # Example
///
/// ```
/// use scirs2_core::data_split::StratifiedKFold;
///
/// let labels = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
/// let skf = StratifiedKFold::new(5, true, Some(42)).expect("skf");
/// let splits: Vec<_> = skf.split(&labels);
/// assert_eq!(splits.len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct StratifiedKFold {
    /// Number of folds
    pub n_splits: usize,
    /// Whether to shuffle within each class
    pub shuffle: bool,
    /// Random seed
    pub seed: Option<u64>,
}

impl StratifiedKFold {
    /// Create a new StratifiedKFold.
    pub fn new(n_splits: usize, shuffle: bool, seed: Option<u64>) -> CoreResult<Self> {
        if n_splits < 2 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "n_splits must be >= 2 for StratifiedKFold",
            )));
        }
        Ok(Self {
            n_splits,
            shuffle,
            seed,
        })
    }

    /// Generate stratified splits.
    pub fn split<L: Eq + Hash + Clone>(&self, labels: &[L]) -> Vec<SplitIndices> {
        let mut class_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut label_to_int: HashMap<&L, usize> = HashMap::new();
        let mut next_id = 0usize;

        for (i, label) in labels.iter().enumerate() {
            let class_id = *label_to_int.entry(label).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            class_indices.entry(class_id).or_default().push(i);
        }

        let mut rng = make_rng(self.seed);
        if self.shuffle {
            for indices in class_indices.values_mut() {
                indices.shuffle(&mut rng);
            }
        }

        // Assign each sample in each class to a fold in round-robin fashion
        let n_samples = labels.len();
        let mut fold_assignment = vec![0usize; n_samples];
        for indices in class_indices.values() {
            for (pos, &idx) in indices.iter().enumerate() {
                fold_assignment[idx] = pos % self.n_splits;
            }
        }

        (0..self.n_splits)
            .map(|k| {
                let mut train = Vec::new();
                let mut test = Vec::new();
                for (i, &fold) in fold_assignment.iter().enumerate() {
                    if fold == k {
                        test.push(i);
                    } else {
                        train.push(i);
                    }
                }
                (train, test)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// LeaveOneOut
// ---------------------------------------------------------------------------

/// Leave-one-out cross-validation.
///
/// Each sample is used once as the test set. Equivalent to KFold with
/// n_splits = n_samples.
///
/// # Example
///
/// ```
/// use scirs2_core::data_split::LeaveOneOut;
///
/// let loo = LeaveOneOut;
/// let splits: Vec<_> = loo.split(5).collect();
/// assert_eq!(splits.len(), 5);
/// for (train, test) in &splits {
///     assert_eq!(test.len(), 1);
///     assert_eq!(train.len(), 4);
/// }
/// ```
pub struct LeaveOneOut;

impl LeaveOneOut {
    /// Generate leave-one-out splits.
    pub fn split(&self, n_samples: usize) -> impl Iterator<Item = SplitIndices> {
        (0..n_samples).map(move |i| {
            let test = vec![i];
            let train: Vec<usize> = (0..n_samples).filter(|&j| j != i).collect();
            (train, test)
        })
    }
}

// ---------------------------------------------------------------------------
// TimeSeriesSplit
// ---------------------------------------------------------------------------

/// Time series split mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeSeriesMode {
    /// Expanding window: training set grows with each split
    Expanding,
    /// Sliding window: training window has fixed maximum size
    Sliding,
}

/// Time series cross-validation splitter.
///
/// Provides train/test indices for time series data where the test set is always
/// in the future relative to the training set.
///
/// # Example
///
/// ```
/// use scirs2_core::data_split::{TimeSeriesSplit, TimeSeriesMode};
///
/// let ts = TimeSeriesSplit::new(3, TimeSeriesMode::Expanding, None).expect("ts");
/// let splits: Vec<_> = ts.split(20);
/// assert_eq!(splits.len(), 3);
/// // Training sets grow: each has more data than the previous
/// ```
#[derive(Debug, Clone)]
pub struct TimeSeriesSplit {
    /// Number of splits
    pub n_splits: usize,
    /// Splitting mode
    pub mode: TimeSeriesMode,
    /// Maximum training set size (only for Sliding mode)
    pub max_train_size: Option<usize>,
    /// Gap between train and test sets
    pub gap: usize,
}

impl TimeSeriesSplit {
    /// Create a new TimeSeriesSplit.
    pub fn new(
        n_splits: usize,
        mode: TimeSeriesMode,
        max_train_size: Option<usize>,
    ) -> CoreResult<Self> {
        if n_splits < 1 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "n_splits must be >= 1 for TimeSeriesSplit",
            )));
        }
        Ok(Self {
            n_splits,
            mode,
            max_train_size,
            gap: 0,
        })
    }

    /// Set the gap between train and test sets.
    #[must_use]
    pub fn with_gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }

    /// Generate time series splits.
    pub fn split(&self, n_samples: usize) -> Vec<SplitIndices> {
        let test_size = n_samples / (self.n_splits + 1);
        let test_size = test_size.max(1);

        let mut splits = Vec::with_capacity(self.n_splits);

        for k in 0..self.n_splits {
            let test_start = (k + 1) * test_size;
            let test_end = ((k + 2) * test_size).min(n_samples);
            if test_start >= n_samples {
                break;
            }
            let train_end = test_start.saturating_sub(self.gap);
            let train_start = match self.mode {
                TimeSeriesMode::Expanding => 0,
                TimeSeriesMode::Sliding => {
                    if let Some(max_size) = self.max_train_size {
                        train_end.saturating_sub(max_size)
                    } else {
                        0
                    }
                }
            };

            if train_start >= train_end || test_start >= test_end {
                continue;
            }

            let train: Vec<usize> = (train_start..train_end).collect();
            let test: Vec<usize> = (test_start..test_end).collect();
            splits.push((train, test));
        }

        splits
    }
}

// ---------------------------------------------------------------------------
// GroupKFold
// ---------------------------------------------------------------------------

/// Group K-fold cross-validation.
///
/// Ensures that the same group is not represented in both training and test
/// sets. Useful when samples from the same subject/experiment should stay together.
///
/// # Example
///
/// ```
/// use scirs2_core::data_split::GroupKFold;
///
/// let groups = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
/// let gkf = GroupKFold::new(5).expect("gkf");
/// let splits = gkf.split(&groups);
/// assert_eq!(splits.len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct GroupKFold {
    /// Number of folds
    pub n_splits: usize,
}

impl GroupKFold {
    /// Create a new GroupKFold.
    pub fn new(n_splits: usize) -> CoreResult<Self> {
        if n_splits < 2 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "n_splits must be >= 2 for GroupKFold",
            )));
        }
        Ok(Self { n_splits })
    }

    /// Generate splits where groups are kept together.
    pub fn split<G: Eq + Hash + Clone>(&self, groups: &[G]) -> Vec<SplitIndices> {
        // Collect unique groups and their sample indices
        let mut group_to_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut group_to_id: HashMap<&G, usize> = HashMap::new();
        let mut next_id = 0usize;

        for (i, group) in groups.iter().enumerate() {
            let gid = *group_to_id.entry(group).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            group_to_indices.entry(gid).or_default().push(i);
        }

        let n_groups = next_id;
        let actual_splits = self.n_splits.min(n_groups);

        // Assign groups to folds
        let mut group_ids: Vec<usize> = (0..n_groups).collect();
        // Sort by group size (largest first) for balanced folds
        group_ids.sort_by(|a, b| {
            let sa = group_to_indices.get(a).map(|v| v.len()).unwrap_or(0);
            let sb = group_to_indices.get(b).map(|v| v.len()).unwrap_or(0);
            sb.cmp(&sa)
        });

        // Greedy assignment: place each group in the fold with fewest samples
        let mut fold_sizes = vec![0usize; actual_splits];
        let mut group_fold = vec![0usize; n_groups];
        for &gid in &group_ids {
            let min_fold = fold_sizes
                .iter()
                .enumerate()
                .min_by_key(|(_, &s)| s)
                .map(|(i, _)| i)
                .unwrap_or(0);
            group_fold[gid] = min_fold;
            fold_sizes[min_fold] += group_to_indices.get(&gid).map(|v| v.len()).unwrap_or(0);
        }

        (0..actual_splits)
            .map(|k| {
                let mut train = Vec::new();
                let mut test = Vec::new();
                for gid in 0..n_groups {
                    let indices = group_to_indices.get(&gid).cloned().unwrap_or_default();
                    if group_fold[gid] == k {
                        test.extend(indices);
                    } else {
                        train.extend(indices);
                    }
                }
                (train, test)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ShuffleSplit
// ---------------------------------------------------------------------------

/// Repeated random train/test splits.
///
/// Generates independent random splits on each iteration.
///
/// # Example
///
/// ```
/// use scirs2_core::data_split::ShuffleSplit;
///
/// let ss = ShuffleSplit::new(10, 0.2, Some(42)).expect("ss");
/// let splits: Vec<_> = ss.split(100);
/// assert_eq!(splits.len(), 10);
/// for (train, test) in &splits {
///     assert_eq!(train.len() + test.len(), 100);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ShuffleSplit {
    /// Number of re-shuffled splits
    pub n_splits: usize,
    /// Fraction for test set
    pub test_size: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl ShuffleSplit {
    /// Create a new ShuffleSplit.
    pub fn new(n_splits: usize, test_size: f64, seed: Option<u64>) -> CoreResult<Self> {
        if n_splits < 1 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "n_splits must be >= 1 for ShuffleSplit",
            )));
        }
        if test_size <= 0.0 || test_size >= 1.0 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "test_size must be between 0 and 1 (exclusive)",
            )));
        }
        Ok(Self {
            n_splits,
            test_size,
            seed,
        })
    }

    /// Generate repeated random splits.
    pub fn split(&self, n_samples: usize) -> Vec<SplitIndices> {
        let n_test = ((n_samples as f64) * self.test_size).round() as usize;
        let n_test = n_test.max(1).min(n_samples - 1);

        let base_seed = self.seed.unwrap_or(0);
        let mut splits = Vec::with_capacity(self.n_splits);

        for k in 0..self.n_splits {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(k as u64));
            indices.shuffle(&mut rng);

            let test = indices[..n_test].to_vec();
            let train = indices[n_test..].to_vec();
            splits.push((train, test));
        }

        splits
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_split_params(n_samples: usize, test_size: f64) -> CoreResult<()> {
    if n_samples < 2 {
        return Err(CoreError::ValueError(ErrorContext::new(
            "Need at least 2 samples to split",
        )));
    }
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(CoreError::ValueError(ErrorContext::new(
            "test_size must be between 0 and 1 (exclusive)",
        )));
    }
    Ok(())
}

fn make_rng(seed: Option<u64>) -> ChaCha8Rng {
    match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::seed_from_u64(rand::rng().random()),
    }
}

fn compute_fold_sizes(n_samples: usize, n_splits: usize) -> Vec<usize> {
    let base_size = n_samples / n_splits;
    let remainder = n_samples % n_splits;
    let mut sizes = vec![base_size; n_splits];
    for i in 0..remainder {
        sizes[i] += 1;
    }
    sizes
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_test_split_basic() {
        let (train, test) = train_test_split(100, 0.2, Some(42)).expect("split");
        assert_eq!(train.len() + test.len(), 100);
        assert_eq!(test.len(), 20);
        // Ensure no duplicates
        let mut all: Vec<usize> = train.iter().chain(test.iter()).copied().collect();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 100);
    }

    #[test]
    fn test_train_test_split_reproducible() {
        let (train1, test1) = train_test_split(50, 0.3, Some(123)).expect("split1");
        let (train2, test2) = train_test_split(50, 0.3, Some(123)).expect("split2");
        assert_eq!(train1, train2);
        assert_eq!(test1, test2);
    }

    #[test]
    fn test_train_test_split_invalid() {
        assert!(train_test_split(1, 0.5, None).is_err());
        assert!(train_test_split(10, 0.0, None).is_err());
        assert!(train_test_split(10, 1.0, None).is_err());
    }

    #[test]
    fn test_stratified_split() {
        let labels = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        let (train, test) = stratified_train_test_split(&labels, 0.4, Some(42)).expect("split");
        assert_eq!(train.len() + test.len(), 10);
        // Both classes should be represented in test
        let test_labels: Vec<i32> = test.iter().map(|&i| labels[i]).collect();
        assert!(test_labels.contains(&0));
        assert!(test_labels.contains(&1));
    }

    #[test]
    fn test_kfold_basic() {
        let kf = KFold::new(5, false, None).expect("kf");
        let splits: Vec<_> = kf.split(100).collect();
        assert_eq!(splits.len(), 5);
        for (train, test) in &splits {
            assert_eq!(train.len() + test.len(), 100);
        }
    }

    #[test]
    fn test_kfold_shuffle() {
        let kf = KFold::new(3, true, Some(42)).expect("kf");
        let splits: Vec<_> = kf.split(30).collect();
        assert_eq!(splits.len(), 3);
        for (train, test) in &splits {
            assert_eq!(train.len() + test.len(), 30);
            assert_eq!(test.len(), 10);
        }
    }

    #[test]
    fn test_kfold_invalid() {
        assert!(KFold::new(1, false, None).is_err());
    }

    #[test]
    fn test_stratified_kfold() {
        let labels = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        let skf = StratifiedKFold::new(5, true, Some(42)).expect("skf");
        let splits = skf.split(&labels);
        assert_eq!(splits.len(), 5);
        for (train, test) in &splits {
            assert_eq!(train.len() + test.len(), 10);
        }
    }

    #[test]
    fn test_leave_one_out() {
        let loo = LeaveOneOut;
        let splits: Vec<_> = loo.split(5).collect();
        assert_eq!(splits.len(), 5);
        for (train, test) in &splits {
            assert_eq!(test.len(), 1);
            assert_eq!(train.len(), 4);
        }
    }

    #[test]
    fn test_time_series_expanding() {
        let ts = TimeSeriesSplit::new(3, TimeSeriesMode::Expanding, None).expect("ts");
        let splits = ts.split(20);
        assert_eq!(splits.len(), 3);
        // In expanding mode, training sets should grow
        let train_sizes: Vec<usize> = splits.iter().map(|(t, _)| t.len()).collect();
        for i in 1..train_sizes.len() {
            assert!(
                train_sizes[i] >= train_sizes[i - 1],
                "expanding training sets should grow"
            );
        }
    }

    #[test]
    fn test_time_series_sliding() {
        let ts = TimeSeriesSplit::new(3, TimeSeriesMode::Sliding, Some(5)).expect("ts");
        let splits = ts.split(20);
        // All training sets should have at most 5 samples
        for (train, _test) in &splits {
            assert!(train.len() <= 5, "sliding window violated max_train_size");
        }
    }

    #[test]
    fn test_time_series_with_gap() {
        let ts = TimeSeriesSplit::new(3, TimeSeriesMode::Expanding, None)
            .expect("ts")
            .with_gap(2);
        let splits = ts.split(20);
        for (train, test) in &splits {
            if !train.is_empty() && !test.is_empty() {
                let train_max = *train.iter().max().unwrap_or(&0);
                let test_min = *test.iter().min().unwrap_or(&0);
                assert!(test_min > train_max, "gap should separate train and test");
            }
        }
    }

    #[test]
    fn test_group_kfold() {
        let groups = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
        let gkf = GroupKFold::new(5).expect("gkf");
        let splits = gkf.split(&groups);
        assert_eq!(splits.len(), 5);

        // Verify no group appears in both train and test
        for (train, test) in &splits {
            let train_groups: std::collections::HashSet<i32> =
                train.iter().map(|&i| groups[i]).collect();
            let test_groups: std::collections::HashSet<i32> =
                test.iter().map(|&i| groups[i]).collect();
            let overlap: Vec<_> = train_groups.intersection(&test_groups).collect();
            assert!(
                overlap.is_empty(),
                "groups should not overlap: {:?}",
                overlap
            );
        }
    }

    #[test]
    fn test_group_kfold_string_groups() {
        let groups = vec!["a", "a", "b", "b", "c", "c"];
        let gkf = GroupKFold::new(3).expect("gkf");
        let splits = gkf.split(&groups);
        assert_eq!(splits.len(), 3);
    }

    #[test]
    fn test_shuffle_split() {
        let ss = ShuffleSplit::new(10, 0.2, Some(42)).expect("ss");
        let splits = ss.split(100);
        assert_eq!(splits.len(), 10);
        for (train, test) in &splits {
            assert_eq!(train.len() + test.len(), 100);
            assert_eq!(test.len(), 20);
        }
    }

    #[test]
    fn test_shuffle_split_different_seeds() {
        let ss = ShuffleSplit::new(3, 0.3, Some(42)).expect("ss");
        let splits = ss.split(50);
        // Each split should be different
        assert_ne!(splits[0].1, splits[1].1);
    }

    #[test]
    fn test_shuffle_split_invalid() {
        assert!(ShuffleSplit::new(0, 0.2, None).is_err());
        assert!(ShuffleSplit::new(5, 0.0, None).is_err());
        assert!(ShuffleSplit::new(5, 1.0, None).is_err());
    }

    #[test]
    fn test_fold_sizes_even() {
        let sizes = compute_fold_sizes(10, 5);
        assert_eq!(sizes, vec![2, 2, 2, 2, 2]);
    }

    #[test]
    fn test_fold_sizes_uneven() {
        let sizes = compute_fold_sizes(13, 5);
        let total: usize = sizes.iter().sum();
        assert_eq!(total, 13);
        // First 3 should be 3, last 2 should be 2
        assert_eq!(sizes, vec![3, 3, 3, 2, 2]);
    }

    #[test]
    fn test_kfold_no_overlap() {
        let kf = KFold::new(4, true, Some(99)).expect("kf");
        let splits: Vec<_> = kf.split(20).collect();
        // All test indices across folds should cover all samples exactly once
        let mut all_test: Vec<usize> = splits.iter().flat_map(|(_, t)| t.iter().copied()).collect();
        all_test.sort();
        all_test.dedup();
        assert_eq!(all_test.len(), 20);
    }

    #[test]
    fn test_stratified_kfold_proportions() {
        // 70% class 0, 30% class 1
        let labels: Vec<i32> = vec![0; 70].into_iter().chain(vec![1; 30]).collect();
        let skf = StratifiedKFold::new(5, false, None).expect("skf");
        let splits = skf.split(&labels);
        for (_, test) in &splits {
            let n_class0 = test.iter().filter(|&&i| labels[i] == 0).count();
            let n_class1 = test.iter().filter(|&&i| labels[i] == 1).count();
            // Proportions should be roughly maintained
            if !test.is_empty() {
                let ratio = n_class0 as f64 / test.len() as f64;
                assert!(
                    ratio > 0.5 && ratio < 0.9,
                    "class 0 ratio {} not within expected range",
                    ratio
                );
            }
        }
    }
}
