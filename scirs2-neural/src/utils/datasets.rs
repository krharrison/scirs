//! Dataset utilities for neural network training
//!
//! This module provides utilities for loading, batching, and preprocessing
//! data for neural network training.

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{s, Array, Array2, ArrayView2, Axis, IxDyn};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Rng, RngExt};
use std::fmt::Debug;
use std::marker::PhantomData;

/// A dataset for neural network training
///
/// Provides efficient batching and shuffling of training data.
///
/// # Examples
///
/// ```rust
/// use scirs2_neural::utils::datasets::Dataset;
/// use scirs2_core::ndarray::Array2;
///
/// // Create a dataset with 100 samples, 10 features
/// let features = Array2::<f64>::zeros((100, 10));
/// let labels = Array2::<f64>::zeros((100, 3));
///
/// let dataset = Dataset::new(features, labels).expect("failed to create dataset");
/// assert_eq!(dataset.len(), 100);
/// ```
#[derive(Debug, Clone)]
pub struct Dataset<F: Float + Debug + NumAssign> {
    /// Feature matrix [num_samples, num_features]
    features: Array2<F>,
    /// Label matrix [num_samples, num_labels]
    labels: Array2<F>,
    /// Indices for shuffling
    indices: Vec<usize>,
}

impl<F: Float + Debug + NumAssign> Dataset<F> {
    /// Create a new dataset from features and labels
    ///
    /// # Arguments
    /// * `features` - Feature matrix [num_samples, num_features]
    /// * `labels` - Label matrix [num_samples, num_labels]
    ///
    /// # Returns
    /// A new Dataset instance
    pub fn new(features: Array2<F>, labels: Array2<F>) -> Result<Self> {
        if features.nrows() != labels.nrows() {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Features and labels must have same number of samples: {} vs {}",
                features.nrows(),
                labels.nrows()
            )));
        }

        let num_samples = features.nrows();
        let indices: Vec<usize> = (0..num_samples).collect();

        Ok(Self {
            features,
            labels,
            indices,
        })
    }

    /// Get the number of samples in the dataset
    pub fn len(&self) -> usize {
        self.features.nrows()
    }

    /// Check if the dataset is empty
    pub fn is_empty(&self) -> bool {
        self.features.nrows() == 0
    }

    /// Get the number of features
    pub fn num_features(&self) -> usize {
        self.features.ncols()
    }

    /// Get the number of labels/outputs
    pub fn num_labels(&self) -> usize {
        self.labels.ncols()
    }

    /// Get a reference to the features
    pub fn features(&self) -> &Array2<F> {
        &self.features
    }

    /// Get a reference to the labels
    pub fn labels(&self) -> &Array2<F> {
        &self.labels
    }

    /// Shuffle the dataset in place
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        let n = self.indices.len();
        for i in (1..n).rev() {
            let j = (rng.random::<f64>() * (i + 1) as f64) as usize;
            self.indices.swap(i, j);
        }
    }

    /// Get a batch of data at the specified indices
    ///
    /// # Arguments
    /// * `start` - Starting index
    /// * `end` - Ending index (exclusive)
    ///
    /// # Returns
    /// A tuple of (features_batch, labels_batch)
    pub fn get_batch(&self, start: usize, end: usize) -> Result<(Array2<F>, Array2<F>)> {
        let end = end.min(self.len());
        if start >= end {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Invalid batch range: {}..{}",
                start, end
            )));
        }

        let batch_indices: Vec<usize> = self.indices[start..end].to_vec();
        let batch_size = batch_indices.len();

        // Extract features for batch
        let mut features_batch = Array2::zeros((batch_size, self.num_features()));
        let mut labels_batch = Array2::zeros((batch_size, self.num_labels()));

        for (batch_idx, &sample_idx) in batch_indices.iter().enumerate() {
            for f in 0..self.num_features() {
                features_batch[[batch_idx, f]] = self.features[[sample_idx, f]];
            }
            for l in 0..self.num_labels() {
                labels_batch[[batch_idx, l]] = self.labels[[sample_idx, l]];
            }
        }

        Ok((features_batch, labels_batch))
    }

    /// Split the dataset into training and validation sets
    ///
    /// # Arguments
    /// * `train_ratio` - Fraction of data to use for training (0.0 to 1.0)
    /// * `rng` - Random number generator for shuffling before split
    ///
    /// # Returns
    /// A tuple of (train_dataset, val_dataset)
    pub fn train_val_split<R: Rng>(
        mut self,
        train_ratio: f64,
        rng: &mut R,
    ) -> Result<(Self, Self)> {
        if !(0.0..=1.0).contains(&train_ratio) {
            return Err(NeuralError::InvalidArchitecture(format!(
                "train_ratio must be between 0 and 1, got {}",
                train_ratio
            )));
        }

        // Shuffle first
        self.shuffle(rng);

        let n = self.len();
        let train_size = (n as f64 * train_ratio) as usize;

        // Get indices for train and val
        let train_indices: Vec<usize> = self.indices[..train_size].to_vec();
        let val_indices: Vec<usize> = self.indices[train_size..].to_vec();

        // Build train dataset
        let mut train_features = Array2::zeros((train_size, self.num_features()));
        let mut train_labels = Array2::zeros((train_size, self.num_labels()));
        for (new_idx, &old_idx) in train_indices.iter().enumerate() {
            for f in 0..self.num_features() {
                train_features[[new_idx, f]] = self.features[[old_idx, f]];
            }
            for l in 0..self.num_labels() {
                train_labels[[new_idx, l]] = self.labels[[old_idx, l]];
            }
        }

        // Build val dataset
        let val_size = n - train_size;
        let mut val_features = Array2::zeros((val_size, self.num_features()));
        let mut val_labels = Array2::zeros((val_size, self.num_labels()));
        for (new_idx, &old_idx) in val_indices.iter().enumerate() {
            for f in 0..self.num_features() {
                val_features[[new_idx, f]] = self.features[[old_idx, f]];
            }
            for l in 0..self.num_labels() {
                val_labels[[new_idx, l]] = self.labels[[old_idx, l]];
            }
        }

        Ok((
            Dataset::new(train_features, train_labels)?,
            Dataset::new(val_features, val_labels)?,
        ))
    }
}

/// Iterator for batching a dataset
///
/// Provides efficient iteration over batches of a dataset.
pub struct BatchIterator<'a, F: Float + Debug + NumAssign> {
    dataset: &'a Dataset<F>,
    batch_size: usize,
    current_idx: usize,
    drop_last: bool,
}

impl<'a, F: Float + Debug + NumAssign> BatchIterator<'a, F> {
    /// Create a new batch iterator
    ///
    /// # Arguments
    /// * `dataset` - The dataset to iterate over
    /// * `batch_size` - Size of each batch
    /// * `drop_last` - Whether to drop the last batch if it's smaller than batch_size
    pub fn new(dataset: &'a Dataset<F>, batch_size: usize, drop_last: bool) -> Self {
        Self {
            dataset,
            batch_size,
            current_idx: 0,
            drop_last,
        }
    }

    /// Get the number of batches
    pub fn num_batches(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            n.div_ceil(self.batch_size)
        }
    }
}

impl<'a, F: Float + Debug + NumAssign> Iterator for BatchIterator<'a, F> {
    type Item = Result<(Array2<F>, Array2<F>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.len() {
            return None;
        }

        let start = self.current_idx;
        let end = (start + self.batch_size).min(self.dataset.len());

        // Check if we should drop this batch
        if self.drop_last && end - start < self.batch_size {
            return None;
        }

        self.current_idx = end;
        Some(self.dataset.get_batch(start, end))
    }
}

/// Data loader for training neural networks
///
/// Provides shuffling, batching, and iteration over datasets.
///
/// # Examples
///
/// ```rust
/// use scirs2_neural::utils::datasets::{Dataset, DataLoader};
/// use scirs2_core::ndarray::Array2;
///
/// let features = Array2::<f64>::zeros((100, 10));
/// let labels = Array2::<f64>::zeros((100, 3));
/// let dataset = Dataset::new(features, labels).expect("failed to create dataset");
///
/// let mut loader = DataLoader::new(dataset, 16, true, true);
///
/// for epoch in 0..2 {
///     for batch_result in loader.iter() {
///         let (x, y) = batch_result.expect("batch failed");
///         // Process batch
///     }
///     loader.on_epoch_end(); // Shuffle for next epoch
/// }
/// ```
pub struct DataLoader<F: Float + Debug + NumAssign> {
    dataset: Dataset<F>,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl<F: Float + Debug + NumAssign> DataLoader<F> {
    /// Create a new data loader
    ///
    /// # Arguments
    /// * `dataset` - The dataset to load
    /// * `batch_size` - Size of each batch
    /// * `shuffle` - Whether to shuffle the data each epoch
    /// * `drop_last` - Whether to drop the last batch if it's incomplete
    pub fn new(dataset: Dataset<F>, batch_size: usize, shuffle: bool, drop_last: bool) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle,
            drop_last,
        }
    }

    /// Get the number of batches per epoch
    pub fn num_batches(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            n.div_ceil(self.batch_size)
        }
    }

    /// Get the dataset size
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Check if the data loader is empty
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }

    /// Get an iterator over batches
    pub fn iter(&self) -> BatchIterator<'_, F> {
        BatchIterator::new(&self.dataset, self.batch_size, self.drop_last)
    }

    /// Call this at the end of each epoch to shuffle data
    pub fn on_epoch_end(&mut self) {
        if self.shuffle {
            let mut rng = scirs2_core::random::rng();
            self.dataset.shuffle(&mut rng);
        }
    }

    /// Get a reference to the underlying dataset
    pub fn dataset(&self) -> &Dataset<F> {
        &self.dataset
    }
}

/// Normalization strategy for features
#[derive(Debug, Clone, Copy)]
pub enum Normalization {
    /// Standard normalization: (x - mean) / std
    StandardScaler,
    /// Min-max normalization: (x - min) / (max - min)
    MinMaxScaler,
    /// No normalization
    None,
}

/// Normalize features according to the specified strategy
///
/// # Arguments
/// * `features` - Feature matrix [num_samples, num_features]
/// * `strategy` - Normalization strategy to apply
///
/// # Returns
/// Normalized feature matrix
pub fn normalize_features<F: Float + Debug + NumAssign>(
    features: &Array2<F>,
    strategy: Normalization,
) -> Array2<F> {
    match strategy {
        Normalization::None => features.clone(),
        Normalization::StandardScaler => {
            let mut result = features.clone();
            for j in 0..features.ncols() {
                // Compute mean
                let mut sum = F::zero();
                for i in 0..features.nrows() {
                    sum += features[[i, j]];
                }
                let mean = sum / F::from(features.nrows()).unwrap_or(F::one());

                // Compute std
                let mut var_sum = F::zero();
                for i in 0..features.nrows() {
                    let diff = features[[i, j]] - mean;
                    var_sum += diff * diff;
                }
                let std = (var_sum / F::from(features.nrows()).unwrap_or(F::one())).sqrt();
                let std = if std < F::from(1e-8).unwrap_or(F::zero()) {
                    F::one()
                } else {
                    std
                };

                // Normalize
                for i in 0..features.nrows() {
                    result[[i, j]] = (features[[i, j]] - mean) / std;
                }
            }
            result
        }
        Normalization::MinMaxScaler => {
            let mut result = features.clone();
            for j in 0..features.ncols() {
                // Find min and max
                let mut min_val = features[[0, j]];
                let mut max_val = features[[0, j]];
                for i in 1..features.nrows() {
                    if features[[i, j]] < min_val {
                        min_val = features[[i, j]];
                    }
                    if features[[i, j]] > max_val {
                        max_val = features[[i, j]];
                    }
                }

                let range = max_val - min_val;
                let range = if range < F::from(1e-8).unwrap_or(F::zero()) {
                    F::one()
                } else {
                    range
                };

                // Normalize
                for i in 0..features.nrows() {
                    result[[i, j]] = (features[[i, j]] - min_val) / range;
                }
            }
            result
        }
    }
}

/// One-hot encode labels
///
/// # Arguments
/// * `labels` - Integer label array
/// * `num_classes` - Number of classes
///
/// # Returns
/// One-hot encoded label matrix
pub fn one_hot_encode<F: Float + Debug + NumAssign>(
    labels: &[usize],
    num_classes: usize,
) -> Array2<F> {
    let n = labels.len();
    let mut encoded = Array2::zeros((n, num_classes));

    for (i, &label) in labels.iter().enumerate() {
        if label < num_classes {
            encoded[[i, label]] = F::one();
        }
    }

    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::rng;

    #[test]
    fn test_dataset_creation() {
        let features = Array2::<f64>::zeros((100, 10));
        let labels = Array2::<f64>::zeros((100, 3));

        let dataset = Dataset::new(features, labels).expect("Operation failed");
        assert_eq!(dataset.len(), 100);
        assert_eq!(dataset.num_features(), 10);
        assert_eq!(dataset.num_labels(), 3);
    }

    #[test]
    fn test_dataset_mismatched_sizes() {
        let features = Array2::<f64>::zeros((100, 10));
        let labels = Array2::<f64>::zeros((50, 3)); // Wrong size

        let result = Dataset::new(features, labels);
        assert!(result.is_err());
    }

    #[test]
    fn test_dataset_shuffle() {
        let mut features = Array2::<f64>::zeros((10, 2));
        for i in 0..10 {
            features[[i, 0]] = i as f64;
        }
        let labels = Array2::<f64>::zeros((10, 1));

        let mut dataset = Dataset::new(features.clone(), labels).expect("Operation failed");
        let original_indices = dataset.indices.clone();

        let mut rng = rng();
        dataset.shuffle(&mut rng);

        // Indices should be different after shuffle (very unlikely to be same)
        assert_ne!(dataset.indices, original_indices);
    }

    #[test]
    fn test_get_batch() {
        let mut features = Array2::<f64>::zeros((10, 2));
        let mut labels = Array2::<f64>::zeros((10, 1));
        for i in 0..10 {
            features[[i, 0]] = i as f64;
            labels[[i, 0]] = i as f64;
        }

        let dataset = Dataset::new(features, labels).expect("Operation failed");
        let (batch_x, batch_y) = dataset.get_batch(0, 5).expect("Operation failed");

        assert_eq!(batch_x.nrows(), 5);
        assert_eq!(batch_y.nrows(), 5);
    }

    #[test]
    fn test_train_val_split() {
        let features = Array2::<f64>::ones((100, 10));
        let labels = Array2::<f64>::zeros((100, 3));

        let dataset = Dataset::new(features, labels).expect("Operation failed");
        let mut rng = rng();
        let (train, val) = dataset
            .train_val_split(0.8, &mut rng)
            .expect("Operation failed");

        assert_eq!(train.len(), 80);
        assert_eq!(val.len(), 20);
    }

    #[test]
    fn test_batch_iterator() {
        let features = Array2::<f64>::zeros((25, 5));
        let labels = Array2::<f64>::zeros((25, 2));

        let dataset = Dataset::new(features, labels).expect("Operation failed");
        let iter = BatchIterator::new(&dataset, 10, false);

        assert_eq!(iter.num_batches(), 3); // 25 / 10 = 2.5, rounded up to 3

        let batches: Vec<_> = iter.collect();
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn test_batch_iterator_drop_last() {
        let features = Array2::<f64>::zeros((25, 5));
        let labels = Array2::<f64>::zeros((25, 2));

        let dataset = Dataset::new(features, labels).expect("Operation failed");
        let iter = BatchIterator::new(&dataset, 10, true);

        assert_eq!(iter.num_batches(), 2); // 25 / 10 = 2 (drop remainder)

        let batches: Vec<_> = iter.collect();
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn test_data_loader() {
        let features = Array2::<f64>::zeros((50, 10));
        let labels = Array2::<f64>::zeros((50, 3));

        let dataset = Dataset::new(features, labels).expect("Operation failed");
        let loader = DataLoader::new(dataset, 16, true, false);

        assert_eq!(loader.len(), 50);
        assert_eq!(loader.num_batches(), 4); // ceil(50/16) = 4
    }

    #[test]
    fn test_standard_scaler() {
        let mut features = Array2::<f64>::zeros((100, 2));
        for i in 0..100 {
            features[[i, 0]] = i as f64;
            features[[i, 1]] = (i as f64) * 2.0;
        }

        let normalized = normalize_features(&features, Normalization::StandardScaler);

        // Check that mean is approximately 0
        let mean_col0: f64 = normalized.column(0).iter().sum::<f64>() / 100.0;
        assert!(mean_col0.abs() < 1e-10);
    }

    #[test]
    fn test_minmax_scaler() {
        let mut features = Array2::<f64>::zeros((10, 1));
        for i in 0..10 {
            features[[i, 0]] = i as f64 * 10.0; // 0, 10, 20, ..., 90
        }

        let normalized = normalize_features(&features, Normalization::MinMaxScaler);

        // Check range is [0, 1]
        let min_val: f64 = normalized.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val: f64 = normalized.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        assert!((min_val - 0.0).abs() < 1e-10);
        assert!((max_val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_one_hot_encode() {
        let labels = vec![0, 1, 2, 0, 1];
        let encoded: Array2<f64> = one_hot_encode(&labels, 3);

        assert_eq!(encoded.nrows(), 5);
        assert_eq!(encoded.ncols(), 3);

        // Check encoding
        assert_eq!(encoded[[0, 0]], 1.0);
        assert_eq!(encoded[[1, 1]], 1.0);
        assert_eq!(encoded[[2, 2]], 1.0);
    }
}
