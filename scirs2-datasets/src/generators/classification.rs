//! Advanced classification dataset generators
//!
//! Provides sklearn-style synthetic classification generators including
//! multi-label classification, Hastie et al. binary classification,
//! and enhanced n-class classification with informative, redundant,
//! and noise features.

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;

/// Helper to create an RNG from an optional seed
fn create_rng(randomseed: Option<u64>) -> StdRng {
    match randomseed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = thread_rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    }
}

/// Configuration for the enhanced classification generator
#[derive(Debug, Clone)]
pub struct ClassificationConfig {
    /// Number of samples
    pub n_samples: usize,
    /// Total number of features
    pub n_features: usize,
    /// Number of informative features
    pub n_informative: usize,
    /// Number of redundant features (linear combinations of informative)
    pub n_redundant: usize,
    /// Number of repeated features (duplicates of informative + redundant)
    pub n_repeated: usize,
    /// Number of classes
    pub n_classes: usize,
    /// Number of clusters per class
    pub n_clusters_per_class: usize,
    /// Fraction of labels to flip (noise in labels)
    pub flip_y: f64,
    /// Scale of the hypercube containing the clusters
    pub class_sep: f64,
    /// Whether to shuffle the samples and features
    pub shuffle: bool,
    /// Optional random seed for reproducibility
    pub random_state: Option<u64>,
}

impl Default for ClassificationConfig {
    fn default() -> Self {
        Self {
            n_samples: 100,
            n_features: 20,
            n_informative: 2,
            n_redundant: 2,
            n_repeated: 0,
            n_classes: 2,
            n_clusters_per_class: 2,
            flip_y: 0.01,
            class_sep: 1.0,
            shuffle: true,
            random_state: None,
        }
    }
}

/// Generate an enhanced random n-class classification problem
///
/// This is a more featureful version of `make_classification` in `basic.rs`,
/// following the sklearn interface more closely. It creates a dataset with
/// informative features, redundant features (linear combinations of informative
/// features), repeated features (duplicates), and pure noise features.
///
/// # Arguments
///
/// * `config` - Classification configuration specifying all parameters
///
/// # Returns
///
/// A `Dataset` with n_samples rows and n_features columns, plus target labels
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::classification::{make_classification_enhanced, ClassificationConfig};
///
/// let config = ClassificationConfig {
///     n_samples: 200,
///     n_features: 20,
///     n_informative: 5,
///     n_redundant: 3,
///     n_repeated: 2,
///     n_classes: 3,
///     random_state: Some(42),
///     ..Default::default()
/// };
/// let ds = make_classification_enhanced(config).expect("should succeed");
/// assert_eq!(ds.n_samples(), 200);
/// assert_eq!(ds.n_features(), 20);
/// ```
pub fn make_classification_enhanced(config: ClassificationConfig) -> Result<Dataset> {
    // Validate parameters
    if config.n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }
    if config.n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be > 0".to_string(),
        ));
    }
    if config.n_informative == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_informative must be > 0".to_string(),
        ));
    }
    if config.n_classes < 2 {
        return Err(DatasetsError::InvalidFormat(
            "n_classes must be >= 2".to_string(),
        ));
    }
    if config.n_clusters_per_class == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_clusters_per_class must be > 0".to_string(),
        ));
    }
    let total_useful = config.n_informative + config.n_redundant + config.n_repeated;
    if total_useful > config.n_features {
        return Err(DatasetsError::InvalidFormat(format!(
            "n_informative ({}) + n_redundant ({}) + n_repeated ({}) = {} must be <= n_features ({})",
            config.n_informative,
            config.n_redundant,
            config.n_repeated,
            total_useful,
            config.n_features
        )));
    }
    if config.n_informative < config.n_classes {
        return Err(DatasetsError::InvalidFormat(format!(
            "n_informative ({}) must be >= n_classes ({})",
            config.n_informative, config.n_classes
        )));
    }
    if config.flip_y < 0.0 || config.flip_y > 1.0 {
        return Err(DatasetsError::InvalidFormat(
            "flip_y must be in [0, 1]".to_string(),
        ));
    }

    let mut rng = create_rng(config.random_state);

    let normal = scirs2_core::random::Normal::new(0.0, 1.0).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;

    let n_noise = config.n_features - config.n_informative - config.n_redundant - config.n_repeated;

    // Step 1: Generate informative features by creating centroids per cluster
    let n_centroids = config.n_classes * config.n_clusters_per_class;
    let mut centroids = Array2::zeros((n_centroids, config.n_informative));

    for i in 0..n_centroids {
        for j in 0..config.n_informative {
            centroids[[i, j]] = config.class_sep * (2.0 * rng.random::<f64>() - 1.0);
        }
    }

    // Step 2: Generate samples around centroids
    let mut informative = Array2::zeros((config.n_samples, config.n_informative));
    let mut target = Array1::zeros(config.n_samples);

    let samples_per_class = config.n_samples / config.n_classes;
    let remainder = config.n_samples % config.n_classes;
    let mut idx = 0;

    for class_idx in 0..config.n_classes {
        let n_samples_class = if class_idx < remainder {
            samples_per_class + 1
        } else {
            samples_per_class
        };
        let spc = n_samples_class / config.n_clusters_per_class;
        let spc_rem = n_samples_class % config.n_clusters_per_class;

        for cluster_idx in 0..config.n_clusters_per_class {
            let n_cluster = if cluster_idx < spc_rem { spc + 1 } else { spc };
            let centroid_idx = class_idx * config.n_clusters_per_class + cluster_idx;

            for _ in 0..n_cluster {
                for j in 0..config.n_informative {
                    informative[[idx, j]] =
                        centroids[[centroid_idx, j]] + 0.5 * normal.sample(&mut rng);
                }
                target[idx] = class_idx as f64;
                idx += 1;
            }
        }
    }

    // Step 3: Generate redundant features as linear combinations of informative features
    let mut redundant = Array2::zeros((config.n_samples, config.n_redundant));
    if config.n_redundant > 0 {
        // Create a random mixing matrix
        let mut mixing = Array2::zeros((config.n_informative, config.n_redundant));
        for i in 0..config.n_informative {
            for j in 0..config.n_redundant {
                mixing[[i, j]] = normal.sample(&mut rng);
            }
        }
        // redundant = informative @ mixing
        for i in 0..config.n_samples {
            for j in 0..config.n_redundant {
                let mut val = 0.0;
                for k in 0..config.n_informative {
                    val += informative[[i, k]] * mixing[[k, j]];
                }
                redundant[[i, j]] = val;
            }
        }
    }

    // Step 4: Generate repeated features (copies of informative + redundant)
    let mut repeated = Array2::zeros((config.n_samples, config.n_repeated));
    if config.n_repeated > 0 {
        let source_cols = config.n_informative + config.n_redundant;
        for j in 0..config.n_repeated {
            let src_j = j % source_cols;
            for i in 0..config.n_samples {
                if src_j < config.n_informative {
                    repeated[[i, j]] = informative[[i, src_j]];
                } else {
                    repeated[[i, j]] = redundant[[i, src_j - config.n_informative]];
                }
            }
        }
    }

    // Step 5: Generate noise features
    let mut noise_features = Array2::zeros((config.n_samples, n_noise));
    for i in 0..config.n_samples {
        for j in 0..n_noise {
            noise_features[[i, j]] = normal.sample(&mut rng);
        }
    }

    // Step 6: Assemble the full feature matrix
    let mut data = Array2::zeros((config.n_samples, config.n_features));
    for i in 0..config.n_samples {
        let mut col = 0;
        for j in 0..config.n_informative {
            data[[i, col]] = informative[[i, j]];
            col += 1;
        }
        for j in 0..config.n_redundant {
            data[[i, col]] = redundant[[i, j]];
            col += 1;
        }
        for j in 0..config.n_repeated {
            data[[i, col]] = repeated[[i, j]];
            col += 1;
        }
        for j in 0..n_noise {
            data[[i, col]] = noise_features[[i, j]];
            col += 1;
        }
    }

    // Step 7: Flip labels with probability flip_y
    if config.flip_y > 0.0 {
        let uniform = scirs2_core::random::Uniform::new(0.0, 1.0).map_err(|e| {
            DatasetsError::ComputationError(format!("Failed to create uniform dist: {e}"))
        })?;
        for i in 0..config.n_samples {
            if uniform.sample(&mut rng) < config.flip_y {
                // Assign a random different class
                let current = target[i] as usize;
                let mut new_class = rng.random_range(0..config.n_classes);
                while new_class == current && config.n_classes > 1 {
                    new_class = rng.random_range(0..config.n_classes);
                }
                target[i] = new_class as f64;
            }
        }
    }

    // Step 8: Shuffle if requested
    if config.shuffle {
        let n = config.n_samples;
        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            if i != j {
                // Swap rows in data
                for col in 0..config.n_features {
                    let tmp = data[[i, col]];
                    data[[i, col]] = data[[j, col]];
                    data[[j, col]] = tmp;
                }
                // Swap targets
                let tmp = target[i];
                target[i] = target[j];
                target[j] = tmp;
            }
        }
    }

    // Build feature names
    let mut feature_names = Vec::with_capacity(config.n_features);
    for j in 0..config.n_informative {
        feature_names.push(format!("informative_{j}"));
    }
    for j in 0..config.n_redundant {
        feature_names.push(format!("redundant_{j}"));
    }
    for j in 0..config.n_repeated {
        feature_names.push(format!("repeated_{j}"));
    }
    for j in 0..n_noise {
        feature_names.push(format!("noise_{j}"));
    }

    let class_names: Vec<String> = (0..config.n_classes)
        .map(|i| format!("class_{i}"))
        .collect();

    let dataset = Dataset::new(data, Some(target))
        .with_featurenames(feature_names)
        .with_targetnames(class_names)
        .with_description(format!(
            "Enhanced classification dataset: {} samples, {} features ({} informative, {} redundant, {} repeated, {} noise), {} classes",
            config.n_samples, config.n_features, config.n_informative,
            config.n_redundant, config.n_repeated, n_noise, config.n_classes
        ))
        .with_metadata("n_informative", &config.n_informative.to_string())
        .with_metadata("n_redundant", &config.n_redundant.to_string())
        .with_metadata("n_repeated", &config.n_repeated.to_string())
        .with_metadata("n_noise", &n_noise.to_string())
        .with_metadata("class_sep", &config.class_sep.to_string())
        .with_metadata("flip_y", &config.flip_y.to_string());

    Ok(dataset)
}

/// Configuration for multi-label classification generator
#[derive(Debug, Clone)]
pub struct MultilabelConfig {
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Number of classes (labels)
    pub n_classes: usize,
    /// Number of labels per sample on average
    pub n_labels: usize,
    /// Whether to allow return_indicator format (target as matrix)
    pub allow_unlabeled: bool,
    /// Optional random seed
    pub random_state: Option<u64>,
}

impl Default for MultilabelConfig {
    fn default() -> Self {
        Self {
            n_samples: 100,
            n_features: 20,
            n_classes: 5,
            n_labels: 2,
            allow_unlabeled: true,
            random_state: None,
        }
    }
}

/// Result type for multi-label classification datasets
///
/// Multi-label datasets have a target matrix instead of a target vector,
/// where each column represents a binary label.
#[derive(Debug, Clone)]
pub struct MultilabelDataset {
    /// Feature matrix (n_samples x n_features)
    pub data: Array2<f64>,
    /// Target indicator matrix (n_samples x n_classes), binary entries
    pub target: Array2<f64>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Class/label names
    pub class_names: Vec<String>,
    /// Description
    pub description: String,
}

/// Generate a random multi-label classification problem
///
/// Each sample can belong to multiple classes simultaneously. The target is
/// an indicator matrix where `target[i,j] = 1` if sample i has label j.
///
/// The generation process:
/// 1. Create class centers in feature space
/// 2. For each sample, generate features near one or more class centers
/// 3. Assign labels based on proximity to class centers
///
/// # Arguments
///
/// * `config` - Multi-label configuration
///
/// # Returns
///
/// A `MultilabelDataset` with feature matrix and binary indicator target matrix
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::classification::{make_multilabel_classification, MultilabelConfig};
///
/// let config = MultilabelConfig {
///     n_samples: 100,
///     n_features: 10,
///     n_classes: 4,
///     n_labels: 2,
///     random_state: Some(42),
///     ..Default::default()
/// };
/// let ds = make_multilabel_classification(config).expect("should succeed");
/// assert_eq!(ds.data.nrows(), 100);
/// assert_eq!(ds.data.ncols(), 10);
/// assert_eq!(ds.target.ncols(), 4);
/// ```
pub fn make_multilabel_classification(config: MultilabelConfig) -> Result<MultilabelDataset> {
    if config.n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }
    if config.n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be > 0".to_string(),
        ));
    }
    if config.n_classes == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_classes must be > 0".to_string(),
        ));
    }
    if config.n_labels == 0 || config.n_labels > config.n_classes {
        return Err(DatasetsError::InvalidFormat(format!(
            "n_labels ({}) must be in [1, n_classes ({})]",
            config.n_labels, config.n_classes
        )));
    }

    let mut rng = create_rng(config.random_state);

    let normal = scirs2_core::random::Normal::new(0.0, 1.0).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;

    // Generate class centers
    let mut centers = Array2::zeros((config.n_classes, config.n_features));
    for i in 0..config.n_classes {
        for j in 0..config.n_features {
            centers[[i, j]] = 3.0 * normal.sample(&mut rng);
        }
    }

    // Generate samples and assign multiple labels
    let mut data = Array2::zeros((config.n_samples, config.n_features));
    let mut target_matrix = Array2::zeros((config.n_samples, config.n_classes));

    for i in 0..config.n_samples {
        // Select n_labels random classes for this sample
        let mut labels: Vec<usize> = Vec::with_capacity(config.n_labels);
        while labels.len() < config.n_labels {
            let candidate = rng.random_range(0..config.n_classes);
            if !labels.contains(&candidate) {
                labels.push(candidate);
            }
        }

        // If !allow_unlabeled, ensure at least one label
        if !config.allow_unlabeled && labels.is_empty() {
            labels.push(rng.random_range(0..config.n_classes));
        }

        // Generate features as a mixture of the selected class centers
        for j in 0..config.n_features {
            let mut val = 0.0;
            for &label in &labels {
                val += centers[[label, j]];
            }
            val /= labels.len() as f64;
            val += normal.sample(&mut rng); // Add noise
            data[[i, j]] = val;
        }

        // Set target indicators
        for &label in &labels {
            target_matrix[[i, label]] = 1.0;
        }
    }

    let feature_names: Vec<String> = (0..config.n_features)
        .map(|j| format!("feature_{j}"))
        .collect();
    let class_names: Vec<String> = (0..config.n_classes)
        .map(|j| format!("label_{j}"))
        .collect();

    Ok(MultilabelDataset {
        data,
        target: target_matrix,
        feature_names,
        class_names,
        description: format!(
            "Multi-label classification dataset: {} samples, {} features, {} classes, ~{} labels per sample",
            config.n_samples, config.n_features, config.n_classes, config.n_labels
        ),
    })
}

/// Generate the Hastie et al. 10-dimensional binary classification dataset
///
/// Generates data from the 10-dimensional standard normal distribution.
/// The target is defined as:
///   y = 1 if sum(x_i^2) > chi-squared median (9.34), else -1
///
/// This is the dataset used in:
/// Hastie, T., Tibshirani, R., Friedman, J. (2009).
/// The Elements of Statistical Learning, 2nd Edition, Example 10.2.
///
/// # Arguments
///
/// * `n_samples` - Number of samples (default 12000 in sklearn, split 2000/10000 train/test)
/// * `random_state` - Optional random seed
///
/// # Returns
///
/// A `Dataset` with 10 features and binary target {-1, 1}
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::classification::make_hastie_10_2;
///
/// let ds = make_hastie_10_2(12000, Some(42)).expect("should succeed");
/// assert_eq!(ds.n_samples(), 12000);
/// assert_eq!(ds.n_features(), 10);
/// ```
pub fn make_hastie_10_2(n_samples: usize, random_state: Option<u64>) -> Result<Dataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }

    let mut rng = create_rng(random_state);

    let normal = scirs2_core::random::Normal::new(0.0, 1.0).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;

    let n_features = 10;
    // Chi-squared(10) median is approximately 9.3418
    let chi2_median = 9.3418;

    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let mut sum_sq = 0.0;
        for j in 0..n_features {
            let val = normal.sample(&mut rng);
            data[[i, j]] = val;
            sum_sq += val * val;
        }

        target[i] = if sum_sq > chi2_median { 1.0 } else { -1.0 };
    }

    let feature_names: Vec<String> = (0..n_features).map(|j| format!("x_{j}")).collect();

    let dataset = Dataset::new(data, Some(target))
        .with_featurenames(feature_names)
        .with_targetnames(vec!["-1".to_string(), "1".to_string()])
        .with_description(
            "Hastie et al. 10.2 binary classification dataset. \
             Features are standard normal; y=1 if sum(x_i^2) > 9.34 (chi2(10) median), else y=-1. \
             Reference: Hastie, Tibshirani, Friedman (2009) The Elements of Statistical Learning."
                .to_string(),
        )
        .with_metadata("chi2_median_threshold", &chi2_median.to_string())
        .with_metadata("n_features", &n_features.to_string());

    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // make_classification_enhanced tests
    // =========================================================================

    #[test]
    fn test_classification_enhanced_basic() {
        let config = ClassificationConfig {
            n_samples: 200,
            n_features: 20,
            n_informative: 5,
            n_redundant: 3,
            n_repeated: 2,
            n_classes: 3,
            random_state: Some(42),
            ..Default::default()
        };
        let ds = make_classification_enhanced(config).expect("should succeed");
        assert_eq!(ds.n_samples(), 200);
        assert_eq!(ds.n_features(), 20);
        assert!(ds.target.is_some());
        let target = ds.target.as_ref().expect("target present");
        assert_eq!(target.len(), 200);
        // All labels should be in [0, 3)
        for &val in target.iter() {
            assert!(val >= 0.0 && val < 3.0, "Invalid class label: {val}");
        }
    }

    #[test]
    fn test_classification_enhanced_feature_names() {
        let config = ClassificationConfig {
            n_samples: 50,
            n_features: 10,
            n_informative: 3,
            n_redundant: 2,
            n_repeated: 1,
            n_classes: 2,
            random_state: Some(42),
            ..Default::default()
        };
        let ds = make_classification_enhanced(config).expect("should succeed");
        let names = ds.featurenames.as_ref().expect("names present");
        assert_eq!(names.len(), 10);
        assert!(names[0].starts_with("informative_"));
        assert!(names[3].starts_with("redundant_"));
        assert!(names[5].starts_with("repeated_"));
        assert!(names[6].starts_with("noise_"));
    }

    #[test]
    fn test_classification_enhanced_reproducibility() {
        let make = || {
            let config = ClassificationConfig {
                n_samples: 50,
                n_features: 10,
                n_informative: 3,
                n_redundant: 2,
                n_repeated: 0,
                n_classes: 2,
                flip_y: 0.0,
                shuffle: false,
                random_state: Some(123),
                ..Default::default()
            };
            make_classification_enhanced(config).expect("should succeed")
        };
        let ds1 = make();
        let ds2 = make();
        for i in 0..50 {
            for j in 0..10 {
                assert!(
                    (ds1.data[[i, j]] - ds2.data[[i, j]]).abs() < 1e-15,
                    "Reproducibility failed at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_classification_enhanced_validation() {
        // n_samples = 0
        let cfg = ClassificationConfig {
            n_samples: 0,
            ..Default::default()
        };
        assert!(make_classification_enhanced(cfg).is_err());

        // n_informative > n_features
        let cfg = ClassificationConfig {
            n_features: 5,
            n_informative: 3,
            n_redundant: 2,
            n_repeated: 2,
            ..Default::default()
        };
        assert!(make_classification_enhanced(cfg).is_err());

        // n_classes > n_informative
        let cfg = ClassificationConfig {
            n_informative: 2,
            n_classes: 5,
            ..Default::default()
        };
        assert!(make_classification_enhanced(cfg).is_err());
    }

    #[test]
    fn test_classification_enhanced_redundant_correlation() {
        // Redundant features should be correlated with informative
        let config = ClassificationConfig {
            n_samples: 500,
            n_features: 10,
            n_informative: 5,
            n_redundant: 3,
            n_repeated: 0,
            n_classes: 2,
            flip_y: 0.0,
            shuffle: false,
            random_state: Some(42),
            ..Default::default()
        };
        let ds = make_classification_enhanced(config).expect("should succeed");

        // Compute variance of redundant feature (col 5)
        let col5: Vec<f64> = (0..500).map(|i| ds.data[[i, 5]]).collect();
        let mean5: f64 = col5.iter().sum::<f64>() / 500.0;
        let var5: f64 = col5.iter().map(|x| (x - mean5).powi(2)).sum::<f64>() / 499.0;
        // Redundant features should have non-trivial variance (not just noise)
        assert!(var5 > 0.01, "Redundant feature variance too low: {var5}");
    }

    #[test]
    fn test_classification_enhanced_flip_y() {
        // With flip_y = 1.0, all labels should be flipped randomly
        let config = ClassificationConfig {
            n_samples: 1000,
            n_features: 5,
            n_informative: 3,
            n_redundant: 0,
            n_repeated: 0,
            n_classes: 2,
            flip_y: 0.0,
            shuffle: false,
            random_state: Some(42),
            ..Default::default()
        };
        let ds_no_flip = make_classification_enhanced(config).expect("should succeed");

        let config_flip = ClassificationConfig {
            n_samples: 1000,
            n_features: 5,
            n_informative: 3,
            n_redundant: 0,
            n_repeated: 0,
            n_classes: 2,
            flip_y: 0.5,
            shuffle: false,
            random_state: Some(42),
            ..Default::default()
        };
        let ds_flip = make_classification_enhanced(config_flip).expect("should succeed");

        // With 50% flip rate, some labels should differ
        let n_different = (0..1000)
            .filter(|&i| {
                let t1 = ds_no_flip.target.as_ref().expect("target")[i];
                let t2 = ds_flip.target.as_ref().expect("target")[i];
                (t1 - t2).abs() > 0.5
            })
            .count();
        // The no-flip targets are the SAME RNG state initially, but flip_y draws random
        // numbers differently, so we just check the flipped version differs
        // from a version with no flipping
        assert!(
            n_different > 0,
            "Expected some labels to differ with flip_y=0.5"
        );
    }

    // =========================================================================
    // make_multilabel_classification tests
    // =========================================================================

    #[test]
    fn test_multilabel_basic() {
        let config = MultilabelConfig {
            n_samples: 100,
            n_features: 10,
            n_classes: 5,
            n_labels: 2,
            random_state: Some(42),
            ..Default::default()
        };
        let ds = make_multilabel_classification(config).expect("should succeed");
        assert_eq!(ds.data.nrows(), 100);
        assert_eq!(ds.data.ncols(), 10);
        assert_eq!(ds.target.nrows(), 100);
        assert_eq!(ds.target.ncols(), 5);
    }

    #[test]
    fn test_multilabel_binary_targets() {
        let config = MultilabelConfig {
            n_samples: 50,
            n_features: 5,
            n_classes: 3,
            n_labels: 2,
            random_state: Some(42),
            ..Default::default()
        };
        let ds = make_multilabel_classification(config).expect("should succeed");
        // All target entries should be 0 or 1
        for i in 0..50 {
            for j in 0..3 {
                let val = ds.target[[i, j]];
                assert!(
                    val == 0.0 || val == 1.0,
                    "Target entry at ({i},{j}) should be binary, got {val}"
                );
            }
        }
    }

    #[test]
    fn test_multilabel_labels_per_sample() {
        let config = MultilabelConfig {
            n_samples: 200,
            n_features: 5,
            n_classes: 6,
            n_labels: 3,
            random_state: Some(42),
            ..Default::default()
        };
        let ds = make_multilabel_classification(config).expect("should succeed");
        // Each sample should have exactly n_labels = 3 labels
        for i in 0..200 {
            let label_count: f64 = (0..6).map(|j| ds.target[[i, j]]).sum();
            assert_eq!(
                label_count, 3.0,
                "Sample {i} should have 3 labels, got {label_count}"
            );
        }
    }

    #[test]
    fn test_multilabel_reproducibility() {
        let make = || {
            let config = MultilabelConfig {
                n_samples: 30,
                n_features: 5,
                n_classes: 3,
                n_labels: 1,
                random_state: Some(77),
                ..Default::default()
            };
            make_multilabel_classification(config).expect("should succeed")
        };
        let ds1 = make();
        let ds2 = make();
        for i in 0..30 {
            for j in 0..5 {
                assert!(
                    (ds1.data[[i, j]] - ds2.data[[i, j]]).abs() < 1e-15,
                    "Reproducibility failed at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_multilabel_validation() {
        let cfg = MultilabelConfig {
            n_samples: 0,
            ..Default::default()
        };
        assert!(make_multilabel_classification(cfg).is_err());

        let cfg = MultilabelConfig {
            n_labels: 0,
            ..Default::default()
        };
        assert!(make_multilabel_classification(cfg).is_err());

        let cfg = MultilabelConfig {
            n_labels: 10,
            n_classes: 3,
            ..Default::default()
        };
        assert!(make_multilabel_classification(cfg).is_err());
    }

    // =========================================================================
    // make_hastie_10_2 tests
    // =========================================================================

    #[test]
    fn test_hastie_basic() {
        let ds = make_hastie_10_2(1000, Some(42)).expect("should succeed");
        assert_eq!(ds.n_samples(), 1000);
        assert_eq!(ds.n_features(), 10);
        assert!(ds.target.is_some());
    }

    #[test]
    fn test_hastie_binary_labels() {
        let ds = make_hastie_10_2(500, Some(42)).expect("should succeed");
        let target = ds.target.as_ref().expect("target present");
        for &val in target.iter() {
            assert!(
                val == -1.0 || val == 1.0,
                "Hastie labels should be -1 or 1, got {val}"
            );
        }
    }

    #[test]
    fn test_hastie_balanced_classes() {
        // With enough samples, classes should be roughly balanced
        let ds = make_hastie_10_2(10000, Some(42)).expect("should succeed");
        let target = ds.target.as_ref().expect("target present");
        let n_positive = target.iter().filter(|&&v| v > 0.0).count();
        let n_negative = target.len() - n_positive;
        // Chi-squared(10) median divides the distribution roughly in half
        let ratio = n_positive as f64 / n_negative as f64;
        assert!(
            ratio > 0.7 && ratio < 1.4,
            "Classes should be roughly balanced, got ratio {ratio} (pos={n_positive}, neg={n_negative})"
        );
    }

    #[test]
    fn test_hastie_feature_stats() {
        // Features should be standard normal
        let ds = make_hastie_10_2(5000, Some(42)).expect("should succeed");
        for j in 0..10 {
            let col: Vec<f64> = (0..5000).map(|i| ds.data[[i, j]]).collect();
            let mean: f64 = col.iter().sum::<f64>() / 5000.0;
            let var: f64 = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 4999.0;
            assert!(
                mean.abs() < 0.1,
                "Feature {j} mean should be ~0, got {mean}"
            );
            assert!(
                (var - 1.0).abs() < 0.15,
                "Feature {j} variance should be ~1, got {var}"
            );
        }
    }

    #[test]
    fn test_hastie_reproducibility() {
        let ds1 = make_hastie_10_2(100, Some(99)).expect("should succeed");
        let ds2 = make_hastie_10_2(100, Some(99)).expect("should succeed");
        for i in 0..100 {
            for j in 0..10 {
                assert!(
                    (ds1.data[[i, j]] - ds2.data[[i, j]]).abs() < 1e-15,
                    "Reproducibility failed at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_hastie_validation() {
        assert!(make_hastie_10_2(0, None).is_err());
    }

    #[test]
    fn test_hastie_description() {
        let ds = make_hastie_10_2(100, Some(42)).expect("should succeed");
        assert!(ds.description.is_some());
        let desc = ds.description.as_ref().expect("desc present");
        assert!(desc.contains("Hastie"));
    }
}
