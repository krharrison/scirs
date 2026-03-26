//! High-dimensional sparse classification data generator
//!
//! Generates synthetic classification datasets where only a small subset of
//! features are truly informative, with the remaining features being exactly
//! zero. This simulates text classification, genomics, and other high-dimensional
//! sparse data settings.

/// Configuration for the sparse high-dimensional classification generator
#[derive(Debug, Clone)]
pub struct SparseClassConfig {
    /// Number of training samples
    pub n_samples: usize,
    /// Total number of features (most will be zero)
    pub n_features: usize,
    /// Number of truly informative features
    pub n_informative: usize,
    /// Number of classes
    pub n_classes: usize,
    /// Class separation multiplier applied to class centroids
    pub class_sep: f64,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for SparseClassConfig {
    fn default() -> Self {
        Self {
            n_samples: 1000,
            n_features: 10000,
            n_informative: 20,
            n_classes: 2,
            class_sep: 1.0,
            seed: 42,
        }
    }
}

/// High-dimensional sparse classification dataset
#[derive(Debug, Clone)]
pub struct SparseClassDataset {
    /// Feature matrix (n_samples × n_features); most columns are exactly zero
    pub x: Vec<Vec<f64>>,
    /// Class labels for each sample
    pub y: Vec<usize>,
    /// Indices of the informative features (columns that can be non-zero)
    pub informative_features: Vec<usize>,
    /// Weight vector over features (non-zero only at informative_features)
    pub feature_weights: Vec<f64>,
}

/// Simple seeded LCG PRNG for deterministic generation
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-10);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn next_usize_below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Generate a high-dimensional sparse classification dataset.
///
/// Only `n_informative` out of `n_features` dimensions carry signal. Each
/// class gets a centroid drawn from N(0,1) projected only onto the informative
/// features. Samples are drawn by adding small Gaussian noise to their class
/// centroid; non-informative features remain exactly 0.
///
/// # Arguments
///
/// * `config` - Generator configuration
///
/// # Returns
///
/// A [`SparseClassDataset`] where `x` has shape (n_samples × n_features).
pub fn make_sparse_classification(config: &SparseClassConfig) -> SparseClassDataset {
    let mut rng = Lcg::new(config.seed);

    // --- Select which features are informative ---
    let n_inf = config.n_informative.min(config.n_features);
    let mut informative_features: Vec<usize> = {
        // Fisher-Yates partial shuffle on 0..n_features
        let mut indices: Vec<usize> = (0..config.n_features).collect();
        for i in 0..n_inf {
            let j = i + rng.next_usize_below(config.n_features - i);
            indices.swap(i, j);
        }
        indices[..n_inf].to_vec()
    };
    informative_features.sort_unstable();

    // --- Generate class centroids (only on informative features) ---
    // centroids[c][i] = value at informative_features[i] for class c
    let centroids: Vec<Vec<f64>> = (0..config.n_classes)
        .map(|_| {
            (0..n_inf)
                .map(|_| rng.next_normal() * config.class_sep)
                .collect()
        })
        .collect();

    // --- Feature weight vector (sparse; non-zero only at informative dims) ---
    let mut feature_weights = vec![0.0f64; config.n_features];
    for (idx, &fi) in informative_features.iter().enumerate() {
        // Weight = mean centroid value across classes at this informative dim
        let mean_val: f64 = centroids.iter().map(|c| c[idx]).sum::<f64>() / config.n_classes as f64;
        feature_weights[fi] = mean_val;
    }

    // --- Generate samples ---
    let n_per_class = config.n_samples / config.n_classes;
    let mut x: Vec<Vec<f64>> = Vec::with_capacity(config.n_samples);
    let mut y: Vec<usize> = Vec::with_capacity(config.n_samples);

    for (class_idx, centroid) in centroids.iter().enumerate() {
        // How many samples for this class (last class may get extras)
        let count = if class_idx == config.n_classes - 1 {
            config.n_samples - n_per_class * (config.n_classes - 1)
        } else {
            n_per_class
        };
        for _ in 0..count {
            let mut sample = vec![0.0f64; config.n_features];
            for (inf_idx, &fi) in informative_features.iter().enumerate() {
                // centroid value + small noise on informative dimensions only
                sample[fi] = centroid[inf_idx] + rng.next_normal() * 0.5;
            }
            x.push(sample);
            y.push(class_idx);
        }
    }

    // --- Shuffle samples together ---
    let n = x.len();
    for i in (1..n).rev() {
        let j = rng.next_usize_below(i + 1);
        x.swap(i, j);
        y.swap(i, j);
    }

    SparseClassDataset {
        x,
        y,
        informative_features,
        feature_weights,
    }
}

/// Compute the fraction of exactly-zero entries in feature matrix X.
///
/// High values (> 0.9) indicate high sparsity, expected for datasets
/// generated with many more features than informative ones.
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples × n_features)
///
/// # Returns
///
/// Fraction of zero entries in [0, 1].
pub fn sparsity_ratio(x: &[Vec<f64>]) -> f64 {
    if x.is_empty() {
        return 1.0;
    }
    let n_cols = x[0].len();
    if n_cols == 0 {
        return 1.0;
    }
    let total = (x.len() * n_cols) as f64;
    let zeros = x
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&v| v == 0.0)
        .count() as f64;
    zeros / total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparsity_high() {
        let config = SparseClassConfig {
            n_samples: 200,
            n_features: 1000,
            n_informative: 10,
            n_classes: 2,
            class_sep: 1.0,
            seed: 42,
        };
        let ds = make_sparse_classification(&config);
        let ratio = sparsity_ratio(&ds.x);
        // 10 / 1000 = 1% informative => >98% zeros
        assert!(ratio > 0.98, "Sparsity ratio should be > 0.98, got {ratio}");
    }

    #[test]
    fn test_label_balance() {
        let config = SparseClassConfig {
            n_samples: 100,
            n_features: 500,
            n_informative: 5,
            n_classes: 2,
            class_sep: 1.0,
            seed: 7,
        };
        let ds = make_sparse_classification(&config);
        assert_eq!(ds.y.len(), 100);
        let class0 = ds.y.iter().filter(|&&c| c == 0).count();
        let class1 = ds.y.iter().filter(|&&c| c == 1).count();
        // Both classes should have roughly 50 samples
        assert!((40..=60).contains(&class0), "Class 0 count: {class0}");
        assert!((40..=60).contains(&class1), "Class 1 count: {class1}");
    }

    #[test]
    fn test_informative_feature_count() {
        let config = SparseClassConfig {
            n_samples: 50,
            n_features: 200,
            n_informative: 8,
            n_classes: 3,
            class_sep: 1.5,
            seed: 99,
        };
        let ds = make_sparse_classification(&config);
        assert_eq!(ds.informative_features.len(), 8);
        // All informative indices must be in range
        for &fi in &ds.informative_features {
            assert!(fi < 200, "Informative feature index out of range: {fi}");
        }
    }

    #[test]
    fn test_non_informative_are_zero() {
        let config = SparseClassConfig {
            n_samples: 20,
            n_features: 100,
            n_informative: 5,
            n_classes: 2,
            class_sep: 1.0,
            seed: 13,
        };
        let ds = make_sparse_classification(&config);
        let inf_set: std::collections::HashSet<usize> =
            ds.informative_features.iter().copied().collect();
        for row in &ds.x {
            for (j, &val) in row.iter().enumerate() {
                if !inf_set.contains(&j) {
                    assert_eq!(val, 0.0, "Non-informative feature {j} should be zero");
                }
            }
        }
    }

    #[test]
    fn test_default_config_shape() {
        let config = SparseClassConfig {
            n_samples: 50,
            n_features: 200,
            n_informative: 10,
            ..Default::default()
        };
        let ds = make_sparse_classification(&config);
        assert_eq!(ds.x.len(), 50);
        assert_eq!(ds.x[0].len(), 200);
    }
}
