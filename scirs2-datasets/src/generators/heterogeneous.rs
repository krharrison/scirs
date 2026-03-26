//! Mixed feature type dataset generator
//!
//! Generates synthetic datasets combining continuous, categorical, ordinal,
//! and binary features. Supports class-conditional generation so each class
//! has different feature distributions, enabling classification benchmarks
//! on heterogeneous tabular data.

/// Describes the type and parameters of a single feature column.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum FeatureType {
    /// Continuous Gaussian feature with given (mean, std)
    Continuous(f64, f64),
    /// Categorical feature with the given number of categories
    Categorical(usize),
    /// Ordinal feature with the given number of levels (0 ..= n_levels-1)
    Ordinal(usize),
    /// Binary Bernoulli feature
    Binary,
}

/// Configuration for the heterogeneous dataset generator
#[derive(Debug, Clone)]
pub struct HeteroConfig {
    /// Number of samples
    pub n_samples: usize,
    /// Explicit feature types; if empty, types are generated automatically
    pub feature_types: Vec<FeatureType>,
    /// Number of features to auto-generate when `feature_types` is empty
    pub n_features: usize,
    /// Number of output classes
    pub n_classes: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for HeteroConfig {
    fn default() -> Self {
        Self {
            n_samples: 500,
            feature_types: Vec::new(),
            n_features: 10,
            n_classes: 2,
            seed: 42,
        }
    }
}

/// A single feature value that may be continuous, integer, or boolean
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum HeteroFeatureValue {
    /// Floating-point (continuous) value
    Float(f64),
    /// Non-negative integer (categorical or ordinal level)
    Int(usize),
    /// Boolean (binary feature)
    Bool(bool),
}

/// Heterogeneous (mixed feature type) classification dataset
#[derive(Debug, Clone)]
pub struct HeteroDataset {
    /// Feature matrix: each row is a sample, each column is a feature
    pub features: Vec<Vec<HeteroFeatureValue>>,
    /// Class label for each sample
    pub labels: Vec<usize>,
    /// Type of each feature column
    pub feature_types: Vec<FeatureType>,
    /// Human-readable name for each feature
    pub feature_names: Vec<String>,
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

/// Auto-generate a list of feature types by cycling through Continuous/Categorical/Ordinal/Binary
fn auto_feature_types(n: usize, rng: &mut Lcg) -> Vec<FeatureType> {
    (0..n)
        .map(|i| match i % 4 {
            0 => {
                let mean = rng.next_normal();
                let std = 0.5 + rng.next_f64();
                FeatureType::Continuous(mean, std)
            }
            1 => {
                let n_cats = 2 + rng.next_usize_below(5); // 2..6
                FeatureType::Categorical(n_cats)
            }
            2 => {
                let n_levels = 3 + rng.next_usize_below(5); // 3..7
                FeatureType::Ordinal(n_levels)
            }
            _ => FeatureType::Binary,
        })
        .collect()
}

/// Generate a heterogeneous (mixed feature type) classification dataset.
///
/// Each class receives different feature distributions:
/// - **Continuous**: each class shifts the mean by a class-specific offset
/// - **Categorical**: each class uses a different Dirichlet-like distribution
/// - **Ordinal**: each class shifts the modal level
/// - **Binary**: each class uses a different Bernoulli probability
///
/// If `config.feature_types` is empty, feature types are auto-assigned in a
/// Continuous/Categorical/Ordinal/Binary pattern.
///
/// # Arguments
///
/// * `config` - Generator configuration
///
/// # Returns
///
/// A [`HeteroDataset`] with mixed feature types and class labels.
pub fn make_heterogeneous(config: &HeteroConfig) -> HeteroDataset {
    let mut rng = Lcg::new(config.seed);

    // Resolve feature types
    let feature_types: Vec<FeatureType> = if config.feature_types.is_empty() {
        auto_feature_types(config.n_features, &mut rng)
    } else {
        config.feature_types.clone()
    };
    let n_features = feature_types.len();

    // Feature names
    let feature_names: Vec<String> = feature_types
        .iter()
        .enumerate()
        .map(|(i, ft)| match ft {
            FeatureType::Continuous(_, _) => format!("cont_{i}"),
            FeatureType::Categorical(_) => format!("cat_{i}"),
            FeatureType::Ordinal(_) => format!("ord_{i}"),
            FeatureType::Binary => format!("bin_{i}"),
        })
        .collect();

    // Per-class, per-feature parameters
    // continuous: class mean offset
    let class_cont_offsets: Vec<Vec<f64>> = (0..config.n_classes)
        .map(|_| (0..n_features).map(|_| rng.next_normal() * 1.5).collect())
        .collect();

    // categorical: per-class probability vector over categories
    let class_cat_probs: Vec<Vec<Vec<f64>>> = (0..config.n_classes)
        .map(|_| {
            feature_types
                .iter()
                .map(|ft| match ft {
                    FeatureType::Categorical(k) | FeatureType::Ordinal(k) => {
                        let mut weights: Vec<f64> = (0..*k).map(|_| rng.next_f64() + 0.1).collect();
                        let s: f64 = weights.iter().sum();
                        for w in &mut weights {
                            *w /= s;
                        }
                        weights
                    }
                    _ => vec![0.5, 0.5], // placeholder for non-categorical
                })
                .collect()
        })
        .collect();

    // binary: per-class Bernoulli probability
    let class_bin_probs: Vec<Vec<f64>> = (0..config.n_classes)
        .map(|_| {
            (0..n_features)
                .map(|_| 0.1 + rng.next_f64() * 0.8)
                .collect()
        })
        .collect();

    // Generate balanced samples
    let n_per_class = config.n_samples / config.n_classes;
    let mut features: Vec<Vec<HeteroFeatureValue>> = Vec::with_capacity(config.n_samples);
    let mut labels: Vec<usize> = Vec::with_capacity(config.n_samples);

    for class_idx in 0..config.n_classes {
        let count = if class_idx == config.n_classes - 1 {
            config.n_samples - n_per_class * (config.n_classes - 1)
        } else {
            n_per_class
        };
        for _ in 0..count {
            let row: Vec<HeteroFeatureValue> = feature_types
                .iter()
                .enumerate()
                .map(|(j, ft)| match ft {
                    FeatureType::Continuous(mean, std) => {
                        let offset = class_cont_offsets[class_idx][j];
                        let val = (mean + offset) + rng.next_normal() * std;
                        HeteroFeatureValue::Float(val)
                    }
                    FeatureType::Categorical(k) => {
                        let probs = &class_cat_probs[class_idx][j];
                        let u = rng.next_f64();
                        let mut cumsum = 0.0;
                        let mut cat = 0;
                        for (idx, &p) in probs.iter().enumerate() {
                            cumsum += p;
                            if u < cumsum {
                                cat = idx;
                                break;
                            }
                            cat = k - 1; // fallback
                        }
                        HeteroFeatureValue::Int(cat)
                    }
                    FeatureType::Ordinal(k) => {
                        let probs = &class_cat_probs[class_idx][j];
                        let u = rng.next_f64();
                        let mut cumsum = 0.0;
                        let mut level = 0;
                        for (idx, &p) in probs.iter().enumerate() {
                            cumsum += p;
                            if u < cumsum {
                                level = idx;
                                break;
                            }
                            level = k - 1;
                        }
                        HeteroFeatureValue::Int(level)
                    }
                    FeatureType::Binary => {
                        let p = class_bin_probs[class_idx][j];
                        HeteroFeatureValue::Bool(rng.next_f64() < p)
                    }
                })
                .collect();
            features.push(row);
            labels.push(class_idx);
        }
    }

    // Shuffle samples
    let n = features.len();
    for i in (1..n).rev() {
        let j = rng.next_usize_below(i + 1);
        features.swap(i, j);
        labels.swap(i, j);
    }

    HeteroDataset {
        features,
        labels,
        feature_types,
        feature_names,
    }
}

/// One-hot encode all features in a heterogeneous dataset to a flat f64 vector.
///
/// Encoding rules:
/// - `Float(v)` → `[v]` (pass through as single value)
/// - `Int(k)` with categorical having `n` categories → one-hot vector of length `n`
/// - `Int(l)` with ordinal having `n_levels` → one-hot vector of length `n_levels`
/// - `Bool(b)` → `[0.0]` or `[1.0]`
///
/// The output vector per sample is wider than `n_features` whenever categorical
/// or ordinal features have more than one category/level.
///
/// # Arguments
///
/// * `dataset` - The heterogeneous dataset to encode
///
/// # Returns
///
/// Dense f64 feature matrix with one-hot encoded categorical/ordinal features.
pub fn encode_one_hot(dataset: &HeteroDataset) -> Vec<Vec<f64>> {
    // Precompute the encoded widths for each feature
    let widths: Vec<usize> = dataset
        .feature_types
        .iter()
        .map(|ft| match ft {
            FeatureType::Continuous(_, _) => 1,
            FeatureType::Categorical(k) => *k,
            FeatureType::Ordinal(k) => *k,
            FeatureType::Binary => 1,
        })
        .collect();

    let total_width: usize = widths.iter().sum();

    dataset
        .features
        .iter()
        .map(|row| {
            let mut encoded = Vec::with_capacity(total_width);
            for (j, val) in row.iter().enumerate() {
                match (&dataset.feature_types[j], val) {
                    (FeatureType::Continuous(_, _), HeteroFeatureValue::Float(v)) => {
                        encoded.push(*v);
                    }
                    (FeatureType::Categorical(k), HeteroFeatureValue::Int(cat)) => {
                        for c in 0..*k {
                            encoded.push(if c == *cat { 1.0 } else { 0.0 });
                        }
                    }
                    (FeatureType::Ordinal(k), HeteroFeatureValue::Int(level)) => {
                        for l in 0..*k {
                            encoded.push(if l == *level { 1.0 } else { 0.0 });
                        }
                    }
                    (FeatureType::Binary, HeteroFeatureValue::Bool(b)) => {
                        encoded.push(if *b { 1.0 } else { 0.0 });
                    }
                    // Fallback for unexpected type combinations
                    (_, HeteroFeatureValue::Float(v)) => encoded.push(*v),
                    (_, HeteroFeatureValue::Int(k)) => encoded.push(*k as f64),
                    (_, HeteroFeatureValue::Bool(b)) => encoded.push(if *b { 1.0 } else { 0.0 }),
                }
            }
            encoded
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heterogeneous_basic() {
        let config = HeteroConfig {
            n_samples: 50,
            feature_types: Vec::new(),
            n_features: 8,
            n_classes: 2,
            seed: 42,
        };
        let ds = make_heterogeneous(&config);
        assert_eq!(ds.features.len(), 50);
        assert_eq!(ds.labels.len(), 50);
        assert_eq!(ds.feature_types.len(), 8);
        assert_eq!(ds.feature_names.len(), 8);
    }

    #[test]
    fn test_encode_one_hot_wider() {
        let config = HeteroConfig {
            n_samples: 20,
            feature_types: vec![
                FeatureType::Continuous(0.0, 1.0),
                FeatureType::Categorical(4),
                FeatureType::Ordinal(3),
                FeatureType::Binary,
            ],
            n_features: 4,
            n_classes: 2,
            seed: 77,
        };
        let ds = make_heterogeneous(&config);
        let encoded = encode_one_hot(&ds);
        // 1 + 4 + 3 + 1 = 9 columns after one-hot
        assert_eq!(
            encoded[0].len(),
            9,
            "Expected 9 columns after one-hot encoding"
        );
        // n_features = 4 < 9 (one-hot is wider)
        assert!(encoded[0].len() > config.n_features);
    }

    #[test]
    fn test_explicit_feature_types() {
        let config = HeteroConfig {
            n_samples: 30,
            feature_types: vec![
                FeatureType::Continuous(2.0, 0.5),
                FeatureType::Categorical(3),
                FeatureType::Binary,
            ],
            n_features: 3,
            n_classes: 2,
            seed: 1,
        };
        let ds = make_heterogeneous(&config);
        // Check continuous feature produces floats
        for row in &ds.features {
            assert!(matches!(row[0], HeteroFeatureValue::Float(_)));
            assert!(matches!(row[1], HeteroFeatureValue::Int(_)));
            assert!(matches!(row[2], HeteroFeatureValue::Bool(_)));
        }
    }

    #[test]
    fn test_label_range() {
        let config = HeteroConfig {
            n_samples: 60,
            feature_types: Vec::new(),
            n_features: 6,
            n_classes: 3,
            seed: 5,
        };
        let ds = make_heterogeneous(&config);
        for &label in &ds.labels {
            assert!(label < 3, "Label {label} out of range [0, 3)");
        }
    }
}
