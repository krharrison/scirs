//! Advanced multi-label classification data generator with label dependencies
//!
//! Generates synthetic multi-label datasets using a latent factor model.
//! Label correlations arise naturally from shared latent structure, producing
//! more realistic co-occurrence patterns than independently sampling labels.

/// Configuration for the advanced multi-label classification generator
#[derive(Debug, Clone)]
pub struct AdvancedMultilabelConfig {
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Number of output labels
    pub n_labels: usize,
    /// Expected fraction of positive labels per sample
    pub label_density: f64,
    /// Dimensionality of the latent space driving label correlations
    pub n_latent: usize,
    /// Whether to allow samples with no positive labels
    pub allow_unlabeled: bool,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for AdvancedMultilabelConfig {
    fn default() -> Self {
        Self {
            n_samples: 500,
            n_features: 20,
            n_labels: 5,
            label_density: 0.3,
            n_latent: 10,
            allow_unlabeled: true,
            seed: 42,
        }
    }
}

/// Advanced multi-label classification dataset
#[derive(Debug, Clone)]
pub struct AdvancedMultilabelDataset {
    /// Feature matrix (n_samples × n_features)
    pub x: Vec<Vec<f64>>,
    /// Label matrix (n_samples × n_labels); true = label is active
    pub y: Vec<Vec<bool>>,
    /// Label co-occurrence frequency matrix (n_labels × n_labels)
    pub label_cooccurrence: Vec<Vec<f64>>,
    /// Mean number of active labels per sample
    pub cardinality: f64,
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
}

/// Sigmoid activation function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Generate an advanced multi-label classification dataset with label correlations.
///
/// The generative model:
/// - W ∈ R^{n_latent × n_features}: feature generation matrix
/// - L ∈ R^{n_labels × n_latent}: label-latent mapping
/// - bias ∈ R^{n_labels}: per-label bias controlling label density
/// - For each sample: z ~ N(0,I) in latent space
///   - x = W^T z + noise
///   - p_k = sigmoid(`L[k,:]` · z + bias_k)
///   - y_k ~ Bernoulli(p_k)
///
/// # Arguments
///
/// * `config` - Generator configuration
///
/// # Returns
///
/// An [`AdvancedMultilabelDataset`] with correlated labels.
pub fn make_advanced_multilabel_classification(
    config: &AdvancedMultilabelConfig,
) -> AdvancedMultilabelDataset {
    let mut rng = Lcg::new(config.seed);
    let n_lat = config.n_latent.max(1);

    // --- Feature generation matrix W (n_latent × n_features) ---
    let w_mat: Vec<Vec<f64>> = (0..n_lat)
        .map(|_| (0..config.n_features).map(|_| rng.next_normal()).collect())
        .collect();

    // --- Label-latent mapping L (n_labels × n_latent) ---
    let l_mat: Vec<Vec<f64>> = (0..config.n_labels)
        .map(|_| (0..n_lat).map(|_| rng.next_normal()).collect())
        .collect();

    // --- Per-label biases calibrated to hit target label_density ---
    // logit(p) = L·z + b; for z=0 bias controls marginal probability
    // logit(label_density) drives the base probability
    let target_logit = {
        let d = config.label_density.clamp(1e-6, 1.0 - 1e-6);
        (d / (1.0 - d)).ln()
    };
    let biases: Vec<f64> = (0..config.n_labels)
        .map(|_| target_logit + rng.next_normal() * 0.2)
        .collect();

    // --- Generate samples ---
    let mut x_all: Vec<Vec<f64>> = Vec::with_capacity(config.n_samples);
    let mut y_all: Vec<Vec<bool>> = Vec::with_capacity(config.n_samples);

    let mut generated = 0;
    let mut attempts = 0;
    let max_attempts = config.n_samples * 10 + 100;

    while generated < config.n_samples && attempts < max_attempts {
        attempts += 1;

        // Latent vector z ~ N(0, I)
        let z: Vec<f64> = (0..n_lat).map(|_| rng.next_normal()).collect();

        // Feature vector x = W^T z + small noise
        let sample_x: Vec<f64> = (0..config.n_features)
            .map(|j| {
                let val: f64 = (0..n_lat).map(|k| w_mat[k][j] * z[k]).sum();
                val + rng.next_normal() * 0.1
            })
            .collect();

        // Label probabilities
        let probs: Vec<f64> = (0..config.n_labels)
            .map(|k| {
                let logit: f64 = (0..n_lat).map(|d| l_mat[k][d] * z[d]).sum::<f64>() + biases[k];
                sigmoid(logit)
            })
            .collect();

        // Sample labels
        let labels: Vec<bool> = probs.iter().map(|&p| rng.next_f64() < p).collect();

        // If allow_unlabeled=false, skip samples with no active labels
        let any_active = labels.iter().any(|&b| b);
        if !config.allow_unlabeled && !any_active {
            continue;
        }

        x_all.push(sample_x);
        y_all.push(labels);
        generated += 1;
    }

    // --- Label co-occurrence matrix ---
    let n_actual = y_all.len();
    let mut cooccur = vec![vec![0.0f64; config.n_labels]; config.n_labels];
    for labels in &y_all {
        for k1 in 0..config.n_labels {
            for k2 in 0..config.n_labels {
                if labels[k1] && labels[k2] {
                    cooccur[k1][k2] += 1.0;
                }
            }
        }
    }
    if n_actual > 0 {
        for row in &mut cooccur {
            for val in row.iter_mut() {
                *val /= n_actual as f64;
            }
        }
    }

    let cardinality = label_cardinality_impl(&y_all);

    AdvancedMultilabelDataset {
        x: x_all,
        y: y_all,
        label_cooccurrence: cooccur,
        cardinality,
    }
}

fn label_cardinality_impl(y: &[Vec<bool>]) -> f64 {
    if y.is_empty() {
        return 0.0;
    }
    let total: usize = y.iter().map(|row| row.iter().filter(|&&b| b).count()).sum();
    total as f64 / y.len() as f64
}

/// Compute the mean number of active labels per sample.
///
/// # Arguments
///
/// * `y` - Label matrix (n_samples × n_labels)
///
/// # Returns
///
/// Mean active labels per sample (label cardinality).
pub fn label_cardinality(y: &[Vec<bool>]) -> f64 {
    label_cardinality_impl(y)
}

/// Compute the fraction of positive labels across all samples and labels.
///
/// # Arguments
///
/// * `y` - Label matrix (n_samples × n_labels)
///
/// # Returns
///
/// Global label density in [0, 1].
pub fn label_density_score(y: &[Vec<bool>]) -> f64 {
    if y.is_empty() {
        return 0.0;
    }
    let n_labels = y[0].len();
    if n_labels == 0 {
        return 0.0;
    }
    let total = (y.len() * n_labels) as f64;
    let positives: usize = y.iter().flat_map(|row| row.iter()).filter(|&&b| b).count();
    positives as f64 / total
}

/// Compute the Hamming loss between two label matrices.
///
/// Hamming loss is the fraction of labels that are incorrectly predicted,
/// averaged over all samples and labels.
///
/// # Arguments
///
/// * `y_true` - Ground truth label matrix
/// * `y_pred` - Predicted label matrix
///
/// # Returns
///
/// Hamming loss in [0, 1].
pub fn hamming_loss(y_true: &[Vec<bool>], y_pred: &[Vec<bool>]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let n_labels = y_true[0].len();
    if n_labels == 0 {
        return 0.0;
    }
    let total = (y_true.len() * n_labels) as f64;
    let wrong: usize = y_true
        .iter()
        .zip(y_pred.iter())
        .flat_map(|(r_t, r_p)| r_t.iter().zip(r_p.iter()))
        .filter(|(&t, &p)| t != p)
        .count();
    wrong as f64 / total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cardinality_near_density() {
        let config = AdvancedMultilabelConfig {
            n_samples: 1000,
            n_features: 15,
            n_labels: 5,
            label_density: 0.3,
            n_latent: 8,
            allow_unlabeled: true,
            seed: 42,
        };
        let ds = make_advanced_multilabel_classification(&config);
        // Expected cardinality ≈ label_density * n_labels = 0.3 * 5 = 1.5
        // Allow generous tolerance due to latent-space correlations
        assert!(
            ds.cardinality > 0.5,
            "Cardinality too low: {}",
            ds.cardinality
        );
        assert!(
            ds.cardinality < 4.5,
            "Cardinality too high: {}",
            ds.cardinality
        );
    }

    #[test]
    fn test_output_shapes() {
        let config = AdvancedMultilabelConfig {
            n_samples: 100,
            n_features: 10,
            n_labels: 4,
            label_density: 0.4,
            n_latent: 5,
            allow_unlabeled: true,
            seed: 7,
        };
        let ds = make_advanced_multilabel_classification(&config);
        assert_eq!(ds.x.len(), 100);
        assert_eq!(ds.x[0].len(), 10);
        assert_eq!(ds.y.len(), 100);
        assert_eq!(ds.y[0].len(), 4);
        assert_eq!(ds.label_cooccurrence.len(), 4);
        assert_eq!(ds.label_cooccurrence[0].len(), 4);
    }

    #[test]
    fn test_hamming_loss_self_zero() {
        let y = vec![
            vec![true, false, true],
            vec![false, false, true],
            vec![true, true, false],
        ];
        assert!((hamming_loss(&y, &y) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_hamming_loss_all_wrong() {
        let y_true = vec![vec![true, true], vec![false, false]];
        let y_pred = vec![vec![false, false], vec![true, true]];
        let loss = hamming_loss(&y_true, &y_pred);
        assert!((loss - 1.0).abs() < 1e-12, "Expected 1.0, got {loss}");
    }

    #[test]
    fn test_label_density_score() {
        let y = vec![
            vec![true, false, false, false],
            vec![false, false, false, false],
        ];
        let d = label_density_score(&y);
        // 1 out of 8 entries are positive
        assert!((d - 0.125).abs() < 1e-12, "Expected 0.125, got {d}");
    }

    #[test]
    fn test_no_unlabeled_when_disabled() {
        let config = AdvancedMultilabelConfig {
            n_samples: 200,
            n_features: 10,
            n_labels: 3,
            label_density: 0.5,
            n_latent: 5,
            allow_unlabeled: false,
            seed: 55,
        };
        let ds = make_advanced_multilabel_classification(&config);
        for (i, labels) in ds.y.iter().enumerate() {
            let any = labels.iter().any(|&b| b);
            assert!(
                any,
                "Sample {i} has no active labels but allow_unlabeled=false"
            );
        }
    }
}
