//! Time series dataset generator with concept drift
//!
//! Generates synthetic regression time series where the underlying
//! data-generating process (concept) changes at specified drift points.
//! Supports abrupt, gradual, recurring, and incremental drift types.

/// The type of concept drift to simulate
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DriftType {
    /// The concept switches instantaneously at drift positions
    Abrupt,
    /// The concept transitions smoothly over `drift_width` samples using a sigmoid
    Gradual,
    /// Concepts cycle periodically through the defined concepts (1→2→1→2→...)
    Recurring,
    /// The linear model weights are linearly interpolated over `drift_width` samples
    Incremental,
}

/// Configuration for the concept drift dataset generator
#[derive(Debug, Clone)]
pub struct ConceptDriftConfig {
    /// Total number of time steps
    pub n_samples: usize,
    /// Number of input features
    pub n_features: usize,
    /// Sample indices where drift begins (e.g., [333, 666] for two drifts)
    pub drift_positions: Vec<usize>,
    /// Type of drift transition
    pub drift_type: DriftType,
    /// Width of transition window for Gradual and Incremental drift
    pub drift_width: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for ConceptDriftConfig {
    fn default() -> Self {
        Self {
            n_samples: 1000,
            n_features: 5,
            drift_positions: vec![333, 666],
            drift_type: DriftType::Abrupt,
            drift_width: 50,
            seed: 42,
        }
    }
}

/// Time series dataset with concept drift
#[derive(Debug, Clone)]
pub struct ConceptDriftDataset {
    /// Input feature matrix (n_samples × n_features)
    pub x: Vec<Vec<f64>>,
    /// Regression target (or binary 0/1 if desired)
    pub y: Vec<f64>,
    /// Concept index per sample (0 = first concept, 1 = second, ...)
    pub drift_labels: Vec<usize>,
    /// Indices where drift was intended to start (mirrors config.drift_positions)
    pub true_drift_points: Vec<usize>,
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

/// Sigmoid function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Generate a time series dataset with concept drift.
///
/// The data-generating process consists of `N_concepts = drift_positions.len() + 1`
/// linear models, each with different weight vectors `w_c` and biases `b_c`.
/// The output at time `t` is `y_t = x_t · w_concept(t) + b_concept(t) + noise`.
///
/// # Drift types
///
/// - **Abrupt**: concept switches immediately at each drift position
/// - **Gradual**: uses a sigmoid-weighted mixture of old and new concept over
///   `drift_width` samples
/// - **Recurring**: concepts cycle (1→2→1→2→...) at each drift position
/// - **Incremental**: linearly interpolates weights/bias over `drift_width` samples
///
/// # Arguments
///
/// * `config` - Generator configuration
///
/// # Returns
///
/// A [`ConceptDriftDataset`] with inputs, targets, concept labels, and drift points.
pub fn make_concept_drift(config: &ConceptDriftConfig) -> ConceptDriftDataset {
    let mut rng = Lcg::new(config.seed);
    let n_concepts = config.drift_positions.len() + 1;

    // Generate concept linear models
    // w_concepts[c] is the weight vector for concept c
    let w_concepts: Vec<Vec<f64>> = (0..n_concepts)
        .map(|_| (0..config.n_features).map(|_| rng.next_normal()).collect())
        .collect();
    let b_concepts: Vec<f64> = (0..n_concepts).map(|_| rng.next_normal()).collect();

    // Generate input features X (iid Gaussian)
    let x: Vec<Vec<f64>> = (0..config.n_samples)
        .map(|_| (0..config.n_features).map(|_| rng.next_normal()).collect())
        .collect();

    // Determine the concept index and mixing weights for each sample
    let mut drift_labels = vec![0usize; config.n_samples];
    let mut y = vec![0.0f64; config.n_samples];

    match &config.drift_type {
        DriftType::Abrupt => {
            for t in 0..config.n_samples {
                let concept = concept_index_abrupt(t, &config.drift_positions);
                drift_labels[t] = concept;
                y[t] = dot(&x[t], &w_concepts[concept])
                    + b_concepts[concept]
                    + rng.next_normal() * 0.1;
            }
        }

        DriftType::Gradual => {
            for t in 0..config.n_samples {
                let (c_old, c_new, alpha) = concept_mixing_gradual(
                    t,
                    &config.drift_positions,
                    config.drift_width,
                    n_concepts,
                );
                // alpha = 0 => old, alpha = 1 => new
                drift_labels[t] = if alpha < 0.5 { c_old } else { c_new };
                let y_old = dot(&x[t], &w_concepts[c_old]) + b_concepts[c_old];
                let y_new = dot(&x[t], &w_concepts[c_new]) + b_concepts[c_new];
                y[t] = (1.0 - alpha) * y_old + alpha * y_new + rng.next_normal() * 0.1;
            }
        }

        DriftType::Recurring => {
            // Recurring: concepts cycle 0,1,0,1,... at each drift position
            for t in 0..config.n_samples {
                let segment = segment_index(t, &config.drift_positions);
                // Cycle through available concepts
                let concept = if n_concepts <= 1 {
                    0
                } else {
                    segment % n_concepts
                };
                drift_labels[t] = concept;
                y[t] = dot(&x[t], &w_concepts[concept])
                    + b_concepts[concept]
                    + rng.next_normal() * 0.1;
            }
        }

        DriftType::Incremental => {
            for t in 0..config.n_samples {
                let (c_old, c_new, alpha) = concept_mixing_incremental(
                    t,
                    &config.drift_positions,
                    config.drift_width,
                    n_concepts,
                );
                drift_labels[t] = if alpha < 0.5 { c_old } else { c_new };
                // Linearly interpolate weights
                let w_interp: Vec<f64> = (0..config.n_features)
                    .map(|j| (1.0 - alpha) * w_concepts[c_old][j] + alpha * w_concepts[c_new][j])
                    .collect();
                let b_interp = (1.0 - alpha) * b_concepts[c_old] + alpha * b_concepts[c_new];
                y[t] = dot(&x[t], &w_interp) + b_interp + rng.next_normal() * 0.1;
            }
        }
    }

    ConceptDriftDataset {
        x,
        y,
        drift_labels,
        true_drift_points: config.drift_positions.clone(),
    }
}

/// Compute the concept index for abrupt drift at sample t
fn concept_index_abrupt(t: usize, drift_positions: &[usize]) -> usize {
    let mut concept = 0;
    for &pos in drift_positions {
        if t >= pos {
            concept += 1;
        }
    }
    concept
}

/// Determine the segment index (0-based count of drift points passed)
fn segment_index(t: usize, drift_positions: &[usize]) -> usize {
    drift_positions.iter().filter(|&&pos| t >= pos).count()
}

/// Compute (old_concept, new_concept, alpha) for gradual drift at sample t.
/// alpha is in [0,1]: 0 = fully old concept, 1 = fully new concept.
fn concept_mixing_gradual(
    t: usize,
    drift_positions: &[usize],
    drift_width: usize,
    n_concepts: usize,
) -> (usize, usize, f64) {
    // Find which drift we're nearest
    let mut base_concept = 0usize;
    let mut in_transition = false;
    let mut transition_alpha = 0.0f64;
    let mut next_concept = 1usize;

    for (i, &pos) in drift_positions.iter().enumerate() {
        let new_c = (i + 1).min(n_concepts - 1);
        let old_c = i;
        if t >= pos {
            if drift_width == 0 || t >= pos + drift_width {
                // Past transition
                base_concept = new_c;
            } else {
                // In transition window
                let progress = (t - pos) as f64 / drift_width as f64;
                // sigmoid-based alpha
                let alpha = sigmoid(10.0 * (progress - 0.5));
                in_transition = true;
                transition_alpha = alpha;
                base_concept = old_c;
                next_concept = new_c;
            }
        }
    }

    if in_transition {
        (base_concept, next_concept, transition_alpha)
    } else {
        (base_concept, base_concept, 0.0)
    }
}

/// Compute (old_concept, new_concept, alpha) for incremental drift at sample t.
/// alpha is linearly interpolated in [0,1] over drift_width samples.
fn concept_mixing_incremental(
    t: usize,
    drift_positions: &[usize],
    drift_width: usize,
    n_concepts: usize,
) -> (usize, usize, f64) {
    let mut base_concept = 0usize;
    let mut in_transition = false;
    let mut transition_alpha = 0.0f64;
    let mut next_concept = 1usize;

    for (i, &pos) in drift_positions.iter().enumerate() {
        let new_c = (i + 1).min(n_concepts - 1);
        let old_c = i;
        if t >= pos {
            if drift_width == 0 || t >= pos + drift_width {
                base_concept = new_c;
            } else {
                let alpha = (t - pos) as f64 / drift_width as f64;
                in_transition = true;
                transition_alpha = alpha;
                base_concept = old_c;
                next_concept = new_c;
            }
        }
    }

    if in_transition {
        (base_concept, next_concept, transition_alpha)
    } else {
        (base_concept, base_concept, 0.0)
    }
}

/// Dot product of two equal-length slices
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute the fraction of true drift points detected within a tolerance window.
///
/// A true drift point is considered "detected" if at least one predicted drift
/// point falls within `[true_point - tolerance, true_point + tolerance]`.
///
/// # Arguments
///
/// * `predicted_drift_points` - Predicted drift point indices
/// * `true_drift_points` - Ground-truth drift point indices
/// * `tolerance` - Maximum allowed distance in sample indices
///
/// # Returns
///
/// Fraction of true drift points detected in [0, 1].
pub fn detect_drift_accuracy(
    predicted_drift_points: &[usize],
    true_drift_points: &[usize],
    tolerance: usize,
) -> f64 {
    if true_drift_points.is_empty() {
        return 1.0;
    }
    let detected = true_drift_points
        .iter()
        .filter(|&&tp| {
            predicted_drift_points.iter().any(|&pp| {
                let diff = pp.abs_diff(tp);
                diff <= tolerance
            })
        })
        .count();
    detected as f64 / true_drift_points.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abrupt_drift_labels() {
        let config = ConceptDriftConfig {
            n_samples: 1000,
            n_features: 4,
            drift_positions: vec![500],
            drift_type: DriftType::Abrupt,
            drift_width: 50,
            seed: 42,
        };
        let ds = make_concept_drift(&config);
        assert_eq!(ds.x.len(), 1000);
        assert_eq!(ds.y.len(), 1000);
        assert_eq!(ds.drift_labels.len(), 1000);
        // Before drift at 500: concept 0
        assert_eq!(ds.drift_labels[499], 0, "Sample 499 should be concept 0");
        // At and after drift point 500: concept 1
        assert_eq!(ds.drift_labels[500], 1, "Sample 500 should be concept 1");
    }

    #[test]
    fn test_two_drift_points() {
        let config = ConceptDriftConfig {
            n_samples: 900,
            n_features: 3,
            drift_positions: vec![300, 600],
            drift_type: DriftType::Abrupt,
            drift_width: 0,
            seed: 7,
        };
        let ds = make_concept_drift(&config);
        assert_eq!(ds.drift_labels[0], 0);
        assert_eq!(ds.drift_labels[300], 1);
        assert_eq!(ds.drift_labels[600], 2);
    }

    #[test]
    fn test_gradual_drift() {
        let config = ConceptDriftConfig {
            n_samples: 500,
            n_features: 3,
            drift_positions: vec![200],
            drift_type: DriftType::Gradual,
            drift_width: 50,
            seed: 13,
        };
        let ds = make_concept_drift(&config);
        // Before transition: concept 0
        assert_eq!(ds.drift_labels[0], 0);
        // After full transition: concept 1
        assert_eq!(ds.drift_labels[499], 1);
    }

    #[test]
    fn test_recurring_drift() {
        let config = ConceptDriftConfig {
            n_samples: 600,
            n_features: 3,
            drift_positions: vec![200, 400],
            drift_type: DriftType::Recurring,
            drift_width: 0,
            seed: 99,
        };
        let ds = make_concept_drift(&config);
        // Segment 0: concept 0
        assert_eq!(ds.drift_labels[0], 0);
        // Segment 1 (after pos 200): concept 1
        assert_eq!(ds.drift_labels[200], 1);
        // Segment 2 (after pos 400): concept 2 % 3 = 2
        assert_eq!(ds.drift_labels[400], 2);
    }

    #[test]
    fn test_incremental_drift_monotone() {
        let config = ConceptDriftConfig {
            n_samples: 400,
            n_features: 3,
            drift_positions: vec![150],
            drift_type: DriftType::Incremental,
            drift_width: 100,
            seed: 5,
        };
        let ds = make_concept_drift(&config);
        // Before drift: concept 0
        assert_eq!(ds.drift_labels[0], 0);
        // After full transition: concept 1
        assert_eq!(ds.drift_labels[399], 1);
    }

    #[test]
    fn test_detect_drift_accuracy_perfect() {
        let accuracy = detect_drift_accuracy(&[333, 666], &[333, 666], 0);
        assert!((accuracy - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_detect_drift_accuracy_with_tolerance() {
        let accuracy = detect_drift_accuracy(&[335], &[333], 5);
        assert!(
            (accuracy - 1.0).abs() < 1e-12,
            "Should detect within tolerance"
        );
    }

    #[test]
    fn test_detect_drift_accuracy_miss() {
        let accuracy = detect_drift_accuracy(&[400], &[333], 5);
        assert!((accuracy - 0.0).abs() < 1e-12, "Should miss drift point");
    }
}
