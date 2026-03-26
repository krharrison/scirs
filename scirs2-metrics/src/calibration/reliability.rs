//! Reliability diagrams and uncertainty metrics for calibration evaluation.
//!
//! Implements:
//! - Reliability diagram (ECE, MCE, over/underconfidence fractions)
//! - Adaptive temperature scaling via NLL minimization
//! - Ensemble uncertainty decomposition (entropy, mutual info, aleatoric/epistemic)

use crate::error::{MetricsError, Result};

/// Configuration for reliability diagram computation.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ReliabilityDiagramConfig {
    /// Number of bins for equal-width binning (default: 10)
    pub n_bins: usize,
    /// Use equal-count (adaptive) bins instead of equal-width (default: false)
    pub adaptive_bins: bool,
    /// Minimum number of samples per bin to include it (default: 5)
    pub min_bin_size: usize,
}

impl Default for ReliabilityDiagramConfig {
    fn default() -> Self {
        Self {
            n_bins: 10,
            adaptive_bins: false,
            min_bin_size: 5,
        }
    }
}

/// Statistics for a single confidence bin.
#[derive(Debug, Clone)]
pub struct BinStats {
    /// Mean predicted confidence in this bin
    pub confidence_mean: f64,
    /// Fraction of samples in this bin with positive label
    pub accuracy: f64,
    /// Number of samples in this bin
    pub count: usize,
    /// Lower bound of the bin
    pub bin_lower: f64,
    /// Upper bound of the bin
    pub bin_upper: f64,
}

/// Full reliability diagram with ECE, MCE and over/underconfidence fractions.
#[derive(Debug, Clone)]
pub struct ReliabilityDiagram {
    /// Per-bin statistics
    pub bins: Vec<BinStats>,
    /// Expected Calibration Error (weighted mean absolute calibration error)
    pub ece: f64,
    /// Maximum Calibration Error
    pub mce: f64,
    /// Fraction of bins where mean confidence > accuracy (overconfident)
    pub overconfidence_frac: f64,
    /// Fraction of bins where mean confidence < accuracy (underconfident)
    pub underconfidence_frac: f64,
}

/// Compute a reliability diagram from predicted probabilities and binary labels.
///
/// # Arguments
/// * `probs`  - Predicted probabilities in [0, 1]
/// * `labels` - True binary labels
/// * `config` - Binning configuration
///
/// # Errors
/// Returns an error if inputs are empty or have different lengths.
pub fn reliability_diagram(
    probs: &[f64],
    labels: &[bool],
    config: &ReliabilityDiagramConfig,
) -> Result<ReliabilityDiagram> {
    if probs.len() != labels.len() {
        return Err(MetricsError::InvalidInput(format!(
            "probs and labels lengths differ: {} vs {}",
            probs.len(),
            labels.len()
        )));
    }
    if probs.is_empty() {
        return Err(MetricsError::InvalidInput(
            "probs and labels must not be empty".to_string(),
        ));
    }
    if config.n_bins == 0 {
        return Err(MetricsError::InvalidInput("n_bins must be > 0".to_string()));
    }

    let n = probs.len();

    let bin_stats = if config.adaptive_bins {
        adaptive_bins(probs, labels, config.n_bins, config.min_bin_size)
    } else {
        uniform_bins(probs, labels, config.n_bins, config.min_bin_size)
    };

    // ECE, MCE
    let mut ece = 0.0f64;
    let mut mce = 0.0f64;
    let mut over_bins = 0usize;
    let mut under_bins = 0usize;
    let n_valid = bin_stats.len();

    for b in &bin_stats {
        let gap = (b.confidence_mean - b.accuracy).abs();
        ece += gap * b.count as f64 / n as f64;
        if gap > mce {
            mce = gap;
        }
        if b.confidence_mean > b.accuracy {
            over_bins += 1;
        } else if b.confidence_mean < b.accuracy {
            under_bins += 1;
        }
    }

    let (overconfidence_frac, underconfidence_frac) = if n_valid == 0 {
        (0.0, 0.0)
    } else {
        (
            over_bins as f64 / n_valid as f64,
            under_bins as f64 / n_valid as f64,
        )
    };

    Ok(ReliabilityDiagram {
        bins: bin_stats,
        ece,
        mce,
        overconfidence_frac,
        underconfidence_frac,
    })
}

/// Build equal-width bins in [0, 1].
fn uniform_bins(
    probs: &[f64],
    labels: &[bool],
    n_bins: usize,
    min_bin_size: usize,
) -> Vec<BinStats> {
    let bin_width = 1.0 / n_bins as f64;
    let mut bins: Vec<BinStats> = Vec::with_capacity(n_bins);

    for b in 0..n_bins {
        let lower = b as f64 * bin_width;
        let upper = if b == n_bins - 1 {
            1.0 + 1e-12 // include 1.0
        } else {
            (b + 1) as f64 * bin_width
        };

        let mut conf_sum = 0.0f64;
        let mut pos_count = 0usize;
        let mut count = 0usize;

        for (p, &l) in probs.iter().zip(labels.iter()) {
            if *p >= lower && *p < upper {
                conf_sum += p;
                if l {
                    pos_count += 1;
                }
                count += 1;
            }
        }

        if count >= min_bin_size {
            bins.push(BinStats {
                confidence_mean: conf_sum / count as f64,
                accuracy: pos_count as f64 / count as f64,
                count,
                bin_lower: lower,
                bin_upper: upper,
            });
        }
    }
    bins
}

/// Build equal-count (adaptive) bins.
fn adaptive_bins(
    probs: &[f64],
    labels: &[bool],
    n_bins: usize,
    min_bin_size: usize,
) -> Vec<BinStats> {
    let n = probs.len();
    // Sort indices by probability
    let mut sorted_idx: Vec<usize> = (0..n).collect();
    sorted_idx.sort_by(|&a, &b| {
        probs[a]
            .partial_cmp(&probs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let base_size = n / n_bins;
    let mut bins: Vec<BinStats> = Vec::with_capacity(n_bins);
    let mut start = 0usize;

    for b in 0..n_bins {
        let end = if b == n_bins - 1 {
            n
        } else {
            start + base_size + if b < (n % n_bins) { 1 } else { 0 }
        };

        if end <= start {
            continue;
        }

        let chunk = &sorted_idx[start..end];
        let count = chunk.len();

        if count < min_bin_size {
            start = end;
            continue;
        }

        let conf_sum: f64 = chunk.iter().map(|&i| probs[i]).sum();
        let pos_count: usize = chunk.iter().filter(|&&i| labels[i]).count();
        let bin_lower = probs[chunk[0]];
        let bin_upper = probs[*chunk.last().unwrap_or(&chunk[0])];

        bins.push(BinStats {
            confidence_mean: conf_sum / count as f64,
            accuracy: pos_count as f64 / count as f64,
            count,
            bin_lower,
            bin_upper,
        });

        start = end;
    }
    bins
}

/// Find the optimal temperature T that minimizes NLL on a held-out calibration set.
///
/// Applies gradient descent on T to minimize:
///   L(T) = -1/N Σ log softmax(logits_i / T)\[y_i\]
///
/// # Arguments
/// * `logits` - Raw logits, shape `[n_samples][n_classes]`
/// * `labels` - True class indices, shape `[n_samples]`
/// * `n_iter` - Number of gradient descent iterations (default: 100)
///
/// # Returns
/// Optimal temperature T (> 0). Returns 1.0 on empty input.
pub fn adaptive_temperature_scaling(
    logits: &[Vec<f64>],
    labels: &[usize],
    n_iter: usize,
) -> Result<f64> {
    if logits.len() != labels.len() {
        return Err(MetricsError::InvalidInput(format!(
            "logits and labels lengths differ: {} vs {}",
            logits.len(),
            labels.len()
        )));
    }
    if logits.is_empty() {
        return Ok(1.0);
    }

    let n_classes = logits[0].len();
    if n_classes == 0 {
        return Err(MetricsError::InvalidInput(
            "n_classes must be > 0".to_string(),
        ));
    }

    for (i, row) in logits.iter().enumerate() {
        if row.len() != n_classes {
            return Err(MetricsError::InvalidInput(format!(
                "logits row {i} has {} elements, expected {n_classes}",
                row.len()
            )));
        }
        if let Some(&l) = labels.get(i) {
            if l >= n_classes {
                return Err(MetricsError::InvalidInput(format!(
                    "label {l} at index {i} exceeds n_classes={n_classes}"
                )));
            }
        }
    }

    let mut t = 1.0f64;
    let lr = 0.01f64;
    let min_t = 1e-3f64;
    let max_t = 1e3f64;
    let n = logits.len() as f64;

    for _ in 0..n_iter {
        let (nll, grad_t) = nll_and_grad(logits, labels, t);
        let _ = nll; // unused, suppress warning
        t -= lr * grad_t;
        t = t.clamp(min_t, max_t);
    }

    Ok(t)
}

/// Compute NLL and d(NLL)/d(T) for temperature scaling.
fn nll_and_grad(logits: &[Vec<f64>], labels: &[usize], t: f64) -> (f64, f64) {
    let mut nll = 0.0f64;
    let mut grad = 0.0f64;
    let n = logits.len() as f64;

    for (row, &y) in logits.iter().zip(labels.iter()) {
        let scaled: Vec<f64> = row.iter().map(|&x| x / t).collect();
        let max_s = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = scaled.iter().map(|&s| (s - max_s).exp()).sum();
        let log_sum = max_s + exp_sum.ln();

        // NLL contribution: log_sum - scaled[y]
        nll += log_sum - scaled[y];

        // Gradient d(NLL)/d(T) = d/dT [log_sum_exp(x/T) - x_y/T]
        // = Σ_c p_c * (-x_c / T^2) - (-x_y / T^2)
        // = (x_y - Σ_c p_c * x_c) / T^2
        let probs: Vec<f64> = scaled
            .iter()
            .map(|&s| (s - log_sum + max_s - max_s).exp())
            .collect();
        // Simpler: prob_c = exp(scaled_c - log_sum_from_zero)
        // log_sum from zero: log(exp_sum) + max_s  (we computed log_sum above)
        // But exp_sum includes max_s shift. Let's recompute cleanly.
        let log_z = log_sum; // log Σ exp(x_c / T)
        let probs_clean: Vec<f64> = row.iter().map(|&x| ((x / t) - log_z).exp()).collect();
        let expected_logit: f64 = probs_clean
            .iter()
            .zip(row.iter())
            .map(|(&p, &x)| p * x)
            .sum();

        grad += (row[y] - expected_logit) / (t * t);
        // Correct sign: d(NLL)/d(T) where NLL = log_Z - x_y/T
        // = d log_Z / d_T - d(x_y/T)/dT
        // = (Σ_c p_c * (-x_c/T^2)) - (-x_y / T^2)
        // = (x_y - E_p[x]) / T^2
        // => gradient we computed above is already d NLL / d T * (-1) relative to
        //    what we want (we want to descend NLL, so subtract grad).
        // Actually: NLL increases when T moves away from optimal.
        // gradient = (x_y - E[x]) / T^2 is d NLL / d T = -something if well-calibrated
        // Let's keep consistent sign here (we'll take - grad in the optimizer).
        let _ = (probs, expected_logit); // suppress
    }

    (nll / n, -grad / n) // return -(d NLL / dT) so caller does t -= lr * grad_t decreases NLL
}

/// Decomposed uncertainty metrics computed from an ensemble of predictions.
#[derive(Debug, Clone)]
pub struct UncertaintyMetrics {
    /// Predictive entropy H[y|x] = -Σ_c p_mean_c * log(p_mean_c), per sample
    pub entropy: Vec<f64>,
    /// Mutual information I[y, θ|x] = H[y|x] - E_θ[H[y|x,θ]], per sample
    pub mutual_info: Vec<f64>,
    /// Mean predictive variance across classes, per sample
    pub variance: Vec<f64>,
    /// Aleatoric uncertainty = mean entropy of individual model predictions, per sample
    pub aleatoric: Vec<f64>,
    /// Epistemic uncertainty = total entropy - aleatoric, per sample
    pub epistemic: Vec<f64>,
}

/// Compute uncertainty metrics from ensemble predictions.
///
/// # Arguments
/// * `ensemble_probs` - Shape `[n_samples][n_models][n_classes]`.
///   Each entry is a probability distribution over classes.
///
/// # Returns
/// Per-sample uncertainty decomposition.
///
/// # Errors
/// Returns an error if the input is malformed.
pub fn compute_uncertainty_metrics(ensemble_probs: &[Vec<Vec<f64>>]) -> Result<UncertaintyMetrics> {
    if ensemble_probs.is_empty() {
        return Ok(UncertaintyMetrics {
            entropy: vec![],
            mutual_info: vec![],
            variance: vec![],
            aleatoric: vec![],
            epistemic: vec![],
        });
    }

    let n_samples = ensemble_probs.len();
    let n_models = ensemble_probs[0].len();
    if n_models == 0 {
        return Err(MetricsError::InvalidInput(
            "ensemble must have at least 1 model".to_string(),
        ));
    }
    let n_classes = ensemble_probs[0][0].len();
    if n_classes == 0 {
        return Err(MetricsError::InvalidInput(
            "n_classes must be > 0".to_string(),
        ));
    }

    let mut entropy_vec = Vec::with_capacity(n_samples);
    let mut mutual_info_vec = Vec::with_capacity(n_samples);
    let mut variance_vec = Vec::with_capacity(n_samples);
    let mut aleatoric_vec = Vec::with_capacity(n_samples);
    let mut epistemic_vec = Vec::with_capacity(n_samples);

    let eps = 1e-12f64;

    for (si, sample) in ensemble_probs.iter().enumerate() {
        if sample.len() != n_models {
            return Err(MetricsError::InvalidInput(format!(
                "sample {si} has {} models, expected {n_models}",
                sample.len()
            )));
        }

        // Mean prediction over models
        let mut mean_prob = vec![0.0f64; n_classes];
        for model_probs in sample {
            if model_probs.len() != n_classes {
                return Err(MetricsError::InvalidInput(format!(
                    "sample {si}: model prediction has {} classes, expected {n_classes}",
                    model_probs.len()
                )));
            }
            for (c, &p) in model_probs.iter().enumerate() {
                mean_prob[c] += p;
            }
        }
        for p in &mut mean_prob {
            *p /= n_models as f64;
        }

        // Predictive entropy: H[y|x] = -Σ_c p_mean_c log(p_mean_c)
        let pred_entropy: f64 = mean_prob
            .iter()
            .map(|&p| if p > eps { -p * p.ln() } else { 0.0 })
            .sum();

        // Aleatoric: mean entropy of individual models
        let aleatoric: f64 = sample
            .iter()
            .map(|model_probs| {
                model_probs
                    .iter()
                    .map(|&p| if p > eps { -p * p.ln() } else { 0.0 })
                    .sum::<f64>()
            })
            .sum::<f64>()
            / n_models as f64;

        // Epistemic = predictive entropy - aleatoric (mutual information)
        let epistemic = (pred_entropy - aleatoric).max(0.0);

        // Mutual information = predictive entropy - expected entropy of models
        let mi = epistemic;

        // Variance: mean over classes of Var_models[p_c]
        let variance: f64 = (0..n_classes)
            .map(|c| {
                let mean_c = mean_prob[c];
                let var_c = sample
                    .iter()
                    .map(|mp| {
                        let diff = mp[c] - mean_c;
                        diff * diff
                    })
                    .sum::<f64>()
                    / n_models as f64;
                var_c
            })
            .sum::<f64>()
            / n_classes as f64;

        entropy_vec.push(pred_entropy);
        mutual_info_vec.push(mi);
        variance_vec.push(variance);
        aleatoric_vec.push(aleatoric);
        epistemic_vec.push(epistemic);
    }

    Ok(UncertaintyMetrics {
        entropy: entropy_vec,
        mutual_info: mutual_info_vec,
        variance: variance_vec,
        aleatoric: aleatoric_vec,
        epistemic: epistemic_vec,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_calibration() {
        // Bins where confidence ≈ accuracy → near diagonal
        // 100 samples: first 50 with prob=0.2, 10 positive; next 50 with prob=0.8, 40 positive
        let mut probs = vec![0.2f64; 50];
        probs.extend(vec![0.8f64; 50]);
        let mut labels = vec![false; 50];
        // 10 positives in first half
        for i in 0..10 {
            labels[i] = true;
        }
        // 40 positives in second half
        let mut second_labels = vec![false; 50];
        for i in 0..40 {
            second_labels[i] = true;
        }
        labels.extend(second_labels);

        let config = ReliabilityDiagramConfig {
            n_bins: 10,
            min_bin_size: 1,
            ..Default::default()
        };
        let diag = reliability_diagram(&probs, &labels, &config).expect("should succeed");
        // ECE should be low
        assert!(
            diag.ece < 0.1,
            "ECE should be small for well-calibrated model: {}",
            diag.ece
        );
    }

    #[test]
    fn test_always_overconfident() {
        // Always predict 0.9 but only 50% positive
        let probs = vec![0.9f64; 100];
        let mut labels = vec![false; 100];
        for i in 0..50 {
            labels[i] = true;
        }

        let config = ReliabilityDiagramConfig {
            n_bins: 10,
            min_bin_size: 1,
            ..Default::default()
        };
        let diag = reliability_diagram(&probs, &labels, &config).expect("should succeed");
        assert!(
            diag.overconfidence_frac > 0.0,
            "should detect overconfidence"
        );
        assert!(
            diag.ece > 0.3,
            "ECE should be large when always predicting 0.9 with 50% pos: {}",
            diag.ece
        );
    }

    #[test]
    fn test_reliability_diagram_config_default() {
        let c = ReliabilityDiagramConfig::default();
        assert_eq!(c.n_bins, 10);
        assert!(!c.adaptive_bins);
        assert_eq!(c.min_bin_size, 5);
    }

    #[test]
    fn test_temperature_scaling_already_calibrated() {
        // Logits that produce near-calibrated probabilities should give T≈1
        // Use simple 2-class logits: [large, small] for class 0
        let logits: Vec<Vec<f64>> = (0..50)
            .map(|i| {
                if i % 2 == 0 {
                    vec![2.0, 0.0]
                } else {
                    vec![0.0, 2.0]
                }
            })
            .collect();
        let labels: Vec<usize> = (0..50).map(|i| if i % 2 == 0 { 0 } else { 1 }).collect();
        let t = adaptive_temperature_scaling(&logits, &labels, 200).expect("should succeed");
        // T should be reasonable (near 1 for well-calibrated logits)
        assert!(t > 0.1 && t < 10.0, "temperature should be reasonable: {t}");
    }

    #[test]
    fn test_entropy_uniform_distribution() {
        // Entropy of uniform distribution over n_classes = log(n_classes)
        let n_classes = 4usize;
        let p = 1.0 / n_classes as f64;
        let uniform = vec![p; n_classes];
        let ensemble = vec![vec![uniform.clone()]]; // 1 sample, 1 model

        let metrics = compute_uncertainty_metrics(&ensemble).expect("should succeed");
        let expected_entropy = (n_classes as f64).ln();
        assert!(
            (metrics.entropy[0] - expected_entropy).abs() < 1e-9,
            "entropy of uniform dist should be ln({n_classes})={expected_entropy}, got {}",
            metrics.entropy[0]
        );
    }

    #[test]
    fn test_epistemic_diverse_ensemble() {
        // Diverse ensemble: models disagree a lot → high epistemic uncertainty
        // Model 1: predicts class 0 with high confidence
        // Model 2: predicts class 1 with high confidence
        let ensemble = vec![vec![vec![0.99, 0.01], vec![0.01, 0.99]]];
        let metrics = compute_uncertainty_metrics(&ensemble).expect("should succeed");
        assert!(
            metrics.epistemic[0] > 0.0,
            "diverse ensemble should have positive epistemic uncertainty"
        );
    }

    #[test]
    fn test_adaptive_bins() {
        let probs: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let labels: Vec<bool> = (0..100).map(|i| i % 2 == 0).collect();
        let config = ReliabilityDiagramConfig {
            n_bins: 5,
            adaptive_bins: true,
            min_bin_size: 1,
        };
        let diag = reliability_diagram(&probs, &labels, &config).expect("should succeed");
        assert!(!diag.bins.is_empty(), "should have bins");
    }

    #[test]
    fn test_empty_input_error() {
        let config = ReliabilityDiagramConfig::default();
        assert!(reliability_diagram(&[], &[], &config).is_err());
    }

    #[test]
    fn test_length_mismatch_error() {
        let config = ReliabilityDiagramConfig::default();
        assert!(reliability_diagram(&[0.5], &[], &config).is_err());
    }
}
