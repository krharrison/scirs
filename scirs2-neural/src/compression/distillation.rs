//! Knowledge distillation utilities.
//!
//! Implements temperature-scaled soft targets for training a student network
//! to mimic a teacher network, following Hinton et al. (2015).

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array1, Array2};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for knowledge distillation.
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    /// Temperature `T` used to soften logits.  Higher values produce softer
    /// probability distributions.  Must be > 0.
    pub temperature: f32,
    /// Weight `α` on the soft-target loss.  Hard-label loss weight is `1 - α`.
    /// Must be in `[0, 1]`.
    pub alpha: f32,
}

impl DistillationConfig {
    /// Create a distillation config.
    ///
    /// # Errors
    /// Returns an error if `temperature <= 0` or `alpha` is not in `[0, 1]`.
    pub fn new(temperature: f32, alpha: f32) -> Result<Self> {
        if temperature <= 0.0 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "temperature must be > 0, got {temperature}"
            )));
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(NeuralError::InvalidArchitecture(format!(
                "alpha must be in [0,1], got {alpha}"
            )));
        }
        Ok(Self { temperature, alpha })
    }

    /// Hinton-style defaults: `temperature = 4.0`, `alpha = 0.5`.
    pub fn default_hinton() -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.5,
        }
    }
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            alpha: 0.5,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Temperature scaling
// ─────────────────────────────────────────────────────────────────────────────

/// Scale logits by temperature and apply softmax.
///
/// `softmax(logits / T)` where T is the temperature.
///
/// # Errors
/// Returns an error if `logits` is empty or `temperature <= 0`.
pub fn temperature_softmax(logits: &Array1<f32>, temperature: f32) -> Result<Array1<f32>> {
    if logits.is_empty() {
        return Err(NeuralError::InvalidArchitecture(
            "temperature_softmax: empty logits".into(),
        ));
    }
    if temperature <= 0.0 {
        return Err(NeuralError::InvalidArchitecture(format!(
            "temperature must be > 0, got {temperature}"
        )));
    }
    softmax_stable(&logits.mapv(|v| v / temperature))
}

/// Numerically-stable softmax (subtracts max before exponentiating).
fn softmax_stable(x: &Array1<f32>) -> Result<Array1<f32>> {
    let max_val = x
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    if !max_val.is_finite() {
        return Err(NeuralError::InvalidArchitecture(
            "softmax: input contains non-finite values".into(),
        ));
    }
    let exp: Array1<f32> = x.mapv(|v| (v - max_val).exp());
    let sum = exp.sum();
    if sum == 0.0 {
        return Err(NeuralError::InvalidArchitecture(
            "softmax: all-zero exponential sum (overflow?)".into(),
        ));
    }
    Ok(exp / sum)
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss functions
// ─────────────────────────────────────────────────────────────────────────────

/// Soft-target KL-divergence loss for a single sample.
///
/// Computes `T² * KL(soft_teacher || soft_student)` where both distributions are
/// obtained via temperature-scaled softmax, following Hinton et al. §2.
/// The `T²` factor preserves gradient magnitude independent of temperature.
///
/// # Errors
/// Returns an error if logit shapes differ or temperature ≤ 0.
pub fn soft_target_loss(
    student_logits: &Array1<f32>,
    teacher_logits: &Array1<f32>,
    temp: f32,
) -> Result<f32> {
    if student_logits.len() != teacher_logits.len() {
        return Err(NeuralError::InvalidArchitecture(format!(
            "soft_target_loss: student {} vs teacher {} logit length mismatch",
            student_logits.len(),
            teacher_logits.len()
        )));
    }
    if temp <= 0.0 {
        return Err(NeuralError::InvalidArchitecture(format!(
            "temperature must be > 0, got {temp}"
        )));
    }
    let p_teacher = temperature_softmax(teacher_logits, temp)?;
    let p_student = temperature_softmax(student_logits, temp)?;

    // KL(p_t || p_s) = Σ p_t * log(p_t / p_s)
    let kl: f32 = p_teacher
        .iter()
        .zip(p_student.iter())
        .map(|(&pt, &ps)| {
            if pt <= 0.0 {
                0.0
            } else {
                let ps_safe = ps.max(1e-12);
                pt * (pt / ps_safe).ln()
            }
        })
        .sum();
    // Multiply by T² to maintain gradient scale.
    Ok(temp * temp * kl)
}

/// Hard-label cross-entropy loss for a single sample.
///
/// `hard_label` is a one-hot class index in `[0, num_classes)`.
///
/// # Errors
/// Returns an error if `hard_label >= student_logits.len()`.
pub fn hard_label_loss(student_logits: &Array1<f32>, hard_label: usize) -> Result<f32> {
    if hard_label >= student_logits.len() {
        return Err(NeuralError::InvalidArchitecture(format!(
            "hard_label {hard_label} out of range [0, {})",
            student_logits.len()
        )));
    }
    let probs = softmax_stable(student_logits)?;
    let p = probs[hard_label].max(1e-12);
    Ok(-p.ln())
}

/// Combined distillation loss for a single sample.
///
/// `loss = α * soft_loss(T) + (1 - α) * hard_loss`
///
/// # Errors
/// Returns an error if any component fails.
pub fn distillation_loss(
    student_logits: &Array1<f32>,
    teacher_logits: &Array1<f32>,
    hard_label: usize,
    config: &DistillationConfig,
) -> Result<f32> {
    let soft = soft_target_loss(student_logits, teacher_logits, config.temperature)?;
    let hard = hard_label_loss(student_logits, hard_label)?;
    Ok(config.alpha * soft + (1.0 - config.alpha) * hard)
}

/// Batch distillation loss — average over a mini-batch.
///
/// `student_logits` and `teacher_logits` are `(batch, num_classes)`;
/// `hard_labels` is a 1-D slice of class indices, one per sample.
///
/// # Errors
/// Returns an error if batch sizes are inconsistent or any per-sample loss fails.
pub fn distillation_loss_batch(
    student_logits: &Array2<f32>,
    teacher_logits: &Array2<f32>,
    hard_labels: &[usize],
    config: &DistillationConfig,
) -> Result<f32> {
    let batch = student_logits.nrows();
    if teacher_logits.nrows() != batch {
        return Err(NeuralError::InvalidArchitecture(format!(
            "distillation_loss_batch: student batch {batch} != teacher batch {}",
            teacher_logits.nrows()
        )));
    }
    if hard_labels.len() != batch {
        return Err(NeuralError::InvalidArchitecture(format!(
            "distillation_loss_batch: batch size {batch} != labels len {}",
            hard_labels.len()
        )));
    }
    if batch == 0 {
        return Err(NeuralError::InvalidArchitecture(
            "distillation_loss_batch: batch size must be > 0".into(),
        ));
    }
    let total: f32 = (0..batch)
        .map(|i| {
            let s = student_logits.row(i).to_owned();
            let t = teacher_logits.row(i).to_owned();
            distillation_loss(&s, &t, hard_labels[i], config)
        })
        .collect::<Result<Vec<f32>>>()?
        .into_iter()
        .sum();
    Ok(total / batch as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn logits_3class() -> Array1<f32> {
        Array1::from_vec(vec![2.0, 1.0, 0.1])
    }

    #[test]
    fn test_temperature_softmax_at_one() {
        let logits = logits_3class();
        let probs = temperature_softmax(&logits, 1.0).expect("failed");
        let sum: f32 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-5, "probs should sum to 1, got {sum}");
        // Highest logit should have highest probability.
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }

    #[test]
    fn test_temperature_softmax_high_temp() {
        let logits = logits_3class();
        let probs = temperature_softmax(&logits, 10.0).expect("failed");
        // High temperature → flatter distribution.
        assert!(probs[0] < 0.5, "high-T distribution should be flat");
    }

    #[test]
    fn test_temperature_softmax_invalid() {
        let logits = logits_3class();
        assert!(temperature_softmax(&logits, 0.0).is_err());
        assert!(temperature_softmax(&logits, -1.0).is_err());
        let empty: Array1<f32> = Array1::from_vec(vec![]);
        assert!(temperature_softmax(&empty, 1.0).is_err());
    }

    #[test]
    fn test_soft_target_loss_identical() {
        let logits = logits_3class();
        // When student == teacher, KL divergence should be ≈ 0.
        let loss = soft_target_loss(&logits, &logits, 2.0).expect("failed");
        assert!(loss < 1e-5, "identical logits should give ~0 KL, got {loss}");
    }

    #[test]
    fn test_soft_target_loss_mismatch_shape() {
        let s = Array1::from_vec(vec![1.0, 2.0]);
        let t = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(soft_target_loss(&s, &t, 1.0).is_err());
    }

    #[test]
    fn test_hard_label_loss_basic() {
        let logits = logits_3class();
        let loss = hard_label_loss(&logits, 0).expect("failed");
        // For strong logit[0]=2.0, loss should be small.
        assert!(loss < 1.0, "correct class loss should be < 1, got {loss}");
        let loss_wrong = hard_label_loss(&logits, 2).expect("failed");
        assert!(
            loss_wrong > loss,
            "wrong class loss should be larger: {loss} vs {loss_wrong}"
        );
    }

    #[test]
    fn test_hard_label_out_of_range() {
        let logits = logits_3class();
        assert!(hard_label_loss(&logits, 3).is_err());
    }

    #[test]
    fn test_distillation_loss_alpha_zero() {
        // α=0 → only hard loss.
        let cfg = DistillationConfig::new(2.0, 0.0).expect("cfg failed");
        let s = logits_3class();
        let t = logits_3class();
        let dl = distillation_loss(&s, &t, 0, &cfg).expect("distillation_loss failed");
        let hl = hard_label_loss(&s, 0).expect("hard label loss failed");
        assert!((dl - hl).abs() < 1e-5);
    }

    #[test]
    fn test_distillation_loss_alpha_one() {
        // α=1 → only soft loss.
        let cfg = DistillationConfig::new(2.0, 1.0).expect("cfg failed");
        let s = logits_3class();
        let t = logits_3class();
        let dl = distillation_loss(&s, &t, 0, &cfg).expect("distillation_loss failed");
        let sl = soft_target_loss(&s, &t, 2.0).expect("soft target loss failed");
        assert!((dl - sl).abs() < 1e-5);
    }

    #[test]
    fn test_distillation_loss_batch() {
        let batch = 4;
        let n_class = 5;
        let student = Array2::from_shape_fn((batch, n_class), |(i, j)| {
            (i * n_class + j + 1) as f32 * 0.1
        });
        let teacher = Array2::from_shape_fn((batch, n_class), |(i, j)| {
            (i * n_class + j) as f32 * 0.12
        });
        let labels: Vec<usize> = (0..batch).map(|i| i % n_class).collect();
        let cfg = DistillationConfig::default_hinton();
        let loss =
            distillation_loss_batch(&student, &teacher, &labels, &cfg).expect("batch loss failed");
        assert!(loss.is_finite(), "batch loss should be finite, got {loss}");
        assert!(loss > 0.0, "batch loss should be positive, got {loss}");
    }

    #[test]
    fn test_distillation_config_validation() {
        assert!(DistillationConfig::new(0.0, 0.5).is_err());
        assert!(DistillationConfig::new(-1.0, 0.5).is_err());
        assert!(DistillationConfig::new(1.0, -0.1).is_err());
        assert!(DistillationConfig::new(1.0, 1.1).is_err());
        assert!(DistillationConfig::new(1.0, 0.5).is_ok());
    }
}
