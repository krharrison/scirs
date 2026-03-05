//! Knowledge distillation for neural network training
//!
//! Provides loss functions and training helpers for distilling knowledge from
//! a large teacher network into a smaller student network.
//!
//! ## Approach
//! - **Soft-target loss** (Hinton et al., 2015): KL divergence between teacher and
//!   student logit distributions after temperature scaling.
//! - **Hard-target loss**: Cross-entropy between student predictions and ground-truth
//!   labels.
//! - **Combined loss**: `alpha * soft_loss + (1 - alpha) * hard_loss`
//! - **Feature distillation**: L2 loss between intermediate layer activations.
//! - **Attention transfer** (Zagoruyko & Komodakis, 2017): Match sum-of-squares
//!   attention maps from teacher and student.
//!
//! ## Quick start
//!
//! ```rust
//! use scirs2_neural::training::knowledge_distillation::{DistillationConfig, distillation_loss};
//! use scirs2_core::ndarray::array;
//!
//! let student_logits = array![[2.0_f64, 1.0, 0.5], [0.1, 3.0, 1.2]];
//! let teacher_logits = array![[1.8_f64, 1.1, 0.6], [0.0, 2.8, 1.0]];
//! let true_labels = vec![0usize, 1];
//! let config = DistillationConfig { temperature: 4.0, alpha: 0.5 };
//! let loss = distillation_loss(
//!     student_logits.view(),
//!     teacher_logits.view(),
//!     &true_labels,
//!     &config,
//! ).expect("distillation loss failed");
//! assert!(loss >= 0.0);
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive, ToPrimitive};
use std::fmt::Debug;

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for knowledge distillation training.
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    /// Temperature for softening probability distributions.
    ///
    /// Higher temperature → softer distributions that reveal more information
    /// about teacher confidence.  Typical range: 2–8.
    pub temperature: f64,
    /// Balance between soft-target loss and hard-target loss.
    ///
    /// `total_loss = alpha * soft_loss + (1 - alpha) * hard_loss`
    ///
    /// - `alpha = 1.0`: pure distillation (ignore true labels)
    /// - `alpha = 0.0`: pure cross-entropy with true labels
    /// - `alpha = 0.5`: equal weighting (common default)
    pub alpha: f64,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.5,
        }
    }
}

impl DistillationConfig {
    /// Validate that temperature and alpha are in sensible ranges.
    pub fn validate(&self) -> Result<()> {
        if self.temperature <= 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "temperature must be > 0, got {}",
                self.temperature
            )));
        }
        if self.alpha < 0.0 || self.alpha > 1.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "alpha must be in [0, 1], got {}",
                self.alpha
            )));
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Numerically-stable softmax helpers
// ────────────────────────────────────────────────────────────────────────────

/// Compute row-wise softmax with temperature scaling.
///
/// Each row is divided by `temperature` before taking exp, then normalized.
fn softmax_with_temperature(logits: ArrayView2<f64>, temperature: f64) -> Array2<f64> {
    let (nrows, ncols) = (logits.nrows(), logits.ncols());
    let mut out = Array2::<f64>::zeros((nrows, ncols));

    for r in 0..nrows {
        // Numerical stability: subtract row max before exp.
        let row_max = logits
            .row(r)
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut sum_exp = 0.0f64;
        for c in 0..ncols {
            let e = ((logits[[r, c]] - row_max) / temperature).exp();
            out[[r, c]] = e;
            sum_exp += e;
        }
        let inv = if sum_exp > 0.0 { 1.0 / sum_exp } else { 1.0 };
        for c in 0..ncols {
            out[[r, c]] *= inv;
        }
    }

    out
}

/// Compute row-wise standard softmax (temperature = 1).
fn softmax(logits: ArrayView2<f64>) -> Array2<f64> {
    softmax_with_temperature(logits, 1.0)
}

// ────────────────────────────────────────────────────────────────────────────
// Soft-target loss (KL divergence with temperature)
// ────────────────────────────────────────────────────────────────────────────

/// Compute the soft-target distillation loss (KL divergence) between student
/// and teacher logits after temperature scaling.
///
/// The loss is scaled by `temperature²` to match the gradient magnitude of the
/// hard-target loss (Hinton et al., 2015).
///
/// Returns the *mean* KL divergence across all samples in the batch.
pub fn soft_target_loss(
    student_logits: ArrayView2<f64>,
    teacher_logits: ArrayView2<f64>,
    temperature: f64,
) -> Result<f64> {
    validate_shapes_2d(student_logits, teacher_logits, "soft_target_loss")?;
    if temperature <= 0.0 {
        return Err(NeuralError::InvalidArgument(format!(
            "temperature must be > 0, got {temperature}"
        )));
    }

    let p_teacher = softmax_with_temperature(teacher_logits, temperature);
    let p_student = softmax_with_temperature(student_logits, temperature);

    let nrows = student_logits.nrows();
    let ncols = student_logits.ncols();
    let mut kl_sum = 0.0f64;

    for r in 0..nrows {
        for c in 0..ncols {
            let pt = p_teacher[[r, c]];
            let ps = p_student[[r, c]].max(1e-40); // avoid log(0)
            if pt > 0.0 {
                kl_sum += pt * (pt / ps).ln();
            }
        }
    }

    // Scale by T² and normalize by batch size.
    Ok(kl_sum * temperature * temperature / nrows as f64)
}

// ────────────────────────────────────────────────────────────────────────────
// Hard-target loss (cross-entropy with true labels)
// ────────────────────────────────────────────────────────────────────────────

/// Compute cross-entropy loss between student logits and true integer class labels.
///
/// Returns the *mean* negative log-likelihood across the batch.
pub fn hard_target_loss(
    student_logits: ArrayView2<f64>,
    true_labels: &[usize],
) -> Result<f64> {
    let nrows = student_logits.nrows();
    let ncols = student_logits.ncols();

    if true_labels.len() != nrows {
        return Err(NeuralError::ShapeMismatch(format!(
            "true_labels length {} != batch size {}",
            true_labels.len(),
            nrows
        )));
    }

    let probs = softmax(student_logits);
    let mut nll_sum = 0.0f64;

    for (r, &label) in true_labels.iter().enumerate() {
        if label >= ncols {
            return Err(NeuralError::InvalidArgument(format!(
                "label {label} out of range for n_classes={ncols}"
            )));
        }
        let p = probs[[r, label]].max(1e-40);
        nll_sum += -p.ln();
    }

    Ok(nll_sum / nrows as f64)
}

// ────────────────────────────────────────────────────────────────────────────
// Combined distillation loss
// ────────────────────────────────────────────────────────────────────────────

/// Compute the combined knowledge-distillation loss:
///
/// `loss = alpha * soft_loss(T) + (1 - alpha) * hard_loss`
///
/// where `soft_loss` is the temperature-scaled KL divergence and `hard_loss`
/// is the cross-entropy against ground-truth labels.
pub fn distillation_loss(
    student_logits: ArrayView2<f64>,
    teacher_logits: ArrayView2<f64>,
    true_labels: &[usize],
    config: &DistillationConfig,
) -> Result<f64> {
    config.validate()?;

    let soft = soft_target_loss(student_logits, teacher_logits, config.temperature)?;
    let hard = hard_target_loss(student_logits, true_labels)?;

    Ok(config.alpha * soft + (1.0 - config.alpha) * hard)
}

/// Component breakdown of the distillation loss.
#[derive(Debug, Clone)]
pub struct DistillationLossComponents {
    /// Soft-target (KL divergence) loss component.
    pub soft_loss: f64,
    /// Hard-target (cross-entropy) loss component.
    pub hard_loss: f64,
    /// Combined weighted loss.
    pub total_loss: f64,
    /// Alpha used to combine components.
    pub alpha: f64,
    /// Temperature used for soft targets.
    pub temperature: f64,
}

/// Like `distillation_loss` but also returns the individual loss components.
pub fn distillation_loss_detailed(
    student_logits: ArrayView2<f64>,
    teacher_logits: ArrayView2<f64>,
    true_labels: &[usize],
    config: &DistillationConfig,
) -> Result<DistillationLossComponents> {
    config.validate()?;

    let soft_loss =
        soft_target_loss(student_logits, teacher_logits, config.temperature)?;
    let hard_loss = hard_target_loss(student_logits, true_labels)?;
    let total_loss = config.alpha * soft_loss + (1.0 - config.alpha) * hard_loss;

    Ok(DistillationLossComponents {
        soft_loss,
        hard_loss,
        total_loss,
        alpha: config.alpha,
        temperature: config.temperature,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Feature distillation
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for intermediate-layer feature distillation.
#[derive(Debug, Clone)]
pub struct FeatureDistillationConfig {
    /// Weight applied to the feature-matching loss.
    pub loss_weight: f64,
    /// Whether to L2-normalize features before computing the loss.
    pub normalize_features: bool,
}

impl Default for FeatureDistillationConfig {
    fn default() -> Self {
        Self {
            loss_weight: 1.0,
            normalize_features: true,
        }
    }
}

/// Intermediate-layer feature distillation.
///
/// Matches teacher and student intermediate activations using a mean-squared-error
/// loss.  If the feature dimensions differ, a linear projection matrix `projection`
/// must be supplied to map the student features into the teacher's feature space.
///
/// `student_features` shape: `[batch, student_dim]`
/// `teacher_features` shape: `[batch, teacher_dim]`
/// `projection` shape (optional): `[student_dim, teacher_dim]`
#[derive(Debug, Clone)]
pub struct FeatureDistillation {
    /// Configuration.
    pub config: FeatureDistillationConfig,
    /// Optional linear projection (student_dim → teacher_dim).
    pub projection: Option<Array2<f64>>,
}

impl FeatureDistillation {
    /// Create a new `FeatureDistillation` without a projection layer.
    pub fn new(config: FeatureDistillationConfig) -> Self {
        Self {
            config,
            projection: None,
        }
    }

    /// Create a `FeatureDistillation` with a projection matrix.
    pub fn with_projection(mut self, proj: Array2<f64>) -> Self {
        self.projection = Some(proj);
        self
    }

    /// Compute the feature-matching loss.
    pub fn loss(
        &self,
        student_features: ArrayView2<f64>,
        teacher_features: ArrayView2<f64>,
    ) -> Result<f64> {
        let projected: Array2<f64> = if let Some(ref proj) = self.projection {
            // Simple matmul: [batch, student_dim] × [student_dim, teacher_dim] → [batch, teacher_dim]
            matmul_2d(student_features, proj.view())?
        } else {
            if student_features.ncols() != teacher_features.ncols() {
                return Err(NeuralError::DimensionMismatch(format!(
                    "student_features cols {} != teacher_features cols {}; supply a projection matrix",
                    student_features.ncols(),
                    teacher_features.ncols()
                )));
            }
            student_features.to_owned()
        };

        let s = if self.config.normalize_features {
            l2_normalize_rows(projected.view())?
        } else {
            projected
        };

        let t = if self.config.normalize_features {
            l2_normalize_rows(teacher_features)?
        } else {
            teacher_features.to_owned()
        };

        validate_shapes_2d(s.view(), t.view(), "feature_distillation_loss")?;

        let mse: f64 = s
            .iter()
            .zip(t.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / s.len() as f64;

        Ok(mse * self.config.loss_weight)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Attention transfer loss
// ────────────────────────────────────────────────────────────────────────────

/// Compute the attention-transfer loss between student and teacher attention maps.
///
/// Based on "Paying More Attention to Attention" (Zagoruyko & Komodakis, 2017).
///
/// The attention map for a 2-D activation `A` is defined as:
/// `F(A) = ||A||_2` normalized to unit L2 norm.
///
/// For 2-D inputs (batch × features), the per-sample L2 norm is used directly.
///
/// Both `student_attn` and `teacher_attn` must have the same shape.
pub fn attention_transfer_loss(
    student_attn: ArrayView2<f64>,
    teacher_attn: ArrayView2<f64>,
) -> Result<f64> {
    validate_shapes_2d(student_attn, teacher_attn, "attention_transfer_loss")?;

    // Compute L2 norm per row, normalize, then compute MSE.
    let s_map = attention_map(student_attn)?;
    let t_map = attention_map(teacher_attn)?;

    let loss: f64 = s_map
        .iter()
        .zip(t_map.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / s_map.len() as f64;

    Ok(loss)
}

/// Compute normalized attention map (row-wise L2-norm then L2-normalize).
fn attention_map(x: ArrayView2<f64>) -> Result<Array1<f64>> {
    let nrows = x.nrows();
    let mut norms = Array1::<f64>::zeros(nrows);

    for r in 0..nrows {
        let n: f64 = x.row(r).iter().map(|v| v * v).sum::<f64>().sqrt();
        norms[r] = n;
    }

    // L2-normalize the vector of norms.
    let total: f64 = norms.iter().map(|v| v * v).sum::<f64>().sqrt();
    if total > 1e-12 {
        for v in norms.iter_mut() {
            *v /= total;
        }
    }

    Ok(norms)
}

// ────────────────────────────────────────────────────────────────────────────
// Teacher–Student framework helper
// ────────────────────────────────────────────────────────────────────────────

/// Accumulated distillation statistics across one training epoch.
#[derive(Debug, Clone, Default)]
pub struct DistillationStats {
    /// Total soft-target loss accumulated this epoch.
    pub total_soft_loss: f64,
    /// Total hard-target loss accumulated this epoch.
    pub total_hard_loss: f64,
    /// Total combined loss accumulated this epoch.
    pub total_loss: f64,
    /// Number of batches processed.
    pub n_batches: usize,
}

impl DistillationStats {
    /// Accumulate one batch's loss components.
    pub fn record(&mut self, components: &DistillationLossComponents) {
        self.total_soft_loss += components.soft_loss;
        self.total_hard_loss += components.hard_loss;
        self.total_loss += components.total_loss;
        self.n_batches += 1;
    }

    /// Average soft-target loss per batch.
    pub fn avg_soft_loss(&self) -> f64 {
        if self.n_batches == 0 {
            0.0
        } else {
            self.total_soft_loss / self.n_batches as f64
        }
    }

    /// Average hard-target loss per batch.
    pub fn avg_hard_loss(&self) -> f64 {
        if self.n_batches == 0 {
            0.0
        } else {
            self.total_hard_loss / self.n_batches as f64
        }
    }

    /// Average combined loss per batch.
    pub fn avg_total_loss(&self) -> f64 {
        if self.n_batches == 0 {
            0.0
        } else {
            self.total_loss / self.n_batches as f64
        }
    }

    /// Reset all accumulators.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Temperature calibration helper
// ────────────────────────────────────────────────────────────────────────────

/// Find the temperature that minimizes the KL divergence between the
/// temperature-scaled student distribution and the teacher distribution,
/// using a simple golden-section grid search.
///
/// `temp_range`: `(min, max)` temperatures to search (e.g. `(0.5, 20.0)`).
/// `n_grid`: number of grid points to evaluate.
pub fn calibrate_temperature(
    student_logits: ArrayView2<f64>,
    teacher_logits: ArrayView2<f64>,
    temp_range: (f64, f64),
    n_grid: usize,
) -> Result<f64> {
    if temp_range.0 <= 0.0 || temp_range.1 <= temp_range.0 {
        return Err(NeuralError::InvalidArgument(format!(
            "temp_range must be (positive, larger): got {:?}",
            temp_range
        )));
    }
    if n_grid < 2 {
        return Err(NeuralError::InvalidArgument(
            "n_grid must be >= 2".to_string(),
        ));
    }

    let (t_min, t_max) = temp_range;
    let step = (t_max - t_min) / (n_grid - 1) as f64;

    let mut best_temp = t_min;
    let mut best_kl = f64::INFINITY;

    for i in 0..n_grid {
        let t = t_min + step * i as f64;
        let kl = soft_target_loss(student_logits, teacher_logits, t)?;
        if kl < best_kl {
            best_kl = kl;
            best_temp = t;
        }
    }

    Ok(best_temp)
}

// ────────────────────────────────────────────────────────────────────────────
// Internal math helpers
// ────────────────────────────────────────────────────────────────────────────

/// Row-wise L2 normalization of a 2-D array.
fn l2_normalize_rows(x: ArrayView2<f64>) -> Result<Array2<f64>> {
    let (nrows, ncols) = (x.nrows(), x.ncols());
    let mut out = Array2::<f64>::zeros((nrows, ncols));

    for r in 0..nrows {
        let norm: f64 = x.row(r).iter().map(|v| v * v).sum::<f64>().sqrt();
        let inv = if norm > 1e-12 { 1.0 / norm } else { 1.0 };
        for c in 0..ncols {
            out[[r, c]] = x[[r, c]] * inv;
        }
    }

    Ok(out)
}

/// Naive 2-D matrix multiplication: `[m, k] × [k, n] → [m, n]`.
fn matmul_2d(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Result<Array2<f64>> {
    let (m, k) = (a.nrows(), a.ncols());
    let (k2, n) = (b.nrows(), b.ncols());
    if k != k2 {
        return Err(NeuralError::DimensionMismatch(format!(
            "matmul: a.ncols={k} != b.nrows={k2}"
        )));
    }

    let mut out = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for p in 0..k {
                s += a[[i, p]] * b[[p, j]];
            }
            out[[i, j]] = s;
        }
    }
    Ok(out)
}

/// Validate that two 2-D arrays have the same shape.
fn validate_shapes_2d(a: ArrayView2<f64>, b: ArrayView2<f64>, ctx: &str) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(NeuralError::ShapeMismatch(format!(
            "{ctx}: shape {:?} != {:?}",
            a.shape(),
            b.shape()
        )));
    }
    Ok(())
}


// ────────────────────────────────────────────────────────────────────────────
// Born-again networks (ensemble / self-distillation)
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for Born-Again Network (BAN) training.
///
/// In BAN training (Furlanello et al., 2018) a sequence of student models is
/// trained via knowledge distillation from the previous generation.  The final
/// prediction is the ensemble (uniform average) of all generations' logits.
///
/// This struct holds the distillation hyper-parameters shared across all
/// generations plus a generation counter.
#[derive(Debug, Clone)]
pub struct BornAgainConfig {
    /// Temperature used for soft-target distillation.
    pub temperature: f64,
    /// Blend coefficient between soft and hard targets (same semantics as
    /// [`DistillationConfig::alpha`]).
    pub alpha: f64,
    /// Total number of BAN generations (≥ 1).  Generation 0 is the initial
    /// model trained on hard labels; subsequent generations distil from the
    /// previous one.
    pub n_generations: usize,
}

impl Default for BornAgainConfig {
    fn default() -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.5,
            n_generations: 3,
        }
    }
}

impl BornAgainConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        if self.temperature <= 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "BAN temperature must be > 0, got {}",
                self.temperature
            )));
        }
        if self.alpha < 0.0 || self.alpha > 1.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "BAN alpha must be in [0, 1], got {}",
                self.alpha
            )));
        }
        if self.n_generations == 0 {
            return Err(NeuralError::InvalidArgument(
                "BAN n_generations must be >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Compute the **ensemble logits** from multiple BAN generations.
///
/// Each entry in `generation_logits` is the raw logit matrix (batch × classes)
/// produced by one generation.  The ensemble is the element-wise mean.
///
/// # Errors
///
/// Returns `Err` if the slice is empty or if the matrices have inconsistent
/// shapes.
pub fn ban_ensemble_logits(
    generation_logits: &[Array2<f64>],
) -> Result<Array2<f64>> {
    if generation_logits.is_empty() {
        return Err(NeuralError::InvalidArgument(
            "generation_logits must contain at least one matrix".to_string(),
        ));
    }
    let (nrows, ncols) = {
        let first = &generation_logits[0];
        (first.nrows(), first.ncols())
    };
    for (i, m) in generation_logits.iter().enumerate() {
        if m.nrows() != nrows || m.ncols() != ncols {
            return Err(NeuralError::ShapeMismatch(format!(
                "generation_logits[{}] has shape ({}, {}) but expected ({}, {})",
                i, m.nrows(), m.ncols(), nrows, ncols
            )));
        }
    }
    let n = generation_logits.len() as f64;
    let mut ensemble = Array2::<f64>::zeros((nrows, ncols));
    for logits in generation_logits {
        for r in 0..nrows {
            for c in 0..ncols {
                ensemble[[r, c]] += logits[[r, c]] / n;
            }
        }
    }
    Ok(ensemble)
}

/// Compute the BAN distillation loss for a single student generation.
///
/// The teacher logits come from the *previous* generation (or the ensemble of
/// all previous generations via [`ban_ensemble_logits`]).  The loss formula is
/// identical to the standard KD loss:
///
/// ```text
/// L = alpha * soft_loss(student, teacher, T) + (1 - alpha) * hard_loss(student, labels)
/// ```
///
/// # Arguments
///
/// * `student_logits`  — Raw logits from the current (student) generation.
/// * `teacher_logits`  — Logits from the previous generation (or their ensemble).
/// * `true_labels`     — Ground-truth class indices (length = batch size).
/// * `config`          — BAN configuration.
///
/// # Errors
///
/// Propagates shape / range errors from the underlying loss functions.
pub fn ban_distillation_loss(
    student_logits: ArrayView2<f64>,
    teacher_logits: ArrayView2<f64>,
    true_labels: &[usize],
    config: &BornAgainConfig,
) -> Result<f64> {
    config.validate()?;
    let kd_config = DistillationConfig {
        temperature: config.temperature,
        alpha: config.alpha,
    };
    distillation_loss(student_logits, teacher_logits, true_labels, &kd_config)
}

/// Compute BAN loss broken into its components (soft + hard + total).
///
/// Equivalent to [`ban_distillation_loss`] but returns the full
/// [`DistillationLossComponents`] breakdown for logging.
pub fn ban_distillation_loss_detailed(
    student_logits: ArrayView2<f64>,
    teacher_logits: ArrayView2<f64>,
    true_labels: &[usize],
    config: &BornAgainConfig,
) -> Result<DistillationLossComponents> {
    config.validate()?;
    let kd_config = DistillationConfig {
        temperature: config.temperature,
        alpha: config.alpha,
    };
    distillation_loss_detailed(student_logits, teacher_logits, true_labels, &kd_config)
}

/// Accumulated BAN training statistics.
///
/// Extends [`DistillationStats`] with a per-generation breakdown so you can
/// track how each generation is progressing.
#[derive(Debug, Clone, Default)]
pub struct BornAgainStats {
    /// Per-generation distillation statistics.
    pub generations: Vec<DistillationStats>,
}

impl BornAgainStats {
    /// Create stats tracker for `n_generations` generations.
    pub fn new(n_generations: usize) -> Self {
        Self {
            generations: vec![DistillationStats::default(); n_generations],
        }
    }

    /// Record a batch result for `generation` (0-indexed).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `generation` is out of range.
    pub fn record(
        &mut self,
        generation: usize,
        components: &DistillationLossComponents,
    ) -> Result<()> {
        if generation >= self.generations.len() {
            return Err(NeuralError::InvalidArgument(format!(
                "generation {} out of range (have {} generations)",
                generation,
                self.generations.len()
            )));
        }
        self.generations[generation].record(components);
        Ok(())
    }

    /// Reset all generation accumulators.
    pub fn reset_all(&mut self) {
        for g in &mut self.generations {
            g.reset();
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn batch_logits() -> (Array2<f64>, Array2<f64>) {
        let student = array![
            [2.0_f64, 1.0, 0.5],
            [0.1, 3.0, 1.2],
            [1.5, 1.5, 1.5]
        ];
        let teacher = array![
            [1.8_f64, 1.1, 0.6],
            [0.0, 2.8, 1.0],
            [1.0, 2.0, 0.5]
        ];
        (student, teacher)
    }

    #[test]
    fn test_soft_target_loss_positive() {
        let (student, teacher) = batch_logits();
        let loss = soft_target_loss(student.view(), teacher.view(), 4.0)
            .expect("soft loss failed");
        assert!(loss >= 0.0, "KL divergence must be non-negative, got {loss}");
    }

    #[test]
    fn test_soft_target_loss_identical_is_zero() {
        let (student, _) = batch_logits();
        // When student == teacher the KL divergence should be ~0.
        let loss = soft_target_loss(student.view(), student.view(), 4.0)
            .expect("soft loss failed");
        assert!(loss < 1e-9, "KL(p||p) should be ~0, got {loss}");
    }

    #[test]
    fn test_hard_target_loss() {
        let (student, _) = batch_logits();
        let labels = vec![0usize, 1, 2];
        let loss = hard_target_loss(student.view(), &labels).expect("hard loss failed");
        assert!(loss > 0.0, "cross-entropy must be positive, got {loss}");
    }

    #[test]
    fn test_distillation_loss_combined() {
        let (student, teacher) = batch_logits();
        let labels = vec![0usize, 1, 2];
        let config = DistillationConfig {
            temperature: 4.0,
            alpha: 0.5,
        };
        let loss =
            distillation_loss(student.view(), teacher.view(), &labels, &config)
                .expect("distillation loss failed");
        assert!(loss > 0.0);
    }

    #[test]
    fn test_distillation_loss_alpha_zero_equals_hard() {
        let (student, teacher) = batch_logits();
        let labels = vec![0usize, 1, 2];
        let config_alpha0 = DistillationConfig {
            temperature: 4.0,
            alpha: 0.0,
        };
        let loss_alpha0 =
            distillation_loss(student.view(), teacher.view(), &labels, &config_alpha0)
                .expect("loss alpha=0");
        let hard = hard_target_loss(student.view(), &labels).expect("hard loss");
        assert!((loss_alpha0 - hard).abs() < 1e-12, "alpha=0 should equal hard loss");
    }

    #[test]
    fn test_distillation_loss_detailed() {
        let (student, teacher) = batch_logits();
        let labels = vec![0usize, 1, 2];
        let config = DistillationConfig::default();
        let detail =
            distillation_loss_detailed(student.view(), teacher.view(), &labels, &config)
                .expect("detailed loss failed");
        assert!(detail.soft_loss >= 0.0);
        assert!(detail.hard_loss > 0.0);
        let expected =
            config.alpha * detail.soft_loss + (1.0 - config.alpha) * detail.hard_loss;
        assert!((detail.total_loss - expected).abs() < 1e-12);
    }

    #[test]
    fn test_feature_distillation_same_dim() {
        let student_feat = array![[1.0_f64, 0.0, -1.0], [0.5, 0.5, 0.0]];
        let teacher_feat = array![[1.1_f64, 0.1, -0.9], [0.6, 0.4, 0.1]];
        let fd = FeatureDistillation::new(FeatureDistillationConfig::default());
        let loss =
            fd.loss(student_feat.view(), teacher_feat.view()).expect("feature loss failed");
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_feature_distillation_with_projection() {
        // student_dim=2, teacher_dim=3
        let student_feat = array![[1.0_f64, 0.0], [0.5, 0.5]];
        let teacher_feat = array![[1.0_f64, 0.5, 0.2], [0.3, 0.7, 0.1]];
        let proj = array![[1.0_f64, 0.0, 0.5], [0.0, 1.0, 0.5]]; // 2×3
        let fd = FeatureDistillation::new(FeatureDistillationConfig::default())
            .with_projection(proj);
        let loss =
            fd.loss(student_feat.view(), teacher_feat.view()).expect("projected feature loss");
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_attention_transfer_loss_identical_is_zero() {
        let attn = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let loss =
            attention_transfer_loss(attn.view(), attn.view()).expect("attn loss identical");
        assert!(loss < 1e-9, "AT loss for identical maps should be ~0, got {loss}");
    }

    #[test]
    fn test_attention_transfer_loss_different() {
        let s = array![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let t = array![[0.0_f64, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let loss = attention_transfer_loss(s.view(), t.view()).expect("attn loss different");
        assert!(loss > 0.0);
    }

    #[test]
    fn test_distillation_stats_accumulation() {
        let (student, teacher) = batch_logits();
        let labels = vec![0usize, 1, 2];
        let config = DistillationConfig::default();

        let mut stats = DistillationStats::default();
        for _ in 0..5 {
            let comp =
                distillation_loss_detailed(student.view(), teacher.view(), &labels, &config)
                    .expect("detail failed");
            stats.record(&comp);
        }
        assert_eq!(stats.n_batches, 5);
        assert!(stats.avg_total_loss() > 0.0);
    }

    #[test]
    fn test_calibrate_temperature() {
        let (student, teacher) = batch_logits();
        let best_t = calibrate_temperature(student.view(), teacher.view(), (1.0, 10.0), 20)
            .expect("calibrate_temperature failed");
        assert!(best_t >= 1.0 && best_t <= 10.0);
    }

    #[test]
    fn test_invalid_alpha() {
        let config = DistillationConfig {
            temperature: 4.0,
            alpha: 1.5,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_temperature() {
        let config = DistillationConfig {
            temperature: -1.0,
            alpha: 0.5,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_shape_mismatch_error() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let labels = vec![0usize, 1];
        let config = DistillationConfig::default();
        assert!(distillation_loss(a.view(), b.view(), &labels, &config).is_err());
    }

    #[test]
    fn test_label_out_of_range_error() {
        let (student, teacher) = batch_logits();
        let labels = vec![0usize, 1, 99]; // 99 is out of range for 3 classes
        let config = DistillationConfig::default();
        assert!(distillation_loss(student.view(), teacher.view(), &labels, &config).is_err());
    }
}
