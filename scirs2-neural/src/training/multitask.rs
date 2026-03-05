//! Multi-task learning for neural network training
//!
//! This module provides algorithms for training a single model on multiple
//! tasks simultaneously, with sophisticated gradient and loss weighting
//! strategies that go beyond naïve summation.
//!
//! ## Algorithms
//!
//! | Name | Paper |
//! |------|-------|
//! | `MultiTaskLoss` | Weighted task loss summation |
//! | `uncertainty_weighting` | Kendall et al., NeurIPS 2018 |
//! | `GradNorm` | Chen et al., ICML 2018 |
//! | `PCGrad` | Yu et al., NeurIPS 2020 |
//! | `MGDA` | Désidéri 2012; Sener & Koltun, NeurIPS 2018 |
//!
//! ## Quick start
//!
//! ```rust
//! use scirs2_neural::training::multitask::{TaskConfig, MultiTaskLoss};
//! use scirs2_core::ndarray::array;
//!
//! let tasks = vec![
//!     TaskConfig::new("depth", 1.0),
//!     TaskConfig::new("semantics", 1.0),
//! ];
//! let losses = vec![0.8_f64, 0.6];
//! let mtl = MultiTaskLoss::new(tasks).expect("mtl init failed");
//! let total = mtl.weighted_sum(&losses).expect("weighted sum failed");
//! assert!(total > 0.0);
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive, ToPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

// ─────────────────────────────────────────────────────────────────────────────
// TaskConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Per-task configuration for multi-task learning.
#[derive(Debug, Clone)]
pub struct TaskConfig {
    /// Unique task identifier string.
    pub task_id: String,
    /// Static loss weight for this task (used when no dynamic weighting is applied).
    pub loss_weight: f64,
    /// Optional display name for metric logging.
    pub metric_name: Option<String>,
}

impl TaskConfig {
    /// Create a new `TaskConfig` with an explicit loss weight.
    pub fn new(task_id: impl Into<String>, loss_weight: f64) -> Self {
        Self {
            task_id: task_id.into(),
            loss_weight,
            metric_name: None,
        }
    }

    /// Attach a metric name used in logging.
    pub fn with_metric(mut self, name: impl Into<String>) -> Self {
        self.metric_name = Some(name.into());
        self
    }

    /// Validate the task configuration.
    pub fn validate(&self) -> Result<()> {
        if self.task_id.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "task_id must not be empty".to_string(),
            ));
        }
        if !self.loss_weight.is_finite() || self.loss_weight < 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "loss_weight must be non-negative and finite, got {}",
                self.loss_weight
            )));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MultiTaskLoss
// ─────────────────────────────────────────────────────────────────────────────

/// Weighted combination of per-task losses.
///
/// The combined loss is:
/// ```text
/// L = Σ_t  w_t * L_t
/// ```
/// where `w_t` is the (optionally normalised) weight for task `t`.
#[derive(Debug, Clone)]
pub struct MultiTaskLoss {
    tasks: Vec<TaskConfig>,
    /// Whether to normalise weights so they sum to 1.
    pub normalise_weights: bool,
}

impl MultiTaskLoss {
    /// Construct from a list of task configurations.
    pub fn new(tasks: Vec<TaskConfig>) -> Result<Self> {
        if tasks.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "MultiTaskLoss requires at least one task".to_string(),
            ));
        }
        for t in &tasks {
            t.validate()?;
        }
        Ok(Self {
            tasks,
            normalise_weights: false,
        })
    }

    /// Enable weight normalisation (weights are divided by their sum before use).
    pub fn with_normalised_weights(mut self) -> Self {
        self.normalise_weights = true;
        self
    }

    /// Number of tasks.
    #[inline]
    pub fn num_tasks(&self) -> usize {
        self.tasks.len()
    }

    /// Task IDs in order.
    pub fn task_ids(&self) -> Vec<&str> {
        self.tasks.iter().map(|t| t.task_id.as_str()).collect()
    }

    /// Compute the weighted sum given one scalar loss per task.
    ///
    /// `losses[i]` corresponds to `tasks[i]`.
    pub fn weighted_sum(&self, losses: &[f64]) -> Result<f64> {
        self.check_len(losses.len())?;
        let weights = self.effective_weights()?;
        let total = losses
            .iter()
            .zip(weights.iter())
            .map(|(&l, &w)| w * l)
            .sum();
        Ok(total)
    }

    /// Compute weighted sum and return per-task contributions.
    pub fn weighted_sum_detailed(&self, losses: &[f64]) -> Result<TaskLossDetail> {
        self.check_len(losses.len())?;
        let weights = self.effective_weights()?;
        let contributions: Vec<f64> = losses
            .iter()
            .zip(weights.iter())
            .map(|(&l, &w)| w * l)
            .collect();
        let total = contributions.iter().sum();
        Ok(TaskLossDetail {
            task_ids: self.task_ids().iter().map(|s| s.to_string()).collect(),
            raw_losses: losses.to_vec(),
            weights: weights.to_vec(),
            contributions,
            total,
        })
    }

    /// Compute effective weights (possibly normalised).
    fn effective_weights(&self) -> Result<Vec<f64>> {
        let raw: Vec<f64> = self.tasks.iter().map(|t| t.loss_weight).collect();
        if self.normalise_weights {
            let sum: f64 = raw.iter().sum();
            if sum <= 0.0 {
                return Err(NeuralError::ComputationError(
                    "sum of loss weights is zero; cannot normalise".to_string(),
                ));
            }
            Ok(raw.iter().map(|&w| w / sum).collect())
        } else {
            Ok(raw)
        }
    }

    fn check_len(&self, n: usize) -> Result<()> {
        if n != self.tasks.len() {
            return Err(NeuralError::ShapeMismatch(format!(
                "expected {} losses (one per task), got {}",
                self.tasks.len(),
                n
            )));
        }
        Ok(())
    }
}

/// Detailed breakdown of a multi-task loss computation.
#[derive(Debug, Clone)]
pub struct TaskLossDetail {
    /// Task identifiers in order.
    pub task_ids: Vec<String>,
    /// Raw (unweighted) per-task scalar losses.
    pub raw_losses: Vec<f64>,
    /// Effective weights used (possibly normalised).
    pub weights: Vec<f64>,
    /// `weight * raw_loss` per task.
    pub contributions: Vec<f64>,
    /// Sum of all contributions.
    pub total: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// uncertainty_weighting  (Kendall et al. 2018)
// ─────────────────────────────────────────────────────────────────────────────

/// Homoscedastic uncertainty weighting (Kendall, Gal & Cipolla, NeurIPS 2018).
///
/// Each task loss is weighted by the learned log-variance `σ²` of its output:
/// ```text
/// L_total = Σ_t  [ L_t / (2 σ_t²)  +  log σ_t ]
/// ```
///
/// In practice the model learns `s_t = log(σ_t²)` for numerical stability:
/// ```text
/// L_total = Σ_t  [ L_t * exp(-s_t)  +  s_t / 2 ]
/// ```
///
/// # Parameters
/// - `losses`: Per-task scalar losses `L_t`.
/// - `log_variances`: Learned parameters `s_t = log σ_t²` (one per task).
///
/// # Returns
/// Total scalar loss and the per-task effective weights `exp(-s_t)`.
pub fn uncertainty_weighting(
    losses: &[f64],
    log_variances: &[f64],
) -> Result<UncertaintyWeightResult> {
    let n = losses.len();
    if n == 0 {
        return Err(NeuralError::InvalidArgument(
            "uncertainty_weighting requires at least one task".to_string(),
        ));
    }
    if log_variances.len() != n {
        return Err(NeuralError::ShapeMismatch(format!(
            "losses length {} != log_variances length {}",
            n,
            log_variances.len()
        )));
    }

    let mut total = 0.0_f64;
    let mut effective_weights = Vec::with_capacity(n);
    let mut per_task_contributions = Vec::with_capacity(n);

    for (&l, &s) in losses.iter().zip(log_variances.iter()) {
        // exp(-s) = 1 / σ²; precision weighting
        let precision = (-s).exp();
        effective_weights.push(precision);
        // regularisation term: log σ = s / 2
        let contrib = l * precision + s * 0.5;
        per_task_contributions.push(contrib);
        total += contrib;
    }

    Ok(UncertaintyWeightResult {
        total_loss: total,
        effective_weights,
        per_task_contributions,
    })
}

/// Output of `uncertainty_weighting`.
#[derive(Debug, Clone)]
pub struct UncertaintyWeightResult {
    /// Scalar total loss (sum of weighted losses + log regularisers).
    pub total_loss: f64,
    /// Effective precision weights `1/σ_t²` for each task.
    pub effective_weights: Vec<f64>,
    /// Per-task contribution to the total loss.
    pub per_task_contributions: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// GradNorm  (Chen et al., ICML 2018)
// ─────────────────────────────────────────────────────────────────────────────

/// State for GradNorm dynamic loss weight adjustment.
///
/// GradNorm balances tasks by forcing the gradient norms of each task's loss
/// (w.r.t. shared parameters) to match a target that is proportional to the
/// task's training speed.
///
/// ## Update rule (per training step)
/// 1. Compute gradient norms `||∇_W (w_t L_t)||` for each task.
/// 2. Compute inverse training rate `r_t = L_t(epoch) / L_t(0)`.
/// 3. Target norm: `G̃_W = mean_t(G_t) * r_t^α`.
/// 4. GradNorm loss: `L_gn = Σ_t |G_t - G̃_W|`.
/// 5. Backprop through `w_t` only (treat `G_t` as constant).
#[derive(Debug, Clone)]
pub struct GradNorm {
    /// Number of tasks.
    pub num_tasks: usize,
    /// Asymmetry hyperparameter (α ≥ 0). Larger = stronger rebalancing.
    pub alpha: f64,
    /// Initial per-task loss values `L_t(0)`.
    initial_losses: Vec<f64>,
    /// Current learned loss weights `w_t`.
    pub weights: Vec<f64>,
    /// Total number of update steps performed.
    pub steps: u64,
}

impl GradNorm {
    /// Initialise GradNorm with the per-task losses at step 0.
    pub fn new(initial_losses: Vec<f64>, alpha: f64) -> Result<Self> {
        let n = initial_losses.len();
        if n == 0 {
            return Err(NeuralError::InvalidArgument(
                "GradNorm requires at least one task".to_string(),
            ));
        }
        if alpha < 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "GradNorm alpha must be >= 0, got {alpha}"
            )));
        }
        for (i, &l) in initial_losses.iter().enumerate() {
            if !l.is_finite() || l <= 0.0 {
                return Err(NeuralError::InvalidArgument(format!(
                    "initial_losses[{i}] must be positive and finite, got {l}"
                )));
            }
        }
        // Start with uniform weights
        let weights = vec![1.0_f64; n];
        Ok(Self {
            num_tasks: n,
            alpha,
            initial_losses,
            weights,
            steps: 0,
        })
    }

    /// Compute GradNorm auxiliary loss and recommended weight gradient.
    ///
    /// # Parameters
    /// - `current_losses`: Scalar task losses at this step.
    /// - `grad_norms`: `||∇_W (w_t * L_t)||` for each task, computed using the
    ///   shared layer's gradient.
    ///
    /// # Returns
    /// [`GradNormOutput`] with updated weights and the GradNorm loss term.
    pub fn compute(
        &mut self,
        current_losses: &[f64],
        grad_norms: &[f64],
    ) -> Result<GradNormOutput> {
        self.check_len("current_losses", current_losses.len())?;
        self.check_len("grad_norms", grad_norms.len())?;

        // Inverse training rates: r_t = L_t(now) / L_t(0)
        let inv_rates: Vec<f64> = current_losses
            .iter()
            .zip(self.initial_losses.iter())
            .map(|(&l, &l0)| l / l0)
            .collect();

        // Mean inverse training rate
        let mean_rate = inv_rates.iter().sum::<f64>() / self.num_tasks as f64;

        // Mean gradient norm over tasks
        let mean_g: f64 = grad_norms.iter().sum::<f64>() / self.num_tasks as f64;

        // Target gradient norms
        let targets: Vec<f64> = inv_rates
            .iter()
            .map(|&r| mean_g * (r / mean_rate).powf(self.alpha))
            .collect();

        // GradNorm loss = Σ_t |G_t - target_t|
        let gradnorm_loss: f64 = grad_norms
            .iter()
            .zip(targets.iter())
            .map(|(&g, &tgt)| (g - tgt).abs())
            .sum();

        // Gradient of GradNorm loss w.r.t. w_t (treat G_t as constant)
        // dL_gn / dw_t = sign(G_t - target_t) * dG_t/dw_t
        // dG_t/dw_t ≈ G_t / w_t  (since G_t = w_t * ||∇ L_t||)
        let weight_grads: Vec<f64> = grad_norms
            .iter()
            .zip(targets.iter())
            .zip(self.weights.iter())
            .map(|((&g, &tgt), &w)| {
                let sign = if g > tgt { 1.0 } else { -1.0 };
                sign * g / w.max(1e-8)
            })
            .collect();

        self.steps += 1;

        Ok(GradNormOutput {
            gradnorm_loss,
            targets,
            weight_grads,
            current_weights: self.weights.clone(),
        })
    }

    /// Apply a gradient-descent update to the weights (learning rate `lr`).
    ///
    /// After updating the raw weights, they are renormalised so that
    /// `Σ_t w_t = num_tasks` (preserving the total loss scale).
    pub fn update_weights(&mut self, weight_grads: &[f64], lr: f64) -> Result<()> {
        self.check_len("weight_grads", weight_grads.len())?;
        for (w, &g) in self.weights.iter_mut().zip(weight_grads.iter()) {
            *w = (*w - lr * g).max(0.0);
        }
        // Renormalise: Σ w_t = num_tasks
        let sum: f64 = self.weights.iter().sum();
        if sum > 0.0 {
            let scale = self.num_tasks as f64 / sum;
            for w in &mut self.weights {
                *w *= scale;
            }
        }
        Ok(())
    }

    fn check_len(&self, name: &str, n: usize) -> Result<()> {
        if n != self.num_tasks {
            return Err(NeuralError::ShapeMismatch(format!(
                "{name}: expected {}, got {}",
                self.num_tasks, n
            )));
        }
        Ok(())
    }
}

/// Output of a [`GradNorm::compute`] call.
#[derive(Debug, Clone)]
pub struct GradNormOutput {
    /// Scalar GradNorm auxiliary loss.
    pub gradnorm_loss: f64,
    /// Target gradient norms `G̃_t` for each task.
    pub targets: Vec<f64>,
    /// Gradient of the GradNorm loss w.r.t. each weight `w_t`.
    pub weight_grads: Vec<f64>,
    /// Current weights at the time of the call.
    pub current_weights: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// PCGrad  (Yu et al., NeurIPS 2020)
// ─────────────────────────────────────────────────────────────────────────────

/// Projecting Conflicting Gradients (PCGrad).
///
/// For each task pair `(i, j)` where gradients conflict (`g_i · g_j < 0`),
/// project `g_i` onto the normal plane of `g_j`:
/// ```text
/// g_i ← g_i  -  (g_i · g_j / ||g_j||²) g_j
/// ```
/// The final gradient is the mean of the (possibly projected) per-task gradients.
///
/// Reference: Yu et al. "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.
pub struct PCGrad {
    /// Number of tasks.
    pub num_tasks: usize,
}

impl PCGrad {
    /// Create a new `PCGrad` instance for `num_tasks` tasks.
    pub fn new(num_tasks: usize) -> Result<Self> {
        if num_tasks < 2 {
            return Err(NeuralError::InvalidArgument(
                "PCGrad requires at least 2 tasks".to_string(),
            ));
        }
        Ok(Self { num_tasks })
    }

    /// Project conflicting gradients and return the combined gradient.
    ///
    /// # Parameters
    /// - `gradients`: A slice of `num_tasks` flat gradient vectors (each `Array1<f64>`).
    ///
    /// # Returns
    /// Combined (summed) gradient after conflict resolution.
    pub fn compute(&self, gradients: &[Array1<f64>]) -> Result<Array1<f64>> {
        if gradients.len() != self.num_tasks {
            return Err(NeuralError::ShapeMismatch(format!(
                "expected {} gradients, got {}",
                self.num_tasks,
                gradients.len()
            )));
        }
        let d = gradients[0].len();
        for (i, g) in gradients.iter().enumerate() {
            if g.len() != d {
                return Err(NeuralError::ShapeMismatch(format!(
                    "gradient {i} has length {} but gradient 0 has length {d}",
                    g.len()
                )));
            }
        }

        // Work on mutable copies
        let mut projected: Vec<Array1<f64>> = gradients.to_vec();

        // For each task i, project onto the normal plane of each conflicting task j
        for i in 0..self.num_tasks {
            for j in 0..self.num_tasks {
                if i == j {
                    continue;
                }
                let dot_ij = projected[i]
                    .iter()
                    .zip(gradients[j].iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>();
                if dot_ij < 0.0 {
                    let norm_j_sq = gradients[j].iter().map(|&x| x * x).sum::<f64>();
                    if norm_j_sq > 0.0 {
                        let scale = dot_ij / norm_j_sq;
                        // projected[i] -= scale * gradients[j]
                        for (pi, &gj) in projected[i].iter_mut().zip(gradients[j].iter()) {
                            *pi -= scale * gj;
                        }
                    }
                }
            }
        }

        // Sum all projected gradients
        let mut combined = Array1::<f64>::zeros(d);
        for g in &projected {
            for (c, &v) in combined.iter_mut().zip(g.iter()) {
                *c += v;
            }
        }
        Ok(combined)
    }

    /// Compute per-task conflict statistics for monitoring.
    ///
    /// Returns a matrix where entry `(i, j)` is `true` if tasks `i` and `j`
    /// have conflicting gradients (dot product < 0).
    pub fn conflict_matrix(&self, gradients: &[Array1<f64>]) -> Result<Array2<bool>> {
        if gradients.len() != self.num_tasks {
            return Err(NeuralError::ShapeMismatch(format!(
                "expected {} gradients, got {}",
                self.num_tasks,
                gradients.len()
            )));
        }
        let n = self.num_tasks;
        let mut mat = Array2::<bool>::from_elem((n, n), false);
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dot = gradients[i]
                        .iter()
                        .zip(gradients[j].iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f64>();
                    mat[[i, j]] = dot < 0.0;
                }
            }
        }
        Ok(mat)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MGDA  (Sener & Koltun, NeurIPS 2018)
// ─────────────────────────────────────────────────────────────────────────────

/// Multiple Gradient Descent Algorithm (MGDA) for Pareto-optimal updates.
///
/// MGDA finds the minimum-norm point in the convex hull of per-task gradients,
/// which is the Pareto-stationary direction.  This corresponds to solving:
/// ```text
/// min_{α ∈ Δ^{T-1}}  ||Σ_t α_t g_t||²
/// ```
/// The solution `α*` then gives a Pareto-improving (or Pareto-stationary) update
/// direction `d = Σ_t α_t* g_t`.
///
/// References:
/// - Désidéri, "Multiple-gradient descent algorithm (MGDA) for multiobjective
///   optimization", Comptes Rendus Mathématique, 2012.
/// - Sener & Koltun, "Multi-Task Learning as Multi-Objective Optimization",
///   NeurIPS 2018.
pub struct MGDA {
    /// Convergence tolerance for the Frank-Wolfe solver.
    pub tol: f64,
    /// Maximum number of Frank-Wolfe iterations.
    pub max_iter: usize,
}

impl Default for MGDA {
    fn default() -> Self {
        Self {
            tol: 1e-5,
            max_iter: 250,
        }
    }
}

impl MGDA {
    /// Create MGDA with custom solver settings.
    pub fn new(tol: f64, max_iter: usize) -> Result<Self> {
        if tol <= 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "tol must be positive, got {tol}"
            )));
        }
        if max_iter == 0 {
            return Err(NeuralError::InvalidArgument(
                "max_iter must be at least 1".to_string(),
            ));
        }
        Ok(Self { tol, max_iter })
    }

    /// Compute the Pareto-stationary combined gradient.
    ///
    /// # Parameters
    /// - `gradients`: One flat `Array1<f64>` per task.
    ///
    /// # Returns
    /// [`MgdaOutput`] with the combined gradient, task weights `α`, and convergence info.
    pub fn compute(&self, gradients: &[Array1<f64>]) -> Result<MgdaOutput> {
        let n = gradients.len();
        if n == 0 {
            return Err(NeuralError::InvalidArgument(
                "MGDA requires at least one gradient".to_string(),
            ));
        }
        let d = gradients[0].len();
        for (i, g) in gradients.iter().enumerate() {
            if g.len() != d {
                return Err(NeuralError::ShapeMismatch(format!(
                    "gradient {i} has length {} but gradient 0 has length {d}",
                    g.len()
                )));
            }
        }

        // Frank-Wolfe algorithm on the unit simplex
        // α starts uniform
        let mut alpha = vec![1.0_f64 / n as f64; n];
        let mut converged = false;
        let mut iters = 0usize;

        for iter in 0..self.max_iter {
            iters = iter + 1;

            // Compute current combined gradient d = Σ_t α_t g_t
            let combined = self.linear_combination(gradients, &alpha, d);

            // Gradient of the objective (||combined||²) w.r.t. α_t  =  2 * combined · g_t
            let grads_obj: Vec<f64> = gradients
                .iter()
                .map(|g| 2.0 * dot_product(&combined, g))
                .collect();

            // Frank-Wolfe: move to vertex minimising linear approximation
            let min_idx = grads_obj
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            // Step size via exact line search
            let s_alpha = {
                let mut s = vec![0.0_f64; n];
                s[min_idx] = 1.0;
                s
            };

            // Direction = s_alpha - alpha
            let direction: Vec<f64> = s_alpha
                .iter()
                .zip(alpha.iter())
                .map(|(&s, &a)| s - a)
                .collect();

            // d_direction = Σ_t direction_t * g_t
            let d_dir = self.linear_combination(gradients, &direction, d);

            // Optimal step size: minimise ||combined + γ d_dir||²
            // γ* = -<combined, d_dir> / <d_dir, d_dir>   clamped to [0, 1]
            let num = -dot_product(&combined, &d_dir);
            let denom = dot_product(&d_dir, &d_dir);

            let gamma = if denom.abs() < 1e-12 {
                0.0
            } else {
                (num / denom).clamp(0.0, 1.0)
            };

            // Convergence check
            if gamma.abs() < self.tol {
                converged = true;
                break;
            }

            for (a, &dir) in alpha.iter_mut().zip(direction.iter()) {
                *a += gamma * dir;
                // Numerical safety: keep on simplex
                *a = a.max(0.0);
            }
            // Renormalise to simplex
            let s: f64 = alpha.iter().sum();
            if s > 0.0 {
                for a in &mut alpha {
                    *a /= s;
                }
            }
        }

        let combined = self.linear_combination(gradients, &alpha, d);
        let norm = dot_product(&combined, &combined).sqrt();

        Ok(MgdaOutput {
            combined_gradient: combined,
            task_weights: alpha,
            pareto_stationary: norm < self.tol * 10.0,
            converged,
            iterations: iters,
        })
    }

    fn linear_combination(
        &self,
        gradients: &[Array1<f64>],
        coeffs: &[f64],
        d: usize,
    ) -> Array1<f64> {
        let mut result = Array1::<f64>::zeros(d);
        for (g, &c) in gradients.iter().zip(coeffs.iter()) {
            for (r, &v) in result.iter_mut().zip(g.iter()) {
                *r += c * v;
            }
        }
        result
    }
}

/// Output of [`MGDA::compute`].
#[derive(Debug, Clone)]
pub struct MgdaOutput {
    /// The Pareto-stationary combined gradient.
    pub combined_gradient: Array1<f64>,
    /// Optimal simplex weights `α_t` for each task.
    pub task_weights: Vec<f64>,
    /// Whether the combined gradient is approximately zero (Pareto-stationary).
    pub pareto_stationary: bool,
    /// Whether the Frank-Wolfe solver converged within `max_iter`.
    pub converged: bool,
    /// Number of Frank-Wolfe iterations used.
    pub iterations: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn dot_product(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-task training statistics tracker
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks per-task losses and weights across training steps.
#[derive(Debug, Clone)]
pub struct MultiTaskStats {
    /// Task identifiers.
    pub task_ids: Vec<String>,
    /// Running sum of per-task losses.
    loss_sums: Vec<f64>,
    /// Running sum of per-task weights.
    weight_sums: Vec<f64>,
    /// Number of accumulated steps.
    pub steps: u64,
}

impl MultiTaskStats {
    /// Create a new stats tracker for the given task IDs.
    pub fn new(task_ids: Vec<String>) -> Self {
        let n = task_ids.len();
        Self {
            task_ids,
            loss_sums: vec![0.0; n],
            weight_sums: vec![0.0; n],
            steps: 0,
        }
    }

    /// Record one step's worth of losses and weights.
    pub fn record(&mut self, losses: &[f64], weights: &[f64]) -> Result<()> {
        let n = self.task_ids.len();
        if losses.len() != n {
            return Err(NeuralError::ShapeMismatch(format!(
                "expected {n} losses, got {}",
                losses.len()
            )));
        }
        if weights.len() != n {
            return Err(NeuralError::ShapeMismatch(format!(
                "expected {n} weights, got {}",
                weights.len()
            )));
        }
        for i in 0..n {
            self.loss_sums[i] += losses[i];
            self.weight_sums[i] += weights[i];
        }
        self.steps += 1;
        Ok(())
    }

    /// Average per-task losses over accumulated steps.
    pub fn avg_losses(&self) -> Vec<f64> {
        if self.steps == 0 {
            return vec![0.0; self.task_ids.len()];
        }
        self.loss_sums
            .iter()
            .map(|&s| s / self.steps as f64)
            .collect()
    }

    /// Average per-task weights over accumulated steps.
    pub fn avg_weights(&self) -> Vec<f64> {
        if self.steps == 0 {
            return vec![0.0; self.task_ids.len()];
        }
        self.weight_sums
            .iter()
            .map(|&s| s / self.steps as f64)
            .collect()
    }

    /// Reset all accumulators.
    pub fn reset(&mut self) {
        for v in self.loss_sums.iter_mut() {
            *v = 0.0;
        }
        for v in self.weight_sums.iter_mut() {
            *v = 0.0;
        }
        self.steps = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // ── TaskConfig ────────────────────────────────────────────────────────

    #[test]
    fn test_task_config_validation_ok() {
        let t = TaskConfig::new("depth", 1.5);
        assert!(t.validate().is_ok());
    }

    #[test]
    fn test_task_config_empty_id() {
        let t = TaskConfig::new("", 1.0);
        assert!(t.validate().is_err());
    }

    #[test]
    fn test_task_config_negative_weight() {
        let t = TaskConfig::new("seg", -0.5);
        assert!(t.validate().is_err());
    }

    // ── MultiTaskLoss ─────────────────────────────────────────────────────

    fn make_mtl() -> MultiTaskLoss {
        MultiTaskLoss::new(vec![
            TaskConfig::new("depth", 1.0),
            TaskConfig::new("seg", 2.0),
            TaskConfig::new("normal", 0.5),
        ])
        .expect("mtl init")
    }

    #[test]
    fn test_multitaskloss_weighted_sum() {
        let mtl = make_mtl();
        let losses = vec![0.4, 0.6, 0.3];
        let total = mtl.weighted_sum(&losses).expect("weighted sum");
        // 1.0*0.4 + 2.0*0.6 + 0.5*0.3 = 0.4 + 1.2 + 0.15 = 1.75
        assert!((total - 1.75).abs() < 1e-10);
    }

    #[test]
    fn test_multitaskloss_normalised() {
        let mtl = make_mtl().with_normalised_weights();
        let losses = vec![1.0, 1.0, 1.0];
        let total = mtl.weighted_sum(&losses).expect("normalised weighted sum");
        // All equal losses → normalised weights sum to 1 → total = 1.0
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_multitaskloss_wrong_len() {
        let mtl = make_mtl();
        assert!(mtl.weighted_sum(&[0.1, 0.2]).is_err());
    }

    #[test]
    fn test_multitaskloss_detailed() {
        let mtl = make_mtl();
        let losses = vec![0.4, 0.6, 0.3];
        let detail = mtl.weighted_sum_detailed(&losses).expect("detailed");
        assert_eq!(detail.task_ids.len(), 3);
        assert!((detail.total - 1.75).abs() < 1e-10);
    }

    // ── uncertainty_weighting ─────────────────────────────────────────────

    #[test]
    fn test_uncertainty_weighting_basic() {
        let losses = vec![0.5, 0.3];
        let log_vars = vec![0.0, 0.0]; // σ² = 1 → weight = 1
        let result = uncertainty_weighting(&losses, &log_vars).expect("uw");
        // L = 0.5 * 1 + 0 + 0.3 * 1 + 0 = 0.8
        assert!((result.total_loss - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_uncertainty_weighting_high_uncertainty() {
        let losses = vec![1.0];
        // log(σ²) = 2.0 → σ² = e² ≈ 7.39 → precision = e^-2 ≈ 0.135
        let log_vars = vec![2.0_f64];
        let result = uncertainty_weighting(&losses, &log_vars).expect("uw high unc");
        // L = 1.0 * e^(-2) + 1.0 = 1.0 * 0.135 + 1.0
        let expected = 1.0_f64 * (-2.0_f64).exp() + 1.0;
        assert!((result.total_loss - expected).abs() < 1e-10);
    }

    #[test]
    fn test_uncertainty_weighting_len_mismatch() {
        assert!(uncertainty_weighting(&[0.5], &[0.0, 0.0]).is_err());
    }

    // ── GradNorm ──────────────────────────────────────────────────────────

    #[test]
    fn test_gradnorm_init_uniform_weights() {
        let gn = GradNorm::new(vec![1.0, 2.0, 3.0], 1.5).expect("gn init");
        assert_eq!(gn.weights, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_gradnorm_compute_returns_output() {
        let mut gn = GradNorm::new(vec![1.0, 2.0], 1.5).expect("gn init");
        let out = gn
            .compute(&[0.9, 2.2], &[0.5, 0.6])
            .expect("gn compute");
        assert!(out.gradnorm_loss >= 0.0);
        assert_eq!(out.targets.len(), 2);
    }

    #[test]
    fn test_gradnorm_update_weights_sums_to_num_tasks() {
        let mut gn = GradNorm::new(vec![1.0, 2.0], 1.5).expect("gn init");
        let out = gn.compute(&[0.9, 2.2], &[0.5, 0.6]).expect("gn compute");
        gn.update_weights(&out.weight_grads, 0.01)
            .expect("update weights");
        let sum: f64 = gn.weights.iter().sum();
        assert!((sum - 2.0).abs() < 1e-10);
    }

    // ── PCGrad ────────────────────────────────────────────────────────────

    #[test]
    fn test_pcgrad_no_conflict() {
        let pcg = PCGrad::new(2).expect("pcg");
        // Parallel gradients → no projection
        let g1 = array![1.0_f64, 0.0];
        let g2 = array![0.5_f64, 0.0];
        let combined = pcg.compute(&[g1.clone(), g2.clone()]).expect("pcg compute");
        // Combined should be sum of both (no conflict)
        assert!((combined[0] - 1.5).abs() < 1e-9);
        assert!(combined[1].abs() < 1e-9);
    }

    #[test]
    fn test_pcgrad_conflict_projection() {
        let pcg = PCGrad::new(2).expect("pcg");
        // Opposite gradients → both projected to zero
        let g1 = array![1.0_f64, 0.0];
        let g2 = array![-1.0_f64, 0.0];
        let combined = pcg.compute(&[g1, g2]).expect("pcg conflict");
        // After projection each should be ~zero along conflict direction
        assert!(combined[0].abs() < 1e-9);
    }

    #[test]
    fn test_pcgrad_conflict_matrix() {
        let pcg = PCGrad::new(2).expect("pcg");
        let g1 = array![1.0_f64, 0.0];
        let g2 = array![-1.0_f64, 0.0];
        let mat = pcg.conflict_matrix(&[g1, g2]).expect("conflict mat");
        assert!(mat[[0, 1]]);
        assert!(mat[[1, 0]]);
        assert!(!mat[[0, 0]]);
    }

    // ── MGDA ──────────────────────────────────────────────────────────────

    #[test]
    fn test_mgda_single_task_is_identity() {
        let mgda = MGDA::default();
        let g = array![0.3_f64, -0.7, 1.2];
        let out = mgda.compute(&[g.clone()]).expect("mgda single");
        for (&c, &orig) in out.combined_gradient.iter().zip(g.iter()) {
            assert!((c - orig).abs() < 1e-10);
        }
        assert!((out.task_weights[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mgda_two_tasks_pareto_direction() {
        let mgda = MGDA::default();
        let g1 = array![1.0_f64, 0.0];
        let g2 = array![0.0_f64, 1.0];
        let out = mgda.compute(&[g1, g2]).expect("mgda two tasks");
        // By symmetry, weights should be approximately equal (0.5 each)
        assert!((out.task_weights[0] - 0.5).abs() < 0.05);
        assert!((out.task_weights[1] - 0.5).abs() < 0.05);
        // Sum of weights = 1
        let sum: f64 = out.task_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mgda_conflicting_tasks_pareto_stationary() {
        let mgda = MGDA::default();
        // Opposite unit gradients → Pareto-stationary at zero
        let g1 = array![1.0_f64, 0.0];
        let g2 = array![-1.0_f64, 0.0];
        let out = mgda.compute(&[g1, g2]).expect("mgda conflict");
        // Combined should be near zero
        let norm: f64 = out
            .combined_gradient
            .iter()
            .map(|&x| x * x)
            .sum::<f64>()
            .sqrt();
        assert!(norm < 0.1, "expected near-zero combined gradient, got norm {norm}");
    }

    // ── MultiTaskStats ────────────────────────────────────────────────────

    #[test]
    fn test_multitaskstats_avg_losses() {
        let mut stats =
            MultiTaskStats::new(vec!["a".to_string(), "b".to_string()]);
        stats.record(&[1.0, 2.0], &[1.0, 1.0]).expect("record 1");
        stats.record(&[3.0, 4.0], &[1.0, 1.0]).expect("record 2");
        let avgs = stats.avg_losses();
        assert!((avgs[0] - 2.0).abs() < 1e-10);
        assert!((avgs[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_multitaskstats_reset() {
        let mut stats = MultiTaskStats::new(vec!["a".to_string()]);
        stats.record(&[5.0], &[1.0]).expect("record");
        stats.reset();
        assert_eq!(stats.steps, 0);
        assert_eq!(stats.avg_losses(), vec![0.0]);
    }
}
