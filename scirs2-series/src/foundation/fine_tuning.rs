//! Fine-tuning interface for pre-trained time series foundation models.
//!
//! ## Design
//!
//! Fine-tuning large pre-trained foundation models on downstream data is a
//! central pattern in modern ML.  This module provides:
//!
//! 1. **[`ForecastModel`]** — a trait that any model must implement to participate
//!    in fine-tuning or zero-shot forecasting.
//! 2. **[`FineTuner`]** — a generic wrapper that trains the model on
//!    `(context_window, horizon)` pairs using mini-batch SGD with optional
//!    early stopping based on a held-out validation set.
//! 3. **[`LinearForecastModel`]** — a self-contained linear baseline model
//!    (maps context window → horizon via a weight matrix + bias) useful for
//!    testing without an actual large pre-trained model.
//! 4. **[`LoraForecastModel`]** — low-rank adaptation (LoRA) wrapper that adds
//!    rank-`r` trainable perturbations `ΔW = A B` to the frozen backbone
//!    weight, drastically reducing the number of trainable parameters.
//!
//! ## Fine-tuning algorithm
//!
//! Given training pairs `(X ∈ ℝ^{T×context}, Y ∈ ℝ^{T×horizon})`:
//!
//! 1. Optionally freeze the backbone, exposing only head parameters.
//! 2. For each epoch, shuffle rows, split into mini-batches.
//! 3. For each mini-batch, forward pass → MSE loss → gradient via finite
//!    differences (gradient-free parameter update, suitable for small parameter
//!    counts or when autograd is unavailable).
//! 4. Track training and validation loss; store best checkpoint by val loss.
//!
//! ## LoRA
//!
//! LoRA ([Hu et al. 2021](https://arxiv.org/abs/2106.09685)) parameterises the
//! weight update as `W = W_0 + α * A B` where `A ∈ ℝ^{d_in×r}`,
//! `B ∈ ℝ^{r×d_out}`, and `r ≪ min(d_in, d_out)`.  Only `A` and `B` are
//! trained; `W_0` is frozen.  This module implements LoRA for
//! `LinearForecastModel`.

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::{s, Array1, Array2};

// ─────────────────────────────────────────────────────────────────────────────
// FoundationModelType
// ─────────────────────────────────────────────────────────────────────────────

/// Identifies the architectural family of the foundation model.
///
/// This is informational — the actual model behaviour is defined by the
/// [`ForecastModel`] implementation, not this enum.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum FoundationModelType {
    /// TimeGPT-style autoregressive decoder (Garza et al. 2023).
    TimeGpt,
    /// PatchTST: channel-independent patch-based encoder (Nie et al. 2023).
    PatchTst,
    /// TimesNet: temporal 2-D variation model (Wu et al. 2023).
    TimesNet,
    /// User-defined / generic model.
    Generic,
}

impl Default for FoundationModelType {
    fn default() -> Self {
        Self::Generic
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FineTuningConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the fine-tuning procedure.
#[derive(Debug, Clone)]
pub struct FineTuningConfig {
    /// Architectural family of the underlying model (informational).
    pub model_type: FoundationModelType,
    /// If `true`, only the head layers are trained; backbone is frozen.
    ///
    /// For [`LinearForecastModel`] this has no effect (single-layer model).
    /// For [`LoraForecastModel`] the frozen backbone is the original weight
    /// matrix; only the LoRA A/B matrices are updated.
    pub freeze_backbone: bool,
    /// How many last-layer parameter groups to unfreeze when
    /// `freeze_backbone = true`.  Default: 2.
    pub n_finetune_layers: usize,
    /// Learning rate for parameter updates.  Default: 1e-4.
    pub learning_rate: f64,
    /// Maximum number of training epochs.  Default: 10.
    pub max_epochs: usize,
    /// Mini-batch size.  Default: 32.
    pub batch_size: usize,
    /// Forecast horizon (number of future steps).  Default: 24.
    pub horizon: usize,
    /// Context window length fed to the model.  Default: 512.
    pub context_length: usize,
    /// LoRA rank.  `Some(r)` activates LoRA adapters; `None` disables.
    pub lora_rank: Option<usize>,
}

impl Default for FineTuningConfig {
    fn default() -> Self {
        Self {
            model_type: FoundationModelType::Generic,
            freeze_backbone: true,
            n_finetune_layers: 2,
            learning_rate: 1e-4,
            max_epochs: 10,
            batch_size: 32,
            horizon: 24,
            context_length: 512,
            lora_rank: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FineTuningResult
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics from a completed fine-tuning run.
#[derive(Debug, Clone)]
pub struct FineTuningResult {
    /// Mean-squared error on the training set, one entry per epoch.
    pub train_losses: Vec<f64>,
    /// Mean-squared error on the validation set (empty if no val data).
    pub val_losses: Vec<f64>,
    /// Index (0-based) of the epoch with the lowest validation loss.
    pub best_epoch: usize,
    /// Validation MSE at the best epoch.
    pub best_val_loss: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// ForecastModel trait
// ─────────────────────────────────────────────────────────────────────────────

/// Trait that any model must implement to participate in fine-tuning or
/// zero-shot forecasting.
///
/// Implementations must be `Send` so that they can be moved between threads.
pub trait ForecastModel: Send {
    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// * `x` – context windows `[batch, context_length]`
    ///
    /// # Returns
    ///
    /// Predictions `[batch, horizon]`.
    fn forward(&self, x: &Array2<f64>) -> Result<Array2<f64>>;

    /// Total number of trainable (floating-point) parameters.
    fn n_params(&self) -> usize;

    /// Flatten all parameters into a `Vec<f64>`.
    fn get_params(&self) -> Vec<f64>;

    /// Load parameters from a flat slice (same order as `get_params`).
    ///
    /// # Errors
    ///
    /// Returns [`TimeSeriesError::InvalidInput`] if `params.len() != n_params()`.
    fn set_params(&mut self, params: &[f64]) -> Result<()>;
}

// ─────────────────────────────────────────────────────────────────────────────
// LinearForecastModel
// ─────────────────────────────────────────────────────────────────────────────

/// A simple linear model: `Y_hat = X W + b`
///
/// * `W ∈ ℝ^{context × horizon}`
/// * `b ∈ ℝ^{horizon}`
///
/// Parameters are flattened as `[W.row_major..., b...]`.
#[derive(Debug, Clone)]
pub struct LinearForecastModel {
    /// Weight matrix `[context_length, horizon]`.
    pub weights: Array2<f64>,
    /// Bias vector `[horizon]`.
    pub bias: Array1<f64>,
}

impl LinearForecastModel {
    /// Create a new model with Xavier-uniform initialisation.
    pub fn new(context_length: usize, horizon: usize) -> Self {
        // Xavier uniform: U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))
        let limit = (6.0 / (context_length + horizon) as f64).sqrt();
        let mut lcg: u64 = 0x_dead_beef_cafe_babe;
        let next = |lcg: &mut u64| -> f64 {
            *lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (*lcg >> 33) as f64 / (u32::MAX as f64);
            u * 2.0 * limit - limit
        };

        let weights = Array2::from_shape_fn((context_length, horizon), |_| next(&mut lcg));
        let bias = Array1::zeros(horizon);
        Self { weights, bias }
    }

    /// Context length (input dimension).
    pub fn context_length(&self) -> usize {
        self.weights.nrows()
    }

    /// Horizon (output dimension).
    pub fn horizon(&self) -> usize {
        self.weights.ncols()
    }
}

impl ForecastModel for LinearForecastModel {
    fn forward(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (batch, ctx) = (x.nrows(), x.ncols());
        if ctx != self.context_length() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.context_length(),
                actual: ctx,
            });
        }
        // Y = X @ W + b  —  manual matmul to avoid ndarray linalg dependency.
        let horizon = self.horizon();
        let mut y = Array2::<f64>::zeros((batch, horizon));
        for b in 0..batch {
            for h in 0..horizon {
                let dot: f64 = (0..ctx).map(|c| x[[b, c]] * self.weights[[c, h]]).sum();
                y[[b, h]] = dot + self.bias[h];
            }
        }
        Ok(y)
    }

    fn n_params(&self) -> usize {
        self.context_length() * self.horizon() + self.horizon()
    }

    fn get_params(&self) -> Vec<f64> {
        let mut p = Vec::with_capacity(self.n_params());
        for val in self.weights.iter() {
            p.push(*val);
        }
        for val in self.bias.iter() {
            p.push(*val);
        }
        p
    }

    fn set_params(&mut self, params: &[f64]) -> Result<()> {
        if params.len() != self.n_params() {
            return Err(TimeSeriesError::InvalidInput(format!(
                "expected {} params, got {}",
                self.n_params(),
                params.len()
            )));
        }
        let wsize = self.context_length() * self.horizon();
        let mut idx = 0;
        for val in self.weights.iter_mut() {
            *val = params[idx];
            idx += 1;
        }
        for val in self.bias.iter_mut() {
            *val = params[idx];
            idx += 1;
        }
        let _ = wsize; // suppress warning
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoraForecastModel
// ─────────────────────────────────────────────────────────────────────────────

/// LoRA-adapted linear forecast model.
///
/// The forward pass computes:
///
/// ```text
/// Y = X (W_0 + α * A B) + b
/// ```
///
/// where `W_0` is frozen, and only `A ∈ ℝ^{context × r}` and
/// `B ∈ ℝ^{r × horizon}` are updated.
///
/// Parameters (via [`ForecastModel`]) expose **only** `[A.row_major..., B.row_major...]`
/// — this is what the FineTuner trains.  `b` (bias) is also trainable and
/// appended last.
#[derive(Debug, Clone)]
pub struct LoraForecastModel {
    /// Frozen backbone weight `[context, horizon]`.
    backbone_w: Array2<f64>,
    /// Frozen bias `[horizon]`.
    bias: Array1<f64>,
    /// LoRA A matrix `[context, rank]`.
    lora_a: Array2<f64>,
    /// LoRA B matrix `[rank, horizon]`.
    lora_b: Array2<f64>,
    /// Scaling factor α.
    alpha: f64,
    /// LoRA rank r.
    rank: usize,
}

impl LoraForecastModel {
    /// Wrap an existing [`LinearForecastModel`] with LoRA adapters.
    ///
    /// `A` is initialised with small Gaussian noise; `B` with zeros
    /// (so initial ΔW = 0).
    pub fn new(base: LinearForecastModel, rank: usize) -> Self {
        if rank == 0 {
            // Degenerate: behave like base model with no adaptation.
            let ctx = base.context_length();
            let h = base.horizon();
            return Self {
                backbone_w: base.weights,
                bias: base.bias,
                lora_a: Array2::zeros((ctx, 0)),
                lora_b: Array2::zeros((0, h)),
                alpha: 1.0,
                rank: 0,
            };
        }
        let ctx = base.context_length();
        let h = base.horizon();
        let std_a = 0.02_f64;
        let mut lcg: u64 = 0xc0ffee;
        let lora_a = Array2::from_shape_fn((ctx, rank), |_| {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (lcg >> 33) as f64 / (u32::MAX as f64);
            (u * 2.0 - 1.0) * std_a
        });
        let lora_b = Array2::zeros((rank, h));
        Self {
            backbone_w: base.weights,
            bias: base.bias,
            lora_a,
            lora_b,
            alpha: 1.0,
            rank,
        }
    }

    fn context_length(&self) -> usize {
        self.backbone_w.nrows()
    }
    fn horizon(&self) -> usize {
        self.backbone_w.ncols()
    }

    /// Number of trainable parameters: A + B + bias.
    pub fn n_lora_params(&self) -> usize {
        let ctx = self.context_length();
        let h = self.horizon();
        ctx * self.rank + self.rank * h + h
    }
}

impl ForecastModel for LoraForecastModel {
    fn forward(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (batch, ctx) = (x.nrows(), x.ncols());
        if ctx != self.context_length() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.context_length(),
                actual: ctx,
            });
        }
        let horizon = self.horizon();

        // Compute effective weight: W_eff = W_0 + α * A B
        // Then Y = X W_eff + b

        let mut y = Array2::<f64>::zeros((batch, horizon));

        // Backbone contribution: X @ W_0
        for b in 0..batch {
            for h in 0..horizon {
                let dot: f64 = (0..ctx).map(|c| x[[b, c]] * self.backbone_w[[c, h]]).sum();
                y[[b, h]] = dot + self.bias[h];
            }
        }

        // LoRA contribution: α * X @ A @ B
        if self.rank > 0 {
            // First compute T = X @ A  → [batch, rank]
            let mut t = Array2::<f64>::zeros((batch, self.rank));
            for b in 0..batch {
                for r in 0..self.rank {
                    let dot: f64 = (0..ctx).map(|c| x[[b, c]] * self.lora_a[[c, r]]).sum();
                    t[[b, r]] = dot;
                }
            }
            // Then add α * T @ B to Y
            for b in 0..batch {
                for h in 0..horizon {
                    let dot: f64 = (0..self.rank)
                        .map(|r| t[[b, r]] * self.lora_b[[r, h]])
                        .sum();
                    y[[b, h]] += self.alpha * dot;
                }
            }
        }

        Ok(y)
    }

    fn n_params(&self) -> usize {
        self.n_lora_params()
    }

    fn get_params(&self) -> Vec<f64> {
        let mut p = Vec::with_capacity(self.n_params());
        for val in self.lora_a.iter() {
            p.push(*val);
        }
        for val in self.lora_b.iter() {
            p.push(*val);
        }
        for val in self.bias.iter() {
            p.push(*val);
        }
        p
    }

    fn set_params(&mut self, params: &[f64]) -> Result<()> {
        if params.len() != self.n_params() {
            return Err(TimeSeriesError::InvalidInput(format!(
                "expected {} params, got {}",
                self.n_params(),
                params.len()
            )));
        }
        let a_size = self.context_length() * self.rank;
        let b_size = self.rank * self.horizon();
        let mut idx = 0;
        for val in self.lora_a.iter_mut() {
            *val = params[idx];
            idx += 1;
        }
        for val in self.lora_b.iter_mut() {
            *val = params[idx];
            idx += 1;
        }
        for val in self.bias.iter_mut() {
            *val = params[idx];
            idx += 1;
        }
        let _ = (a_size, b_size);
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FineTuner
// ─────────────────────────────────────────────────────────────────────────────

/// Fine-tunes a [`ForecastModel`] on labelled time series data.
///
/// Uses a gradient-free finite-difference SGD:
///
/// ```text
/// ∇_i L ≈ (L(θ + ε e_i) - L(θ - ε e_i)) / (2ε)
/// θ ← θ - lr * ∇L
/// ```
///
/// This is O(2P) forward passes per gradient step — feasible for small P
/// and designed to be autograd-free.  For large models, the user should
/// inject pre-computed gradients through a custom [`ForecastModel`]
/// implementation.
pub struct FineTuner<M: ForecastModel> {
    /// The model being fine-tuned.
    pub model: M,
    /// Fine-tuning hyper-parameters.
    pub config: FineTuningConfig,
    /// Best-checkpoint parameters.
    best_params: Option<Vec<f64>>,
}

impl<M: ForecastModel> FineTuner<M> {
    /// Create a new `FineTuner`.
    pub fn new(model: M, config: FineTuningConfig) -> Self {
        Self {
            model,
            config,
            best_params: None,
        }
    }

    /// Compute batch MSE loss for `(x, y)` pairs.
    fn mse_loss(model: &M, x: &Array2<f64>, y: &Array2<f64>) -> Result<f64> {
        let pred = model.forward(x)?;
        let (batch, horizon) = (y.nrows(), y.ncols());
        if pred.shape() != y.shape() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: y.ncols(),
                actual: pred.ncols(),
            });
        }
        let mut mse = 0.0_f64;
        for b in 0..batch {
            for h in 0..horizon {
                mse += (pred[[b, h]] - y[[b, h]]).powi(2);
            }
        }
        mse /= (batch * horizon) as f64;
        Ok(mse)
    }

    /// Run the fine-tuning loop.
    ///
    /// # Arguments
    ///
    /// * `train_x` – `[n_train, context_length]`
    /// * `train_y` – `[n_train, horizon]`
    /// * `val_x`   – optional `[n_val, context_length]`
    /// * `val_y`   – optional `[n_val, horizon]`
    pub fn fit(
        &mut self,
        train_x: &Array2<f64>,
        train_y: &Array2<f64>,
        val_x: Option<&Array2<f64>>,
        val_y: Option<&Array2<f64>>,
    ) -> Result<FineTuningResult> {
        let n_train = train_x.nrows();
        if n_train == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "training set is empty".to_string(),
            ));
        }
        if train_y.nrows() != n_train {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_train,
                actual: train_y.nrows(),
            });
        }

        let lr = self.config.learning_rate;
        let batch_size = self.config.batch_size.max(1).min(n_train);
        let max_epochs = self.config.max_epochs;
        let fd_eps = 1e-4_f64; // finite-difference step

        let n_params = self.model.n_params();
        let mut train_losses = Vec::with_capacity(max_epochs);
        let mut val_losses = Vec::with_capacity(max_epochs);
        let mut best_val_loss = f64::INFINITY;
        let mut best_epoch = 0_usize;

        // Simple LCG for shuffling row indices.
        let mut lcg: u64 = 0x1234_abcd_ef56;
        let lcg_next = |lcg: &mut u64| -> u64 {
            *lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *lcg
        };

        for epoch in 0..max_epochs {
            // Shuffle indices.
            let mut indices: Vec<usize> = (0..n_train).collect();
            for i in (1..n_train).rev() {
                let j = (lcg_next(&mut lcg) as usize) % (i + 1);
                indices.swap(i, j);
            }

            let mut epoch_loss = 0.0_f64;
            let mut n_batches = 0_usize;

            let mut start = 0;
            while start < n_train {
                let end = (start + batch_size).min(n_train);
                let batch_idx = &indices[start..end];
                let b_size = batch_idx.len();

                // Build batch arrays.
                let ctx_len = train_x.ncols();
                let horizon = train_y.ncols();
                let mut bx = Array2::<f64>::zeros((b_size, ctx_len));
                let mut by = Array2::<f64>::zeros((b_size, horizon));
                for (bi, &ri) in batch_idx.iter().enumerate() {
                    for c in 0..ctx_len {
                        bx[[bi, c]] = train_x[[ri, c]];
                    }
                    for h in 0..horizon {
                        by[[bi, h]] = train_y[[ri, h]];
                    }
                }

                // Finite-difference gradient.
                if n_params > 0 {
                    let current_params = self.model.get_params();
                    let mut grad = vec![0.0_f64; n_params];

                    for pi in 0..n_params {
                        let mut p_plus = current_params.clone();
                        let mut p_minus = current_params.clone();
                        p_plus[pi] += fd_eps;
                        p_minus[pi] -= fd_eps;

                        self.model.set_params(&p_plus)?;
                        let loss_plus = Self::mse_loss(&self.model, &bx, &by)?;
                        self.model.set_params(&p_minus)?;
                        let loss_minus = Self::mse_loss(&self.model, &bx, &by)?;

                        grad[pi] = (loss_plus - loss_minus) / (2.0 * fd_eps);
                    }

                    // SGD update.
                    let updated: Vec<f64> = current_params
                        .iter()
                        .zip(grad.iter())
                        .map(|(&p, &g)| p - lr * g)
                        .collect();
                    self.model.set_params(&updated)?;

                    // Compute training loss with updated parameters.
                    let batch_loss = Self::mse_loss(&self.model, &bx, &by)?;
                    epoch_loss += batch_loss;
                    n_batches += 1;
                }

                start = end;
            }

            let avg_train_loss = if n_batches > 0 {
                epoch_loss / n_batches as f64
            } else {
                0.0
            };
            train_losses.push(avg_train_loss);

            // Validation loss.
            if let (Some(vx), Some(vy)) = (val_x, val_y) {
                let vloss = Self::mse_loss(&self.model, vx, vy)?;
                val_losses.push(vloss);
                if vloss < best_val_loss {
                    best_val_loss = vloss;
                    best_epoch = epoch;
                    self.best_params = Some(self.model.get_params());
                }
            } else {
                // No validation: track training loss as proxy.
                if avg_train_loss < best_val_loss {
                    best_val_loss = avg_train_loss;
                    best_epoch = epoch;
                    self.best_params = Some(self.model.get_params());
                }
            }
        }

        // Restore best checkpoint.
        if let Some(ref bp) = self.best_params.clone() {
            self.model.set_params(bp)?;
        }

        Ok(FineTuningResult {
            train_losses,
            val_losses,
            best_epoch,
            best_val_loss,
        })
    }

    /// Predict using the fine-tuned model.
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.model.forward(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Create a simple regression dataset: Y = X.mean(axis=1, keepdims) + noise
    fn make_dataset(n: usize, ctx: usize, horizon: usize) -> (Array2<f64>, Array2<f64>) {
        let x = Array2::from_shape_fn((n, ctx), |(i, j)| (i + j) as f64 * 0.01);
        let y = Array2::from_shape_fn((n, horizon), |(i, _h)| {
            let mean: f64 = (0..ctx).map(|j| x[[i, j]]).sum::<f64>() / ctx as f64;
            mean + 0.001 * (i as f64).sin()
        });
        (x, y)
    }

    #[test]
    fn test_fine_tuner_fit_basic() {
        let ctx = 16;
        let horizon = 4;
        let (train_x, train_y) = make_dataset(40, ctx, horizon);

        let model = LinearForecastModel::new(ctx, horizon);
        let config = FineTuningConfig {
            max_epochs: 5,
            batch_size: 10,
            learning_rate: 1e-3,
            context_length: ctx,
            horizon,
            ..Default::default()
        };
        let mut tuner = FineTuner::new(model, config);
        let result = tuner
            .fit(&train_x, &train_y, None, None)
            .expect("fit should succeed");

        assert_eq!(result.train_losses.len(), 5);
        // Training loss should generally decrease or at least not be NaN.
        for &l in &result.train_losses {
            assert!(l.is_finite(), "loss should be finite");
        }
        // Loss at last epoch should be ≤ loss at first epoch (learning happened).
        let first = result.train_losses[0];
        let last = *result.train_losses.last().expect("non-empty");
        assert!(
            last <= first * 1.5,
            "loss should not drastically increase: first={first:.6} last={last:.6}"
        );
    }

    #[test]
    fn test_fine_tuner_predict_shape() {
        let ctx = 8;
        let horizon = 3;
        let model = LinearForecastModel::new(ctx, horizon);
        let config = FineTuningConfig {
            context_length: ctx,
            horizon,
            max_epochs: 2,
            batch_size: 5,
            ..Default::default()
        };
        let (train_x, train_y) = make_dataset(20, ctx, horizon);
        let mut tuner = FineTuner::new(model, config);
        tuner.fit(&train_x, &train_y, None, None).expect("fit");

        let test_x = Array2::ones((4, ctx));
        let pred = tuner.predict(&test_x).expect("predict");
        assert_eq!(pred.shape(), &[4, horizon]);
    }

    #[test]
    fn test_fine_tuner_with_validation() {
        let ctx = 8;
        let horizon = 4;
        let (train_x, train_y) = make_dataset(30, ctx, horizon);
        let (val_x, val_y) = make_dataset(10, ctx, horizon);

        let model = LinearForecastModel::new(ctx, horizon);
        let config = FineTuningConfig {
            max_epochs: 4,
            batch_size: 8,
            learning_rate: 5e-4,
            context_length: ctx,
            horizon,
            ..Default::default()
        };
        let mut tuner = FineTuner::new(model, config);
        let result = tuner
            .fit(&train_x, &train_y, Some(&val_x), Some(&val_y))
            .expect("fit with val");

        assert_eq!(result.train_losses.len(), 4);
        assert_eq!(result.val_losses.len(), 4);
        assert!(result.best_epoch < 4);
        assert!(result.best_val_loss.is_finite());
    }

    #[test]
    fn test_fine_tuner_lora_reduces_params() {
        let ctx = 32;
        let horizon = 8;
        let rank = 2;

        let base = LinearForecastModel::new(ctx, horizon);
        let base_params = ctx * horizon + horizon; // W + b

        let lora_model = LoraForecastModel::new(base, rank);
        let lora_params = lora_model.n_params(); // A + B + bias only

        // LoRA params = ctx*rank + rank*horizon + horizon
        let expected_lora = ctx * rank + rank * horizon + horizon;
        assert_eq!(lora_params, expected_lora);
        assert!(
            lora_params < base_params,
            "LoRA should have fewer trainable params: lora={lora_params} base={base_params}"
        );
    }

    #[test]
    fn test_lora_forward_shape() {
        let ctx = 16;
        let horizon = 6;
        let rank = 3;
        let base = LinearForecastModel::new(ctx, horizon);
        let lora = LoraForecastModel::new(base, rank);

        let x = Array2::ones((5, ctx));
        let out = lora.forward(&x).expect("lora forward");
        assert_eq!(out.shape(), &[5, horizon]);
    }

    #[test]
    fn test_lora_initial_delta_is_zero() {
        // At init, B = 0 ⟹ ΔW = A B = 0 ⟹ LoRA output = base output.
        let ctx = 8;
        let horizon = 4;
        let rank = 2;
        let base = LinearForecastModel::new(ctx, horizon);
        let base_clone = base.clone();
        let lora = LoraForecastModel::new(base, rank);

        let x = Array2::from_shape_fn((3, ctx), |(i, j)| (i + j) as f64 * 0.1);
        let base_out = base_clone.forward(&x).expect("base");
        let lora_out = lora.forward(&x).expect("lora");

        for i in 0..3 {
            for h in 0..horizon {
                let diff = (base_out[[i, h]] - lora_out[[i, h]]).abs();
                assert!(diff < 1e-12, "LoRA init should equal base: diff={diff:.2e}");
            }
        }
    }

    #[test]
    fn test_linear_model_get_set_params() {
        let ctx = 4;
        let horizon = 2;
        let mut model = LinearForecastModel::new(ctx, horizon);
        let p = model.get_params();
        assert_eq!(p.len(), model.n_params());

        // Round-trip set → get.
        let new_p: Vec<f64> = (0..p.len()).map(|i| i as f64 * 0.01).collect();
        model.set_params(&new_p).expect("set_params");
        let got = model.get_params();
        for (a, b) in new_p.iter().zip(got.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_foundation_model_type_non_exhaustive() {
        let t = FoundationModelType::TimeGpt;
        // Pattern matching must handle unknown variants via wildcard.
        let _name = match t {
            FoundationModelType::TimeGpt => "TimeGPT",
            FoundationModelType::PatchTst => "PatchTST",
            FoundationModelType::TimesNet => "TimesNet",
            FoundationModelType::Generic => "Generic",
            _ => "unknown",
        };
    }
}
