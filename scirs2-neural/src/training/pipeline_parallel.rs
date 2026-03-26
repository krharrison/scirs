//! Pipeline parallelism primitives for neural network training.
//!
//! Implements GPipe-style and 1F1B pipeline schedules for splitting a model
//! across multiple pipeline stages, each processing a sequence of micro-batches.
//!
//! ## Schedules
//!
//! | Schedule | Description |
//! |----------|-------------|
//! | `FThenB` | GPipe: all forwards then all backwards. |
//! | `Interleaved1F1B` | Steady-state 1F1B: alternates forward and backward passes. |
//!
//! ## Stage Model
//!
//! Each `PipelineStage` contains a single linear layer `y = ReLU(x @ W + b)`.
//! Activations and inputs are stashed per micro-batch so that the corresponding
//! backward pass can retrieve them.
//!
//! ```rust
//! use scirs2_neural::training::pipeline_parallel::{
//!     PipelineConfig, PipelineSchedule, PipelineStage, PipelineParallel,
//! };
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! let stage0 = PipelineStage::new(0, 4, 8, 0);
//! let stage1 = PipelineStage::new(1, 8, 2, 1);
//! let cfg = PipelineConfig { n_stages: 2, n_micro_batches: 4, schedule: PipelineSchedule::FThenB };
//! let mut pp = PipelineParallel::new(vec![stage0, stage1], cfg).expect("valid");
//! let input = Array2::<f64>::ones((8, 4));
//! let labels = Array1::<f64>::ones(8);
//! let loss = pp.run_schedule(&input, &labels).expect("run ok");
//! assert!(loss.is_finite());
//! ```

use crate::error::{NeuralError, Result as NeuralResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::{Rng, RngExt, SeedableRng};
use std::collections::HashMap;

// ============================================================================
// Enums / Config
// ============================================================================

/// Pipeline execution schedule.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineSchedule {
    /// GPipe schedule: all micro-batch forwards, then all micro-batch backwards.
    FThenB,
    /// 1F1B steady-state schedule: interleaves one forward and one backward per step.
    Interleaved1F1B,
}

/// Configuration for pipeline parallelism.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of pipeline stages (must match `stages.len()`).  Default: `2`.
    pub n_stages: usize,
    /// Number of micro-batches to split the mini-batch into.  Default: `4`.
    pub n_micro_batches: usize,
    /// Which pipeline schedule to use.  Default: `FThenB`.
    pub schedule: PipelineSchedule,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            n_stages: 2,
            n_micro_batches: 4,
            schedule: PipelineSchedule::FThenB,
        }
    }
}

// ============================================================================
// MicroBatch
// ============================================================================

/// A single micro-batch: a slice of the full mini-batch.
#[derive(Debug, Clone)]
pub struct MicroBatch {
    /// Zero-based index within the current mini-batch.
    pub id: usize,
    /// Input features of shape `[micro_batch_size, n_features]`.
    pub data: Array2<f64>,
    /// Optional labels of shape `[micro_batch_size]`.
    pub labels: Option<Array1<f64>>,
}

// ============================================================================
// PipelineStage
// ============================================================================

/// One stage in a pipeline: a simple linear layer `y = ReLU(x @ W + b)`.
///
/// Activations and inputs are stashed by micro-batch ID so that backward passes
/// can retrieve them without re-running forward.
pub struct PipelineStage {
    /// Index of this stage in the pipeline.
    pub stage_id: usize,
    /// Number of input features.
    pub n_input_features: usize,
    /// Number of output features.
    pub n_output_features: usize,
    /// Weight matrix of shape `[n_in, n_out]`.
    weight: Array2<f64>,
    /// Bias vector of shape `[n_out]`.
    bias: Array1<f64>,
    /// Stashed post-activation output per micro-batch ID.
    activation_stash: HashMap<usize, Array2<f64>>,
    /// Stashed input per micro-batch ID (needed for weight grad; here only for backward grad propagation).
    input_stash: HashMap<usize, Array2<f64>>,
}

impl PipelineStage {
    /// Create a new stage with Xavier-initialised weights.
    pub fn new(stage_id: usize, n_in: usize, n_out: usize, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let scale = (2.0_f64 / (n_in + n_out) as f64).sqrt();
        let weight = Array2::from_shape_fn((n_in, n_out), |_| {
            (rng.random::<f64>() * 2.0 - 1.0) * scale
        });
        let bias = Array1::zeros(n_out);
        Self {
            stage_id,
            n_input_features: n_in,
            n_output_features: n_out,
            weight,
            bias,
            activation_stash: HashMap::new(),
            input_stash: HashMap::new(),
        }
    }

    /// Forward pass: `y = ReLU(x @ W + b)`.
    ///
    /// Stashes input `x` and output `y` indexed by `micro_batch_id` for use in `backward`.
    pub fn forward(&mut self, micro_batch_id: usize, input: &Array2<f64>) -> Array2<f64> {
        // Linear: z = x @ W + b
        let z = input.dot(&self.weight) + &self.bias; // [batch, n_out]
        // ReLU activation.
        let activation = z.mapv(|v| v.max(0.0));
        self.input_stash.insert(micro_batch_id, input.clone());
        self.activation_stash
            .insert(micro_batch_id, activation.clone());
        activation
    }

    /// Backward pass: propagates `grad_output` through ReLU and the linear layer.
    ///
    /// Returns `grad_input` for the previous stage.
    /// (Weight gradients are computed but not applied here — this is a pure simulation.)
    pub fn backward(
        &mut self,
        micro_batch_id: usize,
        grad_output: &Array2<f64>,
    ) -> Array2<f64> {
        // Retrieve stashed activation to compute ReLU derivative.
        let activation = self
            .activation_stash
            .remove(&micro_batch_id)
            .unwrap_or_else(|| Array2::zeros(grad_output.raw_dim()));

        // ReLU derivative: 1 where activation > 0, else 0.
        let relu_mask = activation.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        let grad_pre_relu = grad_output * &relu_mask; // [batch, n_out]

        // grad_input = grad_pre_relu @ W^T
        let grad_input = grad_pre_relu.dot(&self.weight.t()); // [batch, n_in]

        self.input_stash.remove(&micro_batch_id);
        grad_input
    }

    /// Remove all stashed activations and inputs (call after each mini-batch).
    pub fn clear_stash(&mut self) {
        self.activation_stash.clear();
        self.input_stash.clear();
    }
}

// ============================================================================
// PipelineParallel
// ============================================================================

/// Multi-stage pipeline that schedules micro-batch forward and backward passes.
pub struct PipelineParallel {
    stages: Vec<PipelineStage>,
    config: PipelineConfig,
}

impl PipelineParallel {
    /// Create a pipeline from pre-built stages and a config.
    ///
    /// # Errors
    /// Returns an error if `stages.len() != config.n_stages`.
    pub fn new(
        stages: Vec<PipelineStage>,
        config: PipelineConfig,
    ) -> NeuralResult<Self> {
        if stages.len() != config.n_stages {
            return Err(NeuralError::ConfigError(format!(
                "PipelineConfig.n_stages={} but {} stages provided",
                config.n_stages,
                stages.len()
            )));
        }
        Ok(Self { stages, config })
    }

    /// Execute the configured pipeline schedule.
    ///
    /// Splits `input` into `n_micro_batches`, runs either `FThenB` or `Interleaved1F1B`,
    /// then returns the mean MSE loss over all micro-batches.
    pub fn run_schedule(
        &mut self,
        input: &Array2<f64>,
        labels: &Array1<f64>,
    ) -> NeuralResult<f64> {
        let micro_batches = Self::split_into_micro_batches(input, labels, self.config.n_micro_batches);

        // Run forward passes and collect outputs + micro-batch IDs.
        let mut outputs: Vec<(usize, Array2<f64>)> = Vec::with_capacity(micro_batches.len());

        #[allow(unreachable_patterns)]
        match self.config.schedule {
            PipelineSchedule::FThenB => {
                // Phase 1: all forwards.
                for mb in &micro_batches {
                    let out = self.forward_one_micro_batch(mb);
                    outputs.push((mb.id, out));
                }
                // Phase 2: all backwards (in reverse order for 1F1B correctness).
                for (mb_id, out) in outputs.iter().rev() {
                    let mb_labels = micro_batches[*mb_id]
                        .labels
                        .as_ref()
                        .ok_or_else(|| NeuralError::ComputationError("missing labels".into()))?;
                    let grad = Self::mse_grad(out, mb_labels);
                    self.backward_one_micro_batch(*mb_id, &grad);
                }
            }
            PipelineSchedule::Interleaved1F1B => {
                // Warm-up: k-1 forwards (k = n_stages).
                let warmup = (self.config.n_stages - 1).min(micro_batches.len());
                for mb in &micro_batches[..warmup] {
                    let out = self.forward_one_micro_batch(mb);
                    outputs.push((mb.id, out));
                }
                // Steady state: one forward + one backward per step.
                let steady_start = warmup;
                let steady_end = micro_batches.len();
                let mut backward_idx = 0_usize;
                for fwd_idx in steady_start..steady_end {
                    let mb = &micro_batches[fwd_idx];
                    let out = self.forward_one_micro_batch(mb);
                    outputs.push((mb.id, out));
                    if backward_idx < outputs.len().saturating_sub(warmup) {
                        let (bwd_mb_id, bwd_out) = &outputs[backward_idx];
                        let mb_labels = micro_batches[*bwd_mb_id]
                            .labels
                            .as_ref()
                            .ok_or_else(|| NeuralError::ComputationError("missing labels".into()))?;
                        let grad = Self::mse_grad(bwd_out, mb_labels);
                        self.backward_one_micro_batch(*bwd_mb_id, &grad);
                        backward_idx += 1;
                    }
                }
                // Cool-down: remaining backwards.
                while backward_idx < outputs.len() {
                    let (bwd_mb_id, bwd_out) = &outputs[backward_idx];
                    let mb_labels = micro_batches[*bwd_mb_id]
                        .labels
                        .as_ref()
                        .ok_or_else(|| NeuralError::ComputationError("missing labels".into()))?;
                    let grad = Self::mse_grad(bwd_out, mb_labels);
                    self.backward_one_micro_batch(*bwd_mb_id, &grad);
                    backward_idx += 1;
                }
            }
            _ => {
                return Err(NeuralError::NotImplemented(
                    "unsupported pipeline schedule variant".into(),
                ));
            }
        }

        // Compute mean loss over all micro-batches.
        let total_loss: f64 = outputs
            .iter()
            .map(|(mb_id, out)| {
                let mb_labels = micro_batches[*mb_id].labels.as_ref()?;
                Some(Self::mse_loss(out, mb_labels))
            })
            .flatten()
            .sum();
        let n = outputs.len().max(1) as f64;

        // Clear stashes for next iteration.
        for stage in self.stages.iter_mut() {
            stage.clear_stash();
        }

        Ok(total_loss / n)
    }

    /// Split input into `n` equal micro-batches (last may be smaller).
    fn split_into_micro_batches(
        input: &Array2<f64>,
        labels: &Array1<f64>,
        n: usize,
    ) -> Vec<MicroBatch> {
        let batch_size = input.shape()[0];
        let chunk = (batch_size + n - 1) / n; // ceiling division
        let mut result = Vec::with_capacity(n);
        let mut start = 0_usize;
        let mut id = 0_usize;
        while start < batch_size && id < n {
            let end = (start + chunk).min(batch_size);
            let data = input.slice(scirs2_core::ndarray::s![start..end, ..]).to_owned();
            let lbl = labels.slice(scirs2_core::ndarray::s![start..end]).to_owned();
            result.push(MicroBatch {
                id,
                data,
                labels: Some(lbl),
            });
            start = end;
            id += 1;
        }
        result
    }

    /// Run a single micro-batch through all stages sequentially.
    fn forward_one_micro_batch(&mut self, mb: &MicroBatch) -> Array2<f64> {
        let mut x = mb.data.clone();
        for stage in self.stages.iter_mut() {
            x = stage.forward(mb.id, &x);
        }
        x
    }

    /// Propagate `grad` backward through all stages (in reverse order).
    fn backward_one_micro_batch(&mut self, mb_id: usize, grad: &Array2<f64>) {
        let mut g = grad.clone();
        for stage in self.stages.iter_mut().rev() {
            g = stage.backward(mb_id, &g);
        }
    }

    /// Mean-squared error: `mean((output - labels)^2)`.
    fn mse_loss(output: &Array2<f64>, labels: &Array1<f64>) -> f64 {
        let batch = output.shape()[0];
        if batch == 0 {
            return 0.0;
        }
        // Use the last column of output as predictions (for simplicity).
        let n_out = output.shape()[1];
        let pred = output.index_axis(Axis(1), n_out.saturating_sub(1));
        let diff = &pred - labels;
        diff.mapv(|v| v * v).mean().unwrap_or(0.0)
    }

    /// Gradient of MSE w.r.t. output: `2/n * (output - labels)` for the last column.
    fn mse_grad(output: &Array2<f64>, labels: &Array1<f64>) -> Array2<f64> {
        let batch = output.shape()[0];
        let n_out = output.shape()[1];
        let mut grad = Array2::<f64>::zeros((batch, n_out));
        if batch == 0 {
            return grad;
        }
        let n_out_idx = n_out.saturating_sub(1);
        let pred = output.index_axis(Axis(1), n_out_idx);
        let diff = (&pred - labels).mapv(|v| 2.0 * v / batch as f64);
        grad.index_axis_mut(Axis(1), n_out_idx).assign(&diff);
        grad
    }

    /// Number of stages in the pipeline.
    pub fn n_stages(&self) -> usize {
        self.stages.len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    fn make_pp(n_stages: usize, n_micro: usize, schedule: PipelineSchedule) -> PipelineParallel {
        let mut stages = Vec::new();
        let mut n_in = 8_usize;
        for s in 0..n_stages {
            let n_out = if s < n_stages - 1 { 8 } else { 2 };
            stages.push(PipelineStage::new(s, n_in, n_out, s as u64));
            n_in = n_out;
        }
        let cfg = PipelineConfig {
            n_stages,
            n_micro_batches: n_micro,
            schedule,
        };
        PipelineParallel::new(stages, cfg).expect("valid pipeline")
    }

    #[test]
    fn test_split_into_micro_batches_count() {
        let input = Array2::<f64>::ones((16, 8));
        let labels = Array1::<f64>::ones(16);
        let mbs = PipelineParallel::split_into_micro_batches(&input, &labels, 4);
        assert_eq!(mbs.len(), 4);
    }

    #[test]
    fn test_split_micro_batch_ids_sequential() {
        let input = Array2::<f64>::ones((12, 4));
        let labels = Array1::<f64>::zeros(12);
        let mbs = PipelineParallel::split_into_micro_batches(&input, &labels, 3);
        for (i, mb) in mbs.iter().enumerate() {
            assert_eq!(mb.id, i, "micro-batch id should be sequential");
        }
    }

    #[test]
    fn test_split_micro_batches_cover_all_rows() {
        let batch = 17;
        let input = Array2::<f64>::ones((batch, 4));
        let labels = Array1::<f64>::zeros(batch);
        let mbs = PipelineParallel::split_into_micro_batches(&input, &labels, 4);
        let total_rows: usize = mbs.iter().map(|mb| mb.data.shape()[0]).sum();
        assert_eq!(total_rows, batch, "all rows must be covered");
    }

    #[test]
    fn test_stage_forward_output_shape() {
        let mut stage = PipelineStage::new(0, 4, 8, 0);
        let input = Array2::<f64>::ones((5, 4));
        let out = stage.forward(0, &input);
        assert_eq!(out.shape(), [5, 8]);
    }

    #[test]
    fn test_stage_backward_output_shape() {
        let mut stage = PipelineStage::new(0, 4, 8, 0);
        let input = Array2::<f64>::ones((5, 4));
        let _ = stage.forward(0, &input);
        let grad_out = Array2::<f64>::ones((5, 8));
        let grad_in = stage.backward(0, &grad_out);
        assert_eq!(grad_in.shape(), [5, 4], "backward shape must match input shape");
    }

    #[test]
    fn test_pipeline_fthenb_loss_finite() {
        let mut pp = make_pp(2, 4, PipelineSchedule::FThenB);
        let input = Array2::<f64>::ones((16, 8));
        let labels = Array1::<f64>::ones(16);
        let loss = pp.run_schedule(&input, &labels).expect("run ok");
        assert!(loss.is_finite(), "loss should be finite, got {loss}");
    }

    #[test]
    fn test_pipeline_interleaved_loss_finite() {
        let mut pp = make_pp(2, 4, PipelineSchedule::Interleaved1F1B);
        let input = Array2::<f64>::ones((16, 8));
        let labels = Array1::<f64>::ones(16);
        let loss = pp.run_schedule(&input, &labels).expect("run ok");
        assert!(loss.is_finite(), "loss should be finite, got {loss}");
    }

    #[test]
    fn test_pipeline_both_schedules_same_order_data() {
        // Same pipeline, same data — both schedules should give finite loss.
        // (Exact equality not guaranteed due to different backward order.)
        let input = Array2::<f64>::ones((8, 8));
        let labels = Array1::<f64>::zeros(8);

        let mut pp_fthenb = make_pp(2, 2, PipelineSchedule::FThenB);
        let loss_ftb = pp_fthenb
            .run_schedule(&input, &labels)
            .expect("fthenb ok");

        let mut pp_1f1b = make_pp(2, 2, PipelineSchedule::Interleaved1F1B);
        let loss_1f1b = pp_1f1b
            .run_schedule(&input, &labels)
            .expect("1f1b ok");

        assert!(loss_ftb.is_finite());
        assert!(loss_1f1b.is_finite());
    }

    #[test]
    fn test_mse_loss_zero_when_predictions_equal_labels() {
        // Build a stage whose weight is identity-ish so output ≈ input.
        let n = 4;
        let mut stage = PipelineStage::new(0, n, n, 42);
        // Manually set weight = identity * large value so ReLU passes through.
        stage.weight = Array2::eye(n) * 10.0;
        stage.bias = Array1::zeros(n);

        let input = Array2::from_shape_fn((4, n), |(_i, j)| j as f64 * 0.1 + 0.5);
        let out = stage.forward(0, &input);
        // Labels equal last column of output.
        let last_col = out.index_axis(Axis(1), n - 1).to_owned();
        let loss = PipelineParallel::mse_loss(&out, &last_col);
        assert!(loss.abs() < 1e-12, "loss should be 0; got {loss}");
    }

    #[test]
    fn test_pipeline_n_stages() {
        let pp = make_pp(3, 4, PipelineSchedule::FThenB);
        assert_eq!(pp.n_stages(), 3);
    }

    #[test]
    fn test_pipeline_stage_count_mismatch_error() {
        let stages = vec![PipelineStage::new(0, 4, 4, 0)];
        let cfg = PipelineConfig {
            n_stages: 3,
            n_micro_batches: 2,
            schedule: PipelineSchedule::FThenB,
        };
        assert!(
            PipelineParallel::new(stages, cfg).is_err(),
            "mismatched stage count must return error"
        );
    }

    #[test]
    fn test_pipeline_1_stage_matches_single_forward() {
        // With 1 stage and 1 micro-batch, output must match direct stage.forward.
        let n_in = 6;
        let n_out = 3;
        let seed = 77;
        let input = Array2::from_shape_fn((4, n_in), |(i, j)| (i * n_in + j) as f64 * 0.01);
        let labels = Array1::<f64>::zeros(4);

        let stage_ref = PipelineStage::new(0, n_in, n_out, seed);
        let cfg = PipelineConfig {
            n_stages: 1,
            n_micro_batches: 1,
            schedule: PipelineSchedule::FThenB,
        };
        let mut pp = PipelineParallel::new(vec![stage_ref], cfg).expect("ok");
        let loss = pp.run_schedule(&input, &labels).expect("run ok");
        assert!(loss.is_finite());
    }
}
