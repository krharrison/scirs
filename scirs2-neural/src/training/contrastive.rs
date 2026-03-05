//! Contrastive Learning Primitives
//!
//! Implements modern contrastive and metric-learning loss functions together with
//! the supporting trainer scaffolding used by frameworks such as SimCLR, MoCo,
//! BYOL, and SupCon.
//!
//! # Algorithms
//!
//! - **NTXentLoss**: Normalized temperature-scaled cross-entropy (SimCLR loss)
//! - **SimCLRTrainer**: Projection head + augmentation interface + batch loss
//! - **MoCoQueue**: Momentum encoder with FIFO queue of negative keys
//! - **BYOLConfig** / **BYOLUpdate**: Online/target EMA update rule
//! - **SupConLoss**: Supervised contrastive loss with class label support
//! - **TripletMarginLoss**: Online hard/semi-hard triplet mining
//! - **ContrastivePairLoss**: Margin-based pairwise contrastive loss (re-export style)
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::contrastive::{NTXentLoss, SimCLRTrainer, SimCLRConfig};
//! use scirs2_core::ndarray::{Array2};
//!
//! let loss_fn = NTXentLoss::new(0.07);
//!
//! // Two augmented views of batch of 4 samples, each with dim 8
//! let z_i = Array2::<f64>::zeros((4, 8));
//! let z_j = Array2::<f64>::zeros((4, 8));
//! let loss = loss_fn.forward(&z_i, &z_j).expect("NT-Xent forward ok");
//! assert!(loss.is_finite());
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use std::collections::VecDeque;
use std::fmt::Debug;

// ============================================================================
// NTXentLoss (SimCLR)
// ============================================================================

/// Normalized temperature-scaled cross-entropy loss (NT-Xent).
///
/// Given two sets of projected embeddings `z_i` and `z_j` (each of shape
/// `[N, D]`, already L2-normalised or not – this function normalises
/// internally), the loss is:
///
/// ```text
/// L = -1/N Σ_i log [ exp(sim(z_i, z_j_i)/τ) /
///                    Σ_{k≠i} exp(sim(z_i, z_k)/τ) ]
/// ```
///
/// where the denominator sums over the `2N-2` negatives (both views, excluding
/// the positive pair).
///
/// Reference: Chen et al., "A Simple Framework for Contrastive Learning of
/// Visual Representations", ICML 2020.
#[derive(Debug, Clone)]
pub struct NTXentLoss {
    /// Temperature scaling factor τ (typical value: 0.07 – 0.5).
    pub temperature: f64,
}

impl NTXentLoss {
    /// Create a new NT-Xent loss with the given temperature.
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    /// Compute the NT-Xent loss.
    ///
    /// # Arguments
    /// * `z_i` - Projected embeddings from view i, shape `[N, D]`
    /// * `z_j` - Projected embeddings from view j, shape `[N, D]`
    ///
    /// # Returns
    /// Scalar loss value averaged over the batch.
    pub fn forward<F>(&self, z_i: &Array2<F>, z_j: &Array2<F>) -> Result<F>
    where
        F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
    {
        let n = z_i.nrows();
        if z_j.nrows() != n {
            return Err(NeuralError::ShapeMismatch(format!(
                "NT-Xent: z_i has {} rows but z_j has {}",
                n,
                z_j.nrows()
            )));
        }
        if n == 0 {
            return Err(NeuralError::InvalidArgument(
                "NT-Xent: batch size must be > 0".to_string(),
            ));
        }

        let tau = F::from_f64(self.temperature).ok_or_else(|| {
            NeuralError::ComputationError("NT-Xent: cannot convert temperature".to_string())
        })?;

        // L2-normalise both views
        let zi_norm = l2_normalise(z_i)?;
        let zj_norm = l2_normalise(z_j)?;

        // Concatenate: [z_i; z_j] → shape [2N, D]
        let z_all = concatenate_rows(&zi_norm, &zj_norm)?;
        let two_n = z_all.nrows(); // 2N

        // Cosine similarity matrix [2N, 2N]
        let sim = cosine_sim_matrix(&z_all)?;

        let mut total_loss = F::zero();
        let neg_inf = F::from_f64(-1e38).ok_or_else(|| {
            NeuralError::ComputationError("NT-Xent: cannot convert neg_inf".to_string())
        })?;

        for i in 0..two_n {
            // Positive index: if i < N then positive is i+N, else i-N
            let pos_idx = if i < n { i + n } else { i - n };

            // Numerator: sim(i, pos) / tau
            let num_val = sim[[i, pos_idx]] / tau;

            // Denominator: sum over all k ≠ i of exp(sim(i,k)/tau)
            let mut log_denom = neg_inf;
            for k in 0..two_n {
                if k == i {
                    continue;
                }
                let logit = sim[[i, k]] / tau;
                log_denom = log_sum_exp_pair(log_denom, logit);
            }

            total_loss += num_val - log_denom;
        }

        let two_n_f = F::from_usize(two_n).ok_or_else(|| {
            NeuralError::ComputationError("NT-Xent: cannot convert 2N".to_string())
        })?;
        let loss = -(total_loss / two_n_f);
        Ok(loss)
    }
}

impl Default for NTXentLoss {
    fn default() -> Self {
        Self::new(0.1)
    }
}

// ============================================================================
// SimCLR
// ============================================================================

/// Configuration for the SimCLR self-supervised trainer.
#[derive(Debug, Clone)]
pub struct SimCLRConfig {
    /// Dimensionality of the representation (output of backbone encoder).
    pub representation_dim: usize,
    /// Hidden dimension of the MLP projection head.
    pub projection_hidden_dim: usize,
    /// Output dimension of the projection head.
    pub projection_output_dim: usize,
    /// Temperature for NT-Xent loss.
    pub temperature: f64,
    /// L2 weight decay applied to projection head parameters.
    pub weight_decay: f64,
}

impl Default for SimCLRConfig {
    fn default() -> Self {
        Self {
            representation_dim: 512,
            projection_hidden_dim: 2048,
            projection_output_dim: 128,
            temperature: 0.07,
            weight_decay: 1e-6,
        }
    }
}

impl SimCLRConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.representation_dim == 0 {
            return Err(NeuralError::ConfigError(
                "SimCLRConfig: representation_dim must be > 0".to_string(),
            ));
        }
        if self.projection_hidden_dim == 0 {
            return Err(NeuralError::ConfigError(
                "SimCLRConfig: projection_hidden_dim must be > 0".to_string(),
            ));
        }
        if self.projection_output_dim == 0 {
            return Err(NeuralError::ConfigError(
                "SimCLRConfig: projection_output_dim must be > 0".to_string(),
            ));
        }
        if self.temperature <= 0.0 {
            return Err(NeuralError::ConfigError(
                "SimCLRConfig: temperature must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// A lightweight MLP projection head used by SimCLR.
///
/// Architecture: Linear → ReLU → Linear
/// Weights are stored as `Array2<F>` tensors.
#[derive(Debug, Clone)]
pub struct ProjectionHead<F: Float + Debug + NumAssign> {
    /// Weight matrix W1: shape [hidden, in_dim]
    pub w1: Array2<F>,
    /// Bias b1: shape [hidden]
    pub b1: Array1<F>,
    /// Weight matrix W2: shape [out_dim, hidden]
    pub w2: Array2<F>,
    /// Bias b2: shape [out_dim]
    pub b2: Array1<F>,
}

impl<F: Float + Debug + NumAssign + FromPrimitive> ProjectionHead<F> {
    /// Create a new randomly-initialised projection head.
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize) -> Result<Self> {
        // Xavier initialisation scale
        let scale1 = F::from_f64((2.0 / (in_dim + hidden_dim) as f64).sqrt()).ok_or_else(|| {
            NeuralError::ComputationError("ProjectionHead: cannot compute scale1".to_string())
        })?;
        let scale2 =
            F::from_f64((2.0 / (hidden_dim + out_dim) as f64).sqrt()).ok_or_else(|| {
                NeuralError::ComputationError("ProjectionHead: cannot compute scale2".to_string())
            })?;

        // Simple deterministic init (alternating ±scale) for reproducibility
        // without external RNG dependency.
        let w1 = init_weight_matrix(hidden_dim, in_dim, scale1);
        let b1 = Array1::zeros(hidden_dim);
        let w2 = init_weight_matrix(out_dim, hidden_dim, scale2);
        let b2 = Array1::zeros(out_dim);

        Ok(Self { w1, b1, w2, b2 })
    }

    /// Forward pass through the projection head.
    ///
    /// Input shape: `[N, in_dim]`
    /// Output shape: `[N, out_dim]`
    pub fn forward(&self, x: &Array2<F>) -> Result<Array2<F>> {
        // Layer 1: h = relu(x @ W1.T + b1)
        let h = linear_forward(x, &self.w1, &self.b1)?;
        let h = relu2d(&h);
        // Layer 2: out = h @ W2.T + b2
        let out = linear_forward(&h, &self.w2, &self.b2)?;
        Ok(out)
    }
}

/// SimCLR trainer that wraps the projection head and NT-Xent loss computation.
#[derive(Debug, Clone)]
pub struct SimCLRTrainer<F: Float + Debug + NumAssign + FromPrimitive> {
    /// Configuration.
    pub config: SimCLRConfig,
    /// Projection head parameters.
    pub projection_head: ProjectionHead<F>,
    /// NT-Xent loss function.
    pub loss_fn: NTXentLoss,
}

impl<F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive> SimCLRTrainer<F> {
    /// Create a new SimCLR trainer from a validated config.
    pub fn new(config: SimCLRConfig) -> Result<Self> {
        config.validate()?;
        let projection_head = ProjectionHead::new(
            config.representation_dim,
            config.projection_hidden_dim,
            config.projection_output_dim,
        )?;
        let loss_fn = NTXentLoss::new(config.temperature);
        Ok(Self {
            config,
            projection_head,
            loss_fn,
        })
    }

    /// Compute the SimCLR loss for two augmented views of a batch.
    ///
    /// # Arguments
    /// * `rep_i` - Representations from view i, shape `[N, repr_dim]`
    /// * `rep_j` - Representations from view j, shape `[N, repr_dim]`
    ///
    /// # Returns
    /// Scalar NT-Xent loss.
    pub fn batch_loss(&self, rep_i: &Array2<F>, rep_j: &Array2<F>) -> Result<F> {
        let z_i = self.projection_head.forward(rep_i)?;
        let z_j = self.projection_head.forward(rep_j)?;
        self.loss_fn.forward(&z_i, &z_j)
    }
}

// ============================================================================
// MoCo queue
// ============================================================================

/// Momentum Contrastive Learning (MoCo) queue of negative keys.
///
/// Maintains a FIFO queue of encoded keys from past mini-batches that serve as
/// negative examples during contrastive training. The momentum encoder whose
/// weights are an exponential moving average (EMA) of the online encoder's
/// weights produces these keys.
///
/// Reference: He et al., "Momentum Contrast for Unsupervised Visual
/// Representation Learning", CVPR 2020.
#[derive(Debug, Clone)]
pub struct MoCoQueue<F: Float + Debug + NumAssign> {
    /// Maximum queue capacity (number of negative keys).
    pub capacity: usize,
    /// Dimensionality of each key vector.
    pub key_dim: usize,
    /// Momentum coefficient for EMA update of target encoder (default 0.999).
    pub momentum: f64,
    /// Temperature for InfoNCE loss (default 0.07).
    pub temperature: f64,
    /// Internal FIFO storage of key vectors.
    queue: VecDeque<Array1<F>>,
}

impl<F: Float + Debug + NumAssign + FromPrimitive> MoCoQueue<F> {
    /// Create a new MoCo queue.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of negative keys stored in the queue.
    /// * `key_dim`  - Dimensionality of each key (should match encoder output).
    /// * `momentum` - EMA coefficient for target encoder (typical: 0.999).
    /// * `temperature` - Temperature for InfoNCE loss (typical: 0.07).
    pub fn new(capacity: usize, key_dim: usize, momentum: f64, temperature: f64) -> Self {
        Self {
            capacity,
            key_dim,
            momentum,
            temperature,
            queue: VecDeque::with_capacity(capacity),
        }
    }

    /// Enqueue a batch of new keys and dequeue old ones if at capacity.
    ///
    /// Keys are L2-normalised before being stored.
    pub fn enqueue_and_dequeue(&mut self, keys: &Array2<F>) -> Result<()> {
        let n = keys.nrows();
        if keys.ncols() != self.key_dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "MoCoQueue::enqueue: keys have {} dims but queue expects {}",
                keys.ncols(),
                self.key_dim
            )));
        }
        let keys_norm = l2_normalise(keys)?;
        for i in 0..n {
            let key = keys_norm.row(i).to_owned();
            if self.queue.len() == self.capacity {
                self.queue.pop_front();
            }
            self.queue.push_back(key);
        }
        Ok(())
    }

    /// Return the current number of keys in the queue.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Return `true` if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Compute InfoNCE contrastive loss between query embeddings and the queue.
    ///
    /// # Arguments
    /// * `queries` - L2-normalised query embeddings, shape `[N, key_dim]`
    /// * `keys`    - Corresponding positive key embeddings, shape `[N, key_dim]`
    ///
    /// # Returns
    /// Scalar InfoNCE loss averaged over the batch.
    pub fn info_nce_loss(&self, queries: &Array2<F>, keys: &Array2<F>) -> Result<F> {
        let n = queries.nrows();
        if keys.nrows() != n {
            return Err(NeuralError::ShapeMismatch(
                "MoCoQueue::info_nce_loss: queries and keys must have same batch size".to_string(),
            ));
        }
        if self.queue.is_empty() {
            return Err(NeuralError::InvalidState(
                "MoCoQueue::info_nce_loss: queue is empty; enqueue keys first".to_string(),
            ));
        }

        let tau = F::from_f64(self.temperature).ok_or_else(|| {
            NeuralError::ComputationError(
                "MoCoQueue::info_nce_loss: cannot convert temperature".to_string(),
            )
        })?;

        let q_norm = l2_normalise(queries)?;
        let k_norm = l2_normalise(keys)?;

        let mut total_loss = F::zero();
        let neg_inf = F::from_f64(-1e38).ok_or_else(|| {
            NeuralError::ComputationError("MoCoQueue: cannot convert neg_inf".to_string())
        })?;

        for i in 0..n {
            let q = q_norm.row(i);
            // Positive logit: q · k_pos / tau
            let k_pos = k_norm.row(i);
            let pos_sim: F = q.iter().zip(k_pos.iter()).map(|(a, b)| *a * *b).fold(F::zero(), |acc, x| acc + x);
            let pos_logit = pos_sim / tau;

            // Negative logits from queue
            let mut log_denom = pos_logit; // include positive in denominator
            for neg_key in &self.queue {
                let neg_sim: F = q.iter().zip(neg_key.iter()).map(|(a, b)| *a * *b).fold(F::zero(), |acc, x| acc + x);
                let neg_logit = neg_sim / tau;
                log_denom = log_sum_exp_pair(log_denom, neg_logit);
            }

            total_loss += pos_logit - log_denom;
        }

        let n_f = F::from_usize(n).ok_or_else(|| {
            NeuralError::ComputationError("MoCoQueue: cannot convert N".to_string())
        })?;
        Ok(-(total_loss / n_f))
    }

    /// Apply EMA update: `target ← momentum * target + (1 - momentum) * online`.
    ///
    /// Both parameter slices must have identical length.
    pub fn ema_update<P>(&self, online_params: &[P], target_params: &mut [P]) -> Result<()>
    where
        P: Clone
            + std::ops::Mul<f64, Output = P>
            + std::ops::Add<Output = P>,
    {
        if online_params.len() != target_params.len() {
            return Err(NeuralError::ShapeMismatch(format!(
                "MoCoQueue::ema_update: online has {} params but target has {}",
                online_params.len(),
                target_params.len()
            )));
        }
        let m = self.momentum;
        let one_minus_m = 1.0 - m;
        for (t, o) in target_params.iter_mut().zip(online_params.iter()) {
            *t = t.clone() * m + o.clone() * one_minus_m;
        }
        Ok(())
    }
}

// ============================================================================
// BYOL
// ============================================================================

/// Configuration for BYOL (Bootstrap Your Own Latent) self-supervised learning.
///
/// Reference: Grill et al., "Bootstrap Your Own Latent - A New Approach to
/// Self-Supervised Learning", NeurIPS 2020.
#[derive(Debug, Clone)]
pub struct BYOLConfig {
    /// Dimensionality of the representation vector.
    pub representation_dim: usize,
    /// Hidden dim of both projection and prediction heads.
    pub hidden_dim: usize,
    /// Output dim of projection head.
    pub projection_dim: usize,
    /// Initial EMA coefficient for target network update.
    pub initial_tau: f64,
    /// Final EMA coefficient (linearly annealed over training).
    pub final_tau: f64,
    /// Total number of training steps (used for tau annealing).
    pub total_steps: usize,
}

impl Default for BYOLConfig {
    fn default() -> Self {
        Self {
            representation_dim: 512,
            hidden_dim: 4096,
            projection_dim: 256,
            initial_tau: 0.996,
            final_tau: 1.0,
            total_steps: 100_000,
        }
    }
}

impl BYOLConfig {
    /// Validate configuration fields.
    pub fn validate(&self) -> Result<()> {
        if self.representation_dim == 0 {
            return Err(NeuralError::ConfigError(
                "BYOLConfig: representation_dim must be > 0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.initial_tau) || !(0.0..=1.0).contains(&self.final_tau) {
            return Err(NeuralError::ConfigError(
                "BYOLConfig: tau values must be in [0, 1]".to_string(),
            ));
        }
        if self.total_steps == 0 {
            return Err(NeuralError::ConfigError(
                "BYOLConfig: total_steps must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Compute the EMA coefficient at a given training step.
    pub fn tau_at_step(&self, step: usize) -> f64 {
        let t = (step as f64) / (self.total_steps as f64);
        let t = t.clamp(0.0, 1.0);
        // cosine annealing from initial_tau to final_tau
        let cos_val = (std::f64::consts::PI * t).cos();
        self.initial_tau + (self.final_tau - self.initial_tau) * (1.0 - cos_val) / 2.0
    }
}

/// BYOL update result containing the computed loss.
#[derive(Debug, Clone)]
pub struct BYOLUpdate<F: Float + Debug> {
    /// Mean squared error loss between online prediction and target projection.
    pub loss: F,
    /// Current EMA coefficient used in this update.
    pub tau: f64,
    /// Global training step at which this update was computed.
    pub step: usize,
}

impl<F: Float + Debug + NumAssign + FromPrimitive> BYOLUpdate<F> {
    /// Compute a single BYOL update step.
    ///
    /// # Arguments
    /// * `online_pred`  - Online predictor output, shape `[N, projection_dim]`
    /// * `target_proj`  - Target projector output (stop-gradient), shape `[N, projection_dim]`
    /// * `config`       - BYOL configuration used to compute tau.
    /// * `step`         - Current global training step.
    ///
    /// # Returns
    /// `BYOLUpdate` with the symmetric MSE loss and the tau used.
    pub fn compute(
        online_pred: &Array2<F>,
        target_proj: &Array2<F>,
        config: &BYOLConfig,
        step: usize,
    ) -> Result<Self> {
        if online_pred.shape() != target_proj.shape() {
            return Err(NeuralError::ShapeMismatch(format!(
                "BYOLUpdate: online_pred shape {:?} != target_proj shape {:?}",
                online_pred.shape(),
                target_proj.shape()
            )));
        }
        // L2-normalise both
        let p = l2_normalise(online_pred)?;
        let z = l2_normalise(target_proj)?;

        // Symmetric cosine similarity loss: L = 2 - 2 * mean(p · z)
        let n = p.nrows();
        let mut dot_sum = F::zero();
        for i in 0..n {
            let pi = p.row(i);
            let zi = z.row(i);
            let dot: F = pi.iter().zip(zi.iter()).map(|(a, b)| *a * *b).fold(F::zero(), |acc, x| acc + x);
            dot_sum += dot;
        }
        let n_f = F::from_usize(n).ok_or_else(|| {
            NeuralError::ComputationError("BYOLUpdate: cannot convert N".to_string())
        })?;
        let two = F::from_f64(2.0).ok_or_else(|| {
            NeuralError::ComputationError("BYOLUpdate: cannot convert 2.0".to_string())
        })?;
        let mean_dot = dot_sum / n_f;
        let loss = two - two * mean_dot;

        let tau = config.tau_at_step(step);
        Ok(Self { loss, tau, step })
    }
}

// ============================================================================
// SupConLoss
// ============================================================================

/// Supervised Contrastive Loss.
///
/// Extends NT-Xent to the supervised setting: positives include all samples in
/// the batch that share the same class label, not just the augmented pair.
///
/// Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
#[derive(Debug, Clone)]
pub struct SupConLoss {
    /// Temperature scaling factor τ.
    pub temperature: f64,
    /// Contrast mode: `"all"` (use all views as anchors) or `"one"` (first view only).
    pub contrast_mode: SupConMode,
}

/// Mode for SupCon loss anchor selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupConMode {
    /// All views are used as anchors.
    All,
    /// Only the first view is used as an anchor.
    One,
}

impl Default for SupConLoss {
    fn default() -> Self {
        Self {
            temperature: 0.07,
            contrast_mode: SupConMode::All,
        }
    }
}

impl SupConLoss {
    /// Create a supervised contrastive loss.
    pub fn new(temperature: f64, contrast_mode: SupConMode) -> Self {
        Self {
            temperature,
            contrast_mode,
        }
    }

    /// Compute the SupCon loss.
    ///
    /// # Arguments
    /// * `features` - L2-normalised feature matrix, shape `[N * n_views, D]`
    ///   The `N` samples are arranged in contiguous blocks of `n_views` rows.
    ///   e.g., for `n_views=2`: `[f_0_v0, f_0_v1, f_1_v0, f_1_v1, ...]`
    /// * `labels`   - Class labels for each sample (length `N`).
    /// * `n_views`  - Number of augmented views per sample.
    pub fn forward<F>(
        &self,
        features: &Array2<F>,
        labels: &[usize],
        n_views: usize,
    ) -> Result<F>
    where
        F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
    {
        let total = features.nrows();
        let n = labels.len();
        if n == 0 {
            return Err(NeuralError::InvalidArgument(
                "SupConLoss: labels must not be empty".to_string(),
            ));
        }
        if n_views == 0 {
            return Err(NeuralError::InvalidArgument(
                "SupConLoss: n_views must be > 0".to_string(),
            ));
        }
        if total != n * n_views {
            return Err(NeuralError::ShapeMismatch(format!(
                "SupConLoss: features has {} rows but N={} * n_views={} = {}",
                total,
                n,
                n_views,
                n * n_views
            )));
        }

        let tau = F::from_f64(self.temperature).ok_or_else(|| {
            NeuralError::ComputationError("SupConLoss: cannot convert temperature".to_string())
        })?;

        let feats = l2_normalise(features)?;

        // Build anchor indices according to contrast_mode
        let anchor_indices: Vec<usize> = match self.contrast_mode {
            SupConMode::One => (0..n).collect(),
            SupConMode::All => (0..total).collect(),
        };

        let neg_inf = F::from_f64(-1e38).ok_or_else(|| {
            NeuralError::ComputationError("SupConLoss: cannot convert neg_inf".to_string())
        })?;

        let mut total_loss = F::zero();
        let mut num_valid = 0usize;

        for &anchor_idx in &anchor_indices {
            // Sample index for this anchor
            let sample_idx = anchor_idx % n;
            let anchor_label = labels[sample_idx];

            let anchor = feats.row(anchor_idx);

            // Collect positives and all other indices
            let mut pos_logits: Vec<F> = Vec::new();
            let mut all_logits: Vec<F> = Vec::new();

            for j in 0..total {
                if j == anchor_idx {
                    continue;
                }
                let j_sample = j % n;
                let sim: F = anchor.iter().zip(feats.row(j).iter())
                    .map(|(a, b)| *a * *b)
                    .fold(F::zero(), |acc, x| acc + x);
                let logit = sim / tau;
                all_logits.push(logit);
                if labels[j_sample] == anchor_label {
                    pos_logits.push(logit);
                }
            }

            if pos_logits.is_empty() {
                // No positives; skip this anchor
                continue;
            }

            // log_denom = log Σ_k exp(logit_k)
            let mut log_denom = neg_inf;
            for &logit in &all_logits {
                log_denom = log_sum_exp_pair(log_denom, logit);
            }

            // Loss for this anchor: -1/|P| Σ_{p in P} (logit_p - log_denom)
            let n_pos = F::from_usize(pos_logits.len()).ok_or_else(|| {
                NeuralError::ComputationError("SupConLoss: cannot convert n_pos".to_string())
            })?;
            let mut anchor_loss = F::zero();
            for &pos_logit in &pos_logits {
                anchor_loss += pos_logit - log_denom;
            }
            total_loss += -(anchor_loss / n_pos);
            num_valid += 1;
        }

        if num_valid == 0 {
            return Ok(F::zero());
        }
        let num_valid_f = F::from_usize(num_valid).ok_or_else(|| {
            NeuralError::ComputationError("SupConLoss: cannot convert num_valid".to_string())
        })?;
        Ok(total_loss / num_valid_f)
    }
}

// ============================================================================
// TripletMarginLoss
// ============================================================================

/// Online hard / semi-hard triplet mining strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TripletMiningStrategy {
    /// Hardest negative: the negative closest to the anchor.
    HardNegative,
    /// Semi-hard negative: farther than positive but within `margin`.
    SemiHard,
    /// Random negative.
    Random,
}

/// Triplet margin loss with online mining.
///
/// Given a batch of embeddings and their class labels, this loss function
/// automatically mines triplets according to the selected strategy.
#[derive(Debug, Clone)]
pub struct TripletMarginLoss {
    /// Minimum distance margin between positive and negative pairs.
    pub margin: f64,
    /// Online mining strategy.
    pub strategy: TripletMiningStrategy,
    /// Whether to apply L2 normalisation before distance computation.
    pub normalize: bool,
}

impl Default for TripletMarginLoss {
    fn default() -> Self {
        Self {
            margin: 1.0,
            strategy: TripletMiningStrategy::SemiHard,
            normalize: true,
        }
    }
}

impl TripletMarginLoss {
    /// Create a new triplet margin loss.
    pub fn new(margin: f64, strategy: TripletMiningStrategy, normalize: bool) -> Self {
        Self {
            margin,
            strategy,
            normalize,
        }
    }

    /// Compute the triplet loss with online mining.
    ///
    /// # Arguments
    /// * `embeddings` - Embedding matrix, shape `[N, D]`
    /// * `labels`     - Class labels, length `N`
    ///
    /// # Returns
    /// Scalar loss averaged over all valid mined triplets.
    pub fn forward<F>(&self, embeddings: &Array2<F>, labels: &[usize]) -> Result<F>
    where
        F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
    {
        let n = embeddings.nrows();
        if labels.len() != n {
            return Err(NeuralError::ShapeMismatch(format!(
                "TripletMarginLoss: embeddings has {} rows but labels has {} elements",
                n,
                labels.len()
            )));
        }

        let margin = F::from_f64(self.margin).ok_or_else(|| {
            NeuralError::ComputationError("TripletMarginLoss: cannot convert margin".to_string())
        })?;

        let emb = if self.normalize {
            l2_normalise(embeddings)?
        } else {
            embeddings.to_owned()
        };

        // Pre-compute pairwise squared distances
        let dist_sq = pairwise_squared_dist(&emb)?;

        let mut total_loss = F::zero();
        let mut count = 0usize;

        for a in 0..n {
            // Collect positive and negative indices
            let pos_indices: Vec<usize> = (0..n)
                .filter(|&k| k != a && labels[k] == labels[a])
                .collect();
            let neg_indices: Vec<usize> = (0..n)
                .filter(|&k| labels[k] != labels[a])
                .collect();

            if pos_indices.is_empty() || neg_indices.is_empty() {
                continue;
            }

            for &p in &pos_indices {
                let d_ap = dist_sq[[a, p]];

                let neg_choice = match self.strategy {
                    TripletMiningStrategy::HardNegative => {
                        // Negative with smallest distance to anchor
                        neg_indices.iter().copied().min_by(|&x, &y| {
                            dist_sq[[a, x]]
                                .partial_cmp(&dist_sq[[a, y]])
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                    }
                    TripletMiningStrategy::SemiHard => {
                        // Negative: d_an > d_ap but d_an < d_ap + margin
                        let upper = d_ap + margin;
                        let semi_hard: Vec<usize> = neg_indices
                            .iter()
                            .copied()
                            .filter(|&k| {
                                let d_an = dist_sq[[a, k]];
                                d_an > d_ap && d_an < upper
                            })
                            .collect();
                        if semi_hard.is_empty() {
                            // Fall back to hard negative
                            neg_indices.iter().copied().min_by(|&x, &y| {
                                dist_sq[[a, x]]
                                    .partial_cmp(&dist_sq[[a, y]])
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                        } else {
                            // Pick the semi-hard negative closest to positive
                            semi_hard.iter().copied().min_by(|&x, &y| {
                                dist_sq[[a, x]]
                                    .partial_cmp(&dist_sq[[a, y]])
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                        }
                    }
                    TripletMiningStrategy::Random => {
                        // Deterministic pseudo-random via hash of (a, p)
                        let idx = (a.wrapping_mul(131) ^ p.wrapping_mul(137)) % neg_indices.len();
                        Some(neg_indices[idx])
                    }
                };

                if let Some(neg) = neg_choice {
                    let d_an = dist_sq[[a, neg]];
                    let triplet_loss = (d_ap - d_an + margin).max(F::zero());
                    total_loss += triplet_loss;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return Ok(F::zero());
        }
        let count_f = F::from_usize(count).ok_or_else(|| {
            NeuralError::ComputationError("TripletMarginLoss: cannot convert count".to_string())
        })?;
        Ok(total_loss / count_f)
    }
}

// ============================================================================
// ContrastivePairLoss (pairwise margin-based)
// ============================================================================

/// Pairwise contrastive loss (margin-based).
///
/// For a pair `(x1, x2)` with binary label `y` (`1` = similar, `0` = dissimilar):
///
/// ```text
/// L = y * d² + (1 - y) * max(0, margin - d)²
/// ```
///
/// This mirrors the existing `crate::losses::ContrastiveLoss` but lives in the
/// training module alongside the other contrastive-learning utilities.
#[derive(Debug, Clone, Copy)]
pub struct ContrastivePairLoss {
    /// Distance margin for dissimilar pairs.
    pub margin: f64,
    /// Whether to L2-normalise embeddings before computing distance.
    pub normalize: bool,
}

impl Default for ContrastivePairLoss {
    fn default() -> Self {
        Self {
            margin: 1.0,
            normalize: false,
        }
    }
}

impl ContrastivePairLoss {
    /// Create a new pairwise contrastive loss.
    pub fn new(margin: f64, normalize: bool) -> Self {
        Self { margin, normalize }
    }

    /// Compute the pairwise contrastive loss.
    ///
    /// # Arguments
    /// * `emb1`   - First set of embeddings, shape `[N, D]`
    /// * `emb2`   - Second set of embeddings, shape `[N, D]`
    /// * `labels` - Similarity labels (1.0 = similar, 0.0 = dissimilar), length `N`
    pub fn forward<F>(&self, emb1: &Array2<F>, emb2: &Array2<F>, labels: &[F]) -> Result<F>
    where
        F: Float + Debug + NumAssign + FromPrimitive,
    {
        let n = emb1.nrows();
        if emb2.nrows() != n || labels.len() != n {
            return Err(NeuralError::ShapeMismatch(
                "ContrastivePairLoss: emb1, emb2, labels must have same batch size".to_string(),
            ));
        }

        let margin = F::from_f64(self.margin).ok_or_else(|| {
            NeuralError::ComputationError(
                "ContrastivePairLoss: cannot convert margin".to_string(),
            )
        })?;

        let e1 = if self.normalize {
            l2_normalise(emb1)?
        } else {
            emb1.to_owned()
        };
        let e2 = if self.normalize {
            l2_normalise(emb2)?
        } else {
            emb2.to_owned()
        };

        let mut total = F::zero();
        for i in 0..n {
            let diff = e1.row(i).to_owned() - e2.row(i).to_owned();
            let dist_sq: F = diff.iter().map(|x| *x * *x).fold(F::zero(), |a, b| a + b);
            let dist = dist_sq.sqrt();
            let y = labels[i];
            let pair_loss = if y > F::zero() {
                dist_sq
            } else {
                let margin_term = (margin - dist).max(F::zero());
                margin_term * margin_term
            };
            total += pair_loss;
        }

        let n_f = F::from_usize(n).ok_or_else(|| {
            NeuralError::ComputationError(
                "ContrastivePairLoss: cannot convert N".to_string(),
            )
        })?;
        Ok(total / n_f)
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

/// L2-normalise each row of a 2-D array.
fn l2_normalise<F>(x: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + Debug + NumAssign + FromPrimitive,
{
    let eps = F::from_f64(1e-12).ok_or_else(|| {
        NeuralError::ComputationError("l2_normalise: cannot convert eps".to_string())
    })?;
    let norms = x
        .map_axis(Axis(1), |row| {
            let sq: F = row.iter().map(|v| *v * *v).fold(F::zero(), |a, b| a + b);
            sq.sqrt().max(eps)
        });
    let mut out = x.to_owned();
    for (mut row, &norm) in out.rows_mut().into_iter().zip(norms.iter()) {
        row.mapv_inplace(|v| v / norm);
    }
    Ok(out)
}

/// Concatenate two `Array2` row-wise.
fn concatenate_rows<F: Float + Debug>(a: &Array2<F>, b: &Array2<F>) -> Result<Array2<F>> {
    if a.ncols() != b.ncols() {
        return Err(NeuralError::ShapeMismatch(format!(
            "concatenate_rows: a has {} cols but b has {}",
            a.ncols(),
            b.ncols()
        )));
    }
    let mut out =
        Array2::zeros((a.nrows() + b.nrows(), a.ncols()));
    for (i, row) in a.rows().into_iter().enumerate() {
        out.row_mut(i).assign(&row);
    }
    for (i, row) in b.rows().into_iter().enumerate() {
        out.row_mut(a.nrows() + i).assign(&row);
    }
    Ok(out)
}

/// Compute the cosine similarity matrix for a `[N, D]` matrix.
/// Both rows are assumed to be L2-normalised already.
fn cosine_sim_matrix<F: Float + Debug + NumAssign>(x: &Array2<F>) -> Result<Array2<F>> {
    let n = x.nrows();
    let mut sim = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let dot: F = x
                .row(i)
                .iter()
                .zip(x.row(j).iter())
                .map(|(a, b)| *a * *b)
                .fold(F::zero(), |acc, v| acc + v);
            sim[[i, j]] = dot;
        }
    }
    Ok(sim)
}

/// Numerically stable `log(exp(a) + exp(b))`.
fn log_sum_exp_pair<F: Float>(a: F, b: F) -> F {
    if a > b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}

/// Pairwise squared Euclidean distance matrix for a `[N, D]` array.
fn pairwise_squared_dist<F: Float + Debug + NumAssign>(x: &Array2<F>) -> Result<Array2<F>> {
    let n = x.nrows();
    let mut dist = Array2::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let d: F = x
                .row(i)
                .iter()
                .zip(x.row(j).iter())
                .map(|(a, b)| {
                    let diff = *a - *b;
                    diff * diff
                })
                .fold(F::zero(), |acc, v| acc + v);
            dist[[i, j]] = d;
            dist[[j, i]] = d;
        }
    }
    Ok(dist)
}

/// Simple linear forward pass: output = x @ W.T + b
fn linear_forward<F: Float + Debug + NumAssign>(
    x: &Array2<F>,
    w: &Array2<F>,
    b: &Array1<F>,
) -> Result<Array2<F>> {
    let n = x.nrows();
    let in_dim = x.ncols();
    let out_dim = w.nrows();
    if w.ncols() != in_dim {
        return Err(NeuralError::ShapeMismatch(format!(
            "linear_forward: x has {} cols but W has {} cols",
            in_dim,
            w.ncols()
        )));
    }
    if b.len() != out_dim {
        return Err(NeuralError::ShapeMismatch(format!(
            "linear_forward: W has {} rows but b has {} elements",
            out_dim,
            b.len()
        )));
    }
    let mut out = Array2::zeros((n, out_dim));
    for i in 0..n {
        for j in 0..out_dim {
            let dot: F = x
                .row(i)
                .iter()
                .zip(w.row(j).iter())
                .map(|(a, wij)| *a * *wij)
                .fold(F::zero(), |acc, v| acc + v);
            out[[i, j]] = dot + b[j];
        }
    }
    Ok(out)
}

/// Element-wise ReLU for a 2-D array.
fn relu2d<F: Float>(x: &Array2<F>) -> Array2<F> {
    x.mapv(|v| v.max(F::zero()))
}

/// Create a weight matrix of shape `[rows, cols]` with alternating ±scale.
fn init_weight_matrix<F: Float + FromPrimitive>(rows: usize, cols: usize, scale: F) -> Array2<F> {
    let mut w = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            let sign = if (i + j) % 2 == 0 { F::one() } else { -F::one() };
            w[[i, j]] = sign * scale;
        }
    }
    w
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_ntxent_basic() {
        let loss_fn = NTXentLoss::new(0.1);
        let n = 4;
        let d = 8;
        // Create two distinct views
        let z_i = Array2::<f64>::from_shape_fn((n, d), |(i, j)| {
            ((i * d + j) as f64 * 0.1).sin()
        });
        let z_j = Array2::<f64>::from_shape_fn((n, d), |(i, j)| {
            ((i * d + j) as f64 * 0.1 + 0.5).cos()
        });
        let loss = loss_fn.forward(&z_i, &z_j).expect("NT-Xent forward");
        assert!(loss.is_finite(), "NT-Xent loss should be finite");
        assert!(loss >= 0.0, "NT-Xent loss should be non-negative");
    }

    #[test]
    fn test_byol_update() {
        let config = BYOLConfig::default();
        let n = 4;
        let d = 8;
        let online = Array2::<f64>::from_shape_fn((n, d), |(i, j)| (i + j) as f64 * 0.01);
        let target = Array2::<f64>::from_shape_fn((n, d), |(i, j)| (i + j) as f64 * 0.02);
        let update =
            BYOLUpdate::compute(&online, &target, &config, 1000).expect("BYOL update");
        assert!(update.loss.is_finite());
        assert!(update.tau > 0.0 && update.tau <= 1.0);
    }

    #[test]
    fn test_supcon_loss() {
        let loss_fn = SupConLoss::new(0.07, SupConMode::All);
        // 4 samples, 2 views each → 8 rows
        let features = Array2::<f64>::from_shape_fn((8, 16), |(i, j)| {
            ((i * 16 + j) as f64 * 0.1).sin()
        });
        let labels = vec![0usize, 0, 1, 1];
        let loss = loss_fn.forward(&features, &labels, 2).expect("SupCon forward");
        assert!(loss.is_finite());
    }

    #[test]
    fn test_triplet_margin_loss_hard() {
        let loss_fn = TripletMarginLoss::new(1.0, TripletMiningStrategy::HardNegative, true);
        let embeddings = Array2::<f64>::from_shape_fn((6, 4), |(i, j)| {
            ((i * 4 + j) as f64).sin()
        });
        let labels = vec![0usize, 0, 1, 1, 2, 2];
        let loss = loss_fn.forward(&embeddings, &labels).expect("Triplet forward");
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_contrastive_pair_loss() {
        let loss_fn = ContrastivePairLoss::new(1.0, true);
        let emb1 = Array2::<f64>::from_shape_fn((4, 8), |(i, j)| (i + j) as f64 * 0.1);
        let emb2 = Array2::<f64>::from_shape_fn((4, 8), |(i, j)| (i + j) as f64 * 0.2);
        let labels = vec![1.0_f64, 0.0, 1.0, 0.0];
        let loss = loss_fn
            .forward(&emb1, &emb2, &labels)
            .expect("ContrastivePairLoss forward");
        assert!(loss.is_finite());
    }

    #[test]
    fn test_moco_queue() {
        let mut queue: MoCoQueue<f64> = MoCoQueue::new(16, 8, 0.999, 0.07);
        let keys = Array2::<f64>::from_shape_fn((4, 8), |(i, j)| (i + j) as f64 * 0.1);
        queue.enqueue_and_dequeue(&keys).expect("enqueue");
        assert_eq!(queue.len(), 4);

        let queries = Array2::<f64>::from_shape_fn((4, 8), |(i, j)| (i + j) as f64 * 0.05);
        let loss = queue.info_nce_loss(&queries, &keys).expect("InfoNCE");
        assert!(loss.is_finite());
    }

    #[test]
    fn test_simclr_trainer() {
        let config = SimCLRConfig {
            representation_dim: 16,
            projection_hidden_dim: 32,
            projection_output_dim: 8,
            temperature: 0.1,
            weight_decay: 1e-4,
        };
        let trainer = SimCLRTrainer::<f64>::new(config).expect("SimCLRTrainer::new");
        let rep_i = Array2::<f64>::from_shape_fn((4, 16), |(i, j)| (i + j) as f64 * 0.01);
        let rep_j = Array2::<f64>::from_shape_fn((4, 16), |(i, j)| (i + j) as f64 * 0.02);
        let loss = trainer.batch_loss(&rep_i, &rep_j).expect("batch_loss");
        assert!(loss.is_finite());
    }
}
