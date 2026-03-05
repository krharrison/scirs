//! Meta-Learning Algorithms
//!
//! Implements model-agnostic and few-shot learning algorithms:
//!
//! - **MAML**: Model-Agnostic Meta-Learning (Finn et al., ICML 2017)
//! - **PrototypicalNetwork**: Few-shot classification via class prototypes
//! - **RelationNetwork**: Few-shot classification via a learned relation module
//! - **MetaEpisode**: N-way K-shot episode sampler (support + query sets)
//! - **MAMLResult**: Container for inner/outer losses and adapted parameters
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::meta_learning::{
//!     MetaEpisode, PrototypicalNetwork, PrototypicalConfig,
//! };
//! use scirs2_core::ndarray::Array2;
//!
//! // Build 2-way 2-shot episode from 10 labelled embeddings
//! let embeddings = Array2::<f64>::from_shape_fn((10, 8), |(i, j)| (i + j) as f64 * 0.1);
//! let labels: Vec<usize> = (0..10).map(|i| i % 2).collect();
//! let episode = MetaEpisode::sample(&embeddings, &labels, 2, 2, 4, 42)
//!     .expect("episode sampled");
//! assert_eq!(episode.n_way, 2);
//! assert_eq!(episode.k_shot, 2);
//!
//! // Classify query embeddings using prototypical network
//! let config = PrototypicalConfig { distance: DistanceMetric::Euclidean };
//! let net = PrototypicalNetwork::new(config);
//! let preds = net.predict(&episode).expect("proto predict");
//! assert_eq!(preds.len(), episode.query_embeddings.nrows());
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

// ============================================================================
// MetaEpisode
// ============================================================================

/// A single N-way K-shot meta-learning episode.
///
/// An episode contains:
/// * A *support set*: `N × K` labelled embeddings used to define class prototypes
///   or to train an inner-loop model.
/// * A *query set*: `Q` embeddings per class used to evaluate the adapted model.
#[derive(Debug, Clone)]
pub struct MetaEpisode<F: Float + Debug> {
    /// Number of classes in this episode.
    pub n_way: usize,
    /// Number of labelled examples per class in the support set.
    pub k_shot: usize,
    /// Number of query examples per class.
    pub q_query: usize,
    /// Support embeddings, shape `[N * K, D]`.
    pub support_embeddings: Array2<F>,
    /// Support class labels (0-indexed within episode), length `N * K`.
    pub support_labels: Vec<usize>,
    /// Query embeddings, shape `[N * Q, D]`.
    pub query_embeddings: Array2<F>,
    /// Query class labels (0-indexed within episode), length `N * Q`.
    pub query_labels: Vec<usize>,
}

impl<F: Float + Debug + NumAssign + FromPrimitive> MetaEpisode<F> {
    /// Sample a random N-way K-shot episode from a dataset.
    ///
    /// # Arguments
    /// * `embeddings` - Full embedding matrix, shape `[M, D]`
    /// * `labels`     - Class labels for every sample, length `M`
    /// * `n_way`      - Number of classes per episode
    /// * `k_shot`     - Support examples per class
    /// * `q_query`    - Query examples per class
    /// * `seed`       - Deterministic seed for reproducible sampling
    ///
    /// # Returns
    /// A `MetaEpisode` constructed from the sampled indices.
    pub fn sample(
        embeddings: &Array2<F>,
        labels: &[usize],
        n_way: usize,
        k_shot: usize,
        q_query: usize,
        seed: u64,
    ) -> Result<Self> {
        let m = embeddings.nrows();
        if labels.len() != m {
            return Err(NeuralError::ShapeMismatch(format!(
                "MetaEpisode::sample: embeddings has {} rows but labels has {} elements",
                m,
                labels.len()
            )));
        }
        if n_way == 0 || k_shot == 0 || q_query == 0 {
            return Err(NeuralError::InvalidArgument(
                "MetaEpisode::sample: n_way, k_shot, q_query must all be > 0".to_string(),
            ));
        }

        // Group indices by class
        let mut class_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, &lbl) in labels.iter().enumerate() {
            class_map.entry(lbl).or_default().push(idx);
        }

        // Gather classes that have enough samples
        let mut eligible: Vec<usize> = class_map
            .iter()
            .filter(|(_, v)| v.len() >= k_shot + q_query)
            .map(|(k, _)| *k)
            .collect();
        eligible.sort_unstable(); // deterministic order

        if eligible.len() < n_way {
            return Err(NeuralError::InvalidArgument(format!(
                "MetaEpisode::sample: need {} classes with ≥ {} samples, found {}",
                n_way,
                k_shot + q_query,
                eligible.len()
            )));
        }

        // Pseudo-randomly select n_way classes
        let chosen_classes = lcg_sample(&eligible, n_way, seed);

        let d = embeddings.ncols();
        let support_n = n_way * k_shot;
        let query_n = n_way * q_query;

        let mut support_emb = Array2::zeros((support_n, d));
        let mut support_lbl = Vec::with_capacity(support_n);
        let mut query_emb = Array2::zeros((query_n, d));
        let mut query_lbl = Vec::with_capacity(query_n);

        for (ep_class, &global_class) in chosen_classes.iter().enumerate() {
            let samples = &class_map[&global_class];
            // Use LCG to select k_shot + q_query samples without replacement
            let selected = lcg_sample(samples, k_shot + q_query, seed ^ (ep_class as u64 + 1));

            for (i, &sample_idx) in selected[..k_shot].iter().enumerate() {
                let row = ep_class * k_shot + i;
                support_emb.row_mut(row).assign(&embeddings.row(sample_idx));
                support_lbl.push(ep_class);
            }
            for (i, &sample_idx) in selected[k_shot..].iter().enumerate() {
                let row = ep_class * q_query + i;
                query_emb.row_mut(row).assign(&embeddings.row(sample_idx));
                query_lbl.push(ep_class);
            }
        }

        Ok(Self {
            n_way,
            k_shot,
            q_query,
            support_embeddings: support_emb,
            support_labels: support_lbl,
            query_embeddings: query_emb,
            query_labels: query_lbl,
        })
    }
}

// ============================================================================
// PrototypicalNetwork
// ============================================================================

/// Distance metric for prototypical and relation networks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Squared Euclidean distance.
    Euclidean,
    /// Cosine distance (1 - cosine similarity).
    Cosine,
    /// Manhattan (L1) distance.
    Manhattan,
}

/// Configuration for `PrototypicalNetwork`.
#[derive(Debug, Clone)]
pub struct PrototypicalConfig {
    /// Distance metric used to compare query embeddings to prototypes.
    pub distance: DistanceMetric,
}

impl Default for PrototypicalConfig {
    fn default() -> Self {
        Self {
            distance: DistanceMetric::Euclidean,
        }
    }
}

/// Prototypical Network for few-shot classification.
///
/// Class prototypes are the mean of the support embeddings for each class.
/// Query samples are classified by finding the nearest prototype.
///
/// Reference: Snell et al., "Prototypical Networks for Few-shot Learning",
/// NeurIPS 2017.
#[derive(Debug, Clone)]
pub struct PrototypicalNetwork {
    config: PrototypicalConfig,
}

impl PrototypicalNetwork {
    /// Create a new prototypical network.
    pub fn new(config: PrototypicalConfig) -> Self {
        Self { config }
    }

    /// Compute class prototypes from a support set.
    ///
    /// Returns `Array2<F>` of shape `[N, D]` where N = n_way.
    pub fn compute_prototypes<F>(
        &self,
        support_embeddings: &Array2<F>,
        support_labels: &[usize],
        n_way: usize,
    ) -> Result<Array2<F>>
    where
        F: Float + Debug + NumAssign + FromPrimitive,
    {
        let d = support_embeddings.ncols();
        let mut proto = Array2::zeros((n_way, d));
        let mut counts = vec![0usize; n_way];

        for (i, &lbl) in support_labels.iter().enumerate() {
            if lbl >= n_way {
                return Err(NeuralError::InvalidArgument(format!(
                    "PrototypicalNetwork: support label {} exceeds n_way {}",
                    lbl, n_way
                )));
            }
            for j in 0..d {
                proto[[lbl, j]] += support_embeddings[[i, j]];
            }
            counts[lbl] += 1;
        }

        for lbl in 0..n_way {
            if counts[lbl] == 0 {
                return Err(NeuralError::InvalidArgument(format!(
                    "PrototypicalNetwork: class {} has no support samples",
                    lbl
                )));
            }
            let cnt = F::from_usize(counts[lbl]).ok_or_else(|| {
                NeuralError::ComputationError(
                    "PrototypicalNetwork: cannot convert count".to_string(),
                )
            })?;
            for j in 0..d {
                proto[[lbl, j]] /= cnt;
            }
        }

        Ok(proto)
    }

    /// Classify query embeddings using nearest-prototype rule.
    ///
    /// # Arguments
    /// * `episode` - A meta-learning episode containing support and query sets.
    ///
    /// # Returns
    /// Predicted class indices for each query sample.
    pub fn predict<F>(&self, episode: &MetaEpisode<F>) -> Result<Vec<usize>>
    where
        F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
    {
        let prototypes = self.compute_prototypes(
            &episode.support_embeddings,
            &episode.support_labels,
            episode.n_way,
        )?;

        let q = episode.query_embeddings.nrows();
        let mut preds = Vec::with_capacity(q);

        for qi in 0..q {
            let query = episode.query_embeddings.row(qi);
            let mut best_class = 0;
            let mut best_dist = F::infinity();

            for ci in 0..episode.n_way {
                let proto = prototypes.row(ci);
                let dist = compute_distance(&query.to_owned(), &proto.to_owned(), self.config.distance)?;
                if dist < best_dist {
                    best_dist = dist;
                    best_class = ci;
                }
            }
            preds.push(best_class);
        }

        Ok(preds)
    }

    /// Compute the prototypical loss on the query set of an episode.
    ///
    /// Uses softmax over negative distances (log-softmax cross-entropy).
    pub fn episode_loss<F>(&self, episode: &MetaEpisode<F>) -> Result<F>
    where
        F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
    {
        let prototypes = self.compute_prototypes(
            &episode.support_embeddings,
            &episode.support_labels,
            episode.n_way,
        )?;

        let q = episode.query_embeddings.nrows();
        if q == 0 {
            return Ok(F::zero());
        }

        let mut total_loss = F::zero();
        let neg_inf = F::from_f64(-1e38).ok_or_else(|| {
            NeuralError::ComputationError("episode_loss: cannot convert neg_inf".to_string())
        })?;

        for qi in 0..q {
            let query = episode.query_embeddings.row(qi).to_owned();
            let true_class = episode.query_labels[qi];

            // Compute negative distances (logits)
            let mut neg_dists = Vec::with_capacity(episode.n_way);
            for ci in 0..episode.n_way {
                let proto = prototypes.row(ci).to_owned();
                let dist = compute_distance(&query, &proto, self.config.distance)?;
                neg_dists.push(-dist); // negate for softmax
            }

            // Log-softmax cross-entropy
            let mut log_denom = neg_inf;
            for &d in &neg_dists {
                log_denom = log_sum_exp(log_denom, d);
            }
            let log_prob = neg_dists[true_class] - log_denom;
            total_loss += -log_prob;
        }

        let q_f = F::from_usize(q).ok_or_else(|| {
            NeuralError::ComputationError("episode_loss: cannot convert Q".to_string())
        })?;
        Ok(total_loss / q_f)
    }
}

// ============================================================================
// RelationNetwork
// ============================================================================

/// A learned relation module: a 2-layer MLP that scores (query, prototype) pairs.
#[derive(Debug, Clone)]
pub struct RelationModule<F: Float + Debug + NumAssign + FromPrimitive> {
    /// Weight matrix for layer 1, shape `[hidden, 2*D]`
    pub w1: Array2<F>,
    /// Bias for layer 1, shape `[hidden]`
    pub b1: Array1<F>,
    /// Weight matrix for layer 2, shape `[1, hidden]`
    pub w2: Array2<F>,
    /// Bias for layer 2, shape `[1]`
    pub b2: Array1<F>,
}

impl<F: Float + Debug + NumAssign + FromPrimitive> RelationModule<F> {
    /// Create a new relation module.
    ///
    /// # Arguments
    /// * `embedding_dim` - Dimensionality of query and prototype embeddings.
    /// * `hidden_dim`    - Hidden layer size.
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Result<Self> {
        let in_dim = 2 * embedding_dim;
        let scale1 = F::from_f64((2.0 / (in_dim + hidden_dim) as f64).sqrt()).ok_or_else(|| {
            NeuralError::ComputationError("RelationModule: cannot compute scale1".to_string())
        })?;
        let scale2 = F::from_f64((2.0 / (hidden_dim + 1) as f64).sqrt()).ok_or_else(|| {
            NeuralError::ComputationError("RelationModule: cannot compute scale2".to_string())
        })?;

        let w1 = alternating_weight_matrix(hidden_dim, in_dim, scale1);
        let b1 = Array1::zeros(hidden_dim);
        let w2 = alternating_weight_matrix(1, hidden_dim, scale2);
        let b2 = Array1::zeros(1);

        Ok(Self { w1, b1, w2, b2 })
    }

    /// Compute relation scores for `(query, prototype)` pairs.
    ///
    /// # Arguments
    /// * `query`     - Query embedding, shape `[D]`
    /// * `prototype` - Prototype embedding, shape `[D]`
    ///
    /// # Returns
    /// Scalar relation score (higher = more similar).
    pub fn score(&self, query: &Array1<F>, prototype: &Array1<F>) -> Result<F> {
        let d = query.len();
        if prototype.len() != d {
            return Err(NeuralError::ShapeMismatch(format!(
                "RelationModule::score: query dim {} != prototype dim {}",
                d,
                prototype.len()
            )));
        }
        // Concatenate query and prototype → [2D]
        let mut concat = Array1::zeros(2 * d);
        for i in 0..d {
            concat[i] = query[i];
            concat[i + d] = prototype[i];
        }

        // Layer 1: h = relu(W1 @ x + b1)
        let h_dim = self.w1.nrows();
        let mut h = Array1::zeros(h_dim);
        for j in 0..h_dim {
            let dot: F = self
                .w1
                .row(j)
                .iter()
                .zip(concat.iter())
                .map(|(w, x)| *w * *x)
                .fold(F::zero(), |a, b| a + b);
            h[j] = (dot + self.b1[j]).max(F::zero()); // ReLU
        }

        // Layer 2: out = W2 @ h + b2 → sigmoid → scalar
        let dot2: F = self
            .w2
            .row(0)
            .iter()
            .zip(h.iter())
            .map(|(w, x)| *w * *x)
            .fold(F::zero(), |a, b| a + b);
        let logit = dot2 + self.b2[0];
        // Sigmoid activation
        let score = sigmoid(logit);
        Ok(score)
    }
}

/// Relation Network for few-shot learning.
///
/// Compares query embeddings to class prototypes via a learned relation module,
/// rather than using a fixed distance metric.
///
/// Reference: Sung et al., "Learning to Compare: Relation Network for
/// Few-Shot Learning", CVPR 2018.
#[derive(Debug, Clone)]
pub struct RelationNetwork<F: Float + Debug + NumAssign + FromPrimitive> {
    /// Learned relation scoring module.
    pub relation_module: RelationModule<F>,
    /// Hidden dimension of the relation module.
    pub hidden_dim: usize,
}

impl<F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive> RelationNetwork<F> {
    /// Create a new relation network.
    ///
    /// # Arguments
    /// * `embedding_dim` - Dimensionality of embeddings.
    /// * `hidden_dim`    - Hidden size in the relation module.
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Result<Self> {
        let relation_module = RelationModule::new(embedding_dim, hidden_dim)?;
        Ok(Self {
            relation_module,
            hidden_dim,
        })
    }

    /// Classify query embeddings using the relation network.
    ///
    /// Prototypes are computed as class means of the support set.
    ///
    /// # Returns
    /// Predicted class indices for each query sample.
    pub fn predict(&self, episode: &MetaEpisode<F>) -> Result<Vec<usize>>
    where
        F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
    {
        let proto_net = PrototypicalNetwork::new(PrototypicalConfig::default());
        let prototypes = proto_net.compute_prototypes(
            &episode.support_embeddings,
            &episode.support_labels,
            episode.n_way,
        )?;

        let q = episode.query_embeddings.nrows();
        let mut preds = Vec::with_capacity(q);

        for qi in 0..q {
            let query = episode.query_embeddings.row(qi).to_owned();
            let mut best_class = 0;
            let mut best_score = F::neg_infinity();

            for ci in 0..episode.n_way {
                let proto = prototypes.row(ci).to_owned();
                let score = self.relation_module.score(&query, &proto)?;
                if score > best_score {
                    best_score = score;
                    best_class = ci;
                }
            }
            preds.push(best_class);
        }

        Ok(preds)
    }
}

// ============================================================================
// MAML
// ============================================================================

/// Result from one MAML update step.
#[derive(Debug, Clone)]
pub struct MAMLResult<F: Float + Debug> {
    /// Mean inner-loop loss across all tasks in the meta-batch.
    pub inner_loss: F,
    /// Outer (meta) loss after inner-loop adaptation.
    pub outer_loss: F,
    /// Number of tasks processed in this meta-batch.
    pub num_tasks: usize,
    /// Number of inner-loop gradient steps taken.
    pub inner_steps: usize,
}

/// Configuration for MAML.
#[derive(Debug, Clone)]
pub struct MAMLConfig {
    /// Inner-loop learning rate α.
    pub inner_lr: f64,
    /// Number of inner-loop gradient steps.
    pub inner_steps: usize,
    /// Outer-loop (meta) learning rate β.
    pub outer_lr: f64,
    /// Whether to compute second-order gradients (true = MAML, false = FOMAML).
    pub second_order: bool,
    /// Number of tasks per meta-batch.
    pub meta_batch_size: usize,
}

impl Default for MAMLConfig {
    fn default() -> Self {
        Self {
            inner_lr: 0.01,
            inner_steps: 5,
            outer_lr: 1e-3,
            second_order: false, // FOMAML by default (cheap)
            meta_batch_size: 4,
        }
    }
}

impl MAMLConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        if self.inner_lr <= 0.0 || self.outer_lr <= 0.0 {
            return Err(NeuralError::ConfigError(
                "MAMLConfig: learning rates must be > 0".to_string(),
            ));
        }
        if self.inner_steps == 0 {
            return Err(NeuralError::ConfigError(
                "MAMLConfig: inner_steps must be > 0".to_string(),
            ));
        }
        if self.meta_batch_size == 0 {
            return Err(NeuralError::ConfigError(
                "MAMLConfig: meta_batch_size must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Model-Agnostic Meta-Learning (MAML).
///
/// MAML finds an initial set of parameters θ such that a small number of
/// gradient steps on a new task's support set leads to good performance on
/// that task's query set.
///
/// This implementation represents model parameters as a flat `Vec<F>` of
/// weights and provides the inner-loop SGD adaptation and the outer loss
/// aggregation. The actual gradient computation must be supplied by the caller
/// via a closure that returns `(loss, grad)` given `(params, data)`.
///
/// Reference: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation
/// of Deep Networks", ICML 2017.
#[derive(Debug, Clone)]
pub struct MAML<F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive> {
    /// Meta-learning configuration.
    pub config: MAMLConfig,
    /// Current (meta) model parameters θ.
    pub params: Vec<F>,
}

impl<F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive> MAML<F> {
    /// Create a new MAML instance with the given initial parameters.
    pub fn new(config: MAMLConfig, initial_params: Vec<F>) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            params: initial_params,
        })
    }

    /// Perform the inner-loop SGD adaptation for a single task.
    ///
    /// Starting from `init_params`, takes `inner_steps` gradient steps on the
    /// support set using the provided loss/gradient oracle.
    ///
    /// # Arguments
    /// * `init_params`  - Starting parameters (copy of the meta parameters θ)
    /// * `loss_and_grad` - A closure `(params, phase) → (loss, grad)` where
    ///   `phase = "support"` during inner adaptation and `"query"` during
    ///   outer evaluation.
    ///
    /// # Returns
    /// `(adapted_params, inner_loss)` – the adapted parameters and the mean
    /// support loss across inner steps.
    pub fn inner_loop<G>(
        &self,
        init_params: &[F],
        mut loss_and_grad: G,
    ) -> Result<(Vec<F>, F)>
    where
        G: FnMut(&[F], &str) -> Result<(F, Vec<F>)>,
    {
        let alpha = F::from_f64(self.config.inner_lr).ok_or_else(|| {
            NeuralError::ComputationError("MAML::inner_loop: cannot convert inner_lr".to_string())
        })?;

        let mut theta_prime: Vec<F> = init_params.to_vec();
        let mut total_inner_loss = F::zero();

        for _step in 0..self.config.inner_steps {
            let (loss, grad) = loss_and_grad(&theta_prime, "support")?;
            if grad.len() != theta_prime.len() {
                return Err(NeuralError::ShapeMismatch(format!(
                    "MAML::inner_loop: param len {} != grad len {}",
                    theta_prime.len(),
                    grad.len()
                )));
            }
            // SGD step: θ' ← θ' - α * ∇L
            for (p, g) in theta_prime.iter_mut().zip(grad.iter()) {
                *p -= alpha * *g;
            }
            total_inner_loss += loss;
        }

        let steps_f = F::from_usize(self.config.inner_steps).ok_or_else(|| {
            NeuralError::ComputationError("MAML::inner_loop: cannot convert steps".to_string())
        })?;
        let mean_inner_loss = total_inner_loss / steps_f;

        Ok((theta_prime, mean_inner_loss))
    }

    /// Perform one MAML meta-update over a batch of tasks.
    ///
    /// For each task `i`:
    /// 1. Clone θ and adapt via inner-loop SGD on the support set → θ'_i
    /// 2. Evaluate query loss at θ'_i.
    ///
    /// The meta-gradient is the mean query gradient across tasks (FOMAML).
    ///
    /// # Arguments
    /// * `tasks` - A list of loss-and-gradient oracles, one per task.
    ///
    /// # Returns
    /// `MAMLResult` with aggregated losses.
    pub fn meta_update<G>(&mut self, tasks: &mut [G]) -> Result<MAMLResult<F>>
    where
        G: FnMut(&[F], &str) -> Result<(F, Vec<F>)>,
    {
        let num_tasks = tasks.len();
        if num_tasks == 0 {
            return Err(NeuralError::InvalidArgument(
                "MAML::meta_update: tasks list must not be empty".to_string(),
            ));
        }

        let beta = F::from_f64(self.config.outer_lr).ok_or_else(|| {
            NeuralError::ComputationError("MAML::meta_update: cannot convert outer_lr".to_string())
        })?;
        let n_tasks_f = F::from_usize(num_tasks).ok_or_else(|| {
            NeuralError::ComputationError("MAML::meta_update: cannot convert num_tasks".to_string())
        })?;

        let mut total_inner_loss = F::zero();
        let mut total_outer_loss = F::zero();
        let mut meta_grad = vec![F::zero(); self.params.len()];

        for task in tasks.iter_mut() {
            // Inner adaptation
            let (theta_prime, inner_loss) = self.inner_loop(&self.params.clone(), &mut *task)?;
            total_inner_loss += inner_loss;

            // Outer evaluation at adapted parameters
            let (query_loss, query_grad) = task(&theta_prime, "query")?;
            total_outer_loss += query_loss;

            if query_grad.len() != meta_grad.len() {
                return Err(NeuralError::ShapeMismatch(format!(
                    "MAML::meta_update: meta_grad len {} != query_grad len {}",
                    meta_grad.len(),
                    query_grad.len()
                )));
            }
            for (mg, &qg) in meta_grad.iter_mut().zip(query_grad.iter()) {
                *mg += qg;
            }
        }

        // Normalise and apply meta-gradient update: θ ← θ - β * (1/T) Σ ∇L_i(θ'_i)
        for (p, mg) in self.params.iter_mut().zip(meta_grad.iter()) {
            *p -= beta * (*mg / n_tasks_f);
        }

        Ok(MAMLResult {
            inner_loss: total_inner_loss / n_tasks_f,
            outer_loss: total_outer_loss / n_tasks_f,
            num_tasks,
            inner_steps: self.config.inner_steps,
        })
    }

    /// Adapt the current meta-parameters to a new task (inference / fine-tune).
    ///
    /// Returns the adapted parameters without modifying `self.params`.
    pub fn adapt<G>(&self, mut loss_and_grad: G) -> Result<Vec<F>>
    where
        G: FnMut(&[F], &str) -> Result<(F, Vec<F>)>,
    {
        let (adapted, _) = self.inner_loop(&self.params, &mut loss_and_grad)?;
        Ok(adapted)
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Compute the distance between two 1-D vectors.
fn compute_distance<F: Float + Debug + NumAssign + FromPrimitive>(
    a: &Array1<F>,
    b: &Array1<F>,
    metric: DistanceMetric,
) -> Result<F> {
    if a.len() != b.len() {
        return Err(NeuralError::ShapeMismatch(format!(
            "compute_distance: a has {} dims but b has {}",
            a.len(),
            b.len()
        )));
    }
    let eps = F::from_f64(1e-12).ok_or_else(|| {
        NeuralError::ComputationError("compute_distance: cannot convert eps".to_string())
    })?;

    let dist = match metric {
        DistanceMetric::Euclidean => {
            let sq: F = a
                .iter()
                .zip(b.iter())
                .map(|(&ai, &bi)| {
                    let d = ai - bi;
                    d * d
                })
                .fold(F::zero(), |acc, v| acc + v);
            sq
        }
        DistanceMetric::Cosine => {
            let dot: F = a
                .iter()
                .zip(b.iter())
                .map(|(&ai, &bi)| ai * bi)
                .fold(F::zero(), |acc, v| acc + v);
            let na: F = a.iter().map(|&v| v * v).fold(F::zero(), |acc, v| acc + v).sqrt();
            let nb: F = b.iter().map(|&v| v * v).fold(F::zero(), |acc, v| acc + v).sqrt();
            let cos = dot / ((na * nb).max(eps));
            F::one() - cos
        }
        DistanceMetric::Manhattan => a
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).abs())
            .fold(F::zero(), |acc, v| acc + v),
    };
    Ok(dist)
}

/// Numerically stable `log(exp(a) + exp(b))`.
fn log_sum_exp<F: Float>(a: F, b: F) -> F {
    if a > b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}

/// Sigmoid function.
fn sigmoid<F: Float + FromPrimitive>(x: F) -> F {
    let one = F::one();
    one / (one + (-x).exp())
}

/// Sample `k` elements from `pool` using a simple LCG without replacement.
fn lcg_sample<T: Copy>(pool: &[T], k: usize, seed: u64) -> Vec<T> {
    let n = pool.len();
    if k >= n {
        return pool.to_vec();
    }
    // LCG: x_{i+1} = (a*x_i + c) mod m
    const A: u64 = 6364136223846793005;
    const C: u64 = 1442695040888963407;

    let mut state = seed;
    let mut indices: Vec<usize> = (0..n).collect();

    for i in 0..k {
        state = state.wrapping_mul(A).wrapping_add(C);
        let j = i + (state as usize) % (n - i);
        indices.swap(i, j);
    }

    indices[..k].iter().map(|&i| pool[i]).collect()
}

/// Create a weight matrix with alternating ±`scale` for deterministic init.
fn alternating_weight_matrix<F: Float + FromPrimitive>(
    rows: usize,
    cols: usize,
    scale: F,
) -> Array2<F> {
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

    fn make_episode() -> MetaEpisode<f64> {
        // 10 samples, 2 classes (alternating), 8-dim embeddings
        let embeddings = Array2::<f64>::from_shape_fn((12, 8), |(i, j)| {
            let class_offset = if i % 2 == 0 { 0.0 } else { 10.0 };
            class_offset + (i * 8 + j) as f64 * 0.01
        });
        let labels: Vec<usize> = (0..12).map(|i| i % 2).collect();
        MetaEpisode::sample(&embeddings, &labels, 2, 2, 2, 42).expect("episode sample")
    }

    #[test]
    fn test_meta_episode_sample() {
        let episode = make_episode();
        assert_eq!(episode.n_way, 2);
        assert_eq!(episode.k_shot, 2);
        assert_eq!(episode.support_embeddings.nrows(), 4); // 2 * 2
        assert_eq!(episode.query_embeddings.nrows(), 4); // 2 * 2
    }

    #[test]
    fn test_prototypical_network_predict() {
        let episode = make_episode();
        let config = PrototypicalConfig {
            distance: DistanceMetric::Euclidean,
        };
        let net = PrototypicalNetwork::new(config);
        let preds = net.predict(&episode).expect("predict");
        assert_eq!(preds.len(), episode.query_embeddings.nrows());
        // All predictions should be 0 or 1
        for &p in &preds {
            assert!(p < episode.n_way);
        }
    }

    #[test]
    fn test_prototypical_network_episode_loss() {
        let episode = make_episode();
        let net = PrototypicalNetwork::new(PrototypicalConfig::default());
        let loss = net.episode_loss(&episode).expect("episode_loss");
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_relation_network_predict() {
        let episode = make_episode();
        let rel_net = RelationNetwork::<f64>::new(8, 16).expect("RelationNetwork::new");
        let preds = rel_net.predict(&episode).expect("predict");
        assert_eq!(preds.len(), episode.query_embeddings.nrows());
    }

    #[test]
    fn test_maml_inner_loop() {
        let config = MAMLConfig {
            inner_lr: 0.01,
            inner_steps: 3,
            ..Default::default()
        };
        let initial = vec![1.0_f64; 4];
        let maml = MAML::new(config, initial.clone()).expect("MAML::new");

        // Simple quadratic loss: L = ||θ - target||² / 2
        let target = [0.5_f64; 4];
        let oracle = |params: &[f64], _phase: &str| -> Result<(f64, Vec<f64>)> {
            let loss: f64 = params
                .iter()
                .zip(target.iter())
                .map(|(&p, &t)| (p - t) * (p - t))
                .sum::<f64>()
                / 2.0;
            let grad: Vec<f64> = params.iter().zip(target.iter()).map(|(&p, &t)| p - t).collect();
            Ok((loss, grad))
        };

        let (adapted, inner_loss) = maml.inner_loop(&initial, oracle).expect("inner_loop");
        assert_eq!(adapted.len(), 4);
        assert!(inner_loss.is_finite());
        // Adapted params should be closer to target
        let adapted_dist: f64 = adapted
            .iter()
            .zip(target.iter())
            .map(|(&a, &t)| (a - t).abs())
            .sum();
        let init_dist: f64 = initial
            .iter()
            .zip(target.iter())
            .map(|(&a, &t)| (a - t).abs())
            .sum();
        assert!(adapted_dist < init_dist);
    }

    #[test]
    fn test_maml_meta_update() {
        let config = MAMLConfig {
            inner_lr: 0.01,
            inner_steps: 2,
            outer_lr: 0.001,
            meta_batch_size: 2,
            ..Default::default()
        };
        let initial = vec![1.0_f64; 4];
        let mut maml = MAML::new(config, initial).expect("MAML::new");

        // Two tasks: each a quadratic loss towards a different target
        let target_a = [0.0_f64; 4];
        let target_b = [2.0_f64; 4];

        let mut task_a = |params: &[f64], _phase: &str| -> Result<(f64, Vec<f64>)> {
            let loss = params.iter().zip(target_a.iter()).map(|(&p, &t)| (p - t).powi(2)).sum::<f64>() / 2.0;
            let grad: Vec<f64> = params.iter().zip(target_a.iter()).map(|(&p, &t)| p - t).collect();
            Ok((loss, grad))
        };
        let mut task_b = |params: &[f64], _phase: &str| -> Result<(f64, Vec<f64>)> {
            let loss = params.iter().zip(target_b.iter()).map(|(&p, &t)| (p - t).powi(2)).sum::<f64>() / 2.0;
            let grad: Vec<f64> = params.iter().zip(target_b.iter()).map(|(&p, &t)| p - t).collect();
            Ok((loss, grad))
        };

        let mut tasks: Vec<&mut dyn FnMut(&[f64], &str) -> Result<(f64, Vec<f64>)>> =
            vec![&mut task_a, &mut task_b];

        // Wrap in a Vec of boxed closures that satisfy the trait bound
        let result = {
            // Directly invoke the public API by constructing a Vec<G>
            let mut closures: Vec<Box<dyn FnMut(&[f64], &str) -> Result<(f64, Vec<f64>)>>> = vec![
                Box::new(|params: &[f64], phase: &str| {
                    let loss = params.iter().zip(target_a.iter()).map(|(&p, &t)| (p - t).powi(2)).sum::<f64>() / 2.0;
                    let grad: Vec<f64> = params.iter().zip(target_a.iter()).map(|(&p, &t)| p - t).collect();
                    Ok((loss, grad))
                }),
                Box::new(|params: &[f64], phase: &str| {
                    let loss = params.iter().zip(target_b.iter()).map(|(&p, &t)| (p - t).powi(2)).sum::<f64>() / 2.0;
                    let grad: Vec<f64> = params.iter().zip(target_b.iter()).map(|(&p, &t)| p - t).collect();
                    Ok((loss, grad))
                }),
            ];
            let _ = tasks; // suppress unused warning
            maml.meta_update(closures.as_mut_slice())
        };
        let result = result.expect("meta_update");
        assert!(result.inner_loss.is_finite());
        assert!(result.outer_loss.is_finite());
        assert_eq!(result.num_tasks, 2);
    }

    #[test]
    fn test_distance_metrics() {
        let a = Array1::<f64>::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::<f64>::from_vec(vec![0.0, 1.0, 0.0]);

        let euclidean = compute_distance(&a, &b, DistanceMetric::Euclidean).expect("euclidean");
        let cosine = compute_distance(&a, &b, DistanceMetric::Cosine).expect("cosine");
        let manhattan = compute_distance(&a, &b, DistanceMetric::Manhattan).expect("manhattan");

        // Orthogonal vectors: Euclidean² = 2, cosine = 1, Manhattan = 2
        assert!((euclidean - 2.0).abs() < 1e-10);
        assert!((cosine - 1.0).abs() < 1e-10);
        assert!((manhattan - 2.0).abs() < 1e-10);
    }
}
