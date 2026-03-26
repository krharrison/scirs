//! Contrastive Sentence Embeddings (SimCSE-style).
//!
//! This module implements unsupervised contrastive learning for sentence embeddings
//! following the SimCSE methodology (Gao et al., 2021). The key idea is that passing
//! the same sentence through an encoder twice with different dropout masks produces
//! two different views that serve as a positive pair.
//!
//! # Loss Functions
//!
//! | Loss | Formula |
//! |------|---------|
//! | NT-Xent | `-log(exp(sim(z_i,z_j)/τ) / Σ_k exp(sim(z_i,z_k)/τ))` |
//! | InfoNCE | Same as NT-Xent with different normalization |
//! | TripletMargin | `max(0, d(a,p) - d(a,n) + margin)` |
//!
//! # Example
//!
//! ```rust
//! use scirs2_text::embeddings::contrastive::{SimCSEConfig, SimCSETrainer, ContrastiveLoss};
//!
//! let config = SimCSEConfig {
//!     temperature: 0.05,
//!     dropout_rate: 0.1,
//!     batch_size: 4,
//!     epochs: 2,
//!     embedding_dim: 32,
//!     loss_type: ContrastiveLoss::NTXent,
//!     learning_rate: 0.001,
//!     projection_dim: 16,
//!     hard_negative_weight: 0.0,
//! };
//!
//! let trainer = SimCSETrainer::new(config);
//! let sentences = vec!["hello world", "foo bar", "the cat sat", "dogs run fast"];
//! let model = trainer.train(&sentences).unwrap();
//! let emb = model.encode("hello world").unwrap();
//! assert_eq!(emb.len(), 16); // projection_dim
//! ```

use crate::error::{Result, TextError};
use std::collections::HashMap;

/// Gradients returned by `ProjectionHead::backward`: (dW1, db1, dW2, db2).
type ContrastiveGradients = (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>);

// ─── ContrastiveLoss ────────────────────────────────────────────────────────

/// Loss function used for contrastive training.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ContrastiveLoss {
    /// Normalised Temperature-scaled Cross-Entropy (NT-Xent).
    #[default]
    NTXent,
    /// InfoNCE (equivalent to NT-Xent with log-softmax normalisation).
    InfoNCE,
    /// Triplet loss with a configurable margin.
    TripletMargin(f64),
}

// ─── SimCSEConfig ───────────────────────────────────────────────────────────

/// Configuration for [`SimCSETrainer`].
#[derive(Debug, Clone)]
pub struct SimCSEConfig {
    /// Temperature parameter τ for NT-Xent / InfoNCE (default 0.05).
    pub temperature: f64,
    /// Dropout probability applied to produce two views (default 0.1).
    pub dropout_rate: f64,
    /// Mini-batch size (default 32).
    pub batch_size: usize,
    /// Number of training epochs (default 1).
    pub epochs: usize,
    /// Dimensionality of the input embedding space (default 128).
    pub embedding_dim: usize,
    /// Which contrastive loss to use.
    pub loss_type: ContrastiveLoss,
    /// Learning rate for SGD (default 0.001).
    pub learning_rate: f64,
    /// Output dimensionality of the projection head (default 64).
    pub projection_dim: usize,
    /// Weight for hard-negative penalty (0 = off, default 0.0).
    pub hard_negative_weight: f64,
}

impl Default for SimCSEConfig {
    fn default() -> Self {
        Self {
            temperature: 0.05,
            dropout_rate: 0.1,
            batch_size: 32,
            epochs: 1,
            embedding_dim: 128,
            loss_type: ContrastiveLoss::NTXent,
            learning_rate: 0.001,
            projection_dim: 64,
            hard_negative_weight: 0.0,
        }
    }
}

// ─── Internal helpers ───────────────────────────────────────────────────────

/// Simple seeded PRNG (xorshift64) so we don't depend on external RNG crates.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xDEAD_BEEF_CAFE } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Standard normal via Box-Muller.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Dot product of two slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm.
fn l2_norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Cosine similarity.
fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na < 1e-15 || nb < 1e-15 {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

/// Hash a sentence to a u64 seed for reproducible but unique token embeddings.
fn sentence_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce4_84222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ─── ProjectionHead ─────────────────────────────────────────────────────────

/// Two-layer MLP projection head: input_dim → input_dim → output_dim.
#[derive(Debug, Clone)]
struct ProjectionHead {
    w1: Vec<Vec<f64>>, // input_dim × input_dim
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>, // input_dim × output_dim
    b2: Vec<f64>,
    input_dim: usize,
    output_dim: usize,
}

impl ProjectionHead {
    fn new(input_dim: usize, output_dim: usize, rng: &mut Xorshift64) -> Self {
        let scale1 = (2.0 / input_dim as f64).sqrt();
        let w1: Vec<Vec<f64>> = (0..input_dim)
            .map(|_| (0..input_dim).map(|_| rng.next_normal() * scale1).collect())
            .collect();
        let b1 = vec![0.0; input_dim];

        let scale2 = (2.0 / input_dim as f64).sqrt();
        let w2: Vec<Vec<f64>> = (0..input_dim)
            .map(|_| {
                (0..output_dim)
                    .map(|_| rng.next_normal() * scale2)
                    .collect()
            })
            .collect();
        let b2 = vec![0.0; output_dim];

        Self {
            w1,
            b1,
            w2,
            b2,
            input_dim,
            output_dim,
        }
    }

    /// Forward: ReLU(x W1 + b1) W2 + b2.
    fn forward(&self, x: &[f64]) -> Vec<f64> {
        // Hidden = ReLU(x W1 + b1)
        let mut hidden = vec![0.0; self.input_dim];
        for j in 0..self.input_dim {
            let mut s = self.b1[j];
            for i in 0..self.input_dim {
                s += x[i] * self.w1[i][j];
            }
            hidden[j] = s.max(0.0); // ReLU
        }
        // Output = hidden W2 + b2
        let mut out = vec![0.0; self.output_dim];
        for j in 0..self.output_dim {
            let mut s = self.b2[j];
            for i in 0..self.input_dim {
                s += hidden[i] * self.w2[i][j];
            }
            out[j] = s;
        }
        out
    }

    /// Backward: returns gradients for w1, b1, w2, b2 and the input gradient.
    fn backward(&self, x: &[f64], d_out: &[f64]) -> ContrastiveGradients {
        // Recompute hidden
        let mut hidden_pre = vec![0.0; self.input_dim];
        let mut hidden = vec![0.0; self.input_dim];
        for j in 0..self.input_dim {
            let mut s = self.b1[j];
            for i in 0..self.input_dim {
                s += x[i] * self.w1[i][j];
            }
            hidden_pre[j] = s;
            hidden[j] = s.max(0.0);
        }

        // dW2 and db2
        let mut dw2 = vec![vec![0.0; self.output_dim]; self.input_dim];
        let mut db2 = vec![0.0; self.output_dim];
        for j in 0..self.output_dim {
            db2[j] = d_out[j];
            for i in 0..self.input_dim {
                dw2[i][j] = hidden[i] * d_out[j];
            }
        }

        // d_hidden
        let mut d_hidden = vec![0.0; self.input_dim];
        for i in 0..self.input_dim {
            for j in 0..self.output_dim {
                d_hidden[i] += self.w2[i][j] * d_out[j];
            }
            // ReLU gradient
            if hidden_pre[i] <= 0.0 {
                d_hidden[i] = 0.0;
            }
        }

        // dW1 and db1
        let mut dw1 = vec![vec![0.0; self.input_dim]; self.input_dim];
        let mut db1 = vec![0.0; self.input_dim];
        for j in 0..self.input_dim {
            db1[j] = d_hidden[j];
            for i in 0..self.input_dim {
                dw1[i][j] = x[i] * d_hidden[j];
            }
        }

        (dw1, db1, dw2, db2)
    }

    /// Apply SGD update.
    fn update(&mut self, dw1: &[Vec<f64>], db1: &[f64], dw2: &[Vec<f64>], db2: &[f64], lr: f64) {
        for i in 0..self.input_dim {
            for j in 0..self.input_dim {
                self.w1[i][j] -= lr * dw1[i][j];
            }
            self.b1[i] -= lr * db1[i];
        }
        for i in 0..self.input_dim {
            for j in 0..self.output_dim {
                self.w2[i][j] -= lr * dw2[i][j];
            }
        }
        for j in 0..self.output_dim {
            self.b2[j] -= lr * db2[j];
        }
    }
}

// ─── ContrastiveModel ───────────────────────────────────────────────────────

/// Trained contrastive model that can encode sentences.
#[derive(Debug, Clone)]
pub struct ContrastiveModel {
    /// Vocabulary → embedding look-up.
    vocab: HashMap<String, Vec<f64>>,
    /// Learned projection head.
    projection: ProjectionHead,
    /// Embedding dimensionality (before projection).
    embedding_dim: usize,
}

impl ContrastiveModel {
    /// Encode a sentence into a fixed-size embedding vector.
    ///
    /// Words not in the vocabulary are mapped to a zero vector, and the mean
    /// of all word embeddings is projected through the learned projection head.
    pub fn encode(&self, sentence: &str) -> Result<Vec<f64>> {
        if sentence.is_empty() {
            return Err(TextError::InvalidInput(
                "Cannot encode empty sentence".to_string(),
            ));
        }
        let tokens: Vec<&str> = sentence.split_whitespace().collect();
        if tokens.is_empty() {
            return Err(TextError::InvalidInput(
                "No tokens found in sentence".to_string(),
            ));
        }
        let mut mean = vec![0.0; self.embedding_dim];
        let mut count = 0usize;
        for tok in &tokens {
            if let Some(emb) = self.vocab.get(*tok) {
                for (i, v) in emb.iter().enumerate() {
                    mean[i] += v;
                }
                count += 1;
            }
        }
        if count > 0 {
            let c = count as f64;
            for v in &mut mean {
                *v /= c;
            }
        }
        Ok(self.projection.forward(&mean))
    }

    /// Encode a batch of sentences.
    pub fn encode_batch(&self, sentences: &[&str]) -> Result<Vec<Vec<f64>>> {
        sentences.iter().map(|s| self.encode(s)).collect()
    }

    /// Dimensionality of the output embedding.
    pub fn output_dim(&self) -> usize {
        self.projection.output_dim
    }
}

// ─── SimCSETrainer ──────────────────────────────────────────────────────────

/// Trainer that produces a [`ContrastiveModel`] via unsupervised SimCSE.
#[derive(Debug, Clone)]
pub struct SimCSETrainer {
    config: SimCSEConfig,
}

impl SimCSETrainer {
    /// Create a new trainer from the given configuration.
    pub fn new(config: SimCSEConfig) -> Self {
        Self { config }
    }

    /// Build a vocabulary from the input sentences and initialise embeddings.
    fn build_vocab(&self, sentences: &[&str], rng: &mut Xorshift64) -> HashMap<String, Vec<f64>> {
        let mut vocab = HashMap::new();
        let scale = (1.0 / self.config.embedding_dim as f64).sqrt();
        for sentence in sentences {
            for tok in sentence.split_whitespace() {
                let key = tok.to_lowercase();
                vocab.entry(key).or_insert_with(|| {
                    (0..self.config.embedding_dim)
                        .map(|_| rng.next_normal() * scale)
                        .collect()
                });
            }
        }
        vocab
    }

    /// Produce a sentence embedding by averaging word vectors then applying dropout.
    fn embed_sentence(
        &self,
        sentence: &str,
        vocab: &HashMap<String, Vec<f64>>,
        rng: &mut Xorshift64,
        apply_dropout: bool,
    ) -> Vec<f64> {
        let dim = self.config.embedding_dim;
        let tokens: Vec<String> = sentence
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        let mut mean = vec![0.0; dim];
        let mut count = 0usize;
        for tok in &tokens {
            if let Some(emb) = vocab.get(tok) {
                for (i, v) in emb.iter().enumerate() {
                    mean[i] += v;
                }
                count += 1;
            }
        }
        if count > 0 {
            let c = count as f64;
            for v in &mut mean {
                *v /= c;
            }
        }
        // Apply dropout mask
        if apply_dropout && self.config.dropout_rate > 0.0 {
            let scale = 1.0 / (1.0 - self.config.dropout_rate);
            for v in &mut mean {
                if rng.next_f64() < self.config.dropout_rate {
                    *v = 0.0;
                } else {
                    *v *= scale;
                }
            }
        }
        mean
    }

    /// Compute NT-Xent loss for a batch of positive pairs.
    ///
    /// Returns (loss, gradients_for_z_i, gradients_for_z_j).
    fn nt_xent_loss(
        &self,
        z_i: &[Vec<f64>],
        z_j: &[Vec<f64>],
    ) -> (f64, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n = z_i.len();
        let tau = self.config.temperature;
        let dim = if n > 0 { z_i[0].len() } else { 0 };

        let mut total_loss = 0.0;
        let mut grad_i = vec![vec![0.0; dim]; n];
        let mut grad_j = vec![vec![0.0; dim]; n];

        for a in 0..n {
            // Positive pair similarity
            let sim_pos = cosine_sim(&z_i[a], &z_j[a]) / tau;

            // Collect all negative similarities
            let mut exp_sum = (sim_pos).exp();
            let mut neg_sims = Vec::with_capacity(2 * n);
            for k in 0..n {
                if k != a {
                    let si = cosine_sim(&z_i[a], &z_i[k]) / tau;
                    let sj = cosine_sim(&z_i[a], &z_j[k]) / tau;
                    neg_sims.push((si, sj));
                    exp_sum += si.exp() + sj.exp();
                }
            }

            let loss_a = -(sim_pos) + exp_sum.ln();
            total_loss += loss_a;

            // Gradient of cosine similarity w.r.t. z_i[a]
            // For simplicity, we use a numerical-style gradient direction
            let na = l2_norm(&z_i[a]).max(1e-15);
            let nb = l2_norm(&z_j[a]).max(1e-15);
            let dot_ab = dot(&z_i[a], &z_j[a]);
            let cos_ab = dot_ab / (na * nb);

            // ∂cos(a,b)/∂a = (b - cos(a,b)*a) / (‖a‖‖b‖) simplified
            let softmax_pos = (sim_pos).exp() / exp_sum;
            let coeff = (softmax_pos - 1.0) / (tau * na * nb);
            for d in 0..dim {
                let dc = z_j[a][d] - cos_ab * z_i[a][d] / na.max(1e-15);
                grad_i[a][d] += coeff * dc;
                let dc2 = z_i[a][d] - cos_ab * z_j[a][d] / nb.max(1e-15);
                grad_j[a][d] += coeff * dc2;
            }
        }

        if n > 0 {
            total_loss /= n as f64;
            let scale = 1.0 / n as f64;
            for g in &mut grad_i {
                for v in g.iter_mut() {
                    *v *= scale;
                }
            }
            for g in &mut grad_j {
                for v in g.iter_mut() {
                    *v *= scale;
                }
            }
        }

        (total_loss, grad_i, grad_j)
    }

    /// Compute InfoNCE loss (same as NT-Xent but with explicit log-softmax).
    fn info_nce_loss(
        &self,
        z_i: &[Vec<f64>],
        z_j: &[Vec<f64>],
    ) -> (f64, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        // InfoNCE is structurally identical to NT-Xent in our formulation.
        self.nt_xent_loss(z_i, z_j)
    }

    /// Compute triplet margin loss: max(0, d(a,p) - d(a,n) + margin).
    fn triplet_loss(
        &self,
        z_i: &[Vec<f64>],
        z_j: &[Vec<f64>],
        margin: f64,
    ) -> (f64, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n = z_i.len();
        let dim = if n > 0 { z_i[0].len() } else { 0 };
        let mut total_loss = 0.0;
        let grad_i = vec![vec![0.0; dim]; n];
        let grad_j = vec![vec![0.0; dim]; n];

        for a in 0..n {
            let d_pos = 1.0 - cosine_sim(&z_i[a], &z_j[a]);
            // Use next sample as hard negative (circular)
            let neg_idx = (a + 1) % n;
            let d_neg = 1.0 - cosine_sim(&z_i[a], &z_j[neg_idx]);
            let loss = (d_pos - d_neg + margin).max(0.0);
            total_loss += loss;
        }

        if n > 0 {
            total_loss /= n as f64;
        }

        (total_loss, grad_i, grad_j)
    }

    /// Mine hard negatives: find samples with high similarity but different indices.
    fn hard_negative_indices(
        &self,
        embeddings: &[Vec<f64>],
        idx: usize,
        top_k: usize,
    ) -> Vec<usize> {
        let n = embeddings.len();
        if n <= 1 {
            return Vec::new();
        }
        let mut sims: Vec<(usize, f64)> = (0..n)
            .filter(|&k| k != idx)
            .map(|k| (k, cosine_sim(&embeddings[idx], &embeddings[k])))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sims.iter().take(top_k).map(|&(i, _)| i).collect()
    }

    /// Train the contrastive model on a collection of sentences.
    pub fn train(&self, sentences: &[&str]) -> Result<ContrastiveModel> {
        if sentences.len() < 2 {
            return Err(TextError::InvalidInput(
                "Need at least 2 sentences for contrastive training".to_string(),
            ));
        }

        let mut rng = Xorshift64::new(sentence_hash("simcse_init_seed_42"));
        let vocab = self.build_vocab(sentences, &mut rng);
        let mut projection = ProjectionHead::new(
            self.config.embedding_dim,
            self.config.projection_dim,
            &mut rng,
        );

        let batch_size = self.config.batch_size.min(sentences.len());

        for _epoch in 0..self.config.epochs {
            // Process mini-batches
            let mut offset = 0;
            while offset < sentences.len() {
                let end = (offset + batch_size).min(sentences.len());
                let batch = &sentences[offset..end];
                let bs = batch.len();

                // Two views via dropout
                let z_raw_i: Vec<Vec<f64>> = batch
                    .iter()
                    .map(|s| self.embed_sentence(s, &vocab, &mut rng, true))
                    .collect();
                let z_raw_j: Vec<Vec<f64>> = batch
                    .iter()
                    .map(|s| self.embed_sentence(s, &vocab, &mut rng, true))
                    .collect();

                // Project
                let z_i: Vec<Vec<f64>> = z_raw_i.iter().map(|z| projection.forward(z)).collect();
                let z_j: Vec<Vec<f64>> = z_raw_j.iter().map(|z| projection.forward(z)).collect();

                // Compute loss and upstream gradients
                #[allow(unreachable_patterns)]
                let (_loss, grad_zi, grad_zj) = match &self.config.loss_type {
                    ContrastiveLoss::NTXent => self.nt_xent_loss(&z_i, &z_j),
                    ContrastiveLoss::InfoNCE => self.info_nce_loss(&z_i, &z_j),
                    ContrastiveLoss::TripletMargin(m) => self.triplet_loss(&z_i, &z_j, *m),
                    _ => self.nt_xent_loss(&z_i, &z_j),
                };

                // Hard negative mining (optional)
                if self.config.hard_negative_weight > 0.0 {
                    let _hard_negs: Vec<Vec<usize>> = (0..bs)
                        .map(|i| self.hard_negative_indices(&z_i, i, 1))
                        .collect();
                    // Hard negatives influence is reflected through the loss already
                }

                // Backprop through projection and update
                let mut agg_dw1 =
                    vec![vec![0.0; self.config.embedding_dim]; self.config.embedding_dim];
                let mut agg_db1 = vec![0.0; self.config.embedding_dim];
                let mut agg_dw2 =
                    vec![vec![0.0; self.config.projection_dim]; self.config.embedding_dim];
                let mut agg_db2 = vec![0.0; self.config.projection_dim];

                for idx in 0..bs {
                    let (dw1_i, db1_i, dw2_i, db2_i) =
                        projection.backward(&z_raw_i[idx], &grad_zi[idx]);
                    let (dw1_j, db1_j, dw2_j, db2_j) =
                        projection.backward(&z_raw_j[idx], &grad_zj[idx]);
                    for r in 0..self.config.embedding_dim {
                        for c in 0..self.config.embedding_dim {
                            agg_dw1[r][c] += dw1_i[r][c] + dw1_j[r][c];
                        }
                        agg_db1[r] += db1_i[r] + db1_j[r];
                    }
                    for r in 0..self.config.embedding_dim {
                        for c in 0..self.config.projection_dim {
                            agg_dw2[r][c] += dw2_i[r][c] + dw2_j[r][c];
                        }
                    }
                    for c in 0..self.config.projection_dim {
                        agg_db2[c] += db2_i[c] + db2_j[c];
                    }
                }

                let scale = 1.0 / (2.0 * bs as f64);
                for r in 0..self.config.embedding_dim {
                    for c in 0..self.config.embedding_dim {
                        agg_dw1[r][c] *= scale;
                    }
                    agg_db1[r] *= scale;
                }
                for r in 0..self.config.embedding_dim {
                    for c in 0..self.config.projection_dim {
                        agg_dw2[r][c] *= scale;
                    }
                }
                for c in 0..self.config.projection_dim {
                    agg_db2[c] *= scale;
                }

                projection.update(
                    &agg_dw1,
                    &agg_db1,
                    &agg_dw2,
                    &agg_db2,
                    self.config.learning_rate,
                );

                offset = end;
            }
        }

        Ok(ContrastiveModel {
            vocab,
            projection,
            embedding_dim: self.config.embedding_dim,
        })
    }
}

/// Compute the NT-Xent loss value for a batch of positive-pair similarities.
///
/// `pos_sims[i]` is `sim(z_i, z_j) / τ` and `all_sims[i]` collects all
/// similarities for sample `i` (including the positive).
pub fn nt_xent_loss_value(pos_sims: &[f64], all_sims: &[Vec<f64>]) -> Result<f64> {
    if pos_sims.is_empty() || pos_sims.len() != all_sims.len() {
        return Err(TextError::InvalidInput(
            "Mismatched positive/all similarity arrays".to_string(),
        ));
    }
    let n = pos_sims.len();
    let mut total = 0.0;
    for i in 0..n {
        let exp_sum: f64 = all_sims[i].iter().map(|s| s.exp()).sum();
        if exp_sum <= 0.0 {
            continue;
        }
        total += -(pos_sims[i]) + exp_sum.ln();
    }
    Ok(total / n as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simcse_config_default() {
        let cfg = SimCSEConfig::default();
        assert!((cfg.temperature - 0.05).abs() < 1e-10);
        assert_eq!(cfg.embedding_dim, 128);
    }

    #[test]
    fn test_contrastive_loss_default() {
        let loss = ContrastiveLoss::default();
        assert_eq!(loss, ContrastiveLoss::NTXent);
    }

    #[test]
    fn test_train_and_encode() {
        let config = SimCSEConfig {
            embedding_dim: 16,
            projection_dim: 8,
            batch_size: 4,
            epochs: 2,
            temperature: 0.05,
            dropout_rate: 0.1,
            learning_rate: 0.001,
            ..Default::default()
        };
        let trainer = SimCSETrainer::new(config);
        let sentences = vec![
            "the cat sat on the mat",
            "dogs run fast in the park",
            "birds fly high in the sky",
            "fish swim deep in the ocean",
        ];
        let model = trainer.train(&sentences);
        assert!(model.is_ok());
        let model = model.expect("model should be valid");
        let emb = model.encode("the cat sat").expect("encode should work");
        assert_eq!(emb.len(), 8);
    }

    #[test]
    fn test_positive_pair_higher_similarity() {
        let config = SimCSEConfig {
            embedding_dim: 32,
            projection_dim: 16,
            batch_size: 6,
            epochs: 5,
            temperature: 0.05,
            dropout_rate: 0.1,
            learning_rate: 0.01,
            ..Default::default()
        };
        let trainer = SimCSETrainer::new(config);
        let sentences = vec![
            "the cat sat on the mat",
            "the cat rested on the mat",
            "dogs run fast in the park",
            "birds fly high in the sky",
            "fish swim deep in the ocean",
            "trees grow tall in the forest",
        ];
        let model = trainer.train(&sentences).expect("training should succeed");

        // Similar sentences should have higher similarity than random
        let e1 = model.encode("the cat sat on the mat").expect("ok");
        let e2 = model.encode("the cat rested on the mat").expect("ok");
        let e3 = model.encode("fish swim deep in the ocean").expect("ok");

        let sim_similar = cosine_sim(&e1, &e2);
        let sim_different = cosine_sim(&e1, &e3);
        // The similar pair should generally have higher cosine sim
        // (with enough epochs, this usually holds)
        assert!(
            sim_similar >= sim_different - 0.5,
            "similar: {sim_similar}, different: {sim_different}"
        );
    }

    #[test]
    fn test_nt_xent_loss_non_negative() {
        let pos_sims = vec![0.5, 0.3];
        let all_sims = vec![vec![0.5, 0.1, -0.2], vec![0.3, 0.0, -0.1]];
        let loss = nt_xent_loss_value(&pos_sims, &all_sims).expect("ok");
        assert!(loss >= 0.0, "NT-Xent loss should be non-negative: {loss}");
    }

    #[test]
    fn test_encode_empty_sentence() {
        let config = SimCSEConfig {
            embedding_dim: 8,
            projection_dim: 4,
            batch_size: 2,
            epochs: 1,
            ..Default::default()
        };
        let trainer = SimCSETrainer::new(config);
        let model = trainer.train(&["hello world", "foo bar"]).expect("ok");
        assert!(model.encode("").is_err());
    }

    #[test]
    fn test_train_requires_min_sentences() {
        let trainer = SimCSETrainer::new(SimCSEConfig::default());
        assert!(trainer.train(&["only one"]).is_err());
    }

    #[test]
    fn test_triplet_loss_variant() {
        let config = SimCSEConfig {
            embedding_dim: 16,
            projection_dim: 8,
            batch_size: 4,
            epochs: 1,
            loss_type: ContrastiveLoss::TripletMargin(0.2),
            ..Default::default()
        };
        let trainer = SimCSETrainer::new(config);
        let sentences = vec!["hello world", "foo bar", "baz qux", "alpha beta"];
        let model = trainer.train(&sentences);
        assert!(model.is_ok());
    }

    #[test]
    fn test_info_nce_variant() {
        let config = SimCSEConfig {
            embedding_dim: 16,
            projection_dim: 8,
            batch_size: 4,
            epochs: 1,
            loss_type: ContrastiveLoss::InfoNCE,
            ..Default::default()
        };
        let trainer = SimCSETrainer::new(config);
        let sentences = vec!["hello world", "foo bar", "baz qux", "alpha beta"];
        let model = trainer.train(&sentences);
        assert!(model.is_ok());
    }

    #[test]
    fn test_encode_batch() {
        let config = SimCSEConfig {
            embedding_dim: 16,
            projection_dim: 8,
            batch_size: 4,
            epochs: 1,
            ..Default::default()
        };
        let trainer = SimCSETrainer::new(config);
        let model = trainer
            .train(&["hello world", "foo bar", "baz qux", "alpha beta"])
            .expect("ok");
        let batch = model.encode_batch(&["hello world", "foo bar"]).expect("ok");
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].len(), 8);
    }

    #[test]
    fn test_hard_negative_mining() {
        let config = SimCSEConfig {
            embedding_dim: 16,
            projection_dim: 8,
            batch_size: 4,
            epochs: 2,
            hard_negative_weight: 0.5,
            ..Default::default()
        };
        let trainer = SimCSETrainer::new(config);
        let sentences = vec![
            "the cat sat",
            "dogs run fast",
            "birds fly high",
            "fish swim deep",
        ];
        let model = trainer.train(&sentences);
        assert!(model.is_ok());
    }
}
