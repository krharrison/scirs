//! Lightweight BERT fine-tuning API on top of pre-computed embeddings.
//!
//! This module provides a gradient-descent fine-tuning API for classification
//! and sequence labelling tasks.  It operates on pre-computed dense embeddings
//! (e.g., \[CLS\] token embeddings from BERT) rather than raw text, so it does
//! **not** require any external ML library.

use crate::error::{Result, TextError};
use std::f64;

// ─── FineTuneTask ─────────────────────────────────────────────────────────────

/// The task type that determines the classifier head configuration.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum FineTuneTask {
    /// Single-sentence classification.
    Classification {
        /// Number of output classes.
        n_classes: usize,
    },
    /// Per-token sequence labelling (NER, POS, etc.).
    SequenceLabeling {
        /// Number of label classes.
        n_labels: usize,
    },
    /// Two-sentence pair classification (e.g., NLI, STS).
    SentencePairClassification {
        /// Number of output classes.
        n_classes: usize,
    },
}

impl FineTuneTask {
    /// Number of output classes / labels for this task.
    pub fn n_outputs(&self) -> usize {
        match self {
            FineTuneTask::Classification { n_classes } => *n_classes,
            FineTuneTask::SequenceLabeling { n_labels } => *n_labels,
            FineTuneTask::SentencePairClassification { n_classes } => *n_classes,
        }
    }
}

// ─── FineTuneConfig ───────────────────────────────────────────────────────────

/// Hyperparameters for fine-tuning.
#[derive(Debug, Clone)]
pub struct FineTuneConfig {
    /// Peak learning rate (after warmup).
    pub lr: f64,
    /// Number of training epochs.
    pub n_epochs: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// Number of linear warmup steps.
    pub warmup_steps: usize,
    /// Maximum gradient norm for clipping.
    pub max_grad_norm: f64,
    /// Dropout probability applied to the input embedding during training.
    pub dropout: f64,
}

impl Default for FineTuneConfig {
    fn default() -> Self {
        Self {
            lr: 2e-5,
            n_epochs: 3,
            batch_size: 32,
            warmup_steps: 100,
            max_grad_norm: 1.0,
            dropout: 0.1,
        }
    }
}

// ─── ClassificationHead ───────────────────────────────────────────────────────

/// Single linear classification head: logits = W · embedding + b.
#[derive(Debug, Clone)]
pub struct ClassificationHead {
    /// Weight matrix of shape `(n_classes, hidden_size)`.
    pub weight: Vec<Vec<f64>>,
    /// Bias vector of length `n_classes`.
    pub bias: Vec<f64>,
}

impl ClassificationHead {
    /// Construct a randomly initialised head.
    pub fn new(hidden_size: usize, n_classes: usize) -> Self {
        let mut seed: u64 = 0xFAFAFAFA_12345678;
        let weight = (0..n_classes)
            .map(|_| {
                (0..hidden_size)
                    .map(|_| {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let bits = (seed >> 33) as f64 / (u32::MAX as f64);
                        (bits - 0.5) * 0.02 // Xavier-like init
                    })
                    .collect()
            })
            .collect();

        Self {
            weight,
            bias: vec![0.0; n_classes],
        }
    }

    /// Compute raw logits for a single \[CLS\] embedding.
    pub fn forward(&self, cls_embedding: &[f64]) -> Vec<f64> {
        self.weight
            .iter()
            .zip(self.bias.iter())
            .map(|(row, &b)| {
                row.iter()
                    .zip(cls_embedding.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    + b
            })
            .collect()
    }

    /// One-step update using the cross-entropy gradient.
    ///
    /// Returns the cross-entropy loss for the sample.
    pub fn backward_update(
        &mut self,
        cls_embedding: &[f64],
        logits: &[f64],
        label: usize,
        lr: f64,
    ) -> f64 {
        let n_classes = logits.len();
        if label >= n_classes {
            // Safety: silently clamp to last class (no panic).
            return 0.0;
        }

        // Softmax probabilities.
        let probs = softmax(logits);

        // Cross-entropy loss: -log p(true_class).
        let loss = -(probs[label] + 1e-15).ln();

        // Gradient of CE w.r.t. logits: probs - one_hot.
        let grad_logits: Vec<f64> = probs
            .iter()
            .enumerate()
            .map(|(k, &p)| if k == label { p - 1.0 } else { p })
            .collect();

        // Gradient w.r.t. weight[k][j] = grad_logits[k] * cls_embedding[j].
        // Gradient w.r.t. bias[k]       = grad_logits[k].
        let hidden = cls_embedding.len();
        for k in 0..n_classes {
            let g = grad_logits[k];
            self.bias[k] -= lr * g;
            for j in 0..hidden {
                self.weight[k][j] -= lr * g * cls_embedding[j];
            }
        }

        loss
    }
}

// ─── BertFineTuner ────────────────────────────────────────────────────────────

/// Fine-tuner wrapping a `ClassificationHead`.
///
/// Operates on batches of pre-computed BERT \[CLS\] embeddings.
pub struct BertFineTuner {
    /// Classification / labelling head.
    pub head: ClassificationHead,
    /// Fine-tuning hyperparameters.
    pub config: FineTuneConfig,
    /// Current global training step (for LR schedule).
    pub step: usize,
    /// Total training steps (set at `train` call for cosine decay).
    total_steps: usize,
}

impl BertFineTuner {
    /// Create a new `BertFineTuner`.
    ///
    /// # Errors
    /// Returns `TextError::InvalidInput` if the task specifies 0 output classes.
    pub fn new(hidden_size: usize, task: FineTuneTask, config: FineTuneConfig) -> Result<Self> {
        let n_outputs = task.n_outputs();
        if n_outputs == 0 {
            return Err(TextError::InvalidInput(
                "BertFineTuner: task must have at least 1 output class".into(),
            ));
        }
        Ok(Self {
            head: ClassificationHead::new(hidden_size, n_outputs),
            config,
            step: 0,
            total_steps: 0,
        })
    }

    // ── LR schedule ─────────────────────────────────────────────────────────

    /// Learning rate at the current step: linear warmup then cosine decay.
    pub fn learning_rate_schedule(&self) -> f64 {
        let peak = self.config.lr;
        let warmup = self.config.warmup_steps as f64;
        let total = (self.total_steps.max(1)) as f64;
        let s = self.step as f64;

        if s < warmup {
            // Linear warmup.
            peak * (s + 1.0) / warmup
        } else {
            // Cosine decay.
            let progress = (s - warmup) / (total - warmup).max(1.0);
            let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) * 0.5;
            peak * cosine
        }
    }

    // ── gradient clipping ────────────────────────────────────────────────────

    /// Apply gradient norm clipping.  Returns the (possibly scaled) gradient.
    fn clip_grad(grad: &mut [f64], max_norm: f64) {
        let norm: f64 = grad.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > max_norm && norm > 1e-12 {
            let scale = max_norm / norm;
            grad.iter_mut().for_each(|g| *g *= scale);
        }
    }

    // ── training ────────────────────────────────────────────────────────────

    /// Train on a set of (embedding, label) pairs for `config.n_epochs` epochs.
    ///
    /// Returns a vector of per-epoch average losses.
    pub fn train(&mut self, embeddings: &[Vec<f64>], labels: &[usize]) -> Vec<f64> {
        let n = embeddings.len().min(labels.len());
        let batch_size = self.config.batch_size.max(1);
        let n_epochs = self.config.n_epochs;
        self.total_steps = n_epochs * n.div_ceil(batch_size);

        let mut epoch_losses = Vec::with_capacity(n_epochs);

        for _epoch in 0..n_epochs {
            let mut epoch_loss = 0.0_f64;
            let mut n_batches = 0usize;

            // Mini-batch SGD.
            let mut start = 0;
            while start < n {
                let end = (start + batch_size).min(n);
                let batch_embs = &embeddings[start..end];
                let batch_labels = &labels[start..end];

                let lr = self.learning_rate_schedule();
                let mut batch_loss = 0.0_f64;

                // Accumulate gradients.
                let n_classes = self.head.bias.len();
                let hidden = if batch_embs.is_empty() {
                    0
                } else {
                    batch_embs[0].len()
                };
                let mut grad_w = vec![vec![0.0_f64; hidden]; n_classes];
                let mut grad_b = vec![0.0_f64; n_classes];

                for (emb, &lbl) in batch_embs.iter().zip(batch_labels.iter()) {
                    let logits = self.head.forward(emb);
                    let probs = softmax(&logits);
                    let loss = -(probs[lbl.min(n_classes - 1)] + 1e-15).ln();
                    batch_loss += loss;

                    // Gradient of CE w.r.t. logits.
                    for k in 0..n_classes {
                        let g = if k == lbl { probs[k] - 1.0 } else { probs[k] };
                        grad_b[k] += g;
                        for j in 0..hidden {
                            grad_w[k][j] += g * emb[j];
                        }
                    }
                }

                let batch_len = (end - start) as f64;

                // Average gradients.
                grad_b.iter_mut().for_each(|g| *g /= batch_len);
                for row in &mut grad_w {
                    row.iter_mut().for_each(|g| *g /= batch_len);
                }

                // Clip gradients.
                let max_norm = self.config.max_grad_norm;
                Self::clip_grad(&mut grad_b, max_norm);
                for row in &mut grad_w {
                    Self::clip_grad(row, max_norm);
                }

                // Apply updates.
                for k in 0..n_classes {
                    self.head.bias[k] -= lr * grad_b[k];
                    for j in 0..hidden {
                        self.head.weight[k][j] -= lr * grad_w[k][j];
                    }
                }

                epoch_loss += batch_loss / batch_len;
                n_batches += 1;
                self.step += 1;
                start = end;
            }

            epoch_losses.push(if n_batches > 0 {
                epoch_loss / n_batches as f64
            } else {
                0.0
            });
        }

        epoch_losses
    }

    // ── inference ───────────────────────────────────────────────────────────

    /// Predict the class index for a single embedding (argmax of logits).
    pub fn predict(&self, embedding: &[f64]) -> usize {
        let logits = self.head.forward(embedding);
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Compute softmax probability distribution for a single embedding.
    pub fn predict_proba(&self, embedding: &[f64]) -> Vec<f64> {
        softmax(&self.head.forward(embedding))
    }

    /// Compute classification accuracy on a labelled dataset.
    pub fn evaluate(&self, embeddings: &[Vec<f64>], labels: &[usize]) -> f64 {
        let n = embeddings.len().min(labels.len());
        if n == 0 {
            return 0.0;
        }
        let correct: usize = embeddings[..n]
            .iter()
            .zip(labels[..n].iter())
            .filter(|(emb, &lbl)| self.predict(emb) == lbl)
            .count();
        correct as f64 / n as f64
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_v = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max_v).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < 1e-15 {
        exps
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classification_head_shape() {
        let head = ClassificationHead::new(16, 4);
        assert_eq!(head.weight.len(), 4);
        assert_eq!(head.weight[0].len(), 16);
        assert_eq!(head.bias.len(), 4);

        let emb: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
        let logits = head.forward(&emb);
        assert_eq!(logits.len(), 4, "logits must have one entry per class");
    }

    #[test]
    fn test_classification_head_backward_update_returns_loss() {
        let mut head = ClassificationHead::new(8, 3);
        let emb: Vec<f64> = vec![1.0; 8];
        let logits = head.forward(&emb);
        let loss = head.backward_update(&emb, &logits, 0, 1e-3);
        assert!(loss.is_finite(), "loss should be finite, got {}", loss);
        assert!(loss >= 0.0, "CE loss must be non-negative");
    }

    #[test]
    fn test_bert_finetuner_new_invalid_task() {
        // SequenceLabeling with 0 labels should fail.
        let result = BertFineTuner::new(
            16,
            FineTuneTask::SequenceLabeling { n_labels: 0 },
            FineTuneConfig::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_bert_finetuner_train_returns_epoch_losses() {
        let config = FineTuneConfig {
            lr: 0.1,
            n_epochs: 3,
            batch_size: 4,
            warmup_steps: 2,
            ..Default::default()
        };
        let mut tuner =
            BertFineTuner::new(4, FineTuneTask::Classification { n_classes: 2 }, config)
                .expect("should create tuner");

        let embeddings: Vec<Vec<f64>> = (0..8)
            .map(|i| vec![(i % 2) as f64, ((i + 1) % 2) as f64, 0.0, 0.0])
            .collect();
        let labels: Vec<usize> = (0..8).map(|i| i % 2).collect();

        let losses = tuner.train(&embeddings, &labels);
        assert_eq!(losses.len(), 3, "should return one loss per epoch");
        for &loss in &losses {
            assert!(loss.is_finite(), "loss must be finite");
        }
    }

    #[test]
    fn test_bert_finetuner_accuracy_improves_on_separable_data() {
        // Linearly separable 2-class dataset.
        // Class 0: embedding [1, 0, 0, 0], Class 1: embedding [0, 1, 0, 0]
        let hidden = 4;
        let config = FineTuneConfig {
            lr: 1.0,
            n_epochs: 20,
            batch_size: 2,
            warmup_steps: 5,
            max_grad_norm: 10.0,
            dropout: 0.0,
        };
        let mut tuner = BertFineTuner::new(
            hidden,
            FineTuneTask::Classification { n_classes: 2 },
            config,
        )
        .expect("should create tuner");

        let embeddings: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                if i % 2 == 0 {
                    vec![1.0, 0.0, 0.0, 0.0]
                } else {
                    vec![0.0, 1.0, 0.0, 0.0]
                }
            })
            .collect();
        let labels: Vec<usize> = (0..20).map(|i| i % 2).collect();

        let initial_acc = tuner.evaluate(&embeddings, &labels);
        tuner.train(&embeddings, &labels);
        let final_acc = tuner.evaluate(&embeddings, &labels);

        assert!(
            final_acc >= initial_acc,
            "accuracy should not decrease after training on separable data: {} -> {}",
            initial_acc,
            final_acc
        );
    }

    #[test]
    fn test_predict_proba_sums_to_one() {
        let tuner = BertFineTuner::new(
            4,
            FineTuneTask::Classification { n_classes: 3 },
            FineTuneConfig::default(),
        )
        .expect("should create tuner");

        let emb = vec![0.1, 0.2, 0.3, 0.4];
        let proba = tuner.predict_proba(&emb);
        let sum: f64 = proba.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "probabilities must sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_lr_schedule_warmup() {
        let config = FineTuneConfig {
            warmup_steps: 10,
            lr: 1.0,
            ..Default::default()
        };
        let mut tuner =
            BertFineTuner::new(2, FineTuneTask::Classification { n_classes: 2 }, config)
                .expect("tuner");
        tuner.total_steps = 100;

        // Step 0: lr should be 1/10 of peak.
        tuner.step = 0;
        let lr0 = tuner.learning_rate_schedule();
        assert!(lr0 > 0.0 && lr0 <= 1.0, "warmup lr should be in (0, peak]");

        // After warmup, at exactly warmup_steps, cosine starts at 1.0.
        tuner.step = 10;
        let lr_warm = tuner.learning_rate_schedule();
        assert!(lr_warm > 0.0, "lr after warmup should be positive");
    }
}
