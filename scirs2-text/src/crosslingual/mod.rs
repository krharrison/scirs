//! Cross-lingual NER transfer and language-agnostic feature utilities.
//!
//! Provides zero-shot cross-lingual named entity recognition transfer using
//! embedding projection, Procrustes alignment, and character n-gram features.

use crate::error::{Result, TextError};

// ─── Config ──────────────────────────────────────────────────────────────────

/// High-level cross-lingual pipeline configuration.
#[derive(Debug, Clone)]
pub struct CrossLingualConfig {
    /// BCP-47 source language code.
    pub source_lang: String,
    /// BCP-47 target language code.
    pub target_lang: String,
    /// Number of NER label classes.
    pub n_labels: usize,
    /// Whether to freeze the embedding layer during fine-tuning.
    pub freeze_embeddings: bool,
    /// Label-smoothing ε for cross-entropy.
    pub label_smoothing: f64,
}

impl Default for CrossLingualConfig {
    fn default() -> Self {
        Self {
            source_lang: "en".to_string(),
            target_lang: String::new(),
            n_labels: 9,
            freeze_embeddings: false,
            label_smoothing: 0.1,
        }
    }
}

// ─── NerLabel ────────────────────────────────────────────────────────────────

/// BIO-encoded NER label set (CoNLL-2003 style).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NerLabel {
    /// Outside any named entity.
    O,
    /// Beginning of a Person entity.
    BPerson,
    /// Inside a Person entity.
    IPerson,
    /// Beginning of an Organization entity.
    BOrganization,
    /// Inside an Organization entity.
    IOrganization,
    /// Beginning of a Location entity.
    BLocation,
    /// Inside a Location entity.
    ILocation,
    /// Beginning of a Miscellaneous entity.
    BMisc,
    /// Inside a Miscellaneous entity.
    IMisc,
}

impl NerLabel {
    /// Parse a BIO string such as `"B-PER"` or `"O"` into a `NerLabel`.
    pub fn from_bio_str(s: &str) -> Option<Self> {
        match s {
            "O" => Some(NerLabel::O),
            "B-PER" | "B-PERSON" => Some(NerLabel::BPerson),
            "I-PER" | "I-PERSON" => Some(NerLabel::IPerson),
            "B-ORG" | "B-ORGANIZATION" => Some(NerLabel::BOrganization),
            "I-ORG" | "I-ORGANIZATION" => Some(NerLabel::IOrganization),
            "B-LOC" | "B-LOCATION" => Some(NerLabel::BLocation),
            "I-LOC" | "I-LOCATION" => Some(NerLabel::ILocation),
            "B-MISC" => Some(NerLabel::BMisc),
            "I-MISC" => Some(NerLabel::IMisc),
            _ => None,
        }
    }

    /// Canonical BIO string representation.
    pub fn to_bio_str(self) -> &'static str {
        match self {
            NerLabel::O => "O",
            NerLabel::BPerson => "B-PER",
            NerLabel::IPerson => "I-PER",
            NerLabel::BOrganization => "B-ORG",
            NerLabel::IOrganization => "I-ORG",
            NerLabel::BLocation => "B-LOC",
            NerLabel::ILocation => "I-LOC",
            NerLabel::BMisc => "B-MISC",
            NerLabel::IMisc => "I-MISC",
        }
    }

    /// Zero-based integer label id.
    pub fn label_id(self) -> usize {
        match self {
            NerLabel::O => 0,
            NerLabel::BPerson => 1,
            NerLabel::IPerson => 2,
            NerLabel::BOrganization => 3,
            NerLabel::IOrganization => 4,
            NerLabel::BLocation => 5,
            NerLabel::ILocation => 6,
            NerLabel::BMisc => 7,
            NerLabel::IMisc => 8,
        }
    }

    /// Reconstruct a `NerLabel` from its integer id.
    pub fn from_id(id: usize) -> Option<Self> {
        match id {
            0 => Some(NerLabel::O),
            1 => Some(NerLabel::BPerson),
            2 => Some(NerLabel::IPerson),
            3 => Some(NerLabel::BOrganization),
            4 => Some(NerLabel::IOrganization),
            5 => Some(NerLabel::BLocation),
            6 => Some(NerLabel::ILocation),
            7 => Some(NerLabel::BMisc),
            8 => Some(NerLabel::IMisc),
            _ => None,
        }
    }
}

// ─── CrossLingualNerConfig ────────────────────────────────────────────────────

/// Configuration for the cross-lingual NER model.
#[derive(Debug, Clone)]
pub struct CrossLingualNerConfig {
    /// Number of output label classes.
    pub n_labels: usize,
    /// Projection hidden dimension.
    pub hidden_dim: usize,
    /// SGD learning rate.
    pub lr: f64,
    /// Number of fine-tuning epochs.
    pub n_epochs: usize,
}

impl Default for CrossLingualNerConfig {
    fn default() -> Self {
        Self {
            n_labels: 9,
            hidden_dim: 128,
            lr: 0.01,
            n_epochs: 5,
        }
    }
}

// ─── CrossLingualNer ─────────────────────────────────────────────────────────

/// Two-layer linear NER classifier with a language-neutral projection layer.
///
/// Architecture:
/// ```text
/// embeddings  →  W_proj (hidden_dim × embed_dim) → ReLU → W_out (n_labels × hidden_dim) → logits
/// ```
pub struct CrossLingualNer {
    /// Projection weight matrix: `hidden_dim × embed_dim`.
    pub projection_weights: Vec<Vec<f64>>,
    /// Output weight matrix: `n_labels × hidden_dim`.
    pub output_weights: Vec<Vec<f64>>,
    /// Model configuration.
    pub config: CrossLingualNerConfig,
    /// Input embedding dimension (stored for validation).
    embed_dim: usize,
}

/// Tiny deterministic pseudo-random initialiser (LCG) – avoids external deps.
fn lcg_rand(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // Map to [-0.1, 0.1]
    let bits = (*seed >> 33) as f64 / (u32::MAX as f64);
    (bits - 0.5) * 0.2
}

impl CrossLingualNer {
    /// Construct a new model with random weight initialisation.
    pub fn new(embed_dim: usize, config: CrossLingualNerConfig) -> Self {
        let mut seed: u64 = 0xDEAD_BEEF_CAFE_1234;
        let hidden_dim = config.hidden_dim;
        let n_labels = config.n_labels;

        let projection_weights = (0..hidden_dim)
            .map(|_| (0..embed_dim).map(|_| lcg_rand(&mut seed)).collect())
            .collect();

        let output_weights = (0..n_labels)
            .map(|_| (0..hidden_dim).map(|_| lcg_rand(&mut seed)).collect())
            .collect();

        Self {
            projection_weights,
            output_weights,
            config,
            embed_dim,
        }
    }

    // ── internal helpers ────────────────────────────────────────────────────

    fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    fn softmax(logits: &[f64]) -> Vec<f64> {
        let max_v = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&x| (x - max_v).exp()).collect();
        let sum: f64 = exps.iter().sum();
        if sum == 0.0 {
            exps
        } else {
            exps.iter().map(|&e| e / sum).collect()
        }
    }

    fn matvec(mat: &[Vec<f64>], vec: &[f64]) -> Vec<f64> {
        mat.iter()
            .map(|row| row.iter().zip(vec.iter()).map(|(w, x)| w * x).sum())
            .collect()
    }

    /// Compute hidden representation for a single token embedding.
    fn hidden_vec(&self, embedding: &[f64]) -> Vec<f64> {
        CrossLingualNer::matvec(&self.projection_weights, embedding)
            .into_iter()
            .map(Self::relu)
            .collect()
    }

    // ── public API ──────────────────────────────────────────────────────────

    /// Forward pass: `embeddings` shape `(n_tokens, embed_dim)` → logits `(n_tokens, n_labels)`.
    pub fn forward(&self, embeddings: &[Vec<f64>]) -> Vec<Vec<f64>> {
        embeddings
            .iter()
            .map(|emb| {
                let h = self.hidden_vec(emb);
                CrossLingualNer::matvec(&self.output_weights, &h)
            })
            .collect()
    }

    /// Predict a label for each token (argmax of logits).
    pub fn predict(&self, embeddings: &[Vec<f64>]) -> Vec<NerLabel> {
        self.forward(embeddings)
            .iter()
            .map(|logits| {
                let best = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                NerLabel::from_id(best).unwrap_or(NerLabel::O)
            })
            .collect()
    }

    /// Single training step (numerical gradient on output_weights, SGD update).
    ///
    /// Returns the cross-entropy loss (with label smoothing) for this batch.
    pub fn train_step(&mut self, embeddings: &[Vec<f64>], labels: &[usize]) -> f64 {
        assert_eq!(
            embeddings.len(),
            labels.len(),
            "embeddings and labels must have the same length"
        );
        let n_labels = self.config.n_labels;
        let eps = self.config.lr;
        let smooth = 0.1_f64;
        let delta = 1e-5_f64;

        // ── 1. compute loss at current params ──────────────────────────────
        let loss_at = |model: &Self| -> f64 {
            let mut total = 0.0_f64;
            for (emb, &lbl) in embeddings.iter().zip(labels.iter()) {
                let h = model.hidden_vec(emb);
                let logits = CrossLingualNer::matvec(&model.output_weights, &h);
                let probs = Self::softmax(&logits);
                // label-smoothed CE
                for (k, &p) in probs.iter().enumerate() {
                    let target = if k == lbl {
                        1.0 - smooth + smooth / n_labels as f64
                    } else {
                        smooth / n_labels as f64
                    };
                    total -= target * (p + 1e-15).ln();
                }
            }
            total / embeddings.len() as f64
        };

        let base_loss = loss_at(self);

        // ── 2. numerical gradient on output_weights only ───────────────────
        let n_out_rows = self.output_weights.len();
        let n_out_cols = self.output_weights[0].len();
        let mut grad = vec![vec![0.0_f64; n_out_cols]; n_out_rows];

        for i in 0..n_out_rows {
            for j in 0..n_out_cols {
                self.output_weights[i][j] += delta;
                let perturbed = loss_at(self);
                self.output_weights[i][j] -= delta;
                grad[i][j] = (perturbed - base_loss) / delta;
            }
        }

        // ── 3. SGD update ──────────────────────────────────────────────────
        for i in 0..n_out_rows {
            for j in 0..n_out_cols {
                self.output_weights[i][j] -= eps * grad[i][j];
            }
        }

        base_loss
    }

    /// Zero-shot transfer: fine-tune on source embeddings+labels, then predict on target.
    pub fn transfer(
        &self,
        source_embeddings: &[Vec<f64>],
        source_labels: &[usize],
        target_embeddings: &[Vec<f64>],
    ) -> Vec<NerLabel> {
        // Clone the model so the original is not mutated.
        let mut fine_tuned = CrossLingualNer::new(self.embed_dim, self.config.clone());
        // Copy current weights as starting point.
        fine_tuned.projection_weights = self.projection_weights.clone();
        fine_tuned.output_weights = self.output_weights.clone();

        for _ in 0..fine_tuned.config.n_epochs {
            fine_tuned.train_step(source_embeddings, source_labels);
        }
        fine_tuned.predict(target_embeddings)
    }
}

// ─── Language-agnostic features ──────────────────────────────────────────────

/// Compute hashed character n-gram features for a token.
///
/// Returns a 256-dimensional real-valued vector where each dimension corresponds
/// to a hash bucket.  The FNV-1a hash maps each n-gram into one of 256 buckets
/// and the bucket value is incremented by 1.0, then L2-normalised.
///
/// # Example
/// ```
/// use scirs2_text::crosslingual::compute_character_ngram_features;
/// let feat = compute_character_ngram_features("hello", 3);
/// assert_eq!(feat.len(), 256);
/// ```
pub fn compute_character_ngram_features(token: &str, n: usize) -> Vec<f64> {
    const BUCKETS: usize = 256;
    let mut counts = vec![0.0_f64; BUCKETS];

    // Collect unicode chars so that multi-byte characters are handled correctly.
    let chars: Vec<char> = token.chars().collect();

    if chars.len() < n {
        // Token shorter than n → fall back to single characters.
        for ch in &chars {
            let bucket = fnv1a_char_hash(*ch) % BUCKETS;
            counts[bucket] += 1.0;
        }
    } else {
        for window in chars.windows(n) {
            let h = fnv1a_window_hash(window) % BUCKETS;
            counts[h] += 1.0;
        }
    }

    // L2 normalise.
    let norm: f64 = counts.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 {
        counts.iter_mut().for_each(|x| *x /= norm);
    }
    counts
}

/// FNV-1a hash for a single `char`.
fn fnv1a_char_hash(ch: char) -> usize {
    let mut hash: u32 = 2166136261;
    for byte in (ch as u32).to_le_bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(16777619);
    }
    hash as usize
}

/// FNV-1a hash for a slice of `char`s (an n-gram window).
fn fnv1a_window_hash(chars: &[char]) -> usize {
    let mut hash: u32 = 2166136261;
    for &ch in chars {
        for byte in (ch as u32).to_le_bytes() {
            hash ^= byte as u32;
            hash = hash.wrapping_mul(16777619);
        }
    }
    hash as usize
}

// ─── Embedding alignment (Procrustes) ────────────────────────────────────────

/// Align `source` embeddings into the coordinate frame of `target` via
/// Procrustes analysis (SVD-based, orthogonal transform).
///
/// Both slices must have the same embedding dimension.  The function returns
/// the source embeddings rotated/reflected to best match the target.
///
/// # Errors
/// Returns `TextError::InvalidInput` if dimensions are incompatible or either
/// slice is empty.
pub fn align_embeddings(source: &[Vec<f64>], target: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    if source.is_empty() || target.is_empty() {
        return Err(TextError::InvalidInput(
            "align_embeddings: source and target must be non-empty".into(),
        ));
    }
    let d = source[0].len();
    if target[0].len() != d {
        return Err(TextError::InvalidInput(format!(
            "align_embeddings: embedding dimension mismatch ({} vs {})",
            d,
            target[0].len()
        )));
    }

    // We compute M = S^T * T  (d × d cross-covariance).
    // Then SVD(M) = U Σ V^T and the optimal rotation is R = V U^T.
    // We use a simple Jacobi SVD on the d×d matrix (sufficient for small d).

    let n = source.len().min(target.len());

    // Cross-covariance  M[i][j] = Σ_k  source[k][i] * target[k][j]
    let mut m = vec![vec![0.0_f64; d]; d];
    for k in 0..n {
        for i in 0..d {
            for j in 0..d {
                m[i][j] += source[k][i] * target[k][j];
            }
        }
    }

    // Jacobi one-sided SVD for M → U, Σ, V (we only need V U^T = R).
    let (u_mat, _sigma, v_mat) = jacobi_svd(&m);

    // R = V * U^T
    let r = mat_mul_transpose(&v_mat, &u_mat);

    // Apply rotation to each source embedding.
    let aligned = source.iter().map(|s| matvec_t(&r, s)).collect();

    Ok(aligned)
}

/// Multiply matrix `a` (d×d) by the transpose of matrix `b` (d×d): result = A B^T.
fn mat_mul_transpose(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let d = a.len();
    let mut res = vec![vec![0.0_f64; d]; d];
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                res[i][j] += a[i][k] * b[j][k];
            }
        }
    }
    res
}

/// Matrix–vector product  y = M * x.
fn matvec_t(m: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(w, v)| w * v).sum())
        .collect()
}

/// Jacobi one-sided SVD on a square d×d matrix.
/// Returns (U, singular values, V) such that M ≈ U diag(S) V^T.
///
/// This is a simplified Jacobi algorithm adequate for small dense matrices.
fn jacobi_svd(m: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let d = m.len();
    // Work on A = M^T M (normal equations approach for simplicity).
    // Then eigenvectors of A are V, eigenvalues are σ².
    let mut a = mat_mul_mat_t(m); // A = M M^T (d×d)

    // Build V as eigenvectors of A via Jacobi eigendecomposition.
    let mut v = identity(d);
    let max_iter = 100 * d * d;
    for _ in 0..max_iter {
        // Find the largest off-diagonal element.
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_val = 0.0_f64;
        for i in 0..d {
            for j in (i + 1)..d {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-12 {
            break;
        }
        // Compute rotation angle.
        let theta = if (a[q][q] - a[p][p]).abs() < 1e-12 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * a[p][q]) / (a[q][q] - a[p][p])).atan()
        };
        let c = theta.cos();
        let s = theta.sin();
        // Apply Jacobi rotation to A: A ← J^T A J
        jacobi_rotate(&mut a, p, q, c, s);
        // Accumulate in V.
        jacobi_rotate(&mut v, p, q, c, s);
    }

    // Singular values = sqrt of diagonal of A.
    let sigma: Vec<f64> = (0..d).map(|i| a[i][i].abs().sqrt()).collect();

    // U = M V diag(1/sigma) — only needed for the rotation R = V U^T.
    // For our Procrustes purpose R = V U^T; we use a simplified path:
    // U columns = M v_i / σ_i.
    let mut u = vec![vec![0.0_f64; d]; d];
    for j in 0..d {
        let mv: Vec<f64> = matvec_t(m, &v.iter().map(|row| row[j]).collect::<Vec<_>>());
        let sig = sigma[j].max(1e-12);
        for i in 0..d {
            u[i][j] = mv[i] / sig;
        }
    }

    (u, sigma, v)
}

/// M M^T
fn mat_mul_mat_t(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let d = m.len();
    let mut res = vec![vec![0.0_f64; d]; d];
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                res[i][j] += m[i][k] * m[j][k];
            }
        }
    }
    res
}

fn identity(d: usize) -> Vec<Vec<f64>> {
    let mut eye = vec![vec![0.0_f64; d]; d];
    for i in 0..d {
        eye[i][i] = 1.0;
    }
    eye
}

/// In-place column Jacobi rotation on matrix `a`.
fn jacobi_rotate(a: &mut [Vec<f64>], p: usize, q: usize, c: f64, s: f64) {
    let d = a.len();
    // Row rotation.
    for k in 0..d {
        let ap = a[k][p];
        let aq = a[k][q];
        a[k][p] = c * ap - s * aq;
        a[k][q] = s * ap + c * aq;
    }
    // Column rotation (transpose).
    for k in 0..d {
        let ap = a[p][k];
        let aq = a[q][k];
        a[p][k] = c * ap - s * aq;
        a[q][k] = s * ap + c * aq;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ner_label_from_bio_str() {
        assert_eq!(NerLabel::from_bio_str("B-PER"), Some(NerLabel::BPerson));
        assert_eq!(
            NerLabel::from_bio_str("I-ORG"),
            Some(NerLabel::IOrganization)
        );
        assert_eq!(NerLabel::from_bio_str("O"), Some(NerLabel::O));
        assert_eq!(NerLabel::from_bio_str("B-LOC"), Some(NerLabel::BLocation));
        assert_eq!(NerLabel::from_bio_str("UNKNOWN"), None);
    }

    #[test]
    fn test_ner_label_round_trip() {
        let labels = [
            NerLabel::O,
            NerLabel::BPerson,
            NerLabel::IPerson,
            NerLabel::BOrganization,
            NerLabel::IOrganization,
            NerLabel::BLocation,
            NerLabel::ILocation,
            NerLabel::BMisc,
            NerLabel::IMisc,
        ];
        for lbl in &labels {
            assert_eq!(NerLabel::from_id(lbl.label_id()), Some(*lbl));
        }
    }

    #[test]
    fn test_forward_shape() {
        let embed_dim = 16;
        let config = CrossLingualNerConfig {
            n_labels: 9,
            hidden_dim: 32,
            ..Default::default()
        };
        let model = CrossLingualNer::new(embed_dim, config);
        let embeddings: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1_f64; embed_dim]).collect();
        let logits = model.forward(&embeddings);
        assert_eq!(logits.len(), 5);
        assert_eq!(logits[0].len(), 9);
    }

    #[test]
    fn test_predict_length() {
        let embed_dim = 8;
        let model = CrossLingualNer::new(embed_dim, CrossLingualNerConfig::default());
        let embeddings: Vec<Vec<f64>> = (0..3).map(|i| vec![i as f64 * 0.1; embed_dim]).collect();
        let preds = model.predict(&embeddings);
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_train_step_reduces_loss() {
        let embed_dim = 8;
        let config = CrossLingualNerConfig {
            n_labels: 9,
            hidden_dim: 16,
            lr: 0.05,
            n_epochs: 1,
        };
        let mut model = CrossLingualNer::new(embed_dim, config);
        // Trivial single sample: embedding all-zeros, label 0.
        let embeddings = vec![vec![0.5_f64; embed_dim]];
        let labels = vec![0usize];
        let loss1 = model.train_step(&embeddings, &labels);
        let loss2 = model.train_step(&embeddings, &labels);
        // Loss should not increase after a gradient step.
        assert!(
            loss2 <= loss1 + 1e-6,
            "loss should decrease: {} -> {}",
            loss1,
            loss2
        );
    }

    #[test]
    fn test_character_ngram_features_dim() {
        let feat = compute_character_ngram_features("hello", 3);
        assert_eq!(feat.len(), 256, "feature vector must be 256-dimensional");
    }

    #[test]
    fn test_character_ngram_features_short_token() {
        // Token shorter than n → still produces a 256-dim vector.
        let feat = compute_character_ngram_features("ab", 4);
        assert_eq!(feat.len(), 256);
    }

    #[test]
    fn test_character_ngram_features_normalised() {
        let feat = compute_character_ngram_features("hello world", 2);
        let norm: f64 = feat.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-9,
            "features should be L2-normalised, got norm={}",
            norm
        );
    }

    #[test]
    fn test_align_embeddings_identity() {
        // Aligning a set with itself should return (approximately) the same vectors.
        let vecs: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let aligned = align_embeddings(&vecs, &vecs).expect("alignment failed");
        for (a, b) in aligned.iter().zip(vecs.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-6, "identity alignment failed");
            }
        }
    }

    #[test]
    fn test_transfer_returns_correct_length() {
        let embed_dim = 8;
        let model = CrossLingualNer::new(embed_dim, CrossLingualNerConfig::default());
        let src: Vec<Vec<f64>> = (0..4).map(|_| vec![0.1_f64; embed_dim]).collect();
        let src_labels: Vec<usize> = vec![0, 1, 0, 2];
        let tgt: Vec<Vec<f64>> = (0..6).map(|_| vec![0.2_f64; embed_dim]).collect();
        let preds = model.transfer(&src, &src_labels, &tgt);
        assert_eq!(preds.len(), 6);
    }
}
