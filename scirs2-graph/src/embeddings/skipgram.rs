//! Skip-gram embedding with negative sampling
//!
//! This module exposes a lightweight, functional API for training skip-gram
//! word2vec-style embeddings from graph random walks.  It is designed to be
//! used directly with the outputs of `random_walk::biased_random_walk`,
//! `random_walk::generate_walks`, and `random_walk::deepwalk_walks`.
//!
//! # Algorithm
//!
//! For each node occurrence in a walk we maximise the log-probability of its
//! context nodes (within a sliding window) and minimise the log-probability
//! of `k` negatively sampled nodes:
//!
//! ```text
//! L = log σ(eₘ · eₙ) + Σᵢ E[log σ(-eₘ · eᵢ)]
//! ```
//!
//! Negative sampling uses the unigram distribution raised to the 3/4 power
//! (Mikolov et al. 2013).
//!
//! # References
//! - Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation
//!   of word representations in vector space. ICLR 2013.
//! - Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). DeepWalk. KDD 2014.
//! - Grover, A. & Leskovec, J. (2016). node2vec. KDD 2016.

use scirs2_core::random::Rng;

// ─────────────────────────────────────────────────────────────────────────────
// SkipGramEmbedding
// ─────────────────────────────────────────────────────────────────────────────

/// Node embeddings trained with the skip-gram + negative sampling objective.
///
/// After calling [`train`], `embeddings[v]` gives the `dim`-dimensional vector
/// for node `v`.  The complementary `context_embeddings` are stored internally
/// and can be accessed via [`get_context_embedding`] for research purposes, but
/// in most applications only `embeddings` is needed.
#[derive(Debug, Clone)]
pub struct SkipGramEmbedding {
    /// Embedding dimension
    pub dim: usize,
    /// Source embedding matrix: `embeddings[v]` is the embedding of node `v`
    pub embeddings: Vec<Vec<f64>>,
    /// Context (output) embedding matrix
    pub context_embeddings: Vec<Vec<f64>>,
}

impl SkipGramEmbedding {
    /// Get the embedding vector for node `node`.
    ///
    /// Returns an empty slice if `node` is out of range.
    pub fn get_embedding(&self, node: usize) -> &[f64] {
        self.embeddings.get(node).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get the context embedding vector for node `node`.
    pub fn get_context_embedding(&self, node: usize) -> &[f64] {
        self.context_embeddings.get(node).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Number of nodes in the embedding.
    pub fn n_nodes(&self) -> usize {
        self.embeddings.len()
    }

    /// Cosine similarity between the embeddings of nodes `a` and `b`.
    /// Returns `0.0` if either node is out of range or has a zero-norm vector.
    pub fn cosine_similarity(&self, a: usize, b: usize) -> f64 {
        let ea = self.get_embedding(a);
        let eb = self.get_embedding(b);
        if ea.is_empty() || eb.is_empty() {
            return 0.0;
        }
        let dot: f64 = ea.iter().zip(eb.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = ea.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = eb.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a <= 0.0 || norm_b <= 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// train
// ─────────────────────────────────────────────────────────────────────────────

/// Train a skip-gram embedding from a corpus of random walks.
///
/// # Arguments
/// * `walks`   – corpus of random walks (each walk is a slice of node indices)
/// * `n_nodes` – total number of nodes in the graph
/// * `dim`     – embedding dimension
/// * `window`  – skip-gram context window size
/// * `epochs`  – number of passes over the walk corpus
/// * `lr`      – initial learning rate (linearly decayed to `lr * 0.0001`)
///
/// # Returns
/// A `SkipGramEmbedding` with trained vectors.  All node indices must be
/// `< n_nodes`; out-of-range indices in a walk are silently skipped.
pub fn train(
    walks: &[Vec<usize>],
    n_nodes: usize,
    dim: usize,
    window: usize,
    epochs: usize,
    lr: f64,
) -> SkipGramEmbedding {
    let mut rng = scirs2_core::random::rng();

    // ── Initialise embeddings ──────────────────────────────────────────────
    let mut emb: Vec<Vec<f64>> = (0..n_nodes)
        .map(|_| {
            (0..dim)
                .map(|_| (rng.random::<f64>() - 0.5) / dim as f64)
                .collect()
        })
        .collect();

    let mut ctx_emb: Vec<Vec<f64>> = vec![vec![0.0f64; dim]; n_nodes];

    // ── Build unigram noise distribution (freq^0.75) ───────────────────────
    let mut freq = vec![0usize; n_nodes];
    for walk in walks {
        for &v in walk {
            if v < n_nodes {
                freq[v] += 1;
            }
        }
    }
    let noise_dist = build_noise_distribution(&freq);

    // ── Training loop ──────────────────────────────────────────────────────
    let total_walks = walks.len() * epochs;
    let neg_k = 5usize; // 5 negative samples per positive

    for epoch in 0..epochs {
        for (walk_idx, walk) in walks.iter().enumerate() {
            // Linear learning rate decay
            let step = epoch * walks.len() + walk_idx;
            let progress = step as f64 / total_walks.max(1) as f64;
            let current_lr = (lr * (1.0 - progress)).max(lr * 0.0001);

            for (pos, &center) in walk.iter().enumerate() {
                if center >= n_nodes {
                    continue;
                }

                // Iterate over context window
                let win_start = pos.saturating_sub(window);
                let win_end = (pos + window + 1).min(walk.len());

                for ctx_pos in win_start..win_end {
                    if ctx_pos == pos {
                        continue;
                    }
                    let ctx = walk[ctx_pos];
                    if ctx >= n_nodes {
                        continue;
                    }

                    // ── Positive sample update ─────────────────────────────
                    let dot = dot_product(&emb[center], &ctx_emb[ctx], dim);
                    let sig = sigmoid(dot);
                    let g_pos = current_lr * (1.0 - sig);

                    // Accumulate gradient for center (applied after negatives)
                    let mut grad_center = vec![0.0f64; dim];
                    for d in 0..dim {
                        grad_center[d] += g_pos * ctx_emb[ctx][d];
                        ctx_emb[ctx][d] += g_pos * emb[center][d];
                    }

                    // ── Negative samples ───────────────────────────────────
                    for _ in 0..neg_k {
                        let neg = sample_noise(&noise_dist, &mut rng, n_nodes);
                        if neg == center || neg == ctx {
                            continue;
                        }
                        let dot_neg = dot_product(&emb[center], &ctx_emb[neg], dim);
                        let sig_neg = sigmoid(dot_neg);
                        let g_neg = current_lr * (-sig_neg);

                        for d in 0..dim {
                            grad_center[d] += g_neg * ctx_emb[neg][d];
                            ctx_emb[neg][d] += g_neg * emb[center][d];
                        }
                    }

                    // Apply accumulated gradient to center embedding
                    for d in 0..dim {
                        emb[center][d] += grad_center[d];
                    }
                }
            }
        }
    }

    SkipGramEmbedding {
        dim,
        embeddings: emb,
        context_embeddings: ctx_emb,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build an alias table for the unigram^(3/4) noise distribution.
/// Returns `(prob, alias)` vectors for Vose's method.
fn build_noise_distribution(freq: &[usize]) -> (Vec<f64>, Vec<usize>) {
    let n = freq.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let powered: Vec<f64> = freq.iter().map(|&f| (f as f64).powf(0.75) + 1e-10).collect();
    let total: f64 = powered.iter().sum();

    // Vose's alias method
    let mut prob = vec![0.0f64; n];
    let mut alias = vec![0usize; n];
    let mut work: Vec<f64> = powered.iter().map(|p| p / total * n as f64).collect();

    let mut small: Vec<usize> = Vec::new();
    let mut large: Vec<usize> = Vec::new();

    for (i, &w) in work.iter().enumerate() {
        if w < 1.0 {
            small.push(i);
        } else {
            large.push(i);
        }
    }

    while !small.is_empty() && !large.is_empty() {
        let s = small.pop().unwrap_or(0);
        let l = large.pop().unwrap_or(0);
        prob[s] = work[s];
        alias[s] = l;
        work[l] = (work[l] + work[s]) - 1.0;
        if work[l] < 1.0 {
            small.push(l);
        } else {
            large.push(l);
        }
    }
    for &i in large.iter().chain(small.iter()) {
        prob[i] = 1.0;
    }

    (prob, alias)
}

/// Sample one index from the noise distribution using O(1) alias method.
fn sample_noise(
    noise: &(Vec<f64>, Vec<usize>),
    rng: &mut impl Rng,
    n_nodes: usize,
) -> usize {
    let (prob, alias) = noise;
    if prob.is_empty() {
        return rng.random_range(0..n_nodes.max(1));
    }
    let n = prob.len();
    let i = rng.random_range(0..n);
    if rng.random::<f64>() < prob[i] {
        i
    } else {
        alias[i]
    }
}

/// Dot product of the first `dim` elements of two slices.
#[inline]
fn dot_product(a: &[f64], b: &[f64], dim: usize) -> f64 {
    a.iter().take(dim).zip(b.iter().take(dim)).map(|(x, y)| x * y).sum()
}

/// Numerically stable sigmoid function.
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x > 6.0 {
        1.0 - 1e-10
    } else if x < -6.0 {
        1e-10
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_walks() -> Vec<Vec<usize>> {
        vec![
            vec![0, 1, 2, 1, 0],
            vec![1, 2, 3, 2, 1],
            vec![0, 1, 3, 2, 0],
            vec![2, 3, 0, 1, 2],
        ]
    }

    #[test]
    fn test_train_produces_embeddings() {
        let walks = make_walks();
        let emb = train(&walks, 4, 8, 2, 2, 0.025);
        assert_eq!(emb.n_nodes(), 4);
        assert_eq!(emb.dim, 8);
    }

    #[test]
    fn test_get_embedding_length() {
        let walks = make_walks();
        let emb = train(&walks, 4, 16, 2, 1, 0.025);
        for v in 0..4 {
            assert_eq!(
                emb.get_embedding(v).len(),
                16,
                "node {v} embedding should have length 16"
            );
        }
    }

    #[test]
    fn test_get_embedding_out_of_range() {
        let walks = make_walks();
        let emb = train(&walks, 4, 8, 2, 1, 0.025);
        let e = emb.get_embedding(99);
        assert!(e.is_empty(), "out-of-range node should return empty slice");
    }

    #[test]
    fn test_cosine_similarity_self() {
        let walks = make_walks();
        let emb = train(&walks, 4, 8, 2, 2, 0.025);
        let sim = emb.cosine_similarity(0, 0);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "cosine similarity with self should be ~1.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_range() {
        let walks = make_walks();
        let emb = train(&walks, 4, 8, 2, 2, 0.025);
        for a in 0..4 {
            for b in 0..4 {
                let sim = emb.cosine_similarity(a, b);
                assert!(
                    (-1.0 - 1e-6..=1.0 + 1e-6).contains(&sim),
                    "cosine similarity should be in [-1,1], got {sim}"
                );
            }
        }
    }

    #[test]
    fn test_train_empty_walks() {
        let emb = train(&[], 4, 8, 2, 1, 0.025);
        assert_eq!(emb.n_nodes(), 4);
        // Embeddings should be initialised (random, non-zero)
        let e = emb.get_embedding(0);
        assert_eq!(e.len(), 8);
    }

    #[test]
    fn test_noise_distribution() {
        let freq = vec![10usize, 5, 2, 1];
        let (prob, alias) = build_noise_distribution(&freq);
        assert_eq!(prob.len(), 4);
        assert_eq!(alias.len(), 4);
        for &p in &prob {
            assert!((0.0..=1.0 + 1e-9).contains(&p), "prob should be in [0,1]");
        }
    }
}
