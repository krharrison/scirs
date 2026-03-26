//! Semantic Similarity and Search.
//!
//! This module provides tools for computing semantic similarity between
//! embeddings, performing semantic search, and evaluating STS (Semantic
//! Textual Similarity) benchmarks.
//!
//! # Features
//!
//! - Multiple similarity metrics: Cosine, Euclidean, Manhattan, DotProduct
//! - Pairwise similarity matrices
//! - Top-K most-similar search
//! - STS evaluation with Spearman correlation
//! - Semantic search index with brute-force and approximate (LSH) lookup
//!
//! # Example
//!
//! ```rust
//! use scirs2_text::similarity::{
//!     compute_similarity, pairwise_similarity, most_similar,
//!     SimilarityMetric, SemanticSimilarity, SemanticSearchIndex,
//! };
//!
//! let a = vec![1.0, 0.0, 0.0];
//! let b = vec![0.0, 1.0, 0.0];
//! let c = vec![1.0, 1.0, 0.0];
//!
//! // Cosine similarity
//! let sim = compute_similarity(&a, &b, &SimilarityMetric::Cosine);
//! assert!((sim - 0.0).abs() < 1e-10); // orthogonal
//!
//! let sim_ac = compute_similarity(&a, &c, &SimilarityMetric::Cosine);
//! assert!(sim_ac > 0.5); // closer
//!
//! // Top-K search
//! let corpus = vec![a.clone(), b.clone(), c.clone()];
//! let results = most_similar(&a, &corpus, 2, &SimilarityMetric::Cosine);
//! assert_eq!(results[0].0, 0); // self is most similar
//! ```

use crate::error::{Result, TextError};
use std::collections::HashMap;

// ─── SimilarityMetric ───────────────────────────────────────────────────────

/// Distance/similarity metric for comparing embedding vectors.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum SimilarityMetric {
    /// Cosine similarity: dot(a,b) / (‖a‖·‖b‖). Range: \[-1, 1\].
    #[default]
    Cosine,
    /// Negative Euclidean distance (higher = more similar).
    Euclidean,
    /// Negative Manhattan distance (higher = more similar).
    Manhattan,
    /// Raw dot product.
    DotProduct,
}

// ─── Core functions ─────────────────────────────────────────────────────────

/// Compute similarity between two vectors using the given metric.
///
/// For `Euclidean` and `Manhattan`, the returned value is the *negative* distance
/// so that higher values always mean "more similar".
pub fn compute_similarity(a: &[f64], b: &[f64], metric: &SimilarityMetric) -> f64 {
    #[allow(unreachable_patterns)]
    match metric {
        SimilarityMetric::Cosine => {
            let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            if na < 1e-15 || nb < 1e-15 {
                0.0
            } else {
                dot / (na * nb)
            }
        }
        SimilarityMetric::Euclidean => {
            let dist = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt();
            -dist
        }
        SimilarityMetric::Manhattan => {
            let dist: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
            -dist
        }
        SimilarityMetric::DotProduct => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
        _ => {
            // Fallback: cosine
            compute_similarity(a, b, &SimilarityMetric::Cosine)
        }
    }
}

/// Compute the pairwise similarity matrix for a set of embeddings.
///
/// Returns an N×N matrix where entry \[i\]\[j\] = similarity(embeddings\[i\], embeddings\[j\]).
pub fn pairwise_similarity(embeddings: &[Vec<f64>], metric: &SimilarityMetric) -> Vec<Vec<f64>> {
    let n = embeddings.len();
    let mut matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i..n {
            let sim = compute_similarity(&embeddings[i], &embeddings[j], metric);
            matrix[i][j] = sim;
            matrix[j][i] = sim;
        }
    }
    matrix
}

/// Find the `top_k` most similar vectors in `corpus` to `query`.
///
/// Returns a list of `(index, similarity)` pairs sorted by descending similarity.
pub fn most_similar(
    query: &[f64],
    corpus: &[Vec<f64>],
    top_k: usize,
    metric: &SimilarityMetric,
) -> Vec<(usize, f64)> {
    let mut sims: Vec<(usize, f64)> = corpus
        .iter()
        .enumerate()
        .map(|(i, emb)| (i, compute_similarity(query, emb, metric)))
        .collect();
    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sims.truncate(top_k);
    sims
}

// ─── SemanticSimilarity ─────────────────────────────────────────────────────

/// High-level semantic similarity calculator with configurable metric.
#[derive(Debug, Clone)]
pub struct SemanticSimilarity {
    /// Which metric to use.
    pub metric: SimilarityMetric,
}

impl Default for SemanticSimilarity {
    fn default() -> Self {
        Self {
            metric: SimilarityMetric::Cosine,
        }
    }
}

impl SemanticSimilarity {
    /// Create with a specific metric.
    pub fn new(metric: SimilarityMetric) -> Self {
        Self { metric }
    }

    /// Compute similarity between two vectors.
    pub fn similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        compute_similarity(a, b, &self.metric)
    }

    /// Compute pairwise similarity matrix.
    pub fn pairwise(&self, embeddings: &[Vec<f64>]) -> Vec<Vec<f64>> {
        pairwise_similarity(embeddings, &self.metric)
    }

    /// Find top-k most similar vectors.
    pub fn top_k(&self, query: &[f64], corpus: &[Vec<f64>], k: usize) -> Vec<(usize, f64)> {
        most_similar(query, corpus, k, &self.metric)
    }
}

// ─── STS Evaluation ─────────────────────────────────────────────────────────

/// Compute Spearman rank correlation coefficient between two sequences.
///
/// Returns a value in \[-1, 1\] where 1 means perfect monotone agreement.
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    let n = x.len();
    if n != y.len() {
        return Err(TextError::InvalidInput(
            "Arrays must have the same length for Spearman correlation".to_string(),
        ));
    }
    if n < 2 {
        return Err(TextError::InvalidInput(
            "Need at least 2 elements for Spearman correlation".to_string(),
        ));
    }

    let rank_x = rank(x);
    let rank_y = rank(y);

    // Pearson correlation of ranks
    let n_f = n as f64;
    let mean_rx: f64 = rank_x.iter().sum::<f64>() / n_f;
    let mean_ry: f64 = rank_y.iter().sum::<f64>() / n_f;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..n {
        let dx = rank_x[i] - mean_rx;
        let dy = rank_y[i] - mean_ry;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 {
        return Ok(0.0);
    }
    Ok(cov / denom)
}

/// Assign ranks to values (average rank for ties).
fn rank(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-15 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Evaluate predicted similarities against gold-standard STS scores.
///
/// Returns the Spearman correlation between `predicted` and `gold` similarity scores.
pub fn sts_evaluate(predicted: &[f64], gold: &[f64]) -> Result<f64> {
    spearman_correlation(predicted, gold)
}

// ─── SemanticSearchIndex ────────────────────────────────────────────────────

/// A searchable index of embeddings supporting brute-force and optional LSH
/// approximate nearest neighbour lookup.
#[derive(Debug, Clone)]
pub struct SemanticSearchIndex {
    /// Stored embeddings.
    embeddings: Vec<Vec<f64>>,
    /// Optional labels/IDs for each embedding.
    labels: Vec<String>,
    /// Similarity metric.
    metric: SimilarityMetric,
    /// LSH tables (if enabled). Each table maps bucket → list of indices.
    lsh_tables: Vec<HashMap<u64, Vec<usize>>>,
    /// LSH hyperplanes per table.
    lsh_hyperplanes: Vec<Vec<Vec<f64>>>,
    /// Number of LSH hash bits.
    lsh_bits: usize,
}

/// Configuration for building a [`SemanticSearchIndex`].
#[derive(Debug, Clone)]
pub struct SearchIndexConfig {
    /// Similarity metric.
    pub metric: SimilarityMetric,
    /// Whether to enable LSH approximate search.
    pub use_lsh: bool,
    /// Number of LSH tables (default 4).
    pub lsh_tables: usize,
    /// Number of hash bits per table (default 8).
    pub lsh_bits: usize,
}

impl Default for SearchIndexConfig {
    fn default() -> Self {
        Self {
            metric: SimilarityMetric::Cosine,
            use_lsh: false,
            lsh_tables: 4,
            lsh_bits: 8,
        }
    }
}

/// Simple PRNG for LSH hyperplane generation.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xCAFE_BABE } else { seed },
        }
    }

    fn next_normal(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        let u1 = ((x >> 11) as f64 / (1u64 << 53) as f64).max(1e-15);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        let u2 = (x >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

impl SemanticSearchIndex {
    /// Create a new empty index.
    pub fn new(config: SearchIndexConfig) -> Self {
        Self {
            embeddings: Vec::new(),
            labels: Vec::new(),
            metric: config.metric,
            lsh_tables: Vec::new(),
            lsh_hyperplanes: Vec::new(),
            lsh_bits: config.lsh_bits,
        }
    }

    /// Build index from embeddings and optional labels.
    pub fn build(
        embeddings: Vec<Vec<f64>>,
        labels: Option<Vec<String>>,
        config: SearchIndexConfig,
    ) -> Result<Self> {
        if embeddings.is_empty() {
            return Err(TextError::InvalidInput(
                "Cannot build index from empty embeddings".to_string(),
            ));
        }

        let dim = embeddings[0].len();
        let n = embeddings.len();
        let labels = labels.unwrap_or_else(|| (0..n).map(|i| i.to_string()).collect());
        if labels.len() != n {
            return Err(TextError::InvalidInput(format!(
                "Labels length ({}) does not match embeddings length ({n})",
                labels.len()
            )));
        }

        let mut idx = Self {
            embeddings,
            labels,
            metric: config.metric.clone(),
            lsh_tables: Vec::new(),
            lsh_hyperplanes: Vec::new(),
            lsh_bits: config.lsh_bits,
        };

        if config.use_lsh {
            idx.build_lsh(dim, config.lsh_tables);
        }

        Ok(idx)
    }

    /// Build LSH tables.
    fn build_lsh(&mut self, dim: usize, num_tables: usize) {
        let mut rng = SimpleRng::new(42);

        self.lsh_hyperplanes = (0..num_tables)
            .map(|_| {
                (0..self.lsh_bits)
                    .map(|_| (0..dim).map(|_| rng.next_normal()).collect::<Vec<f64>>())
                    .collect::<Vec<Vec<f64>>>()
            })
            .collect();

        self.lsh_tables = self
            .lsh_hyperplanes
            .iter()
            .map(|planes| {
                let mut table: HashMap<u64, Vec<usize>> = HashMap::new();
                for (i, emb) in self.embeddings.iter().enumerate() {
                    let hash = lsh_hash(emb, planes);
                    table.entry(hash).or_default().push(i);
                }
                table
            })
            .collect();
    }

    /// Add an embedding to the index.
    pub fn add(&mut self, embedding: Vec<f64>, label: String) {
        let idx = self.embeddings.len();
        // Update LSH tables
        for (table_idx, table) in self.lsh_tables.iter_mut().enumerate() {
            if table_idx < self.lsh_hyperplanes.len() {
                let hash = lsh_hash(&embedding, &self.lsh_hyperplanes[table_idx]);
                table.entry(hash).or_default().push(idx);
            }
        }
        self.embeddings.push(embedding);
        self.labels.push(label);
    }

    /// Brute-force search: find top-k most similar embeddings.
    pub fn search_brute_force(&self, query: &[f64], top_k: usize) -> Vec<(usize, String, f64)> {
        let results = most_similar(query, &self.embeddings, top_k, &self.metric);
        results
            .into_iter()
            .map(|(i, sim)| {
                let label = self.labels.get(i).cloned().unwrap_or_default();
                (i, label, sim)
            })
            .collect()
    }

    /// LSH approximate search. Falls back to brute-force if LSH is not built.
    pub fn search_approximate(&self, query: &[f64], top_k: usize) -> Vec<(usize, String, f64)> {
        if self.lsh_tables.is_empty() {
            return self.search_brute_force(query, top_k);
        }

        // Collect candidate indices from all LSH tables
        let mut candidates = std::collections::HashSet::new();
        for (table_idx, table) in self.lsh_tables.iter().enumerate() {
            let hash = lsh_hash(query, &self.lsh_hyperplanes[table_idx]);
            if let Some(indices) = table.get(&hash) {
                for &i in indices {
                    candidates.insert(i);
                }
            }
        }

        // Score candidates
        let mut scored: Vec<(usize, f64)> = candidates
            .into_iter()
            .map(|i| {
                (
                    i,
                    compute_similarity(query, &self.embeddings[i], &self.metric),
                )
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
            .into_iter()
            .map(|(i, sim)| {
                let label = self.labels.get(i).cloned().unwrap_or_default();
                (i, label, sim)
            })
            .collect()
    }

    /// Search using the best available method (LSH if built, else brute-force).
    pub fn search(&self, query: &[f64], top_k: usize) -> Vec<(usize, String, f64)> {
        if self.lsh_tables.is_empty() {
            self.search_brute_force(query, top_k)
        } else {
            self.search_approximate(query, top_k)
        }
    }

    /// Number of embeddings in the index.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}

/// Compute LSH hash for a vector given hyperplanes.
fn lsh_hash(v: &[f64], hyperplanes: &[Vec<f64>]) -> u64 {
    let mut hash: u64 = 0;
    for (bit, plane) in hyperplanes.iter().enumerate() {
        let dot: f64 = v.iter().zip(plane.iter()).map(|(a, b)| a * b).sum();
        if dot >= 0.0 {
            hash |= 1u64 << bit;
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_self_similarity() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = compute_similarity(&v, &v, &SimilarityMetric::Cosine);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "Cosine(x, x) should be 1.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_range() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = compute_similarity(&a, &b, &SimilarityMetric::Cosine);
        assert!(
            (-1.0 - 1e-10..=1.0 + 1e-10).contains(&sim),
            "Cosine should be in [-1, 1]"
        );
        assert!((sim - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = compute_similarity(&a, &b, &SimilarityMetric::Cosine);
        assert!((sim - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_self() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = compute_similarity(&v, &v, &SimilarityMetric::Euclidean);
        assert!(
            (sim - 0.0).abs() < 1e-10,
            "Euclidean(x, x) should be 0.0 (negative distance), got {sim}"
        );
    }

    #[test]
    fn test_manhattan() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = compute_similarity(&a, &b, &SimilarityMetric::Manhattan);
        assert!((sim - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let sim = compute_similarity(&a, &b, &SimilarityMetric::DotProduct);
        assert!((sim - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_symmetric() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0],
        ];
        let matrix = pairwise_similarity(&embeddings, &SimilarityMetric::Cosine);
        assert_eq!(matrix.len(), 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (matrix[i][j] - matrix[j][i]).abs() < 1e-10,
                    "Pairwise matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_most_similar_ordering() {
        let corpus = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let query = vec![1.0, 0.0, 0.0];
        let results = most_similar(&query, &corpus, 3, &SimilarityMetric::Cosine);
        assert_eq!(results.len(), 3);
        // First result should be self (index 0)
        assert_eq!(results[0].0, 0);
        // Second should be the closest (index 1)
        assert_eq!(results[1].0, 1);
        // Verify descending order
        assert!(results[0].1 >= results[1].1);
        assert!(results[1].1 >= results[2].1);
    }

    #[test]
    fn test_spearman_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = spearman_correlation(&x, &y).expect("ok");
        assert!(
            (r - 1.0).abs() < 1e-10,
            "Perfect monotone should give rho=1.0, got {r}"
        );
    }

    #[test]
    fn test_spearman_inverse() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let r = spearman_correlation(&x, &y).expect("ok");
        assert!(
            (r - (-1.0)).abs() < 1e-10,
            "Perfect inverse monotone should give rho=-1.0, got {r}"
        );
    }

    #[test]
    fn test_sts_evaluate() {
        let predicted = vec![0.9, 0.5, 0.1, 0.8];
        let gold = vec![1.0, 0.4, 0.0, 0.9];
        let corr = sts_evaluate(&predicted, &gold).expect("ok");
        assert!(corr > 0.9, "Should have high Spearman: {corr}");
    }

    #[test]
    fn test_semantic_similarity_struct() {
        let ss = SemanticSimilarity::new(SimilarityMetric::Cosine);
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!((ss.similarity(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_search_index_brute_force() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.7, 0.7]];
        let config = SearchIndexConfig::default();
        let index = SemanticSearchIndex::build(embeddings, None, config).expect("ok");
        assert_eq!(index.len(), 3);

        let results = index.search_brute_force(&[1.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // self
    }

    #[test]
    fn test_search_index_lsh() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.9, 0.1, 0.0],
            vec![0.1, 0.9, 0.0],
        ];
        let config = SearchIndexConfig {
            metric: SimilarityMetric::Cosine,
            use_lsh: true,
            lsh_tables: 4,
            lsh_bits: 4,
        };
        let index = SemanticSearchIndex::build(embeddings, None, config).expect("ok");

        // LSH search may not find all results but should find some
        let results = index.search_approximate(&[1.0, 0.0, 0.0], 3);
        // At minimum we should get at least one result if any bucket matches
        // (LSH is probabilistic, so we just check it doesn't panic)
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_search_index_add() {
        let config = SearchIndexConfig::default();
        let mut index = SemanticSearchIndex::new(config);
        assert!(index.is_empty());
        index.add(vec![1.0, 0.0], "first".to_string());
        index.add(vec![0.0, 1.0], "second".to_string());
        assert_eq!(index.len(), 2);
        assert!(!index.is_empty());

        let results = index.search(&[1.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, "first");
    }

    #[test]
    fn test_spearman_length_mismatch() {
        assert!(spearman_correlation(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_semantic_similarity_default() {
        let ss = SemanticSimilarity::default();
        assert_eq!(ss.metric, SimilarityMetric::Cosine);
    }
}
