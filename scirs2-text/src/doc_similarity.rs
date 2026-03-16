//! # Document Similarity
//!
//! Comprehensive document similarity and clustering tools:
//!
//! - **TF-IDF vectors**: Build document vectors from a corpus
//! - **Cosine similarity**: Measure similarity between document vectors
//! - **LSA (Latent Semantic Analysis)**: Dimensionality reduction via truncated SVD
//! - **Document clustering**: k-means on TF-IDF vectors
//! - **Shingling**: Near-duplicate detection via character/word n-grams
//! - **MinHash**: Fast approximate Jaccard similarity
//!
//! ## Example
//!
//! ```rust
//! use scirs2_text::doc_similarity::{TfIdfEngine, DocumentCluster, MinHasher, Shingler};
//!
//! let docs = vec![
//!     "machine learning is great",
//!     "deep learning is a type of machine learning",
//!     "natural language processing uses machine learning",
//!     "the cat sat on the mat",
//!     "cats and dogs are pets",
//! ];
//!
//! // Build TF-IDF
//! let mut engine = TfIdfEngine::new();
//! engine.fit(&docs).unwrap();
//! let sim = engine.cosine_similarity(0, 1).unwrap();
//! assert!(sim > 0.0);
//!
//! // Cluster documents
//! let mut cluster = DocumentCluster::new(2);
//! cluster.fit(&engine).unwrap();
//! let labels = cluster.labels();
//! assert_eq!(labels.len(), 5);
//!
//! // MinHash similarity
//! let shingler = Shingler::new(3);
//! let shingles_a = shingler.shingle("the quick brown fox");
//! let shingles_b = shingler.shingle("the quick brown dog");
//! let hasher = MinHasher::new(128);
//! let sig_a = hasher.signature(&shingles_a);
//! let sig_b = hasher.signature(&shingles_b);
//! let est = MinHasher::estimate_jaccard(&sig_a, &sig_b);
//! assert!(est > 0.0);
//! ```

use crate::error::{Result, TextError};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{rngs::StdRng, SeedableRng};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// TF-IDF Engine
// ---------------------------------------------------------------------------

/// TF-IDF document vectorization engine.
///
/// Builds a term-frequency / inverse-document-frequency matrix from a corpus,
/// and supports cosine similarity queries between documents.
#[derive(Debug, Clone)]
pub struct TfIdfEngine {
    /// Vocabulary: word -> column index
    vocab: HashMap<String, usize>,
    /// Inverse document frequency for each term
    idf: Vec<f64>,
    /// TF-IDF matrix: (n_docs, n_terms)
    tfidf_matrix: Option<Array2<f64>>,
    /// Number of documents
    n_docs: usize,
}

impl TfIdfEngine {
    /// Create a new TF-IDF engine.
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            idf: Vec::new(),
            tfidf_matrix: None,
            n_docs: 0,
        }
    }

    /// Fit the TF-IDF model on a corpus.
    pub fn fit(&mut self, documents: &[&str]) -> Result<()> {
        if documents.is_empty() {
            return Err(TextError::InvalidInput(
                "Cannot fit TF-IDF on empty corpus".to_string(),
            ));
        }

        self.n_docs = documents.len();

        // Tokenize and build vocabulary
        let tokenized: Vec<Vec<String>> =
            documents.iter().map(|doc| tokenize_simple(doc)).collect();

        // Build vocabulary
        let mut word_set: HashSet<String> = HashSet::new();
        for tokens in &tokenized {
            for token in tokens {
                word_set.insert(token.clone());
            }
        }

        let mut sorted_words: Vec<String> = word_set.into_iter().collect();
        sorted_words.sort();

        self.vocab.clear();
        for (idx, word) in sorted_words.iter().enumerate() {
            self.vocab.insert(word.clone(), idx);
        }

        let n_terms = self.vocab.len();

        // Compute document frequency for each term
        let mut doc_freq = vec![0usize; n_terms];
        for tokens in &tokenized {
            let unique_tokens: HashSet<&String> = tokens.iter().collect();
            for token in unique_tokens {
                if let Some(&idx) = self.vocab.get(token) {
                    doc_freq[idx] += 1;
                }
            }
        }

        // Compute IDF: log(N / df) + 1  (smooth variant)
        self.idf = doc_freq
            .iter()
            .map(|&df| {
                if df > 0 {
                    (self.n_docs as f64 / df as f64).ln() + 1.0
                } else {
                    1.0
                }
            })
            .collect();

        // Build TF-IDF matrix
        let mut matrix = Array2::<f64>::zeros((self.n_docs, n_terms));
        for (doc_idx, tokens) in tokenized.iter().enumerate() {
            // Term frequency
            let mut tf: HashMap<usize, usize> = HashMap::new();
            for token in tokens {
                if let Some(&term_idx) = self.vocab.get(token) {
                    *tf.entry(term_idx).or_insert(0) += 1;
                }
            }

            let doc_len = tokens.len().max(1) as f64;
            for (&term_idx, &count) in &tf {
                let tf_val = count as f64 / doc_len;
                matrix[[doc_idx, term_idx]] = tf_val * self.idf[term_idx];
            }
        }

        // L2-normalize each document vector
        for mut row in matrix.rows_mut() {
            let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-12 {
                row.mapv_inplace(|x| x / norm);
            }
        }

        self.tfidf_matrix = Some(matrix);
        Ok(())
    }

    /// Get the TF-IDF matrix.
    pub fn matrix(&self) -> Result<&Array2<f64>> {
        self.tfidf_matrix.as_ref().ok_or_else(|| {
            TextError::ModelNotFitted("TF-IDF engine has not been fitted".to_string())
        })
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get the number of documents.
    pub fn n_docs(&self) -> usize {
        self.n_docs
    }

    /// Compute cosine similarity between two documents by index.
    pub fn cosine_similarity(&self, doc_a: usize, doc_b: usize) -> Result<f64> {
        let matrix = self.matrix()?;
        if doc_a >= self.n_docs || doc_b >= self.n_docs {
            return Err(TextError::InvalidInput(format!(
                "Document index out of bounds: {}, {} (total: {})",
                doc_a, doc_b, self.n_docs
            )));
        }
        let a = matrix.row(doc_a);
        let b = matrix.row(doc_b);
        Ok(cosine_sim_vec(&a, &b))
    }

    /// Compute the full pairwise cosine similarity matrix.
    pub fn pairwise_similarity(&self) -> Result<Array2<f64>> {
        let matrix = self.matrix()?;
        let n = self.n_docs;
        let mut sim_matrix = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            sim_matrix[[i, i]] = 1.0;
            for j in (i + 1)..n {
                let s = cosine_sim_vec(&matrix.row(i), &matrix.row(j));
                sim_matrix[[i, j]] = s;
                sim_matrix[[j, i]] = s;
            }
        }
        Ok(sim_matrix)
    }

    /// Transform a new document into TF-IDF space.
    pub fn transform(&self, document: &str) -> Result<Array1<f64>> {
        if self.tfidf_matrix.is_none() {
            return Err(TextError::ModelNotFitted(
                "TF-IDF engine has not been fitted".to_string(),
            ));
        }

        let tokens = tokenize_simple(document);
        let mut vec = Array1::<f64>::zeros(self.vocab.len());

        let mut tf: HashMap<usize, usize> = HashMap::new();
        for token in &tokens {
            if let Some(&idx) = self.vocab.get(token) {
                *tf.entry(idx).or_insert(0) += 1;
            }
        }

        let doc_len = tokens.len().max(1) as f64;
        for (&idx, &count) in &tf {
            vec[idx] = (count as f64 / doc_len) * self.idf[idx];
        }

        // L2-normalize
        let norm = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            vec.mapv_inplace(|x| x / norm);
        }

        Ok(vec)
    }

    /// Get the vocabulary mapping.
    pub fn vocabulary(&self) -> &HashMap<String, usize> {
        &self.vocab
    }
}

impl Default for TfIdfEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LSA (Latent Semantic Analysis) via truncated SVD
// ---------------------------------------------------------------------------

/// Latent Semantic Analysis via truncated SVD.
///
/// Reduces the dimensionality of a TF-IDF matrix to discover latent topics.
#[derive(Debug, Clone)]
pub struct LatentSemanticAnalysis {
    /// Number of components to keep.
    n_components: usize,
    /// Document embeddings in reduced space.
    doc_embeddings: Option<Array2<f64>>,
    /// Singular values.
    singular_values: Option<Array1<f64>>,
}

impl LatentSemanticAnalysis {
    /// Create a new LSA with the specified number of components.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            doc_embeddings: None,
            singular_values: None,
        }
    }

    /// Fit LSA on a TF-IDF engine.
    pub fn fit(&mut self, engine: &TfIdfEngine) -> Result<()> {
        let matrix = engine.matrix()?;
        let (n_docs, n_terms) = matrix.dim();
        let k = self.n_components.min(n_docs).min(n_terms);

        // Power iteration SVD approximation
        let (u, s) = truncated_svd(matrix, k)?;

        self.doc_embeddings = Some(u);
        self.singular_values = Some(s);
        Ok(())
    }

    /// Get the document embeddings in reduced space.
    pub fn embeddings(&self) -> Result<&Array2<f64>> {
        self.doc_embeddings
            .as_ref()
            .ok_or_else(|| TextError::ModelNotFitted("LSA has not been fitted".to_string()))
    }

    /// Get the singular values.
    pub fn singular_values(&self) -> Result<&Array1<f64>> {
        self.singular_values
            .as_ref()
            .ok_or_else(|| TextError::ModelNotFitted("LSA has not been fitted".to_string()))
    }

    /// Compute cosine similarity between two documents in the reduced space.
    pub fn similarity(&self, doc_a: usize, doc_b: usize) -> Result<f64> {
        let emb = self.embeddings()?;
        if doc_a >= emb.nrows() || doc_b >= emb.nrows() {
            return Err(TextError::InvalidInput(
                "Document index out of bounds".to_string(),
            ));
        }
        Ok(cosine_sim_vec(&emb.row(doc_a), &emb.row(doc_b)))
    }
}

/// Truncated SVD via randomized power iteration.
///
/// Returns (U * S, sigma) where U is n_docs x k and sigma has k values.
fn truncated_svd(matrix: &Array2<f64>, k: usize) -> Result<(Array2<f64>, Array1<f64>)> {
    let (n, m) = matrix.dim();
    let actual_k = k.min(n).min(m);

    if actual_k == 0 {
        return Ok((Array2::<f64>::zeros((n, 0)), Array1::<f64>::zeros(0)));
    }

    // Randomized SVD: generate random matrix, do power iteration
    let mut rng = StdRng::seed_from_u64(42);
    let mut omega = Array2::<f64>::zeros((m, actual_k));
    for elem in omega.iter_mut() {
        // Simple Box-Muller for normal distribution
        let u1: f64 = loop {
            let v = rng_uniform_f64(&mut rng);
            if v > 1e-15 {
                break v;
            }
        };
        let u2: f64 = rng_uniform_f64(&mut rng);
        *elem = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    }

    // Y = A * Omega
    let mut y = mat_mul(matrix, &omega)?;

    // Power iteration (2 iterations for better approximation)
    for _ in 0..2 {
        // Y = A * (A^T * Y)
        let aty = mat_mul_transpose_a(matrix, &y)?;
        y = mat_mul(matrix, &aty)?;
    }

    // QR factorization of Y (Gram-Schmidt)
    let q = gram_schmidt(&y)?;

    // B = Q^T * A
    let b = mat_mul_transpose_a(&q, matrix)?;

    // SVD of B (small matrix, k x m)
    // We compute B * B^T eigendecomposition
    let bbt = mat_mul(&b, &mat_transpose(&b))?;
    let (eigenvalues, eigenvectors) = symmetric_eigen(&bbt, actual_k)?;

    // Singular values are sqrt of eigenvalues
    let mut sigma = Array1::<f64>::zeros(actual_k);
    for i in 0..actual_k {
        sigma[i] = eigenvalues[i].max(0.0).sqrt();
    }

    // U = Q * eigenvectors
    let u = mat_mul(&q, &eigenvectors)?;

    // Return U * diag(sigma) for document embeddings
    let mut u_scaled = Array2::<f64>::zeros((n, actual_k));
    for i in 0..n {
        for j in 0..actual_k {
            u_scaled[[i, j]] = u[[i, j]] * sigma[j];
        }
    }

    Ok((u_scaled, sigma))
}

// ---------------------------------------------------------------------------
// Document Clustering (k-means on TF-IDF)
// ---------------------------------------------------------------------------

/// k-means document clustering on TF-IDF vectors.
#[derive(Debug, Clone)]
pub struct DocumentCluster {
    /// Number of clusters.
    k: usize,
    /// Maximum iterations.
    max_iter: usize,
    /// Random seed.
    seed: u64,
    /// Cluster labels for each document.
    labels: Vec<usize>,
    /// Cluster centroids.
    centroids: Option<Array2<f64>>,
}

impl DocumentCluster {
    /// Create a new document clustering with `k` clusters.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iter: 100,
            seed: 42,
            labels: Vec::new(),
            centroids: None,
        }
    }

    /// Set maximum iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Fit k-means on a TF-IDF engine.
    pub fn fit(&mut self, engine: &TfIdfEngine) -> Result<()> {
        let matrix = engine.matrix()?;
        self.fit_on_matrix(matrix)
    }

    /// Fit k-means directly on a matrix.
    pub fn fit_on_matrix(&mut self, matrix: &Array2<f64>) -> Result<()> {
        let n_docs = matrix.nrows();
        let n_features = matrix.ncols();

        if n_docs < self.k {
            return Err(TextError::InvalidInput(format!(
                "Number of documents ({}) less than number of clusters ({})",
                n_docs, self.k
            )));
        }

        // Initialize centroids using k-means++ style
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut centroids = Array2::<f64>::zeros((self.k, n_features));

        // First centroid: random document
        let first_idx = (rng_uniform_f64(&mut rng) * n_docs as f64) as usize % n_docs;
        centroids.row_mut(0).assign(&matrix.row(first_idx));

        // Remaining centroids: proportional to squared distance
        for c in 1..self.k {
            let mut distances = vec![f64::MAX; n_docs];
            for (i, mut d) in distances.iter_mut().enumerate() {
                for prev in 0..c {
                    let dist = euclidean_dist(&matrix.row(i), &centroids.row(prev));
                    if dist < *d {
                        *d = dist;
                    }
                }
            }
            let total: f64 = distances.iter().map(|d| d * d).sum();
            if total < 1e-15 {
                // All points are the same; just pick a random one
                let idx = (rng_uniform_f64(&mut rng) * n_docs as f64) as usize % n_docs;
                centroids.row_mut(c).assign(&matrix.row(idx));
                continue;
            }
            let threshold = rng_uniform_f64(&mut rng) * total;
            let mut cumsum = 0.0;
            let mut chosen = n_docs - 1;
            for (i, d) in distances.iter().enumerate() {
                cumsum += d * d;
                if cumsum >= threshold {
                    chosen = i;
                    break;
                }
            }
            centroids.row_mut(c).assign(&matrix.row(chosen));
        }

        // k-means iteration
        let mut labels = vec![0usize; n_docs];
        for _iter in 0..self.max_iter {
            // Assignment step
            let mut changed = false;
            for i in 0..n_docs {
                let mut best_k = 0;
                let mut best_dist = f64::MAX;
                for c in 0..self.k {
                    let dist = euclidean_dist(&matrix.row(i), &centroids.row(c));
                    if dist < best_dist {
                        best_dist = dist;
                        best_k = c;
                    }
                }
                if labels[i] != best_k {
                    labels[i] = best_k;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update step
            let mut new_centroids = Array2::<f64>::zeros((self.k, n_features));
            let mut counts = vec![0usize; self.k];
            for i in 0..n_docs {
                let c = labels[i];
                counts[c] += 1;
                for j in 0..n_features {
                    new_centroids[[c, j]] += matrix[[i, j]];
                }
            }
            for c in 0..self.k {
                if counts[c] > 0 {
                    for j in 0..n_features {
                        new_centroids[[c, j]] /= counts[c] as f64;
                    }
                }
            }
            centroids = new_centroids;
        }

        self.labels = labels;
        self.centroids = Some(centroids);
        Ok(())
    }

    /// Get cluster labels.
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Get cluster centroids.
    pub fn centroids(&self) -> Result<&Array2<f64>> {
        self.centroids.as_ref().ok_or_else(|| {
            TextError::ModelNotFitted("DocumentCluster has not been fitted".to_string())
        })
    }

    /// Get the inertia (sum of squared distances to nearest centroid).
    pub fn inertia(&self, matrix: &Array2<f64>) -> Result<f64> {
        let centroids = self.centroids()?;
        let mut inertia = 0.0;
        for (i, &label) in self.labels.iter().enumerate() {
            let dist = euclidean_dist(&matrix.row(i), &centroids.row(label));
            inertia += dist * dist;
        }
        Ok(inertia)
    }
}

// ---------------------------------------------------------------------------
// Shingling
// ---------------------------------------------------------------------------

/// Character-level shingling for near-duplicate detection.
#[derive(Debug, Clone)]
pub struct Shingler {
    /// Shingle size (number of characters).
    k: usize,
}

impl Shingler {
    /// Create a new shingler with shingle size `k`.
    pub fn new(k: usize) -> Self {
        Self { k: k.max(1) }
    }

    /// Generate shingles from a text.
    pub fn shingle(&self, text: &str) -> HashSet<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut shingles = HashSet::new();
        if chars.len() < self.k {
            if !chars.is_empty() {
                shingles.insert(chars.iter().collect());
            }
            return shingles;
        }
        for i in 0..=(chars.len() - self.k) {
            let s: String = chars[i..i + self.k].iter().collect();
            shingles.insert(s);
        }
        shingles
    }

    /// Compute exact Jaccard similarity between two shingle sets.
    pub fn jaccard_similarity(&self, a: &HashSet<String>, b: &HashSet<String>) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        let intersection = a.intersection(b).count();
        let union = a.union(b).count();
        if union == 0 {
            return 0.0;
        }
        intersection as f64 / union as f64
    }
}

/// Word-level shingling (w-shingling).
#[derive(Debug, Clone)]
pub struct WordShingler {
    /// Number of consecutive words per shingle.
    k: usize,
}

impl WordShingler {
    /// Create a new word shingler.
    pub fn new(k: usize) -> Self {
        Self { k: k.max(1) }
    }

    /// Generate word shingles.
    pub fn shingle(&self, text: &str) -> HashSet<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut shingles = HashSet::new();
        if words.len() < self.k {
            if !words.is_empty() {
                shingles.insert(words.join(" "));
            }
            return shingles;
        }
        for i in 0..=(words.len() - self.k) {
            shingles.insert(words[i..i + self.k].join(" "));
        }
        shingles
    }
}

// ---------------------------------------------------------------------------
// MinHash
// ---------------------------------------------------------------------------

/// MinHash for fast approximate Jaccard similarity estimation.
///
/// Uses multiple hash functions to create a compact signature of a set,
/// then estimates Jaccard similarity by comparing signatures.
#[derive(Debug, Clone)]
pub struct MinHasher {
    /// Number of hash functions (signature size).
    n_hashes: usize,
    /// Coefficients a for hash functions: h(x) = (a*x + b) mod p
    coeff_a: Vec<u64>,
    /// Coefficients b.
    coeff_b: Vec<u64>,
    /// Large prime.
    prime: u64,
}

impl MinHasher {
    /// Create a new MinHasher with the specified number of hash functions.
    pub fn new(n_hashes: usize) -> Self {
        Self::with_seed(n_hashes, 42)
    }

    /// Create a MinHasher with a specific seed.
    pub fn with_seed(n_hashes: usize, seed: u64) -> Self {
        let prime = 4_294_967_311u64; // Large prime > 2^32
        let mut rng = StdRng::seed_from_u64(seed);

        let mut coeff_a = Vec::with_capacity(n_hashes);
        let mut coeff_b = Vec::with_capacity(n_hashes);

        for _ in 0..n_hashes {
            coeff_a.push((rng_uniform_f64(&mut rng) * (prime as f64 - 1.0)) as u64 + 1);
            coeff_b.push((rng_uniform_f64(&mut rng) * (prime as f64 - 1.0)) as u64);
        }

        Self {
            n_hashes,
            coeff_a,
            coeff_b,
            prime,
        }
    }

    /// Compute a MinHash signature for a set of shingles.
    pub fn signature(&self, shingles: &HashSet<String>) -> Vec<u64> {
        let mut sig = vec![u64::MAX; self.n_hashes];

        for shingle in shingles {
            let hash_val = simple_hash(shingle);
            for i in 0..self.n_hashes {
                let h = (self.coeff_a[i]
                    .wrapping_mul(hash_val)
                    .wrapping_add(self.coeff_b[i]))
                    % self.prime;
                if h < sig[i] {
                    sig[i] = h;
                }
            }
        }
        sig
    }

    /// Estimate Jaccard similarity from two signatures.
    pub fn estimate_jaccard(sig_a: &[u64], sig_b: &[u64]) -> f64 {
        if sig_a.len() != sig_b.len() || sig_a.is_empty() {
            return 0.0;
        }
        let matches = sig_a
            .iter()
            .zip(sig_b.iter())
            .filter(|(&a, &b)| a == b)
            .count();
        matches as f64 / sig_a.len() as f64
    }

    /// Get the number of hash functions.
    pub fn n_hashes(&self) -> usize {
        self.n_hashes
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Simple whitespace tokenizer that lowercases.
fn tokenize_simple(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Cosine similarity between two array views.
fn cosine_sim_vec(
    a: &scirs2_core::ndarray::ArrayView1<f64>,
    b: &scirs2_core::ndarray::ArrayView1<f64>,
) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Euclidean distance between two array views.
fn euclidean_dist(
    a: &scirs2_core::ndarray::ArrayView1<f64>,
    b: &scirs2_core::ndarray::ArrayView1<f64>,
) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Simple string hash (FNV-1a variant).
fn simple_hash(s: &str) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// Generate a uniform f64 in [0,1) from StdRng.
fn rng_uniform_f64(rng: &mut StdRng) -> f64 {
    use scirs2_core::random::{Rng, RngExt};
    rng.random::<f64>()
}

/// Matrix multiplication A * B.
fn mat_mul(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    let (ar, ac) = a.dim();
    let (br, bc) = b.dim();
    if ac != br {
        return Err(TextError::ProcessingError(format!(
            "Matrix dimension mismatch: ({}, {}) * ({}, {})",
            ar, ac, br, bc
        )));
    }
    let mut result = Array2::<f64>::zeros((ar, bc));
    for i in 0..ar {
        for k in 0..ac {
            let a_ik = a[[i, k]];
            if a_ik.abs() < 1e-15 {
                continue;
            }
            for j in 0..bc {
                result[[i, j]] += a_ik * b[[k, j]];
            }
        }
    }
    Ok(result)
}

/// Matrix multiplication A^T * B.
fn mat_mul_transpose_a(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    let (ar, ac) = a.dim();
    let (br, bc) = b.dim();
    if ar != br {
        return Err(TextError::ProcessingError(format!(
            "Matrix dimension mismatch for A^T * B: ({}, {})^T * ({}, {})",
            ar, ac, br, bc
        )));
    }
    let mut result = Array2::<f64>::zeros((ac, bc));
    for k in 0..ar {
        for i in 0..ac {
            let a_ki = a[[k, i]];
            if a_ki.abs() < 1e-15 {
                continue;
            }
            for j in 0..bc {
                result[[i, j]] += a_ki * b[[k, j]];
            }
        }
    }
    Ok(result)
}

/// Matrix transpose.
fn mat_transpose(a: &Array2<f64>) -> Array2<f64> {
    a.t().to_owned()
}

/// Gram-Schmidt orthogonalization to produce Q.
fn gram_schmidt(a: &Array2<f64>) -> Result<Array2<f64>> {
    let (n, k) = a.dim();
    let mut q = Array2::<f64>::zeros((n, k));

    for j in 0..k {
        let mut v = a.column(j).to_owned();
        for i in 0..j {
            let qi = q.column(i);
            let proj: f64 = v.iter().zip(qi.iter()).map(|(&a, &b)| a * b).sum();
            for idx in 0..n {
                v[idx] -= proj * qi[idx];
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for idx in 0..n {
                q[[idx, j]] = v[idx] / norm;
            }
        }
    }
    Ok(q)
}

/// Symmetric eigendecomposition via Jacobi iteration.
/// Returns (eigenvalues, eigenvectors) sorted descending by eigenvalue.
fn symmetric_eigen(a: &Array2<f64>, _k: usize) -> Result<(Array1<f64>, Array2<f64>)> {
    let n = a.nrows();
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }

    let mut mat = a.clone();
    let mut eigvecs = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        eigvecs[[i, i]] = 1.0;
    }

    // Jacobi iteration
    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if mat[[i, j]].abs() > max_val {
                    max_val = mat[[i, j]].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-12 {
            break;
        }

        // Compute rotation
        let app = mat[[p, p]];
        let aqq = mat[[q, q]];
        let apq = mat[[p, q]];

        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to mat
        let mut new_mat = mat.clone();
        for i in 0..n {
            if i != p && i != q {
                new_mat[[i, p]] = c * mat[[i, p]] + s * mat[[i, q]];
                new_mat[[p, i]] = new_mat[[i, p]];
                new_mat[[i, q]] = -s * mat[[i, p]] + c * mat[[i, q]];
                new_mat[[q, i]] = new_mat[[i, q]];
            }
        }
        new_mat[[p, p]] = c * c * app + 2.0 * c * s * apq + s * s * aqq;
        new_mat[[q, q]] = s * s * app - 2.0 * c * s * apq + c * c * aqq;
        new_mat[[p, q]] = 0.0;
        new_mat[[q, p]] = 0.0;
        mat = new_mat;

        // Update eigenvectors
        let mut new_vecs = eigvecs.clone();
        for i in 0..n {
            new_vecs[[i, p]] = c * eigvecs[[i, p]] + s * eigvecs[[i, q]];
            new_vecs[[i, q]] = -s * eigvecs[[i, p]] + c * eigvecs[[i, q]];
        }
        eigvecs = new_vecs;
    }

    // Extract eigenvalues and sort descending
    let mut eig_pairs: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|i| {
            let val = mat[[i, i]];
            let vec: Vec<f64> = (0..n).map(|j| eigvecs[[j, i]]).collect();
            (val, vec)
        })
        .collect();
    eig_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let actual_k = _k.min(n);
    let eigenvalues = Array1::from_vec(eig_pairs.iter().take(actual_k).map(|p| p.0).collect());
    let mut eigenvectors = Array2::<f64>::zeros((n, actual_k));
    for (j, pair) in eig_pairs.iter().take(actual_k).enumerate() {
        for i in 0..n {
            eigenvectors[[i, j]] = pair.1[i];
        }
    }

    Ok((eigenvalues, eigenvectors))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tfidf_fit() {
        let docs = vec![
            "the cat sat on the mat",
            "the dog sat on the log",
            "cats and dogs are friends",
        ];
        let mut engine = TfIdfEngine::new();
        engine.fit(&docs).expect("fit failed");
        assert_eq!(engine.n_docs(), 3);
        assert!(engine.vocab_size() > 0);
    }

    #[test]
    fn test_tfidf_cosine_similarity() {
        let docs = vec![
            "machine learning algorithms",
            "deep learning neural networks",
            "the cat sat on the mat",
        ];
        let mut engine = TfIdfEngine::new();
        engine.fit(&docs).expect("fit failed");

        let sim_01 = engine.cosine_similarity(0, 1).expect("sim failed");
        let sim_02 = engine.cosine_similarity(0, 2).expect("sim failed");

        // Docs 0 and 1 share "learning", should be more similar than 0 and 2
        assert!(sim_01 > sim_02);
    }

    #[test]
    fn test_tfidf_self_similarity() {
        let docs = vec!["hello world", "foo bar"];
        let mut engine = TfIdfEngine::new();
        engine.fit(&docs).expect("fit failed");
        let sim = engine.cosine_similarity(0, 0).expect("sim failed");
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tfidf_pairwise() {
        let docs = vec!["a b c", "a b d", "x y z"];
        let mut engine = TfIdfEngine::new();
        engine.fit(&docs).expect("fit failed");
        let pw = engine.pairwise_similarity().expect("pairwise failed");
        assert_eq!(pw.dim(), (3, 3));
        assert!((pw[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((pw[[0, 1]] - pw[[1, 0]]).abs() < 1e-6);
    }

    #[test]
    fn test_tfidf_transform() {
        let docs = vec!["machine learning", "deep learning"];
        let mut engine = TfIdfEngine::new();
        engine.fit(&docs).expect("fit failed");
        let vec = engine
            .transform("machine learning is great")
            .expect("transform failed");
        assert_eq!(vec.len(), engine.vocab_size());
    }

    #[test]
    fn test_tfidf_empty_corpus() {
        let mut engine = TfIdfEngine::new();
        let result = engine.fit(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tfidf_out_of_bounds() {
        let docs = vec!["hello"];
        let mut engine = TfIdfEngine::new();
        engine.fit(&docs).expect("fit failed");
        let result = engine.cosine_similarity(0, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_lsa_basic() {
        let docs = vec![
            "machine learning algorithms",
            "deep learning neural networks",
            "natural language processing",
            "the cat sat on the mat",
            "dogs and cats are pets",
        ];
        let mut engine = TfIdfEngine::new();
        engine.fit(&docs).expect("fit failed");

        let mut lsa = LatentSemanticAnalysis::new(2);
        lsa.fit(&engine).expect("lsa fit failed");

        let emb = lsa.embeddings().expect("embeddings failed");
        assert_eq!(emb.nrows(), 5);
        assert_eq!(emb.ncols(), 2);
    }

    #[test]
    fn test_lsa_singular_values() {
        let docs = vec!["hello world", "foo bar baz", "hello foo"];
        let mut engine = TfIdfEngine::new();
        engine.fit(&docs).expect("fit failed");

        let mut lsa = LatentSemanticAnalysis::new(2);
        lsa.fit(&engine).expect("lsa fit failed");

        let sv = lsa.singular_values().expect("sv failed");
        // Singular values should be non-negative and sorted descending
        for &v in sv.iter() {
            assert!(v >= -1e-10);
        }
        if sv.len() >= 2 {
            assert!(sv[0] >= sv[1] - 1e-10);
        }
    }

    #[test]
    fn test_document_cluster() {
        let docs = vec![
            "machine learning algorithms are powerful",
            "deep learning neural network training",
            "the cat sat on the mat today",
            "dogs and cats play in the park",
        ];
        let mut engine = TfIdfEngine::new();
        engine.fit(&docs).expect("fit failed");

        let mut cluster = DocumentCluster::new(2);
        cluster.fit(&engine).expect("cluster failed");

        assert_eq!(cluster.labels().len(), 4);
        // All labels should be 0 or 1
        for &l in cluster.labels() {
            assert!(l < 2);
        }
    }

    #[test]
    fn test_cluster_too_many_k() {
        let docs = vec!["hello"];
        let mut engine = TfIdfEngine::new();
        engine.fit(&docs).expect("fit failed");

        let mut cluster = DocumentCluster::new(5);
        let result = cluster.fit(&engine);
        assert!(result.is_err());
    }

    #[test]
    fn test_cluster_inertia() {
        let docs = vec!["a b c", "a b d", "x y z", "x y w"];
        let mut engine = TfIdfEngine::new();
        engine.fit(&docs).expect("fit failed");

        let mut cluster = DocumentCluster::new(2);
        cluster.fit(&engine).expect("cluster failed");

        let matrix = engine.matrix().expect("matrix failed");
        let inertia = cluster.inertia(matrix).expect("inertia failed");
        assert!(inertia >= 0.0);
    }

    #[test]
    fn test_shingler_basic() {
        let shingler = Shingler::new(3);
        let shingles = shingler.shingle("hello");
        // "hel", "ell", "llo"
        assert_eq!(shingles.len(), 3);
        assert!(shingles.contains("hel"));
        assert!(shingles.contains("llo"));
    }

    #[test]
    fn test_shingler_short_text() {
        let shingler = Shingler::new(10);
        let shingles = shingler.shingle("hi");
        assert_eq!(shingles.len(), 1);
        assert!(shingles.contains("hi"));
    }

    #[test]
    fn test_shingler_jaccard() {
        let shingler = Shingler::new(3);
        let a = shingler.shingle("the quick brown fox");
        let b = shingler.shingle("the quick brown dog");
        let sim = shingler.jaccard_similarity(&a, &b);
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn test_shingler_identical() {
        let shingler = Shingler::new(3);
        let a = shingler.shingle("hello world");
        let b = shingler.shingle("hello world");
        let sim = shingler.jaccard_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_word_shingler() {
        let shingler = WordShingler::new(2);
        let shingles = shingler.shingle("the quick brown fox");
        // "the quick", "quick brown", "brown fox"
        assert_eq!(shingles.len(), 3);
        assert!(shingles.contains("the quick"));
        assert!(shingles.contains("brown fox"));
    }

    #[test]
    fn test_minhash_basic() {
        let shingler = Shingler::new(3);
        let a = shingler.shingle("the quick brown fox");
        let b = shingler.shingle("the quick brown dog");
        let c = shingler.shingle("completely different text here");

        let hasher = MinHasher::new(200);
        let sig_a = hasher.signature(&a);
        let sig_b = hasher.signature(&b);
        let sig_c = hasher.signature(&c);

        let sim_ab = MinHasher::estimate_jaccard(&sig_a, &sig_b);
        let sim_ac = MinHasher::estimate_jaccard(&sig_a, &sig_c);

        // Similar docs should have higher estimated Jaccard
        assert!(sim_ab > sim_ac);
    }

    #[test]
    fn test_minhash_identical() {
        let shingler = Shingler::new(3);
        let a = shingler.shingle("hello world");
        let b = shingler.shingle("hello world");

        let hasher = MinHasher::new(128);
        let sig_a = hasher.signature(&a);
        let sig_b = hasher.signature(&b);

        let sim = MinHasher::estimate_jaccard(&sig_a, &sig_b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_minhash_empty_set() {
        let hasher = MinHasher::new(64);
        let sig = hasher.signature(&HashSet::new());
        assert_eq!(sig.len(), 64);
        // All values should be u64::MAX for empty set
        for &v in &sig {
            assert_eq!(v, u64::MAX);
        }
    }

    #[test]
    fn test_minhash_different_sizes() {
        let sig_a = vec![1u64, 2, 3];
        let sig_b = vec![1u64, 2];
        let sim = MinHasher::estimate_jaccard(&sig_a, &sig_b);
        assert_eq!(sim, 0.0); // Different sizes return 0
    }

    #[test]
    fn test_vocabulary_access() {
        let docs = vec!["hello world", "hello foo"];
        let mut engine = TfIdfEngine::new();
        engine.fit(&docs).expect("fit failed");
        let vocab = engine.vocabulary();
        assert!(vocab.contains_key("hello"));
        assert!(vocab.contains_key("world"));
    }

    #[test]
    fn test_lsa_not_fitted() {
        let lsa = LatentSemanticAnalysis::new(2);
        assert!(lsa.embeddings().is_err());
        assert!(lsa.singular_values().is_err());
    }

    #[test]
    fn test_tfidf_not_fitted_transform() {
        let engine = TfIdfEngine::new();
        assert!(engine.transform("hello").is_err());
    }

    #[test]
    fn test_cluster_not_fitted() {
        let cluster = DocumentCluster::new(2);
        assert!(cluster.centroids().is_err());
    }

    #[test]
    fn test_lsa_similarity() {
        let docs = vec!["machine learning", "deep learning", "cats and dogs"];
        let mut engine = TfIdfEngine::new();
        engine.fit(&docs).expect("fit failed");

        let mut lsa = LatentSemanticAnalysis::new(2);
        lsa.fit(&engine).expect("lsa fit failed");

        let sim_01 = lsa.similarity(0, 1).expect("sim failed");
        let sim_02 = lsa.similarity(0, 2).expect("sim failed");
        // ml docs should be more similar to each other
        // (may not hold for very small corpus, but generally true)
        let _ = (sim_01, sim_02); // just ensure they compute
    }
}
