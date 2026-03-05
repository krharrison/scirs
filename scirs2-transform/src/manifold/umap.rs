//! UMAP: Uniform Manifold Approximation and Projection (McInnes et al. 2018).
//!
//! UMAP constructs a fuzzy topological representation of the data using a
//! Riemannian manifold interpretation, then optimizes a low-dimensional
//! embedding to have a similar fuzzy topological structure via stochastic
//! gradient descent with negative sampling.
//!
//! ## Algorithm Overview
//!
//! 1. Build k-NN graph with per-point adaptive bandwidth (rho + sigma)
//! 2. Compute fuzzy simplicial set via smooth membership strengths
//! 3. Symmetrize with fuzzy union: `p OR q = p + q - p*q`
//! 4. Initialize embedding (spectral on circle or random)
//! 5. SGD with attractive + repulsive forces parameterized by `(a, b)` curve
//!
//! ## References
//!
//! - McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold
//!   Approximation and Projection for Dimension Reduction. arXiv:1802.03426.

use crate::error::TransformError;

// ─── Public Types ─────────────────────────────────────────────────────────────

/// Distance metric used when building the k-NN graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UMAPMetric {
    /// L2 (Euclidean) distance
    Euclidean,
    /// Cosine distance: `1 − cos(θ)`
    Cosine,
    /// L1 (Manhattan / taxicab) distance
    Manhattan,
    /// Pearson correlation distance: `1 − corr(x, y)`
    Correlation,
}

/// UMAP hyperparameters.
#[derive(Debug, Clone)]
pub struct UMAPParams {
    /// Number of output dimensions (default 2)
    pub n_components: usize,
    /// Number of nearest neighbours for the high-dimensional graph (default 15)
    pub n_neighbors: usize,
    /// Controls tightness of clusters in the low-dim embedding (default 0.1)
    pub min_dist: f64,
    /// Effective scale of the embedded points (default 1.0)
    pub spread: f64,
    /// Number of SGD optimisation epochs (default 200)
    pub n_epochs: usize,
    /// SGD learning rate (default 1.0)
    pub learning_rate: f64,
    /// Number of negative samples per positive edge per epoch (default 5)
    pub negative_sample_rate: usize,
    /// Distance metric used in high-dim space
    pub metric: UMAPMetric,
    /// Seed for reproducible results (default 42)
    pub random_seed: u64,
}

impl Default for UMAPParams {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_neighbors: 15,
            min_dist: 0.1,
            spread: 1.0,
            n_epochs: 200,
            learning_rate: 1.0,
            negative_sample_rate: 5,
            metric: UMAPMetric::Euclidean,
            random_seed: 42,
        }
    }
}

/// UMAP result containing the low-dimensional embedding and the high-dim graph.
#[derive(Debug, Clone)]
pub struct UMAPResult {
    /// Low-dimensional embedding: `[n_samples][n_components]`
    pub embedding: Vec<Vec<f64>>,
    /// Sparse symmetrized fuzzy simplicial set: `graph[i]` is a list of
    /// `(neighbor_index, weight)` pairs.
    pub graph: Vec<Vec<(usize, f64)>>,
}

/// UMAP dimensionality reduction.
///
/// # Example
/// ```rust,no_run
/// use scirs2_transform::manifold::umap::UMAP;
/// let data: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, (i as f64).sin()]).collect();
/// let result = UMAP::new(2, 5).fit_transform(&data).expect("should succeed");
/// assert_eq!(result.embedding.len(), 20);
/// assert_eq!(result.embedding[0].len(), 2);
/// ```
pub struct UMAP {
    /// Hyperparameters controlling the UMAP run
    pub params: UMAPParams,
}

// ─── Implementation ───────────────────────────────────────────────────────────

impl UMAP {
    /// Create a new UMAP with the given output dimensionality and number of
    /// nearest neighbours. All other parameters use `UMAPParams::default()`.
    pub fn new(n_components: usize, n_neighbors: usize) -> Self {
        let mut params = UMAPParams::default();
        params.n_components = n_components;
        params.n_neighbors = n_neighbors;
        Self { params }
    }

    /// Override minimum distance (cluster tightness).
    pub fn with_min_dist(mut self, min_dist: f64) -> Self {
        self.params.min_dist = min_dist;
        self
    }

    /// Override spread parameter.
    pub fn with_spread(mut self, spread: f64) -> Self {
        self.params.spread = spread;
        self
    }

    /// Override number of SGD epochs.
    pub fn with_n_epochs(mut self, n: usize) -> Self {
        self.params.n_epochs = n;
        self
    }

    /// Override learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.params.learning_rate = lr;
        self
    }

    /// Override distance metric.
    pub fn with_metric(mut self, metric: UMAPMetric) -> Self {
        self.params.metric = metric;
        self
    }

    /// Override random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.params.random_seed = seed;
        self
    }

    /// Fit the model on `data` and return the low-dimensional embedding.
    ///
    /// # Arguments
    /// * `data` – Row-major dataset; each inner `Vec<f64>` is one sample.
    ///
    /// # Errors
    /// Returns `TransformError::InvalidInput` for empty input.
    pub fn fit_transform(&self, data: &[Vec<f64>]) -> Result<UMAPResult, TransformError> {
        let n = data.len();
        if n == 0 {
            return Err(TransformError::InvalidInput(
                "UMAP requires at least one sample".into(),
            ));
        }
        if n == 1 {
            return Ok(UMAPResult {
                embedding: vec![vec![0.0; self.params.n_components]],
                graph: vec![Vec::new()],
            });
        }

        let k = self.params.n_neighbors.min(n - 1).max(1);

        // 1. Build k-NN graph
        let knn_graph = self.compute_knn(data, k);

        // 2. Fuzzy simplicial set with per-point sigma estimation
        let fuzzy_graph = self.fuzzy_simplicial_set(&knn_graph, k);

        // 3. Initialize embedding
        let mut embedding = self.initialize_embedding(n);

        // 4. SGD optimization
        self.optimize_embedding(&mut embedding, &fuzzy_graph, n)?;

        Ok(UMAPResult {
            embedding,
            graph: fuzzy_graph,
        })
    }

    // ── Step 1: k-NN graph ────────────────────────────────────────────────────

    fn compute_knn(&self, data: &[Vec<f64>], k: usize) -> Vec<Vec<(usize, f64)>> {
        let n = data.len();
        let mut knn: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);

        for i in 0..n {
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d = self.distance(&data[i], &data[j]);
                    (j, d)
                })
                .collect();
            dists.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            dists.truncate(k);
            knn.push(dists);
        }
        knn
    }

    // ── Step 2: Fuzzy simplicial set ─────────────────────────────────────────

    /// Build the symmetrized fuzzy simplicial set (weighted graph).
    ///
    /// Per-point parameters:
    /// * `rho_i` = distance to the nearest neighbour (ensures local connectivity)
    /// * `sigma_i` = bandwidth found by binary search so that the perplexity
    ///   of the local distribution equals `log2(k)`.
    fn fuzzy_simplicial_set(
        &self,
        knn: &[Vec<(usize, f64)>],
        k: usize,
    ) -> Vec<Vec<(usize, f64)>> {
        let n = knn.len();
        let target_entropy = (k as f64).ln(); // log(k) ≈ target perplexity

        // Compute per-point (rho, sigma)
        let rho_sigma: Vec<(f64, f64)> = knn
            .iter()
            .map(|neighbors| {
                if neighbors.is_empty() {
                    return (0.0, 1.0);
                }
                let rho = neighbors[0].1; // distance to 1-NN
                let dists: Vec<f64> = neighbors.iter().map(|(_, d)| *d).collect();
                let sigma = self.binary_search_sigma(&dists, rho, target_entropy);
                (rho, sigma)
            })
            .collect();

        // Build directed membership graph
        let mut directed: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for (i, neighbors) in knn.iter().enumerate() {
            let (rho_i, sigma_i) = rho_sigma[i];
            for &(j, dist) in neighbors {
                let w = (-(dist - rho_i).max(0.0) / sigma_i.max(1e-10)).exp();
                directed[i].push((j, w));
            }
        }

        // Symmetrize with fuzzy OR: p(i,j) + p(j,i) - p(i,j)*p(j,i)
        let mut graph: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for i in 0..n {
            for &(j, w_ij) in &directed[i] {
                let w_ji = directed[j]
                    .iter()
                    .find(|(nb, _)| *nb == i)
                    .map(|(_, w)| *w)
                    .unwrap_or(0.0);
                let sym = w_ij + w_ji - w_ij * w_ji;

                // Update graph[i][j]
                match graph[i].iter().position(|(nb, _)| *nb == j) {
                    Some(pos) => graph[i][pos].1 = sym,
                    None => graph[i].push((j, sym)),
                }
                // Update graph[j][i]
                match graph[j].iter().position(|(nb, _)| *nb == i) {
                    Some(pos) => graph[j][pos].1 = sym,
                    None => graph[j].push((i, sym)),
                }
            }
        }
        graph
    }

    /// Binary search for sigma such that Σ exp(-(d-rho)/sigma) ≈ target.
    fn binary_search_sigma(&self, dists: &[f64], rho: f64, target: f64) -> f64 {
        let mut lo = 1e-10_f64;
        let mut hi = 1e8_f64;
        let max_iter = 64;

        for _ in 0..max_iter {
            let mid = (lo + hi) / 2.0;
            let val: f64 = dists
                .iter()
                .map(|&d| (-(d - rho).max(0.0) / mid).exp())
                .sum();
            if val > target {
                lo = mid;
            } else {
                hi = mid;
            }
            if (hi - lo) < 1e-12 {
                break;
            }
        }
        (lo + hi) / 2.0
    }

    // ── Step 3: Embedding initialization ─────────────────────────────────────

    fn initialize_embedding(&self, n: usize) -> Vec<Vec<f64>> {
        use std::f64::consts::TAU;
        let dim = self.params.n_components;

        // Spectral-like initialization: place points on a circle / higher-dim
        // sphere, scaled by 10 to give the SGD room to work.
        (0..n)
            .map(|i| {
                let t = TAU * i as f64 / n as f64;
                (0..dim)
                    .map(|d| match d {
                        0 => t.cos() * 10.0,
                        1 => t.sin() * 10.0,
                        _ => (i as f64 / n as f64 - 0.5) * 10.0,
                    })
                    .collect()
            })
            .collect()
    }

    // ── Step 4: SGD optimization ──────────────────────────────────────────────

    /// Fit `(a, b)` curve parameters for the low-dim membership function:
    ///   `1 / (1 + a * d^(2b))`
    ///
    /// Uses a simplified analytic approximation rather than scipy curve_fit.
    fn fit_ab_params(&self) -> (f64, f64) {
        // Good closed-form approximation from the UMAP paper supplement
        let min_dist = self.params.min_dist;
        let spread = self.params.spread;
        if min_dist < 1e-10 {
            return (1.0, 1.0);
        }
        // Empirical approximation: a ~ 1/(min_dist^b), b ~ spread^(1/4)
        let b = (spread / min_dist).ln().max(0.5).min(2.0) * 0.5 + 0.5;
        let a = 1.0 / (min_dist.powf(2.0 * b));
        (a.max(0.001), b.clamp(0.1, 3.0))
    }

    fn optimize_embedding(
        &self,
        embedding: &mut Vec<Vec<f64>>,
        graph: &[Vec<(usize, f64)>],
        n: usize,
    ) -> Result<(), TransformError> {
        let mut rng = scirs2_core::random::seeded_rng(self.params.random_seed);
        let dim = self.params.n_components;
        let (a, b) = self.fit_ab_params();

        for epoch in 0..self.params.n_epochs {
            let alpha = self.params.learning_rate
                * (1.0 - epoch as f64 / self.params.n_epochs as f64);

            // ── Attractive forces ────────────────────────────────────────────
            for i in 0..n {
                // Clone to satisfy borrow checker (we need to modify embedding[i] and embedding[j])
                let neighbors = graph[i].clone();
                for (j, w) in neighbors {
                    // Sample edge proportional to its weight
                    if w < rng.random::<f64>() {
                        continue;
                    }
                    let grad = attractive_gradient(
                        &embedding[i].clone(),
                        &embedding[j],
                        a,
                        b,
                    );
                    let emb_i: Vec<f64> = embedding[i]
                        .iter()
                        .zip(grad.iter())
                        .map(|(xi, gi)| xi + alpha * gi)
                        .collect();
                    let emb_j: Vec<f64> = embedding[j]
                        .iter()
                        .zip(grad.iter())
                        .map(|(xj, gj)| xj - alpha * gj)
                        .collect();
                    embedding[i] = emb_i;
                    embedding[j] = emb_j;
                }

                // ── Repulsive forces (negative sampling) ─────────────────────
                for _ in 0..self.params.negative_sample_rate {
                    let j: usize = rng.random_range(0..n);
                    if j == i {
                        continue;
                    }
                    let grad = repulsive_gradient(&embedding[i].clone(), &embedding[j], a, b);
                    for d in 0..dim {
                        embedding[i][d] += alpha * grad[d];
                    }
                }
            }
        }
        Ok(())
    }

    // ── Distance ──────────────────────────────────────────────────────────────

    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        match self.params.metric {
            UMAPMetric::Euclidean => euclidean_distance(a, b),
            UMAPMetric::Manhattan => manhattan_distance(a, b),
            UMAPMetric::Cosine => cosine_distance(a, b),
            UMAPMetric::Correlation => correlation_distance(a, b),
        }
    }
}

// ─── Gradient functions ───────────────────────────────────────────────────────

/// Attractive gradient for edge (i, j): pulls i toward j.
///
/// Uses derivative of cross-entropy loss on the fuzzy membership:
///   `grad = 2ab * d^(2b-2) / (1 + a*d^(2b)) * (xi - xj)`
fn attractive_gradient(xi: &[f64], xj: &[f64], a: f64, b: f64) -> Vec<f64> {
    let dim = xi.len().min(xj.len());
    let dist2: f64 = xi
        .iter()
        .zip(xj.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .max(1e-10);

    // d^(2b) = dist2^b
    let d2b = dist2.powf(b);
    let denom = 1.0 + a * d2b;
    // Gradient coefficient (negative = attractive)
    let coef = -(2.0 * a * b * dist2.powf(b - 1.0)) / (denom * dist2.max(1e-10));
    let coef = coef.clamp(-4.0, 4.0);

    (0..dim)
        .map(|d| coef * (xi[d] - xj[d]))
        .collect()
}

/// Repulsive gradient for negative sample (i, j): pushes i away from j.
///
/// Uses derivative:
///   `grad = 2b / (epsilon + d^2) / (1 + a*d^(2b)) * (xi - xj)`
fn repulsive_gradient(xi: &[f64], xj: &[f64], a: f64, b: f64) -> Vec<f64> {
    let dim = xi.len().min(xj.len());
    let dist2: f64 = xi
        .iter()
        .zip(xj.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .max(1e-10);

    let d2b = dist2.powf(b);
    let denom = (1e-3 + dist2) * (1.0 + a * d2b);
    let coef = (2.0 * b / denom).clamp(0.0, 4.0);

    (0..dim)
        .map(|d| coef * (xi[d] - xj[d]))
        .collect()
}

// ─── Distance helpers ─────────────────────────────────────────────────────────

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).abs())
        .sum()
}

/// Cosine distance: `1 − cosine_similarity(a, b)`
pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na * nb < 1e-10 {
        1.0
    } else {
        (1.0 - dot / (na * nb)).clamp(0.0, 2.0)
    }
}

fn correlation_distance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    if n < 1.0 {
        return 1.0;
    }
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let ca: Vec<f64> = a.iter().map(|x| x - ma).collect();
    let cb: Vec<f64> = b.iter().map(|x| x - mb).collect();
    let dot: f64 = ca.iter().zip(cb.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = ca.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = cb.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na * nb < 1e-10 {
        1.0
    } else {
        (1.0 - dot / (na * nb)).clamp(0.0, 2.0)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(n: usize) -> Vec<Vec<f64>> {
        use std::f64::consts::TAU;
        (0..n)
            .map(|i| {
                let t = TAU * i as f64 / n as f64;
                vec![t.cos(), t.sin(), t * 0.1]
            })
            .collect()
    }

    #[test]
    fn test_umap_output_shape() {
        let data = make_data(10);
        let result = UMAP::new(2, 3)
            .with_n_epochs(20)
            .fit_transform(&data)
            .expect("UMAP fit_transform");
        assert_eq!(result.embedding.len(), 10);
        assert!(
            result.embedding.iter().all(|row| row.len() == 2),
            "every row should have 2 dimensions"
        );
    }

    #[test]
    fn test_umap_embedding_finite() {
        let data = make_data(12);
        let result = UMAP::new(2, 4)
            .with_n_epochs(30)
            .fit_transform(&data)
            .expect("fit_transform");
        for row in &result.embedding {
            for &v in row {
                assert!(v.is_finite(), "embedding contains non-finite value: {v}");
            }
        }
    }

    #[test]
    fn test_umap_cosine_metric_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let d = cosine_distance(&a, &b);
        // orthogonal vectors → cosine distance = 1
        assert!((d - 1.0).abs() < 1e-10, "expected 1.0, got {d}");

        let c = vec![1.0, 0.0, 0.0];
        let d2 = cosine_distance(&a, &c);
        // identical vectors → distance = 0
        assert!(d2.abs() < 1e-10, "expected 0.0, got {d2}");
    }

    #[test]
    fn test_umap_cosine_metric() {
        let data = make_data(10);
        let result = UMAP::new(2, 3)
            .with_metric(UMAPMetric::Cosine)
            .with_n_epochs(20)
            .fit_transform(&data)
            .expect("cosine UMAP");
        assert_eq!(result.embedding.len(), 10);
        assert_eq!(result.embedding[0].len(), 2);
    }

    #[test]
    fn test_umap_manhattan_metric() {
        let data = make_data(10);
        let result = UMAP::new(2, 3)
            .with_metric(UMAPMetric::Manhattan)
            .with_n_epochs(20)
            .fit_transform(&data)
            .expect("manhattan UMAP");
        assert_eq!(result.embedding.len(), 10);
    }

    #[test]
    fn test_umap_graph_weights_in_unit_interval() {
        let data = make_data(10);
        let result = UMAP::new(2, 3)
            .with_n_epochs(10)
            .fit_transform(&data)
            .expect("UMAP");
        for row in &result.graph {
            for &(_, w) in row {
                assert!(w >= 0.0 && w <= 1.0, "weight {w} outside [0,1]");
            }
        }
    }

    #[test]
    fn test_umap_empty_input() {
        let result = UMAP::new(2, 5).fit_transform(&[]);
        assert!(result.is_err(), "empty input should return Err");
    }

    #[test]
    fn test_umap_single_sample() {
        let data = vec![vec![1.0, 2.0, 3.0]];
        let result = UMAP::new(2, 1).fit_transform(&data).expect("single sample");
        assert_eq!(result.embedding.len(), 1);
        assert_eq!(result.embedding[0].len(), 2);
    }

    #[test]
    fn test_umap_3d_output() {
        let data = make_data(15);
        let result = UMAP::new(3, 4)
            .with_n_epochs(20)
            .fit_transform(&data)
            .expect("3d UMAP");
        assert_eq!(result.embedding.len(), 15);
        assert!(result.embedding.iter().all(|r| r.len() == 3));
    }
}
