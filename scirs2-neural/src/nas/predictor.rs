//! Predictor-based Neural Architecture Search
//!
//! Predictor-based NAS uses a surrogate model to predict the validation
//! performance of architectures without training them fully. The search
//! loop alternates between:
//!  1. Fitting the predictor on evaluated (arch, score) pairs.
//!  2. Using an acquisition function to select promising candidates.
//!  3. Evaluating the top candidates and adding them to the dataset.
//!
//! Two predictor types are provided:
//!  - `GpPredictor`: Gaussian Process with RBF kernel (provides uncertainty)
//!  - `MlpPredictor`: 3-layer MLP trained with SGD on MSE loss
//!
//! Three acquisition functions are provided:
//!  - `ExpectedImprovement` (EI) — balances exploration and exploitation
//!  - `UpperConfidenceBound` (UCB) — trades off mean and std with kappa
//!  - `ThompsonSampling` — samples from the posterior predictive

use crate::error::{NeuralError, Result};
use scirs2_core::random::{ChaCha20Rng, Rng, RngExt, SeedableRng};

// ── Acquisition function ───────────────────────────────────────────────────

/// Which acquisition function to use when selecting candidate architectures
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquisitionFunction {
    /// Expected Improvement over the current best (Mockus 1974)
    ExpectedImprovement,
    /// Upper Confidence Bound with parameter kappa (Srinivas et al. 2010)
    UpperConfidenceBound,
    /// Thompson Sampling: sample from the posterior predictive distribution
    ThompsonSampling,
}

// ── Predictor type ─────────────────────────────────────────────────────────

/// Which surrogate predictor model to use
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictorType {
    /// Gaussian Process with RBF kernel (provides calibrated uncertainty)
    GaussianProcess,
    /// 3-layer MLP regressor trained with SGD on MSE loss
    MlpRegressor,
}

// ── Configuration ──────────────────────────────────────────────────────────

/// Configuration for predictor-based NAS
#[derive(Debug, Clone)]
pub struct PredictorNasConfig {
    /// Number of random architectures to evaluate before fitting the predictor
    pub n_initial_architectures: usize,
    /// Number of Bayesian optimization iterations after the initial phase
    pub n_iterations: usize,
    /// Number of random candidates generated per iteration for acquisition scoring
    pub n_candidates_per_iter: usize,
    /// Acquisition function to use
    pub acquisition: AcquisitionFunction,
    /// Surrogate predictor type
    pub predictor_type: PredictorType,
    /// Exploration parameter kappa for the UCB acquisition function
    pub ucb_kappa: f64,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for PredictorNasConfig {
    fn default() -> Self {
        Self {
            n_initial_architectures: 10,
            n_iterations: 20,
            n_candidates_per_iter: 100,
            acquisition: AcquisitionFunction::ExpectedImprovement,
            predictor_type: PredictorType::GaussianProcess,
            ucb_kappa: 2.0,
            seed: 42,
        }
    }
}

// ── Architecture feature extractor ─────────────────────────────────────────

/// Converts a discrete architecture (op indices) to a flat one-hot feature vector.
///
/// For an architecture with `n_edges` edges and `n_ops` candidate operations,
/// the feature vector has length `n_edges * n_ops`.
/// Feature `[edge * n_ops + op]` is 1.0 iff edge `edge` uses operation `op`.
pub struct ArchFeatureExtractor {
    /// Number of edges in the search DAG
    pub n_edges: usize,
    /// Number of candidate operations per edge
    pub n_ops: usize,
}

impl ArchFeatureExtractor {
    /// Create a new feature extractor
    pub fn new(n_edges: usize, n_ops: usize) -> Self {
        Self { n_edges, n_ops }
    }

    /// Encode an architecture as a one-hot feature vector.
    ///
    /// `arch[i]` is the operation index for edge `i`. The returned vector
    /// has length `n_edges * n_ops`.
    pub fn encode(&self, arch: &[usize]) -> Vec<f64> {
        let mut features = vec![0.0_f64; self.feature_dim()];
        for (edge_idx, &op_idx) in arch.iter().enumerate().take(self.n_edges) {
            let clamped_op = op_idx.min(self.n_ops.saturating_sub(1));
            features[edge_idx * self.n_ops + clamped_op] = 1.0;
        }
        features
    }

    /// Dimensionality of the feature vector
    pub fn feature_dim(&self) -> usize {
        self.n_edges * self.n_ops
    }
}

// ── Gaussian Process predictor ─────────────────────────────────────────────

/// Gaussian Process surrogate predictor with RBF (squared-exponential) kernel.
///
/// Kernel: k(x, x') = exp(-||x - x'||^2 / (2 * l^2))
///
/// The GP is fitted by computing the Cholesky factorization of the kernel matrix
/// K + sigma^2 * I and solving for the dual coefficients alpha = (K + sigma^2 I)^{-1} y.
pub struct GpPredictor {
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    /// RBF kernel length scale
    length_scale: f64,
    /// Observation noise variance
    noise_var: f64,
    /// Dual coefficients: (K + sigma^2 I)^{-1} y
    dual_alpha: Vec<f64>,
    /// Lower Cholesky factor L such that L L^T = K + sigma^2 I
    chol_l: Vec<Vec<f64>>,
}

impl GpPredictor {
    /// Create a new GP predictor with given length scale and noise variance.
    pub fn new(length_scale: f64, noise_var: f64) -> Self {
        Self {
            x_train: Vec::new(),
            y_train: Vec::new(),
            length_scale,
            noise_var,
            dual_alpha: Vec::new(),
            chol_l: Vec::new(),
        }
    }

    /// RBF kernel value between two feature vectors.
    fn rbf_kernel(&self, a: &[f64], b: &[f64]) -> f64 {
        let sq_dist: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi) * (ai - bi))
            .sum();
        let two_l2 = 2.0 * self.length_scale * self.length_scale;
        (-sq_dist / two_l2).exp()
    }

    /// Fit the GP on a set of (feature, score) observations.
    ///
    /// Computes the Cholesky decomposition of K + sigma^2 I and stores
    /// the dual coefficients for fast prediction.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        let n = x.len();
        if n == 0 || n != y.len() {
            return;
        }
        self.x_train = x.to_vec();
        self.y_train = y.to_vec();

        // Build kernel matrix K
        let mut k_mat = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                k_mat[i][j] = self.rbf_kernel(&x[i], &x[j]);
            }
            k_mat[i][i] += self.noise_var;
        }

        // Cholesky decomposition: L L^T = K + sigma^2 I
        self.chol_l = cholesky_decompose(&k_mat);

        // Solve (K + sigma^2 I) alpha = y  via  L L^T alpha = y
        self.dual_alpha = cholesky_solve(&self.chol_l, y);
    }

    /// Predict mean and standard deviation at a new point.
    ///
    /// Returns `(mean, std)`.
    ///
    /// The variance is approximated as:
    /// var ≈ k(x,x) - k^T * (K + sigma^2 I)^{-1} * k
    ///       = 1 - ||L^{-1} k||^2
    pub fn predict(&self, x: &[f64]) -> (f64, f64) {
        if self.x_train.is_empty() {
            return (0.0, 1.0);
        }

        // Compute kernel vector k_star = [k(x, x_train_i)]
        let k_star: Vec<f64> = self
            .x_train
            .iter()
            .map(|xi| self.rbf_kernel(x, xi))
            .collect();

        // Mean: mu = k_star^T * dual_alpha
        let mean: f64 = k_star
            .iter()
            .zip(self.dual_alpha.iter())
            .map(|(&ki, &ai)| ki * ai)
            .sum();

        // Variance: var = k(x,x) - ||L^{-1} k_star||^2
        let k_xx = self.rbf_kernel(x, x);
        let v = forward_solve(&self.chol_l, &k_star);
        let v_sq_norm: f64 = v.iter().map(|&vi| vi * vi).sum();
        let variance = (k_xx - v_sq_norm).max(1e-10);
        let std_dev = variance.sqrt();

        (mean, std_dev)
    }

    /// Number of training points
    pub fn n_train(&self) -> usize {
        self.x_train.len()
    }
}

// ── MLP predictor ──────────────────────────────────────────────────────────

/// 3-layer MLP regression predictor.
///
/// Architecture: input -> hidden (ReLU) -> hidden (ReLU) -> 1 (linear)
///
/// Trained with mini-batch SGD on MSE loss using numerical gradients.
/// Does not provide calibrated uncertainty (always returns fixed std=0.1).
pub struct MlpPredictor {
    /// First layer weights: shape [hidden_size x input_dim]
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    /// Second layer weights: shape [hidden_size x hidden_size]
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    /// Output layer weights: shape [hidden_size]
    w3: Vec<f64>,
    b3: f64,
    hidden_size: usize,
}

impl MlpPredictor {
    /// Create a new MLP predictor with random weight initialization (He init).
    pub fn new(input_dim: usize, hidden_size: usize, rng: &mut impl Rng) -> Self {
        let w1 = random_matrix(hidden_size, input_dim, rng, (2.0 / input_dim as f64).sqrt());
        let b1 = vec![0.0_f64; hidden_size];
        let w2 = random_matrix(hidden_size, hidden_size, rng, (2.0 / hidden_size as f64).sqrt());
        let b2 = vec![0.0_f64; hidden_size];
        let w3 = random_vector(hidden_size, rng, (2.0 / hidden_size as f64).sqrt());
        let b3 = 0.0;

        Self {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            hidden_size,
        }
    }

    /// Compute the scalar output for one input vector (forward pass).
    pub fn forward(&self, x: &[f64]) -> f64 {
        // Layer 1
        let h1: Vec<f64> = (0..self.hidden_size)
            .map(|j| {
                let pre_act: f64 = self.w1[j]
                    .iter()
                    .zip(x.iter())
                    .map(|(&w, &xi)| w * xi)
                    .sum::<f64>()
                    + self.b1[j];
                relu(pre_act)
            })
            .collect();

        // Layer 2
        let h2: Vec<f64> = (0..self.hidden_size)
            .map(|j| {
                let pre_act: f64 = self.w2[j]
                    .iter()
                    .zip(h1.iter())
                    .map(|(&w, &hi)| w * hi)
                    .sum::<f64>()
                    + self.b2[j];
                relu(pre_act)
            })
            .collect();

        // Output layer (linear)
        self.w3
            .iter()
            .zip(h2.iter())
            .map(|(&w, &h)| w * h)
            .sum::<f64>()
            + self.b3
    }

    /// Train the MLP on a dataset using SGD with numerical gradients.
    ///
    /// Uses finite differences (epsilon=1e-5) to compute gradients.
    /// Each epoch iterates over all samples in order.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64], n_epochs: usize, lr: f64) {
        if x.is_empty() || x.len() != y.len() {
            return;
        }
        let eps = 1e-5;
        let n = x.len();

        for _ in 0..n_epochs {
            for sample_idx in 0..n {
                let xi = &x[sample_idx];
                let yi = y[sample_idx];
                let pred = self.forward(xi);
                let loss_scale = 2.0 * (pred - yi); // d(MSE)/d(pred)

                // Numerical gradient for w3 and b3
                for k in 0..self.w3.len() {
                    let orig = self.w3[k];
                    self.w3[k] = orig + eps;
                    let p_plus = self.forward(xi);
                    self.w3[k] = orig - eps;
                    let p_minus = self.forward(xi);
                    self.w3[k] = orig;
                    let grad = loss_scale * (p_plus - p_minus) / (2.0 * eps);
                    self.w3[k] -= lr * grad;
                }
                {
                    let orig = self.b3;
                    self.b3 = orig + eps;
                    let p_plus = self.forward(xi);
                    self.b3 = orig - eps;
                    let p_minus = self.forward(xi);
                    self.b3 = orig;
                    let grad = loss_scale * (p_plus - p_minus) / (2.0 * eps);
                    self.b3 -= lr * grad;
                }

                // Numerical gradient for w2 and b2
                for j in 0..self.hidden_size {
                    for k in 0..self.hidden_size {
                        let orig = self.w2[j][k];
                        self.w2[j][k] = orig + eps;
                        let p_plus = self.forward(xi);
                        self.w2[j][k] = orig - eps;
                        let p_minus = self.forward(xi);
                        self.w2[j][k] = orig;
                        let grad = loss_scale * (p_plus - p_minus) / (2.0 * eps);
                        self.w2[j][k] -= lr * grad;
                    }
                    let orig = self.b2[j];
                    self.b2[j] = orig + eps;
                    let p_plus = self.forward(xi);
                    self.b2[j] = orig - eps;
                    let p_minus = self.forward(xi);
                    self.b2[j] = orig;
                    let grad = loss_scale * (p_plus - p_minus) / (2.0 * eps);
                    self.b2[j] -= lr * grad;
                }

                // Numerical gradient for w1 and b1
                for j in 0..self.hidden_size {
                    for k in 0..xi.len() {
                        let orig = self.w1[j][k];
                        self.w1[j][k] = orig + eps;
                        let p_plus = self.forward(xi);
                        self.w1[j][k] = orig - eps;
                        let p_minus = self.forward(xi);
                        self.w1[j][k] = orig;
                        let grad = loss_scale * (p_plus - p_minus) / (2.0 * eps);
                        self.w1[j][k] -= lr * grad;
                    }
                    let orig = self.b1[j];
                    self.b1[j] = orig + eps;
                    let p_plus = self.forward(xi);
                    self.b1[j] = orig - eps;
                    let p_minus = self.forward(xi);
                    self.b1[j] = orig;
                    let grad = loss_scale * (p_plus - p_minus) / (2.0 * eps);
                    self.b1[j] -= lr * grad;
                }
            }
        }
    }

    /// Predict mean and a fixed standard deviation (MLP has no uncertainty).
    ///
    /// Returns `(prediction, 0.1)`.
    pub fn predict(&self, x: &[f64]) -> (f64, f64) {
        (self.forward(x), 0.1)
    }

    /// Number of hidden units
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

// ── Predictor NAS search ───────────────────────────────────────────────────

/// Predictor-based NAS search engine.
///
/// Maintains a dataset of evaluated architectures and a surrogate predictor.
/// At each iteration, the predictor is fitted on the dataset, and new
/// candidate architectures are ranked by an acquisition function.
pub struct PredictorNasSearch {
    config: PredictorNasConfig,
    extractor: ArchFeatureExtractor,
    /// History of evaluated architectures and their validation scores
    evaluated: Vec<(Vec<usize>, f64)>,
    rng: ChaCha20Rng,
}

impl PredictorNasSearch {
    /// Create a new predictor-based NAS search.
    pub fn new(config: PredictorNasConfig, n_edges: usize, n_ops: usize) -> Self {
        let seed = config.seed;
        Self {
            extractor: ArchFeatureExtractor::new(n_edges, n_ops),
            evaluated: Vec::new(),
            rng: ChaCha20Rng::seed_from_u64(seed),
            config,
        }
    }

    /// Record a completed architecture evaluation.
    pub fn record_evaluation(&mut self, arch: Vec<usize>, score: f64) {
        self.evaluated.push((arch, score));
    }

    /// Number of architectures evaluated so far.
    pub fn n_evaluated(&self) -> usize {
        self.evaluated.len()
    }

    /// Return the architecture with the highest score recorded so far.
    pub fn best_architecture(&self) -> Option<(&Vec<usize>, f64)> {
        self.evaluated
            .iter()
            .max_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(arch, score)| (arch, *score))
    }

    /// Propose the next `n_proposals` architectures to evaluate.
    ///
    /// If fewer than 2 architectures have been evaluated the predictor cannot
    /// be fitted, so random proposals are returned.
    ///
    /// Otherwise:
    ///  1. Fit the predictor on evaluated data.
    ///  2. Sample `n_candidates_per_iter` random architectures.
    ///  3. Score each candidate using the acquisition function.
    ///  4. Return the top `n_proposals` candidates.
    pub fn propose_next_architectures(&mut self, n_proposals: usize) -> Result<Vec<Vec<usize>>> {
        let n_edges = self.extractor.n_edges;
        let n_ops = self.extractor.n_ops;

        // Not enough data to fit predictor – return random proposals
        if self.evaluated.len() < 2 {
            return Ok(self.random_architectures(n_proposals));
        }

        // Build training data for the predictor
        let x_train: Vec<Vec<f64>> = self
            .evaluated
            .iter()
            .map(|(arch, _)| self.extractor.encode(arch))
            .collect();
        let y_train: Vec<f64> = self.evaluated.iter().map(|(_, s)| *s).collect();

        let best_score = y_train
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        // Generate candidate architectures
        let n_candidates = self.config.n_candidates_per_iter.max(n_proposals);
        let candidates = self.random_architectures(n_candidates);

        // Score candidates using the acquisition function
        let feature_dim = self.extractor.feature_dim();
        let scored: Vec<(f64, Vec<usize>)> = match self.config.predictor_type {
            PredictorType::GaussianProcess => {
                let mut gp = GpPredictor::new(1.0, 0.01);
                gp.fit(&x_train, &y_train);
                candidates
                    .into_iter()
                    .map(|arch| {
                        let feat = self.extractor.encode(&arch);
                        let (mu, sigma) = gp.predict(&feat);
                        let acq = self.acquisition_score(
                            mu,
                            sigma,
                            best_score,
                            &mut ChaCha20Rng::seed_from_u64(42),
                        );
                        (acq, arch)
                    })
                    .collect()
            }
            PredictorType::MlpRegressor => {
                let hidden = 32_usize.min(feature_dim.max(4));
                let mut mlp =
                    MlpPredictor::new(feature_dim, hidden, &mut self.rng.clone());
                mlp.fit(&x_train, &y_train, 20, 1e-3);
                candidates
                    .into_iter()
                    .map(|arch| {
                        let feat = self.extractor.encode(&arch);
                        let (mu, sigma) = mlp.predict(&feat);
                        let acq = self.acquisition_score(
                            mu,
                            sigma,
                            best_score,
                            &mut ChaCha20Rng::seed_from_u64(42),
                        );
                        (acq, arch)
                    })
                    .collect()
            }
        };

        // Sort by acquisition score descending and take top n_proposals
        let mut scored = scored;
        scored.sort_by(|(s1, _), (s2, _)| s2.partial_cmp(s1).unwrap_or(std::cmp::Ordering::Equal));

        let proposals: Vec<Vec<usize>> = scored
            .into_iter()
            .take(n_proposals)
            .map(|(_, arch)| arch)
            .collect();

        if proposals.is_empty() {
            return Err(NeuralError::ComputationError(
                "No proposals generated — check n_candidates_per_iter".to_string(),
            ));
        }

        Ok(proposals)
    }

    // ── Private helpers ──────────────────────────────────────────────────

    /// Sample `n` random architectures uniformly.
    fn random_architectures(&mut self, n: usize) -> Vec<Vec<usize>> {
        let n_edges = self.extractor.n_edges;
        let n_ops = self.extractor.n_ops;
        (0..n)
            .map(|_| {
                (0..n_edges)
                    .map(|_| self.rng.random_range(0..n_ops))
                    .collect()
            })
            .collect()
    }

    /// Compute acquisition function score for a candidate with predicted (mu, sigma).
    fn acquisition_score(
        &self,
        mu: f64,
        sigma: f64,
        best: f64,
        rng: &mut impl Rng,
    ) -> f64 {
        match self.config.acquisition {
            AcquisitionFunction::ExpectedImprovement => Self::expected_improvement(mu, sigma, best),
            AcquisitionFunction::UpperConfidenceBound => mu + self.config.ucb_kappa * sigma,
            AcquisitionFunction::ThompsonSampling => {
                // Sample from N(mu, sigma^2)
                let u: f64 = rng.random();
                let u2: f64 = rng.random();
                let u_clamped = u.max(1e-40);
                let normal_sample = (-2.0 * u_clamped.ln()).sqrt()
                    * (2.0 * std::f64::consts::PI * u2).cos();
                mu + sigma * normal_sample
            }
        }
    }

    /// Expected Improvement acquisition function.
    ///
    /// EI(x) = (mu - best) * Phi(z) + sigma * phi(z)
    ///         where z = (mu - best) / sigma
    ///
    /// When sigma < 1e-6, returns max(mu - best, 0).
    fn expected_improvement(mu: f64, sigma: f64, best: f64) -> f64 {
        if sigma < 1e-6 {
            return (mu - best).max(0.0);
        }
        let z = (mu - best) / sigma;
        let phi_z = standard_normal_pdf(z);
        let big_phi_z = standard_normal_cdf(z);
        (mu - best) * big_phi_z + sigma * phi_z
    }
}

// ── Normal distribution helpers ────────────────────────────────────────────

/// Standard normal PDF: phi(z) = exp(-z^2/2) / sqrt(2*pi)
fn standard_normal_pdf(z: f64) -> f64 {
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    inv_sqrt_2pi * (-0.5 * z * z).exp()
}

/// Standard normal CDF approximation (Abramowitz & Stegun 26.2.17)
fn standard_normal_cdf(z: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let poly = t * (0.319_381_53
        + t * (-0.356_563_782
            + t * (1.781_477_937
                + t * (-1.821_255_978 + t * 1.330_274_429))));
    let phi = 1.0 - standard_normal_pdf(z) * poly;
    if z >= 0.0 {
        phi
    } else {
        1.0 - phi
    }
}

// ── Linear algebra helpers ─────────────────────────────────────────────────

/// In-place Cholesky decomposition (Cholesky-Banachiewicz algorithm).
///
/// Returns the lower triangular factor L such that A = L L^T.
/// If the matrix is not positive definite, a small regularisation is added.
fn cholesky_decompose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut l = vec![vec![0.0_f64; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let diag = a[i][i] - sum;
                // Ensure positive diagonal (add small jitter if needed)
                l[i][j] = diag.max(1e-12).sqrt();
            } else {
                l[i][j] = if l[j][j].abs() < 1e-12 {
                    0.0
                } else {
                    (a[i][j] - sum) / l[j][j]
                };
            }
        }
    }
    l
}

/// Solve L v = b (forward substitution) for lower triangular L.
fn forward_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut v = vec![0.0_f64; n];
    for i in 0..n {
        let sum: f64 = (0..i).map(|k| l[i][k] * v[k]).sum();
        v[i] = if l[i][i].abs() < 1e-12 {
            0.0
        } else {
            (b[i] - sum) / l[i][i]
        };
    }
    v
}

/// Solve L^T v = b (backward substitution) for lower triangular L.
fn backward_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut v = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let sum: f64 = (i + 1..n).map(|k| l[k][i] * v[k]).sum();
        v[i] = if l[i][i].abs() < 1e-12 {
            0.0
        } else {
            (b[i] - sum) / l[i][i]
        };
    }
    v
}

/// Solve (L L^T) x = b using the pre-computed Cholesky factor L.
fn cholesky_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let v = forward_solve(l, b);
    backward_solve(l, &v)
}

// ── Random initialisation helpers ──────────────────────────────────────────

fn random_matrix(rows: usize, cols: usize, rng: &mut impl Rng, scale: f64) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    let u: f64 = rng.random();
                    let u2: f64 = rng.random();
                    let u_clamped = u.max(1e-40);
                    let normal = (-2.0 * u_clamped.ln()).sqrt()
                        * (2.0 * std::f64::consts::PI * u2).cos();
                    scale * normal
                })
                .collect()
        })
        .collect()
}

fn random_vector(len: usize, rng: &mut impl Rng, scale: f64) -> Vec<f64> {
    (0..len)
        .map(|_| {
            let u: f64 = rng.random();
            let u2: f64 = rng.random();
            let u_clamped = u.max(1e-40);
            let normal =
                (-2.0 * u_clamped.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            scale * normal
        })
        .collect()
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    fn make_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    // ── GDAS tests (in predictor.rs as required by spec) ──────────────

    #[test]
    fn test_gdas_gumbel_top1_returns_valid_index() {
        use crate::nas::gdas::{GdasConfig, GdasSearch};
        let config = GdasConfig::default();
        let search = GdasSearch::new(config.clone());
        let mut rng = make_rng(0);
        let logits = vec![1.0, 2.0, 0.5, 3.0, 1.5, 0.0, 2.5, 1.0];
        let (idx, weights) = search.gumbel_top1_sample(&logits, &mut rng).unwrap();
        assert!(idx < config.n_ops);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert_eq!(weights[idx], 1.0);
    }

    #[test]
    fn test_gdas_anneal_temperature_decreases() {
        use crate::nas::gdas::{GdasConfig, GdasSearch, TemperatureSchedule};
        let mut config = GdasConfig::default();
        config.schedule = TemperatureSchedule::Exponential;
        let mut search = GdasSearch::new(config);
        let initial = search.temperature;
        search.anneal_temperature(50);
        assert!(search.temperature < initial);
    }

    #[test]
    fn test_gdas_derive_architecture_argmax() {
        use crate::nas::gdas::{GdasConfig, GdasSearch};
        let config = GdasConfig {
            n_nodes: 3,
            n_ops: 4,
            ..GdasConfig::default()
        };
        let mut search = GdasSearch::new(config);
        search.alpha[0] = vec![0.1, 0.2, 5.0, 0.0];
        search.alpha[1] = vec![3.0, 0.1, 0.2, 0.0];
        search.alpha[2] = vec![0.0, 0.1, 0.2, 7.0];
        let arch = search.derive_architecture();
        assert_eq!(arch, vec![2, 0, 3]);
    }

    #[test]
    fn test_gdas_n_edges_correct() {
        use crate::nas::gdas::{GdasConfig, GdasSearch};
        let config = GdasConfig {
            n_nodes: 4,
            ..GdasConfig::default()
        };
        let search = GdasSearch::new(config);
        assert_eq!(search.n_edges(), 6); // 4 * 3 / 2 = 6
    }

    // ── SNAS tests ─────────────────────────────────────────────────────

    #[test]
    fn test_snas_gumbel_softmax_sums_to_one() {
        use crate::nas::snas::{SnasConfig, SnasSearch};
        let config = SnasConfig::default();
        let search = SnasSearch::new(config);
        let mut rng = make_rng(42);
        let logits = vec![1.0, 0.5, -1.0, 2.0, 0.0, -0.5, 1.5, 0.3];
        let weights = search.gumbel_softmax_sample(&logits, 1.0, &mut rng).unwrap();
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_snas_kl_divergence_uniform_alpha_near_zero() {
        use crate::nas::snas::{SnasConfig, SnasSearch};
        let config = SnasConfig::default();
        let search = SnasSearch::new(config);
        let kl = search.kl_divergence_from_uniform(0).unwrap();
        assert!(kl < 1e-10, "KL from uniform alpha should be ~0, got {kl}");
    }

    #[test]
    fn test_snas_kl_divergence_peaked_positive() {
        use crate::nas::snas::{SnasConfig, SnasSearch};
        let config = SnasConfig {
            n_nodes: 3,
            n_ops: 4,
            ..SnasConfig::default()
        };
        let mut search = SnasSearch::new(config);
        search.alpha[0] = vec![20.0, -10.0, -10.0, -10.0];
        let kl = search.kl_divergence_from_uniform(0).unwrap();
        assert!(kl > 0.1, "Peaked distribution should have positive KL, got {kl}");
    }

    #[test]
    fn test_snas_architecture_entropy_uniform_max() {
        use crate::nas::snas::{SnasConfig, SnasSearch};
        let config = SnasConfig {
            n_nodes: 3,
            n_ops: 8,
            ..SnasConfig::default()
        };
        let search = SnasSearch::new(config);
        let entropy = search.architecture_entropy();
        let expected = (8.0_f64).ln();
        assert!((entropy - expected).abs() < 1e-6);
    }

    // ── ArchFeatureExtractor tests ──────────────────────────────────────

    #[test]
    fn test_arch_feature_extractor_length() {
        let extractor = ArchFeatureExtractor::new(6, 8);
        assert_eq!(extractor.feature_dim(), 48);
        let arch = vec![0, 1, 2, 3, 4, 5];
        let features = extractor.encode(&arch);
        assert_eq!(features.len(), 48);
    }

    #[test]
    fn test_arch_feature_extractor_deterministic() {
        let extractor = ArchFeatureExtractor::new(3, 4);
        let arch = vec![1, 0, 3];
        let f1 = extractor.encode(&arch);
        let f2 = extractor.encode(&arch);
        assert_eq!(f1, f2);
    }

    #[test]
    fn test_arch_feature_extractor_distinct_architectures() {
        let extractor = ArchFeatureExtractor::new(3, 4);
        let arch1 = vec![0, 1, 2];
        let arch2 = vec![1, 0, 2];
        let f1 = extractor.encode(&arch1);
        let f2 = extractor.encode(&arch2);
        assert_ne!(f1, f2, "Different architectures should produce different features");
    }

    #[test]
    fn test_arch_feature_extractor_one_hot_property() {
        let extractor = ArchFeatureExtractor::new(4, 5);
        let arch = vec![2, 0, 4, 1];
        let features = extractor.encode(&arch);
        // Each edge-block should have exactly one 1.0
        for edge in 0..4 {
            let block = &features[edge * 5..(edge + 1) * 5];
            let sum: f64 = block.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "One-hot property failed at edge {edge}");
        }
    }

    // ── GP predictor tests ──────────────────────────────────────────────

    #[test]
    fn test_gp_predictor_fit_and_predict() {
        let mut gp = GpPredictor::new(1.0, 0.01);
        let x = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let y = vec![0.0, 1.0, 1.0, 2.0];
        gp.fit(&x, &y);
        let (mean, std) = gp.predict(&[0.5, 0.5]);
        assert!(std > 0.0, "Standard deviation should be positive");
        // Mean should be roughly in the range of training labels
        assert!(mean > -0.5 && mean < 2.5, "Mean {mean} should be in plausible range");
    }

    #[test]
    fn test_gp_predictor_variance_at_training_points_small() {
        let mut gp = GpPredictor::new(1.0, 1e-6);
        let x = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y = vec![1.0, 2.0, 3.0];
        gp.fit(&x, &y);
        // At training points, uncertainty should be small
        let (_, std_at_train) = gp.predict(&[0.0]);
        assert!(std_at_train < 0.5, "Variance at training point should be small, got {std_at_train}");
    }

    // ── MLP predictor tests ─────────────────────────────────────────────

    #[test]
    fn test_mlp_predictor_overfit() {
        let mut rng = make_rng(0);
        let input_dim = 4;
        let mut mlp = MlpPredictor::new(input_dim, 16, &mut rng);
        let x = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let y = vec![1.0, 2.0, 3.0];
        mlp.fit(&x, &y, 200, 0.01);
        for (xi, &yi) in x.iter().zip(y.iter()) {
            let (pred, _) = mlp.predict(xi);
            assert!(
                (pred - yi).abs() < 1.0,
                "MLP should approximately fit training data: pred={pred:.3}, target={yi}"
            );
        }
    }

    // ── Acquisition function tests ──────────────────────────────────────

    #[test]
    fn test_expected_improvement_at_best_zero() {
        // When mu == best and sigma is tiny, EI should be ~0
        let ei = PredictorNasSearch::expected_improvement(1.0, 1e-9, 1.0);
        assert!(ei.abs() < 1e-6, "EI at current best with tiny sigma should be ~0, got {ei}");
    }

    #[test]
    fn test_expected_improvement_positive_with_uncertainty() {
        // When mu > best, EI should be positive
        let ei = PredictorNasSearch::expected_improvement(2.0, 0.5, 1.0);
        assert!(ei > 0.0, "EI should be positive when mu > best, got {ei}");
    }

    #[test]
    fn test_expected_improvement_zero_sigma_clamp() {
        // sigma < 1e-6 uses simplified formula: max(mu - best, 0)
        let ei_above = PredictorNasSearch::expected_improvement(1.5, 1e-8, 1.0);
        assert!((ei_above - 0.5).abs() < 1e-6, "Expected 0.5, got {ei_above}");
        let ei_below = PredictorNasSearch::expected_improvement(0.5, 1e-8, 1.0);
        assert!(ei_below == 0.0, "EI should be 0 when mu < best with tiny sigma");
    }

    // ── PredictorNasSearch tests ────────────────────────────────────────

    #[test]
    fn test_predictor_nas_record_evaluation() {
        let config = PredictorNasConfig::default();
        let mut search = PredictorNasSearch::new(config, 6, 8);
        assert_eq!(search.n_evaluated(), 0);
        search.record_evaluation(vec![0, 1, 2, 3, 4, 5], 0.75);
        assert_eq!(search.n_evaluated(), 1);
    }

    #[test]
    fn test_predictor_nas_best_architecture() {
        let config = PredictorNasConfig::default();
        let mut search = PredictorNasSearch::new(config, 3, 4);
        search.record_evaluation(vec![0, 1, 2], 0.80);
        search.record_evaluation(vec![1, 0, 3], 0.95);
        search.record_evaluation(vec![2, 2, 1], 0.70);
        let (best_arch, best_score) = search.best_architecture().unwrap();
        assert!((best_score - 0.95).abs() < 1e-10);
        assert_eq!(best_arch, &vec![1, 0, 3]);
    }

    #[test]
    fn test_predictor_nas_propose_returns_valid_arches() {
        let config = PredictorNasConfig {
            predictor_type: PredictorType::GaussianProcess,
            n_candidates_per_iter: 20,
            ..PredictorNasConfig::default()
        };
        let mut search = PredictorNasSearch::new(config, 4, 5);
        // Seed with a few evaluations
        for i in 0..5 {
            let arch: Vec<usize> = (0..4).map(|j| (i + j) % 5).collect();
            search.record_evaluation(arch, 0.5 + i as f64 * 0.05);
        }
        let proposals = search.propose_next_architectures(3).unwrap();
        assert_eq!(proposals.len(), 3);
        for arch in &proposals {
            assert_eq!(arch.len(), 4);
            for &op in arch {
                assert!(op < 5, "Operation index {op} out of range for n_ops=5");
            }
        }
    }

    #[test]
    fn test_predictor_nas_n_evaluated_increases() {
        let config = PredictorNasConfig::default();
        let mut search = PredictorNasSearch::new(config, 3, 4);
        for i in 0..10 {
            search.record_evaluation(vec![i % 4, (i + 1) % 4, (i + 2) % 4], i as f64 * 0.1);
        }
        assert_eq!(search.n_evaluated(), 10);
    }

    #[test]
    fn test_predictor_nas_no_data_returns_random() {
        let config = PredictorNasConfig {
            n_candidates_per_iter: 10,
            ..PredictorNasConfig::default()
        };
        let mut search = PredictorNasSearch::new(config, 4, 5);
        // With no evaluations, should fall back to random proposals
        let proposals = search.propose_next_architectures(5).unwrap();
        assert_eq!(proposals.len(), 5);
    }
}
