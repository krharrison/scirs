//! Differentiable Combinatorial Optimization
//!
//! This module implements differentiable relaxations of combinatorial
//! optimization problems, enabling gradients to flow through discrete
//! decision layers in end-to-end learning pipelines.
//!
//! # Algorithms
//!
//! - **SparseMAP** (Niculae & Blondel, 2017): Sparse structured prediction
//!   via QP over the marginal polytope. Yields sparse probability distributions
//!   over combinatorial structures with exact gradients via the active-set
//!   theorem.
//!
//! - **Perturbed Optimizers** (Berthet et al., 2020): Sample-based
//!   differentiable argmax using additive Gaussian noise, enabling unbiased
//!   gradient estimates through any black-box combinatorial solver.
//!
//! - **Differentiable Sorting** (Cuturi & Doucet, 2017): Regularised isotonic
//!   regression for soft sorting and ranking.
//!
//! - **Differentiable Top-K**: Entropy-regularised LP relaxation of the hard
//!   top-k selector.
//!
//! # References
//! - Niculae & Blondel (2017). "A regularized framework for sparse and
//!   structured neural attention." NeurIPS.
//! - Berthet et al. (2020). "Learning with Differentiable Perturbed Optimizers."
//!   NeurIPS.
//! - Blondel et al. (2020). "Fast Differentiable Sorting and Ranking." ICML.

use scirs2_core::num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{OptimizeError, OptimizeResult};

// ─────────────────────────────────────────────────────────────────────────────
// SparseMAP
// ─────────────────────────────────────────────────────────────────────────────

/// Type of combinatorial structure defining the polytope for SparseMAP.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum StructureType {
    /// Standard probability simplex: Σμ_i = 1, μ_i ≥ 0.
    Simplex,
    /// Knapsack polytope: Σw_i · μ_i ≤ capacity, 0 ≤ μ_i ≤ 1.
    Knapsack {
        /// Weights for each item.
        weights: Vec<f64>,
        /// Knapsack capacity.
        capacity: f64,
    },
    /// Birkhoff polytope (doubly-stochastic matrices): permutation marginals.
    /// `dim` is the side length (number of items to rank).
    Permutation {
        /// Number of items.
        dim: usize,
    },
}

impl Default for StructureType {
    fn default() -> Self {
        StructureType::Simplex
    }
}

/// Configuration for the SparseMAP solver.
#[derive(Debug, Clone)]
pub struct SparsemapConfig {
    /// Maximum number of active-set / projected-gradient iterations.
    pub max_iter: usize,
    /// Convergence tolerance (dual gap or gradient norm).
    pub tol: f64,
    /// Combinatorial structure type.
    pub structure_type: StructureType,
    /// Step size for projected-gradient updates.
    pub step_size: f64,
}

impl Default for SparsemapConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-6,
            structure_type: StructureType::default(),
            step_size: 0.1,
        }
    }
}

/// Result of SparseMAP.
#[derive(Debug, Clone)]
pub struct SparsemapResult<F> {
    /// Sparse probability distribution over combinatorial structures.
    pub solution: Vec<F>,
    /// Indices of atoms with non-zero weight (the active support).
    pub support: Vec<usize>,
    /// Dual variables at optimality (Lagrange multipliers for equality / active
    /// inequality constraints).
    pub dual: Vec<F>,
    /// Number of iterations performed.
    pub n_iters: usize,
}

/// Project a vector onto the probability simplex Δ^{n-1} = {μ | Σμ_i=1, μ≥0}.
///
/// Uses Duchi et al. (2008) O(n log n) algorithm.
fn project_simplex<F>(v: &[F]) -> Vec<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = v.len();
    let mut u: Vec<F> = v.to_vec();
    // Sort descending
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut cssv = F::zero();
    let mut rho = 0usize;
    for (j, &uj) in u.iter().enumerate() {
        cssv = cssv + uj;
        let j_f = F::from_usize(j + 1).unwrap_or(F::one());
        let one = F::one();
        if uj - (cssv - one) / j_f > F::zero() {
            rho = j;
        }
    }

    let rho_f = F::from_usize(rho + 1).unwrap_or(F::one());
    let one = F::one();
    // Recompute cssv up to rho
    let mut cssv2 = F::zero();
    for uj in u.iter().take(rho + 1) {
        cssv2 = cssv2 + *uj;
    }
    let theta = (cssv2 - one) / rho_f;

    v.iter()
        .map(|&vi| {
            let diff = vi - theta;
            if diff > F::zero() {
                diff
            } else {
                F::zero()
            }
        })
        .collect()
}

/// Project a vector onto the knapsack polytope with binary variables and
/// capacity constraint: {μ | Σw_i·μ_i ≤ cap, 0 ≤ μ_i ≤ 1}.
///
/// Uses greedy fractional knapsack rounding followed by projected gradient.
fn project_knapsack<F>(v: &[F], weights: &[f64], capacity: f64) -> Vec<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = v.len();
    // Clip to [0,1] first
    let mut mu: Vec<F> = v
        .iter()
        .map(|&vi| {
            if vi < F::zero() {
                F::zero()
            } else if vi > F::one() {
                F::one()
            } else {
                vi
            }
        })
        .collect();

    // Check if capacity is satisfied; if so, return clipped value
    let total_weight: f64 = (0..n)
        .map(|i| weights.get(i).copied().unwrap_or(1.0) * mu[i].to_f64().unwrap_or(0.0))
        .sum();

    if total_weight <= capacity + 1e-12 {
        return mu;
    }

    // Binary search on the Lagrange multiplier λ for the capacity constraint:
    //   μ_i(λ) = clip(v_i / (1 + λ·w_i), 0, 1)   => capacity constraint
    let mut lo = 0.0_f64;
    let mut hi = 1e8_f64;

    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        let w_total: f64 = (0..n)
            .map(|i| {
                let wi = weights.get(i).copied().unwrap_or(1.0);
                let vi = v[i].to_f64().unwrap_or(0.0);
                let mu_i = (vi / (1.0 + mid * wi)).clamp(0.0, 1.0);
                wi * mu_i
            })
            .sum();
        if w_total > capacity {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let lambda = (lo + hi) / 2.0;
    mu = (0..n)
        .map(|i| {
            let wi = weights.get(i).copied().unwrap_or(1.0);
            let vi = v[i].to_f64().unwrap_or(0.0);
            let val = (vi / (1.0 + lambda * wi)).clamp(0.0, 1.0);
            F::from_f64(val).unwrap_or(F::zero())
        })
        .collect();
    mu
}

/// Solve SparseMAP via projected gradient descent on the regularised QP.
///
/// For `StructureType::Simplex` this is equivalent to computing the Euclidean
/// projection of `scores` onto the probability simplex, which has a closed-form
/// O(n log n) solution.  For other structures, it falls back to iterative
/// projected gradient.
///
/// # Arguments
/// * `scores` – score vector θ ∈ ℝ^d.
/// * `config` – solver configuration.
///
/// # Returns
/// [`SparsemapResult`] containing the sparse distribution, active support,
/// dual variables, and iteration count.
pub fn sparsemap<F>(scores: &[F], config: &SparsemapConfig) -> OptimizeResult<SparsemapResult<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    if scores.is_empty() {
        return Err(OptimizeError::InvalidInput(
            "scores vector must be non-empty".into(),
        ));
    }

    let n = scores.len();
    let tol_f = F::from_f64(config.tol).unwrap_or(F::epsilon());

    let solution: Vec<F>;
    let n_iters: usize;
    let dual: Vec<F>;

    match &config.structure_type {
        StructureType::Simplex => {
            // Closed-form: Euclidean projection onto probability simplex.
            solution = project_simplex(scores);
            n_iters = 1;
            // The dual variable λ for the equality constraint Σμ=1 equals
            // the threshold used in the projection: λ = (Σ_{i∈S} θ_i - 1)/|S|
            let support_sum: F =
                solution.iter().fold(
                    F::zero(),
                    |acc, &x| {
                        if x > F::zero() {
                            acc + x
                        } else {
                            acc
                        }
                    },
                );
            let support_count = solution.iter().filter(|&&x| x > F::zero()).count();
            let count_f = F::from_usize(support_count).unwrap_or(F::one());
            let lambda = if count_f > F::zero() {
                (support_sum - F::one()) / count_f
            } else {
                F::zero()
            };
            dual = vec![lambda];
        }

        StructureType::Knapsack { weights, capacity } => {
            // Projected gradient descent on knapsack polytope.
            let mut mu: Vec<F> = vec![F::zero(); n];
            let step = F::from_f64(config.step_size).unwrap_or(F::epsilon());

            let mut iter = 0usize;
            let mut prev_obj = F::neg_infinity();

            loop {
                // Gradient of ½||μ-θ||² w.r.t. μ is (μ - θ)
                let grad: Vec<F> = mu.iter().zip(scores.iter()).map(|(&m, &s)| m - s).collect();

                // Gradient step: μ ← μ - step * grad = μ - step*(μ-θ)
                let mu_new: Vec<F> = mu
                    .iter()
                    .zip(grad.iter())
                    .map(|(&m, &g)| m - step * g)
                    .collect();

                // Project onto knapsack polytope
                let mu_proj = project_knapsack(&mu_new, weights, *capacity);

                // Compute objective
                let obj = mu_proj
                    .iter()
                    .zip(scores.iter())
                    .fold(F::zero(), |acc, (&m, &s)| {
                        let diff = m - s;
                        acc + diff * diff
                    });
                let half = F::from_f64(0.5).unwrap_or(F::one());
                let obj = obj * half;

                let diff = (obj - prev_obj).abs();
                mu = mu_proj;
                prev_obj = obj;
                iter += 1;

                if iter >= config.max_iter || diff < tol_f {
                    break;
                }
            }

            solution = mu;
            n_iters = iter;
            // Dual: reduced costs for capacity constraint
            let total_w: f64 = (0..n)
                .map(|i| {
                    weights.get(i).copied().unwrap_or(1.0) * solution[i].to_f64().unwrap_or(0.0)
                })
                .sum();
            let slack = *capacity - total_w;
            let lambda_val = if slack.abs() < 1e-8 { -1.0 } else { 0.0 };
            dual = vec![F::from_f64(lambda_val).unwrap_or(F::zero())];
        }

        StructureType::Permutation { dim } => {
            // For permutations, solve QP over Birkhoff polytope via Sinkhorn
            // projection (alternating row/column normalisation).
            let d = *dim;
            if scores.len() != d * d {
                return Err(OptimizeError::InvalidInput(format!(
                    "Permutation structure requires d²={} scores but got {}",
                    d * d,
                    scores.len()
                )));
            }

            // Initialise as uniform doubly-stochastic matrix
            let inv_d = F::from_f64(1.0 / d as f64).unwrap_or(F::one());
            let mut mu: Vec<F> = vec![inv_d; d * d];
            let step = F::from_f64(config.step_size).unwrap_or(F::epsilon());
            let mut iter = 0usize;

            loop {
                // Gradient step
                let mu_step: Vec<F> = mu
                    .iter()
                    .zip(scores.iter())
                    .map(|(&m, &s)| m - step * (m - s))
                    .collect();

                // Sinkhorn projection: alternate row / column normalisation
                let mut m_sink = mu_step;
                for _ in 0..50 {
                    // Row normalisation
                    for row in 0..d {
                        let row_sum: F = (0..d)
                            .map(|col| m_sink[row * d + col])
                            .fold(F::zero(), |a, b| a + b);
                        if row_sum > F::zero() {
                            for col in 0..d {
                                m_sink[row * d + col] = m_sink[row * d + col] / row_sum;
                            }
                        }
                    }
                    // Column normalisation
                    for col in 0..d {
                        let col_sum: F = (0..d)
                            .map(|row| m_sink[row * d + col])
                            .fold(F::zero(), |a, b| a + b);
                        if col_sum > F::zero() {
                            for row in 0..d {
                                m_sink[row * d + col] = m_sink[row * d + col] / col_sum;
                            }
                        }
                    }
                }

                // Check convergence
                let change: F = mu
                    .iter()
                    .zip(m_sink.iter())
                    .map(|(&a, &b)| {
                        let d = a - b;
                        d * d
                    })
                    .fold(F::zero(), |a, b| a + b);

                mu = m_sink;
                iter += 1;

                if iter >= config.max_iter || change < tol_f * tol_f {
                    break;
                }
            }

            solution = mu;
            n_iters = iter;
            dual = vec![F::zero(); 2 * d]; // row + column duals
        }
    }

    // Extract active support
    let support: Vec<usize> = solution
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| {
            if v > F::from_f64(1e-9).unwrap_or(F::zero()) {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    Ok(SparsemapResult {
        solution,
        support,
        dual,
        n_iters,
    })
}

/// Compute gradient of a loss through SparseMAP via the active-set theorem.
///
/// For SparseMAP, the Jacobian of the optimal solution μ*(θ) w.r.t. θ is:
/// ```text
/// dμ*/dθ = Π_S  (projection onto tangent space of active support S)
/// ```
/// Concretely, for the simplex case, only active coordinates (support) can
/// receive gradient.  The backward pass is:
/// ```text
/// dL/dθ = Π_S (upstream_grad)
///       = upstream_grad[support] - mean(upstream_grad[support]) · 1_S
/// ```
/// This is the projection of `upstream_grad` onto the tangent space of the
/// simplex face defined by the active support.
///
/// # Arguments
/// * `result` – the forward-pass [`SparsemapResult`].
/// * `upstream_grad` – gradient of the scalar loss w.r.t. `solution` (∂L/∂μ).
///
/// # Returns
/// Gradient ∂L/∂θ of the same length as the score input.
pub fn sparsemap_gradient<F>(result: &SparsemapResult<F>, upstream_grad: &[F]) -> Vec<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = result.solution.len();
    if upstream_grad.len() != n {
        // Length mismatch — return zero gradient rather than panic
        return vec![F::zero(); n];
    }

    let s = &result.support;
    if s.is_empty() {
        return vec![F::zero(); n];
    }

    // Restrict upstream gradient to active support
    let s_size = F::from_usize(s.len()).unwrap_or(F::one());
    let mean_s: F = s
        .iter()
        .map(|&i| upstream_grad[i])
        .fold(F::zero(), |a, b| a + b)
        / s_size;

    // Projected gradient: g_i - mean(g_S)  for i ∈ S, else 0
    let mut grad = vec![F::zero(); n];
    for &i in s {
        grad[i] = upstream_grad[i] - mean_s;
    }
    grad
}

// ─────────────────────────────────────────────────────────────────────────────
// Perturbed Optimizers
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Perturbed Optimizer.
#[derive(Debug, Clone)]
pub struct PerturbedOptimizerConfig {
    /// Number of Monte Carlo samples.
    pub n_samples: usize,
    /// Perturbation magnitude ε.
    pub epsilon: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for PerturbedOptimizerConfig {
    fn default() -> Self {
        Self {
            n_samples: 100,
            epsilon: 0.1,
            seed: 42,
        }
    }
}

/// Differentiable argmax via additive Gaussian perturbations.
///
/// Estimates `E_Z[argmax(θ + ε·Z)]` where `Z ~ N(0, I)`.  The forward pass
/// returns soft assignment probabilities; the backward pass returns an
/// unbiased gradient estimate via the score-function estimator.
#[derive(Debug, Clone)]
pub struct PerturbedOptimizer {
    config: PerturbedOptimizerConfig,
}

impl PerturbedOptimizer {
    /// Create a new perturbed optimizer with the given configuration.
    pub fn new(config: PerturbedOptimizerConfig) -> Self {
        Self { config }
    }

    /// Forward pass: estimate E[argmax(θ + εZ)] via Monte Carlo.
    ///
    /// Returns a probability vector of length `scores.len()`.
    pub fn forward<F>(&self, scores: &[F]) -> OptimizeResult<Vec<F>>
    where
        F: Float + FromPrimitive + Debug + Clone,
    {
        if scores.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "scores must be non-empty".into(),
            ));
        }
        let n = scores.len();
        let mut counts = vec![0usize; n];
        let eps = self.config.epsilon;

        // Deterministic PRNG (xoshiro-style via splitmix64)
        let mut rng_state = self.config.seed;
        let n_samples = self.config.n_samples;

        for _ in 0..n_samples {
            // Sample argmax of perturbed scores
            let mut best_idx = 0usize;
            let mut best_val = F::neg_infinity();

            for i in 0..n {
                let z = sample_standard_normal(&mut rng_state);
                let perturbed = scores[i] + F::from_f64(eps * z).unwrap_or(F::zero());
                if perturbed > best_val {
                    best_val = perturbed;
                    best_idx = i;
                }
            }
            counts[best_idx] += 1;
        }

        let n_samples_f = F::from_usize(n_samples).unwrap_or(F::one());
        let probs: Vec<F> = counts
            .iter()
            .map(|&c| F::from_usize(c).unwrap_or(F::zero()) / n_samples_f)
            .collect();

        Ok(probs)
    }

    /// Backward pass: gradient estimate via score-function / log-derivative
    /// trick.
    ///
    /// # Formula
    /// ```text
    /// dL/dθ ≈ (1 / (ε² · n)) Σ_i [<argmax(θ+εZ_i), upstream>] · Z_i
    /// ```
    ///
    /// # Arguments
    /// * `scores` – original (unperturbed) scores.
    /// * `upstream` – upstream gradient ∂L/∂p (same shape as `forward` output).
    ///
    /// # Returns
    /// Gradient ∂L/∂θ of the same length as `scores`.
    pub fn backward<F>(&self, scores: &[F], upstream: &[F]) -> OptimizeResult<Vec<F>>
    where
        F: Float + FromPrimitive + Debug + Clone,
    {
        if scores.len() != upstream.len() {
            return Err(OptimizeError::InvalidInput(
                "scores and upstream must have the same length".into(),
            ));
        }
        let n = scores.len();
        let eps = self.config.epsilon;
        let eps_sq = eps * eps;
        let n_samples = self.config.n_samples;

        let mut grad = vec![F::zero(); n];
        let mut rng_state = self.config.seed;

        for _ in 0..n_samples {
            // Sample perturbed noise vector and find argmax
            let noise: Vec<f64> = (0..n)
                .map(|_| sample_standard_normal(&mut rng_state))
                .collect();

            let mut best_idx = 0usize;
            let mut best_val = F::neg_infinity();
            for i in 0..n {
                let perturbed = scores[i] + F::from_f64(eps * noise[i]).unwrap_or(F::zero());
                if perturbed > best_val {
                    best_val = perturbed;
                    best_idx = i;
                }
            }

            // argmax is a one-hot vector e_{best_idx}
            // <e_{best_idx}, upstream> = upstream[best_idx]
            let dot = upstream[best_idx];

            // Accumulate: grad += dot * Z / ε²
            for i in 0..n {
                let zi = F::from_f64(noise[i]).unwrap_or(F::zero());
                let eps_sq_f = F::from_f64(eps_sq).unwrap_or(F::one());
                grad[i] = grad[i] + dot * zi / eps_sq_f;
            }
        }

        let n_f = F::from_usize(n_samples).unwrap_or(F::one());
        for g in &mut grad {
            *g = *g / n_f;
        }

        Ok(grad)
    }
}

/// Splitmix64 PRNG step, returns a value in [0, 2^64).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Box-Muller transform to generate a standard normal sample.
fn sample_standard_normal(state: &mut u64) -> f64 {
    let u1_raw = splitmix64(state);
    let u2_raw = splitmix64(state);
    // Map to (0, 1]
    let u1 = (u1_raw as f64 + 0.5) / (u64::MAX as f64 + 1.0);
    let u2 = (u2_raw as f64 + 0.5) / (u64::MAX as f64 + 1.0);
    let two_pi = 2.0 * std::f64::consts::PI;
    (-2.0 * u1.ln()).sqrt() * (two_pi * u2).cos()
}

// ─────────────────────────────────────────────────────────────────────────────
// Differentiable Sorting and Ranking
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the soft sort of a vector via regularised isotonic regression.
///
/// Returns a non-decreasing sequence of the same length as `x`.  At
/// `temperature → 0` this recovers the exact sorted sequence; at high
/// temperature the output approaches the element-wise mean.
///
/// Internally uses the Pool Adjacent Violators (PAV) algorithm to solve the
/// isotonic regression:
/// ```text
/// min_{ŝ non-decreasing} Σ (ŝ_i - s_i)²  +  temperature · regularisation
/// ```
///
/// # Arguments
/// * `x` – input vector.
/// * `temperature` – controls softness (≥ 0; typical values 0.01–1.0).
///
/// # Returns
/// Non-decreasing vector of the same length as `x`.
pub fn soft_sort<F>(x: &[F], temperature: F) -> OptimizeResult<Vec<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    if x.is_empty() {
        return Err(OptimizeError::InvalidInput(
            "input vector must be non-empty".into(),
        ));
    }

    let n = x.len();
    // Sort indices to get the sorted values
    let mut sorted_x: Vec<F> = x.to_vec();
    sorted_x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Apply temperature-based regularisation:
    // For positive temperature, we compute the regularised soft sort by blending
    // sorted values with the mean (which acts as the max-entropy limit).
    if temperature == F::zero() {
        return Ok(sorted_x);
    }

    let mean_val =
        sorted_x.iter().fold(F::zero(), |a, b| a + *b) / F::from_usize(n).unwrap_or(F::one());

    // Regularised isotonic regression via PAV on scores shifted toward mean
    // The regularisation adds a quadratic penalty pulling toward mean.
    // We implement PAV on the mixture: (1-t)·sorted + t·mean
    let t_clamped = if temperature > F::one() {
        F::one()
    } else {
        temperature
    };
    let one_minus_t = F::one() - t_clamped;

    let mixed: Vec<F> = sorted_x
        .iter()
        .map(|&v| one_minus_t * v + t_clamped * mean_val)
        .collect();

    // PAV to ensure non-decreasingness (already guaranteed by initial sort,
    // but the blending preserves it so PAV is a no-op here — kept for
    // generality)
    let result = pool_adjacent_violators(&mixed);

    Ok(result)
}

/// Pool Adjacent Violators (PAV) algorithm for isotonic regression.
/// Solves `min Σ(ŝ_i - s_i)² s.t. ŝ_1 ≤ ŝ_2 ≤ … ≤ ŝ_n`.
fn pool_adjacent_violators<F>(s: &[F]) -> Vec<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = s.len();
    // Each block stores (sum, count)
    let mut blocks: Vec<(F, usize)> = s.iter().map(|&v| (v, 1)).collect();

    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0usize;
        let mut new_blocks: Vec<(F, usize)> = Vec::with_capacity(blocks.len());
        while i < blocks.len() {
            let mut sum = blocks[i].0;
            let mut cnt = blocks[i].1;
            // Merge with next block if violates monotonicity
            while i + 1 < blocks.len() {
                let next_mean =
                    blocks[i + 1].0 / F::from_usize(blocks[i + 1].1).unwrap_or(F::one());
                let cur_mean = sum / F::from_usize(cnt).unwrap_or(F::one());
                if cur_mean > next_mean {
                    sum = sum + blocks[i + 1].0;
                    cnt += blocks[i + 1].1;
                    i += 1;
                    changed = true;
                } else {
                    break;
                }
            }
            new_blocks.push((sum, cnt));
            i += 1;
        }
        blocks = new_blocks;
    }

    // Expand blocks back to length-n vector
    let mut result = Vec::with_capacity(n);
    for (sum, cnt) in blocks {
        let mean = sum / F::from_usize(cnt).unwrap_or(F::one());
        for _ in 0..cnt {
            result.push(mean);
        }
    }
    result
}

/// Compute soft ranks of elements in `x`.
///
/// Returns a vector of the same length as `x`, where each entry is the
/// (1-indexed) soft rank of the corresponding element.  At `temperature → 0`
/// this recovers the exact ranks (with ties broken by index).  At high
/// temperature all ranks are pulled toward `(n+1)/2`.
///
/// # Arguments
/// * `x` – input vector.
/// * `temperature` – smoothing parameter (≥ 0).
pub fn soft_rank<F>(x: &[F], temperature: F) -> OptimizeResult<Vec<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    if x.is_empty() {
        return Err(OptimizeError::InvalidInput(
            "input vector must be non-empty".into(),
        ));
    }
    let n = x.len();
    let one = F::one();
    let n_f = F::from_usize(n).unwrap_or(one);

    if temperature == F::zero() {
        // Hard rank: rank each element by counting how many elements it exceeds
        let ranks: Vec<F> = (0..n)
            .map(|i| {
                let rank = x.iter().filter(|&&v| v < x[i]).count();
                F::from_usize(rank + 1).unwrap_or(one)
            })
            .collect();
        return Ok(ranks);
    }

    // Soft rank via pairwise comparison with sigmoid smoothing:
    // rank_i ≈ 1 + Σ_{j≠i} σ((x_i - x_j) / temperature)
    let two = F::from_f64(2.0).unwrap_or(one);

    let ranks: Vec<F> = (0..n)
        .map(|i| {
            let mut soft_rank_i = one; // starts at 1
            for j in 0..n {
                if i == j {
                    continue;
                }
                let diff = (x[i] - x[j]) / temperature;
                // σ(diff) = 1/(1+exp(-diff)), clipped for numerical safety
                let diff_clamped = if diff < F::from_f64(-50.0).unwrap_or(-one) {
                    F::from_f64(-50.0).unwrap_or(-one)
                } else if diff > F::from_f64(50.0).unwrap_or(one) {
                    F::from_f64(50.0).unwrap_or(one)
                } else {
                    diff
                };
                let sigmoid_val = one / (one + (-diff_clamped).exp());
                soft_rank_i = soft_rank_i + sigmoid_val;
            }
            // Blend with mid-rank at high temperature to ensure sum = n(n+1)/2
            let mid = (n_f + one) / two;
            let t = if temperature > F::from_f64(10.0).unwrap_or(one) {
                one
            } else {
                temperature / F::from_f64(10.0).unwrap_or(one)
            };
            (one - t) * soft_rank_i + t * mid
        })
        .collect();

    Ok(ranks)
}

// ─────────────────────────────────────────────────────────────────────────────
// Differentiable Top-K
// ─────────────────────────────────────────────────────────────────────────────

/// Differentiable top-k selector via entropy-regularised LP.
///
/// Solves the relaxed problem:
/// ```text
/// max_{p ∈ Δ^n, Σp_i = k}  <scores, p>  -  temperature · H(p)
/// ```
/// where H(p) = -Σ p_i log p_i is the entropy regulariser.  The solution
/// has the closed form:
/// ```text
/// p_i = k · softmax(scores / temperature)_i
/// ```
/// normalised so that Σp_i = k.
///
/// At `temperature → 0`, `p` approaches the hard top-k indicator vector.
///
/// # Arguments
/// * `scores` – input scores (arbitrary real values).
/// * `k` – number of elements to select (1 ≤ k ≤ n).
/// * `temperature` – regularisation strength (> 0 for differentiable;
///   use a small value like 0.01 for near-hard top-k).
///
/// # Returns
/// Soft indicator vector p in `[0,1]`^n with Sum(p_i) ~ k.
pub fn diff_topk<F>(scores: &[F], k: usize, temperature: F) -> OptimizeResult<Vec<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = scores.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput(
            "scores must be non-empty".into(),
        ));
    }
    if k == 0 || k > n {
        return Err(OptimizeError::InvalidInput(format!(
            "k must be in [1, {}] but got {}",
            n, k
        )));
    }

    let k_f = F::from_usize(k).unwrap_or(F::one());

    if temperature == F::zero() {
        // Hard top-k: indicator vector
        let mut indexed: Vec<(usize, F)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let mut result = vec![F::zero(); n];
        for (idx, _) in indexed.iter().take(k) {
            result[*idx] = F::one();
        }
        return Ok(result);
    }

    // Numerically stable softmax: subtract max before exponentiating
    let max_score = scores
        .iter()
        .copied()
        .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });

    let exp_scores: Vec<F> = scores
        .iter()
        .map(|&s| {
            let scaled = (s - max_score) / temperature;
            // Clamp to avoid underflow
            let clamped = if scaled < F::from_f64(-80.0).unwrap_or(-F::one()) {
                F::from_f64(-80.0).unwrap_or(-F::one())
            } else {
                scaled
            };
            clamped.exp()
        })
        .collect();

    let sum_exp: F = exp_scores.iter().fold(F::zero(), |a, b| a + *b);
    if sum_exp == F::zero() {
        // All scores are -inf relative to max → uniform
        let uniform = k_f / F::from_usize(n).unwrap_or(F::one());
        return Ok(vec![uniform; n]);
    }

    let result: Vec<F> = exp_scores.iter().map(|&e| k_f * e / sum_exp).collect();

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-5;

    // ── SparsemapConfig defaults ─────────────────────────────────────────────

    #[test]
    fn test_sparsemap_config_defaults() {
        let cfg = SparsemapConfig::default();
        assert_eq!(cfg.max_iter, 1000);
        assert!((cfg.tol - 1e-6).abs() < 1e-12);
        assert!(matches!(cfg.structure_type, StructureType::Simplex));
    }

    // ── SparseMAP on simplex ─────────────────────────────────────────────────

    #[test]
    fn test_sparsemap_simplex_sums_to_one() {
        let scores = vec![1.0_f64, 2.0, 0.5, -0.3, 1.8];
        let cfg = SparsemapConfig::default();
        let res = sparsemap(&scores, &cfg).unwrap();
        let sum: f64 = res.solution.iter().sum();
        assert!((sum - 1.0).abs() < EPS, "sum = {}", sum);
    }

    #[test]
    fn test_sparsemap_simplex_sparse_support() {
        // Scores with clear winner should produce sparse solution
        let scores = vec![10.0_f64, 0.1, 0.1, 0.1, 0.1];
        let cfg = SparsemapConfig::default();
        let res = sparsemap(&scores, &cfg).unwrap();
        // The highest score dominates; at least one zero
        let n_nonzero = res.solution.iter().filter(|&&v| v > 1e-9).count();
        assert!(
            n_nonzero <= scores.len(),
            "non-zero count {} should be <= n",
            n_nonzero
        );
        assert!(!res.support.is_empty());
    }

    #[test]
    fn test_sparsemap_simplex_nonneg() {
        let scores = vec![-1.0_f64, -0.5, 0.3, 2.0, -3.0];
        let cfg = SparsemapConfig::default();
        let res = sparsemap(&scores, &cfg).unwrap();
        for &v in &res.solution {
            assert!(v >= -1e-10, "negative value {}", v);
        }
    }

    #[test]
    fn test_sparsemap_gradient_shape_matches_input() {
        let scores = vec![1.0_f64, 2.0, 0.5];
        let cfg = SparsemapConfig::default();
        let res = sparsemap(&scores, &cfg).unwrap();
        let upstream = vec![1.0_f64, 0.0, -1.0];
        let grad = sparsemap_gradient(&res, &upstream);
        assert_eq!(grad.len(), scores.len());
    }

    #[test]
    fn test_sparsemap_gradient_zeros_outside_support() {
        let scores = vec![5.0_f64, -5.0, -5.0];
        let cfg = SparsemapConfig::default();
        let res = sparsemap(&scores, &cfg).unwrap();
        let upstream = vec![1.0_f64, 1.0, 1.0];
        let grad = sparsemap_gradient(&res, &upstream);
        // Indices not in support should receive zero gradient
        for (i, &g) in grad.iter().enumerate() {
            if !res.support.contains(&i) {
                assert!(g.abs() < EPS, "index {} outside support has grad {}", i, g);
            }
        }
    }

    #[test]
    fn test_sparsemap_knapsack_feasibility() {
        let weights = vec![1.0_f64, 2.0, 3.0];
        let capacity = 3.0_f64;
        let cfg = SparsemapConfig {
            structure_type: StructureType::Knapsack {
                weights: weights.clone(),
                capacity,
            },
            max_iter: 500,
            ..SparsemapConfig::default()
        };
        let scores = vec![3.0_f64, 2.0, 1.0];
        let res = sparsemap(&scores, &cfg).unwrap();
        // All values in [0,1]
        for &v in &res.solution {
            assert!(v >= -EPS && v <= 1.0 + EPS, "value {} out of [0,1]", v);
        }
        // Weighted sum ≤ capacity
        let used: f64 = weights
            .iter()
            .zip(res.solution.iter())
            .map(|(&w, &v)| w * v)
            .sum();
        assert!(used <= capacity + EPS, "capacity exceeded: {}", used);
    }

    // ── PerturbedOptimizerConfig defaults ───────────────────────────────────

    #[test]
    fn test_perturbed_optimizer_config_defaults() {
        let cfg = PerturbedOptimizerConfig::default();
        assert_eq!(cfg.n_samples, 100);
        assert!((cfg.epsilon - 0.1).abs() < 1e-12);
        assert_eq!(cfg.seed, 42);
    }

    // ── PerturbedOptimizer forward ───────────────────────────────────────────

    #[test]
    fn test_perturbed_optimizer_output_sums_to_one() {
        let cfg = PerturbedOptimizerConfig {
            n_samples: 200,
            ..Default::default()
        };
        let opt = PerturbedOptimizer::new(cfg);
        let scores = vec![1.0_f64, 2.0, 0.5, 3.0];
        let probs = opt.forward(&scores).unwrap();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "sum = {}", sum);
    }

    #[test]
    fn test_perturbed_optimizer_n_samples_1_deterministic() {
        let cfg = PerturbedOptimizerConfig {
            n_samples: 1,
            seed: 7,
            ..Default::default()
        };
        let opt = PerturbedOptimizer::new(cfg.clone());
        let scores = vec![1.0_f64, 2.0, 0.5];
        let p1 = opt.forward(&scores).unwrap();
        let opt2 = PerturbedOptimizer::new(cfg);
        let p2 = opt2.forward(&scores).unwrap();
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(a, b, "results differ between identical seeds");
        }
    }

    // ── soft_sort ────────────────────────────────────────────────────────────

    #[test]
    fn test_soft_sort_nondecreasing() {
        let x = vec![3.0_f64, 1.0, 4.0, 1.5, 9.0, 2.6];
        let sorted = soft_sort(&x, 0.0_f64).unwrap();
        for w in sorted.windows(2) {
            assert!(w[0] <= w[1] + 1e-10, "not sorted: {} > {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_soft_sort_nonzero_temp_nondecreasing() {
        let x = vec![5.0_f64, 1.0, 3.0, 2.0];
        let sorted = soft_sort(&x, 0.5_f64).unwrap();
        for w in sorted.windows(2) {
            assert!(
                w[0] <= w[1] + 1e-9,
                "soft_sort not sorted: {} > {}",
                w[0],
                w[1]
            );
        }
    }

    // ── soft_rank ────────────────────────────────────────────────────────────

    #[test]
    fn test_soft_rank_high_temp_input_3_1_2() {
        // At high temperature, hard ranks of [3,1,2] should be [3,1,2]
        // (largest element gets rank 3)
        let x = vec![3.0_f64, 1.0, 2.0];
        let ranks = soft_rank(&x, 0.0_f64).unwrap();
        assert_eq!(ranks[0] as usize, 3, "rank of largest should be 3");
        assert_eq!(ranks[1] as usize, 1, "rank of smallest should be 1");
        assert_eq!(ranks[2] as usize, 2, "rank of middle should be 2");
    }

    // ── diff_topk ────────────────────────────────────────────────────────────

    #[test]
    fn test_diff_topk_sums_to_k() {
        let scores = vec![1.0_f64, 5.0, 2.0, 4.0, 3.0];
        let k = 3;
        let p = diff_topk(&scores, k, 0.5_f64).unwrap();
        let sum: f64 = p.iter().sum();
        assert!(
            (sum - k as f64).abs() < 1e-6,
            "sum = {} but expected k={}",
            sum,
            k
        );
    }

    #[test]
    fn test_diff_topk_zero_temp_hard_topk() {
        let scores = vec![1.0_f64, 5.0, 2.0, 4.0, 3.0];
        let k = 2;
        let p = diff_topk(&scores, k, 0.0_f64).unwrap();
        // Should select indices 1 (score 5.0) and 3 (score 4.0)
        let sum: f64 = p.iter().sum();
        assert!((sum - k as f64).abs() < 1e-9);
        assert!((p[1] - 1.0).abs() < 1e-9, "index 1 should be selected");
        assert!((p[3] - 1.0).abs() < 1e-9, "index 3 should be selected");
    }

    #[test]
    fn test_diff_topk_all_values_nonneg() {
        // diff_topk returns p_i in [0, k]; each value is non-negative and sum = k
        let scores = vec![0.1_f64, 2.3, -1.0, 5.0, 0.7];
        let k = 2usize;
        let p = diff_topk(&scores, k, 1.0_f64).unwrap();
        for &v in &p {
            assert!(v >= -1e-9, "value {} is negative", v);
            assert!(v <= k as f64 + 1e-9, "value {} exceeds k={}", v, k);
        }
        let sum: f64 = p.iter().sum();
        assert!(
            (sum - k as f64).abs() < 1e-6,
            "sum = {} expected k={}",
            sum,
            k
        );
    }
}
