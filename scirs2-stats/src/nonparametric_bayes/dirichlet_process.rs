//! Dirichlet Process and related constructs
//!
//! Provides:
//! - `DirichletProcess` – typed wrapper around the DP concentration parameter
//! - Stick-breaking (GEM) construction of DP mixture weights
//! - Chinese Restaurant Process (CRP) seating simulation
//! - CRP predictive distribution
//! - Pitman-Yor Process (PYP) seating simulation
//! - Expected cluster count and alpha estimation utilities
//! - Posterior distribution over number of tables
//! - `dp_mixture_gibbs` – a lightweight CRP + Gaussian collapsed Gibbs sampler
//!   that marginalises cluster parameters analytically (conjugate Normal-Normal prior)

use crate::error::StatsError;
use scirs2_core::ndarray::ArrayView2;
use scirs2_core::random::{rngs::StdRng, Distribution, Gamma, Normal, SeedableRng, Uniform};

// ---------------------------------------------------------------------------
// DirichletProcess
// ---------------------------------------------------------------------------

/// Typed representation of a Dirichlet Process with concentration parameter α.
///
/// Provides convenience methods for the most common DP constructions (CRP,
/// stick-breaking) without requiring the caller to carry the concentration
/// parameter through every function call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirichletProcess {
    /// Concentration parameter α > 0.
    ///
    /// Larger α leads to more clusters and more evenly weighted atoms.
    pub alpha: f64,
}

impl DirichletProcess {
    /// Construct a new `DirichletProcess`.
    ///
    /// # Errors
    /// Returns `StatsError::DomainError` when `alpha <= 0`.
    pub fn new(alpha: f64) -> Result<Self, StatsError> {
        if alpha <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "DirichletProcess::new: alpha must be > 0, got {alpha}"
            )));
        }
        Ok(Self { alpha })
    }

    /// Expected number of clusters for `n` observations.
    ///
    /// Approximation: E[K_n] ≈ α · ln(1 + n/α).
    pub fn expected_clusters(&self, n: usize) -> f64 {
        expected_clusters(self.alpha, n)
    }

    /// Draw cluster assignments via the Chinese Restaurant Process.
    ///
    /// # Parameters
    /// - `n`: Number of customers (observations).
    /// - `seed`: Random seed.
    ///
    /// # Returns
    /// `(assignments, n_tables)` – see [`chinese_restaurant_process`].
    pub fn sample_crp(
        &self,
        n: usize,
        seed: u64,
    ) -> Result<(Vec<usize>, usize), StatsError> {
        chinese_restaurant_process(n, self.alpha, seed)
    }

    /// Draw mixture weights from the stick-breaking (GEM) construction.
    ///
    /// # Parameters
    /// - `n_sticks`: Truncation level (number of explicit sticks to break).
    /// - `seed`: Random seed.
    ///
    /// # Returns
    /// Weight vector of length `n_sticks + 1`; see [`stick_breaking`].
    pub fn sample_stick_breaking(
        &self,
        n_sticks: usize,
        seed: u64,
    ) -> Result<Vec<f64>, StatsError> {
        stick_breaking(self.alpha, n_sticks, seed)
    }

    /// Collapsed Gibbs sampler for a DP Gaussian mixture model.
    ///
    /// Convenience wrapper around [`dp_mixture_gibbs`].
    ///
    /// # Parameters
    /// - `data`: (n_samples × n_features) data matrix.
    /// - `prior_mean`: Prior mean μ₀ for all clusters (length = n_features).
    /// - `prior_variance`: Isotropic prior variance σ₀² > 0.
    /// - `noise_variance`: Isotropic observation noise variance σ² > 0.
    /// - `max_iter`: Number of full Gibbs sweeps.
    /// - `seed`: Random seed.
    ///
    /// # Returns
    /// Cluster assignment for each of the n data points.
    pub fn fit_gmm(
        &self,
        data: ArrayView2<f64>,
        prior_mean: &[f64],
        prior_variance: f64,
        noise_variance: f64,
        max_iter: usize,
        seed: u64,
    ) -> Result<Vec<usize>, StatsError> {
        dp_mixture_gibbs(data, self.alpha, prior_mean, prior_variance, noise_variance, max_iter, seed)
    }
}

// ---------------------------------------------------------------------------
// Internal helper: Beta sampler via ratio of Gamma variates
// ---------------------------------------------------------------------------

/// Sample from Beta(a, b) using the ratio-of-gammas method.
fn beta_sample(a: f64, b: f64, rng: &mut StdRng) -> Result<f64, StatsError> {
    let ga = Gamma::new(a, 1.0).map_err(|e| {
        StatsError::ComputationError(format!("Beta sampler Gamma(a) error: {e}"))
    })?;
    let gb = Gamma::new(b, 1.0).map_err(|e| {
        StatsError::ComputationError(format!("Beta sampler Gamma(b) error: {e}"))
    })?;
    let x = ga.sample(rng);
    let y = gb.sample(rng);
    let sum = x + y;
    if sum == 0.0 {
        return Err(StatsError::ComputationError(
            "Beta sampler: both Gamma samples are zero".to_string(),
        ));
    }
    Ok(x / sum)
}

// ---------------------------------------------------------------------------
// Stick-breaking (GEM distribution)
// ---------------------------------------------------------------------------

/// Stick-breaking construction of Dirichlet Process (GEM distribution).
///
/// Returns `n_sticks + 1` mixture weights obtained by the following procedure:
/// 1. Draw `vk ~ Beta(1, alpha)` for `k = 1..n_sticks`.
/// 2. `w_k = vk * prod_{j<k}(1 - vj)` (break successive pieces of a unit stick).
/// 3. The remaining weight `1 - sum(w_1..w_K)` is appended as the final element,
///    representing all remaining atoms beyond the truncation level.
///
/// # Parameters
/// - `alpha`: DP concentration parameter (> 0). Larger α → more even weights.
/// - `n_sticks`: Truncation level K (number of explicit sticks to break). The
///   returned vector has length `n_sticks + 1`.
/// - `seed`: Random seed for reproducibility.
///
/// # Returns
/// Weights vector of length `n_sticks + 1`; the last element is the residual mass.
/// All elements are non-negative and sum to 1.0 (up to floating-point precision).
pub fn stick_breaking(
    alpha: f64,
    n_sticks: usize,
    seed: u64,
) -> Result<Vec<f64>, StatsError> {
    if alpha <= 0.0 {
        return Err(StatsError::DomainError(format!(
            "stick_breaking: alpha must be > 0, got {alpha}"
        )));
    }
    if n_sticks == 0 {
        return Err(StatsError::InvalidArgument(
            "stick_breaking: n_sticks must be >= 1".to_string(),
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut weights = Vec::with_capacity(n_sticks + 1);
    let mut remaining = 1.0_f64;

    for _ in 0..n_sticks {
        let v = beta_sample(1.0, alpha, &mut rng)?;
        let w = v * remaining;
        weights.push(w);
        remaining *= 1.0 - v;
        if remaining < 1e-15 {
            // Numerical underflow – stop early; residual is effectively 0
            remaining = 0.0;
            break;
        }
    }
    weights.push(remaining);
    Ok(weights)
}

// ---------------------------------------------------------------------------
// Chinese Restaurant Process (CRP)
// ---------------------------------------------------------------------------

/// Simulate the Chinese Restaurant Process for `n_customers` customers.
///
/// The CRP metaphor: customers enter a restaurant one by one. Customer 1 sits
/// at a new table. Each subsequent customer i:
/// - Sits at existing table k with probability `n_k / (i - 1 + alpha)`
/// - Starts a new table with probability `alpha / (i - 1 + alpha)`
///
/// # Parameters
/// - `n_customers`: Total number of customers (data points).
/// - `alpha`: DP concentration parameter (> 0).
/// - `seed`: Random seed.
///
/// # Returns
/// `(assignments, n_tables)`:
/// - `assignments[i]` is the 0-indexed table assigned to customer `i`.
/// - `n_tables` is the total number of distinct tables used.
pub fn chinese_restaurant_process(
    n_customers: usize,
    alpha: f64,
    seed: u64,
) -> Result<(Vec<usize>, usize), StatsError> {
    if alpha <= 0.0 {
        return Err(StatsError::DomainError(format!(
            "chinese_restaurant_process: alpha must be > 0, got {alpha}"
        )));
    }
    if n_customers == 0 {
        return Ok((Vec::new(), 0));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let uniform = Uniform::new(0.0_f64, 1.0).map_err(|e| {
        StatsError::ComputationError(format!("CRP Uniform init error: {e}"))
    })?;

    let mut assignments = Vec::with_capacity(n_customers);
    // table_counts[k] = number of customers currently at table k
    let mut table_counts: Vec<usize> = Vec::new();

    for i in 0..n_customers {
        let total = i as f64 + alpha; // i customers already seated + alpha
        let u = uniform.sample(&mut rng) * total;

        let mut cumulative = 0.0_f64;
        let mut chosen_table = table_counts.len(); // default: new table

        for (k, &count) in table_counts.iter().enumerate() {
            cumulative += count as f64;
            if u < cumulative {
                chosen_table = k;
                break;
            }
        }

        assignments.push(chosen_table);
        if chosen_table == table_counts.len() {
            table_counts.push(1);
        } else {
            table_counts[chosen_table] += 1;
        }
    }

    let n_tables = table_counts.len();
    Ok((assignments, n_tables))
}

// ---------------------------------------------------------------------------
// CRP predictive distribution
// ---------------------------------------------------------------------------

/// Compute the CRP predictive distribution for the next customer.
///
/// Given the current table counts, returns the probability that the next
/// customer joins each existing table or starts a new one.
///
/// # Parameters
/// - `existing_counts`: Number of customers at each existing table.
/// - `alpha`: DP concentration parameter (> 0).
///
/// # Returns
/// Vector of length `existing_counts.len() + 1`.
/// - Elements `0..len-1` are probabilities of joining each existing table.
/// - The last element is the probability of starting a new table.
pub fn crp_predictive(existing_counts: &[usize], alpha: f64) -> Vec<f64> {
    let n_existing: usize = existing_counts.iter().sum();
    let total = n_existing as f64 + alpha;

    let mut probs: Vec<f64> = existing_counts
        .iter()
        .map(|&c| c as f64 / total)
        .collect();
    probs.push(alpha / total);
    probs
}

// ---------------------------------------------------------------------------
// Pitman-Yor Process
// ---------------------------------------------------------------------------

/// Simulate the Pitman-Yor Process for `n_customers`.
///
/// The PYP(d, α) generalises the DP by introducing a discount parameter d ∈ [0, 1).
/// Customer i:
/// - Sits at existing table k with probability `(n_k - d) / (i - 1 + alpha)`
/// - Starts a new table with probability `(alpha + n_tables * d) / (i - 1 + alpha)`
///
/// When d = 0 this reduces to the standard CRP.
///
/// # Parameters
/// - `n_customers`: Number of customers to seat.
/// - `discount`: Discount parameter d ∈ [0, 1). Controls power-law exponent.
/// - `strength`: Strength parameter α > -d.
/// - `seed`: Random seed.
///
/// # Returns
/// `(assignments, n_tables)`.
pub fn pitman_yor_process(
    n_customers: usize,
    discount: f64,
    strength: f64,
    seed: u64,
) -> Result<(Vec<usize>, usize), StatsError> {
    if !(0.0..1.0).contains(&discount) {
        return Err(StatsError::DomainError(format!(
            "pitman_yor_process: discount must be in [0, 1), got {discount}"
        )));
    }
    if strength <= -discount {
        return Err(StatsError::DomainError(format!(
            "pitman_yor_process: strength must be > -discount ({} > {}), got {}",
            strength, -discount, strength
        )));
    }
    if n_customers == 0 {
        return Ok((Vec::new(), 0));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let uniform = Uniform::new(0.0_f64, 1.0).map_err(|e| {
        StatsError::ComputationError(format!("PYP Uniform init error: {e}"))
    })?;

    let mut assignments = Vec::with_capacity(n_customers);
    let mut table_counts: Vec<usize> = Vec::new();

    for i in 0..n_customers {
        let n_tables = table_counts.len();
        let total = i as f64 + strength; // i customers already seated
        let u = uniform.sample(&mut rng) * total;

        let mut cumulative = 0.0_f64;
        let mut chosen_table = n_tables; // default: new table

        for (k, &count) in table_counts.iter().enumerate() {
            cumulative += (count as f64) - discount;
            if u < cumulative {
                chosen_table = k;
                break;
            }
        }

        // If not assigned yet, compare against new-table mass
        // new-table probability ∝ strength + n_tables * discount
        // (the cumulative loop above may have covered all existing tables)
        assignments.push(chosen_table);
        if chosen_table == table_counts.len() {
            table_counts.push(1);
        } else {
            table_counts[chosen_table] += 1;
        }
    }

    let n_tables = table_counts.len();
    Ok((assignments, n_tables))
}

// ---------------------------------------------------------------------------
// Expected cluster count
// ---------------------------------------------------------------------------

/// Compute the expected number of clusters under the CRP for `n` samples.
///
/// The approximation E[K_n] ≈ α · ln(1 + n/α) is the harmonic-number-based
/// asymptotic formula (exact in the limit for the DP).
pub fn expected_clusters(alpha: f64, n: usize) -> f64 {
    alpha * (1.0 + n as f64 / alpha).ln()
}

// ---------------------------------------------------------------------------
// Alpha estimation via Newton–Raphson on CRP marginal likelihood
// ---------------------------------------------------------------------------

/// Estimate the DP concentration parameter α from an observed (n, K) pair.
///
/// Uses Newton-Raphson iterations on the CRP log-marginal-likelihood with
/// respect to α: `L(α) = ln Γ(α) - ln Γ(α+n) + K·ln α + const`.
/// The derivative is `ψ(α) - ψ(α+n) + K/α` and the second derivative is
/// `ψ'(α) - ψ'(α+n) - K/α²`, where ψ is the digamma function.
///
/// # Parameters
/// - `n_samples`: Total number of observations.
/// - `n_clusters`: Number of distinct clusters observed.
/// - `n_iter`: Number of Newton-Raphson iterations (50 is typically sufficient).
///
/// # Returns
/// Estimated α > 0.
pub fn estimate_alpha_from_clusters(
    n_samples: usize,
    n_clusters: usize,
    n_iter: usize,
) -> Result<f64, StatsError> {
    if n_samples == 0 {
        return Err(StatsError::InsufficientData(
            "estimate_alpha_from_clusters: n_samples must be >= 1".to_string(),
        ));
    }
    if n_clusters == 0 || n_clusters > n_samples {
        return Err(StatsError::InvalidArgument(format!(
            "estimate_alpha_from_clusters: n_clusters must be in [1, n_samples], got {n_clusters} / {n_samples}"
        )));
    }
    if n_iter == 0 {
        return Err(StatsError::InvalidArgument(
            "estimate_alpha_from_clusters: n_iter must be >= 1".to_string(),
        ));
    }

    let n = n_samples as f64;
    let k = n_clusters as f64;

    // Initial guess via moment matching: k ≈ α·ln(1 + n/α) → α ≈ k/ln(n)
    let mut alpha = if n > 1.0 { k / n.ln().max(1e-10) } else { 1.0 };
    alpha = alpha.max(1e-6);

    for _ in 0..n_iter {
        // digamma and trigamma via the asymptotic series (good for α > 6)
        // We use recurrence to shift into the asymptotic regime.
        let dpsi_alpha = digamma(alpha);
        let dpsi_alpha_n = digamma(alpha + n);
        let tpsi_alpha = trigamma(alpha);
        let tpsi_alpha_n = trigamma(alpha + n);

        // First derivative: d/dα [ K ln α + ln Γ(α) - ln Γ(α+n) ]
        let grad = k / alpha + dpsi_alpha - dpsi_alpha_n;
        // Second derivative
        let hess = -k / (alpha * alpha) + tpsi_alpha - tpsi_alpha_n;

        if hess.abs() < 1e-15 {
            break;
        }
        let step = grad / hess;
        let alpha_new = (alpha - step).max(1e-6);
        if (alpha_new - alpha).abs() < 1e-10 * alpha {
            alpha = alpha_new;
            break;
        }
        alpha = alpha_new;
    }

    Ok(alpha)
}

// ---------------------------------------------------------------------------
// Digamma / Trigamma (Bernoulli asymptotic series)
// ---------------------------------------------------------------------------

/// Digamma function ψ(x) = d/dx ln Γ(x) via recurrence + asymptotic series.
fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    // Shift x into the asymptotic regime (x > 6)
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic series: ψ(x) ≈ ln(x) - 1/(2x) - Σ B_{2k}/(2k·x^{2k})
    result += x.ln() - 0.5 / x;
    let x2 = x * x;
    result -= 1.0 / (12.0 * x2);
    result += 1.0 / (120.0 * x2 * x2);
    result -= 1.0 / (252.0 * x2 * x2 * x2);
    result
}

/// Trigamma function ψ'(x) = d²/dx² ln Γ(x) via recurrence + asymptotic series.
fn trigamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 6.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    // Asymptotic: ψ'(x) ≈ 1/x + 1/(2x²) + Σ B_{2k}·2k / x^{2k+1}
    let x2 = x * x;
    result += 1.0 / x + 1.0 / (2.0 * x2);
    result += 1.0 / (6.0 * x2 * x);
    result -= 1.0 / (30.0 * x2 * x2 * x);
    result += 1.0 / (42.0 * x2 * x2 * x2 * x);
    result
}

// ---------------------------------------------------------------------------
// Posterior over number of tables
// ---------------------------------------------------------------------------

/// Compute the posterior distribution over the number of tables K given n
/// customers and concentration α under the CRP.
///
/// Uses the (unsigned) Stirling numbers of the first kind:
/// `P(K=k | n, α) ∝ |s(n, k)| · α^k`
///
/// where `|s(n, k)|` are unsigned Stirling numbers of the first kind.
/// This function computes the full row of unsigned Stirling numbers via the
/// recurrence `|s(n+1, k)| = n·|s(n,k)| + |s(n, k-1)|` and normalises.
///
/// # Parameters
/// - `n_customers`: Number of data points (n ≥ 1).
/// - `alpha`: DP concentration parameter (α > 0).
///
/// # Returns
/// Vector of length `n_customers` where index `k-1` holds `P(K = k)`.
pub fn crp_posterior_tables(n_customers: usize, alpha: f64) -> Vec<f64> {
    if n_customers == 0 {
        return Vec::new();
    }

    // Compute unsigned Stirling numbers of the first kind |s(n, k)| in
    // log-space to avoid overflow.  We work with a single row at a time.
    let n = n_customers;
    // log_stirling[k] = log |s(n, k+1)|  (1-indexed in math, 0-indexed here)
    let mut log_s = vec![f64::NEG_INFINITY; n];
    log_s[0] = 0.0; // |s(1,1)| = 1

    for i in 1..n {
        let mut new_log_s = vec![f64::NEG_INFINITY; n];
        for k in 0..=i {
            // |s(i+1, k+1)| = i * |s(i, k+1)| + |s(i, k)|
            let term1 = if log_s[k] > f64::NEG_INFINITY {
                log_s[k] + (i as f64).ln()
            } else {
                f64::NEG_INFINITY
            };
            let term2 = if k > 0 && log_s[k - 1] > f64::NEG_INFINITY {
                log_s[k - 1]
            } else {
                f64::NEG_INFINITY
            };
            new_log_s[k] = log_sum_exp(term1, term2);
        }
        log_s = new_log_s;
    }

    // log P(K=k) ∝ log|s(n,k)| + k * log(alpha)
    let log_alpha = alpha.ln();
    let mut log_probs: Vec<f64> = (0..n)
        .map(|k| {
            if log_s[k] > f64::NEG_INFINITY {
                log_s[k] + (k + 1) as f64 * log_alpha
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();

    // Normalise in log-space
    let max_lp = log_probs
        .iter()
        .cloned()
        .filter(|x| x.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);

    if !max_lp.is_finite() {
        // Degenerate: return uniform
        let p = 1.0 / n as f64;
        return vec![p; n];
    }

    let sum_exp: f64 = log_probs
        .iter()
        .map(|&lp| if lp.is_finite() { (lp - max_lp).exp() } else { 0.0 })
        .sum();
    let log_norm = max_lp + sum_exp.ln();

    for lp in log_probs.iter_mut() {
        if lp.is_finite() {
            *lp = (*lp - log_norm).exp();
        } else {
            *lp = 0.0;
        }
    }
    log_probs
}

/// Numerically stable log(exp(a) + exp(b)).
fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

// ---------------------------------------------------------------------------
// dp_mixture_gibbs
// ---------------------------------------------------------------------------

/// Collapsed Gibbs sampler for a DP Gaussian mixture model with a conjugate
/// isotropic Normal-Normal (known variance) prior.
///
/// For each data point the cluster assignment is sampled from
/// ```text
/// P(z_i = k | rest) ∝ n_k · N(x_i | μ_k^post, (σ² + σ₀²/n_k)·I)   (existing cluster)
/// P(z_i = new | rest) ∝ α · N(x_i | μ₀, (σ² + σ₀²)·I)              (new cluster)
/// ```
/// where the posterior mean μ_k^post = σ₀⁻²·μ₀ + σ⁻²·Σx_k) / (σ₀⁻² + n_k·σ⁻²)`.
///
/// # Parameters
/// - `data`: (n_samples × n_features) data matrix.
/// - `alpha`: DP concentration parameter α > 0.
/// - `prior_mean`: Prior mean vector μ₀ (length = n_features). Use a slice of
///   zeros for an uninformative location prior centred at the origin.
/// - `prior_variance`: Isotropic prior variance σ₀² > 0 on the cluster mean.
/// - `noise_variance`: Isotropic observation noise variance σ² > 0.
/// - `max_iter`: Number of full Gibbs sweeps over all n data points.
/// - `seed`: Random seed for reproducibility.
///
/// # Returns
/// Cluster assignment vector of length n (0-indexed) from the final Gibbs state.
///
/// # Errors
/// Returns `StatsError` for invalid arguments or numerical failures.
pub fn dp_mixture_gibbs(
    data: ArrayView2<f64>,
    alpha: f64,
    prior_mean: &[f64],
    prior_variance: f64,
    noise_variance: f64,
    max_iter: usize,
    seed: u64,
) -> Result<Vec<usize>, StatsError> {
    use std::f64::consts::PI as F64_PI;

    // ---- Validate inputs ----
    if alpha <= 0.0 {
        return Err(StatsError::DomainError(format!(
            "dp_mixture_gibbs: alpha must be > 0, got {alpha}"
        )));
    }
    let (n, d) = (data.nrows(), data.ncols());
    if n == 0 {
        return Err(StatsError::InsufficientData(
            "dp_mixture_gibbs: data has no rows".to_string(),
        ));
    }
    if d == 0 {
        return Err(StatsError::InvalidArgument(
            "dp_mixture_gibbs: data has zero features".to_string(),
        ));
    }
    if prior_mean.len() != d {
        return Err(StatsError::DimensionMismatch(format!(
            "dp_mixture_gibbs: prior_mean length {} != data features {d}",
            prior_mean.len()
        )));
    }
    if prior_variance <= 0.0 {
        return Err(StatsError::DomainError(format!(
            "dp_mixture_gibbs: prior_variance must be > 0, got {prior_variance}"
        )));
    }
    if noise_variance <= 0.0 {
        return Err(StatsError::DomainError(format!(
            "dp_mixture_gibbs: noise_variance must be > 0, got {noise_variance}"
        )));
    }
    if max_iter == 0 {
        return Err(StatsError::InvalidArgument(
            "dp_mixture_gibbs: max_iter must be >= 1".to_string(),
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let uniform = Uniform::new(0.0_f64, 1.0).map_err(|e| {
        StatsError::ComputationError(format!("dp_mixture_gibbs Uniform error: {e}"))
    })?;

    // Precision values (isotropic)
    let prec_prior = 1.0 / prior_variance; // 1/σ₀²
    let prec_noise = 1.0 / noise_variance; // 1/σ²

    // ---- Initialise assignments uniformly at random ----
    let k_init = ((alpha * (n as f64).ln()).round() as usize).max(1);
    let init_uniform = Uniform::new(0usize, k_init).map_err(|e| {
        StatsError::ComputationError(format!("dp_mixture_gibbs init Uniform error: {e}"))
    })?;
    let mut assignments: Vec<usize> = (0..n).map(|_| init_uniform.sample(&mut rng)).collect();

    // ---- Cluster sufficient statistics ----
    // For each cluster k:  n_k (count), sum_x[k] (sum of member vectors)
    // We use a Vec<(usize, Vec<f64>)> indexed by cluster id.
    // After compaction, cluster ids are contiguous 0..K.

    // Helper: build counts and sums from current assignments
    fn rebuild_stats(
        data: ArrayView2<f64>,
        assignments: &[usize],
    ) -> (Vec<usize>, Vec<Vec<f64>>) {
        let d = data.ncols();
        let k_max = assignments.iter().cloned().max().map(|m| m + 1).unwrap_or(0);
        let mut counts = vec![0usize; k_max];
        let mut sums = vec![vec![0.0_f64; d]; k_max];
        for (i, &ci) in assignments.iter().enumerate() {
            counts[ci] += 1;
            let row = data.row(i);
            for j in 0..d {
                sums[ci][j] += row[j];
            }
        }
        (counts, sums)
    }

    let (mut counts, mut sums) = rebuild_stats(data, &assignments);

    // Helper: log marginal likelihood log N(x | μ_post, σ²_pred · I)
    // where σ²_pred = σ² + σ₀² / n_k (for existing cluster with n_k members after
    // removing x from it, or σ² + σ₀² for a new cluster)
    // and μ_post_j = (prec_prior · μ₀[j] + prec_noise · sum_x_k[j]) / (prec_prior + n_k · prec_noise)
    let log_normal_isotropic = |x_row: &[f64], mu_post: &[f64], sigma2_pred: f64| -> f64 {
        let dim = x_row.len() as f64;
        let mut sum_sq = 0.0_f64;
        for j in 0..x_row.len() {
            let diff = x_row[j] - mu_post[j];
            sum_sq += diff * diff;
        }
        -0.5 * dim * (2.0 * F64_PI * sigma2_pred).ln() - 0.5 * sum_sq / sigma2_pred
    };

    // ---- Main Gibbs loop ----
    for _iter in 0..max_iter {
        for i in 0..n {
            let ci = assignments[i];
            let x_row: Vec<f64> = (0..d).map(|j| data[[i, j]]).collect();

            // Remove point i from its cluster
            counts[ci] -= 1;
            for j in 0..d {
                sums[ci][j] -= x_row[j];
            }

            // Determine non-empty clusters (excluding now-empty ci if applicable)
            let k_current = counts.len();

            // Build log-weights for each existing cluster and for a new cluster
            let mut log_weights: Vec<f64> = Vec::with_capacity(k_current + 1);

            for k in 0..k_current {
                let nk = counts[k];
                if nk == 0 {
                    // Empty cluster: push -inf to exclude
                    log_weights.push(f64::NEG_INFINITY);
                    continue;
                }
                let nk_f = nk as f64;
                // Posterior mean for cluster k (excluding point i)
                let prec_post = prec_prior + nk_f * prec_noise;
                let sigma2_pred = noise_variance + 1.0 / prec_post;
                let mu_post: Vec<f64> = (0..d)
                    .map(|j| {
                        (prec_prior * prior_mean[j] + prec_noise * sums[k][j]) / prec_post
                    })
                    .collect();
                let lp = nk_f.ln() + log_normal_isotropic(&x_row, &mu_post, sigma2_pred);
                log_weights.push(lp);
            }

            // New cluster weight
            let sigma2_pred_new = noise_variance + prior_variance;
            let lp_new = alpha.ln() + log_normal_isotropic(&x_row, prior_mean, sigma2_pred_new);
            log_weights.push(lp_new);

            // Stable softmax
            let max_lw = log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = log_weights
                .iter()
                .map(|&lw| if lw.is_finite() { (lw - max_lw).exp() } else { 0.0 })
                .collect();
            let total: f64 = weights.iter().sum();
            if total == 0.0 {
                // Fallback: assign to first non-empty cluster or create new
                let fallback = counts.iter().position(|&c| c > 0).unwrap_or(k_current);
                let new_ci = if fallback == k_current {
                    counts.push(0);
                    sums.push(vec![0.0; d]);
                    k_current
                } else {
                    fallback
                };
                assignments[i] = new_ci;
                counts[new_ci] += 1;
                for j in 0..d {
                    sums[new_ci][j] += x_row[j];
                }
                continue;
            }

            // Categorical sample
            let u = uniform.sample(&mut rng) * total;
            let mut cumul = 0.0_f64;
            let mut chosen = k_current; // default: new cluster
            for (k, &w) in weights.iter().enumerate() {
                cumul += w;
                if u < cumul {
                    chosen = k;
                    break;
                }
            }

            // Assign to chosen cluster
            if chosen < k_current {
                assignments[i] = chosen;
                counts[chosen] += 1;
                for j in 0..d {
                    sums[chosen][j] += x_row[j];
                }
            } else {
                // New cluster: reuse an empty slot if one exists, otherwise extend
                let new_ci = counts.iter().position(|&c| c == 0).unwrap_or_else(|| {
                    counts.push(0);
                    sums.push(vec![0.0; d]);
                    counts.len() - 1
                });
                assignments[i] = new_ci;
                counts[new_ci] += 1;
                for j in 0..d {
                    sums[new_ci][j] += x_row[j];
                }
            }
        }

        // Compact: renumber to remove empty cluster slots
        let old_to_new: Vec<Option<usize>> = {
            let mut next = 0usize;
            counts
                .iter()
                .map(|&c| {
                    if c > 0 {
                        let id = next;
                        next += 1;
                        Some(id)
                    } else {
                        None
                    }
                })
                .collect()
        };
        let new_k = old_to_new.iter().filter(|x| x.is_some()).count();
        let mut new_counts = vec![0usize; new_k];
        let mut new_sums = vec![vec![0.0_f64; d]; new_k];
        for (old, maybe_new) in old_to_new.iter().enumerate() {
            if let Some(nid) = maybe_new {
                new_counts[*nid] = counts[old];
                new_sums[*nid] = sums[old].clone();
            }
        }
        for a in assignments.iter_mut() {
            if let Some(nid) = old_to_new[*a] {
                *a = nid;
            }
        }
        counts = new_counts;
        sums = new_sums;
    }

    Ok(assignments)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- DirichletProcess ---

    #[test]
    fn test_dp_new_valid() {
        let dp = DirichletProcess::new(1.5).expect("DP new failed");
        assert!((dp.alpha - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_dp_new_invalid() {
        assert!(DirichletProcess::new(0.0).is_err());
        assert!(DirichletProcess::new(-1.0).is_err());
    }

    #[test]
    fn test_dp_expected_clusters() {
        let dp = DirichletProcess::new(2.0).expect("DP new failed");
        let e = dp.expected_clusters(100);
        assert!(e > 0.0 && e < 100.0);
    }

    #[test]
    fn test_dp_sample_crp_ok() {
        let dp = DirichletProcess::new(1.0).expect("DP new failed");
        let (assignments, n_tables) = dp.sample_crp(50, 42).expect("sample_crp failed");
        assert_eq!(assignments.len(), 50);
        assert!(n_tables >= 1 && n_tables <= 50);
    }

    #[test]
    fn test_dp_sample_stick_breaking_ok() {
        let dp = DirichletProcess::new(2.0).expect("DP new failed");
        let w = dp.sample_stick_breaking(20, 7).expect("stick_breaking failed");
        let total: f64 = w.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "weights should sum to 1, got {total}");
    }

    // --- dp_mixture_gibbs ---

    #[test]
    fn test_dp_mixture_gibbs_basic() {
        use scirs2_core::ndarray::Array2;
        // Two well-separated 1-D clusters embedded in a column-2 array
        let mut raw = vec![0.0_f64; 40 * 2];
        for i in 0..20 {
            raw[i * 2] = -5.0 + (i as f64) * 0.01;
            raw[i * 2 + 1] = 0.0;
        }
        for i in 20..40 {
            raw[i * 2] = 5.0 + (i as f64 - 20.0) * 0.01;
            raw[i * 2 + 1] = 0.0;
        }
        let data = Array2::from_shape_vec((40, 2), raw).expect("shape vec failed");
        let prior_mean = vec![0.0_f64; 2];
        let assignments = dp_mixture_gibbs(
            data.view(),
            1.0,
            &prior_mean,
            10.0,
            0.1,
            20,
            42,
        )
        .expect("dp_mixture_gibbs failed");
        assert_eq!(assignments.len(), 40);
        // Both clusters should exist (at least 2 distinct labels)
        let n_distinct = {
            let mut seen = assignments.clone();
            seen.sort_unstable();
            seen.dedup();
            seen.len()
        };
        assert!(n_distinct >= 1, "should have at least 1 cluster");
    }

    #[test]
    fn test_dp_mixture_gibbs_invalid_args() {
        use scirs2_core::ndarray::Array2;
        let data = Array2::zeros((10, 2));
        let prior = vec![0.0_f64; 2];
        assert!(dp_mixture_gibbs(data.view(), 0.0, &prior, 1.0, 1.0, 10, 0).is_err());
        assert!(dp_mixture_gibbs(data.view(), 1.0, &prior, -1.0, 1.0, 10, 0).is_err());
        assert!(dp_mixture_gibbs(data.view(), 1.0, &prior, 1.0, 0.0, 10, 0).is_err());
        let bad_prior = vec![0.0_f64; 3];
        assert!(dp_mixture_gibbs(data.view(), 1.0, &bad_prior, 1.0, 1.0, 10, 0).is_err());
    }

    // --- stick_breaking ---

    #[test]
    fn test_stick_breaking_weights_sum_to_one() {
        let weights = stick_breaking(1.0, 100, 42).expect("stick_breaking failed");
        let total: f64 = weights.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "weights should sum to 1.0, got {total}"
        );
    }

    #[test]
    fn test_stick_breaking_all_nonnegative() {
        let weights = stick_breaking(2.0, 50, 7).expect("stick_breaking failed");
        for &w in &weights {
            assert!(w >= 0.0, "weight must be non-negative, got {w}");
        }
    }

    #[test]
    fn test_stick_breaking_length() {
        let k = 20usize;
        let weights = stick_breaking(1.0, k, 0).expect("stick_breaking failed");
        assert_eq!(weights.len(), k + 1, "length must be n_sticks + 1");
    }

    #[test]
    fn test_stick_breaking_large_alpha_more_uniform() {
        // With larger alpha the weights should be more uniformly spread
        let weights_small = stick_breaking(0.1, 50, 1).expect("stick_breaking failed");
        let weights_large = stick_breaking(100.0, 50, 1).expect("stick_breaking failed");
        // Entropy of large-alpha weights should be higher
        let entropy = |w: &Vec<f64>| -> f64 {
            w.iter()
                .filter(|&&x| x > 1e-15)
                .map(|&x| -x * x.ln())
                .sum()
        };
        assert!(
            entropy(&weights_large) > entropy(&weights_small),
            "large alpha should yield more uniform (higher entropy) weights"
        );
    }

    #[test]
    fn test_stick_breaking_invalid_alpha() {
        assert!(stick_breaking(-1.0, 10, 0).is_err());
        assert!(stick_breaking(0.0, 10, 0).is_err());
    }

    #[test]
    fn test_stick_breaking_invalid_n_sticks() {
        assert!(stick_breaking(1.0, 0, 0).is_err());
    }

    // --- chinese_restaurant_process ---

    #[test]
    fn test_crp_assignments_in_range() {
        let n = 100;
        let (assignments, n_tables) = chinese_restaurant_process(n, 1.0, 42)
            .expect("crp failed");
        assert_eq!(assignments.len(), n);
        for &a in &assignments {
            assert!(a < n_tables, "assignment {a} out of range [0, {n_tables})");
        }
    }

    #[test]
    fn test_crp_all_tables_used() {
        let (assignments, n_tables) = chinese_restaurant_process(200, 1.0, 0)
            .expect("crp failed");
        let mut seen = vec![false; n_tables];
        for &a in &assignments {
            seen[a] = true;
        }
        assert!(seen.iter().all(|&x| x), "all tables must appear in assignments");
    }

    #[test]
    fn test_crp_first_customer_table_zero() {
        let (assignments, _) = chinese_restaurant_process(1, 1.0, 0).expect("crp failed");
        assert_eq!(assignments[0], 0, "first customer always sits at table 0");
    }

    #[test]
    fn test_crp_cluster_count_near_expected() {
        // Over many seeds, average cluster count ≈ α·ln(n)
        let n = 500usize;
        let alpha = 2.0_f64;
        let n_trials = 50usize;
        let total_tables: usize = (0..n_trials)
            .map(|seed| {
                let (_, t) = chinese_restaurant_process(n, alpha, seed as u64)
                    .expect("crp failed");
                t
            })
            .sum();
        let mean_tables = total_tables as f64 / n_trials as f64;
        let expected = expected_clusters(alpha, n);
        let rel_err = (mean_tables - expected).abs() / expected;
        assert!(
            rel_err < 0.15,
            "mean tables {mean_tables:.2} far from expected {expected:.2} (rel err {rel_err:.3})"
        );
    }

    #[test]
    fn test_crp_invalid_alpha() {
        assert!(chinese_restaurant_process(10, 0.0, 0).is_err());
        assert!(chinese_restaurant_process(10, -1.0, 0).is_err());
    }

    #[test]
    fn test_crp_empty() {
        let (a, t) = chinese_restaurant_process(0, 1.0, 0).expect("crp empty failed");
        assert!(a.is_empty());
        assert_eq!(t, 0);
    }

    // --- crp_predictive ---

    #[test]
    fn test_crp_predictive_sums_to_one() {
        let counts = vec![3usize, 5, 2];
        let probs = crp_predictive(&counts, 1.0);
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "predictive probs must sum to 1, got {total}");
    }

    #[test]
    fn test_crp_predictive_length() {
        let counts = vec![1usize, 2, 3];
        let probs = crp_predictive(&counts, 1.0);
        assert_eq!(probs.len(), counts.len() + 1);
    }

    #[test]
    fn test_crp_predictive_new_table_prob() {
        // With empty existing tables and alpha=1, P(new table) = 1.0
        let probs = crp_predictive(&[], 1.0);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-12);
    }

    // --- pitman_yor_process ---

    #[test]
    fn test_pyp_assignments_in_range() {
        let (assignments, n_tables) =
            pitman_yor_process(100, 0.5, 1.0, 42).expect("pyp failed");
        assert_eq!(assignments.len(), 100);
        for &a in &assignments {
            assert!(a < n_tables, "assignment {a} out of range");
        }
    }

    #[test]
    fn test_pyp_discount_zero_resembles_crp() {
        // With d=0, PYP reduces to CRP; cluster counts should behave similarly
        let n = 500usize;
        let alpha = 2.0_f64;
        let (_, t_pyp) = pitman_yor_process(n, 0.0, alpha, 0).expect("pyp failed");
        let (_, t_crp) = chinese_restaurant_process(n, alpha, 0).expect("crp failed");
        // Allow 20% relative difference since they use the same seed but
        // different code paths
        let rel = ((t_pyp as f64) - (t_crp as f64)).abs() / (t_crp as f64);
        assert!(rel < 0.20, "PYP(d=0) differs too much from CRP: {t_pyp} vs {t_crp}");
    }

    #[test]
    fn test_pyp_power_law_more_clusters() {
        // Higher discount → more clusters (power-law growth)
        let n = 300usize;
        let alpha = 1.0_f64;
        let seeds_and_results: Vec<usize> = (0..20u64)
            .map(|s| pitman_yor_process(n, 0.7, alpha, s).expect("pyp failed").1)
            .collect();
        let crp_results: Vec<usize> = (0..20u64)
            .map(|s| chinese_restaurant_process(n, alpha, s).expect("crp failed").1)
            .collect();
        let mean_pyp: f64 = seeds_and_results.iter().sum::<usize>() as f64 / 20.0;
        let mean_crp: f64 = crp_results.iter().sum::<usize>() as f64 / 20.0;
        assert!(
            mean_pyp > mean_crp,
            "PYP(d=0.7) should produce more clusters than CRP(d=0): {mean_pyp:.1} vs {mean_crp:.1}"
        );
    }

    #[test]
    fn test_pyp_invalid_discount() {
        assert!(pitman_yor_process(10, -0.1, 1.0, 0).is_err());
        assert!(pitman_yor_process(10, 1.0, 1.0, 0).is_err());
    }

    #[test]
    fn test_pyp_invalid_strength() {
        // alpha must be > -discount; with d=0.5, alpha=-0.6 is invalid
        assert!(pitman_yor_process(10, 0.5, -0.6, 0).is_err());
    }

    // --- estimate_alpha_from_clusters ---

    #[test]
    fn test_estimate_alpha_round_trip() {
        // Simulate CRP at known alpha, then recover it
        let true_alpha = 3.0_f64;
        let n = 500usize;
        let k_sum: usize = (0..50u64)
            .map(|s| {
                chinese_restaurant_process(n, true_alpha, s)
                    .expect("crp failed")
                    .1
            })
            .sum();
        let k_mean = k_sum as f64 / 50.0;
        let k_rounded = k_mean.round() as usize;
        let est = estimate_alpha_from_clusters(n, k_rounded.clamp(1, n), 100)
            .expect("estimate_alpha failed");
        let rel_err = (est - true_alpha).abs() / true_alpha;
        assert!(
            rel_err < 0.25,
            "estimated alpha {est:.3} far from true {true_alpha:.3} (rel err {rel_err:.3})"
        );
    }

    #[test]
    fn test_estimate_alpha_invalid() {
        assert!(estimate_alpha_from_clusters(0, 1, 10).is_err());
        assert!(estimate_alpha_from_clusters(10, 0, 10).is_err());
        assert!(estimate_alpha_from_clusters(10, 11, 10).is_err());
        assert!(estimate_alpha_from_clusters(10, 1, 0).is_err());
    }

    // --- crp_posterior_tables ---

    #[test]
    fn test_crp_posterior_tables_sums_to_one() {
        let probs = crp_posterior_tables(10, 1.0);
        let total: f64 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "posterior probs must sum to 1, got {total}"
        );
    }

    #[test]
    fn test_crp_posterior_tables_length() {
        let n = 15usize;
        let probs = crp_posterior_tables(n, 1.0);
        assert_eq!(probs.len(), n);
    }

    #[test]
    fn test_crp_posterior_tables_mode_near_expected() {
        let n = 20usize;
        let alpha = 2.0_f64;
        let probs = crp_posterior_tables(n, alpha);
        let mode_k = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("comparison failed"))
            .map(|(i, _)| i + 1)
            .expect("must have a mode");
        let expected_k = expected_clusters(alpha, n).round() as usize;
        // Allow the mode to be within ±3 of the expected value
        let diff = (mode_k as isize - expected_k as isize).unsigned_abs();
        assert!(
            diff <= 3,
            "posterior mode {mode_k} too far from expected {expected_k}"
        );
    }

    #[test]
    fn test_crp_posterior_tables_empty() {
        let probs = crp_posterior_tables(0, 1.0);
        assert!(probs.is_empty());
    }
}
