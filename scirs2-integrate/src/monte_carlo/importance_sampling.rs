//! Advanced importance sampling methods for Monte Carlo integration.
//!
//! Importance sampling replaces the uniform measure with a proposal distribution `q`
//! that concentrates mass where `|f · p|` is large, reducing variance compared to
//! naive Monte Carlo.
//!
//! ## Methods
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`importance_sampling_integral`] | Standard IS: `E_q[f(x)·p(x)/q(x)]` |
//! | [`self_normalized_is`] | SNIS for unnormalized target density |
//! | [`sequential_is`] | Sequential (weighted) IS with resampling |
//! | [`effective_sample_size`] | ESS = `(Σwᵢ)² / Σwᵢ²` |
//! | [`stratified_sampling`] | Stratified partition of the domain for variance reduction |
//!
//! ## References
//!
//! - Owen, A. B. (2013). *Monte Carlo theory, methods and examples.*
//! - Doucet, A., de Freitas, N., Gordon, N. (2001). *Sequential Monte Carlo Methods
//!   in Practice.* Springer.

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution, Normal, Uniform};

// ─────────────────────────────────────────────────────────────────────────────
// Result types
// ─────────────────────────────────────────────────────────────────────────────

/// Result of an importance sampling integration.
#[derive(Debug, Clone)]
pub struct ImportanceSamplingResult {
    /// Estimated integral value.
    pub value: f64,
    /// Estimated variance of the estimator.
    pub variance: f64,
    /// Effective sample size: measures how many i.i.d. samples the weighted
    /// sample is equivalent to.  A small ESS indicates weight degeneracy.
    pub effective_sample_size: f64,
    /// Number of function evaluations used.
    pub n_evals: usize,
    /// Weights used in the estimator (normalised so they sum to 1).
    pub normalised_weights: Vec<f64>,
}

/// Result of a stratified sampling integration.
#[derive(Debug, Clone)]
pub struct StratifiedResult {
    /// Estimated integral value.
    pub value: f64,
    /// Estimated standard error.
    pub std_error: f64,
    /// Per-stratum estimates.
    pub stratum_values: Vec<f64>,
    /// Per-stratum standard errors.
    pub stratum_errors: Vec<f64>,
    /// Total number of function evaluations.
    pub n_evals: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// effective_sample_size
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the **effective sample size** (ESS) for a set of importance weights.
///
/// The ESS is defined as
///
/// ```text
/// ESS = (Σᵢ wᵢ)² / Σᵢ wᵢ²
/// ```
///
/// A value close to `weights.len()` means the weights are nearly uniform
/// (good), while a value close to 1 indicates severe weight degeneracy.
///
/// # Arguments
///
/// * `weights` – Raw (unnormalised) importance weights `wᵢ = p(xᵢ) / q(xᵢ)`.
///
/// # Returns
///
/// ESS ∈ `[1, weights.len()]`.
///
/// # Examples
///
/// ```
/// use scirs2_integrate::monte_carlo::importance_sampling::effective_sample_size;
///
/// // Uniform weights → ESS equals N.
/// let w = vec![1.0_f64; 100];
/// let ess = effective_sample_size(&w);
/// assert!((ess - 100.0).abs() < 1e-10);
/// ```
pub fn effective_sample_size(weights: &[f64]) -> f64 {
    if weights.is_empty() {
        return 0.0;
    }
    let sum: f64 = weights.iter().sum();
    let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
    if sum_sq == 0.0 {
        return 0.0;
    }
    (sum * sum) / sum_sq
}

// ─────────────────────────────────────────────────────────────────────────────
// importance_sampling_integral
// ─────────────────────────────────────────────────────────────────────────────

/// Standard importance sampling estimator of ∫ f(x) p(x) dx.
///
/// Samples are drawn from proposal `q` and the estimate is
///
/// ```text
/// Î = (1/N) Σᵢ f(xᵢ) · w(xᵢ)   where w(xᵢ) = p(xᵢ) / q(xᵢ),  xᵢ ~ q.
/// ```
///
/// # Type parameters
///
/// * `F`       – integrand `f : ℝᵈ → ℝ`
/// * `P`       – target density `p : ℝᵈ → ℝ≥0`
/// * `Q`       – proposal density `q : ℝᵈ → ℝ>0`
/// * `Sampler` – samples a point from `q` given an RNG
///
/// # Arguments
///
/// * `f`            – the integrand
/// * `target_p`     – target density (need not be normalised; see [`self_normalized_is`])
/// * `proposal_q`   – proposal density (must be normalised)
/// * `sampler`      – draws `x ~ q`
/// * `n_samples`    – number of samples
/// * `seed`         – optional RNG seed for reproducibility
///
/// # Examples
///
/// ```
/// use scirs2_integrate::monte_carlo::importance_sampling::importance_sampling_integral;
/// use scirs2_core::ndarray::{Array1, ArrayView1};
/// use scirs2_core::random::prelude::*;
/// use std::f64::consts::PI;
///
/// // Integrate exp(-x²) over ℝ using N(0,1) as proposal.
/// // p(x) = 1 (Lebesgue measure), q(x) = N(0,1).
/// // The true value is √π ≈ 1.7725.
/// let result = importance_sampling_integral(
///     |x: ArrayView1<f64>| (-x[0] * x[0]).exp(),
///     |_x: ArrayView1<f64>| 1.0_f64,                      // p = Lebesgue
///     |x: ArrayView1<f64>| {
///         let z = x[0];
///         (-0.5 * z * z).exp() / (2.0 * PI).sqrt()        // q = N(0,1)
///     },
///     |rng: &mut StdRng, _d: usize| {
///         let n = Normal::new(0.0, 1.0).expect("valid params");
///         Array1::from_elem(1, n.sample(rng))
///     },
///     50000,
///     Some(42),
/// ).expect("IS integration failed");
///
/// assert!((result.value - PI.sqrt()).abs() < 0.05);
/// ```
pub fn importance_sampling_integral<F, P, Q, Sampler>(
    f: F,
    target_p: P,
    proposal_q: Q,
    sampler: Sampler,
    n_samples: usize,
    seed: Option<u64>,
) -> IntegrateResult<ImportanceSamplingResult>
where
    F: Fn(ArrayView1<f64>) -> f64,
    P: Fn(ArrayView1<f64>) -> f64,
    Q: Fn(ArrayView1<f64>) -> f64,
    Sampler: Fn(&mut StdRng, usize) -> Array1<f64>,
{
    if n_samples == 0 {
        return Err(IntegrateError::ValueError(
            "n_samples must be positive".to_string(),
        ));
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut os = scirs2_core::random::rng();
            StdRng::from_rng(&mut os)
        }
    };

    // We don't know the dimension upfront; sample one point to discover it.
    let probe = sampler(&mut rng, 1);
    let dim = probe.len().max(1);

    let mut weights: Vec<f64> = Vec::with_capacity(n_samples);
    let mut weighted_f: Vec<f64> = Vec::with_capacity(n_samples);
    let mut n_valid = 0usize;

    // Include the probe sample we already drew.
    {
        let x = probe.view();
        let pval = target_p(x);
        let qval = proposal_q(x);
        if qval > 1e-300 && pval.is_finite() && qval.is_finite() {
            let w = pval / qval;
            let fval = f(x);
            if fval.is_finite() && w.is_finite() {
                weights.push(w);
                weighted_f.push(fval * w);
                n_valid += 1;
            }
        }
    }

    for _ in 1..n_samples {
        let x = sampler(&mut rng, dim);
        let pval = target_p(x.view());
        let qval = proposal_q(x.view());

        if qval <= 1e-300 || !pval.is_finite() || !qval.is_finite() {
            continue;
        }

        let w = pval / qval;
        let fval = f(x.view());

        if !fval.is_finite() || !w.is_finite() {
            continue;
        }

        weights.push(w);
        weighted_f.push(fval * w);
        n_valid += 1;
    }

    if n_valid < 2 {
        return Err(IntegrateError::ConvergenceError(
            "Too few valid IS samples (fewer than 2); check that proposal covers target support"
                .to_string(),
        ));
    }

    let n_f = n_valid as f64;
    let sum_wf: f64 = weighted_f.iter().sum();
    let value = sum_wf / n_f;

    // Variance estimate via the delta method.
    let sum_sq: f64 = weighted_f.iter().map(|v| (v - value) * (v - value)).sum();
    let variance = sum_sq / (n_f * (n_f - 1.0));

    let ess = effective_sample_size(&weights);

    // Normalise weights for diagnostics.
    let w_sum: f64 = weights.iter().sum();
    let normalised_weights: Vec<f64> = weights
        .iter()
        .map(|&w| if w_sum > 0.0 { w / w_sum } else { 0.0 })
        .collect();

    Ok(ImportanceSamplingResult {
        value,
        variance,
        effective_sample_size: ess,
        n_evals: n_valid,
        normalised_weights,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// self_normalized_is
// ─────────────────────────────────────────────────────────────────────────────

/// Self-normalised importance sampling (SNIS) for an **unnormalised** target.
///
/// When the target `π(x)` is only known up to a constant `Z`:
/// `p(x) = π(x) / Z`, SNIS estimates the integral without knowing `Z`:
///
/// ```text
/// Î = Σᵢ f(xᵢ) w̃(xᵢ) / Σᵢ w̃(xᵢ)     where  w̃(xᵢ) = π(xᵢ) / q(xᵢ).
/// ```
///
/// This estimator is **biased** but consistent.  The bias is
/// `O(1/N)` and typically negligible for moderate `N`.
///
/// # Arguments
///
/// * `f`           – integrand
/// * `unnorm_pi`   – unnormalised target π (need not integrate to 1)
/// * `proposal_q`  – normalised proposal density
/// * `sampler`     – draws from `q`
/// * `n_samples`   – Monte Carlo sample size
/// * `seed`        – optional seed
///
/// # Examples
///
/// ```
/// use scirs2_integrate::monte_carlo::importance_sampling::self_normalized_is;
/// use scirs2_core::ndarray::{Array1, ArrayView1};
/// use scirs2_core::random::prelude::*;
/// use std::f64::consts::PI;
///
/// // Estimate E[x²] under N(1, 0.5²) using N(0,1) as proposal.
/// // True value: Var + mean² = 0.25 + 1 = 1.25.
/// let result = self_normalized_is(
///     |x: ArrayView1<f64>| x[0] * x[0],
///     |x: ArrayView1<f64>| {
///         let z = (x[0] - 1.0) / 0.5;
///         (-0.5 * z * z).exp()   // unnormalised N(1, 0.5²)
///     },
///     |x: ArrayView1<f64>| {
///         let z = x[0];
///         (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
///     },
///     |rng: &mut StdRng, _d: usize| {
///         let n = Normal::new(0.0, 1.0).expect("valid params");
///         Array1::from_elem(1, n.sample(rng))
///     },
///     50000,
///     Some(7),
/// ).expect("SNIS failed");
///
/// assert!((result.value - 1.25).abs() < 0.05);
/// ```
pub fn self_normalized_is<F, Pi, Q, Sampler>(
    f: F,
    unnorm_pi: Pi,
    proposal_q: Q,
    sampler: Sampler,
    n_samples: usize,
    seed: Option<u64>,
) -> IntegrateResult<ImportanceSamplingResult>
where
    F: Fn(ArrayView1<f64>) -> f64,
    Pi: Fn(ArrayView1<f64>) -> f64,
    Q: Fn(ArrayView1<f64>) -> f64,
    Sampler: Fn(&mut StdRng, usize) -> Array1<f64>,
{
    if n_samples == 0 {
        return Err(IntegrateError::ValueError(
            "n_samples must be positive".to_string(),
        ));
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut os = scirs2_core::random::rng();
            StdRng::from_rng(&mut os)
        }
    };

    let probe = sampler(&mut rng, 1);
    let dim = probe.len().max(1);

    let mut raw_weights: Vec<f64> = Vec::with_capacity(n_samples);
    let mut wf_vals: Vec<f64> = Vec::with_capacity(n_samples);

    let process = |x: ArrayView1<f64>, rw: &mut Vec<f64>, wf: &mut Vec<f64>| -> bool {
        let pi_val = unnorm_pi(x);
        let q_val = proposal_q(x);
        if q_val <= 1e-300 || !pi_val.is_finite() || !q_val.is_finite() {
            return false;
        }
        let w = pi_val / q_val;
        let fval = f(x);
        if !fval.is_finite() || !w.is_finite() {
            return false;
        }
        rw.push(w);
        wf.push(fval * w);
        true
    };

    process(probe.view(), &mut raw_weights, &mut wf_vals);

    for _ in 1..n_samples {
        let x = sampler(&mut rng, dim);
        process(x.view(), &mut raw_weights, &mut wf_vals);
    }

    let n_valid = raw_weights.len();

    if n_valid < 2 {
        return Err(IntegrateError::ConvergenceError(
            "Too few valid SNIS samples".to_string(),
        ));
    }

    let w_sum: f64 = raw_weights.iter().sum();
    if w_sum == 0.0 {
        return Err(IntegrateError::ConvergenceError(
            "All importance weights are zero".to_string(),
        ));
    }

    // SNIS estimate: Σ f·w̃ / Σ w̃
    let wf_sum: f64 = wf_vals.iter().sum();
    let value = wf_sum / w_sum;

    // Variance of the SNIS estimator via the delta-method approximation.
    let n_f = n_valid as f64;
    let var_fw: f64 = wf_vals
        .iter()
        .map(|&x| (x / w_sum - value / n_f) * (x / w_sum - value / n_f))
        .sum::<f64>()
        * n_f
        / (n_f - 1.0);
    let variance = var_fw / n_f;

    let ess = effective_sample_size(&raw_weights);

    let normalised_weights: Vec<f64> = raw_weights.iter().map(|&w| w / w_sum).collect();

    Ok(ImportanceSamplingResult {
        value,
        variance,
        effective_sample_size: ess,
        n_evals: n_valid,
        normalised_weights,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// sequential_is
// ─────────────────────────────────────────────────────────────────────────────

/// Sequential Importance Sampling (SIS) with **systematic resampling**.
///
/// Propagates `n_particles` weighted particles through `n_steps` transition
/// steps, resampling when the ESS drops below `resample_threshold * n_particles`.
/// Returns the final estimate of `E[f(x)]` under the final target.
///
/// # Algorithm
///
/// 1. Initialise particles from `initial_sampler`.
/// 2. For each step `t`:
///    a. Propagate each particle through `transition`.
///    b. Reweight: `w̃ᵢ ∝ wᵢ · (target_t(xᵢ) / proposal_t(xᵢ))`.
///    c. If `ESS < threshold`, do systematic resampling.
/// 3. Return `Σ wᵢ f(xᵢ) / Σ wᵢ`.
///
/// # Arguments
///
/// * `f`                  – the function whose expectation we estimate
/// * `initial_sampler`    – samples the initial particle cloud
/// * `transition`         – propagates a particle one step forward: `(x, rng) → x_new`
/// * `incremental_weight` – log incremental weight for each step `t` and particle `x`:
///   `log(γ_t(x) / γ_{t-1}(x) / q_t(x|x_{t-1}))`
/// * `n_particles`        – number of particles
/// * `n_steps`            – number of SMC steps
/// * `resample_threshold` – resample when `ESS < threshold * n_particles` (default 0.5)
/// * `seed`               – optional seed
///
/// # Returns
///
/// Final estimate and diagnostics packaged in [`ImportanceSamplingResult`].
pub fn sequential_is<F, InitSampler, Transition, IncWeight>(
    f: F,
    initial_sampler: InitSampler,
    transition: Transition,
    incremental_weight: IncWeight,
    n_particles: usize,
    n_steps: usize,
    resample_threshold: Option<f64>,
    seed: Option<u64>,
) -> IntegrateResult<ImportanceSamplingResult>
where
    F: Fn(ArrayView1<f64>) -> f64,
    InitSampler: Fn(&mut StdRng) -> Array1<f64>,
    Transition: Fn(ArrayView1<f64>, &mut StdRng) -> Array1<f64>,
    IncWeight: Fn(ArrayView1<f64>, usize) -> f64,
{
    if n_particles == 0 {
        return Err(IntegrateError::ValueError(
            "n_particles must be positive".to_string(),
        ));
    }

    let threshold = resample_threshold.unwrap_or(0.5) * (n_particles as f64);

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut os = scirs2_core::random::rng();
            StdRng::from_rng(&mut os)
        }
    };

    // Initialise
    let mut particles: Vec<Array1<f64>> = (0..n_particles)
        .map(|_| initial_sampler(&mut rng))
        .collect();
    let mut log_weights: Vec<f64> = vec![-(n_particles as f64).ln(); n_particles];

    for step in 0..n_steps {
        // Propagate + reweight
        let mut new_particles: Vec<Array1<f64>> = Vec::with_capacity(n_particles);
        let mut new_log_w: Vec<f64> = Vec::with_capacity(n_particles);

        for (i, particle) in particles.iter().enumerate() {
            let x_new = transition(particle.view(), &mut rng);
            let lw_inc = incremental_weight(x_new.view(), step);
            if lw_inc.is_nan() {
                // Propagate but zero weight
                new_particles.push(x_new);
                new_log_w.push(f64::NEG_INFINITY);
            } else {
                new_log_w.push(log_weights[i] + lw_inc);
                new_particles.push(x_new);
            }
        }

        // Normalise in log space (shift by max to avoid underflow)
        let lw_max = new_log_w
            .iter()
            .cloned()
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);

        if !lw_max.is_finite() {
            return Err(IntegrateError::ConvergenceError(
                "All particle weights collapsed to zero at SIS step".to_string(),
            ));
        }

        let raw_w: Vec<f64> = new_log_w.iter().map(|&lw| (lw - lw_max).exp()).collect();
        let w_sum: f64 = raw_w.iter().sum();
        let norm_w: Vec<f64> = raw_w.iter().map(|&w| w / w_sum).collect();

        let ess = effective_sample_size(&norm_w);

        particles = new_particles;
        log_weights = norm_w.iter().map(|&w| w.ln()).collect();

        // Systematic resampling if ESS below threshold
        if ess < threshold {
            let resampled = systematic_resample(&norm_w, n_particles, &mut rng);
            particles = resampled
                .iter()
                .map(|&idx| particles[idx].clone())
                .collect();
            let log_uniform = -(n_particles as f64).ln();
            log_weights = vec![log_uniform; n_particles];
        }
    }

    // Final estimate
    let lw_max = log_weights
        .iter()
        .cloned()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);

    let raw_w: Vec<f64> = log_weights.iter().map(|&lw| (lw - lw_max).exp()).collect();
    let w_sum: f64 = raw_w.iter().sum();
    let norm_w: Vec<f64> = raw_w.iter().map(|&w| w / w_sum).collect();

    let value: f64 = particles
        .iter()
        .zip(norm_w.iter())
        .map(|(x, &w)| w * f(x.view()))
        .sum();

    let variance: f64 = {
        let n_f = n_particles as f64;
        particles
            .iter()
            .zip(norm_w.iter())
            .map(|(x, &w)| {
                let diff = w * f(x.view()) - value / n_f;
                diff * diff
            })
            .sum::<f64>()
            / (n_f - 1.0)
            / n_f
    };

    let ess = effective_sample_size(&norm_w);

    Ok(ImportanceSamplingResult {
        value,
        variance,
        effective_sample_size: ess,
        n_evals: n_particles * (n_steps + 1),
        normalised_weights: norm_w,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// stratified_sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Stratified Monte Carlo integration in one dimension.
///
/// Divides `[a, b]` into `n_strata` equal sub-intervals and draws
/// `samples_per_stratum` uniform points from each, then combines stratum
/// estimates.  The variance is reduced by a factor of up to `n_strata`
/// compared to simple Monte Carlo.
///
/// For `d`-dimensional integrands, stratification is applied to the **first**
/// dimension and uniform sampling is used for the remaining dimensions.
///
/// # Arguments
///
/// * `f`                   – integrand `ℝᵈ → ℝ`
/// * `ranges`              – integration domain `[(a₁,b₁), …, (aₐ,bₐ)]`
/// * `n_strata`            – number of strata along the first dimension
/// * `samples_per_stratum` – i.i.d. samples drawn within each stratum
/// * `seed`                – optional seed
///
/// # Examples
///
/// ```
/// use scirs2_integrate::monte_carlo::importance_sampling::stratified_sampling;
/// use scirs2_core::ndarray::ArrayView1;
///
/// // Integrate sin(x) over [0, π]: exact value = 2.
/// let result = stratified_sampling(
///     |x: ArrayView1<f64>| x[0].sin(),
///     &[(0.0, std::f64::consts::PI)],
///     20,
///     500,
///     Some(13),
/// ).expect("stratified sampling failed");
///
/// assert!((result.value - 2.0).abs() < 0.01);
/// ```
pub fn stratified_sampling<F>(
    f: F,
    ranges: &[(f64, f64)],
    n_strata: usize,
    samples_per_stratum: usize,
    seed: Option<u64>,
) -> IntegrateResult<StratifiedResult>
where
    F: Fn(ArrayView1<f64>) -> f64,
{
    if ranges.is_empty() {
        return Err(IntegrateError::ValueError(
            "ranges cannot be empty".to_string(),
        ));
    }
    if n_strata == 0 || samples_per_stratum == 0 {
        return Err(IntegrateError::ValueError(
            "n_strata and samples_per_stratum must be positive".to_string(),
        ));
    }

    let n_dims = ranges.len();
    let (a0, b0) = ranges[0];
    let stratum_width = (b0 - a0) / (n_strata as f64);

    // Volume of the remaining dimensions (dims 1..d)
    let remaining_volume: f64 = ranges[1..].iter().map(|(a, b)| b - a).product();

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut os = scirs2_core::random::rng();
            StdRng::from_rng(&mut os)
        }
    };

    let uniform01 = Uniform::new(0.0_f64, 1.0).expect("valid uniform bounds [0,1)");

    let mut stratum_values = Vec::with_capacity(n_strata);
    let mut stratum_errors = Vec::with_capacity(n_strata);
    let mut total_evals = 0usize;

    for k in 0..n_strata {
        let lo = a0 + k as f64 * stratum_width;
        let hi = lo + stratum_width;

        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;

        for _ in 0..samples_per_stratum {
            let mut point = Array1::<f64>::zeros(n_dims);
            // Stratified sample in dimension 0
            point[0] = lo + (hi - lo) * uniform01.sample(&mut rng);
            // Uniform samples in remaining dimensions
            for (j, &(aj, bj)) in ranges[1..].iter().enumerate() {
                point[j + 1] = aj + (bj - aj) * uniform01.sample(&mut rng);
            }
            let val = f(point.view());
            sum += val;
            sum_sq += val * val;
        }

        let n_f = samples_per_stratum as f64;
        let mean = sum / n_f;
        // Variance of the mean within this stratum
        let var_mean = if samples_per_stratum > 1 {
            (sum_sq - sum * sum / n_f) / (n_f * (n_f - 1.0))
        } else {
            0.0
        };

        // Scale by stratum volume
        let stratum_vol = stratum_width * remaining_volume;
        stratum_values.push(mean * stratum_vol);
        stratum_errors.push(var_mean.sqrt() * stratum_vol);
        total_evals += samples_per_stratum;
    }

    let value: f64 = stratum_values.iter().sum();
    // Variance of total estimate = sum of variances (stratums are independent)
    let total_var: f64 = stratum_errors.iter().map(|e| e * e).sum();
    let std_error = total_var.sqrt();

    Ok(StratifiedResult {
        value,
        std_error,
        stratum_values,
        stratum_errors,
        n_evals: total_evals,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Systematic resampling: given normalised weights, draw `n` indices.
/// Uses O(N) time and preserves diversity better than multinomial resampling.
fn systematic_resample(weights: &[f64], n: usize, rng: &mut StdRng) -> Vec<usize> {
    let uniform01 = Uniform::new(0.0_f64, 1.0).expect("valid bounds");
    let u0: f64 = uniform01.sample(rng);

    let mut indices = Vec::with_capacity(n);
    let step = 1.0 / (n as f64);
    let mut cumsum = 0.0_f64;
    let mut j = 0usize;

    for i in 0..n {
        let u = (u0 + i as f64) * step;
        while cumsum < u && j < weights.len() - 1 {
            cumsum += weights[j];
            j += 1;
        }
        indices.push(j.min(weights.len() - 1));
    }

    indices
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_ess_uniform() {
        let w = vec![1.0_f64; 100];
        let ess = effective_sample_size(&w);
        assert!((ess - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_ess_degenerate() {
        let mut w = vec![0.0_f64; 99];
        w.push(1.0);
        let ess = effective_sample_size(&w);
        // All weight on one particle → ESS ≈ 1
        assert!((ess - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_importance_sampling_integral_gaussian() {
        // Integrate exp(-x²) over ℝ using N(0,1) proposal.
        // True value: √π.
        let result = importance_sampling_integral(
            |x: ArrayView1<f64>| (-x[0] * x[0]).exp(),
            |_x: ArrayView1<f64>| 1.0_f64,
            |x: ArrayView1<f64>| {
                let z = x[0];
                (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
            },
            |rng: &mut StdRng, _d: usize| {
                let n = Normal::new(0.0, 1.0).expect("valid");
                Array1::from_elem(1, n.sample(rng))
            },
            50000,
            Some(42),
        )
        .expect("IS failed");

        assert!((result.value - PI.sqrt()).abs() < 0.05);
        assert!(result.effective_sample_size > 1.0);
    }

    #[test]
    fn test_self_normalized_is() {
        // E[x²] under N(1, 0.5²) = Var + mean² = 0.25 + 1 = 1.25
        let result = self_normalized_is(
            |x: ArrayView1<f64>| x[0] * x[0],
            |x: ArrayView1<f64>| {
                let z = (x[0] - 1.0) / 0.5;
                (-0.5 * z * z).exp()
            },
            |x: ArrayView1<f64>| {
                let z = x[0];
                (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
            },
            |rng: &mut StdRng, _d: usize| {
                let n = Normal::new(0.0, 1.0).expect("valid");
                Array1::from_elem(1, n.sample(rng))
            },
            50000,
            Some(7),
        )
        .expect("SNIS failed");

        assert!((result.value - 1.25).abs() < 0.05);
    }

    #[test]
    fn test_stratified_sampling_sin() {
        let result = stratified_sampling(
            |x: ArrayView1<f64>| x[0].sin(),
            &[(0.0, PI)],
            20,
            500,
            Some(13),
        )
        .expect("stratified failed");

        assert!((result.value - 2.0).abs() < 0.01);
        assert_eq!(result.stratum_values.len(), 20);
    }

    #[test]
    fn test_sequential_is_simple() {
        // SIS for estimating E[x²] under N(0,1) — trivially = 1.
        // Particles stay at their initial values (identity transition).
        // Incremental weight = 0 (log ratio of N(0,1)/N(0,1)).
        let n_particles = 500;
        let result = sequential_is(
            |x: ArrayView1<f64>| x[0] * x[0],
            |rng: &mut StdRng| {
                let n = Normal::new(0.0, 1.0).expect("valid");
                Array1::from_elem(1, n.sample(rng))
            },
            |x: ArrayView1<f64>, _rng: &mut StdRng| {
                // identity: particles don't move
                x.to_owned()
            },
            |_x: ArrayView1<f64>, _step: usize| 0.0_f64, // log weight increment = 0
            n_particles,
            3,
            Some(0.5),
            Some(99),
        )
        .expect("SIS failed");

        // E[x²] under N(0,1) = 1 — allow generous tolerance since SIS is biased
        assert!(
            (result.value - 1.0).abs() < 0.3,
            "SIS estimate {} too far from 1.0",
            result.value
        );
    }
}
