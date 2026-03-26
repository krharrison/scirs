//! SWAG — Stochastic Weight Averaging Gaussian (Maddox et al. 2019).
//!
//! SWAG approximates the posterior over neural network weights by collecting
//! a trajectory of SGD weight snapshots and fitting a Gaussian with
//! diagonal + low-rank covariance:
//!
//! ```text
//!   Σ ≈ diag(σ²_diag) / 2 + D̂ D̂ᵀ / (2(C−1))
//! ```
//!
//! where:
//! - `θ_SWA = (1/T) Σₜ θₜ` (SWA mean)
//! - `σ²_diag = max(θ²_bar − θ_SWA², 0)` (diagonal variance)
//! - `D̂` = matrix of last C mean-subtracted snapshots (low-rank factor)
//!
//! Sampling:
//! ```text
//!   θ ~ θ_SWA + 1/√2 · diag(σ_diag) z₁ + 1/√(2(C−1)) · D̂ z₂
//! ```
//! where `z₁ ~ N(0, I_d)` and `z₂ ~ N(0, I_C)`.

use crate::error::{StatsError, StatsResult};

use super::types::{BnnApproxResult, SwagConfig};

// ============================================================================
// LCG PRNG (no external rand crate)
// ============================================================================

/// 64-bit LCG for SWAG sampling.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
    }

    fn randn(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ============================================================================
// SwagState — frozen posterior statistics
// ============================================================================

/// Frozen SWAG posterior statistics built from `SwagCollector::finalize`.
#[derive(Debug, Clone)]
pub struct SwagState {
    /// SWA mean: θ_SWA = (1/T) Σₜ θₜ
    pub swa_mean: Vec<f64>,
    /// Second moment: θ²_bar = (1/T) Σₜ θₜ²
    pub sq_mean: Vec<f64>,
    /// Low-rank deviation columns: last C snapshots after subtracting θ_SWA.
    /// Each inner Vec has length `n_params`.
    pub deviation_cols: Vec<Vec<f64>>,
    /// Number of parameters
    pub n_params: usize,
    /// Number of snapshots used
    pub n_snapshots: usize,
}

impl SwagState {
    /// Diagonal variance: σ²_diag = max(θ²_bar − θ_SWA², 0).
    pub fn diagonal_variance(&self) -> Vec<f64> {
        self.swa_mean
            .iter()
            .zip(&self.sq_mean)
            .map(|(&m, &m2)| (m2 - m * m).max(0.0))
            .collect()
    }

    /// Number of deviation columns stored (rank of low-rank factor).
    pub fn rank(&self) -> usize {
        self.deviation_cols.len()
    }
}

// ============================================================================
// SwagCollector — online snapshot collector
// ============================================================================

/// Online collector that accumulates SGD weight snapshots and builds the
/// SWAG posterior statistics.
///
/// # Usage
/// ```rust
/// use scirs2_stats::bayesian_approx::swag::{SwagCollector, sample_weights};
///
/// let mut collector = SwagCollector::new(4, 5); // 4 params, rank-5
/// for t in 0..20 {
///     let weights = vec![t as f64 * 0.1; 4];
///     collector.update(&weights);
/// }
/// let state = collector.finalize().expect("finalize");
/// let samples = sample_weights(&state, 10, 42).expect("samples");
/// assert_eq!(samples.len(), 10);
/// ```
#[derive(Debug, Clone)]
pub struct SwagCollector {
    /// Running first moment (SWA mean), length = n_params
    mean: Vec<f64>,
    /// Running second moment (θ²_bar), length = n_params
    sq_mean: Vec<f64>,
    /// Buffer of raw weight snapshots (FIFO, max capacity = rank)
    snapshots: std::collections::VecDeque<Vec<f64>>,
    /// Maximum low-rank columns to keep
    rank: usize,
    /// Number of snapshots collected so far
    n_collected: usize,
    /// Number of parameters
    n_params: usize,
}

impl SwagCollector {
    /// Create a new collector for `n_params` parameters with low-rank factor of `rank`.
    ///
    /// # Panics
    /// Panics if `n_params == 0`.
    pub fn new(n_params: usize, rank: usize) -> Self {
        assert!(n_params > 0, "n_params must be > 0");
        Self {
            mean: vec![0.0; n_params],
            sq_mean: vec![0.0; n_params],
            snapshots: std::collections::VecDeque::new(),
            rank,
            n_collected: 0,
            n_params,
        }
    }

    /// Create a collector from a [`SwagConfig`].
    pub fn from_config(n_params: usize, config: &SwagConfig) -> Self {
        Self::new(n_params, config.c)
    }

    /// Update the running moments and snapshot buffer with a new weight vector.
    ///
    /// Online Welford-style update:
    /// ```text
    ///   mean_{t+1}    = (t · mean_t + θ) / (t+1)
    ///   sq_mean_{t+1} = (t · sq_mean_t + θ²) / (t+1)
    /// ```
    pub fn update(&mut self, weights: &[f64]) {
        debug_assert_eq!(
            weights.len(),
            self.n_params,
            "weight vector length must equal n_params"
        );

        let n = self.n_collected as f64;
        let n1 = n + 1.0;

        for i in 0..self.n_params {
            self.mean[i] = (n * self.mean[i] + weights[i]) / n1;
            self.sq_mean[i] = (n * self.sq_mean[i] + weights[i] * weights[i]) / n1;
        }

        // FIFO snapshot buffer for deviation columns
        if self.snapshots.len() >= self.rank && self.rank > 0 {
            self.snapshots.pop_front();
        }
        if self.rank > 0 {
            self.snapshots.push_back(weights.to_vec());
        }

        self.n_collected += 1;
    }

    /// Finalize the SWAG state.
    ///
    /// Computes deviation columns `D̂[:,k] = snapshot_k − θ_SWA`.
    ///
    /// # Errors
    /// Returns an error if fewer than 2 snapshots have been collected.
    pub fn finalize(&self) -> StatsResult<SwagState> {
        if self.n_collected < 2 {
            return Err(StatsError::invalid_argument(
                "SWAG requires at least 2 weight snapshots before finalize()",
            ));
        }

        // Compute deviation columns relative to the current SWA mean
        let deviation_cols: Vec<Vec<f64>> = self
            .snapshots
            .iter()
            .map(|snap| snap.iter().zip(&self.mean).map(|(&s, &m)| s - m).collect())
            .collect();

        Ok(SwagState {
            swa_mean: self.mean.clone(),
            sq_mean: self.sq_mean.clone(),
            deviation_cols,
            n_params: self.n_params,
            n_snapshots: self.n_collected,
        })
    }
}

// ============================================================================
// Sampling from the SWAG posterior
// ============================================================================

/// Draw `n` samples from the SWAG posterior.
///
/// The sampling formula is:
/// ```text
///   θ = θ_SWA
///       + (1/√2) · diag(σ_diag) · z₁       (diagonal contribution)
///       + (1/√(2(C−1))) · D̂ · z₂            (low-rank contribution)
/// ```
/// where `z₁ ~ N(0, I_d)`, `z₂ ~ N(0, I_C)`.
///
/// # Errors
/// Returns an error if `n_snapshots < 2`.
pub fn sample_weights(state: &SwagState, n: usize, seed: u64) -> StatsResult<Vec<Vec<f64>>> {
    if state.n_snapshots < 2 {
        return Err(StatsError::invalid_argument(
            "SWAG: need at least 2 snapshots to sample",
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut rng = Lcg::new(seed);
    let diag_var = state.diagonal_variance();
    let diag_std: Vec<f64> = diag_var.iter().map(|&v| v.sqrt()).collect();

    let c = state.deviation_cols.len();
    let c_scale = if c >= 2 {
        1.0 / (2.0 * (c - 1) as f64).sqrt()
    } else {
        0.0
    };
    let diag_scale = 1.0 / 2.0_f64.sqrt();

    let d = state.n_params;
    let mut samples = Vec::with_capacity(n);

    for _ in 0..n {
        // z₁ ~ N(0, I_d)
        let z1: Vec<f64> = (0..d).map(|_| rng.randn()).collect();

        // Diagonal contribution: diag_scale · diag_std · z₁
        let diag_part: Vec<f64> = (0..d).map(|i| diag_scale * diag_std[i] * z1[i]).collect();

        // Low-rank contribution: c_scale · D̂ z₂
        let mut lr_part = vec![0.0f64; d];
        if c >= 2 {
            let z2: Vec<f64> = (0..c).map(|_| rng.randn()).collect();
            for (k, col) in state.deviation_cols.iter().enumerate() {
                for i in 0..d {
                    lr_part[i] += col[i] * z2[k];
                }
            }
            for v in &mut lr_part {
                *v *= c_scale;
            }
        }

        // θ = θ_SWA + diag_part + lr_part
        let theta: Vec<f64> = (0..d)
            .map(|i| state.swa_mean[i] + diag_part[i] + lr_part[i])
            .collect();

        samples.push(theta);
    }

    Ok(samples)
}

// ============================================================================
// Ensemble prediction
// ============================================================================

/// Compute the ensemble mean and variance of `model_fn(x)` over `n_samples`
/// weight samples drawn from the SWAG posterior.
///
/// # Arguments
/// * `state` — Finalized SWAG state.
/// * `model_fn` — Closure `(weights, x) -> f64` evaluating the model.
/// * `x` — Input to the model.
/// * `n_samples` — Number of Monte Carlo weight samples.
/// * `seed` — RNG seed.
///
/// # Returns
/// `(mean, variance)` of the predictive distribution.
///
/// # Errors
/// Propagates sampling errors.
pub fn predict_ensemble(
    state: &SwagState,
    model_fn: &dyn Fn(&[f64], &[f64]) -> f64,
    x: &[f64],
    n_samples: usize,
    seed: u64,
) -> StatsResult<(f64, f64)> {
    if n_samples == 0 {
        return Err(StatsError::invalid_argument(
            "predict_ensemble: n_samples must be > 0",
        ));
    }

    let weight_samples = sample_weights(state, n_samples, seed)?;
    let preds: Vec<f64> = weight_samples.iter().map(|w| model_fn(w, x)).collect();

    let n = preds.len() as f64;
    let mean = preds.iter().sum::<f64>() / n;
    let variance = preds.iter().map(|&p| (p - mean) * (p - mean)).sum::<f64>() / n;

    Ok((mean, variance))
}

// ============================================================================
// High-level fit_swag convenience wrapper
// ============================================================================

/// Fit a SWAG posterior by collecting `weights_trajectory` snapshots.
///
/// # Arguments
/// * `weights_trajectory` — Slice of weight vectors collected during training.
/// * `config` — SWAG configuration.
///
/// # Errors
/// Returns an error if fewer than 2 snapshots are provided.
pub fn fit_swag(
    weights_trajectory: &[Vec<f64>],
    config: &SwagConfig,
) -> StatsResult<BnnApproxResult> {
    if weights_trajectory.len() < 2 {
        return Err(StatsError::invalid_argument(
            "fit_swag: need at least 2 weight snapshots",
        ));
    }

    let n_params = weights_trajectory[0].len();
    let mut collector = SwagCollector::from_config(n_params, config);
    for w in weights_trajectory {
        collector.update(w);
    }

    let state = collector.finalize()?;
    let uncertainty = state.diagonal_variance();

    Ok(BnnApproxResult {
        mean_weights: state.swa_mean,
        uncertainty,
        method: "SWAG".to_string(),
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_traj(n: usize, d: usize, scale: f64) -> Vec<Vec<f64>> {
        let mut lcg = Lcg::new(77);
        (0..n)
            .map(|_| (0..d).map(|_| lcg.randn() * scale).collect())
            .collect()
    }

    #[test]
    fn test_swag_mean_converges() {
        // SWA mean should converge toward the time-average of the trajectory
        let d = 4;
        let n = 100;
        // All weight vectors are the same constant → SWA mean = that constant
        let constant = vec![3.14f64; d];
        let traj: Vec<Vec<f64>> = (0..n).map(|_| constant.clone()).collect();

        let mut collector = SwagCollector::new(d, 20);
        for w in &traj {
            collector.update(w);
        }
        let state = collector.finalize().expect("finalize");
        for &m in &state.swa_mean {
            assert!((m - 3.14).abs() < 1e-10, "SWA mean should be 3.14, got {m}");
        }
    }

    #[test]
    fn test_swag_sq_mean_ge_mean_sq() {
        // Jensen's inequality: E[θ²] ≥ E[θ]²
        let traj = make_traj(50, 3, 1.0);
        let mut collector = SwagCollector::new(3, 10);
        for w in &traj {
            collector.update(w);
        }
        let state = collector.finalize().expect("finalize");
        for i in 0..3 {
            let mean_sq = state.swa_mean[i] * state.swa_mean[i];
            assert!(
                state.sq_mean[i] >= mean_sq - 1e-12,
                "E[θ²] ({}) < E[θ]² ({}) at dim {}",
                state.sq_mean[i],
                mean_sq,
                i
            );
        }
    }

    #[test]
    fn test_swag_diagonal_variance_nonneg() {
        let traj = make_traj(30, 5, 2.0);
        let mut collector = SwagCollector::new(5, 10);
        for w in &traj {
            collector.update(w);
        }
        let state = collector.finalize().expect("finalize");
        let var = state.diagonal_variance();
        for (i, &v) in var.iter().enumerate() {
            assert!(v >= 0.0, "Variance[{i}] = {v} < 0");
        }
    }

    #[test]
    fn test_swag_sample_shape() {
        let traj = make_traj(20, 4, 1.0);
        let mut collector = SwagCollector::new(4, 5);
        for w in &traj {
            collector.update(w);
        }
        let state = collector.finalize().expect("finalize");
        let samples = sample_weights(&state, 15, 42).expect("sample");
        assert_eq!(samples.len(), 15);
        for s in &samples {
            assert_eq!(s.len(), 4);
        }
    }

    #[test]
    fn test_swag_samples_reasonable_range() {
        // Samples should stay within ±5σ of the SWA mean for most dimensions.
        let traj = make_traj(50, 3, 0.5);
        let mut collector = SwagCollector::new(3, 10);
        for w in &traj {
            collector.update(w);
        }
        let state = collector.finalize().expect("finalize");
        let var = state.diagonal_variance();
        let samples = sample_weights(&state, 200, 1).expect("sample");

        let mut out_of_range = 0usize;
        for s in &samples {
            for i in 0..3 {
                let std = var[i].sqrt().max(1e-6) * 5.0;
                if (s[i] - state.swa_mean[i]).abs() > std + 5.0 * state.swa_mean[i].abs().max(1.0) {
                    out_of_range += 1;
                }
            }
        }
        // Allow a small fraction of samples to be "far" (due to low-rank component)
        assert!(
            out_of_range < 50,
            "Too many out-of-range samples: {out_of_range}"
        );
    }

    #[test]
    fn test_swag_collector_update_deviation_cols() {
        // After collecting rank+5 snapshots, the deviation_cols should have
        // exactly `rank` entries (FIFO).
        let rank = 3;
        let d = 2;
        let mut collector = SwagCollector::new(d, rank);
        for i in 0..10usize {
            collector.update(&[i as f64, (i * 2) as f64]);
        }
        let state = collector.finalize().expect("finalize");
        assert_eq!(state.deviation_cols.len(), rank);
        for col in &state.deviation_cols {
            assert_eq!(col.len(), d);
        }
    }

    #[test]
    fn test_swag_fewer_than_2_snapshots_error() {
        let mut collector = SwagCollector::new(2, 5);
        collector.update(&[1.0, 2.0]);
        assert!(collector.finalize().is_err());
    }

    #[test]
    fn test_swag_zero_samples() {
        let traj = make_traj(5, 2, 1.0);
        let mut collector = SwagCollector::new(2, 3);
        for w in &traj {
            collector.update(w);
        }
        let state = collector.finalize().expect("finalize");
        let samples = sample_weights(&state, 0, 1).expect("ok");
        assert!(samples.is_empty());
    }

    #[test]
    fn test_fit_swag_result() {
        let traj = make_traj(25, 3, 1.0);
        let config = SwagConfig::default();
        let result = fit_swag(&traj, &config).expect("fit");
        assert_eq!(result.mean_weights.len(), 3);
        assert_eq!(result.uncertainty.len(), 3);
        for &v in &result.uncertainty {
            assert!(v >= 0.0);
        }
        assert_eq!(result.method, "SWAG");
    }

    #[test]
    fn test_predict_ensemble() {
        // Simple linear model: f(w, x) = w[0] * x[0]
        let traj: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1]).collect();
        let config = SwagConfig {
            c: 5,
            ..SwagConfig::default()
        };
        let mut collector = SwagCollector::from_config(1, &config);
        for w in &traj {
            collector.update(w);
        }
        let state = collector.finalize().expect("finalize");

        let model_fn = |w: &[f64], x: &[f64]| w[0] * x[0];
        let x = vec![1.0];
        let (mean, var) = predict_ensemble(&state, &model_fn, &x, 100, 42).expect("predict");
        assert!(mean.is_finite(), "ensemble mean should be finite");
        assert!(var >= 0.0, "ensemble variance should be non-negative");
    }
}
