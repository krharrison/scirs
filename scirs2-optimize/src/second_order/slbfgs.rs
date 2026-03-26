//! Stochastic L-BFGS (S-L-BFGS) with SVRG-style variance reduction.
//!
//! Implements the algorithm of Moritz, Nishihara & Jordan (2016) combining:
//! - Mini-batch stochastic gradients
//! - L-BFGS curvature estimates from larger batches
//! - SVRG variance reduction to stabilize curvature pairs
//!
//! ## References
//!
//! - Moritz, P., Nishihara, R., Jordan, M.I. (2016).
//!   "A linearly-convergent stochastic L-BFGS algorithm." AISTATS.
//! - Johnson, R., Zhang, T. (2013).
//!   "Accelerating stochastic gradient descent using predictive variance reduction."
//!   NIPS 26.

use super::lbfgsb::hv_product;
use super::types::{OptResult, SlbfgsConfig};
use crate::error::OptimizeError;

// ─── LCG pseudo-random number generator ──────────────────────────────────────

/// Linear congruential generator for mini-batch selection.
///
/// Uses parameters from Numerical Recipes: a=1664525, c=1013904223, m=2^32.
pub struct Lcg {
    state: u64,
}

impl Lcg {
    /// Create a new LCG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance the state and return the next value in [0, 2^32).
    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223)
            & 0xFFFF_FFFF;
        self.state as u32
    }

    /// Return a random index in [0, n).
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }

    /// Fill `buf` with distinct random indices drawn from [0, n) without replacement.
    ///
    /// Uses a partial Fisher-Yates shuffle on a temporary index array.
    pub fn sample_without_replacement(&mut self, n: usize, k: usize, buf: &mut Vec<usize>) {
        buf.clear();
        if k == 0 || n == 0 {
            return;
        }
        let k = k.min(n);
        let mut pool: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + self.next_usize(n - i);
            pool.swap(i, j);
            buf.push(pool[i]);
        }
    }
}

// ─── SVRG variance reduction ─────────────────────────────────────────────────

/// Compute the SVRG-corrected stochastic gradient.
///
/// g_corrected = ∇f_{i}(x_k) − ∇f_{i}(x̃) + ∇F(x̃)
///
/// where x̃ is the snapshot point and ∇F(x̃) is the full (snapshot) gradient.
/// This correction gives an unbiased estimate of ∇F(x_k) with lower variance
/// than the raw stochastic gradient.
fn svrg_gradient(
    stoch_f_and_g: &dyn Fn(&[f64], &[usize]) -> (f64, Vec<f64>),
    x_k: &[f64],
    x_snap: &[f64],
    g_snap: &[f64], // ∇F(x̃) — full gradient at snapshot
    batch: &[usize],
) -> Vec<f64> {
    let (_, g_k) = stoch_f_and_g(x_k, batch);
    let (_, g_s) = stoch_f_and_g(x_snap, batch);
    g_k.iter()
        .zip(g_s.iter())
        .zip(g_snap.iter())
        .map(|((gki, gsi), gfi)| gki - gsi + gfi)
        .collect()
}

// ─── Curvature pair computation ───────────────────────────────────────────────

/// Compute a curvature pair (y_k) using a large mini-batch.
///
/// y_k = (1/|B|) Σ_{i∈B} (∇f_i(x_{k+1}) − ∇f_i(x_k))
///
/// Uses a larger batch than the gradient estimate to reduce noise in the
/// curvature estimate, which is crucial for quasi-Newton stability.
fn curvature_y(
    stoch_f_and_g: &dyn Fn(&[f64], &[usize]) -> (f64, Vec<f64>),
    x_new: &[f64],
    x_old: &[f64],
    batch: &[usize],
) -> Vec<f64> {
    let n = x_new.len();
    if batch.is_empty() {
        return vec![0.0; n];
    }
    let (_, g_new) = stoch_f_and_g(x_new, batch);
    let (_, g_old) = stoch_f_and_g(x_old, batch);
    g_new
        .iter()
        .zip(g_old.iter())
        .map(|(gn, go)| gn - go)
        .collect()
}

// ─── S-L-BFGS optimizer ───────────────────────────────────────────────────────

/// Stochastic L-BFGS optimizer with optional SVRG variance reduction.
pub struct SlbfgsOptimizer {
    /// Algorithm configuration.
    pub config: SlbfgsConfig,
}

impl SlbfgsOptimizer {
    /// Create with given configuration.
    pub fn new(config: SlbfgsConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self {
            config: SlbfgsConfig::default(),
        }
    }

    /// Minimize a stochastic objective function using S-L-BFGS.
    ///
    /// # Arguments
    /// * `stoch_f_and_g` — stochastic oracle: given `x` and a batch of indices
    ///   (subset of `0..n_samples`), returns (f, ∇f) on that mini-batch.
    /// * `full_grad_fn` — deterministic oracle for full gradient at snapshot:
    ///   `(x) → (f, ∇f)` over the full dataset.
    /// * `n_samples` — total number of data points.
    /// * `x0` — initial point.
    ///
    /// # Returns
    /// An `OptResult` describing the found minimizer and convergence status.
    pub fn minimize(
        &self,
        stoch_f_and_g: &dyn Fn(&[f64], &[usize]) -> (f64, Vec<f64>),
        full_grad_fn: &dyn Fn(&[f64]) -> (f64, Vec<f64>),
        n_samples: usize,
        x0: &[f64],
    ) -> Result<OptResult, OptimizeError> {
        let n = x0.len();
        let cfg = &self.config;
        let m = cfg.m;

        if n_samples == 0 {
            return Err(OptimizeError::ValueError(
                "n_samples must be positive".to_string(),
            ));
        }

        let mut x = x0.to_vec();
        let mut rng = Lcg::new(cfg.seed);

        // L-BFGS circular buffer
        let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut rho_hist: Vec<f64> = Vec::with_capacity(m);
        let mut gamma = 1.0_f64;

        // Snapshot for SVRG
        let mut x_snap = x.clone();
        let (mut f_snap, mut g_snap) = full_grad_fn(&x_snap);

        let mut n_iter = 0usize;
        let mut converged = false;
        let mut batch_buf: Vec<usize> = Vec::with_capacity(cfg.batch_size);
        let mut curv_batch_buf: Vec<usize> = Vec::with_capacity(cfg.curvature_batch_size);

        // Track best solution seen
        let mut best_x = x.clone();
        let mut best_f = f_snap;

        for iter in 0..cfg.max_iter {
            n_iter = iter;

            // Re-snapshot for SVRG
            if cfg.variance_reduction && iter % cfg.snapshot_freq == 0 {
                let (fs, gs) = full_grad_fn(&x);
                x_snap = x.clone();
                f_snap = fs;
                g_snap = gs;
            }

            // Convergence check (on snapshot gradient)
            let gn = g_snap.iter().map(|g| g * g).sum::<f64>().sqrt();
            if gn < cfg.tol {
                converged = true;
                break;
            }

            // Sample mini-batch for gradient
            rng.sample_without_replacement(n_samples, cfg.batch_size, &mut batch_buf);

            // Compute stochastic gradient (SVRG-corrected or plain)
            let g_k = if cfg.variance_reduction {
                svrg_gradient(stoch_f_and_g, &x, &x_snap, &g_snap, &batch_buf)
            } else {
                let (_, gk) = stoch_f_and_g(&x, &batch_buf);
                gk
            };

            // Compute L-BFGS search direction
            let hg = hv_product(&g_k, &s_hist, &y_hist, &rho_hist, gamma);
            let d: Vec<f64> = hg.iter().map(|v| -v).collect();

            // Check descent
            let slope: f64 = g_k.iter().zip(d.iter()).map(|(gi, di)| gi * di).sum();
            let d = if slope >= 0.0 {
                g_k.iter().map(|gi| -gi).collect::<Vec<f64>>()
            } else {
                d
            };

            // Step
            let x_new: Vec<f64> = x
                .iter()
                .zip(d.iter())
                .map(|(xi, di)| xi + cfg.lr * di)
                .collect();

            // Compute curvature pair with a larger batch
            rng.sample_without_replacement(
                n_samples,
                cfg.curvature_batch_size,
                &mut curv_batch_buf,
            );
            let s_k: Vec<f64> = (0..n).map(|i| x_new[i] - x[i]).collect();
            let y_k = curvature_y(stoch_f_and_g, &x_new, &x, &curv_batch_buf);

            // Curvature condition: y^T s > 0
            let sy: f64 = s_k.iter().zip(y_k.iter()).map(|(si, yi)| si * yi).sum();
            if sy > 1e-14 * s_k.iter().map(|si| si * si).sum::<f64>().sqrt() {
                if s_hist.len() == m {
                    s_hist.remove(0);
                    y_hist.remove(0);
                    rho_hist.remove(0);
                }
                let yy: f64 = y_k.iter().map(|yi| yi * yi).sum();
                if yy > 1e-14 {
                    gamma = sy / yy;
                }
                rho_hist.push(1.0 / sy);
                s_hist.push(s_k);
                y_hist.push(y_k);
            }

            x = x_new;

            // Update best
            let (f_curr, _) = full_grad_fn(&x);
            if f_curr < best_f {
                best_f = f_curr;
                best_x = x.clone();
            }
        }

        let (_, g_final) = full_grad_fn(&best_x);
        let grad_norm = g_final.iter().map(|gi| gi * gi).sum::<f64>().sqrt();

        Ok(OptResult {
            x: best_x,
            f_val: best_f,
            grad_norm,
            n_iter,
            converged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::second_order::types::SlbfgsConfig;

    /// Simple separable stochastic problem: f(x) = sum_i (x_i - 1)^2 / n
    /// Each sample i contributes (x_i - 1)^2.
    fn stoch_quad(x: &[f64], batch: &[usize]) -> (f64, Vec<f64>) {
        let n = x.len();
        if batch.is_empty() {
            let f: f64 = x.iter().map(|xi| (xi - 1.0).powi(2)).sum::<f64>() / n as f64;
            let g: Vec<f64> = x.iter().map(|xi| 2.0 * (xi - 1.0) / n as f64).collect();
            return (f, g);
        }
        let bs = batch.len() as f64;
        let f: f64 = batch.iter().map(|&i| (x[i % n] - 1.0).powi(2)).sum::<f64>() / bs;
        let mut g = vec![0.0_f64; n];
        for &idx in batch {
            g[idx % n] += 2.0 * (x[idx % n] - 1.0) / bs;
        }
        (f, g)
    }

    fn full_quad(x: &[f64]) -> (f64, Vec<f64>) {
        let n = x.len();
        let all: Vec<usize> = (0..n).collect();
        stoch_quad(x, &all)
    }

    #[test]
    fn test_slbfgs_gradient_variance_reduction() {
        // SVRG-corrected gradient should have zero mean at the optimum
        let x_star = vec![1.0; 4];
        let x_snap = vec![1.0; 4];
        let (_, g_snap) = full_quad(&x_snap);
        let batch = vec![0, 1, 2, 3];
        let g_corr = svrg_gradient(&stoch_quad, &x_star, &x_snap, &g_snap, &batch);
        for gi in &g_corr {
            assert!(
                gi.abs() < 1e-12,
                "Corrected gradient should be zero at optimum: got {}",
                gi
            );
        }
    }

    #[test]
    fn test_slbfgs_curvature_condition() {
        // y^T s > 0 must hold for a strongly convex quadratic
        let x_old = vec![2.0, 3.0];
        let x_new = vec![1.5, 2.5];
        let all_batch: Vec<usize> = (0..2).collect();
        let y = curvature_y(&stoch_quad, &x_new, &x_old, &all_batch);
        let s: Vec<f64> = x_new
            .iter()
            .zip(x_old.iter())
            .map(|(xn, xo)| xn - xo)
            .collect();
        let sy: f64 = s.iter().zip(y.iter()).map(|(si, yi)| si * yi).sum();
        assert!(sy > 0.0, "Curvature condition y^T s > 0 violated: {}", sy);
    }

    #[test]
    fn test_slbfgs_stochastic_convergence() {
        let mut cfg = SlbfgsConfig::default();
        cfg.max_iter = 300;
        cfg.lr = 0.05;
        cfg.batch_size = 4;
        cfg.curvature_batch_size = 8;
        cfg.variance_reduction = true;
        cfg.tol = 1e-4;
        let opt = SlbfgsOptimizer::new(cfg);

        let x0 = vec![0.0_f64; 4];
        let result = opt
            .minimize(&stoch_quad, &full_quad, 4, &x0)
            .expect("S-L-BFGS failed");
        for xi in &result.x {
            assert!(
                (xi - 1.0).abs() < 0.2,
                "S-L-BFGS did not converge: x={:?}",
                result.x
            );
        }
    }

    #[test]
    fn test_second_order_config_default() {
        use crate::second_order::types::{LbfgsBConfig, SlbfgsConfig, Sr1Config};
        let _c1 = LbfgsBConfig::default();
        let _c2 = Sr1Config::default();
        let _c3 = SlbfgsConfig::default();
        // All should construct without error
    }

    #[test]
    fn test_slbfgs_batch_selection() {
        let mut rng = Lcg::new(12345);
        let mut buf = Vec::new();
        rng.sample_without_replacement(100, 10, &mut buf);
        // Check: length = k
        assert_eq!(buf.len(), 10);
        // Check: no duplicates
        let mut sorted = buf.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 10, "Duplicate indices in batch selection");
        // Check: all within [0, 100)
        for &idx in &buf {
            assert!(idx < 100, "Index out of bounds: {}", idx);
        }
    }
}
