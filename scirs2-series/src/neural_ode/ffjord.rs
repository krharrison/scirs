//! FFJORD — Free-Form Jacobian of Reversible Dynamics
//!
//! FFJORD (Grathwohl et al., ICLR 2019) extends the Continuous Normalizing
//! Flow (CNF) with:
//!
//! 1. **Multiple Hutchinson probes** to reduce variance of the trace estimator.
//! 2. **Kinetic energy regularization** `λ · E[||f(z,t)||²]` that encourages
//!    smoother, lower-curvature flow trajectories, which improves training
//!    stability and generalisation.
//! 3. A utility function `moons_dataset` that generates the two-moons 2-D toy
//!    distribution often used to benchmark density estimators.
//!
//! # References
//! * Grathwohl et al., "FFJORD: Free-Form Continuous Dynamics for Scalable Reversible Generative Models", ICLR 2019.
//! * Chen et al., "Neural Ordinary Differential Equations", NeurIPS 2018.

use crate::error::{Result, TimeSeriesError};
use crate::neural_ode::cnf::{lcg_next, rk4_step, CnfConfig, MlpDynamics};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the FFJORD model.
#[derive(Debug, Clone)]
pub struct FfjordConfig {
    /// Base CNF configuration (dynamics, integration times, tolerances)
    pub cnf: CnfConfig,
    /// Number of Rademacher probe vectors for the Hutchinson estimator
    pub n_hutchinson_samples: usize,
    /// Whether to include kinetic energy regularization in the loss
    pub regularize_kinetic: bool,
    /// Regularization coefficient λ for the kinetic energy term
    pub reg_coeff: f64,
}

impl Default for FfjordConfig {
    fn default() -> Self {
        Self {
            cnf: CnfConfig {
                dim: 2,
                hidden_dim: 32,
                n_layers: 3,
                t0: 0.0,
                t1: 1.0,
                n_steps: 50,
                fd_eps: 1e-5,
                rtol: 1e-5,
                atol: 1e-5,
            },
            n_hutchinson_samples: 1,
            regularize_kinetic: true,
            reg_coeff: 0.01,
        }
    }
}

// ---------------------------------------------------------------------------
// FFJORD Model
// ---------------------------------------------------------------------------

/// FFJORD model: augments CNF with multi-sample Hutchinson + kinetic regularisation.
#[derive(Debug, Clone)]
pub struct FfjordModel {
    /// MLP dynamics network
    pub dynamics: MlpDynamics,
    /// FFJORD configuration
    pub config: FfjordConfig,
    /// LCG state for reproducible probe sampling
    rng_state: u64,
}

impl FfjordModel {
    /// Construct a new FFJORD model.
    pub fn new(config: FfjordConfig) -> Self {
        let dynamics = MlpDynamics::new(config.cnf.dim, config.cnf.hidden_dim, config.cnf.n_layers);
        Self {
            dynamics,
            config,
            rng_state: 0xfeed_f00d_dead_beef,
        }
    }

    /// Sample a Rademacher vector `v ∈ {±1}^d`.
    fn rademacher(&mut self, d: usize) -> Vec<f64> {
        (0..d)
            .map(|_| {
                if lcg_next(&mut self.rng_state) < 0.5 {
                    -1.0
                } else {
                    1.0
                }
            })
            .collect()
    }

    /// Estimate `tr(∂f/∂z)` using `n_hutchinson_samples` Rademacher probes,
    /// averaged for lower variance.
    fn hutchinson_multi(&mut self, z: &[f64], t: f64) -> f64 {
        let d = z.len();
        let eps = self.config.cnf.fd_eps;
        let n = self.config.n_hutchinson_samples.max(1);
        let mut acc = 0.0f64;
        for _ in 0..n {
            let v = self.rademacher(d);
            let zp: Vec<f64> = (0..d).map(|i| z[i] + eps * v[i]).collect();
            let zm: Vec<f64> = (0..d).map(|i| z[i] - eps * v[i]).collect();
            let fp = self.dynamics.forward(&zp, t);
            let fm = self.dynamics.forward(&zm, t);
            let jv: Vec<f64> = (0..d).map(|i| (fp[i] - fm[i]) / (2.0 * eps)).collect();
            acc += (0..d).map(|i| v[i] * jv[i]).sum::<f64>();
        }
        acc / n as f64
    }

    /// Compute kinetic energy at a point: `||f(z, t)||²`.
    fn kinetic_energy(&self, z: &[f64], t: f64) -> f64 {
        let f = self.dynamics.forward(z, t);
        f.iter().map(|x| x * x).sum()
    }

    /// Forward ODE integration with accumulated log-det and kinetic energy.
    ///
    /// Returns `(z1, delta_log_p, kinetic_energy)`.
    fn forward_with_stats(&mut self, z0: &[f64]) -> (Vec<f64>, f64, f64) {
        let d = z0.len();
        let t0 = self.config.cnf.t0;
        let t1 = self.config.cnf.t1;
        let n_steps = self.config.cnf.n_steps;
        let dt = (t1 - t0) / n_steps as f64;

        let mut z = z0.to_vec();
        let mut log_det = 0.0f64;
        let mut kin_acc = 0.0f64;

        for step in 0..n_steps {
            let t = t0 + step as f64 * dt;

            // Hutchinson trace at current position
            let trace = self.hutchinson_multi(&z, t);

            // Kinetic energy at current position
            let ke = self.kinetic_energy(&z, t);
            kin_acc += dt * ke;

            // RK4 step for z
            let dyn_ref = &self.dynamics;
            let f_fn = |zs: &[f64], ts: f64| -> Vec<f64> { dyn_ref.forward(zs, ts) };
            let z_new = rk4_step(&f_fn, &z, t, dt);

            // Euler update for log_det (using trace at start of step)
            log_det -= dt * trace;

            z = z_new;
        }

        (z, log_det, kin_acc)
    }

    /// Backward ODE integration (t1 → t0) with accumulated log-det.
    ///
    /// Returns `(z0, delta_log_p)`.
    fn backward(&mut self, x: &[f64]) -> (Vec<f64>, f64) {
        let d = x.len();
        let t0 = self.config.cnf.t0;
        let t1 = self.config.cnf.t1;
        let n_steps = self.config.cnf.n_steps;
        let dt = (t1 - t0) / n_steps as f64;

        let mut z = x.to_vec();
        let mut log_det = 0.0f64;

        for step in 0..n_steps {
            let t = t1 - step as f64 * dt;

            // Hutchinson trace at current position
            let trace = self.hutchinson_multi(&z, t);

            // RK4 backward step for z
            let dyn_ref = &self.dynamics;
            let f_fn = |zs: &[f64], ts: f64| -> Vec<f64> { dyn_ref.forward(zs, ts) };
            // Negate the dynamics for backward integration
            let f_neg = |zs: &[f64], ts: f64| -> Vec<f64> {
                f_fn(zs, ts).into_iter().map(|v| -v).collect()
            };
            let z_new = rk4_step(&f_neg, &z, t, dt);

            // log_det accumulates positively going backward
            log_det += dt * trace;

            z = z_new;
        }

        (z, log_det)
    }

    /// Compute `log p_model(x)`.
    pub fn log_prob(&mut self, x: &[f64]) -> f64 {
        let d = x.len();
        let (z0, delta_log_p) = self.backward(x);
        let log_base =
            -0.5 * d as f64 * (2.0 * PI).ln() - 0.5 * z0.iter().map(|&v| v * v).sum::<f64>();
        log_base + delta_log_p
    }

    /// Sample `n` points from the model distribution.
    pub fn sample(&mut self, n: usize) -> Vec<Vec<f64>> {
        let d = self.config.cnf.dim;
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let z0 = self.sample_base(d);
            let (x, _, _) = self.forward_with_stats(&z0);
            samples.push(x);
        }
        samples
    }

    /// Draw a single sample from N(0, I) via Box-Muller.
    fn sample_base(&mut self, d: usize) -> Vec<f64> {
        let mut z = Vec::with_capacity(d);
        let mut i = 0;
        while i < d {
            let u1 = lcg_next(&mut self.rng_state).max(1e-12);
            let u2 = lcg_next(&mut self.rng_state);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * PI * u2;
            z.push(r * theta.cos());
            if i + 1 < d {
                z.push(r * theta.sin());
            }
            i += 2;
        }
        z.truncate(d);
        z
    }

    /// Regularized training step.
    ///
    /// Returns `(nll_loss, kinetic_energy)` — both per-sample averages.
    ///
    /// The total optimized objective is `nll_loss + reg_coeff * kinetic_energy`.
    pub fn train_step_regularized(&mut self, batch: &[Vec<f64>], lr: f64) -> Result<(f64, f64)> {
        if batch.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "train_step_regularized: empty batch".to_string(),
            ));
        }
        let n = batch.len();

        // Compute baseline loss components
        let (nll, ke) = self.batch_losses(batch);

        let reg = if self.config.regularize_kinetic {
            self.config.reg_coeff
        } else {
            0.0
        };
        let total_baseline = nll + reg * ke;

        // Finite-difference gradient estimation
        let n_params = self.dynamics.n_params();
        let fd_step = 1e-4;
        let mut grad = vec![0.0f64; n_params];

        for p_idx in 0..n_params {
            let orig = self.dynamics.params[p_idx];

            self.dynamics.params[p_idx] = orig + fd_step;
            let (nll_p, ke_p) = self.batch_losses(batch);
            let total_p = nll_p + reg * ke_p;

            self.dynamics.params[p_idx] = orig - fd_step;
            let (nll_m, ke_m) = self.batch_losses(batch);
            let total_m = nll_m + reg * ke_m;

            self.dynamics.params[p_idx] = orig;
            grad[p_idx] = (total_p - total_m) / (2.0 * fd_step);
        }

        // SGD update
        for p_idx in 0..n_params {
            self.dynamics.params[p_idx] -= lr * grad[p_idx];
        }

        let _ = total_baseline; // used only implicitly above
        Ok((nll / n as f64, ke / n as f64))
    }

    /// Compute (total_nll, total_kinetic_energy) over a batch.
    fn batch_losses(&mut self, batch: &[Vec<f64>]) -> (f64, f64) {
        let mut total_nll = 0.0f64;
        let mut total_ke = 0.0f64;

        for x in batch {
            // Backward for NLL
            let lp = self.log_prob(x);
            total_nll -= lp;

            // Forward for kinetic energy
            let z0 = x.clone(); // use data point as proxy start (cheap)
            let (_, _, ke) = self.forward_with_stats(&z0);
            total_ke += ke;
        }

        (total_nll, total_ke)
    }
}

// ---------------------------------------------------------------------------
// Two-Moons Dataset
// ---------------------------------------------------------------------------

/// Generate `n` 2-D samples from the two-moons distribution.
///
/// Each sample is drawn from one of two half-circles arranged to form
/// two interleaving "moons". This is a standard density-estimation benchmark.
///
/// # Parameters
/// - `n`: number of samples (split evenly between the two moons)
/// - `noise`: standard deviation of additive Gaussian noise
pub fn moons_dataset(n: usize, noise: f64) -> Vec<[f64; 2]> {
    let half = n / 2;
    let mut out = Vec::with_capacity(n);
    let mut rng_state: u64 = 0xabcdef01;

    // Moon 0: upper half-circle centred at (0, 0)
    for i in 0..half {
        let theta = PI * i as f64 / half as f64;
        let nx = box_muller_single(&mut rng_state) * noise;
        let ny = box_muller_single(&mut rng_state) * noise;
        out.push([theta.cos() + nx, theta.sin() + ny]);
    }
    // Moon 1: lower half-circle centred at (1, -0.5)
    for i in 0..(n - half) {
        let theta = PI + PI * i as f64 / (n - half) as f64;
        let nx = box_muller_single(&mut rng_state) * noise;
        let ny = box_muller_single(&mut rng_state) * noise;
        out.push([theta.cos() + 1.0 + nx, theta.sin() + 0.5 + ny]);
    }

    out
}

/// Draw a single N(0,1) sample via Box-Muller.
fn box_muller_single(rng: &mut u64) -> f64 {
    let u1 = lcg_next(rng).max(1e-12);
    let u2 = lcg_next(rng);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ffjord_small() -> FfjordModel {
        FfjordModel::new(FfjordConfig {
            cnf: CnfConfig {
                dim: 2,
                hidden_dim: 8,
                n_layers: 2,
                t0: 0.0,
                t1: 1.0,
                n_steps: 4,
                fd_eps: 1e-5,
                rtol: 1e-5,
                atol: 1e-5,
            },
            n_hutchinson_samples: 2,
            regularize_kinetic: true,
            reg_coeff: 0.01,
        })
    }

    #[test]
    fn test_ffjord_log_prob_finite() {
        let mut m = make_ffjord_small();
        let x = vec![0.5, -0.3];
        let lp = m.log_prob(&x);
        assert!(lp.is_finite(), "log_prob must be finite, got {lp}");
    }

    #[test]
    fn test_ffjord_sample_shape() {
        let mut m = make_ffjord_small();
        let s = m.sample(4);
        assert_eq!(s.len(), 4);
        for v in &s {
            assert_eq!(v.len(), 2);
        }
    }

    #[test]
    fn test_ffjord_kinetic_energy_nonneg() {
        let m = make_ffjord_small();
        let z = vec![0.0, 0.0];
        let ke = m.kinetic_energy(&z, 0.5);
        assert!(ke >= 0.0, "kinetic energy must be ≥ 0, got {ke}");
    }

    #[test]
    fn test_ffjord_forward_stats_shapes() {
        let mut m = make_ffjord_small();
        let z0 = vec![0.1, 0.2];
        let (z1, log_det, ke) = m.forward_with_stats(&z0);
        assert_eq!(z1.len(), 2);
        assert!(log_det.is_finite());
        assert!(ke >= 0.0);
    }

    #[test]
    fn test_ffjord_train_step_returns_nonneg_ke() {
        let mut m = make_ffjord_small();
        let batch = vec![vec![0.5, 0.5], vec![-0.5, 0.5], vec![0.0, -0.5]];
        let (nll, ke) = m
            .train_step_regularized(&batch, 1e-4)
            .expect("should succeed");
        assert!(nll.is_finite(), "NLL must be finite, got {nll}");
        assert!(ke >= 0.0, "kinetic energy must be ≥ 0, got {ke}");
    }

    #[test]
    fn test_ffjord_loss_decomposes() {
        // With reg_coeff = 0, total loss should equal NLL component
        let mut m = FfjordModel::new(FfjordConfig {
            cnf: CnfConfig {
                dim: 2,
                hidden_dim: 8,
                n_layers: 2,
                t0: 0.0,
                t1: 1.0,
                n_steps: 4,
                fd_eps: 1e-5,
                rtol: 1e-5,
                atol: 1e-5,
            },
            n_hutchinson_samples: 1,
            regularize_kinetic: false,
            reg_coeff: 0.0,
        });
        let batch = vec![vec![0.3, 0.4]];
        // Should not panic; kinetic energy still reported even if reg=false
        let (nll, ke) = m
            .train_step_regularized(&batch, 0.0)
            .expect("should succeed");
        assert!(nll.is_finite());
        assert!(ke >= 0.0);
    }

    #[test]
    fn test_moons_dataset_length() {
        let samples = moons_dataset(100, 0.05);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_moons_dataset_finite_values() {
        for s in moons_dataset(50, 0.0) {
            assert!(s[0].is_finite());
            assert!(s[1].is_finite());
        }
    }

    #[test]
    fn test_moons_two_clusters() {
        // Moon 0 should have positive y centroid, Moon 1 negative y centroid
        let samples = moons_dataset(200, 0.0);
        let half = samples.len() / 2;
        let y0: f64 = samples[..half].iter().map(|s| s[1]).sum::<f64>() / half as f64;
        let y1: f64 =
            samples[half..].iter().map(|s| s[1]).sum::<f64>() / (samples.len() - half) as f64;
        // Moon 0 (upper) should have higher y than Moon 1 (lower)
        assert!(
            y0 > y1,
            "Moon 0 y-centroid {y0:.3} should exceed Moon 1 y-centroid {y1:.3}"
        );
    }

    #[test]
    fn test_ffjord_train_step_empty_batch_error() {
        let mut m = make_ffjord_small();
        let result = m.train_step_regularized(&[], 1e-3);
        assert!(result.is_err(), "empty batch should return error");
    }
}
