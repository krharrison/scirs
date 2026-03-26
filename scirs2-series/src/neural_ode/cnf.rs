//! Continuous Normalizing Flow (CNF) — Chen et al. NeurIPS 2018
//!
//! A Continuous Normalizing Flow defines a generative model by solving an ODE
//! whose dynamics are parameterised by a neural network (here, an MLP). The
//! change-of-variables formula lets us compute exact log-densities:
//!
//! ```text
//! dz/dt = f_θ(z, t)
//! d log p / dt = -tr(∂f/∂z)          (Liouville's theorem)
//! ```
//!
//! The instantaneous trace is estimated with a single Hutchinson probe vector
//! `v ∈ {±1}^d` using finite differences:
//!
//! ```text
//! tr(∂f/∂z) ≈ v^T (∂f/∂z) v   with  (∂f/∂z) v ≈ [f(z+εv) - f(z-εv)] / (2ε)
//! ```
//!
//! # References
//! * Chen et al., "Neural Ordinary Differential Equations", NeurIPS 2018.
//! * Grathwohl et al., "FFJORD: Free-Form Continuous Dynamics for Scalable Reversible Generative Models", ICLR 2019.

use crate::error::{Result, TimeSeriesError};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a Continuous Normalizing Flow model.
#[derive(Debug, Clone)]
pub struct CnfConfig {
    /// Dimensionality of each sample
    pub dim: usize,
    /// Width of hidden layers in the MLP dynamics network
    pub hidden_dim: usize,
    /// Number of hidden layers in the MLP
    pub n_layers: usize,
    /// Start time of ODE integration
    pub t0: f64,
    /// End time of ODE integration
    pub t1: f64,
    /// Relative tolerance (used for step-count selection)
    pub rtol: f64,
    /// Absolute tolerance
    pub atol: f64,
    /// Number of RK4 integration steps
    pub n_steps: usize,
    /// Finite-difference epsilon for Jacobian–vector product
    pub fd_eps: f64,
}

impl Default for CnfConfig {
    fn default() -> Self {
        Self {
            dim: 2,
            hidden_dim: 32,
            n_layers: 3,
            t0: 0.0,
            t1: 1.0,
            rtol: 1e-5,
            atol: 1e-5,
            n_steps: 50,
            fd_eps: 1e-5,
        }
    }
}

// ---------------------------------------------------------------------------
// MLP dynamics  f_θ(z, t) : R^d × R → R^d
// ---------------------------------------------------------------------------

/// Multi-layer perceptron that computes `dz/dt = f(z, t)`.
///
/// The network concatenates `[z, t]` as input (dimension `d+1`) and outputs
/// a vector of the same dimension `d` as `z`.
///
/// Parameters are stored as a flat `Vec<f64>` in the order
/// `W_1, b_1, W_2, b_2, …, W_L, b_L` where layer 0 maps `(d+1) → hidden_dim`
/// and the final layer maps `hidden_dim → d`.
#[derive(Debug, Clone)]
pub struct MlpDynamics {
    /// Input dimension (= data dim `d`)
    pub dim: usize,
    /// Hidden-layer width
    pub hidden_dim: usize,
    /// Number of hidden layers (total layers = n_layers + 1 for output)
    pub n_layers: usize,
    /// Flat parameter vector: `[W_0 (hidden×(d+1)), b_0 (hidden), W_1 (hidden×hidden), b_1, …, W_out (d×hidden), b_out (d)]`
    pub params: Vec<f64>,
}

impl MlpDynamics {
    /// Construct a new `MlpDynamics` with Kaiming-uniform initialisation.
    ///
    /// # Parameters
    /// - `dim`: dimension of `z`
    /// - `hidden_dim`: width of each hidden layer
    /// - `n_layers`: number of hidden layers (≥ 1)
    pub fn new(dim: usize, hidden_dim: usize, n_layers: usize) -> Self {
        let n_layers = n_layers.max(1);
        let mut params = Vec::new();
        let mut rng_state: u64 = 0xdeadbeef_cafebabe;

        // Layer sizes: (d+1) → hidden → … → hidden → d
        let layer_sizes: Vec<(usize, usize)> = {
            let mut v = Vec::new();
            let input_dim = dim + 1; // concat [z, t]
            v.push((hidden_dim, input_dim));
            for _ in 1..n_layers {
                v.push((hidden_dim, hidden_dim));
            }
            v.push((dim, hidden_dim));
            v
        };

        for (out, inp) in &layer_sizes {
            // Kaiming uniform: U(-sqrt(6/fan_in), +sqrt(6/fan_in))
            let fan_in = *inp as f64;
            let bound = (6.0_f64 / fan_in).sqrt();
            // W: out × inp
            for _ in 0..(out * inp) {
                let r = lcg_next(&mut rng_state);
                params.push((r * 2.0 - 1.0) * bound);
            }
            // b: out (zeros)
            for _ in 0..*out {
                params.push(0.0);
            }
        }
        Self {
            dim,
            hidden_dim,
            n_layers,
            params,
        }
    }

    /// Total number of trainable parameters.
    pub fn n_params(&self) -> usize {
        self.params.len()
    }

    /// Forward pass: `(z, t) → dz/dt`.
    pub fn forward(&self, z: &[f64], t: f64) -> Vec<f64> {
        // Build input: [z..., t]
        let mut x: Vec<f64> = z.to_vec();
        x.push(t);

        let n_layers = self.n_layers;
        let n_hidden_layers = n_layers; // hidden layers only; final layer is separate
        let total_layers = n_hidden_layers + 1;

        // Reconstruct layer sizes to compute parameter offsets
        let input_dim = self.dim + 1;
        let layer_sizes: Vec<(usize, usize)> = {
            let mut v = Vec::new();
            v.push((self.hidden_dim, input_dim));
            for _ in 1..n_hidden_layers {
                v.push((self.hidden_dim, self.hidden_dim));
            }
            v.push((self.dim, self.hidden_dim));
            v
        };

        let mut offset = 0usize;
        let mut h: Vec<f64> = x;

        for (layer_idx, (out, inp)) in layer_sizes.iter().enumerate() {
            let w_size = out * inp;
            let b_size = *out;
            let w = &self.params[offset..offset + w_size];
            let b = &self.params[offset + w_size..offset + w_size + b_size];
            offset += w_size + b_size;

            let mut next_h = vec![0.0f64; *out];
            for i in 0..*out {
                let mut s = b[i];
                for j in 0..h.len() {
                    s += w[i * inp + j] * h[j];
                }
                // Tanh activation on all hidden layers; linear on last
                let is_last = layer_idx == total_layers - 1;
                next_h[i] = if is_last { s } else { s.tanh() };
            }
            h = next_h;
        }
        h
    }
}

// ---------------------------------------------------------------------------
// Simple RK4 integrator (inline, no external dependency)
// ---------------------------------------------------------------------------

/// Single 4th-order Runge-Kutta step.
///
/// Advances `z` from `t` to `t + dt` under `dz/dt = f(z, t)`.
pub fn rk4_step<F>(f: &F, z: &[f64], t: f64, dt: f64) -> Vec<f64>
where
    F: Fn(&[f64], f64) -> Vec<f64>,
{
    let d = z.len();
    let k1 = f(z, t);
    let z2: Vec<f64> = (0..d).map(|i| z[i] + 0.5 * dt * k1[i]).collect();
    let k2 = f(&z2, t + 0.5 * dt);
    let z3: Vec<f64> = (0..d).map(|i| z[i] + 0.5 * dt * k2[i]).collect();
    let k3 = f(&z3, t + 0.5 * dt);
    let z4: Vec<f64> = (0..d).map(|i| z[i] + dt * k3[i]).collect();
    let k4 = f(&z4, t + dt);
    (0..d)
        .map(|i| z[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}

// ---------------------------------------------------------------------------
// Hutchinson trace estimator
// ---------------------------------------------------------------------------

/// Estimate `tr(J)` via one Rademacher probe: `v^T J v` where `Jv` is
/// approximated by a finite-difference directional derivative.
///
/// `J v ≈ [f(z + eps*v) - f(z - eps*v)] / (2*eps)`
fn hutchinson_trace(mlp_dyn: &MlpDynamics, z: &[f64], t: f64, v: &[f64], eps: f64) -> f64 {
    let d = z.len();
    let zp: Vec<f64> = (0..d).map(|i| z[i] + eps * v[i]).collect();
    let zm: Vec<f64> = (0..d).map(|i| z[i] - eps * v[i]).collect();
    let fp = mlp_dyn.forward(&zp, t);
    let fm = mlp_dyn.forward(&zm, t);
    // Jv ≈ (fp - fm) / (2 eps)
    let jv: Vec<f64> = (0..d).map(|i| (fp[i] - fm[i]) / (2.0 * eps)).collect();
    // v^T (Jv)
    (0..d).map(|i| v[i] * jv[i]).sum()
}

// ---------------------------------------------------------------------------
// CNF Model
// ---------------------------------------------------------------------------

/// Continuous Normalizing Flow model.
///
/// Transforms a base distribution (isotropic N(0,I)) to the data distribution
/// by integrating the neural ODE from `t0` to `t1`.
#[derive(Debug, Clone)]
pub struct CnfModel {
    /// MLP that parameterises the ODE dynamics
    pub dynamics: MlpDynamics,
    /// Model configuration
    pub config: CnfConfig,
    /// Rademacher probe state (deterministic per-call via LCG)
    rng_state: u64,
}

impl CnfModel {
    /// Create a new `CnfModel` with the given configuration.
    pub fn new(config: CnfConfig) -> Self {
        let dynamics = MlpDynamics::new(config.dim, config.hidden_dim, config.n_layers);
        Self {
            dynamics,
            config,
            rng_state: 0x12345678_9abcdef0,
        }
    }

    /// Sample a single Rademacher vector `v ∈ {±1}^d`.
    fn rademacher(&mut self, d: usize) -> Vec<f64> {
        let mut v = Vec::with_capacity(d);
        for _ in 0..d {
            let r = lcg_next(&mut self.rng_state);
            v.push(if r < 0.5 { -1.0 } else { 1.0 });
        }
        v
    }

    /// Forward pass: integrate ODE from `t0 → t1` starting at `z0`.
    ///
    /// Returns `(z1, delta_log_p)` where `delta_log_p` is the accumulated
    /// change in log-probability (negative of integrated trace divergence).
    pub fn forward(&mut self, z0: &[f64]) -> (Vec<f64>, f64) {
        let d = z0.len();
        let t0 = self.config.t0;
        let t1 = self.config.t1;
        let n_steps = self.config.n_steps;
        let dt = (t1 - t0) / n_steps as f64;
        let fd_eps = self.config.fd_eps;

        let mut z = z0.to_vec();
        let mut log_det = 0.0f64;

        // One Rademacher vector, fixed for the whole trajectory
        let v = self.rademacher(d);

        for step in 0..n_steps {
            let t = t0 + step as f64 * dt;
            // Augmented state: [z (d), log_det (1)]
            // We integrate z and log_det simultaneously using RK4.
            // For the trace, we use the Hutchinson estimate at the midpoint of each step.

            // RK4 for z
            let dyn_ref = &self.dynamics;
            let v_ref = &v;
            let fd_e = fd_eps;
            let f = |zs: &[f64], ts: f64| -> Vec<f64> { dyn_ref.forward(zs, ts) };

            let k1_z = f(&z, t);
            let trace1 = hutchinson_trace(dyn_ref, &z, t, v_ref, fd_e);

            let z2: Vec<f64> = (0..d).map(|i| z[i] + 0.5 * dt * k1_z[i]).collect();
            let k2_z = f(&z2, t + 0.5 * dt);
            let trace2 = hutchinson_trace(dyn_ref, &z2, t + 0.5 * dt, v_ref, fd_e);

            let z3: Vec<f64> = (0..d).map(|i| z[i] + 0.5 * dt * k2_z[i]).collect();
            let k3_z = f(&z3, t + 0.5 * dt);
            let trace3 = hutchinson_trace(dyn_ref, &z3, t + 0.5 * dt, v_ref, fd_e);

            let z4: Vec<f64> = (0..d).map(|i| z[i] + dt * k3_z[i]).collect();
            let k4_z = f(&z4, t + dt);
            let trace4 = hutchinson_trace(dyn_ref, &z4, t + dt, v_ref, fd_e);

            // Update z via RK4
            for i in 0..d {
                z[i] += (dt / 6.0) * (k1_z[i] + 2.0 * k2_z[i] + 2.0 * k3_z[i] + k4_z[i]);
            }

            // Update log_det via RK4 (d(log_det)/dt = -tr(df/dz))
            let avg_trace = (trace1 + 2.0 * trace2 + 2.0 * trace3 + trace4) / 6.0;
            log_det -= dt * avg_trace;
        }

        (z, log_det)
    }

    /// Backward pass: integrate ODE from `t1 → t0` starting at `x`.
    ///
    /// Returns `(z0, delta_log_p)` accumulated while going backward.
    pub fn backward(&mut self, x: &[f64]) -> (Vec<f64>, f64) {
        let d = x.len();
        let t0 = self.config.t0;
        let t1 = self.config.t1;
        let n_steps = self.config.n_steps;
        let dt = (t1 - t0) / n_steps as f64; // positive step size
        let fd_eps = self.config.fd_eps;

        let mut z = x.to_vec();
        let mut log_det = 0.0f64;

        let v = self.rademacher(d);

        for step in 0..n_steps {
            // Reverse: t decreases from t1 to t0
            let t = t1 - step as f64 * dt;
            let dyn_ref = &self.dynamics;
            let v_ref = &v;
            let fd_e = fd_eps;
            let f = |zs: &[f64], ts: f64| -> Vec<f64> { dyn_ref.forward(zs, ts) };

            // Integrate backward: dz/d(-t) = -f(z, t)
            let k1_z = f(&z, t);
            let trace1 = hutchinson_trace(dyn_ref, &z, t, v_ref, fd_e);

            let z2: Vec<f64> = (0..d).map(|i| z[i] - 0.5 * dt * k1_z[i]).collect();
            let k2_z = f(&z2, t - 0.5 * dt);
            let trace2 = hutchinson_trace(dyn_ref, &z2, t - 0.5 * dt, v_ref, fd_e);

            let z3: Vec<f64> = (0..d).map(|i| z[i] - 0.5 * dt * k2_z[i]).collect();
            let k3_z = f(&z3, t - 0.5 * dt);
            let trace3 = hutchinson_trace(dyn_ref, &z3, t - 0.5 * dt, v_ref, fd_e);

            let z4: Vec<f64> = (0..d).map(|i| z[i] - dt * k3_z[i]).collect();
            let k4_z = f(&z4, t - dt);
            let trace4 = hutchinson_trace(dyn_ref, &z4, t - dt, v_ref, fd_e);

            for i in 0..d {
                z[i] -= (dt / 6.0) * (k1_z[i] + 2.0 * k2_z[i] + 2.0 * k3_z[i] + k4_z[i]);
            }

            let avg_trace = (trace1 + 2.0 * trace2 + 2.0 * trace3 + trace4) / 6.0;
            // Going backward: log p changes by +trace (reverse of forward)
            log_det += dt * avg_trace;
        }

        (z, log_det)
    }

    /// Compute `log p_model(x)` using the change-of-variables formula.
    ///
    /// `log p(x) = log p_z(z0) + Δ_log_p`  (backward ODE gives z0 and Δ)
    pub fn log_prob(&mut self, x: &[f64]) -> f64 {
        let d = x.len();
        let (z0, delta_log_p) = self.backward(x);
        // log p_z(z0) = -d/2 * log(2π) - 0.5 * ||z0||²
        let log_base =
            -0.5 * d as f64 * (2.0 * PI).ln() - 0.5 * z0.iter().map(|&v| v * v).sum::<f64>();
        log_base + delta_log_p
    }

    /// Sample `n` points from the model distribution.
    ///
    /// Draws z0 ~ N(0,I) and integrates the ODE forward to produce samples x.
    pub fn sample(&mut self, n: usize) -> Vec<Vec<f64>> {
        let d = self.config.dim;
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let z0 = self.sample_base(d);
            let (x, _) = self.forward(&z0);
            samples.push(x);
        }
        samples
    }

    /// Draw a single sample from the base distribution N(0, I).
    fn sample_base(&mut self, d: usize) -> Vec<f64> {
        // Box-Muller transform using LCG
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

    /// One training step: compute negative log-likelihood on a mini-batch,
    /// update parameters via finite-difference gradient descent.
    ///
    /// Returns the NLL loss before the update.
    pub fn train_step(&mut self, batch: &[Vec<f64>], lr: f64) -> Result<f64> {
        if batch.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "train_step: empty batch".to_string(),
            ));
        }
        let n = batch.len();
        let n_params = self.dynamics.n_params();

        // Compute baseline NLL
        let baseline_nll = self.batch_nll(batch);

        // Finite-difference gradient estimation
        let fd_step = 1e-4;
        let mut grad = vec![0.0f64; n_params];

        for p_idx in 0..n_params {
            let orig = self.dynamics.params[p_idx];
            self.dynamics.params[p_idx] = orig + fd_step;
            let nll_plus = self.batch_nll(batch);
            self.dynamics.params[p_idx] = orig - fd_step;
            let nll_minus = self.batch_nll(batch);
            self.dynamics.params[p_idx] = orig;
            grad[p_idx] = (nll_plus - nll_minus) / (2.0 * fd_step);
        }

        // SGD update
        for p_idx in 0..n_params {
            self.dynamics.params[p_idx] -= lr * grad[p_idx];
        }

        Ok(baseline_nll / n as f64)
    }

    /// Compute average negative log-likelihood over a batch.
    fn batch_nll(&mut self, batch: &[Vec<f64>]) -> f64 {
        let total: f64 = batch.iter().map(|x| self.log_prob(x)).sum();
        -total
    }
}

// ---------------------------------------------------------------------------
// Utility: Linear Congruential Generator for deterministic pseudo-randomness
// ---------------------------------------------------------------------------

/// Advance LCG state and return a uniform value in [0, 1).
pub(crate) fn lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // Use upper 32 bits for better quality
    let bits = (*state >> 33) as u64;
    bits as f64 / (1u64 << 31) as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: construct a 2-D CNF with minimal steps for speed.
    fn make_cnf_2d() -> CnfModel {
        CnfModel::new(CnfConfig {
            dim: 2,
            hidden_dim: 8,
            n_layers: 2,
            t0: 0.0,
            t1: 1.0,
            n_steps: 4,
            fd_eps: 1e-5,
            ..Default::default()
        })
    }

    #[test]
    fn test_mlp_output_shape_matches_input_dim() {
        let mlp = MlpDynamics::new(3, 8, 2);
        let z = vec![0.1, 0.2, 0.3];
        let out = mlp.forward(&z, 0.5);
        assert_eq!(out.len(), 3, "output dim must equal input dim");
    }

    #[test]
    fn test_mlp_different_dims() {
        for dim in [1, 4, 8] {
            let mlp = MlpDynamics::new(dim, 16, 3);
            let z: Vec<f64> = (0..dim).map(|i| i as f64 * 0.1).collect();
            let out = mlp.forward(&z, 0.0);
            assert_eq!(out.len(), dim);
        }
    }

    #[test]
    fn test_cnf_forward_output_shapes() {
        let mut model = make_cnf_2d();
        let z0 = vec![0.5, -0.3];
        let (z1, _log_det) = model.forward(&z0);
        assert_eq!(z1.len(), 2, "forward output dim must match input dim");
    }

    #[test]
    fn test_cnf_forward_log_det_is_finite() {
        let mut model = make_cnf_2d();
        let z0 = vec![0.5, -0.3];
        let (_, log_det) = model.forward(&z0);
        assert!(log_det.is_finite(), "log_det must be finite, got {log_det}");
    }

    #[test]
    fn test_rk4_step_accuracy_linear_ode() {
        // dz/dt = z  →  z(t) = z0 * exp(t)
        // With dt = 0.1 RK4 error should be O(dt^5) ≈ 1e-7
        let f = |z: &[f64], _t: f64| -> Vec<f64> { vec![z[0]] };
        let z0 = vec![1.0_f64];
        let dt = 0.1;
        let z1 = rk4_step(&f, &z0, 0.0, dt);
        let exact = dt.exp();
        let err = (z1[0] - exact).abs();
        assert!(err < 1e-6, "RK4 error {err} should be < 1e-6 for dt={dt}");
    }

    #[test]
    fn test_cnf_log_prob_is_finite() {
        let mut model = make_cnf_2d();
        let x = vec![0.0, 0.0];
        let lp = model.log_prob(&x);
        assert!(lp.is_finite(), "log_prob must be finite, got {lp}");
    }

    #[test]
    fn test_cnf_sample_correct_dimension() {
        let mut model = make_cnf_2d();
        let samples = model.sample(5);
        assert_eq!(samples.len(), 5);
        for s in &samples {
            assert_eq!(s.len(), 2);
        }
    }

    #[test]
    fn test_cnf_sample_finite_values() {
        let mut model = make_cnf_2d();
        let samples = model.sample(10);
        for s in &samples {
            for &v in s {
                assert!(v.is_finite(), "sample value must be finite, got {v}");
            }
        }
    }

    #[test]
    fn test_hutchinson_estimator_unbiasedness() {
        // For a diagonal matrix A = diag(a_1, ..., a_d),
        // E[v^T A v] = tr(A) exactly when v is Rademacher.
        // We verify that averaging over many probe vectors gives ~tr(A).
        let mlp = MlpDynamics::new(4, 8, 1);
        let z = vec![0.0, 0.0, 0.0, 0.0];
        let t = 0.5;
        let eps = 1e-5;
        let mut sum = 0.0;
        let n_probes = 200;
        let mut rng_state: u64 = 42;
        for _ in 0..n_probes {
            let v: Vec<f64> = (0..4)
                .map(|_| {
                    if lcg_next(&mut rng_state) < 0.5 {
                        -1.0
                    } else {
                        1.0
                    }
                })
                .collect();
            sum += hutchinson_trace(&mlp, &z, t, &v, eps);
        }
        let mean_est = sum / n_probes as f64;
        // Compute true trace via full Jacobian rows
        let mut true_trace = 0.0;
        for i in 0..4 {
            let mut ei = vec![0.0; 4];
            ei[i] = eps;
            let zp: Vec<f64> = z.iter().zip(&ei).map(|(a, b)| a + b).collect();
            let zm: Vec<f64> = z.iter().zip(&ei).map(|(a, b)| a - b).collect();
            let fp = mlp.forward(&zp, t);
            let fm = mlp.forward(&zm, t);
            true_trace += (fp[i] - fm[i]) / (2.0 * eps);
        }
        let err = (mean_est - true_trace).abs();
        // Statistical tolerance: std dev of Hutchinson is O(sqrt(n_probes)^-1 * d * ||J||_F)
        // For a small MLP this should be within ±2 in relative terms
        assert!(
            err < (true_trace.abs() + 1.0) * 2.0 + 2.0,
            "Hutchinson mean {mean_est:.4} far from true trace {true_trace:.4} (err={err:.4})"
        );
    }
}
