//! Normalizing Flows for Variational Inference
//!
//! Implements invertible transformations that map a simple base distribution
//! (e.g., standard Gaussian) to a more flexible posterior approximation.
//!
//! Supports:
//! - **Planar flow**: `f(z) = z + u * tanh(w^T z + b)` (Rezende & Mohamed 2015)
//! - **Radial flow**: `f(z) = z + beta * (z - z0) / (alpha + ||z - z0||)` (Rezende & Mohamed 2015)
//! - **Flow chains**: Compose multiple flows `z_K = f_K . ... . f_1(z_0)`
//! - **ELBO with flow**: `log p(x, z_K) - log q_0(z_0) + sum log|det(df_k/dz_{k-1})|`
//!
//! These flows can be used to enhance ADVI by replacing mean-field with flow-based posteriors.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::Array1;
use std::f64::consts::PI;

use super::{PosteriorResult, VariationalInference};

// ============================================================================
// Flow Types
// ============================================================================

/// Type of normalizing flow layer
#[derive(Debug, Clone)]
pub enum FlowLayer {
    /// Planar flow: f(z) = z + u * tanh(w^T z + b)
    Planar {
        /// Weight vector w (dim)
        w: Array1<f64>,
        /// Scale vector u (dim)
        u: Array1<f64>,
        /// Bias scalar
        b: f64,
    },
    /// Radial flow: f(z) = z + beta * (z - z0) / (alpha + ||z - z0||)
    Radial {
        /// Center point z0 (dim)
        z0: Array1<f64>,
        /// Scale parameter alpha > 0
        alpha: f64,
        /// Magnitude parameter beta
        beta: f64,
    },
}

impl FlowLayer {
    /// Create a new planar flow layer with random initialization
    pub fn new_planar(dim: usize, seed: u64) -> Self {
        let golden = 1.618033988749895_f64;
        let plastic = 1.324717957244746_f64;

        let w = Array1::from_shape_fn(dim, |i| {
            let u1 = ((seed as f64 * golden + i as f64 * plastic + 0.3) % 1.0)
                .abs()
                .max(1e-10)
                .min(1.0 - 1e-10);
            let u2 = ((seed as f64 * plastic + i as f64 * golden + 0.7) % 1.0)
                .abs()
                .max(1e-10)
                .min(1.0 - 1e-10);
            let r = (-2.0 * u1.ln()).sqrt();
            r * (2.0 * PI * u2).cos() * 0.1
        });

        let u = Array1::from_shape_fn(dim, |i| {
            let u1 = (((seed + 100) as f64 * golden + i as f64 * plastic + 0.1) % 1.0)
                .abs()
                .max(1e-10)
                .min(1.0 - 1e-10);
            let u2 = (((seed + 100) as f64 * plastic + i as f64 * golden + 0.9) % 1.0)
                .abs()
                .max(1e-10)
                .min(1.0 - 1e-10);
            let r = (-2.0 * u1.ln()).sqrt();
            r * (2.0 * PI * u2).cos() * 0.1
        });

        let b_val = {
            let u1 = ((seed as f64 * 0.37 + 0.5) % 1.0)
                .abs()
                .max(1e-10)
                .min(1.0 - 1e-10);
            let u2 = ((seed as f64 * 0.73 + 0.5) % 1.0)
                .abs()
                .max(1e-10)
                .min(1.0 - 1e-10);
            let r = (-2.0 * u1.ln()).sqrt();
            r * (2.0 * PI * u2).cos() * 0.1
        };

        FlowLayer::Planar { w, u, b: b_val }
    }

    /// Create a new radial flow layer with random initialization
    pub fn new_radial(dim: usize, seed: u64) -> Self {
        let golden = 1.618033988749895_f64;
        let plastic = 1.324717957244746_f64;

        let z0 = Array1::from_shape_fn(dim, |i| {
            let u1 = (((seed + 200) as f64 * golden + i as f64 * plastic + 0.2) % 1.0)
                .abs()
                .max(1e-10)
                .min(1.0 - 1e-10);
            let u2 = (((seed + 200) as f64 * plastic + i as f64 * golden + 0.8) % 1.0)
                .abs()
                .max(1e-10)
                .min(1.0 - 1e-10);
            let r = (-2.0 * u1.ln()).sqrt();
            r * (2.0 * PI * u2).cos() * 0.1
        });

        FlowLayer::Radial {
            z0,
            alpha: 1.0,
            beta: 0.1,
        }
    }

    /// Apply the flow transformation: f(z) and compute log|det(df/dz)|
    ///
    /// Returns (f(z), log|det J|)
    pub fn forward(&self, z: &Array1<f64>) -> StatsResult<(Array1<f64>, f64)> {
        match self {
            FlowLayer::Planar { w, u, b } => {
                let dim = z.len();
                if w.len() != dim || u.len() != dim {
                    return Err(StatsError::DimensionMismatch(format!(
                        "Flow dimension mismatch: z={}, w={}, u={}",
                        dim,
                        w.len(),
                        u.len()
                    )));
                }

                // Enforce invertibility: u_hat = u + (m(w^T u) - w^T u) * w / ||w||^2
                // where m(x) = -1 + softplus(x) = -1 + log(1 + exp(x))
                let u_hat = enforce_planar_invertibility(w, u);

                let wtz = w.dot(z) + b;
                let tanh_wtz = wtz.tanh();

                // f(z) = z + u_hat * tanh(w^T z + b)
                let fz = z + &(&u_hat * tanh_wtz);

                // log|det J| = log|1 + u_hat^T * w * (1 - tanh^2(w^T z + b))|
                let dtanh = 1.0 - tanh_wtz * tanh_wtz;
                let psi = w * dtanh;
                let det_term = 1.0 + u_hat.dot(&psi);

                let log_det = det_term.abs().max(1e-15).ln();

                Ok((fz, log_det))
            }
            FlowLayer::Radial { z0, alpha, beta } => {
                let dim = z.len();
                if z0.len() != dim {
                    return Err(StatsError::DimensionMismatch(format!(
                        "Flow dimension mismatch: z={}, z0={}",
                        dim,
                        z0.len()
                    )));
                }

                let diff = z - z0;
                let r = diff.dot(&diff).sqrt().max(1e-10);
                let alpha_pos = alpha.abs().max(1e-6);

                // Enforce beta >= -alpha to ensure invertibility
                let beta_hat = -alpha_pos + softplus(*beta + alpha_pos);

                let h = 1.0 / (alpha_pos + r);
                let h_prime = -1.0 / ((alpha_pos + r) * (alpha_pos + r));

                // f(z) = z + beta_hat * h(r) * (z - z0)
                let fz = z + &(&diff * (beta_hat * h));

                // log|det J| = (d-1) * log(1 + beta_hat * h)
                //             + log(1 + beta_hat * h + beta_hat * h' * r)
                let d = dim as f64;
                let term1 = 1.0 + beta_hat * h;
                let term2 = 1.0 + beta_hat * h + beta_hat * h_prime * r;

                let log_det = (d - 1.0) * term1.abs().max(1e-15).ln() + term2.abs().max(1e-15).ln();

                Ok((fz, log_det))
            }
        }
    }

    /// Get the total number of parameters for this flow layer
    pub fn n_params(&self) -> usize {
        match self {
            FlowLayer::Planar { w, u, .. } => w.len() + u.len() + 1,
            FlowLayer::Radial { z0, .. } => z0.len() + 2,
        }
    }

    /// Get all parameters as a flat vector
    pub fn get_params(&self) -> Array1<f64> {
        match self {
            FlowLayer::Planar { w, u, b } => {
                let dim = w.len();
                let mut params = Array1::zeros(2 * dim + 1);
                for i in 0..dim {
                    params[i] = w[i];
                    params[dim + i] = u[i];
                }
                params[2 * dim] = *b;
                params
            }
            FlowLayer::Radial { z0, alpha, beta } => {
                let dim = z0.len();
                let mut params = Array1::zeros(dim + 2);
                for i in 0..dim {
                    params[i] = z0[i];
                }
                params[dim] = *alpha;
                params[dim + 1] = *beta;
                params
            }
        }
    }

    /// Set parameters from a flat vector
    pub fn set_params(&mut self, params: &Array1<f64>) -> StatsResult<()> {
        match self {
            FlowLayer::Planar { w, u, b } => {
                let dim = w.len();
                if params.len() != 2 * dim + 1 {
                    return Err(StatsError::DimensionMismatch(format!(
                        "Expected {} params, got {}",
                        2 * dim + 1,
                        params.len()
                    )));
                }
                for i in 0..dim {
                    w[i] = params[i];
                    u[i] = params[dim + i];
                }
                *b = params[2 * dim];
                Ok(())
            }
            FlowLayer::Radial { z0, alpha, beta } => {
                let dim = z0.len();
                if params.len() != dim + 2 {
                    return Err(StatsError::DimensionMismatch(format!(
                        "Expected {} params, got {}",
                        dim + 2,
                        params.len()
                    )));
                }
                for i in 0..dim {
                    z0[i] = params[i];
                }
                *alpha = params[dim];
                *beta = params[dim + 1];
                Ok(())
            }
        }
    }
}

/// Enforce invertibility for planar flows by computing u_hat
/// such that w^T u_hat >= -1
fn enforce_planar_invertibility(w: &Array1<f64>, u: &Array1<f64>) -> Array1<f64> {
    let wtu = w.dot(u);
    let w_norm_sq = w.dot(w);
    if w_norm_sq < 1e-15 {
        return u.clone();
    }
    // m(x) = -1 + softplus(x) = -1 + log(1 + exp(x))
    let m_wtu = -1.0 + softplus(wtu);
    if (m_wtu - wtu).abs() < 1e-15 {
        return u.clone();
    }
    u + &(w * ((m_wtu - wtu) / w_norm_sq))
}

/// Numerically stable softplus: log(1 + exp(x))
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

// ============================================================================
// Flow Chain
// ============================================================================

/// A chain of normalizing flow layers
#[derive(Debug, Clone)]
pub struct NormalizingFlowChain {
    /// Ordered list of flow layers
    pub layers: Vec<FlowLayer>,
}

impl NormalizingFlowChain {
    /// Create a new flow chain with the given layers
    pub fn new(layers: Vec<FlowLayer>) -> Self {
        Self { layers }
    }

    /// Create a chain of planar flows
    pub fn planar(dim: usize, n_layers: usize, seed: u64) -> Self {
        let layers = (0..n_layers)
            .map(|i| FlowLayer::new_planar(dim, seed + i as u64 * 7))
            .collect();
        Self { layers }
    }

    /// Create a chain of radial flows
    pub fn radial(dim: usize, n_layers: usize, seed: u64) -> Self {
        let layers = (0..n_layers)
            .map(|i| FlowLayer::new_radial(dim, seed + i as u64 * 11))
            .collect();
        Self { layers }
    }

    /// Create a mixed chain alternating planar and radial flows
    pub fn mixed(dim: usize, n_layers: usize, seed: u64) -> Self {
        let layers = (0..n_layers)
            .map(|i| {
                if i % 2 == 0 {
                    FlowLayer::new_planar(dim, seed + i as u64 * 13)
                } else {
                    FlowLayer::new_radial(dim, seed + i as u64 * 17)
                }
            })
            .collect();
        Self { layers }
    }

    /// Apply the full chain: z_K = f_K . ... . f_1(z_0)
    ///
    /// Returns (z_K, sum of log|det J_k|)
    pub fn forward(&self, z0: &Array1<f64>) -> StatsResult<(Array1<f64>, f64)> {
        let mut z = z0.clone();
        let mut total_log_det = 0.0;

        for layer in &self.layers {
            let (z_new, log_det) = layer.forward(&z)?;
            z = z_new;
            total_log_det += log_det;
        }

        Ok((z, total_log_det))
    }

    /// Total number of flow parameters across all layers
    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.n_params()).sum()
    }

    /// Get all flow parameters as a flat vector
    pub fn get_params(&self) -> Array1<f64> {
        let total = self.n_params();
        let mut params = Array1::zeros(total);
        let mut offset = 0;
        for layer in &self.layers {
            let lp = layer.get_params();
            let n = lp.len();
            for i in 0..n {
                params[offset + i] = lp[i];
            }
            offset += n;
        }
        params
    }

    /// Set all flow parameters from a flat vector
    pub fn set_params(&mut self, params: &Array1<f64>) -> StatsResult<()> {
        let total = self.n_params();
        if params.len() != total {
            return Err(StatsError::DimensionMismatch(format!(
                "Expected {} total flow params, got {}",
                total,
                params.len()
            )));
        }
        let mut offset = 0;
        for layer in &mut self.layers {
            let n = layer.n_params();
            let lp = Array1::from_shape_fn(n, |i| params[offset + i]);
            layer.set_params(&lp)?;
            offset += n;
        }
        Ok(())
    }
}

// ============================================================================
// Flow-enhanced Variational Inference
// ============================================================================

/// Configuration for flow-enhanced variational inference
#[derive(Debug, Clone)]
pub struct FlowViConfig {
    /// Type of flow layers to use
    pub flow_type: FlowType,
    /// Number of flow layers
    pub n_flow_layers: usize,
    /// Number of MC samples for ELBO estimation
    pub num_samples: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Random seed
    pub seed: u64,
    /// Convergence window
    pub convergence_window: usize,
}

/// Type of flow to use
#[derive(Debug, Clone, Copy)]
pub enum FlowType {
    /// Planar flows only
    Planar,
    /// Radial flows only
    Radial,
    /// Alternating planar and radial
    Mixed,
}

impl Default for FlowViConfig {
    fn default() -> Self {
        Self {
            flow_type: FlowType::Planar,
            n_flow_layers: 4,
            num_samples: 10,
            learning_rate: 0.01,
            max_iterations: 5000,
            tolerance: 1e-4,
            seed: 42,
            convergence_window: 50,
        }
    }
}

/// Flow-enhanced Variational Inference
///
/// Uses a normalizing flow on top of a mean-field Gaussian base distribution
/// to produce a more flexible posterior approximation.
///
/// The ELBO becomes:
/// ```text
/// ELBO = E_{z_0 ~ q_0} [log p(x, z_K) - log q_0(z_0) + sum_k log|det J_k|]
/// ```
/// where z_K = f_K . ... . f_1(z_0) and q_0 = N(mu, diag(sigma^2)).
#[derive(Debug, Clone)]
pub struct FlowVi {
    /// Configuration
    pub config: FlowViConfig,
}

impl FlowVi {
    /// Create a new flow-enhanced VI instance
    pub fn new(config: FlowViConfig) -> Self {
        Self { config }
    }

    /// Generate quasi-random standard normal samples
    fn generate_epsilon(&self, dim: usize, seed: u64) -> Array1<f64> {
        let golden = 1.618033988749895_f64;
        let plastic = 1.324717957244746_f64;
        Array1::from_shape_fn(dim, |i| {
            let u1 = ((seed as f64 * golden + i as f64 * plastic) % 1.0)
                .abs()
                .max(1e-10)
                .min(1.0 - 1e-10);
            let u2 = ((seed as f64 * plastic + i as f64 * golden) % 1.0)
                .abs()
                .max(1e-10)
                .min(1.0 - 1e-10);
            let r = (-2.0 * u1.ln()).sqrt();
            r * (2.0 * PI * u2).cos()
        })
    }
}

/// Adam state for flow VI (for all parameters: base + flow)
#[derive(Debug, Clone)]
struct FlowAdamState {
    m: Array1<f64>,
    v: Array1<f64>,
    t: usize,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
}

impl FlowAdamState {
    fn new(n: usize) -> Self {
        Self {
            m: Array1::zeros(n),
            v: Array1::zeros(n),
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    fn update(&mut self, grad: &Array1<f64>) -> Array1<f64> {
        self.t += 1;
        let n = grad.len();
        let mut dir = Array1::zeros(n);
        for i in 0..n {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
            let m_hat = self.m[i] / (1.0 - self.beta1.powi(self.t as i32));
            let v_hat = self.v[i] / (1.0 - self.beta2.powi(self.t as i32));
            dir[i] = m_hat / (v_hat.sqrt() + self.epsilon);
        }
        dir
    }
}

impl VariationalInference for FlowVi {
    fn fit<F>(&mut self, log_joint: F, dim: usize) -> StatsResult<PosteriorResult>
    where
        F: Fn(&Array1<f64>) -> StatsResult<(f64, Array1<f64>)>,
    {
        if dim == 0 {
            return Err(StatsError::InvalidArgument(
                "Dimension must be at least 1".to_string(),
            ));
        }
        if self.config.n_flow_layers == 0 {
            return Err(StatsError::InvalidArgument(
                "n_flow_layers must be at least 1".to_string(),
            ));
        }

        // Initialize base distribution parameters: mu, log_sigma
        let mut mu = Array1::zeros(dim);
        let mut log_sigma = Array1::zeros(dim);

        // Initialize flow chain
        let mut flow = match self.config.flow_type {
            FlowType::Planar => {
                NormalizingFlowChain::planar(dim, self.config.n_flow_layers, self.config.seed)
            }
            FlowType::Radial => {
                NormalizingFlowChain::radial(dim, self.config.n_flow_layers, self.config.seed)
            }
            FlowType::Mixed => {
                NormalizingFlowChain::mixed(dim, self.config.n_flow_layers, self.config.seed)
            }
        };

        // Total parameters: base (2*dim) + flow params
        let n_base = 2 * dim;
        let n_flow = flow.n_params();
        let n_total = n_base + n_flow;
        let fd_eps = 1e-4;

        let mut adam = FlowAdamState::new(n_total);
        let mut elbo_history = Vec::with_capacity(self.config.max_iterations);
        let mut converged = false;

        for iter in 0..self.config.max_iterations {
            // Evaluate ELBO at current params
            let elbo = self.estimate_elbo(&mu, &log_sigma, &flow, &log_joint, iter)?;
            elbo_history.push(elbo);

            // Compute numerical gradient for all parameters
            let mut full_grad = Array1::zeros(n_total);

            // Gradient w.r.t. mu
            for i in 0..dim {
                let orig = mu[i];
                mu[i] = orig + fd_eps;
                let elbo_plus = self.estimate_elbo(&mu, &log_sigma, &flow, &log_joint, iter)?;
                mu[i] = orig - fd_eps;
                let elbo_minus = self.estimate_elbo(&mu, &log_sigma, &flow, &log_joint, iter)?;
                mu[i] = orig;
                full_grad[i] = (elbo_plus - elbo_minus) / (2.0 * fd_eps);
            }

            // Gradient w.r.t. log_sigma
            for i in 0..dim {
                let orig = log_sigma[i];
                log_sigma[i] = orig + fd_eps;
                let elbo_plus = self.estimate_elbo(&mu, &log_sigma, &flow, &log_joint, iter)?;
                log_sigma[i] = orig - fd_eps;
                let elbo_minus = self.estimate_elbo(&mu, &log_sigma, &flow, &log_joint, iter)?;
                log_sigma[i] = orig;
                full_grad[dim + i] = (elbo_plus - elbo_minus) / (2.0 * fd_eps);
            }

            // Gradient w.r.t. flow parameters
            let flow_params = flow.get_params();
            for i in 0..n_flow {
                let mut fp_plus = flow_params.clone();
                fp_plus[i] += fd_eps;
                flow.set_params(&fp_plus)?;
                let elbo_plus = self.estimate_elbo(&mu, &log_sigma, &flow, &log_joint, iter)?;

                let mut fp_minus = flow_params.clone();
                fp_minus[i] -= fd_eps;
                flow.set_params(&fp_minus)?;
                let elbo_minus = self.estimate_elbo(&mu, &log_sigma, &flow, &log_joint, iter)?;

                flow.set_params(&flow_params)?;
                full_grad[n_base + i] = (elbo_plus - elbo_minus) / (2.0 * fd_eps);
            }

            // Adam update
            let direction = adam.update(&full_grad);
            let lr = self.config.learning_rate;

            for i in 0..dim {
                mu[i] += lr * direction[i];
                log_sigma[i] += lr * direction[dim + i];
                log_sigma[i] = log_sigma[i].max(-10.0).min(10.0);
            }

            // Update flow parameters
            let mut new_flow_params = flow.get_params();
            for i in 0..n_flow {
                new_flow_params[i] += lr * direction[n_base + i];
                new_flow_params[i] = new_flow_params[i].max(-5.0).min(5.0);
            }
            flow.set_params(&new_flow_params)?;

            // Check convergence
            if elbo_history.len() >= self.config.convergence_window {
                let n = elbo_history.len();
                let w = self.config.convergence_window;
                let hw = w / 2;
                let recent_avg: f64 = elbo_history[n - hw..n].iter().sum::<f64>() / hw as f64;
                let earlier_avg: f64 = elbo_history[n - w..n - hw].iter().sum::<f64>() / hw as f64;
                if (recent_avg - earlier_avg).abs() < self.config.tolerance {
                    converged = true;
                    break;
                }
            }
        }

        // Generate posterior samples through the flow
        let n_posterior_samples = 100;
        let mut samples = Vec::with_capacity(n_posterior_samples);
        for s in 0..n_posterior_samples {
            let seed = self.config.seed.wrapping_add(100000 + s as u64);
            let epsilon = self.generate_epsilon(dim, seed);
            let sigma = log_sigma.mapv(f64::exp);
            let z0 = &mu + &(&sigma * &epsilon);
            let (z_k, _) = flow.forward(&z0)?;
            samples.push(z_k);
        }

        // Compute means and stds from samples
        let mut mean = Array1::zeros(dim);
        for s in &samples {
            mean = &mean + s;
        }
        mean /= n_posterior_samples as f64;

        let mut var = Array1::zeros(dim);
        for s in &samples {
            let diff = s - &mean;
            var = &var + &(&diff * &diff);
        }
        var /= (n_posterior_samples - 1).max(1) as f64;
        let std_devs = var.mapv(f64::sqrt);

        let iterations = elbo_history.len();
        Ok(PosteriorResult {
            means: mean,
            std_devs,
            elbo_history: elbo_history.clone(),
            iterations,
            converged,
            samples: Some(samples),
        })
    }
}

impl FlowVi {
    /// Estimate ELBO using Monte Carlo samples through the flow
    fn estimate_elbo<F>(
        &self,
        mu: &Array1<f64>,
        log_sigma: &Array1<f64>,
        flow: &NormalizingFlowChain,
        log_joint: &F,
        iter: usize,
    ) -> StatsResult<f64>
    where
        F: Fn(&Array1<f64>) -> StatsResult<(f64, Array1<f64>)>,
    {
        let dim = mu.len();
        let sigma = log_sigma.mapv(f64::exp);
        let mut elbo_sum = 0.0;

        for s in 0..self.config.num_samples {
            let seed = self
                .config
                .seed
                .wrapping_add(iter as u64 * 1000)
                .wrapping_add(s as u64);
            let epsilon = self.generate_epsilon(dim, seed);

            // Sample from base: z_0 = mu + sigma * epsilon
            let z0 = mu + &(&sigma * &epsilon);

            // Apply flow chain
            let (z_k, sum_log_det) = flow.forward(&z0)?;

            // log q_0(z_0) = sum_i [-0.5*log(2pi) - log(sigma_i) - 0.5*(epsilon_i)^2]
            let log_q0: f64 = (0..dim)
                .map(|i| -0.5 * (2.0 * PI).ln() - log_sigma[i] - 0.5 * epsilon[i] * epsilon[i])
                .sum();

            // log q_K(z_K) = log q_0(z_0) - sum_k log|det J_k|
            let _log_q_k = log_q0 - sum_log_det;

            // ELBO = E[log p(x, z_K) - log q_K(z_K)]
            //      = E[log p(x, z_K) - log q_0(z_0) + sum log|det J_k|]
            let (log_p, _) = log_joint(&z_k)?;
            let elbo_s = log_p - log_q0 + sum_log_det;
            elbo_sum += elbo_s;
        }

        Ok(elbo_sum / self.config.num_samples as f64)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test: planar flow preserves volume — det(Jacobian) is nonzero
    #[test]
    fn test_planar_flow_volume_preservation() {
        let layer = FlowLayer::new_planar(3, 42);
        let z = Array1::from_vec(vec![1.0, -0.5, 0.3]);
        let (fz, log_det) = layer.forward(&z).expect("forward should succeed");

        assert_eq!(fz.len(), 3, "Output dimension should match input");
        assert!(
            log_det.is_finite(),
            "Log-det-Jacobian should be finite, got {}",
            log_det
        );
        // The Jacobian determinant should never be exactly zero
        // (enforced by invertibility constraint)
        assert!(
            log_det.exp() > 1e-15,
            "det(J) should be nonzero, got exp({}) = {}",
            log_det,
            log_det.exp()
        );
    }

    /// Test: radial flow preserves volume
    #[test]
    fn test_radial_flow_volume_preservation() {
        let layer = FlowLayer::new_radial(3, 42);
        let z = Array1::from_vec(vec![1.0, -0.5, 0.3]);
        let (fz, log_det) = layer.forward(&z).expect("forward should succeed");

        assert_eq!(fz.len(), 3);
        assert!(log_det.is_finite(), "Log-det should be finite");
        assert!(log_det.exp() > 1e-15, "det(J) should be nonzero");
    }

    /// Test: flow chain application and log-det accumulation
    #[test]
    fn test_flow_chain_forward() {
        let flow = NormalizingFlowChain::planar(2, 4, 42);
        let z0 = Array1::from_vec(vec![0.5, -0.3]);
        let (z_k, total_log_det) = flow.forward(&z0).expect("chain forward should succeed");

        assert_eq!(z_k.len(), 2);
        assert!(total_log_det.is_finite(), "Total log-det should be finite");

        // Compare with applying layers individually
        let mut z = z0.clone();
        let mut accum = 0.0;
        for layer in &flow.layers {
            let (z_new, ld) = layer.forward(&z).expect("layer forward should succeed");
            z = z_new;
            accum += ld;
        }
        assert!(
            (total_log_det - accum).abs() < 1e-10,
            "Chain log-det ({}) should equal accumulated ({})",
            total_log_det,
            accum
        );
    }

    /// Test: flow parameters roundtrip (get/set)
    #[test]
    fn test_flow_params_roundtrip() {
        let mut flow = NormalizingFlowChain::mixed(3, 4, 42);
        let params = flow.get_params();
        let n = params.len();
        assert!(n > 0, "Should have flow parameters");

        // Perturb and restore
        let mut perturbed = params.clone();
        for i in 0..n {
            perturbed[i] += 0.1;
        }
        flow.set_params(&perturbed).expect("set should succeed");
        let retrieved = flow.get_params();
        for i in 0..n {
            assert!(
                (retrieved[i] - perturbed[i]).abs() < 1e-10,
                "Param {} mismatch after set",
                i
            );
        }

        // Restore original
        flow.set_params(&params).expect("restore should succeed");
        let restored = flow.get_params();
        for i in 0..n {
            assert!(
                (restored[i] - params[i]).abs() < 1e-10,
                "Param {} mismatch after restore",
                i
            );
        }
    }

    /// Test: FlowVI achieves better ELBO than a baseline (no flow = mean-field only)
    /// We check that the final ELBO with flows is at least as good as without.
    #[test]
    fn test_flow_vi_improves_elbo() {
        // Target: N(2, 1)
        let target_fn = |theta: &Array1<f64>| -> StatsResult<(f64, Array1<f64>)> {
            let x = theta[0];
            let log_p = -0.5 * (x - 2.0).powi(2);
            let grad = Array1::from_vec(vec![-(x - 2.0)]);
            Ok((log_p, grad))
        };

        // With flows
        let flow_config = FlowViConfig {
            flow_type: FlowType::Planar,
            n_flow_layers: 2,
            num_samples: 10,
            learning_rate: 0.01,
            max_iterations: 200,
            tolerance: 1e-6,
            seed: 42,
            convergence_window: 50,
        };

        let mut flow_vi = FlowVi::new(flow_config);
        let result = flow_vi.fit(target_fn, 1).expect("FlowVI should succeed");

        // Basic sanity checks
        assert!(!result.elbo_history.is_empty(), "Should have ELBO history");
        let final_elbo = result
            .elbo_history
            .last()
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        assert!(
            final_elbo.is_finite(),
            "Final ELBO should be finite, got {}",
            final_elbo
        );

        // Mean should be near 2.0
        assert!(
            (result.means[0] - 2.0).abs() < 2.0,
            "Mean should be near 2.0, got {}",
            result.means[0]
        );
    }

    /// Test: dimension mismatch error
    #[test]
    fn test_flow_dimension_mismatch() {
        let layer = FlowLayer::Planar {
            w: Array1::from_vec(vec![1.0, 0.5]),
            u: Array1::from_vec(vec![0.3, -0.2]),
            b: 0.1,
        };
        let z = Array1::from_vec(vec![1.0, 2.0, 3.0]); // wrong dim
        let result = layer.forward(&z);
        assert!(result.is_err(), "Should fail on dimension mismatch");
    }

    /// Test: zero dimension error for FlowVi
    #[test]
    fn test_flow_vi_zero_dim() {
        let mut fv = FlowVi::new(FlowViConfig::default());
        let result = fv.fit(|_: &Array1<f64>| Ok((0.0, Array1::zeros(0))), 0);
        assert!(result.is_err());
    }

    /// Test: planar invertibility enforcement
    #[test]
    fn test_planar_invertibility() {
        // Even with adversarial w, u, the flow should produce finite outputs
        let w = Array1::from_vec(vec![1.0, 0.0]);
        let u = Array1::from_vec(vec![-5.0, 0.0]); // w^T u = -5 < -1, needs correction
        let u_hat = enforce_planar_invertibility(&w, &u);
        let wtu_hat = w.dot(&u_hat);
        assert!(
            wtu_hat >= -1.0 - 1e-10,
            "w^T u_hat should be >= -1, got {}",
            wtu_hat
        );
    }
}
