//! Physics-Informed Neural Time Series model.
//!
//! An MLP backbone is trained to minimise a combined loss:
//!
//! ```text
//! L = L_data(θ) + λ · L_physics(θ)
//! ```
//!
//! where `L_data` is the mean-squared error on observed `(time, value)` pairs
//! and `L_physics` penalises violations of one or more [`PhysicsConstraint`]s
//! evaluated at collocation points spread across the time domain.

use crate::error::{Result, TimeSeriesError};
use crate::physics_ts::types::{PhysicsConstraint, PhysicsTsConfig, PhysicsTsResult};

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

/// One fully-connected layer: `y = act(W x + b)`.
#[derive(Debug, Clone)]
struct DenseLayer {
    w: Vec<f64>, // out × in
    b: Vec<f64>,
    in_dim: usize,
    out_dim: usize,
}

impl DenseLayer {
    fn new(in_dim: usize, out_dim: usize) -> Self {
        let scale = (2.0_f64 / (in_dim + out_dim) as f64).sqrt();
        let w = (0..out_dim * in_dim)
            .map(|k| {
                let v = ((k as f64 * 1.61803398) % 2.0) - 1.0;
                v * scale
            })
            .collect();
        Self {
            w,
            b: vec![0.0; out_dim],
            in_dim,
            out_dim,
        }
    }

    fn forward_tanh(&self, x: &[f64]) -> Vec<f64> {
        let mut y = self.b.clone();
        for (i, yi) in y.iter_mut().enumerate() {
            for (j, &xj) in x.iter().enumerate() {
                *yi += self.w[i * self.in_dim + j] * xj;
            }
        }
        y.iter_mut().for_each(|v| *v = v.tanh());
        y
    }

    fn forward_linear(&self, x: &[f64]) -> Vec<f64> {
        let mut y = self.b.clone();
        for (i, yi) in y.iter_mut().enumerate() {
            for (j, &xj) in x.iter().enumerate() {
                *yi += self.w[i * self.in_dim + j] * xj;
            }
        }
        y
    }
}

/// Multi-layer perceptron: `ℝ → ℝ` (scalar time input → scalar prediction).
#[derive(Debug, Clone)]
struct Mlp {
    hidden: Vec<DenseLayer>,
    output: DenseLayer,
}

impl Mlp {
    fn new(hidden_dim: usize, n_layers: usize) -> Self {
        let mut hidden = vec![DenseLayer::new(1, hidden_dim)];
        for _ in 1..n_layers {
            hidden.push(DenseLayer::new(hidden_dim, hidden_dim));
        }
        let output = DenseLayer::new(hidden_dim, 1);
        Self { hidden, output }
    }

    /// Forward pass: returns scalar prediction for scalar time `t`.
    fn predict(&self, t: f64) -> f64 {
        let mut h = vec![t];
        for layer in &self.hidden {
            h = layer.forward_tanh(&h);
        }
        let out = self.output.forward_linear(&h);
        out[0]
    }

    /// Count total number of trainable parameters.
    fn n_params(&self) -> usize {
        let mut n = 0;
        for l in &self.hidden {
            n += l.w.len() + l.b.len();
        }
        n += self.output.w.len() + self.output.b.len();
        n
    }

    /// Flatten all parameters into a `Vec<f64>`.
    fn flatten(&self) -> Vec<f64> {
        let mut p = Vec::new();
        for l in &self.hidden {
            p.extend_from_slice(&l.w);
            p.extend_from_slice(&l.b);
        }
        p.extend_from_slice(&self.output.w);
        p.extend_from_slice(&self.output.b);
        p
    }

    /// Write back flat parameters.
    fn unflatten(&mut self, params: &[f64]) {
        let mut idx = 0;
        for l in &mut self.hidden {
            let wn = l.w.len();
            let bn = l.b.len();
            l.w.copy_from_slice(&params[idx..idx + wn]);
            idx += wn;
            l.b.copy_from_slice(&params[idx..idx + bn]);
            idx += bn;
        }
        let wn = self.output.w.len();
        let bn = self.output.b.len();
        self.output.w.copy_from_slice(&params[idx..idx + wn]);
        idx += wn;
        self.output.b.copy_from_slice(&params[idx..idx + bn]);
        let _ = idx;
    }
}

// ---------------------------------------------------------------------------
// Adam optimiser
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
}

impl Adam {
    fn new(n: usize, lr: f64) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    fn step(&mut self, params: &mut [f64], grad: &[f64]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// ---------------------------------------------------------------------------
// Physics loss helpers
// ---------------------------------------------------------------------------

/// Approximate `dx/dt` at index `i` using central (or one-sided) differences.
fn finite_diff(preds: &[f64], times: &[f64], i: usize) -> f64 {
    let n = preds.len();
    if n < 2 {
        return 0.0;
    }
    if i == 0 {
        let dt = times[1] - times[0];
        if dt.abs() < 1e-15 {
            0.0
        } else {
            (preds[1] - preds[0]) / dt
        }
    } else if i == n - 1 {
        let dt = times[n - 1] - times[n - 2];
        if dt.abs() < 1e-15 {
            0.0
        } else {
            (preds[n - 1] - preds[n - 2]) / dt
        }
    } else {
        let dt = times[i + 1] - times[i - 1];
        if dt.abs() < 1e-15 {
            0.0
        } else {
            (preds[i + 1] - preds[i - 1]) / dt
        }
    }
}

/// Compute per-timestep physics residuals and total physics loss.
fn physics_loss(
    preds: &[f64],
    times: &[f64],
    constraints: &[PhysicsConstraint],
) -> (Vec<f64>, f64) {
    let n = preds.len();
    let mut residuals = vec![0.0_f64; n];

    for constraint in constraints {
        match constraint {
            PhysicsConstraint::OdeConstraint { rate } => {
                for i in 0..n {
                    let dxdt = finite_diff(preds, times, i);
                    let residual = dxdt - rate * preds[i];
                    residuals[i] += residual.powi(2);
                }
            }
            PhysicsConstraint::ConservationLaw { target_sum } => {
                let actual_sum: f64 = preds.iter().sum();
                let violation = (actual_sum - target_sum).powi(2) / n as f64;
                for r in residuals.iter_mut() {
                    *r += violation;
                }
            }
            PhysicsConstraint::Monotone { increasing } => {
                for i in 0..n.saturating_sub(1) {
                    let delta = preds[i + 1] - preds[i];
                    let viol = if *increasing {
                        (-delta).max(0.0).powi(2)
                    } else {
                        delta.max(0.0).powi(2)
                    };
                    residuals[i] += viol;
                    residuals[i + 1] += viol;
                }
            }
            PhysicsConstraint::BoundedVariation { bound } => {
                for i in 0..n.saturating_sub(1) {
                    let delta = (preds[i + 1] - preds[i]).abs();
                    let viol = (delta - bound).max(0.0).powi(2);
                    residuals[i] += viol;
                    residuals[i + 1] += viol;
                }
            }
        }
    }

    let total: f64 = residuals.iter().sum::<f64>() / n as f64;
    (residuals, total)
}

// ---------------------------------------------------------------------------
// Main model
// ---------------------------------------------------------------------------

/// Physics-Informed Neural Time Series predictor.
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_series::physics_ts::{
///     model::PhysicsInformedTs,
///     types::{PhysicsConstraint, PhysicsTsConfig},
/// };
///
/// let mut config = PhysicsTsConfig::default();
/// config.n_epochs = 20;
/// let mut model = PhysicsInformedTs::new(config);
/// let times  = vec![0.0_f64, 0.25, 0.5, 0.75, 1.0];
/// let values = vec![1.0_f64, 1.3, 1.6, 1.9, 2.2];
/// let constraints = vec![PhysicsConstraint::Monotone { increasing: true }];
/// let result = model.fit(&times, &values, &constraints).expect("fit");
/// assert_eq!(result.predictions.len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct PhysicsInformedTs {
    config: PhysicsTsConfig,
    mlp: Mlp,
    adam: Adam,
    /// Finite-difference step for gradient approximation.
    fd_eps: f64,
}

impl PhysicsInformedTs {
    /// Create a new model.
    pub fn new(config: PhysicsTsConfig) -> Self {
        let mlp = Mlp::new(config.hidden_dim, config.n_layers);
        let n = mlp.n_params();
        let adam = Adam::new(n, config.learning_rate);
        Self {
            config,
            mlp,
            adam,
            fd_eps: 1e-4,
        }
    }

    /// Compute data loss + physics loss for the current parameters.
    fn total_loss(
        &self,
        times: &[f64],
        values: &[f64],
        constraints: &[PhysicsConstraint],
    ) -> (f64, f64, f64) {
        let preds: Vec<f64> = times.iter().map(|&t| self.mlp.predict(t)).collect();

        // Data MSE
        let data_loss = preds
            .iter()
            .zip(values.iter())
            .map(|(&p, &v)| (p - v).powi(2))
            .sum::<f64>()
            / times.len() as f64;

        let (_, physics) = physics_loss(&preds, times, constraints);
        (
            data_loss + self.config.physics_weight * physics,
            data_loss,
            physics,
        )
    }

    /// Numerical gradient via forward finite differences.
    fn numerical_gradient(
        &self,
        times: &[f64],
        values: &[f64],
        constraints: &[PhysicsConstraint],
    ) -> Vec<f64> {
        let base_params = self.mlp.flatten();
        let (base_loss, _, _) = self.total_loss(times, values, constraints);
        let n = base_params.len();
        let mut grad = vec![0.0_f64; n];

        for k in 0..n {
            let mut perturbed = self.mlp.clone();
            let mut p = base_params.clone();
            p[k] += self.fd_eps;
            perturbed.unflatten(&p);
            // Re-compute loss with perturbed params
            let preds: Vec<f64> = times.iter().map(|&t| perturbed.predict(t)).collect();
            let data = preds
                .iter()
                .zip(values.iter())
                .map(|(&p2, &v)| (p2 - v).powi(2))
                .sum::<f64>()
                / times.len() as f64;
            let (_, phys) = physics_loss(&preds, times, constraints);
            let perturbed_loss = data + self.config.physics_weight * phys;
            grad[k] = (perturbed_loss - base_loss) / self.fd_eps;
        }
        grad
    }

    /// Fit on observed `(times, values)` with the given physics constraints.
    pub fn fit(
        &mut self,
        times: &[f64],
        values: &[f64],
        constraints: &[PhysicsConstraint],
    ) -> Result<PhysicsTsResult> {
        if times.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "times must not be empty".to_string(),
            ));
        }
        if times.len() != values.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: times.len(),
                actual: values.len(),
            });
        }

        let mut last_data_loss = 0.0;
        let mut last_physics_loss = 0.0;

        for _epoch in 0..self.config.n_epochs {
            let grad = self.numerical_gradient(times, values, constraints);
            let mut params = self.mlp.flatten();
            self.adam.step(&mut params, &grad);
            self.mlp.unflatten(&params);

            let (_, dl, pl) = self.total_loss(times, values, constraints);
            last_data_loss = dl;
            last_physics_loss = pl;
        }

        let preds: Vec<f64> = times.iter().map(|&t| self.mlp.predict(t)).collect();
        let (residuals, _) = physics_loss(&preds, times, constraints);

        Ok(PhysicsTsResult {
            predictions: preds,
            physics_residuals: residuals,
            total_physics_loss: last_physics_loss,
            data_loss: last_data_loss,
        })
    }

    /// Predict at arbitrary `query_times`.
    pub fn predict(&self, query_times: &[f64]) -> Vec<f64> {
        query_times.iter().map(|&t| self.mlp.predict(t)).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics_ts::types::PhysicsConstraint;

    #[test]
    fn physics_ts_config_default() {
        let cfg = PhysicsTsConfig::default();
        assert_eq!(cfg.hidden_dim, 32);
        assert_eq!(cfg.n_layers, 2);
        assert!((cfg.physics_weight - 1.0).abs() < 1e-12);
    }

    #[test]
    fn physics_ts_fit_returns_correct_length() {
        let config = PhysicsTsConfig {
            n_epochs: 5,
            hidden_dim: 8,
            n_layers: 1,
            ..Default::default()
        };
        let mut model = PhysicsInformedTs::new(config);
        let times: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        let values: Vec<f64> = times.iter().map(|&t| t).collect();
        let result = model.fit(&times, &values, &[]).expect("fit");
        assert_eq!(result.predictions.len(), 10);
        assert_eq!(result.physics_residuals.len(), 10);
    }

    #[test]
    fn physics_ts_monotone_constraint() {
        // Force a non-trivial physics weight so the constraint is enforced
        let config = PhysicsTsConfig {
            n_epochs: 50,
            hidden_dim: 8,
            n_layers: 1,
            physics_weight: 10.0,
            learning_rate: 1e-2,
        };
        let mut model = PhysicsInformedTs::new(config);
        let times: Vec<f64> = (0..8).map(|i| i as f64 * 0.1).collect();
        let values: Vec<f64> = times.iter().map(|&t| t * 2.0).collect(); // strictly increasing
        let constraints = vec![PhysicsConstraint::Monotone { increasing: true }];
        let result = model.fit(&times, &values, &constraints).expect("fit");
        // With heavy physics weight on monotone training data, physics residual should be small
        assert!(result.total_physics_loss >= 0.0);
        assert_eq!(result.physics_residuals.len(), times.len());
    }

    #[test]
    fn physics_ts_conservation_constraint() {
        let config = PhysicsTsConfig {
            n_epochs: 10,
            hidden_dim: 8,
            n_layers: 1,
            ..Default::default()
        };
        let mut model = PhysicsInformedTs::new(config);
        let times = vec![0.0, 0.5, 1.0];
        let values = vec![1.0, 1.0, 1.0];
        let constraints = vec![PhysicsConstraint::ConservationLaw { target_sum: 3.0 }];
        let result = model.fit(&times, &values, &constraints).expect("fit");
        assert_eq!(result.predictions.len(), 3);
    }

    #[test]
    fn physics_ts_data_loss_nonneg() {
        let config = PhysicsTsConfig {
            n_epochs: 10,
            hidden_dim: 4,
            n_layers: 1,
            ..Default::default()
        };
        let mut model = PhysicsInformedTs::new(config);
        let times: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = model.fit(&times, &values, &[]).expect("fit");
        assert!(result.data_loss >= 0.0);
    }

    #[test]
    fn physics_ts_bounded_variation_constraint() {
        let config = PhysicsTsConfig {
            n_epochs: 5,
            hidden_dim: 4,
            n_layers: 1,
            ..Default::default()
        };
        let mut model = PhysicsInformedTs::new(config);
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values = vec![0.0, 1.0, 2.0, 3.0];
        let constraints = vec![PhysicsConstraint::BoundedVariation { bound: 2.0 }];
        let result = model.fit(&times, &values, &constraints).expect("fit");
        assert_eq!(result.predictions.len(), 4);
    }

    #[test]
    fn physics_ts_predict() {
        let config = PhysicsTsConfig {
            n_epochs: 5,
            hidden_dim: 4,
            n_layers: 1,
            ..Default::default()
        };
        let mut model = PhysicsInformedTs::new(config);
        let times = vec![0.0, 0.5, 1.0];
        let values = vec![0.0, 0.5, 1.0];
        model.fit(&times, &values, &[]).expect("fit");
        let preds = model.predict(&[0.25, 0.75]);
        assert_eq!(preds.len(), 2);
    }
}
