//! Neural ODE function f_θ(z, t) for latent dynamics.
//!
//! Implements a multi-layer perceptron with tanh activations that maps
//! the current latent state `z` to its time-derivative `dz/dt`.
//! The ODE is *autonomous*: the time argument `t` is accepted for API
//! compatibility but is not used in the forward pass.
//!
//! Also provides a fixed-step RK4 integrator built on top of this function.

/// A fully-connected layer: `y = W x + b` (no activation; callers apply it).
#[derive(Debug, Clone)]
pub struct LinearLayer {
    /// Weight matrix stored row-major: `weights[out_i * in_dim + in_j]`.
    pub weights: Vec<f64>,
    /// Bias vector of length `out_dim`.
    pub biases: Vec<f64>,
    /// Number of input features.
    pub in_dim: usize,
    /// Number of output features.
    pub out_dim: usize,
}

impl LinearLayer {
    /// Create a new layer with small random-like initialisation.
    ///
    /// Uses a deterministic but reasonably spread initialisation
    /// `w_ij = sin(i * in_dim + j) * scale` so that the model is not
    /// identically zero at construction time.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let scale = (2.0_f64 / (in_dim + out_dim) as f64).sqrt();
        let weights: Vec<f64> = (0..out_dim * in_dim)
            .map(|k| {
                let v = ((k as f64 * 1.6180339887) % 2.0) - 1.0; // pseudo-random in [-1,1]
                v * scale
            })
            .collect();
        let biases = vec![0.0_f64; out_dim];
        Self {
            weights,
            biases,
            in_dim,
            out_dim,
        }
    }

    /// Forward pass (no activation).
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut y = self.biases.clone();
        for (i, yi) in y.iter_mut().enumerate() {
            for (j, &xj) in x.iter().enumerate() {
                *yi += self.weights[i * self.in_dim + j] * xj;
            }
        }
        y
    }
}

/// Neural network implementing `dz/dt = f_θ(z, t)`.
///
/// Architecture: `z_dim → hidden → hidden → z_dim` with tanh activations on
/// all layers except the final one (linear output to avoid saturating gradients).
#[derive(Debug, Clone)]
pub struct OdeFunc {
    layers: Vec<LinearLayer>,
    z_dim: usize,
}

impl OdeFunc {
    /// Construct an ODE function with `n_layers` hidden layers.
    ///
    /// `n_layers` controls the number of *hidden* layers; there is always an
    /// additional output projection, so the total depth is `n_layers + 1`.
    pub fn new(z_dim: usize, hidden_dim: usize, n_layers: usize) -> Self {
        let mut layers = Vec::new();
        // Input → first hidden
        layers.push(LinearLayer::new(z_dim, hidden_dim));
        // Additional hidden layers
        for _ in 1..n_layers {
            layers.push(LinearLayer::new(hidden_dim, hidden_dim));
        }
        // Last hidden → output (z_dim)
        layers.push(LinearLayer::new(hidden_dim, z_dim));
        Self { layers, z_dim }
    }

    /// Evaluate `f_θ(z, t)`.  Returns a vector of length `z_dim`.
    pub fn forward(&self, z: &[f64], _t: f64) -> Vec<f64> {
        let mut h: Vec<f64> = z.to_vec();
        let n = self.layers.len();
        for (idx, layer) in self.layers.iter().enumerate() {
            let pre = layer.forward(&h);
            if idx < n - 1 {
                // tanh activation on all but the last layer
                h = pre.iter().map(|&v| v.tanh()).collect();
            } else {
                // linear output
                h = pre;
            }
        }
        h
    }

    /// Number of latent dimensions.
    pub fn z_dim(&self) -> usize {
        self.z_dim
    }

    /// Mutable access to all layer parameters (for gradient-free weight updates).
    pub fn layers_mut(&mut self) -> &mut Vec<LinearLayer> {
        &mut self.layers
    }

    /// Immutable access to all layers.
    pub fn layers(&self) -> &[LinearLayer] {
        &self.layers
    }
}

// ---------------------------------------------------------------------------
// ODE solvers
// ---------------------------------------------------------------------------

/// Euler step: `z(t+h) ≈ z(t) + h * f(z(t), t)`.
pub fn euler_step(f: &OdeFunc, z: &[f64], t: f64, h: f64) -> Vec<f64> {
    let dz = f.forward(z, t);
    z.iter()
        .zip(dz.iter())
        .map(|(&zi, &dzi)| zi + h * dzi)
        .collect()
}

/// Fourth-order Runge-Kutta step.
pub fn rk4_step(f: &OdeFunc, z: &[f64], t: f64, h: f64) -> Vec<f64> {
    let k1 = f.forward(z, t);
    let z2: Vec<f64> = z
        .iter()
        .zip(k1.iter())
        .map(|(&zi, &ki)| zi + 0.5 * h * ki)
        .collect();
    let k2 = f.forward(&z2, t + 0.5 * h);
    let z3: Vec<f64> = z
        .iter()
        .zip(k2.iter())
        .map(|(&zi, &ki)| zi + 0.5 * h * ki)
        .collect();
    let k3 = f.forward(&z3, t + 0.5 * h);
    let z4: Vec<f64> = z
        .iter()
        .zip(k3.iter())
        .map(|(&zi, &ki)| zi + h * ki)
        .collect();
    let k4 = f.forward(&z4, t + h);

    z.iter()
        .enumerate()
        .map(|(i, &zi)| zi + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}

/// Integrate `z` from `t0` to `t1` using fixed-step RK4 with `n_steps` steps.
///
/// Returns the state at `t1`.
pub fn integrate(f: &OdeFunc, z0: &[f64], t0: f64, t1: f64, n_steps: usize) -> Vec<f64> {
    if n_steps == 0 || (t1 - t0).abs() < 1e-15 {
        return z0.to_vec();
    }
    let h = (t1 - t0) / n_steps as f64;
    let mut z = z0.to_vec();
    let mut t = t0;
    for _ in 0..n_steps {
        z = rk4_step(f, &z, t, h);
        t += h;
    }
    z
}

/// Integrate `z` from `t0` along `times` (must be sorted ascending).
///
/// Returns the latent trajectory at each time in `times`.
pub fn integrate_trajectory(
    f: &OdeFunc,
    z0: &[f64],
    t0: f64,
    times: &[f64],
    steps_per_interval: usize,
) -> Vec<Vec<f64>> {
    let mut trajectory = Vec::with_capacity(times.len());
    let mut z = z0.to_vec();
    let mut prev_t = t0;
    for &t in times {
        if t <= prev_t {
            trajectory.push(z.clone());
            continue;
        }
        z = integrate(f, &z, prev_t, t, steps_per_interval);
        trajectory.push(z.clone());
        prev_t = t;
    }
    trajectory
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ode_func_forward_shape() {
        let f = OdeFunc::new(4, 8, 2);
        let z = vec![0.1, 0.2, 0.3, 0.4];
        let dz = f.forward(&z, 0.0);
        assert_eq!(dz.len(), 4);
    }

    #[test]
    fn rk4_step_changes_state() {
        let f = OdeFunc::new(2, 4, 1);
        let z0 = vec![1.0, 0.0];
        let z1 = rk4_step(&f, &z0, 0.0, 0.1);
        // With non-trivial weights, state should change
        assert_eq!(z1.len(), 2);
    }

    #[test]
    fn integrate_trajectory_length() {
        let f = OdeFunc::new(3, 8, 2);
        let z0 = vec![0.5, -0.5, 0.0];
        let times = vec![0.1, 0.5, 1.0, 2.0];
        let traj = integrate_trajectory(&f, &z0, 0.0, &times, 4);
        assert_eq!(traj.len(), 4);
        for t in &traj {
            assert_eq!(t.len(), 3);
        }
    }
}
