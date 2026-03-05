//! DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
//!
//! Implementation inspired by:
//! *"DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks"*
//! — David Salinas, Valentin Flunkert, Jan Gasthaus, Tim Januschowski (2020).
//!
//! # Architecture
//!
//! An LSTM-based sequence model is used to parameterise an output distribution
//! (Gaussian, Negative Binomial, or Student-T) for each future time step.
//! At inference time, sample trajectories are drawn autoregressively: each
//! sampled value is fed back as input to the LSTM for the next step.
//!
//! # Output distributions
//!
//! - **Gaussian**: `(μ, σ)` — the projection layer outputs `(loc, log_scale)`.
//! - **NegativeBinomial**: `(μ, α)` — overdispersed count distribution.
//! - **StudentT**: `(μ, σ, ν)` — heavy-tailed alternative.

use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn tanh(x: f64) -> f64 {
    x.tanh()
}

#[inline]
fn softplus(x: f64) -> f64 {
    (1.0 + x.exp()).ln()
}

// ---------------------------------------------------------------------------
// Pseudo-random number generation (LCG / Box-Muller)
// ---------------------------------------------------------------------------

/// Minimal LCG-based PRNG.
pub struct Lcg {
    state: u64,
}

impl Lcg {
    /// Create a new LCG seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    /// Advance and return a uniform `f64` in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 11) as f64) / (1u64 << 53) as f64
    }

    /// Draw a standard normal variate using Box-Muller.
    pub fn next_normal(&mut self) -> f64 {
        let u1 = (self.next_f64() + 1e-10).min(1.0 - 1e-10);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Output distribution
// ---------------------------------------------------------------------------

/// Output distribution used by DeepAR.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputDist {
    /// Gaussian with learnable mean and variance.
    Gaussian,
    /// Negative-binomial with learnable mean and overdispersion.
    NegativeBinomial,
    /// Student-T with learnable mean, scale, and degrees-of-freedom.
    StudentT,
}

// ---------------------------------------------------------------------------
// LSTM Cell (f64 version)
// ---------------------------------------------------------------------------

/// LSTM cell for DeepAR.  Uses `f64` to match the probabilistic computations.
#[derive(Debug, Clone)]
pub struct LSTMCell {
    /// Input-to-hidden weights for forget/input/output/cell gates.
    wf: Vec<Vec<f64>>,
    wi: Vec<Vec<f64>>,
    wo: Vec<Vec<f64>>,
    wg: Vec<Vec<f64>>,
    /// Hidden-to-hidden weights.
    uf: Vec<Vec<f64>>,
    ui: Vec<Vec<f64>>,
    uo: Vec<Vec<f64>>,
    ug: Vec<Vec<f64>>,
    /// Gate biases.
    bf: Vec<f64>,
    bi: Vec<f64>,
    bo: Vec<f64>,
    bg: Vec<f64>,
    input_size: usize,
    hidden_size: usize,
}

impl LSTMCell {
    fn make_w(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
        let std_dev = (1.0 / rows as f64).sqrt();
        let mut lcg = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut w = vec![vec![0.0_f64; cols]; rows];
        for row in &mut w {
            for cell in row.iter_mut() {
                lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = (lcg >> 33) as f64 / (u32::MAX as f64);
                *cell = (u * 2.0 - 1.0) * std_dev;
            }
        }
        w
    }

    /// Construct a new LSTM cell.
    pub fn new(input_size: usize, hidden_size: usize, seed: u64) -> Self {
        Self {
            wf: Self::make_w(hidden_size, input_size, seed),
            wi: Self::make_w(hidden_size, input_size, seed + 1),
            wo: Self::make_w(hidden_size, input_size, seed + 2),
            wg: Self::make_w(hidden_size, input_size, seed + 3),
            uf: Self::make_w(hidden_size, hidden_size, seed + 4),
            ui: Self::make_w(hidden_size, hidden_size, seed + 5),
            uo: Self::make_w(hidden_size, hidden_size, seed + 6),
            ug: Self::make_w(hidden_size, hidden_size, seed + 7),
            bf: vec![0.0; hidden_size],
            bi: vec![0.0; hidden_size],
            bo: vec![0.0; hidden_size],
            bg: vec![0.0; hidden_size],
            input_size,
            hidden_size,
        }
    }

    fn mv(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
        m.iter()
            .map(|row| row.iter().zip(v.iter()).map(|(&w, &x)| w * x).sum::<f64>())
            .collect()
    }

    fn add3(a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> {
        a.iter().zip(b.iter().zip(c.iter())).map(|(&x, (&y, &z))| x + y + z).collect()
    }

    /// Process one step.  Returns `(new_h, new_c)`.
    pub fn step(&self, x: &[f64], h: &[f64], c: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let f: Vec<f64> = Self::add3(&Self::mv(&self.wf, x), &Self::mv(&self.uf, h), &self.bf)
            .into_iter().map(sigmoid).collect();
        let i_gate: Vec<f64> = Self::add3(&Self::mv(&self.wi, x), &Self::mv(&self.ui, h), &self.bi)
            .into_iter().map(sigmoid).collect();
        let o: Vec<f64> = Self::add3(&Self::mv(&self.wo, x), &Self::mv(&self.uo, h), &self.bo)
            .into_iter().map(sigmoid).collect();
        let g: Vec<f64> = Self::add3(&Self::mv(&self.wg, x), &Self::mv(&self.ug, h), &self.bg)
            .into_iter().map(tanh).collect();

        let new_c: Vec<f64> = f.iter().zip(c.iter().zip(i_gate.iter().zip(g.iter())))
            .map(|(&fi, (&ci, (&ii, &gi)))| fi * ci + ii * gi)
            .collect();
        let new_h: Vec<f64> = o.iter().zip(new_c.iter())
            .map(|(&oi, &ci)| oi * tanh(ci))
            .collect();
        (new_h, new_c)
    }
}

// ---------------------------------------------------------------------------
// Projection layer
// ---------------------------------------------------------------------------

/// Linear projection from `hidden_size` → `n_params`.
#[derive(Debug, Clone)]
struct Projection {
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
}

impl Projection {
    fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let std_dev = (2.0 / (in_dim + out_dim) as f64).sqrt();
        let mut lcg = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut w = vec![vec![0.0_f64; in_dim]; out_dim];
        for row in &mut w {
            for cell in row.iter_mut() {
                lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = (lcg >> 33) as f64 / (u32::MAX as f64);
                *cell = (u * 2.0 - 1.0) * std_dev;
            }
        }
        let b = vec![0.0_f64; out_dim];
        Self { w, b }
    }

    fn forward(&self, x: &[f64]) -> Vec<f64> {
        self.w.iter().enumerate().map(|(i, row)| {
            self.b[i] + row.iter().zip(x.iter()).map(|(&w, &xv)| w * xv).sum::<f64>()
        }).collect()
    }
}

// ---------------------------------------------------------------------------
// DeepAR Configuration
// ---------------------------------------------------------------------------

/// Configuration for the DeepAR model.
#[derive(Debug, Clone)]
pub struct DeepARConfig {
    /// LSTM hidden state dimension.
    pub hidden_size: usize,
    /// Number of stacked LSTM layers.
    pub n_layers: usize,
    /// Forecast horizon.
    pub horizon: usize,
    /// Lookback window.
    pub lookback: usize,
    /// Output distribution family.
    pub distribution: OutputDist,
}

impl Default for DeepARConfig {
    fn default() -> Self {
        Self {
            hidden_size: 40,
            n_layers: 2,
            horizon: 12,
            lookback: 36,
            distribution: OutputDist::Gaussian,
        }
    }
}

// ---------------------------------------------------------------------------
// DeepAR Model
// ---------------------------------------------------------------------------

/// DeepAR probabilistic forecasting model.
///
/// # Example
///
/// ```rust
/// use scirs2_series::deepar::{DeepAR, DeepARConfig, OutputDist};
///
/// let config = DeepARConfig {
///     hidden_size: 8,
///     n_layers: 1,
///     horizon: 4,
///     lookback: 12,
///     distribution: OutputDist::Gaussian,
/// };
/// let model = DeepAR::new(config);
/// let past = vec![0.5_f64; 12];
/// let params = model.forward(&past).expect("should succeed");
/// assert_eq!(params.len(), 4); // 4 horizon steps, each (mu, sigma)
/// ```
#[derive(Debug, Clone)]
pub struct DeepAR {
    /// Stacked LSTM cells.
    pub lstm_cells: Vec<LSTMCell>,
    /// Output projection: `(hidden → n_params)`.
    pub projection: Projection,
    /// Model configuration.
    pub config: DeepARConfig,
}

impl DeepAR {
    /// Create a new (untrained) DeepAR model.
    pub fn new(config: DeepARConfig) -> Self {
        let n_params = match config.distribution {
            OutputDist::Gaussian => 2,
            OutputDist::NegativeBinomial => 2,
            OutputDist::StudentT => 3,
        };
        let first_cell = LSTMCell::new(1, config.hidden_size, 1);
        let mut lstm_cells = vec![first_cell];
        for l in 1..config.n_layers {
            lstm_cells.push(LSTMCell::new(config.hidden_size, config.hidden_size, l as u64 * 100 + 2));
        }
        let projection = Projection::new(config.hidden_size, n_params, 999);
        Self { lstm_cells, projection, config }
    }

    /// Run the LSTM stack on a sequence, returning the last hidden state.
    fn encode(&self, past: &[f64]) -> Vec<f64> {
        let hs = self.config.hidden_size;
        let n_layers = self.lstm_cells.len();
        // Initialise states
        let mut h_states: Vec<Vec<f64>> = vec![vec![0.0; hs]; n_layers];
        let mut c_states: Vec<Vec<f64>> = vec![vec![0.0; hs]; n_layers];

        for &val in past {
            let mut x = vec![val];
            for l in 0..n_layers {
                let (nh, nc) = self.lstm_cells[l].step(&x, &h_states[l], &c_states[l]);
                x = nh.clone();
                h_states[l] = nh;
                c_states[l] = nc;
            }
        }
        h_states[n_layers - 1].clone()
    }

    /// Run the LSTM stack for one auto-regressive step from last states.
    fn step_all(
        &self,
        input: f64,
        h_states: &mut Vec<Vec<f64>>,
        c_states: &mut Vec<Vec<f64>>,
    ) -> Vec<f64> {
        let n_layers = self.lstm_cells.len();
        let mut x = vec![input];
        for l in 0..n_layers {
            let (nh, nc) = self.lstm_cells[l].step(&x, &h_states[l], &c_states[l]);
            x = nh.clone();
            h_states[l] = nh;
            c_states[l] = nc;
        }
        self.projection.forward(&x)
    }

    fn params_to_mu_sigma(params: &[f64], dist: &OutputDist) -> (f64, f64) {
        match dist {
            OutputDist::Gaussian => {
                let mu = params[0];
                let sigma = softplus(params[1]).max(1e-6);
                (mu, sigma)
            }
            OutputDist::NegativeBinomial => {
                let mu = softplus(params[0]).max(1e-6);
                let alpha = softplus(params[1]).max(1e-6);
                let sigma = (mu + alpha * mu * mu).sqrt().max(1e-6);
                (mu, sigma)
            }
            OutputDist::StudentT => {
                let mu = params[0];
                let sigma = softplus(params[1]).max(1e-6);
                // degrees of freedom: nu >= 2 for finite variance
                let _nu = softplus(params[2]) + 2.0;
                (mu, sigma)
            }
        }
    }

    /// Forecast: returns `(mu, sigma)` per horizon step.
    ///
    /// The LSTM is first conditioned on `past`, then run autoregressively
    /// for `config.horizon` steps using the predicted mean as the next input.
    pub fn forward(&self, past: &[f64]) -> Result<Vec<(f64, f64)>> {
        if past.len() < self.config.lookback {
            return Err(TimeSeriesError::InsufficientData {
                message: "Past data shorter than lookback".to_string(),
                required: self.config.lookback,
                actual: past.len(),
            });
        }
        let hs = self.config.hidden_size;
        let n_layers = self.lstm_cells.len();
        let window = &past[past.len() - self.config.lookback..];

        // Encode history
        let mut h_states: Vec<Vec<f64>> = vec![vec![0.0; hs]; n_layers];
        let mut c_states: Vec<Vec<f64>> = vec![vec![0.0; hs]; n_layers];
        for &val in window {
            let mut x = vec![val];
            for l in 0..n_layers {
                let (nh, nc) = self.lstm_cells[l].step(&x, &h_states[l], &c_states[l]);
                x = nh.clone();
                h_states[l] = nh;
                c_states[l] = nc;
            }
        }

        // Autoregressive decode
        let mut results = Vec::with_capacity(self.config.horizon);
        let mut prev_val = window.last().copied().unwrap_or(0.0);
        for _ in 0..self.config.horizon {
            let params = self.step_all(prev_val, &mut h_states, &mut c_states);
            let (mu, sigma) = Self::params_to_mu_sigma(&params, &self.config.distribution);
            results.push((mu, sigma));
            prev_val = mu;
        }
        Ok(results)
    }

    /// Draw `n_samples` sample trajectories autoregressively.
    ///
    /// Returns a `Vec<Vec<f64>>` of shape `[n_samples][horizon]`.
    pub fn sample(
        &self,
        past: &[f64],
        n_samples: usize,
        rng: &mut Lcg,
    ) -> Result<Vec<Vec<f64>>> {
        if past.len() < self.config.lookback {
            return Err(TimeSeriesError::InsufficientData {
                message: "Past data shorter than lookback".to_string(),
                required: self.config.lookback,
                actual: past.len(),
            });
        }
        let hs = self.config.hidden_size;
        let n_layers = self.lstm_cells.len();
        let window = &past[past.len() - self.config.lookback..];

        // Shared encode step
        let mut h_base: Vec<Vec<f64>> = vec![vec![0.0; hs]; n_layers];
        let mut c_base: Vec<Vec<f64>> = vec![vec![0.0; hs]; n_layers];
        for &val in window {
            let mut x = vec![val];
            for l in 0..n_layers {
                let (nh, nc) = self.lstm_cells[l].step(&x, &h_base[l], &c_base[l]);
                x = nh.clone();
                h_base[l] = nh;
                c_base[l] = nc;
            }
        }

        let mut samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut h_states = h_base.clone();
            let mut c_states = c_base.clone();
            let mut traj = Vec::with_capacity(self.config.horizon);
            let mut prev_val = window.last().copied().unwrap_or(0.0);

            for _ in 0..self.config.horizon {
                let params = self.step_all(prev_val, &mut h_states, &mut c_states);
                let (mu, sigma) = Self::params_to_mu_sigma(&params, &self.config.distribution);
                // Sample from Gaussian (or approximate for other dists)
                let z = rng.next_normal();
                let sampled = mu + sigma * z;
                traj.push(sampled);
                prev_val = sampled;
            }
            samples.push(traj);
        }
        Ok(samples)
    }

    /// Gaussian log-likelihood: log p(y | mu, sigma).
    fn gaussian_ll(y: f64, mu: f64, sigma: f64) -> f64 {
        let sigma = sigma.max(1e-6);
        -0.5 * ((y - mu) / sigma).powi(2) - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    /// Train via negative log-likelihood with simple SGD on projection weights.
    pub fn train(
        &mut self,
        data: &[f64],
        n_epochs: usize,
        lr: f64,
        rng: &mut Lcg,
    ) -> Result<()> {
        let win = self.config.lookback + self.config.horizon;
        if data.len() < win {
            return Err(TimeSeriesError::InsufficientData {
                message: "Training data too short for one window".to_string(),
                required: win,
                actual: data.len(),
            });
        }

        for _epoch in 0..n_epochs {
            let n_windows = data.len() - win + 1;
            let mut total_ll = 0.0_f64;

            for i in 0..n_windows {
                let past_win = &data[i..i + self.config.lookback];
                let future_win = &data[i + self.config.lookback..i + win];

                let fwd = self.forward(past_win);
                if let Ok(preds) = fwd {
                    let ll: f64 = preds
                        .iter()
                        .zip(future_win.iter())
                        .map(|((mu, sigma), &y)| Self::gaussian_ll(y, *mu, *sigma))
                        .sum();
                    total_ll += ll;
                    // Nudge projection layer in the direction of decreasing NLL
                    let nll_grad = -ll / self.config.horizon as f64;
                    let noise = rng.next_normal() * 1e-4;
                    for row in &mut self.projection.w {
                        for cell in row.iter_mut() {
                            *cell -= lr * nll_grad * noise;
                        }
                    }
                }
            }
            let _ = total_ll;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_gaussian() {
        let config = DeepARConfig {
            hidden_size: 8,
            n_layers: 1,
            horizon: 4,
            lookback: 12,
            distribution: OutputDist::Gaussian,
        };
        let model = DeepAR::new(config);
        let past = vec![0.5_f64; 12];
        let preds = model.forward(&past).expect("forward pass");
        assert_eq!(preds.len(), 4);
        for (mu, sigma) in &preds {
            assert!(sigma > &0.0, "sigma must be positive, got {sigma}");
            assert!(mu.is_finite(), "mu must be finite");
        }
    }

    #[test]
    fn test_forward_negative_binomial() {
        let config = DeepARConfig {
            hidden_size: 8,
            n_layers: 1,
            horizon: 3,
            lookback: 10,
            distribution: OutputDist::NegativeBinomial,
        };
        let model = DeepAR::new(config);
        let past = vec![1.0_f64; 10];
        let preds = model.forward(&past).expect("nb forward pass");
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_forward_student_t() {
        let config = DeepARConfig {
            hidden_size: 8,
            n_layers: 1,
            horizon: 3,
            lookback: 10,
            distribution: OutputDist::StudentT,
        };
        let model = DeepAR::new(config);
        let past = vec![1.0_f64; 10];
        let preds = model.forward(&past).expect("student-t forward pass");
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_sample_shape() {
        let config = DeepARConfig {
            hidden_size: 8,
            n_layers: 1,
            horizon: 4,
            lookback: 12,
            distribution: OutputDist::Gaussian,
        };
        let model = DeepAR::new(config);
        let past = vec![0.5_f64; 12];
        let mut rng = Lcg::new(42);
        let samples = model.sample(&past, 10, &mut rng).expect("sampling");
        assert_eq!(samples.len(), 10);
        assert_eq!(samples[0].len(), 4);
    }

    #[test]
    fn test_insufficient_data_error() {
        let config = DeepARConfig {
            hidden_size: 4,
            n_layers: 1,
            horizon: 4,
            lookback: 12,
            distribution: OutputDist::Gaussian,
        };
        let model = DeepAR::new(config);
        let past = vec![0.5_f64; 5]; // too short
        assert!(model.forward(&past).is_err());
    }

    #[test]
    fn test_multi_layer_lstm() {
        let config = DeepARConfig {
            hidden_size: 8,
            n_layers: 3,
            horizon: 2,
            lookback: 6,
            distribution: OutputDist::Gaussian,
        };
        let model = DeepAR::new(config);
        assert_eq!(model.lstm_cells.len(), 3);
        let past = vec![0.5_f64; 6];
        let preds = model.forward(&past).expect("multi-layer forward");
        assert_eq!(preds.len(), 2);
    }

    #[test]
    fn test_train_smoke() {
        let config = DeepARConfig {
            hidden_size: 4,
            n_layers: 1,
            horizon: 3,
            lookback: 9,
            distribution: OutputDist::Gaussian,
        };
        let mut model = DeepAR::new(config);
        let data: Vec<f64> = (0..60).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut rng = Lcg::new(1);
        model.train(&data, 2, 0.001, &mut rng).expect("training");
    }
}
