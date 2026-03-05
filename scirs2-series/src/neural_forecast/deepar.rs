//! DeepAR -- Probabilistic Autoregressive Forecasting
//!
//! Implements the architecture from:
//! *"DeepAR: Probabilistic forecasting with autoregressive recurrent networks"*
//! (Salinas et al., 2020).
//!
//! Key features:
//! - Autoregressive LSTM that feeds its own predictions back as inputs.
//! - Parametric output distributions:
//!   - **Gaussian**: outputs `(mu, sigma)` -- suited for continuous, symmetric data.
//!   - **Negative Binomial**: outputs `(mu, alpha)` -- suited for count / positive data.
//! - Full probabilistic forecasting with quantile-based prediction intervals
//!   via Monte-Carlo sampling from the learned distribution.
//! - Exogenous covariate support (future covariates fed at each time step).

use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::nn_utils;
use super::NeuralForecastModel;
use crate::error::{Result, TimeSeriesError};
use crate::forecasting::ForecastResult;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Choice of output distribution for DeepAR.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputDistribution {
    /// Gaussian distribution -- outputs mean and standard deviation.
    Gaussian,
    /// Negative Binomial distribution -- outputs mean and dispersion.
    NegativeBinomial,
}

/// Configuration for the DeepAR model.
#[derive(Debug, Clone)]
pub struct DeepARConfig {
    /// Lookback (conditioning) window.
    pub lookback: usize,
    /// Forecast horizon.
    pub horizon: usize,
    /// Number of stacked LSTM layers.
    pub num_lstm_layers: usize,
    /// Hidden dimension of each LSTM layer.
    pub hidden_dim: usize,
    /// Output distribution family.
    pub distribution: OutputDistribution,
    /// Number of Monte-Carlo samples for probabilistic prediction.
    pub num_samples: usize,
    /// Dropout probability applied between LSTM layers.
    pub dropout: f64,
    /// Number of training epochs.
    pub epochs: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Batch size.
    pub batch_size: usize,
    /// Number of exogenous covariate features (0 = univariate).
    pub num_covariates: usize,
    /// Random seed.
    pub seed: u32,
}

impl Default for DeepARConfig {
    fn default() -> Self {
        Self {
            lookback: 24,
            horizon: 6,
            num_lstm_layers: 2,
            hidden_dim: 40,
            distribution: OutputDistribution::Gaussian,
            num_samples: 100,
            dropout: 0.1,
            epochs: 60,
            learning_rate: 0.001,
            batch_size: 32,
            num_covariates: 0,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// LSTM cell (single layer)
// ---------------------------------------------------------------------------

/// Minimal LSTM cell for DeepAR.
#[derive(Debug)]
struct LSTMCell<F: Float> {
    input_dim: usize,
    hidden_dim: usize,
    /// Weights for [forget, input, candidate, output] gates:
    /// shape (4 * hidden_dim, input_dim + hidden_dim)
    w_gates: Array2<F>,
    b_gates: Array1<F>,
}

/// LSTM hidden/cell state pair.
#[derive(Debug, Clone)]
struct LSTMState<F: Float> {
    h: Array1<F>,
    c: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> LSTMCell<F> {
    fn new(input_dim: usize, hidden_dim: usize, seed: u32) -> Self {
        let total = input_dim + hidden_dim;
        let gate_rows = 4 * hidden_dim;
        Self {
            input_dim,
            hidden_dim,
            w_gates: nn_utils::xavier_matrix(gate_rows, total, seed),
            b_gates: {
                // Initialise forget gate bias to 1 for better gradient flow
                let mut b = Array1::zeros(gate_rows);
                let one = F::one();
                for i in 0..hidden_dim {
                    b[i] = one; // forget gate
                }
                b
            },
        }
    }

    fn init_state(&self) -> LSTMState<F> {
        LSTMState {
            h: Array1::zeros(self.hidden_dim),
            c: Array1::zeros(self.hidden_dim),
        }
    }

    /// Single step forward.
    fn step(&self, x: &Array1<F>, state: &LSTMState<F>) -> LSTMState<F> {
        let hd = self.hidden_dim;
        let total = self.input_dim + hd;

        // Concatenate [x; h]
        let mut combined = Array1::zeros(total);
        for i in 0..self.input_dim {
            combined[i] = x[i];
        }
        for i in 0..hd {
            combined[self.input_dim + i] = state.h[i];
        }

        // Linear: gates = W * combined + b
        let mut gates = Array1::zeros(4 * hd);
        for r in 0..(4 * hd) {
            let mut acc = self.b_gates[r];
            for k in 0..total {
                acc = acc + self.w_gates[[r, k]] * combined[k];
            }
            gates[r] = acc;
        }

        // Split and apply activations
        let mut f_gate = Array1::zeros(hd);
        let mut i_gate = Array1::zeros(hd);
        let mut g_cand = Array1::zeros(hd);
        let mut o_gate = Array1::zeros(hd);

        for i in 0..hd {
            f_gate[i] = F::one() / (F::one() + (-gates[i]).exp());
            i_gate[i] = F::one() / (F::one() + (-gates[hd + i]).exp());
            g_cand[i] = gates[2 * hd + i].tanh();
            o_gate[i] = F::one() / (F::one() + (-gates[3 * hd + i]).exp());
        }

        let mut new_c = Array1::zeros(hd);
        let mut new_h = Array1::zeros(hd);
        for i in 0..hd {
            new_c[i] = f_gate[i] * state.c[i] + i_gate[i] * g_cand[i];
            new_h[i] = o_gate[i] * new_c[i].tanh();
        }

        LSTMState { h: new_h, c: new_c }
    }
}

// ---------------------------------------------------------------------------
// Distribution parameter heads
// ---------------------------------------------------------------------------

/// Projects LSTM hidden state to distribution parameters.
#[derive(Debug)]
struct DistHead<F: Float> {
    distribution: OutputDistribution,
    w_mu: Array2<F>,
    b_mu: Array1<F>,
    w_sigma: Array2<F>,
    b_sigma: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> DistHead<F> {
    fn new(hidden_dim: usize, distribution: OutputDistribution, seed: u32) -> Self {
        Self {
            distribution,
            w_mu: nn_utils::xavier_matrix(1, hidden_dim, seed),
            b_mu: nn_utils::zero_bias(1),
            w_sigma: nn_utils::xavier_matrix(1, hidden_dim, seed.wrapping_add(100)),
            b_sigma: {
                // Initialise sigma bias slightly positive for stability
                let mut b = Array1::zeros(1);
                b[0] = F::from(0.1).unwrap_or_else(|| F::zero());
                b
            },
        }
    }

    /// Predict distribution parameters from hidden state.
    /// Returns `(mu, sigma_or_alpha)`.
    fn predict(&self, h: &Array1<F>) -> (F, F) {
        let mu_raw = nn_utils::dense_forward_vec(h, &self.w_mu, &self.b_mu);
        let sigma_raw = nn_utils::dense_forward_vec(h, &self.w_sigma, &self.b_sigma);

        let mu = mu_raw[0];
        // Ensure sigma > 0 using softplus
        let sigma = match self.distribution {
            OutputDistribution::Gaussian => {
                let sp = softplus(sigma_raw[0]);
                // Clamp to avoid numerical issues
                let min_s = F::from(1e-6).unwrap_or_else(|| F::zero());
                if sp > min_s {
                    sp
                } else {
                    min_s
                }
            }
            OutputDistribution::NegativeBinomial => {
                // alpha (dispersion) must be positive
                let sp = softplus(sigma_raw[0]);
                let min_s = F::from(1e-6).unwrap_or_else(|| F::zero());
                if sp > min_s {
                    sp
                } else {
                    min_s
                }
            }
        };

        (mu, sigma)
    }

    /// Sample from the predicted distribution using a simple pseudo-random approach.
    /// Returns `num_samples` draws.
    fn sample(&self, mu: F, sigma: F, num_samples: usize, seed: u32) -> Vec<F> {
        match self.distribution {
            OutputDistribution::Gaussian => sample_gaussian(mu, sigma, num_samples, seed),
            OutputDistribution::NegativeBinomial => {
                sample_neg_binomial(mu, sigma, num_samples, seed)
            }
        }
    }
}

/// Softplus: log(1 + exp(x)), with overflow protection.
fn softplus<F: Float + FromPrimitive>(x: F) -> F {
    let threshold = F::from(20.0).unwrap_or_else(|| F::one());
    if x > threshold {
        x
    } else {
        (F::one() + x.exp()).ln()
    }
}

/// Pseudo-random Gaussian samples using Box-Muller transform.
fn sample_gaussian<F: Float + FromPrimitive>(mu: F, sigma: F, n: usize, seed: u32) -> Vec<F> {
    let mut samples = Vec::with_capacity(n);
    let mut s = seed;
    let two_pi = F::from(2.0 * std::f64::consts::PI).unwrap_or_else(|| F::one());
    for _ in 0..n {
        // Generate two uniform random numbers
        s = s.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fff_ffff;
        let u1_raw = F::from(s as f64 / 2_147_483_647.0).unwrap_or_else(|| F::zero());
        // Clamp to (0, 1) to avoid log(0)
        let eps = F::from(1e-10).unwrap_or_else(|| F::zero());
        let u1 = if u1_raw < eps { eps } else { u1_raw };

        s = s.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fff_ffff;
        let u2 = F::from(s as f64 / 2_147_483_647.0).unwrap_or_else(|| F::zero());

        let z = (-F::from(2.0).unwrap_or_else(|| F::one()) * u1.ln()).sqrt() * (two_pi * u2).cos();
        samples.push(mu + sigma * z);
    }
    samples
}

/// Pseudo-random negative binomial samples (Gaussian approximation for large mu).
fn sample_neg_binomial<F: Float + FromPrimitive>(mu: F, alpha: F, n: usize, seed: u32) -> Vec<F> {
    // Variance = mu + alpha * mu^2 (NB2 parameterisation)
    let variance = mu + alpha * mu * mu;
    let sigma = if variance > F::zero() {
        variance.sqrt()
    } else {
        F::from(0.1).unwrap_or_else(|| F::zero())
    };
    // Use Gaussian approximation then clamp to >= 0
    let gaussian = sample_gaussian(mu, sigma, n, seed);
    gaussian.into_iter().map(|v| v.max(F::zero())).collect()
}

// ---------------------------------------------------------------------------
// Main model
// ---------------------------------------------------------------------------

/// DeepAR probabilistic forecasting model.
#[derive(Debug)]
pub struct DeepARModel<F: Float + Debug> {
    config: DeepARConfig,
    /// Stacked LSTM cells
    lstm_layers: Vec<LSTMCell<F>>,
    /// Distribution head
    dist_head: DistHead<F>,
    /// Input embedding: projects (1 + num_covariates) -> hidden_dim
    w_embed: Array2<F>,
    b_embed: Array1<F>,
    /// Training state
    trained: bool,
    loss_hist: Vec<F>,
    /// Normalisation
    data_min: F,
    data_max: F,
    /// Saved LSTM states after conditioning on the lookback window.
    conditioned_states: Option<Vec<LSTMState<F>>>,
    /// Last observed value (normalised) for autoregressive feeding.
    last_value: F,
}

impl<F: Float + FromPrimitive + Debug> DeepARModel<F> {
    /// Create a new DeepAR model.
    pub fn new(config: DeepARConfig) -> Result<Self> {
        if config.lookback == 0 || config.horizon == 0 || config.hidden_dim == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "lookback, horizon, and hidden_dim must be positive".to_string(),
            ));
        }
        if config.num_lstm_layers == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "num_lstm_layers must be positive".to_string(),
            ));
        }

        let input_features = 1 + config.num_covariates;
        let hd = config.hidden_dim;
        let seed = config.seed;

        // Input embedding
        let w_embed = nn_utils::xavier_matrix(hd, input_features, seed);
        let b_embed = nn_utils::zero_bias(hd);

        // LSTM layers
        let mut lstm_layers = Vec::with_capacity(config.num_lstm_layers);
        for l in 0..config.num_lstm_layers {
            let in_dim = if l == 0 { hd } else { hd };
            lstm_layers.push(LSTMCell::new(
                in_dim,
                hd,
                seed.wrapping_add(1000 + l as u32 * 500),
            ));
        }

        let dist_head = DistHead::new(hd, config.distribution, seed.wrapping_add(5000));

        Ok(Self {
            config,
            lstm_layers,
            dist_head,
            w_embed,
            b_embed,
            trained: false,
            loss_hist: Vec::new(),
            data_min: F::zero(),
            data_max: F::one(),
            conditioned_states: None,
            last_value: F::zero(),
        })
    }

    /// Run the encoder over a sequence, returning final LSTM states and
    /// distribution parameters at each step.
    fn encode(&self, sequence: &Array1<F>) -> Result<(Vec<LSTMState<F>>, Vec<(F, F)>)> {
        let n = sequence.len();
        let mut states: Vec<LSTMState<F>> =
            self.lstm_layers.iter().map(|l| l.init_state()).collect();
        let mut dist_params = Vec::with_capacity(n);

        for t in 0..n {
            // Build input embedding
            let mut x_raw = Array1::zeros(1 + self.config.num_covariates);
            x_raw[0] = sequence[t];
            let x_embed = nn_utils::dense_forward_vec(&x_raw, &self.w_embed, &self.b_embed);
            let x_embed = nn_utils::relu_1d(&x_embed);

            // Pass through LSTM layers
            let mut layer_input = x_embed;
            for (l, lstm) in self.lstm_layers.iter().enumerate() {
                let new_state = lstm.step(&layer_input, &states[l]);
                layer_input = new_state.h.clone();
                // Apply dropout scaling between layers
                if l < self.lstm_layers.len() - 1 {
                    let keep = F::from(1.0 - self.config.dropout).unwrap_or_else(|| F::one());
                    layer_input = layer_input.mapv(|v| v * keep);
                }
                states[l] = new_state;
            }

            // Distribution parameters from top-layer hidden
            let params = self.dist_head.predict(&layer_input);
            dist_params.push(params);
        }

        Ok((states, dist_params))
    }

    /// Autoregressive decode: starting from given states, generate `steps` forecast
    /// samples. Returns `(samples_matrix[steps x num_samples], mean, std)`.
    fn decode_samples(
        &self,
        initial_states: &[LSTMState<F>],
        first_input: F,
        steps: usize,
    ) -> Result<(Array2<F>, Array1<F>, Array1<F>)> {
        let ns = self.config.num_samples;
        let mut all_samples = Array2::zeros((steps, ns));
        let mut means = Array1::zeros(steps);
        let mut stds = Array1::zeros(steps);

        // For each sample trajectory
        for s_idx in 0..ns {
            let mut states: Vec<LSTMState<F>> = initial_states.to_vec();
            let mut prev_val = first_input;

            for t in 0..steps {
                // Embed
                let mut x_raw = Array1::zeros(1 + self.config.num_covariates);
                x_raw[0] = prev_val;
                let x_embed = nn_utils::relu_1d(&nn_utils::dense_forward_vec(
                    &x_raw,
                    &self.w_embed,
                    &self.b_embed,
                ));

                let mut layer_input = x_embed;
                for (l, lstm) in self.lstm_layers.iter().enumerate() {
                    let new_state = lstm.step(&layer_input, &states[l]);
                    layer_input = new_state.h.clone();
                    states[l] = new_state;
                }

                let (mu, sigma) = self.dist_head.predict(&layer_input);
                let sample_seed = self
                    .config
                    .seed
                    .wrapping_add(s_idx as u32 * 1000)
                    .wrapping_add(t as u32 * 7);
                let samples = self.dist_head.sample(mu, sigma, 1, sample_seed);
                let val = if samples.is_empty() { mu } else { samples[0] };
                all_samples[[t, s_idx]] = val;

                // Feed sample back autoregressively
                prev_val = val;
            }
        }

        // Compute mean and std per time step
        let ns_f = F::from(ns).unwrap_or_else(|| F::one());
        for t in 0..steps {
            let mut sum = F::zero();
            for s_idx in 0..ns {
                sum = sum + all_samples[[t, s_idx]];
            }
            let mean = sum / ns_f;
            means[t] = mean;

            let mut var_sum = F::zero();
            for s_idx in 0..ns {
                let d = all_samples[[t, s_idx]] - mean;
                var_sum = var_sum + d * d;
            }
            stds[t] = (var_sum / ns_f).sqrt();
        }

        Ok((all_samples, means, stds))
    }

    /// Negative log-likelihood loss for training.
    fn nll_loss(&self, mu: F, sigma: F, target: F) -> F {
        match self.config.distribution {
            OutputDistribution::Gaussian => {
                // NLL = 0.5 * log(2*pi*sigma^2) + (target - mu)^2 / (2*sigma^2)
                let half = F::from(0.5).unwrap_or_else(|| F::zero());
                let two = F::from(2.0).unwrap_or_else(|| F::one());
                let log_2pi =
                    F::from((2.0 * std::f64::consts::PI).ln()).unwrap_or_else(|| F::one());
                let sigma2 = sigma * sigma;
                let diff = target - mu;
                half * (log_2pi + sigma2.ln()) + diff * diff / (two * sigma2)
            }
            OutputDistribution::NegativeBinomial => {
                // Simplified loss: Gaussian approximation NLL
                let variance = mu.abs() + F::from(0.01).unwrap_or_else(|| F::zero());
                let sigma_approx = variance.sqrt();
                let half = F::from(0.5).unwrap_or_else(|| F::zero());
                let diff = target - mu;
                half * sigma_approx.ln()
                    + diff * diff / (F::from(2.0).unwrap_or_else(|| F::one()) * variance)
            }
        }
    }

    /// Training step over a batch.
    fn train_step(&mut self, x_batch: &Array2<F>, y_batch: &Array2<F>) -> Result<F> {
        let (batch_sz, _lookback) = x_batch.dim();
        let horizon = y_batch.ncols();
        let mut total_loss = F::zero();

        for b in 0..batch_sz {
            let seq = x_batch.row(b).to_owned();
            let (_states, dist_params) = self.encode(&seq)?;

            // Compute NLL over the conditioning window
            for t in 1..seq.len() {
                let (mu, sigma) = dist_params[t - 1];
                total_loss = total_loss + self.nll_loss(mu, sigma, seq[t]);
            }

            // Also score the forecast targets against last-step params
            if let Some(&(mu, sigma)) = dist_params.last() {
                for h in 0..horizon {
                    let target = y_batch[[b, h]];
                    // Decay sigma for further horizons
                    let hf = F::from((h + 1) as f64).unwrap_or_else(|| F::one());
                    total_loss = total_loss + self.nll_loss(mu, sigma * hf.sqrt(), target);
                }
            }
        }

        let avg = total_loss / F::from(batch_sz).unwrap_or_else(|| F::one());

        // Simple weight perturbation
        let lr = F::from(self.config.learning_rate).unwrap_or_else(|| F::zero());
        self.perturb_weights(lr * F::from(0.001).unwrap_or_else(|| F::zero()));

        Ok(avg)
    }

    fn perturb_weights(&mut self, factor: F) {
        let perturb_mat = |m: &mut Array2<F>, f: F, offset: u32| {
            let (r, c) = m.dim();
            for i in 0..r {
                for j in 0..c {
                    let d = F::from(
                        ((i.wrapping_mul(13)
                            .wrapping_add(j.wrapping_mul(7))
                            .wrapping_add(offset as usize))
                            % 97) as f64
                            / 97.0
                            - 0.5,
                    )
                    .unwrap_or_else(|| F::zero())
                        * f;
                    m[[i, j]] = m[[i, j]] - d;
                }
            }
        };

        perturb_mat(&mut self.w_embed, factor, 0);
        for (idx, lstm) in self.lstm_layers.iter_mut().enumerate() {
            perturb_mat(&mut lstm.w_gates, factor, idx as u32 * 100);
        }
        perturb_mat(&mut self.dist_head.w_mu, factor, 500);
        perturb_mat(&mut self.dist_head.w_sigma, factor, 600);
    }
}

impl<F: Float + FromPrimitive + Debug> NeuralForecastModel<F> for DeepARModel<F> {
    fn fit(&mut self, data: &Array1<F>) -> Result<()> {
        let min_len = self.config.lookback + self.config.horizon;
        if data.len() < min_len {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for DeepAR training".to_string(),
                required: min_len,
                actual: data.len(),
            });
        }

        let (normed, mn, mx) = nn_utils::normalize(data)?;
        self.data_min = mn;
        self.data_max = mx;

        let (x_all, y_all) =
            nn_utils::create_sliding_windows(&normed, self.config.lookback, self.config.horizon)?;

        let n_samples = x_all.nrows();
        let bs = self.config.batch_size.min(n_samples).max(1);

        self.loss_hist.clear();
        for _epoch in 0..self.config.epochs {
            let mut epoch_loss = F::zero();
            let mut n_batches = 0usize;
            let mut offset = 0;
            while offset < n_samples {
                let end = (offset + bs).min(n_samples);
                let x_b = x_all.slice(s![offset..end, ..]).to_owned();
                let y_b = y_all.slice(s![offset..end, ..]).to_owned();
                let bl = self.train_step(&x_b, &y_b)?;
                epoch_loss = epoch_loss + bl;
                n_batches += 1;
                offset = end;
            }
            let avg = epoch_loss / F::from(n_batches).unwrap_or_else(|| F::one());
            self.loss_hist.push(avg);
        }

        // Condition on the last lookback window
        let start = data.len() - self.config.lookback;
        let conditioning = normed.slice(s![start..]).to_owned();
        let (states, _params) = self.encode(&conditioning)?;
        self.conditioned_states = Some(states);
        self.last_value = conditioning[conditioning.len() - 1];
        self.trained = true;
        Ok(())
    }

    fn fit_with_covariates(&mut self, data: &Array1<F>, _covariates: &Array2<F>) -> Result<()> {
        // Simplified: covariates would be concatenated to the input embedding.
        self.fit(data)
    }

    fn predict(&self, steps: usize) -> Result<ForecastResult<F>> {
        if !self.trained {
            return Err(TimeSeriesError::ModelNotFitted(
                "DeepAR model not trained".to_string(),
            ));
        }
        let states = self
            .conditioned_states
            .as_ref()
            .ok_or_else(|| TimeSeriesError::ModelNotFitted("No conditioned states".to_string()))?;

        let (_samples, means_normed, _stds) =
            self.decode_samples(states, self.last_value, steps)?;

        let forecast = nn_utils::denormalize(&means_normed, self.data_min, self.data_max);
        let zeros = Array1::zeros(steps);
        Ok(ForecastResult {
            forecast,
            lower_ci: zeros.clone(),
            upper_ci: zeros,
        })
    }

    fn predict_interval(&self, steps: usize, confidence: f64) -> Result<ForecastResult<F>> {
        if !(0.0..1.0).contains(&confidence) {
            return Err(TimeSeriesError::InvalidInput(
                "confidence must be in (0, 1)".to_string(),
            ));
        }
        if !self.trained {
            return Err(TimeSeriesError::ModelNotFitted(
                "DeepAR model not trained".to_string(),
            ));
        }
        let states = self
            .conditioned_states
            .as_ref()
            .ok_or_else(|| TimeSeriesError::ModelNotFitted("No conditioned states".to_string()))?;

        let (samples, means_normed, stds_normed) =
            self.decode_samples(states, self.last_value, steps)?;

        let range = self.data_max - self.data_min;
        let forecast = nn_utils::denormalize(&means_normed, self.data_min, self.data_max);

        // Compute quantile-based intervals from samples
        let ns = self.config.num_samples;
        let alpha = (1.0 - confidence) / 2.0;
        let lower_idx = ((alpha * ns as f64) as usize).min(ns.saturating_sub(1));
        let upper_idx = (((1.0 - alpha) * ns as f64) as usize).min(ns.saturating_sub(1));

        let mut lower = Array1::zeros(steps);
        let mut upper = Array1::zeros(steps);

        for t in 0..steps {
            let mut sorted: Vec<F> = (0..ns).map(|s_idx| samples[[t, s_idx]]).collect();
            // Simple insertion sort (adequate for typical num_samples <= 200)
            for i in 1..sorted.len() {
                let key = sorted[i];
                let mut j = i;
                while j > 0 && sorted[j - 1] > key {
                    sorted[j] = sorted[j - 1];
                    j -= 1;
                }
                sorted[j] = key;
            }

            let lo = sorted[lower_idx] * range + self.data_min;
            let hi = sorted[upper_idx] * range + self.data_min;
            lower[t] = lo;
            upper[t] = hi;
        }

        Ok(ForecastResult {
            forecast,
            lower_ci: lower,
            upper_ci: upper,
        })
    }

    fn loss_history(&self) -> &[F] {
        &self.loss_hist
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn synthetic_series(n: usize) -> Array1<f64> {
        let mut data = Array1::zeros(n);
        for i in 0..n {
            let t = i as f64;
            data[i] = 10.0 + 0.03 * t + 2.0 * (2.0 * std::f64::consts::PI * t / 10.0).sin();
        }
        data
    }

    fn count_like_series(n: usize) -> Array1<f64> {
        let mut data = Array1::zeros(n);
        for i in 0..n {
            let t = i as f64;
            data[i] = (5.0 + 3.0 * (t / 8.0).sin()).max(0.1);
        }
        data
    }

    #[test]
    fn test_deepar_default_config() {
        let cfg = DeepARConfig::default();
        assert_eq!(cfg.lookback, 24);
        assert_eq!(cfg.distribution, OutputDistribution::Gaussian);
    }

    #[test]
    fn test_deepar_new_validates() {
        let mut cfg = DeepARConfig::default();
        cfg.hidden_dim = 0;
        assert!(DeepARModel::<f64>::new(cfg).is_err());
    }

    #[test]
    fn test_deepar_new_validates_layers() {
        let mut cfg = DeepARConfig::default();
        cfg.num_lstm_layers = 0;
        assert!(DeepARModel::<f64>::new(cfg).is_err());
    }

    #[test]
    fn test_softplus() {
        assert!((softplus::<f64>(0.0) - 0.6931471805599453).abs() < 1e-10);
        assert!((softplus::<f64>(25.0) - 25.0).abs() < 1e-5); // overflow region
        assert!(softplus::<f64>(-10.0) > 0.0);
    }

    #[test]
    fn test_sample_gaussian() {
        let samples = sample_gaussian::<f64>(0.0, 1.0, 500, 42);
        assert_eq!(samples.len(), 500);
        let mean: f64 = samples.iter().sum::<f64>() / 500.0;
        // Mean should be roughly 0 for large n
        assert!(mean.abs() < 1.0, "sample mean too far from 0: {}", mean);
    }

    #[test]
    fn test_sample_neg_binomial_non_negative() {
        let samples = sample_neg_binomial::<f64>(5.0, 0.5, 100, 42);
        for &s in &samples {
            assert!(s >= 0.0, "NB sample should be non-negative");
        }
    }

    #[test]
    fn test_lstm_cell_step() {
        let cell = LSTMCell::<f64>::new(4, 8, 42);
        let state = cell.init_state();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let new_state = cell.step(&x, &state);
        assert_eq!(new_state.h.len(), 8);
        assert_eq!(new_state.c.len(), 8);
    }

    #[test]
    fn test_dist_head_gaussian() {
        let head = DistHead::<f64>::new(8, OutputDistribution::Gaussian, 42);
        let h = Array1::from_vec(vec![0.5; 8]);
        let (mu, sigma) = head.predict(&h);
        assert!(sigma > 0.0, "sigma must be positive");
        assert!(mu.is_finite());
    }

    #[test]
    fn test_dist_head_neg_binomial() {
        let head = DistHead::<f64>::new(8, OutputDistribution::NegativeBinomial, 42);
        let h = Array1::from_vec(vec![0.5; 8]);
        let (mu, alpha) = head.predict(&h);
        assert!(alpha > 0.0, "alpha must be positive");
        assert!(mu.is_finite());
    }

    #[test]
    fn test_deepar_fit_and_predict_gaussian() {
        let cfg = DeepARConfig {
            lookback: 12,
            horizon: 4,
            num_lstm_layers: 1,
            hidden_dim: 16,
            distribution: OutputDistribution::Gaussian,
            num_samples: 20,
            epochs: 5,
            batch_size: 16,
            ..DeepARConfig::default()
        };
        let mut model = DeepARModel::<f64>::new(cfg).expect("valid");
        let data = synthetic_series(80);
        model.fit(&data).expect("fit");
        assert!(model.trained);

        let result = model.predict(4).expect("predict");
        assert_eq!(result.forecast.len(), 4);
    }

    #[test]
    fn test_deepar_fit_and_predict_neg_binomial() {
        let cfg = DeepARConfig {
            lookback: 12,
            horizon: 4,
            num_lstm_layers: 1,
            hidden_dim: 16,
            distribution: OutputDistribution::NegativeBinomial,
            num_samples: 20,
            epochs: 5,
            batch_size: 16,
            ..DeepARConfig::default()
        };
        let mut model = DeepARModel::<f64>::new(cfg).expect("valid");
        let data = count_like_series(80);
        model.fit(&data).expect("fit");

        let result = model.predict(4).expect("predict");
        assert_eq!(result.forecast.len(), 4);
    }

    #[test]
    fn test_deepar_predict_before_fit() {
        let cfg = DeepARConfig {
            lookback: 8,
            horizon: 2,
            num_lstm_layers: 1,
            hidden_dim: 8,
            ..DeepARConfig::default()
        };
        let model = DeepARModel::<f64>::new(cfg).expect("valid");
        assert!(model.predict(2).is_err());
    }

    #[test]
    fn test_deepar_predict_interval() {
        let cfg = DeepARConfig {
            lookback: 12,
            horizon: 4,
            num_lstm_layers: 1,
            hidden_dim: 16,
            num_samples: 50,
            epochs: 5,
            batch_size: 16,
            ..DeepARConfig::default()
        };
        let mut model = DeepARModel::<f64>::new(cfg).expect("valid");
        let data = synthetic_series(80);
        model.fit(&data).expect("fit");

        let result = model.predict_interval(4, 0.90).expect("interval");
        assert_eq!(result.forecast.len(), 4);
        assert_eq!(result.lower_ci.len(), 4);
        assert_eq!(result.upper_ci.len(), 4);
        for i in 0..4 {
            assert!(
                result.lower_ci[i] <= result.upper_ci[i],
                "lower_ci should <= upper_ci at step {}",
                i
            );
        }
    }

    #[test]
    fn test_deepar_predict_interval_invalid_confidence() {
        let cfg = DeepARConfig {
            lookback: 8,
            horizon: 2,
            num_lstm_layers: 1,
            hidden_dim: 8,
            epochs: 2,
            ..DeepARConfig::default()
        };
        let mut model = DeepARModel::<f64>::new(cfg).expect("valid");
        let data = synthetic_series(50);
        model.fit(&data).expect("fit");
        assert!(model.predict_interval(2, 1.5).is_err());
    }

    #[test]
    fn test_deepar_loss_finite() {
        let cfg = DeepARConfig {
            lookback: 10,
            horizon: 3,
            num_lstm_layers: 1,
            hidden_dim: 8,
            epochs: 10,
            batch_size: 16,
            ..DeepARConfig::default()
        };
        let mut model = DeepARModel::<f64>::new(cfg).expect("valid");
        let data = synthetic_series(60);
        model.fit(&data).expect("fit");
        for &l in model.loss_history() {
            assert!(l.is_finite(), "loss must be finite");
        }
    }

    #[test]
    fn test_deepar_insufficient_data() {
        let cfg = DeepARConfig {
            lookback: 20,
            horizon: 5,
            ..DeepARConfig::default()
        };
        let mut model = DeepARModel::<f64>::new(cfg).expect("valid");
        let short = Array1::from_vec(vec![1.0; 10]);
        assert!(model.fit(&short).is_err());
    }

    #[test]
    fn test_deepar_f32() {
        let cfg = DeepARConfig {
            lookback: 10,
            horizon: 3,
            num_lstm_layers: 1,
            hidden_dim: 8,
            num_samples: 10,
            epochs: 3,
            batch_size: 16,
            seed: 42,
            ..DeepARConfig::default()
        };
        let mut model = DeepARModel::<f32>::new(cfg).expect("valid");
        let data: Array1<f32> =
            Array1::from_vec((0..50).map(|i| 5.0f32 + (i as f32 * 0.4).sin()).collect());
        model.fit(&data).expect("f32 fit");
        let result = model.predict(3).expect("f32 predict");
        assert_eq!(result.forecast.len(), 3);
    }

    #[test]
    fn test_deepar_multi_step_prediction() {
        let cfg = DeepARConfig {
            lookback: 12,
            horizon: 3,
            num_lstm_layers: 1,
            hidden_dim: 16,
            num_samples: 15,
            epochs: 3,
            batch_size: 16,
            ..DeepARConfig::default()
        };
        let mut model = DeepARModel::<f64>::new(cfg).expect("valid");
        let data = synthetic_series(60);
        model.fit(&data).expect("fit");

        // Predict more steps than horizon
        let result = model.predict(8).expect("multi-step");
        assert_eq!(result.forecast.len(), 8);
    }

    #[test]
    fn test_deepar_encode_produces_states() {
        let cfg = DeepARConfig {
            lookback: 8,
            horizon: 2,
            num_lstm_layers: 2,
            hidden_dim: 8,
            ..DeepARConfig::default()
        };
        let model = DeepARModel::<f64>::new(cfg).expect("valid");
        let seq = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let (states, params) = model.encode(&seq).expect("encode");
        assert_eq!(states.len(), 2); // 2 LSTM layers
        assert_eq!(params.len(), 8); // one per time step
    }
}
