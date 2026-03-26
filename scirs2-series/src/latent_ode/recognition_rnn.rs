//! Reverse-time GRU encoder for the Latent ODE model.
//!
//! Processes observations `{(t_i, x_i)}` in **reverse** time order and
//! outputs the mean `μ` and log-variance `log σ²` of the approximate
//! posterior `q(z₀ | observations)`.

/// A single GRU cell operating on raw `f64` slices.
///
/// Equations (standard GRU):
/// ```text
/// z  = σ(W_z · [h, x] + b_z)      (update gate)
/// r  = σ(W_r · [h, x] + b_r)      (reset gate)
/// h̃  = tanh(W_h · [r⊙h, x] + b_h)  (candidate hidden)
/// h' = (1 - z) ⊙ h + z ⊙ h̃
/// ```
#[derive(Debug, Clone)]
pub struct GruCell {
    hidden_dim: usize,
    input_dim: usize,
    // Update gate parameters
    w_z: Vec<f64>, // (hidden_dim) × (hidden_dim + input_dim)
    b_z: Vec<f64>, // hidden_dim
    // Reset gate parameters
    w_r: Vec<f64>,
    b_r: Vec<f64>,
    // Candidate hidden state parameters
    w_h: Vec<f64>,
    b_h: Vec<f64>,
}

impl GruCell {
    /// Create a new GRU cell.
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let combined = hidden_dim + input_dim;
        let scale = (1.0_f64 / combined as f64).sqrt();
        let init_weights = |size: usize| -> Vec<f64> {
            (0..size)
                .map(|k| {
                    let v = ((k as f64 * 2.3999632) % 2.0) - 1.0;
                    v * scale
                })
                .collect()
        };
        Self {
            hidden_dim,
            input_dim,
            w_z: init_weights(hidden_dim * combined),
            b_z: vec![0.0; hidden_dim],
            w_r: init_weights(hidden_dim * combined),
            b_r: vec![0.0; hidden_dim],
            w_h: init_weights(hidden_dim * combined),
            b_h: vec![0.0; hidden_dim],
        }
    }

    /// Sigmoid of a scalar.
    #[inline]
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Linear projection: `W [h; x] + b`.
    fn linear(&self, w: &[f64], b: &[f64], h: &[f64], x: &[f64]) -> Vec<f64> {
        let combined = self.hidden_dim + self.input_dim;
        let mut out = b.to_vec();
        for (i, oi) in out.iter_mut().enumerate() {
            for (j, &hj) in h.iter().enumerate() {
                *oi += w[i * combined + j] * hj;
            }
            for (j, &xj) in x.iter().enumerate() {
                *oi += w[i * combined + self.hidden_dim + j] * xj;
            }
        }
        out
    }

    /// One GRU step: returns the new hidden state.
    pub fn step(&self, h: &[f64], x: &[f64]) -> Vec<f64> {
        // Update gate
        let z_pre = self.linear(&self.w_z, &self.b_z, h, x);
        let z: Vec<f64> = z_pre.iter().map(|&v| Self::sigmoid(v)).collect();

        // Reset gate
        let r_pre = self.linear(&self.w_r, &self.b_r, h, x);
        let r: Vec<f64> = r_pre.iter().map(|&v| Self::sigmoid(v)).collect();

        // Candidate: use r ⊙ h as the hidden input
        let rh: Vec<f64> = r.iter().zip(h.iter()).map(|(&ri, &hi)| ri * hi).collect();
        let h_cand_pre = self.linear(&self.w_h, &self.b_h, &rh, x);
        let h_cand: Vec<f64> = h_cand_pre.iter().map(|&v| v.tanh()).collect();

        // New hidden: (1 - z) ⊙ h + z ⊙ h̃
        z.iter()
            .enumerate()
            .map(|(i, &zi)| (1.0 - zi) * h[i] + zi * h_cand[i])
            .collect()
    }

    /// Immutable access to hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
}

/// Output projection that maps from `hidden_dim` to `2 * latent_dim`
/// (mean and log-variance of `q(z₀)`).
#[derive(Debug, Clone)]
pub struct OutputProjection {
    w: Vec<f64>,
    b: Vec<f64>,
    in_dim: usize,
    out_dim: usize,
}

impl OutputProjection {
    /// Create a new linear output projection.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let scale = (1.0_f64 / in_dim as f64).sqrt();
        let w = (0..in_dim * out_dim)
            .map(|k| {
                let v = ((k as f64 * 1.41421356) % 2.0) - 1.0;
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

    /// Forward pass.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut y = self.b.clone();
        for (i, yi) in y.iter_mut().enumerate() {
            for (j, &xj) in x.iter().enumerate() {
                *yi += self.w[i * self.in_dim + j] * xj;
            }
        }
        y
    }
}

/// Reverse-time GRU encoder.
///
/// Reads `(time, observation)` pairs from *last to first* and outputs
/// `(μ, log σ)` for the approximate posterior over the initial latent state.
#[derive(Debug, Clone)]
pub struct RecognitionRnn {
    gru: GruCell,
    proj: OutputProjection,
    hidden_dim: usize,
    latent_dim: usize,
}

impl RecognitionRnn {
    /// Create a new recognition RNN.
    ///
    /// - `input_dim`: observation dimensionality + 1 (for time delta).
    /// - `hidden_dim`: GRU hidden size.
    /// - `latent_dim`: size of the latent space.
    pub fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize) -> Self {
        let gru = GruCell::new(input_dim, hidden_dim);
        let proj = OutputProjection::new(hidden_dim, 2 * latent_dim);
        Self {
            gru,
            proj,
            hidden_dim,
            latent_dim,
        }
    }

    /// Encode a sequence of `(time, observation)` pairs.
    ///
    /// Returns `(mu, log_sigma)` each of length `latent_dim`.
    ///
    /// Observations are processed in **reverse** chronological order.
    /// The time delta to the *next* (earlier) observation is appended to each
    /// input vector.
    pub fn encode(&self, obs: &[(f64, Vec<f64>)]) -> (Vec<f64>, Vec<f64>) {
        if obs.is_empty() {
            return (vec![0.0; self.latent_dim], vec![-2.0; self.latent_dim]);
        }

        let mut h = vec![0.0_f64; self.hidden_dim];

        // Iterate in reverse order
        let n = obs.len();
        for i in (0..n).rev() {
            let (t_cur, ref x) = obs[i];
            let dt = if i > 0 { t_cur - obs[i - 1].0 } else { 0.0 };

            // Concatenate observation and time delta
            let mut inp = x.clone();
            inp.push(dt);

            h = self.gru.step(&h, &inp);
        }

        // Project final hidden state to posterior parameters
        let out = self.proj.forward(&h);
        let mu: Vec<f64> = out[..self.latent_dim].to_vec();
        let log_sigma: Vec<f64> = out[self.latent_dim..].to_vec();

        (mu, log_sigma)
    }

    /// Mutable access to the GRU cell for weight updates.
    pub fn gru_mut(&mut self) -> &mut GruCell {
        &mut self.gru
    }

    /// Mutable access to the output projection for weight updates.
    pub fn proj_mut(&mut self) -> &mut OutputProjection {
        &mut self.proj
    }

    /// Latent dimension.
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gru_cell_step_shape() {
        let cell = GruCell::new(3, 8);
        let h = vec![0.0; 8];
        let x = vec![1.0, -0.5, 0.2];
        let h_new = cell.step(&h, &x);
        assert_eq!(h_new.len(), 8);
    }

    #[test]
    fn recognition_encode_returns_correct_dims() {
        let rnn = RecognitionRnn::new(4, 16, 8); // obs_dim=3, +1 for dt
        let obs = vec![
            (0.0, vec![1.0, 2.0, 3.0]),
            (0.5, vec![1.1, 1.9, 2.8]),
            (1.0, vec![1.2, 1.8, 2.6]),
        ];
        let (mu, log_sigma) = rnn.encode(&obs);
        assert_eq!(mu.len(), 8);
        assert_eq!(log_sigma.len(), 8);
    }

    #[test]
    fn recognition_encode_empty_returns_defaults() {
        let rnn = RecognitionRnn::new(4, 16, 8);
        let (mu, log_sigma) = rnn.encode(&[]);
        assert_eq!(mu.len(), 8);
        assert_eq!(log_sigma.len(), 8);
        for &m in &mu {
            assert_eq!(m, 0.0);
        }
    }
}
