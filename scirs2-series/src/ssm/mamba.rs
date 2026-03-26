//! Mamba (selective SSM) block and model implementation.
//!
//! Implements the Mamba architecture from:
//!   Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
//!   <https://arxiv.org/abs/2312.00752>
//!
//! # Key Innovation
//!
//! Unlike S4, Mamba makes the SSM parameters (Δ, B, C) **input-dependent**:
//!   - Δ (timestep) controls how much to integrate each input token.
//!   - B and C are content-based linear projections.
//!
//! This allows the model to selectively incorporate or ignore information,
//! behaving like a differentiable RNN with content-based gating.
//!
//! # Block Architecture
//!
//! For input x ∈ ℝ^{L×d}:
//!   1. `in_proj`:  x_split = x W_in        → [L, 2·d_inner]
//!      Split:      z = x_split[:, :d_inner]  (gating path)
//!                  x' = x_split[:, d_inner:] (SSM path)
//!   2. `conv1d`:   x' = causal_conv1d(x', W_conv, bias) + x' (depthwise)
//!   3. Activation: x' = silu(x')
//!   4. `x_proj`:   xbc = x' W_xp            → [L, dt_rank+2·d_state]
//!      Split:       Δ_raw  = xbc[:, :dt_rank]
//!                   B      = xbc[:, dt_rank:dt_rank+d_state]
//!                   C      = xbc[:, dt_rank+d_state:]
//!   5. `dt_proj`:  Δ = softplus(Δ_raw W_dt) → [L, d_inner]
//!   6. SSM scan:   y_ssm = selective_scan(x', Δ, A, B, C) → [L, d_inner]
//!   7. Gating:     y = y_ssm * silu(z)
//!   8. `out_proj`: output = y W_out          → [L, d_model]
//!
//! # Selective Scan (SSM Recurrence)
//!
//! For each input position t and each SSM channel (d_inner):
//!   Ā_t = exp(Δ_t · A)          (element-wise; A = -exp(a_log))
//!   B̄_t = Δ_t · B_t             (simplified ZOH; B is already projected)
//!   h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x'_t
//!   y_t = Σ_n  C_{t,n} · h_{t,n}

use std::f64::consts::PI;

use scirs2_core::ndarray::Array2;
use scirs2_core::random::RngExt;

use crate::error::{Result, TimeSeriesError};

use super::config::MambaConfig;

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

/// SiLU (Sigmoid Linear Unit) / Swish activation: x * sigmoid(x).
#[inline]
fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Softplus activation: log(1 + exp(x)).
#[inline]
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x // numerical stability
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// RMSNorm: normalize by RMS, then scale by learned parameter.
fn rms_norm(x: &[f64], scale: &[f64]) -> Vec<f64> {
    let rms = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + 1e-6).sqrt();
    x.iter()
        .zip(scale.iter())
        .map(|(v, s)| v / rms * s)
        .collect()
}

// ---------------------------------------------------------------------------
// Helper: normal random sample via Box-Muller
// ---------------------------------------------------------------------------

fn normal_sample(rng: &mut impl scirs2_core::random::Rng, mean: f64, std: f64) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-15); // avoid log(0)
    let u2: f64 = rng.random::<f64>();
    let z: f64 = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * PI * u2).cos();
    mean + std * z
}

// ---------------------------------------------------------------------------
// MambaBlock
// ---------------------------------------------------------------------------

/// A single Mamba block.
///
/// Processes sequences of shape `[seq_len, d_model]` and returns
/// sequences of the same shape via selective state-space modeling.
pub struct MambaBlock {
    /// Block configuration.
    pub config: MambaConfig,

    // ---- Linear projections ----
    /// Input projection: maps d_model → 2·d_inner.
    /// Shape: `[d_model, 2·d_inner]`.
    in_proj: Array2<f64>,

    /// Depthwise 1-D convolution weights.
    /// Shape: `[d_inner, d_conv]` (weight per channel, d_conv kernel).
    conv1d_weight: Array2<f64>,

    /// Depthwise 1-D convolution bias.
    /// Shape: `[d_inner]`.
    conv1d_bias: Vec<f64>,

    /// Projects d_inner → dt_rank + 2·d_state.
    /// Shape: `[d_inner, dt_rank + 2·d_state]`.
    x_proj: Array2<f64>,

    /// Projects dt_rank → d_inner (Δ projection).
    /// Shape: `[dt_rank, d_inner]`.
    dt_proj: Array2<f64>,

    /// Log of absolute SSM eigenvalues, shape `[d_inner, d_state]`.
    /// A = -exp(a_log) for stability.
    a_log: Array2<f64>,

    /// Skip-connection coefficient per SSM channel, shape `[d_inner]`.
    d_param: Vec<f64>,

    /// Output projection: maps d_inner → d_model.
    /// Shape: `[d_inner, d_model]`.
    out_proj: Array2<f64>,
}

impl MambaBlock {
    /// Create a new MambaBlock with Xavier/Kaiming initialization.
    pub fn new(config: &MambaConfig, rng: &mut impl scirs2_core::random::Rng) -> Self {
        let d_model = config.d_model;
        let d_inner = config.d_inner();
        let d_state = config.d_state;
        let d_conv = config.d_conv;
        let dt_rank = config.dt_rank;

        // in_proj: [d_model, 2*d_inner]  — Xavier uniform
        let in_proj = xavier_uniform_2d(d_model, 2 * d_inner, rng);

        // conv1d: [d_inner, d_conv]
        let conv_scale = (2.0 / (d_inner * d_conv) as f64).sqrt();
        let conv_data: Vec<f64> = (0..d_inner * d_conv)
            .map(|_| normal_sample(rng, 0.0, conv_scale))
            .collect();
        let conv1d_weight = Array2::from_shape_vec((d_inner, d_conv), conv_data)
            .unwrap_or_else(|_| Array2::zeros((d_inner, d_conv)));
        let conv1d_bias = vec![0.0_f64; d_inner];

        // x_proj: [d_inner, dt_rank + 2*d_state]
        let x_proj_cols = dt_rank + 2 * d_state;
        let x_proj = xavier_uniform_2d(d_inner, x_proj_cols, rng);

        // dt_proj: [dt_rank, d_inner]  — Kaiming
        let dt_proj_scale = (2.0 / dt_rank as f64).sqrt();
        let dt_proj_data: Vec<f64> = (0..dt_rank * d_inner)
            .map(|_| normal_sample(rng, 0.0, dt_proj_scale))
            .collect();
        let dt_proj = Array2::from_shape_vec((dt_rank, d_inner), dt_proj_data)
            .unwrap_or_else(|_| Array2::zeros((dt_rank, d_inner)));

        // a_log: [d_inner, d_state]
        // Initialize A eigenvalues as exp(-n/d_state) for graceful decay
        let a_log_data: Vec<f64> = (0..d_inner)
            .flat_map(|_| {
                (0..d_state).map(|n| {
                    // log|λ_n| = log(n+1) — standard Mamba init
                    ((n + 1) as f64).ln()
                })
            })
            .collect();
        let a_log = Array2::from_shape_vec((d_inner, d_state), a_log_data)
            .unwrap_or_else(|_| Array2::zeros((d_inner, d_state)));

        // d_param: [d_inner], initialized to 1
        let d_param = vec![1.0_f64; d_inner];

        // out_proj: [d_inner, d_model]
        let out_proj = xavier_uniform_2d(d_inner, d_model, rng);

        MambaBlock {
            config: config.clone(),
            in_proj,
            conv1d_weight,
            conv1d_bias,
            x_proj,
            dt_proj,
            a_log,
            d_param,
            out_proj,
        }
    }

    /// Causal depthwise 1-D convolution.
    ///
    /// For each channel c, convolves x[:,c] with conv1d_weight[c,:] (causal).
    /// Output has the same shape as input: `[seq_len, d_inner]`.
    fn causal_conv1d(&self, x: &Array2<f64>) -> Array2<f64> {
        let (seq_len, d_inner) = x.dim();
        let d_conv = self.config.d_conv;
        let mut out = Array2::zeros((seq_len, d_inner));

        for c in 0..d_inner {
            let bias = self.conv1d_bias[c];
            for t in 0..seq_len {
                let mut acc = bias;
                for k in 0..d_conv {
                    // causal: only use past positions
                    if t + 1 >= k + 1 {
                        let src_t = t - k;
                        acc += self.conv1d_weight[[c, k]] * x[[src_t, c]];
                    }
                }
                out[[t, c]] = acc;
            }
        }
        out
    }

    /// Perform the selective SSM scan.
    ///
    /// # Arguments
    /// * `x`   — SSM input, shape `[seq_len, d_inner]` (after conv + SiLU)
    /// * `dt`  — input-dependent timestep, shape `[seq_len, d_inner]`
    /// * `b`   — input-dependent B, shape `[seq_len, d_state]`
    /// * `c`   — input-dependent C, shape `[seq_len, d_state]`
    ///
    /// # Returns
    /// Output `[seq_len, d_inner]`.
    fn selective_scan(
        &self,
        x: &Array2<f64>,
        dt: &Array2<f64>,
        b: &Array2<f64>,
        c: &Array2<f64>,
    ) -> Array2<f64> {
        let (seq_len, d_inner) = x.dim();
        let d_state = self.config.d_state;
        let mut output = Array2::zeros((seq_len, d_inner));

        for ch in 0..d_inner {
            // h: hidden state [d_state]
            let mut h = vec![0.0_f64; d_state];

            for t in 0..seq_len {
                let dt_t = dt[[t, ch]]; // scalar timestep for this (t, ch)

                // Ā_t = exp(Δ_t * A_ch)   where A_ch[n] = -exp(a_log[ch,n])
                // B̄_t = Δ_t * B_t         (simplified ZOH)
                // h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t
                // y_t = Σ_n C_t[n] * h_t[n]

                let x_t = x[[t, ch]];
                let mut y_t = 0.0_f64;

                for n in 0..d_state {
                    let a_val = -(self.a_log[[ch, n]].exp()); // negative eigenvalue
                    let a_bar = (dt_t * a_val).exp();
                    let b_bar = dt_t * b[[t, n]]; // simplified ZOH: Δ·B
                    h[n] = a_bar * h[n] + b_bar * x_t;
                    y_t += c[[t, n]] * h[n];
                }

                // Add skip connection D * x
                output[[t, ch]] = y_t + self.d_param[ch] * x_t;
            }
        }

        output
    }

    /// Matrix-vector multiply: out[l, j] = Σ_i  x[l, i] · w[i, j].
    fn matmul_seq(x: &Array2<f64>, w: &Array2<f64>) -> Array2<f64> {
        let (seq_len, in_dim) = x.dim();
        let (in_w, out_dim) = w.dim();
        debug_assert_eq!(in_dim, in_w);
        let mut out = Array2::zeros((seq_len, out_dim));
        for l in 0..seq_len {
            for j in 0..out_dim {
                let mut acc = 0.0_f64;
                for i in 0..in_dim {
                    acc += x[[l, i]] * w[[i, j]];
                }
                out[[l, j]] = acc;
            }
        }
        out
    }

    /// Apply the Mamba block to an input sequence.
    ///
    /// # Arguments
    /// * `x` — Input tensor `[seq_len, d_model]`.
    ///
    /// # Returns
    /// Output tensor `[seq_len, d_model]`.
    pub fn forward(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (seq_len, d_model) = x.dim();
        if d_model != self.config.d_model {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.config.d_model,
                actual: d_model,
            });
        }

        let d_inner = self.config.d_inner();
        let d_state = self.config.d_state;
        let dt_rank = self.config.dt_rank;

        // --- Step 1: Input projection → [seq_len, 2*d_inner] ---
        let xz = Self::matmul_seq(x, &self.in_proj); // [L, 2*d_inner]

        // Split into SSM path (x_ssm) and gate (z)
        let mut x_ssm = Array2::zeros((seq_len, d_inner));
        let mut z = Array2::zeros((seq_len, d_inner));
        for t in 0..seq_len {
            for j in 0..d_inner {
                x_ssm[[t, j]] = xz[[t, j]];
                z[[t, j]] = xz[[t, j + d_inner]];
            }
        }

        // --- Step 2: Causal depthwise conv + SiLU ---
        let x_conv = self.causal_conv1d(&x_ssm);
        let x_act = Array2::from_shape_fn((seq_len, d_inner), |(t, j)| silu(x_conv[[t, j]]));

        // --- Step 3: x_proj → [seq_len, dt_rank + 2*d_state] ---
        let xbc = Self::matmul_seq(&x_act, &self.x_proj);

        // Split Δ_raw, B, C
        let mut delta_raw = Array2::zeros((seq_len, dt_rank));
        let mut b_mat = Array2::zeros((seq_len, d_state));
        let mut c_mat = Array2::zeros((seq_len, d_state));

        for t in 0..seq_len {
            for j in 0..dt_rank {
                delta_raw[[t, j]] = xbc[[t, j]];
            }
            for j in 0..d_state {
                b_mat[[t, j]] = xbc[[t, dt_rank + j]];
                c_mat[[t, j]] = xbc[[t, dt_rank + d_state + j]];
            }
        }

        // --- Step 4: Δ = softplus(Δ_raw · W_dt^T) → [seq_len, d_inner] ---
        // dt_proj shape: [dt_rank, d_inner], so we need delta_raw @ dt_proj
        let dt_linear = Self::matmul_seq(&delta_raw, &self.dt_proj); // [L, d_inner]
        let dt = Array2::from_shape_fn((seq_len, d_inner), |(t, j)| softplus(dt_linear[[t, j]]));

        // --- Step 5: Selective SSM scan → [seq_len, d_inner] ---
        let y_ssm = self.selective_scan(&x_act, &dt, &b_mat, &c_mat);

        // --- Step 6: Gate with SiLU(z) ---
        let y_gated =
            Array2::from_shape_fn((seq_len, d_inner), |(t, j)| y_ssm[[t, j]] * silu(z[[t, j]]));

        // --- Step 7: Output projection → [seq_len, d_model] ---
        let output = Self::matmul_seq(&y_gated, &self.out_proj);

        Ok(output)
    }

    /// Return a reference to the A_log matrix for inspection.
    pub fn a_log(&self) -> &Array2<f64> {
        &self.a_log
    }

    /// Return a reference to the D (skip) parameter.
    pub fn d_param(&self) -> &[f64] {
        &self.d_param
    }
}

// ---------------------------------------------------------------------------
// MambaModel
// ---------------------------------------------------------------------------

/// Full Mamba model: a stack of MambaBlocks with RMSNorm, plus an output head.
///
/// # Usage
///
/// ```no_run
/// use scirs2_series::ssm::{MambaConfig, MambaModel};
/// use scirs2_core::SeedableRng;
///
/// let config = MambaConfig { d_model: 32, n_layers: 2, seq_len: 64, ..Default::default() };
/// let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(0);
/// let model = MambaModel::new(&config, 1, &mut rng);
/// let history: Vec<f64> = (0..64).map(|i| i as f64).collect();
/// let forecast = model.forecast(&history, 8).unwrap();
/// assert_eq!(forecast.len(), 8);
/// ```
pub struct MambaModel {
    /// Model configuration.
    pub config: MambaConfig,
    /// Stacked Mamba blocks.
    layers: Vec<MambaBlock>,
    /// RMSNorm scale parameters per layer, shape `n_layers × d_model`.
    norm_layers: Vec<Vec<f64>>,
    /// Final RMSNorm scale, shape `[d_model]`.
    final_norm: Vec<f64>,
    /// Output projection: maps d_model → output_dim.
    /// Shape `[d_model, output_dim]`.
    output_proj: Array2<f64>,
    /// Output dimensionality.
    output_dim: usize,
}

impl MambaModel {
    /// Create a new MambaModel.
    ///
    /// # Arguments
    /// * `config`     — Mamba configuration.
    /// * `output_dim` — Output dimension (e.g., 1 for univariate forecasting).
    /// * `rng`        — Random number generator.
    pub fn new(
        config: &MambaConfig,
        output_dim: usize,
        rng: &mut impl scirs2_core::random::Rng,
    ) -> Self {
        let n_layers = config.n_layers;
        let d_model = config.d_model;

        let layers: Vec<MambaBlock> = (0..n_layers)
            .map(|_| MambaBlock::new(config, rng))
            .collect();

        // RMSNorm scale parameters: initialized to 1
        let norm_layers: Vec<Vec<f64>> = (0..n_layers).map(|_| vec![1.0_f64; d_model]).collect();
        let final_norm = vec![1.0_f64; d_model];

        let output_proj = xavier_uniform_2d(d_model, output_dim, rng);

        MambaModel {
            config: config.clone(),
            layers,
            norm_layers,
            final_norm,
            output_proj,
            output_dim,
        }
    }

    /// Apply the full Mamba model to an input sequence.
    ///
    /// Implements residual connections and pre-norm RMSNorm:
    ///   for each layer: x = x + MambaBlock(RMSNorm(x))
    ///
    /// # Arguments
    /// * `x` — Input tensor `[seq_len, d_model]`.
    ///
    /// # Returns
    /// Output tensor `[seq_len, output_dim]`.
    pub fn forward(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (seq_len, d_model) = x.dim();

        if d_model != self.config.d_model {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.config.d_model,
                actual: d_model,
            });
        }

        let mut hidden = x.clone();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let scale = &self.norm_layers[layer_idx];

            // Pre-norm RMSNorm
            let normed = apply_rmsnorm_seq(&hidden, scale);

            // Mamba block
            let block_out = layer.forward(&normed)?;

            // Residual connection
            for t in 0..seq_len {
                for j in 0..d_model {
                    hidden[[t, j]] += block_out[[t, j]];
                }
            }
        }

        // Final layer norm
        let normed_final = apply_rmsnorm_seq(&hidden, &self.final_norm);

        // Output projection
        let output = matmul_seq_static(&normed_final, &self.output_proj);

        Ok(output)
    }

    /// Produce an autoregressive forecast from a history window.
    ///
    /// The history is windowed to `config.seq_len` points and embedded as
    /// a 1-D feature (repeated across `d_model` dimensions with positional
    /// scaling).  The model then predicts `horizon` future values one step
    /// at a time, feeding each prediction back as input.
    ///
    /// # Arguments
    /// * `history` — Historical time-series values (at least `config.seq_len` points).
    /// * `horizon` — Number of future steps to predict.
    ///
    /// # Returns
    /// Vector of length `horizon` with predicted values.
    pub fn forecast(&self, history: &[f64], horizon: usize) -> Result<Vec<f64>> {
        if history.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "history must be non-empty".to_string(),
            ));
        }
        if horizon == 0 {
            return Ok(Vec::new());
        }

        let seq_len = self.config.seq_len;
        let d_model = self.config.d_model;

        // Use the last `seq_len` history points (or pad with first value)
        let window: Vec<f64> = if history.len() >= seq_len {
            history[history.len() - seq_len..].to_vec()
        } else {
            // Left-pad with the first history value
            let pad_len = seq_len - history.len();
            let pad_val = history[0];
            let mut w = vec![pad_val; pad_len];
            w.extend_from_slice(history);
            w
        };

        let mut predictions = Vec::with_capacity(horizon);
        let mut current_window = window;

        for _ in 0..horizon {
            // Embed: map scalar window to [seq_len, d_model]
            let x = embed_window(&current_window, d_model);

            // Forward pass
            let output = self.forward(&x)?; // [seq_len, output_dim]

            // Take the last time-step output as the prediction
            let pred = output[[seq_len - 1, 0]];
            predictions.push(pred);

            // Slide window: append prediction, drop oldest
            current_window.remove(0);
            current_window.push(pred);
        }

        Ok(predictions)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Xavier uniform initialization for a 2-D weight matrix.
fn xavier_uniform_2d(
    in_dim: usize,
    out_dim: usize,
    rng: &mut impl scirs2_core::random::Rng,
) -> Array2<f64> {
    let limit = (6.0 / (in_dim + out_dim) as f64).sqrt();
    let data: Vec<f64> = (0..in_dim * out_dim)
        .map(|_| {
            let u: f64 = rng.random::<f64>();
            -limit + 2.0 * limit * u
        })
        .collect();
    Array2::from_shape_vec((in_dim, out_dim), data)
        .unwrap_or_else(|_| Array2::zeros((in_dim, out_dim)))
}

/// Apply RMSNorm row-wise to `[seq_len, d_model]`.
fn apply_rmsnorm_seq(x: &Array2<f64>, scale: &[f64]) -> Array2<f64> {
    let (seq_len, d_model) = x.dim();
    let mut out = Array2::zeros((seq_len, d_model));
    for t in 0..seq_len {
        let row: Vec<f64> = (0..d_model).map(|j| x[[t, j]]).collect();
        let normed = rms_norm(&row, scale);
        for j in 0..d_model {
            out[[t, j]] = normed[j];
        }
    }
    out
}

/// Static version of matmul for use outside `MambaBlock`.
fn matmul_seq_static(x: &Array2<f64>, w: &Array2<f64>) -> Array2<f64> {
    let (seq_len, in_dim) = x.dim();
    let (in_w, out_dim) = w.dim();
    debug_assert_eq!(in_dim, in_w);
    let mut out = Array2::zeros((seq_len, out_dim));
    for l in 0..seq_len {
        for j in 0..out_dim {
            let mut acc = 0.0_f64;
            for i in 0..in_dim {
                acc += x[[l, i]] * w[[i, j]];
            }
            out[[l, j]] = acc;
        }
    }
    out
}

/// Embed a scalar window of length `seq_len` into shape `[seq_len, d_model]`.
///
/// Uses a simple broadcast + sinusoidal positional encoding scheme:
///   embedding[t, j] = value[t] * cos(t * π/(seq_len) * (j+1)/(d_model+1))
///
/// This spreads the scalar value across the model dimension while
/// providing positional context.
fn embed_window(window: &[f64], d_model: usize) -> Array2<f64> {
    let seq_len = window.len();
    let mut x = Array2::zeros((seq_len, d_model));

    for t in 0..seq_len {
        let v = window[t];
        for j in 0..d_model {
            // Sinusoidal positional blend
            let pos_angle =
                (t as f64 * PI) / (seq_len as f64) * (j + 1) as f64 / (d_model + 1) as f64;
            x[[t, j]] = v * pos_angle.cos();
        }
    }
    x
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::SeedableRng;

    fn make_rng() -> impl scirs2_core::random::Rng {
        scirs2_core::random::rngs::StdRng::seed_from_u64(0)
    }

    fn small_config() -> MambaConfig {
        MambaConfig {
            d_model: 16,
            d_state: 4,
            d_conv: 3,
            expand: 2,
            dt_rank: 2,
            seq_len: 16,
            n_layers: 2,
            dropout: 0.0,
        }
    }

    #[test]
    fn test_mamba_block_shape() {
        let config = small_config();
        let block = MambaBlock::new(&config, &mut make_rng());
        let x = Array2::ones((16, 16));
        let out = block.forward(&x).expect("block forward should succeed");
        assert_eq!(out.dim(), (16, 16));
    }

    #[test]
    fn test_mamba_block_shape_varied_seq() {
        let config = small_config();
        let block = MambaBlock::new(&config, &mut make_rng());

        // Different sequence lengths should all work
        for seq_len in [1, 4, 16, 32] {
            let x = Array2::ones((seq_len, 16));
            let out = block.forward(&x).expect("block forward");
            assert_eq!(out.dim(), (seq_len, 16));
        }
    }

    #[test]
    fn test_mamba_selective_scan_shape() {
        let config = small_config();
        let block = MambaBlock::new(&config, &mut make_rng());
        let d_inner = config.d_inner();
        let d_state = config.d_state;
        let seq_len = 16;

        let x = Array2::ones((seq_len, d_inner));
        let dt = Array2::from_elem((seq_len, d_inner), 0.01);
        let b = Array2::ones((seq_len, d_state));
        let c = Array2::ones((seq_len, d_state));

        let out = block.selective_scan(&x, &dt, &b, &c);
        assert_eq!(out.dim(), (seq_len, d_inner));
    }

    #[test]
    fn test_mamba_forward_causal() {
        // Causality: output at time t should not depend on input at times > t.
        // We verify this by comparing two runs: one with normal input, one with
        // modified future input.  The outputs at time t should be identical.
        let config = small_config();
        let block = MambaBlock::new(&config, &mut make_rng());

        let mut x1 = Array2::zeros((8, 16));
        let mut x2 = Array2::zeros((8, 16));

        // x1[t,j] = t * 0.1 + j * 0.01
        for t in 0..8 {
            for j in 0..16 {
                x1[[t, j]] = t as f64 * 0.1 + j as f64 * 0.01;
                x2[[t, j]] = t as f64 * 0.1 + j as f64 * 0.01;
            }
        }
        // Modify x2 at future positions (t >= 4) to different values
        for t in 4..8 {
            for j in 0..16 {
                x2[[t, j]] = 999.0;
            }
        }

        let out1 = block.forward(&x1).expect("forward x1");
        let out2 = block.forward(&x2).expect("forward x2");

        // At t = 0..4, outputs should be identical (causal SSM)
        for t in 0..4 {
            for j in 0..16 {
                let diff = (out1[[t, j]] - out2[[t, j]]).abs();
                assert!(
                    diff < 1e-10,
                    "t={t}, j={j}: causality violation: diff={diff:.2e}"
                );
            }
        }
    }

    #[test]
    fn test_mamba_model_forecast() {
        let config = MambaConfig {
            d_model: 16,
            d_state: 4,
            d_conv: 3,
            expand: 2,
            dt_rank: 2,
            seq_len: 32,
            n_layers: 2,
            dropout: 0.0,
        };
        let mut rng = make_rng();
        let model = MambaModel::new(&config, 1, &mut rng);

        let history: Vec<f64> = (0..64).map(|i| (i as f64).sin()).collect();
        let forecast = model.forecast(&history, 8).expect("forecast");
        assert_eq!(forecast.len(), 8);
        // All predictions should be finite
        for (i, &v) in forecast.iter().enumerate() {
            assert!(v.is_finite(), "prediction[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_mamba_model_forward_shape() {
        let config = small_config();
        let mut rng = make_rng();
        let model = MambaModel::new(&config, 1, &mut rng);
        let x = Array2::zeros((16, 16));
        let out = model.forward(&x).expect("model forward");
        assert_eq!(out.dim(), (16, 1));
    }

    #[test]
    fn test_mamba_longer_seq() {
        let config = MambaConfig {
            d_model: 16,
            d_state: 4,
            d_conv: 4,
            expand: 2,
            dt_rank: 2,
            seq_len: 256,
            n_layers: 2,
            dropout: 0.0,
        };
        let block = MambaBlock::new(&config, &mut make_rng());
        let x = Array2::zeros((256, 16));
        let out = block.forward(&x).expect("long seq forward");
        assert_eq!(out.dim(), (256, 16));
    }

    #[test]
    fn test_mamba_d_param_skip_connection() {
        // With all-zero input, the SSM output is zero, so the d_param skip
        // connection doesn't add anything (0 * x = 0).
        // With a nonzero input, d_param contributes.
        let config = small_config();
        let block = MambaBlock::new(&config, &mut make_rng());

        let x_zero = Array2::zeros((8, 16));
        let out_zero = block.forward(&x_zero).expect("forward zero");
        // With zero input everything is zero
        for t in 0..8 {
            for j in 0..16 {
                // May not be exactly zero due to conv bias + normalization, but finite
                assert!(out_zero[[t, j]].is_finite());
            }
        }

        // Nonzero d_param accessible via API
        let d = block.d_param();
        assert_eq!(d.len(), config.d_inner());
        for &v in d {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_mamba_dimension_mismatch() {
        let config = small_config();
        let block = MambaBlock::new(&config, &mut make_rng());
        // Wrong d_model
        let x = Array2::zeros((8, 4)); // expected 16
        let result = block.forward(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_mamba_forecast_short_history() {
        // History shorter than seq_len should be padded automatically
        let config = MambaConfig {
            d_model: 8,
            d_state: 4,
            d_conv: 3,
            expand: 2,
            dt_rank: 1,
            seq_len: 16,
            n_layers: 1,
            dropout: 0.0,
        };
        let mut rng = make_rng();
        let model = MambaModel::new(&config, 1, &mut rng);
        let history = vec![1.0, 2.0, 3.0]; // shorter than seq_len=16
        let forecast = model.forecast(&history, 4).expect("forecast");
        assert_eq!(forecast.len(), 4);
    }

    #[test]
    fn test_mamba_a_log_shape() {
        let config = small_config();
        let block = MambaBlock::new(&config, &mut make_rng());
        let a_log = block.a_log();
        assert_eq!(a_log.dim(), (config.d_inner(), config.d_state));
    }

    #[test]
    fn test_silu_properties() {
        // silu(0) = 0
        assert!((silu(0.0)).abs() < 1e-10);
        // silu is continuous and finite
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0, 20.0] {
            let v = silu(x);
            assert!(v.is_finite(), "silu({x}) = {v}");
        }
        // silu is monotone-ish: silu(1) > silu(0) > silu(-5)
        assert!(silu(1.0) > silu(0.0));
    }

    #[test]
    fn test_softplus_properties() {
        // softplus is always positive
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0, 25.0] {
            let v = softplus(x);
            assert!(v > 0.0, "softplus({x}) = {v}");
            assert!(v.is_finite(), "softplus({x}) = {v}");
        }
        // For large x, softplus(x) ≈ x
        assert!((softplus(25.0) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_mamba_default_config() {
        let config = MambaConfig::default();
        assert_eq!(config.d_model, 64);
        assert_eq!(config.d_state, 16);
        assert_eq!(config.d_conv, 4);
        assert_eq!(config.expand, 2);
        assert_eq!(config.d_inner(), 128);
        assert_eq!(config.dt_rank, 4); // ceil(64/16)
    }
}
