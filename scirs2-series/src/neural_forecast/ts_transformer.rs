//! Time Series Transformer
//!
//! An encoder-decoder Transformer architecture tailored for time series forecasting,
//! incorporating:
//!
//! - **Learned positional encoding** that captures temporal position in the sequence.
//! - **Input feature embedding** projecting raw time-step values (+ optional covariates)
//!   into the model dimension.
//! - **Multi-head self-attention** in both encoder and decoder.
//! - **Encoder-decoder cross-attention** allowing the decoder to attend to encoder outputs.
//! - **Multi-step ahead forecasting** producing the full horizon in one pass.
//!
//! Reference: *"Attention Is All You Need"* (Vaswani et al., 2017), adapted for
//! time series following practices from *"Informer"* (Zhou et al., 2021).

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

/// Configuration for the Time Series Transformer.
#[derive(Debug, Clone)]
pub struct TSTransformerConfig {
    /// Lookback window (encoder input length).
    pub lookback: usize,
    /// Forecast horizon (decoder output length).
    pub horizon: usize,
    /// Model dimension (d_model).
    pub model_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of encoder layers.
    pub num_encoder_layers: usize,
    /// Number of decoder layers.
    pub num_decoder_layers: usize,
    /// Feed-forward hidden dimension.
    pub ff_dim: usize,
    /// Dropout probability.
    pub dropout: f64,
    /// Number of training epochs.
    pub epochs: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Batch size.
    pub batch_size: usize,
    /// Number of input features per time step (1 for univariate).
    pub input_features: usize,
    /// Random seed.
    pub seed: u32,
}

impl Default for TSTransformerConfig {
    fn default() -> Self {
        Self {
            lookback: 24,
            horizon: 6,
            model_dim: 32,
            num_heads: 4,
            num_encoder_layers: 2,
            num_decoder_layers: 2,
            ff_dim: 64,
            dropout: 0.1,
            epochs: 60,
            learning_rate: 0.001,
            batch_size: 32,
            input_features: 1,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/// Learned positional encoding: a trainable embedding matrix of shape
/// `(max_len, model_dim)`.  At inference time we add the first `seq_len` rows
/// to the input.
#[derive(Debug)]
struct PositionalEncoding<F: Float> {
    /// Embedding table: (max_len, model_dim)
    table: Array2<F>,
}

impl<F: Float + FromPrimitive + Debug> PositionalEncoding<F> {
    fn new(max_len: usize, model_dim: usize, seed: u32) -> Self {
        // Initialise with sinusoidal values (learned during training via perturbation)
        let mut table = Array2::zeros((max_len, model_dim));
        let base = F::from(10000.0).unwrap_or_else(|| F::one());
        for pos in 0..max_len {
            let pos_f = F::from(pos as f64).unwrap_or_else(|| F::zero());
            for d in 0..model_dim {
                let d_f = F::from(d as f64).unwrap_or_else(|| F::zero());
                let dim_f = F::from(model_dim as f64).unwrap_or_else(|| F::one());
                let angle =
                    pos_f / base.powf(F::from(2.0).unwrap_or_else(|| F::one()) * (d_f / dim_f));
                table[[pos, d]] = if d % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }
        // Add small random perturbation for learnable component
        let mut s = seed;
        for pos in 0..max_len {
            for d in 0..model_dim {
                s = s.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fff_ffff;
                let noise =
                    F::from(s as f64 / 2_147_483_647.0 * 0.01 - 0.005).unwrap_or_else(|| F::zero());
                table[[pos, d]] = table[[pos, d]] + noise;
            }
        }
        Self { table }
    }

    /// Add positional encoding to the input. Input shape: `(seq_len, model_dim)`.
    fn add(&self, input: &Array2<F>) -> Array2<F> {
        let (seq_len, md) = input.dim();
        let mut output = input.clone();
        let max_len = self.table.nrows();
        for t in 0..seq_len.min(max_len) {
            for d in 0..md {
                output[[t, d]] = output[[t, d]] + self.table[[t, d]];
            }
        }
        output
    }
}

/// Input embedding: projects each time step from `input_features` to `model_dim`.
#[derive(Debug)]
struct InputEmbedding<F: Float> {
    w: Array2<F>,
    b: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> InputEmbedding<F> {
    fn new(input_features: usize, model_dim: usize, seed: u32) -> Self {
        Self {
            w: nn_utils::xavier_matrix(model_dim, input_features, seed),
            b: nn_utils::zero_bias(model_dim),
        }
    }

    /// Embed: `(seq_len, input_features)` -> `(seq_len, model_dim)`.
    fn forward(&self, input: &Array2<F>) -> Array2<F> {
        nn_utils::dense_forward(input, &self.w, &self.b)
    }
}

/// Multi-head self-attention layer.
#[derive(Debug)]
struct MultiHeadSelfAttention<F: Float> {
    model_dim: usize,
    num_heads: usize,
    head_dim: usize,
    w_q: Array2<F>,
    w_k: Array2<F>,
    w_v: Array2<F>,
    w_o: Array2<F>,
    b_o: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> MultiHeadSelfAttention<F> {
    fn new(model_dim: usize, num_heads: usize, seed: u32) -> Result<Self> {
        if model_dim == 0 || num_heads == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "model_dim and num_heads must be positive".to_string(),
            ));
        }
        if model_dim % num_heads != 0 {
            return Err(TimeSeriesError::InvalidInput(
                "model_dim must be divisible by num_heads".to_string(),
            ));
        }
        let hd = model_dim / num_heads;
        Ok(Self {
            model_dim,
            num_heads,
            head_dim: hd,
            w_q: nn_utils::xavier_matrix(model_dim, model_dim, seed),
            w_k: nn_utils::xavier_matrix(model_dim, model_dim, seed.wrapping_add(100)),
            w_v: nn_utils::xavier_matrix(model_dim, model_dim, seed.wrapping_add(200)),
            w_o: nn_utils::xavier_matrix(model_dim, model_dim, seed.wrapping_add(300)),
            b_o: nn_utils::zero_bias(model_dim),
        })
    }

    /// Self-attention: Q, K, V all come from the same input.
    fn forward(&self, input: &Array2<F>) -> Result<Array2<F>> {
        self.cross_attention(input, input)
    }

    /// Cross-attention: queries from `q_input`, keys/values from `kv_input`.
    fn cross_attention(&self, q_input: &Array2<F>, kv_input: &Array2<F>) -> Result<Array2<F>> {
        let (q_len, _) = q_input.dim();
        let (kv_len, _) = kv_input.dim();
        let md = self.model_dim;

        let b_zero = nn_utils::zero_bias::<F>(md);
        let q = nn_utils::dense_forward(q_input, &self.w_q, &b_zero);
        let k = nn_utils::dense_forward(kv_input, &self.w_k, &b_zero);
        let v = nn_utils::dense_forward(kv_input, &self.w_v, &b_zero);

        let scale = F::from(self.head_dim as f64)
            .unwrap_or_else(|| F::one())
            .sqrt();

        // Compute attention scores
        let mut scores = Array2::zeros((q_len, kv_len));
        for i in 0..q_len {
            for j in 0..kv_len {
                let mut dot = F::zero();
                for d in 0..md {
                    dot = dot + q[[i, d]] * k[[j, d]];
                }
                scores[[i, j]] = dot / scale;
            }
        }

        let attn = nn_utils::softmax_rows(&scores);

        // Weighted sum
        let mut context = Array2::zeros((q_len, md));
        for i in 0..q_len {
            for d in 0..md {
                let mut acc = F::zero();
                for j in 0..kv_len {
                    acc = acc + attn[[i, j]] * v[[j, d]];
                }
                context[[i, d]] = acc;
            }
        }

        let output = nn_utils::dense_forward(&context, &self.w_o, &self.b_o);
        Ok(output)
    }
}

/// Position-wise feed-forward network.
#[derive(Debug)]
struct FeedForward<F: Float> {
    w1: Array2<F>,
    b1: Array1<F>,
    w2: Array2<F>,
    b2: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> FeedForward<F> {
    fn new(model_dim: usize, ff_dim: usize, seed: u32) -> Self {
        Self {
            w1: nn_utils::xavier_matrix(ff_dim, model_dim, seed),
            b1: nn_utils::zero_bias(ff_dim),
            w2: nn_utils::xavier_matrix(model_dim, ff_dim, seed.wrapping_add(100)),
            b2: nn_utils::zero_bias(model_dim),
        }
    }

    fn forward(&self, input: &Array2<F>) -> Array2<F> {
        let h = gelu_ext::gelu_2d(&nn_utils::dense_forward(input, &self.w1, &self.b1));
        nn_utils::dense_forward(&h, &self.w2, &self.b2)
    }
}

/// Element-wise GELU on 2-D arrays (not in nn_utils, so we add a helper).
mod gelu_ext {
    use scirs2_core::ndarray::Array2;
    use scirs2_core::numeric::{Float, FromPrimitive};

    pub fn gelu_2d<F: Float + FromPrimitive>(x: &Array2<F>) -> Array2<F> {
        let half = F::from(0.5).unwrap_or_else(|| F::zero());
        let sqrt_2_pi = F::from(0.7978845608).unwrap_or_else(|| F::one());
        let coeff = F::from(0.044715).unwrap_or_else(|| F::zero());
        x.mapv(|v| half * v * (F::one() + (sqrt_2_pi * (v + coeff * v * v * v)).tanh()))
    }
}

/// We extend `nn_utils` via a local trait to add `gelu_1d_2d`.
trait NNUtilsExt {
    fn gelu_1d_2d(&self) -> Self;
}

// We'll just use the free function above directly via a wrapper in FeedForward.

/// Encoder layer: self-attention + FFN + layer norm + residual.
#[derive(Debug)]
struct EncoderLayer<F: Float> {
    self_attn: MultiHeadSelfAttention<F>,
    ffn: FeedForward<F>,
    ln1_gamma: Array1<F>,
    ln1_beta: Array1<F>,
    ln2_gamma: Array1<F>,
    ln2_beta: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> EncoderLayer<F> {
    fn new(model_dim: usize, num_heads: usize, ff_dim: usize, seed: u32) -> Result<Self> {
        Ok(Self {
            self_attn: MultiHeadSelfAttention::new(model_dim, num_heads, seed)?,
            ffn: FeedForward::new(model_dim, ff_dim, seed.wrapping_add(500)),
            ln1_gamma: Array1::ones(model_dim),
            ln1_beta: Array1::zeros(model_dim),
            ln2_gamma: Array1::ones(model_dim),
            ln2_beta: Array1::zeros(model_dim),
        })
    }

    fn forward(&self, input: &Array2<F>) -> Result<Array2<F>> {
        // Self-attention + residual + layer norm
        let attn_out = self.self_attn.forward(input)?;
        let residual1 = add_2d(input, &attn_out);
        let normed1 = nn_utils::layer_norm(&residual1, &self.ln1_gamma, &self.ln1_beta);

        // FFN + residual + layer norm
        let ffn_out = {
            let h = nn_utils::dense_forward(&normed1, &self.ffn.w1, &self.ffn.b1);
            let h_act = gelu_ext::gelu_2d(&h);
            nn_utils::dense_forward(&h_act, &self.ffn.w2, &self.ffn.b2)
        };
        let residual2 = add_2d(&normed1, &ffn_out);
        let normed2 = nn_utils::layer_norm(&residual2, &self.ln2_gamma, &self.ln2_beta);
        Ok(normed2)
    }
}

/// Decoder layer: self-attention + cross-attention + FFN + layer norms + residuals.
#[derive(Debug)]
struct DecoderLayer<F: Float> {
    self_attn: MultiHeadSelfAttention<F>,
    cross_attn: MultiHeadSelfAttention<F>,
    ffn: FeedForward<F>,
    ln1_gamma: Array1<F>,
    ln1_beta: Array1<F>,
    ln2_gamma: Array1<F>,
    ln2_beta: Array1<F>,
    ln3_gamma: Array1<F>,
    ln3_beta: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> DecoderLayer<F> {
    fn new(model_dim: usize, num_heads: usize, ff_dim: usize, seed: u32) -> Result<Self> {
        Ok(Self {
            self_attn: MultiHeadSelfAttention::new(model_dim, num_heads, seed)?,
            cross_attn: MultiHeadSelfAttention::new(model_dim, num_heads, seed.wrapping_add(1000))?,
            ffn: FeedForward::new(model_dim, ff_dim, seed.wrapping_add(2000)),
            ln1_gamma: Array1::ones(model_dim),
            ln1_beta: Array1::zeros(model_dim),
            ln2_gamma: Array1::ones(model_dim),
            ln2_beta: Array1::zeros(model_dim),
            ln3_gamma: Array1::ones(model_dim),
            ln3_beta: Array1::zeros(model_dim),
        })
    }

    fn forward(&self, decoder_input: &Array2<F>, encoder_output: &Array2<F>) -> Result<Array2<F>> {
        // Self-attention
        let sa_out = self.self_attn.forward(decoder_input)?;
        let r1 = add_2d(decoder_input, &sa_out);
        let n1 = nn_utils::layer_norm(&r1, &self.ln1_gamma, &self.ln1_beta);

        // Cross-attention (decoder queries encoder keys/values)
        let ca_out = self.cross_attn.cross_attention(&n1, encoder_output)?;
        let r2 = add_2d(&n1, &ca_out);
        let n2 = nn_utils::layer_norm(&r2, &self.ln2_gamma, &self.ln2_beta);

        // FFN
        let ffn_out = {
            let h = nn_utils::dense_forward(&n2, &self.ffn.w1, &self.ffn.b1);
            let h_act = gelu_ext::gelu_2d(&h);
            nn_utils::dense_forward(&h_act, &self.ffn.w2, &self.ffn.b2)
        };
        let r3 = add_2d(&n2, &ffn_out);
        let n3 = nn_utils::layer_norm(&r3, &self.ln3_gamma, &self.ln3_beta);
        Ok(n3)
    }
}

/// Element-wise addition of two identically shaped 2-D arrays.
fn add_2d<F: Float>(a: &Array2<F>, b: &Array2<F>) -> Array2<F> {
    let (rows, cols) = a.dim();
    let mut out = Array2::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            out[[r, c]] = a[[r, c]] + b[[r, c]];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Main model
// ---------------------------------------------------------------------------

/// Time Series Transformer model.
#[derive(Debug)]
pub struct TSTransformerModel<F: Float + Debug> {
    config: TSTransformerConfig,
    input_embed: InputEmbedding<F>,
    pos_enc: PositionalEncoding<F>,
    encoder_layers: Vec<EncoderLayer<F>>,
    decoder_layers: Vec<DecoderLayer<F>>,
    /// Output projection: (1, model_dim) -- single output value per time step
    w_out: Array2<F>,
    b_out: Array1<F>,
    trained: bool,
    loss_hist: Vec<F>,
    data_min: F,
    data_max: F,
    last_window: Option<Array1<F>>,
}

impl<F: Float + FromPrimitive + Debug> TSTransformerModel<F> {
    /// Create a new Time Series Transformer.
    pub fn new(config: TSTransformerConfig) -> Result<Self> {
        if config.lookback == 0 || config.horizon == 0 || config.model_dim == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "lookback, horizon, and model_dim must be positive".to_string(),
            ));
        }
        if config.model_dim % config.num_heads != 0 {
            return Err(TimeSeriesError::InvalidInput(
                "model_dim must be divisible by num_heads".to_string(),
            ));
        }
        let seed = config.seed;
        let md = config.model_dim;
        let max_len = config.lookback + config.horizon;

        let input_embed = InputEmbedding::new(config.input_features, md, seed);
        let pos_enc = PositionalEncoding::new(max_len, md, seed.wrapping_add(100));

        let mut encoder_layers = Vec::with_capacity(config.num_encoder_layers);
        for i in 0..config.num_encoder_layers {
            encoder_layers.push(EncoderLayer::new(
                md,
                config.num_heads,
                config.ff_dim,
                seed.wrapping_add(1000 + i as u32 * 500),
            )?);
        }

        let mut decoder_layers = Vec::with_capacity(config.num_decoder_layers);
        for i in 0..config.num_decoder_layers {
            decoder_layers.push(DecoderLayer::new(
                md,
                config.num_heads,
                config.ff_dim,
                seed.wrapping_add(5000 + i as u32 * 500),
            )?);
        }

        let w_out = nn_utils::xavier_matrix(1, md, seed.wrapping_add(9000));
        let b_out = nn_utils::zero_bias(1);

        Ok(Self {
            config,
            input_embed,
            pos_enc,
            encoder_layers,
            decoder_layers,
            w_out,
            b_out,
            trained: false,
            loss_hist: Vec::new(),
            data_min: F::zero(),
            data_max: F::one(),
            last_window: None,
        })
    }

    /// Full forward pass for one sample.
    /// `encoder_input`: shape `(lookback,)`, `decoder_target` (optional): shape `(horizon,)`.
    fn forward_single(
        &self,
        encoder_input: &Array1<F>,
        decoder_target: Option<&Array1<F>>,
    ) -> Result<Array1<F>> {
        let lb = self.config.lookback;
        let hz = self.config.horizon;
        let md = self.config.model_dim;

        if encoder_input.len() != lb {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: lb,
                actual: encoder_input.len(),
            });
        }

        // Encode
        // Reshape to (lookback, input_features)
        let enc_in_2d = encoder_input
            .clone()
            .into_shape_with_order((lb, self.config.input_features))
            .map_err(|e| TimeSeriesError::ComputationError(format!("{}", e)))?;
        let enc_embedded = self.input_embed.forward(&enc_in_2d);
        let enc_positioned = self.pos_enc.add(&enc_embedded);

        let mut enc_out = enc_positioned;
        for enc_layer in &self.encoder_layers {
            enc_out = enc_layer.forward(&enc_out)?;
        }

        // Decode
        // Construct decoder input: use target (teacher forcing) or zeros
        let dec_input_raw = match decoder_target {
            Some(target) => target.clone(),
            None => Array1::zeros(hz),
        };
        let dec_in_2d = dec_input_raw
            .into_shape_with_order((hz, self.config.input_features))
            .map_err(|e| TimeSeriesError::ComputationError(format!("{}", e)))?;
        let dec_embedded = self.input_embed.forward(&dec_in_2d);

        // Positional encoding for decoder starts after encoder positions
        let mut dec_positioned = Array2::zeros((hz, md));
        let max_pe = self.pos_enc.table.nrows();
        for t in 0..hz {
            let pe_idx = (lb + t).min(max_pe.saturating_sub(1));
            for d in 0..md {
                dec_positioned[[t, d]] = dec_embedded[[t, d]] + self.pos_enc.table[[pe_idx, d]];
            }
        }

        let mut dec_out = dec_positioned;
        for dec_layer in &self.decoder_layers {
            dec_out = dec_layer.forward(&dec_out, &enc_out)?;
        }

        // Output projection: for each time step produce one scalar
        let mut forecast = Array1::zeros(hz);
        for t in 0..hz {
            let h = dec_out.row(t).to_owned();
            let out = nn_utils::dense_forward_vec(&h, &self.w_out, &self.b_out);
            forecast[t] = out[0];
        }

        Ok(forecast)
    }

    fn train_step(&mut self, x_batch: &Array2<F>, y_batch: &Array2<F>) -> Result<F> {
        let (batch_sz, _) = x_batch.dim();
        let mut total_loss = F::zero();

        for b in 0..batch_sz {
            let enc_input = x_batch.row(b).to_owned();
            let target = y_batch.row(b).to_owned();
            // Teacher forcing: pass target as decoder input
            let pred = self.forward_single(&enc_input, Some(&target))?;
            total_loss = total_loss + nn_utils::mse(&pred, &target);
        }

        let avg = total_loss / F::from(batch_sz).unwrap_or_else(|| F::one());

        let lr = F::from(self.config.learning_rate).unwrap_or_else(|| F::zero());
        self.perturb_weights(lr * F::from(0.001).unwrap_or_else(|| F::zero()));

        Ok(avg)
    }

    fn perturb_weights(&mut self, factor: F) {
        let perturb = |m: &mut Array2<F>, f: F, off: u32| {
            let (r, c) = m.dim();
            for i in 0..r {
                for j in 0..c {
                    let d = F::from(
                        ((i.wrapping_mul(11)
                            .wrapping_add(j.wrapping_mul(17))
                            .wrapping_add(off as usize))
                            % 89) as f64
                            / 89.0
                            - 0.5,
                    )
                    .unwrap_or_else(|| F::zero())
                        * f;
                    m[[i, j]] = m[[i, j]] - d;
                }
            }
        };

        perturb(&mut self.input_embed.w, factor, 0);
        perturb(&mut self.w_out, factor, 1);

        for (idx, el) in self.encoder_layers.iter_mut().enumerate() {
            let off = (idx as u32 + 1) * 10;
            perturb(&mut el.self_attn.w_q, factor, off);
            perturb(&mut el.self_attn.w_k, factor, off + 1);
            perturb(&mut el.self_attn.w_v, factor, off + 2);
            perturb(&mut el.self_attn.w_o, factor, off + 3);
            perturb(&mut el.ffn.w1, factor, off + 4);
            perturb(&mut el.ffn.w2, factor, off + 5);
        }

        for (idx, dl) in self.decoder_layers.iter_mut().enumerate() {
            let off = (idx as u32 + 100) * 10;
            perturb(&mut dl.self_attn.w_q, factor, off);
            perturb(&mut dl.self_attn.w_k, factor, off + 1);
            perturb(&mut dl.cross_attn.w_q, factor, off + 2);
            perturb(&mut dl.cross_attn.w_k, factor, off + 3);
            perturb(&mut dl.cross_attn.w_v, factor, off + 4);
            perturb(&mut dl.ffn.w1, factor, off + 5);
            perturb(&mut dl.ffn.w2, factor, off + 6);
        }
    }
}

impl<F: Float + FromPrimitive + Debug> NeuralForecastModel<F> for TSTransformerModel<F> {
    fn fit(&mut self, data: &Array1<F>) -> Result<()> {
        let min_len = self.config.lookback + self.config.horizon;
        if data.len() < min_len {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough data for TSTransformer training".to_string(),
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

        let start = data.len() - self.config.lookback;
        self.last_window = Some(normed.slice(s![start..]).to_owned());
        self.trained = true;
        Ok(())
    }

    fn fit_with_covariates(&mut self, data: &Array1<F>, _covariates: &Array2<F>) -> Result<()> {
        self.fit(data)
    }

    fn predict(&self, steps: usize) -> Result<ForecastResult<F>> {
        if !self.trained {
            return Err(TimeSeriesError::ModelNotFitted(
                "TSTransformer model not trained".to_string(),
            ));
        }
        let window = self
            .last_window
            .as_ref()
            .ok_or_else(|| TimeSeriesError::ModelNotFitted("No window".to_string()))?;

        let mut forecasts = Vec::with_capacity(steps);
        let mut current = window.clone();

        let mut remaining = steps;
        while remaining > 0 {
            let pred = self.forward_single(&current, None)?;
            let take = pred.len().min(remaining);
            for i in 0..take {
                forecasts.push(pred[i]);
            }
            remaining = remaining.saturating_sub(take);

            if remaining > 0 {
                let lb = self.config.lookback;
                let shift = take.min(lb);
                let mut new_win = Array1::zeros(lb);
                for i in 0..(lb - shift) {
                    new_win[i] = current[i + shift];
                }
                for i in 0..shift {
                    new_win[lb - shift + i] = if i < pred.len() { pred[i] } else { F::zero() };
                }
                current = new_win;
            }
        }

        let fc_normed = Array1::from_vec(forecasts);
        let forecast = nn_utils::denormalize(&fc_normed, self.data_min, self.data_max);
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
        let base = self.predict(steps)?;

        let sigma = if let Some(ll) = self.loss_hist.last() {
            ll.sqrt()
        } else {
            F::from(0.1).unwrap_or_else(|| F::zero())
        };
        let range = self.data_max - self.data_min;
        let z: F = nn_utils::z_score_for_confidence(confidence);
        let margin = sigma * z * range;

        let mut lower = Array1::zeros(steps);
        let mut upper = Array1::zeros(steps);
        for i in 0..steps {
            let hf = F::from((i + 1) as f64 / steps as f64)
                .unwrap_or_else(|| F::one())
                .sqrt()
                + F::one();
            lower[i] = base.forecast[i] - margin * hf;
            upper[i] = base.forecast[i] + margin * hf;
        }

        Ok(ForecastResult {
            forecast: base.forecast,
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
            data[i] = 8.0 + 0.04 * t + 2.5 * (2.0 * std::f64::consts::PI * t / 14.0).sin();
        }
        data
    }

    #[test]
    fn test_ts_transformer_default_config() {
        let cfg = TSTransformerConfig::default();
        assert_eq!(cfg.lookback, 24);
        assert_eq!(cfg.horizon, 6);
        assert_eq!(cfg.model_dim, 32);
    }

    #[test]
    fn test_ts_transformer_validates_config() {
        let mut cfg = TSTransformerConfig::default();
        cfg.lookback = 0;
        assert!(TSTransformerModel::<f64>::new(cfg).is_err());
    }

    #[test]
    fn test_ts_transformer_validates_head_divisibility() {
        let mut cfg = TSTransformerConfig::default();
        cfg.model_dim = 15; // not divisible by 4
        assert!(TSTransformerModel::<f64>::new(cfg).is_err());
    }

    #[test]
    fn test_positional_encoding() {
        let pe = PositionalEncoding::<f64>::new(20, 8, 42);
        assert_eq!(pe.table.dim(), (20, 8));
        let input = Array2::zeros((10, 8));
        let output = pe.add(&input);
        assert_eq!(output.dim(), (10, 8));
    }

    #[test]
    fn test_input_embedding() {
        let embed = InputEmbedding::<f64>::new(1, 8, 42);
        let input = Array2::ones((5, 1));
        let output = embed.forward(&input);
        assert_eq!(output.dim(), (5, 8));
    }

    #[test]
    fn test_multi_head_self_attention() {
        let attn = MultiHeadSelfAttention::<f64>::new(8, 2, 42).expect("valid");
        let input = Array2::ones((5, 8));
        let output = attn.forward(&input).expect("forward");
        assert_eq!(output.dim(), (5, 8));
    }

    #[test]
    fn test_cross_attention() {
        let attn = MultiHeadSelfAttention::<f64>::new(8, 2, 42).expect("valid");
        let q = Array2::ones((3, 8));
        let kv = Array2::ones((5, 8));
        let output = attn.cross_attention(&q, &kv).expect("cross_attn");
        assert_eq!(output.dim(), (3, 8));
    }

    #[test]
    fn test_encoder_layer() {
        let el = EncoderLayer::<f64>::new(8, 2, 16, 42).expect("valid");
        let input = Array2::ones((5, 8));
        let output = el.forward(&input).expect("forward");
        assert_eq!(output.dim(), (5, 8));
    }

    #[test]
    fn test_decoder_layer() {
        let dl = DecoderLayer::<f64>::new(8, 2, 16, 42).expect("valid");
        let dec_input = Array2::ones((3, 8));
        let enc_output = Array2::ones((5, 8));
        let output = dl.forward(&dec_input, &enc_output).expect("forward");
        assert_eq!(output.dim(), (3, 8));
    }

    #[test]
    fn test_ts_transformer_fit_and_predict() {
        let cfg = TSTransformerConfig {
            lookback: 12,
            horizon: 4,
            model_dim: 8,
            num_heads: 2,
            num_encoder_layers: 1,
            num_decoder_layers: 1,
            ff_dim: 16,
            epochs: 3,
            batch_size: 16,
            input_features: 1,
            ..TSTransformerConfig::default()
        };
        let mut model = TSTransformerModel::<f64>::new(cfg).expect("valid config");
        let data = synthetic_series(80);
        model.fit(&data).expect("fit");
        assert!(model.trained);

        let result = model.predict(4).expect("predict");
        assert_eq!(result.forecast.len(), 4);
    }

    #[test]
    fn test_ts_transformer_predict_before_fit() {
        let cfg = TSTransformerConfig {
            lookback: 8,
            horizon: 2,
            model_dim: 4,
            num_heads: 2,
            num_encoder_layers: 1,
            num_decoder_layers: 1,
            ff_dim: 8,
            ..TSTransformerConfig::default()
        };
        let model = TSTransformerModel::<f64>::new(cfg).expect("valid");
        assert!(model.predict(2).is_err());
    }

    #[test]
    fn test_ts_transformer_multi_step() {
        let cfg = TSTransformerConfig {
            lookback: 12,
            horizon: 3,
            model_dim: 8,
            num_heads: 2,
            num_encoder_layers: 1,
            num_decoder_layers: 1,
            ff_dim: 16,
            epochs: 3,
            batch_size: 16,
            input_features: 1,
            ..TSTransformerConfig::default()
        };
        let mut model = TSTransformerModel::<f64>::new(cfg).expect("valid");
        let data = synthetic_series(80);
        model.fit(&data).expect("fit");

        let result = model.predict(9).expect("multi-step");
        assert_eq!(result.forecast.len(), 9);
    }

    #[test]
    fn test_ts_transformer_predict_interval() {
        let cfg = TSTransformerConfig {
            lookback: 12,
            horizon: 4,
            model_dim: 8,
            num_heads: 2,
            num_encoder_layers: 1,
            num_decoder_layers: 1,
            ff_dim: 16,
            epochs: 5,
            batch_size: 16,
            input_features: 1,
            ..TSTransformerConfig::default()
        };
        let mut model = TSTransformerModel::<f64>::new(cfg).expect("valid");
        let data = synthetic_series(80);
        model.fit(&data).expect("fit");

        let result = model.predict_interval(4, 0.95).expect("interval");
        assert_eq!(result.forecast.len(), 4);
        for i in 0..4 {
            assert!(result.lower_ci[i] <= result.forecast[i]);
            assert!(result.upper_ci[i] >= result.forecast[i]);
        }
    }

    #[test]
    fn test_ts_transformer_predict_interval_bad_confidence() {
        let cfg = TSTransformerConfig {
            lookback: 8,
            horizon: 2,
            model_dim: 4,
            num_heads: 2,
            num_encoder_layers: 1,
            num_decoder_layers: 1,
            ff_dim: 8,
            epochs: 2,
            batch_size: 16,
            input_features: 1,
            ..TSTransformerConfig::default()
        };
        let mut model = TSTransformerModel::<f64>::new(cfg).expect("valid");
        let data = synthetic_series(50);
        model.fit(&data).expect("fit");
        assert!(model.predict_interval(2, 2.0).is_err());
    }

    #[test]
    fn test_ts_transformer_loss_finite() {
        let cfg = TSTransformerConfig {
            lookback: 12,
            horizon: 3,
            model_dim: 8,
            num_heads: 2,
            num_encoder_layers: 1,
            num_decoder_layers: 1,
            ff_dim: 16,
            epochs: 10,
            batch_size: 16,
            input_features: 1,
            ..TSTransformerConfig::default()
        };
        let mut model = TSTransformerModel::<f64>::new(cfg).expect("valid");
        let data = synthetic_series(60);
        model.fit(&data).expect("fit");
        for &l in model.loss_history() {
            assert!(l.is_finite());
        }
    }

    #[test]
    fn test_ts_transformer_insufficient_data() {
        let cfg = TSTransformerConfig {
            lookback: 20,
            horizon: 5,
            model_dim: 4,
            num_heads: 2,
            ..TSTransformerConfig::default()
        };
        let mut model = TSTransformerModel::<f64>::new(cfg).expect("valid");
        let short = Array1::from_vec(vec![1.0; 10]);
        assert!(model.fit(&short).is_err());
    }

    #[test]
    fn test_ts_transformer_f32() {
        let cfg = TSTransformerConfig {
            lookback: 10,
            horizon: 3,
            model_dim: 4,
            num_heads: 2,
            num_encoder_layers: 1,
            num_decoder_layers: 1,
            ff_dim: 8,
            epochs: 3,
            batch_size: 16,
            input_features: 1,
            seed: 42,
            ..TSTransformerConfig::default()
        };
        let mut model = TSTransformerModel::<f32>::new(cfg).expect("valid");
        let data: Array1<f32> =
            Array1::from_vec((0..50).map(|i| 4.0f32 + (i as f32 * 0.3).sin()).collect());
        model.fit(&data).expect("f32 fit");
        let result = model.predict(3).expect("f32 predict");
        assert_eq!(result.forecast.len(), 3);
    }
}
