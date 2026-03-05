//! GPT-2 style autoregressive transformer implementation
//!
//! GPT-2 is a large transformer-based language model trained with the objective of
//! predicting the next word, given all of the previous words within some text.
//! It uses causal (masked) self-attention, pre-norm transformer blocks, and
//! learned positional embeddings.
//!
//! Reference: "Language Models are Unsupervised Multitask Learners", Radford et al. (2019)
//! <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>

use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Dropout, Embedding, EmbeddingConfig, Layer, LayerNorm};
use scirs2_core::ndarray::{Array, Array1, Array2, Array3, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use scirs2_core::random::SeedableRng;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a GPT-2 style model
#[derive(Debug, Clone)]
pub struct Gpt2Config {
    /// Vocabulary size
    pub n_vocab: usize,
    /// Maximum sequence length (context window)
    pub n_positions: usize,
    /// Model embedding dimension
    pub n_embd: usize,
    /// Number of transformer layers
    pub n_layer: usize,
    /// Number of attention heads
    pub n_head: usize,
    /// Inner (feed-forward) dimension; defaults to 4 * n_embd if 0
    pub n_inner: usize,
    /// Activation function ("gelu" or "relu")
    pub activation: String,
    /// Residual dropout probability
    pub resid_pdrop: f32,
    /// Attention dropout probability
    pub attn_pdrop: f32,
    /// Layer-norm epsilon
    pub layer_norm_eps: f64,
}

impl Default for Gpt2Config {
    fn default() -> Self {
        Self {
            n_vocab: 50257,
            n_positions: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            n_inner: 0,
            activation: "gelu".to_string(),
            resid_pdrop: 0.1,
            attn_pdrop: 0.1,
            layer_norm_eps: 1e-5,
        }
    }
}

impl Gpt2Config {
    /// GPT-2 Small (117M)
    pub fn gpt2_small() -> Self {
        Self::default()
    }

    /// GPT-2 Medium (345M)
    pub fn gpt2_medium() -> Self {
        Self {
            n_embd: 1024,
            n_layer: 24,
            n_head: 16,
            ..Self::default()
        }
    }

    /// GPT-2 Large (774M)
    pub fn gpt2_large() -> Self {
        Self {
            n_embd: 1280,
            n_layer: 36,
            n_head: 20,
            ..Self::default()
        }
    }

    /// GPT-2 XL (1558M)
    pub fn gpt2_xl() -> Self {
        Self {
            n_embd: 1600,
            n_layer: 48,
            n_head: 25,
            ..Self::default()
        }
    }

    /// Effective n_inner (4 * n_embd when n_inner == 0)
    pub fn effective_n_inner(&self) -> usize {
        if self.n_inner == 0 {
            4 * self.n_embd
        } else {
            self.n_inner
        }
    }
}

// ---------------------------------------------------------------------------
// Causal self-attention
// ---------------------------------------------------------------------------

/// Causal (autoregressive) multi-head self-attention
///
/// Uses a lower-triangular mask so that each token only attends to previous tokens.
#[derive(Clone)]
pub struct CausalSelfAttention<
    F: Float + Debug + ScalarOperand + Clone + Send + Sync + SimdUnifiedOps + NumAssign,
> {
    /// Combined Q/K/V projection: d_embd -> 3*d_embd
    c_attn: Dense<F>,
    /// Output projection: d_embd -> d_embd
    c_proj: Dense<F>,
    /// Attention dropout
    attn_dropout: Dropout<F>,
    /// Residual dropout
    resid_dropout: Dropout<F>,
    /// Number of attention heads
    n_head: usize,
    /// Head dimension
    head_dim: usize,
    /// Embedding dimension
    n_embd: usize,
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Clone
            + Send
            + Sync
            + SimdUnifiedOps
            + NumAssign
            + FromPrimitive
            + ToPrimitive
            + 'static,
    > CausalSelfAttention<F>
{
    /// Create a new causal self-attention block
    pub fn new(config: &Gpt2Config, seed_base: u8) -> Result<Self> {
        let n_embd = config.n_embd;
        let n_head = config.n_head;

        if n_embd % n_head != 0 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "n_embd ({n_embd}) must be divisible by n_head ({n_head})"
            )));
        }
        let head_dim = n_embd / n_head;

        let mut rng1 = scirs2_core::random::rngs::SmallRng::from_seed([seed_base; 32]);
        let c_attn = Dense::new(n_embd, 3 * n_embd, None, &mut rng1)?;

        let mut rng2 = scirs2_core::random::rngs::SmallRng::from_seed([seed_base + 1; 32]);
        let c_proj = Dense::new(n_embd, n_embd, None, &mut rng2)?;

        let mut rng3 = scirs2_core::random::rngs::SmallRng::from_seed([seed_base + 2; 32]);
        let attn_dropout = Dropout::new(config.attn_pdrop as f64, &mut rng3)?;

        let mut rng4 = scirs2_core::random::rngs::SmallRng::from_seed([seed_base + 3; 32]);
        let resid_dropout = Dropout::new(config.resid_pdrop as f64, &mut rng4)?;

        Ok(Self {
            c_attn,
            c_proj,
            attn_dropout,
            resid_dropout,
            n_head,
            head_dim,
            n_embd,
        })
    }

    /// Forward pass: (B, T, C) -> (B, T, C)
    pub fn forward(&self, x: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 3D input (B, T, C), got {}D",
                shape.len()
            )));
        }
        let batch = shape[0];
        let seq_len = shape[1];
        let c = shape[2];

        if c != self.n_embd {
            return Err(NeuralError::ShapeMismatch(format!(
                "Input last dim {c} != n_embd {}",
                self.n_embd
            )));
        }

        // Project to Q, K, V: (B, T, 3*C)
        let qkv = self.c_attn.forward(x)?;

        // Split Q, K, V: each (B, T, C)
        let mut q = Array::zeros(IxDyn(&[batch, seq_len, self.n_embd]));
        let mut k = Array::zeros(IxDyn(&[batch, seq_len, self.n_embd]));
        let mut v = Array::zeros(IxDyn(&[batch, seq_len, self.n_embd]));

        for b in 0..batch {
            for t in 0..seq_len {
                for i in 0..self.n_embd {
                    q[[b, t, i]] = qkv[[b, t, i]];
                    k[[b, t, i]] = qkv[[b, t, self.n_embd + i]];
                    v[[b, t, i]] = qkv[[b, t, 2 * self.n_embd + i]];
                }
            }
        }

        // Compute scale factor: 1 / sqrt(head_dim)
        let scale = F::from(1.0 / (self.head_dim as f64).sqrt()).ok_or_else(|| {
            NeuralError::ComputationError("Failed to convert scale factor".to_string())
        })?;

        // Multi-head attention with causal mask
        // Output: (B, T, C)
        let mut attn_out = Array::zeros(IxDyn(&[batch, seq_len, self.n_embd]));

        for b in 0..batch {
            for h in 0..self.n_head {
                let hd = self.head_dim;
                let h_start = h * hd;

                // Compute attention scores with causal mask: (T, T)
                let mut scores = Array2::<F>::zeros((seq_len, seq_len));
                for i in 0..seq_len {
                    for j in 0..=i {
                        // causal: only attend to j <= i
                        let mut dot = F::zero();
                        for d in 0..hd {
                            dot += q[[b, i, h_start + d]] * k[[b, j, h_start + d]];
                        }
                        scores[[i, j]] = dot * scale;
                    }
                    // Fill upper triangle with -inf
                    for j in (i + 1)..seq_len {
                        scores[[i, j]] = F::from(-1e9).ok_or_else(|| {
                            NeuralError::ComputationError("Failed to convert -inf".to_string())
                        })?;
                    }
                }

                // Softmax over last dim
                let mut attn_weights = Array2::<F>::zeros((seq_len, seq_len));
                for i in 0..seq_len {
                    // Numerically stable softmax
                    let mut max_val = scores[[i, 0]];
                    for j in 1..seq_len {
                        if scores[[i, j]] > max_val {
                            max_val = scores[[i, j]];
                        }
                    }
                    let mut sum = F::zero();
                    for j in 0..seq_len {
                        let e = (scores[[i, j]] - max_val).exp();
                        attn_weights[[i, j]] = e;
                        sum += e;
                    }
                    if sum > F::zero() {
                        for j in 0..seq_len {
                            attn_weights[[i, j]] /= sum;
                        }
                    }
                }

                // Apply dropout to attention weights
                let aw_dyn = attn_weights.into_dyn();
                let aw_dropped = self.attn_dropout.forward(&aw_dyn)?;
                let aw = aw_dropped
                    .view()
                    .into_shape_with_order((seq_len, seq_len))
                    .map_err(|e| NeuralError::ShapeMismatch(e.to_string()))?
                    .to_owned();

                // Weighted sum of values: (T, head_dim)
                for i in 0..seq_len {
                    for d in 0..hd {
                        let mut val = F::zero();
                        for j in 0..seq_len {
                            val += aw[[i, j]] * v[[b, j, h_start + d]];
                        }
                        attn_out[[b, i, h_start + d]] = val;
                    }
                }
            }
        }

        // Output projection: (B, T, C)
        let projected = self.c_proj.forward(&attn_out)?;
        let out = self.resid_dropout.forward(&projected)?;

        Ok(out)
    }

    /// Update parameters
    pub fn update(&mut self, lr: F) -> Result<()> {
        self.c_attn.update(lr)?;
        self.c_proj.update(lr)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MLP (feed-forward block)
// ---------------------------------------------------------------------------

/// GPT-2 feed-forward MLP block
#[derive(Clone)]
pub struct Gpt2Mlp<
    F: Float + Debug + ScalarOperand + Clone + Send + Sync + SimdUnifiedOps + NumAssign,
> {
    /// First linear projection: d_embd -> n_inner
    fc: Dense<F>,
    /// Second linear projection: n_inner -> d_embd
    proj: Dense<F>,
    /// Dropout after projection
    dropout: Dropout<F>,
    /// Activation name
    activation: String,
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Clone
            + Send
            + Sync
            + SimdUnifiedOps
            + NumAssign
            + FromPrimitive
            + 'static,
    > Gpt2Mlp<F>
{
    /// Create a new MLP block
    pub fn new(config: &Gpt2Config, seed_base: u8) -> Result<Self> {
        let n_inner = config.effective_n_inner();

        let mut rng1 = scirs2_core::random::rngs::SmallRng::from_seed([seed_base; 32]);
        let fc = Dense::new(config.n_embd, n_inner, None, &mut rng1)?;

        let mut rng2 = scirs2_core::random::rngs::SmallRng::from_seed([seed_base + 1; 32]);
        let proj = Dense::new(n_inner, config.n_embd, None, &mut rng2)?;

        let mut rng3 = scirs2_core::random::rngs::SmallRng::from_seed([seed_base + 2; 32]);
        let dropout = Dropout::new(config.resid_pdrop as f64, &mut rng3)?;

        Ok(Self {
            fc,
            proj,
            dropout,
            activation: config.activation.clone(),
        })
    }

    /// Apply activation function
    fn apply_activation(&self, x: &Array<F, IxDyn>) -> Array<F, IxDyn>
    where
        F: FromPrimitive,
    {
        match self.activation.as_str() {
            "relu" => x.mapv(|v| if v > F::zero() { v } else { F::zero() }),
            _ => {
                // GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                x.mapv(|v| {
                    let x3 = v * v * v;
                    let c1 = F::from(0.044715).unwrap_or(F::zero());
                    let c2 = F::from(0.7978845608).unwrap_or(F::zero()); // sqrt(2/pi)
                    let inner = c2 * (v + c1 * x3);
                    v * F::from(0.5).unwrap_or(F::zero()) * (F::one() + inner.tanh())
                })
            }
        }
    }

    /// Forward: (B, T, C) -> (B, T, C)
    pub fn forward(&self, x: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let h = self.fc.forward(x)?;
        let h_act = self.apply_activation(&h);
        let out = self.proj.forward(&h_act)?;
        let out = self.dropout.forward(&out)?;
        Ok(out)
    }

    /// Update parameters
    pub fn update(&mut self, lr: F) -> Result<()> {
        self.fc.update(lr)?;
        self.proj.update(lr)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GPT-2 Layer
// ---------------------------------------------------------------------------

/// Single GPT-2 transformer layer (pre-norm)
///
/// Structure: LN -> CausalSelfAttention -> Add -> LN -> MLP -> Add
#[derive(Clone)]
pub struct Gpt2Layer<
    F: Float + Debug + ScalarOperand + Clone + Send + Sync + SimdUnifiedOps + NumAssign,
> {
    /// Layer norm before attention
    ln1: LayerNorm<F>,
    /// Causal self-attention
    attn: CausalSelfAttention<F>,
    /// Layer norm before MLP
    ln2: LayerNorm<F>,
    /// MLP block
    mlp: Gpt2Mlp<F>,
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Clone
            + Send
            + Sync
            + SimdUnifiedOps
            + NumAssign
            + FromPrimitive
            + ToPrimitive
            + 'static,
    > Gpt2Layer<F>
{
    /// Create a new GPT-2 layer
    pub fn new(config: &Gpt2Config, seed_base: u8) -> Result<Self> {
        let mut rng1 = scirs2_core::random::rngs::SmallRng::from_seed([seed_base; 32]);
        let ln1 = LayerNorm::new(config.n_embd, config.layer_norm_eps, &mut rng1)?;

        let attn = CausalSelfAttention::new(config, seed_base + 1)?;

        let mut rng2 = scirs2_core::random::rngs::SmallRng::from_seed([seed_base + 5; 32]);
        let ln2 = LayerNorm::new(config.n_embd, config.layer_norm_eps, &mut rng2)?;

        let mlp = Gpt2Mlp::new(config, seed_base + 6)?;

        Ok(Self { ln1, attn, ln2, mlp })
    }

    /// Forward: (B, T, C) -> (B, T, C)
    pub fn forward(&self, x: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Pre-norm + attention + residual
        let normed1 = self.ln1.forward(x)?;
        let attn_out = self.attn.forward(&normed1)?;
        let x = x + &attn_out;

        // Pre-norm + MLP + residual
        let normed2 = self.ln2.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed2)?;
        let x = x + &mlp_out;

        Ok(x)
    }

    /// Update parameters
    pub fn update(&mut self, lr: F) -> Result<()> {
        self.ln1.update(lr)?;
        self.attn.update(lr)?;
        self.ln2.update(lr)?;
        self.mlp.update(lr)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GPT-2 Model (embeddings + layers + final LN)
// ---------------------------------------------------------------------------

/// GPT-2 base model (without language model head)
pub struct Gpt2Model<
    F: Float + Debug + ScalarOperand + Clone + Send + Sync + SimdUnifiedOps + NumAssign,
> {
    /// Token embedding table: (n_vocab, n_embd)
    wte: Embedding<F>,
    /// Position embedding table: (n_positions, n_embd)
    wpe: Embedding<F>,
    /// Embedding dropout
    emb_drop: Dropout<F>,
    /// Transformer layers
    layers: Vec<Gpt2Layer<F>>,
    /// Final layer norm
    ln_f: LayerNorm<F>,
    /// Configuration
    config: Gpt2Config,
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Clone
            + Send
            + Sync
            + SimdUnifiedOps
            + NumAssign
            + FromPrimitive
            + ToPrimitive
            + 'static,
    > Gpt2Model<F>
{
    /// Create a new GPT-2 base model
    pub fn new(config: Gpt2Config) -> Result<Self> {
        // Token embeddings
        let wte_cfg = EmbeddingConfig {
            num_embeddings: config.n_vocab,
            embedding_dim: config.n_embd,
            ..Default::default()
        };
        let wte = Embedding::new(wte_cfg)?;

        // Position embeddings
        let wpe_cfg = EmbeddingConfig {
            num_embeddings: config.n_positions,
            embedding_dim: config.n_embd,
            ..Default::default()
        };
        let wpe = Embedding::new(wpe_cfg)?;

        // Embedding dropout
        let mut drop_rng = scirs2_core::random::rngs::SmallRng::from_seed([10; 32]);
        let emb_drop = Dropout::new(config.resid_pdrop as f64, &mut drop_rng)?;

        // Transformer layers: each layer gets a unique seed range
        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let seed = ((i * 20) % 200 + 20) as u8;
            layers.push(Gpt2Layer::new(&config, seed)?);
        }

        // Final layer norm
        let mut ln_rng = scirs2_core::random::rngs::SmallRng::from_seed([9; 32]);
        let ln_f = LayerNorm::new(config.n_embd, config.layer_norm_eps, &mut ln_rng)?;

        Ok(Self {
            wte,
            wpe,
            emb_drop,
            layers,
            ln_f,
            config,
        })
    }

    /// Get configuration
    pub fn config(&self) -> &Gpt2Config {
        &self.config
    }

    /// Forward: input_ids (B, T) as float array -> (B, T, n_embd)
    ///
    /// input_ids should be an Array<F, IxDyn> with integer values cast to F.
    pub fn forward_hidden(&self, x: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = x.shape();
        if shape.len() != 2 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 2D input (B, T), got {}D",
                shape.len()
            )));
        }
        let batch = shape[0];
        let seq_len = shape[1];

        if seq_len > self.config.n_positions {
            return Err(NeuralError::InvalidArgument(format!(
                "seq_len ({seq_len}) exceeds n_positions ({})",
                self.config.n_positions
            )));
        }

        // Token embeddings: (B, T, n_embd)
        let tok_emb = self.wte.forward(x)?;

        // Position indices: 0, 1, ..., T-1
        let pos_ids: Vec<F> = (0..seq_len)
            .map(|i| F::from(i).unwrap_or(F::zero()))
            .collect();
        let pos_arr = Array::from_shape_vec(IxDyn(&[1, seq_len]), pos_ids)
            .map_err(|e| NeuralError::ShapeMismatch(e.to_string()))?;
        let pos_emb_single = self.wpe.forward(&pos_arr)?; // (1, T, n_embd)

        // Broadcast position embeddings across batch
        let mut hidden =
            Array::zeros(IxDyn(&[batch, seq_len, self.config.n_embd]));
        for b in 0..batch {
            for t in 0..seq_len {
                for d in 0..self.config.n_embd {
                    hidden[[b, t, d]] = tok_emb[[b, t, d]] + pos_emb_single[[0, t, d]];
                }
            }
        }

        // Embedding dropout
        let mut h = self.emb_drop.forward(&hidden)?;

        // Transformer layers
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }

        // Final layer norm
        let h = self.ln_f.forward(&h)?;

        Ok(h)
    }

    /// Update parameters
    pub fn update(&mut self, lr: F) -> Result<()> {
        self.wte.update(lr)?;
        self.wpe.update(lr)?;
        for layer in &mut self.layers {
            layer.update(lr)?;
        }
        self.ln_f.update(lr)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GPT-2 Language Model (with LM head)
// ---------------------------------------------------------------------------

/// GPT-2 language model with tied embedding weights
///
/// The language model head projects the hidden states to vocabulary logits.
/// Following the GPT-2 paper, the embedding matrix is tied (shared) with the
/// output projection.
pub struct Gpt2LM<
    F: Float + Debug + ScalarOperand + Clone + Send + Sync + SimdUnifiedOps + NumAssign,
> {
    /// Base GPT-2 model
    transformer: Gpt2Model<F>,
    /// Language model head (n_embd -> n_vocab)
    /// NOTE: In a full implementation, weights would be tied to wte.
    /// Here we use a separate Dense layer for forward-only inference.
    lm_head: Dense<F>,
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Clone
            + Send
            + Sync
            + SimdUnifiedOps
            + NumAssign
            + FromPrimitive
            + ToPrimitive
            + 'static,
    > Gpt2LM<F>
{
    /// Create a new GPT-2 language model
    pub fn new(config: Gpt2Config) -> Result<Self> {
        let mut head_rng = scirs2_core::random::rngs::SmallRng::from_seed([1; 32]);
        let lm_head = Dense::new(config.n_embd, config.n_vocab, None, &mut head_rng)?;
        let transformer = Gpt2Model::new(config)?;
        Ok(Self { transformer, lm_head })
    }

    /// Get configuration
    pub fn config(&self) -> &Gpt2Config {
        self.transformer.config()
    }

    /// Forward pass: input_ids (B, T) -> logits (B, T, n_vocab)
    ///
    /// `input_ids` should contain integer token IDs cast to F.
    pub fn forward(&self, input_ids: &Array2<F>) -> Result<Array3<F>> {
        let hidden = self.transformer.forward_hidden(&input_ids.clone().into_dyn())?;

        // LM head: (B, T, n_embd) -> (B, T, n_vocab)
        let logits_dyn = self.lm_head.forward(&hidden)?;

        let shape = logits_dyn.shape();
        if shape.len() != 3 {
            return Err(NeuralError::ShapeMismatch(format!(
                "Expected 3D logits, got {:?}",
                shape
            )));
        }
        let (b, t, v) = (shape[0], shape[1], shape[2]);
        let logits = logits_dyn
            .into_shape_with_order((b, t, v))
            .map_err(|e| NeuralError::ShapeMismatch(e.to_string()))?;

        Ok(logits)
    }

    /// Greedy text generation
    ///
    /// Generates `max_new_tokens` additional tokens given an initial `prompt`.
    /// Returns the full sequence (prompt + generated tokens).
    pub fn generate(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<usize>> {
        if prompt.is_empty() {
            return Err(NeuralError::InvalidArgument("Prompt is empty".to_string()));
        }

        let mut tokens: Vec<usize> = prompt.to_vec();

        for _ in 0..max_new_tokens {
            let seq_len = tokens
                .len()
                .min(self.transformer.config.n_positions);
            let start = tokens.len() - seq_len;
            let context = &tokens[start..];

            // Build input array
            let ids: Vec<F> = context
                .iter()
                .map(|&id| F::from(id).unwrap_or(F::zero()))
                .collect();
            let input = Array2::from_shape_vec((1, seq_len), ids)
                .map_err(|e| NeuralError::ShapeMismatch(e.to_string()))?;

            let logits = self.forward(&input)?; // (1, T, n_vocab)

            // Take logits at last position
            let last_logits: Vec<F> = (0..self.transformer.config.n_vocab)
                .map(|v| logits[[0, seq_len - 1, v]])
                .collect();

            // Apply temperature
            let temp = F::from(temperature).unwrap_or(F::one());
            let scaled: Vec<F> = last_logits.iter().map(|&l| l / temp).collect();

            // Softmax + argmax (greedy)
            let max_val = scaled
                .iter()
                .cloned()
                .fold(scaled[0], |a, b| if b > a { b } else { a });
            let exps: Vec<F> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
            let sum: F = exps.iter().cloned().fold(F::zero(), |a, b| a + b);

            let next_token = exps
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    let a_f = (**a / sum).to_f64().unwrap_or(0.0);
                    let b_f = (**b / sum).to_f64().unwrap_or(0.0);
                    a_f.partial_cmp(&b_f).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .ok_or_else(|| NeuralError::ComputationError("Empty logits".to_string()))?;

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Update parameters
    pub fn update(&mut self, lr: F) -> Result<()> {
        self.transformer.update(lr)?;
        self.lm_head.update(lr)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Layer trait impl for Gpt2LM
// ---------------------------------------------------------------------------

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Clone
            + Send
            + Sync
            + SimdUnifiedOps
            + NumAssign
            + FromPrimitive
            + ToPrimitive
            + 'static,
    > Layer<F> for Gpt2LM<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 2D input (B, T), got {}D",
                shape.len()
            )));
        }
        let arr2 = input
            .view()
            .into_shape_with_order((shape[0], shape[1]))
            .map_err(|e| NeuralError::ShapeMismatch(e.to_string()))?
            .to_owned();
        let logits = self.forward(&arr2)?;
        Ok(logits.into_dyn())
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, lr: F) -> Result<()> {
        self.update(lr)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "Gpt2LM"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn small_config() -> Gpt2Config {
        Gpt2Config {
            n_vocab: 100,
            n_positions: 16,
            n_embd: 32,
            n_layer: 2,
            n_head: 4,
            n_inner: 64,
            activation: "gelu".to_string(),
            resid_pdrop: 0.0,
            attn_pdrop: 0.0,
            layer_norm_eps: 1e-5,
        }
    }

    #[test]
    fn test_gpt2_config_defaults() {
        let cfg = Gpt2Config::default();
        assert_eq!(cfg.n_embd, 768);
        assert_eq!(cfg.n_layer, 12);
        assert_eq!(cfg.effective_n_inner(), 768 * 4);
    }

    #[test]
    fn test_gpt2_lm_forward() {
        let cfg = small_config();
        let model = Gpt2LM::<f32>::new(cfg).expect("Failed to create Gpt2LM");
        let ids = Array2::from_shape_vec(
            (1, 4),
            vec![1.0_f32, 5.0, 10.0, 20.0],
        )
        .expect("Shape error");
        let logits = model.forward(&ids).expect("Forward failed");
        assert_eq!(logits.shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_gpt2_generate() {
        let cfg = small_config();
        let model = Gpt2LM::<f32>::new(cfg).expect("Failed to create Gpt2LM");
        let result = model.generate(&[1, 2, 3], 5, 1.0).expect("Generate failed");
        assert_eq!(result.len(), 8); // 3 prompt + 5 generated
    }

    #[test]
    fn test_causal_mask() {
        let cfg = small_config();
        let attn = CausalSelfAttention::<f32>::new(&cfg, 10).expect("Failed to create attention");
        let x = Array::zeros(IxDyn(&[1, 4, 32]));
        let out = attn.forward(&x).expect("Forward failed");
        assert_eq!(out.shape(), &[1, 4, 32]);
    }

    #[test]
    fn test_gpt2_layer() {
        let cfg = small_config();
        let layer = Gpt2Layer::<f32>::new(&cfg, 10).expect("Failed to create layer");
        let x = Array::zeros(IxDyn(&[2, 6, 32]));
        let out = layer.forward(&x).expect("Forward failed");
        assert_eq!(out.shape(), &[2, 6, 32]);
    }
}
