//! Swin Transformer implementation
//!
//! The Swin Transformer is a hierarchical vision transformer that computes
//! self-attention within local windows and shifts windows between layers to
//! allow cross-window connections. It serves as a general-purpose backbone for
//! computer vision tasks.
//!
//! Reference: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
//! Liu et al. (2021) <https://arxiv.org/abs/2103.14030>

use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Dropout, Layer, LayerNorm};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::simd_ops::SimdUnifiedOps;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Swin Transformer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwinConfig {
    /// Input image size (height, width)
    pub img_size: (usize, usize),
    /// Patch size (height, width) for the patch partition step
    pub patch_size: (usize, usize),
    /// Number of input channels (e.g., 3 for RGB)
    pub in_channels: usize,
    /// Number of output classes for classification
    pub num_classes: usize,
    /// Base embedding dimension after patch embedding
    pub embed_dim: usize,
    /// Number of Swin Transformer blocks in each stage
    pub depths: Vec<usize>,
    /// Number of attention heads in each stage
    pub num_heads: Vec<usize>,
    /// Window size for local self-attention
    pub window_size: usize,
    /// MLP expansion ratio
    pub mlp_ratio: f64,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Attention dropout rate
    pub attention_dropout_rate: f64,
    /// Drop path (stochastic depth) rate
    pub drop_path_rate: f64,
    /// QKV bias
    pub qkv_bias: bool,
}

impl SwinConfig {
    /// Swin-Tiny: a small model suitable for resource-constrained scenarios
    pub fn swin_tiny(img_size: (usize, usize), num_classes: usize) -> Self {
        Self {
            img_size,
            patch_size: (4, 4),
            in_channels: 3,
            num_classes,
            embed_dim: 96,
            depths: vec![2, 2, 6, 2],
            num_heads: vec![3, 6, 12, 24],
            window_size: 7,
            mlp_ratio: 4.0,
            dropout_rate: 0.0,
            attention_dropout_rate: 0.0,
            drop_path_rate: 0.2,
            qkv_bias: true,
        }
    }

    /// Swin-Small
    pub fn swin_small(img_size: (usize, usize), num_classes: usize) -> Self {
        Self {
            img_size,
            patch_size: (4, 4),
            in_channels: 3,
            num_classes,
            embed_dim: 96,
            depths: vec![2, 2, 18, 2],
            num_heads: vec![3, 6, 12, 24],
            window_size: 7,
            mlp_ratio: 4.0,
            dropout_rate: 0.0,
            attention_dropout_rate: 0.0,
            drop_path_rate: 0.3,
            qkv_bias: true,
        }
    }

    /// Swin-Base
    pub fn swin_base(img_size: (usize, usize), num_classes: usize) -> Self {
        Self {
            img_size,
            patch_size: (4, 4),
            in_channels: 3,
            num_classes,
            embed_dim: 128,
            depths: vec![2, 2, 18, 2],
            num_heads: vec![4, 8, 16, 32],
            window_size: 7,
            mlp_ratio: 4.0,
            dropout_rate: 0.0,
            attention_dropout_rate: 0.0,
            drop_path_rate: 0.5,
            qkv_bias: true,
        }
    }

    /// Swin-Large
    pub fn swin_large(img_size: (usize, usize), num_classes: usize) -> Self {
        Self {
            img_size,
            patch_size: (4, 4),
            in_channels: 3,
            num_classes,
            embed_dim: 192,
            depths: vec![2, 2, 18, 2],
            num_heads: vec![6, 12, 24, 48],
            window_size: 7,
            mlp_ratio: 4.0,
            dropout_rate: 0.0,
            attention_dropout_rate: 0.0,
            drop_path_rate: 0.5,
            qkv_bias: true,
        }
    }

    /// Return the number of stages (same as depths.len())
    pub fn num_stages(&self) -> usize {
        self.depths.len()
    }

    /// Feature dimension at a given stage
    pub fn stage_dim(&self, stage: usize) -> usize {
        self.embed_dim * (1 << stage)
    }

    /// Number of patches (H/patch_h * W/patch_w)
    pub fn num_patches(&self) -> (usize, usize) {
        (
            self.img_size.0 / self.patch_size.0,
            self.img_size.1 / self.patch_size.1,
        )
    }
}

// ---------------------------------------------------------------------------
// Window Multi-Head Self-Attention (W-MSA / SW-MSA)
// ---------------------------------------------------------------------------

/// Window-based multi-head self-attention with optional shift
///
/// Each window attends only to tokens within itself. For shifted windows,
/// a cyclic shift is applied before partitioning so tokens can communicate
/// across window boundaries.
#[derive(Debug, Clone)]
pub struct WindowAttention<F: Float + Debug + Send + Sync + NumAssign + ScalarOperand> {
    /// Embedding dimension
    dim: usize,
    /// Window size
    window_size: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Per-head scaling factor
    scale: F,
    /// QKV projection: [dim, 3 * dim]
    qkv_weight: Array<F, IxDyn>,
    /// QKV bias: [3 * dim]
    qkv_bias: Option<Array<F, IxDyn>>,
    /// Output projection: [dim, dim]
    proj_weight: Array<F, IxDyn>,
    /// Relative position bias table: [(2*win-1)^2, num_heads]
    rel_pos_bias: Array<F, IxDyn>,
    /// Attention dropout probability (stored for reference)
    attn_drop_prob: f64,
    /// Projection dropout
    proj_drop: Dropout<F>,
}

impl<
        F: Float
            + Debug
            + Send
            + Sync
            + NumAssign
            + ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + 'static,
    > WindowAttention<F>
{
    /// Create a new window attention module
    pub fn new<R: Rng + Clone + Send + Sync + 'static>(
        dim: usize,
        window_size: usize,
        num_heads: usize,
        attn_drop: f64,
        proj_drop: f64,
        use_bias: bool,
        rng: &mut R,
    ) -> Result<Self> {
        if dim % num_heads != 0 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "dim ({dim}) must be divisible by num_heads ({num_heads})"
            )));
        }
        let head_dim = dim / num_heads;
        let scale = F::from(1.0 / (head_dim as f64).sqrt()).ok_or_else(|| {
            NeuralError::InvalidArchitecture("scale conversion failed".to_string())
        })?;

        // Xavier uniform initialization helper
        let xavier_init = |rows: usize, cols: usize, rng: &mut R| -> Array<F, IxDyn> {
            let limit = (6.0_f64 / (rows + cols) as f64).sqrt();
            let data: Vec<F> = (0..rows * cols)
                .map(|_| {
                    let v = rng.random::<f64>() * 2.0 * limit - limit;
                    F::from(v).unwrap_or(F::zero())
                })
                .collect();
            Array::from_shape_vec(IxDyn(&[rows, cols]), data)
                .unwrap_or_else(|_| Array::zeros(IxDyn(&[rows, cols])))
        };

        let qkv_weight = xavier_init(dim, 3 * dim, rng);
        let qkv_bias = if use_bias {
            Some(Array::zeros(IxDyn(&[3 * dim])))
        } else {
            None
        };
        let proj_weight = xavier_init(dim, dim, rng);

        // Relative position bias table: (2*W-1)^2 entries per head
        let bias_table_size = (2 * window_size - 1) * (2 * window_size - 1);
        let rel_pos_bias = Array::zeros(IxDyn(&[bias_table_size, num_heads]));

        let proj_drop_layer = Dropout::<F>::new(proj_drop, rng).map_err(|e| {
            NeuralError::InvalidArchitecture(format!("Dropout creation failed: {e}"))
        })?;

        Ok(Self {
            dim,
            window_size,
            num_heads,
            scale,
            qkv_weight,
            qkv_bias,
            proj_weight,
            rel_pos_bias,
            attn_drop_prob: attn_drop,
            proj_drop: proj_drop_layer,
        })
    }

    /// Compute window attention on a batch of windows
    ///
    /// `x` shape: [num_windows * batch, window_size^2, dim]
    pub fn forward(&self, x: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InvalidArgument(format!(
                "WindowAttention expects 3D input [B*nW, Wh*Ww, C], got {:?}",
                shape
            )));
        }
        let bw = shape[0]; // batch * num_windows
        let n = shape[1];  // window_size^2
        let c = shape[2];  // dim

        if c != self.dim {
            return Err(NeuralError::InvalidArgument(format!(
                "Expected channel dim {}, got {c}",
                self.dim
            )));
        }

        // QKV projection: [bw, n, 3*dim]
        let x_2d = x
            .view()
            .into_shape_with_order(IxDyn(&[bw * n, c]))
            .map_err(|e| NeuralError::InvalidArgument(format!("reshape error: {e}")))?
            .to_owned();
        let qkv_2d = matmul_2d(&x_2d, &self.qkv_weight)?; // [bw*n, 3*dim]

        // Add bias if present
        let qkv_2d = if let Some(ref bias) = self.qkv_bias {
            let mut out = qkv_2d;
            for col in 0..(3 * self.dim) {
                for row in 0..(bw * n) {
                    out[[row, col]] = out[[row, col]] + bias[[col]];
                }
            }
            out
        } else {
            qkv_2d
        };

        // Reshape to [bw, n, 3, num_heads, head_dim] then split Q,K,V
        let head_dim = self.dim / self.num_heads;
        let mut q = Array::zeros(IxDyn(&[bw, self.num_heads, n, head_dim]));
        let mut k = Array::zeros(IxDyn(&[bw, self.num_heads, n, head_dim]));
        let mut v = Array::zeros(IxDyn(&[bw, self.num_heads, n, head_dim]));

        for b in 0..bw {
            for i in 0..n {
                let row = b * n + i;
                for h in 0..self.num_heads {
                    for d in 0..head_dim {
                        q[[b, h, i, d]] = qkv_2d[[row, h * head_dim + d]];
                        k[[b, h, i, d]] = qkv_2d[[row, self.dim + h * head_dim + d]];
                        v[[b, h, i, d]] = qkv_2d[[row, 2 * self.dim + h * head_dim + d]];
                    }
                }
            }
        }

        // Scale Q
        let scale = self.scale;
        q.mapv_inplace(|x| x * scale);

        // Attention scores: [bw, num_heads, n, n]
        let mut attn = Array::zeros(IxDyn(&[bw, self.num_heads, n, n]));
        for b in 0..bw {
            for h in 0..self.num_heads {
                for i in 0..n {
                    for j in 0..n {
                        let mut dot = F::zero();
                        for d in 0..head_dim {
                            dot = dot + q[[b, h, i, d]] * k[[b, h, j, d]];
                        }
                        attn[[b, h, i, j]] = dot;
                    }
                }
            }
        }

        // Add relative position bias: broadcast over batch and windows
        // bias table is indexed by (row_offset + W-1) * (2W-1) + (col_offset + W-1)
        let w = self.window_size;
        for i in 0..n {
            let ri = i / w;
            let ci = i % w;
            for j in 0..n {
                let rj = j / w;
                let cj = j % w;
                let row_off = (ri as isize) - (rj as isize) + (w as isize - 1);
                let col_off = (ci as isize) - (cj as isize) + (w as isize - 1);
                let bias_idx = (row_off * (2 * w as isize - 1) + col_off) as usize;
                for b in 0..bw {
                    for h in 0..self.num_heads {
                        attn[[b, h, i, j]] = attn[[b, h, i, j]] + self.rel_pos_bias[[bias_idx, h]];
                    }
                }
            }
        }

        // Softmax over last dim
        softmax_last_dim(&mut attn)?;

        // Weighted sum: [bw, num_heads, n, head_dim]
        let mut out = Array::zeros(IxDyn(&[bw, self.num_heads, n, head_dim]));
        for b in 0..bw {
            for h in 0..self.num_heads {
                for i in 0..n {
                    for d in 0..head_dim {
                        let mut acc = F::zero();
                        for j in 0..n {
                            acc = acc + attn[[b, h, i, j]] * v[[b, h, j, d]];
                        }
                        out[[b, h, i, d]] = acc;
                    }
                }
            }
        }

        // Merge heads: [bw, n, dim]
        let mut merged = Array::zeros(IxDyn(&[bw * n, self.dim]));
        for b in 0..bw {
            for i in 0..n {
                for h in 0..self.num_heads {
                    for d in 0..head_dim {
                        merged[[b * n + i, h * head_dim + d]] = out[[b, h, i, d]];
                    }
                }
            }
        }

        // Output projection: [bw*n, dim]
        let proj_out = matmul_2d(&merged, &self.proj_weight)?;
        let proj_3d = proj_out
            .into_shape_with_order(IxDyn(&[bw, n, self.dim]))
            .map_err(|e| NeuralError::InvalidArgument(format!("reshape error: {e}")))?
            .to_owned();

        self.proj_drop.forward(&proj_3d)
    }

    /// Update parameters with gradient descent
    pub fn update(&mut self, _lr: F) -> Result<()> {
        // Gradient updates would be applied by an optimizer in a full training loop
        Ok(())
    }

    /// Count trainable parameters
    pub fn parameter_count(&self) -> usize {
        let qkv = self.dim * 3 * self.dim;
        let bias = if self.qkv_bias.is_some() { 3 * self.dim } else { 0 };
        let proj = self.dim * self.dim;
        let rel = self.rel_pos_bias.len();
        qkv + bias + proj + rel
    }
}

// ---------------------------------------------------------------------------
// MLP block used inside each Swin block
// ---------------------------------------------------------------------------

/// Two-layer MLP with GELU activation
#[derive(Debug, Clone)]
pub struct SwinMlp<F: Float + Debug + Send + Sync + NumAssign + ScalarOperand> {
    fc1: Dense<F>,
    fc2: Dense<F>,
    drop: Dropout<F>,
}

impl<
        F: Float
            + Debug
            + Send
            + Sync
            + NumAssign
            + ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + 'static,
    > SwinMlp<F>
{
    pub fn new<R: Rng + Clone + Send + Sync + 'static>(
        in_features: usize,
        hidden_features: usize,
        drop: f64,
        rng: &mut R,
    ) -> Result<Self> {
        let fc1 = Dense::<F>::new(in_features, hidden_features, Some("gelu"), rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("fc1 failed: {e}")))?;
        let fc2 = Dense::<F>::new(hidden_features, in_features, None, rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("fc2 failed: {e}")))?;
        let drop = Dropout::<F>::new(drop, rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("dropout failed: {e}")))?;
        Ok(Self { fc1, fc2, drop })
    }

    pub fn forward(&self, x: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let h1 = self.fc1.forward(x)?;
        let h1d = self.drop.forward(&h1)?;
        let h2 = self.fc2.forward(&h1d)?;
        self.drop.forward(&h2)
    }

    pub fn update(&mut self, lr: F) -> Result<()> {
        self.fc1.update(lr)?;
        self.fc2.update(lr)?;
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        self.fc1.parameter_count() + self.fc2.parameter_count()
    }
}

// ---------------------------------------------------------------------------
// Swin Transformer Block
// ---------------------------------------------------------------------------

/// A single Swin Transformer block
///
/// Each block applies:
/// 1. Layer Norm
/// 2. Window (or Shifted-Window) Multi-Head Self-Attention
/// 3. Residual connection
/// 4. Layer Norm
/// 5. MLP
/// 6. Residual connection
#[derive(Debug, Clone)]
pub struct SwinBlock<F: Float + Debug + Send + Sync + NumAssign + ScalarOperand + SimdUnifiedOps> {
    norm1: LayerNorm<F>,
    attn: WindowAttention<F>,
    norm2: LayerNorm<F>,
    mlp: SwinMlp<F>,
    /// Whether this block uses shifted windows
    shift_size: usize,
    /// Feature map height (in patches)
    feat_h: usize,
    /// Feature map width (in patches)
    feat_w: usize,
    /// Window size
    window_size: usize,
    /// Drop path probability
    drop_path_prob: f64,
}

impl<
        F: Float
            + Debug
            + Send
            + Sync
            + NumAssign
            + ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + SimdUnifiedOps
            + 'static,
    > SwinBlock<F>
{
    /// Create a new Swin Transformer block
    pub fn new<R: Rng + Clone + Send + Sync + 'static>(
        dim: usize,
        feat_h: usize,
        feat_w: usize,
        num_heads: usize,
        window_size: usize,
        shift_size: usize,
        mlp_ratio: f64,
        drop: f64,
        attn_drop: f64,
        drop_path: f64,
        use_qkv_bias: bool,
        rng: &mut R,
    ) -> Result<Self> {
        let norm1 = LayerNorm::<F>::new(dim, 1e-5, rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("norm1 failed: {e}")))?;
        let attn = WindowAttention::<F>::new(
            dim, window_size, num_heads, attn_drop, drop, use_qkv_bias, rng,
        )?;
        let norm2 = LayerNorm::<F>::new(dim, 1e-5, rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("norm2 failed: {e}")))?;
        let hidden_dim = (dim as f64 * mlp_ratio) as usize;
        let mlp = SwinMlp::<F>::new(dim, hidden_dim, drop, rng)?;

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            shift_size,
            feat_h,
            feat_w,
            window_size,
            drop_path_prob: drop_path,
        })
    }

    /// Forward pass through the block
    ///
    /// `x` shape: [batch, feat_h * feat_w, dim]
    pub fn forward(&self, x: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InvalidArgument(format!(
                "SwinBlock expects 3D input [B, L, C], got {:?}",
                shape
            )));
        }
        let batch = shape[0];
        let l = shape[1];
        let c = shape[2];

        let h = self.feat_h;
        let w = self.feat_w;
        if l != h * w {
            return Err(NeuralError::InvalidArgument(format!(
                "Sequence length {l} does not match feat_h*feat_w = {h}*{w} = {}",
                h * w
            )));
        }

        // ---- Norm 1 + window attention ----
        let shortcut = x.clone();
        let xn = self.norm1.forward(x)?;

        // Reshape to [batch, h, w, c]
        let xn_4d = xn
            .into_shape_with_order(IxDyn(&[batch, h, w, c]))
            .map_err(|e| NeuralError::InvalidArgument(format!("reshape error: {e}")))?
            .to_owned();

        // Apply cyclic shift for shifted-window attention
        let xn_shifted = if self.shift_size > 0 {
            cyclic_shift(&xn_4d, self.shift_size)?
        } else {
            xn_4d
        };

        // Partition into windows: [num_windows*batch, window_size^2, c]
        let ws = self.window_size;
        let nw_h = h / ws;
        let nw_w = w / ws;
        let num_windows = nw_h * nw_w;
        let win_tokens = ws * ws;
        let mut windows = Array::zeros(IxDyn(&[batch * num_windows, win_tokens, c]));

        for b in 0..batch {
            for wh in 0..nw_h {
                for ww in 0..nw_w {
                    let win_idx = b * num_windows + wh * nw_w + ww;
                    for ph in 0..ws {
                        for pw in 0..ws {
                            let tok = ph * ws + pw;
                            for ch in 0..c {
                                windows[[win_idx, tok, ch]] =
                                    xn_shifted[[b, wh * ws + ph, ww * ws + pw, ch]];
                            }
                        }
                    }
                }
            }
        }

        // Window attention
        let attn_out = self.attn.forward(&windows)?;

        // Reverse window partition: [batch, h, w, c]
        let mut x4d = Array::zeros(IxDyn(&[batch, h, w, c]));
        for b in 0..batch {
            for wh in 0..nw_h {
                for ww in 0..nw_w {
                    let win_idx = b * num_windows + wh * nw_w + ww;
                    for ph in 0..ws {
                        for pw in 0..ws {
                            let tok = ph * ws + pw;
                            for ch in 0..c {
                                x4d[[b, wh * ws + ph, ww * ws + pw, ch]] =
                                    attn_out[[win_idx, tok, ch]];
                            }
                        }
                    }
                }
            }
        }

        // Reverse cyclic shift
        let x4d = if self.shift_size > 0 {
            cyclic_shift_reverse(&x4d, self.shift_size)?
        } else {
            x4d
        };

        // Reshape back to [batch, l, c]
        let x_flat = x4d
            .into_shape_with_order(IxDyn(&[batch, l, c]))
            .map_err(|e| NeuralError::InvalidArgument(format!("reshape error: {e}")))?
            .to_owned();

        // Residual 1 (with optional drop path)
        let drop_path_scale = if self.drop_path_prob > 0.0 {
            F::from(1.0 - self.drop_path_prob).unwrap_or(F::one())
        } else {
            F::one()
        };
        let x = shortcut + x_flat.mapv(|v| v * drop_path_scale);

        // ---- Norm 2 + MLP ----
        let shortcut2 = x.clone();
        let xn2 = self.norm2.forward(&x)?;
        let mlp_out = self.mlp.forward(&xn2)?;
        let x = shortcut2 + mlp_out.mapv(|v| v * drop_path_scale);

        Ok(x)
    }

    pub fn update(&mut self, lr: F) -> Result<()> {
        self.norm1.update(lr)?;
        self.norm2.update(lr)?;
        self.attn.update(lr)?;
        self.mlp.update(lr)?;
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        self.norm1.parameter_count()
            + self.norm2.parameter_count()
            + self.attn.parameter_count()
            + self.mlp.parameter_count()
    }
}

// ---------------------------------------------------------------------------
// Patch Merging (downsampling between stages)
// ---------------------------------------------------------------------------

/// Patch merging layer that halves the spatial resolution and doubles channels
///
/// Takes [B, H, W, C] and outputs [B, H/2, W/2, 2*C].
#[derive(Debug, Clone)]
pub struct PatchMerging<F: Float + Debug + Send + Sync + NumAssign + ScalarOperand + SimdUnifiedOps> {
    norm: LayerNorm<F>,
    reduction: Dense<F>,
    in_dim: usize,
}

impl<
        F: Float
            + Debug
            + Send
            + Sync
            + NumAssign
            + ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + SimdUnifiedOps
            + 'static,
    > PatchMerging<F>
{
    pub fn new<R: Rng + Clone + Send + Sync + 'static>(in_dim: usize, rng: &mut R) -> Result<Self> {
        let norm = LayerNorm::<F>::new(4 * in_dim, 1e-5, rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("norm failed: {e}")))?;
        let reduction = Dense::<F>::new(4 * in_dim, 2 * in_dim, None, rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("reduction failed: {e}")))?;
        Ok(Self {
            norm,
            reduction,
            in_dim,
        })
    }

    /// Forward: input [B, H*W, C] → output [B, H/2*W/2, 2*C]
    pub fn forward(&self, x: &Array<F, IxDyn>, h: usize, w: usize) -> Result<Array<F, IxDyn>> {
        let shape = x.shape();
        let batch = shape[0];
        let c = self.in_dim;

        // Reshape to [B, H, W, C]
        let x4d = x
            .view()
            .into_shape_with_order(IxDyn(&[batch, h, w, c]))
            .map_err(|e| NeuralError::InvalidArgument(format!("reshape error: {e}")))?
            .to_owned();

        let h2 = h / 2;
        let w2 = w / 2;

        // Gather 2x2 patches: [B, H/2, W/2, 4*C]
        let mut merged = Array::zeros(IxDyn(&[batch, h2 * w2, 4 * c]));
        for b in 0..batch {
            for ih in 0..h2 {
                for iw in 0..w2 {
                    let tok = ih * w2 + iw;
                    for ch in 0..c {
                        merged[[b, tok, ch]] = x4d[[b, ih * 2, iw * 2, ch]];
                        merged[[b, tok, c + ch]] = x4d[[b, ih * 2, iw * 2 + 1, ch]];
                        merged[[b, tok, 2 * c + ch]] = x4d[[b, ih * 2 + 1, iw * 2, ch]];
                        merged[[b, tok, 3 * c + ch]] = x4d[[b, ih * 2 + 1, iw * 2 + 1, ch]];
                    }
                }
            }
        }

        // Norm + linear reduction
        let normed = self.norm.forward(&merged)?;
        self.reduction.forward(&normed)
    }

    pub fn update(&mut self, lr: F) -> Result<()> {
        self.norm.update(lr)?;
        self.reduction.update(lr)?;
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        self.norm.parameter_count() + self.reduction.parameter_count()
    }
}

// ---------------------------------------------------------------------------
// Swin Transformer Stage
// ---------------------------------------------------------------------------

/// A Swin Transformer stage: multiple blocks + optional downsampling
#[derive(Debug, Clone)]
pub struct SwinStage<F: Float + Debug + Send + Sync + NumAssign + ScalarOperand + SimdUnifiedOps> {
    blocks: Vec<SwinBlock<F>>,
    downsample: Option<PatchMerging<F>>,
    /// Current feature map dimensions
    feat_h: usize,
    feat_w: usize,
    dim: usize,
}

impl<
        F: Float
            + Debug
            + Send
            + Sync
            + NumAssign
            + ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + SimdUnifiedOps
            + 'static,
    > SwinStage<F>
{
    #[allow(clippy::too_many_arguments)]
    pub fn new<R: Rng + Clone + Send + Sync + 'static>(
        dim: usize,
        depth: usize,
        num_heads: usize,
        feat_h: usize,
        feat_w: usize,
        window_size: usize,
        mlp_ratio: f64,
        drop: f64,
        attn_drop: f64,
        drop_path_rates: &[f64],
        use_qkv_bias: bool,
        downsample: bool,
        rng: &mut R,
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let shift = if i % 2 == 0 { 0 } else { window_size / 2 };
            let dp = drop_path_rates.get(i).copied().unwrap_or(0.0);
            let block = SwinBlock::<F>::new(
                dim, feat_h, feat_w, num_heads, window_size, shift, mlp_ratio, drop, attn_drop,
                dp, use_qkv_bias, rng,
            )?;
            blocks.push(block);
        }

        let ds = if downsample {
            Some(PatchMerging::<F>::new(dim, rng)?)
        } else {
            None
        };

        Ok(Self {
            blocks,
            downsample: ds,
            feat_h,
            feat_w,
            dim,
        })
    }

    /// Forward: [B, H*W, C] → [B, H'*W', C'] (after optional downsampling)
    pub fn forward(&self, x: &Array<F, IxDyn>) -> Result<(Array<F, IxDyn>, usize, usize)> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        if let Some(ref ds) = self.downsample {
            let out = ds.forward(&x, self.feat_h, self.feat_w)?;
            Ok((out, self.feat_h / 2, self.feat_w / 2))
        } else {
            Ok((x, self.feat_h, self.feat_w))
        }
    }

    pub fn update(&mut self, lr: F) -> Result<()> {
        for block in &mut self.blocks {
            block.update(lr)?;
        }
        if let Some(ref mut ds) = self.downsample {
            ds.update(lr)?;
        }
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        let block_params: usize = self.blocks.iter().map(|b| b.parameter_count()).sum();
        let ds_params = self
            .downsample
            .as_ref()
            .map(|d| d.parameter_count())
            .unwrap_or(0);
        block_params + ds_params
    }

    /// Output dimension of this stage (after optional downsampling)
    pub fn out_dim(&self) -> usize {
        if self.downsample.is_some() {
            self.dim * 2
        } else {
            self.dim
        }
    }
}

// ---------------------------------------------------------------------------
// Patch Embedding
// ---------------------------------------------------------------------------

/// Linear patch embedding: [B, H, W, C_in] → [B, num_patches, embed_dim]
///
/// Implemented as a Dense layer applied on flattened patches
/// (patch_h * patch_w * in_channels → embed_dim).
#[derive(Debug, Clone)]
pub struct SwinPatchEmbed<F: Float + Debug + Send + Sync + NumAssign + ScalarOperand + SimdUnifiedOps> {
    proj: Dense<F>,
    norm: LayerNorm<F>,
    patch_h: usize,
    patch_w: usize,
    in_channels: usize,
    embed_dim: usize,
}

impl<
        F: Float
            + Debug
            + Send
            + Sync
            + NumAssign
            + ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + SimdUnifiedOps
            + 'static,
    > SwinPatchEmbed<F>
{
    pub fn new<R: Rng + Clone + Send + Sync + 'static>(
        patch_h: usize,
        patch_w: usize,
        in_channels: usize,
        embed_dim: usize,
        rng: &mut R,
    ) -> Result<Self> {
        let patch_dim = patch_h * patch_w * in_channels;
        let proj = Dense::<F>::new(patch_dim, embed_dim, None, rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("proj failed: {e}")))?;
        let norm = LayerNorm::<F>::new(embed_dim, 1e-5, rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("norm failed: {e}")))?;
        Ok(Self {
            proj,
            norm,
            patch_h,
            patch_w,
            in_channels,
            embed_dim,
        })
    }

    /// Forward: x shape [B, img_h, img_w, in_channels] → [B, nH*nW, embed_dim]
    pub fn forward(
        &self,
        x: &Array<F, IxDyn>,
        img_h: usize,
        img_w: usize,
    ) -> Result<Array<F, IxDyn>> {
        let shape = x.shape();
        let batch = shape[0];
        let nw_h = img_h / self.patch_h;
        let nw_w = img_w / self.patch_w;
        let num_patches = nw_h * nw_w;
        let patch_dim = self.patch_h * self.patch_w * self.in_channels;

        // Flatten patches: [B, num_patches, patch_dim]
        let mut patches = Array::zeros(IxDyn(&[batch, num_patches, patch_dim]));
        for b in 0..batch {
            for ph in 0..nw_h {
                for pw in 0..nw_w {
                    let p_idx = ph * nw_w + pw;
                    for dh in 0..self.patch_h {
                        for dw in 0..self.patch_w {
                            for ch in 0..self.in_channels {
                                let d_idx =
                                    dh * self.patch_w * self.in_channels + dw * self.in_channels + ch;
                                let img_h_idx = ph * self.patch_h + dh;
                                let img_w_idx = pw * self.patch_w + dw;
                                if img_h_idx < shape[1] && img_w_idx < shape[2] {
                                    patches[[b, p_idx, d_idx]] =
                                        x[[b, img_h_idx, img_w_idx, ch]];
                                }
                            }
                        }
                    }
                }
            }
        }

        let projected = self.proj.forward(&patches)?;
        self.norm.forward(&projected)
    }

    pub fn update(&mut self, lr: F) -> Result<()> {
        self.proj.update(lr)?;
        self.norm.update(lr)
    }

    pub fn parameter_count(&self) -> usize {
        self.proj.parameter_count() + self.norm.parameter_count()
    }

    #[allow(dead_code)]
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

// ---------------------------------------------------------------------------
// Swin Transformer (full model)
// ---------------------------------------------------------------------------

/// Swin Transformer model for image classification
///
/// Hierarchical transformer that uses window-based self-attention with cyclic
/// shifting to capture both local and global context efficiently.
#[derive(Debug, Clone)]
pub struct SwinTransformer<F: Float + Debug + Send + Sync + NumAssign + ScalarOperand + SimdUnifiedOps> {
    config: SwinConfig,
    patch_embed: SwinPatchEmbed<F>,
    pos_drop: Dropout<F>,
    stages: Vec<SwinStage<F>>,
    norm: LayerNorm<F>,
    head: Dense<F>,
}

impl<
        F: Float
            + Debug
            + Send
            + Sync
            + NumAssign
            + ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + SimdUnifiedOps
            + 'static,
    > SwinTransformer<F>
{
    /// Create a new Swin Transformer from a configuration
    pub fn new(config: SwinConfig) -> Result<Self> {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42u8; 32]);

        let (patch_h, patch_w) = config.patch_size;
        let patch_embed = SwinPatchEmbed::<F>::new(
            patch_h,
            patch_w,
            config.in_channels,
            config.embed_dim,
            &mut rng,
        )?;

        let pos_drop = Dropout::<F>::new(config.dropout_rate, &mut rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("pos_drop failed: {e}")))?;

        let (np_h, np_w) = config.num_patches();

        // Compute total drop-path rates distributed over all blocks
        let total_depth: usize = config.depths.iter().sum();
        let dp_rates: Vec<f64> = (0..total_depth)
            .map(|i| i as f64 * config.drop_path_rate / (total_depth.saturating_sub(1).max(1)) as f64)
            .collect();

        let mut stages = Vec::with_capacity(config.num_stages());
        let mut cur_dim = config.embed_dim;
        let mut cur_h = np_h;
        let mut cur_w = np_w;
        let mut dp_offset = 0;

        for stage_idx in 0..config.num_stages() {
            let depth = config.depths[stage_idx];
            let num_heads = config.num_heads[stage_idx];
            let has_downsample = stage_idx < config.num_stages() - 1;

            // Clamp feature map to be divisible by window_size
            let w_h = (cur_h / config.window_size).max(1) * config.window_size;
            let w_w = (cur_w / config.window_size).max(1) * config.window_size;

            let dp_slice = &dp_rates[dp_offset..(dp_offset + depth).min(dp_rates.len())];

            let stage = SwinStage::<F>::new(
                cur_dim,
                depth,
                num_heads,
                w_h,
                w_w,
                config.window_size,
                config.mlp_ratio,
                config.dropout_rate,
                config.attention_dropout_rate,
                dp_slice,
                config.qkv_bias,
                has_downsample,
                &mut rng,
            )?;

            dp_offset += depth;

            if has_downsample {
                cur_dim *= 2;
                cur_h /= 2;
                cur_w /= 2;
            }

            stages.push(stage);
        }

        // Final norm operates on the last stage's output dim
        let norm = LayerNorm::<F>::new(cur_dim, 1e-5, &mut rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("norm failed: {e}")))?;

        let head = Dense::<F>::new(cur_dim, config.num_classes, None, &mut rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("head failed: {e}")))?;

        Ok(Self {
            config,
            patch_embed,
            pos_drop,
            stages,
            norm,
            head,
        })
    }

    /// Forward pass
    ///
    /// Input `x`: [batch, img_h, img_w, in_channels]
    /// Output: [batch, num_classes]
    pub fn forward(&self, x: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = x.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InvalidArgument(format!(
                "SwinTransformer expects 4D input [B, H, W, C], got {:?}",
                shape
            )));
        }
        let batch = shape[0];
        let img_h = shape[1];
        let img_w = shape[2];

        // Patch embedding
        let mut tokens = self.patch_embed.forward(x, img_h, img_w)?;
        tokens = self.pos_drop.forward(&tokens)?;

        // Process through stages
        for stage in &self.stages {
            let (out, _, _) = stage.forward(&tokens)?;
            tokens = out;
        }

        // Global average pooling: [B, L, C] → [B, C]
        let c = tokens.shape()[2];
        let l = tokens.shape()[1];
        let mut pooled = Array::zeros(IxDyn(&[batch, c]));
        let scale = F::from(1.0 / l as f64).unwrap_or(F::one());
        for b in 0..batch {
            for i in 0..l {
                for ch in 0..c {
                    pooled[[b, ch]] = pooled[[b, ch]] + tokens[[b, i, ch]] * scale;
                }
            }
        }

        // Norm + head
        let pooled_3d = pooled
            .into_shape_with_order(IxDyn(&[batch, 1, c]))
            .map_err(|e| NeuralError::InvalidArgument(format!("reshape error: {e}")))?
            .to_owned();
        let normed = self.norm.forward(&pooled_3d)?;
        let normed_2d = normed
            .into_shape_with_order(IxDyn(&[batch, c]))
            .map_err(|e| NeuralError::InvalidArgument(format!("reshape error: {e}")))?
            .to_owned();
        self.head.forward(&normed_2d)
    }

    /// Return the configuration
    pub fn config(&self) -> &SwinConfig {
        &self.config
    }

    /// Count trainable parameters
    pub fn parameter_count(&self) -> usize {
        self.patch_embed.parameter_count()
            + self.stages.iter().map(|s| s.parameter_count()).sum::<usize>()
            + self.norm.parameter_count()
            + self.head.parameter_count()
    }
}

impl<
        F: Float
            + Debug
            + Send
            + Sync
            + NumAssign
            + ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + SimdUnifiedOps
            + 'static,
    > Layer<F> for SwinTransformer<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.forward(input)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, lr: F) -> Result<()> {
        self.patch_embed.update(lr)?;
        for stage in &mut self.stages {
            stage.update(lr)?;
        }
        self.norm.update(lr)?;
        self.head.update(lr)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "SwinTransformer"
    }

    fn parameter_count(&self) -> usize {
        self.parameter_count()
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Matrix multiplication: A [m, k] × B [k, n] → [m, n]
fn matmul_2d<F: Float + Debug + NumAssign + ScalarOperand>(
    a: &Array<F, IxDyn>,
    b: &Array<F, IxDyn>,
) -> Result<Array<F, IxDyn>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(NeuralError::InvalidArgument(
            "matmul_2d requires 2D arrays".to_string(),
        ));
    }
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];
    if b_shape[0] != k {
        return Err(NeuralError::InvalidArgument(format!(
            "matmul_2d: incompatible shapes [{m},{k}] × [{},{}]",
            b_shape[0], n
        )));
    }

    let mut out = Array::zeros(IxDyn(&[m, n]));
    for i in 0..m {
        for j in 0..n {
            let mut acc = F::zero();
            for p in 0..k {
                acc = acc + a[[i, p]] * b[[p, j]];
            }
            out[[i, j]] = acc;
        }
    }
    Ok(out)
}

/// Apply softmax in-place along the last dimension of a 4D array
fn softmax_last_dim<F: Float + Debug + NumAssign + ScalarOperand>(
    x: &mut Array<F, IxDyn>,
) -> Result<()> {
    let shape = x.shape().to_vec();
    let ndim = shape.len();
    if ndim < 1 {
        return Err(NeuralError::InvalidArgument(
            "softmax_last_dim requires non-empty array".to_string(),
        ));
    }
    let last = shape[ndim - 1];
    let outer: usize = shape[..ndim - 1].iter().product();

    for i in 0..outer {
        // Convert flat index i to multi-dim index (excluding last dim)
        let mut rem = i;
        let mut prefix = vec![0usize; ndim - 1];
        for d in (0..ndim - 1).rev() {
            prefix[d] = rem % shape[d];
            rem /= shape[d];
        }

        // Compute max for numerical stability
        let mut max_val = F::neg_infinity();
        for j in 0..last {
            let mut idx = prefix.clone();
            idx.push(j);
            let v = x[IxDyn(&idx)];
            if v > max_val {
                max_val = v;
            }
        }

        // Exp and sum
        let mut sum = F::zero();
        for j in 0..last {
            let mut idx = prefix.clone();
            idx.push(j);
            let v = (x[IxDyn(&idx)] - max_val).exp();
            x[IxDyn(&idx)] = v;
            sum = sum + v;
        }

        // Normalize
        let eps = F::from(1e-9).unwrap_or(F::zero());
        let inv_sum = F::one() / (sum + eps);
        for j in 0..last {
            let mut idx = prefix.clone();
            idx.push(j);
            x[IxDyn(&idx)] = x[IxDyn(&idx)] * inv_sum;
        }
    }
    Ok(())
}

/// Cyclic shift of a 4D [B, H, W, C] array by `shift` positions along H and W
fn cyclic_shift<F: Float + Debug + NumAssign + ScalarOperand>(
    x: &Array<F, IxDyn>,
    shift: usize,
) -> Result<Array<F, IxDyn>> {
    let shape = x.shape();
    let (batch, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
    let mut out = Array::zeros(IxDyn(&[batch, h, w, c]));
    for b in 0..batch {
        for i in 0..h {
            for j in 0..w {
                let ni = (i + h - shift) % h;
                let nj = (j + w - shift) % w;
                for ch in 0..c {
                    out[[b, ni, nj, ch]] = x[[b, i, j, ch]];
                }
            }
        }
    }
    Ok(out)
}

/// Reverse cyclic shift
fn cyclic_shift_reverse<F: Float + Debug + NumAssign + ScalarOperand>(
    x: &Array<F, IxDyn>,
    shift: usize,
) -> Result<Array<F, IxDyn>> {
    let shape = x.shape();
    let (batch, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
    let mut out = Array::zeros(IxDyn(&[batch, h, w, c]));
    for b in 0..batch {
        for i in 0..h {
            for j in 0..w {
                let ni = (i + shift) % h;
                let nj = (j + shift) % w;
                for ch in 0..c {
                    out[[b, ni, nj, ch]] = x[[b, i, j, ch]];
                }
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> SwinConfig {
        SwinConfig {
            img_size: (56, 56),
            patch_size: (4, 4),
            in_channels: 3,
            num_classes: 10,
            embed_dim: 32,
            depths: vec![2, 2],
            num_heads: vec![2, 4],
            window_size: 7,
            mlp_ratio: 2.0,
            dropout_rate: 0.0,
            attention_dropout_rate: 0.0,
            drop_path_rate: 0.0,
            qkv_bias: true,
        }
    }

    #[test]
    fn test_swin_config_tiny() {
        let cfg = SwinConfig::swin_tiny((224, 224), 1000);
        assert_eq!(cfg.embed_dim, 96);
        assert_eq!(cfg.depths, vec![2, 2, 6, 2]);
        assert_eq!(cfg.num_heads, vec![3, 6, 12, 24]);
    }

    #[test]
    fn test_swin_config_base() {
        let cfg = SwinConfig::swin_base((224, 224), 1000);
        assert_eq!(cfg.embed_dim, 128);
        assert_eq!(cfg.depths.len(), 4);
    }

    #[test]
    fn test_swin_config_stage_dim() {
        let cfg = tiny_config();
        assert_eq!(cfg.stage_dim(0), cfg.embed_dim);
        assert_eq!(cfg.stage_dim(1), cfg.embed_dim * 2);
    }

    #[test]
    fn test_swin_config_num_patches() {
        let cfg = tiny_config();
        let (np_h, np_w) = cfg.num_patches();
        assert_eq!(np_h, 56 / 4);
        assert_eq!(np_w, 56 / 4);
    }

    #[test]
    fn test_window_attention_forward() {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([0u8; 32]);
        let ws = 7;
        let dim = 32;
        let num_heads = 2;
        let attn = WindowAttention::<f32>::new(dim, ws, num_heads, 0.0, 0.0, true, &mut rng)
            .expect("Failed to create WindowAttention");
        // 1 batch, 1 window
        let x = Array::zeros(IxDyn(&[1, ws * ws, dim]));
        let out = attn.forward(&x).expect("Forward failed");
        assert_eq!(out.shape(), &[1, ws * ws, dim]);
    }

    #[test]
    fn test_swin_mlp_forward() {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([0u8; 32]);
        let mlp = SwinMlp::<f32>::new(32, 64, 0.0, &mut rng).expect("MLP failed");
        let x = Array::zeros(IxDyn(&[2, 10, 32]));
        let out = mlp.forward(&x).expect("Forward failed");
        assert_eq!(out.shape(), &[2, 10, 32]);
    }

    #[test]
    fn test_patch_embed_forward() {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([0u8; 32]);
        let embed = SwinPatchEmbed::<f32>::new(4, 4, 3, 32, &mut rng).expect("Embed failed");
        // [B, H, W, C]
        let x = Array::zeros(IxDyn(&[1, 56, 56, 3]));
        let out = embed.forward(&x, 56, 56).expect("Forward failed");
        // should be [1, (56/4)*(56/4), 32] = [1, 196, 32]
        assert_eq!(out.shape(), &[1, 196, 32]);
    }

    #[test]
    fn test_patch_merging_forward() {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([0u8; 32]);
        let pm = PatchMerging::<f32>::new(32, &mut rng).expect("PatchMerging failed");
        let x = Array::zeros(IxDyn(&[1, 14 * 14, 32]));
        let out = pm.forward(&x, 14, 14).expect("Forward failed");
        // Should be [1, 7*7, 64]
        assert_eq!(out.shape(), &[1, 49, 64]);
    }

    #[test]
    fn test_swin_block_no_shift() {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([0u8; 32]);
        let block = SwinBlock::<f32>::new(
            32, 14, 14, 2, 7, 0, 2.0, 0.0, 0.0, 0.0, true, &mut rng,
        )
        .expect("Block failed");
        let x = Array::zeros(IxDyn(&[1, 196, 32]));
        let out = block.forward(&x).expect("Forward failed");
        assert_eq!(out.shape(), &[1, 196, 32]);
    }

    #[test]
    fn test_swin_block_with_shift() {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([0u8; 32]);
        let block = SwinBlock::<f32>::new(
            32, 14, 14, 2, 7, 3, 2.0, 0.0, 0.0, 0.0, true, &mut rng,
        )
        .expect("Block failed");
        let x = Array::zeros(IxDyn(&[1, 196, 32]));
        let out = block.forward(&x).expect("Forward failed");
        assert_eq!(out.shape(), &[1, 196, 32]);
    }

    #[test]
    fn test_swin_transformer_forward() {
        let cfg = tiny_config();
        let model = SwinTransformer::<f32>::new(cfg).expect("Model creation failed");
        // [B, H, W, C]
        let x = Array::zeros(IxDyn(&[1, 56, 56, 3]));
        let out = model.forward(&x).expect("Forward failed");
        // Output: [1, num_classes=10]
        assert_eq!(out.shape(), &[1, 10]);
    }

    #[test]
    fn test_swin_transformer_parameter_count() {
        let cfg = tiny_config();
        let model = SwinTransformer::<f32>::new(cfg).expect("Model creation failed");
        let count = model.parameter_count();
        assert!(count > 0, "Model should have parameters");
    }

    #[test]
    fn test_swin_layer_impl() {
        use crate::layers::Layer;
        let cfg = tiny_config();
        let model = SwinTransformer::<f32>::new(cfg).expect("Model creation failed");
        assert_eq!(model.layer_type(), "SwinTransformer");
    }
}
