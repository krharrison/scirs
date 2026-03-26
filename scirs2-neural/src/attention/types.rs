//! Common types for attention mechanisms.
//!
//! This module provides shared configuration and output types used across
//! Flash Attention, RoPE, ALiBi, sliding-window, and other attention
//! variants in this crate.

/// Configuration for scaled dot-product attention layers.
///
/// # Defaults
///
/// ```rust
/// use scirs2_neural::attention::types::AttentionConfig;
/// let cfg = AttentionConfig::default();
/// assert_eq!(cfg.num_heads, 8);
/// assert_eq!(cfg.head_dim, 64);
/// assert_eq!(cfg.dropout_prob, 0.0);
/// assert!(!cfg.causal);
/// assert!(cfg.use_flash);
/// assert!(cfg.scale.is_none());
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,

    /// Dimensionality of each attention head.
    pub head_dim: usize,

    /// Dropout probability applied to attention weights (0.0 = disabled).
    pub dropout_prob: f64,

    /// Whether to use causal (autoregressive) masking.
    pub causal: bool,

    /// Whether to use Flash Attention 2 for memory-efficient computation.
    pub use_flash: bool,

    /// Optional custom attention scale factor.  When `None` the standard
    /// `1 / sqrt(head_dim)` is used.
    pub scale: Option<f64>,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            dropout_prob: 0.0,
            causal: false,
            use_flash: true,
            scale: None,
        }
    }
}

impl AttentionConfig {
    /// Compute the effective scale factor.
    ///
    /// Returns `scale` if explicitly set, otherwise `1 / sqrt(head_dim)`.
    pub fn effective_scale(&self) -> f64 {
        self.scale
            .unwrap_or_else(|| 1.0 / (self.head_dim as f64).sqrt())
    }
}

// ---------------------------------------------------------------------------
// AttentionMask
// ---------------------------------------------------------------------------

/// Attention mask variants controlling which query–key pairs are visible.
///
/// The `#[non_exhaustive]` attribute ensures that adding new variants is
/// backwards-compatible.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum AttentionMask {
    /// No masking — every position can attend to every other position.
    None,

    /// Causal (autoregressive) mask — position `i` may only attend to
    /// positions `j ≤ i`.
    Causal,

    /// Custom boolean mask.  `mask[i][j] == true` means query `i` may attend
    /// to key `j`.
    Custom(Vec<Vec<bool>>),

    /// Padding mask expressed as the *valid* sequence length per batch item.
    /// Positions `>= lengths[b]` for batch item `b` are masked.
    PaddingMask(Vec<usize>),
}

// ---------------------------------------------------------------------------
// AttentionOutput
// ---------------------------------------------------------------------------

/// Output of an attention operation.
///
/// `output` always contains the context vectors.  `attention_weights` is
/// populated only when the caller requests it (e.g. for visualisation), since
/// materialising the full weight tensor is expensive for long sequences.
#[derive(Debug, Clone)]
pub struct AttentionOutput {
    /// Context vectors with logical shape `[seq_len, embed_dim]`, stored as
    /// a 2-D `Vec<Vec<f64>>`.
    pub output: Vec<Vec<f64>>,

    /// Optional attention weights with shape `[num_heads, seq_len, seq_len]`.
    /// Each element `weights[h][i][j]` is the softmax weight that query `i`
    /// assigns to key `j` under head `h`.
    pub attention_weights: Option<Vec<Vec<Vec<f64>>>>,
}

impl AttentionOutput {
    /// Construct an output with no stored weights.
    pub fn new(output: Vec<Vec<f64>>) -> Self {
        Self {
            output,
            attention_weights: None,
        }
    }

    /// Construct an output that includes the full weight tensor.
    pub fn with_weights(output: Vec<Vec<f64>>, weights: Vec<Vec<Vec<f64>>>) -> Self {
        Self {
            output,
            attention_weights: Some(weights),
        }
    }
}

// ---------------------------------------------------------------------------
// PositionEncoding
// ---------------------------------------------------------------------------

/// Position-encoding strategies supported by attention layers in this crate.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionEncoding {
    /// Classic sinusoidal position embeddings (Vaswani et al., 2017).
    Sinusoidal,

    /// Learned absolute position embeddings (BERT, GPT-2 style).
    Learned,

    /// Rotary Position Embedding — RoPE (Su et al., 2021).
    RoPE,

    /// Attention with Linear Biases — ALiBi (Press et al., 2021).
    ALiBi,

    /// No position encoding (NoPE) — relies on relative information in the
    /// input or is provided externally.
    NoPE,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config_default() {
        let cfg = AttentionConfig::default();
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.dropout_prob, 0.0);
        assert!(!cfg.causal);
        assert!(cfg.use_flash);
        assert!(cfg.scale.is_none());
    }

    #[test]
    fn test_attention_config_effective_scale_default() {
        let cfg = AttentionConfig::default();
        let expected = 1.0 / (64.0_f64).sqrt();
        assert!((cfg.effective_scale() - expected).abs() < 1e-12);
    }

    #[test]
    fn test_attention_config_effective_scale_custom() {
        let cfg = AttentionConfig {
            scale: Some(0.5),
            ..Default::default()
        };
        assert!((cfg.effective_scale() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_attention_output_struct() {
        let output = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let ao = AttentionOutput::new(output.clone());
        assert_eq!(ao.output, output);
        assert!(ao.attention_weights.is_none());
    }

    #[test]
    fn test_attention_output_with_weights() {
        let output = vec![vec![0.5; 4]];
        let w = vec![vec![vec![0.25; 4]; 4]; 2];
        let ao = AttentionOutput::with_weights(output, w.clone());
        assert!(ao.attention_weights.is_some());
        assert_eq!(ao.attention_weights.as_ref().map(|x| x.len()), Some(2));
    }

    #[test]
    fn test_position_encoding_variants() {
        // Just ensure all variants can be constructed and compared.
        let variants = [
            PositionEncoding::Sinusoidal,
            PositionEncoding::Learned,
            PositionEncoding::RoPE,
            PositionEncoding::ALiBi,
            PositionEncoding::NoPE,
        ];
        for v in &variants {
            assert_eq!(v, v);
        }
    }

    #[test]
    fn test_attention_mask_causal_variant() {
        let mask = AttentionMask::Causal;
        // Pattern-matching must cover `_` arm for #[non_exhaustive] enums
        // used outside the crate; inside the crate we can match explicitly.
        match mask {
            AttentionMask::Causal => {}
            _ => panic!("expected Causal"),
        }
    }
}
