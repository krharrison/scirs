//! Sparse attention for long sequences (BigBird / Longformer style).
//!
//! This module provides configurable sparse attention patterns that avoid the
//! O(n²) memory and compute cost of full dense attention.  Three complementary
//! sub-modules expose the functionality:
//!
//! * [`types`]    — [`SparsePattern`], [`SparseAttentionConfig`], and
//!   [`SparseAttentionMask`] definitions.
//! * [`mask`]     — [`AttentionMaskBuilder`] that generates the sparsity
//!   patterns (local window, global+local, random, block-sparse).
//! * [`attention`] — [`SparseAttention`] forward computation over a prebuilt mask.
//!
//! ## Quick start
//!
//! ```rust
//! use scirs2_neural::attention::sparse::{
//!     SparseAttention, SparseAttentionConfig, SparsePattern,
//! };
//!
//! let mut cfg = SparseAttentionConfig::default();
//! cfg.pattern = SparsePattern::LocalWindow;
//! cfg.window_size = 4;
//! cfg.n_heads = 2;
//! cfg.head_dim = 8;
//! let sa = SparseAttention::new(cfg);
//! let seq_len = 16;
//! let n = seq_len * 2 * 8;
//! let q = vec![0.1f64; n];
//! let out = sa.forward(&q, &q, &q, seq_len, &[]);
//! assert_eq!(out.len(), n);
//! ```

pub mod attention;
pub mod mask;
pub mod types;

pub use attention::SparseAttention;
pub use mask::AttentionMaskBuilder;
pub use types::{SparseAttentionConfig, SparseAttentionMask, SparsePattern};
