//! Attention mechanisms for neural networks.
//!
//! This module groups attention implementations by style:
//!
//! * [`sparse`]          — Sparse / long-sequence attention (BigBird, Longformer).
//! * [`types`]           — Shared configuration and output types (`AttentionConfig`,
//!   `AttentionMask`, `AttentionOutput`, `PositionEncoding`).
//! * [`flash_v2`]        — Flash Attention 2 (tiled online-softmax, O(n) memory).
//! * [`rope`]            — Rotary Position Embedding (RoPE / RoFormer).
//! * [`alibi`]           — Attention with Linear Biases (ALiBi).
//! * [`sliding_window`]  — Sliding-window local attention (Longformer / Mistral style).
//!
//! For standard multi-head attention layers, see [`crate::layers`].

pub mod alibi;
pub mod flash_v2;
pub mod rope;
pub mod sliding_window;
pub mod sparse;
pub mod types;

pub use alibi::{AlibiConfig, AlibiSlopes};
pub use flash_v2::{naive_attention, FlashAttentionV2};
pub use rope::{RopeConfig, RopeEmbedding};
pub use sliding_window::{SlidingWindowAttention, SlidingWindowConfig};
pub use types::{AttentionConfig, AttentionMask, AttentionOutput, PositionEncoding};
