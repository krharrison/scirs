//! Attention mechanisms for time series models.
//!
//! This module provides memory-efficient attention implementations suitable
//! for long time series sequences, including FlashAttention-style tiled
//! online softmax computation that avoids materializing the full N×N matrix.

pub mod flash;

pub use flash::{flash_attention, multi_head_flash_attention, FlashAttentionConfig};
