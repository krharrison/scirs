//! LLM inference utilities: paged KV cache, block manager, prefix caching.
//!
//! This module implements a production-grade paged KV cache for LLM inference,
//! inspired by vLLM's PagedAttention architecture. Key features:
//!
//! - **Non-contiguous memory**: Keys/values stored in fixed-size pages, allowing
//!   flexible memory allocation without fragmentation.
//! - **Block manager**: Tracks page chains per sequence, handles allocation and eviction.
//! - **Prefix sharing**: Shared prefix cache enables KV reuse across requests with
//!   common prefixes (e.g., system prompts).
//! - **Paged attention**: Attention computation over non-contiguous page chains.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                  KvPagePool                         │
//! │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐      │
//! │  │ Page 0 │ │ Page 1 │ │ Page 2 │ │ Page 3 │ ...  │
//! │  │[bs,H,D]│ │[bs,H,D]│ │[bs,H,D]│ │[bs,H,D]│      │
//! │  └────────┘ └────────┘ └────────┘ └────────┘      │
//! └─────────────────────────────────────────────────────┘
//!            ↑
//!    BlockManager maps SeqId → [PageId, PageId, ...]
//! ```
//!
//! ## Example
//!
//! ```rust
//! use scirs2_neural::inference::{
//!     KvPageConfig, KvPagePool, BlockManagerConfig, BlockManager,
//!     PagedAttentionConfig, PagedAttentionForward,
//! };
//!
//! // Configure pages: block_size=16 tokens, 8 heads, head_dim=64
//! let page_cfg = KvPageConfig {
//!     block_size: 16,
//!     num_heads: 8,
//!     head_dim: 64,
//!     dtype_bytes: 4,
//! };
//! let pool = KvPagePool::<f32>::new(128, page_cfg);
//! let bm_cfg = BlockManagerConfig {
//!     max_sequences: 32,
//!     max_pages_per_seq: 64,
//! };
//! let _manager = BlockManager::<f32>::new(pool, bm_cfg);
//! ```

pub mod block_manager;
pub mod kv_page;
pub mod paged_attention;
pub mod speculative;

pub use block_manager::{BlockManager, BlockManagerConfig, SharedPrefixCache};
pub use kv_page::{KvPage, KvPageConfig, KvPagePool, PageId};
pub use paged_attention::{PagedAttentionConfig, PagedAttentionForward};

use crate::NeuralError;

/// Errors specific to inference / paged KV cache operations.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    /// No free pages remaining in the pool.
    #[error("Out of memory: no free pages in pool")]
    Oom,

    /// The requested sequence ID is not tracked by the block manager.
    #[error("Sequence not found: {0}")]
    SequenceNotFound(u64),

    /// A page was freed more than once.
    #[error("Page {0} already freed")]
    DoubleFree(PageId),

    /// Page ID is out of bounds for the pool.
    #[error("Page index {0} out of bounds (pool size {1})")]
    PageOutOfBounds(PageId, usize),

    /// Slot position exceeds page capacity.
    #[error("Slot {slot} out of range for page capacity {capacity}")]
    SlotOutOfRange {
        /// Requested slot index.
        slot: usize,
        /// Page capacity (block_size).
        capacity: usize,
    },

    /// Shape mismatch when writing key/value tensors.
    #[error(
        "Shape mismatch: expected [{expected_heads}, {expected_dim}], got [{got_heads}, {got_dim}]"
    )]
    KvShapeMismatch {
        /// Expected number of heads.
        expected_heads: usize,
        /// Expected head dimension.
        expected_dim: usize,
        /// Actual number of heads.
        got_heads: usize,
        /// Actual head dimension.
        got_dim: usize,
    },

    /// The sequence already has the maximum number of pages allocated.
    #[error("Sequence {0} has reached max pages per sequence")]
    MaxPagesExceeded(u64),

    /// A neural layer error propagated from elsewhere in the crate.
    #[error("Neural error: {0}")]
    Neural(#[from] NeuralError),
}

/// Convenience alias for `Result<T, InferenceError>`.
pub type InferenceResult<T> = Result<T, InferenceError>;
