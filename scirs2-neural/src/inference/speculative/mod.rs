//! Closure-based speculative decoding for language-model inference.
//!
//! This sub-module provides a lightweight, closure-driven API for speculative
//! decoding.  It complements the trait-based [`crate::speculative`] module by
//! requiring no model trait implementations — the caller passes ordinary Rust
//! closures for the draft and target forward passes.
//!
//! ## Quick start
//!
//! ```rust
//! use scirs2_neural::inference::speculative::{SpeculativeConfig, SpeculativeDecoder};
//!
//! let logits = vec![1.0f64, 2.0, 0.5, 0.1]; // 4-token vocabulary
//! let mut cfg = SpeculativeConfig::default();
//! cfg.max_tokens = 4;
//! cfg.draft_steps = 2;
//!
//! let result = SpeculativeDecoder::decode(
//!     &[0u32],                       // prompt
//!     |_ctx| vec![                   // draft_fn: 2 tokens per step
//!         (1, logits.clone()),
//!         (2, logits.clone()),
//!     ],
//!     |_ctx, draft| draft.iter().map(|_| logits.clone()).collect(), // target_fn
//!     &cfg,
//! );
//! assert!(!result.accepted_tokens.is_empty());
//! ```

pub mod rejection_sampling;
pub mod types;

pub use rejection_sampling::{SpeculativeDecoder, TokenDist};
pub use types::{SpeculativeConfig, SpeculativeResult};
