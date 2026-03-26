//! Speculative decoding for accelerated autoregressive generation.
//!
//! Speculative decoding uses a fast **draft model** to propose candidate tokens,
//! which are then verified in parallel by a slower but more accurate **target model**.
//! Through rejection sampling, the output distribution is mathematically equivalent
//! to sampling from the target model alone, while reducing the number of expensive
//! target-model evaluations.
//!
//! ## Overview
//!
//! The decoding loop works as follows:
//!
//! 1. The draft model generates `k` candidate tokens autoregressively.
//! 2. The target model evaluates all `k` positions in a single forward pass.
//! 3. Rejection sampling accepts a prefix of the draft tokens and, on rejection,
//!    resamples from the adjusted distribution `max(0, p_target - p_draft)`.
//! 4. The process repeats until the desired output length is reached.
//!
//! ## Key Types
//!
//! - [`SpeculativeConfig`] — configuration for the decoding loop.
//! - [`DraftModel`] — trait for fast draft models.
//! - [`TargetModel`] — trait for the authoritative target model.
//! - [`SpeculativeDecoder`] — orchestrator that runs the full loop.
//! - [`SpeculativeVerifier`] — rejection-sampling verifier.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_neural::speculative::{
//!     SpeculativeConfig, SpeculativeDecoder, UniformDraftModel,
//! };
//!
//! // For a real use case, implement TargetModel for your LLM.
//! // Here we just demonstrate the configuration:
//! let config = SpeculativeConfig {
//!     draft_length: 4,
//!     temperature: 0.8,
//!     top_k: 50,
//!     max_tokens: 256,
//!     adaptive_draft: true,
//! };
//! assert_eq!(config.draft_length, 4);
//! ```

pub mod decoder;
pub mod draft;
pub mod types;
pub mod verifier;

// Re-export key types
pub use decoder::SpeculativeDecoder;
pub use draft::{build_ngram_table, DraftModel, NGramDraftModel, UniformDraftModel, Xorshift64};
pub use types::{DecodingStats, SpeculativeConfig, TokenDistribution, VerificationResult};
pub use verifier::{compute_adjusted_distribution, SpeculativeVerifier, TargetModel};
