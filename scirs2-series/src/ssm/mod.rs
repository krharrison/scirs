//! State-Space Sequence Models (S4 and Mamba) for linear-time long-range dependencies.
//!
//! This module implements two classes of structured state-space models:
//!
//! - **S4** (Structured State Space Sequence Model): Uses HiPPO matrix initialization
//!   and convolutional representation for efficient long-range dependency modeling.
//!   Reference: Gu et al. (2021) "Efficiently Modeling Long Sequences with Structured State Spaces"
//!
//! - **Mamba** (Selective SSM): Extends S4 with input-dependent (selective) state transitions,
//!   enabling the model to focus on relevant information in the input.
//!   Reference: Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
//!
//! Both models operate in O(L) time per layer (linear in sequence length), making them
//! efficient alternatives to O(L²) attention-based transformers for long sequences.

pub mod config;
pub mod mamba;
pub mod s4;

pub use config::{MambaConfig, S4Config};
pub use mamba::{MambaBlock, MambaModel};
pub use s4::S4Layer;
