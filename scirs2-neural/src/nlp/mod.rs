//! NLP utilities for text processing and sequence generation.
//!
//! This module provides:
//! - Tokenizers: BPE, character-level, word-level
//! - Beam search and sampling decoding strategies
//! - Language model evaluation utilities (perplexity, BLEU, NLL)

pub mod beam_search;
pub mod language_model;
pub mod tokenizer;
