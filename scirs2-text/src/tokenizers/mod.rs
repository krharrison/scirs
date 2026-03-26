//! Transformer tokenizers.
//!
//! This module provides production-quality tokenizer implementations for
//! modern transformer architectures:
//!
//! - [`bert`]: BERT-style WordPiece tokenizer with `[CLS]` / `[SEP]` / `[MASK]`
//!   special tokens, batch encoding with padding/truncation, pair encoding with
//!   `token_type_ids`.
//!
//! - [`roberta`]: RoBERTa byte-level BPE tokenizer with `<s>` / `</s>` special
//!   tokens and GPT-2 compatible byte encoding.

/// BERT-style WordPiece tokenizer.
pub mod bert;
/// HuggingFace tokenizers JSON serialization format.
pub mod hf_json;
/// RoBERTa byte-level BPE tokenizer.
pub mod roberta;

pub use bert::{BatchEncoding, BertEncoding, BertTokenizer};
pub use hf_json::{HfAddedToken, HfNormalizerConfig, HfTokenizerJson};
pub use roberta::RobertaTokenizer;
