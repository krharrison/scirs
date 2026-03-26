//! # NLP Evaluation Metrics and Online LDA
//!
//! This module provides standard NLP evaluation metrics for machine translation
//! and text summarization, along with online topic modeling.
//!
//! ## Metrics
//!
//! - **BLEU** (Bilingual Evaluation Understudy): Measures n-gram precision of
//!   generated text against reference translations (Papineni et al. 2002).
//! - **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation): Measures
//!   n-gram recall for summarization evaluation.
//! - **METEOR** (Metric for Evaluation of Translation with Explicit ORdering):
//!   Alignment-based metric with stemming and synonym matching.
//!
//! ## Topic Modeling
//!
//! - **Online LDA**: Streaming Latent Dirichlet Allocation using stochastic
//!   variational inference (Hoffman et al. 2010).

pub mod bleu;
pub mod meteor;
pub mod ner;
pub mod online_lda;
pub mod rouge;

pub use bleu::{corpus_bleu, sentence_bleu, BleuConfig, SmoothingMethod as BleuSmoothingMethod};
pub use meteor::{meteor_score, MeteorConfig, MeteorScore};
pub use online_lda::{OnlineLda, OnlineLdaConfig};
pub use rouge::{rouge_l, rouge_l_summary, rouge_n, RougeScore};
