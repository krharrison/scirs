//! Sequence labeling algorithms: CRF, HMM, and Viterbi decoding.
//!
//! This module provides probabilistic sequence labeling tools suitable for
//! POS tagging, NER, chunking, and related structured prediction tasks.

pub mod crf;
pub mod hmm_tagger;
pub mod viterbi;

pub use crf::{CRFFeature, LinearChainCRF};
pub use hmm_tagger::HMMTagger;
pub use viterbi::{beam_search, viterbi_decode};

pub mod advanced;

pub use advanced::{
    iob2_to_spans,
    spans_to_iob2,
    CrfTagger,
    HiddenMarkovModel,
};
