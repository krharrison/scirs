//! Advanced string and text algorithms for SciRS2.
//!
//! This module provides efficient implementations of fundamental string algorithms
//! suitable for bioinformatics, text mining, and data processing.
//!
//! # Modules
//!
//! - [`aho_corasick`] — Multi-pattern string search via the Aho-Corasick automaton.
//! - [`suffix_array`] — Suffix array construction, LCP arrays, and BWT.
//! - [`rolling_hash`] — Rabin-Karp rolling hash, edit distance, LCS, and k-mer utilities.

pub mod aho_corasick;
pub mod suffix_array;
pub mod rolling_hash;

pub use aho_corasick::*;
pub use suffix_array::*;
pub use rolling_hash::*;
