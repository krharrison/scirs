//! Bioinformatics utilities for SciRS2.
//!
//! This module provides a comprehensive set of tools for analysing biological
//! sequences (DNA, RNA, protein), performing sequence alignment, and computing
//! phylogenetic distances.
//!
//! ## Submodules
//!
//! | Submodule | Description |
//! |-----------|-------------|
//! | `sequence` | Sequence types, GC content, complement, translation, k-mer counting |
//! | `alignment` | Needleman-Wunsch, Smith-Waterman, Levenshtein edit distance |
//! | `stats` | Nucleotide / di-nucleotide frequencies, codon usage bias |
//! | `phylo` | Hamming distance, Jukes-Cantor model, pairwise distance matrix |
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_core::bioinformatics::sequence::{
//!     NucleotideSequence, SequenceType, gc_content, reverse_complement, translate,
//! };
//! use scirs2_core::bioinformatics::alignment::{needleman_wunsch, edit_distance};
//! use scirs2_core::bioinformatics::stats::nucleotide_frequencies;
//! use scirs2_core::bioinformatics::phylo::distance_matrix;
//!
//! // Validate and wrap a sequence
//! let dna = NucleotideSequence::new(b"ATGCATGC", SequenceType::Dna).expect("should succeed");
//! assert_eq!(dna.gc_content(), 0.5);
//!
//! // Reverse complement
//! let rc = reverse_complement(b"ATGCTT");
//! assert_eq!(rc, b"AAGCAT");
//!
//! // Global alignment
//! let (score, a1, a2) = needleman_wunsch(b"AGCT", b"AGT", 1, -1, -2).expect("should succeed");
//! assert_eq!(a1.len(), a2.len());
//!
//! // Edit distance
//! assert_eq!(edit_distance(b"ATGC", b"ATCC"), 1);
//!
//! // Nucleotide frequencies [A, C, G, T]
//! let freqs = nucleotide_frequencies(b"AACGT");
//! assert!((freqs[0] - 0.4).abs() < 1e-10);
//!
//! // Pairwise distance matrix
//! let seqs: &[&[u8]] = &[b"ATGC", b"ATGC", b"TTGT"];
//! let mat = distance_matrix(seqs).expect("should succeed");
//! assert_eq!(mat[[0, 0]], 0.0);
//! ```

pub mod alignment;
pub mod phylo;
pub mod sequence;
pub mod stats;

// Convenience re-exports for the most commonly used items.
pub use alignment::{edit_distance, needleman_wunsch, smith_waterman};
pub use phylo::{distance_matrix, hamming_distance, jukes_cantor_distance};
pub use sequence::{
    build_codon_table, complement, gc_content, kmer_count, reverse_complement, translate,
    AminoAcid, NucleotideSequence, SequenceType,
};
pub use stats::{codon_usage_table, dinucleotide_frequencies, nucleotide_frequencies};
