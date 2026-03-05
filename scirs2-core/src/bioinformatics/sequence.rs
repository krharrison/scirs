//! Sequence analysis for bioinformatics
//!
//! Provides types and functions for working with biological sequences:
//! DNA, RNA, and protein sequences with validation, transformation,
//! and analysis operations.

use std::collections::HashMap;
use std::fmt;

use crate::error::{CoreError, CoreResult};

/// Represents a single amino acid using the standard one-letter code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AminoAcid {
    /// Alanine (A)
    Ala,
    /// Arginine (R)
    Arg,
    /// Asparagine (N)
    Asn,
    /// Aspartic acid (D)
    Asp,
    /// Cysteine (C)
    Cys,
    /// Glutamine (Q)
    Gln,
    /// Glutamic acid (E)
    Glu,
    /// Glycine (G)
    Gly,
    /// Histidine (H)
    His,
    /// Isoleucine (I)
    Ile,
    /// Leucine (L)
    Leu,
    /// Lysine (K)
    Lys,
    /// Methionine (M) — also start codon
    Met,
    /// Phenylalanine (F)
    Phe,
    /// Proline (P)
    Pro,
    /// Serine (S)
    Ser,
    /// Threonine (T)
    Thr,
    /// Tryptophan (W)
    Trp,
    /// Tyrosine (Y)
    Tyr,
    /// Valine (V)
    Val,
    /// Stop codon (*)
    Stop,
}

impl AminoAcid {
    /// Returns the one-letter code for this amino acid.
    ///
    /// Stop codons are represented as `*`.
    #[must_use]
    pub fn one_letter(&self) -> char {
        match self {
            AminoAcid::Ala => 'A',
            AminoAcid::Arg => 'R',
            AminoAcid::Asn => 'N',
            AminoAcid::Asp => 'D',
            AminoAcid::Cys => 'C',
            AminoAcid::Gln => 'Q',
            AminoAcid::Glu => 'E',
            AminoAcid::Gly => 'G',
            AminoAcid::His => 'H',
            AminoAcid::Ile => 'I',
            AminoAcid::Leu => 'L',
            AminoAcid::Lys => 'K',
            AminoAcid::Met => 'M',
            AminoAcid::Phe => 'F',
            AminoAcid::Pro => 'P',
            AminoAcid::Ser => 'S',
            AminoAcid::Thr => 'T',
            AminoAcid::Trp => 'W',
            AminoAcid::Tyr => 'Y',
            AminoAcid::Val => 'V',
            AminoAcid::Stop => '*',
        }
    }

    /// Returns the three-letter code for this amino acid.
    #[must_use]
    pub fn three_letter(&self) -> &'static str {
        match self {
            AminoAcid::Ala => "Ala",
            AminoAcid::Arg => "Arg",
            AminoAcid::Asn => "Asn",
            AminoAcid::Asp => "Asp",
            AminoAcid::Cys => "Cys",
            AminoAcid::Gln => "Gln",
            AminoAcid::Glu => "Glu",
            AminoAcid::Gly => "Gly",
            AminoAcid::His => "His",
            AminoAcid::Ile => "Ile",
            AminoAcid::Leu => "Leu",
            AminoAcid::Lys => "Lys",
            AminoAcid::Met => "Met",
            AminoAcid::Phe => "Phe",
            AminoAcid::Pro => "Pro",
            AminoAcid::Ser => "Ser",
            AminoAcid::Thr => "Thr",
            AminoAcid::Trp => "Trp",
            AminoAcid::Tyr => "Tyr",
            AminoAcid::Val => "Val",
            AminoAcid::Stop => "Stop",
        }
    }

    /// Returns `true` if this amino acid is a stop codon.
    #[must_use]
    pub fn is_stop(&self) -> bool {
        matches!(self, AminoAcid::Stop)
    }
}

impl fmt::Display for AminoAcid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.one_letter())
    }
}

/// The type of nucleotide sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceType {
    /// DNA: A, T, G, C (and N for ambiguous)
    Dna,
    /// RNA: A, U, G, C (and N for ambiguous)
    Rna,
}

/// A validated nucleotide sequence (DNA or RNA).
///
/// The sequence is stored as uppercase ASCII bytes. Valid bases are:
/// - DNA: A, T, G, C, N (ambiguous)
/// - RNA: A, U, G, C, N (ambiguous)
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::sequence::{NucleotideSequence, SequenceType};
///
/// let dna = NucleotideSequence::new(b"ATGCATGC", SequenceType::Dna).expect("should succeed");
/// assert_eq!(dna.len(), 8);
/// assert_eq!(dna.gc_content(), 0.5);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NucleotideSequence {
    data: Vec<u8>,
    seq_type: SequenceType,
}

impl NucleotideSequence {
    /// Creates a new validated nucleotide sequence.
    ///
    /// Accepts both upper and lowercase input; internally stores uppercase.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ValueError` if the sequence contains invalid characters
    /// for the given sequence type.
    pub fn new(seq: &[u8], seq_type: SequenceType) -> CoreResult<Self> {
        let data: Vec<u8> = seq.iter().map(|&b| b.to_ascii_uppercase()).collect();
        for (i, &base) in data.iter().enumerate() {
            if !is_valid_base(base, seq_type) {
                return Err(CoreError::ValueError(crate::error_context!(format!(
                    "Invalid {} base '{}' at position {}",
                    match seq_type {
                        SequenceType::Dna => "DNA",
                        SequenceType::Rna => "RNA",
                    },
                    base as char,
                    i
                ))));
            }
        }
        Ok(Self { data, seq_type })
    }

    /// Returns the underlying byte slice.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Returns the length of the sequence.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the sequence is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the sequence type (DNA or RNA).
    #[must_use]
    pub fn seq_type(&self) -> SequenceType {
        self.seq_type
    }

    /// Computes the GC content (fraction of G+C bases) of this sequence.
    ///
    /// Returns 0.0 if the sequence is empty.
    #[must_use]
    pub fn gc_content(&self) -> f64 {
        gc_content(&self.data)
    }

    /// Returns the reverse complement of this DNA sequence.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ValueError` if called on an RNA sequence.
    pub fn reverse_complement(&self) -> CoreResult<NucleotideSequence> {
        if self.seq_type != SequenceType::Dna {
            return Err(CoreError::ValueError(crate::error_context!(
                "reverse_complement is only defined for DNA sequences"
            )));
        }
        let rc = reverse_complement(&self.data);
        Ok(NucleotideSequence {
            data: rc,
            seq_type: SequenceType::Dna,
        })
    }

    /// Translates this DNA sequence into a protein sequence.
    ///
    /// Reads codons starting from position 0.  Translation stops at the first
    /// stop codon or when fewer than 3 bases remain.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ValueError` if this is an RNA sequence.
    pub fn translate(&self) -> CoreResult<Vec<AminoAcid>> {
        if self.seq_type != SequenceType::Dna {
            return Err(CoreError::ValueError(crate::error_context!(
                "translate is only defined for DNA sequences"
            )));
        }
        translate(&self.data)
    }

    /// Counts all k-mers (substrings of length `k`) in the sequence.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ValueError` if `k` is 0 or exceeds the sequence length.
    pub fn kmer_count(&self, k: usize) -> CoreResult<HashMap<Vec<u8>, usize>> {
        kmer_count(&self.data, k)
    }
}

impl fmt::Display for NucleotideSequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: data contains only valid ASCII because new() validated it.
        let s = std::str::from_utf8(&self.data).map_err(|_| fmt::Error)?;
        write!(f, "{s}")
    }
}

// ─── Free-standing functions ──────────────────────────────────────────────────

/// Returns `true` if `base` is a valid nucleotide for the given sequence type.
///
/// Accepts uppercase letters only (callers are responsible for normalisation).
#[must_use]
pub fn is_valid_base(base: u8, seq_type: SequenceType) -> bool {
    match seq_type {
        SequenceType::Dna => matches!(base, b'A' | b'T' | b'G' | b'C' | b'N'),
        SequenceType::Rna => matches!(base, b'A' | b'U' | b'G' | b'C' | b'N'),
    }
}

/// Returns the Watson-Crick complement of a DNA base.
///
/// Non-standard bases (including `N`) are left unchanged.
#[must_use]
pub fn complement_base(base: u8) -> u8 {
    match base.to_ascii_uppercase() {
        b'A' => b'T',
        b'T' => b'A',
        b'G' => b'C',
        b'C' => b'G',
        other => other,
    }
}

/// Returns the complement of a DNA sequence.
///
/// Each base is complemented in-place without reversing.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::sequence::complement;
///
/// let seq = b"ATGC";
/// let comp = complement(seq);
/// assert_eq!(comp, b"TACG");
/// ```
#[must_use]
pub fn complement(seq: &[u8]) -> Vec<u8> {
    seq.iter().map(|&b| complement_base(b)).collect()
}

/// Returns the reverse complement of a DNA sequence.
///
/// Equivalent to `complement(seq)` reversed.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::sequence::reverse_complement;
///
/// let seq = b"ATGCTT";
/// let rc = reverse_complement(seq);
/// assert_eq!(rc, b"AAGCAT");
/// ```
#[must_use]
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter().rev().map(|&b| complement_base(b)).collect()
}

/// Returns the GC content (fraction of G+C bases) in `seq`.
///
/// Returns `0.0` for empty sequences.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::sequence::gc_content;
///
/// let seq = b"ATGCATGC";
/// let gc = gc_content(seq);
/// assert!((gc - 0.5).abs() < 1e-10);
/// ```
#[must_use]
pub fn gc_content(seq: &[u8]) -> f64 {
    if seq.is_empty() {
        return 0.0;
    }
    let gc = seq
        .iter()
        .filter(|&&b| {
            let ub = b.to_ascii_uppercase();
            ub == b'G' || ub == b'C'
        })
        .count();
    gc as f64 / seq.len() as f64
}

/// Translates a DNA sequence (5'→3') into a protein sequence using the
/// standard genetic code (NCBI translation table 1).
///
/// Translation begins at position 0 and reads complete codons.  The resulting
/// vector includes `AminoAcid::Stop` for stop codons.  After the first stop
/// codon no further codons are translated.
///
/// # Errors
///
/// Returns `CoreError::ValueError` for unknown codons (e.g. sequences
/// containing `N`).
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::sequence::{translate, AminoAcid};
///
/// let protein = translate(b"ATGAAATAA").expect("should succeed");
/// assert_eq!(protein, vec![AminoAcid::Met, AminoAcid::Lys, AminoAcid::Stop]);
/// ```
pub fn translate(dna: &[u8]) -> CoreResult<Vec<AminoAcid>> {
    let codon_table = build_codon_table();
    let mut protein = Vec::new();

    let upper: Vec<u8> = dna.iter().map(|&b| b.to_ascii_uppercase()).collect();

    let mut i = 0;
    while i + 3 <= upper.len() {
        let codon: [u8; 3] = [upper[i], upper[i + 1], upper[i + 2]];
        let aa = codon_table.get(&codon).copied().ok_or_else(|| {
            CoreError::ValueError(crate::error_context!(format!(
                "Unknown codon: {}{}{}",
                codon[0] as char, codon[1] as char, codon[2] as char
            )))
        })?;
        protein.push(aa);
        if aa == AminoAcid::Stop {
            break;
        }
        i += 3;
    }
    Ok(protein)
}

/// Counts all k-mers (substrings of length `k`) in `seq`.
///
/// # Errors
///
/// Returns `CoreError::ValueError` if `k` is zero or greater than the sequence length.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::sequence::kmer_count;
///
/// let counts = kmer_count(b"ATAT", 2).expect("should succeed");
/// assert_eq!(*counts.get(b"AT".as_ref()).expect("should succeed"), 2);
/// assert_eq!(*counts.get(b"TA".as_ref()).expect("should succeed"), 1);
/// ```
pub fn kmer_count(seq: &[u8], k: usize) -> CoreResult<HashMap<Vec<u8>, usize>> {
    if k == 0 {
        return Err(CoreError::ValueError(crate::error_context!(
            "k must be at least 1"
        )));
    }
    if k > seq.len() {
        return Err(CoreError::ValueError(crate::error_context!(format!(
            "k ({k}) must not exceed sequence length ({})",
            seq.len()
        ))));
    }

    let upper: Vec<u8> = seq.iter().map(|&b| b.to_ascii_uppercase()).collect();
    let mut counts: HashMap<Vec<u8>, usize> = HashMap::new();

    for window in upper.windows(k) {
        *counts.entry(window.to_vec()).or_insert(0) += 1;
    }
    Ok(counts)
}

// ─── Standard codon table ─────────────────────────────────────────────────────

/// Builds the standard genetic code (NCBI table 1) mapping `[u8; 3]` codons
/// to `AminoAcid` values.
#[must_use]
pub fn build_codon_table() -> HashMap<[u8; 3], AminoAcid> {
    let mut t = HashMap::new();

    // TTx
    t.insert(*b"TTT", AminoAcid::Phe);
    t.insert(*b"TTC", AminoAcid::Phe);
    t.insert(*b"TTA", AminoAcid::Leu);
    t.insert(*b"TTG", AminoAcid::Leu);

    // TCx
    t.insert(*b"TCT", AminoAcid::Ser);
    t.insert(*b"TCC", AminoAcid::Ser);
    t.insert(*b"TCA", AminoAcid::Ser);
    t.insert(*b"TCG", AminoAcid::Ser);

    // TAx
    t.insert(*b"TAT", AminoAcid::Tyr);
    t.insert(*b"TAC", AminoAcid::Tyr);
    t.insert(*b"TAA", AminoAcid::Stop);
    t.insert(*b"TAG", AminoAcid::Stop);

    // TGx
    t.insert(*b"TGT", AminoAcid::Cys);
    t.insert(*b"TGC", AminoAcid::Cys);
    t.insert(*b"TGA", AminoAcid::Stop);
    t.insert(*b"TGG", AminoAcid::Trp);

    // CTx
    t.insert(*b"CTT", AminoAcid::Leu);
    t.insert(*b"CTC", AminoAcid::Leu);
    t.insert(*b"CTA", AminoAcid::Leu);
    t.insert(*b"CTG", AminoAcid::Leu);

    // CCx
    t.insert(*b"CCT", AminoAcid::Pro);
    t.insert(*b"CCC", AminoAcid::Pro);
    t.insert(*b"CCA", AminoAcid::Pro);
    t.insert(*b"CCG", AminoAcid::Pro);

    // CAx
    t.insert(*b"CAT", AminoAcid::His);
    t.insert(*b"CAC", AminoAcid::His);
    t.insert(*b"CAA", AminoAcid::Gln);
    t.insert(*b"CAG", AminoAcid::Gln);

    // CGx
    t.insert(*b"CGT", AminoAcid::Arg);
    t.insert(*b"CGC", AminoAcid::Arg);
    t.insert(*b"CGA", AminoAcid::Arg);
    t.insert(*b"CGG", AminoAcid::Arg);

    // ATx
    t.insert(*b"ATT", AminoAcid::Ile);
    t.insert(*b"ATC", AminoAcid::Ile);
    t.insert(*b"ATA", AminoAcid::Ile);
    t.insert(*b"ATG", AminoAcid::Met);

    // ACx
    t.insert(*b"ACT", AminoAcid::Thr);
    t.insert(*b"ACC", AminoAcid::Thr);
    t.insert(*b"ACA", AminoAcid::Thr);
    t.insert(*b"ACG", AminoAcid::Thr);

    // AAx
    t.insert(*b"AAT", AminoAcid::Asn);
    t.insert(*b"AAC", AminoAcid::Asn);
    t.insert(*b"AAA", AminoAcid::Lys);
    t.insert(*b"AAG", AminoAcid::Lys);

    // AGx
    t.insert(*b"AGT", AminoAcid::Ser);
    t.insert(*b"AGC", AminoAcid::Ser);
    t.insert(*b"AGA", AminoAcid::Arg);
    t.insert(*b"AGG", AminoAcid::Arg);

    // GTx
    t.insert(*b"GTT", AminoAcid::Val);
    t.insert(*b"GTC", AminoAcid::Val);
    t.insert(*b"GTA", AminoAcid::Val);
    t.insert(*b"GTG", AminoAcid::Val);

    // GCx
    t.insert(*b"GCT", AminoAcid::Ala);
    t.insert(*b"GCC", AminoAcid::Ala);
    t.insert(*b"GCA", AminoAcid::Ala);
    t.insert(*b"GCG", AminoAcid::Ala);

    // GAx
    t.insert(*b"GAT", AminoAcid::Asp);
    t.insert(*b"GAC", AminoAcid::Asp);
    t.insert(*b"GAA", AminoAcid::Glu);
    t.insert(*b"GAG", AminoAcid::Glu);

    // GGx
    t.insert(*b"GGT", AminoAcid::Gly);
    t.insert(*b"GGC", AminoAcid::Gly);
    t.insert(*b"GGA", AminoAcid::Gly);
    t.insert(*b"GGG", AminoAcid::Gly);

    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nucleotide_sequence_valid_dna() {
        let seq = NucleotideSequence::new(b"ATGCATGC", SequenceType::Dna);
        assert!(seq.is_ok());
        let s = seq.expect("sequence creation failed");
        assert_eq!(s.len(), 8);
        assert_eq!(s.seq_type(), SequenceType::Dna);
    }

    #[test]
    fn test_nucleotide_sequence_invalid_dna() {
        // U is not valid in DNA
        let result = NucleotideSequence::new(b"AUGC", SequenceType::Dna);
        assert!(result.is_err());
    }

    #[test]
    fn test_nucleotide_sequence_valid_rna() {
        let seq = NucleotideSequence::new(b"AUGCAUGC", SequenceType::Rna);
        assert!(seq.is_ok());
    }

    #[test]
    fn test_complement_basic() {
        assert_eq!(complement(b"ATGC"), b"TACG");
        assert_eq!(complement(b"AAAA"), b"TTTT");
        assert_eq!(complement(b"CCCC"), b"GGGG");
    }

    #[test]
    fn test_reverse_complement() {
        // 5'-ATGCTT-3' → complement TACGAA → reverse AAGCAT
        let rc = reverse_complement(b"ATGCTT");
        assert_eq!(rc, b"AAGCAT");
    }

    #[test]
    fn test_reverse_complement_palindrome() {
        // 5'-GAATTC-3' is a palindrome (EcoRI site)
        let rc = reverse_complement(b"GAATTC");
        assert_eq!(rc, b"GAATTC");
    }

    #[test]
    fn test_gc_content_half() {
        let gc = gc_content(b"ATGCATGC");
        assert!((gc - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gc_content_zero() {
        let gc = gc_content(b"AAAA");
        assert!((gc - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gc_content_one() {
        let gc = gc_content(b"GCGC");
        assert!((gc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gc_content_empty() {
        assert_eq!(gc_content(b""), 0.0);
    }

    #[test]
    fn test_translate_met_stop() {
        let protein = translate(b"ATGAAATAA").expect("translation failed");
        assert_eq!(
            protein,
            vec![AminoAcid::Met, AminoAcid::Lys, AminoAcid::Stop]
        );
    }

    #[test]
    fn test_translate_stops_at_first_stop() {
        // ATG = Met, TAA = Stop, AAA = Lys (should not appear)
        let protein = translate(b"ATGTAAAAAA").expect("translation failed");
        assert_eq!(protein, vec![AminoAcid::Met, AminoAcid::Stop]);
    }

    #[test]
    fn test_translate_incomplete_codon_ignored() {
        // 7 bases → 2 complete codons + 1 leftover base (ignored)
        let protein = translate(b"ATGAAAT").expect("translation failed");
        assert_eq!(protein, vec![AminoAcid::Met, AminoAcid::Lys]);
    }

    #[test]
    fn test_kmer_count_basic() {
        let counts = kmer_count(b"ATAT", 2).expect("kmer_count failed");
        assert_eq!(*counts.get(b"AT".as_ref()).expect("AT not found"), 2);
        assert_eq!(*counts.get(b"TA".as_ref()).expect("TA not found"), 1);
    }

    #[test]
    fn test_kmer_count_k_equals_length() {
        let counts = kmer_count(b"ATGC", 4).expect("kmer_count failed");
        assert_eq!(counts.len(), 1);
        assert_eq!(*counts.get(b"ATGC".as_ref()).expect("ATGC not found"), 1);
    }

    #[test]
    fn test_kmer_count_k_zero_errors() {
        let result = kmer_count(b"ATGC", 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_kmer_count_k_too_large_errors() {
        let result = kmer_count(b"AT", 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_amino_acid_one_letter() {
        assert_eq!(AminoAcid::Met.one_letter(), 'M');
        assert_eq!(AminoAcid::Stop.one_letter(), '*');
    }

    #[test]
    fn test_codon_table_size() {
        let table = build_codon_table();
        // 64 codons in standard code
        assert_eq!(table.len(), 64);
    }

    #[test]
    fn test_lowercase_input_normalised() {
        // Lower-case input should be accepted and normalised
        let seq = NucleotideSequence::new(b"atgc", SequenceType::Dna).expect("should succeed");
        assert_eq!(seq.as_bytes(), b"ATGC");
    }
}
