//! Rich biological sequence types.
//!
//! Provides `DnaSequence`, `RnaSequence`, and `ProteinSequence` with
//! comprehensive analysis operations: GC content, Shannon entropy, k-mer
//! frequencies, translation, isoelectric point, hydrophobicity, etc.

use std::collections::HashMap;

use crate::error::{CoreError, CoreResult};

use super::sequence::{
    build_codon_table, complement_base, gc_content, is_valid_base, reverse_complement, AminoAcid,
    SequenceType,
};

// ─── DnaSequence ─────────────────────────────────────────────────────────────

/// A validated DNA sequence with rich analysis capabilities.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::sequence_types::DnaSequence;
///
/// let dna = DnaSequence::new("seq1", "ATGCATGC").expect("should succeed");
/// assert_eq!(dna.len(), 8);
/// assert!((dna.gc_content() - 0.5).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DnaSequence {
    data: Vec<u8>,
    id: String,
}

impl DnaSequence {
    /// Constructs a new DNA sequence from a string.
    ///
    /// Accepts upper and lowercase ATGCN; converts to uppercase internally.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ValueError` if any character is not a valid DNA base.
    pub fn new(id: &str, sequence: &str) -> CoreResult<Self> {
        let data: Vec<u8> = sequence.bytes().map(|b| b.to_ascii_uppercase()).collect();
        for (i, &b) in data.iter().enumerate() {
            if !is_valid_base(b, SequenceType::Dna) {
                return Err(CoreError::ValueError(crate::error_context!(format!(
                    "Invalid DNA base '{}' at position {}",
                    b as char,
                    i
                ))));
            }
        }
        Ok(Self {
            data,
            id: id.to_string(),
        })
    }

    /// Constructs a new DNA sequence from a byte slice.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ValueError` if any byte is not a valid DNA base.
    pub fn from_bytes(id: &str, bytes: &[u8]) -> CoreResult<Self> {
        let data: Vec<u8> = bytes.iter().map(|&b| b.to_ascii_uppercase()).collect();
        for (i, &b) in data.iter().enumerate() {
            if !is_valid_base(b, SequenceType::Dna) {
                return Err(CoreError::ValueError(crate::error_context!(format!(
                    "Invalid DNA byte {} at position {i}",
                    b
                ))));
            }
        }
        Ok(Self {
            data,
            id: id.to_string(),
        })
    }

    /// Returns the length of the sequence.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the sequence has no bases.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the sequence identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the sequence as an uppercase ASCII `String`.
    #[must_use]
    pub fn as_str(&self) -> String {
        // SAFETY: data contains only valid ASCII (validated in constructors).
        String::from_utf8(self.data.clone()).unwrap_or_default()
    }

    /// Returns the underlying byte slice.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Computes the GC content (fraction of G+C bases).
    #[must_use]
    pub fn gc_content(&self) -> f64 {
        gc_content(&self.data)
    }

    /// Returns the reverse complement.
    #[must_use]
    pub fn reverse_complement(&self) -> Self {
        Self {
            data: reverse_complement(&self.data),
            id: format!("{}_rc", self.id),
        }
    }

    /// Transcribes this DNA sequence to RNA (T → U).
    #[must_use]
    pub fn transcribe(&self) -> RnaSequence {
        let rna_data: Vec<u8> = self
            .data
            .iter()
            .map(|&b| if b == b'T' { b'U' } else { b })
            .collect();
        RnaSequence {
            data: rna_data,
            id: format!("{}_rna", self.id),
        }
    }

    /// Returns a subsequence `[start, end)`.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::IndexError` if indices are out of range.
    pub fn subsequence(&self, start: usize, end: usize) -> CoreResult<Self> {
        if end > self.data.len() || start > end {
            return Err(CoreError::IndexError(crate::error_context!(format!(
                "subsequence [{start}, {end}) out of range for length {}",
                self.data.len()
            ))));
        }
        Ok(Self {
            data: self.data[start..end].to_vec(),
            id: format!("{}_sub_{start}_{end}", self.id),
        })
    }

    /// Finds all occurrences of `pattern` in the sequence.
    ///
    /// Returns the 0-based start positions.
    #[must_use]
    pub fn find_pattern(&self, pattern: &str) -> Vec<usize> {
        let pat: Vec<u8> = pattern
            .bytes()
            .map(|b| b.to_ascii_uppercase())
            .collect();
        if pat.is_empty() || pat.len() > self.data.len() {
            return Vec::new();
        }
        self.data
            .windows(pat.len())
            .enumerate()
            .filter(|(_, w)| *w == pat.as_slice())
            .map(|(i, _)| i)
            .collect()
    }

    /// Counts all k-mers in the sequence.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ValueError` if `k` is 0 or exceeds sequence length.
    pub fn kmer_frequency(&self, k: usize) -> CoreResult<HashMap<String, usize>> {
        if k == 0 {
            return Err(CoreError::ValueError(crate::error_context!(
                "k must be at least 1"
            )));
        }
        if k > self.data.len() {
            return Err(CoreError::ValueError(crate::error_context!(format!(
                "k ({k}) exceeds sequence length ({})",
                self.data.len()
            ))));
        }
        let mut counts: HashMap<String, usize> = HashMap::new();
        for window in self.data.windows(k) {
            // SAFETY: window contains valid ASCII only.
            let kmer = String::from_utf8(window.to_vec()).unwrap_or_default();
            *counts.entry(kmer).or_insert(0) += 1;
        }
        Ok(counts)
    }

    /// Shannon entropy of the k-mer distribution (bits).
    ///
    /// Uses k=2 dinucleotides. Returns 0 for sequences too short to have
    /// any k-mers.
    #[must_use]
    pub fn complexity_entropy(&self) -> f64 {
        let k = 2;
        if self.data.len() < k {
            return 0.0;
        }
        let mut counts: HashMap<[u8; 2], usize> = HashMap::new();
        for w in self.data.windows(k) {
            let key = [w[0], w[1]];
            *counts.entry(key).or_insert(0) += 1;
        }
        let total = counts.values().sum::<usize>() as f64;
        if total <= 0.0 {
            return 0.0;
        }
        counts
            .values()
            .map(|&c| {
                let p = c as f64 / total;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum()
    }
}

// ─── RnaSequence ─────────────────────────────────────────────────────────────

/// A validated RNA sequence.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::sequence_types::RnaSequence;
///
/// let rna = RnaSequence::new("mrna1", "AUGCAUGC").expect("should succeed");
/// assert_eq!(rna.len(), 8);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnaSequence {
    data: Vec<u8>,
    id: String,
}

impl RnaSequence {
    /// Constructs a new RNA sequence from a string.
    ///
    /// Accepts upper and lowercase AUGCN; converts to uppercase internally.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ValueError` if any character is not a valid RNA base.
    pub fn new(id: &str, sequence: &str) -> CoreResult<Self> {
        let data: Vec<u8> = sequence.bytes().map(|b| b.to_ascii_uppercase()).collect();
        for (i, &b) in data.iter().enumerate() {
            if !is_valid_base(b, SequenceType::Rna) {
                return Err(CoreError::ValueError(crate::error_context!(format!(
                    "Invalid RNA base '{}' at position {i}",
                    b as char
                ))));
            }
        }
        Ok(Self {
            data,
            id: id.to_string(),
        })
    }

    /// Returns the length of the sequence.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the sequence has no bases.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the sequence identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the sequence as an uppercase ASCII `String`.
    #[must_use]
    pub fn as_str(&self) -> String {
        String::from_utf8(self.data.clone()).unwrap_or_default()
    }

    /// Returns the reverse complement (U ↔ A, G ↔ C, reversed).
    #[must_use]
    pub fn reverse_complement(&self) -> Self {
        let rc: Vec<u8> = self
            .data
            .iter()
            .rev()
            .map(|&b| rna_complement_base(b))
            .collect();
        Self {
            data: rc,
            id: format!("{}_rc", self.id),
        }
    }

    /// Computes GC content.
    #[must_use]
    pub fn gc_content(&self) -> f64 {
        gc_content(&self.data)
    }

    /// Translates this RNA sequence into a protein sequence.
    ///
    /// Starts translation at the first AUG codon.  Stops at the first stop
    /// codon or when fewer than 3 bases remain.  The stop codon residue is
    /// included in the result.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ValueError` if no start codon is found or if
    /// an unknown codon is encountered.
    pub fn translate(&self) -> CoreResult<ProteinSequence> {
        // Find first AUG
        let start = self
            .data
            .windows(3)
            .position(|w| w == b"AUG")
            .ok_or_else(|| {
                CoreError::ValueError(crate::error_context!(
                    "no start codon (AUG) found in RNA sequence"
                ))
            })?;

        let codon_table = build_codon_table();
        let mut residues: Vec<AminoAcid> = Vec::new();
        let mut i = start;
        while i + 3 <= self.data.len() {
            // Convert U→T for lookup in the DNA codon table
            let codon: [u8; 3] = [
                if self.data[i] == b'U' { b'T' } else { self.data[i] },
                if self.data[i + 1] == b'U' { b'T' } else { self.data[i + 1] },
                if self.data[i + 2] == b'U' { b'T' } else { self.data[i + 2] },
            ];
            let aa = codon_table.get(&codon).copied().ok_or_else(|| {
                CoreError::ValueError(crate::error_context!(format!(
                    "unknown codon: {}{}{}",
                    codon[0] as char,
                    codon[1] as char,
                    codon[2] as char
                )))
            })?;
            residues.push(aa);
            if aa == AminoAcid::Stop {
                break;
            }
            i += 3;
        }

        Ok(ProteinSequence {
            residues,
            id: format!("{}_protein", self.id),
        })
    }

    /// Rough estimate of minimum free energy secondary structure.
    ///
    /// Uses a simplified nearest-neighbour stack counting heuristic.
    /// Each G-C or A-U stem stack contributes a small negative energy.
    /// Returns `None` if the sequence is too short to form any stems.
    ///
    /// Note: This is a rough approximation for illustrative purposes; it is
    /// not the full Zuker/Vienna thermodynamic model.
    #[must_use]
    pub fn secondary_structure_mfe_naive(&self) -> Option<f64> {
        let n = self.data.len();
        if n < 6 {
            return None;
        }
        // Count complementary pairs as a heuristic
        let mut energy = 0.0f64;
        let half = n / 2;
        for i in 0..half {
            let j = n - 1 - i;
            let a = self.data[i];
            let b = self.data[j];
            energy += match (a, b) {
                (b'G', b'C') | (b'C', b'G') => -3.4,
                (b'A', b'U') | (b'U', b'A') => -2.1,
                (b'G', b'U') | (b'U', b'G') => -1.4, // wobble
                _ => 0.0,
            };
        }
        if energy < 0.0 {
            Some(energy)
        } else {
            None
        }
    }
}

/// Returns the Watson-Crick complement of an RNA base.
fn rna_complement_base(base: u8) -> u8 {
    match base.to_ascii_uppercase() {
        b'A' => b'U',
        b'U' => b'A',
        b'G' => b'C',
        b'C' => b'G',
        other => other,
    }
}

// ─── AminoAcidExt trait / helpers ────────────────────────────────────────────

/// Returns the monoisotopic molecular weight (Da) of a single amino acid residue.
///
/// Values are average masses minus water (residue mass).
#[must_use]
pub fn amino_acid_molecular_weight(aa: AminoAcid) -> f64 {
    match aa {
        AminoAcid::Ala => 89.094,
        AminoAcid::Arg => 174.201,
        AminoAcid::Asn => 132.119,
        AminoAcid::Asp => 133.104,
        AminoAcid::Cys => 121.158,
        AminoAcid::Gln => 146.146,
        AminoAcid::Glu => 147.130,
        AminoAcid::Gly => 75.032,
        AminoAcid::His => 155.156,
        AminoAcid::Ile => 131.174,
        AminoAcid::Leu => 131.174,
        AminoAcid::Lys => 146.188,
        AminoAcid::Met => 149.208,
        AminoAcid::Phe => 165.192,
        AminoAcid::Pro => 115.132,
        AminoAcid::Ser => 105.093,
        AminoAcid::Thr => 119.119,
        AminoAcid::Trp => 204.228,
        AminoAcid::Tyr => 181.191,
        AminoAcid::Val => 117.148,
        AminoAcid::Stop => 0.0,
    }
}

/// Returns `true` if the amino acid is hydrophobic (Kyte-Doolittle > 0).
#[must_use]
pub fn is_hydrophobic(aa: AminoAcid) -> bool {
    matches!(
        aa,
        AminoAcid::Ala
            | AminoAcid::Val
            | AminoAcid::Ile
            | AminoAcid::Leu
            | AminoAcid::Met
            | AminoAcid::Phe
            | AminoAcid::Trp
            | AminoAcid::Pro
            | AminoAcid::Cys
    )
}

/// Returns `true` if the amino acid carries a charge at physiological pH.
#[must_use]
pub fn is_charged(aa: AminoAcid) -> bool {
    matches!(
        aa,
        AminoAcid::Asp | AminoAcid::Glu | AminoAcid::Lys | AminoAcid::Arg | AminoAcid::His
    )
}

/// Kyte-Doolittle hydrophobicity scale values.
///
/// Reference: J. Kyte, R.F. Doolittle, J. Mol. Biol. 157:105–132 (1982).
#[must_use]
pub fn kyte_doolittle(aa: AminoAcid) -> f64 {
    match aa {
        AminoAcid::Ile => 4.5,
        AminoAcid::Val => 4.2,
        AminoAcid::Leu => 3.8,
        AminoAcid::Phe => 2.8,
        AminoAcid::Cys => 2.5,
        AminoAcid::Met => 1.9,
        AminoAcid::Ala => 1.8,
        AminoAcid::Gly => -0.4,
        AminoAcid::Thr => -0.7,
        AminoAcid::Ser => -0.8,
        AminoAcid::Trp => -0.9,
        AminoAcid::Tyr => -1.3,
        AminoAcid::Pro => -1.6,
        AminoAcid::His => -3.2,
        AminoAcid::Glu => -3.5,
        AminoAcid::Gln => -3.5,
        AminoAcid::Asp => -3.5,
        AminoAcid::Asn => -3.5,
        AminoAcid::Lys => -3.9,
        AminoAcid::Arg => -4.5,
        AminoAcid::Stop => 0.0,
    }
}

// ─── ProteinSequence ─────────────────────────────────────────────────────────

/// A protein sequence with physicochemical analysis capabilities.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::sequence_types::ProteinSequence;
///
/// let prot = ProteinSequence::new("p1", "MKVL").expect("should succeed");
/// assert_eq!(prot.len(), 4);
/// let mw = prot.molecular_weight();
/// assert!(mw > 0.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ProteinSequence {
    residues: Vec<AminoAcid>,
    id: String,
}

impl ProteinSequence {
    /// Constructs a protein sequence from a one-letter code string.
    ///
    /// Accepts standard IUPAC one-letter codes (case insensitive).
    /// `*` is accepted as a stop codon.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ValueError` for unknown characters.
    pub fn new(id: &str, sequence: &str) -> CoreResult<Self> {
        let mut residues = Vec::with_capacity(sequence.len());
        for (i, ch) in sequence.chars().enumerate() {
            let aa = one_letter_to_amino_acid(ch).ok_or_else(|| {
                CoreError::ValueError(crate::error_context!(format!(
                    "Unknown amino acid code '{}' at position {i}",
                    ch
                )))
            })?;
            residues.push(aa);
        }
        Ok(Self {
            residues,
            id: id.to_string(),
        })
    }

    /// Returns the length (number of residues).
    #[must_use]
    pub fn len(&self) -> usize {
        self.residues.len()
    }

    /// Returns `true` if there are no residues.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.residues.is_empty()
    }

    /// Returns the sequence identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the sequence as a one-letter code string.
    #[must_use]
    pub fn as_str(&self) -> String {
        self.residues.iter().map(|aa| aa.one_letter()).collect()
    }

    /// Returns the residues slice.
    #[must_use]
    pub fn residues(&self) -> &[AminoAcid] {
        &self.residues
    }

    /// Computes the molecular weight (Da) as the sum of residue masses plus
    /// one water molecule (18.015 Da).
    #[must_use]
    pub fn molecular_weight(&self) -> f64 {
        let residue_sum: f64 = self
            .residues
            .iter()
            .filter(|&&aa| aa != AminoAcid::Stop)
            .map(|&aa| amino_acid_molecular_weight(aa))
            .sum();
        // Add water (H2O = 18.015) and subtract (n-1) waters lost in peptide bonds
        // Simplified: sum of residue masses + 18.015
        residue_sum + 18.015
    }

    /// Approximate isoelectric point (pI) using the Henderson-Hasselbalch
    /// equation.
    ///
    /// Uses average pKa values for ionisable groups.  The result is the pH at
    /// which the net charge is approximately zero.
    #[must_use]
    pub fn isoelectric_point_approx(&self) -> f64 {
        // pKa values (simplified, Bjellqvist scale)
        const PKA_NTERM: f64 = 8.0;
        const PKA_CTERM: f64 = 3.1;
        const PKA_ASP: f64 = 3.9;
        const PKA_GLU: f64 = 4.1;
        const PKA_HIS: f64 = 6.5;
        const PKA_CYS: f64 = 8.3;
        const PKA_TYR: f64 = 10.1;
        const PKA_LYS: f64 = 10.5;
        const PKA_ARG: f64 = 12.5;

        let comp = self.composition();
        let n_asp = *comp.get(&AminoAcid::Asp).unwrap_or(&0) as f64;
        let n_glu = *comp.get(&AminoAcid::Glu).unwrap_or(&0) as f64;
        let n_his = *comp.get(&AminoAcid::His).unwrap_or(&0) as f64;
        let n_cys = *comp.get(&AminoAcid::Cys).unwrap_or(&0) as f64;
        let n_tyr = *comp.get(&AminoAcid::Tyr).unwrap_or(&0) as f64;
        let n_lys = *comp.get(&AminoAcid::Lys).unwrap_or(&0) as f64;
        let n_arg = *comp.get(&AminoAcid::Arg).unwrap_or(&0) as f64;

        // Binary search for the pH where net charge ≈ 0
        let charge_at_ph = |ph: f64| -> f64 {
            // Positive contributions
            let pos = 1.0 / (1.0 + 10.0_f64.powf(ph - PKA_NTERM))
                + n_his / (1.0 + 10.0_f64.powf(ph - PKA_HIS))
                + n_lys / (1.0 + 10.0_f64.powf(ph - PKA_LYS))
                + n_arg / (1.0 + 10.0_f64.powf(ph - PKA_ARG));
            // Negative contributions
            let neg = 1.0 / (1.0 + 10.0_f64.powf(PKA_CTERM - ph))
                + n_asp / (1.0 + 10.0_f64.powf(PKA_ASP - ph))
                + n_glu / (1.0 + 10.0_f64.powf(PKA_GLU - ph))
                + n_cys / (1.0 + 10.0_f64.powf(PKA_CYS - ph))
                + n_tyr / (1.0 + 10.0_f64.powf(PKA_TYR - ph));
            pos - neg
        };

        let mut lo = 0.0f64;
        let mut hi = 14.0f64;
        for _ in 0..100 {
            let mid = (lo + hi) / 2.0;
            if charge_at_ph(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo + hi) / 2.0
    }

    /// Mean Kyte-Doolittle hydrophobicity of the sequence.
    ///
    /// Returns 0.0 for empty sequences.
    #[must_use]
    pub fn hydrophobicity_kyte_doolittle(&self) -> f64 {
        let non_stop: Vec<AminoAcid> = self
            .residues
            .iter()
            .copied()
            .filter(|&aa| aa != AminoAcid::Stop)
            .collect();
        if non_stop.is_empty() {
            return 0.0;
        }
        let sum: f64 = non_stop.iter().map(|&aa| kyte_doolittle(aa)).sum();
        sum / non_stop.len() as f64
    }

    /// Returns the amino acid composition as a count map.
    #[must_use]
    pub fn composition(&self) -> HashMap<AminoAcid, usize> {
        let mut counts: HashMap<AminoAcid, usize> = HashMap::new();
        for &aa in &self.residues {
            *counts.entry(aa).or_insert(0) += 1;
        }
        counts
    }
}

// ─── One-letter to AminoAcid conversion ──────────────────────────────────────

/// Converts a one-letter amino acid code to `AminoAcid`.
///
/// Returns `None` for unknown characters.
#[must_use]
pub fn one_letter_to_amino_acid(c: char) -> Option<AminoAcid> {
    match c.to_ascii_uppercase() {
        'A' => Some(AminoAcid::Ala),
        'R' => Some(AminoAcid::Arg),
        'N' => Some(AminoAcid::Asn),
        'D' => Some(AminoAcid::Asp),
        'C' => Some(AminoAcid::Cys),
        'Q' => Some(AminoAcid::Gln),
        'E' => Some(AminoAcid::Glu),
        'G' => Some(AminoAcid::Gly),
        'H' => Some(AminoAcid::His),
        'I' => Some(AminoAcid::Ile),
        'L' => Some(AminoAcid::Leu),
        'K' => Some(AminoAcid::Lys),
        'M' => Some(AminoAcid::Met),
        'F' => Some(AminoAcid::Phe),
        'P' => Some(AminoAcid::Pro),
        'S' => Some(AminoAcid::Ser),
        'T' => Some(AminoAcid::Thr),
        'W' => Some(AminoAcid::Trp),
        'Y' => Some(AminoAcid::Tyr),
        'V' => Some(AminoAcid::Val),
        '*' => Some(AminoAcid::Stop),
        _ => None,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DnaSequence ────────────────────────────────────────────────────────

    #[test]
    fn test_dna_new_valid() {
        let dna = DnaSequence::new("s1", "ATGCATGC").expect("valid DNA");
        assert_eq!(dna.len(), 8);
        assert_eq!(dna.id(), "s1");
    }

    #[test]
    fn test_dna_new_lowercase() {
        let dna = DnaSequence::new("s1", "atgc").expect("lowercase DNA");
        assert_eq!(dna.as_str(), "ATGC");
    }

    #[test]
    fn test_dna_new_invalid_base() {
        let result = DnaSequence::new("s1", "AUGC"); // U not valid in DNA
        assert!(result.is_err());
    }

    #[test]
    fn test_dna_from_bytes() {
        let dna = DnaSequence::from_bytes("b1", b"ATGC").expect("from_bytes failed");
        assert_eq!(dna.len(), 4);
    }

    #[test]
    fn test_dna_gc_content() {
        let dna = DnaSequence::new("s", "ATGCATGC").expect("should succeed");
        assert!((dna.gc_content() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_dna_reverse_complement() {
        let dna = DnaSequence::new("s", "ATGCTT").expect("should succeed");
        let rc = dna.reverse_complement();
        assert_eq!(rc.as_str(), "AAGCAT");
    }

    #[test]
    fn test_dna_transcribe() {
        let dna = DnaSequence::new("s", "ATGCAT").expect("should succeed");
        let rna = dna.transcribe();
        assert_eq!(rna.as_str(), "AUGCAU");
    }

    #[test]
    fn test_dna_subsequence() {
        let dna = DnaSequence::new("s", "ATGCATGC").expect("should succeed");
        let sub = dna.subsequence(2, 6).expect("valid subsequence");
        assert_eq!(sub.as_str(), "GCAT");
    }

    #[test]
    fn test_dna_subsequence_out_of_range() {
        let dna = DnaSequence::new("s", "ATGC").expect("should succeed");
        let result = dna.subsequence(0, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_dna_find_pattern() {
        let dna = DnaSequence::new("s", "ATGATGATG").expect("should succeed");
        let positions = dna.find_pattern("ATG");
        assert_eq!(positions, vec![0, 3, 6]);
    }

    #[test]
    fn test_dna_find_pattern_not_found() {
        let dna = DnaSequence::new("s", "ATGC").expect("should succeed");
        let positions = dna.find_pattern("TTT");
        assert!(positions.is_empty());
    }

    #[test]
    fn test_dna_kmer_frequency() {
        let dna = DnaSequence::new("s", "ATAT").expect("should succeed");
        let freqs = dna.kmer_frequency(2).expect("kmer_frequency failed");
        assert_eq!(*freqs.get("AT").expect("AT missing"), 2);
        assert_eq!(*freqs.get("TA").expect("TA missing"), 1);
    }

    #[test]
    fn test_dna_kmer_frequency_zero_error() {
        let dna = DnaSequence::new("s", "ATGC").expect("should succeed");
        let result = dna.kmer_frequency(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_dna_complexity_entropy() {
        let dna = DnaSequence::new("s", "ATGCATGCATGC").expect("should succeed");
        let entropy = dna.complexity_entropy();
        assert!(entropy >= 0.0);
        assert!(entropy.is_finite());
    }

    #[test]
    fn test_dna_is_empty() {
        let dna = DnaSequence::new("s", "ATGC").expect("should succeed");
        assert!(!dna.is_empty());
    }

    // ── RnaSequence ────────────────────────────────────────────────────────

    #[test]
    fn test_rna_new_valid() {
        let rna = RnaSequence::new("r1", "AUGCAUGC").expect("valid RNA");
        assert_eq!(rna.len(), 8);
    }

    #[test]
    fn test_rna_new_invalid() {
        let result = RnaSequence::new("r1", "ATGC"); // T not valid in RNA
        assert!(result.is_err());
    }

    #[test]
    fn test_rna_reverse_complement() {
        // 5'-AUGCAU-3' → complement UACGUA → reverse AUGCAU (palindrome-like check)
        let rna = RnaSequence::new("r", "AUGCAU").expect("should succeed");
        let rc = rna.reverse_complement();
        // complement of AUGCAU reversed: A→U,U→A,G→C,C→G,A→U,U→A = UACGUA reversed = AUGCAU
        assert_eq!(rc.as_str(), "AUGCAU");
    }

    #[test]
    fn test_rna_gc_content() {
        let rna = RnaSequence::new("r", "AUGCAUGC").expect("should succeed");
        assert!((rna.gc_content() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_rna_translate_met_stop() {
        let rna = RnaSequence::new("r", "AUGAAAUAA").expect("valid RNA");
        let prot = rna.translate().expect("translation failed");
        assert_eq!(prot.residues()[0], AminoAcid::Met);
        assert_eq!(prot.residues()[1], AminoAcid::Lys);
        assert_eq!(prot.residues()[2], AminoAcid::Stop);
    }

    #[test]
    fn test_rna_translate_no_start_codon_error() {
        let rna = RnaSequence::new("r", "AAACCC").expect("should succeed"); // no AUG
        let result = rna.translate();
        assert!(result.is_err());
    }

    #[test]
    fn test_rna_mfe_negative_for_complementary() {
        // Sequence that forms a perfect hairpin: GCGCGC...GCGCGC
        let rna = RnaSequence::new("r", "GCGCGCGCGCGCGCGCGCGC").expect("should succeed");
        let mfe = rna.secondary_structure_mfe_naive();
        assert!(mfe.is_some());
        assert!(mfe.expect("mfe expected") < 0.0);
    }

    #[test]
    fn test_rna_mfe_none_for_short() {
        let rna = RnaSequence::new("r", "AUG").expect("should succeed");
        let mfe = rna.secondary_structure_mfe_naive();
        assert!(mfe.is_none());
    }

    // ── ProteinSequence ────────────────────────────────────────────────────

    #[test]
    fn test_protein_new_valid() {
        let prot = ProteinSequence::new("p1", "MKVL").expect("valid protein");
        assert_eq!(prot.len(), 4);
    }

    #[test]
    fn test_protein_new_invalid_char() {
        let result = ProteinSequence::new("p1", "MXVL"); // X not standard
        assert!(result.is_err());
    }

    #[test]
    fn test_protein_as_str_roundtrip() {
        let seq = "MKVLWSTREK";
        let prot = ProteinSequence::new("p", seq).expect("should succeed");
        assert_eq!(prot.as_str(), seq);
    }

    #[test]
    fn test_protein_molecular_weight_positive() {
        let prot = ProteinSequence::new("p", "MKVL").expect("should succeed");
        let mw = prot.molecular_weight();
        assert!(mw > 0.0);
    }

    #[test]
    fn test_protein_isoelectric_point_range() {
        // Should be in physiological range
        let prot = ProteinSequence::new("p", "MKVLWSTREK").expect("should succeed");
        let pi = prot.isoelectric_point_approx();
        assert!(pi >= 0.0 && pi <= 14.0, "pI {pi} out of [0,14] range");
    }

    #[test]
    fn test_protein_hydrophobicity_hydrophobic() {
        // IILVVF = all hydrophobic → positive mean KD
        let prot = ProteinSequence::new("p", "IILVVF").expect("should succeed");
        let hyd = prot.hydrophobicity_kyte_doolittle();
        assert!(hyd > 0.0, "expected positive hydrophobicity");
    }

    #[test]
    fn test_protein_hydrophobicity_hydrophilic() {
        // RKDE = all charged/hydrophilic → negative mean KD
        let prot = ProteinSequence::new("p", "RKDE").expect("should succeed");
        let hyd = prot.hydrophobicity_kyte_doolittle();
        assert!(hyd < 0.0, "expected negative hydrophobicity");
    }

    #[test]
    fn test_protein_composition() {
        let prot = ProteinSequence::new("p", "MMMKVL").expect("should succeed");
        let comp = prot.composition();
        assert_eq!(*comp.get(&AminoAcid::Met).expect("Met missing"), 3);
    }

    #[test]
    fn test_one_letter_conversion() {
        assert_eq!(one_letter_to_amino_acid('M'), Some(AminoAcid::Met));
        assert_eq!(one_letter_to_amino_acid('*'), Some(AminoAcid::Stop));
        assert_eq!(one_letter_to_amino_acid('X'), None);
    }

    #[test]
    fn test_kyte_doolittle_values() {
        assert!((kyte_doolittle(AminoAcid::Ile) - 4.5).abs() < 1e-6);
        assert!((kyte_doolittle(AminoAcid::Arg) - (-4.5)).abs() < 1e-6);
    }
}
