//! Statistical analysis of biological sequences.
//!
//! Provides functions to characterise sequence composition:
//! - Mono-nucleotide frequencies (ACGT)
//! - Di-nucleotide frequencies
//! - Codon usage bias tables

use std::collections::HashMap;

use crate::error::{CoreError, CoreResult};

// ─── Nucleotide frequencies ───────────────────────────────────────────────────

/// Computes the relative frequencies of A, C, G, T in `seq`.
///
/// The returned array is indexed as `[A, C, G, T]` (alphabetical order).
/// Ambiguous bases (`N`) and any non-ACGT characters are ignored when
/// computing the denominator.
///
/// Returns `[0.0; 4]` if `seq` contains no ACGT bases.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::stats::nucleotide_frequencies;
///
/// let freqs = nucleotide_frequencies(b"AACGT");
/// // A=2, C=1, G=1, T=1 out of 5 total
/// assert!((freqs[0] - 0.4).abs() < 1e-10);
/// ```
#[must_use]
pub fn nucleotide_frequencies(seq: &[u8]) -> [f64; 4] {
    let mut counts = [0u64; 4]; // [A, C, G, T]
    for &b in seq {
        match b.to_ascii_uppercase() {
            b'A' => counts[0] += 1,
            b'C' => counts[1] += 1,
            b'G' => counts[2] += 1,
            b'T' => counts[3] += 1,
            _ => {}
        }
    }
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return [0.0; 4];
    }
    let t = total as f64;
    [
        counts[0] as f64 / t,
        counts[1] as f64 / t,
        counts[2] as f64 / t,
        counts[3] as f64 / t,
    ]
}

// ─── Di-nucleotide frequencies ────────────────────────────────────────────────

/// Computes the relative frequencies of all observed di-nucleotides in `seq`.
///
/// Each consecutive pair `seq[i], seq[i+1]` (with both bases being ACGT) is
/// counted.  The resulting HashMap maps `[b1, b2]` (uppercase) to the fraction
/// of all di-nucleotides represented by that pair.
///
/// Returns an empty map if `seq` contains fewer than two bases.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::stats::dinucleotide_frequencies;
///
/// let freqs = dinucleotide_frequencies(b"ATCG");
/// // Pairs: AT, TC, CG → each 1/3
/// assert_eq!(freqs.len(), 3);
/// ```
#[must_use]
pub fn dinucleotide_frequencies(seq: &[u8]) -> HashMap<[u8; 2], f64> {
    let mut counts: HashMap<[u8; 2], u64> = HashMap::new();

    if seq.len() < 2 {
        return HashMap::new();
    }

    let upper: Vec<u8> = seq.iter().map(|&b| b.to_ascii_uppercase()).collect();
    let mut total = 0u64;

    for window in upper.windows(2) {
        let b0 = window[0];
        let b1 = window[1];
        if is_acgt(b0) && is_acgt(b1) {
            *counts.entry([b0, b1]).or_insert(0) += 1;
            total += 1;
        }
    }

    if total == 0 {
        return HashMap::new();
    }

    let t = total as f64;
    counts.into_iter().map(|(k, v)| (k, v as f64 / t)).collect()
}

// ─── Codon usage table ────────────────────────────────────────────────────────

/// Computes the codon usage bias table for a DNA sequence.
///
/// Reading frame 0 is used (i.e. codons start at position 0).  Only complete
/// codons composed entirely of ACGT bases are counted.  The result maps each
/// observed 3-mer to its relative frequency among all counted codons.
///
/// Returns an empty map if there are no complete ACGT codons.
///
/// # Errors
///
/// Returns `CoreError::ValueError` if `dna` contains non-ASCII bytes.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::stats::codon_usage_table;
///
/// // ATG appears twice, TAA once → ATG = 2/3, TAA = 1/3
/// let table = codon_usage_table(b"ATGATGTAA").expect("should succeed");
/// let atg_freq = table[&[b'A', b'T', b'G']];
/// assert!((atg_freq - 2.0/3.0).abs() < 1e-10);
/// ```
pub fn codon_usage_table(dna: &[u8]) -> CoreResult<HashMap<[u8; 3], f64>> {
    let upper: Vec<u8> = dna.iter().map(|&b| b.to_ascii_uppercase()).collect();

    let mut counts: HashMap<[u8; 3], u64> = HashMap::new();
    let mut total = 0u64;

    let mut i = 0;
    while i + 3 <= upper.len() {
        let b0 = upper[i];
        let b1 = upper[i + 1];
        let b2 = upper[i + 2];
        if is_acgt(b0) && is_acgt(b1) && is_acgt(b2) {
            *counts.entry([b0, b1, b2]).or_insert(0) += 1;
            total += 1;
        } else {
            // Validate that the byte is at least valid ASCII; ambiguous (N) is
            // allowed but skipped silently, other chars are errors.
            for &b in &[b0, b1, b2] {
                if !b.is_ascii() {
                    return Err(CoreError::ValueError(crate::error_context!(format!(
                        "Non-ASCII byte 0x{b:02X} found in DNA sequence"
                    ))));
                }
            }
        }
        i += 3;
    }

    if total == 0 {
        return Ok(HashMap::new());
    }

    let t = total as f64;
    Ok(counts.into_iter().map(|(k, v)| (k, v as f64 / t)).collect())
}

// ─── Helper ───────────────────────────────────────────────────────────────────

#[inline]
fn is_acgt(b: u8) -> bool {
    matches!(b, b'A' | b'C' | b'G' | b'T')
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── nucleotide_frequencies ─────────────────────────────────────────────

    #[test]
    fn test_nuc_freq_uniform() {
        // Each base appears exactly once
        let freqs = nucleotide_frequencies(b"ACGT");
        for &f in &freqs {
            assert!((f - 0.25).abs() < 1e-10, "expected 0.25, got {f}");
        }
    }

    #[test]
    fn test_nuc_freq_only_a() {
        let freqs = nucleotide_frequencies(b"AAAA");
        assert!((freqs[0] - 1.0).abs() < 1e-10);
        assert_eq!(freqs[1], 0.0);
        assert_eq!(freqs[2], 0.0);
        assert_eq!(freqs[3], 0.0);
    }

    #[test]
    fn test_nuc_freq_empty() {
        let freqs = nucleotide_frequencies(b"");
        assert_eq!(freqs, [0.0; 4]);
    }

    #[test]
    fn test_nuc_freq_ignores_n() {
        // N should be excluded from both numerator and denominator
        let freqs = nucleotide_frequencies(b"ACGTN");
        let sum: f64 = freqs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "sum should be 1.0, got {sum}");
    }

    #[test]
    fn test_nuc_freq_order_acgt() {
        // freqs[0]=A, freqs[1]=C, freqs[2]=G, freqs[3]=T
        let freqs = nucleotide_frequencies(b"AACGT");
        assert!((freqs[0] - 2.0 / 5.0).abs() < 1e-10); // A = 2/5
        assert!((freqs[1] - 1.0 / 5.0).abs() < 1e-10); // C = 1/5
        assert!((freqs[2] - 1.0 / 5.0).abs() < 1e-10); // G = 1/5
        assert!((freqs[3] - 1.0 / 5.0).abs() < 1e-10); // T = 1/5
    }

    // ── dinucleotide_frequencies ───────────────────────────────────────────

    #[test]
    fn test_dinuc_freq_basic() {
        let freqs = dinucleotide_frequencies(b"ATCG");
        // Pairs: AT, TC, CG – each has frequency 1/3
        assert_eq!(freqs.len(), 3);
        for &v in freqs.values() {
            assert!((v - 1.0 / 3.0).abs() < 1e-10, "expected 1/3, got {v}");
        }
    }

    #[test]
    fn test_dinuc_freq_repeated() {
        let freqs = dinucleotide_frequencies(b"ATATAT");
        // Pairs: AT, TA, AT, TA, AT → AT = 3/5, TA = 2/5
        let at_freq = freqs[&[b'A', b'T']];
        let ta_freq = freqs[&[b'T', b'A']];
        assert!((at_freq - 3.0 / 5.0).abs() < 1e-10);
        assert!((ta_freq - 2.0 / 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_dinuc_freq_empty_sequence() {
        let freqs = dinucleotide_frequencies(b"");
        assert!(freqs.is_empty());
    }

    #[test]
    fn test_dinuc_freq_single_base() {
        let freqs = dinucleotide_frequencies(b"A");
        assert!(freqs.is_empty());
    }

    #[test]
    fn test_dinuc_freq_sums_to_one() {
        let freqs = dinucleotide_frequencies(b"ATGCATGCNN");
        if !freqs.is_empty() {
            let sum: f64 = freqs.values().sum();
            assert!((sum - 1.0).abs() < 1e-10, "sum should be 1.0, got {sum}");
        }
    }

    // ── codon_usage_table ─────────────────────────────────────────────────

    #[test]
    fn test_codon_usage_two_atg_one_taa() {
        let table = codon_usage_table(b"ATGATGTAA").expect("codon_usage_table failed");
        let atg = table[&[b'A', b'T', b'G']];
        let taa = table[&[b'T', b'A', b'A']];
        assert!(
            (atg - 2.0 / 3.0).abs() < 1e-10,
            "ATG expected 2/3, got {atg}"
        );
        assert!(
            (taa - 1.0 / 3.0).abs() < 1e-10,
            "TAA expected 1/3, got {taa}"
        );
    }

    #[test]
    fn test_codon_usage_empty() {
        let table = codon_usage_table(b"").expect("codon_usage_table failed");
        assert!(table.is_empty());
    }

    #[test]
    fn test_codon_usage_sums_to_one() {
        let table = codon_usage_table(b"ATGAAACCCGGG").expect("codon_usage_table failed");
        let sum: f64 = table.values().sum();
        assert!((sum - 1.0).abs() < 1e-10, "sum should be 1.0, got {sum}");
    }

    #[test]
    fn test_codon_usage_skips_ambiguous() {
        // NNN should be skipped; ATG counts
        let table = codon_usage_table(b"ATGNNN").expect("codon_usage_table failed");
        assert_eq!(table.len(), 1);
        let atg = table[&[b'A', b'T', b'G']];
        assert!((atg - 1.0).abs() < 1e-10);
    }
}
