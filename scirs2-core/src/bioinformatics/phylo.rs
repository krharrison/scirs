//! Phylogenetic distance calculations for bioinformatics.
//!
//! Provides:
//! - Hamming distance – count of positional differences.
//! - Jukes-Cantor distance – evolutionary correction for multiple substitutions.
//! - Pairwise distance matrix over a collection of sequences.

use crate::error::{CoreError, CoreResult};
use crate::ndarray_ext::Array2;

// ─── Hamming distance ─────────────────────────────────────────────────────────

/// Returns the Hamming distance between two sequences of equal length.
///
/// The Hamming distance is the number of positions at which corresponding
/// characters differ.
///
/// # Errors
///
/// Returns `CoreError::DimensionError` if `s1` and `s2` have different lengths.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::phylo::hamming_distance;
///
/// assert_eq!(hamming_distance(b"ATGC", b"ATGC").expect("should succeed"), 0);
/// assert_eq!(hamming_distance(b"ATGC", b"TTGT").expect("should succeed"), 2);
/// ```
pub fn hamming_distance(s1: &[u8], s2: &[u8]) -> CoreResult<usize> {
    if s1.len() != s2.len() {
        return Err(CoreError::DimensionError(crate::error_context!(format!(
            "hamming_distance: sequences must have equal length ({} vs {})",
            s1.len(),
            s2.len()
        ))));
    }
    let dist = s1
        .iter()
        .zip(s2.iter())
        .filter(|(&a, &b)| !a.eq_ignore_ascii_case(&b))
        .count();
    Ok(dist)
}

// ─── Jukes-Cantor distance ────────────────────────────────────────────────────

/// Converts a p-distance (fraction of differing sites) to a Jukes-Cantor
/// evolutionary distance.
///
/// The Jukes-Cantor model (1969) corrects for the possibility of multiple
/// substitutions at the same site under the assumption of equal base
/// frequencies and equal substitution rates.
///
/// ## Formula
///
/// ```text
/// d_JC = -3/4 × ln(1 − 4p/3)
/// ```
///
/// The formula is undefined for `p >= 0.75`.
///
/// # Parameters
///
/// - `p` – observed p-distance (fraction of differing positions), in `[0, 1)`.
///
/// # Errors
///
/// Returns `CoreError::DomainError` if `p < 0`, `p >= 0.75`, or `p` is NaN.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::phylo::jukes_cantor_distance;
///
/// // p=0 → d=0
/// assert!((jukes_cantor_distance(0.0).expect("should succeed") - 0.0).abs() < 1e-10);
///
/// // p=0.1 → d ≈ 0.10733 (-3/4 * ln(1 - 4/3 * 0.1))
/// let d = jukes_cantor_distance(0.1).expect("should succeed");
/// assert!((d - 0.10733).abs() < 1e-4);
/// ```
pub fn jukes_cantor_distance(p: f64) -> CoreResult<f64> {
    if p.is_nan() || p < 0.0 || p >= 0.75 {
        return Err(CoreError::DomainError(crate::error_context!(format!(
            "jukes_cantor_distance: p must be in [0, 0.75), got {p}"
        ))));
    }
    if p == 0.0 {
        return Ok(0.0);
    }
    let d = -0.75 * (1.0 - (4.0 / 3.0) * p).ln();
    Ok(d)
}

// ─── Pairwise distance matrix ─────────────────────────────────────────────────

/// Computes the pairwise Jukes-Cantor distance matrix for a set of sequences.
///
/// All sequences must have the same length.  The matrix is symmetric with
/// zero on the diagonal.  For each pair `(i, j)` the Hamming p-distance is
/// computed and then converted to a Jukes-Cantor distance.
///
/// When the p-distance is ≥ 0.75 (no valid JC estimate), the entry is set to
/// `f64::INFINITY`.
///
/// # Parameters
///
/// - `sequences` – slice of byte slices, all of the same length.
///
/// # Errors
///
/// Returns `CoreError::DimensionError` if the input is empty or if sequences
/// have different lengths.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::phylo::distance_matrix;
///
/// let seqs: &[&[u8]] = &[b"ATGC", b"ATGC", b"TTGT"];
/// let mat = distance_matrix(seqs).expect("should succeed");
/// assert_eq!(mat.shape(), &[3, 3]);
/// assert_eq!(mat[[0, 0]], 0.0);
/// assert_eq!(mat[[0, 1]], 0.0); // identical
/// assert!(mat[[0, 2]] > 0.0);   // differ at 2/4 sites
/// ```
pub fn distance_matrix(sequences: &[&[u8]]) -> CoreResult<Array2<f64>> {
    let n = sequences.len();
    if n == 0 {
        return Err(CoreError::DimensionError(crate::error_context!(
            "distance_matrix: sequences slice must not be empty"
        )));
    }

    let seq_len = sequences[0].len();
    for (idx, seq) in sequences.iter().enumerate() {
        if seq.len() != seq_len {
            return Err(CoreError::DimensionError(crate::error_context!(format!(
                "distance_matrix: sequence {idx} has length {} but expected {seq_len}",
                seq.len()
            ))));
        }
    }

    let mut mat = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in (i + 1)..n {
            let h = hamming_distance(sequences[i], sequences[j])?;
            let p = if seq_len == 0 {
                0.0
            } else {
                h as f64 / seq_len as f64
            };
            let d = jukes_cantor_distance(p).unwrap_or(f64::INFINITY);
            mat[[i, j]] = d;
            mat[[j, i]] = d;
        }
    }

    Ok(mat)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── hamming_distance ───────────────────────────────────────────────────

    #[test]
    fn test_hamming_identical() {
        assert_eq!(
            hamming_distance(b"ATGC", b"ATGC").expect("hamming failed"),
            0
        );
    }

    #[test]
    fn test_hamming_all_different() {
        assert_eq!(
            hamming_distance(b"ATGC", b"TACG").expect("hamming failed"),
            4
        );
    }

    #[test]
    fn test_hamming_two_differences() {
        assert_eq!(
            hamming_distance(b"ATGC", b"TTGT").expect("hamming failed"),
            2
        );
    }

    #[test]
    fn test_hamming_length_mismatch() {
        let result = hamming_distance(b"ATG", b"AT");
        assert!(result.is_err());
    }

    #[test]
    fn test_hamming_empty() {
        assert_eq!(hamming_distance(b"", b"").expect("hamming failed"), 0);
    }

    #[test]
    fn test_hamming_case_insensitive() {
        assert_eq!(
            hamming_distance(b"atgc", b"ATGC").expect("hamming failed"),
            0
        );
    }

    // ── jukes_cantor_distance ──────────────────────────────────────────────

    #[test]
    fn test_jc_zero_p() {
        assert!((jukes_cantor_distance(0.0).expect("jc failed") - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_jc_small_p() {
        // p=0.1 → d = -3/4 * ln(1 - 4/3 * 0.1) ≈ 0.107326
        let d = jukes_cantor_distance(0.1).expect("jc failed");
        assert!((d - 0.107326).abs() < 1e-4, "got {d}");
    }

    #[test]
    fn test_jc_p_above_threshold_errors() {
        let result = jukes_cantor_distance(0.75);
        assert!(result.is_err());
    }

    #[test]
    fn test_jc_negative_p_errors() {
        let result = jukes_cantor_distance(-0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_jc_nan_p_errors() {
        let result = jukes_cantor_distance(f64::NAN);
        assert!(result.is_err());
    }

    #[test]
    fn test_jc_monotone_increasing() {
        let d1 = jukes_cantor_distance(0.1).expect("jc failed");
        let d2 = jukes_cantor_distance(0.2).expect("jc failed");
        let d3 = jukes_cantor_distance(0.5).expect("jc failed");
        assert!(d1 < d2, "JC distance should be monotonically increasing");
        assert!(d2 < d3, "JC distance should be monotonically increasing");
    }

    // ── distance_matrix ────────────────────────────────────────────────────

    #[test]
    fn test_distance_matrix_shape() {
        let seqs: &[&[u8]] = &[b"ATGC", b"ATGC", b"TTGT"];
        let mat = distance_matrix(seqs).expect("distance_matrix failed");
        assert_eq!(mat.shape(), &[3, 3]);
    }

    #[test]
    fn test_distance_matrix_zero_diagonal() {
        let seqs: &[&[u8]] = &[b"ATGC", b"TTGT", b"GGCC"];
        let mat = distance_matrix(seqs).expect("distance_matrix failed");
        for i in 0..3 {
            assert_eq!(mat[[i, i]], 0.0, "diagonal must be zero at ({i},{i})");
        }
    }

    #[test]
    fn test_distance_matrix_symmetric() {
        let seqs: &[&[u8]] = &[b"ATGC", b"TTGT", b"GGCC"];
        let mat = distance_matrix(seqs).expect("distance_matrix failed");
        for i in 0..3 {
            for j in 0..3 {
                let a = mat[[i, j]];
                let b = mat[[j, i]];
                let symmetric = if a.is_infinite() && b.is_infinite() {
                    a.signum() == b.signum()
                } else {
                    (a - b).abs() < 1e-15
                };
                assert!(symmetric, "matrix must be symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_distance_matrix_identical_sequences() {
        let seqs: &[&[u8]] = &[b"ATGC", b"ATGC"];
        let mat = distance_matrix(seqs).expect("distance_matrix failed");
        assert_eq!(mat[[0, 1]], 0.0);
        assert_eq!(mat[[1, 0]], 0.0);
    }

    #[test]
    fn test_distance_matrix_length_mismatch_errors() {
        let seqs: &[&[u8]] = &[b"ATGC", b"ATG"];
        let result = distance_matrix(seqs);
        assert!(result.is_err());
    }

    #[test]
    fn test_distance_matrix_empty_errors() {
        let seqs: &[&[u8]] = &[];
        let result = distance_matrix(seqs);
        assert!(result.is_err());
    }
}
