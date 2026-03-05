//! Sequence alignment algorithms for bioinformatics.
//!
//! Provides classic dynamic-programming alignment methods:
//! - **Needleman-Wunsch** – global pairwise alignment (Needleman & Wunsch, 1970)
//! - **Smith-Waterman** – local pairwise alignment (Smith & Waterman, 1981)
//! - **Edit distance** – Levenshtein distance for byte sequences

use crate::error::CoreResult;

// ─── Needleman-Wunsch global alignment ───────────────────────────────────────

/// Performs global pairwise sequence alignment using the Needleman-Wunsch
/// dynamic-programming algorithm.
///
/// # Parameters
///
/// - `seq1`, `seq2` – the two sequences to align (ASCII bytes, case-sensitive).
/// - `match_score`  – score added for matching characters.
/// - `mismatch`     – score added for mismatching characters (typically negative).
/// - `gap`          – gap penalty applied to each gap character introduced
///   (typically negative; the same value is used for opening and
///   extension, i.e. a linear gap model).
///
/// # Returns
///
/// `(score, aligned1, aligned2)` where `aligned1` and `aligned2` are the
/// aligned sequences in ASCII; gaps are represented by `'-'`.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::alignment::needleman_wunsch;
///
/// let (score, a1, a2) = needleman_wunsch(b"AGCT", b"AGT", 1, -1, -2).expect("should succeed");
/// assert!(score > 0);
/// assert_eq!(a1.len(), a2.len());
/// ```
pub fn needleman_wunsch(
    seq1: &[u8],
    seq2: &[u8],
    match_score: i32,
    mismatch: i32,
    gap: i32,
) -> CoreResult<(i32, String, String)> {
    let m = seq1.len();
    let n = seq2.len();

    // Allocate score matrix using flat Vec<i32> for cache efficiency.
    let mut dp = vec![0i32; (m + 1) * (n + 1)];

    // Initialise first row and column with cumulative gap penalties.
    for i in 0..=m {
        dp[i * (n + 1)] = i as i32 * gap;
    }
    for j in 0..=n {
        dp[j] = j as i32 * gap;
    }

    // Fill the DP table.
    for i in 1..=m {
        for j in 1..=n {
            let diag = dp[(i - 1) * (n + 1) + (j - 1)]
                + if seq1[i - 1] == seq2[j - 1] {
                    match_score
                } else {
                    mismatch
                };
            let up = dp[(i - 1) * (n + 1) + j] + gap;
            let left = dp[i * (n + 1) + (j - 1)] + gap;
            dp[i * (n + 1) + j] = diag.max(up).max(left);
        }
    }

    let score = dp[m * (n + 1) + n];
    let (a1, a2) = traceback_nw(&dp, seq1, seq2, m, n, match_score, mismatch, gap);
    Ok((score, a1, a2))
}

/// Traces back through the Needleman-Wunsch DP table to reconstruct the
/// optimal global alignment.
fn traceback_nw(
    dp: &[i32],
    seq1: &[u8],
    seq2: &[u8],
    m: usize,
    n: usize,
    match_score: i32,
    mismatch: i32,
    gap: i32,
) -> (String, String) {
    let cols = n + 1;
    let mut aligned1 = Vec::new();
    let mut aligned2 = Vec::new();

    let mut i = m;
    let mut j = n;

    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let expected_diag = dp[(i - 1) * cols + (j - 1)]
                + if seq1[i - 1] == seq2[j - 1] {
                    match_score
                } else {
                    mismatch
                };
            if dp[i * cols + j] == expected_diag {
                aligned1.push(seq1[i - 1]);
                aligned2.push(seq2[j - 1]);
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && dp[i * cols + j] == dp[(i - 1) * cols + j] + gap {
            aligned1.push(seq1[i - 1]);
            aligned2.push(b'-');
            i -= 1;
        } else {
            // j > 0, move left
            aligned1.push(b'-');
            aligned2.push(seq2[j - 1]);
            j -= 1;
        }
    }

    aligned1.reverse();
    aligned2.reverse();

    // SAFETY: we only push valid ASCII bytes (from inputs) or b'-'.
    let s1 = String::from_utf8(aligned1).unwrap_or_default();
    let s2 = String::from_utf8(aligned2).unwrap_or_default();
    (s1, s2)
}

// ─── Smith-Waterman local alignment ──────────────────────────────────────────

/// Performs local pairwise sequence alignment using the Smith-Waterman
/// dynamic-programming algorithm.
///
/// # Parameters
///
/// Same as [`needleman_wunsch`]; a linear gap model is used.
///
/// # Returns
///
/// `(score, aligned1, aligned2)` containing the highest-scoring local
/// alignment.  Returns empty strings if no positive-scoring alignment exists.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::alignment::smith_waterman;
///
/// let (score, a1, a2) = smith_waterman(b"TGTTACGG", b"GGTTGACTA", 3, -3, -2).expect("should succeed");
/// assert!(score > 0);
/// assert_eq!(a1.len(), a2.len());
/// ```
pub fn smith_waterman(
    seq1: &[u8],
    seq2: &[u8],
    match_score: i32,
    mismatch: i32,
    gap: i32,
) -> CoreResult<(i32, String, String)> {
    let m = seq1.len();
    let n = seq2.len();

    let mut dp = vec![0i32; (m + 1) * (n + 1)];
    let mut best_score = 0i32;
    let mut best_i = 0usize;
    let mut best_j = 0usize;

    for i in 1..=m {
        for j in 1..=n {
            let diag = dp[(i - 1) * (n + 1) + (j - 1)]
                + if seq1[i - 1] == seq2[j - 1] {
                    match_score
                } else {
                    mismatch
                };
            let up = dp[(i - 1) * (n + 1) + j] + gap;
            let left = dp[i * (n + 1) + (j - 1)] + gap;
            let cell = diag.max(up).max(left).max(0);
            dp[i * (n + 1) + j] = cell;
            if cell > best_score {
                best_score = cell;
                best_i = i;
                best_j = j;
            }
        }
    }

    if best_score == 0 {
        return Ok((0, String::new(), String::new()));
    }

    let (a1, a2) = traceback_sw(
        &dp,
        seq1,
        seq2,
        best_i,
        best_j,
        n + 1,
        match_score,
        mismatch,
        gap,
    );
    Ok((best_score, a1, a2))
}

/// Traces back from the highest-scoring cell in the Smith-Waterman DP table
/// until a cell with value 0 is reached.
fn traceback_sw(
    dp: &[i32],
    seq1: &[u8],
    seq2: &[u8],
    start_i: usize,
    start_j: usize,
    cols: usize,
    match_score: i32,
    mismatch: i32,
    gap: i32,
) -> (String, String) {
    let mut aligned1 = Vec::new();
    let mut aligned2 = Vec::new();

    let mut i = start_i;
    let mut j = start_j;

    while i > 0 && j > 0 && dp[i * cols + j] > 0 {
        let expected_diag = dp[(i - 1) * cols + (j - 1)]
            + if seq1[i - 1] == seq2[j - 1] {
                match_score
            } else {
                mismatch
            };
        if dp[i * cols + j] == expected_diag {
            aligned1.push(seq1[i - 1]);
            aligned2.push(seq2[j - 1]);
            i -= 1;
            j -= 1;
        } else if i > 0 && dp[i * cols + j] == dp[(i - 1) * cols + j] + gap {
            aligned1.push(seq1[i - 1]);
            aligned2.push(b'-');
            i -= 1;
        } else {
            aligned1.push(b'-');
            aligned2.push(seq2[j - 1]);
            j -= 1;
        }
    }

    aligned1.reverse();
    aligned2.reverse();

    let s1 = String::from_utf8(aligned1).unwrap_or_default();
    let s2 = String::from_utf8(aligned2).unwrap_or_default();
    (s1, s2)
}

// ─── Levenshtein (edit) distance ─────────────────────────────────────────────

/// Computes the Levenshtein edit distance between two byte sequences.
///
/// The edit distance is the minimum number of single-character edits
/// (insertions, deletions, or substitutions) needed to transform `s1` into `s2`.
///
/// Uses O(min(|s1|, |s2|)) space via the two-row optimisation.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::alignment::edit_distance;
///
/// assert_eq!(edit_distance(b"kitten", b"sitting"), 3);
/// assert_eq!(edit_distance(b"ATGC", b"ATGC"), 0);
/// assert_eq!(edit_distance(b"", b"ABC"), 3);
/// ```
#[must_use]
pub fn edit_distance(s1: &[u8], s2: &[u8]) -> usize {
    // Ensure s1 is the shorter sequence to minimise memory.
    if s1.len() > s2.len() {
        return edit_distance(s2, s1);
    }

    let m = s1.len();
    let n = s2.len();

    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr = vec![0usize; m + 1];

    for j in 1..=n {
        curr[0] = j;
        for i in 1..=m {
            let cost = if s1[i - 1] == s2[j - 1] { 0 } else { 1 };
            curr[i] = (prev[i] + 1)           // deletion
                .min(curr[i - 1] + 1)          // insertion
                .min(prev[i - 1] + cost); // substitution
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m]
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── edit_distance ──────────────────────────────────────────────────────

    #[test]
    fn test_edit_distance_identical() {
        assert_eq!(edit_distance(b"ATGC", b"ATGC"), 0);
    }

    #[test]
    fn test_edit_distance_one_deletion() {
        assert_eq!(edit_distance(b"ATGC", b"ATG"), 1);
    }

    #[test]
    fn test_edit_distance_one_insertion() {
        assert_eq!(edit_distance(b"ATG", b"ATGC"), 1);
    }

    #[test]
    fn test_edit_distance_one_substitution() {
        assert_eq!(edit_distance(b"ATGC", b"TTGC"), 1);
    }

    #[test]
    fn test_edit_distance_kitten_sitting() {
        assert_eq!(edit_distance(b"kitten", b"sitting"), 3);
    }

    #[test]
    fn test_edit_distance_empty_strings() {
        assert_eq!(edit_distance(b"", b""), 0);
    }

    #[test]
    fn test_edit_distance_one_empty() {
        assert_eq!(edit_distance(b"", b"ATGC"), 4);
        assert_eq!(edit_distance(b"ATGC", b""), 4);
    }

    // ── needleman_wunsch ───────────────────────────────────────────────────

    #[test]
    fn test_nw_identical_sequences() {
        let (score, a1, a2) = needleman_wunsch(b"ATGC", b"ATGC", 1, -1, -2).expect("NW failed");
        assert_eq!(score, 4);
        assert_eq!(a1, "ATGC");
        assert_eq!(a2, "ATGC");
    }

    #[test]
    fn test_nw_single_gap() {
        // AGCT vs AGT: one deletion expected
        let (score, a1, a2) = needleman_wunsch(b"AGCT", b"AGT", 1, -1, -2).expect("NW failed");
        assert_eq!(
            a1.len(),
            a2.len(),
            "aligned sequences must have equal length"
        );
        // Optimal: AGCT / AG-T → score = 1+1+(-2)+1 = 1
        assert_eq!(score, 1);
    }

    #[test]
    fn test_nw_aligned_lengths_equal() {
        let (_, a1, a2) = needleman_wunsch(b"GCATGCU", b"GATTACA", 1, -1, -1).expect("NW failed");
        assert_eq!(a1.len(), a2.len());
    }

    #[test]
    fn test_nw_empty_sequences() {
        let (score, a1, a2) = needleman_wunsch(b"ATGC", b"", 1, -1, -2).expect("NW failed");
        assert_eq!(score, -8); // 4 gaps × (-2)
        assert_eq!(a1, "ATGC");
        assert_eq!(a2, "----");
    }

    // ── smith_waterman ─────────────────────────────────────────────────────

    #[test]
    fn test_sw_classic_example() {
        // Classic example from Smith-Waterman (1981)
        let (score, a1, a2) =
            smith_waterman(b"TGTTACGG", b"GGTTGACTA", 3, -3, -2).expect("SW failed");
        assert!(score > 0, "SW score must be positive");
        assert_eq!(
            a1.len(),
            a2.len(),
            "aligned sequences must have equal length"
        );
    }

    #[test]
    fn test_sw_identical_sequences() {
        let (score, a1, a2) = smith_waterman(b"ATGC", b"ATGC", 2, -1, -1).expect("SW failed");
        assert_eq!(score, 8); // 4 matches × 2
        assert_eq!(a1, "ATGC");
        assert_eq!(a2, "ATGC");
    }

    #[test]
    fn test_sw_no_match_returns_empty() {
        // Completely different sequences with high penalties should score 0
        let (score, a1, a2) = smith_waterman(b"AAAA", b"TTTT", 1, -10, -10).expect("SW failed");
        assert_eq!(score, 0);
        assert!(a1.is_empty());
        assert!(a2.is_empty());
    }

    #[test]
    fn test_sw_local_region() {
        // seq1 contains seq2 as a substring – SW should find perfect match
        let (score, a1, a2) =
            smith_waterman(b"NNNNATGCNNNN", b"ATGC", 2, -1, -1).expect("SW failed");
        assert_eq!(score, 8); // 4 matches × 2
        assert_eq!(a1, "ATGC");
        assert_eq!(a2, "ATGC");
    }
}
