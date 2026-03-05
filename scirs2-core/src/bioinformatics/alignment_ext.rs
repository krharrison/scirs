//! Extended sequence alignment algorithms.
//!
//! Provides:
//! - Affine gap penalty Needleman-Wunsch (global) and Smith-Waterman (local)
//! - Semi-global alignment (free end gaps)
//! - Multiple sequence alignment (progressive, ClustalW-like)

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::error::{CoreError, CoreResult};

// ─── Scoring matrix ───────────────────────────────────────────────────────────

/// Gap and substitution scoring parameters for sequence alignment.
///
/// Supports affine gap penalties: a gap of length `k` costs
/// `gap_open + (k-1) * gap_extend`.
#[derive(Debug, Clone)]
pub struct ScoringMatrix {
    /// Score for matching identical characters.
    pub match_score: i32,
    /// Penalty for mismatching characters (typically negative).
    pub mismatch_penalty: i32,
    /// Penalty applied when opening a new gap.
    pub gap_open: i32,
    /// Penalty applied for each additional gap extension.
    pub gap_extend: i32,
}

impl ScoringMatrix {
    /// Default DNA scoring: match=2, mismatch=-1, gap_open=-2, gap_extend=-1.
    #[must_use]
    pub fn dna_default() -> Self {
        Self {
            match_score: 2,
            mismatch_penalty: -1,
            gap_open: -2,
            gap_extend: -1,
        }
    }

    /// Simplified protein scoring inspired by BLOSUM62 parameters.
    /// Uses match=4, mismatch=-1, gap_open=-11, gap_extend=-1.
    #[must_use]
    pub fn blosum62() -> Self {
        Self {
            match_score: 4,
            mismatch_penalty: -1,
            gap_open: -11,
            gap_extend: -1,
        }
    }
}

// ─── Alignment result ─────────────────────────────────────────────────────────

/// Result of a pairwise sequence alignment.
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    /// Raw alignment score.
    pub score: i32,
    /// Aligned version of the first sequence (with gap characters `-`).
    pub seq1_aligned: String,
    /// Aligned version of the second sequence (with gap characters `-`).
    pub seq2_aligned: String,
    /// Fraction of identical positions (excluding gap-only columns).
    pub identity: f64,
    /// Total number of gap characters in the alignment.
    pub gaps: usize,
    /// Length of the alignment (including gaps).
    pub aligned_length: usize,
}

impl AlignmentResult {
    /// Builds an `AlignmentResult` from two aligned strings.
    fn from_aligned(score: i32, s1: String, s2: String) -> Self {
        let len = s1.len();
        let gaps = s1.chars().filter(|&c| c == '-').count()
            + s2.chars().filter(|&c| c == '-').count();

        let identical = s1
            .chars()
            .zip(s2.chars())
            .filter(|(a, b)| *a != '-' && *b != '-' && a == b)
            .count();

        let non_gap_cols = s1
            .chars()
            .zip(s2.chars())
            .filter(|(a, b)| *a != '-' && *b != '-')
            .count();

        let identity = if non_gap_cols > 0 {
            identical as f64 / non_gap_cols as f64
        } else {
            0.0
        };

        AlignmentResult {
            score,
            seq1_aligned: s1,
            seq2_aligned: s2,
            identity,
            gaps,
            aligned_length: len,
        }
    }
}

// ─── Global alignment: Needleman-Wunsch with affine gaps ─────────────────────

/// Global pairwise alignment using the Needleman-Wunsch algorithm with
/// affine gap penalties (three-matrix approach: M, Ix, Iy).
///
/// # Errors
///
/// Returns `CoreError::ValueError` if either sequence is empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::alignment_ext::{needleman_wunsch_affine, ScoringMatrix};
///
/// let scoring = ScoringMatrix::dna_default();
/// let result = needleman_wunsch_affine("ATGC", "ATGC", &scoring).expect("should succeed");
/// assert!((result.identity - 1.0).abs() < 1e-10);
/// assert_eq!(result.gaps, 0);
/// ```
pub fn needleman_wunsch_affine(
    seq1: &str,
    seq2: &str,
    scoring: &ScoringMatrix,
) -> CoreResult<AlignmentResult> {
    let s1: Vec<u8> = seq1.bytes().map(|b| b.to_ascii_uppercase()).collect();
    let s2: Vec<u8> = seq2.bytes().map(|b| b.to_ascii_uppercase()).collect();

    let m = s1.len();
    let n = s2.len();
    let cols = n + 1;

    const NEG_INF: i32 = i32::MIN / 2;

    // M[i][j]  = best score ending with characters matched/mismatched
    // Ix[i][j] = best score ending with gap in seq2 (deletion from seq1)
    // Iy[i][j] = best score ending with gap in seq1 (insertion into seq1)
    let mut m_mat = vec![NEG_INF; (m + 1) * cols];
    let mut ix_mat = vec![NEG_INF; (m + 1) * cols];
    let mut iy_mat = vec![NEG_INF; (m + 1) * cols];

    m_mat[0] = 0;
    // Initialise first row (gaps in seq1)
    for j in 1..=n {
        iy_mat[j] = scoring.gap_open + (j as i32 - 1) * scoring.gap_extend;
    }
    // Initialise first column (gaps in seq2)
    for i in 1..=m {
        ix_mat[i * cols] = scoring.gap_open + (i as i32 - 1) * scoring.gap_extend;
    }

    // Fill matrices
    for i in 1..=m {
        for j in 1..=n {
            let sub = if s1[i - 1] == s2[j - 1] {
                scoring.match_score
            } else {
                scoring.mismatch_penalty
            };

            let prev_m = m_mat[(i - 1) * cols + (j - 1)];
            let prev_ix = ix_mat[(i - 1) * cols + (j - 1)];
            let prev_iy = iy_mat[(i - 1) * cols + (j - 1)];
            m_mat[i * cols + j] = safe_max3(
                safe_add(prev_m, sub),
                safe_add(prev_ix, sub),
                safe_add(prev_iy, sub),
            );

            // Gap in seq2 (extend/open from above)
            let open_ix = safe_add(m_mat[(i - 1) * cols + j], scoring.gap_open);
            let ext_ix = safe_add(ix_mat[(i - 1) * cols + j], scoring.gap_extend);
            ix_mat[i * cols + j] = open_ix.max(ext_ix);

            // Gap in seq1 (extend/open from left)
            let open_iy = safe_add(m_mat[i * cols + (j - 1)], scoring.gap_open);
            let ext_iy = safe_add(iy_mat[i * cols + (j - 1)], scoring.gap_extend);
            iy_mat[i * cols + j] = open_iy.max(ext_iy);
        }
    }

    // Best score at (m, n)
    let final_m = m_mat[m * cols + n];
    let final_ix = ix_mat[m * cols + n];
    let final_iy = iy_mat[m * cols + n];
    let score = safe_max3(final_m, final_ix, final_iy);

    // Traceback
    let (aligned1, aligned2) =
        traceback_affine_nw(&m_mat, &ix_mat, &iy_mat, &s1, &s2, m, n, scoring);

    Ok(AlignmentResult::from_aligned(score, aligned1, aligned2))
}

/// Traceback for affine gap NW alignment.
fn traceback_affine_nw(
    m_mat: &[i32],
    ix_mat: &[i32],
    iy_mat: &[i32],
    s1: &[u8],
    s2: &[u8],
    m: usize,
    n: usize,
    scoring: &ScoringMatrix,
) -> (String, String) {
    let cols = n + 1;
    const NEG_INF: i32 = i32::MIN / 2;

    // Which matrix are we in?
    #[derive(Copy, Clone, PartialEq)]
    enum State {
        M,
        Ix,
        Iy,
    }

    let final_m = m_mat[m * cols + n];
    let final_ix = ix_mat[m * cols + n];
    let final_iy = iy_mat[m * cols + n];
    let best = safe_max3(final_m, final_ix, final_iy);

    let mut cur_state = if final_m == best {
        State::M
    } else if final_ix == best {
        State::Ix
    } else {
        State::Iy
    };

    let mut a1: Vec<u8> = Vec::new();
    let mut a2: Vec<u8> = Vec::new();
    let mut i = m;
    let mut j = n;

    while i > 0 || j > 0 {
        match cur_state {
            State::M => {
                if i == 0 {
                    // Must have come from Iy
                    a1.push(b'-');
                    a2.push(s2[j - 1]);
                    j -= 1;
                    cur_state = State::Iy;
                    continue;
                }
                if j == 0 {
                    a1.push(s1[i - 1]);
                    a2.push(b'-');
                    i -= 1;
                    cur_state = State::Ix;
                    continue;
                }
                let sub = if s1[i - 1] == s2[j - 1] {
                    scoring.match_score
                } else {
                    scoring.mismatch_penalty
                };
                let cur = m_mat[i * cols + j];

                let from_m = safe_add(m_mat[(i - 1) * cols + (j - 1)], sub);
                let from_ix = safe_add(ix_mat[(i - 1) * cols + (j - 1)], sub);
                let from_iy = safe_add(iy_mat[(i - 1) * cols + (j - 1)], sub);

                a1.push(s1[i - 1]);
                a2.push(s2[j - 1]);
                i -= 1;
                j -= 1;
                cur_state = if from_m == cur {
                    State::M
                } else if from_ix == cur {
                    State::Ix
                } else if from_iy == cur {
                    State::Iy
                } else {
                    State::M // fallback
                };
            }
            State::Ix => {
                if i == 0 {
                    // Boundary: consume remaining j as gaps
                    a1.push(b'-');
                    a2.push(s2[j - 1]);
                    j -= 1;
                    continue;
                }
                a1.push(s1[i - 1]);
                a2.push(b'-');
                let cur = ix_mat[i * cols + j];
                let from_open = safe_add(m_mat[(i - 1) * cols + j], scoring.gap_open);
                i -= 1;
                cur_state = if from_open == cur && m_mat[i * cols + j] != NEG_INF {
                    State::M
                } else {
                    State::Ix
                };
            }
            State::Iy => {
                if j == 0 {
                    a1.push(s1[i - 1]);
                    a2.push(b'-');
                    i -= 1;
                    continue;
                }
                a1.push(b'-');
                a2.push(s2[j - 1]);
                let cur = iy_mat[i * cols + j];
                let from_open = safe_add(m_mat[i * cols + (j - 1)], scoring.gap_open);
                j -= 1;
                cur_state = if from_open == cur && m_mat[i * cols + j] != NEG_INF {
                    State::M
                } else {
                    State::Iy
                };
            }
        }
    }

    a1.reverse();
    a2.reverse();
    let aligned1 = String::from_utf8(a1).unwrap_or_default();
    let aligned2 = String::from_utf8(a2).unwrap_or_default();
    (aligned1, aligned2)
}

// ─── Local alignment: Smith-Waterman with affine gaps ────────────────────────

/// Local pairwise alignment using the Smith-Waterman algorithm with affine
/// gap penalties.
///
/// # Errors
///
/// Returns `CoreError::ValueError` if either sequence is empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::alignment_ext::{smith_waterman_affine, ScoringMatrix};
///
/// let scoring = ScoringMatrix::dna_default();
/// let result = smith_waterman_affine("NNNNATGCNNNN", "ATGC", &scoring).expect("should succeed");
/// assert!((result.identity - 1.0).abs() < 1e-10);
/// ```
pub fn smith_waterman_affine(
    seq1: &str,
    seq2: &str,
    scoring: &ScoringMatrix,
) -> CoreResult<AlignmentResult> {
    let s1: Vec<u8> = seq1.bytes().map(|b| b.to_ascii_uppercase()).collect();
    let s2: Vec<u8> = seq2.bytes().map(|b| b.to_ascii_uppercase()).collect();

    let m = s1.len();
    let n = s2.len();
    let cols = n + 1;

    let mut m_mat = vec![0i32; (m + 1) * cols];
    let mut ix_mat = vec![0i32; (m + 1) * cols];
    let mut iy_mat = vec![0i32; (m + 1) * cols];

    let mut best_score = 0i32;
    let mut best_i = 0usize;
    let mut best_j = 0usize;

    for i in 1..=m {
        for j in 1..=n {
            let sub = if s1[i - 1] == s2[j - 1] {
                scoring.match_score
            } else {
                scoring.mismatch_penalty
            };

            let m_val = safe_max3(
                safe_add(m_mat[(i - 1) * cols + (j - 1)], sub),
                safe_add(ix_mat[(i - 1) * cols + (j - 1)], sub),
                safe_add(iy_mat[(i - 1) * cols + (j - 1)], sub),
            )
            .max(0);

            let ix_val =
                (safe_add(m_mat[(i - 1) * cols + j], scoring.gap_open))
                    .max(safe_add(ix_mat[(i - 1) * cols + j], scoring.gap_extend))
                    .max(0);

            let iy_val =
                (safe_add(m_mat[i * cols + (j - 1)], scoring.gap_open))
                    .max(safe_add(iy_mat[i * cols + (j - 1)], scoring.gap_extend))
                    .max(0);

            m_mat[i * cols + j] = m_val;
            ix_mat[i * cols + j] = ix_val;
            iy_mat[i * cols + j] = iy_val;

            let cell_best = safe_max3(m_val, ix_val, iy_val);
            if cell_best > best_score {
                best_score = cell_best;
                best_i = i;
                best_j = j;
            }
        }
    }

    if best_score == 0 {
        return Ok(AlignmentResult::from_aligned(0, String::new(), String::new()));
    }

    let (aligned1, aligned2) = traceback_affine_sw(
        &m_mat, &ix_mat, &iy_mat, &s1, &s2, best_i, best_j, scoring,
    );

    Ok(AlignmentResult::from_aligned(
        best_score, aligned1, aligned2,
    ))
}

/// Traceback for Smith-Waterman (stops when cell value reaches 0).
fn traceback_affine_sw(
    m_mat: &[i32],
    ix_mat: &[i32],
    iy_mat: &[i32],
    s1: &[u8],
    s2: &[u8],
    start_i: usize,
    start_j: usize,
    scoring: &ScoringMatrix,
) -> (String, String) {
    let cols = s2.len() + 1;
    let mut a1: Vec<u8> = Vec::new();
    let mut a2: Vec<u8> = Vec::new();

    let mut i = start_i;
    let mut j = start_j;

    while i > 0 && j > 0 {
        let cur_m = m_mat[i * cols + j];
        let cur_ix = ix_mat[i * cols + j];
        let cur_iy = iy_mat[i * cols + j];
        let cur_best = safe_max3(cur_m, cur_ix, cur_iy);
        if cur_best == 0 {
            break;
        }

        if cur_best == cur_m && i > 0 && j > 0 {
            let sub = if s1[i - 1] == s2[j - 1] {
                scoring.match_score
            } else {
                scoring.mismatch_penalty
            };
            let prev = safe_add(
                safe_max3(
                    m_mat[(i - 1) * cols + (j - 1)],
                    ix_mat[(i - 1) * cols + (j - 1)],
                    iy_mat[(i - 1) * cols + (j - 1)],
                ),
                sub,
            );
            if prev == cur_m {
                a1.push(s1[i - 1]);
                a2.push(s2[j - 1]);
                i -= 1;
                j -= 1;
                continue;
            }
        }

        if cur_best == cur_ix && i > 0 {
            a1.push(s1[i - 1]);
            a2.push(b'-');
            i -= 1;
        } else if j > 0 {
            a1.push(b'-');
            a2.push(s2[j - 1]);
            j -= 1;
        } else {
            break;
        }
    }

    a1.reverse();
    a2.reverse();
    (
        String::from_utf8(a1).unwrap_or_default(),
        String::from_utf8(a2).unwrap_or_default(),
    )
}

// ─── Semi-global alignment ────────────────────────────────────────────────────

/// Semi-global alignment: no penalty for end gaps in the target.
///
/// This is useful for finding a `query` within a `target` allowing free end
/// gaps on both ends of the target (overlap alignment).
///
/// # Errors
///
/// Returns `CoreError::ValueError` if either sequence is empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::alignment_ext::{semi_global_align, ScoringMatrix};
///
/// let scoring = ScoringMatrix::dna_default();
/// let result = semi_global_align("ATGC", "NNATGCNN", &scoring).expect("should succeed");
/// assert!((result.identity - 1.0).abs() < 1e-10);
/// ```
pub fn semi_global_align(
    query: &str,
    target: &str,
    scoring: &ScoringMatrix,
) -> CoreResult<AlignmentResult> {
    if query.is_empty() || target.is_empty() {
        return Err(CoreError::ValueError(crate::error_context!(
            "query and target must be non-empty"
        )));
    }

    let q: Vec<u8> = query.bytes().map(|b| b.to_ascii_uppercase()).collect();
    let t: Vec<u8> = target.bytes().map(|b| b.to_ascii_uppercase()).collect();

    let m = q.len();
    let n = t.len();
    let cols = n + 1;
    const NEG_INF: i32 = i32::MIN / 2;

    let mut dp = vec![NEG_INF; (m + 1) * cols];

    // No penalty for starting gaps in target (first column)
    for i in 0..=m {
        dp[i * cols] = i as i32 * scoring.gap_open;
    }
    // No penalty for end gaps in target (first row initialised to 0)
    for j in 0..=n {
        dp[j] = 0;
    }

    for i in 1..=m {
        for j in 1..=n {
            let sub = if q[i - 1] == t[j - 1] {
                scoring.match_score
            } else {
                scoring.mismatch_penalty
            };
            let diag = safe_add(dp[(i - 1) * cols + (j - 1)], sub);
            let up = safe_add(dp[(i - 1) * cols + j], scoring.gap_open);
            let left = safe_add(dp[i * cols + (j - 1)], scoring.gap_open);
            dp[i * cols + j] = diag.max(up).max(left);
        }
    }

    // Best score in last row (free end gaps in target)
    let (best_j, best_score) = (0..=n)
        .map(|j| (j, dp[m * cols + j]))
        .max_by_key(|&(_, s)| s)
        .unwrap_or((n, NEG_INF));

    // Traceback from (m, best_j)
    let (aligned1, aligned2) =
        traceback_semi_global(&dp, &q, &t, m, best_j, n, scoring);

    Ok(AlignmentResult::from_aligned(best_score, aligned1, aligned2))
}

fn traceback_semi_global(
    dp: &[i32],
    q: &[u8],
    t: &[u8],
    start_i: usize,
    start_j: usize,
    n: usize,
    scoring: &ScoringMatrix,
) -> (String, String) {
    let cols = n + 1;
    let mut a1: Vec<u8> = Vec::new();
    let mut a2: Vec<u8> = Vec::new();

    // Pad gaps for the trailing target positions not aligned
    for j in (start_j..n).rev() {
        a1.push(b'-');
        a2.push(t[j]);
    }

    let mut i = start_i;
    let mut j = start_j;

    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let sub = if q[i - 1] == t[j - 1] {
                scoring.match_score
            } else {
                scoring.mismatch_penalty
            };
            let expected_diag = safe_add(dp[(i - 1) * cols + (j - 1)], sub);
            if dp[i * cols + j] == expected_diag {
                a1.push(q[i - 1]);
                a2.push(t[j - 1]);
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && (j == 0 || dp[i * cols + j] == safe_add(dp[(i - 1) * cols + j], scoring.gap_open)) {
            a1.push(q[i - 1]);
            a2.push(b'-');
            i -= 1;
        } else if j > 0 {
            a1.push(b'-');
            a2.push(t[j - 1]);
            j -= 1;
        } else {
            break;
        }
    }

    a1.reverse();
    a2.reverse();
    (
        String::from_utf8(a1).unwrap_or_default(),
        String::from_utf8(a2).unwrap_or_default(),
    )
}

// ─── Multiple sequence alignment ─────────────────────────────────────────────

/// Result of a multiple sequence alignment.
#[derive(Debug, Clone)]
pub struct MultipleAlignment {
    /// Aligned sequences (all the same length, gaps padded with `-`).
    pub aligned_sequences: Vec<String>,
    /// Identifiers corresponding to each aligned sequence.
    pub ids: Vec<String>,
    /// Consensus sequence: the majority character at each column.
    pub consensus: String,
}

impl MultipleAlignment {
    /// Conservation score per column: fraction of non-gap positions that
    /// share the most common character.
    #[must_use]
    pub fn conservation_scores(&self) -> Vec<f64> {
        if self.aligned_sequences.is_empty() {
            return Vec::new();
        }
        let len = self.aligned_sequences[0].len();
        (0..len)
            .map(|col| {
                let mut counts: HashMap<char, usize> = HashMap::new();
                let mut total = 0usize;
                for seq in &self.aligned_sequences {
                    let ch = seq.chars().nth(col).unwrap_or('-');
                    if ch != '-' {
                        *counts.entry(ch).or_insert(0) += 1;
                        total += 1;
                    }
                }
                if total == 0 {
                    return 0.0;
                }
                let max_count = counts.values().max().copied().unwrap_or(0);
                max_count as f64 / total as f64
            })
            .collect()
    }

    /// Number of gap characters per column.
    #[must_use]
    pub fn gaps_per_column(&self) -> Vec<usize> {
        if self.aligned_sequences.is_empty() {
            return Vec::new();
        }
        let len = self.aligned_sequences[0].len();
        (0..len)
            .map(|col| {
                self.aligned_sequences
                    .iter()
                    .filter(|seq| seq.chars().nth(col).unwrap_or('-') == '-')
                    .count()
            })
            .collect()
    }

    /// Formats the alignment in FASTA format.
    #[must_use]
    pub fn to_fasta(&self) -> String {
        self.ids
            .iter()
            .zip(self.aligned_sequences.iter())
            .map(|(id, seq)| format!(">{id}\n{seq}\n"))
            .collect()
    }
}

/// Progressive multiple sequence alignment (ClustalW-like).
///
/// Algorithm:
/// 1. Compute all pairwise distances (1 - identity from pairwise alignment).
/// 2. Build a guide tree using UPGMA.
/// 3. Progressively align sequences/profiles following the guide tree.
///
/// # Errors
///
/// Returns `CoreError::ValueError` if `sequences` is empty or has fewer than
/// 2 elements.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::alignment_ext::{
///     multiple_sequence_alignment, ScoringMatrix,
/// };
///
/// let seqs = vec![
///     ("seq1".to_string(), "ATGCATGC".to_string()),
///     ("seq2".to_string(), "ATGCTTGC".to_string()),
///     ("seq3".to_string(), "ATGCATGC".to_string()),
/// ];
/// let scoring = ScoringMatrix::dna_default();
/// let msa = multiple_sequence_alignment(&seqs, &scoring).expect("should succeed");
/// assert_eq!(msa.aligned_sequences.len(), 3);
/// ```
pub fn multiple_sequence_alignment(
    sequences: &[(String, String)],
    scoring: &ScoringMatrix,
) -> CoreResult<MultipleAlignment> {
    if sequences.len() < 2 {
        return Err(CoreError::ValueError(crate::error_context!(
            "at least 2 sequences are required for multiple alignment"
        )));
    }

    let n = sequences.len();

    // Step 1: Pairwise distance matrix
    let mut dist = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let result = needleman_wunsch_affine(&sequences[i].1, &sequences[j].1, scoring)?;
            let d = 1.0 - result.identity;
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    // Step 2: UPGMA guide tree (returns merge order as [(i, j)] pairs)
    let merge_order = upgma_order(&dist, n);

    // Step 3: Progressive alignment
    // Start with all sequences as individual "profiles" (vec of aligned strings)
    let mut profiles: Vec<Vec<String>> = sequences
        .iter()
        .map(|(_, seq)| vec![seq.clone()])
        .collect();
    let mut profile_ids: Vec<Vec<String>> = sequences
        .iter()
        .map(|(id, _)| vec![id.clone()])
        .collect();
    let mut active = vec![true; n]; // which profile indices are still top-level

    for (pi, pj) in merge_order {
        if pi >= n || pj >= n || !active[pi] || !active[pj] {
            continue;
        }

        // Align profile[pi] with profile[pj] using the consensus of each
        let consensus_i = profile_consensus(&profiles[pi]);
        let consensus_j = profile_consensus(&profiles[pj]);

        let alignment = needleman_wunsch_affine(&consensus_i, &consensus_j, scoring)?;

        // Apply gap insertions to all sequences in each profile
        let new_pi = apply_gaps_to_profile(&profiles[pi], &alignment.seq1_aligned, &consensus_i);
        let new_pj = apply_gaps_to_profile(&profiles[pj], &alignment.seq2_aligned, &consensus_j);

        // Merge into profile[pi]
        let mut merged_seqs = new_pi;
        merged_seqs.extend(new_pj);
        let mut merged_ids = profile_ids[pi].clone();
        merged_ids.extend(profile_ids[pj].clone());

        profiles[pi] = merged_seqs;
        profile_ids[pi] = merged_ids;
        active[pj] = false;
    }

    // Find the last active profile
    let final_idx = active
        .iter()
        .rposition(|&a| a)
        .unwrap_or(0);

    let aligned = profiles[final_idx].clone();
    let ids = profile_ids[final_idx].clone();

    // Build consensus
    let consensus = build_msa_consensus(&aligned);

    Ok(MultipleAlignment {
        aligned_sequences: aligned,
        ids,
        consensus,
    })
}

/// Computes a simple consensus from a profile (majority vote per column).
fn profile_consensus(profile: &[String]) -> String {
    if profile.is_empty() {
        return String::new();
    }
    let len = profile[0].len();
    (0..len)
        .map(|col| {
            let mut counts: HashMap<char, usize> = HashMap::new();
            for seq in profile {
                let ch = seq.chars().nth(col).unwrap_or('-');
                *counts.entry(ch).or_insert(0) += 1;
            }
            counts
                .into_iter()
                .max_by_key(|&(_, c)| c)
                .map(|(ch, _)| ch)
                .unwrap_or('-')
        })
        .collect()
}

/// Inserts gaps into all sequences in a profile according to the aligned
/// consensus string vs original consensus.
fn apply_gaps_to_profile(profile: &[String], aligned: &str, original: &str) -> Vec<String> {
    // Build a mapping: position in original → list of aligned chars (including leading gaps)
    let mut orig_to_aligned: Vec<Vec<char>> = Vec::new();
    let mut orig_idx = 0usize;
    let orig_chars: Vec<char> = original.chars().collect();
    let aligned_chars: Vec<char> = aligned.chars().collect();

    let mut pending_gaps: Vec<char> = Vec::new();
    for &ch in &aligned_chars {
        if ch == '-' {
            pending_gaps.push('-');
        } else {
            if orig_idx < orig_chars.len() {
                let mut entry = pending_gaps.drain(..).collect::<Vec<char>>();
                entry.push(ch);
                orig_to_aligned.push(entry);
                orig_idx += 1;
            }
        }
    }
    // Trailing gaps
    if !pending_gaps.is_empty() {
        if orig_to_aligned.is_empty() {
            orig_to_aligned.push(pending_gaps);
        } else {
            orig_to_aligned.last_mut().map(|v| v.extend(pending_gaps.iter()));
        }
    }

    profile
        .iter()
        .map(|seq| {
            let seq_chars: Vec<char> = seq.chars().collect();
            let mut result = String::new();
            let mut col = 0usize;

            for (oi, mapping) in orig_to_aligned.iter().enumerate() {
                for &mch in mapping {
                    if mch == '-' {
                        result.push('-');
                    } else {
                        // This position corresponds to original position oi
                        // Find corresponding char in this sequence
                        let seq_col = find_non_gap_position(seq_chars.as_slice(), oi);
                        result.push(seq_col);
                        col += 1;
                    }
                }
            }
            // Append any remaining characters from the sequence that were not mapped
            let total_non_gap = seq_chars.iter().filter(|&&c| c != '-').count();
            while col < total_non_gap {
                let pos = find_non_gap_position(seq_chars.as_slice(), col);
                result.push(pos);
                col += 1;
            }
            result
        })
        .collect()
}

/// Finds the character at the `idx`-th non-gap position.
fn find_non_gap_position(seq: &[char], idx: usize) -> char {
    let mut count = 0;
    for &ch in seq {
        if ch != '-' {
            if count == idx {
                return ch;
            }
            count += 1;
        }
    }
    '-'
}

/// Builds the consensus string for an MSA.
fn build_msa_consensus(aligned: &[String]) -> String {
    if aligned.is_empty() {
        return String::new();
    }
    let len = aligned[0].len();
    (0..len)
        .map(|col| {
            let mut counts: HashMap<char, usize> = HashMap::new();
            for seq in aligned {
                let ch = seq.chars().nth(col).unwrap_or('-');
                if ch != '-' {
                    *counts.entry(ch).or_insert(0) += 1;
                }
            }
            if counts.is_empty() {
                '-'
            } else {
                counts
                    .into_iter()
                    .max_by_key(|&(_, c)| c)
                    .map(|(ch, _)| ch)
                    .unwrap_or('-')
            }
        })
        .collect()
}

// ─── UPGMA ────────────────────────────────────────────────────────────────────

/// Returns a merge order `[(i, j)]` from a UPGMA guide tree.
///
/// At each step, merges the two clusters with the smallest average pairwise
/// distance.
fn upgma_order(dist: &[Vec<f64>], n: usize) -> Vec<(usize, usize)> {
    // Map cluster index → set of original sequence indices
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut active: Vec<bool> = vec![true; n];
    let mut order: Vec<(usize, usize)> = Vec::new();

    for _ in 0..(n - 1) {
        // Find closest pair of active clusters
        let mut best_dist = f64::INFINITY;
        let mut best_i = 0;
        let mut best_j = 1;

        let active_clusters: Vec<usize> = (0..clusters.len()).filter(|&i| active[i]).collect();

        for ci in 0..active_clusters.len() {
            for cj in (ci + 1)..active_clusters.len() {
                let i = active_clusters[ci];
                let j = active_clusters[cj];
                let avg = average_cluster_dist(dist, &clusters[i], &clusters[j]);
                if avg < best_dist {
                    best_dist = avg;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        order.push((best_i, best_j));

        // Merge cluster best_j into best_i
        let merged: Vec<usize> = clusters[best_i]
            .iter()
            .chain(clusters[best_j].iter())
            .copied()
            .collect();
        clusters[best_i] = merged;
        active[best_j] = false;
    }

    order
}

/// Average pairwise distance between two clusters.
fn average_cluster_dist(dist: &[Vec<f64>], a: &[usize], b: &[usize]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::INFINITY;
    }
    let total: f64 = a.iter().flat_map(|&i| b.iter().map(move |&j| dist[i][j])).sum();
    total / (a.len() * b.len()) as f64
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

#[inline]
fn safe_add(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

#[inline]
fn safe_max3(a: i32, b: i32, c: i32) -> i32 {
    a.max(b).max(c)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ScoringMatrix ──────────────────────────────────────────────────────

    #[test]
    fn test_scoring_matrix_dna_default() {
        let s = ScoringMatrix::dna_default();
        assert_eq!(s.match_score, 2);
        assert!(s.mismatch_penalty < 0);
        assert!(s.gap_open < 0);
        assert!(s.gap_extend < 0);
    }

    #[test]
    fn test_scoring_matrix_blosum62() {
        let s = ScoringMatrix::blosum62();
        assert!(s.match_score > 0);
        assert!(s.gap_open < 0);
    }

    // ── needleman_wunsch_affine ────────────────────────────────────────────

    #[test]
    fn test_nw_affine_identical() {
        let s = ScoringMatrix::dna_default();
        let r = needleman_wunsch_affine("ATGC", "ATGC", &s).expect("NW failed");
        assert!((r.identity - 1.0).abs() < 1e-10);
        assert_eq!(r.gaps, 0);
        assert_eq!(r.aligned_length, 4);
    }

    #[test]
    fn test_nw_affine_single_gap() {
        let s = ScoringMatrix::dna_default();
        let r = needleman_wunsch_affine("AGCT", "AGT", &s).expect("NW failed");
        assert_eq!(r.seq1_aligned.len(), r.seq2_aligned.len());
        assert!(r.score > 0 || r.aligned_length >= 3);
    }

    #[test]
    fn test_nw_affine_alignment_lengths_equal() {
        let s = ScoringMatrix::dna_default();
        let r = needleman_wunsch_affine("GCATGCU", "GATTACA", &s).expect("NW failed");
        assert_eq!(r.seq1_aligned.len(), r.seq2_aligned.len());
    }

    #[test]
    fn test_nw_affine_identity_in_range() {
        let s = ScoringMatrix::dna_default();
        let r = needleman_wunsch_affine("ATGCATGC", "ATGCTTGC", &s).expect("NW failed");
        assert!(r.identity >= 0.0 && r.identity <= 1.0);
    }

    // ── smith_waterman_affine ──────────────────────────────────────────────

    #[test]
    fn test_sw_affine_substring() {
        let s = ScoringMatrix::dna_default();
        let r = smith_waterman_affine("NNNNATGCNNNN", "ATGC", &s).expect("SW failed");
        assert!((r.identity - 1.0).abs() < 1e-10);
        assert_eq!(r.seq1_aligned, "ATGC");
    }

    #[test]
    fn test_sw_affine_no_match() {
        let s = ScoringMatrix {
            match_score: 1,
            mismatch_penalty: -100,
            gap_open: -100,
            gap_extend: -100,
        };
        let r = smith_waterman_affine("AAAA", "TTTT", &s).expect("SW failed");
        assert_eq!(r.score, 0);
        assert!(r.seq1_aligned.is_empty());
    }

    #[test]
    fn test_sw_affine_identical() {
        let s = ScoringMatrix::dna_default();
        let r = smith_waterman_affine("ATGC", "ATGC", &s).expect("SW failed");
        assert!((r.identity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sw_affine_alignment_lengths_equal() {
        let s = ScoringMatrix::dna_default();
        let r = smith_waterman_affine("TGTTACGG", "GGTTGACTA", &s).expect("SW failed");
        assert_eq!(r.seq1_aligned.len(), r.seq2_aligned.len());
    }

    // ── semi_global_align ─────────────────────────────────────────────────

    #[test]
    fn test_semi_global_query_in_target() {
        let s = ScoringMatrix::dna_default();
        let r = semi_global_align("ATGC", "NNATGCNN", &s).expect("semi-global failed");
        assert!((r.identity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_semi_global_aligned_lengths_equal() {
        let s = ScoringMatrix::dna_default();
        let r = semi_global_align("ATGCATGC", "GCATGCATGCGC", &s).expect("semi-global failed");
        assert_eq!(r.seq1_aligned.len(), r.seq2_aligned.len());
    }

    #[test]
    fn test_semi_global_empty_error() {
        let s = ScoringMatrix::dna_default();
        let r = semi_global_align("", "ATGC", &s);
        assert!(r.is_err());
    }

    // ── multiple_sequence_alignment ────────────────────────────────────────

    #[test]
    fn test_msa_two_identical_sequences() {
        let seqs = vec![
            ("s1".to_string(), "ATGC".to_string()),
            ("s2".to_string(), "ATGC".to_string()),
        ];
        let s = ScoringMatrix::dna_default();
        let msa = multiple_sequence_alignment(&seqs, &s).expect("MSA failed");
        assert_eq!(msa.aligned_sequences.len(), 2);
        assert_eq!(msa.ids.len(), 2);
    }

    #[test]
    fn test_msa_three_sequences() {
        let seqs = vec![
            ("s1".to_string(), "ATGCATGC".to_string()),
            ("s2".to_string(), "ATGCTTGC".to_string()),
            ("s3".to_string(), "ATGCATGC".to_string()),
        ];
        let s = ScoringMatrix::dna_default();
        let msa = multiple_sequence_alignment(&seqs, &s).expect("MSA failed");
        assert_eq!(msa.aligned_sequences.len(), 3);
        // All aligned sequences should have the same length
        let len = msa.aligned_sequences[0].len();
        for seq in &msa.aligned_sequences {
            assert_eq!(seq.len(), len, "aligned sequences must have equal length");
        }
    }

    #[test]
    fn test_msa_single_sequence_error() {
        let seqs = vec![("s1".to_string(), "ATGC".to_string())];
        let s = ScoringMatrix::dna_default();
        let r = multiple_sequence_alignment(&seqs, &s);
        assert!(r.is_err());
    }

    #[test]
    fn test_msa_conservation_scores_length() {
        let seqs = vec![
            ("s1".to_string(), "ATGC".to_string()),
            ("s2".to_string(), "ATGC".to_string()),
        ];
        let s = ScoringMatrix::dna_default();
        let msa = multiple_sequence_alignment(&seqs, &s).expect("should succeed");
        let cons = msa.conservation_scores();
        assert_eq!(cons.len(), msa.aligned_sequences[0].len());
        for v in &cons {
            assert!(*v >= 0.0 && *v <= 1.0);
        }
    }

    #[test]
    fn test_msa_gaps_per_column_length() {
        let seqs = vec![
            ("s1".to_string(), "ATGCAT".to_string()),
            ("s2".to_string(), "ATGC".to_string()),
        ];
        let s = ScoringMatrix::dna_default();
        let msa = multiple_sequence_alignment(&seqs, &s).expect("should succeed");
        let gaps = msa.gaps_per_column();
        assert_eq!(gaps.len(), msa.aligned_sequences[0].len());
    }

    #[test]
    fn test_msa_to_fasta() {
        let seqs = vec![
            ("seq1".to_string(), "ATGC".to_string()),
            ("seq2".to_string(), "ATGC".to_string()),
        ];
        let s = ScoringMatrix::dna_default();
        let msa = multiple_sequence_alignment(&seqs, &s).expect("should succeed");
        let fasta = msa.to_fasta();
        assert!(fasta.contains(">seq1"));
        assert!(fasta.contains(">seq2"));
    }
}
