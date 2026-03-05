//! Sequence motif finding and scoring.
//!
//! Provides position weight matrices (PWMs), motif scanning, EM-based de novo
//! motif discovery (MEME-like), Gibbs sampling, and over-represented k-mer
//! analysis.

use std::collections::HashMap;

use crate::error::{CoreError, CoreResult};

// ─── Position Weight Matrix ───────────────────────────────────────────────────

/// Position Weight Matrix (PWM) for motif representation.
///
/// Each row corresponds to a position in the motif; each column to a character
/// in the alphabet.  Values are log-odds scores (log2(freq / background)).
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::motif::PositionWeightMatrix;
///
/// let seqs = ["ACGT", "ACGT", "ACGG"];
/// let alphabet = ['A', 'C', 'G', 'T'];
/// let pwm = PositionWeightMatrix::from_sequences(&seqs, &alphabet, 0.1).expect("should succeed");
/// assert_eq!(pwm.length, 4);
/// let score = pwm.score("ACGT").expect("should succeed");
/// assert!(score.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct PositionWeightMatrix {
    /// `rows[position][alphabet_index]` – log-odds score.
    pub rows: Vec<Vec<f64>>,
    /// Ordered list of characters making up the alphabet.
    pub alphabet: Vec<char>,
    /// Length of the motif (number of positions).
    pub length: usize,
}

impl PositionWeightMatrix {
    /// Builds a PWM from a set of aligned sequences.
    ///
    /// Counts character occurrences at each position, adds a `pseudocount` to
    /// avoid log(0), normalises to frequencies, then converts to log-odds
    /// relative to a uniform background.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ValueError` if:
    /// - `sequences` is empty.
    /// - `alphabet` is empty.
    /// - The sequences have different lengths.
    pub fn from_sequences(
        sequences: &[&str],
        alphabet: &[char],
        pseudocount: f64,
    ) -> CoreResult<Self> {
        if sequences.is_empty() {
            return Err(CoreError::ValueError(crate::error_context!(
                "sequences must not be empty"
            )));
        }
        if alphabet.is_empty() {
            return Err(CoreError::ValueError(crate::error_context!(
                "alphabet must not be empty"
            )));
        }

        let length = sequences[0].chars().count();
        for (i, seq) in sequences.iter().enumerate() {
            if seq.chars().count() != length {
                return Err(CoreError::ValueError(crate::error_context!(format!(
                    "sequence {i} has length {} but expected {length}",
                    seq.chars().count()
                ))));
            }
        }

        let a = alphabet.len();
        let alpha_idx: HashMap<char, usize> =
            alphabet.iter().enumerate().map(|(i, &c)| (c, i)).collect();

        // Count matrix: counts[position][alpha_index]
        let mut counts = vec![vec![0usize; a]; length];
        for seq in sequences {
            for (pos, ch) in seq.chars().enumerate() {
                let uch = ch.to_ascii_uppercase();
                if let Some(&idx) = alpha_idx.get(&uch) {
                    counts[pos][idx] += 1;
                }
                // unknown characters are skipped
            }
        }

        let n = sequences.len() as f64;
        let bg = 1.0 / a as f64; // uniform background
        let mut rows = Vec::with_capacity(length);

        for pos_counts in &counts {
            let total = n + pseudocount * a as f64;
            let mut row = Vec::with_capacity(a);
            for &c in pos_counts {
                let freq = (c as f64 + pseudocount) / total;
                row.push((freq / bg).log2());
            }
            rows.push(row);
        }

        Ok(Self {
            rows,
            alphabet: alphabet.to_vec(),
            length,
        })
    }

    /// Constructs a PWM directly from a pre-computed frequency matrix.
    ///
    /// `freq[position][alphabet_index]` should contain non-negative frequencies.
    /// Values are converted to log-odds relative to a uniform background.
    #[must_use]
    pub fn from_frequency_matrix(freq: Vec<Vec<f64>>, alphabet: Vec<char>) -> Self {
        let length = freq.len();
        let a = alphabet.len();
        let bg = if a > 0 { 1.0 / a as f64 } else { 1.0 };

        let rows = freq
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|f| {
                        let safe_f = f.max(1e-300);
                        (safe_f / bg).log2()
                    })
                    .collect()
            })
            .collect();

        Self {
            rows,
            alphabet,
            length,
        }
    }

    /// Scores a subsequence against this PWM.
    ///
    /// Returns the sum of log-odds scores at each position.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ValueError` if `subsequence` has a different length
    /// from the PWM or contains characters outside the alphabet.
    pub fn score(&self, subsequence: &str) -> CoreResult<f64> {
        let chars: Vec<char> = subsequence.chars().collect();
        if chars.len() != self.length {
            return Err(CoreError::ValueError(crate::error_context!(format!(
                "subsequence length {} does not match PWM length {}",
                chars.len(),
                self.length
            ))));
        }

        let alpha_idx: HashMap<char, usize> = self
            .alphabet
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        let mut total = 0.0f64;
        for (pos, ch) in chars.iter().enumerate() {
            let uch = ch.to_ascii_uppercase();
            let idx = alpha_idx.get(&uch).ok_or_else(|| {
                CoreError::ValueError(crate::error_context!(format!(
                    "character '{}' not in alphabet",
                    uch
                )))
            })?;
            total += self.rows[pos][*idx];
        }
        Ok(total)
    }

    /// Returns the maximum possible score achievable by this PWM.
    #[must_use]
    pub fn max_score(&self) -> f64 {
        self.rows
            .iter()
            .map(|row| row.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            .sum()
    }

    /// Returns the minimum possible score achievable by this PWM.
    #[must_use]
    pub fn min_score(&self) -> f64 {
        self.rows
            .iter()
            .map(|row| row.iter().cloned().fold(f64::INFINITY, f64::min))
            .sum()
    }

    /// Returns the consensus string: the highest-scoring character at each position.
    #[must_use]
    pub fn consensus(&self) -> String {
        self.rows
            .iter()
            .map(|row| {
                let best_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                *self.alphabet.get(best_idx).unwrap_or(&'?')
            })
            .collect()
    }

    /// Information content (bits) at each position.
    ///
    /// IC = log2(|alphabet|) + Σ freq * log2(freq).
    #[must_use]
    pub fn ic_per_position(&self) -> Vec<f64> {
        let a = self.alphabet.len() as f64;
        let max_ic = a.log2();

        self.rows
            .iter()
            .map(|row| {
                // Convert log-odds back to frequency: freq = 2^(log_odds) * bg
                let bg = 1.0 / a;
                let freqs: Vec<f64> = row.iter().map(|&lo| (2.0_f64.powf(lo)) * bg).collect();
                let sum: f64 = freqs.iter().sum();
                // Normalise in case of rounding drift
                let ic: f64 = freqs
                    .iter()
                    .map(|&f| {
                        let nf = f / sum;
                        if nf > 1e-300 {
                            nf * nf.log2()
                        } else {
                            0.0
                        }
                    })
                    .sum();
                max_ic + ic // Shannon uncertainty
            })
            .collect()
    }

    /// Total information content (sum over all positions).
    #[must_use]
    pub fn total_ic(&self) -> f64 {
        self.ic_per_position().iter().sum()
    }
}

// ─── Motif scanning ───────────────────────────────────────────────────────────

/// Scans `sequence` for positions where the PWM score exceeds `threshold`.
///
/// Returns a vector of `(position, score)` pairs sorted by position.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::motif::{PositionWeightMatrix, scan_for_motif};
///
/// let seqs = ["ACGT", "ACGT"];
/// let alphabet = ['A', 'C', 'G', 'T'];
/// let pwm = PositionWeightMatrix::from_sequences(&seqs, &alphabet, 0.1).expect("should succeed");
/// let hits = scan_for_motif("NNNACGTNNN", &pwm, 0.0);
/// assert!(!hits.is_empty());
/// ```
#[must_use]
pub fn scan_for_motif(
    sequence: &str,
    pwm: &PositionWeightMatrix,
    threshold: f64,
) -> Vec<(usize, f64)> {
    let chars: Vec<char> = sequence.chars().collect();
    let len = chars.len();
    let motif_len = pwm.length;

    if motif_len == 0 || motif_len > len {
        return Vec::new();
    }

    let alpha_idx: HashMap<char, usize> = pwm
        .alphabet
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    let mut results = Vec::new();

    for start in 0..=(len - motif_len) {
        let window: String = chars[start..start + motif_len].iter().collect();
        // Score the window; skip positions with unknown characters
        let mut ok = true;
        let mut score = 0.0f64;
        for (pos, ch) in window.chars().enumerate() {
            let uch = ch.to_ascii_uppercase();
            match alpha_idx.get(&uch) {
                Some(&idx) => score += pwm.rows[pos][idx],
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok && score >= threshold {
            results.push((start, score));
        }
    }

    results
}

// ─── EM motif discovery (MEME-like) ──────────────────────────────────────────

/// MEME-like de novo motif discovery using expectation maximisation.
///
/// Initialises `n_motifs` random PWMs from subsequences, then iterates
/// E-step (score every window) / M-step (update counts) for `n_iterations`
/// rounds.  Returns the `n_motifs` final PWMs.
///
/// # Errors
///
/// Returns `CoreError::ValueError` if:
/// - `sequences` is empty.
/// - `motif_length` is 0 or greater than the shortest sequence.
/// - `n_motifs` is 0.
pub fn discover_motifs_em(
    sequences: &[&str],
    motif_length: usize,
    n_motifs: usize,
    n_iterations: usize,
    seed: u64,
) -> CoreResult<Vec<PositionWeightMatrix>> {
    if sequences.is_empty() {
        return Err(CoreError::ValueError(crate::error_context!(
            "sequences must not be empty"
        )));
    }
    if motif_length == 0 {
        return Err(CoreError::ValueError(crate::error_context!(
            "motif_length must be at least 1"
        )));
    }
    if n_motifs == 0 {
        return Err(CoreError::ValueError(crate::error_context!(
            "n_motifs must be at least 1"
        )));
    }

    let min_len = sequences.iter().map(|s| s.len()).min().unwrap_or(0);
    if motif_length > min_len {
        return Err(CoreError::ValueError(crate::error_context!(format!(
            "motif_length {motif_length} exceeds shortest sequence length {min_len}"
        ))));
    }

    let alphabet = ['A', 'C', 'G', 'T'];
    let a = alphabet.len();
    let alpha_idx: HashMap<char, usize> =
        alphabet.iter().enumerate().map(|(i, &c)| (c, i)).collect();

    // Simple LCG RNG (no external crate required)
    let mut rng_state = seed.wrapping_add(1);
    let lcg_next = |state: &mut u64| -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    };

    let mut motifs = Vec::with_capacity(n_motifs);

    // Collect all windows across all sequences as character vectors
    let all_windows: Vec<(usize, usize, Vec<char>)> = sequences
        .iter()
        .enumerate()
        .flat_map(|(si, seq)| {
            let chars: Vec<char> = seq.chars().collect();
            let n = chars.len();
            (0..=n.saturating_sub(motif_length))
                .map(|start| {
                    (
                        si,
                        start,
                        chars[start..start + motif_length].to_vec(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();

    if all_windows.is_empty() {
        return Err(CoreError::ValueError(crate::error_context!(
            "no valid windows found in sequences"
        )));
    }

    for motif_idx in 0..n_motifs {
        let _ = motif_idx;

        // Initialise PWM: pick a random window as seed
        let init_pos = (lcg_next(&mut rng_state) as usize) % all_windows.len();
        let seed_window = &all_windows[init_pos].2;

        // Counts[position][alpha_index]
        let mut counts = vec![vec![1.0f64; a]; motif_length]; // pseudocount=1
        for (pos, &ch) in seed_window.iter().enumerate() {
            let uch = ch.to_ascii_uppercase();
            if let Some(&idx) = alpha_idx.get(&uch) {
                counts[pos][idx] += 2.0; // extra weight for seed
            }
        }

        for _iter in 0..n_iterations {
            // Normalise counts → frequencies
            let freqs: Vec<Vec<f64>> = counts
                .iter()
                .map(|row| {
                    let total: f64 = row.iter().sum();
                    row.iter().map(|&c| c / total).collect()
                })
                .collect();

            // E-step: score each window, accumulate weighted counts
            let mut new_counts = vec![vec![1.0f64; a]; motif_length]; // pseudocount
            let bg = 1.0 / a as f64;

            let mut weights: Vec<f64> = Vec::with_capacity(all_windows.len());
            for (_, _, window) in &all_windows {
                let mut log_score = 0.0f64;
                let mut valid = true;
                for (pos, &ch) in window.iter().enumerate() {
                    let uch = ch.to_ascii_uppercase();
                    match alpha_idx.get(&uch) {
                        Some(&idx) => {
                            let f = freqs[pos][idx].max(1e-300);
                            log_score += (f / bg).ln();
                        }
                        None => {
                            valid = false;
                            break;
                        }
                    }
                }
                weights.push(if valid { log_score.exp() } else { 0.0 });
            }

            // Normalise weights to probabilities
            let weight_sum: f64 = weights.iter().sum();
            if weight_sum > 1e-300 {
                for w in &mut weights {
                    *w /= weight_sum;
                }
            }

            // M-step: weighted count accumulation
            for (wi, (_, _, window)) in all_windows.iter().enumerate() {
                let wt = weights[wi];
                if wt <= 1e-300 {
                    continue;
                }
                for (pos, &ch) in window.iter().enumerate() {
                    let uch = ch.to_ascii_uppercase();
                    if let Some(&idx) = alpha_idx.get(&uch) {
                        new_counts[pos][idx] += wt;
                    }
                }
            }

            counts = new_counts;
        }

        // Build final PWM from converged counts
        let freqs: Vec<Vec<f64>> = counts
            .iter()
            .map(|row| {
                let total: f64 = row.iter().sum();
                row.iter().map(|&c| c / total).collect()
            })
            .collect();

        motifs.push(PositionWeightMatrix::from_frequency_matrix(
            freqs,
            alphabet.to_vec(),
        ));
    }

    Ok(motifs)
}

// ─── Gibbs sampling motif finder ─────────────────────────────────────────────

/// Gibbs sampling motif finder.
///
/// Iteratively:
/// 1. Excludes one sequence.
/// 2. Builds a PWM from the current motif positions in all other sequences.
/// 3. Scores all windows in the excluded sequence and samples a new position
///    proportional to the scores.
///
/// Returns the final PWM after `n_iterations`.
///
/// # Errors
///
/// Returns `CoreError::ValueError` if inputs are invalid (empty sequences,
/// motif longer than shortest sequence, etc.).
pub fn gibbs_motif_sampler(
    sequences: &[&str],
    motif_length: usize,
    n_iterations: usize,
    seed: u64,
) -> CoreResult<PositionWeightMatrix> {
    if sequences.len() < 2 {
        return Err(CoreError::ValueError(crate::error_context!(
            "at least 2 sequences are required for Gibbs sampling"
        )));
    }
    if motif_length == 0 {
        return Err(CoreError::ValueError(crate::error_context!(
            "motif_length must be at least 1"
        )));
    }

    let min_len = sequences.iter().map(|s| s.len()).min().unwrap_or(0);
    if motif_length > min_len {
        return Err(CoreError::ValueError(crate::error_context!(format!(
            "motif_length {motif_length} exceeds shortest sequence length {min_len}"
        ))));
    }

    let ns = sequences.len();
    let alphabet = ['A', 'C', 'G', 'T'];
    let a = alphabet.len();
    let alpha_idx: HashMap<char, usize> =
        alphabet.iter().enumerate().map(|(i, &c)| (c, i)).collect();

    // LCG RNG
    let mut rng_state = seed.wrapping_add(7);
    let mut lcg = move || -> u64 {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        rng_state
    };

    // Convert sequences to char vecs
    let seqs_chars: Vec<Vec<char>> = sequences.iter().map(|s| s.chars().collect()).collect();

    // Initialise random starting positions for each sequence
    let mut positions: Vec<usize> = seqs_chars
        .iter()
        .map(|seq| {
            let max_pos = seq.len().saturating_sub(motif_length);
            if max_pos == 0 {
                0
            } else {
                (lcg() as usize) % (max_pos + 1)
            }
        })
        .collect();

    // Helper: build PWM from positions excluding sequence `exclude_idx`
    let build_pwm = |positions: &[usize], exclude: usize| -> Vec<Vec<f64>> {
        let mut counts = vec![vec![1.0f64; a]; motif_length]; // pseudocount
        for (si, (&pos, seq)) in positions.iter().zip(seqs_chars.iter()).enumerate() {
            if si == exclude {
                continue;
            }
            let end = (pos + motif_length).min(seq.len());
            let window = &seq[pos..end];
            if window.len() < motif_length {
                continue;
            }
            for (p, &ch) in window.iter().enumerate() {
                let uch = ch.to_ascii_uppercase();
                if let Some(&idx) = alpha_idx.get(&uch) {
                    counts[p][idx] += 1.0;
                }
            }
        }
        counts
            .iter()
            .map(|row| {
                let total: f64 = row.iter().sum();
                row.iter().map(|&c| c / total).collect()
            })
            .collect()
    };

    // Helper: score a window in a sequence given a frequency matrix
    let score_window = |window: &[char], freqs: &Vec<Vec<f64>>| -> f64 {
        if window.len() != motif_length {
            return 0.0;
        }
        let bg = 1.0 / a as f64;
        let mut log_score = 0.0f64;
        for (pos, &ch) in window.iter().enumerate() {
            let uch = ch.to_ascii_uppercase();
            match alpha_idx.get(&uch) {
                Some(&idx) => {
                    let f = freqs[pos][idx].max(1e-300);
                    log_score += (f / bg).ln();
                }
                None => return 0.0,
            }
        }
        log_score.exp().max(0.0)
    };

    for _iter in 0..n_iterations {
        let exclude = (lcg() as usize) % ns;
        let freqs = build_pwm(&positions, exclude);

        let seq = &seqs_chars[exclude];
        let n_windows = seq.len().saturating_sub(motif_length) + 1;

        // Score all windows in the excluded sequence
        let mut scores: Vec<f64> = (0..n_windows)
            .map(|start| {
                let window = &seq[start..(start + motif_length).min(seq.len())];
                score_window(window, &freqs)
            })
            .collect();

        // Sample new position proportional to scores
        let total: f64 = scores.iter().sum();
        if total > 1e-300 {
            for s in &mut scores {
                *s /= total;
            }
            let r = (lcg() as f64) / u64::MAX as f64;
            let mut cumsum = 0.0f64;
            let mut new_pos = 0;
            for (idx, &s) in scores.iter().enumerate() {
                cumsum += s;
                if r <= cumsum {
                    new_pos = idx;
                    break;
                }
            }
            positions[exclude] = new_pos;
        }
    }

    // Build final PWM from all positions
    let mut final_counts = vec![vec![1.0f64; a]; motif_length];
    for (si, (&pos, seq)) in positions.iter().zip(seqs_chars.iter()).enumerate() {
        let _ = si;
        let end = (pos + motif_length).min(seq.len());
        let window = &seq[pos..end];
        if window.len() < motif_length {
            continue;
        }
        for (p, &ch) in window.iter().enumerate() {
            let uch = ch.to_ascii_uppercase();
            if let Some(&idx) = alpha_idx.get(&uch) {
                final_counts[p][idx] += 1.0;
            }
        }
    }

    let freqs: Vec<Vec<f64>> = final_counts
        .iter()
        .map(|row| {
            let total: f64 = row.iter().sum();
            row.iter().map(|&c| c / total).collect()
        })
        .collect();

    Ok(PositionWeightMatrix::from_frequency_matrix(
        freqs,
        alphabet.to_vec(),
    ))
}

// ─── Over-represented k-mer finding ──────────────────────────────────────────

/// Finds over-represented k-mers in `foreground` sequences relative to
/// `background` sequences.
///
/// Computes `log2(fg_freq / bg_freq)` for each k-mer, using a pseudocount of
/// 1 to avoid division by zero.  Returns the top `top_n` k-mers sorted by
/// descending enrichment ratio.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::bioinformatics::motif::find_enriched_kmers;
///
/// let fg = ["ACGTACGT", "ACGTTTTT"];
/// let bg = ["TTTTTTTT", "GGGGGGGG"];
/// let results = find_enriched_kmers(&fg, &bg, 3, 5);
/// assert!(!results.is_empty());
/// ```
#[must_use]
pub fn find_enriched_kmers(
    foreground: &[&str],
    background: &[&str],
    k: usize,
    top_n: usize,
) -> Vec<(String, f64)> {
    if k == 0 || foreground.is_empty() || background.is_empty() {
        return Vec::new();
    }

    let count_kmers = |seqs: &[&str]| -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for seq in seqs {
            let chars: Vec<char> = seq.chars().collect();
            if chars.len() < k {
                continue;
            }
            for start in 0..=(chars.len() - k) {
                let kmer: String = chars[start..start + k]
                    .iter()
                    .map(|c| c.to_ascii_uppercase())
                    .collect();
                *counts.entry(kmer).or_insert(0) += 1;
            }
        }
        counts
    };

    let fg_counts = count_kmers(foreground);
    let bg_counts = count_kmers(background);

    let fg_total: usize = fg_counts.values().sum();
    let bg_total: usize = bg_counts.values().sum();

    let fg_total_f = (fg_total + 1) as f64;
    let bg_total_f = (bg_total + 1) as f64;

    // Collect all k-mers from foreground
    let mut enrichments: Vec<(String, f64)> = fg_counts
        .iter()
        .map(|(kmer, &fg_c)| {
            let bg_c = *bg_counts.get(kmer).unwrap_or(&0);
            let fg_freq = (fg_c + 1) as f64 / fg_total_f;
            let bg_freq = (bg_c + 1) as f64 / bg_total_f;
            let log_ratio = (fg_freq / bg_freq).log2();
            (kmer.clone(), log_ratio)
        })
        .collect();

    // Sort by descending enrichment
    enrichments.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    enrichments.truncate(top_n);
    enrichments
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PositionWeightMatrix ───────────────────────────────────────────────

    #[test]
    fn test_pwm_from_sequences_basic() {
        let seqs = ["ACGT", "ACGT", "ACGT"];
        let alphabet = ['A', 'C', 'G', 'T'];
        let pwm = PositionWeightMatrix::from_sequences(&seqs, &alphabet, 0.1)
            .expect("PWM construction failed");
        assert_eq!(pwm.length, 4);
        assert_eq!(pwm.alphabet, alphabet);
    }

    #[test]
    fn test_pwm_from_sequences_empty_error() {
        let seqs: &[&str] = &[];
        let alphabet = ['A', 'C', 'G', 'T'];
        let result = PositionWeightMatrix::from_sequences(seqs, &alphabet, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_pwm_from_sequences_length_mismatch_error() {
        let seqs = ["ACGT", "ACG"]; // different lengths
        let alphabet = ['A', 'C', 'G', 'T'];
        let result = PositionWeightMatrix::from_sequences(&seqs, &alphabet, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_pwm_score_correct_length() {
        let seqs = ["ACGT", "ACGT"];
        let alphabet = ['A', 'C', 'G', 'T'];
        let pwm = PositionWeightMatrix::from_sequences(&seqs, &alphabet, 0.1).expect("should succeed");
        let score = pwm.score("ACGT").expect("score failed");
        assert!(score.is_finite());
    }

    #[test]
    fn test_pwm_score_wrong_length_error() {
        let seqs = ["ACGT"];
        let alphabet = ['A', 'C', 'G', 'T'];
        let pwm = PositionWeightMatrix::from_sequences(&seqs, &alphabet, 0.1).expect("should succeed");
        let result = pwm.score("ACG"); // too short
        assert!(result.is_err());
    }

    #[test]
    fn test_pwm_score_unknown_char_error() {
        let seqs = ["ACGT"];
        let alphabet = ['A', 'C', 'G', 'T'];
        let pwm = PositionWeightMatrix::from_sequences(&seqs, &alphabet, 0.1).expect("should succeed");
        let result = pwm.score("XYZW"); // chars not in alphabet
        assert!(result.is_err());
    }

    #[test]
    fn test_pwm_max_score_geq_min_score() {
        let seqs = ["ACGT", "TGCA", "AATT"];
        let alphabet = ['A', 'C', 'G', 'T'];
        let pwm = PositionWeightMatrix::from_sequences(&seqs, &alphabet, 0.1).expect("should succeed");
        assert!(pwm.max_score() >= pwm.min_score());
    }

    #[test]
    fn test_pwm_consensus_length() {
        let seqs = ["ACGT", "ACGT"];
        let alphabet = ['A', 'C', 'G', 'T'];
        let pwm = PositionWeightMatrix::from_sequences(&seqs, &alphabet, 0.1).expect("should succeed");
        assert_eq!(pwm.consensus().len(), pwm.length);
    }

    #[test]
    fn test_pwm_ic_per_position_length() {
        let seqs = ["ACGT", "ACGT"];
        let alphabet = ['A', 'C', 'G', 'T'];
        let pwm = PositionWeightMatrix::from_sequences(&seqs, &alphabet, 0.1).expect("should succeed");
        let ic = pwm.ic_per_position();
        assert_eq!(ic.len(), pwm.length);
        for v in &ic {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_pwm_total_ic_non_negative() {
        let seqs = ["ACGT", "ACGT"];
        let alphabet = ['A', 'C', 'G', 'T'];
        let pwm = PositionWeightMatrix::from_sequences(&seqs, &alphabet, 0.1).expect("should succeed");
        assert!(pwm.total_ic() >= 0.0);
    }

    #[test]
    fn test_pwm_from_frequency_matrix() {
        // Uniform frequency matrix → zero log-odds
        let freq = vec![vec![0.25, 0.25, 0.25, 0.25]; 3];
        let alphabet = vec!['A', 'C', 'G', 'T'];
        let pwm = PositionWeightMatrix::from_frequency_matrix(freq, alphabet);
        assert_eq!(pwm.length, 3);
        for row in &pwm.rows {
            for &v in row {
                assert!((v).abs() < 1e-10, "uniform freq should give 0 log-odds");
            }
        }
    }

    // ── scan_for_motif ─────────────────────────────────────────────────────

    #[test]
    fn test_scan_finds_perfect_match() {
        let seqs = ["ACGT", "ACGT", "ACGT"];
        let alphabet = ['A', 'C', 'G', 'T'];
        let pwm = PositionWeightMatrix::from_sequences(&seqs, &alphabet, 0.01).expect("should succeed");
        let hits = scan_for_motif("NNNNACGTNNNN", &pwm, 0.0);
        assert!(
            hits.iter().any(|&(pos, _)| pos == 4),
            "expected hit at position 4"
        );
    }

    #[test]
    fn test_scan_empty_result_high_threshold() {
        let seqs = ["ACGT"];
        let alphabet = ['A', 'C', 'G', 'T'];
        let pwm = PositionWeightMatrix::from_sequences(&seqs, &alphabet, 0.1).expect("should succeed");
        let max = pwm.max_score();
        // threshold above max_score → no hits
        let hits = scan_for_motif("ACGTACGT", &pwm, max + 100.0);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_scan_short_sequence() {
        let seqs = ["ACGT"];
        let alphabet = ['A', 'C', 'G', 'T'];
        let pwm = PositionWeightMatrix::from_sequences(&seqs, &alphabet, 0.1).expect("should succeed");
        // Sequence shorter than motif
        let hits = scan_for_motif("AC", &pwm, 0.0);
        assert!(hits.is_empty());
    }

    // ── discover_motifs_em ─────────────────────────────────────────────────

    #[test]
    fn test_em_motif_discovery_returns_correct_count() {
        let seqs = [
            "ACGTACGTACGT",
            "TTACGTACGTTT",
            "GGACGTACGTGG",
            "CCACGTACGTCC",
        ];
        let result = discover_motifs_em(&seqs, 4, 2, 10, 42);
        assert!(result.is_ok());
        let motifs = result.expect("should succeed");
        assert_eq!(motifs.len(), 2);
        for m in &motifs {
            assert_eq!(m.length, 4);
        }
    }

    #[test]
    fn test_em_empty_sequences_error() {
        let seqs: &[&str] = &[];
        let result = discover_motifs_em(seqs, 4, 1, 10, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_em_motif_too_long_error() {
        let seqs = ["ACGT"]; // length 4
        let result = discover_motifs_em(&seqs, 10, 1, 5, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_em_zero_motifs_error() {
        let seqs = ["ACGTACGTACGT"];
        let result = discover_motifs_em(&seqs, 4, 0, 5, 42);
        assert!(result.is_err());
    }

    // ── gibbs_motif_sampler ────────────────────────────────────────────────

    #[test]
    fn test_gibbs_returns_pwm_correct_length() {
        let seqs = [
            "ACGTACGTACGT",
            "TTACGTACGTTT",
            "GGACGTACGTGG",
        ];
        let pwm = gibbs_motif_sampler(&seqs, 4, 50, 123).expect("Gibbs failed");
        assert_eq!(pwm.length, 4);
    }

    #[test]
    fn test_gibbs_single_sequence_error() {
        let seqs = ["ACGTACGTACGT"];
        let result = gibbs_motif_sampler(&seqs, 4, 10, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_gibbs_motif_too_long_error() {
        let seqs = ["ACGT", "ACGT"];
        let result = gibbs_motif_sampler(&seqs, 10, 10, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_gibbs_pwm_alphabet_correct() {
        let seqs = ["ACGTACGT", "TGCATGCA", "AACCGGTT"];
        let pwm = gibbs_motif_sampler(&seqs, 3, 20, 7).expect("Gibbs failed");
        assert_eq!(pwm.alphabet, ['A', 'C', 'G', 'T']);
    }

    // ── find_enriched_kmers ────────────────────────────────────────────────

    #[test]
    fn test_enriched_kmers_returns_top_n() {
        let fg = ["ACGTACGT", "ACGTACGT", "ACGTACGT"];
        let bg = ["TTTTTTTT", "GGGGGGGG", "CCCCCCCC"];
        let results = find_enriched_kmers(&fg, &bg, 3, 5);
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_enriched_kmers_sorted_descending() {
        let fg = ["ACGTACGT", "ACGTACGT"];
        let bg = ["TTTTTTTT", "GGGGGGGG"];
        let results = find_enriched_kmers(&fg, &bg, 2, 10);
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 >= results[i].1,
                "results should be sorted in descending order"
            );
        }
    }

    #[test]
    fn test_enriched_kmers_empty_foreground() {
        let fg: &[&str] = &[];
        let bg = ["TTTTTTTT"];
        let results = find_enriched_kmers(fg, &bg, 3, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_enriched_kmers_k_zero() {
        let fg = ["ACGT"];
        let bg = ["TTTT"];
        let results = find_enriched_kmers(&fg, &bg, 0, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_enriched_kmers_fg_enrichment() {
        // A 3-mer that only appears in fg should have positive log ratio
        let fg = ["ACGACGACG", "ACGACGACG"];
        let bg = ["TTTTTTTTTT", "GGGGGGGGGG"];
        let results = find_enriched_kmers(&fg, &bg, 3, 3);
        // The top kmer should have positive enrichment
        assert!(!results.is_empty());
        assert!(results[0].1 > 0.0, "expected positive enrichment");
    }
}
