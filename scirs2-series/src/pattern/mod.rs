//! Temporal pattern mining for time series
//!
//! Provides:
//! - `TimeSeriesMotif`: Matrix profile-based motif discovery
//! - `Discord`: Time series discord (unusual subsequence) detection
//! - `ShapePattern`: Shape-based pattern matching using DTW
//! - `SAX`: Symbolic Aggregate approXimation
//! - `MotifResult`: Motif pairs, indices, distances

use crate::error::{Result, TimeSeriesError};

// ============================================================================
// Result types
// ============================================================================

/// A discovered motif (repeating pattern) in a time series
#[derive(Debug, Clone)]
pub struct MotifPair {
    /// Index of the first occurrence
    pub index_a: usize,
    /// Index of the second (nearest neighbor) occurrence
    pub index_b: usize,
    /// Distance between the two occurrences
    pub distance: f64,
}

/// Result of motif discovery
#[derive(Debug, Clone)]
pub struct MotifResult {
    /// All discovered motif pairs (sorted by distance, best motif first)
    pub motif_pairs: Vec<MotifPair>,
    /// Number of motifs found
    pub n_motifs: usize,
    /// Subsequence length used
    pub subsequence_len: usize,
}

/// A discord: the most unusual subsequence in a time series
#[derive(Debug, Clone)]
pub struct DiscordResult {
    /// Index of the discord
    pub index: usize,
    /// Nearest neighbor distance (larger = more anomalous)
    pub nn_distance: f64,
    /// Length of the discord subsequence
    pub subsequence_len: usize,
}

// ============================================================================
// Matrix Profile helpers
// ============================================================================

/// Z-normalize a subsequence
fn z_normalize(sub: &[f64]) -> Vec<f64> {
    let n = sub.len() as f64;
    let mean = sub.iter().sum::<f64>() / n;
    let std = {
        let var = sub.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        var.sqrt().max(1e-10)
    };
    sub.iter().map(|&x| (x - mean) / std).collect()
}

/// Euclidean distance between two slices
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

// ============================================================================
// TimeSeriesMotif (Matrix Profile based)
// ============================================================================

/// Configuration for motif discovery
#[derive(Debug, Clone)]
pub struct MotifConfig {
    /// Subsequence length
    pub subsequence_len: usize,
    /// Exclusion zone as fraction of subsequence_len (typical: 0.5)
    pub exclusion_zone: f64,
    /// Maximum number of top motifs to return
    pub top_k: usize,
}

impl Default for MotifConfig {
    fn default() -> Self {
        Self {
            subsequence_len: 20,
            exclusion_zone: 0.5,
            top_k: 5,
        }
    }
}

/// TimeSeriesMotif: discovers repeating subsequence patterns via matrix profile
///
/// Uses a simplified O(n^2 m) STAMP-style computation:
/// for each subsequence, find the nearest non-self neighbor.
/// The top-k pairs with smallest distances are returned as motifs.
pub struct TimeSeriesMotif {
    config: MotifConfig,
}

impl TimeSeriesMotif {
    /// Create a new motif discovery instance
    pub fn new(config: MotifConfig) -> Self {
        Self { config }
    }

    /// Compute the matrix profile (nearest neighbor distances and indices)
    fn compute_matrix_profile(&self, data: &[f64]) -> (Vec<f64>, Vec<usize>) {
        let n = data.len();
        let m = self.config.subsequence_len;
        let n_subs = n - m + 1;
        let excl = (self.config.exclusion_zone * m as f64).ceil() as usize;

        let mut mp = vec![f64::INFINITY; n_subs];
        let mut mpi = vec![0usize; n_subs];

        // Precompute z-normalized subsequences
        let normalized: Vec<Vec<f64>> = (0..n_subs)
            .map(|i| z_normalize(&data[i..i + m]))
            .collect();

        for i in 0..n_subs {
            for j in 0..n_subs {
                if (i as isize - j as isize).unsigned_abs() <= excl {
                    continue;
                }
                let d = euclidean_distance(&normalized[i], &normalized[j]);
                if d < mp[i] {
                    mp[i] = d;
                    mpi[i] = j;
                }
            }
        }
        (mp, mpi)
    }

    /// Discover motifs in the time series
    pub fn find_motifs(&self, data: &[f64]) -> Result<MotifResult> {
        let n = data.len();
        let m = self.config.subsequence_len;

        if n < m * 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Series too short for motif discovery".to_string(),
                required: m * 2,
                actual: n,
            });
        }

        let (mp, mpi) = self.compute_matrix_profile(data);
        let n_subs = mp.len();
        let excl = (self.config.exclusion_zone * m as f64).ceil() as usize;

        // Collect motif candidates sorted by distance
        let mut candidates: Vec<(usize, f64)> = mp.iter().cloned().enumerate()
            .filter(|(_, d)| d.is_finite())
            .collect();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top-k non-overlapping motifs
        let mut motif_pairs: Vec<MotifPair> = Vec::new();
        let mut used: Vec<bool> = vec![false; n_subs];

        for (idx, dist) in candidates {
            if used[idx] {
                continue;
            }
            let nn = mpi[idx];
            if used[nn] {
                continue;
            }
            // Ensure exclusion zone is respected
            let too_close = motif_pairs.iter().any(|mp_pair| {
                (idx as isize - mp_pair.index_a as isize).unsigned_abs() <= excl
                    || (nn as isize - mp_pair.index_b as isize).unsigned_abs() <= excl
            });
            if too_close {
                continue;
            }

            motif_pairs.push(MotifPair { index_a: idx, index_b: nn, distance: dist });

            // Mark exclusion zone around this motif
            for j in idx.saturating_sub(excl)..=(idx + excl).min(n_subs - 1) {
                used[j] = true;
            }
            for j in nn.saturating_sub(excl)..=(nn + excl).min(n_subs - 1) {
                used[j] = true;
            }

            if motif_pairs.len() >= self.config.top_k {
                break;
            }
        }

        let n_motifs = motif_pairs.len();
        Ok(MotifResult { motif_pairs, n_motifs, subsequence_len: m })
    }
}

// ============================================================================
// Discord detection
// ============================================================================

/// Configuration for discord detection
#[derive(Debug, Clone)]
pub struct DiscordConfig {
    /// Subsequence length
    pub subsequence_len: usize,
    /// Exclusion zone as fraction of subsequence_len
    pub exclusion_zone: f64,
    /// Number of top discords to return
    pub top_k: usize,
}

impl Default for DiscordConfig {
    fn default() -> Self {
        Self {
            subsequence_len: 20,
            exclusion_zone: 0.5,
            top_k: 3,
        }
    }
}

/// Discord: detects the most unusual subsequences in a time series
///
/// A discord is a subsequence whose nearest non-self neighbor is farther
/// than any other subsequence's nearest non-self neighbor.
pub struct Discord {
    config: DiscordConfig,
}

impl Discord {
    /// Create a new discord detector
    pub fn new(config: DiscordConfig) -> Self {
        Self { config }
    }

    /// Find top-k discords in the time series
    pub fn find_discords(&self, data: &[f64]) -> Result<Vec<DiscordResult>> {
        let n = data.len();
        let m = self.config.subsequence_len;

        if n < m * 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Series too short for discord detection".to_string(),
                required: m * 2,
                actual: n,
            });
        }

        let motif_finder = TimeSeriesMotif::new(MotifConfig {
            subsequence_len: m,
            exclusion_zone: self.config.exclusion_zone,
            top_k: 1,
        });

        let (mp, _mpi) = motif_finder.compute_matrix_profile(data);
        let n_subs = mp.len();
        let excl = (self.config.exclusion_zone * m as f64).ceil() as usize;

        // Sort by descending distance (largest nearest-neighbor distance = most unusual)
        let mut candidates: Vec<(usize, f64)> = mp.iter().cloned().enumerate()
            .filter(|(_, d)| d.is_finite())
            .collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut discords: Vec<DiscordResult> = Vec::new();
        let mut used: Vec<bool> = vec![false; n_subs];

        for (idx, dist) in candidates {
            if used[idx] {
                continue;
            }
            discords.push(DiscordResult {
                index: idx,
                nn_distance: dist,
                subsequence_len: m,
            });

            // Exclude nearby subsequences
            for j in idx.saturating_sub(excl)..=(idx + excl).min(n_subs - 1) {
                used[j] = true;
            }

            if discords.len() >= self.config.top_k {
                break;
            }
        }

        Ok(discords)
    }
}

// ============================================================================
// ShapePattern: DTW-based pattern matching
// ============================================================================

/// Configuration for shape-based pattern matching
#[derive(Debug, Clone)]
pub struct ShapePatternConfig {
    /// Sakoe-Chiba band width as fraction of pattern length (0 = no constraint)
    pub warping_window: f64,
    /// Maximum distance for a match
    pub distance_threshold: Option<f64>,
}

impl Default for ShapePatternConfig {
    fn default() -> Self {
        Self {
            warping_window: 0.1,
            distance_threshold: None,
        }
    }
}

/// Match result from shape pattern search
#[derive(Debug, Clone)]
pub struct ShapeMatchResult {
    /// Starting index of the match in the time series
    pub start_index: usize,
    /// DTW distance of the match
    pub distance: f64,
    /// Length of the matched subsequence
    pub length: usize,
}

/// ShapePattern: DTW-based template matching in time series
pub struct ShapePattern {
    config: ShapePatternConfig,
}

impl ShapePattern {
    /// Create a new ShapePattern matcher
    pub fn new(config: ShapePatternConfig) -> Self {
        Self { config }
    }

    /// Compute DTW distance between two sequences with Sakoe-Chiba band
    pub fn dtw_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        let n = a.len();
        let m = b.len();
        let window = ((self.config.warping_window * n.max(m) as f64).ceil() as usize).max(1);

        let mut dtw = vec![vec![f64::INFINITY; m + 1]; n + 1];
        dtw[0][0] = 0.0;

        for i in 1..=n {
            let j_start = i.saturating_sub(window);
            let j_end = (i + window).min(m);
            for j in j_start..=j_end {
                if j == 0 {
                    continue;
                }
                let cost = (a[i - 1] - b[j - 1]).powi(2);
                let best_prev = dtw[i - 1][j].min(dtw[i][j - 1]).min(dtw[i - 1][j - 1]);
                dtw[i][j] = cost + best_prev;
            }
        }

        if dtw[n][m].is_infinite() {
            // Retry without window constraint
            let mut dtw2 = vec![vec![f64::INFINITY; m + 1]; n + 1];
            dtw2[0][0] = 0.0;
            for i in 1..=n {
                for j in 1..=m {
                    let cost = (a[i - 1] - b[j - 1]).powi(2);
                    let best_prev = dtw2[i - 1][j].min(dtw2[i][j - 1]).min(dtw2[i - 1][j - 1]);
                    dtw2[i][j] = cost + best_prev;
                }
            }
            dtw2[n][m].sqrt()
        } else {
            dtw[n][m].sqrt()
        }
    }

    /// Find all occurrences of the pattern in the time series within the distance threshold
    pub fn find_pattern(&self, data: &[f64], pattern: &[f64]) -> Result<Vec<ShapeMatchResult>> {
        let n = data.len();
        let m = pattern.len();

        if n < m {
            return Err(TimeSeriesError::InsufficientData {
                message: "Time series shorter than pattern".to_string(),
                required: m,
                actual: n,
            });
        }

        let norm_pattern = z_normalize(pattern);

        let mut matches: Vec<ShapeMatchResult> = Vec::new();
        for start in 0..=n - m {
            let sub = z_normalize(&data[start..start + m]);
            let dist = self.dtw_distance(&sub, &norm_pattern);

            let include = match self.config.distance_threshold {
                Some(thr) => dist <= thr,
                None => true,
            };
            if include {
                matches.push(ShapeMatchResult { start_index: start, distance: dist, length: m });
            }
        }

        matches.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        Ok(matches)
    }
}

// ============================================================================
// SAX: Symbolic Aggregate approXimation
// ============================================================================

/// Configuration for SAX
#[derive(Debug, Clone)]
pub struct SAXConfig {
    /// PAA (Piecewise Aggregate Approximation) word size
    pub word_size: usize,
    /// Alphabet size (e.g., 4 gives letters a-d)
    pub alphabet_size: usize,
    /// PAA reduction window (if None, derived from word_size and data length)
    pub window_size: Option<usize>,
}

impl Default for SAXConfig {
    fn default() -> Self {
        Self {
            word_size: 8,
            alphabet_size: 4,
            window_size: None,
        }
    }
}

/// SAX breakpoints for Normal(0,1) distribution
/// For alphabet size a, there are a-1 breakpoints
fn normal_breakpoints(alphabet_size: usize) -> Vec<f64> {
    // Precomputed quantiles for alphabet sizes 2..=10
    match alphabet_size {
        2 => vec![0.0],
        3 => vec![-0.4307, 0.4307],
        4 => vec![-0.6745, 0.0, 0.6745],
        5 => vec![-0.8416, -0.2533, 0.2533, 0.8416],
        6 => vec![-0.9674, -0.4307, 0.0, 0.4307, 0.9674],
        7 => vec![-1.0676, -0.5659, -0.1800, 0.1800, 0.5659, 1.0676],
        8 => vec![-1.1503, -0.6745, -0.3186, 0.0, 0.3186, 0.6745, 1.1503],
        9 => vec![-1.2206, -0.7648, -0.4307, -0.1397, 0.1397, 0.4307, 0.7648, 1.2206],
        10 => vec![-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816],
        _ => {
            // Approximate using inverse normal CDF
            let n = alphabet_size - 1;
            (1..=n).map(|i| {
                let p = i as f64 / alphabet_size as f64;
                // Beasley-Springer-Moro approximation
                let p_adj = p.clamp(1e-10, 1.0 - 1e-10);
                let t = (-2.0 * (p_adj.min(1.0 - p_adj)).ln()).sqrt();
                let c = [2.515517, 0.802853, 0.010328];
                let d = [1.432788, 0.189269, 0.001308];
                let num = c[0] + c[1] * t + c[2] * t * t;
                let den = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t;
                let z = t - num / den;
                if p >= 0.5 { z } else { -z }
            }).collect()
        }
    }
}

/// SAX representation of a time series
#[derive(Debug, Clone)]
pub struct SAXRepresentation {
    /// SAX string (each character is a letter from 'a' to 'a'+alphabet_size)
    pub word: String,
    /// Alphabet size used
    pub alphabet_size: usize,
    /// Word size (number of letters)
    pub word_size: usize,
    /// PAA values (before symbol assignment)
    pub paa_values: Vec<f64>,
}

/// SAX distance lookup table entry
#[derive(Debug, Clone)]
pub struct SAXDistanceInfo {
    /// Distance between two SAX words
    pub distance: f64,
}

/// Symbolic Aggregate approXimation (SAX)
///
/// Converts a time series into a discrete symbolic string by:
/// 1. Z-normalizing the data
/// 2. Applying PAA (Piecewise Aggregate Approximation)
/// 3. Mapping PAA values to symbols using equiprobable breakpoints of Normal(0,1)
pub struct SAX {
    config: SAXConfig,
}

impl SAX {
    /// Create a new SAX encoder
    pub fn new(config: SAXConfig) -> Self {
        Self { config }
    }

    /// PAA: reduce n-length series to w segments by averaging
    fn paa(data: &[f64], word_size: usize) -> Vec<f64> {
        let n = data.len();
        if word_size >= n {
            return data.to_vec();
        }
        let seg_len = n as f64 / word_size as f64;
        (0..word_size).map(|i| {
            let start = (i as f64 * seg_len).round() as usize;
            let end = ((i + 1) as f64 * seg_len).round() as usize;
            let end = end.min(n);
            let slice = &data[start..end];
            if slice.is_empty() {
                0.0
            } else {
                slice.iter().sum::<f64>() / slice.len() as f64
            }
        }).collect()
    }

    /// Map a PAA value to a SAX symbol index
    fn to_symbol(value: f64, breakpoints: &[f64]) -> usize {
        // Binary search for the appropriate symbol
        let pos = breakpoints.partition_point(|&bp| value >= bp);
        pos
    }

    /// Encode a time series as a SAX word
    pub fn encode(&self, data: &[f64]) -> Result<SAXRepresentation> {
        let n = data.len();
        let w = self.config.word_size;

        if n < w {
            return Err(TimeSeriesError::InsufficientData {
                message: "Series shorter than SAX word size".to_string(),
                required: w,
                actual: n,
            });
        }

        // Z-normalize
        let normalized = z_normalize(data);

        // PAA reduction
        let paa_values = Self::paa(&normalized, w);

        // Get breakpoints
        let breakpoints = normal_breakpoints(self.config.alphabet_size);

        // Map to symbols
        let word: String = paa_values.iter()
            .map(|&v| {
                let sym_idx = Self::to_symbol(v, &breakpoints);
                (b'a' + sym_idx as u8) as char
            })
            .collect();

        Ok(SAXRepresentation {
            word,
            alphabet_size: self.config.alphabet_size,
            word_size: w,
            paa_values,
        })
    }

    /// Compute MINDIST between two SAX words
    ///
    /// MINDIST(Q̂, Ĉ) = sqrt(n/w) * sqrt(sum_i dist_tab[q_i][c_i]^2)
    pub fn mindist(&self, rep_a: &SAXRepresentation, rep_b: &SAXRepresentation, n: usize) -> Result<f64> {
        if rep_a.word_size != rep_b.word_size {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: rep_a.word_size,
                actual: rep_b.word_size,
            });
        }

        let breakpoints = normal_breakpoints(self.config.alphabet_size);
        let w = rep_a.word_size;

        // Distance table between symbols
        let mut sum_sq = 0.0_f64;
        for (ca, cb) in rep_a.word.chars().zip(rep_b.word.chars()) {
            let ia = (ca as u8 - b'a') as usize;
            let ib = (cb as u8 - b'a') as usize;
            let dist = if ia.abs_diff(ib) <= 1 {
                0.0
            } else {
                let (lo, hi) = if ia < ib { (ia, ib) } else { (ib, ia) };
                // Distance between non-adjacent symbols
                breakpoints.get(hi - 1).cloned().unwrap_or(0.0)
                    - breakpoints.get(lo).cloned().unwrap_or(0.0)
            };
            sum_sq += dist * dist;
        }

        Ok(((n as f64 / w as f64) * sum_sq).sqrt())
    }

    /// Encode a sliding window of subsequences as SAX words
    pub fn encode_sliding_window(&self, data: &[f64], window_size: usize) -> Result<Vec<SAXRepresentation>> {
        let n = data.len();
        let w = self.config.word_size;

        if window_size < w {
            return Err(TimeSeriesError::InvalidInput(
                "Window size must be >= SAX word size".to_string()
            ));
        }
        if n < window_size {
            return Err(TimeSeriesError::InsufficientData {
                message: "Series shorter than window size".to_string(),
                required: window_size,
                actual: n,
            });
        }

        (0..=n - window_size)
            .map(|start| self.encode(&data[start..start + window_size]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(n: usize) -> Vec<f64> {
        (0..n).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 20.0).sin()).collect()
    }

    #[test]
    fn test_motif_discovery() {
        // Create data with a repeating pattern
        let mut data = make_sine(100);
        // Repeat a pattern starting at indices 10 and 50
        let pattern: Vec<f64> = (0..20).map(|i| (i as f64 * 0.5).sin() * 5.0).collect();
        for (i, &v) in pattern.iter().enumerate() {
            if i < data.len() { data[i + 10] = v; }
            if i + 50 < data.len() { data[i + 50] = v; }
        }

        let motif_finder = TimeSeriesMotif::new(MotifConfig {
            subsequence_len: 20,
            top_k: 3,
            ..Default::default()
        });
        let result = motif_finder.find_motifs(&data).expect("Motif discovery failed");
        assert!(!result.motif_pairs.is_empty(), "No motifs found");
        assert!(result.motif_pairs[0].distance >= 0.0);
    }

    #[test]
    fn test_discord_detection() {
        let mut data = make_sine(100);
        // Inject an unusual subsequence
        for i in 50..70 {
            data[i] = 100.0 * ((i - 50) as f64 * 0.3).cos();
        }

        let discord = Discord::new(DiscordConfig {
            subsequence_len: 20,
            top_k: 3,
            ..Default::default()
        });
        let results = discord.find_discords(&data).expect("Discord detection failed");
        assert!(!results.is_empty(), "No discords found");
    }

    #[test]
    fn test_shape_pattern_dtw() {
        let pattern = vec![0.0, 1.0, 2.0, 1.0, 0.0];
        let data: Vec<f64> = (0..50).map(|i| {
            if i >= 10 && i < 15 {
                [0.0, 1.0, 2.0, 1.0, 0.0][i - 10]
            } else {
                (i as f64 * 0.1).sin() * 0.1
            }
        }).collect();

        let matcher = ShapePattern::new(ShapePatternConfig {
            distance_threshold: Some(2.0),
            ..Default::default()
        });
        let matches = matcher.find_pattern(&data, &pattern).expect("Pattern matching failed");
        assert!(!matches.is_empty(), "No matches found");
    }

    #[test]
    fn test_sax_encode() {
        let data = make_sine(100);
        let sax = SAX::new(SAXConfig {
            word_size: 8,
            alphabet_size: 4,
            window_size: None,
        });
        let rep = sax.encode(&data).expect("SAX encoding failed");
        assert_eq!(rep.word.len(), 8);
        assert!(rep.word.chars().all(|c| c >= 'a' && c <= 'd'));
        assert_eq!(rep.paa_values.len(), 8);
    }

    #[test]
    fn test_sax_mindist() {
        let data1 = make_sine(100);
        let data2: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 2.0).collect();
        let sax = SAX::new(SAXConfig::default());
        let rep1 = sax.encode(&data1).expect("Failed");
        let rep2 = sax.encode(&data2).expect("Failed");
        let dist = sax.mindist(&rep1, &rep2, 100).expect("MINDIST failed");
        assert!(dist >= 0.0, "Distance should be non-negative");
    }

    #[test]
    fn test_sax_sliding_window() {
        let data = make_sine(100);
        let sax = SAX::new(SAXConfig { word_size: 4, alphabet_size: 4, window_size: None });
        let windows = sax.encode_sliding_window(&data, 20).expect("Sliding window SAX failed");
        assert_eq!(windows.len(), 81); // 100 - 20 + 1
    }
}
