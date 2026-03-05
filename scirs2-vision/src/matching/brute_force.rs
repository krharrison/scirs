//! Brute-force descriptor matching.
//!
//! Provides O(n×m) nearest-neighbour matching for both float (L2) and binary
//! (Hamming) descriptors, plus a mutual-best-match cross-check filter.

use crate::error::VisionError;

// ─── Distance metric ─────────────────────────────────────────────────────────

/// Distance metric selector used in [`BruteForceMatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance — for float descriptors.
    L2,
    /// Hamming distance — for binary (packed u64) descriptors.
    Hamming,
}

/// Brute-force descriptor matcher.
#[derive(Debug, Clone)]
pub struct BruteForceMatch {
    /// Distance metric to use.
    pub distance: DistanceMetric,
}

impl Default for BruteForceMatch {
    fn default() -> Self {
        Self {
            distance: DistanceMetric::L2,
        }
    }
}

// ─── L2 matching ─────────────────────────────────────────────────────────────

/// Compute the squared L2 distance between two float descriptor vectors.
#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Match float descriptors using L2 distance.
///
/// For each descriptor in `desc1`, finds the single nearest neighbour in
/// `desc2`.  Returns a sorted list of `(idx1, idx2, l2_distance)` triples
/// ordered by ascending distance.
///
/// # Errors
/// Returns [`VisionError`] if descriptor lengths are inconsistent.
pub fn match_descriptors_l2(
    desc1: &[Vec<f32>],
    desc2: &[Vec<f32>],
) -> Result<Vec<(usize, usize, f32)>, VisionError> {
    if desc1.is_empty() || desc2.is_empty() {
        return Ok(vec![]);
    }
    let dim = desc1[0].len();
    for d in desc1.iter().chain(desc2.iter()) {
        if d.len() != dim {
            return Err(VisionError::InvalidInput(
                "Inconsistent descriptor lengths in L2 matching".into(),
            ));
        }
    }

    let mut matches: Vec<(usize, usize, f32)> = Vec::with_capacity(desc1.len());
    for (i, d1) in desc1.iter().enumerate() {
        let (best_j, best_dist) = desc2
            .iter()
            .enumerate()
            .map(|(j, d2)| (j, l2_sq(d1, d2).sqrt()))
            .fold((0_usize, f32::INFINITY), |(bj, bd), (j, d)| {
                if d < bd {
                    (j, d)
                } else {
                    (bj, bd)
                }
            });
        matches.push((i, best_j, best_dist));
    }
    matches.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    Ok(matches)
}

// ─── Hamming matching ────────────────────────────────────────────────────────

/// Compute the Hamming distance between two packed-u64 binary descriptors.
#[inline]
fn hamming_u64(a: &[u64], b: &[u64]) -> u32 {
    let len = a.len().max(b.len());
    (0..len)
        .map(|i| {
            let wa = a.get(i).copied().unwrap_or(0);
            let wb = b.get(i).copied().unwrap_or(0);
            (wa ^ wb).count_ones()
        })
        .sum()
}

/// Match binary descriptors using Hamming distance.
///
/// `bits_per_word` is informational (typically 64); matching uses the full
/// u64 word array.  Returns `(idx1, idx2, hamming_distance)` sorted ascending.
///
/// # Errors
/// Returns [`VisionError`] if either input is empty.
pub fn match_descriptors_hamming(
    desc1: &[Vec<u64>],
    desc2: &[Vec<u64>],
    _bits_per_word: usize,
) -> Result<Vec<(usize, usize, u32)>, VisionError> {
    if desc1.is_empty() || desc2.is_empty() {
        return Ok(vec![]);
    }

    let mut matches: Vec<(usize, usize, u32)> = Vec::with_capacity(desc1.len());
    for (i, d1) in desc1.iter().enumerate() {
        let (best_j, best_dist) = desc2
            .iter()
            .enumerate()
            .map(|(j, d2)| (j, hamming_u64(d1, d2)))
            .fold(
                (0_usize, u32::MAX),
                |(bj, bd), (j, d)| if d < bd { (j, d) } else { (bj, bd) },
            );
        matches.push((i, best_j, best_dist));
    }
    matches.sort_by_key(|m| m.2);
    Ok(matches)
}

// ─── Cross-check filter ──────────────────────────────────────────────────────

/// Mutual-best-match (cross-check) filter for L2 matches.
///
/// Keeps match `(i, j, d)` from `matches_12` only if the best match from
/// `matches_21` for index `j` also points back to `i`.
pub fn cross_check_filter(
    matches_12: &[(usize, usize, f32)],
    matches_21: &[(usize, usize, f32)],
) -> Vec<(usize, usize, f32)> {
    // Build reverse lookup: best match for each desc2 index → desc1 index.
    let mut best_21: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for &(j, i, _) in matches_21 {
        // matches_21 is sorted; only keep the first (best) entry per j.
        best_21.entry(j).or_insert(i);
    }

    matches_12
        .iter()
        .filter(|&&(i, j, _)| best_21.get(&j).copied() == Some(i))
        .copied()
        .collect()
}

/// Cross-check filter for Hamming matches.
pub fn cross_check_filter_hamming(
    matches_12: &[(usize, usize, u32)],
    matches_21: &[(usize, usize, u32)],
) -> Vec<(usize, usize, u32)> {
    let mut best_21: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for &(j, i, _) in matches_21 {
        best_21.entry(j).or_insert(i);
    }
    matches_12
        .iter()
        .filter(|&&(i, j, _)| best_21.get(&j).copied() == Some(i))
        .copied()
        .collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(dim: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        v[hot % dim] = 1.0;
        v
    }

    #[test]
    fn test_l2_self_matching() {
        let descs: Vec<Vec<f32>> = (0..5).map(|i| unit_vec(8, i)).collect();
        let matches =
            match_descriptors_l2(&descs, &descs).expect("match_descriptors_l2 should succeed");
        // Every descriptor should match itself.
        for &(i, j, d) in &matches {
            assert_eq!(i, j);
            assert!(d < 1e-5);
        }
    }

    #[test]
    fn test_l2_sorted() {
        let d1 = vec![vec![1.0_f32; 4], vec![0.0; 4]];
        let d2 = vec![vec![0.9_f32; 4], vec![0.1; 4], vec![0.5; 4]];
        let matches = match_descriptors_l2(&d1, &d2).expect("match_descriptors_l2 should succeed");
        for w in matches.windows(2) {
            assert!(w[0].2 <= w[1].2);
        }
    }

    #[test]
    fn test_hamming_self_matching() {
        let descs: Vec<Vec<u64>> = (0..4).map(|i| vec![1_u64 << i]).collect();
        let matches = match_descriptors_hamming(&descs, &descs, 64)
            .expect("match_descriptors_hamming should succeed");
        for &(i, j, d) in &matches {
            assert_eq!(i, j);
            assert_eq!(d, 0);
        }
    }

    #[test]
    fn test_cross_check_symmetry() {
        let descs: Vec<Vec<f32>> = (0..4).map(|i| unit_vec(8, i)).collect();
        let m12 = match_descriptors_l2(&descs, &descs)
            .expect("match_descriptors_l2 should succeed for cross-check");
        let m21 = match_descriptors_l2(&descs, &descs)
            .expect("match_descriptors_l2 should succeed for cross-check");
        let filtered = cross_check_filter(&m12, &m21);
        // Symmetric; all matches should survive.
        assert_eq!(filtered.len(), descs.len());
    }

    #[test]
    fn test_l2_empty() {
        let d1: Vec<Vec<f32>> = vec![];
        let d2: Vec<Vec<f32>> = vec![vec![1.0]];
        let matches = match_descriptors_l2(&d1, &d2)
            .expect("match_descriptors_l2 should succeed on empty query");
        assert!(matches.is_empty());
    }

    #[test]
    fn test_l2_dim_mismatch() {
        let d1 = vec![vec![1.0_f32; 4]];
        let d2 = vec![vec![1.0_f32; 3]];
        assert!(match_descriptors_l2(&d1, &d2).is_err());
    }
}
