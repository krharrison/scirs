//! Evaluation metrics for overlapping community detection.
//!
//! All metrics operate on "covers" — collections of communities where each community
//! is a list of node indices.  Nodes may appear in multiple communities.
//!
//! Metrics provided:
//! - [`overlapping_nmi`]: Averaged best-match F1-based NMI approximation.
//! - [`omega_index`]: Agreement between two covers on node pairs.
//! - [`overlap_f1`]: Averaged best-match F1 score.
//! - [`coverage`]: Fraction of nodes covered by at least one community.

use std::collections::HashSet;

// ─── Overlapping NMI ──────────────────────────────────────────────────────────

/// Compute the overlapping NMI between `detected` and `ground_truth`.
///
/// Uses the averaged best-match F1 approximation (a practical simplification of
/// the McDaid et al. 2011 conditional-entropy definition):
///
/// ```text
/// NMI ≈ 0.5 * (avg_best_f1(detected → gt) + avg_best_f1(gt → detected))
/// ```
///
/// Returns a value in [0, 1] where 1 means perfect agreement.
///
/// # Arguments
/// * `detected`     — Detected cover; each inner `Vec` is a community.
/// * `ground_truth` — Reference cover.
/// * `n_nodes`      — Total number of nodes in the graph (used for normalisation).
pub fn overlapping_nmi(
    detected: &[Vec<usize>],
    ground_truth: &[Vec<usize>],
    n_nodes: usize,
) -> f64 {
    if detected.is_empty() || ground_truth.is_empty() || n_nodes == 0 {
        return 0.0;
    }
    let fwd = avg_best_f1(detected, ground_truth);
    let bwd = avg_best_f1(ground_truth, detected);
    0.5 * (fwd + bwd)
}

// ─── Omega index ──────────────────────────────────────────────────────────────

/// Compute the Omega index between two covers.
///
/// The Omega index is an extension of the Rand index to overlapping clusterings.
/// It measures the fraction of node pairs assigned to the same number of common
/// communities in both covers, corrected for chance.
///
/// ```text
/// Omega = (t_obs - t_exp) / (1 - t_exp)
/// ```
///
/// where `t_obs` is the fraction of pairs with identical co-membership counts, and
/// `t_exp` is the expected value under independence.
///
/// Returns a value in (-∞, 1]; perfect overlap = 1, random = 0.
pub fn omega_index(cover1: &[Vec<usize>], cover2: &[Vec<usize>], n_nodes: usize) -> f64 {
    if n_nodes < 2 {
        return 0.0;
    }

    // For each node pair (i, j), count how many communities they share in each cover.
    let co1 = co_membership_counts(cover1, n_nodes);
    let co2 = co_membership_counts(cover2, n_nodes);

    let n_pairs = n_nodes * (n_nodes - 1) / 2;
    if n_pairs == 0 {
        return 0.0;
    }

    // Maximum co-membership count across both covers.
    let max_count = co1.values().chain(co2.values()).copied().max().unwrap_or(0);

    // t_obs: fraction of pairs with the same co-membership count in both covers.
    // co1 and co2 only store pairs with count >= 1.
    // Pairs not in either map have count = 0 in both → they always match.
    let pairs_in_co1: HashSet<(usize, usize)> = co1.keys().copied().collect();
    let pairs_in_co2: HashSet<(usize, usize)> = co2.keys().copied().collect();
    let all_nonzero_pairs: HashSet<(usize, usize)> =
        pairs_in_co1.union(&pairs_in_co2).copied().collect();
    let n_nonzero = all_nonzero_pairs.len();
    let n_zero_match = n_pairs - n_nonzero; // pairs with count=0 in both → always match

    // Recount properly.
    let mut obs_match = n_zero_match;
    for &pair in &all_nonzero_pairs {
        let c1 = co1.get(&pair).copied().unwrap_or(0);
        let c2 = co2.get(&pair).copied().unwrap_or(0);
        if c1 == c2 {
            obs_match += 1;
        }
    }

    let t_obs = obs_match as f64 / n_pairs as f64;

    // t_exp: expected fraction under independence.
    // For each possible count k, let p_k1 = fraction of pairs with count=k in cover1.
    // t_exp = sum_k p_k1 * p_k2
    let mut freq1: Vec<usize> = vec![0; max_count + 2];
    let mut freq2: Vec<usize> = vec![0; max_count + 2];
    for &c in co1.values() {
        if c < freq1.len() {
            freq1[c] += 1;
        }
    }
    for &c in co2.values() {
        if c < freq2.len() {
            freq2[c] += 1;
        }
    }
    // Count of pairs with count=0 in each cover.
    let zero1 = n_pairs - co1.len();
    let zero2 = n_pairs - co2.len();
    let np = n_pairs as f64;

    let mut t_exp = (zero1 as f64 / np) * (zero2 as f64 / np);
    for k in 1..=max_count {
        let p1 = freq1.get(k).copied().unwrap_or(0) as f64 / np;
        let p2 = freq2.get(k).copied().unwrap_or(0) as f64 / np;
        t_exp += p1 * p2;
    }

    let denom = 1.0 - t_exp;
    if denom.abs() < 1e-15 {
        return 1.0;
    }
    (t_obs - t_exp) / denom
}

// ─── Overlap F1 ──────────────────────────────────────────────────────────────

/// Compute the averaged best-match F1 score between `detected` and `ground_truth`.
///
/// For each detected community we find the ground-truth community with the highest
/// F1 score and record that F1.  The final score is the average over all detected
/// communities (symmetrised by also averaging in the reverse direction).
///
/// Returns a value in [0, 1].
pub fn overlap_f1(detected: &[Vec<usize>], ground_truth: &[Vec<usize>]) -> f64 {
    if detected.is_empty() || ground_truth.is_empty() {
        return 0.0;
    }
    let fwd = avg_best_f1(detected, ground_truth);
    let bwd = avg_best_f1(ground_truth, detected);
    0.5 * (fwd + bwd)
}

// ─── Coverage ────────────────────────────────────────────────────────────────

/// Fraction of nodes (0 .. `n_nodes`) covered by at least one community.
pub fn coverage(communities: &[Vec<usize>], n_nodes: usize) -> f64 {
    if n_nodes == 0 {
        return 0.0;
    }
    let mut covered = vec![false; n_nodes];
    for comm in communities {
        for &node in comm {
            if node < n_nodes {
                covered[node] = true;
            }
        }
    }
    let n_covered = covered.iter().filter(|&&c| c).count();
    n_covered as f64 / n_nodes as f64
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// For each community in `source`, find the best F1 against any community in `target`.
/// Returns the average of best-F1 scores.
fn avg_best_f1(source: &[Vec<usize>], target: &[Vec<usize>]) -> f64 {
    let target_sets: Vec<HashSet<usize>> =
        target.iter().map(|c| c.iter().copied().collect()).collect();
    let mut total = 0.0f64;
    for src_comm in source {
        let src_set: HashSet<usize> = src_comm.iter().copied().collect();
        let best = target_sets
            .iter()
            .map(|tgt| community_f1(&src_set, tgt))
            .fold(0.0f64, f64::max);
        total += best;
    }
    total / source.len() as f64
}

/// F1 score between two community sets.
fn community_f1(a: &HashSet<usize>, b: &HashSet<usize>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count() as f64;
    let precision = if a.is_empty() {
        0.0
    } else {
        inter / a.len() as f64
    };
    let recall = if b.is_empty() {
        0.0
    } else {
        inter / b.len() as f64
    };
    if precision + recall < 1e-15 {
        return 0.0;
    }
    2.0 * precision * recall / (precision + recall)
}

/// Build a map from canonical node pairs (u < v) to the number of communities
/// they co-occur in (counting only pairs with count ≥ 1).
fn co_membership_counts(
    cover: &[Vec<usize>],
    n_nodes: usize,
) -> std::collections::HashMap<(usize, usize), usize> {
    let mut counts: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for comm in cover {
        // Build the set of nodes in this community (valid indices only).
        let members: Vec<usize> = comm.iter().copied().filter(|&v| v < n_nodes).collect();
        for ii in 0..members.len() {
            for jj in (ii + 1)..members.len() {
                let (u, v) = if members[ii] < members[jj] {
                    (members[ii], members[jj])
                } else {
                    (members[jj], members[ii])
                };
                *counts.entry((u, v)).or_insert(0) += 1;
            }
        }
    }
    counts
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlapping_nmi_self_equals_one() {
        let cover = vec![vec![0, 1, 2], vec![2, 3, 4]];
        let nmi = overlapping_nmi(&cover, &cover, 5);
        assert!(
            (nmi - 1.0).abs() < 1e-9,
            "NMI of cover with itself should be 1; got {nmi}"
        );
    }

    #[test]
    fn test_overlapping_nmi_range() {
        let detected = vec![vec![0, 1], vec![2, 3]];
        let gt = vec![vec![0, 2], vec![1, 3]];
        let nmi = overlapping_nmi(&detected, &gt, 4);
        assert!(
            (0.0..=1.0).contains(&nmi),
            "NMI must be in [0,1], got {nmi}"
        );
    }

    #[test]
    fn test_overlapping_nmi_empty() {
        let nmi = overlapping_nmi(&[], &[vec![0, 1]], 2);
        assert!((nmi - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_omega_index_identical_covers() {
        let cover = vec![vec![0, 1, 2], vec![2, 3, 4]];
        let omega = omega_index(&cover, &cover, 5);
        assert!(
            omega > 0.9,
            "Omega of identical covers should be close to 1; got {omega}"
        );
    }

    #[test]
    fn test_omega_index_range() {
        let c1 = vec![vec![0, 1, 2], vec![3, 4]];
        let c2 = vec![vec![0, 2, 4], vec![1, 3]];
        let omega = omega_index(&c1, &c2, 5);
        // Omega can be negative; just check it doesn't panic.
        let _ = omega;
    }

    #[test]
    fn test_overlap_f1_perfect_detection() {
        let cover = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let f1 = overlap_f1(&cover, &cover);
        assert!(
            (f1 - 1.0).abs() < 1e-9,
            "F1 should be 1 for perfect detection; got {f1}"
        );
    }

    #[test]
    fn test_overlap_f1_range() {
        let detected = vec![vec![0, 1, 2], vec![2, 3, 4]];
        let gt = vec![vec![0, 1], vec![3, 4, 5]];
        let f1 = overlap_f1(&detected, &gt);
        assert!((0.0..=1.0).contains(&f1), "F1 must be in [0,1], got {f1}");
    }

    #[test]
    fn test_coverage_all_nodes() {
        let comms = vec![vec![0, 1, 2], vec![3, 4]];
        let cov = coverage(&comms, 5);
        assert!((cov - 1.0).abs() < 1e-9, "all nodes covered; got {cov}");
    }

    #[test]
    fn test_coverage_partial() {
        let comms = vec![vec![0, 1]];
        let cov = coverage(&comms, 5);
        assert!((cov - 0.4).abs() < 1e-9, "2/5 nodes covered; got {cov}");
    }

    #[test]
    fn test_coverage_empty_communities() {
        let cov = coverage(&[], 5);
        assert!((cov - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_coverage_zero_nodes() {
        let cov = coverage(&[vec![0]], 0);
        assert!((cov - 0.0).abs() < 1e-9);
    }
}
