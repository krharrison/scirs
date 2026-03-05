//! Feature descriptor matching algorithms
//!
//! Provides multiple matching strategies for both floating-point (SIFT-like)
//! and binary (ORB-like) descriptors:
//!
//! - **`BruteForce`** – exact O(n·m) exhaustive search
//! - **`FlannLike`** – approximate nearest-neighbour via a randomised
//!   kd-tree forest (float descriptors) or multi-probe LSH (binary descriptors)
//! - **`RatioTest`** – Lowe's ratio test: a match is retained only when the
//!   best-match distance is < `ratio × second-best-match distance`.
//!
//! All three methods return `(idx1, idx2, distance)` triples.

use crate::error::{Result, VisionError};
use crate::features::orb_like::{hamming_distance, OrbLikeDescriptor, DESC_WORDS};
use crate::features::sift_like::SIFTDescriptor;

// ─── Match method enum ────────────────────────────────────────────────────────

/// Strategy used to match feature descriptors.
#[derive(Debug, Clone)]
pub enum MatchMethod {
    /// Exhaustive brute-force matching (exact nearest neighbour).
    BruteForce,
    /// Approximate nearest-neighbour with a randomised forest / LSH index.
    FlannLike {
        /// Number of kd-trees (float) or hash tables (binary)
        num_trees: usize,
        /// Number of candidate checks per query
        checks: usize,
    },
    /// Lowe's ratio test: keep matches where `d_best / d_second < ratio`.
    RatioTest {
        /// Ratio threshold (typical value: 0.75)
        ratio: f64,
    },
}

impl Default for MatchMethod {
    fn default() -> Self {
        MatchMethod::RatioTest { ratio: 0.75 }
    }
}

// ─── Float descriptor matching (SIFT-like) ───────────────────────────────────

/// Match two sets of SIFT-like descriptors.
///
/// # Arguments
///
/// * `desc1` – Query descriptors
/// * `desc2` – Train descriptors
/// * `method` – Matching strategy
///
/// # Returns
///
/// Vector of `(idx1, idx2, distance)` triples sorted by distance ascending.
pub fn match_descriptors(
    desc1: &[SIFTDescriptor],
    desc2: &[SIFTDescriptor],
    method: &MatchMethod,
) -> Result<Vec<(usize, usize, f64)>> {
    if desc1.is_empty() || desc2.is_empty() {
        return Ok(Vec::new());
    }

    // Extract raw float vectors for matching
    let vecs1: Vec<&[f32]> = desc1.iter().map(|d| d.descriptor.as_slice()).collect();
    let vecs2: Vec<&[f32]> = desc2.iter().map(|d| d.descriptor.as_slice()).collect();

    let dim = vecs1[0].len();
    for v in vecs1.iter().chain(vecs2.iter()) {
        if v.len() != dim {
            return Err(VisionError::InvalidParameter(format!(
                "Descriptor dimension mismatch: expected {dim}, got {}",
                v.len()
            )));
        }
    }

    let matches = match method {
        MatchMethod::BruteForce => brute_force_float(&vecs1, &vecs2),
        MatchMethod::FlannLike { num_trees, checks } => {
            flann_like_float(&vecs1, &vecs2, *num_trees, *checks)
        }
        MatchMethod::RatioTest { ratio } => ratio_test_float(&vecs1, &vecs2, *ratio),
    };

    Ok(matches)
}

/// Exhaustive L2 nearest-neighbour.
fn brute_force_float(q: &[&[f32]], t: &[&[f32]]) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::with_capacity(q.len());
    for (i, qi) in q.iter().enumerate() {
        let mut best_dist = f64::MAX;
        let mut best_j = 0usize;
        for (j, tj) in t.iter().enumerate() {
            let d = l2_distance_f32(qi, tj);
            if d < best_dist {
                best_dist = d;
                best_j = j;
            }
        }
        out.push((i, best_j, best_dist));
    }
    out.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Approximate nearest-neighbour using a lightweight randomised kd-tree forest.
///
/// For each query, traversal with `checks` node evaluations is performed.
/// Falls back to brute-force for small descriptor sets.
fn flann_like_float(
    q: &[&[f32]],
    t: &[&[f32]],
    num_trees: usize,
    checks: usize,
) -> Vec<(usize, usize, f64)> {
    if t.len() < 16 {
        return brute_force_float(q, t);
    }

    // Build multiple randomised kd-trees
    let trees: Vec<KdNode> = (0..num_trees.max(1))
        .map(|seed| build_kdtree(t, seed as u64))
        .collect();

    let mut out = Vec::with_capacity(q.len());
    for (i, qi) in q.iter().enumerate() {
        let mut best_dist = f64::MAX;
        let mut best_j = 0usize;

        for tree in &trees {
            let (j, d) = search_kdtree(tree, qi, t, checks);
            if d < best_dist {
                best_dist = d;
                best_j = j;
            }
        }
        out.push((i, best_j, best_dist));
    }
    out.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Lowe's ratio test: keep a match only if d1 / d2 < ratio.
fn ratio_test_float(q: &[&[f32]], t: &[&[f32]], ratio: f64) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    for (i, qi) in q.iter().enumerate() {
        let mut first = (f64::MAX, 0usize);
        let mut second = f64::MAX;

        for (j, tj) in t.iter().enumerate() {
            let d = l2_distance_f32(qi, tj);
            if d < first.0 {
                second = first.0;
                first = (d, j);
            } else if d < second {
                second = d;
            }
        }

        if second > 0.0 && first.0 / second < ratio {
            out.push((i, first.1, first.0));
        }
    }
    out.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    out
}

// ─── Binary descriptor matching (ORB-like) ───────────────────────────────────

/// Match two sets of ORB-like binary descriptors using Hamming distance.
///
/// # Arguments
///
/// * `desc1` – Query descriptors
/// * `desc2` – Train descriptors
/// * `method` – Matching strategy
///
/// # Returns
///
/// Vector of `(idx1, idx2, distance)` where distance is the Hamming distance
/// (number of differing bits, 0–256).
pub fn match_binary_descriptors(
    desc1: &[OrbLikeDescriptor],
    desc2: &[OrbLikeDescriptor],
    method: &MatchMethod,
) -> Result<Vec<(usize, usize, f64)>> {
    if desc1.is_empty() || desc2.is_empty() {
        return Ok(Vec::new());
    }

    let refs1: Vec<&[u32; DESC_WORDS]> = desc1.iter().map(|d| &d.descriptor).collect();
    let refs2: Vec<&[u32; DESC_WORDS]> = desc2.iter().map(|d| &d.descriptor).collect();

    let matches = match method {
        MatchMethod::BruteForce => brute_force_binary(&refs1, &refs2),
        MatchMethod::FlannLike { checks, .. } => {
            // For binary: approximate via multi-table LSH
            lsh_binary(&refs1, &refs2, *checks)
        }
        MatchMethod::RatioTest { ratio } => ratio_test_binary(&refs1, &refs2, *ratio),
    };

    Ok(matches)
}

/// Exhaustive Hamming nearest-neighbour for binary descriptors.
fn brute_force_binary(
    q: &[&[u32; DESC_WORDS]],
    t: &[&[u32; DESC_WORDS]],
) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::with_capacity(q.len());
    for (i, qi) in q.iter().enumerate() {
        let mut best = (u32::MAX, 0usize);
        for (j, tj) in t.iter().enumerate() {
            let d = hamming_distance(qi, tj);
            if d < best.0 {
                best = (d, j);
            }
        }
        out.push((i, best.1, best.0 as f64));
    }
    out.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Approximate Hamming matching via a simple multi-probe approach:
/// for each query, randomly sub-sample `checks` train descriptors plus a
/// guided sweep of the top hash-bucket.
fn lsh_binary(
    q: &[&[u32; DESC_WORDS]],
    t: &[&[u32; DESC_WORDS]],
    checks: usize,
) -> Vec<(usize, usize, f64)> {
    if t.len() <= checks {
        return brute_force_binary(q, t);
    }

    // Build a simple random-projection table for approximate bucketing
    // Hash = XOR of selected descriptor words
    let bucket_count = (t.len() / 4).next_power_of_two().max(64);
    let mask = bucket_count - 1;

    // Assign train descriptors to buckets (based on first word XOR hash)
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); bucket_count];
    for (j, tj) in t.iter().enumerate() {
        let hash = (tj[0] ^ tj[1].rotate_left(8) ^ tj[2].rotate_left(16)) as usize & mask;
        buckets[hash].push(j);
    }

    let mut out = Vec::with_capacity(q.len());
    for (i, qi) in q.iter().enumerate() {
        let query_hash = (qi[0] ^ qi[1].rotate_left(8) ^ qi[2].rotate_left(16)) as usize & mask;

        let mut candidates: Vec<usize> = buckets[query_hash].clone();
        // Also probe neighbouring buckets
        let nb1 = (query_hash + 1) & mask;
        let nb2 = (query_hash + bucket_count - 1) & mask;
        candidates.extend_from_slice(&buckets[nb1]);
        candidates.extend_from_slice(&buckets[nb2]);

        // Supplement with uniformly spaced samples if we have too few candidates
        if candidates.len() < checks {
            let step = t.len() / (checks - candidates.len()).max(1);
            for k in (0..t.len()).step_by(step.max(1)) {
                candidates.push(k);
            }
        }

        candidates.sort_unstable();
        candidates.dedup();

        let mut best = (u32::MAX, 0usize);
        for j in candidates {
            if j < t.len() {
                let d = hamming_distance(qi, t[j]);
                if d < best.0 {
                    best = (d, j);
                }
            }
        }

        out.push((i, best.1, best.0 as f64));
    }

    out.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Lowe's ratio test for binary (Hamming) descriptors.
fn ratio_test_binary(
    q: &[&[u32; DESC_WORDS]],
    t: &[&[u32; DESC_WORDS]],
    ratio: f64,
) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    for (i, qi) in q.iter().enumerate() {
        let mut first = (u32::MAX, 0usize);
        let mut second = u32::MAX;

        for (j, tj) in t.iter().enumerate() {
            let d = hamming_distance(qi, tj);
            if d < first.0 {
                second = first.0;
                first = (d, j);
            } else if d < second {
                second = d;
            }
        }

        if second > 0 {
            let r = first.0 as f64 / second as f64;
            if r < ratio {
                out.push((i, first.1, first.0 as f64));
            }
        }
    }
    out.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    out
}

// ─── Symmetric / cross-check filtering ───────────────────────────────────────

/// Filter a set of SIFT-like matches using symmetric (cross-check) validation.
///
/// Keeps only matches (i → j) where the reverse match (j → i) is also i.
/// This removes many spurious correspondences at the cost of slightly
/// fewer total matches.
pub fn symmetric_filter(
    desc1: &[SIFTDescriptor],
    desc2: &[SIFTDescriptor],
    method: &MatchMethod,
) -> Result<Vec<(usize, usize, f64)>> {
    let fwd = match_descriptors(desc1, desc2, method)?;
    let rev = match_descriptors(desc2, desc1, method)?;

    // Build reverse lookup: j → i
    let mut rev_map: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::with_capacity(rev.len());
    for (j, i, _) in &rev {
        rev_map.insert(*j, *i);
    }

    let symmetric: Vec<(usize, usize, f64)> = fwd
        .into_iter()
        .filter(|(i, j, _)| rev_map.get(j).is_some_and(|&ri| ri == *i))
        .collect();

    Ok(symmetric)
}

// ─── kd-tree (float, randomised) ─────────────────────────────────────────────

/// Node in a randomised kd-tree.
enum KdNode {
    Leaf {
        indices: Vec<usize>,
    },
    Internal {
        axis: usize,
        split_val: f32,
        left: Box<KdNode>,
        right: Box<KdNode>,
    },
}

/// Build a randomised kd-tree over `vecs` (indexed by position).
fn build_kdtree(vecs: &[&[f32]], seed: u64) -> KdNode {
    let indices: Vec<usize> = (0..vecs.len()).collect();
    build_kdtree_rec(&indices, vecs, seed, 0)
}

fn build_kdtree_rec(indices: &[usize], vecs: &[&[f32]], seed: u64, depth: usize) -> KdNode {
    const LEAF_SIZE: usize = 8;
    if indices.len() <= LEAF_SIZE {
        return KdNode::Leaf {
            indices: indices.to_vec(),
        };
    }

    let dim = vecs[0].len();
    // Randomised axis selection: pick the axis with highest variance among a
    // random subset of 5 dimensions.
    let axis = choose_split_axis(indices, vecs, seed, depth, dim);

    // Split at the median
    let mut vals: Vec<f32> = indices.iter().map(|&i| vecs[i][axis]).collect();
    vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let split_val = vals[vals.len() / 2];

    let left_idx: Vec<usize> = indices
        .iter()
        .copied()
        .filter(|&i| vecs[i][axis] < split_val)
        .collect();
    let right_idx: Vec<usize> = indices
        .iter()
        .copied()
        .filter(|&i| vecs[i][axis] >= split_val)
        .collect();

    // Guard against degenerate splits
    if left_idx.is_empty() || right_idx.is_empty() {
        return KdNode::Leaf {
            indices: indices.to_vec(),
        };
    }

    KdNode::Internal {
        axis,
        split_val,
        left: Box::new(build_kdtree_rec(&left_idx, vecs, seed, depth + 1)),
        right: Box::new(build_kdtree_rec(&right_idx, vecs, seed, depth + 1)),
    }
}

fn choose_split_axis(
    indices: &[usize],
    vecs: &[&[f32]],
    seed: u64,
    depth: usize,
    dim: usize,
) -> usize {
    // Sample up to 5 random dimensions and pick the one with highest variance
    let n_sample = 5usize.min(dim);
    let mut rng_state = seed.wrapping_add(depth as u64 * 6_364_136_223_846_793_005);

    let sample_n = indices.len().min(32);

    let mut best_axis = 0usize;
    let mut best_var = f64::NEG_INFINITY;

    for _ in 0..n_sample {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let axis = (rng_state >> 33) as usize % dim;

        // Compute variance of this axis over a subset
        let step = indices.len() / sample_n + 1;
        let sampled: Vec<f64> = indices
            .iter()
            .step_by(step)
            .take(sample_n)
            .map(|&i| vecs[i][axis] as f64)
            .collect();

        let n = sampled.len() as f64;
        if n < 2.0 {
            continue;
        }
        let mean = sampled.iter().sum::<f64>() / n;
        let var = sampled.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;

        if var > best_var {
            best_var = var;
            best_axis = axis;
        }
    }

    best_axis
}

/// Approximate nearest-neighbour search in a kd-tree with `checks` leaf evaluations.
fn search_kdtree(root: &KdNode, query: &[f32], vecs: &[&[f32]], checks: usize) -> (usize, f64) {
    let mut best = (f64::MAX, 0usize);
    let mut evals = 0usize;
    search_kdtree_rec(root, query, vecs, &mut best, &mut evals, checks);
    (best.1, best.0)
}

fn search_kdtree_rec(
    node: &KdNode,
    query: &[f32],
    vecs: &[&[f32]],
    best: &mut (f64, usize),
    evals: &mut usize,
    max_evals: usize,
) {
    if *evals >= max_evals {
        return;
    }

    match node {
        KdNode::Leaf { indices } => {
            for &i in indices {
                *evals += 1;
                let d = l2_distance_f32(query, vecs[i]);
                if d < best.0 {
                    *best = (d, i);
                }
                if *evals >= max_evals {
                    return;
                }
            }
        }
        KdNode::Internal {
            axis,
            split_val,
            left,
            right,
        } => {
            let q_val = query[*axis];
            let (near, far) = if q_val < *split_val {
                (left.as_ref(), right.as_ref())
            } else {
                (right.as_ref(), left.as_ref())
            };

            search_kdtree_rec(near, query, vecs, best, evals, max_evals);

            // Backtrack to far side if potentially useful
            let plane_dist = (q_val - split_val).powi(2) as f64;
            if plane_dist < best.0 && *evals < max_evals {
                search_kdtree_rec(far, query, vecs, best, evals, max_evals);
            }
        }
    }
}

// ─── Distance helpers ─────────────────────────────────────────────────────────

/// Squared L2 distance between two float descriptor vectors.
fn l2_distance_f32(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            (d * d) as f64
        })
        .sum::<f64>()
        .sqrt()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::sift_like::{Keypoint, SIFTDescriptor};

    fn make_sift_desc(id: usize, perturb: f32) -> SIFTDescriptor {
        let mut desc = vec![0.0f32; 128];
        desc[id % 128] = 1.0;
        // tiny perturbation so ratio test can differentiate
        desc[(id + 1) % 128] = perturb;
        // normalise
        let norm: f32 = desc.iter().map(|v| v * v).sum::<f32>().sqrt();
        for v in &mut desc {
            *v /= norm;
        }
        SIFTDescriptor {
            keypoint: Keypoint {
                x: id as f64,
                y: id as f64,
                scale: 1.0,
                orientation: 0.0,
                response: 1.0,
                octave: 0,
            },
            descriptor: desc,
        }
    }

    fn make_orb_desc(id: usize) -> OrbLikeDescriptor {
        let mut words = [0u32; DESC_WORDS];
        words[id % DESC_WORDS] = (id as u32).wrapping_mul(0x12345678);
        OrbLikeDescriptor {
            keypoint: crate::features::orb_like::OrbKeypoint {
                x: id as f64,
                y: id as f64,
                score: 1.0,
                orientation: 0.0,
                level: 0,
            },
            descriptor: words,
        }
    }

    #[test]
    fn test_brute_force_exact_match() {
        let descs: Vec<SIFTDescriptor> = (0..5).map(|i| make_sift_desc(i, 0.0)).collect();
        let matches = match_descriptors(&descs, &descs, &MatchMethod::BruteForce)
            .expect("match_descriptors should succeed");
        // Each descriptor should match itself (distance 0)
        for (i, j, d) in &matches {
            assert_eq!(i, j, "Self-match expected at index {i}");
            assert!(*d < 1e-6, "Self-match distance should be ~0, got {d}");
        }
    }

    #[test]
    fn test_ratio_test_returns_matches() {
        let q: Vec<SIFTDescriptor> = (0..4).map(|i| make_sift_desc(i, 0.01)).collect();
        let t: Vec<SIFTDescriptor> = (0..8).map(|i| make_sift_desc(i, 0.01)).collect();
        let m = match_descriptors(&q, &t, &MatchMethod::RatioTest { ratio: 0.9 })
            .expect("match_descriptors with ratio test should succeed");
        // Should return at least some matches
        assert!(!m.is_empty());
    }

    #[test]
    fn test_flann_like_consistent_with_brute_force_small() {
        // For a tiny set, FLANN should fall back to brute-force and give the same result
        let descs: Vec<SIFTDescriptor> = (0..5).map(|i| make_sift_desc(i, 0.0)).collect();
        let bf = match_descriptors(&descs, &descs, &MatchMethod::BruteForce)
            .expect("brute force match should succeed");
        let fl = match_descriptors(
            &descs,
            &descs,
            &MatchMethod::FlannLike {
                num_trees: 2,
                checks: 50,
            },
        )
        .expect("flann-like match should succeed");
        // Both should match each descriptor to itself
        assert_eq!(bf.len(), fl.len());
    }

    #[test]
    fn test_binary_brute_force() {
        let d: Vec<OrbLikeDescriptor> = (0..4).map(|i| make_orb_desc(i)).collect();
        let matches = match_binary_descriptors(&d, &d, &MatchMethod::BruteForce)
            .expect("match_binary_descriptors should succeed");
        for (i, j, dist) in &matches {
            assert_eq!(i, j, "Binary self-match expected");
            assert_eq!(*dist, 0.0, "Hamming self-distance should be 0");
        }
    }

    #[test]
    fn test_empty_match_set() {
        let empty: Vec<SIFTDescriptor> = Vec::new();
        let q: Vec<SIFTDescriptor> = (0..3).map(|i| make_sift_desc(i, 0.0)).collect();
        let m1 = match_descriptors(&empty, &q, &MatchMethod::BruteForce)
            .expect("match_descriptors should succeed with empty query");
        let m2 = match_descriptors(&q, &empty, &MatchMethod::BruteForce)
            .expect("match_descriptors should succeed with empty target");
        assert!(m1.is_empty());
        assert!(m2.is_empty());
    }

    #[test]
    fn test_symmetric_filter() {
        let descs: Vec<SIFTDescriptor> = (0..6).map(|i| make_sift_desc(i, 0.0)).collect();
        let sym = symmetric_filter(&descs, &descs, &MatchMethod::BruteForce)
            .expect("symmetric_filter should succeed");
        // All self-matches are symmetric
        for (i, j, d) in &sym {
            assert_eq!(i, j);
            assert!(*d < 1e-6);
        }
    }
}
