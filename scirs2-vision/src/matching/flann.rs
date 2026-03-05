//! FLANN-style approximate nearest-neighbour matching.
//!
//! Provides a KD-tree based approximate matcher for float descriptors, which
//! is significantly faster than brute force for large descriptor sets.
//!
//! Note: this is a simplified KD-tree implementation optimised for descriptor
//! matching (fixed-dim float vectors).  For a full FLANN implementation with
//! randomised KD-forests and LSH tables, a dedicated library would be needed.

use crate::error::VisionError;

// ─── KD-tree ─────────────────────────────────────────────────────────────────

/// A node in the KD-tree.
#[derive(Debug)]
enum KDNode {
    Leaf {
        indices: Vec<usize>,
    },
    Internal {
        axis: usize,
        threshold: f32,
        left: Box<KDNode>,
        right: Box<KDNode>,
    },
}

/// Build a KD-tree over `data[indices]`, splitting along the axis with maximum
/// variance.  Leaf nodes hold at most `leaf_size` points.
fn build_kdtree(data: &[Vec<f32>], indices: Vec<usize>, leaf_size: usize, depth: usize) -> KDNode {
    if indices.len() <= leaf_size || depth > 32 {
        return KDNode::Leaf { indices };
    }

    let dim = data[0].len();
    if dim == 0 {
        return KDNode::Leaf { indices };
    }

    // Choose split axis: axis with maximum variance.
    let n = indices.len() as f64;
    let axis = (0..dim)
        .max_by_key(|&d| {
            let mean = indices.iter().map(|&i| data[i][d] as f64).sum::<f64>() / n;
            let var = indices
                .iter()
                .map(|&i| (data[i][d] as f64 - mean).powi(2))
                .sum::<f64>();
            ordered_float::NotNan::new(var).unwrap_or_else(|_| ordered_float::NotNan::default())
        })
        .unwrap_or(depth % dim);

    // Split at median.
    let mut vals: Vec<f32> = indices.iter().map(|&i| data[i][axis]).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = vals[vals.len() / 2];

    let (left_idx, right_idx): (Vec<usize>, Vec<usize>) =
        indices.into_iter().partition(|&i| data[i][axis] < median);

    // Prevent infinite recursion if all values equal.
    if left_idx.is_empty() || right_idx.is_empty() {
        let (l, r) = if left_idx.is_empty() {
            right_idx.split_at(right_idx.len() / 2)
        } else {
            left_idx.split_at(left_idx.len() / 2)
        };
        let (lv, rv) = (l.to_vec(), r.to_vec());
        return KDNode::Internal {
            axis,
            threshold: median,
            left: Box::new(build_kdtree(data, lv, leaf_size, depth + 1)),
            right: Box::new(build_kdtree(data, rv, leaf_size, depth + 1)),
        };
    }

    KDNode::Internal {
        axis,
        threshold: median,
        left: Box::new(build_kdtree(data, left_idx, leaf_size, depth + 1)),
        right: Box::new(build_kdtree(data, right_idx, leaf_size, depth + 1)),
    }
}

/// Find the `k` nearest neighbours of `query` in the KD-tree.
/// Returns `(index, squared_distance)` pairs.
fn knn_search(
    node: &KDNode,
    data: &[Vec<f32>],
    query: &[f32],
    k: usize,
    heap: &mut Vec<(usize, f32)>,
) {
    match node {
        KDNode::Leaf { indices } => {
            for &idx in indices {
                let d2: f32 = query
                    .iter()
                    .zip(data[idx].iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                if heap.len() < k {
                    heap.push((idx, d2));
                    // Keep as max-heap (largest distance first; for equal distances, larger index first).
                    if heap.len() == k {
                        heap.sort_by(|a, b| {
                            match b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal) {
                                std::cmp::Ordering::Equal => b.0.cmp(&a.0),
                                other => other,
                            }
                        });
                    }
                } else if d2 < heap[0].1 || (d2 == heap[0].1 && idx < heap[0].0) {
                    heap[0] = (idx, d2);
                    heap.sort_by(|a, b| {
                        match b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal) {
                            std::cmp::Ordering::Equal => b.0.cmp(&a.0),
                            other => other,
                        }
                    });
                }
            }
        }
        KDNode::Internal {
            axis,
            threshold,
            left,
            right,
        } => {
            let diff = query[*axis] - threshold;
            let (near, far) = if diff < 0.0 {
                (left.as_ref(), right.as_ref())
            } else {
                (right.as_ref(), left.as_ref())
            };
            knn_search(near, data, query, k, heap);
            // Prune: only explore far branch if it could contain closer point.
            let worst = heap.first().map_or(f32::INFINITY, |h| h.1);
            if heap.len() < k || diff * diff < worst {
                knn_search(far, data, query, k, heap);
            }
        }
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// FLANN-style approximate KD-tree matcher.
pub struct FlannMatcher {
    tree: KDNode,
    data: Vec<Vec<f32>>,
}

impl FlannMatcher {
    /// Build the index from a set of float descriptors.
    ///
    /// # Errors
    /// Returns [`VisionError`] if `descriptors` is empty.
    pub fn build(descriptors: Vec<Vec<f32>>) -> Result<Self, VisionError> {
        if descriptors.is_empty() {
            return Err(VisionError::InvalidInput(
                "Empty descriptor set for FLANN".into(),
            ));
        }
        let indices: Vec<usize> = (0..descriptors.len()).collect();
        let tree = build_kdtree(&descriptors, indices, 10, 0);
        Ok(Self {
            tree,
            data: descriptors,
        })
    }

    /// Find the `k` nearest neighbours of `query` in the index.
    ///
    /// Returns `(index, l2_distance)` pairs sorted by ascending distance.
    ///
    /// # Errors
    /// Returns [`VisionError`] if `query` has a different dimension than the
    /// indexed descriptors.
    pub fn knn_match(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, VisionError> {
        if query.len() != self.data[0].len() {
            return Err(VisionError::InvalidInput(
                "Query dimension mismatch in FLANN knn_match".into(),
            ));
        }
        let mut heap: Vec<(usize, f32)> = Vec::with_capacity(k);
        knn_search(&self.tree, &self.data, query, k, &mut heap);
        heap.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        // Return L2 (not squared).
        let result = heap.into_iter().map(|(i, d2)| (i, d2.sqrt())).collect();
        Ok(result)
    }

    /// Match all descriptors in `queries` against the index using ratio test.
    ///
    /// Returns `(query_idx, match_idx, distance)` triples passing the ratio test.
    ///
    /// # Errors
    /// Propagates errors from `knn_match`.
    pub fn match_with_ratio(
        &self,
        queries: &[Vec<f32>],
        ratio: f32,
    ) -> Result<Vec<(usize, usize, f32)>, VisionError> {
        let mut results = Vec::new();
        for (qi, query) in queries.iter().enumerate() {
            let nn = self.knn_match(query, 2)?;
            if nn.len() < 2 {
                if let Some(&(mi, dist)) = nn.first() {
                    results.push((qi, mi, dist));
                }
                continue;
            }
            if nn[0].1 < ratio * nn[1].1 {
                results.push((qi, nn[0].0, nn[0].1));
            }
        }
        results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_descs(n: usize, dim: usize) -> Vec<Vec<f32>> {
        // Generate descriptors that are unique (no duplicates even when n > dim)
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| {
                        // Use a combination that makes each descriptor unique
                        let base = if d == (i % dim) { 1.0_f32 } else { 0.0_f32 };
                        base + (i as f32 * 0.01) + (d as f32 * 0.001)
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_flann_build_and_query() {
        let data = unique_descs(20, 8);
        let matcher = FlannMatcher::build(data.clone()).expect("build FLANN");
        let nn = matcher.knn_match(&data[3], 1).expect("knn_match");
        assert!(!nn.is_empty());
        assert_eq!(nn[0].0, 3); // Should find self.
        assert!(nn[0].1 < 1e-4);
    }

    #[test]
    fn test_flann_knn_returns_k() {
        let data: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![(i as f32) / 50.0, 0.0, 0.0, 0.0])
            .collect();
        let matcher = FlannMatcher::build(data.clone())
            .expect("FlannMatcher::build should succeed with valid data");
        let nn = matcher
            .knn_match(&data[10], 5)
            .expect("knn_match should succeed");
        assert!(nn.len() <= 5);
    }

    #[test]
    fn test_flann_ratio_match() {
        let data: Vec<Vec<f32>> = (0..30).map(|i| vec![(i as f32) / 30.0; 4]).collect();
        let matcher = FlannMatcher::build(data.clone())
            .expect("FlannMatcher::build should succeed with valid data");
        let matches = matcher
            .match_with_ratio(&data, 0.8)
            .expect("match_with_ratio should succeed");
        // At least some matches should survive.
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_flann_empty_error() {
        assert!(FlannMatcher::build(vec![]).is_err());
    }

    #[test]
    fn test_flann_dim_mismatch() {
        let data = vec![vec![1.0_f32; 4]];
        let matcher = FlannMatcher::build(data)
            .expect("FlannMatcher::build should succeed with single-point data");
        assert!(matcher.knn_match(&[1.0_f32, 0.0], 1).is_err());
    }
}
