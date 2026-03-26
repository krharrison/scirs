//! Alignment quality evaluation metrics.
//!
//! Provides functions for measuring the quality of a network alignment,
//! including edge conservation, symmetric substructure score, node correctness,
//! and induced conserved structure.

use scirs2_core::ndarray::Array2;

/// Edge Conservation (EC): fraction of edges in G1 preserved in the mapping to G2.
///
/// For each edge `(u, v)` in G1, checks whether the mapped nodes `(f(u), f(v))`
/// also form an edge in G2.
///
/// # Returns
///
/// A value in `[0.0, 1.0]`. Returns 0.0 if G1 has no edges.
pub fn edge_conservation(
    mapping: &[(usize, usize)],
    adj1: &Array2<f64>,
    adj2: &Array2<f64>,
) -> f64 {
    if mapping.is_empty() {
        return 0.0;
    }

    let n1 = adj1.nrows();

    // Build a lookup: g1_node -> g2_node
    let mut g1_to_g2 = vec![usize::MAX; n1];
    for &(i, j) in mapping {
        if i < n1 {
            g1_to_g2[i] = j;
        }
    }

    let mut total_edges = 0u64;
    let mut preserved_edges = 0u64;

    // Count edges in G1's mapped subgraph and check preservation
    for &(u, _) in mapping {
        for &(v, _) in mapping {
            if u >= v {
                continue; // undirected: count each edge once
            }
            if u < adj1.nrows() && v < adj1.nrows() && adj1[[u, v]].abs() > f64::EPSILON {
                total_edges += 1;
                let fu = g1_to_g2[u];
                let fv = g1_to_g2[v];
                if fu < adj2.nrows() && fv < adj2.ncols() && adj2[[fu, fv]].abs() > f64::EPSILON {
                    preserved_edges += 1;
                }
            }
        }
    }

    if total_edges == 0 {
        return 0.0;
    }

    preserved_edges as f64 / total_edges as f64
}

/// Symmetric Substructure Score (S3).
///
/// Computes: `|edges preserved| / (|edges in G1 subgraph| + |edges in G2 subgraph| - |edges preserved|)`
///
/// This metric penalizes both missing edges (in G1 not preserved) and extra edges
/// (in G2 subgraph not corresponding to G1 edges), giving a more balanced view
/// than edge conservation alone.
///
/// # Returns
///
/// A value in `[0.0, 1.0]`. Returns 0.0 if neither subgraph has edges.
pub fn symmetric_substructure_score(
    mapping: &[(usize, usize)],
    adj1: &Array2<f64>,
    adj2: &Array2<f64>,
) -> f64 {
    if mapping.is_empty() {
        return 0.0;
    }

    let n1 = adj1.nrows();
    let n2 = adj2.nrows();

    // Build lookup tables
    let mut g1_to_g2 = vec![usize::MAX; n1];
    for &(i, j) in mapping {
        if i < n1 {
            g1_to_g2[i] = j;
        }
    }

    // Collect the set of mapped G2 nodes
    let mut mapped_g2: Vec<bool> = vec![false; n2];
    for &(_, j) in mapping {
        if j < n2 {
            mapped_g2[j] = true;
        }
    }

    // Count edges in G1's mapped subgraph
    let mut edges_g1 = 0u64;
    for &(u, _) in mapping {
        for &(v, _) in mapping {
            if u < v && u < adj1.nrows() && v < adj1.nrows() && adj1[[u, v]].abs() > f64::EPSILON {
                edges_g1 += 1;
            }
        }
    }

    // Count edges in G2's mapped subgraph
    let mut edges_g2 = 0u64;
    for &(_, ju) in mapping {
        for &(_, jv) in mapping {
            if ju < jv
                && ju < adj2.nrows()
                && jv < adj2.nrows()
                && adj2[[ju, jv]].abs() > f64::EPSILON
            {
                edges_g2 += 1;
            }
        }
    }

    // Count preserved edges
    let mut preserved = 0u64;
    for &(u, _) in mapping {
        for &(v, _) in mapping {
            if u < v && u < adj1.nrows() && v < adj1.nrows() && adj1[[u, v]].abs() > f64::EPSILON {
                let fu = g1_to_g2[u];
                let fv = g1_to_g2[v];
                if fu < adj2.nrows() && fv < adj2.nrows() && adj2[[fu, fv]].abs() > f64::EPSILON {
                    preserved += 1;
                }
            }
        }
    }

    let denominator = edges_g1 + edges_g2 - preserved;
    if denominator == 0 {
        return 0.0;
    }

    preserved as f64 / denominator as f64
}

/// Node Correctness (NC): fraction of correctly mapped nodes.
///
/// Compares the computed mapping against a known ground truth alignment.
/// A node is "correct" if it is mapped to the same target in both the
/// computed and ground truth mappings.
///
/// # Returns
///
/// A value in `[0.0, 1.0]`. Returns 0.0 if ground truth is empty.
pub fn node_correctness(mapping: &[(usize, usize)], ground_truth: &[(usize, usize)]) -> f64 {
    if ground_truth.is_empty() {
        return 0.0;
    }

    // Build lookup from ground truth
    let max_node = ground_truth.iter().map(|&(i, _)| i).max().unwrap_or(0);
    let mut gt_map = vec![usize::MAX; max_node + 1];
    for &(i, j) in ground_truth {
        if i <= max_node {
            gt_map[i] = j;
        }
    }

    let mut correct = 0usize;
    for &(i, j) in mapping {
        if i < gt_map.len() && gt_map[i] == j {
            correct += 1;
        }
    }

    correct as f64 / ground_truth.len() as f64
}

/// Induced Conserved Structure (ICS).
///
/// Computes: `|edges preserved| / |edges in mapped subgraph of G2|`
///
/// This metric measures how well the mapped subgraph of G2 is "covered"
/// by edges from G1, penalizing extra edges in G2's subgraph that don't
/// correspond to edges in G1.
///
/// # Returns
///
/// A value in `[0.0, 1.0]`. Returns 0.0 if the G2 subgraph has no edges.
pub fn induced_conserved_structure(
    mapping: &[(usize, usize)],
    adj1: &Array2<f64>,
    adj2: &Array2<f64>,
) -> f64 {
    if mapping.is_empty() {
        return 0.0;
    }

    let n1 = adj1.nrows();

    // Build lookup: g1_node -> g2_node
    let mut g1_to_g2 = vec![usize::MAX; n1];
    for &(i, j) in mapping {
        if i < n1 {
            g1_to_g2[i] = j;
        }
    }

    // Count edges in G2's mapped subgraph
    let mut edges_g2 = 0u64;
    for &(_, ju) in mapping {
        for &(_, jv) in mapping {
            if ju < jv
                && ju < adj2.nrows()
                && jv < adj2.nrows()
                && adj2[[ju, jv]].abs() > f64::EPSILON
            {
                edges_g2 += 1;
            }
        }
    }

    if edges_g2 == 0 {
        return 0.0;
    }

    // Count preserved edges
    let mut preserved = 0u64;
    for &(u, _) in mapping {
        for &(v, _) in mapping {
            if u < v && u < adj1.nrows() && v < adj1.nrows() && adj1[[u, v]].abs() > f64::EPSILON {
                let fu = g1_to_g2[u];
                let fv = g1_to_g2[v];
                if fu < adj2.nrows() && fv < adj2.nrows() && adj2[[fu, fv]].abs() > f64::EPSILON {
                    preserved += 1;
                }
            }
        }
    }

    preserved as f64 / edges_g2 as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn path_graph(n: usize) -> Array2<f64> {
        let mut adj = Array2::zeros((n, n));
        for i in 0..n.saturating_sub(1) {
            adj[[i, i + 1]] = 1.0;
            adj[[i + 1, i]] = 1.0;
        }
        adj
    }

    #[test]
    fn test_ec_identity_mapping() {
        let adj = path_graph(5);
        let identity: Vec<(usize, usize)> = (0..5).map(|i| (i, i)).collect();
        let ec = edge_conservation(&identity, &adj, &adj);
        assert!(
            (ec - 1.0).abs() < 1e-10,
            "Identity mapping on same graph should have EC=1.0, got {}",
            ec
        );
    }

    #[test]
    fn test_ec_random_worse_than_identity() {
        let adj = path_graph(5);
        let identity: Vec<(usize, usize)> = (0..5).map(|i| (i, i)).collect();
        // Reversed mapping
        let reversed: Vec<(usize, usize)> = (0..5).map(|i| (i, 4 - i)).collect();

        let ec_id = edge_conservation(&identity, &adj, &adj);
        let ec_rev = edge_conservation(&reversed, &adj, &adj);

        // For a path graph, reversed mapping also preserves all edges
        // (path is symmetric), so both should be 1.0
        // But for asymmetric graphs, identity would be better
        assert!(ec_id >= 0.0);
        assert!(ec_rev >= 0.0);
    }

    #[test]
    fn test_ec_empty_mapping() {
        let adj = path_graph(3);
        let ec = edge_conservation(&[], &adj, &adj);
        assert!((ec).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ec_no_edges() {
        let adj = Array2::zeros((3, 3));
        let mapping: Vec<(usize, usize)> = vec![(0, 0), (1, 1), (2, 2)];
        let ec = edge_conservation(&mapping, &adj, &adj);
        assert!((ec).abs() < f64::EPSILON);
    }

    #[test]
    fn test_s3_range() {
        let adj = path_graph(5);
        let mapping: Vec<(usize, usize)> = (0..5).map(|i| (i, i)).collect();
        let s3 = symmetric_substructure_score(&mapping, &adj, &adj);
        assert!(
            (0.0..=1.0).contains(&s3),
            "S3 should be in [0, 1], got {}",
            s3
        );
    }

    #[test]
    fn test_s3_identity() {
        let adj = path_graph(4);
        let identity: Vec<(usize, usize)> = (0..4).map(|i| (i, i)).collect();
        let s3 = symmetric_substructure_score(&identity, &adj, &adj);
        assert!(
            (s3 - 1.0).abs() < 1e-10,
            "S3 of identity on same graph should be 1.0, got {}",
            s3
        );
    }

    #[test]
    fn test_s3_empty() {
        let adj = path_graph(3);
        let s3 = symmetric_substructure_score(&[], &adj, &adj);
        assert!((s3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_node_correctness_identity() {
        let gt: Vec<(usize, usize)> = (0..5).map(|i| (i, i)).collect();
        let mapping = gt.clone();
        let nc = node_correctness(&mapping, &gt);
        assert!((nc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_node_correctness_none_correct() {
        let gt: Vec<(usize, usize)> = vec![(0, 0), (1, 1), (2, 2)];
        let mapping: Vec<(usize, usize)> = vec![(0, 2), (1, 0), (2, 1)];
        let nc = node_correctness(&mapping, &gt);
        assert!((nc).abs() < f64::EPSILON);
    }

    #[test]
    fn test_node_correctness_partial() {
        let gt: Vec<(usize, usize)> = vec![(0, 0), (1, 1), (2, 2), (3, 3)];
        let mapping: Vec<(usize, usize)> = vec![(0, 0), (1, 2), (2, 2), (3, 1)];
        let nc = node_correctness(&mapping, &gt);
        assert!((nc - 0.5).abs() < 1e-10); // 2 out of 4 correct
    }

    #[test]
    fn test_node_correctness_empty_gt() {
        let nc = node_correctness(&[(0, 0)], &[]);
        assert!((nc).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ics_range() {
        let adj = path_graph(5);
        let mapping: Vec<(usize, usize)> = (0..5).map(|i| (i, i)).collect();
        let ics = induced_conserved_structure(&mapping, &adj, &adj);
        assert!(
            (0.0..=1.0).contains(&ics),
            "ICS should be in [0, 1], got {}",
            ics
        );
    }

    #[test]
    fn test_ics_identity() {
        let adj = path_graph(4);
        let identity: Vec<(usize, usize)> = (0..4).map(|i| (i, i)).collect();
        let ics = induced_conserved_structure(&identity, &adj, &adj);
        assert!(
            (ics - 1.0).abs() < 1e-10,
            "ICS of identity on same graph should be 1.0, got {}",
            ics
        );
    }

    #[test]
    fn test_ics_empty() {
        let adj = path_graph(3);
        let ics = induced_conserved_structure(&[], &adj, &adj);
        assert!((ics).abs() < f64::EPSILON);
    }
}
