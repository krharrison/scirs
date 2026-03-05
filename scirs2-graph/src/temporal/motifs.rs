//! Temporal motif counting
//!
//! Counts occurrences of small time-respecting subgraph patterns (motifs) in
//! temporal graphs.  The algorithms are based on the δ-temporal motif framework
//! from Paranjape et al. (2017), which requires that all edges of a motif occur
//! within a time window of width δ.
//!
//! # Supported patterns
//!
//! We enumerate all **3-node, 3-edge** undirected temporal motifs (8 types) as
//! defined in the directed version by collapsing edge directions:
//!
//! | ID | Name            | Description                                 |
//! |----|-----------------|---------------------------------------------|
//! | M1 | TriangleFwd     | u→v, v→w, u→w (closing triangle)           |
//! | M2 | TriangleBck     | u→v, w→v, u→w (shared target, then close)  |
//! | M3 | StarOut         | u→v, u→w, v→w (star with closing edge)     |
//! | M4 | StarIn          | v→u, w→u, v→w (shared source)              |
//! | M5 | PathFwd         | u→v, v→w (open 2-hop path)                 |
//! | M6 | PathBck         | v→u, v→w (diverging from hub)              |
//! | M7 | PathIn          | u→v, w→v (converging to hub)               |
//! | M8 | PathMix         | u→v, w→u (chain reversed)                 |
//!
//! # References
//! - Paranjape, A., Benson, A. R., & Leskovec, J. (2017).
//!   Motifs in temporal networks. WSDM 2017.

use super::graph::TemporalGraph;

// ─────────────────────────────────────────────────────────────────────────────
// TemporalMotifCounts
// ─────────────────────────────────────────────────────────────────────────────

/// Counts for each of the 8 types of 3-node, 3-edge temporal motifs.
#[derive(Debug, Clone, Default)]
pub struct TemporalMotifCounts {
    /// M1: Forward triangle  u→v, v→w, u→w  (3-clique, edges in "closing" order)
    pub m1_triangle_fwd: usize,
    /// M2: Backward triangle  u→v, w→v, u→w  (two paths to v, then close)
    pub m2_triangle_bck: usize,
    /// M3: Out-star triangle  u→v, u→w, v→w  (hub fans out, leaves close)
    pub m3_star_out: usize,
    /// M4: In-star triangle  v→u, w→u, v→w  (hub collects, leaves close)
    pub m4_star_in: usize,
    /// M5: Forward path  u→v, v→w  (temporal 2-hop)
    pub m5_path_fwd: usize,
    /// M6: Diverge path  v→u, v→w  (hub fans out in time)
    pub m6_path_bck: usize,
    /// M7: Converge path  u→v, w→v  (two sources arrive at hub)
    pub m7_path_in: usize,
    /// M8: Mixed path  u→v, w→u  (chain reversed)
    pub m8_path_mix: usize,
}

impl TemporalMotifCounts {
    /// Total number of motif instances counted.
    pub fn total(&self) -> usize {
        self.m1_triangle_fwd
            + self.m2_triangle_bck
            + self.m3_star_out
            + self.m4_star_in
            + self.m5_path_fwd
            + self.m6_path_bck
            + self.m7_path_in
            + self.m8_path_mix
    }

    /// Number of triangle motifs (M1–M4).
    pub fn triangle_count(&self) -> usize {
        self.m1_triangle_fwd + self.m2_triangle_bck + self.m3_star_out + self.m4_star_in
    }

    /// Number of path motifs (M5–M8).
    pub fn path_count(&self) -> usize {
        self.m5_path_fwd + self.m6_path_bck + self.m7_path_in + self.m8_path_mix
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// count_temporal_triangles
// ─────────────────────────────────────────────────────────────────────────────

/// Count temporal triangles: triples of edges `(e1, e2, e3)` that form a
/// triangle on 3 distinct nodes and all occur within a time window of `delta`.
///
/// An ordered triple is considered a temporal triangle when:
/// - The three edges form a closed triangle on distinct nodes u, v, w.
/// - All three timestamps satisfy `t3 - t1 ≤ delta` (time-window constraint).
/// - The timestamps are non-decreasing: `t1 ≤ t2 ≤ t3`.
///
/// # Arguments
/// * `tg`    – mutable reference to the temporal graph
/// * `delta` – maximum duration of a temporal motif window
pub fn count_temporal_triangles(tg: &mut TemporalGraph, delta: f64) -> usize {
    tg.ensure_sorted();
    let edges = tg.edges.clone(); // avoid borrow issues
    let m = edges.len();
    let n = tg.nodes;

    // Build adjacency set for quick triangle closure check
    // adj_set[u] = set of v such that {u,v} appeared in *any* edge (for triangle test)
    let mut adj_set: Vec<std::collections::HashSet<usize>> =
        vec![std::collections::HashSet::new(); n];
    for e in &edges {
        adj_set[e.source].insert(e.target);
        adj_set[e.target].insert(e.source);
    }

    let mut count = 0usize;

    // Enumerate all ordered triples (i, j, k) with ti ≤ tj ≤ tk ≤ ti + delta
    for i in 0..m {
        let e1 = &edges[i];
        // e2 must have timestamp in [e1.timestamp, e1.timestamp + delta]
        for j in (i + 1)..m {
            let e2 = &edges[j];
            if e2.timestamp > e1.timestamp + delta {
                break; // edges are sorted; no point going further
            }
            // e3 must have timestamp in [e2.timestamp, e1.timestamp + delta]
            for k in (j + 1)..m {
                let e3 = &edges[k];
                if e3.timestamp > e1.timestamp + delta {
                    break;
                }

                // Check that all three edges form a triangle on 3 distinct nodes
                let nodes_used = distinct_triangle_nodes(
                    e1.source, e1.target, e2.source, e2.target, e3.source, e3.target,
                );

                if let Some((u, v, w)) = nodes_used {
                    // Verify all three edges of the triangle are present:
                    // e1 connects two of {u,v,w}, e2 connects two others, e3 the last pair
                    let _ = (u, v, w); // already validated by distinct_triangle_nodes
                    count += 1;
                }
            }
        }
    }

    count
}

/// Check whether the six endpoints form a triangle on exactly 3 distinct nodes.
/// Returns `Some((a, b, c))` if yes, `None` otherwise.
fn distinct_triangle_nodes(
    s1: usize,
    t1: usize,
    s2: usize,
    t2: usize,
    s3: usize,
    t3: usize,
) -> Option<(usize, usize, usize)> {
    // Collect all involved nodes (each edge contributes a pair)
    let mut nodes = [s1, t1, s2, t2, s3, t3];
    nodes.sort_unstable();

    // We need exactly 3 distinct nodes
    let mut distinct: Vec<usize> = Vec::with_capacity(3);
    for &n in &nodes {
        if distinct.last() != Some(&n) {
            distinct.push(n);
        }
    }
    if distinct.len() != 3 {
        return None;
    }
    let (a, b, c) = (distinct[0], distinct[1], distinct[2]);

    // Check that each of the 3 edges connects exactly two of {a, b, c}
    let pair_ab = (s1 == a && t1 == b) || (s1 == b && t1 == a);
    let pair_ac = (s1 == a && t1 == c) || (s1 == c && t1 == a);
    let pair_bc = (s1 == b && t1 == c) || (s1 == c && t1 == b);

    let e1_ok = pair_ab || pair_ac || pair_bc;
    if !e1_ok {
        return None;
    }

    let pair2_ab = (s2 == a && t2 == b) || (s2 == b && t2 == a);
    let pair2_ac = (s2 == a && t2 == c) || (s2 == c && t2 == a);
    let pair2_bc = (s2 == b && t2 == c) || (s2 == c && t2 == b);
    let e2_ok = pair2_ab || pair2_ac || pair2_bc;
    if !e2_ok {
        return None;
    }

    let pair3_ab = (s3 == a && t3 == b) || (s3 == b && t3 == a);
    let pair3_ac = (s3 == a && t3 == c) || (s3 == c && t3 == a);
    let pair3_bc = (s3 == b && t3 == c) || (s3 == c && t3 == b);
    let e3_ok = pair3_ab || pair3_ac || pair3_bc;
    if !e3_ok {
        return None;
    }

    // Verify all three *pairs* of nodes are covered (no duplicate edge)
    let pairs = [
        ordered_pair(s1, t1),
        ordered_pair(s2, t2),
        ordered_pair(s3, t3),
    ];
    let p_ab = ordered_pair(a, b);
    let p_ac = ordered_pair(a, c);
    let p_bc = ordered_pair(b, c);

    let covers_ab = pairs.contains(&p_ab);
    let covers_ac = pairs.contains(&p_ac);
    let covers_bc = pairs.contains(&p_bc);

    if covers_ab && covers_ac && covers_bc {
        Some((a, b, c))
    } else {
        None
    }
}

#[inline]
fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

// ─────────────────────────────────────────────────────────────────────────────
// temporal_motif_count  (all 8 types)
// ─────────────────────────────────────────────────────────────────────────────

/// Count all 8 types of 3-node, 3-edge temporal motifs within time window `delta`.
///
/// The algorithm enumerates all ordered triples of edges whose timestamps span
/// at most `delta` and classifies each triple by its topological pattern.
///
/// Complexity: O(m³) in the worst case, but in practice fast for sparse graphs
/// with small `delta` due to early termination on the time window constraint.
///
/// # Arguments
/// * `tg`    – mutable reference to the temporal graph
/// * `delta` – maximum duration of the motif time window
pub fn temporal_motif_count(tg: &mut TemporalGraph, delta: f64) -> TemporalMotifCounts {
    tg.ensure_sorted();
    let edges = tg.edges.clone();
    let m = edges.len();

    let mut counts = TemporalMotifCounts::default();

    for i in 0..m {
        let e1 = &edges[i];
        for j in (i + 1)..m {
            let e2 = &edges[j];
            if e2.timestamp > e1.timestamp + delta {
                break;
            }
            // Check 2-edge patterns (open paths / stars) at each step
            classify_two_edge_motif(e1.source, e1.target, e2.source, e2.target, &mut counts);

            for k in (j + 1)..m {
                let e3 = &edges[k];
                if e3.timestamp > e1.timestamp + delta {
                    break;
                }
                // Check 3-edge patterns (triangles)
                classify_three_edge_motif(
                    e1.source,
                    e1.target,
                    e2.source,
                    e2.target,
                    e3.source,
                    e3.target,
                    &mut counts,
                );
            }
        }
    }

    counts
}

/// Classify a 2-edge motif and increment the appropriate counter.
/// This counts path motifs M5–M8.
fn classify_two_edge_motif(
    s1: usize,
    t1: usize,
    s2: usize,
    t2: usize,
    counts: &mut TemporalMotifCounts,
) {
    // M5: forward path  e1 = (u→v), e2 = (v→w):  t1 < t2, shared node v is target of e1, source of e2
    if t1 == s2 && s1 != t2 {
        counts.m5_path_fwd += 1;
        return;
    }
    // M8: mixed path  e1 = (u→v), e2 = (w→u):  t1 < t2, shared node u is source of e1, target of e2
    if s1 == t2 && t1 != s2 {
        counts.m8_path_mix += 1;
        return;
    }
    // M6: diverge path  e1 = (v→u), e2 = (v→w): shared node v is source of both
    if s1 == s2 && t1 != t2 {
        counts.m6_path_bck += 1;
        return;
    }
    // M7: converge path  e1 = (u→v), e2 = (w→v): shared node v is target of both
    if t1 == t2 && s1 != s2 {
        counts.m7_path_in += 1;
    }
}

/// Classify a 3-edge triangle motif and increment the appropriate counter.
fn classify_three_edge_motif(
    s1: usize,
    t1: usize,
    s2: usize,
    t2: usize,
    s3: usize,
    t3: usize,
    counts: &mut TemporalMotifCounts,
) {
    if distinct_triangle_nodes(s1, t1, s2, t2, s3, t3).is_none() {
        return;
    }

    // Determine the 3 distinct nodes
    let mut node_set = [s1, t1, s2, t2, s3, t3];
    node_set.sort_unstable();
    let mut nodes: Vec<usize> = node_set.to_vec();
    nodes.dedup();
    if nodes.len() != 3 {
        return;
    }

    // M1: u→v at t1, v→w at t2, u→w at t3
    if t1 == s2 && s1 == s3 && t2 == t3 {
        counts.m1_triangle_fwd += 1;
        return;
    }
    // M2: u→v at t1, w→v at t2, u→w at t3
    if t1 == t2 && s1 == s3 && s2 == t3 {
        counts.m2_triangle_bck += 1;
        return;
    }
    // M3: u→v at t1, u→w at t2, v→w at t3
    if s1 == s2 && t1 == s3 && t2 == t3 {
        counts.m3_star_out += 1;
        return;
    }
    // M4: v→u at t1, w→u at t2, v→w at t3
    if t1 == t2 && s1 == s3 && s2 == t3 {
        counts.m4_star_in += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::graph::TemporalEdge;
    use super::*;

    fn make_triangle_fwd() -> TemporalGraph {
        // u=0, v=1, w=2
        // M1: 0→1 at t=1, 1→2 at t=2, 0→2 at t=3
        let mut tg = TemporalGraph::new(3);
        tg.add_edge(TemporalEdge::new(0, 1, 1.0));
        tg.add_edge(TemporalEdge::new(1, 2, 2.0));
        tg.add_edge(TemporalEdge::new(0, 2, 3.0));
        tg
    }

    fn make_path() -> TemporalGraph {
        // M5: 0→1 at t=1, 1→2 at t=2
        let mut tg = TemporalGraph::new(3);
        tg.add_edge(TemporalEdge::new(0, 1, 1.0));
        tg.add_edge(TemporalEdge::new(1, 2, 2.0));
        tg
    }

    #[test]
    fn test_count_temporal_triangles_basic() {
        let mut tg = make_triangle_fwd();
        let count = count_temporal_triangles(&mut tg, 5.0);
        assert!(count >= 1, "should find at least one temporal triangle");
    }

    #[test]
    fn test_count_temporal_triangles_no_match() {
        let mut tg = make_path();
        let count = count_temporal_triangles(&mut tg, 5.0);
        assert_eq!(count, 0, "open path has no triangle");
    }

    #[test]
    fn test_count_temporal_triangles_delta_too_small() {
        let mut tg = make_triangle_fwd();
        // With delta=0.5 the 3 edges at t=1,2,3 don't all fit
        let count = count_temporal_triangles(&mut tg, 0.5);
        assert_eq!(count, 0, "edges spread > 0.5 should give no triangle");
    }

    #[test]
    fn test_temporal_motif_count_path() {
        let mut tg = make_path();
        let counts = temporal_motif_count(&mut tg, 5.0);
        // Should detect M5 (forward path 0→1→2)
        assert!(
            counts.m5_path_fwd >= 1,
            "expected at least one M5 forward path, got {}",
            counts.m5_path_fwd
        );
    }

    #[test]
    fn test_temporal_motif_count_total_is_sum() {
        let mut tg = make_triangle_fwd();
        let counts = temporal_motif_count(&mut tg, 5.0);
        assert_eq!(
            counts.total(),
            counts.triangle_count() + counts.path_count(),
            "total should equal triangle + path counts"
        );
    }

    #[test]
    fn test_temporal_motif_count_no_edges() {
        let mut tg = TemporalGraph::new(5);
        let counts = temporal_motif_count(&mut tg, 10.0);
        assert_eq!(counts.total(), 0);
    }

    #[test]
    fn test_motif_counts_struct() {
        let c = TemporalMotifCounts {
            m1_triangle_fwd: 2,
            m5_path_fwd: 3,
            ..Default::default()
        };
        assert_eq!(c.total(), 5);
        assert_eq!(c.triangle_count(), 2);
        assert_eq!(c.path_count(), 3);
    }
}
