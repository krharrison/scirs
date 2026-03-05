//! Projective dependency parsing via the Eisner algorithm (O(n³)).
//!
//! Also provides Chu-Liu/Edmonds for maximum spanning arborescence
//! (non-projective).
//!
//! # References
//!
//! * Eisner, J. (1996). Three new probabilistic models for dependency parsing.
//! * Chu & Liu (1965); Edmonds (1967) – maximum spanning arborescence.

use super::graph::{DepLabel, DependencyGraph};

// ---------------------------------------------------------------------------
// Score matrix
// ---------------------------------------------------------------------------

/// Dense score matrix of size `n × n` (ROOT at index 0, tokens at 1..n).
///
/// `scores[h][d]` is the score for the arc `h → d`.
#[derive(Debug, Clone)]
pub struct ScoreMatrix {
    /// Dense `n x n` matrix where `scores[h][d]` is the score for arc `h -> d`.
    pub scores: Vec<Vec<f64>>,
    /// Total size including the virtual ROOT.
    pub n: usize,
}

impl ScoreMatrix {
    /// Create an all-`-∞` matrix of size `n × n`.
    pub fn new(n: usize) -> Self {
        Self {
            scores: vec![vec![f64::NEG_INFINITY; n]; n],
            n,
        }
    }

    /// Set the score for arc `head → dep`.
    pub fn set(&mut self, head: usize, dep: usize, score: f64) {
        if head < self.n && dep < self.n {
            self.scores[head][dep] = score;
        }
    }

    /// Build a distance-based heuristic score matrix for `n_tokens` real tokens.
    ///
    /// Score is `-ln(|h - d|)` — closer attachments are preferred.
    pub fn from_distance_heuristic(n_tokens: usize) -> Self {
        let n = n_tokens + 1; // index 0 = ROOT
        let mut m = Self::new(n);
        for h in 0..n {
            for d in 1..n {
                if h == d {
                    continue;
                }
                let dist = (h as f64 - d as f64).abs();
                m.scores[h][d] = -(dist.max(1.0).ln());
            }
        }
        m
    }
}

// ---------------------------------------------------------------------------
// Eisner algorithm
// ---------------------------------------------------------------------------

/// Maximum projective spanning tree parser using the Eisner (1996) algorithm.
///
/// Time complexity: O(n³).
pub struct EisnerParser {
    scores: ScoreMatrix,
}

/// Backtrack structure storing the split point and arc direction.
#[derive(Clone)]
struct Chart {
    score: f64,
    split: usize,
}

impl Chart {
    fn new() -> Self {
        Self { score: f64::NEG_INFINITY, split: 0 }
    }
}

impl EisnerParser {
    /// Construct an `EisnerParser` with the given score matrix.
    pub fn new(scores: ScoreMatrix) -> Self {
        Self { scores }
    }

    /// Construct an `EisnerParser` using a distance heuristic for `n_tokens` tokens.
    pub fn from_heuristic(n_tokens: usize) -> Self {
        Self::new(ScoreMatrix::from_distance_heuristic(n_tokens))
    }

    /// Run Eisner's algorithm and return the head array (1-indexed).
    ///
    /// `heads[i]` is the head of token `i` (0 = ROOT).  Index 0 is unused.
    pub fn parse(&self, n_tokens: usize) -> Vec<usize> {
        let n = n_tokens + 1; // +1 for ROOT at 0
        if n <= 1 {
            return vec![0usize; n];
        }

        // c[i][j][dir]: best complete span [i,j] with head at i (dir=0) or j (dir=1).
        // ic[i][j][dir]: best incomplete span [i,j] with attachment in direction dir.
        let mut c:  Vec<Vec<[Chart; 2]>> = vec![vec![[Chart::new(), Chart::new()]; n]; n];
        let mut ic: Vec<Vec<[Chart; 2]>> = vec![vec![[Chart::new(), Chart::new()]; n]; n];

        // Base case: zero-length spans.
        for i in 0..n {
            c[i][i][0].score = 0.0;
            c[i][i][1].score = 0.0;
        }

        // Fill chart by span length.
        for length in 1..n {
            for i in 0..(n - length) {
                let j = i + length;

                // Incomplete span: arc i → j (right attachment)
                {
                    let arc_score = self.scores.scores[i][j];
                    let mut best = Chart::new();
                    for k in i..j {
                        let v = c[i][k][0].score + c[k + 1][j][1].score + arc_score;
                        if v > best.score {
                            best.score = v;
                            best.split = k;
                        }
                    }
                    ic[i][j][0] = best;
                }

                // Incomplete span: arc j → i (left attachment)
                {
                    let arc_score = self.scores.scores[j][i];
                    let mut best = Chart::new();
                    for k in i..j {
                        let v = c[i][k][0].score + c[k + 1][j][1].score + arc_score;
                        if v > best.score {
                            best.score = v;
                            best.split = k;
                        }
                    }
                    ic[i][j][1] = best;
                }

                // Complete span, head at i (right)
                {
                    let mut best = Chart::new();
                    for k in i..j {
                        let v = c[i][k][0].score + ic[k][j][0].score;
                        if v > best.score {
                            best.score = v;
                            best.split = k;
                        }
                    }
                    c[i][j][0] = best;
                }

                // Complete span, head at j (left)
                {
                    let mut best = Chart::new();
                    for k in (i + 1)..=j {
                        let v = ic[i][k][1].score + c[k][j][1].score;
                        if v > best.score {
                            best.score = v;
                            best.split = k;
                        }
                    }
                    c[i][j][1] = best;
                }
            }
        }

        // Backtrack to recover head assignments.
        let mut heads = vec![0usize; n];
        // The root span is [0, n-1] with head at 0.
        Self::backtrack_complete(&c, &ic, &mut heads, 0, n - 1, 0);
        heads
    }

    fn backtrack_complete(
        c:     &Vec<Vec<[Chart; 2]>>,
        ic:    &Vec<Vec<[Chart; 2]>>,
        heads: &mut Vec<usize>,
        i: usize, j: usize, dir: usize,
    ) {
        if i == j {
            return;
        }
        let k = c[i][j][dir].split;
        if dir == 0 {
            // c[i][j][0] = c[i][k][0] + ic[k][j][0]
            Self::backtrack_complete(c, ic, heads, i, k, 0);
            Self::backtrack_incomplete(c, ic, heads, k, j, 0);
        } else {
            // c[i][j][1] = ic[i][k][1] + c[k][j][1]
            Self::backtrack_incomplete(c, ic, heads, i, k, 1);
            Self::backtrack_complete(c, ic, heads, k, j, 1);
        }
    }

    fn backtrack_incomplete(
        c:     &Vec<Vec<[Chart; 2]>>,
        ic:    &Vec<Vec<[Chart; 2]>>,
        heads: &mut Vec<usize>,
        i: usize, j: usize, dir: usize,
    ) {
        if i == j {
            return;
        }
        let k = ic[i][j][dir].split;
        if dir == 0 {
            // arc i → j; split at k: c[i][k][0] + c[k+1][j][1]
            heads[j] = i;
            Self::backtrack_complete(c, ic, heads, i, k, 0);
            if k + 1 <= j {
                Self::backtrack_complete(c, ic, heads, k + 1, j, 1);
            }
        } else {
            // arc j → i; split at k: c[i][k][0] + c[k+1][j][1]
            heads[i] = j;
            Self::backtrack_complete(c, ic, heads, i, k, 0);
            if k + 1 <= j {
                Self::backtrack_complete(c, ic, heads, k + 1, j, 1);
            }
        }
    }

    /// Build a `DependencyGraph` from the Eisner parse, using `Dep` as label for
    /// all arcs.  Token indices in the returned graph are 1-based.
    pub fn parse_to_graph(&self, tokens: Vec<String>, pos_tags: Vec<String>) -> DependencyGraph {
        let n = tokens.len();
        let heads = self.parse(n);
        let mut g = DependencyGraph::new(tokens, pos_tags);
        for dep in 1..=n {
            let head = if dep < heads.len() { heads[dep] } else { 0 };
            g.add_arc(head, dep, DepLabel::Dep, self.scores.scores[head][dep]);
        }
        g
    }
}

// ---------------------------------------------------------------------------
// Chu-Liu / Edmonds
// ---------------------------------------------------------------------------

/// Maximum spanning arborescence rooted at node 0 via Chu-Liu/Edmonds algorithm.
///
/// Handles non-projective trees.  The implementation uses the standard
/// contraction-based approach (Tarjan 1977).
pub struct ChuLiuEdmonds;

impl ChuLiuEdmonds {
    /// Find the maximum spanning arborescence of the score matrix.
    ///
    /// Returns a `heads` vector (length = `scores.n`): `heads[i]` is the
    /// best head for node `i`; `heads[0]` is 0 (ROOT has no head).
    pub fn max_arborescence(scores: &ScoreMatrix) -> Vec<usize> {
        let n = scores.n;
        if n <= 1 {
            return vec![0; n];
        }
        chu_liu_edmonds_impl(&scores.scores, n)
    }

    /// Build a `DependencyGraph` from the Chu-Liu/Edmonds max arborescence.
    pub fn parse_to_graph(
        scores: &ScoreMatrix,
        tokens: Vec<String>,
        pos_tags: Vec<String>,
    ) -> DependencyGraph {
        let n = tokens.len();
        let heads = Self::max_arborescence(scores);
        let mut g = DependencyGraph::new(tokens, pos_tags);
        for dep in 1..=n {
            let head = if dep < heads.len() { heads[dep] } else { 0 };
            let score = if head < scores.n && dep < scores.n {
                scores.scores[head][dep]
            } else {
                0.0
            };
            g.add_arc(head, dep, DepLabel::Dep, score);
        }
        g
    }
}

// ---------------------------------------------------------------------------
// Chu-Liu/Edmonds implementation
// ---------------------------------------------------------------------------

/// Full Chu-Liu/Edmonds with cycle contraction.
///
/// This is Tarjan's efficient version.  Nodes are contracted iteratively
/// until no cycles remain.
fn chu_liu_edmonds_impl(scores: &Vec<Vec<f64>>, n: usize) -> Vec<usize> {
    // Step 1: for each node i != 0, pick the highest-scoring incoming arc.
    let best_head: Vec<usize> = (0..n)
        .map(|i| {
            if i == 0 {
                0
            } else {
                (0..n)
                    .filter(|&j| j != i)
                    .max_by(|&a, &b| {
                        scores[a][i]
                            .partial_cmp(&scores[b][i])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0)
            }
        })
        .collect();

    // Step 2: detect a cycle in `best_head`.
    let cycle = find_cycle(&best_head, n);

    if let Some(cycle_nodes) = cycle {
        // Contract the cycle into a single super-node and recurse.
        contract_and_recurse(scores, n, &best_head, &cycle_nodes)
    } else {
        best_head
    }
}

/// Find a cycle in the `heads` array.  Returns `None` if the graph is acyclic.
fn find_cycle(heads: &[usize], n: usize) -> Option<Vec<usize>> {
    let mut color = vec![0u8; n]; // 0=white, 1=grey, 2=black
    for start in 1..n {
        if color[start] != 0 {
            continue;
        }
        // Walk the chain from `start`.
        let mut path = Vec::new();
        let mut cur = start;
        loop {
            if color[cur] == 2 {
                break;
            }
            if color[cur] == 1 {
                // Found cycle — collect it.
                let pos = path.iter().position(|&x| x == cur).unwrap_or(0);
                return Some(path[pos..].to_vec());
            }
            color[cur] = 1;
            path.push(cur);
            if cur == 0 { break; }
            cur = heads[cur];
        }
        for &v in &path {
            color[v] = 2;
        }
    }
    None
}

/// Contract a cycle into a super-node, recurse, then expand.
fn contract_and_recurse(
    scores: &Vec<Vec<f64>>,
    n: usize,
    best_head: &[usize],
    cycle: &[usize],
) -> Vec<usize> {
    // Map original nodes to contracted nodes.
    let cycle_id = n; // super-node id
    let cycle_set: std::collections::HashSet<usize> = cycle.iter().copied().collect();

    // score of cycle node best_head[v] -> v for each v in cycle.
    let cycle_score: f64 = cycle.iter().map(|&v| scores[best_head[v]][v]).sum();

    // Build contracted score matrix (size = n+1 after contraction).
    // Contracted node ids: non-cycle nodes keep their ids; cycle → cycle_id.
    let contracted_n = n - cycle.len() + 1;
    let id_map: Vec<usize> = (0..n).collect();
    let mut new_id = 0usize;
    let mut remap = vec![0usize; n + 1];
    for i in 0..n {
        if cycle_set.contains(&i) {
            remap[i] = contracted_n - 1; // last slot = super-node
        } else {
            remap[i] = new_id;
            new_id += 1;
        }
    }
    let _ = id_map; // suppress unused warning; remap is used instead.
    let _ = cycle_id;

    let mut new_scores: Vec<Vec<f64>> =
        vec![vec![f64::NEG_INFINITY; contracted_n]; contracted_n];
    let super_id = contracted_n - 1;

    for h in 0..n {
        for d in 0..n {
            if h == d { continue; }
            let nh = remap[h];
            let nd = remap[d];
            if nh == nd { continue; } // both in cycle → skip

            if cycle_set.contains(&d) {
                // Arc into cycle: benefit = scores[h][d] - scores[best_head[d]][d] + cycle_score
                let benefit = scores[h][d] - scores[best_head[d]][d] + cycle_score;
                if benefit > new_scores[nh][super_id] {
                    new_scores[nh][super_id] = benefit;
                }
            } else {
                if scores[h][d] > new_scores[nh][nd] {
                    new_scores[nh][nd] = scores[h][d];
                }
            }
        }
    }

    // Recurse.
    let contracted_heads = chu_liu_edmonds_impl(&new_scores, contracted_n);

    // Expand: re-map contracted heads back to original nodes.
    let mut final_heads = best_head.to_vec();

    // For non-cycle nodes, use the contracted solution.
    for orig in 0..n {
        if cycle_set.contains(&orig) { continue; }
        let contracted_node = remap[orig];
        if contracted_node < contracted_heads.len() {
            let contracted_parent = contracted_heads[contracted_node];
            // Map contracted_parent back to an original node.
            // We stored nh = remap[h] for non-cycle nodes.
            // Find original node whose remap == contracted_parent.
            let orig_parent = (0..n)
                .filter(|&x| !cycle_set.contains(&x) && remap[x] == contracted_parent)
                .next()
                .unwrap_or(0);
            final_heads[orig] = orig_parent;
        }
    }

    // The cycle node that receives the external arc gets its head re-assigned.
    // All other cycle nodes keep their best_head.
    // Collect candidates without nested move closures to avoid borrow issues.
    let mut best_ext: Option<(usize, usize)> = None;
    let mut best_score = f64::NEG_INFINITY;
    for h in 0..n {
        if cycle_set.contains(&h) { continue; }
        let nh = remap[h];
        if contracted_heads.get(nh).copied().unwrap_or(usize::MAX) == super_id {
            for &d in cycle {
                let s = scores[h][d];
                if s > best_score {
                    best_score = s;
                    best_ext = Some((h, d));
                }
            }
        }
    }

    if let Some((ext_h, ext_d)) = best_ext {
        final_heads[ext_d] = ext_h;
    }

    final_heads
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_matrix_heuristic() {
        let m = ScoreMatrix::from_distance_heuristic(4);
        assert_eq!(m.n, 5);
        // Self-arcs are -∞.
        assert!(m.scores[1][1].is_infinite());
        // ROOT → token 1: distance = 1, score = -ln(1) = 0.
        assert!((m.scores[0][1] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_eisner_single_token() {
        let parser = EisnerParser::from_heuristic(1);
        let heads  = parser.parse(1);
        // Token 1 should be attached to ROOT (0).
        assert_eq!(heads[1], 0);
    }

    #[test]
    fn test_eisner_three_tokens() {
        let parser = EisnerParser::from_heuristic(3);
        let heads  = parser.parse(3);
        // Every token 1..3 must have a valid head in 0..3.
        for i in 1..=3 {
            assert!(heads[i] < 4, "token {} head {} out of range", i, heads[i]);
            assert_ne!(heads[i], i, "token {} is its own head", i);
        }
    }

    #[test]
    fn test_eisner_parse_to_graph() {
        let parser  = EisnerParser::from_heuristic(3);
        let tokens  = ["The", "cat", "sat"].map(String::from).to_vec();
        let pos     = ["DT", "NN", "VBD"].map(String::from).to_vec();
        let graph   = parser.parse_to_graph(tokens, pos);
        assert_eq!(graph.n_tokens, 3);
        for i in 1..=3 {
            assert!(graph.head_of(i).is_some(), "token {} has no head", i);
        }
        // A distance-based parse is always projective.
        assert!(graph.is_projective());
    }

    #[test]
    fn test_cle_single_token() {
        let scores = ScoreMatrix::from_distance_heuristic(1);
        let heads  = ChuLiuEdmonds::max_arborescence(&scores);
        assert_eq!(heads[1], 0);
    }

    #[test]
    fn test_cle_three_tokens() {
        let scores = ScoreMatrix::from_distance_heuristic(3);
        let heads  = ChuLiuEdmonds::max_arborescence(&scores);
        for i in 1..=3 {
            if i < heads.len() {
                assert!(heads[i] < 4, "CLE head out of range");
            }
        }
    }

    #[test]
    fn test_cle_parse_to_graph() {
        let scores = ScoreMatrix::from_distance_heuristic(3);
        let tokens = ["The", "cat", "sat"].map(String::from).to_vec();
        let pos    = ["DT", "NN", "VBD"].map(String::from).to_vec();
        let graph  = ChuLiuEdmonds::parse_to_graph(&scores, tokens, pos);
        assert_eq!(graph.n_tokens, 3);
        for i in 1..=3 {
            assert!(graph.head_of(i).is_some());
        }
    }

    #[test]
    fn test_find_cycle() {
        // Build a trivial cycle 1 → 2 → 1.
        let heads = vec![0, 2, 1, 0]; // node 0 = ROOT
        let cycle = find_cycle(&heads, 4);
        assert!(cycle.is_some());
    }

    #[test]
    fn test_no_cycle() {
        // Tree: ROOT → 1, 1 → 2, 2 → 3 — no cycle.
        let heads = vec![0, 0, 1, 2];
        let cycle = find_cycle(&heads, 4);
        assert!(cycle.is_none());
    }
}
