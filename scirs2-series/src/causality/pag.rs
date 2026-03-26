//! Partial Ancestral Graph (PAG) data structure for FCI algorithm
//!
//! A PAG represents the equivalence class of Maximal Ancestral Graphs (MAGs)
//! that are Markov-equivalent. It uses three types of edge marks:
//! - **Tail** (`--`): definite non-ancestor endpoint
//! - **Arrowhead** (`->`): definite ancestor endpoint
//! - **Circle** (`o`): unknown (could be tail or arrowhead)
//!
//! Common edge types in PAGs:
//! - `o-o` : completely undetermined
//! - `o->` : partial direction known
//! - `-->` : directed edge (definite causal direction)
//! - `<->` : bidirected edge (latent common cause)

use std::collections::HashMap;

/// Edge mark types for PAG edges
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeMark {
    /// Tail mark (`--`): indicates definite non-ancestor relationship
    Tail,
    /// Arrowhead mark (`->`): indicates definite ancestor relationship
    Arrowhead,
    /// Circle mark (`o`): unknown — could be tail or arrowhead
    Circle,
}

/// A node in the lagged causal graph: variable at a specific time lag
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LaggedNode {
    /// Index of the variable
    pub var_idx: usize,
    /// Time lag: 0 = contemporaneous, negative = past
    pub lag: i32,
}

/// An edge in the PAG with directional marks at each endpoint
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagEdge {
    /// Mark at the 'from' node end of this edge
    pub from_mark: EdgeMark,
    /// Mark at the 'to' node end of this edge
    pub to_mark: EdgeMark,
}

/// Partial Ancestral Graph representation
///
/// Edges are stored with a canonical key `(min(i,j), max(i,j))` but with
/// directional marks stored relative to the canonical orientation.
/// The "from" mark is at the smaller-index node end,
/// and the "to" mark is at the larger-index node end.
#[derive(Debug, Clone, Default)]
pub struct PartialAncestralGraph {
    /// Number of original variables (not counting lags)
    pub n_vars: usize,
    /// Maximum lag considered
    pub tau_max: usize,
    /// Total number of nodes in the PAG: n_vars * (tau_max + 1)
    pub n_nodes: usize,
    /// Adjacency: canonical key (min_idx, max_idx) → PagEdge
    /// The from_mark corresponds to the min_idx node end,
    /// the to_mark to the max_idx node end.
    adjacency: HashMap<(usize, usize), PagEdge>,
    /// Separation sets from skeleton discovery phase
    pub sep_sets: HashMap<(usize, usize), Vec<usize>>,
}

impl PartialAncestralGraph {
    /// Create a new empty PAG with given number of nodes
    pub fn new(n_nodes: usize) -> Self {
        Self {
            n_vars: n_nodes,
            tau_max: 0,
            n_nodes,
            adjacency: HashMap::new(),
            sep_sets: HashMap::new(),
        }
    }

    /// Create a PAG with explicit n_vars and tau_max, computing n_nodes automatically
    pub fn with_vars_and_lags(n_vars: usize, tau_max: usize) -> Self {
        let n_nodes = n_vars * (tau_max + 1);
        Self {
            n_vars,
            tau_max,
            n_nodes,
            adjacency: HashMap::new(),
            sep_sets: HashMap::new(),
        }
    }

    /// Create a PAG from a skeleton adjacency matrix,
    /// initializing all skeleton edges with Circle-Circle marks
    pub fn initialize_from_skeleton(skeleton_adj: &[Vec<bool>], n_nodes: usize) -> Self {
        let mut pag = Self {
            n_vars: n_nodes,
            tau_max: 0,
            n_nodes,
            adjacency: HashMap::new(),
            sep_sets: HashMap::new(),
        };
        for i in 0..n_nodes {
            if i >= skeleton_adj.len() {
                break;
            }
            for j in (i + 1)..n_nodes {
                if j >= skeleton_adj[i].len() {
                    break;
                }
                if skeleton_adj[i][j] {
                    pag.add_edge(i, j, EdgeMark::Circle, EdgeMark::Circle);
                }
            }
        }
        pag
    }

    /// Add or overwrite an edge between nodes i and j
    ///
    /// The from_mark is at node i and the to_mark is at node j.
    /// Internally stored with canonical key (min(i,j), max(i,j)).
    pub fn add_edge(&mut self, i: usize, j: usize, from_mark: EdgeMark, to_mark: EdgeMark) {
        let (key, edge) = if i <= j {
            ((i, j), PagEdge { from_mark, to_mark })
        } else {
            // Swap nodes: the canonical from is now j, canonical to is i
            (
                (j, i),
                PagEdge {
                    from_mark: to_mark,
                    to_mark: from_mark,
                },
            )
        };
        self.adjacency.insert(key, edge);
    }

    /// Remove the edge between nodes i and j
    pub fn remove_edge(&mut self, i: usize, j: usize) {
        let key = Self::canonical_key(i, j);
        self.adjacency.remove(&key);
    }

    /// Check if there is an edge between nodes i and j
    pub fn has_edge(&self, i: usize, j: usize) -> bool {
        let key = Self::canonical_key(i, j);
        self.adjacency.contains_key(&key)
    }

    /// Get the edge between nodes i and j, if it exists.
    ///
    /// Returns the edge with from_mark at node i, to_mark at node j.
    pub fn get_edge(&self, i: usize, j: usize) -> Option<PagEdge> {
        let key = Self::canonical_key(i, j);
        self.adjacency.get(&key).map(|e| {
            if i <= j {
                e.clone()
            } else {
                PagEdge {
                    from_mark: e.to_mark,
                    to_mark: e.from_mark,
                }
            }
        })
    }

    /// Set the mark at node `to` on the edge from `from` → `to`
    ///
    /// The mark is placed at the `to` endpoint.
    pub fn set_mark(&mut self, from: usize, to: usize, mark: EdgeMark) {
        let key = Self::canonical_key(from, to);
        if let Some(edge) = self.adjacency.get_mut(&key) {
            if from <= to {
                // canonical: from=small, to=large → to_mark is at `to`
                edge.to_mark = mark;
            } else {
                // canonical: from=large stored at key.1, to=small stored at key.0
                // in canonical (key.0, key.1): key.0=to, key.1=from
                // from_mark is at key.0 = to endpoint
                edge.from_mark = mark;
            }
        }
    }

    /// Get the mark at node `to` on the edge from `from` → `to`
    pub fn get_mark_at(&self, from: usize, to: usize) -> Option<EdgeMark> {
        let key = Self::canonical_key(from, to);
        self.adjacency
            .get(&key)
            .map(|e| if from <= to { e.to_mark } else { e.from_mark })
    }

    /// Get all nodes adjacent to the given node
    pub fn adjacent_nodes(&self, node: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();
        for &(a, b) in self.adjacency.keys() {
            if a == node {
                neighbors.push(b);
            } else if b == node {
                neighbors.push(a);
            }
        }
        neighbors.sort_unstable();
        neighbors.dedup();
        neighbors
    }

    /// Check if `parent` is a definite parent of `child`
    ///
    /// True when the edge has Tail at the parent end and Arrowhead at the child end.
    pub fn is_parent(&self, parent: usize, child: usize) -> bool {
        if let Some(edge) = self.get_edge(parent, child) {
            edge.from_mark == EdgeMark::Tail && edge.to_mark == EdgeMark::Arrowhead
        } else {
            false
        }
    }

    /// Count the number of bidirected edges (Arrowhead on both ends)
    pub fn n_bidirected_edges(&self) -> usize {
        self.adjacency
            .values()
            .filter(|e| e.from_mark == EdgeMark::Arrowhead && e.to_mark == EdgeMark::Arrowhead)
            .count()
    }

    /// Count the total number of Circle marks remaining in the PAG
    pub fn n_circle_marks(&self) -> usize {
        self.adjacency.values().fold(0, |acc, e| {
            let from_circle = usize::from(e.from_mark == EdgeMark::Circle);
            let to_circle = usize::from(e.to_mark == EdgeMark::Circle);
            acc + from_circle + to_circle
        })
    }

    /// Find all possible ancestors of a node
    ///
    /// A node X is a possible ancestor of Y if there exists a path from X to Y
    /// along which all edges are oriented towards Y (via Arrowhead or Circle marks
    /// at the Y-side). This follows Arrowhead and Circle marks leading toward `node`.
    pub fn possible_ancestors(&self, node: usize) -> Vec<usize> {
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![node];
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            // A node X might be an ancestor of `current` if there's an edge X *-> current
            // i.e., the mark at `current` end is Arrowhead or Circle
            for neighbor in self.adjacent_nodes(current) {
                if let Some(mark_at_current) = self.get_mark_at(neighbor, current) {
                    if mark_at_current == EdgeMark::Arrowhead || mark_at_current == EdgeMark::Circle
                    {
                        if !visited.contains(&neighbor) {
                            stack.push(neighbor);
                        }
                    }
                }
            }
        }
        visited.remove(&node);
        let mut ancestors: Vec<usize> = visited.into_iter().collect();
        ancestors.sort_unstable();
        ancestors
    }

    /// Return all edges as (from, to, PagEdge) triples.
    ///
    /// Each edge appears once with from < to in canonical form.
    pub fn edges(&self) -> impl Iterator<Item = (usize, usize, &PagEdge)> {
        self.adjacency.iter().map(|(&(a, b), edge)| (a, b, edge))
    }

    /// Return all node pairs (i, j) with i < j that have edges
    pub fn edge_node_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs: Vec<(usize, usize)> = self.adjacency.keys().copied().collect();
        pairs.sort_unstable();
        pairs
    }

    fn canonical_key(i: usize, j: usize) -> (usize, usize) {
        if i <= j {
            (i, j)
        } else {
            (j, i)
        }
    }
}

#[cfg(test)]
mod pag_tests {
    use super::*;

    #[test]
    fn test_pag_new_empty() {
        let pag = PartialAncestralGraph::new(4);
        assert_eq!(pag.n_nodes, 4);
        assert_eq!(pag.n_bidirected_edges(), 0);
        assert_eq!(pag.n_circle_marks(), 0);
    }

    #[test]
    fn test_pag_add_remove_edge() {
        let mut pag = PartialAncestralGraph::new(3);
        pag.add_edge(0, 1, EdgeMark::Circle, EdgeMark::Circle);
        assert!(pag.has_edge(0, 1));
        assert!(pag.has_edge(1, 0)); // symmetric
        pag.remove_edge(0, 1);
        assert!(!pag.has_edge(0, 1));
    }

    #[test]
    fn test_pag_initialize_from_skeleton_all_circles() {
        let adj = vec![
            vec![false, true, false],
            vec![true, false, true],
            vec![false, true, false],
        ];
        let pag = PartialAncestralGraph::initialize_from_skeleton(&adj, 3);
        // Edges 0-1 and 1-2 should exist with Circle-Circle
        assert!(pag.has_edge(0, 1));
        assert!(pag.has_edge(1, 2));
        assert!(!pag.has_edge(0, 2));
        // All marks should be Circle
        assert_eq!(pag.n_circle_marks(), 4); // 2 edges × 2 marks each
    }

    #[test]
    fn test_pag_set_and_get_mark() {
        let mut pag = PartialAncestralGraph::new(3);
        pag.add_edge(0, 1, EdgeMark::Circle, EdgeMark::Circle);
        // Set mark at node 1 on edge 0->1 to Arrowhead
        pag.set_mark(0, 1, EdgeMark::Arrowhead);
        let mark = pag.get_mark_at(0, 1);
        assert_eq!(mark, Some(EdgeMark::Arrowhead));
        // The mark at node 0 (from 1->0 direction) should still be Circle
        let mark0 = pag.get_mark_at(1, 0);
        assert_eq!(mark0, Some(EdgeMark::Circle));
    }

    #[test]
    fn test_pag_is_parent_true() {
        let mut pag = PartialAncestralGraph::new(3);
        // Edge: 0 ---> 1  (Tail at 0, Arrowhead at 1)
        pag.add_edge(0, 1, EdgeMark::Tail, EdgeMark::Arrowhead);
        assert!(pag.is_parent(0, 1));
        assert!(!pag.is_parent(1, 0));
    }

    #[test]
    fn test_pag_n_bidirected_edges_after_orient() {
        let mut pag = PartialAncestralGraph::new(4);
        pag.add_edge(0, 1, EdgeMark::Arrowhead, EdgeMark::Arrowhead); // bidirected
        pag.add_edge(1, 2, EdgeMark::Tail, EdgeMark::Arrowhead); // directed
        pag.add_edge(2, 3, EdgeMark::Circle, EdgeMark::Circle); // undetermined
        assert_eq!(pag.n_bidirected_edges(), 1);
    }

    #[test]
    fn test_pag_n_circle_marks() {
        let mut pag = PartialAncestralGraph::new(4);
        pag.add_edge(0, 1, EdgeMark::Circle, EdgeMark::Circle); // 2 circles
        pag.add_edge(1, 2, EdgeMark::Circle, EdgeMark::Arrowhead); // 1 circle
        pag.add_edge(2, 3, EdgeMark::Tail, EdgeMark::Arrowhead); // 0 circles
        assert_eq!(pag.n_circle_marks(), 3);
    }

    #[test]
    fn test_pag_adjacent_nodes() {
        let mut pag = PartialAncestralGraph::new(4);
        pag.add_edge(0, 1, EdgeMark::Circle, EdgeMark::Circle);
        pag.add_edge(0, 2, EdgeMark::Circle, EdgeMark::Circle);
        pag.add_edge(1, 3, EdgeMark::Circle, EdgeMark::Circle);
        let adj0 = pag.adjacent_nodes(0);
        assert_eq!(adj0, vec![1, 2]);
        let adj1 = pag.adjacent_nodes(1);
        assert_eq!(adj1, vec![0, 3]);
    }
}
