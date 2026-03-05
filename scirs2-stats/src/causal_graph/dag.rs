//! Directed Acyclic Graph (DAG) for causal structure representation.
//!
//! # Overview
//!
//! A [`CausalDAG`] stores a causal graph where nodes represent random variables
//! and directed edges `parent → child` represent direct causal influences.
//!
//! Key operations:
//! - Topology queries: parents, children, ancestors, descendants
//! - D-separation via the **Bayes Ball** algorithm (Shachter 1998)
//! - Markov blanket computation
//! - Cycle detection (Kahn's algorithm) to enforce the acyclicity constraint
//!
//! # References
//!
//! - Pearl, J. (2000). *Causality*. Cambridge University Press.
//! - Shachter, R.D. (1998). Bayes-Ball: The Rational Pastime.
//!   *Proc. UAI 1998*, pp. 480-487.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// CausalDAG
// ---------------------------------------------------------------------------

/// A directed acyclic graph (DAG) representing a causal structure.
///
/// Each **node** is identified by a unique string name.  Each **edge**
/// `(parent, child)` represents a direct causal link `parent → child`.
///
/// The struct enforces acyclicity at insertion time via a DFS-based cycle
/// check, so every valid `CausalDAG` is guaranteed to be a true DAG.
#[derive(Debug, Clone)]
pub struct CausalDAG {
    /// Ordered list of node names (index == node id)
    nodes: Vec<String>,
    /// Edges stored as (parent_idx, child_idx)
    edges: Vec<(usize, usize)>,
    /// Map from name → index
    node_map: HashMap<String, usize>,
}

impl Default for CausalDAG {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalDAG {
    /// Create an empty DAG.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            node_map: HashMap::new(),
        }
    }

    // ------------------------------------------------------------------
    // Node / edge management
    // ------------------------------------------------------------------

    /// Add a node by name, returning its index.
    /// If the node already exists the existing index is returned.
    pub fn add_node(&mut self, name: &str) -> usize {
        if let Some(&idx) = self.node_map.get(name) {
            return idx;
        }
        let idx = self.nodes.len();
        self.nodes.push(name.to_owned());
        self.node_map.insert(name.to_owned(), idx);
        idx
    }

    /// Add a directed edge `parent → child`.
    ///
    /// Both nodes are created automatically if they do not yet exist.
    /// Returns `Err` if adding the edge would introduce a cycle.
    pub fn add_edge(&mut self, parent: &str, child: &str) -> StatsResult<()> {
        let p = self.add_node(parent);
        let c = self.add_node(child);
        if p == c {
            return Err(StatsError::InvalidArgument(
                "Self-loops are not allowed in a DAG".to_owned(),
            ));
        }
        // Temporarily add the edge then check for cycles.
        self.edges.push((p, c));
        if self.has_cycle() {
            self.edges.pop();
            return Err(StatsError::InvalidArgument(format!(
                "Adding edge {parent} → {child} would create a cycle"
            )));
        }
        Ok(())
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Returns the name of node `idx`, or `None` if out of range.
    pub fn node_name(&self, idx: usize) -> Option<&str> {
        self.nodes.get(idx).map(String::as_str)
    }

    /// Returns the index of a node by name, or `None` if unknown.
    pub fn node_index(&self, name: &str) -> Option<usize> {
        self.node_map.get(name).copied()
    }

    /// All node names in insertion order.
    pub fn node_names(&self) -> Vec<&str> {
        self.nodes.iter().map(String::as_str).collect()
    }

    /// All edges as `(parent_name, child_name)`.
    pub fn edge_list(&self) -> Vec<(&str, &str)> {
        self.edges
            .iter()
            .map(|&(p, c)| (self.nodes[p].as_str(), self.nodes[c].as_str()))
            .collect()
    }

    // ------------------------------------------------------------------
    // Adjacency helpers (private, index-based)
    // ------------------------------------------------------------------

    fn parent_indices(&self, idx: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|&&(_, c)| c == idx)
            .map(|&(p, _)| p)
            .collect()
    }

    fn child_indices(&self, idx: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|&&(p, _)| p == idx)
            .map(|&(_, c)| c)
            .collect()
    }

    // ------------------------------------------------------------------
    // Public topology queries
    // ------------------------------------------------------------------

    /// Direct parents of `node`.
    pub fn parents(&self, node: &str) -> Vec<&str> {
        match self.node_map.get(node) {
            None => Vec::new(),
            Some(&idx) => self
                .parent_indices(idx)
                .into_iter()
                .map(|i| self.nodes[i].as_str())
                .collect(),
        }
    }

    /// Direct children of `node`.
    pub fn children(&self, node: &str) -> Vec<&str> {
        match self.node_map.get(node) {
            None => Vec::new(),
            Some(&idx) => self
                .child_indices(idx)
                .into_iter()
                .map(|i| self.nodes[i].as_str())
                .collect(),
        }
    }

    /// All ancestors of `node` (not including `node` itself).
    pub fn ancestors(&self, node: &str) -> HashSet<usize> {
        let start = match self.node_map.get(node) {
            None => return HashSet::new(),
            Some(&i) => i,
        };
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        for p in self.parent_indices(start) {
            queue.push_back(p);
        }
        while let Some(cur) = queue.pop_front() {
            if visited.insert(cur) {
                for p in self.parent_indices(cur) {
                    if !visited.contains(&p) {
                        queue.push_back(p);
                    }
                }
            }
        }
        visited
    }

    /// All descendants of `node` (not including `node` itself).
    pub fn descendants(&self, node: &str) -> HashSet<usize> {
        let start = match self.node_map.get(node) {
            None => return HashSet::new(),
            Some(&i) => i,
        };
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        for c in self.child_indices(start) {
            queue.push_back(c);
        }
        while let Some(cur) = queue.pop_front() {
            if visited.insert(cur) {
                for c in self.child_indices(cur) {
                    if !visited.contains(&c) {
                        queue.push_back(c);
                    }
                }
            }
        }
        visited
    }

    // ------------------------------------------------------------------
    // D-separation — Bayes Ball algorithm
    // ------------------------------------------------------------------

    /// Test whether `x` and `y` are **d-separated** given the set `z`
    /// of observed (conditioned) variables, using the Bayes Ball algorithm.
    ///
    /// Returns `true` when X ⊥ Y | Z holds in every distribution
    /// compatible with the DAG.
    pub fn is_d_separated(&self, x: &str, y: &str, z: &[&str]) -> bool {
        let xi = match self.node_map.get(x) {
            None => return true,
            Some(&i) => i,
        };
        let yi = match self.node_map.get(y) {
            None => return true,
            Some(&i) => i,
        };
        let observed: HashSet<usize> = z
            .iter()
            .filter_map(|name| self.node_map.get(*name).copied())
            .collect();

        // Compute the set of ancestors of observed variables (needed for
        // v-structure activation).
        let observed_ancestors = self.ancestors_of_set(&observed);

        // Bayes Ball state: (node_idx, came_from_child: bool)
        // came_from_child=true  → ball arrived via an upward (child→parent) move
        // came_from_child=false → ball arrived via a downward (parent→child) move
        let mut visited: HashSet<(usize, bool)> = HashSet::new();
        let mut queue: VecDeque<(usize, bool)> = VecDeque::new();

        // Start at xi travelling upward (as if we came from a child).
        queue.push_back((xi, true));
        queue.push_back((xi, false));

        while let Some((node, via_child)) = queue.pop_front() {
            if !visited.insert((node, via_child)) {
                continue;
            }
            if node == yi {
                // Found a path — not d-separated.
                return false;
            }

            let is_observed = observed.contains(&node);
            let is_anc_obs = observed_ancestors.contains(&node);

            if via_child && !is_observed {
                // Ball came from child and node is NOT observed.
                // Pass to parents (chain/fork continue upstream).
                for p in self.parent_indices(node) {
                    queue.push_back((p, true));
                }
                // Pass to children (chain continues downstream).
                for c in self.child_indices(node) {
                    queue.push_back((c, false));
                }
            }

            if !via_child && !is_observed {
                // Ball came from parent and node is NOT observed.
                for c in self.child_indices(node) {
                    queue.push_back((c, false));
                }
            }

            if !via_child && is_observed {
                // Ball came from parent and node IS observed (blocks chain/fork).
                // Pass back upstream.
                for p in self.parent_indices(node) {
                    queue.push_back((p, true));
                }
            }

            // V-structure (collider): node has multiple parents.
            // Activated only when node or one of its descendants is observed.
            if via_child && (is_observed || is_anc_obs) {
                for p in self.parent_indices(node) {
                    queue.push_back((p, true));
                }
            }
        }

        true
    }

    // ------------------------------------------------------------------
    // Markov blanket
    // ------------------------------------------------------------------

    /// **Markov blanket** of `node`: its parents, its children, and the
    /// other parents of its children (spouses).
    pub fn markov_blanket(&self, node: &str) -> Vec<&str> {
        let idx = match self.node_map.get(node) {
            None => return Vec::new(),
            Some(&i) => i,
        };
        let mut mb = HashSet::new();
        // Parents
        for p in self.parent_indices(idx) {
            mb.insert(p);
        }
        // Children + their other parents (co-parents / spouses)
        for c in self.child_indices(idx) {
            mb.insert(c);
            for p in self.parent_indices(c) {
                if p != idx {
                    mb.insert(p);
                }
            }
        }
        mb.into_iter()
            .map(|i| self.nodes[i].as_str())
            .collect()
    }

    // ------------------------------------------------------------------
    // Topological sort (Kahn's algorithm)
    // ------------------------------------------------------------------

    /// Return a topological ordering of all nodes.
    ///
    /// Returns `Err` if the graph contains a cycle (should not happen if
    /// all edges were added through [`add_edge`]).
    pub fn topological_sort(&self) -> Vec<&str> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        for &(_, c) in &self.edges {
            in_degree[c] += 1;
        }
        let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for c in self.child_indices(u) {
                in_degree[c] -= 1;
                if in_degree[c] == 0 {
                    queue.push_back(c);
                }
            }
        }
        order
            .into_iter()
            .map(|i| self.nodes[i].as_str())
            .collect()
    }

    // ------------------------------------------------------------------
    // C-components (for Tian-Pearl identification)
    // ------------------------------------------------------------------

    /// Partition the nodes into **c-components** (connected components of
    /// the bidirected part of an ADMG — here approximated by bidirected
    /// edges derived from pairs sharing a common latent parent).
    ///
    /// In the pure-DAG case every node is its own c-component.  This
    /// method is extended in the identification module.
    pub fn c_components(&self) -> Vec<HashSet<usize>> {
        let n = self.nodes.len();
        let mut component = vec![usize::MAX; n];
        let mut next_comp = 0usize;
        for i in 0..n {
            if component[i] == usize::MAX {
                component[i] = next_comp;
                next_comp += 1;
            }
        }
        let mut comps: Vec<HashSet<usize>> = Vec::new();
        for i in 0..n {
            let c = component[i];
            while comps.len() <= c {
                comps.push(HashSet::new());
            }
            comps[c].insert(i);
        }
        comps
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// Check whether the graph currently contains a cycle (DFS colouring).
    fn has_cycle(&self) -> bool {
        let n = self.nodes.len();
        // 0 = unvisited, 1 = in stack, 2 = done
        let mut colour = vec![0u8; n];
        for start in 0..n {
            if colour[start] == 0 && self.dfs_cycle(start, &mut colour) {
                return true;
            }
        }
        false
    }

    fn dfs_cycle(&self, node: usize, colour: &mut Vec<u8>) -> bool {
        colour[node] = 1; // grey
        for c in self.child_indices(node) {
            if colour[c] == 1 {
                return true; // back-edge → cycle
            }
            if colour[c] == 0 && self.dfs_cycle(c, colour) {
                return true;
            }
        }
        colour[node] = 2; // black
        false
    }

    /// Compute all ancestors of a *set* of nodes.
    fn ancestors_of_set(&self, nodes: &HashSet<usize>) -> HashSet<usize> {
        let mut ancestors = HashSet::new();
        let mut queue: VecDeque<usize> = nodes.iter().copied().collect();
        while let Some(cur) = queue.pop_front() {
            for p in self.parent_indices(cur) {
                if ancestors.insert(p) {
                    queue.push_back(p);
                }
            }
        }
        ancestors
    }

    /// Returns (parent_indices_vec, child_indices_vec) for external use.
    pub(crate) fn adjacency(&self, idx: usize) -> (Vec<usize>, Vec<usize>) {
        (self.parent_indices(idx), self.child_indices(idx))
    }

    /// Remove all edges where the child node index is in `target_indices`.
    pub(crate) fn remove_incoming_edges_for(&mut self, target_indices: &std::collections::HashSet<usize>) {
        self.edges.retain(|&(_, c)| !target_indices.contains(&c));
    }

    /// Remove all edges where the parent node index is in `target_indices`.
    pub(crate) fn remove_outgoing_edges_for(&mut self, target_indices: &std::collections::HashSet<usize>) {
        self.edges.retain(|&(p, _)| !target_indices.contains(&p));
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_chain() -> CausalDAG {
        // X → Y → Z
        let mut dag = CausalDAG::new();
        dag.add_edge("X", "Y").unwrap();
        dag.add_edge("Y", "Z").unwrap();
        dag
    }

    fn build_fork() -> CausalDAG {
        // X → Y, X → Z
        let mut dag = CausalDAG::new();
        dag.add_edge("X", "Y").unwrap();
        dag.add_edge("X", "Z").unwrap();
        dag
    }

    fn build_collider() -> CausalDAG {
        // X → Z ← Y
        let mut dag = CausalDAG::new();
        dag.add_edge("X", "Z").unwrap();
        dag.add_edge("Y", "Z").unwrap();
        dag
    }

    #[test]
    fn test_cycle_detection() {
        let mut dag = CausalDAG::new();
        dag.add_edge("A", "B").unwrap();
        dag.add_edge("B", "C").unwrap();
        let res = dag.add_edge("C", "A");
        assert!(res.is_err(), "Should reject cycle A→B→C→A");
    }

    #[test]
    fn test_self_loop_rejected() {
        let mut dag = CausalDAG::new();
        assert!(dag.add_edge("A", "A").is_err());
    }

    #[test]
    fn test_parents_children() {
        let mut dag = CausalDAG::new();
        dag.add_edge("A", "B").unwrap();
        dag.add_edge("A", "C").unwrap();
        dag.add_edge("B", "C").unwrap();

        let mut pa_c = dag.parents("C");
        pa_c.sort();
        assert_eq!(pa_c, vec!["A", "B"]);

        let mut ch_a = dag.children("A");
        ch_a.sort();
        assert_eq!(ch_a, vec!["B", "C"]);
    }

    #[test]
    fn test_ancestors_descendants() {
        let dag = build_chain();
        let xi = dag.node_index("X").unwrap();
        let yi = dag.node_index("Y").unwrap();
        let zi = dag.node_index("Z").unwrap();

        let anc_z = dag.ancestors("Z");
        assert!(anc_z.contains(&xi));
        assert!(anc_z.contains(&yi));

        let desc_x = dag.descendants("X");
        assert!(desc_x.contains(&yi));
        assert!(desc_x.contains(&zi));
    }

    #[test]
    fn test_d_separation_chain() {
        let dag = build_chain();
        // X ⊥ Z | Y in a chain
        assert!(dag.is_d_separated("X", "Z", &["Y"]));
        // X and Z are NOT d-separated when Y is unobserved
        assert!(!dag.is_d_separated("X", "Z", &[]));
    }

    #[test]
    fn test_d_separation_fork() {
        let dag = build_fork();
        // Y ⊥ Z | X in a fork
        assert!(dag.is_d_separated("Y", "Z", &["X"]));
        assert!(!dag.is_d_separated("Y", "Z", &[]));
    }

    #[test]
    fn test_d_separation_collider() {
        let dag = build_collider();
        // X ⊥ Y unconditionally (collider Z blocks)
        assert!(dag.is_d_separated("X", "Y", &[]));
        // Conditioning on the collider opens the path
        assert!(!dag.is_d_separated("X", "Y", &["Z"]));
    }

    #[test]
    fn test_markov_blanket() {
        let mut dag = CausalDAG::new();
        dag.add_edge("A", "B").unwrap();
        dag.add_edge("C", "B").unwrap();
        dag.add_edge("B", "D").unwrap();
        dag.add_edge("E", "D").unwrap();

        let mut mb = dag.markov_blanket("B");
        mb.sort();
        // Parents: A, C; Children: D; Spouses (other parents of D): E
        assert_eq!(mb, vec!["A", "C", "D", "E"]);
    }

    #[test]
    fn test_topological_sort() {
        let dag = build_chain();
        let order = dag.topological_sort();
        // X must come before Y and Y before Z
        let xi = order.iter().position(|&s| s == "X").unwrap();
        let yi = order.iter().position(|&s| s == "Y").unwrap();
        let zi = order.iter().position(|&s| s == "Z").unwrap();
        assert!(xi < yi && yi < zi);
    }
}
