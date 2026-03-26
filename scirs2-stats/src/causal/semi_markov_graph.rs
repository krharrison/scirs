//! Semi-Markovian Causal Graph
//!
//! A [`SemiMarkovGraph`] represents a Semi-Markovian causal model (SMCM) where:
//!
//! - **Directed edges** (`→`) represent direct causal influences between observed variables.
//! - **Bidirected edges** (`↔`) represent latent common causes (hidden confounders)
//!   shared between two observed variables.
//!
//! This is the appropriate representation for the ID algorithm and do-calculus
//! identification procedures.
//!
//! # References
//!
//! - Tian, J. & Pearl, J. (2002). A General Identification Condition for
//!   Causal Effects. *AAAI 2002*.
//! - Shpitser, I. & Pearl, J. (2006). Identification of Joint Interventional
//!   Distributions in Recursive Semi-Markovian Causal Models. *AAAI 2006*.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

// ---------------------------------------------------------------------------
// SemiMarkovGraph
// ---------------------------------------------------------------------------

/// A semi-Markovian causal graph supporting directed (→) and bidirected (↔) edges.
///
/// Node names are arbitrary strings. Both directed and bidirected adjacency
/// are stored as sorted `BTreeSet`s for deterministic iteration.
#[derive(Debug, Clone)]
pub struct SemiMarkovGraph {
    /// Nodes in insertion order.
    node_list: Vec<String>,
    /// Set of all node names (for O(log n) membership testing).
    node_set: BTreeSet<String>,
    /// Directed adjacency: directed_children[v] = {w : v → w}.
    directed_children: BTreeMap<String, BTreeSet<String>>,
    /// Directed adjacency: directed_parents[v] = {w : w → v}.
    directed_parents: BTreeMap<String, BTreeSet<String>>,
    /// Bidirected adjacency: bidirected[v] = {w : v ↔ w}.
    bidirected: BTreeMap<String, BTreeSet<String>>,
}

impl Default for SemiMarkovGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl SemiMarkovGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            node_list: Vec::new(),
            node_set: BTreeSet::new(),
            directed_children: BTreeMap::new(),
            directed_parents: BTreeMap::new(),
            bidirected: BTreeMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Node management
    // -----------------------------------------------------------------------

    /// Add a node if it does not already exist.
    pub fn add_node(&mut self, name: &str) {
        if self.node_set.insert(name.to_owned()) {
            self.node_list.push(name.to_owned());
            self.directed_children.entry(name.to_owned()).or_default();
            self.directed_parents.entry(name.to_owned()).or_default();
            self.bidirected.entry(name.to_owned()).or_default();
        }
    }

    /// Returns `true` if `name` is a node in the graph.
    pub fn has_node(&self, name: &str) -> bool {
        self.node_set.contains(name)
    }

    /// Iterator over all node names (in insertion order).
    pub fn nodes(&self) -> impl Iterator<Item = &String> {
        self.node_list.iter()
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.node_list.len()
    }

    // -----------------------------------------------------------------------
    // Directed edge management
    // -----------------------------------------------------------------------

    /// Add a directed edge `from → to`.
    ///
    /// Both nodes are added automatically if they do not yet exist.
    /// Self-loops are silently ignored.
    pub fn add_directed(&mut self, from: &str, to: &str) {
        if from == to {
            return;
        }
        self.add_node(from);
        self.add_node(to);
        self.directed_children
            .entry(from.to_owned())
            .or_default()
            .insert(to.to_owned());
        self.directed_parents
            .entry(to.to_owned())
            .or_default()
            .insert(from.to_owned());
    }

    /// Remove the directed edge `from → to` if it exists.
    /// Returns `true` if the edge was present.
    pub fn remove_directed(&mut self, from: &str, to: &str) -> bool {
        let removed = self
            .directed_children
            .get_mut(from)
            .map(|set| set.remove(to))
            .unwrap_or(false);
        if removed {
            if let Some(set) = self.directed_parents.get_mut(to) {
                set.remove(from);
            }
        }
        removed
    }

    /// Returns `true` if the directed edge `from → to` exists.
    pub fn has_directed(&self, from: &str, to: &str) -> bool {
        self.directed_children
            .get(from)
            .map(|s| s.contains(to))
            .unwrap_or(false)
    }

    // -----------------------------------------------------------------------
    // Bidirected edge management
    // -----------------------------------------------------------------------

    /// Add a bidirected edge `a ↔ b`.
    ///
    /// Both nodes are added automatically if they do not yet exist.
    /// Self-loops are silently ignored.
    pub fn add_bidirected(&mut self, a: &str, b: &str) {
        if a == b {
            return;
        }
        self.add_node(a);
        self.add_node(b);
        self.bidirected
            .entry(a.to_owned())
            .or_default()
            .insert(b.to_owned());
        self.bidirected
            .entry(b.to_owned())
            .or_default()
            .insert(a.to_owned());
    }

    /// Remove the bidirected edge `a ↔ b` if it exists.
    pub fn remove_bidirected(&mut self, a: &str, b: &str) -> bool {
        let removed = self
            .bidirected
            .get_mut(a)
            .map(|set| set.remove(b))
            .unwrap_or(false);
        if removed {
            if let Some(set) = self.bidirected.get_mut(b) {
                set.remove(a);
            }
        }
        removed
    }

    /// Returns `true` if the bidirected edge `a ↔ b` exists.
    pub fn has_bidirected(&self, a: &str, b: &str) -> bool {
        self.bidirected
            .get(a)
            .map(|s| s.contains(b))
            .unwrap_or(false)
    }

    // -----------------------------------------------------------------------
    // Adjacency queries
    // -----------------------------------------------------------------------

    /// Direct children of `node` via directed edges (v → child).
    pub fn children<'a>(&'a self, node: &str) -> impl Iterator<Item = String> + 'a {
        self.directed_children
            .get(node)
            .into_iter()
            .flat_map(|s| s.iter().cloned())
    }

    /// Direct parents of `node` via directed edges (parent → v).
    pub fn parents<'a>(&'a self, node: &str) -> impl Iterator<Item = String> + 'a {
        self.directed_parents
            .get(node)
            .into_iter()
            .flat_map(|s| s.iter().cloned())
    }

    /// Bidirected neighbors of `node` (v ↔ neighbor).
    pub fn bidirected_neighbors<'a>(&'a self, node: &str) -> impl Iterator<Item = String> + 'a {
        self.bidirected
            .get(node)
            .into_iter()
            .flat_map(|s| s.iter().cloned())
    }

    // -----------------------------------------------------------------------
    // Graph transformation helpers for ID algorithm
    // -----------------------------------------------------------------------

    /// Return a copy of the graph restricted to the given node set.
    ///
    /// Only nodes in `vars` are kept. All directed and bidirected edges
    /// between nodes outside `vars` are discarded.
    pub fn subgraph(&self, vars: &BTreeSet<String>) -> Self {
        let mut g = SemiMarkovGraph::new();
        for v in vars {
            if self.has_node(v) {
                g.add_node(v);
            }
        }
        // Copy directed edges within vars
        for v in vars {
            for child in self.children(v) {
                if vars.contains(&child) {
                    g.add_directed(v, &child);
                }
            }
        }
        // Copy bidirected edges within vars
        for v in vars {
            for nb in self.bidirected_neighbors(v) {
                if vars.contains(&nb) {
                    g.add_bidirected(v, &nb);
                }
            }
        }
        g
    }

    /// Return the "mutilated" graph do(x_vars):
    /// all incoming directed edges to each node in `x_vars` are removed.
    /// Bidirected edges are left intact (they model latent confounders that
    /// do-calculus does not cut unless explicitly stated).
    pub fn mutilate(&self, x_vars: &BTreeSet<String>) -> Self {
        let mut g = self.clone();
        for x in x_vars {
            // Collect parents before mutating
            let parents: Vec<String> = g.parents(x).collect();
            for parent in parents {
                g.remove_directed(&parent, x);
            }
        }
        g
    }

    /// All ancestors of the given nodes (inclusive), following directed edges
    /// backward.
    pub fn ancestors(&self, y: &BTreeSet<String>) -> BTreeSet<String> {
        let mut visited: BTreeSet<String> = BTreeSet::new();
        let mut queue: VecDeque<String> = y.iter().cloned().collect();
        while let Some(node) = queue.pop_front() {
            if visited.insert(node.clone()) {
                for parent in self.parents(&node) {
                    if !visited.contains(&parent) {
                        queue.push_back(parent);
                    }
                }
            }
        }
        visited
    }

    /// All descendants of the given nodes (inclusive), following directed edges
    /// forward.
    pub fn descendants(&self, x: &BTreeSet<String>) -> BTreeSet<String> {
        let mut visited: BTreeSet<String> = BTreeSet::new();
        let mut queue: VecDeque<String> = x.iter().cloned().collect();
        while let Some(node) = queue.pop_front() {
            if visited.insert(node.clone()) {
                for child in self.children(&node) {
                    if !visited.contains(&child) {
                        queue.push_back(child);
                    }
                }
            }
        }
        visited
    }

    /// Return the set of all node names.
    pub fn node_set(&self) -> BTreeSet<String> {
        self.node_set.clone()
    }

    /// Return all directed edges as (from, to) pairs.
    pub fn directed_edges(&self) -> Vec<(String, String)> {
        let mut edges = Vec::new();
        for (from, children) in &self.directed_children {
            for to in children {
                edges.push((from.clone(), to.clone()));
            }
        }
        edges.sort();
        edges
    }

    /// Return all bidirected edges as sorted (a, b) pairs (a < b lexicographically).
    pub fn bidirected_edges(&self) -> Vec<(String, String)> {
        let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
        for (a, neighbors) in &self.bidirected {
            for b in neighbors {
                let pair = if a <= b {
                    (a.clone(), b.clone())
                } else {
                    (b.clone(), a.clone())
                };
                seen.insert(pair);
            }
        }
        seen.into_iter().collect()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_nodes_and_edges() {
        let mut g = SemiMarkovGraph::new();
        g.add_directed("X", "Y");
        g.add_directed("Y", "Z");
        g.add_bidirected("X", "Z");

        assert!(g.has_node("X"));
        assert!(g.has_node("Y"));
        assert!(g.has_node("Z"));
        assert!(g.has_directed("X", "Y"));
        assert!(g.has_directed("Y", "Z"));
        assert!(!g.has_directed("X", "Z"));
        assert!(g.has_bidirected("X", "Z"));
        assert!(g.has_bidirected("Z", "X")); // symmetric
        assert!(!g.has_bidirected("X", "Y"));
    }

    #[test]
    fn test_subgraph() {
        let mut g = SemiMarkovGraph::new();
        g.add_directed("X", "Y");
        g.add_directed("Y", "Z");
        g.add_bidirected("X", "Z");

        let vars: BTreeSet<String> = ["X", "Y"].iter().map(|s| s.to_string()).collect();
        let sub = g.subgraph(&vars);

        assert!(sub.has_node("X"));
        assert!(sub.has_node("Y"));
        assert!(!sub.has_node("Z"));
        assert!(sub.has_directed("X", "Y"));
        assert!(!sub.has_directed("Y", "Z"));
        // Bidirected X ↔ Z is dropped since Z is not in vars
        assert!(!sub.has_bidirected("X", "Z"));
    }

    #[test]
    fn test_mutilate() {
        let mut g = SemiMarkovGraph::new();
        g.add_directed("Z", "X");
        g.add_directed("X", "Y");
        g.add_bidirected("X", "Y");

        let x_vars: BTreeSet<String> = ["X".to_string()].into();
        let m = g.mutilate(&x_vars);

        // Z → X should be removed (incoming edge to X)
        assert!(!m.has_directed("Z", "X"), "Z→X should be cut");
        // X → Y should remain
        assert!(m.has_directed("X", "Y"), "X→Y should remain");
        // Bidirected X ↔ Y should remain
        assert!(m.has_bidirected("X", "Y"), "X↔Y should remain");
    }

    #[test]
    fn test_ancestors() {
        let mut g = SemiMarkovGraph::new();
        g.add_directed("X", "M");
        g.add_directed("M", "Y");

        let y_set: BTreeSet<String> = ["Y".to_string()].into();
        let anc = g.ancestors(&y_set);

        assert!(anc.contains("X"), "X is an ancestor of Y");
        assert!(anc.contains("M"), "M is an ancestor of Y");
        assert!(anc.contains("Y"), "Y is included");
    }

    #[test]
    fn test_descendants() {
        let mut g = SemiMarkovGraph::new();
        g.add_directed("X", "M");
        g.add_directed("M", "Y");

        let x_set: BTreeSet<String> = ["X".to_string()].into();
        let desc = g.descendants(&x_set);

        assert!(desc.contains("X"), "X is included");
        assert!(desc.contains("M"), "M is a descendant");
        assert!(desc.contains("Y"), "Y is a descendant");
    }

    #[test]
    fn test_parents_and_children() {
        let mut g = SemiMarkovGraph::new();
        g.add_directed("Z", "X");
        g.add_directed("X", "Y");

        let parents_x: Vec<String> = g.parents("X").collect();
        assert!(parents_x.contains(&"Z".to_string()));

        let children_x: Vec<String> = g.children("X").collect();
        assert!(children_x.contains(&"Y".to_string()));
    }
}
