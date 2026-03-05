//! Directed Acyclic Graph (DAG) for Bayesian Networks.
//!
//! Provides a lightweight, index-based DAG with:
//! - Cycle detection on edge insertion
//! - Topological sort (Kahn's algorithm)
//! - D-separation via the Bayes-ball algorithm
//! - Moral graph construction
//! - Markov blanket computation
//! - V-structure enumeration

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use crate::StatsError;

/// Directed Acyclic Graph for Bayesian network structure.
///
/// Nodes are identified by integer indices. Names are optional but stored
/// for display/debugging purposes.
#[derive(Debug, Clone)]
pub struct DAG {
    /// Number of nodes.
    pub n_nodes: usize,
    /// `parents[i]` = sorted list of parent indices for node i.
    pub parents: Vec<Vec<usize>>,
    /// `children[i]` = sorted list of child indices for node i.
    pub children: Vec<Vec<usize>>,
    /// Human-readable node names (may be empty).
    pub node_names: Vec<String>,
}

impl DAG {
    /// Create a new DAG with `n` nodes and no edges.
    ///
    /// Node names default to `"X_0"`, `"X_1"`, …
    pub fn new(n: usize) -> Self {
        let node_names = (0..n).map(|i| format!("X_{i}")).collect();
        Self {
            n_nodes: n,
            parents: vec![Vec::new(); n],
            children: vec![Vec::new(); n],
            node_names,
        }
    }

    /// Create a DAG with explicit node names.
    pub fn with_names(names: Vec<String>) -> Self {
        let n = names.len();
        Self {
            n_nodes: n,
            parents: vec![Vec::new(); n],
            children: vec![Vec::new(); n],
            node_names: names,
        }
    }

    /// Add a directed edge `from → to`.
    ///
    /// Returns `Err` if the edge would create a cycle or is a self-loop.
    pub fn add_edge(&mut self, from: usize, to: usize) -> Result<(), StatsError> {
        if from >= self.n_nodes || to >= self.n_nodes {
            return Err(StatsError::InvalidInput(format!(
                "Node index out of range: from={from}, to={to}, n={}", self.n_nodes
            )));
        }
        if from == to {
            return Err(StatsError::InvalidInput(
                "Self-loops are not allowed in a DAG".to_string(),
            ));
        }
        // Check if edge already exists
        if self.parents[to].contains(&from) {
            return Ok(()); // idempotent
        }
        // Cycle check: can we reach `from` from `to`?
        if self.can_reach(to, from) {
            return Err(StatsError::InvalidInput(format!(
                "Adding edge {from}→{to} would create a cycle"
            )));
        }
        // Insert in sorted order for determinism
        let pos = self.parents[to].partition_point(|&p| p < from);
        self.parents[to].insert(pos, from);
        let pos = self.children[from].partition_point(|&c| c < to);
        self.children[from].insert(pos, to);
        Ok(())
    }

    /// Remove an edge `from → to` if it exists.
    pub fn remove_edge(&mut self, from: usize, to: usize) -> bool {
        let before = self.parents[to].len();
        self.parents[to].retain(|&p| p != from);
        self.children[from].retain(|&c| c != to);
        self.parents[to].len() < before
    }

    /// Returns `true` if the graph is acyclic (always true if only `add_edge` was used).
    pub fn is_dag(&self) -> bool {
        self.topological_sort_full().is_some()
    }

    /// Topological sort via Kahn's algorithm. Returns nodes in topological order.
    pub fn topological_sort(&self) -> Vec<usize> {
        self.topological_sort_full().unwrap_or_default()
    }

    /// Internal: returns `None` if a cycle is detected.
    fn topological_sort_full(&self) -> Option<Vec<usize>> {
        let mut in_degree: Vec<usize> = self.parents.iter().map(|p| p.len()).collect();
        let mut queue: VecDeque<usize> = (0..self.n_nodes)
            .filter(|&i| in_degree[i] == 0)
            .collect();
        let mut order = Vec::with_capacity(self.n_nodes);
        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &child in &self.children[node] {
                in_degree[child] -= 1;
                if in_degree[child] == 0 {
                    queue.push_back(child);
                }
            }
        }
        if order.len() == self.n_nodes {
            Some(order)
        } else {
            None // cycle detected
        }
    }

    /// Check whether node `target` is reachable from `start` via directed edges.
    fn can_reach(&self, start: usize, target: usize) -> bool {
        let mut visited = vec![false; self.n_nodes];
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            if node == target {
                return true;
            }
            if visited[node] {
                continue;
            }
            visited[node] = true;
            for &child in &self.children[node] {
                stack.push(child);
            }
        }
        false
    }

    /// D-separation test: is X ⊥ Y | Z (conditioned on the set Z)?
    ///
    /// Uses the Bayes-ball algorithm:
    /// - In a chain A→B→C or fork A←B→C, B blocks the path when observed.
    /// - In a v-structure A→B←C (collider), B *opens* a path when B or any
    ///   of its descendants is observed.
    pub fn d_separation(&self, x: usize, y: usize, z: &[usize]) -> bool {
        // D-separation via the Bayes Ball algorithm (Shachter 1998).
        //
        // Returns true iff X ⊥ Y | Z  (no active path from X to Y given Z).
        //
        // Key rules for the ball traversal at node n:
        //   Ball coming UP from child:
        //     n NOT in Z → pass to parents (up) and children (down)
        //     n IN Z     → blocked (do not propagate)
        //   Ball coming DOWN from parent:
        //     n NOT in Z → pass to children (down)
        //     n IN Z or has descendant in Z → activate collider: pass to parents (up)
        //     n IN Z     → also blocked for downstream (do not pass to children)
        let z_set: HashSet<usize> = z.iter().copied().collect();

        // Pre-compute: the set of all nodes that are in Z or are ancestors of Z,
        // which is equivalent to "has a descendant (or self) in Z" for collider activation.
        let mut z_or_ancestor_of_z: HashSet<usize> = z_set.clone();
        for &zn in &z_set {
            self.collect_ancestors_into(zn, &mut z_or_ancestor_of_z);
        }

        // BFS over (node, direction) pairs.
        // direction: true = "going up (came from child)", false = "going down (came from parent)"
        let mut visited: HashSet<(usize, bool)> = HashSet::new();
        let mut queue: VecDeque<(usize, bool)> = VecDeque::new();

        // Start from X going in both directions
        queue.push_back((x, true));
        queue.push_back((x, false));

        while let Some((node, going_up)) = queue.pop_front() {
            if node == y {
                return false; // active path found
            }
            if visited.contains(&(node, going_up)) {
                continue;
            }
            visited.insert((node, going_up));

            if going_up {
                // Ball came from a child (traveling toward parents)
                if !z_set.contains(&node) {
                    // Non-observed: pass through
                    for &parent in &self.parents[node] {
                        queue.push_back((parent, true));
                    }
                    for &child in &self.children[node] {
                        queue.push_back((child, false));
                    }
                }
                // If node IS in Z: blocked (do nothing)
            } else {
                // Ball came from a parent (traveling toward children)
                if !z_set.contains(&node) {
                    // Non-observed: continue downstream
                    for &child in &self.children[node] {
                        queue.push_back((child, false));
                    }
                }
                // Collider activation: if node is in Z (or has a descendant in Z),
                // the ball bounces back UP through parents.
                if z_or_ancestor_of_z.contains(&node) {
                    for &parent in &self.parents[node] {
                        queue.push_back((parent, true));
                    }
                }
            }
        }
        true // no active path found → d-separated
    }

    /// Collect all ancestors of `node` into `set` (excluding `node` itself).
    fn collect_ancestors_into(&self, node: usize, set: &mut HashSet<usize>) {
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            for &parent in &self.parents[n] {
                if set.insert(parent) {
                    stack.push(parent);
                }
            }
        }
    }

    /// Collect all descendants of `node` into `set`.
    fn collect_descendants(&self, node: usize, set: &mut HashSet<usize>) {
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            for &child in &self.children[n] {
                if set.insert(child) {
                    stack.push(child);
                }
            }
        }
    }

    /// Collect all ancestors of `node` (exclusive).
    pub fn ancestors(&self, node: usize) -> HashSet<usize> {
        let mut set = HashSet::new();
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            for &parent in &self.parents[n] {
                if set.insert(parent) {
                    stack.push(parent);
                }
            }
        }
        set
    }

    /// Collect all descendants of `node` (exclusive).
    pub fn descendants(&self, node: usize) -> HashSet<usize> {
        let mut set = HashSet::new();
        self.collect_descendants(node, &mut set);
        set
    }

    /// Construct the moral graph: undirected graph where we connect all pairs
    /// of parents for every node ("marrying" parents).
    ///
    /// Returns an adjacency matrix (symmetric boolean n×n).
    pub fn moral_graph(&self) -> Vec<Vec<bool>> {
        let n = self.n_nodes;
        let mut adj = vec![vec![false; n]; n];
        // Add undirected versions of all directed edges
        for node in 0..n {
            for &parent in &self.parents[node] {
                adj[node][parent] = true;
                adj[parent][node] = true;
            }
            // Connect all pairs of parents (marry them)
            let parents = &self.parents[node];
            for (i, &p1) in parents.iter().enumerate() {
                for &p2 in &parents[(i + 1)..] {
                    adj[p1][p2] = true;
                    adj[p2][p1] = true;
                }
            }
        }
        adj
    }

    /// Markov blanket of `node`: parents ∪ children ∪ other parents of children.
    pub fn markov_blanket(&self, node: usize) -> Vec<usize> {
        let mut blanket: BTreeSet<usize> = BTreeSet::new();
        // Parents
        for &p in &self.parents[node] {
            blanket.insert(p);
        }
        // Children and their other parents
        for &child in &self.children[node] {
            blanket.insert(child);
            for &co_parent in &self.parents[child] {
                if co_parent != node {
                    blanket.insert(co_parent);
                }
            }
        }
        blanket.into_iter().collect()
    }

    /// Enumerate all v-structures (colliders): triples (parent1, child, parent2)
    /// where parent1 → child ← parent2 and parent1 and parent2 are NOT adjacent.
    pub fn v_structures(&self) -> Vec<(usize, usize, usize)> {
        let mut result = Vec::new();
        for node in 0..self.n_nodes {
            let parents = &self.parents[node];
            for (i, &p1) in parents.iter().enumerate() {
                for &p2 in &parents[(i + 1)..] {
                    // Check that p1 and p2 are NOT adjacent (no directed edge either way)
                    if !self.children[p1].contains(&p2) && !self.children[p2].contains(&p1) {
                        result.push((p1, node, p2));
                    }
                }
            }
        }
        result
    }

    /// Return the index of a node by name (if names are set).
    pub fn node_index(&self, name: &str) -> Option<usize> {
        self.node_names.iter().position(|n| n == name)
    }

    /// Number of edges in the DAG.
    pub fn n_edges(&self) -> usize {
        self.children.iter().map(|c| c.len()).sum()
    }

    /// Check if there is a directed edge from → to.
    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        self.children[from].contains(&to)
    }

    /// Build a node-name → index lookup map.
    pub fn name_map(&self) -> HashMap<&str, usize> {
        self.node_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn chain_dag() -> DAG {
        // 0 → 1 → 2
        let mut dag = DAG::new(3);
        dag.add_edge(0, 1).unwrap();
        dag.add_edge(1, 2).unwrap();
        dag
    }

    fn fork_dag() -> DAG {
        // 1 ← 0 → 2
        let mut dag = DAG::new(3);
        dag.add_edge(0, 1).unwrap();
        dag.add_edge(0, 2).unwrap();
        dag
    }

    fn collider_dag() -> DAG {
        // 0 → 2 ← 1
        let mut dag = DAG::new(3);
        dag.add_edge(0, 2).unwrap();
        dag.add_edge(1, 2).unwrap();
        dag
    }

    #[test]
    fn test_add_edge_cycle() {
        let mut dag = DAG::new(3);
        dag.add_edge(0, 1).unwrap();
        dag.add_edge(1, 2).unwrap();
        assert!(dag.add_edge(2, 0).is_err(), "cycle should be rejected");
    }

    #[test]
    fn test_add_edge_self_loop() {
        let mut dag = DAG::new(2);
        assert!(dag.add_edge(0, 0).is_err());
    }

    #[test]
    fn test_topological_sort_chain() {
        let dag = chain_dag();
        let order = dag.topological_sort();
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn test_is_dag() {
        let dag = chain_dag();
        assert!(dag.is_dag());
    }

    #[test]
    fn test_d_separation_chain() {
        let dag = chain_dag();
        // 0 ⊥ 2 | 1 in a chain
        assert!(dag.d_separation(0, 2, &[1]));
        // Not d-separated without conditioning
        assert!(!dag.d_separation(0, 2, &[]));
    }

    #[test]
    fn test_d_separation_fork() {
        let dag = fork_dag();
        // 1 ⊥ 2 | 0 in a fork
        assert!(dag.d_separation(1, 2, &[0]));
        // Not d-separated without conditioning
        assert!(!dag.d_separation(1, 2, &[]));
    }

    #[test]
    fn test_d_separation_collider() {
        let dag = collider_dag();
        // 0 ⊥ 1 marginally (collider at 2)
        assert!(dag.d_separation(0, 1, &[]));
        // Conditioning on collider opens the path
        assert!(!dag.d_separation(0, 1, &[2]));
    }

    #[test]
    fn test_moral_graph() {
        let dag = collider_dag();
        // Parents of node 2 are 0 and 1; they should be connected in moral graph
        let moral = dag.moral_graph();
        assert!(moral[0][1], "0 and 1 should be connected in moral graph");
        assert!(moral[1][0]);
    }

    #[test]
    fn test_markov_blanket() {
        // 0 → 2 ← 1 → 3
        let mut dag = DAG::new(4);
        dag.add_edge(0, 2).unwrap();
        dag.add_edge(1, 2).unwrap();
        dag.add_edge(1, 3).unwrap();
        // MB(2): parents={0,1}, children={}, co-parents={}
        let mb2 = dag.markov_blanket(2);
        assert!(mb2.contains(&0));
        assert!(mb2.contains(&1));
        // MB(1): parents={}, children={2,3}, co-parents of 2={0}
        let mb1 = dag.markov_blanket(1);
        assert!(mb1.contains(&2));
        assert!(mb1.contains(&3));
        assert!(mb1.contains(&0));
    }

    #[test]
    fn test_v_structures() {
        let dag = collider_dag();
        let vs = dag.v_structures();
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0], (0, 2, 1));
    }

    #[test]
    fn test_ancestors_descendants() {
        let dag = chain_dag();
        let anc = dag.ancestors(2);
        assert!(anc.contains(&0));
        assert!(anc.contains(&1));
        let desc = dag.descendants(0);
        assert!(desc.contains(&1));
        assert!(desc.contains(&2));
    }

    #[test]
    fn test_remove_edge() {
        let mut dag = chain_dag();
        assert!(dag.remove_edge(0, 1));
        assert!(!dag.has_edge(0, 1));
        assert!(!dag.parents[1].contains(&0));
    }
}
