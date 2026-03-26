//! Hedge Certificate for Non-Identifiability in Do-Calculus
//!
//! A **hedge** is a witness to non-identifiability of a causal effect
//! P(y | do(x)). Two forests F and F' form a hedge for (x, y) if:
//!
//! 1. Both F and F' are c-forests rooted at a set with non-empty intersection with
//!    ancestors(Y) in G.
//! 2. F' ⊆ F.
//! 3. X intersects F but not F'.
//!
//! # References
//!
//! - Shpitser, I. & Pearl, J. (2006). Identification of Joint Interventional
//!   Distributions in Recursive Semi-Markovian Causal Models. *AAAI 2006*.
//! - Shpitser, I. & Pearl, J. (2006). Identification of Conditional Interventional
//!   Distributions. *UAI 2006*.

use std::collections::{BTreeSet, VecDeque};

use crate::causal::semi_markov_graph::SemiMarkovGraph;

// ---------------------------------------------------------------------------
// HedgeCertificate
// ---------------------------------------------------------------------------

/// A hedge certificate that proves causal non-identifiability.
///
/// When the ID algorithm encounters a hedge, it means that no
/// identification formula can be derived from the observational
/// distribution for the given interventional query.
#[derive(Debug, Clone)]
pub struct HedgeCertificate {
    /// The c-component (connected component in the bidirected-edge subgraph)
    /// that contains nodes from ancestors(Y) but is intersected by X.
    /// This is the "problematic" set S that forms the hedge.
    pub s_component: BTreeSet<String>,

    /// The set of intervention variables X that intersect the c-component,
    /// making it impossible to "cut" the confounding paths.
    pub blocking_x: BTreeSet<String>,

    /// The outcome variable set Y.
    pub outcome_y: BTreeSet<String>,

    /// Human-readable explanation of why the hedge exists.
    pub explanation: String,
}

// ---------------------------------------------------------------------------
// HedgeError
// ---------------------------------------------------------------------------

/// Detailed error returned when the ID algorithm detects a hedge.
///
/// Contains the full certificate plus a precise description of
/// which c-component blocks identification.
#[derive(Debug, Clone)]
pub struct HedgeError {
    /// The hedge certificate.
    pub certificate: HedgeCertificate,
}

impl std::fmt::Display for HedgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Non-identifiable: hedge found in c-component {:?} blocking effect of {:?} on {:?}. {}",
            self.certificate.s_component,
            self.certificate.blocking_x,
            self.certificate.outcome_y,
            self.certificate.explanation,
        )
    }
}

impl std::error::Error for HedgeError {}

// ---------------------------------------------------------------------------
// HedgeFinder
// ---------------------------------------------------------------------------

/// Detects hedge certificates for non-identifiability.
///
/// Implements Theorem 4 of Shpitser & Pearl (AAAI 2006):
/// P(y | do(x)) is NOT identifiable iff there exists a hedge formed by
/// a c-component S in G (or a subgraph thereof) such that:
///
/// - S intersects An(Y) in G
/// - X ∩ S ≠ ∅ (intervention variables appear inside S)
/// - (V \ X) restricted to S's c-component still connects to Y's ancestors
pub struct HedgeFinder;

impl HedgeFinder {
    /// Try to find a hedge certificate demonstrating that P(y | do(x)) is
    /// not identifiable in the graph `graph`.
    ///
    /// Returns `Some(HedgeCertificate)` if a hedge is found (not identifiable),
    /// or `None` if no hedge exists (the effect may be identifiable — the absence
    /// of a hedge alone does not guarantee identifiability; use the full ID
    /// algorithm for that).
    pub fn find(graph: &SemiMarkovGraph, y: &[String], x: &[String]) -> Option<HedgeCertificate> {
        let x_set: BTreeSet<String> = x.iter().cloned().collect();
        let y_set: BTreeSet<String> = y.iter().cloned().collect();

        // Step 1: Compute ancestors of Y in the full graph
        let anc_y = ancestors_of(graph, y);

        // Step 2: Compute c-components of G (based on bidirected edges only)
        let all_vars: BTreeSet<String> = graph.nodes().cloned().collect();
        let components = c_components_in_subgraph(graph, &all_vars);

        // Step 3: For each c-component S, check hedge conditions.
        //
        // A non-trivial hedge requires:
        // A. S has size ≥ 2 (i.e., there is actual bidirected confounding — a singleton
        //    component has no bidirected edges and forms no hedge).
        // B. S has non-empty intersection with An(Y) (the confounding reaches Y's ancestors).
        // C. X ∩ S ≠ ∅ (the intervention variables are "inside" the confounded component,
        //    so cutting them cannot remove the bidirected confounding).
        // D. S also contains at least one node that is an ancestor of Y but NOT in X
        //    (i.e., there's a confounded path from within S to Y that X cannot block).
        for comp in &components {
            // Condition A: component must contain at least 2 nodes (actual bidirected confounding)
            if comp.len() < 2 {
                continue;
            }

            // Condition B: S must have non-empty intersection with An(Y) (including Y itself)
            let intersects_anc_y = comp.iter().any(|v| anc_y.contains(v) || y_set.contains(v));
            if !intersects_anc_y {
                continue;
            }

            // Condition C: X must intersect S
            let x_intersects_s: BTreeSet<String> = comp.intersection(&x_set).cloned().collect();
            if x_intersects_s.is_empty() {
                continue;
            }

            // Condition D: there must be a node in S that is an ancestor of Y (or Y itself)
            // but is NOT entirely covered by X — meaning the confounding path still exists
            let has_unblocked_anc: bool = comp
                .iter()
                .any(|v| (anc_y.contains(v) || y_set.contains(v)) && !x_set.contains(v));
            if !has_unblocked_anc {
                continue;
            }

            // All conditions met: this component forms a hedge.
            // The bidirected confounders inside this component connect X to ancestors of Y,
            // and the intervention on X cannot remove those bidirected paths.
            let explanation = format!(
                "C-component {comp:?} (size {}) contains ancestors of Y and is intersected by \
                 intervention variables {x_intersects_s:?}. The bidirected confounders \
                 inside this component cannot be eliminated by do(X), hence P(y|do(x)) \
                 is not identifiable from the observational distribution.",
                comp.len()
            );
            return Some(HedgeCertificate {
                s_component: comp.clone(),
                blocking_x: x_intersects_s,
                outcome_y: y_set,
                explanation,
            });
        }

        None
    }
}

// ---------------------------------------------------------------------------
// Graph helper functions (also used by id_algorithm)
// ---------------------------------------------------------------------------

/// Compute the set of ancestors of all nodes in `y` (inclusive of y itself).
///
/// Uses BFS traversal following directed edges backwards (i.e., parent edges).
pub fn ancestors_of(graph: &SemiMarkovGraph, y: &[String]) -> BTreeSet<String> {
    let mut visited: BTreeSet<String> = BTreeSet::new();
    let mut queue: VecDeque<String> = y.iter().cloned().collect();

    while let Some(node) = queue.pop_front() {
        if visited.insert(node.clone()) {
            // Add all direct parents
            for parent in graph.parents(&node) {
                if !visited.contains(&parent) {
                    queue.push_back(parent);
                }
            }
        }
    }
    visited
}

/// Compute the topological order of all nodes in the graph.
///
/// Uses Kahn's algorithm (BFS-based topological sort).
/// Returns nodes from roots to leaves.
pub fn topological_order(graph: &SemiMarkovGraph) -> Vec<String> {
    let nodes: Vec<String> = graph.nodes().cloned().collect();
    let n = nodes.len();

    // Build in-degree map
    let mut in_degree: std::collections::HashMap<String, usize> =
        nodes.iter().map(|v| (v.clone(), 0)).collect();

    for node in &nodes {
        for child in graph.children(node) {
            *in_degree.entry(child).or_insert(0) += 1;
        }
    }

    // Initialize queue with zero-in-degree nodes (sorted for determinism)
    let mut queue: VecDeque<String> = nodes
        .iter()
        .filter(|v| in_degree.get(*v).copied().unwrap_or(0) == 0)
        .cloned()
        .collect();

    // Sort for determinism
    let mut queue_sorted: Vec<String> = queue.drain(..).collect();
    queue_sorted.sort();
    let mut queue: VecDeque<String> = queue_sorted.into();

    let mut order = Vec::with_capacity(n);

    while let Some(node) = queue.pop_front() {
        order.push(node.clone());
        let mut children: Vec<String> = graph.children(&node).collect();
        children.sort();
        for child in children {
            let deg = in_degree.entry(child.clone()).or_insert(1);
            *deg = deg.saturating_sub(1);
            if *deg == 0 {
                queue.push_back(child);
            }
        }
    }

    order
}

/// Compute the c-components (bidirected-connected components) of the subgraph
/// restricted to the variables in `vars`.
///
/// A c-component groups variables that are connected via bidirected edges (↔),
/// corresponding to shared latent common causes.
pub fn c_components_in_subgraph(
    graph: &SemiMarkovGraph,
    vars: &BTreeSet<String>,
) -> Vec<BTreeSet<String>> {
    // Union-Find over `vars`
    let var_list: Vec<String> = vars.iter().cloned().collect();
    let n = var_list.len();
    let mut parent: Vec<usize> = (0..n).collect();
    let var_index: std::collections::HashMap<String, usize> = var_list
        .iter()
        .enumerate()
        .map(|(i, v)| (v.clone(), i))
        .collect();

    fn find(parent: &mut Vec<usize>, i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }

    fn union(parent: &mut Vec<usize>, a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }

    // Unite nodes connected by bidirected edges
    for var in vars {
        for neighbor in graph.bidirected_neighbors(var) {
            if vars.contains(&neighbor) {
                if let (Some(&i), Some(&j)) = (var_index.get(var), var_index.get(&neighbor)) {
                    union(&mut parent, i, j);
                }
            }
        }
    }

    // Group by root
    let mut components: std::collections::HashMap<usize, BTreeSet<String>> =
        std::collections::HashMap::new();
    for (i, var) in var_list.iter().enumerate() {
        let root = find(&mut parent, i);
        components.entry(root).or_default().insert(var.clone());
    }

    components.into_values().collect()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::semi_markov_graph::SemiMarkovGraph;

    fn chain_graph() -> SemiMarkovGraph {
        // X → Y → Z, no bidirected edges
        let mut g = SemiMarkovGraph::new();
        g.add_node("X");
        g.add_node("Y");
        g.add_node("Z");
        g.add_directed("X", "Y");
        g.add_directed("Y", "Z");
        g
    }

    fn confounded_graph() -> SemiMarkovGraph {
        // X → Y, X ↔ Y (hidden confounder U → X and U → Y)
        let mut g = SemiMarkovGraph::new();
        g.add_node("X");
        g.add_node("Y");
        g.add_directed("X", "Y");
        g.add_bidirected("X", "Y");
        g
    }

    fn bidirected_chain_graph() -> SemiMarkovGraph {
        // X ↔ Y ↔ Z (fully bidirected-connected)
        let mut g = SemiMarkovGraph::new();
        g.add_node("X");
        g.add_node("Y");
        g.add_node("Z");
        g.add_bidirected("X", "Y");
        g.add_bidirected("Y", "Z");
        g
    }

    // -----------------------------------------------------------------------
    // c_components tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_c_components_chain_no_bidirected() {
        let g = chain_graph();
        let vars: BTreeSet<String> = ["X", "Y", "Z"].iter().map(|s| s.to_string()).collect();
        let comps = c_components_in_subgraph(&g, &vars);
        // No bidirected edges → each node is its own singleton component
        assert_eq!(
            comps.len(),
            3,
            "Expected 3 singleton components, got {}",
            comps.len()
        );
        for comp in &comps {
            assert_eq!(
                comp.len(),
                1,
                "Each component should be a singleton, got {:?}",
                comp
            );
        }
    }

    #[test]
    fn test_c_components_fully_bidirected() {
        let g = bidirected_chain_graph();
        let vars: BTreeSet<String> = ["X", "Y", "Z"].iter().map(|s| s.to_string()).collect();
        let comps = c_components_in_subgraph(&g, &vars);
        // X ↔ Y ↔ Z → all in one component
        assert_eq!(comps.len(), 1, "Expected 1 component, got {}", comps.len());
        let comp = &comps[0];
        assert_eq!(
            comp.len(),
            3,
            "Component should contain all 3 nodes, got {:?}",
            comp
        );
    }

    #[test]
    fn test_c_components_partial_bidirected() {
        let g = confounded_graph();
        let vars: BTreeSet<String> = ["X", "Y"].iter().map(|s| s.to_string()).collect();
        let comps = c_components_in_subgraph(&g, &vars);
        // X ↔ Y → one component containing both
        assert_eq!(comps.len(), 1, "Expected 1 component, got {}", comps.len());
        assert!(comps[0].contains("X") && comps[0].contains("Y"));
    }

    // -----------------------------------------------------------------------
    // ancestors_of tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ancestors_of_chain() {
        let g = chain_graph();
        let anc = ancestors_of(&g, &["Z".to_string()]);
        // ancestors(Z) should be {X, Y, Z}
        assert!(anc.contains("X"), "X should be an ancestor of Z");
        assert!(anc.contains("Y"), "Y should be an ancestor of Z");
        assert!(anc.contains("Z"), "Z should be included");
    }

    #[test]
    fn test_ancestors_of_root() {
        let g = chain_graph();
        let anc = ancestors_of(&g, &["X".to_string()]);
        // X has no parents; ancestors = {X}
        assert_eq!(anc.len(), 1);
        assert!(anc.contains("X"));
    }

    // -----------------------------------------------------------------------
    // topological_order tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_topological_order_chain() {
        let g = chain_graph();
        let order = topological_order(&g);
        assert_eq!(order.len(), 3, "Order should have 3 elements");
        // X must come before Y, Y before Z
        let x_pos = order.iter().position(|v| v == "X").expect("X not in order");
        let y_pos = order.iter().position(|v| v == "Y").expect("Y not in order");
        let z_pos = order.iter().position(|v| v == "Z").expect("Z not in order");
        assert!(x_pos < y_pos, "X must precede Y in topological order");
        assert!(y_pos < z_pos, "Y must precede Z in topological order");
    }

    // -----------------------------------------------------------------------
    // HedgeFinder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hedge_finder_identifiable_chain() {
        // X → Y → Z with no confounders — P(Z | do(X)) is identifiable
        let g = chain_graph();
        let cert = HedgeFinder::find(&g, &["Z".to_string()], &["X".to_string()]);
        // No bidirected edges into ancestors of Z from X → no hedge
        assert!(
            cert.is_none(),
            "Chain with no bidirected edges should have no hedge"
        );
    }

    #[test]
    fn test_hedge_finder_confounded_not_identifiable() {
        // X → Y, X ↔ Y — P(Y | do(X)) is NOT identifiable
        // The c-component {X, Y} intersects both An(Y) and the intervention set {X}
        let g = confounded_graph();
        let cert = HedgeFinder::find(&g, &["Y".to_string()], &["X".to_string()]);
        assert!(
            cert.is_some(),
            "Confounded graph X↔Y should produce a hedge certificate"
        );
        let cert = cert.expect("certificate");
        assert!(
            cert.s_component.contains("X") || cert.s_component.contains("Y"),
            "Hedge component should include X or Y: {:?}",
            cert.s_component
        );
        assert!(
            cert.blocking_x.contains("X"),
            "Blocking X should include X: {:?}",
            cert.blocking_x
        );
    }
}
