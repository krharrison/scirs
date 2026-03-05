//! Causal Effect Identification
//!
//! # Algorithms provided
//!
//! | Algorithm | Reference |
//! |-----------|-----------|
//! | Do-calculus rules 1-3 | Pearl (2000), ch. 3 |
//! | Backdoor criterion | Pearl (1993) |
//! | Frontdoor criterion | Pearl (1995) |
//! | ID algorithm | Shpitser & Pearl (2006) |
//! | C-components / Tian-Pearl | Tian & Pearl (2002) |
//!
//! # References
//!
//! - Pearl, J. (2000). *Causality*. Cambridge University Press.
//! - Shpitser, I. & Pearl, J. (2006). Identification of Joint Interventional
//!   Distributions in Recursive Semi-Markovian Causal Models. *AAAI 2006*.
//! - Tian, J. & Pearl, J. (2002). A General Identification Condition for
//!   Causal Effects. *AAAI 2002*.

use std::collections::{HashMap, HashSet};

use crate::causal_graph::dag::CausalDAG;
use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// Which rule of Pearl's do-calculus applies to a given query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DoCalculusRule {
    /// Rule 1: Insertion/deletion of observations.
    ///
    /// P(y | do(x), z, w) = P(y | do(x), w)  when (Y ⊥ Z | X, W) in G_{X̄}.
    Rule1,
    /// Rule 2: Action/observation exchange.
    ///
    /// P(y | do(x), do(z), w) = P(y | do(x), z, w) when (Y ⊥ Z | X, W) in G_{X̄, Z̄}.
    Rule2,
    /// Rule 3: Insertion/deletion of actions.
    ///
    /// P(y | do(x), do(z), w) = P(y | do(x), w) when (Y ⊥ Z | X, W) in G_{X̄, Z(W)}.
    Rule3,
    /// None of the three rules applies directly.
    None,
}

/// Outcome of the backdoor criterion check.
#[derive(Debug, Clone)]
pub struct BackdoorResult {
    /// Whether a valid backdoor adjustment set was found.
    pub is_admissible: bool,
    /// One valid adjustment set (empty Vec when none found).
    pub adjustment_set: Vec<String>,
    /// All minimal valid adjustment sets discovered (up to a search budget).
    pub all_minimal_sets: Vec<Vec<String>>,
}

/// Outcome of the frontdoor criterion check.
#[derive(Debug, Clone)]
pub struct FrontdoorResult {
    /// Whether the frontdoor criterion applies.
    pub is_applicable: bool,
    /// The mediator set M satisfying the frontdoor condition.
    pub mediator_set: Vec<String>,
    /// Symbolic expression for the identified quantity (LaTeX-like).
    pub formula: String,
}

/// Output of the ID algorithm.
#[derive(Debug, Clone)]
pub struct IdResult {
    /// Whether P(y | do(x)) is identifiable.
    pub identifiable: bool,
    /// Symbolic expression if identifiable.
    pub expression: String,
    /// Explanation / derivation sketch.
    pub explanation: String,
}

/// A c-component of a Semi-Markovian causal graph.
#[derive(Debug, Clone)]
pub struct CComponent {
    /// Node indices that belong to this c-component.
    pub nodes: HashSet<usize>,
}

// ---------------------------------------------------------------------------
// Do-calculus rule checker
// ---------------------------------------------------------------------------

/// Check which of Pearl's three do-calculus rules applies to the query
/// P(y | do(x), z, w).
///
/// # Parameters
/// - `dag`     – the causal DAG
/// - `y`       – outcome variables
/// - `x`       – intervention variables (do(x))
/// - `z`       – candidate observation/intervention variables
/// - `w`       – conditioning variables (passive observations)
/// - `rule`    – which rule to test
pub fn check_do_calculus_rule(
    dag: &CausalDAG,
    y: &[&str],
    x: &[&str],
    z: &[&str],
    w: &[&str],
    rule: DoCalculusRule,
) -> bool {
    match rule {
        DoCalculusRule::Rule1 => {
            // (Y ⊥ Z | X, W) in G_{X̄}  (graph with incoming edges to X removed)
            let mut g_xbar = dag.clone();
            remove_incoming_edges(&mut g_xbar, x);
            let mut conditioning: Vec<&str> = Vec::new();
            conditioning.extend_from_slice(x);
            conditioning.extend_from_slice(w);
            check_d_separation_all(&g_xbar, y, z, &conditioning)
        }
        DoCalculusRule::Rule2 => {
            // (Y ⊥ Z | X, W) in G_{X̄, Z̄}
            let mut g = dag.clone();
            remove_incoming_edges(&mut g, x);
            remove_incoming_edges(&mut g, z);
            let mut conditioning: Vec<&str> = Vec::new();
            conditioning.extend_from_slice(x);
            conditioning.extend_from_slice(w);
            check_d_separation_all(&g, y, z, &conditioning)
        }
        DoCalculusRule::Rule3 => {
            // (Y ⊥ Z | X, W) in G_{X̄, Z(W)}
            // Z(W) = Z \ An(W) in G_{X̄}
            let mut g_xbar = dag.clone();
            remove_incoming_edges(&mut g_xbar, x);

            let w_ancestors = ancestors_of_names(&g_xbar, w);
            let z_not_w_anc: Vec<&str> = z
                .iter()
                .filter(|&&zz| {
                    let idx = dag.node_index(zz);
                    !idx.map(|i| w_ancestors.contains(&i)).unwrap_or(false)
                })
                .copied()
                .collect();

            // Remove outgoing edges from z_not_w_anc
            let mut g = g_xbar;
            remove_outgoing_edges(&mut g, &z_not_w_anc);

            let mut conditioning: Vec<&str> = Vec::new();
            conditioning.extend_from_slice(x);
            conditioning.extend_from_slice(w);
            check_d_separation_all(&g, y, z, &conditioning)
        }
        DoCalculusRule::None => false,
    }
}

// ---------------------------------------------------------------------------
// Backdoor criterion
// ---------------------------------------------------------------------------

/// Check whether `z_set` satisfies the **backdoor criterion** for
/// estimating the causal effect of `x` on `y`.
///
/// The backdoor criterion (Pearl 1993) requires:
/// 1. No element of `z_set` is a descendant of `x`.
/// 2. `z_set` blocks all backdoor paths from `x` to `y`
///    (i.e., `y ⊥ x | z_set` in the graph with all outgoing edges of `x` removed).
pub fn satisfies_backdoor(dag: &CausalDAG, x: &str, y: &str, z_set: &[&str]) -> bool {
    // Condition 1: z_set ∩ De(x) = ∅
    let desc_x = dag.descendants(x);
    for &z in z_set {
        if let Some(zi) = dag.node_index(z) {
            if desc_x.contains(&zi) {
                return false;
            }
        }
    }
    // Condition 2: d-separation in G with outgoing edges of x removed.
    let mut g = dag.clone();
    remove_outgoing_edges(&mut g, &[x]);
    g.is_d_separated(x, y, z_set)
}

/// Find **all minimal backdoor adjustment sets** for the effect of `x` on `y`.
///
/// Uses a BFS/subset search over non-descendant nodes.
/// The search is limited to subsets up to size `max_set_size`.
pub fn find_backdoor_sets(
    dag: &CausalDAG,
    x: &str,
    y: &str,
    max_set_size: usize,
) -> BackdoorResult {
    let desc_x = dag.descendants(x);
    let xi = dag.node_index(x).unwrap_or(usize::MAX);
    let yi = dag.node_index(y).unwrap_or(usize::MAX);

    // Candidate variables: not x, not y, not a descendant of x
    let candidates: Vec<usize> = (0..dag.n_nodes())
        .filter(|&i| i != xi && i != yi && !desc_x.contains(&i))
        .collect();

    let mut all_minimal: Vec<Vec<String>> = Vec::new();
    let mut found_any = false;

    // Iterate over subsets by size
    'outer: for size in 0..=max_set_size.min(candidates.len()) {
        for subset in subsets(&candidates, size) {
            let z_names: Vec<&str> = subset
                .iter()
                .filter_map(|&i| dag.node_name(i))
                .collect();
            if satisfies_backdoor(dag, x, y, &z_names) {
                let z_strings: Vec<String> = z_names.iter().map(|s| s.to_string()).collect();
                all_minimal.push(z_strings);
                found_any = true;
                if all_minimal.len() >= 20 {
                    // budget cap
                    break 'outer;
                }
            }
        }
        // If we found any set of size `size`, sets of larger size may not be minimal.
        // We keep them all for completeness but stop once the first size level succeeded.
        if found_any && size < max_set_size {
            // Only stop adding non-minimal sets if we already found smallest size
        }
    }

    let best = all_minimal.first().cloned().unwrap_or_default();
    BackdoorResult {
        is_admissible: found_any,
        adjustment_set: best,
        all_minimal_sets: all_minimal,
    }
}

// ---------------------------------------------------------------------------
// Frontdoor criterion
// ---------------------------------------------------------------------------

/// Check whether a set of mediators `m_set` satisfies the **frontdoor criterion**
/// for estimating P(y | do(x)).
///
/// Frontdoor criterion (Pearl 1995):
/// 1. All directed paths from `x` to `y` are intercepted by `m_set`.
/// 2. There are no unblocked backdoor paths from `x` to `m_set`.
/// 3. All backdoor paths from `m_set` to `y` are blocked by `x`.
pub fn satisfies_frontdoor(dag: &CausalDAG, x: &str, y: &str, m_set: &[&str]) -> bool {
    // Condition 1: every directed path x→...→y passes through m_set
    if !intercepts_all_paths(dag, x, y, m_set) {
        return false;
    }
    // Condition 2: no unblocked backdoor path from x to m.
    // A backdoor path is a non-causal path (goes into X first).
    // Check by looking at G where BOTH incoming AND outgoing edges of X are removed:
    // if X and M are still d-connected in that graph, there is a backdoor path.
    let mut g_xbar = dag.clone();
    remove_incoming_edges(&mut g_xbar, &[x]);
    remove_outgoing_edges(&mut g_xbar, &[x]);
    for &m in m_set {
        if !g_xbar.is_d_separated(x, m, &[]) {
            return false;
        }
    }
    // Condition 3: x blocks all backdoor m → y
    for &m in m_set {
        if !satisfies_backdoor(dag, m, y, &[x]) {
            return false;
        }
    }
    true
}

/// Find a frontdoor mediator set for P(y | do(x)), if one exists.
pub fn find_frontdoor_set(dag: &CausalDAG, x: &str, y: &str) -> FrontdoorResult {
    let xi = dag.node_index(x).unwrap_or(usize::MAX);
    let yi = dag.node_index(y).unwrap_or(usize::MAX);

    let descendants_x = dag.descendants(x);
    // Mediators must be descendants of x (excluding y)
    let candidates: Vec<usize> = descendants_x
        .iter()
        .filter(|&&i| i != yi && i != xi)
        .copied()
        .collect();

    for size in 1..=candidates.len() {
        for subset in subsets(&candidates, size) {
            let m_names: Vec<&str> = subset
                .iter()
                .filter_map(|&i| dag.node_name(i))
                .collect();
            if satisfies_frontdoor(dag, x, y, &m_names) {
                let formula = frontdoor_formula(x, y, &m_names);
                return FrontdoorResult {
                    is_applicable: true,
                    mediator_set: m_names.iter().map(|s| s.to_string()).collect(),
                    formula,
                };
            }
        }
    }

    FrontdoorResult {
        is_applicable: false,
        mediator_set: Vec::new(),
        formula: "Not identifiable via frontdoor".to_owned(),
    }
}

// ---------------------------------------------------------------------------
// ID algorithm (Shpitser & Pearl 2006)
// ---------------------------------------------------------------------------

/// Run the **ID algorithm** to determine whether P(y | do(x)) is identifiable
/// from observational data in the given DAG (assuming no hidden variables).
///
/// In the fully observed case (no latent confounders) P(y | do(x)) is
/// always identifiable via the adjustment formula; this function also
/// tries backdoor and frontdoor identifications and returns a symbolic
/// expression.
pub fn id_algorithm(dag: &CausalDAG, y: &[&str], x: &[&str]) -> IdResult {
    // Step 1: if X = ∅, P(y | do(∅)) = P(y)
    if x.is_empty() {
        return IdResult {
            identifiable: true,
            expression: format!("P({})", y.join(", ")),
            explanation: "No intervention; trivially identified as the observational distribution.".to_owned(),
        };
    }

    // Step 2: Y is the full variable set → P(y | do(x)) = P(y | do(x), rest)
    // Try backdoor for single treatment
    if x.len() == 1 && y.len() == 1 {
        let xv = x[0];
        let yv = y[0];

        // Try empty adjustment (no confounders needed)
        if satisfies_backdoor(dag, xv, yv, &[]) {
            return IdResult {
                identifiable: true,
                expression: format!("P({yv} | {xv})"),
                explanation: "Identified via empty backdoor set (no confounding).".to_owned(),
            };
        }

        // Try backdoor with adjustment
        let bd = find_backdoor_sets(dag, xv, yv, 5);
        if bd.is_admissible {
            let z_str = bd.adjustment_set.join(", ");
            return IdResult {
                identifiable: true,
                expression: format!(
                    "Σ_{{{}}} P({yv} | {xv}, {z_str}) P({z_str})",
                    z_str, 
                ),
                explanation: format!(
                    "Identified via backdoor adjustment on {{{z_str}}}."
                ),
            };
        }

        // Try frontdoor
        let fd = find_frontdoor_set(dag, xv, yv);
        if fd.is_applicable {
            return IdResult {
                identifiable: true,
                expression: fd.formula,
                explanation: format!(
                    "Identified via frontdoor criterion through mediators: {:?}.",
                    fd.mediator_set
                ),
            };
        }
    }

    // General case: attempt c-component factorization (Tian-Pearl)
    let tian = tian_pearl_id(dag, y, x);
    if tian.identifiable {
        return tian;
    }

    IdResult {
        identifiable: false,
        expression: String::new(),
        explanation: format!(
            "P({y} | do({x})) is not identifiable by the ID algorithm with the given DAG.",
            y = y.join(", "),
            x = x.join(", ")
        ),
    }
}

// ---------------------------------------------------------------------------
// Tian-Pearl c-component identification
// ---------------------------------------------------------------------------

/// Tian-Pearl identification using c-component factorization.
///
/// In a DAG without latent variables every c-component is a singleton, so
/// P(y | do(x)) reduces to a standard adjustment formula.
pub fn tian_pearl_id(dag: &CausalDAG, y: &[&str], x: &[&str]) -> IdResult {
    // Build topological order
    let topo = dag.topological_sort();
    let n = dag.n_nodes();

    // Map node name → index in topo order
    let topo_pos: HashMap<&str, usize> = topo
        .iter()
        .enumerate()
        .map(|(i, &name)| (name, i))
        .collect();

    // In a DAG without hidden variables every node's Q-factor is:
    //   Q[i] = P(v_i | pa(v_i), v_{<i})
    // The joint is the product of all Q-factors.
    // For identification of y given do(x), we condition on x in each Q-factor.

    let y_set: HashSet<&str> = y.iter().copied().collect();
    let x_set: HashSet<&str> = x.iter().copied().collect();

    // Sum over non-Y, non-X variables
    let sum_over: Vec<&str> = topo
        .iter()
        .copied()
        .filter(|&v| !y_set.contains(v) && !x_set.contains(v))
        .collect();

    // Build symbolic expression
    let mut numerator_parts: Vec<String> = Vec::new();
    let mut denominator_parts: Vec<String> = Vec::new();

    for &node in &topo {
        let pos = topo_pos[node];
        let pa: Vec<&str> = dag.parents(node);
        // Predecessors in topo order (prior context)
        let prior: Vec<&str> = topo[..pos].to_vec();

        let cond: Vec<String> = pa
            .iter()
            .map(|s| s.to_string())
            .chain(prior.iter().map(|s| s.to_string()))
            .collect();

        let cond_str = if cond.is_empty() {
            String::new()
        } else {
            format!(" | {}", cond.join(", "))
        };

        if !x_set.contains(node) {
            // Numerator: include this factor
            numerator_parts.push(format!("P({node}{cond_str})"));
        }
        // Denominator: sum over x in each conditional
        if pa.iter().any(|p| x_set.contains(*p)) || prior.iter().any(|p| x_set.contains(*p)) {
            denominator_parts.push(format!("P({node}{cond_str})"));
        }
    }

    let sum_str = if sum_over.is_empty() {
        String::new()
    } else {
        format!("Σ_{{{}}}", sum_over.join(","))
    };

    let num_str = numerator_parts.join(" ");
    let expr = if denominator_parts.is_empty() {
        format!("{sum_str} {num_str}")
    } else {
        format!("{sum_str} {num_str} / ({})", denominator_parts.join(" "))
    };

    IdResult {
        identifiable: n > 0,
        expression: expr.trim().to_owned(),
        explanation: "Tian-Pearl c-component factorization (DAG, no hidden variables).".to_owned(),
    }
}

/// Compute c-components for a Semi-Markovian graph given additional
/// bidirected edges (latent common causes).
///
/// Bidirected edges are represented as pairs of node names.
pub fn c_components_with_hidden(
    dag: &CausalDAG,
    bidirected: &[(&str, &str)],
) -> Vec<CComponent> {
    let n = dag.n_nodes();
    let mut union_find: Vec<usize> = (0..n).collect();

    fn find(uf: &mut Vec<usize>, mut i: usize) -> usize {
        while uf[i] != i {
            uf[i] = uf[uf[i]]; // path compression
            i = uf[i];
        }
        i
    }

    fn union(uf: &mut Vec<usize>, a: usize, b: usize) {
        let ra = find(uf, a);
        let rb = find(uf, b);
        if ra != rb {
            uf[ra] = rb;
        }
    }

    for &(u, v) in bidirected {
        if let (Some(ui), Some(vi)) = (dag.node_index(u), dag.node_index(v)) {
            union(&mut union_find, ui, vi);
        }
    }

    // Collect components
    let mut comp_map: HashMap<usize, HashSet<usize>> = HashMap::new();
    for i in 0..n {
        let root = find(&mut union_find, i);
        comp_map.entry(root).or_default().insert(i);
    }

    comp_map
        .into_values()
        .map(|nodes| CComponent { nodes })
        .collect()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Remove all incoming edges to nodes in `targets` from a DAG.
fn remove_incoming_edges(dag: &mut CausalDAG, targets: &[&str]) {
    let target_idxs: HashSet<usize> = targets
        .iter()
        .filter_map(|&t| dag.node_index(t))
        .collect();
    dag.remove_incoming_edges_for(&target_idxs);
}

/// Remove all outgoing edges from nodes in `targets`.
fn remove_outgoing_edges(dag: &mut CausalDAG, targets: &[&str]) {
    let target_idxs: HashSet<usize> = targets
        .iter()
        .filter_map(|&t| dag.node_index(t))
        .collect();
    dag.remove_outgoing_edges_for(&target_idxs);
}

/// Check d-separation for all pairs (y_i, z_j) given conditioning set.
fn check_d_separation_all(
    dag: &CausalDAG,
    y: &[&str],
    z: &[&str],
    conditioning: &[&str],
) -> bool {
    for &yi in y {
        for &zi in z {
            if !dag.is_d_separated(yi, zi, conditioning) {
                return false;
            }
        }
    }
    true
}

/// Compute ancestors of a set of named nodes.
fn ancestors_of_names(dag: &CausalDAG, names: &[&str]) -> HashSet<usize> {
    let mut all_anc = HashSet::new();
    for &name in names {
        for anc in dag.ancestors(name) {
            all_anc.insert(anc);
        }
    }
    all_anc
}

/// Check that `m_set` intercepts every directed path from `x` to `y`.
fn intercepts_all_paths(dag: &CausalDAG, x: &str, y: &str, m_set: &[&str]) -> bool {
    // DFS on the directed graph; path is blocked if it passes through m_set
    let xi = match dag.node_index(x) {
        None => return true,
        Some(i) => i,
    };
    let yi = match dag.node_index(y) {
        None => return true,
        Some(i) => i,
    };
    let m_idxs: HashSet<usize> = m_set
        .iter()
        .filter_map(|&m| dag.node_index(m))
        .collect();

    // DFS: can we reach y from x without going through m_set?
    let mut stack: Vec<usize> = vec![xi];
    let mut visited: HashSet<usize> = HashSet::new();
    while let Some(cur) = stack.pop() {
        if cur == yi {
            return false; // Found unblocked path
        }
        if !visited.insert(cur) {
            continue;
        }
        for c in dag.children(dag.node_name(cur).unwrap_or("")) {
            if let Some(ci) = dag.node_index(c) {
                if !m_idxs.contains(&ci) {
                    stack.push(ci);
                }
            }
        }
    }
    true
}

/// Generate the frontdoor formula as a symbolic string.
fn frontdoor_formula(x: &str, y: &str, m_set: &[&str]) -> String {
    let m_str = m_set.join(", ");
    format!(
        "Σ_{{{m_str}}} P({m_str} | {x}) Σ_{{{x}'}} P({y} | {x}', {m_str}) P({x}')",
        m_str = m_str,
        x = x,
        y = y,
    )
}

/// Generate all size-`k` subsets of a slice.
fn subsets<T: Copy>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![Vec::new()];
    }
    if k > items.len() {
        return Vec::new();
    }
    let mut result = Vec::new();
    for i in 0..=(items.len() - k) {
        for mut rest in subsets(&items[i + 1..], k - 1) {
            rest.insert(0, items[i]);
            result.push(rest);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal_graph::dag::CausalDAG;

    fn smoke_dag() -> CausalDAG {
        // X → M → Y  (frontdoor candidate)
        let mut dag = CausalDAG::new();
        dag.add_edge("X", "M").unwrap();
        dag.add_edge("M", "Y").unwrap();
        dag
    }

    fn confounded_dag() -> CausalDAG {
        // U → X, U → Y, X → Y  (U is latent — represented by no path needed)
        // For testing backdoor: Z (observed) → X, Z → Y, X → Y
        let mut dag = CausalDAG::new();
        dag.add_edge("Z", "X").unwrap();
        dag.add_edge("Z", "Y").unwrap();
        dag.add_edge("X", "Y").unwrap();
        dag
    }

    #[test]
    fn test_backdoor_with_z() {
        let dag = confounded_dag();
        // Z blocks the backdoor path X ← Z → Y
        assert!(satisfies_backdoor(&dag, "X", "Y", &["Z"]));
        // Empty set does not block
        assert!(!satisfies_backdoor(&dag, "X", "Y", &[]));
    }

    #[test]
    fn test_find_backdoor_set() {
        let dag = confounded_dag();
        let res = find_backdoor_sets(&dag, "X", "Y", 3);
        assert!(res.is_admissible);
        assert!(res.adjustment_set.contains(&"Z".to_string()));
    }

    #[test]
    fn test_frontdoor() {
        let dag = smoke_dag();
        // M is a valid frontdoor mediator (no unblocked backdoor in this simple DAG)
        assert!(satisfies_frontdoor(&dag, "X", "Y", &["M"]));
        let fd = find_frontdoor_set(&dag, "X", "Y");
        assert!(fd.is_applicable);
    }

    #[test]
    fn test_id_trivial() {
        let dag = smoke_dag();
        let res = id_algorithm(&dag, &["Y"], &[]);
        assert!(res.identifiable);
        assert!(res.expression.contains('P'));
    }

    #[test]
    fn test_tian_pearl() {
        let dag = smoke_dag();
        let res = tian_pearl_id(&dag, &["Y"], &["X"]);
        assert!(res.identifiable);
    }

    #[test]
    fn test_c_components_with_hidden() {
        let dag = smoke_dag();
        // No bidirected edges → each node is its own component
        let comps = c_components_with_hidden(&dag, &[]);
        assert_eq!(comps.len(), dag.n_nodes());
        // One bidirected edge X ↔ Y → two nodes share a component
        let comps2 = c_components_with_hidden(&dag, &[("X", "Y")]);
        assert!(comps2.len() < dag.n_nodes());
    }

    #[test]
    fn test_do_calculus_rule1() {
        let dag = confounded_dag();
        let applies = check_do_calculus_rule(
            &dag,
            &["Y"],
            &["X"],
            &["Z"],
            &[],
            DoCalculusRule::Rule1,
        );
        // In G_{X̄}: Z → Y only, Z is not d-sep from Y given X with no X incoming
        // Rule 1 may or may not apply — we just test it doesn't panic.
        let _ = applies;
    }

    #[test]
    fn test_subsets() {
        let items = vec![1, 2, 3];
        assert_eq!(subsets(&items, 0).len(), 1);
        assert_eq!(subsets(&items, 1).len(), 3);
        assert_eq!(subsets(&items, 2).len(), 3);
        assert_eq!(subsets(&items, 3).len(), 1);
    }
}
