//! Link prediction algorithms for graph analysis
//!
//! This module implements various similarity-based link prediction methods
//! and evaluation metrics. Given a graph, link prediction estimates the
//! likelihood of a link between two nodes that are not currently connected.
//!
//! # Algorithms
//! - **Common Neighbors**: Count of shared neighbors
//! - **Jaccard Coefficient**: Common neighbors normalized by union
//! - **Adamic-Adar Index**: Weighted common neighbors (inverse log degree)
//! - **Preferential Attachment**: Product of degrees
//! - **Resource Allocation Index**: Similar to Adamic-Adar with 1/degree
//! - **Katz Similarity**: Truncated path-based similarity
//! - **SimRank**: Recursive structural similarity
//! - **ROC/AUC Evaluation**: Quality assessment for link prediction

use crate::base::{EdgeWeight, Graph, IndexType, Node};
use crate::error::{GraphError, Result};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// A scored node pair for link prediction
#[derive(Debug, Clone)]
pub struct LinkScore<N: Node> {
    /// First node
    pub node_a: N,
    /// Second node
    pub node_b: N,
    /// Prediction score (higher = more likely to form link)
    pub score: f64,
}

/// Result of link prediction evaluation
#[derive(Debug, Clone)]
pub struct LinkPredictionEval {
    /// Area Under the ROC Curve
    pub auc: f64,
    /// Average Precision
    pub average_precision: f64,
    /// Number of true positive predictions
    pub true_positives: usize,
    /// Number of false positive predictions
    pub false_positives: usize,
    /// Total positive examples
    pub total_positives: usize,
    /// Total negative examples
    pub total_negatives: usize,
}

/// Configuration for link prediction
#[derive(Debug, Clone)]
pub struct LinkPredictionConfig {
    /// Maximum number of predictions to return
    pub max_predictions: usize,
    /// Minimum score threshold for predictions
    pub min_score: f64,
    /// Whether to include self-loops in predictions
    pub include_self_loops: bool,
}

impl Default for LinkPredictionConfig {
    fn default() -> Self {
        Self {
            max_predictions: 100,
            min_score: 0.0,
            include_self_loops: false,
        }
    }
}

// ============================================================================
// Common Neighbors
// ============================================================================

/// Common neighbors score between two nodes.
///
/// The score is the number of shared neighbors between nodes u and v.
/// Higher score suggests higher likelihood of a future link.
///
/// Score(u, v) = |N(u) intersection N(v)|
pub fn common_neighbors_score<N, E, Ix>(graph: &Graph<N, E, Ix>, u: &N, v: &N) -> Result<f64>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    validate_nodes(graph, u, v)?;

    let neighbors_u: HashSet<N> = graph.neighbors(u)?.into_iter().collect();
    let neighbors_v: HashSet<N> = graph.neighbors(v)?.into_iter().collect();

    let common = neighbors_u.intersection(&neighbors_v).count();
    Ok(common as f64)
}

/// Compute common neighbors scores for all non-connected node pairs.
pub fn common_neighbors_all<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    config: &LinkPredictionConfig,
) -> Vec<LinkScore<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    compute_all_scores(graph, config, |g, u, v| {
        common_neighbors_score(g, u, v).unwrap_or(0.0)
    })
}

// ============================================================================
// Jaccard Coefficient
// ============================================================================

/// Jaccard coefficient between two nodes.
///
/// Score(u, v) = |N(u) intersection N(v)| / |N(u) union N(v)|
///
/// Returns 0.0 if both nodes have no neighbors.
pub fn jaccard_coefficient<N, E, Ix>(graph: &Graph<N, E, Ix>, u: &N, v: &N) -> Result<f64>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    validate_nodes(graph, u, v)?;

    let neighbors_u: HashSet<N> = graph.neighbors(u)?.into_iter().collect();
    let neighbors_v: HashSet<N> = graph.neighbors(v)?.into_iter().collect();

    let intersection = neighbors_u.intersection(&neighbors_v).count();
    let union = neighbors_u.union(&neighbors_v).count();

    if union == 0 {
        Ok(0.0)
    } else {
        Ok(intersection as f64 / union as f64)
    }
}

/// Compute Jaccard coefficient for all non-connected node pairs.
pub fn jaccard_coefficient_all<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    config: &LinkPredictionConfig,
) -> Vec<LinkScore<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    compute_all_scores(graph, config, |g, u, v| {
        jaccard_coefficient(g, u, v).unwrap_or(0.0)
    })
}

// ============================================================================
// Adamic-Adar Index
// ============================================================================

/// Adamic-Adar index between two nodes.
///
/// Sums 1/log(degree) for each common neighbor. Nodes with fewer
/// connections contribute more to the score.
///
/// Score(u, v) = sum_{w in N(u) intersection N(v)} 1 / log(|N(w)|)
pub fn adamic_adar_index<N, E, Ix>(graph: &Graph<N, E, Ix>, u: &N, v: &N) -> Result<f64>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    validate_nodes(graph, u, v)?;

    let neighbors_u: HashSet<N> = graph.neighbors(u)?.into_iter().collect();
    let neighbors_v: HashSet<N> = graph.neighbors(v)?.into_iter().collect();

    let mut score = 0.0;
    for common in neighbors_u.intersection(&neighbors_v) {
        let degree = graph.degree(common);
        if degree > 1 {
            score += 1.0 / (degree as f64).ln();
        }
    }

    Ok(score)
}

/// Compute Adamic-Adar index for all non-connected node pairs.
pub fn adamic_adar_all<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    config: &LinkPredictionConfig,
) -> Vec<LinkScore<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    compute_all_scores(graph, config, |g, u, v| {
        adamic_adar_index(g, u, v).unwrap_or(0.0)
    })
}

// ============================================================================
// Preferential Attachment
// ============================================================================

/// Preferential attachment score between two nodes.
///
/// Based on the Barabasi-Albert model: nodes with more connections
/// are more likely to form new connections.
///
/// Score(u, v) = |N(u)| * |N(v)|
pub fn preferential_attachment<N, E, Ix>(graph: &Graph<N, E, Ix>, u: &N, v: &N) -> Result<f64>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    validate_nodes(graph, u, v)?;

    let deg_u = graph.degree(u);
    let deg_v = graph.degree(v);

    Ok((deg_u * deg_v) as f64)
}

/// Compute preferential attachment for all non-connected node pairs.
pub fn preferential_attachment_all<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    config: &LinkPredictionConfig,
) -> Vec<LinkScore<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    compute_all_scores(graph, config, |g, u, v| {
        preferential_attachment(g, u, v).unwrap_or(0.0)
    })
}

// ============================================================================
// Resource Allocation Index
// ============================================================================

/// Resource allocation index between two nodes.
///
/// Similar to Adamic-Adar but uses 1/degree instead of 1/log(degree).
/// Proposed by Zhou, Lu, and Zhang (2009).
///
/// Score(u, v) = sum_{w in N(u) intersection N(v)} 1 / |N(w)|
pub fn resource_allocation_index<N, E, Ix>(graph: &Graph<N, E, Ix>, u: &N, v: &N) -> Result<f64>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    validate_nodes(graph, u, v)?;

    let neighbors_u: HashSet<N> = graph.neighbors(u)?.into_iter().collect();
    let neighbors_v: HashSet<N> = graph.neighbors(v)?.into_iter().collect();

    let mut score = 0.0;
    for common in neighbors_u.intersection(&neighbors_v) {
        let degree = graph.degree(common);
        if degree > 0 {
            score += 1.0 / degree as f64;
        }
    }

    Ok(score)
}

/// Compute resource allocation for all non-connected node pairs.
pub fn resource_allocation_all<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    config: &LinkPredictionConfig,
) -> Vec<LinkScore<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    compute_all_scores(graph, config, |g, u, v| {
        resource_allocation_index(g, u, v).unwrap_or(0.0)
    })
}

// ============================================================================
// Katz Similarity (Truncated)
// ============================================================================

/// Truncated Katz similarity between two nodes.
///
/// Considers paths of different lengths weighted by a damping factor beta.
/// Truncated at `max_path_length` for efficiency.
///
/// Score(u, v) = sum_{l=1}^{L} beta^l * |paths_l(u, v)|
///
/// # Arguments
/// * `graph` - The graph
/// * `u`, `v` - The node pair
/// * `beta` - Damping factor (typically 0.001 to 0.1)
/// * `max_path_length` - Maximum path length to consider
pub fn katz_similarity<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    u: &N,
    v: &N,
    beta: f64,
    max_path_length: usize,
) -> Result<f64>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    validate_nodes(graph, u, v)?;

    if beta <= 0.0 || beta >= 1.0 {
        return Err(GraphError::InvalidGraph(
            "Beta must be in (0, 1)".to_string(),
        ));
    }

    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();
    let node_to_idx: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    let u_idx = node_to_idx
        .get(u)
        .ok_or_else(|| GraphError::node_not_found(format!("{u:?}")))?;
    let v_idx = node_to_idx
        .get(v)
        .ok_or_else(|| GraphError::node_not_found(format!("{v:?}")))?;

    // Build adjacency matrix as sparse representation
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for (i, node) in nodes.iter().enumerate() {
        if let Ok(neighbors) = graph.neighbors(node) {
            for neighbor in &neighbors {
                if let Some(&j) = node_to_idx.get(neighbor) {
                    adj[i].push(j);
                }
            }
        }
    }

    // Count paths of each length using matrix power approach
    // paths_l[i] = number of paths of length l from u to node i
    let mut score = 0.0;
    let mut current = vec![0.0f64; n];
    current[*u_idx] = 1.0;

    for l in 1..=max_path_length {
        let mut next = vec![0.0f64; n];
        for (i, &count) in current.iter().enumerate() {
            if count > 0.0 {
                for &j in &adj[i] {
                    next[j] += count;
                }
            }
        }

        let beta_l = beta.powi(l as i32);
        score += beta_l * next[*v_idx];
        current = next;
    }

    Ok(score)
}

/// Compute Katz similarity for all non-connected node pairs.
pub fn katz_similarity_all<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    beta: f64,
    max_path_length: usize,
    config: &LinkPredictionConfig,
) -> Vec<LinkScore<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    compute_all_scores(graph, config, |g, u, v| {
        katz_similarity(g, u, v, beta, max_path_length).unwrap_or(0.0)
    })
}

// ============================================================================
// SimRank
// ============================================================================

/// SimRank: structural similarity based on the idea that two nodes are similar
/// if they are referenced by similar nodes.
///
/// Uses iterative computation with damping factor C.
/// SimRank(u, u) = 1, SimRank(u, v) = C / (|N(u)| * |N(v)|) * sum SimRank(a, b)
///
/// # Arguments
/// * `graph` - The graph
/// * `decay` - Decay/damping factor (typically 0.8)
/// * `max_iterations` - Maximum iterations for convergence
/// * `tolerance` - Convergence tolerance
pub fn simrank<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    decay: f64,
    max_iterations: usize,
    tolerance: f64,
) -> Result<HashMap<(N, N), f64>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    if decay <= 0.0 || decay > 1.0 {
        return Err(GraphError::InvalidGraph(
            "Decay must be in (0, 1]".to_string(),
        ));
    }

    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();
    let node_to_idx: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    // Build adjacency
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for (i, node) in nodes.iter().enumerate() {
        if let Ok(neighbors) = graph.neighbors(node) {
            for neighbor in &neighbors {
                if let Some(&j) = node_to_idx.get(neighbor) {
                    adj[i].push(j);
                }
            }
        }
    }

    // Initialize SimRank: S(a, a) = 1, S(a, b) = 0 for a != b
    let mut sim = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        sim[i][i] = 1.0;
    }

    // Iterate
    for _ in 0..max_iterations {
        let mut new_sim = vec![vec![0.0f64; n]; n];
        let mut max_diff = 0.0f64;

        for i in 0..n {
            new_sim[i][i] = 1.0;
            for j in (i + 1)..n {
                let deg_i = adj[i].len();
                let deg_j = adj[j].len();

                if deg_i == 0 || deg_j == 0 {
                    new_sim[i][j] = 0.0;
                    new_sim[j][i] = 0.0;
                    continue;
                }

                let mut sum = 0.0;
                for &ni in &adj[i] {
                    for &nj in &adj[j] {
                        sum += sim[ni][nj];
                    }
                }

                let new_val = decay * sum / (deg_i * deg_j) as f64;
                new_sim[i][j] = new_val;
                new_sim[j][i] = new_val;

                let diff = (new_val - sim[i][j]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        sim = new_sim;

        if max_diff < tolerance {
            break;
        }
    }

    // Convert to HashMap
    let mut result = HashMap::new();
    for i in 0..n {
        for j in i..n {
            result.insert((nodes[i].clone(), nodes[j].clone()), sim[i][j]);
            if i != j {
                result.insert((nodes[j].clone(), nodes[i].clone()), sim[i][j]);
            }
        }
    }

    Ok(result)
}

/// SimRank score between a specific pair of nodes.
pub fn simrank_score<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    u: &N,
    v: &N,
    decay: f64,
    max_iterations: usize,
) -> Result<f64>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    let all_scores = simrank(graph, decay, max_iterations, 1e-6)?;
    all_scores
        .get(&(u.clone(), v.clone()))
        .copied()
        .ok_or_else(|| GraphError::node_not_found(format!("{u:?}")))
}

// ============================================================================
// ROC/AUC Evaluation
// ============================================================================

/// Evaluate link prediction quality using ROC AUC.
///
/// Compares predicted scores against known positive (existing) and
/// negative (non-existing) edges.
///
/// # Arguments
/// * `scores` - Predicted link scores for node pairs
/// * `positive_edges` - Known positive edges (existing links)
/// * `negative_edges` - Known negative edges (non-existing links)
pub fn evaluate_link_prediction<N>(
    scores: &[LinkScore<N>],
    positive_edges: &HashSet<(N, N)>,
    negative_edges: &HashSet<(N, N)>,
) -> LinkPredictionEval
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
{
    if positive_edges.is_empty() || negative_edges.is_empty() {
        return LinkPredictionEval {
            auc: 0.5,
            average_precision: 0.0,
            true_positives: 0,
            false_positives: 0,
            total_positives: positive_edges.len(),
            total_negatives: negative_edges.len(),
        };
    }

    // Build scored list with labels
    let mut scored_labels: Vec<(f64, bool)> = Vec::new();

    for score in scores {
        let pair = (score.node_a.clone(), score.node_b.clone());
        let reverse_pair = (score.node_b.clone(), score.node_a.clone());

        let is_positive = positive_edges.contains(&pair) || positive_edges.contains(&reverse_pair);
        let is_negative = negative_edges.contains(&pair) || negative_edges.contains(&reverse_pair);

        if is_positive || is_negative {
            scored_labels.push((score.score, is_positive));
        }
    }

    // Sort by score descending
    scored_labels.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Compute AUC using the trapezoidal rule
    let total_positives = scored_labels.iter().filter(|(_, label)| *label).count();
    let total_negatives = scored_labels.iter().filter(|(_, label)| !*label).count();

    if total_positives == 0 || total_negatives == 0 {
        return LinkPredictionEval {
            auc: 0.5,
            average_precision: 0.0,
            true_positives: 0,
            false_positives: 0,
            total_positives,
            total_negatives,
        };
    }

    let mut auc = 0.0;
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut prev_fpr = 0.0;
    let mut prev_tpr = 0.0;

    // Compute average precision
    let mut ap = 0.0;
    let mut running_tp = 0;

    for (i, &(_, is_positive)) in scored_labels.iter().enumerate() {
        if is_positive {
            tp += 1;
            running_tp += 1;
            ap += running_tp as f64 / (i + 1) as f64;
        } else {
            fp += 1;
        }

        let tpr = tp as f64 / total_positives as f64;
        let fpr = fp as f64 / total_negatives as f64;

        // Trapezoidal rule
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
        prev_fpr = fpr;
        prev_tpr = tpr;
    }

    // Complete the curve to (1, 1)
    auc += (1.0 - prev_fpr) * (1.0 + prev_tpr) / 2.0;

    let average_precision = if total_positives > 0 {
        ap / total_positives as f64
    } else {
        0.0
    };

    LinkPredictionEval {
        auc,
        average_precision,
        true_positives: tp,
        false_positives: fp,
        total_positives,
        total_negatives,
    }
}

/// Simplified AUC computation: randomly sample positive and negative pairs,
/// score them, and estimate AUC.
pub fn compute_auc<N, E, Ix, F>(
    graph: &Graph<N, E, Ix>,
    test_edges: &[(N, N)],
    non_edges: &[(N, N)],
    score_fn: F,
) -> f64
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
    F: Fn(&Graph<N, E, Ix>, &N, &N) -> f64,
{
    if test_edges.is_empty() || non_edges.is_empty() {
        return 0.5;
    }

    let mut n_correct = 0usize;
    let mut n_tie = 0usize;
    let mut n_total = 0usize;

    for (pu, pv) in test_edges {
        let pos_score = score_fn(graph, pu, pv);
        for (nu, nv) in non_edges {
            let neg_score = score_fn(graph, nu, nv);
            n_total += 1;
            if pos_score > neg_score + 1e-12 {
                n_correct += 1;
            } else if (pos_score - neg_score).abs() <= 1e-12 {
                n_tie += 1;
            }
        }
    }

    if n_total == 0 {
        return 0.5;
    }

    (n_correct as f64 + 0.5 * n_tie as f64) / n_total as f64
}

// ============================================================================
// Internal helpers
// ============================================================================

fn validate_nodes<N, E, Ix>(graph: &Graph<N, E, Ix>, u: &N, v: &N) -> Result<()>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    if !graph.has_node(u) {
        return Err(GraphError::node_not_found(format!("{u:?}")));
    }
    if !graph.has_node(v) {
        return Err(GraphError::node_not_found(format!("{v:?}")));
    }
    Ok(())
}

fn compute_all_scores<N, E, Ix, F>(
    graph: &Graph<N, E, Ix>,
    config: &LinkPredictionConfig,
    score_fn: F,
) -> Vec<LinkScore<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
    F: Fn(&Graph<N, E, Ix>, &N, &N) -> f64,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let mut scores = Vec::new();

    for (i, u) in nodes.iter().enumerate() {
        for v in nodes.iter().skip(i + 1) {
            if !config.include_self_loops && u == v {
                continue;
            }
            // Only predict for non-connected pairs
            if graph.has_edge(u, v) {
                continue;
            }

            let score = score_fn(graph, u, v);
            if score >= config.min_score {
                scores.push(LinkScore {
                    node_a: u.clone(),
                    node_b: v.clone(),
                    score,
                });
            }
        }
    }

    // Sort by score descending
    scores.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Truncate to max
    scores.truncate(config.max_predictions);
    scores
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as GraphResult;
    use crate::generators::create_graph;

    fn build_test_graph() -> Graph<i32, ()> {
        let mut g = create_graph::<i32, ()>();
        // Build a small social network:
        //   0 -- 1 -- 2
        //   |    |    |
        //   3 -- 4 -- 5
        let _ = g.add_edge(0, 1, ());
        let _ = g.add_edge(1, 2, ());
        let _ = g.add_edge(0, 3, ());
        let _ = g.add_edge(1, 4, ());
        let _ = g.add_edge(2, 5, ());
        let _ = g.add_edge(3, 4, ());
        let _ = g.add_edge(4, 5, ());
        g
    }

    #[test]
    fn test_common_neighbors() -> GraphResult<()> {
        let g = build_test_graph();

        // Nodes 0 and 2 share neighbor 1
        let score = common_neighbors_score(&g, &0, &2)?;
        assert!((score - 1.0).abs() < 1e-6);

        // Nodes 0 and 4 share neighbors 1 and 3
        let score = common_neighbors_score(&g, &0, &4)?;
        assert!((score - 2.0).abs() < 1e-6);

        // Nodes 0 and 5 share no neighbors
        let score = common_neighbors_score(&g, &0, &5)?;
        assert!((score - 0.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_jaccard_coefficient() -> GraphResult<()> {
        let g = build_test_graph();

        // Nodes 0 and 4: intersection={1,3}, union={1,2,3,5} (based on graph)
        let score = jaccard_coefficient(&g, &0, &4)?;
        assert!(score > 0.0 && score <= 1.0);

        // Self-similarity should be handled
        let score = jaccard_coefficient(&g, &0, &0)?;
        assert!((score - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_adamic_adar() -> GraphResult<()> {
        let g = build_test_graph();

        let score = adamic_adar_index(&g, &0, &4)?;
        assert!(score > 0.0);

        // Nodes with no common neighbors should have 0 score
        let score = adamic_adar_index(&g, &0, &5)?;
        assert!((score - 0.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_preferential_attachment() -> GraphResult<()> {
        let g = build_test_graph();

        // Node 0 has degree 2, node 4 has degree 3
        let score = preferential_attachment(&g, &0, &4)?;
        assert!((score - 6.0).abs() < 1e-6);

        // Node 1 has degree 3, node 4 has degree 3
        let score = preferential_attachment(&g, &1, &4)?;
        assert!((score - 9.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_resource_allocation() -> GraphResult<()> {
        let g = build_test_graph();

        let score = resource_allocation_index(&g, &0, &4)?;
        assert!(score > 0.0);

        let score = resource_allocation_index(&g, &0, &5)?;
        assert!((score - 0.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_katz_similarity() -> GraphResult<()> {
        let g = build_test_graph();

        let score = katz_similarity(&g, &0, &2, 0.05, 3)?;
        assert!(score > 0.0);

        // Closer nodes should have higher Katz similarity
        let score_near = katz_similarity(&g, &0, &1, 0.05, 3)?;
        let score_far = katz_similarity(&g, &0, &5, 0.05, 3)?;
        assert!(score_near > score_far);
        Ok(())
    }

    #[test]
    fn test_katz_invalid_beta() {
        let g = build_test_graph();
        assert!(katz_similarity(&g, &0, &1, 0.0, 3).is_err());
        assert!(katz_similarity(&g, &0, &1, 1.0, 3).is_err());
    }

    #[test]
    fn test_simrank() -> GraphResult<()> {
        let g = build_test_graph();

        let scores = simrank(&g, 0.8, 10, 1e-4)?;

        // Self-similarity should be 1.0
        let self_score = scores.get(&(0, 0)).copied().unwrap_or(0.0);
        assert!((self_score - 1.0).abs() < 1e-6);

        // Structural similarity should be non-negative
        for (_, &score) in &scores {
            assert!(score >= -1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_simrank_score() -> GraphResult<()> {
        let g = build_test_graph();
        let score = simrank_score(&g, &0, &2, 0.8, 10)?;
        assert!(score >= 0.0);
        assert!(score <= 1.0);
        Ok(())
    }

    #[test]
    fn test_evaluate_link_prediction() {
        let scores = vec![
            LinkScore {
                node_a: 0,
                node_b: 1,
                score: 0.9,
            },
            LinkScore {
                node_a: 0,
                node_b: 2,
                score: 0.8,
            },
            LinkScore {
                node_a: 0,
                node_b: 3,
                score: 0.3,
            },
            LinkScore {
                node_a: 1,
                node_b: 3,
                score: 0.2,
            },
        ];

        let mut positives = HashSet::new();
        positives.insert((0, 1));
        positives.insert((0, 2));

        let mut negatives = HashSet::new();
        negatives.insert((0, 3));
        negatives.insert((1, 3));

        let eval = evaluate_link_prediction(&scores, &positives, &negatives);
        assert!(eval.auc >= 0.5); // Should be better than random
        assert!(eval.true_positives > 0);
    }

    #[test]
    fn test_compute_auc() -> GraphResult<()> {
        let g = build_test_graph();

        // Remove edge 0-2 conceptually (it doesn't exist, so 0-4 is already non-edge)
        // Positive test: nodes that share many neighbors
        let test_edges = vec![(0, 4)]; // 0 and 4 share neighbors 1,3
        let non_edges = vec![(0, 5)]; // 0 and 5 share no neighbors

        let auc = compute_auc(&g, &test_edges, &non_edges, |g, u, v| {
            common_neighbors_score(g, u, v).unwrap_or(0.0)
        });

        assert!(auc >= 0.5); // Should predict correctly
        Ok(())
    }

    #[test]
    fn test_common_neighbors_all() {
        let g = build_test_graph();
        let config = LinkPredictionConfig {
            max_predictions: 10,
            min_score: 0.0,
            include_self_loops: false,
        };

        let scores = common_neighbors_all(&g, &config);
        // Should only include non-connected pairs
        for score in &scores {
            assert!(!g.has_edge(&score.node_a, &score.node_b));
        }
        // Should be sorted by score descending
        for window in scores.windows(2) {
            assert!(window[0].score >= window[1].score);
        }
    }

    #[test]
    fn test_invalid_nodes() {
        let g = build_test_graph();
        assert!(common_neighbors_score(&g, &0, &99).is_err());
        assert!(jaccard_coefficient(&g, &99, &0).is_err());
        assert!(adamic_adar_index(&g, &0, &99).is_err());
    }

    #[test]
    fn test_empty_graph_link_prediction() -> GraphResult<()> {
        let mut g = create_graph::<i32, ()>();
        let _ = g.add_node(0);

        let config = LinkPredictionConfig::default();
        let scores = common_neighbors_all(&g, &config);
        assert!(scores.is_empty());
        Ok(())
    }

    #[test]
    fn test_all_methods_consistency() -> GraphResult<()> {
        let g = build_test_graph();

        // All methods should return non-negative scores for same pair
        let cn = common_neighbors_score(&g, &0, &4)?;
        let jc = jaccard_coefficient(&g, &0, &4)?;
        let aa = adamic_adar_index(&g, &0, &4)?;
        let pa = preferential_attachment(&g, &0, &4)?;
        let ra = resource_allocation_index(&g, &0, &4)?;
        let kz = katz_similarity(&g, &0, &4, 0.05, 3)?;

        assert!(cn >= 0.0);
        assert!(jc >= 0.0);
        assert!(aa >= 0.0);
        assert!(pa >= 0.0);
        assert!(ra >= 0.0);
        assert!(kz >= 0.0);

        // Pairs with common neighbors should score > 0 for relevant methods
        assert!(cn > 0.0);
        assert!(jc > 0.0);
        assert!(aa > 0.0);
        assert!(ra > 0.0);
        Ok(())
    }
}
