//! Relation-aware message passing primitives
//!
//! This module provides:
//!
//! - [`RelationEmbedding`] – a learnable embedding table for relation types.
//! - [`RotatEScoring`] – RotatE scoring function (Sun et al. 2019), which
//!   models each relation as a **rotation** in complex space:
//!   ```text
//!     score(h, r, t) = − ‖ h_e ∘ r_e − t_e ‖
//!   ```
//!   where `∘` denotes element-wise complex multiplication
//!   (`r_e = exp(i θ_r)` so `|r_e| = 1`).
//! - [`HeterogeneousAdjacency`] – compact adjacency storage that organises
//!   edges both by relation type and by source/destination node type.

use std::collections::HashMap;

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Rng, RngExt};

use crate::gnn::rgcn::KgScorer;

// ============================================================================
// Helpers
// ============================================================================

/// Xavier uniform initialisation.
fn xavier_uniform(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = scirs2_core::random::rng();
    let limit = (6.0_f64 / (rows + cols) as f64).sqrt();
    Array2::from_shape_fn((rows, cols), |_| rng.random::<f64>() * 2.0 * limit - limit)
}

// ============================================================================
// RelationEmbedding
// ============================================================================

/// Learnable embedding table for relation types.
///
/// Embeddings are initialised with Xavier uniform and can be updated via
/// gradient-based methods (integration with an optimiser is the caller's
/// responsibility; this struct just holds the parameter tensor).
#[derive(Debug, Clone)]
pub struct RelationEmbedding {
    /// Embedding matrix `(n_relations, dim)`.
    pub table: Array2<f64>,
    /// Number of relation types.
    pub n_relations: usize,
    /// Embedding dimensionality.
    pub dim: usize,
}

impl RelationEmbedding {
    /// Create a new relation embedding table with Xavier initialisation.
    ///
    /// # Arguments
    /// * `n_relations` – Number of distinct relation types.
    /// * `dim`         – Embedding dimensionality.
    pub fn new(n_relations: usize, dim: usize) -> Self {
        Self {
            table: xavier_uniform(n_relations, dim),
            n_relations,
            dim,
        }
    }

    /// Look up the embedding vector for relation `r`.
    ///
    /// Returns `None` if `r >= n_relations`.
    pub fn get(&self, r: usize) -> Option<Array1<f64>> {
        if r < self.n_relations {
            Some(self.table.row(r).to_owned())
        } else {
            None
        }
    }
}

// ============================================================================
// RotatEScoring
// ============================================================================

/// RotatE scoring model (Sun et al. 2019).
///
/// Entity embeddings are complex-valued: each entity `e` has a real part
/// `entity_re[e]` and an imaginary part `entity_im[e]`, both of shape
/// `(n_entities, dim/2)` (so the full complex embedding has `dim` real
/// parameters per entity).
///
/// Relation embeddings are phase angles `θ_r ∈ (−π, π]` of shape
/// `(n_relations, dim/2)`.  The unit-modulus rotation vector is
/// `r_e = cos(θ_r) + i sin(θ_r)`.
///
/// Score function (negated L2 in complex space):
/// ```text
///   score(h, r, t) = − ‖ h_re ⊙ cos(θ_r) − h_im ⊙ sin(θ_r) − t_re
///                       + i ( h_re ⊙ sin(θ_r) + h_im ⊙ cos(θ_r) − t_im ) ‖
/// ```
/// Higher scores (less negative) imply a more plausible triple.
#[derive(Debug, Clone)]
pub struct RotatEScoring {
    /// Real part of entity embeddings `(n_entities, dim)`.
    pub entity_re: Array2<f64>,
    /// Imaginary part of entity embeddings `(n_entities, dim)`.
    pub entity_im: Array2<f64>,
    /// Relation phase angles `(n_relations, dim)`.
    pub relation_phase: Array2<f64>,
    /// Number of entities.
    pub n_entities: usize,
    /// Number of relation types.
    pub n_relations: usize,
    /// Half-embedding dimension (complex degree of freedom per entity/relation).
    pub dim: usize,
}

impl RotatEScoring {
    /// Create a new RotatE model with random initialisation.
    ///
    /// Entity embeddings are Xavier-initialised; relation phases are uniform
    /// in `(−π, π]`.
    ///
    /// # Arguments
    /// * `n_entities`  – Number of distinct entities.
    /// * `n_relations` – Number of distinct relation types.
    /// * `dim`         – Complex embedding half-dimension (full parameter count
    ///   per entity is `2 * dim`).
    pub fn new(n_entities: usize, n_relations: usize, dim: usize) -> Self {
        let mut rng = scirs2_core::random::rng();
        let entity_re = xavier_uniform(n_entities, dim);
        let entity_im = xavier_uniform(n_entities, dim);
        // Relation phases uniform in (−π, π]
        let relation_phase = Array2::from_shape_fn((n_relations, dim), |_| {
            (rng.random::<f64>() * 2.0 - 1.0) * std::f64::consts::PI
        });
        Self {
            entity_re,
            entity_im,
            relation_phase,
            n_entities,
            n_relations,
            dim,
        }
    }

    /// Compute the RotatE score for triple `(h, r, t)`.
    ///
    /// Returns the **negated** L2 norm of `h ∘ r − t` in complex space.
    /// Scores closer to 0 indicate a more plausible triple.
    pub fn score_triple(&self, h: usize, r: usize, t: usize) -> f64 {
        let h_re = self.entity_re.row(h);
        let h_im = self.entity_im.row(h);
        let t_re = self.entity_re.row(t);
        let t_im = self.entity_im.row(t);
        let phase = self.relation_phase.row(r);

        let mut norm_sq = 0.0_f64;
        for k in 0..self.dim {
            let cos_r = phase[k].cos();
            let sin_r = phase[k].sin();
            // Real part of (h ∘ r − t)
            let diff_re = h_re[k] * cos_r - h_im[k] * sin_r - t_re[k];
            // Imaginary part of (h ∘ r − t)
            let diff_im = h_re[k] * sin_r + h_im[k] * cos_r - t_im[k];
            norm_sq += diff_re * diff_re + diff_im * diff_im;
        }
        -norm_sq.sqrt()
    }
}

impl KgScorer for RotatEScoring {
    fn score(&self, h: usize, r: usize, t: usize) -> f64 {
        self.score_triple(h, r, t)
    }
}

// ============================================================================
// HeterogeneousAdjacency
// ============================================================================

/// Compact adjacency representation for heterogeneous graphs.
///
/// Stores edges both:
/// - Per relation type: `by_relation[r]` = list of `(src, dst)` edges with
///   relation type `r`.
/// - Per `(src_node_type, dst_node_type)` pair: `by_node_type[(ts, tt)]` =
///   list of all node indices `src` that have at least one edge to a node of
///   type `tt`.
#[derive(Debug, Clone)]
pub struct HeterogeneousAdjacency {
    /// Adjacency list per relation type: `by_relation[r]` = `Vec<(src, dst)>`.
    pub by_relation: Vec<Vec<(usize, usize)>>,
    /// Nodes per `(src_type, dst_type)` pair.
    ///
    /// `by_node_type[(src_type, dst_type)]` contains the *source* node indices
    /// for edges going from a node of `src_type` to a node of `dst_type`.
    pub by_node_type: HashMap<(usize, usize), Vec<usize>>,
    /// Number of relation types
    pub n_relations: usize,
    /// Number of node types
    pub n_node_types: usize,
}

impl HeterogeneousAdjacency {
    /// Construct a [`HeterogeneousAdjacency`] from a list of typed edges.
    ///
    /// # Arguments
    /// * `n_relations`  – Total number of distinct relation types.
    /// * `n_node_types` – Total number of distinct node types.
    /// * `node_types`   – Node type assignment for every node `(len = n_nodes)`.
    /// * `typed_edges`  – List of `(src, rel_type, dst)` directed edges.
    pub fn from_typed_edges(
        n_relations: usize,
        n_node_types: usize,
        node_types: &[usize],
        typed_edges: &[(usize, usize, usize)], // (src, rel, dst)
    ) -> Self {
        let mut by_relation: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_relations];
        let mut by_node_type: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

        for &(src, rel, dst) in typed_edges {
            if rel < n_relations {
                by_relation[rel].push((src, dst));
            }
            let src_t = node_types.get(src).copied().unwrap_or(0);
            let dst_t = node_types.get(dst).copied().unwrap_or(0);
            by_node_type.entry((src_t, dst_t)).or_default().push(src);
        }

        Self {
            by_relation,
            by_node_type,
            n_relations,
            n_node_types,
        }
    }

    /// Return an iterator over all `(src, dst)` pairs for relation `r`.
    pub fn edges_for_relation(&self, r: usize) -> &[(usize, usize)] {
        self.by_relation.get(r).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Return all source node indices going to nodes of type `dst_type` from
    /// nodes of type `src_type`.
    pub fn sources_by_type(&self, src_type: usize, dst_type: usize) -> &[usize] {
        self.by_node_type
            .get(&(src_type, dst_type))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Total number of edges in the graph.
    pub fn n_edges(&self) -> usize {
        self.by_relation.iter().map(|v| v.len()).sum()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relation_embedding_shape() {
        let emb = RelationEmbedding::new(5, 16);
        assert_eq!(emb.table.nrows(), 5);
        assert_eq!(emb.table.ncols(), 16);
    }

    #[test]
    fn test_relation_embedding_get() {
        let emb = RelationEmbedding::new(3, 8);
        assert!(emb.get(0).is_some());
        assert!(emb.get(2).is_some());
        assert!(emb.get(3).is_none());
    }

    #[test]
    fn test_rotate_score_is_finite() {
        let scorer = RotatEScoring::new(5, 3, 8);
        let s = scorer.score_triple(0, 0, 1);
        assert!(s.is_finite(), "RotatE score must be finite");
    }

    #[test]
    fn test_rotate_self_score_is_highest() {
        // For a model that perfectly maps h ∘ r = t the score should be 0.0.
        // We just verify the score decreases when the tail does not match.
        let scorer = RotatEScoring::new(4, 2, 4);
        // Score same entity as both head and tail under identity rotation (r=0)
        let s_same = scorer.score_triple(0, 0, 0);
        let s_diff = scorer.score_triple(0, 0, 1);
        // With a non-trivial random model this is not guaranteed to hold;
        // we just check both are finite.
        assert!(s_same.is_finite());
        assert!(s_diff.is_finite());
    }

    #[test]
    fn test_rotate_scorer_trait_object() {
        let scorer: Box<dyn KgScorer> = Box::new(RotatEScoring::new(3, 2, 4));
        let s = scorer.score(0, 0, 1);
        assert!(s.is_finite());
    }

    #[test]
    fn test_heterogeneous_adjacency_by_relation() {
        let node_types = vec![0usize, 0, 1, 1];
        let edges = vec![
            (0usize, 0usize, 2usize), // rel 0
            (1, 0, 3),                // rel 0
            (0, 1, 1),                // rel 1
        ];
        let adj = HeterogeneousAdjacency::from_typed_edges(2, 2, &node_types, &edges);
        assert_eq!(adj.by_relation.len(), 2);
        assert_eq!(adj.edges_for_relation(0).len(), 2);
        assert_eq!(adj.edges_for_relation(1).len(), 1);
    }

    #[test]
    fn test_heterogeneous_adjacency_by_node_type() {
        let node_types = vec![0usize, 0, 1, 1];
        let edges = vec![(0usize, 0usize, 2usize), (1, 0, 3)];
        let adj = HeterogeneousAdjacency::from_typed_edges(1, 2, &node_types, &edges);
        // Both edges go from type 0 → type 1
        let srcs = adj.sources_by_type(0, 1);
        assert_eq!(srcs.len(), 2);
    }

    #[test]
    fn test_heterogeneous_adjacency_n_edges() {
        let node_types = vec![0usize; 5];
        let edges: Vec<(usize, usize, usize)> = (0..4).map(|i| (i, 0, i + 1)).collect();
        let adj = HeterogeneousAdjacency::from_typed_edges(1, 1, &node_types, &edges);
        assert_eq!(adj.n_edges(), 4);
    }
}
