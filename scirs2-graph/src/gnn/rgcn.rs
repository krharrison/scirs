//! Relational Graph Convolutional Network (R-GCN)
//!
//! Implements the encoder from Schlichtkrull et al. (2018),
//! "Modeling Relational Data with Graph Convolutional Networks".
//!
//! R-GCN extends GCN to handle multi-relational graphs (knowledge graphs)
//! by maintaining a separate weight matrix per relation type.  To keep the
//! parameter count manageable, **basis decomposition** is used:
//!
//! ```text
//!   W_r = Σ_b  a_{r,b}  V_b
//! ```
//!
//! where `V_b` are `n_bases` shared basis matrices and `a_{r,b}` are
//! per-relation scalar coefficients.
//!
//! The layer update rule is:
//! ```text
//!   h_i^{(l+1)} = ReLU( Σ_r  Σ_{j ∈ N_r(i)}  (W_r h_j^{(l)}) / |N_r(i)|
//!                       + W_0 h_i^{(l)} )
//! ```

use std::collections::HashMap;

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Rng, RngExt};

use crate::error::{GraphError, Result};

// ============================================================================
// Helpers
// ============================================================================

/// Xavier uniform initialisation: U[-gain*sqrt(6/(fan_in+fan_out)), ...]
fn xavier_uniform(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = scirs2_core::random::rng();
    let limit = (6.0_f64 / (rows + cols) as f64).sqrt();
    Array2::from_shape_fn((rows, cols), |_| rng.random::<f64>() * 2.0 * limit - limit)
}

/// ReLU activation applied element-wise to a 2-D array.
fn relu2(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.max(0.0))
}

// ============================================================================
// RgcnConfig
// ============================================================================

/// Configuration for an R-GCN encoder.
#[derive(Debug, Clone)]
pub struct RgcnConfig {
    /// Hidden (output) dimensionality of each layer
    pub hidden_dim: usize,
    /// Number of basis matrices for weight decomposition
    pub n_bases: usize,
    /// Number of R-GCN layers to stack
    pub n_layers: usize,
    /// Dropout probability (applied on node features between layers)
    pub dropout: f64,
    /// Whether to include a self-loop weight W_0
    pub self_loop: bool,
}

impl Default for RgcnConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 64,
            n_bases: 4,
            n_layers: 2,
            dropout: 0.1,
            self_loop: true,
        }
    }
}

// ============================================================================
// RgcnBasisDecomposition
// ============================================================================

/// Basis-decomposition weight for a single R-GCN layer.
///
/// Stores `n_bases` shared matrices `V_b` (each `out_dim × in_dim`) and
/// per-relation coefficient vectors `a_r` (length `n_bases`).  The effective
/// weight for relation `r` is:
/// ```text
///   W_r = Σ_b  a_{r,b}  V_b
/// ```
#[derive(Debug, Clone)]
pub struct RgcnBasisDecomposition {
    /// Shared basis matrices V_b, each of shape `(out_dim, in_dim)`.
    pub basis_matrices: Vec<Array2<f64>>,
    /// Per-relation coefficients: `coefficients[r][b] = a_{r,b}`.
    pub coefficients: Vec<Vec<f64>>,
    /// Input feature dimension.
    pub in_dim: usize,
    /// Output feature dimension.
    pub out_dim: usize,
}

impl RgcnBasisDecomposition {
    /// Build a new basis decomposition for `n_relations` relation types.
    ///
    /// # Arguments
    /// * `in_dim`      – Input feature dimensionality.
    /// * `out_dim`     – Output feature dimensionality.
    /// * `n_bases`     – Number of shared basis matrices.
    /// * `n_relations` – Number of distinct relation types.
    pub fn new(in_dim: usize, out_dim: usize, n_bases: usize, n_relations: usize) -> Result<Self> {
        if n_bases == 0 {
            return Err(GraphError::InvalidParameter {
                param: "n_bases".to_string(),
                value: "0".to_string(),
                expected: ">= 1".to_string(),
                context: "RgcnBasisDecomposition::new".to_string(),
            });
        }

        let basis_matrices: Vec<Array2<f64>> = (0..n_bases)
            .map(|_| xavier_uniform(out_dim, in_dim))
            .collect();

        let mut rng = scirs2_core::random::rng();
        let coefficients: Vec<Vec<f64>> = (0..n_relations)
            .map(|_| (0..n_bases).map(|_| rng.random::<f64>() * 0.1).collect())
            .collect();

        Ok(Self {
            basis_matrices,
            coefficients,
            in_dim,
            out_dim,
        })
    }

    /// Compute the effective weight matrix for `relation`.
    ///
    /// Returns `W_r = Σ_b a_{r,b} V_b` as an `(out_dim, in_dim)` matrix.
    pub fn effective_weight(&self, relation: usize) -> Result<Array2<f64>> {
        let coeffs =
            self.coefficients
                .get(relation)
                .ok_or_else(|| GraphError::InvalidParameter {
                    param: "relation".to_string(),
                    value: relation.to_string(),
                    expected: format!("< {}", self.coefficients.len()),
                    context: "RgcnBasisDecomposition::effective_weight".to_string(),
                })?;

        let mut w = Array2::<f64>::zeros((self.out_dim, self.in_dim));
        for (b, &coeff) in coeffs.iter().enumerate() {
            w = w + coeff * &self.basis_matrices[b];
        }
        Ok(w)
    }
}

// ============================================================================
// RgcnLayer
// ============================================================================

/// Single R-GCN layer.
///
/// # Forward pass
/// For each relation `r` and each node `i`:
/// 1. Aggregate messages:  `M_{r,i} = mean_{j ∈ N_r(i)} W_r h_j`
/// 2. Combine:             `h_i' = ReLU( Σ_r M_{r,i} + W_0 h_i )`
#[derive(Debug, Clone)]
pub struct RgcnLayer {
    /// Basis decomposition (handles all relation-specific weights)
    pub basis_decomp: RgcnBasisDecomposition,
    /// Self-loop weight W_0 (shape `out_dim × in_dim`), `None` when `self_loop=false`
    pub self_loop_weight: Option<Array2<f64>>,
    /// Bias term (length `out_dim`)
    pub bias: Array1<f64>,
    /// Number of relation types
    pub n_relations: usize,
    /// Output dimensionality
    pub out_dim: usize,
}

impl RgcnLayer {
    /// Construct a new R-GCN layer.
    ///
    /// # Arguments
    /// * `in_dim`      – Input feature dimensionality.
    /// * `out_dim`     – Output feature dimensionality.
    /// * `n_relations` – Number of distinct relation types.
    /// * `n_bases`     – Number of basis matrices for weight decomposition.
    /// * `self_loop`   – Whether to include a self-loop weight.
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        n_relations: usize,
        n_bases: usize,
        self_loop: bool,
    ) -> Result<Self> {
        let basis_decomp = RgcnBasisDecomposition::new(in_dim, out_dim, n_bases, n_relations)?;
        let self_loop_weight = if self_loop {
            Some(xavier_uniform(out_dim, in_dim))
        } else {
            None
        };
        Ok(Self {
            basis_decomp,
            self_loop_weight,
            bias: Array1::zeros(out_dim),
            n_relations,
            out_dim,
        })
    }

    /// Forward pass over a heterogeneous graph.
    ///
    /// # Arguments
    /// * `node_feats`       – Node feature matrix `(n_nodes, in_dim)`.
    /// * `adj_by_relation`  – For each relation `r`, a list of `(src, dst)` edges.
    ///
    /// # Returns
    /// Updated node feature matrix `(n_nodes, out_dim)`.
    pub fn forward(
        &self,
        node_feats: &Array2<f64>,
        adj_by_relation: &[Vec<(usize, usize)>],
    ) -> Result<Array2<f64>> {
        let n_nodes = node_feats.nrows();
        let in_dim = node_feats.ncols();

        if in_dim != self.basis_decomp.in_dim {
            return Err(GraphError::InvalidParameter {
                param: "node_feats".to_string(),
                value: format!("in_dim={}", in_dim),
                expected: format!("in_dim={}", self.basis_decomp.in_dim),
                context: "RgcnLayer::forward".to_string(),
            });
        }

        // Accumulator for the combined relational aggregation
        let mut combined = Array2::<f64>::zeros((n_nodes, self.out_dim));

        // ---- Relational aggregation ----------------------------------------
        for (r, edges) in adj_by_relation.iter().enumerate() {
            if r >= self.n_relations {
                break;
            }
            let w_r = self.basis_decomp.effective_weight(r)?;

            // Count in-degree per destination for normalisation
            let mut in_deg: Vec<usize> = vec![0usize; n_nodes];
            for &(_, dst) in edges {
                if dst < n_nodes {
                    in_deg[dst] += 1;
                }
            }

            // Aggregate: sum W_r h_j for each destination
            for &(src, dst) in edges {
                if src >= n_nodes || dst >= n_nodes {
                    continue;
                }
                let h_j = node_feats.row(src);
                // w_r has shape (out_dim, in_dim); multiply w_r @ h_j
                let msg = w_r.dot(&h_j);
                let deg = in_deg[dst].max(1) as f64;
                let mut row = combined.row_mut(dst);
                row.zip_mut_with(&msg, |acc, &m| *acc += m / deg);
            }
        }

        // ---- Self-loop contribution ----------------------------------------
        if let Some(ref w0) = self.self_loop_weight {
            for i in 0..n_nodes {
                let h_i = node_feats.row(i);
                let self_msg = w0.dot(&h_i);
                let mut row = combined.row_mut(i);
                row.zip_mut_with(&self_msg, |acc, &v| *acc += v);
            }
        }

        // ---- Bias + ReLU ---------------------------------------------------
        for mut row in combined.rows_mut() {
            row.zip_mut_with(&self.bias, |v, &b| *v += b);
        }

        Ok(relu2(&combined))
    }
}

// ============================================================================
// Rgcn — stacked R-GCN
// ============================================================================

/// Multi-layer R-GCN encoder.
///
/// Stacks `n_layers` [`RgcnLayer`]s, projecting node features from
/// `in_dim` → `hidden_dim` → … → `hidden_dim`.
#[derive(Debug, Clone)]
pub struct Rgcn {
    /// Ordered list of R-GCN layers
    pub layers: Vec<RgcnLayer>,
}

impl Rgcn {
    /// Build an R-GCN from a [`RgcnConfig`].
    ///
    /// # Arguments
    /// * `in_dim`      – Initial node feature dimensionality.
    /// * `n_relations` – Number of relation types in the graph.
    /// * `config`      – Hyper-parameter configuration.
    pub fn new(in_dim: usize, n_relations: usize, config: &RgcnConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let layer_in = if i == 0 { in_dim } else { config.hidden_dim };
            let layer = RgcnLayer::new(
                layer_in,
                config.hidden_dim,
                n_relations,
                config.n_bases,
                config.self_loop,
            )?;
            layers.push(layer);
        }
        Ok(Self { layers })
    }

    /// Forward pass through all layers.
    ///
    /// # Arguments
    /// * `node_feats`      – Initial node features `(n_nodes, in_dim)`.
    /// * `adj_by_relation` – Per-relation edge lists `(src, dst)`.
    ///
    /// # Returns
    /// Final node embeddings `(n_nodes, hidden_dim)`.
    pub fn forward(
        &self,
        node_feats: &Array2<f64>,
        adj_by_relation: &[Vec<(usize, usize)>],
    ) -> Result<Array2<f64>> {
        let mut h = node_feats.clone();
        for layer in &self.layers {
            h = layer.forward(&h, adj_by_relation)?;
        }
        Ok(h)
    }
}

// ============================================================================
// DistMult scoring
// ============================================================================

/// DistMult bilinear scoring model (Yang et al. 2015).
///
/// Score function:
/// ```text
///   score(h, r, t) = Σ_k  h_k · r_k · t_k
/// ```
///
/// This is a symmetric scoring function: `score(h,r,t) = score(t,r,h)`.
#[derive(Debug, Clone)]
pub struct DistMultScoring {
    /// Entity embedding table `(n_entities, dim)`
    pub entity_embeds: Array2<f64>,
    /// Relation embedding table `(n_relations, dim)`
    pub relation_embeds: Array2<f64>,
}

impl DistMultScoring {
    /// Create a new DistMult scorer with Xavier-initialised embeddings.
    pub fn new(n_entities: usize, n_relations: usize, dim: usize) -> Self {
        Self {
            entity_embeds: xavier_uniform(n_entities, dim),
            relation_embeds: xavier_uniform(n_relations, dim),
        }
    }

    /// Compute DistMult score for triple `(h, r, t)`.
    pub fn score(&self, h: usize, r: usize, t: usize) -> f64 {
        let h_emb = self.entity_embeds.row(h);
        let r_emb = self.relation_embeds.row(r);
        let t_emb = self.entity_embeds.row(t);
        h_emb
            .iter()
            .zip(r_emb.iter())
            .zip(t_emb.iter())
            .map(|((&hk, &rk), &tk)| hk * rk * tk)
            .sum()
    }
}

// ============================================================================
// KgScorer trait (also used by kg_completion module)
// ============================================================================

/// Trait for knowledge-graph triple scoring models.
pub trait KgScorer: Send + Sync {
    /// Return a scalar score for the triple `(h, r, t)`.
    ///
    /// Higher scores indicate more plausible triples.
    fn score(&self, h: usize, r: usize, t: usize) -> f64;
}

impl KgScorer for DistMultScoring {
    fn score(&self, h: usize, r: usize, t: usize) -> f64 {
        DistMultScoring::score(self, h, r, t)
    }
}

// ============================================================================
// RgcnLinkPredictor
// ============================================================================

/// End-to-end R-GCN encoder + DistMult decoder for link prediction.
///
/// The encoder produces node embeddings with R-GCN; the decoder scores triples
/// using the DistMult bilinear form.
#[derive(Debug, Clone)]
pub struct RgcnLinkPredictor {
    /// R-GCN encoder
    pub encoder: Rgcn,
    /// DistMult decoder
    pub decoder: DistMultScoring,
    /// Cached node embeddings (populated after calling `encode`)
    pub node_embeddings: Option<Array2<f64>>,
    /// Relation embedding dimension (same as hidden_dim)
    pub dim: usize,
    /// Number of relation types
    pub n_relations: usize,
}

impl RgcnLinkPredictor {
    /// Create a new link predictor.
    ///
    /// # Arguments
    /// * `in_dim`      – Raw input node feature dimensionality.
    /// * `n_entities`  – Number of entities in the KG.
    /// * `n_relations` – Number of relation types.
    /// * `config`      – R-GCN hyper-parameters.
    pub fn new(
        in_dim: usize,
        n_entities: usize,
        n_relations: usize,
        config: &RgcnConfig,
    ) -> Result<Self> {
        let encoder = Rgcn::new(in_dim, n_relations, config)?;
        let decoder = DistMultScoring::new(n_entities, n_relations, config.hidden_dim);
        Ok(Self {
            encoder,
            decoder,
            node_embeddings: None,
            dim: config.hidden_dim,
            n_relations,
        })
    }

    /// Run the R-GCN encoder and cache the node embeddings.
    pub fn encode(
        &mut self,
        node_feats: &Array2<f64>,
        adj_by_relation: &[Vec<(usize, usize)>],
    ) -> Result<()> {
        let h = self.encoder.forward(node_feats, adj_by_relation)?;
        self.node_embeddings = Some(h);
        Ok(())
    }

    /// Score a triple using encoder embeddings (falls back to DistMult table if
    /// `encode` has not been called).
    pub fn score_triple(&self, h: usize, r: usize, t: usize) -> f64 {
        match &self.node_embeddings {
            Some(embeds) => {
                let h_emb = embeds.row(h);
                let r_emb = self.decoder.relation_embeds.row(r);
                let t_emb = embeds.row(t);
                h_emb
                    .iter()
                    .zip(r_emb.iter())
                    .zip(t_emb.iter())
                    .map(|((&hk, &rk), &tk)| hk * rk * tk)
                    .sum()
            }
            None => self.decoder.score(h, r, t),
        }
    }
}

impl KgScorer for RgcnLinkPredictor {
    fn score(&self, h: usize, r: usize, t: usize) -> f64 {
        self.score_triple(h, r, t)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn eye_feats(n: usize, dim: usize) -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((n, dim));
        for i in 0..n.min(dim) {
            m[(i, i)] = 1.0;
        }
        m
    }

    fn single_relation_adj(n: usize) -> Vec<Vec<(usize, usize)>> {
        // A ring graph with one relation type
        let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
        vec![edges]
    }

    #[test]
    fn test_basis_decomp_single_basis_recovers_weight() {
        // With n_bases=1 and coefficient 1.0 the effective weight equals the
        // single basis matrix exactly.
        let mut decomp = RgcnBasisDecomposition::new(4, 4, 1, 1).expect("decomp");
        // Force coefficient to 1.0
        decomp.coefficients[0][0] = 1.0;
        let w = decomp.effective_weight(0).expect("w");
        let diff = (&w - &decomp.basis_matrices[0]).mapv(|v| v.abs()).sum();
        assert!(
            diff < 1e-10,
            "effective_weight should equal basis[0] when a=1"
        );
    }

    #[test]
    fn test_rgcn_layer_output_shape() {
        let feats = eye_feats(5, 8);
        let adj = single_relation_adj(5);
        let layer = RgcnLayer::new(8, 16, 1, 2, true).expect("layer");
        let out = layer.forward(&feats, &adj).expect("forward");
        assert_eq!(out.nrows(), 5);
        assert_eq!(out.ncols(), 16);
    }

    #[test]
    fn test_rgcn_layer_isolated_node_self_loop() {
        // Isolated node (no edges) should still get a non-zero output via self-loop.
        let feats = eye_feats(3, 4);
        let adj: Vec<Vec<(usize, usize)>> = vec![vec![]]; // no edges
        let layer = RgcnLayer::new(4, 4, 1, 1, true).expect("layer");
        let out = layer.forward(&feats, &adj).expect("forward");
        // Not all-zero (due to self-loop + ReLU + non-zero weights)
        let row_norm: f64 = out.row(0).iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(row_norm >= 0.0, "isolated node output must be finite");
        assert_eq!(out.nrows(), 3);
    }

    #[test]
    fn test_rgcn_stacked_output_shape() {
        let config = RgcnConfig {
            hidden_dim: 8,
            n_bases: 2,
            n_layers: 3,
            ..Default::default()
        };
        let feats = eye_feats(6, 4);
        let adj = single_relation_adj(6);
        let rgcn = Rgcn::new(4, 1, &config).expect("rgcn");
        let out = rgcn.forward(&feats, &adj).expect("forward");
        assert_eq!(out.nrows(), 6);
        assert_eq!(out.ncols(), 8);
    }

    #[test]
    fn test_distmult_symmetry() {
        // DistMult scoring is symmetric: score(h,r,t) == score(t,r,h)
        let dm = DistMultScoring::new(4, 2, 8);
        let s1 = dm.score(0, 0, 1);
        let s2 = dm.score(1, 0, 0);
        assert!((s1 - s2).abs() < 1e-10, "DistMult should be symmetric");
    }

    #[test]
    fn test_rgcn_link_predictor_encode() {
        let config = RgcnConfig::default();
        let mut predictor = RgcnLinkPredictor::new(4, 5, 2, &config).expect("predictor");
        let feats = eye_feats(5, 4);
        let adj: Vec<Vec<(usize, usize)>> = vec![vec![(0, 1), (1, 2)], vec![(2, 3)]];
        predictor.encode(&feats, &adj).expect("encode");
        // After encoding, node_embeddings should be populated
        assert!(predictor.node_embeddings.is_some());
        let embeds = predictor.node_embeddings.as_ref().expect("embeds");
        assert_eq!(embeds.nrows(), 5);
        assert_eq!(embeds.ncols(), config.hidden_dim);
        // Score should be finite
        let s = predictor.score_triple(0, 0, 1);
        assert!(s.is_finite());
    }
}
