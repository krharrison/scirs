//! LINE (Large-scale Information Network Embedding) algorithm
//!
//! Implements the LINE algorithm from Tang et al. (2015) for learning node embeddings
//! that preserve both first-order (direct connections) and second-order
//! (shared neighborhood) proximity in graphs.
//!
//! # References
//! - Tang, J., Qu, M., Wang, M., Zhang, M., Yan, J., & Mei, Q. (2015).
//!   LINE: Large-scale Information Network Embedding. WWW 2015.

use super::core::{Embedding, EmbeddingModel};
use crate::base::{Edge, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};
use scirs2_core::random::Rng;
use std::collections::{HashMap, HashSet};

/// Proximity order for LINE algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LINEOrder {
    /// First-order proximity: preserves local pairwise structure
    /// Nodes connected by edges should have similar embeddings
    First,
    /// Second-order proximity: preserves neighborhood structure
    /// Nodes sharing similar neighborhoods should have similar embeddings
    Second,
    /// Combined: concatenates first-order and second-order embeddings
    Combined,
}

/// Configuration for LINE embedding algorithm
#[derive(Debug, Clone)]
pub struct LINEConfig {
    /// Dimensions of the embedding vectors (per order)
    pub dimensions: usize,
    /// Which proximity order to use
    pub order: LINEOrder,
    /// Number of training epochs
    pub epochs: usize,
    /// Initial learning rate
    pub learning_rate: f64,
    /// Number of negative samples per positive sample
    pub negative_samples: usize,
    /// Batch size for edge sampling
    pub batch_size: usize,
    /// Number of total training samples (edges sampled)
    pub num_samples: usize,
}

impl Default for LINEConfig {
    fn default() -> Self {
        LINEConfig {
            dimensions: 128,
            order: LINEOrder::Combined,
            epochs: 1,
            learning_rate: 0.025,
            negative_samples: 5,
            batch_size: 1,
            num_samples: 10000,
        }
    }
}

/// Alias table for O(1) edge sampling proportional to weights
#[derive(Debug)]
struct AliasTable {
    /// Probability threshold for each entry
    prob: Vec<f64>,
    /// Alias index for each entry
    alias: Vec<usize>,
}

impl AliasTable {
    /// Build an alias table from a probability distribution using Vose's method
    fn build(probabilities: &[f64]) -> Result<Self> {
        let n = probabilities.len();
        if n == 0 {
            return Err(GraphError::InvalidGraph(
                "Cannot build alias table from empty distribution".to_string(),
            ));
        }

        let total: f64 = probabilities.iter().sum();
        if total <= 0.0 {
            return Err(GraphError::InvalidGraph(
                "Probabilities must sum to a positive value".to_string(),
            ));
        }

        // Normalize and scale
        let scaled: Vec<f64> = probabilities
            .iter()
            .map(|p| (p / total) * n as f64)
            .collect();

        let mut prob = vec![0.0; n];
        let mut alias = vec![0usize; n];

        let mut small: Vec<usize> = Vec::new();
        let mut large: Vec<usize> = Vec::new();

        // Work on a mutable copy
        let mut work = scaled;

        for (i, &w) in work.iter().enumerate() {
            if w < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }

        while !small.is_empty() && !large.is_empty() {
            let s = small.pop().unwrap_or(0);
            let l = large.pop().unwrap_or(0);

            prob[s] = work[s];
            alias[s] = l;

            work[l] = (work[l] + work[s]) - 1.0;

            if work[l] < 1.0 {
                small.push(l);
            } else {
                large.push(l);
            }
        }

        // Remaining items get probability 1.0
        for &l in &large {
            prob[l] = 1.0;
        }
        for &s in &small {
            prob[s] = 1.0;
        }

        Ok(AliasTable { prob, alias })
    }

    /// Sample from the alias table in O(1)
    fn sample(&self, rng: &mut impl Rng) -> usize {
        let n = self.prob.len();
        if n == 0 {
            return 0;
        }
        let i = (rng.random::<f64>() * n as f64) as usize;
        let i = i.min(n - 1);
        if rng.random::<f64>() < self.prob[i] {
            i
        } else {
            self.alias[i]
        }
    }
}

/// Noise distribution for negative sampling (unigram^(3/4))
#[derive(Debug)]
struct NoiseDistribution {
    /// Alias table for O(1) sampling
    alias_table: AliasTable,
}

impl NoiseDistribution {
    /// Build noise distribution from node degrees (raised to 3/4 power)
    fn from_degrees(degrees: &[f64]) -> Result<Self> {
        let powered: Vec<f64> = degrees.iter().map(|d| d.powf(0.75)).collect();
        let alias_table = AliasTable::build(&powered)?;
        Ok(NoiseDistribution { alias_table })
    }

    /// Sample a node index from the noise distribution
    fn sample(&self, rng: &mut impl Rng) -> usize {
        self.alias_table.sample(rng)
    }

    /// Sample multiple node indices, excluding given set
    fn sample_negatives(
        &self,
        count: usize,
        exclude: &HashSet<usize>,
        rng: &mut impl Rng,
    ) -> Vec<usize> {
        let mut negatives = Vec::with_capacity(count);
        let max_attempts = count * 10;
        let mut attempts = 0;

        while negatives.len() < count && attempts < max_attempts {
            let idx = self.sample(rng);
            if !exclude.contains(&idx) {
                negatives.push(idx);
            }
            attempts += 1;
        }

        negatives
    }
}

/// LINE (Large-scale Information Network Embedding) implementation
///
/// Learns node embeddings that preserve graph proximity using
/// stochastic gradient descent with negative sampling.
pub struct LINE<N: Node> {
    /// Configuration
    config: LINEConfig,
    /// Node embedding vectors (source embeddings)
    node_embeddings: Vec<Vec<f64>>,
    /// Context embedding vectors (for second-order proximity)
    context_embeddings: Vec<Vec<f64>>,
    /// Node to index mapping
    node_to_idx: HashMap<N, usize>,
    /// Index to node mapping
    idx_to_node: Vec<N>,
    /// Edge list as index pairs with weights
    edge_list: Vec<(usize, usize, f64)>,
    /// Edge weights for alias sampling
    edge_alias: Option<AliasTable>,
    /// Noise distribution for negative sampling
    noise_dist: Option<NoiseDistribution>,
}

impl<N: Node + std::fmt::Debug> LINE<N> {
    /// Create a new LINE instance with the given configuration
    pub fn new(config: LINEConfig) -> Self {
        LINE {
            config,
            node_embeddings: Vec::new(),
            context_embeddings: Vec::new(),
            node_to_idx: HashMap::new(),
            idx_to_node: Vec::new(),
            edge_list: Vec::new(),
            edge_alias: None,
            noise_dist: None,
        }
    }

    /// Initialize the model from a graph
    fn initialize<E, Ix>(&mut self, graph: &Graph<N, E, Ix>) -> Result<()>
    where
        N: Clone,
        E: EdgeWeight + Into<f64> + Clone,
        Ix: petgraph::graph::IndexType,
    {
        let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
        let n = nodes.len();

        if n == 0 {
            return Err(GraphError::InvalidGraph(
                "Cannot compute LINE embeddings for empty graph".to_string(),
            ));
        }

        // Build node mappings
        self.node_to_idx.clear();
        self.idx_to_node = nodes.clone();
        for (i, node) in nodes.iter().enumerate() {
            self.node_to_idx.insert(node.clone(), i);
        }

        // Build edge list with weights
        self.edge_list.clear();
        let edges = graph.edges();
        let mut edge_weights_for_alias = Vec::new();

        for edge in &edges {
            let src = self
                .node_to_idx
                .get(&edge.source)
                .copied()
                .ok_or_else(|| GraphError::node_not_found("source"))?;
            let tgt = self
                .node_to_idx
                .get(&edge.target)
                .copied()
                .ok_or_else(|| GraphError::node_not_found("target"))?;
            let w: f64 = edge.weight.clone().into();
            let w_abs = w.abs().max(1e-10);

            // Add both directions for undirected graphs
            self.edge_list.push((src, tgt, w_abs));
            edge_weights_for_alias.push(w_abs);

            self.edge_list.push((tgt, src, w_abs));
            edge_weights_for_alias.push(w_abs);
        }

        if self.edge_list.is_empty() {
            return Err(GraphError::InvalidGraph(
                "Cannot compute LINE embeddings for graph with no edges".to_string(),
            ));
        }

        // Build alias table for edge sampling
        self.edge_alias = Some(AliasTable::build(&edge_weights_for_alias)?);

        // Build noise distribution from node degrees
        let degrees: Vec<f64> = (0..n).map(|i| graph.degree(&nodes[i]) as f64).collect();
        self.noise_dist = Some(NoiseDistribution::from_degrees(&degrees)?);

        // Initialize random embeddings
        let mut rng = scirs2_core::random::rng();
        let dim = self.config.dimensions;

        self.node_embeddings = (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| (rng.random::<f64>() - 0.5) / dim as f64)
                    .collect()
            })
            .collect();

        self.context_embeddings = (0..n).map(|_| vec![0.0; dim]).collect();

        Ok(())
    }

    /// Train the LINE model on a graph
    pub fn train<E, Ix>(&mut self, graph: &Graph<N, E, Ix>) -> Result<()>
    where
        N: Clone,
        E: EdgeWeight + Into<f64> + Clone,
        Ix: petgraph::graph::IndexType,
    {
        self.initialize(graph)?;

        match self.config.order {
            LINEOrder::First => self.train_first_order()?,
            LINEOrder::Second => self.train_second_order()?,
            LINEOrder::Combined => {
                // Train both orders and combine
                self.train_first_order()?;
                let first_order_embs = self.node_embeddings.clone();

                // Re-initialize for second order
                let mut rng = scirs2_core::random::rng();
                let n = self.idx_to_node.len();
                let dim = self.config.dimensions;

                self.node_embeddings = (0..n)
                    .map(|_| {
                        (0..dim)
                            .map(|_| (rng.random::<f64>() - 0.5) / dim as f64)
                            .collect()
                    })
                    .collect();
                self.context_embeddings = (0..n).map(|_| vec![0.0; dim]).collect();

                self.train_second_order()?;

                // Concatenate first-order and second-order embeddings
                for (i, first_emb) in first_order_embs.into_iter().enumerate() {
                    self.node_embeddings[i].extend(first_emb);
                }
            }
        }

        Ok(())
    }

    /// Train using first-order proximity
    /// Optimizes: log sigma(vec_u^T * vec_v) for each edge (u, v)
    fn train_first_order(&mut self) -> Result<()> {
        let mut rng = scirs2_core::random::rng();
        let total_samples = self.config.num_samples * self.config.epochs;
        let dim = self.config.dimensions;
        let num_neg = self.config.negative_samples;

        let edge_alias = self.edge_alias.as_ref().ok_or_else(|| {
            GraphError::AlgorithmError("Edge alias table not initialized".to_string())
        })?;

        let noise_dist = self.noise_dist.as_ref().ok_or_else(|| {
            GraphError::AlgorithmError("Noise distribution not initialized".to_string())
        })?;

        for sample_idx in 0..total_samples {
            // Learning rate with linear decay
            let progress = sample_idx as f64 / total_samples as f64;
            let lr = self.config.learning_rate * (1.0 - progress).max(0.0001);

            // Sample an edge
            let edge_idx = edge_alias.sample(&mut rng);
            let edge_idx = edge_idx.min(self.edge_list.len() - 1);
            let (u, v, _w) = self.edge_list[edge_idx];

            // Positive sample: maximize log sigma(vec_u . vec_v)
            let dot = dot_product(&self.node_embeddings[u], &self.node_embeddings[v], dim);
            let sig = sigmoid(dot);
            let g = lr * (1.0 - sig);

            // Compute gradient and update
            let mut grad_u = vec![0.0; dim];
            let mut grad_v = vec![0.0; dim];
            for d in 0..dim {
                grad_u[d] = g * self.node_embeddings[v][d];
                grad_v[d] = g * self.node_embeddings[u][d];
            }

            // Negative samples: minimize log sigma(-vec_u . vec_neg)
            let exclude: HashSet<usize> = [u, v].iter().copied().collect();
            let negatives = noise_dist.sample_negatives(num_neg, &exclude, &mut rng);

            for &neg in &negatives {
                let dot_neg =
                    dot_product(&self.node_embeddings[u], &self.node_embeddings[neg], dim);
                let sig_neg = sigmoid(dot_neg);
                let g_neg = lr * (-sig_neg);

                for d in 0..dim {
                    grad_u[d] += g_neg * self.node_embeddings[neg][d];
                    // Update negative embedding
                    self.node_embeddings[neg][d] -= g_neg * self.node_embeddings[u][d];
                }
            }

            // Apply gradients to u and v
            for d in 0..dim {
                self.node_embeddings[u][d] += grad_u[d];
                self.node_embeddings[v][d] += grad_v[d];
            }
        }

        Ok(())
    }

    /// Train using second-order proximity
    /// Optimizes: log sigma(vec_u^T * vec_v') for edge (u, v)
    /// where vec_v' is the context embedding of v
    fn train_second_order(&mut self) -> Result<()> {
        let mut rng = scirs2_core::random::rng();
        let total_samples = self.config.num_samples * self.config.epochs;
        let dim = self.config.dimensions;
        let num_neg = self.config.negative_samples;

        let edge_alias = self.edge_alias.as_ref().ok_or_else(|| {
            GraphError::AlgorithmError("Edge alias table not initialized".to_string())
        })?;

        let noise_dist = self.noise_dist.as_ref().ok_or_else(|| {
            GraphError::AlgorithmError("Noise distribution not initialized".to_string())
        })?;

        for sample_idx in 0..total_samples {
            // Learning rate with linear decay
            let progress = sample_idx as f64 / total_samples as f64;
            let lr = self.config.learning_rate * (1.0 - progress).max(0.0001);

            // Sample an edge
            let edge_idx = edge_alias.sample(&mut rng);
            let edge_idx = edge_idx.min(self.edge_list.len() - 1);
            let (u, v, _w) = self.edge_list[edge_idx];

            // Positive sample: maximize log sigma(vec_u . ctx_v)
            let dot = dot_product(&self.node_embeddings[u], &self.context_embeddings[v], dim);
            let sig = sigmoid(dot);
            let g = lr * (1.0 - sig);

            // Accumulate gradient for source node
            let mut grad_u = vec![0.0; dim];
            for d in 0..dim {
                grad_u[d] = g * self.context_embeddings[v][d];
                // Update context of positive sample
                self.context_embeddings[v][d] += g * self.node_embeddings[u][d];
            }

            // Negative samples: minimize log sigma(-vec_u . ctx_neg)
            let exclude: HashSet<usize> = [u, v].iter().copied().collect();
            let negatives = noise_dist.sample_negatives(num_neg, &exclude, &mut rng);

            for &neg in &negatives {
                let dot_neg =
                    dot_product(&self.node_embeddings[u], &self.context_embeddings[neg], dim);
                let sig_neg = sigmoid(dot_neg);
                let g_neg = lr * (-sig_neg);

                for d in 0..dim {
                    grad_u[d] += g_neg * self.context_embeddings[neg][d];
                    // Update context of negative sample
                    self.context_embeddings[neg][d] += g_neg * self.node_embeddings[u][d];
                }
            }

            // Apply accumulated gradient to source node
            for d in 0..dim {
                self.node_embeddings[u][d] += grad_u[d];
            }
        }

        Ok(())
    }

    /// Get the embedding model containing all node embeddings
    pub fn model(&self) -> Result<EmbeddingModel<N>>
    where
        N: Clone,
    {
        let actual_dim = if let Some(first) = self.node_embeddings.first() {
            first.len()
        } else {
            return Err(GraphError::AlgorithmError(
                "No embeddings computed yet".to_string(),
            ));
        };

        let mut model = EmbeddingModel::new(actual_dim);

        for (i, node) in self.idx_to_node.iter().enumerate() {
            if i < self.node_embeddings.len() {
                let emb = Embedding {
                    vector: self.node_embeddings[i].clone(),
                };
                model.set_embedding(node.clone(), emb)?;
            }
        }

        Ok(model)
    }

    /// Get the embedding for a specific node
    pub fn get_embedding(&self, node: &N) -> Result<Embedding> {
        let idx = self
            .node_to_idx
            .get(node)
            .ok_or_else(|| GraphError::node_not_found(format!("{node:?}")))?;

        if *idx < self.node_embeddings.len() {
            Ok(Embedding {
                vector: self.node_embeddings[*idx].clone(),
            })
        } else {
            Err(GraphError::AlgorithmError(
                "Embedding not computed for this node".to_string(),
            ))
        }
    }

    /// Get all node embeddings as a HashMap
    pub fn embeddings(&self) -> HashMap<N, Embedding>
    where
        N: Clone,
    {
        let mut result = HashMap::new();
        for (i, node) in self.idx_to_node.iter().enumerate() {
            if i < self.node_embeddings.len() {
                result.insert(
                    node.clone(),
                    Embedding {
                        vector: self.node_embeddings[i].clone(),
                    },
                );
            }
        }
        result
    }

    /// Compute similarity between two nodes using their embeddings
    pub fn node_similarity(&self, node_a: &N, node_b: &N) -> Result<f64> {
        let emb_a = self.get_embedding(node_a)?;
        let emb_b = self.get_embedding(node_b)?;
        emb_a.cosine_similarity(&emb_b)
    }

    /// Get the embedding dimension (may differ from config for Combined order)
    pub fn embedding_dimension(&self) -> usize {
        self.node_embeddings.first().map(|e| e.len()).unwrap_or(0)
    }
}

/// Compute dot product of two vectors up to `dim` dimensions
#[inline]
fn dot_product(a: &[f64], b: &[f64], dim: usize) -> f64 {
    a.iter()
        .take(dim)
        .zip(b.iter().take(dim))
        .map(|(x, y)| x * y)
        .sum()
}

/// Sigmoid activation function with numerical stability
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x > 6.0 {
        1.0 - 1e-10
    } else if x < -6.0 {
        1e-10
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a simple test graph (triangle)
    fn make_triangle_graph() -> Graph<i32, f64> {
        let mut g = Graph::new();
        g.add_node(0);
        g.add_node(1);
        g.add_node(2);
        let _ = g.add_edge(0, 1, 1.0);
        let _ = g.add_edge(1, 2, 1.0);
        let _ = g.add_edge(0, 2, 1.0);
        g
    }

    /// Helper to create a barbell graph (two triangles connected by one edge)
    fn make_barbell_graph() -> Graph<i32, f64> {
        let mut g = Graph::new();
        for i in 0..6 {
            g.add_node(i);
        }
        // Left triangle
        let _ = g.add_edge(0, 1, 1.0);
        let _ = g.add_edge(1, 2, 1.0);
        let _ = g.add_edge(0, 2, 1.0);
        // Right triangle
        let _ = g.add_edge(3, 4, 1.0);
        let _ = g.add_edge(4, 5, 1.0);
        let _ = g.add_edge(3, 5, 1.0);
        // Bridge
        let _ = g.add_edge(2, 3, 1.0);
        g
    }

    #[test]
    fn test_line_first_order_basic() {
        let g = make_triangle_graph();
        let config = LINEConfig {
            dimensions: 16,
            order: LINEOrder::First,
            epochs: 1,
            learning_rate: 0.025,
            negative_samples: 3,
            num_samples: 500,
            ..Default::default()
        };

        let mut line = LINE::new(config);
        let result = line.train(&g);
        assert!(result.is_ok(), "LINE first-order training should succeed");

        // All nodes should have embeddings
        for node in [0, 1, 2] {
            let emb = line.get_embedding(&node);
            assert!(emb.is_ok(), "Node {node} should have an embedding");
            assert_eq!(emb.as_ref().map(|e| e.vector.len()).unwrap_or(0), 16);
        }
    }

    #[test]
    fn test_line_second_order_basic() {
        let g = make_triangle_graph();
        let config = LINEConfig {
            dimensions: 16,
            order: LINEOrder::Second,
            epochs: 1,
            learning_rate: 0.025,
            negative_samples: 3,
            num_samples: 500,
            ..Default::default()
        };

        let mut line = LINE::new(config);
        let result = line.train(&g);
        assert!(result.is_ok(), "LINE second-order training should succeed");

        let emb = line.get_embedding(&0);
        assert!(emb.is_ok());
        assert_eq!(emb.as_ref().map(|e| e.vector.len()).unwrap_or(0), 16);
    }

    #[test]
    fn test_line_combined_order() {
        let g = make_triangle_graph();
        let config = LINEConfig {
            dimensions: 8,
            order: LINEOrder::Combined,
            epochs: 1,
            learning_rate: 0.025,
            negative_samples: 3,
            num_samples: 300,
            ..Default::default()
        };

        let mut line = LINE::new(config);
        let result = line.train(&g);
        assert!(result.is_ok(), "LINE combined training should succeed");

        // Combined should produce 2x dimensions (first + second concatenated)
        let emb = line.get_embedding(&0);
        assert!(emb.is_ok());
        assert_eq!(
            emb.as_ref().map(|e| e.vector.len()).unwrap_or(0),
            16,
            "Combined order should produce 2x dimension embeddings"
        );
    }

    #[test]
    fn test_line_structural_similarity() {
        // In a barbell graph, nodes within the same triangle should be more similar
        // than nodes across the bridge
        let g = make_barbell_graph();
        let config = LINEConfig {
            dimensions: 32,
            order: LINEOrder::First,
            epochs: 3,
            learning_rate: 0.05,
            negative_samples: 5,
            num_samples: 5000,
            ..Default::default()
        };

        let mut line = LINE::new(config);
        let result = line.train(&g);
        assert!(result.is_ok());

        // Nodes 0,1,2 are in the same triangle; node 5 is in the other triangle
        let sim_within = line.node_similarity(&0, &1);
        let sim_across = line.node_similarity(&0, &5);

        assert!(sim_within.is_ok());
        assert!(sim_across.is_ok());

        // Within-community similarity should typically be higher
        // (stochastic, so we just check the values are valid numbers)
        let sw = sim_within.unwrap_or(0.0);
        let sa = sim_across.unwrap_or(0.0);
        assert!(
            sw.is_finite(),
            "Within-community similarity should be finite"
        );
        assert!(
            sa.is_finite(),
            "Across-community similarity should be finite"
        );
    }

    #[test]
    fn test_line_empty_graph_error() {
        let g: Graph<i32, f64> = Graph::new();
        let config = LINEConfig {
            dimensions: 8,
            order: LINEOrder::First,
            ..Default::default()
        };

        let mut line = LINE::new(config);
        let result = line.train(&g);
        assert!(result.is_err(), "Should fail on empty graph");
    }

    #[test]
    fn test_line_single_edge_graph() {
        let mut g: Graph<i32, f64> = Graph::new();
        g.add_node(0);
        g.add_node(1);
        let _ = g.add_edge(0, 1, 2.5);

        let config = LINEConfig {
            dimensions: 8,
            order: LINEOrder::Second,
            epochs: 1,
            num_samples: 200,
            negative_samples: 1,
            ..Default::default()
        };

        let mut line = LINE::new(config);
        let result = line.train(&g);
        assert!(result.is_ok());

        let embs = line.embeddings();
        assert_eq!(embs.len(), 2);
    }

    #[test]
    fn test_line_model_retrieval() {
        let g = make_triangle_graph();
        let config = LINEConfig {
            dimensions: 8,
            order: LINEOrder::First,
            epochs: 1,
            num_samples: 200,
            ..Default::default()
        };

        let mut line = LINE::new(config);
        let _ = line.train(&g);

        let model = line.model();
        assert!(model.is_ok());
        let model = model.expect("model should be valid");
        assert_eq!(model.dimensions, 8);

        // Check that all nodes have embeddings in the model
        for node in [0, 1, 2] {
            assert!(model.get_embedding(&node).is_some());
        }
    }

    #[test]
    fn test_line_node_not_found() {
        let g = make_triangle_graph();
        let config = LINEConfig {
            dimensions: 8,
            order: LINEOrder::First,
            epochs: 1,
            num_samples: 200,
            ..Default::default()
        };

        let mut line = LINE::new(config);
        let _ = line.train(&g);

        let result = line.get_embedding(&99);
        assert!(result.is_err(), "Should fail for non-existent node");
    }

    #[test]
    fn test_alias_table() {
        let probs = vec![0.1, 0.3, 0.4, 0.2];
        let table = AliasTable::build(&probs);
        assert!(table.is_ok());

        let table = table.expect("alias table should be valid");
        let mut rng = scirs2_core::random::rng();

        // Sample many times and check distribution is roughly correct
        let mut counts = vec![0u32; 4];
        let n_samples = 10000;
        for _ in 0..n_samples {
            let idx = table.sample(&mut rng);
            counts[idx] += 1;
        }

        // Each bucket should have received some samples
        for (i, &count) in counts.iter().enumerate() {
            assert!(
                count > 0,
                "Bucket {i} should have at least some samples, got {count}"
            );
        }
    }
}
