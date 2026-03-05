//! Node2Vec graph embedding algorithm
//!
//! Implements the Node2Vec algorithm from Grover & Leskovec (2016) for learning
//! continuous feature representations for nodes in networks. Uses biased random
//! walks with return parameter p and in-out parameter q to explore neighborhoods.
//!
//! # References
//! - Grover, A. & Leskovec, J. (2016). node2vec: Scalable Feature Learning for Networks. KDD 2016.

use super::core::EmbeddingModel;
use super::negative_sampling::NegativeSampler;
use super::random_walk::RandomWalkGenerator;
use super::types::{Node2VecConfig, RandomWalk};
use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::Result;
use scirs2_core::random::seq::SliceRandom;

/// Node2Vec embedding algorithm
///
/// Learns node embeddings using biased second-order random walks followed
/// by skip-gram optimization with negative sampling.
pub struct Node2Vec<N: Node> {
    config: Node2VecConfig,
    model: EmbeddingModel<N>,
    walk_generator: RandomWalkGenerator<N>,
}

impl<N: Node> Node2Vec<N> {
    /// Create a new Node2Vec instance
    pub fn new(config: Node2VecConfig) -> Self {
        Node2Vec {
            model: EmbeddingModel::new(config.dimensions),
            config,
            walk_generator: RandomWalkGenerator::new(),
        }
    }

    /// Generate training data (biased random walks) for Node2Vec on undirected graph
    pub fn generate_walks<E, Ix>(&mut self, graph: &Graph<N, E, Ix>) -> Result<Vec<RandomWalk<N>>>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight + Into<f64>,
        Ix: petgraph::graph::IndexType,
    {
        let mut all_walks = Vec::new();

        for node in graph.nodes() {
            for _ in 0..self.config.num_walks {
                let walk = self.walk_generator.node2vec_walk(
                    graph,
                    node,
                    self.config.walk_length,
                    self.config.p,
                    self.config.q,
                )?;
                all_walks.push(walk);
            }
        }

        Ok(all_walks)
    }

    /// Generate training data (biased random walks) for Node2Vec on directed graph
    pub fn generate_walks_digraph<E, Ix>(
        &mut self,
        graph: &DiGraph<N, E, Ix>,
    ) -> Result<Vec<RandomWalk<N>>>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight + Into<f64>,
        Ix: petgraph::graph::IndexType,
    {
        let mut all_walks = Vec::new();

        for node in graph.nodes() {
            for _ in 0..self.config.num_walks {
                let walk = self.walk_generator.node2vec_walk_digraph(
                    graph,
                    node,
                    self.config.walk_length,
                    self.config.p,
                    self.config.q,
                )?;
                all_walks.push(walk);
            }
        }

        Ok(all_walks)
    }

    /// Train the Node2Vec model on an undirected graph
    pub fn train<E, Ix>(&mut self, graph: &Graph<N, E, Ix>) -> Result<()>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight + Into<f64>,
        Ix: petgraph::graph::IndexType,
    {
        // Initialize random embeddings
        let mut rng = scirs2_core::random::rng();
        self.model.initialize_random(graph, &mut rng);

        // Create negative sampler
        let negative_sampler = NegativeSampler::new(graph);

        // Training loop over epochs
        for epoch in 0..self.config.epochs {
            // Generate walks for this epoch
            let walks = self.generate_walks(graph)?;

            // Generate context pairs from walks
            let context_pairs =
                EmbeddingModel::generate_context_pairs(&walks, self.config.window_size);

            // Shuffle pairs for better training
            let mut shuffled_pairs = context_pairs;
            shuffled_pairs.shuffle(&mut rng);

            // Train skip-gram model with negative sampling
            // Linear learning rate decay
            let current_lr = self.config.learning_rate
                * (1.0 - epoch as f64 / self.config.epochs as f64).max(0.0001);

            self.model.train_skip_gram(
                &shuffled_pairs,
                &negative_sampler,
                current_lr,
                self.config.negative_samples,
                &mut rng,
            )?;
        }

        Ok(())
    }

    /// Train the Node2Vec model on a directed graph
    pub fn train_digraph<E, Ix>(&mut self, graph: &DiGraph<N, E, Ix>) -> Result<()>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight + Into<f64>,
        Ix: petgraph::graph::IndexType,
    {
        // Initialize random embeddings for directed graph
        let mut rng = scirs2_core::random::rng();
        self.model.initialize_random_digraph(graph, &mut rng);

        // Create negative sampler from the undirected view
        // For DiGraph, we build a temporary sampler from node degrees
        let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
        let node_degrees: Vec<f64> = nodes.iter().map(|n| graph.degree(n) as f64).collect();

        // Build cumulative distribution for negative sampling
        let total_degree: f64 = node_degrees.iter().sum();
        let frequencies: Vec<f64> = node_degrees
            .iter()
            .map(|d| (d / total_degree.max(1.0)).powf(0.75))
            .collect();
        let total_freq: f64 = frequencies.iter().sum();
        let normalized: Vec<f64> = frequencies
            .iter()
            .map(|f| f / total_freq.max(1e-10))
            .collect();

        let mut cumulative = vec![0.0; normalized.len()];
        if !cumulative.is_empty() {
            cumulative[0] = normalized[0];
            for i in 1..normalized.len() {
                cumulative[i] = cumulative[i - 1] + normalized[i];
            }
        }

        // Training loop
        for epoch in 0..self.config.epochs {
            let walks = self.generate_walks_digraph(graph)?;
            let context_pairs =
                EmbeddingModel::generate_context_pairs(&walks, self.config.window_size);

            let mut shuffled_pairs = context_pairs;
            shuffled_pairs.shuffle(&mut rng);

            let current_lr = self.config.learning_rate
                * (1.0 - epoch as f64 / self.config.epochs as f64).max(0.0001);

            // Manual skip-gram training for directed graphs
            // (since NegativeSampler is built for Graph, not DiGraph)
            for pair in &shuffled_pairs {
                self.train_pair_digraph(
                    pair,
                    &nodes,
                    &cumulative,
                    current_lr,
                    self.config.negative_samples,
                    &mut rng,
                );
            }
        }

        Ok(())
    }

    /// Train on a single context pair for directed graphs
    fn train_pair_digraph(
        &mut self,
        pair: &super::types::ContextPair<N>,
        nodes: &[N],
        cumulative: &[f64],
        learning_rate: f64,
        num_negative: usize,
        rng: &mut impl scirs2_core::random::Rng,
    ) where
        N: Clone,
    {
        let dim = self.config.dimensions;

        // Get target embedding
        let target_emb = match self.model.embeddings.get(&pair.target) {
            Some(e) => e.clone(),
            None => return,
        };

        // Get context embedding
        let context_emb = match self.model.context_embeddings.get(&pair.context) {
            Some(e) => e.clone(),
            None => return,
        };

        // Positive sample gradient
        let dot: f64 = target_emb
            .vector
            .iter()
            .zip(context_emb.vector.iter())
            .map(|(a, b)| a * b)
            .sum();
        let sig = 1.0 / (1.0 + (-dot).exp());
        let g = learning_rate * (1.0 - sig);

        let mut target_grad = vec![0.0; dim];
        for d in 0..dim {
            target_grad[d] += g * context_emb.vector[d];
        }

        // Update context embedding
        if let Some(ctx) = self.model.context_embeddings.get_mut(&pair.context) {
            for d in 0..dim {
                ctx.vector[d] += g * target_emb.vector[d];
            }
        }

        // Negative samples
        for _ in 0..num_negative {
            let r = rng.random::<f64>();
            let neg_idx = cumulative
                .iter()
                .position(|&c| r <= c)
                .unwrap_or(cumulative.len().saturating_sub(1));

            if neg_idx >= nodes.len() {
                continue;
            }

            let neg_node = &nodes[neg_idx];
            if neg_node == &pair.target || neg_node == &pair.context {
                continue;
            }

            if let Some(neg_emb) = self.model.context_embeddings.get(neg_node) {
                let neg_dot: f64 = target_emb
                    .vector
                    .iter()
                    .zip(neg_emb.vector.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let neg_sig = 1.0 / (1.0 + (-neg_dot).exp());
                let neg_g = learning_rate * (-neg_sig);

                for d in 0..dim {
                    target_grad[d] += neg_g * neg_emb.vector[d];
                }

                // Update negative context
                if let Some(neg_ctx) = self.model.context_embeddings.get_mut(neg_node) {
                    for d in 0..dim {
                        neg_ctx.vector[d] += neg_g * target_emb.vector[d];
                    }
                }
            }
        }

        // Apply accumulated gradient to target
        if let Some(target) = self.model.embeddings.get_mut(&pair.target) {
            for d in 0..dim {
                target.vector[d] += target_grad[d];
            }
        }
    }

    /// Get the trained model
    pub fn model(&self) -> &EmbeddingModel<N> {
        &self.model
    }

    /// Get mutable reference to the model
    pub fn model_mut(&mut self) -> &mut EmbeddingModel<N> {
        &mut self.model
    }

    /// Get the configuration
    pub fn config(&self) -> &Node2VecConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triangle() -> Graph<i32, f64> {
        let mut g = Graph::new();
        for i in 0..3 {
            g.add_node(i);
        }
        let _ = g.add_edge(0, 1, 1.0);
        let _ = g.add_edge(1, 2, 1.0);
        let _ = g.add_edge(0, 2, 1.0);
        g
    }

    fn make_star_graph() -> Graph<i32, f64> {
        let mut g = Graph::new();
        for i in 0..5 {
            g.add_node(i);
        }
        // Node 0 is the center
        for i in 1..5 {
            let _ = g.add_edge(0, i, 1.0);
        }
        g
    }

    fn make_directed_chain() -> DiGraph<i32, f64> {
        let mut g = DiGraph::new();
        for i in 0..5 {
            g.add_node(i);
        }
        let _ = g.add_edge(0, 1, 1.0);
        let _ = g.add_edge(1, 2, 1.0);
        let _ = g.add_edge(2, 3, 1.0);
        let _ = g.add_edge(3, 4, 1.0);
        g
    }

    #[test]
    fn test_node2vec_train_basic() {
        let g = make_triangle();
        let config = Node2VecConfig {
            dimensions: 8,
            walk_length: 5,
            num_walks: 3,
            window_size: 2,
            p: 1.0,
            q: 1.0,
            epochs: 2,
            learning_rate: 0.025,
            negative_samples: 2,
        };

        let mut n2v = Node2Vec::new(config);
        let result = n2v.train(&g);
        assert!(result.is_ok(), "Node2Vec training should succeed");

        // All nodes should have embeddings
        for node in [0, 1, 2] {
            assert!(
                n2v.model().get_embedding(&node).is_some(),
                "Node {node} should have an embedding"
            );
        }
    }

    #[test]
    fn test_node2vec_walk_generation() {
        let g = make_triangle();
        let config = Node2VecConfig {
            dimensions: 8,
            walk_length: 10,
            num_walks: 2,
            p: 1.0,
            q: 1.0,
            ..Default::default()
        };

        let mut n2v = Node2Vec::new(config);
        let walks = n2v.generate_walks(&g);
        assert!(walks.is_ok());

        let walks = walks.expect("walks should be valid");
        // 3 nodes * 2 walks per node = 6 walks total
        assert_eq!(walks.len(), 6);

        // Each walk should have at most walk_length nodes
        for walk in &walks {
            assert!(walk.nodes.len() <= 10);
            assert!(!walk.nodes.is_empty());
        }
    }

    #[test]
    fn test_node2vec_biased_walks() {
        // With p=0.5 (low), walks should favor returning to previous nodes
        // With q=2.0 (high), walks should favor local (BFS-like) exploration
        let g = make_star_graph();
        let config = Node2VecConfig {
            dimensions: 8,
            walk_length: 20,
            num_walks: 5,
            p: 0.5,
            q: 2.0,
            ..Default::default()
        };

        let mut n2v = Node2Vec::new(config);
        let walks = n2v.generate_walks(&g);
        assert!(walks.is_ok());

        let walks = walks.expect("walks should be valid");
        assert!(!walks.is_empty());

        // Verify walks contain valid nodes
        for walk in &walks {
            for node in &walk.nodes {
                assert!(
                    (0..5).contains(node),
                    "Walk should only contain valid nodes, got {node}"
                );
            }
        }
    }

    #[test]
    fn test_node2vec_embedding_similarity() {
        let g = make_triangle();
        let config = Node2VecConfig {
            dimensions: 16,
            walk_length: 10,
            num_walks: 10,
            window_size: 3,
            p: 1.0,
            q: 1.0,
            epochs: 5,
            learning_rate: 0.05,
            negative_samples: 3,
        };

        let mut n2v = Node2Vec::new(config);
        let _ = n2v.train(&g);

        // In a triangle, all nodes are structurally equivalent
        // so similarities should be computable (not NaN)
        let model = n2v.model();
        let sim_01 = model.most_similar(&0, 2);
        assert!(sim_01.is_ok());

        let sim_01 = sim_01.expect("similarity should be valid");
        assert_eq!(sim_01.len(), 2, "Should find 2 most similar nodes");

        for (node, score) in &sim_01 {
            assert!(
                score.is_finite(),
                "Similarity for node {node} should be finite"
            );
        }
    }

    #[test]
    fn test_node2vec_digraph_train() {
        let g = make_directed_chain();
        let config = Node2VecConfig {
            dimensions: 8,
            walk_length: 4,
            num_walks: 3,
            window_size: 2,
            p: 1.0,
            q: 1.0,
            epochs: 2,
            learning_rate: 0.025,
            negative_samples: 2,
        };

        let mut n2v = Node2Vec::new(config);
        let result = n2v.train_digraph(&g);
        assert!(result.is_ok(), "DiGraph Node2Vec training should succeed");

        // All nodes should have embeddings
        for node in 0..5 {
            assert!(
                n2v.model().get_embedding(&node).is_some(),
                "Node {node} should have an embedding in directed graph"
            );
        }
    }

    #[test]
    fn test_node2vec_config() {
        let config = Node2VecConfig::default();
        assert_eq!(config.dimensions, 128);
        assert_eq!(config.walk_length, 80);
        assert_eq!(config.p, 1.0);
        assert_eq!(config.q, 1.0);

        let n2v: Node2Vec<i32> = Node2Vec::new(config);
        assert_eq!(n2v.config().dimensions, 128);
    }
}
