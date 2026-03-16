//! DeepWalk graph embedding algorithm
//!
//! Implements the DeepWalk algorithm from Perozzi et al. (2014) for learning
//! latent representations of nodes using short uniform random walks.
//! Supports both negative sampling and hierarchical softmax approximation.
//!
//! # References
//! - Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). DeepWalk: Online Learning
//!   of Social Representations. KDD 2014.

use super::core::{Embedding, EmbeddingModel};
use super::negative_sampling::NegativeSampler;
use super::random_walk::RandomWalkGenerator;
use super::types::{DeepWalkConfig, RandomWalk};
use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::{Rng, RngExt};
use std::collections::HashMap;

/// Training mode for DeepWalk
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepWalkMode {
    /// Standard negative sampling (default, simpler and often faster)
    NegativeSampling,
    /// Hierarchical softmax using a Huffman tree built from node frequencies
    HierarchicalSoftmax,
}

/// A node in the Huffman tree used for hierarchical softmax
#[derive(Debug, Clone)]
struct HuffmanNode {
    /// Path from root (true = right, false = left)
    code: Vec<bool>,
    /// Indices of internal nodes along the path
    point: Vec<usize>,
}

/// Huffman tree for hierarchical softmax
#[derive(Debug)]
struct HuffmanTree {
    /// Huffman encoding for each leaf node (indexed by node index)
    codes: Vec<HuffmanNode>,
    /// Number of internal nodes
    num_internal: usize,
}

impl HuffmanTree {
    /// Build a Huffman tree from node frequencies
    fn build(frequencies: &[f64]) -> Result<Self> {
        let n = frequencies.len();
        if n == 0 {
            return Err(GraphError::InvalidGraph(
                "Cannot build Huffman tree from empty frequency list".to_string(),
            ));
        }

        if n == 1 {
            // Single node: trivial encoding
            let codes = vec![HuffmanNode {
                code: vec![false],
                point: vec![0],
            }];
            return Ok(HuffmanTree {
                codes,
                num_internal: 1,
            });
        }

        // Build Huffman tree using a priority queue simulation
        // Total nodes = n leaves + (n-1) internal nodes
        let total = 2 * n - 1;
        let mut count = vec![0.0f64; total];
        let mut parent = vec![0usize; total];
        let mut binary = vec![false; total]; // true = right child

        // Initialize leaf frequencies
        for (i, &freq) in frequencies.iter().enumerate() {
            count[i] = freq.max(1e-10); // Avoid zero frequencies
        }

        // Initialize internal node counts to large value
        for i in n..total {
            count[i] = f64::MAX;
        }

        // Build tree bottom-up
        let mut pos1 = n - 1; // Position scanning leaves (right to left, sorted by freq)
        let mut pos2 = n; // Position scanning internal nodes

        // Sort leaf indices by frequency (ascending)
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| {
            count[a]
                .partial_cmp(&count[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Reorder counts by sorted order
        let mut sorted_counts = vec![0.0; n];
        let mut reverse_map = vec![0usize; n]; // original_index -> sorted_position
        for (sorted_pos, &orig_idx) in sorted_indices.iter().enumerate() {
            sorted_counts[sorted_pos] = count[orig_idx];
            reverse_map[orig_idx] = sorted_pos;
        }
        count[..n].copy_from_slice(&sorted_counts[..n]);

        // Build internal nodes
        for internal_idx in n..total {
            // Find two nodes with smallest counts
            let min1;
            let min2;

            // First minimum
            if pos1 < n && (pos2 >= internal_idx || count[pos1] < count[pos2]) {
                min1 = pos1;
                pos1 = pos1.wrapping_sub(1); // will wrap to usize::MAX when 0
                if pos1 == usize::MAX {
                    pos1 = n; // sentinel: no more leaves
                }
            } else {
                min1 = pos2;
                pos2 += 1;
            }

            // Second minimum
            if pos1 < n && (pos2 >= internal_idx || count[pos1] < count[pos2]) {
                min2 = pos1;
                pos1 = pos1.wrapping_sub(1);
                if pos1 == usize::MAX {
                    pos1 = n;
                }
            } else if pos2 < internal_idx {
                min2 = pos2;
                pos2 += 1;
            } else {
                min2 = min1; // Fallback (shouldn't happen with valid input)
            }

            count[internal_idx] = count[min1] + count[min2];
            parent[min1] = internal_idx;
            parent[min2] = internal_idx;
            binary[min2] = true; // Right child
        }

        // Generate codes by traversing from each leaf to root
        let mut codes = vec![
            HuffmanNode {
                code: Vec::new(),
                point: Vec::new(),
            };
            n
        ];

        for sorted_pos in 0..n {
            let mut code = Vec::new();
            let mut point = Vec::new();

            let mut current = sorted_pos;
            while current < total - 1 {
                // Not root
                code.push(binary[current]);
                let par = parent[current];
                // Internal node index = par - n (0-indexed)
                if par >= n {
                    point.push(par - n);
                }
                current = par;
            }

            // Reverse to get root-to-leaf order
            code.reverse();
            point.reverse();

            // Map back from sorted position to original index
            let orig_idx = sorted_indices[sorted_pos];
            codes[orig_idx] = HuffmanNode { code, point };
        }

        Ok(HuffmanTree {
            codes,
            num_internal: n - 1,
        })
    }
}

/// DeepWalk embedding algorithm
///
/// Learns node embeddings using uniform random walks followed by
/// skip-gram optimization. Supports both negative sampling and
/// hierarchical softmax.
pub struct DeepWalk<N: Node> {
    config: DeepWalkConfig,
    model: EmbeddingModel<N>,
    walk_generator: RandomWalkGenerator<N>,
    /// Training mode
    mode: DeepWalkMode,
    /// Internal node vectors for hierarchical softmax
    internal_vectors: Vec<Vec<f64>>,
}

impl<N: Node> DeepWalk<N> {
    /// Create a new DeepWalk instance with negative sampling (default)
    pub fn new(config: DeepWalkConfig) -> Self {
        DeepWalk {
            model: EmbeddingModel::new(config.dimensions),
            config,
            walk_generator: RandomWalkGenerator::new(),
            mode: DeepWalkMode::NegativeSampling,
            internal_vectors: Vec::new(),
        }
    }

    /// Create a new DeepWalk instance with hierarchical softmax
    pub fn with_hierarchical_softmax(config: DeepWalkConfig) -> Self {
        DeepWalk {
            model: EmbeddingModel::new(config.dimensions),
            config,
            walk_generator: RandomWalkGenerator::new(),
            mode: DeepWalkMode::HierarchicalSoftmax,
            internal_vectors: Vec::new(),
        }
    }

    /// Set the training mode
    pub fn set_mode(&mut self, mode: DeepWalkMode) {
        self.mode = mode;
    }

    /// Get the current training mode
    pub fn mode(&self) -> DeepWalkMode {
        self.mode
    }

    /// Generate training data (uniform random walks) on undirected graph
    pub fn generate_walks<E, Ix>(&mut self, graph: &Graph<N, E, Ix>) -> Result<Vec<RandomWalk<N>>>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let mut all_walks = Vec::new();

        for node in graph.nodes() {
            for _ in 0..self.config.num_walks {
                let walk =
                    self.walk_generator
                        .simple_random_walk(graph, node, self.config.walk_length)?;
                all_walks.push(walk);
            }
        }

        Ok(all_walks)
    }

    /// Generate training data (uniform random walks) on directed graph
    pub fn generate_walks_digraph<E, Ix>(
        &mut self,
        graph: &DiGraph<N, E, Ix>,
    ) -> Result<Vec<RandomWalk<N>>>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let mut all_walks = Vec::new();

        for node in graph.nodes() {
            for _ in 0..self.config.num_walks {
                let walk = self.walk_generator.simple_random_walk_digraph(
                    graph,
                    node,
                    self.config.walk_length,
                )?;
                all_walks.push(walk);
            }
        }

        Ok(all_walks)
    }

    /// Train the DeepWalk model on an undirected graph
    pub fn train<E, Ix>(&mut self, graph: &Graph<N, E, Ix>) -> Result<()>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        // Initialize random embeddings
        let mut rng = scirs2_core::random::rng();
        self.model.initialize_random(graph, &mut rng);

        match self.mode {
            DeepWalkMode::NegativeSampling => {
                self.train_negative_sampling(graph, &mut rng)?;
            }
            DeepWalkMode::HierarchicalSoftmax => {
                self.train_hierarchical_softmax(graph, &mut rng)?;
            }
        }

        Ok(())
    }

    /// Train using negative sampling
    fn train_negative_sampling<E, Ix>(
        &mut self,
        graph: &Graph<N, E, Ix>,
        rng: &mut impl Rng,
    ) -> Result<()>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let negative_sampler = NegativeSampler::new(graph);

        for epoch in 0..self.config.epochs {
            let walks = self.generate_walks(graph)?;
            let context_pairs =
                EmbeddingModel::generate_context_pairs(&walks, self.config.window_size);

            let mut shuffled_pairs = context_pairs;
            shuffled_pairs.shuffle(rng);

            let current_lr = self.config.learning_rate
                * (1.0 - epoch as f64 / self.config.epochs as f64).max(0.0001);

            self.model.train_skip_gram(
                &shuffled_pairs,
                &negative_sampler,
                current_lr,
                self.config.negative_samples,
                rng,
            )?;
        }

        Ok(())
    }

    /// Train using hierarchical softmax approximation
    fn train_hierarchical_softmax<E, Ix>(
        &mut self,
        graph: &Graph<N, E, Ix>,
        rng: &mut impl Rng,
    ) -> Result<()>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
        let n = nodes.len();

        if n == 0 {
            return Err(GraphError::InvalidGraph(
                "Cannot train on empty graph".to_string(),
            ));
        }

        // Build node-to-index mapping
        let node_to_idx: HashMap<N, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();

        // Compute node frequencies (degree-based)
        let frequencies: Vec<f64> = nodes.iter().map(|n| graph.degree(n) as f64 + 1.0).collect();

        // Build Huffman tree
        let huffman = HuffmanTree::build(&frequencies)?;

        // Initialize internal node vectors (one per internal node)
        let dim = self.config.dimensions;
        self.internal_vectors = (0..huffman.num_internal).map(|_| vec![0.0; dim]).collect();

        // Training loop
        for epoch in 0..self.config.epochs {
            let walks = self.generate_walks(graph)?;

            let current_lr = self.config.learning_rate
                * (1.0 - epoch as f64 / self.config.epochs as f64).max(0.0001);

            // Process each walk
            for walk in &walks {
                let walk_indices: Vec<usize> = walk
                    .nodes
                    .iter()
                    .filter_map(|n| node_to_idx.get(n).copied())
                    .collect();

                // Generate (target, context) pairs from walk
                for (i, &target_idx) in walk_indices.iter().enumerate() {
                    let start = i.saturating_sub(self.config.window_size);
                    let end = (i + self.config.window_size + 1).min(walk_indices.len());

                    for j in start..end {
                        if i == j {
                            continue;
                        }

                        let context_idx = walk_indices[j];
                        self.hierarchical_softmax_update(
                            &nodes[target_idx],
                            context_idx,
                            &huffman,
                            current_lr,
                        );
                    }
                }
            }

            // Shuffle walks for next epoch
            let _ = rng; // Use rng to avoid unused warning
        }

        Ok(())
    }

    /// Update embeddings using hierarchical softmax for one (target, context) pair
    fn hierarchical_softmax_update(
        &mut self,
        target_node: &N,
        context_idx: usize,
        huffman: &HuffmanTree,
        learning_rate: f64,
    ) where
        N: Clone,
    {
        let dim = self.config.dimensions;

        if context_idx >= huffman.codes.len() {
            return;
        }

        let huffman_node = &huffman.codes[context_idx];

        // Get target embedding
        let target_emb = match self.model.embeddings.get(target_node) {
            Some(e) => e.vector.clone(),
            None => return,
        };

        let mut grad = vec![0.0; dim];

        // Walk along the Huffman tree path
        for (step, (&is_right, &internal_idx)) in huffman_node
            .code
            .iter()
            .zip(huffman_node.point.iter())
            .enumerate()
        {
            if internal_idx >= self.internal_vectors.len() {
                continue;
            }

            // Compute dot product: target . internal_node
            let dot: f64 = target_emb
                .iter()
                .zip(self.internal_vectors[internal_idx].iter())
                .map(|(a, b)| a * b)
                .sum();

            let sig = 1.0 / (1.0 + (-dot).exp());

            // Label: 1 for left child (code=false), 0 for right child (code=true)
            let label = if is_right { 0.0 } else { 1.0 };
            let g = learning_rate * (label - sig);

            // Accumulate gradient for target embedding
            for d in 0..dim {
                grad[d] += g * self.internal_vectors[internal_idx][d];
            }

            // Update internal node vector
            for d in 0..dim {
                self.internal_vectors[internal_idx][d] += g * target_emb[d];
            }

            let _ = step; // Consume step variable
        }

        // Apply accumulated gradient to target embedding
        if let Some(emb) = self.model.embeddings.get_mut(target_node) {
            for d in 0..dim {
                emb.vector[d] += grad[d];
            }
        }
    }

    /// Train the DeepWalk model on a directed graph
    pub fn train_digraph<E, Ix>(&mut self, graph: &DiGraph<N, E, Ix>) -> Result<()>
    where
        N: Clone + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        let mut rng = scirs2_core::random::rng();
        self.model.initialize_random_digraph(graph, &mut rng);

        // For directed graphs, we only support negative sampling for now
        // Build a manual negative sampler from DiGraph
        let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
        let degrees: Vec<f64> = nodes.iter().map(|n| graph.degree(n) as f64 + 1.0).collect();
        let total: f64 = degrees.iter().sum();
        let powered: Vec<f64> = degrees.iter().map(|d| (d / total).powf(0.75)).collect();
        let total_powered: f64 = powered.iter().sum();
        let probs: Vec<f64> = powered.iter().map(|p| p / total_powered).collect();

        let mut cumulative = vec![0.0; probs.len()];
        if !cumulative.is_empty() {
            cumulative[0] = probs[0];
            for i in 1..probs.len() {
                cumulative[i] = cumulative[i - 1] + probs[i];
            }
        }

        for epoch in 0..self.config.epochs {
            let walks = self.generate_walks_digraph(graph)?;
            let context_pairs =
                EmbeddingModel::generate_context_pairs(&walks, self.config.window_size);

            let mut shuffled_pairs = context_pairs;
            shuffled_pairs.shuffle(&mut rng);

            let current_lr = self.config.learning_rate
                * (1.0 - epoch as f64 / self.config.epochs as f64).max(0.0001);

            let dim = self.config.dimensions;
            let num_neg = self.config.negative_samples;

            // Manual skip-gram with negative sampling
            for pair in &shuffled_pairs {
                let target_emb = match self.model.embeddings.get(&pair.target) {
                    Some(e) => e.clone(),
                    None => continue,
                };
                let context_emb = match self.model.context_embeddings.get(&pair.context) {
                    Some(e) => e.clone(),
                    None => continue,
                };

                let dot: f64 = target_emb
                    .vector
                    .iter()
                    .zip(context_emb.vector.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let sig = 1.0 / (1.0 + (-dot).exp());
                let g = current_lr * (1.0 - sig);

                let mut target_grad = vec![0.0; dim];
                for d in 0..dim {
                    target_grad[d] = g * context_emb.vector[d];
                }

                if let Some(ctx) = self.model.context_embeddings.get_mut(&pair.context) {
                    for d in 0..dim {
                        ctx.vector[d] += g * target_emb.vector[d];
                    }
                }

                // Negative samples
                for _ in 0..num_neg {
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
                        let neg_g = current_lr * (-neg_sig);

                        for d in 0..dim {
                            target_grad[d] += neg_g * neg_emb.vector[d];
                        }

                        if let Some(neg_ctx) = self.model.context_embeddings.get_mut(neg_node) {
                            for d in 0..dim {
                                neg_ctx.vector[d] += neg_g * target_emb.vector[d];
                            }
                        }
                    }
                }

                if let Some(target) = self.model.embeddings.get_mut(&pair.target) {
                    for d in 0..dim {
                        target.vector[d] += target_grad[d];
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the trained model
    pub fn model(&self) -> &EmbeddingModel<N> {
        &self.model
    }

    /// Get mutable reference to the model
    pub fn model_mut(&mut self) -> &mut EmbeddingModel<N> {
        &mut self.model
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

    fn make_path_graph() -> Graph<i32, f64> {
        let mut g = Graph::new();
        for i in 0..5 {
            g.add_node(i);
        }
        let _ = g.add_edge(0, 1, 1.0);
        let _ = g.add_edge(1, 2, 1.0);
        let _ = g.add_edge(2, 3, 1.0);
        let _ = g.add_edge(3, 4, 1.0);
        g
    }

    fn make_directed_cycle() -> DiGraph<i32, f64> {
        let mut g = DiGraph::new();
        for i in 0..4 {
            g.add_node(i);
        }
        let _ = g.add_edge(0, 1, 1.0);
        let _ = g.add_edge(1, 2, 1.0);
        let _ = g.add_edge(2, 3, 1.0);
        let _ = g.add_edge(3, 0, 1.0);
        g
    }

    #[test]
    fn test_deepwalk_negative_sampling() {
        let g = make_triangle();
        let config = DeepWalkConfig {
            dimensions: 8,
            walk_length: 5,
            num_walks: 3,
            window_size: 2,
            epochs: 2,
            learning_rate: 0.025,
            negative_samples: 2,
        };

        let mut dw = DeepWalk::new(config);
        assert_eq!(dw.mode(), DeepWalkMode::NegativeSampling);

        let result = dw.train(&g);
        assert!(
            result.is_ok(),
            "DeepWalk negative sampling training should succeed"
        );

        for node in [0, 1, 2] {
            assert!(
                dw.model().get_embedding(&node).is_some(),
                "Node {node} should have embedding"
            );
        }
    }

    #[test]
    fn test_deepwalk_hierarchical_softmax() {
        let g = make_triangle();
        let config = DeepWalkConfig {
            dimensions: 8,
            walk_length: 5,
            num_walks: 3,
            window_size: 2,
            epochs: 2,
            learning_rate: 0.025,
            negative_samples: 2,
        };

        let mut dw = DeepWalk::with_hierarchical_softmax(config);
        assert_eq!(dw.mode(), DeepWalkMode::HierarchicalSoftmax);

        let result = dw.train(&g);
        assert!(
            result.is_ok(),
            "DeepWalk hierarchical softmax training should succeed"
        );

        for node in [0, 1, 2] {
            assert!(
                dw.model().get_embedding(&node).is_some(),
                "Node {node} should have embedding"
            );
        }
    }

    #[test]
    fn test_deepwalk_walk_generation() {
        let g = make_path_graph();
        let config = DeepWalkConfig {
            dimensions: 8,
            walk_length: 4,
            num_walks: 2,
            ..Default::default()
        };

        let mut dw = DeepWalk::new(config);
        let walks = dw.generate_walks(&g);
        assert!(walks.is_ok());

        let walks = walks.expect("walks should be valid");
        // 5 nodes * 2 walks = 10 walks
        assert_eq!(walks.len(), 10);

        for walk in &walks {
            assert!(!walk.nodes.is_empty());
            assert!(walk.nodes.len() <= 4);
            // All nodes should be valid
            for node in &walk.nodes {
                assert!((0..5).contains(node));
            }
        }
    }

    #[test]
    fn test_deepwalk_digraph() {
        let g = make_directed_cycle();
        let config = DeepWalkConfig {
            dimensions: 8,
            walk_length: 6,
            num_walks: 3,
            window_size: 2,
            epochs: 2,
            learning_rate: 0.025,
            negative_samples: 2,
        };

        let mut dw = DeepWalk::new(config);
        let result = dw.train_digraph(&g);
        assert!(result.is_ok(), "DiGraph DeepWalk training should succeed");

        for node in 0..4 {
            assert!(
                dw.model().get_embedding(&node).is_some(),
                "Node {node} should have embedding in directed graph"
            );
        }
    }

    #[test]
    fn test_deepwalk_mode_switching() {
        let g = make_triangle();
        let config = DeepWalkConfig {
            dimensions: 8,
            walk_length: 5,
            num_walks: 2,
            epochs: 1,
            ..Default::default()
        };

        let mut dw = DeepWalk::new(config);
        assert_eq!(dw.mode(), DeepWalkMode::NegativeSampling);

        dw.set_mode(DeepWalkMode::HierarchicalSoftmax);
        assert_eq!(dw.mode(), DeepWalkMode::HierarchicalSoftmax);

        let result = dw.train(&g);
        assert!(result.is_ok());
    }

    #[test]
    fn test_deepwalk_embedding_dimensions() {
        let g = make_triangle();
        let config = DeepWalkConfig {
            dimensions: 32,
            walk_length: 5,
            num_walks: 2,
            epochs: 1,
            ..Default::default()
        };

        let mut dw = DeepWalk::new(config);
        let _ = dw.train(&g);

        for node in [0, 1, 2] {
            let emb = dw.model().get_embedding(&node);
            assert!(emb.is_some());
            assert_eq!(emb.map(|e| e.dimensions()).unwrap_or(0), 32);
        }
    }

    #[test]
    fn test_huffman_tree_basic() {
        let freqs = vec![5.0, 2.0, 1.0, 3.0];
        let tree = HuffmanTree::build(&freqs);
        assert!(tree.is_ok());

        let tree = tree.expect("tree should be valid");
        assert_eq!(tree.codes.len(), 4);
        assert_eq!(tree.num_internal, 3);

        // Each code should be non-empty
        for (i, code) in tree.codes.iter().enumerate() {
            assert!(
                !code.code.is_empty(),
                "Node {i} should have non-empty Huffman code"
            );
            assert!(
                !code.point.is_empty(),
                "Node {i} should have non-empty path"
            );
        }
    }

    #[test]
    fn test_huffman_tree_single_node() {
        let freqs = vec![1.0];
        let tree = HuffmanTree::build(&freqs);
        assert!(tree.is_ok());

        let tree = tree.expect("tree should be valid");
        assert_eq!(tree.codes.len(), 1);
    }
}
