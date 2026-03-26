//! Shared types and configuration structs for graph transformers.

/// Configuration for a GPS (General, Powerful, Scalable) graph transformer.
#[derive(Debug, Clone)]
pub struct GraphTransformerConfig {
    /// Number of attention heads
    pub n_heads: usize,
    /// Hidden (embedding) dimensionality
    pub hidden_dim: usize,
    /// Number of transformer layers to stack
    pub n_layers: usize,
    /// Dropout rate (currently informational; applied as weight noise during init)
    pub dropout: f64,
    /// Positional encoding type to use
    pub pe_type: PeType,
    /// Number of positional encoding dimensions (LapPE/RWPE walk steps)
    pub pe_dim: usize,
}

impl Default for GraphTransformerConfig {
    fn default() -> Self {
        Self {
            n_heads: 4,
            hidden_dim: 64,
            n_layers: 2,
            dropout: 0.1,
            pe_type: PeType::LapPE,
            pe_dim: 8,
        }
    }
}

/// Configuration for the Graphormer model.
#[derive(Debug, Clone)]
pub struct GraphormerConfig {
    /// Maximum node degree for degree embedding table
    pub max_degree: usize,
    /// Maximum shortest-path distance for spatial encoding table (disconnected → this value)
    pub max_shortest_path: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Hidden (embedding) dimensionality
    pub hidden_dim: usize,
    /// Number of transformer layers
    pub n_layers: usize,
}

impl Default for GraphormerConfig {
    fn default() -> Self {
        Self {
            max_degree: 16,
            max_shortest_path: 20,
            n_heads: 4,
            hidden_dim: 64,
            n_layers: 2,
        }
    }
}

/// Output produced by a graph transformer forward pass.
#[derive(Debug, Clone)]
pub struct GraphTransformerOutput {
    /// Per-node embeddings, shape `[n_nodes][hidden_dim]`
    pub node_embeddings: Vec<Vec<f64>>,
    /// Graph-level embedding obtained by mean-pooling node embeddings, shape `[hidden_dim]`
    pub graph_embedding: Vec<f64>,
}

/// Type of positional encoding to augment node features.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeType {
    /// Laplacian eigenvector positional encoding (LapPE)
    LapPE,
    /// Random-walk landing probability positional encoding (RWPE)
    RWPE,
    /// SignNet-style sign-invariant Laplacian PE (placeholder, falls back to LapPE)
    SignNet,
}

/// A graph representation suitable for graph transformers.
///
/// Stores an *undirected* adjacency list and per-node feature vectors.
#[derive(Debug, Clone)]
pub struct GraphForTransformer {
    /// Undirected adjacency list: `adjacency[i]` = list of neighbor indices
    pub adjacency: Vec<Vec<usize>>,
    /// Per-node feature vectors (all must have the same length)
    pub node_features: Vec<Vec<f64>>,
    /// Number of nodes
    pub n_nodes: usize,
}

impl GraphForTransformer {
    /// Create a new graph from an adjacency list and feature matrix.
    ///
    /// # Errors
    /// Returns an error string if feature rows have inconsistent dimensionality or
    /// the adjacency list length does not match `node_features`.
    pub fn new(adjacency: Vec<Vec<usize>>, node_features: Vec<Vec<f64>>) -> Result<Self, String> {
        let n = node_features.len();
        if adjacency.len() != n {
            return Err(format!(
                "adjacency list length {} != node_features length {}",
                adjacency.len(),
                n
            ));
        }
        if n > 0 {
            let dim = node_features[0].len();
            for (i, row) in node_features.iter().enumerate() {
                if row.len() != dim {
                    return Err(format!(
                        "node_features row {} has {} dims, expected {}",
                        i,
                        row.len(),
                        dim
                    ));
                }
            }
        }
        Ok(Self {
            n_nodes: n,
            adjacency,
            node_features,
        })
    }
}
