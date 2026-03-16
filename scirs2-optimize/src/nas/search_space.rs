//! Neural architecture search space definition.
//!
//! Provides a DAG-based representation of neural architectures
//! and a configurable search space (DARTS-like by default).

use std::collections::VecDeque;

/// Operation types in a NAS cell
#[derive(Debug, Clone, PartialEq)]
pub enum OpType {
    /// Pass-through with no transformation
    Identity,
    /// 3x3 convolution
    Conv3x3,
    /// 5x5 convolution
    Conv5x5,
    /// Dilated 3x3 convolution (dilation=2)
    DilatedConv3x3,
    /// Depthwise separable 3x3 convolution
    DepthwiseSep3x3,
    /// 3x3 max pooling
    MaxPool3x3,
    /// 3x3 average pooling
    AvgPool3x3,
    /// Skip connection (residual)
    Skip,
    /// Zero connection (no gradient flow)
    Zero,
    /// Fully-connected layer with given output size
    Linear(usize),
    /// Gated Recurrent Unit cell
    GRU,
    /// Long Short-Term Memory cell
    LSTM,
}

impl OpType {
    /// Approximate number of trainable parameters for this op.
    pub fn num_params(&self, in_channels: usize) -> usize {
        match self {
            Self::Identity | Self::Skip | Self::Zero | Self::MaxPool3x3 | Self::AvgPool3x3 => 0,
            Self::Conv3x3 => 9 * in_channels * in_channels,
            Self::Conv5x5 => 25 * in_channels * in_channels,
            Self::DilatedConv3x3 => 9 * in_channels * in_channels,
            Self::DepthwiseSep3x3 => 9 * in_channels + in_channels * in_channels,
            Self::Linear(out) => in_channels * out,
            Self::GRU | Self::LSTM => 4 * in_channels * in_channels,
        }
    }

    /// Approximate FLOPs for this op given input channels and spatial size.
    pub fn flops(&self, in_channels: usize, spatial: usize) -> usize {
        let spatial_sq = spatial * spatial;
        match self {
            Self::Conv3x3 => 9 * 2 * in_channels * in_channels * spatial_sq,
            Self::Conv5x5 => 25 * 2 * in_channels * in_channels * spatial_sq,
            Self::DilatedConv3x3 => 9 * 2 * in_channels * in_channels * spatial_sq,
            Self::DepthwiseSep3x3 => (9 * in_channels + in_channels * in_channels) * spatial_sq,
            Self::Linear(out) => in_channels * out * 2,
            Self::GRU | Self::LSTM => 4 * in_channels * in_channels * 2,
            _ => in_channels * spatial_sq,
        }
    }
}

/// A node in the architecture DAG
#[derive(Debug, Clone)]
pub struct ArchNode {
    /// Unique identifier for the node
    pub id: usize,
    /// Human-readable node name
    pub name: String,
    /// Number of output channels produced by this node
    pub output_channels: usize,
}

/// A directed edge (operation) in the architecture DAG
#[derive(Debug, Clone)]
pub struct ArchEdge {
    /// Source node id
    pub from: usize,
    /// Destination node id
    pub to: usize,
    /// Operation applied along this edge
    pub op: OpType,
}

/// A complete architecture specification as a DAG
#[derive(Debug, Clone)]
pub struct Architecture {
    /// All nodes in the architecture
    pub nodes: Vec<ArchNode>,
    /// All directed edges (operations) in the architecture
    pub edges: Vec<ArchEdge>,
    /// Number of cells in the architecture
    pub n_cells: usize,
    /// Channel width
    pub channels: usize,
    /// Number of output classes
    pub n_classes: usize,
}

impl Architecture {
    /// Create a new empty architecture.
    pub fn new(n_cells: usize, channels: usize, n_classes: usize) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            n_cells,
            channels,
            n_classes,
        }
    }

    /// Sum of parameter counts across all edges.
    pub fn total_params(&self) -> usize {
        self.edges
            .iter()
            .map(|e| e.op.num_params(self.channels))
            .sum()
    }

    /// Sum of FLOPs across all edges for given spatial dimension.
    pub fn total_flops(&self, spatial: usize) -> usize {
        self.edges
            .iter()
            .map(|e| e.op.flops(self.channels, spatial))
            .sum()
    }

    /// Kahn's algorithm topological sort of DAG nodes.
    ///
    /// Returns a sorted list of node ids. If the graph contains a cycle,
    /// the returned list will be shorter than `nodes.len()`.
    pub fn topological_sort(&self) -> Vec<usize> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        for e in &self.edges {
            if e.from < n && e.to < n {
                adj[e.from].push(e.to);
                in_degree[e.to] = in_degree[e.to].saturating_add(1);
            }
        }

        let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);

        while let Some(v) = queue.pop_front() {
            order.push(v);
            for &u in &adj[v] {
                in_degree[u] -= 1;
                if in_degree[u] == 0 {
                    queue.push_back(u);
                }
            }
        }
        order
    }

    /// Returns `true` if the architecture DAG contains no cycles.
    pub fn is_valid(&self) -> bool {
        self.topological_sort().len() == self.nodes.len()
    }
}

/// Defines the search space: the set of valid architecture configurations.
pub struct SearchSpace {
    /// Available operation types
    pub operations: Vec<OpType>,
    /// Number of intermediate nodes per cell
    pub n_nodes_per_cell: usize,
    /// How many previous outputs each node can take as input
    pub n_input_nodes: usize,
    /// Possible channel widths
    pub channels: Vec<usize>,
    /// (min, max) number of cells (inclusive)
    pub n_cells_range: (usize, usize),
}

impl SearchSpace {
    /// Create a DARTS-like search space with the specified number of intermediate nodes.
    pub fn darts_like(n_nodes: usize) -> Self {
        Self {
            operations: vec![
                OpType::Skip,
                OpType::Zero,
                OpType::MaxPool3x3,
                OpType::AvgPool3x3,
                OpType::Conv3x3,
                OpType::Conv5x5,
                OpType::DilatedConv3x3,
                OpType::DepthwiseSep3x3,
            ],
            n_nodes_per_cell: n_nodes,
            n_input_nodes: 2,
            channels: vec![16, 32, 64, 128],
            n_cells_range: (2, 20),
        }
    }

    /// Upper bound on the number of distinct architectures in this search space.
    pub fn n_architectures(&self) -> u64 {
        let n_ops = self.operations.len() as u64;
        let n = self.n_nodes_per_cell;
        // edges: n_input_nodes inputs per node across all nodes
        let n_edges = (self.n_input_nodes * n) as u64;
        n_ops.saturating_pow(n_edges as u32)
    }

    /// Sample a random architecture from this search space.
    pub fn sample_random(
        &self,
        rng: &mut (impl scirs2_core::random::Rng + ?Sized),
    ) -> Architecture {
        use scirs2_core::random::{Rng, RngExt};

        let cells_lo = self.n_cells_range.0;
        let cells_hi = self.n_cells_range.1;
        let n_cells = if cells_lo >= cells_hi {
            cells_lo
        } else {
            rng.random_range(cells_lo..=cells_hi)
        };

        let ch_idx = rng.random_range(0..self.channels.len());
        let channels = self.channels[ch_idx];
        let n_classes = 10;

        let mut arch = Architecture::new(n_cells, channels, n_classes);

        // Add nodes: n_nodes_per_cell per cell
        for c in 0..n_cells {
            for j in 0..self.n_nodes_per_cell {
                arch.nodes.push(ArchNode {
                    id: c * self.n_nodes_per_cell + j,
                    name: format!("cell{}_node{}", c, j),
                    output_channels: channels,
                });
            }
        }

        // Add directed edges with randomly selected operations
        for c in 0..n_cells {
            for j in 0..self.n_nodes_per_cell {
                // Each node gets up to n_input_nodes incoming edges from earlier nodes
                let n_inputs = j.min(self.n_input_nodes);
                for k in 0..n_inputs.max(1) {
                    let from_offset = if n_inputs == 0 {
                        // first node in cell: connect to beginning (node 0)
                        0
                    } else {
                        c * self.n_nodes_per_cell + (j.saturating_sub(k + 1))
                    };
                    let to = c * self.n_nodes_per_cell + j;
                    if from_offset != to {
                        let op_idx = rng.random_range(0..self.operations.len());
                        arch.edges.push(ArchEdge {
                            from: from_offset,
                            to,
                            op: self.operations[op_idx].clone(),
                        });
                    }
                }
            }
        }

        arch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_search_space_sample_produces_arch() {
        let space = SearchSpace::darts_like(4);
        let mut rng = StdRng::seed_from_u64(42);
        let arch = space.sample_random(&mut rng);

        assert!(arch.n_cells > 0);
        assert!(arch.channels > 0);
        assert!(!arch.nodes.is_empty());
    }

    #[test]
    fn test_architecture_params_nonzero_for_conv() {
        let mut arch = Architecture::new(2, 32, 10);
        arch.nodes.push(ArchNode {
            id: 0,
            name: "node0".into(),
            output_channels: 32,
        });
        arch.nodes.push(ArchNode {
            id: 1,
            name: "node1".into(),
            output_channels: 32,
        });
        arch.edges.push(ArchEdge {
            from: 0,
            to: 1,
            op: OpType::Conv3x3,
        });

        // Conv3x3: 9 * 32 * 32 = 9216 params
        assert_eq!(arch.total_params(), 9 * 32 * 32);
        assert!(arch.total_flops(8) > 0);
    }

    #[test]
    fn test_topological_sort_linear_dag() {
        let mut arch = Architecture::new(1, 32, 10);
        for i in 0..3_usize {
            arch.nodes.push(ArchNode {
                id: i,
                name: format!("n{}", i),
                output_channels: 32,
            });
        }
        arch.edges.push(ArchEdge {
            from: 0,
            to: 1,
            op: OpType::Skip,
        });
        arch.edges.push(ArchEdge {
            from: 1,
            to: 2,
            op: OpType::Conv3x3,
        });

        assert!(arch.is_valid());
        let order = arch.topological_sort();
        assert_eq!(order.len(), 3);
        assert_eq!(order[0], 0);
    }

    #[test]
    fn test_n_architectures_positive() {
        let space = SearchSpace::darts_like(4);
        assert!(space.n_architectures() > 0);
    }

    #[test]
    fn test_op_type_zero_params_for_pooling() {
        assert_eq!(OpType::MaxPool3x3.num_params(64), 0);
        assert_eq!(OpType::AvgPool3x3.num_params(64), 0);
        assert_eq!(OpType::Skip.num_params(64), 0);
    }
}
