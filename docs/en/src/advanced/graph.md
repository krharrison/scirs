# Graph Computing (scirs2-graph)

`scirs2-graph` provides graph algorithms, graph neural networks (GNNs), and graph
transformers. It covers classical algorithms (shortest paths, centrality, community
detection) and modern GNN architectures (GCN, GAT, GraphSAGE, GraphGPS).

## Graph Construction

```rust,ignore
use scirs2_graph::{Graph, EdgeType};

// Undirected graph
let mut g = Graph::new(EdgeType::Undirected);
g.add_edge(0, 1, 1.0)?;
g.add_edge(1, 2, 2.0)?;
g.add_edge(0, 2, 3.0)?;

println!("Nodes: {}, Edges: {}", g.num_nodes(), g.num_edges());
```

## Classical Algorithms

### Shortest Paths

```rust,ignore
use scirs2_graph::algorithms::{dijkstra, bellman_ford, floyd_warshall};

// Single-source shortest paths
let distances = dijkstra(&graph, source_node)?;

// Negative weights allowed
let distances = bellman_ford(&graph, source_node)?;

// All-pairs shortest paths
let dist_matrix = floyd_warshall(&graph)?;
```

### Centrality

```rust,ignore
use scirs2_graph::algorithms::{pagerank, betweenness_centrality, degree_centrality};

// PageRank
let ranks = pagerank(&graph, 0.85, 100)?;

// Betweenness centrality
let bc = betweenness_centrality(&graph)?;
```

### Community Detection

```rust,ignore
use scirs2_graph::algorithms::{louvain, label_propagation};
use scirs2_graph::ssl::{DeepWalk, Node2Vec};

// Louvain community detection
let communities = louvain(&graph, None)?;

// Label propagation
let labels = label_propagation(&graph, max_iter)?;
```

## Graph Neural Networks

### GCN (Graph Convolutional Network)

```rust,ignore
use scirs2_graph::gnn::{GCNLayer, GCNModel};

// Single GCN layer
let gcn = GCNLayer::new(input_dim, output_dim)?;
let output = gcn.forward(&node_features, &adjacency)?;

// Multi-layer GCN model
let model = GCNModel::new(&[64, 32, num_classes], dropout)?;
let logits = model.forward(&features, &adjacency)?;
```

### GAT (Graph Attention Network)

```rust,ignore
use scirs2_graph::gnn::{GATLayer, GATModel};

let gat = GATLayer::new(input_dim, output_dim, num_heads)?;
let output = gat.forward(&node_features, &adjacency)?;
```

### GraphSAGE

```rust,ignore
use scirs2_graph::gnn::{GraphSAGELayer, Aggregator};

let sage = GraphSAGELayer::new(
    input_dim, output_dim, Aggregator::Mean
)?;
let output = sage.forward(&features, &neighbor_lists)?;
```

### R-GCN (Relational GCN)

For multi-relational graphs (knowledge graphs, heterogeneous networks):

```rust,ignore
use scirs2_graph::gnn::rgcn::RGCNLayer;

let rgcn = RGCNLayer::new(input_dim, output_dim, num_relations, num_bases)?;
let output = rgcn.forward(&features, &adjacency_per_relation)?;
```

## Graph Transformers

### GraphGPS

Combines message passing with global Transformer attention:

```rust,ignore
use scirs2_graph::gnn::graphgps::{GraphGPS, GraphGPSConfig};

let config = GraphGPSConfig {
    hidden_dim: 64,
    num_heads: 8,
    num_layers: 4,
    local_model: "gcn",
    global_model: "transformer",
};
let model = GraphGPS::new(config)?;
let output = model.forward(&features, &edge_index, &batch)?;
```

### Graphormer

With spatial and degree encoding:

```rust,ignore
use scirs2_graph::gnn::graphormer::{Graphormer, GraphormerConfig};

let model = Graphormer::new(GraphormerConfig {
    hidden_dim: 64,
    num_heads: 8,
    num_layers: 6,
    max_degree: 512,
})?;
let output = model.forward(&features, &spd_matrix, &degree)?;
```

## Graph Partitioning

```rust,ignore
use scirs2_graph::partitioning::{metis_partition, fennel_streaming};

// METIS-style multilevel partitioning
let partition = metis_partition(&graph, num_parts)?;

// FENNEL streaming partitioner (for very large graphs)
let partition = fennel_streaming(&edge_stream, num_parts, num_nodes)?;
```

## Signed and Directed Graphs

```rust,ignore
use scirs2_graph::gnn::signed::{SPONGE, SGCN};

// SPONGE spectral embedding for signed graphs
let embedding = SPONGE::new(embed_dim)?.embed(&signed_adjacency)?;

// SGCN: Signed Graph Convolutional Network
let sgcn = SGCN::new(input_dim, hidden_dim, output_dim)?;
let output = sgcn.forward(&features, &pos_edges, &neg_edges)?;
```
