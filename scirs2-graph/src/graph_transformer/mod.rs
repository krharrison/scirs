//! Graph Transformer models: GraphGPS and Graphormer.
//!
//! Provides two state-of-the-art graph transformer architectures:
//!
//! - **GraphGPS** (Rampasek et al. 2022): combines a local MPNN branch with a
//!   global self-attention Transformer, augmented with positional encodings.
//! - **Graphormer** (Ying et al. 2021): uses structural encodings (degree
//!   embedding, spatial encoding via SPD, virtual graph token) injected directly
//!   into the standard Transformer architecture.
//!
//! Positional encodings are provided in the [`positional_encoding`] sub-module:
//! Laplacian PE, Random-Walk PE, and all-pairs shortest paths (BFS).

pub mod graphgps;
pub mod graphormer;
pub mod positional_encoding;
pub mod types;

// Convenience re-exports
pub use graphgps::GpsModel;
pub use graphormer::GraphormerModel;
pub use positional_encoding::{all_pairs_shortest_path, laplacian_pe, rwpe};
pub use types::{
    GraphForTransformer, GraphTransformerConfig, GraphTransformerOutput, GraphormerConfig, PeType,
};
