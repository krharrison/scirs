//! Graph Transformer architectures and Temporal Graph Neural Networks
//!
//! This module implements state-of-the-art graph transformer architectures:
//!
//! - [`graphormer`] - Graphormer-style positional encodings with spatial/centrality/edge bias
//! - [`gps`] - GPS (General Powerful Scalable) hybrid MPNN + global attention
//! - [`temporal_gnn`] - Temporal Graph Neural Networks with TGN-style memory modules

pub mod gps;
pub mod graphormer;
pub mod temporal_gnn;

pub use gps::{GpsConfig, GpsLayer, GpsModel, LaplacianPe, LocalModel, RandomWalkPe};
pub use graphormer::{
    CentralityEncoding, EdgeEncoding, GraphormerConfig, GraphormerLayer, GraphormerModel,
    SpatialEncoding,
};
pub use temporal_gnn::{
    MemoryModule, MemoryUpdateMethod, TemporalAttention, TemporalEvent, TemporalGnnConfig,
    TemporalGnnModel, TimeEncoding, TimeEncodingType,
};
