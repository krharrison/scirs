//! Enhanced streaming (online) clustering algorithms.
//!
//! This module provides a richer suite of streaming clustering methods that
//! complement the basic implementations in `streaming_cluster`:
//!
//! - [`clustream`] – CluStream (Aggarwal et al. 2003): micro/macro-cluster
//!   based stream clustering with pyramid time windows.
//! - [`denstream`] – DenStream (Cao et al. 2006): density-based stream
//!   clustering with potential / outlier micro-clusters and exponential fading.
//! - [`birkm`] – BIRCH (Zhang et al. 1996) and mini-batch K-means with
//!   streaming updates and chunk-based processing.

pub mod birkm;
pub mod clustream;
pub mod denstream;

// Re-export key types for convenience
pub use birkm::{
    BirchConfig, BirchResult, BIRCH, CF, CFNode, CFTree, ChunkClustering, ChunkClusteringConfig,
    ChunkClusteringResult, StreamingKMeans as BirchStreamingKMeans,
    StreamingKMeansConfig as BirchStreamingKMeansConfig,
    StreamingKMeansResult as BirchStreamingKMeansResult,
};
pub use clustream::{
    CluStream, CluStreamConfig, CluStreamResult, MacroKMeans, MicroCluster as CluStreamMicroCluster,
    PyramidTimeWindow, Snapshot,
};
pub use denstream::{
    CoreMicroCluster, DenStream, DenStreamConfig, DenStreamResult, Fading,
    OutlierMicroCluster,
};
