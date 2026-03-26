//! Streaming graph processing — core module and advanced algorithms.
//!
//! This module re-exports all types from the original streaming core and adds:
//! - [`GraphStream`]: iterator-backed edge stream abstraction
//! - [`StreamingTriangleCounter`]: reservoir+window-based exact/approximate triangle counting
//! - [`StreamingUnionFind`]: online connected-components via Union-Find
//! - [`streaming_bfs`]: multi-pass memory-efficient BFS over an edge stream
//! - [`StreamingDegreeEstimator`]: count-min sketch degree estimation
//!
//! The original sliding-window and Doulion/MASCOT triangle counters, streaming
//! graph, and degree-distribution types are all still available via [`core`].

// Re-export the original streaming core unchanged
pub mod core;
pub use core::*;

// New streaming algorithm extensions
pub mod algorithms;
pub use algorithms::{
    streaming_bfs, GraphStream, StreamBfsResult, StreamConfig, StreamingBfsConfig,
    StreamingDegreeEstimator, StreamingTriangleCounter, StreamingUnionFind,
};
