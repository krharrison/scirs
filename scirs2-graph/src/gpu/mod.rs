//! GPU-accelerated and parallel graph algorithms.
//!
//! This module provides parallel graph traversal and shortest-path algorithms
//! designed with a GPU-ready interface. Current implementations use CPU-parallel
//! execution via Rayon-compatible patterns and are ready to be backed by actual
//! GPU kernels in future releases.
//!
//! # Algorithms
//!
//! - [`algorithms::gpu_bfs`] — Parallel BFS (level-synchronous, frontier-based)
//! - [`algorithms::gpu_sssp_bellman_ford`] — Bellman-Ford SSSP (GPU-friendly, detects
//!   negative cycles)
//! - [`algorithms::gpu_sssp_delta_stepping`] — Delta-stepping SSSP (highly parallel,
//!   better cache behavior than Dijkstra)
//!
//! # Graph Format
//!
//! Algorithms in this module accept graphs in **Compressed Sparse Row (CSR)** format
//! for BFS/Bellman-Ford, or adjacency-list format for delta-stepping. CSR is the
//! preferred format for GPU workloads due to coalesced memory access.
//!
//! ```rust,no_run
//! use scirs2_graph::gpu::algorithms::{gpu_bfs, GpuBfsConfig};
//!
//! // 3-node path: 0 -> 1 -> 2
//! let row_ptr = vec![0, 1, 2, 2];
//! let col_idx = vec![1, 2];
//! let config = GpuBfsConfig::default();
//! let dist = gpu_bfs(&row_ptr, &col_idx, 0, &config).expect("bfs failed");
//! assert_eq!(dist[0], 0);
//! assert_eq!(dist[1], 1);
//! assert_eq!(dist[2], 2);
//! ```

pub mod algorithms;

pub use algorithms::{
    gpu_bfs, gpu_sssp_bellman_ford, gpu_sssp_delta_stepping, GpuBfsConfig, GpuGraphBackend,
};
