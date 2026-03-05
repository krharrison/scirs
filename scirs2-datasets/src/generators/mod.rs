//! Dataset generators
//!
//! This module provides comprehensive dataset generation functionality for machine learning
//! and data science applications. It includes:
//!
//! - **Basic generators**: Classification, regression, clustering, and time series datasets
//! - **Manifold generators**: Swiss roll, S-curve, torus, and other manifold datasets
//! - **Noise injection**: Missing data and outlier injection utilities
//! - **GPU acceleration**: GPU-accelerated versions of basic generators
//! - **Configuration**: Types and utilities for generator configuration
//! - **Time series generators**: Sine wave, random walk, AR process, seasonal signals
//! - **Graph generators**: Karate club, Erdos-Renyi, Barabasi-Albert, Watts-Strogatz
//! - **Sparse matrix generators**: SPD, banded, Laplacian matrices

pub mod basic;
/// Advanced classification generators (multi-label, Hastie, enhanced n-class)
pub mod classification;
pub mod config;
pub mod gpu;
/// Graph dataset generators (karate club, random graph, Barabasi-Albert, Watts-Strogatz)
pub mod graph;
pub mod manifold;
pub mod noise;
/// Advanced regression generators (Friedman benchmarks, sparse uncorrelated, low-rank)
pub mod regression;
/// Sparse matrix dataset generators (SPD, banded, Laplacian)
pub mod sparse;
/// Structured data generators (biclusters, checkerboard, SPD matrices, sparse coded signals)
pub mod structured;
/// Time series dataset generators (sine wave, random walk, AR process, seasonal)
pub mod time_series;

// Re-export all public functions and types for backward compatibility
pub use basic::*;
pub use config::*;
pub use gpu::*;
pub use graph::*;
pub use manifold::*;
pub use noise::*;
pub use sparse::*;
pub use time_series::*;
