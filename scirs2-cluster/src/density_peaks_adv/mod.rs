//! Density Peaks Clustering (Rodriguez & Laio, Science 2014).
//!
//! Key insight: cluster centers are local density maxima that are
//! far from points with higher density.
//!
//! # References
//!
//! * Rodriguez, A., & Laio, A. (2014). Clustering by fast search and find of density peaks.
//!   *Science*, 344(6191), 1492-1496.

pub mod algorithm;
pub mod decision_graph;

pub use algorithm::{DensityPeaksAdv, DensityPeaksAdvResult, KernelType};
pub use decision_graph::DecisionGraph;
