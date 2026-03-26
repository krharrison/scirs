//! Distributional Distance Metrics
//!
//! This module provides distance metrics between probability distributions,
//! including optimal transport distances, time-warping distances, and
//! kernel-based distances.
//!
//! # Modules
//!
//! - [`wasserstein`]: Wasserstein (Earth Mover's) distance
//! - [`sinkhorn`]: Sinkhorn divergence and entropic optimal transport
//! - [`dtw`]: Dynamic Time Warping distance
//! - [`energy`]: Energy distance and Maximum Mean Discrepancy
//!
//! # Quick Example
//!
//! ```
//! use scirs2_metrics::distributional::wasserstein::wasserstein_1d;
//! use scirs2_metrics::distributional::dtw::{dtw_distance, DtwConfig};
//! use scirs2_metrics::distributional::energy::energy_distance;
//!
//! // Wasserstein distance between two 1D distributions
//! let a = vec![1.0, 2.0, 3.0];
//! let b = vec![2.0, 3.0, 4.0];
//! let w1 = wasserstein_1d(&a, &b, 1).expect("should succeed");
//!
//! // DTW distance between two time series
//! let ts1 = vec![1.0, 2.0, 3.0, 2.0, 1.0];
//! let ts2 = vec![1.0, 2.0, 3.0, 2.0, 1.0];
//! let dtw = dtw_distance(&ts1, &ts2, &DtwConfig::default()).expect("should succeed");
//!
//! // Energy distance between two sample sets
//! let x = vec![1.0, 2.0, 3.0];
//! let y = vec![1.5, 2.5, 3.5];
//! let ed = energy_distance(&x, &y).expect("should succeed");
//! ```

pub mod dtw;
pub mod energy;
pub mod sinkhorn;
pub mod wasserstein;

// Re-export key types for convenience
pub use dtw::{DistanceFunction, DtwConfig, DtwConstraint, DtwResult};
pub use energy::{energy_distance, energy_distance_nd, mmd_gaussian, mmd_gaussian_nd};
pub use sinkhorn::{sinkhorn_distance, sinkhorn_divergence, SinkhornConfig, SinkhornResult};
pub use wasserstein::{
    earth_movers_distance, wasserstein_1d, wasserstein_from_samples, wasserstein_nd,
};
