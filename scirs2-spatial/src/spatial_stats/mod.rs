//! Spatial statistics module for analyzing spatial patterns and relationships
//!
//! This module provides statistical measures commonly used in spatial analysis,
//! including measures of spatial autocorrelation, clustering, pattern analysis,
//! kernel density estimation, and cluster detection.
//!
//! # Submodules
//!
//! - `core` - Basic spatial autocorrelation (Moran's I, Geary's C), weights matrices,
//!   Clark-Evans index, Ripley's K/L, average nearest neighbor
//! - `global_autocorrelation` - Enhanced global spatial autocorrelation with significance tests
//! - `lisa` - Local Indicators of Spatial Association with permutation tests and cluster maps
//! - `spatial_kde` - 2D Kernel Density Estimation on spatial data
//! - `scan_statistic` - Kulldorff's spatial scan statistic for cluster detection

pub mod core;
pub mod global_autocorrelation;
pub mod lisa;
pub mod scan_statistic;
pub mod spatial_kde;

// Re-export everything from core (backward compatibility)
pub use self::core::*;

// Re-export from global_autocorrelation
pub use global_autocorrelation::{
    distance_band_weights, geary_test, inverse_distance_weights, moran_test,
    row_standardize_weights, GlobalAutocorrelationResult, SpatialWeights,
};

// Re-export from lisa
pub use lisa::{
    getis_ord_gi_star, lisa_cluster_map, local_moran_permutation_test, GetisOrdResult, LisaCluster,
    LisaClusterMap, LisaResult,
};

// Re-export from spatial_kde
pub use spatial_kde::{
    kde_at_point, kde_on_grid, select_bandwidth, BandwidthMethod, KdeGrid, KernelType,
    SpatialKdeConfig,
};

// Re-export from scan_statistic
pub use scan_statistic::{kulldorff_scan, ScanCluster, ScanModel, ScanResult, ScanStatisticConfig};
