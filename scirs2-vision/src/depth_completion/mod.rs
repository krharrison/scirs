//! Depth completion algorithms.
//!
//! Fill sparse depth measurements into dense depth maps using various
//! interpolation, variational, and fusion strategies.
//!
//! - **Propagation**: nearest-neighbour, bilateral guided, inverse-distance-weighted
//! - **Variational**: total-variation regularised completion (Chambolle-Pock)
//! - **Fusion**: Kalman-based multi-source depth fusion

pub mod completion;
mod fusion;
mod propagation;
mod types;
mod variational;

pub use fusion::{confidence_weighted_average, fuse_sources, KalmanDepthFusion};
pub use propagation::{bilateral_upsample, inverse_distance_weighted, nearest_neighbor_fill};
pub use types::{
    CompletionMethod, CompletionResult, DepthSource, FusionConfig, SparseDepthMap,
    SparseMeasurement,
};
// Re-export the existing DepthCompletionConfig under a different name to avoid
// ambiguity with the new completion::DepthCompletionConfig.
pub use types::DepthCompletionConfig as DepthCompletionConfigV1;
pub use variational::tv_completion;

// New high-level sparse-to-dense API (f32-based).
pub use completion::{
    apply_bilateral_filter, fill_holes_morphological, DepthCompleter, DepthCompletionConfig,
    DepthMethod, DepthResult,
};
