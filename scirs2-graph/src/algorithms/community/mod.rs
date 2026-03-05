//! Community detection algorithms
//!
//! This module contains algorithms for detecting community structure in graphs.

// Community detection algorithm modules
pub mod fluid;
pub mod girvan_newman;
pub mod hierarchical;
pub mod infomap;
pub mod label_propagation;
pub mod louvain;
pub mod modularity;
pub mod parallel;
pub mod types;

// Re-export core types for backward compatibility and convenience
pub use types::{CommunityResult, CommunityStructure};

// Re-export all main algorithm functions
#[allow(deprecated)]
pub use fluid::fluid_communities;
pub use fluid::fluid_communities_result;

#[allow(deprecated)]
pub use hierarchical::hierarchical_communities;
pub use hierarchical::hierarchical_communities_result;
pub use infomap::{infomap_communities, InfomapResult};

#[allow(deprecated)]
pub use label_propagation::label_propagation;
pub use label_propagation::label_propagation_result;

#[allow(deprecated)]
pub use louvain::louvain_communities;
pub use louvain::louvain_communities_result;

#[allow(deprecated)]
pub use modularity::{greedy_modularity_optimization, modularity_optimization};
pub use modularity::{
    greedy_modularity_optimization_result, modularity, modularity_optimization_result,
};

pub use girvan_newman::{
    girvan_newman_communities_result, girvan_newman_result, DendrogramLevel, GirvanNewmanConfig,
    GirvanNewmanResult,
};

#[allow(deprecated)]
pub use parallel::parallel_louvain_communities;
pub use parallel::parallel_louvain_communities_result;

#[cfg(feature = "parallel")]
pub use parallel::{parallel_label_propagation_result, parallel_modularity};
