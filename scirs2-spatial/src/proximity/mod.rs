//! Proximity queries and spatial graph construction
//!
//! This module provides algorithms for computing distances between geometric objects
//! and constructing proximity-based graphs from point sets.
//!
//! # Features
//!
//! * **Hausdorff distance** between point sets
//! * **Frechet distance** between curves (discrete approximation)
//! * **Euclidean Minimum Spanning Tree** (MST)
//! * **Gabriel graph**
//! * **Relative neighborhood graph**
//! * **Alpha shapes** (concave hull generalization)

mod distance;
mod graphs;

pub use distance::{discrete_frechet_distance, hausdorff_distance_detailed, HausdorffResult};
pub use graphs::{
    alpha_shape_edges, euclidean_mst, gabriel_graph, relative_neighborhood_graph, MstEdge,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Type alias for the hausdorff_distance_detailed function signature
    type HausdorffFn = fn(&[[f64; 2]], &[[f64; 2]]) -> crate::error::SpatialResult<HausdorffResult>;

    #[test]
    fn test_module_imports() {
        // Ensure all public types are accessible
        let _: HausdorffFn = hausdorff_distance_detailed;
    }
}
