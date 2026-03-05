//! Descriptor matching algorithms.
//!
//! # Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`brute_force`] | O(n·m) L2 and Hamming matching with cross-check filter |
//! | [`flann`] | KD-tree approximate nearest-neighbour matching |
//! | `ratio_test` | Lowe's ratio test, RANSAC fundamental matrix, essential matrix |

pub mod brute_force;
pub mod flann;
pub mod ratio_test;

// Convenience re-exports
pub use brute_force::{
    cross_check_filter, cross_check_filter_hamming, match_descriptors_hamming,
    match_descriptors_l2, BruteForceMatch, DistanceMetric,
};
pub use flann::FlannMatcher;
pub use ratio_test::{essential_from_fundamental, ransac_fundamental, ratio_test};
