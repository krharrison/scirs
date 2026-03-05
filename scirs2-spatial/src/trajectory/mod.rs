//! Trajectory analysis and similarity
//!
//! This module provides algorithms for working with sequences of 2-D points
//! (trajectories), covering:
//!
//! - **Similarity measures** ([`similarity`]): DTW, Fréchet, Hausdorff, EDR,
//!   ERP.
//! - **Compression** ([`compression`]): Douglas–Peucker, Visvalingam–Whyatt,
//!   Dead Reckoning, online/streaming simplification.
//! - **Clustering** ([`clustering`]): k-medoids with DTW, TRACLUS line-segment
//!   clustering, common sub-trajectory detection.
//! - **Movement patterns** ([`patterns`]): stop detection, turn detection,
//!   speed profiles, convex hull, turning function.
//!
//! # Quick start
//!
//! ```
//! use scirs2_spatial::trajectory::similarity::{dtw_distance, frechet_distance};
//!
//! let t1 = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
//! let t2 = vec![[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]];
//!
//! let dtw  = dtw_distance(&t1, &t2).unwrap();
//! let fre  = frechet_distance(&t1, &t2).unwrap();
//! println!("DTW={dtw:.2}, Fréchet={fre:.2}");
//! ```

pub mod clustering;
pub mod compression;
pub mod patterns;
pub mod similarity;

// Re-export the most commonly used types and functions at the module level.
pub use similarity::{
    continuous_frechet, directed_hausdorff_distance, dtw_distance, dtw_with_window,
    edr_distance, erp_distance, frechet_distance, hausdorff_distance, Point2D, Trajectory,
};

pub use compression::{
    compression_ratio, dead_reckoning, douglas_peucker, visvalingam_whyatt,
    OnlineDouglasPeucker,
};

pub use clustering::{
    trajectory_dtw_matrix, trajectory_kmedoids, traclus_cluster, sub_trajectory_cluster,
    KMedoidsResult, TraclusResult, CommonSubTrajectory, LineSegment,
};

pub use patterns::{
    convex_hull_trajectory, detect_stops, detect_turns, speed_profile, turning_function,
    SpeedSample, Stop, TurnPoint, TurningFunctionSample,
};
