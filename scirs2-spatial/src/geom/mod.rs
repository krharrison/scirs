//! Computational geometry algorithms for 2D spatial data.
//!
//! This module provides fundamental geometric algorithms commonly used in
//! spatial computing, robotics, GIS, and scientific computing:
//!
//! - [`convex_hull`] — Andrew's monotone chain, Graham scan, Jarvis march,
//!   point-set diameter.
//! - [`polygon`] — Shoelace area, point-in-polygon, centroid, perimeter,
//!   [`Polygon`] struct with hole support.
//! - [`closest_pair`] — O(n log n) divide-and-conquer closest pair,
//!   pairwise distances, farthest pair.
//! - [`triangulation`] — Ear clipping, fan triangulation, circumcircle,
//!   incircle, triangle quality.
//!
//! # Examples
//!
//! ## Convex hull
//!
//! ```
//! use scirs2_spatial::geom::convex_hull_2d;
//!
//! let pts = vec![[0.0_f64,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0],[0.5,0.5]];
//! let hull = convex_hull_2d(pts);
//! assert_eq!(hull.len(), 4);
//! ```
//!
//! ## Polygon area
//!
//! ```
//! use scirs2_spatial::geom::polygon_area;
//!
//! let sq = [[0.0_f64,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]];
//! assert!((polygon_area(&sq) - 1.0).abs() < 1e-12);
//! ```
//!
//! ## Closest pair
//!
//! ```
//! use scirs2_spatial::geom::closest_pair;
//!
//! let pts = [[0.0_f64,0.0],[3.0,4.0],[1.0,1.0]];
//! let (i, j, d) = closest_pair(&pts).unwrap();
//! assert!((d - 2_f64.sqrt()).abs() < 1e-9);
//! ```
//!
//! ## Triangulation
//!
//! ```
//! use scirs2_spatial::geom::ear_clipping;
//!
//! let sq = [[0.0_f64,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]];
//! let tris = ear_clipping(&sq);
//! assert_eq!(tris.len(), 2);
//! ```

pub mod convex_hull;
pub mod polygon;
pub mod closest_pair;
pub mod triangulation;

// ── Re-exports ─────────────────────────────────────────────────────────────────

pub use convex_hull::{convex_hull_2d, GrahamScan, JarvisMarch, point_set_diameter};
pub use polygon::{polygon_area, point_in_polygon, polygon_centroid, polygon_perimeter, Polygon};
pub use closest_pair::{closest_pair, closest_pair_brute, pairwise_distances, farthest_pair};
pub use triangulation::{
    ear_clipping, fan_triangulation, triangle_area, circumcircle, incircle_radius,
    triangle_quality, point_cloud_triangulation,
};
