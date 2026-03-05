//! Computational geometry algorithms
//!
//! This module provides a collection of fundamental computational geometry algorithms
//! including sweep line intersection detection, bounding rectangle computation,
//! Fortune's algorithm for Voronoi diagrams, and incremental 3D convex hull construction.
//!
//! # Modules
//!
//! - [`sweep_line`] - Bentley-Ottmann sweep line algorithm for line segment intersections
//! - [`bounding`] - Minimum bounding rectangles, AABB, polygon perimeter/area utilities
//! - [`fortune_voronoi`] - Fortune's sweep line algorithm for Voronoi diagram construction
//! - [`incremental_hull_3d`] - Incremental 3D convex hull using DCEL data structure
//!
//! # Examples
//!
//! ## Line Segment Intersection Detection
//!
//! ```
//! use scirs2_spatial::computational_geometry::sweep_line::{Segment2D, find_all_intersections};
//!
//! let segments = vec![
//!     Segment2D::new(0.0, 0.0, 2.0, 2.0),
//!     Segment2D::new(0.0, 2.0, 2.0, 0.0),
//! ];
//!
//! let intersections = find_all_intersections(&segments).expect("Operation failed");
//! assert_eq!(intersections.len(), 1);
//! ```
//!
//! ## Bounding Rectangle
//!
//! ```
//! use scirs2_spatial::computational_geometry::bounding::{axis_aligned_bounding_box, polygon_perimeter};
//! use scirs2_core::ndarray::array;
//!
//! let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
//! let aabb = axis_aligned_bounding_box(&points.view()).expect("Operation failed");
//! assert!((aabb.area() - 1.0).abs() < 1e-10);
//! ```
//!
//! ## Fortune's Voronoi
//!
//! ```
//! use scirs2_spatial::computational_geometry::fortune_voronoi::fortune_voronoi_2d;
//!
//! let sites = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
//! let diagram = fortune_voronoi_2d(&sites).expect("Operation failed");
//! assert_eq!(diagram.num_sites(), 3);
//! ```
//!
//! ## 3D Convex Hull
//!
//! ```
//! use scirs2_spatial::computational_geometry::incremental_hull_3d::IncrementalHull3D;
//!
//! let points = vec![
//!     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
//!     [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
//! ];
//! let hull = IncrementalHull3D::new(&points).expect("Operation failed");
//! assert_eq!(hull.num_faces(), 4);
//! ```

pub mod bounding;
pub mod fortune_voronoi;
pub mod incremental_hull_3d;
pub mod sweep_line;

// Re-export key types for convenience
pub use bounding::{
    axis_aligned_bounding_box, convex_hull_area, convex_hull_area_perimeter, convex_hull_perimeter,
    is_convex, minimum_bounding_rectangle, point_set_diameter, point_set_width, polygon_perimeter,
    signed_polygon_area, OrientedBoundingRect, AABB,
};

pub use sweep_line::{
    count_intersections, find_all_intersections, find_all_intersections_brute_force,
    has_any_intersection, segment_intersection, Intersection, Segment2D,
};

pub use fortune_voronoi::{
    fortune_voronoi_2d, fortune_voronoi_from_array, VoronoiCell, VoronoiDiagram, VoronoiEdge,
    VoronoiVertex,
};

pub use incremental_hull_3d::{HullFace, IncrementalHull3D};
