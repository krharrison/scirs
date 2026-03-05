//! Geographic bounding box and polygon operations.
//!
//! Provides:
//! - `BoundingBox` for axis-aligned geographic rectangles
//! - `polygon_area` using the Shoelace formula (result in m²)
//! - `point_in_polygon` using ray-casting (works for (lat, lon) pairs)

use super::coordinates::EARTH_RADIUS_M;
use crate::error::{SpatialError, SpatialResult};
use std::f64::consts::PI;

/// An axis-aligned geographic bounding box defined by min/max lat and lon.
///
/// Coordinates are in degrees (latitude: [-90, 90], longitude: [-180, 180]).
/// This struct does **not** handle anti-meridian crossing (bounding boxes
/// that wrap around longitude ±180°). For those use cases, split into two boxes.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geo::BoundingBox;
///
/// let bbox = BoundingBox::new(48.0, 52.0, -1.0, 3.0).unwrap();
/// assert!(bbox.contains(50.0, 1.0));
/// assert!(!bbox.contains(53.0, 1.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    /// Minimum latitude in degrees
    pub min_lat: f64,
    /// Maximum latitude in degrees
    pub max_lat: f64,
    /// Minimum longitude in degrees
    pub min_lon: f64,
    /// Maximum longitude in degrees
    pub max_lon: f64,
}

impl BoundingBox {
    /// Create a new `BoundingBox`.
    ///
    /// # Arguments
    ///
    /// * `min_lat` - Minimum latitude in degrees
    /// * `max_lat` - Maximum latitude in degrees
    /// * `min_lon` - Minimum longitude in degrees
    /// * `max_lon` - Maximum longitude in degrees
    ///
    /// # Errors
    ///
    /// Returns `SpatialError::ValueError` if:
    /// - `min_lat > max_lat`
    /// - `min_lon > max_lon`
    /// - Any value is out of range (lat ∉ [-90, 90], lon ∉ [-180, 180])
    pub fn new(min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> SpatialResult<Self> {
        if min_lat > max_lat {
            return Err(SpatialError::ValueError(format!(
                "min_lat ({min_lat}) > max_lat ({max_lat})"
            )));
        }
        if min_lon > max_lon {
            return Err(SpatialError::ValueError(format!(
                "min_lon ({min_lon}) > max_lon ({max_lon})"
            )));
        }
        if !(-90.0..=90.0).contains(&min_lat) || !(-90.0..=90.0).contains(&max_lat) {
            return Err(SpatialError::ValueError(
                "Latitude values must be in range [-90, 90]".to_string(),
            ));
        }
        if !(-180.0..=180.0).contains(&min_lon) || !(-180.0..=180.0).contains(&max_lon) {
            return Err(SpatialError::ValueError(
                "Longitude values must be in range [-180, 180]".to_string(),
            ));
        }
        Ok(BoundingBox {
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        })
    }

    /// Create a `BoundingBox` from a set of geographic points.
    ///
    /// # Arguments
    ///
    /// * `points` - Slice of `(lat, lon)` pairs in degrees
    ///
    /// # Errors
    ///
    /// Returns `SpatialError::ValueError` if the slice is empty.
    pub fn from_points(points: &[(f64, f64)]) -> SpatialResult<Self> {
        if points.is_empty() {
            return Err(SpatialError::ValueError(
                "Cannot create BoundingBox from empty point set".to_string(),
            ));
        }
        let mut min_lat = f64::INFINITY;
        let mut max_lat = f64::NEG_INFINITY;
        let mut min_lon = f64::INFINITY;
        let mut max_lon = f64::NEG_INFINITY;

        for &(lat, lon) in points {
            if lat < min_lat {
                min_lat = lat;
            }
            if lat > max_lat {
                max_lat = lat;
            }
            if lon < min_lon {
                min_lon = lon;
            }
            if lon > max_lon {
                max_lon = lon;
            }
        }
        Ok(BoundingBox {
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        })
    }

    /// Test whether a geographic point is inside (or on the boundary of) this box.
    ///
    /// # Arguments
    ///
    /// * `lat` - Latitude in degrees
    /// * `lon` - Longitude in degrees
    pub fn contains(&self, lat: f64, lon: f64) -> bool {
        lat >= self.min_lat && lat <= self.max_lat && lon >= self.min_lon && lon <= self.max_lon
    }

    /// Test whether two bounding boxes overlap (share any area or boundary point).
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        self.min_lat <= other.max_lat
            && self.max_lat >= other.min_lat
            && self.min_lon <= other.max_lon
            && self.max_lon >= other.min_lon
    }

    /// Return a new `BoundingBox` expanded by `margin_deg` degrees in all four directions.
    ///
    /// Clamps the result to valid lat/lon ranges so the returned box is always valid.
    ///
    /// # Arguments
    ///
    /// * `margin_deg` - Expansion amount in degrees (must be ≥ 0)
    ///
    /// # Errors
    ///
    /// Returns `SpatialError::ValueError` if `margin_deg` is negative.
    pub fn expand(&self, margin_deg: f64) -> SpatialResult<BoundingBox> {
        if margin_deg < 0.0 {
            return Err(SpatialError::ValueError(format!(
                "margin_deg ({margin_deg}) must be non-negative"
            )));
        }
        Ok(BoundingBox {
            min_lat: (self.min_lat - margin_deg).max(-90.0),
            max_lat: (self.max_lat + margin_deg).min(90.0),
            min_lon: (self.min_lon - margin_deg).max(-180.0),
            max_lon: (self.max_lon + margin_deg).min(180.0),
        })
    }

    /// Return the center `(lat, lon)` of the bounding box.
    pub fn center(&self) -> (f64, f64) {
        (
            (self.min_lat + self.max_lat) / 2.0,
            (self.min_lon + self.max_lon) / 2.0,
        )
    }

    /// Return the width of the bounding box in degrees (longitude span).
    pub fn width_deg(&self) -> f64 {
        self.max_lon - self.min_lon
    }

    /// Return the height of the bounding box in degrees (latitude span).
    pub fn height_deg(&self) -> f64 {
        self.max_lat - self.min_lat
    }

    /// Approximate area of the bounding box in square metres.
    ///
    /// Uses the spherical Earth model. Suitable for boxes up to a few hundred km wide.
    pub fn area_m2(&self) -> f64 {
        let lat_center = (self.min_lat + self.max_lat) / 2.0;
        let m_per_deg_lat = PI / 180.0 * EARTH_RADIUS_M;
        let m_per_deg_lon = PI / 180.0 * EARTH_RADIUS_M * lat_center.to_radians().cos();
        self.height_deg() * m_per_deg_lat * self.width_deg() * m_per_deg_lon
    }
}

/// Compute the area of a planar polygon given as `(lat, lon)` pairs using the
/// Shoelace (Gauss's area) formula projected onto the Earth's surface.
///
/// This function treats coordinates as planar (i.e. uses a flat-Earth approximation
/// at the centroid latitude) and returns a result in square metres. It is accurate
/// for polygons up to ~100 km across. For larger polygons, use spherical methods.
///
/// The polygon can be open (first ≠ last) or closed (first == last); the algorithm
/// handles both.
///
/// # Arguments
///
/// * `coords` - Slice of `(lat, lon)` pairs in degrees. Must have at least 3 points.
///
/// # Returns
///
/// Area in square metres (always positive regardless of vertex winding).
///
/// # Errors
///
/// Returns `SpatialError::ValueError` if fewer than 3 coordinate pairs are given.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geo::polygon_area;
///
/// // Approximately 1° × 1° square near the equator (~12,321 km²)
/// let square = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)];
/// let area = polygon_area(&square).unwrap();
/// assert!((area - 12_321_000_000.0_f64).abs() / 12_321_000_000.0 < 0.01);
/// ```
pub fn polygon_area(coords: &[(f64, f64)]) -> SpatialResult<f64> {
    if coords.len() < 3 {
        return Err(SpatialError::ValueError(
            "Polygon must have at least 3 vertices".to_string(),
        ));
    }

    // Compute the centroid latitude for the projection
    let centroid_lat = coords.iter().map(|(lat, _)| lat).sum::<f64>() / coords.len() as f64;
    let m_per_deg_lat = PI / 180.0 * EARTH_RADIUS_M;
    let m_per_deg_lon = PI / 180.0 * EARTH_RADIUS_M * centroid_lat.to_radians().cos();

    // Shoelace formula in projected (metre) space
    let n = coords.len();
    let mut area2 = 0.0_f64;
    for i in 0..n {
        let j = (i + 1) % n;
        let xi = coords[i].1 * m_per_deg_lon; // lon → x (easting)
        let yi = coords[i].0 * m_per_deg_lat; // lat → y (northing)
        let xj = coords[j].1 * m_per_deg_lon;
        let yj = coords[j].0 * m_per_deg_lat;
        area2 += xi * yj - xj * yi;
    }

    Ok(area2.abs() / 2.0)
}

/// Test whether a point is inside a polygon using the ray-casting algorithm.
///
/// Works for general (non-convex) polygons in geographic coordinates. The polygon
/// can be open or closed (first vertex need not equal last). This uses a planar
/// algorithm which is appropriate for polygons smaller than a continent.
///
/// Points exactly on an edge are considered inside.
///
/// # Arguments
///
/// * `lat` - Latitude of the test point in degrees
/// * `lon` - Longitude of the test point in degrees
/// * `polygon` - Slice of `(lat, lon)` pairs defining the polygon boundary
///
/// # Returns
///
/// `true` if the point is inside (or on the boundary of) the polygon.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geo::point_in_polygon;
///
/// let square = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
///
/// assert!(point_in_polygon(5.0, 5.0, &square));   // inside
/// assert!(!point_in_polygon(11.0, 5.0, &square)); // outside
/// assert!(point_in_polygon(0.0, 0.0, &square));   // corner
/// ```
pub fn point_in_polygon(lat: f64, lon: f64, polygon: &[(f64, f64)]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }

    // Check corners explicitly for boundary detection
    for &(vlat, vlon) in polygon {
        if (vlat - lat).abs() < f64::EPSILON && (vlon - lon).abs() < f64::EPSILON {
            return true;
        }
    }

    let mut inside = false;
    let mut j = n - 1;

    for i in 0..n {
        let (yi, xi) = polygon[i];
        let (yj, xj) = polygon[j];

        // Check if point is on this edge
        if is_on_segment(lat, lon, yi, xi, yj, xj) {
            return true;
        }

        // Standard ray-casting
        if ((yi > lat) != (yj > lat)) && (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }

    inside
}

/// Test whether `(py, px)` lies on the line segment `(y0, x0) – (y1, x1)`.
fn is_on_segment(py: f64, px: f64, y0: f64, x0: f64, y1: f64, x1: f64) -> bool {
    // Cross-product zero means collinear
    let cross = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0);
    if cross.abs() > 1e-10 {
        return false;
    }
    // Check that point is within the bounding box of the segment
    let min_x = x0.min(x1);
    let max_x = x0.max(x1);
    let min_y = y0.min(y1);
    let max_y = y0.max(y1);
    px >= min_x && px <= max_x && py >= min_y && py <= max_y
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- BoundingBox ---

    #[test]
    fn test_bbox_contains() {
        let bbox = BoundingBox::new(48.0, 52.0, -1.0, 3.0).expect("bbox");
        assert!(bbox.contains(50.0, 1.0));
        assert!(bbox.contains(48.0, -1.0)); // corner
        assert!(!bbox.contains(53.0, 1.0));
        assert!(!bbox.contains(50.0, 4.0));
    }

    #[test]
    fn test_bbox_intersects() {
        let a = BoundingBox::new(0.0, 2.0, 0.0, 2.0).expect("a");
        let b = BoundingBox::new(1.0, 3.0, 1.0, 3.0).expect("b");
        let c = BoundingBox::new(3.0, 5.0, 3.0, 5.0).expect("c");
        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_bbox_expand() {
        let bbox = BoundingBox::new(10.0, 20.0, 10.0, 20.0).expect("bbox");
        let expanded = bbox.expand(5.0).expect("expand");
        assert!((expanded.min_lat - 5.0).abs() < 1e-10);
        assert!((expanded.max_lat - 25.0).abs() < 1e-10);
        assert!((expanded.min_lon - 5.0).abs() < 1e-10);
        assert!((expanded.max_lon - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbox_expand_clamped() {
        let bbox = BoundingBox::new(-89.0, 89.0, -179.0, 179.0).expect("bbox");
        let expanded = bbox.expand(5.0).expect("expand");
        assert!(expanded.min_lat >= -90.0);
        assert!(expanded.max_lat <= 90.0);
        assert!(expanded.min_lon >= -180.0);
        assert!(expanded.max_lon <= 180.0);
    }

    #[test]
    fn test_bbox_from_points() {
        let points = [(1.0, 2.0), (3.0, 4.0), (2.0, 1.0)];
        let bbox = BoundingBox::from_points(&points).expect("from_points");
        assert!((bbox.min_lat - 1.0).abs() < 1e-10);
        assert!((bbox.max_lat - 3.0).abs() < 1e-10);
        assert!((bbox.min_lon - 1.0).abs() < 1e-10);
        assert!((bbox.max_lon - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbox_center() {
        let bbox = BoundingBox::new(0.0, 10.0, 0.0, 20.0).expect("bbox");
        let (lat, lon) = bbox.center();
        assert!((lat - 5.0).abs() < 1e-10);
        assert!((lon - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbox_invalid() {
        assert!(BoundingBox::new(10.0, 5.0, 0.0, 10.0).is_err());
        assert!(BoundingBox::new(0.0, 10.0, 10.0, 5.0).is_err());
        assert!(BoundingBox::new(-91.0, 10.0, 0.0, 10.0).is_err());
        assert!(BoundingBox::new(0.0, 10.0, 0.0, 181.0).is_err());
    }

    // --- polygon_area ---

    #[test]
    fn test_polygon_area_unit_degree_square() {
        // 1°×1° square near equator
        let square = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)];
        let area = polygon_area(&square).expect("area");
        // Expected ~12,321 km² = 1.2321e10 m²
        let expected = 12_321_000_000.0;
        assert!(
            (area - expected).abs() / expected < 0.01,
            "area={area}, expected≈{expected}"
        );
    }

    #[test]
    fn test_polygon_area_winding_independent() {
        // CW and CCW should give same area
        let cw = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let ccw = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)];
        let a1 = polygon_area(&cw).expect("area cw");
        let a2 = polygon_area(&ccw).expect("area ccw");
        assert!((a1 - a2).abs() < 1.0, "areas differ: {a1} vs {a2}");
    }

    #[test]
    fn test_polygon_area_too_few_points() {
        assert!(polygon_area(&[(0.0, 0.0), (1.0, 0.0)]).is_err());
    }

    // --- point_in_polygon ---

    #[test]
    fn test_point_in_polygon_inside() {
        let square = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
        assert!(point_in_polygon(5.0, 5.0, &square));
    }

    #[test]
    fn test_point_in_polygon_outside() {
        let square = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
        assert!(!point_in_polygon(11.0, 5.0, &square));
        assert!(!point_in_polygon(5.0, 11.0, &square));
    }

    #[test]
    fn test_point_in_polygon_corner() {
        let square = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
        assert!(point_in_polygon(0.0, 0.0, &square));
        assert!(point_in_polygon(10.0, 10.0, &square));
    }

    #[test]
    fn test_point_in_polygon_nonconvex() {
        // L-shaped polygon
        let poly = [
            (0.0, 0.0),
            (0.0, 6.0),
            (3.0, 6.0),
            (3.0, 3.0),
            (6.0, 3.0),
            (6.0, 0.0),
        ];
        assert!(point_in_polygon(1.0, 1.0, &poly)); // in bottom-left
        assert!(point_in_polygon(1.0, 5.0, &poly)); // in top-left
        assert!(!point_in_polygon(4.0, 5.0, &poly)); // in notch (outside)
    }
}
