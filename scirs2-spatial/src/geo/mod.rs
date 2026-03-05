//! Geospatial analysis components for SciRS2.
//!
//! This module provides comprehensive geospatial functionality:
//!
//! ## Coordinate Transformations
//!
//! - [`lat_lon_to_xyz`] — Geographic (WGS84) to ECEF Cartesian coordinates
//! - [`xyz_to_lat_lon`] — ECEF Cartesian to geographic (WGS84) coordinates
//! - [`utm_to_lat_lon`] — UTM easting/northing to geographic
//! - [`lat_lon_to_utm`] — Geographic to UTM easting/northing + zone
//!
//! ## Distance Calculations
//!
//! - [`haversine_distance`] — Fast great-circle distance (spherical Earth)
//! - [`vincenty_distance`] — Accurate geodesic distance (WGS84 ellipsoid)
//! - [`bearing`] — Initial bearing in degrees (clockwise from North)
//! - [`destination_point`] — Destination given start, bearing, and distance
//!
//! ## Spatial Indexing
//!
//! - [`GeoHash`] — Compact base-32 spatial index cell
//! - [`geohash_encode`] — Encode lat/lon to geohash string
//! - [`geohash_decode`] — Decode geohash string to center lat/lon
//! - [`geohash_neighbors`] — 8 neighbouring geohash cells
//!
//! ## Bounding Box and Polygon Operations
//!
//! - [`BoundingBox`] — Axis-aligned geographic rectangle
//! - [`polygon_area`] — Area of a lat/lon polygon in m²
//! - [`point_in_polygon`] — Ray-casting point-in-polygon test
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::geo::{
//!     haversine_distance, vincenty_distance, bearing, destination_point,
//!     lat_lon_to_xyz, xyz_to_lat_lon,
//!     lat_lon_to_utm, utm_to_lat_lon,
//!     geohash_encode, geohash_decode, geohash_neighbors,
//!     BoundingBox, polygon_area, point_in_polygon,
//! };
//!
//! // Distance between London and Paris
//! let dist_m = haversine_distance(51.5074, -0.1278, 48.8566, 2.3522);
//! println!("London–Paris: {:.1} km", dist_m / 1000.0);
//!
//! // Convert to ECEF and back
//! let (x, y, z) = lat_lon_to_xyz(51.5074, -0.1278, 0.0);
//! let (lat, lon, alt) = xyz_to_lat_lon(x, y, z);
//!
//! // GeoHash indexing
//! let hash = geohash_encode(48.8566, 2.3522, 6).unwrap();
//! let (dlat, dlon) = geohash_decode(&hash).unwrap();
//! let neighbors = geohash_neighbors(&hash).unwrap();
//!
//! // Bounding box
//! let bbox = BoundingBox::new(48.0, 53.0, -2.0, 4.0).unwrap();
//! assert!(bbox.contains(51.5, 0.0));
//!
//! // Polygon area
//! let poly = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)];
//! let area_m2 = polygon_area(&poly).unwrap();
//! ```

pub mod bbox;
pub mod coordinates;
pub mod distances;
pub mod geohash;

// Re-export coordinate transformations
pub use coordinates::{
    lat_lon_to_utm, lat_lon_to_xyz, utm_to_lat_lon, xyz_to_lat_lon, EARTH_RADIUS_M, WGS84_A,
    WGS84_B, WGS84_E2, WGS84_F,
};

// Re-export distance calculations
pub use distances::{bearing, destination_point, haversine_distance, vincenty_distance};

// Re-export geohash types and helpers
pub use geohash::{geohash_decode, geohash_encode, geohash_neighbors, GeoHash, GeoHashCenter};

// Re-export bounding box and polygon operations
pub use bbox::{point_in_polygon, polygon_area, BoundingBox};

#[cfg(test)]
mod tests {
    use super::*;

    /// Integration test: full pipeline from coordinates to geohash to bbox.
    #[test]
    fn test_full_pipeline_london() {
        let lat = 51.5074_f64;
        let lon = -0.1278_f64;

        // ECEF round-trip
        let (x, y, z) = lat_lon_to_xyz(lat, lon, 50.0);
        let (lat2, lon2, alt2) = xyz_to_lat_lon(x, y, z);
        assert!((lat2 - lat).abs() < 1e-7, "lat={lat2}");
        assert!((lon2 - lon).abs() < 1e-7, "lon={lon2}");
        assert!((alt2 - 50.0).abs() < 0.001, "alt={alt2}");

        // UTM round-trip
        let (e, n, zone, north) = lat_lon_to_utm(lat, lon).expect("utm forward");
        let (lat3, lon3) = utm_to_lat_lon(e, n, zone, north).expect("utm inverse");
        assert!((lat3 - lat).abs() < 1e-5, "utm lat={lat3}");
        assert!((lon3 - lon).abs() < 1e-5, "utm lon={lon3}");

        // Haversine distance to Paris
        let dist = haversine_distance(lat, lon, 48.8566, 2.3522);
        assert!((dist - 343_000.0).abs() < 5_000.0, "dist={dist}");

        // Vincenty distance close to haversine
        let vd = vincenty_distance(lat, lon, 48.8566, 2.3522).expect("vincenty");
        assert!((vd - dist).abs() / dist < 0.005, "vincenty diff");

        // GeoHash encode/decode
        let hash = geohash_encode(lat, lon, 7).expect("encode");
        let (dlat, dlon) = geohash_decode(&hash).expect("decode");
        assert!((dlat - lat).abs() < 0.001, "geohash lat={dlat}");
        assert!((dlon - lon).abs() < 0.001, "geohash lon={dlon}");

        // 8 neighbors
        let neighbors = geohash_neighbors(&hash).expect("neighbors");
        assert_eq!(neighbors.len(), 8);

        // Bounding box
        let bbox = BoundingBox::new(50.0, 53.0, -2.0, 2.0).expect("bbox");
        assert!(bbox.contains(lat, lon));
        assert!(!bbox.contains(48.8566, 2.3522)); // Paris outside
    }

    #[test]
    fn test_destination_and_bearing_inverse() {
        let lat1 = 40.0;
        let lon1 = -70.0;
        let b = 135.0; // Southeast
        let d = 200_000.0; // 200 km

        let (lat2, lon2) = destination_point(lat1, lon1, b, d);
        // bearing from lat1,lon1 to lat2,lon2 should be close to b
        let b2 = bearing(lat1, lon1, lat2, lon2);
        assert!((b2 - b).abs() < 0.5, "bearing roundtrip: {b2} vs {b}");
        // distance should be close to d
        let d2 = haversine_distance(lat1, lon1, lat2, lon2);
        assert!((d2 - d).abs() < 500.0, "distance roundtrip: {d2} vs {d}");
    }

    #[test]
    fn test_polygon_area_and_containment() {
        // Roughly 10°×10° box near equator (~1.2e12 m²)
        let poly = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
        let area = polygon_area(&poly).expect("area");
        assert!(area > 1e11 && area < 2e12, "area={area}");

        assert!(point_in_polygon(5.0, 5.0, &poly));
        assert!(!point_in_polygon(11.0, 5.0, &poly));
    }

    #[test]
    fn test_geohash_neighbor_count() {
        let hash = geohash_encode(35.6762, 139.6503, 5).expect("encode tokyo");
        let neighbors = geohash_neighbors(&hash).expect("neighbors");
        assert_eq!(neighbors.len(), 8);
        for n in &neighbors {
            assert!(!n.is_empty());
            assert_eq!(n.len(), 5);
        }
    }
}
