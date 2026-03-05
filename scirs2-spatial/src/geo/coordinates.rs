//! Coordinate system transformations for geospatial analysis
//!
//! This module provides conversions between:
//! - Geographic (lat/lon) and ECEF (Earth-Centered, Earth-Fixed) Cartesian coordinates
//! - Geographic (lat/lon) and UTM (Universal Transverse Mercator) coordinates
//!
//! All angles are in degrees unless stated otherwise.

use crate::error::{SpatialError, SpatialResult};
use std::f64::consts::PI;

// WGS84 ellipsoid constants
/// WGS84 semi-major axis (equatorial radius) in meters
pub const WGS84_A: f64 = 6_378_137.0;
/// WGS84 flattening
pub const WGS84_F: f64 = 1.0 / 298.257_223_563;
/// WGS84 semi-minor axis (polar radius) in meters
pub const WGS84_B: f64 = WGS84_A * (1.0 - WGS84_F);
/// WGS84 first eccentricity squared
pub const WGS84_E2: f64 = 2.0 * WGS84_F - WGS84_F * WGS84_F;
/// UTM scale factor
const UTM_K0: f64 = 0.9996;
/// UTM false easting
const UTM_FALSE_EASTING: f64 = 500_000.0;
/// UTM false northing for southern hemisphere
const UTM_FALSE_NORTHING_SOUTH: f64 = 10_000_000.0;

/// Convert geographic coordinates (lat/lon + altitude) to ECEF Cartesian coordinates.
///
/// The Earth-Centered, Earth-Fixed (ECEF) coordinate system has its origin at
/// Earth's center of mass, the Z-axis pointing toward the North Pole, and
/// the X-axis pointing toward the intersection of the equator and prime meridian.
///
/// # Arguments
///
/// * `lat` - Geodetic latitude in degrees (positive north)
/// * `lon` - Longitude in degrees (positive east)
/// * `altitude` - Height above WGS84 ellipsoid in meters
///
/// # Returns
///
/// `(x, y, z)` ECEF coordinates in meters.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geo::lat_lon_to_xyz;
///
/// // Convert London's coordinates to ECEF
/// let (x, y, z) = lat_lon_to_xyz(51.5074, -0.1278, 0.0);
/// // X should be approximately 3975000 m, Y ~ -8700 m, Z ~ 4969000 m
/// assert!((x - 3_975_000.0).abs() < 5_000.0);
/// ```
pub fn lat_lon_to_xyz(lat: f64, lon: f64, altitude: f64) -> (f64, f64, f64) {
    let lat_rad = lat.to_radians();
    let lon_rad = lon.to_radians();

    // Prime vertical radius of curvature
    let n = WGS84_A / (1.0 - WGS84_E2 * lat_rad.sin().powi(2)).sqrt();

    let x = (n + altitude) * lat_rad.cos() * lon_rad.cos();
    let y = (n + altitude) * lat_rad.cos() * lon_rad.sin();
    let z = (n * (1.0 - WGS84_E2) + altitude) * lat_rad.sin();

    (x, y, z)
}

/// Convert ECEF Cartesian coordinates to geographic (lat/lon + altitude).
///
/// Uses the iterative Bowring method to handle the ellipsoid accurately.
///
/// # Arguments
///
/// * `x` - ECEF X coordinate in meters
/// * `y` - ECEF Y coordinate in meters
/// * `z` - ECEF Z coordinate in meters
///
/// # Returns
///
/// `(lat, lon, altitude)` where lat/lon are in degrees and altitude is in meters
/// above the WGS84 ellipsoid.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geo::{lat_lon_to_xyz, xyz_to_lat_lon};
///
/// let (x, y, z) = lat_lon_to_xyz(51.5074, -0.1278, 100.0);
/// let (lat, lon, alt) = xyz_to_lat_lon(x, y, z);
///
/// assert!((lat - 51.5074).abs() < 1e-6);
/// assert!((lon - (-0.1278)).abs() < 1e-6);
/// assert!((alt - 100.0).abs() < 1e-3);
/// ```
pub fn xyz_to_lat_lon(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let lon = y.atan2(x);

    let p = (x * x + y * y).sqrt();

    // Initial estimate for latitude using Bowring's method
    let mut lat = z.atan2(p * (1.0 - WGS84_E2));

    // Iterate to convergence (typically 2-3 iterations suffice)
    for _ in 0..10 {
        let sin_lat = lat.sin();
        let n = WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
        let lat_new = (z + WGS84_E2 * n * sin_lat).atan2(p);
        if (lat_new - lat).abs() < 1e-12 {
            lat = lat_new;
            break;
        }
        lat = lat_new;
    }

    let sin_lat = lat.sin();
    let n = WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
    let altitude = if lat.cos().abs() > 1e-10 {
        p / lat.cos() - n
    } else {
        z.abs() / lat.sin().abs() - n * (1.0 - WGS84_E2)
    };

    (lat.to_degrees(), lon.to_degrees(), altitude)
}

/// Convert UTM coordinates to geographic latitude/longitude.
///
/// # Arguments
///
/// * `easting` - UTM easting in meters (typically 100,000 – 900,000)
/// * `northing` - UTM northing in meters (0 – 10,000,000)
/// * `zone` - UTM zone number (1–60)
/// * `north` - `true` for northern hemisphere, `false` for southern hemisphere
///
/// # Returns
///
/// `(latitude, longitude)` in degrees.
///
/// # Errors
///
/// Returns `SpatialError::ValueError` if the zone number is out of range [1, 60].
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geo::utm_to_lat_lon;
///
/// // Zone 30N, New York area
/// let (lat, lon) = utm_to_lat_lon(583960.0, 4507523.0, 18, true).unwrap();
/// assert!((lat - 40.7128).abs() < 0.01);
/// assert!((lon - (-74.0060)).abs() < 0.01);
/// ```
pub fn utm_to_lat_lon(
    easting: f64,
    northing: f64,
    zone: i32,
    north: bool,
) -> SpatialResult<(f64, f64)> {
    if !(1..=60).contains(&zone) {
        return Err(SpatialError::ValueError(format!(
            "UTM zone {zone} is out of valid range [1, 60]"
        )));
    }

    let a = WGS84_A;
    let e2 = WGS84_E2;
    let e4 = e2 * e2;
    let e6 = e4 * e2;

    let e1 = (1.0 - (1.0 - e2).sqrt()) / (1.0 + (1.0 - e2).sqrt());

    let x = easting - UTM_FALSE_EASTING;
    let y = if north {
        northing
    } else {
        northing - UTM_FALSE_NORTHING_SOUTH
    };

    let m = y / UTM_K0;
    let mu = m / (a * (1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0));

    let phi1 = mu
        + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
        + (21.0 * e1.powi(2) / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
        + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin()
        + (1097.0 * e1.powi(4) / 512.0) * (8.0 * mu).sin();

    let sin_phi1 = phi1.sin();
    let cos_phi1 = phi1.cos();
    let tan_phi1 = phi1.tan();

    let n1 = a / (1.0 - e2 * sin_phi1.powi(2)).sqrt();
    let t1 = tan_phi1.powi(2);
    let c1 = e2 * cos_phi1.powi(2) / (1.0 - e2);
    let r1 = a * (1.0 - e2) / (1.0 - e2 * sin_phi1.powi(2)).powf(1.5);
    let d = x / (n1 * UTM_K0);

    let lat = phi1
        - (n1 * tan_phi1 / r1)
            * (d.powi(2) / 2.0
                - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1.powi(2) - 9.0 * e2 / (1.0 - e2))
                    * d.powi(4)
                    / 24.0
                + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1.powi(2)
                    - 252.0 * e2 / (1.0 - e2)
                    - 3.0 * c1.powi(2))
                    * d.powi(6)
                    / 720.0);

    let central_meridian = ((zone - 1) * 6 - 180 + 3) as f64;
    let lon = central_meridian.to_radians()
        + (d - (1.0 + 2.0 * t1 + c1) * d.powi(3) / 6.0
            + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1.powi(2)
                + 8.0 * e2 / (1.0 - e2)
                + 24.0 * t1.powi(2))
                * d.powi(5)
                / 120.0)
            / cos_phi1;

    Ok((lat.to_degrees(), lon.to_degrees()))
}

/// Convert geographic latitude/longitude to UTM coordinates.
///
/// # Arguments
///
/// * `lat` - Latitude in degrees (must be in range [-80, 84])
/// * `lon` - Longitude in degrees
///
/// # Returns
///
/// `(easting, northing, zone, north)` where:
/// - `easting` is in meters
/// - `northing` is in meters
/// - `zone` is the UTM zone number (1–60)
/// - `north` is `true` for northern hemisphere, `false` for southern
///
/// # Errors
///
/// Returns `SpatialError::ValueError` if latitude is outside UTM range [-80, 84].
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geo::lat_lon_to_utm;
///
/// let (easting, northing, zone, north) = lat_lon_to_utm(40.7128, -74.0060).unwrap();
/// assert_eq!(zone, 18);
/// assert!(north);
/// assert!((easting - 583_960.0).abs() < 1_000.0);
/// ```
pub fn lat_lon_to_utm(lat: f64, lon: f64) -> SpatialResult<(f64, f64, i32, bool)> {
    if !(-80.0..=84.0).contains(&lat) {
        return Err(SpatialError::ValueError(format!(
            "Latitude {lat} is outside UTM range [-80, 84]"
        )));
    }

    let zone = compute_utm_zone(lat, lon);
    let north = lat >= 0.0;

    let a = WGS84_A;
    let e2 = WGS84_E2;
    let e4 = e2 * e2;
    let e6 = e4 * e2;

    let lat_rad = lat.to_radians();
    let central_meridian = (((zone - 1) * 6 - 180 + 3) as f64).to_radians();
    let lon_rad = lon.to_radians();

    let sin_lat = lat_rad.sin();
    let cos_lat = lat_rad.cos();
    let tan_lat = lat_rad.tan();

    let n = a / (1.0 - e2 * sin_lat.powi(2)).sqrt();
    let t = tan_lat.powi(2);
    let c = e2 * cos_lat.powi(2) / (1.0 - e2);
    let a_coeff = cos_lat * (lon_rad - central_meridian);

    let m = a
        * ((1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * lat_rad
            - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * (2.0 * lat_rad).sin()
            + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * (4.0 * lat_rad).sin()
            - (35.0 * e6 / 3072.0) * (6.0 * lat_rad).sin());

    let easting = UTM_K0
        * n
        * (a_coeff
            + (1.0 - t + c) * a_coeff.powi(3) / 6.0
            + (5.0 - 18.0 * t + t.powi(2) + 72.0 * c - 58.0 * e2 / (1.0 - e2)) * a_coeff.powi(5)
                / 120.0)
        + UTM_FALSE_EASTING;

    let northing_raw = UTM_K0
        * (m + n
            * tan_lat
            * (a_coeff.powi(2) / 2.0
                + (5.0 - t + 9.0 * c + 4.0 * c.powi(2)) * a_coeff.powi(4) / 24.0
                + (61.0 - 58.0 * t + t.powi(2) + 600.0 * c - 330.0 * e2 / (1.0 - e2))
                    * a_coeff.powi(6)
                    / 720.0));

    let northing = if north {
        northing_raw
    } else {
        northing_raw + UTM_FALSE_NORTHING_SOUTH
    };

    Ok((easting, northing, zone, north))
}

/// Compute the UTM zone number for a given lat/lon pair.
///
/// Handles the special zones for Norway (32V) and Svalbard (31X-37X).
fn compute_utm_zone(lat: f64, lon: f64) -> i32 {
    let mut zone = ((lon + 180.0) / 6.0).floor() as i32 + 1;

    // Special cases for Norway and Svalbard
    if (56.0..64.0).contains(&lat) && (3.0..12.0).contains(&lon) {
        zone = 32;
    }

    if (72.0..84.0).contains(&lat) {
        if (0.0..9.0).contains(&lon) {
            zone = 31;
        } else if (9.0..21.0).contains(&lon) {
            zone = 33;
        } else if (21.0..33.0).contains(&lon) {
            zone = 35;
        } else if (33.0..42.0).contains(&lon) {
            zone = 37;
        }
    }

    zone
}

/// Earth radius in meters (WGS84 mean radius)
pub const EARTH_RADIUS_M: f64 = 6_371_008.8;

/// Haversine formula helper: returns `sin²(Δ/2)` for an angle difference
#[inline]
pub(super) fn hav(delta: f64) -> f64 {
    let s = (delta / 2.0).sin();
    s * s
}

/// Convert degrees to radians (local helper)
#[inline]
pub(super) fn to_rad(deg: f64) -> f64 {
    deg * PI / 180.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lat_lon_to_xyz_equator_prime_meridian() {
        // At equator and prime meridian, x should equal Earth radius
        let (x, y, z) = lat_lon_to_xyz(0.0, 0.0, 0.0);
        assert!((x - WGS84_A).abs() < 1.0, "x={x}");
        assert!(y.abs() < 1.0, "y={y}");
        assert!(z.abs() < 1.0, "z={z}");
    }

    #[test]
    fn test_lat_lon_to_xyz_north_pole() {
        // At North Pole, z should approximately equal polar radius
        let (x, y, z) = lat_lon_to_xyz(90.0, 0.0, 0.0);
        assert!(x.abs() < 1.0, "x={x}");
        assert!(y.abs() < 1.0, "y={y}");
        assert!((z - WGS84_B).abs() < 1.0, "z={z}");
    }

    #[test]
    fn test_xyz_round_trip() {
        let test_points = [
            (0.0, 0.0, 0.0),
            (51.5074, -0.1278, 100.0),
            (-33.8688, 151.2093, 50.0),
            (40.7128, -74.0060, 10.0),
        ];
        for (lat, lon, alt) in test_points {
            let (x, y, z) = lat_lon_to_xyz(lat, lon, alt);
            let (lat2, lon2, alt2) = xyz_to_lat_lon(x, y, z);
            assert!(
                (lat - lat2).abs() < 1e-8,
                "lat mismatch: {lat} vs {lat2} (original: lat={lat}, lon={lon}, alt={alt})"
            );
            assert!(
                (lon - lon2).abs() < 1e-8,
                "lon mismatch: {lon} vs {lon2} (original: lat={lat}, lon={lon}, alt={alt})"
            );
            assert!(
                (alt - alt2).abs() < 1e-3,
                "alt mismatch: {alt} vs {alt2} (original: lat={lat}, lon={lon}, alt={alt})"
            );
        }
    }

    #[test]
    fn test_utm_round_trip() {
        // New York City
        let (e, n, zone, north) = lat_lon_to_utm(40.7128, -74.0060).expect("UTM conversion");
        assert_eq!(zone, 18);
        assert!(north);
        let (lat2, lon2) = utm_to_lat_lon(e, n, zone, north).expect("UTM inverse");
        assert!((lat2 - 40.7128).abs() < 1e-5, "lat mismatch: {lat2}");
        assert!((lon2 - (-74.0060)).abs() < 1e-5, "lon mismatch: {lon2}");
    }

    #[test]
    fn test_utm_southern_hemisphere() {
        // Sydney, Australia (southern hemisphere)
        let (e, n, zone, north) = lat_lon_to_utm(-33.8688, 151.2093).expect("UTM conversion");
        assert_eq!(zone, 56);
        assert!(!north);
        let (lat2, lon2) = utm_to_lat_lon(e, n, zone, north).expect("UTM inverse");
        assert!((lat2 - (-33.8688)).abs() < 1e-5, "lat mismatch: {lat2}");
        assert!((lon2 - 151.2093).abs() < 1e-5, "lon mismatch: {lon2}");
    }

    #[test]
    fn test_utm_invalid_latitude() {
        assert!(lat_lon_to_utm(85.0, 0.0).is_err());
        assert!(lat_lon_to_utm(-85.0, 0.0).is_err());
    }

    #[test]
    fn test_utm_to_lat_lon_invalid_zone() {
        assert!(utm_to_lat_lon(500000.0, 4000000.0, 0, true).is_err());
        assert!(utm_to_lat_lon(500000.0, 4000000.0, 61, true).is_err());
    }
}
