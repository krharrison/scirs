//! Geodesic and great-circle distance calculations.
//!
//! Implements:
//! - Haversine formula (fast, spherical Earth)
//! - Vincenty inverse formula (accurate, WGS84 ellipsoid)
//! - Initial bearing (forward azimuth)
//! - Destination point given distance and bearing

use super::coordinates::{to_rad, EARTH_RADIUS_M, WGS84_A, WGS84_B, WGS84_F};
use crate::error::{SpatialError, SpatialResult};

/// Calculate the great-circle distance between two points using the Haversine formula.
///
/// This is the standard formula for calculating distances on a sphere. It is
/// numerically stable even for very short distances.
///
/// # Arguments
///
/// * `lat1` - Latitude of first point in degrees (positive north)
/// * `lon1` - Longitude of first point in degrees (positive east)
/// * `lat2` - Latitude of second point in degrees
/// * `lon2` - Longitude of second point in degrees
///
/// # Returns
///
/// Distance in meters.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geo::haversine_distance;
///
/// // New York to London
/// let dist = haversine_distance(40.7128, -74.0060, 51.5074, -0.1278);
/// // Should be approximately 5,570 km
/// assert!((dist - 5_570_000.0).abs() < 50_000.0);
/// ```
pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dlat = to_rad(lat2 - lat1);
    let dlon = to_rad(lon2 - lon1);
    let lat1_rad = to_rad(lat1);
    let lat2_rad = to_rad(lat2);

    let a =
        (dlat / 2.0).sin().powi(2) + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();

    EARTH_RADIUS_M * c
}

/// Calculate the geodesic distance between two points using Vincenty's inverse formula.
///
/// More accurate than Haversine for long distances as it uses the WGS84 ellipsoid
/// rather than a spherical approximation.
///
/// # Arguments
///
/// * `lat1` - Latitude of first point in degrees
/// * `lon1` - Longitude of first point in degrees
/// * `lat2` - Latitude of second point in degrees
/// * `lon2` - Longitude of second point in degrees
///
/// # Returns
///
/// Distance in meters, or an error if the formula fails to converge (can happen
/// for nearly antipodal points).
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geo::vincenty_distance;
///
/// // London to Paris (~340 km)
/// let dist = vincenty_distance(51.5074, -0.1278, 48.8566, 2.3522).unwrap();
/// assert!((dist - 340_000.0).abs() < 5_000.0);
/// ```
pub fn vincenty_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> SpatialResult<f64> {
    let lat1_rad = to_rad(lat1);
    let lat2_rad = to_rad(lat2);
    let lon1_rad = to_rad(lon1);
    let lon2_rad = to_rad(lon2);

    let f = WGS84_F;
    let a = WGS84_A;
    let b = WGS84_B;

    let l = lon2_rad - lon1_rad;
    let u1 = ((1.0 - f) * lat1_rad.tan()).atan();
    let u2 = ((1.0 - f) * lat2_rad.tan()).atan();

    let sin_u1 = u1.sin();
    let cos_u1 = u1.cos();
    let sin_u2 = u2.sin();
    let cos_u2 = u2.cos();

    let mut lambda = l;

    'convergence: {
        for _ in 0..1000 {
            let sin_lambda = lambda.sin();
            let cos_lambda = lambda.cos();

            let sin_sigma_sq = (cos_u2 * sin_lambda).powi(2)
                + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda).powi(2);
            let sin_sigma = sin_sigma_sq.sqrt();

            if sin_sigma.abs() < 1e-12 {
                // Co-incident points
                break 'convergence Ok(0.0);
            }

            let cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
            let sigma = sin_sigma.atan2(cos_sigma);

            let sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma;
            let cos_sq_alpha = 1.0 - sin_alpha.powi(2);

            let cos_2sigma_m = if cos_sq_alpha.abs() < 1e-12 {
                0.0 // equatorial line
            } else {
                cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha
            };

            let c = f / 16.0 * cos_sq_alpha * (4.0 + f * (4.0 - 3.0 * cos_sq_alpha));

            let lambda_new = l
                + (1.0 - c)
                    * f
                    * sin_alpha
                    * (sigma
                        + c * sin_sigma
                            * (cos_2sigma_m + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m.powi(2))));

            if (lambda_new - lambda).abs() < 1e-12 {
                // Converged — compute distance
                let u_sq = cos_sq_alpha * (a * a - b * b) / (b * b);
                let big_a = 1.0
                    + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
                let big_b = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));
                let delta_sigma = big_b
                    * sin_sigma
                    * (cos_2sigma_m
                        + big_b / 4.0
                            * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m.powi(2))
                                - big_b / 6.0
                                    * cos_2sigma_m
                                    * (-3.0 + 4.0 * sin_sigma.powi(2))
                                    * (-3.0 + 4.0 * cos_2sigma_m.powi(2))));
                break 'convergence Ok(b * big_a * (sigma - delta_sigma));
            }

            lambda = lambda_new;
        }

        Err(SpatialError::ComputationError(
            "Vincenty formula did not converge (nearly antipodal points?)".to_string(),
        ))
    }
}

/// Calculate the initial bearing (forward azimuth) from point 1 to point 2.
///
/// The bearing is measured clockwise from North (0°) through East (90°),
/// South (180°) and West (270°).
///
/// # Arguments
///
/// * `lat1` - Latitude of first point in degrees
/// * `lon1` - Longitude of first point in degrees
/// * `lat2` - Latitude of second point in degrees
/// * `lon2` - Longitude of second point in degrees
///
/// # Returns
///
/// Bearing in degrees [0, 360).
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geo::bearing;
///
/// // Bearing from equator due north
/// let b = bearing(0.0, 0.0, 1.0, 0.0);
/// assert!((b - 0.0).abs() < 1e-6 || (b - 360.0).abs() < 1e-6);
///
/// // Bearing due east
/// let b = bearing(0.0, 0.0, 0.0, 1.0);
/// assert!((b - 90.0).abs() < 0.01);
/// ```
pub fn bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let lat1_rad = to_rad(lat1);
    let lat2_rad = to_rad(lat2);
    let dlon = to_rad(lon2 - lon1);

    let y = dlon.sin() * lat2_rad.cos();
    let x = lat1_rad.cos() * lat2_rad.sin() - lat1_rad.sin() * lat2_rad.cos() * dlon.cos();

    let theta = y.atan2(x).to_degrees();
    // Normalize to [0, 360)
    (theta + 360.0) % 360.0
}

/// Calculate the destination point given a starting point, bearing, and distance.
///
/// Uses the spherical Earth model with Earth's mean radius.
///
/// # Arguments
///
/// * `lat` - Starting latitude in degrees
/// * `lon` - Starting longitude in degrees
/// * `bearing_deg` - Bearing in degrees (0 = North, 90 = East)
/// * `distance` - Distance to travel in meters
///
/// # Returns
///
/// `(latitude, longitude)` of the destination point in degrees.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geo::destination_point;
///
/// // Travel 100 km due north from (0, 0)
/// let (lat, lon) = destination_point(0.0, 0.0, 0.0, 100_000.0);
/// assert!(lat > 0.0);
/// assert!(lon.abs() < 0.001);
/// ```
pub fn destination_point(lat: f64, lon: f64, bearing_deg: f64, distance: f64) -> (f64, f64) {
    let lat_rad = to_rad(lat);
    let lon_rad = to_rad(lon);
    let bearing_rad = to_rad(bearing_deg);

    let angular_dist = distance / EARTH_RADIUS_M;

    let lat2_rad = (lat_rad.sin() * angular_dist.cos()
        + lat_rad.cos() * angular_dist.sin() * bearing_rad.cos())
    .asin();

    let lon2_rad = lon_rad
        + (bearing_rad.sin() * angular_dist.sin() * lat_rad.cos())
            .atan2(angular_dist.cos() - lat_rad.sin() * lat2_rad.sin());

    (lat2_rad.to_degrees(), lon2_rad.to_degrees())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine_same_point() {
        assert!(haversine_distance(51.5074, -0.1278, 51.5074, -0.1278) < 1e-6);
    }

    #[test]
    fn test_haversine_london_paris() {
        let dist = haversine_distance(51.5074, -0.1278, 48.8566, 2.3522);
        // Known ~343 km
        assert!((dist - 343_000.0).abs() < 5_000.0, "dist={dist}");
    }

    #[test]
    fn test_haversine_ny_london() {
        let dist = haversine_distance(40.7128, -74.0060, 51.5074, -0.1278);
        // Known ~5570 km
        assert!((dist - 5_570_000.0).abs() < 50_000.0, "dist={dist}");
    }

    #[test]
    fn test_vincenty_vs_haversine_short_distance() {
        let h = haversine_distance(51.5074, -0.1278, 48.8566, 2.3522);
        let v = vincenty_distance(51.5074, -0.1278, 48.8566, 2.3522).expect("vincenty");
        // Should agree within 0.5%
        let diff_pct = (h - v).abs() / v * 100.0;
        assert!(diff_pct < 0.5, "diff={diff_pct}%");
    }

    #[test]
    fn test_vincenty_same_point() {
        let d = vincenty_distance(0.0, 0.0, 0.0, 0.0).expect("vincenty");
        assert!(d < 1e-6, "d={d}");
    }

    #[test]
    fn test_bearing_north() {
        let b = bearing(0.0, 0.0, 1.0, 0.0);
        assert!(b < 1e-6 || (b - 360.0).abs() < 1e-6, "b={b}");
    }

    #[test]
    fn test_bearing_east() {
        let b = bearing(0.0, 0.0, 0.0, 1.0);
        assert!((b - 90.0).abs() < 0.01, "b={b}");
    }

    #[test]
    fn test_bearing_south() {
        let b = bearing(1.0, 0.0, 0.0, 0.0);
        assert!((b - 180.0).abs() < 0.01, "b={b}");
    }

    #[test]
    fn test_bearing_west() {
        let b = bearing(0.0, 1.0, 0.0, 0.0);
        assert!((b - 270.0).abs() < 0.01, "b={b}");
    }

    #[test]
    fn test_destination_point_north() {
        let (lat2, lon2) = destination_point(0.0, 0.0, 0.0, 100_000.0);
        assert!(lat2 > 0.0, "lat2={lat2}");
        assert!(lon2.abs() < 0.001, "lon2={lon2}");
        // Approximately 0.899 degrees of latitude per 100 km
        let back = haversine_distance(0.0, 0.0, lat2, lon2);
        assert!((back - 100_000.0).abs() < 100.0, "back={back}");
    }

    #[test]
    fn test_destination_roundtrip() {
        let (lat2, lon2) = destination_point(40.0, -74.0, 45.0, 500_000.0);
        let b = bearing(40.0, -74.0, lat2, lon2);
        assert!((b - 45.0).abs() < 0.1, "bearing={b}");
        let d = haversine_distance(40.0, -74.0, lat2, lon2);
        assert!((d - 500_000.0).abs() < 1000.0, "dist={d}");
    }
}
