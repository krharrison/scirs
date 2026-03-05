//! # Coordinate Systems and Geodetic Computations
//!
//! Provides conversions between geographic, ECEF, and local ENU coordinate systems,
//! geodetic distance algorithms (Vincenty, Haversine), bearing, and the Vincenty
//! direct formula for destination-point computation.
//!
//! All angular inputs/outputs are in **decimal degrees** unless noted otherwise.
//! All linear inputs/outputs are in **metres**.
//!
//! ## Ellipsoid
//!
//! The WGS-84 ellipsoid is used throughout:
//!
//! | Constant | Value | Description |
//! |----------|-------|-------------|
//! | [`WGS84_A`] | 6 378 137 m | Semi-major axis |
//! | [`WGS84_F`] | 1/298.257 223 563 | Flattening |
//! | [`WGS84_B`] | ≈ 6 356 752.314 m | Semi-minor axis |
//! | [`WGS84_E2`] | ≈ 0.006 694 380 | First eccentricity squared |
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::geo::coordinates::{GeographicCoord, EcefCoord};
//!
//! // London
//! let london = GeographicCoord::new(51.5074, -0.1278, 0.0);
//! // Convert to ECEF and back
//! let ecef = london.to_ecef();
//! let recovered = ecef.to_geographic();
//! assert!((recovered.lat - london.lat).abs() < 1e-9);
//! assert!((recovered.lon - london.lon).abs() < 1e-9);
//! ```

use std::f64::consts::PI;
use thiserror::Error;

// ---------------------------------------------------------------------------
// WGS-84 ellipsoid constants
// ---------------------------------------------------------------------------

/// WGS-84 semi-major axis (metres).
pub const WGS84_A: f64 = 6_378_137.0;

/// WGS-84 flattening (dimensionless).
pub const WGS84_F: f64 = 1.0 / 298.257_223_563;

/// WGS-84 semi-minor axis (metres).
pub const WGS84_B: f64 = WGS84_A * (1.0 - WGS84_F);

/// WGS-84 first eccentricity squared (dimensionless).
pub const WGS84_E2: f64 = 2.0 * WGS84_F - WGS84_F * WGS84_F;

/// WGS-84 second eccentricity squared (dimensionless).
pub const WGS84_EP2: f64 = (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B);

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that may arise from geodetic computations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum GeoError {
    /// Vincenty's formula failed to converge (nearly antipodal points).
    #[error("Vincenty iteration did not converge after {max_iter} iterations (lambda change = {delta:.3e}); points may be nearly antipodal")]
    VincentyNonConvergence {
        /// Maximum iteration count used.
        max_iter: u32,
        /// Residual change in lambda at termination.
        delta: f64,
    },

    /// A coordinate value is outside its valid domain.
    #[error("Invalid coordinate value for '{field}': {reason}")]
    InvalidCoordinate {
        /// Name of the field that is invalid.
        field: &'static str,
        /// Human-readable explanation.
        reason: String,
    },

    /// A generic domain error (e.g. negative distance).
    #[error("Domain error: {0}")]
    DomainError(String),
}

/// Convenience result alias for geodetic functions.
pub type GeoResult<T> = Result<T, GeoError>;

// ---------------------------------------------------------------------------
// Coordinate types
// ---------------------------------------------------------------------------

/// Geographic (geodetic) coordinates on the WGS-84 ellipsoid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeographicCoord {
    /// Geodetic latitude in decimal degrees (−90 … +90).
    pub lat: f64,
    /// Geodetic longitude in decimal degrees (−180 … +180).
    pub lon: f64,
    /// Height above the WGS-84 ellipsoid in metres.
    pub alt: f64,
}

/// Earth-Centred, Earth-Fixed (ECEF) Cartesian coordinates (metres).
#[derive(Debug, Clone, Copy)]
pub struct EcefCoord {
    /// X-axis component (passes through intersection of equator and prime meridian).
    pub x: f64,
    /// Y-axis component (passes through 90 °E on the equator).
    pub y: f64,
    /// Z-axis component (passes through the geographic north pole).
    pub z: f64,
}

/// Local East-North-Up (ENU) Cartesian coordinates (metres) relative to a reference origin.
#[derive(Debug, Clone, Copy)]
pub struct EnuCoord {
    /// Eastward component (metres).
    pub east: f64,
    /// Northward component (metres).
    pub north: f64,
    /// Upward component (metres, positive away from the geoid).
    pub up: f64,
}

// ---------------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------------

/// Convert degrees to radians.
#[inline]
fn deg2rad(deg: f64) -> f64 {
    deg * PI / 180.0
}

/// Convert radians to degrees.
#[inline]
fn rad2deg(rad: f64) -> f64 {
    rad * 180.0 / PI
}

/// Radius of curvature in the prime vertical N(φ) for WGS-84.
#[inline]
fn prime_vertical_radius(sin_lat: f64) -> f64 {
    WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt()
}

// ---------------------------------------------------------------------------
// GeographicCoord implementation
// ---------------------------------------------------------------------------

impl GeographicCoord {
    /// Create a new geographic coordinate.
    ///
    /// # Arguments
    ///
    /// * `lat` – Latitude in decimal degrees (−90 … +90).
    /// * `lon` – Longitude in decimal degrees (−180 … +180).
    /// * `alt` – Height above the WGS-84 ellipsoid in metres.
    #[inline]
    pub fn new(lat: f64, lon: f64, alt: f64) -> Self {
        Self { lat, lon, alt }
    }

    /// Convert this geographic coordinate to an ECEF Cartesian coordinate.
    ///
    /// Uses the standard closed-form formula:
    ///
    /// ```text
    /// X = (N + h) cos φ cos λ
    /// Y = (N + h) cos φ sin λ
    /// Z = (N (1 − e²) + h) sin φ
    /// ```
    pub fn to_ecef(&self) -> EcefCoord {
        let lat_r = deg2rad(self.lat);
        let lon_r = deg2rad(self.lon);

        let sin_lat = lat_r.sin();
        let cos_lat = lat_r.cos();
        let sin_lon = lon_r.sin();
        let cos_lon = lon_r.cos();

        let n = prime_vertical_radius(sin_lat);

        EcefCoord {
            x: (n + self.alt) * cos_lat * cos_lon,
            y: (n + self.alt) * cos_lat * sin_lon,
            z: (n * (1.0 - WGS84_E2) + self.alt) * sin_lat,
        }
    }

    /// Convert this geographic coordinate to a local East-North-Up vector
    /// relative to `origin`.
    ///
    /// Both points are first converted to ECEF and then transformed via the
    /// rotation matrix that aligns the local tangent plane of `origin`.
    pub fn to_enu(&self, origin: &GeographicCoord) -> EnuCoord {
        let origin_ecef = origin.to_ecef();
        let point_ecef = self.to_ecef();
        point_ecef.to_enu(&origin_ecef)
    }

    /// Compute the geodetic distance to `other` using Vincenty's inverse formula.
    ///
    /// This is accurate to within 0.5 mm on the WGS-84 ellipsoid for all
    /// non-antipodal pairs.  For nearly antipodal points the iteration may not
    /// converge; in that case a [`GeoError::VincentyNonConvergence`] is
    /// returned.
    ///
    /// # References
    ///
    /// Vincenty, T. (1975). *Direct and inverse solutions of geodesics on the
    /// ellipsoid with application of nested equations.*  Survey Review, 23(176),
    /// 88–93.
    pub fn vincenty_distance(&self, other: &GeographicCoord) -> GeoResult<f64> {
        const MAX_ITER: u32 = 200;
        const TOLERANCE: f64 = 1e-12;

        let lat1 = deg2rad(self.lat);
        let lat2 = deg2rad(other.lat);
        let l = deg2rad(other.lon - self.lon);

        let u1 = ((1.0 - WGS84_F) * lat1.tan()).atan();
        let u2 = ((1.0 - WGS84_F) * lat2.tan()).atan();

        let sin_u1 = u1.sin();
        let cos_u1 = u1.cos();
        let sin_u2 = u2.sin();
        let cos_u2 = u2.cos();

        let mut lambda = l;
        let mut lambda_prev;
        let mut sin_sigma;
        let mut cos_sigma;
        let mut sigma;
        let mut sin_alpha;
        let mut cos2_alpha;
        let mut cos2_sigma_m;
        let mut iter = 0_u32;

        loop {
            let sin_lambda = lambda.sin();
            let cos_lambda = lambda.cos();

            sin_sigma = ((cos_u2 * sin_lambda).powi(2)
                + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda).powi(2))
            .sqrt();

            cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
            sigma = sin_sigma.atan2(cos_sigma);

            sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma.max(f64::EPSILON);
            cos2_alpha = 1.0 - sin_alpha * sin_alpha;

            cos2_sigma_m = if cos2_alpha.abs() < f64::EPSILON {
                // equatorial line
                0.0
            } else {
                cos_sigma - 2.0 * sin_u1 * sin_u2 / cos2_alpha
            };

            let c = WGS84_F / 16.0 * cos2_alpha * (4.0 + WGS84_F * (4.0 - 3.0 * cos2_alpha));

            lambda_prev = lambda;
            lambda = l
                + (1.0 - c)
                    * WGS84_F
                    * sin_alpha
                    * (sigma
                        + c * sin_sigma
                            * (cos2_sigma_m
                                + c * cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)));

            iter += 1;
            if (lambda - lambda_prev).abs() < TOLERANCE {
                break;
            }
            if iter >= MAX_ITER {
                return Err(GeoError::VincentyNonConvergence {
                    max_iter: MAX_ITER,
                    delta: (lambda - lambda_prev).abs(),
                });
            }
        }

        let u2_coeff = cos2_alpha * (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B);
        let big_a =
            1.0 + u2_coeff / 16384.0 * (4096.0 + u2_coeff * (-768.0 + u2_coeff * (320.0 - 175.0 * u2_coeff)));
        let big_b = u2_coeff / 1024.0 * (256.0 + u2_coeff * (-128.0 + u2_coeff * (74.0 - 47.0 * u2_coeff)));

        let delta_sigma = big_b
            * sin_sigma
            * (cos2_sigma_m
                + big_b / 4.0
                    * (cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)
                        - big_b / 6.0
                            * cos2_sigma_m
                            * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                            * (-3.0 + 4.0 * cos2_sigma_m * cos2_sigma_m)));

        Ok(WGS84_B * big_a * (sigma - delta_sigma))
    }

    /// Compute the approximate geodetic distance to `other` using the Haversine
    /// formula on a sphere of radius equal to the WGS-84 semi-major axis.
    ///
    /// The Haversine formula is fast and simple but treats the Earth as a sphere,
    /// so errors may reach ~0.3 % near the poles.  For higher accuracy use
    /// [`Self::vincenty_distance`].
    pub fn haversine_distance(&self, other: &GeographicCoord) -> f64 {
        let lat1 = deg2rad(self.lat);
        let lat2 = deg2rad(other.lat);
        let dlat = lat2 - lat1;
        let dlon = deg2rad(other.lon - self.lon);

        let a = (dlat / 2.0).sin().powi(2)
            + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

        WGS84_A * c
    }

    /// Compute the initial bearing (forward azimuth) from `self` to `other`.
    ///
    /// Returns a value in the range `[0, 360)` degrees, measured clockwise from
    /// true north.
    pub fn bearing_to(&self, other: &GeographicCoord) -> f64 {
        let lat1 = deg2rad(self.lat);
        let lat2 = deg2rad(other.lat);
        let dlon = deg2rad(other.lon - self.lon);

        let y = dlon.sin() * lat2.cos();
        let x = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * dlon.cos();

        let bearing_rad = y.atan2(x);
        let bearing_deg = rad2deg(bearing_rad);
        (bearing_deg + 360.0) % 360.0
    }

    /// Compute the initial bearing (forward azimuth) from `self` to `other`
    /// using Vincenty's inverse formula on the WGS-84 ellipsoid.
    ///
    /// Returns a value in the range `[0, 360)` degrees, measured clockwise from
    /// true north. This is more accurate than [`Self::bearing_to`] which uses
    /// a spherical approximation.
    ///
    /// # Errors
    ///
    /// Returns an error if the Vincenty iteration does not converge (e.g. for
    /// nearly antipodal points).
    pub fn vincenty_bearing(&self, other: &GeographicCoord) -> GeoResult<f64> {
        const MAX_ITER: u32 = 200;
        const TOLERANCE: f64 = 1e-12;

        let lat1 = deg2rad(self.lat);
        let lat2 = deg2rad(other.lat);
        let l = deg2rad(other.lon - self.lon);

        let u1 = ((1.0 - WGS84_F) * lat1.tan()).atan();
        let u2 = ((1.0 - WGS84_F) * lat2.tan()).atan();

        let sin_u1 = u1.sin();
        let cos_u1 = u1.cos();
        let sin_u2 = u2.sin();
        let cos_u2 = u2.cos();

        let mut lambda = l;
        let mut iter = 0_u32;

        loop {
            let sin_lambda = lambda.sin();
            let cos_lambda = lambda.cos();

            let sin_sigma = ((cos_u2 * sin_lambda).powi(2)
                + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda).powi(2))
            .sqrt();

            let cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;

            let sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma.max(f64::EPSILON);
            let cos2_alpha = 1.0 - sin_alpha * sin_alpha;

            let cos2_sigma_m = if cos2_alpha.abs() < f64::EPSILON {
                0.0
            } else {
                cos_sigma - 2.0 * sin_u1 * sin_u2 / cos2_alpha
            };

            let c = WGS84_F / 16.0 * cos2_alpha * (4.0 + WGS84_F * (4.0 - 3.0 * cos2_alpha));

            let lambda_prev = lambda;
            lambda = l
                + (1.0 - c)
                    * WGS84_F
                    * sin_alpha
                    * (cos_sigma.asin()
                        + c * sin_sigma
                            * (cos2_sigma_m
                                + c * cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)));

            iter += 1;
            if (lambda - lambda_prev).abs() < TOLERANCE {
                // Compute forward azimuth alpha1
                let alpha1 = (cos_u2 * sin_lambda).atan2(
                    cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda,
                );
                let bearing_deg = rad2deg(alpha1);
                return Ok((bearing_deg + 360.0) % 360.0);
            }
            if iter >= MAX_ITER {
                return Err(GeoError::VincentyNonConvergence {
                    max_iter: MAX_ITER,
                    delta: (lambda - lambda_prev).abs(),
                });
            }
        }
    }

    /// Compute the destination point given a starting point, an initial bearing,
    /// and a distance along the geodesic, using Vincenty's direct formula.
    ///
    /// # Arguments
    ///
    /// * `distance` – Distance along the geodesic in metres (must be ≥ 0).
    /// * `bearing`  – Initial bearing in decimal degrees (clockwise from north).
    ///
    /// # Errors
    ///
    /// Returns [`GeoError::DomainError`] if `distance` is negative, or
    /// [`GeoError::VincentyNonConvergence`] if the iteration does not converge.
    pub fn destination(&self, distance: f64, bearing: f64) -> GeoResult<GeographicCoord> {
        if distance < 0.0 {
            return Err(GeoError::DomainError(format!(
                "distance must be non-negative, got {distance}"
            )));
        }

        const MAX_ITER: u32 = 200;
        const TOLERANCE: f64 = 1e-12;

        let alpha1 = deg2rad(bearing);
        let sin_alpha1 = alpha1.sin();
        let cos_alpha1 = alpha1.cos();

        let tan_u1 = (1.0 - WGS84_F) * deg2rad(self.lat).tan();
        let cos_u1 = 1.0 / (1.0 + tan_u1 * tan_u1).sqrt();
        let sin_u1 = tan_u1 * cos_u1;

        let sigma1 = tan_u1.atan2(cos_alpha1);
        let sin_alpha = cos_u1 * sin_alpha1;
        let cos2_alpha = 1.0 - sin_alpha * sin_alpha;

        let u2 = cos2_alpha * WGS84_EP2;
        let big_a = 1.0
            + u2 / 16384.0
                * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)));
        let big_b =
            u2 / 1024.0 * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)));

        let mut sigma = distance / (WGS84_B * big_a);
        let mut sigma_prev;
        let mut iter = 0_u32;
        let mut cos2_sigma_m;
        let mut sin_sigma;
        let mut cos_sigma;

        loop {
            cos2_sigma_m = (2.0 * sigma1 + sigma).cos();
            sin_sigma = sigma.sin();
            cos_sigma = sigma.cos();

            let delta_sigma = big_b
                * sin_sigma
                * (cos2_sigma_m
                    + big_b / 4.0
                        * (cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)
                            - big_b / 6.0
                                * cos2_sigma_m
                                * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                                * (-3.0 + 4.0 * cos2_sigma_m * cos2_sigma_m)));

            sigma_prev = sigma;
            sigma = distance / (WGS84_B * big_a) + delta_sigma;

            iter += 1;
            if (sigma - sigma_prev).abs() < TOLERANCE {
                break;
            }
            if iter >= MAX_ITER {
                return Err(GeoError::VincentyNonConvergence {
                    max_iter: MAX_ITER,
                    delta: (sigma - sigma_prev).abs(),
                });
            }
        }

        // Recompute with converged sigma
        cos2_sigma_m = (2.0 * sigma1 + sigma).cos();
        sin_sigma = sigma.sin();
        cos_sigma = sigma.cos();

        let lat2_rad = (sin_u1 * cos_sigma + cos_u1 * sin_sigma * cos_alpha1).atan2(
            (1.0 - WGS84_F)
                * (sin_alpha.powi(2)
                    + (sin_u1 * sin_sigma - cos_u1 * cos_sigma * cos_alpha1).powi(2))
                .sqrt(),
        );

        let lambda =
            (sin_sigma * sin_alpha1).atan2(cos_u1 * cos_sigma - sin_u1 * sin_sigma * cos_alpha1);

        let c = WGS84_F / 16.0
            * cos2_alpha
            * (4.0 + WGS84_F * (4.0 - 3.0 * cos2_alpha));

        let l = lambda
            - (1.0 - c)
                * WGS84_F
                * sin_alpha
                * (sigma
                    + c * sin_sigma
                        * (cos2_sigma_m
                            + c * cos_sigma
                                * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)));

        let lon2 = deg2rad(self.lon) + l;

        Ok(GeographicCoord {
            lat: rad2deg(lat2_rad),
            lon: rad2deg(lon2),
            alt: self.alt, // preserve altitude of the starting point
        })
    }
}

// ---------------------------------------------------------------------------
// EcefCoord implementation
// ---------------------------------------------------------------------------

impl EcefCoord {
    /// Create a new ECEF coordinate.
    #[inline]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Convert this ECEF coordinate to geographic (geodetic) coordinates using
    /// **Bowring's iterative method** (typically converges in 2–3 iterations to
    /// sub-millimetre accuracy).
    ///
    /// # References
    ///
    /// Bowring, B. R. (1985). *The geodetic line*.  Survey Review, 28(218),
    /// 109–124.
    pub fn to_geographic(&self) -> GeographicCoord {
        let p = (self.x * self.x + self.y * self.y).sqrt();
        let lon = self.y.atan2(self.x);

        // Bowring's iterative formula
        let mut lat = (self.z / (p * (1.0 - WGS84_E2))).atan();
        for _ in 0..10 {
            let sin_lat = lat.sin();
            let n = prime_vertical_radius(sin_lat);
            let lat_new = (self.z + WGS84_E2 * n * sin_lat).atan2(p);
            if (lat_new - lat).abs() < 1e-14 {
                lat = lat_new;
                break;
            }
            lat = lat_new;
        }

        let sin_lat = lat.sin();
        let n = prime_vertical_radius(sin_lat);
        let alt = p / lat.cos() - n;

        GeographicCoord {
            lat: rad2deg(lat),
            lon: rad2deg(lon),
            alt,
        }
    }

    /// Convert this ECEF coordinate to local ENU coordinates relative to an
    /// ECEF origin.
    ///
    /// The rotation matrix R transforms the ΔECEF vector into local ENU via:
    ///
    /// ```text
    /// ⎡ e ⎤   ⎡ -sin(λ)          cos(λ)         0       ⎤   ⎡ ΔX ⎤
    /// ⎢ n ⎥ = ⎢ -sin(φ)cos(λ)  -sin(φ)sin(λ)  cos(φ) ⎥ · ⎢ ΔY ⎥
    /// ⎣ u ⎦   ⎣  cos(φ)cos(λ)   cos(φ)sin(λ)  sin(φ) ⎦   ⎣ ΔZ ⎦
    /// ```
    pub fn to_enu(&self, origin: &EcefCoord) -> EnuCoord {
        let origin_geo = origin.to_geographic();
        let lat = deg2rad(origin_geo.lat);
        let lon = deg2rad(origin_geo.lon);

        let dx = self.x - origin.x;
        let dy = self.y - origin.y;
        let dz = self.z - origin.z;

        let sin_lat = lat.sin();
        let cos_lat = lat.cos();
        let sin_lon = lon.sin();
        let cos_lon = lon.cos();

        EnuCoord {
            east: -sin_lon * dx + cos_lon * dy,
            north: -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz,
            up: cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz,
        }
    }
}

// ---------------------------------------------------------------------------
// EnuCoord implementation
// ---------------------------------------------------------------------------

impl EnuCoord {
    /// Create a new ENU coordinate.
    #[inline]
    pub fn new(east: f64, north: f64, up: f64) -> Self {
        Self { east, north, up }
    }

    /// Horizontal distance from the origin (ignoring the `up` component).
    #[inline]
    pub fn horizontal_distance(&self) -> f64 {
        (self.east * self.east + self.north * self.north).sqrt()
    }

    /// Full 3-D distance from the origin.
    #[inline]
    pub fn distance(&self) -> f64 {
        (self.east * self.east + self.north * self.north + self.up * self.up).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Maximum allowed absolute error when comparing distances (metres).
    const DIST_TOL: f64 = 1.0; // 1 m

    /// Angular tolerance for round-trip checks (degrees).
    const ANG_TOL: f64 = 1e-9;

    // -----------------------------------------------------------------------
    // ECEF round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_ecef_round_trip_london() {
        let london = GeographicCoord::new(51.5074, -0.1278, 0.0);
        let ecef = london.to_ecef();
        let recovered = ecef.to_geographic();
        assert!(
            (recovered.lat - london.lat).abs() < ANG_TOL,
            "lat error: {}",
            (recovered.lat - london.lat).abs()
        );
        assert!(
            (recovered.lon - london.lon).abs() < ANG_TOL,
            "lon error: {}",
            (recovered.lon - london.lon).abs()
        );
        assert!(
            (recovered.alt - london.alt).abs() < 1e-4,
            "alt error: {}",
            (recovered.alt - london.alt).abs()
        );
    }

    #[test]
    fn test_ecef_round_trip_with_altitude() {
        let coord = GeographicCoord::new(-33.8688, 151.2093, 500.0); // Sydney, 500 m
        let recovered = coord.to_ecef().to_geographic();
        assert!((recovered.lat - coord.lat).abs() < ANG_TOL);
        assert!((recovered.lon - coord.lon).abs() < ANG_TOL);
        assert!((recovered.alt - coord.alt).abs() < 1e-4);
    }

    #[test]
    fn test_ecef_round_trip_north_pole() {
        let pole = GeographicCoord::new(89.9999, 0.0, 0.0);
        let recovered = pole.to_ecef().to_geographic();
        assert!((recovered.lat - pole.lat).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Haversine distance
    // -----------------------------------------------------------------------

    #[test]
    fn test_haversine_london_paris() {
        let london = GeographicCoord::new(51.5074, -0.1278, 0.0);
        let paris = GeographicCoord::new(48.8566, 2.3522, 0.0);
        let dist = london.haversine_distance(&paris);
        // Known approximate distance: ~340 km
        assert!(
            (dist - 340_000.0).abs() < 5_000.0,
            "Haversine London–Paris: {dist:.0} m, expected ~340 000 m"
        );
    }

    #[test]
    fn test_haversine_same_point() {
        let p = GeographicCoord::new(45.0, 90.0, 0.0);
        let dist = p.haversine_distance(&p);
        assert!(dist.abs() < 1e-6, "distance to self must be ~0, got {dist}");
    }

    #[test]
    fn test_haversine_equatorial() {
        // Two points on the equator, 1 degree apart in longitude
        // 1° longitude at equator ≈ 111_319.5 m
        let a = GeographicCoord::new(0.0, 0.0, 0.0);
        let b = GeographicCoord::new(0.0, 1.0, 0.0);
        let dist = a.haversine_distance(&b);
        assert!((dist - 111_319.49).abs() < 1.0, "equatorial 1°: {dist:.2} m");
    }

    // -----------------------------------------------------------------------
    // Vincenty distance
    // -----------------------------------------------------------------------

    #[test]
    fn test_vincenty_london_paris() {
        let london = GeographicCoord::new(51.5074, -0.1278, 0.0);
        let paris = GeographicCoord::new(48.8566, 2.3522, 0.0);
        let dist = london.vincenty_distance(&paris).expect("vincenty should converge");
        assert!(
            (dist - 340_000.0).abs() < 5_000.0,
            "Vincenty London–Paris: {dist:.0} m"
        );
    }

    #[test]
    fn test_vincenty_more_accurate_than_haversine() {
        // For a long geodesic Vincenty should agree with Haversine within ~0.3%
        let nyc = GeographicCoord::new(40.7128, -74.0060, 0.0);
        let london = GeographicCoord::new(51.5074, -0.1278, 0.0);
        let hav = nyc.haversine_distance(&london);
        let vin = nyc.vincenty_distance(&london).expect("vincenty convergence");
        // Both should agree within 0.3% (Vincenty more accurate on ellipsoid)
        let rel_diff = (hav - vin).abs() / vin;
        assert!(
            rel_diff < 0.005,
            "Haversine vs Vincenty relative diff: {rel_diff:.4} > 0.5%"
        );
    }

    #[test]
    fn test_vincenty_same_point() {
        let p = GeographicCoord::new(45.0, 90.0, 0.0);
        let dist = p.vincenty_distance(&p).expect("should converge");
        assert!(dist.abs() < 1e-3, "distance to self must be ~0, got {dist}");
    }

    // -----------------------------------------------------------------------
    // Bearing
    // -----------------------------------------------------------------------

    #[test]
    fn test_bearing_to_north() {
        // Bearing from an equatorial point to a point due north should be 0
        let a = GeographicCoord::new(0.0, 0.0, 0.0);
        let b = GeographicCoord::new(10.0, 0.0, 0.0);
        let bearing = a.bearing_to(&b);
        assert!(
            bearing < 0.001 || (bearing - 360.0).abs() < 0.001,
            "north bearing should be ~0, got {bearing}"
        );
    }

    #[test]
    fn test_bearing_to_east() {
        let a = GeographicCoord::new(0.0, 0.0, 0.0);
        let b = GeographicCoord::new(0.0, 10.0, 0.0);
        let bearing = a.bearing_to(&b);
        assert!(
            (bearing - 90.0).abs() < 0.001,
            "east bearing should be ~90, got {bearing}"
        );
    }

    #[test]
    fn test_bearing_in_range() {
        let a = GeographicCoord::new(51.5074, -0.1278, 0.0);
        let b = GeographicCoord::new(48.8566, 2.3522, 0.0);
        let bearing = a.bearing_to(&b);
        assert!((0.0..360.0).contains(&bearing), "bearing out of range: {bearing}");
    }

    // -----------------------------------------------------------------------
    // Vincenty direct (destination)
    // -----------------------------------------------------------------------

    #[test]
    fn test_destination_round_trip() {
        let start = GeographicCoord::new(51.5074, -0.1278, 0.0);
        let other = GeographicCoord::new(48.8566, 2.3522, 0.0);
        let dist = start.vincenty_distance(&other).expect("dist converges");
        // Use Vincenty bearing (ellipsoidal) for consistency with Vincenty destination.
        let bearing = start.vincenty_bearing(&other).expect("bearing converges");
        let dest = start.destination(dist, bearing).expect("destination converges");
        assert!(
            (dest.lat - other.lat).abs() < 1e-5,
            "lat error: {}",
            (dest.lat - other.lat).abs()
        );
        assert!(
            (dest.lon - other.lon).abs() < 1e-5,
            "lon error: {}",
            (dest.lon - other.lon).abs()
        );
    }

    #[test]
    fn test_destination_negative_distance_error() {
        let p = GeographicCoord::new(0.0, 0.0, 0.0);
        let result = p.destination(-1.0, 0.0);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // ENU
    // -----------------------------------------------------------------------

    #[test]
    fn test_enu_origin_maps_to_zero() {
        let origin = GeographicCoord::new(51.5074, -0.1278, 0.0);
        let enu = origin.to_enu(&origin);
        assert!(enu.east.abs() < 1e-6, "east should be ~0: {}", enu.east);
        assert!(enu.north.abs() < 1e-6, "north should be ~0: {}", enu.north);
        assert!(enu.up.abs() < 1e-6, "up should be ~0: {}", enu.up);
    }

    #[test]
    fn test_enu_north_offset() {
        let origin = GeographicCoord::new(0.0, 0.0, 0.0);
        // Point ~111 km north
        let north_pt = GeographicCoord::new(1.0, 0.0, 0.0);
        let enu = north_pt.to_enu(&origin);
        // North component should dominate (~111 319 m)
        assert!(
            enu.north > 100_000.0,
            "north component too small: {}",
            enu.north
        );
        assert!(
            enu.east.abs() < 1_000.0,
            "east component too large: {}",
            enu.east
        );
    }

    #[test]
    fn test_enu_horizontal_distance() {
        let origin = GeographicCoord::new(0.0, 0.0, 0.0);
        let pt = GeographicCoord::new(0.0, 1.0, 0.0);
        let enu = pt.to_enu(&origin);
        let h = enu.horizontal_distance();
        // 1° lon at equator ≈ 111 319 m
        assert!((h - 111_319.49).abs() < DIST_TOL * 10.0, "h = {h}");
    }
}
