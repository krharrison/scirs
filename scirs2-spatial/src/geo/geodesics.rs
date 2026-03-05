//! Geodesic calculations on the WGS84 ellipsoid.
//!
//! This module extends the basic haversine / bearing functionality already
//! present in [`distances`] with:
//!
//! - **Vincenty direct / inverse** (ellipsoidal, millimetre accuracy)
//! - **Rhumb lines** (constant-bearing loxodromic paths)
//! - `initial_bearing` / `final_bearing`
//! - `midpoint` — geodesic midpoint on a sphere
//! - `destination_point` — offset a point by bearing + distance
//!
//! # Design choices
//!
//! We re-export and supplement the haversine / vincenty functions from
//! [`distances`] so that callers can use this single module for all
//! geodesic operations.

use crate::error::{SpatialError, SpatialResult};
use std::f64::consts::PI;

// ── WGS84 constants ──────────────────────────────────────────────────────────
/// WGS84 semi-major axis (m)
pub const WGS84_A: f64 = 6_378_137.0;
/// WGS84 flattening
pub const WGS84_F: f64 = 1.0 / 298.257_223_563;
/// WGS84 semi-minor axis (m)
pub const WGS84_B: f64 = WGS84_A * (1.0 - WGS84_F);
/// Mean Earth radius used for spherical approximations (m)
pub const EARTH_R: f64 = 6_371_008.8;

#[inline]
fn to_rad(d: f64) -> f64 {
    d * PI / 180.0
}

#[inline]
fn to_deg(r: f64) -> f64 {
    r * 180.0 / PI
}

/// Wrap an angle in radians to [−π, π].
#[inline]
fn wrap_pi(a: f64) -> f64 {
    let mut v = a % (2.0 * PI);
    if v > PI {
        v -= 2.0 * PI;
    } else if v < -PI {
        v += 2.0 * PI;
    }
    v
}

/// Wrap an angle in degrees to [0, 360).
#[inline]
fn wrap_360(a: f64) -> f64 {
    ((a % 360.0) + 360.0) % 360.0
}

// ═══════════════════════════════════════════════════════════════════════════════
//  1. Vincenty inverse (distance + azimuths)
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of the Vincenty inverse problem.
#[derive(Debug, Clone, Copy)]
pub struct VincentyResult {
    /// Geodesic distance in metres.
    pub distance: f64,
    /// Initial (forward) azimuth in degrees [0, 360).
    pub initial_bearing: f64,
    /// Final (reverse) azimuth in degrees [0, 360).
    pub final_bearing: f64,
}

/// Solve the **Vincenty inverse problem**: compute geodesic distance and
/// azimuths between two points on the WGS84 ellipsoid.
///
/// # Arguments
/// * `lat1`, `lon1` – first  point in degrees
/// * `lat2`, `lon2` – second point in degrees
///
/// # Returns
/// [`VincentyResult`] or `Err` if the iteration fails to converge (near-antipodal
/// points).
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geo::geodesics::vincenty_inverse;
///
/// let r = vincenty_inverse(51.5074, -0.1278, 48.8566, 2.3522).unwrap();
/// // London ↔ Paris ≈ 340 km
/// assert!((r.distance - 340_000.0).abs() < 5_000.0);
/// ```
pub fn vincenty_inverse(
    lat1: f64,
    lon1: f64,
    lat2: f64,
    lon2: f64,
) -> SpatialResult<VincentyResult> {
    let phi1 = to_rad(lat1);
    let phi2 = to_rad(lat2);
    let l = to_rad(lon2 - lon1);

    let u1 = ((1.0 - WGS84_F) * phi1.tan()).atan();
    let u2 = ((1.0 - WGS84_F) * phi2.tan()).atan();

    let sin_u1 = u1.sin();
    let cos_u1 = u1.cos();
    let sin_u2 = u2.sin();
    let cos_u2 = u2.cos();

    let mut lam = l;
    let mut lam_prev;
    let mut sin_sigma = 0.0_f64;
    let mut cos_sigma = 0.0_f64;
    let mut sigma = 0.0_f64;
    let mut sin_alpha = 0.0_f64;
    let mut cos2_alpha = 0.0_f64;
    let mut cos_2sigma_m = 0.0_f64;

    let mut converged = false;
    for _ in 0..1000 {
        sin_sigma = ((cos_u2 * lam.sin()).powi(2)
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * lam.cos()).powi(2))
        .sqrt();

        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * lam.cos();
        sigma = sin_sigma.atan2(cos_sigma);

        sin_alpha = if sin_sigma.abs() < 1e-15 {
            0.0
        } else {
            cos_u1 * cos_u2 * lam.sin() / sin_sigma
        };
        cos2_alpha = 1.0 - sin_alpha * sin_alpha;

        cos_2sigma_m = if cos2_alpha.abs() < 1e-15 {
            0.0 // equatorial line
        } else {
            cos_sigma - 2.0 * sin_u1 * sin_u2 / cos2_alpha
        };

        let c = WGS84_F / 16.0 * cos2_alpha * (4.0 + WGS84_F * (4.0 - 3.0 * cos2_alpha));

        lam_prev = lam;
        lam = l
            + (1.0 - c)
                * WGS84_F
                * sin_alpha
                * (sigma
                    + c * sin_sigma
                        * (cos_2sigma_m
                            + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)));

        if (lam - lam_prev).abs() < 1e-12 {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(SpatialError::ComputationError(
            "Vincenty inverse failed to converge (near-antipodal points)".to_string(),
        ));
    }

    let u2_sq = cos2_alpha * (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B);
    let k1 = ((1.0 + u2_sq).sqrt() - 1.0) / ((1.0 + u2_sq).sqrt() + 1.0);
    let a_coeff = (1.0 + k1 * k1 / 4.0) / (1.0 - k1);
    let b_coeff = k1 * (1.0 - 3.0 * k1 * k1 / 8.0);

    let delta_sigma = b_coeff
        * sin_sigma
        * (cos_2sigma_m
            + b_coeff / 4.0
                * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)
                    - b_coeff / 6.0
                        * cos_2sigma_m
                        * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                        * (-3.0 + 4.0 * cos_2sigma_m * cos_2sigma_m)));

    let s = WGS84_B * a_coeff * (sigma - delta_sigma);

    let az1 = to_deg(wrap_pi(
        (cos_u2 * lam.sin()).atan2(cos_u1 * sin_u2 - sin_u1 * cos_u2 * lam.cos()),
    ));
    let az2 = to_deg(wrap_pi(
        (cos_u1 * lam.sin()).atan2(-sin_u1 * cos_u2 + cos_u1 * sin_u2 * lam.cos()),
    ));

    Ok(VincentyResult {
        distance: s,
        initial_bearing: wrap_360(az1),
        final_bearing: wrap_360(az2 + 180.0),
    })
}

/// Solve the **Vincenty direct problem**: given a start point, initial azimuth
/// and geodesic distance, find the destination.
///
/// # Arguments
/// * `lat1`, `lon1` – origin in degrees
/// * `bearing_deg`  – initial azimuth in degrees (clockwise from north)
/// * `distance`     – geodesic distance in metres
///
/// # Returns
/// `(lat2, lon2, final_bearing)` all in degrees.
pub fn vincenty_direct(
    lat1: f64,
    lon1: f64,
    bearing_deg: f64,
    distance: f64,
) -> SpatialResult<(f64, f64, f64)> {
    let phi1 = to_rad(lat1);
    let alpha1 = to_rad(bearing_deg);

    let u1 = ((1.0 - WGS84_F) * phi1.tan()).atan();
    let sigma1 = u1.tan().atan2(alpha1.cos());
    let sin_alpha = u1.cos() * alpha1.sin();
    let cos2_alpha = 1.0 - sin_alpha * sin_alpha;
    let u2_sq = cos2_alpha * (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B);

    let k1 = ((1.0 + u2_sq).sqrt() - 1.0) / ((1.0 + u2_sq).sqrt() + 1.0);
    let a_coeff = (1.0 + k1 * k1 / 4.0) / (1.0 - k1);
    let b_coeff = k1 * (1.0 - 3.0 * k1 * k1 / 8.0);

    let mut sigma = distance / (WGS84_B * a_coeff);

    let mut sigma_prev;
    for _ in 0..1000 {
        let cos_2sigma_m = (2.0 * sigma1 + sigma).cos();
        let delta_sigma = b_coeff
            * sigma.sin()
            * (cos_2sigma_m
                + b_coeff / 4.0
                    * (sigma.cos() * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)
                        - b_coeff / 6.0
                            * cos_2sigma_m
                            * (-3.0 + 4.0 * sigma.sin().powi(2))
                            * (-3.0 + 4.0 * cos_2sigma_m * cos_2sigma_m)));
        sigma_prev = sigma;
        sigma = distance / (WGS84_B * a_coeff) + delta_sigma;
        if (sigma - sigma_prev).abs() < 1e-12 {
            break;
        }
    }

    let cos_2sigma_m = (2.0 * sigma1 + sigma).cos();
    let sin_u1 = u1.sin();
    let cos_u1 = u1.cos();

    let lat2 = (sin_u1 * sigma.cos() + cos_u1 * sigma.sin() * alpha1.cos()).atan2(
        (1.0 - WGS84_F)
            * (sin_alpha.powi(2)
                + (sin_u1 * sigma.sin() - cos_u1 * sigma.cos() * alpha1.cos()).powi(2))
            .sqrt(),
    );

    let lam = (sigma.sin() * alpha1.sin()).atan2(
        cos_u1 * sigma.cos() - sin_u1 * sigma.sin() * alpha1.cos(),
    );

    let c = WGS84_F / 16.0 * cos2_alpha * (4.0 + WGS84_F * (4.0 - 3.0 * cos2_alpha));

    let lon2 = to_rad(lon1)
        + lam
        - (1.0 - c)
            * WGS84_F
            * sin_alpha
            * (sigma
                + c * sigma.sin()
                    * (cos_2sigma_m
                        + c * sigma.cos()
                            * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)));

    let az2 = alpha1.sin().atan2(-sin_u1 * sigma.sin() + cos_u1 * sigma.cos() * alpha1.cos());

    Ok((
        to_deg(lat2),
        to_deg(lon2),
        wrap_360(to_deg(az2) + 180.0),
    ))
}

// ═══════════════════════════════════════════════════════════════════════════════
//  2. Haversine (spherical great-circle)
// ═══════════════════════════════════════════════════════════════════════════════

/// Fast spherical great-circle distance (Haversine formula).
///
/// # Arguments
/// * `lat1`, `lon1` – first  point in degrees
/// * `lat2`, `lon2` – second point in degrees
///
/// # Returns
/// Distance in metres (spherical approximation; error < 0.5 %).
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geo::geodesics::haversine;
/// let d = haversine(51.5074, -0.1278, 48.8566, 2.3522);
/// assert!((d - 343_000.0).abs() < 5_000.0);
/// ```
pub fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let phi1 = to_rad(lat1);
    let phi2 = to_rad(lat2);
    let dphi = to_rad(lat2 - lat1);
    let dlam = to_rad(lon2 - lon1);
    let a = (dphi / 2.0).sin().powi(2) + phi1.cos() * phi2.cos() * (dlam / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    EARTH_R * c
}

// ═══════════════════════════════════════════════════════════════════════════════
//  3. Rhumb lines (loxodromes)
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the **rhumb-line bearing** from point 1 to point 2.
///
/// A rhumb line is a path that crosses every meridian at the same angle
/// (constant compass bearing).
///
/// # Returns
/// Bearing in degrees [0, 360).
pub fn rhumb_bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dphi = to_rad(lat2) - to_rad(lat1);
    let dlam = to_rad(lon2 - lon1);

    // Isometric latitude difference Δψ = ψ₂ − ψ₁.
    // The rhumb bearing = atan2(Δλ, Δψ).
    // When Δψ → 0 (equatorial track), atan2(Δλ, 0) = ±90° (due east/west) as expected.
    let dpsi = {
        let psi1 = ((PI / 4.0 + to_rad(lat1) / 2.0).tan()).ln();
        let psi2 = ((PI / 4.0 + to_rad(lat2) / 2.0).tan()).ln();
        psi2 - psi1
    };

    // Correct for crossing the anti-meridian
    let dlam_adj = if dlam.abs() > PI {
        if dlam > 0.0 {
            dlam - 2.0 * PI
        } else {
            dlam + 2.0 * PI
        }
    } else {
        dlam
    };

    wrap_360(to_deg(dlam_adj.atan2(dpsi)))
}

/// Compute the **rhumb-line distance** between two points.
///
/// # Returns
/// Distance in metres.
pub fn rhumb_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dphi = to_rad(lat2 - lat1);
    let dlam = to_rad(lon2 - lon1);

    let q = if dphi.abs() < 1e-12 {
        to_rad(lat1).cos()
    } else {
        let psi1 = ((PI / 4.0 + to_rad(lat1) / 2.0).tan()).ln();
        let psi2 = ((PI / 4.0 + to_rad(lat2) / 2.0).tan()).ln();
        dphi / (psi2 - psi1)
    };

    let dlam_adj = if dlam.abs() > PI {
        if dlam > 0.0 {
            dlam - 2.0 * PI
        } else {
            dlam + 2.0 * PI
        }
    } else {
        dlam
    };

    EARTH_R * (dphi.powi(2) + (q * dlam_adj).powi(2)).sqrt()
}

/// Compute the destination point along a **rhumb line**.
///
/// # Arguments
/// * `lat`, `lon`   – origin in degrees
/// * `bearing_deg`  – constant bearing in degrees
/// * `distance`     – distance in metres
///
/// # Returns
/// `(lat2, lon2)` in degrees.
pub fn rhumb_destination(lat: f64, lon: f64, bearing_deg: f64, distance: f64) -> (f64, f64) {
    let phi1 = to_rad(lat);
    let lam1 = to_rad(lon);
    let theta = to_rad(bearing_deg);

    let delta = distance / EARTH_R;
    let dphi = delta * theta.cos();
    let phi2 = phi1 + dphi;

    // Limit to poles
    let phi2_clamped = phi2.clamp(-PI / 2.0 + 1e-10, PI / 2.0 - 1e-10);

    let dpsi = if dphi.abs() < 1e-12 {
        phi1.cos()
    } else {
        let psi1 = ((PI / 4.0 + phi1 / 2.0).tan()).ln();
        let psi2 = ((PI / 4.0 + phi2_clamped / 2.0).tan()).ln();
        if (psi2 - psi1).abs() < 1e-12 {
            phi1.cos()
        } else {
            dphi / (psi2 - psi1)
        }
    };

    let dlam = delta * theta.sin() / dpsi;
    let lam2 = ((lam1 + dlam + PI) % (2.0 * PI)) - PI;

    (to_deg(phi2_clamped), to_deg(lam2))
}

// ═══════════════════════════════════════════════════════════════════════════════
//  4. Forward / reverse azimuth helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Initial (forward) azimuth from point 1 to point 2 on a sphere, in degrees.
///
/// Equivalent to the bearing at the departure point.
///
/// # Returns
/// Angle in degrees [0, 360), clockwise from north.
pub fn initial_bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let phi1 = to_rad(lat1);
    let phi2 = to_rad(lat2);
    let dlam = to_rad(lon2 - lon1);

    let y = dlam.sin() * phi2.cos();
    let x = phi1.cos() * phi2.sin() - phi1.sin() * phi2.cos() * dlam.cos();
    wrap_360(to_deg(y.atan2(x)))
}

/// Final (reverse) azimuth at the **destination** of the great-circle path
/// from point 1 to point 2.
///
/// # Returns
/// Angle in degrees [0, 360), clockwise from north.
pub fn final_bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    // The final bearing is the bearing at the destination looking back toward the origin.
    // This equals the initial bearing from destination to origin.
    wrap_360(initial_bearing(lat2, lon2, lat1, lon1))
}

// ═══════════════════════════════════════════════════════════════════════════════
//  5. Geodesic midpoint
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the **geodesic midpoint** of two points (spherical).
///
/// The midpoint lies on the great-circle arc halfway between the two points.
///
/// # Returns
/// `(lat, lon)` in degrees.
pub fn midpoint(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> (f64, f64) {
    let phi1 = to_rad(lat1);
    let phi2 = to_rad(lat2);
    let dlam = to_rad(lon2 - lon1);
    let lam1 = to_rad(lon1);

    let bx = phi2.cos() * dlam.cos();
    let by = phi2.cos() * dlam.sin();

    let phi_m = (phi1.sin() + phi2.sin()).atan2(((phi1.cos() + bx).powi(2) + by * by).sqrt());
    let lam_m = lam1 + by.atan2(phi1.cos() + bx);

    (to_deg(phi_m), to_deg(lam_m))
}

// ═══════════════════════════════════════════════════════════════════════════════
//  6. Destination point (spherical)
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the destination point given start, bearing and distance.
///
/// Uses the spherical Earth model (suitable for navigation purposes).
///
/// # Arguments
/// * `lat`, `lon`   – origin in degrees
/// * `bearing_deg`  – initial azimuth clockwise from north, degrees
/// * `distance`     – arc distance in metres
///
/// # Returns
/// `(lat, lon)` of the destination in degrees.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geo::geodesics::destination_point;
///
/// let (lat2, lon2) = destination_point(51.5074, -0.1278, 108.0, 200_000.0);
/// println!("Destination: {lat2:.4}, {lon2:.4}");
/// ```
pub fn destination_point(lat: f64, lon: f64, bearing_deg: f64, distance: f64) -> (f64, f64) {
    let phi1 = to_rad(lat);
    let lam1 = to_rad(lon);
    let theta = to_rad(bearing_deg);
    let delta = distance / EARTH_R;

    let phi2 =
        (phi1.sin() * delta.cos() + phi1.cos() * delta.sin() * theta.cos()).asin();
    let lam2 = lam1
        + (theta.sin() * delta.sin() * phi1.cos())
            .atan2(delta.cos() - phi1.sin() * phi2.sin());

    (to_deg(phi2), to_deg(lam2))
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vincenty_inverse_london_paris() {
        let r = vincenty_inverse(51.5074, -0.1278, 48.8566, 2.3522).expect("vincenty");
        // Known ≈ 340 km
        assert!((r.distance - 340_000.0).abs() < 5_000.0, "dist={}", r.distance);
    }

    #[test]
    fn test_vincenty_inverse_same_point() {
        let r = vincenty_inverse(40.0, -74.0, 40.0, -74.0).expect("vincenty");
        assert!(r.distance < 1.0, "same point distance={}", r.distance);
    }

    #[test]
    fn test_vincenty_direct_roundtrip() {
        let (lat2, lon2, _) = vincenty_direct(51.5074, -0.1278, 108.0, 200_000.0).expect("direct");
        let r = vincenty_inverse(51.5074, -0.1278, lat2, lon2).expect("inverse");
        assert!((r.distance - 200_000.0).abs() < 1.0, "dist={}", r.distance);
    }

    #[test]
    fn test_haversine_known_distances() {
        // London ↔ Paris ≈ 343 km
        let d = haversine(51.5074, -0.1278, 48.8566, 2.3522);
        assert!((d - 343_000.0).abs() < 5_000.0, "d={d}");
        // Same point
        assert!(haversine(0.0, 0.0, 0.0, 0.0) < 1e-6);
    }

    #[test]
    fn test_initial_bearing_cardinal() {
        assert!((initial_bearing(0.0, 0.0, 1.0, 0.0)).abs() < 0.01);       // north
        assert!((initial_bearing(0.0, 0.0, 0.0, 1.0) - 90.0).abs() < 0.01); // east
        assert!((initial_bearing(1.0, 0.0, 0.0, 0.0) - 180.0).abs() < 0.01); // south
        assert!((initial_bearing(0.0, 1.0, 0.0, 0.0) - 270.0).abs() < 0.01); // west
    }

    #[test]
    fn test_final_bearing() {
        // After travelling north from equator, the reverse bearing is south
        let fb = final_bearing(0.0, 0.0, 10.0, 0.0);
        assert!((fb - 180.0).abs() < 0.1, "fb={fb}");
    }

    #[test]
    fn test_midpoint_equatorial() {
        // Midpoint between (0,0) and (0,20) should be (0,10)
        let (lat_m, lon_m) = midpoint(0.0, 0.0, 0.0, 20.0);
        assert!(lat_m.abs() < 1e-6, "lat={lat_m}");
        assert!((lon_m - 10.0).abs() < 1e-6, "lon={lon_m}");
    }

    #[test]
    fn test_midpoint_symmetric() {
        let (m1_lat, m1_lon) = midpoint(10.0, 20.0, 30.0, 40.0);
        let (m2_lat, m2_lon) = midpoint(30.0, 40.0, 10.0, 20.0);
        assert!((m1_lat - m2_lat).abs() < 1e-8, "lat symmetry");
        assert!((m1_lon - m2_lon).abs() < 1e-8, "lon symmetry");
    }

    #[test]
    fn test_destination_point_north() {
        let (lat2, lon2) = destination_point(0.0, 0.0, 0.0, 100_000.0);
        assert!(lat2 > 0.0, "lat2={lat2}");
        assert!(lon2.abs() < 0.001, "lon2={lon2}");
    }

    #[test]
    fn test_destination_roundtrip() {
        let (lat2, lon2) = destination_point(40.0, -74.0, 45.0, 500_000.0);
        let b = initial_bearing(40.0, -74.0, lat2, lon2);
        assert!((b - 45.0).abs() < 0.1, "bearing={b}");
        let d = haversine(40.0, -74.0, lat2, lon2);
        assert!((d - 500_000.0).abs() < 1000.0, "dist={d}");
    }

    #[test]
    fn test_rhumb_bearing_east() {
        let b = rhumb_bearing(0.0, 0.0, 0.0, 10.0);
        assert!((b - 90.0).abs() < 0.01, "b={b}");
    }

    #[test]
    fn test_rhumb_bearing_north() {
        let b = rhumb_bearing(0.0, 0.0, 10.0, 0.0);
        assert!(b < 0.01 || (b - 360.0).abs() < 0.01, "b={b}");
    }

    #[test]
    fn test_rhumb_distance_sanity() {
        // Roughly 10° longitude at equator ≈ 1_111 km
        let d = rhumb_distance(0.0, 0.0, 0.0, 10.0);
        assert!((d - 1_111_000.0).abs() < 20_000.0, "d={d}");
    }

    #[test]
    fn test_rhumb_destination_roundtrip() {
        let bearing = 45.0;
        let dist = 300_000.0;
        let (lat2, lon2) = rhumb_destination(10.0, 20.0, bearing, dist);
        let b_back = rhumb_bearing(10.0, 20.0, lat2, lon2);
        assert!((b_back - bearing).abs() < 0.01, "b_back={b_back}");
        let d_back = rhumb_distance(10.0, 20.0, lat2, lon2);
        assert!((d_back - dist).abs() < 100.0, "d_back={d_back}");
    }
}
