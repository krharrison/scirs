//! Coordinate reference systems and map projections.
//!
//! This module provides forward and inverse transforms for several widely-used
//! cartographic projections and datum conversions:
//!
//! - **Mercator** (conformal, cylindrical, Web Mercator variant)
//! - **Transverse Mercator / UTM** (extended API: zone forced by caller)
//! - **Mollweide** (equal-area pseudo-cylindrical)
//! - **Azimuthal Equidistant** (geodesic distances preserved from a centre point)
//! - **Datum shift** WGS84 → NAD83 (Helmert 7-parameter)
//!
//! All angular inputs/outputs are in **degrees**; Cartesian results are in **metres**.

use crate::error::{SpatialError, SpatialResult};
use std::f64::consts::PI;

// ── WGS84 ellipsoid constants (reuse from coordinates) ──────────────────────
/// WGS84 semi-major axis (m)
pub const A: f64 = 6_378_137.0;
/// WGS84 flattening
pub const F: f64 = 1.0 / 298.257_223_563;
/// WGS84 semi-minor axis (m)
pub const B: f64 = A * (1.0 - F);
/// WGS84 first eccentricity squared
pub const E2: f64 = 2.0 * F - F * F;
/// WGS84 eccentricity
pub const E: f64 = 0.081_819_190_842_622;

// ── Helper ────────────────────────────────────────────────────────────────────

#[inline]
fn deg_to_rad(d: f64) -> f64 {
    d * PI / 180.0
}

#[inline]
fn rad_to_deg(r: f64) -> f64 {
    r * 180.0 / PI
}

/// Prime vertical radius of curvature N(φ) for WGS84.
#[inline]
fn prime_vertical_radius(lat_rad: f64) -> f64 {
    A / (1.0 - E2 * lat_rad.sin().powi(2)).sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════════
//  1.  Geographic ↔ ECEF  (aliases so callers can import from projections too)
// ═══════════════════════════════════════════════════════════════════════════════

/// Convert geographic (WGS84) to ECEF Cartesian.
///
/// # Arguments
/// * `lat` – geodetic latitude in degrees (north positive)
/// * `lon` – longitude in degrees (east positive)
/// * `alt` – ellipsoidal height in metres
///
/// # Returns
/// `(x, y, z)` in metres.
pub fn geo_to_ecef(lat: f64, lon: f64, alt: f64) -> (f64, f64, f64) {
    let phi = deg_to_rad(lat);
    let lam = deg_to_rad(lon);
    let n = prime_vertical_radius(phi);
    let x = (n + alt) * phi.cos() * lam.cos();
    let y = (n + alt) * phi.cos() * lam.sin();
    let z = (n * (1.0 - E2) + alt) * phi.sin();
    (x, y, z)
}

/// Convert ECEF Cartesian to geographic (WGS84).
///
/// Uses the iterative Bowring method (typically 3 iterations suffice).
///
/// # Returns
/// `(lat_deg, lon_deg, alt_m)`
pub fn ecef_to_geo(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let lon = y.atan2(x);
    let p = (x * x + y * y).sqrt();

    // Initial latitude estimate (Bowring)
    let mut phi = (z / p * (1.0 - E2).recip()).atan();

    for _ in 0..10 {
        let n = prime_vertical_radius(phi);
        let phi_new = (z + E2 * n * phi.sin()).atan2(p);
        if (phi_new - phi).abs() < 1e-12 {
            phi = phi_new;
            break;
        }
        phi = phi_new;
    }

    let n = prime_vertical_radius(phi);
    let alt = if phi.cos().abs() > 1e-10 {
        p / phi.cos() - n
    } else {
        z / phi.sin() - n * (1.0 - E2)
    };

    (rad_to_deg(phi), rad_to_deg(lon), alt)
}

// ═══════════════════════════════════════════════════════════════════════════════
//  2.  Mercator projection  (spherical / conformal cylindrical)
// ═══════════════════════════════════════════════════════════════════════════════

/// Mercator (ellipsoidal) forward projection.
///
/// Uses the standard (true-scale) Mercator with the WGS84 ellipsoid.  The
/// false origin is at (lon₀, φ₀) = (0°, 0°).
///
/// # Arguments
/// * `lon` – longitude in degrees
/// * `lat` – geodetic latitude in degrees (must be outside ±85.051°)
///
/// # Returns
/// `(easting, northing)` in metres with respect to the equatorial origin at
/// the prime meridian.
pub fn mercator_forward(lon: f64, lat: f64) -> SpatialResult<(f64, f64)> {
    if lat.abs() >= 89.9 {
        return Err(SpatialError::ValueError(
            "Mercator is undefined at or near the poles".to_string(),
        ));
    }
    let phi = deg_to_rad(lat);
    let lam = deg_to_rad(lon);

    let e_sin_phi = E * phi.sin();
    // Isometric latitude
    let psi = ((PI / 4.0 + phi / 2.0).tan()
        * ((1.0 - e_sin_phi) / (1.0 + e_sin_phi)).powf(E / 2.0))
    .ln();

    let x = A * lam;
    let y = A * psi;
    Ok((x, y))
}

/// Mercator inverse projection.
///
/// # Arguments
/// * `x` – easting in metres
/// * `y` – northing in metres
///
/// # Returns
/// `(lon, lat)` in degrees.
pub fn mercator_inverse(x: f64, y: f64) -> SpatialResult<(f64, f64)> {
    let lon = rad_to_deg(x / A);

    // Iterative inversion of the isometric latitude
    let t = (-y / A).exp();                    // t = tan(π/4 - φ/2)  for sphere
    let mut phi = PI / 2.0 - 2.0 * t.atan();  // spherical first guess

    for _ in 0..20 {
        let e_sin_phi = E * phi.sin();
        let phi_new = PI / 2.0
            - 2.0
                * (t * ((1.0 - e_sin_phi) / (1.0 + e_sin_phi)).powf(E / 2.0)).atan();
        if (phi_new - phi).abs() < 1e-12 {
            phi = phi_new;
            break;
        }
        phi = phi_new;
    }

    Ok((lon, rad_to_deg(phi)))
}

// ═══════════════════════════════════════════════════════════════════════════════
//  3.  Transverse Mercator / UTM
// ═══════════════════════════════════════════════════════════════════════════════

/// UTM scale factor at the central meridian.
const K0: f64 = 0.9996;
/// UTM false easting (m).
const FALSE_EASTING: f64 = 500_000.0;
/// UTM false northing for the southern hemisphere (m).
const FALSE_NORTHING_S: f64 = 10_000_000.0;

/// Compute the meridional arc M from equator to latitude φ (Helmert series).
fn meridional_arc(phi: f64) -> f64 {
    let n = (A - B) / (A + B);
    let n2 = n * n;
    let n3 = n2 * n;
    let n4 = n2 * n2;

    A / (1.0 + n)
        * ((1.0 + n2 / 4.0 + n4 / 64.0) * phi
            - (3.0 / 2.0 * (n - n3 / 8.0)) * (2.0 * phi).sin()
            + (15.0 / 16.0 * (n2 - n4 / 4.0)) * (4.0 * phi).sin()
            - 35.0 / 48.0 * n3 * (6.0 * phi).sin()
            + 315.0 / 512.0 * n4 * (8.0 * phi).sin())
}

/// Transverse Mercator forward projection with explicit zone / central meridian.
///
/// # Arguments
/// * `lon`  – longitude in degrees
/// * `lat`  – geodetic latitude in degrees
/// * `zone` – UTM zone number 1–60; the central meridian is `(zone*6 − 183)°`
///
/// # Returns
/// `(easting, northing)` in metres (with UTM false origins applied).
///
/// # Errors
/// Returns an error if `zone` is outside 1–60 or latitude outside ±84°.
pub fn utm_forward(lon: f64, lat: f64, zone: u8) -> SpatialResult<(f64, f64)> {
    if zone < 1 || zone > 60 {
        return Err(SpatialError::ValueError(format!(
            "UTM zone must be 1–60, got {zone}"
        )));
    }
    if lat < -84.0 || lat > 84.0 {
        return Err(SpatialError::ValueError(format!(
            "UTM is defined only for latitudes −84°..84°, got {lat}"
        )));
    }
    let phi = deg_to_rad(lat);
    let lon0 = deg_to_rad((zone as f64 * 6.0) - 183.0);
    let dlam = deg_to_rad(lon) - lon0;

    let n_radius = prime_vertical_radius(phi);
    let t = phi.tan();
    let t2 = t * t;
    let c = E2 / (1.0 - E2) * phi.cos().powi(2);
    let a_coef = phi.cos() * dlam;
    let m = meridional_arc(phi);

    let x = K0
        * n_radius
        * (a_coef
            + (1.0 - t2 + c) * a_coef.powi(3) / 6.0
            + (5.0 - 18.0 * t2 + t2 * t2 + 72.0 * c - 58.0 * E2 / (1.0 - E2))
                * a_coef.powi(5)
                / 120.0)
        + FALSE_EASTING;

    let y0 = if lat < 0.0 { FALSE_NORTHING_S } else { 0.0 };
    let y = K0
        * (m
            + n_radius
                * t
                * (a_coef.powi(2) / 2.0
                    + (5.0 - t2 + 9.0 * c + 4.0 * c * c) * a_coef.powi(4) / 24.0
                    + (61.0 - 58.0 * t2 + t2 * t2 + 600.0 * c - 330.0 * E2 / (1.0 - E2))
                        * a_coef.powi(6)
                        / 720.0))
        + y0;

    Ok((x, y))
}

/// Transverse Mercator inverse projection with explicit UTM zone.
///
/// # Arguments
/// * `easting`  – UTM easting in metres
/// * `northing` – UTM northing in metres (with false northing already included)
/// * `zone`     – UTM zone number 1–60
/// * `northern` – `true` for northern hemisphere, `false` for southern
///
/// # Returns
/// `(lon, lat)` in degrees.
pub fn utm_inverse(
    easting: f64,
    northing: f64,
    zone: u8,
    northern: bool,
) -> SpatialResult<(f64, f64)> {
    if zone < 1 || zone > 60 {
        return Err(SpatialError::ValueError(format!(
            "UTM zone must be 1–60, got {zone}"
        )));
    }

    let lon0 = deg_to_rad((zone as f64 * 6.0) - 183.0);
    let x = easting - FALSE_EASTING;
    let y = if northern {
        northing
    } else {
        northing - FALSE_NORTHING_S
    };

    // Footprint latitude (iterate on meridional arc)
    let m = y / K0;
    let mu = m / (A * (1.0 - E2 / 4.0 - 3.0 * E2 * E2 / 64.0 - 5.0 * E2.powi(3) / 256.0));

    let e1 = (1.0 - (1.0 - E2).sqrt()) / (1.0 + (1.0 - E2).sqrt());
    let phi1 = mu
        + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
        + (21.0 * e1 * e1 / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
        + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin()
        + (1097.0 * e1.powi(4) / 512.0) * (8.0 * mu).sin();

    let n1 = prime_vertical_radius(phi1);
    let t1 = phi1.tan();
    let t1_2 = t1 * t1;
    let c1 = E2 / (1.0 - E2) * phi1.cos().powi(2);
    let r1 = A * (1.0 - E2) / (1.0 - E2 * phi1.sin().powi(2)).powf(1.5);
    let d = x / (n1 * K0);

    let lat = phi1
        - (n1 * t1 / r1)
            * (d * d / 2.0
                - (5.0 + 3.0 * t1_2 + 10.0 * c1 - 4.0 * c1 * c1 - 9.0 * E2 / (1.0 - E2))
                    * d.powi(4)
                    / 24.0
                + (61.0 + 90.0 * t1_2 + 298.0 * c1 + 45.0 * t1_2 * t1_2
                    - 252.0 * E2 / (1.0 - E2)
                    - 3.0 * c1 * c1)
                    * d.powi(6)
                    / 720.0);

    let lon = lon0
        + (d - (1.0 + 2.0 * t1_2 + c1) * d.powi(3) / 6.0
            + (5.0 - 2.0 * c1 + 28.0 * t1_2 - 3.0 * c1 * c1 + 8.0 * E2 / (1.0 - E2)
                + 24.0 * t1_2 * t1_2)
                * d.powi(5)
                / 120.0)
            / phi1.cos();

    Ok((rad_to_deg(lon), rad_to_deg(lat)))
}

// ═══════════════════════════════════════════════════════════════════════════════
//  4.  Mollweide equal-area projection
// ═══════════════════════════════════════════════════════════════════════════════

/// Mollweide forward projection (equal-area pseudo-cylindrical).
///
/// Projects the entire globe onto an ellipse with aspect ratio 2:1.
/// The central meridian is λ₀ = 0°.
///
/// # Arguments
/// * `lon` – longitude in degrees (−180..180)
/// * `lat` – geodetic latitude in degrees (−90..90)
///
/// # Returns
/// `(x, y)` in metres, on a unit sphere scaled to Earth's radius.
pub fn mollweide_forward(lon: f64, lat: f64) -> SpatialResult<(f64, f64)> {
    if lat < -90.0 || lat > 90.0 {
        return Err(SpatialError::ValueError(format!(
            "latitude must be −90..90, got {lat}"
        )));
    }
    let phi = deg_to_rad(lat);
    let lam = deg_to_rad(lon);

    // Solve 2θ + sin(2θ) = π sin(φ) via Newton–Raphson
    let target = PI * phi.sin();
    let mut theta = phi; // good initial guess
    for _ in 0..50 {
        let f = 2.0 * theta + (2.0 * theta).sin() - target;
        let fp = 2.0 + 2.0 * (2.0 * theta).cos();
        if fp.abs() < 1e-14 {
            break;
        }
        let delta = -f / fp;
        theta += delta;
        if delta.abs() < 1e-12 {
            break;
        }
    }

    let r = A * (2.0_f64.sqrt()); // scaling radius so equator maps to [-2R, 2R]
    let x = r * 2.0_f64.sqrt() / PI * lam * theta.cos();
    let y = r * 2.0_f64.sqrt() * theta.sin();
    Ok((x, y))
}

/// Mollweide inverse projection.
///
/// # Arguments
/// * `x` – projected x in metres
/// * `y` – projected y in metres
///
/// # Returns
/// `(lon, lat)` in degrees.
pub fn mollweide_inverse(x: f64, y: f64) -> SpatialResult<(f64, f64)> {
    let r = A * 2.0_f64.sqrt();
    let sin_theta = y / (r * 2.0_f64.sqrt());
    if sin_theta.abs() > 1.0 + 1e-10 {
        return Err(SpatialError::ValueError(
            "Mollweide inverse: point outside valid ellipse".to_string(),
        ));
    }
    let sin_theta_clamped = sin_theta.clamp(-1.0, 1.0);
    let theta = sin_theta_clamped.asin();

    let lat = (2.0 * theta + (2.0 * theta).sin()) / PI;
    let lat_clamped = lat.clamp(-1.0, 1.0).asin();

    let cos_theta = theta.cos();
    let lon = if cos_theta.abs() < 1e-10 {
        0.0
    } else {
        PI / (2.0_f64.sqrt()) * x / (r * cos_theta)
    };

    if lon.abs() > PI + 1e-10 {
        return Err(SpatialError::ValueError(
            "Mollweide inverse: point outside valid ellipse".to_string(),
        ));
    }

    Ok((rad_to_deg(lon), rad_to_deg(lat_clamped)))
}

// ═══════════════════════════════════════════════════════════════════════════════
//  5.  Azimuthal Equidistant projection
// ═══════════════════════════════════════════════════════════════════════════════

/// Azimuthal Equidistant forward projection.
///
/// Distances and directions from the projection centre are preserved exactly
/// (on a sphere).
///
/// # Arguments
/// * `lon`        – point longitude in degrees
/// * `lat`        – point latitude in degrees
/// * `centre_lon` – projection centre longitude in degrees
/// * `centre_lat` – projection centre latitude in degrees
///
/// # Returns
/// `(x, y)` in metres from the projection centre.
pub fn azimuthal_equidistant_forward(
    lon: f64,
    lat: f64,
    centre_lon: f64,
    centre_lat: f64,
) -> SpatialResult<(f64, f64)> {
    let phi = deg_to_rad(lat);
    let lam = deg_to_rad(lon);
    let phi0 = deg_to_rad(centre_lat);
    let lam0 = deg_to_rad(centre_lon);
    let dlam = lam - lam0;

    let cos_c = phi0.sin() * phi.sin() + phi0.cos() * phi.cos() * dlam.cos();
    // Angular distance c from centre
    let c = cos_c.clamp(-1.0, 1.0).acos();

    if c < 1e-12 {
        return Ok((0.0, 0.0));
    }

    let k = c / c.sin(); // k = c / sin(c) — scale factor
    let x = A * k * phi.cos() * dlam.sin();
    let y = A * k * (phi0.cos() * phi.sin() - phi0.sin() * phi.cos() * dlam.cos());
    Ok((x, y))
}

/// Azimuthal Equidistant inverse projection.
///
/// # Arguments
/// * `x`          – easting in metres from projection centre
/// * `y`          – northing in metres from projection centre
/// * `centre_lon` – projection centre longitude in degrees
/// * `centre_lat` – projection centre latitude in degrees
///
/// # Returns
/// `(lon, lat)` in degrees.
pub fn azimuthal_equidistant_inverse(
    x: f64,
    y: f64,
    centre_lon: f64,
    centre_lat: f64,
) -> SpatialResult<(f64, f64)> {
    let phi0 = deg_to_rad(centre_lat);
    let lam0 = deg_to_rad(centre_lon);

    let rho = (x * x + y * y).sqrt();
    if rho < 1e-10 {
        return Ok((centre_lon, centre_lat));
    }

    let c = rho / A;
    if c > PI + 1e-6 {
        return Err(SpatialError::ValueError(
            "Azimuthal equidistant inverse: point beyond antipodal distance".to_string(),
        ));
    }

    let phi = (c.cos() * phi0.sin() + y / rho * c.sin() * phi0.cos()).asin();
    let lam = lam0
        + (x * c.sin()).atan2(rho * phi0.cos() * c.cos() - y * phi0.sin() * c.sin());

    Ok((rad_to_deg(lam), rad_to_deg(phi)))
}

// ═══════════════════════════════════════════════════════════════════════════════
//  6.  Datum transformation  WGS84 → NAD83  (Helmert 7-parameter)
// ═══════════════════════════════════════════════════════════════════════════════

/// Helmert 7-parameter transformation: WGS84 ECEF → NAD83 ECEF.
///
/// Parameters from the official EPSG dataset (EPSG:1173).
/// Translation in metres, rotations in arc-seconds, scale in ppm.
///
/// # Arguments
/// `(x, y, z)` in WGS84 ECEF metres.
///
/// # Returns
/// `(x, y, z)` in NAD83 ECEF metres.
pub fn wgs84_to_nad83_ecef(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    // EPSG:1173  WGS 84 to NAD 83 (1)
    let tx = 0.9956;      // m
    let ty = -1.9013;     // m
    let tz = -0.5215;     // m
    let rx = 0.025915e-6; // rad  (≈ 0.025915 mas)
    let ry = 0.009426e-6; // rad
    let rz = 0.011599e-6; // rad
    let ds = 0.00062e-6;  // scale (ppm * 1e-6)

    let xn = (1.0 + ds) * (x - rz * y + ry * z) + tx;
    let yn = (1.0 + ds) * (rz * x + y - rx * z) + ty;
    let zn = (1.0 + ds) * (-ry * x + rx * y + z) + tz;
    (xn, yn, zn)
}

/// Full datum transformation WGS84 (lat/lon/alt) → NAD83 (lat/lon/alt).
///
/// Uses the Helmert 7-parameter shift then converts back to geographic
/// using the GRS80 ellipsoid (which defines NAD83).
pub fn wgs84_to_nad83(lat: f64, lon: f64, alt: f64) -> (f64, f64, f64) {
    let (x, y, z) = geo_to_ecef(lat, lon, alt);
    let (xn, yn, zn) = wgs84_to_nad83_ecef(x, y, z);
    // GRS80 constants (NAD83 datum)
    // Nearly identical to WGS84; difference is < 0.1 mm.
    ecef_to_geo(xn, yn, zn)
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ecef_roundtrip() {
        let cases = [
            (0.0, 0.0, 0.0),
            (51.5074, -0.1278, 45.0),
            (-33.8688, 151.2093, 10.0),
            (89.9, 0.0, 100.0),
            (-89.9, 180.0, 0.0),
        ];
        for (lat, lon, alt) in cases {
            let (x, y, z) = geo_to_ecef(lat, lon, alt);
            let (lat2, lon2, alt2) = ecef_to_geo(x, y, z);
            assert!(
                (lat2 - lat).abs() < 1e-7,
                "lat mismatch: {lat2} vs {lat}"
            );
            assert!(
                (lon2 - lon).abs() < 1e-7,
                "lon mismatch: {lon2} vs {lon}"
            );
            assert!((alt2 - alt).abs() < 1e-3, "alt mismatch: {alt2} vs {alt}");
        }
    }

    #[test]
    fn test_mercator_roundtrip() {
        let cases = [(0.0, 0.0), (51.5074, -0.1278), (-33.8688, 151.2093)];
        for (lat, lon) in cases {
            let (x, y) = mercator_forward(lon, lat).expect("forward");
            let (lon2, lat2) = mercator_inverse(x, y).expect("inverse");
            assert!((lat2 - lat).abs() < 1e-7, "lat: {lat2} vs {lat}");
            assert!((lon2 - lon).abs() < 1e-7, "lon: {lon2} vs {lon}");
        }
    }

    #[test]
    fn test_mercator_equator() {
        // At equator, x should be proportional to longitude
        let (x1, y1) = mercator_forward(1.0, 0.0).expect("fwd");
        let (x2, _) = mercator_forward(2.0, 0.0).expect("fwd");
        assert!((y1).abs() < 1.0, "equator northing ≈ 0: {y1}");
        assert!((x2 / x1 - 2.0).abs() < 1e-8, "linear in lon: {}", x2 / x1);
    }

    #[test]
    fn test_mercator_pole_rejected() {
        assert!(mercator_forward(0.0, 90.0).is_err());
        assert!(mercator_forward(0.0, -90.0).is_err());
    }

    #[test]
    fn test_utm_forward_inverse_nyc() {
        // New York City
        let (e, n) = utm_forward(-74.006, 40.7128, 18).expect("utm fwd");
        let (lon2, lat2) = utm_inverse(e, n, 18, true).expect("utm inv");
        assert!((lat2 - 40.7128).abs() < 1e-5, "lat: {lat2}");
        assert!((lon2 - (-74.006)).abs() < 1e-5, "lon: {lon2}");
    }

    #[test]
    fn test_utm_forward_inverse_sydney() {
        let (e, n) = utm_forward(151.2093, -33.8688, 56).expect("utm fwd");
        let (lon2, lat2) = utm_inverse(e, n, 56, false).expect("utm inv");
        assert!((lat2 - (-33.8688)).abs() < 1e-5, "lat: {lat2}");
        assert!((lon2 - 151.2093).abs() < 1e-5, "lon: {lon2}");
    }

    #[test]
    fn test_utm_invalid_zone() {
        assert!(utm_forward(0.0, 0.0, 0).is_err());
        assert!(utm_forward(0.0, 0.0, 61).is_err());
        assert!(utm_inverse(500_000.0, 0.0, 0, true).is_err());
    }

    #[test]
    fn test_mollweide_roundtrip() {
        let cases = [(0.0, 0.0), (48.8566, 2.3522), (-33.0, 151.0), (0.0, 180.0)];
        for (lat, lon) in cases {
            let (x, y) = mollweide_forward(lon, lat).expect("fwd");
            let (lon2, lat2) = mollweide_inverse(x, y).expect("inv");
            assert!((lat2 - lat).abs() < 1e-5, "lat: {lat2} vs {lat}");
            assert!((lon2 - lon).abs() < 1e-5, "lon: {lon2} vs {lon}");
        }
    }

    #[test]
    fn test_mollweide_poles() {
        // At poles, x must be 0 (on central meridian)
        let (x_n, _) = mollweide_forward(0.0, 90.0).expect("north pole");
        assert!(x_n.abs() < 1.0, "x at north pole: {x_n}");
        let (x_s, _) = mollweide_forward(0.0, -90.0).expect("south pole");
        assert!(x_s.abs() < 1.0, "x at south pole: {x_s}");
    }

    #[test]
    fn test_azimuthal_equidistant_roundtrip() {
        let centre = (0.0, 0.0); // (lon, lat)
        let pts = [(10.0, 20.0), (-30.0, 45.0), (90.0, -60.0)];
        for (lon, lat) in pts {
            let (x, y) =
                azimuthal_equidistant_forward(lon, lat, centre.0, centre.1).expect("fwd");
            let (lon2, lat2) =
                azimuthal_equidistant_inverse(x, y, centre.0, centre.1).expect("inv");
            assert!((lat2 - lat).abs() < 1e-5, "lat: {lat2} vs {lat}");
            assert!((lon2 - lon).abs() < 1e-5, "lon: {lon2} vs {lon}");
        }
    }

    #[test]
    fn test_azimuthal_equidistant_distance_preserved() {
        use crate::geo::distances::haversine_distance;
        let centre = (2.3522, 48.8566); // Paris
        let dest = (13.4050, 52.5200); // Berlin

        let (x, y) = azimuthal_equidistant_forward(
            dest.0, dest.1, centre.0, centre.1,
        )
        .expect("fwd");
        let projected_dist = (x * x + y * y).sqrt();
        let haversine = haversine_distance(centre.1, centre.0, dest.1, dest.0);
        // Should agree within 1 % (spherical approximation)
        assert!(
            (projected_dist - haversine).abs() / haversine < 0.01,
            "dist mismatch: projected={projected_dist}, haversine={haversine}"
        );
    }

    #[test]
    fn test_wgs84_to_nad83_small_shift() {
        // WGS84 and NAD83 differ by < 2 m at the surface
        let lat = 40.0;
        let lon = -75.0;
        let alt = 0.0;
        let (lat2, lon2, alt2) = wgs84_to_nad83(lat, lon, alt);
        assert!((lat2 - lat).abs() < 0.00002, "lat shift: {}", lat2 - lat);
        assert!((lon2 - lon).abs() < 0.00002, "lon shift: {}", lon2 - lon);
        assert!((alt2 - alt).abs() < 2.0, "alt shift: {}", alt2 - alt);
    }
}
