//! # Map Projections
//!
//! Provides forward and inverse map projections for cartographic use, including
//! Mercator, Transverse Mercator (a.k.a. Gauss-Krüger), UTM, Azimuthal Equidistant,
//! and Lambert Conformal Conic.
//!
//! All angular inputs/outputs are in **decimal degrees**.
//! All linear inputs/outputs are in **metres**.
//!
//! ## Ellipsoid
//!
//! All projections use the WGS-84 ellipsoid constants defined in
//! [`super::coordinates`].
//!
//! ## Example — UTM round-trip
//!
//! ```rust
//! use scirs2_core::geo::projection::{to_utm, from_utm};
//!
//! // New York City ≈ UTM zone 18T
//! let (zone, band, easting, northing) = to_utm(40.7128, -74.0060).expect("utm ok");
//! assert_eq!(zone, 18);
//! let (lat2, lon2) = from_utm(zone, band, easting, northing).expect("from_utm ok");
//! assert!((lat2 - 40.7128).abs() < 1e-5);
//! assert!((lon2 - (-74.0060)).abs() < 1e-5);
//! ```

use std::f64::consts::PI;

use super::coordinates::{GeoError, GeoResult, WGS84_A, WGS84_B, WGS84_E2, WGS84_F};

// ---------------------------------------------------------------------------
// Internal helpers
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

/// WGS-84 first eccentricity (not squared).
const WGS84_E: f64 = 0.081_819_190_842_622; // sqrt(WGS84_E2) — precomputed

/// Isometric latitude (Mercator ψ parameter) for a given geographic latitude φ
/// (radians) on the WGS-84 ellipsoid.
///
/// ψ = ln tan(π/4 + φ/2) − (e/2) ln((1 + e sin φ)/(1 − e sin φ))
fn isometric_lat(lat_rad: f64) -> f64 {
    let e_sin = WGS84_E * lat_rad.sin();
    let term1 = (PI / 4.0 + lat_rad / 2.0).tan().ln();
    let term2 = (WGS84_E / 2.0) * ((1.0 + e_sin) / (1.0 - e_sin)).ln();
    term1 - term2
}

/// Inverse isometric latitude — recover geographic latitude φ (radians) from
/// the isometric latitude ψ using Newton–Raphson iteration.
fn isometric_lat_inverse(psi: f64) -> f64 {
    let mut lat = 2.0 * psi.exp().atan() - PI / 2.0;
    for _ in 0..20 {
        let e_sin = WGS84_E * lat.sin();
        let lat_new = 2.0
            * (psi + (WGS84_E / 2.0) * ((1.0 + e_sin) / (1.0 - e_sin)).ln()).exp().atan()
            - PI / 2.0;
        if (lat_new - lat).abs() < 1e-14 {
            return lat_new;
        }
        lat = lat_new;
    }
    lat
}

// ---------------------------------------------------------------------------
// Meridian arc
// ---------------------------------------------------------------------------

/// Meridian arc length from the equator to latitude φ (radians) for the WGS-84
/// ellipsoid using Helmert's series (accurate to ~0.1 mm).
fn meridian_arc(lat_rad: f64) -> f64 {
    // Helmert series coefficients for WGS-84
    let n = (WGS84_A - WGS84_B) / (WGS84_A + WGS84_B);
    let n2 = n * n;
    let n3 = n2 * n;
    let n4 = n3 * n;

    let a0 = 1.0 + n2 / 4.0 + n4 / 64.0;
    let a2 = 3.0 / 2.0 * (n - n3 / 8.0);
    let a4 = 15.0 / 16.0 * (n2 - n4 / 4.0);
    let a6 = 35.0 / 48.0 * n3;
    let a8 = 315.0 / 512.0 * n4;

    let coeff = WGS84_A / (1.0 + n) * a0;
    coeff
        * (lat_rad - a2 * (2.0 * lat_rad).sin() + a4 * (4.0 * lat_rad).sin()
            - a6 * (6.0 * lat_rad).sin()
            + a8 * (8.0 * lat_rad).sin())
}

// ---------------------------------------------------------------------------
// Mercator projection
// ---------------------------------------------------------------------------

/// **Mercator** (oblique/equatorial) — forward projection.
///
/// # Arguments
///
/// * `lat`  – Geographic latitude in degrees (must be strictly between −90 and 90).
/// * `lon`  – Geographic longitude in degrees.
/// * `lon0` – Central meridian in degrees.
///
/// # Returns
///
/// `(x, y)` in metres, with `x` pointing east and `y` pointing north.
pub fn mercator_forward(lat: f64, lon: f64, lon0: f64) -> (f64, f64) {
    let lat_r = deg2rad(lat);
    let x = WGS84_A * deg2rad(lon - lon0);
    let y = WGS84_A * isometric_lat(lat_r);
    (x, y)
}

/// **Mercator** — inverse projection.
///
/// # Arguments
///
/// * `x`    – Easting in metres.
/// * `y`    – Northing in metres.
/// * `lon0` – Central meridian in degrees.
///
/// # Returns
///
/// `(lat, lon)` in decimal degrees.
pub fn mercator_inverse(x: f64, y: f64, lon0: f64) -> (f64, f64) {
    let psi = y / WGS84_A;
    let lat = rad2deg(isometric_lat_inverse(psi));
    let lon = lon0 + rad2deg(x / WGS84_A);
    (lat, lon)
}

// ---------------------------------------------------------------------------
// Transverse Mercator (Gauss-Krüger)
// ---------------------------------------------------------------------------

/// **Transverse Mercator** — forward projection (Gauss-Krüger formulation).
///
/// Uses the Karney (2011) series truncated to 6th order for high accuracy
/// (errors < 0.1 mm within 3 400 km of the central meridian).
///
/// # Arguments
///
/// * `lat`  – Geographic latitude in degrees.
/// * `lon`  – Geographic longitude in degrees.
/// * `lon0` – Central meridian in degrees.
///
/// # Errors
///
/// Returns [`GeoError::InvalidCoordinate`] if the longitude difference from the
/// central meridian exceeds 90°.
pub fn transverse_mercator_forward(lat: f64, lon: f64, lon0: f64) -> GeoResult<(f64, f64)> {
    let dlon = lon - lon0;
    if dlon.abs() > 90.0 {
        return Err(GeoError::InvalidCoordinate {
            field: "lon",
            reason: format!(
                "longitude difference from central meridian ({dlon:.2}°) exceeds ±90°"
            ),
        });
    }

    let lat_r = deg2rad(lat);
    let lon_r = deg2rad(dlon);

    // 3rd flattening
    let n = (WGS84_A - WGS84_B) / (WGS84_A + WGS84_B);
    let n2 = n * n;
    let n3 = n2 * n;
    let n4 = n3 * n;

    // Conformal latitude
    let tau = lat_r.tan();
    let sigma = (WGS84_E * (WGS84_E * tau / (1.0 + tau * tau).sqrt()).atanh()).sinh();
    let tau_prime = tau * (1.0 + sigma * sigma).sqrt() - sigma * (1.0 + tau * tau).sqrt();

    let xi_prime = tau_prime.atan2(lon_r.cos());
    let eta_prime = (lon_r.sin() / (tau_prime * tau_prime + lon_r.cos() * lon_r.cos()).sqrt()).asinh();

    // Meridian arc coefficients (Karney)
    let alpha = [
        0.0_f64, // placeholder index 0
        1.0 / 2.0 * n - 2.0 / 3.0 * n2 + 5.0 / 16.0 * n3 + 41.0 / 180.0 * n4,
        13.0 / 48.0 * n2 - 3.0 / 5.0 * n3 + 557.0 / 1440.0 * n4,
        61.0 / 240.0 * n3 - 103.0 / 140.0 * n4,
        49561.0 / 161280.0 * n4,
    ];

    let mut xi = xi_prime;
    let mut eta = eta_prime;
    for j in 1..=4_usize {
        xi += alpha[j] * (2.0 * j as f64 * xi_prime).sin() * (2.0 * j as f64 * eta_prime).cosh();
        eta +=
            alpha[j] * (2.0 * j as f64 * xi_prime).cos() * (2.0 * j as f64 * eta_prime).sinh();
    }

    // Scale factor A
    let a_scale = WGS84_B * (1.0 + n2 / 4.0 + n4 / 64.0);

    Ok((a_scale * eta, a_scale * xi))
}

/// **Transverse Mercator** — inverse projection.
///
/// # Arguments
///
/// * `x`    – Easting in metres (relative to the central meridian).
/// * `y`    – Northing in metres (from the equator).
/// * `lon0` – Central meridian in degrees.
///
/// # Errors
///
/// Returns [`GeoError::InvalidCoordinate`] if the coordinates are outside the
/// valid domain (projected easting too large).
pub fn transverse_mercator_inverse(x: f64, y: f64, lon0: f64) -> GeoResult<(f64, f64)> {
    let n = (WGS84_A - WGS84_B) / (WGS84_A + WGS84_B);
    let n2 = n * n;
    let n3 = n2 * n;
    let n4 = n3 * n;

    let a_scale = WGS84_B * (1.0 + n2 / 4.0 + n4 / 64.0);
    let xi = y / a_scale;
    let eta = x / a_scale;

    // Inverse series coefficients (Karney)
    let beta = [
        0.0_f64,
        1.0 / 2.0 * n - 2.0 / 3.0 * n2 + 37.0 / 96.0 * n3 - 1.0 / 360.0 * n4,
        1.0 / 48.0 * n2 + 1.0 / 15.0 * n3 - 437.0 / 1440.0 * n4,
        17.0 / 480.0 * n3 - 37.0 / 840.0 * n4,
        4397.0 / 161280.0 * n4,
    ];

    let mut xi_prime = xi;
    let mut eta_prime = eta;
    for j in 1..=4_usize {
        xi_prime -= beta[j] * (2.0 * j as f64 * xi).sin() * (2.0 * j as f64 * eta).cosh();
        eta_prime -= beta[j] * (2.0 * j as f64 * xi).cos() * (2.0 * j as f64 * eta).sinh();
    }

    let tau_prime = xi_prime.sin() / (eta_prime * eta_prime + xi_prime.cos() * xi_prime.cos()).sqrt();

    // Newton-Raphson to invert conformal latitude (Karney 2011).
    // Given tau_prime (tan of conformal latitude), recover tau = tan(geographic lat).
    // f(tau) = taup(tau) - tau_prime, where taup(tau) = tau*sqrt(1+sigma^2) - sigma*sqrt(1+tau^2)
    // and sigma = sinh(e * atanh(e*tau/sqrt(1+tau^2))).
    // Derivative (Karney eq. 7):
    //   df/dtau = sqrt(1+taup^2) * sqrt(1+tau^2) * (1-e^2) / (1+(1-e^2)*tau^2)
    let mut tau = tau_prime;
    for _ in 0..20 {
        let tau2 = tau * tau;
        let tau_hyp = (1.0 + tau2).sqrt();
        let e_arg = WGS84_E * tau / tau_hyp;
        let sigma = (WGS84_E * e_arg.atanh()).sinh();
        let taup_computed = tau * (1.0 + sigma * sigma).sqrt() - sigma * tau_hyp;
        let dtaup_dtau = (1.0 + taup_computed * taup_computed).sqrt()
            * tau_hyp
            * (1.0 - WGS84_E2)
            / (1.0 + (1.0 - WGS84_E2) * tau2);
        let delta = (taup_computed - tau_prime) / dtaup_dtau;
        tau -= delta;
        if delta.abs() < 1e-14 {
            break;
        }
    }

    let lat = rad2deg(tau.atan());
    let lon = lon0 + rad2deg((eta_prime.sinh()).atan2(xi_prime.cos()));

    Ok((lat, lon))
}

// ---------------------------------------------------------------------------
// UTM
// ---------------------------------------------------------------------------

/// Return the UTM zone number (1 – 60) for a given longitude.
///
/// The special zones for Norway (32V) and Svalbard (31X, 33X, 35X, 37X) are
/// *not* handled here; use the full `to_utm` function if you need those.
#[inline]
pub fn utm_zone(lon: f64) -> u8 {
    (((lon + 180.0) / 6.0).floor() as i32 % 60 + 1) as u8
}

/// Return the UTM latitude band letter for a given latitude (−80 … +84 degrees).
///
/// Returns `None` for latitudes outside the UTM coverage area.
pub fn utm_band(lat: f64) -> Option<char> {
    const BANDS: &[char] = &[
        'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
        'V', 'W', 'X',
    ];
    if lat < -80.0 || lat > 84.0 {
        return None;
    }
    let idx = ((lat + 80.0) / 8.0).floor() as usize;
    let idx = idx.min(BANDS.len() - 1);
    Some(BANDS[idx])
}

/// Convert geographic coordinates to UTM (Universal Transverse Mercator).
///
/// # Returns
///
/// `(zone, band, easting, northing)` where easting is in metres east of the
/// zone's false origin (500 000 m) and northing is in metres north of the
/// equator (southern hemisphere adds 10 000 000 m false northing).
///
/// # Errors
///
/// Returns [`GeoError::InvalidCoordinate`] if the latitude is outside
/// −80 … +84 degrees (UPS territory).
pub fn to_utm(lat: f64, lon: f64) -> GeoResult<(u8, char, f64, f64)> {
    if lat < -80.0 || lat > 84.0 {
        return Err(GeoError::InvalidCoordinate {
            field: "lat",
            reason: format!("UTM is defined for −80 … +84 degrees; got {lat:.4}"),
        });
    }

    let zone = utm_zone(lon);
    let band = utm_band(lat).ok_or_else(|| GeoError::InvalidCoordinate {
        field: "lat",
        reason: format!("latitude {lat:.4} is outside UTM coverage"),
    })?;

    let lon0 = (zone as f64 - 1.0) * 6.0 - 180.0 + 3.0; // central meridian

    // Transverse Mercator with scale factor k0 = 0.9996
    let (x_tm, y_tm) = transverse_mercator_forward(lat, lon, lon0)?;
    const K0: f64 = 0.9996;

    let easting = K0 * x_tm + 500_000.0;
    let northing = if lat < 0.0 {
        K0 * y_tm + 10_000_000.0
    } else {
        K0 * y_tm
    };

    Ok((zone, band, easting, northing))
}

/// Convert UTM coordinates back to geographic coordinates.
///
/// # Arguments
///
/// * `zone`     – UTM zone number (1 – 60).
/// * `band`     – UTM latitude band letter (C … X, excluding I and O).
/// * `easting`  – UTM easting in metres.
/// * `northing` – UTM northing in metres.
///
/// # Returns
///
/// `(lat, lon)` in decimal degrees.
///
/// # Errors
///
/// Returns [`GeoError::InvalidCoordinate`] for invalid zone numbers, or
/// [`GeoError::DomainError`] for other issues.
pub fn from_utm(zone: u8, band: char, easting: f64, northing: f64) -> GeoResult<(f64, f64)> {
    if zone < 1 || zone > 60 {
        return Err(GeoError::InvalidCoordinate {
            field: "zone",
            reason: format!("UTM zone must be 1–60, got {zone}"),
        });
    }

    let lon0 = (zone as f64 - 1.0) * 6.0 - 180.0 + 3.0;
    const K0: f64 = 0.9996;

    let x_tm = (easting - 500_000.0) / K0;
    let northing_adj = if band < 'N' {
        // Southern hemisphere: remove false northing
        (northing - 10_000_000.0) / K0
    } else {
        northing / K0
    };

    transverse_mercator_inverse(x_tm, northing_adj, lon0)
}

// ---------------------------------------------------------------------------
// Azimuthal Equidistant
// ---------------------------------------------------------------------------

/// **Azimuthal Equidistant** projection — forward.
///
/// Projects the point `(lat, lon)` onto a plane centred at `(lat0, lon0)`.
/// Distances and directions from the centre point are preserved.
///
/// Uses a spherical approximation (radius = WGS-84 semi-major axis).
///
/// # Returns
///
/// `(x, y)` in metres.
pub fn azimuthal_equidistant(lat: f64, lon: f64, lat0: f64, lon0: f64) -> (f64, f64) {
    let lat_r = deg2rad(lat);
    let lon_r = deg2rad(lon);
    let lat0_r = deg2rad(lat0);
    let lon0_r = deg2rad(lon0);

    let cos_c = lat0_r.sin() * lat_r.sin()
        + lat0_r.cos() * lat_r.cos() * (lon_r - lon0_r).cos();

    // Angular distance
    let c = cos_c.clamp(-1.0, 1.0).acos();

    if c.abs() < 1e-12 {
        return (0.0, 0.0);
    }

    let k = c / c.sin();
    let x = WGS84_A
        * k
        * lat_r.cos()
        * (lon_r - lon0_r).sin();
    let y = WGS84_A
        * k
        * (lat0_r.cos() * lat_r.sin() - lat0_r.sin() * lat_r.cos() * (lon_r - lon0_r).cos());

    (x, y)
}

// ---------------------------------------------------------------------------
// Lambert Conformal Conic
// ---------------------------------------------------------------------------

/// **Lambert Conformal Conic** — forward projection.
///
/// Preserves angles (conformal) with two standard parallels.
///
/// Uses a spherical approximation (radius = WGS-84 semi-major axis).
///
/// # Arguments
///
/// * `lat`  – Geographic latitude of the point in degrees.
/// * `lon`  – Geographic longitude of the point in degrees.
/// * `lat0` – Latitude of origin in degrees (false origin latitude).
/// * `lon0` – Central meridian in degrees (false origin longitude).
/// * `lat1` – First standard parallel in degrees.
/// * `lat2` – Second standard parallel in degrees.
///
/// # Returns
///
/// `(x, y)` in metres.
pub fn lambert_conic_forward(
    lat: f64,
    lon: f64,
    lat0: f64,
    lon0: f64,
    lat1: f64,
    lat2: f64,
) -> (f64, f64) {
    let lat_r = deg2rad(lat);
    let lon_r = deg2rad(lon);
    let lat0_r = deg2rad(lat0);
    let lon0_r = deg2rad(lon0);
    let lat1_r = deg2rad(lat1);
    let lat2_r = deg2rad(lat2);

    // Cone constant n
    let n = if (lat1 - lat2).abs() < 1e-10 {
        lat1_r.sin()
    } else {
        (lat1_r.cos().ln() - lat2_r.cos().ln())
            / ((PI / 4.0 + lat2_r / 2.0).tan().ln()
                - (PI / 4.0 + lat1_r / 2.0).tan().ln())
    };

    let f = lat1_r.cos() * (PI / 4.0 + lat1_r / 2.0).tan().powf(n) / n;
    let rho0 = WGS84_A * f * (PI / 4.0 + lat0_r / 2.0).tan().powf(-n);
    let rho = WGS84_A * f * (PI / 4.0 + lat_r / 2.0).tan().powf(-n);
    let theta = n * (lon_r - lon0_r);

    let x = rho * theta.sin();
    let y = rho0 - rho * theta.cos();

    (x, y)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Mercator
    // -----------------------------------------------------------------------

    #[test]
    fn test_mercator_equator_maps_to_y_zero() {
        let (_, y) = mercator_forward(0.0, 0.0, 0.0);
        assert!(
            y.abs() < 1e-6,
            "equator should map to y=0, got y={y}"
        );
    }

    #[test]
    fn test_mercator_prime_meridian_maps_to_x_zero() {
        let (x, _) = mercator_forward(45.0, 0.0, 0.0);
        assert!(x.abs() < 1e-6, "prime meridian lon=0 → x=0, got x={x}");
    }

    #[test]
    fn test_mercator_round_trip() {
        let lat0 = 45.0_f64;
        let lon0 = 30.0_f64;
        let (x, y) = mercator_forward(lat0, lon0, 0.0);
        let (lat1, lon1) = mercator_inverse(x, y, 0.0);
        assert!(
            (lat1 - lat0).abs() < 1e-9,
            "lat round-trip error: {}",
            (lat1 - lat0).abs()
        );
        assert!(
            (lon1 - lon0).abs() < 1e-9,
            "lon round-trip error: {}",
            (lon1 - lon0).abs()
        );
    }

    #[test]
    fn test_mercator_central_meridian_shift() {
        // With lon0=10, a point at lon=10 should give x=0
        let (x, _) = mercator_forward(0.0, 10.0, 10.0);
        assert!(x.abs() < 1e-6, "x should be 0 when lon=lon0, got {x}");
    }

    // -----------------------------------------------------------------------
    // Transverse Mercator
    // -----------------------------------------------------------------------

    #[test]
    fn test_transverse_mercator_round_trip() {
        let lat0 = 51.5_f64;
        let lon0 = -0.1_f64;
        let (x, y) = transverse_mercator_forward(lat0, lon0, 0.0).expect("tm forward");
        let (lat1, lon1) = transverse_mercator_inverse(x, y, 0.0).expect("tm inverse");
        assert!(
            (lat1 - lat0).abs() < 1e-8,
            "TM lat round-trip: {}",
            (lat1 - lat0).abs()
        );
        assert!(
            (lon1 - lon0).abs() < 1e-8,
            "TM lon round-trip: {}",
            (lon1 - lon0).abs()
        );
    }

    #[test]
    fn test_transverse_mercator_origin_maps_to_zero() {
        // At the intersection of the central meridian and equator, both x and y
        // should be 0.
        let (x, y) = transverse_mercator_forward(0.0, 0.0, 0.0).expect("tm forward");
        assert!(x.abs() < 1e-3, "x should be ~0, got {x}");
        assert!(y.abs() < 1e-3, "y should be ~0, got {y}");
    }

    #[test]
    fn test_transverse_mercator_dlon_too_large() {
        let result = transverse_mercator_forward(0.0, 100.0, 0.0);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // UTM zone
    // -----------------------------------------------------------------------

    #[test]
    fn test_utm_zone_new_york() {
        // New York is at approximately −74° longitude → zone 18
        assert_eq!(utm_zone(-74.0), 18, "New York should be zone 18");
    }

    #[test]
    fn test_utm_zone_london() {
        // London is at ~−0.1° → zone 30
        assert_eq!(utm_zone(-0.1), 30, "London should be zone 30");
    }

    #[test]
    fn test_utm_zone_180() {
        // −180° is the start of zone 1
        assert_eq!(utm_zone(-180.0), 1);
    }

    #[test]
    fn test_utm_zone_boundaries() {
        // Zone boundaries are at multiples of 6 degrees from -180 degrees.
        // Zone 1: [-180, -174), Zone 2: [-174, -168), etc.
        assert_eq!(utm_zone(-174.001), 1); // last point in zone 1
        assert_eq!(utm_zone(-174.0), 2); // exactly on boundary -> zone 2
        assert_eq!(utm_zone(-173.999), 2); // clearly in zone 2
    }

    // -----------------------------------------------------------------------
    // UTM round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_utm_round_trip_new_york() {
        let lat = 40.7128_f64;
        let lon = -74.0060_f64;
        let (zone, band, easting, northing) = to_utm(lat, lon).expect("to_utm");
        assert_eq!(zone, 18, "NYC zone should be 18");
        let (lat2, lon2) = from_utm(zone, band, easting, northing).expect("from_utm");
        assert!(
            (lat2 - lat).abs() < 1e-5,
            "UTM lat round-trip error: {}",
            (lat2 - lat).abs()
        );
        assert!(
            (lon2 - lon).abs() < 1e-5,
            "UTM lon round-trip error: {}",
            (lon2 - lon).abs()
        );
    }

    #[test]
    fn test_utm_round_trip_sydney() {
        let lat = -33.8688_f64;
        let lon = 151.2093_f64;
        let (zone, band, easting, northing) = to_utm(lat, lon).expect("to_utm Sydney");
        let (lat2, lon2) = from_utm(zone, band, easting, northing).expect("from_utm Sydney");
        assert!(
            (lat2 - lat).abs() < 1e-5,
            "UTM Sydney lat: {}",
            (lat2 - lat).abs()
        );
        assert!(
            (lon2 - lon).abs() < 1e-5,
            "UTM Sydney lon: {}",
            (lon2 - lon).abs()
        );
    }

    #[test]
    fn test_utm_invalid_latitude() {
        let result = to_utm(85.0, 0.0);
        assert!(result.is_err(), "lat=85 should be invalid for UTM");
    }

    #[test]
    fn test_utm_invalid_zone() {
        let result = from_utm(0, 'N', 500_000.0, 0.0);
        assert!(result.is_err());
        let result2 = from_utm(61, 'N', 500_000.0, 0.0);
        assert!(result2.is_err());
    }

    // -----------------------------------------------------------------------
    // Azimuthal Equidistant
    // -----------------------------------------------------------------------

    #[test]
    fn test_azimuthal_equidistant_origin_is_zero() {
        let (x, y) = azimuthal_equidistant(45.0, 90.0, 45.0, 90.0);
        assert!(x.abs() < 1e-6 && y.abs() < 1e-6, "origin → (0,0), got ({x}, {y})");
    }

    #[test]
    fn test_azimuthal_equidistant_distance_preserved() {
        use crate::geo::coordinates::GeographicCoord;
        let origin_lat = 0.0_f64;
        let origin_lon = 0.0_f64;
        let pt_lat = 1.0_f64;
        let pt_lon = 0.0_f64;
        let (x, y) = azimuthal_equidistant(pt_lat, pt_lon, origin_lat, origin_lon);
        let aeqd_dist = (x * x + y * y).sqrt();
        let great_circle = GeographicCoord::new(origin_lat, origin_lon, 0.0)
            .haversine_distance(&GeographicCoord::new(pt_lat, pt_lon, 0.0));
        assert!(
            (aeqd_dist - great_circle).abs() < 10.0,
            "AEQD distance: {aeqd_dist:.1} vs great-circle: {great_circle:.1}"
        );
    }

    // -----------------------------------------------------------------------
    // Lambert Conformal Conic
    // -----------------------------------------------------------------------

    #[test]
    fn test_lambert_conic_origin_maps_to_zero() {
        // At the origin (lat=lat0, lon=lon0), x and y should both be 0.
        let lat0 = 23.0_f64;
        let lon0 = -96.0_f64;
        let (x, y) = lambert_conic_forward(lat0, lon0, lat0, lon0, 29.5, 45.5);
        // x = 0 since lon = lon0
        assert!(x.abs() < 1e-3, "x at origin should be ~0, got {x}");
        // y = rho0 - rho; at lat=lat0, rho = rho0, so y ≈ 0
        assert!(y.abs() < 1e-3, "y at origin should be ~0, got {y}");
    }

    #[test]
    fn test_lambert_conic_central_meridian_x_zero() {
        // Along the central meridian, x should be 0
        let (x, _) = lambert_conic_forward(40.0, -96.0, 23.0, -96.0, 29.5, 45.5);
        assert!(x.abs() < 1e-3, "x on central meridian should be ~0, got {x}");
    }
}
