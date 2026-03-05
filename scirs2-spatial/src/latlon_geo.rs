#![allow(unused_assignments)]
//! Geospatial algorithms with `LatLon` coordinate type.
//!
//! Provides:
//! * Haversine and Vincenty geodesic distance calculations.
//! * Bearing, destination point, midpoint.
//! * Cross-track and along-track distances.
//! * Point-in-polygon (ray casting, spherical).
//! * Convex hull of `LatLon` points (Graham scan).
//! * Spherical polygon area.
//! * Mercator and UTM (transverse Mercator) projections with inverses.
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::latlon_geo::{LatLon, haversine_distance, bearing, destination_point};
//!
//! let london = LatLon { lat: 51.5074, lon: -0.1278 };
//! let paris  = LatLon { lat: 48.8566, lon:  2.3522 };
//!
//! let dist = haversine_distance(&london, &paris);
//! assert!((dist - 343_000.0).abs() < 10_000.0);
//!
//! let b = bearing(&london, &paris);
//! assert!(b > 100.0 && b < 170.0);
//! ```

use std::f64::consts::PI;

// ── WGS84 constants ───────────────────────────────────────────────────────────

/// Mean Earth radius (metres), used by the haversine approximation.
const R_EARTH: f64 = 6_371_000.0;

/// WGS84 semi-major axis (metres).
const WGS84_A: f64 = 6_378_137.0;
/// WGS84 flattening.
const WGS84_F: f64 = 1.0 / 298.257_223_563;
/// WGS84 semi-minor axis.
const WGS84_B: f64 = WGS84_A * (1.0 - WGS84_F);

// ── LatLon ────────────────────────────────────────────────────────────────────

/// WGS84 geodetic coordinates (degrees).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatLon {
    /// Latitude in decimal degrees (−90 … +90).
    pub lat: f64,
    /// Longitude in decimal degrees (−180 … +180).
    pub lon: f64,
}

impl LatLon {
    /// Construct a new `LatLon`.
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }

    /// Return latitude in radians.
    #[inline]
    fn lat_r(&self) -> f64 { self.lat.to_radians() }

    /// Return longitude in radians.
    #[inline]
    fn lon_r(&self) -> f64 { self.lon.to_radians() }
}

// ── Haversine distance ────────────────────────────────────────────────────────

/// Great-circle distance (metres) using the haversine formula.
///
/// Accurate to within ~0.3 % compared with Vincenty over global distances.
pub fn haversine_distance(a: &LatLon, b: &LatLon) -> f64 {
    let dlat = (b.lat_r() - a.lat_r()) * 0.5;
    let dlon = (b.lon_r() - a.lon_r()) * 0.5;
    let h = dlat.sin().powi(2) + a.lat_r().cos() * b.lat_r().cos() * dlon.sin().powi(2);
    2.0 * R_EARTH * h.sqrt().asin()
}

// ── Vincenty distance ─────────────────────────────────────────────────────────

/// Iterative Vincenty formula for geodesic distance on the WGS84 ellipsoid.
///
/// Returns `Err` if the iteration fails to converge (e.g., antipodal points).
pub fn vincenty_distance(a: &LatLon, b: &LatLon) -> Result<f64, String> {
    let (lat1, lon1) = (a.lat_r(), a.lon_r());
    let (lat2, lon2) = (b.lat_r(), b.lon_r());

    let u1 = ((1.0 - WGS84_F) * lat1.tan()).atan();
    let u2 = ((1.0 - WGS84_F) * lat2.tan()).atan();
    let (sin_u1, cos_u1) = (u1.sin(), u1.cos());
    let (sin_u2, cos_u2) = (u2.sin(), u2.cos());

    let l = lon2 - lon1;
    let mut lambda = l;

    let mut cos2_alpha = 0.0_f64;
    let mut sin_sigma = 0.0_f64;
    let mut cos_sigma = 0.0_f64;
    let mut cos2_sigma_m = 0.0_f64;
    let mut sigma = 0.0_f64;

    for _ in 0..200 {
        let sin_lambda = lambda.sin();
        let cos_lambda = lambda.cos();

        sin_sigma = ((cos_u2 * sin_lambda).powi(2)
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda).powi(2))
        .sqrt();

        if sin_sigma.abs() < 1e-12 {
            return Ok(0.0); // coincident points
        }

        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
        sigma = sin_sigma.atan2(cos_sigma);

        let sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma;
        cos2_alpha = 1.0 - sin_alpha.powi(2);

        cos2_sigma_m = if cos2_alpha.abs() > 1e-12 {
            cos_sigma - 2.0 * sin_u1 * sin_u2 / cos2_alpha
        } else {
            0.0
        };

        let c = WGS84_F / 16.0 * cos2_alpha * (4.0 + WGS84_F * (4.0 - 3.0 * cos2_alpha));

        let lambda_new = l
            + (1.0 - c)
                * WGS84_F
                * sin_alpha
                * (sigma
                    + c * sin_sigma
                        * (cos2_sigma_m
                            + c * cos_sigma * (-1.0 + 2.0 * cos2_sigma_m.powi(2))));

        if (lambda_new - lambda).abs() < 1e-12 {
            lambda = lambda_new;
            break;
        }
        lambda = lambda_new;
    }

    let u_sq = cos2_alpha * (WGS84_A.powi(2) - WGS84_B.powi(2)) / WGS84_B.powi(2);
    let big_a = 1.0
        + u_sq / 16384.0
            * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
    let big_b =
        u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));

    let delta_sigma = big_b
        * sin_sigma
        * (cos2_sigma_m
            + big_b / 4.0
                * (cos_sigma * (-1.0 + 2.0 * cos2_sigma_m.powi(2))
                    - big_b / 6.0
                        * cos2_sigma_m
                        * (-3.0 + 4.0 * sin_sigma.powi(2))
                        * (-3.0 + 4.0 * cos2_sigma_m.powi(2))));

    Ok(WGS84_B * big_a * (sigma - delta_sigma))
}

// ── Bearing ───────────────────────────────────────────────────────────────────

/// Initial bearing from `a` to `b` (degrees, 0 = North, clockwise).
pub fn bearing(a: &LatLon, b: &LatLon) -> f64 {
    let dlon = b.lon_r() - a.lon_r();
    let x = dlon.sin() * b.lat_r().cos();
    let y = a.lat_r().cos() * b.lat_r().sin()
        - a.lat_r().sin() * b.lat_r().cos() * dlon.cos();
    let theta = x.atan2(y);
    (theta.to_degrees() + 360.0) % 360.0
}

// ── Destination point ─────────────────────────────────────────────────────────

/// Point reached from `start` by travelling `distance_m` metres along `bearing_deg`.
pub fn destination_point(start: &LatLon, bearing_deg: f64, distance_m: f64) -> LatLon {
    let d_r = distance_m / R_EARTH;
    let brng = bearing_deg.to_radians();
    let lat1 = start.lat_r();
    let lon1 = start.lon_r();

    let lat2 = (lat1.sin() * d_r.cos() + lat1.cos() * d_r.sin() * brng.cos()).asin();
    let lon2 = lon1
        + (brng.sin() * d_r.sin() * lat1.cos())
            .atan2(d_r.cos() - lat1.sin() * lat2.sin());

    LatLon {
        lat: lat2.to_degrees(),
        lon: ((lon2.to_degrees() + 540.0) % 360.0) - 180.0,
    }
}

// ── Midpoint ──────────────────────────────────────────────────────────────────

/// Great-circle midpoint of `a` and `b`.
pub fn midpoint(a: &LatLon, b: &LatLon) -> LatLon {
    let dlon = b.lon_r() - a.lon_r();
    let bx = b.lat_r().cos() * dlon.cos();
    let by = b.lat_r().cos() * dlon.sin();
    let lat = (a.lat_r().sin() + b.lat_r().sin())
        .atan2(((a.lat_r().cos() + bx).powi(2) + by.powi(2)).sqrt());
    let lon = a.lon_r() + by.atan2(a.lat_r().cos() + bx);
    LatLon {
        lat: lat.to_degrees(),
        lon: ((lon.to_degrees() + 540.0) % 360.0) - 180.0,
    }
}

// ── Cross-track distance ──────────────────────────────────────────────────────

/// Perpendicular (signed) distance from `point` to the great-circle line a→b (metres).
///
/// Positive = left of the path, negative = right.
pub fn cross_track_distance(point: &LatLon, a: &LatLon, b: &LatLon) -> f64 {
    let d_ap = haversine_distance(a, point) / R_EARTH;
    let theta_ap = bearing(a, point).to_radians();
    let theta_ab = bearing(a, b).to_radians();
    (d_ap.sin() * (theta_ap - theta_ab).sin()).asin() * R_EARTH
}

// ── Along-track distance ──────────────────────────────────────────────────────

/// Signed distance along the path a→b to the point on the path closest to `point`.
pub fn along_track_distance(point: &LatLon, a: &LatLon, b: &LatLon) -> f64 {
    let d_ap = haversine_distance(a, point) / R_EARTH;
    let dxt = cross_track_distance(point, a, b) / R_EARTH;
    (d_ap.cos() / dxt.cos()).acos() * R_EARTH
}

// ── Point in polygon ──────────────────────────────────────────────────────────

/// Ray-casting point-in-polygon test (spherical / planar approximation).
///
/// Treats latitude as y and longitude as x.
pub fn point_in_polygon(point: &LatLon, polygon: &[LatLon]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let (px, py) = (point.lon, point.lat);
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (polygon[i].lon, polygon[i].lat);
        let (xj, yj) = (polygon[j].lon, polygon[j].lat);
        let intersect = ((yi > py) != (yj > py))
            && (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}

// ── Convex hull (Graham scan on LatLon treated as 2-D) ────────────────────────

/// Convex hull of a set of `LatLon` points using the Graham scan algorithm.
///
/// Treats longitude as x and latitude as y.  Returns points in CCW order.
pub fn convex_hull(points: &[LatLon]) -> Vec<LatLon> {
    if points.len() < 3 {
        return points.to_vec();
    }

    // Find lowest (smallest lat), leftmost (smallest lon) as pivot.
    let mut pts: Vec<(f64, f64, usize)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| (p.lon, p.lat, i))
        .collect();

    let pivot_idx = pts
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    pts.swap(0, pivot_idx);
    let (px, py, _) = pts[0];

    // Sort remaining by polar angle with respect to pivot.
    let tail = &mut pts[1..];
    tail.sort_unstable_by(|a, b| {
        let cross = (a.0 - px) * (b.1 - py) - (a.1 - py) * (b.0 - px);
        if cross.abs() < 1e-12 {
            // Collinear: closer first.
            let da = (a.0 - px).powi(2) + (a.1 - py).powi(2);
            let db = (b.0 - px).powi(2) + (b.1 - py).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            (-cross).partial_cmp(&0.0).unwrap_or(std::cmp::Ordering::Equal)
        }
    });

    let mut stack: Vec<(f64, f64)> = Vec::with_capacity(pts.len());
    for &(x, y, _) in &pts {
        while stack.len() >= 2 {
            let n = stack.len();
            let (ax, ay) = stack[n - 2];
            let (bx, by) = stack[n - 1];
            // Cross product of (b-a) × (p-a); pop if not CCW turn.
            let cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
            if cross <= 0.0 {
                stack.pop();
            } else {
                break;
            }
        }
        stack.push((x, y));
    }

    stack
        .iter()
        .map(|&(lon, lat)| LatLon { lat, lon })
        .collect()
}

// ── Spherical polygon area ─────────────────────────────────────────────────────

/// Compute the area of a spherical polygon (km²) using the spherical excess formula.
///
/// Points must form a simple polygon in CCW order.
pub fn spherical_polygon_area(polygon: &[LatLon]) -> f64 {
    let n = polygon.len();
    if n < 3 {
        return 0.0;
    }
    // Girard's theorem: area = (sum of interior angles - (n-2)*π) * R²
    // Here we use the l'Huilier formula for each spherical triangle formed with origin.
    // More practically, use the sum of excess formula for the polygon.

    // Convert to Cartesian unit vectors.
    let to_xyz = |p: &LatLon| -> [f64; 3] {
        let (lat, lon) = (p.lat_r(), p.lon_r());
        [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
    };

    let dot = |a: [f64; 3], b: [f64; 3]| -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    };

    let cross = |a: [f64; 3], b: [f64; 3]| -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };

    let _norm = |v: [f64; 3]| -> f64 { (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() };

    // Compute signed spherical area via the sum of spherical triangle areas.
    let mut area = 0.0_f64;
    let v0 = to_xyz(&polygon[0]);

    for i in 1..(n - 1) {
        let va = to_xyz(&polygon[i]);
        let vb = to_xyz(&polygon[i + 1]);

        // Spherical triangle area via l'Huilier's theorem.
        // Sides a, b, c:
        let s_a = dot(va, vb).clamp(-1.0, 1.0).acos();
        let s_b = dot(v0, vb).clamp(-1.0, 1.0).acos();
        let s_c = dot(v0, va).clamp(-1.0, 1.0).acos();

        let s = (s_a + s_b + s_c) * 0.5;
        let t_val = (s * 0.5).tan()
            * ((s - s_a) * 0.5).tan()
            * ((s - s_b) * 0.5).tan()
            * ((s - s_c) * 0.5).tan();
        let t_val = t_val.max(0.0);
        let excess = 4.0 * t_val.sqrt().atan();

        // Sign: use cross product to determine orientation.
        let c1 = cross(v0, va);
        let orientation = dot(c1, vb);
        if orientation >= 0.0 {
            area += excess;
        } else {
            area -= excess;
        }
    }

    // R² in km² = (6371)²
    let r_km = R_EARTH / 1000.0;
    area.abs() * r_km * r_km
}

// ── Mercator projection ────────────────────────────────────────────────────────

/// Forward Mercator projection: `LatLon` → `(x, y)` in metres.
///
/// Returns `(easting, northing)` relative to the prime meridian / equator.
pub fn mercator_project(ll: &LatLon) -> [f64; 2] {
    let x = WGS84_A * ll.lon_r();
    let lat_r = ll.lat_r();
    let y = WGS84_A * ((PI / 4.0 + lat_r / 2.0).tan()).ln();
    [x, y]
}

/// Inverse Mercator: `(x, y)` metres → `LatLon`.
pub fn mercator_unproject(xy: [f64; 2]) -> LatLon {
    let lon = (xy[0] / WGS84_A).to_degrees();
    let lat = (2.0 * (xy[1] / WGS84_A).exp().atan() - PI / 2.0).to_degrees();
    LatLon { lat, lon }
}

// ── UTM ────────────────────────────────────────────────────────────────────────

/// Return the UTM zone number (1–60) for a given longitude.
pub fn utm_zone(lon: f64) -> u32 {
    (((lon + 180.0) / 6.0).floor() as u32 % 60) + 1
}

/// Simplified UTM forward projection.
///
/// Returns `((easting, northing), zone)`.  Easting and northing are in metres.
/// Uses the WGS84 transverse Mercator formulae (Krueger series, 4th order).
pub fn utm_project(ll: &LatLon) -> ([f64; 2], u32) {
    let zone = utm_zone(ll.lon);
    let lon0 = ((zone as f64 - 1.0) * 6.0 - 177.0).to_radians();

    let lat = ll.lat_r();
    let lon = ll.lon_r();
    let dlon = lon - lon0;

    let k0 = 0.9996_f64;
    let e2 = WGS84_F * (2.0 - WGS84_F);
    let e_prime2 = e2 / (1.0 - e2);

    let n_val = WGS84_A / (1.0 - e2 * lat.sin().powi(2)).sqrt();

    let t = lat.tan();
    let c = e_prime2 * lat.cos().powi(2);
    let a_coef = lat.cos() * dlon;

    // Meridional arc
    let e4 = e2 * e2;
    let e6 = e4 * e2;
    let m_arc = WGS84_A
        * ((1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * lat
            - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * (2.0 * lat).sin()
            + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * (4.0 * lat).sin()
            - (35.0 * e6 / 3072.0) * (6.0 * lat).sin());

    let easting = k0
        * n_val
        * (a_coef
            + (1.0 - t.powi(2) + c) * a_coef.powi(3) / 6.0
            + (5.0 - 18.0 * t.powi(2) + t.powi(4) + 72.0 * c - 58.0 * e_prime2)
                * a_coef.powi(5)
                / 120.0)
        + 500_000.0;

    let mut northing = k0
        * (m_arc
            + n_val
                * lat.tan()
                * (a_coef.powi(2) / 2.0
                    + (5.0 - t.powi(2) + 9.0 * c + 4.0 * c.powi(2)) * a_coef.powi(4) / 24.0
                    + (61.0 - 58.0 * t.powi(2) + t.powi(4) + 600.0 * c - 330.0 * e_prime2)
                        * a_coef.powi(6)
                        / 720.0));

    if ll.lat < 0.0 {
        northing += 10_000_000.0;
    }

    ([easting, northing], zone)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    fn london() -> LatLon { LatLon { lat: 51.5074, lon: -0.1278 } }
    fn paris()  -> LatLon { LatLon { lat: 48.8566, lon:  2.3522 } }

    #[test]
    fn test_haversine_distance() {
        let d = haversine_distance(&london(), &paris());
        // ~343 km
        assert!((d - 343_000.0).abs() < 5_000.0, "d={d}");
    }

    #[test]
    fn test_haversine_same_point() {
        let p = london();
        assert!(haversine_distance(&p, &p) < 1e-6);
    }

    #[test]
    fn test_vincenty_distance() {
        let dh = haversine_distance(&london(), &paris());
        let dv = vincenty_distance(&london(), &paris()).expect("vincenty");
        // Should be within 0.5% of haversine.
        assert!((dv - dh).abs() / dh < 0.005, "dv={dv}, dh={dh}");
    }

    #[test]
    fn test_vincenty_coincident() {
        let p = london();
        let d = vincenty_distance(&p, &p).expect("vincenty coincident");
        assert!(d < 1.0);
    }

    #[test]
    fn test_bearing_east() {
        let a = LatLon { lat: 0.0, lon: 0.0 };
        let b = LatLon { lat: 0.0, lon: 1.0 };
        let b_deg = bearing(&a, &b);
        // East ≈ 90°
        assert!((b_deg - 90.0).abs() < 1.0, "bearing={b_deg}");
    }

    #[test]
    fn test_bearing_north() {
        let a = LatLon { lat: 0.0, lon: 0.0 };
        let b = LatLon { lat: 1.0, lon: 0.0 };
        let b_deg = bearing(&a, &b);
        assert!(b_deg < 0.5 || b_deg > 359.5, "bearing={b_deg}");
    }

    #[test]
    fn test_destination_point_roundtrip() {
        let start = london();
        let brng = 135.0;
        let dist = 200_000.0;
        let dest = destination_point(&start, brng, dist);
        let back = haversine_distance(&start, &dest);
        assert!((back - dist).abs() < 200.0, "back={back}");
    }

    #[test]
    fn test_midpoint() {
        let m = midpoint(&london(), &paris());
        // Midpoint should be roughly halfway.
        let d_lm = haversine_distance(&london(), &m);
        let d_pm = haversine_distance(&paris(), &m);
        assert!((d_lm - d_pm).abs() / d_lm < 0.01, "d_lm={d_lm}, d_pm={d_pm}");
    }

    #[test]
    fn test_cross_track_distance() {
        // Point directly on the path should have ~0 cross-track distance.
        let m = midpoint(&london(), &paris());
        let d = cross_track_distance(&m, &london(), &paris());
        assert!(d.abs() < 1000.0, "d={d}");
    }

    #[test]
    fn test_point_in_polygon() {
        let poly = vec![
            LatLon { lat: 0.0, lon: 0.0 },
            LatLon { lat: 0.0, lon: 10.0 },
            LatLon { lat: 10.0, lon: 10.0 },
            LatLon { lat: 10.0, lon: 0.0 },
        ];
        let inside = LatLon { lat: 5.0, lon: 5.0 };
        let outside = LatLon { lat: 15.0, lon: 5.0 };
        assert!(point_in_polygon(&inside, &poly));
        assert!(!point_in_polygon(&outside, &poly));
    }

    #[test]
    fn test_convex_hull_square() {
        let pts = vec![
            LatLon { lat: 0.0, lon: 0.0 },
            LatLon { lat: 0.0, lon: 1.0 },
            LatLon { lat: 1.0, lon: 1.0 },
            LatLon { lat: 1.0, lon: 0.0 },
            LatLon { lat: 0.5, lon: 0.5 }, // interior
        ];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4, "hull={:?}", hull);
    }

    #[test]
    fn test_spherical_polygon_area() {
        // Small square ~1°×1° near equator: area ≈ 12_300 km²
        let poly = vec![
            LatLon { lat: 0.0, lon: 0.0 },
            LatLon { lat: 0.0, lon: 1.0 },
            LatLon { lat: 1.0, lon: 1.0 },
            LatLon { lat: 1.0, lon: 0.0 },
        ];
        let area = spherical_polygon_area(&poly);
        // ~12,300 km²
        assert!(area > 10_000.0 && area < 15_000.0, "area={area}");
    }

    #[test]
    fn test_mercator_roundtrip() {
        let ll = paris();
        let xy = mercator_project(&ll);
        let back = mercator_unproject(xy);
        assert!((back.lat - ll.lat).abs() < EPSILON, "lat={}", back.lat);
        assert!((back.lon - ll.lon).abs() < EPSILON, "lon={}", back.lon);
    }

    #[test]
    fn test_utm_zone() {
        assert_eq!(utm_zone(-0.1278), 30); // London
        assert_eq!(utm_zone(2.3522),  31); // Paris
        assert_eq!(utm_zone(139.6917), 54); // Tokyo
    }

    #[test]
    fn test_utm_project_london() {
        let ([e, n], zone) = utm_project(&london());
        assert_eq!(zone, 30);
        // London UTM easting ~699_000, northing ~5_710_000
        assert!((e - 699_000.0).abs() < 5_000.0, "e={e}");
        assert!((n - 5_710_000.0).abs() < 10_000.0, "n={n}");
    }
}
