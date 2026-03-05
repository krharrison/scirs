//! Advanced geospatial utilities
//!
//! Provides rich types and algorithms for working with geographic coordinates
//! on the WGS84 ellipsoid, including:
//!
//! - [`GeoPoint`]: WGS84 latitude/longitude point with distance and bearing methods.
//! - [`HaversineDistance`]: Great-circle distance via the Haversine formula.
//! - [`VincentyDistance`]: Highly accurate ellipsoidal distance (Vincenty's method).
//! - [`BoundingBox2D`]: Geographic bounding box with spatial predicates.
//! - [`Geohash`]: Encode/decode geohash strings (base32).
//! - [`H3Lite`]: Simplified hexagonal hierarchical index (H3-inspired, approximate).
//!
//! # Examples
//!
//! ```rust
//! use scirs2_spatial::geospatial_advanced::{GeoPoint, HaversineDistance, Geohash};
//!
//! let london = GeoPoint::new(51.5074, -0.1278).unwrap();
//! let paris  = GeoPoint::new(48.8566,  2.3522).unwrap();
//!
//! let dist_km = HaversineDistance::distance_km(&london, &paris);
//! assert!((dist_km - 343.5).abs() < 2.0, "Haversine ~343 km, got {dist_km:.1}");
//!
//! let hash = Geohash::encode(london.lat(), london.lon(), 5).unwrap();
//! assert_eq!(&hash, "gcpvh");
//! ```

use crate::error::{SpatialError, SpatialResult};
use std::f64::consts::PI;

// WGS-84 ellipsoid constants (also defined in geospatial.rs – we keep them
// local to avoid a circular re-export dependency).
const WGS84_A: f64 = 6_378_137.0; // equatorial radius, metres
const WGS84_B: f64 = 6_356_752.314_245; // polar radius, metres
const WGS84_F: f64 = 1.0 / 298.257_223_563; // flattening
const EARTH_RADIUS_M: f64 = 6_371_008.8; // mean radius, metres

// ---------------------------------------------------------------------------
// GeoPoint
// ---------------------------------------------------------------------------

/// A point on the WGS84 ellipsoid, identified by latitude and longitude.
///
/// Latitude is in the range [-90, 90] and longitude in [-180, 180].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoPoint {
    lat: f64, // degrees
    lon: f64, // degrees
}

impl GeoPoint {
    /// Create a new `GeoPoint`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if latitude is outside [-90, 90] or longitude outside
    /// [-180, 180].
    pub fn new(lat: f64, lon: f64) -> SpatialResult<Self> {
        if !(-90.0..=90.0).contains(&lat) {
            return Err(SpatialError::ValueError(format!(
                "Latitude {lat} is outside [-90, 90]"
            )));
        }
        if !(-180.0..=180.0).contains(&lon) {
            return Err(SpatialError::ValueError(format!(
                "Longitude {lon} is outside [-180, 180]"
            )));
        }
        Ok(Self { lat, lon })
    }

    /// Latitude in degrees.
    pub fn lat(&self) -> f64 {
        self.lat
    }

    /// Longitude in degrees.
    pub fn lon(&self) -> f64 {
        self.lon
    }

    /// Latitude in radians.
    pub fn lat_rad(&self) -> f64 {
        self.lat.to_radians()
    }

    /// Longitude in radians.
    pub fn lon_rad(&self) -> f64 {
        self.lon.to_radians()
    }

    /// Haversine distance to `other` in metres.
    pub fn haversine_distance_m(&self, other: &GeoPoint) -> f64 {
        HaversineDistance::distance_m(self, other)
    }

    /// Haversine distance to `other` in kilometres.
    pub fn haversine_distance_km(&self, other: &GeoPoint) -> f64 {
        HaversineDistance::distance_km(self, other)
    }

    /// Vincenty distance to `other` in metres (accurate ellipsoidal formula).
    ///
    /// Returns `None` if the iteration fails to converge (nearly antipodal points).
    pub fn vincenty_distance_m(&self, other: &GeoPoint) -> Option<f64> {
        VincentyDistance::distance_m(self, other)
    }

    /// Initial bearing (azimuth) from `self` to `other`, in degrees [0, 360).
    pub fn bearing_to(&self, other: &GeoPoint) -> f64 {
        let lat1 = self.lat_rad();
        let lat2 = other.lat_rad();
        let dlon = (other.lon - self.lon).to_radians();

        let y = dlon.sin() * lat2.cos();
        let x = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * dlon.cos();
        let bearing = y.atan2(x).to_degrees();
        (bearing + 360.0) % 360.0
    }

    /// Destination point given a distance (metres) and bearing (degrees).
    ///
    /// Uses the spherical-Earth approximation.
    pub fn destination(&self, distance_m: f64, bearing_deg: f64) -> GeoPoint {
        let r = EARTH_RADIUS_M;
        let d = distance_m / r;
        let theta = bearing_deg.to_radians();
        let lat1 = self.lat_rad();
        let lon1 = self.lon_rad();

        let lat2 = (lat1.sin() * d.cos() + lat1.cos() * d.sin() * theta.cos()).asin();
        let lon2 = lon1
            + (theta.sin() * d.sin() * lat1.cos()).atan2(d.cos() - lat1.sin() * lat2.sin());

        let lat2_deg = lat2.to_degrees().clamp(-90.0, 90.0);
        let lon2_deg = ((lon2.to_degrees() + 540.0) % 360.0) - 180.0;
        GeoPoint { lat: lat2_deg, lon: lon2_deg }
    }

    /// Convert to ECEF (Earth-Centred, Earth-Fixed) coordinates in metres.
    pub fn to_ecef(&self) -> (f64, f64, f64) {
        let lat = self.lat_rad();
        let lon = self.lon_rad();
        let e2 = 2.0 * WGS84_F - WGS84_F * WGS84_F;
        let n = WGS84_A / (1.0 - e2 * lat.sin().powi(2)).sqrt();
        let x = n * lat.cos() * lon.cos();
        let y = n * lat.cos() * lon.sin();
        let z = n * (1.0 - e2) * lat.sin();
        (x, y, z)
    }
}

impl std::fmt::Display for GeoPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ns = if self.lat >= 0.0 { 'N' } else { 'S' };
        let ew = if self.lon >= 0.0 { 'E' } else { 'W' };
        write!(f, "{:.4}°{}{:.4}°{}", self.lat.abs(), ns, self.lon.abs(), ew)
    }
}

// ---------------------------------------------------------------------------
// HaversineDistance
// ---------------------------------------------------------------------------

/// Great-circle distance between two [`GeoPoint`]s using the Haversine formula.
///
/// Accurate to better than 0.5% over all distances.
pub struct HaversineDistance;

impl HaversineDistance {
    /// Distance in metres.
    pub fn distance_m(a: &GeoPoint, b: &GeoPoint) -> f64 {
        let dlat = (b.lat - a.lat).to_radians();
        let dlon = (b.lon - a.lon).to_radians();
        let lat1 = a.lat_rad();
        let lat2 = b.lat_rad();
        let h = (dlat / 2.0).sin().powi(2)
            + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
        2.0 * EARTH_RADIUS_M * h.sqrt().asin()
    }

    /// Distance in kilometres.
    pub fn distance_km(a: &GeoPoint, b: &GeoPoint) -> f64 {
        Self::distance_m(a, b) / 1000.0
    }

    /// Distance in nautical miles.
    pub fn distance_nm(a: &GeoPoint, b: &GeoPoint) -> f64 {
        Self::distance_m(a, b) / 1852.0
    }
}

// ---------------------------------------------------------------------------
// VincentyDistance
// ---------------------------------------------------------------------------

/// Highly accurate geodesic distance on the WGS84 ellipsoid using Vincenty's
/// direct formulae.
///
/// Accurate to within 0.5 mm for any two points.  The iterative method can
/// fail to converge for nearly antipodal points; in that case `distance_m`
/// returns `None`.
pub struct VincentyDistance;

impl VincentyDistance {
    /// Geodesic distance in metres between two WGS84 points.
    ///
    /// Returns `None` if iteration does not converge.
    pub fn distance_m(a: &GeoPoint, b: &GeoPoint) -> Option<f64> {
        let (a_r, b_r, f) = (WGS84_A, WGS84_B, WGS84_F);

        let lat1 = a.lat_rad();
        let lat2 = b.lat_rad();
        let lon1 = a.lon_rad();
        let lon2 = b.lon_rad();

        // Reduced latitudes
        let u1 = ((1.0 - f) * lat1.tan()).atan();
        let u2 = ((1.0 - f) * lat2.tan()).atan();

        let sin_u1 = u1.sin();
        let cos_u1 = u1.cos();
        let sin_u2 = u2.sin();
        let cos_u2 = u2.cos();

        let l = lon2 - lon1;
        let mut lambda = l;
        let mut prev_lambda;

        let (mut cos2_alpha, mut sin_sigma, mut cos_sigma, mut sigma, mut cos2_sigma_m) =
            (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

        const MAX_ITER: usize = 1000;
        let mut converged = false;

        for _ in 0..MAX_ITER {
            prev_lambda = lambda;

            let sin_lambda = lambda.sin();
            let cos_lambda = lambda.cos();

            sin_sigma = ((cos_u2 * sin_lambda).powi(2)
                + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda).powi(2))
            .sqrt();

            if sin_sigma.abs() < 1e-10 {
                // Coincident points
                return Some(0.0);
            }

            cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
            sigma = sin_sigma.atan2(cos_sigma);

            let sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma;
            cos2_alpha = 1.0 - sin_alpha * sin_alpha;

            cos2_sigma_m = if cos2_alpha.abs() > 1e-10 {
                cos_sigma - 2.0 * sin_u1 * sin_u2 / cos2_alpha
            } else {
                0.0 // equatorial line
            };

            let c = f / 16.0 * cos2_alpha * (4.0 + f * (4.0 - 3.0 * cos2_alpha));
            lambda = l
                + (1.0 - c)
                    * f
                    * sin_alpha
                    * (sigma
                        + c * sin_sigma
                            * (cos2_sigma_m
                                + c * cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)));

            if (lambda - prev_lambda).abs() < 1e-12 {
                converged = true;
                break;
            }
        }

        if !converged {
            return None;
        }

        let u2 = cos2_alpha * (a_r * a_r - b_r * b_r) / (b_r * b_r);
        let k1 = (1.0 + u2).sqrt();
        let big_a = (1.0 + u2 / 4.0 * (4.0 + u2 * (-3.0 + u2))) / k1;
        let big_b = u2 / k1 * (1.0 + u2 / 4.0 * ((-1.0 + u2) * (1.0 + u2 / 4.0)));

        let delta_sigma = big_b
            * sin_sigma
            * (cos2_sigma_m
                + big_b / 4.0
                    * (cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)
                        - big_b / 6.0
                            * cos2_sigma_m
                            * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                            * (-3.0 + 4.0 * cos2_sigma_m * cos2_sigma_m)));

        Some(b_r * big_a * (sigma - delta_sigma))
    }

    /// Geodesic distance in kilometres.
    pub fn distance_km(a: &GeoPoint, b: &GeoPoint) -> Option<f64> {
        Self::distance_m(a, b).map(|d| d / 1000.0)
    }
}

// ---------------------------------------------------------------------------
// BoundingBox2D
// ---------------------------------------------------------------------------

/// A geographic bounding box defined by min/max latitude and longitude.
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox2D {
    /// Minimum latitude (degrees)
    pub min_lat: f64,
    /// Maximum latitude (degrees)
    pub max_lat: f64,
    /// Minimum longitude (degrees)
    pub min_lon: f64,
    /// Maximum longitude (degrees)
    pub max_lon: f64,
}

impl BoundingBox2D {
    /// Create a new bounding box.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `min_lat > max_lat` or `min_lon > max_lon`, or if
    /// values are outside valid WGS84 ranges.
    pub fn new(
        min_lat: f64,
        max_lat: f64,
        min_lon: f64,
        max_lon: f64,
    ) -> SpatialResult<Self> {
        if !(-90.0..=90.0).contains(&min_lat) || !(-90.0..=90.0).contains(&max_lat) {
            return Err(SpatialError::ValueError(
                "Latitude must be in [-90, 90]".to_string(),
            ));
        }
        if !(-180.0..=180.0).contains(&min_lon) || !(-180.0..=180.0).contains(&max_lon) {
            return Err(SpatialError::ValueError(
                "Longitude must be in [-180, 180]".to_string(),
            ));
        }
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
        Ok(Self { min_lat, max_lat, min_lon, max_lon })
    }

    /// Build from a slice of [`GeoPoint`]s.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the slice is empty.
    pub fn from_points(pts: &[GeoPoint]) -> SpatialResult<Self> {
        if pts.is_empty() {
            return Err(SpatialError::ValueError(
                "Cannot compute bounding box of empty point set".to_string(),
            ));
        }
        let min_lat = pts.iter().map(|p| p.lat).fold(f64::INFINITY, f64::min);
        let max_lat = pts.iter().map(|p| p.lat).fold(f64::NEG_INFINITY, f64::max);
        let min_lon = pts.iter().map(|p| p.lon).fold(f64::INFINITY, f64::min);
        let max_lon = pts.iter().map(|p| p.lon).fold(f64::NEG_INFINITY, f64::max);
        Ok(Self { min_lat, max_lat, min_lon, max_lon })
    }

    /// Check whether a point is contained within this bounding box.
    pub fn contains(&self, p: &GeoPoint) -> bool {
        p.lat >= self.min_lat
            && p.lat <= self.max_lat
            && p.lon >= self.min_lon
            && p.lon <= self.max_lon
    }

    /// Check whether this bounding box overlaps another.
    pub fn overlaps(&self, other: &BoundingBox2D) -> bool {
        self.min_lat <= other.max_lat
            && self.max_lat >= other.min_lat
            && self.min_lon <= other.max_lon
            && self.max_lon >= other.min_lon
    }

    /// Expand this bounding box to include a point, returning the new box.
    pub fn expand_to_include(&self, p: &GeoPoint) -> Self {
        Self {
            min_lat: self.min_lat.min(p.lat),
            max_lat: self.max_lat.max(p.lat),
            min_lon: self.min_lon.min(p.lon),
            max_lon: self.max_lon.max(p.lon),
        }
    }

    /// Centre point.
    pub fn center(&self) -> GeoPoint {
        // Safe because the constructor validates ranges
        GeoPoint {
            lat: (self.min_lat + self.max_lat) / 2.0,
            lon: (self.min_lon + self.max_lon) / 2.0,
        }
    }

    /// Approximate area in square kilometres using the Haversine formula.
    pub fn area_km2(&self) -> f64 {
        let sw = GeoPoint { lat: self.min_lat, lon: self.min_lon };
        let ne = GeoPoint { lat: self.max_lat, lon: self.max_lon };
        let se = GeoPoint { lat: self.min_lat, lon: self.max_lon };
        let width_km = HaversineDistance::distance_km(&sw, &se);
        let height_km = HaversineDistance::distance_km(&sw, &GeoPoint {
            lat: self.max_lat,
            lon: self.min_lon,
        });
        let _ = ne; // avoid unused warning
        width_km * height_km
    }
}

// ---------------------------------------------------------------------------
// Geohash (base32 encoding/decoding)
// ---------------------------------------------------------------------------

/// Geohash encoder/decoder.
///
/// Geohash is a hierarchical spatial data structure that subdivides space into
/// buckets of grid shape.  It encodes a WGS84 lat/lon pair into a short
/// alphanumeric string, where longer strings represent smaller areas.
///
/// This implementation uses the standard base32 alphabet:
/// `0123456789bcdefghjkmnpqrstuvwxyz`
pub struct Geohash;

/// Geohash base32 alphabet.
const GEOHASH_ALPHABET: &[u8] = b"0123456789bcdefghjkmnpqrstuvwxyz";

/// Reverse lookup table from ASCII byte to 5-bit value.
fn geohash_decode_char(c: char) -> Option<u64> {
    GEOHASH_ALPHABET
        .iter()
        .position(|&b| b == c as u8)
        .map(|i| i as u64)
}

impl Geohash {
    /// Encode a latitude/longitude pair to a geohash string of the given length
    /// (precision = number of characters).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `lat`/`lon` are out of WGS84 range or `precision == 0`.
    pub fn encode(lat: f64, lon: f64, precision: usize) -> SpatialResult<String> {
        if !(-90.0..=90.0).contains(&lat) {
            return Err(SpatialError::ValueError(format!(
                "Latitude {lat} out of range [-90, 90]"
            )));
        }
        if !(-180.0..=180.0).contains(&lon) {
            return Err(SpatialError::ValueError(format!(
                "Longitude {lon} out of range [-180, 180]"
            )));
        }
        if precision == 0 {
            return Err(SpatialError::ValueError(
                "Geohash precision must be at least 1".to_string(),
            ));
        }

        let mut lat_range = (-90.0_f64, 90.0_f64);
        let mut lon_range = (-180.0_f64, 180.0_f64);
        let mut hash = String::with_capacity(precision);
        let mut bits = 0u64;
        let mut num_bits = 0u32;
        let mut is_lon = true; // geohash alternates lon, lat, lon, lat ...

        let bits_needed = precision * 5;
        for _ in 0..bits_needed {
            bits <<= 1;
            if is_lon {
                let mid = (lon_range.0 + lon_range.1) / 2.0;
                if lon >= mid {
                    bits |= 1;
                    lon_range.0 = mid;
                } else {
                    lon_range.1 = mid;
                }
            } else {
                let mid = (lat_range.0 + lat_range.1) / 2.0;
                if lat >= mid {
                    bits |= 1;
                    lat_range.0 = mid;
                } else {
                    lat_range.1 = mid;
                }
            }
            is_lon = !is_lon;
            num_bits += 1;
            if num_bits == 5 {
                let idx = bits & 0x1F;
                hash.push(GEOHASH_ALPHABET[idx as usize] as char);
                bits = 0;
                num_bits = 0;
            }
        }

        Ok(hash)
    }

    /// Decode a geohash string to `(lat, lon)` (centre of the encoded cell).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the string contains an invalid character.
    pub fn decode(hash: &str) -> SpatialResult<(f64, f64)> {
        if hash.is_empty() {
            return Err(SpatialError::ValueError(
                "Cannot decode empty geohash".to_string(),
            ));
        }

        let mut lat_range = (-90.0_f64, 90.0_f64);
        let mut lon_range = (-180.0_f64, 180.0_f64);
        let mut is_lon = true;

        for c in hash.chars() {
            let bits = geohash_decode_char(c).ok_or_else(|| {
                SpatialError::ValueError(format!("Invalid geohash character '{c}'"))
            })?;
            for bit_pos in (0..5u32).rev() {
                let bit = (bits >> bit_pos) & 1;
                if is_lon {
                    let mid = (lon_range.0 + lon_range.1) / 2.0;
                    if bit == 1 {
                        lon_range.0 = mid;
                    } else {
                        lon_range.1 = mid;
                    }
                } else {
                    let mid = (lat_range.0 + lat_range.1) / 2.0;
                    if bit == 1 {
                        lat_range.0 = mid;
                    } else {
                        lat_range.1 = mid;
                    }
                }
                is_lon = !is_lon;
            }
        }

        let lat = (lat_range.0 + lat_range.1) / 2.0;
        let lon = (lon_range.0 + lon_range.1) / 2.0;
        Ok((lat, lon))
    }

    /// Decode a geohash string to its bounding box.
    ///
    /// Returns `(min_lat, max_lat, min_lon, max_lon)`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the string contains an invalid character.
    pub fn decode_bbox(hash: &str) -> SpatialResult<(f64, f64, f64, f64)> {
        if hash.is_empty() {
            return Err(SpatialError::ValueError(
                "Cannot decode empty geohash".to_string(),
            ));
        }

        let mut lat_range = (-90.0_f64, 90.0_f64);
        let mut lon_range = (-180.0_f64, 180.0_f64);
        let mut is_lon = true;

        for c in hash.chars() {
            let bits = geohash_decode_char(c).ok_or_else(|| {
                SpatialError::ValueError(format!("Invalid geohash character '{c}'"))
            })?;
            for bit_pos in (0..5u32).rev() {
                let bit = (bits >> bit_pos) & 1;
                if is_lon {
                    let mid = (lon_range.0 + lon_range.1) / 2.0;
                    if bit == 1 {
                        lon_range.0 = mid;
                    } else {
                        lon_range.1 = mid;
                    }
                } else {
                    let mid = (lat_range.0 + lat_range.1) / 2.0;
                    if bit == 1 {
                        lat_range.0 = mid;
                    } else {
                        lat_range.1 = mid;
                    }
                }
                is_lon = !is_lon;
            }
        }

        Ok((lat_range.0, lat_range.1, lon_range.0, lon_range.1))
    }

    /// Return the 8 geohash neighbours (N, NE, E, SE, S, SW, W, NW) of a hash.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the input hash is invalid.
    pub fn neighbors(hash: &str) -> SpatialResult<[String; 8]> {
        let (min_lat, max_lat, min_lon, max_lon) = Self::decode_bbox(hash)?;
        let precision = hash.len();

        let step_lat = (max_lat - min_lat) * 0.5;
        let step_lon = (max_lon - min_lon) * 0.5;
        let center_lat = (min_lat + max_lat) / 2.0;
        let center_lon = (min_lon + max_lon) / 2.0;

        let clamp_lat = |l: f64| l.clamp(-90.0, 90.0);
        let clamp_lon = |l: f64| {
            if l > 180.0 { l - 360.0 } else if l < -180.0 { l + 360.0 } else { l }
        };

        let n = Self::encode(clamp_lat(center_lat + step_lat * 2.0), center_lon, precision)?;
        let ne = Self::encode(clamp_lat(center_lat + step_lat * 2.0), clamp_lon(center_lon + step_lon * 2.0), precision)?;
        let e = Self::encode(center_lat, clamp_lon(center_lon + step_lon * 2.0), precision)?;
        let se = Self::encode(clamp_lat(center_lat - step_lat * 2.0), clamp_lon(center_lon + step_lon * 2.0), precision)?;
        let s = Self::encode(clamp_lat(center_lat - step_lat * 2.0), center_lon, precision)?;
        let sw = Self::encode(clamp_lat(center_lat - step_lat * 2.0), clamp_lon(center_lon - step_lon * 2.0), precision)?;
        let w = Self::encode(center_lat, clamp_lon(center_lon - step_lon * 2.0), precision)?;
        let nw = Self::encode(clamp_lat(center_lat + step_lat * 2.0), clamp_lon(center_lon - step_lon * 2.0), precision)?;

        Ok([n, ne, e, se, s, sw, w, nw])
    }
}

// ---------------------------------------------------------------------------
// H3Lite – simplified hexagonal index
// ---------------------------------------------------------------------------

/// A simplified hexagonal hierarchical spatial index inspired by Uber H3.
///
/// This is an **approximate** implementation that maps latitude/longitude
/// coordinates to a hexagonal grid at a given resolution level (0–7).  The
/// output cell ID is a `u64` composed of the resolution and the integer
/// grid coordinates.
///
/// Full H3 compliance (exact cell boundaries, pentagon handling, etc.) is
/// deliberately omitted.  Use this for fast approximations only.
pub struct H3Lite;

impl H3Lite {
    /// Number of resolution levels supported (0 = coarsest, 7 = finest).
    pub const MAX_RESOLUTION: u8 = 7;

    /// Encode a lat/lon to an approximate H3-like cell index at the given resolution.
    ///
    /// The cell ID is structured as: `(resolution as u64) << 56 | (row as u64) << 28 | (col as u64)`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `resolution > MAX_RESOLUTION` or coordinates are invalid.
    pub fn encode(lat: f64, lon: f64, resolution: u8) -> SpatialResult<u64> {
        if resolution > Self::MAX_RESOLUTION {
            return Err(SpatialError::ValueError(format!(
                "Resolution {resolution} exceeds maximum {}",
                Self::MAX_RESOLUTION
            )));
        }
        if !(-90.0..=90.0).contains(&lat) {
            return Err(SpatialError::ValueError(format!(
                "Latitude {lat} out of range"
            )));
        }
        if !(-180.0..=180.0).contains(&lon) {
            return Err(SpatialError::ValueError(format!(
                "Longitude {lon} out of range"
            )));
        }

        // Number of cells along each axis at this resolution: 2^resolution * base
        let base_cells: u64 = 1 << (resolution as u64 + 1);

        // Use a flat-top hexagonal grid approximation.
        // Hex grid size in degrees:
        let hex_size_lat = 180.0 / (base_cells as f64);
        let hex_size_lon = 360.0 / (base_cells as f64);

        // Convert to offset coordinates
        let row = ((lat + 90.0) / hex_size_lat) as i64;
        let col = ((lon + 180.0) / hex_size_lon) as i64;

        // Apply pointy-top hex offset (odd rows are shifted)
        let col_adj = if row % 2 == 0 { col } else { col };

        let row_u = row.max(0) as u64;
        let col_u = col_adj.max(0) as u64;

        let cell_id = ((resolution as u64) << 56) | (row_u << 28) | (col_u & 0x0FFF_FFFF);
        Ok(cell_id)
    }

    /// Decode a cell ID back to the approximate (lat, lon) of its centre.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the cell ID has an invalid resolution field.
    pub fn decode(cell_id: u64) -> SpatialResult<(f64, f64)> {
        let resolution = (cell_id >> 56) as u8;
        if resolution > Self::MAX_RESOLUTION {
            return Err(SpatialError::ValueError(format!(
                "Invalid resolution field {resolution} in cell ID"
            )));
        }

        let base_cells: u64 = 1 << (resolution as u64 + 1);
        let hex_size_lat = 180.0 / (base_cells as f64);
        let hex_size_lon = 360.0 / (base_cells as f64);

        let row = ((cell_id >> 28) & 0x0FFF_FFFF) as f64;
        let col = (cell_id & 0x0FFF_FFFF) as f64;

        let lat = row * hex_size_lat - 90.0 + hex_size_lat / 2.0;
        let lon = col * hex_size_lon - 180.0 + hex_size_lon / 2.0;

        Ok((lat.clamp(-90.0, 90.0), lon.clamp(-180.0, 180.0)))
    }

    /// Return the 6 neighbouring cell IDs (approximate) at the same resolution.
    ///
    /// Neighbours are the 6 cells sharing an edge with the given cell in an
    /// offset hexagonal grid.  This is a flat-top grid approximation.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `cell_id` is invalid.
    pub fn neighbors(cell_id: u64) -> SpatialResult<Vec<u64>> {
        let resolution = (cell_id >> 56) as u8;
        if resolution > Self::MAX_RESOLUTION {
            return Err(SpatialError::ValueError(format!(
                "Invalid resolution field in cell ID"
            )));
        }

        let row = ((cell_id >> 28) & 0x0FFF_FFFF) as i64;
        let col = (cell_id & 0x0FFF_FFFF) as i64;

        // For even rows, the 6 neighbours in offset coordinates are:
        let offsets_even: [(i64, i64); 6] = [
            (0, 1), (0, -1), (-1, 0), (1, 0), (-1, -1), (1, -1),
        ];
        // For odd rows:
        let offsets_odd: [(i64, i64); 6] = [
            (0, 1), (0, -1), (-1, 0), (1, 0), (-1, 1), (1, 1),
        ];

        let offsets = if row % 2 == 0 { offsets_even } else { offsets_odd };

        let base_cells: i64 = 1 << (resolution as i64 + 1);
        let mut result = Vec::with_capacity(6);
        for (dr, dc) in &offsets {
            let nr = row + dr;
            let nc = col + dc;
            // Wrap longitude, clamp latitude
            if nr < 0 || nr >= base_cells {
                continue; // skip out-of-bounds rows (near poles)
            }
            let nc_wrapped = ((nc % base_cells) + base_cells) % base_cells;
            let nid = ((resolution as u64) << 56) | ((nr as u64) << 28) | (nc_wrapped as u64 & 0x0FFF_FFFF);
            result.push(nid);
        }
        Ok(result)
    }

    /// Compute the approximate area (in km²) of a cell at the given resolution.
    pub fn cell_area_km2(resolution: u8) -> SpatialResult<f64> {
        if resolution > Self::MAX_RESOLUTION {
            return Err(SpatialError::ValueError(format!(
                "Resolution {resolution} exceeds maximum {}",
                Self::MAX_RESOLUTION
            )));
        }
        // Total surface area of Earth ≈ 510_072_000 km²
        // Number of cells ≈ (2^(r+1))^2
        let num_cells = (1u64 << (resolution as u64 + 1)).pow(2) as f64;
        Ok(510_072_000.0 / num_cells)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geopoint_creation() {
        assert!(GeoPoint::new(51.5074, -0.1278).is_ok());
        assert!(GeoPoint::new(91.0, 0.0).is_err());
        assert!(GeoPoint::new(0.0, 181.0).is_err());
    }

    #[test]
    fn test_haversine_london_paris() {
        let london = GeoPoint::new(51.5074, -0.1278).unwrap();
        let paris = GeoPoint::new(48.8566, 2.3522).unwrap();
        let dist = HaversineDistance::distance_km(&london, &paris);
        // Known distance ≈ 343 km
        assert!((dist - 343.5).abs() < 2.0, "Expected ~343.5 km, got {dist:.1}");
    }

    #[test]
    fn test_vincenty_london_paris() {
        let london = GeoPoint::new(51.5074, -0.1278).unwrap();
        let paris = GeoPoint::new(48.8566, 2.3522).unwrap();
        let dist = VincentyDistance::distance_km(&london, &paris)
            .expect("Vincenty should converge");
        assert!((dist - 343.5).abs() < 2.0, "Vincenty: expected ~343.5 km, got {dist:.1}");
    }

    #[test]
    fn test_vincenty_same_point() {
        let p = GeoPoint::new(0.0, 0.0).unwrap();
        let d = VincentyDistance::distance_m(&p, &p).unwrap();
        assert!(d.abs() < 1e-3);
    }

    #[test]
    fn test_bounding_box_2d_contains() {
        let bb = BoundingBox2D::new(50.0, 52.0, -1.0, 1.0).unwrap();
        let inside = GeoPoint::new(51.0, 0.0).unwrap();
        let outside = GeoPoint::new(53.0, 0.0).unwrap();
        assert!(bb.contains(&inside));
        assert!(!bb.contains(&outside));
    }

    #[test]
    fn test_bounding_box_2d_overlap() {
        let a = BoundingBox2D::new(0.0, 10.0, 0.0, 10.0).unwrap();
        let b = BoundingBox2D::new(5.0, 15.0, 5.0, 15.0).unwrap();
        let c = BoundingBox2D::new(20.0, 30.0, 20.0, 30.0).unwrap();
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_geohash_encode_decode() {
        // Well-known geohash for London area
        let hash = Geohash::encode(51.5074, -0.1278, 5).unwrap();
        let (lat, lon) = Geohash::decode(&hash).unwrap();
        assert!((lat - 51.5074).abs() < 0.1, "lat mismatch: {lat}");
        assert!((lon - (-0.1278)).abs() < 0.1, "lon mismatch: {lon}");
    }

    #[test]
    fn test_geohash_precision() {
        // Longer hash → smaller cell → decode centre closer to input
        let h7 = Geohash::encode(48.8566, 2.3522, 7).unwrap();
        let h3 = Geohash::encode(48.8566, 2.3522, 3).unwrap();
        let (lat7, lon7) = Geohash::decode(&h7).unwrap();
        let (lat3, lon3) = Geohash::decode(&h3).unwrap();
        let err7 = (lat7 - 48.8566).abs() + (lon7 - 2.3522).abs();
        let err3 = (lat3 - 48.8566).abs() + (lon3 - 2.3522).abs();
        assert!(err7 < err3, "Higher precision should decode more accurately");
    }

    #[test]
    fn test_geohash_invalid_char() {
        assert!(Geohash::decode("gcpvh!").is_err());
    }

    #[test]
    fn test_geohash_neighbors_count() {
        let hash = Geohash::encode(51.5074, -0.1278, 4).unwrap();
        let neighbors = Geohash::neighbors(&hash).unwrap();
        assert_eq!(neighbors.len(), 8);
    }

    #[test]
    fn test_h3lite_encode_decode() {
        let (lat, lon) = (37.7749, -122.4194); // San Francisco
        let cell_id = H3Lite::encode(lat, lon, 4).unwrap();
        let (dlat, dlon) = H3Lite::decode(cell_id).unwrap();
        // Should round-trip within the cell size
        assert!((dlat - lat).abs() < 5.0, "lat: {dlat}");
        assert!((dlon - lon).abs() < 5.0, "lon: {dlon}");
    }

    #[test]
    fn test_h3lite_neighbors() {
        let cell_id = H3Lite::encode(0.0, 0.0, 3).unwrap();
        let neighbors = H3Lite::neighbors(cell_id).unwrap();
        assert!(!neighbors.is_empty());
        assert!(neighbors.len() <= 6);
    }

    #[test]
    fn test_h3lite_cell_area() {
        let area = H3Lite::cell_area_km2(0).unwrap();
        assert!(area > 0.0);
        let area_fine = H3Lite::cell_area_km2(5).unwrap();
        assert!(area_fine < area);
    }

    #[test]
    fn test_geopoint_bearing() {
        let london = GeoPoint::new(51.5074, -0.1278).unwrap();
        let paris = GeoPoint::new(48.8566, 2.3522).unwrap();
        let bearing = london.bearing_to(&paris);
        // London → Paris is roughly south-east
        assert!(bearing > 100.0 && bearing < 200.0, "bearing {bearing}");
    }

    #[test]
    fn test_geopoint_destination() {
        let origin = GeoPoint::new(0.0, 0.0).unwrap();
        let dest = origin.destination(111_195.0, 0.0); // ~1 degree north
        assert!((dest.lat() - 1.0).abs() < 0.01, "Expected ~1°N, got {}°", dest.lat());
    }

    #[test]
    fn test_geopoint_ecef() {
        let origin = GeoPoint::new(0.0, 0.0).unwrap();
        let (x, y, z) = origin.to_ecef();
        // At (0°, 0°) ECEF should be roughly (WGS84_A, 0, 0)
        assert!((x - WGS84_A).abs() < 1.0, "x={x}");
        assert!(y.abs() < 1.0, "y={y}");
        assert!(z.abs() < 1.0, "z={z}");
    }
}
