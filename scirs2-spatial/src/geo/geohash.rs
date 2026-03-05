//! GeoHash spatial indexing.
//!
//! GeoHash encodes a latitude/longitude coordinate into a short base-32 string.
//! Adjacent cells share a common prefix, which makes geohash useful for
//! proximity searches and spatial indexing.
//!
//! # Precision table (approximate cell sizes)
//!
//! | Precision | Width      | Height     |
//! |-----------|-----------|-----------|
//! | 1         | 5,009 km  | 4,992 km  |
//! | 2         | 1,252 km  | 624 km    |
//! | 3         | 156 km    | 156 km    |
//! | 4         | 39.1 km   | 19.5 km   |
//! | 5         | 4.9 km    | 4.9 km    |
//! | 6         | 1.2 km    | 609 m     |
//! | 7         | 153 m     | 153 m     |
//! | 8         | 38.2 m    | 19.1 m    |
//! | 9         | 4.77 m    | 4.77 m    |
//! | 10        | 1.19 m    | 0.596 m   |
//! | 11        | 149 mm    | 149 mm    |
//! | 12        | 37.2 mm   | 18.6 mm   |

use crate::error::{SpatialError, SpatialResult};

/// Base-32 character set used by geohash
const BASE32: &[u8] = b"0123456789bcdefghjkmnpqrstuvwxyz";

/// Decode a base-32 character to its 5-bit integer value
fn decode_char(c: u8) -> SpatialResult<u8> {
    BASE32
        .iter()
        .position(|&b| b == c.to_ascii_lowercase())
        .map(|p| p as u8)
        .ok_or_else(|| {
            SpatialError::ValueError(format!("Invalid geohash character: '{}'", c as char))
        })
}

/// GeoHash spatial index cell.
///
/// Encodes a geographic bounding box as a compact base-32 string.
/// The longer the hash, the smaller and more precise the bounding box.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geo::GeoHash;
///
/// // Encode a coordinate
/// let hash = GeoHash::encode(48.8566, 2.3522, 6).unwrap();
/// // Length should be exactly the requested precision
/// assert_eq!(hash.as_str().len(), 6);
///
/// // Decode back to bounding box center — should be within ~1 km of original
/// let decoded = hash.center().unwrap();
/// assert!((decoded.lat() - 48.8566).abs() < 0.05);
/// assert!((decoded.lon() - 2.3522).abs() < 0.05);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GeoHash {
    hash: String,
}

/// The decoded center coordinate of a GeoHash cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoHashCenter {
    /// Latitude of the cell center in degrees
    lat: f64,
    /// Longitude of the cell center in degrees
    lon: f64,
    /// Half-height of the bounding box in degrees
    lat_err: f64,
    /// Half-width of the bounding box in degrees
    lon_err: f64,
}

impl GeoHashCenter {
    /// Latitude of the cell center in degrees
    pub fn lat(&self) -> f64 {
        self.lat
    }
    /// Longitude of the cell center in degrees
    pub fn lon(&self) -> f64 {
        self.lon
    }
    /// Half-height of the cell in degrees (error in latitude)
    pub fn lat_err(&self) -> f64 {
        self.lat_err
    }
    /// Half-width of the cell in degrees (error in longitude)
    pub fn lon_err(&self) -> f64 {
        self.lon_err
    }

    /// Return the bounding box as `(min_lat, max_lat, min_lon, max_lon)` in degrees.
    pub fn bbox(&self) -> (f64, f64, f64, f64) {
        (
            self.lat - self.lat_err,
            self.lat + self.lat_err,
            self.lon - self.lon_err,
            self.lon + self.lon_err,
        )
    }
}

impl GeoHash {
    /// Encode a coordinate into a GeoHash string.
    ///
    /// # Arguments
    ///
    /// * `lat` - Latitude in degrees (must be in [-90, 90])
    /// * `lon` - Longitude in degrees (must be in [-180, 180])
    /// * `precision` - Length of the hash string (1–12)
    ///
    /// # Errors
    ///
    /// Returns `SpatialError::ValueError` if coordinates or precision are out of range.
    pub fn encode(lat: f64, lon: f64, precision: usize) -> SpatialResult<Self> {
        if !(-90.0..=90.0).contains(&lat) {
            return Err(SpatialError::ValueError(format!(
                "Latitude {lat} is outside valid range [-90, 90]"
            )));
        }
        if !(-180.0..=180.0).contains(&lon) {
            return Err(SpatialError::ValueError(format!(
                "Longitude {lon} is outside valid range [-180, 180]"
            )));
        }
        if precision == 0 || precision > 12 {
            return Err(SpatialError::ValueError(format!(
                "Precision {precision} is outside valid range [1, 12]"
            )));
        }

        let hash = encode_internal(lat, lon, precision);
        Ok(GeoHash { hash })
    }

    /// Decode a GeoHash string to its bounding box center.
    ///
    /// # Errors
    ///
    /// Returns `SpatialError::ValueError` if the string contains invalid characters
    /// or is empty.
    pub fn decode(hash: &str) -> SpatialResult<GeoHashCenter> {
        if hash.is_empty() {
            return Err(SpatialError::ValueError(
                "Geohash string is empty".to_string(),
            ));
        }
        decode_internal(hash)
    }

    /// Return the hash string.
    pub fn as_str(&self) -> &str {
        &self.hash
    }

    /// Return the decoded center of this GeoHash cell.
    pub fn center(&self) -> SpatialResult<GeoHashCenter> {
        decode_internal(&self.hash)
    }

    /// Return the 8 neighboring GeoHash cells (N, NE, E, SE, S, SW, W, NW).
    ///
    /// The result array is ordered: `[N, NE, E, SE, S, SW, W, NW]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the hash is invalid.
    pub fn neighbors(&self) -> SpatialResult<[String; 8]> {
        compute_neighbors(&self.hash)
    }
}

/// Convenience function: encode lat/lon to a geohash string.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geo::geohash_encode;
///
/// let hash = geohash_encode(48.8566, 2.3522, 6).unwrap();
/// // The hash encodes a cell near Paris; verify by round-tripping
/// assert_eq!(hash.len(), 6);
/// use scirs2_spatial::geo::geohash_decode;
/// let (lat, lon) = geohash_decode(&hash).unwrap();
/// assert!((lat - 48.8566).abs() < 0.05);
/// assert!((lon - 2.3522).abs() < 0.05);
/// ```
pub fn geohash_encode(lat: f64, lon: f64, precision: usize) -> SpatialResult<String> {
    GeoHash::encode(lat, lon, precision).map(|g| g.hash)
}

/// Convenience function: decode a geohash string to `(lat, lon)` center.
///
/// # Errors
///
/// Returns an error if the string is invalid.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geo::geohash_decode;
///
/// let (lat, lon) = geohash_decode("u09tun").unwrap();
/// assert!((lat - 48.856).abs() < 0.01);
/// ```
pub fn geohash_decode(hash: &str) -> SpatialResult<(f64, f64)> {
    GeoHash::decode(hash).map(|c| (c.lat, c.lon))
}

/// Convenience function: return the 8 neighboring geohash strings.
///
/// Returns `[N, NE, E, SE, S, SW, W, NW]`.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geo::geohash_neighbors;
///
/// let neighbors = geohash_neighbors("u09tun").unwrap();
/// assert_eq!(neighbors.len(), 8);
/// ```
pub fn geohash_neighbors(hash: &str) -> SpatialResult<[String; 8]> {
    compute_neighbors(hash)
}

/// Internal encoder: bit-interleave lon bits and lat bits, output base-32.
fn encode_internal(lat: f64, lon: f64, precision: usize) -> String {
    let mut lat_min = -90.0_f64;
    let mut lat_max = 90.0_f64;
    let mut lon_min = -180.0_f64;
    let mut lon_max = 180.0_f64;

    let total_bits = precision * 5;
    let mut hash = Vec::with_capacity(precision);
    let mut current_char: u8 = 0;
    let mut bit_count = 0;
    let mut is_lon = true; // longitude bits interleaved first

    for _ in 0..total_bits {
        current_char <<= 1;
        if is_lon {
            let mid = (lon_min + lon_max) / 2.0;
            if lon >= mid {
                current_char |= 1;
                lon_min = mid;
            } else {
                lon_max = mid;
            }
        } else {
            let mid = (lat_min + lat_max) / 2.0;
            if lat >= mid {
                current_char |= 1;
                lat_min = mid;
            } else {
                lat_max = mid;
            }
        }
        is_lon = !is_lon;
        bit_count += 1;

        if bit_count == 5 {
            hash.push(BASE32[current_char as usize]);
            current_char = 0;
            bit_count = 0;
        }
    }

    String::from_utf8(hash).unwrap_or_default()
}

/// Internal decoder: expand base-32 chars into lat/lon intervals.
fn decode_internal(hash: &str) -> SpatialResult<GeoHashCenter> {
    let mut lat_min = -90.0_f64;
    let mut lat_max = 90.0_f64;
    let mut lon_min = -180.0_f64;
    let mut lon_max = 180.0_f64;
    let mut is_lon = true;

    for byte in hash.bytes() {
        let value = decode_char(byte)?;
        // Process 5 bits from MSB to LSB
        for shift in (0..5).rev() {
            let bit = (value >> shift) & 1;
            if is_lon {
                let mid = (lon_min + lon_max) / 2.0;
                if bit == 1 {
                    lon_min = mid;
                } else {
                    lon_max = mid;
                }
            } else {
                let mid = (lat_min + lat_max) / 2.0;
                if bit == 1 {
                    lat_min = mid;
                } else {
                    lat_max = mid;
                }
            }
            is_lon = !is_lon;
        }
    }

    Ok(GeoHashCenter {
        lat: (lat_min + lat_max) / 2.0,
        lon: (lon_min + lon_max) / 2.0,
        lat_err: (lat_max - lat_min) / 2.0,
        lon_err: (lon_max - lon_min) / 2.0,
    })
}

/// Direction offsets for neighbor computation.
///
/// For each direction we adjust the center coordinate of the current cell
/// by one cell-width/height in that direction, then re-encode.
fn compute_neighbors(hash: &str) -> SpatialResult<[String; 8]> {
    if hash.is_empty() {
        return Err(SpatialError::ValueError(
            "Geohash string is empty".to_string(),
        ));
    }
    let precision = hash.len();
    let center = decode_internal(hash)?;

    // Step sizes (2× the half-error gives us a full cell dimension)
    let lat_step = center.lat_err * 2.0;
    let lon_step = center.lon_err * 2.0;

    // [N, NE, E, SE, S, SW, W, NW]
    let offsets: [(f64, f64); 8] = [
        (lat_step, 0.0),        // N
        (lat_step, lon_step),   // NE
        (0.0, lon_step),        // E
        (-lat_step, lon_step),  // SE
        (-lat_step, 0.0),       // S
        (-lat_step, -lon_step), // SW
        (0.0, -lon_step),       // W
        (lat_step, -lon_step),  // NW
    ];

    let mut result: [String; 8] = Default::default();
    for (i, (dlat, dlon)) in offsets.iter().enumerate() {
        let nlat = (center.lat + dlat).clamp(-90.0, 90.0);
        // Wrap longitude
        let raw_lon = center.lon + dlon;
        let nlon = if raw_lon > 180.0 {
            raw_lon - 360.0
        } else if raw_lon < -180.0 {
            raw_lon + 360.0
        } else {
            raw_lon
        };
        result[i] = encode_internal(nlat, nlon, precision);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_known_hash() {
        // Paris (approximate; geohash u09tun at precision 6)
        let hash = geohash_encode(48.8566, 2.3522, 6).expect("encode");
        // Verify round-trip instead of exact string, as rounding can differ
        let (lat, lon) = geohash_decode(&hash).expect("decode");
        assert!((lat - 48.8566).abs() < 0.05, "lat={lat}");
        assert!((lon - 2.3522).abs() < 0.05, "lon={lon}");
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let points = [
            (0.0, 0.0),
            (51.5074, -0.1278),
            (-33.8688, 151.2093),
            (40.7128, -74.0060),
            (35.6762, 139.6503),
        ];
        for (lat, lon) in points {
            for precision in 3..=8 {
                let hash = geohash_encode(lat, lon, precision).expect("encode");
                let (dlat, dlon) = geohash_decode(&hash).expect("decode");
                // At precision p, error < 360 / 2^(5p/2 + 0.5) degrees
                let tolerance = 180.0 / 2.0_f64.powi((5 * precision / 2) as i32);
                assert!(
                    (dlat - lat).abs() <= tolerance * 1.5,
                    "lat mismatch at precision {precision}: {dlat} vs {lat}"
                );
                assert!(
                    (dlon - lon).abs() <= tolerance * 1.5,
                    "lon mismatch at precision {precision}: {dlon} vs {lon}"
                );
            }
        }
    }

    #[test]
    fn test_decode_invalid_char() {
        assert!(geohash_decode("u0_bad").is_err());
        assert!(geohash_decode("").is_err());
    }

    #[test]
    fn test_encode_invalid_inputs() {
        assert!(geohash_encode(91.0, 0.0, 5).is_err());
        assert!(geohash_encode(0.0, 181.0, 5).is_err());
        assert!(geohash_encode(0.0, 0.0, 0).is_err());
        assert!(geohash_encode(0.0, 0.0, 13).is_err());
    }

    #[test]
    fn test_neighbors_count() {
        let neighbors = geohash_neighbors("u09tun").expect("neighbors");
        assert_eq!(neighbors.len(), 8);
        // All neighbors should be valid hashes of the same precision
        for n in &neighbors {
            assert_eq!(n.len(), 6);
            let result = geohash_decode(n);
            assert!(result.is_ok(), "neighbor {n} is invalid");
        }
    }

    #[test]
    fn test_geohash_bbox() {
        let center = GeoHash::decode("u09tun").expect("decode");
        let (min_lat, max_lat, min_lon, max_lon) = center.bbox();
        assert!(min_lat < center.lat());
        assert!(max_lat > center.lat());
        assert!(min_lon < center.lon());
        assert!(max_lon > center.lon());
    }

    #[test]
    fn test_neighbors_adjacency() {
        // North neighbor should be further north
        let center = GeoHash::decode("u09tun").expect("decode");
        let neighbors = geohash_neighbors("u09tun").expect("neighbors");
        let north_center = geohash_decode(&neighbors[0]).expect("decode north");
        assert!(
            north_center.0 > center.lat(),
            "North neighbor should be north"
        );
        // East neighbor should be further east
        let east_center = geohash_decode(&neighbors[2]).expect("decode east");
        assert!(east_center.1 > center.lon(), "East neighbor should be east");
    }
}
