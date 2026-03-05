//! # Geospatial and Geodesy Utilities
//!
//! This module provides a comprehensive set of geospatial and geodetic
//! primitives for the SciRS2 scientific computing ecosystem.
//!
//! | Submodule | Contents |
//! |-----------|---------|
//! | [`coordinates`] | Geographic/ECEF/ENU coordinate systems, Vincenty/Haversine distances |
//! | [`projection`]  | Mercator, Transverse Mercator, UTM, Azimuthal Equidistant, Lambert Conic |
//!
//! ## Design Principles
//!
//! * All angular values are in **decimal degrees** unless explicitly noted.
//! * All linear values are in **metres**.
//! * The **WGS-84** ellipsoid is used throughout.
//! * Every operation that can fail returns a [`GeoResult`] / [`GeoError`].
//! * No external geospatial crate dependencies — pure Rust implementation.
//!
//! ## Quick Start
//!
//! ### Coordinate conversion and geodetic distance
//!
//! ```rust
//! use scirs2_core::geo::coordinates::{GeographicCoord, EcefCoord};
//!
//! let london = GeographicCoord::new(51.5074, -0.1278, 0.0);
//! let paris  = GeographicCoord::new(48.8566,  2.3522, 0.0);
//!
//! // Haversine distance (fast, spherical approximation)
//! let hav_m = london.haversine_distance(&paris);
//! assert!((hav_m - 340_000.0).abs() < 5_000.0);
//!
//! // Vincenty distance (ellipsoidal, ~0.5 mm accuracy)
//! let vin_m = london.vincenty_distance(&paris).expect("converges");
//! assert!((vin_m - 340_000.0).abs() < 5_000.0);
//!
//! // ECEF round-trip
//! let ecef      = london.to_ecef();
//! let recovered = ecef.to_geographic();
//! assert!((recovered.lat - london.lat).abs() < 1e-9);
//!
//! // ENU: point relative to an origin
//! let enu = paris.to_enu(&london);
//! println!("Paris from London: east={:.0}m, north={:.0}m, up={:.0}m",
//!          enu.east, enu.north, enu.up);
//! ```
//!
//! ### Map projections
//!
//! ```rust
//! use scirs2_core::geo::projection::{mercator_forward, mercator_inverse, to_utm, from_utm};
//!
//! // Mercator: equator maps to y = 0
//! let (_, y) = mercator_forward(0.0, 0.0, 0.0);
//! assert!(y.abs() < 1e-6);
//!
//! // UTM round-trip (New York City)
//! let (zone, band, e, n) = to_utm(40.7128, -74.0060).expect("to_utm");
//! assert_eq!(zone, 18);
//! let (lat2, lon2) = from_utm(zone, band, e, n).expect("from_utm");
//! assert!((lat2 - 40.7128).abs() < 1e-5);
//! assert!((lon2 - (-74.0060)).abs() < 1e-5);
//! ```

pub mod coordinates;
pub mod projection;

// Flat re-exports so callers can use `scirs2_core::geo::GeoError` etc. without
// needing to drill into the submodule.
pub use coordinates::{
    EcefCoord, EnuCoord, GeographicCoord, GeoError, GeoResult, WGS84_A, WGS84_B, WGS84_E2,
    WGS84_F,
};
pub use projection::{
    azimuthal_equidistant, from_utm, lambert_conic_forward, mercator_forward, mercator_inverse,
    to_utm, transverse_mercator_forward, transverse_mercator_inverse, utm_band, utm_zone,
};
