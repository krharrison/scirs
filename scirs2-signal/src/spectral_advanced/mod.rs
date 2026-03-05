//! Enhanced spectral analysis methods
//!
//! This module provides advanced spectral estimation algorithms including:
//!
//! ## Multitaper Spectral Estimation (Thomson's method)
//! - DPSS (discrete prolate spheroidal sequences) windows
//! - Adaptive weighting (Thomson 1982)
//! - F-test for line component detection
//!
//! ## Lomb-Scargle Periodogram
//! - Standard and generalized versions for unevenly-sampled data
//! - False alarm probability estimation (Baluev, Davies, Naive methods)
//! - Floating mean correction
//!
//! ## Parametric Spectral Methods
//! - Burg's method (maximum entropy spectral estimation)
//! - Yule-Walker AR estimation with Levinson-Durbin recursion
//! - MUSIC (Multiple Signal Classification) — subspace-based super-resolution
//! - ESPRIT (Estimation of Signal Parameters via Rotational Invariance)
//!
//! # Example
//!
//! ```
//! use scirs2_signal::spectral_advanced::{burg_spectral, BurgConfig};
//!
//! // Estimate PSD using Burg's method
//! let signal: Vec<f64> = (0..256).map(|i| {
//!     (2.0 * std::f64::consts::PI * 30.0 * i as f64 / 256.0).sin()
//! }).collect();
//!
//! let config = BurgConfig { order: 10, fs: 256.0, nfft: 512 };
//! let result = burg_spectral(&signal, &config).expect("Burg failed");
//! assert!(!result.psd.is_empty());
//! ```

pub mod lomb_scargle;
pub mod multitaper_est;
pub mod parametric_methods;
pub mod types;

// Re-export all public types
pub use types::*;

// Re-export multitaper functions
pub use multitaper_est::{multitaper_ftest_line_detection, multitaper_psd};

// Re-export Lomb-Scargle functions
pub use lomb_scargle::{false_alarm_level, false_alarm_probability, lomb_scargle_periodogram};

// Re-export parametric methods
pub use parametric_methods::{
    burg_spectral, esprit_spectral, music_spectral, yule_walker_spectral,
};
