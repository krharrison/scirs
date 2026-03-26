//! Phase estimation: ESPRIT, MUSIC, and instantaneous frequency.
//!
//! This module provides high-resolution spectral estimation algorithms for
//! identifying the frequencies, amplitudes, and phases of sinusoidal components
//! embedded in a signal.
//!
//! # Algorithms
//!
//! | Algorithm | Type | Best For |
//! |-----------|------|----------|
//! | [`esprit::EspritEstimator`] | Subspace | Narrow-band tones, high SNR |
//! | [`music::MusicEstimator`] | Subspace pseudospectrum | Multiple closely-spaced tones |
//! | [`instantaneous::instantaneous_frequency`] | Hilbert | Wideband / single-component IF |
//! | [`instantaneous::teager_kaiser_if`] | TKEO | Fast, low-complexity IF tracking |
//!
//! # Example
//!
//! ```rust
//! use scirs2_signal::phase_estimation::{EspritEstimator, MusicEstimator};
//! use scirs2_signal::phase_estimation::instantaneous_frequency;
//!
//! let fs = 1000.0_f64;
//! let n = 256_usize;
//! let signal: Vec<f64> = (0..n)
//!     .map(|i| (2.0 * std::f64::consts::PI * 100.0 / fs * i as f64).sin())
//!     .collect();
//!
//! // ESPRIT
//! let est = EspritEstimator::new(1, fs);
//! let result = est.estimate(&signal).expect("ESPRIT failed");
//! println!("ESPRIT: {:.2} Hz", result.components[0].frequency);
//!
//! // Instantaneous frequency
//! let if_track = instantaneous_frequency(&signal, fs);
//! println!("IF mean: {:.2} Hz", if_track.mean());
//! ```

pub mod esprit;
pub mod instantaneous;
pub mod music;
pub mod types;

// Convenient flat re-exports.
pub use esprit::EspritEstimator;
pub use instantaneous::{
    hilbert_transform, instantaneous_frequency, instantaneous_phase, phase_unwrap,
    teager_kaiser_energy, teager_kaiser_if,
};
pub use music::MusicEstimator;
pub use types::{
    FrequencyComponent, InstantaneousFreq, PhaseEstConfig, PhaseEstResult, PhaseMethod,
};
