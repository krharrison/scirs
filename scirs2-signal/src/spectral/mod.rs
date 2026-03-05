//! Advanced Spectral Analysis Methods
//!
//! This module provides specialised spectral estimation algorithms:
//!
//! | Sub-module | Contents |
//! |------------|---------|
//! | [`multitaper`] | DPSS windows, multitaper PSD, adaptive weighting, F-test |
//! | [`lomb_scargle`] | Periodogram for unevenly sampled data, FAP estimation |
//! | [`parametric`] | AR/ARMA PSD, Burg, Yule-Walker, MUSIC pseudospectrum |
//! | [`reassignment`] | Reassigned spectrogram, synchrosqueezed STFT |
//! | [`wavelet_packet`] | Wavelet packet transform, best-basis selection, energy map |
//!
//! The `legacy` sub-module preserves the original `periodogram`, `welch`,
//! `stft`, and `spectrogram` functions.
//!
//! # Quick Examples
//!
//! ## Multitaper PSD
//!
//! ```rust
//! use scirs2_signal::spectral::multitaper::multitaper_psd;
//!
//! let n = 256usize;
//! let fs = 256.0f64;
//! let x: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 40.0 * i as f64 / fs).sin()).collect();
//! let (freqs, psd) = multitaper_psd(&x, fs, 4.0, 7).expect("multitaper_psd failed");
//! assert!(!psd.is_empty());
//! ```
//!
//! ## Lomb-Scargle Periodogram
//!
//! ```rust
//! use scirs2_signal::spectral::lomb_scargle::lomb_scargle_auto;
//!
//! let t: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
//! let y: Vec<f64> = t.iter().map(|&ti| (2.0 * std::f64::consts::PI * 2.0 * ti).sin()).collect();
//! let (freqs, power) = lomb_scargle_auto(&t, &y, 200).expect("lomb_scargle_auto failed");
//! assert_eq!(freqs.len(), 200);
//! ```
//!
//! ## Wavelet Packet Transform
//!
//! ```rust
//! use scirs2_signal::spectral::wavelet_packet::{decompose, best_basis, WaveletType, CostFunction};
//!
//! let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.3).sin()).collect();
//! let wp = decompose(&signal, WaveletType::Haar, 3).expect("decompose");
//! let basis = best_basis(&wp, CostFunction::Shannon).expect("best_basis");
//! assert!(!basis.is_empty());
//! ```

// New specialised sub-modules
pub mod lomb_scargle;
pub mod multitaper;
pub mod parametric;
pub mod reassignment;
pub mod wavelet_packet;

// Legacy module: preserves the original periodogram / welch / stft / spectrogram API
pub mod legacy;

// ---------------------------------------------------------------------------
// Re-exports: legacy (used by lib.rs `pub use spectral::{...}`)
// ---------------------------------------------------------------------------
pub use legacy::{get_window_simd_ultra, periodogram, spectrogram, stft, welch};

// ---------------------------------------------------------------------------
// Convenience re-exports: new algorithms
// ---------------------------------------------------------------------------

// --- Multitaper ---
pub use multitaper::{adaptive_multitaper_psd, dpss_windows, f_test_statistic, multitaper_psd};

// --- Lomb-Scargle ---
pub use lomb_scargle::{
    false_alarm_probability, fast_lomb_scargle, lomb_scargle as lomb_scargle_power,
    lomb_scargle_auto, significance_level,
};

// --- Parametric ---
pub use parametric::{ar_psd, arma_psd, burg_method, music_algorithm, yule_walker};

// --- Reassignment ---
pub use reassignment::{reassigned_spectrogram, synchrosqueezed_stft};

// --- Wavelet packet ---
pub use wavelet_packet::{
    best_basis, decompose as wpt_decompose, energy_map, reconstruct as wpt_reconstruct,
    CostFunction, WaveletPacket, WaveletType,
};
