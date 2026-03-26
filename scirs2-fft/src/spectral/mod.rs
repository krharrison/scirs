//! Advanced spectral analysis methods.
//!
//! This module provides:
//!
//! - **Cyclostationary analysis**: Spectral Correlation Function (SCF), cyclic spectral density,
//!   spectral coherence, and cyclic frequency detection via the FFT Accumulation Method (FAM).
//! - **Short-Time Fractional Fourier Transform (STFRFT)**: Fractional Fourier Transform with
//!   time-localization via windowing, using the Ozaktas decomposition algorithm.
//! - **Ambiguity function**: Narrowband and wideband ambiguity functions for radar/sonar
//!   signal analysis, including auto- and cross-ambiguity computations.

pub mod ambiguity;
pub mod cyclostationary;
pub mod stfrft;

pub use ambiguity::{
    auto_ambiguity, auto_ambiguity_surface, cross_ambiguity, cross_ambiguity_surface,
    wideband_ambiguity, AmbiguityConfig, AmbiguityResult, WidebandAmbiguityResult,
};
pub use cyclostationary::{
    cyclic_frequency_detection, cyclic_spectral_density, spectral_coherence_cyclic,
    spectral_correlation_function, CyclostationaryConfig, CyclostationaryResult,
};
pub use stfrft::{frft as spectral_frft, ifrft, stfrft, FrftConfig, StfrftConfig, StfrftResult};
