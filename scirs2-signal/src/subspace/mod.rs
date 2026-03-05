//! Subspace methods for DOA estimation and high-resolution spectral analysis
//!
//! This module provides a comprehensive suite of subspace-based methods for:
//! - **Direction-of-Arrival (DOA) estimation** from sensor array data
//! - **High-resolution spectral analysis** of time series
//!
//! # Module Structure
//!
//! | Submodule | Description |
//! |-----------|-------------|
//! | [`array_processing`] | Array manifolds, covariance estimation, source number detection |
//! | [`music`] | MUSIC, Root-MUSIC, SS-MUSIC (spatially smoothed MUSIC) |
//! | [`esprit`] | ESPRIT-1D, TLS-ESPRIT, ESPRIT-2D for planar arrays |
//! | [`pisarenko`] | Pisarenko, Min-Norm, MVDR/Capon, Forward-Backward averaging |
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_signal::subspace::{
//!     array_processing::UniformLinearArray,
//!     music::{MUSICEstimator, MUSICConfig},
//! };
//! use scirs2_core::numeric::Complex64;
//! use std::f64::consts::PI;
//!
//! // Build a 8-element ULA
//! let ula = UniformLinearArray::new(8, 0.5).expect("ula");
//!
//! // Steering vector at broadside (0°)
//! let sv = ula.steering_vector(0.0).expect("sv");
//! assert_eq!(sv.len(), 8);
//! ```

pub mod array_processing;
pub mod esprit;
pub mod music;
pub mod pisarenko;

// Convenience re-exports

// Array processing
pub use array_processing::{
    ArrayManifold, SourceNumberEstimation, SpatialCovariance, UniformCircularArray,
    UniformLinearArray,
};

// MUSIC variants
pub use music::{
    music_spectral, MUSICConfig, MUSICEstimator, MUSICResult, RootMUSIC, RootMUSICResult, SSMUSIC,
};

// ESPRIT variants
pub use esprit::{ESPRIT1D, ESPRIT2D, ESPRIT2DResult, ESPRITResult, TlsEsprit};

// Classical methods
pub use pisarenko::{
    Capon, ForwardBackward, MVDRMode, MVDRResult, MinNorm, MinNormMode, MinNormResult, MVDR,
    Pisarenko, PisarenkoResult,
};
