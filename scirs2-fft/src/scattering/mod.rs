//! Wavelet Scattering Transform
//!
//! Implementation of Mallat's scattering transform (2012), which provides
//! translation-invariant and deformation-stable signal representations.
//!
//! The scattering transform cascades wavelet convolutions with modulus
//! nonlinearities and low-pass averaging to produce feature vectors that are:
//! - Translation invariant up to scale 2^J
//! - Lipschitz continuous with respect to deformations
//! - Information preserving (captures modulation structure via higher orders)
//!
//! # Architecture
//!
//! - [`FilterBank`]: Constructs Morlet wavelet filter banks in the frequency domain
//! - [`ScatteringTransform`]: Computes zeroth, first, and second-order scattering coefficients
//! - [`ScatteringFeatures`]: Extracts and normalizes feature vectors for downstream tasks
//!
//! # References
//!
//! - Mallat, S. (2012). Group invariant scattering. Communications on Pure and Applied Mathematics.
//! - Andén, J. & Mallat, S. (2014). Deep scattering spectrum. IEEE Trans. Signal Processing.

mod features;
mod filter_bank;
#[allow(clippy::module_inception)]
mod scattering;

pub use features::{
    FeatureNormalization, JointScatteringFeatures, ScatteringFeatureExtractor, ScatteringFeatures,
    TimeFrequencyMode,
};
pub use filter_bank::{FilterBank, FilterBankConfig, MorletWavelet};
pub use scattering::{
    ScatteringCoefficients, ScatteringConfig, ScatteringOrder, ScatteringResult,
    ScatteringTransform,
};
