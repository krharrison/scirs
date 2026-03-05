//! Wavelet Packet Decomposition and Multiresolution Analysis
//!
//! This module provides a comprehensive implementation of:
//!
//! - **Wavelet Packet Transform (WPT)**: Full binary-tree decomposition where
//!   both approximation and detail subbands are recursively split, unlike the
//!   standard DWT that only recurses on approximations.
//! - **Multiresolution Analysis (MRA)**: Mallat's fast algorithm, two-channel
//!   filter banks, and perfect reconstruction verification.
//! - **Lifting Scheme**: In-place second-generation wavelet transforms including
//!   Haar, CDF 9/7 (JPEG 2000), and data-adaptive variants.
//!
//! # Module Layout
//!
//! | Sub-module | Highlights |
//! |------------|-----------|
//! | [`wpt`]       | `WaveletPacketTree`, best-basis selection, cost functions |
//! | [`mra`]       | `MRAFilter`, Mallat algorithm, PR verification |
//! | [`lifting`]   | `LiftingWavelet`, Haar, CDF 9/7, adaptive lifting |
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_signal::wavelet_packets::wpt::{WaveletPacketTree, CostMeasure};
//! use scirs2_signal::dwt::Wavelet;
//!
//! let signal: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
//! let tree = WaveletPacketTree::build(&signal, Wavelet::DB(4), 4, "symmetric")
//!     .expect("tree build failed");
//! let basis = tree.best_basis(CostMeasure::Shannon).expect("best basis failed");
//! println!("best basis has {} leaves", basis.len());
//! ```

pub mod lifting;
pub mod mra;
pub mod wpt;

// Convenience re-exports
pub use lifting::{
    adaptive_lifting, cdf97_lifting, forward_lifting, haar_lifting, inverse_lifting,
    LiftingStep, LiftingWavelet,
};
pub use mra::{
    mallat_algorithm, mallat_reconstruct, perfect_reconstruction_check, two_channel_filter_bank,
    upsampling_convolution, MRAFilter,
};
pub use wpt::{
    best_basis_selection, wpt_cost_function, wpt_decompose, wpt_reconstruct, CostMeasure,
    WaveletPacketNode, WaveletPacketTree,
};
