//! Medical imaging utilities for SciRS2-NDImage.
//!
//! This module provides lightweight tools for working with medical images:
//!
//! - [`dicom_lite`]: minimal DICOM-like metadata, bias correction, HU
//!   classification, window/level display adjustment, and per-tissue statistics.
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_ndimage::medical::dicom_lite::{
//!     DicomHeader, HounsfieldUnits, MedicalVolume, WindowLeveling,
//! };
//! ```

pub mod dicom_lite;

// Re-export the most commonly used items at the module level
pub use dicom_lite::{
    DicomHeader, HounsfieldUnits, MedicalVolume, N4BiasCorrection, N4Config, Tissue,
    TissueClassStats, VolumeStats as MedicalVolumeStats, WindowLeveling,
};
