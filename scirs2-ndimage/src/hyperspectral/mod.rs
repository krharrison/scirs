//! Hyperspectral Imaging Analysis
//!
//! This module provides a comprehensive toolkit for hyperspectral image analysis
//! including spectral unmixing, preprocessing, and classification.
//!
//! # Module Structure
//!
//! | Sub-module | Description |
//! |------------|-------------|
//! | [`unmixing`]       | Endmember extraction (VCA, N-FINDR, SISAL) and abundance estimation (UCLS, NCLS, FCLS) |
//! | [`preprocessing`]  | Noise reduction (MNF), whitening, band removal, radiometric correction |
//! | [`classification`] | Pixel classification (SAM, SID, SCM, MSD) and accuracy metrics |
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use scirs2_ndimage::hyperspectral::{
//!     unmixing::{HyperspectralImage, vertex_component_analysis, abundance_estimation_fcls},
//!     preprocessing::{minimum_noise_fraction, radiometric_correction, RadiometricCalibration},
//!     classification::{sam_classifier, abundance_map_to_class},
//! };
//! use ndarray::Array2;
//!
//! // Synthetic 100-pixel, 20-band image.
//! let data = Array2::<f64>::ones((100, 20));
//! let img = HyperspectralImage::new(data);
//!
//! // Extract 3 endmembers via VCA.
//! let endmembers = vertex_component_analysis(&img, 3).unwrap();
//!
//! // Estimate FCLS abundances.
//! let delta = 10.0;
//! let abundances = abundance_estimation_fcls(&img, &endmembers, delta).unwrap();
//!
//! // Hard classification.
//! let class_map = abundance_map_to_class(&abundances, 0.3).unwrap();
//! ```

pub mod classification;
pub mod preprocessing;
pub mod unmixing;

// ── Re-exports ──────────────────────────────────────────────────────────────

// Core data structure.
pub use self::unmixing::HyperspectralImage;

// Endmember extraction.
pub use self::unmixing::{
    nfindr,
    sisal,
    vertex_component_analysis,
};

// Abundance estimation.
pub use self::unmixing::{
    abundance_estimation_fcls,
    abundance_estimation_ncls,
    abundance_estimation_ucls,
};

// Preprocessing.
pub use self::preprocessing::{
    MnfResult,
    RadiometricCalibration,
    cube_to_pixels,
    dark_object_subtraction,
    minimum_noise_fraction,
    pixels_to_cube,
    radiometric_correction,
    remove_absorption_bands,
    remove_bands,
    spatial_smoothing,
    whiten_hyperspectral,
};

// Classification.
pub use self::classification::{
    ClassificationMap,
    abundance_map_to_class,
    classification_accuracy,
    sam_classifier,
    sam_sid_classifier,
    sid_classifier,
    spectral_correlation_mapper,
    subspace_detector,
};
