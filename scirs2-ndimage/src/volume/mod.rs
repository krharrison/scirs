//! # 3D Volumetric Image Processing (`scirs2-ndimage::volume`)
//!
//! This module provides a comprehensive, production-quality API for processing
//! and analysing 3-D volumetric images.  It is organised into four
//! sub-modules:
//!
//! | Sub-module | Contents |
//! |------------|---------|
//! | [`filters3d`] | Gaussian, median, Sobel, LoG, bilateral, anisotropic diffusion |
//! | [`morphology3d`] | Erosion, dilation, opening, closing, binary variants, CC labeling, skeleton |
//! | [`measurements3d`] | Label, region properties, moment-of-inertia tensor |
//! | [`surface`] | Marching cubes, surface-normal estimation, isosurface extraction |
//!
//! ## Quick Example
//!
//! ```rust,no_run
//! use scirs2_ndimage::volume::{
//!     gaussian_filter_3d, median_filter_3d, sobel_3d,
//!     binary_erosion_3d, connected_components_3d, StructElem3D,
//!     label_3d, region_props_3d,
//!     marching_cubes, isosurface_extraction,
//! };
//! use scirs2_core::ndarray::Array3;
//!
//! // Build a small test volume
//! let vol = Array3::<f64>::from_elem((16, 16, 16), 1.0);
//!
//! // 3-D Gaussian smoothing
//! let smoothed = gaussian_filter_3d(vol.view(), 1.5).unwrap();
//!
//! // 3-D median filter
//! let med = median_filter_3d(vol.view(), 3).unwrap();
//!
//! // Binary mask operations
//! let mask = Array3::<bool>::from_elem((8, 8, 8), true);
//! let se = StructElem3D::Cross26;
//! let eroded = binary_erosion_3d(mask.view(), &se).unwrap();
//!
//! // Connected-component labeling
//! let (labels, n) = connected_components_3d(mask.view()).unwrap();
//! ```

pub mod filters3d;
pub mod measurements3d;
pub mod morphology3d;
pub mod surface;

// ── Filters ──────────────────────────────────────────────────────────────────

pub use filters3d::{
    anisotropic_diffusion_3d,
    bilateral_filter_3d,
    gaussian_filter_3d,
    gradient_magnitude_3d,
    laplacian_3d,
    median_filter_3d,
    sobel_3d,
};

// ── Morphology ───────────────────────────────────────────────────────────────

pub use morphology3d::{
    StructElem3D,
    // Grayscale
    closing_3d,
    dilation_3d,
    erosion_3d,
    opening_3d,
    // Binary
    binary_dilation_3d,
    binary_erosion_3d,
    // Analysis
    connected_components_3d,
    skeleton_3d,
};

// ── Measurements ─────────────────────────────────────────────────────────────

pub use measurements3d::{
    BBox3D,
    RegionProps3D,
    label_3d,
    moment_of_inertia_3d,
    region_props_3d,
};

// ── Surface extraction ───────────────────────────────────────────────────────

pub use surface::{
    SurfaceMesh,
    estimate_surface_normals,
    isosurface_extraction,
    marching_cubes,
};
