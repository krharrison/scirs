//! # SciPy / NumPy Migration Guide
//!
//! This module provides a programmatic equivalence table mapping SciPy and NumPy
//! function names to their SciRS2 equivalents, including usage examples and notes
//! about API differences.
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_core::scipy_migration::scipy_equiv::{search_scipy, by_category, MigrationCategory};
//!
//! // Find the SciRS2 equivalent of scipy.linalg.det
//! let results = search_scipy("linalg.det");
//! assert!(!results.is_empty());
//!
//! // Browse all linear algebra equivalents
//! let linalg_entries = by_category(MigrationCategory::LinearAlgebra);
//! assert!(!linalg_entries.is_empty());
//! ```
//!
//! ## Supported Categories
//!
//! - Linear Algebra (`scipy.linalg`, `numpy.linalg`)
//! - Statistics (`scipy.stats`)
//! - Signal Processing (`scipy.signal`)
//! - FFT (`scipy.fft`, `numpy.fft`)
//! - Optimization (`scipy.optimize`)
//! - Integration (`scipy.integrate`)
//! - Interpolation (`scipy.interpolate`)
//! - Special Functions (`scipy.special`)
//! - Sparse Matrices (`scipy.sparse`)
//! - Image Processing (`scipy.ndimage`)
//!
//! ## NumPy Core
//!
//! NumPy array operations are mapped to `ndarray` (re-exported via `scirs2_core::ndarray`)
//! and to SciRS2 utility functions. See [`numpy_equiv`] for the complete reference.

pub mod numpy_equiv;
pub mod scipy_equiv;

pub use numpy_equiv::{numpy_table, search_numpy, NumpyEntry};
pub use scipy_equiv::{
    by_category, migration_table, search_scipy, MigrationCategory, MigrationEntry,
};
