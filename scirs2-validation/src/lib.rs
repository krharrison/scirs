// Reference values module uses precise float literals and known constants as data points
#![allow(clippy::excessive_precision, clippy::approx_constant)]

//! Shared statistical validation framework for the COOLJAPAN ecosystem.
//!
//! This crate provides a standalone, dependency-light framework for validating
//! statistical distribution implementations against mathematically-derived
//! reference values. It is designed to be used by any stats library (scirs2-stats,
//! NumRS2, QuantRS2, etc.) without introducing circular dependencies.
//!
//! # Modules
//!
//! - [`reference_values`]: Pre-computed reference values for 15+ standard distributions
//! - [`validators`]: Generic validation traits and property-test helpers
//! - [`report`]: Human-readable and JSON report generation
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_validation::validators::{ValidatableDistribution, validate_distribution};
//! use scirs2_validation::reference_values::NORMAL_STANDARD;
//!
//! // Implement ValidatableDistribution for your distribution type, then:
//! // let result = validate_distribution(&my_normal, &NORMAL_STANDARD, 1e-9, 1e-9);
//! // assert!(result.passed);
//! ```

pub mod reference_values;
pub mod report;
pub mod validators;

// Re-export key types for convenience
pub use reference_values::DistributionReference;
pub use report::{generate_json_report, generate_report, ValidationReport};
pub use validators::{
    validate_cdf_monotone, validate_distribution, validate_pdf_integral, validate_ppf_roundtrip,
    ValidatableDistribution, ValidationResult,
};
