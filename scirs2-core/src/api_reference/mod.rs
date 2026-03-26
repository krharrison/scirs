//! # SciRS2 API Reference
//!
//! Programmatic catalog of all major public APIs in the SciRS2 ecosystem,
//! with mathematical references, usage examples, and cross-references.
//!
//! ## Usage
//!
//! ```rust
//! use scirs2_core::api_reference::{api_catalog, search_api, by_crate, by_category, ApiCategory};
//!
//! // Browse full catalog
//! let catalog = api_catalog();
//! assert!(!catalog.is_empty());
//!
//! // Search by name
//! let results = search_api("svd");
//! assert!(!results.is_empty());
//!
//! // Filter by crate
//! let linalg_apis = by_crate("scirs2-linalg");
//! assert!(!linalg_apis.is_empty());
//!
//! // Filter by category
//! let stats_apis = by_category(ApiCategory::Statistics);
//! assert!(!stats_apis.is_empty());
//! ```

pub mod catalog;
pub mod math_reference;

pub use catalog::{api_catalog, by_category, by_crate, search_api, ApiCategory, ApiEntry};
pub use math_reference::{math_references, MathReference};

#[cfg(test)]
mod tests;
