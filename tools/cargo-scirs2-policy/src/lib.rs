//! `cargo-scirs2-policy` library crate.
//!
//! Exposes the linter modules for use in integration tests and as a library.
//!
//! The binary entry point is `src/main.rs`.

pub mod checks;
pub mod report;
pub mod version_policy;
pub mod violation;
pub mod workspace;

// Internal modules (not part of the public API but re-exported here so that
// integration tests can access them via `cargo_scirs2_policy::rules`, etc.)
pub mod bench_regression;
pub mod dep_audit;
pub mod output;
pub mod rules;
