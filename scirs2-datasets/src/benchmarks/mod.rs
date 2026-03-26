//! Benchmarking utilities and synthetic benchmark dataset generators.
//!
//! This module contains two complementary components:
//!
//! 1. **[`performance`]** — `BenchmarkRunner`, `BenchmarkSuite`, `BenchmarkResult`,
//!    `PerformanceComparison`: tools for timing dataset operations.
//!
//! 2. **[`synthetic`]** — `make_friedman1`, `make_friedman2`, `make_blobs`,
//!    `make_moons`, `make_swiss_roll`, `make_checkerboard`: standard synthetic
//!    benchmark datasets for regression, classification, and manifold learning.

pub mod performance;
pub mod synthetic;

// Re-export performance benchmarking types (preserves backwards compatibility).
pub use performance::{BenchmarkResult, BenchmarkRunner, BenchmarkSuite, PerformanceComparison};

// Re-export synthetic benchmark generators.
pub use synthetic::{
    make_blobs as make_blobs_bench, make_checkerboard as make_checkerboard_bench,
    make_friedman1 as make_friedman1_bench, make_friedman2 as make_friedman2_bench,
    make_moons as make_moons_bench, make_swiss_roll as make_swiss_roll_bench,
};
