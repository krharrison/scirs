//! Benchmark dataset generators for optimization and machine learning.
//!
//! # Contents
//!
//! - [`test_functions`] — Optimization test functions: ZDT multi-objective,
//!   DTLZ, Ackley, Rastrigin, Griewank, Levy, and many more.
//! - [`ml_benchmarks`] — ML dataset generators: Friedman, moons, circles,
//!   Swiss roll, S-curve, imbalanced classification, concept drift.

pub mod ml_benchmarks;
pub mod test_functions;
