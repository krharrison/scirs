//! GPU-accelerated GEMM and mixed-precision CPU/GPU dispatch
//!
//! This module provides:
//! - Cache-blocked general matrix multiply (GEMM) with CPU fallback
//! - Batched GEMM for stacked matrix computations
//! - Symmetric GEMM (A * A^T)
//! - Adaptive precision dispatch based on condition number estimates
//! - Mixed-precision solve with iterative refinement
//!
//! When GPU hardware features (`cuda`, `opencl`, etc.) are not available,
//! all operations fall back to a high-performance cache-blocked CPU implementation.

pub mod dispatch;
pub mod gemm;

pub use dispatch::{
    adaptive_gemm, condition_number_estimate_1norm, gemm_f32_accum_f64, mixed_precision_solve,
    DispatchResult, PrecisionDispatchConfig, PrecisionMode,
};
pub use gemm::{batched_gemm, gemm, symm_gemm, GemmBackend, GemmConfig};
