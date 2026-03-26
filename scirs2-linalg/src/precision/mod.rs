//! Extended and compensated precision linear algebra operations.
//!
//! This module provides algorithms that use error-free transformations (EFTs)
//! to achieve accuracy beyond standard f64 arithmetic in key linear algebra
//! kernels such as GEMM.
//!
//! # Modules
//!
//! - [`compensated_gemm`]: GEMM with Kahan / double-double / TwoFold accumulation.

pub mod compensated_gemm;

pub use compensated_gemm::{AccumulationMode, CompensatedGemm, CompensatedGemmConfig};
