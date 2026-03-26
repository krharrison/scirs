//! GPU-accelerated matrix operations (pure-Rust simulated backend)
//!
//! This module implements the "GPU-accelerated matrix operations via OxiBLAS GPU
//! backend" roadmap item (TODO line 355).  It provides a complete, always-available
//! simulation of the GPU programming model without requiring any C/Fortran
//! dependencies or hardware feature flags.  When real GPU backends (CUDA, OpenCL,
//! Metal, ROCm) are enabled, the same public API routes through OxiBLAS.
//!
//! # Public surface
//!
//! | Item | Kind | Description |
//! |------|------|-------------|
//! | [`GpuBackendKind`] | `enum` | Backend selector: CPU / Simulated / OxiBLAS |
//! | [`GpuMatrixConfig`] | `struct` | Tuning config (tile size, threshold, …) |
//! | [`GpuMatrixBuffer`] | `struct` | CPU-backed matrix "device buffer" |
//! | [`GpuError`] | `enum` | Error variants for all GPU operations |
//! | [`GpuResult`] | `type` | `Result<T, GpuError>` alias |
//! | [`GpuCapabilities`] | `struct` | Simulated device capability report |
//! | [`detect_gpu_capabilities`] | `fn` | Query simulated device capabilities |
//! | [`gpu_sgemm`] | `fn` | Single-precision tiled GEMM |
//! | [`gpu_dgemm`] | `fn` | Double-precision tiled GEMM |
//! | [`gpu_batched_gemm`] | `fn` | Batched GEMM over a slice of buffer pairs |
//! | [`upload_matrix`] | `fn` | Slice → `GpuMatrixBuffer<f64>` |
//! | [`download_matrix`] | `fn` | `GpuMatrixBuffer<f64>` → `Vec<f64>` |
//! | [`adaptive_gemm`] | `fn` | GEMM with auto CPU/GPU dispatch |
//! | [`gpu_matmul`] | `fn` | Convenience `C = A · B` wrapper |
//! | [`gpu_transpose`] | `fn` | Cache-oblivious matrix transpose |
//! | [`gpu_axpy`] | `fn` | BLAS-1 `y += α·x` |
//! | [`GpuDispatcher`] | `struct` | Reusable dispatcher for repeated calls |
//!
//! # Feature flags
//!
//! No feature flags are required.  All operations compile and run on any platform
//! using the `Simulated` backend.  The `gpu` feature gates additional OxiBLAS GPU
//! runtime integration when hardware is present.
//!
//! # Example
//!
//! ```rust
//! use scirs2_linalg::gpu_accel::{
//!     GpuMatrixBuffer, GpuMatrixConfig, gpu_dgemm, gpu_matmul, gpu_transpose, gpu_axpy,
//!     detect_gpu_capabilities,
//! };
//!
//! // Query simulated device capabilities
//! let caps = detect_gpu_capabilities();
//! assert!(caps.vram_gb > 0.0);
//!
//! // Allocate and fill buffers
//! let a = GpuMatrixBuffer::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], 2, 2).unwrap();
//! let b = GpuMatrixBuffer::from_slice(&[5.0_f64, 6.0, 7.0, 8.0], 2, 2).unwrap();
//! let mut c = GpuMatrixBuffer::<f64>::zeros(2, 2);
//!
//! // GEMM: C = A * B
//! gpu_dgemm(&a, &b, 1.0, 0.0, &mut c).unwrap();
//! assert!((c.as_slice()[0] - 19.0).abs() < 1e-12);
//!
//! // Flat-slice convenience wrapper
//! let result = gpu_matmul(&[1.0, 2.0, 3.0, 4.0], &[1.0, 0.0, 0.0, 1.0], 2, 2, 2).unwrap();
//! assert!((result[0] - 1.0).abs() < 1e-12);
//!
//! // Transpose
//! let t = gpu_transpose(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
//! assert_eq!(t.len(), 6);
//!
//! // AXPY
//! let mut y = vec![1.0_f64, 2.0, 3.0];
//! gpu_axpy(2.0, &[10.0, 20.0, 30.0], &mut y);
//! assert!((y[0] - 21.0).abs() < 1e-12);
//! ```

pub mod dispatch;
pub mod gemm;
pub mod types;

// Re-export types
pub use types::{
    detect_gpu_capabilities, GpuBackendKind, GpuCapabilities, GpuError, GpuMatrixBuffer,
    GpuMatrixConfig, GpuResult,
};

// Re-export GEMM operations
pub use gemm::{download_matrix, gpu_batched_gemm, gpu_dgemm, gpu_sgemm, upload_matrix};

// Re-export dispatch utilities
pub use dispatch::{adaptive_gemm, gpu_axpy, gpu_matmul, gpu_transpose, GpuDispatcher};
