//! Linear algebra abstractions using OxiBLAS backend.
//!
//! This module provides BLAS and LAPACK operations for SciRS2 using the pure Rust
//! OxiBLAS library as the backend. All other SciRS2 crates should use this module
//! instead of importing OxiBLAS directly (SciRS2 POLICY compliance).
//!
//! # Features
//!
//! - **BLAS Level 1**: Vector-vector operations (dot, nrm2, asum, axpy, scal)
//! - **BLAS Level 2**: Matrix-vector operations (gemv, matvec)
//! - **BLAS Level 3**: Matrix-matrix operations (gemm, matmul)
//! - **LAPACK Decompositions**: LU, QR, SVD, Cholesky, EVD, Schur
//! - **LAPACK Solvers**: Linear system solvers, least squares
//! - **Matrix Operations**: Inverse, determinant, condition number, rank
//!
//! # Example
//!
//! ```rust,ignore
//! use scirs2_core::linalg::prelude::*;
//! use scirs2_core::ndarray::Array2;
//!
//! // Matrix multiplication
//! let a = Array2::<f64>::eye(3);
//! let b = Array2::<f64>::ones((3, 3));
//! let c = matmul(&a, &b);
//!
//! // Linear solve
//! let a = array![[2.0, 1.0], [1.0, 3.0]];
//! let b = array![5.0, 7.0];
//! let x = solve_ndarray(&a, &b).expect("should succeed");
//!
//! // SVD decomposition
//! let svd = svd_ndarray(&a).expect("should succeed");
//! ```
//!
//! # SciRS2 POLICY Compliance
//!
//! This module is the ONLY place where OxiBLAS is imported directly.
//! All other SciRS2 crates must use `scirs2_core::linalg::*` imports.
//!
//! ```rust,ignore
//! // ✅ CORRECT: Use scirs2-core linalg abstraction
//! use scirs2_core::linalg::prelude::*;
//!
//! // ❌ WRONG: Don't import oxiblas directly in other crates
//! // use oxiblas_ndarray::prelude::*;  // NO!
//! ```

pub mod blas;
pub mod lapack;
pub mod prelude;

// Re-export prelude for convenience
pub use prelude::*;
