//! Advanced eigensolvers for SciRS2 linear algebra.
//!
//! This module provides three high-performance eigensolver implementations:
//!
//! - **Divide-and-Conquer** (`divide_conquer`): Cuppen's D&C algorithm for symmetric
//!   tridiagonal eigenvalue problems. O(n²) for eigenvalues, O(n³) for eigenvectors.
//!
//! - **FEAST** (`feast`): Spectral slicing via contour integration (Polizzi 2009).
//!   Finds all eigenvalues/vectors in a user-specified interval [a, b].
//!
//! - **Randomized** (`randomized_eig`): Nyström-based randomized eigensolver
//!   (Williams & Seeger 2001). Efficiently computes approximate top-k eigenvalues.
//!
//! # References
//!
//! - Cuppen, J.J.M. (1981). A divide and conquer method for the symmetric tridiagonal
//!   eigenproblem. *Numerische Mathematik*, 36(2), 177–195.
//! - Polizzi, E. (2009). Density-matrix-based algorithm for solving eigenvalue problems.
//!   *Physical Review B*, 79(11), 115112.
//! - Williams, C.K.I. & Seeger, M. (2001). Using the Nyström method to speed up kernel
//!   machines. *Advances in Neural Information Processing Systems*, 13.

pub mod divide_conquer;
pub mod feast;
pub mod randomized_eig;

pub use divide_conquer::{dc_eig_symmetric, dc_eig_tridiag, DcConfig};
pub use feast::{feast_eig, FeastConfig, FeastResult};
pub use randomized_eig::{randomized_eig_symmetric, RandomizedEigConfig};
