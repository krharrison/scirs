//! Kernel Methods
//!
//! This module provides kernel-based algorithms for nonlinear machine learning:
//!
//! - **Kernel Functions** (`kernels`): A library of kernel functions (Linear, Polynomial,
//!   RBF/Gaussian, Laplacian, Sigmoid), Gram matrix computation, and kernel centering.
//!
//! - **Kernel PCA** (`kpca`): Nonlinear dimensionality reduction via the kernel trick,
//!   with pre-image estimation and automatic parameter selection.
//!
//! - **Kernel Ridge Regression** (`kernel_ridge`): Tikhonov-regularized regression in
//!   kernel space, with closed-form LOO-CV and multi-output support.

/// Kernel functions library (Linear, Polynomial, RBF, Laplacian, Sigmoid)
pub mod kernels;

/// Kernel PCA for nonlinear dimensionality reduction
pub mod kpca;

/// Kernel Ridge Regression
pub mod kernel_ridge;

// Re-exports for convenience
pub use kernel_ridge::KernelRidgeRegression;
pub use kernels::{
    center_kernel_matrix, center_kernel_matrix_test, cross_gram_matrix, estimate_rbf_gamma,
    gram_matrix, is_positive_semidefinite, kernel_alignment, kernel_diagonal, kernel_eval,
    KernelType,
};
pub use kpca::KernelPCA;
