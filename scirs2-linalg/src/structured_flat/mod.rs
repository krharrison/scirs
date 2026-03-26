//! Flat (`Vec<f64>`) structured matrix solvers for Toeplitz, Circulant, and Hankel matrices.
//!
//! This module provides plain-data (no generics, no ndarray) implementations of:
//!
//! - [`FlatToeplitz`]: `T[i,j] = t[i-j]`, with O(N log N) matvec (circulant embedding)
//!   and O(N²) Levinson-Durbin solve.
//! - [`FlatCirculant`]: `C[i,j] = c[(j-i+n)%n]`, with O(N log N) matvec/solve via FFT.
//! - [`FlatHankel`]: `H[i,j] = h[i+j]`, with O(N²) direct matvec.
//!
//! The FFT used internally is a self-contained radix-2 Cooley-Tukey implementation
//! that degrades gracefully to O(N²) DFT for non-power-of-two sizes.

mod circulant;
mod hankel;
mod toeplitz;
pub mod types;

pub use toeplitz::levinson_durbin;
pub use types::{FlatCirculant, FlatHankel, FlatToeplitz};
