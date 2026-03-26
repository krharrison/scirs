//! High-performance multidimensional FFT with cache-oblivious tiling.
//!
//! Provides a from-scratch mixed-radix N-dimensional FFT engine:
//!
//! - [`mixed_radix`] — 1-D mixed-radix engine (radix-2/4/generic + Bluestein)
//! - [`ndim`]        — N-D row-column decomposition with cache-oblivious tiling
//! - [`parallel_ndim`] — Thread-parallel N-D FFT via `std::thread::scope`
//! - [`types`]       — Config structs and result types

pub mod mixed_radix;
pub mod ndim;
pub mod parallel_ndim;
pub mod types;

// Convenient top-level re-exports
pub use ndim::{fftn, fftn_norm, ifftn, ifftn_norm, irfftn, rfftn, tiled_2d_fft};
pub use parallel_ndim::{parallel_fftn, parallel_fftn_norm, parallel_ifftn};
pub use types::{FftAxis, FftPlan, NdimFftConfig, NdimFftResult, NormMode};
