//! Short-Time Fractional Fourier Transform (STFRFT) and related algorithms.
//!
//! The fractional Fourier transform (FrFT) generalises the ordinary Fourier
//! transform by a continuous rotation angle α in the time-frequency plane.
//! This module provides:
//!
//! - [`stfrft()`] – Short-Time FrFT (sliding window FrFT analysis).
//! - [`istfrft`] – Inverse STFRFT via overlap-add.
//! - [`dfrft`] – Discrete FrFT via Grünbaum tridiagonal eigenvector method.

pub mod stfrft;

pub use stfrft::*;
