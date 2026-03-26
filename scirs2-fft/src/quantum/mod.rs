//! Quantum-inspired Fourier Transform algorithms.
//!
//! This module provides circuit-level simulation of Quantum Fourier Transform (QFT)
//! and related quantum algorithms implemented in pure Rust on classical hardware.
//!
//! # Key algorithms
//! - **QFT**: n-qubit Quantum Fourier Transform with statevector simulation
//! - **IQFT**: Inverse Quantum Fourier Transform
//! - **QPE**: Quantum Phase Estimation using QFT as the readout register

pub mod phase_estimation;
pub mod qft;

pub use phase_estimation::*;
pub use qft::*;
