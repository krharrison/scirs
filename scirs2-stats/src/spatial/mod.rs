//! Spatial Statistics
//!
//! This module provides a comprehensive suite of spatial analysis tools:
//!
//! - **`autocorrelation`**: Moran's I (global & LISA), Geary's C, Ripley's K/L functions
//! - **`point_process`**: CSR envelope tests, kernel intensity estimation, G/F functions
//! - **`kriging`**: Variogram modelling, ordinary kriging interpolation
//!
//! # References
//! - Moran, P.A.P. (1950). Notes on Continuous Stochastic Phenomena.
//! - Geary, R.C. (1954). The Contiguity Ratio and Statistical Mapping.
//! - Ripley, B.D. (1976). The Second-Order Analysis of Stationary Point Processes.
//! - Matheron, G. (1963). Principles of Geostatistics.
//! - Diggle, P.J. (2003). Statistical Analysis of Spatial Point Patterns.

pub mod autocorrelation;
pub mod kriging;
pub mod point_process;

// Re-export primary types and functions for convenience
pub use autocorrelation::{
    gearys_c, local_morans_i, morans_i, ripleys_k, ripleys_l, ClusterType, GearyResult,
    LocalMoranResult, MoranResult,
};
pub use kriging::{
    empirical_variogram, EmpiricalVariogram, OrdinaryKriging, VariogramModel,
};
pub use point_process::{
    csr_envelope, f_function, g_function, kernel_intensity, CsrEnvelope,
};

/// Error type for spatial statistics operations.
#[derive(Debug, thiserror::Error)]
pub enum SpatialError {
    /// Input arrays have mismatched dimensions.
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Input data is invalid (e.g., empty, non-finite values).
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Numerical failure during computation (e.g., singular matrix).
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Convergence failure during iterative fitting.
    #[error("Convergence error: {0}")]
    ConvergenceError(String),

    /// Insufficient data to perform the operation.
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
}

/// Convenience result alias for spatial operations.
pub type SpatialResult<T> = Result<T, SpatialError>;
