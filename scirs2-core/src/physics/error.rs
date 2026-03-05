//! Error types for the physics module.

use thiserror::Error;

/// Errors that can arise from physics computations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum PhysicsError {
    /// A physical parameter had an invalid value (e.g. negative mass).
    #[error("Invalid parameter '{param}': {reason}")]
    InvalidParameter {
        /// Name of the parameter.
        param: &'static str,
        /// Human-readable explanation.
        reason: String,
    },

    /// The requested quantum number is out of range.
    #[error("Quantum number out of range: {0}")]
    QuantumNumberOutOfRange(String),

    /// The velocity is at or exceeds the speed of light (relativistic singularity).
    #[error("Velocity {velocity:.6e} m/s is >= speed of light {c:.6e} m/s")]
    SuperluminalVelocity {
        /// Supplied velocity (m/s).
        velocity: f64,
        /// Speed of light (m/s).
        c: f64,
    },

    /// A general domain error (argument outside the domain of the function).
    #[error("Domain error: {0}")]
    DomainError(String),
}

/// Convenience result alias for physics functions.
pub type PhysicsResult<T> = Result<T, PhysicsError>;
