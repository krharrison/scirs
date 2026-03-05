//! Error types for the quantum simulation module.

use thiserror::Error;

/// Errors that can arise from quantum simulation operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum QuantumError {
    /// The statevector has zero norm and cannot be normalised.
    #[error("Quantum state vector has zero norm — unphysical state")]
    ZeroStateVector,

    /// A qubit index was out of range for the register size.
    #[error("Qubit index {index} is out of range for a {n_qubits}-qubit register")]
    QubitIndexOutOfRange {
        /// The supplied index.
        index: usize,
        /// Number of qubits in the register.
        n_qubits: usize,
    },

    /// The statevector dimension did not match 2^n_qubits.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension encountered.
        actual: usize,
    },

    /// A basis-state index exceeded the Hilbert-space dimension.
    #[error("Basis index {index} is out of range for dim={dim}")]
    BasisIndexOutOfRange {
        /// Supplied index.
        index: usize,
        /// Hilbert-space dimension 2^n.
        dim: usize,
    },

    /// Too many qubits were requested (would overflow usize).
    #[error("Too many qubits requested: {0} would exceed usize capacity")]
    TooManyQubits(usize),

    /// An invalid qubit count was supplied (e.g. 0).
    #[error("Invalid qubit count: {0}")]
    InvalidQubitCount(usize),

    /// Gate arity does not match the number of target qubits supplied.
    #[error("Gate arity mismatch: gate acts on {gate_qubits} qubit(s) but {supplied} target(s) were provided")]
    GateArityMismatch {
        /// Number of qubits the gate acts on.
        gate_qubits: usize,
        /// Number of target indices supplied by the caller.
        supplied: usize,
    },

    /// Two target-qubit indices in a gate application were identical.
    #[error("Duplicate qubit index {index} in gate target list")]
    DuplicateQubitIndex {
        /// The repeated index.
        index: usize,
    },

    /// The gate matrix is not unitary (failed validation).
    #[error("Gate matrix is not unitary (max deviation = {deviation:.3e})")]
    NonUnitaryGate {
        /// Maximum entry-wise deviation from U†U = I.
        deviation: f64,
    },

    /// A circuit operation was applied to a register with the wrong qubit count.
    #[error(
        "Circuit was built for {circuit_qubits} qubit(s) but the register has {register_qubits}"
    )]
    CircuitRegisterMismatch {
        /// Circuit qubit count.
        circuit_qubits: usize,
        /// Register qubit count.
        register_qubits: usize,
    },

    /// A general domain / parameter error.
    #[error("Domain error: {0}")]
    DomainError(String),
}

/// Convenience result alias for quantum functions.
pub type QuantumResult<T> = Result<T, QuantumError>;
