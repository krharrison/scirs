//! Quantum-Classical Hybrid Optimization
//!
//! This module provides classical simulation of quantum optimization algorithms:
//! - QAOA (Quantum Approximate Optimization Algorithm) for combinatorial problems
//! - VQE (Variational Quantum Eigensolver) for ground-state energy estimation
//! - Tensor network methods (MPS/DMRG) for quantum many-body systems
//!
//! # Overview
//!
//! The module implements exact statevector simulation of quantum circuits, enabling
//! benchmarking of quantum optimization protocols on small problem instances (up to
//! ~20 qubits on classical hardware).
//!
//! # Example: MaxCut with QAOA
//!
//! ```rust
//! use scirs2_optimize::quantum_classical::qaoa::{MaxCutProblem, QaoaConfig, QaoaCircuit};
//!
//! // Triangle graph: edges (0,1), (1,2), (0,2)
//! let problem = MaxCutProblem::new(3, vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]);
//! let config = QaoaConfig::default();
//! let circuit = QaoaCircuit::new(problem, config);
//! let result = circuit.optimize().expect("QAOA should converge");
//! println!("Expected cut value: {:.4}", result.optimal_value);
//! ```

pub mod qaoa;
pub mod statevector;
pub mod tensor_network;
pub mod vqe;

use crate::error::OptimizeError;

/// Result type for quantum-classical optimization operations
pub type QcResult<T> = Result<T, OptimizeError>;

/// Configuration for quantum-classical optimizers
#[derive(Debug, Clone)]
pub struct QcConfig {
    /// Maximum number of outer optimization iterations
    pub max_iter: usize,
    /// Convergence tolerance for parameter updates
    pub tol: f64,
    /// Whether to print iteration progress
    pub verbose: bool,
}

impl Default for QcConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-6,
            verbose: false,
        }
    }
}

/// Result of a quantum-classical optimization run
#[derive(Debug, Clone)]
pub struct QcOptResult {
    /// Optimal circuit parameters found
    pub optimal_params: Vec<f64>,
    /// Optimal objective value achieved
    pub optimal_value: f64,
    /// Total number of circuit evaluations performed
    pub n_evaluations: usize,
    /// Whether the optimizer converged within tolerance
    pub converged: bool,
}

/// Trait for quantum-classical hybrid optimizers
pub trait QuantumClassicalOptimizer {
    /// Run the optimization and return a result
    fn optimize(&mut self) -> QcResult<QcOptResult>;

    /// Evaluate the objective at a given parameter vector
    fn evaluate(&self, params: &[f64]) -> f64;

    /// Return the number of parameters in the circuit
    fn n_params(&self) -> usize;
}

// Re-export the most commonly used types
pub use qaoa::{MaxCutProblem, QaoaCircuit, QaoaConfig, QaoaResult};
pub use statevector::Statevector;
pub use tensor_network::{ising_1d_mpo, MPS};
pub use vqe::{HardwareEfficientAnsatz, PauliHamiltonian, PauliOp, VqeOptimizer, VqeResult};
