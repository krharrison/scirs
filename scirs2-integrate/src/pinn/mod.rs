//! Physics-Informed Neural Networks (PINN) for PDE solving.
//!
//! This module provides a PINN-based approach to solving partial differential equations.
//! PINNs encode physics (the PDE residual) directly into the neural network loss function,
//! enabling mesh-free solutions that can incorporate sparse observational data.
//!
//! # Overview
//!
//! - **`types`**: Configuration, boundary conditions, and result structures
//! - **`network`**: Feed-forward neural network with finite-difference derivatives
//! - **`solver`**: Training loop with Adam optimizer and collocation point generation
//! - **`problems`**: Pre-built residual functions for common PDEs (Laplace, Poisson, Heat, Burgers)
//!
//! # Example
//!
//! ```rust,ignore
//! use scirs2_integrate::pinn::{PINNSolver, PINNConfig, problems};
//!
//! let problem = problems::laplace_problem_2d((0.0, 1.0, 0.0, 1.0));
//! let config = PINNConfig::default();
//! let mut solver = PINNSolver::new(&problem, config).unwrap();
//! let result = solver.train(&problems::laplace_residual, &problem, None).unwrap();
//! ```

pub(crate) mod high_level;
pub(crate) mod network;
pub(crate) mod problems;
pub(crate) mod solver;
pub(crate) mod types;

pub use high_level::{Pinn, PinnSolveResult};
pub use network::PINNNetwork;
pub use problems::{
    burgers_residual, heat_problem_1d, heat_residual, laplace_problem_2d, laplace_residual,
    poisson_problem_2d, poisson_residual,
};
pub use solver::{generate_boundary_points, generate_collocation_points, PINNSolver};
pub use types::{
    Boundary, BoundaryCondition, BoundarySide, CollocationStrategy, PDEProblem, PINNConfig,
    PINNResult,
};
