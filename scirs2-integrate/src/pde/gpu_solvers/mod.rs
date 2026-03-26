//! GPU-accelerated PDE solvers for FEM and FDM.
//!
//! This module provides simulated-GPU implementations of Finite Difference Method (FDM)
//! and Finite Element Method (FEM) PDE solvers, using tiled parallelism via
//! `std::thread::scope` to approximate GPU thread-block execution patterns.
//!
//! ## Submodules
//! - [`types`]: shared types (config, grid, boundary conditions, errors, stats)
//! - [`fdm_gpu`]: GPU-accelerated FDM for Poisson, heat, and wave equations
//! - [`fem_gpu`]: GPU-accelerated FEM for the Poisson equation on triangular meshes
//!
//! ## Quick example (FDM Poisson)
//! ```rust
//! use scirs2_integrate::pde::gpu_solvers::{
//!     types::{BoundaryCondition, GpuGrid2D, GpuPdeConfig, GridSpec},
//!     fdm_gpu::solve_poisson_2d,
//! };
//!
//! let spec = GridSpec::new(11, 11, 0.1, 0.1, 0.0, 0.0).unwrap();
//! let rhs  = GpuGrid2D::zeros(spec).unwrap();
//! let bc   = [[BoundaryCondition::Dirichlet(0.0); 2]; 2];
//! let cfg  = GpuPdeConfig::default();
//! let (sol, stats) = solve_poisson_2d(&rhs, &bc, &cfg).unwrap();
//! assert!(stats.converged);
//! ```

pub mod fdm_gpu;
pub mod fem_gpu;
pub mod types;

// Re-export the most commonly used items at the submodule level.
pub use fdm_gpu::{solve_heat_2d, solve_poisson_2d, solve_wave_2d, GpuFdmSolver};
pub use fem_gpu::{
    apply_dirichlet_gpu, assemble_stiffness_gpu, conjugate_gradient_gpu, solve_fem_poisson,
    uniform_rect_mesh, FemMesh,
};
pub use types::{
    BoundaryCondition, GpuGrid2D, GpuPdeConfig, GridSpec, PdeSolverError, PdeSolverResult,
    SolverStats,
};
