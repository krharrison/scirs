//! Types for GPU-accelerated PDE solvers
//!
//! Defines configuration, grid representations, boundary conditions,
//! error types, and solver statistics used across FDM and FEM GPU solvers.

use std::fmt;

/// Configuration for GPU-accelerated PDE solvers.
///
/// Simulates GPU tiling and parallelism via CPU multi-threading.
#[derive(Debug, Clone)]
pub struct GpuPdeConfig {
    /// Number of threads per tile (analogous to CUDA block size)
    pub tile_size: usize,
    /// Maximum number of solver iterations
    pub max_iterations: usize,
    /// Convergence tolerance (L∞ norm of residual)
    pub tolerance: f64,
    /// Whether to use parallel thread-based updates
    pub use_parallel: bool,
}

impl Default for GpuPdeConfig {
    fn default() -> Self {
        GpuPdeConfig {
            tile_size: 16,
            max_iterations: 10_000,
            tolerance: 1e-8,
            use_parallel: true,
        }
    }
}

/// Specification for a 2D uniform Cartesian grid.
#[derive(Debug, Clone, Copy)]
pub struct GridSpec {
    /// Number of grid points in x direction
    pub nx: usize,
    /// Number of grid points in y direction
    pub ny: usize,
    /// Grid spacing in x
    pub dx: f64,
    /// Grid spacing in y
    pub dy: f64,
    /// X coordinate of the first grid point
    pub x0: f64,
    /// Y coordinate of the first grid point
    pub y0: f64,
}

impl GridSpec {
    /// Create a new GridSpec.
    ///
    /// # Errors
    /// Returns `PdeSolverError::InvalidGrid` if nx < 3, ny < 3, dx <= 0 or dy <= 0.
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64, x0: f64, y0: f64) -> PdeSolverResult<Self> {
        if nx < 3 || ny < 3 {
            return Err(PdeSolverError::InvalidGrid);
        }
        if dx <= 0.0 || dy <= 0.0 {
            return Err(PdeSolverError::InvalidGrid);
        }
        Ok(GridSpec { nx, ny, dx, dy, x0, y0 })
    }

    /// X coordinate at grid index i.
    #[inline]
    pub fn x_coord(&self, i: usize) -> f64 {
        self.x0 + i as f64 * self.dx
    }

    /// Y coordinate at grid index j.
    #[inline]
    pub fn y_coord(&self, j: usize) -> f64 {
        self.y0 + j as f64 * self.dy
    }

    /// Flat row-major index for (i, j).
    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// Total number of grid points.
    #[inline]
    pub fn total(&self) -> usize {
        self.nx * self.ny
    }
}

/// Boundary condition variants for PDE problems.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition {
    /// Fixed value: u = value at the boundary.
    Dirichlet(f64),
    /// Fixed normal derivative: du/dn = value at the boundary.
    Neumann(f64),
    /// Periodic boundary (value wraps around).
    Periodic,
    /// Robin (mixed): a·u + b·du/dn = 0 at the boundary.
    Robin {
        /// Coefficient for u term.
        a: f64,
        /// Coefficient for du/dn term.
        b: f64,
    },
}

/// Errors from GPU PDE solvers.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum PdeSolverError {
    /// Iterative solver did not converge within the allowed iteration count.
    NotConverged {
        /// Number of iterations attempted.
        iterations: usize,
    },
    /// Invalid grid specification (e.g., too few points, non-positive spacing).
    InvalidGrid,
    /// Boundary conditions do not match the grid or each other.
    BoundaryMismatch,
    /// Linear system is singular or near-singular.
    SingularSystem,
}

impl fmt::Display for PdeSolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PdeSolverError::NotConverged { iterations } => {
                write!(f, "Solver did not converge after {iterations} iterations")
            }
            PdeSolverError::InvalidGrid => write!(f, "Invalid grid specification"),
            PdeSolverError::BoundaryMismatch => write!(f, "Boundary condition mismatch"),
            PdeSolverError::SingularSystem => write!(f, "Singular or near-singular system"),
        }
    }
}

impl std::error::Error for PdeSolverError {}

/// Convenience result type for GPU PDE solvers.
pub type PdeSolverResult<T> = Result<T, PdeSolverError>;

/// Statistics reported after solver completion.
#[derive(Debug, Clone)]
pub struct SolverStats {
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final L∞ residual achieved.
    pub final_residual: f64,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
}

impl SolverStats {
    /// Construct a converged stats record.
    pub fn converged(iterations: usize, final_residual: f64) -> Self {
        SolverStats { iterations, final_residual, converged: true }
    }

    /// Construct a non-converged stats record.
    pub fn not_converged(iterations: usize, final_residual: f64) -> Self {
        SolverStats { iterations, final_residual, converged: false }
    }
}

/// A 2D field stored on a uniform Cartesian grid (simulating GPU device memory).
///
/// Data is stored in row-major order: `data[j * nx + i]` is the value at `(i, j)`.
#[derive(Debug, Clone)]
pub struct GpuGrid2D {
    /// Flat row-major data array.
    pub data: Vec<f64>,
    /// Grid specification.
    pub spec: GridSpec,
}

impl GpuGrid2D {
    /// Create a zero-initialised grid.
    ///
    /// # Errors
    /// Returns `PdeSolverError::InvalidGrid` if the spec is invalid.
    pub fn zeros(spec: GridSpec) -> PdeSolverResult<Self> {
        let n = spec.total();
        if n == 0 {
            return Err(PdeSolverError::InvalidGrid);
        }
        Ok(GpuGrid2D { data: vec![0.0_f64; n], spec })
    }

    /// Create a grid from a flat data vector.
    ///
    /// # Errors
    /// Returns `PdeSolverError::InvalidGrid` if the data length doesn't match the spec.
    pub fn from_data(data: Vec<f64>, spec: GridSpec) -> PdeSolverResult<Self> {
        if data.len() != spec.total() {
            return Err(PdeSolverError::InvalidGrid);
        }
        Ok(GpuGrid2D { data, spec })
    }

    /// Get value at grid position `(i, j)`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[self.spec.idx(i, j)]
    }

    /// Set value at grid position `(i, j)`.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        let idx = self.spec.idx(i, j);
        self.data[idx] = val;
    }

    /// Return the L∞ norm difference between `self` and `other`.
    ///
    /// # Errors
    /// Returns `PdeSolverError::InvalidGrid` if the grids have different sizes.
    pub fn linf_diff(&self, other: &GpuGrid2D) -> PdeSolverResult<f64> {
        if self.data.len() != other.data.len() {
            return Err(PdeSolverError::InvalidGrid);
        }
        let max_diff = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        Ok(max_diff)
    }

    /// L∞ norm of this grid.
    pub fn linf_norm(&self) -> f64 {
        self.data.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
    }

    /// Sum of all values (useful for conservation checks).
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Number of x grid points.
    pub fn nx(&self) -> usize {
        self.spec.nx
    }

    /// Number of y grid points.
    pub fn ny(&self) -> usize {
        self.spec.ny
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_pde_config_default() {
        let cfg = GpuPdeConfig::default();
        assert_eq!(cfg.tile_size, 16);
        assert_eq!(cfg.max_iterations, 10_000);
        assert!((cfg.tolerance - 1e-8).abs() < 1e-20);
        assert!(cfg.use_parallel);
    }

    #[test]
    fn test_grid_spec_coordinates() {
        let spec = GridSpec::new(5, 5, 0.25, 0.25, 0.0, 0.0).expect("valid spec");
        assert!((spec.x_coord(2) - 0.5).abs() < 1e-15);
        assert!((spec.y_coord(4) - 1.0).abs() < 1e-15);
        assert_eq!(spec.idx(2, 3), 3 * 5 + 2);
        assert_eq!(spec.total(), 25);
    }

    #[test]
    fn test_grid_spec_invalid() {
        assert!(GridSpec::new(2, 5, 0.25, 0.25, 0.0, 0.0).is_err());
        assert!(GridSpec::new(5, 5, -0.1, 0.25, 0.0, 0.0).is_err());
    }

    #[test]
    fn test_gpu_grid_2d_create() {
        let spec = GridSpec::new(4, 4, 0.1, 0.1, 0.0, 0.0).expect("valid");
        let grid = GpuGrid2D::zeros(spec).expect("created");
        assert_eq!(grid.data.len(), 16);
        assert_eq!(grid.nx(), 4);
        assert_eq!(grid.ny(), 4);
        assert!((grid.linf_norm() - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_solver_stats_not_converged() {
        let stats = SolverStats::not_converged(500, 1e-3);
        assert!(!stats.converged);
        assert_eq!(stats.iterations, 500);
        assert!((stats.final_residual - 1e-3).abs() < 1e-15);
    }

    #[test]
    fn test_boundary_condition_periodic() {
        let bc = BoundaryCondition::Periodic;
        assert_eq!(bc, BoundaryCondition::Periodic);
        let bc2 = BoundaryCondition::Dirichlet(1.5);
        assert_ne!(bc, bc2);
    }
}
