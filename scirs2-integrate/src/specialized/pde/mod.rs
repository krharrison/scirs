//! Advanced PDE solvers (specialized)
//!
//! This module extends the base PDE infrastructure with:
//!
//! - **Multigrid** ([`multigrid`]): geometric multigrid (V/W/F cycles) for
//!   elliptic problems on structured grids.
//! - **Spectral Element 1D** ([`spectral_element`]): high-order GLL-based
//!   spectral element discretization for 1D BVPs.
//! - **Operator Splitting** ([`splitting`]): Lie-Trotter, Strang, Yoshida, and
//!   adaptive Strang splitting for `du/dt = A(u) + B(u)` systems.

pub mod multigrid;
pub mod spectral_element;
pub mod splitting;

// Re-exports for convenience
pub use multigrid::{
    solve_elliptic_2d, solve_poisson_2d, BoundaryCondition2D, CycleType, EllipticCoeffs2D,
    MultigridConfig, MultigridStats,
};
pub use spectral_element::{gll_nodes_weights, SpectralElement1D};
pub use splitting::{
    lie_trotter_split, strang_split, strang_split_adaptive, yoshida_split,
};
