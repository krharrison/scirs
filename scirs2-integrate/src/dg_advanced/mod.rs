//! Advanced Discontinuous Galerkin methods
//!
//! This module provides improvements over the basic DG implementation in `crate::dg`:
//!
//! - **Curved elements**: Isoparametric mappings for curved triangular/quad elements,
//!   Gordon-Hall blended transfinite interpolation, Jacobian computations.
//! - **High-order DG**: Legendre polynomial basis, modal mass matrix, p-refinement
//!   with the Persson-Peraire troubled-cell indicator.
//! - **Entropy-stable schemes**: SBP operators satisfying Q + Q^T = B,
//!   Tadmor entropy-conserving flux, entropy-stable flux differencing (Gassner 2013),
//!   and entropy rate monitoring.
//!
//! ## References
//!
//! - Hesthaven & Warburton, "Nodal Discontinuous Galerkin Methods" (2008)
//! - Tadmor, "Entropy functions for symmetric systems of conservation laws" (1987)
//! - Gassner, "A skew-symmetric discontinuous Galerkin spectral element discretization
//!   and its relation to SBP-SAT finite difference methods" (2013)
//! - Carpenter & Fisher, "High-order entropy stable finite difference schemes for
//!   nonlinear conservation laws" (2013)
//! - Persson & Peraire, "Sub-cell shock capturing for DG methods" (2006)

pub mod curved_elements;
pub mod entropy_stable;
pub mod high_order_dg;
pub mod types;

// Re-export primary types
pub use types::{
    CurvedElement, DgAdvancedConfig, DgSolution, EntropyStableConfig, FluxType, GeometricMap,
};

// Re-export curved element utilities
pub use curved_elements::{
    arc_boundary_map, blended_transfinite_interpolation, curved_quad_points_weights, det_jacobian,
    gauss_legendre_1d, inv_jacobian, isoparametric_map, jacobian, lagrange_basis_triangle,
    lagrange_nodes_triangle,
};

// Re-export entropy-stable scheme components
pub use entropy_stable::{
    burgers_flux, differentiation_matrix_lgl, entropy_stable_flux_burgers, legendre_gauss_lobatto,
    rusanov_flux, EntropyStableDg1D, SbpOperator,
};

// Re-export high-order DG utilities
pub use high_order_dg::{
    high_order_dg_convergence_test, l2_error, legendre, legendre_deriv,
    legendre_mass_matrix_diagonal, modal_to_nodal_eval, nodal_to_modal, p_refine_step,
    troubled_cell_indicator,
};
