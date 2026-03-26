//! Physics-informed interpolation methods.
//!
//! This module extends the existing physics-informed interpolation with:
//! - PDE-constrained RBF interpolation (`pde_constrained`)
//! - Online/streaming RBF updates (`streaming`)
//! - ANOVA decomposition for adaptive sparse grids (`anova`)
//! - Automatic interpolation method selection (`auto_select`)

pub mod anova;
pub mod auto_select;
pub mod legacy;
pub mod pde_constrained;
pub mod streaming;

// Re-export legacy content at the module level for backward compatibility
pub use legacy::{
    make_mass_conserving_interpolator, make_monotonic_physics_interpolator,
    make_smooth_physics_interpolator, ConservationLaw, PhysicalConstraint, PhysicsInformedConfig,
    PhysicsInformedInterpolator, PhysicsInformedResult,
};

// Re-export new additions
pub use anova::{AnovaConfig, AnovaDecomposition};
pub use auto_select::{
    analyze_data, recommend_method, recommend_with_rationale, DataProfile, InterpolationMethod,
};
pub use pde_constrained::{PdeConfig, PdeConstrainedRbf, PdeType};
pub use streaming::{StreamingRbf, StreamingRbfConfig};
