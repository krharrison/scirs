//! Kolmogorov-Arnold Networks (KAN).
//!
//! Replaces fixed activation functions with learnable univariate functions on
//! each network edge.  Each edge carries a B-spline or rational activation that
//! is learned during training.
//!
//! # Architecture
//!
//! A KAN layer with `n_in` inputs and `n_out` outputs stores `n_in × n_out`
//! independent activation functions `phi_{i,j}`.  The output is:
//!
//! ```text
//! y_j = Σ_i  phi_{i,j}(x_i)
//! ```
//!
//! Stacking multiple KAN layers forms a `KanNetwork`.
//!
//! # Supported activation families
//!
//! | Variant             | Description                                      |
//! |---------------------|--------------------------------------------------|
//! | [`BSplineActivation`] | Piecewise-polynomial (cubic by default); smooth |
//! | [`RationalActivation`] | Padé-type; globally smooth; compact support    |
//!
//! # Reference
//!
//! Liu et al. (2024) *"KAN: Kolmogorov-Arnold Networks"*
//! <https://arxiv.org/abs/2404.19756>

pub mod layer;
pub mod rational;
pub mod spline;

pub use layer::{ActivationType, KanConfig, KanLayer, KanLayerConfig, KanNetwork};
pub use rational::{RationalActivation, RationalConfig};
pub use spline::{BSplineBasis, BSplineActivation, SplineConfig};

use crate::NeuralError;

/// Convenience result type for KAN operations.
pub type KanResult<T> = Result<T, NeuralError>;
