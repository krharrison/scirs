//! Types for Physics-Informed Neural Time Series.

/// A physics-based constraint to be enforced during model training.
///
/// Each variant penalises a specific type of constraint violation during
/// training of [`crate::physics_ts::model::PhysicsInformedTs`].
///
/// This enum is `#[non_exhaustive]` — new variants may be added in future
/// versions without breaking existing match arms.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum PhysicsConstraint {
    /// Penalise deviations from `dx/dt = c * x` (exponential growth/decay).
    ///
    /// The constraint loss at collocation point `i` is
    /// `(dx_dt_i - rate * x_i)²`.
    OdeConstraint {
        /// Growth / decay rate coefficient `c`.
        rate: f64,
    },

    /// Enforce conservation: the sum (or integral) of the time series stays
    /// constant.  Penalises `(∑ x - target_sum)²`.
    ConservationLaw {
        /// Expected total sum of the series.
        target_sum: f64,
    },

    /// Enforce monotone increasing (`increasing = true`) or decreasing
    /// (`increasing = false`) predictions.
    ///
    /// Penalises each `max(0, -Δx)` (for increasing) or `max(0, Δx)` (for
    /// decreasing) with a squared penalty.
    Monotone {
        /// If `true`, enforce non-decreasing; otherwise non-increasing.
        increasing: bool,
    },

    /// Enforce bounded variation: `|x_{t+1} - x_t| ≤ bound`.
    ///
    /// Penalises `max(0, |Δx| - bound)²`.
    BoundedVariation {
        /// Maximum permitted step size.
        bound: f64,
    },
}

/// Configuration for [`crate::physics_ts::model::PhysicsInformedTs`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct PhysicsTsConfig {
    /// Number of hidden units per layer in the MLP backbone.  Default: 32.
    pub hidden_dim: usize,
    /// Number of hidden layers.  Default: 2.
    pub n_layers: usize,
    /// Weight `λ` applied to the physics constraint loss.  Default: 1.0.
    pub physics_weight: f64,
    /// Number of training epochs.  Default: 100.
    pub n_epochs: usize,
    /// Adam learning rate.  Default: 1e-3.
    pub learning_rate: f64,
}

impl Default for PhysicsTsConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 32,
            n_layers: 2,
            physics_weight: 1.0,
            n_epochs: 100,
            learning_rate: 1e-3,
        }
    }
}

/// Result returned by [`crate::physics_ts::model::PhysicsInformedTs::fit`] and
/// [`crate::physics_ts::model::PhysicsInformedTs::predict`].
#[derive(Debug, Clone)]
pub struct PhysicsTsResult {
    /// Predicted values at the queried / training time points.
    pub predictions: Vec<f64>,
    /// Per-timestep constraint violation magnitudes.
    pub physics_residuals: Vec<f64>,
    /// Total physics penalty term from all constraints.
    pub total_physics_loss: f64,
    /// Data fit (MSE on observed points).
    pub data_loss: f64,
}
