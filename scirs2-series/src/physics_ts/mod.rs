//! Physics-Informed Neural Time Series.
//!
//! Trains a small MLP to fit a time series while simultaneously penalising
//! violations of user-supplied physics constraints such as:
//!
//! - **ODE constraint**: `dx/dt ≈ c · x` (exponential growth / decay)
//! - **Conservation law**: `∑ xᵢ ≈ constant`
//! - **Monotonicity**: predictions must be non-decreasing / non-increasing
//! - **Bounded variation**: consecutive differences `|Δx| ≤ bound`
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use scirs2_series::physics_ts::{
//!     model::PhysicsInformedTs,
//!     types::{PhysicsConstraint, PhysicsTsConfig},
//! };
//!
//! let mut config = PhysicsTsConfig::default();
//! config.n_epochs = 50;
//! let mut model = PhysicsInformedTs::new(config);
//!
//! let times  = vec![0.0_f64, 0.25, 0.5, 0.75, 1.0];
//! let values = vec![1.0_f64, 1.2, 1.4, 1.7, 2.0];
//! let constraints = vec![PhysicsConstraint::Monotone { increasing: true }];
//!
//! let result = model.fit(&times, &values, &constraints).expect("fit");
//! println!("Data loss: {}", result.data_loss);
//! println!("Physics loss: {}", result.total_physics_loss);
//! ```

pub mod model;
pub mod ode_predictor;
pub mod types;

pub use model::PhysicsInformedTs;
pub use ode_predictor::{
    ConservationLaw, ODESystem, PhysicsInformedConfig, PhysicsInformedPredictor, PredictorResult,
};
pub use types::{PhysicsConstraint, PhysicsTsConfig, PhysicsTsResult};
