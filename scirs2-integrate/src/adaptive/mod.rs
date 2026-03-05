//! Adaptive and implicit ODE solvers.
//!
//! This module collects all adaptive step-size and stiff ODE methods
//! available in `scirs2-integrate`.
//!
//! # Sub-modules
//!
//! | Module         | Contents                                                  |
//! |----------------|-----------------------------------------------------------|
//! | [`embedded_rk`]| Dormand-Prince (DOPRI5), Bogacki-Shampine (BS23), Cash-Karp (RKCK) |
//! | [`stiff`]      | Implicit Euler, Trapezoidal (Crank-Nicolson), BDF-2       |
//! | [`events`]     | Zero-crossing detection via the Illinois algorithm        |
//!
//! # Quick start
//!
//! ```rust
//! use scirs2_integrate::adaptive::embedded_rk::dopri5;
//! use scirs2_integrate::adaptive::stiff::implicit_euler;
//! use scirs2_integrate::adaptive::events::{EventDirection, EventSpec, EventSet, dopri5_with_events};
//!
//! // --- Adaptive RK: exponential decay ---
//! let result = dopri5(
//!     |_t, y: &[f64]| vec![-y[0]],
//!     0.0, &[1.0], 5.0, 1e-8, 1e-10,
//! ).expect("dopri5 failed");
//! println!("y(5) ≈ {:.6}", result.y.last().expect("empty")[0]);  // ≈ exp(-5)
//!
//! // --- Implicit Euler: stiff decay ---
//! let result2 = implicit_euler(
//!     |_t, y: &[f64]| vec![-1000.0 * y[0]],
//!     |_t, _y: &[f64]| vec![vec![-1000.0]],
//!     0.0, &[1.0], 0.01, 0.001,
//! ).expect("implicit_euler failed");
//! println!("y(0.01) ≈ {:.6e}", result2.y.last().expect("empty")[0]);
//!
//! // --- Event detection: stop when y < 0.5 ---
//! let threshold = EventSpec {
//!     func: Box::new(|_t, y: &[f64]| y[0] - 0.5),
//!     direction: EventDirection::Falling,
//!     terminal: true,
//! };
//! let ev_result = dopri5_with_events(
//!     |_t, y: &[f64]| vec![-y[0]],
//!     0.0, &[1.0], 5.0, 1e-8, 1e-10,
//!     EventSet::new(vec![threshold]),
//! ).expect("event integration failed");
//! println!("stopped at t ≈ {:.4}", ev_result.ode.t.last().expect("empty"));  // ≈ ln(2)
//! ```

pub mod embedded_rk;
pub mod events;
pub mod stiff;

// Re-export the most commonly used items at the module level for convenience.
pub use embedded_rk::{dopri5, OdeResult};
pub use events::{
    dopri5_with_events, EventDirection, EventResult, EventSet, EventSpec, OdeEventResult,
};
pub use stiff::{bdf2, implicit_euler, trapezoidal};
