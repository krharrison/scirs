//! Interval arithmetic (validated numerics) for rigorous scientific computing.
//!
//! This module provides *validated numerics* — a technique where every
//! floating-point computation produces an *interval* that is guaranteed to
//! contain the mathematically exact result.  It is the foundation for
//! *verified computing*, where software proofs replace error-prone manual
//! analysis.
//!
//! # Submodules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`interval`] | Core `Interval<T>` type and all arithmetic ops |
//! | [`vector`] | `IntervalVector`, `IntervalMatrix`, and verified linear solving |
//! | [`functions`] | Verified elementary functions and Taylor models |
//!
//! # Quick start
//!
//! ```rust,ignore
//! use scirs2_core::interval::{Interval, IntervalVector, IntervalMatrix};
//! use scirs2_core::interval::functions::{verified_exp, enclose_polynomial};
//!
//! // Arithmetic with outward rounding
//! let a = Interval::new(1.0_f64, 2.0);
//! let b = Interval::new(3.0_f64, 4.0);
//! let c = a + b;
//! assert!(c.contains(5.5));   // 1.5 + 3.5 = 5 — must be enclosed
//!
//! // Elementary functions
//! let e_interval = verified_exp(Interval::new(0.0, 1.0)).expect("should succeed");
//! assert!(e_interval.contains(std::f64::consts::E));
//!
//! // Polynomial range
//! // p(x) = x^2 on [-1, 2]
//! let range = enclose_polynomial(&[0.0, 0.0, 1.0], Interval::new(-1.0, 2.0)).expect("should succeed");
//! assert!(range.lo <= 0.0 && range.hi >= 4.0);
//! ```
//!
//! # Outward rounding
//!
//! Stable Rust does not expose IEEE-754 directed rounding modes.  We instead
//! implement outward rounding conservatively by adjusting each result by one
//! ULP (unit in the last place) after the computation:
//!
//! * Lower bound decremented by one ULP (moves toward −∞).
//! * Upper bound incremented by one ULP (moves toward +∞).
//!
//! This guarantees containment at the cost of intervals that are slightly wider
//! than necessary.  The widening per operation is exactly one ULP, so for
//! reasonably short computations the final interval width is still very tight.
//!
//! # References
//!
//! * Moore, R. E. (1966). *Interval Analysis*. Prentice-Hall.
//! * Neumaier, A. (1990). *Interval Methods for Systems of Equations*.
//!   Cambridge University Press.
//! * Rump, S. M. (1999). INTLAB — INTerval LABoratory. In T. Csendes (Ed.),
//!   *Developments in Reliable Computing*, pp. 77–104. Kluwer Academic.

// Re-export submodules
pub mod interval;
pub mod vector;
pub mod functions;

// Convenience re-exports at the `interval` crate level
pub use interval::{
    next_float,
    prev_float,
    FloatBits,
    Interval,
};

pub use vector::{
    gaussian_elimination_interval,
    IntervalMatrix,
    IntervalVector,
};

pub use functions::{
    compose_verified,
    enclose_polynomial,
    enclose_polynomial_iv,
    polynomial_range,
    taylor_cos,
    taylor_exp,
    taylor_ln,
    taylor_sin,
    taylor_verified,
    TaylorFunction,
    TaylorModel,
    verified_atan,
    verified_cos,
    verified_exp,
    verified_ln,
    verified_sin,
    verified_sqrt,
};
