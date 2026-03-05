//! Second-order optimization methods using automatic differentiation.
//!
//! This module provides quasi-Newton and second-order methods that exploit
//! curvature information to accelerate convergence beyond first-order gradient
//! descent.  All optimizers operate on plain `Vec<f64>` closures so they
//! integrate with any differentiable function, including those produced by
//! the autograd graph.
//!
//! # Available Methods
//!
//! | Type | Method | Description |
//! |------|--------|-------------|
//! | Quasi-Newton | [`LBFGS`] | Limited-memory BFGS with Wolfe line search |
//! | Newton | [`NewtonCG`] | Truncated Newton via conjugate gradients |
//! | Information geometry | [`NaturalGradient`] | Fisher-preconditioned gradient descent |
//! | Trust region | [`TrustRegionOptimizer`] | Steihaug–Toint CG trust region |

pub mod lbfgs;
pub mod natural_gradient;
pub mod newton;
pub mod trust_region;

pub use lbfgs::LBFGS;
pub use natural_gradient::NaturalGradient;
pub use newton::NewtonCG;
pub use trust_region::TrustRegionOptimizer;
