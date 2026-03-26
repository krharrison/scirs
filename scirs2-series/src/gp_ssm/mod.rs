//! Gaussian Process State-Space Models (GP-SSM)
//!
//! This module provides an implementation of the GP-SSM that uses a Gaussian
//! Process as the latent transition function inside a nonlinear state-space
//! model.  Inference is performed with a Bootstrap Particle Filter (BPF) –
//! also known as a Sequential Importance Resampler (SIR).
//!
//! # Model
//!
//! ```text
//! x_{t+1} = f(x_t) + ε_t,    ε_t ~ N(0, σ_proc² I)
//! y_t     = C x_t  + η_t,    η_t ~ N(0, σ_obs² I)
//! ```
//!
//! where `f` is modelled as a sparse GP with a set of inducing inputs **Z**
//! and an RBF (squared-exponential) kernel.
//!
//! # Reference
//!
//! Turner, R., Deisenroth, M. & Rasmussen, C.E. (2010).
//! "State-space inference and learning with Gaussian processes."
//! *ECML PKDD 2010*, Lecture Notes in AI 6323, pp. 30–45.

pub mod particle_filter;

pub use particle_filter::{GpSsm, GpSsmConfig, GpSsmResult};
