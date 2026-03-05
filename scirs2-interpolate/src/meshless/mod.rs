//! Meshless (mesh-free) interpolation methods
//!
//! This module groups advanced interpolation techniques that do not require a
//! mesh or grid structure.  They operate directly on scattered point clouds
//! in any number of dimensions.
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`rbf_interpolant`] | Advanced RBF: polyharmonic splines, Wendland CSRBFs, RBF-FD, Hermite RBF |
//! | [`kriging_interp`]  | Kriging variants: ordinary, universal, co-kriging, external drift |
//! | [`partition_unity`] | Partition-of-Unity (PU) blended local RBF patches |
//! | [`shepard`]         | Shepard's method: global IDW, modified Shepard, Franke-Little weights |
//!
//! ## Quick Reference
//!
//! ### RBF with Polynomial Augmentation
//!
//! ```rust,ignore
//! use scirs2_interpolate::meshless::rbf_interpolant::{
//!     GlobalRbfInterpolant, PolyharmonicKernel, PolyDegree,
//! };
//! use scirs2_core::ndarray::{array, Array2};
//!
//! let pts = Array2::from_shape_vec((4, 2), vec![
//!     0.0, 0.0,  1.0, 0.0,  0.0, 1.0,  1.0, 1.0,
//! ]).expect("doc example: should succeed");
//! let vals = array![0.0, 1.0, 1.0, 2.0];
//! let interp = GlobalRbfInterpolant::new_polyharmonic(
//!     &pts.view(), &vals.view(),
//!     PolyharmonicKernel::ThinPlate, PolyDegree::Linear,
//! ).expect("doc example: should succeed");
//! let v = interp.evaluate(&[0.5, 0.5]).expect("doc example: should succeed");
//! assert!((v - 1.0).abs() < 1e-9);
//! ```
//!
//! ### Ordinary Kriging
//!
//! ```rust,ignore
//! use scirs2_interpolate::meshless::kriging_interp::{OrdinaryKriging, Variogram};
//! use scirs2_core::ndarray::{array, Array2};
//!
//! let pts = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).expect("doc example: should succeed");
//! let vals = array![0.0, 1.0, 4.0, 9.0];
//! let vgm = Variogram::Gaussian { nugget: 0.0, sill: 1.0, range: 5.0 };
//! let ok = OrdinaryKriging::new(&pts.view(), &vals.view(), vgm).expect("doc example: should succeed");
//! let (pred, var) = ok.predict(&[1.5]).expect("doc example: should succeed");
//! ```
//!
//! ### Partition of Unity
//!
//! ```rust,ignore
//! use scirs2_interpolate::meshless::partition_unity::{
//!     PartitionUnityInterpolant, BumpFunction,
//! };
//! // ... (see module docs)
//! ```
//!
//! ### Shepard's Method
//!
//! ```rust,ignore
//! use scirs2_interpolate::meshless::shepard::{ShepardInterpolant, ShepardMode};
//! // ...
//! ```

pub mod kriging_interp;
pub mod partition_unity;
pub mod rbf_interpolant;
pub mod shepard;

// ---------------------------------------------------------------------------
// Convenience re-exports
// ---------------------------------------------------------------------------

// RBF interpolant types
pub use rbf_interpolant::{
    GlobalRbfInterpolant, HermiteRbfInterpolant, PolyDegree, PolyharmonicKernel,
    WendlandInterpolant, WendlandKernel, rbf_fd_weights,
};

// Kriging types
pub use kriging_interp::{
    CoKriging, KrigingExternalDrift, OrdinaryKriging, TrendDegree, UniversalKriging, Variogram,
};

// Partition of unity types
pub use partition_unity::{BumpFunction, PartitionUnityInterpolant};

// Shepard types
pub use shepard::{
    ShepardInterpolant, ShepardMode,
    basic_shepard, franke_little_shepard, modified_shepard,
};
