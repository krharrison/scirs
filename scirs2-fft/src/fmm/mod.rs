//! Fast Multipole Method (FMM) for O(N) computation of N-body interactions.
//!
//! The Fast Multipole Method approximates N-body Laplace interactions
//! φ_i = Σ_j q_j · ln|x_i − x_j|  in O(N log N) time — down from O(N²) for
//! direct summation — by combining a hierarchical spatial decomposition
//! (quad-tree) with multipole expansions that compactly represent the potential
//! of a cluster of distant charges.
//!
//! ## Modules
//!
//! * [`tree`]      – Quad-tree (2D) and Oct-tree (3D) data structures.
//! * [`multipole`] – Multipole and local expansions; M2M / M2L / L2L operators.
//! * [`fmm2d`]     – High-level 2D FMM solver using the Laplace kernel.
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_fft::fmm::FMM2D;
//!
//! // 4 unit charges at the corners of a square.
//! let sources: Vec<[f64; 2]> = vec![
//!     [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0],
//! ];
//! let charges = vec![1.0_f64; 4];
//! let targets: Vec<[f64; 2]> = vec![[5.0, 0.0], [0.0, 5.0]];
//!
//! let fmm = FMM2D::new(8);
//! let phi = fmm.compute_potentials(&sources, &charges, &targets).expect("valid input");
//!
//! // Compare to direct summation.
//! let phi_direct = FMM2D::direct_sum(&sources, &charges, &targets);
//! for (a, b) in phi.iter().zip(phi_direct.iter()) {
//!     assert!((a - b).abs() < 0.5);
//! }
//! ```

pub mod fmm2d;
pub mod multipole;
pub mod tree;

pub use fmm2d::FMM2D;
pub use multipole::{LocalExpansion, MultipoleExpansion};
pub use tree::{OctTree, QuadTree, TreeNode};
