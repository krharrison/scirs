//! Capsule Networks (CapsNet)
//!
//! Implements the capsule network architecture from:
//! - Sabour, Frosst & Hinton (2017), "Dynamic Routing Between Capsules"
//!   <https://arxiv.org/abs/1710.09829>
//!
//! A capsule is a group of neurons whose **activity vector** encodes both the
//! existence probability (via its length) and the instantiation parameters
//! (via its orientation) of a visual entity.
//!
//! ## Architecture
//!
//! ```text
//! Input features
//!      │
//!  PrimaryCaps          (squash activation, produces u_j)
//!      │
//!  DynamicRouting       (iterative agreement routing, b_ij → c_ij)
//!      │
//!  DigitCaps / ClassCaps (class-level capsules v_i)
//!      │
//!  MarginLoss           (per-class hinge-style reconstruction loss)
//! ```
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_neural::capsule::{CapsNet, CapsNetConfig, MarginLoss};
//!
//! let cfg = CapsNetConfig {
//!     input_size: 16,
//!     primary_caps: 8,
//!     primary_dim: 8,
//!     digit_caps: 10,
//!     digit_dim: 16,
//!     routing_iters: 3,
//! };
//! let net = CapsNet::new(cfg).expect("operation should succeed");
//! let input = vec![0.1_f32; 16];
//! let capsules = net.forward(&input).expect("operation should succeed");
//! println!("Output capsules: {}", capsules.len());
//! ```

pub mod dynamic_routing;
pub mod layers;
pub mod loss;
pub mod network;

// Re-exports
pub use dynamic_routing::DynamicRouting;
pub use layers::{DigitCaps, PrimaryCaps};
pub use loss::MarginLoss;
pub use network::{CapsNet, CapsNetConfig, Capsule};
