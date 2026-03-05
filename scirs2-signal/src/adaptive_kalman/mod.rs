//! Advanced adaptive filtering algorithms for online signal processing.
//!
//! This module provides a comprehensive suite of adaptive filter implementations
//! covering classical and modern algorithms:
//!
//! | Filter | Algorithm | Complexity | Convergence |
//! |--------|-----------|------------|-------------|
//! | [`LmsFilter`]        | Standard LMS                 | O(N)   | Slow, robust |
//! | [`LmsLeakyFilter`]   | Leaky LMS                    | O(N)   | Bounded weights |
//! | [`LmsSignFilter`]    | Sign-Error LMS               | O(N)   | Integer-friendly |
//! | [`RlsFilter`]        | Recursive Least Squares      | O(N²)  | Very fast |
//! | [`NlmsFilter`]       | Normalised LMS               | O(N)   | Scale-invariant |
//! | [`AffineProjFilter`] | Affine Projection (K-th order)| O(KN+K²)| Fast for correlated |
//!
//! # Quick Start
//!
//! ## LMS
//! ```
//! use scirs2_signal::adaptive_kalman::LmsFilter;
//!
//! let mut lms = LmsFilter::new(8, 0.01).expect("create LMS");
//! for n in 0..100 {
//!     let x = (n as f64 * 0.1).sin();
//!     let d = 0.5 * x;
//!     let y = lms.update(x, d).expect("step");
//! }
//! ```
//!
//! ## RLS
//! ```
//! use scirs2_signal::adaptive_kalman::RlsFilter;
//!
//! let mut rls = RlsFilter::new(8, 0.99, 100.0).expect("create RLS");
//! for n in 0..50 {
//!     let x = (n as f64 * 0.1).sin();
//!     let d = 0.5 * x;
//!     rls.update(x, d).expect("step");
//! }
//! ```

pub mod affine_projection;
pub mod lms;
pub mod nlms;
pub mod rls;

pub use affine_projection::AffineProjFilter;
pub use lms::{LmsFilter, LmsLeakyFilter, LmsSignFilter};
pub use nlms::NlmsFilter;
pub use rls::RlsFilter;
