//! System Identification Module
//!
//! Comprehensive system identification algorithms for estimating mathematical
//! models of dynamic systems from measured input-output data.
//!
//! # Algorithms
//!
//! ## Parametric Models
//! - **ARX**: AutoRegressive with eXogenous input
//! - **ARMAX**: AutoRegressive Moving Average with eXogenous input
//! - **OE**: Output-Error model
//! - **PEM**: Prediction Error Method (general-purpose)
//!
//! ## Subspace Methods
//! - **N4SID**: Numerical algorithms for Subspace State Space System IDentification
//!
//! ## Online / Recursive Methods
//! - **RLS**: Recursive Least Squares with forgetting factor
//!
//! # Example
//!
//! ```rust
//! use scirs2_signal::system_identification::{arx_estimate, ArxConfig};
//! use scirs2_core::ndarray::Array1;
//!
//! // Generate test data from a known system: y[k] = 0.8*y[k-1] + 0.5*u[k-1]
//! let n = 200;
//! let mut u = Array1::<f64>::zeros(n);
//! let mut y = Array1::<f64>::zeros(n);
//! for i in 0..n {
//!     u[i] = ((i as f64) * 0.1).sin();
//! }
//! for i in 1..n {
//!     y[i] = 0.8 * y[i - 1] + 0.5 * u[i - 1];
//! }
//!
//! let config = ArxConfig { na: 1, nb: 1, nk: 1 };
//! let result = arx_estimate(&y, &u, &config).expect("ARX estimation failed");
//! assert!(result.fit_percentage > 90.0);
//! ```

mod armax;
mod arx;
mod n4sid;
mod oe;
mod pem;
mod rls;
mod types;

pub use armax::*;
pub use arx::*;
pub use n4sid::*;
pub use oe::*;
pub use pem::*;
pub use rls::*;
pub use types::*;
