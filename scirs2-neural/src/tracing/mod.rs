//! Static computation graph tracing (TorchScript-like).
//!
//! Provides a symbolic tracing API for capturing neural network computations
//! as a directed acyclic graph (DAG) that can be inspected, optimized, and
//! executed independently of the original Rust code that described it.
//!
//! ## Modules
//!
//! - [`types`] — core data structures (`TensorSpec`, `OpNode`, `StaticGraph`, …)
//! - [`graph_builder`] — `GraphBuilder` for recording operations symbolically
//! - [`executor`] — `GraphExecutor` for running a traced graph; `optimize()`

pub mod executor;
pub mod graph_builder;
pub mod types;

pub use executor::{optimize, GraphExecutor};
pub use graph_builder::{GraphBuilder, Tensor};
pub use types::{DType, OpAttr, OpNode, OpType, StaticGraph, TensorSpec, TraceConfig};
