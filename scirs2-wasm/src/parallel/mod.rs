//! Parallel computation utilities using `std::thread` (compatible with
//! `wasm32-wasi` / wasm-threads enabled targets).
//!
//! # SharedArrayBuffer note
//! `SharedArrayBuffer` requires the browser to set `Cross-Origin-Opener-Policy`
//! and `Cross-Origin-Embedder-Policy` headers.  The Rust implementation here
//! uses `std::sync` primitives that map to the underlying platform's threading
//! primitives (pthreads on native, Atomics.wait on wasm-threads).

pub mod coordinator;
pub mod types;

pub use coordinator::{AtomicCounter, ParallelCoordinator};
pub use types::{ParallelConfig, SyncPrimitive, WorkerMessage, WorkerOp};
