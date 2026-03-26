//! GPU-style memory pool with defragmentation and async allocation.
//!
//! This module provides a pure-Rust simulation of GPU memory management patterns,
//! implementing concepts commonly found in graphics and compute APIs:
//!
//! * **Arena allocator** with size-class free lists (see [`arena`])
//! * **Defragmentation** — both full compaction and incremental (see [`defragmenter`])
//! * **Buddy allocator** for large power-of-2 allocations (see [`defragmenter::BuddyAllocator`])
//! * **Async allocation queue** with priority scheduling and pressure callbacks (see [`async_pool`])
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_core::memory_pool::{
//!     arena::ArenaAllocator,
//!     types::{PoolConfig, AsyncAllocRequest},
//!     async_pool::AsyncPool,
//! };
//!
//! // Arena allocator
//! let mut arena = ArenaAllocator::new(PoolConfig::default());
//! let id = arena.alloc(1024, 256).expect("alloc failed");
//! let stats = arena.stats();
//! println!("used: {} / total: {}", stats.used, stats.total);
//! arena.free(id).expect("free failed");
//!
//! // Async pool with priority queue
//! let mut pool = AsyncPool::new(PoolConfig::default());
//! let req = AsyncAllocRequest::new(4096, /* priority */ 10);
//! let handle = pool.enqueue(req).expect("enqueue failed");
//! let completed = pool.process_queue(1);
//! if let Some(alloc_id) = pool.get_result(handle) {
//!     pool.free(alloc_id).expect("free failed");
//! }
//! ```

pub mod arena;
pub mod async_pool;
pub mod defragmenter;
pub mod types;

// Convenient re-exports.
pub use arena::ArenaAllocator;
pub use async_pool::AsyncPool;
pub use defragmenter::{compact, fragmentation_score, incremental_defrag, BuddyAllocator};
pub use types::{
    AllocError, AllocationId, AllocationStats, AsyncAllocRequest, BlockState, MemoryBlock,
    PoolConfig, RequestHandle,
};
