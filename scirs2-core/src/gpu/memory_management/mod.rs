//! GPU memory management and buffer pooling
//!
//! This module provides comprehensive memory management for GPU operations including
//! buffer pools, allocators, memory arenas, and transfer optimizations.
//!
//! # Submodules
//!
//! - [`pool`]: Core pool types (`BufferPool`, `BufferAllocator`, `MemoryArena`, error types)
//! - [`transfer`]: Transfer queue, buffer lifetimes, and memory pressure tracking
//! - [`allocators`]: Advanced allocators (`BuddyAllocator`, `SlabAllocator`,
//!   `CompactionAllocator`, `HybridAllocator`) and the `benchmarks` submodule
//!
//! # Advanced Allocator Strategies
//!
//! ## BuddyAllocator
//! Binary buddy system: O(log n) allocation, low fragmentation for medium-large allocations.
//!
//! ## SlabAllocator
//! Fixed-size slabs: O(1) allocation, ideal for small allocations with known sizes.
//!
//! ## CompactionAllocator
//! Defragmenting allocator that can relocate buffers during idle periods.
//!
//! ## HybridAllocator
//! Automatically selects the best strategy based on allocation size.

pub mod allocators;
pub mod pool;
pub mod transfer;

// Re-export all public items to preserve the original flat public API
pub use allocators::{
    benchmarks, BuddyAllocator, BuddyAllocatorStatistics, CompactionAllocator,
    CompactionAllocatorStatistics, HybridAllocator, HybridAllocatorStatistics, SlabAllocator,
    SlabAllocatorStatistics, SlabSizeStatistics,
};
pub use pool::{
    AllocatorStatistics, ArenaAllocation, BufferAllocator, BufferHandle, BufferPool,
    BufferPoolStatistics, EvictionPolicy, MemoryArena, MemoryError, MemoryResult,
};
pub use transfer::{
    BufferLifetime, MemoryPressure, MemoryPressureLevel, TransferDirection, TransferQueue,
    TransferStatistics,
};
