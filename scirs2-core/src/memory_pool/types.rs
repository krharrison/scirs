//! Core types for the GPU-style memory pool.
//!
//! This module defines the fundamental data structures used throughout the memory pool,
//! including allocation handles, block states, configuration, and error types.

use std::fmt;

/// Unique handle identifying a single allocation in the pool.
///
/// Handles are opaque; callers should treat them as opaque tokens.
/// They are invalidated by compaction unless the caller updates them
/// via the relocation map returned by `compact`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AllocationId(pub(crate) usize);

impl AllocationId {
    /// Return the raw numeric value of this id.
    #[inline]
    pub fn raw(self) -> usize {
        self.0
    }
}

impl fmt::Display for AllocationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AllocationId({})", self.0)
    }
}

/// State of a single memory block inside the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BlockState {
    /// Block is available for allocation.
    Free,
    /// Block is currently in use by a caller.
    Allocated,
    /// Block is partially unusable due to alignment padding or fragmentation.
    Fragmented,
}

/// A contiguous region within the simulated memory heap.
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    /// Unique id for this block.
    pub id: AllocationId,
    /// Byte offset from the start of the heap.
    pub offset: usize,
    /// Size of this block in bytes (may be larger than the requested size due to size-class rounding).
    pub size: usize,
    /// Current lifecycle state.
    pub state: BlockState,
    /// Required byte alignment for the data stored in this block.
    pub alignment: usize,
    /// Actual number of bytes requested by the caller (≤ size).
    pub requested_size: usize,
    /// Size-class index this block belongs to (`usize::MAX` for buddy-allocated blocks).
    pub size_class_index: usize,
}

impl MemoryBlock {
    /// Construct a new block in the `Allocated` state.
    pub fn new(
        id: AllocationId,
        offset: usize,
        size: usize,
        alignment: usize,
        requested_size: usize,
        size_class_index: usize,
    ) -> Self {
        Self {
            id,
            offset,
            size,
            state: BlockState::Allocated,
            alignment,
            requested_size,
            size_class_index,
        }
    }
}

/// Configuration parameters for the memory pool.
///
/// All fields have sensible defaults — use `PoolConfig::default()` and override
/// only the fields you care about.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Total heap size in bytes (default 64 MiB).
    pub total_size: usize,
    /// Minimum allocatable block size in bytes (default 64 B).
    pub min_block_size: usize,
    /// Default alignment in bytes for allocations that do not specify one (default 256 B,
    /// matching typical GPU memory transaction granularity).
    pub alignment: usize,
    /// Fragmentation ratio [0.0, 1.0] above which automatic defragmentation is triggered
    /// (default 0.4 → 40%).
    pub defrag_threshold: f64,
    /// Capacity of the async request queue (default 1024).
    pub async_queue_size: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            total_size: 64 * 1024 * 1024, // 64 MiB
            min_block_size: 64,
            alignment: 256,
            defrag_threshold: 0.4,
            async_queue_size: 1024,
        }
    }
}

/// Snapshot of allocation statistics for the pool.
#[derive(Debug, Clone)]
pub struct AllocationStats {
    /// Total heap capacity in bytes.
    pub total: usize,
    /// Bytes currently allocated to live blocks.
    pub used: usize,
    /// Bytes available for new allocations.
    pub free: usize,
    /// Number of live (allocated) blocks.
    pub n_blocks: usize,
    /// Fragmentation score in [0.0, 1.0]; 0.0 = perfectly compact.
    pub fragmentation: f64,
}

impl AllocationStats {
    /// Utilisation ratio: `used / total`.
    pub fn utilization(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.used as f64 / self.total as f64
        }
    }
}

/// Errors returned by pool operations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AllocError {
    /// The pool does not have enough free space to satisfy the request.
    OutOfMemory {
        /// Bytes requested.
        requested: usize,
        /// Bytes available at the time of the request.
        available: usize,
    },
    /// The requested alignment is invalid (e.g. not a power of two, or zero).
    AlignmentError {
        /// The invalid alignment value.
        alignment: usize,
    },
    /// The given `AllocationId` does not correspond to any live block.
    InvalidId(AllocationId),
    /// The async queue has reached its capacity limit.
    PoolFull,
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocError::OutOfMemory {
                requested,
                available,
            } => write!(
                f,
                "out of memory: requested {} bytes, only {} bytes available",
                requested, available
            ),
            AllocError::AlignmentError { alignment } => {
                write!(
                    f,
                    "invalid alignment: {} (must be a non-zero power of 2)",
                    alignment
                )
            }
            AllocError::InvalidId(id) => write!(f, "invalid allocation id: {}", id),
            AllocError::PoolFull => write!(f, "async allocation queue is full"),
        }
    }
}

impl std::error::Error for AllocError {}

/// An async allocation request that can be enqueued for deferred processing.
#[derive(Debug, Clone)]
pub struct AsyncAllocRequest {
    /// Number of bytes to allocate.
    pub size: usize,
    /// Required alignment (must be a non-zero power of two).
    pub alignment: usize,
    /// Scheduling priority; higher values are processed first.
    pub priority: u8,
}

impl AsyncAllocRequest {
    /// Convenience constructor with default alignment (256 bytes).
    pub fn new(size: usize, priority: u8) -> Self {
        Self {
            size,
            alignment: 256,
            priority,
        }
    }

    /// Constructor with explicit alignment.
    pub fn with_alignment(size: usize, alignment: usize, priority: u8) -> Self {
        Self {
            size,
            alignment,
            priority,
        }
    }
}

/// Opaque handle for a submitted async allocation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestHandle(pub(crate) usize);

impl RequestHandle {
    /// Return the raw numeric value of this handle.
    pub fn raw(self) -> usize {
        self.0
    }
}

impl fmt::Display for RequestHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RequestHandle({})", self.0)
    }
}
