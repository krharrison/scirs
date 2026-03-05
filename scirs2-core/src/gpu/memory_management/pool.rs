//! Buffer pool and arena memory management for GPU operations.
//!
//! This submodule provides core pool-based allocation types: [`BufferPool`],
//! [`BufferAllocator`], [`MemoryArena`], and associated error/result types.

use crate::gpu::{GpuBuffer, GpuDataType, GpuError};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MemoryError {
    /// Out of memory
    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    /// Invalid allocation size
    #[error("Invalid allocation size: {0}")]
    InvalidSize(usize),

    /// Buffer not found
    #[error("Buffer not found: {0}")]
    BufferNotFound(u64),

    /// Pool is full
    #[error("Pool is full")]
    PoolFull,

    /// Fragmentation threshold exceeded
    #[error("Fragmentation threshold exceeded: {0:.2}%")]
    FragmentationExceeded(f64),

    /// GPU error
    #[error("GPU error: {0}")]
    GpuError(#[from] GpuError),
}

/// Result type for memory operations
pub type MemoryResult<T> = Result<T, MemoryError>;

/// Buffer handle for tracking allocated buffers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(u64);

impl BufferHandle {
    /// Create a new buffer handle
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Get the raw handle value
    pub fn raw(&self) -> u64 {
        self.0
    }
}

/// Buffer metadata for pool management
#[derive(Debug, Clone)]
struct BufferMetadata {
    handle: BufferHandle,
    size: usize,
    allocated_at: Instant,
    last_used: Instant,
    use_count: usize,
    is_pinned: bool,
}

impl BufferMetadata {
    fn new(handle: BufferHandle, size: usize) -> Self {
        let now = Instant::now();
        Self {
            handle,
            size,
            allocated_at: now,
            last_used: now,
            use_count: 0,
            is_pinned: false,
        }
    }

    fn mark_used(&mut self) {
        self.last_used = Instant::now();
        self.use_count += 1;
    }

    fn age(&self) -> Duration {
        self.allocated_at.elapsed()
    }

    fn idle_time(&self) -> Duration {
        self.last_used.elapsed()
    }
}

/// Buffer pool for reusing GPU buffers
#[derive(Debug)]
pub struct BufferPool<T: GpuDataType> {
    // Size-stratified pools (size -> list of available buffers)
    pools: Arc<Mutex<BTreeMap<usize, VecDeque<(BufferHandle, Arc<GpuBuffer<T>>)>>>>,
    // Active buffers being used
    active_buffers: Arc<Mutex<HashMap<BufferHandle, BufferMetadata>>>,
    // Statistics
    total_allocated: Arc<AtomicUsize>,
    total_reused: Arc<AtomicUsize>,
    max_pool_size: usize,
    eviction_policy: EvictionPolicy,
}

/// Buffer eviction policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    Lru,
    /// Least Frequently Used
    Lfu,
    /// First In First Out
    Fifo,
}

impl<T: GpuDataType> BufferPool<T> {
    /// Create a new buffer pool
    pub fn new() -> Self {
        Self::with_capacity_and_policy(1024, EvictionPolicy::Lru)
    }

    /// Create a new buffer pool with capacity
    pub fn with_capacity(max_pool_size: usize) -> Self {
        Self::with_capacity_and_policy(max_pool_size, EvictionPolicy::Lru)
    }

    /// Create a new buffer pool with capacity and eviction policy
    pub fn with_capacity_and_policy(max_pool_size: usize, eviction_policy: EvictionPolicy) -> Self {
        Self {
            pools: Arc::new(Mutex::new(BTreeMap::new())),
            active_buffers: Arc::new(Mutex::new(HashMap::new())),
            total_allocated: Arc::new(AtomicUsize::new(0)),
            total_reused: Arc::new(AtomicUsize::new(0)),
            max_pool_size,
            eviction_policy,
        }
    }

    /// Allocate a buffer from the pool or create a new one
    pub fn allocate(&self, size: usize) -> MemoryResult<(BufferHandle, Arc<GpuBuffer<T>>)> {
        if size == 0 {
            return Err(MemoryError::InvalidSize(size));
        }

        // Try to reuse an existing buffer
        let mut pools = self.pools.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other("Failed to lock pools".to_string()))
        })?;

        // Look for a buffer of the exact size or slightly larger
        let reusable = pools
            .range_mut(size..)
            .next()
            .and_then(|(_, buffers)| buffers.pop_front());

        if let Some((handle, buffer)) = reusable {
            self.total_reused.fetch_add(1, Ordering::Relaxed);

            let mut active = self.active_buffers.lock().map_err(|_| {
                MemoryError::GpuError(GpuError::Other("Failed to lock active buffers".to_string()))
            })?;

            if let Some(metadata) = active.get_mut(&handle) {
                metadata.mark_used();
            } else {
                let metadata = BufferMetadata::new(handle, size);
                active.insert(handle, metadata);
            }

            return Ok((handle, buffer));
        }

        drop(pools);

        // No reusable buffer found, allocate a new one
        // Note: In a real implementation, this would call the GPU backend
        // For now, we create a placeholder buffer
        self.allocate_new_buffer(size)
    }

    /// Allocate a new buffer
    fn allocate_new_buffer(&self, size: usize) -> MemoryResult<(BufferHandle, Arc<GpuBuffer<T>>)> {
        self.total_allocated.fetch_add(1, Ordering::Relaxed);

        // In a real implementation, this would use the GPU backend to allocate
        // For now, we create a dummy buffer handle
        let handle = BufferHandle::new();

        // Since we can't create a real GpuBuffer without a backend,
        // we'll need to modify this in practice to use the actual GPU context
        // For the type system, we'll return an error indicating this needs backend support
        Err(MemoryError::GpuError(GpuError::Other(
            "Buffer allocation requires GPU backend context".to_string(),
        )))
    }

    /// Return a buffer to the pool
    pub fn deallocate(&self, handle: BufferHandle, buffer: Arc<GpuBuffer<T>>) -> MemoryResult<()> {
        let mut active = self.active_buffers.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other("Failed to lock active buffers".to_string()))
        })?;

        let metadata = active
            .remove(&handle)
            .ok_or(MemoryError::BufferNotFound(handle.raw()))?;

        drop(active);

        let mut pools = self.pools.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other("Failed to lock pools".to_string()))
        })?;

        let pool = pools.entry(metadata.size).or_insert_with(VecDeque::new);

        if pool.len() >= self.max_pool_size {
            // Pool is full, apply eviction policy
            self.evict_buffer(pool)?;
        }

        pool.push_back((handle, buffer));
        Ok(())
    }

    /// Evict a buffer based on the eviction policy
    fn evict_buffer(
        &self,
        pool: &mut VecDeque<(BufferHandle, Arc<GpuBuffer<T>>)>,
    ) -> MemoryResult<()> {
        match self.eviction_policy {
            EvictionPolicy::Lru | EvictionPolicy::Lfu | EvictionPolicy::Fifo => {
                // For FIFO and LRU, remove the front
                pool.pop_front();
                Ok(())
            }
        }
    }

    /// Get pool statistics
    pub fn statistics(&self) -> BufferPoolStatistics {
        let pools = self.pools.lock().expect("Failed to lock pools");
        let active = self
            .active_buffers
            .lock()
            .expect("Failed to lock active buffers");

        let total_pooled: usize = pools.values().map(|v| v.len()).sum();
        let total_active = active.len();

        BufferPoolStatistics {
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
            total_reused: self.total_reused.load(Ordering::Relaxed),
            pooled_buffers: total_pooled,
            active_buffers: total_active,
            pool_size_distribution: pools
                .iter()
                .map(|(size, buffers)| (*size, buffers.len()))
                .collect(),
        }
    }

    /// Clear the pool
    pub fn clear(&self) -> MemoryResult<()> {
        let mut pools = self.pools.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other("Failed to lock pools".to_string()))
        })?;
        pools.clear();
        Ok(())
    }

    /// Get the total number of pooled buffers
    pub fn pooled_count(&self) -> MemoryResult<usize> {
        let pools = self.pools.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other("Failed to lock pools".to_string()))
        })?;
        Ok(pools.values().map(|v| v.len()).sum())
    }
}

impl<T: GpuDataType> Default for BufferPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Buffer pool statistics
#[derive(Debug, Clone)]
pub struct BufferPoolStatistics {
    pub total_allocated: usize,
    pub total_reused: usize,
    pub pooled_buffers: usize,
    pub active_buffers: usize,
    pub pool_size_distribution: Vec<(usize, usize)>,
}

/// Buffer allocator with size stratification
#[derive(Debug)]
pub struct BufferAllocator {
    // Size classes (powers of 2)
    size_classes: Vec<usize>,
    // Memory statistics
    allocated_bytes: Arc<AtomicUsize>,
    freed_bytes: Arc<AtomicUsize>,
    peak_usage: Arc<AtomicUsize>,
    // Fragmentation tracking
    fragmentation_threshold: f64,
}

impl BufferAllocator {
    /// Create a new buffer allocator
    pub fn new() -> Self {
        Self::with_size_classes(Self::default_size_classes())
    }

    /// Create allocator with custom size classes
    pub fn with_size_classes(size_classes: Vec<usize>) -> Self {
        Self {
            size_classes,
            allocated_bytes: Arc::new(AtomicUsize::new(0)),
            freed_bytes: Arc::new(AtomicUsize::new(0)),
            peak_usage: Arc::new(AtomicUsize::new(0)),
            fragmentation_threshold: 0.3, // 30% fragmentation threshold
        }
    }

    /// Get default size classes (powers of 2 from 256 bytes to 1 GB)
    fn default_size_classes() -> Vec<usize> {
        let mut classes = Vec::new();
        let mut size = 256;
        while size <= 1024 * 1024 * 1024 {
            classes.push(size);
            size *= 2;
        }
        classes
    }

    /// Find the appropriate size class for a requested size
    pub fn size_class_for(&self, size: usize) -> usize {
        self.size_classes
            .iter()
            .find(|&&class_size| class_size >= size)
            .copied()
            .unwrap_or_else(|| {
                // Round up to next power of 2
                size.next_power_of_two()
            })
    }

    /// Record an allocation
    pub fn record_allocation(&self, size: usize) {
        let new_allocated = self.allocated_bytes.fetch_add(size, Ordering::Relaxed) + size;

        // Update peak usage
        let mut peak = self.peak_usage.load(Ordering::Relaxed);
        while new_allocated > peak {
            match self.peak_usage.compare_exchange_weak(
                peak,
                new_allocated,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current) => peak = current,
            }
        }
    }

    /// Record a deallocation
    pub fn record_deallocation(&self, size: usize) {
        self.freed_bytes.fetch_add(size, Ordering::Relaxed);
    }

    /// Get current allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes.load(Ordering::Relaxed)
    }

    /// Get freed bytes
    pub fn freed_bytes(&self) -> usize {
        self.freed_bytes.load(Ordering::Relaxed)
    }

    /// Get current usage (allocated - freed)
    pub fn current_usage(&self) -> usize {
        self.allocated_bytes() - self.freed_bytes()
    }

    /// Get peak usage
    pub fn peak_usage(&self) -> usize {
        self.peak_usage.load(Ordering::Relaxed)
    }

    /// Calculate fragmentation ratio
    pub fn fragmentation_ratio(&self) -> f64 {
        let allocated = self.allocated_bytes() as f64;
        let current = self.current_usage() as f64;

        if allocated == 0.0 {
            return 0.0;
        }

        (allocated - current) / allocated
    }

    /// Check if defragmentation is needed
    pub fn needs_defragmentation(&self) -> bool {
        self.fragmentation_ratio() > self.fragmentation_threshold
    }

    /// Get allocator statistics
    pub fn statistics(&self) -> AllocatorStatistics {
        AllocatorStatistics {
            allocated_bytes: self.allocated_bytes(),
            freed_bytes: self.freed_bytes(),
            current_usage: self.current_usage(),
            peak_usage: self.peak_usage(),
            fragmentation_ratio: self.fragmentation_ratio(),
        }
    }
}

impl Default for BufferAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Allocator statistics
#[derive(Debug, Clone)]
pub struct AllocatorStatistics {
    pub allocated_bytes: usize,
    pub freed_bytes: usize,
    pub current_usage: usize,
    pub peak_usage: usize,
    pub fragmentation_ratio: f64,
}

/// Memory arena for temporary scratch space
#[derive(Debug)]
pub struct MemoryArena<T: GpuDataType> {
    // Arena buffers
    buffers: Arc<Mutex<Vec<Arc<GpuBuffer<T>>>>>,
    // Current offset in the active buffer
    current_offset: Arc<AtomicUsize>,
    // Size of each arena buffer
    buffer_size: usize,
    // Total allocated in this arena
    total_allocated: Arc<AtomicUsize>,
}

impl<T: GpuDataType> MemoryArena<T> {
    /// Create a new memory arena
    pub fn new(buffer_size: usize) -> Self {
        Self {
            buffers: Arc::new(Mutex::new(Vec::new())),
            current_offset: Arc::new(AtomicUsize::new(0)),
            buffer_size,
            total_allocated: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Allocate temporary space in the arena
    pub fn allocate_temp(&self, size: usize) -> MemoryResult<ArenaAllocation> {
        if size > self.buffer_size {
            return Err(MemoryError::InvalidSize(size));
        }

        let offset = self.current_offset.fetch_add(size, Ordering::Relaxed);

        // Check if we need a new buffer
        if offset + size > self.buffer_size {
            // Would need to allocate a new buffer in the arena
            // Reset offset and allocate from start of new buffer
            self.current_offset.store(size, Ordering::Relaxed);
            self.total_allocated
                .fetch_add(self.buffer_size, Ordering::Relaxed);

            Ok(ArenaAllocation {
                buffer_index: 0, // In practice, would track which buffer
                offset: 0,
                size,
            })
        } else {
            Ok(ArenaAllocation {
                buffer_index: 0,
                offset,
                size,
            })
        }
    }

    /// Reset the arena (deallocate all temporary allocations)
    pub fn reset(&self) {
        self.current_offset.store(0, Ordering::Relaxed);
    }

    /// Get total allocated bytes
    pub fn total_allocated(&self) -> usize {
        self.total_allocated.load(Ordering::Relaxed)
    }

    /// Get current usage
    pub fn current_usage(&self) -> usize {
        self.current_offset.load(Ordering::Relaxed)
    }
}

/// Arena allocation handle
#[derive(Debug, Clone, Copy)]
pub struct ArenaAllocation {
    buffer_index: usize,
    offset: usize,
    size: usize,
}

impl ArenaAllocation {
    /// Get the offset in the buffer
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the size of the allocation
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the buffer index
    pub fn buffer_index(&self) -> usize {
        self.buffer_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_buffer_handle_creation() {
        let handle1 = BufferHandle::new();
        let handle2 = BufferHandle::new();
        assert_ne!(handle1, handle2);
    }

    #[test]
    fn test_buffer_metadata() {
        let handle = BufferHandle::new();
        let mut metadata = BufferMetadata::new(handle, 1024);

        assert_eq!(metadata.handle, handle);
        assert_eq!(metadata.size, 1024);
        assert_eq!(metadata.use_count, 0);
        assert!(!metadata.is_pinned);

        metadata.mark_used();
        assert_eq!(metadata.use_count, 1);
    }

    #[test]
    fn test_buffer_allocator_size_classes() {
        let allocator = BufferAllocator::new();

        assert_eq!(allocator.size_class_for(100), 256);
        assert_eq!(allocator.size_class_for(256), 256);
        assert_eq!(allocator.size_class_for(257), 512);
        assert_eq!(allocator.size_class_for(1000), 1024);
    }

    #[test]
    fn test_buffer_allocator_statistics() {
        let allocator = BufferAllocator::new();

        allocator.record_allocation(1024);
        allocator.record_allocation(2048);

        assert_eq!(allocator.allocated_bytes(), 3072);
        assert_eq!(allocator.current_usage(), 3072);
        assert_eq!(allocator.peak_usage(), 3072);

        allocator.record_deallocation(1024);
        assert_eq!(allocator.current_usage(), 2048);
        assert_eq!(allocator.freed_bytes(), 1024);
    }

    #[test]
    fn test_buffer_allocator_fragmentation() {
        let allocator = BufferAllocator::new();

        allocator.record_allocation(10000);
        allocator.record_deallocation(3000);

        let frag = allocator.fragmentation_ratio();
        assert!(frag > 0.0 && frag < 1.0);
    }

    #[test]
    fn test_memory_arena() {
        let arena = MemoryArena::<f32>::new(4096);

        let alloc1 = arena.allocate_temp(1024).expect("Failed to allocate");
        assert_eq!(alloc1.size(), 1024);
        assert_eq!(alloc1.offset(), 0);

        let alloc2 = arena.allocate_temp(512).expect("Failed to allocate");
        assert_eq!(alloc2.size(), 512);
        assert_eq!(alloc2.offset(), 1024);

        arena.reset();
        assert_eq!(arena.current_usage(), 0);
    }

    #[test]
    fn test_memory_arena_overflow() {
        let arena = MemoryArena::<f32>::new(1024);

        let result = arena.allocate_temp(2048);
        assert!(result.is_err());
    }

    fn test_eviction_policy() {
        let pool = BufferPool::<f32>::with_capacity_and_policy(10, EvictionPolicy::Lru);
        let stats = pool.statistics();

        assert_eq!(stats.pooled_buffers, 0);
        assert_eq!(stats.active_buffers, 0);
    }
}
