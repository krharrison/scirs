//! GPU memory pooling for efficient buffer reuse
//!
//! This module provides memory pool management for GPU buffers, reducing
//! allocation/deallocation overhead and improving training performance.

use crate::error::{NeuralError, Result};
#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuBuffer, GpuContext, GpuDataType};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Buffer size class for memory pooling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SizeClass {
    /// Size in elements (not bytes)
    pub size: usize,
}

impl SizeClass {
    /// Create a new size class
    pub fn new(size: usize) -> Self {
        Self { size }
    }

    /// Round up size to the nearest power of 2 for efficient pooling
    pub fn from_size_rounded(size: usize) -> Self {
        let rounded = if size == 0 {
            1
        } else {
            size.next_power_of_two()
        };
        Self { size: rounded }
    }

    /// Get the actual size in elements
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Memory pool for GPU buffers (GPU feature required)
#[cfg(feature = "gpu")]
pub struct GpuMemoryPool<T: GpuDataType> {
    /// GPU context for buffer allocation
    gpu_context: Arc<GpuContext>,
    /// Available buffers organized by size class
    available_buffers: Arc<Mutex<HashMap<SizeClass, Vec<GpuBuffer<T>>>>>,
    /// Total allocated memory in bytes
    total_allocated: Arc<AtomicU64>,
    /// Total memory currently in use
    total_in_use: Arc<AtomicU64>,
    /// Peak memory usage
    peak_usage: Arc<AtomicU64>,
    /// Number of allocations
    num_allocations: Arc<AtomicU64>,
    /// Number of deallocations
    num_deallocations: Arc<AtomicU64>,
    /// Number of cache hits
    cache_hits: Arc<AtomicU64>,
    /// Number of cache misses
    cache_misses: Arc<AtomicU64>,
    /// Maximum pool size in bytes (0 = unlimited)
    max_pool_size: u64,
    /// Whether to enable automatic cleanup
    auto_cleanup: bool,
}

#[cfg(feature = "gpu")]
impl<T: GpuDataType> GpuMemoryPool<T> {
    /// Create a new GPU memory pool
    pub fn new(gpu_context: Arc<GpuContext>, max_pool_size: u64) -> Self {
        Self {
            gpu_context,
            available_buffers: Arc::new(Mutex::new(HashMap::new())),
            total_allocated: Arc::new(AtomicU64::new(0)),
            total_in_use: Arc::new(AtomicU64::new(0)),
            peak_usage: Arc::new(AtomicU64::new(0)),
            num_allocations: Arc::new(AtomicU64::new(0)),
            num_deallocations: Arc::new(AtomicU64::new(0)),
            cache_hits: Arc::new(AtomicU64::new(0)),
            cache_misses: Arc::new(AtomicU64::new(0)),
            max_pool_size,
            auto_cleanup: true,
        }
    }

    /// Allocate a buffer from the pool
    pub fn allocate(&self, size: usize) -> Result<PooledBuffer<T>> {
        self.num_allocations.fetch_add(1, Ordering::Relaxed);

        let size_class = SizeClass::from_size_rounded(size);
        let actual_size = size_class.size();

        // Try to get from pool
        let mut buffers = self.available_buffers.lock().map_err(|_| {
            NeuralError::TrainingError("Failed to lock available buffers".to_string())
        })?;

        let buffer = if let Some(pool) = buffers.get_mut(&size_class) {
            if let Some(buf) = pool.pop() {
                // Cache hit
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                buf
            } else {
                // Cache miss - allocate new buffer
                self.cache_misses.fetch_add(1, Ordering::Relaxed);
                drop(buffers); // Release lock before GPU allocation
                self.allocate_new_buffer(actual_size)?
            }
        } else {
            // Cache miss - allocate new buffer
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
            drop(buffers); // Release lock before GPU allocation
            self.allocate_new_buffer(actual_size)?
        };

        // Update usage statistics
        let buffer_size = actual_size * std::mem::size_of::<T>();
        let current_usage = self
            .total_in_use
            .fetch_add(buffer_size as u64, Ordering::Relaxed)
            + buffer_size as u64;

        // Update peak usage
        let mut peak = self.peak_usage.load(Ordering::Relaxed);
        while current_usage > peak {
            match self.peak_usage.compare_exchange_weak(
                peak,
                current_usage,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }

        Ok(PooledBuffer {
            buffer,
            size_class,
            pool: Arc::downgrade(&self.available_buffers),
            total_in_use: Arc::clone(&self.total_in_use),
            num_deallocations: Arc::clone(&self.num_deallocations),
        })
    }

    /// Allocate a new GPU buffer
    fn allocate_new_buffer(&self, size: usize) -> Result<GpuBuffer<T>> {
        let buffer_size = size * std::mem::size_of::<T>();

        // Check pool size limit
        if self.max_pool_size > 0 {
            let total_allocated = self.total_allocated.load(Ordering::Relaxed);
            if total_allocated + buffer_size as u64 > self.max_pool_size {
                // Try to free some memory
                if self.auto_cleanup {
                    self.cleanup_oldest_buffers(buffer_size as u64)?;
                } else {
                    return Err(NeuralError::TrainingError(format!(
                        "Memory pool size limit exceeded: {} + {} > {}",
                        total_allocated, buffer_size, self.max_pool_size
                    )));
                }
            }
        }

        let buffer = self.gpu_context.create_buffer::<T>(size);
        self.total_allocated
            .fetch_add(buffer_size as u64, Ordering::Relaxed);

        Ok(buffer)
    }

    /// Return a buffer to the pool
    fn return_buffer(&self, buffer: GpuBuffer<T>, size_class: SizeClass) -> Result<()> {
        let mut buffers = self.available_buffers.lock().map_err(|_| {
            NeuralError::TrainingError("Failed to lock available buffers".to_string())
        })?;

        buffers
            .entry(size_class)
            .or_insert_with(Vec::new)
            .push(buffer);

        Ok(())
    }

    /// Clean up oldest buffers to free memory
    fn cleanup_oldest_buffers(&self, required_space: u64) -> Result<()> {
        let mut buffers = self.available_buffers.lock().map_err(|_| {
            NeuralError::TrainingError("Failed to lock available buffers".to_string())
        })?;

        let mut freed_space = 0u64;

        // Sort size classes by size (largest first) for efficient cleanup
        let mut size_classes: Vec<_> = buffers.keys().cloned().collect();
        size_classes.sort_by_key(|sc| std::cmp::Reverse(sc.size));

        for size_class in size_classes {
            if freed_space >= required_space {
                break;
            }

            if let Some(pool) = buffers.get_mut(&size_class) {
                while let Some(buffer) = pool.pop() {
                    let buffer_size = (buffer.len() * std::mem::size_of::<T>()) as u64;
                    freed_space += buffer_size;
                    self.total_allocated
                        .fetch_sub(buffer_size, Ordering::Relaxed);

                    if freed_space >= required_space {
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Clear all cached buffers
    pub fn clear(&self) -> Result<()> {
        let mut buffers = self.available_buffers.lock().map_err(|_| {
            NeuralError::TrainingError("Failed to lock available buffers".to_string())
        })?;

        for (size_class, pool) in buffers.drain() {
            let buffer_size = (size_class.size * std::mem::size_of::<T>()) as u64;
            let count = pool.len() as u64;
            self.total_allocated
                .fetch_sub(buffer_size * count, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Get memory pool statistics
    pub fn get_statistics(&self) -> PoolStatistics {
        let buffers = self
            .available_buffers
            .lock()
            .expect("Failed to lock buffers");

        let cached_buffers: usize = buffers.values().map(|pool| pool.len()).sum();

        PoolStatistics {
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
            total_in_use: self.total_in_use.load(Ordering::Relaxed),
            peak_usage: self.peak_usage.load(Ordering::Relaxed),
            cached_buffers,
            num_allocations: self.num_allocations.load(Ordering::Relaxed),
            num_deallocations: self.num_deallocations.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            cache_hit_rate: {
                let hits = self.cache_hits.load(Ordering::Relaxed) as f64;
                let total = hits + self.cache_misses.load(Ordering::Relaxed) as f64;
                if total > 0.0 {
                    hits / total
                } else {
                    0.0
                }
            },
        }
    }

    /// Enable or disable automatic cleanup
    pub fn set_auto_cleanup(&mut self, enabled: bool) {
        self.auto_cleanup = enabled;
    }

    /// Get maximum pool size
    pub fn max_pool_size(&self) -> u64 {
        self.max_pool_size
    }

    /// Set maximum pool size
    pub fn set_max_pool_size(&mut self, size: u64) {
        self.max_pool_size = size;
    }
}

/// Statistics for memory pool
#[derive(Debug, Clone)]
pub struct PoolStatistics {
    /// Total allocated memory in bytes
    pub total_allocated: u64,
    /// Total memory currently in use
    pub total_in_use: u64,
    /// Peak memory usage
    pub peak_usage: u64,
    /// Number of cached buffers
    pub cached_buffers: usize,
    /// Number of allocations
    pub num_allocations: u64,
    /// Number of deallocations
    pub num_deallocations: u64,
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of cache misses
    pub cache_misses: u64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
}

/// A pooled GPU buffer that automatically returns to the pool when dropped (GPU feature required)
#[cfg(feature = "gpu")]
pub struct PooledBuffer<T: GpuDataType> {
    /// The actual GPU buffer
    buffer: GpuBuffer<T>,
    /// Size class for returning to pool
    size_class: SizeClass,
    /// Weak reference to the pool
    pool: std::sync::Weak<Mutex<HashMap<SizeClass, Vec<GpuBuffer<T>>>>>,
    /// Reference to in-use counter
    total_in_use: Arc<AtomicU64>,
    /// Reference to deallocation counter
    num_deallocations: Arc<AtomicU64>,
}

#[cfg(feature = "gpu")]
impl<T: GpuDataType> PooledBuffer<T> {
    /// Get a reference to the underlying buffer
    pub fn buffer(&self) -> &GpuBuffer<T> {
        &self.buffer
    }

    /// Get the size class
    pub fn size_class(&self) -> SizeClass {
        self.size_class
    }
}

#[cfg(feature = "gpu")]
impl<T: GpuDataType> Drop for PooledBuffer<T> {
    fn drop(&mut self) {
        self.num_deallocations.fetch_add(1, Ordering::Relaxed);

        let buffer_size = (self.buffer.len() * std::mem::size_of::<T>()) as u64;
        self.total_in_use.fetch_sub(buffer_size, Ordering::Relaxed);

        // Return buffer to pool if pool still exists
        if let Some(pool) = self.pool.upgrade() {
            if let Ok(mut buffers) = pool.lock() {
                // Clone the buffer and return it to the pool for reuse
                let recycled = self.buffer.clone();
                buffers
                    .entry(self.size_class)
                    .or_insert_with(Vec::new)
                    .push(recycled);
            }
        }
    }
}

#[cfg(feature = "gpu")]
impl<T: GpuDataType> std::ops::Deref for PooledBuffer<T> {
    type Target = GpuBuffer<T>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use scirs2_core::gpu::GpuBackend;

    #[test]
    fn test_size_class() {
        let size_class = SizeClass::new(100);
        assert_eq!(size_class.size(), 100);

        let rounded = SizeClass::from_size_rounded(100);
        assert_eq!(rounded.size(), 128); // Next power of 2
    }

    #[test]
    fn test_memory_pool_creation() {
        let context = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let pool = GpuMemoryPool::<f32>::new(Arc::new(context), 1024 * 1024 * 1024);

        let stats = pool.get_statistics();
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.cached_buffers, 0);
    }

    #[test]
    fn test_buffer_allocation() {
        let context = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let pool = GpuMemoryPool::<f32>::new(Arc::new(context), 1024 * 1024 * 1024);

        let buffer = pool.allocate(1000).expect("Failed to allocate");
        assert_eq!(buffer.size_class().size(), 1024); // Rounded to power of 2

        let stats = pool.get_statistics();
        assert_eq!(stats.num_allocations, 1);
        assert!(stats.total_in_use > 0);
    }

    #[test]
    fn test_buffer_reuse() {
        let context = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let pool = Arc::new(GpuMemoryPool::<f32>::new(
            Arc::new(context),
            1024 * 1024 * 1024,
        ));

        // Allocate and drop buffer
        {
            let _buffer = pool.allocate(1000).expect("Failed to allocate");
        }

        // Allocate again - should reuse
        let _buffer2 = pool.allocate(1000).expect("Failed to allocate");

        let stats = pool.get_statistics();
        assert_eq!(stats.num_allocations, 2);
        assert_eq!(stats.cache_hits, 1);
    }

    #[test]
    fn test_pool_statistics() {
        let context = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let pool = GpuMemoryPool::<f32>::new(Arc::new(context), 1024 * 1024 * 1024);

        let _buffer1 = pool.allocate(1000).expect("Failed to allocate");
        let _buffer2 = pool.allocate(2000).expect("Failed to allocate");

        let stats = pool.get_statistics();
        assert_eq!(stats.num_allocations, 2);
        assert!(stats.total_in_use > 0);
        assert!(stats.peak_usage >= stats.total_in_use);
    }

    #[test]
    fn test_pool_clear() {
        let context = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let pool = GpuMemoryPool::<f32>::new(Arc::new(context), 1024 * 1024 * 1024);

        {
            let _buffer = pool.allocate(1000).expect("Failed to allocate");
        }

        pool.clear().expect("Failed to clear pool");

        let stats = pool.get_statistics();
        assert_eq!(stats.cached_buffers, 0);
    }

    #[test]
    fn test_pooled_buffer_deref() {
        let context = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let pool = GpuMemoryPool::<f32>::new(Arc::new(context), 1024 * 1024 * 1024);

        let buffer = pool.allocate(1000).expect("Failed to allocate");

        // Test deref
        assert!(!buffer.is_empty());
        assert!(!buffer.is_empty());
    }
}
