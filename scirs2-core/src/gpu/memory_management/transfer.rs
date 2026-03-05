//! Transfer queue and buffer lifetime management for GPU memory.
//!
//! This submodule provides [`TransferQueue`], [`BufferLifetime`],
//! and [`MemoryPressure`] for managing GPU memory transfers and lifetimes.

use super::pool::{BufferHandle, MemoryError, MemoryResult};
use crate::gpu::{GpuBuffer, GpuDataType, GpuError};
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

/// Transfer queue for optimizing CPU↔GPU transfers
#[derive(Debug)]
pub struct TransferQueue {
    // Pending transfers
    pending_transfers: Arc<Mutex<VecDeque<Transfer>>>,
    // Completed transfers
    completed_count: Arc<AtomicUsize>,
    // Total bytes transferred
    total_bytes_transferred: Arc<AtomicUsize>,
    // Use pinned memory for transfers
    use_pinned_memory: bool,
}

/// Transfer operation
#[derive(Debug, Clone)]
struct Transfer {
    id: u64,
    direction: TransferDirection,
    size: usize,
    queued_at: Instant,
}

/// Transfer direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

impl TransferQueue {
    /// Create a new transfer queue
    pub fn new() -> Self {
        Self::with_pinned_memory(true)
    }

    /// Create a transfer queue with pinned memory option
    pub fn with_pinned_memory(use_pinned_memory: bool) -> Self {
        Self {
            pending_transfers: Arc::new(Mutex::new(VecDeque::new())),
            completed_count: Arc::new(AtomicUsize::new(0)),
            total_bytes_transferred: Arc::new(AtomicUsize::new(0)),
            use_pinned_memory,
        }
    }

    /// Queue a transfer
    pub fn queue_transfer(&self, direction: TransferDirection, size: usize) -> MemoryResult<u64> {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);

        let transfer = Transfer {
            id,
            direction,
            size,
            queued_at: Instant::now(),
        };

        let mut pending = self.pending_transfers.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other("Failed to lock transfers".to_string()))
        })?;

        pending.push_back(transfer);
        Ok(id)
    }

    /// Process the next transfer
    pub fn process_next(&self) -> MemoryResult<Option<u64>> {
        let mut pending = self.pending_transfers.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other("Failed to lock transfers".to_string()))
        })?;

        if let Some(transfer) = pending.pop_front() {
            self.completed_count.fetch_add(1, Ordering::Relaxed);
            self.total_bytes_transferred
                .fetch_add(transfer.size, Ordering::Relaxed);
            Ok(Some(transfer.id))
        } else {
            Ok(None)
        }
    }

    /// Get the number of pending transfers
    pub fn pending_count(&self) -> MemoryResult<usize> {
        let pending = self.pending_transfers.lock().map_err(|_| {
            MemoryError::GpuError(GpuError::Other("Failed to lock transfers".to_string()))
        })?;
        Ok(pending.len())
    }

    /// Get the number of completed transfers
    pub fn completed_count(&self) -> usize {
        self.completed_count.load(Ordering::Relaxed)
    }

    /// Get total bytes transferred
    pub fn total_bytes_transferred(&self) -> usize {
        self.total_bytes_transferred.load(Ordering::Relaxed)
    }

    /// Check if using pinned memory
    pub fn uses_pinned_memory(&self) -> bool {
        self.use_pinned_memory
    }

    /// Get transfer statistics
    pub fn statistics(&self) -> TransferStatistics {
        TransferStatistics {
            pending_transfers: self.pending_count().unwrap_or(0),
            completed_transfers: self.completed_count(),
            total_bytes_transferred: self.total_bytes_transferred(),
            uses_pinned_memory: self.uses_pinned_memory(),
        }
    }
}

impl Default for TransferQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Transfer queue statistics
#[derive(Debug, Clone)]
pub struct TransferStatistics {
    pub pending_transfers: usize,
    pub completed_transfers: usize,
    pub total_bytes_transferred: usize,
    pub uses_pinned_memory: bool,
}

/// RAII buffer lifetime management
pub struct BufferLifetime<T: GpuDataType> {
    handle: BufferHandle,
    buffer: Arc<GpuBuffer<T>>,
    pool: Weak<Mutex<BTreeMap<usize, VecDeque<(BufferHandle, Arc<GpuBuffer<T>>)>>>>,
    size: usize,
}

impl<T: GpuDataType> BufferLifetime<T> {
    /// Create a new buffer lifetime guard
    pub fn new(
        handle: BufferHandle,
        buffer: Arc<GpuBuffer<T>>,
        pool: Weak<Mutex<BTreeMap<usize, VecDeque<(BufferHandle, Arc<GpuBuffer<T>>)>>>>,
        size: usize,
    ) -> Self {
        Self {
            handle,
            buffer,
            pool,
            size,
        }
    }

    /// Get the buffer handle
    pub fn handle(&self) -> BufferHandle {
        self.handle
    }

    /// Get a reference to the buffer
    pub fn buffer(&self) -> &Arc<GpuBuffer<T>> {
        &self.buffer
    }
}

impl<T: GpuDataType> Drop for BufferLifetime<T> {
    fn drop(&mut self) {
        // Return buffer to pool if pool still exists
        if let Some(pool_arc) = self.pool.upgrade() {
            if let Ok(mut pool) = pool_arc.lock() {
                let buffers = pool.entry(self.size).or_insert_with(VecDeque::new);
                buffers.push_back((self.handle, self.buffer.clone()));
            }
        }
    }
}

/// Memory pressure tracker
#[derive(Debug)]
pub struct MemoryPressure {
    // Current memory usage
    current_usage: Arc<AtomicUsize>,
    // Memory limit
    memory_limit: usize,
    // Pressure thresholds
    warning_threshold: f64,
    critical_threshold: f64,
}

impl MemoryPressure {
    /// Create a new memory pressure tracker
    pub fn new(memory_limit: usize) -> Self {
        Self {
            current_usage: Arc::new(AtomicUsize::new(0)),
            memory_limit,
            warning_threshold: 0.7,  // 70%
            critical_threshold: 0.9, // 90%
        }
    }

    /// Record memory allocation
    pub fn allocate(&self, size: usize) {
        self.current_usage.fetch_add(size, Ordering::Relaxed);
    }

    /// Record memory deallocation
    pub fn deallocate(&self, size: usize) {
        self.current_usage.fetch_sub(size, Ordering::Relaxed);
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_usage.load(Ordering::Relaxed)
    }

    /// Get memory usage ratio
    pub fn usage_ratio(&self) -> f64 {
        self.current_usage() as f64 / self.memory_limit as f64
    }

    /// Get memory pressure level
    pub fn pressure_level(&self) -> MemoryPressureLevel {
        let ratio = self.usage_ratio();

        if ratio >= self.critical_threshold {
            MemoryPressureLevel::Critical
        } else if ratio >= self.warning_threshold {
            MemoryPressureLevel::Warning
        } else {
            MemoryPressureLevel::Normal
        }
    }

    /// Check if under memory pressure
    pub fn is_under_pressure(&self) -> bool {
        matches!(
            self.pressure_level(),
            MemoryPressureLevel::Warning | MemoryPressureLevel::Critical
        )
    }

    /// Get available memory
    pub fn available_memory(&self) -> usize {
        self.memory_limit.saturating_sub(self.current_usage())
    }
}

/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressureLevel {
    Normal,
    Warning,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_transfer_queue() {
        let queue = TransferQueue::new();

        let id1 = queue
            .queue_transfer(TransferDirection::HostToDevice, 1024)
            .expect("Failed to queue transfer");
        let id2 = queue
            .queue_transfer(TransferDirection::DeviceToHost, 2048)
            .expect("Failed to queue transfer");

        assert_eq!(queue.pending_count().expect("Failed to get count"), 2);

        let processed = queue.process_next().expect("Failed to process");
        assert_eq!(processed, Some(id1));

        assert_eq!(queue.pending_count().expect("Failed to get count"), 1);
        assert_eq!(queue.completed_count(), 1);
    }

    #[test]
    fn test_transfer_queue_statistics() {
        let queue = TransferQueue::new();

        queue
            .queue_transfer(TransferDirection::HostToDevice, 1024)
            .expect("Failed to queue");
        queue.process_next().expect("Failed to process");

        let stats = queue.statistics();
        assert_eq!(stats.completed_transfers, 1);
        assert_eq!(stats.total_bytes_transferred, 1024);
        assert!(stats.uses_pinned_memory);
    }

    #[test]
    fn test_memory_pressure() {
        let pressure = MemoryPressure::new(10000);

        assert_eq!(pressure.pressure_level(), MemoryPressureLevel::Normal);
        assert!(!pressure.is_under_pressure());

        pressure.allocate(7500);
        assert_eq!(pressure.pressure_level(), MemoryPressureLevel::Warning);
        assert!(pressure.is_under_pressure());

        pressure.allocate(2000);
        assert_eq!(pressure.pressure_level(), MemoryPressureLevel::Critical);

        pressure.deallocate(5000);
        assert_eq!(pressure.pressure_level(), MemoryPressureLevel::Normal);
    }

    #[test]
    fn test_memory_pressure_available() {
        let pressure = MemoryPressure::new(10000);

        assert_eq!(pressure.available_memory(), 10000);

        pressure.allocate(3000);
        assert_eq!(pressure.available_memory(), 7000);

        pressure.deallocate(1000);
        assert_eq!(pressure.available_memory(), 8000);
    }
}
