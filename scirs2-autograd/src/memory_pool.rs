//! Tensor memory pool for reducing allocation pressure during training.
//!
//! This module provides a thread-safe memory pool that reuses gradient buffers
//! and intermediate tensor allocations. During training loops, the same shapes
//! are allocated and deallocated repeatedly; the pool caches these buffers
//! by shape so subsequent requests can skip the allocator entirely.
//!
//! # Architecture
//!
//! - **`TensorPool`**: The core pool, keyed by `(shape, TypeId)`. Each bucket
//!   holds a `Vec` of recycled `NdArray<F>` buffers. Protected by a
//!   `parking_lot::Mutex` for low-overhead locking.
//!
//! - **`PooledArray<F>`**: An RAII wrapper around `NdArray<F>` that, on drop,
//!   returns its buffer to the pool for reuse. Implements `Deref` / `DerefMut`
//!   so it can be used transparently wherever `NdArray<F>` is expected.
//!
//! - **`PoolStats`**: Lightweight counters exposed via `TensorPool::stats()`.
//!
//! # Thread Safety
//!
//! The pool is `Send + Sync`. A global singleton is provided via
//! [`global_pool`] for convenience, but callers may also create dedicated
//! pools.
//!
//! # Example
//!
//! ```rust
//! use scirs2_autograd::memory_pool::{global_pool, PooledArray};
//!
//! // Acquire a buffer with shape [64, 128]
//! let buf: PooledArray<f64> = global_pool().acquire(&[64, 128]);
//! assert_eq!(buf.shape(), &[64, 128]);
//!
//! // When `buf` is dropped it is returned to the pool automatically.
//! drop(buf);
//!
//! // Next acquire with the same shape reuses the buffer (zero fresh allocations).
//! let buf2: PooledArray<f64> = global_pool().acquire(&[64, 128]);
//! assert_eq!(buf2.shape(), &[64, 128]);
//! ```

use crate::ndarray_ext::NdArray;
use crate::Float;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::any::TypeId;
use std::collections::HashMap;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// PoolStats
// ---------------------------------------------------------------------------

/// Cumulative statistics for a [`TensorPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolStats {
    /// Total number of `acquire` calls.
    pub n_acquired: u64,
    /// Total number of `release` calls (including automatic drops from `PooledArray`).
    pub n_released: u64,
    /// Number of times a fresh allocation was required (pool miss).
    pub n_allocated: u64,
    /// Number of times an existing buffer was reused (pool hit).
    pub n_reused: u64,
    /// Approximate total bytes currently held *inside* the pool (not checked out).
    pub pool_bytes: u64,
    /// Number of distinct shape buckets.
    pub n_buckets: u64,
    /// Total buffers currently sitting in the pool.
    pub n_pooled_buffers: u64,
}

impl fmt::Display for PoolStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PoolStats {{ acquired: {}, released: {}, allocated: {}, reused: {}, \
             pool_bytes: {}, buckets: {}, pooled_buffers: {} }}",
            self.n_acquired,
            self.n_released,
            self.n_allocated,
            self.n_reused,
            self.pool_bytes,
            self.n_buckets,
            self.n_pooled_buffers,
        )
    }
}

// ---------------------------------------------------------------------------
// BucketKey
// ---------------------------------------------------------------------------

/// A key for the internal bucket map: (shape, element TypeId).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BucketKey {
    shape: Vec<usize>,
    type_id: TypeId,
}

// ---------------------------------------------------------------------------
// TensorPool
// ---------------------------------------------------------------------------

/// Thread-safe pool that caches `NdArray<F>` buffers by shape.
///
/// The pool is cheap to clone (it is internally reference-counted).
pub struct TensorPool {
    inner: Arc<TensorPoolInner>,
}

struct TensorPoolInner {
    /// Buckets keyed by (shape, TypeId).
    buckets: Mutex<HashMap<BucketKey, Vec<ErasedArray>>>,
    // Atomic counters so stats queries do not need the lock.
    n_acquired: AtomicU64,
    n_released: AtomicU64,
    n_allocated: AtomicU64,
    n_reused: AtomicU64,
    /// Maximum number of buffers retained per bucket (0 = unlimited).
    max_per_bucket: usize,
}

/// Type-erased array storage. We store the raw `Vec<u8>` backing plus
/// the shape, and reconstruct the typed array on retrieval.
///
/// Safety: the `data` buffer was originally allocated as `Vec<F>` for some
/// concrete `F: Float`. We only hand it back when the caller requests the
/// same `TypeId`.
struct ErasedArray {
    /// Raw bytes. Length = num_elements * size_of::<F>().
    data: Vec<u8>,
    /// The shape that was used when the array was created.
    shape: Vec<usize>,
    /// size_of one element (used for byte-level accounting).
    elem_size: usize,
}

impl ErasedArray {
    /// Approximate byte size of the buffer.
    fn byte_size(&self) -> usize {
        self.data.len()
    }
}

impl Clone for TensorPool {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

// Explicit Send + Sync: ErasedArray contains only a Vec<u8> and Vec<usize>.
// The Mutex ensures synchronized access.
unsafe impl Send for TensorPoolInner {}
unsafe impl Sync for TensorPoolInner {}

impl fmt::Debug for TensorPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.stats();
        f.debug_struct("TensorPool").field("stats", &stats).finish()
    }
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorPool {
    /// Create a new, empty pool with no per-bucket limit.
    pub fn new() -> Self {
        Self::with_max_per_bucket(0)
    }

    /// Create a new pool that retains at most `max` buffers per shape bucket.
    ///
    /// A value of `0` means unlimited.
    pub fn with_max_per_bucket(max: usize) -> Self {
        Self {
            inner: Arc::new(TensorPoolInner {
                buckets: Mutex::new(HashMap::new()),
                n_acquired: AtomicU64::new(0),
                n_released: AtomicU64::new(0),
                n_allocated: AtomicU64::new(0),
                n_reused: AtomicU64::new(0),
                max_per_bucket: max,
            }),
        }
    }

    /// Acquire a zeroed buffer with the given `shape`.
    ///
    /// If the pool contains a buffer with a matching shape and type it is
    /// reused (and zeroed); otherwise a fresh allocation is made.
    ///
    /// The returned [`PooledArray`] will automatically return its buffer to
    /// this pool when dropped.
    pub fn acquire<F: Float>(&self, shape: &[usize]) -> PooledArray<F> {
        self.inner.n_acquired.fetch_add(1, Ordering::Relaxed);

        let key = BucketKey {
            shape: shape.to_vec(),
            type_id: TypeId::of::<F>(),
        };

        let maybe_erased = {
            let mut buckets = self.inner.buckets.lock();
            buckets.get_mut(&key).and_then(|v| v.pop())
        };

        let array = if let Some(erased) = maybe_erased {
            self.inner.n_reused.fetch_add(1, Ordering::Relaxed);
            erased_to_ndarray::<F>(erased)
        } else {
            self.inner.n_allocated.fetch_add(1, Ordering::Relaxed);
            NdArray::<F>::zeros(scirs2_core::ndarray::IxDyn(shape))
        };

        PooledArray {
            array: Some(array),
            pool: self.clone(),
        }
    }

    /// Manually return a buffer to the pool for later reuse.
    ///
    /// Prefer relying on [`PooledArray`]'s `Drop` impl instead of calling
    /// this directly. This method is useful when you have a bare `NdArray`
    /// obtained from elsewhere.
    pub fn release<F: Float>(&self, array: NdArray<F>) {
        self.inner.n_released.fetch_add(1, Ordering::Relaxed);
        self.release_inner::<F>(array);
    }

    /// Internal release that inserts into the bucket.
    fn release_inner<F: Float>(&self, array: NdArray<F>) {
        let key = BucketKey {
            shape: array.shape().to_vec(),
            type_id: TypeId::of::<F>(),
        };

        let erased = ndarray_to_erased(array);

        let mut buckets = self.inner.buckets.lock();
        let bucket = buckets.entry(key).or_default();

        // Respect max_per_bucket (0 means unlimited).
        if self.inner.max_per_bucket == 0 || bucket.len() < self.inner.max_per_bucket {
            bucket.push(erased);
        }
        // else: buffer is simply dropped (deallocated).
    }

    /// Remove all cached buffers from the pool.
    pub fn clear(&self) {
        let mut buckets = self.inner.buckets.lock();
        buckets.clear();
    }

    /// Return a snapshot of current pool statistics.
    pub fn stats(&self) -> PoolStats {
        let buckets = self.inner.buckets.lock();
        let mut pool_bytes: u64 = 0;
        let mut n_pooled_buffers: u64 = 0;
        for bucket in buckets.values() {
            for erased in bucket {
                pool_bytes = pool_bytes.saturating_add(erased.byte_size() as u64);
            }
            n_pooled_buffers = n_pooled_buffers.saturating_add(bucket.len() as u64);
        }

        PoolStats {
            n_acquired: self.inner.n_acquired.load(Ordering::Relaxed),
            n_released: self.inner.n_released.load(Ordering::Relaxed),
            n_allocated: self.inner.n_allocated.load(Ordering::Relaxed),
            n_reused: self.inner.n_reused.load(Ordering::Relaxed),
            pool_bytes,
            n_buckets: buckets.len() as u64,
            n_pooled_buffers,
        }
    }

    /// Reset the atomic counters (acquired, released, allocated, reused) to
    /// zero. Does *not* clear the pooled buffers.
    pub fn reset_stats(&self) {
        self.inner.n_acquired.store(0, Ordering::Relaxed);
        self.inner.n_released.store(0, Ordering::Relaxed);
        self.inner.n_allocated.store(0, Ordering::Relaxed);
        self.inner.n_reused.store(0, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Erased <-> Typed conversions
// ---------------------------------------------------------------------------

/// Convert a typed `NdArray<F>` into an `ErasedArray` without re-allocating.
fn ndarray_to_erased<F: Float>(array: NdArray<F>) -> ErasedArray {
    let shape = array.shape().to_vec();
    let elem_size = std::mem::size_of::<F>();

    // Convert the ndarray into a flat Vec<F>, then transmute to Vec<u8>.
    let vec_f: Vec<F> = array.into_raw_vec_and_offset().0;
    let len = vec_f.len();
    let cap = vec_f.capacity();

    let ptr = vec_f.as_ptr();
    std::mem::forget(vec_f);

    // Safety: F is Copy + sized, so reinterpreting as bytes is sound.
    let data = unsafe { Vec::from_raw_parts(ptr as *mut u8, len * elem_size, cap * elem_size) };

    ErasedArray {
        data,
        shape,
        elem_size,
    }
}

/// Reconstruct a typed `NdArray<F>` from an `ErasedArray` and zero its contents.
fn erased_to_ndarray<F: Float>(erased: ErasedArray) -> NdArray<F> {
    let elem_size = std::mem::size_of::<F>();
    debug_assert_eq!(erased.elem_size, elem_size);

    let byte_len = erased.data.len();
    let byte_cap = erased.data.capacity();
    let ptr = erased.data.as_ptr();
    std::mem::forget(erased.data);

    let f_len = byte_len / elem_size;
    let f_cap = byte_cap / elem_size;

    // Safety: the bytes were originally produced from a Vec<F>.
    let mut vec_f: Vec<F> = unsafe { Vec::from_raw_parts(ptr as *mut F, f_len, f_cap) };

    // Zero the buffer so the caller gets a clean slate.
    for v in vec_f.iter_mut() {
        *v = F::zero();
    }

    NdArray::<F>::from_shape_vec(scirs2_core::ndarray::IxDyn(&erased.shape), vec_f).unwrap_or_else(
        |_| {
            // Fallback: allocate a fresh zero array. This path should be
            // unreachable because we preserved the original shape.
            NdArray::<F>::zeros(scirs2_core::ndarray::IxDyn(&erased.shape))
        },
    )
}

// ---------------------------------------------------------------------------
// PooledArray
// ---------------------------------------------------------------------------

/// An RAII wrapper around `NdArray<F>` that returns its buffer to a
/// [`TensorPool`] when dropped.
///
/// `PooledArray<F>` dereferences to `NdArray<F>`, so it can be used
/// transparently in any context that expects `&NdArray<F>` or
/// `&mut NdArray<F>`.
pub struct PooledArray<F: Float> {
    /// `Some` while alive; taken in `Drop::drop`.
    array: Option<NdArray<F>>,
    pool: TensorPool,
}

impl<F: Float> PooledArray<F> {
    /// Consume the wrapper and return the inner `NdArray<F>` **without**
    /// returning it to the pool. The caller takes ownership.
    pub fn into_inner(mut self) -> NdArray<F> {
        // Take the array so Drop does not recycle it.
        self.array
            .take()
            .expect("PooledArray inner array already taken")
    }

    /// Get a reference to the inner array shape.
    pub fn shape(&self) -> &[usize] {
        match &self.array {
            Some(a) => a.shape(),
            None => &[],
        }
    }
}

impl<F: Float> Deref for PooledArray<F> {
    type Target = NdArray<F>;

    fn deref(&self) -> &Self::Target {
        self.array
            .as_ref()
            .expect("PooledArray inner array already taken")
    }
}

impl<F: Float> DerefMut for PooledArray<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.array
            .as_mut()
            .expect("PooledArray inner array already taken")
    }
}

impl<F: Float> Drop for PooledArray<F> {
    fn drop(&mut self) {
        if let Some(array) = self.array.take() {
            self.pool.inner.n_released.fetch_add(1, Ordering::Relaxed);
            self.pool.release_inner::<F>(array);
        }
    }
}

impl<F: Float> fmt::Debug for PooledArray<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.array {
            Some(a) => write!(f, "PooledArray(shape={:?})", a.shape()),
            None => write!(f, "PooledArray(<taken>)"),
        }
    }
}

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

/// The process-wide global tensor pool.
static GLOBAL_POOL: Lazy<TensorPool> = Lazy::new(TensorPool::new);

/// Returns a reference to the process-wide global [`TensorPool`].
///
/// This is a convenience for the common case where a single shared pool is
/// sufficient.
pub fn global_pool() -> &'static TensorPool {
    &GLOBAL_POOL
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acquire_returns_zero_array() {
        let pool = TensorPool::new();
        let buf: PooledArray<f64> = pool.acquire(&[3, 4]);
        assert_eq!(buf.shape(), &[3, 4]);
        // All elements should be zero.
        for &v in buf.iter() {
            assert!((v - 0.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_acquire_release_reuse_cycle() {
        let pool = TensorPool::new();

        // First acquire: fresh allocation.
        let buf1: PooledArray<f64> = pool.acquire(&[8, 16]);
        let stats1 = pool.stats();
        assert_eq!(stats1.n_acquired, 1);
        assert_eq!(stats1.n_allocated, 1);
        assert_eq!(stats1.n_reused, 0);

        // Drop returns to pool.
        drop(buf1);
        let stats2 = pool.stats();
        assert_eq!(stats2.n_released, 1);
        assert_eq!(stats2.n_pooled_buffers, 1);

        // Second acquire with same shape: reuse.
        let buf2: PooledArray<f64> = pool.acquire(&[8, 16]);
        let stats3 = pool.stats();
        assert_eq!(stats3.n_acquired, 2);
        assert_eq!(stats3.n_allocated, 1); // still 1
        assert_eq!(stats3.n_reused, 1);

        // The reused buffer should be zeroed.
        for &v in buf2.iter() {
            assert!((v - 0.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_different_shapes_get_different_buckets() {
        let pool = TensorPool::new();

        let a: PooledArray<f64> = pool.acquire(&[2, 3]);
        let b: PooledArray<f64> = pool.acquire(&[3, 2]);

        drop(a);
        drop(b);

        let stats = pool.stats();
        assert_eq!(stats.n_buckets, 2);
        assert_eq!(stats.n_pooled_buffers, 2);
    }

    #[test]
    fn test_different_types_get_different_buckets() {
        let pool = TensorPool::new();

        let a: PooledArray<f32> = pool.acquire(&[4, 4]);
        let b: PooledArray<f64> = pool.acquire(&[4, 4]);

        drop(a);
        drop(b);

        let stats = pool.stats();
        assert_eq!(stats.n_buckets, 2);
    }

    #[test]
    fn test_manual_release() {
        let pool = TensorPool::new();
        let arr: NdArray<f64> = NdArray::zeros(scirs2_core::ndarray::IxDyn(&[5, 5]));
        pool.release(arr);

        let stats = pool.stats();
        assert_eq!(stats.n_released, 1);
        assert_eq!(stats.n_pooled_buffers, 1);

        // Acquire should reuse.
        let buf: PooledArray<f64> = pool.acquire(&[5, 5]);
        let stats2 = pool.stats();
        assert_eq!(stats2.n_reused, 1);
        assert_eq!(stats2.n_allocated, 0);
        drop(buf);
    }

    #[test]
    fn test_clear_empties_pool() {
        let pool = TensorPool::new();

        let a: PooledArray<f64> = pool.acquire(&[10, 10]);
        drop(a);

        assert_eq!(pool.stats().n_pooled_buffers, 1);

        pool.clear();

        assert_eq!(pool.stats().n_pooled_buffers, 0);
        assert_eq!(pool.stats().n_buckets, 0);
    }

    #[test]
    fn test_into_inner_does_not_return_to_pool() {
        let pool = TensorPool::new();

        let buf: PooledArray<f64> = pool.acquire(&[3, 3]);
        let _arr: NdArray<f64> = buf.into_inner();

        // No release should have happened.
        let stats = pool.stats();
        assert_eq!(stats.n_released, 0);
        assert_eq!(stats.n_pooled_buffers, 0);
    }

    #[test]
    fn test_stats_display() {
        let pool = TensorPool::new();
        let _a: PooledArray<f64> = pool.acquire(&[2]);
        let display = format!("{}", pool.stats());
        assert!(display.contains("acquired: 1"));
    }

    #[test]
    fn test_pool_stats_pool_bytes() {
        let pool = TensorPool::new();

        let buf: PooledArray<f64> = pool.acquire(&[100]);
        drop(buf);

        let stats = pool.stats();
        // 100 f64 elements = 800 bytes
        assert_eq!(stats.pool_bytes, 800);
    }

    #[test]
    fn test_reset_stats() {
        let pool = TensorPool::new();

        let buf: PooledArray<f64> = pool.acquire(&[4]);
        drop(buf);

        pool.reset_stats();

        let stats = pool.stats();
        assert_eq!(stats.n_acquired, 0);
        assert_eq!(stats.n_released, 0);
        assert_eq!(stats.n_allocated, 0);
        assert_eq!(stats.n_reused, 0);
        // Buffers are still in the pool (reset_stats does not clear).
        assert_eq!(stats.n_pooled_buffers, 1);
    }

    #[test]
    fn test_max_per_bucket() {
        let pool = TensorPool::with_max_per_bucket(2);

        // Allocate and release 5 buffers of the same shape.
        for _ in 0..5 {
            let buf: PooledArray<f64> = pool.acquire(&[10]);
            drop(buf);
        }

        // Only 2 should be retained.
        assert!(pool.stats().n_pooled_buffers <= 2);
    }

    #[test]
    fn test_global_pool_accessible() {
        let pool = global_pool();
        let _buf: PooledArray<f64> = pool.acquire(&[1]);
        // Just verify it doesn't panic.
    }

    #[test]
    fn test_deref_mut() {
        let pool = TensorPool::new();
        let mut buf: PooledArray<f64> = pool.acquire(&[3]);

        // Write through DerefMut.
        buf[[0]] = 42.0;
        assert!((buf[[0]] - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_debug_format() {
        let pool = TensorPool::new();
        let buf: PooledArray<f64> = pool.acquire(&[2, 3]);
        let dbg = format!("{:?}", buf);
        assert!(dbg.contains("PooledArray"));
        assert!(dbg.contains("[2, 3]"));
    }

    #[test]
    fn test_pool_debug_format() {
        let pool = TensorPool::new();
        let dbg = format!("{:?}", pool);
        assert!(dbg.contains("TensorPool"));
    }

    #[test]
    fn test_pool_clone_shares_state() {
        let pool1 = TensorPool::new();
        let pool2 = pool1.clone();

        let buf: PooledArray<f64> = pool1.acquire(&[4]);
        drop(buf);

        // pool2 should see the same stats because they share the Arc.
        let stats = pool2.stats();
        assert_eq!(stats.n_acquired, 1);
        assert_eq!(stats.n_released, 1);
        assert_eq!(stats.n_pooled_buffers, 1);
    }

    #[test]
    fn test_scalar_shape() {
        let pool = TensorPool::new();
        let buf: PooledArray<f64> = pool.acquire(&[]);
        assert_eq!(buf.shape(), &[] as &[usize]);
        drop(buf);

        let buf2: PooledArray<f64> = pool.acquire(&[]);
        assert_eq!(pool.stats().n_reused, 1);
        drop(buf2);
    }

    #[test]
    fn test_f32_pool() {
        let pool = TensorPool::new();
        let buf: PooledArray<f32> = pool.acquire(&[5, 5]);
        assert_eq!(buf.shape(), &[5, 5]);
        for &v in buf.iter() {
            assert!((v - 0.0f32).abs() < f32::EPSILON);
        }
        drop(buf);

        let stats = pool.stats();
        assert_eq!(stats.pool_bytes, 100); // 25 * 4 bytes
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let pool = Arc::new(TensorPool::new());
        let n_threads = 8;
        let n_ops_per_thread = 100;

        let mut handles = Vec::with_capacity(n_threads);

        for _ in 0..n_threads {
            let pool = Arc::clone(&pool);
            handles.push(thread::spawn(move || {
                for i in 0..n_ops_per_thread {
                    // Use a few different shapes to create contention.
                    let shape = match i % 3 {
                        0 => vec![16, 32],
                        1 => vec![32, 16],
                        _ => vec![64],
                    };
                    let mut buf: PooledArray<f64> = pool.acquire(&shape);
                    // Do a tiny bit of work using the first element via iter_mut.
                    if let Some(v) = buf.iter_mut().next() {
                        *v = 1.0;
                    }
                    drop(buf);
                }
            }));
        }

        for h in handles {
            h.join().expect("thread panicked");
        }

        let stats = pool.stats();
        assert_eq!(stats.n_acquired, (n_threads * n_ops_per_thread) as u64,);
        assert_eq!(stats.n_acquired, stats.n_allocated + stats.n_reused);
        assert_eq!(stats.n_released, stats.n_acquired);
    }

    #[test]
    fn test_concurrent_mixed_types() {
        use std::sync::Arc;
        use std::thread;

        let pool = Arc::new(TensorPool::new());
        let n_threads = 4;
        let n_ops = 50;

        let mut handles = Vec::with_capacity(n_threads * 2);

        // f64 threads
        for _ in 0..n_threads {
            let pool = Arc::clone(&pool);
            handles.push(thread::spawn(move || {
                for _ in 0..n_ops {
                    let buf: PooledArray<f64> = pool.acquire(&[8, 8]);
                    drop(buf);
                }
            }));
        }

        // f32 threads
        for _ in 0..n_threads {
            let pool = Arc::clone(&pool);
            handles.push(thread::spawn(move || {
                for _ in 0..n_ops {
                    let buf: PooledArray<f32> = pool.acquire(&[8, 8]);
                    drop(buf);
                }
            }));
        }

        for h in handles {
            h.join().expect("thread panicked");
        }

        let stats = pool.stats();
        let total_ops = (n_threads * 2 * n_ops) as u64;
        assert_eq!(stats.n_acquired, total_ops);
    }

    #[test]
    fn test_large_shape() {
        let pool = TensorPool::new();
        let buf: PooledArray<f64> = pool.acquire(&[256, 256]);
        assert_eq!(buf.shape(), &[256, 256]);
        assert_eq!(buf.len(), 256 * 256);
        drop(buf);

        let stats = pool.stats();
        assert_eq!(stats.pool_bytes, (256 * 256 * 8) as u64);
    }

    #[test]
    fn test_reused_buffer_is_zeroed() {
        let pool = TensorPool::new();

        // Acquire, fill with non-zero, release.
        let mut buf: PooledArray<f64> = pool.acquire(&[4]);
        buf[[0]] = 99.0;
        buf[[1]] = 88.0;
        buf[[2]] = 77.0;
        buf[[3]] = 66.0;
        drop(buf);

        // Re-acquire: should be zeroed.
        let buf2: PooledArray<f64> = pool.acquire(&[4]);
        for &v in buf2.iter() {
            assert!((v - 0.0).abs() < f64::EPSILON, "expected zero, got {}", v);
        }
    }

    #[test]
    fn test_multiple_buffers_same_shape() {
        let pool = TensorPool::new();

        // Release multiple buffers of the same shape.
        for _ in 0..5 {
            let arr: NdArray<f64> = NdArray::zeros(scirs2_core::ndarray::IxDyn(&[3]));
            pool.release(arr);
        }

        assert_eq!(pool.stats().n_pooled_buffers, 5);

        // Acquire them all at once (keeping each alive so it is not returned).
        let mut held: Vec<PooledArray<f64>> = Vec::with_capacity(5);
        for i in 0..5 {
            held.push(pool.acquire(&[3]));
            assert_eq!(pool.stats().n_pooled_buffers, 4 - i as u64);
        }
        // Drop all at once; all 5 buffers are returned to the pool.
        drop(held);
        assert_eq!(pool.stats().n_pooled_buffers, 5);
    }
}
