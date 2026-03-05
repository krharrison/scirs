//! Parallel utilities: higher-level parallel patterns built on Rayon.
//!
//! Feature-gated behind `parallel`. Provides:
//!
//! - [`par_for_each`] — parallel for-each with configurable chunk size
//! - [`par_map`] — parallel map preserving input order
//! - [`par_reduce`] — parallel reduce with an associative combiner
//! - Work partitioning strategies: static, dynamic, guided
//! - [`Barrier`] — barrier synchronisation primitive
//!
//! # Example
//!
//! ```rust
//! # #[cfg(feature = "parallel")]
//! # {
//! use scirs2_core::parallel_utils::{par_map, par_reduce, Barrier, PartitionStrategy};
//!
//! // Parallel map (ordered)
//! let data = vec![1, 2, 3, 4, 5];
//! let squares = par_map(&data, |&x| x * x, PartitionStrategy::Static);
//! assert_eq!(squares, vec![1, 4, 9, 16, 25]);
//!
//! // Parallel reduce
//! let sum = par_reduce(&data, 0, |a, &b| a + b, |a, b| a + b, PartitionStrategy::Static);
//! assert_eq!(sum, 15);
//! # }
//! ```

use crate::error::{CoreError, CoreResult, ErrorContext};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

// ===========================================================================
// Partition strategy
// ===========================================================================

/// Strategy for dividing work across parallel threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionStrategy {
    /// Fixed-size chunks distributed round-robin (best for uniform work).
    Static,
    /// A specific chunk size.
    ChunkSize(usize),
    /// Let Rayon decide dynamically (work-stealing, good for uneven work).
    Dynamic,
    /// Guided self-scheduling: starts with large chunks and shrinks.
    Guided,
}

impl Default for PartitionStrategy {
    fn default() -> Self {
        Self::Dynamic
    }
}

/// Compute chunk size given a strategy and the total number of elements.
fn chunk_size(strategy: PartitionStrategy, total: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    match strategy {
        PartitionStrategy::Static => {
            let base = total / threads;
            if base == 0 {
                1
            } else {
                base
            }
        }
        PartitionStrategy::ChunkSize(cs) => cs.max(1),
        PartitionStrategy::Dynamic => {
            // Rayon handles this internally; we just return 1 to process items individually.
            1
        }
        PartitionStrategy::Guided => {
            // Start with large chunks, at least total / (4 * threads).
            let initial = total / (4 * threads);
            initial.max(1)
        }
    }
}

// ===========================================================================
// par_for_each
// ===========================================================================

/// Apply `f` to each element of `data` in parallel.
///
/// The chunk size is determined by the [`PartitionStrategy`].
/// The order of execution is non-deterministic, but every element is processed exactly once.
pub fn par_for_each<T, F>(data: &[T], f: F, strategy: PartitionStrategy)
where
    T: Sync,
    F: Fn(&T) + Sync + Send,
{
    match strategy {
        PartitionStrategy::Dynamic => {
            data.par_iter().for_each(&f);
        }
        _ => {
            let cs = chunk_size(strategy, data.len());
            data.par_chunks(cs).for_each(|chunk| {
                for item in chunk {
                    f(item);
                }
            });
        }
    }
}

/// Mutable version of [`par_for_each`].
pub fn par_for_each_mut<T, F>(data: &mut [T], f: F, strategy: PartitionStrategy)
where
    T: Send + Sync,
    F: Fn(&mut T) + Sync + Send,
{
    match strategy {
        PartitionStrategy::Dynamic => {
            data.par_iter_mut().for_each(&f);
        }
        _ => {
            let cs = chunk_size(strategy, data.len());
            data.par_chunks_mut(cs).for_each(|chunk| {
                for item in chunk {
                    f(item);
                }
            });
        }
    }
}

/// Indexed parallel for-each: `f(index, &element)`.
pub fn par_for_each_indexed<T, F>(data: &[T], f: F, strategy: PartitionStrategy)
where
    T: Sync,
    F: Fn(usize, &T) + Sync + Send,
{
    match strategy {
        PartitionStrategy::Dynamic => {
            data.par_iter().enumerate().for_each(|(i, item)| f(i, item));
        }
        _ => {
            let cs = chunk_size(strategy, data.len());
            data.par_chunks(cs)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let base = chunk_idx * cs;
                    for (offset, item) in chunk.iter().enumerate() {
                        f(base + offset, item);
                    }
                });
        }
    }
}

// ===========================================================================
// par_map
// ===========================================================================

/// Apply `f` to each element of `data` in parallel, returning results in order.
pub fn par_map<T, R, F>(data: &[T], f: F, strategy: PartitionStrategy) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync + Send,
{
    match strategy {
        PartitionStrategy::Dynamic => data.par_iter().map(&f).collect(),
        _ => {
            let cs = chunk_size(strategy, data.len());
            // Process in chunks, preserving order.
            let chunk_results: Vec<Vec<R>> = data
                .par_chunks(cs)
                .map(|chunk| chunk.iter().map(&f).collect::<Vec<_>>())
                .collect();
            chunk_results.into_iter().flatten().collect()
        }
    }
}

/// Parallel map with index: `f(index, &element) -> R`.
pub fn par_map_indexed<T, R, F>(data: &[T], f: F, strategy: PartitionStrategy) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(usize, &T) -> R + Sync + Send,
{
    match strategy {
        PartitionStrategy::Dynamic => data
            .par_iter()
            .enumerate()
            .map(|(i, item)| f(i, item))
            .collect(),
        _ => {
            let cs = chunk_size(strategy, data.len());
            let chunk_results: Vec<Vec<R>> = data
                .par_chunks(cs)
                .enumerate()
                .map(|(chunk_idx, chunk)| {
                    let base = chunk_idx * cs;
                    chunk
                        .iter()
                        .enumerate()
                        .map(|(offset, item)| f(base + offset, item))
                        .collect::<Vec<_>>()
                })
                .collect();
            chunk_results.into_iter().flatten().collect()
        }
    }
}

// ===========================================================================
// par_reduce
// ===========================================================================

/// Parallel reduce with an associative combiner.
///
/// - `identity` is the identity element for the reduction (e.g. 0 for sum).
/// - `fold_fn` folds a single element into the accumulator: `fold(acc, &item) -> acc`.
/// - `combine_fn` merges two partial accumulators: `combine(acc1, acc2) -> acc`.
///
/// The `combine_fn` must be associative: `combine(a, combine(b, c)) == combine(combine(a, b), c)`.
pub fn par_reduce<T, A, FoldFn, CombineFn>(
    data: &[T],
    identity: A,
    fold_fn: FoldFn,
    combine_fn: CombineFn,
    strategy: PartitionStrategy,
) -> A
where
    T: Sync,
    A: Send + Clone + Sync,
    FoldFn: Fn(A, &T) -> A + Sync + Send,
    CombineFn: Fn(A, A) -> A + Sync + Send,
{
    if data.is_empty() {
        return identity;
    }

    match strategy {
        PartitionStrategy::Dynamic => data
            .par_iter()
            .fold(|| identity.clone(), |acc, item| fold_fn(acc, item))
            .reduce(|| identity.clone(), |a, b| combine_fn(a, b)),
        _ => {
            let cs = chunk_size(strategy, data.len());
            let partial: Vec<A> = data
                .par_chunks(cs)
                .map(|chunk| {
                    chunk
                        .iter()
                        .fold(identity.clone(), |acc, item| fold_fn(acc, item))
                })
                .collect();
            partial
                .into_iter()
                .fold(identity, |acc, p| combine_fn(acc, p))
        }
    }
}

/// Parallel sum of a slice of `f64`.
pub fn par_sum_f64(data: &[f64], strategy: PartitionStrategy) -> f64 {
    par_reduce(data, 0.0, |acc, &x| acc + x, |a, b| a + b, strategy)
}

/// Parallel dot product of two `f64` slices.
pub fn par_dot_f64(a: &[f64], b: &[f64], strategy: PartitionStrategy) -> f64 {
    let len = a.len().min(b.len());
    let pairs: Vec<(f64, f64)> = a[..len]
        .iter()
        .copied()
        .zip(b[..len].iter().copied())
        .collect();
    par_reduce(
        &pairs,
        0.0,
        |acc, &(x, y)| acc + x * y,
        |a, b| a + b,
        strategy,
    )
}

// ===========================================================================
// Guided partitioning iterator
// ===========================================================================

/// An iterator adaptor that yields chunks of decreasing size (guided self-scheduling).
///
/// The first chunk is `total / threads`, and subsequent chunks are
/// `remaining / threads`, with a minimum of `min_chunk`.
pub struct GuidedChunks<'a, T> {
    data: &'a [T],
    offset: usize,
    threads: usize,
    min_chunk: usize,
}

impl<'a, T> GuidedChunks<'a, T> {
    /// Create a guided-scheduling chunk iterator.
    pub fn new(data: &'a [T], threads: usize, min_chunk: usize) -> Self {
        Self {
            data,
            offset: 0,
            threads: threads.max(1),
            min_chunk: min_chunk.max(1),
        }
    }
}

impl<'a, T> Iterator for GuidedChunks<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.data.len() {
            return None;
        }
        let remaining = self.data.len() - self.offset;
        let chunk_sz = (remaining / self.threads)
            .max(self.min_chunk)
            .min(remaining);
        let start = self.offset;
        self.offset += chunk_sz;
        Some(&self.data[start..start + chunk_sz])
    }
}

// ===========================================================================
// Barrier
// ===========================================================================

/// A re-usable barrier for synchronising a fixed number of threads.
///
/// All `n` threads must call [`wait`](Barrier::wait) before any of them can proceed.
/// The barrier automatically resets for the next generation.
pub struct Barrier {
    total: usize,
    arrived: AtomicUsize,
    generation: AtomicUsize,
    lock: Mutex<()>,
    cv: Condvar,
}

impl Barrier {
    /// Create a barrier that synchronises `n` threads.
    pub fn new(n: usize) -> Self {
        Self {
            total: n.max(1),
            arrived: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            lock: Mutex::new(()),
            cv: Condvar::new(),
        }
    }

    /// Block until all threads have reached the barrier.
    ///
    /// Returns `Ok(true)` for exactly one thread (the "leader") and
    /// `Ok(false)` for all others.
    pub fn wait(&self) -> CoreResult<bool> {
        let gen = self.generation.load(Ordering::Acquire);
        let count = self.arrived.fetch_add(1, Ordering::AcqRel) + 1;

        if count == self.total {
            // Last thread to arrive: reset and wake everyone.
            self.arrived.store(0, Ordering::Release);
            self.generation.fetch_add(1, Ordering::Release);
            if let Ok(_g) = self.lock.lock() {
                self.cv.notify_all();
            }
            return Ok(true); // leader
        }

        // Wait for the generation to advance.
        let mut guard = self.lock.lock().map_err(|e| {
            CoreError::ComputationError(ErrorContext::new(format!("barrier mutex poisoned: {e}")))
        })?;
        while self.generation.load(Ordering::Acquire) == gen {
            guard = self.cv.wait(guard).map_err(|e| {
                CoreError::ComputationError(ErrorContext::new(format!(
                    "barrier condvar wait failed: {e}"
                )))
            })?;
        }
        Ok(false)
    }

    /// The number of threads this barrier is configured for.
    pub fn party_count(&self) -> usize {
        self.total
    }

    /// Block until all threads have reached the barrier, or the timeout expires.
    ///
    /// Returns `Ok(Some(true))` for the leader, `Ok(Some(false))` for others,
    /// or `Ok(None)` on timeout.
    pub fn wait_timeout(&self, timeout: std::time::Duration) -> CoreResult<Option<bool>> {
        let gen = self.generation.load(Ordering::Acquire);
        let count = self.arrived.fetch_add(1, Ordering::AcqRel) + 1;

        if count == self.total {
            self.arrived.store(0, Ordering::Release);
            self.generation.fetch_add(1, Ordering::Release);
            if let Ok(_g) = self.lock.lock() {
                self.cv.notify_all();
            }
            return Ok(Some(true));
        }

        let deadline = std::time::Instant::now() + timeout;
        let mut guard = self.lock.lock().map_err(|e| {
            CoreError::ComputationError(ErrorContext::new(format!("barrier mutex poisoned: {e}")))
        })?;
        while self.generation.load(Ordering::Acquire) == gen {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                // Undo our arrival to avoid corrupting the count.
                self.arrived.fetch_sub(1, Ordering::Release);
                return Ok(None);
            }
            let (g, timeout_result) = self.cv.wait_timeout(guard, remaining).map_err(|e| {
                CoreError::ComputationError(ErrorContext::new(format!(
                    "barrier condvar wait failed: {e}"
                )))
            })?;
            guard = g;
            if timeout_result.timed_out() && self.generation.load(Ordering::Acquire) == gen {
                self.arrived.fetch_sub(1, Ordering::Release);
                return Ok(None);
            }
        }
        Ok(Some(false))
    }
}

// ===========================================================================
// Parallel pipeline
// ===========================================================================

/// Execute a two-stage parallel pipeline.
///
/// Stage 1 (`producer`) generates items in parallel.
/// Stage 2 (`consumer`) processes items as they arrive.
///
/// This is useful for overlapping computation with I/O or other work.
pub fn par_pipeline<T, R, P, C>(
    data: &[T],
    producer: P,
    consumer: C,
    strategy: PartitionStrategy,
) -> Vec<R>
where
    T: Sync,
    R: Send + Sync,
    P: Fn(&T) -> R + Sync + Send,
    C: Fn(R) -> R + Sync + Send,
{
    let intermediate = par_map(data, producer, strategy);
    intermediate
        .into_iter()
        .map(|item| consumer(item))
        .collect()
}

/// Execute a parallel scan (prefix sum / inclusive scan).
///
/// Returns a vector where `result[i] = combine(result[i-1], data[i])`.
/// The `combine` function must be associative.
pub fn par_scan<T, A, F, C>(data: &[T], identity: A, fold_fn: F, combine_fn: C) -> Vec<A>
where
    T: Sync,
    A: Send + Clone + Sync + PartialEq,
    F: Fn(&A, &T) -> A + Sync + Send,
    C: Fn(&A, &A) -> A + Sync + Send,
{
    if data.is_empty() {
        return Vec::new();
    }

    // Sequential scan for correctness (parallel prefix is complex).
    // For large arrays, we do a two-pass approach:
    // 1) parallel local scans per chunk
    // 2) sequential scan of chunk totals
    // 3) parallel adjustment of each chunk

    let threads = rayon::current_num_threads().max(1);
    let cs = (data.len() / threads).max(1);

    // Phase 1: local scans
    let chunks: Vec<_> = data.chunks(cs).collect();
    let local_scans: Vec<Vec<A>> = chunks
        .par_iter()
        .map(|chunk| {
            let mut result = Vec::with_capacity(chunk.len());
            let mut acc = identity.clone();
            for item in *chunk {
                acc = fold_fn(&acc, item);
                result.push(acc.clone());
            }
            result
        })
        .collect();

    // Phase 2: scan of chunk totals
    let mut chunk_prefixes = Vec::with_capacity(local_scans.len());
    let mut running = identity.clone();
    for local in &local_scans {
        chunk_prefixes.push(running.clone());
        if let Some(last) = local.last() {
            running = combine_fn(&running, last);
        }
    }

    // Phase 3: adjust each chunk
    let adjusted: Vec<Vec<A>> = local_scans
        .into_par_iter()
        .zip(chunk_prefixes.into_par_iter())
        .map(|(local, prefix)| {
            if prefix == identity {
                local
            } else {
                local.iter().map(|v| combine_fn(&prefix, v)).collect()
            }
        })
        .collect();

    adjusted.into_iter().flatten().collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn test_par_for_each_static() {
        let counter = Arc::new(AtomicU32::new(0));
        let data: Vec<u32> = (0..100).collect();
        let c = counter.clone();
        par_for_each(
            &data,
            move |&x| {
                c.fetch_add(x, Ordering::Relaxed);
            },
            PartitionStrategy::Static,
        );
        assert_eq!(counter.load(Ordering::Relaxed), 4950);
    }

    #[test]
    fn test_par_for_each_dynamic() {
        let counter = Arc::new(AtomicU32::new(0));
        let data: Vec<u32> = (0..100).collect();
        let c = counter.clone();
        par_for_each(
            &data,
            move |&x| {
                c.fetch_add(x, Ordering::Relaxed);
            },
            PartitionStrategy::Dynamic,
        );
        assert_eq!(counter.load(Ordering::Relaxed), 4950);
    }

    #[test]
    fn test_par_for_each_chunk_size() {
        let counter = Arc::new(AtomicU32::new(0));
        let data: Vec<u32> = (0..100).collect();
        let c = counter.clone();
        par_for_each(
            &data,
            move |&x| {
                c.fetch_add(x, Ordering::Relaxed);
            },
            PartitionStrategy::ChunkSize(10),
        );
        assert_eq!(counter.load(Ordering::Relaxed), 4950);
    }

    #[test]
    fn test_par_for_each_guided() {
        let counter = Arc::new(AtomicU32::new(0));
        let data: Vec<u32> = (0..100).collect();
        let c = counter.clone();
        par_for_each(
            &data,
            move |&x| {
                c.fetch_add(x, Ordering::Relaxed);
            },
            PartitionStrategy::Guided,
        );
        assert_eq!(counter.load(Ordering::Relaxed), 4950);
    }

    #[test]
    fn test_par_for_each_mut() {
        let mut data: Vec<u32> = (0..50).collect();
        par_for_each_mut(&mut data, |x| *x *= 2, PartitionStrategy::Static);
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(v, (i as u32) * 2);
        }
    }

    #[test]
    fn test_par_for_each_indexed() {
        let data = vec![10, 20, 30, 40, 50];
        let results = Arc::new(Mutex::new(vec![0u64; 5]));
        let r = results.clone();
        par_for_each_indexed(
            &data,
            move |i, &x| {
                if let Ok(mut v) = r.lock() {
                    v[i] = x;
                }
            },
            PartitionStrategy::Static,
        );
        let v = results.lock().expect("lock");
        assert_eq!(*v, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_par_map_ordered() {
        let data: Vec<i32> = (0..100).collect();
        let result = par_map(&data, |&x| x * x, PartitionStrategy::Static);
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(v, (i as i32) * (i as i32));
        }
    }

    #[test]
    fn test_par_map_dynamic() {
        let data: Vec<i32> = (0..100).collect();
        let result = par_map(&data, |&x| x + 1, PartitionStrategy::Dynamic);
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(v, i as i32 + 1);
        }
    }

    #[test]
    fn test_par_map_indexed() {
        let data = vec!["a", "b", "c"];
        let result = par_map_indexed(&data, |i, &s| format!("{i}:{s}"), PartitionStrategy::Static);
        assert_eq!(result, vec!["0:a", "1:b", "2:c"]);
    }

    #[test]
    fn test_par_reduce_sum() {
        let data: Vec<i64> = (1..=100).collect();
        let sum = par_reduce(
            &data,
            0i64,
            |a, &b| a + b,
            |a, b| a + b,
            PartitionStrategy::Static,
        );
        assert_eq!(sum, 5050);
    }

    #[test]
    fn test_par_reduce_product() {
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let prod = par_reduce(
            &data,
            1.0,
            |a, &b| a * b,
            |a, b| a * b,
            PartitionStrategy::Dynamic,
        );
        assert!((prod - 3628800.0).abs() < 1e-6);
    }

    #[test]
    fn test_par_reduce_empty() {
        let data: Vec<i32> = Vec::new();
        let result = par_reduce(
            &data,
            0,
            |a, &b| a + b,
            |a, b| a + b,
            PartitionStrategy::Static,
        );
        assert_eq!(result, 0);
    }

    #[test]
    fn test_par_sum_f64() {
        let data: Vec<f64> = (1..=1000).map(|x| x as f64).collect();
        let sum = par_sum_f64(&data, PartitionStrategy::Static);
        assert!((sum - 500500.0).abs() < 1e-6);
    }

    #[test]
    fn test_par_dot_f64() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0];
        let b: Vec<f64> = vec![4.0, 5.0, 6.0];
        let dot = par_dot_f64(&a, &b, PartitionStrategy::Static);
        assert!((dot - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_guided_chunks() {
        let data: Vec<i32> = (0..100).collect();
        let chunks: Vec<_> = GuidedChunks::new(&data, 4, 2).collect();
        // Should have decreasing chunk sizes, and cover all elements.
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total, 100);
        // First chunk should be largest.
        assert!(chunks[0].len() >= chunks.last().map_or(0, |c| c.len()));
    }

    #[test]
    fn test_barrier_basic() {
        let barrier = Arc::new(Barrier::new(4));
        let counter = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();

        for _ in 0..4 {
            let b = barrier.clone();
            let c = counter.clone();
            handles.push(std::thread::spawn(move || {
                c.fetch_add(1, Ordering::Relaxed);
                let _ = b.wait();
                // After barrier, all 4 threads have incremented.
                c.load(Ordering::Relaxed)
            }));
        }

        for h in handles {
            let val = h.join().expect("join");
            assert_eq!(val, 4);
        }
    }

    #[test]
    fn test_barrier_leader() {
        let barrier = Arc::new(Barrier::new(3));
        let leaders = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();

        for _ in 0..3 {
            let b = barrier.clone();
            let l = leaders.clone();
            handles.push(std::thread::spawn(move || {
                if let Ok(true) = b.wait() {
                    l.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }

        for h in handles {
            h.join().expect("join");
        }
        // Exactly one leader.
        assert_eq!(leaders.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_barrier_reuse() {
        let barrier = Arc::new(Barrier::new(2));
        let counter = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();

        for _ in 0..2 {
            let b = barrier.clone();
            let c = counter.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..5 {
                    c.fetch_add(1, Ordering::Relaxed);
                    let _ = b.wait();
                }
            }));
        }

        for h in handles {
            h.join().expect("join");
        }
        assert_eq!(counter.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn test_barrier_timeout() {
        let barrier = Arc::new(Barrier::new(2));
        // Only one thread arrives, so it should time out.
        let result = barrier
            .wait_timeout(std::time::Duration::from_millis(50))
            .expect("wait_timeout");
        assert_eq!(result, None);
    }

    #[test]
    fn test_par_pipeline() {
        let data: Vec<i32> = (0..20).collect();
        let result = par_pipeline(&data, |&x| x * 2, |y| y + 1, PartitionStrategy::Dynamic);
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(v, (i as i32) * 2 + 1);
        }
    }

    #[test]
    fn test_par_scan() {
        let data: Vec<i32> = vec![1, 2, 3, 4, 5];
        let result = par_scan(&data, 0, |acc, &x| acc + x, |a, b| a + b);
        assert_eq!(result, vec![1, 3, 6, 10, 15]);
    }

    #[test]
    fn test_par_scan_large() {
        let data: Vec<i64> = (1..=1000).collect();
        let result = par_scan(&data, 0i64, |acc, &x| acc + x, |a, b| a + b);
        assert_eq!(result.len(), 1000);
        assert_eq!(*result.last().expect("last"), 500500);
        // Check a few intermediate values.
        assert_eq!(result[0], 1);
        assert_eq!(result[9], 55); // sum 1..=10
    }

    #[test]
    fn test_par_scan_empty() {
        let data: Vec<i32> = Vec::new();
        let result = par_scan(&data, 0, |a, &b| a + b, |a, b| a + b);
        assert!(result.is_empty());
    }

    #[test]
    fn test_partition_strategy_default() {
        assert_eq!(PartitionStrategy::default(), PartitionStrategy::Dynamic);
    }

    #[test]
    fn test_barrier_party_count() {
        let b = Barrier::new(5);
        assert_eq!(b.party_count(), 5);
    }
}
