//! Collective Operations for Distributed Computing
//!
//! Simulated MPI-like collective operations using Rust threads, channels, and
//! synchronisation primitives.  All operations are purely in-process and use
//! multi-threading to model what would be network-based collectives in a real
//! distributed system.
//!
//! ## Provided Operations
//!
//! | Operation | Description |
//! |-----------|-------------|
//! | [`AllReduceOp`] | Ring-barrier all-reduce (sum / mean / max) |
//! | [`broadcast`] | Rank-0 sends value to all workers |
//! | [`scatter`] | Distribute equal-sized chunks to workers |
//! | [`gather`] | Collect chunks from all workers onto root (worker 0) |
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::distributed::collective::AllReduceOp;
//! use std::sync::{Arc, Barrier};
//! use std::thread;
//!
//! let n = 4;
//! let op = Arc::new(AllReduceOp::new(n));
//! let barrier = Arc::clone(op.barrier());
//!
//! let handles: Vec<_> = (0..n)
//!     .map(|id| {
//!         let op_ref = Arc::clone(&op);
//!         thread::spawn(move || {
//!             let local = vec![1.0_f64, 2.0, 3.0];
//!             op_ref.all_reduce_sum(&local, id)
//!         })
//!     })
//!     .collect();
//!
//! for h in handles {
//!     let result = h.join().expect("thread panicked").expect("all_reduce_sum failed");
//!     assert_eq!(result, vec![4.0, 8.0, 12.0]); // 4 workers × [1,2,3]
//! }
//! ```

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::sync::{Arc, Barrier, Mutex};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helper: lock a Mutex, converting poison errors to CoreError
// ─────────────────────────────────────────────────────────────────────────────

fn lock_or_err<T>(m: &Mutex<T>) -> CoreResult<std::sync::MutexGuard<'_, T>> {
    m.lock().map_err(|_| {
        CoreError::ComputationError(ErrorContext::new("collective: mutex lock poisoned"))
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// AllReduceOp
// ─────────────────────────────────────────────────────────────────────────────

/// All-reduce operation across worker threads.
///
/// Workers call [`all_reduce_sum`](AllReduceOp::all_reduce_sum),
/// [`all_reduce_mean`](AllReduceOp::all_reduce_mean), or
/// [`all_reduce_max`](AllReduceOp::all_reduce_max) concurrently.  The barrier
/// inside the struct ensures every worker has contributed its local data before
/// anyone reads the global accumulator.
///
/// The shared buffer is pre-allocated to `n_workers * chunk_len` entries so
/// that each worker writes into a distinct slot, avoiding any locking during
/// the write phase.
pub struct AllReduceOp {
    n_workers: usize,
    /// Barrier that synchronises the two phases: write then read.
    barrier: Arc<Barrier>,
    /// Flat buffer laid out as `[worker_0_data, worker_1_data, …]`.
    /// Length is determined on the first call and then fixed for the lifetime
    /// of this op.  Guarded by a Mutex only for the initial resize; the write
    /// phase is single-indexed and therefore safe once the size is set.
    shared_buffer: Arc<Mutex<Vec<f64>>>,
}

impl AllReduceOp {
    /// Create a new `AllReduceOp` for `n_workers` concurrent callers.
    ///
    /// # Panics
    ///
    /// Panics if `n_workers == 0`.
    pub fn new(n_workers: usize) -> Self {
        assert!(n_workers > 0, "n_workers must be > 0");
        Self {
            n_workers,
            barrier: Arc::new(Barrier::new(n_workers)),
            shared_buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Borrow the inner [`Arc<Barrier>`] — useful when callers need to reuse
    /// the same barrier for sequencing outside of this struct.
    pub fn barrier(&self) -> &Arc<Barrier> {
        &self.barrier
    }

    /// Number of workers this op was configured for.
    pub fn n_workers(&self) -> usize {
        self.n_workers
    }

    /// Perform an all-reduce **sum** across all workers.
    ///
    /// Each worker contributes `local_data`; all workers receive the
    /// element-wise sum.  All slices must have the same length.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::ValueError`] if `worker_id >= n_workers` or if
    /// `local_data` is empty.
    /// Returns [`CoreError::ComputationError`] on internal mutex poisoning.
    pub fn all_reduce_sum(&self, local_data: &[f64], worker_id: usize) -> CoreResult<Vec<f64>> {
        self.validate_args(local_data, worker_id)?;
        let chunk_len = local_data.len();

        // ── Phase 1: write local data into the shared buffer ─────────────────
        {
            let mut buf = lock_or_err(&self.shared_buffer)?;
            let required = self.n_workers * chunk_len;
            if buf.len() < required {
                buf.resize(required, 0.0_f64);
            }
            let start = worker_id * chunk_len;
            buf[start..start + chunk_len].copy_from_slice(local_data);
        }

        // ── Synchronise: wait for every worker to write ───────────────────────
        self.barrier.wait();

        // ── Phase 2: reduce ───────────────────────────────────────────────────
        let buf = lock_or_err(&self.shared_buffer)?;
        let result = self.reduce_sum_from_buf(&buf, chunk_len)?;
        drop(buf);

        // ── Synchronise again so no worker can start reusing the buffer while
        //    others are still reading ─────────────────────────────────────────
        self.barrier.wait();

        Ok(result)
    }

    /// Perform an all-reduce **mean** across all workers.
    pub fn all_reduce_mean(&self, local_data: &[f64], worker_id: usize) -> CoreResult<Vec<f64>> {
        let sum = self.all_reduce_sum(local_data, worker_id)?;
        let n = self.n_workers as f64;
        Ok(sum.into_iter().map(|v| v / n).collect())
    }

    /// Perform an all-reduce **max** across all workers (element-wise maximum).
    pub fn all_reduce_max(&self, local_data: &[f64], worker_id: usize) -> CoreResult<Vec<f64>> {
        self.validate_args(local_data, worker_id)?;
        let chunk_len = local_data.len();

        {
            let mut buf = lock_or_err(&self.shared_buffer)?;
            let required = self.n_workers * chunk_len;
            if buf.len() < required {
                buf.resize(required, f64::NEG_INFINITY);
            }
            let start = worker_id * chunk_len;
            buf[start..start + chunk_len].copy_from_slice(local_data);
        }

        self.barrier.wait();

        let buf = lock_or_err(&self.shared_buffer)?;
        let result = self.reduce_max_from_buf(&buf, chunk_len)?;
        drop(buf);

        self.barrier.wait();

        Ok(result)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn validate_args(&self, local_data: &[f64], worker_id: usize) -> CoreResult<()> {
        if worker_id >= self.n_workers {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "worker_id {worker_id} >= n_workers {}",
                self.n_workers
            ))));
        }
        if local_data.is_empty() {
            return Err(CoreError::ValueError(ErrorContext::new(
                "local_data must not be empty",
            )));
        }
        Ok(())
    }

    fn reduce_sum_from_buf(&self, buf: &[f64], chunk_len: usize) -> CoreResult<Vec<f64>> {
        let mut result = vec![0.0_f64; chunk_len];
        for w in 0..self.n_workers {
            let start = w * chunk_len;
            let end = start + chunk_len;
            if end > buf.len() {
                return Err(CoreError::ComputationError(ErrorContext::new(
                    "shared buffer smaller than expected during reduce",
                )));
            }
            for (acc, &val) in result.iter_mut().zip(buf[start..end].iter()) {
                *acc += val;
            }
        }
        Ok(result)
    }

    fn reduce_max_from_buf(&self, buf: &[f64], chunk_len: usize) -> CoreResult<Vec<f64>> {
        let mut result = vec![f64::NEG_INFINITY; chunk_len];
        for w in 0..self.n_workers {
            let start = w * chunk_len;
            let end = start + chunk_len;
            if end > buf.len() {
                return Err(CoreError::ComputationError(ErrorContext::new(
                    "shared buffer smaller than expected during reduce",
                )));
            }
            for (acc, &val) in result.iter_mut().zip(buf[start..end].iter()) {
                if val > *acc {
                    *acc = val;
                }
            }
        }
        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// broadcast
// ─────────────────────────────────────────────────────────────────────────────

/// Broadcast: the source worker (`src_worker`) sends `data` to all workers.
///
/// Returns a `Vec<T>` with `n_workers` elements, all cloned from `data`.
/// This is a local simulation — in a real distributed system each worker would
/// receive exactly one copy; here all copies are returned in a vector indexed
/// by worker id.
///
/// # Panics
///
/// Panics if `n_workers == 0` or `src_worker >= n_workers`.
///
/// # Example
///
/// ```rust
/// use scirs2_core::distributed::collective::broadcast;
///
/// let copies = broadcast(42_u32, 0, 3);
/// assert_eq!(copies, vec![42, 42, 42]);
/// ```
pub fn broadcast<T: Clone + Send>(data: T, src_worker: usize, n_workers: usize) -> Vec<T> {
    assert!(n_workers > 0, "n_workers must be > 0");
    assert!(
        src_worker < n_workers,
        "src_worker {src_worker} >= n_workers {n_workers}"
    );
    (0..n_workers).map(|_| data.clone()).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// scatter
// ─────────────────────────────────────────────────────────────────────────────

/// Scatter: distribute chunks of `data` to `n_workers` workers.
///
/// The data is split as evenly as possible: if `data.len()` is not a multiple
/// of `n_workers`, the last worker receives fewer elements.
///
/// Returns a `Vec<Vec<T>>` of length `n_workers`.
///
/// # Panics
///
/// Panics if `n_workers == 0`.
///
/// # Example
///
/// ```rust
/// use scirs2_core::distributed::collective::scatter;
///
/// let chunks = scatter(&[1, 2, 3, 4, 5, 6], 3);
/// assert_eq!(chunks[0], vec![1, 2]);
/// assert_eq!(chunks[1], vec![3, 4]);
/// assert_eq!(chunks[2], vec![5, 6]);
/// ```
pub fn scatter<T: Clone + Send>(data: &[T], n_workers: usize) -> Vec<Vec<T>> {
    assert!(n_workers > 0, "n_workers must be > 0");
    let total = data.len();
    // base chunk size (floor division); remainder spread over first workers
    let base = total / n_workers;
    let remainder = total % n_workers;

    let mut result = Vec::with_capacity(n_workers);
    let mut offset = 0_usize;
    for w in 0..n_workers {
        // First `remainder` workers get one extra element
        let this_len = if w < remainder { base + 1 } else { base };
        result.push(data[offset..offset + this_len].to_vec());
        offset += this_len;
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// gather
// ─────────────────────────────────────────────────────────────────────────────

/// Gather: collect chunks from all workers onto root (worker 0).
///
/// Each worker calls this function with its own `local_data`.  The function
/// uses a shared `Mutex<Vec<Option<Vec<T>>>>` and a `Barrier` to synchronise.
///
/// - On worker 0 (`worker_id == 0`) returns `Some(gathered_data)`.
/// - On all other workers returns `None`.
///
/// The caller is responsible for creating the shared state before spawning
/// worker threads:
///
/// ```rust
/// use scirs2_core::distributed::collective::gather;
/// use std::sync::{Arc, Barrier, Mutex};
/// use std::thread;
///
/// let n = 4_usize;
/// let barrier = Arc::new(Barrier::new(n));
/// let shared: Arc<Mutex<Vec<Option<Vec<u32>>>>> =
///     Arc::new(Mutex::new(vec![None; n]));
///
/// let handles: Vec<_> = (0..n)
///     .map(|id| {
///         let b = Arc::clone(&barrier);
///         let s = Arc::clone(&shared);
///         thread::spawn(move || gather(vec![id as u32 * 10], id, n, s, b))
///     })
///     .collect();
///
/// let results: Vec<_> = handles.into_iter().map(|h| h.join().expect("should succeed")).collect();
/// // Worker 0 receives Some(gathered); others receive Ok(None)
/// if let Ok(Some(gathered)) = &results[0] {
///     assert_eq!(gathered.len(), 4 * 1); // 4 workers × 1 element each
/// }
/// ```
pub fn gather<T: Clone + Send>(
    local_data: Vec<T>,
    worker_id: usize,
    n_workers: usize,
    shared: Arc<Mutex<Vec<Option<Vec<T>>>>>,
    barrier: Arc<Barrier>,
) -> CoreResult<Option<Vec<T>>> {
    if worker_id >= n_workers {
        return Err(CoreError::ValueError(ErrorContext::new(format!(
            "worker_id {worker_id} >= n_workers {n_workers}"
        ))));
    }

    // ── Write phase ───────────────────────────────────────────────────────────
    {
        let mut buf = shared.lock().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("gather: shared mutex poisoned"))
        })?;
        if buf.len() != n_workers {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "shared buffer has length {} but n_workers is {n_workers}",
                buf.len()
            ))));
        }
        buf[worker_id] = Some(local_data);
    }

    // ── Synchronise ───────────────────────────────────────────────────────────
    barrier.wait();

    // ── Read phase: only root collects ────────────────────────────────────────
    if worker_id == 0 {
        let buf = shared.lock().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("gather: shared mutex poisoned on read"))
        })?;
        let mut gathered: Vec<T> = Vec::new();
        for slot in buf.iter() {
            match slot {
                Some(chunk) => gathered.extend(chunk.iter().cloned()),
                None => {
                    return Err(CoreError::ComputationError(ErrorContext::new(
                        "gather: a worker slot was empty after barrier",
                    )));
                }
            }
        }
        Ok(Some(gathered))
    } else {
        Ok(None)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier, Mutex};
    use std::thread;

    // ── AllReduceOp::all_reduce_sum ───────────────────────────────────────────

    #[test]
    fn test_all_reduce_sum_4_workers() {
        let n = 4_usize;
        let op = Arc::new(AllReduceOp::new(n));

        let handles: Vec<_> = (0..n)
            .map(|id| {
                let op_ref = Arc::clone(&op);
                thread::spawn(move || {
                    // worker i contributes [(i+1)*1.0, (i+1)*2.0, (i+1)*3.0]
                    let local: Vec<f64> = vec![
                        (id + 1) as f64,
                        (id + 1) as f64 * 2.0,
                        (id + 1) as f64 * 3.0,
                    ];
                    op_ref.all_reduce_sum(&local, id)
                })
            })
            .collect();

        // sum over workers: element[0] = 1+2+3+4 = 10, etc.
        for h in handles {
            let result = h
                .join()
                .expect("thread panicked")
                .expect("all_reduce_sum failed");
            assert_eq!(result, vec![10.0, 20.0, 30.0]);
        }
    }

    #[test]
    fn test_all_reduce_sum_single_element() {
        let n = 4_usize;
        let op = Arc::new(AllReduceOp::new(n));

        let handles: Vec<_> = (0..n)
            .map(|id| {
                let op_ref = Arc::clone(&op);
                thread::spawn(move || op_ref.all_reduce_sum(&[1.0_f64], id))
            })
            .collect();

        for h in handles {
            let r = h.join().expect("panic").expect("error");
            // 4 workers × 1.0 = 4.0
            assert_eq!(r, vec![4.0]);
        }
    }

    // ── AllReduceOp::all_reduce_mean ──────────────────────────────────────────

    #[test]
    fn test_all_reduce_mean_4_workers() {
        let n = 4_usize;
        let op = Arc::new(AllReduceOp::new(n));

        // Each worker contributes [4.0]; sum = 16.0; mean = 4.0
        let handles: Vec<_> = (0..n)
            .map(|id| {
                let op_ref = Arc::clone(&op);
                thread::spawn(move || op_ref.all_reduce_mean(&[4.0_f64], id))
            })
            .collect();

        for h in handles {
            let r = h.join().expect("panic").expect("error");
            // sum = 4 * 4.0 = 16.0, mean = 16.0 / 4 = 4.0
            let diff = (r[0] - 4.0_f64).abs();
            assert!(diff < 1e-10, "expected 4.0, got {}", r[0]);
        }
    }

    #[test]
    fn test_all_reduce_mean_heterogeneous() {
        // 4 workers: [1.0], [3.0], [5.0], [7.0] → sum = 16.0, mean = 4.0
        let n = 4_usize;
        let op = Arc::new(AllReduceOp::new(n));
        let inputs: Vec<f64> = vec![1.0, 3.0, 5.0, 7.0];

        let handles: Vec<_> = (0..n)
            .map(|id| {
                let op_ref = Arc::clone(&op);
                let val = inputs[id];
                thread::spawn(move || op_ref.all_reduce_mean(&[val], id))
            })
            .collect();

        for h in handles {
            let r = h.join().expect("panic").expect("error");
            let diff = (r[0] - 4.0_f64).abs();
            assert!(diff < 1e-10, "expected 4.0, got {}", r[0]);
        }
    }

    // ── AllReduceOp::all_reduce_max ───────────────────────────────────────────

    #[test]
    fn test_all_reduce_max() {
        let n = 4_usize;
        let op = Arc::new(AllReduceOp::new(n));
        let inputs: Vec<Vec<f64>> = vec![
            vec![1.0, 9.0],
            vec![3.0, 2.0],
            vec![7.0, 5.0],
            vec![4.0, 8.0],
        ];

        let handles: Vec<_> = (0..n)
            .map(|id| {
                let op_ref = Arc::clone(&op);
                let local = inputs[id].clone();
                thread::spawn(move || op_ref.all_reduce_max(&local, id))
            })
            .collect();

        for h in handles {
            let r = h.join().expect("panic").expect("error");
            assert_eq!(r, vec![7.0, 9.0]);
        }
    }

    // ── broadcast ────────────────────────────────────────────────────────────

    #[test]
    fn test_broadcast_copies_to_all() {
        let copies = broadcast(42_u32, 0, 5);
        assert_eq!(copies.len(), 5);
        assert!(copies.iter().all(|&v| v == 42));
    }

    #[test]
    fn test_broadcast_vec_cloned() {
        let copies = broadcast(vec![1.0_f64, 2.0], 0, 3);
        assert_eq!(copies.len(), 3);
        for c in &copies {
            assert_eq!(c, &vec![1.0_f64, 2.0]);
        }
    }

    // ── scatter ───────────────────────────────────────────────────────────────

    #[test]
    fn test_scatter_even_split() {
        let chunks = scatter(&[10, 20, 30, 40, 50, 60], 3);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], vec![10, 20]);
        assert_eq!(chunks[1], vec![30, 40]);
        assert_eq!(chunks[2], vec![50, 60]);
    }

    #[test]
    fn test_scatter_uneven_split() {
        // 7 elements, 3 workers → [3,2,2]
        let chunks = scatter(&[1, 2, 3, 4, 5, 6, 7], 3);
        assert_eq!(chunks[0], vec![1, 2, 3]);
        assert_eq!(chunks[1], vec![4, 5]);
        assert_eq!(chunks[2], vec![6, 7]);
    }

    #[test]
    fn test_scatter_single_element_each() {
        let chunks = scatter(&[100_u8, 200, 150], 3);
        assert_eq!(chunks[0], vec![100_u8]);
        assert_eq!(chunks[1], vec![200_u8]);
        assert_eq!(chunks[2], vec![150_u8]);
    }

    #[test]
    fn test_scatter_more_workers_than_elements() {
        // 2 elements, 4 workers → two workers get 1 element, two get none
        let chunks = scatter(&[1_i32, 2], 4);
        assert_eq!(chunks[0], vec![1_i32]);
        assert_eq!(chunks[1], vec![2_i32]);
        assert_eq!(chunks[2], Vec::<i32>::new());
        assert_eq!(chunks[3], Vec::<i32>::new());
    }

    // ── gather ────────────────────────────────────────────────────────────────

    #[test]
    fn test_gather_root_collects() {
        let n = 4_usize;
        let barrier = Arc::new(Barrier::new(n));
        let shared: Arc<Mutex<Vec<Option<Vec<u32>>>>> =
            Arc::new(Mutex::new(vec![None; n]));

        let handles: Vec<_> = (0..n)
            .map(|id| {
                let b = Arc::clone(&barrier);
                let s = Arc::clone(&shared);
                thread::spawn(move || {
                    gather(vec![id as u32 * 10, id as u32 * 10 + 1], id, n, s, b)
                })
            })
            .collect();

        let mut root_data: Option<Vec<u32>> = None;
        for (id, h) in handles.into_iter().enumerate() {
            let res = h.join().expect("panic").expect("gather error");
            if id == 0 {
                root_data = res;
            } else {
                assert!(res.is_none(), "non-root worker should return None");
            }
        }

        // Root should have all 4 workers × 2 elements = 8 elements
        let gathered = root_data.expect("root must have data");
        assert_eq!(gathered.len(), 8);
        // Data order: worker 0 → [0,1], worker 1 → [10,11], etc.
        assert_eq!(gathered[0], 0);
        assert_eq!(gathered[1], 1);
        assert_eq!(gathered[2], 10);
    }

    // ── Error paths ───────────────────────────────────────────────────────────

    #[test]
    fn test_all_reduce_invalid_worker_id() {
        let op = AllReduceOp::new(2);
        // We can call directly from the single-threaded test (only one worker)
        // but the barrier would deadlock with n=2; create n=1 for this check.
        let op1 = AllReduceOp::new(1);
        let err = op1.all_reduce_sum(&[1.0], 99);
        assert!(err.is_err());
        // Original op with n=2, just test the validation path directly
        let _ = op; // silence unused warning
    }

    #[test]
    fn test_all_reduce_empty_local_data() {
        let op = AllReduceOp::new(1);
        let err = op.all_reduce_sum(&[], 0);
        assert!(err.is_err());
    }
}
