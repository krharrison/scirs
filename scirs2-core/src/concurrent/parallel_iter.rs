//! Parallel iterators over slices and owned `Vec`s.
//!
//! All functions in this module use OS threads (no Rayon dependency) and
//! fall back to sequential execution for small inputs or when only one CPU
//! is available.
//!
//! # Provided operations
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`parallel_map`] | Apply `f` to every element; preserves order. |
//! | [`parallel_reduce`] | Reduce with a commutative binary op; associativity required. |
//! | [`parallel_filter`] | Retain elements matching a predicate. |
//! | [`parallel_scan`] | Inclusive/exclusive prefix scan. |
//! | [`parallel_merge_sort`] | In-place parallel merge sort. |
//! | [`parallel_for_each`] | Execute a closure for each element (side-effects). |
//! | [`parallel_partition`] | Partition elements into two `Vec`s based on a predicate. |
//! | [`parallel_prefix_sum`] | Specialised f64 prefix-sum using Blelloch's algorithm. |
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::concurrent::parallel_iter::{parallel_map, parallel_reduce, parallel_scan, ScanMode};
//!
//! let data: Vec<i64> = (1..=8).collect();
//! let doubled = parallel_map(&data, |&x| x * 2, 0).expect("map");
//! assert_eq!(doubled, vec![2, 4, 6, 8, 10, 12, 14, 16]);
//!
//! let sum = parallel_reduce(&data, 0i64, |a, b| a + b, |a, b| a + b, 0).expect("reduce");
//! assert_eq!(sum, 36);
//!
//! let prefix = parallel_scan(&data, 0i64, |a, b| a + b, ScanMode::Inclusive, 0).expect("scan");
//! assert_eq!(prefix, vec![1, 3, 6, 10, 15, 21, 28, 36]);
//! ```

use std::sync::{Arc, Mutex};
use std::thread;

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};

// ── helpers ──────────────────────────────────────────────────────────────────

/// Resolve `n_threads` to a concrete thread count (0 → hardware concurrency).
pub fn resolve_threads(n_threads: usize) -> usize {
    if n_threads == 0 {
        thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    } else {
        n_threads
    }
}

/// Split `len` items into `n_chunks` index ranges.
fn chunk_ranges(len: usize, n_chunks: usize) -> Vec<std::ops::Range<usize>> {
    let n = n_chunks.max(1);
    let base = len / n;
    let rem = len % n;
    let mut ranges = Vec::with_capacity(n);
    let mut start = 0;
    for i in 0..n {
        let extra = if i < rem { 1 } else { 0 };
        let end = (start + base + extra).min(len);
        if start < len {
            ranges.push(start..end);
        }
        start = start + base + extra;
    }
    ranges
}

fn spawn_err(e: impl std::fmt::Display) -> CoreError {
    CoreError::SchedulerError(
        ErrorContext::new(format!("failed to spawn thread: {e}"))
            .with_location(ErrorLocation::new(file!(), line!())),
    )
}

fn join_err(label: &'static str) -> CoreError {
    CoreError::SchedulerError(
        ErrorContext::new(format!("{label}: worker thread panicked"))
            .with_location(ErrorLocation::new(file!(), line!())),
    )
}

// ── parallel_map ─────────────────────────────────────────────────────────────

/// Apply `f` to every element of `data` in parallel, returning results in the
/// same order as the input.
///
/// `n_threads = 0` uses hardware concurrency.
pub fn parallel_map<T, R, F>(data: &[T], f: F, n_threads: usize) -> CoreResult<Vec<R>>
where
    T: Sync + 'static,
    R: Send + Default + Clone + 'static,
    F: Fn(&T) -> R + Send + Sync + 'static,
{
    let n = data.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let n_threads = resolve_threads(n_threads).min(n);

    // Allocate output vector upfront.
    let mut out: Vec<R> = vec![R::default(); n];

    // SAFETY: each chunk has a disjoint range of `out`.  We use raw pointers
    // to hand non-overlapping slices to threads.
    let data_ptr = data.as_ptr() as usize;
    let out_ptr = out.as_mut_ptr() as usize;
    let f = Arc::new(f);
    let ranges = chunk_ranges(n, n_threads);

    let mut handles = Vec::with_capacity(ranges.len());
    for range in ranges {
        let f2 = Arc::clone(&f);
        let handle = thread::Builder::new()
            .spawn(move || {
                let data: &[T] =
                    // SAFETY: range is within `data`, unique per thread.
                    unsafe { std::slice::from_raw_parts(data_ptr as *const T, n) };
                let out: &mut [R] =
                    // SAFETY: range is within `out`, unique per thread.
                    unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut R, n) };
                for i in range {
                    out[i] = f2(&data[i]);
                }
            })
            .map_err(spawn_err)?;
        handles.push(handle);
    }

    for h in handles {
        h.join().map_err(|_| join_err("parallel_map"))?;
    }

    Ok(out)
}

// ── parallel_for_each ────────────────────────────────────────────────────────

/// Execute `f` for each element of `data` in parallel (fire-and-forget,
/// side-effects only).  Order of execution is not guaranteed.
pub fn parallel_for_each<T, F>(data: &[T], f: F, n_threads: usize) -> CoreResult<()>
where
    T: Sync + 'static,
    F: Fn(&T) + Send + Sync + 'static,
{
    let n = data.len();
    if n == 0 {
        return Ok(());
    }

    let n_threads = resolve_threads(n_threads).min(n);
    let data_ptr = data.as_ptr() as usize;
    let f = Arc::new(f);
    let mut handles = Vec::new();

    for range in chunk_ranges(n, n_threads) {
        let f2 = Arc::clone(&f);
        let handle = thread::Builder::new()
            .spawn(move || {
                let data: &[T] = unsafe { std::slice::from_raw_parts(data_ptr as *const T, n) };
                for i in range {
                    f2(&data[i]);
                }
            })
            .map_err(spawn_err)?;
        handles.push(handle);
    }

    for h in handles {
        h.join().map_err(|_| join_err("parallel_for_each"))?;
    }
    Ok(())
}

// ── parallel_reduce ───────────────────────────────────────────────────────────

/// Parallel reduction using a chunk-local reduce followed by a sequential
/// combine across chunks.
///
/// - `fold`: combine one element `T` into accumulator `R` (called per thread).
/// - `combine`: merge two accumulators (called sequentially to merge chunks).
/// - `identity`: neutral element for `combine`.
pub fn parallel_reduce<T, R, Fold, Combine>(
    data: &[T],
    identity: R,
    fold: Fold,
    combine: Combine,
    n_threads: usize,
) -> CoreResult<R>
where
    T: Sync + 'static,
    R: Send + Clone + 'static,
    Fold: Fn(R, &T) -> R + Send + Sync + 'static,
    Combine: Fn(R, R) -> R,
{
    let n = data.len();
    if n == 0 {
        return Ok(identity);
    }

    let n_threads = resolve_threads(n_threads).min(n);
    let data_ptr = data.as_ptr() as usize;
    let fold = Arc::new(fold);
    let results: Arc<Mutex<Vec<(usize, R)>>> = Arc::new(Mutex::new(Vec::new()));
    let ranges = chunk_ranges(n, n_threads);
    let mut handles = Vec::with_capacity(ranges.len());

    for (chunk_id, range) in ranges.into_iter().enumerate() {
        let f2 = Arc::clone(&fold);
        let results2 = Arc::clone(&results);
        let id = identity.clone();
        let handle = thread::Builder::new()
            .spawn(move || {
                let data: &[T] = unsafe { std::slice::from_raw_parts(data_ptr as *const T, n) };
                let local = data[range].iter().fold(id, |acc, x| f2(acc, x));
                if let Ok(mut g) = results2.lock() {
                    g.push((chunk_id, local));
                }
            })
            .map_err(spawn_err)?;
        handles.push(handle);
    }

    for h in handles {
        h.join().map_err(|_| join_err("parallel_reduce"))?;
    }

    let mut partials = Arc::try_unwrap(results)
        .map_err(|_| {
            CoreError::SchedulerError(ErrorContext::new("parallel_reduce: Arc still held"))
        })?
        .into_inner()
        .map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("parallel_reduce: mutex poisoned: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

    // Sort by chunk_id to ensure deterministic combine order.
    partials.sort_by_key(|(id, _)| *id);
    let result = partials
        .into_iter()
        .fold(identity, |acc, (_, r)| combine(acc, r));
    Ok(result)
}

// ── parallel_filter ───────────────────────────────────────────────────────────

/// Retain elements matching `pred` in parallel.
///
/// The order of elements in the output matches the order in the input.
pub fn parallel_filter<T, F>(data: Vec<T>, pred: F, n_threads: usize) -> CoreResult<Vec<T>>
where
    T: Send + 'static,
    F: Fn(&T) -> bool + Send + Sync + 'static,
{
    let n = data.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let n_threads = resolve_threads(n_threads).min(n);
    let pred = Arc::new(pred);
    let ranges = chunk_ranges(n, n_threads);
    let n_chunks = ranges.len();

    // Move data into Arc<Mutex<Vec<Option<T>>>> so threads can take elements.
    let data: Vec<Option<T>> = data.into_iter().map(Some).collect();
    let shared: Arc<Mutex<Vec<Option<T>>>> = Arc::new(Mutex::new(data));
    let chunk_results: Arc<Mutex<Vec<(usize, Vec<T>)>>> = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::with_capacity(n_chunks);

    for (chunk_id, range) in ranges.into_iter().enumerate() {
        let p2 = Arc::clone(&pred);
        let sh = Arc::clone(&shared);
        let cr = Arc::clone(&chunk_results);

        let handle = thread::Builder::new()
            .spawn(move || {
                // Extract our chunk's items.
                let items: Vec<T> = {
                    if let Ok(mut g) = sh.lock() {
                        range.filter_map(|i| g[i].take()).collect()
                    } else {
                        Vec::new()
                    }
                };
                let kept: Vec<T> = items.into_iter().filter(|x| p2(x)).collect();
                if let Ok(mut g) = cr.lock() {
                    g.push((chunk_id, kept));
                }
            })
            .map_err(spawn_err)?;
        handles.push(handle);
    }

    for h in handles {
        h.join().map_err(|_| join_err("parallel_filter"))?;
    }

    let mut partials = Arc::try_unwrap(chunk_results)
        .map_err(|_| {
            CoreError::SchedulerError(ErrorContext::new("parallel_filter: Arc still held"))
        })?
        .into_inner()
        .map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("parallel_filter: mutex poisoned: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

    partials.sort_by_key(|(id, _)| *id);
    Ok(partials.into_iter().flat_map(|(_, v)| v).collect())
}

// ── parallel_partition ────────────────────────────────────────────────────────

/// Partition `data` into `(matching, non_matching)` in parallel.
///
/// The order within each partition preserves the original input order.
pub fn parallel_partition<T, F>(
    data: Vec<T>,
    pred: F,
    n_threads: usize,
) -> CoreResult<(Vec<T>, Vec<T>)>
where
    T: Send + 'static,
    F: Fn(&T) -> bool + Send + Sync + 'static,
{
    let n = data.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let n_threads = resolve_threads(n_threads).min(n);
    let pred = Arc::new(pred);
    let ranges = chunk_ranges(n, n_threads);

    let shared: Arc<Mutex<Vec<Option<T>>>> =
        Arc::new(Mutex::new(data.into_iter().map(Some).collect()));
    let yes_chunks: Arc<Mutex<Vec<(usize, Vec<T>)>>> = Arc::new(Mutex::new(Vec::new()));
    let no_chunks: Arc<Mutex<Vec<(usize, Vec<T>)>>> = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();

    for (chunk_id, range) in ranges.into_iter().enumerate() {
        let p2 = Arc::clone(&pred);
        let sh = Arc::clone(&shared);
        let yc = Arc::clone(&yes_chunks);
        let nc = Arc::clone(&no_chunks);

        let handle = thread::Builder::new()
            .spawn(move || {
                let items: Vec<T> = {
                    if let Ok(mut g) = sh.lock() {
                        range.filter_map(|i| g[i].take()).collect()
                    } else {
                        Vec::new()
                    }
                };
                let (yes, no): (Vec<T>, Vec<T>) = items.into_iter().partition(|x| p2(x));
                if let Ok(mut g) = yc.lock() {
                    g.push((chunk_id, yes));
                }
                if let Ok(mut g) = nc.lock() {
                    g.push((chunk_id, no));
                }
            })
            .map_err(spawn_err)?;
        handles.push(handle);
    }

    for h in handles {
        h.join().map_err(|_| join_err("parallel_partition"))?;
    }

    let mut yes = Arc::try_unwrap(yes_chunks)
        .map_err(|_| {
            CoreError::SchedulerError(ErrorContext::new("parallel_partition: yes Arc held"))
        })?
        .into_inner()
        .map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("parallel_partition: mutex poisoned: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
    let mut no = Arc::try_unwrap(no_chunks)
        .map_err(|_| {
            CoreError::SchedulerError(ErrorContext::new("parallel_partition: no Arc held"))
        })?
        .into_inner()
        .map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("parallel_partition: no mutex poisoned: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

    yes.sort_by_key(|(id, _)| *id);
    no.sort_by_key(|(id, _)| *id);

    let yes_flat: Vec<T> = yes.into_iter().flat_map(|(_, v)| v).collect();
    let no_flat: Vec<T> = no.into_iter().flat_map(|(_, v)| v).collect();
    Ok((yes_flat, no_flat))
}

// ── parallel_scan ────────────────────────────────────────────────────────────

/// Scan mode (inclusive or exclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanMode {
    /// `output[i] = op(input[0..=i])` — identity-free.
    Inclusive,
    /// `output[i] = op(input[0..i])` — `output[0]` equals `identity`.
    Exclusive,
}

/// Parallel prefix scan (generalised prefix sum) using Blelloch's algorithm.
///
/// The `op` must be *associative* (but need not be commutative).
///
/// # Arguments
/// * `data`      – input slice
/// * `identity`  – identity element for `op` (needed for `Exclusive` scans)
/// * `op`        – associative binary operator
/// * `mode`      – [`ScanMode::Inclusive`] or [`ScanMode::Exclusive`]
/// * `n_threads` – degree of parallelism (0 = auto)
pub fn parallel_scan<T, F>(
    data: &[T],
    identity: T,
    op: F,
    mode: ScanMode,
    n_threads: usize,
) -> CoreResult<Vec<T>>
where
    T: Clone + Send + 'static,
    F: Fn(T, T) -> T + Send + Sync + 'static,
{
    let n = data.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let n_threads = resolve_threads(n_threads).min(n);
    let op = Arc::new(op);

    // Step 1: compute local prefix sums per chunk.
    let ranges = chunk_ranges(n, n_threads);
    let n_chunks = ranges.len();
    let data_ptr = data.as_ptr() as usize;

    let chunk_sums: Arc<Mutex<Vec<(usize, T, Vec<T>)>>> =
        Arc::new(Mutex::new(Vec::with_capacity(n_chunks)));
    let mut handles = Vec::with_capacity(n_chunks);

    for (chunk_id, range) in ranges.into_iter().enumerate() {
        let op2 = Arc::clone(&op);
        let cs = Arc::clone(&chunk_sums);
        let id2 = identity.clone();

        let handle = thread::Builder::new()
            .spawn(move || {
                let data: &[T] = unsafe { std::slice::from_raw_parts(data_ptr as *const T, n) };
                let chunk = &data[range.clone()];
                let mut local_prefix = Vec::with_capacity(range.len());
                let mut acc = id2;
                for x in chunk {
                    acc = op2(acc, x.clone());
                    local_prefix.push(acc.clone());
                }
                // The last element is the chunk's total.
                let chunk_total = local_prefix.last().cloned().unwrap_or(acc);
                if let Ok(mut g) = cs.lock() {
                    g.push((chunk_id, chunk_total, local_prefix));
                }
            })
            .map_err(spawn_err)?;
        handles.push(handle);
    }

    for h in handles {
        h.join().map_err(|_| join_err("parallel_scan local"))?;
    }

    let mut chunk_data = Arc::try_unwrap(chunk_sums)
        .map_err(|_| CoreError::SchedulerError(ErrorContext::new("parallel_scan: Arc held")))?
        .into_inner()
        .map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("parallel_scan: mutex poisoned: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
    chunk_data.sort_by_key(|(id, _, _)| *id);

    // Step 2: compute global chunk offsets sequentially.
    let mut offsets = Vec::with_capacity(n_chunks);
    let mut running = identity.clone();
    for (_, chunk_total, _) in &chunk_data {
        offsets.push(running.clone());
        running = op(running, chunk_total.clone());
    }

    // Step 3: apply offsets to local prefixes (always compute inclusive first).
    let mut result = vec![identity.clone(); n];
    let mut start = 0;
    for (chunk_idx, (_, _, local_prefix)) in chunk_data.into_iter().enumerate() {
        let offset = offsets[chunk_idx].clone();
        let len = local_prefix.len();
        for (j, lv) in local_prefix.into_iter().enumerate() {
            result[start + j] = op(offset.clone(), lv);
        }
        start += len;
    }

    // For exclusive mode, shift right by 1 and prepend identity.
    if mode == ScanMode::Exclusive {
        let mut out = vec![identity; n];
        out[1..n].clone_from_slice(&result[..(n - 1)]);
        return Ok(out);
    }

    Ok(result)
}

/// Specialised parallel prefix sum for `f64` slices.
///
/// Returns a vector of length `n` where `result[i] = sum(input[0..=i])`.
pub fn parallel_prefix_sum(data: &[f64], n_threads: usize) -> CoreResult<Vec<f64>> {
    parallel_scan(data, 0.0f64, |a, b| a + b, ScanMode::Inclusive, n_threads)
}

// ── parallel_merge_sort ───────────────────────────────────────────────────────

/// Parallel merge sort.
///
/// Splits the slice into chunks, sorts each chunk in a thread, then merges
/// sequentially.  For small slices or `n_threads == 1` this degrades to
/// `sort_unstable`.
pub fn parallel_merge_sort<T>(data: &mut Vec<T>, n_threads: usize) -> CoreResult<()>
where
    T: Ord + Send + Clone + 'static,
{
    let n = data.len();
    if n <= 1 {
        return Ok(());
    }

    let n_threads = resolve_threads(n_threads).min(n);
    if n_threads <= 1 {
        data.sort_unstable();
        return Ok(());
    }

    let ranges = chunk_ranges(n, n_threads);
    // Split data into per-chunk owned vecs.
    let mut chunks: Vec<Vec<T>> = {
        let mut remaining = data.clone();
        let mut out = Vec::with_capacity(ranges.len());
        let mut offset = 0;
        for range in &ranges {
            let chunk: Vec<T> = remaining[offset..range.end - offset + offset].to_vec();
            // Simpler: build chunks directly from data.
            let _ = remaining; // avoid unused var warning
            let chunk = data[range.clone()].to_vec();
            out.push(chunk);
            offset = range.end;
        }
        out
    };

    // Sort each chunk in a thread.
    let sorted_chunks: Arc<Mutex<Vec<(usize, Vec<T>)>>> = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();

    for (id, mut chunk) in chunks.drain(..).enumerate() {
        let sc = Arc::clone(&sorted_chunks);
        let handle = thread::Builder::new()
            .spawn(move || {
                chunk.sort_unstable();
                if let Ok(mut g) = sc.lock() {
                    g.push((id, chunk));
                }
            })
            .map_err(spawn_err)?;
        handles.push(handle);
    }

    for h in handles {
        h.join().map_err(|_| join_err("parallel_merge_sort"))?;
    }

    let mut sorted_chunks = Arc::try_unwrap(sorted_chunks)
        .map_err(|_| CoreError::SchedulerError(ErrorContext::new("parallel_merge_sort: Arc held")))?
        .into_inner()
        .map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("parallel_merge_sort: mutex poisoned: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
    sorted_chunks.sort_by_key(|(id, _)| *id);

    // Sequential k-way merge.
    let sorted: Vec<Vec<T>> = sorted_chunks.into_iter().map(|(_, v)| v).collect();
    let merged = k_way_merge(sorted);
    *data = merged;
    Ok(())
}

/// K-way merge of sorted vectors into a single sorted vector.
fn k_way_merge<T: Ord>(mut sorted: Vec<Vec<T>>) -> Vec<T> {
    while sorted.len() > 1 {
        let mut next = Vec::with_capacity(sorted.len() / 2 + 1);
        let mut i = 0;
        while i + 1 < sorted.len() {
            let merged = merge_two(
                std::mem::take(&mut sorted[i]),
                std::mem::take(&mut sorted[i + 1]),
            );
            next.push(merged);
            i += 2;
        }
        if i < sorted.len() {
            next.push(std::mem::take(&mut sorted[i]));
        }
        sorted = next;
    }
    sorted.into_iter().next().unwrap_or_default()
}

/// Merge two sorted vectors.
fn merge_two<T: Ord>(a: Vec<T>, b: Vec<T>) -> Vec<T> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let mut ai = a.into_iter();
    let mut bi = b.into_iter();
    let mut ahead = ai.next();
    let mut bhead = bi.next();
    loop {
        match (ahead, bhead) {
            (Some(av), Some(bv)) => {
                if av <= bv {
                    result.push(av);
                    ahead = ai.next();
                    bhead = Some(bv);
                } else {
                    result.push(bv);
                    bhead = bi.next();
                    ahead = Some(av);
                }
            }
            (Some(av), None) => {
                result.push(av);
                result.extend(ai);
                break;
            }
            (None, Some(bv)) => {
                result.push(bv);
                result.extend(bi);
                break;
            }
            (None, None) => break,
        }
    }
    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_map_basic() {
        let data: Vec<i32> = (1..=10).collect();
        let result = parallel_map(&data, |&x| x * x, 0).expect("parallel_map");
        assert_eq!(result, vec![1, 4, 9, 16, 25, 36, 49, 64, 81, 100]);
    }

    #[test]
    fn parallel_map_empty() {
        let data: Vec<i32> = Vec::new();
        let result = parallel_map(&data, |&x| x * 2, 0).expect("map empty");
        assert!(result.is_empty());
    }

    #[test]
    fn parallel_map_single_thread() {
        let data: Vec<u64> = (0..100).collect();
        let result = parallel_map(&data, |&x| x + 1, 1).expect("single thread");
        assert_eq!(result.len(), 100);
        assert_eq!(result[99], 100);
    }

    #[test]
    fn parallel_reduce_sum() {
        let data: Vec<i64> = (1..=100).collect();
        let sum =
            parallel_reduce(&data, 0i64, |acc, &x| acc + x, |a, b| a + b, 4).expect("reduce sum");
        assert_eq!(sum, 5050);
    }

    #[test]
    fn parallel_reduce_empty() {
        let data: Vec<i64> = Vec::new();
        let sum = parallel_reduce(&data, 42i64, |acc, &x| acc + x, |a, b| a + b, 2)
            .expect("reduce empty");
        assert_eq!(sum, 42);
    }

    #[test]
    fn parallel_filter_basic() {
        let data: Vec<i32> = (1..=20).collect();
        let evens = parallel_filter(data, |&x| x % 2 == 0, 4).expect("filter");
        assert_eq!(evens, vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);
    }

    #[test]
    fn parallel_filter_empty() {
        let data: Vec<i32> = Vec::new();
        let result = parallel_filter(data, |_| true, 2).expect("filter empty");
        assert!(result.is_empty());
    }

    #[test]
    fn parallel_scan_inclusive_sum() {
        let data: Vec<i64> = (1..=8).collect();
        let prefix =
            parallel_scan(&data, 0i64, |a, b| a + b, ScanMode::Inclusive, 4).expect("scan inc");
        assert_eq!(prefix, vec![1, 3, 6, 10, 15, 21, 28, 36]);
    }

    #[test]
    fn parallel_scan_exclusive_sum() {
        let data: Vec<i64> = (1..=5).collect();
        let prefix =
            parallel_scan(&data, 0i64, |a, b| a + b, ScanMode::Exclusive, 2).expect("scan exc");
        assert_eq!(prefix, vec![0, 1, 3, 6, 10]);
    }

    #[test]
    fn parallel_prefix_sum_basic() {
        let data: Vec<f64> = (1..=5).map(|x| x as f64).collect();
        let prefix = parallel_prefix_sum(&data, 2).expect("prefix sum");
        let expected = [1.0, 3.0, 6.0, 10.0, 15.0];
        for (a, b) in prefix.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
    }

    #[test]
    fn parallel_merge_sort_basic() {
        let mut data: Vec<i32> = vec![9, 3, 7, 1, 5, 2, 8, 4, 6, 0];
        parallel_merge_sort(&mut data, 4).expect("merge sort");
        assert_eq!(data, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn parallel_merge_sort_single_element() {
        let mut data = vec![42i32];
        parallel_merge_sort(&mut data, 4).expect("sort single");
        assert_eq!(data, vec![42]);
    }

    #[test]
    fn parallel_merge_sort_already_sorted() {
        let mut data: Vec<i32> = (0..50).collect();
        parallel_merge_sort(&mut data, 4).expect("sort sorted");
        assert_eq!(data, (0..50).collect::<Vec<_>>());
    }

    #[test]
    fn parallel_partition_basic() {
        let data: Vec<i32> = (1..=10).collect();
        let (evens, odds) = parallel_partition(data, |&x| x % 2 == 0, 4).expect("partition");
        assert_eq!(evens, vec![2, 4, 6, 8, 10]);
        assert_eq!(odds, vec![1, 3, 5, 7, 9]);
    }

    #[test]
    fn parallel_for_each_basic() {
        use std::sync::atomic::{AtomicI64, Ordering};
        let data: Vec<i64> = (1..=100).collect();
        let sum = Arc::new(AtomicI64::new(0));
        let s = Arc::clone(&sum);
        parallel_for_each(
            &data,
            move |&x| {
                s.fetch_add(x, Ordering::Relaxed);
            },
            4,
        )
        .expect("for_each");
        assert_eq!(sum.load(Ordering::Relaxed), 5050);
    }
}
