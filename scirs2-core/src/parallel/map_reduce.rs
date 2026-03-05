//! Parallel map, reduce, filter, sort and prefix-sum operations.
//!
//! All functions accept an `n_threads` argument that controls the degree of
//! parallelism.  Pass `0` to use the hardware-concurrency count.

use std::sync::{Arc, Mutex};
use std::thread;

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve `n_threads` to a concrete thread count.
fn resolve_threads(n_threads: usize) -> usize {
    if n_threads == 0 {
        thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    } else {
        n_threads
    }
}

/// Split `data` into `n_chunks` non-overlapping index ranges.
fn chunk_ranges(len: usize, n_chunks: usize) -> Vec<std::ops::Range<usize>> {
    let n_chunks = n_chunks.max(1);
    let base = len / n_chunks;
    let remainder = len % n_chunks;
    let mut ranges = Vec::with_capacity(n_chunks);
    let mut start = 0;
    for i in 0..n_chunks {
        let extra = if i < remainder { 1 } else { 0 };
        let end = start + base + extra;
        if start < len {
            ranges.push(start..end.min(len));
        }
        start = end;
    }
    ranges
}

// ─────────────────────────────────────────────────────────────────────────────
// parallel_map
// ─────────────────────────────────────────────────────────────────────────────

/// Apply `f` to every element of `data` in parallel, returning a `Vec<R>` in
/// the same order as the input.
///
/// # Errors
/// Returns `CoreError::SchedulerError` if a worker thread panics.
///
/// # Example
/// ```rust
/// use scirs2_core::parallel::map_reduce::parallel_map;
/// let v = vec![1u64, 2, 3, 4, 5];
/// let sq = parallel_map(&v, |x| x * x, 4).expect("should succeed");
/// assert_eq!(sq, vec![1, 4, 9, 16, 25]);
/// ```
pub fn parallel_map<T, R, F>(data: &[T], f: F, n_threads: usize) -> CoreResult<Vec<R>>
where
    T: Sync + 'static,
    R: Send + 'static,
    F: Fn(&T) -> R + Send + Sync + 'static,
{
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let n_threads = resolve_threads(n_threads).min(data.len());
    let ranges = chunk_ranges(data.len(), n_threads);

    // SAFETY: we slice `data` into non-overlapping ranges and each slice is
    // read-only; f only receives shared references.
    let data_ptr = data.as_ptr() as usize;
    let data_len = data.len();

    let f = Arc::new(f);
    let results: Arc<Mutex<Vec<(usize, Vec<R>)>>> = Arc::new(Mutex::new(Vec::new()));

    let handles: Vec<_> = ranges
        .into_iter()
        .enumerate()
        .map(|(chunk_idx, range)| {
            let f = Arc::clone(&f);
            let results = Arc::clone(&results);
            thread::Builder::new()
                .name(format!("par-map-{chunk_idx}"))
                .spawn(move || {
                    // Reconstruct the slice from the raw pointer.
                    // SAFETY: pointer is valid for the duration of the thread;
                    // data outlives all threads (joined before returning).
                    let slice = unsafe {
                        std::slice::from_raw_parts(data_ptr as *const T, data_len)
                    };
                    let chunk = &slice[range];
                    let mapped: Vec<R> = chunk.iter().map(|x| f(x)).collect();
                    if let Ok(mut guard) = results.lock() {
                        guard.push((chunk_idx, mapped));
                    }
                })
        })
        .collect::<Result<_, _>>()
        .map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("thread spawn failed: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

    for h in handles {
        h.join().map_err(|_| {
            CoreError::SchedulerError(
                ErrorContext::new("worker thread panicked".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
    }

    // Reconstruct in original order.
    let mut chunks = Arc::try_unwrap(results)
        .map_err(|_| {
            CoreError::SchedulerError(
                ErrorContext::new("results arc still owned".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?
        .into_inner()
        .map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("results mutex poisoned: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

    chunks.sort_unstable_by_key(|(idx, _)| *idx);
    let mut out = Vec::with_capacity(data.len());
    for (_, chunk) in chunks {
        out.extend(chunk);
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// parallel_reduce
// ─────────────────────────────────────────────────────────────────────────────

/// Map each element and then tree-reduce the results.
///
/// `identity` must satisfy `reduce_fn(identity.clone(), x) == x` for all `x`.
///
/// # Errors
/// Returns `CoreError::SchedulerError` if a worker thread panics.
///
/// # Example
/// ```rust
/// use scirs2_core::parallel::map_reduce::parallel_reduce;
/// let v: Vec<f64> = (1..=5).map(|x| x as f64).collect();
/// let sum = parallel_reduce(&v, |x| *x, |a, b| a + b, 0.0_f64, 4).expect("should succeed");
/// assert!((sum - 15.0).abs() < 1e-10);
/// ```
pub fn parallel_reduce<T, R, F, G>(
    data: &[T],
    map_fn: F,
    reduce_fn: G,
    identity: R,
    n_threads: usize,
) -> CoreResult<R>
where
    T: Sync + 'static,
    R: Send + Sync + Clone + 'static,
    F: Fn(&T) -> R + Send + Sync + 'static,
    G: Fn(R, R) -> R + Send + Sync + 'static,
{
    if data.is_empty() {
        return Ok(identity);
    }

    let n_threads = resolve_threads(n_threads).min(data.len());
    let ranges = chunk_ranges(data.len(), n_threads);

    let data_ptr = data.as_ptr() as usize;
    let data_len = data.len();

    let map_fn = Arc::new(map_fn);
    let reduce_fn = Arc::new(reduce_fn);
    let identity_arc = Arc::new(identity);

    let partial_results: Arc<Mutex<Vec<(usize, R)>>> = Arc::new(Mutex::new(Vec::new()));

    let handles: Vec<_> = ranges
        .into_iter()
        .enumerate()
        .map(|(chunk_idx, range)| {
            let map_fn = Arc::clone(&map_fn);
            let reduce_fn = Arc::clone(&reduce_fn);
            let identity_arc = Arc::clone(&identity_arc);
            let partial_results = Arc::clone(&partial_results);
            thread::Builder::new()
                .name(format!("par-reduce-{chunk_idx}"))
                .spawn(move || {
                    let slice = unsafe {
                        std::slice::from_raw_parts(data_ptr as *const T, data_len)
                    };
                    let chunk = &slice[range];
                    let partial = chunk.iter().fold((*identity_arc).clone(), |acc, x| {
                        reduce_fn(acc, map_fn(x))
                    });
                    if let Ok(mut guard) = partial_results.lock() {
                        guard.push((chunk_idx, partial));
                    }
                })
        })
        .collect::<Result<_, _>>()
        .map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("thread spawn failed: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

    for h in handles {
        h.join().map_err(|_| {
            CoreError::SchedulerError(
                ErrorContext::new("worker panicked".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
    }

    let identity_val = Arc::try_unwrap(identity_arc)
        .unwrap_or_else(|a| (*a).clone());

    let mut partials = Arc::try_unwrap(partial_results)
        .map_err(|_| {
            CoreError::SchedulerError(
                ErrorContext::new("partial_results arc still owned".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?
        .into_inner()
        .map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("partial_results mutex poisoned: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

    partials.sort_unstable_by_key(|(idx, _)| *idx);
    let result = partials
        .into_iter()
        .fold(identity_val, |acc, (_, v)| (*reduce_fn)(acc, v));

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// parallel_filter
// ─────────────────────────────────────────────────────────────────────────────

/// Retain only elements for which `pred` returns `true`, in parallel.
///
/// The output preserves the original relative order.
///
/// # Errors
/// Returns `CoreError::SchedulerError` if a worker thread panics.
///
/// # Example
/// ```rust
/// use scirs2_core::parallel::map_reduce::parallel_filter;
/// let v = vec![1i32, 2, 3, 4, 5, 6];
/// let evens = parallel_filter(v, |x| x % 2 == 0, 2).expect("should succeed");
/// assert_eq!(evens, vec![2, 4, 6]);
/// ```
pub fn parallel_filter<T, F>(data: Vec<T>, pred: F, n_threads: usize) -> CoreResult<Vec<T>>
where
    T: Send + Sync + 'static,
    F: Fn(&T) -> bool + Send + Sync + 'static,
{
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let n_threads = resolve_threads(n_threads).min(data.len());
    let data = Arc::new(data);
    let pred = Arc::new(pred);

    let ranges = chunk_ranges(data.len(), n_threads);
    let partial_results: Arc<Mutex<Vec<(usize, Vec<usize>)>>> =
        Arc::new(Mutex::new(Vec::new()));

    let handles: Vec<_> = ranges
        .into_iter()
        .enumerate()
        .map(|(chunk_idx, range)| {
            let data = Arc::clone(&data);
            let pred = Arc::clone(&pred);
            let partial_results = Arc::clone(&partial_results);
            thread::Builder::new()
                .name(format!("par-filter-{chunk_idx}"))
                .spawn(move || {
                    let indices: Vec<usize> = range
                        .filter(|&i| pred(&data[i]))
                        .collect();
                    if let Ok(mut guard) = partial_results.lock() {
                        guard.push((chunk_idx, indices));
                    }
                })
        })
        .collect::<Result<_, _>>()
        .map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("thread spawn failed: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

    for h in handles {
        h.join().map_err(|_| {
            CoreError::SchedulerError(
                ErrorContext::new("filter worker panicked".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
    }

    let mut chunk_results = Arc::try_unwrap(partial_results)
        .map_err(|_| {
            CoreError::SchedulerError(
                ErrorContext::new("partial_results arc still owned".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?
        .into_inner()
        .map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("partial_results mutex poisoned: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

    chunk_results.sort_unstable_by_key(|(idx, _)| *idx);

    // Collect all passing indices in original order.
    let mut passing: Vec<usize> = chunk_results
        .into_iter()
        .flat_map(|(_, v)| v)
        .collect();
    passing.sort_unstable();

    // Unwrap the Arc – we are the only remaining owner.
    let data = Arc::try_unwrap(data).map_err(|_| {
        CoreError::SchedulerError(
            ErrorContext::new("data arc still owned after joining threads".to_string())
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    })?;

    // Build output by index (requires moving out of the Vec by index).
    // We convert to a Vec of Options to allow indexed moves.
    let mut data_opts: Vec<Option<T>> = data.into_iter().map(Some).collect();
    let mut out = Vec::with_capacity(passing.len());
    for i in passing {
        if let Some(val) = data_opts.get_mut(i).and_then(|opt| opt.take()) {
            out.push(val);
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// parallel_sort – parallel merge sort
// ─────────────────────────────────────────────────────────────────────────────

/// Sort `data` in-place using a parallel merge sort.
///
/// Falls back to `sort_unstable` for small inputs or when `n_threads == 1`.
///
/// # Errors
/// Returns `CoreError::SchedulerError` if a worker thread panics.
///
/// # Example
/// ```rust
/// use scirs2_core::parallel::map_reduce::parallel_sort;
/// let mut v = vec![5i32, 1, 4, 2, 3];
/// parallel_sort(&mut v, 2).expect("should succeed");
/// assert_eq!(v, vec![1, 2, 3, 4, 5]);
/// ```
pub fn parallel_sort<T: Ord + Send + 'static>(data: &mut Vec<T>, n_threads: usize) -> CoreResult<()> {
    let n = data.len();
    let n_threads = resolve_threads(n_threads);

    if n <= 1 || n_threads <= 1 {
        data.sort_unstable();
        return Ok(());
    }

    // Sequential threshold: below 2048 elements, sequential is faster.
    if n < 2048 {
        data.sort_unstable();
        return Ok(());
    }

    parallel_merge_sort(data, n_threads)
}

fn parallel_merge_sort<T: Ord + Send + 'static>(data: &mut Vec<T>, n_threads: usize) -> CoreResult<()> {
    let n = data.len();
    if n <= 1 {
        return Ok(());
    }

    // Split into two halves, sort each in a thread, then merge.
    let mid = n / 2;

    // Move data out of the Vec to avoid aliasing; we'll put it back.
    let mut left: Vec<T> = data.drain(..mid).collect();
    let mut right: Vec<T> = data.drain(..).collect();

    if n_threads >= 2 {
        // SAFETY: we move the Vecs into threads and join before returning.
        // `T: Send` is required.
        let left_handle = {
            // We need to move `left` into the thread.  Use a channel to get
            // it back after sorting.
            let (tx, rx) = std::sync::mpsc::channel();
            thread::Builder::new()
                .name("par-sort-left".to_string())
                .spawn(move || {
                    let mut v = left;
                    parallel_merge_sort_seq(&mut v);
                    let _ = tx.send(v);
                })
                .map_err(|e| {
                    CoreError::SchedulerError(
                        ErrorContext::new(format!("sort thread spawn failed: {e}"))
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;
            rx
        };

        parallel_merge_sort_seq(&mut right);

        left = left_handle.recv().map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("sort thread recv failed: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
    } else {
        parallel_merge_sort_seq(&mut left);
        parallel_merge_sort_seq(&mut right);
    }

    // Merge sorted halves back into `data`.
    *data = merge_sorted(left, right);
    Ok(())
}

fn parallel_merge_sort_seq<T: Ord>(data: &mut Vec<T>) {
    data.sort_unstable();
}

fn merge_sorted<T: Ord>(mut left: Vec<T>, mut right: Vec<T>) -> Vec<T> {
    let mut result = Vec::with_capacity(left.len() + right.len());
    let mut li = 0;
    let mut ri = 0;
    while li < left.len() && ri < right.len() {
        if left[li] <= right[ri] {
            li += 1;
        } else {
            ri += 1;
        }
    }
    // Drain in merge order using indices.
    let mut l_iter = left.drain(..);
    let mut r_iter = right.drain(..);
    let mut l_buf: Option<T> = l_iter.next();
    let mut r_buf: Option<T> = r_iter.next();
    loop {
        match (l_buf.take(), r_buf.take()) {
            (Some(l), Some(r)) => {
                if l <= r {
                    result.push(l);
                    r_buf = Some(r);
                    l_buf = l_iter.next();
                } else {
                    result.push(r);
                    l_buf = Some(l);
                    r_buf = r_iter.next();
                }
            }
            (Some(l), None) => {
                result.push(l);
                l_buf = l_iter.next();
            }
            (None, Some(r)) => {
                result.push(r);
                r_buf = r_iter.next();
            }
            (None, None) => break,
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// parallel_prefix_sum – parallel scan (exclusive)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute an **inclusive** parallel prefix sum of `data`.
///
/// `result[i] = data[0] + data[1] + … + data[i]`.
///
/// Uses a two-pass algorithm: chunk sums in parallel, then sequential prefix
/// on the partial sums, then add the prefix to each chunk in parallel.
///
/// # Errors
/// Returns `CoreError::SchedulerError` if a worker thread panics.
///
/// # Example
/// ```rust
/// use scirs2_core::parallel::map_reduce::parallel_prefix_sum;
/// let v = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let ps = parallel_prefix_sum(&v, 4).expect("should succeed");
/// assert!((ps[4] - 15.0).abs() < 1e-10);
/// ```
pub fn parallel_prefix_sum(data: &[f64], n_threads: usize) -> CoreResult<Vec<f64>> {
    let n = data.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let n_threads = resolve_threads(n_threads).min(n);
    let ranges = chunk_ranges(n, n_threads);
    let n_chunks = ranges.len();

    // Pass 1: compute the sum of each chunk in parallel.
    let data_ptr = data.as_ptr() as usize;
    let chunk_sums: Arc<Mutex<Vec<(usize, f64)>>> = Arc::new(Mutex::new(Vec::new()));

    let handles: Vec<_> = ranges
        .iter()
        .cloned()
        .enumerate()
        .map(|(chunk_idx, range)| {
            let chunk_sums = Arc::clone(&chunk_sums);
            thread::Builder::new()
                .name(format!("par-scan-{chunk_idx}"))
                .spawn(move || {
                    let slice =
                        unsafe { std::slice::from_raw_parts(data_ptr as *const f64, n) };
                    // Kahan summation for each chunk.
                    let mut sum = 0.0_f64;
                    let mut c = 0.0_f64;
                    for &v in &slice[range] {
                        let y = v - c;
                        let t = sum + y;
                        c = (t - sum) - y;
                        sum = t;
                    }
                    if let Ok(mut guard) = chunk_sums.lock() {
                        guard.push((chunk_idx, sum));
                    }
                })
        })
        .collect::<Result<_, _>>()
        .map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("thread spawn failed: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

    for h in handles {
        h.join().map_err(|_| {
            CoreError::SchedulerError(
                ErrorContext::new("scan worker panicked".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
    }

    let mut chunk_sums_vec = Arc::try_unwrap(chunk_sums)
        .map_err(|_| {
            CoreError::SchedulerError(
                ErrorContext::new("chunk_sums arc still owned".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?
        .into_inner()
        .map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("chunk_sums mutex poisoned: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

    chunk_sums_vec.sort_unstable_by_key(|(idx, _)| *idx);
    let chunk_sums_sorted: Vec<f64> = chunk_sums_vec.into_iter().map(|(_, s)| s).collect();

    // Pass 2: sequential exclusive prefix sum over chunk sums.
    let mut chunk_prefix = vec![0.0_f64; n_chunks];
    for i in 1..n_chunks {
        chunk_prefix[i] = chunk_prefix[i - 1] + chunk_sums_sorted[i - 1];
    }

    // Pass 3: compute per-element inclusive prefix within each chunk,
    // adding the chunk prefix offset, in parallel.
    let mut output = vec![0.0_f64; n];
    let output_ptr = output.as_mut_ptr() as usize;

    let handles: Vec<_> = ranges
        .into_iter()
        .enumerate()
        .map(|(chunk_idx, range)| {
            let offset = chunk_prefix[chunk_idx];
            thread::Builder::new()
                .name(format!("par-scan2-{chunk_idx}"))
                .spawn(move || {
                    let slice_in =
                        unsafe { std::slice::from_raw_parts(data_ptr as *const f64, n) };
                    // SAFETY: each chunk writes to a distinct, non-overlapping
                    // sub-range of `output`.
                    let slice_out =
                        unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f64, n) };
                    let mut running = offset;
                    for i in range {
                        running += slice_in[i];
                        slice_out[i] = running;
                    }
                })
        })
        .collect::<Result<_, _>>()
        .map_err(|e| {
            CoreError::SchedulerError(
                ErrorContext::new(format!("thread spawn failed: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

    for h in handles {
        h.join().map_err(|_| {
            CoreError::SchedulerError(
                ErrorContext::new("scan pass-3 worker panicked".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_map_squares() {
        let data: Vec<u64> = (1..=10).collect();
        let result = parallel_map(&data, |x| x * x, 4).expect("should succeed");
        let expected: Vec<u64> = data.iter().map(|x| x * x).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parallel_map_empty() {
        let data: Vec<i32> = vec![];
        let result = parallel_map(&data, |x| x + 1, 2).expect("should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_parallel_reduce_sum() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let sum = parallel_reduce(&data, |x| *x, |a, b| a + b, 0.0, 4).expect("should succeed");
        assert!((sum - 5050.0).abs() < 1e-6);
    }

    #[test]
    fn test_parallel_filter_even() {
        let data: Vec<i32> = (1..=10).collect();
        let evens = parallel_filter(data, |x| x % 2 == 0, 3).expect("should succeed");
        assert_eq!(evens, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_parallel_sort() {
        let mut data: Vec<i32> = vec![9, 3, 7, 1, 5, 2, 8, 4, 6, 0];
        parallel_sort(&mut data, 2).expect("should succeed");
        assert_eq!(data, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_parallel_prefix_sum() {
        let data: Vec<f64> = (1..=5).map(|x| x as f64).collect();
        let ps = parallel_prefix_sum(&data, 4).expect("should succeed");
        assert!((ps[0] - 1.0).abs() < 1e-10);
        assert!((ps[1] - 3.0).abs() < 1e-10);
        assert!((ps[2] - 6.0).abs() < 1e-10);
        assert!((ps[3] - 10.0).abs() < 1e-10);
        assert!((ps[4] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_prefix_sum_empty() {
        let ps = parallel_prefix_sum(&[], 2).expect("should succeed");
        assert!(ps.is_empty());
    }
}
