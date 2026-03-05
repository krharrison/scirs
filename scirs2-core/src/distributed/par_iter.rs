//! Standalone parallel iterator combinators
//!
//! This module provides simple, ergonomic parallel `map`, `filter`, and `fold`
//! operations that use OS threads (via [`std::thread::scope`]) without
//! requiring an external runtime or the `parallel` feature gate.
//!
//! These combinators complement the rayon-based operations in
//! `crate::parallel_ops` by working without rayon and providing explicit
//! control over chunk sizes and thread counts.
//!
//! ## Design
//!
//! - Data is split into contiguous chunks.
//! - Each chunk is processed on a scoped thread.
//! - Results are concatenated in input order.
//! - No `unsafe` code is used.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::distributed::par_iter::{par_map, par_filter, par_fold};
//!
//! let data: Vec<i32> = (0..1000).collect();
//!
//! // Parallel map
//! let squares = par_map(&data, |&x| x * x, None);
//! assert_eq!(squares[0], 0);
//! assert_eq!(squares[10], 100);
//!
//! // Parallel filter
//! let evens = par_filter(&data, |&&x| x % 2 == 0, None);
//! assert_eq!(evens.len(), 500);
//!
//! // Parallel fold
//! let sum = par_fold(&data, 0i64, |acc, &x| acc + x as i64, |a, b| a + b, None);
//! assert_eq!(sum, 499_500);
//! ```

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Default number of worker threads (logical CPU count, minimum 1).
fn default_num_workers() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Compute an appropriate chunk size given `data_len` and optional `num_workers`.
fn chunk_size(data_len: usize, num_workers: Option<usize>) -> usize {
    let workers = num_workers.unwrap_or_else(default_num_workers).max(1);
    // Target ~4 chunks per worker for good load balancing
    let target_chunks = workers.saturating_mul(4).max(1);
    (data_len / target_chunks).max(1)
}

// ─────────────────────────────────────────────────────────────────────────────
// par_map
// ─────────────────────────────────────────────────────────────────────────────

/// Apply `f` to each element of `data` in parallel, returning the mapped
/// results in input order.
///
/// `num_workers` controls the number of threads.  Pass `None` to use the
/// system default (number of logical CPUs).
///
/// For small inputs (< 64 elements), the map is executed sequentially on
/// the calling thread.
pub fn par_map<T, U, F>(data: &[T], f: F, num_workers: Option<usize>) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    if n < 64 {
        return data.iter().map(&f).collect();
    }

    let cs = chunk_size(n, num_workers);

    std::thread::scope(|s| {
        let handles: Vec<_> = data
            .chunks(cs)
            .map(|chunk| s.spawn(|| chunk.iter().map(&f).collect::<Vec<U>>()))
            .collect();

        let mut result = Vec::with_capacity(n);
        for handle in handles {
            match handle.join() {
                Ok(partial) => result.extend(partial),
                Err(_) => {
                    // Thread panicked — fill with nothing (best effort)
                    // In practice this shouldn't happen since f is safe.
                }
            }
        }
        result
    })
}

/// `par_map` that returns `CoreResult`.
pub fn try_par_map<T, U, F>(data: &[T], f: F, num_workers: Option<usize>) -> CoreResult<Vec<U>>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> CoreResult<U> + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n < 64 {
        return data.iter().map(&f).collect();
    }

    let cs = chunk_size(n, num_workers);

    std::thread::scope(|s| {
        let handles: Vec<_> = data
            .chunks(cs)
            .map(|chunk| s.spawn(|| chunk.iter().map(&f).collect::<CoreResult<Vec<U>>>()))
            .collect();

        let mut result = Vec::with_capacity(n);
        for handle in handles {
            match handle.join() {
                Ok(Ok(partial)) => result.extend(partial),
                Ok(Err(e)) => return Err(e),
                Err(_) => {
                    return Err(CoreError::ThreadError(
                        ErrorContext::new("par_map worker thread panicked".to_string())
                            .with_location(ErrorLocation::new(file!(), line!())),
                    ))
                }
            }
        }
        Ok(result)
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// par_filter
// ─────────────────────────────────────────────────────────────────────────────

/// Filter elements of `data` in parallel, returning those for which
/// `predicate` returns `true`, in input order.
///
/// `num_workers` controls the number of threads.  Pass `None` to use the
/// system default.
pub fn par_filter<T, F>(data: &[T], predicate: F, num_workers: Option<usize>) -> Vec<T>
where
    T: Clone + Sync + Send,
    F: Fn(&&T) -> bool + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    if n < 64 {
        return data.iter().filter(&predicate).cloned().collect();
    }

    let cs = chunk_size(n, num_workers);

    std::thread::scope(|s| {
        let handles: Vec<_> = data
            .chunks(cs)
            .map(|chunk| {
                s.spawn(|| {
                    chunk
                        .iter()
                        .filter(|item| predicate(item))
                        .cloned()
                        .collect::<Vec<T>>()
                })
            })
            .collect();

        let mut result = Vec::new();
        for handle in handles {
            if let Ok(partial) = handle.join() {
                result.extend(partial);
            }
        }
        result
    })
}

/// `par_filter` variant that also transforms each passing element.
pub fn par_filter_map<T, U, F>(data: &[T], f: F, num_workers: Option<usize>) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> Option<U> + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    if n < 64 {
        return data.iter().filter_map(&f).collect();
    }

    let cs = chunk_size(n, num_workers);

    std::thread::scope(|s| {
        let handles: Vec<_> = data
            .chunks(cs)
            .map(|chunk| s.spawn(|| chunk.iter().filter_map(&f).collect::<Vec<U>>()))
            .collect();

        let mut result = Vec::new();
        for handle in handles {
            if let Ok(partial) = handle.join() {
                result.extend(partial);
            }
        }
        result
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// par_fold
// ─────────────────────────────────────────────────────────────────────────────

/// Parallel fold (reduce) operation.
///
/// - Each chunk is folded with `fold_op(accumulator, &element)`.
/// - Chunk results are then combined with `combine_op(left, right)`.
///
/// The `fold_op` and `combine_op` should form a monoid with `init` as
/// the identity element.
///
/// `num_workers` controls the number of threads.  Pass `None` for default.
pub fn par_fold<T, A, FoldOp, CombineOp>(
    data: &[T],
    init: A,
    fold_op: FoldOp,
    combine_op: CombineOp,
    num_workers: Option<usize>,
) -> A
where
    T: Sync,
    A: Clone + Send + Sync,
    FoldOp: Fn(A, &T) -> A + Send + Sync,
    CombineOp: Fn(A, A) -> A + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return init;
    }
    if n < 64 {
        return data.iter().fold(init, &fold_op);
    }

    let cs = chunk_size(n, num_workers);
    let init_ref = &init;

    std::thread::scope(|s| {
        let handles: Vec<_> = data
            .chunks(cs)
            .map(|chunk| {
                s.spawn(|| {
                    let local_init = init_ref.clone();
                    chunk.iter().fold(local_init, &fold_op)
                })
            })
            .collect();

        let mut acc = init.clone();
        for handle in handles {
            if let Ok(partial) = handle.join() {
                acc = combine_op(acc, partial);
            }
        }
        acc
    })
}

/// `par_fold` that returns `CoreResult`.
pub fn try_par_fold<T, A, FoldOp, CombineOp>(
    data: &[T],
    init: A,
    fold_op: FoldOp,
    combine_op: CombineOp,
    num_workers: Option<usize>,
) -> CoreResult<A>
where
    T: Sync,
    A: Clone + Send + Sync,
    FoldOp: Fn(A, &T) -> CoreResult<A> + Send + Sync,
    CombineOp: Fn(A, A) -> A + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return Ok(init);
    }
    if n < 64 {
        let mut acc = init;
        for item in data {
            acc = fold_op(acc, item)?;
        }
        return Ok(acc);
    }

    let cs = chunk_size(n, num_workers);
    let init_ref = &init;
    let fold_ref = &fold_op;

    std::thread::scope(|s| {
        let handles: Vec<_> = data
            .chunks(cs)
            .map(|chunk| {
                s.spawn(move || {
                    let mut local_acc = init_ref.clone();
                    for item in chunk {
                        local_acc = fold_ref(local_acc, item)?;
                    }
                    Ok(local_acc)
                })
            })
            .collect();

        let mut acc = init.clone();
        for handle in handles {
            match handle.join() {
                Ok(Ok(partial)) => acc = combine_op(acc, partial),
                Ok(Err(e)) => return Err(e),
                Err(_) => {
                    return Err(CoreError::ThreadError(
                        ErrorContext::new("par_fold worker thread panicked".to_string())
                            .with_location(ErrorLocation::new(file!(), line!())),
                    ))
                }
            }
        }
        Ok(acc)
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// par_for_each
// ─────────────────────────────────────────────────────────────────────────────

/// Apply `f` to each element of `data` in parallel (no return values).
///
/// `num_workers` controls the number of threads.  Pass `None` for default.
pub fn par_for_each<T, F>(data: &[T], f: F, num_workers: Option<usize>)
where
    T: Sync,
    F: Fn(&T) + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return;
    }
    if n < 64 {
        data.iter().for_each(&f);
        return;
    }

    let cs = chunk_size(n, num_workers);

    std::thread::scope(|s| {
        let handles: Vec<_> = data
            .chunks(cs)
            .map(|chunk| s.spawn(|| chunk.iter().for_each(&f)))
            .collect();

        for handle in handles {
            let _ = handle.join();
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// par_any / par_all
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `true` if `predicate` returns `true` for any element (parallel).
pub fn par_any<T, F>(data: &[T], predicate: F, num_workers: Option<usize>) -> bool
where
    T: Sync,
    F: Fn(&T) -> bool + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return false;
    }
    if n < 64 {
        return data.iter().any(&predicate);
    }

    let cs = chunk_size(n, num_workers);

    std::thread::scope(|s| {
        let handles: Vec<_> = data
            .chunks(cs)
            .map(|chunk| s.spawn(|| chunk.iter().any(&predicate)))
            .collect();

        for handle in handles {
            if let Ok(true) = handle.join() {
                return true;
            }
        }
        false
    })
}

/// Returns `true` if `predicate` returns `true` for all elements (parallel).
pub fn par_all<T, F>(data: &[T], predicate: F, num_workers: Option<usize>) -> bool
where
    T: Sync,
    F: Fn(&T) -> bool + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return true;
    }
    if n < 64 {
        return data.iter().all(&predicate);
    }

    let cs = chunk_size(n, num_workers);

    std::thread::scope(|s| {
        let handles: Vec<_> = data
            .chunks(cs)
            .map(|chunk| s.spawn(|| chunk.iter().all(&predicate)))
            .collect();

        for handle in handles {
            if let Ok(false) = handle.join() {
                return false;
            }
        }
        true
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// par_sort
// ─────────────────────────────────────────────────────────────────────────────

/// Parallel merge sort that sorts `data` in place.
///
/// Uses a divide-and-conquer approach: splits the data, sorts each half
/// in parallel, then merges.
///
/// `num_workers` is the depth of parallelism.  `None` defaults to
/// `log2(num_cpus)`.
pub fn par_sort<T>(data: &mut [T], num_workers: Option<usize>)
where
    T: Ord + Send + Clone,
{
    let depth = num_workers
        .unwrap_or_else(|| {
            let cpus = default_num_workers();
            (cpus as f64).log2().ceil() as usize
        })
        .max(1);

    par_sort_recursive(data, depth);
}

/// Parallel sort with a custom comparator.
pub fn par_sort_by<T, F>(data: &mut [T], compare: F, num_workers: Option<usize>)
where
    T: Send + Clone,
    F: Fn(&T, &T) -> std::cmp::Ordering + Send + Sync + Clone,
{
    let depth = num_workers
        .unwrap_or_else(|| {
            let cpus = default_num_workers();
            (cpus as f64).log2().ceil() as usize
        })
        .max(1);

    par_sort_by_recursive(data, &compare, depth);
}

fn par_sort_recursive<T: Ord + Send + Clone>(data: &mut [T], depth: usize) {
    if data.len() <= 32 || depth == 0 {
        data.sort();
        return;
    }

    let mid = data.len() / 2;
    let (left, right) = data.split_at_mut(mid);

    std::thread::scope(|s| {
        let handle = s.spawn(|| par_sort_recursive(left, depth - 1));
        par_sort_recursive(right, depth - 1);
        let _ = handle.join();
    });

    // Merge in place using a temporary buffer
    let merged = merge_sorted(&data[..mid], &data[mid..], |a, b| a.cmp(b));
    data[..merged.len()].clone_from_slice(&merged);
}

fn par_sort_by_recursive<T, F>(data: &mut [T], compare: &F, depth: usize)
where
    T: Send + Clone,
    F: Fn(&T, &T) -> std::cmp::Ordering + Send + Sync,
{
    if data.len() <= 32 || depth == 0 {
        data.sort_by(compare);
        return;
    }

    let mid = data.len() / 2;
    let (left, right) = data.split_at_mut(mid);

    std::thread::scope(|s| {
        let handle = s.spawn(|| par_sort_by_recursive(left, compare, depth - 1));
        par_sort_by_recursive(right, compare, depth - 1);
        let _ = handle.join();
    });

    let merged = merge_sorted(&data[..mid], &data[mid..], compare);
    data[..merged.len()].clone_from_slice(&merged);
}

fn merge_sorted<T, F>(left: &[T], right: &[T], compare: F) -> Vec<T>
where
    T: Clone,
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    let mut result = Vec::with_capacity(left.len() + right.len());
    let (mut i, mut j) = (0, 0);

    while i < left.len() && j < right.len() {
        if compare(&left[i], &right[j]) != std::cmp::Ordering::Greater {
            result.push(left[i].clone());
            i += 1;
        } else {
            result.push(right[j].clone());
            j += 1;
        }
    }

    while i < left.len() {
        result.push(left[i].clone());
        i += 1;
    }
    while j < right.len() {
        result.push(right[j].clone());
        j += 1;
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── par_map ──────────────────────────────────────────────────────────────

    #[test]
    fn test_par_map_empty() {
        let data: Vec<i32> = Vec::new();
        let result = par_map(&data, |&x| x * 2, None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_par_map_small() {
        let data = vec![1, 2, 3, 4, 5];
        let result = par_map(&data, |&x| x * x, None);
        assert_eq!(result, vec![1, 4, 9, 16, 25]);
    }

    #[test]
    fn test_par_map_large() {
        let data: Vec<i32> = (0..10_000).collect();
        let result = par_map(&data, |&x| x * 2, Some(4));
        let expected: Vec<i32> = (0..10_000).map(|x| x * 2).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_par_map_preserves_order() {
        let data: Vec<usize> = (0..500).collect();
        let result = par_map(&data, |&x| x, Some(8));
        assert_eq!(result, data);
    }

    #[test]
    fn test_par_map_string_transform() {
        let data: Vec<i32> = (0..200).collect();
        let result = par_map(&data, |x| format!("v{x}"), Some(4));
        for (i, s) in result.iter().enumerate() {
            assert_eq!(s, &format!("v{i}"));
        }
    }

    #[test]
    fn test_try_par_map_success() {
        let data: Vec<i32> = (0..200).collect();
        let result = try_par_map(&data, |&x| Ok(x as f64 * 0.5), Some(2)).expect("should succeed");
        assert_eq!(result.len(), 200);
        assert!((result[10] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_try_par_map_error() {
        let data: Vec<i32> = (0..200).collect();
        let result = try_par_map(
            &data,
            |&x| {
                if x == 150 {
                    Err(CoreError::ValueError(
                        ErrorContext::new("bad value".to_string())
                            .with_location(ErrorLocation::new(file!(), line!())),
                    ))
                } else {
                    Ok(x)
                }
            },
            Some(2),
        );
        assert!(result.is_err());
    }

    // ── par_filter ───────────────────────────────────────────────────────────

    #[test]
    fn test_par_filter_empty() {
        let data: Vec<i32> = Vec::new();
        let result = par_filter(&data, |_| true, None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_par_filter_evens() {
        let data: Vec<i32> = (0..1000).collect();
        let result = par_filter(&data, |&&x| x % 2 == 0, Some(4));
        assert_eq!(result.len(), 500);
        for v in &result {
            assert_eq!(v % 2, 0);
        }
    }

    #[test]
    fn test_par_filter_none_pass() {
        let data: Vec<i32> = (0..100).collect();
        let result = par_filter(&data, |&&x| x > 1000, None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_par_filter_all_pass() {
        let data: Vec<i32> = (0..100).collect();
        let result = par_filter(&data, |_| true, None);
        assert_eq!(result, data);
    }

    #[test]
    fn test_par_filter_preserves_order() {
        let data: Vec<i32> = (0..500).collect();
        let result = par_filter(&data, |&&x| x % 3 == 0, Some(4));
        let expected: Vec<i32> = (0..500).filter(|x| x % 3 == 0).collect();
        assert_eq!(result, expected);
    }

    // ── par_filter_map ───────────────────────────────────────────────────────

    #[test]
    fn test_par_filter_map() {
        let data: Vec<i32> = (0..200).collect();
        let result = par_filter_map(
            &data,
            |&x| if x % 2 == 0 { Some(x / 2) } else { None },
            Some(4),
        );
        let expected: Vec<i32> = (0..200).filter(|x| x % 2 == 0).map(|x| x / 2).collect();
        assert_eq!(result, expected);
    }

    // ── par_fold ─────────────────────────────────────────────────────────────

    #[test]
    fn test_par_fold_empty() {
        let data: Vec<i32> = Vec::new();
        let result = par_fold(&data, 0i64, |acc, _| acc, |a, b| a + b, None);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_par_fold_sum() {
        let data: Vec<i32> = (1..=100).collect();
        let sum = par_fold(&data, 0i64, |acc, &x| acc + x as i64, |a, b| a + b, Some(4));
        assert_eq!(sum, 5050);
    }

    #[test]
    fn test_par_fold_large() {
        let n = 100_000i64;
        let data: Vec<i64> = (1..=n).collect();
        let sum = par_fold(&data, 0i64, |acc, &x| acc + x, |a, b| a + b, Some(8));
        assert_eq!(sum, n * (n + 1) / 2);
    }

    #[test]
    fn test_par_fold_product() {
        let data = vec![1, 2, 3, 4, 5];
        let product = par_fold(&data, 1i64, |acc, &x| acc * x as i64, |a, b| a * b, None);
        assert_eq!(product, 120);
    }

    #[test]
    fn test_par_fold_max() {
        let data: Vec<i32> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        let max_val = par_fold(&data, i32::MIN, |acc, &x| acc.max(x), |a, b| a.max(b), None);
        assert_eq!(max_val, 9);
    }

    #[test]
    fn test_try_par_fold_success() {
        let data: Vec<i32> = (1..=10).collect();
        let sum = try_par_fold(
            &data,
            0i64,
            |acc, &x| Ok(acc + x as i64),
            |a, b| a + b,
            Some(2),
        )
        .expect("should succeed");
        assert_eq!(sum, 55);
    }

    // ── par_for_each ─────────────────────────────────────────────────────────

    #[test]
    fn test_par_for_each() {
        use std::sync::atomic::{AtomicI64, Ordering};
        let data: Vec<i32> = (1..=100).collect();
        let sum = AtomicI64::new(0);
        par_for_each(
            &data,
            |&x| {
                sum.fetch_add(x as i64, Ordering::Relaxed);
            },
            Some(4),
        );
        assert_eq!(sum.load(Ordering::Relaxed), 5050);
    }

    // ── par_any / par_all ────────────────────────────────────────────────────

    #[test]
    fn test_par_any_found() {
        let data: Vec<i32> = (0..1000).collect();
        assert!(par_any(&data, |&x| x == 500, Some(4)));
    }

    #[test]
    fn test_par_any_not_found() {
        let data: Vec<i32> = (0..1000).collect();
        assert!(!par_any(&data, |&x| x == 2000, Some(4)));
    }

    #[test]
    fn test_par_all_true() {
        let data: Vec<i32> = (0..1000).collect();
        assert!(par_all(&data, |&x| x < 1000, Some(4)));
    }

    #[test]
    fn test_par_all_false() {
        let data: Vec<i32> = (0..1000).collect();
        assert!(!par_all(&data, |&x| x < 500, Some(4)));
    }

    // ── par_sort ─────────────────────────────────────────────────────────────

    #[test]
    fn test_par_sort_empty() {
        let mut data: Vec<i32> = Vec::new();
        par_sort(&mut data, None);
        assert!(data.is_empty());
    }

    #[test]
    fn test_par_sort_sorted() {
        let mut data: Vec<i32> = (0..1000).rev().collect();
        par_sort(&mut data, Some(4));
        let expected: Vec<i32> = (0..1000).collect();
        assert_eq!(data, expected);
    }

    #[test]
    fn test_par_sort_already_sorted() {
        let mut data: Vec<i32> = (0..100).collect();
        par_sort(&mut data, None);
        let expected: Vec<i32> = (0..100).collect();
        assert_eq!(data, expected);
    }

    #[test]
    fn test_par_sort_by_descending() {
        let mut data: Vec<i32> = (0..500).collect();
        par_sort_by(&mut data, |a, b| b.cmp(a), Some(4));
        let expected: Vec<i32> = (0..500).rev().collect();
        assert_eq!(data, expected);
    }

    #[test]
    fn test_par_sort_large_random() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let n = 10_000;
        let mut data: Vec<i64> = (0..n)
            .map(|i| {
                let mut hasher = DefaultHasher::new();
                i.hash(&mut hasher);
                hasher.finish() as i64
            })
            .collect();
        let mut expected = data.clone();
        expected.sort();

        par_sort(&mut data, Some(4));
        assert_eq!(data, expected);
    }
}
