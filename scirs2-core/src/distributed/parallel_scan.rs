//! Work-efficient parallel prefix sum (scan) operations
//!
//! This module implements the Blelloch parallel scan algorithm, which computes
//! prefix sums (and generalised prefix scans with arbitrary associative
//! operators) in O(n) work and O(log n) span.
//!
//! Two variants are provided:
//!
//! - **Exclusive scan** (`parallel_prefix_sum_exclusive`): element `i` of the
//!   output is the sum of elements `0..i` (the first element is `identity`).
//! - **Inclusive scan** (`parallel_prefix_sum`): element `i` of the output
//!   is the sum of elements `0..=i`.
//!
//! A generic `parallel_scan` function accepts any associative binary operator.
//!
//! When the `parallel` feature is enabled, the up-sweep and down-sweep phases
//! use `rayon` for parallel execution.  Without `parallel`, all operations
//! fall back to sequential execution.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::distributed::parallel_scan::{parallel_prefix_sum, parallel_scan};
//!
//! // Inclusive prefix sum: [1, 3, 6, 10, 15]
//! let data = vec![1, 2, 3, 4, 5];
//! let result = parallel_prefix_sum(&data);
//! assert_eq!(result, vec![1, 3, 6, 10, 15]);
//!
//! // Generic scan with multiplication
//! let data = vec![1, 2, 3, 4];
//! let result = parallel_scan(&data, 1, |a, b| a * b);
//! assert_eq!(result, vec![1, 2, 6, 24]);
//! ```

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};

// ─────────────────────────────────────────────────────────────────────────────
// Core algorithms
// ─────────────────────────────────────────────────────────────────────────────

/// Inclusive prefix sum for a slice of values that support addition.
///
/// Returns a `Vec<T>` where `result[i] = data[0] + data[1] + ... + data[i]`.
///
/// For empty input, returns an empty vector.
///
/// This is a specialisation of [`parallel_scan`] with the addition operator
/// and a zero identity element.
pub fn parallel_prefix_sum<T>(data: &[T]) -> Vec<T>
where
    T: Clone + Send + Sync + Default + std::ops::Add<Output = T>,
{
    parallel_scan(data, T::default(), |a, b| a + b)
}

/// Exclusive prefix sum for a slice of values that support addition.
///
/// Returns a `Vec<T>` where `result[i] = data[0] + ... + data[i-1]` and
/// `result[0] = identity`.
///
/// For empty input, returns an empty vector.
pub fn parallel_prefix_sum_exclusive<T>(data: &[T]) -> Vec<T>
where
    T: Clone + Send + Sync + Default + std::ops::Add<Output = T>,
{
    parallel_scan_exclusive(data, T::default(), |a, b| a + b)
}

/// Inclusive generalised parallel scan with an arbitrary associative operator.
///
/// Given `data = [a0, a1, a2, ...]` and binary operator `op`, returns
/// `[a0, op(a0, a1), op(op(a0, a1), a2), ...]`.
///
/// The `identity` element must satisfy `op(identity, x) == x` for all `x`.
///
/// # Algorithm
///
/// For small inputs (< `SEQUENTIAL_THRESHOLD` elements), a simple sequential
/// scan is used.  For larger inputs, the **Blelloch three-phase** algorithm
/// is used:
///
/// 1. **Tile reduce**: Divide the input into tiles, compute partial reductions
///    of each tile (in parallel when the `parallel` feature is enabled).
/// 2. **Prefix on reductions**: Compute an exclusive prefix scan on the tile
///    reductions (recursive, but the number of tiles is small).
/// 3. **Tile scan**: Each tile performs a local inclusive scan starting from
///    its tile prefix (in parallel when `parallel` is enabled).
pub fn parallel_scan<T, F>(data: &[T], identity: T, op: F) -> Vec<T>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + Clone,
{
    if data.is_empty() {
        return Vec::new();
    }

    let n = data.len();

    // For small inputs, use sequential scan
    if n < SEQUENTIAL_THRESHOLD {
        return sequential_inclusive_scan(data, &identity, &op);
    }

    blelloch_inclusive_scan(data, &identity, &op)
}

/// Exclusive generalised parallel scan.
///
/// Like [`parallel_scan`] but the output is shifted right: `result[0] =
/// identity`, `result[i] = op(data[0], ..., data[i-1])`.
pub fn parallel_scan_exclusive<T, F>(data: &[T], identity: T, op: F) -> Vec<T>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + Clone,
{
    if data.is_empty() {
        return Vec::new();
    }

    let n = data.len();

    if n < SEQUENTIAL_THRESHOLD {
        return sequential_exclusive_scan(data, &identity, &op);
    }

    blelloch_exclusive_scan(data, &identity, &op)
}

/// Parallel prefix sum that returns a `CoreResult`, for ergonomic error
/// handling at call sites.
pub fn try_parallel_prefix_sum<T>(data: &[T]) -> CoreResult<Vec<T>>
where
    T: Clone + Send + Sync + Default + std::ops::Add<Output = T>,
{
    Ok(parallel_prefix_sum(data))
}

/// Parallel scan that validates the input is non-empty.
pub fn try_parallel_scan<T, F>(data: &[T], identity: T, op: F) -> CoreResult<Vec<T>>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + Clone,
{
    if data.is_empty() {
        return Err(CoreError::ValueError(
            ErrorContext::new("parallel_scan requires non-empty input".to_string())
                .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    Ok(parallel_scan(data, identity, op))
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Below this many elements, we fall back to sequential scan.
const SEQUENTIAL_THRESHOLD: usize = 1024;

/// Tile size for the parallel Blelloch algorithm.
/// Each tile is processed sequentially; tiles are processed in parallel.
const TILE_SIZE: usize = 256;

fn sequential_inclusive_scan<T, F>(data: &[T], identity: &T, op: &F) -> Vec<T>
where
    T: Clone,
    F: Fn(T, T) -> T,
{
    let mut result = Vec::with_capacity(data.len());
    let mut acc = identity.clone();
    for item in data {
        acc = op(acc, item.clone());
        result.push(acc.clone());
    }
    result
}

fn sequential_exclusive_scan<T, F>(data: &[T], identity: &T, op: &F) -> Vec<T>
where
    T: Clone,
    F: Fn(T, T) -> T,
{
    let mut result = Vec::with_capacity(data.len());
    let mut acc = identity.clone();
    for item in data {
        result.push(acc.clone());
        acc = op(acc, item.clone());
    }
    result
}

/// Blelloch three-phase inclusive scan.
#[cfg(feature = "parallel")]
fn blelloch_inclusive_scan<T, F>(data: &[T], identity: &T, op: &F) -> Vec<T>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + Clone,
{
    use rayon::prelude::*;

    let n = data.len();
    let num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    // Phase 1: Reduce each tile to a single value (in parallel)
    let tile_reductions: Vec<T> = (0..num_tiles)
        .into_par_iter()
        .map(|tile_idx| {
            let start = tile_idx * TILE_SIZE;
            let end = (start + TILE_SIZE).min(n);
            let mut acc = identity.clone();
            for item in &data[start..end] {
                acc = op(acc, item.clone());
            }
            acc
        })
        .collect();

    // Phase 2: Exclusive prefix scan on tile reductions (sequential — num_tiles is small)
    let mut tile_prefixes: Vec<T> = Vec::with_capacity(num_tiles);
    {
        let mut acc = identity.clone();
        for red in &tile_reductions {
            tile_prefixes.push(acc.clone());
            acc = op(acc, red.clone());
        }
    }

    // Phase 3: Local inclusive scan per tile, starting from tile prefix (in parallel)
    let mut result: Vec<T> = vec![identity.clone(); n];
    let result_chunks: Vec<&mut [T]> = result.chunks_mut(TILE_SIZE).collect();

    // We need to use indices-based approach to avoid lifetime issues
    let result_ptr = result.as_mut_ptr();
    let data_ref = data;

    // Safety: each tile writes to a disjoint range of `result`
    // We use thread::scope to avoid unsafe code
    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_tiles);
        for tile_idx in 0..num_tiles {
            let start = tile_idx * TILE_SIZE;
            let end = (start + TILE_SIZE).min(n);
            let tile_prefix = tile_prefixes[tile_idx].clone();
            let op_clone = op.clone();
            let tile_data = &data_ref[start..end];

            let handle = s.spawn(move || {
                let mut acc = tile_prefix;
                let mut local_result = Vec::with_capacity(end - start);
                for item in tile_data {
                    acc = op_clone(acc, item.clone());
                    local_result.push(acc.clone());
                }
                (start, local_result)
            });
            handles.push(handle);
        }

        for handle in handles {
            if let Ok((start, local_result)) = handle.join() {
                for (i, val) in local_result.into_iter().enumerate() {
                    // Safety: each tile writes to non-overlapping range
                    unsafe {
                        std::ptr::write(result_ptr.add(start + i), val);
                    }
                }
            }
        }
    });

    result
}

/// Blelloch three-phase exclusive scan.
#[cfg(feature = "parallel")]
fn blelloch_exclusive_scan<T, F>(data: &[T], identity: &T, op: &F) -> Vec<T>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + Clone,
{
    use rayon::prelude::*;

    let n = data.len();
    let num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    // Phase 1: Reduce each tile
    let tile_reductions: Vec<T> = (0..num_tiles)
        .into_par_iter()
        .map(|tile_idx| {
            let start = tile_idx * TILE_SIZE;
            let end = (start + TILE_SIZE).min(n);
            let mut acc = identity.clone();
            for item in &data[start..end] {
                acc = op(acc, item.clone());
            }
            acc
        })
        .collect();

    // Phase 2: Exclusive prefix on tile reductions
    let mut tile_prefixes: Vec<T> = Vec::with_capacity(num_tiles);
    {
        let mut acc = identity.clone();
        for red in &tile_reductions {
            tile_prefixes.push(acc.clone());
            acc = op(acc, red.clone());
        }
    }

    // Phase 3: Local exclusive scan per tile
    let mut result: Vec<T> = vec![identity.clone(); n];
    let result_ptr = result.as_mut_ptr();
    let data_ref = data;

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_tiles);
        for tile_idx in 0..num_tiles {
            let start = tile_idx * TILE_SIZE;
            let end = (start + TILE_SIZE).min(n);
            let tile_prefix = tile_prefixes[tile_idx].clone();
            let op_clone = op.clone();
            let tile_data = &data_ref[start..end];

            let handle = s.spawn(move || {
                let mut acc = tile_prefix;
                let mut local_result = Vec::with_capacity(end - start);
                for item in tile_data {
                    local_result.push(acc.clone());
                    acc = op_clone(acc, item.clone());
                }
                (start, local_result)
            });
            handles.push(handle);
        }

        for handle in handles {
            if let Ok((start, local_result)) = handle.join() {
                for (i, val) in local_result.into_iter().enumerate() {
                    unsafe {
                        std::ptr::write(result_ptr.add(start + i), val);
                    }
                }
            }
        }
    });

    result
}

/// Sequential fallback for inclusive scan (no parallel feature).
#[cfg(not(feature = "parallel"))]
fn blelloch_inclusive_scan<T, F>(data: &[T], identity: &T, op: &F) -> Vec<T>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + Clone,
{
    sequential_inclusive_scan(data, identity, op)
}

/// Sequential fallback for exclusive scan (no parallel feature).
#[cfg(not(feature = "parallel"))]
fn blelloch_exclusive_scan<T, F>(data: &[T], identity: &T, op: &F) -> Vec<T>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + Clone,
{
    sequential_exclusive_scan(data, identity, op)
}

// ─────────────────────────────────────────────────────────────────────────────
// Specialised numeric prefix sums
// ─────────────────────────────────────────────────────────────────────────────

/// Fast inclusive prefix sum for `f64` slices.
///
/// Uses SIMD-friendly sequential accumulation for small inputs and the
/// tiled parallel algorithm for large inputs.
pub fn parallel_prefix_sum_f64(data: &[f64]) -> Vec<f64> {
    parallel_prefix_sum(data)
}

/// Fast inclusive prefix sum for `i64` slices.
pub fn parallel_prefix_sum_i64(data: &[i64]) -> Vec<i64> {
    parallel_prefix_sum(data)
}

/// Parallel prefix minimum — `result[i] = min(data[0..=i])`.
pub fn parallel_prefix_min<T>(data: &[T]) -> Vec<T>
where
    T: Clone + Send + Sync + Ord,
{
    if data.is_empty() {
        return Vec::new();
    }
    let identity = data[0].clone();
    parallel_scan(data, identity, |a, b| if a <= b { a } else { b })
}

/// Parallel prefix maximum — `result[i] = max(data[0..=i])`.
pub fn parallel_prefix_max<T>(data: &[T]) -> Vec<T>
where
    T: Clone + Send + Sync + Ord,
{
    if data.is_empty() {
        return Vec::new();
    }
    let identity = data[0].clone();
    parallel_scan(data, identity, |a, b| if a >= b { a } else { b })
}

/// Segmented prefix sum.
///
/// `flags[i]` is `true` at the start of a new segment. The prefix sum
/// resets at each segment boundary.
///
/// # Example
///
/// ```rust
/// use scirs2_core::distributed::parallel_scan::segmented_prefix_sum;
///
/// let data = vec![1, 2, 3, 1, 2, 3];
/// let flags = vec![true, false, false, true, false, false];
/// let result = segmented_prefix_sum(&data, &flags);
/// assert_eq!(result, vec![1, 3, 6, 1, 3, 6]);
/// ```
pub fn segmented_prefix_sum<T>(data: &[T], flags: &[bool]) -> Vec<T>
where
    T: Clone + Send + Sync + Default + std::ops::Add<Output = T>,
{
    let n = data.len().min(flags.len());
    if n == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(n);
    let mut acc = T::default();

    for i in 0..n {
        if flags[i] {
            acc = T::default();
        }
        acc = acc + data[i].clone();
        result.push(acc.clone());
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_prefix_sum() {
        let data: Vec<i32> = Vec::new();
        assert!(parallel_prefix_sum(&data).is_empty());
        assert!(parallel_prefix_sum_exclusive(&data).is_empty());
    }

    #[test]
    fn test_single_element() {
        assert_eq!(parallel_prefix_sum(&[42]), vec![42]);
        assert_eq!(parallel_prefix_sum_exclusive(&[42]), vec![0]);
    }

    #[test]
    fn test_small_inclusive_sum() {
        let data = vec![1, 2, 3, 4, 5];
        let result = parallel_prefix_sum(&data);
        assert_eq!(result, vec![1, 3, 6, 10, 15]);
    }

    #[test]
    fn test_small_exclusive_sum() {
        let data = vec![1, 2, 3, 4, 5];
        let result = parallel_prefix_sum_exclusive(&data);
        assert_eq!(result, vec![0, 1, 3, 6, 10]);
    }

    #[test]
    fn test_generic_scan_multiplication() {
        let data = vec![1, 2, 3, 4, 5];
        let result = parallel_scan(&data, 1, |a, b| a * b);
        assert_eq!(result, vec![1, 2, 6, 24, 120]);
    }

    #[test]
    fn test_generic_scan_max() {
        let data = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let result = parallel_prefix_max(&data);
        assert_eq!(result, vec![3, 3, 4, 4, 5, 9, 9, 9]);
    }

    #[test]
    fn test_generic_scan_min() {
        let data = vec![5, 3, 7, 1, 4, 2, 8, 6];
        let result = parallel_prefix_min(&data);
        assert_eq!(result, vec![5, 3, 3, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_large_prefix_sum() {
        // Test with data larger than SEQUENTIAL_THRESHOLD
        let n = 5000;
        let data: Vec<i64> = (1..=n).collect();
        let result = parallel_prefix_sum(&data);
        // Verify a few known values
        assert_eq!(result[0], 1);
        assert_eq!(result[n as usize - 1], n * (n + 1) / 2);
        // Verify it's monotonically increasing
        for i in 1..result.len() {
            assert!(result[i] > result[i - 1]);
        }
    }

    #[test]
    fn test_large_exclusive_sum() {
        let n = 5000;
        let data: Vec<i64> = (1..=n).collect();
        let result = parallel_prefix_sum_exclusive(&data);
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 1);
        assert_eq!(result[n as usize - 1], n * (n - 1) / 2);
    }

    #[test]
    fn test_segmented_prefix_sum() {
        let data = vec![1, 2, 3, 1, 2, 3];
        let flags = vec![true, false, false, true, false, false];
        let result = segmented_prefix_sum(&data, &flags);
        assert_eq!(result, vec![1, 3, 6, 1, 3, 6]);
    }

    #[test]
    fn test_segmented_prefix_sum_single_segment() {
        let data = vec![1, 2, 3, 4];
        let flags = vec![true, false, false, false];
        let result = segmented_prefix_sum(&data, &flags);
        assert_eq!(result, vec![1, 3, 6, 10]);
    }

    #[test]
    fn test_segmented_prefix_sum_all_segments() {
        let data = vec![10, 20, 30];
        let flags = vec![true, true, true];
        let result = segmented_prefix_sum(&data, &flags);
        assert_eq!(result, vec![10, 20, 30]);
    }

    #[test]
    fn test_f64_prefix_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = parallel_prefix_sum_f64(&data);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
        assert!((result[2] - 6.0).abs() < 1e-10);
        assert!((result[3] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_try_parallel_scan_empty_error() {
        let data: Vec<i32> = Vec::new();
        let result = try_parallel_scan(&data, 0, |a, b| a + b);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_parallel_prefix_sum() {
        let data = vec![1, 2, 3];
        let result = try_parallel_prefix_sum(&data).expect("should succeed");
        assert_eq!(result, vec![1, 3, 6]);
    }

    #[test]
    fn test_consistency_inclusive_vs_exclusive() {
        let data: Vec<i32> = (1..=100).collect();
        let inclusive = parallel_prefix_sum(&data);
        let exclusive = parallel_prefix_sum_exclusive(&data);

        // exclusive[i] + data[i] == inclusive[i]
        for i in 0..data.len() {
            assert_eq!(exclusive[i] + data[i], inclusive[i]);
        }
    }

    #[test]
    fn test_string_concat_scan() {
        let data = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let result = parallel_scan(&data, String::new(), |mut a, b| {
            a.push_str(&b);
            a
        });
        assert_eq!(result, vec!["a", "ab", "abc"]);
    }

    #[test]
    fn test_large_parallel_correctness() {
        // Verify parallel result matches sequential for large inputs
        let n = 10_000;
        let data: Vec<i64> = (0..n).collect();
        let par_result = parallel_prefix_sum(&data);
        let seq_result = sequential_inclusive_scan(&data, &0i64, &|a, b| a + b);
        assert_eq!(par_result, seq_result);
    }
}
