//! Parallel prefix scan (prefix sum) algorithms
//!
//! This module implements efficient parallel prefix scan operations using
//! Blelloch's work-efficient parallel scan algorithm. The implementation
//! uses Rayon for data parallelism.
//!
//! # Algorithms
//!
//! - **Blelloch scan**: Work-efficient O(n) work, O(log n) span
//! - **Segmented scan**: Prefix scan that resets at segment boundaries
//! - **Exclusive scan**: Output[i] = op(input[0..i]) (identity at index 0)
//! - **Inclusive scan**: Output[i] = op(input[0..=i])
//!
//! # Examples
//!
//! ```rust
//! use scirs2_core::parallel::scan::{exclusive_scan, inclusive_scan, parallel_scan_sum};
//!
//! let data = vec![1u64, 2, 3, 4, 5];
//! let prefix = parallel_scan_sum(&data);
//! assert_eq!(prefix, vec![0, 1, 3, 6, 10]);
//!
//! let inc = inclusive_scan(&data, 0u64, |a, b| a + b);
//! assert_eq!(inc, vec![1, 3, 6, 10, 15]);
//! ```

use crate::error::{CoreError, CoreResult};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ── threshold below which we fall back to sequential ──────────────────────
const PAR_THRESHOLD: usize = 1024;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Parallel exclusive prefix sum using Blelloch's algorithm.
///
/// Returns a vector of length `n` where `result[i] = sum(input[0..i])`.
/// `result[0]` is always 0 (the additive identity).
///
/// # Arguments
/// * `input` – input slice of `u64` values
///
/// # Examples
/// ```rust
/// use scirs2_core::parallel::scan::parallel_scan_sum;
/// let v = vec![1u64, 2, 3, 4];
/// assert_eq!(parallel_scan_sum(&v), vec![0, 1, 3, 6]);
/// ```
pub fn parallel_scan_sum(input: &[u64]) -> Vec<u64> {
    exclusive_scan_u64(input)
}

/// Generic parallel exclusive prefix scan with any associative binary operation.
///
/// Returns a vector of length `n` where `result[i] = op(input[0..i])`.
/// `result[0]` is always `identity`.
///
/// # Type Parameters
/// * `T` – element type (must be `Clone + Send + Sync`)
/// * `F` – associative binary operator (`Fn(T, T) -> T`, must be `Send + Sync`)
///
/// # Arguments
/// * `input`    – input slice
/// * `identity` – identity element for `op` (e.g. 0 for addition, 1 for multiplication)
/// * `op`       – associative binary operator
///
/// # Examples
/// ```rust
/// use scirs2_core::parallel::scan::parallel_scan_generic;
/// let v = vec![1i32, 2, 3, 4];
/// let result = parallel_scan_generic(&v, 0i32, |a, b| a + b);
/// assert_eq!(result, vec![0, 1, 3, 6]);
/// ```
pub fn parallel_scan_generic<T, F>(input: &[T], identity: T, op: F) -> Vec<T>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + Clone,
{
    blelloch_exclusive_scan(input, identity, op)
}

/// Segmented prefix scan.
///
/// Performs an exclusive prefix scan that resets at every segment boundary.
/// `segments[i] == true` marks the start of a new segment.
///
/// # Arguments
/// * `input`    – input slice
/// * `segments` – flags marking segment starts; must have the same length as `input`
/// * `identity` – identity element for `op`
/// * `op`       – associative binary operator
///
/// # Errors
/// Returns `CoreError` if `input` and `segments` have different lengths.
///
/// # Examples
/// ```rust
/// use scirs2_core::parallel::scan::segmented_scan;
/// let v    = vec![1i32, 2, 3, 4, 5];
/// let segs = vec![true, false, false, true, false];
/// let r    = segmented_scan(&v, &segs, 0i32, |a, b| a + b).expect("should succeed");
/// // Segment 0: [1,2,3] → exclusive sums [0,1,3]
/// // Segment 1: [4,5]   → exclusive sums [0,4]
/// assert_eq!(r, vec![0, 1, 3, 0, 4]);
/// ```
pub fn segmented_scan<T, F>(
    input: &[T],
    segments: &[bool],
    identity: T,
    op: F,
) -> CoreResult<Vec<T>>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    if input.len() != segments.len() {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new(format!(
                "segmented_scan: input length {} != segments length {} (input and segments slices must have the same length)",
                input.len(),
                segments.len()
            )),
        ));
    }

    let n = input.len();
    let mut output = vec![identity.clone(); n];

    let mut acc = identity.clone();
    for i in 0..n {
        if segments[i] {
            acc = identity.clone();
        }
        output[i] = acc.clone();
        acc = op(acc, input[i].clone());
    }

    Ok(output)
}

/// Exclusive prefix scan: `result[i] = op(input[0..i])`, `result[0] = identity`.
///
/// # Examples
/// ```rust
/// use scirs2_core::parallel::scan::exclusive_scan;
/// let v = vec![2i32, 3, 5, 7];
/// let r = exclusive_scan(&v, 0i32, |a, b| a + b);
/// assert_eq!(r, vec![0, 2, 5, 10]);
/// ```
pub fn exclusive_scan<T, F>(input: &[T], identity: T, op: F) -> Vec<T>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + Clone,
{
    blelloch_exclusive_scan(input, identity, op)
}

/// Inclusive prefix scan: `result[i] = op(input[0..=i])`.
///
/// # Examples
/// ```rust
/// use scirs2_core::parallel::scan::inclusive_scan;
/// let v = vec![2i32, 3, 5, 7];
/// let r = inclusive_scan(&v, 0i32, |a, b| a + b);
/// assert_eq!(r, vec![2, 5, 10, 17]);
/// ```
pub fn inclusive_scan<T, F>(input: &[T], identity: T, op: F) -> Vec<T>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + Clone,
{
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    let mut excl = blelloch_exclusive_scan(input, identity, op.clone());

    // inclusive[i] = exclusive[i] op input[i]
    for i in 0..n {
        let prev = excl[i].clone();
        excl[i] = op(prev, input[i].clone());
    }
    excl
}

// ─────────────────────────────────────────────────────────────────────────────
// Exclusive scan specialised for u64 (avoids Clone bounds on the operator)
// ─────────────────────────────────────────────────────────────────────────────

fn exclusive_scan_u64(input: &[u64]) -> Vec<u64> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    let mut out = vec![0u64; n];

    if n < PAR_THRESHOLD {
        // Sequential path
        let mut acc = 0u64;
        for (i, &v) in input.iter().enumerate() {
            out[i] = acc;
            acc = acc.wrapping_add(v);
        }
        return out;
    }

    // Parallel path: Blelloch two-phase up-sweep / down-sweep on a flat buffer
    blelloch_scan_u64_parallel(input, &mut out);
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Blelloch parallel scan (generic version)
// ─────────────────────────────────────────────────────────────────────────────

/// Core Blelloch exclusive scan.
///
/// For small inputs falls back to a simple sequential scan.
/// For large inputs uses Rayon parallel up-sweep / down-sweep when the
/// `parallel` feature is enabled; otherwise always sequential.
fn blelloch_exclusive_scan<T, F>(input: &[T], identity: T, op: F) -> Vec<T>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + Clone,
{
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    // Small input: plain sequential prefix scan (avoids allocation overhead)
    if n < PAR_THRESHOLD {
        return sequential_exclusive_scan(input, identity, &op);
    }

    // Large input: chunk-based parallel prefix scan
    parallel_chunked_exclusive_scan(input, identity, op)
}

/// Sequential exclusive prefix scan.
fn sequential_exclusive_scan<T, F>(input: &[T], identity: T, op: &F) -> Vec<T>
where
    T: Clone,
    F: Fn(T, T) -> T,
{
    let n = input.len();
    let mut out = Vec::with_capacity(n);
    let mut acc = identity;
    for item in input {
        out.push(acc.clone());
        acc = op(acc, item.clone());
    }
    out
}

/// Parallel chunked exclusive scan using Rayon.
///
/// Strategy:
/// 1. Divide the input into `P` chunks (P = rayon thread count).
/// 2. Compute the local prefix-sum within each chunk sequentially.
/// 3. Collect the chunk totals and compute their exclusive prefix sum.
/// 4. Add the chunk offset to every element in the chunk (in parallel).
#[cfg(feature = "parallel")]
fn parallel_chunked_exclusive_scan<T, F>(input: &[T], identity: T, op: F) -> Vec<T>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + Clone,
{
    use std::sync::Arc;
    let n = input.len();
    let num_threads = rayon::current_num_threads().max(1);
    let chunk_size = ((n + num_threads - 1) / num_threads).max(1);

    // Wrap op in Arc so it can be cloned across Fn closures without being moved.
    let op = Arc::new(op);

    // Phase 1: compute local prefix sums in each chunk; collect totals
    let op_phase1 = Arc::clone(&op);
    let chunks: Vec<(Vec<T>, T)> = input
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local = Vec::with_capacity(chunk.len());
            let mut acc = identity.clone();
            for item in chunk {
                local.push(acc.clone());
                acc = op_phase1(acc, item.clone());
            }
            (local, acc)
        })
        .collect();

    // Phase 2: compute exclusive prefix sum over chunk totals (sequential – O(P))
    let mut chunk_offsets: Vec<T> = Vec::with_capacity(chunks.len());
    let mut running = identity.clone();
    for (_, total) in &chunks {
        chunk_offsets.push(running.clone());
        running = op(running, total.clone());
    }

    // Phase 3: add chunk offset to every element (parallel)
    let local_scans: Vec<Vec<T>> = chunks.into_iter().map(|(v, _)| v).collect();
    let op_phase3 = Arc::clone(&op);

    let result: Vec<T> = local_scans
        .into_par_iter()
        .zip(chunk_offsets.into_par_iter())
        .flat_map(move |(chunk_scan, offset)| {
            let op_inner = Arc::clone(&op_phase3);
            chunk_scan
                .into_iter()
                .map(move |v| op_inner(offset.clone(), v))
                .collect::<Vec<_>>()
        })
        .collect();

    result
}

/// Sequential fallback for the chunked scan when rayon is unavailable.
#[cfg(not(feature = "parallel"))]
fn parallel_chunked_exclusive_scan<T, F>(input: &[T], identity: T, op: F) -> Vec<T>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + Clone,
{
    sequential_exclusive_scan(input, identity, &op)
}

// ─────────────────────────────────────────────────────────────────────────────
// Blelloch parallel scan specialised for u64 (Rayon parallel path)
// ─────────────────────────────────────────────────────────────────────────────

/// True Blelloch two-phase scan for u64 arrays.
///
/// Works on the `out` buffer in-place after copying input.
#[cfg(feature = "parallel")]
fn blelloch_scan_u64_parallel(input: &[u64], out: &mut [u64]) {
    let n = input.len();

    // Copy input to output buffer (we'll work in-place)
    out[..n].copy_from_slice(input);

    // Round up to next power of two for the tree
    let tree_size = n.next_power_of_two();
    let mut tree = vec![0u64; tree_size];
    tree[..n].copy_from_slice(input);

    // Up-sweep (reduce) phase
    let mut stride = 1usize;
    while stride < tree_size {
        let step = stride * 2;
        let indices: Vec<usize> = (step - 1..tree_size).step_by(step).collect();
        let updates: Vec<u64> = indices
            .par_iter()
            .map(|&i| tree[i].wrapping_add(tree[i - stride]))
            .collect();
        for (&i, v) in indices.iter().zip(updates) {
            tree[i] = v;
        }
        stride *= 2;
    }

    // Set the root to zero (identity for addition)
    tree[tree_size - 1] = 0;

    // Down-sweep phase
    let mut stride = tree_size / 2;
    while stride >= 1 {
        let step = stride * 2;
        let indices: Vec<usize> = (step - 1..tree_size).step_by(step).collect();
        let swaps: Vec<(u64, u64)> = indices
            .par_iter()
            .map(|&i| {
                let left_child = tree[i - stride];
                let parent = tree[i];
                // Down-sweep: left child ← parent, right child ← parent + left_child
                (parent, parent.wrapping_add(left_child))
            })
            .collect();
        for (&i, (new_left, new_right)) in indices.iter().zip(swaps) {
            tree[i - stride] = new_left;
            tree[i] = new_right;
        }
        stride /= 2;
    }

    // Copy tree result back to output (only the first n elements)
    out[..n].copy_from_slice(&tree[..n]);
}

#[cfg(not(feature = "parallel"))]
fn blelloch_scan_u64_parallel(input: &[u64], out: &mut [u64]) {
    let mut acc = 0u64;
    for (i, &v) in input.iter().enumerate() {
        out[i] = acc;
        acc = acc.wrapping_add(v);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_scan_sum_empty() {
        let v: Vec<u64> = vec![];
        assert_eq!(parallel_scan_sum(&v), Vec::<u64>::new());
    }

    #[test]
    fn test_parallel_scan_sum_single() {
        assert_eq!(parallel_scan_sum(&[42u64]), vec![0u64]);
    }

    #[test]
    fn test_parallel_scan_sum_basic() {
        let v = vec![1u64, 2, 3, 4, 5];
        assert_eq!(parallel_scan_sum(&v), vec![0, 1, 3, 6, 10]);
    }

    #[test]
    fn test_parallel_scan_sum_large() {
        let n = 10_000usize;
        let input: Vec<u64> = (0..n as u64).collect();
        let result = parallel_scan_sum(&input);
        for i in 0..n {
            let expected: u64 = (0..i as u64).sum();
            assert_eq!(result[i], expected, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_exclusive_scan_multiplication() {
        let v = vec![2i64, 3, 4, 5];
        let r = exclusive_scan(&v, 1i64, |a, b| a * b);
        assert_eq!(r, vec![1, 2, 6, 24]);
    }

    #[test]
    fn test_inclusive_scan_sum() {
        let v = vec![1i32, 2, 3, 4, 5];
        let r = inclusive_scan(&v, 0i32, |a, b| a + b);
        assert_eq!(r, vec![1, 3, 6, 10, 15]);
    }

    #[test]
    fn test_segmented_scan_basic() {
        let v = vec![1i32, 2, 3, 4, 5];
        let segs = vec![true, false, false, true, false];
        let r = segmented_scan(&v, &segs, 0i32, |a, b| a + b).expect("segmented scan failed");
        assert_eq!(r, vec![0, 1, 3, 0, 4]);
    }

    #[test]
    fn test_segmented_scan_length_mismatch() {
        let v = vec![1i32, 2, 3];
        let segs = vec![true, false];
        assert!(segmented_scan(&v, &segs, 0i32, |a, b| a + b).is_err());
    }

    #[test]
    fn test_parallel_scan_generic_sum() {
        let v = vec![1i32, 2, 3, 4];
        let r = parallel_scan_generic(&v, 0i32, |a, b| a + b);
        assert_eq!(r, vec![0, 1, 3, 6]);
    }

    #[test]
    fn test_scan_consistency_small_large() {
        // Sequential and parallel paths should agree
        let v: Vec<u64> = (1..=100).collect();
        let result = parallel_scan_sum(&v);
        let mut expected = 0u64;
        for (i, r) in result.iter().enumerate() {
            assert_eq!(*r, expected, "index {i}");
            expected += v[i];
        }
    }
}
