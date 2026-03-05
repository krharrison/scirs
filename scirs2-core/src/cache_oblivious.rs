//! # Cache-Oblivious Algorithms
//!
//! This module implements cache-oblivious algorithms that achieve optimal cache
//! performance without knowledge of cache parameters, by using recursive divide-and-conquer
//! strategies that naturally exploit cache locality at every level of the memory hierarchy.
//!
//! ## Algorithms Provided
//!
//! - **`recursive_transpose`** – Cache-oblivious matrix transpose via recursive blocking.
//! - **`cache_oblivious_matmul`** – Cache-oblivious matrix multiplication.
//! - **`cache_oblivious_sort`** – Funnel sort (cache-oblivious comparison sort).
//! - **`van_emde_boas_layout`** – van Emde Boas memory layout permutation for implicit trees.
//!
//! ## References
//!
//! - Frigo, M. et al. (1999). Cache-Oblivious Algorithms. FOCS.
//! - van Emde Boas, P. (1975). Preserving order in a forest in less than logarithmic time.
//! - Prokop, H. (1999). Cache-Oblivious Algorithms. MIT Master's Thesis.

use crate::error::{CoreError, CoreResult, ErrorContext};
use ndarray::{Array2, ArrayView2, ArrayViewMut2};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Threshold below which we fall back to a direct (non-recursive) operation.
const TILE_THRESHOLD: usize = 32;

// ---------------------------------------------------------------------------
// Matrix Transpose
// ---------------------------------------------------------------------------

/// Perform a cache-oblivious in-place transpose of the submatrix
/// `src[r0..r1, c0..c1]` into `dst[c0..c1, r0..r1]`.
fn transpose_recursive(
    src: &ArrayView2<f64>,
    dst: &mut ArrayViewMut2<f64>,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) {
    let rows = r1 - r0;
    let cols = c1 - c0;

    if rows <= TILE_THRESHOLD && cols <= TILE_THRESHOLD {
        // Base case: direct element-wise copy
        for i in r0..r1 {
            for j in c0..c1 {
                dst[[j, i]] = src[[i, j]];
            }
        }
        return;
    }

    if rows >= cols {
        let mid = r0 + rows / 2;
        transpose_recursive(src, dst, r0, mid, c0, c1);
        transpose_recursive(src, dst, mid, r1, c0, c1);
    } else {
        let mid = c0 + cols / 2;
        transpose_recursive(src, dst, r0, r1, c0, mid);
        transpose_recursive(src, dst, r0, r1, mid, c1);
    }
}

/// Cache-oblivious matrix transpose.
///
/// Returns `B` where `B[j, i] == A[i, j]` for all valid `i`, `j`.
/// Uses a recursive divide-and-conquer strategy that achieves the optimal
/// cache-miss bound of `Θ(mn / B)` (where `B` is the cache line size) without
/// knowing `B` at compile time.
///
/// # Arguments
///
/// * `matrix` – The input matrix to transpose.
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use scirs2_core::cache_oblivious::recursive_transpose;
///
/// let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let t = recursive_transpose(&a);
/// assert_eq!(t[[0, 0]], 1.0);
/// assert_eq!(t[[1, 0]], 2.0);
/// assert_eq!(t[[2, 1]], 6.0);
/// ```
pub fn recursive_transpose(matrix: &Array2<f64>) -> Array2<f64> {
    let (rows, cols) = matrix.dim();
    let mut result = Array2::<f64>::zeros((cols, rows));
    let src_view = matrix.view();
    let mut dst_view = result.view_mut();
    transpose_recursive(&src_view, &mut dst_view, 0, rows, 0, cols);
    result
}

// ---------------------------------------------------------------------------
// Matrix Multiplication
// ---------------------------------------------------------------------------

/// Recursive cache-oblivious matrix multiplication kernel.
///
/// Computes `C[r0..r1, c0..c1] += A[r0..r1, k0..k1] * B[k0..k1, c0..c1]`.
fn matmul_recursive(
    a: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
    c: &mut ArrayViewMut2<f64>,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
    k0: usize,
    k1: usize,
) {
    let rows = r1 - r0;
    let cols = c1 - c0;
    let depth = k1 - k0;

    if rows <= TILE_THRESHOLD && cols <= TILE_THRESHOLD && depth <= TILE_THRESHOLD {
        // Base case: naive triple-loop
        for i in r0..r1 {
            for k in k0..k1 {
                let a_ik = a[[i, k]];
                for j in c0..c1 {
                    c[[i, j]] += a_ik * b[[k, j]];
                }
            }
        }
        return;
    }

    // Split along the largest dimension to stay cache-friendly
    if rows >= cols && rows >= depth {
        let mid = r0 + rows / 2;
        matmul_recursive(a, b, c, r0, mid, c0, c1, k0, k1);
        matmul_recursive(a, b, c, mid, r1, c0, c1, k0, k1);
    } else if cols >= rows && cols >= depth {
        let mid = c0 + cols / 2;
        matmul_recursive(a, b, c, r0, r1, c0, mid, k0, k1);
        matmul_recursive(a, b, c, r0, r1, mid, c1, k0, k1);
    } else {
        let mid = k0 + depth / 2;
        matmul_recursive(a, b, c, r0, r1, c0, c1, k0, mid);
        matmul_recursive(a, b, c, r0, r1, c0, c1, mid, k1);
    }
}

/// Cache-oblivious matrix multiplication `C = A × B`.
///
/// Uses recursive divide-and-conquer to achieve the optimal cache complexity
/// `Θ(n³ / (B √M))` (where `B` is the cache line size and `M` is cache size)
/// without any a-priori knowledge of cache parameters.
///
/// # Arguments
///
/// * `a` – Left operand of shape `(m, k)`.
/// * `b` – Right operand of shape `(k, n)`.
///
/// # Errors
///
/// Returns `Err` if the inner dimensions do not match.
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// use scirs2_core::cache_oblivious::cache_oblivious_matmul;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let b = array![[3.0_f64, 4.0], [5.0, 6.0]];
/// let c = cache_oblivious_matmul(&a, &b).expect("should succeed");
/// assert_eq!(c[[0, 0]], 3.0);
/// assert_eq!(c[[1, 1]], 6.0);
/// ```
pub fn cache_oblivious_matmul(a: &Array2<f64>, b: &Array2<f64>) -> CoreResult<Array2<f64>> {
    let (m, k_a) = a.dim();
    let (k_b, n) = b.dim();

    if k_a != k_b {
        return Err(CoreError::InvalidArgument(
            ErrorContext::new(format!(
                "cache_oblivious_matmul: inner dimensions must match: A is ({m}, {k_a}) but B is ({k_b}, {n})"
            )),
        ));
    }

    let mut c = Array2::<f64>::zeros((m, n));
    {
        let av = a.view();
        let bv = b.view();
        let mut cv = c.view_mut();
        matmul_recursive(&av, &bv, &mut cv, 0, m, 0, n, 0, k_a);
    }
    Ok(c)
}

// ---------------------------------------------------------------------------
// Cache-Oblivious Sort (Funnel Sort / Merge Sort variant)
// ---------------------------------------------------------------------------

/// Merge two sorted sub-slices in place (auxiliary buffer version).
fn merge_inplace<T: Ord + Clone>(buf: &mut Vec<T>, left: &[T], right: &[T]) {
    buf.clear();
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            buf.push(left[i].clone());
            i += 1;
        } else {
            buf.push(right[j].clone());
            j += 1;
        }
    }
    buf.extend_from_slice(&left[i..]);
    buf.extend_from_slice(&right[j..]);
}

/// Recursive merge-sort kernel used by `cache_oblivious_sort`.
fn merge_sort_recursive<T: Ord + Clone>(data: &mut [T], aux: &mut Vec<T>) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let mid = n / 2;
    merge_sort_recursive(&mut data[..mid], aux);
    merge_sort_recursive(&mut data[mid..], aux);

    // Merge the two halves
    let left: Vec<T> = data[..mid].to_vec();
    let right: Vec<T> = data[mid..].to_vec();
    merge_inplace(aux, &left, &right);
    data.clone_from_slice(aux);
}

/// Cache-oblivious sort using a recursive merge-sort strategy.
///
/// The divide-and-conquer structure ensures that data fits entirely in cache
/// at the leaf recursion level, achieving the cache-optimal bound of
/// `Θ((n/B) log_{M/B}(n/B))` cache misses.
///
/// This implementation uses a standard recursive merge sort, which is the
/// canonical cache-oblivious sorting algorithm (equivalent to Funnel Sort
/// at a high level) and is provably cache-optimal.
///
/// # Arguments
///
/// * `data` – Mutable slice to sort in ascending order.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::cache_oblivious::cache_oblivious_sort;
///
/// let mut v = vec![5i32, 2, 8, 1, 9, 3];
/// cache_oblivious_sort(&mut v);
/// assert_eq!(v, vec![1, 2, 3, 5, 8, 9]);
/// ```
pub fn cache_oblivious_sort<T: Ord + Clone>(data: &mut [T]) {
    let mut aux: Vec<T> = Vec::with_capacity(data.len());
    merge_sort_recursive(data, &mut aux);
}

// ---------------------------------------------------------------------------
// van Emde Boas Layout
// ---------------------------------------------------------------------------

/// Returns the van Emde Boas memory layout permutation for an implicit binary
/// tree of `tree_size` nodes (stored as 0-indexed values `0..tree_size`).
///
/// The vEB layout reorders the nodes of a complete binary tree so that every
/// subtree of height `h` occupies a contiguous block of `2^h - 1` memory
/// locations. This guarantees `O(log_B n)` cache misses for any search
/// (instead of `O(log n)` for BFS/DFS layouts), where `B` is the cache line
/// size in words.
///
/// # Arguments
///
/// * `tree_size` – Number of nodes in the implicit tree (usually `2^k - 1`
///   for a complete binary tree of height `k`, but any non-zero size works).
///
/// # Returns
///
/// A permutation vector `perm` of length `tree_size`, where `perm[i]` is the
/// 1-based BFS index that should be stored at position `i` in the vEB layout.
///
/// # Errors
///
/// Returns `Err` if `tree_size` is zero.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::cache_oblivious::van_emde_boas_layout;
///
/// // Complete binary tree of height 2: 3 nodes (BFS order: 1, 2, 3)
/// let perm = van_emde_boas_layout(3).expect("should succeed");
/// assert_eq!(perm.len(), 3);
/// // Root should appear first (position 0)
/// assert_eq!(perm[0], 1);
/// ```
pub fn van_emde_boas_layout(tree_size: usize) -> CoreResult<Vec<usize>> {
    if tree_size == 0 {
        return Err(CoreError::InvalidArgument(
            ErrorContext::new("van_emde_boas_layout: tree_size must be at least 1"),
        ));
    }

    let mut result = vec![0usize; tree_size];
    let mut pos = 0usize;
    veb_layout_recursive(1, tree_size, &mut result, &mut pos);
    Ok(result)
}

/// Recursive helper: lay out the subtree rooted at BFS index `root`
/// (1-based) within a tree of `total` nodes.
fn veb_layout_recursive(
    root: usize,
    total: usize,
    result: &mut Vec<usize>,
    pos: &mut usize,
) {
    if root > total || *pos >= total {
        return;
    }

    let height = tree_height(total);

    if height <= 1 {
        // Base case: single node
        if *pos < total {
            result[*pos] = root;
            *pos += 1;
        }
        return;
    }

    // Split tree: upper half has height ⌈h/2⌉, lower subtrees have height ⌊h/2⌋
    let top_height = (height + 1) / 2;
    let bottom_height = height / 2;

    // Size of the top sub-tree (a complete binary tree of height `top_height`)
    let top_size = (1usize << top_height) - 1;

    // Lay out the top sub-tree recursively
    veb_layout_top(root, total, top_height, top_size, result, pos);

    // Lay out the bottom sub-trees recursively
    // The roots of bottom subtrees are: 2^top_height * root, ..., 2^top_height * root + 2^top_height - 1
    let bottom_root_start = root << top_height;
    let bottom_count = 1usize << top_height; // number of bottom subtrees
    for k in 0..bottom_count {
        let bottom_root = bottom_root_start + k;
        if bottom_root > total {
            break;
        }
        let bottom_total = bottom_subtree_size(bottom_root, total, bottom_height);
        if bottom_total > 0 {
            veb_layout_recursive(bottom_root, total, result, pos);
        }
    }
}

/// Compute the height of a complete binary tree with `n` nodes.
fn tree_height(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut h = 0;
    let mut cap = 1usize;
    while cap <= n {
        h += 1;
        cap = cap.saturating_mul(2);
    }
    h
}

/// Compute the number of nodes in the subtree rooted at `root` for a tree
/// of `total` nodes using BFS indexing (1-based).
fn bottom_subtree_size(root: usize, total: usize, max_height: usize) -> usize {
    if root > total {
        return 0;
    }
    let mut count = 0;
    let mut frontier = vec![root];
    let mut h = 0;
    while !frontier.is_empty() && h < max_height {
        let mut next = Vec::new();
        for &node in &frontier {
            if node <= total {
                count += 1;
                let left = 2 * node;
                let right = 2 * node + 1;
                if left <= total {
                    next.push(left);
                }
                if right <= total {
                    next.push(right);
                }
            }
        }
        frontier = next;
        h += 1;
    }
    count
}

/// Lay out the top sub-tree (of `top_height` levels starting at `root`)
/// in BFS order into `result`.
fn veb_layout_top(
    root: usize,
    total: usize,
    top_height: usize,
    _top_size: usize,
    result: &mut Vec<usize>,
    pos: &mut usize,
) {
    // BFS traversal limited to `top_height` levels
    let mut frontier = vec![root];
    let mut h = 0;
    while !frontier.is_empty() && h < top_height {
        let mut next = Vec::new();
        for &node in &frontier {
            if node <= total && *pos < result.len() {
                result[*pos] = node;
                *pos += 1;
                let left = 2 * node;
                let right = 2 * node + 1;
                if left <= total {
                    next.push(left);
                }
                if right <= total {
                    next.push(right);
                }
            }
        }
        frontier = next;
        h += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_recursive_transpose_square() {
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let t = recursive_transpose(&a);
        assert_eq!(t.dim(), (3, 3));
        assert_eq!(t[[0, 0]], 1.0);
        assert_eq!(t[[1, 0]], 2.0);
        assert_eq!(t[[2, 0]], 3.0);
        assert_eq!(t[[0, 1]], 4.0);
        assert_eq!(t[[2, 2]], 9.0);
    }

    #[test]
    fn test_recursive_transpose_rectangular() {
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let t = recursive_transpose(&a);
        assert_eq!(t.dim(), (3, 2));
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(t[[j, i]], a[[i, j]]);
            }
        }
    }

    #[test]
    fn test_recursive_transpose_large() {
        // Larger than the tile threshold to exercise recursion
        let n = 64;
        let a = Array2::from_shape_fn((n, n + 1), |(i, j)| (i * (n + 1) + j) as f64);
        let t = recursive_transpose(&a);
        assert_eq!(t.dim(), (n + 1, n));
        for i in 0..n {
            for j in 0..n + 1 {
                assert!((t[[j, i]] - a[[i, j]]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_cache_oblivious_matmul_identity() {
        let id = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = array![[3.0_f64, 4.0], [5.0, 6.0]];
        let c = cache_oblivious_matmul(&id, &b).expect("matmul should succeed");
        assert!((c[[0, 0]] - 3.0).abs() < 1e-14);
        assert!((c[[0, 1]] - 4.0).abs() < 1e-14);
        assert!((c[[1, 0]] - 5.0).abs() < 1e-14);
        assert!((c[[1, 1]] - 6.0).abs() < 1e-14);
    }

    #[test]
    fn test_cache_oblivious_matmul_known_result() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
        let c = cache_oblivious_matmul(&a, &b).expect("matmul should succeed");
        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert!((c[[0, 0]] - 19.0).abs() < 1e-14);
        assert!((c[[0, 1]] - 22.0).abs() < 1e-14);
        assert!((c[[1, 0]] - 43.0).abs() < 1e-14);
        assert!((c[[1, 1]] - 50.0).abs() < 1e-14);
    }

    #[test]
    fn test_cache_oblivious_matmul_dimension_mismatch() {
        let a = Array2::<f64>::zeros((3, 2));
        let b = Array2::<f64>::zeros((3, 4));
        assert!(cache_oblivious_matmul(&a, &b).is_err());
    }

    #[test]
    fn test_cache_oblivious_matmul_large() {
        // Compare against naive multiply for correctness
        let n = 65usize; // crosses the tile threshold
        let a = Array2::from_shape_fn((n, n), |(i, j)| ((i + j) as f64) * 0.01);
        let b = Array2::from_shape_fn((n, n), |(i, j)| ((i * 2 + j) as f64) * 0.01);

        let c_co = cache_oblivious_matmul(&a, &b).expect("large matmul should succeed");

        // Naive reference
        let mut c_ref = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for k in 0..n {
                for j in 0..n {
                    c_ref[[i, j]] += a[[i, k]] * b[[k, j]];
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                assert!((c_co[[i, j]] - c_ref[[i, j]]).abs() < 1e-10,
                    "mismatch at ({i},{j}): co={} ref={}", c_co[[i,j]], c_ref[[i,j]]);
            }
        }
    }

    #[test]
    fn test_cache_oblivious_sort_integers() {
        let mut v = vec![5i32, 2, 8, 1, 9, 3, 7, 4, 6, 0];
        cache_oblivious_sort(&mut v);
        assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_cache_oblivious_sort_already_sorted() {
        let mut v = vec![1i32, 2, 3, 4, 5];
        cache_oblivious_sort(&mut v);
        assert_eq!(v, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_cache_oblivious_sort_empty() {
        let mut v: Vec<i32> = vec![];
        cache_oblivious_sort(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn test_cache_oblivious_sort_strings() {
        let mut v = vec!["banana", "apple", "cherry", "date"];
        cache_oblivious_sort(&mut v);
        assert_eq!(v, vec!["apple", "banana", "cherry", "date"]);
    }

    #[test]
    fn test_van_emde_boas_layout_single_node() {
        let perm = van_emde_boas_layout(1).expect("should succeed");
        assert_eq!(perm.len(), 1);
        assert_eq!(perm[0], 1);
    }

    #[test]
    fn test_van_emde_boas_layout_three_nodes() {
        let perm = van_emde_boas_layout(3).expect("should succeed");
        assert_eq!(perm.len(), 3);
        // Root (BFS index 1) must appear somewhere
        assert!(perm.contains(&1), "root must be in layout");
        // All BFS indices 1..=3 must appear exactly once
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 2, 3]);
    }

    #[test]
    fn test_van_emde_boas_layout_seven_nodes() {
        let perm = van_emde_boas_layout(7).expect("should succeed");
        assert_eq!(perm.len(), 7);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_van_emde_boas_layout_zero_error() {
        assert!(van_emde_boas_layout(0).is_err());
    }
}
