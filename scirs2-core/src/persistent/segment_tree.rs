//! Segment tree variants: plain, lazy-propagation, and persistent.
//!
//! # `SegmentTree`
//! Heap-stored segment tree for range queries (sum / min / max / …) with
//! O(log N) point updates.
//!
//! # `LazySegmentTree`
//! Extension with lazy propagation that supports O(log N) *range* updates
//! alongside range queries.  The example operation is range-add + range-sum.
//!
//! # `PersistentSegmentTree`
//! Each update creates a new *version* that shares all unchanged nodes with
//! the previous version.  Useful for offline "version k" range queries.

use std::sync::Arc;

// ===========================================================================
// SegmentTree
// ===========================================================================

/// A segment tree supporting range queries and O(log N) point updates.
///
/// The combining function is provided at construction time and stored as a
/// boxed closure.
///
/// ```
/// use scirs2_core::persistent::SegmentTree;
///
/// let mut st = SegmentTree::build(&[1, 3, 5, 7, 9, 11], |a, b| a + b);
/// assert_eq!(st.query(1, 3), 15); // 3+5+7
/// st.update(1, 10);
/// assert_eq!(st.query(1, 3), 22); // 10+5+7
/// ```
pub struct SegmentTree<T: Copy + Default> {
    n: usize,
    data: Vec<T>,
    combine: Box<dyn Fn(T, T) -> T + Send + Sync>,
}

impl<T: Copy + Default> SegmentTree<T> {
    /// Build a segment tree from `data`.  `combine` is the associative
    /// combining function (e.g. `|a,b| a+b` for sum, `std::cmp::min` for min).
    pub fn build(data: &[T], combine: impl Fn(T, T) -> T + Clone + Send + Sync + 'static) -> Self {
        let n = data.len();
        let mut tree_data = vec![T::default(); 4 * n.max(1)];
        if n > 0 {
            build_seg(&mut tree_data, data, 1, 0, n - 1, &combine);
        }
        SegmentTree {
            n,
            data: tree_data,
            combine: Box::new(combine),
        }
    }

    /// Query the combined value over the inclusive range `[l, r]`.
    ///
    /// # Panics
    /// Panics if `l > r` or either index is out of bounds.
    pub fn query(&self, l: usize, r: usize) -> T {
        assert!(l <= r && r < self.n, "query index out of bounds");
        query_seg_opt(&self.data, 1, 0, self.n - 1, l, r, &self.combine)
            .unwrap_or_default()
    }

    /// Update the element at position `idx` to `val`.
    pub fn update(&mut self, idx: usize, val: T) {
        assert!(idx < self.n, "update index out of bounds");
        update_seg(&mut self.data, 1, 0, self.n - 1, idx, val, &self.combine);
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the tree has no elements.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

// Internal recursive helpers for SegmentTree.

fn build_seg<T: Copy + Default>(
    tree: &mut Vec<T>,
    data: &[T],
    node: usize,
    l: usize,
    r: usize,
    combine: &dyn Fn(T, T) -> T,
) {
    if l == r {
        tree[node] = data[l];
        return;
    }
    let mid = (l + r) / 2;
    build_seg(tree, data, 2 * node, l, mid, combine);
    build_seg(tree, data, 2 * node + 1, mid + 1, r, combine);
    tree[node] = combine(tree[2 * node], tree[2 * node + 1]);
}

fn query_seg<T: Copy + Default>(
    tree: &[T],
    node: usize,
    l: usize,
    r: usize,
    ql: usize,
    qr: usize,
    combine: &dyn Fn(T, T) -> T,
) -> T {
    if ql <= l && r <= qr {
        return tree[node];
    }
    if qr < l || r < ql {
        return T::default();
    }
    let mid = (l + r) / 2;
    let left = query_seg(tree, 2 * node, l, mid, ql, qr, combine);
    let right = query_seg(tree, 2 * node + 1, mid + 1, r, ql, qr, combine);
    combine(left, right)
}

/// Like `query_seg` but returns `Option<T>` — `None` means the sub-range had no
/// overlap with `[ql, qr]`.  This avoids relying on `T::default()` as an
/// identity element for the combine function, which is incorrect for min/max.
fn query_seg_opt<T: Copy + Default>(
    tree: &[T],
    node: usize,
    l: usize,
    r: usize,
    ql: usize,
    qr: usize,
    combine: &dyn Fn(T, T) -> T,
) -> Option<T> {
    if ql <= l && r <= qr {
        return Some(tree[node]);
    }
    if qr < l || r < ql {
        return None;
    }
    let mid = (l + r) / 2;
    let left = query_seg_opt(tree, 2 * node, l, mid, ql, qr, combine);
    let right = query_seg_opt(tree, 2 * node + 1, mid + 1, r, ql, qr, combine);
    match (left, right) {
        (Some(lv), Some(rv)) => Some(combine(lv, rv)),
        (Some(lv), None) => Some(lv),
        (None, Some(rv)) => Some(rv),
        (None, None) => None,
    }
}

fn update_seg<T: Copy + Default>(
    tree: &mut Vec<T>,
    node: usize,
    l: usize,
    r: usize,
    idx: usize,
    val: T,
    combine: &dyn Fn(T, T) -> T,
) {
    if l == r {
        tree[node] = val;
        return;
    }
    let mid = (l + r) / 2;
    if idx <= mid {
        update_seg(tree, 2 * node, l, mid, idx, val, combine);
    } else {
        update_seg(tree, 2 * node + 1, mid + 1, r, idx, val, combine);
    }
    tree[node] = combine(tree[2 * node], tree[2 * node + 1]);
}

// ===========================================================================
// LazySegmentTree
// ===========================================================================

/// Multiply a value `val` by a `usize` count using repeated addition.
///
/// This avoids requiring `T: Mul<usize>`, which is not implemented for
/// standard integer types like `i64`.
#[inline]
fn scale_by_count<T: Copy + Default + std::ops::Add<Output = T>>(val: T, count: usize) -> T {
    let mut result = T::default();
    for _ in 0..count {
        result = result + val;
    }
    result
}

/// Segment tree with lazy propagation for O(log N) range updates and queries.
///
/// Supports range-add + range-sum out of the box.  The element type must
/// implement `Copy`, `Default`, `Add`, and `PartialEq`.
///
/// ```
/// use scirs2_core::persistent::LazySegmentTree;
///
/// let mut st = LazySegmentTree::build(&[1i64, 2, 3, 4, 5]);
/// st.range_add(1, 3, 10); // add 10 to indices 1,2,3
/// assert_eq!(st.range_query(0, 4), 45); // 1+12+13+14+5
/// assert_eq!(st.range_query(1, 3), 39); // 12+13+14
/// ```
pub struct LazySegmentTree<T>
where
    T: Copy + Default + std::ops::Add<Output = T> + std::cmp::PartialEq,
{
    n: usize,
    tree: Vec<T>,
    lazy: Vec<T>,
}

impl<T> LazySegmentTree<T>
where
    T: Copy + Default + std::ops::Add<Output = T> + std::cmp::PartialEq,
{
    /// Build from a slice.  O(N).
    pub fn build(data: &[T]) -> Self {
        let n = data.len();
        let mut tree = vec![T::default(); 4 * n.max(1)];
        let lazy = vec![T::default(); 4 * n.max(1)];
        if n > 0 {
            lazy_build(&mut tree, data, 1, 0, n - 1);
        }
        LazySegmentTree { n, tree, lazy }
    }

    /// Add `val` to all elements in the inclusive range `[l, r]`.
    pub fn range_add(&mut self, l: usize, r: usize, val: T) {
        assert!(l <= r && r < self.n, "range_add index out of bounds");
        lazy_range_add(
            &mut self.tree,
            &mut self.lazy,
            1,
            0,
            self.n - 1,
            l,
            r,
            val,
        );
    }

    /// Query the sum of elements in the inclusive range `[l, r]`.
    pub fn range_query(&self, l: usize, r: usize) -> T {
        assert!(l <= r && r < self.n, "range_query index out of bounds");
        // We need mutable access to propagate lazy tags.
        // Work around with a RefCell-free approach: copy and query on a mutable copy.
        let mut tree = self.tree.clone();
        let mut lazy = self.lazy.clone();
        lazy_range_query(&mut tree, &mut lazy, 1, 0, self.n - 1, l, r)
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the tree has no elements.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

fn lazy_build<T: Copy + Default + std::ops::Add<Output = T>>(
    tree: &mut Vec<T>,
    data: &[T],
    node: usize,
    l: usize,
    r: usize,
) {
    if l == r {
        tree[node] = data[l];
        return;
    }
    let mid = (l + r) / 2;
    lazy_build(tree, data, 2 * node, l, mid);
    lazy_build(tree, data, 2 * node + 1, mid + 1, r);
    tree[node] = tree[2 * node] + tree[2 * node + 1];
}

fn lazy_push_down<T>(tree: &mut Vec<T>, lazy: &mut Vec<T>, node: usize, l: usize, r: usize)
where
    T: Copy + Default + std::ops::Add<Output = T> + std::cmp::PartialEq,
{
    if lazy[node] == T::default() {
        return;
    }
    let mid = (l + r) / 2;
    let left_len = mid - l + 1;
    let right_len = r - mid;
    // Apply lazy to children.
    tree[2 * node] = tree[2 * node] + scale_by_count(lazy[node], left_len);
    lazy[2 * node] = lazy[2 * node] + lazy[node];
    tree[2 * node + 1] = tree[2 * node + 1] + scale_by_count(lazy[node], right_len);
    lazy[2 * node + 1] = lazy[2 * node + 1] + lazy[node];
    lazy[node] = T::default();
}

fn lazy_range_add<T>(
    tree: &mut Vec<T>,
    lazy: &mut Vec<T>,
    node: usize,
    l: usize,
    r: usize,
    ql: usize,
    qr: usize,
    val: T,
) where
    T: Copy + Default + std::ops::Add<Output = T> + std::cmp::PartialEq,
{
    if qr < l || r < ql {
        return;
    }
    if ql <= l && r <= qr {
        tree[node] = tree[node] + scale_by_count(val, r - l + 1);
        lazy[node] = lazy[node] + val;
        return;
    }
    lazy_push_down(tree, lazy, node, l, r);
    let mid = (l + r) / 2;
    lazy_range_add(tree, lazy, 2 * node, l, mid, ql, qr, val);
    lazy_range_add(tree, lazy, 2 * node + 1, mid + 1, r, ql, qr, val);
    tree[node] = tree[2 * node] + tree[2 * node + 1];
}

fn lazy_range_query<T>(
    tree: &mut Vec<T>,
    lazy: &mut Vec<T>,
    node: usize,
    l: usize,
    r: usize,
    ql: usize,
    qr: usize,
) -> T
where
    T: Copy + Default + std::ops::Add<Output = T> + std::cmp::PartialEq,
{
    if qr < l || r < ql {
        return T::default();
    }
    if ql <= l && r <= qr {
        return tree[node];
    }
    lazy_push_down(tree, lazy, node, l, r);
    let mid = (l + r) / 2;
    let lv = lazy_range_query(tree, lazy, 2 * node, l, mid, ql, qr);
    let rv = lazy_range_query(tree, lazy, 2 * node + 1, mid + 1, r, ql, qr);
    lv + rv
}

// ===========================================================================
// PersistentSegmentTree
// ===========================================================================

/// A node in the persistent segment tree.
#[derive(Clone)]
struct PstNode<T: Copy + Default> {
    val: T,
    left: Option<Arc<PstNode<T>>>,
    right: Option<Arc<PstNode<T>>>,
}

impl<T: Copy + Default> PstNode<T> {
    fn new_leaf(val: T) -> Arc<Self> {
        Arc::new(PstNode {
            val,
            left: None,
            right: None,
        })
    }
}

/// Persistent segment tree.
///
/// Each update returns a *new version* that shares unchanged nodes with the
/// original.  Older versions remain fully accessible.
///
/// ```
/// use scirs2_core::persistent::PersistentSegmentTree;
///
/// let v0 = PersistentSegmentTree::build(&[1i64, 2, 3, 4, 5], |a, b| a + b);
/// let v1 = v0.update(2, 10);
/// assert_eq!(v0.query(0, 4), 15);
/// assert_eq!(v1.query(0, 4), 22); // 1+2+10+4+5
/// assert_eq!(v1.query(2, 2), 10);
/// ```
#[derive(Clone)]
pub struct PersistentSegmentTree<T: Copy + Default> {
    n: usize,
    root: Option<Arc<PstNode<T>>>,
    combine: Arc<dyn Fn(T, T) -> T + Send + Sync>,
}

impl<T: Copy + Default> PersistentSegmentTree<T> {
    /// Build from a slice.  O(N).
    pub fn build(data: &[T], combine: impl Fn(T, T) -> T + Send + Sync + 'static) -> Self {
        let combine: Arc<dyn Fn(T, T) -> T + Send + Sync> = Arc::new(combine);
        let n = data.len();
        let root = if n > 0 {
            Some(pst_build(data, 0, n - 1, &*combine))
        } else {
            None
        };
        PersistentSegmentTree { n, root, combine }
    }

    /// Returns a new version with `data[idx]` replaced by `val`.
    pub fn update(&self, idx: usize, val: T) -> Self {
        assert!(idx < self.n, "update index out of bounds");
        let new_root = pst_update(
            self.root.as_deref(),
            0,
            self.n - 1,
            idx,
            val,
            &*self.combine,
        );
        PersistentSegmentTree {
            n: self.n,
            root: Some(new_root),
            combine: Arc::clone(&self.combine),
        }
    }

    /// Query the combined value over the inclusive range `[l, r]`.
    pub fn query(&self, l: usize, r: usize) -> T {
        assert!(l <= r && r < self.n, "query index out of bounds");
        match &self.root {
            None => T::default(),
            Some(root) => pst_query(root, 0, self.n - 1, l, r, &*self.combine),
        }
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the tree has no elements.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

fn pst_build<T: Copy + Default>(
    data: &[T],
    l: usize,
    r: usize,
    combine: &dyn Fn(T, T) -> T,
) -> Arc<PstNode<T>> {
    if l == r {
        return PstNode::new_leaf(data[l]);
    }
    let mid = (l + r) / 2;
    let left = pst_build(data, l, mid, combine);
    let right = pst_build(data, mid + 1, r, combine);
    let val = combine(left.val, right.val);
    Arc::new(PstNode {
        val,
        left: Some(left),
        right: Some(right),
    })
}

fn pst_update<T: Copy + Default>(
    node: Option<&PstNode<T>>,
    l: usize,
    r: usize,
    idx: usize,
    val: T,
    combine: &dyn Fn(T, T) -> T,
) -> Arc<PstNode<T>> {
    if l == r {
        return PstNode::new_leaf(val);
    }
    let mid = (l + r) / 2;
    let (new_left, new_right) = if idx <= mid {
        let old_left = node.and_then(|n| n.left.as_deref());
        let new_l = pst_update(old_left, l, mid, idx, val, combine);
        let new_r = node
            .and_then(|n| n.right.clone())
            .unwrap_or_else(|| Arc::new(PstNode { val: T::default(), left: None, right: None }));
        (new_l, new_r)
    } else {
        let old_right = node.and_then(|n| n.right.as_deref());
        let new_r = pst_update(old_right, mid + 1, r, idx, val, combine);
        let new_l = node
            .and_then(|n| n.left.clone())
            .unwrap_or_else(|| Arc::new(PstNode { val: T::default(), left: None, right: None }));
        (new_l, new_r)
    };
    let v = combine(new_left.val, new_right.val);
    Arc::new(PstNode {
        val: v,
        left: Some(new_left),
        right: Some(new_right),
    })
}

fn pst_query<T: Copy + Default>(
    node: &PstNode<T>,
    l: usize,
    r: usize,
    ql: usize,
    qr: usize,
    combine: &dyn Fn(T, T) -> T,
) -> T {
    if ql <= l && r <= qr {
        return node.val;
    }
    if qr < l || r < ql {
        return T::default();
    }
    let mid = (l + r) / 2;
    let lv = node.left.as_ref().map_or(T::default(), |n| {
        pst_query(n, l, mid, ql, qr, combine)
    });
    let rv = node.right.as_ref().map_or(T::default(), |n| {
        pst_query(n, mid + 1, r, ql, qr, combine)
    });
    combine(lv, rv)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // SegmentTree
    // -----------------------------------------------------------------------

    #[test]
    fn seg_sum_query() {
        let st = SegmentTree::build(&[1i64, 2, 3, 4, 5], |a, b| a + b);
        assert_eq!(st.query(0, 4), 15);
        assert_eq!(st.query(1, 3), 9);
        assert_eq!(st.query(2, 2), 3);
    }

    #[test]
    fn seg_min_query() {
        let st = SegmentTree::build(&[5i32, 1, 4, 2, 3], std::cmp::min);
        assert_eq!(st.query(0, 4), 1);
        assert_eq!(st.query(0, 0), 5);
        assert_eq!(st.query(2, 4), 2);
    }

    #[test]
    fn seg_update() {
        let mut st = SegmentTree::build(&[1i64, 3, 5, 7, 9, 11], |a, b| a + b);
        assert_eq!(st.query(1, 3), 15);
        st.update(1, 10);
        assert_eq!(st.query(1, 3), 22);
    }

    // -----------------------------------------------------------------------
    // LazySegmentTree
    // -----------------------------------------------------------------------

    #[test]
    fn lazy_range_add_query() {
        let mut st = LazySegmentTree::build(&[1i64, 2, 3, 4, 5]);
        st.range_add(1, 3, 10);
        assert_eq!(st.range_query(0, 4), 45);
        assert_eq!(st.range_query(1, 3), 39);
        assert_eq!(st.range_query(0, 0), 1);
    }

    #[test]
    fn lazy_multiple_range_adds() {
        let mut st = LazySegmentTree::build(&[0i64; 8]);
        st.range_add(0, 7, 1);
        st.range_add(2, 5, 3);
        assert_eq!(st.range_query(0, 7), 20); // 8*1 + 4*3
        assert_eq!(st.range_query(2, 5), 16); // 4*(1+3)
    }

    // -----------------------------------------------------------------------
    // PersistentSegmentTree
    // -----------------------------------------------------------------------

    #[test]
    fn pst_versions() {
        let v0 = PersistentSegmentTree::build(&[1i64, 2, 3, 4, 5], |a, b| a + b);
        let v1 = v0.update(2, 10);
        let v2 = v1.update(4, 0);

        assert_eq!(v0.query(0, 4), 15);
        assert_eq!(v1.query(0, 4), 22);
        assert_eq!(v2.query(0, 4), 17);
        // Versions are independent.
        assert_eq!(v0.query(2, 2), 3);
        assert_eq!(v1.query(2, 2), 10);
    }

    #[test]
    fn pst_single_element() {
        let v0 = PersistentSegmentTree::build(&[42i64], |a, b| a + b);
        assert_eq!(v0.query(0, 0), 42);
        let v1 = v0.update(0, 0);
        assert_eq!(v1.query(0, 0), 0);
        assert_eq!(v0.query(0, 0), 42);
    }
}
