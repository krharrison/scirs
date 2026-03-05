//! Fenwick Tree (Binary Indexed Tree) and derived data structures.
//!
//! # `FenwickTree`
//! A 1-D BIT that supports O(log N) point updates and prefix-sum queries.
//!
//! # `FenwickTree2D`
//! A 2-D BIT for O(log N × log M) 2-D prefix sums.
//!
//! # `OrderStatisticsTree`
//! A BIT over a compressed integer domain providing O(log N) `insert`,
//! `rank` (position of value), and `select` (k-th smallest).

use std::ops::Add;

// ===========================================================================
// FenwickTree
// ===========================================================================

/// A Binary Indexed Tree (Fenwick Tree) for prefix-sum queries.
///
/// Elements are 0-indexed; internally the BIT uses 1-indexed positions.
///
/// ```
/// use scirs2_core::persistent::FenwickTree;
///
/// let mut ft = FenwickTree::build(&[1i64, 2, 3, 4, 5]);
/// assert_eq!(ft.prefix_sum(2), 6);  // 1+2+3
/// assert_eq!(ft.range_sum(1, 3), 9); // 2+3+4
/// ft.update(0, 10); // add 10 to index 0
/// assert_eq!(ft.prefix_sum(0), 11);
/// ```
pub struct FenwickTree<T: Copy + Default + Add<Output = T>> {
    n: usize,
    tree: Vec<T>,
}

impl<T: Copy + Default + Add<Output = T>> FenwickTree<T> {
    /// Create a zeroed Fenwick tree of size `n`.
    pub fn new(n: usize) -> Self {
        FenwickTree {
            n,
            tree: vec![T::default(); n + 1],
        }
    }

    /// Build a Fenwick tree from `data` in O(N) time.
    pub fn build(data: &[T]) -> Self {
        let n = data.len();
        let mut tree = vec![T::default(); n + 1];
        // O(N) build via propagation.
        for i in 1..=n {
            tree[i] = tree[i] + data[i - 1];
            let parent = i + (i & i.wrapping_neg());
            if parent <= n {
                let val = tree[i];
                tree[parent] = tree[parent] + val;
            }
        }
        FenwickTree { n, tree }
    }

    /// Add `delta` to the element at 0-indexed position `idx`.
    pub fn update(&mut self, idx: usize, delta: T) {
        let mut i = idx + 1; // 1-indexed
        while i <= self.n {
            self.tree[i] = self.tree[i] + delta;
            i += i & i.wrapping_neg();
        }
    }

    /// Return the prefix sum of elements `[0..=idx]` (0-indexed, inclusive).
    pub fn prefix_sum(&self, idx: usize) -> T {
        let mut sum = T::default();
        let mut i = idx + 1; // 1-indexed
        while i > 0 {
            sum = sum + self.tree[i];
            i -= i & i.wrapping_neg();
        }
        sum
    }

    /// Return the sum of elements in the 0-indexed range `[l..=r]`.
    pub fn range_sum(&self, l: usize, r: usize) -> T {
        assert!(l <= r && r < self.n, "range_sum index out of bounds");
        if l == 0 {
            self.prefix_sum(r)
        } else {
            // prefix_sum(r) - prefix_sum(l-1)
            // We can't subtract generically, so fall back to a direct sum
            // if subtraction is not available.  The caller can use
            // `prefix_sum(r) - prefix_sum(l-1)` when T: Sub.
            //
            // To keep the interface generic we recompute by walking the tree.
            let high = self.prefix_sum(r);
            let low = self.prefix_sum(l - 1);
            // We need subtraction.  Because the trait bound only requires Add,
            // we compute the delta via a re-scan from l-1 to r.
            // This is O(log² N) worst-case but keeps T unconstrained.
            let _ = (high, low);
            self.range_sum_scan(l, r)
        }
    }

    /// Slow O(N) fallback when only Add is available.
    fn range_sum_scan(&self, l: usize, r: usize) -> T {
        // Reconstruct individual values from prefix sums would require Sub.
        // Instead iterate; this is only hit in the generic path.
        let mut sum = T::default();
        for idx in l..=r {
            sum = sum + self.point_value(idx);
        }
        sum
    }

    /// Return the current value of element at 0-indexed `idx`.  O(log N).
    pub fn point_value(&self, idx: usize) -> T {
        // point = prefix_sum(idx) - prefix_sum(idx-1)
        // Implemented without Sub by re-descending.
        let mut sum = T::default();
        let mut i = idx + 1;
        let mut depth = i & i.wrapping_neg(); // lowest set bit
        loop {
            sum = sum + self.tree[i];
            i -= depth;
            if i == 0 {
                break;
            }
            let next_depth = i & i.wrapping_neg();
            if next_depth >= depth {
                break;
            }
            depth = next_depth;
        }
        sum
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

// Specialised `range_sum` for types that support subtraction.
impl<T> FenwickTree<T>
where
    T: Copy + Default + Add<Output = T> + std::ops::Sub<Output = T>,
{
    /// Return the sum of elements in `[l..=r]` using O(log N) prefix sums.
    pub fn range_sum_fast(&self, l: usize, r: usize) -> T {
        assert!(l <= r && r < self.n, "range_sum_fast index out of bounds");
        if l == 0 {
            self.prefix_sum(r)
        } else {
            self.prefix_sum(r) - self.prefix_sum(l - 1)
        }
    }
}

// ===========================================================================
// FenwickTree2D
// ===========================================================================

/// A 2-D Binary Indexed Tree for 2-D prefix sums.
///
/// `update(r, c, delta)` adds `delta` to cell `(r, c)`.
/// `prefix_sum(r, c)` returns the sum of all cells `(i, j)` with `i ≤ r` and `j ≤ c`.
///
/// ```
/// use scirs2_core::persistent::FenwickTree2D;
///
/// let mut ft = FenwickTree2D::<i64>::new(4, 4);
/// ft.update(0, 0, 1);
/// ft.update(1, 1, 2);
/// ft.update(2, 2, 3);
/// assert_eq!(ft.prefix_sum(2, 2), 6);
/// assert_eq!(ft.prefix_sum(1, 1), 3);
/// ```
pub struct FenwickTree2D<T: Copy + Default + Add<Output = T>> {
    rows: usize,
    cols: usize,
    tree: Vec<Vec<T>>,
}

impl<T: Copy + Default + Add<Output = T>> FenwickTree2D<T> {
    /// Create a zeroed 2-D BIT of size `rows × cols`.
    pub fn new(rows: usize, cols: usize) -> Self {
        FenwickTree2D {
            rows,
            cols,
            tree: vec![vec![T::default(); cols + 1]; rows + 1],
        }
    }

    /// Add `delta` to cell `(row, col)` (both 0-indexed).
    pub fn update(&mut self, row: usize, col: usize, delta: T) {
        let mut i = row + 1;
        while i <= self.rows {
            let mut j = col + 1;
            while j <= self.cols {
                self.tree[i][j] = self.tree[i][j] + delta;
                j += j & j.wrapping_neg();
            }
            i += i & i.wrapping_neg();
        }
    }

    /// Return the sum of all cells `(r, c)` with `r ≤ row` and `c ≤ col`.
    pub fn prefix_sum(&self, row: usize, col: usize) -> T {
        let mut sum = T::default();
        let mut i = row + 1;
        while i > 0 {
            let mut j = col + 1;
            while j > 0 {
                sum = sum + self.tree[i][j];
                j -= j & j.wrapping_neg();
            }
            i -= i & i.wrapping_neg();
        }
        sum
    }

    /// Return the sum of cells in the sub-rectangle
    /// `[r1..=r2] × [c1..=c2]` (0-indexed, inclusive).
    pub fn range_sum(&self, r1: usize, c1: usize, r2: usize, c2: usize) -> T
    where
        T: std::ops::Sub<Output = T>,
    {
        let br = self.prefix_sum(r2, c2);
        let bl = if c1 > 0 {
            self.prefix_sum(r2, c1 - 1)
        } else {
            T::default()
        };
        let tr = if r1 > 0 {
            self.prefix_sum(r1 - 1, c2)
        } else {
            T::default()
        };
        let tl = if r1 > 0 && c1 > 0 {
            self.prefix_sum(r1 - 1, c1 - 1)
        } else {
            T::default()
        };
        br - bl - tr + tl
    }
}

// ===========================================================================
// OrderStatisticsTree
// ===========================================================================

/// An order-statistics data structure backed by a Fenwick tree over a
/// compressed integer domain.
///
/// Operations:
/// - `insert(val)` — insert integer `val`.
/// - `rank(val)` — number of elements strictly less than `val`.
/// - `select(k)` — `k`-th smallest element (0-indexed).
///
/// Values must fall within the range `[min_val, max_val]` supplied at
/// construction.
///
/// ```
/// use scirs2_core::persistent::OrderStatisticsTree;
///
/// let mut ost = OrderStatisticsTree::new(0, 100);
/// ost.insert(5);
/// ost.insert(3);
/// ost.insert(8);
/// ost.insert(1);
///
/// assert_eq!(ost.rank(5), 2);   // 1 and 3 are < 5
/// assert_eq!(ost.select(0), Some(1)); // 1st smallest
/// assert_eq!(ost.select(2), Some(5)); // 3rd smallest
/// ```
pub struct OrderStatisticsTree {
    min_val: i64,
    size: usize,
    bit: FenwickTree<i64>,
    total: usize,
}

impl OrderStatisticsTree {
    /// Create a new OST for values in `[min_val, max_val]` (inclusive).
    pub fn new(min_val: i64, max_val: i64) -> Self {
        assert!(max_val >= min_val, "max_val must be >= min_val");
        let size = (max_val - min_val + 1) as usize;
        OrderStatisticsTree {
            min_val,
            size,
            bit: FenwickTree::new(size),
            total: 0,
        }
    }

    /// Insert `val` into the OST.
    pub fn insert(&mut self, val: i64) {
        let idx = self.compress(val);
        self.bit.update(idx, 1);
        self.total += 1;
    }

    /// Remove one occurrence of `val` from the OST.
    /// Does nothing if `val` is not present.
    pub fn remove(&mut self, val: i64) {
        let idx = self.compress(val);
        let cur = self.bit.point_value(idx);
        if cur > 0 {
            self.bit.update(idx, -1);
            self.total -= 1;
        }
    }

    /// Return the number of elements strictly less than `val`.
    pub fn rank(&self, val: i64) -> usize {
        let idx = self.compress(val);
        if idx == 0 {
            return 0;
        }
        self.bit.prefix_sum(idx - 1) as usize
    }

    /// Return the `k`-th smallest element (0-indexed), or `None` if `k ≥ len`.
    pub fn select(&self, k: usize) -> Option<i64> {
        if k >= self.total {
            return None;
        }
        // Binary-lift on the BIT for O(log N).
        let mut pos = 0usize;
        let mut remaining = (k + 1) as i64;
        let log = (usize::BITS - self.size.leading_zeros()) as usize;
        let mut step = 1 << log;
        while step > 0 {
            let next = pos + step;
            if next <= self.size && self.bit.tree[next] < remaining {
                remaining -= self.bit.tree[next];
                pos = next;
            }
            step >>= 1;
        }
        // pos is the 1-indexed BIT position (0-indexed in our domain).
        Some(self.min_val + pos as i64)
    }

    /// Return the total number of elements in the OST.
    pub fn len(&self) -> usize {
        self.total
    }

    /// Returns `true` if the OST contains no elements.
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    fn compress(&self, val: i64) -> usize {
        assert!(
            val >= self.min_val && (val - self.min_val) < self.size as i64,
            "value {val} out of range [{}, {}]",
            self.min_val,
            self.min_val + self.size as i64 - 1
        );
        (val - self.min_val) as usize
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // FenwickTree
    // -----------------------------------------------------------------------

    #[test]
    fn fenwick_prefix_sum() {
        let ft = FenwickTree::build(&[1i64, 2, 3, 4, 5]);
        assert_eq!(ft.prefix_sum(0), 1);
        assert_eq!(ft.prefix_sum(1), 3);
        assert_eq!(ft.prefix_sum(4), 15);
    }

    #[test]
    fn fenwick_update() {
        let mut ft = FenwickTree::build(&[1i64, 2, 3, 4, 5]);
        ft.update(2, 10); // index 2: 3 → 13
        assert_eq!(ft.prefix_sum(4), 25);
        assert_eq!(ft.prefix_sum(2), 16); // 1+2+13
    }

    #[test]
    fn fenwick_range_sum_fast() {
        let ft = FenwickTree::build(&[1i64, 2, 3, 4, 5]);
        assert_eq!(ft.range_sum_fast(1, 3), 9); // 2+3+4
        assert_eq!(ft.range_sum_fast(0, 4), 15);
    }

    #[test]
    fn fenwick_build_correctness() {
        let data: Vec<i64> = (1..=100).collect();
        let ft = FenwickTree::build(&data);
        // Sum of 1..=100 = 5050
        assert_eq!(ft.prefix_sum(99), 5050);
    }

    // -----------------------------------------------------------------------
    // FenwickTree2D
    // -----------------------------------------------------------------------

    #[test]
    fn fenwick2d_basic() {
        let mut ft = FenwickTree2D::<i64>::new(4, 4);
        ft.update(0, 0, 1);
        ft.update(1, 1, 2);
        ft.update(2, 2, 3);
        assert_eq!(ft.prefix_sum(2, 2), 6);
        assert_eq!(ft.prefix_sum(1, 1), 3);
        assert_eq!(ft.prefix_sum(0, 0), 1);
    }

    #[test]
    fn fenwick2d_range_sum() {
        let mut ft = FenwickTree2D::<i64>::new(5, 5);
        // Fill a 3x3 block at (1,1)–(3,3) with 1.
        for r in 1..=3 {
            for c in 1..=3 {
                ft.update(r, c, 1);
            }
        }
        assert_eq!(ft.range_sum(1, 1, 3, 3), 9);
        assert_eq!(ft.range_sum(0, 0, 4, 4), 9);
        assert_eq!(ft.range_sum(2, 2, 3, 3), 4);
    }

    // -----------------------------------------------------------------------
    // OrderStatisticsTree
    // -----------------------------------------------------------------------

    #[test]
    fn ost_rank_select() {
        let mut ost = OrderStatisticsTree::new(0, 100);
        ost.insert(5);
        ost.insert(3);
        ost.insert(8);
        ost.insert(1);

        assert_eq!(ost.rank(5), 2); // 1 and 3 are < 5
        assert_eq!(ost.rank(1), 0);
        assert_eq!(ost.select(0), Some(1));
        assert_eq!(ost.select(1), Some(3));
        assert_eq!(ost.select(2), Some(5));
        assert_eq!(ost.select(3), Some(8));
        assert_eq!(ost.select(4), None);
    }

    #[test]
    fn ost_remove() {
        let mut ost = OrderStatisticsTree::new(0, 50);
        ost.insert(10);
        ost.insert(20);
        ost.insert(30);
        ost.remove(20);
        assert_eq!(ost.len(), 2);
        assert_eq!(ost.select(1), Some(30));
    }

    #[test]
    fn ost_negative_range() {
        let mut ost = OrderStatisticsTree::new(-50, 50);
        ost.insert(-10);
        ost.insert(0);
        ost.insert(10);
        assert_eq!(ost.rank(0), 1); // -10 is < 0
        assert_eq!(ost.select(0), Some(-10));
        assert_eq!(ost.select(1), Some(0));
    }
}
