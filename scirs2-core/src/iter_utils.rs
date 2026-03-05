//! # Iterator Utilities
//!
//! This module provides a rich collection of lazy iterator adaptors that are
//! commonly needed in scientific computing pipelines but absent from the
//! standard library.
//!
//! ## Adaptors Overview
//!
//! | Type / Function              | Description                                              |
//! |------------------------------|----------------------------------------------------------|
//! | [`WindowedIterator`]         | Sliding window with configurable step size               |
//! | [`ChunkedParallelIterator`]  | Parallel processing of non-overlapping chunks            |
//! | [`ZipLongest`]               | Zip two iterators, continuing past the shorter one       |
//! | [`GroupBy`]                  | Group consecutive equal-keyed elements                   |
//! | [`FlatScan`]                 | Stateful `flat_map` (like `scan` + `flat_map` combined)  |
//! | [`take_while_inclusive`]     | `take_while` that includes the first non-matching item   |
//!
//! ## Design Goals
//!
//! - Zero allocation in the hot path wherever possible.
//! - No `unwrap()` usage.
//! - Compatible with Rust's standard `Iterator` trait.
//! - `Send + Sync` where the underlying types allow it.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------

/// The value produced by [`ZipLongest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EitherOrBoth<A, B> {
    /// Both iterators yielded a value.
    Both(A, B),
    /// Only the left iterator yielded a value (right was exhausted).
    Left(A),
    /// Only the right iterator yielded a value (left was exhausted).
    Right(B),
}

// ---------------------------------------------------------------------------
// WindowedIterator
// ---------------------------------------------------------------------------

/// A sliding-window iterator that yields `Vec<T>` windows of size `window`
/// advancing by `step` elements each time.
///
/// - Windows smaller than `window` (at the end) are **not** emitted.
/// - `step` must be ≥ 1; values are clamped to 1 if zero is supplied.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::iter_utils::WindowedIterator;
///
/// let v: Vec<i32> = (1..=6).collect();
/// let windows: Vec<_> = WindowedIterator::new(v.iter().copied(), 3, 1).collect();
/// assert_eq!(windows, vec![vec![1,2,3], vec![2,3,4], vec![3,4,5], vec![4,5,6]]);
/// ```
pub struct WindowedIterator<I: Iterator> {
    iter: I,
    window: usize,
    step: usize,
    buf: VecDeque<I::Item>,
    done: bool,
}

impl<I: Iterator> WindowedIterator<I>
where
    I::Item: Clone,
{
    /// Create a new `WindowedIterator`.
    ///
    /// # Arguments
    ///
    /// * `iter`   – Source iterator.
    /// * `window` – Window size (number of elements per yielded slice). Must be ≥ 1.
    /// * `step`   – Advance step between consecutive windows. Must be ≥ 1.
    pub fn new(iter: I, window: usize, step: usize) -> Self {
        let window = window.max(1);
        let step = step.max(1);
        Self {
            iter,
            window,
            step,
            buf: VecDeque::new(),
            done: false,
        }
    }
}

impl<I: Iterator> Iterator for WindowedIterator<I>
where
    I::Item: Clone,
{
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Fill the buffer up to `window` elements
        while self.buf.len() < self.window {
            match self.iter.next() {
                Some(item) => self.buf.push_back(item),
                None => {
                    self.done = true;
                    return None; // Not enough elements for a full window
                }
            }
        }

        let result: Vec<I::Item> = self.buf.iter().cloned().collect();

        // Advance by `step`
        for _ in 0..self.step {
            self.buf.pop_front();
        }

        Some(result)
    }
}

// ---------------------------------------------------------------------------
// ChunkedParallelIterator
// ---------------------------------------------------------------------------

/// An iterator that processes non-overlapping chunks of size `chunk_size`
/// using a user-supplied mapping function, potentially in parallel.
///
/// Each chunk is mapped to an output `Vec<Out>` by `f`, and the results are
/// yielded in-order.  When `parallel` is `true` the chunks are processed via
/// [`std::thread::scope`] threads; otherwise they are processed sequentially.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::iter_utils::ChunkedParallelIterator;
///
/// let data: Vec<i32> = (1..=9).collect();
/// let mut results: Vec<i32> = Vec::new();
/// for chunk_result in ChunkedParallelIterator::new(data.as_slice(), 3, false, |chunk| {
///     chunk.iter().map(|&x| x * 2).collect()
/// }) {
///     results.extend(chunk_result);
/// }
/// assert_eq!(results, vec![2, 4, 6, 8, 10, 12, 14, 16, 18]);
/// ```
pub struct ChunkedParallelIterator<'a, T, Out, F>
where
    F: Fn(&[T]) -> Vec<Out>,
{
    data: &'a [T],
    chunk_size: usize,
    parallel: bool,
    f: F,
    pos: usize,
    _out: std::marker::PhantomData<Out>,
}

impl<'a, T, Out, F> ChunkedParallelIterator<'a, T, Out, F>
where
    F: Fn(&[T]) -> Vec<Out>,
{
    /// Create a new `ChunkedParallelIterator`.
    ///
    /// # Arguments
    ///
    /// * `data`       – Input data slice.
    /// * `chunk_size` – Number of elements per chunk (clamped to ≥ 1).
    /// * `parallel`   – If `true`, spawn a thread per chunk (requires `T: Send + Sync`, `F: Sync`, `Out: Send`).
    /// * `f`          – Mapping function applied to each chunk.
    pub fn new(data: &'a [T], chunk_size: usize, parallel: bool, f: F) -> Self {
        Self {
            data,
            chunk_size: chunk_size.max(1),
            parallel,
            f,
            pos: 0,
            _out: std::marker::PhantomData,
        }
    }
}

impl<'a, T, Out, F> Iterator for ChunkedParallelIterator<'a, T, Out, F>
where
    T: Send + Sync,
    Out: Send,
    F: Fn(&[T]) -> Vec<Out> + Sync,
{
    type Item = Vec<Out>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }

        let end = (self.pos + self.chunk_size).min(self.data.len());
        let chunk = &self.data[self.pos..end];
        self.pos = end;

        if self.parallel && chunk.len() >= 2 {
            // Spawn a thread for this single chunk to simulate parallel dispatch.
            // In a real pipeline you would batch multiple chunks across threads;
            // this implementation demonstrates the thread-safety contract.
            let result = std::thread::scope(|s| {
                s.spawn(|| (self.f)(chunk)).join()
            });
            // If the thread panicked, fall back to sequential execution
            match result {
                Ok(v) => Some(v),
                Err(_) => Some((self.f)(chunk)),
            }
        } else {
            Some((self.f)(chunk))
        }
    }
}

// ---------------------------------------------------------------------------
// ZipLongest
// ---------------------------------------------------------------------------

/// Zip two iterators together, continuing past the shorter one.
///
/// Yields [`EitherOrBoth::Both`] when both iterators have values,
/// [`EitherOrBoth::Left`] when only `A` remains, and
/// [`EitherOrBoth::Right`] when only `B` remains.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::iter_utils::{ZipLongest, EitherOrBoth};
///
/// let a = vec![1i32, 2, 3];
/// let b = vec![10i32, 20];
/// let zipped: Vec<_> = ZipLongest::new(a.into_iter(), b.into_iter()).collect();
/// assert_eq!(zipped, vec![
///     EitherOrBoth::Both(1, 10),
///     EitherOrBoth::Both(2, 20),
///     EitherOrBoth::Left(3),
/// ]);
/// ```
pub struct ZipLongest<A: Iterator, B: Iterator> {
    a: A,
    b: B,
    a_done: bool,
    b_done: bool,
}

impl<A: Iterator, B: Iterator> ZipLongest<A, B> {
    /// Create a new `ZipLongest` adaptor.
    pub fn new(a: A, b: B) -> Self {
        Self {
            a,
            b,
            a_done: false,
            b_done: false,
        }
    }
}

impl<A: Iterator, B: Iterator> Iterator for ZipLongest<A, B> {
    type Item = EitherOrBoth<A::Item, B::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        let a_val = if self.a_done {
            None
        } else {
            let v = self.a.next();
            if v.is_none() {
                self.a_done = true;
            }
            v
        };

        let b_val = if self.b_done {
            None
        } else {
            let v = self.b.next();
            if v.is_none() {
                self.b_done = true;
            }
            v
        };

        match (a_val, b_val) {
            (Some(a), Some(b)) => Some(EitherOrBoth::Both(a, b)),
            (Some(a), None) => Some(EitherOrBoth::Left(a)),
            (None, Some(b)) => Some(EitherOrBoth::Right(b)),
            (None, None) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// GroupBy
// ---------------------------------------------------------------------------

/// Group consecutive elements with the same key into `(key, Vec<item>)` pairs.
///
/// Only **consecutive** runs are grouped; elements with equal keys separated
/// by differing keys form independent groups (same semantics as Python's
/// `itertools.groupby`).
///
/// # Examples
///
/// ```rust
/// use scirs2_core::iter_utils::GroupBy;
///
/// let data = vec![1i32, 1, 2, 2, 2, 1];
/// let groups: Vec<_> = GroupBy::new(data.into_iter(), |&x| x).collect();
/// assert_eq!(groups, vec![
///     (1, vec![1, 1]),
///     (2, vec![2, 2, 2]),
///     (1, vec![1]),
/// ]);
/// ```
pub struct GroupBy<I, K, F>
where
    I: Iterator,
    K: PartialEq,
    F: Fn(&I::Item) -> K,
{
    iter: I,
    key_fn: F,
    peeked: Option<(K, I::Item)>,
}

impl<I, K, F> GroupBy<I, K, F>
where
    I: Iterator,
    K: PartialEq,
    F: Fn(&I::Item) -> K,
{
    /// Create a new `GroupBy` adaptor.
    ///
    /// # Arguments
    ///
    /// * `iter`   – Source iterator.
    /// * `key_fn` – Function mapping each element to its group key.
    pub fn new(iter: I, key_fn: F) -> Self {
        Self {
            iter,
            key_fn,
            peeked: None,
        }
    }
}

impl<I, K, F> Iterator for GroupBy<I, K, F>
where
    I: Iterator,
    K: PartialEq + Clone,
    F: Fn(&I::Item) -> K,
    I::Item: Clone,
{
    type Item = (K, Vec<I::Item>);

    fn next(&mut self) -> Option<Self::Item> {
        // Get the first element of the next group
        let (current_key, first_item) = match self.peeked.take() {
            Some(peeked) => peeked,
            None => {
                let item = self.iter.next()?;
                let key = (self.key_fn)(&item);
                (key, item)
            }
        };

        let mut group = vec![first_item];

        // Consume all consecutive elements with the same key
        loop {
            match self.iter.next() {
                None => break,
                Some(item) => {
                    let key = (self.key_fn)(&item);
                    if key == current_key {
                        group.push(item);
                    } else {
                        // Different key: store for next call
                        self.peeked = Some((key, item));
                        break;
                    }
                }
            }
        }

        Some((current_key, group))
    }
}

// ---------------------------------------------------------------------------
// FlatScan
// ---------------------------------------------------------------------------

/// A stateful `flat_map`: carries mutable state across calls and flattens
/// the returned iterators.
///
/// Like `Iterator::scan` but with a flattening step: `f` takes `(&mut state, item)`
/// and returns an `IntoIterator`.  All items from the returned iterator are
/// yielded before calling `f` again.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::iter_utils::FlatScan;
///
/// // Running product: for each value x, emit [running_product, x]
/// let data = vec![2i64, 3, 4];
/// let result: Vec<i64> = FlatScan::new(
///     data.into_iter(),
///     1i64,
///     |state, x| {
///         *state *= x;
///         vec![*state, x]
///     },
/// ).collect();
/// assert_eq!(result, vec![2, 2, 6, 3, 24, 4]);
/// ```
pub struct FlatScan<I, S, F, Iter>
where
    I: Iterator,
    F: FnMut(&mut S, I::Item) -> Iter,
    Iter: IntoIterator,
{
    iter: I,
    state: S,
    f: F,
    buffer: std::collections::VecDeque<Iter::Item>,
}

impl<I, S, F, Iter> FlatScan<I, S, F, Iter>
where
    I: Iterator,
    F: FnMut(&mut S, I::Item) -> Iter,
    Iter: IntoIterator,
{
    /// Create a new `FlatScan` adaptor.
    ///
    /// # Arguments
    ///
    /// * `iter`  – Source iterator.
    /// * `state` – Initial state value.
    /// * `f`     – Function `(&mut state, item) -> impl IntoIterator`.
    pub fn new(iter: I, state: S, f: F) -> Self {
        Self {
            iter,
            state,
            f,
            buffer: VecDeque::new(),
        }
    }
}

impl<I, S, F, Iter> Iterator for FlatScan<I, S, F, Iter>
where
    I: Iterator,
    F: FnMut(&mut S, I::Item) -> Iter,
    Iter: IntoIterator,
{
    type Item = Iter::Item;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Drain any buffered output first
            if let Some(item) = self.buffer.pop_front() {
                return Some(item);
            }

            // Advance the source iterator
            let source_item = self.iter.next()?;
            let produced = (self.f)(&mut self.state, source_item);
            self.buffer.extend(produced);
        }
    }
}

// ---------------------------------------------------------------------------
// take_while_inclusive
// ---------------------------------------------------------------------------

/// Extension trait that adds [`take_while_inclusive`](IteratorExt::take_while_inclusive)
/// to all iterators.
pub trait IteratorExt: Iterator + Sized {
    /// Like `take_while`, but **includes** the first element for which the
    /// predicate returns `false`.
    ///
    /// Subsequent elements (after the first failing one) are discarded.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_core::iter_utils::IteratorExt;
    ///
    /// let v: Vec<i32> = (1..=10).take_while_inclusive(|&x| x < 5).collect();
    /// assert_eq!(v, vec![1, 2, 3, 4, 5]);
    /// ```
    fn take_while_inclusive<P: FnMut(&Self::Item) -> bool>(
        self,
        predicate: P,
    ) -> TakeWhileInclusive<Self, P>;
}

impl<I: Iterator> IteratorExt for I {
    fn take_while_inclusive<P: FnMut(&Self::Item) -> bool>(
        self,
        predicate: P,
    ) -> TakeWhileInclusive<Self, P> {
        TakeWhileInclusive::new(self, predicate)
    }
}

/// Iterator returned by [`IteratorExt::take_while_inclusive`].
pub struct TakeWhileInclusive<I, P>
where
    I: Iterator,
    P: FnMut(&I::Item) -> bool,
{
    iter: I,
    predicate: P,
    done: bool,
}

impl<I, P> TakeWhileInclusive<I, P>
where
    I: Iterator,
    P: FnMut(&I::Item) -> bool,
{
    fn new(iter: I, predicate: P) -> Self {
        Self {
            iter,
            predicate,
            done: false,
        }
    }
}

impl<I, P> Iterator for TakeWhileInclusive<I, P>
where
    I: Iterator,
    P: FnMut(&I::Item) -> bool,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let item = self.iter.next()?;
        if !(self.predicate)(&item) {
            self.done = true;
        }
        Some(item)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- WindowedIterator ---

    #[test]
    fn test_windowed_basic() {
        let v: Vec<i32> = (1..=5).collect();
        let windows: Vec<Vec<i32>> = WindowedIterator::new(v.into_iter(), 3, 1).collect();
        assert_eq!(
            windows,
            vec![vec![1, 2, 3], vec![2, 3, 4], vec![3, 4, 5]]
        );
    }

    #[test]
    fn test_windowed_step_2() {
        let v: Vec<i32> = (1..=6).collect();
        let windows: Vec<Vec<i32>> = WindowedIterator::new(v.into_iter(), 3, 2).collect();
        assert_eq!(windows, vec![vec![1, 2, 3], vec![3, 4, 5]]);
    }

    #[test]
    fn test_windowed_empty() {
        let v: Vec<i32> = vec![];
        let windows: Vec<Vec<i32>> = WindowedIterator::new(v.into_iter(), 3, 1).collect();
        assert!(windows.is_empty());
    }

    #[test]
    fn test_windowed_not_enough_elements() {
        let v = vec![1i32, 2];
        let windows: Vec<Vec<i32>> = WindowedIterator::new(v.into_iter(), 3, 1).collect();
        assert!(windows.is_empty());
    }

    // --- ChunkedParallelIterator ---

    #[test]
    fn test_chunked_sequential() {
        let data: Vec<i32> = (1..=9).collect();
        let mut results: Vec<i32> = Vec::new();
        for chunk_result in
            ChunkedParallelIterator::new(data.as_slice(), 3, false, |chunk| {
                chunk.iter().map(|&x| x * 2).collect()
            })
        {
            results.extend(chunk_result);
        }
        assert_eq!(results, vec![2, 4, 6, 8, 10, 12, 14, 16, 18]);
    }

    #[test]
    fn test_chunked_parallel() {
        let data: Vec<i32> = (1..=6).collect();
        let mut results: Vec<i32> = Vec::new();
        for chunk_result in
            ChunkedParallelIterator::new(data.as_slice(), 2, true, |chunk| {
                chunk.iter().map(|&x| x + 10).collect()
            })
        {
            results.extend(chunk_result);
        }
        assert_eq!(results, vec![11, 12, 13, 14, 15, 16]);
    }

    #[test]
    fn test_chunked_uneven() {
        let data: Vec<i32> = (1..=7).collect();
        let mut results: Vec<i32> = Vec::new();
        for chunk_result in
            ChunkedParallelIterator::new(data.as_slice(), 3, false, |chunk| {
                vec![chunk.iter().sum::<i32>()]
            })
        {
            results.extend(chunk_result);
        }
        // Sums of [1,2,3], [4,5,6], [7]
        assert_eq!(results, vec![6, 15, 7]);
    }

    // --- ZipLongest ---

    #[test]
    fn test_zip_longest_equal() {
        let a = vec![1i32, 2, 3];
        let b = vec![10i32, 20, 30];
        let zipped: Vec<_> = ZipLongest::new(a.into_iter(), b.into_iter()).collect();
        assert_eq!(
            zipped,
            vec![
                EitherOrBoth::Both(1, 10),
                EitherOrBoth::Both(2, 20),
                EitherOrBoth::Both(3, 30),
            ]
        );
    }

    #[test]
    fn test_zip_longest_left_longer() {
        let a = vec![1i32, 2, 3];
        let b = vec![10i32];
        let zipped: Vec<_> = ZipLongest::new(a.into_iter(), b.into_iter()).collect();
        assert_eq!(
            zipped,
            vec![
                EitherOrBoth::Both(1, 10),
                EitherOrBoth::Left(2),
                EitherOrBoth::Left(3),
            ]
        );
    }

    #[test]
    fn test_zip_longest_right_longer() {
        let a: Vec<i32> = vec![1];
        let b = vec![10i32, 20, 30];
        let zipped: Vec<_> = ZipLongest::new(a.into_iter(), b.into_iter()).collect();
        assert_eq!(
            zipped,
            vec![
                EitherOrBoth::Both(1, 10),
                EitherOrBoth::Right(20),
                EitherOrBoth::Right(30),
            ]
        );
    }

    #[test]
    fn test_zip_longest_empty() {
        let a: Vec<i32> = vec![];
        let b: Vec<i32> = vec![];
        let zipped: Vec<_> = ZipLongest::new(a.into_iter(), b.into_iter()).collect();
        assert!(zipped.is_empty());
    }

    // --- GroupBy ---

    #[test]
    fn test_group_by_consecutive() {
        let data = vec![1i32, 1, 2, 2, 2, 1];
        let groups: Vec<_> = GroupBy::new(data.into_iter(), |&x| x).collect();
        assert_eq!(
            groups,
            vec![(1, vec![1, 1]), (2, vec![2, 2, 2]), (1, vec![1]),]
        );
    }

    #[test]
    fn test_group_by_all_same() {
        let data = vec![5i32; 4];
        let groups: Vec<_> = GroupBy::new(data.into_iter(), |&x| x).collect();
        assert_eq!(groups, vec![(5, vec![5, 5, 5, 5])]);
    }

    #[test]
    fn test_group_by_all_different() {
        let data = vec![1i32, 2, 3];
        let groups: Vec<_> = GroupBy::new(data.into_iter(), |&x| x).collect();
        assert_eq!(
            groups,
            vec![(1, vec![1]), (2, vec![2]), (3, vec![3]),]
        );
    }

    #[test]
    fn test_group_by_empty() {
        let data: Vec<i32> = vec![];
        let groups: Vec<_> = GroupBy::new(data.into_iter(), |&x| x).collect();
        assert!(groups.is_empty());
    }

    // --- FlatScan ---

    #[test]
    fn test_flat_scan_running_product() {
        let data = vec![2i64, 3, 4];
        let result: Vec<i64> = FlatScan::new(data.into_iter(), 1i64, |state, x| {
            *state *= x;
            vec![*state, x]
        })
        .collect();
        assert_eq!(result, vec![2, 2, 6, 3, 24, 4]);
    }

    #[test]
    fn test_flat_scan_empty_output() {
        // f returns empty vec for odd elements, [x*10] for even
        let data = vec![1i32, 2, 3, 4];
        let result: Vec<i32> = FlatScan::new(data.into_iter(), (), |_state, x| {
            if x % 2 == 0 {
                vec![x * 10]
            } else {
                vec![]
            }
        })
        .collect();
        assert_eq!(result, vec![20, 40]);
    }

    #[test]
    fn test_flat_scan_empty_input() {
        let data: Vec<i32> = vec![];
        let result: Vec<i32> =
            FlatScan::new(data.into_iter(), 0i32, |_, x| vec![x]).collect();
        assert!(result.is_empty());
    }

    // --- TakeWhileInclusive ---

    #[test]
    fn test_take_while_inclusive_basic() {
        let v: Vec<i32> = (1..=10).take_while_inclusive(|&x| x < 5).collect();
        assert_eq!(v, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_take_while_inclusive_all_pass() {
        let v: Vec<i32> = (1..=5).take_while_inclusive(|&x| x < 100).collect();
        assert_eq!(v, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_take_while_inclusive_first_fails() {
        let v: Vec<i32> = (1..=5).take_while_inclusive(|&x| x < 1).collect();
        assert_eq!(v, vec![1]);
    }

    #[test]
    fn test_take_while_inclusive_empty() {
        let v: Vec<i32> = std::iter::empty::<i32>()
            .take_while_inclusive(|_| true)
            .collect();
        assert!(v.is_empty());
    }
}
