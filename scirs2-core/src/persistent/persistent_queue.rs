//! Persistent functional queue with amortized O(1) operations.
//!
//! This module implements a purely functional persistent queue using the
//! classic **two-list** technique:
//!
//! - `front`: a `Vec<T>` in FIFO order (`front[0]` is the queue head).
//! - `rear`:  a `Vec<T>` in push order (`rear[0]` is the oldest element in
//!   the rear portion).
//!
//! `push_back` appends to `rear` (O(1)).  `pop_front` removes `front[0]`
//! (O(1)); when `front` becomes empty we swap `rear` into `front` and clear
//! `rear`.  This is amortized O(1) per element because each element moves
//! from `rear` to `front` exactly once.
//!
//! Because `front` and `rear` are stored as `Arc<Vec<T>>`, all slicing
//! creates new `Arc<Vec<T>>` snapshots.  Old versions of the queue keep
//! their own `Arc` references alive, so they remain fully usable after any
//! number of subsequent operations on derived queues.
//!
//! # Complexity
//! - `push_back` — O(1)
//! - `pop_front` — amortized O(1)
//! - `peek` — O(1)
//! - `len` — O(1)
//! - All operations are fully persistent.

use std::sync::Arc;

/// Persistent functional queue.
///
/// # Examples
///
/// ```
/// use scirs2_core::persistent::persistent_queue::PersistentQueue;
///
/// let q0 = PersistentQueue::new();
/// let q1 = q0.push_back(1u32);
/// let q2 = q1.push_back(2).push_back(3);
///
/// assert_eq!(q2.peek(), Some(&1));
///
/// let (val, q3) = q2.pop_front().expect("should succeed");
/// assert_eq!(val, 1);
/// assert_eq!(q3.peek(), Some(&2));
///
/// // q2 is unaffected by the pop.
/// assert_eq!(q2.len(), 3);
/// assert_eq!(q2.peek(), Some(&1));
/// ```
#[derive(Clone, Debug)]
pub struct PersistentQueue<T: Clone> {
    /// Elements available for `pop_front`, in FIFO order.
    /// `front[0]` is the queue head.
    front: Arc<Vec<T>>,
    /// Index into `front` — elements before this index have been logically
    /// removed.  Using an offset avoids copying the entire front vector on
    /// each pop.
    front_start: usize,
    /// Elements added since the last rotation, in push order.
    /// `rear[0]` is the oldest element in this portion.
    rear: Arc<Vec<T>>,
    /// Cached total element count.
    len: usize,
}

impl<T: Clone> Default for PersistentQueue<T> {
    fn default() -> Self {
        PersistentQueue::new()
    }
}

impl<T: Clone> PersistentQueue<T> {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create an empty queue.
    pub fn new() -> Self {
        PersistentQueue {
            front: Arc::new(Vec::new()),
            front_start: 0,
            rear: Arc::new(Vec::new()),
            len: 0,
        }
    }

    /// Build a queue from an iterator, preserving insertion order.
    pub fn from_iter(iter: impl IntoIterator<Item = T>) -> Self {
        let items: Vec<T> = iter.into_iter().collect();
        let len = items.len();
        PersistentQueue {
            front: Arc::new(items),
            front_start: 0,
            rear: Arc::new(Vec::new()),
            len,
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the queue contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Peek at the front element without removing it.
    ///
    /// Returns `None` if the queue is empty.
    pub fn peek(&self) -> Option<&T> {
        // Active front elements.
        let front_slice = &self.front[self.front_start..];
        if !front_slice.is_empty() {
            return Some(&front_slice[0]);
        }
        // Front is exhausted: look at the oldest element in rear (index 0).
        self.rear.first()
    }

    // -----------------------------------------------------------------------
    // Operations
    // -----------------------------------------------------------------------

    /// Return a new queue with `item` appended at the back.
    ///
    /// O(1).
    pub fn push_back(&self, item: T) -> Self {
        let mut new_rear = (*self.rear).clone();
        new_rear.push(item);
        PersistentQueue {
            front: self.front.clone(),
            front_start: self.front_start,
            rear: Arc::new(new_rear),
            len: self.len + 1,
        }
    }

    /// Return `Some((front_element, new_queue))`, or `None` if the queue is
    /// empty.
    ///
    /// Amortized O(1).
    pub fn pop_front(&self) -> Option<(T, Self)> {
        if self.len == 0 {
            return None;
        }

        let front_slice = &self.front[self.front_start..];

        if !front_slice.is_empty() {
            // Pop from the active portion of front.
            let head = front_slice[0].clone();
            let new_start = self.front_start + 1;
            let remaining_front = self.front.len() - new_start;

            let q = if remaining_front == 0 {
                // front exhausted: rotate rear → front.
                if self.rear.is_empty() {
                    PersistentQueue {
                        front: Arc::new(Vec::new()),
                        front_start: 0,
                        rear: Arc::new(Vec::new()),
                        len: 0,
                    }
                } else {
                    PersistentQueue {
                        front: self.rear.clone(),
                        front_start: 0,
                        rear: Arc::new(Vec::new()),
                        len: self.len - 1,
                    }
                }
            } else {
                PersistentQueue {
                    front: self.front.clone(),
                    front_start: new_start,
                    rear: self.rear.clone(),
                    len: self.len - 1,
                }
            };

            return Some((head, q));
        }

        // front_slice is empty but len > 0 → there must be elements in rear.
        // Rotate rear → front, then pop.
        debug_assert!(!self.rear.is_empty());
        let rotated = PersistentQueue {
            front: self.rear.clone(),
            front_start: 0,
            rear: Arc::new(Vec::new()),
            len: self.len,
        };
        rotated.pop_front()
    }

    // -----------------------------------------------------------------------
    // Iteration
    // -----------------------------------------------------------------------

    /// Iterate over elements in FIFO order, yielding cloned values.
    pub fn iter(&self) -> QueueIter<T> {
        // Active front elements.
        let front_slice = &self.front[self.front_start..];
        let mut items: Vec<T> = front_slice.to_vec();
        // Rear elements are in push order (oldest first).
        items.extend(self.rear.iter().cloned());
        QueueIter { items, index: 0 }
    }

    /// Consuming iterator via repeated `pop_front`.
    pub fn into_pop_iter(self) -> PopIterator<T> {
        PopIterator { queue: self }
    }
}

// ---------------------------------------------------------------------------
// Iterators
// ---------------------------------------------------------------------------

/// FIFO iterator that yields cloned elements.
pub struct QueueIter<T: Clone> {
    items: Vec<T>,
    index: usize,
}

impl<T: Clone> Iterator for QueueIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.items.len() {
            let val = self.items[self.index].clone();
            self.index += 1;
            Some(val)
        } else {
            None
        }
    }
}

/// An iterator that repeatedly calls `pop_front`, consuming the queue.
pub struct PopIterator<T: Clone> {
    queue: PersistentQueue<T>,
}

impl<T: Clone> Iterator for PopIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let (val, new_queue) = self.queue.pop_front()?;
        self.queue = new_queue;
        Some(val)
    }
}

impl<T: Clone> IntoIterator for PersistentQueue<T> {
    type Item = T;
    type IntoIter = QueueIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_queue() {
        let q: PersistentQueue<i32> = PersistentQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        assert_eq!(q.peek(), None);
        assert!(q.pop_front().is_none());
    }

    #[test]
    fn test_push_and_pop_single() {
        let q = PersistentQueue::new().push_back(42i32);
        assert_eq!(q.len(), 1);
        assert_eq!(q.peek(), Some(&42));

        let (val, q2) = q.pop_front().expect("should succeed");
        assert_eq!(val, 42);
        assert!(q2.is_empty());
    }

    #[test]
    fn test_fifo_order() {
        let mut q = PersistentQueue::new();
        for i in 0..10i32 {
            q = q.push_back(i);
        }
        assert_eq!(q.len(), 10);

        for expected in 0..10i32 {
            let (val, new_q) = q.pop_front().expect("should succeed");
            assert_eq!(val, expected);
            q = new_q;
        }
        assert!(q.is_empty());
    }

    #[test]
    fn test_persistence() {
        let q0: PersistentQueue<i32> = PersistentQueue::new();
        let q1 = q0.push_back(1);
        let q2 = q1.push_back(2).push_back(3);

        // q1 still has only one element.
        assert_eq!(q1.len(), 1);
        assert_eq!(q1.peek(), Some(&1));

        // q2 has three.
        assert_eq!(q2.len(), 3);
        assert_eq!(q2.peek(), Some(&1));

        // Pop from q2 does not affect q1.
        let (_, q3) = q2.pop_front().expect("should succeed");
        assert_eq!(q1.len(), 1);
        assert_eq!(q3.len(), 2);

        // q2 is unaffected.
        assert_eq!(q2.len(), 3);
        assert_eq!(q2.peek(), Some(&1));
    }

    #[test]
    fn test_interleaved_push_pop() {
        let q = PersistentQueue::new()
            .push_back(1i32)
            .push_back(2)
            .push_back(3);

        let (v1, q) = q.pop_front().expect("should succeed");
        let q = q.push_back(4).push_back(5);
        let (v2, q) = q.pop_front().expect("should succeed");
        let (v3, q) = q.pop_front().expect("should succeed");
        let (v4, q) = q.pop_front().expect("should succeed");
        let (v5, q) = q.pop_front().expect("should succeed");

        assert_eq!(v1, 1);
        assert_eq!(v2, 2);
        assert_eq!(v3, 3);
        assert_eq!(v4, 4);
        assert_eq!(v5, 5);
        assert!(q.is_empty());
    }

    #[test]
    fn test_from_iter() {
        let q = PersistentQueue::from_iter(0..5i32);
        assert_eq!(q.len(), 5);
        let vals: Vec<i32> = q.into_pop_iter().collect();
        assert_eq!(vals, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_iter_order() {
        let q = PersistentQueue::from_iter(vec!["a", "b", "c", "d"]);
        let vals: Vec<&str> = q.iter().collect();
        assert_eq!(vals, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn test_large_queue_fifo() {
        let n = 1000usize;
        let q = PersistentQueue::from_iter(0..n);
        assert_eq!(q.len(), n);

        let mut current = q;
        for expected in 0..n {
            let (val, new_q) = current.pop_front().expect("should succeed");
            assert_eq!(val, expected);
            current = new_q;
        }
        assert!(current.is_empty());
    }

    #[test]
    fn test_pop_iter_fifo() {
        let q = PersistentQueue::from_iter(10..20usize);
        let vals: Vec<usize> = q.into_pop_iter().collect();
        assert_eq!(vals, (10..20).collect::<Vec<_>>());
    }

    #[test]
    fn test_shared_versions_independent() {
        let base = PersistentQueue::from_iter(1..4i32); // [1, 2, 3]

        // Two different continuations from the same base.
        let branch_a = base.push_back(10).push_back(11);
        let branch_b = base.push_back(20).push_back(21);

        let a_vals: Vec<i32> = branch_a.into_pop_iter().collect();
        let b_vals: Vec<i32> = branch_b.into_pop_iter().collect();

        assert_eq!(a_vals, vec![1, 2, 3, 10, 11]);
        assert_eq!(b_vals, vec![1, 2, 3, 20, 21]);
    }

    #[test]
    fn test_peek_after_pop() {
        let q = PersistentQueue::from_iter(vec![100i32, 200, 300]);
        assert_eq!(q.peek(), Some(&100));

        let (_, q) = q.pop_front().expect("should succeed");
        assert_eq!(q.peek(), Some(&200));

        let (_, q) = q.pop_front().expect("should succeed");
        assert_eq!(q.peek(), Some(&300));

        let (_, q) = q.pop_front().expect("should succeed");
        assert_eq!(q.peek(), None);
    }

    #[test]
    fn test_push_then_multiple_pops_across_rotation() {
        // Push enough elements that we cross a rotation boundary.
        let mut q = PersistentQueue::new();
        for i in 0..20i32 {
            q = q.push_back(i);
        }
        // Pop 10.
        for i in 0..10i32 {
            let (v, nq) = q.pop_front().expect("should succeed");
            assert_eq!(v, i);
            q = nq;
        }
        // Push 10 more.
        for i in 20..30i32 {
            q = q.push_back(i);
        }
        // Pop remaining 20.
        for i in 10..30i32 {
            let (v, nq) = q.pop_front().expect("should succeed");
            assert_eq!(v, i);
            q = nq;
        }
        assert!(q.is_empty());
    }

    #[test]
    fn test_pure_push_back_peek() {
        // Verify that push_back into an empty queue makes peek() work.
        let q = PersistentQueue::<i32>::new();
        assert_eq!(q.peek(), None);
        let q = q.push_back(7);
        assert_eq!(q.peek(), Some(&7));
        let q = q.push_back(8);
        assert_eq!(q.peek(), Some(&7)); // still the head
    }
}
