//! Lock-free MPMC bounded queue using a ring buffer with atomic operations.
//!
//! This module implements a high-performance multi-producer, multi-consumer
//! bounded queue that avoids mutexes on the hot path.  Each slot has its own
//! sequence number so producers and consumers can independently claim a slot
//! without blocking one another.
//!
//! # Algorithm
//!
//! Each ring-buffer slot stores:
//! - `sequence: AtomicUsize` — an ever-increasing stamp that encodes the slot
//!   state (empty/ready-to-read).
//! - `value: UnsafeCell<MaybeUninit<T>>` — the payload.
//!
//! A producer:
//!   1. Atomically increments the shared `tail`.
//!   2. Waits (spin) until `slot.sequence == tail` (the slot was last read by
//!      `tail - capacity` ago, so it is now free).
//!   3. Writes the value and sets `slot.sequence = tail + 1` (signals the
//!      consumer that the slot is ready).
//!
//! A consumer mirrors the process using `head`.
//!
//! This is the classic Dmitry Vyukov MPMC queue design.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Cache-line padding to avoid false sharing between hot atomic fields.
#[repr(align(64))]
struct Padded<T>(T);

/// One slot in the ring buffer.
struct Slot<T> {
    sequence: AtomicUsize,
    value: UnsafeCell<MaybeUninit<T>>,
}

impl<T> Slot<T> {
    fn new(seq: usize) -> Self {
        Slot {
            sequence: AtomicUsize::new(seq),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }
}

// SAFETY: `Slot` is only accessed through carefully sequenced atomic ops.
unsafe impl<T: Send> Send for Slot<T> {}
unsafe impl<T: Send> Sync for Slot<T> {}

/// A bounded, lock-free multi-producer / multi-consumer queue.
///
/// `T` must be `Send`.  All operations are wait-free from the caller's point
/// of view: if the queue is full `push` returns `false`; if it is empty `pop`
/// returns `None`.  There is no spinning inside the public API.
///
/// # Example
///
/// ```rust
/// use scirs2_core::concurrent::LockFreeQueue;
///
/// let q: LockFreeQueue<i32> = LockFreeQueue::new(4);
/// assert!(q.push(1));
/// assert!(q.push(2));
/// assert_eq!(q.pop(), Some(1));
/// assert_eq!(q.pop(), Some(2));
/// assert_eq!(q.pop(), None);
/// ```
pub struct LockFreeQueue<T> {
    buffer: Vec<Slot<T>>,
    capacity: usize,
    mask: usize,
    head: Padded<AtomicUsize>,
    tail: Padded<AtomicUsize>,
}

// SAFETY: the internal slots are only mutated while holding the implicit
// sequence-number "lock", so the queue is safe to share across threads.
unsafe impl<T: Send> Send for LockFreeQueue<T> {}
unsafe impl<T: Send> Sync for LockFreeQueue<T> {}

impl<T> LockFreeQueue<T> {
    /// Create a new queue with `capacity` rounded up to the next power of two.
    ///
    /// The minimum capacity is 1; if `capacity` is 0 it is treated as 1.
    pub fn new(capacity: usize) -> Self {
        // The Vyukov MPMC queue requires capacity >= 2 to correctly
        // distinguish "slot free for producer" from "slot has data for consumer."
        // With capacity 1, the sequence-number check becomes ambiguous and the
        // queue deadlocks.
        let cap = capacity.max(2).next_power_of_two();
        let buffer: Vec<Slot<T>> = (0..cap).map(|i| Slot::new(i)).collect();
        LockFreeQueue {
            buffer,
            capacity: cap,
            mask: cap - 1,
            head: Padded(AtomicUsize::new(0)),
            tail: Padded(AtomicUsize::new(0)),
        }
    }

    /// Attempt to push `val` onto the queue.
    ///
    /// Returns `false` without modifying the queue if it is full.
    pub fn push(&self, val: T) -> bool {
        let mut pos = self.tail.0.load(Ordering::Relaxed);
        loop {
            let slot = &self.buffer[pos & self.mask];
            let seq = slot.sequence.load(Ordering::Acquire);
            let diff = seq as isize - pos as isize;
            match diff.cmp(&0) {
                std::cmp::Ordering::Equal => {
                    // Slot is free — try to claim it.
                    match self.tail.0.compare_exchange_weak(
                        pos,
                        pos.wrapping_add(1),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            // We own the slot; write value and publish.
                            unsafe {
                                (*slot.value.get()).write(val);
                            }
                            slot.sequence.store(pos.wrapping_add(1), Ordering::Release);
                            return true;
                        }
                        Err(updated) => {
                            pos = updated;
                        }
                    }
                }
                std::cmp::Ordering::Less => {
                    // Queue full.
                    return false;
                }
                std::cmp::Ordering::Greater => {
                    // Another producer moved tail; reload.
                    pos = self.tail.0.load(Ordering::Relaxed);
                }
            }
        }
    }

    /// Attempt to pop a value from the queue.
    ///
    /// Returns `None` if the queue is currently empty.
    pub fn pop(&self) -> Option<T> {
        let mut pos = self.head.0.load(Ordering::Relaxed);
        loop {
            let slot = &self.buffer[pos & self.mask];
            let seq = slot.sequence.load(Ordering::Acquire);
            let diff = seq as isize - pos.wrapping_add(1) as isize;
            match diff.cmp(&0) {
                std::cmp::Ordering::Equal => {
                    // Slot has data — try to claim it.
                    match self.head.0.compare_exchange_weak(
                        pos,
                        pos.wrapping_add(1),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            // We own the slot; read value and free.
                            let val = unsafe { (*slot.value.get()).assume_init_read() };
                            slot.sequence
                                .store(pos.wrapping_add(self.capacity), Ordering::Release);
                            return Some(val);
                        }
                        Err(updated) => {
                            pos = updated;
                        }
                    }
                }
                std::cmp::Ordering::Less => {
                    // Queue empty.
                    return None;
                }
                std::cmp::Ordering::Greater => {
                    pos = self.head.0.load(Ordering::Relaxed);
                }
            }
        }
    }

    /// Return the number of items currently in the queue (approximate under
    /// concurrent access).
    pub fn len(&self) -> usize {
        let tail = self.tail.0.load(Ordering::Relaxed);
        let head = self.head.0.load(Ordering::Relaxed);
        tail.saturating_sub(head)
    }

    /// Return `true` if the queue appears empty (approximate).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the maximum number of items this queue can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T> Drop for LockFreeQueue<T> {
    fn drop(&mut self) {
        // Drain remaining items so their destructors run.
        while self.pop().is_some() {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_basic_push_pop() {
        let q: LockFreeQueue<u32> = LockFreeQueue::new(4);
        assert!(q.is_empty());
        assert!(q.push(10));
        assert!(q.push(20));
        assert_eq!(q.len(), 2);
        assert_eq!(q.pop(), Some(10));
        assert_eq!(q.pop(), Some(20));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn test_capacity_limit() {
        let q: LockFreeQueue<i32> = LockFreeQueue::new(4);
        // capacity rounds up to next power-of-two = 4
        assert_eq!(q.capacity(), 4);
        assert!(q.push(1));
        assert!(q.push(2));
        assert!(q.push(3));
        assert!(q.push(4));
        // Fifth push must fail.
        assert!(!q.push(5));
        // After a pop, one push should succeed.
        assert_eq!(q.pop(), Some(1));
        assert!(q.push(5));
    }

    #[test]
    fn test_fifo_order() {
        let q: LockFreeQueue<usize> = LockFreeQueue::new(16);
        for i in 0..10 {
            assert!(q.push(i));
        }
        for i in 0..10 {
            assert_eq!(q.pop(), Some(i));
        }
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn test_concurrent_mpmc() {
        const PRODUCERS: usize = 4;
        const ITEMS_PER_PRODUCER: usize = 1_000;
        const CAPACITY: usize = 256;

        let q = Arc::new(LockFreeQueue::<usize>::new(CAPACITY));
        let total = PRODUCERS * ITEMS_PER_PRODUCER;

        // Spawn producers.
        let handles: Vec<_> = (0..PRODUCERS)
            .map(|_p| {
                let q2 = Arc::clone(&q);
                thread::spawn(move || {
                    let mut sent = 0usize;
                    while sent < ITEMS_PER_PRODUCER {
                        if q2.push(1) {
                            sent += 1;
                        } else {
                            thread::yield_now();
                        }
                    }
                })
            })
            .collect();

        // Consumer on main thread.
        let mut received = 0usize;
        while received < total {
            if let Some(_) = q.pop() {
                received += 1;
            } else {
                thread::yield_now();
            }
        }

        for h in handles {
            h.join().expect("producer thread panicked");
        }

        assert_eq!(received, total);
        assert!(q.is_empty());
    }

    #[test]
    fn test_drop_runs_destructors() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;

        let counter = Arc::new(AtomicUsize::new(0));

        struct Tracker(Arc<AtomicUsize>);
        impl Drop for Tracker {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        {
            let q: LockFreeQueue<Tracker> = LockFreeQueue::new(8);
            q.push(Tracker(Arc::clone(&counter)));
            q.push(Tracker(Arc::clone(&counter)));
            // Queue is dropped here with 2 items inside.
        }

        assert_eq!(counter.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_zero_capacity_becomes_one() {
        let q: LockFreeQueue<u8> = LockFreeQueue::new(0);
        // Minimum capacity is 2 (Vyukov MPMC requires >= 2 to avoid
        // sequence-number ambiguity); next_power_of_two(2) == 2.
        assert_eq!(q.capacity(), 2);
        assert!(q.push(42));
        assert!(q.push(43));
        // Now full — third push must fail.
        assert!(!q.push(44));
        assert_eq!(q.pop(), Some(42));
        assert_eq!(q.pop(), Some(43));
        assert_eq!(q.pop(), None);
    }
}
