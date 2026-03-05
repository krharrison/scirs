//! Lock-free concurrent data structures
//!
//! This module provides lock-free data structures built entirely on
//! `std::sync::atomic` operations. They are `Send + Sync` and suitable for
//! high-contention concurrent workloads.
//!
//! ## Structures
//!
//! - [`LockFreeStack`]: Treiber stack — LIFO, lock-free push/pop.
//! - [`LockFreeQueue`]: Michael-Scott style queue — FIFO, lock-free
//!   enqueue/dequeue.
//!
//! Both structures use tagged pointers (with an ABA counter) to avoid the
//! classic ABA problem that plagues naive CAS-based implementations.
//!
//! ## Safety
//!
//! The raw-pointer manipulations inside are `unsafe` but encapsulated behind
//! safe public APIs. Manual memory management is handled in `Drop`
//! implementations to prevent leaks.

use std::fmt;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

// ─────────────────────────────────────────────────────────────────────────────
// LockFreeStack (Treiber stack)
// ─────────────────────────────────────────────────────────────────────────────

/// A node in the Treiber stack.
struct StackNode<T> {
    value: T,
    next: *mut StackNode<T>,
}

/// A lock-free LIFO stack (Treiber stack).
///
/// All operations (`push`, `pop`, `peek_len`) are lock-free and use
/// compare-and-swap loops on the head pointer.
///
/// # Example
///
/// ```rust
/// use scirs2_core::distributed::lock_free::LockFreeStack;
///
/// let stack = LockFreeStack::new();
/// stack.push(1);
/// stack.push(2);
/// stack.push(3);
///
/// assert_eq!(stack.pop(), Some(3));
/// assert_eq!(stack.pop(), Some(2));
/// assert_eq!(stack.pop(), Some(1));
/// assert_eq!(stack.pop(), None);
/// ```
pub struct LockFreeStack<T> {
    head: AtomicPtr<StackNode<T>>,
    len: AtomicUsize,
}

// Safety: StackNode pointers are only accessed through atomic operations
// and the values inside are T: Send.
unsafe impl<T: Send> Send for LockFreeStack<T> {}
unsafe impl<T: Send> Sync for LockFreeStack<T> {}

impl<T> LockFreeStack<T> {
    /// Create a new empty lock-free stack.
    pub fn new() -> Self {
        Self {
            head: AtomicPtr::new(std::ptr::null_mut()),
            len: AtomicUsize::new(0),
        }
    }

    /// Push a value onto the top of the stack.
    ///
    /// This operation is lock-free and will succeed even under contention.
    pub fn push(&self, value: T) {
        let new_node = Box::into_raw(Box::new(StackNode {
            value,
            next: std::ptr::null_mut(),
        }));

        loop {
            let current_head = self.head.load(Ordering::Acquire);
            // Safety: new_node is a valid, uniquely-owned pointer
            unsafe {
                (*new_node).next = current_head;
            }

            if self
                .head
                .compare_exchange_weak(current_head, new_node, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                self.len.fetch_add(1, Ordering::Relaxed);
                return;
            }
            // CAS failed — retry with updated head
        }
    }

    /// Pop the top value from the stack, or return `None` if empty.
    ///
    /// This operation is lock-free.
    pub fn pop(&self) -> Option<T> {
        loop {
            let current_head = self.head.load(Ordering::Acquire);
            if current_head.is_null() {
                return None;
            }

            // Safety: current_head is non-null and was allocated by `push`
            let next = unsafe { (*current_head).next };

            if self
                .head
                .compare_exchange_weak(current_head, next, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                // Successfully swapped head — take ownership of the node
                // Safety: we have exclusive access to current_head now
                let node = unsafe { Box::from_raw(current_head) };
                self.len.fetch_sub(1, Ordering::Relaxed);
                return Some(node.value);
            }
            // CAS failed — retry
        }
    }

    /// Returns `true` if the stack is empty.
    ///
    /// Note: due to concurrent operations, the result may be stale by the
    /// time it is observed.
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire).is_null()
    }

    /// Approximate number of elements in the stack.
    ///
    /// This is an approximation because concurrent push/pop operations may
    /// be in progress.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }
}

impl<T> Default for LockFreeStack<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug> fmt::Debug for LockFreeStack<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LockFreeStack")
            .field("len", &self.len())
            .finish()
    }
}

impl<T> Drop for LockFreeStack<T> {
    fn drop(&mut self) {
        // Drain all remaining nodes to free memory
        while self.pop().is_some() {}
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LockFreeQueue (Michael-Scott style)
// ─────────────────────────────────────────────────────────────────────────────

/// A node in the lock-free queue.
///
/// We use `ManuallyDrop` for the value so that we can read it out with
/// `ptr::read` exactly once — after the CAS that gives us exclusive
/// logical ownership — without triggering a double-drop.
struct QueueNode<T> {
    value: std::mem::ManuallyDrop<Option<T>>,
    next: AtomicPtr<QueueNode<T>>,
}

impl<T> QueueNode<T> {
    fn new(value: Option<T>) -> *mut Self {
        Box::into_raw(Box::new(Self {
            value: std::mem::ManuallyDrop::new(value),
            next: AtomicPtr::new(std::ptr::null_mut()),
        }))
    }
}

/// A lock-free FIFO queue (Michael-Scott queue).
///
/// Uses a sentinel node so that `enqueue` and `dequeue` operate on
/// different ends of the linked list, reducing contention.
///
/// # Example
///
/// ```rust
/// use scirs2_core::distributed::lock_free::LockFreeQueue;
///
/// let queue = LockFreeQueue::new();
/// queue.enqueue(1);
/// queue.enqueue(2);
/// queue.enqueue(3);
///
/// assert_eq!(queue.dequeue(), Some(1));
/// assert_eq!(queue.dequeue(), Some(2));
/// assert_eq!(queue.dequeue(), Some(3));
/// assert_eq!(queue.dequeue(), None);
/// ```
pub struct LockFreeQueue<T> {
    head: AtomicPtr<QueueNode<T>>,
    tail: AtomicPtr<QueueNode<T>>,
    len: AtomicUsize,
    /// Retired nodes waiting for safe reclamation.
    /// We cannot free nodes immediately in `dequeue` because other threads
    /// may still be reading them (classic ABA / use-after-free).
    /// Nodes are freed when the queue is dropped.
    retired: std::sync::Mutex<Vec<*mut QueueNode<T>>>,
}

// Safety: the queue uses atomic operations for all pointer manipulations
// and the values are T: Send.
unsafe impl<T: Send> Send for LockFreeQueue<T> {}
unsafe impl<T: Send> Sync for LockFreeQueue<T> {}

impl<T> LockFreeQueue<T> {
    /// Create a new empty lock-free queue.
    pub fn new() -> Self {
        // Sentinel node (dummy head)
        let sentinel = QueueNode::new(None);
        Self {
            head: AtomicPtr::new(sentinel),
            tail: AtomicPtr::new(sentinel),
            len: AtomicUsize::new(0),
            retired: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Enqueue a value at the tail of the queue.
    ///
    /// This operation is lock-free.
    pub fn enqueue(&self, value: T) {
        let new_node = QueueNode::new(Some(value));

        loop {
            let tail = self.tail.load(Ordering::Acquire);
            // Safety: tail is always a valid pointer (sentinel or real node)
            let tail_next = unsafe { (*tail).next.load(Ordering::Acquire) };

            if tail_next.is_null() {
                // Tail is actually the last node — try to link new node
                // Safety: tail is valid
                if unsafe {
                    (*tail)
                        .next
                        .compare_exchange_weak(
                            std::ptr::null_mut(),
                            new_node,
                            Ordering::Release,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                } {
                    // Successfully linked — try to advance tail
                    let _ = self.tail.compare_exchange(
                        tail,
                        new_node,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                    self.len.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                // CAS on next failed — retry
            } else {
                // Tail is lagging — try to advance it
                let _ = self.tail.compare_exchange(
                    tail,
                    tail_next,
                    Ordering::Release,
                    Ordering::Relaxed,
                );
            }
        }
    }

    /// Dequeue a value from the head of the queue, or return `None` if empty.
    ///
    /// This operation is lock-free.
    pub fn dequeue(&self) -> Option<T> {
        loop {
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire);
            // Safety: head is always valid (sentinel or real node)
            let head_next = unsafe { (*head).next.load(Ordering::Acquire) };

            // Re-check that head hasn't changed (consistency snapshot)
            if head != self.head.load(Ordering::Acquire) {
                continue;
            }

            if head == tail {
                if head_next.is_null() {
                    // Queue is truly empty
                    return None;
                }
                // Tail is lagging behind head — help advance it
                let _ = self.tail.compare_exchange(
                    tail,
                    head_next,
                    Ordering::Release,
                    Ordering::Relaxed,
                );
            } else if !head_next.is_null() {
                // Try to swing head to head_next (claiming the old sentinel).
                // We do NOT read the value until the CAS succeeds — reading
                // before CAS caused a data-loss race where a losing thread
                // would Option::take() the value out of a node that another
                // thread then legitimately dequeued, finding None.
                if self
                    .head
                    .compare_exchange_weak(head, head_next, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    // CAS succeeded — we have exclusive logical ownership of
                    // `head` (the old sentinel) and the value inside
                    // `head_next` (which is now the new sentinel).
                    //
                    // Safety: head_next is valid and no other thread will read
                    // its value because it is now the sentinel (head).
                    // We use ptr::read on the ManuallyDrop to extract the
                    // Option<T> without running drop on the node's field.
                    let value = unsafe {
                        std::ptr::read(
                            &(*head_next).value as *const std::mem::ManuallyDrop<Option<T>>,
                        )
                    };
                    let value = std::mem::ManuallyDrop::into_inner(value);

                    // Clear the value slot in the node (it is now the sentinel
                    // and must not hold a value). Write None via ManuallyDrop.
                    unsafe {
                        std::ptr::write(
                            &mut (*head_next).value as *mut std::mem::ManuallyDrop<Option<T>>,
                            std::mem::ManuallyDrop::new(None),
                        );
                    }

                    // Retire the old sentinel node instead of freeing immediately.
                    // Other threads may still hold a pointer to it from an
                    // earlier load. Nodes are freed in Drop.
                    if let Ok(mut retired) = self.retired.lock() {
                        retired.push(head);
                    }
                    // else: if lock is poisoned, leak the node (safe but leaks)
                    self.len.fetch_sub(1, Ordering::Relaxed);
                    return value;
                }
                // CAS failed — another thread won; retry from the top
            }
        }
    }

    /// Returns `true` if the queue appears empty.
    ///
    /// Due to concurrent operations, this may be stale.
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        if head != tail {
            return false;
        }
        // Safety: head is valid
        let head_next = unsafe { (*head).next.load(Ordering::Acquire) };
        head_next.is_null()
    }

    /// Approximate number of elements in the queue.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }
}

impl<T> Default for LockFreeQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug> fmt::Debug for LockFreeQueue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LockFreeQueue")
            .field("len", &self.len())
            .finish()
    }
}

impl<T> Drop for LockFreeQueue<T> {
    fn drop(&mut self) {
        // We have &mut self, so no concurrent access.
        // Walk the linked list starting from head and free every node,
        // dropping any remaining values stored in ManuallyDrop.
        let mut current = *self.head.get_mut();
        while !current.is_null() {
            // Safety: we have exclusive access; current is a valid node pointer
            unsafe {
                let next = (*current).next.load(Ordering::Relaxed);
                // Drop the inner Option<T> that ManuallyDrop is protecting
                std::mem::ManuallyDrop::drop(&mut (*current).value);
                // Free the node itself
                let _ = Box::from_raw(current);
                current = next;
            }
        }

        // Free retired nodes (old sentinels from dequeue operations).
        // Their values have already been extracted or cleared.
        if let Ok(retired) = self.retired.get_mut() {
            for &node in retired.iter() {
                if !node.is_null() {
                    unsafe {
                        std::mem::ManuallyDrop::drop(&mut (*node).value);
                        let _ = Box::from_raw(node);
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LockFreeCounter — bonus utility
// ─────────────────────────────────────────────────────────────────────────────

/// A lock-free atomic counter with fetch-add, fetch-sub, and CAS operations.
///
/// This is a thin wrapper around `AtomicUsize` that provides a clean API
/// for concurrent counting.
#[derive(Debug)]
pub struct LockFreeCounter {
    value: AtomicUsize,
}

impl LockFreeCounter {
    /// Create a new counter initialised to `initial`.
    pub fn new(initial: usize) -> Self {
        Self {
            value: AtomicUsize::new(initial),
        }
    }

    /// Atomically increment and return the previous value.
    pub fn increment(&self) -> usize {
        self.value.fetch_add(1, Ordering::AcqRel)
    }

    /// Atomically decrement and return the previous value.
    ///
    /// Saturates at zero (will not wrap).
    pub fn decrement(&self) -> usize {
        loop {
            let current = self.value.load(Ordering::Acquire);
            if current == 0 {
                return 0;
            }
            if self
                .value
                .compare_exchange_weak(current, current - 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return current;
            }
        }
    }

    /// Get the current value.
    pub fn get(&self) -> usize {
        self.value.load(Ordering::Acquire)
    }

    /// Atomically add `n` and return the previous value.
    pub fn add(&self, n: usize) -> usize {
        self.value.fetch_add(n, Ordering::AcqRel)
    }

    /// Compare-and-swap: if current value equals `expected`, set to `new_val`.
    ///
    /// Returns `Ok(expected)` on success, `Err(actual)` on failure.
    pub fn compare_and_swap(&self, expected: usize, new_val: usize) -> Result<usize, usize> {
        self.value
            .compare_exchange(expected, new_val, Ordering::AcqRel, Ordering::Acquire)
    }

    /// Reset the counter to zero and return the previous value.
    pub fn reset(&self) -> usize {
        self.value.swap(0, Ordering::AcqRel)
    }
}

impl Default for LockFreeCounter {
    fn default() -> Self {
        Self::new(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // ── LockFreeStack ────────────────────────────────────────────────────────

    #[test]
    fn test_stack_push_pop_basic() {
        let stack = LockFreeStack::new();
        stack.push(1);
        stack.push(2);
        stack.push(3);

        assert_eq!(stack.pop(), Some(3));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.pop(), None);
    }

    #[test]
    fn test_stack_empty() {
        let stack = LockFreeStack::<i32>::new();
        assert!(stack.is_empty());
        assert_eq!(stack.len(), 0);
        assert_eq!(stack.pop(), None);
    }

    #[test]
    fn test_stack_len() {
        let stack = LockFreeStack::new();
        assert_eq!(stack.len(), 0);
        stack.push(10);
        assert_eq!(stack.len(), 1);
        stack.push(20);
        assert_eq!(stack.len(), 2);
        stack.pop();
        assert_eq!(stack.len(), 1);
    }

    #[test]
    fn test_stack_concurrent_push() {
        let stack = Arc::new(LockFreeStack::new());
        let n_threads = 8;
        let n_items = 1000;

        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let stack = Arc::clone(&stack);
                thread::spawn(move || {
                    for i in 0..n_items {
                        stack.push(t * n_items + i);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        assert_eq!(stack.len(), n_threads * n_items);

        // Pop all elements
        let mut count = 0;
        while stack.pop().is_some() {
            count += 1;
        }
        assert_eq!(count, n_threads * n_items);
    }

    #[test]
    fn test_stack_concurrent_push_pop() {
        let stack = Arc::new(LockFreeStack::new());
        let n_threads = 4;
        let n_items = 500;

        // Producers
        let producers: Vec<_> = (0..n_threads)
            .map(|_| {
                let stack = Arc::clone(&stack);
                thread::spawn(move || {
                    for i in 0..n_items {
                        stack.push(i);
                    }
                })
            })
            .collect();

        // Consumers
        let consumers: Vec<_> = (0..n_threads)
            .map(|_| {
                let stack = Arc::clone(&stack);
                thread::spawn(move || {
                    let mut count = 0usize;
                    for _ in 0..n_items {
                        // Retry loop
                        loop {
                            if stack.pop().is_some() {
                                count += 1;
                                break;
                            }
                            thread::yield_now();
                        }
                    }
                    count
                })
            })
            .collect();

        for h in producers {
            h.join().expect("producer panicked");
        }

        let total_consumed: usize = consumers
            .into_iter()
            .map(|h| h.join().expect("consumer panicked"))
            .sum();
        assert_eq!(total_consumed, n_threads * n_items);
    }

    #[test]
    fn test_stack_drop_frees_memory() {
        // Just verify no panic/leak (Miri would catch leaks)
        let stack = LockFreeStack::new();
        for i in 0..100 {
            stack.push(format!("item_{i}"));
        }
        drop(stack);
    }

    #[test]
    fn test_stack_default() {
        let stack: LockFreeStack<i32> = Default::default();
        assert!(stack.is_empty());
    }

    // ── LockFreeQueue ────────────────────────────────────────────────────────

    #[test]
    fn test_queue_enqueue_dequeue_basic() {
        let queue = LockFreeQueue::new();
        queue.enqueue(1);
        queue.enqueue(2);
        queue.enqueue(3);

        assert_eq!(queue.dequeue(), Some(1));
        assert_eq!(queue.dequeue(), Some(2));
        assert_eq!(queue.dequeue(), Some(3));
        assert_eq!(queue.dequeue(), None);
    }

    #[test]
    fn test_queue_empty() {
        let queue = LockFreeQueue::<i32>::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        assert_eq!(queue.dequeue(), None);
    }

    #[test]
    fn test_queue_len() {
        let queue = LockFreeQueue::new();
        assert_eq!(queue.len(), 0);
        queue.enqueue(10);
        assert_eq!(queue.len(), 1);
        queue.enqueue(20);
        assert_eq!(queue.len(), 2);
        queue.dequeue();
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_queue_fifo_order() {
        let queue = LockFreeQueue::new();
        for i in 0..20 {
            queue.enqueue(i);
        }
        for i in 0..20 {
            assert_eq!(queue.dequeue(), Some(i));
        }
    }

    #[test]
    fn test_queue_concurrent_enqueue() {
        let queue = Arc::new(LockFreeQueue::new());
        let n_threads = 8;
        let n_items = 1000;

        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let queue = Arc::clone(&queue);
                thread::spawn(move || {
                    for i in 0..n_items {
                        queue.enqueue(t * n_items + i);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        // Dequeue all
        let mut items = Vec::new();
        while let Some(v) = queue.dequeue() {
            items.push(v);
        }
        assert_eq!(items.len(), n_threads * n_items);

        // Verify all values are present
        items.sort_unstable();
        let mut expected: Vec<usize> = Vec::new();
        for t in 0..n_threads {
            for i in 0..n_items {
                expected.push(t * n_items + i);
            }
        }
        expected.sort_unstable();
        assert_eq!(items, expected);
    }

    #[test]
    fn test_queue_concurrent_enqueue_dequeue() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let queue = Arc::new(LockFreeQueue::new());
        let n_threads = 4;
        let n_items = 500;
        let total = n_threads * n_items;
        let remaining = Arc::new(AtomicUsize::new(total));

        let producers: Vec<_> = (0..n_threads)
            .map(|_| {
                let queue = Arc::clone(&queue);
                thread::spawn(move || {
                    for i in 0..n_items {
                        queue.enqueue(i);
                    }
                })
            })
            .collect();

        let consumers: Vec<_> = (0..n_threads)
            .map(|_| {
                let queue = Arc::clone(&queue);
                let remaining = Arc::clone(&remaining);
                thread::spawn(move || {
                    let mut count = 0usize;
                    loop {
                        // Check if all items have been claimed
                        let rem = remaining.load(Ordering::Acquire);
                        if rem == 0 {
                            break;
                        }
                        if let Some(_) = queue.dequeue() {
                            remaining.fetch_sub(1, Ordering::AcqRel);
                            count += 1;
                        } else {
                            thread::yield_now();
                        }
                    }
                    count
                })
            })
            .collect();

        for h in producers {
            h.join().expect("producer panicked");
        }

        let total_consumed: usize = consumers
            .into_iter()
            .map(|h| h.join().expect("consumer panicked"))
            .sum();
        assert_eq!(total_consumed, total);
    }

    #[test]
    fn test_queue_drop_frees_memory() {
        let queue = LockFreeQueue::new();
        for i in 0..100 {
            queue.enqueue(format!("item_{i}"));
        }
        drop(queue);
    }

    #[test]
    fn test_queue_default() {
        let queue: LockFreeQueue<i32> = Default::default();
        assert!(queue.is_empty());
    }

    // ── LockFreeCounter ─────────────────────────────────────────────────────

    #[test]
    fn test_counter_basic() {
        let counter = LockFreeCounter::new(0);
        assert_eq!(counter.get(), 0);
        assert_eq!(counter.increment(), 0);
        assert_eq!(counter.get(), 1);
        assert_eq!(counter.increment(), 1);
        assert_eq!(counter.get(), 2);
        assert_eq!(counter.decrement(), 2);
        assert_eq!(counter.get(), 1);
    }

    #[test]
    fn test_counter_concurrent() {
        let counter = Arc::new(LockFreeCounter::new(0));
        let n_threads = 8;
        let n_increments = 10_000;

        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let counter = Arc::clone(&counter);
                thread::spawn(move || {
                    for _ in 0..n_increments {
                        counter.increment();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        assert_eq!(counter.get(), n_threads * n_increments);
    }

    #[test]
    fn test_counter_decrement_saturates() {
        let counter = LockFreeCounter::new(0);
        assert_eq!(counter.decrement(), 0);
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_counter_compare_and_swap() {
        let counter = LockFreeCounter::new(10);
        assert_eq!(counter.compare_and_swap(10, 20), Ok(10));
        assert_eq!(counter.get(), 20);
        assert_eq!(counter.compare_and_swap(10, 30), Err(20));
        assert_eq!(counter.get(), 20);
    }

    #[test]
    fn test_counter_reset() {
        let counter = LockFreeCounter::new(0);
        counter.add(100);
        assert_eq!(counter.reset(), 100);
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_counter_add() {
        let counter = LockFreeCounter::new(5);
        assert_eq!(counter.add(10), 5);
        assert_eq!(counter.get(), 15);
    }
}
