//! Treiber lock-free stack with versioned pointer ABA mitigation.
//!
//! # ABA Mitigation
//!
//! On 64-bit systems the pointer value only uses the lower 48 bits (canonical
//! virtual addresses).  We exploit the remaining 16 high bits as a monotonically
//! increasing version counter that is incremented on every push.  This reduces
//! the probability of a false CAS success due to ABA to effectively zero for
//! any practical workload.
//!
//! The scheme packs `(version: u16, ptr: *mut Node<T>)` into a single `u64`
//! which can be stored in one `AtomicU64`.
//!
//! # Safety Contract
//!
//! Nodes are allocated via `Box::into_raw` and freed with `Box::from_raw`.
//! The only mutable access to a node's fields happens immediately after
//! allocation (on push) or immediately after a successful CAS-pop before the
//! pointer is published to any other thread.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// Versioned pointer encoding
// ---------------------------------------------------------------------------

/// On 64-bit Linux/macOS, user-space virtual addresses use at most 48 bits.
/// We store the 16-bit version in bits [63:48].
const PTR_MASK: u64 = (1u64 << 48) - 1;
const VER_SHIFT: u32 = 48;

#[inline]
fn pack(version: u16, ptr: *mut u8) -> u64 {
    ((version as u64) << VER_SHIFT) | (ptr as u64 & PTR_MASK)
}

#[inline]
fn unpack_ptr<T>(raw: u64) -> *mut Node<T> {
    // Sign-extend bits [47:0] to reconstruct a canonical pointer.
    let addr = raw & PTR_MASK;
    // On x86-64 / AArch64 user space the top bit of the 48-bit range is 0,
    // so no sign-extension is needed for user-space pointers.
    addr as *mut Node<T>
}

#[inline]
fn unpack_version(raw: u64) -> u16 {
    (raw >> VER_SHIFT) as u16
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

struct Node<T> {
    value: T,
    /// Raw packed (version, next) of the *next* node.
    next: u64,
}

// ---------------------------------------------------------------------------
// LockFreeStack
// ---------------------------------------------------------------------------

/// A lock-free stack (LIFO) using the Treiber algorithm with versioned
/// pointer ABA mitigation.
///
/// `T` must be `Send`.
///
/// # Example
///
/// ```rust
/// use scirs2_core::concurrent::LockFreeStack;
///
/// let s: LockFreeStack<i32> = LockFreeStack::new();
/// s.push(1);
/// s.push(2);
/// assert_eq!(s.pop(), Some(2));
/// assert_eq!(s.pop(), Some(1));
/// assert_eq!(s.pop(), None);
/// ```
pub struct LockFreeStack<T> {
    _phantom: std::marker::PhantomData<T>,
    /// Packed (version:16, ptr:48).  0 means null / empty.
    head: AtomicU64,
    len: AtomicUsize,
}

unsafe impl<T: Send> Send for LockFreeStack<T> {}
unsafe impl<T: Send> Sync for LockFreeStack<T> {}

impl<T> LockFreeStack<T> {
    /// Create a new empty stack.
    pub fn new() -> Self {
        LockFreeStack {
            _phantom: std::marker::PhantomData,
            head: AtomicU64::new(0),
            len: AtomicUsize::new(0),
        }
    }

    /// Push `val` onto the top of the stack.
    pub fn push(&self, val: T) {
        let node = Box::into_raw(Box::new(Node { value: val, next: 0 }));

        loop {
            let old_head = self.head.load(Ordering::Acquire);
            // Embed the existing head as next.
            unsafe { (*node).next = old_head };

            let old_version = unpack_version(old_head);
            let new_version = old_version.wrapping_add(1);
            let new_head = pack(new_version, node as *mut u8);

            match self.head.compare_exchange_weak(
                old_head,
                new_head,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.len.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                Err(_) => {
                    // Retry; node.next will be overwritten in the next iteration.
                }
            }
        }
    }

    /// Pop the top value from the stack, returning `None` if empty.
    pub fn pop(&self) -> Option<T> {
        loop {
            let old_head = self.head.load(Ordering::Acquire);
            if old_head & PTR_MASK == 0 {
                return None; // empty
            }

            let node_ptr = unpack_ptr::<T>(old_head);
            // SAFETY: The pointer is non-null and was created by Box::into_raw
            // in push().  We verify non-null above.
            let next = unsafe { (*node_ptr).next };

            // Keep version from next (it already has the version embedded).
            match self.head.compare_exchange_weak(
                old_head,
                next,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.len.fetch_sub(1, Ordering::Relaxed);
                    // Reclaim the node.
                    let boxed = unsafe { Box::from_raw(node_ptr) };
                    return Some(boxed.value);
                }
                Err(_) => {
                    // Retry.
                }
            }
        }
    }

    /// Returns `true` if the stack is empty.
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Relaxed) & PTR_MASK == 0
    }

    /// Returns the number of elements in the stack (approximate under
    /// concurrent modification).
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }
}

impl<T> Default for LockFreeStack<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for LockFreeStack<T> {
    fn drop(&mut self) {
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
        let s: LockFreeStack<i32> = LockFreeStack::new();
        assert!(s.is_empty());
        s.push(1);
        s.push(2);
        s.push(3);
        assert_eq!(s.len(), 3);
        assert_eq!(s.pop(), Some(3));
        assert_eq!(s.pop(), Some(2));
        assert_eq!(s.pop(), Some(1));
        assert_eq!(s.pop(), None);
        assert!(s.is_empty());
    }

    #[test]
    fn test_lifo_ordering() {
        let s: LockFreeStack<usize> = LockFreeStack::new();
        for i in 0..100 {
            s.push(i);
        }
        for i in (0..100).rev() {
            assert_eq!(s.pop(), Some(i));
        }
    }

    #[test]
    fn test_concurrent_push_pop() {
        const THREADS: usize = 8;
        const OPS: usize = 10_000;

        let stack = Arc::new(LockFreeStack::<usize>::new());
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let s = Arc::clone(&stack);
                let c = Arc::clone(&counter);
                thread::spawn(move || {
                    for _ in 0..OPS {
                        s.push(1);
                    }
                    // Pop as many as we can.
                    loop {
                        match s.pop() {
                            Some(v) => {
                                c.fetch_add(v, std::sync::atomic::Ordering::Relaxed);
                            }
                            None => break,
                        }
                        thread::yield_now();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        // Drain remainder from main thread.
        while stack.pop().is_some() {}

        assert!(stack.is_empty());
    }

    #[test]
    fn test_drop_runs_destructors() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let drops = Arc::new(AtomicUsize::new(0));

        struct Bomb(Arc<AtomicUsize>);
        impl Drop for Bomb {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        {
            let s: LockFreeStack<Bomb> = LockFreeStack::new();
            for _ in 0..5 {
                s.push(Bomb(Arc::clone(&drops)));
            }
            // Drop the stack; all 5 Bombs should be freed.
        }
        assert_eq!(drops.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_default_is_empty() {
        let s: LockFreeStack<String> = LockFreeStack::default();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn test_single_element_roundtrip() {
        let s: LockFreeStack<String> = LockFreeStack::new();
        s.push("hello".to_string());
        assert_eq!(s.len(), 1);
        assert_eq!(s.pop(), Some("hello".to_string()));
        assert_eq!(s.pop(), None);
    }
}
