//! `TinyVec<T, N>` — a hybrid stack/heap vector optimised for small sizes.
//!
//! Up to `N` elements are stored inline in a fixed-size array on the stack.
//! When the number of elements exceeds `N` the backing storage automatically
//! spills onto the heap and the inline array is abandoned.
//!
//! # Design goals
//!
//! - **Zero heap allocation** for collections that stay within `N` elements.
//! - **Transparent API** that mirrors `Vec<T>` for push / pop / index / iter.
//! - **No `unsafe` needed for the API** — the `MaybeUninit` usage is fully
//!   contained in this module and carefully justified.

use std::fmt;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut, Index, IndexMut};

// ============================================================================
// TinyVec
// ============================================================================

/// The maximum number of elements stored inline when `N == 0` at compile time.
/// This case is degenerate; we still support it by treating capacity as 0.

enum Storage<T, const N: usize> {
    Inline {
        data: [MaybeUninit<T>; N],
        len: usize,
    },
    Heap(Vec<T>),
}

/// A hybrid stack/heap vector.
///
/// Elements are stored inline (on the stack) until `N` elements are exceeded,
/// at which point the entire buffer is moved to the heap.
///
/// # Example
///
/// ```rust
/// use scirs2_core::collections::TinyVec;
///
/// let mut v: TinyVec<i32, 4> = TinyVec::new();
/// v.push(1);
/// v.push(2);
/// v.push(3);
/// v.push(4); // still inline
/// v.push(5); // spills to heap
///
/// assert_eq!(v.len(), 5);
/// assert_eq!(v[0], 1);
/// assert_eq!(v[4], 5);
/// ```
pub struct TinyVec<T, const N: usize> {
    storage: Storage<T, N>,
}

impl<T, const N: usize> TinyVec<T, N> {
    /// Creates a new, empty `TinyVec`.
    pub fn new() -> Self {
        TinyVec {
            // SAFETY: An array of MaybeUninit is always valid in its uninitialised state.
            storage: Storage::Inline {
                data: unsafe { MaybeUninit::uninit().assume_init() },
                len: 0,
            },
        }
    }

    /// Creates a `TinyVec` from an existing `Vec<T>`, immediately using heap storage.
    pub fn from_vec(v: Vec<T>) -> Self {
        TinyVec {
            storage: Storage::Heap(v),
        }
    }

    /// Returns the number of elements stored.
    pub fn len(&self) -> usize {
        match &self.storage {
            Storage::Inline { len, .. } => *len,
            Storage::Heap(v) => v.len(),
        }
    }

    /// Returns `true` if there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if the backing storage is currently on the stack.
    pub fn is_inline(&self) -> bool {
        matches!(&self.storage, Storage::Inline { .. })
    }

    /// Appends `value` to the end of the vector.
    ///
    /// If the inline buffer is full the vector spills to the heap automatically.
    pub fn push(&mut self, value: T) {
        match &mut self.storage {
            Storage::Inline { data, len } => {
                if *len < N {
                    // SAFETY: `*len < N` so `data[*len]` is within bounds.
                    unsafe {
                        data[*len].as_mut_ptr().write(value);
                    }
                    *len += 1;
                } else {
                    // Spill to heap.
                    self.spill_to_heap(value);
                }
            }
            Storage::Heap(v) => v.push(value),
        }
    }

    /// Removes and returns the last element, or `None` if empty.
    pub fn pop(&mut self) -> Option<T> {
        match &mut self.storage {
            Storage::Inline { data, len } => {
                if *len == 0 {
                    return None;
                }
                *len -= 1;
                // SAFETY: The element at index `*len` was initialised by a prior push.
                let value = unsafe { data[*len].as_ptr().read() };
                Some(value)
            }
            Storage::Heap(v) => v.pop(),
        }
    }

    /// Returns a reference to the element at `index`, or `None` if out of range.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len() {
            return None;
        }
        match &self.storage {
            // SAFETY: index < len, so the slot is initialised.
            Storage::Inline { data, .. } => Some(unsafe { &*data[index].as_ptr() }),
            Storage::Heap(v) => v.get(index),
        }
    }

    /// Returns a mutable reference to the element at `index`, or `None` if out of range.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len() {
            return None;
        }
        match &mut self.storage {
            // SAFETY: index < len, so the slot is initialised.
            Storage::Inline { data, .. } => Some(unsafe { &mut *data[index].as_mut_ptr() }),
            Storage::Heap(v) => v.get_mut(index),
        }
    }

    /// Returns a slice view of all elements.
    pub fn as_slice(&self) -> &[T] {
        match &self.storage {
            // SAFETY: The first `len` elements are initialised.
            Storage::Inline { data, len } => unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const T, *len)
            },
            Storage::Heap(v) => v.as_slice(),
        }
    }

    /// Returns a mutable slice view of all elements.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match &mut self.storage {
            // SAFETY: The first `len` elements are initialised.
            Storage::Inline { data, len } => unsafe {
                std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut T, *len)
            },
            Storage::Heap(v) => v.as_mut_slice(),
        }
    }

    /// Clears the vector, dropping all elements.
    pub fn clear(&mut self) {
        match &mut self.storage {
            Storage::Inline { data, len } => {
                // Drop all initialised elements.
                for i in 0..*len {
                    // SAFETY: elements 0..*len are initialised.
                    unsafe { data[i].as_mut_ptr().drop_in_place() };
                }
                *len = 0;
            }
            Storage::Heap(v) => v.clear(),
        }
    }

    /// Converts the `TinyVec` into an owned `Vec<T>`, which may involve a heap
    /// allocation if the data is currently stored inline.
    pub fn into_vec(mut self) -> Vec<T> {
        // Use ManuallyDrop to prevent the Drop impl from running after we
        // move the storage out via ptr::read.
        let storage = unsafe { std::ptr::read(&self.storage) };
        // Prevent our Drop from double-freeing.
        std::mem::forget(self);

        match storage {
            Storage::Inline { data, len } => {
                let mut v = Vec::with_capacity(len);
                for i in 0..len {
                    // SAFETY: elements 0..len are initialised; we consume them.
                    v.push(unsafe { data[i].as_ptr().read() });
                }
                // MaybeUninit<T> does not implement Drop, so no explicit
                // forget is needed — the array is consumed by value above.
                v
            }
            Storage::Heap(v) => v,
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Moves all inline elements to a new heap `Vec` and then pushes `extra`.
    fn spill_to_heap(&mut self, extra: T) {
        // We temporarily move `self.storage` out; this is safe because we
        // immediately replace it.
        let old_storage = std::mem::replace(
            &mut self.storage,
            // Placeholder — will be replaced before we return.
            Storage::Heap(Vec::new()),
        );

        if let Storage::Inline { data, len } = old_storage {
            let mut v = Vec::with_capacity(len + 1);
            for i in 0..len {
                // SAFETY: elements 0..len were initialised.
                v.push(unsafe { data[i].as_ptr().read() });
            }
            // MaybeUninit<T> does not implement Drop — no forget needed.
            v.push(extra);
            self.storage = Storage::Heap(v);
        }
        // The placeholder `Storage::Heap(Vec::new())` case should never be reached.
    }
}

// ============================================================================
// Drop
// ============================================================================

impl<T, const N: usize> Drop for TinyVec<T, N> {
    fn drop(&mut self) {
        // The Heap variant drops T elements automatically via Vec's Drop.
        // For the Inline variant we must manually drop initialised elements.
        if let Storage::Inline { data, len } = &mut self.storage {
            for i in 0..*len {
                // SAFETY: elements 0..*len are initialised.
                unsafe { data[i].as_mut_ptr().drop_in_place() };
            }
            // Set len = 0 so a hypothetical future double-drop is a no-op.
            *len = 0;
        }
    }
}

// ============================================================================
// Trait implementations
// ============================================================================

impl<T, const N: usize> Deref for TinyVec<T, N> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize> DerefMut for TinyVec<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, const N: usize> Index<usize> for TinyVec<T, N> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        self.get(index).expect("TinyVec: index out of bounds")
    }
}

impl<T, const N: usize> IndexMut<usize> for TinyVec<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.get_mut(index).expect("TinyVec: index out of bounds")
    }
}

impl<T: Clone, const N: usize> Clone for TinyVec<T, N> {
    fn clone(&self) -> Self {
        let mut out = TinyVec::new();
        for elem in self.as_slice() {
            out.push(elem.clone());
        }
        out
    }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for TinyVec<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), f)
    }
}

impl<T: PartialEq, const N: usize> PartialEq for TinyVec<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq, const N: usize> Eq for TinyVec<T, N> {}

impl<T, const N: usize> FromIterator<T> for TinyVec<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut v = TinyVec::new();
        for item in iter {
            v.push(item);
        }
        v
    }
}

impl<T, const N: usize> IntoIterator for TinyVec<T, N> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a TinyVec<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_within_inline() {
        let mut v: TinyVec<i32, 4> = TinyVec::new();
        v.push(10);
        v.push(20);
        v.push(30);
        assert!(v.is_inline());
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 10);
        assert_eq!(v[2], 30);
    }

    #[test]
    fn test_spill_to_heap() {
        let mut v: TinyVec<i32, 2> = TinyVec::new();
        v.push(1);
        v.push(2);
        assert!(v.is_inline());
        v.push(3); // triggers spill
        assert!(!v.is_inline());
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 3);
    }

    #[test]
    fn test_pop() {
        let mut v: TinyVec<i32, 4> = TinyVec::new();
        v.push(42);
        assert_eq!(v.pop(), Some(42));
        assert_eq!(v.pop(), None);
    }

    #[test]
    fn test_pop_after_spill() {
        let mut v: TinyVec<i32, 1> = TinyVec::new();
        v.push(1);
        v.push(2);
        assert_eq!(v.pop(), Some(2));
        assert_eq!(v.pop(), Some(1));
        assert_eq!(v.pop(), None);
    }

    #[test]
    fn test_clear() {
        let mut v: TinyVec<String, 4> = TinyVec::new();
        v.push("hello".to_string());
        v.push("world".to_string());
        v.clear();
        assert!(v.is_empty());
    }

    #[test]
    fn test_drop_non_copy() {
        // Ensure that strings are properly dropped (checked via Miri in CI).
        let mut v: TinyVec<String, 2> = TinyVec::new();
        v.push("a".to_string());
        v.push("b".to_string());
        v.push("c".to_string()); // spill
        drop(v);
    }

    #[test]
    fn test_iter() {
        let mut v: TinyVec<i32, 4> = TinyVec::new();
        for i in 0..6 {
            v.push(i);
        }
        let collected: Vec<_> = v.iter().copied().collect();
        assert_eq!(collected, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_from_iter() {
        let v: TinyVec<i32, 4> = (0..8).collect();
        assert_eq!(v.len(), 8);
        for (i, &x) in v.iter().enumerate() {
            assert_eq!(x, i as i32);
        }
    }

    #[test]
    fn test_clone() {
        let mut v: TinyVec<i32, 4> = TinyVec::new();
        v.push(1);
        v.push(2);
        let w = v.clone();
        assert_eq!(v, w);
    }

    #[test]
    fn test_zero_capacity() {
        // N = 0: every push immediately triggers a heap allocation.
        let mut v: TinyVec<i32, 0> = TinyVec::new();
        v.push(99);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], 99);
    }
}
