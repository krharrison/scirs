//! Slab allocator — O(1) fixed-size object pool with compact iteration.
//!
//! A [`Slab<T>`] maintains a contiguous `Vec` of *entries* where each entry is
//! either **occupied** (holds a `T` value) or **vacant** (part of an intrusive
//! free list).  This layout gives:
//!
//! * O(1) `insert` — pop from the free list or push to the end.
//! * O(1) `remove` — mark entry as vacant, push index onto free list.
//! * O(1) `get` / `get_mut` — direct index into the backing `Vec`.
//! * Dense iteration — occupied entries are visited in insertion-index order.
//!
//! Keys are 32-bit integers (`SlabKey(u32)`), keeping them compact in data
//! structures that store many keys.
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::memory::slab::{Slab, SlabKey};
//!
//! let mut slab: Slab<&str> = Slab::new();
//! let k1: SlabKey = slab.insert("hello");
//! let k2: SlabKey = slab.insert("world");
//! assert_eq!(slab.get(k1), Some(&"hello"));
//! let removed = slab.remove(k1);
//! assert_eq!(removed, "hello");
//! // k1 is now invalid; k2 still works.
//! assert_eq!(slab.get(k2), Some(&"world"));
//! ```

/// An opaque 32-bit key into a [`Slab<T>`].
///
/// Keys are invalidated when the corresponding entry is [`Slab::remove`]d.
/// Using a stale key returns `None` from `get` / `get_mut`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SlabKey(u32);

impl SlabKey {
    /// Create a key from a raw index.  Intended for testing and serialization.
    #[inline]
    pub fn from_raw(idx: u32) -> Self {
        SlabKey(idx)
    }

    /// Return the raw index value.
    #[inline]
    pub fn into_raw(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for SlabKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SlabKey({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Internal entry representation
// ---------------------------------------------------------------------------

/// Each slot in the backing `Vec` is either occupied or a node in the
/// intrusive free list.
enum Entry<T> {
    /// The slot holds a live value together with a *generation* counter that
    /// lets us detect use-after-remove if callers store keys across removes.
    Occupied {
        value: T,
        generation: u32,
    },
    /// The slot is free.  `next_free` is the index of the next free slot
    /// (`u32::MAX` ≙ end of list).
    Vacant {
        next_free: u32,
        /// Generation at which this slot was freed.  The *next* occupant will
        /// have generation `freed_gen + 1` so old keys do not accidentally
        /// alias new data.
        freed_gen: u32,
    },
}

// ---------------------------------------------------------------------------
// Slab<T>
// ---------------------------------------------------------------------------

/// A slab allocator for homogeneous objects.
///
/// See the [module-level documentation](self) for an overview.
pub struct Slab<T> {
    entries: Vec<Entry<T>>,
    /// Head of the intrusive free list (`u32::MAX` ≙ empty).
    free_head: u32,
    /// Number of occupied entries.
    len: usize,
}

impl<T> Slab<T> {
    /// Create an empty slab.
    pub fn new() -> Self {
        Slab {
            entries: Vec::new(),
            free_head: u32::MAX,
            len: 0,
        }
    }

    /// Create a slab with pre-allocated capacity for `cap` objects.
    pub fn with_capacity(cap: usize) -> Self {
        Slab {
            entries: Vec::with_capacity(cap),
            free_head: u32::MAX,
            len: 0,
        }
    }

    /// Insert `value` and return its key.
    ///
    /// # Panics
    ///
    /// Panics if the slab contains `u32::MAX` (4 294 967 295) occupied entries,
    /// which is a practical impossibility.
    pub fn insert(&mut self, value: T) -> SlabKey {
        if self.free_head != u32::MAX {
            // Recycle a vacant slot.
            let idx = self.free_head as usize;
            let (next_free, new_gen) = match &self.entries[idx] {
                Entry::Vacant { next_free, freed_gen } => (*next_free, freed_gen.wrapping_add(1)),
                Entry::Occupied { .. } => unreachable!("free list must point to vacant entries"),
            };
            self.free_head = next_free;
            self.entries[idx] = Entry::Occupied {
                value,
                generation: new_gen,
            };
            self.len += 1;
            SlabKey(idx as u32)
        } else {
            // Append a new slot at the end.
            let idx = self.entries.len();
            assert!(idx < u32::MAX as usize, "slab capacity overflow");
            self.entries.push(Entry::Occupied {
                value,
                generation: 0,
            });
            self.len += 1;
            SlabKey(idx as u32)
        }
    }

    /// Remove the value associated with `key` and return it.
    ///
    /// # Panics
    ///
    /// Panics if `key` is out of range or already vacant (use-after-remove).
    pub fn remove(&mut self, key: SlabKey) -> T {
        let idx = key.0 as usize;
        assert!(idx < self.entries.len(), "SlabKey out of range");
        let freed_gen = match &self.entries[idx] {
            Entry::Occupied { generation, .. } => *generation,
            Entry::Vacant { .. } => panic!("attempted to remove already-vacant entry {key}"),
        };
        // Swap in a Vacant entry, retrieving the old value.
        let old = std::mem::replace(
            &mut self.entries[idx],
            Entry::Vacant {
                next_free: self.free_head,
                freed_gen,
            },
        );
        self.free_head = idx as u32;
        self.len -= 1;
        match old {
            Entry::Occupied { value, .. } => value,
            Entry::Vacant { .. } => unreachable!(),
        }
    }

    /// Return a reference to the value at `key`, or `None` if vacant or out of range.
    pub fn get(&self, key: SlabKey) -> Option<&T> {
        let idx = key.0 as usize;
        match self.entries.get(idx)? {
            Entry::Occupied { value, .. } => Some(value),
            Entry::Vacant { .. } => None,
        }
    }

    /// Return a mutable reference to the value at `key`, or `None` if vacant / out of range.
    pub fn get_mut(&mut self, key: SlabKey) -> Option<&mut T> {
        let idx = key.0 as usize;
        match self.entries.get_mut(idx)? {
            Entry::Occupied { value, .. } => Some(value),
            Entry::Vacant { .. } => None,
        }
    }

    /// Returns `true` if the slab contains no occupied entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of occupied entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Current capacity of the backing storage.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.entries.capacity()
    }

    /// Iterate over all occupied entries as `(SlabKey, &T)` pairs.
    ///
    /// Entries are yielded in insertion-index order (not necessarily insertion
    /// chronological order after removes).
    pub fn iter(&self) -> SlabIter<'_, T> {
        SlabIter {
            entries: &self.entries,
            index: 0,
        }
    }

    /// Iterate over all occupied entries as `(SlabKey, &mut T)` pairs.
    pub fn iter_mut(&mut self) -> SlabIterMut<'_, T> {
        SlabIterMut {
            entries: &mut self.entries,
            index: 0,
        }
    }

    /// Remove all entries, leaving the backing storage allocated.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.free_head = u32::MAX;
        self.len = 0;
    }

    /// Returns `true` if the key points to an occupied entry.
    pub fn contains(&self, key: SlabKey) -> bool {
        self.get(key).is_some()
    }
}

impl<T> Default for Slab<T> {
    fn default() -> Self {
        Slab::new()
    }
}

// ---------------------------------------------------------------------------
// Iterators
// ---------------------------------------------------------------------------

/// Immutable iterator over occupied slab entries.
pub struct SlabIter<'a, T> {
    entries: &'a [Entry<T>],
    index: usize,
}

impl<'a, T> Iterator for SlabIter<'a, T> {
    type Item = (SlabKey, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let idx = self.index;
            if idx >= self.entries.len() {
                return None;
            }
            self.index += 1;
            if let Entry::Occupied { value, .. } = &self.entries[idx] {
                return Some((SlabKey(idx as u32), value));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.entries.len() - self.index))
    }
}

/// Mutable iterator over occupied slab entries.
pub struct SlabIterMut<'a, T> {
    entries: &'a mut [Entry<T>],
    index: usize,
}

impl<'a, T> Iterator for SlabIterMut<'a, T> {
    type Item = (SlabKey, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let idx = self.index;
            if idx >= self.entries.len() {
                return None;
            }
            self.index += 1;
            // SAFETY: we advance `index` each iteration so each slot is
            // accessed at most once, satisfying the aliasing rules for `&mut`.
            let entry = unsafe { &mut *(self.entries.as_mut_ptr().add(idx)) };
            if let Entry::Occupied { value, .. } = entry {
                return Some((SlabKey(idx as u32), value));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.entries.len() - self.index))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slab_insert_get() {
        let mut s: Slab<i32> = Slab::new();
        let k1 = s.insert(10);
        let k2 = s.insert(20);
        let k3 = s.insert(30);
        assert_eq!(s.get(k1), Some(&10));
        assert_eq!(s.get(k2), Some(&20));
        assert_eq!(s.get(k3), Some(&30));
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn slab_remove_and_reinsert() {
        let mut s: Slab<String> = Slab::new();
        let k1 = s.insert("alpha".to_string());
        let k2 = s.insert("beta".to_string());
        assert_eq!(s.remove(k1), "alpha");
        assert_eq!(s.len(), 1);
        assert_eq!(s.get(k1), None); // stale key
        // Reinsert reuses the slot.
        let k3 = s.insert("gamma".to_string());
        assert_eq!(k3, k1, "slot k1 should have been recycled");
        assert_eq!(s.get(k3), Some(&"gamma".to_string()));
        assert_eq!(s.get(k2), Some(&"beta".to_string()));
    }

    #[test]
    fn slab_get_mut() {
        let mut s: Slab<i32> = Slab::new();
        let k = s.insert(42);
        *s.get_mut(k).expect("entry should exist") = 99;
        assert_eq!(s.get(k), Some(&99));
    }

    #[test]
    fn slab_iter() {
        let mut s: Slab<u32> = Slab::new();
        let k0 = s.insert(0_u32);
        let k1 = s.insert(1_u32);
        let k2 = s.insert(2_u32);
        s.remove(k1);
        let pairs: Vec<_> = s.iter().collect();
        assert_eq!(pairs.len(), 2);
        assert!(pairs.contains(&(k0, &0)));
        assert!(pairs.contains(&(k2, &2)));
    }

    #[test]
    fn slab_iter_mut() {
        let mut s: Slab<i32> = Slab::new();
        s.insert(1);
        s.insert(2);
        s.insert(3);
        for (_k, v) in s.iter_mut() {
            *v *= 10;
        }
        let vals: Vec<i32> = s.iter().map(|(_, v)| *v).collect();
        assert_eq!(vals, vec![10, 20, 30]);
    }

    #[test]
    fn slab_clear() {
        let mut s: Slab<i32> = Slab::new();
        let _k = s.insert(5);
        s.clear();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn slab_contains() {
        let mut s: Slab<()> = Slab::new();
        let k = s.insert(());
        assert!(s.contains(k));
        s.remove(k);
        assert!(!s.contains(k));
    }

    #[test]
    #[should_panic(expected = "already-vacant")]
    fn slab_double_remove_panics() {
        let mut s: Slab<i32> = Slab::new();
        let k = s.insert(1);
        s.remove(k);
        s.remove(k); // must panic
    }

    #[test]
    fn slab_key_raw_round_trip() {
        let k = SlabKey::from_raw(42);
        assert_eq!(k.into_raw(), 42);
    }

    #[test]
    fn slab_large_sequence() {
        const N: usize = 1_000;
        let mut s: Slab<usize> = Slab::with_capacity(N);
        let keys: Vec<SlabKey> = (0..N).map(|i| s.insert(i)).collect();
        assert_eq!(s.len(), N);
        // Remove even-indexed entries.
        for i in (0..N).step_by(2) {
            let v = s.remove(keys[i]);
            assert_eq!(v, i);
        }
        assert_eq!(s.len(), N / 2);
        // All odd entries still accessible.
        for i in (1..N).step_by(2) {
            assert_eq!(s.get(keys[i]), Some(&i));
        }
    }
}
