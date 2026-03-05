//! `FlatMap<K, V>` — a sorted-Vec–based map optimised for small sizes.
//!
//! For maps with fewer than ~20 entries a sorted `Vec` of `(K, V)` pairs
//! consistently outperforms a `HashMap` because:
//!
//! - No heap allocation for the hash table itself.
//! - Better cache locality (all entries packed contiguously).
//! - Binary-search O(log n) is fast and branch-predictable at small n.
//! - Simple insertion into a sorted array is O(n) but n is tiny.
//!
//! When the map grows beyond its small-map sweet spot (roughly 20 entries) a
//! `HashMap` becomes faster; callers who expect large maps should migrate to
//! `HashMap` at that point.  `FlatMap` does not enforce a size limit — it
//! continues to work correctly at any size.
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::collections::FlatMap;
//!
//! let mut m: FlatMap<&str, u32> = FlatMap::new();
//! m.insert("b", 2);
//! m.insert("a", 1);
//! m.insert("c", 3);
//!
//! assert_eq!(m.get(&"a"), Some(&1));
//! assert_eq!(m.get(&"d"), None);
//! assert_eq!(m.len(), 3);
//!
//! // Keys are kept in sorted order.
//! let keys: Vec<_> = m.keys().copied().collect();
//! assert_eq!(keys, vec!["a", "b", "c"]);
//! ```

use std::fmt;

// ============================================================================
// FlatMap
// ============================================================================

/// A sorted `Vec`-backed ordered map suitable for collections with fewer than
/// roughly 20 entries.
pub struct FlatMap<K, V> {
    /// Pairs stored in ascending key order.
    pairs: Vec<(K, V)>,
}

impl<K: Ord, V> FlatMap<K, V> {
    /// Creates a new, empty `FlatMap`.
    pub fn new() -> Self {
        FlatMap { pairs: Vec::new() }
    }

    /// Creates a new `FlatMap` pre-allocated for `cap` entries.
    pub fn with_capacity(cap: usize) -> Self {
        FlatMap {
            pairs: Vec::with_capacity(cap),
        }
    }

    /// Inserts `(k, v)` into the map.
    ///
    /// If `k` already exists the old value is replaced and returned.
    /// If `k` is new, `None` is returned.
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        match self.pairs.binary_search_by(|(pk, _)| pk.cmp(&k)) {
            Ok(pos) => {
                // Key already present — replace and return old value.
                let old = std::mem::replace(&mut self.pairs[pos].1, v);
                Some(old)
            }
            Err(pos) => {
                // New key — insert at sorted position.
                self.pairs.insert(pos, (k, v));
                None
            }
        }
    }

    /// Returns a reference to the value for `k`, or `None` if absent.
    pub fn get(&self, k: &K) -> Option<&V> {
        self.pairs
            .binary_search_by(|(pk, _)| pk.cmp(k))
            .ok()
            .map(|pos| &self.pairs[pos].1)
    }

    /// Returns a mutable reference to the value for `k`, or `None` if absent.
    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        self.pairs
            .binary_search_by(|(pk, _)| pk.cmp(k))
            .ok()
            .map(|pos| &mut self.pairs[pos].1)
    }

    /// Returns `true` if `k` is present in the map.
    pub fn contains_key(&self, k: &K) -> bool {
        self.pairs
            .binary_search_by(|(pk, _)| pk.cmp(k))
            .is_ok()
    }

    /// Removes `k` from the map, returning its value if it was present.
    pub fn remove(&mut self, k: &K) -> Option<V> {
        self.pairs
            .binary_search_by(|(pk, _)| pk.cmp(k))
            .ok()
            .map(|pos| self.pairs.remove(pos).1)
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Returns `true` if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.pairs.clear();
    }

    /// Returns an iterator over `(&K, &V)` pairs in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.pairs.iter().map(|(k, v)| (k, v))
    }

    /// Returns an iterator over `(&K, &mut V)` pairs in key order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.pairs.iter_mut().map(|(k, v)| (k as &K, v))
    }

    /// Returns an iterator over keys in sorted order.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.pairs.iter().map(|(k, _)| k)
    }

    /// Returns an iterator over values in key order.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.pairs.iter().map(|(_, v)| v)
    }

    /// Returns a reference to the entry with the smallest key, or `None`.
    pub fn first(&self) -> Option<(&K, &V)> {
        self.pairs.first().map(|(k, v)| (k, v))
    }

    /// Returns a reference to the entry with the largest key, or `None`.
    pub fn last(&self) -> Option<(&K, &V)> {
        self.pairs.last().map(|(k, v)| (k, v))
    }

    /// Returns an iterator over `(K, V)` pairs consuming the map, in key order.
    pub fn into_iter(self) -> impl Iterator<Item = (K, V)> {
        self.pairs.into_iter()
    }

    /// Returns a slice view of the internal `(K, V)` pairs (sorted by key).
    pub fn as_slice(&self) -> &[(K, V)] {
        &self.pairs
    }

    /// Returns an entry-like interface for the given key.
    ///
    /// If the key is present, returns `Ok(&mut V)`.
    /// If it is absent, inserts `default` and returns `Ok(&mut V)`.
    pub fn entry_or_insert(&mut self, k: K, default: V) -> &mut V {
        match self.pairs.binary_search_by(|(pk, _)| pk.cmp(&k)) {
            Ok(pos) => &mut self.pairs[pos].1,
            Err(pos) => {
                self.pairs.insert(pos, (k, default));
                &mut self.pairs[pos].1
            }
        }
    }

    /// Retains only entries for which `f` returns `true`.
    pub fn retain<F: FnMut(&K, &mut V) -> bool>(&mut self, mut f: F) {
        self.pairs.retain_mut(|(k, v)| f(k, v));
    }
}

// ============================================================================
// Trait implementations
// ============================================================================

impl<K: Ord + fmt::Debug, V: fmt::Debug> fmt::Debug for FlatMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.pairs.iter().map(|(k, v)| (k, v)))
            .finish()
    }
}

impl<K: Ord + Clone, V: Clone> Clone for FlatMap<K, V> {
    fn clone(&self) -> Self {
        FlatMap {
            pairs: self.pairs.clone(),
        }
    }
}

impl<K: Ord + PartialEq, V: PartialEq> PartialEq for FlatMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.pairs == other.pairs
    }
}

impl<K: Ord + Eq, V: Eq> Eq for FlatMap<K, V> {}

impl<K: Ord, V> FromIterator<(K, V)> for FlatMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut m = FlatMap::new();
        for (k, v) in iter {
            m.insert(k, v);
        }
        m
    }
}

impl<K: Ord, V> Extend<(K, V)> for FlatMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let mut m: FlatMap<&str, u32> = FlatMap::new();
        m.insert("b", 2);
        m.insert("a", 1);
        m.insert("c", 3);

        assert_eq!(m.get(&"a"), Some(&1));
        assert_eq!(m.get(&"b"), Some(&2));
        assert_eq!(m.get(&"c"), Some(&3));
        assert_eq!(m.get(&"d"), None);
    }

    #[test]
    fn test_sorted_keys() {
        let mut m: FlatMap<i32, &str> = FlatMap::new();
        m.insert(5, "five");
        m.insert(1, "one");
        m.insert(3, "three");
        m.insert(2, "two");
        m.insert(4, "four");

        let keys: Vec<i32> = m.keys().copied().collect();
        assert_eq!(keys, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_replace_existing() {
        let mut m: FlatMap<i32, &str> = FlatMap::new();
        let old = m.insert(1, "one");
        assert_eq!(old, None);
        let old2 = m.insert(1, "ONE");
        assert_eq!(old2, Some("one"));
        assert_eq!(m.get(&1), Some(&"ONE"));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_remove() {
        let mut m: FlatMap<i32, i32> = FlatMap::new();
        m.insert(1, 10);
        m.insert(2, 20);
        let removed = m.remove(&1);
        assert_eq!(removed, Some(10));
        assert!(!m.contains_key(&1));
        assert_eq!(m.len(), 1);
        assert_eq!(m.remove(&99), None);
    }

    #[test]
    fn test_contains_key() {
        let mut m: FlatMap<&str, u8> = FlatMap::new();
        m.insert("x", 0);
        assert!(m.contains_key(&"x"));
        assert!(!m.contains_key(&"y"));
    }

    #[test]
    fn test_entry_or_insert() {
        let mut m: FlatMap<i32, i32> = FlatMap::new();
        *m.entry_or_insert(1, 0) += 1;
        *m.entry_or_insert(1, 0) += 1;
        assert_eq!(m.get(&1), Some(&2));
    }

    #[test]
    fn test_iter_order() {
        let m: FlatMap<i32, i32> = [(3, 30), (1, 10), (2, 20)].iter().cloned().collect();
        let pairs: Vec<_> = m.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(pairs, vec![(1, 10), (2, 20), (3, 30)]);
    }

    #[test]
    fn test_retain() {
        let mut m: FlatMap<i32, i32> = [(1, 1), (2, 4), (3, 9)].iter().cloned().collect();
        m.retain(|_, v| *v > 3);
        assert_eq!(m.len(), 2);
        assert!(!m.contains_key(&1));
    }

    #[test]
    fn test_clone_eq() {
        let m: FlatMap<i32, i32> = [(1, 10), (2, 20)].iter().cloned().collect();
        let n = m.clone();
        assert_eq!(m, n);
    }

    #[test]
    fn test_first_last() {
        let m: FlatMap<i32, &str> = [(1, "a"), (3, "b"), (2, "c")]
            .iter()
            .cloned()
            .collect();
        assert_eq!(m.first(), Some((&1, &"a")));
        assert_eq!(m.last(), Some((&3, &"b")));
    }
}
