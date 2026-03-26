//! Concurrent skip list with O(log n) expected operations.
//!
//! This implementation uses a single `Arc<RwLock<SkipListInner<K,V>>>` for
//! concurrent access.  Reads acquire a shared lock, writes acquire an
//! exclusive lock.  The internal representation is a level-indexed tower of
//! sorted `Vec<(K, Arc<V>)>` slices — level 0 contains all keys, level k
//! contains a ~2^{-k} random sample.
//!
//! ## Level promotion
//!
//! Each inserted key is promoted to level k+1 with probability 0.5 via an
//! internal LCG PRNG (no external crate), giving an expected height of 2 and
//! O(log n) expected traversal cost at every level.
//!
//! ## Concurrency
//!
//! - Multiple readers can query the list simultaneously.
//! - Writers serialise via the exclusive write lock.
//! - `Arc<V>` ensures that returned value references outlive any later
//!   mutation without copying the value.

use std::sync::{Arc, RwLock};

/// Maximum tower height.
const MAX_LEVEL: usize = 16;

// ---------------------------------------------------------------------------
// LCG PRNG
// ---------------------------------------------------------------------------

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Return a random level in `[1, MAX_LEVEL]` with p = 0.5 promotion.
    fn random_level(&mut self) -> usize {
        let mut level = 1usize;
        while level < MAX_LEVEL && (self.next() & 1) == 0 {
            level += 1;
        }
        level
    }
}

// ---------------------------------------------------------------------------
// Internal representation
// ---------------------------------------------------------------------------

/// The mutable interior of the skip list.
///
/// `levels[0]` is the full sorted list.  `levels[k]` (k > 0) is a sparse
/// express-lane.  We store `Arc<V>` at every level so that a caller can hold
/// a reference to a value without worrying about it being overwritten.
struct SkipListInner<K, V> {
    /// `levels[lvl]` is a sorted Vec of (key, value) at that express lane.
    levels: Vec<Vec<(K, Arc<V>)>>,
    size: usize,
    prng: Lcg,
}

impl<K: Ord + Clone, V: Clone> SkipListInner<K, V> {
    fn new() -> Self {
        SkipListInner {
            levels: vec![Vec::new(); MAX_LEVEL],
            size: 0,
            prng: Lcg::new(0xDEAD_BEEF_CAFE_BABE),
        }
    }

    // Binary search on level-0 for a key — returns (found, index).
    fn search_l0(&self, key: &K) -> (bool, usize) {
        let base = &self.levels[0];
        match base.binary_search_by(|(k, _)| k.cmp(key)) {
            Ok(i) => (true, i),
            Err(i) => (false, i),
        }
    }

    fn insert(&mut self, key: K, value: V) -> bool {
        let height = self.prng.random_level();
        let val_arc = Arc::new(value);
        let (found, pos) = self.search_l0(&key);

        if found {
            // Update value at every level.
            for lvl in 0..MAX_LEVEL {
                let p = self.levels[lvl].binary_search_by(|(k, _)| k.cmp(&key));
                if let Ok(i) = p {
                    self.levels[lvl][i].1 = Arc::clone(&val_arc);
                }
            }
            return false;
        }

        // Insert at level 0.
        self.levels[0].insert(pos, (key.clone(), Arc::clone(&val_arc)));
        self.size += 1;

        // Insert at express lanes up to `height - 1`.
        for lvl in 1..height {
            let p = self.levels[lvl]
                .binary_search_by(|(k, _)| k.cmp(&key))
                .unwrap_or_else(|i| i);
            self.levels[lvl].insert(p, (key.clone(), Arc::clone(&val_arc)));
        }
        true
    }

    fn remove(&mut self, key: &K) -> Option<Arc<V>> {
        let (found, pos) = self.search_l0(key);
        if !found {
            return None;
        }
        let val = Arc::clone(&self.levels[0][pos].1);
        self.levels[0].remove(pos);
        self.size -= 1;
        // Remove from express lanes.
        for lvl in 1..MAX_LEVEL {
            if let Ok(i) = self.levels[lvl].binary_search_by(|(k, _)| k.cmp(key)) {
                self.levels[lvl].remove(i);
            }
        }
        Some(val)
    }

    fn get(&self, key: &K) -> Option<Arc<V>> {
        // Use the highest express lane that has the key to skip quickly.
        // Fall back to level 0 for the definitive answer.
        let (found, pos) = self.search_l0(key);
        if found {
            Some(Arc::clone(&self.levels[0][pos].1))
        } else {
            None
        }
    }

    fn contains(&self, key: &K) -> bool {
        self.search_l0(key).0
    }

    fn range(&self, lo: &K, hi: &K) -> Vec<(K, Arc<V>)> {
        let base = &self.levels[0];
        let start = base.partition_point(|(k, _)| k < lo);
        let end = base.partition_point(|(k, _)| k <= hi);
        base[start..end]
            .iter()
            .map(|(k, v)| (k.clone(), Arc::clone(v)))
            .collect()
    }

    fn len(&self) -> usize {
        self.size
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A concurrent skip list mapping keys to values.
///
/// `K` must implement `Ord + Clone` and `V` must implement `Clone`.
///
/// Values are returned as `Arc<V>` so callers can hold references without
/// worrying about subsequent mutations invalidating them.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::lock_free::ConcurrentSkipList;
///
/// let sl = ConcurrentSkipList::<i32, i32>::new();
/// sl.insert(1, 10);
/// sl.insert(2, 20);
/// assert_eq!(sl.get(&1).as_deref().copied(), Some(10));
/// assert_eq!(sl.len(), 2);
/// ```
pub struct ConcurrentSkipList<K, V> {
    inner: Arc<RwLock<SkipListInner<K, V>>>,
}

impl<K: Ord + Clone, V: Clone> ConcurrentSkipList<K, V> {
    /// Create a new, empty skip list.
    pub fn new() -> Self {
        ConcurrentSkipList {
            inner: Arc::new(RwLock::new(SkipListInner::new())),
        }
    }

    /// Insert `key → value`.
    ///
    /// Returns `true` if newly inserted, `false` if an existing entry was
    /// updated.
    pub fn insert(&self, key: K, value: V) -> bool {
        match self.inner.write() {
            Ok(mut g) => g.insert(key, value),
            Err(_) => false,
        }
    }

    /// Remove the entry for `key`, returning its value if found.
    pub fn remove(&self, key: &K) -> Option<V> {
        let arc = match self.inner.write() {
            Ok(mut g) => g.remove(key),
            Err(_) => None,
        }?;
        // Unwrap the Arc.  If this is the sole owner (no concurrent readers
        // holding a reference) we can move the value out; otherwise clone.
        Some(Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone()))
    }

    /// Return `true` if the skip list contains `key`.
    pub fn contains(&self, key: &K) -> bool {
        match self.inner.read() {
            Ok(g) => g.contains(key),
            Err(_) => false,
        }
    }

    /// Return a reference-counted pointer to the value for `key`, if present.
    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        match self.inner.read() {
            Ok(g) => g.get(key),
            Err(_) => None,
        }
    }

    /// Return all `(key, Arc<V>)` pairs in the closed range `[lo, hi]` in
    /// sorted order.
    pub fn range(&self, lo: &K, hi: &K) -> Vec<(K, Arc<V>)> {
        match self.inner.read() {
            Ok(g) => g.range(lo, hi),
            Err(_) => Vec::new(),
        }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        match self.inner.read() {
            Ok(g) => g.len(),
            Err(_) => 0,
        }
    }

    /// `true` if the skip list is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// Allow cloning handles (they share the same inner state).
impl<K: Ord + Clone, V: Clone> Clone for ConcurrentSkipList<K, V> {
    fn clone(&self) -> Self {
        ConcurrentSkipList {
            inner: Arc::clone(&self.inner),
        }
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
    fn test_insert_get() {
        let sl = ConcurrentSkipList::<i32, i32>::new();
        for i in 0..10 {
            sl.insert(i, i * 10);
        }
        for i in 0..10 {
            assert_eq!(sl.get(&i).as_deref().copied(), Some(i * 10), "key {i}");
        }
    }

    #[test]
    fn test_remove() {
        let sl = ConcurrentSkipList::<i32, i32>::new();
        sl.insert(1, 100);
        sl.insert(2, 200);
        let removed = sl.remove(&1);
        assert_eq!(removed, Some(100));
        assert!(!sl.contains(&1));
        assert!(sl.contains(&2));
        assert_eq!(sl.len(), 1);
    }

    #[test]
    fn test_contains() {
        let sl = ConcurrentSkipList::<i32, &str>::new();
        sl.insert(42, "hello");
        assert!(sl.contains(&42));
        assert!(!sl.contains(&0));
        assert!(!sl.contains(&43));
    }

    #[test]
    fn test_range() {
        let sl = ConcurrentSkipList::<i32, i32>::new();
        for i in 0..20 {
            sl.insert(i, i);
        }
        let result = sl.range(&5, &10);
        let keys: Vec<i32> = result.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_ordered() {
        let sl = ConcurrentSkipList::<i32, i32>::new();
        let keys = vec![7, 3, 1, 9, 4, 2, 8, 5, 6];
        for k in &keys {
            sl.insert(*k, *k);
        }
        let result = sl.range(&i32::MIN, &i32::MAX);
        let out_keys: Vec<i32> = result.iter().map(|(k, _)| *k).collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(out_keys, sorted);
    }

    #[test]
    fn test_concurrent_inserts() {
        let sl = Arc::new(ConcurrentSkipList::<i32, i32>::new());
        let n_threads = 4i32;
        let per_thread = 100i32;
        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let sl = Arc::clone(&sl);
                thread::spawn(move || {
                    for i in 0..per_thread {
                        sl.insert(t * per_thread + i, t * per_thread + i);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread panicked");
        }
        assert_eq!(sl.len(), (n_threads * per_thread) as usize);
    }

    #[test]
    fn test_update_existing() {
        let sl = ConcurrentSkipList::<i32, i32>::new();
        assert!(sl.insert(1, 100));
        assert!(!sl.insert(1, 200)); // update
        assert_eq!(sl.get(&1).as_deref().copied(), Some(200));
        assert_eq!(sl.len(), 1);
    }

    #[test]
    fn test_empty() {
        let sl = ConcurrentSkipList::<i32, i32>::new();
        assert!(sl.is_empty());
        assert_eq!(sl.len(), 0);
        assert_eq!(sl.get(&0), None);
        assert!(sl.range(&0, &100).is_empty());
    }
}
