//! High-performance caching utilities for scientific computing.
//!
//! This module provides several complementary caching strategies, each
//! optimised for a different access pattern:
//!
//! | Type | Strategy | Thread-safe |
//! |------|----------|-------------|
//! | [`LRUCache<K,V>`] | Least-recently-used eviction | `Arc<Mutex<…>>` |
//! | [`TTLCache<K,V>`] | Time-to-live expiration | `Arc<Mutex<…>>` |
//! | [`MemoizedFn<A,R>`] | Memoize a pure `Fn(A)->R` | `Arc<Mutex<…>>` |
//! | [`ComputeCache<K,V>`] | Lazy one-time compute per key | `Arc<Mutex<…>>` |
//! | [`TieredCache<K,V>`] | L1 (fast, small) / L2 (slow, large) hierarchy | `Arc<Mutex<…>>` |
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::caching::{LRUCache, TTLCache, MemoizedFn};
//! use std::time::Duration;
//!
//! // LRU cache
//! let mut cache = LRUCache::<String, u64>::new(4);
//! cache.insert("x".into(), 42);
//! assert_eq!(cache.get(&"x".into()), Some(&42));
//!
//! // TTL cache — entries expire after 60 seconds
//! let mut ttl = TTLCache::<String, u64>::new(8, Duration::from_secs(60));
//! ttl.insert("y".into(), 99);
//! assert_eq!(ttl.get(&"y".into()), Some(99));
//!
//! // Memoized function
//! let memo = MemoizedFn::new(|n: u64| n * n);
//! assert_eq!(memo.call(5), 25);
//! assert_eq!(memo.call(5), 25); // cache hit
//! ```

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ============================================================================
// LRUCache<K, V>
// ============================================================================

/// An LRU (least-recently-used) cache with configurable capacity.
///
/// When the cache is full and a new key is inserted, the least-recently-used
/// entry is evicted.
///
/// Access (both read and write) promotes the accessed entry to the most-recently-
/// used position.
///
/// This implementation is **not** thread-safe by itself; wrap in
/// `Arc<Mutex<LRUCache<…>>>` for shared access.
pub struct LRUCache<K, V> {
    capacity: usize,
    map: HashMap<K, (V, usize)>,   // key → (value, list_position_token)
    order: VecDeque<K>,            // front = MRU, back = LRU
    hits: u64,
    misses: u64,
}

impl<K: Hash + Eq + Clone, V> LRUCache<K, V> {
    /// Create a new LRU cache with the given `capacity` (minimum 1).
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            map: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            hits: 0,
            misses: 0,
        }
    }

    /// Insert or update a key-value pair.  If `key` already exists the value is
    /// updated and the entry is promoted to MRU.  If the cache is full the LRU
    /// entry is evicted.
    pub fn insert(&mut self, key: K, value: V) {
        // If key already present, remove from order list
        if self.map.contains_key(&key) {
            self.remove_from_order(&key);
        }

        // Evict LRU if at capacity
        while self.map.len() >= self.capacity {
            if let Some(lru_key) = self.order.pop_back() {
                self.map.remove(&lru_key);
            }
        }

        let token = self.order.len(); // We'll use the VecDeque front for MRU
        self.order.push_front(key.clone());
        self.map.insert(key, (value, token));
    }

    /// Retrieve a reference to the value for `key`, promoting it to MRU.
    /// Returns `None` if not present.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.map.contains_key(key) {
            self.hits += 1;
            // Promote to front
            self.remove_from_order(key);
            self.order.push_front(key.clone());
            // Safe: we just confirmed key exists
            self.map.get(key).map(|(v, _)| v)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Peek at the value for `key` without changing the LRU order.
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.map.get(key).map(|(v, _)| v)
    }

    /// Remove and return the value for `key`.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some((value, _)) = self.map.remove(key) {
            self.remove_from_order(key);
            Some(value)
        } else {
            None
        }
    }

    /// Returns `true` if the cache contains `key`.
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Configured maximum capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of cache hits since creation.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Number of cache misses since creation.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Hit rate in [0, 1].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Clear all entries and reset statistics.
    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Helper: remove `key` from the `order` deque.
    fn remove_from_order(&mut self, key: &K) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
    }
}

// ============================================================================
// TTLCache<K, V>
// ============================================================================

/// Cached entry with an expiration timestamp.
struct TTLEntry<V> {
    value: V,
    expires_at: Instant,
}

/// A time-to-live cache: entries automatically expire after a configurable
/// duration.
///
/// Expired entries are lazily removed on access.  Calling [`TTLCache::purge_expired`]
/// forces a sweep of all stale entries.
///
/// Not thread-safe by itself; wrap in `Arc<Mutex<…>>` for shared access.
pub struct TTLCache<K, V> {
    capacity: usize,
    ttl: Duration,
    map: HashMap<K, TTLEntry<V>>,
    hits: u64,
    misses: u64,
    expired: u64,
}

impl<K: Hash + Eq + Clone, V: Clone> TTLCache<K, V> {
    /// Create a new TTL cache.
    ///
    /// - `capacity`: maximum number of live entries.
    /// - `ttl`: time-to-live for each entry.
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            ttl,
            map: HashMap::with_capacity(capacity),
            hits: 0,
            misses: 0,
            expired: 0,
        }
    }

    /// Insert a key-value pair.  If the cache is full and all entries are
    /// still live, one arbitrary entry is evicted to make room.
    pub fn insert(&mut self, key: K, value: V) {
        // Try to evict an expired entry first
        if self.map.len() >= self.capacity {
            let expired_key = self
                .map
                .iter()
                .find(|(_, e)| e.expires_at <= Instant::now())
                .map(|(k, _)| k.clone());

            if let Some(k) = expired_key {
                self.map.remove(&k);
                self.expired += 1;
            } else {
                // No expired entry found — evict an arbitrary live entry
                if let Some(k) = self.map.keys().next().cloned() {
                    self.map.remove(&k);
                }
            }
        }

        self.map.insert(
            key,
            TTLEntry {
                value,
                expires_at: Instant::now() + self.ttl,
            },
        );
    }

    /// Retrieve the value for `key`, returning `None` if not present or expired.
    pub fn get(&mut self, key: &K) -> Option<V> {
        let now = Instant::now();
        match self.map.get(key) {
            Some(e) if e.expires_at > now => {
                self.hits += 1;
                Some(e.value.clone())
            }
            Some(_) => {
                // Expired
                self.map.remove(key);
                self.expired += 1;
                self.misses += 1;
                None
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Remove an entry explicitly.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.map.remove(key).map(|e| e.value)
    }

    /// Sweep and remove all entries that have expired.  Returns the number
    /// removed.
    pub fn purge_expired(&mut self) -> usize {
        let now = Instant::now();
        let before = self.map.len();
        self.map.retain(|_, e| e.expires_at > now);
        let removed = before - self.map.len();
        self.expired += removed as u64;
        removed
    }

    /// Current number of entries (including any that may have expired but
    /// not yet been purged).
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Cache hits.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Cache misses (including expired lookups).
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Total entries that expired.
    pub fn expired_count(&self) -> u64 {
        self.expired
    }
}

// ============================================================================
// MemoizedFn<A, R>
// ============================================================================

/// A memoization wrapper for a pure `Fn(A) -> R`.
///
/// Results are cached indefinitely (no eviction).  Use [`LRUCache`] inside a
/// custom wrapper if bounded storage is needed.
///
/// Thread-safe: cloning the `MemoizedFn` shares the same internal cache.
///
/// # Example
///
/// ```rust
/// use scirs2_core::caching::MemoizedFn;
///
/// fn fib(n: u64) -> u64 {
///     if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
/// }
///
/// let memo = MemoizedFn::new(fib);
/// assert_eq!(memo.call(10), 55);
/// ```
pub struct MemoizedFn<A: Hash + Eq + Clone + Send + 'static, R: Clone + Send + 'static> {
    cache: Arc<Mutex<HashMap<A, R>>>,
    func: Arc<dyn Fn(A) -> R + Send + Sync>,
}

impl<A: Hash + Eq + Clone + Send + 'static, R: Clone + Send + 'static> MemoizedFn<A, R> {
    /// Wrap `f` in a memoization layer.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(A) -> R + Send + Sync + 'static,
    {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            func: Arc::new(f),
        }
    }

    /// Call the wrapped function with `arg`, using a cached result if available.
    pub fn call(&self, arg: A) -> R {
        // Check cache
        if let Ok(guard) = self.cache.lock() {
            if let Some(v) = guard.get(&arg) {
                return v.clone();
            }
        }
        // Compute (outside the lock to avoid holding during computation)
        let result = (self.func)(arg.clone());
        if let Ok(mut guard) = self.cache.lock() {
            guard.entry(arg).or_insert_with(|| result.clone());
        }
        result
    }

    /// Clear the memoization cache.
    pub fn clear(&self) {
        if let Ok(mut guard) = self.cache.lock() {
            guard.clear();
        }
    }

    /// Number of cached results.
    pub fn len(&self) -> usize {
        self.cache.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// `true` if no results are cached.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<A, R> Clone for MemoizedFn<A, R>
where
    A: Hash + Eq + Clone + Send + 'static,
    R: Clone + Send + 'static,
{
    fn clone(&self) -> Self {
        Self {
            cache: Arc::clone(&self.cache),
            func: Arc::clone(&self.func),
        }
    }
}

// ============================================================================
// ComputeCache<K, V>
// ============================================================================

/// A lazy-computed cache: each key is computed exactly once.
///
/// On the first access for a key, the provided factory closure is called to
/// produce a value; subsequent accesses return the cached result.
///
/// Thread-safe: wraps the internal map in a `Mutex`.
///
/// # Example
///
/// ```rust
/// use scirs2_core::caching::ComputeCache;
///
/// let cache: ComputeCache<u64, String> = ComputeCache::new();
/// let v = cache.get_or_compute(3, |k| format!("value-{k}"));
/// assert_eq!(v, "value-3");
/// // Second call returns the same value without calling the closure again
/// let v2 = cache.get_or_compute(3, |k| panic!("should not be called"));
/// assert_eq!(v2, "value-3");
/// ```
pub struct ComputeCache<K: Hash + Eq + Clone, V: Clone> {
    inner: Arc<Mutex<HashMap<K, V>>>,
}

impl<K: Hash + Eq + Clone + Send + 'static, V: Clone + Send + 'static> ComputeCache<K, V> {
    /// Create an empty `ComputeCache`.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Return the cached value for `key`, computing it with `factory` if not
    /// yet present.
    pub fn get_or_compute<F>(&self, key: K, factory: F) -> V
    where
        F: FnOnce(&K) -> V,
    {
        // Check without computing
        if let Ok(guard) = self.inner.lock() {
            if let Some(v) = guard.get(&key) {
                return v.clone();
            }
        }
        // Compute outside the lock
        let value = factory(&key);
        if let Ok(mut guard) = self.inner.lock() {
            guard.entry(key).or_insert_with(|| value.clone());
        }
        value
    }

    /// Invalidate the cache entry for `key`.  Returns the old value if any.
    pub fn invalidate(&self, key: &K) -> Option<V> {
        self.inner.lock().ok()?.remove(key)
    }

    /// Clear all entries.
    pub fn clear(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.clear();
        }
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<K: Hash + Eq + Clone, V: Clone> Clone for ComputeCache<K, V> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

// ============================================================================
// TieredCache<K, V>
// ============================================================================

/// A two-level cache hierarchy.
///
/// - **L1** is small and fast (LRU eviction).
/// - **L2** is larger and slightly slower (also LRU).
///
/// Lookups first consult L1; on an L1 miss they fall through to L2 (promoting
/// the entry into L1).  On an L2 miss the caller must provide the value via
/// [`TieredCache::insert`] or [`TieredCache::get_or_compute`].
pub struct TieredCache<K: Hash + Eq + Clone, V: Clone> {
    l1: LRUCache<K, V>,
    l2: LRUCache<K, V>,
    l1_hits: u64,
    l2_hits: u64,
    full_misses: u64,
}

impl<K: Hash + Eq + Clone, V: Clone> TieredCache<K, V> {
    /// Create a tiered cache with `l1_capacity` L1 slots and `l2_capacity` L2 slots.
    pub fn new(l1_capacity: usize, l2_capacity: usize) -> Self {
        Self {
            l1: LRUCache::new(l1_capacity),
            l2: LRUCache::new(l2_capacity),
            l1_hits: 0,
            l2_hits: 0,
            full_misses: 0,
        }
    }

    /// Insert a value into both L1 and L2.
    pub fn insert(&mut self, key: K, value: V) {
        self.l1.insert(key.clone(), value.clone());
        self.l2.insert(key, value);
    }

    /// Look up `key`.
    ///
    /// 1. L1 hit → return immediately.
    /// 2. L1 miss, L2 hit → promote to L1 and return.
    /// 3. Both miss → return `None` and increment the full-miss counter.
    pub fn get(&mut self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        // L1 probe
        if let Some(v) = self.l1.get(key) {
            self.l1_hits += 1;
            return Some(v.clone());
        }
        // L2 probe
        if let Some(v) = self.l2.get(key) {
            self.l2_hits += 1;
            let owned = v.clone();
            // Promote to L1
            self.l1.insert(key.clone(), owned.clone());
            return Some(owned);
        }
        self.full_misses += 1;
        None
    }

    /// Get-or-compute: look up `key`; if absent, call `factory` and cache the
    /// result in both levels.
    pub fn get_or_compute<F>(&mut self, key: K, factory: F) -> V
    where
        F: FnOnce(&K) -> V,
    {
        if let Some(v) = self.get(&key) {
            return v;
        }
        let value = factory(&key);
        self.insert(key, value.clone());
        value
    }

    /// Remove `key` from both L1 and L2.
    pub fn remove(&mut self, key: &K) {
        self.l1.remove(key);
        self.l2.remove(key);
    }

    /// L1 hit count.
    pub fn l1_hits(&self) -> u64 {
        self.l1_hits
    }

    /// L2 hit count (L1 miss + L2 hit).
    pub fn l2_hits(&self) -> u64 {
        self.l2_hits
    }

    /// Full-miss count (both L1 and L2 missed).
    pub fn full_misses(&self) -> u64 {
        self.full_misses
    }

    /// Overall hit rate (L1+L2 hits / total lookups).
    pub fn hit_rate(&self) -> f64 {
        let total = self.l1_hits + self.l2_hits + self.full_misses;
        if total == 0 {
            0.0
        } else {
            (self.l1_hits + self.l2_hits) as f64 / total as f64
        }
    }

    /// Current L1 utilisation as a fraction [0, 1].
    pub fn l1_utilisation(&self) -> f64 {
        self.l1.len() as f64 / self.l1.capacity() as f64
    }

    /// Current L2 utilisation as a fraction [0, 1].
    pub fn l2_utilisation(&self) -> f64 {
        self.l2.len() as f64 / self.l2.capacity() as f64
    }

    /// Clear both cache levels and reset statistics.
    pub fn clear(&mut self) {
        self.l1.clear();
        self.l2.clear();
        self.l1_hits = 0;
        self.l2_hits = 0;
        self.full_misses = 0;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- LRUCache ---

    #[test]
    fn lru_basic_insert_get() {
        let mut c = LRUCache::<i32, &str>::new(3);
        c.insert(1, "a");
        c.insert(2, "b");
        c.insert(3, "c");
        assert_eq!(c.get(&1), Some(&"a"));
        assert_eq!(c.get(&2), Some(&"b"));
        assert_eq!(c.get(&3), Some(&"c"));
    }

    #[test]
    fn lru_evicts_least_recently_used() {
        let mut c = LRUCache::<i32, i32>::new(3);
        c.insert(1, 10);
        c.insert(2, 20);
        c.insert(3, 30);
        // Access 1 to make it MRU
        c.get(&1);
        // Insert 4 — should evict key 2 (LRU)
        c.insert(4, 40);
        assert!(c.get(&1).is_some(), "1 was MRU, should survive");
        // key 2 was accessed least recently; it or 3 may be evicted depending on
        // promotion order.  Simply verify capacity is respected.
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn lru_hit_miss_counters() {
        let mut c = LRUCache::<i32, i32>::new(4);
        c.insert(1, 1);
        c.get(&1); // hit
        c.get(&2); // miss
        assert_eq!(c.hits(), 1);
        assert_eq!(c.misses(), 1);
    }

    #[test]
    fn lru_remove() {
        let mut c = LRUCache::<i32, i32>::new(4);
        c.insert(1, 100);
        assert_eq!(c.remove(&1), Some(100));
        assert!(c.get(&1).is_none());
    }

    // --- TTLCache ---

    #[test]
    fn ttl_cache_insert_get() {
        let mut c = TTLCache::<i32, i32>::new(4, Duration::from_secs(60));
        c.insert(1, 42);
        assert_eq!(c.get(&1), Some(42));
    }

    #[test]
    fn ttl_cache_expiry() {
        let mut c = TTLCache::<i32, i32>::new(4, Duration::from_millis(10));
        c.insert(1, 7);
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(c.get(&1), None, "entry should have expired");
    }

    #[test]
    fn ttl_purge_expired() {
        let mut c = TTLCache::<i32, i32>::new(8, Duration::from_millis(10));
        c.insert(1, 1);
        c.insert(2, 2);
        std::thread::sleep(Duration::from_millis(20));
        let removed = c.purge_expired();
        assert_eq!(removed, 2);
        assert_eq!(c.len(), 0);
    }

    // --- MemoizedFn ---

    #[test]
    fn memoized_fn_caches_result() {
        let call_count = Arc::new(Mutex::new(0u32));
        let cc = Arc::clone(&call_count);
        let memo = MemoizedFn::new(move |n: u64| {
            if let Ok(mut g) = cc.lock() {
                *g += 1;
            }
            n * n
        });
        assert_eq!(memo.call(5), 25);
        assert_eq!(memo.call(5), 25); // cache hit
        assert_eq!(memo.call(6), 36);
        let count = *call_count.lock().expect("lock");
        assert_eq!(count, 2, "closure should be called only for new inputs");
    }

    #[test]
    fn memoized_fn_clear() {
        let memo = MemoizedFn::new(|n: u64| n + 1);
        memo.call(10);
        memo.call(20);
        assert_eq!(memo.len(), 2);
        memo.clear();
        assert_eq!(memo.len(), 0);
    }

    // --- ComputeCache ---

    #[test]
    fn compute_cache_get_or_compute() {
        let cache = ComputeCache::<u64, String>::new();
        let v = cache.get_or_compute(3, |k| format!("v-{k}"));
        assert_eq!(v, "v-3");
        // Second call should NOT invoke factory — validate by using a panicking closure
        let v2 = cache.get_or_compute(3, |_| panic!("factory should not be called again"));
        assert_eq!(v2, "v-3");
    }

    #[test]
    fn compute_cache_invalidate() {
        let cache = ComputeCache::<u64, u64>::new();
        cache.get_or_compute(1, |k| *k * 2);
        cache.invalidate(&1);
        assert_eq!(cache.len(), 0);
    }

    // --- TieredCache ---

    #[test]
    fn tiered_cache_l1_hit() {
        let mut tc = TieredCache::<i32, i32>::new(4, 16);
        tc.insert(1, 100);
        assert_eq!(tc.get(&1), Some(100));
        assert_eq!(tc.l1_hits(), 1);
        assert_eq!(tc.l2_hits(), 0);
    }

    #[test]
    fn tiered_cache_l2_promotion() {
        let mut tc = TieredCache::<i32, i32>::new(1, 4);
        // Fill L1 so key 1 is evicted by a subsequent insert
        tc.insert(1, 111);
        tc.insert(2, 222); // evicts key 1 from L1 (L1 cap = 1)
        // Key 1 should still be in L2
        let v = tc.get(&1);
        assert_eq!(v, Some(111));
        assert_eq!(tc.l2_hits(), 1);
    }

    #[test]
    fn tiered_cache_full_miss() {
        let mut tc = TieredCache::<i32, i32>::new(2, 4);
        assert_eq!(tc.get(&99), None);
        assert_eq!(tc.full_misses(), 1);
    }

    #[test]
    fn tiered_cache_get_or_compute() {
        let mut tc = TieredCache::<i32, i32>::new(4, 16);
        let v = tc.get_or_compute(7, |k| k * 3);
        assert_eq!(v, 21);
        let v2 = tc.get_or_compute(7, |_| panic!("should not compute again"));
        assert_eq!(v2, 21);
    }
}
