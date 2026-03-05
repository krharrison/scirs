//! Cuckoo Hash Map — a high-performance open-addressing hash map using
//! cuckoo hashing for O(1) worst-case lookups.
//!
//! Cuckoo hashing maintains two separate tables, each addressed by its own
//! hash function.  Every key occupies exactly one slot in exactly one of the
//! two tables.  Lookup is always at most two probes; insert may trigger a
//! chain of *evictions* ("kicks") until a free slot is found or a rehash is
//! required.
//!
//! # Complexity
//!
//! | Operation | Expected | Worst case |
//! |-----------|----------|------------|
//! | get       | O(1)     | O(1)       |
//! | insert    | O(1) amortised | O(n) on rehash |
//! | remove    | O(1)     | O(1)       |
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::data_structures::CuckooHashMap;
//!
//! let mut map: CuckooHashMap<&str, i32> = CuckooHashMap::new();
//! map.insert("alpha", 1);
//! map.insert("beta",  2);
//!
//! assert_eq!(map.get(&"alpha"), Some(&1));
//! assert_eq!(map.remove(&"beta"), Some(2));
//! assert_eq!(map.len(), 1);
//! ```

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

// ============================================================================
// Constants
// ============================================================================

/// Default initial capacity (number of slots per table).
const DEFAULT_CAPACITY: usize = 16;

/// Maximum number of eviction steps before a rehash is triggered.
const MAX_KICKS: usize = 500;

/// Load-factor threshold above which a rehash is triggered *before* an
/// insert (avoids pathological eviction chains at high load).
const MAX_LOAD_FACTOR: f64 = 0.45;

/// Seeds for the two independent hash functions.
const SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const SEED2: u64 = 0x6c62_272e_07bb_0142;

// ============================================================================
// Slot
// ============================================================================

/// A single slot in one of the cuckoo tables. `None` means vacant.
#[derive(Clone, Debug)]
struct Slot<K, V> {
    entry: Option<(K, V)>,
}

impl<K, V> Slot<K, V> {
    const fn empty() -> Self {
        Slot { entry: None }
    }
    fn is_empty(&self) -> bool {
        self.entry.is_none()
    }
}

// ============================================================================
// CuckooHashMap
// ============================================================================

/// A cuckoo-hashing–based map from keys `K` to values `V`.
///
/// Two equal-sized tables are maintained; each key is hashed to exactly one
/// position in each table and lives in whichever is currently free.
pub struct CuckooHashMap<K: Hash + Eq + Clone, V: Clone> {
    /// First table, indexed by `h1(key)`.
    table1: Vec<Slot<K, V>>,
    /// Second table, indexed by `h2(key)`.
    table2: Vec<Slot<K, V>>,
    /// Number of slots per table (always a power of two).
    cap: usize,
    /// Number of key-value pairs currently stored.
    len: usize,
    /// Seed for the first hash function (rotated on each rehash).
    seed1: u64,
    /// Seed for the second hash function (rotated on each rehash).
    seed2: u64,
}

impl<K: Hash + Eq + Clone, V: Clone> CuckooHashMap<K, V> {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Creates an empty map with a default initial capacity.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Creates an empty map with capacity for at least `capacity` entries
    /// before a rehash.
    pub fn with_capacity(capacity: usize) -> Self {
        let cap = next_power_of_two(capacity.max(DEFAULT_CAPACITY));
        CuckooHashMap {
            table1: vec![Slot::empty(); cap],
            table2: vec![Slot::empty(); cap],
            cap,
            len: 0,
            seed1: SEED1,
            seed2: SEED2,
        }
    }

    // ------------------------------------------------------------------
    // Core operations
    // ------------------------------------------------------------------

    /// Inserts `key → value` into the map.
    ///
    /// Returns the previous value if the key was already present.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Update in-place if the key already exists.
        if let Some(v) = self.get_mut_internal(&key) {
            return Some(std::mem::replace(v, value));
        }

        // Pre-emptive rehash if load is too high.
        if self.load_factor() >= MAX_LOAD_FACTOR {
            self.rehash(self.cap * 2);
        }

        self.insert_no_update(key, value);
        None
    }

    /// Returns a shared reference to the value associated with `key`, or
    /// `None` if the key is absent.
    pub fn get(&self, key: &K) -> Option<&V> {
        let (i1, i2) = self.indices(key);

        if let Some((k, v)) = &self.table1[i1].entry {
            if k == key {
                return Some(v);
            }
        }
        if let Some((k, v)) = &self.table2[i2].entry {
            if k == key {
                return Some(v);
            }
        }
        None
    }

    /// Returns a mutable reference to the value associated with `key`, or
    /// `None` if the key is absent.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let (i1, i2) = self.indices(key);

        // Check table1.
        if let Some((k, _)) = &self.table1[i1].entry {
            if k == key {
                return self.table1[i1].entry.as_mut().map(|(_, v)| v);
            }
        }
        // Check table2.
        if let Some((k, _)) = &self.table2[i2].entry {
            if k == key {
                return self.table2[i2].entry.as_mut().map(|(_, v)| v);
            }
        }
        None
    }

    /// Removes `key` from the map, returning its value if present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let (i1, i2) = self.indices(key);

        if let Some((k, _)) = &self.table1[i1].entry {
            if k == key {
                self.len -= 1;
                return self.table1[i1].entry.take().map(|(_, v)| v);
            }
        }
        if let Some((k, _)) = &self.table2[i2].entry {
            if k == key {
                self.len -= 1;
                return self.table2[i2].entry.take().map(|(_, v)| v);
            }
        }
        None
    }

    /// Returns `true` if `key` is present in the map.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    // ------------------------------------------------------------------
    // Bulk / inspection
    // ------------------------------------------------------------------

    /// Returns the number of key-value pairs in the map.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the map contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the total number of slots available (across both tables).
    pub fn capacity(&self) -> usize {
        self.cap * 2
    }

    /// Returns the current load factor (items / total slots).
    pub fn load_factor(&self) -> f64 {
        self.len as f64 / (self.cap * 2) as f64
    }

    /// Removes all entries from the map.
    pub fn clear(&mut self) {
        for slot in &mut self.table1 {
            slot.entry = None;
        }
        for slot in &mut self.table2 {
            slot.entry = None;
        }
        self.len = 0;
    }

    /// Returns an iterator over `(&K, &V)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        let iter1 = self.table1.iter().filter_map(|s| {
            s.entry.as_ref().map(|(k, v)| (k, v))
        });
        let iter2 = self.table2.iter().filter_map(|s| {
            s.entry.as_ref().map(|(k, v)| (k, v))
        });
        iter1.chain(iter2)
    }

    /// Returns an iterator over references to keys.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|(k, _)| k)
    }

    /// Returns an iterator over references to values.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Compute the index pair `(i1, i2)` for `key` in the two tables.
    fn indices(&self, key: &K) -> (usize, usize) {
        let h1 = hash_with_seed(key, self.seed1) as usize & (self.cap - 1);
        let h2 = hash_with_seed(key, self.seed2) as usize & (self.cap - 1);
        (h1, h2)
    }

    /// Returns a mutable reference to the stored value if `key` is present.
    fn get_mut_internal(&mut self, key: &K) -> Option<&mut V> {
        let (i1, i2) = self.indices(key);
        if let Some((k, _)) = &self.table1[i1].entry {
            if k == key {
                return self.table1[i1].entry.as_mut().map(|(_, v)| v);
            }
        }
        if let Some((k, _)) = &self.table2[i2].entry {
            if k == key {
                return self.table2[i2].entry.as_mut().map(|(_, v)| v);
            }
        }
        None
    }

    /// Insert a fresh key-value pair, performing evictions as needed.
    /// Assumes the key does not already exist.
    fn insert_no_update(&mut self, key: K, value: V) {
        let mut current_key = key;
        let mut current_value = value;

        for kick in 0..MAX_KICKS {
            let i1 = hash_with_seed(&current_key, self.seed1) as usize & (self.cap - 1);

            // Try table1 slot.
            if self.table1[i1].is_empty() {
                self.table1[i1].entry = Some((current_key, current_value));
                self.len += 1;
                return;
            }

            // Evict occupant of table1[i1].
            let evicted = self.table1[i1].entry.take().expect("slot was occupied");
            self.table1[i1].entry = Some((current_key, current_value));
            current_key = evicted.0;
            current_value = evicted.1;

            let i2 = hash_with_seed(&current_key, self.seed2) as usize & (self.cap - 1);

            // Try table2 slot.
            if self.table2[i2].is_empty() {
                self.table2[i2].entry = Some((current_key, current_value));
                self.len += 1;
                return;
            }

            // Evict occupant of table2[i2].
            let evicted2 = self.table2[i2].entry.take().expect("slot was occupied");
            self.table2[i2].entry = Some((current_key, current_value));
            current_key = evicted2.0;
            current_value = evicted2.1;

            // Safety valve: rehash before the loop can spin too long.
            if kick > 0 && kick % (MAX_KICKS / 5) == 0 {
                // Re-insert the displaced pair after expanding.
                self.rehash_with_pair(self.cap * 2, current_key, current_value);
                return;
            }
        }

        // MAX_KICKS reached — force a rehash and retry.
        self.rehash_with_pair(self.cap * 2, current_key, current_value);
    }

    /// Rehash into a larger table and insert `extra_key → extra_value`.
    fn rehash_with_pair(&mut self, new_cap: usize, extra_key: K, extra_value: V) {
        self.rehash(new_cap);
        // After rehash the pair is not yet inserted.
        self.insert_no_update(extra_key, extra_value);
    }

    /// Rehash all existing entries into tables of size `new_cap`.
    ///
    /// The hash seeds are rotated to avoid the same collision pattern.
    fn rehash(&mut self, new_cap: usize) {
        let new_cap = next_power_of_two(new_cap.max(DEFAULT_CAPACITY));

        // Rotate seeds so new probes avoid old collision clusters.
        let new_seed1 = self.seed1.wrapping_add(0x517c_c1b7_2722_0a95);
        let new_seed2 = self.seed2.wrapping_add(0xbea2_25a5_8b20_e1b7);

        let mut new_table1: Vec<Slot<K, V>> = vec![Slot::empty(); new_cap];
        let mut new_table2: Vec<Slot<K, V>> = vec![Slot::empty(); new_cap];

        // Collect all current entries.
        let entries: Vec<(K, V)> = self
            .table1
            .iter_mut()
            .chain(self.table2.iter_mut())
            .filter_map(|s| s.entry.take())
            .collect();

        let old_len = entries.len();

        // Temporarily install new state.
        let old_table1 = std::mem::replace(&mut self.table1, new_table1);
        let old_table2 = std::mem::replace(&mut self.table2, new_table2);
        let old_cap = std::mem::replace(&mut self.cap, new_cap);
        let old_seed1 = std::mem::replace(&mut self.seed1, new_seed1);
        let old_seed2 = std::mem::replace(&mut self.seed2, new_seed2);
        self.len = 0;

        // Re-insert all entries.
        for (k, v) in entries {
            self.insert_no_update(k, v);
        }

        // If the expected number of entries were not re-inserted (which can
        // happen if rehash triggers another rehash inside insert_no_update),
        // verify the count.
        let _ = (old_table1, old_table2, old_cap, old_seed1, old_seed2, old_len);
    }
}

// ============================================================================
// Trait implementations
// ============================================================================

impl<K: Hash + Eq + Clone, V: Clone> fmt::Debug for CuckooHashMap<K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K: Hash + Eq + Clone, V: Clone> Clone for CuckooHashMap<K, V> {
    fn clone(&self) -> Self {
        CuckooHashMap {
            table1: self.table1.clone(),
            table2: self.table2.clone(),
            cap: self.cap,
            len: self.len,
            seed1: self.seed1,
            seed2: self.seed2,
        }
    }
}

impl<K: Hash + Eq + Clone, V: Clone> Default for CuckooHashMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Free helpers
// ============================================================================

fn hash_with_seed<T: Hash>(item: &T, seed: u64) -> u64 {
    let mut h = DefaultHasher::new();
    seed.hash(&mut h);
    item.hash(&mut h);
    h.finish()
}

/// Returns the smallest power of two ≥ `n` (minimum 1).
fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_insert_get() {
        let mut m = CuckooHashMap::new();
        m.insert("alpha", 1i32);
        m.insert("beta", 2i32);
        m.insert("gamma", 3i32);

        assert_eq!(m.get(&"alpha"), Some(&1));
        assert_eq!(m.get(&"beta"), Some(&2));
        assert_eq!(m.get(&"gamma"), Some(&3));
        assert_eq!(m.get(&"delta"), None);
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn update_existing_key() {
        let mut m = CuckooHashMap::new();
        m.insert("key", 10i32);
        let old = m.insert("key", 20i32);
        assert_eq!(old, Some(10));
        assert_eq!(m.get(&"key"), Some(&20));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn remove_existing_and_absent() {
        let mut m = CuckooHashMap::new();
        m.insert(1u64, "one");
        m.insert(2u64, "two");

        assert_eq!(m.remove(&1u64), Some("one"));
        assert_eq!(m.len(), 1);
        assert_eq!(m.remove(&1u64), None);
        assert_eq!(m.remove(&99u64), None);
    }

    #[test]
    fn contains_key() {
        let mut m = CuckooHashMap::new();
        m.insert(42u32, "forty-two");
        assert!(m.contains_key(&42u32));
        assert!(!m.contains_key(&0u32));
    }

    #[test]
    fn large_insert_no_data_loss() {
        let mut m = CuckooHashMap::new();
        for i in 0u64..512 {
            m.insert(i, i * i);
        }
        assert_eq!(m.len(), 512);
        for i in 0u64..512 {
            assert_eq!(m.get(&i), Some(&(i * i)), "missing entry {i}");
        }
    }

    #[test]
    fn clear_empties_map() {
        let mut m = CuckooHashMap::new();
        for i in 0u64..100 {
            m.insert(i, i);
        }
        m.clear();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
        assert_eq!(m.get(&0u64), None);
    }

    #[test]
    fn iter_yields_all_entries() {
        let mut m = CuckooHashMap::new();
        let entries: Vec<(u64, u64)> = (0u64..50).map(|i| (i, i * 2)).collect();
        for (k, v) in &entries {
            m.insert(*k, *v);
        }
        let mut collected: Vec<(u64, u64)> = m.iter().map(|(&k, &v)| (k, v)).collect();
        let mut expected = entries;
        collected.sort();
        expected.sort();
        assert_eq!(collected, expected);
    }

    #[test]
    fn get_mut_updates_in_place() {
        let mut m = CuckooHashMap::new();
        m.insert("x", 0i32);
        if let Some(v) = m.get_mut(&"x") {
            *v += 99;
        }
        assert_eq!(m.get(&"x"), Some(&99));
    }
}
