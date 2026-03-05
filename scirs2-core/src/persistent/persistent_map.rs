//! Persistent Hash Map based on a Hash Array Mapped Trie (HAMT).
//!
//! A HAMT partitions a key's hash 5 bits at a time.  At each trie level a
//! *bitmap* records which of the 32 possible slots are occupied; a *dense*
//! array holds only the occupied children.  The position of a child in the
//! dense array is `popcount(bitmap & ((1 << slot) - 1))`.
//!
//! Every "mutating" operation returns a brand-new `PersistentMap` while
//! sharing all unchanged subtrees with the original via `Arc`.  Complexity
//! per operation is O(log₃₂ N) = O(log N).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

const BITS: u32 = 5;
const BRANCHING: usize = 1 << BITS; // 32

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum HamtNode<K: Clone, V: Clone> {
    /// A key-value pair that belongs in this slot.
    Leaf(K, V),
    /// Compressed 32-way trie node.
    Internal {
        bitmap: u32,
        children: Arc<[Arc<HamtNode<K, V>>]>,
    },
    /// Collision bucket — multiple keys with the same hash.
    Collision(Arc<[(K, V)]>),
}

impl<K: Clone + Hash + Eq, V: Clone> HamtNode<K, V> {
    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn hash_of(key: &K) -> u64 {
        let mut h = DefaultHasher::new();
        key.hash(&mut h);
        h.finish()
    }

    /// Index (0..32) for the current trie level.
    fn level_index(hash: u64, shift: u32) -> usize {
        ((hash >> shift) & (BRANCHING as u64 - 1)) as usize
    }

    /// Dense array index via popcount.
    fn dense_index(bitmap: u32, slot: usize) -> usize {
        let mask = (1u32 << slot).wrapping_sub(1);
        (bitmap & mask).count_ones() as usize
    }

    // ------------------------------------------------------------------
    // Lookup
    // ------------------------------------------------------------------

    fn get<'a>(&'a self, key: &K, hash: u64, shift: u32) -> Option<&'a V> {
        match self {
            HamtNode::Leaf(k, v) => {
                if k == key {
                    Some(v)
                } else {
                    None
                }
            }
            HamtNode::Internal { bitmap, children } => {
                let slot = Self::level_index(hash, shift);
                let bit = 1u32 << slot;
                if bitmap & bit == 0 {
                    return None;
                }
                let idx = Self::dense_index(*bitmap, slot);
                children[idx].get(key, hash, shift + BITS)
            }
            HamtNode::Collision(pairs) => {
                for (k, v) in pairs.iter() {
                    if k == key {
                        return Some(v);
                    }
                }
                None
            }
        }
    }

    // ------------------------------------------------------------------
    // Insert
    // ------------------------------------------------------------------

    fn insert(&self, key: K, val: V, hash: u64, shift: u32) -> (Arc<HamtNode<K, V>>, bool) {
        match self {
            HamtNode::Leaf(k, _v) => {
                if k == &key {
                    // Replace existing.
                    return (Arc::new(HamtNode::Leaf(key, val)), false);
                }
                // Two different keys at the same node — expand into Internal or Collision.
                let other_hash = Self::hash_of(k);
                if other_hash == hash {
                    // Hash collision: create collision bucket.
                    let pairs: Vec<(K, V)> = vec![(k.clone(), _v.clone()), (key, val)];
                    return (
                        Arc::new(HamtNode::Collision(Arc::from(pairs.as_slice()))),
                        true,
                    );
                }
                // Distinct hashes: create a new Internal containing both leaves.
                let new_node = HamtNode::Internal {
                    bitmap: 0,
                    children: Arc::from([] as [Arc<HamtNode<K, V>>; 0]),
                };
                // Insert the existing leaf, then the new one.
                let (n1, _) = new_node.insert(k.clone(), _v.clone(), other_hash, shift);
                let (n2, added) = n1.insert(key, val, hash, shift);
                (n2, added)
            }

            HamtNode::Internal { bitmap, children } => {
                let slot = Self::level_index(hash, shift);
                let bit = 1u32 << slot;
                let dense = Self::dense_index(*bitmap, slot);

                let mut new_children: Vec<Arc<HamtNode<K, V>>> = children.to_vec();

                if bitmap & bit == 0 {
                    // Empty slot — just insert a new leaf.
                    new_children.insert(dense, Arc::new(HamtNode::Leaf(key, val)));
                    let new_bitmap = bitmap | bit;
                    (
                        Arc::new(HamtNode::Internal {
                            bitmap: new_bitmap,
                            children: Arc::from(new_children.as_slice()),
                        }),
                        true,
                    )
                } else {
                    // Occupied — recurse.
                    let (new_child, added) =
                        new_children[dense].insert(key, val, hash, shift + BITS);
                    new_children[dense] = new_child;
                    (
                        Arc::new(HamtNode::Internal {
                            bitmap: *bitmap,
                            children: Arc::from(new_children.as_slice()),
                        }),
                        added,
                    )
                }
            }

            HamtNode::Collision(pairs) => {
                // Check if key already exists.
                for (k, _) in pairs.iter() {
                    if k == &key {
                        let new_pairs: Vec<(K, V)> = pairs
                            .iter()
                            .map(|(ek, ev)| {
                                if ek == &key {
                                    (key.clone(), val.clone())
                                } else {
                                    (ek.clone(), ev.clone())
                                }
                            })
                            .collect();
                        return (
                            Arc::new(HamtNode::Collision(Arc::from(new_pairs.as_slice()))),
                            false,
                        );
                    }
                }
                let mut new_pairs: Vec<(K, V)> = pairs.to_vec();
                new_pairs.push((key, val));
                (
                    Arc::new(HamtNode::Collision(Arc::from(new_pairs.as_slice()))),
                    true,
                )
            }
        }
    }

    // ------------------------------------------------------------------
    // Remove
    // ------------------------------------------------------------------

    /// Returns `(new_node, size_delta)`.  `new_node = None` means the node
    /// should be removed from its parent.
    fn remove(&self, key: &K, hash: u64, shift: u32) -> (Option<Arc<HamtNode<K, V>>>, bool) {
        match self {
            HamtNode::Leaf(k, _) => {
                if k == key {
                    (None, true)
                } else {
                    (Some(Arc::new(self.clone())), false)
                }
            }

            HamtNode::Internal { bitmap, children } => {
                let slot = Self::level_index(hash, shift);
                let bit = 1u32 << slot;
                if bitmap & bit == 0 {
                    // Key not here.
                    return (Some(Arc::new(self.clone())), false);
                }
                let dense = Self::dense_index(*bitmap, slot);
                let (child_result, removed) =
                    children[dense].remove(key, hash, shift + BITS);

                if !removed {
                    return (Some(Arc::new(self.clone())), false);
                }

                let mut new_children: Vec<Arc<HamtNode<K, V>>> = children.to_vec();
                let new_bitmap;
                match child_result {
                    None => {
                        new_children.remove(dense);
                        new_bitmap = bitmap & !bit;
                    }
                    Some(child) => {
                        new_children[dense] = child;
                        new_bitmap = *bitmap;
                    }
                }

                if new_children.is_empty() {
                    (None, true)
                } else if new_children.len() == 1 {
                    // Collapse if the single remaining child is a leaf.
                    match *new_children[0] {
                        HamtNode::Leaf(_, _) => (Some(Arc::clone(&new_children[0])), true),
                        _ => (
                            Some(Arc::new(HamtNode::Internal {
                                bitmap: new_bitmap,
                                children: Arc::from(new_children.as_slice()),
                            })),
                            true,
                        ),
                    }
                } else {
                    (
                        Some(Arc::new(HamtNode::Internal {
                            bitmap: new_bitmap,
                            children: Arc::from(new_children.as_slice()),
                        })),
                        true,
                    )
                }
            }

            HamtNode::Collision(pairs) => {
                let new_pairs: Vec<(K, V)> =
                    pairs.iter().filter(|(k, _)| k != key).cloned().collect();
                if new_pairs.len() == pairs.len() {
                    return (Some(Arc::new(self.clone())), false);
                }
                if new_pairs.len() == 1 {
                    let (k, v) = new_pairs.into_iter().next().expect("len==1");
                    return (Some(Arc::new(HamtNode::Leaf(k, v))), true);
                }
                (
                    Some(Arc::new(HamtNode::Collision(Arc::from(new_pairs.as_slice())))),
                    true,
                )
            }
        }
    }

    // ------------------------------------------------------------------
    // Iteration
    // ------------------------------------------------------------------

    fn collect_entries<'a>(&'a self, out: &mut Vec<(&'a K, &'a V)>) {
        match self {
            HamtNode::Leaf(k, v) => out.push((k, v)),
            HamtNode::Internal { children, .. } => {
                for child in children.iter() {
                    child.collect_entries(out);
                }
            }
            HamtNode::Collision(pairs) => {
                for (k, v) in pairs.iter() {
                    out.push((k, v));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Persistent hash map backed by a Hash Array Mapped Trie (HAMT).
///
/// Every operation returns a new `PersistentMap` while sharing unchanged
/// subtrees with the original.
///
/// ```
/// use scirs2_core::persistent::PersistentMap;
///
/// let m0 = PersistentMap::new();
/// let m1 = m0.insert("a", 1u32);
/// let m2 = m1.insert("b", 2u32);
/// let m3 = m2.remove(&"a");
///
/// assert_eq!(m2.get(&"a"), Some(&1));
/// assert_eq!(m3.get(&"a"), None);
/// assert_eq!(m3.get(&"b"), Some(&2));
/// assert_eq!(m2.len(), 2);
/// ```
#[derive(Clone)]
pub struct PersistentMap<K: Clone + Hash + Eq, V: Clone> {
    len: usize,
    root: Option<Arc<HamtNode<K, V>>>,
}

impl<K: Clone + Hash + Eq, V: Clone> PersistentMap<K, V> {
    /// Creates an empty `PersistentMap`.
    pub fn new() -> Self {
        PersistentMap { len: 0, root: None }
    }

    /// Returns the number of key-value pairs.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the map contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a reference to the value associated with `key`, or `None`.
    pub fn get(&self, key: &K) -> Option<&V> {
        let hash = HamtNode::<K, V>::hash_of(key);
        self.root.as_ref()?.get(key, hash, 0)
    }

    /// Returns `true` if the map contains `key`.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Returns a new map with `key → val` inserted (or updated).
    pub fn insert(&self, key: K, val: V) -> Self {
        let hash = HamtNode::<K, V>::hash_of(&key);
        match &self.root {
            None => {
                let node = Arc::new(HamtNode::Leaf(key, val));
                PersistentMap {
                    len: 1,
                    root: Some(node),
                }
            }
            Some(root) => {
                let (new_root, added) = root.insert(key, val, hash, 0);
                PersistentMap {
                    len: if added { self.len + 1 } else { self.len },
                    root: Some(new_root),
                }
            }
        }
    }

    /// Returns a new map with `key` removed.
    pub fn remove(&self, key: &K) -> Self {
        let hash = HamtNode::<K, V>::hash_of(key);
        match &self.root {
            None => self.clone(),
            Some(root) => {
                let (new_root, removed) = root.remove(key, hash, 0);
                PersistentMap {
                    len: if removed { self.len - 1 } else { self.len },
                    root: new_root,
                }
            }
        }
    }

    /// Returns an iterator over all `(&K, &V)` pairs.
    ///
    /// The iteration order is unspecified (hash-dependent).
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        let mut entries: Vec<(&K, &V)> = Vec::with_capacity(self.len);
        if let Some(root) = &self.root {
            root.collect_entries(&mut entries);
        }
        entries.into_iter()
    }
}

impl<K: Clone + Hash + Eq, V: Clone> Default for PersistentMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let m: PersistentMap<&str, i32> = PersistentMap::new();
        assert!(m.is_empty());
        assert_eq!(m.get(&"x"), None);
    }

    #[test]
    fn test_insert_and_get() {
        let m = PersistentMap::new()
            .insert("a", 1)
            .insert("b", 2)
            .insert("c", 3);
        assert_eq!(m.get(&"a"), Some(&1));
        assert_eq!(m.get(&"b"), Some(&2));
        assert_eq!(m.get(&"c"), Some(&3));
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn test_update() {
        let m0 = PersistentMap::new().insert("x", 1);
        let m1 = m0.insert("x", 99);
        assert_eq!(m0.get(&"x"), Some(&1));
        assert_eq!(m1.get(&"x"), Some(&99));
        assert_eq!(m1.len(), 1);
    }

    #[test]
    fn test_remove() {
        let m = PersistentMap::new().insert("a", 1).insert("b", 2);
        let m2 = m.remove(&"a");
        assert_eq!(m2.get(&"a"), None);
        assert_eq!(m2.get(&"b"), Some(&2));
        assert_eq!(m.get(&"a"), Some(&1)); // original unchanged
        assert_eq!(m2.len(), 1);
    }

    #[test]
    fn test_large_map() {
        let mut m = PersistentMap::new();
        for i in 0..500_i32 {
            m = m.insert(i, i * i);
        }
        assert_eq!(m.len(), 500);
        for i in 0..500_i32 {
            assert_eq!(m.get(&i), Some(&(i * i)));
        }
    }

    #[test]
    fn test_iter() {
        let m = PersistentMap::new()
            .insert("a", 1)
            .insert("b", 2)
            .insert("c", 3);
        let mut pairs: Vec<(&&str, &i32)> = m.iter().collect();
        pairs.sort_by_key(|(k, _)| **k);
        assert_eq!(pairs, vec![(&"a", &1), (&"b", &2), (&"c", &3)]);
    }

    #[test]
    fn test_remove_nonexistent() {
        let m = PersistentMap::new().insert("a", 1);
        let m2 = m.remove(&"z");
        assert_eq!(m2.len(), 1);
    }
}
