//! Persistent Hash Array Mapped Trie (HAMT).
//!
//! A HAMT partitions a key's hash 5 bits at a time.  At each trie level a
//! bitmap records which of the 32 possible child slots are occupied; the
//! actual children are stored in a dense array.  The position of a child in
//! the dense array is `popcount(bitmap & ((1 << slot) - 1))`.
//!
//! Every "mutating" operation returns a brand-new `PersistentHashMap` while
//! sharing all unchanged subtrees via `Arc`.  Complexity per operation is
//! O(log₃₂ N) ≈ O(log N).
//!
//! # Node taxonomy
//!
//! | Variant    | Description |
//! |------------|-------------|
//! | `Leaf`     | A single `(hash, key, value)` |
//! | `Bitmap`   | Compressed 32-way trie (< 32 children); uses a popcount index |
//! | `Full`     | All 32 slots occupied — no bitmap needed |
//! | `Collision`| Multiple keys with the exact same hash |

#![allow(dead_code)] // Some node variants are documented even if unused in minimal paths.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BITS: u32 = 5;
const BRANCHING: usize = 1 << BITS; // 32

// ---------------------------------------------------------------------------
// Internal node types
// ---------------------------------------------------------------------------

/// A node in the HAMT trie.
#[derive(Clone, Debug)]
enum HamtNode<K: Clone, V: Clone> {
    /// A single key-value pair.
    Leaf { hash: u64, key: K, val: V },
    /// Compressed 32-way trie node.  `children` is dense; its length equals
    /// `bitmap.count_ones()`.
    Bitmap {
        bitmap: u32,
        children: Arc<Vec<Arc<HamtNode<K, V>>>>,
    },
    /// All 32 slots occupied — no bitmap needed.
    Full { children: Arc<Vec<Arc<HamtNode<K, V>>>> },
    /// Multiple keys with the *same* hash value (true collision bucket).
    Collision { hash: u64, pairs: Arc<Vec<(K, V)>> },
}

// ---------------------------------------------------------------------------
// Hashing helpers
// ---------------------------------------------------------------------------

fn hash_of<K: Hash>(key: &K) -> u64 {
    let mut h = DefaultHasher::new();
    key.hash(&mut h);
    h.finish()
}

#[inline]
fn level_idx(hash: u64, shift: u32) -> usize {
    ((hash >> shift) & (BRANCHING as u64 - 1)) as usize
}

#[inline]
fn dense_idx(bitmap: u32, slot: usize) -> usize {
    let mask = (1u32 << slot).wrapping_sub(1);
    (bitmap & mask).count_ones() as usize
}

// ---------------------------------------------------------------------------
// Node operations
// ---------------------------------------------------------------------------

impl<K: Clone + Hash + Eq, V: Clone> HamtNode<K, V> {
    // ------------------------------------------------------------------
    // get
    // ------------------------------------------------------------------

    fn get<'a>(&'a self, key: &K, hash: u64, shift: u32) -> Option<&'a V> {
        match self {
            HamtNode::Leaf { key: k, val, .. } => {
                if k == key { Some(val) } else { None }
            }
            HamtNode::Bitmap { bitmap, children } => {
                let slot = level_idx(hash, shift);
                let bit = 1u32 << slot;
                if bitmap & bit == 0 {
                    return None;
                }
                let idx = dense_idx(*bitmap, slot);
                children[idx].get(key, hash, shift + BITS)
            }
            HamtNode::Full { children } => {
                let slot = level_idx(hash, shift);
                children[slot].get(key, hash, shift + BITS)
            }
            HamtNode::Collision { pairs, .. } => {
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
    // insert  — returns (new_node, is_new_key)
    // ------------------------------------------------------------------

    fn insert(&self, key: K, val: V, hash: u64, shift: u32) -> (Arc<HamtNode<K, V>>, bool) {
        match self {
            HamtNode::Leaf { hash: lh, key: lk, val: lv } => {
                if lk == &key {
                    // Replace value (same key).
                    return (Arc::new(HamtNode::Leaf { hash, key, val }), false);
                }
                if *lh == hash {
                    // True hash collision.
                    return (
                        Arc::new(HamtNode::Collision {
                            hash,
                            pairs: Arc::new(vec![(lk.clone(), lv.clone()), (key, val)]),
                        }),
                        true,
                    );
                }
                // Different hashes: expand into a Bitmap node.
                let existing =
                    Arc::new(HamtNode::Leaf { hash: *lh, key: lk.clone(), val: lv.clone() });
                let new_leaf = Arc::new(HamtNode::Leaf { hash, key, val });
                let expanded = make_bitmap_from_two(existing, *lh, new_leaf, hash, shift);
                (expanded, true)
            }

            HamtNode::Bitmap { bitmap, children } => {
                let slot = level_idx(hash, shift);
                let bit = 1u32 << slot;

                if bitmap & bit == 0 {
                    // Empty slot: insert a new leaf.
                    let idx = dense_idx(*bitmap, slot);
                    let new_leaf = Arc::new(HamtNode::Leaf { hash, key, val });
                    let mut new_children = (**children).clone();
                    new_children.insert(idx, new_leaf);
                    let new_bitmap = bitmap | bit;
                    if new_children.len() == BRANCHING {
                        // Upgrade to Full.
                        return (
                            Arc::new(HamtNode::Full { children: Arc::new(new_children) }),
                            true,
                        );
                    }
                    return (
                        Arc::new(HamtNode::Bitmap {
                            bitmap: new_bitmap,
                            children: Arc::new(new_children),
                        }),
                        true,
                    );
                }

                // Recurse into existing child.
                let idx = dense_idx(*bitmap, slot);
                let (new_child, is_new) = children[idx].insert(key, val, hash, shift + BITS);
                let mut new_children = (**children).clone();
                new_children[idx] = new_child;
                (
                    Arc::new(HamtNode::Bitmap {
                        bitmap: *bitmap,
                        children: Arc::new(new_children),
                    }),
                    is_new,
                )
            }

            HamtNode::Full { children } => {
                let slot = level_idx(hash, shift);
                let (new_child, is_new) = children[slot].insert(key, val, hash, shift + BITS);
                let mut new_children = (**children).clone();
                new_children[slot] = new_child;
                (Arc::new(HamtNode::Full { children: Arc::new(new_children) }), is_new)
            }

            HamtNode::Collision { hash: ch, pairs } => {
                if *ch != hash {
                    // Expand: wrap existing collision and new leaf into a Bitmap.
                    let coll_node = Arc::new(HamtNode::Collision {
                        hash: *ch,
                        pairs: pairs.clone(),
                    });
                    let new_leaf = Arc::new(HamtNode::Leaf { hash, key, val });
                    let expanded = make_bitmap_from_two(coll_node, *ch, new_leaf, hash, shift);
                    return (expanded, true);
                }
                // Same hash — update or append to the collision bucket.
                let mut new_pairs = (**pairs).clone();
                for (k, v) in new_pairs.iter_mut() {
                    if k == &key {
                        *v = val;
                        return (
                            Arc::new(HamtNode::Collision { hash, pairs: Arc::new(new_pairs) }),
                            false,
                        );
                    }
                }
                new_pairs.push((key, val));
                (Arc::new(HamtNode::Collision { hash, pairs: Arc::new(new_pairs) }), true)
            }
        }
    }

    // ------------------------------------------------------------------
    // delete  — returns (Option<new_node>, was_removed)
    //   None means the slot is completely vacated.
    // ------------------------------------------------------------------

    fn delete(&self, key: &K, hash: u64, shift: u32) -> (Option<Arc<HamtNode<K, V>>>, bool) {
        match self {
            HamtNode::Leaf { key: k, .. } => {
                if k == key {
                    (None, true)
                } else {
                    (Some(Arc::new(self.clone())), false)
                }
            }

            HamtNode::Bitmap { bitmap, children } => {
                let slot = level_idx(hash, shift);
                let bit = 1u32 << slot;
                if bitmap & bit == 0 {
                    // Key not in this subtree.
                    return (Some(Arc::new(self.clone())), false);
                }
                let idx = dense_idx(*bitmap, slot);
                let (new_child_opt, removed) = children[idx].delete(key, hash, shift + BITS);
                if !removed {
                    return (Some(Arc::new(self.clone())), false);
                }
                match new_child_opt {
                    None => {
                        // Remove that slot entirely.
                        let mut new_children = (**children).clone();
                        new_children.remove(idx);
                        if new_children.is_empty() {
                            return (None, true);
                        }
                        let new_bitmap = bitmap & !bit;
                        // Compression: inline a lone Leaf or Collision child.
                        if new_children.len() == 1 {
                            match new_children[0].as_ref() {
                                HamtNode::Leaf { .. } | HamtNode::Collision { .. } => {
                                    return (Some(new_children.remove(0)), true);
                                }
                                _ => {}
                            }
                        }
                        (
                            Some(Arc::new(HamtNode::Bitmap {
                                bitmap: new_bitmap,
                                children: Arc::new(new_children),
                            })),
                            true,
                        )
                    }
                    Some(new_child) => {
                        let mut new_children = (**children).clone();
                        new_children[idx] = new_child;
                        (
                            Some(Arc::new(HamtNode::Bitmap {
                                bitmap: *bitmap,
                                children: Arc::new(new_children),
                            })),
                            true,
                        )
                    }
                }
            }

            HamtNode::Full { children } => {
                let slot = level_idx(hash, shift);
                let (new_child_opt, removed) = children[slot].delete(key, hash, shift + BITS);
                if !removed {
                    return (Some(Arc::new(self.clone())), false);
                }
                match new_child_opt {
                    None => {
                        // Downgrade Full → Bitmap with one slot removed.
                        let bit = 1u32 << slot;
                        let new_bitmap = !bit; // 31 bits set
                        let mut new_children: Vec<Arc<HamtNode<K, V>>> =
                            Vec::with_capacity(BRANCHING - 1);
                        for (i, child) in children.iter().enumerate() {
                            if i != slot {
                                new_children.push(child.clone());
                            }
                        }
                        (
                            Some(Arc::new(HamtNode::Bitmap {
                                bitmap: new_bitmap,
                                children: Arc::new(new_children),
                            })),
                            true,
                        )
                    }
                    Some(new_child) => {
                        // Replace the child, stay Full.
                        let mut new_children = (**children).clone();
                        new_children[slot] = new_child;
                        (Some(Arc::new(HamtNode::Full { children: Arc::new(new_children) })), true)
                    }
                }
            }

            HamtNode::Collision { hash: ch, pairs } => {
                if *ch != hash {
                    return (Some(Arc::new(self.clone())), false);
                }
                let mut new_pairs = (**pairs).clone();
                let before = new_pairs.len();
                new_pairs.retain(|(k, _)| k != key);
                if new_pairs.len() == before {
                    return (Some(Arc::new(self.clone())), false);
                }
                if new_pairs.is_empty() {
                    return (None, true);
                }
                if new_pairs.len() == 1 {
                    let (k, v) = new_pairs.remove(0);
                    return (
                        Some(Arc::new(HamtNode::Leaf { hash: *ch, key: k, val: v })),
                        true,
                    );
                }
                (
                    Some(Arc::new(HamtNode::Collision {
                        hash: *ch,
                        pairs: Arc::new(new_pairs),
                    })),
                    true,
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Build a Bitmap node from two children at a given trie level.
// ---------------------------------------------------------------------------

fn make_bitmap_from_two<K: Clone + Hash + Eq, V: Clone>(
    child_a: Arc<HamtNode<K, V>>,
    hash_a: u64,
    child_b: Arc<HamtNode<K, V>>,
    hash_b: u64,
    shift: u32,
) -> Arc<HamtNode<K, V>> {
    let slot_a = level_idx(hash_a, shift);
    let slot_b = level_idx(hash_b, shift);

    if slot_a == slot_b {
        // Still colliding at this level — recurse one level deeper.
        let inner = make_bitmap_from_two(child_a, hash_a, child_b, hash_b, shift + BITS);
        let bit = 1u32 << slot_a;
        return Arc::new(HamtNode::Bitmap {
            bitmap: bit,
            children: Arc::new(vec![inner]),
        });
    }

    let (bit_a, bit_b) = (1u32 << slot_a, 1u32 << slot_b);
    let bitmap = bit_a | bit_b;
    let children = if slot_a < slot_b {
        vec![child_a, child_b]
    } else {
        vec![child_b, child_a]
    };
    Arc::new(HamtNode::Bitmap { bitmap, children: Arc::new(children) })
}

// ---------------------------------------------------------------------------
// Iterator
// ---------------------------------------------------------------------------

enum StackItem<'a, K: Clone, V: Clone> {
    Node(&'a HamtNode<K, V>),
    Pair(&'a K, &'a V),
}

/// Iterates over all `(&K, &V)` pairs in the map (order is unspecified).
pub struct Iter<'a, K: Clone, V: Clone> {
    stack: Vec<StackItem<'a, K, V>>,
}

impl<'a, K: Clone + Hash + Eq, V: Clone> Iter<'a, K, V> {
    fn new(root: Option<&'a Arc<HamtNode<K, V>>>) -> Self {
        let mut stack = Vec::new();
        if let Some(r) = root {
            stack.push(StackItem::Node(r.as_ref()));
        }
        Iter { stack }
    }
}

impl<'a, K: Clone + Hash + Eq, V: Clone> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let item = self.stack.pop()?;
            match item {
                StackItem::Pair(k, v) => return Some((k, v)),
                StackItem::Node(node) => match node {
                    HamtNode::Leaf { key, val, .. } => return Some((key, val)),
                    HamtNode::Bitmap { children, .. } => {
                        for child in children.iter().rev() {
                            self.stack.push(StackItem::Node(child.as_ref()));
                        }
                    }
                    HamtNode::Full { children } => {
                        for child in children.iter().rev() {
                            self.stack.push(StackItem::Node(child.as_ref()));
                        }
                    }
                    HamtNode::Collision { pairs, .. } => {
                        for (k, v) in pairs.iter().rev() {
                            self.stack.push(StackItem::Pair(k, v));
                        }
                    }
                },
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Persistent Hash Map based on a Hash Array Mapped Trie (HAMT).
///
/// Every "mutating" operation returns a brand-new `PersistentHashMap` while
/// sharing all unchanged subtrees with the original via `Arc`.  Complexity
/// per operation is O(log₃₂ N) ≈ O(log N).
///
/// # Examples
///
/// ```
/// use scirs2_core::persistent::persistent_hamt::PersistentHashMap;
///
/// let m0 = PersistentHashMap::new();
/// let m1 = m0.insert("hello", 1u32);
/// let m2 = m1.insert("world", 2u32);
///
/// assert_eq!(m2.get(&"hello"), Some(&1));
/// assert_eq!(m2.get(&"world"), Some(&2));
///
/// // m1 is unaffected.
/// assert_eq!(m1.get(&"world"), None);
/// ```
#[derive(Clone, Debug)]
pub struct PersistentHashMap<K: Clone + Hash + Eq, V: Clone> {
    root: Option<Arc<HamtNode<K, V>>>,
    len: usize,
}

impl<K: Clone + Hash + Eq, V: Clone> Default for PersistentHashMap<K, V> {
    fn default() -> Self {
        PersistentHashMap::new()
    }
}

impl<K: Clone + Hash + Eq, V: Clone> PersistentHashMap<K, V> {
    /// Create an empty map.
    pub fn new() -> Self {
        PersistentHashMap { root: None, len: 0 }
    }

    /// Number of key-value pairs.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the map contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Look up a key and return a reference to its value.
    pub fn get(&self, key: &K) -> Option<&V> {
        let hash = hash_of(key);
        self.root.as_ref()?.get(key, hash, 0)
    }

    /// Returns `true` if the map contains `key`.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Return a new map with `(key, value)` inserted / updated.
    pub fn insert(&self, key: K, value: V) -> Self {
        let hash = hash_of(&key);
        let (new_root, is_new) = match &self.root {
            None => {
                let leaf = Arc::new(HamtNode::Leaf { hash, key, val: value });
                (leaf, true)
            }
            Some(r) => r.insert(key, value, hash, 0),
        };
        PersistentHashMap {
            root: Some(new_root),
            len: if is_new { self.len + 1 } else { self.len },
        }
    }

    /// Return a new map with `key` removed.
    pub fn delete(&self, key: &K) -> Self {
        let hash = hash_of(key);
        match &self.root {
            None => self.clone(),
            Some(r) => {
                let (new_root_opt, removed) = r.delete(key, hash, 0);
                if !removed {
                    return self.clone();
                }
                PersistentHashMap {
                    root: new_root_opt,
                    len: self.len - 1,
                }
            }
        }
    }

    /// Iterator over `(&K, &V)` pairs (order is unspecified).
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter::new(self.root.as_ref())
    }
}

impl<K: Clone + Hash + Eq, V: Clone> FromIterator<(K, V)> for PersistentHashMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut map = PersistentHashMap::new();
        for (k, v) in iter {
            map = map.insert(k, v);
        }
        map
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_map() {
        let m: PersistentHashMap<&str, i32> = PersistentHashMap::new();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
        assert_eq!(m.get(&"key"), None);
        assert!(!m.contains_key(&"key"));
    }

    #[test]
    fn test_basic_insert_and_get() {
        let m = PersistentHashMap::new()
            .insert("a", 1)
            .insert("b", 2)
            .insert("c", 3);

        assert_eq!(m.get(&"a"), Some(&1));
        assert_eq!(m.get(&"b"), Some(&2));
        assert_eq!(m.get(&"c"), Some(&3));
        assert_eq!(m.get(&"d"), None);
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn test_persistence() {
        let m0: PersistentHashMap<i32, i32> = PersistentHashMap::new();
        let m1 = m0.insert(1, 10);
        let m2 = m1.insert(2, 20);
        let m3 = m2.insert(3, 30);

        assert_eq!(m0.len(), 0);
        assert_eq!(m1.len(), 1);
        assert_eq!(m2.len(), 2);
        assert_eq!(m3.len(), 3);

        assert_eq!(m1.get(&1), Some(&10));
        assert_eq!(m1.get(&2), None);
        assert_eq!(m2.get(&2), Some(&20));
        assert_eq!(m2.get(&3), None);
        assert_eq!(m3.get(&3), Some(&30));
    }

    #[test]
    fn test_update_existing_key() {
        let m1 = PersistentHashMap::new().insert(42, "first");
        let m2 = m1.insert(42, "second");

        assert_eq!(m1.get(&42), Some(&"first"));
        assert_eq!(m2.get(&42), Some(&"second"));
        assert_eq!(m2.len(), 1);
    }

    #[test]
    fn test_delete_basic() {
        let m = PersistentHashMap::new()
            .insert(1, "a")
            .insert(2, "b")
            .insert(3, "c");

        let m2 = m.delete(&2);
        assert_eq!(m2.len(), 2);
        assert_eq!(m2.get(&1), Some(&"a"));
        assert_eq!(m2.get(&2), None);
        assert_eq!(m2.get(&3), Some(&"c"));

        // Original unchanged.
        assert_eq!(m.get(&2), Some(&"b"));
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn test_delete_nonexistent() {
        let m = PersistentHashMap::new().insert(1, 1).insert(2, 2);
        let m2 = m.delete(&99);
        assert_eq!(m2.len(), 2);
    }

    #[test]
    fn test_delete_all() {
        let m = PersistentHashMap::new()
            .insert(1, 1)
            .insert(2, 2)
            .insert(3, 3);
        let m = m.delete(&1).delete(&2).delete(&3);
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn test_iter_collects_all() {
        let n = 50usize;
        let m: PersistentHashMap<usize, usize> = (0..n).map(|i| (i, i * 2)).collect();
        assert_eq!(m.len(), n);

        let mut collected: Vec<(usize, usize)> =
            m.iter().map(|(&k, &v)| (k, v)).collect();
        collected.sort_by_key(|&(k, _)| k);
        let expected: Vec<(usize, usize)> = (0..n).map(|i| (i, i * 2)).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_large_map() {
        let n = 10_000usize;
        let m: PersistentHashMap<usize, usize> = (0..n).map(|i| (i, i)).collect();
        assert_eq!(m.len(), n);
        for i in 0..n {
            assert_eq!(m.get(&i), Some(&i));
        }
    }

    #[test]
    fn test_large_delete() {
        let n = 500usize;
        let m: PersistentHashMap<usize, usize> = (0..n).map(|i| (i, i)).collect();
        let mut m = m;
        for i in 0..n {
            m = m.delete(&i);
        }
        assert!(m.is_empty());
    }

    #[test]
    fn test_collision_handling() {
        // Insert many keys to exercise Bitmap→Full upgrades and collision paths.
        let mut m = PersistentHashMap::new();
        for i in 0..200i32 {
            m = m.insert(i, i * 3);
        }
        assert_eq!(m.len(), 200);
        for i in 0..200i32 {
            assert_eq!(m.get(&i), Some(&(i * 3)));
        }
    }

    #[test]
    fn test_from_iterator() {
        let pairs = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let m: PersistentHashMap<i32, char> = pairs.into_iter().collect();
        assert_eq!(m.len(), 3);
        assert_eq!(m.get(&1), Some(&'a'));
        assert_eq!(m.get(&2), Some(&'b'));
        assert_eq!(m.get(&3), Some(&'c'));
    }
}
