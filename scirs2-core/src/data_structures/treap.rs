//! Treap — a randomised binary search tree with expected O(log n) operations.
//!
//! A treap assigns each node a uniformly random *priority* at insertion time.
//! It then maintains both the **BST** property on keys and the **heap**
//! property on priorities simultaneously. This gives an expected tree height
//! of O(log n) with high probability, avoiding the worst-case linear depth
//! of a plain BST.
//!
//! # Operations
//!
//! | Operation       | Expected time | Notes                         |
//! |-----------------|---------------|-------------------------------|
//! | `insert`        | O(log n)      |                               |
//! | `remove`        | O(log n)      |                               |
//! | `get`           | O(log n)      |                               |
//! | `range_query`   | O(log n + k)  | k results returned            |
//! | `split`         | O(log n)      | Consumes the treap            |
//! | `merge`         | O(log n)      | Left keys < all right keys    |
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::data_structures::Treap;
//!
//! let mut t = Treap::new();
//! t.insert(5, "five");
//! t.insert(3, "three");
//! t.insert(7, "seven");
//!
//! assert_eq!(t.get(&5), Some(&"five"));
//! assert_eq!(t.len(), 3);
//!
//! let pairs = t.range_query(&3, &6);
//! assert_eq!(pairs.len(), 2);
//!
//! let (left, right) = t.split(&5);
//! assert_eq!(left.len(), 1);  // only key 3
//! assert_eq!(right.len(), 2); // keys 5 and 7
//! ```

use std::fmt;

// ============================================================================
// Priority / RNG
// ============================================================================

/// A minimal splitmix64 PRNG used to generate node priorities.
///
/// Using a deterministic sequence seeded by a simple mix means the treap is
/// reproducible and does not require the `rand` crate.
struct Splitmix64 {
    state: u64,
}

impl Splitmix64 {
    fn new(seed: u64) -> Self {
        Splitmix64 { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

// Thread-local RNG so that inserts from any thread are independent.
use std::cell::Cell;
thread_local! {
    static RNG: Cell<Splitmix64> = Cell::new(Splitmix64::new(0xdead_beef_cafe_babe));
}

fn next_priority() -> u64 {
    RNG.with(|cell| {
        // SAFETY: Cell<T> is only accessed from the current thread.
        let mut rng = unsafe { std::ptr::read(cell.as_ptr()) };
        let p = rng.next_u64();
        unsafe { std::ptr::write(cell.as_ptr(), rng) };
        p
    })
}

// ============================================================================
// TreapNode
// ============================================================================

struct Node<K, V> {
    key: K,
    value: V,
    priority: u64,
    size: usize,
    left: NodePtr<K, V>,
    right: NodePtr<K, V>,
}

type NodePtr<K, V> = Option<Box<Node<K, V>>>;

impl<K: Ord, V> Node<K, V> {
    fn new(key: K, value: V) -> Box<Self> {
        Box::new(Node {
            key,
            value,
            priority: next_priority(),
            size: 1,
            left: None,
            right: None,
        })
    }

    fn update_size(&mut self) {
        self.size = 1
            + self.left.as_deref().map_or(0, |n| n.size)
            + self.right.as_deref().map_or(0, |n| n.size);
    }
}

// ============================================================================
// Core split / merge primitives
// ============================================================================

/// Splits tree `root` into two trees:
/// - `left`:  contains all keys **< split_key**
/// - `right`: contains all keys **>= split_key**
///
/// Consumes `root`.
fn split<K: Ord + Clone, V>(
    root: NodePtr<K, V>,
    split_key: &K,
) -> (NodePtr<K, V>, NodePtr<K, V>) {
    match root {
        None => (None, None),
        Some(mut node) => {
            if node.key < *split_key {
                let (rl, rr) = split(node.right.take(), split_key);
                node.right = rl;
                node.update_size();
                (Some(node), rr)
            } else {
                let (ll, lr) = split(node.left.take(), split_key);
                node.left = lr;
                node.update_size();
                (ll, Some(node))
            }
        }
    }
}

/// Merges two treaps where all keys in `left` are **strictly less than** all
/// keys in `right`.
///
/// The heap invariant on priorities is maintained.
fn merge<K: Ord, V>(left: NodePtr<K, V>, right: NodePtr<K, V>) -> NodePtr<K, V> {
    match (left, right) {
        (None, r) => r,
        (l, None) => l,
        (Some(mut l), Some(mut r)) => {
            if l.priority >= r.priority {
                l.right = merge(l.right.take(), Some(r));
                l.update_size();
                Some(l)
            } else {
                r.left = merge(Some(l), r.left.take());
                r.update_size();
                Some(r)
            }
        }
    }
}

// ============================================================================
// Treap
// ============================================================================

/// A randomised binary search tree.
///
/// Keys are kept in sorted order; arbitrary range queries and O(log n) splits
/// and merges are supported.
pub struct Treap<K: Ord, V> {
    root: NodePtr<K, V>,
}

impl<K: Ord + Clone, V> Treap<K, V> {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Creates an empty treap.
    pub fn new() -> Self {
        Treap { root: None }
    }

    // ------------------------------------------------------------------
    // Core operations
    // ------------------------------------------------------------------

    /// Inserts `key → value`.  If the key is already present the value is
    /// updated and the old value is returned.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let (old, new_root) = insert_node(self.root.take(), key, value);
        self.root = new_root;
        old
    }

    /// Removes and returns the value for `key`, or `None` if absent.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let (removed, new_root) = remove_node(self.root.take(), key);
        self.root = new_root;
        removed
    }

    /// Returns a shared reference to the value for `key`, or `None`.
    pub fn get(&self, key: &K) -> Option<&V> {
        let mut cur = self.root.as_deref();
        while let Some(node) = cur {
            if key < &node.key {
                cur = node.left.as_deref();
            } else if key > &node.key {
                cur = node.right.as_deref();
            } else {
                return Some(&node.value);
            }
        }
        None
    }

    /// Returns a mutable reference to the value for `key`, or `None`.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        get_mut_node(self.root.as_deref_mut(), key)
    }

    /// Returns `true` if `key` is in the treap.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    // ------------------------------------------------------------------
    // Range query
    // ------------------------------------------------------------------

    /// Returns all `(&K, &V)` pairs whose key lies in the **inclusive** range
    /// `[lo, hi]`, in ascending key order.
    pub fn range_query<'a>(&'a self, lo: &K, hi: &K) -> Vec<(&'a K, &'a V)> {
        let mut result = Vec::new();
        range_collect(self.root.as_deref(), lo, hi, &mut result);
        result
    }

    // ------------------------------------------------------------------
    // Split / merge
    // ------------------------------------------------------------------

    /// Splits `self` into two treaps at `key`:
    /// - **left**: all keys `< key`
    /// - **right**: all keys `>= key`
    ///
    /// Consumes `self`.
    pub fn split(mut self, key: &K) -> (Treap<K, V>, Treap<K, V>) {
        let (l, r) = split(self.root.take(), key);
        (Treap { root: l }, Treap { root: r })
    }

    /// Merges two treaps where all keys in `left` are strictly less than all
    /// keys in `right`.
    ///
    /// **Caller is responsible** for ensuring the key order invariant.
    pub fn merge(left: Treap<K, V>, right: Treap<K, V>) -> Treap<K, V> {
        Treap {
            root: merge(left.root, right.root),
        }
    }

    // ------------------------------------------------------------------
    // Inspection
    // ------------------------------------------------------------------

    /// Returns the number of key-value pairs.
    pub fn len(&self) -> usize {
        self.root.as_deref().map_or(0, |n| n.size)
    }

    /// Returns `true` if the treap contains no entries.
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Returns a reference to the minimum key and its value, or `None`.
    pub fn min(&self) -> Option<(&K, &V)> {
        let mut cur = self.root.as_deref()?;
        while let Some(left) = cur.left.as_deref() {
            cur = left;
        }
        Some((&cur.key, &cur.value))
    }

    /// Returns a reference to the maximum key and its value, or `None`.
    pub fn max(&self) -> Option<(&K, &V)> {
        let mut cur = self.root.as_deref()?;
        while let Some(right) = cur.right.as_deref() {
            cur = right;
        }
        Some((&cur.key, &cur.value))
    }

    /// Returns an in-order traversal of all `(&K, &V)` pairs.
    pub fn inorder(&self) -> Vec<(&K, &V)> {
        let mut result = Vec::with_capacity(self.len());
        inorder_collect(self.root.as_deref(), &mut result);
        result
    }
}

// ============================================================================
// Internal recursive helpers
// ============================================================================

/// Insert `key → value` into the subtree at `root`.
/// Returns `(old_value_if_updated, new_root)`.
fn insert_node<K: Ord + Clone, V>(
    root: NodePtr<K, V>,
    key: K,
    value: V,
) -> (Option<V>, NodePtr<K, V>) {
    match root {
        None => (None, Some(Node::new(key, value))),
        Some(mut node) => {
            if key < node.key {
                let (old, new_left) = insert_node(node.left.take(), key, value);
                node.left = new_left;
                node.update_size();
                // Rotate right if left child has higher priority.
                let result = if node.left.as_deref().map_or(false, |l| l.priority > node.priority) {
                    rotate_right(node)
                } else {
                    node
                };
                (old, Some(result))
            } else if key > node.key {
                let (old, new_right) = insert_node(node.right.take(), key, value);
                node.right = new_right;
                node.update_size();
                // Rotate left if right child has higher priority.
                let result = if node.right.as_deref().map_or(false, |r| r.priority > node.priority) {
                    rotate_left(node)
                } else {
                    node
                };
                (old, Some(result))
            } else {
                // Key exists: update in-place.
                let old = std::mem::replace(&mut node.value, value);
                (Some(old), Some(node))
            }
        }
    }
}

/// Remove `key` from the subtree at `root`.
/// Returns `(removed_value, new_root)`.
fn remove_node<K: Ord, V>(root: NodePtr<K, V>, key: &K) -> (Option<V>, NodePtr<K, V>) {
    match root {
        None => (None, None),
        Some(mut node) => {
            if key < &node.key {
                let (removed, new_left) = remove_node(node.left.take(), key);
                node.left = new_left;
                node.update_size();
                (removed, Some(node))
            } else if key > &node.key {
                let (removed, new_right) = remove_node(node.right.take(), key);
                node.right = new_right;
                node.update_size();
                (removed, Some(node))
            } else {
                // Found: merge the two children to replace this node.
                let merged = merge(node.left.take(), node.right.take());
                (Some(node.value), merged)
            }
        }
    }
}

fn get_mut_node<'a, K: Ord, V>(root: Option<&'a mut Node<K, V>>, key: &K) -> Option<&'a mut V> {
    match root {
        None => None,
        Some(node) => {
            if key < &node.key {
                get_mut_node(node.left.as_deref_mut(), key)
            } else if key > &node.key {
                get_mut_node(node.right.as_deref_mut(), key)
            } else {
                Some(&mut node.value)
            }
        }
    }
}

fn inorder_collect<'a, K: Ord, V>(
    root: Option<&'a Node<K, V>>,
    result: &mut Vec<(&'a K, &'a V)>,
) {
    if let Some(node) = root {
        inorder_collect(node.left.as_deref(), result);
        result.push((&node.key, &node.value));
        inorder_collect(node.right.as_deref(), result);
    }
}

fn range_collect<'a, K: Ord, V>(
    root: Option<&'a Node<K, V>>,
    lo: &K,
    hi: &K,
    result: &mut Vec<(&'a K, &'a V)>,
) {
    let node = match root {
        None => return,
        Some(n) => n,
    };
    // Prune left subtree if all keys there are < lo.
    if &node.key >= lo {
        range_collect(node.left.as_deref(), lo, hi, result);
    }
    if &node.key >= lo && &node.key <= hi {
        result.push((&node.key, &node.value));
    }
    // Prune right subtree if all keys there are > hi.
    if &node.key <= hi {
        range_collect(node.right.as_deref(), lo, hi, result);
    }
}

// ============================================================================
// Rotations (used during insertion to restore heap invariant)
// ============================================================================

/// Right rotation:
/// ```text
///     y              x
///    / \            / \
///   x   C    →    A   y
///  / \                / \
/// A   B              B   C
/// ```
fn rotate_right<K: Ord, V>(mut y: Box<Node<K, V>>) -> Box<Node<K, V>> {
    let mut x = y.left.take().expect("rotate_right requires left child");
    y.left = x.right.take();
    y.update_size();
    x.right = Some(y);
    x.update_size();
    x
}

/// Left rotation:
/// ```text
///   x                y
///  / \              / \
/// A   y     →      x   C
///    / \          / \
///   B   C        A   B
/// ```
fn rotate_left<K: Ord, V>(mut x: Box<Node<K, V>>) -> Box<Node<K, V>> {
    let mut y = x.right.take().expect("rotate_left requires right child");
    x.right = y.left.take();
    x.update_size();
    y.left = Some(x);
    y.update_size();
    y
}

// ============================================================================
// Trait implementations
// ============================================================================

impl<K: Ord + Clone, V> Default for Treap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord + Clone + fmt::Debug, V: fmt::Debug> fmt::Debug for Treap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.inorder()).finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_insert_get() {
        let mut t = Treap::new();
        t.insert(5i32, "five");
        t.insert(3i32, "three");
        t.insert(8i32, "eight");

        assert_eq!(t.get(&5), Some(&"five"));
        assert_eq!(t.get(&3), Some(&"three"));
        assert_eq!(t.get(&8), Some(&"eight"));
        assert_eq!(t.get(&99), None);
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn update_existing_key() {
        let mut t = Treap::new();
        t.insert(10i32, "old");
        let prev = t.insert(10i32, "new");
        assert_eq!(prev, Some("old"));
        assert_eq!(t.get(&10), Some(&"new"));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn remove_present_and_absent() {
        let mut t = Treap::new();
        t.insert(1i32, 100);
        t.insert(2i32, 200);
        t.insert(3i32, 300);

        assert_eq!(t.remove(&2), Some(200));
        assert_eq!(t.len(), 2);
        assert_eq!(t.remove(&2), None);
        assert!(!t.contains_key(&2));
    }

    #[test]
    fn inorder_yields_sorted_keys() {
        let mut t = Treap::new();
        let keys = [7i32, 3, 15, 1, 5, 10, 20];
        for &k in &keys {
            t.insert(k, k * 2);
        }
        let order: Vec<i32> = t.inorder().iter().map(|(&k, _)| k).collect();
        let mut expected = keys.to_vec();
        expected.sort();
        assert_eq!(order, expected);
    }

    #[test]
    fn range_query_correct() {
        let mut t = Treap::new();
        for k in 0i32..20 {
            t.insert(k, k);
        }
        let results = t.range_query(&5, &10);
        let keys: Vec<i32> = results.iter().map(|(&k, _)| k).collect();
        assert_eq!(keys, (5i32..=10).collect::<Vec<_>>());
    }

    #[test]
    fn range_query_empty_range() {
        let mut t = Treap::new();
        for k in [1i32, 3, 5, 7, 9] {
            t.insert(k, k);
        }
        // Range [2, 2] contains no odd keys.
        let results = t.range_query(&2, &2);
        assert!(results.is_empty());
    }

    #[test]
    fn split_and_merge_roundtrip() {
        let mut t = Treap::new();
        for k in 0i32..10 {
            t.insert(k, k * 10);
        }
        let (left, right) = t.split(&5);
        // left: keys [0, 5); right: keys [5, 10)
        assert!(left.inorder().iter().all(|(&k, _)| k < 5));
        assert!(right.inorder().iter().all(|(&k, _)| k >= 5));

        let merged = Treap::merge(left, right);
        assert_eq!(merged.len(), 10);
        let order: Vec<i32> = merged.inorder().iter().map(|(&k, _)| k).collect();
        assert_eq!(order, (0i32..10).collect::<Vec<_>>());
    }

    #[test]
    fn min_max() {
        let mut t = Treap::new();
        for k in [4i32, 2, 9, 1, 7] {
            t.insert(k, ());
        }
        assert_eq!(t.min().map(|(&k, _)| k), Some(1));
        assert_eq!(t.max().map(|(&k, _)| k), Some(9));
    }

    #[test]
    fn large_insert_remove() {
        let mut t = Treap::new();
        for i in 0u64..1000 {
            t.insert(i, i * i);
        }
        assert_eq!(t.len(), 1000);
        for i in (0u64..1000).step_by(2) {
            assert_eq!(t.remove(&i), Some(i * i));
        }
        assert_eq!(t.len(), 500);
        // Verify remaining keys.
        let keys: Vec<u64> = t.inorder().iter().map(|(&k, _)| k).collect();
        let expected: Vec<u64> = (1u64..1000).step_by(2).collect();
        assert_eq!(keys, expected);
    }

    #[test]
    fn get_mut_updates_value() {
        let mut t = Treap::new();
        t.insert(42i32, 0i32);
        if let Some(v) = t.get_mut(&42) {
            *v += 100;
        }
        assert_eq!(t.get(&42), Some(&100));
    }

    #[test]
    fn empty_treap_invariants() {
        let t: Treap<i32, i32> = Treap::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert_eq!(t.min(), None);
        assert_eq!(t.max(), None);
        assert!(t.inorder().is_empty());
        assert!(t.range_query(&0, &100).is_empty());
    }
}
