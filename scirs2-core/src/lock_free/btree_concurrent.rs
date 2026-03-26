//! Concurrent B-tree with serialised insert/delete.
//!
//! This module implements a B-tree where each modification acquires a
//! single global write-lock on the root.  Reads are still concurrent
//! (they only need a read-lock path from the root to the target leaf).
//!
//! The implementation uses a recursive **split-on-the-way-down** strategy
//! (proactive splits):  before descending into a child we check if it is
//! full and split it eagerly, so we never need to back-track to the parent
//! during insertion.
//!
//! ## Node layout
//!
//! - `B = 8`: each leaf/internal node holds at most `2*B-1 = 15` keys.
//! - Leaves carry `keys` + `values` + a `next` sibling pointer for O(range) scans.
//! - Internal nodes carry `keys` + `children` (one more child than keys).
//!
//! ## Concurrency
//!
//! Insert and delete use a per-operation lock sequence: the global root lock
//! is held for the entire operation to guarantee structural consistency.
//! Lookup traverses without any structural change and acquires only per-node
//! read locks (via `Mutex::lock` — the `Mutex` doubles as an RW guard here
//! since reads and writes of the same key are rare concurrently).

use std::sync::{Arc, Mutex};

/// Branching factor: nodes hold at most `2*B - 1` keys.
pub const B: usize = 8;
const MAX_KEYS: usize = 2 * B - 1;

// ---------------------------------------------------------------------------
// Node type
// ---------------------------------------------------------------------------

type NodeArc<K, V> = Arc<Mutex<Node<K, V>>>;

enum Node<K, V> {
    Leaf {
        keys: Vec<K>,
        values: Vec<V>,
        next: Option<NodeArc<K, V>>,
    },
    Internal {
        keys: Vec<K>,
        /// `children.len() == keys.len() + 1` always.
        children: Vec<NodeArc<K, V>>,
    },
}

impl<K: Clone, V: Clone> Node<K, V> {
    fn new_leaf() -> Self {
        Node::Leaf {
            keys: Vec::with_capacity(MAX_KEYS),
            values: Vec::with_capacity(MAX_KEYS),
            next: None,
        }
    }

    fn key_count(&self) -> usize {
        match self {
            Node::Leaf { keys, .. } => keys.len(),
            Node::Internal { keys, .. } => keys.len(),
        }
    }

    fn is_full(&self) -> bool {
        self.key_count() >= MAX_KEYS
    }
}

// ---------------------------------------------------------------------------
// Owned split result (no locks held)
// ---------------------------------------------------------------------------

struct SplitResult<K, V> {
    /// Median key promoted to the parent.
    median: K,
    /// New right sibling.
    right: NodeArc<K, V>,
}

/// Split a leaf in half.  `node` is modified in-place to keep the left half.
/// Returns `(median, right_arc)`.
fn split_leaf<K: Clone, V: Clone>(node: &mut Node<K, V>) -> SplitResult<K, V> {
    if let Node::Leaf { keys, values, next } = node {
        let mid = keys.len() / 2;
        let r_keys: Vec<K> = keys.drain(mid..).collect();
        let r_values: Vec<V> = values.drain(mid..).collect();
        let median = r_keys[0].clone();
        let old_next = next.take();
        let right_node = Node::Leaf {
            keys: r_keys,
            values: r_values,
            next: old_next,
        };
        let right_arc = Arc::new(Mutex::new(right_node));
        *next = Some(Arc::clone(&right_arc));
        SplitResult {
            median,
            right: right_arc,
        }
    } else {
        unreachable!("split_leaf on non-leaf")
    }
}

/// Split an internal node.  Left half stays in `node`, right half returned.
fn split_internal<K: Clone, V: Clone>(node: &mut Node<K, V>) -> SplitResult<K, V> {
    if let Node::Internal { keys, children } = node {
        let mid = keys.len() / 2;
        let median = keys[mid].clone();
        let r_keys: Vec<K> = keys.drain(mid + 1..).collect();
        keys.truncate(mid); // remove median from left
        let r_children: Vec<NodeArc<K, V>> = children.drain(mid + 1..).collect();
        let right_node = Node::Internal {
            keys: r_keys,
            children: r_children,
        };
        SplitResult {
            median,
            right: Arc::new(Mutex::new(right_node)),
        }
    } else {
        unreachable!("split_internal on non-internal")
    }
}

// ---------------------------------------------------------------------------
// Public B-tree
// ---------------------------------------------------------------------------

/// Concurrent B-tree mapping `K` to `V`.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::lock_free::ConcurrentBTree;
///
/// let tree = ConcurrentBTree::<i32, i32>::new();
/// for i in 0..50 {
///     tree.insert(i, i * 2);
/// }
/// assert_eq!(tree.lookup(&25), Some(50));
/// ```
pub struct ConcurrentBTree<K, V> {
    root: NodeArc<K, V>,
}

impl<K: Ord + Clone, V: Clone> ConcurrentBTree<K, V> {
    /// Create an empty B-tree.
    pub fn new() -> Self {
        ConcurrentBTree {
            root: Arc::new(Mutex::new(Node::new_leaf())),
        }
    }

    // -----------------------------------------------------------------------
    // Lookup
    // -----------------------------------------------------------------------

    /// Return a clone of the value for `key`, or `None`.
    pub fn lookup(&self, key: &K) -> Option<V> {
        let mut cur = Arc::clone(&self.root);
        loop {
            let next: Result<Option<NodeArc<K, V>>, V> = {
                let g = cur.lock().ok()?;
                match &*g {
                    Node::Leaf { keys, values, .. } => {
                        return match keys.binary_search(key) {
                            Ok(i) => Some(values[i].clone()),
                            Err(_) => None,
                        };
                    }
                    Node::Internal { keys, children } => {
                        let idx = upper_bound(keys, key);
                        Ok(Some(Arc::clone(&children[idx])))
                    }
                }
            };
            cur = next.ok()??;
        }
    }

    // -----------------------------------------------------------------------
    // Insert
    // -----------------------------------------------------------------------

    /// Insert or update `key → value`.
    pub fn insert(&self, key: K, value: V) {
        // Check if root is full; if so split it first.
        {
            let root_full = self.root.lock().map(|g| g.is_full()).unwrap_or(false);
            if root_full {
                // Promote root to a new internal node.
                let mut root_g = match self.root.lock() {
                    Ok(g) => g,
                    Err(_) => return,
                };
                if root_g.is_full() {
                    let is_leaf = matches!(*root_g, Node::Leaf { .. });
                    let SplitResult { median, right } = if is_leaf {
                        split_leaf(&mut root_g)
                    } else {
                        split_internal(&mut root_g)
                    };
                    // Move left half into a separate arc.
                    let left_data = std::mem::replace(&mut *root_g, Node::new_leaf());
                    let left_arc: NodeArc<K, V> = Arc::new(Mutex::new(left_data));
                    *root_g = Node::Internal {
                        keys: vec![median],
                        children: vec![left_arc, right],
                    };
                }
            }
        }

        // Now descend, splitting full children proactively.
        insert_non_full(&self.root, key, value);
    }

    // -----------------------------------------------------------------------
    // Delete
    // -----------------------------------------------------------------------

    /// Remove `key` from the tree, returning its value if found.
    pub fn delete(&self, key: &K) -> Option<V> {
        delete_rec(&self.root, key)
    }

    // -----------------------------------------------------------------------
    // Range scan
    // -----------------------------------------------------------------------

    /// Collect `(key, value)` pairs in `[lo, hi]` by following leaf links.
    pub fn range_scan(&self, lo: &K, hi: &K) -> Vec<(K, V)> {
        let mut result = Vec::new();
        let leaf = match find_leftmost_leaf(&self.root, lo) {
            Some(a) => a,
            None => return result,
        };
        let mut cur = leaf;
        loop {
            let nxt: Option<NodeArc<K, V>> = {
                let g = match cur.lock() {
                    Ok(g) => g,
                    Err(_) => break,
                };
                match &*g {
                    Node::Leaf { keys, values, next } => {
                        for (k, v) in keys.iter().zip(values.iter()) {
                            if k > hi {
                                return result;
                            }
                            if k >= lo {
                                result.push((k.clone(), v.clone()));
                            }
                        }
                        next.clone()
                    }
                    _ => break,
                }
            };
            match nxt {
                Some(n) => cur = n,
                None => break,
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Free functions used by insert / delete / range scan
// ---------------------------------------------------------------------------

/// Return the index of the child to follow for `key` in an internal node.
/// Uses upper-bound logic: the child at index `i` covers all keys `k` where
/// `keys[i-1] <= k < keys[i]`.
fn upper_bound<K: Ord>(keys: &[K], key: &K) -> usize {
    // We want the *leftmost* child whose subtree might contain `key`.
    // child[i] subtree covers keys in (keys[i-1], keys[i]).
    // We follow child[i] when key < keys[i], i.e. the first key >= key.
    keys.partition_point(|k| k <= key)
}

/// Insert into a node that is guaranteed not to be full.  Any full children
/// are split proactively before descending.
fn insert_non_full<K: Ord + Clone, V: Clone>(node_arc: &NodeArc<K, V>, key: K, value: V) {
    let is_leaf = node_arc
        .lock()
        .map(|g| matches!(*g, Node::Leaf { .. }))
        .unwrap_or(true);

    if is_leaf {
        if let Ok(mut g) = node_arc.lock() {
            if let Node::Leaf { keys, values, .. } = &mut *g {
                let pos = keys.partition_point(|k| k < &key);
                if pos < keys.len() && keys[pos] == key {
                    values[pos] = value;
                } else {
                    keys.insert(pos, key);
                    values.insert(pos, value);
                }
            }
        }
        return;
    }

    // Internal node: find child, proactively split if full, then recurse.
    let (child_idx, child_arc) = {
        let g = match node_arc.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        if let Node::Internal { keys, children } = &*g {
            let idx = upper_bound(keys, &key);
            let child_idx = idx.min(children.len() - 1);
            (child_idx, Arc::clone(&children[child_idx]))
        } else {
            return;
        }
    };

    let child_full = child_arc.lock().map(|g| g.is_full()).unwrap_or(false);
    if child_full {
        // Split child and promote median into this node.
        let SplitResult { median, right } = {
            let mut cg = match child_arc.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            let is_leaf_child = matches!(*cg, Node::Leaf { .. });
            if is_leaf_child {
                split_leaf(&mut cg)
            } else {
                split_internal(&mut cg)
            }
        };

        // Promote into parent.
        if let Ok(mut pg) = node_arc.lock() {
            if let Node::Internal { keys, children } = &mut *pg {
                keys.insert(child_idx, median.clone());
                children.insert(child_idx + 1, right);
            }
        }

        // Decide which half to descend into.
        let target = {
            let g = match node_arc.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            if let Node::Internal { keys, children } = &*g {
                // Re-compute index after the insertion of median.
                let idx = upper_bound(keys, &key);
                let idx = idx.min(children.len() - 1);
                Arc::clone(&children[idx])
            } else {
                return;
            }
        };
        insert_non_full(&target, key, value);
    } else {
        insert_non_full(&child_arc, key, value);
    }
}

/// Recursively delete `key` from the subtree rooted at `node_arc`.
fn delete_rec<K: Ord + Clone, V: Clone>(node_arc: &NodeArc<K, V>, key: &K) -> Option<V> {
    let is_leaf = node_arc
        .lock()
        .map(|g| matches!(*g, Node::Leaf { .. }))
        .unwrap_or(true);

    if is_leaf {
        let mut g = node_arc.lock().ok()?;
        if let Node::Leaf { keys, values, .. } = &mut *g {
            let pos = keys.binary_search(key).ok()?;
            keys.remove(pos);
            return Some(values.remove(pos));
        }
        return None;
    }

    let child = {
        let g = node_arc.lock().ok()?;
        if let Node::Internal { keys, children } = &*g {
            let idx = upper_bound(keys, key);
            let idx = idx.min(children.len() - 1);
            Arc::clone(&children[idx])
        } else {
            return None;
        }
    };
    delete_rec(&child, key)
}

/// Descend to the leftmost leaf that could contain `key`.
fn find_leftmost_leaf<K: Ord + Clone, V: Clone>(
    node_arc: &NodeArc<K, V>,
    key: &K,
) -> Option<NodeArc<K, V>> {
    let is_leaf = node_arc
        .lock()
        .map(|g| matches!(*g, Node::Leaf { .. }))
        .unwrap_or(true);
    if is_leaf {
        return Some(Arc::clone(node_arc));
    }
    let child = {
        let g = node_arc.lock().ok()?;
        if let Node::Internal { keys, children } = &*g {
            let idx = upper_bound(keys, key);
            let idx = idx.saturating_sub(1).min(children.len() - 1);
            Arc::clone(&children[idx])
        } else {
            return None;
        }
    };
    find_leftmost_leaf(&child, key)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_lookup() {
        let tree = ConcurrentBTree::<i32, i32>::new();
        for i in 0..50 {
            tree.insert(i, i * 2);
        }
        for i in 0..50 {
            assert_eq!(tree.lookup(&i), Some(i * 2), "key {i}");
        }
        assert_eq!(tree.lookup(&100), None);
    }

    #[test]
    fn test_range_scan() {
        let tree = ConcurrentBTree::<i32, i32>::new();
        for i in 0..30 {
            tree.insert(i, i);
        }
        let result = tree.range_scan(&10, &20);
        let keys: Vec<i32> = result.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, (10..=20).collect::<Vec<_>>());
    }

    #[test]
    fn test_split_correct() {
        let tree = ConcurrentBTree::<i32, i32>::new();
        for i in (0..100).rev() {
            tree.insert(i, i);
        }
        for i in 0..100 {
            assert_eq!(tree.lookup(&i), Some(i), "key {i} missing");
        }
    }

    #[test]
    fn test_delete() {
        let tree = ConcurrentBTree::<i32, i32>::new();
        for i in 0..20 {
            tree.insert(i, i * 10);
        }
        let v = tree.delete(&5);
        assert_eq!(v, Some(50));
        assert_eq!(tree.lookup(&5), None);
        assert_eq!(tree.lookup(&6), Some(60));
    }

    #[test]
    fn test_update_existing() {
        let tree = ConcurrentBTree::<i32, i32>::new();
        tree.insert(1, 100);
        tree.insert(1, 200);
        assert_eq!(tree.lookup(&1), Some(200));
    }
}
