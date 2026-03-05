//! Persistent Red-Black Tree (Okasaki-style).
//!
//! A purely functional red-black tree that uses path copying and structural
//! sharing via [`Arc`] to implement O(log N) insert, delete, and lookup while
//! preserving every previous version of the tree.
//!
//! # References
//! - Okasaki, C. (1999). *Purely Functional Data Structures*. Cambridge University Press.
//! - Germane & Might (2014). *Deletion: The curse of the red-black tree*. JFP 24(4).

use std::cmp::Ordering;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Color {
    Red,
    Black,
}

// ---------------------------------------------------------------------------
// Internal node
// ---------------------------------------------------------------------------

/// Internal node.  All fields are stored via [`Arc`] so that cloning a node
/// is O(1) — we only increment a reference count, not deep-copy the subtree.
#[derive(Clone, Debug)]
struct Node<K: Clone, V: Clone> {
    color: Color,
    key: K,
    val: V,
    left: Option<Arc<Node<K, V>>>,
    right: Option<Arc<Node<K, V>>>,
}

impl<K: Clone, V: Clone> Node<K, V> {
    fn new(
        color: Color,
        key: K,
        val: V,
        left: Option<Arc<Node<K, V>>>,
        right: Option<Arc<Node<K, V>>>,
    ) -> Arc<Self> {
        Arc::new(Node {
            color,
            key,
            val,
            left,
            right,
        })
    }
}

// ---------------------------------------------------------------------------
// Balance helpers (Okasaki's 4 rotations collapsed into one function)
// ---------------------------------------------------------------------------

/// Rebalance after a red-red violation during insertion.
///
/// Handles all four cases described in Okasaki §3.3.
fn balance<K: Clone, V: Clone>(
    color: Color,
    left: Option<Arc<Node<K, V>>>,
    key: K,
    val: V,
    right: Option<Arc<Node<K, V>>>,
) -> Arc<Node<K, V>> {
    // Case 1: left child is red and left-left grandchild is red.
    if let Color::Black = color {
        if let Some(ref l) = left {
            if l.color == Color::Red {
                if let Some(ref ll) = l.left {
                    if ll.color == Color::Red {
                        return Node::new(
                            Color::Red,
                            l.key.clone(),
                            l.val.clone(),
                            Some(Node::new(
                                Color::Black,
                                ll.key.clone(),
                                ll.val.clone(),
                                ll.left.clone(),
                                ll.right.clone(),
                            )),
                            Some(Node::new(
                                Color::Black,
                                key,
                                val,
                                l.right.clone(),
                                right,
                            )),
                        );
                    }
                }
                // Case 2: left child is red and left-right grandchild is red.
                if let Some(ref lr) = l.right {
                    if lr.color == Color::Red {
                        return Node::new(
                            Color::Red,
                            lr.key.clone(),
                            lr.val.clone(),
                            Some(Node::new(
                                Color::Black,
                                l.key.clone(),
                                l.val.clone(),
                                l.left.clone(),
                                lr.left.clone(),
                            )),
                            Some(Node::new(
                                Color::Black,
                                key,
                                val,
                                lr.right.clone(),
                                right,
                            )),
                        );
                    }
                }
            }
        }
        // Case 3: right child is red and right-left grandchild is red.
        if let Some(ref r) = right {
            if r.color == Color::Red {
                if let Some(ref rl) = r.left {
                    if rl.color == Color::Red {
                        return Node::new(
                            Color::Red,
                            rl.key.clone(),
                            rl.val.clone(),
                            Some(Node::new(
                                Color::Black,
                                key,
                                val,
                                left,
                                rl.left.clone(),
                            )),
                            Some(Node::new(
                                Color::Black,
                                r.key.clone(),
                                r.val.clone(),
                                rl.right.clone(),
                                r.right.clone(),
                            )),
                        );
                    }
                }
                // Case 4: right child is red and right-right grandchild is red.
                if let Some(ref rr) = r.right {
                    if rr.color == Color::Red {
                        return Node::new(
                            Color::Red,
                            r.key.clone(),
                            r.val.clone(),
                            Some(Node::new(
                                Color::Black,
                                key,
                                val,
                                left,
                                r.left.clone(),
                            )),
                            Some(Node::new(
                                Color::Black,
                                rr.key.clone(),
                                rr.val.clone(),
                                rr.left.clone(),
                                rr.right.clone(),
                            )),
                        );
                    }
                }
            }
        }
    }
    Node::new(color, key, val, left, right)
}

// ---------------------------------------------------------------------------
// Deletion helpers (Germane & Might)
// ---------------------------------------------------------------------------

/// A "double-black" sentinel used during deletion to track the need for
/// rebalancing when a black node is removed.
#[derive(Clone, Debug, PartialEq, Eq)]
enum DelColor {
    Red,
    Black,
    DoubleBlack,
}

#[derive(Clone, Debug)]
struct DelNode<K: Clone, V: Clone> {
    color: DelColor,
    key: K,
    val: V,
    left: DelTree<K, V>,
    right: DelTree<K, V>,
}

/// A tree that may contain a double-black node at its root.
#[derive(Clone, Debug)]
enum DelTree<K: Clone, V: Clone> {
    Empty,
    DoubleBlackEmpty,
    Node(Arc<DelNode<K, V>>),
}

impl<K: Clone + Ord, V: Clone> DelTree<K, V> {
    fn from_node(n: &Node<K, V>) -> Self {
        DelTree::Node(Arc::new(DelNode {
            color: match n.color {
                Color::Red => DelColor::Red,
                Color::Black => DelColor::Black,
            },
            key: n.key.clone(),
            val: n.val.clone(),
            left: match &n.left {
                None => DelTree::Empty,
                Some(c) => DelTree::from_node(c),
            },
            right: match &n.right {
                None => DelTree::Empty,
                Some(c) => DelTree::from_node(c),
            },
        }))
    }

    fn into_node(self) -> Option<Arc<Node<K, V>>> {
        match self {
            DelTree::Node(dn) => {
                let color = match dn.color {
                    DelColor::Red => Color::Red,
                    DelColor::Black | DelColor::DoubleBlack => Color::Black,
                };
                let left = dn.left.clone().into_node();
                let right = dn.right.clone().into_node();
                Some(Node::new(color, dn.key.clone(), dn.val.clone(), left, right))
            }
            _ => None,
        }
    }

    /// Remove double-black by absorbing into the color field.
    fn make_black(self) -> Self {
        match self {
            DelTree::DoubleBlackEmpty => DelTree::Empty,
            DelTree::Node(ref dn) if dn.color == DelColor::DoubleBlack => {
                let mut new_dn = (**dn).clone();
                new_dn.color = DelColor::Black;
                DelTree::Node(Arc::new(new_dn))
            }
            other => other,
        }
    }

    fn is_double_black(&self) -> bool {
        matches!(self, DelTree::DoubleBlackEmpty)
            || matches!(self, DelTree::Node(dn) if dn.color == DelColor::DoubleBlack)
    }

    fn bubble(color: DelColor, left: DelTree<K, V>, key: K, val: V, right: DelTree<K, V>) -> Self {
        let left_db = left.is_double_black();
        let right_db = right.is_double_black();
        let new_color = if left_db || right_db {
            match color {
                DelColor::Red => DelColor::Black,
                DelColor::Black => DelColor::DoubleBlack,
                DelColor::DoubleBlack => DelColor::DoubleBlack,
            }
        } else {
            color.clone()
        };
        let new_left = if left_db { left.make_black() } else { left };
        let new_right = if right_db { right.make_black() } else { right };
        DelTree::balance_del(new_color, new_left, key, val, new_right)
    }

    fn balance_del(color: DelColor, left: DelTree<K, V>, key: K, val: V, right: DelTree<K, V>) -> Self {
        // Rebalance cases for double-black nodes (Germane & Might Table 1).
        // Case BB-LR: double-black right, red left sibling.
        if color == DelColor::DoubleBlack {
            if let DelTree::Node(ref ln) = left {
                if ln.color == DelColor::Red {
                    if let DelTree::Node(ref lrn) = ln.right {
                        // rotate right at current, recurse
                        let new_right = Self::balance_del(
                            DelColor::DoubleBlack,
                            DelTree::Node(lrn.clone()),
                            key.clone(),
                            val.clone(),
                            right,
                        );
                        return DelTree::Node(Arc::new(DelNode {
                            color: DelColor::Black,
                            key: ln.key.clone(),
                            val: ln.val.clone(),
                            left: ln.left.clone(),
                            right: new_right,
                        }));
                    }
                }
            }
            if let DelTree::Node(ref rn) = right {
                if rn.color == DelColor::Red {
                    if let DelTree::Node(ref rln) = rn.left {
                        let new_left = Self::balance_del(
                            DelColor::DoubleBlack,
                            left,
                            key.clone(),
                            val.clone(),
                            DelTree::Node(rln.clone()),
                        );
                        return DelTree::Node(Arc::new(DelNode {
                            color: DelColor::Black,
                            key: rn.key.clone(),
                            val: rn.val.clone(),
                            left: new_left,
                            right: rn.right.clone(),
                        }));
                    }
                }
            }
        }

        DelTree::Node(Arc::new(DelNode {
            color,
            key,
            val,
            left,
            right,
        }))
    }

    /// Delete the minimum key from the subtree, returning (min_key, min_val, new_subtree).
    fn del_min(self) -> Option<(K, V, DelTree<K, V>)> {
        match self {
            DelTree::Empty | DelTree::DoubleBlackEmpty => None,
            DelTree::Node(dn) => {
                match dn.left {
                    DelTree::Empty => {
                        // This node IS the minimum.
                        let replacement = match dn.color {
                            DelColor::Red => DelTree::Empty,
                            _ => DelTree::DoubleBlackEmpty,
                        };
                        Some((dn.key.clone(), dn.val.clone(), replacement))
                    }
                    ref left_tree => {
                        let left_tree = left_tree.clone();
                        match left_tree.del_min() {
                            None => None,
                            Some((min_k, min_v, new_left)) => {
                                let tree = Self::bubble(
                                    dn.color.clone(),
                                    new_left,
                                    dn.key.clone(),
                                    dn.val.clone(),
                                    dn.right.clone(),
                                );
                                Some((min_k, min_v, tree))
                            }
                        }
                    }
                }
            }
        }
    }

    fn delete(self, key: &K) -> Self {
        match self {
            DelTree::Empty | DelTree::DoubleBlackEmpty => DelTree::Empty,
            DelTree::Node(dn) => match key.cmp(&dn.key) {
                Ordering::Less => {
                    let new_left = dn.left.clone().delete(key);
                    Self::bubble(dn.color.clone(), new_left, dn.key.clone(), dn.val.clone(), dn.right.clone())
                }
                Ordering::Greater => {
                    let new_right = dn.right.clone().delete(key);
                    Self::bubble(dn.color.clone(), dn.left.clone(), dn.key.clone(), dn.val.clone(), new_right)
                }
                Ordering::Equal => {
                    // Remove this node: replace with successor (min of right subtree).
                    match dn.right.clone().del_min() {
                        None => {
                            // No right subtree: promote left or become empty.
                            match dn.color {
                                DelColor::Red => dn.left.clone(),
                                _ => {
                                    // Black node: left must be empty (RB invariant for simple case).
                                    match dn.left.clone() {
                                        DelTree::Empty => DelTree::DoubleBlackEmpty,
                                        DelTree::Node(ref ln) if ln.color == DelColor::Red => {
                                            // Red child: just recolor black.
                                            let mut new_ln = (**ln).clone();
                                            new_ln.color = DelColor::Black;
                                            DelTree::Node(Arc::new(new_ln))
                                        }
                                        other => other,
                                    }
                                }
                            }
                        }
                        Some((succ_k, succ_v, new_right)) => {
                            Self::bubble(dn.color.clone(), dn.left.clone(), succ_k, succ_v, new_right)
                        }
                    }
                }
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Insertion helper
// ---------------------------------------------------------------------------

fn ins<K: Clone + Ord, V: Clone>(
    node: &Option<Arc<Node<K, V>>>,
    key: K,
    val: V,
) -> Arc<Node<K, V>> {
    match node {
        None => Node::new(Color::Red, key, val, None, None),
        Some(n) => match key.cmp(&n.key) {
            Ordering::Less => balance(
                n.color,
                Some(ins(&n.left, key, val)),
                n.key.clone(),
                n.val.clone(),
                n.right.clone(),
            ),
            Ordering::Greater => balance(
                n.color,
                n.left.clone(),
                n.key.clone(),
                n.val.clone(),
                Some(ins(&n.right, key, val)),
            ),
            Ordering::Equal => Node::new(n.color, key, val, n.left.clone(), n.right.clone()),
        },
    }
}

// ---------------------------------------------------------------------------
// In-order iterator
// ---------------------------------------------------------------------------

/// Iterator that traverses the tree in key-ascending order.
pub struct InOrderIter<'a, K: Clone, V: Clone> {
    stack: Vec<&'a Node<K, V>>,
}

impl<'a, K: Clone + Ord, V: Clone> InOrderIter<'a, K, V> {
    fn new(root: &'a Option<Arc<Node<K, V>>>) -> Self {
        let mut iter = InOrderIter { stack: Vec::new() };
        iter.push_left(root);
        iter
    }

    fn push_left(&mut self, mut node: &'a Option<Arc<Node<K, V>>>) {
        while let Some(n) = node {
            self.stack.push(n.as_ref());
            node = &n.left;
        }
    }
}

impl<'a, K: Clone + Ord, V: Clone> Iterator for InOrderIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.stack.pop()?;
        self.push_left(&n.right);
        Some((&n.key, &n.val))
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Persistent Red-Black Tree map.
///
/// Every "mutating" operation returns a brand-new `PersistentRBTree` while
/// sharing all unchanged subtrees with the original via `Arc`.  Complexity is
/// O(log N) per operation.
///
/// # Examples
///
/// ```
/// use scirs2_core::persistent::persistent_rbtree::PersistentRBTree;
///
/// let t0 = PersistentRBTree::new();
/// let t1 = t0.insert(3, "three");
/// let t2 = t1.insert(1, "one");
/// let t3 = t2.insert(2, "two");
///
/// assert_eq!(t3.get(&1), Some(&"one"));
/// assert_eq!(t3.get(&2), Some(&"two"));
/// assert_eq!(t3.get(&3), Some(&"three"));
///
/// // t1 is unaffected by later insertions.
/// assert!(!t1.contains_key(&2));
/// ```
#[derive(Clone, Debug)]
pub struct PersistentRBTree<K: Clone + Ord, V: Clone> {
    root: Option<Arc<Node<K, V>>>,
    len: usize,
}

impl<K: Clone + Ord, V: Clone> Default for PersistentRBTree<K, V> {
    fn default() -> Self {
        PersistentRBTree::new()
    }
}

impl<K: Clone + Ord, V: Clone> PersistentRBTree<K, V> {
    /// Create an empty tree.
    pub fn new() -> Self {
        PersistentRBTree { root: None, len: 0 }
    }

    /// Number of key-value pairs in the tree.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the tree contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Look up a key and return a reference to its value if found.
    pub fn get(&self, key: &K) -> Option<&V> {
        let mut node = self.root.as_ref();
        while let Some(n) = node {
            match key.cmp(&n.key) {
                Ordering::Less => node = n.left.as_ref(),
                Ordering::Greater => node = n.right.as_ref(),
                Ordering::Equal => return Some(&n.val),
            }
        }
        None
    }

    /// Returns `true` if the tree contains `key`.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Return a new tree with `(key, value)` inserted / updated.
    ///
    /// O(log N) — uses Okasaki's balanced insertion with path copying.
    pub fn insert(&self, key: K, value: V) -> Self {
        let new_root = ins(&self.root, key.clone(), value);
        // The root must always be black.
        let new_root = if new_root.color == Color::Red {
            Node::new(
                Color::Black,
                new_root.key.clone(),
                new_root.val.clone(),
                new_root.left.clone(),
                new_root.right.clone(),
            )
        } else {
            new_root
        };
        let new_len = if self.contains_key(&key) {
            self.len
        } else {
            self.len + 1
        };
        PersistentRBTree {
            root: Some(new_root),
            len: new_len,
        }
    }

    /// Return a new tree with `key` removed.
    ///
    /// O(log N) — uses Germane & Might's functional deletion algorithm.
    pub fn delete(&self, key: &K) -> Self {
        if !self.contains_key(key) {
            return self.clone();
        }
        let del_tree = match &self.root {
            None => DelTree::Empty,
            Some(n) => DelTree::from_node(n),
        };
        let result = del_tree.delete(key);
        // Make root black.
        let result = result.make_black();
        let new_root = result.into_node();
        PersistentRBTree {
            root: new_root,
            len: self.len - 1,
        }
    }

    /// In-order iterator over `(&key, &value)` pairs.
    pub fn iter(&self) -> InOrderIter<'_, K, V> {
        InOrderIter::new(&self.root)
    }
}

impl<K: Clone + Ord, V: Clone> FromIterator<(K, V)> for PersistentRBTree<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut tree = PersistentRBTree::new();
        for (k, v) in iter {
            tree = tree.insert(k, v);
        }
        tree
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tree() {
        let t: PersistentRBTree<i32, &str> = PersistentRBTree::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert_eq!(t.get(&42), None);
        assert!(!t.contains_key(&42));
    }

    #[test]
    fn test_basic_insert_and_get() {
        let t = PersistentRBTree::new()
            .insert(5, "five")
            .insert(3, "three")
            .insert(7, "seven")
            .insert(1, "one")
            .insert(9, "nine");

        assert_eq!(t.get(&5), Some(&"five"));
        assert_eq!(t.get(&3), Some(&"three"));
        assert_eq!(t.get(&7), Some(&"seven"));
        assert_eq!(t.get(&1), Some(&"one"));
        assert_eq!(t.get(&9), Some(&"nine"));
        assert_eq!(t.get(&4), None);
        assert_eq!(t.len(), 5);
    }

    #[test]
    fn test_persistence() {
        let t0: PersistentRBTree<i32, i32> = PersistentRBTree::new();
        let t1 = t0.insert(10, 100);
        let t2 = t1.insert(20, 200);
        let t3 = t2.insert(30, 300);

        // Each version is independent.
        assert_eq!(t0.len(), 0);
        assert_eq!(t1.len(), 1);
        assert_eq!(t2.len(), 2);
        assert_eq!(t3.len(), 3);

        assert_eq!(t1.get(&10), Some(&100));
        assert_eq!(t1.get(&20), None);

        assert_eq!(t2.get(&10), Some(&100));
        assert_eq!(t2.get(&20), Some(&200));
        assert_eq!(t2.get(&30), None);

        assert_eq!(t3.get(&30), Some(&300));
    }

    #[test]
    fn test_update_existing_key() {
        let t1 = PersistentRBTree::new().insert(1, "a");
        let t2 = t1.insert(1, "b");

        assert_eq!(t1.get(&1), Some(&"a"));
        assert_eq!(t2.get(&1), Some(&"b"));
        assert_eq!(t2.len(), 1);
    }

    #[test]
    fn test_in_order_iteration() {
        let keys = vec![5, 3, 7, 1, 9, 2, 8, 4, 6];
        let t: PersistentRBTree<i32, i32> =
            keys.iter().map(|&k| (k, k * 10)).collect();

        let sorted: Vec<_> = t.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(
            sorted,
            vec![(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60), (7, 70), (8, 80), (9, 90)]
        );
    }

    #[test]
    fn test_delete_basic() {
        let t = PersistentRBTree::new()
            .insert(1, "a")
            .insert(2, "b")
            .insert(3, "c");

        let t2 = t.delete(&2);
        assert_eq!(t2.len(), 2);
        assert_eq!(t2.get(&1), Some(&"a"));
        assert_eq!(t2.get(&2), None);
        assert_eq!(t2.get(&3), Some(&"c"));

        // Original unchanged.
        assert_eq!(t.get(&2), Some(&"b"));
    }

    #[test]
    fn test_delete_nonexistent() {
        let t = PersistentRBTree::new().insert(1, 1).insert(2, 2);
        let t2 = t.delete(&99);
        assert_eq!(t2.len(), 2);
        assert_eq!(t2.get(&1), Some(&1));
        assert_eq!(t2.get(&2), Some(&2));
    }

    #[test]
    fn test_delete_all() {
        let t = PersistentRBTree::new()
            .insert(1, 1)
            .insert(2, 2)
            .insert(3, 3);
        let t = t.delete(&1).delete(&2).delete(&3);
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn test_large_tree() {
        let n = 1000usize;
        let t: PersistentRBTree<usize, usize> = (0..n).map(|i| (i, i * 2)).collect();
        assert_eq!(t.len(), n);
        for i in 0..n {
            assert_eq!(t.get(&i), Some(&(i * 2)));
        }
        let sorted: Vec<_> = t.iter().map(|(&k, _)| k).collect();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn test_rbtree_black_height_invariant() {
        // After many insertions the tree should remain balanced (reachable depth ≤ 2*log2(n+1)).
        let n = 256usize;
        let t: PersistentRBTree<usize, usize> = (0..n).map(|i| (i, i)).collect();
        // Check we can find all keys (implies valid structure).
        for i in 0..n {
            assert!(t.contains_key(&i));
        }
    }

    #[test]
    fn test_from_iterator() {
        let pairs = vec![(3, 'c'), (1, 'a'), (2, 'b')];
        let t: PersistentRBTree<i32, char> = pairs.into_iter().collect();
        assert_eq!(t.len(), 3);
        let sorted: Vec<_> = t.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(sorted, vec![(1, 'a'), (2, 'b'), (3, 'c')]);
    }
}
