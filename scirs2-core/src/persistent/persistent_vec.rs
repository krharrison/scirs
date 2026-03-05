//! Persistent vector based on a 32-ary trie (Clojure-style PersistentVector).
//!
//! Every "mutating" operation returns a brand-new `PersistentVec` that shares
//! all unchanged subtrees with the original via `Arc`.  The amortised cost of
//! `push`, `pop`, `get`, and `set` is O(log₃₂ N) node allocations.
//!
//! # Representation
//!
//! ```text
//! depth=0  leaf: [T; ≤ BRANCHING]
//! depth=1  internal: [Arc<Node<T>>; ≤ BRANCHING]
//!          …
//! ```
//!
//! The "tail" is a detached leaf that accumulates elements before being pushed
//! into the trie, giving true O(1) amortised `push`.

use std::sync::Arc;

/// Branching factor (2^5 = 32).
const BRANCHING: usize = 32;
/// Bit-mask for one level of index.
const MASK: usize = BRANCHING - 1;
/// Bits consumed per level.
const BITS: usize = 5;

// ---------------------------------------------------------------------------
// Internal node types
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Node<T: Clone> {
    Internal(Arc<[Arc<Node<T>>]>),
    Leaf(Arc<[T]>),
}

impl<T: Clone> Node<T> {
    /// Return a slice of child nodes (panics if called on a leaf).
    fn children(&self) -> &[Arc<Node<T>>] {
        match self {
            Node::Internal(ch) => ch,
            Node::Leaf(_) => panic!("children() called on Leaf"),
        }
    }

    /// Return a slice of elements (panics if called on an internal node).
    fn elements(&self) -> &[T] {
        match self {
            Node::Leaf(elems) => elems,
            Node::Internal(_) => panic!("elements() called on Internal"),
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Persistent vector backed by a 32-ary trie.
///
/// All operations are non-destructive: each "update" returns a new version
/// while the old version remains valid.
///
/// ```
/// use scirs2_core::persistent::PersistentVec;
///
/// let v0 = PersistentVec::new();
/// let v1 = v0.push(1u32);
/// let v2 = v1.push(2u32);
/// let v3 = v2.set(0, 42u32);
///
/// assert_eq!(v3.get(0), Some(&42));
/// assert_eq!(v3.get(1), Some(&2));
/// assert_eq!(v2.get(0), Some(&1)); // v2 is unchanged
/// ```
#[derive(Clone)]
pub struct PersistentVec<T: Clone> {
    /// Total number of elements.
    len: usize,
    /// Height of the trie (0 means root is the only leaf / trie is empty).
    shift: usize,
    /// Root of the trie (may be an empty internal node for the empty vec).
    root: Arc<Node<T>>,
    /// Tail buffer — not yet pushed into the trie.
    tail: Arc<[T]>,
}

impl<T: Clone> PersistentVec<T> {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Creates an empty `PersistentVec`.
    pub fn new() -> Self {
        PersistentVec {
            len: 0,
            shift: BITS,
            root: Arc::new(Node::Internal(Arc::from([] as [Arc<Node<T>>; 0]))),
            tail: Arc::from([] as [T; 0]),
        }
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a reference to the element at `idx`, or `None` if out of bounds.
    pub fn get(&self, idx: usize) -> Option<&T> {
        if idx >= self.len {
            return None;
        }
        // Elements in the tail?
        if idx >= self.tail_offset() {
            let tail_idx = idx - self.tail_offset();
            return self.tail.get(tail_idx);
        }
        // Navigate trie using references to avoid local Arc lifetime issues.
        let mut node: &Node<T> = &self.root;
        let mut level = self.shift;
        loop {
            match node {
                Node::Internal(children) => {
                    let child_idx = (idx >> level) & MASK;
                    if level == BITS {
                        // Next level is leaves.
                        node = &children[child_idx];
                        break;
                    }
                    node = &children[child_idx];
                    level -= BITS;
                }
                Node::Leaf(_) => break,
            }
        }
        node.elements().get(idx & MASK)
    }

    // ------------------------------------------------------------------
    // Persistent updates
    // ------------------------------------------------------------------

    /// Returns a new vector with `val` appended to the end.
    pub fn push(&self, val: T) -> Self {
        // If there is room in the tail, just extend it.
        if self.tail.len() < BRANCHING {
            let mut new_tail = self.tail.to_vec();
            new_tail.push(val);
            return PersistentVec {
                len: self.len + 1,
                shift: self.shift,
                root: Arc::clone(&self.root),
                tail: Arc::from(new_tail.as_slice()),
            };
        }

        // Tail is full — push it into the trie, then start a new tail.
        let tail_node = Arc::new(Node::Leaf(Arc::clone(&self.tail)));
        let (new_root, new_shift) = self.push_tail(Arc::clone(&self.root), tail_node, self.shift);

        PersistentVec {
            len: self.len + 1,
            shift: new_shift,
            root: new_root,
            tail: Arc::from([val].as_slice()),
        }
    }

    /// Returns a new vector with the last element removed.
    /// The removed element is returned alongside the new vector.
    pub fn pop(&self) -> (Self, Option<T>) {
        if self.len == 0 {
            return (self.clone(), None);
        }
        let last = self.get(self.len - 1).cloned();

        if self.tail.len() > 1 {
            // Just trim the tail.
            let new_tail: Arc<[T]> = Arc::from(&self.tail[..self.tail.len() - 1]);
            let new_vec = PersistentVec {
                len: self.len - 1,
                shift: self.shift,
                root: Arc::clone(&self.root),
                tail: new_tail,
            };
            return (new_vec, last);
        }

        // Tail has exactly one element — pull a new tail from the trie.
        if self.tail_offset() == 0 {
            // Trie is empty after pop; just remove the tail entirely.
            let new_vec = PersistentVec {
                len: self.len - 1,
                shift: BITS,
                root: Arc::new(Node::Internal(Arc::from([] as [Arc<Node<T>>; 0]))),
                tail: Arc::from([] as [T; 0]),
            };
            return (new_vec, last);
        }

        // Find the rightmost leaf in the trie as the new tail.
        let new_tail_leaf = self.find_leaf(self.tail_offset() - 1);
        let new_tail: Arc<[T]> = Arc::clone(
            if let Node::Leaf(ref elems) = *new_tail_leaf {
                elems
            } else {
                unreachable!("find_leaf must return a Leaf")
            },
        );

        let (new_root, new_shift) =
            self.pop_tail(Arc::clone(&self.root), self.tail_offset() - 1, self.shift);

        let new_vec = PersistentVec {
            len: self.len - 1,
            shift: new_shift,
            root: new_root,
            tail: new_tail,
        };
        (new_vec, last)
    }

    /// Returns a new vector where the element at `idx` is replaced by `val`.
    /// Returns `None` if `idx` is out of bounds.
    pub fn set(&self, idx: usize, val: T) -> Self {
        assert!(idx < self.len, "index out of bounds");

        if idx >= self.tail_offset() {
            // Update in the tail.
            let mut new_tail = self.tail.to_vec();
            new_tail[idx - self.tail_offset()] = val;
            return PersistentVec {
                len: self.len,
                shift: self.shift,
                root: Arc::clone(&self.root),
                tail: Arc::from(new_tail.as_slice()),
            };
        }

        // Path-copy through the trie.
        let new_root = self.update_trie(Arc::clone(&self.root), idx, val, self.shift);
        PersistentVec {
            len: self.len,
            shift: self.shift,
            root: new_root,
            tail: Arc::clone(&self.tail),
        }
    }

    // ------------------------------------------------------------------
    // Iterator
    // ------------------------------------------------------------------

    /// Returns an iterator over all elements (by reference).
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            vec: self,
            idx: 0,
        }
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// The index at which the tail starts.
    fn tail_offset(&self) -> usize {
        if self.len < BRANCHING {
            0
        } else {
            ((self.len - 1) >> BITS) << BITS
        }
    }

    /// Walk the trie and return the leaf node containing index `idx`.
    fn find_leaf(&self, idx: usize) -> Arc<Node<T>> {
        let mut node = Arc::clone(&self.root);
        let mut level = self.shift;
        loop {
            let child_idx = (idx >> level) & MASK;
            match *node.clone() {
                Node::Internal(ref children) => {
                    if level == BITS {
                        // Next level is leaves.
                        node = Arc::clone(&children[child_idx]);
                        break;
                    }
                    node = Arc::clone(&children[child_idx]);
                    level -= BITS;
                }
                Node::Leaf(_) => break,
            }
        }
        node
    }

    /// Push a full tail leaf into the trie, returning the new root and shift.
    fn push_tail(
        &self,
        node: Arc<Node<T>>,
        tail: Arc<Node<T>>,
        shift: usize,
    ) -> (Arc<Node<T>>, usize) {
        // Check if trie has overflow (root needs expansion).
        let trie_len = self.tail_offset();
        if trie_len == 0 {
            // Trie was empty.
            let new_root: Arc<Node<T>> = Arc::new(Node::Internal(Arc::from(
                vec![tail].as_slice(),
            )));
            return (new_root, BITS);
        }

        // Does current root have capacity?
        let capacity = 1 << (shift + BITS);
        if trie_len < capacity {
            let new_root = self.insert_into_node(node, tail, trie_len, shift);
            return (new_root, shift);
        }

        // Need to grow the root by one level.
        let new_shift = shift + BITS;
        let sub = self.new_path(tail, shift);
        let new_root: Arc<Node<T>> = Arc::new(Node::Internal(Arc::from(
            vec![node, sub].as_slice(),
        )));
        (new_root, new_shift)
    }

    /// Insert `leaf` into the trie at position `idx`, doing path-copying.
    fn insert_into_node(
        &self,
        node: Arc<Node<T>>,
        leaf: Arc<Node<T>>,
        idx: usize,
        shift: usize,
    ) -> Arc<Node<T>> {
        let child_idx = (idx >> shift) & MASK;
        match *node {
            Node::Internal(ref children) => {
                let mut new_children: Vec<Arc<Node<T>>> = children.to_vec();
                if shift == BITS {
                    // Next slot becomes the leaf.
                    if child_idx < new_children.len() {
                        new_children[child_idx] = leaf;
                    } else {
                        new_children.push(leaf);
                    }
                } else if child_idx < new_children.len() {
                    new_children[child_idx] = self.insert_into_node(
                        Arc::clone(&new_children[child_idx]),
                        leaf,
                        idx,
                        shift - BITS,
                    );
                } else {
                    new_children.push(self.new_path(leaf, shift - BITS));
                }
                Arc::new(Node::Internal(Arc::from(new_children.as_slice())))
            }
            Node::Leaf(_) => Arc::new(Node::Internal(Arc::from(vec![leaf].as_slice()))),
        }
    }

    /// Create a left-spine path of internal nodes down to `leaf`.
    fn new_path(&self, leaf: Arc<Node<T>>, shift: usize) -> Arc<Node<T>> {
        if shift == 0 {
            return leaf;
        }
        let child = self.new_path(leaf, shift - BITS);
        Arc::new(Node::Internal(Arc::from(vec![child].as_slice())))
    }

    /// Pop the tail stored at `idx` from the trie (path-copy).
    fn pop_tail(
        &self,
        node: Arc<Node<T>>,
        idx: usize,
        shift: usize,
    ) -> (Arc<Node<T>>, usize) {
        let child_idx = (idx >> shift) & MASK;
        match *node {
            Node::Internal(ref children) => {
                let mut new_children: Vec<Arc<Node<T>>> = children.to_vec();
                if shift == BITS {
                    new_children.truncate(child_idx);
                } else {
                    let (new_child, _) = self.pop_tail(
                        Arc::clone(&new_children[child_idx]),
                        idx,
                        shift - BITS,
                    );
                    new_children[child_idx] = new_child;
                }
                let new_root = Arc::new(Node::Internal(Arc::from(new_children.as_slice())));
                // Collapse single-child roots.
                let new_shift = if shift > BITS
                    && matches!(*new_root, Node::Internal(ref ch) if ch.len() == 1)
                {
                    shift - BITS
                } else {
                    shift
                };
                (new_root, new_shift)
            }
            Node::Leaf(_) => (node, shift),
        }
    }

    /// Path-copy an update to a single element inside the trie.
    fn update_trie(&self, node: Arc<Node<T>>, idx: usize, val: T, shift: usize) -> Arc<Node<T>> {
        match *node {
            Node::Internal(ref children) => {
                let child_idx = (idx >> shift) & MASK;
                let mut new_children: Vec<Arc<Node<T>>> = children.to_vec();
                if shift == BITS {
                    // Child is a leaf.
                    let leaf = Arc::clone(&new_children[child_idx]);
                    let new_leaf = self.update_leaf(leaf, idx & MASK, val);
                    new_children[child_idx] = new_leaf;
                } else {
                    new_children[child_idx] = self.update_trie(
                        Arc::clone(&new_children[child_idx]),
                        idx,
                        val,
                        shift - BITS,
                    );
                }
                Arc::new(Node::Internal(Arc::from(new_children.as_slice())))
            }
            Node::Leaf(ref elems) => {
                let leaf_idx = idx & MASK;
                let mut new_elems = elems.to_vec();
                if leaf_idx < new_elems.len() {
                    new_elems[leaf_idx] = val;
                }
                Arc::new(Node::Leaf(Arc::from(new_elems.as_slice())))
            }
        }
    }

    /// Replace one element in a leaf node, returning a new leaf.
    fn update_leaf(&self, node: Arc<Node<T>>, leaf_idx: usize, val: T) -> Arc<Node<T>> {
        match *node {
            Node::Leaf(ref elems) => {
                let mut new_elems = elems.to_vec();
                if leaf_idx < new_elems.len() {
                    new_elems[leaf_idx] = val;
                }
                Arc::new(Node::Leaf(Arc::from(new_elems.as_slice())))
            }
            _ => node,
        }
    }
}

impl<T: Clone> Default for PersistentVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Iterator
// ---------------------------------------------------------------------------

/// Iterator over the elements of a [`PersistentVec`].
pub struct Iter<'a, T: Clone> {
    vec: &'a PersistentVec<T>,
    idx: usize,
}

impl<'a, T: Clone> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.vec.get(self.idx)?;
        self.idx += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.len.saturating_sub(self.idx);
        (remaining, Some(remaining))
    }
}

impl<'a, T: Clone> ExactSizeIterator for Iter<'a, T> {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_vec() {
        let v: PersistentVec<i32> = PersistentVec::new();
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
        assert_eq!(v.get(0), None);
    }

    #[test]
    fn test_push_and_get() {
        let mut v = PersistentVec::new();
        for i in 0..100 {
            v = v.push(i);
        }
        assert_eq!(v.len(), 100);
        for i in 0..100 {
            assert_eq!(v.get(i), Some(&i));
        }
    }

    #[test]
    fn test_persistence() {
        let v0 = PersistentVec::new();
        let v1 = v0.push(1);
        let v2 = v1.push(2);
        let v3 = v2.set(0, 99);

        // Older versions are unchanged.
        assert_eq!(v1.len(), 1);
        assert_eq!(v1.get(0), Some(&1));
        assert_eq!(v2.get(0), Some(&1));
        assert_eq!(v3.get(0), Some(&99));
        assert_eq!(v3.get(1), Some(&2));
    }

    #[test]
    fn test_pop() {
        let v = (0..10).fold(PersistentVec::new(), |acc, x| acc.push(x));
        let (v2, last) = v.pop();
        assert_eq!(last, Some(9));
        assert_eq!(v2.len(), 9);
        // Original unchanged.
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn test_large_push_cross_trie_boundary() {
        // Push BRANCHING+1 elements to force a trie node.
        let mut v = PersistentVec::new();
        for i in 0..(BRANCHING + 5) {
            v = v.push(i as i64);
        }
        for i in 0..(BRANCHING + 5) {
            assert_eq!(v.get(i), Some(&(i as i64)), "failed at idx={i}");
        }
    }

    #[test]
    fn test_iter() {
        let v = (0..50_i32).fold(PersistentVec::new(), |a, x| a.push(x));
        let collected: Vec<i32> = v.iter().copied().collect();
        let expected: Vec<i32> = (0..50).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_pop_empty() {
        let v: PersistentVec<i32> = PersistentVec::new();
        let (v2, val) = v.pop();
        assert_eq!(val, None);
        assert_eq!(v2.len(), 0);
    }
}
