//! Persistent radix-balanced vector (RRB-tree).
//!
//! Implements an immutable persistent vector with:
//! - O(log N) element access
//! - O(log N) amortised `push_back`
//! - O(log N) structural-sharing `update` (returns a new version while the
//!   original remains intact)
//!
//! The branching factor is `BRANCHING = 32` (5-bit trie).
//!
//! # Structural Sharing
//!
//! All nodes are reference-counted via `std::rc::Rc`.  `update` copies only
//! the O(log N) nodes on the path to the changed element; all other nodes are
//! shared with the original tree.

use std::rc::Rc;

/// Branching factor of the trie (must be a power of two).
const BRANCHING: usize = 32;

/// log2(BRANCHING) — number of bits consumed per trie level.
const BITS: usize = 5;

/// Bit-mask for selecting a `BITS`-wide index at one trie level.
const MASK: usize = BRANCHING - 1;

// ─────────────────────────────────────────────────────────────────────────────
// Internal node
// ─────────────────────────────────────────────────────────────────────────────

/// A node in the radix-balanced trie.
#[derive(Clone)]
enum RrbNode {
    /// Leaf node holding up to `BRANCHING` elements.
    Leaf(Vec<u64>),
    /// Internal node with up to `BRANCHING` child subtrees.
    Internal(Vec<Rc<RrbNode>>),
}

impl RrbNode {
    /// Retrieve the element at `index` within a subtree of the given `height`.
    ///
    /// `height == 0` means this node is a [`RrbNode::Leaf`].
    fn get(&self, index: usize, height: usize) -> Option<u64> {
        match self {
            RrbNode::Leaf(data) => data.get(index).copied(),
            RrbNode::Internal(children) => {
                let shift = height * BITS;
                let child_idx = (index >> shift) & MASK;
                let remainder = index & ((1 << shift) - 1);
                children
                    .get(child_idx)
                    .and_then(|c| c.get(remainder, height - 1))
            }
        }
    }

    /// Return a new node (sharing structure) with `index` updated to `value`.
    fn update(&self, index: usize, value: u64, height: usize) -> Option<Rc<RrbNode>> {
        match self {
            RrbNode::Leaf(data) => {
                if index >= data.len() {
                    return None;
                }
                let mut new_data = data.clone();
                new_data[index] = value;
                Some(Rc::new(RrbNode::Leaf(new_data)))
            }
            RrbNode::Internal(children) => {
                let shift = height * BITS;
                let child_idx = (index >> shift) & MASK;
                let remainder = index & ((1 << shift) - 1);
                let new_child = children
                    .get(child_idx)
                    .and_then(|c| c.update(remainder, value, height - 1))?;
                let mut new_children = children.clone();
                new_children[child_idx] = new_child;
                Some(Rc::new(RrbNode::Internal(new_children)))
            }
        }
    }

    /// Collect all elements into `out` in order.
    fn collect(&self, out: &mut Vec<u64>) {
        match self {
            RrbNode::Leaf(data) => out.extend_from_slice(data),
            RrbNode::Internal(children) => {
                for c in children {
                    c.collect(out);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RrbVec
// ─────────────────────────────────────────────────────────────────────────────

/// Persistent immutable vector backed by a 32-way radix-balanced trie.
///
/// All modifying operations return a *new* `RrbVec`; the original is
/// unaffected.  Internal subtrees are reference-counted and shared between
/// versions wherever possible.
#[derive(Clone)]
pub struct RrbVec {
    root: Option<Rc<RrbNode>>,
    /// Total number of elements stored.
    length: usize,
    /// Height of the trie (0 = root is a leaf, 1 = one level of internal
    /// nodes above leaves, …).
    height: usize,
}

impl Default for RrbVec {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for RrbVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RrbVec")
            .field("length", &self.length)
            .field("height", &self.height)
            .finish()
    }
}

impl RrbVec {
    // ── constructors ──────────────────────────────────────────────────────

    /// Create an empty vector.
    pub fn new() -> Self {
        RrbVec {
            root: None,
            length: 0,
            height: 0,
        }
    }

    /// Build a vector from a slice of `u64` values.
    pub fn from_slice(data: &[u64]) -> Self {
        let mut v = RrbVec::new();
        for &x in data {
            v = v.push_back(x);
        }
        v
    }

    // ── queries ───────────────────────────────────────────────────────────

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.length
    }

    /// True if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// O(log N) element access.
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.length {
            return None;
        }
        self.root.as_ref().and_then(|r| r.get(index, self.height))
    }

    // ── persistent mutations ──────────────────────────────────────────────

    /// Return a new vector with `value` appended to the end.
    ///
    /// Amortised O(log N).
    pub fn push_back(&self, value: u64) -> Self {
        let new_root = match &self.root {
            None => {
                // First element: create a single-element leaf.
                Rc::new(RrbNode::Leaf(vec![value]))
            }
            Some(root) => {
                // Try to insert into the existing tree.
                match push_node(root, self.length, self.height, value) {
                    PushResult::Inserted(new_root) => new_root,
                    PushResult::NeedsNewRoot(sibling) => {
                        // The tree is full at the current height — grow by one level.
                        Rc::new(RrbNode::Internal(vec![Rc::clone(root), sibling]))
                    }
                }
            }
        };
        RrbVec {
            root: Some(new_root),
            length: self.length + 1,
            height: if self.root.is_none() {
                0
            } else {
                // Height may have grown inside push_node logic
                new_height_after_push(self.length + 1, self.height)
            },
        }
    }

    /// Return a new vector with the element at `index` replaced by `value`.
    ///
    /// Returns `None` if `index` is out of bounds.  Runs in O(log N) and
    /// shares structure with the original.
    pub fn update(&self, index: usize, value: u64) -> Option<Self> {
        if index >= self.length {
            return None;
        }
        let new_root = self
            .root
            .as_ref()
            .and_then(|r| r.update(index, value, self.height))?;
        Some(RrbVec {
            root: Some(new_root),
            length: self.length,
            height: self.height,
        })
    }

    // ── iteration / conversion ────────────────────────────────────────────

    /// Iterate over elements in order.
    pub fn iter(&self) -> RrbVecIter<'_> {
        RrbVecIter { vec: self, pos: 0 }
    }

    /// Collect all elements into a `Vec<u64>`.
    pub fn to_vec(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.length);
        if let Some(root) = &self.root {
            root.collect(&mut out);
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Iterator
// ─────────────────────────────────────────────────────────────────────────────

/// Iterator over an [`RrbVec`].
pub struct RrbVecIter<'a> {
    vec: &'a RrbVec,
    pos: usize,
}

impl<'a> Iterator for RrbVecIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.vec.length {
            return None;
        }
        let val = self.vec.get(self.pos);
        self.pos += 1;
        val
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.length - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for RrbVecIter<'a> {}

// ─────────────────────────────────────────────────────────────────────────────
// Internal push helpers
// ─────────────────────────────────────────────────────────────────────────────

enum PushResult {
    /// The value was inserted; the returned node is the updated subtree root.
    Inserted(Rc<RrbNode>),
    /// The current subtree was full; a new sibling node carrying the value is
    /// returned and must be attached at the parent level.
    NeedsNewRoot(Rc<RrbNode>),
}

/// Capacity of a full subtree of the given height.
fn subtree_capacity(height: usize) -> usize {
    BRANCHING.pow((height + 1) as u32)
}

/// Recursively insert `value` into the subtree rooted at `node`.
///
/// `length` is the current number of elements in the subtree.
/// `height` is the height of `node` (0 = leaf).
fn push_node(node: &Rc<RrbNode>, length: usize, height: usize, value: u64) -> PushResult {
    if height == 0 {
        // Leaf node
        match node.as_ref() {
            RrbNode::Leaf(data) => {
                if data.len() < BRANCHING {
                    let mut new_data = data.clone();
                    new_data.push(value);
                    PushResult::Inserted(Rc::new(RrbNode::Leaf(new_data)))
                } else {
                    // Leaf is full — create a new sibling leaf
                    PushResult::NeedsNewRoot(Rc::new(RrbNode::Leaf(vec![value])))
                }
            }
            RrbNode::Internal(_) => unreachable!("height 0 must be a leaf"),
        }
    } else {
        // Internal node
        match node.as_ref() {
            RrbNode::Internal(children) => {
                let child_cap = subtree_capacity(height - 1);
                let last_child_idx = if children.is_empty() {
                    return PushResult::NeedsNewRoot(make_path(height, value));
                } else {
                    children.len() - 1
                };
                let last_child_len = length - last_child_idx * child_cap;

                match push_node(&children[last_child_idx], last_child_len, height - 1, value) {
                    PushResult::Inserted(new_child) => {
                        let mut new_children = children.clone();
                        new_children[last_child_idx] = new_child;
                        PushResult::Inserted(Rc::new(RrbNode::Internal(new_children)))
                    }
                    PushResult::NeedsNewRoot(sibling) => {
                        if children.len() < BRANCHING {
                            let mut new_children = children.clone();
                            new_children.push(sibling);
                            PushResult::Inserted(Rc::new(RrbNode::Internal(new_children)))
                        } else {
                            // This internal node is also full
                            PushResult::NeedsNewRoot(make_path(height, value))
                        }
                    }
                }
            }
            RrbNode::Leaf(_) => unreachable!("height > 0 must be internal"),
        }
    }
}

/// Create a leftmost path of internal nodes down to a single-element leaf.
fn make_path(height: usize, value: u64) -> Rc<RrbNode> {
    if height == 0 {
        Rc::new(RrbNode::Leaf(vec![value]))
    } else {
        Rc::new(RrbNode::Internal(vec![make_path(height - 1, value)]))
    }
}

/// Determine the height of a balanced trie holding `n` elements.
fn new_height_after_push(n: usize, old_height: usize) -> usize {
    // A trie of height h can hold at most BRANCHING^(h+1) elements.
    let mut h = old_height;
    while subtree_capacity(h) < n {
        h += 1;
    }
    // If the new element triggered a root split, the height increased.
    h
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_vec() {
        let v = RrbVec::new();
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
        assert_eq!(v.get(0), None);
    }

    #[test]
    fn test_push_back_sequential_0_to_99() {
        let mut v = RrbVec::new();
        for i in 0..100u64 {
            v = v.push_back(i);
        }
        assert_eq!(v.len(), 100);
        for i in 0..100u64 {
            assert_eq!(v.get(i as usize), Some(i), "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_get_after_push() {
        let v = RrbVec::new().push_back(42).push_back(99).push_back(7);
        assert_eq!(v.get(0), Some(42));
        assert_eq!(v.get(1), Some(99));
        assert_eq!(v.get(2), Some(7));
        assert_eq!(v.get(3), None);
    }

    #[test]
    fn test_update_does_not_modify_original() {
        let v1 = RrbVec::from_slice(&[10, 20, 30]);
        let v2 = v1.update(1, 999).expect("update failed");
        // v1 must be unchanged
        assert_eq!(v1.get(1), Some(20));
        // v2 reflects the change
        assert_eq!(v2.get(1), Some(999));
        assert_eq!(v2.get(0), Some(10));
        assert_eq!(v2.get(2), Some(30));
    }

    #[test]
    fn test_iter_matches_to_vec() {
        let data: Vec<u64> = (0..50).collect();
        let v = RrbVec::from_slice(&data);
        let from_iter: Vec<u64> = v.iter().collect();
        assert_eq!(from_iter, v.to_vec());
        assert_eq!(from_iter, data);
    }

    #[test]
    fn test_from_slice_and_iter() {
        let data = vec![100u64, 200, 300, 400, 500];
        let v = RrbVec::from_slice(&data);
        assert_eq!(v.len(), 5);
        let out: Vec<u64> = v.iter().collect();
        assert_eq!(out, data);
    }

    #[test]
    fn test_large_vector_1000_elements() {
        let mut v = RrbVec::new();
        for i in 0..1000u64 {
            v = v.push_back(i * 3 + 7);
        }
        assert_eq!(v.len(), 1000);
        for i in 0..1000u64 {
            let expected = i * 3 + 7;
            assert_eq!(v.get(i as usize), Some(expected), "mismatch at {}", i);
        }
    }

    #[test]
    fn test_structural_sharing_update_creates_new_vec_original_unchanged() {
        let original = RrbVec::from_slice(&[1, 2, 3, 4, 5]);
        let updated = original.update(2, 99).expect("update failed");
        // The updated version has the new value
        assert_eq!(updated.get(2), Some(99));
        // The original is entirely unaffected (structural sharing)
        assert_eq!(original.get(2), Some(3));
        // All other elements are shared/equal
        for i in [0, 1, 3, 4] {
            assert_eq!(original.get(i), updated.get(i), "element {} differs", i);
        }
    }
}
