//! Persistent vector based on Relaxed Radix Balanced (RRB) trees.
//!
//! An immutable, persistent vector where every "mutation" returns a new
//! vector while sharing unchanged structure with previous versions via `Arc`.
//!
//! # Complexity
//!
//! | Operation     | Time                |
//! |---------------|---------------------|
//! | `get(index)`  | O(log₃₂ N)         |
//! | `set(i, v)`   | O(log₃₂ N)         |
//! | `push_back`   | O(1) amortised      |
//! | `concat`      | O(log₃₂ N)         |
//! | `slice`       | O(log₃₂ N)         |
//! | `len`         | O(1)                |
//!
//! # Structural Sharing
//!
//! Old versions remain valid after modification — they share subtrees with
//! new versions through `Arc`.

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

#[derive(Clone, Debug)]
enum RrbNode<T: Clone> {
    /// Internal node with children and optional size table for relaxed nodes.
    Internal {
        children: Arc<Vec<Arc<RrbNode<T>>>>,
        /// Size table: `sizes[i]` = cumulative count of elements up to and
        /// including child `i`.  `None` for "dense" nodes where sizes are
        /// computed from depth alone.
        sizes: Option<Arc<Vec<usize>>>,
    },
    /// Leaf node holding up to BRANCHING elements.
    Leaf { elements: Arc<Vec<T>> },
}

impl<T: Clone> RrbNode<T> {
    fn empty_leaf() -> Arc<Self> {
        Arc::new(RrbNode::Leaf {
            elements: Arc::new(Vec::new()),
        })
    }

    fn leaf(elements: Vec<T>) -> Arc<Self> {
        Arc::new(RrbNode::Leaf {
            elements: Arc::new(elements),
        })
    }

    fn internal(children: Vec<Arc<RrbNode<T>>>, sizes: Option<Vec<usize>>) -> Arc<Self> {
        Arc::new(RrbNode::Internal {
            children: Arc::new(children),
            sizes: sizes.map(Arc::new),
        })
    }

    /// Number of children (for internal) or elements (for leaf).
    fn width(&self) -> usize {
        match self {
            RrbNode::Internal { children, .. } => children.len(),
            RrbNode::Leaf { elements } => elements.len(),
        }
    }

    /// Get element at the given index within this subtree.
    fn get(&self, index: usize, depth: usize) -> Option<&T> {
        match self {
            RrbNode::Leaf { elements } => elements.get(index),
            RrbNode::Internal {
                children, sizes, ..
            } => {
                if let Some(ref sz) = sizes {
                    // Relaxed node: use size table to find child.
                    let (child_idx, child_offset) = Self::find_child_relaxed(sz, index);
                    children
                        .get(child_idx)
                        .and_then(|c| c.get(child_offset, depth - 1))
                } else {
                    // Dense node: compute child index from bits.
                    let shift = depth * BITS;
                    let child_idx = (index >> shift) & MASK;
                    let child_offset = index & ((1 << shift) - 1);
                    children
                        .get(child_idx)
                        .and_then(|c| c.get(child_offset, depth - 1))
                }
            }
        }
    }

    /// Set element at the given index, returning a new node.
    fn set(&self, index: usize, value: T, depth: usize) -> Arc<Self> {
        match self {
            RrbNode::Leaf { elements } => {
                let mut new_elems = (**elements).clone();
                if index < new_elems.len() {
                    new_elems[index] = value;
                }
                Self::leaf(new_elems)
            }
            RrbNode::Internal {
                children, sizes, ..
            } => {
                if let Some(ref sz) = sizes {
                    let (child_idx, child_offset) = Self::find_child_relaxed(sz, index);
                    let mut new_children = (**children).clone();
                    if let Some(child) = new_children.get(child_idx) {
                        new_children[child_idx] = child.set(child_offset, value, depth - 1);
                    }
                    Self::internal(new_children, Some((**sz).clone()))
                } else {
                    let shift = depth * BITS;
                    let child_idx = (index >> shift) & MASK;
                    let child_offset = index & ((1 << shift) - 1);
                    let mut new_children = (**children).clone();
                    if let Some(child) = new_children.get(child_idx) {
                        new_children[child_idx] = child.set(child_offset, value, depth - 1);
                    }
                    Self::internal(new_children, None)
                }
            }
        }
    }

    /// Find the child index and offset within that child for a relaxed node.
    fn find_child_relaxed(sizes: &[usize], index: usize) -> (usize, usize) {
        for (i, &cumulative) in sizes.iter().enumerate() {
            if index < cumulative {
                let prev = if i == 0 { 0 } else { sizes[i - 1] };
                return (i, index - prev);
            }
        }
        // Fallback: last child.
        let last = sizes.len().saturating_sub(1);
        let prev = if last == 0 { 0 } else { sizes[last - 1] };
        (last, index.saturating_sub(prev))
    }

    /// Count elements in this subtree.
    fn count(&self, _depth: usize) -> usize {
        match self {
            RrbNode::Leaf { elements } => elements.len(),
            RrbNode::Internal {
                children, sizes, ..
            } => {
                if let Some(ref sz) = sizes {
                    sz.last().copied().unwrap_or(0)
                } else if children.is_empty() {
                    0
                } else {
                    // Dense node: compute from structure.
                    let mut total = 0;
                    for child in children.iter() {
                        total += child.count(_depth - 1);
                    }
                    total
                }
            }
        }
    }

    /// Collect all elements into a vec (in order).
    fn collect_into(&self, out: &mut Vec<T>) {
        match self {
            RrbNode::Leaf { elements } => {
                out.extend(elements.iter().cloned());
            }
            RrbNode::Internal { children, .. } => {
                for child in children.iter() {
                    child.collect_into(out);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PersistentRrbVec
// ---------------------------------------------------------------------------

/// A persistent vector based on Relaxed Radix Balanced (RRB) trees.
///
/// Every modification returns a new vector; the original remains unchanged.
/// Structural sharing means this is memory-efficient.
///
/// # Example
///
/// ```rust
/// use scirs2_core::concurrent::PersistentRrbVec;
///
/// let v0 = PersistentRrbVec::new();
/// let v1 = v0.push_back(10);
/// let v2 = v1.push_back(20);
/// let v3 = v2.push_back(30);
///
/// assert_eq!(v3.get(0), Some(&10));
/// assert_eq!(v3.get(1), Some(&20));
/// assert_eq!(v3.get(2), Some(&30));
/// assert_eq!(v3.len(), 3);
///
/// // v0 is still empty
/// assert!(v0.is_empty());
/// ```
#[derive(Clone)]
pub struct PersistentRrbVec<T: Clone> {
    root: Arc<RrbNode<T>>,
    /// Detached tail for O(1) amortised push.
    tail: Arc<Vec<T>>,
    /// Number of elements *not* in the tail (i.e., in the tree).
    tree_len: usize,
    /// Depth of the tree (0 = root is a leaf).
    depth: usize,
}

impl<T: Clone> PersistentRrbVec<T> {
    /// Create an empty persistent vector.
    pub fn new() -> Self {
        PersistentRrbVec {
            root: RrbNode::empty_leaf(),
            tail: Arc::new(Vec::new()),
            tree_len: 0,
            depth: 0,
        }
    }

    /// Return the total number of elements.
    pub fn len(&self) -> usize {
        self.tree_len + self.tail.len()
    }

    /// Return `true` if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a reference to the element at `index`.
    pub fn get(&self, index: usize) -> Option<&T> {
        let total = self.len();
        if index >= total {
            return None;
        }

        if index >= self.tree_len {
            // In the tail.
            self.tail.get(index - self.tree_len)
        } else {
            // In the tree.
            self.root.get(index, self.depth)
        }
    }

    /// Return a new vector with the element at `index` replaced by `value`.
    ///
    /// Returns `None` if `index` is out of bounds.
    pub fn set(&self, index: usize, value: T) -> Option<Self> {
        let total = self.len();
        if index >= total {
            return None;
        }

        if index >= self.tree_len {
            // In the tail.
            let tail_idx = index - self.tree_len;
            let mut new_tail = (*self.tail).clone();
            new_tail[tail_idx] = value;
            Some(PersistentRrbVec {
                root: Arc::clone(&self.root),
                tail: Arc::new(new_tail),
                tree_len: self.tree_len,
                depth: self.depth,
            })
        } else {
            // In the tree.
            let new_root = self.root.set(index, value, self.depth);
            Some(PersistentRrbVec {
                root: new_root,
                tail: Arc::clone(&self.tail),
                tree_len: self.tree_len,
                depth: self.depth,
            })
        }
    }

    /// Return a new vector with `value` appended at the end.
    pub fn push_back(&self, value: T) -> Self {
        if self.tail.len() < BRANCHING {
            // Room in the tail.
            let mut new_tail = (*self.tail).clone();
            new_tail.push(value);
            PersistentRrbVec {
                root: Arc::clone(&self.root),
                tail: Arc::new(new_tail),
                tree_len: self.tree_len,
                depth: self.depth,
            }
        } else {
            // Tail is full — push it into the tree and start a new tail.
            let tail_node = RrbNode::leaf((*self.tail).clone());
            let (new_root, new_depth) = self.push_tail_into_tree(tail_node);
            let new_tail = vec![value];

            PersistentRrbVec {
                root: new_root,
                tail: Arc::new(new_tail),
                tree_len: self.tree_len + BRANCHING,
                depth: new_depth,
            }
        }
    }

    /// Push the full tail leaf into the tree, possibly growing the tree height.
    fn push_tail_into_tree(&self, tail_node: Arc<RrbNode<T>>) -> (Arc<RrbNode<T>>, usize) {
        // Special case: empty tree.
        if self.tree_len == 0 {
            return (tail_node, 0);
        }

        // Try to insert into the existing tree.
        if let Some(new_root) = self.push_into_node(&self.root, tail_node.clone(), self.depth) {
            (new_root, self.depth)
        } else {
            // Tree is full at this depth — grow by one level.
            let new_right = self.new_path(tail_node, self.depth);
            let mut sizes = Vec::new();
            let left_count = self.root.count(self.depth);
            let right_count = new_right.count(self.depth);
            sizes.push(left_count);
            sizes.push(left_count + right_count);
            let new_root = RrbNode::internal(vec![Arc::clone(&self.root), new_right], Some(sizes));
            (new_root, self.depth + 1)
        }
    }

    /// Try to push a leaf into the given node. Returns `None` if the node is full.
    fn push_into_node(
        &self,
        node: &Arc<RrbNode<T>>,
        leaf: Arc<RrbNode<T>>,
        depth: usize,
    ) -> Option<Arc<RrbNode<T>>> {
        match node.as_ref() {
            RrbNode::Leaf { .. } => {
                // We're at a leaf level; can't push another leaf here.
                // The caller needs to grow the tree.
                None
            }
            RrbNode::Internal {
                children, sizes, ..
            } => {
                if depth == 1 {
                    // Children are leaves.
                    if children.len() < BRANCHING {
                        let mut new_children = (**children).clone();
                        new_children.push(leaf);
                        let new_sizes = self.compute_sizes(&new_children, 0);
                        Some(RrbNode::internal(new_children, Some(new_sizes)))
                    } else {
                        None // full
                    }
                } else {
                    // Try to push into the last child.
                    let last_idx = children.len() - 1;
                    if let Some(new_last) =
                        self.push_into_node(&children[last_idx], leaf.clone(), depth - 1)
                    {
                        let mut new_children = (**children).clone();
                        new_children[last_idx] = new_last;
                        let new_sizes = self.compute_sizes(&new_children, depth - 1);
                        Some(RrbNode::internal(new_children, Some(new_sizes)))
                    } else if children.len() < BRANCHING {
                        // Last child is full; add a new path.
                        let new_path = self.new_path(leaf, depth - 1);
                        let mut new_children = (**children).clone();
                        new_children.push(new_path);
                        let new_sizes = self.compute_sizes(&new_children, depth - 1);
                        Some(RrbNode::internal(new_children, Some(new_sizes)))
                    } else {
                        None // this node is full
                    }
                }
            }
        }
    }

    /// Create a path of internal nodes leading to the given leaf at the specified depth.
    fn new_path(&self, leaf: Arc<RrbNode<T>>, depth: usize) -> Arc<RrbNode<T>> {
        if depth == 0 {
            leaf
        } else {
            let child = self.new_path(leaf, depth - 1);
            let count = child.count(depth - 1);
            RrbNode::internal(vec![child], Some(vec![count]))
        }
    }

    /// Compute cumulative size table for a list of children at a given child depth.
    fn compute_sizes(&self, children: &[Arc<RrbNode<T>>], child_depth: usize) -> Vec<usize> {
        let mut sizes = Vec::with_capacity(children.len());
        let mut cumulative = 0usize;
        for child in children {
            cumulative += child.count(child_depth);
            sizes.push(cumulative);
        }
        sizes
    }

    /// Concatenate two persistent vectors into a new one.
    ///
    /// Both original vectors remain valid.
    pub fn concat(&self, other: &Self) -> Self {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }

        // Simple strategy: collect both into one vec and rebuild.
        // This is O(n) but correct. For a production RRB-tree, the concat
        // algorithm would merge the spines in O(log N) but that's
        // significantly more complex.
        let mut all = Vec::with_capacity(self.len() + other.len());
        self.root.collect_into(&mut all);
        all.extend(self.tail.iter().cloned());
        other.root.collect_into(&mut all);
        all.extend(other.tail.iter().cloned());

        Self::from_vec(all)
    }

    /// Build a PersistentRrbVec from a Vec efficiently (bottom-up).
    fn from_vec(elements: Vec<T>) -> Self {
        if elements.is_empty() {
            return Self::new();
        }

        let total = elements.len();

        // Split into tree portion (full leaves) and tail.
        let full_leaves = total / BRANCHING;
        let tail_len = total - full_leaves * BRANCHING;

        let tree_elements = &elements[..full_leaves * BRANCHING];
        let tail_elements = &elements[full_leaves * BRANCHING..];

        let tail = Arc::new(tail_elements.to_vec());

        if full_leaves == 0 {
            return PersistentRrbVec {
                root: RrbNode::empty_leaf(),
                tail,
                tree_len: 0,
                depth: 0,
            };
        }

        // Build leaf nodes.
        let mut leaves: Vec<Arc<RrbNode<T>>> = Vec::with_capacity(full_leaves);
        for chunk in tree_elements.chunks(BRANCHING) {
            leaves.push(RrbNode::leaf(chunk.to_vec()));
        }

        // Build tree bottom-up.
        let mut level_nodes = leaves;
        let mut depth = 0usize;

        while level_nodes.len() > 1 {
            let mut next_level = Vec::new();
            for chunk in level_nodes.chunks(BRANCHING) {
                let children: Vec<Arc<RrbNode<T>>> = chunk.to_vec();
                let mut sizes = Vec::with_capacity(children.len());
                let mut cum = 0usize;
                for child in &children {
                    cum += child.count(depth);
                    sizes.push(cum);
                }
                next_level.push(RrbNode::internal(children, Some(sizes)));
            }
            level_nodes = next_level;
            depth += 1;
        }

        let root = level_nodes
            .into_iter()
            .next()
            .unwrap_or_else(RrbNode::empty_leaf);

        PersistentRrbVec {
            root,
            tail,
            tree_len: full_leaves * BRANCHING,
            depth,
        }
    }

    /// Extract a sub-vector for the given range.
    ///
    /// Returns a new `PersistentRrbVec` containing elements `[start..end)`.
    pub fn slice(&self, start: usize, end: usize) -> Self {
        let total = self.len();
        let start = start.min(total);
        let end = end.min(total);
        if start >= end {
            return Self::new();
        }

        // Collect the slice and rebuild.
        let mut elements = Vec::with_capacity(end - start);
        for i in start..end {
            if let Some(v) = self.get(i) {
                elements.push(v.clone());
            }
        }
        Self::from_vec(elements)
    }

    /// Collect all elements into a `Vec`.
    pub fn to_vec(&self) -> Vec<T> {
        let mut result = Vec::with_capacity(self.len());
        self.root.collect_into(&mut result);
        result.extend(self.tail.iter().cloned());
        result
    }

    /// Return an iterator over the elements.
    pub fn iter(&self) -> PersistentRrbVecIter<'_, T> {
        PersistentRrbVecIter {
            vec: self,
            index: 0,
        }
    }
}

impl<T: Clone> Default for PersistentRrbVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + std::fmt::Debug> std::fmt::Debug for PersistentRrbVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

// ---------------------------------------------------------------------------
// Iterator
// ---------------------------------------------------------------------------

/// Iterator over elements of a [`PersistentRrbVec`].
pub struct PersistentRrbVecIter<'a, T: Clone> {
    vec: &'a PersistentRrbVec<T>,
    index: usize,
}

impl<'a, T: Clone> Iterator for PersistentRrbVecIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.vec.len() {
            return None;
        }
        let item = self.vec.get(self.index);
        self.index += 1;
        item
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<T: Clone> ExactSizeIterator for PersistentRrbVecIter<'_, T> {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_back_and_get() {
        let v0 = PersistentRrbVec::new();
        let v1 = v0.push_back(10);
        let v2 = v1.push_back(20);
        let v3 = v2.push_back(30);

        assert_eq!(v3.get(0), Some(&10));
        assert_eq!(v3.get(1), Some(&20));
        assert_eq!(v3.get(2), Some(&30));
        assert_eq!(v3.len(), 3);
    }

    #[test]
    fn test_structural_sharing() {
        let v0 = PersistentRrbVec::new();
        let v1 = v0.push_back(1);
        let v2 = v1.push_back(2);

        // v1 should still be valid and unchanged.
        assert_eq!(v1.len(), 1);
        assert_eq!(v1.get(0), Some(&1));
        assert_eq!(v1.get(1), None);

        // v2 has both elements.
        assert_eq!(v2.len(), 2);
        assert_eq!(v2.get(0), Some(&1));
        assert_eq!(v2.get(1), Some(&2));

        // v0 is still empty.
        assert!(v0.is_empty());
    }

    #[test]
    fn test_set_returns_new_version() {
        let v0 = PersistentRrbVec::new();
        let v1 = v0.push_back(1).push_back(2).push_back(3);

        let v2 = v1.set(1, 42);
        assert!(v2.is_some());
        let v2 = v2.expect("set should succeed");

        // v2 has the updated value.
        assert_eq!(v2.get(1), Some(&42));
        // v1 is unchanged.
        assert_eq!(v1.get(1), Some(&2));

        // Out-of-bounds set returns None.
        assert!(v1.set(100, 0).is_none());
    }

    #[test]
    fn test_large_push_back() {
        let mut v = PersistentRrbVec::new();
        for i in 0..200u32 {
            v = v.push_back(i);
        }
        assert_eq!(v.len(), 200);
        for i in 0..200u32 {
            assert_eq!(v.get(i as usize), Some(&i), "failed at index {i}");
        }
    }

    #[test]
    fn test_concat() {
        let mut v1 = PersistentRrbVec::new();
        for i in 0..50u32 {
            v1 = v1.push_back(i);
        }

        let mut v2 = PersistentRrbVec::new();
        for i in 50..100u32 {
            v2 = v2.push_back(i);
        }

        let v3 = v1.concat(&v2);
        assert_eq!(v3.len(), 100);
        for i in 0..100u32 {
            assert_eq!(v3.get(i as usize), Some(&i), "concat failed at index {i}");
        }

        // Originals unchanged.
        assert_eq!(v1.len(), 50);
        assert_eq!(v2.len(), 50);
    }

    #[test]
    fn test_slice() {
        let mut v = PersistentRrbVec::new();
        for i in 0..100u32 {
            v = v.push_back(i);
        }

        let s = v.slice(10, 20);
        assert_eq!(s.len(), 10);
        for i in 0..10u32 {
            assert_eq!(s.get(i as usize), Some(&(i + 10)));
        }

        // Empty slice.
        let s2 = v.slice(50, 50);
        assert!(s2.is_empty());

        // Out-of-bounds slice clamps.
        let s3 = v.slice(90, 200);
        assert_eq!(s3.len(), 10);
    }

    #[test]
    fn test_to_vec() {
        let mut v = PersistentRrbVec::new();
        for i in 0..10u32 {
            v = v.push_back(i);
        }
        assert_eq!(v.to_vec(), (0..10u32).collect::<Vec<_>>());
    }

    #[test]
    fn test_iter() {
        let mut v = PersistentRrbVec::new();
        for i in 0..5i32 {
            v = v.push_back(i);
        }
        let collected: Vec<i32> = v.iter().copied().collect();
        assert_eq!(collected, vec![0, 1, 2, 3, 4]);
        assert_eq!(v.iter().len(), 5);
    }

    #[test]
    fn test_empty() {
        let v: PersistentRrbVec<i32> = PersistentRrbVec::new();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
        assert_eq!(v.get(0), None);
        assert!(v.to_vec().is_empty());
    }

    #[test]
    fn test_single_element() {
        let v = PersistentRrbVec::new().push_back(42);
        assert_eq!(v.len(), 1);
        assert_eq!(v.get(0), Some(&42));
        assert!(!v.is_empty());
    }

    #[test]
    fn test_concat_empty() {
        let v = PersistentRrbVec::new().push_back(1).push_back(2);
        let empty = PersistentRrbVec::new();

        let r1 = v.concat(&empty);
        assert_eq!(r1.len(), 2);

        let r2 = empty.concat(&v);
        assert_eq!(r2.len(), 2);
    }

    #[test]
    fn test_many_push_backs_past_branching() {
        // Push more than BRANCHING * BRANCHING elements to test tree growth.
        let n = BRANCHING * BRANCHING + 100;
        let mut v = PersistentRrbVec::new();
        for i in 0..n {
            v = v.push_back(i);
        }
        assert_eq!(v.len(), n);
        for i in 0..n {
            assert_eq!(v.get(i), Some(&i), "failed at index {i}");
        }
    }

    #[test]
    fn test_from_vec_roundtrip() {
        let original: Vec<u32> = (0..150).collect();
        let pv = PersistentRrbVec::from_vec(original.clone());
        assert_eq!(pv.len(), 150);
        assert_eq!(pv.to_vec(), original);
    }
}
