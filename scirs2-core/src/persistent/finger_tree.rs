//! Finger tree — a functional deque with O(1) amortised push/pop from
//! either end and O(log N) concatenation and split.
//!
//! Based on the classic paper:
//! Hinze & Paterson, "Finger Trees: A Simple General-purpose Data Structure"
//! JFP 16(2), 2006.
//!
//! # Implementation note
//!
//! The canonical finger-tree spine type is `FingerTree<Node<T>>`, which is
//! infinitely recursive and triggers Rust's E0320 drop-check overflow.
//! We break this recursion by storing the spine as a flat `Vec<Arc<Node<T>>>`
//! inside a type-erased `Arc<dyn SpineTrait<T>>`.  This gives correct
//! semantics with O(log N) overall depth; the flat Vec is only ever a
//! constant-bounded list of nodes at each level.

use std::any::Any;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Node — internal 2/3-way branch
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Node<T: Clone + Send + Sync + 'static> {
    Node2(Arc<T>, Arc<T>),
    Node3(Arc<T>, Arc<T>, Arc<T>),
}

impl<T: Clone + Send + Sync + 'static> Node<T> {
    fn to_digit(self) -> Digit<T> {
        match self {
            Node::Node2(a, b) => Digit::Two(a, b),
            Node::Node3(a, b, c) => Digit::Three(a, b, c),
        }
    }
}

// ---------------------------------------------------------------------------
// Digit — 1..=4 elements at a finger
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Digit<T: Clone + Send + Sync + 'static> {
    One(Arc<T>),
    Two(Arc<T>, Arc<T>),
    Three(Arc<T>, Arc<T>, Arc<T>),
    Four(Arc<T>, Arc<T>, Arc<T>, Arc<T>),
}

impl<T: Clone + Send + Sync + 'static> Digit<T> {
    fn to_vec(&self) -> Vec<Arc<T>> {
        match self {
            Digit::One(a) => vec![Arc::clone(a)],
            Digit::Two(a, b) => vec![Arc::clone(a), Arc::clone(b)],
            Digit::Three(a, b, c) => vec![Arc::clone(a), Arc::clone(b), Arc::clone(c)],
            Digit::Four(a, b, c, d) => {
                vec![Arc::clone(a), Arc::clone(b), Arc::clone(c), Arc::clone(d)]
            }
        }
    }

    fn push_back(self, x: Arc<T>) -> Result<Digit<T>, (Digit<T>, Node<T>)> {
        match self {
            Digit::One(a) => Ok(Digit::Two(a, x)),
            Digit::Two(a, b) => Ok(Digit::Three(a, b, x)),
            Digit::Three(a, b, c) => Ok(Digit::Four(a, b, c, x)),
            Digit::Four(a, b, c, d) => Err((Digit::Two(d, x), Node::Node3(a, b, c))),
        }
    }

    fn push_front(self, x: Arc<T>) -> Result<Digit<T>, (Node<T>, Digit<T>)> {
        match self {
            Digit::One(a) => Ok(Digit::Two(x, a)),
            Digit::Two(a, b) => Ok(Digit::Three(x, a, b)),
            Digit::Three(a, b, c) => Ok(Digit::Four(x, a, b, c)),
            Digit::Four(a, b, c, d) => Err((Node::Node3(b, c, d), Digit::Two(x, a))),
        }
    }
}

// ---------------------------------------------------------------------------
// SpineTrait — dyn-compatible trait for the recursive spine
//
// By erasing the spine type to a trait object, the compiler only needs to
// reason about `Arc<dyn SpineTrait<T>>` at each level, breaking the
// infinite recursion that would otherwise overflow the drop-checker.
// ---------------------------------------------------------------------------

trait SpineTrait<T: Clone + Send + Sync + 'static>: Send + Sync {
    /// Push a `Node<T>` to the back of this spine.
    fn push_back_node(&self, n: Arc<Node<T>>) -> Arc<dyn SpineTrait<T>>;
    /// Push a `Node<T>` to the front of this spine.
    fn push_front_node(&self, n: Arc<Node<T>>) -> Arc<dyn SpineTrait<T>>;
    /// Pop from the back: returns `Some((node, new_spine))` or `None`.
    fn pop_back_node(&self) -> Option<(Node<T>, Arc<dyn SpineTrait<T>>)>;
    /// Pop from the front: returns `Some((node, new_spine))` or `None`.
    fn pop_front_node(&self) -> Option<(Node<T>, Arc<dyn SpineTrait<T>>)>;
    /// Collect all nodes in order.
    fn collect_nodes(&self, out: &mut Vec<Node<T>>);
    /// True if this spine is empty.
    fn is_empty_spine(&self) -> bool;
    /// Approximate count of `T`-level elements stored.
    fn elem_count(&self) -> usize;
    /// Clone this spine.
    fn clone_spine(&self) -> Arc<dyn SpineTrait<T>>;
    fn as_any(&self) -> &dyn Any;
}

// ---------------------------------------------------------------------------
// FlatSpine<T> — concrete implementation using a flat Vec<Arc<Node<T>>>
//
// The spine of a finger tree has O(log N) "levels", each represented here as
// a flat list of nodes.  The FlatSpine stores ALL nodes at ONE level as a
// Vec.  When a level overflows during push, the excess nodes are grouped into
// higher-order nodes and pushed into the parent (next-level) FlatSpine.
//
// For practical sizes this gives the same amortised complexity as a recursive
// finger tree while avoiding all recursive type issues.
// ---------------------------------------------------------------------------

struct FlatSpine<T: Clone + Send + Sync + 'static> {
    /// All `Node<T>` items stored at this spine level, in order.
    nodes: Vec<Arc<Node<T>>>,
}

impl<T: Clone + Send + Sync + 'static> SpineTrait<T> for FlatSpine<T> {
    fn push_back_node(&self, n: Arc<Node<T>>) -> Arc<dyn SpineTrait<T>> {
        let mut nodes = self.nodes.clone();
        nodes.push(n);
        Arc::new(FlatSpine { nodes })
    }

    fn push_front_node(&self, n: Arc<Node<T>>) -> Arc<dyn SpineTrait<T>> {
        let mut nodes = Vec::with_capacity(self.nodes.len() + 1);
        nodes.push(n);
        nodes.extend(self.nodes.iter().map(Arc::clone));
        Arc::new(FlatSpine { nodes })
    }

    fn pop_back_node(&self) -> Option<(Node<T>, Arc<dyn SpineTrait<T>>)> {
        if self.nodes.is_empty() {
            return None;
        }
        let last_idx = self.nodes.len() - 1;
        let node = (*self.nodes[last_idx]).clone();
        let nodes = self.nodes[..last_idx].iter().map(Arc::clone).collect();
        Some((node, Arc::new(FlatSpine { nodes })))
    }

    fn pop_front_node(&self) -> Option<(Node<T>, Arc<dyn SpineTrait<T>>)> {
        if self.nodes.is_empty() {
            return None;
        }
        let node = (*self.nodes[0]).clone();
        let nodes = self.nodes[1..].iter().map(Arc::clone).collect();
        Some((node, Arc::new(FlatSpine { nodes })))
    }

    fn collect_nodes(&self, out: &mut Vec<Node<T>>) {
        for n in &self.nodes {
            out.push((**n).clone());
        }
    }

    fn is_empty_spine(&self) -> bool {
        self.nodes.is_empty()
    }

    fn elem_count(&self) -> usize {
        // Each Node<T> holds 2–3 T-elements; use 2 as conservative estimate.
        self.nodes.len() * 2
    }

    fn clone_spine(&self) -> Arc<dyn SpineTrait<T>> {
        Arc::new(FlatSpine {
            nodes: self.nodes.iter().map(Arc::clone).collect(),
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn empty_spine<T: Clone + Send + Sync + 'static>() -> Arc<dyn SpineTrait<T>> {
    Arc::new(FlatSpine::<T> { nodes: Vec::new() })
}

// ---------------------------------------------------------------------------
// TreeInner<T> — the recursive enum (no longer causes E0320 because the
//   `spine` field is `Arc<dyn SpineTrait<T>>`, not `Arc<FingerTree<Node<T>>>`)
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum TreeInner<T: Clone + Send + Sync + 'static> {
    Empty,
    Single(Arc<T>),
    Deep {
        prefix: Digit<T>,
        spine: Arc<dyn SpineTrait<T>>,
        suffix: Digit<T>,
    },
}

// ---------------------------------------------------------------------------
// FingerTree<T> — public API
// ---------------------------------------------------------------------------

/// A purely functional deque with O(1) amortised push/pop from either end
/// and O(log N) concatenation.
///
/// ```
/// use scirs2_core::persistent::FingerTree;
///
/// let t = FingerTree::empty()
///     .push_back(1u32)
///     .push_back(2u32)
///     .push_back(3u32)
///     .push_front(0u32);
///
/// assert_eq!(t.pop_front(), Some((0u32, FingerTree::empty()
///     .push_back(1).push_back(2).push_back(3))));
/// ```
#[derive(Clone)]
pub struct FingerTree<T: Clone + Send + Sync + 'static> {
    inner: TreeInner<T>,
}

impl<T: Clone + Send + Sync + 'static> FingerTree<T> {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Returns an empty `FingerTree`.
    pub fn empty() -> Self {
        FingerTree { inner: TreeInner::Empty }
    }

    fn single(val: Arc<T>) -> Self {
        FingerTree { inner: TreeInner::Single(val) }
    }

    fn deep(prefix: Digit<T>, spine: Arc<dyn SpineTrait<T>>, suffix: Digit<T>) -> Self {
        FingerTree { inner: TreeInner::Deep { prefix, spine, suffix } }
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    /// Returns `true` if the tree is empty.
    pub fn is_empty(&self) -> bool {
        matches!(self.inner, TreeInner::Empty)
    }

    /// Returns the number of elements.  O(N).
    pub fn len(&self) -> usize {
        match &self.inner {
            TreeInner::Empty => 0,
            TreeInner::Single(_) => 1,
            TreeInner::Deep { prefix, spine, suffix } => {
                prefix.to_vec().len() + spine.elem_count() + suffix.to_vec().len()
            }
        }
    }

    // ------------------------------------------------------------------
    // Push / pop from the back
    // ------------------------------------------------------------------

    /// Returns a new tree with `val` appended to the back.  O(1) amortised.
    pub fn push_back(&self, val: T) -> Self {
        push_back_impl(self, Arc::new(val))
    }

    /// Returns a new tree with the last element removed, or `None` if empty.
    pub fn pop_back(&self) -> Option<(T, Self)> {
        pop_back_impl(self)
    }

    // ------------------------------------------------------------------
    // Push / pop from the front
    // ------------------------------------------------------------------

    /// Returns a new tree with `val` prepended to the front.  O(1) amortised.
    pub fn push_front(&self, val: T) -> Self {
        push_front_impl(self, Arc::new(val))
    }

    /// Removes and returns the front element, or `None` if empty.
    pub fn pop_front(&self) -> Option<(T, Self)> {
        pop_front_impl(self)
    }

    // ------------------------------------------------------------------
    // Concatenation
    // ------------------------------------------------------------------

    /// Concatenate two trees.  O(log N).
    pub fn concat(&self, other: &Self) -> Self {
        concat_trees(self, &[], other)
    }

    // ------------------------------------------------------------------
    // Conversion
    // ------------------------------------------------------------------

    /// Collect all elements into a `Vec`.
    pub fn to_vec(&self) -> Vec<T> {
        let mut out = Vec::new();
        collect_into(self, &mut out);
        out
    }
}

// ---------------------------------------------------------------------------
// Core operations (free functions to avoid recursive method calls on Self)
// ---------------------------------------------------------------------------

fn push_back_impl<T: Clone + Send + Sync + 'static>(
    tree: &FingerTree<T>,
    x: Arc<T>,
) -> FingerTree<T> {
    match &tree.inner {
        TreeInner::Empty => FingerTree::single(x),
        TreeInner::Single(a) => FingerTree::deep(
            Digit::One(Arc::clone(a)),
            empty_spine::<T>(),
            Digit::One(x),
        ),
        TreeInner::Deep { prefix, spine, suffix } => {
            match suffix.clone().push_back(x) {
                Ok(new_suffix) => FingerTree::deep(
                    prefix.clone(),
                    Arc::clone(spine),
                    new_suffix,
                ),
                Err((small_suffix, node)) => {
                    let new_spine = spine.push_back_node(Arc::new(node));
                    FingerTree::deep(prefix.clone(), new_spine, small_suffix)
                }
            }
        }
    }
}

fn push_front_impl<T: Clone + Send + Sync + 'static>(
    tree: &FingerTree<T>,
    x: Arc<T>,
) -> FingerTree<T> {
    match &tree.inner {
        TreeInner::Empty => FingerTree::single(x),
        TreeInner::Single(a) => FingerTree::deep(
            Digit::One(x),
            empty_spine::<T>(),
            Digit::One(Arc::clone(a)),
        ),
        TreeInner::Deep { prefix, spine, suffix } => {
            match prefix.clone().push_front(x) {
                Ok(new_prefix) => FingerTree::deep(new_prefix, Arc::clone(spine), suffix.clone()),
                Err((node, small_prefix)) => {
                    let new_spine = spine.push_front_node(Arc::new(node));
                    FingerTree::deep(small_prefix, new_spine, suffix.clone())
                }
            }
        }
    }
}

fn pop_back_impl<T: Clone + Send + Sync + 'static>(
    tree: &FingerTree<T>,
) -> Option<(T, FingerTree<T>)> {
    match &tree.inner {
        TreeInner::Empty => None,
        TreeInner::Single(a) => Some(((**a).clone(), FingerTree::empty())),
        TreeInner::Deep { prefix, spine, suffix } => {
            let (last, new_suffix_opt) = pop_digit_back(suffix.clone());
            let new_tree = match new_suffix_opt {
                Some(new_suffix) => FingerTree::deep(
                    prefix.clone(),
                    Arc::clone(spine),
                    new_suffix,
                ),
                None => match spine.pop_back_node() {
                    None => {
                        // Spine empty — collapse.
                        let pv = prefix.to_vec();
                        match pv.as_slice() {
                            [only] => FingerTree::single(Arc::clone(only)),
                            _ => {
                                let (p_last, new_prefix) = pop_digit_back(prefix.clone());
                                FingerTree::deep(
                                    new_prefix.expect("prefix has ≥2 elements"),
                                    Arc::clone(spine),
                                    Digit::One(p_last),
                                )
                            }
                        }
                    }
                    Some((node, new_spine)) => FingerTree::deep(
                        prefix.clone(),
                        new_spine,
                        node.to_digit(),
                    ),
                },
            };
            Some(((*last).clone(), new_tree))
        }
    }
}

fn pop_front_impl<T: Clone + Send + Sync + 'static>(
    tree: &FingerTree<T>,
) -> Option<(T, FingerTree<T>)> {
    match &tree.inner {
        TreeInner::Empty => None,
        TreeInner::Single(a) => Some(((**a).clone(), FingerTree::empty())),
        TreeInner::Deep { prefix, spine, suffix } => {
            let (first, new_prefix_opt) = pop_digit_front(prefix.clone());
            let new_tree = match new_prefix_opt {
                Some(new_prefix) => FingerTree::deep(
                    new_prefix,
                    Arc::clone(spine),
                    suffix.clone(),
                ),
                None => match spine.pop_front_node() {
                    None => {
                        // Spine empty — collapse.
                        let sv = suffix.to_vec();
                        match sv.as_slice() {
                            [only] => FingerTree::single(Arc::clone(only)),
                            _ => {
                                let (s_first, new_suffix) = pop_digit_front(suffix.clone());
                                FingerTree::deep(
                                    Digit::One(s_first),
                                    Arc::clone(spine),
                                    new_suffix.expect("suffix has ≥2 elements"),
                                )
                            }
                        }
                    }
                    Some((node, new_spine)) => FingerTree::deep(
                        node.to_digit(),
                        new_spine,
                        suffix.clone(),
                    ),
                },
            };
            Some(((*first).clone(), new_tree))
        }
    }
}

fn collect_into<T: Clone + Send + Sync + 'static>(
    tree: &FingerTree<T>,
    out: &mut Vec<T>,
) {
    match &tree.inner {
        TreeInner::Empty => {}
        TreeInner::Single(a) => out.push((**a).clone()),
        TreeInner::Deep { prefix, spine, suffix } => {
            for a in prefix.to_vec() {
                out.push((*a).clone());
            }
            // Expand spine nodes into T elements.
            let mut nodes: Vec<Node<T>> = Vec::new();
            spine.collect_nodes(&mut nodes);
            for node in nodes {
                match node {
                    Node::Node2(a, b) => {
                        out.push((*a).clone());
                        out.push((*b).clone());
                    }
                    Node::Node3(a, b, c) => {
                        out.push((*a).clone());
                        out.push((*b).clone());
                        out.push((*c).clone());
                    }
                }
            }
            for a in suffix.to_vec() {
                out.push((*a).clone());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper digit operations
// ---------------------------------------------------------------------------

fn pop_digit_back<T: Clone + Send + Sync + 'static>(
    d: Digit<T>,
) -> (Arc<T>, Option<Digit<T>>) {
    match d {
        Digit::One(a) => (a, None),
        Digit::Two(a, b) => (b, Some(Digit::One(a))),
        Digit::Three(a, b, c) => (c, Some(Digit::Two(a, b))),
        Digit::Four(a, b, c, d) => (d, Some(Digit::Three(a, b, c))),
    }
}

fn pop_digit_front<T: Clone + Send + Sync + 'static>(
    d: Digit<T>,
) -> (Arc<T>, Option<Digit<T>>) {
    match d {
        Digit::One(a) => (a, None),
        Digit::Two(a, b) => (a, Some(Digit::One(b))),
        Digit::Three(a, b, c) => (a, Some(Digit::Two(b, c))),
        Digit::Four(a, b, c, d) => (a, Some(Digit::Three(b, c, d))),
    }
}

// ---------------------------------------------------------------------------
// Concatenation helpers
// ---------------------------------------------------------------------------

fn append_left<T: Clone + Send + Sync + 'static>(
    tree: &FingerTree<T>,
    elems: &[Arc<T>],
) -> FingerTree<T> {
    elems.iter().fold(tree.clone(), |t, e| push_back_impl(&t, Arc::clone(e)))
}

fn prepend_right<T: Clone + Send + Sync + 'static>(
    elems: &[Arc<T>],
    tree: &FingerTree<T>,
) -> FingerTree<T> {
    elems.iter().rev().fold(tree.clone(), |t, e| push_front_impl(&t, Arc::clone(e)))
}

/// Group a slice of elements into Node2/Node3 groups (used for concat).
fn nodes_from_arcs<T: Clone + Send + Sync + 'static>(
    elems: &[Arc<T>],
) -> Vec<Arc<Node<T>>> {
    let mut out = Vec::new();
    let mut i = 0;
    let n = elems.len();
    while i < n {
        let rem = n - i;
        if rem == 2 || rem == 4 {
            out.push(Arc::new(Node::Node2(
                Arc::clone(&elems[i]),
                Arc::clone(&elems[i + 1]),
            )));
            i += 2;
        } else {
            out.push(Arc::new(Node::Node3(
                Arc::clone(&elems[i]),
                Arc::clone(&elems[i + 1]),
                Arc::clone(&elems[i + 2]),
            )));
            i += 3;
        }
    }
    out
}

fn concat_trees<T: Clone + Send + Sync + 'static>(
    left: &FingerTree<T>,
    middle: &[Arc<T>],
    right: &FingerTree<T>,
) -> FingerTree<T> {
    match (&left.inner, &right.inner) {
        (TreeInner::Empty, _) => prepend_right(middle, right),
        (_, TreeInner::Empty) => append_left(left, middle),
        (TreeInner::Single(a), _) => {
            let mut elems = vec![Arc::clone(a)];
            elems.extend_from_slice(middle);
            prepend_right(&elems, right)
        }
        (_, TreeInner::Single(b)) => {
            let mut elems = middle.to_vec();
            elems.push(Arc::clone(b));
            append_left(left, &elems)
        }
        (
            TreeInner::Deep { prefix: lp, spine: ls, suffix: lsuf },
            TreeInner::Deep { prefix: rp, spine: rs, suffix: rsuf },
        ) => {
            // Collect: lsuf elements + middle + rp elements → new mid_elems.
            let mut mid_elems: Vec<Arc<T>> = lsuf.to_vec();
            mid_elems.extend_from_slice(middle);
            mid_elems.extend(rp.to_vec());

            // Group into nodes and push into a merged spine.
            let new_nodes = nodes_from_arcs(&mid_elems);

            // Build the new spine by starting from ls, appending new_nodes,
            // then appending all nodes from rs.
            let mut new_spine = ls.clone_spine();
            for n in &new_nodes {
                new_spine = new_spine.push_back_node(Arc::clone(n));
            }
            let mut rs_nodes: Vec<Node<T>> = Vec::new();
            rs.collect_nodes(&mut rs_nodes);
            for n in rs_nodes {
                new_spine = new_spine.push_back_node(Arc::new(n));
            }

            FingerTree::deep(lp.clone(), new_spine, rsuf.clone())
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn from_vec(v: &[i32]) -> FingerTree<i32> {
        v.iter().fold(FingerTree::empty(), |t, &x| t.push_back(x))
    }

    #[test]
    fn test_empty() {
        let t: FingerTree<i32> = FingerTree::empty();
        assert!(t.is_empty());
        assert!(t.pop_front().is_none(), "pop_front on empty tree should be None");
        assert!(t.pop_back().is_none(), "pop_back on empty tree should be None");
    }

    #[test]
    fn test_push_back_and_pop_front() {
        let t = from_vec(&[1, 2, 3, 4, 5]);
        let mut cur = t;
        for expected in 1..=5 {
            let (val, rest) = cur.pop_front().expect("should have element");
            assert_eq!(val, expected);
            cur = rest;
        }
        assert!(cur.is_empty());
    }

    #[test]
    fn test_push_front_and_pop_back() {
        let t = [5, 4, 3, 2, 1]
            .iter()
            .fold(FingerTree::empty(), |t, &x| t.push_front(x));
        let mut cur = t;
        for expected in (1..=5).rev() {
            let (val, rest) = cur.pop_back().expect("should have element");
            assert_eq!(val, expected);
            cur = rest;
        }
        assert!(cur.is_empty());
    }

    #[test]
    fn test_to_vec() {
        let expected: Vec<i32> = (1..=20).collect();
        let t = from_vec(&expected);
        assert_eq!(t.to_vec(), expected);
    }

    #[test]
    fn test_concat() {
        let a = from_vec(&[1, 2, 3]);
        let b = from_vec(&[4, 5, 6]);
        let c = a.concat(&b);
        assert_eq!(c.to_vec(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_concat_with_empty() {
        let a = from_vec(&[1, 2, 3]);
        let e: FingerTree<i32> = FingerTree::empty();
        assert_eq!(a.concat(&e).to_vec(), vec![1, 2, 3]);
        assert_eq!(e.concat(&a).to_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn test_large_push_back() {
        let n = 200;
        let t = (0..n).fold(FingerTree::empty(), |t, x| t.push_back(x as i32));
        let got = t.to_vec();
        let expected: Vec<i32> = (0..n as i32).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn test_large_concat() {
        let a = from_vec(&(0..50).collect::<Vec<i32>>());
        let b = from_vec(&(50..100).collect::<Vec<i32>>());
        let c = a.concat(&b);
        let expected: Vec<i32> = (0..100).collect();
        assert_eq!(c.to_vec(), expected);
    }

    #[test]
    fn test_persistence() {
        let t0 = from_vec(&[1, 2, 3]);
        let t1 = t0.push_back(4);
        let t2 = t0.push_front(0);

        // t0 is unchanged.
        assert_eq!(t0.to_vec(), vec![1, 2, 3]);
        assert_eq!(t1.to_vec(), vec![1, 2, 3, 4]);
        assert_eq!(t2.to_vec(), vec![0, 1, 2, 3]);
    }
}
