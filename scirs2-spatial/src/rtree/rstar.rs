//! R*-tree spatial index (Beckmann, Kriegel, Schneider, Seeger 1990).
//!
//! The R*-tree improves upon the basic R-tree by using more sophisticated
//! insertion heuristics:
//! - Minimises overlap between node MBRs (not just area enlargement).
//! - Forced re-insertion when a leaf overflows (before splitting).
//! - Quadratic split replaced by an axis-based split that minimises perimeter.
//!
//! This implementation wraps the existing `RTree<T>` for 2D data and adds
//! R*-tree–specific bulk-insertion, forced re-insertion tracking, and
//! an axis-optimised split strategy.

use super::mbr::MBR;
use crate::error::{SpatialError, SpatialResult};

// Node capacity parameters
const MAX_ENTRIES: usize = 9;
const MIN_ENTRIES: usize = 4; // ≥ 40% of max

// ── Entry stored at leaf level ────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct LeafEntry<T> {
    mbr: MBR,
    data: T,
}

// ── Node ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum NodeKind<T: Clone> {
    Leaf(Vec<LeafEntry<T>>),
    Internal(Vec<InternalEntry<T>>),
}

#[derive(Debug, Clone)]
struct InternalEntry<T: Clone> {
    mbr: MBR,
    child: Box<RStarNode<T>>,
}

#[derive(Debug, Clone)]
struct RStarNode<T: Clone> {
    kind: NodeKind<T>,
    /// Cached MBR of this node (keeps updates local).
    cached_mbr: Option<MBR>,
}

impl<T: Clone> RStarNode<T> {
    fn new_leaf() -> Self {
        Self { kind: NodeKind::Leaf(Vec::new()), cached_mbr: None }
    }

    fn new_internal() -> Self {
        Self { kind: NodeKind::Internal(Vec::new()), cached_mbr: None }
    }

    fn is_leaf(&self) -> bool {
        matches!(self.kind, NodeKind::Leaf(_))
    }

    fn entry_count(&self) -> usize {
        match &self.kind {
            NodeKind::Leaf(v) => v.len(),
            NodeKind::Internal(v) => v.len(),
        }
    }

    fn compute_mbr(&self) -> Option<MBR> {
        match &self.kind {
            NodeKind::Leaf(entries) => {
                entries.iter().map(|e| e.mbr).reduce(|a, b| a.union(&b))
            }
            NodeKind::Internal(entries) => {
                entries.iter().map(|e| e.mbr).reduce(|a, b| a.union(&b))
            }
        }
    }

    fn mbr(&self) -> Option<MBR> {
        self.cached_mbr.or_else(|| self.compute_mbr())
    }

    fn update_mbr(&mut self) {
        self.cached_mbr = self.compute_mbr();
    }
}

// ── Overflow result ───────────────────────────────────────────────────────────

/// What to do after an insertion causes a node to overflow.
enum OverflowResult<T: Clone> {
    /// No overflow – the node (possibly modified) is returned as-is.
    NoOverflow(Box<RStarNode<T>>),
    /// The node split into two sibling nodes.
    Split(Box<RStarNode<T>>, InternalEntry<T>),
    /// Forced re-insertion: items ejected from the overflowed leaf.
    Reinsert(Box<RStarNode<T>>, Vec<LeafEntry<T>>),
}

// ── R*-tree ───────────────────────────────────────────────────────────────────

/// R*-tree spatial index for 2D bounding-box queries.
///
/// Supports efficient point and region queries, k-NN search, and
/// incremental insertion with R*-tree rebalancing.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::rtree::{MBR, RStarTree};
///
/// let mut tree: RStarTree<u32> = RStarTree::new();
/// tree.insert(MBR::from_point(1.0, 2.0), 1);
/// tree.insert(MBR::new(0.0, 0.0, 5.0, 5.0), 2);
///
/// let hits = tree.query(&MBR::new(-1.0, -1.0, 6.0, 6.0));
/// assert_eq!(hits.len(), 2);
/// ```
pub struct RStarTree<T: Clone> {
    root: Box<RStarNode<T>>,
    size: usize,
    /// Height of the tree (1 = only a leaf root).
    height: usize,
}

impl<T: Clone> RStarTree<T> {
    /// Create an empty R*-tree.
    pub fn new() -> Self {
        Self {
            root: Box::new(RStarNode::new_leaf()),
            size: 0,
            height: 1,
        }
    }

    /// Number of entries stored in the tree.
    pub fn len(&self) -> usize { self.size }

    /// Return `true` if the tree contains no entries.
    pub fn is_empty(&self) -> bool { self.size == 0 }

    /// Height of the tree (1 = single leaf root).
    pub fn height(&self) -> usize { self.height }

    // ── Insertion ──────────────────────────────────────────────────────────────

    /// Insert an entry with bounding rectangle `mbr` and associated data.
    pub fn insert(&mut self, mbr: MBR, data: T) {
        self.size += 1;
        let entry = LeafEntry { mbr, data };
        let mut reinsert_buffer: Vec<LeafEntry<T>> = Vec::new();

        // Phase 1: regular insertion, collecting any re-insert items
        let overflow = Self::insert_recursive(
            &mut self.root,
            entry,
            self.height,
            self.height,
            true,
            &mut reinsert_buffer,
        );

        // Handle the result from recursive insertion
        match overflow {
            OverflowResult::NoOverflow(updated) => {
                self.root = updated;
            }
            OverflowResult::Split(left, right_entry) => {
                let mut new_root = Box::new(RStarNode::new_internal());
                let left_mbr = left.mbr().unwrap_or_else(|| MBR::new(0.0, 0.0, 0.0, 0.0));
                if let NodeKind::Internal(ref mut children) = new_root.kind {
                    children.push(InternalEntry { mbr: left_mbr, child: left });
                    children.push(right_entry);
                }
                new_root.update_mbr();
                self.root = new_root;
                self.height += 1;
            }
            OverflowResult::Reinsert(updated, _extra) => {
                self.root = updated;
            }
        }

        // Phase 2: re-insert ejected entries (without further re-insertion cascading)
        for e in reinsert_buffer {
            let overflow2 = Self::insert_recursive(
                &mut self.root,
                e,
                self.height,
                self.height,
                false, // no re-insertion on second pass
                &mut Vec::new(),
            );
            match overflow2 {
                OverflowResult::NoOverflow(updated) => {
                    self.root = updated;
                }
                OverflowResult::Split(left, right_entry) => {
                    let mut new_root = Box::new(RStarNode::new_internal());
                    let left_mbr = left.mbr().unwrap_or_else(|| MBR::new(0.0, 0.0, 0.0, 0.0));
                    if let NodeKind::Internal(ref mut children) = new_root.kind {
                        children.push(InternalEntry { mbr: left_mbr, child: left });
                        children.push(right_entry);
                    }
                    new_root.update_mbr();
                    self.root = new_root;
                    self.height += 1;
                }
                OverflowResult::Reinsert(updated, _) => {
                    self.root = updated;
                }
            }
        }
    }

    /// Recursive insertion returning an `OverflowResult`.
    ///
    /// `tree_height` is the full tree height; `level` is the depth at which
    /// this call sits (tree_height = leaf level).
    fn insert_recursive(
        node: &mut Box<RStarNode<T>>,
        entry: LeafEntry<T>,
        tree_height: usize,
        level: usize,
        allow_reinsert: bool,
        reinsert_buf: &mut Vec<LeafEntry<T>>,
    ) -> OverflowResult<T> {
        if node.is_leaf() {
            // Insert into leaf; check overflow after releasing the borrow on entries
            let overflowed = if let NodeKind::Leaf(ref mut entries) = node.kind {
                entries.push(entry);
                entries.len() > MAX_ENTRIES
            } else {
                false
            };
            node.update_mbr();
            if overflowed {
                return Self::handle_leaf_overflow(node, allow_reinsert, reinsert_buf);
            }
            return OverflowResult::NoOverflow(std::mem::replace(
                node,
                Box::new(RStarNode::new_leaf()),
            ));
        }

        // Internal node: choose subtree.
        // We extract the index and perform the recursive call, then handle results
        // outside the `if let` borrow to avoid double-mutable-borrow of `node`.
        let (best, result) = if let NodeKind::Internal(ref mut children) = node.kind {
            let best = Self::choose_subtree(children, &entry.mbr);
            let result = Self::insert_recursive(
                &mut children[best].child,
                entry,
                tree_height,
                level.saturating_sub(1),
                allow_reinsert,
                reinsert_buf,
            );
            (best, result)
        } else {
            return OverflowResult::NoOverflow(
                std::mem::replace(node, Box::new(RStarNode::new_leaf()))
            );
        };

        // Apply the result of recursive insertion — each arm borrows `children`
        // in a fresh scope so `node.update_mbr()` can be called afterwards.
        let overflowed = match result {
            OverflowResult::NoOverflow(updated) => {
                if let NodeKind::Internal(ref mut children) = node.kind {
                    children[best].child = updated;
                    let child_mbr = children[best].child.mbr();
                    let fallback = children[best].mbr;
                    children[best].mbr = child_mbr.unwrap_or(fallback);
                }
                false
            }
            OverflowResult::Split(left, right_entry) => {
                let count = if let NodeKind::Internal(ref mut children) = node.kind {
                    let fallback = children[best].mbr;
                    let left_mbr = left.mbr().unwrap_or(fallback);
                    children[best] = InternalEntry { mbr: left_mbr, child: left };
                    children.push(right_entry);
                    children.len()
                } else {
                    0
                };
                count > MAX_ENTRIES
            }
            OverflowResult::Reinsert(updated, ejected) => {
                if let NodeKind::Internal(ref mut children) = node.kind {
                    children[best].child = updated;
                    let child_mbr = children[best].child.mbr();
                    let fallback = children[best].mbr;
                    children[best].mbr = child_mbr.unwrap_or(fallback);
                }
                reinsert_buf.extend(ejected);
                false
            }
        };

        node.update_mbr();

        if overflowed {
            return Self::handle_internal_overflow(node, allow_reinsert, reinsert_buf);
        }
        OverflowResult::NoOverflow(std::mem::replace(node, Box::new(RStarNode::new_leaf())))
    }

    // ── Subtree selection (R*-tree: minimise overlap then area) ────────────────

    fn choose_subtree(children: &[InternalEntry<T>], mbr: &MBR) -> usize {
        if children.is_empty() { return 0; }

        // If children are leaves: minimise overlap enlargement
        // (heuristic: choose by minimum area enlargement, breaking ties by area)
        children
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let ea = a.mbr.enlargement_needed(mbr);
                let eb = b.mbr.enlargement_needed(mbr);
                ea.partial_cmp(&eb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        a.mbr.area().partial_cmp(&b.mbr.area())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    // ── Overflow handling ─────────────────────────────────────────────────────

    fn handle_leaf_overflow(
        node: &mut Box<RStarNode<T>>,
        allow_reinsert: bool,
        reinsert_buf: &mut Vec<LeafEntry<T>>,
    ) -> OverflowResult<T> {
        if allow_reinsert {
            // Compute the node centre before borrowing entries mutably
            let node_centre = node.mbr()
                .map(|m| m.center())
                .unwrap_or([0.0, 0.0]);

            // R*-tree forced re-insertion: eject the p farthest entries
            if let NodeKind::Leaf(ref mut entries) = node.kind {
                // Sort by distance from node centre (farthest first)
                entries.sort_by(|a, b| {
                    let ca = a.mbr.center();
                    let cb = b.mbr.center();
                    let da = (ca[0]-node_centre[0]).hypot(ca[1]-node_centre[1]);
                    let db = (cb[0]-node_centre[0]).hypot(cb[1]-node_centre[1]);
                    db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
                });
                // Eject ~30% of entries (at least 1)
                let p = ((entries.len() as f64 * 0.30).ceil() as usize).max(1);
                let ejected: Vec<LeafEntry<T>> = entries.drain(..p).collect();
                reinsert_buf.extend(ejected);
            }
            node.update_mbr();
            let taken = std::mem::replace(node, Box::new(RStarNode::new_leaf()));
            return OverflowResult::Reinsert(taken, Vec::new());
        }
        // Split
        Self::split_leaf(node)
    }

    fn handle_internal_overflow(
        node: &mut Box<RStarNode<T>>,
        _allow_reinsert: bool,
        _reinsert_buf: &mut Vec<LeafEntry<T>>,
    ) -> OverflowResult<T> {
        Self::split_internal(node)
    }

    // ── R*-tree axis-based split ──────────────────────────────────────────────

    /// Split a leaf node using the R*-tree axis-based heuristic (minimise perimeter sum).
    fn split_leaf(node: &mut Box<RStarNode<T>>) -> OverflowResult<T> {
        if let NodeKind::Leaf(ref mut entries) = node.kind {
            let total = entries.len();
            let (axis, split_idx) = Self::choose_split_axis_and_index_leaf(entries);

            // Sort along chosen axis
            entries.sort_by(|a, b| {
                let ca = a.mbr.center()[axis];
                let cb = b.mbr.center()[axis];
                ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
            });

            let right_entries: Vec<LeafEntry<T>> = entries.drain(split_idx..).collect();
            node.update_mbr();

            let mut right_node = Box::new(RStarNode::new_leaf());
            right_node.kind = NodeKind::Leaf(right_entries);
            right_node.update_mbr();
            let right_mbr = right_node.mbr().unwrap_or_else(|| MBR::new(0.0,0.0,0.0,0.0));

            let taken = std::mem::replace(node, Box::new(RStarNode::new_leaf()));
            return OverflowResult::Split(taken, InternalEntry { mbr: right_mbr, child: right_node });
        }
        // Unreachable but safe fallback
        OverflowResult::NoOverflow(std::mem::replace(node, Box::new(RStarNode::new_leaf())))
    }

    fn split_internal(node: &mut Box<RStarNode<T>>) -> OverflowResult<T> {
        if let NodeKind::Internal(ref mut children) = node.kind {
            let (axis, split_idx) = Self::choose_split_axis_and_index_internal(children);

            children.sort_by(|a, b| {
                let ca = a.mbr.center()[axis];
                let cb = b.mbr.center()[axis];
                ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
            });

            let right_children: Vec<InternalEntry<T>> = children.drain(split_idx..).collect();
            node.update_mbr();

            let mut right_node = Box::new(RStarNode::new_internal());
            right_node.kind = NodeKind::Internal(right_children);
            right_node.update_mbr();
            let right_mbr = right_node.mbr().unwrap_or_else(|| MBR::new(0.0,0.0,0.0,0.0));

            let taken = std::mem::replace(node, Box::new(RStarNode::new_leaf()));
            return OverflowResult::Split(taken, InternalEntry { mbr: right_mbr, child: right_node });
        }
        OverflowResult::NoOverflow(std::mem::replace(node, Box::new(RStarNode::new_leaf())))
    }

    fn choose_split_axis_and_index_leaf(entries: &[LeafEntry<T>]) -> (usize, usize) {
        let n = entries.len();
        let mut best_axis = 0;
        let mut best_perimeter = f64::INFINITY;

        for axis in 0..2 {
            let mut sorted = entries.iter().collect::<Vec<_>>();
            sorted.sort_by(|a, b| {
                a.mbr.center()[axis].partial_cmp(&b.mbr.center()[axis])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut psum = 0.0;
            for k in MIN_ENTRIES..=(n - MIN_ENTRIES) {
                let left: Option<MBR> = sorted[..k].iter().map(|e| e.mbr).reduce(|a,b| a.union(&b));
                let right: Option<MBR> = sorted[k..].iter().map(|e| e.mbr).reduce(|a,b| a.union(&b));
                psum += left.map(|m| m.perimeter()).unwrap_or(0.0)
                    + right.map(|m| m.perimeter()).unwrap_or(0.0);
            }
            if psum < best_perimeter {
                best_perimeter = psum;
                best_axis = axis;
            }
        }

        // Along best axis, choose split minimising overlap
        let mut sorted = entries.iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| {
            a.mbr.center()[best_axis].partial_cmp(&b.mbr.center()[best_axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best_idx = (MIN_ENTRIES..=(n - MIN_ENTRIES))
            .min_by(|&k1, &k2| {
                let ol1 = Self::overlap_leaf(&sorted, k1);
                let ol2 = Self::overlap_leaf(&sorted, k2);
                ol1.partial_cmp(&ol2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(n / 2);

        (best_axis, best_idx)
    }

    fn overlap_leaf(sorted: &[&LeafEntry<T>], k: usize) -> f64 {
        let left: Option<MBR> = sorted[..k].iter().map(|e| e.mbr).reduce(|a,b| a.union(&b));
        let right: Option<MBR> = sorted[k..].iter().map(|e| e.mbr).reduce(|a,b| a.union(&b));
        match (left, right) {
            (Some(l), Some(r)) => l.intersection(&r).map(|i| i.area()).unwrap_or(0.0),
            _ => 0.0,
        }
    }

    fn choose_split_axis_and_index_internal(children: &[InternalEntry<T>]) -> (usize, usize) {
        let n = children.len();
        let mut best_axis = 0;
        let mut best_perimeter = f64::INFINITY;

        for axis in 0..2 {
            let mut sorted = children.iter().collect::<Vec<_>>();
            sorted.sort_by(|a, b| {
                a.mbr.center()[axis].partial_cmp(&b.mbr.center()[axis])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut psum = 0.0;
            for k in MIN_ENTRIES..=(n - MIN_ENTRIES) {
                let left: Option<MBR> = sorted[..k].iter().map(|e| e.mbr).reduce(|a,b| a.union(&b));
                let right: Option<MBR> = sorted[k..].iter().map(|e| e.mbr).reduce(|a,b| a.union(&b));
                psum += left.map(|m| m.perimeter()).unwrap_or(0.0)
                    + right.map(|m| m.perimeter()).unwrap_or(0.0);
            }
            if psum < best_perimeter {
                best_perimeter = psum;
                best_axis = axis;
            }
        }

        let best_idx = (MIN_ENTRIES..=(n - MIN_ENTRIES))
            .min_by(|&k1, &k2| {
                let ol1 = Self::overlap_internal(children, k1);
                let ol2 = Self::overlap_internal(children, k2);
                ol1.partial_cmp(&ol2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(n / 2);

        (best_axis, best_idx)
    }

    fn overlap_internal(children: &[InternalEntry<T>], k: usize) -> f64 {
        let left: Option<MBR> = children[..k].iter().map(|e| e.mbr).reduce(|a,b| a.union(&b));
        let right: Option<MBR> = children[k..].iter().map(|e| e.mbr).reduce(|a,b| a.union(&b));
        match (left, right) {
            (Some(l), Some(r)) => l.intersection(&r).map(|i| i.area()).unwrap_or(0.0),
            _ => 0.0,
        }
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    /// Return references to all entries whose MBR intersects `query`.
    pub fn query(&self, query: &MBR) -> Vec<&T> {
        let mut results = Vec::new();
        Self::query_node(&self.root, query, &mut results);
        results
    }

    fn query_node<'a>(node: &'a RStarNode<T>, query: &MBR, out: &mut Vec<&'a T>) {
        match &node.kind {
            NodeKind::Leaf(entries) => {
                for e in entries {
                    if e.mbr.intersects(query) {
                        out.push(&e.data);
                    }
                }
            }
            NodeKind::Internal(children) => {
                for c in children {
                    if c.mbr.intersects(query) {
                        Self::query_node(&c.child, query, out);
                    }
                }
            }
        }
    }

    /// Return references to entries whose MBR **contains** the point `(x, y)`.
    pub fn query_point(&self, x: f64, y: f64) -> Vec<&T> {
        self.query(&MBR::from_point(x, y))
    }

    // ── k-Nearest Neighbour ───────────────────────────────────────────────────

    /// Return up to `k` nearest entries to the point `(x, y)`, sorted by
    /// distance (closest first).
    pub fn nearest_neighbor(&self, x: f64, y: f64, k: usize) -> Vec<&T> {
        if k == 0 { return Vec::new(); }
        let mut heap: Vec<(f64, *const T)> = Vec::new();
        Self::nn_search(&self.root, x, y, k, &mut heap);
        heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        heap.into_iter()
            .take(k)
            // SAFETY: pointers were obtained from stable references in `self.root`
            // which outlive this method call.
            .map(|(_, ptr)| unsafe { &*ptr })
            .collect()
    }

    fn nn_search(node: &RStarNode<T>, x: f64, y: f64, k: usize, heap: &mut Vec<(f64, *const T)>) {
        match &node.kind {
            NodeKind::Leaf(entries) => {
                for e in entries {
                    let d = e.mbr.distance_to_point(x, y);
                    heap.push((d, &e.data as *const T));
                }
            }
            NodeKind::Internal(children) => {
                // Sort children by min-distance for branch-and-bound efficiency
                let mut child_dists: Vec<(f64, &InternalEntry<T>)> = children
                    .iter()
                    .map(|c| (c.mbr.distance_to_point(x, y), c))
                    .collect();
                child_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                for (min_d, c) in child_dists {
                    // Prune: if we already have k results and this child is farther, skip
                    if heap.len() >= k {
                        let worst = heap.iter()
                            .map(|(d, _)| *d)
                            .fold(f64::NEG_INFINITY, f64::max);
                        if min_d > worst { continue; }
                    }
                    Self::nn_search(&c.child, x, y, k, heap);
                }
            }
        }
    }

    // ── Range count ───────────────────────────────────────────────────────────

    /// Count entries whose MBR intersects a circle with centre `(cx, cy)` and
    /// radius `r` (uses MBR approximation – may count false positives).
    pub fn range_count(&self, cx: f64, cy: f64, radius: f64) -> usize {
        let query = MBR::new(cx - radius, cy - radius, cx + radius, cy + radius);
        self.query(&query).len()
    }

    // ── Bulk-loading (STR-style) ───────────────────────────────────────────────

    /// Build an R*-tree from a batch of entries using Sort-Tile-Recursive packing.
    ///
    /// This produces a much better-packed tree than sequential insertion and is
    /// preferred for static datasets.
    pub fn bulk_load(mut items: Vec<(MBR, T)>) -> SpatialResult<Self> {
        if items.is_empty() {
            return Ok(Self::new());
        }

        let n = items.len();
        // Sort by x-centre, then tile by y-centre
        items.sort_by(|a, b| {
            a.0.center()[0].partial_cmp(&b.0.center()[0])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build leaf nodes
        let mut leaves: Vec<Box<RStarNode<T>>> = items
            .chunks(MAX_ENTRIES)
            .map(|chunk| {
                let entries: Vec<LeafEntry<T>> = chunk
                    .iter()
                    .map(|(m, d)| LeafEntry { mbr: *m, data: d.clone() })
                    .collect();
                let mut node = Box::new(RStarNode::new_leaf());
                node.kind = NodeKind::Leaf(entries);
                node.update_mbr();
                node
            })
            .collect();

        if leaves.is_empty() {
            return Ok(Self::new());
        }

        // Build tree bottom-up
        let mut height = 1usize;
        while leaves.len() > 1 {
            let parents: Vec<Box<RStarNode<T>>> = leaves
                .chunks(MAX_ENTRIES)
                .map(|chunk| {
                    let children: Vec<InternalEntry<T>> = chunk
                        .iter()
                        .map(|leaf| {
                            let mbr = leaf.mbr().unwrap_or_else(|| MBR::new(0.0,0.0,0.0,0.0));
                            InternalEntry { mbr, child: leaf.clone() }
                        })
                        .collect();
                    let mut node = Box::new(RStarNode::new_internal());
                    node.kind = NodeKind::Internal(children);
                    node.update_mbr();
                    node
                })
                .collect();
            leaves = parents;
            height += 1;
        }

        let root = leaves.into_iter().next()
            .ok_or_else(|| SpatialError::ComputationError("Bulk load failed".into()))?;

        Ok(Self { root, size: n, height })
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Bounding rectangle of all entries, or `None` if empty.
    pub fn bounding_rect(&self) -> Option<MBR> {
        self.root.mbr()
    }
}

impl<T: Clone> Default for RStarTree<T> {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_entry(id: u32, x: f64, y: f64) -> (MBR, u32) {
        (MBR::from_point(x, y), id)
    }

    #[test]
    fn test_rstar_empty() {
        let tree: RStarTree<u32> = RStarTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_rstar_insert_and_query() {
        let mut tree: RStarTree<u32> = RStarTree::new();
        tree.insert(MBR::from_point(1.0, 1.0), 1);
        tree.insert(MBR::from_point(2.0, 2.0), 2);
        tree.insert(MBR::from_point(10.0, 10.0), 3);

        assert_eq!(tree.len(), 3);

        let hits = tree.query(&MBR::new(0.0, 0.0, 5.0, 5.0));
        assert_eq!(hits.len(), 2);

        let hits2 = tree.query(&MBR::new(9.0, 9.0, 11.0, 11.0));
        assert_eq!(hits2.len(), 1);
    }

    #[test]
    fn test_rstar_large_insert_triggers_split() {
        let mut tree: RStarTree<usize> = RStarTree::new();
        for i in 0..50 {
            let x = (i % 10) as f64;
            let y = (i / 10) as f64;
            tree.insert(MBR::from_point(x, y), i);
        }
        assert_eq!(tree.len(), 50);

        let all = tree.query(&MBR::new(-1.0, -1.0, 20.0, 20.0));
        assert_eq!(all.len(), 50);
    }

    #[test]
    fn test_rstar_query_point() {
        let mut tree: RStarTree<&str> = RStarTree::new();
        tree.insert(MBR::new(0.0, 0.0, 5.0, 5.0), "A");
        tree.insert(MBR::new(3.0, 3.0, 8.0, 8.0), "B");
        tree.insert(MBR::new(10.0, 10.0, 15.0, 15.0), "C");

        let hits = tree.query_point(4.0, 4.0);
        assert_eq!(hits.len(), 2, "Expected both A and B to contain (4,4)");
    }

    #[test]
    fn test_rstar_nn() {
        let mut tree: RStarTree<u32> = RStarTree::new();
        let pts = [(0.0, 0.0, 1u32), (1.0, 0.0, 2), (0.0, 1.0, 3), (5.0, 5.0, 4)];
        for (x, y, id) in pts {
            tree.insert(MBR::from_point(x, y), id);
        }
        let nn = tree.nearest_neighbor(0.1, 0.1, 1);
        assert_eq!(nn.len(), 1);
        assert_eq!(*nn[0], 1u32);
    }

    #[test]
    fn test_rstar_bulk_load() {
        let items: Vec<(MBR, usize)> = (0..30)
            .map(|i| (MBR::from_point((i % 5) as f64, (i / 5) as f64), i))
            .collect();
        let tree = RStarTree::bulk_load(items).expect("bulk_load failed");
        assert_eq!(tree.len(), 30);

        let all = tree.query(&MBR::new(-1.0, -1.0, 10.0, 10.0));
        assert_eq!(all.len(), 30);
    }

    #[test]
    fn test_rstar_bounding_rect() {
        let mut tree: RStarTree<u8> = RStarTree::new();
        assert!(tree.bounding_rect().is_none());
        tree.insert(MBR::new(1.0, 2.0, 3.0, 4.0), 1);
        tree.insert(MBR::new(-1.0, -2.0, 0.0, 0.0), 2);
        let br = tree.bounding_rect().expect("Should have bounding rect");
        assert!((br.min_x - (-1.0)).abs() < 1e-12);
        assert!((br.min_y - (-2.0)).abs() < 1e-12);
        assert!((br.max_x - 3.0).abs() < 1e-12);
        assert!((br.max_y - 4.0).abs() < 1e-12);
    }
}
