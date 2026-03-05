//! Van Emde Boas Tree — O(log log u) integer set operations over a bounded
//! universe [0, u).
//!
//! A Van Emde Boas (vEB) tree supports the following operations in
//! **O(log log u)** time where `u` is the universe size:
//!
//! - `insert(x)` / `delete(x)` / `member(x)`
//! - `minimum()` / `maximum()`
//! - `successor(x)` / `predecessor(x)`
//!
//! This implementation supports universes up to **u = 2^16 = 65 536**, which
//! keeps the recursive structure shallow (depth ≤ 16 for the worst case)
//! while remaining practically useful for dense integer sets.
//!
//! # Memory
//!
//! The recursive structure allocates only for non-empty sub-trees; the worst
//! case is O(u) but average-case usage for sparse sets is much lower.
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::data_structures::VEBTree;
//!
//! let mut t = VEBTree::new(64);
//! t.insert(5);
//! t.insert(10);
//! t.insert(3);
//!
//! assert_eq!(t.minimum(), Some(3));
//! assert_eq!(t.maximum(), Some(10));
//! assert_eq!(t.successor(5), Some(10));
//! assert_eq!(t.predecessor(10), Some(5));
//! t.delete(5);
//! assert!(!t.member(5));
//! ```

use std::fmt;

// ============================================================================
// Universe limit
// ============================================================================

/// Maximum universe size supported.
const MAX_UNIVERSE: u32 = 1 << 16; // 65 536

// ============================================================================
// VEBTree
// ============================================================================

/// A Van Emde Boas tree for integer keys in the range `[0, universe)`.
///
/// The universe size is rounded up to the next perfect square at each
/// recursive level following the standard algorithm.
pub struct VEBTree {
    /// Universe size for this node.
    universe: u32,
    /// Minimum element stored at this node (not propagated to children).
    min: Option<u32>,
    /// Maximum element stored at this node (not propagated to children).
    max: Option<u32>,
    /// Recursive sub-trees indexed by the *high* part of each key.
    clusters: Vec<Option<Box<VEBTree>>>,
    /// Summary tree that records which clusters are non-empty.
    summary: Option<Box<VEBTree>>,
}

impl VEBTree {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Creates a new vEB tree over the universe `[0, universe)`.
    ///
    /// `universe` is clamped to [`MAX_UNIVERSE`] and rounded up to at least 2
    /// internally.
    ///
    /// # Panics
    ///
    /// Does not panic — zero is silently treated as 2.
    pub fn new(universe: u32) -> Self {
        let u = universe.clamp(2, MAX_UNIVERSE);
        VEBTree::create(u)
    }

    /// Internal recursive constructor.
    fn create(u: u32) -> Self {
        if u <= 2 {
            // Base case: clusters and summary are not needed.
            return VEBTree {
                universe: 2,
                min: None,
                max: None,
                clusters: Vec::new(),
                summary: None,
            };
        }

        let upper = upper_sqrt(u);
        let lower = lower_sqrt(u);

        // Allocate `upper` empty cluster slots (lazily allocated).
        let clusters = (0..upper).map(|_| None).collect();

        VEBTree {
            universe: u,
            min: None,
            max: None,
            clusters,
            summary: Some(Box::new(VEBTree::create(upper))),
        }
    }

    // ------------------------------------------------------------------
    // Core operations
    // ------------------------------------------------------------------

    /// Returns the universe size of this tree.
    pub fn universe(&self) -> u32 {
        self.universe
    }

    /// Returns `true` if `x` is in the set.
    pub fn member(&self, x: u32) -> bool {
        if x >= self.universe {
            return false;
        }
        match (self.min, self.max) {
            (Some(mn), _) if mn == x => true,
            (_, Some(mx)) if mx == x => true,
            _ => {
                if self.universe <= 2 {
                    return false;
                }
                let hi = self.high(x);
                let lo = self.low(x);
                match self.clusters.get(hi as usize).and_then(|c| c.as_deref()) {
                    Some(cluster) => cluster.member(lo),
                    None => false,
                }
            }
        }
    }

    /// Inserts `x` into the set.  No-op if already present.
    pub fn insert(&mut self, x: u32) {
        if x >= self.universe {
            return;
        }
        self.insert_internal(x);
    }

    /// Deletes `x` from the set.  No-op if not present.
    pub fn delete(&mut self, x: u32) {
        if x >= self.universe {
            return;
        }
        self.delete_internal(x);
    }

    /// Returns the minimum element, or `None` if the set is empty.
    pub fn minimum(&self) -> Option<u32> {
        self.min
    }

    /// Returns the maximum element, or `None` if the set is empty.
    pub fn maximum(&self) -> Option<u32> {
        self.max
    }

    /// Returns the smallest element strictly greater than `x`, or `None`.
    pub fn successor(&self, x: u32) -> Option<u32> {
        if self.universe <= 2 {
            if x == 0 && self.max == Some(1) {
                return Some(1);
            }
            return None;
        }
        // If x is less than the minimum, the minimum is the successor.
        if let Some(mn) = self.min {
            if x < mn {
                return Some(mn);
            }
        }
        let hi = self.high(x);
        let lo = self.low(x);
        let max_lo = self
            .clusters
            .get(hi as usize)
            .and_then(|c| c.as_deref())
            .and_then(|c| c.maximum());

        if let Some(max_low) = max_lo {
            if lo < max_low {
                // Successor is in the same cluster.
                let succ_lo = self.clusters[hi as usize]
                    .as_deref()
                    .and_then(|c| c.successor(lo))?;
                return Some(self.index(hi, succ_lo));
            }
        }
        // Successor is in a later cluster.
        let succ_cluster = self.summary.as_deref().and_then(|s| s.successor(hi))?;
        let min_lo = self.clusters[succ_cluster as usize]
            .as_deref()
            .and_then(|c| c.minimum())?;
        Some(self.index(succ_cluster, min_lo))
    }

    /// Returns the largest element strictly less than `x`, or `None`.
    pub fn predecessor(&self, x: u32) -> Option<u32> {
        if self.universe <= 2 {
            if x == 1 && self.min == Some(0) {
                return Some(0);
            }
            return None;
        }
        if let Some(mx) = self.max {
            if x > mx {
                return Some(mx);
            }
        }
        let hi = self.high(x);
        let lo = self.low(x);
        let min_lo = self
            .clusters
            .get(hi as usize)
            .and_then(|c| c.as_deref())
            .and_then(|c| c.minimum());

        if let Some(min_low) = min_lo {
            if lo > min_low {
                let pred_lo = self.clusters[hi as usize]
                    .as_deref()
                    .and_then(|c| c.predecessor(lo))?;
                return Some(self.index(hi, pred_lo));
            }
        }
        // Predecessor is in an earlier cluster or is self.min.
        let pred_cluster = self.summary.as_deref().and_then(|s| s.predecessor(hi));
        match pred_cluster {
            Some(pc) => {
                let max_lo = self.clusters[pc as usize]
                    .as_deref()
                    .and_then(|c| c.maximum())?;
                Some(self.index(pc, max_lo))
            }
            None => {
                // The only predecessor can be min (not stored in clusters).
                if let Some(mn) = self.min {
                    if x > mn {
                        return Some(mn);
                    }
                }
                None
            }
        }
    }

    /// Returns `true` if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.min.is_none()
    }

    // ------------------------------------------------------------------
    // Internal recursive helpers
    // ------------------------------------------------------------------

    fn insert_internal(&mut self, mut x: u32) {
        match self.min {
            None => {
                // Empty set: store as both min and max.
                self.min = Some(x);
                self.max = Some(x);
                return;
            }
            Some(mn) => {
                if x == mn {
                    return; // already present
                }
                if x < mn {
                    // x becomes the new minimum; push old min into clusters.
                    self.min = Some(x);
                    x = mn;
                }
            }
        }
        if self.universe > 2 {
            let hi = self.high(x);
            let lo = self.low(x);
            let lower = lower_sqrt(self.universe);
            let upper = upper_sqrt(self.universe);

            let cluster = self.clusters[hi as usize].get_or_insert_with(|| {
                Box::new(VEBTree::create(lower))
            });

            if cluster.is_empty() {
                // Also insert hi into the summary.
                if let Some(ref mut summary) = self.summary {
                    summary.insert_internal(hi);
                }
            }
            cluster.insert_internal(lo);

            let _ = upper; // suppress unused warning
        }
        if let Some(mx) = self.max {
            if x > mx {
                self.max = Some(x);
            }
        } else {
            self.max = Some(x);
        }
    }

    fn delete_internal(&mut self, x: u32) {
        let (mn, mx) = match (self.min, self.max) {
            (Some(mn), Some(mx)) => (mn, mx),
            _ => return, // empty tree
        };

        if mn == mx {
            // Only one element.
            if x == mn {
                self.min = None;
                self.max = None;
            }
            return;
        }

        if self.universe <= 2 {
            // Base case: two elements, remove x.
            if x == 0 {
                self.min = Some(1);
            } else {
                self.max = Some(0);
            }
            self.min = self.max; // keep consistent
            // Re-assign correctly:
            if x == mn {
                self.min = Some(mx);
                self.max = Some(mx);
            } else if x == mx {
                self.min = Some(mn);
                self.max = Some(mn);
            }
            return;
        }

        let mut to_delete = x;

        if to_delete == mn {
            // The element to delete is the stored minimum.
            // Replace min with the actual cluster minimum.
            let first_cluster = self
                .summary
                .as_deref()
                .and_then(|s| s.minimum());
            match first_cluster {
                None => {
                    // No clusters — min was the only element.
                    self.min = Some(mx);
                    return;
                }
                Some(fc) => {
                    let cluster_min = self.clusters[fc as usize]
                        .as_deref()
                        .and_then(|c| c.minimum())
                        .unwrap_or(0);
                    to_delete = self.index(fc, cluster_min);
                    self.min = Some(to_delete);
                }
            }
        }

        let hi = self.high(to_delete);
        let lo = self.low(to_delete);

        if let Some(ref mut cluster) = self.clusters[hi as usize] {
            cluster.delete_internal(lo);
            if cluster.is_empty() {
                // Remove hi from summary.
                if let Some(ref mut summary) = self.summary {
                    summary.delete_internal(hi);
                }
                // Optionally free the cluster.
                self.clusters[hi as usize] = None;
            }
        }

        if to_delete == mx {
            // Recompute max.
            let summary_max = self.summary.as_deref().and_then(|s| s.maximum());
            match summary_max {
                None => {
                    // No more clusters; only min remains.
                    self.max = self.min;
                }
                Some(sm) => {
                    let cluster_max = self.clusters[sm as usize]
                        .as_deref()
                        .and_then(|c| c.maximum())
                        .unwrap_or(0);
                    self.max = Some(self.index(sm, cluster_max));
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Index helpers
    // ------------------------------------------------------------------

    /// High part: cluster index.
    #[inline]
    fn high(&self, x: u32) -> u32 {
        x / lower_sqrt(self.universe)
    }

    /// Low part: position within cluster.
    #[inline]
    fn low(&self, x: u32) -> u32 {
        x % lower_sqrt(self.universe)
    }

    /// Reconstruct original key from cluster index and position.
    #[inline]
    fn index(&self, hi: u32, lo: u32) -> u32 {
        hi * lower_sqrt(self.universe) + lo
    }
}

// ============================================================================
// Math helpers
// ============================================================================

/// Returns ⌈√u⌉ (upper square root) used as the number of clusters.
#[inline]
fn upper_sqrt(u: u32) -> u32 {
    let bits = u32::BITS - u.leading_zeros();
    let half = (bits + 1) / 2;
    1u32 << half
}

/// Returns ⌊√u⌋ (lower square root) used as each cluster's universe size.
#[inline]
fn lower_sqrt(u: u32) -> u32 {
    let bits = u32::BITS - u.leading_zeros();
    let half = bits / 2;
    1u32 << half
}

// ============================================================================
// Debug
// ============================================================================

impl fmt::Debug for VEBTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VEBTree")
            .field("universe", &self.universe)
            .field("min", &self.min)
            .field("max", &self.max)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tree() {
        let t = VEBTree::new(16);
        assert!(t.is_empty());
        assert_eq!(t.minimum(), None);
        assert_eq!(t.maximum(), None);
        assert!(!t.member(0));
        assert_eq!(t.successor(0), None);
        assert_eq!(t.predecessor(15), None);
    }

    #[test]
    fn single_element() {
        let mut t = VEBTree::new(16);
        t.insert(7);
        assert_eq!(t.minimum(), Some(7));
        assert_eq!(t.maximum(), Some(7));
        assert!(t.member(7));
        assert!(!t.member(0));
        assert_eq!(t.successor(7), None);
        assert_eq!(t.predecessor(7), None);
    }

    #[test]
    fn two_elements() {
        let mut t = VEBTree::new(16);
        t.insert(3);
        t.insert(11);
        assert_eq!(t.minimum(), Some(3));
        assert_eq!(t.maximum(), Some(11));
        assert_eq!(t.successor(3), Some(11));
        assert_eq!(t.predecessor(11), Some(3));
        assert_eq!(t.successor(11), None);
        assert_eq!(t.predecessor(3), None);
    }

    #[test]
    fn sequential_insert_successor_predecessor() {
        let values = [0u32, 1, 5, 10, 15, 20, 31];
        let mut t = VEBTree::new(64);
        for &v in &values {
            t.insert(v);
        }
        assert_eq!(t.minimum(), Some(0));
        assert_eq!(t.maximum(), Some(31));

        // Successor chain.
        let mut cur = t.minimum();
        let mut chain = Vec::new();
        while let Some(v) = cur {
            chain.push(v);
            cur = t.successor(v);
        }
        assert_eq!(chain, values);

        // Predecessor chain (reverse).
        let mut cur = t.maximum();
        let mut rev_chain = Vec::new();
        while let Some(v) = cur {
            rev_chain.push(v);
            cur = t.predecessor(v);
        }
        let mut expected_rev: Vec<u32> = values.iter().copied().rev().collect();
        assert_eq!(rev_chain, expected_rev);
    }

    #[test]
    fn delete_min() {
        let mut t = VEBTree::new(64);
        for v in [5u32, 10, 20, 30] {
            t.insert(v);
        }
        t.delete(5);
        assert!(!t.member(5));
        assert_eq!(t.minimum(), Some(10));
    }

    #[test]
    fn delete_max() {
        let mut t = VEBTree::new(64);
        for v in [5u32, 10, 20, 30] {
            t.insert(v);
        }
        t.delete(30);
        assert!(!t.member(30));
        assert_eq!(t.maximum(), Some(20));
    }

    #[test]
    fn delete_middle() {
        let mut t = VEBTree::new(64);
        for v in [1u32, 2, 3, 4, 5] {
            t.insert(v);
        }
        t.delete(3);
        assert!(!t.member(3));
        assert_eq!(t.successor(2), Some(4));
        assert_eq!(t.predecessor(4), Some(2));
    }

    #[test]
    fn duplicate_insert_idempotent() {
        let mut t = VEBTree::new(64);
        t.insert(42);
        t.insert(42);
        assert_eq!(t.minimum(), Some(42));
        assert_eq!(t.maximum(), Some(42));
        // Both min and max should be 42 — no duplicate stored.
        let mut chain = Vec::new();
        let mut cur = t.minimum();
        while let Some(v) = cur {
            chain.push(v);
            cur = t.successor(v);
        }
        assert_eq!(chain, vec![42u32]);
    }

    #[test]
    fn large_universe() {
        let mut t = VEBTree::new(1024);
        let vals: Vec<u32> = (0..100).map(|i| i * 10).collect();
        for &v in &vals {
            t.insert(v);
        }
        assert_eq!(t.minimum(), Some(0));
        assert_eq!(t.maximum(), Some(990));
        assert_eq!(t.successor(500), Some(510));
        assert_eq!(t.predecessor(500), Some(490));
    }

    #[test]
    fn out_of_range_ignored() {
        let mut t = VEBTree::new(16);
        t.insert(16); // out of range
        t.insert(100);
        assert!(t.is_empty());
        t.delete(16); // no-op
    }

    #[test]
    fn full_delete_sequence() {
        let mut t = VEBTree::new(64);
        let vals = [3u32, 7, 12, 19, 25, 40, 55, 63];
        for &v in &vals {
            t.insert(v);
        }
        for &v in &vals {
            t.delete(v);
        }
        assert!(t.is_empty());
        for &v in &vals {
            assert!(!t.member(v), "member({v}) should be false after full delete");
        }
    }
}
