//! Persistent and functional data structures.
//!
//! This module provides immutable, persistent data structures that use structural
//! sharing (via `Arc`) to make "updates" efficient — each operation returns a new
//! version while reusing unchanged subtrees from the old version.
//!
//! # Structures
//!
//! - [`PersistentVec`](persistent_vec::PersistentVec) — 32-ary trie vector (Clojure-style).
//! - [`PersistentMap`](persistent_map::PersistentMap) — Hash Array Mapped Trie (HAMT).
//! - [`PersistentRBTree`](persistent_rbtree::PersistentRBTree) — Okasaki-style persistent red-black tree map.
//! - [`PersistentHashMap`](persistent_hamt::PersistentHashMap) — Full-featured HAMT with Bitmap/Full/Collision nodes.
//! - [`PersistentQueue`](persistent_queue::PersistentQueue) — Functional queue with amortized O(1) push/pop.
//! - [`SegmentTree`](segment_tree::SegmentTree) — range query / point update.
//! - [`LazySegmentTree`](segment_tree::LazySegmentTree) — range query / range update with lazy propagation.
//! - [`PersistentSegmentTree`](segment_tree::PersistentSegmentTree) — versioned segment tree.
//! - [`FenwickTree`](fenwick_tree::FenwickTree) — Binary Indexed Tree for prefix sums.
//! - [`FenwickTree2D`](fenwick_tree::FenwickTree2D) — 2-D BIT.
//! - [`OrderStatisticsTree`](fenwick_tree::OrderStatisticsTree) — rank / select via BIT.
//! - [`FingerTree`](finger_tree::FingerTree) — functional deque with O(log N) split/concat.

pub mod fenwick_tree;
pub mod finger_tree;
pub mod persistent_hamt;
pub mod persistent_map;
pub mod persistent_queue;
pub mod persistent_rbtree;
pub mod persistent_vec;
pub mod segment_tree;

pub use fenwick_tree::{FenwickTree, FenwickTree2D, OrderStatisticsTree};
pub use finger_tree::FingerTree;
pub use persistent_hamt::PersistentHashMap;
pub use persistent_map::PersistentMap;
pub use persistent_queue::PersistentQueue;
pub use persistent_rbtree::PersistentRBTree;
pub use persistent_vec::PersistentVec;
pub use segment_tree::{LazySegmentTree, PersistentSegmentTree, SegmentTree};
