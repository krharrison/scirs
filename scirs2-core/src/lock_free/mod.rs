//! Lock-free and fine-grained concurrent data structures.
//!
//! This module provides high-performance concurrent data structures designed
//! for use in multi-threaded scientific computing workloads:
//!
//! - [`ConcurrentSkipList`]: A concurrent skip list with O(log n) expected operations
//!   using per-node fine-grained locking and an LCG-based level promotion scheme.
//! - [`ConcurrentBTree`]: A B-link tree with crab-locking for concurrent insert,
//!   lookup, delete, and range-scan operations.

pub mod btree_concurrent;
pub mod skiplist;

pub use btree_concurrent::ConcurrentBTree;
pub use skiplist::ConcurrentSkipList;
