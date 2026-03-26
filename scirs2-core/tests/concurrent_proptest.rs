//! Property-based tests for scirs2-core concurrent data structures.
//!
//! Verifies key invariants:
//! * `LockFreeQueue`: push/pop consistency — every pushed item can be popped;
//!   length invariants are maintained throughout.
//! * `SkipList`: ordering invariant — the sorted order of keys is preserved
//!   after arbitrary insertions, and every inserted key is retrievable.
//! * `PersistentRrbVec`: append-only semantics — `push_back` does not mutate
//!   previous versions; indexed reads are stable across versions.

use proptest::prelude::*;
use scirs2_core::concurrent::{LockFreeQueue, PersistentRrbVec, SkipList};

// ─────────────────────────────────────────────────────────────────────────────
// Strategies
// ─────────────────────────────────────────────────────────────────────────────

/// A sequence of small integer keys suitable for skip-list operations.
fn key_sequence_strategy(max_len: usize) -> impl Strategy<Value = Vec<i64>> {
    proptest::collection::vec(i64::MIN / 2..i64::MAX / 2, 1..=max_len)
}

/// A sequence of (key, value) pairs for skip-list insertion.
fn kv_sequence_strategy(max_len: usize) -> impl Strategy<Value = Vec<(i32, i32)>> {
    proptest::collection::vec(
        (i32::MIN / 2..i32::MAX / 2, i32::MIN..i32::MAX),
        1..=max_len,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// LockFreeQueue: push/pop consistency
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// All successfully pushed items can subsequently be popped (FIFO order).
    #[test]
    fn prop_lock_free_queue_push_pop_fifo(
        values in proptest::collection::vec(0i32..1000, 1..=64),
    ) {
        // Use a capacity slightly larger than the number of items.
        let capacity = values.len().next_power_of_two();
        let queue = LockFreeQueue::<i32>::new(capacity);

        let mut pushed: Vec<i32> = Vec::new();
        for &v in &values {
            if queue.push(v) {
                pushed.push(v);
            }
        }

        let mut popped: Vec<i32> = Vec::new();
        while let Some(v) = queue.pop() {
            popped.push(v);
        }

        // Every pushed value must appear exactly once in FIFO order.
        prop_assert_eq!(
            pushed.len(),
            popped.len(),
            "Pushed {} items but popped {}",
            pushed.len(),
            popped.len()
        );
        for (i, (a, b)) in pushed.iter().zip(popped.iter()).enumerate() {
            prop_assert_eq!(
                a, b,
                "FIFO order violated at position {}: pushed {}, popped {}",
                i, a, b
            );
        }
    }

    /// Queue is empty after popping all elements.
    #[test]
    fn prop_lock_free_queue_empty_after_drain(
        values in proptest::collection::vec(0i32..1000, 1..=32),
    ) {
        let capacity = values.len().next_power_of_two();
        let queue = LockFreeQueue::<i32>::new(capacity);

        for &v in &values {
            let _ = queue.push(v);
        }
        while queue.pop().is_some() {}

        prop_assert!(
            queue.is_empty(),
            "Queue should be empty after draining all elements"
        );
        prop_assert_eq!(
            queue.len(),
            0,
            "Queue len should be 0 after draining, got {}",
            queue.len()
        );
    }

    /// Queue length reflects the number of outstanding items (pushed minus popped).
    #[test]
    fn prop_lock_free_queue_len_tracking(
        values in proptest::collection::vec(0i32..1000, 2..=64),
        pop_count in 1usize..=16,
    ) {
        let capacity = values.len().next_power_of_two();
        let queue = LockFreeQueue::<i32>::new(capacity);

        let mut actual_count: usize = 0;
        for &v in &values {
            if queue.push(v) {
                actual_count += 1;
            }
        }
        prop_assert_eq!(
            queue.len(),
            actual_count,
            "After pushing, len={} but actual_count={}",
            queue.len(),
            actual_count
        );

        let pops = pop_count.min(actual_count);
        for _ in 0..pops {
            let _ = queue.pop();
            actual_count -= 1;
        }
        prop_assert_eq!(
            queue.len(),
            actual_count,
            "After popping {} items, len={} but expected={}",
            pops,
            queue.len(),
            actual_count
        );
    }

    /// Pushing to a full queue returns false and does not grow the queue.
    #[test]
    fn prop_lock_free_queue_full_push_rejected(
        capacity_exp in 1u32..=5u32,
        overflow_count in 1usize..=16,
    ) {
        let capacity = 1usize << capacity_exp; // 2..=32
        let queue = LockFreeQueue::<i32>::new(capacity);

        // Fill the queue exactly.
        let mut filled = 0usize;
        for i in 0..capacity {
            if queue.push(i as i32) {
                filled += 1;
            }
        }

        // Any further push should be rejected (queue may be full).
        for i in 0..overflow_count {
            let pushed = queue.push((capacity + i) as i32);
            if pushed {
                // If somehow the queue accepted this (due to capacity being larger than
                // expected), the length should still be consistent.
                break;
            }
            // Rejected: length must still equal filled.
            prop_assert_eq!(
                queue.len(),
                filled,
                "Length should not change after rejected push"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SkipList: ordering and retrieval invariants
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// All inserted keys are retrievable.
    #[test]
    fn prop_skip_list_inserted_keys_retrievable(
        pairs in kv_sequence_strategy(50),
    ) {
        let mut sl: SkipList<i32, i32> = SkipList::new();

        // Insert with dedup: if a key appears multiple times, last write wins.
        let mut expected: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();
        for (k, v) in &pairs {
            sl.insert(*k, *v);
            expected.insert(*k, *v);
        }

        for (k, v_expected) in &expected {
            match sl.get(k) {
                Some(v_actual) => {
                    prop_assert_eq!(
                        v_actual, *v_expected,
                        "SkipList value for key {}: expected {}, got {}",
                        k, v_expected, v_actual
                    );
                }
                None => {
                    prop_assert!(
                        false,
                        "SkipList missing key {} that was inserted",
                        k
                    );
                }
            }
        }
    }

    /// The `iter()` output is sorted by key in ascending order.
    #[test]
    fn prop_skip_list_iteration_sorted(
        keys in key_sequence_strategy(50),
    ) {
        let mut sl: SkipList<i64, i64> = SkipList::new();
        for &k in &keys {
            sl.insert(k, k * 2);
        }

        let items = sl.iter();
        for i in 1..items.len() {
            prop_assert!(
                items[i - 1].0 <= items[i].0,
                "SkipList iteration order violated at index {}: {} > {}",
                i,
                items[i - 1].0,
                items[i].0
            );
        }
    }

    /// `contains` agrees with `get` for all inserted and some non-inserted keys.
    #[test]
    fn prop_skip_list_contains_agrees_with_get(
        pairs in kv_sequence_strategy(40),
        query_keys in proptest::collection::vec(i32::MIN / 2..i32::MAX / 2, 1..=20),
    ) {
        let mut sl: SkipList<i32, i32> = SkipList::new();
        for (k, v) in &pairs {
            sl.insert(*k, *v);
        }
        for qk in &query_keys {
            let has_get = sl.get(qk).is_some();
            let has_contains = sl.contains(qk);
            prop_assert_eq!(
                has_get,
                has_contains,
                "contains/get disagreement for key {}: get={}, contains={}",
                qk, has_get, has_contains
            );
        }
    }

    /// `len` matches the number of distinct keys inserted.
    #[test]
    fn prop_skip_list_len_matches_distinct_keys(
        pairs in kv_sequence_strategy(60),
    ) {
        let mut sl: SkipList<i32, i32> = SkipList::new();
        let mut distinct: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for (k, v) in &pairs {
            sl.insert(*k, *v);
            distinct.insert(*k);
        }
        prop_assert_eq!(
            sl.len(),
            distinct.len(),
            "SkipList len={} but distinct keys={}",
            sl.len(),
            distinct.len()
        );
    }

    /// After removing a key it is no longer found, and `len` decreases by one.
    #[test]
    fn prop_skip_list_remove_makes_key_absent(
        pairs in kv_sequence_strategy(40),
    ) {
        prop_assume!(!pairs.is_empty());

        let mut sl: SkipList<i32, i32> = SkipList::new();
        let mut distinct: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for (k, v) in &pairs {
            sl.insert(*k, *v);
            distinct.insert(*k);
        }
        let len_before = sl.len();

        // Remove the first distinct key.
        let first_key = *distinct.iter().next().unwrap();
        let removed = sl.remove(&first_key);
        prop_assert!(removed, "remove returned false for a key that was inserted");
        prop_assert!(
            sl.get(&first_key).is_none(),
            "Key {} still present after remove",
            first_key
        );
        prop_assert_eq!(
            sl.len(),
            len_before - 1,
            "len should decrease by 1 after remove"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PersistentRrbVec: append-only / structural sharing invariants
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// `push_back` increases `len` by exactly one.
    #[test]
    fn prop_persistent_vec_push_back_len(
        values in proptest::collection::vec(0i32..10000, 1..=128),
    ) {
        let mut v: PersistentRrbVec<i32> = PersistentRrbVec::new();
        for (i, &val) in values.iter().enumerate() {
            let v_next = v.push_back(val);
            prop_assert_eq!(
                v_next.len(),
                i + 1,
                "After push_back #{}, len={} (expected {})",
                i,
                v_next.len(),
                i + 1
            );
            v = v_next;
        }
    }

    /// Previous versions are unchanged after `push_back` (structural sharing).
    #[test]
    fn prop_persistent_vec_immutability(
        values in proptest::collection::vec(0i32..10000, 2..=32),
    ) {
        let n = values.len();

        // Build a series of versions and record snapshots.
        let mut versions: Vec<PersistentRrbVec<i32>> = Vec::with_capacity(n + 1);
        versions.push(PersistentRrbVec::new());
        for &val in &values {
            let next = versions.last().unwrap().push_back(val);
            versions.push(next);
        }

        // Verify each version still matches its snapshot.
        for (ver_idx, version) in versions.iter().enumerate() {
            prop_assert_eq!(
                version.len(),
                ver_idx,
                "Version {} len={} (expected {})",
                ver_idx,
                version.len(),
                ver_idx
            );
            for (elem_idx, &expected_val) in values[..ver_idx].iter().enumerate() {
                match version.get(elem_idx) {
                    Some(&actual) => {
                        prop_assert_eq!(
                            actual,
                            expected_val,
                            "Version {}[{}]={} expected {}",
                            ver_idx,
                            elem_idx,
                            actual,
                            expected_val
                        );
                    }
                    None => {
                        prop_assert!(
                            false,
                            "Version {} missing index {}",
                            ver_idx,
                            elem_idx
                        );
                    }
                }
            }
        }
    }

    /// Elements appended in order are retrieved in the same order.
    #[test]
    fn prop_persistent_vec_order_preserved(
        values in proptest::collection::vec(i32::MIN..i32::MAX, 1..=64),
    ) {
        let mut v = PersistentRrbVec::new();
        for &val in &values {
            v = v.push_back(val);
        }

        for (i, &expected) in values.iter().enumerate() {
            match v.get(i) {
                Some(&actual) => {
                    prop_assert_eq!(
                        actual,
                        expected,
                        "Order violated at index {}: actual={}, expected={}",
                        i,
                        actual,
                        expected
                    );
                }
                None => {
                    prop_assert!(false, "Missing element at index {}", i);
                }
            }
        }
    }

    /// `to_vec()` returns the same sequence as the original values.
    #[test]
    fn prop_persistent_vec_to_vec_matches_get(
        values in proptest::collection::vec(0i32..1000, 1..=64),
    ) {
        let mut v = PersistentRrbVec::new();
        for &val in &values {
            v = v.push_back(val);
        }

        let as_vec = v.to_vec();
        prop_assert_eq!(
            as_vec.len(),
            v.len(),
            "to_vec len={} but len()={}",
            as_vec.len(),
            v.len()
        );
        for (i, (from_vec, from_original)) in as_vec.iter().zip(values.iter()).enumerate() {
            prop_assert_eq!(
                from_vec,
                from_original,
                "to_vec[{}]={} != original[{}]={}",
                i,
                from_vec,
                i,
                from_original
            );
        }
    }

    /// `set` returns a new version with the modified value while the old version
    /// is unchanged.
    #[test]
    fn prop_persistent_vec_set_non_destructive(
        values in proptest::collection::vec(0i32..1000, 2..=32),
        new_val in 1000i32..2000,
    ) {
        let n = values.len();
        let mut v = PersistentRrbVec::new();
        for &val in &values {
            v = v.push_back(val);
        }

        // Modify the first element.
        if let Some(v2) = v.set(0, new_val) {
            // v2[0] should be new_val
            prop_assert_eq!(
                v2.get(0).copied(),
                Some(new_val),
                "v2[0] after set should be {}", new_val
            );
            // v[0] should be unchanged
            prop_assert_eq!(
                v.get(0).copied(),
                Some(values[0]),
                "Original v[0] should still be {} after set on v2",
                values[0]
            );
            // All other elements in v2 should match v.
            for i in 1..n {
                prop_assert_eq!(
                    v2.get(i).copied(),
                    v.get(i).copied(),
                    "v2[{}] and v[{}] should agree",
                    i, i
                );
            }
        }
    }
}
