//! Stream join operations.
//!
//! Provides hash-based and interval-based joins for streaming record pairs.
//! Hash join maintains an in-memory lookup table per side and produces
//! results as soon as a matching record arrives on either side.
//! Interval join matches records from two streams whose timestamps fall within
//! a configurable time window of each other.

use std::collections::HashMap;

/// Hash join for two typed streams.
///
/// Maintains an in-memory state table for each stream side. When a record
/// arrives on the right, all matching left records are returned immediately.
/// Left-side records are also indexed for future right-side arrivals.
pub struct HashJoin<K, L, R>
where
    K: Eq + std::hash::Hash + Clone,
    L: Clone,
    R: Clone,
{
    left_table: HashMap<K, Vec<L>>,
    right_table: HashMap<K, Vec<R>>,
}

impl<K, L, R> HashJoin<K, L, R>
where
    K: Eq + std::hash::Hash + Clone,
    L: Clone,
    R: Clone,
{
    /// Create a new, empty `HashJoin`.
    pub fn new() -> Self {
        Self {
            left_table: HashMap::new(),
            right_table: HashMap::new(),
        }
    }

    /// Index a left-side record.
    pub fn insert_left(&mut self, key: K, value: L) {
        self.left_table.entry(key).or_default().push(value);
    }

    /// Insert a right-side record and immediately return all (left, right) pairs.
    pub fn insert_right(&mut self, key: K, value: R) -> Vec<(L, R)> {
        let matches: Vec<(L, R)> = if let Some(left_records) = self.left_table.get(&key) {
            left_records
                .iter()
                .map(|l| (l.clone(), value.clone()))
                .collect()
        } else {
            Vec::new()
        };

        self.right_table.entry(key).or_default().push(value);
        matches
    }

    /// Perform a full join for a given key across both tables.
    pub fn join_key(&self, key: &K) -> Vec<(L, R)> {
        match (self.left_table.get(key), self.right_table.get(key)) {
            (Some(lefts), Some(rights)) => lefts
                .iter()
                .flat_map(|l| rights.iter().map(move |r| (l.clone(), r.clone())))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Evict left-side state for the given keys (e.g., after watermark advance).
    pub fn evict_left(&mut self, keys: &[K]) {
        for k in keys {
            self.left_table.remove(k);
        }
    }

    /// Evict right-side state for the given keys.
    pub fn evict_right(&mut self, keys: &[K]) {
        for k in keys {
            self.right_table.remove(k);
        }
    }

    /// Number of distinct keys in the left table.
    pub fn left_key_count(&self) -> usize {
        self.left_table.len()
    }

    /// Number of distinct keys in the right table.
    pub fn right_key_count(&self) -> usize {
        self.right_table.len()
    }
}

impl<K, L, R> Default for HashJoin<K, L, R>
where
    K: Eq + std::hash::Hash + Clone,
    L: Clone,
    R: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Interval join: matches records from two streams whose timestamps differ
/// by at most `window` seconds and share the same key.
///
/// Internally buffers records from both sides for `2 * window` seconds.
pub struct IntervalJoin {
    window: f64,
    left_buffer: Vec<TimedRecord>,
    right_buffer: Vec<TimedRecord>,
}

#[derive(Debug, Clone)]
struct TimedRecord {
    timestamp: f64,
    key: String,
    value: f64,
}

impl IntervalJoin {
    /// Create a new interval join that matches records within `window` seconds of each other.
    pub fn new(window: f64) -> Self {
        assert!(window >= 0.0, "window must be non-negative");
        Self {
            window,
            left_buffer: Vec::new(),
            right_buffer: Vec::new(),
        }
    }

    /// Add a left record and return matching (left_val, right_val) pairs.
    pub fn add_left(&mut self, ts: f64, key: String, val: f64) -> Vec<(f64, f64)> {
        let matches: Vec<(f64, f64)> = self
            .right_buffer
            .iter()
            .filter(|r| r.key == key && (r.timestamp - ts).abs() <= self.window)
            .map(|r| (val, r.value))
            .collect();

        self.left_buffer.push(TimedRecord {
            timestamp: ts,
            key,
            value: val,
        });
        self.evict_old(ts);
        matches
    }

    /// Add a right record and return matching (left_val, right_val) pairs.
    pub fn add_right(&mut self, ts: f64, key: String, val: f64) -> Vec<(f64, f64)> {
        let matches: Vec<(f64, f64)> = self
            .left_buffer
            .iter()
            .filter(|r| r.key == key && (r.timestamp - ts).abs() <= self.window)
            .map(|r| (r.value, val))
            .collect();

        self.right_buffer.push(TimedRecord {
            timestamp: ts,
            key,
            value: val,
        });
        self.evict_old(ts);
        matches
    }

    fn evict_old(&mut self, current_ts: f64) {
        let cutoff = current_ts - self.window * 2.0;
        self.left_buffer.retain(|r| r.timestamp >= cutoff);
        self.right_buffer.retain(|r| r.timestamp >= cutoff);
    }

    /// Number of records currently buffered from the left stream.
    pub fn left_buffer_size(&self) -> usize {
        self.left_buffer.len()
    }

    /// Number of records currently buffered from the right stream.
    pub fn right_buffer_size(&self) -> usize {
        self.right_buffer.len()
    }
}

/// Type alias: a `HashJoin` with `String` keys (most common case for stream joins).
pub type StreamJoin<L, R> = HashJoin<String, L, R>;

/// Type alias: sort-merge join is approximated by HashJoin for the streaming case.
/// A true sort-merge join requires globally sorted input, which is unusual in
/// practice; the hash-based implementation is used here.
pub type SortMergeJoin<L, R> = HashJoin<String, L, R>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_join_matches() {
        let mut join: HashJoin<String, f64, f64> = HashJoin::new();

        join.insert_left("user_1".to_string(), 100.0);
        join.insert_left("user_2".to_string(), 200.0);

        // Right record for user_1 should produce one match
        let matches = join.insert_right("user_1".to_string(), 42.0);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0], (100.0, 42.0));

        // Right record for user_3 (no left) should produce no match
        let no_matches = join.insert_right("user_3".to_string(), 99.0);
        assert!(no_matches.is_empty());

        // join_key should return cross-product
        join.insert_left("user_1".to_string(), 150.0);
        let all = join.join_key(&"user_1".to_string());
        assert_eq!(all.len(), 2); // (100,42) and (150,42)
    }

    #[test]
    fn test_hash_join_eviction() {
        let mut join: HashJoin<String, i32, i32> = HashJoin::new();
        join.insert_left("k1".to_string(), 1);
        join.insert_left("k2".to_string(), 2);
        assert_eq!(join.left_key_count(), 2);

        join.evict_left(&["k1".to_string()]);
        assert_eq!(join.left_key_count(), 1);
        assert!(join.join_key(&"k1".to_string()).is_empty());
    }

    #[test]
    fn test_interval_join_matches_within_window() {
        let mut ij = IntervalJoin::new(3.0);

        // Left at t=10, right at t=12 (within window=3)
        ij.add_left(10.0, "a".to_string(), 1.0);
        let matches = ij.add_right(12.0, "a".to_string(), 2.0);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0], (1.0, 2.0));
    }

    #[test]
    fn test_interval_join_no_match_outside_window() {
        let mut ij = IntervalJoin::new(3.0);

        ij.add_left(10.0, "a".to_string(), 1.0);
        // t=20 is 10 seconds away — outside the 3.0 window
        let matches = ij.add_right(20.0, "a".to_string(), 2.0);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_stream_join_alias() {
        let mut join: StreamJoin<u64, u64> = StreamJoin::new();
        join.insert_left("event".to_string(), 1u64);
        let matches = join.insert_right("event".to_string(), 2u64);
        assert_eq!(matches.len(), 1);
    }
}
