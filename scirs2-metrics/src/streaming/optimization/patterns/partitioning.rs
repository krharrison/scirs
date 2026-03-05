//! Data partitioning strategies for parallel streaming processing
//!
//! This module provides mechanisms that route incoming stream elements to
//! parallel workers (partitions) for load-balanced, concurrent metric
//! computation:
//!
//! - [`HashPartitioner`]     – deterministic assignment via element key hashing
//! - [`RoundRobinPartitioner`] – cyclic distribution across partitions
//! - [`RangePartitioner`]    – assigns elements to partitions based on numeric ranges
//! - [`LoadBalancedPartitioner`] – adaptive routing that minimises queue depth
//! - [`PartitionRegistry`]   – aggregates per-partition statistics for monitoring

use crate::error::{MetricsError, Result};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};

// ── PartitionAssignment ───────────────────────────────────────────────────────

/// The result of routing an element through a partitioner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartitionAssignment {
    /// Zero-based index of the partition the element was routed to.
    pub partition_id: usize,
}

// ── PartitionStats ────────────────────────────────────────────────────────────

/// Per-partition cumulative statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PartitionStats {
    /// Total elements routed to this partition.
    pub element_count: u64,
    /// Number of times this partition was selected as the least-loaded one
    /// (relevant for load-balanced routing).
    pub load_balance_selections: u64,
    /// Reported current queue depth (updated externally via
    /// [`LoadBalancedPartitioner::report_queue_depth`]).
    pub current_queue_depth: usize,
}

// ── HashPartitioner ───────────────────────────────────────────────────────────

/// Routes elements to partitions by hashing a key derived from each element.
///
/// The same key always maps to the same partition, making this suitable for
/// workloads that require locality (e.g. join keys, user-ID-based grouping).
#[derive(Debug, Clone)]
pub struct HashPartitioner {
    num_partitions: usize,
    stats: Vec<PartitionStats>,
}

impl HashPartitioner {
    /// Create a new hash partitioner.
    ///
    /// # Arguments
    /// * `num_partitions` – number of downstream partitions (must be >= 1)
    pub fn new(num_partitions: usize) -> Result<Self> {
        if num_partitions == 0 {
            return Err(MetricsError::InvalidInput(
                "HashPartitioner requires at least 1 partition".to_string(),
            ));
        }
        Ok(Self {
            num_partitions,
            stats: vec![PartitionStats::default(); num_partitions],
        })
    }

    /// Route `element` to a partition using its [`Hash`] impl.
    pub fn assign<K: Hash>(&mut self, key: &K) -> PartitionAssignment {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let partition_id = (hasher.finish() as usize) % self.num_partitions;
        self.stats[partition_id].element_count += 1;
        PartitionAssignment { partition_id }
    }

    /// Statistics for each partition.
    #[inline]
    pub fn stats(&self) -> &[PartitionStats] {
        &self.stats
    }

    /// Number of configured partitions.
    #[inline]
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }
}

// ── RoundRobinPartitioner ─────────────────────────────────────────────────────

/// Distributes elements evenly across partitions in a cyclic order.
///
/// No key is required; each call to [`assign`](RoundRobinPartitioner::assign)
/// advances the cursor by one.
#[derive(Debug, Clone)]
pub struct RoundRobinPartitioner {
    num_partitions: usize,
    cursor: usize,
    stats: Vec<PartitionStats>,
}

impl RoundRobinPartitioner {
    /// Create a new round-robin partitioner.
    pub fn new(num_partitions: usize) -> Result<Self> {
        if num_partitions == 0 {
            return Err(MetricsError::InvalidInput(
                "RoundRobinPartitioner requires at least 1 partition".to_string(),
            ));
        }
        Ok(Self {
            num_partitions,
            cursor: 0,
            stats: vec![PartitionStats::default(); num_partitions],
        })
    }

    /// Assign the next element in round-robin fashion.
    pub fn assign(&mut self) -> PartitionAssignment {
        let partition_id = self.cursor;
        self.cursor = (self.cursor + 1) % self.num_partitions;
        self.stats[partition_id].element_count += 1;
        PartitionAssignment { partition_id }
    }

    /// Statistics for each partition.
    #[inline]
    pub fn stats(&self) -> &[PartitionStats] {
        &self.stats
    }

    /// Number of configured partitions.
    #[inline]
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }
}

// ── RangePartitioner ─────────────────────────────────────────────────────────

/// Assigns numeric values to partitions based on ordered ranges.
///
/// Ranges are defined by `num_partitions - 1` split points dividing the domain
/// `[domain_min, domain_max]` into equal-width buckets.  Values outside the
/// domain are clamped to the first or last partition.
#[derive(Debug, Clone)]
pub struct RangePartitioner<F: Float + std::fmt::Debug + Copy + PartialOrd> {
    split_points: Vec<F>,
    num_partitions: usize,
    stats: Vec<PartitionStats>,
}

impl<F: Float + std::fmt::Debug + Copy + PartialOrd> RangePartitioner<F> {
    /// Create a new range partitioner with automatically computed equal-width
    /// split points.
    ///
    /// # Arguments
    /// * `num_partitions` – number of downstream partitions (must be >= 1)
    /// * `domain_min`     – inclusive lower bound of the value domain
    /// * `domain_max`     – exclusive upper bound of the value domain
    ///
    /// Returns `Err` if `num_partitions` is 0 or `domain_max <= domain_min`.
    pub fn new(num_partitions: usize, domain_min: F, domain_max: F) -> Result<Self> {
        if num_partitions == 0 {
            return Err(MetricsError::InvalidInput(
                "RangePartitioner requires at least 1 partition".to_string(),
            ));
        }
        if domain_max <= domain_min {
            return Err(MetricsError::InvalidInput(
                "RangePartitioner domain_max must be > domain_min".to_string(),
            ));
        }
        let range = domain_max - domain_min;
        let n_f = F::from(num_partitions).expect("usize fits in F");
        let step = range / n_f;
        // n-1 split points for n partitions
        let split_points: Vec<F> = (1..num_partitions)
            .map(|i| domain_min + step * F::from(i).expect("usize fits in F"))
            .collect();

        Ok(Self {
            split_points,
            num_partitions,
            stats: vec![PartitionStats::default(); num_partitions],
        })
    }

    /// Create a partitioner with explicitly provided split points.
    ///
    /// `split_points` must be sorted in strictly ascending order and its length
    /// must equal `num_partitions - 1`.
    pub fn from_split_points(split_points: Vec<F>) -> Result<Self> {
        let num_partitions = split_points.len() + 1;
        // Validate ordering
        for window in split_points.windows(2) {
            if window[0] >= window[1] {
                return Err(MetricsError::InvalidInput(
                    "RangePartitioner split_points must be strictly ascending".to_string(),
                ));
            }
        }
        Ok(Self {
            split_points,
            num_partitions,
            stats: vec![PartitionStats::default(); num_partitions],
        })
    }

    /// Route `value` to the appropriate partition.
    pub fn assign(&mut self, value: F) -> PartitionAssignment {
        let partition_id = self
            .split_points
            .iter()
            .position(|&sp| value < sp)
            .unwrap_or(self.num_partitions - 1);

        self.stats[partition_id].element_count += 1;
        PartitionAssignment { partition_id }
    }

    /// Statistics for each partition.
    #[inline]
    pub fn stats(&self) -> &[PartitionStats] {
        &self.stats
    }

    /// Number of configured partitions.
    #[inline]
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }
}

// ── LoadBalancedPartitioner ───────────────────────────────────────────────────

/// Routes elements to the partition with the smallest current queue depth.
///
/// Workers call [`report_queue_depth`](LoadBalancedPartitioner::report_queue_depth)
/// to update their reported load.  New elements are then routed to the least
/// loaded partition.  Ties are broken by partition index (lowest wins).
#[derive(Debug, Clone)]
pub struct LoadBalancedPartitioner {
    num_partitions: usize,
    queue_depths: Vec<usize>,
    stats: Vec<PartitionStats>,
}

impl LoadBalancedPartitioner {
    /// Create a new load-balanced partitioner.
    pub fn new(num_partitions: usize) -> Result<Self> {
        if num_partitions == 0 {
            return Err(MetricsError::InvalidInput(
                "LoadBalancedPartitioner requires at least 1 partition".to_string(),
            ));
        }
        Ok(Self {
            num_partitions,
            queue_depths: vec![0; num_partitions],
            stats: vec![PartitionStats::default(); num_partitions],
        })
    }

    /// Route the next element to the least-loaded partition.
    pub fn assign(&mut self) -> PartitionAssignment {
        // Find partition with minimum queue depth; ties broken by lowest index
        let partition_id = self
            .queue_depths
            .iter()
            .enumerate()
            .min_by_key(|&(_, &depth)| depth)
            .map(|(id, _)| id)
            .unwrap_or(0);

        self.stats[partition_id].element_count += 1;
        self.stats[partition_id].load_balance_selections += 1;
        PartitionAssignment { partition_id }
    }

    /// Update the reported queue depth for a partition.
    ///
    /// Workers should call this after dequeuing elements so that future routing
    /// decisions reflect the current load distribution.
    pub fn report_queue_depth(&mut self, partition_id: usize, depth: usize) -> Result<()> {
        if partition_id >= self.num_partitions {
            return Err(MetricsError::InvalidInput(format!(
                "partition_id {partition_id} out of range (num_partitions={})",
                self.num_partitions
            )));
        }
        self.queue_depths[partition_id] = depth;
        self.stats[partition_id].current_queue_depth = depth;
        Ok(())
    }

    /// Current reported queue depths (indexed by partition ID).
    #[inline]
    pub fn queue_depths(&self) -> &[usize] {
        &self.queue_depths
    }

    /// Statistics for each partition.
    #[inline]
    pub fn stats(&self) -> &[PartitionStats] {
        &self.stats
    }

    /// Number of configured partitions.
    #[inline]
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }
}

// ── PartitionRegistry ─────────────────────────────────────────────────────────

/// A registry that aggregates statistics across multiple named partitioners for
/// monitoring and observability.
#[derive(Debug, Default)]
pub struct PartitionRegistry {
    partitioner_stats: HashMap<String, Vec<PartitionStats>>,
}

impl PartitionRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            partitioner_stats: HashMap::new(),
        }
    }

    /// Register a snapshot of per-partition statistics under `name`.
    pub fn register(&mut self, name: impl Into<String>, stats: Vec<PartitionStats>) {
        self.partitioner_stats.insert(name.into(), stats);
    }

    /// Get the statistics for a named partitioner.
    pub fn get(&self, name: &str) -> Option<&[PartitionStats]> {
        self.partitioner_stats.get(name).map(|v| v.as_slice())
    }

    /// Compute an imbalance score for a named partitioner.
    ///
    /// The score is the ratio `max_count / mean_count` (minus 1), where a
    /// score of 0.0 means perfect balance.  Returns `None` when the partitioner
    /// is unknown or has no elements.
    pub fn imbalance_score(&self, name: &str) -> Option<f64> {
        let stats = self.partitioner_stats.get(name)?;
        if stats.is_empty() {
            return None;
        }
        let total: u64 = stats.iter().map(|s| s.element_count).sum();
        if total == 0 {
            return Some(0.0);
        }
        let n = stats.len() as f64;
        let mean = total as f64 / n;
        let max = stats.iter().map(|s| s.element_count).max().unwrap_or(0) as f64;
        Some((max / mean.max(f64::EPSILON)) - 1.0)
    }

    /// Names of all registered partitioners.
    pub fn registered_names(&self) -> Vec<&str> {
        self.partitioner_stats.keys().map(|s| s.as_str()).collect()
    }
}

// ── PartitionedStream ─────────────────────────────────────────────────────────

/// A lightweight helper that maintains one logical queue per partition and
/// routes elements to those queues via a [`RoundRobinPartitioner`].
///
/// Useful for testing partitioning logic or as a simple in-process fan-out.
#[derive(Debug)]
pub struct PartitionedStream<T> {
    queues: Vec<VecDeque<T>>,
    partitioner: RoundRobinPartitioner,
}

impl<T: Clone> PartitionedStream<T> {
    /// Create a new partitioned stream with `num_partitions` queues.
    pub fn new(num_partitions: usize) -> Result<Self> {
        let partitioner = RoundRobinPartitioner::new(num_partitions)?;
        Ok(Self {
            queues: vec![VecDeque::new(); num_partitions],
            partitioner,
        })
    }

    /// Route `value` to the next partition's queue.
    pub fn push(&mut self, value: T) {
        let assignment = self.partitioner.assign();
        self.queues[assignment.partition_id].push_back(value);
    }

    /// Pop an element from the specified partition's queue.
    pub fn pop(&mut self, partition_id: usize) -> Result<Option<T>> {
        if partition_id >= self.queues.len() {
            return Err(MetricsError::InvalidInput(format!(
                "partition_id {partition_id} out of range"
            )));
        }
        Ok(self.queues[partition_id].pop_front())
    }

    /// Number of elements queued in a specific partition.
    pub fn queue_depth(&self, partition_id: usize) -> Result<usize> {
        if partition_id >= self.queues.len() {
            return Err(MetricsError::InvalidInput(format!(
                "partition_id {partition_id} out of range"
            )));
        }
        Ok(self.queues[partition_id].len())
    }

    /// Number of partitions.
    #[inline]
    pub fn num_partitions(&self) -> usize {
        self.queues.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_partitioner_deterministic() {
        let mut p = HashPartitioner::new(4).expect("valid");
        let a1 = p.assign(&"hello");
        let a2 = p.assign(&"hello");
        assert_eq!(a1.partition_id, a2.partition_id, "same key same partition");
        let a3 = p.assign(&"world");
        // Different key may land in a different partition (not guaranteed, but
        // we just check it is in range)
        assert!(a3.partition_id < 4);
    }

    #[test]
    fn hash_partitioner_stats() {
        let mut p = HashPartitioner::new(3).expect("valid");
        for i in 0_u64..30 {
            p.assign(&i);
        }
        let total: u64 = p.stats().iter().map(|s| s.element_count).sum();
        assert_eq!(total, 30);
    }

    #[test]
    fn round_robin_even_distribution() {
        let mut p = RoundRobinPartitioner::new(3).expect("valid");
        let assignments: Vec<usize> = (0..9).map(|_| p.assign().partition_id).collect();
        assert_eq!(assignments, vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn round_robin_single_partition() {
        let mut p = RoundRobinPartitioner::new(1).expect("valid");
        for _ in 0..5 {
            assert_eq!(p.assign().partition_id, 0);
        }
    }

    #[test]
    fn range_partitioner_equal_width() {
        // 3 partitions over [0, 9): splits at 3, 6
        let mut p = RangePartitioner::<f64>::new(3, 0.0, 9.0).expect("valid");
        assert_eq!(p.assign(1.0).partition_id, 0);
        assert_eq!(p.assign(3.0).partition_id, 1);
        assert_eq!(p.assign(6.0).partition_id, 2);
        assert_eq!(p.assign(8.9).partition_id, 2);
    }

    #[test]
    fn range_partitioner_clamping() {
        let mut p = RangePartitioner::<f64>::new(3, 0.0, 9.0).expect("valid");
        // Value below domain → partition 0
        assert_eq!(p.assign(-5.0).partition_id, 0);
        // Value above domain → last partition
        assert_eq!(p.assign(100.0).partition_id, 2);
    }

    #[test]
    fn range_partitioner_invalid_domain() {
        assert!(RangePartitioner::<f64>::new(3, 5.0, 5.0).is_err());
        assert!(RangePartitioner::<f64>::new(3, 10.0, 5.0).is_err());
    }

    #[test]
    fn range_partitioner_from_split_points() {
        let mut p = RangePartitioner::<f64>::from_split_points(vec![10.0, 20.0]).expect("valid");
        assert_eq!(p.assign(5.0).partition_id, 0);
        assert_eq!(p.assign(15.0).partition_id, 1);
        assert_eq!(p.assign(25.0).partition_id, 2);
    }

    #[test]
    fn load_balanced_routes_to_least_loaded() {
        let mut p = LoadBalancedPartitioner::new(3).expect("valid");
        // Initially all at depth 0 → routes to partition 0
        assert_eq!(p.assign().partition_id, 0);
        // Report partition 0 as loaded
        p.report_queue_depth(0, 10).expect("valid id");
        // Next should go to partition 1 (or 2, both at 0)
        let next = p.assign().partition_id;
        assert!(next == 1 || next == 2);
    }

    #[test]
    fn load_balanced_invalid_partition_id() {
        let mut p = LoadBalancedPartitioner::new(2).expect("valid");
        assert!(p.report_queue_depth(5, 10).is_err());
    }

    #[test]
    fn partition_registry_imbalance() {
        let mut registry = PartitionRegistry::new();
        let stats = vec![
            PartitionStats { element_count: 100, ..Default::default() },
            PartitionStats { element_count: 100, ..Default::default() },
            PartitionStats { element_count: 100, ..Default::default() },
        ];
        registry.register("rr", stats);
        let score = registry.imbalance_score("rr").expect("exists");
        // Perfectly balanced → score should be 0
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn partitioned_stream_fan_out() {
        let mut stream: PartitionedStream<i32> = PartitionedStream::new(3).expect("valid");
        for i in 0..9 {
            stream.push(i);
        }
        // Each partition should have received 3 elements
        for pid in 0..3 {
            assert_eq!(stream.queue_depth(pid).expect("valid"), 3);
        }
    }

    #[test]
    fn partitioned_stream_pop() {
        let mut stream: PartitionedStream<i32> = PartitionedStream::new(2).expect("valid");
        stream.push(10);
        stream.push(20);
        let v0 = stream.pop(0).expect("ok").expect("some");
        let v1 = stream.pop(1).expect("ok").expect("some");
        assert_eq!(v0, 10);
        assert_eq!(v1, 20);
    }
}
