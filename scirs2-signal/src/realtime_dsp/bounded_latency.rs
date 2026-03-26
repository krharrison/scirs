//! Bounded-latency guarantees: latency analysis, overrun detection, and
//! earliest-deadline-first block scheduling.

use crate::realtime_dsp::types::SignalBlock;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

// ── LatencyGuarantee ──────────────────────────────────────────────────────────

/// A statistical latency guarantee for a real-time processing component.
#[derive(Debug, Clone, Default)]
pub struct LatencyGuarantee {
    /// Hard worst-case bound, in samples.
    pub worst_case_samples: usize,
    /// Typical (median) latency, in samples.
    pub typical_samples: usize,
    /// Peak-to-peak jitter = max − min over the observation window, in samples.
    pub jitter_samples: usize,
}

impl LatencyGuarantee {
    /// Create a guarantee from raw observations.
    pub fn from_observations(
        worst_case_samples: usize,
        typical_samples: usize,
        jitter_samples: usize,
    ) -> Self {
        Self {
            worst_case_samples,
            typical_samples,
            jitter_samples,
        }
    }
}

// ── LatencyAnalyzer ───────────────────────────────────────────────────────────

/// Tracks per-block latency observations and computes statistical measures.
///
/// Stores up to `window_size` recent observations in a circular buffer.
pub struct LatencyAnalyzer {
    window_size: usize,
    history: Vec<usize>,
    write_ptr: usize,
    count: usize,
    /// Running maximum over all blocks ever seen (not just the window).
    global_max: usize,
    /// Running minimum over all blocks ever seen.
    global_min: usize,
}

impl LatencyAnalyzer {
    /// Create an analyzer with the given history window.
    pub fn new(window_size: usize) -> Self {
        let window_size = window_size.max(1);
        Self {
            window_size,
            history: vec![0; window_size],
            write_ptr: 0,
            count: 0,
            global_max: 0,
            global_min: usize::MAX,
        }
    }

    /// Record one latency observation (in samples).
    pub fn record(&mut self, actual_latency_samples: usize) {
        self.history[self.write_ptr] = actual_latency_samples;
        self.write_ptr = (self.write_ptr + 1) % self.window_size;
        self.count += 1;
        if actual_latency_samples > self.global_max {
            self.global_max = actual_latency_samples;
        }
        if actual_latency_samples < self.global_min {
            self.global_min = actual_latency_samples;
        }
    }

    /// Worst-case latency over all blocks ever observed.
    pub fn worst_case(&self) -> usize {
        self.global_max
    }

    /// Return the 99th-percentile latency of the current history window.
    pub fn percentile_99(&self) -> usize {
        let valid = self.valid_observations();
        if valid.is_empty() {
            return 0;
        }
        let mut sorted = valid.to_vec();
        sorted.sort_unstable();
        let idx = ((sorted.len() as f64 * 0.99).ceil() as usize).saturating_sub(1);
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Peak-to-peak jitter = max − min over the current history window.
    pub fn jitter(&self) -> usize {
        let valid = self.valid_observations();
        if valid.len() < 2 {
            return 0;
        }
        let min = *valid.iter().min().unwrap_or(&0);
        let max = *valid.iter().max().unwrap_or(&0);
        max.saturating_sub(min)
    }

    /// Build the full [`LatencyGuarantee`] from current statistics.
    pub fn guarantee(&self) -> LatencyGuarantee {
        LatencyGuarantee {
            worst_case_samples: self.worst_case(),
            typical_samples: self.median(),
            jitter_samples: self.jitter(),
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    fn valid_observations(&self) -> &[usize] {
        let n = self.count.min(self.window_size);
        &self.history[..n]
    }

    fn median(&self) -> usize {
        let valid = self.valid_observations();
        if valid.is_empty() {
            return 0;
        }
        let mut sorted = valid.to_vec();
        sorted.sort_unstable();
        sorted[sorted.len() / 2]
    }
}

// ── OverrunDetector ───────────────────────────────────────────────────────────

/// Detects when the actual processing time exceeds a hard time budget.
#[derive(Debug, Default)]
pub struct OverrunDetector {
    overrun_count: u64,
    consecutive_overruns: u64,
    max_consecutive: u64,
}

impl OverrunDetector {
    /// Create a new detector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check whether `processing_time_ns` exceeds `budget_ns`.
    ///
    /// Returns `true` (overrun detected) and increments the internal counter
    /// if so.
    pub fn check(&mut self, processing_time_ns: u64, budget_ns: u64) -> bool {
        let overrun = processing_time_ns > budget_ns;
        if overrun {
            self.overrun_count += 1;
            self.consecutive_overruns += 1;
            if self.consecutive_overruns > self.max_consecutive {
                self.max_consecutive = self.consecutive_overruns;
            }
        } else {
            self.consecutive_overruns = 0;
        }
        overrun
    }

    /// Total number of overruns detected so far.
    pub fn overrun_count(&self) -> u64 {
        self.overrun_count
    }

    /// Longest consecutive run of overruns.
    pub fn max_consecutive_overruns(&self) -> u64 {
        self.max_consecutive
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ── RealtimeScheduler ─────────────────────────────────────────────────────────

/// Pending-block entry held inside the scheduler.
#[derive(Debug)]
struct SchedulerEntry {
    /// Block identifier (also determines priority: lower = earlier deadline).
    block_id: u64,
    block: SignalBlock,
}

impl PartialEq for SchedulerEntry {
    fn eq(&self, other: &Self) -> bool {
        self.block_id == other.block_id
    }
}
impl Eq for SchedulerEntry {}

impl PartialOrd for SchedulerEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SchedulerEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // We want a min-heap (earliest deadline = lowest block_id first).
        // Wrap with Reverse via explicit ordering.
        other.block_id.cmp(&self.block_id)
    }
}

/// Earliest-Deadline-First (EDF) scheduler for pending signal blocks.
///
/// Blocks are queued with their `block_id` used as the deadline proxy; the
/// block with the smallest id is always returned first.  Late blocks (whose
/// age exceeds `max_age_blocks`) can be purged with `drop_if_late`.
pub struct RealtimeScheduler {
    queue: BinaryHeap<SchedulerEntry>,
    next_expected_id: u64,
}

impl RealtimeScheduler {
    /// Create a new scheduler.
    pub fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
            next_expected_id: 0,
        }
    }

    /// Enqueue a block.
    pub fn submit(&mut self, block: SignalBlock) {
        let block_id = block.block_id;
        self.queue.push(SchedulerEntry { block_id, block });
    }

    /// Dequeue and return the block with the earliest deadline (lowest
    /// `block_id`), if any.
    pub fn next_due(&mut self) -> Option<SignalBlock> {
        self.queue.pop().map(|e| {
            self.next_expected_id = e.block_id + 1;
            e.block
        })
    }

    /// Drop all blocks whose `block_id` is more than `max_age_blocks`
    /// behind the next expected block id.  Returns the count dropped.
    pub fn drop_if_late(&mut self, max_age_blocks: usize) -> usize {
        let threshold = self.next_expected_id.saturating_add(max_age_blocks as u64);
        let before = self.queue.len();
        // Re-build without the stale entries.
        let fresh: Vec<SchedulerEntry> = self
            .queue
            .drain()
            .filter(|e| e.block_id <= threshold)
            .collect();
        let dropped = before - fresh.len();
        self.queue.extend(fresh);
        dropped
    }

    /// Number of blocks currently waiting in the queue.
    pub fn pending(&self) -> usize {
        self.queue.len()
    }
}

impl Default for RealtimeScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_analyzer_worst_case() {
        let mut analyzer = LatencyAnalyzer::new(128);
        for v in [10, 50, 30, 80, 20] {
            analyzer.record(v);
        }
        assert_eq!(analyzer.worst_case(), 80);
    }

    #[test]
    fn test_latency_percentile() {
        let mut analyzer = LatencyAnalyzer::new(200);
        for v in 0_usize..100 {
            analyzer.record(v);
        }
        let p99 = analyzer.percentile_99();
        let worst = analyzer.worst_case();
        assert!(p99 <= worst, "99th percentile must be ≤ worst case");
        assert!(
            p99 >= 97,
            "99th percentile of 0..99 should be ≥ 97, got {p99}"
        );
    }

    #[test]
    fn test_overrun_detector_count() {
        let mut det = OverrunDetector::new();
        assert!(!det.check(100, 200)); // no overrun
        assert!(det.check(300, 200)); // overrun
        assert!(det.check(250, 200)); // overrun
        assert!(!det.check(100, 200)); // no overrun
        assert_eq!(det.overrun_count(), 2);
    }

    #[test]
    fn test_realtime_scheduler_edd() {
        let mut sched = RealtimeScheduler::new();
        // Submit out of order.
        sched.submit(SignalBlock::new(vec![], 300, 3));
        sched.submit(SignalBlock::new(vec![], 100, 1));
        sched.submit(SignalBlock::new(vec![], 200, 2));

        // Should come out in ascending block_id order.
        let b1 = sched.next_due().expect("block 1");
        let b2 = sched.next_due().expect("block 2");
        let b3 = sched.next_due().expect("block 3");
        assert_eq!(b1.block_id, 1);
        assert_eq!(b2.block_id, 2);
        assert_eq!(b3.block_id, 3);
    }
}
