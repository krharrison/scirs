//! Ring All-Reduce for Distributed Gradient Averaging
//!
//! This module implements the bandwidth-optimal **ring all-reduce** algorithm
//! used in data-parallel deep learning (e.g., as in Baidu's deep-speech paper
//! and Horovod).  The implementation is in-process and models what would be
//! inter-process communication using shared-memory and threads.
//!
//! ## Algorithm Overview
//!
//! The ring all-reduce consists of two phases:
//!
//! 1. **Scatter-Reduce** — `n-1` rounds.  Each worker sends a *chunk* to its
//!    right neighbour and accumulates the chunk it receives from its left
//!    neighbour.  After this phase every worker holds one fully-reduced shard.
//!
//! 2. **All-Gather** — `n-1` rounds.  Each worker propagates its fully-reduced
//!    shard around the ring so that every worker ends up with all shards.
//!
//! The total data communicated per worker is `2 × (n-1)/n × tensor_size`, which
//! approaches `2 × tensor_size` as `n → ∞`, independent of the ring size.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::distributed::ring_allreduce::RingAllReduce;
//!
//! let ring = RingAllReduce::new(4);
//! let local_tensors = vec![
//!     vec![1.0_f64, 2.0, 3.0],
//!     vec![4.0,     5.0, 6.0],
//!     vec![7.0,     8.0, 9.0],
//!     vec![2.0,     1.0, 0.0],
//! ];
//! let sums = ring.reduce_sum(local_tensors).expect("ring allreduce failed");
//! // Every worker should hold [14.0, 16.0, 18.0]
//! for s in &sums {
//!     assert_eq!(s, &vec![14.0, 16.0, 18.0]);
//! }
//! ```

use crate::error::{CoreError, CoreResult, ErrorContext};

// ─────────────────────────────────────────────────────────────────────────────
// RingAllReduce
// ─────────────────────────────────────────────────────────────────────────────

/// Ring all-reduce implementation for bandwidth-optimal gradient averaging.
///
/// The struct is intentionally lightweight — it only stores the worker count.
/// Actual data exchange is performed inside [`reduce_sum`](RingAllReduce::reduce_sum)
/// and [`reduce_mean`](RingAllReduce::reduce_mean).
#[derive(Debug, Clone, Copy)]
pub struct RingAllReduce {
    n_workers: usize,
}

impl RingAllReduce {
    /// Create a new `RingAllReduce` for `n_workers` participants.
    ///
    /// # Panics
    ///
    /// Panics if `n_workers == 0`.
    pub fn new(n_workers: usize) -> Self {
        assert!(n_workers > 0, "n_workers must be > 0");
        Self { n_workers }
    }

    /// Number of workers this ring was configured for.
    pub fn n_workers(&self) -> usize {
        self.n_workers
    }

    // ── Public interface ─────────────────────────────────────────────────────

    /// Run ring all-reduce and return the **sum** for every worker.
    ///
    /// `local_tensors[i]` is the tensor held by worker `i`.  All tensors must
    /// have the same length.  Returns a `Vec` with `n_workers` entries, each
    /// containing the element-wise sum of all input tensors.
    ///
    /// # Errors
    ///
    /// - [`CoreError::ValueError`] if `local_tensors.len() != n_workers` or if
    ///   tensor lengths are not uniform.
    pub fn reduce_sum(&self, local_tensors: Vec<Vec<f64>>) -> CoreResult<Vec<Vec<f64>>> {
        self.validate(&local_tensors)?;

        // Edge case: single worker — trivially already "reduced"
        if self.n_workers == 1 {
            return Ok(local_tensors);
        }

        let tensor_len = local_tensors[0].len();
        self.ring_allreduce_impl(local_tensors, tensor_len, ReduceOp::Sum)
    }

    /// Run ring all-reduce and return the element-wise **mean** for every worker.
    pub fn reduce_mean(&self, local_tensors: Vec<Vec<f64>>) -> CoreResult<Vec<Vec<f64>>> {
        let n = self.n_workers as f64;
        let sums = self.reduce_sum(local_tensors)?;
        Ok(sums
            .into_iter()
            .map(|v| v.into_iter().map(|x| x / n).collect())
            .collect())
    }

    /// Theoretical bandwidth consumed per worker for a tensor of `tensor_size`
    /// elements (send + receive in both phases combined).
    ///
    /// Returns the number of scalar elements transmitted per worker.
    pub fn bandwidth_per_worker(&self, tensor_size: usize) -> f64 {
        if self.n_workers <= 1 {
            return 0.0;
        }
        let n = self.n_workers as f64;
        // Each of the two phases transmits (n-1)/n fraction of the tensor
        2.0 * (n - 1.0) / n * tensor_size as f64
    }

    // ── Core algorithm ───────────────────────────────────────────────────────

    fn ring_allreduce_impl(
        &self,
        local_tensors: Vec<Vec<f64>>,
        tensor_len: usize,
        op: ReduceOp,
    ) -> CoreResult<Vec<Vec<f64>>> {
        let n = self.n_workers;

        // ── Compute chunk boundaries ─────────────────────────────────────────
        // Split the tensor into n chunks.  The last chunk may be larger if the
        // tensor length is not a multiple of n.
        let base_chunk = tensor_len / n;
        let remainder = tensor_len % n;

        // chunk_start[i] = start index of chunk i; chunk_start[n] = tensor_len
        let mut chunk_start = Vec::with_capacity(n + 1);
        let mut offset = 0_usize;
        for i in 0..n {
            chunk_start.push(offset);
            offset += if i < remainder { base_chunk + 1 } else { base_chunk };
        }
        chunk_start.push(tensor_len);

        // ── Working buffers: one per worker ─────────────────────────────────
        let mut buffers: Vec<Vec<f64>> = local_tensors;

        // ── Phase 1: Scatter-Reduce ──────────────────────────────────────────
        //
        // In each of n-1 rounds, worker i sends chunk (i - round) mod n to
        // its right neighbour (i+1) mod n, and accumulates the chunk it
        // receives from its left neighbour (i-1) mod n.
        //
        // We simulate all workers simultaneously by iterating over workers
        // in the inner loop and using a temporary send buffer.

        for round in 0..(n - 1) {
            // For each worker compute which chunk it will send this round.
            // After scatter-reduce phase, worker i owns the fully-reduced
            // shard at chunk index (i + 1) mod n  (standard formulation).
            // Send chunk index for worker i in round r = (i - r + n) mod n.

            // Collect all chunks that will be sent this round (before mutation)
            let sent_chunks: Vec<(usize, Vec<f64>)> = (0..n)
                .map(|worker| {
                    let send_chunk_idx = (worker + n - round) % n;
                    let start = chunk_start[send_chunk_idx];
                    let end = chunk_start[send_chunk_idx + 1];
                    let data = buffers[worker][start..end].to_vec();
                    (worker, data)
                })
                .collect();

            // Apply received chunks to destination workers
            for (sender, chunk_data) in &sent_chunks {
                let receiver = (*sender + 1) % n;
                let recv_chunk_idx = (*sender + n - round) % n;
                let start = chunk_start[recv_chunk_idx];
                let end = chunk_start[recv_chunk_idx + 1];

                match op {
                    ReduceOp::Sum => {
                        for (acc, &val) in buffers[receiver][start..end]
                            .iter_mut()
                            .zip(chunk_data.iter())
                        {
                            *acc += val;
                        }
                    }
                }
            }
        }

        // ── Phase 2: All-Gather ──────────────────────────────────────────────
        //
        // In each of n-1 rounds, worker i sends chunk (i + 1 - round) mod n
        // to its right neighbour, which copies (not accumulates) the data.

        for round in 0..(n - 1) {
            let sent_chunks: Vec<(usize, Vec<f64>)> = (0..n)
                .map(|worker| {
                    let send_chunk_idx = (worker + 1 + n - round) % n;
                    let start = chunk_start[send_chunk_idx];
                    let end = chunk_start[send_chunk_idx + 1];
                    let data = buffers[worker][start..end].to_vec();
                    (worker, data)
                })
                .collect();

            for (sender, chunk_data) in &sent_chunks {
                let receiver = (*sender + 1) % n;
                let recv_chunk_idx = (*sender + 1 + n - round) % n;
                let start = chunk_start[recv_chunk_idx];
                let end = chunk_start[recv_chunk_idx + 1];
                buffers[receiver][start..end].copy_from_slice(chunk_data);
            }
        }

        Ok(buffers)
    }

    // ── Validation ───────────────────────────────────────────────────────────

    fn validate(&self, tensors: &[Vec<f64>]) -> CoreResult<()> {
        if tensors.len() != self.n_workers {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "expected {} tensors (one per worker), got {}",
                self.n_workers,
                tensors.len()
            ))));
        }
        if tensors.is_empty() {
            return Err(CoreError::ValueError(ErrorContext::new(
                "tensor list must not be empty",
            )));
        }
        let expected_len = tensors[0].len();
        if expected_len == 0 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "tensors must have at least one element",
            )));
        }
        for (i, t) in tensors.iter().enumerate() {
            if t.len() != expected_len {
                return Err(CoreError::ValueError(ErrorContext::new(format!(
                    "tensor {i} has length {} but expected {expected_len}",
                    t.len()
                ))));
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal enum for reduce operation kind
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum ReduceOp {
    Sum,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── reduce_sum ────────────────────────────────────────────────────────────

    #[test]
    fn test_reduce_sum_4_workers() {
        let ring = RingAllReduce::new(4);
        let tensors = vec![
            vec![1.0_f64, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![2.0, 1.0, 0.0],
        ];
        let expected = vec![14.0_f64, 16.0, 18.0]; // 1+4+7+2, 2+5+8+1, 3+6+9+0
        let results = ring.reduce_sum(tensors).expect("reduce_sum failed");
        assert_eq!(results.len(), 4);
        for r in &results {
            for (a, b) in r.iter().zip(expected.iter()) {
                assert!(
                    (a - b).abs() < 1e-10,
                    "mismatch: got {a}, expected {b}"
                );
            }
        }
    }

    #[test]
    fn test_reduce_sum_2_workers() {
        let ring = RingAllReduce::new(2);
        let tensors = vec![vec![3.0_f64, 0.0], vec![1.0, 4.0]];
        let expected = vec![4.0_f64, 4.0];
        let results = ring.reduce_sum(tensors).expect("reduce_sum 2-worker failed");
        assert_eq!(results.len(), 2);
        for r in &results {
            for (a, b) in r.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_reduce_sum_single_worker() {
        let ring = RingAllReduce::new(1);
        let tensors = vec![vec![5.0_f64, 6.0, 7.0]];
        let results = ring.reduce_sum(tensors.clone()).expect("single worker failed");
        assert_eq!(results, tensors);
    }

    #[test]
    fn test_reduce_sum_matches_naive_sum() {
        // Compare ring result against naive element-wise sum
        let n = 5_usize;
        let ring = RingAllReduce::new(n);
        let tensors: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![(i as f64) * 1.1, (i as f64) * 2.2 + 0.5])
            .collect();

        // Naive sum
        let mut naive = vec![0.0_f64; 2];
        for t in &tensors {
            naive[0] += t[0];
            naive[1] += t[1];
        }

        let results = ring.reduce_sum(tensors).expect("ring sum failed");
        for r in &results {
            assert!((r[0] - naive[0]).abs() < 1e-9);
            assert!((r[1] - naive[1]).abs() < 1e-9);
        }
    }

    // ── reduce_mean ───────────────────────────────────────────────────────────

    #[test]
    fn test_reduce_mean_4_workers() {
        let ring = RingAllReduce::new(4);
        let tensors = vec![
            vec![4.0_f64],
            vec![4.0],
            vec![4.0],
            vec![4.0],
        ];
        // sum = 16.0, mean = 4.0
        let results = ring.reduce_mean(tensors).expect("reduce_mean failed");
        for r in &results {
            assert!((r[0] - 4.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_reduce_mean_heterogeneous() {
        let ring = RingAllReduce::new(3);
        let tensors = vec![vec![3.0_f64], vec![6.0], vec![9.0]];
        // sum = 18.0, mean = 6.0
        let results = ring.reduce_mean(tensors).expect("reduce_mean failed");
        for r in &results {
            assert!((r[0] - 6.0).abs() < 1e-10);
        }
    }

    // ── bandwidth_per_worker ──────────────────────────────────────────────────

    #[test]
    fn test_bandwidth_per_worker_formula() {
        // For n=4, tensor_size=1000: bandwidth = 2 * (3/4) * 1000 = 1500
        let ring = RingAllReduce::new(4);
        let bw = ring.bandwidth_per_worker(1000);
        assert!((bw - 1500.0).abs() < 1e-9, "expected 1500.0 got {bw}");
    }

    #[test]
    fn test_bandwidth_single_worker_is_zero() {
        let ring = RingAllReduce::new(1);
        assert_eq!(ring.bandwidth_per_worker(100), 0.0);
    }

    // ── validation ────────────────────────────────────────────────────────────

    #[test]
    fn test_wrong_tensor_count_returns_error() {
        let ring = RingAllReduce::new(3);
        let tensors = vec![vec![1.0_f64], vec![2.0]]; // only 2, need 3
        assert!(ring.reduce_sum(tensors).is_err());
    }

    #[test]
    fn test_mismatched_tensor_lengths_returns_error() {
        let ring = RingAllReduce::new(2);
        let tensors = vec![vec![1.0_f64, 2.0], vec![3.0]]; // unequal lengths
        assert!(ring.reduce_sum(tensors).is_err());
    }

    // ── non-divisible tensor length ────────────────────────────────────────────

    #[test]
    fn test_reduce_sum_non_divisible_tensor_length() {
        // tensor_len = 7, n_workers = 3  →  chunks: [3, 2, 2]
        let n = 3_usize;
        let ring = RingAllReduce::new(n);
        let tensors: Vec<Vec<f64>> = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        ];
        let expected = vec![6.0_f64; 7]; // 1+2+3 = 6 for every element
        let results = ring.reduce_sum(tensors).expect("non-divisible failed");
        for r in &results {
            for (a, b) in r.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-10, "got {a}, expected {b}");
            }
        }
    }
}
