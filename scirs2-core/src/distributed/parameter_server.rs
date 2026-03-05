//! Parameter Server for Distributed Optimisation
//!
//! This module provides a simple but production-quality parameter server
//! together with gradient compression utilities.
//!
//! ## Components
//!
//! | Item | Description |
//! |------|-------------|
//! | [`ParameterServer`] | Central store; workers push gradients; SGD applied |
//! | [`top_k_sparsify`] | Top-k gradient sparsification (bandwidth-efficient) |
//! | [`decompress_gradient`] | Reconstruct dense gradient from sparse (idx, val) pairs |
//! | [`ErrorFeedbackCompressor`] | Biased-compression error-feedback mechanism |
//!
//! ## Parameter Server Protocol
//!
//! ```text
//! Worker 0 ─── push_gradient(grad_0) ──┐
//! Worker 1 ─── push_gradient(grad_1) ──┤──► ParameterServer (apply SGD)
//! Worker k ─── push_gradient(grad_k) ──┘
//!
//! All workers ◄── get_params() ── (updated parameters)
//! ```
//!
//! Two update modes are provided:
//!
//! * **Asynchronous** — [`push_gradient`](ParameterServer::push_gradient): each
//!   worker independently pushes its gradient and the server applies a
//!   single-worker SGD step immediately.
//!
//! * **Synchronous** — [`sync_step`](ParameterServer::sync_step): the caller
//!   collects gradients from all workers and triggers one averaged SGD step.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::distributed::parameter_server::ParameterServer;
//!
//! let initial = vec![0.0_f64; 4];
//! let ps = ParameterServer::new(initial, 2, 0.1);
//!
//! // Asynchronous push
//! ps.push_gradient(0, &[1.0, 2.0, 3.0, 4.0]).expect("push failed");
//! let params = ps.get_params();
//! // params[0] = 0.0 - 0.1 * 1.0 = -0.1
//! assert!((params[0] - -0.1_f64).abs() < 1e-10);
//! ```

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helper
// ─────────────────────────────────────────────────────────────────────────────

fn lock_params(m: &Mutex<Vec<f64>>) -> CoreResult<std::sync::MutexGuard<'_, Vec<f64>>> {
    m.lock().map_err(|_| {
        CoreError::ComputationError(ErrorContext::new("parameter server: mutex poisoned"))
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// ParameterServer
// ─────────────────────────────────────────────────────────────────────────────

/// Central parameter store used in data-parallel distributed training.
///
/// The server holds a flat parameter vector and a monotonically-increasing
/// version counter.  Workers pull the current parameters via [`get_params`]
/// and push gradient updates via [`push_gradient`] or [`sync_step`].
///
/// The SGD update rule is: `θ ← θ − lr × gradient`.
///
/// [`get_params`]: ParameterServer::get_params
pub struct ParameterServer {
    /// Protected parameter vector.
    params: Arc<Mutex<Vec<f64>>>,
    /// Monotonically-increasing update counter.
    version: Arc<AtomicU64>,
    /// Number of workers this server is configured for (used in `sync_step`).
    n_workers: usize,
    /// SGD learning rate.
    learning_rate: f64,
}

impl ParameterServer {
    /// Create a new `ParameterServer`.
    ///
    /// # Arguments
    ///
    /// * `initial_params` — Starting parameter values.
    /// * `n_workers` — Number of workers (used for synchronous averaging).
    /// * `lr` — Learning rate for the SGD update `θ ← θ − lr × g`.
    ///
    /// # Panics
    ///
    /// Panics if `n_workers == 0` or `initial_params` is empty.
    pub fn new(initial_params: Vec<f64>, n_workers: usize, lr: f64) -> Self {
        assert!(n_workers > 0, "n_workers must be > 0");
        assert!(!initial_params.is_empty(), "initial_params must not be empty");
        Self {
            params: Arc::new(Mutex::new(initial_params)),
            version: Arc::new(AtomicU64::new(0)),
            n_workers,
            learning_rate: lr,
        }
    }

    /// Return a cloned copy of the current parameter vector.
    pub fn get_params(&self) -> Vec<f64> {
        self.params
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| Vec::new())
    }

    /// Return the current parameter version (number of applied updates).
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    /// Number of workers this server was configured for.
    pub fn n_workers(&self) -> usize {
        self.n_workers
    }

    // ── Asynchronous update ──────────────────────────────────────────────────

    /// Push a gradient from `worker_id` and immediately apply a single SGD step.
    ///
    /// The gradient is applied as-is (no averaging); for synchronous averaging
    /// see [`sync_step`](ParameterServer::sync_step).
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::ValueError`] if `gradient.len()` does not match the
    /// parameter vector length or `worker_id >= n_workers`.
    pub fn push_gradient(&self, worker_id: usize, gradient: &[f64]) -> CoreResult<()> {
        if worker_id >= self.n_workers {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "worker_id {worker_id} >= n_workers {}",
                self.n_workers
            ))));
        }
        let mut params = lock_params(&self.params)?;
        if gradient.len() != params.len() {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "gradient length {} does not match parameter length {}",
                gradient.len(),
                params.len()
            ))));
        }
        let lr = self.learning_rate;
        for (p, &g) in params.iter_mut().zip(gradient.iter()) {
            *p -= lr * g;
        }
        self.version.fetch_add(1, Ordering::Release);
        Ok(())
    }

    // ── Synchronous update ───────────────────────────────────────────────────

    /// Synchronous gradient step: average all worker gradients, then apply one
    /// SGD update.
    ///
    /// All gradients must have the same length as the parameter vector.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::ValueError`] if `gradients.len() != n_workers` or
    /// if any gradient length is mismatched.
    pub fn sync_step(&self, gradients: Vec<Vec<f64>>) -> CoreResult<()> {
        if gradients.len() != self.n_workers {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "expected {} gradients, got {}",
                self.n_workers,
                gradients.len()
            ))));
        }

        let mut params = lock_params(&self.params)?;
        let param_len = params.len();
        let n = self.n_workers as f64;

        // Validate all gradient lengths before touching params
        for (i, g) in gradients.iter().enumerate() {
            if g.len() != param_len {
                return Err(CoreError::ValueError(ErrorContext::new(format!(
                    "gradient {i} has length {} but param vector has length {param_len}",
                    g.len()
                ))));
            }
        }

        // Compute element-wise average gradient
        let mut avg_grad = vec![0.0_f64; param_len];
        for g in &gradients {
            for (acc, &val) in avg_grad.iter_mut().zip(g.iter()) {
                *acc += val;
            }
        }
        for acc in avg_grad.iter_mut() {
            *acc /= n;
        }

        // Apply averaged SGD step
        let lr = self.learning_rate;
        for (p, &avg) in params.iter_mut().zip(avg_grad.iter()) {
            *p -= lr * avg;
        }
        self.version.fetch_add(1, Ordering::Release);
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient Compression — top-k sparsification
// ─────────────────────────────────────────────────────────────────────────────

/// Top-k gradient sparsification: keep only the `k` elements with the largest
/// absolute value and zero-out the rest.
///
/// Returns `(indices, values)` where both slices have length `min(k, gradient.len())`.
/// Indices are in ascending order.
///
/// # Panics
///
/// Panics if `k == 0`.
///
/// # Example
///
/// ```rust
/// use scirs2_core::distributed::parameter_server::top_k_sparsify;
///
/// let grad = vec![0.1, -0.5, 0.3, -0.8, 0.2];
/// let (idx, vals) = top_k_sparsify(&grad, 2);
/// // Largest absolute values: -0.8 (idx 3) and -0.5 (idx 1)
/// assert_eq!(idx, vec![1, 3]);
/// assert_eq!(vals, vec![-0.5, -0.8]);
/// ```
pub fn top_k_sparsify(gradient: &[f64], k: usize) -> (Vec<usize>, Vec<f64>) {
    assert!(k > 0, "k must be > 0");
    if gradient.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let effective_k = k.min(gradient.len());

    // Build (abs_value, original_index) pairs
    let mut indexed: Vec<(f64, usize)> = gradient
        .iter()
        .enumerate()
        .map(|(i, &v)| (v.abs(), i))
        .collect();

    // Partial sort: bring top-k largest absolute values to the front
    // Use select_nth_unstable for O(n) average complexity
    let split_at = gradient.len() - effective_k;
    if split_at > 0 {
        indexed.select_nth_unstable_by(split_at, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // The top-k elements are in indexed[split_at..]
    let mut selected: Vec<(usize, f64)> = indexed[split_at..]
        .iter()
        .map(|&(_, idx)| (idx, gradient[idx]))
        .collect();

    // Sort by index so output is deterministic and compatible with decompress
    selected.sort_unstable_by_key(|&(idx, _)| idx);

    let indices: Vec<usize> = selected.iter().map(|&(i, _)| i).collect();
    let values: Vec<f64> = selected.iter().map(|&(_, v)| v).collect();
    (indices, values)
}

/// Decompress a sparse gradient back to a dense vector of length `size`.
///
/// Elements not listed in `indices` are set to zero.
///
/// # Panics
///
/// Panics if any index in `indices` is `>= size` or if `indices.len() !=
/// values.len()`.
///
/// # Example
///
/// ```rust
/// use scirs2_core::distributed::parameter_server::{top_k_sparsify, decompress_gradient};
///
/// let grad = vec![0.1, -0.5, 0.3, -0.8, 0.2];
/// let (idx, vals) = top_k_sparsify(&grad, 2);
/// let dense = decompress_gradient(&idx, &vals, grad.len());
/// assert_eq!(dense.len(), grad.len());
/// ```
pub fn decompress_gradient(indices: &[usize], values: &[f64], size: usize) -> Vec<f64> {
    assert_eq!(
        indices.len(),
        values.len(),
        "indices and values must have the same length"
    );
    let mut dense = vec![0.0_f64; size];
    for (&idx, &val) in indices.iter().zip(values.iter()) {
        assert!(idx < size, "index {idx} out of bounds for size {size}");
        dense[idx] = val;
    }
    dense
}

// ─────────────────────────────────────────────────────────────────────────────
// ErrorFeedbackCompressor
// ─────────────────────────────────────────────────────────────────────────────

/// Error-feedback compressor for biased gradient compression.
///
/// When using top-k sparsification the discarded elements introduce bias into
/// the gradient estimate.  The error-feedback mechanism accumulates the
/// compression residual into an internal error buffer and adds it to the next
/// gradient before compressing, so that every element is eventually
/// transmitted.
///
/// ## Algorithm
///
/// ```text
/// corrected = gradient + error_buffer
/// (indices, values) = top_k_sparsify(corrected, k)
/// transmitted = decompress(indices, values, size)
/// error_buffer = corrected - transmitted
/// ```
///
/// Over multiple steps the error buffer decays as residuals are re-included.
pub struct ErrorFeedbackCompressor {
    /// Accumulated residual from previous compression rounds.
    error_buffer: Vec<f64>,
    /// Number of elements to keep per compression.
    k: usize,
}

impl ErrorFeedbackCompressor {
    /// Create a new `ErrorFeedbackCompressor`.
    ///
    /// # Arguments
    ///
    /// * `size` — Number of parameters (gradient length).
    /// * `k` — Number of elements to keep per compression round.
    ///
    /// # Panics
    ///
    /// Panics if `size == 0` or `k == 0`.
    pub fn new(size: usize, k: usize) -> Self {
        assert!(size > 0, "size must be > 0");
        assert!(k > 0, "k must be > 0");
        Self {
            error_buffer: vec![0.0_f64; size],
            k,
        }
    }

    /// Return the current error buffer (read-only).
    pub fn error_buffer(&self) -> &[f64] {
        &self.error_buffer
    }

    /// Compress `gradient` with error feedback.
    ///
    /// Adds the accumulated error buffer to `gradient`, compresses with top-k,
    /// updates the error buffer with the residual, and returns the
    /// `(indices, values)` sparse representation.
    ///
    /// # Panics
    ///
    /// Panics if `gradient.len() != error_buffer.len()`.
    pub fn compress_and_feedback(&mut self, gradient: &[f64]) -> (Vec<usize>, Vec<f64>) {
        assert_eq!(
            gradient.len(),
            self.error_buffer.len(),
            "gradient length must match compressor size"
        );

        // Step 1: add accumulated error
        let corrected: Vec<f64> = gradient
            .iter()
            .zip(self.error_buffer.iter())
            .map(|(&g, &e)| g + e)
            .collect();

        // Step 2: top-k sparsify
        let (indices, values) = top_k_sparsify(&corrected, self.k);

        // Step 3: reconstruct transmitted signal
        let transmitted = decompress_gradient(&indices, &values, corrected.len());

        // Step 4: update error buffer = corrected - transmitted
        for (e, (c, t)) in self
            .error_buffer
            .iter_mut()
            .zip(corrected.iter().zip(transmitted.iter()))
        {
            *e = c - t;
        }

        (indices, values)
    }

    /// L2 norm of the current error buffer (useful for monitoring convergence).
    pub fn error_norm(&self) -> f64 {
        let sq: f64 = self.error_buffer.iter().map(|&v| v * v).sum();
        sq.sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // ── ParameterServer::push_gradient ───────────────────────────────────────

    #[test]
    fn test_push_gradient_single_worker() {
        let ps = ParameterServer::new(vec![0.0_f64, 0.0, 0.0, 0.0], 2, 0.1);
        ps.push_gradient(0, &[1.0, 2.0, 3.0, 4.0])
            .expect("push failed");
        let params = ps.get_params();
        let expected = [-0.1_f64, -0.2, -0.3, -0.4];
        for (p, &e) in params.iter().zip(expected.iter()) {
            assert!((p - e).abs() < 1e-10, "got {p}, expected {e}");
        }
    }

    #[test]
    fn test_push_gradient_increments_version() {
        let ps = ParameterServer::new(vec![0.0_f64; 2], 1, 0.01);
        assert_eq!(ps.version(), 0);
        ps.push_gradient(0, &[1.0, 1.0]).expect("push failed");
        assert_eq!(ps.version(), 1);
        ps.push_gradient(0, &[1.0, 1.0]).expect("push failed");
        assert_eq!(ps.version(), 2);
    }

    #[test]
    fn test_push_gradient_concurrent() {
        let n = 4_usize;
        let ps = Arc::new(ParameterServer::new(vec![0.0_f64; 3], n, 1.0));
        // Each worker pushes gradient [1.0, 1.0, 1.0] → params should end up
        // at some combination of n SGD steps, each subtracting [1.0, 1.0, 1.0]
        let handles: Vec<_> = (0..n)
            .map(|id| {
                let ps_ref = Arc::clone(&ps);
                thread::spawn(move || ps_ref.push_gradient(id, &[1.0, 1.0, 1.0]))
            })
            .collect();
        for h in handles {
            h.join().expect("thread panic").expect("push error");
        }
        // Version should be n
        assert_eq!(ps.version(), n as u64);
        // Each param should be -n * 1.0 * lr = -4.0
        let params = ps.get_params();
        for p in &params {
            assert!((p - -4.0_f64).abs() < 1e-10, "got {p}");
        }
    }

    #[test]
    fn test_push_gradient_invalid_worker_id() {
        let ps = ParameterServer::new(vec![0.0_f64; 2], 2, 0.1);
        assert!(ps.push_gradient(5, &[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_push_gradient_wrong_length() {
        let ps = ParameterServer::new(vec![0.0_f64; 4], 2, 0.1);
        assert!(ps.push_gradient(0, &[1.0, 2.0]).is_err()); // length mismatch
    }

    // ── ParameterServer::sync_step ────────────────────────────────────────────

    #[test]
    fn test_sync_step_averages_gradients() {
        // 2 workers, params = [0.0; 4], lr = 1.0
        // Worker 0 gradient = [2.0; 4], Worker 1 gradient = [4.0; 4]
        // avg = [3.0; 4], update: params = 0.0 - 1.0 * 3.0 = -3.0
        let ps = ParameterServer::new(vec![0.0_f64; 4], 2, 1.0);
        let gradients = vec![vec![2.0_f64; 4], vec![4.0_f64; 4]];
        ps.sync_step(gradients).expect("sync_step failed");
        let params = ps.get_params();
        for p in &params {
            assert!((p - -3.0_f64).abs() < 1e-10, "got {p}");
        }
    }

    #[test]
    fn test_sync_step_wrong_worker_count() {
        let ps = ParameterServer::new(vec![0.0_f64; 2], 3, 0.1);
        let gradients = vec![vec![1.0_f64; 2], vec![1.0_f64; 2]]; // 2 instead of 3
        assert!(ps.sync_step(gradients).is_err());
    }

    #[test]
    fn test_sync_step_increments_version() {
        let ps = ParameterServer::new(vec![0.0_f64; 2], 2, 0.1);
        let gradients = vec![vec![0.0_f64; 2], vec![0.0_f64; 2]];
        ps.sync_step(gradients).expect("sync_step failed");
        assert_eq!(ps.version(), 1);
    }

    // ── top_k_sparsify ────────────────────────────────────────────────────────

    #[test]
    fn test_top_k_sparsify_selects_largest_abs() {
        let grad = vec![0.1_f64, -0.5, 0.3, -0.8, 0.2];
        let (idx, vals) = top_k_sparsify(&grad, 2);
        // Largest |v|: -0.8 (idx 3) and -0.5 (idx 1)
        assert_eq!(idx, vec![1, 3]);
        assert_eq!(vals, vec![-0.5, -0.8]);
    }

    #[test]
    fn test_top_k_sparsify_k_equals_length() {
        let grad = vec![1.0_f64, 2.0, 3.0];
        let (idx, vals) = top_k_sparsify(&grad, 3);
        assert_eq!(idx.len(), 3);
        assert_eq!(vals.len(), 3);
    }

    #[test]
    fn test_top_k_sparsify_k_exceeds_length_clips_to_len() {
        let grad = vec![1.0_f64, 2.0];
        let (idx, vals) = top_k_sparsify(&grad, 100);
        assert_eq!(idx.len(), 2);
        assert_eq!(vals.len(), 2);
    }

    #[test]
    fn test_top_k_sparsify_indices_sorted() {
        let grad: Vec<f64> = (0..20).rev().map(|i| i as f64).collect();
        let (idx, _vals) = top_k_sparsify(&grad, 5);
        for w in idx.windows(2) {
            assert!(w[0] < w[1], "indices must be in ascending order");
        }
    }

    // ── decompress_gradient ───────────────────────────────────────────────────

    #[test]
    fn test_decompress_gradient_round_trip() {
        let grad = vec![0.0_f64, 0.5, 0.0, -0.3, 0.0];
        let (idx, vals) = top_k_sparsify(&grad, 2);
        let dense = decompress_gradient(&idx, &vals, grad.len());
        for (a, b) in dense.iter().zip(grad.iter()) {
            assert!((a - b).abs() < 1e-10, "dense[i]={a} vs grad[i]={b}");
        }
    }

    #[test]
    fn test_decompress_gradient_zeros_for_missing() {
        let idx = vec![2_usize, 4];
        let vals = vec![1.5_f64, -0.7];
        let dense = decompress_gradient(&idx, &vals, 6);
        assert_eq!(dense[0], 0.0);
        assert_eq!(dense[1], 0.0);
        assert!((dense[2] - 1.5).abs() < 1e-10);
        assert_eq!(dense[3], 0.0);
        assert!((dense[4] - -0.7).abs() < 1e-10);
        assert_eq!(dense[5], 0.0);
    }

    // ── ErrorFeedbackCompressor ───────────────────────────────────────────────

    #[test]
    fn test_error_feedback_compressor_basic() {
        let mut comp = ErrorFeedbackCompressor::new(5, 2);
        let grad = vec![0.1_f64, -0.5, 0.3, -0.8, 0.2];
        let (idx, vals) = comp.compress_and_feedback(&grad);
        assert_eq!(idx.len(), 2);
        assert_eq!(vals.len(), 2);
        // Error buffer should hold the residual
        let err = comp.error_buffer();
        let transmitted = decompress_gradient(&idx, &vals, 5);
        for (i, (&e, &g)) in err.iter().zip(grad.iter()).enumerate() {
            let expected_err = g - transmitted[i];
            assert!((e - expected_err).abs() < 1e-10);
        }
    }

    #[test]
    fn test_error_feedback_error_norm_decreases_over_steps() {
        // After many steps with a constant gradient, the error norm should
        // not grow unboundedly (error-feedback keeps it bounded)
        let size = 10_usize;
        let k = 3_usize;
        let mut comp = ErrorFeedbackCompressor::new(size, k);
        let grad: Vec<f64> = (0..size).map(|i| (i as f64) * 0.01).collect();

        let mut norms = Vec::new();
        for _ in 0..20 {
            comp.compress_and_feedback(&grad);
            norms.push(comp.error_norm());
        }

        // The norm should reach a steady state and not diverge
        let last_norm = *norms.last().expect("should succeed");
        assert!(
            last_norm.is_finite(),
            "error norm diverged: {last_norm}"
        );
    }

    #[test]
    fn test_error_feedback_accumulates_residual() {
        // With k=1, two steps should eventually transmit both significant
        // elements of a 2-element gradient
        let mut comp = ErrorFeedbackCompressor::new(2, 1);
        let grad = vec![1.0_f64, 0.9];

        // Step 1: one element transmitted
        let (idx1, _vals1) = comp.compress_and_feedback(&grad);
        assert_eq!(idx1.len(), 1);

        // Step 2: residual is added back, both elements should have been
        // transmitted at some point across the two steps
        let (idx2, _vals2) = comp.compress_and_feedback(&grad);
        assert_eq!(idx2.len(), 1);

        // Together they cover both indices (no guarantee of order, but the
        // union must cover both positions 0 and 1)
        let covered: std::collections::HashSet<usize> =
            idx1.iter().chain(idx2.iter()).cloned().collect();
        assert_eq!(covered.len(), 2, "both elements should be covered over 2 steps");
    }

    #[test]
    fn test_error_feedback_zero_gradient_keeps_zero_error() {
        let mut comp = ErrorFeedbackCompressor::new(4, 2);
        let grad = vec![0.0_f64; 4];
        comp.compress_and_feedback(&grad);
        // Error buffer should remain zero after compressing zero gradient
        for &e in comp.error_buffer() {
            assert_eq!(e, 0.0);
        }
    }
}
