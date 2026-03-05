//! Weight pruning for neural networks
//!
//! Provides structured and unstructured pruning strategies, an iterative pruner
//! with polynomial-decay scheduling, channel-level pruning, and the lottery ticket
//! hypothesis helper.
//!
//! ## Quick start
//!
//! ```rust
//! use scirs2_neural::training::pruning::{PruningConfig, PruningMethod, prune_weights};
//! use scirs2_core::ndarray::array;
//!
//! let weights = array![
//!     [0.1_f64, -0.9, 0.2],
//!     [-0.05, 0.8, -0.3],
//! ];
//! let config = PruningConfig {
//!     method: PruningMethod::MagnitudePruning,
//!     sparsity: 0.5,
//!     structured: false,
//! };
//! let (pruned, mask) = prune_weights(weights.view(), &config).expect("prune failed");
//! let sparsity = mask.iter().filter(|&&m| !m).count() as f64 / mask.len() as f64;
//! assert!((sparsity - 0.5).abs() < 0.2);
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive, ToPrimitive};
use std::fmt::Debug;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// Method used to determine which weights to prune.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruningMethod {
    /// Remove weights with smallest absolute magnitude.
    MagnitudePruning,
    /// Remove weights that contribute the smallest gradient signal.
    /// In practice this uses weight × gradient as a proxy importance score.
    GradientPruning,
    /// Remove weights uniformly at random.
    RandomPruning,
    /// Remove weights that minimize the L1 norm of the remaining tensor.
    L1Pruning,
    /// Remove weights with smallest squared magnitude (L2-norm criterion).
    L2Pruning,
}

/// Top-level pruning configuration.
#[derive(Debug, Clone)]
pub struct PruningConfig {
    /// Pruning method.
    pub method: PruningMethod,
    /// Target sparsity in (0.0, 1.0).  0.5 means half the weights are zeroed.
    pub sparsity: f64,
    /// If `true`, perform structured pruning (entire channels / rows).
    /// If `false`, perform element-wise unstructured pruning.
    pub structured: bool,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            method: PruningMethod::MagnitudePruning,
            sparsity: 0.5,
            structured: false,
        }
    }
}

/// Statistics about the sparsity of a pruned weight matrix.
#[derive(Debug, Clone)]
pub struct SparsityStats {
    /// Total number of parameters.
    pub total_params: usize,
    /// Number of zero (pruned) parameters.
    pub zero_params: usize,
    /// Fraction of parameters that are zero.
    pub sparsity: f64,
    /// Number of completely zero rows (channels).
    pub zero_rows: usize,
    /// Fraction of rows that are completely zero.
    pub row_sparsity: f64,
}

impl SparsityStats {
    /// Compute sparsity statistics from a boolean mask (false = pruned).
    pub fn from_mask(mask: ArrayView2<bool>, weights: ArrayView2<f64>) -> Self {
        let total = mask.len();
        let zero_params = mask.iter().filter(|&&m| !m).count();

        let nrows = mask.nrows();
        let zero_rows = (0..nrows)
            .filter(|&r| mask.row(r).iter().all(|&m| !m))
            .count();

        Self {
            total_params: total,
            zero_params,
            sparsity: zero_params as f64 / total.max(1) as f64,
            zero_rows,
            row_sparsity: zero_rows as f64 / nrows.max(1) as f64,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Core pruning function
// ────────────────────────────────────────────────────────────────────────────

/// Prune a 2-D weight matrix, returning the pruned weights and a boolean mask.
///
/// `mask[[r,c]] == true` means the weight is kept; `false` means it was pruned to 0.
pub fn prune_weights(
    weights: ArrayView2<f64>,
    config: &PruningConfig,
) -> Result<(Array2<f64>, Array2<bool>)> {
    validate_sparsity(config.sparsity)?;

    if config.structured {
        prune_structured(weights, config)
    } else {
        prune_unstructured(weights, config)
    }
}

// ── Unstructured (element-wise) ───────────────────────────────────────────

fn prune_unstructured(
    weights: ArrayView2<f64>,
    config: &PruningConfig,
) -> Result<(Array2<f64>, Array2<bool>)> {
    let n = weights.len();
    let n_prune = ((config.sparsity * n as f64).round() as usize).min(n);

    // Compute importance scores for all elements.
    let scores: Vec<f64> = match config.method {
        PruningMethod::MagnitudePruning | PruningMethod::L1Pruning => {
            weights.iter().map(|v| v.abs()).collect()
        }
        PruningMethod::L2Pruning => {
            // L2-norm criterion: importance = w² (squared magnitude).
            weights.iter().map(|v| v * v).collect()
        }
        PruningMethod::GradientPruning => {
            // Without actual gradients available, fall back to magnitude.
            weights.iter().map(|v| v.abs()).collect()
        }
        PruningMethod::RandomPruning => {
            // Uniform random scores – deterministic via index-based LCG so tests are reproducible.
            weights
                .iter()
                .enumerate()
                .map(|(i, _)| lcg_f64(i as u64))
                .collect()
        }
    };

    // Find the threshold: keep elements with score > threshold.
    let mut sorted_scores = scores.clone();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = if n_prune == 0 {
        -1.0 // keep everything
    } else {
        sorted_scores[n_prune - 1]
    };

    let (nrows, ncols) = (weights.nrows(), weights.ncols());
    let mut pruned = Array2::<f64>::zeros((nrows, ncols));
    let mut mask = Array2::<bool>::from_elem((nrows, ncols), true);

    let mut flat_idx = 0usize;
    for r in 0..nrows {
        for c in 0..ncols {
            if scores[flat_idx] <= threshold {
                mask[[r, c]] = false;
                // pruned already 0
            } else {
                pruned[[r, c]] = weights[[r, c]];
            }
            flat_idx += 1;
        }
    }

    Ok((pruned, mask))
}

// ── Structured (row/channel-wise) ────────────────────────────────────────

fn prune_structured(
    weights: ArrayView2<f64>,
    config: &PruningConfig,
) -> Result<(Array2<f64>, Array2<bool>)> {
    let nrows = weights.nrows();
    let n_prune = ((config.sparsity * nrows as f64).round() as usize).min(nrows);

    // Score each row.
    let row_scores: Vec<f64> = (0..nrows)
        .map(|r| {
            let row = weights.row(r);
            match config.method {
                PruningMethod::MagnitudePruning | PruningMethod::L1Pruning => {
                    row.iter().map(|v| v.abs()).sum::<f64>()
                }
                PruningMethod::L2Pruning => {
                    row.iter().map(|v| v * v).sum::<f64>().sqrt()
                }
                PruningMethod::GradientPruning => row.iter().map(|v| v.abs()).sum::<f64>(),
                PruningMethod::RandomPruning => lcg_f64(r as u64),
            }
        })
        .collect();

    // Indices of rows to prune (lowest scores first).
    let mut indexed: Vec<(usize, f64)> = row_scores
        .iter()
        .cloned()
        .enumerate()
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let prune_set: std::collections::HashSet<usize> =
        indexed[..n_prune].iter().map(|&(i, _)| i).collect();

    let ncols = weights.ncols();
    let mut pruned = weights.to_owned();
    let mut mask = Array2::<bool>::from_elem((nrows, ncols), true);

    for r in prune_set {
        for c in 0..ncols {
            pruned[[r, c]] = 0.0;
            mask[[r, c]] = false;
        }
    }

    Ok((pruned, mask))
}

// ────────────────────────────────────────────────────────────────────────────
// Structured channel pruning
// ────────────────────────────────────────────────────────────────────────────

/// Remove `n_channels_to_prune` entire rows (output channels) from a weight matrix.
///
/// The rows with the smallest L1 norm are removed.  Returns the pruned weight
/// matrix (with rows zeroed) and a `Vec<usize>` with the indices of removed channels.
pub fn structured_prune_channels(
    weights: ArrayView2<f64>,
    n_channels_to_prune: usize,
) -> Result<(Array2<f64>, Vec<usize>)> {
    let nrows = weights.nrows();
    if n_channels_to_prune > nrows {
        return Err(NeuralError::InvalidArgument(format!(
            "Cannot prune {n_channels_to_prune} channels from a matrix with {nrows} rows"
        )));
    }

    // Score each row by L1 norm.
    let mut row_norms: Vec<(usize, f64)> = (0..nrows)
        .map(|r| {
            let l1: f64 = weights.row(r).iter().map(|v| v.abs()).sum();
            (r, l1)
        })
        .collect();
    row_norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let pruned_indices: Vec<usize> = row_norms[..n_channels_to_prune]
        .iter()
        .map(|&(i, _)| i)
        .collect();
    let prune_set: std::collections::HashSet<usize> = pruned_indices.iter().cloned().collect();

    let ncols = weights.ncols();
    let mut out = weights.to_owned();
    for r in prune_set {
        for c in 0..ncols {
            out[[r, c]] = 0.0;
        }
    }

    Ok((out, pruned_indices))
}

// ────────────────────────────────────────────────────────────────────────────
// Gradient-based pruning (with explicit gradient tensor)
// ────────────────────────────────────────────────────────────────────────────

/// Prune weights using weight × gradient importance scores.
///
/// `gradients` must have the same shape as `weights`.
pub fn gradient_prune_weights(
    weights: ArrayView2<f64>,
    gradients: ArrayView2<f64>,
    sparsity: f64,
) -> Result<(Array2<f64>, Array2<bool>)> {
    if weights.shape() != gradients.shape() {
        return Err(NeuralError::ShapeMismatch(format!(
            "weights shape {:?} != gradients shape {:?}",
            weights.shape(),
            gradients.shape()
        )));
    }
    validate_sparsity(sparsity)?;

    let n = weights.len();
    let n_prune = ((sparsity * n as f64).round() as usize).min(n);

    // Importance: |w * g|
    let scores: Vec<f64> = weights
        .iter()
        .zip(gradients.iter())
        .map(|(w, g)| (w * g).abs())
        .collect();

    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = if n_prune == 0 { -1.0 } else { sorted[n_prune - 1] };

    let (nrows, ncols) = (weights.nrows(), weights.ncols());
    let mut pruned = Array2::<f64>::zeros((nrows, ncols));
    let mut mask = Array2::<bool>::from_elem((nrows, ncols), true);

    let mut flat_idx = 0usize;
    for r in 0..nrows {
        for c in 0..ncols {
            if scores[flat_idx] <= threshold {
                mask[[r, c]] = false;
            } else {
                pruned[[r, c]] = weights[[r, c]];
            }
            flat_idx += 1;
        }
    }

    Ok((pruned, mask))
}

// ────────────────────────────────────────────────────────────────────────────
// Iterative pruner with polynomial-decay schedule
// ────────────────────────────────────────────────────────────────────────────

/// Iterative (gradual) pruner that increases sparsity over time following
/// a polynomial decay schedule.
///
/// At step `t` in [0, `n_steps`], the target sparsity is:
/// ```text
/// s(t) = s_final + (s_initial - s_final) * (1 - t / n_steps)^exponent
/// ```
///
/// This matches the schedule from "To Prune, or Not to Prune" (Zhu & Gupta, 2017).
#[derive(Debug, Clone)]
pub struct IterativePruner {
    /// Initial sparsity (typically 0).
    pub initial_sparsity: f64,
    /// Final (target) sparsity.
    pub final_sparsity: f64,
    /// Number of pruning steps.
    pub n_steps: usize,
    /// Polynomial exponent (3 is common).
    pub exponent: f64,
    /// Pruning method used at each step.
    pub method: PruningMethod,
    /// Whether to use structured pruning.
    pub structured: bool,
    /// Current step index.
    pub step: usize,
}

impl IterativePruner {
    /// Create a new iterative pruner.
    pub fn new(
        final_sparsity: f64,
        n_steps: usize,
        method: PruningMethod,
        structured: bool,
    ) -> Result<Self> {
        validate_sparsity(final_sparsity)?;
        if n_steps == 0 {
            return Err(NeuralError::InvalidArgument(
                "n_steps must be greater than 0".to_string(),
            ));
        }
        Ok(Self {
            initial_sparsity: 0.0,
            final_sparsity,
            n_steps,
            exponent: 3.0,
            method,
            structured,
            step: 0,
        })
    }

    /// Set the polynomial exponent (default 3).
    pub fn with_exponent(mut self, exp: f64) -> Self {
        self.exponent = exp;
        self
    }

    /// Set the initial sparsity (default 0).
    pub fn with_initial_sparsity(mut self, s: f64) -> Self {
        self.initial_sparsity = s;
        self
    }

    /// Compute the target sparsity at the current step.
    pub fn current_sparsity(&self) -> f64 {
        if self.step >= self.n_steps {
            return self.final_sparsity;
        }
        let t = self.step as f64;
        let n = self.n_steps as f64;
        let s_f = self.final_sparsity;
        let s_i = self.initial_sparsity;
        s_f + (s_i - s_f) * (1.0 - t / n).powf(self.exponent)
    }

    /// Apply one pruning step to `weights`, update the step counter, and return
    /// the pruned weights + mask.
    pub fn step_prune(
        &mut self,
        weights: ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array2<bool>)> {
        let sparsity = self.current_sparsity();
        let config = PruningConfig {
            method: self.method,
            sparsity,
            structured: self.structured,
        };
        let result = prune_weights(weights, &config)?;
        self.step += 1;
        Ok(result)
    }

    /// Whether all pruning steps have been applied.
    pub fn is_done(&self) -> bool {
        self.step >= self.n_steps
    }

    /// Reset the step counter.
    pub fn reset(&mut self) {
        self.step = 0;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Lottery ticket hypothesis
// ────────────────────────────────────────────────────────────────────────────

/// Apply a pruning mask to initial weights to find the "winning ticket".
///
/// Per the Lottery Ticket Hypothesis (Frankle & Carlin, 2019), after training
/// and pruning, resetting the surviving weights to their *initial* values can
/// yield a sparse subnetwork that trains to similar or better accuracy.
///
/// `initial_weights` and `mask` must have the same shape.
/// `mask[[r,c]] == true` → weight is kept; `false` → zeroed.
pub fn find_winning_ticket(
    initial_weights: ArrayView2<f64>,
    mask: ArrayView2<bool>,
) -> Result<Array2<f64>> {
    if initial_weights.shape() != mask.shape() {
        return Err(NeuralError::ShapeMismatch(format!(
            "initial_weights shape {:?} != mask shape {:?}",
            initial_weights.shape(),
            mask.shape()
        )));
    }

    let (nrows, ncols) = (initial_weights.nrows(), initial_weights.ncols());
    let mut ticket = Array2::<f64>::zeros((nrows, ncols));

    for r in 0..nrows {
        for c in 0..ncols {
            if mask[[r, c]] {
                ticket[[r, c]] = initial_weights[[r, c]];
            }
        }
    }

    Ok(ticket)
}

/// Apply an existing mask to (possibly updated) weights without recomputing
/// the pruning schedule.
///
/// Useful for re-zeroing weights that were inadvertently updated by the
/// optimizer after the pruning step.
pub fn apply_mask(weights: ArrayView2<f64>, mask: ArrayView2<bool>) -> Result<Array2<f64>> {
    if weights.shape() != mask.shape() {
        return Err(NeuralError::ShapeMismatch(format!(
            "weights shape {:?} != mask shape {:?}",
            weights.shape(),
            mask.shape()
        )));
    }

    let (nrows, ncols) = (weights.nrows(), weights.ncols());
    let mut out = Array2::<f64>::zeros((nrows, ncols));
    for r in 0..nrows {
        for c in 0..ncols {
            if mask[[r, c]] {
                out[[r, c]] = weights[[r, c]];
            }
        }
    }
    Ok(out)
}

// ────────────────────────────────────────────────────────────────────────────
// Sparsity reporting
// ────────────────────────────────────────────────────────────────────────────

/// Compute the actual sparsity of a weight matrix (fraction of near-zero elements).
///
/// Elements with |w| < `threshold` are counted as zero.
pub fn compute_sparsity(weights: ArrayView2<f64>, threshold: f64) -> f64 {
    let n = weights.len();
    if n == 0 {
        return 0.0;
    }
    let zeros = weights.iter().filter(|&&v| v.abs() < threshold).count();
    zeros as f64 / n as f64
}

/// Print a human-readable sparsity report to a `String`.
pub fn sparsity_report(
    weights: &[ArrayView2<f64>],
    names: &[&str],
    threshold: f64,
) -> Result<String> {
    if weights.len() != names.len() {
        return Err(NeuralError::InvalidArgument(format!(
            "weights.len() ({}) != names.len() ({})",
            weights.len(),
            names.len()
        )));
    }

    let mut lines = Vec::new();
    let mut total_params = 0usize;
    let mut total_zeros = 0usize;

    lines.push("Sparsity Report".to_string());
    lines.push("───────────────────────────────────────────────────────".to_string());

    for (w, name) in weights.iter().zip(names.iter()) {
        let n = w.len();
        let zeros = w.iter().filter(|&&v| v.abs() < threshold).count();
        let sparsity = zeros as f64 / n.max(1) as f64;
        total_params += n;
        total_zeros += zeros;
        lines.push(format!(
            "  {name:30}  params={n:8}  zeros={zeros:8}  sparsity={:.2}%",
            sparsity * 100.0
        ));
    }

    lines.push("───────────────────────────────────────────────────────".to_string());
    let overall = total_zeros as f64 / total_params.max(1) as f64;
    lines.push(format!(
        "  Total:                          params={total_params:8}  zeros={total_zeros:8}  sparsity={:.2}%",
        overall * 100.0
    ));

    Ok(lines.join("\n"))
}

// ────────────────────────────────────────────────────────────────────────────
// Internal utilities
// ────────────────────────────────────────────────────────────────────────────

fn validate_sparsity(s: f64) -> Result<()> {
    if s < 0.0 || s > 1.0 {
        return Err(NeuralError::InvalidArgument(format!(
            "sparsity must be in [0, 1], got {s}"
        )));
    }
    Ok(())
}

/// Deterministic LCG pseudo-random number generator in [0, 1) – used to make
/// RandomPruning deterministic without pulling in an RNG dependency.
fn lcg_f64(seed: u64) -> f64 {
    const A: u64 = 6_364_136_223_846_793_005;
    const C: u64 = 1_442_695_040_888_963_407;
    let x = A.wrapping_mul(seed).wrapping_add(C);
    (x >> 11) as f64 / (1u64 << 53) as f64
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn make_weights() -> Array2<f64> {
        array![
            [0.1, -0.9, 0.2, -0.05, 0.8],
            [-0.3, 0.7, 0.0, 0.6, -0.8],
            [0.5, 0.5, -0.5, -0.5, 0.1]
        ]
    }

    #[test]
    fn test_magnitude_pruning_unstructured() {
        let w = make_weights();
        let config = PruningConfig {
            method: PruningMethod::MagnitudePruning,
            sparsity: 0.4,
            structured: false,
        };
        let (pruned, mask) = prune_weights(w.view(), &config).expect("prune failed");
        let actual_sparsity = mask.iter().filter(|&&m| !m).count() as f64 / mask.len() as f64;
        assert!(
            (actual_sparsity - 0.4).abs() <= 0.2,
            "sparsity={actual_sparsity}"
        );
        // Pruned weights should be zero where mask is false.
        for r in 0..pruned.nrows() {
            for c in 0..pruned.ncols() {
                if !mask[[r, c]] {
                    assert_eq!(pruned[[r, c]], 0.0, "pruned element should be 0");
                }
            }
        }
    }

    #[test]
    fn test_structured_pruning() {
        let w = make_weights();
        let config = PruningConfig {
            method: PruningMethod::MagnitudePruning,
            sparsity: 1.0 / 3.0, // prune 1 of 3 rows
            structured: true,
        };
        let (pruned, mask) = prune_weights(w.view(), &config).expect("prune failed");
        let zero_rows = (0..pruned.nrows())
            .filter(|&r| pruned.row(r).iter().all(|&v| v == 0.0))
            .count();
        assert_eq!(zero_rows, 1, "exactly 1 row should be zeroed");
        let _ = mask;
    }

    #[test]
    fn test_structured_prune_channels() {
        let w = make_weights();
        let (pruned, removed) =
            structured_prune_channels(w.view(), 2).expect("channel prune failed");
        assert_eq!(removed.len(), 2);
        for &r in &removed {
            assert!(
                pruned.row(r).iter().all(|&v| v == 0.0),
                "pruned row {r} should be zero"
            );
        }
    }

    #[test]
    fn test_gradient_pruning() {
        let w = make_weights();
        let g = make_weights(); // reuse as fake gradients
        let (pruned, mask) = gradient_prune_weights(w.view(), g.view(), 0.5)
            .expect("gradient prune failed");
        assert_eq!(pruned.shape(), w.shape());
        assert_eq!(mask.shape(), w.shape());
    }

    #[test]
    fn test_iterative_pruner_schedule() {
        let mut pruner =
            IterativePruner::new(0.9, 10, PruningMethod::MagnitudePruning, false)
                .expect("pruner failed");
        // At step 0 sparsity should be near initial (0).
        assert!((pruner.current_sparsity()).abs() < 0.01);
        for _ in 0..10 {
            let w = make_weights();
            let _ = pruner.step_prune(w.view()).expect("step failed");
        }
        assert!(pruner.is_done());
        // After all steps, current sparsity should be final sparsity.
        assert!((pruner.current_sparsity() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_lottery_ticket() {
        let initial = make_weights();
        let w = array![
            [0.2, -1.1, 0.3, -0.1, 0.9],
            [-0.4, 0.8, 0.1, 0.7, -0.9],
            [0.6, 0.4, -0.6, -0.4, 0.2]
        ];
        let config = PruningConfig {
            method: PruningMethod::MagnitudePruning,
            sparsity: 0.5,
            structured: false,
        };
        let (_, mask) = prune_weights(w.view(), &config).expect("prune failed");
        let ticket =
            find_winning_ticket(initial.view(), mask.view()).expect("ticket failed");

        // Ticket should have the same shape as initial weights.
        assert_eq!(ticket.shape(), initial.shape());
        // Masked-out elements should be zero in the ticket.
        for r in 0..mask.nrows() {
            for c in 0..mask.ncols() {
                if !mask[[r, c]] {
                    assert_eq!(ticket[[r, c]], 0.0);
                } else {
                    assert_eq!(ticket[[r, c]], initial[[r, c]]);
                }
            }
        }
    }

    #[test]
    fn test_apply_mask() {
        let w = make_weights();
        let mut mask = Array2::<bool>::from_elem(w.raw_dim(), true);
        mask[[0, 0]] = false;
        mask[[1, 1]] = false;
        let out = apply_mask(w.view(), mask.view()).expect("apply mask failed");
        assert_eq!(out[[0, 0]], 0.0);
        assert_eq!(out[[1, 1]], 0.0);
        assert_eq!(out[[0, 1]], w[[0, 1]]);
    }

    #[test]
    fn test_compute_sparsity() {
        let w = array![[0.0_f64, 1.0, 0.0], [2.0, 0.0, 3.0]];
        let s = compute_sparsity(w.view(), 1e-9);
        assert!((s - 0.5).abs() < 1e-9, "sparsity={s}");
    }

    #[test]
    fn test_sparsity_report() {
        let w1 = make_weights();
        let w2 = make_weights();
        let views: Vec<ArrayView2<f64>> = vec![w1.view(), w2.view()];
        let report =
            sparsity_report(&views, &["layer1", "layer2"], 1e-9).expect("report failed");
        assert!(report.contains("layer1"));
        assert!(report.contains("layer2"));
        assert!(report.contains("Sparsity Report"));
    }

    #[test]
    fn test_invalid_sparsity() {
        let w = make_weights();
        let config = PruningConfig {
            method: PruningMethod::MagnitudePruning,
            sparsity: 1.5, // invalid
            structured: false,
        };
        assert!(prune_weights(w.view(), &config).is_err());
    }

    #[test]
    fn test_zero_sparsity_keeps_all() {
        let w = make_weights();
        let config = PruningConfig {
            method: PruningMethod::MagnitudePruning,
            sparsity: 0.0,
            structured: false,
        };
        let (pruned, mask) = prune_weights(w.view(), &config).expect("prune failed");
        assert!(mask.iter().all(|&m| m), "all weights should be kept");
    }

    #[test]
    fn test_random_pruning() {
        let w = make_weights();
        let config = PruningConfig {
            method: PruningMethod::RandomPruning,
            sparsity: 0.5,
            structured: false,
        };
        let (_, mask) = prune_weights(w.view(), &config).expect("prune failed");
        let pruned_count = mask.iter().filter(|&&m| !m).count();
        assert!(pruned_count > 0, "should have pruned some weights");
    }
}

// ────────────────────────────────────────────────────────────────────────────
// L2-norm weight pruning
// ────────────────────────────────────────────────────────────────────────────

/// Prune a weight matrix using L2-norm as the importance criterion.
///
/// For unstructured pruning every element's squared value `w²` is the score.
/// For structured (row) pruning, the row's L2 norm `||row||₂` is the score.
///
/// Rows/elements with the *smallest* L2-norm score are removed.
pub fn l2_prune_weights(
    weights: ArrayView2<f64>,
    sparsity: f64,
    structured: bool,
) -> Result<(Array2<f64>, Array2<bool>)> {
    validate_sparsity(sparsity)?;

    if structured {
        let nrows = weights.nrows();
        let ncols = weights.ncols();
        let n_prune = ((sparsity * nrows as f64).round() as usize).min(nrows);

        let row_norms: Vec<f64> = (0..nrows)
            .map(|r| weights.row(r).iter().map(|v| v * v).sum::<f64>().sqrt())
            .collect();

        let mut indexed: Vec<(usize, f64)> = row_norms.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let prune_set: std::collections::HashSet<usize> =
            indexed[..n_prune].iter().map(|&(i, _)| i).collect();

        let mut pruned = weights.to_owned();
        let mut mask = Array2::<bool>::from_elem((nrows, ncols), true);

        for r in prune_set {
            for c in 0..ncols {
                pruned[[r, c]] = 0.0;
                mask[[r, c]] = false;
            }
        }

        Ok((pruned, mask))
    } else {
        let n = weights.len();
        let n_prune = ((sparsity * n as f64).round() as usize).min(n);

        // Importance score = w² (L2 contribution).
        let scores: Vec<f64> = weights.iter().map(|v| v * v).collect();

        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = if n_prune == 0 { -1.0 } else { sorted[n_prune - 1] };

        let (nrows, ncols) = (weights.nrows(), weights.ncols());
        let mut pruned = Array2::<f64>::zeros((nrows, ncols));
        let mut mask = Array2::<bool>::from_elem((nrows, ncols), true);

        let mut flat = 0usize;
        for r in 0..nrows {
            for c in 0..ncols {
                if scores[flat] <= threshold {
                    mask[[r, c]] = false;
                } else {
                    pruned[[r, c]] = weights[[r, c]];
                }
                flat += 1;
            }
        }

        Ok((pruned, mask))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Pruning scheduler — decoupled schedule from the pruner
// ────────────────────────────────────────────────────────────────────────────

/// Schedule type for a [`PruningScheduler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruningScheduleType {
    /// Polynomial decay:  s(t) = s_f + (s_i - s_f)·(1 - t/T)^p
    Polynomial,
    /// Linear interpolation from `s_initial` to `s_final`.
    Linear,
    /// Cosine annealing:  s(t) = s_f + (s_i - s_f)·½·(1 + cos(π·t/T))
    Cosine,
    /// Step-wise: sparsity increases by a fixed delta every `step_interval` steps.
    StepWise,
    /// Cubic polynomial (exponent = 3 always, classic "To Prune, or Not to Prune").
    Cubic,
}

/// Standalone pruning scheduler that separates the *schedule* concern from the
/// *pruner* concern.  At each training step the caller asks for the target
/// sparsity, then applies it with whichever pruner they prefer.
///
/// # Example
///
/// ```rust
/// use scirs2_neural::training::pruning::{PruningScheduler, PruningScheduleType};
///
/// let mut sched = PruningScheduler::new(0.0, 0.9, 100, PruningScheduleType::Cosine)
///     .expect("valid scheduler");
///
/// // At step 0 we start at ~s_initial.
/// let s0 = sched.current_sparsity();
/// assert!(s0 >= 0.0 && s0 <= 1.0);
///
/// // Advance by 50 steps.
/// for _ in 0..50 { sched.advance(); }
/// let s50 = sched.current_sparsity();
/// assert!(s50 > s0, "sparsity should increase");
/// ```
#[derive(Debug, Clone)]
pub struct PruningScheduler {
    /// Starting sparsity (often 0.0).
    pub initial_sparsity: f64,
    /// Target sparsity at the end of the schedule.
    pub final_sparsity: f64,
    /// Total number of steps in the schedule.
    pub total_steps: usize,
    /// Current step.
    pub current_step: usize,
    /// Schedule type.
    pub schedule_type: PruningScheduleType,
    /// Polynomial exponent (only used for `Polynomial`; default 3.0).
    pub exponent: f64,
    /// Number of steps between increases for `StepWise` schedule.
    pub step_interval: usize,
    /// Number of warmup steps before pruning begins.
    pub warmup_steps: usize,
}

impl PruningScheduler {
    /// Create a new pruning scheduler.
    ///
    /// # Errors
    ///
    /// Returns `Err` if either sparsity value is outside `[0, 1]` or
    /// `total_steps` is 0.
    pub fn new(
        initial_sparsity: f64,
        final_sparsity: f64,
        total_steps: usize,
        schedule_type: PruningScheduleType,
    ) -> Result<Self> {
        validate_sparsity(initial_sparsity)?;
        validate_sparsity(final_sparsity)?;
        if total_steps == 0 {
            return Err(NeuralError::InvalidArgument(
                "total_steps must be > 0".to_string(),
            ));
        }
        Ok(Self {
            initial_sparsity,
            final_sparsity,
            total_steps,
            current_step: 0,
            schedule_type,
            exponent: 3.0,
            step_interval: 1,
            warmup_steps: 0,
        })
    }

    /// Set the polynomial exponent (relevant for `Polynomial` and `Cubic` schedules).
    pub fn with_exponent(mut self, exp: f64) -> Self {
        self.exponent = exp;
        self
    }

    /// Set the interval between step-wise increases.
    pub fn with_step_interval(mut self, interval: usize) -> Self {
        self.step_interval = interval.max(1);
        self
    }

    /// Set a number of warmup steps during which sparsity stays at `initial_sparsity`.
    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup_steps = warmup;
        self
    }

    /// Advance the internal step counter by 1.
    pub fn advance(&mut self) {
        if self.current_step < self.total_steps {
            self.current_step += 1;
        }
    }

    /// Reset the schedule back to step 0.
    pub fn reset(&mut self) {
        self.current_step = 0;
    }

    /// Return `true` if the schedule has reached `total_steps`.
    pub fn is_done(&self) -> bool {
        self.current_step >= self.total_steps
    }

    /// Compute the target sparsity for the **current** step.
    ///
    /// If `current_step < warmup_steps`, returns `initial_sparsity`.
    /// Once `current_step >= total_steps`, returns `final_sparsity`.
    pub fn current_sparsity(&self) -> f64 {
        if self.current_step < self.warmup_steps {
            return self.initial_sparsity;
        }
        if self.current_step >= self.total_steps {
            return self.final_sparsity;
        }

        let s_i = self.initial_sparsity;
        let s_f = self.final_sparsity;
        // Effective progress after warmup.
        let effective_total = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let effective_step = self.current_step - self.warmup_steps;
        let t = effective_step as f64;
        let n = effective_total as f64;

        match self.schedule_type {
            PruningScheduleType::Polynomial => {
                s_f + (s_i - s_f) * (1.0 - t / n).powf(self.exponent)
            }
            PruningScheduleType::Cubic => s_f + (s_i - s_f) * (1.0 - t / n).powi(3),
            PruningScheduleType::Linear => s_i + (s_f - s_i) * (t / n),
            PruningScheduleType::Cosine => {
                s_f + (s_i - s_f) * 0.5 * (1.0 + (std::f64::consts::PI * t / n).cos())
            }
            PruningScheduleType::StepWise => {
                let interval = self.step_interval.max(1) as f64;
                let n_increments = (n / interval).ceil() as usize;
                if n_increments == 0 {
                    return s_f;
                }
                let completed =
                    ((effective_step as f64 / interval).floor() as usize).min(n_increments);
                let delta = (s_f - s_i) / n_increments as f64;
                (s_i + delta * completed as f64).clamp(s_i.min(s_f), s_i.max(s_f))
            }
        }
    }

    /// Convenience: advance then return the new sparsity.
    pub fn step_sparsity(&mut self) -> f64 {
        self.advance();
        self.current_sparsity()
    }
}
