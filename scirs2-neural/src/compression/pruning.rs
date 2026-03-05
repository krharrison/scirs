//! Weight pruning for model compression.
//!
//! Provides magnitude-based unstructured pruning, structured pruning of
//! neurons/filters, gradual pruning schedules, and binary mask tracking.

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};

// ─────────────────────────────────────────────────────────────────────────────
// PruningMask
// ─────────────────────────────────────────────────────────────────────────────

/// Binary mask indicating which weights are *active* (true) vs. pruned (false).
#[derive(Debug, Clone)]
pub struct PruningMask {
    /// `true` → weight is kept; `false` → weight is pruned.
    pub mask: Array2<bool>,
}

impl PruningMask {
    /// Create a new all-ones (unpruned) mask with the given shape.
    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            mask: Array2::from_elem((rows, cols), true),
        }
    }

    /// Total number of entries.
    pub fn total(&self) -> usize {
        self.mask.len()
    }

    /// Number of active (non-pruned) entries.
    pub fn active(&self) -> usize {
        self.mask.iter().filter(|&&v| v).count()
    }

    /// Number of pruned entries.
    pub fn pruned(&self) -> usize {
        self.total() - self.active()
    }

    /// Fraction of pruned weights in `[0, 1]`.
    pub fn sparsity(&self) -> f64 {
        if self.total() == 0 {
            return 0.0;
        }
        self.pruned() as f64 / self.total() as f64
    }

    /// Apply this mask to a weight matrix element-wise (zero-out pruned weights).
    pub fn apply(&self, weights: &Array2<f32>) -> Result<Array2<f32>> {
        if weights.shape() != self.mask.shape() {
            return Err(NeuralError::InvalidArchitecture(format!(
                "weight shape {:?} does not match mask shape {:?}",
                weights.shape(),
                self.mask.shape()
            )));
        }
        let result = weights.mapv(|_| 0.0_f32)
            + &weights
                .indexed_iter()
                .fold(Array2::zeros(weights.raw_dim()), |mut acc, ((r, c), &v)| {
                    if self.mask[(r, c)] {
                        acc[(r, c)] = v;
                    }
                    acc
                });
        Ok(result)
    }

    /// Merge two masks with logical AND (intersection of kept weights).
    pub fn intersect(&self, other: &PruningMask) -> Result<PruningMask> {
        if self.mask.shape() != other.mask.shape() {
            return Err(NeuralError::InvalidArchitecture(
                "cannot intersect masks of different shapes".into(),
            ));
        }
        let combined = self
            .mask
            .iter()
            .zip(other.mask.iter())
            .map(|(&a, &b)| a && b)
            .collect::<Vec<bool>>();
        let shape = (self.mask.nrows(), self.mask.ncols());
        Ok(PruningMask {
            mask: Array2::from_shape_vec(shape, combined).map_err(|e| {
                NeuralError::InvalidArchitecture(format!("mask reshape error: {e}"))
            })?,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Free functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the fraction of near-zero weights in a weight matrix.
///
/// A weight is considered zero if `|w| < 1e-8`.
pub fn compute_sparsity(weights: &Array2<f32>) -> f64 {
    if weights.len() == 0 {
        return 0.0;
    }
    let zeros = weights.iter().filter(|&&v| v.abs() < 1e-8_f32).count();
    zeros as f64 / weights.len() as f64
}

/// Magnitude-based unstructured pruning.
///
/// Sets the bottom `sparsity` fraction of weights (by absolute value) to zero
/// and returns both the pruned weight matrix and the corresponding mask.
///
/// # Errors
/// Returns an error if `sparsity` is not in `[0, 1]`.
pub fn prune_magnitude(
    weights: &Array2<f32>,
    sparsity: f64,
) -> Result<(Array2<f32>, PruningMask)> {
    if !(0.0..=1.0).contains(&sparsity) {
        return Err(NeuralError::InvalidArchitecture(format!(
            "sparsity must be in [0,1], got {sparsity}"
        )));
    }

    let (nrows, ncols) = (weights.nrows(), weights.ncols());
    let n = weights.len();

    // Collect (absolute value, flat index) and sort ascending.
    let mut abs_vals: Vec<(f32, usize)> = weights
        .iter()
        .enumerate()
        .map(|(i, &v)| (v.abs(), i))
        .collect();
    abs_vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_prune = (sparsity * n as f64).round() as usize;

    let mut mask_flat = vec![true; n];
    for &(_, idx) in abs_vals.iter().take(n_prune) {
        mask_flat[idx] = false;
    }

    let flat_weights: Vec<f32> = weights
        .iter()
        .zip(mask_flat.iter())
        .map(|(&w, &keep)| if keep { w } else { 0.0_f32 })
        .collect();

    let pruned = Array2::from_shape_vec((nrows, ncols), flat_weights).map_err(|e| {
        NeuralError::InvalidArchitecture(format!("pruned weight reshape error: {e}"))
    })?;
    let mask_arr = Array2::from_shape_vec((nrows, ncols), mask_flat).map_err(|e| {
        NeuralError::InvalidArchitecture(format!("mask reshape error: {e}"))
    })?;

    Ok((pruned, PruningMask { mask: mask_arr }))
}

/// Structured pruning: remove entire rows (axis=0) or columns (axis=1).
///
/// Returns the pruned weight matrix (with zeroed-out rows/columns) and the
/// indices of the kept rows/columns.
///
/// # Errors
/// Returns an error if `sparsity` is not in `[0, 1]` or `axis` > 1.
pub fn prune_structured(
    weights: &Array2<f32>,
    sparsity: f64,
    axis: usize,
) -> Result<(Array2<f32>, Vec<usize>)> {
    if !(0.0..=1.0).contains(&sparsity) {
        return Err(NeuralError::InvalidArchitecture(format!(
            "sparsity must be in [0,1], got {sparsity}"
        )));
    }
    if axis > 1 {
        return Err(NeuralError::InvalidArchitecture(format!(
            "axis must be 0 or 1 for 2-D weight matrix, got {axis}"
        )));
    }

    let n_slices = weights.shape()[axis];
    let n_prune = (sparsity * n_slices as f64).round() as usize;

    // Compute L2 norm of each row/column.
    let norms: Vec<(f32, usize)> = (0..n_slices)
        .map(|i| {
            let norm = if axis == 0 {
                weights.row(i).iter().map(|&v| v * v).sum::<f32>().sqrt()
            } else {
                weights.column(i).iter().map(|&v| v * v).sum::<f32>().sqrt()
            };
            (norm, i)
        })
        .collect();

    let mut sorted_norms = norms.clone();
    sorted_norms.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Indices to prune (smallest norms).
    let pruned_indices: std::collections::HashSet<usize> = sorted_norms
        .iter()
        .take(n_prune)
        .map(|&(_, idx)| idx)
        .collect();

    let kept: Vec<usize> = (0..n_slices)
        .filter(|i| !pruned_indices.contains(i))
        .collect();

    // Build output with zeroed rows/columns.
    let (nrows, ncols) = (weights.nrows(), weights.ncols());
    let mut out = Array2::zeros((nrows, ncols));
    for (&keep_idx, (r, c, v)) in weights.indexed_iter().map(|((r, c), v)| (r, c, v)).enumerate()
    {
        // Re-bind: indexed_iter gives ((row, col), val)
        let _ = keep_idx;
        let slice_idx = if axis == 0 { r } else { c };
        if !pruned_indices.contains(&slice_idx) {
            out[(r, c)] = *v;
        }
    }
    // Redo iteration correctly.
    let mut out = Array2::zeros((nrows, ncols));
    for ((r, c), &v) in weights.indexed_iter() {
        let slice_idx = if axis == 0 { r } else { c };
        if !pruned_indices.contains(&slice_idx) {
            out[(r, c)] = v;
        }
    }

    Ok((out, kept))
}

// ─────────────────────────────────────────────────────────────────────────────
// MagnitudePruner
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for magnitude-based pruning.
#[derive(Debug, Clone)]
pub struct MagnitudePruner {
    /// Target sparsity in `[0, 1]`.
    pub target_sparsity: f64,
    /// Absolute-value threshold below which weights are pruned.
    /// If `None`, the threshold is derived from `target_sparsity`.
    pub threshold: Option<f32>,
}

impl MagnitudePruner {
    /// Create a new `MagnitudePruner` with a sparsity target.
    pub fn new(target_sparsity: f64) -> Self {
        Self {
            target_sparsity,
            threshold: None,
        }
    }

    /// Create a `MagnitudePruner` with a fixed threshold.
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            target_sparsity: 0.0,
            threshold: Some(threshold),
        }
    }

    /// Prune weights and return `(pruned_weights, mask)`.
    pub fn prune(&self, weights: &Array2<f32>) -> Result<(Array2<f32>, PruningMask)> {
        if let Some(thresh) = self.threshold {
            // Threshold-based mode.
            let (nrows, ncols) = (weights.nrows(), weights.ncols());
            let (flat_w, flat_m): (Vec<f32>, Vec<bool>) = weights
                .iter()
                .map(|&v| {
                    let keep = v.abs() >= thresh;
                    (if keep { v } else { 0.0_f32 }, keep)
                })
                .unzip();
            let pruned = Array2::from_shape_vec((nrows, ncols), flat_w).map_err(|e| {
                NeuralError::InvalidArchitecture(format!("reshape error: {e}"))
            })?;
            let mask = Array2::from_shape_vec((nrows, ncols), flat_m).map_err(|e| {
                NeuralError::InvalidArchitecture(format!("reshape error: {e}"))
            })?;
            Ok((pruned, PruningMask { mask }))
        } else {
            prune_magnitude(weights, self.target_sparsity)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StructuredPruner
// ─────────────────────────────────────────────────────────────────────────────

/// Structured pruner that removes complete neurons (rows) or filters (columns).
#[derive(Debug, Clone)]
pub struct StructuredPruner {
    /// Target sparsity in `[0, 1]`.
    pub target_sparsity: f64,
    /// Axis along which to prune: 0 = rows (output neurons), 1 = columns (input neurons).
    pub axis: usize,
}

impl StructuredPruner {
    /// Create a new structured pruner.
    pub fn new(target_sparsity: f64, axis: usize) -> Self {
        Self {
            target_sparsity,
            axis,
        }
    }

    /// Prune the weight matrix and return `(pruned_weights, kept_indices)`.
    pub fn prune(&self, weights: &Array2<f32>) -> Result<(Array2<f32>, Vec<usize>)> {
        prune_structured(weights, self.target_sparsity, self.axis)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GradualPruning
// ─────────────────────────────────────────────────────────────────────────────

/// Schedule shape for gradual pruning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PruningSchedule {
    /// Linearly increase sparsity from `initial_sparsity` to `target_sparsity`.
    Linear,
    /// Polynomial increase: sparsity grows as `(step/total_steps)^exponent`.
    Polynomial { exponent: f64 },
    /// Cubic schedule (polynomial with exponent = 3).
    Cubic,
}

/// Gradual pruning that ramps sparsity over a number of training steps.
#[derive(Debug, Clone)]
pub struct GradualPruning {
    /// Sparsity at step 0 (typically 0).
    pub initial_sparsity: f64,
    /// Desired sparsity after `total_steps`.
    pub target_sparsity: f64,
    /// Total number of pruning steps.
    pub total_steps: usize,
    /// Frequency (in steps) at which pruning is applied.
    pub pruning_frequency: usize,
    /// Schedule shape.
    pub schedule: PruningSchedule,
    /// Current step counter.
    current_step: usize,
}

impl GradualPruning {
    /// Create a new gradual pruning schedule.
    ///
    /// # Errors
    /// Returns an error if sparsities are out of range or inconsistent.
    pub fn new(
        initial_sparsity: f64,
        target_sparsity: f64,
        total_steps: usize,
        pruning_frequency: usize,
        schedule: PruningSchedule,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&initial_sparsity) {
            return Err(NeuralError::InvalidArchitecture(format!(
                "initial_sparsity must be in [0,1], got {initial_sparsity}"
            )));
        }
        if !(0.0..=1.0).contains(&target_sparsity) {
            return Err(NeuralError::InvalidArchitecture(format!(
                "target_sparsity must be in [0,1], got {target_sparsity}"
            )));
        }
        if initial_sparsity > target_sparsity {
            return Err(NeuralError::InvalidArchitecture(
                "initial_sparsity must be <= target_sparsity".into(),
            ));
        }
        if total_steps == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "total_steps must be > 0".into(),
            ));
        }
        if pruning_frequency == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "pruning_frequency must be > 0".into(),
            ));
        }
        Ok(Self {
            initial_sparsity,
            target_sparsity,
            total_steps,
            pruning_frequency,
            schedule,
            current_step: 0,
        })
    }

    /// Compute the target sparsity for a given step.
    pub fn sparsity_at_step(&self, step: usize) -> f64 {
        let t = step.min(self.total_steps) as f64 / self.total_steps as f64;
        let delta = self.target_sparsity - self.initial_sparsity;
        let fraction = match self.schedule {
            PruningSchedule::Linear => t,
            PruningSchedule::Polynomial { exponent } => t.powf(exponent),
            PruningSchedule::Cubic => t.powi(3),
        };
        self.initial_sparsity + delta * fraction
    }

    /// Advance one step; if this is a pruning step, apply pruning and return
    /// the new mask. Returns `None` on non-pruning steps.
    pub fn step(
        &mut self,
        weights: &Array2<f32>,
    ) -> Result<Option<(Array2<f32>, PruningMask)>> {
        self.current_step += 1;
        if self.current_step % self.pruning_frequency == 0 {
            let sparsity = self.sparsity_at_step(self.current_step);
            let result = prune_magnitude(weights, sparsity)?;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Current step count.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Reset step counter.
    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_weights() -> Array2<f32> {
        Array2::from_shape_vec(
            (3, 4),
            vec![
                0.1, -0.5, 0.3, -0.9, 0.8, 0.2, -0.4, 0.6, -0.7, 0.05, 0.95, -0.15,
            ],
        )
        .expect("static test data")
    }

    #[test]
    fn test_compute_sparsity_all_nonzero() {
        let w = sample_weights();
        let s = compute_sparsity(&w);
        assert!(s < 0.01, "no zeros expected, got sparsity={s}");
    }

    #[test]
    fn test_compute_sparsity_with_zeros() {
        let w = Array2::from_shape_vec(
            (2, 4),
            vec![0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0],
        )
        .expect("static test data");
        let s = compute_sparsity(&w);
        // 4 zeros out of 8
        assert!((s - 0.5).abs() < 0.01, "expected 0.5, got {s}");
    }

    #[test]
    fn test_prune_magnitude_zero_sparsity() {
        let w = sample_weights();
        let (pruned, mask) = prune_magnitude(&w, 0.0).expect("prune_magnitude failed");
        assert_eq!(pruned, w);
        assert_eq!(mask.sparsity(), 0.0);
    }

    #[test]
    fn test_prune_magnitude_fifty_percent() {
        let w = sample_weights();
        let (pruned, mask) = prune_magnitude(&w, 0.5).expect("prune_magnitude failed");
        let actual = compute_sparsity(&pruned);
        // Should be approximately 50 % sparse (6 out of 12 zeroed).
        assert!(actual >= 0.45 && actual <= 0.55, "expected ~0.5, got {actual}");
        assert!((mask.sparsity() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_prune_magnitude_full_sparsity() {
        let w = sample_weights();
        let (pruned, mask) = prune_magnitude(&w, 1.0).expect("prune_magnitude failed");
        assert!(pruned.iter().all(|&v| v == 0.0), "all weights should be zero");
        assert!((mask.sparsity() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_prune_magnitude_invalid_sparsity() {
        let w = sample_weights();
        assert!(prune_magnitude(&w, 1.5).is_err());
        assert!(prune_magnitude(&w, -0.1).is_err());
    }

    #[test]
    fn test_prune_structured_rows() {
        let w = Array2::from_shape_vec(
            (4, 3),
            vec![
                0.01, 0.01, 0.01, // row 0 – tiny norm
                1.0, 2.0, 3.0, // row 1 – large norm
                0.5, 0.5, 0.5, // row 2 – medium norm
                4.0, 5.0, 6.0, // row 3 – largest norm
            ],
        )
        .expect("static test data");
        let (pruned, kept) = prune_structured(&w, 0.25, 0).expect("prune_structured failed");
        // 25% of 4 rows = 1 row pruned (row 0 has smallest norm).
        assert_eq!(kept.len(), 3);
        assert!(!kept.contains(&0), "row 0 should be pruned");
        // Row 0 should be zero in the output.
        assert!(pruned.row(0).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_prune_structured_cols() {
        let w = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.01, 1.0, 0.5, 4.0,
                0.01, 2.0, 0.5, 5.0,
                0.01, 3.0, 0.5, 6.0,
            ],
        )
        .expect("static test data");
        let (pruned, kept) = prune_structured(&w, 0.25, 1).expect("prune_structured failed");
        assert_eq!(kept.len(), 3);
        assert!(!kept.contains(&0));
        assert!(pruned.column(0).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_pruning_mask_apply() {
        let w = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("static test data");
        let mut mask = PruningMask::ones(2, 2);
        mask.mask[(0, 0)] = false;
        mask.mask[(1, 1)] = false;
        let applied = mask.apply(&w).expect("apply failed");
        assert_eq!(applied[(0, 0)], 0.0);
        assert_eq!(applied[(0, 1)], 2.0);
        assert_eq!(applied[(1, 0)], 3.0);
        assert_eq!(applied[(1, 1)], 0.0);
    }

    #[test]
    fn test_magnitude_pruner() {
        let w = sample_weights();
        let pruner = MagnitudePruner::new(0.5);
        let (pruned, mask) = pruner.prune(&w).expect("prune failed");
        assert!((mask.sparsity() - 0.5).abs() < 0.1);
        let _ = pruned;
    }

    #[test]
    fn test_gradual_pruning_schedule_linear() {
        let mut gp = GradualPruning::new(0.0, 0.9, 100, 10, PruningSchedule::Linear)
            .expect("GradualPruning::new failed");
        // At step 50 sparsity should be ~0.45.
        let s50 = gp.sparsity_at_step(50);
        assert!((s50 - 0.45).abs() < 0.01, "expected 0.45, got {s50}");
        let s100 = gp.sparsity_at_step(100);
        assert!((s100 - 0.9).abs() < 0.01, "expected 0.9, got {s100}");
        let _ = gp.reset();
    }

    #[test]
    fn test_gradual_pruning_schedule_cubic() {
        let gp = GradualPruning::new(0.0, 0.9, 100, 10, PruningSchedule::Cubic)
            .expect("GradualPruning::new failed");
        // Cubic schedule is slower at first – at 50% of steps sparsity < 0.45.
        let s50 = gp.sparsity_at_step(50);
        assert!(s50 < 0.45, "cubic should be slow initially, got {s50}");
    }

    #[test]
    fn test_gradual_pruning_step() {
        let w = sample_weights();
        let mut gp = GradualPruning::new(0.0, 0.6, 10, 2, PruningSchedule::Linear)
            .expect("GradualPruning::new failed");
        // Step 1: not a pruning step (frequency=2).
        let r1 = gp.step(&w).expect("step failed");
        assert!(r1.is_none());
        // Step 2: pruning step.
        let r2 = gp.step(&w).expect("step failed");
        assert!(r2.is_some());
    }
}
