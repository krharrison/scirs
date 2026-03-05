//! Preference Data Structures for RLHF
//!
//! This module provides data structures and utilities for working with human
//! preference data used in Reinforcement Learning from Human Feedback (RLHF),
//! including Direct Preference Optimization (DPO) and reward model training.
//!
//! # Overview
//!
//! Human preference data typically consists of:
//! - A **prompt** (context / instruction)
//! - A **chosen** response that the annotator preferred
//! - A **rejected** response that the annotator disliked
//!
//! This module provides:
//! - [`Preference`] — a single (prompt, chosen, rejected) triple with optional
//!   metadata (score, annotator id, etc.)
//! - [`PreferenceDataset`] — a collection of preferences with batch sampling,
//!   shuffling, and splitting utilities
//! - [`reward_from_preferences`] — converts binary preference labels to a
//!   scalar reward signal suitable for Bradley-Terry training
//! - [`PreferenceMiniBatch`] — a collated mini-batch of preferences ready for
//!   loss computation
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::preference_data::{
//!     Preference, PreferenceDataset, reward_from_preferences,
//! };
//! use scirs2_core::ndarray::Array1;
//!
//! let pref = Preference {
//!     prompt_ids: vec![1u32, 2, 3],
//!     chosen_ids: vec![4u32, 5],
//!     rejected_ids: vec![6u32, 7, 8],
//!     chosen_score: Some(1.0),
//!     rejected_score: Some(0.0),
//!     annotator_id: None,
//!     metadata: std::collections::HashMap::new(),
//! };
//!
//! let mut dataset = PreferenceDataset::new();
//! dataset.push(pref);
//! assert_eq!(dataset.len(), 1);
//!
//! let batch = dataset.sample_batch(1, None).expect("batch ok");
//! assert_eq!(batch.batch_size(), 1);
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

// ============================================================================
// Preference record
// ============================================================================

/// A single human preference example.
///
/// The `chosen` and `rejected` fields are stored as token-ID sequences (or any
/// integer sequence identifying the response).  Dense embedding / log-prob
/// tensors are stored separately in [`PreferenceMiniBatch`] after encoding.
#[derive(Debug, Clone)]
pub struct Preference {
    /// Token IDs of the shared prompt / context.
    pub prompt_ids: Vec<u32>,
    /// Token IDs of the preferred (chosen) response.
    pub chosen_ids: Vec<u32>,
    /// Token IDs of the dispreferred (rejected) response.
    pub rejected_ids: Vec<u32>,
    /// Optional quality score for the chosen response (higher = better).
    pub chosen_score: Option<f64>,
    /// Optional quality score for the rejected response.
    pub rejected_score: Option<f64>,
    /// Optional annotator identifier.
    pub annotator_id: Option<u32>,
    /// Free-form metadata key–value pairs.
    pub metadata: HashMap<String, String>,
}

impl Preference {
    /// Construct a minimal preference example without score or metadata.
    pub fn new(
        prompt_ids: Vec<u32>,
        chosen_ids: Vec<u32>,
        rejected_ids: Vec<u32>,
    ) -> Self {
        Self {
            prompt_ids,
            chosen_ids,
            rejected_ids,
            chosen_score: None,
            rejected_score: None,
            annotator_id: None,
            metadata: HashMap::new(),
        }
    }

    /// Total token length (prompt + chosen + rejected).
    pub fn total_length(&self) -> usize {
        self.prompt_ids.len() + self.chosen_ids.len() + self.rejected_ids.len()
    }

    /// Length of the chosen response (excluding prompt).
    pub fn chosen_length(&self) -> usize {
        self.chosen_ids.len()
    }

    /// Length of the rejected response (excluding prompt).
    pub fn rejected_length(&self) -> usize {
        self.rejected_ids.len()
    }

    /// Return `true` when both quality scores are present and chosen > rejected.
    pub fn is_consistent(&self) -> bool {
        match (self.chosen_score, self.rejected_score) {
            (Some(c), Some(r)) => c > r,
            _ => true, // No scores → assume consistent
        }
    }
}

// ============================================================================
// Mini-batch of preferences
// ============================================================================

/// A collated mini-batch of preference examples.
///
/// All token-ID sequences are **padded** to the maximum length within the
/// batch (with `pad_id` filling shorter sequences).
#[derive(Debug, Clone)]
pub struct PreferenceMiniBatch {
    /// Padded prompt token IDs `[batch, max_prompt_len]`.
    pub prompt_ids: Array2<u32>,
    /// Padded chosen token IDs `[batch, max_chosen_len]`.
    pub chosen_ids: Array2<u32>,
    /// Padded rejected token IDs `[batch, max_rejected_len]`.
    pub rejected_ids: Array2<u32>,
    /// Attention masks for chosen sequences (1=real, 0=padding) `[batch, max_chosen_len]`.
    pub chosen_mask: Array2<u8>,
    /// Attention masks for rejected sequences `[batch, max_rejected_len]`.
    pub rejected_mask: Array2<u8>,
    /// Original chosen sequence lengths before padding.
    pub chosen_lengths: Vec<usize>,
    /// Original rejected sequence lengths before padding.
    pub rejected_lengths: Vec<usize>,
    /// Per-sample chosen quality scores (NaN when not provided).
    pub chosen_scores: Array1<f64>,
    /// Per-sample rejected quality scores (NaN when not provided).
    pub rejected_scores: Array1<f64>,
}

impl PreferenceMiniBatch {
    /// Number of preference pairs in this batch.
    pub fn batch_size(&self) -> usize {
        self.chosen_ids.nrows()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.batch_size() == 0
    }
}

// ============================================================================
// Preference Dataset
// ============================================================================

/// A dataset of human preference triples.
///
/// Supports batch sampling, train/validation splitting, and conversion to
/// log-probability tensors for DPO / reward-model training.
#[derive(Debug, Clone)]
pub struct PreferenceDataset {
    /// Stored preference examples.
    samples: Vec<Preference>,
    /// Token ID used for padding sequences in a batch.
    pub pad_id: u32,
}

impl PreferenceDataset {
    /// Create a new empty dataset with the default pad token (`0`).
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            pad_id: 0,
        }
    }

    /// Create a new empty dataset with a custom pad token.
    pub fn with_pad_id(pad_id: u32) -> Self {
        Self {
            samples: Vec::new(),
            pad_id,
        }
    }

    /// Append a preference example to the dataset.
    pub fn push(&mut self, pref: Preference) {
        self.samples.push(pref);
    }

    /// Extend the dataset from an iterator.
    pub fn extend<I: IntoIterator<Item = Preference>>(&mut self, iter: I) {
        self.samples.extend(iter);
    }

    /// Total number of examples in the dataset.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Return `true` when the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Access the i-th preference (bounds-checked).
    pub fn get(&self, idx: usize) -> Option<&Preference> {
        self.samples.get(idx)
    }

    /// Return an iterator over all samples.
    pub fn iter(&self) -> impl Iterator<Item = &Preference> {
        self.samples.iter()
    }

    /// Return the number of consistent preferences (chosen_score > rejected_score).
    pub fn consistency_count(&self) -> usize {
        self.samples.iter().filter(|p| p.is_consistent()).count()
    }

    /// Sample a mini-batch of `batch_size` preferences.
    ///
    /// # Arguments
    /// - `batch_size` – number of preference pairs in the batch
    /// - `seed`       – optional deterministic RNG seed for reproducibility
    ///                  (`None` → sequential from the start)
    ///
    /// # Returns
    /// A collated [`PreferenceMiniBatch`].
    pub fn sample_batch(
        &self,
        batch_size: usize,
        seed: Option<u64>,
    ) -> Result<PreferenceMiniBatch> {
        if self.samples.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "sample_batch: dataset is empty".to_string(),
            ));
        }
        let n = self.samples.len();
        let actual_batch = batch_size.min(n);
        if actual_batch == 0 {
            return Err(NeuralError::InvalidArgument(
                "sample_batch: batch_size must be > 0".to_string(),
            ));
        }

        // Generate indices (deterministic shuffle when seed is provided)
        let indices: Vec<usize> = if let Some(s) = seed {
            deterministic_shuffle(n, s)
                .into_iter()
                .take(actual_batch)
                .collect()
        } else {
            (0..actual_batch).collect()
        };

        self.collate(&indices)
    }

    /// Return a sequential mini-batch starting at `offset`.
    ///
    /// Wraps around if `offset + batch_size > len()`.
    pub fn sequential_batch(
        &self,
        offset: usize,
        batch_size: usize,
    ) -> Result<PreferenceMiniBatch> {
        if self.samples.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "sequential_batch: dataset is empty".to_string(),
            ));
        }
        let n = self.samples.len();
        let actual_batch = batch_size.min(n);
        let indices: Vec<usize> = (0..actual_batch)
            .map(|i| (offset + i) % n)
            .collect();
        self.collate(&indices)
    }

    /// Split the dataset into train and validation subsets.
    ///
    /// # Arguments
    /// - `val_fraction` – fraction of data to use for validation `(0, 1)`
    /// - `seed`         – optional shuffle seed
    ///
    /// # Returns
    /// `(train_dataset, val_dataset)` tuple.
    pub fn train_val_split(
        &self,
        val_fraction: f64,
        seed: Option<u64>,
    ) -> Result<(PreferenceDataset, PreferenceDataset)> {
        if !(0.0 < val_fraction && val_fraction < 1.0) {
            return Err(NeuralError::InvalidArgument(format!(
                "train_val_split: val_fraction must be in (0, 1), got {val_fraction}"
            )));
        }
        let n = self.samples.len();
        if n < 2 {
            return Err(NeuralError::InvalidArgument(
                "train_val_split: dataset must have at least 2 examples".to_string(),
            ));
        }

        let indices = if let Some(s) = seed {
            deterministic_shuffle(n, s)
        } else {
            (0..n).collect()
        };

        let val_count = ((n as f64 * val_fraction).round() as usize).max(1).min(n - 1);
        let val_indices = &indices[..val_count];
        let train_indices = &indices[val_count..];

        let mut train = PreferenceDataset::with_pad_id(self.pad_id);
        for &i in train_indices {
            train.push(self.samples[i].clone());
        }

        let mut val = PreferenceDataset::with_pad_id(self.pad_id);
        for &i in val_indices {
            val.push(self.samples[i].clone());
        }

        Ok((train, val))
    }

    /// Return dataset statistics.
    pub fn stats(&self) -> PreferenceDatasetStats {
        let n = self.samples.len();
        if n == 0 {
            return PreferenceDatasetStats::default();
        }

        let total_chosen_len: usize = self.samples.iter().map(|p| p.chosen_length()).sum();
        let total_rejected_len: usize = self.samples.iter().map(|p| p.rejected_length()).sum();
        let total_prompt_len: usize = self.samples.iter().map(|p| p.prompt_ids.len()).sum();
        let consistent = self.consistency_count();

        PreferenceDatasetStats {
            num_samples: n,
            avg_prompt_len: total_prompt_len as f64 / n as f64,
            avg_chosen_len: total_chosen_len as f64 / n as f64,
            avg_rejected_len: total_rejected_len as f64 / n as f64,
            consistency_rate: consistent as f64 / n as f64,
        }
    }

    // -----------------------------------------------------------------------
    // Internal: collate a list of indices into a mini-batch
    // -----------------------------------------------------------------------
    fn collate(&self, indices: &[usize]) -> Result<PreferenceMiniBatch> {
        let batch = indices.len();
        if batch == 0 {
            return Err(NeuralError::InvalidArgument(
                "collate: indices must be non-empty".to_string(),
            ));
        }

        let max_chosen_len = indices
            .iter()
            .map(|&i| self.samples[i].chosen_length())
            .max()
            .unwrap_or(1)
            .max(1);
        let max_rejected_len = indices
            .iter()
            .map(|&i| self.samples[i].rejected_length())
            .max()
            .unwrap_or(1)
            .max(1);
        let max_prompt_len = indices
            .iter()
            .map(|&i| self.samples[i].prompt_ids.len())
            .max()
            .unwrap_or(1)
            .max(1);

        let mut prompt_ids = Array2::from_elem((batch, max_prompt_len), self.pad_id);
        let mut chosen_ids = Array2::from_elem((batch, max_chosen_len), self.pad_id);
        let mut rejected_ids = Array2::from_elem((batch, max_rejected_len), self.pad_id);
        let mut chosen_mask = Array2::zeros((batch, max_chosen_len));
        let mut rejected_mask = Array2::zeros((batch, max_rejected_len));
        let mut chosen_lengths = Vec::with_capacity(batch);
        let mut rejected_lengths = Vec::with_capacity(batch);
        let mut chosen_scores = Vec::with_capacity(batch);
        let mut rejected_scores = Vec::with_capacity(batch);

        for (row, &idx) in indices.iter().enumerate() {
            let pref = &self.samples[idx];

            // Prompt
            for (j, &tid) in pref.prompt_ids.iter().enumerate() {
                if j < max_prompt_len {
                    prompt_ids[[row, j]] = tid;
                }
            }

            // Chosen
            let clen = pref.chosen_length();
            for (j, &tid) in pref.chosen_ids.iter().enumerate() {
                if j < max_chosen_len {
                    chosen_ids[[row, j]] = tid;
                    chosen_mask[[row, j]] = 1u8;
                }
            }
            chosen_lengths.push(clen);

            // Rejected
            let rlen = pref.rejected_length();
            for (j, &tid) in pref.rejected_ids.iter().enumerate() {
                if j < max_rejected_len {
                    rejected_ids[[row, j]] = tid;
                    rejected_mask[[row, j]] = 1u8;
                }
            }
            rejected_lengths.push(rlen);

            chosen_scores.push(pref.chosen_score.unwrap_or(f64::NAN));
            rejected_scores.push(pref.rejected_score.unwrap_or(f64::NAN));
        }

        Ok(PreferenceMiniBatch {
            prompt_ids,
            chosen_ids,
            rejected_ids,
            chosen_mask,
            rejected_mask,
            chosen_lengths,
            rejected_lengths,
            chosen_scores: Array1::from(chosen_scores),
            rejected_scores: Array1::from(rejected_scores),
        })
    }
}

impl Default for PreferenceDataset {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Dataset statistics
// ============================================================================

/// Summary statistics for a preference dataset.
#[derive(Debug, Clone, Default)]
pub struct PreferenceDatasetStats {
    /// Total number of preference pairs.
    pub num_samples: usize,
    /// Mean prompt token length.
    pub avg_prompt_len: f64,
    /// Mean chosen response length.
    pub avg_chosen_len: f64,
    /// Mean rejected response length.
    pub avg_rejected_len: f64,
    /// Fraction of samples where chosen_score > rejected_score.
    pub consistency_rate: f64,
}

// ============================================================================
// Reward signal from preferences
// ============================================================================

/// Convert a batch of preference pairs to a reward signal.
///
/// Each pair `(chosen_score, rejected_score)` is mapped to a tuple
/// `(r_chosen, r_rejected)` where scores are rescaled to a desired output range.
///
/// Specifically:
/// ```text
/// r = (score - min_score) / (max_score - min_score) * (high - low) + low
/// ```
///
/// When scores are not present (NaN), the default values `high` (for chosen)
/// and `low` (for rejected) are used.
///
/// # Arguments
/// - `chosen_scores`   – raw annotator scores for the chosen responses
/// - `rejected_scores` – raw annotator scores for the rejected responses
/// - `low`             – lower bound of the output reward range
/// - `high`            – upper bound of the output reward range
///
/// # Returns
/// `(r_chosen, r_rejected)` arrays of shape `[batch]`.
pub fn reward_from_preferences<F>(
    chosen_scores: &Array1<f64>,
    rejected_scores: &Array1<f64>,
    low: f64,
    high: f64,
) -> Result<(Array1<F>, Array1<F>)>
where
    F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
{
    let n = chosen_scores.len();
    if n == 0 {
        return Err(NeuralError::InvalidArgument(
            "reward_from_preferences: empty input".to_string(),
        ));
    }
    if rejected_scores.len() != n {
        return Err(NeuralError::DimensionMismatch(format!(
            "reward_from_preferences: length mismatch {} vs {}",
            n,
            rejected_scores.len()
        )));
    }
    if high <= low {
        return Err(NeuralError::InvalidArgument(format!(
            "reward_from_preferences: high ({high}) must be > low ({low})"
        )));
    }

    // Compute global score range (ignoring NaN)
    let all_valid: Vec<f64> = chosen_scores
        .iter()
        .chain(rejected_scores.iter())
        .copied()
        .filter(|v| !v.is_nan())
        .collect();

    let (score_min, score_max) = if all_valid.is_empty() {
        // No scores available — use 0/1 defaults
        (0.0_f64, 1.0_f64)
    } else {
        let mn = all_valid.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = all_valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (mn, mx)
    };

    let range = if (score_max - score_min).abs() < 1e-12 {
        1.0_f64
    } else {
        score_max - score_min
    };

    let low_f = F::from_f64(low).ok_or_else(|| {
        NeuralError::ComputationError("reward_from_preferences: cannot convert low".to_string())
    })?;
    let high_f = F::from_f64(high).ok_or_else(|| {
        NeuralError::ComputationError("reward_from_preferences: cannot convert high".to_string())
    })?;
    let output_range_f = high_f - low_f;

    let map_score = |score: f64, default: f64| -> Result<F> {
        let s = if score.is_nan() { default } else { score };
        let normalised = (s - score_min) / range;
        let scaled = normalised * (high - low) + low;
        let clamped = scaled.max(low).min(high);
        F::from_f64(clamped).ok_or_else(|| {
            NeuralError::ComputationError(
                "reward_from_preferences: cannot convert reward".to_string(),
            )
        })
    };

    let _ = output_range_f; // suppress unused warning

    let mut r_chosen: Vec<F> = Vec::with_capacity(n);
    let mut r_rejected: Vec<F> = Vec::with_capacity(n);
    for i in 0..n {
        r_chosen.push(map_score(chosen_scores[i], high)?);
        r_rejected.push(map_score(rejected_scores[i], low)?);
    }

    Ok((Array1::from(r_chosen), Array1::from(r_rejected)))
}

/// Build (chosen_log_probs, rejected_log_probs) from a batch by averaging
/// per-token log-probabilities over each sequence length.
///
/// When working with language model log-probs (shape `[batch, seq_len]`), it
/// is standard to sum / average over the response tokens only (not the prompt).
///
/// # Arguments
/// - `chosen_token_lps`   – per-token log-probs for chosen responses `[batch, seq_len]`
/// - `rejected_token_lps` – per-token log-probs for rejected responses `[batch, seq_len]`
/// - `chosen_lengths`     – actual sequence lengths before padding
/// - `rejected_lengths`   – actual sequence lengths before padding
///
/// # Returns
/// `(lp_chosen, lp_rejected)` mean log-prob vectors of shape `[batch]`.
pub fn aggregate_sequence_log_probs<F>(
    chosen_token_lps: &Array2<F>,
    rejected_token_lps: &Array2<F>,
    chosen_lengths: &[usize],
    rejected_lengths: &[usize],
) -> Result<(Array1<F>, Array1<F>)>
where
    F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
{
    let batch = chosen_token_lps.nrows();
    if batch == 0 {
        return Err(NeuralError::InvalidArgument(
            "aggregate_sequence_log_probs: empty batch".to_string(),
        ));
    }
    if rejected_token_lps.nrows() != batch {
        return Err(NeuralError::DimensionMismatch(format!(
            "aggregate_sequence_log_probs: batch size mismatch {} vs {}",
            batch,
            rejected_token_lps.nrows()
        )));
    }
    if chosen_lengths.len() != batch || rejected_lengths.len() != batch {
        return Err(NeuralError::DimensionMismatch(
            "aggregate_sequence_log_probs: lengths mismatch".to_string(),
        ));
    }

    let mut lp_chosen: Vec<F> = Vec::with_capacity(batch);
    let mut lp_rejected: Vec<F> = Vec::with_capacity(batch);

    for i in 0..batch {
        let clen = chosen_lengths[i].min(chosen_token_lps.ncols());
        let rlen = rejected_lengths[i].min(rejected_token_lps.ncols());

        if clen == 0 {
            return Err(NeuralError::InvalidArgument(format!(
                "aggregate_sequence_log_probs: zero chosen length at index {i}"
            )));
        }
        if rlen == 0 {
            return Err(NeuralError::InvalidArgument(format!(
                "aggregate_sequence_log_probs: zero rejected length at index {i}"
            )));
        }

        let clen_f = F::from_usize(clen).ok_or_else(|| {
            NeuralError::ComputationError("cannot convert clen".to_string())
        })?;
        let rlen_f = F::from_usize(rlen).ok_or_else(|| {
            NeuralError::ComputationError("cannot convert rlen".to_string())
        })?;

        let mut c_sum = F::zero();
        for j in 0..clen {
            c_sum += chosen_token_lps[[i, j]];
        }
        let mut r_sum = F::zero();
        for j in 0..rlen {
            r_sum += rejected_token_lps[[i, j]];
        }

        lp_chosen.push(c_sum / clen_f);
        lp_rejected.push(r_sum / rlen_f);
    }

    Ok((Array1::from(lp_chosen), Array1::from(lp_rejected)))
}

// ============================================================================
// Internal: deterministic shuffle
// ============================================================================

/// Produce a deterministic permutation of `0..n` using a simple LCG / Knuth
/// shuffle so that no external RNG crate is needed.
fn deterministic_shuffle(n: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    if n <= 1 {
        return indices;
    }
    // LCG: xₙ₊₁ = (a * xₙ + c) mod m
    let mut state: u64 = seed ^ 0xDEAD_BEEF_CAFE_BABE;
    let a: u64 = 6364136223846793005;
    let c: u64 = 1442695040888963407;

    for i in (1..n).rev() {
        state = state.wrapping_mul(a).wrapping_add(c);
        let j = (state >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
    indices
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_pref(c_len: usize, r_len: usize) -> Preference {
        Preference {
            prompt_ids: vec![1, 2, 3],
            chosen_ids: vec![10u32; c_len],
            rejected_ids: vec![20u32; r_len],
            chosen_score: Some(1.0),
            rejected_score: Some(0.0),
            annotator_id: None,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_preference_new() {
        let p = Preference::new(vec![1, 2], vec![3, 4, 5], vec![6]);
        assert_eq!(p.prompt_ids.len(), 2);
        assert_eq!(p.chosen_ids.len(), 3);
        assert_eq!(p.rejected_ids.len(), 1);
        assert!(p.is_consistent());
    }

    #[test]
    fn test_dataset_push_and_len() {
        let mut ds = PreferenceDataset::new();
        ds.push(make_pref(3, 2));
        ds.push(make_pref(4, 5));
        assert_eq!(ds.len(), 2);
        assert!(!ds.is_empty());
    }

    #[test]
    fn test_sample_batch_sequential() {
        let mut ds = PreferenceDataset::new();
        for _ in 0..5 {
            ds.push(make_pref(3, 2));
        }
        let batch = ds.sample_batch(3, None).expect("batch");
        assert_eq!(batch.batch_size(), 3);
        assert_eq!(batch.prompt_ids.nrows(), 3);
        assert_eq!(batch.chosen_ids.nrows(), 3);
        assert_eq!(batch.rejected_ids.nrows(), 3);
    }

    #[test]
    fn test_sample_batch_with_seed() {
        let mut ds = PreferenceDataset::new();
        for i in 0..10u32 {
            let p = Preference::new(vec![i], vec![i + 10], vec![i + 20]);
            ds.push(p);
        }
        let b1 = ds.sample_batch(4, Some(42)).expect("b1");
        let b2 = ds.sample_batch(4, Some(42)).expect("b2");
        // Same seed → same indices → same first chosen_ids row
        assert_eq!(b1.chosen_ids.row(0), b2.chosen_ids.row(0));
    }

    #[test]
    fn test_sequential_batch() {
        let mut ds = PreferenceDataset::new();
        for _ in 0..6 {
            ds.push(make_pref(3, 2));
        }
        let batch = ds.sequential_batch(2, 3).expect("batch");
        assert_eq!(batch.batch_size(), 3);
    }

    #[test]
    fn test_padding_correctness() {
        let mut ds = PreferenceDataset::with_pad_id(0);
        // Variable-length chosen responses
        ds.push(Preference::new(vec![1], vec![10, 11], vec![20]));
        ds.push(Preference::new(vec![1], vec![10, 11, 12, 13], vec![20]));
        let batch = ds.sample_batch(2, None).expect("batch");
        // max_chosen_len = 4
        assert_eq!(batch.chosen_ids.ncols(), 4);
        // First row should be padded with 0 in columns 2-3
        assert_eq!(batch.chosen_ids[[0, 2]], 0u32);
        assert_eq!(batch.chosen_ids[[0, 3]], 0u32);
        // Mask should be 1 only where real tokens exist
        assert_eq!(batch.chosen_mask[[0, 0]], 1u8);
        assert_eq!(batch.chosen_mask[[0, 2]], 0u8);
    }

    #[test]
    fn test_train_val_split() {
        let mut ds = PreferenceDataset::new();
        for _ in 0..20 {
            ds.push(make_pref(3, 2));
        }
        let (train, val) = ds.train_val_split(0.2, Some(0)).expect("split");
        assert_eq!(train.len() + val.len(), 20);
        assert!(val.len() >= 1);
        assert!(train.len() >= 1);
    }

    #[test]
    fn test_train_val_split_invalid_fraction() {
        let mut ds = PreferenceDataset::new();
        ds.push(make_pref(3, 2));
        ds.push(make_pref(3, 2));
        let result = ds.train_val_split(0.0, None);
        assert!(result.is_err());
        let result2 = ds.train_val_split(1.0, None);
        assert!(result2.is_err());
    }

    #[test]
    fn test_reward_from_preferences() {
        let chosen = Array1::from(vec![1.0_f64, 0.8, 0.9]);
        let rejected = Array1::from(vec![0.2_f64, 0.3, 0.1]);
        let (r_c, r_r) = reward_from_preferences::<f64>(&chosen, &rejected, -1.0, 1.0)
            .expect("rewards");
        assert_eq!(r_c.len(), 3);
        assert_eq!(r_r.len(), 3);
        for (&c, &r) in r_c.iter().zip(r_r.iter()) {
            assert!(c >= -1.0 && c <= 1.0, "c={c}");
            assert!(r >= -1.0 && r <= 1.0, "r={r}");
        }
    }

    #[test]
    fn test_reward_from_preferences_with_nans() {
        let chosen = Array1::from(vec![f64::NAN, 0.8]);
        let rejected = Array1::from(vec![0.2_f64, f64::NAN]);
        let (r_c, r_r) = reward_from_preferences::<f64>(&chosen, &rejected, 0.0, 1.0)
            .expect("rewards");
        // NaN chosen → default high=1.0
        assert!((r_c[0] - 1.0).abs() < 1e-9, "r_c[0]={}", r_c[0]);
        // NaN rejected → default low=0.0
        assert!((r_r[1] - 0.0).abs() < 1e-9, "r_r[1]={}", r_r[1]);
    }

    #[test]
    fn test_aggregate_sequence_log_probs() {
        let chosen_lps = Array2::from_shape_vec(
            (2, 4),
            vec![-1.0_f64, -2.0, 0.0, 0.0, -0.5, -1.5, 0.0, 0.0],
        )
        .expect("arr");
        let rejected_lps = Array2::from_shape_vec(
            (2, 4),
            vec![-3.0_f64, -4.0, 0.0, 0.0, -2.0, -3.0, -4.0, 0.0],
        )
        .expect("arr");
        let c_lens = vec![2, 2];
        let r_lens = vec![2, 3];
        let (lp_c, lp_r) =
            aggregate_sequence_log_probs::<f64>(&chosen_lps, &rejected_lps, &c_lens, &r_lens)
                .expect("agg");
        assert!((lp_c[0] - (-1.5)).abs() < 1e-9, "lp_c[0]={}", lp_c[0]);
        assert!((lp_c[1] - (-1.0)).abs() < 1e-9, "lp_c[1]={}", lp_c[1]);
        assert!((lp_r[0] - (-3.5)).abs() < 1e-9, "lp_r[0]={}", lp_r[0]);
        let expected_r1 = (-2.0 - 3.0 - 4.0) / 3.0;
        assert!((lp_r[1] - expected_r1).abs() < 1e-9, "lp_r[1]={}", lp_r[1]);
    }

    #[test]
    fn test_dataset_stats() {
        let mut ds = PreferenceDataset::new();
        ds.push(make_pref(3, 2));
        ds.push(make_pref(4, 5));
        let stats = ds.stats();
        assert_eq!(stats.num_samples, 2);
        assert!((stats.avg_chosen_len - 3.5).abs() < 1e-9);
        assert!((stats.avg_rejected_len - 3.5).abs() < 1e-9);
        assert!((stats.consistency_rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_dataset_error() {
        let ds = PreferenceDataset::new();
        let result = ds.sample_batch(4, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_deterministic_shuffle_reproducible() {
        let a = deterministic_shuffle(10, 12345);
        let b = deterministic_shuffle(10, 12345);
        assert_eq!(a, b);
    }

    #[test]
    fn test_deterministic_shuffle_is_permutation() {
        let n = 20;
        let mut perm = deterministic_shuffle(n, 999);
        perm.sort_unstable();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(perm, expected);
    }
}
