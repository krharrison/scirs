//! Overlap-save convolution for efficient streaming FIR filtering.
//!
//! The overlap-save method (also called overlap-scrap) divides a long input
//! signal into overlapping blocks, applies the FFT-based circular convolution
//! to each block, and discards the invalid (aliased) portion of each output
//! block.  This yields the same result as direct linear convolution but with
//! O(N log N) cost per block instead of O(N M) for an M-tap FIR filter.
//!
//! ## Usage
//!
//! ```
//! use scirs2_signal::streaming::overlap_save::OverlapSave;
//!
//! let filter = vec![0.25, 0.5, 0.25];
//! let block_len = 64;
//! let mut ols = OverlapSave::new(&filter, block_len).expect("valid params");
//!
//! let input = vec![1.0; 64];
//! let output = ols.process_block(&input).expect("process");
//! assert_eq!(output.len(), 64);
//! ```
//!
//! ## Algorithm
//!
//! 1. Choose FFT size `N >= filter_len + block_len - 1`.
//! 2. Zero-pad the filter to length N and pre-compute `H = FFT(h)`.
//! 3. For each input block of length `block_len`:
//!    a. Form a length-N input segment: `[overlap | new_block]`.
//!    b. Compute `X = FFT(segment)`.
//!    c. Point-wise multiply `Y = X * H`.
//!    d. Compute `y = IFFT(Y)`.
//!    e. Discard the first `filter_len - 1` samples (the "overlap" or
//!       aliased region).
//!    f. Output the remaining `block_len` valid samples.

use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;

/// Overlap-save streaming convolver.
///
/// Internally uses `scirs2_fft` (OxiFFT backend) for all FFT operations.
#[derive(Debug, Clone)]
pub struct OverlapSave {
    /// FFT size (N >= filter_len + block_len - 1).
    fft_size: usize,
    /// Length of the FIR filter.
    filter_len: usize,
    /// Number of new input samples per block.
    block_len: usize,
    /// Pre-computed filter spectrum, length = fft_size/2 + 1 (from rfft).
    filter_spectrum: Vec<Complex64>,
    /// Overlap buffer: the last `filter_len - 1` samples from the previous
    /// input segment.
    overlap: Vec<f64>,
    /// Whether the first block has been processed (the very first output
    /// requires special handling since there is no prior overlap).
    first_block: bool,
    /// Total blocks processed.
    blocks_processed: u64,
}

impl OverlapSave {
    /// Create a new overlap-save convolver.
    ///
    /// # Arguments
    ///
    /// * `filter` - FIR filter coefficients.  Must be non-empty.
    /// * `block_len` - Number of new input samples per block.  Must be > 0.
    ///
    /// The FFT size is chosen as the smallest power of two that is
    /// `>= filter.len() + block_len - 1`.
    ///
    /// # Errors
    ///
    /// Returns an error if `filter` is empty or `block_len` is zero.
    pub fn new(filter: &[f64], block_len: usize) -> SignalResult<Self> {
        if filter.is_empty() {
            return Err(SignalError::ValueError(
                "Filter coefficients must not be empty".to_string(),
            ));
        }
        if block_len == 0 {
            return Err(SignalError::ValueError("block_len must be > 0".to_string()));
        }

        let filter_len = filter.len();
        let min_fft_size = filter_len + block_len - 1;
        let fft_size = next_power_of_two(min_fft_size);

        // Zero-pad filter and compute its spectrum
        let mut h_padded = vec![0.0; fft_size];
        h_padded[..filter_len].copy_from_slice(filter);

        let filter_spectrum = scirs2_fft::rfft(&h_padded, None)
            .map_err(|e| SignalError::ComputationError(format!("FFT of filter failed: {e}")))?;

        let overlap_len = filter_len.saturating_sub(1);

        Ok(Self {
            fft_size,
            filter_len,
            block_len,
            filter_spectrum,
            overlap: vec![0.0; overlap_len],
            first_block: true,
            blocks_processed: 0,
        })
    }

    /// Create an overlap-save convolver with a specific FFT size.
    ///
    /// # Arguments
    ///
    /// * `filter` - FIR filter coefficients.
    /// * `block_len` - Number of new input samples per block.
    /// * `fft_size` - FFT size.  Must be >= `filter.len() + block_len - 1`.
    ///
    /// # Errors
    ///
    /// Returns an error if constraints are violated.
    pub fn with_fft_size(filter: &[f64], block_len: usize, fft_size: usize) -> SignalResult<Self> {
        if filter.is_empty() {
            return Err(SignalError::ValueError(
                "Filter coefficients must not be empty".to_string(),
            ));
        }
        if block_len == 0 {
            return Err(SignalError::ValueError("block_len must be > 0".to_string()));
        }
        let filter_len = filter.len();
        let min_fft_size = filter_len + block_len - 1;
        if fft_size < min_fft_size {
            return Err(SignalError::ValueError(format!(
                "fft_size ({fft_size}) must be >= filter_len + block_len - 1 ({min_fft_size})"
            )));
        }

        let mut h_padded = vec![0.0; fft_size];
        h_padded[..filter_len].copy_from_slice(filter);

        let filter_spectrum = scirs2_fft::rfft(&h_padded, None)
            .map_err(|e| SignalError::ComputationError(format!("FFT of filter failed: {e}")))?;

        let overlap_len = filter_len.saturating_sub(1);

        Ok(Self {
            fft_size,
            filter_len,
            block_len,
            filter_spectrum,
            overlap: vec![0.0; overlap_len],
            first_block: true,
            blocks_processed: 0,
        })
    }

    /// Process one block of input samples and return the valid convolution
    /// output.
    ///
    /// The input slice **must** have exactly `block_len` elements.
    ///
    /// Returns a `Vec<f64>` of length `block_len` containing the valid
    /// output samples.
    ///
    /// # Errors
    ///
    /// Returns an error if the input length does not match `block_len` or if
    /// an FFT computation fails.
    pub fn process_block(&mut self, input: &[f64]) -> SignalResult<Vec<f64>> {
        if input.len() != self.block_len {
            return Err(SignalError::DimensionMismatch(format!(
                "Expected block of {} samples, got {}",
                self.block_len,
                input.len()
            )));
        }

        let overlap_len = self.filter_len.saturating_sub(1);

        // Build the full input segment of length fft_size:
        //   [overlap (filter_len-1 samples) | new block (block_len samples) | zero pad]
        let mut segment = vec![0.0; self.fft_size];
        segment[..overlap_len].copy_from_slice(&self.overlap);
        segment[overlap_len..overlap_len + self.block_len].copy_from_slice(input);

        // Forward FFT
        let x_spectrum = scirs2_fft::rfft(&segment, None).map_err(|e| {
            SignalError::ComputationError(format!("FFT of input block failed: {e}"))
        })?;

        // Point-wise multiply
        let y_spectrum: Vec<Complex64> = x_spectrum
            .iter()
            .zip(self.filter_spectrum.iter())
            .map(|(&x, &h)| x * h)
            .collect();

        // Inverse FFT
        let y_full = scirs2_fft::irfft(&y_spectrum, Some(self.fft_size))
            .map_err(|e| SignalError::ComputationError(format!("IFFT failed: {e}")))?;

        // The valid output samples are at indices [overlap_len .. overlap_len + block_len]
        let valid_start = overlap_len;
        let valid_end = valid_start + self.block_len;
        let output = y_full[valid_start..valid_end].to_vec();

        // Update the overlap buffer with the last `overlap_len` samples from
        // the *input* segment (not the output).
        // In overlap-save, the overlap comes from the tail of the current
        // input block.
        if overlap_len > 0 {
            if self.block_len >= overlap_len {
                self.overlap
                    .copy_from_slice(&input[self.block_len - overlap_len..]);
            } else {
                // Block is shorter than overlap: shift existing overlap and
                // append new samples.
                let shift = overlap_len - self.block_len;
                for i in 0..shift {
                    self.overlap[i] = self.overlap[i + self.block_len];
                }
                self.overlap[shift..].copy_from_slice(input);
            }
        }

        self.first_block = false;
        self.blocks_processed += 1;

        Ok(output)
    }

    /// Reset the convolver state.
    pub fn reset(&mut self) {
        self.overlap.iter_mut().for_each(|v| *v = 0.0);
        self.first_block = true;
        self.blocks_processed = 0;
    }

    /// FFT size used internally.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Filter length.
    pub fn filter_len(&self) -> usize {
        self.filter_len
    }

    /// Block length (number of new input samples per call).
    pub fn block_len(&self) -> usize {
        self.block_len
    }

    /// Total blocks processed.
    pub fn blocks_processed(&self) -> u64 {
        self.blocks_processed
    }
}

// ============================================================================
// StreamProcessor trait impl
// ============================================================================

impl super::StreamProcessor for OverlapSave {
    fn process_block(&mut self, input: &[f64]) -> SignalResult<Vec<f64>> {
        self.process_block(input)
    }

    fn reset(&mut self) {
        self.reset();
    }
}

// ============================================================================
// Helper
// ============================================================================

/// Return the smallest power of two >= n.
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

// ============================================================================
// Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlap_save_creation() {
        let ols = OverlapSave::new(&[1.0, 0.5], 64);
        assert!(ols.is_ok());
        let ols = ols.expect("should succeed");
        assert_eq!(ols.block_len(), 64);
        assert_eq!(ols.filter_len(), 2);
        assert!(ols.fft_size() >= 64 + 2 - 1);
    }

    #[test]
    fn test_overlap_save_empty_filter_error() {
        assert!(OverlapSave::new(&[], 64).is_err());
    }

    #[test]
    fn test_overlap_save_zero_block_error() {
        assert!(OverlapSave::new(&[1.0], 0).is_err());
    }

    #[test]
    fn test_overlap_save_fft_size_too_small() {
        assert!(OverlapSave::with_fft_size(&[1.0, 0.5, 0.25], 8, 4).is_err());
    }

    #[test]
    fn test_overlap_save_identity_filter() {
        // Filter = [1.0] => output should equal input (with possible tiny numerical error)
        let mut ols = OverlapSave::new(&[1.0], 8).expect("create OverlapSave");
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let output = ols.process_block(&input).expect("process");
        for (i, (&y, &x)) in output.iter().zip(input.iter()).enumerate() {
            assert!(
                (y - x).abs() < 1e-10,
                "Identity mismatch at {i}: got {y}, expected {x}"
            );
        }
    }

    #[test]
    fn test_overlap_save_vs_direct_convolution() {
        // Compare overlap-save output with direct (non-streaming) convolution
        // for a multi-block signal.
        let filter = vec![0.25, 0.5, 0.25];
        let block_len = 16;
        let num_blocks = 4;
        let total_len = block_len * num_blocks;

        // Generate test signal
        let signal: Vec<f64> = (0..total_len).map(|i| (i as f64 * 0.2).sin()).collect();

        // Stream through overlap-save
        let mut ols = OverlapSave::new(&filter, block_len).expect("create OverlapSave");
        let mut streamed = Vec::with_capacity(total_len);
        for block_idx in 0..num_blocks {
            let start = block_idx * block_len;
            let end = start + block_len;
            let output = ols.process_block(&signal[start..end]).expect("process");
            streamed.extend_from_slice(&output);
        }

        // Direct convolution (manual, "full" mode then trim to "same"-like)
        let direct = direct_convolve(&signal, &filter);

        // Compare valid region (skip the first filter_len-1 samples due to startup transient)
        let skip = filter.len() - 1;
        for i in skip..total_len {
            assert!(
                (streamed[i] - direct[i]).abs() < 1e-8,
                "Mismatch at index {i}: streamed={}, direct={}",
                streamed[i],
                direct[i]
            );
        }
    }

    #[test]
    fn test_overlap_save_wrong_block_size() {
        let mut ols = OverlapSave::new(&[1.0], 8).expect("create OverlapSave");
        assert!(ols.process_block(&[1.0; 4]).is_err());
    }

    #[test]
    fn test_overlap_save_reset() {
        let mut ols = OverlapSave::new(&[0.5, 0.5], 8).expect("create OverlapSave");
        let _ = ols.process_block(&[1.0; 8]).expect("process");
        assert_eq!(ols.blocks_processed(), 1);

        ols.reset();
        assert_eq!(ols.blocks_processed(), 0);

        // After reset, output should be the same as the first block ever
        let mut fresh = OverlapSave::new(&[0.5, 0.5], 8).expect("create fresh");
        let input = vec![1.0; 8];
        let out_reset = ols.process_block(&input).expect("after reset");
        let out_fresh = fresh.process_block(&input).expect("fresh");

        for (i, (&r, &f)) in out_reset.iter().zip(out_fresh.iter()).enumerate() {
            assert!((r - f).abs() < 1e-12, "Reset mismatch at {i}: {r} vs {f}");
        }
    }

    #[test]
    fn test_overlap_save_long_filter() {
        // Test with a longer filter (16 taps)
        let filter: Vec<f64> = (0..16).map(|i| 1.0 / (i as f64 + 1.0)).collect();
        let block_len = 32;
        let mut ols = OverlapSave::new(&filter, block_len).expect("create OverlapSave");

        let input: Vec<f64> = (0..block_len).map(|i| (i as f64 * 0.1).sin()).collect();
        let output = ols.process_block(&input).expect("process");
        assert_eq!(output.len(), block_len);

        // Output should be finite
        for &v in &output {
            assert!(v.is_finite(), "Output should be finite, got {v}");
        }
    }

    #[test]
    fn test_overlap_save_custom_fft_size() {
        let filter = vec![0.5, 0.5];
        let block_len = 8;
        let fft_size = 32; // Larger than minimum
        let ols = OverlapSave::with_fft_size(&filter, block_len, fft_size);
        assert!(ols.is_ok());
        let ols = ols.expect("should succeed");
        assert_eq!(ols.fft_size(), 32);
    }

    // Helper: direct linear convolution (output length = signal.len(), "same"-like)
    fn direct_convolve(signal: &[f64], filter: &[f64]) -> Vec<f64> {
        let n = signal.len();
        let m = filter.len();
        let mut out = vec![0.0; n];
        for i in 0..n {
            for j in 0..m {
                if i >= j {
                    out[i] += filter[j] * signal[i - j];
                }
            }
        }
        out
    }
}
