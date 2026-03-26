//! Block-based FIR and IIR filters for streaming signal processing.
//!
//! Unlike the sample-by-sample [`StreamingFIR`](super::StreamingFIR) and
//! [`StreamingIIR`](super::StreamingIIR), the block filters in this module
//! process data in **fixed-size blocks**, making them suitable for real-time
//! audio callbacks, hardware DMA buffers, and other scenarios where samples
//! arrive in fixed-length chunks.
//!
//! ## Provided filters
//!
//! | Struct | Description |
//! |--------|-------------|
//! | [`BlockFIR`] | FIR filter with fixed block processing and delay-line state |
//! | [`BlockIIR`] | IIR filter (general b/a form) with fixed block processing |
//!
//! Both filters support **state save/restore** for pause/resume workflows.

use crate::error::{SignalError, SignalResult};
use serde::{Deserialize, Serialize};

// ============================================================================
// BlockFIR
// ============================================================================

/// Saved state of a [`BlockFIR`] for pause/resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockFIRState {
    /// Delay line contents.
    delay_line: Vec<f64>,
    /// Write position in the delay line.
    pos: usize,
    /// Total samples processed.
    samples_processed: u64,
}

/// Block-based FIR filter.
///
/// Processes input in fixed-size blocks through an FIR filter while maintaining
/// a delay line between successive calls.  The block size is configurable at
/// construction time.
///
/// # Example
///
/// ```
/// use scirs2_signal::streaming::block_filter::BlockFIR;
///
/// let coeffs = vec![0.25, 0.5, 0.25]; // simple 3-tap smoothing filter
/// let block_size = 64;
/// let mut fir = BlockFIR::new(&coeffs, block_size).expect("valid filter");
///
/// let input = vec![1.0; 64];
/// let output = fir.process_block(&input).expect("process");
/// assert_eq!(output.len(), 64);
/// ```
#[derive(Debug, Clone)]
pub struct BlockFIR {
    /// FIR coefficients (taps).
    coeffs: Vec<f64>,
    /// Delay line (circular buffer) of length `num_taps`.
    delay_line: Vec<f64>,
    /// Current write position in the delay line.
    pos: usize,
    /// Configured block size.
    block_size: usize,
    /// Total samples processed.
    samples_processed: u64,
}

impl BlockFIR {
    /// Create a new block-based FIR filter.
    ///
    /// # Arguments
    ///
    /// * `coefficients` - FIR tap weights.  Must be non-empty.
    /// * `block_size`   - Number of samples per processing block.  Must be > 0.
    ///
    /// # Errors
    ///
    /// Returns `SignalError::ValueError` if `coefficients` is empty or
    /// `block_size` is zero.
    pub fn new(coefficients: &[f64], block_size: usize) -> SignalResult<Self> {
        if coefficients.is_empty() {
            return Err(SignalError::ValueError(
                "FIR coefficients must not be empty".to_string(),
            ));
        }
        if block_size == 0 {
            return Err(SignalError::ValueError(
                "block_size must be > 0".to_string(),
            ));
        }
        let n = coefficients.len();
        Ok(Self {
            coeffs: coefficients.to_vec(),
            delay_line: vec![0.0; n],
            pos: 0,
            block_size,
            samples_processed: 0,
        })
    }

    /// Process one block of input samples.
    ///
    /// The input slice **must** have exactly `block_size` elements.
    ///
    /// # Errors
    ///
    /// Returns `SignalError::DimensionMismatch` if the input length does not
    /// match the configured block size.
    pub fn process_block(&mut self, input: &[f64]) -> SignalResult<Vec<f64>> {
        if input.len() != self.block_size {
            return Err(SignalError::DimensionMismatch(format!(
                "Expected block of {} samples, got {}",
                self.block_size,
                input.len()
            )));
        }
        let n = self.coeffs.len();
        let mut output = Vec::with_capacity(self.block_size);

        for &sample in input {
            self.delay_line[self.pos] = sample;
            let mut y = 0.0;
            for (k, &coeff) in self.coeffs.iter().enumerate() {
                let idx = (self.pos + n - k) % n;
                y += coeff * self.delay_line[idx];
            }
            self.pos = (self.pos + 1) % n;
            self.samples_processed += 1;
            output.push(y);
        }

        Ok(output)
    }

    /// Reset the filter state (delay line) to zero.
    pub fn reset(&mut self) {
        self.delay_line.iter_mut().for_each(|v| *v = 0.0);
        self.pos = 0;
        self.samples_processed = 0;
    }

    /// Save the current filter state for later restoration.
    pub fn save_state(&self) -> BlockFIRState {
        BlockFIRState {
            delay_line: self.delay_line.clone(),
            pos: self.pos,
            samples_processed: self.samples_processed,
        }
    }

    /// Restore a previously saved filter state.
    ///
    /// # Errors
    ///
    /// Returns an error if the saved state has a different delay-line length
    /// (i.e. it was saved from a filter with different coefficients).
    pub fn restore_state(&mut self, state: &BlockFIRState) -> SignalResult<()> {
        if state.delay_line.len() != self.delay_line.len() {
            return Err(SignalError::ValueError(format!(
                "State delay-line length {} does not match filter length {}",
                state.delay_line.len(),
                self.delay_line.len()
            )));
        }
        self.delay_line.copy_from_slice(&state.delay_line);
        self.pos = state.pos;
        self.samples_processed = state.samples_processed;
        Ok(())
    }

    /// Configured block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of FIR taps.
    pub fn num_taps(&self) -> usize {
        self.coeffs.len()
    }

    /// Total samples processed.
    pub fn samples_processed(&self) -> u64 {
        self.samples_processed
    }
}

// ============================================================================
// BlockIIR
// ============================================================================

/// Saved state of a [`BlockIIR`] for pause/resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockIIRState {
    /// Input delay line (x history).
    x_history: Vec<f64>,
    /// Output delay line (y history).
    y_history: Vec<f64>,
    /// Total samples processed.
    samples_processed: u64,
}

/// Block-based IIR filter using the Direct Form I difference equation.
///
/// The filter is defined by numerator coefficients `b` and denominator
/// coefficients `a` such that:
///
/// ```text
/// a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... - a[1]*y[n-1] - a[2]*y[n-2] - ...
/// ```
///
/// `a[0]` is normalised internally so that it need not be 1.
///
/// # Example
///
/// ```
/// use scirs2_signal::streaming::block_filter::BlockIIR;
///
/// // First-order lowpass: y[n] = 0.1*x[n] + 0.9*y[n-1]
/// // b = [0.1], a = [1.0, -0.9]
/// let mut iir = BlockIIR::new(&[0.1], &[1.0, -0.9], 32).expect("valid filter");
/// let input = vec![1.0; 32];
/// let output = iir.process_block(&input).expect("process");
/// assert_eq!(output.len(), 32);
/// ```
#[derive(Debug, Clone)]
pub struct BlockIIR {
    /// Numerator (feedforward) coefficients, normalised by a[0].
    b: Vec<f64>,
    /// Denominator (feedback) coefficients, normalised by a[0].
    /// a[0] is always 1.0 after normalisation; we store from index 1 onward.
    a: Vec<f64>,
    /// Input history ring buffer (length = len(b)).
    x_history: Vec<f64>,
    /// Output history ring buffer (length = max(1, len(a) - 1)).
    y_history: Vec<f64>,
    /// Write position for x_history.
    x_pos: usize,
    /// Write position for y_history.
    y_pos: usize,
    /// Configured block size.
    block_size: usize,
    /// Total samples processed.
    samples_processed: u64,
}

impl BlockIIR {
    /// Create a new block-based IIR filter.
    ///
    /// # Arguments
    ///
    /// * `b` - Numerator (feedforward) coefficients.  Must be non-empty.
    /// * `a` - Denominator (feedback) coefficients.  Must be non-empty and `a[0] != 0`.
    /// * `block_size` - Number of samples per processing block.
    ///
    /// # Errors
    ///
    /// Returns `SignalError::ValueError` if `b` or `a` are empty, `a[0]` is
    /// zero, or `block_size` is zero.
    pub fn new(b: &[f64], a: &[f64], block_size: usize) -> SignalResult<Self> {
        if b.is_empty() {
            return Err(SignalError::ValueError(
                "IIR numerator coefficients must not be empty".to_string(),
            ));
        }
        if a.is_empty() {
            return Err(SignalError::ValueError(
                "IIR denominator coefficients must not be empty".to_string(),
            ));
        }
        if a[0].abs() < 1e-30 {
            return Err(SignalError::ValueError(
                "IIR denominator a[0] must be non-zero".to_string(),
            ));
        }
        if block_size == 0 {
            return Err(SignalError::ValueError(
                "block_size must be > 0".to_string(),
            ));
        }

        let inv_a0 = 1.0 / a[0];
        let b_norm: Vec<f64> = b.iter().map(|&v| v * inv_a0).collect();
        // Store a[1..] normalised (a[0] = 1 is implicit).
        let a_norm: Vec<f64> = a.iter().skip(1).map(|&v| v * inv_a0).collect();

        let nb = b_norm.len();
        let na = a_norm.len();

        Ok(Self {
            b: b_norm,
            a: a_norm,
            x_history: vec![0.0; nb],
            y_history: vec![0.0; na.max(1)],
            x_pos: 0,
            y_pos: 0,
            block_size,
            samples_processed: 0,
        })
    }

    /// Process one block of input samples.
    ///
    /// The input slice **must** have exactly `block_size` elements.
    ///
    /// # Errors
    ///
    /// Returns `SignalError::DimensionMismatch` if the input length does not
    /// match the configured block size.
    pub fn process_block(&mut self, input: &[f64]) -> SignalResult<Vec<f64>> {
        if input.len() != self.block_size {
            return Err(SignalError::DimensionMismatch(format!(
                "Expected block of {} samples, got {}",
                self.block_size,
                input.len()
            )));
        }

        let nb = self.b.len();
        let na = self.a.len();
        let ny = self.y_history.len();

        let mut output = Vec::with_capacity(self.block_size);

        for &sample in input {
            // Store input
            self.x_history[self.x_pos] = sample;

            // Feedforward sum: sum b[k] * x[n-k]
            let mut y = 0.0;
            for k in 0..nb {
                let idx = (self.x_pos + nb - k) % nb;
                y += self.b[k] * self.x_history[idx];
            }

            // Feedback sum: - sum a[k] * y[n-k]  (a[0]=1 is implicit, a stored from index 1)
            for k in 0..na {
                let idx = (self.y_pos + ny - k) % ny;
                y -= self.a[k] * self.y_history[idx];
            }

            // Advance positions
            self.x_pos = (self.x_pos + 1) % nb;
            if ny > 0 {
                self.y_pos = (self.y_pos + 1) % ny;
                self.y_history[self.y_pos] = y;
            }

            self.samples_processed += 1;
            output.push(y);
        }

        Ok(output)
    }

    /// Reset the filter state to zero.
    pub fn reset(&mut self) {
        self.x_history.iter_mut().for_each(|v| *v = 0.0);
        self.y_history.iter_mut().for_each(|v| *v = 0.0);
        self.x_pos = 0;
        self.y_pos = 0;
        self.samples_processed = 0;
    }

    /// Save the current filter state.
    pub fn save_state(&self) -> BlockIIRState {
        BlockIIRState {
            x_history: self.x_history.clone(),
            y_history: self.y_history.clone(),
            samples_processed: self.samples_processed,
        }
    }

    /// Restore a previously saved filter state.
    ///
    /// # Errors
    ///
    /// Returns an error if the saved state dimensions do not match.
    pub fn restore_state(&mut self, state: &BlockIIRState) -> SignalResult<()> {
        if state.x_history.len() != self.x_history.len() {
            return Err(SignalError::ValueError(format!(
                "State x_history length {} does not match filter length {}",
                state.x_history.len(),
                self.x_history.len()
            )));
        }
        if state.y_history.len() != self.y_history.len() {
            return Err(SignalError::ValueError(format!(
                "State y_history length {} does not match filter length {}",
                state.y_history.len(),
                self.y_history.len()
            )));
        }
        self.x_history.copy_from_slice(&state.x_history);
        self.y_history.copy_from_slice(&state.y_history);
        // Reset positions to 0 since saved history is a snapshot
        self.x_pos = 0;
        self.y_pos = 0;
        self.samples_processed = state.samples_processed;
        Ok(())
    }

    /// Configured block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Total samples processed.
    pub fn samples_processed(&self) -> u64 {
        self.samples_processed
    }
}

// ============================================================================
// StreamProcessor trait impl
// ============================================================================

impl super::StreamProcessor for BlockFIR {
    fn process_block(&mut self, input: &[f64]) -> SignalResult<Vec<f64>> {
        self.process_block(input)
    }

    fn reset(&mut self) {
        self.reset();
    }
}

impl super::StreamProcessor for BlockIIR {
    fn process_block(&mut self, input: &[f64]) -> SignalResult<Vec<f64>> {
        self.process_block(input)
    }

    fn reset(&mut self) {
        self.reset();
    }
}

// ============================================================================
// Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // ---- BlockFIR ----

    #[test]
    fn test_block_fir_identity() {
        // Identity filter: [1.0]
        let mut fir = BlockFIR::new(&[1.0], 4).expect("create BlockFIR");
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = fir.process_block(&input).expect("process");
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - input[i]).abs() < 1e-12,
                "Identity filter mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_block_fir_matches_sample_by_sample() {
        // Compare block output with sample-by-sample streaming FIR
        let coeffs = [0.3, 0.5, 0.2];
        let block_size = 8;
        let mut block_fir = BlockFIR::new(&coeffs, block_size).expect("create BlockFIR");
        let mut stream_fir =
            crate::streaming::filters::StreamingFIR::new(&coeffs).expect("create StreamingFIR");

        let input: Vec<f64> = (0..block_size).map(|i| (i as f64 * 0.7).sin()).collect();
        let block_out = block_fir.process_block(&input).expect("block process");
        let stream_out = stream_fir.process_chunk(&input);

        for (i, (&b, &s)) in block_out.iter().zip(stream_out.iter()).enumerate() {
            assert!(
                (b - s).abs() < 1e-12,
                "Mismatch at index {i}: block={b}, stream={s}"
            );
        }
    }

    #[test]
    fn test_block_fir_continuity_across_blocks() {
        // State should persist between blocks
        let coeffs = [0.5, 0.5];
        let mut fir = BlockFIR::new(&coeffs, 2).expect("create BlockFIR");

        let out1 = fir.process_block(&[1.0, 0.0]).expect("block 1");
        // y[0] = 0.5*1 + 0.5*0 = 0.5
        // y[1] = 0.5*0 + 0.5*1 = 0.5
        assert!((out1[0] - 0.5).abs() < 1e-12);
        assert!((out1[1] - 0.5).abs() < 1e-12);

        let out2 = fir.process_block(&[2.0, 0.0]).expect("block 2");
        // y[2] = 0.5*2 + 0.5*0 = 1.0  (x[n-1] = last sample of prev block = 0.0)
        assert!((out2[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_block_fir_wrong_block_size() {
        let mut fir = BlockFIR::new(&[1.0], 4).expect("create BlockFIR");
        let result = fir.process_block(&[1.0, 2.0]); // wrong size
        assert!(result.is_err());
    }

    #[test]
    fn test_block_fir_empty_coeffs_error() {
        assert!(BlockFIR::new(&[], 4).is_err());
    }

    #[test]
    fn test_block_fir_zero_block_size_error() {
        assert!(BlockFIR::new(&[1.0], 0).is_err());
    }

    #[test]
    fn test_block_fir_save_restore() {
        let coeffs = [0.5, 0.3, 0.2];
        let mut fir = BlockFIR::new(&coeffs, 4).expect("create BlockFIR");
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let _ = fir.process_block(&input).expect("process");

        // Save state
        let state = fir.save_state();

        // Process more data
        let _ = fir.process_block(&[5.0, 6.0, 7.0, 8.0]).expect("process");

        // Create a new filter and restore state
        let mut fir2 = BlockFIR::new(&coeffs, 4).expect("create BlockFIR");
        fir2.restore_state(&state).expect("restore");

        // Process the same data on the restored filter
        let out_restored = fir2.process_block(&[5.0, 6.0, 7.0, 8.0]).expect("process");

        // Reset original and reprocess from saved state
        fir.reset();
        let _ = fir.process_block(&input).expect("reprocess");
        let out_original = fir.process_block(&[5.0, 6.0, 7.0, 8.0]).expect("process");

        for (i, (&r, &o)) in out_restored.iter().zip(out_original.iter()).enumerate() {
            assert!(
                (r - o).abs() < 1e-12,
                "Restored output mismatch at index {i}: {r} vs {o}"
            );
        }
    }

    #[test]
    fn test_block_fir_reset() {
        let mut fir = BlockFIR::new(&[0.5, 0.5], 2).expect("create BlockFIR");
        let _ = fir.process_block(&[10.0, 20.0]).expect("process");
        fir.reset();
        assert_eq!(fir.samples_processed(), 0);
        let out = fir.process_block(&[1.0, 0.0]).expect("after reset");
        // After reset, delay line is zero:  y[0] = 0.5*1 + 0.5*0 = 0.5
        assert!((out[0] - 0.5).abs() < 1e-12);
    }

    // ---- BlockIIR ----

    #[test]
    fn test_block_iir_passthrough() {
        // b=[1], a=[1] => y[n] = x[n]
        let mut iir = BlockIIR::new(&[1.0], &[1.0], 4).expect("create BlockIIR");
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = iir.process_block(&input).expect("process");
        for (i, (&y, &x)) in output.iter().zip(input.iter()).enumerate() {
            assert!((y - x).abs() < 1e-12, "Passthrough mismatch at index {i}");
        }
    }

    #[test]
    fn test_block_iir_first_order_lowpass() {
        // y[n] = 0.1*x[n] + 0.9*y[n-1]
        // b = [0.1], a = [1.0, -0.9]
        let mut iir = BlockIIR::new(&[0.1], &[1.0, -0.9], 64).expect("create BlockIIR");

        // Step response should converge towards 1.0
        let mut last = 0.0;
        for _ in 0..10 {
            let input = vec![1.0; 64];
            let output = iir.process_block(&input).expect("process");
            last = output[output.len() - 1];
        }
        assert!(
            (last - 1.0).abs() < 0.01,
            "Step response should converge to 1.0, got {last}"
        );
    }

    #[test]
    fn test_block_iir_continuity_across_blocks() {
        // Verify state persists across block boundaries
        let mut iir = BlockIIR::new(&[0.1], &[1.0, -0.9], 1).expect("create BlockIIR");

        let y0 = iir.process_block(&[1.0]).expect("block 0")[0];
        let y1 = iir.process_block(&[1.0]).expect("block 1")[0];

        // y[0] = 0.1*1 = 0.1
        // y[1] = 0.1*1 + 0.9*0.1 = 0.19
        assert!((y0 - 0.1).abs() < 1e-12, "y[0] should be 0.1, got {y0}");
        assert!((y1 - 0.19).abs() < 1e-12, "y[1] should be 0.19, got {y1}");
    }

    #[test]
    fn test_block_iir_wrong_block_size() {
        let mut iir = BlockIIR::new(&[1.0], &[1.0], 4).expect("create BlockIIR");
        assert!(iir.process_block(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_block_iir_empty_coeffs_error() {
        assert!(BlockIIR::new(&[], &[1.0], 4).is_err());
        assert!(BlockIIR::new(&[1.0], &[], 4).is_err());
    }

    #[test]
    fn test_block_iir_a0_zero_error() {
        assert!(BlockIIR::new(&[1.0], &[0.0, 1.0], 4).is_err());
    }

    #[test]
    fn test_block_iir_save_restore() {
        let mut iir = BlockIIR::new(&[0.1], &[1.0, -0.9], 4).expect("create BlockIIR");
        let _ = iir.process_block(&[1.0; 4]).expect("process");

        let state = iir.save_state();

        let out1 = iir.process_block(&[0.5; 4]).expect("process");

        let mut iir2 = BlockIIR::new(&[0.1], &[1.0, -0.9], 4).expect("create BlockIIR");
        iir2.restore_state(&state).expect("restore");
        let out2 = iir2.process_block(&[0.5; 4]).expect("process");

        for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "Restored output mismatch at index {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_block_iir_reset() {
        let mut iir = BlockIIR::new(&[0.1], &[1.0, -0.9], 4).expect("create BlockIIR");
        let _ = iir.process_block(&[1.0; 4]).expect("process");
        iir.reset();
        assert_eq!(iir.samples_processed(), 0);
        let out = iir.process_block(&[1.0; 4]).expect("after reset");
        // After reset: y[0] = 0.1*1 = 0.1
        assert!((out[0] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_block_iir_normalisation() {
        // a[0] = 2 should be normalised: b=[0.2], a=[2, -1.8] => same as b=[0.1], a=[1, -0.9]
        let mut iir1 = BlockIIR::new(&[0.1], &[1.0, -0.9], 4).expect("iir1");
        let mut iir2 = BlockIIR::new(&[0.2], &[2.0, -1.8], 4).expect("iir2");

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out1 = iir1.process_block(&input).expect("process 1");
        let out2 = iir2.process_block(&input).expect("process 2");

        for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "Normalisation mismatch at index {i}: {a} vs {b}"
            );
        }
    }
}
