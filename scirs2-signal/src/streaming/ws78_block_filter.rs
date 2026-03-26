//! Block-based FIR/IIR filters with state preservation for WS78.
//!
//! This module provides:
//!
//! - [`StreamProcessor`] — trait for block-processing components
//! - [`IirState`] — per-SOS-section state vector
//! - [`StatefulIirFilter`] — SOS cascade IIR with preserved state across blocks
//! - [`OverlapSaveFilter`] — FFT-domain FIR using overlap-save with internal ring buffer
//! - [`StatefulFirFilter`] — time-domain FIR using a delay-line ring buffer

use crate::error::{SignalError, SignalResult};
use crate::streaming::ring_buffer::RingBuffer;

// ============================================================================
// StreamProcessor trait
// ============================================================================

/// A block-oriented stream processor.
///
/// Implementations maintain internal state so that successive calls to
/// [`StreamProcessor::process_block`] produce the same result as processing the
/// concatenation of all input blocks in a single call.
pub trait StreamProcessor {
    /// Process one block of input samples, appending results to `output`.
    ///
    /// # Errors
    ///
    /// Implementations may return an error for dimension mismatches or
    /// numerical issues.
    fn process_block(&mut self, input: &[f64], output: &mut Vec<f64>) -> SignalResult<()>;

    /// Reset all internal state to zero.
    fn reset(&mut self);
}

// ============================================================================
// IirState
// ============================================================================

/// Per-section transversal state for one second-order section (SOS) of an IIR
/// filter in Direct Form II Transposed.
///
/// For a single SOS section the recurrence is:
///
/// ```text
/// y[n] = b0*x[n] + w1[n-1]
/// w1[n] = b1*x[n] - a1*y[n] + w2[n-1]
/// w2[n] = b2*x[n] - a2*y[n]
/// ```
///
/// where `zi[0] = w1` and `zi[1] = w2`.
#[derive(Debug, Clone, Default)]
pub struct IirState {
    /// State vector — length = 2 for a second-order section.
    pub zi: Vec<f64>,
}

impl IirState {
    /// Create a new zeroed state for a second-order section.
    pub fn new() -> Self {
        Self { zi: vec![0.0; 2] }
    }

    /// Reset state to zero.
    pub fn reset(&mut self) {
        self.zi.iter_mut().for_each(|v| *v = 0.0);
    }
}

// ============================================================================
// StatefulIirFilter
// ============================================================================

/// IIR filter defined by a Second-Order-Section (SOS) cascade with state
/// preservation across block boundaries.
///
/// The SOS matrix has one row per biquad section; each row contains
/// `[b0, b1, b2, a0, a1, a2]` where `a0` is typically 1.0 (normalised).
///
/// # Example
///
/// ```
/// use scirs2_signal::streaming::ws78_block_filter::{StatefulIirFilter, StreamProcessor};
///
/// // Single SOS section: simple first-order lowpass padded to 2nd order
/// // b = [0.1, 0.0, 0.0],  a = [1.0, -0.9, 0.0]
/// let sos = vec![[0.1_f64, 0.0, 0.0, 1.0, -0.9, 0.0]];
/// let mut filt = StatefulIirFilter::new(sos).expect("valid SOS");
///
/// let input = vec![1.0_f64; 8];
/// let mut out = Vec::new();
/// filt.process_block(&input, &mut out).expect("process");
/// assert_eq!(out.len(), 8);
/// ```
#[derive(Debug, Clone)]
pub struct StatefulIirFilter {
    /// SOS coefficients: one `[b0, b1, b2, a0, a1, a2]` per section.
    sos: Vec<[f64; 6]>,
    /// One state per SOS section.
    states: Vec<IirState>,
}

impl StatefulIirFilter {
    /// Create a new `StatefulIirFilter` from an SOS matrix.
    ///
    /// Each row is `[b0, b1, b2, a0, a1, a2]`.  `a0` must be non-zero.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ValueError`] if `sos` is empty or any `a0` is
    /// zero.
    pub fn new(sos: Vec<[f64; 6]>) -> SignalResult<Self> {
        if sos.is_empty() {
            return Err(SignalError::ValueError(
                "SOS matrix must have at least one section".to_string(),
            ));
        }
        for (i, row) in sos.iter().enumerate() {
            if row[3].abs() < 1e-30 {
                return Err(SignalError::ValueError(format!(
                    "SOS section {i}: a0 ({}) must be non-zero",
                    row[3]
                )));
            }
        }

        let n = sos.len();

        // Normalise so that a0 = 1 for every section.
        let sos_norm: Vec<[f64; 6]> = sos
            .iter()
            .map(|row| {
                let inv_a0 = 1.0 / row[3];
                [
                    row[0] * inv_a0,
                    row[1] * inv_a0,
                    row[2] * inv_a0,
                    1.0,
                    row[4] * inv_a0,
                    row[5] * inv_a0,
                ]
            })
            .collect();

        let states = vec![IirState::new(); n];

        Ok(Self {
            sos: sos_norm,
            states,
        })
    }

    /// Reset all SOS section states to zero.
    pub fn reset(&mut self) {
        self.states.iter_mut().for_each(|s| s.reset());
    }

    /// Number of SOS sections.
    pub fn num_sections(&self) -> usize {
        self.sos.len()
    }
}

impl StreamProcessor for StatefulIirFilter {
    /// Process a block of samples through the SOS cascade.
    ///
    /// The output is appended to `output`.  State is preserved so that
    /// consecutive calls with adjoining blocks give the same result as
    /// processing the full signal at once.
    fn process_block(&mut self, input: &[f64], output: &mut Vec<f64>) -> SignalResult<()> {
        // Work on a mutable copy so we can pipe through sections.
        let mut buf: Vec<f64> = input.to_vec();

        for (sec_idx, row) in self.sos.iter().enumerate() {
            let b0 = row[0];
            let b1 = row[1];
            let b2 = row[2];
            // row[3] == 1.0 (normalised)
            let a1 = row[4];
            let a2 = row[5];

            let state = &mut self.states[sec_idx];

            for x in &mut buf {
                let xv = *x;
                let w1 = state.zi[0];
                let w2 = state.zi[1];
                let y = b0 * xv + w1;
                state.zi[0] = b1 * xv - a1 * y + w2;
                state.zi[1] = b2 * xv - a2 * y;
                *x = y;
            }
        }

        output.extend_from_slice(&buf);
        Ok(())
    }

    fn reset(&mut self) {
        self.reset();
    }
}

// ============================================================================
// OverlapSaveFilter
// ============================================================================

/// FIR filter implemented with the overlap-save method.
///
/// The filter impulse response `h` is stored as a pre-computed frequency-domain
/// representation `H[k] = FFT(h, N)`, where `N` is the next power-of-two
/// above `2 * len(h)`.  On each call to [`OverlapSaveFilter::process_block`]
/// an internal [`RingBuffer`] holds the last `len(h) - 1` input samples to
/// reconstruct the required overlap.
///
/// # Valid samples per block
///
/// `block_size() = N - len(h) + 1`
///
/// # Example
///
/// ```
/// use scirs2_signal::streaming::ws78_block_filter::{OverlapSaveFilter, StreamProcessor};
///
/// let h = vec![1.0_f64, 0.0]; // 2-tap filter → block_size = 2
/// let mut filt = OverlapSaveFilter::new(h).expect("valid filter");
/// let bs = filt.block_size();
/// let input = vec![1.0_f64; bs];
/// let mut out = Vec::new();
/// filt.process_block(&input, &mut out).expect("process");
/// assert_eq!(out.len(), bs);
/// ```
pub struct OverlapSaveFilter {
    /// Filter length M.
    filter_len: usize,
    /// FFT size N (next power-of-two above 2*M).
    fft_size: usize,
    /// Pre-computed complex filter spectrum H[k], length = N/2+1.
    filter_spectrum: Vec<scirs2_core::numeric::Complex64>,
    /// Ring buffer holding the last M-1 input samples.
    overlap_buf: RingBuffer<f64>,
    /// Number of valid output samples per block: N - M + 1.
    block_size: usize,
}

impl OverlapSaveFilter {
    /// Create a new overlap-save FIR filter.
    ///
    /// # Arguments
    ///
    /// * `h` — impulse response coefficients (length M).  Must be non-empty.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ValueError`] if `h` is empty.
    pub fn new(h: Vec<f64>) -> SignalResult<Self> {
        if h.is_empty() {
            return Err(SignalError::ValueError(
                "Filter impulse response must not be empty".to_string(),
            ));
        }

        let m = h.len();
        // N = next power-of-two above 2*M so that block_size = N - M + 1 >= M
        let n = next_pow2((2 * m).max(2));

        // Zero-pad h to length N and compute its spectrum.
        let mut h_padded = vec![0.0_f64; n];
        h_padded[..m].copy_from_slice(&h);
        let filter_spectrum = scirs2_fft::rfft(&h_padded, None)
            .map_err(|e| SignalError::ComputationError(format!("FFT of filter failed: {e}")))?;

        let block_size = n - m + 1;

        // Ring buffer needs to hold M-1 past samples (overlap region).
        let overlap_len = m.saturating_sub(1).max(1);
        let overlap_buf = RingBuffer::<f64>::new(overlap_len).map_err(|e| {
            SignalError::ValueError(format!("Failed to create overlap ring buffer: {e}"))
        })?;

        Ok(Self {
            filter_len: m,
            fft_size: n,
            filter_spectrum,
            overlap_buf,
            block_size,
        })
    }

    /// Number of valid output samples produced per call to `process_block`.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Filter length M.
    pub fn filter_len(&self) -> usize {
        self.filter_len
    }

    /// FFT size N.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Reset internal state (overlap buffer).
    pub fn reset(&mut self) {
        self.overlap_buf.clear();
    }
}

impl StreamProcessor for OverlapSaveFilter {
    /// Process a block of exactly `block_size()` input samples.
    ///
    /// The overlap-save method:
    /// 1. Prepend the last `M-1` stored input samples.
    /// 2. FFT the resulting N-sample frame.
    /// 3. Multiply by `H[k]`.
    /// 4. IFFT.
    /// 5. Return the last `block_size` samples (valid region).
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::DimensionMismatch`] if `input.len() != block_size()`.
    fn process_block(&mut self, input: &[f64], output: &mut Vec<f64>) -> SignalResult<()> {
        if input.len() != self.block_size {
            return Err(SignalError::DimensionMismatch(format!(
                "OverlapSaveFilter::process_block: expected {} samples, got {}",
                self.block_size,
                input.len()
            )));
        }

        let m = self.filter_len;
        let n = self.fft_size;
        let overlap_len = m.saturating_sub(1);

        // Build the N-length frame: [overlap | new_block]
        let mut frame = vec![0.0_f64; n];

        // Fill overlap region from ring buffer (oldest first).
        let stored = self.overlap_buf.as_ordered_vec();
        let stored_len = stored.len();
        // If ring buffer has fewer than overlap_len samples (startup), zero-pad.
        if stored_len <= overlap_len {
            let pad = overlap_len - stored_len;
            for (i, &v) in stored.iter().enumerate() {
                frame[pad + i] = v;
            }
        } else {
            for (i, &v) in stored.iter().enumerate() {
                frame[i] = v;
            }
        }

        // Fill new block.
        frame[overlap_len..overlap_len + self.block_size].copy_from_slice(input);

        // Forward FFT of frame.
        let x_spec = scirs2_fft::rfft(&frame, None)
            .map_err(|e| SignalError::ComputationError(format!("rfft failed: {e}")))?;

        // Pointwise multiply by H.
        let y_spec: Vec<scirs2_core::numeric::Complex64> = x_spec
            .iter()
            .zip(self.filter_spectrum.iter())
            .map(|(&x, &h)| x * h)
            .collect();

        // Inverse FFT.
        let y_time = scirs2_fft::irfft(&y_spec, Some(n))
            .map_err(|e| SignalError::ComputationError(format!("irfft failed: {e}")))?;

        // Valid output: last block_size samples of the N-sample result.
        let valid_start = overlap_len;
        output.extend_from_slice(&y_time[valid_start..valid_start + self.block_size]);

        // Update overlap buffer: store last M-1 samples from *input*.
        if overlap_len > 0 {
            // Push as much of the input tail as possible.
            let from = if self.block_size >= overlap_len {
                self.block_size - overlap_len
            } else {
                0
            };
            for &v in &input[from..] {
                self.overlap_buf.push(v);
            }
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.reset();
    }
}

// ============================================================================
// StatefulFirFilter
// ============================================================================

/// Time-domain FIR filter using a ring-buffer delay line.
///
/// Unlike [`OverlapSaveFilter`], this operates sample-by-sample within each
/// block and is suitable for short filters where FFT overhead is undesirable.
///
/// # Example
///
/// ```
/// use scirs2_signal::streaming::ws78_block_filter::{StatefulFirFilter, StreamProcessor};
///
/// let taps = vec![0.25_f64, 0.5, 0.25]; // 3-tap smoothing
/// let mut filt = StatefulFirFilter::new(taps).expect("valid");
/// let input = vec![1.0_f64, 2.0, 3.0, 4.0];
/// let mut out = Vec::new();
/// filt.process_block(&input, &mut out).expect("process");
/// assert_eq!(out.len(), 4);
/// ```
pub struct StatefulFirFilter {
    /// Filter coefficients (taps).
    coeffs: Vec<f64>,
    /// Circular delay-line ring buffer, length = num_taps.
    delay_line: RingBuffer<f64>,
}

impl StatefulFirFilter {
    /// Create a new time-domain FIR filter with the given tap weights.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::ValueError`] if `taps` is empty.
    pub fn new(taps: Vec<f64>) -> SignalResult<Self> {
        if taps.is_empty() {
            return Err(SignalError::ValueError(
                "FIR taps must not be empty".to_string(),
            ));
        }
        let n = taps.len();
        let delay_line = RingBuffer::<f64>::new(n).map_err(|e| {
            SignalError::ValueError(format!("Failed to create FIR delay line: {e}"))
        })?;
        Ok(Self {
            coeffs: taps,
            delay_line,
        })
    }

    /// Number of taps.
    pub fn num_taps(&self) -> usize {
        self.coeffs.len()
    }

    /// Reset delay line to zero.
    pub fn reset(&mut self) {
        self.delay_line.clear();
    }
}

impl StreamProcessor for StatefulFirFilter {
    fn process_block(&mut self, input: &[f64], output: &mut Vec<f64>) -> SignalResult<()> {
        let n = self.coeffs.len();
        for &x in input {
            self.delay_line.push(x);
            // Compute dot product with tap weights.
            // delay_line newest element is at logical index len-1 = n-1.
            let mut y = 0.0_f64;
            let len = self.delay_line.len();
            for (k, &coeff) in self.coeffs.iter().enumerate() {
                // tap k corresponds to delay k: newest - k
                if k < len {
                    let idx = len - 1 - k;
                    let val = self.delay_line.get(idx).map(|v| *v).unwrap_or(0.0);
                    y += coeff * val;
                }
            }
            output.push(y);
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.reset();
    }
}

// ============================================================================
// Helper
// ============================================================================

fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1usize;
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

    // ---- StatefulIirFilter ----

    #[test]
    fn test_stateful_iir_output_shape() {
        // Single SOS section: b=[0.1,0,0], a=[1,-0.9,0]
        let sos = vec![[0.1_f64, 0.0, 0.0, 1.0, -0.9, 0.0]];
        let mut filt = StatefulIirFilter::new(sos).expect("create");
        let input: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let mut out = Vec::new();
        filt.process_block(&input, &mut out).expect("process");
        assert_eq!(out.len(), input.len(), "Output length must match input");
    }

    #[test]
    fn test_stateful_iir_state_preserved_across_blocks() {
        // Processing two consecutive blocks must equal processing the
        // concatenated signal in one shot.
        let sos = vec![[0.5_f64, 0.3, 0.0, 1.0, -0.5, 0.0]];

        let full_input: Vec<f64> = (0..32).map(|i| (i as f64 * 0.3).sin()).collect();
        let block1 = &full_input[..16];
        let block2 = &full_input[16..];

        // One-shot
        let mut filt_one = StatefulIirFilter::new(sos.clone()).expect("create");
        let mut out_one = Vec::new();
        filt_one
            .process_block(&full_input, &mut out_one)
            .expect("one-shot");

        // Two blocks
        let mut filt_two = StatefulIirFilter::new(sos).expect("create");
        let mut out_two = Vec::new();
        filt_two.process_block(block1, &mut out_two).expect("b1");
        filt_two.process_block(block2, &mut out_two).expect("b2");

        assert_eq!(out_one.len(), out_two.len());
        for (i, (&a, &b)) in out_one.iter().zip(out_two.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "Mismatch at index {i}: one-shot={a}, two-block={b}"
            );
        }
    }

    #[test]
    fn test_stateful_iir_reset_clears_state() {
        let sos = vec![[0.1_f64, 0.0, 0.0, 1.0, -0.9, 0.0]];
        let mut filt = StatefulIirFilter::new(sos).expect("create");

        let input = vec![1.0_f64; 8];
        let mut out1 = Vec::new();
        filt.process_block(&input, &mut out1).expect("first");
        filt.reset();

        let mut out2 = Vec::new();
        filt.process_block(&input, &mut out2).expect("after reset");

        // After reset, the two outputs should be identical.
        for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "After reset, outputs differ at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_stateful_iir_empty_sos_error() {
        assert!(StatefulIirFilter::new(vec![]).is_err());
    }

    #[test]
    fn test_stateful_iir_zero_a0_error() {
        let bad_sos = vec![[0.1_f64, 0.0, 0.0, 0.0, -0.9, 0.0]];
        assert!(StatefulIirFilter::new(bad_sos).is_err());
    }

    // ---- OverlapSaveFilter ----

    #[test]
    fn test_overlap_save_filter_identity() {
        // h=[1.0]: identity filter
        let mut filt = OverlapSaveFilter::new(vec![1.0_f64]).expect("create");
        let input: Vec<f64> = (0..filt.block_size()).map(|i| i as f64 + 1.0).collect();
        let mut out = Vec::new();
        filt.process_block(&input, &mut out).expect("process");
        assert_eq!(out.len(), input.len());
        for (i, (&y, &x)) in out.iter().zip(input.iter()).enumerate() {
            assert!(
                (y - x).abs() < 1e-8,
                "Identity filter mismatch at {i}: got {y}, expected {x}"
            );
        }
    }

    #[test]
    fn test_overlap_save_filter_dimension_mismatch() {
        let mut filt = OverlapSaveFilter::new(vec![1.0_f64, 0.5]).expect("create");
        let bad_input = vec![1.0_f64; filt.block_size() + 1];
        let mut out = Vec::new();
        assert!(filt.process_block(&bad_input, &mut out).is_err());
    }

    #[test]
    fn test_overlap_save_filter_block_size_positive() {
        let filt = OverlapSaveFilter::new(vec![0.25_f64, 0.5, 0.25]).expect("create");
        assert!(filt.block_size() > 0, "block_size must be > 0");
    }

    #[test]
    fn test_overlap_save_filter_reset() {
        let mut filt = OverlapSaveFilter::new(vec![0.5_f64, 0.5]).expect("create");
        let bs = filt.block_size();
        let input: Vec<f64> = vec![1.0; bs];
        let mut out = Vec::new();
        filt.process_block(&input, &mut out).expect("process");
        filt.reset();

        // After reset, a fresh filter should give identical output.
        let mut fresh = OverlapSaveFilter::new(vec![0.5_f64, 0.5]).expect("create");
        let mut out_after = Vec::new();
        let mut out_fresh = Vec::new();
        filt.process_block(&input, &mut out_after)
            .expect("after reset");
        fresh.process_block(&input, &mut out_fresh).expect("fresh");

        for (i, (&a, &b)) in out_after.iter().zip(out_fresh.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "Reset output mismatch at {i}: {a} vs {b}"
            );
        }
    }

    // ---- StatefulFirFilter ----

    #[test]
    fn test_stateful_fir_output_shape() {
        let taps = vec![0.25_f64, 0.5, 0.25];
        let mut filt = StatefulFirFilter::new(taps).expect("create");
        let input: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let mut out = Vec::new();
        filt.process_block(&input, &mut out).expect("process");
        assert_eq!(out.len(), input.len());
    }

    #[test]
    fn test_stateful_fir_state_preserved_across_blocks() {
        let taps = vec![0.3_f64, 0.5, 0.2];
        let full: Vec<f64> = (0..24).map(|i| (i as f64 * 0.2).cos()).collect();

        let mut filt_one = StatefulFirFilter::new(taps.clone()).expect("create");
        let mut out_one = Vec::new();
        filt_one.process_block(&full, &mut out_one).expect("one");

        let mut filt_two = StatefulFirFilter::new(taps).expect("create");
        let mut out_two = Vec::new();
        filt_two
            .process_block(&full[..12], &mut out_two)
            .expect("b1");
        filt_two
            .process_block(&full[12..], &mut out_two)
            .expect("b2");

        assert_eq!(out_one.len(), out_two.len());
        for (i, (&a, &b)) in out_one.iter().zip(out_two.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "FIR state mismatch at {i}: one={a} two={b}"
            );
        }
    }

    #[test]
    fn test_stateful_fir_reset() {
        let taps = vec![0.5_f64, 0.5];
        let mut filt = StatefulFirFilter::new(taps).expect("create");
        let input = vec![1.0_f64; 4];
        let mut out = Vec::new();
        filt.process_block(&input, &mut out).expect("first block");
        filt.reset();

        let mut out2 = Vec::new();
        filt.process_block(&input, &mut out2).expect("after reset");
        // After reset, output should match first run.
        for (i, (&a, &b)) in out.iter().zip(out2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "FIR reset mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_stateful_fir_empty_taps_error() {
        assert!(StatefulFirFilter::new(vec![]).is_err());
    }
}
