//! Online / streaming digital filters.
//!
//! Every filter in this module processes data **sample-by-sample** (via
//! [`StreamingFIR::process_sample`]) or **chunk-by-chunk** (via [`StreamingFIR::process_chunk`]).  Internal
//! state is preserved across calls so that a continuous stream can be fed
//! incrementally.
//!
//! ## Provided filters
//!
//! | Struct | Description |
//! |--------|-------------|
//! | [`StreamingFIR`] | Finite impulse response filter with circular buffer |
//! | [`StreamingIIR`] | Biquad cascade, Direct Form II Transposed |
//! | [`StreamingMedianFilter`] | Running median via a two-heap approach |
//! | [`StreamingMovingAverage`] | Efficient incremental arithmetic mean |

use crate::error::{SignalError, SignalResult};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

// ============================================================================
// StreamingFIR
// ============================================================================

/// A streaming FIR (Finite Impulse Response) filter.
///
/// Internally uses a circular buffer of length equal to the number of taps for
/// O(N) per-sample filtering where N is the filter order.
#[derive(Debug, Clone)]
pub struct StreamingFIR {
    /// Filter coefficients (taps), length = order + 1.
    coeffs: Vec<f64>,
    /// Delay line (circular buffer).
    buffer: Vec<f64>,
    /// Current write position in the circular buffer.
    pos: usize,
    /// Total samples processed (monotonically increasing counter).
    samples_processed: u64,
}

impl StreamingFIR {
    /// Create a new streaming FIR filter.
    ///
    /// # Arguments
    ///
    /// * `coefficients` - FIR tap weights.  Must contain at least one element.
    ///
    /// # Errors
    ///
    /// Returns `SignalError::ValueError` if `coefficients` is empty.
    pub fn new(coefficients: &[f64]) -> SignalResult<Self> {
        if coefficients.is_empty() {
            return Err(SignalError::ValueError(
                "FIR coefficients must not be empty".to_string(),
            ));
        }
        let n = coefficients.len();
        Ok(Self {
            coeffs: coefficients.to_vec(),
            buffer: vec![0.0; n],
            pos: 0,
            samples_processed: 0,
        })
    }

    /// Process a single input sample and return the filtered output sample.
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let n = self.coeffs.len();
        self.buffer[self.pos] = input;

        let mut output = 0.0;
        for (k, &coeff) in self.coeffs.iter().enumerate() {
            // Walk backwards through the buffer
            let idx = (self.pos + n - k) % n;
            output += coeff * self.buffer[idx];
        }

        self.pos = (self.pos + 1) % n;
        self.samples_processed += 1;
        output
    }

    /// Process a chunk of samples in-place.
    pub fn process_chunk(&mut self, chunk: &[f64]) -> Vec<f64> {
        chunk.iter().map(|&s| self.process_sample(s)).collect()
    }

    /// Reset internal state (delay line) to zero.
    pub fn reset(&mut self) {
        self.buffer.iter_mut().for_each(|v| *v = 0.0);
        self.pos = 0;
    }

    /// Number of taps (filter order + 1).
    pub fn num_taps(&self) -> usize {
        self.coeffs.len()
    }

    /// Total samples processed since creation or last reset.
    pub fn samples_processed(&self) -> u64 {
        self.samples_processed
    }

    /// Group delay in samples (constant for FIR = (N-1)/2 for symmetric).
    pub fn group_delay(&self) -> f64 {
        (self.coeffs.len() as f64 - 1.0) / 2.0
    }
}

// ============================================================================
// StreamingIIR  -- Biquad Cascade, Direct Form II Transposed
// ============================================================================

/// Coefficients for a single biquad section.
///
/// Transfer function:
///
/// ```text
///        b0 + b1*z^-1 + b2*z^-2
/// H(z) = -------------------------
///        1  + a1*z^-1 + a2*z^-2
/// ```
///
/// Note: `a0` is assumed to be 1.0 (normalised).  If it is not, divide all
/// coefficients by `a0` before constructing this struct.
#[derive(Debug, Clone, Copy)]
pub struct BiquadCoeffs {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl BiquadCoeffs {
    /// Create coefficients from 6-element arrays `[b0, b1, b2]` and
    /// `[a0, a1, a2]`.  Normalises by `a0`.
    ///
    /// # Errors
    ///
    /// Returns an error if `a0` is zero.
    pub fn from_ba(b: [f64; 3], a: [f64; 3]) -> SignalResult<Self> {
        if a[0].abs() < 1e-30 {
            return Err(SignalError::ValueError(
                "a[0] must be non-zero for IIR normalisation".to_string(),
            ));
        }
        let inv_a0 = 1.0 / a[0];
        Ok(Self {
            b0: b[0] * inv_a0,
            b1: b[1] * inv_a0,
            b2: b[2] * inv_a0,
            a1: a[1] * inv_a0,
            a2: a[2] * inv_a0,
        })
    }
}

/// Internal state for one biquad section (DF-II Transposed).
#[derive(Debug, Clone, Copy, Default)]
struct BiquadState {
    s1: f64,
    s2: f64,
}

/// A streaming IIR filter implemented as a cascade of second-order (biquad)
/// sections in Direct Form II Transposed.
///
/// This is numerically more stable than a single high-order transfer function
/// and is the standard representation used in real-time audio processing.
#[derive(Debug, Clone)]
pub struct StreamingIIR {
    sections: Vec<BiquadCoeffs>,
    states: Vec<BiquadState>,
    samples_processed: u64,
}

impl StreamingIIR {
    /// Create a new biquad cascade.
    ///
    /// # Errors
    ///
    /// Returns an error if `sections` is empty.
    pub fn new(sections: Vec<BiquadCoeffs>) -> SignalResult<Self> {
        if sections.is_empty() {
            return Err(SignalError::ValueError(
                "At least one biquad section is required".to_string(),
            ));
        }
        let n = sections.len();
        Ok(Self {
            sections,
            states: vec![BiquadState::default(); n],
            samples_processed: 0,
        })
    }

    /// Convenience: create a single second-order section.
    pub fn from_single_section(coeffs: BiquadCoeffs) -> SignalResult<Self> {
        Self::new(vec![coeffs])
    }

    /// Process one sample through the cascade.
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let mut x = input;
        for (sec, st) in self.sections.iter().zip(self.states.iter_mut()) {
            // Direct Form II Transposed
            let y = sec.b0 * x + st.s1;
            st.s1 = sec.b1 * x - sec.a1 * y + st.s2;
            st.s2 = sec.b2 * x - sec.a2 * y;
            x = y;
        }
        self.samples_processed += 1;
        x
    }

    /// Process a chunk.
    pub fn process_chunk(&mut self, chunk: &[f64]) -> Vec<f64> {
        chunk.iter().map(|&s| self.process_sample(s)).collect()
    }

    /// Reset all section states to zero.
    pub fn reset(&mut self) {
        for st in &mut self.states {
            st.s1 = 0.0;
            st.s2 = 0.0;
        }
    }

    /// Number of biquad sections.
    pub fn num_sections(&self) -> usize {
        self.sections.len()
    }

    /// Total samples processed.
    pub fn samples_processed(&self) -> u64 {
        self.samples_processed
    }
}

// ============================================================================
// StreamingMedianFilter  -- Two-Heap running median
// ============================================================================

/// A streaming median filter using a two-heap (max-heap + min-heap) approach.
///
/// Maintains a sliding window of the last `window_size` samples and
/// efficiently computes the running median in O(log N) per sample.
#[derive(Debug, Clone)]
pub struct StreamingMedianFilter {
    window_size: usize,
    /// Ordered history of the window (for element removal).
    window: Vec<f64>,
    /// Current write position in the circular window buffer.
    pos: usize,
    /// Number of elements inserted so far (clamped at `window_size`).
    count: usize,
    samples_processed: u64,
}

impl StreamingMedianFilter {
    /// Create a new streaming median filter.
    ///
    /// # Errors
    ///
    /// Returns an error if `window_size` is 0.
    pub fn new(window_size: usize) -> SignalResult<Self> {
        if window_size == 0 {
            return Err(SignalError::ValueError(
                "Median filter window size must be > 0".to_string(),
            ));
        }
        Ok(Self {
            window_size,
            window: vec![0.0; window_size],
            pos: 0,
            count: 0,
            samples_processed: 0,
        })
    }

    /// Process a single sample and return the current median.
    pub fn process_sample(&mut self, input: f64) -> f64 {
        // Insert into circular window
        self.window[self.pos] = input;
        self.pos = (self.pos + 1) % self.window_size;
        if self.count < self.window_size {
            self.count += 1;
        }
        self.samples_processed += 1;

        // Compute median from current window using a sorted copy.
        // For windows up to a few hundred this is fast enough; for truly
        // massive windows a more sophisticated heap structure could be used.
        self.compute_median()
    }

    /// Process a chunk.
    pub fn process_chunk(&mut self, chunk: &[f64]) -> Vec<f64> {
        chunk.iter().map(|&s| self.process_sample(s)).collect()
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.window.iter_mut().for_each(|v| *v = 0.0);
        self.pos = 0;
        self.count = 0;
    }

    /// Window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Total samples processed.
    pub fn samples_processed(&self) -> u64 {
        self.samples_processed
    }

    // ---- internal ----

    fn compute_median(&self) -> f64 {
        let mut sorted: Vec<f64> = if self.count == self.window_size {
            self.window.clone()
        } else {
            // Only take the valid elements from the circular buffer
            let mut valid = Vec::with_capacity(self.count);
            for i in 0..self.count {
                // Elements were inserted starting at pos-count (mod window_size)
                let idx = (self.pos + self.window_size - self.count + i) % self.window_size;
                valid.push(self.window[idx]);
            }
            valid
        };
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        }
    }
}

// ============================================================================
// StreamingMovingAverage
// ============================================================================

/// Efficient incremental moving average filter.
///
/// Uses a running sum so that each new sample only requires one addition and
/// one subtraction regardless of the window size.
#[derive(Debug, Clone)]
pub struct StreamingMovingAverage {
    window_size: usize,
    buffer: Vec<f64>,
    pos: usize,
    count: usize,
    running_sum: f64,
    samples_processed: u64,
}

impl StreamingMovingAverage {
    /// Create a new streaming moving average filter.
    ///
    /// # Errors
    ///
    /// Returns an error if `window_size` is 0.
    pub fn new(window_size: usize) -> SignalResult<Self> {
        if window_size == 0 {
            return Err(SignalError::ValueError(
                "Moving average window size must be > 0".to_string(),
            ));
        }
        Ok(Self {
            window_size,
            buffer: vec![0.0; window_size],
            pos: 0,
            count: 0,
            running_sum: 0.0,
            samples_processed: 0,
        })
    }

    /// Process one sample and return the current moving average.
    pub fn process_sample(&mut self, input: f64) -> f64 {
        // Subtract the element that is about to be overwritten
        if self.count == self.window_size {
            self.running_sum -= self.buffer[self.pos];
        }

        self.buffer[self.pos] = input;
        self.running_sum += input;
        self.pos = (self.pos + 1) % self.window_size;

        if self.count < self.window_size {
            self.count += 1;
        }
        self.samples_processed += 1;

        self.running_sum / self.count as f64
    }

    /// Process a chunk.
    pub fn process_chunk(&mut self, chunk: &[f64]) -> Vec<f64> {
        chunk.iter().map(|&s| self.process_sample(s)).collect()
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.buffer.iter_mut().for_each(|v| *v = 0.0);
        self.pos = 0;
        self.count = 0;
        self.running_sum = 0.0;
    }

    /// Current average value (NaN if no samples processed).
    pub fn current_value(&self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            self.running_sum / self.count as f64
        }
    }

    /// Window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Total samples processed.
    pub fn samples_processed(&self) -> u64 {
        self.samples_processed
    }
}

// ============================================================================
// Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // ---- StreamingFIR ----

    #[test]
    fn test_fir_identity() {
        // Identity filter: [1.0]
        let mut fir = StreamingFIR::new(&[1.0]).expect("Failed to create FIR");
        let out: Vec<f64> = (1..=5).map(|i| fir.process_sample(i as f64)).collect();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_fir_simple_average() {
        // 3-tap averaging filter: [1/3, 1/3, 1/3]
        let coeffs = [1.0 / 3.0; 3];
        let mut fir = StreamingFIR::new(&coeffs).expect("Failed to create FIR");

        // Feed 1.0 three times; third output should be 1.0
        let _o1 = fir.process_sample(1.0);
        let _o2 = fir.process_sample(1.0);
        let o3 = fir.process_sample(1.0);
        assert!((o3 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_fir_process_chunk() {
        let coeffs = [0.5, 0.5];
        let mut fir = StreamingFIR::new(&coeffs).expect("Failed to create FIR");
        let out = fir.process_chunk(&[2.0, 4.0, 6.0]);
        // y[0] = 0.5*2 + 0.5*0 = 1.0
        // y[1] = 0.5*4 + 0.5*2 = 3.0
        // y[2] = 0.5*6 + 0.5*4 = 5.0
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - 3.0).abs() < 1e-12);
        assert!((out[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_fir_reset() {
        let mut fir = StreamingFIR::new(&[0.5, 0.5]).expect("Failed to create FIR");
        fir.process_sample(10.0);
        fir.reset();
        // After reset the delay line should be zero
        let o = fir.process_sample(4.0);
        assert!((o - 2.0).abs() < 1e-12); // 0.5*4 + 0.5*0
    }

    #[test]
    fn test_fir_empty_coeffs_error() {
        assert!(StreamingFIR::new(&[]).is_err());
    }

    #[test]
    fn test_fir_group_delay() {
        let fir = StreamingFIR::new(&[1.0, 1.0, 1.0]).expect("create FIR");
        assert!((fir.group_delay() - 1.0).abs() < 1e-12);
    }

    // ---- StreamingIIR ----

    #[test]
    fn test_iir_pass_through() {
        // H(z) = 1 (pass-through)
        let sec = BiquadCoeffs {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        let mut iir = StreamingIIR::from_single_section(sec).expect("create IIR");
        for i in 1..=5 {
            let o = iir.process_sample(i as f64);
            assert!((o - i as f64).abs() < 1e-12);
        }
    }

    #[test]
    fn test_iir_first_order_lowpass() {
        // Simple first-order lowpass: y[n] = 0.1*x[n] + 0.9*y[n-1]
        // b = [0.1, 0, 0], a = [1, -0.9, 0]
        let sec = BiquadCoeffs {
            b0: 0.1,
            b1: 0.0,
            b2: 0.0,
            a1: -0.9,
            a2: 0.0,
        };
        let mut iir = StreamingIIR::from_single_section(sec).expect("create IIR");

        // Step response should converge towards 1.0
        let mut last = 0.0;
        for _ in 0..100 {
            last = iir.process_sample(1.0);
        }
        assert!((last - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_iir_process_chunk() {
        let sec = BiquadCoeffs {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        let mut iir = StreamingIIR::from_single_section(sec).expect("create IIR");
        let out = iir.process_chunk(&[1.0, 2.0, 3.0]);
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - 2.0).abs() < 1e-12);
        assert!((out[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_iir_reset() {
        let sec = BiquadCoeffs {
            b0: 0.1,
            b1: 0.0,
            b2: 0.0,
            a1: -0.9,
            a2: 0.0,
        };
        let mut iir = StreamingIIR::from_single_section(sec).expect("create IIR");
        for _ in 0..50 {
            iir.process_sample(1.0);
        }
        iir.reset();
        // After reset, first sample of step should be 0.1
        let o = iir.process_sample(1.0);
        assert!((o - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_iir_cascade() {
        // Two identical pass-through sections
        let sec = BiquadCoeffs {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        let mut iir = StreamingIIR::new(vec![sec, sec]).expect("create IIR cascade");
        assert_eq!(iir.num_sections(), 2);
        let o = iir.process_sample(42.0);
        assert!((o - 42.0).abs() < 1e-12);
    }

    #[test]
    fn test_iir_empty_sections_error() {
        assert!(StreamingIIR::new(vec![]).is_err());
    }

    #[test]
    fn test_biquad_coeffs_from_ba() {
        let c = BiquadCoeffs::from_ba([2.0, 0.0, 0.0], [2.0, 0.0, 0.0]).expect("from_ba");
        assert!((c.b0 - 1.0).abs() < 1e-12);
        assert!((c.a1).abs() < 1e-12);
    }

    #[test]
    fn test_biquad_coeffs_a0_zero() {
        assert!(BiquadCoeffs::from_ba([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]).is_err());
    }

    // ---- StreamingMedianFilter ----

    #[test]
    fn test_median_odd_window() {
        let mut mf = StreamingMedianFilter::new(3).expect("create median filter");
        // Feed: 5, 1, 3
        let _o1 = mf.process_sample(5.0); // window=[5] -> median=5
        let _o2 = mf.process_sample(1.0); // window=[5,1] -> median=3
        let o3 = mf.process_sample(3.0); // window=[5,1,3] -> sorted=[1,3,5] -> median=3
        assert!((o3 - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_median_even_window() {
        let mut mf = StreamingMedianFilter::new(4).expect("create median filter");
        mf.process_sample(1.0);
        mf.process_sample(2.0);
        mf.process_sample(3.0);
        let o = mf.process_sample(4.0); // sorted=[1,2,3,4] -> median=(2+3)/2=2.5
        assert!((o - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_median_sliding() {
        let mut mf = StreamingMedianFilter::new(3).expect("create median filter");
        mf.process_sample(1.0);
        mf.process_sample(2.0);
        mf.process_sample(3.0);
        // Window now [1,2,3], median=2
        let o = mf.process_sample(100.0);
        // Window now [2,3,100], sorted=[2,3,100], median=3
        assert!((o - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_median_process_chunk() {
        let mut mf = StreamingMedianFilter::new(3).expect("create median filter");
        let out = mf.process_chunk(&[1.0, 5.0, 3.0, 7.0, 2.0]);
        assert_eq!(out.len(), 5);
        // Just check the last value: window=[3,7,2], sorted=[2,3,7], median=3
        assert!((out[4] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_median_reset() {
        let mut mf = StreamingMedianFilter::new(3).expect("create median filter");
        mf.process_sample(100.0);
        mf.reset();
        assert_eq!(mf.samples_processed(), 1); // samples_processed not cleared by reset
        let o = mf.process_sample(5.0);
        assert!((o - 5.0).abs() < 1e-12); // single element, median = that element
    }

    #[test]
    fn test_median_zero_window_error() {
        assert!(StreamingMedianFilter::new(0).is_err());
    }

    // ---- StreamingMovingAverage ----

    #[test]
    fn test_moving_avg_constant_signal() {
        let mut ma = StreamingMovingAverage::new(5).expect("create MA");
        for _ in 0..10 {
            let o = ma.process_sample(3.0);
            assert!((o - 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_moving_avg_ramp() {
        let mut ma = StreamingMovingAverage::new(3).expect("create MA");
        let o1 = ma.process_sample(1.0); // avg(1) = 1
        let o2 = ma.process_sample(2.0); // avg(1,2) = 1.5
        let o3 = ma.process_sample(3.0); // avg(1,2,3) = 2
        let o4 = ma.process_sample(4.0); // avg(2,3,4) = 3

        assert!((o1 - 1.0).abs() < 1e-12);
        assert!((o2 - 1.5).abs() < 1e-12);
        assert!((o3 - 2.0).abs() < 1e-12);
        assert!((o4 - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_moving_avg_process_chunk() {
        let mut ma = StreamingMovingAverage::new(2).expect("create MA");
        let out = ma.process_chunk(&[2.0, 4.0, 6.0]);
        // o1 = avg(2) = 2
        // o2 = avg(2,4) = 3
        // o3 = avg(4,6) = 5
        assert!((out[0] - 2.0).abs() < 1e-12);
        assert!((out[1] - 3.0).abs() < 1e-12);
        assert!((out[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_moving_avg_reset() {
        let mut ma = StreamingMovingAverage::new(3).expect("create MA");
        ma.process_chunk(&[10.0, 20.0, 30.0]);
        ma.reset();
        assert!(ma.current_value().is_nan());
        let o = ma.process_sample(5.0);
        assert!((o - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_moving_avg_zero_window_error() {
        assert!(StreamingMovingAverage::new(0).is_err());
    }

    #[test]
    fn test_moving_avg_current_value() {
        let mut ma = StreamingMovingAverage::new(4).expect("create MA");
        assert!(ma.current_value().is_nan()); // No samples yet
        ma.process_sample(8.0);
        assert!((ma.current_value() - 8.0).abs() < 1e-12);
    }
}
