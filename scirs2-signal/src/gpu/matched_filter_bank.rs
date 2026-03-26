// FFT-based matched filter bank
//
// Pre-computes the FFT of each template, then correlates with incoming signals
// via overlap-save: IFFT(FFT(signal) · conj(FFT(template))).
//
// Because the input / templates are real, we use scirs2-fft's `rfft` / `irfft`
// and obtain the one-sided spectrum; `irfft` recovers the real cross-correlation.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::num_complex::Complex64;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`MatchedFilterBank`].
#[derive(Debug, Clone)]
pub struct MatchedFilterConfig {
    /// When `true` each template is normalised to unit energy before the FFT.
    /// Default: `true`.
    pub normalize_templates: bool,
    /// SNR threshold for peak detection (multiples of σ). Default: `3.0`.
    pub peak_threshold_snr: f32,
}

impl Default for MatchedFilterConfig {
    fn default() -> Self {
        Self {
            normalize_templates: true,
            peak_threshold_snr: 3.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Detection result
// ---------------------------------------------------------------------------

/// A single detection event returned by [`MatchedFilterBank::detect_peaks`].
#[derive(Debug, Clone)]
pub struct Detection {
    /// Index of the template that produced this detection.
    pub template_idx: usize,
    /// Sample offset of the peak in the output cross-correlation.
    pub sample_offset: usize,
    /// Correlation value at the peak.
    pub correlation_value: f32,
    /// Estimated SNR: `correlation_value / noise_rms`.
    pub snr: f32,
}

// ---------------------------------------------------------------------------
// Helper: next power of two
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// MatchedFilterBank
// ---------------------------------------------------------------------------

/// A bank of matched filters processed via FFT cross-correlation.
pub struct MatchedFilterBank {
    /// One-sided FFT of each zero-padded template: shape \[n_templates\]\[fft_len/2+1\].
    templates_fft: Vec<Vec<Complex64>>,
    /// Per-template energy norms (after optional normalisation).
    template_norms: Vec<f32>,
    /// Length of the input signal this bank was built for.
    signal_length: usize,
    /// Zero-padded FFT length = next_pow2(signal_length + template_length - 1).
    fft_length: usize,
    config: MatchedFilterConfig,
}

impl MatchedFilterBank {
    /// Build the filter bank.
    ///
    /// # Arguments
    ///
    /// * `templates`      – `[n_templates, template_length]`
    /// * `signal_length`  – Number of samples in each input signal
    /// * `config`         – See [`MatchedFilterConfig`]
    pub fn new(
        templates: &Array2<f32>,
        signal_length: usize,
        config: MatchedFilterConfig,
    ) -> SignalResult<Self> {
        let n_templates = templates.nrows();
        let template_length = templates.ncols();

        if n_templates == 0 {
            return Err(SignalError::InvalidArgument(
                "templates must have at least one row".into(),
            ));
        }
        if template_length == 0 {
            return Err(SignalError::InvalidArgument(
                "template_length must be > 0".into(),
            ));
        }
        if signal_length == 0 {
            return Err(SignalError::InvalidArgument(
                "signal_length must be > 0".into(),
            ));
        }

        // FFT length for linear (non-circular) correlation
        let fft_length = next_pow2(signal_length + template_length - 1);
        let n_freq = fft_length / 2 + 1;

        let mut templates_fft: Vec<Vec<Complex64>> = Vec::with_capacity(n_templates);
        let mut template_norms: Vec<f32> = Vec::with_capacity(n_templates);

        for t in 0..n_templates {
            let row = templates.row(t);

            // Compute energy norm
            let energy: f32 = row.iter().map(|&v| v * v).sum();
            let norm = energy.sqrt().max(f32::EPSILON);
            template_norms.push(norm);

            // Build zero-padded buffer (f64 for rfft)
            let mut padded = vec![0.0f64; fft_length];
            for (i, &v) in row.iter().enumerate() {
                if config.normalize_templates {
                    padded[i] = (v / norm) as f64;
                } else {
                    padded[i] = v as f64;
                }
            }

            // RFFT → one-sided complex spectrum
            let spectrum = scirs2_fft::rfft(&padded, None)
                .map_err(|e| SignalError::ComputationError(format!("template rfft error: {e}")))?;

            // Sanity-check length
            if spectrum.len() < n_freq {
                return Err(SignalError::ComputationError(
                    "rfft returned fewer bins than expected".into(),
                ));
            }

            templates_fft.push(spectrum.into_iter().take(n_freq).collect());
        }

        Ok(Self {
            templates_fft,
            template_norms,
            signal_length,
            fft_length,
            config,
        })
    }

    /// Number of templates in the bank.
    #[inline]
    pub fn n_templates(&self) -> usize {
        self.templates_fft.len()
    }

    /// Compute cross-correlations between the signal and all templates.
    ///
    /// # Returns
    ///
    /// `[n_templates, signal_length]` array of cross-correlation values.
    pub fn correlate_all(&self, signal: &Array1<f32>) -> SignalResult<Array2<f32>> {
        if signal.len() != self.signal_length {
            return Err(SignalError::DimensionMismatch(format!(
                "expected signal of length {}, got {}",
                self.signal_length,
                signal.len()
            )));
        }

        let n_freq = self.fft_length / 2 + 1;

        // Zero-pad signal to fft_length and compute its RFFT
        let mut sig_padded = vec![0.0f64; self.fft_length];
        for (i, &v) in signal.iter().enumerate() {
            sig_padded[i] = v as f64;
        }
        let sig_fft = scirs2_fft::rfft(&sig_padded, None)
            .map_err(|e| SignalError::ComputationError(format!("signal rfft error: {e}")))?;

        let n_templates = self.n_templates();
        let mut output = Array2::<f32>::zeros((n_templates, self.signal_length));

        for (t, tmpl_fft) in self.templates_fft.iter().enumerate() {
            // Pointwise: signal_fft * conj(template_fft)
            let mut product: Vec<Complex64> = sig_fft
                .iter()
                .take(n_freq)
                .zip(tmpl_fft.iter())
                .map(|(&s, &tmpl)| s * tmpl.conj())
                .collect();

            // Pad to n_freq if necessary (shouldn't happen but safety first)
            product.resize(n_freq, Complex64::new(0.0, 0.0));

            // IRFFT → real cross-correlation (length fft_length)
            let xcorr = scirs2_fft::irfft(&product, Some(self.fft_length)).map_err(|e| {
                SignalError::ComputationError(format!("irfft error in matched filter: {e}"))
            })?;

            // Trim to signal_length (the relevant portion)
            for i in 0..self.signal_length.min(xcorr.len()) {
                output[[t, i]] = xcorr[i] as f32;
            }
        }

        Ok(output)
    }

    /// Detect peaks in a cross-correlation array.
    ///
    /// A peak is any sample whose absolute value exceeds
    /// `threshold * mean(|correlations|)` (or the config SNR × noise_rms,
    /// whichever is larger).
    ///
    /// # Arguments
    ///
    /// * `correlations` – `[n_templates, signal_length]` from `correlate_all`
    /// * `threshold`    – Additional multiplicative threshold on the mean magnitude.
    pub fn detect_peaks(&self, correlations: &Array2<f32>, threshold: f32) -> Vec<Detection> {
        let mut detections = Vec::new();

        let n_t = correlations.nrows();
        let sig_len = correlations.ncols();

        for t in 0..n_t {
            let row = correlations.row(t);

            // Noise estimate: mean absolute value
            let mean_abs: f32 = row.iter().map(|v| v.abs()).sum::<f32>() / sig_len as f32;
            let rms: f32 = {
                let sum_sq: f32 = row.iter().map(|v| v * v).sum();
                (sum_sq / sig_len as f32).sqrt()
            };

            let abs_threshold = (threshold * mean_abs).max(self.config.peak_threshold_snr * rms);

            for i in 0..sig_len {
                let v = row[i];
                if v.abs() > abs_threshold {
                    // Simple local-maximum check (±1 neighbour)
                    let left = if i > 0 { row[i - 1].abs() } else { 0.0 };
                    let right = if i + 1 < sig_len {
                        row[i + 1].abs()
                    } else {
                        0.0
                    };
                    if v.abs() >= left && v.abs() >= right {
                        let snr = if rms > f32::EPSILON {
                            v.abs() / rms
                        } else {
                            0.0
                        };
                        detections.push(Detection {
                            template_idx: t,
                            sample_offset: i,
                            correlation_value: v,
                            snr,
                        });
                    }
                }
            }
        }

        detections
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{Array1, Array2};

    fn identity_template(len: usize) -> Array2<f32> {
        // Template = [1, 0, 0, …] (Dirac delta → self-correlation is flat)
        Array2::from_shape_fn((1, len), |(_, j)| if j == 0 { 1.0 } else { 0.0 })
    }

    fn rect_template(len: usize) -> Array2<f32> {
        Array2::from_elem((1, len), 1.0 / len as f32)
    }

    #[test]
    fn test_matched_filter_n_templates() {
        let templates = Array2::zeros((5, 32));
        let bank =
            MatchedFilterBank::new(&templates, 128, MatchedFilterConfig::default()).expect("bank");
        assert_eq!(bank.n_templates(), 5);
    }

    #[test]
    fn test_matched_filter_bank_output_shape() {
        let n_templates = 3;
        let template_len = 16;
        let signal_len = 256;

        let templates = Array2::from_shape_fn((n_templates, template_len), |(_, j)| j as f32);
        let bank = MatchedFilterBank::new(&templates, signal_len, MatchedFilterConfig::default())
            .expect("bank");

        let signal = Array1::from_vec((0..signal_len).map(|i| i as f32 * 0.01).collect());
        let corr = bank.correlate_all(&signal).expect("correlate");
        assert_eq!(corr.shape(), &[n_templates, signal_len]);
    }

    #[test]
    fn test_matched_filter_bank_self_correlation_peak_at_zero() {
        // A Dirac-delta template should yield peak at sample 0 when correlated
        // with a copy of itself embedded in the signal.
        let template_len = 32;
        let signal_len = 256;

        // Template = unit impulse
        let templates = identity_template(template_len);
        let config = MatchedFilterConfig {
            normalize_templates: false,
            ..Default::default()
        };
        let bank = MatchedFilterBank::new(&templates, signal_len, config).expect("bank");

        // Signal = unit impulse at sample 0
        let mut sig = vec![0.0f32; signal_len];
        sig[0] = 1.0;
        let signal = Array1::from_vec(sig);

        let corr = bank.correlate_all(&signal).expect("correlate");

        // Peak should be at sample 0
        let peak_idx = corr
            .row(0)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).expect("cmp"))
            .map(|(i, _)| i)
            .expect("peak");

        assert_eq!(peak_idx, 0, "Expected peak at sample 0, got {peak_idx}");
    }

    #[test]
    fn test_matched_filter_bank_detect_peaks() {
        let template_len = 8;
        let signal_len = 64;

        let templates = rect_template(template_len);
        let config = MatchedFilterConfig {
            normalize_templates: false,
            peak_threshold_snr: 2.0,
        };
        let bank = MatchedFilterBank::new(&templates, signal_len, config).expect("bank");

        // Insert a rectangular pulse at position 20
        let mut sig = vec![0.0f32; signal_len];
        for i in 20..(20 + template_len).min(signal_len) {
            sig[i] = 1.0;
        }
        let signal = Array1::from_vec(sig);

        let corr = bank.correlate_all(&signal).expect("correlate");
        let detections = bank.detect_peaks(&corr, 2.0);

        // Should find at least one detection near sample 20
        assert!(
            !detections.is_empty(),
            "Expected at least one detection, got 0"
        );
    }

    #[test]
    fn test_matched_filter_bank_wrong_signal_length_errors() {
        let templates = Array2::zeros((1, 8));
        let bank =
            MatchedFilterBank::new(&templates, 64, MatchedFilterConfig::default()).expect("bank");
        let wrong_signal = Array1::zeros(32);
        assert!(bank.correlate_all(&wrong_signal).is_err());
    }

    #[test]
    fn test_matched_filter_bank_zero_signal_near_zero_correlations() {
        let template_len = 16;
        let signal_len = 128;
        let templates = Array2::from_elem((2, template_len), 0.5f32);
        let bank = MatchedFilterBank::new(&templates, signal_len, MatchedFilterConfig::default())
            .expect("bank");
        let signal = Array1::zeros(signal_len);
        let corr = bank.correlate_all(&signal).expect("correlate");
        for &v in corr.iter() {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-5);
        }
    }
}
