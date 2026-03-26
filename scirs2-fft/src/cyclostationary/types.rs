//! Types for Cyclostationary Spectral Analysis.

use scirs2_core::numeric::Complex64;

/// Configuration for cyclostationary spectral analysis.
///
/// # Examples
///
/// ```
/// use scirs2_fft::cyclostationary::CyclostationaryConfig;
///
/// let config = CyclostationaryConfig::default();
/// assert_eq!(config.n_fft, 512);
/// assert!((config.overlap - 0.5).abs() < 1e-10);
/// assert!((config.alpha_resolution - 0.01).abs() < 1e-10);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CyclostationaryConfig {
    /// Cyclic frequencies (α) to analyse.
    ///
    /// If `None`, cyclic frequencies are auto-detected from the signal.
    pub cyclic_freqs: Option<Vec<f64>>,
    /// Resolution of the cyclic frequency axis for auto-detection (Hz).
    ///
    /// Smaller values provide finer cyclic-frequency resolution but increase
    /// computation time.  Default: 0.01.
    pub alpha_resolution: f64,
    /// FFT size for each segment (power of two recommended).  Default: 512.
    pub n_fft: usize,
    /// Fractional overlap between consecutive segments `[0, 1)`.  Default: 0.5.
    pub overlap: f64,
    /// Sampling frequency of the input signal (Hz).  Default: 1.0.
    pub fs: f64,
    /// Detection threshold (fraction of peak power) for cyclic frequency peaks.
    ///
    /// Only cyclic frequencies with power exceeding `threshold * max_power` are
    /// returned.  Default: 0.1.
    pub detection_threshold: f64,
}

impl Default for CyclostationaryConfig {
    fn default() -> Self {
        Self {
            cyclic_freqs: None,
            alpha_resolution: 0.01,
            n_fft: 512,
            overlap: 0.5,
            fs: 1.0,
            detection_threshold: 0.1,
        }
    }
}

/// Result of a Spectral Correlation Density (SCD) analysis.
///
/// The SCD matrix `scd[alpha_idx][freq_idx]` gives the complex spectral
/// correlation at cyclic frequency `cyclic_frequencies[alpha_idx]` and
/// spectral frequency `spectral_frequencies[freq_idx]`.
#[derive(Debug, Clone)]
pub struct SpectralCorrelationResult {
    /// Spectral Correlation Density matrix: `scd[α_idx][f_idx]`.
    pub scd: Vec<Vec<Complex64>>,
    /// Cyclic frequencies axis (Hz).
    pub cyclic_frequencies: Vec<f64>,
    /// Spectral frequency axis (Hz).
    pub spectral_frequencies: Vec<f64>,
}

impl SpectralCorrelationResult {
    /// Number of cyclic frequency bins.
    pub fn n_alphas(&self) -> usize {
        self.scd.len()
    }

    /// Number of spectral frequency bins.
    pub fn n_freqs(&self) -> usize {
        self.scd.first().map(|v| v.len()).unwrap_or(0)
    }

    /// Compute the cyclic power spectrum (α-profile): `|S_x(f=0; α)|` for each α.
    ///
    /// This collapses the SCD over the spectral frequency axis to produce a
    /// one-dimensional cyclic spectral signature useful for cyclic frequency detection.
    pub fn cyclic_power_spectrum(&self) -> Vec<f64> {
        self.scd
            .iter()
            .map(|row| {
                let sum_sq: f64 = row.iter().map(|c| c.re * c.re + c.im * c.im).sum();
                (sum_sq / row.len().max(1) as f64).sqrt()
            })
            .collect()
    }
}
