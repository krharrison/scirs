//! Type definitions for phase estimation algorithms.

use std::fmt;

/// Configuration for phase estimation algorithms.
#[derive(Debug, Clone)]
pub struct PhaseEstConfig {
    /// Number of signal components to extract.
    pub num_components: usize,
    /// Sample rate in Hz.
    pub fs: f64,
    /// Subspace dimension (must be >= num_components).
    /// If 0, defaults to 2 * num_components.
    pub subspace_dim: usize,
    /// Frequency resolution for MUSIC pseudospectrum (number of points).
    pub freq_resolution: usize,
    /// Maximum power-iteration steps for SVD/EVD.
    pub max_iter: usize,
    /// Convergence tolerance for iterative methods.
    pub tol: f64,
}

impl Default for PhaseEstConfig {
    fn default() -> Self {
        Self {
            num_components: 1,
            fs: 1.0,
            subspace_dim: 0,
            freq_resolution: 1024,
            max_iter: 200,
            tol: 1e-10,
        }
    }
}

/// A single recovered sinusoidal component.
#[derive(Debug, Clone, PartialEq)]
pub struct FrequencyComponent {
    /// Frequency in cycles per sample (normalised to [0, 0.5]).
    /// Multiply by `fs` to obtain Hz.
    pub frequency: f64,
    /// Amplitude (linear scale).
    pub amplitude: f64,
    /// Phase in radians in [-π, π].
    pub phase: f64,
}

impl fmt::Display for FrequencyComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FrequencyComponent {{ freq={:.4}, amp={:.4}, phase={:.4} }}",
            self.frequency, self.amplitude, self.phase
        )
    }
}

/// Instantaneous frequency track over time.
#[derive(Debug, Clone)]
pub struct InstantaneousFreq {
    /// Per-sample instantaneous frequency values (same units as requested `fs`).
    pub samples: Vec<f64>,
    /// Sample rate used for computing the IF.
    pub fs: f64,
}

impl InstantaneousFreq {
    /// Construct from raw samples and sample rate.
    pub fn new(samples: Vec<f64>, fs: f64) -> Self {
        Self { samples, fs }
    }

    /// Mean instantaneous frequency.
    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().copied().sum::<f64>() / self.samples.len() as f64
    }

    /// Standard deviation of instantaneous frequency.
    pub fn std_dev(&self) -> f64 {
        let n = self.samples.len();
        if n < 2 {
            return 0.0;
        }
        let m = self.mean();
        let var = self.samples.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64;
        var.sqrt()
    }
}

/// Result returned by ESPRIT and MUSIC estimators.
#[derive(Debug, Clone)]
pub struct PhaseEstResult {
    /// Extracted frequency components, sorted by frequency.
    pub components: Vec<FrequencyComponent>,
    /// Which algorithm produced this result.
    pub method: PhaseMethod,
}

impl PhaseEstResult {
    /// Create a new result.
    pub fn new(mut components: Vec<FrequencyComponent>, method: PhaseMethod) -> Self {
        components.sort_by(|a, b| {
            a.frequency
                .partial_cmp(&b.frequency)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { components, method }
    }

    /// Number of detected components.
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// True if no components were found.
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }
}

/// Phase estimation method identifier.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhaseMethod {
    /// Estimation of Signal Parameters via Rotational Invariance Techniques.
    Esprit,
    /// Multiple Signal Classification.
    Music,
    /// Hilbert transform (analytic signal) approach.
    HilbertTransform,
    /// Autocorrelation-based approach.
    AutoCorrelation,
}

impl fmt::Display for PhaseMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhaseMethod::Esprit => write!(f, "ESPRIT"),
            PhaseMethod::Music => write!(f, "MUSIC"),
            PhaseMethod::HilbertTransform => write!(f, "HilbertTransform"),
            PhaseMethod::AutoCorrelation => write!(f, "AutoCorrelation"),
            _ => write!(f, "Unknown"),
        }
    }
}
