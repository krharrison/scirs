//! Type definitions for enhanced spectral analysis methods
//!
//! This module defines all configuration, result, and parameter types used
//! across the multitaper, Lomb-Scargle, and parametric spectral estimation methods.

use scirs2_core::ndarray::{Array1, Array2};

// =============================================================================
// Multitaper types
// =============================================================================

/// Configuration for multitaper spectral estimation (Thomson's method)
#[derive(Debug, Clone)]
pub struct MultitaperConfig {
    /// Time-bandwidth product (NW). Typical values: 2.0 to 8.0.
    /// Higher values give better spectral leakage suppression but lower resolution.
    pub nw: f64,
    /// Number of DPSS tapers (K). Should satisfy K <= 2*NW.
    /// If None, defaults to 2*NW - 1.
    pub k: Option<usize>,
    /// Sampling frequency in Hz. Default: 1.0.
    pub fs: f64,
    /// Number of FFT points. If None, defaults to next power of 2 >= signal length.
    pub nfft: Option<usize>,
    /// Use adaptive weighting (Thomson's adaptive algorithm).
    /// When true, weights are computed to minimize broadband bias.
    pub adaptive: bool,
    /// Maximum number of iterations for adaptive weighting. Default: 150.
    pub max_adaptive_iter: usize,
    /// Convergence tolerance for adaptive weighting. Default: 1e-7.
    pub adaptive_tol: f64,
}

impl Default for MultitaperConfig {
    fn default() -> Self {
        Self {
            nw: 4.0,
            k: None,
            fs: 1.0,
            nfft: None,
            adaptive: true,
            max_adaptive_iter: 150,
            adaptive_tol: 1e-7,
        }
    }
}

/// Result from multitaper spectral estimation
#[derive(Debug, Clone)]
pub struct MultitaperResult {
    /// Frequency vector (Hz)
    pub frequencies: Array1<f64>,
    /// Power spectral density estimate
    pub psd: Array1<f64>,
    /// Adaptive weights for each taper at each frequency (K x nfreq)
    pub weights: Option<Array2<f64>>,
    /// Individual eigenspectra for each taper (K x nfreq)
    pub eigenspectra: Option<Array2<f64>>,
    /// DPSS concentration ratios (eigenvalues)
    pub concentration_ratios: Array1<f64>,
}

/// Result from multitaper F-test for line components
#[derive(Debug, Clone)]
pub struct FTestResult {
    /// Frequency vector (Hz)
    pub frequencies: Array1<f64>,
    /// F-statistic at each frequency
    pub f_statistic: Array1<f64>,
    /// p-values for each frequency
    pub p_values: Array1<f64>,
    /// Estimated amplitude of line component at each frequency
    pub line_amplitudes: Array1<f64>,
    /// Indices of detected line components (where p < significance_level)
    pub significant_indices: Vec<usize>,
}

// =============================================================================
// Lomb-Scargle types
// =============================================================================

/// Configuration for Lomb-Scargle periodogram
#[derive(Debug, Clone)]
pub struct LombScargleConfig {
    /// Normalization method
    pub normalization: LombScargleNormalization,
    /// Whether to fit a floating mean (generalized Lomb-Scargle)
    pub fit_mean: bool,
    /// Whether to center data before analysis
    pub center_data: bool,
    /// Oversampling factor for automatic frequency grid (default: 5)
    pub oversampling: f64,
    /// Nyquist factor for maximum frequency (default: 1.0)
    pub nyquist_factor: f64,
    /// Number of frequency points. If None, determined automatically.
    pub n_frequencies: Option<usize>,
    /// Minimum frequency. If None, determined from data span.
    pub f_min: Option<f64>,
    /// Maximum frequency. If None, determined from average Nyquist.
    pub f_max: Option<f64>,
}

impl Default for LombScargleConfig {
    fn default() -> Self {
        Self {
            normalization: LombScargleNormalization::Standard,
            fit_mean: true,
            center_data: true,
            oversampling: 5.0,
            nyquist_factor: 1.0,
            n_frequencies: None,
            f_min: None,
            f_max: None,
        }
    }
}

/// Normalization methods for Lomb-Scargle periodogram
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LombScargleNormalization {
    /// Standard normalization: P(f) in [0, 1]
    Standard,
    /// Model normalization: chi-squared residual model
    Model,
    /// Log normalization: natural log of model normalization
    Log,
    /// PSD normalization: power spectral density units
    Psd,
}

/// Result from Lomb-Scargle periodogram
#[derive(Debug, Clone)]
pub struct LombScargleResult {
    /// Angular frequencies evaluated
    pub frequencies: Array1<f64>,
    /// Periodogram power at each frequency
    pub power: Array1<f64>,
    /// Normalization used
    pub normalization: LombScargleNormalization,
    /// Whether floating mean correction was applied
    pub fit_mean: bool,
}

/// Result from false alarm probability estimation
#[derive(Debug, Clone)]
pub struct FalseAlarmResult {
    /// False alarm probability for each power level
    pub fap: Array1<f64>,
    /// The power levels queried
    pub power_levels: Array1<f64>,
    /// Method used for FAP calculation
    pub method: FapMethod,
}

/// Method for false alarm probability calculation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FapMethod {
    /// Baluev (2008) analytical approximation
    Baluev,
    /// Davies (1977) bound
    Davies,
    /// Naive (single-frequency Exponential)
    Naive,
}

// =============================================================================
// Parametric spectral types
// =============================================================================

/// Configuration for Burg's method (maximum entropy spectral estimation)
#[derive(Debug, Clone)]
pub struct BurgConfig {
    /// AR model order
    pub order: usize,
    /// Sampling frequency (Hz). Default: 1.0
    pub fs: f64,
    /// Number of frequency evaluation points. Default: 512
    pub nfft: usize,
}

impl Default for BurgConfig {
    fn default() -> Self {
        Self {
            order: 10,
            fs: 1.0,
            nfft: 512,
        }
    }
}

/// Result from Burg's method
#[derive(Debug, Clone)]
pub struct BurgResult {
    /// AR coefficients [1, a1, a2, ..., ap]
    pub ar_coeffs: Array1<f64>,
    /// Reflection coefficients
    pub reflection_coeffs: Array1<f64>,
    /// Estimated prediction error variance (noise power)
    pub variance: f64,
    /// Frequency vector (Hz)
    pub frequencies: Array1<f64>,
    /// Power spectral density
    pub psd: Array1<f64>,
}

/// Configuration for Yule-Walker AR estimation
#[derive(Debug, Clone)]
pub struct YuleWalkerConfig {
    /// AR model order
    pub order: usize,
    /// Sampling frequency (Hz). Default: 1.0
    pub fs: f64,
    /// Number of frequency evaluation points. Default: 512
    pub nfft: usize,
}

impl Default for YuleWalkerConfig {
    fn default() -> Self {
        Self {
            order: 10,
            fs: 1.0,
            nfft: 512,
        }
    }
}

/// Result from Yule-Walker AR estimation
#[derive(Debug, Clone)]
pub struct YuleWalkerResult {
    /// AR coefficients [1, a1, a2, ..., ap]
    pub ar_coeffs: Array1<f64>,
    /// Reflection coefficients from Levinson-Durbin
    pub reflection_coeffs: Array1<f64>,
    /// Estimated noise variance
    pub variance: f64,
    /// Frequency vector (Hz)
    pub frequencies: Array1<f64>,
    /// Power spectral density
    pub psd: Array1<f64>,
}

/// Configuration for MUSIC (Multiple Signal Classification)
#[derive(Debug, Clone)]
pub struct MusicConfig {
    /// Number of signal sources to estimate
    pub n_signals: usize,
    /// Subspace dimension (covariance matrix size). If None, uses signal_length / 3
    pub subspace_dim: Option<usize>,
    /// Number of frequency points to evaluate. Default: 1024
    pub n_frequencies: usize,
    /// Sampling frequency (Hz). Default: 1.0
    pub fs: f64,
    /// Use forward-backward averaging for improved covariance estimate
    pub forward_backward: bool,
    /// Frequency range as (min, max) in Hz. If None, uses (0, fs/2)
    pub frequency_range: Option<(f64, f64)>,
}

impl Default for MusicConfig {
    fn default() -> Self {
        Self {
            n_signals: 2,
            subspace_dim: None,
            n_frequencies: 1024,
            fs: 1.0,
            forward_backward: true,
            frequency_range: None,
        }
    }
}

/// Result from MUSIC spectral estimation
#[derive(Debug, Clone)]
pub struct MusicResult {
    /// Frequency vector (Hz)
    pub frequencies: Array1<f64>,
    /// Pseudospectrum values (dB)
    pub pseudospectrum: Array1<f64>,
    /// Estimated signal frequencies (Hz)
    pub signal_frequencies: Array1<f64>,
    /// Eigenvalues of the covariance matrix (sorted descending)
    pub eigenvalues: Array1<f64>,
    /// Number of detected signals
    pub n_signals: usize,
    /// Subspace dimension used
    pub subspace_dim: usize,
}

/// Configuration for ESPRIT (Estimation of Signal Parameters via Rotational Invariance)
#[derive(Debug, Clone)]
pub struct EspritConfig {
    /// Number of signal sources to estimate
    pub n_signals: usize,
    /// Subspace dimension (covariance matrix size). If None, uses signal_length / 3
    pub subspace_dim: Option<usize>,
    /// Sampling frequency (Hz). Default: 1.0
    pub fs: f64,
    /// Use forward-backward averaging for improved covariance estimate
    pub forward_backward: bool,
    /// Use total least squares (TLS-ESPRIT) instead of standard LS-ESPRIT
    pub total_least_squares: bool,
}

impl Default for EspritConfig {
    fn default() -> Self {
        Self {
            n_signals: 2,
            subspace_dim: None,
            fs: 1.0,
            forward_backward: true,
            total_least_squares: false,
        }
    }
}

/// Result from ESPRIT frequency estimation
#[derive(Debug, Clone)]
pub struct EspritResult {
    /// Estimated signal frequencies (Hz), sorted ascending
    pub frequencies: Array1<f64>,
    /// Estimated signal amplitudes (complex, encoding amplitude and phase)
    pub amplitudes: Option<Array1<f64>>,
    /// Eigenvalues of the rotation matrix (on unit circle for clean signals)
    pub rotation_eigenvalues: Array1<f64>,
    /// Eigenvalues of the covariance matrix (sorted descending)
    pub covariance_eigenvalues: Array1<f64>,
    /// Subspace dimension used
    pub subspace_dim: usize,
}
