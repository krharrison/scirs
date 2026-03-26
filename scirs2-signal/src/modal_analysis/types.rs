//! Core types for Operational Modal Analysis (OMA).
//!
//! This module defines the fundamental data structures used across all OMA
//! sub-modules: method selector, configuration, modal mode representation,
//! analysis results, and the Modal Assurance Criterion matrix.

use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// OMA method selector
// ---------------------------------------------------------------------------

/// Selects which OMA algorithm to use.
///
/// All variants are data-free; configuration is supplied via [`OmaConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OmaMethod {
    /// Frequency Domain Decomposition — fast, peak-picking in SVD spectrum.
    Fdd,
    /// Enhanced FDD — fits SDOF power spectral density bell curves around each peak.
    Efdd,
    /// Covariance-driven Stochastic Subspace Identification.
    SsiCov,
    /// Data-driven Stochastic Subspace Identification.
    SsiData,
}

impl Default for OmaMethod {
    fn default() -> Self {
        OmaMethod::Fdd
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration shared by all OMA algorithms.
#[derive(Debug, Clone)]
pub struct OmaConfig {
    /// Expected number of structural modes to identify.
    pub n_modes: usize,
    /// Sampling frequency of the measured data (Hz).
    pub fs: f64,
    /// Number of correlation lags used in SSI covariance estimation and block-Hankel
    /// construction.  For FDD this is the number of Welch segments.
    pub n_lags: usize,
    /// Minimum frequency (Hz) for mode search (inclusive).  Modes below this limit
    /// are discarded.
    pub freq_min: f64,
    /// Maximum frequency (Hz) for mode search (exclusive).  Defaults to Nyquist.
    pub freq_max: Option<f64>,
    /// Stabilisation tolerance for frequency (relative) in SSI.
    pub stab_freq_tol: f64,
    /// Stabilisation tolerance for damping ratio (relative) in SSI.
    pub stab_damp_tol: f64,
    /// Welch segment overlap fraction (0 – 1) for FDD cross-PSD.
    pub overlap: f64,
}

impl Default for OmaConfig {
    fn default() -> Self {
        Self {
            n_modes: 4,
            fs: 100.0,
            n_lags: 64,
            freq_min: 0.1,
            freq_max: None,
            stab_freq_tol: 0.01,
            stab_damp_tol: 0.05,
            overlap: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Individual modal mode
// ---------------------------------------------------------------------------

/// Identified modal parameters for a single structural mode.
#[derive(Debug, Clone)]
pub struct ModalMode {
    /// Natural frequency (Hz).
    pub freq: f64,
    /// Damping ratio (dimensionless, e.g. 0.02 = 2 %).
    pub damping: f64,
    /// Complex mode shape vector (real part only for FDD; full complex for SSI).
    /// Length equals the number of measurement channels.
    pub mode_shape: Array1<f64>,
}

impl ModalMode {
    /// Create a new modal mode.
    pub fn new(freq: f64, damping: f64, mode_shape: Array1<f64>) -> Self {
        Self {
            freq,
            damping,
            mode_shape,
        }
    }

    /// Normalise the mode shape to unit Euclidean norm in-place.
    pub fn normalise(&mut self) {
        let norm: f64 = self.mode_shape.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > f64::EPSILON {
            self.mode_shape.mapv_inplace(|v| v / norm);
        }
    }
}

// ---------------------------------------------------------------------------
// OMA result
// ---------------------------------------------------------------------------

/// Output of an OMA algorithm: a list of identified modes plus metadata.
#[derive(Debug, Clone)]
pub struct OmaResult {
    /// Identified structural modes, sorted by ascending frequency.
    pub modes: Vec<ModalMode>,
    /// Algorithm used to produce this result.
    pub method: OmaMethod,
    /// Frequency resolution of the underlying spectral analysis (Hz).
    /// Zero for time-domain methods.
    pub freq_resolution: f64,
}

impl OmaResult {
    /// Create a new `OmaResult`.
    pub fn new(modes: Vec<ModalMode>, method: OmaMethod, freq_resolution: f64) -> Self {
        let mut result = Self {
            modes,
            method,
            freq_resolution,
        };
        result.modes.sort_by(|a, b| {
            a.freq
                .partial_cmp(&b.freq)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }

    /// Return the number of identified modes.
    #[inline]
    pub fn n_modes(&self) -> usize {
        self.modes.len()
    }
}

// ---------------------------------------------------------------------------
// MAC matrix
// ---------------------------------------------------------------------------

/// Modal Assurance Criterion matrix between two sets of mode shapes.
///
/// `matrix[i, j]` is the MAC value between mode `i` from one set and mode `j`
/// from another set.  Values are in [0, 1]; 1 indicates identical directions.
#[derive(Debug, Clone)]
pub struct MacMatrix {
    /// MAC value matrix (n_modes_a × n_modes_b).
    pub matrix: Array2<f64>,
}

impl MacMatrix {
    /// Construct from a pre-computed matrix.
    pub fn from_array(matrix: Array2<f64>) -> Self {
        Self { matrix }
    }

    /// Number of rows (modes in first set).
    pub fn n_rows(&self) -> usize {
        self.matrix.nrows()
    }

    /// Number of columns (modes in second set).
    pub fn n_cols(&self) -> usize {
        self.matrix.ncols()
    }

    /// Retrieve a single MAC value.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.matrix[[row, col]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_oma_method_default() {
        assert_eq!(OmaMethod::default(), OmaMethod::Fdd);
    }

    #[test]
    fn test_oma_config_default() {
        let cfg = OmaConfig::default();
        assert_eq!(cfg.n_modes, 4);
        assert!(cfg.fs > 0.0);
        assert!(cfg.n_lags > 0);
        assert!(cfg.overlap > 0.0 && cfg.overlap < 1.0);
    }

    #[test]
    fn test_modal_mode_normalise() {
        let shape = array![3.0, 4.0];
        let mut mode = ModalMode::new(5.0, 0.02, shape);
        mode.normalise();
        let norm: f64 = mode.mode_shape.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12, "norm={norm}");
    }

    #[test]
    fn test_oma_result_sorted() {
        let modes = vec![
            ModalMode::new(10.0, 0.02, array![1.0, 0.0]),
            ModalMode::new(2.0, 0.03, array![0.0, 1.0]),
            ModalMode::new(5.0, 0.01, array![1.0, 1.0]),
        ];
        let result = OmaResult::new(modes, OmaMethod::Fdd, 0.25);
        assert_eq!(result.modes[0].freq, 2.0);
        assert_eq!(result.modes[1].freq, 5.0);
        assert_eq!(result.modes[2].freq, 10.0);
    }

    #[test]
    fn test_mac_matrix_from_array() {
        let m = Array2::from_elem((3, 3), 0.5_f64);
        let mac = MacMatrix::from_array(m);
        assert_eq!(mac.n_rows(), 3);
        assert_eq!(mac.n_cols(), 3);
        assert!((mac.get(1, 2) - 0.5).abs() < 1e-12);
    }
}
