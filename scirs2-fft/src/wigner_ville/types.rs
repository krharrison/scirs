//! Types for the Wigner-Ville Distribution module.

/// Configuration for Wigner-Ville Distribution computation.
///
/// # Examples
///
/// ```
/// use scirs2_fft::wigner_ville::WvdConfig;
///
/// let config = WvdConfig::default();
/// assert!(config.analytic);
/// assert_eq!(config.smooth_window, 0);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct WvdConfig {
    /// Number of frequency bins in the output.
    ///
    /// Defaults to the signal length.  Values smaller than the signal length
    /// cause zero-padding to be avoided, resulting in a coarser frequency grid.
    pub n_freqs: Option<usize>,
    /// Gaussian smoothing window half-length for the Pseudo-WVD (PWVD).
    ///
    /// Set to `0` (the default) for the unsmoothed WVD.  Larger values reduce
    /// cross-terms at the expense of frequency resolution.
    pub smooth_window: usize,
    /// Whether to compute the analytic (Hilbert-extended) signal before the WVD.
    ///
    /// Setting this to `true` (the default) suppresses negative-frequency
    /// artefacts and cross-terms from the real-signal WVD.
    pub analytic: bool,
}

impl Default for WvdConfig {
    fn default() -> Self {
        Self {
            n_freqs: None,
            smooth_window: 0,
            analytic: true,
        }
    }
}

/// Result of a Wigner-Ville (or Pseudo-WVD) computation.
///
/// The `wvd` field is indexed as `wvd[time_index][freq_index]` and can contain
/// negative values (the WVD is a real-valued but not necessarily non-negative
/// time-frequency distribution).
#[derive(Debug, Clone)]
pub struct WvdResult {
    /// Time-frequency distribution: `wvd[t][f]`.
    ///
    /// Real-valued; can be negative (interference terms).
    pub wvd: Vec<Vec<f64>>,
    /// Time axis values in seconds (or sample indices if `fs = 1.0`).
    pub times: Vec<f64>,
    /// Frequency axis values in Hz (or normalised if `fs = 1.0`).
    pub frequencies: Vec<f64>,
}

impl WvdResult {
    /// Number of time frames in the distribution.
    pub fn n_times(&self) -> usize {
        self.wvd.len()
    }

    /// Number of frequency bins in the distribution.
    pub fn n_freqs(&self) -> usize {
        self.wvd.first().map(|v| v.len()).unwrap_or(0)
    }
}
