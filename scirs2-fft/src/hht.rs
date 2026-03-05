//! Hilbert-Huang Transform (HHT) — dedicated public API module
//!
//! This module exposes a clean, struct-based API for the Hilbert-Huang
//! Transform (HHT), Empirical Mode Decomposition (EMD), and Ensemble EMD
//! (EEMD).  The core mathematical routines are implemented here; they
//! supplement (and do not replace) the functions in [`crate::hilbert_enhanced`].
//!
//! # Algorithm summary
//!
//! 1. **EMD** — The signal is decomposed into Intrinsic Mode Functions (IMFs)
//!    via the *sifting process*: the mean envelope (cubic spline through
//!    local maxima and minima) is repeatedly subtracted until the candidate
//!    IMF satisfies the two IMF conditions (Huang et al., 1998).
//! 2. **Hilbert spectral analysis** — Each IMF is passed through the analytic
//!    signal (FFT-based Hilbert transform) to obtain instantaneous frequency
//!    and amplitude, yielding a 2D time-frequency energy density.
//! 3. **EEMD** — White noise is added to the signal, EMD is performed many
//!    times, and the results are averaged to suppress mode mixing.
//!
//! # References
//!
//! * Huang, N. E. *et al.* (1998). "The empirical mode decomposition and the
//!   Hilbert spectrum for nonlinear and non-stationary time series analysis."
//!   *Proc. R. Soc. Lond. A* **454**, 903–995.
//! * Wu, Z. & Huang, N. E. (2009). "Ensemble empirical mode decomposition: a
//!   noise-assisted data analysis method." *Adv. Adapt. Data Anal.* **1**,
//!   1–41.

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
//  Stopping criterion types
// ─────────────────────────────────────────────────────────────────────────────

/// Stopping criterion for the sifting process.
///
/// The sifting process extracts one IMF by iteratively subtracting the mean
/// envelope.  Different stopping criteria have been proposed in the literature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StoppingCriterion {
    /// Cauchy convergence: sifting stops when the normalised squared difference
    /// between consecutive candidates falls below `threshold`.
    /// Typical value: 0.05 (Huang et al., 1998).
    Cauchy {
        /// Convergence threshold ε.  Range (0, 1); typical default 0.05.
        threshold: f64,
        /// *S-number* — the criterion must be satisfied for `s_number`
        /// consecutive iterations before stopping.
        s_number: usize,
    },
    /// Fixed number of sifting iterations per IMF.
    FixedIterations {
        /// Number of sifting iterations to perform.
        n_iterations: usize,
    },
    /// Energy ratio: stop when the residual energy drops below a fraction of
    /// the original signal energy.
    EnergyRatio {
        /// Fraction of signal energy below which sifting stops.
        ratio: f64,
    },
}

impl Default for StoppingCriterion {
    fn default() -> Self {
        Self::Cauchy {
            threshold: 0.05,
            s_number: 4,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  EmpiricalModeDecomposition struct
// ─────────────────────────────────────────────────────────────────────────────

/// Builder-style configuration struct for Empirical Mode Decomposition.
///
/// ```
/// use scirs2_fft::hht::{EmpiricalModeDecomposition, StoppingCriterion};
///
/// let emd = EmpiricalModeDecomposition::new()
///     .max_imfs(10)
///     .max_sifts(200)
///     .stopping_criterion(StoppingCriterion::Cauchy { threshold: 0.05, s_number: 4 });
/// ```
#[derive(Debug, Clone)]
pub struct EmpiricalModeDecomposition {
    /// Maximum number of IMFs to extract.
    pub max_imfs: usize,
    /// Maximum sifting iterations per IMF.
    pub max_sifts: usize,
    /// Stopping criterion for the sifting loop.
    pub stopping_criterion: StoppingCriterion,
}

impl Default for EmpiricalModeDecomposition {
    fn default() -> Self {
        Self {
            max_imfs: 20,
            max_sifts: 500,
            stopping_criterion: StoppingCriterion::default(),
        }
    }
}

impl EmpiricalModeDecomposition {
    /// Create a new `EmpiricalModeDecomposition` with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of IMFs to extract.
    pub fn max_imfs(mut self, n: usize) -> Self {
        self.max_imfs = n;
        self
    }

    /// Set the maximum number of sifting iterations per IMF.
    pub fn max_sifts(mut self, n: usize) -> Self {
        self.max_sifts = n;
        self
    }

    /// Set the stopping criterion for the sifting process.
    pub fn stopping_criterion(mut self, criterion: StoppingCriterion) -> Self {
        self.stopping_criterion = criterion;
        self
    }

    /// Decompose `signal` into IMFs using the configured parameters.
    ///
    /// This is a convenience method that delegates to the free function [`emd`].
    ///
    /// # Errors
    ///
    /// Returns an error if the signal is too short (< 4 samples).
    pub fn decompose(&self, signal: &[f64]) -> FFTResult<Vec<Vec<f64>>> {
        emd(signal, self.max_imfs, self.stopping_criterion)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Free-function API
// ─────────────────────────────────────────────────────────────────────────────

/// Decompose a signal into Intrinsic Mode Functions (IMFs) via EMD.
///
/// The signal is iteratively decomposed: at each step the *sifting process*
/// extracts one IMF (the highest-frequency component) and the residual is
/// passed back for the next iteration.
///
/// # Arguments
///
/// * `signal`   - Input signal slice.
/// * `max_imfs` - Maximum number of IMFs to extract.  Set to a large value
///                (e.g. `usize::MAX`) to extract until the residual is monotonic.
/// * `stopping_criterion` - Stopping criterion for the sifting process.
///
/// # Returns
///
/// A `Vec<Vec<f64>>` where each inner `Vec` is one IMF, ordered from the
/// highest-frequency component to the lowest.  The *residual* is appended as
/// the last element.
///
/// # Errors
///
/// Returns an error if `signal.len() < 4`.
///
/// # Examples
///
/// ```
/// use scirs2_fft::hht::{emd, StoppingCriterion};
/// use std::f64::consts::PI;
///
/// let n = 256;
/// let signal: Vec<f64> = (0..n).map(|i| {
///     let t = i as f64 / 256.0;
///     (2.0 * PI * 8.0 * t).sin() + 0.5 * (2.0 * PI * 30.0 * t).sin()
/// }).collect();
///
/// let imfs = emd(&signal, 10, StoppingCriterion::default()).expect("EMD failed");
/// assert!(!imfs.is_empty(), "Should produce at least one IMF + residual");
///
/// // The sum of all components should reconstruct the original
/// let mut reconstructed = vec![0.0_f64; n];
/// for imf in &imfs {
///     for (i, &v) in imf.iter().enumerate() {
///         reconstructed[i] += v;
///     }
/// }
/// for i in 0..n {
///     assert!((reconstructed[i] - signal[i]).abs() < 1e-8,
///         "Reconstruction failed at index {i}");
/// }
/// ```
pub fn emd(
    signal: &[f64],
    max_imfs: usize,
    stopping_criterion: StoppingCriterion,
) -> FFTResult<Vec<Vec<f64>>> {
    let n = signal.len();
    if n < 4 {
        return Err(FFTError::ValueError(
            "Signal must have at least 4 samples for EMD".to_string(),
        ));
    }

    let max_sifts = match stopping_criterion {
        StoppingCriterion::FixedIterations { n_iterations } => n_iterations,
        _ => 500,
    };

    let mut components: Vec<Vec<f64>> = Vec::new();
    let mut residual = signal.to_vec();

    for _imf_idx in 0..max_imfs {
        // Check if residual has enough extrema to continue
        let n_extrema = count_extrema(&residual);
        if n_extrema < 2 {
            break;
        }

        let imf = sifting_process(&residual, max_sifts, stopping_criterion)?;

        // Subtract IMF from residual
        for i in 0..n {
            residual[i] -= imf[i];
        }

        components.push(imf);

        // Early exit: residual energy negligible
        let residual_energy: f64 = residual.iter().map(|&v| v * v).sum();
        let signal_energy: f64 = signal.iter().map(|&v| v * v).sum();
        if signal_energy > 0.0 && residual_energy / signal_energy < 1e-12 {
            break;
        }

        if count_extrema(&residual) < 2 {
            break;
        }
    }

    // Append the residual as the final component
    components.push(residual);
    Ok(components)
}

/// Extract one IMF from `signal` using the sifting process.
///
/// The sifting process repeatedly subtracts the mean envelope (arithmetic mean
/// of the upper and lower cubic-spline envelopes through local extrema) until a
/// stopping criterion is met.
///
/// # Arguments
///
/// * `signal`   - Input signal slice (length ≥ 4).
/// * `max_sifts` - Hard cap on the number of sifting iterations.
/// * `stopping_criterion` - When to stop sifting.
///
/// # Returns
///
/// A `Vec<f64>` of the same length as `signal` containing one IMF.
///
/// # Errors
///
/// Returns an error if the signal is too short.
///
/// # Examples
///
/// ```
/// use scirs2_fft::hht::{sifting_process, StoppingCriterion};
/// use std::f64::consts::PI;
///
/// let n = 128;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 10.0 * i as f64 / n as f64).sin())
///     .collect();
///
/// let imf = sifting_process(&signal, 100, StoppingCriterion::default())
///     .expect("sifting failed");
/// assert_eq!(imf.len(), signal.len());
/// ```
pub fn sifting_process(
    signal: &[f64],
    max_sifts: usize,
    stopping_criterion: StoppingCriterion,
) -> FFTResult<Vec<f64>> {
    let n = signal.len();
    if n < 4 {
        return Err(FFTError::ValueError(
            "Signal must have at least 4 samples for sifting".to_string(),
        ));
    }

    let mut h = signal.to_vec();
    let mut prev_h = h.clone();
    let mut s_count = 0usize;

    for _iter in 0..max_sifts {
        let (max_idx, max_val) = find_local_maxima_full(&h);
        let (min_idx, min_val) = find_local_minima_full(&h);

        if max_idx.len() < 2 || min_idx.len() < 2 {
            break;
        }

        let upper_env = cubic_spline_envelope(&max_idx, &max_val, n)?;
        let lower_env = cubic_spline_envelope(&min_idx, &min_val, n)?;

        // Subtract mean envelope
        for i in 0..n {
            h[i] -= (upper_env[i] + lower_env[i]) / 2.0;
        }

        // Check stopping criterion
        match stopping_criterion {
            StoppingCriterion::Cauchy { threshold, s_number } => {
                let diff_energy: f64 = h
                    .iter()
                    .zip(prev_h.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                let h_energy: f64 = prev_h.iter().map(|&v| v * v).sum();

                if h_energy > 0.0 && diff_energy / h_energy < threshold {
                    s_count += 1;
                    if s_count >= s_number {
                        return Ok(h);
                    }
                } else {
                    s_count = 0;
                }
            }
            StoppingCriterion::FixedIterations { .. } => {
                // Fixed iterations: just keep going until max_sifts
            }
            StoppingCriterion::EnergyRatio { ratio } => {
                let orig_energy: f64 = signal.iter().map(|&v| v * v).sum();
                let h_energy: f64 = h.iter().map(|&v| v * v).sum();
                if orig_energy > 0.0 && (h_energy / orig_energy - 1.0).abs() < ratio {
                    return Ok(h);
                }
            }
        }

        prev_h.clone_from(&h);
    }

    Ok(h)
}

/// Find indices and values of local maxima, with mirrored boundary conditions.
///
/// Boundary extrema are added by reflecting the first/last internal extremum,
/// which reduces boundary effects on the spline interpolation.
///
/// # Arguments
///
/// * `signal` - Input signal.
///
/// # Returns
///
/// `(max_indices, max_values)` where each index is a `usize` sample index.
///
/// # Examples
///
/// ```
/// use scirs2_fft::hht::find_extrema;
///
/// let signal = vec![0.0_f64, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0];
/// let (maxima, minima) = find_extrema(&signal);
/// assert!(!maxima.0.is_empty());
/// assert!(!minima.0.is_empty());
/// ```
pub fn find_extrema(signal: &[f64]) -> ((Vec<usize>, Vec<f64>), (Vec<usize>, Vec<f64>)) {
    let maxima = find_local_maxima_full(signal);
    let minima = find_local_minima_full(signal);
    (maxima, minima)
}

/// Internal: find local maxima (positions as `usize`) with boundary padding.
fn find_local_maxima_full(signal: &[f64]) -> (Vec<usize>, Vec<f64>) {
    let n = signal.len();
    let mut idx = Vec::new();
    let mut val = Vec::new();

    for i in 1..n.saturating_sub(1) {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            idx.push(i);
            val.push(signal[i]);
        }
    }

    // Boundary: add mirrored boundary points so the spline covers the full signal
    if idx.is_empty() {
        // No interior maxima — use endpoints
        let (start_val, end_val) = if signal[0] >= signal[n - 1] {
            (signal[0], signal[n - 1])
        } else {
            (signal[n - 1], signal[0])
        };
        return (vec![0, n - 1], vec![start_val, end_val]);
    }

    let first_idx = idx[0];
    let first_val = val[0];
    let last_idx = *idx.last().unwrap_or(&0);
    let last_val = *val.last().unwrap_or(&0.0);

    // Mirror the first/last interior maximum across the boundary
    let left_mirror_idx = if first_idx > 0 { first_idx - 1 } else { 0 };
    let right_mirror_idx = if last_idx + 1 < n { last_idx + 1 } else { n - 1 };

    let mut all_idx = vec![left_mirror_idx];
    let mut all_val = vec![first_val]; // Use the same value (reflection)
    all_idx.extend_from_slice(&idx);
    all_val.extend_from_slice(&val);
    all_idx.push(right_mirror_idx);
    all_val.push(last_val);

    (all_idx, all_val)
}

/// Internal: find local minima (positions as `usize`) with boundary padding.
fn find_local_minima_full(signal: &[f64]) -> (Vec<usize>, Vec<f64>) {
    let n = signal.len();
    let mut idx = Vec::new();
    let mut val = Vec::new();

    for i in 1..n.saturating_sub(1) {
        if signal[i] < signal[i - 1] && signal[i] < signal[i + 1] {
            idx.push(i);
            val.push(signal[i]);
        }
    }

    if idx.is_empty() {
        let (start_val, end_val) = if signal[0] <= signal[n - 1] {
            (signal[0], signal[n - 1])
        } else {
            (signal[n - 1], signal[0])
        };
        return (vec![0, n - 1], vec![start_val, end_val]);
    }

    let first_idx = idx[0];
    let first_val = val[0];
    let last_idx = *idx.last().unwrap_or(&0);
    let last_val = *val.last().unwrap_or(&0.0);

    let left_mirror_idx = if first_idx > 0 { first_idx - 1 } else { 0 };
    let right_mirror_idx = if last_idx + 1 < n { last_idx + 1 } else { n - 1 };

    let mut all_idx = vec![left_mirror_idx];
    let mut all_val = vec![first_val];
    all_idx.extend_from_slice(&idx);
    all_val.extend_from_slice(&val);
    all_idx.push(right_mirror_idx);
    all_val.push(last_val);

    (all_idx, all_val)
}

/// Count the number of local extrema (maxima + minima) in a signal.
fn count_extrema(signal: &[f64]) -> usize {
    if signal.len() < 3 {
        return 0;
    }
    let mut count = 0usize;
    for i in 1..signal.len() - 1 {
        if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1])
            || (signal[i] < signal[i - 1] && signal[i] < signal[i + 1])
        {
            count += 1;
        }
    }
    count
}

/// Interpolate a cubic spline envelope through extrema and evaluate at all sample points.
///
/// The spline uses natural boundary conditions (`c_0 = c_{m-1} = 0`).
///
/// # Arguments
///
/// * `extrema_indices` - Sample indices of the extrema (monotonically increasing).
/// * `extrema_values`  - Signal values at those extrema.
/// * `n`               - Length of the output vector (number of samples).
///
/// # Returns
///
/// A `Vec<f64>` of length `n` containing the spline-interpolated envelope.
///
/// # Errors
///
/// Returns an error if fewer than 2 knots are provided.
///
/// # Examples
///
/// ```
/// use scirs2_fft::hht::cubic_spline_envelope;
///
/// let indices = vec![0, 3, 7];
/// let values  = vec![1.0, 2.0, 0.5];
/// let env = cubic_spline_envelope(&indices, &values, 8).expect("spline failed");
/// assert_eq!(env.len(), 8);
/// // Knot values are reproduced
/// assert!((env[0] - 1.0).abs() < 1e-9);
/// assert!((env[3] - 2.0).abs() < 1e-9);
/// assert!((env[7] - 0.5).abs() < 1e-9);
/// ```
pub fn cubic_spline_envelope(
    extrema_indices: &[usize],
    extrema_values: &[f64],
    n: usize,
) -> FFTResult<Vec<f64>> {
    let m = extrema_indices.len();
    if m < 2 {
        return Err(FFTError::ValueError(
            "Need at least 2 extrema for spline interpolation".to_string(),
        ));
    }
    if m != extrema_values.len() {
        return Err(FFTError::ValueError(
            "extrema_indices and extrema_values must have the same length".to_string(),
        ));
    }

    // Convert usize indices to f64 knot positions
    let x_knots: Vec<f64> = extrema_indices.iter().map(|&i| i as f64).collect();
    let y_knots = extrema_values;

    // Fall back to linear for 2 knots
    if m == 2 {
        return linear_interp(&x_knots, y_knots, n);
    }

    // Compute interval widths h[i] = x[i+1] - x[i]
    let mut h = Vec::with_capacity(m - 1);
    for i in 0..m - 1 {
        let hi = x_knots[i + 1] - x_knots[i];
        if hi <= 0.0 {
            return linear_interp(&x_knots, y_knots, n);
        }
        h.push(hi);
    }

    // Set up natural spline system for interior knots c[1..m-2]
    let n_interior = m - 2;
    if n_interior == 0 {
        return linear_interp(&x_knots, y_knots, n);
    }

    let mut diag = vec![0.0f64; n_interior];
    let mut upper = vec![0.0f64; n_interior.saturating_sub(1)];
    let mut lower = vec![0.0f64; n_interior.saturating_sub(1)];
    let mut rhs = vec![0.0f64; n_interior];

    for i in 0..n_interior {
        diag[i] = 2.0 * (h[i] + h[i + 1]);
        rhs[i] = 3.0
            * ((y_knots[i + 2] - y_knots[i + 1]) / h[i + 1]
                - (y_knots[i + 1] - y_knots[i]) / h[i]);
    }
    for i in 0..n_interior.saturating_sub(1) {
        upper[i] = h[i + 1];
        lower[i] = h[i + 1];
    }

    let c_interior = solve_tridiagonal(&lower, &diag, &upper, &rhs)?;

    // Full second-derivative array (natural BC: c[0] = c[m-1] = 0)
    let mut c = vec![0.0f64; m];
    for i in 0..n_interior {
        c[i + 1] = c_interior[i];
    }

    // Compute b and d spline coefficients
    let mut b = vec![0.0f64; m - 1];
    let mut d = vec![0.0f64; m - 1];
    for i in 0..m - 1 {
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
        b[i] = (y_knots[i + 1] - y_knots[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0;
    }

    // Evaluate the spline at each integer sample index 0..n
    let mut result = Vec::with_capacity(n);
    for t in 0..n {
        let tf = t as f64;
        let seg = find_segment(&x_knots, tf);
        let dx = tf - x_knots[seg];
        let val = y_knots[seg]
            + b[seg] * dx
            + c[seg] * dx * dx
            + d[seg] * dx * dx * dx;
        result.push(val);
    }

    Ok(result)
}

/// Solve the tridiagonal system using the Thomas algorithm.
fn solve_tridiagonal(
    lower: &[f64],
    diag: &[f64],
    upper: &[f64],
    rhs: &[f64],
) -> FFTResult<Vec<f64>> {
    let n = diag.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        if diag[0].abs() < 1e-15 {
            return Err(FFTError::ComputationError(
                "Singular tridiagonal system".to_string(),
            ));
        }
        return Ok(vec![rhs[0] / diag[0]]);
    }

    let mut c_prime = vec![0.0f64; n];
    let mut d_prime = vec![0.0f64; n];

    if diag[0].abs() < 1e-15 {
        return Err(FFTError::ComputationError(
            "Zero pivot in tridiagonal solve".to_string(),
        ));
    }
    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n {
        let l_val = if i <= lower.len() { lower[i - 1] } else { 0.0 };
        let denom = diag[i] - l_val * c_prime[i - 1];
        if denom.abs() < 1e-15 {
            return Err(FFTError::ComputationError(
                "Near-singular tridiagonal system".to_string(),
            ));
        }
        c_prime[i] = if i < n - 1 && i < upper.len() {
            upper[i] / denom
        } else {
            0.0
        };
        d_prime[i] = (rhs[i] - l_val * d_prime[i - 1]) / denom;
    }

    let mut x = vec![0.0f64; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    Ok(x)
}

/// Find the spline segment index for a given query point.
fn find_segment(x_knots: &[f64], t: f64) -> usize {
    if t <= x_knots[0] {
        return 0;
    }
    for i in 0..x_knots.len() - 1 {
        if t >= x_knots[i] && t < x_knots[i + 1] {
            return i;
        }
    }
    x_knots.len().saturating_sub(2)
}

/// Linear interpolation fallback.
fn linear_interp(x_knots: &[f64], y_knots: &[f64], n_out: usize) -> FFTResult<Vec<f64>> {
    let m = x_knots.len();
    let mut result = Vec::with_capacity(n_out);
    for t in 0..n_out {
        let tf = t as f64;
        if tf <= x_knots[0] {
            result.push(y_knots[0]);
        } else if tf >= x_knots[m - 1] {
            result.push(y_knots[m - 1]);
        } else {
            let seg = find_segment(x_knots, tf);
            let dx = x_knots[seg + 1] - x_knots[seg];
            let frac = if dx > 0.0 {
                (tf - x_knots[seg]) / dx
            } else {
                0.0
            };
            result.push(y_knots[seg] + frac * (y_knots[seg + 1] - y_knots[seg]));
        }
    }
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Instantaneous frequency via Hilbert transform
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the instantaneous frequency of an IMF via the analytic signal.
///
/// The analytic signal z(t) = x(t) + j·H{x}(t) is obtained via FFT-based
/// Hilbert transform.  The instantaneous frequency is then:
///
/// ```text
/// f_inst(t) = (1 / 2π) · dφ/dt
/// ```
///
/// where φ = arg(z(t)) (unwrapped phase).  Central differences are used in
/// the interior; forward/backward differences at the endpoints.
///
/// # Arguments
///
/// * `imf` - One IMF (real-valued signal slice).
/// * `fs`  - Sampling frequency in Hz.
///
/// # Returns
///
/// A `Vec<f64>` of the same length as `imf` containing the instantaneous
/// frequency in Hz (clamped to `[0, fs/2]`).
///
/// # Errors
///
/// Returns an error if `imf` is empty or if `fs ≤ 0`.
///
/// # Examples
///
/// ```
/// use scirs2_fft::hht::instantaneous_frequency_hht;
/// use std::f64::consts::PI;
///
/// let fs = 256.0;
/// let freq = 10.0;
/// let n = 256;
/// let imf: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
///     .collect();
///
/// let inst_freq = instantaneous_frequency_hht(&imf, fs).expect("inst_freq failed");
/// assert_eq!(inst_freq.len(), imf.len());
///
/// // In the middle of the signal the estimated frequency should be near 10 Hz
/// let mid_start = n / 4;
/// let mid_end   = 3 * n / 4;
/// let avg: f64  = inst_freq[mid_start..mid_end].iter().sum::<f64>()
///                 / (mid_end - mid_start) as f64;
/// assert!((avg - freq).abs() < 3.0, "Average frequency {avg:.1} should be near {freq}");
/// ```
pub fn instantaneous_frequency_hht(imf: &[f64], fs: f64) -> FFTResult<Vec<f64>> {
    if imf.is_empty() {
        return Err(FFTError::ValueError("IMF cannot be empty".to_string()));
    }
    if fs <= 0.0 {
        return Err(FFTError::ValueError(
            "Sampling frequency must be positive".to_string(),
        ));
    }

    let analytic = analytic_signal_fft(imf)?;
    let phase = unwrap_phase(&analytic);

    let nyquist = fs / 2.0;
    let n = phase.len();
    let mut freq = Vec::with_capacity(n);

    for i in 0..n {
        let f = if i == 0 && n > 1 {
            (phase[1] - phase[0]) * fs / (2.0 * PI)
        } else if i == n - 1 && n > 1 {
            (phase[n - 1] - phase[n - 2]) * fs / (2.0 * PI)
        } else if n > 2 {
            (phase[i + 1] - phase[i - 1]) * fs / (4.0 * PI)
        } else {
            0.0
        };
        freq.push(f.clamp(0.0, nyquist));
    }

    Ok(freq)
}

/// FFT-based analytic signal computation.
fn analytic_signal_fft(signal: &[f64]) -> FFTResult<Vec<Complex64>> {
    let n = signal.len();
    let complex_in: Vec<Complex64> = signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();

    let mut spectrum = fft(&complex_in, Some(n))?;

    // Hilbert filter in frequency domain
    if n % 2 == 0 {
        // Even length
        spectrum[0] *= 1.0; // DC unchanged
        spectrum[n / 2] *= 1.0; // Nyquist unchanged
        for k in 1..n / 2 {
            spectrum[k] *= Complex64::new(2.0, 0.0);
        }
        for k in (n / 2 + 1)..n {
            spectrum[k] = Complex64::new(0.0, 0.0);
        }
    } else {
        // Odd length
        spectrum[0] *= 1.0;
        let half = (n + 1) / 2;
        for k in 1..half {
            spectrum[k] *= Complex64::new(2.0, 0.0);
        }
        for k in half..n {
            spectrum[k] = Complex64::new(0.0, 0.0);
        }
    }

    ifft(&spectrum, Some(n))
}

/// Unwrap phase from a complex analytic signal.
fn unwrap_phase(analytic: &[Complex64]) -> Vec<f64> {
    let n = analytic.len();
    if n == 0 {
        return Vec::new();
    }

    let mut phase = Vec::with_capacity(n);
    let mut prev_angle = analytic[0].im.atan2(analytic[0].re);
    let mut cumulative = 0.0f64;
    phase.push(prev_angle);

    for c in analytic.iter().skip(1) {
        let angle = c.im.atan2(c.re);
        let mut diff = angle - prev_angle;

        // Wrap diff to [-π, π]
        while diff > PI {
            diff -= 2.0 * PI;
            cumulative -= 2.0 * PI;
        }
        while diff < -PI {
            diff += 2.0 * PI;
            cumulative += 2.0 * PI;
        }

        phase.push(angle + cumulative);
        prev_angle = angle;
    }

    phase
}

// ─────────────────────────────────────────────────────────────────────────────
//  HHT spectrum
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Hilbert-Huang marginal spectrum from a set of IMFs.
///
/// For each IMF the instantaneous frequency and amplitude (envelope) are
/// computed, and the squared amplitude is accumulated into a 2D time-frequency
/// energy density matrix of shape `(n_time, n_freq_bins)`.
///
/// # Arguments
///
/// * `imfs`        - Slice of IMF vectors (each of the same length `n_time`).
///                   The residual (last component from [`emd`]) should be
///                   excluded since it carries no oscillatory information.
/// * `fs`          - Sampling frequency in Hz.
/// * `n_freq_bins` - Number of frequency bins spanning `[0, fs/2]`.
///
/// # Returns
///
/// A 2D array of shape `(n_time, n_freq_bins)` where entry `[t, k]` is the
/// total squared amplitude contributed by all IMFs at time `t` and frequency
/// bin `k`.
///
/// # Errors
///
/// Returns an error if `fs ≤ 0`, `n_freq_bins == 0`, or if the IMFs have
/// inconsistent lengths.
///
/// # Examples
///
/// ```
/// use scirs2_fft::hht::{emd, hht_spectrum, StoppingCriterion};
/// use std::f64::consts::PI;
///
/// let fs = 256.0;
/// let n  = 256;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 10.0 * i as f64 / fs).sin())
///     .collect();
///
/// let components = emd(&signal, 8, StoppingCriterion::default()).expect("EMD failed");
/// // Drop the residual (last component)
/// let imfs = &components[..components.len().saturating_sub(1)];
///
/// let spectrum = hht_spectrum(imfs, fs, 64).expect("hht_spectrum failed");
/// assert_eq!(spectrum.shape()[0], n);
/// assert_eq!(spectrum.shape()[1], 64);
/// ```
pub fn hht_spectrum(
    imfs: &[Vec<f64>],
    fs: f64,
    n_freq_bins: usize,
) -> FFTResult<Array2<f64>> {
    if fs <= 0.0 {
        return Err(FFTError::ValueError(
            "Sampling frequency must be positive".to_string(),
        ));
    }
    if n_freq_bins == 0 {
        return Err(FFTError::ValueError(
            "n_freq_bins must be positive".to_string(),
        ));
    }
    if imfs.is_empty() {
        return Ok(Array2::zeros((0, n_freq_bins)));
    }

    let n_time = imfs[0].len();
    // Validate all IMFs have the same length
    for (idx, imf) in imfs.iter().enumerate() {
        if imf.len() != n_time {
            return Err(FFTError::ValueError(format!(
                "IMF {idx} has length {} but expected {n_time}",
                imf.len()
            )));
        }
    }

    let nyquist = fs / 2.0;
    let freq_step = nyquist / n_freq_bins as f64;

    let mut energy = Array2::<f64>::zeros((n_time, n_freq_bins));

    for imf in imfs {
        let analytic = analytic_signal_fft(imf)?;
        let phase = unwrap_phase(&analytic);

        // Compute instantaneous frequency at each sample
        let n = phase.len();
        for t in 0..n_time.min(n) {
            // Central differences for interior; forward/backward at endpoints
            let f = if t == 0 && n > 1 {
                (phase[1] - phase[0]) * fs / (2.0 * PI)
            } else if t == n - 1 && n > 1 {
                (phase[n - 1] - phase[n - 2]) * fs / (2.0 * PI)
            } else if n > 2 {
                (phase[t + 1] - phase[t - 1]) * fs / (4.0 * PI)
            } else {
                0.0
            };
            let f_clamped = f.clamp(0.0, nyquist);

            let amplitude = analytic[t].norm();
            let bin = (f_clamped / freq_step).floor() as usize;
            let bin = bin.min(n_freq_bins - 1);
            energy[[t, bin]] += amplitude * amplitude;
        }
    }

    Ok(energy)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Ensemble EMD (EEMD)
// ─────────────────────────────────────────────────────────────────────────────

/// Perform Ensemble EMD (EEMD) to mitigate mode mixing.
///
/// White Gaussian noise with standard deviation `noise_std` is added to the
/// signal `n_ensemble` times, EMD is performed on each noisy copy, and the
/// IMFs are averaged across ensembles.  The averaged IMFs better separate
/// frequency scales than a single EMD run.
///
/// # Arguments
///
/// * `signal`     - Input signal.
/// * `noise_std`  - Absolute standard deviation of the added white noise.
/// * `n_ensemble` - Number of noise realisations (typically 50–300).
/// * `rng`        - Mutable reference to a 64-bit seed for the simple LCG RNG
///                  used internally.  Pass `&mut 42` for reproducible results.
///
/// # Returns
///
/// A `Vec<Vec<f64>>` of averaged IMFs (the residual is included as the last
/// element, following the same convention as [`emd`]).
///
/// # Errors
///
/// Returns an error if `n_ensemble == 0` or if `signal.len() < 4`.
///
/// # Examples
///
/// ```
/// use scirs2_fft::hht::ensemble_emd;
/// use std::f64::consts::PI;
///
/// let n = 256;
/// let signal: Vec<f64> = (0..n).map(|i| {
///     let t = i as f64 / 256.0;
///     (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 25.0 * t).sin()
/// }).collect();
///
/// let imfs = ensemble_emd(&signal, 0.1, 10, &mut 42).expect("EEMD failed");
/// assert!(!imfs.is_empty(), "EEMD should produce components");
/// ```
pub fn ensemble_emd(
    signal: &[f64],
    noise_std: f64,
    n_ensemble: usize,
    rng: &mut u64,
) -> FFTResult<Vec<Vec<f64>>> {
    let n = signal.len();
    if n < 4 {
        return Err(FFTError::ValueError(
            "Signal must have at least 4 samples for EEMD".to_string(),
        ));
    }
    if n_ensemble == 0 {
        return Err(FFTError::ValueError(
            "n_ensemble must be at least 1".to_string(),
        ));
    }

    let stopping = StoppingCriterion::default();
    let max_imfs: usize = 20;

    let mut all_components: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_ensemble);
    let mut max_components = 0usize;

    for _e in 0..n_ensemble {
        let mut noisy = signal.to_vec();
        for sample in &mut noisy {
            let g = gaussian_sample(rng);
            *sample += noise_std * g;
        }
        let components = emd(&noisy, max_imfs, stopping)?;
        if components.len() > max_components {
            max_components = components.len();
        }
        all_components.push(components);
    }

    if max_components == 0 {
        return Ok(vec![signal.to_vec()]);
    }

    // Average component by component (zero-pad missing components)
    let mut averaged: Vec<Vec<f64>> = vec![vec![0.0f64; n]; max_components];
    let mut counts: Vec<usize> = vec![0usize; max_components];

    for components in &all_components {
        for (k, comp) in components.iter().enumerate() {
            for (i, &v) in comp.iter().enumerate() {
                if i < n {
                    averaged[k][i] += v;
                }
            }
            counts[k] += 1;
        }
    }

    for k in 0..max_components {
        if counts[k] > 0 {
            let c = counts[k] as f64;
            for v in &mut averaged[k] {
                *v /= c;
            }
        }
    }

    Ok(averaged)
}

/// Box-Muller Gaussian sample using internal LCG.
fn gaussian_sample(state: &mut u64) -> f64 {
    let u1 = lcg_f64(state);
    let u2 = lcg_f64(state);
    // Avoid log(0)
    let u1_safe = u1.max(f64::MIN_POSITIVE);
    (-2.0 * u1_safe.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// LCG random number generator producing f64 in (0, 1).
fn lcg_f64(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let v = ((*state >> 11) as f64) / ((1u64 << 53) as f64);
    v.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── cubic_spline_envelope ─────────────────────────────────────────────────

    #[test]
    fn test_cubic_spline_reproduces_knots() {
        let idx = vec![0, 3, 7];
        let val = vec![1.0_f64, 2.0, 0.5];
        let env = cubic_spline_envelope(&idx, &val, 8).expect("spline");
        assert_abs_diff_eq!(env[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(env[3], 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(env[7], 0.5, epsilon = 1e-9);
    }

    #[test]
    fn test_cubic_spline_two_knots_fallback() {
        let idx = vec![0, 7];
        let val = vec![1.0_f64, 3.0];
        let env = cubic_spline_envelope(&idx, &val, 8).expect("linear fallback");
        assert_abs_diff_eq!(env[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(env[7], 3.0, epsilon = 1e-9);
        // Midpoint should be between
        assert!(env[3] > 1.0 && env[3] < 3.0);
    }

    #[test]
    fn test_cubic_spline_too_few_knots() {
        let idx = vec![0];
        let val = vec![1.0];
        assert!(cubic_spline_envelope(&idx, &val, 8).is_err());
    }

    // ── find_extrema ─────────────────────────────────────────────────────────

    #[test]
    fn test_find_extrema_sine() {
        let n = 64;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 4.0 * i as f64 / n as f64).sin())
            .collect();
        let (maxima, minima) = find_extrema(&signal);
        assert!(!maxima.0.is_empty(), "Should find maxima");
        assert!(!minima.0.is_empty(), "Should find minima");
    }

    // ── sifting_process ──────────────────────────────────────────────────────

    #[test]
    fn test_sifting_process_length_preserved() {
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 8.0 * i as f64 / n as f64).sin())
            .collect();
        let imf =
            sifting_process(&signal, 50, StoppingCriterion::default()).expect("sifting");
        assert_eq!(imf.len(), n);
    }

    #[test]
    fn test_sifting_process_too_short() {
        assert!(sifting_process(&[1.0, 2.0, 3.0], 50, StoppingCriterion::default()).is_err());
    }

    // ── emd ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_emd_reconstruction() {
        let n = 256;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / 256.0;
                (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 25.0 * t).sin()
            })
            .collect();

        let components = emd(&signal, 10, StoppingCriterion::default()).expect("EMD");
        assert!(!components.is_empty());

        // Sum of all components (IMFs + residual) should reconstruct signal
        let mut reconstructed = vec![0.0f64; n];
        for comp in &components {
            for (i, &v) in comp.iter().enumerate() {
                if i < n {
                    reconstructed[i] += v;
                }
            }
        }
        for i in 0..n {
            assert_abs_diff_eq!(reconstructed[i], signal[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_emd_too_short() {
        assert!(emd(&[1.0, 2.0], 10, StoppingCriterion::default()).is_err());
    }

    #[test]
    fn test_emd_linear_signal() {
        let n = 64;
        let signal: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let components = emd(&signal, 5, StoppingCriterion::default()).expect("EMD linear");
        // Linear signal has no extrema → only the residual should be produced
        assert!(!components.is_empty());
    }

    #[test]
    fn test_emd_fixed_iterations_criterion() {
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / n as f64).sin())
            .collect();
        let components = emd(
            &signal,
            5,
            StoppingCriterion::FixedIterations { n_iterations: 10 },
        )
        .expect("EMD fixed iter");
        assert!(!components.is_empty());
    }

    // ── instantaneous_frequency_hht ─────────────────────────────────────────

    #[test]
    fn test_inst_freq_sine_accuracy() {
        let fs = 512.0;
        let n = 512;
        let freq = 20.0_f64;
        let imf: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect();

        let inst_freq = instantaneous_frequency_hht(&imf, fs).expect("inst_freq");
        assert_eq!(inst_freq.len(), n);

        // Average frequency in the middle should be near the true frequency
        let mid_start = n / 4;
        let mid_end = 3 * n / 4;
        let avg: f64 = inst_freq[mid_start..mid_end].iter().sum::<f64>()
            / (mid_end - mid_start) as f64;
        assert!(
            (avg - freq).abs() < 5.0,
            "Avg inst freq {avg:.1} should be near {freq}"
        );
    }

    #[test]
    fn test_inst_freq_empty_imf() {
        assert!(instantaneous_frequency_hht(&[], 256.0).is_err());
    }

    #[test]
    fn test_inst_freq_invalid_fs() {
        let imf = vec![1.0; 32];
        assert!(instantaneous_frequency_hht(&imf, 0.0).is_err());
        assert!(instantaneous_frequency_hht(&imf, -1.0).is_err());
    }

    #[test]
    fn test_inst_freq_non_negative() {
        let fs = 256.0;
        let n = 256;
        let imf: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 15.0 * i as f64 / fs).sin())
            .collect();
        let inst_freq = instantaneous_frequency_hht(&imf, fs).expect("inst_freq");
        for &f in &inst_freq {
            assert!(
                f >= 0.0 && f <= fs / 2.0,
                "inst_freq {f} out of [0, Nyquist]"
            );
        }
    }

    // ── hht_spectrum ─────────────────────────────────────────────────────────

    #[test]
    fn test_hht_spectrum_shape() {
        let fs = 256.0;
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / fs).sin())
            .collect();
        let components = emd(&signal, 5, StoppingCriterion::default()).expect("EMD");
        let imfs = &components[..components.len().saturating_sub(1)];

        let spectrum = hht_spectrum(imfs, fs, 64).expect("hht_spectrum");
        assert_eq!(spectrum.shape()[0], n);
        assert_eq!(spectrum.shape()[1], 64);
    }

    #[test]
    fn test_hht_spectrum_non_negative() {
        let fs = 256.0;
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / fs).sin())
            .collect();
        let components = emd(&signal, 5, StoppingCriterion::default()).expect("EMD");
        let imfs = &components[..components.len().saturating_sub(1)];

        let spectrum = hht_spectrum(imfs, fs, 32).expect("hht_spectrum");
        for &v in spectrum.iter() {
            assert!(v >= 0.0, "HHT spectrum must be non-negative, got {v}");
        }
    }

    #[test]
    fn test_hht_spectrum_inconsistent_imf_lengths() {
        let imfs = vec![vec![1.0; 10], vec![1.0; 12]];
        assert!(hht_spectrum(&imfs, 256.0, 32).is_err());
    }

    #[test]
    fn test_hht_spectrum_invalid_params() {
        let imfs = vec![vec![1.0; 10]];
        assert!(hht_spectrum(&imfs, 0.0, 32).is_err());
        assert!(hht_spectrum(&imfs, 256.0, 0).is_err());
    }

    // ── ensemble_emd ─────────────────────────────────────────────────────────

    #[test]
    fn test_ensemble_emd_basic() {
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / 128.0;
                (2.0 * PI * 5.0 * t).sin() + 0.3 * (2.0 * PI * 20.0 * t).sin()
            })
            .collect();

        let imfs = ensemble_emd(&signal, 0.1, 5, &mut 42).expect("EEMD");
        assert!(!imfs.is_empty());
    }

    #[test]
    fn test_ensemble_emd_zero_ensembles() {
        let signal = vec![1.0; 32];
        assert!(ensemble_emd(&signal, 0.1, 0, &mut 42).is_err());
    }

    #[test]
    fn test_ensemble_emd_too_short() {
        assert!(ensemble_emd(&[1.0, 2.0], 0.1, 5, &mut 42).is_err());
    }

    #[test]
    fn test_ensemble_emd_deterministic_rng() {
        let n = 64;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 8.0 * i as f64 / 64.0).sin())
            .collect();

        let imfs1 = ensemble_emd(&signal, 0.1, 3, &mut 12345).expect("EEMD 1");
        let imfs2 = ensemble_emd(&signal, 0.1, 3, &mut 12345).expect("EEMD 2");

        assert_eq!(imfs1.len(), imfs2.len());
        for (a, b) in imfs1.iter().zip(imfs2.iter()) {
            for (&va, &vb) in a.iter().zip(b.iter()) {
                assert_abs_diff_eq!(va, vb, epsilon = 1e-12);
            }
        }
    }

    // ── EmpiricalModeDecomposition struct API ─────────────────────────────────

    #[test]
    fn test_emd_struct_decompose() {
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / 128.0).sin())
            .collect();

        let emd_runner = EmpiricalModeDecomposition::new()
            .max_imfs(5)
            .max_sifts(100)
            .stopping_criterion(StoppingCriterion::Cauchy {
                threshold: 0.1,
                s_number: 2,
            });

        let components = emd_runner.decompose(&signal).expect("decompose");
        assert!(!components.is_empty());
    }
}
