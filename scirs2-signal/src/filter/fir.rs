// FIR (Finite Impulse Response) filter design functions
//
// This module provides comprehensive FIR filter design capabilities including
// window-based design (firwin) and optimal equiripple design (Parks-McClellan/Remez).
// FIR filters offer linear phase response and guaranteed stability.

use super::common::validation::validate_cutoff_frequency;
use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::{Float, NumCast};
use std::fmt::Debug;

/// Minimum absolute difference for barycentric weight computation to avoid division by zero.
const BARY_EPSILON: f64 = 1e-15;
/// Minimum denominator magnitude for barycentric interpolation to avoid division by zero.
const BARY_MIN_DENOM: f64 = 1e-30;

#[allow(unused_imports)]
/// FIR filter design using window method
///
/// Designs a linear phase FIR filter using the window method. The filter
/// is obtained by truncating and windowing the ideal impulse response.
///
/// # Arguments
///
/// * `numtaps` - Number of filter taps (filter order + 1)
/// * `cutoff` - Cutoff frequency (normalized from 0 to 1, where 1 is Nyquist frequency)
/// * `window` - Window function name ("hamming", "hann", "blackman", "kaiser", etc.)
/// * `pass_zero` - If true, the filter is lowpass; if false, highpass
///
/// # Returns
///
/// * Filter coefficients as a vector
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::fir::firwin;
///
/// // Design a 65-tap lowpass filter with Hamming window
/// let h = firwin(65, 0.3, "hamming", true).expect("Operation failed");
///
/// // Design a highpass filter
/// let h = firwin(65, 0.3, "hamming", false).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn firwin<T>(
    _numtaps: usize,
    cutoff: T,
    window: &str,
    pass_zero: bool,
) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    if _numtaps < 3 {
        return Err(SignalError::ValueError(
            "Number of taps must be at least 3".to_string(),
        ));
    }

    let wc = validate_cutoff_frequency(cutoff)?;

    // Calculate the ideal impulse response
    let mid = (_numtaps - 1) as f64 / 2.0;
    let mut h = vec![0.0; _numtaps];

    for (i, item) in h.iter_mut().enumerate() {
        let n = i as f64 - mid;

        if n == 0.0 {
            // At n=0, use L'Hôpital's rule result
            *item = if pass_zero {
                wc / std::f64::consts::PI
            } else {
                1.0 - wc / std::f64::consts::PI
            };
        } else {
            // General case: sinc function
            let sinc_val = (wc * std::f64::consts::PI * n).sin() / (std::f64::consts::PI * n);
            *item = if pass_zero {
                sinc_val
            } else {
                // Highpass: subtract lowpass from delta function
                if i == _numtaps / 2 {
                    1.0 - sinc_val
                } else {
                    -sinc_val
                }
            };
        }
    }

    // Apply window function
    let window_coeffs = generate_window(_numtaps, window)?;
    for (i, coeff) in h.iter_mut().enumerate() {
        *coeff *= window_coeffs[i];
    }

    // Normalize to ensure unity gain at DC (for lowpass) or Nyquist (for highpass)
    let sum: f64 = h.iter().sum();
    if pass_zero && sum.abs() > 1e-10 {
        for coeff in &mut h {
            *coeff /= sum;
        }
    } else if !pass_zero {
        // For highpass, normalize for unity gain at Nyquist
        let nyquist_response: f64 = h
            .iter()
            .enumerate()
            .map(|(i, &coeff)| coeff * (-1.0_f64).powi(i as i32))
            .sum();
        if nyquist_response.abs() > 1e-10 {
            for coeff in &mut h {
                *coeff /= nyquist_response;
            }
        }
    }

    Ok(h)
}

/// Parks-McClellan optimal FIR filter design (Remez exchange algorithm)
///
/// Design a linear phase FIR filter using the Parks-McClellan algorithm.
/// The algorithm finds the filter coefficients that minimize the maximum
/// error between the desired and actual frequency response.
///
/// # Arguments
///
/// * `numtaps` - Number of filter taps (filter order + 1)
/// * `bands` - Frequency bands specified as pairs of band edges (0 to 1, where 1 is Nyquist)
/// * `desired` - Desired gain for each band
/// * `weights` - Relative weights for each band. Can be either `num_bands` weights (one per band)
///   or `bands.len()` weights (per-edge, averaged per band).
/// * `max_iter` - Maximum number of iterations (default: 25)
/// * `grid_density` - Grid density for frequency sampling (default: 16)
///
/// # Returns
///
/// * Filter coefficients as a vector
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::fir::remez;
///
/// // Design a 65-tap lowpass filter
/// // Passband: 0-0.4, Stopband: 0.45-1.0
/// let bands = vec![0.0, 0.4, 0.45, 1.0];
/// let desired = vec![1.0, 1.0, 0.0, 0.0];
/// let h = remez(65, &bands, &desired, None, None, None).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn remez(
    numtaps: usize,
    bands: &[f64],
    desired: &[f64],
    weights: Option<&[f64]>,
    max_iter: Option<usize>,
    grid_density: Option<usize>,
) -> SignalResult<Vec<f64>> {
    // Validate inputs
    if numtaps < 3 {
        return Err(SignalError::ValueError(
            "Number of taps must be at least 3".to_string(),
        ));
    }

    if !bands.len().is_multiple_of(2) || bands.len() < 2 {
        return Err(SignalError::ValueError(
            "Bands must be specified as pairs of edges".to_string(),
        ));
    }

    if desired.len() != bands.len() {
        return Err(SignalError::ValueError(
            "Desired array must have same length as bands".to_string(),
        ));
    }

    // Check that bands are monotonically increasing
    for i in 1..bands.len() {
        if bands[i] <= bands[i - 1] {
            return Err(SignalError::ValueError(
                "Band edges must be monotonically increasing".to_string(),
            ));
        }
    }

    // Check that bands are within [0, 1]
    if bands[0] < 0.0 || bands[bands.len() - 1] > 1.0 {
        return Err(SignalError::ValueError(
            "Band edges must be between 0 and 1".to_string(),
        ));
    }

    let num_bands = bands.len() / 2;

    // Resolve per-band weights
    let band_weights: Vec<f64> = if let Some(w) = weights {
        if w.len() == num_bands {
            // One weight per band
            w.to_vec()
        } else if w.len() == bands.len() {
            // Per-edge weights: average each pair
            (0..num_bands)
                .map(|i| (w[2 * i] + w[2 * i + 1]) / 2.0)
                .collect()
        } else {
            return Err(SignalError::ValueError(format!(
                "Weights must have {} (per band) or {} (per edge) elements, got {}",
                num_bands,
                bands.len(),
                w.len()
            )));
        }
    } else {
        vec![1.0; num_bands]
    };

    let max_iter = max_iter.unwrap_or(25);
    let grid_density = grid_density.unwrap_or(16);

    // Calculate filter half-order
    let filter_order = numtaps - 1;
    let m = filter_order / 2;

    // Alternation theorem: M+2 extremal frequencies
    let r = m + 2;

    // Set up the dense frequency grid
    let grid_size = grid_density * filter_order;
    let mut omega_grid = Vec::with_capacity(grid_size);
    let mut desired_grid = Vec::with_capacity(grid_size);
    let mut weight_grid = Vec::with_capacity(grid_size);
    let mut band_index_grid = Vec::with_capacity(grid_size);

    // Build the frequency grid for each band
    for band_idx in 0..num_bands {
        let band_start = bands[2 * band_idx];
        let band_end = bands[2 * band_idx + 1];
        let band_points = ((band_end - band_start) * grid_size as f64)
            .round()
            .max(2.0) as usize;

        for i in 0..band_points {
            let frac = if band_points > 1 {
                i as f64 / (band_points as f64 - 1.0)
            } else {
                0.0
            };
            let omega = band_start + (band_end - band_start) * frac;
            omega_grid.push(omega * std::f64::consts::PI);

            // Linear interpolation for desired response
            let des = desired[2 * band_idx] * (1.0 - frac) + desired[2 * band_idx + 1] * frac;
            desired_grid.push(des);

            // Use per-band weight
            weight_grid.push(band_weights[band_idx]);
            band_index_grid.push(band_idx);
        }
    }

    let grid_len = omega_grid.len();
    if grid_len < r {
        return Err(SignalError::ValueError(
            "Grid too small for the requested filter order".to_string(),
        ));
    }

    // Initialize extremal frequencies uniformly across the grid
    let mut extremal_freqs: Vec<usize> = (0..r).map(|i| i * (grid_len - 1) / (r - 1)).collect();

    // Remez exchange algorithm
    let mut prev_delta = f64::MAX;

    for _iter in 0..max_iter {
        // Step 1: Compute x = cos(omega) at extremal points
        let x_ext: Vec<f64> = extremal_freqs
            .iter()
            .map(|&idx| omega_grid[idx].cos())
            .collect();

        // Step 2: Compute barycentric weights
        let bary = compute_barycentric_weights(&x_ext);

        // Step 3: Gather desired and weight values at extremal points
        let d_ext: Vec<f64> = extremal_freqs
            .iter()
            .map(|&idx| desired_grid[idx])
            .collect();
        let w_ext: Vec<f64> = extremal_freqs.iter().map(|&idx| weight_grid[idx]).collect();

        // Step 4: Compute delta via alternation formula
        let (num, den) = delta_numerator_denominator(&bary, &d_ext, &w_ext);
        let delta = if den.abs() < BARY_MIN_DENOM {
            0.0
        } else {
            num / den
        };

        // Step 5: Compute adjusted values E_i = D_i - (-1)^i * delta / W_i
        let e_ext: Vec<f64> = (0..r)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                d_ext[i] - sign * delta / w_ext[i]
            })
            .collect();

        // Step 6: Evaluate P on dense grid using barycentric interpolation with E values
        let mut errors = Vec::with_capacity(grid_len);
        for i in 0..grid_len {
            let xg = omega_grid[i].cos();
            let p_val = barycentric_eval(&bary, &x_ext, &e_ext, xg);
            let err = weight_grid[i] * (desired_grid[i] - p_val);
            errors.push(err);
        }

        // Step 7: Find new extremal set - local maxima of |weighted error|
        let abs_errors: Vec<f64> = errors.iter().map(|e| e.abs()).collect();
        let mut candidates: Vec<(usize, f64)> = Vec::new();

        // Check boundaries
        if grid_len > 1 && abs_errors[0] >= abs_errors[1] {
            candidates.push((0, abs_errors[0]));
        }
        if grid_len > 1 && abs_errors[grid_len - 1] >= abs_errors[grid_len - 2] {
            candidates.push((grid_len - 1, abs_errors[grid_len - 1]));
        }

        // Interior local maxima
        for i in 1..(grid_len - 1) {
            if abs_errors[i] >= abs_errors[i - 1] && abs_errors[i] >= abs_errors[i + 1] {
                candidates.push((i, abs_errors[i]));
            }
        }

        // Sort by error magnitude descending, keep r largest
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(r);

        // Sort by grid index for proper ordering
        candidates.sort_by_key(|c| c.0);

        if candidates.len() >= r {
            extremal_freqs = candidates.iter().map(|c| c.0).collect();
        }

        // Convergence check: relative change in delta
        let abs_delta = delta.abs();
        if abs_delta > 0.0 && (prev_delta - abs_delta).abs() < 1e-10 * abs_delta {
            break;
        }
        prev_delta = abs_delta;
    }

    // Final pass: compute the polynomial values at extremal points one more time
    let x_ext: Vec<f64> = extremal_freqs
        .iter()
        .map(|&idx| omega_grid[idx].cos())
        .collect();
    let bary = compute_barycentric_weights(&x_ext);
    let d_ext: Vec<f64> = extremal_freqs
        .iter()
        .map(|&idx| desired_grid[idx])
        .collect();
    let w_ext: Vec<f64> = extremal_freqs.iter().map(|&idx| weight_grid[idx]).collect();
    let (num, den) = delta_numerator_denominator(&bary, &d_ext, &w_ext);
    let delta = if den.abs() < BARY_MIN_DENOM {
        0.0
    } else {
        num / den
    };
    let e_ext: Vec<f64> = (0..r)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            d_ext[i] - sign * delta / w_ext[i]
        })
        .collect();

    // Coefficient extraction via inverse DCT-I
    // Sample P at M+1 evenly-spaced DCT-I nodes
    let m_plus_1 = m + 1;
    let mut p_samples = Vec::with_capacity(m_plus_1);
    for k in 0..m_plus_1 {
        let theta = std::f64::consts::PI * k as f64 / m as f64;
        let xg = theta.cos();
        let p_val = barycentric_eval(&bary, &x_ext, &e_ext, xg);
        p_samples.push(p_val);
    }

    // Inverse DCT-I to get cosine-series coefficients
    let mut a_coeffs = vec![0.0; m_plus_1];
    for k in 0..m_plus_1 {
        let mut sum = 0.0;
        for (n, &p_n) in p_samples.iter().enumerate() {
            let cos_val = (std::f64::consts::PI * k as f64 * n as f64 / m as f64).cos();
            let weight = if n == 0 || n == m { 0.5 } else { 1.0 };
            sum += weight * p_n * cos_val;
        }
        a_coeffs[k] = sum * 2.0 / m as f64;
    }
    // The first coefficient needs halving (DCT-I normalization)
    a_coeffs[0] /= 2.0;

    // Map to symmetric FIR taps: h[M] = a[0], h[M +/- k] = a[k]/2
    let mut h = vec![0.0; numtaps];
    h[m] = a_coeffs[0];
    for k in 1..m_plus_1 {
        let val = a_coeffs[k] / 2.0;
        if m + k < numtaps {
            h[m + k] = val;
        }
        if m >= k {
            h[m - k] = val;
        }
    }

    Ok(h)
}

/// Compute barycentric weights for interpolation nodes.
///
/// Given nodes `x[0..n]`, returns weights `lambda[i] = 1 / prod_{j != i}(x[i] - x[j])`.
fn compute_barycentric_weights(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut w = vec![1.0; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let diff = x[i] - x[j];
                if diff.abs() < BARY_EPSILON {
                    // Avoid division by zero; use a large but finite weight contribution
                    w[i] /= if diff >= 0.0 {
                        BARY_EPSILON
                    } else {
                        -BARY_EPSILON
                    };
                } else {
                    w[i] /= diff;
                }
            }
        }
    }
    w
}

/// Evaluate a polynomial at `xg` via barycentric interpolation.
///
/// `bary` are the barycentric weights, `x` are the nodes, `y` are the function values at nodes.
fn barycentric_eval(bary: &[f64], x: &[f64], y: &[f64], xg: f64) -> f64 {
    let n = x.len();
    let mut num = 0.0;
    let mut den = 0.0;

    for i in 0..n {
        let diff = xg - x[i];
        if diff.abs() < BARY_EPSILON {
            // xg is (nearly) coincident with node i; return the value directly
            return y[i];
        }
        let term = bary[i] / diff;
        num += term * y[i];
        den += term;
    }

    if den.abs() < BARY_MIN_DENOM {
        // Fallback: return average of y values
        let sum: f64 = y.iter().sum();
        return sum / n as f64;
    }

    num / den
}

/// Compute the numerator and denominator for the Remez delta formula.
///
/// `delta = sum(lambda_i * D_i) / sum((-1)^i * lambda_i / W_i)`
fn delta_numerator_denominator(bary: &[f64], d: &[f64], w: &[f64]) -> (f64, f64) {
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..bary.len() {
        num += bary[i] * d[i];
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        den += sign * bary[i] / w[i];
    }
    (num, den)
}

/// Generate a window function
///
/// Creates a window function of the specified type and length.
///
/// # Arguments
///
/// * `length` - Window length
/// * `window_type` - Window type ("hamming", "hann", "blackman", "kaiser", etc.)
///
/// # Returns
///
/// * Window coefficients as a vector
#[allow(dead_code)]
fn generate_window(_length: usize, windowtype: &str) -> SignalResult<Vec<f64>> {
    let mut window = vec![0.0; _length];

    match windowtype.to_lowercase().as_str() {
        "hamming" => {
            for (i, w) in window.iter_mut().enumerate() {
                let n = i as f64;
                let total = _length as f64;
                *w = 0.54 - 0.46 * (2.0 * std::f64::consts::PI * n / (total - 1.0)).cos();
            }
        }
        "hann" | "hanning" => {
            for (i, w) in window.iter_mut().enumerate() {
                let n = i as f64;
                let total = _length as f64;
                *w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * n / (total - 1.0)).cos());
            }
        }
        "blackman" => {
            for (i, w) in window.iter_mut().enumerate() {
                let n = i as f64;
                let total = _length as f64;
                let arg = 2.0 * std::f64::consts::PI * n / (total - 1.0);
                *w = 0.42 - 0.5 * arg.cos() + 0.08 * (2.0 * arg).cos();
            }
        }
        "rectangular" | "boxcar" => {
            window.fill(1.0);
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown window type: {}. Supported types: hamming, hann, blackman, rectangular",
                windowtype
            )));
        }
    }

    Ok(window)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remez_length() {
        let bands = vec![0.0, 0.3, 0.4, 1.0];
        let desired = vec![1.0, 1.0, 0.0, 0.0];
        let h = remez(65, &bands, &desired, None, None, None).unwrap();
        assert_eq!(h.len(), 65);

        let h2 = remez(101, &bands, &desired, None, None, None).unwrap();
        assert_eq!(h2.len(), 101);
    }

    #[test]
    fn test_remez_symmetric() {
        let bands = vec![0.0, 0.3, 0.4, 1.0];
        let desired = vec![1.0, 1.0, 0.0, 0.0];
        let h = remez(65, &bands, &desired, None, None, None).unwrap();

        let n = h.len();
        for i in 0..n / 2 {
            let diff = (h[i] - h[n - 1 - i]).abs();
            assert!(
                diff < 1e-12,
                "Filter not symmetric at index {}: h[{}]={}, h[{}]={}",
                i,
                i,
                h[i],
                n - 1 - i,
                h[n - 1 - i]
            );
        }
    }

    #[test]
    fn test_remez_lowpass_frequency_response() {
        let bands = vec![0.0, 0.3, 0.4, 1.0];
        let desired = vec![1.0, 1.0, 0.0, 0.0];
        let weights = vec![1.0, 10.0];
        let h = remez(65, &bands, &desired, Some(&weights), Some(40), None).unwrap();

        // Compute frequency response at a few points
        let n = h.len();

        // Passband: f=0.15 (well inside passband)
        let f_pass = 0.15 * std::f64::consts::PI;
        let gain_pass: f64 = h
            .iter()
            .enumerate()
            .map(|(i, &c)| c * (f_pass * i as f64).cos())
            .sum();
        assert!(
            gain_pass.abs() > 0.75,
            "Passband gain too low: {}",
            gain_pass.abs()
        );

        // Stopband: f=0.6 (well inside stopband)
        let f_stop = 0.6 * std::f64::consts::PI;
        let gain_stop: f64 = h
            .iter()
            .enumerate()
            .map(|(i, &c)| c * (f_stop * (i as f64 - (n - 1) as f64 / 2.0)).cos())
            .sum();
        assert!(
            gain_stop.abs() < 0.15,
            "Stopband gain too high: {}",
            gain_stop.abs()
        );
    }

    #[test]
    fn test_remez_bandpass_frequency_response() {
        let bands = vec![0.0, 0.15, 0.2, 0.4, 0.45, 1.0];
        let desired = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let weights = vec![10.0, 1.0, 10.0];
        let h = remez(101, &bands, &desired, Some(&weights), Some(40), None).unwrap();

        let n = h.len();
        let mid = (n - 1) as f64 / 2.0;

        // Passband center: f=0.3 (center of passband)
        let f_pass = 0.3 * std::f64::consts::PI;
        let gain_pass: f64 = h
            .iter()
            .enumerate()
            .map(|(i, &c)| c * (f_pass * (i as f64 - mid)).cos())
            .sum();
        assert!(
            gain_pass.abs() > 0.85,
            "Bandpass passband gain too low: {}",
            gain_pass.abs()
        );

        // Lower stopband: f=0.05
        let f_stop_lo = 0.05 * std::f64::consts::PI;
        let gain_stop_lo: f64 = h
            .iter()
            .enumerate()
            .map(|(i, &c)| c * (f_stop_lo * (i as f64 - mid)).cos())
            .sum();
        assert!(
            gain_stop_lo.abs() < 0.15,
            "Lower stopband gain too high: {}",
            gain_stop_lo.abs()
        );

        // Upper stopband: f=0.7
        let f_stop_hi = 0.7 * std::f64::consts::PI;
        let gain_stop_hi: f64 = h
            .iter()
            .enumerate()
            .map(|(i, &c)| c * (f_stop_hi * (i as f64 - mid)).cos())
            .sum();
        assert!(
            gain_stop_hi.abs() < 0.15,
            "Upper stopband gain too high: {}",
            gain_stop_hi.abs()
        );
    }
}
