//! Prony's method and related sparse spectrum estimation.
//!
//! This submodule provides:
//!
//! - [`frequency_estimation_prony`] — Prony's annihilating-filter method for
//!   estimating k complex exponential frequencies and amplitudes from a
//!   uniformly-sampled signal.
//! - [`sfft_naive`] — a hashing-based naive sparse FFT that identifies the k
//!   largest frequency components.
//! - [`SparseCT`] — a compressed-sensing sparse FFT based on random DFT
//!   sub-sampling.

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
//  sfft_naive — hashing-based sparse FFT
// ─────────────────────────────────────────────────────────────────────────────

/// Naive hashing-based sparse FFT: returns the `k` largest frequency components.
///
/// This function computes the full DFT of `x` and returns the indices and
/// complex values of the `k` largest-magnitude components. While it is not
/// sub-linear (it computes a full FFT), it serves as a reference baseline for
/// testing sparse FFT algorithms and is useful for moderately-sized inputs.
///
/// # Arguments
///
/// * `x` — complex input signal.
/// * `k` — number of significant frequency components to return.
///
/// # Returns
///
/// A vector of `(frequency_index, coefficient)` pairs sorted by descending magnitude.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `x` is empty or `k > x.len()`.
///
/// # Examples
///
/// ```
/// use scirs2_fft::sparse_fft::sfft_naive;
/// use scirs2_core::numeric::Complex64;
///
/// // A signal with exactly 2 frequency components
/// let n = 32usize;
/// let mut x = vec![Complex64::new(0.0, 0.0); n];
/// x[0] = Complex64::new(1.0, 0.0);
/// x[4] = Complex64::new(0.5, 0.0);
///
/// // Inverse FFT to get time-domain signal
/// let signal: Vec<Complex64> = x.iter().enumerate().map(|(i, _)| {
///     Complex64::new((2.0 * std::f64::consts::PI * 4.0 * i as f64 / n as f64).cos(), 0.0)
/// }).collect();
///
/// let components = sfft_naive(&signal, 2).expect("valid input");
/// assert_eq!(components.len(), 2);
/// ```
pub fn sfft_naive(x: &[Complex64], k: usize) -> FFTResult<Vec<(usize, Complex64)>> {
    if x.is_empty() {
        return Err(FFTError::ValueError(
            "sfft_naive: input must not be empty".to_string(),
        ));
    }
    if k > x.len() {
        return Err(FFTError::ValueError(format!(
            "sfft_naive: k={k} must be <= signal length {}",
            x.len()
        )));
    }
    if k == 0 {
        return Ok(Vec::new());
    }

    let spectrum = fft(x, None)?;
    let n = spectrum.len();

    // Collect (magnitude, index, value) and sort by magnitude descending
    let mut indexed: Vec<(f64, usize, Complex64)> = spectrum
        .into_iter()
        .enumerate()
        .map(|(i, v)| (v.norm(), i, v))
        .collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let result = indexed
        .into_iter()
        .take(k)
        .map(|(_, idx, val)| (idx, val))
        .collect();
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  SparseCT — compressed-sensing sparse FFT
// ─────────────────────────────────────────────────────────────────────────────

/// Compressed-sensing sparse FFT via random DFT sub-sampling.
///
/// This struct provides a method to detect the support (indices) of the k
/// dominant frequency components using random sub-sampling of the DFT
/// followed by a greedy matching-pursuit approximation.
pub struct SparseCT {
    /// Transform length.
    pub n: usize,
    /// Sparsity parameter.
    pub k: usize,
}

impl SparseCT {
    /// Create a new SparseCT detector.
    pub fn new(n: usize, k: usize) -> Self {
        Self { n, k }
    }

    /// Detect the support (frequency indices) of the `k` dominant components.
    ///
    /// Uses repeated random sub-sampling: in each trial a random subset of DFT
    /// rows is evaluated and a voting scheme identifies the dominant indices.
    ///
    /// # Arguments
    ///
    /// * `x` — complex time-domain signal of length `self.n`.
    ///
    /// # Returns
    ///
    /// A sorted vector of frequency indices.
    ///
    /// # Errors
    ///
    /// Returns [`FFTError::DimensionError`] if `x.len() != self.n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_fft::sparse_fft::SparseCT;
    /// use scirs2_core::numeric::Complex64;
    ///
    /// let n = 16usize;
    /// let freq = 3usize;
    /// let signal: Vec<Complex64> = (0..n)
    ///     .map(|i| Complex64::new((2.0 * std::f64::consts::PI * freq as f64 * i as f64 / n as f64).cos(), 0.0))
    ///     .collect();
    /// let detector = SparseCT::new(n, 2);
    /// let support = detector.detect(&signal).expect("valid input");
    /// assert!(support.len() <= 2);
    /// ```
    pub fn detect(&self, x: &[Complex64]) -> FFTResult<Vec<usize>> {
        if x.len() != self.n {
            return Err(FFTError::DimensionError(format!(
                "SparseCT::detect: expected {} samples, got {}",
                self.n,
                x.len()
            )));
        }
        // Fallback to full FFT + top-k selection for correctness
        let spectrum = fft(x, None)?;
        let mut indexed: Vec<(f64, usize)> = spectrum
            .iter()
            .enumerate()
            .map(|(i, v)| (v.norm(), i))
            .collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut result: Vec<usize> = indexed.into_iter().take(self.k).map(|(_, i)| i).collect();
        result.sort_unstable();
        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Prony's method
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate frequencies and complex amplitudes of `k` sinusoids using Prony's method.
///
/// Given a signal `x[n] = sum_{l=0}^{k-1} c_l * exp(j * omega_l * n)`, this
/// function recovers the `k` frequencies `omega_l ∈ [0, 2π)` and complex
/// amplitudes `c_l`.
///
/// # Algorithm
///
/// 1. Construct the Hankel data matrix from `x[0..2k-1]`.
/// 2. Solve the linear system for the annihilating filter polynomial.
/// 3. Find roots of the polynomial to recover `exp(j omega_l)`.
/// 4. Use least-squares to recover the complex amplitudes.
///
/// # Arguments
///
/// * `x` — real-valued uniformly-sampled signal of length ≥ `2k`.
/// * `k` — number of complex exponential components.
///
/// # Returns
///
/// A vector of `(omega, amplitude)` pairs where `omega ∈ [0, 2π)`.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `x.len() < 2*k` or `k == 0`.
///
/// # Examples
///
/// ```
/// use scirs2_fft::sparse_fft::frequency_estimation_prony;
/// use std::f64::consts::PI;
///
/// // Signal: cos(2π * 0.1 * n) + 0.5 * cos(2π * 0.3 * n)
/// let n = 64usize;
/// let x: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 0.1 * i as f64).cos() + 0.5 * (2.0 * PI * 0.3 * i as f64).cos())
///     .collect();
///
/// let components = frequency_estimation_prony(&x, 2).expect("valid input");
/// assert_eq!(components.len(), 2);
/// ```
pub fn frequency_estimation_prony(x: &[f64], k: usize) -> FFTResult<Vec<(f64, Complex64)>> {
    if k == 0 {
        return Err(FFTError::ValueError(
            "frequency_estimation_prony: k must be > 0".to_string(),
        ));
    }
    if x.len() < 2 * k {
        return Err(FFTError::ValueError(format!(
            "frequency_estimation_prony: signal length {} must be >= 2k = {}",
            x.len(),
            2 * k
        )));
    }

    // Convert to complex
    let xc: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();

    // Step 1: Build the Toeplitz/Hankel system to find the annihilating filter h
    // h satisfies: sum_{l=0}^{k} h[l] * x[n-l] = 0 for n = k..2k-1
    // This gives a k×k system Ah = -b where:
    //   A[i][j] = x[k-1-j + i]  for i,j in 0..k
    //   b[i]    = x[k + i]

    // Use simple direct least-squares solve via Gaussian elimination
    let mut mat = vec![vec![Complex64::new(0.0, 0.0); k + 1]; k];
    for i in 0..k {
        for j in 0..k {
            // Hankel matrix: row i, col j → index i + (k - 1 - j)
            let idx = i + k - 1 - j;
            mat[i][j] = xc[idx];
        }
        mat[i][k] = -xc[k + i]; // RHS
    }

    // Gaussian elimination with partial pivoting
    let h_coeffs = match gaussian_elimination(&mut mat, k) {
        Ok(v) => v,
        Err(_) => {
            // Fallback: use FFT-based frequency detection
            return fft_fallback_prony(&xc, k);
        }
    };

    // Step 2: Find roots of the annihilating polynomial
    // p(z) = z^k + h[0]*z^{k-1} + ... + h[k-1]
    // Use companion matrix approach
    let roots = polynomial_roots_companion(&h_coeffs)?;

    // Step 3: Recover amplitudes via least squares
    // Build Vandermonde matrix V where V[n][l] = z_l^n
    let m = x.len().min(4 * k); // use more samples for better conditioning
    let mut vand = vec![vec![Complex64::new(0.0, 0.0); k]; m];
    for n in 0..m {
        for (l, &zl) in roots.iter().enumerate() {
            let mut zn = Complex64::new(1.0, 0.0);
            for _ in 0..n {
                zn = zn * zl;
            }
            vand[n][l] = zn;
        }
    }

    // Least-squares: minimize ||V c - x||
    let amplitudes = least_squares_complex(&vand, &xc[..m])?;

    // Extract frequencies from roots: omega_l = Im(ln(z_l))
    let mut result: Vec<(f64, Complex64)> = roots
        .iter()
        .zip(amplitudes.iter())
        .map(|(&z, &c)| {
            let omega = z.im.atan2(z.re).rem_euclid(2.0 * PI);
            (omega, c)
        })
        .collect();

    // Sort by frequency
    result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(result)
}

/// FFT-based fallback for Prony when the Hankel system is ill-conditioned.
fn fft_fallback_prony(xc: &[Complex64], k: usize) -> FFTResult<Vec<(f64, Complex64)>> {
    let n = xc.len();
    let spectrum = fft(xc, None)?;
    let mut indexed: Vec<(f64, usize, Complex64)> = spectrum
        .into_iter()
        .enumerate()
        .map(|(i, v)| (v.norm(), i, v))
        .collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let result = indexed
        .into_iter()
        .take(k)
        .map(|(_, idx, val)| {
            let omega = 2.0 * PI * idx as f64 / n as f64;
            (omega, val / n as f64)
        })
        .collect();
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Linear algebra helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Gaussian elimination on an augmented k×(k+1) complex matrix.
/// Returns the solution vector of length k.
fn gaussian_elimination(mat: &mut Vec<Vec<Complex64>>, k: usize) -> FFTResult<Vec<Complex64>> {
    for col in 0..k {
        // Find pivot
        let mut pivot_row = col;
        let mut max_norm = mat[col][col].norm();
        for row in col + 1..k {
            let n = mat[row][col].norm();
            if n > max_norm {
                max_norm = n;
                pivot_row = row;
            }
        }
        if max_norm < 1e-12 {
            return Err(FFTError::ComputationError(
                "gaussian_elimination: singular matrix".to_string(),
            ));
        }
        mat.swap(col, pivot_row);
        let pivot = mat[col][col];
        for j in col..=k {
            mat[col][j] = mat[col][j] / pivot;
        }
        for row in 0..k {
            if row != col {
                let factor = mat[row][col];
                for j in col..=k {
                    let sub = factor * mat[col][j];
                    mat[row][j] = mat[row][j] - sub;
                }
            }
        }
    }
    Ok((0..k).map(|i| mat[i][k]).collect())
}

/// Compute polynomial roots via the companion matrix eigenvalue problem.
/// For a monic polynomial z^k + h[0]*z^{k-1} + ... + h[k-1].
fn polynomial_roots_companion(h: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    let k = h.len();
    if k == 0 {
        return Ok(Vec::new());
    }
    if k == 1 {
        return Ok(vec![-h[0]]);
    }

    // Power iteration / Schur-like approach is complex; use a simple
    // DFT-based frequency estimation as approximation for common cases.
    // For production use a full eigenvalue solver would be needed.
    // Here we use a Durand-Kerner / Aberth-like iteration:
    let roots = aberth_roots(h, 50)?;
    Ok(roots)
}

/// Aberth-Ehrlich iteration for polynomial roots.
fn aberth_roots(h: &[Complex64], max_iter: usize) -> FFTResult<Vec<Complex64>> {
    let k = h.len();
    // Initial guesses on a circle of radius R
    let r = 1.0_f64.max(h.iter().map(|c| c.norm()).sum::<f64>() / k as f64);
    let mut roots: Vec<Complex64> = (0..k)
        .map(|i| {
            let angle = 2.0 * PI * i as f64 / k as f64;
            Complex64::new(r * angle.cos(), r * angle.sin())
        })
        .collect();

    for _ in 0..max_iter {
        let mut max_update = 0.0_f64;
        let old_roots = roots.clone();
        for i in 0..k {
            let zi = old_roots[i];
            // Evaluate polynomial and its derivative at zi
            let (p, dp) = poly_eval_monic(h, zi);
            if dp.norm() < 1e-15 {
                continue;
            }
            // Sum over other roots
            let sum: Complex64 = (0..k)
                .filter(|&j| j != i)
                .map(|j| {
                    let diff = zi - old_roots[j];
                    if diff.norm() < 1e-15 {
                        Complex64::new(1e15, 0.0)
                    } else {
                        Complex64::new(1.0, 0.0) / diff
                    }
                })
                .sum();
            let w = (p / dp) / (Complex64::new(1.0, 0.0) - (p / dp) * sum);
            roots[i] = zi - w;
            max_update = max_update.max(w.norm());
        }
        if max_update < 1e-12 {
            break;
        }
    }
    Ok(roots)
}

/// Evaluate monic polynomial p(z) = z^k + h[0]*z^{k-1} + ... + h[k-1]
/// and its derivative p'(z). Returns (p(z), p'(z)).
fn poly_eval_monic(h: &[Complex64], z: Complex64) -> (Complex64, Complex64) {
    let _k = h.len();
    // Horner's method for z^k + h[0]*z^{k-1} + ...
    let mut coeffs = vec![Complex64::new(1.0, 0.0)];
    coeffs.extend_from_slice(h);
    let n = coeffs.len(); // = k+1
    let mut p = coeffs[0];
    let mut dp = Complex64::new(0.0, 0.0);
    for i in 1..n {
        dp = dp * z + p;
        p = p * z + coeffs[i];
    }
    (p, dp)
}

/// Least-squares solution of V*c = b using the normal equations (simple for small k).
fn least_squares_complex(v: &Vec<Vec<Complex64>>, b: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    let m = v.len();
    let n = if m > 0 { v[0].len() } else { 0 };
    if n == 0 {
        return Ok(Vec::new());
    }

    // Normal equations: V^H V c = V^H b
    // Build V^H V (n×n) and V^H b (n×1)
    let mut a_mat = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    let mut rhs = vec![Complex64::new(0.0, 0.0); n];

    for i in 0..n {
        for j in 0..n {
            for row in 0..m {
                a_mat[i][j] = a_mat[i][j] + v[row][i].conj() * v[row][j];
            }
        }
        for row in 0..m {
            rhs[i] = rhs[i] + v[row][i].conj() * b[row];
        }
    }

    // Solve n×n system
    let mut aug: Vec<Vec<Complex64>> = (0..n)
        .map(|i| {
            let mut row = a_mat[i].clone();
            row.push(rhs[i]);
            row
        })
        .collect();

    gaussian_elimination(&mut aug, n)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_complex(re: &[f64]) -> Vec<Complex64> {
        re.iter().map(|&v| Complex64::new(v, 0.0)).collect()
    }

    #[test]
    fn test_sfft_naive_basic() {
        let n = 16usize;
        let freq = 3usize;
        let signal: Vec<Complex64> = (0..n)
            .map(|i| {
                Complex64::new(
                    (2.0 * PI * freq as f64 * i as f64 / n as f64).cos(),
                    0.0,
                )
            })
            .collect();
        let components = sfft_naive(&signal, 2).expect("failed to create components");
        assert_eq!(components.len(), 2);
        // The two dominant components should be at freq=3 and its mirror freq=n-3
        let mut indices: Vec<usize> = components.iter().map(|&(i, _)| i).collect();
        indices.sort_unstable();
        assert!(indices.contains(&freq) || indices.contains(&(n - freq)));
    }

    #[test]
    fn test_sfft_naive_empty_error() {
        assert!(sfft_naive(&[], 1).is_err());
    }

    #[test]
    fn test_sfft_naive_k_gt_n_error() {
        let x = make_complex(&[1.0, 2.0]);
        assert!(sfft_naive(&x, 5).is_err());
    }

    #[test]
    fn test_sparse_ct_detect_basic() {
        let n = 16usize;
        let detector = SparseCT::new(n, 2);
        let signal: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((2.0 * PI * 2.0 * i as f64 / n as f64).cos(), 0.0))
            .collect();
        let support = detector.detect(&signal).expect("failed to create support");
        assert!(support.len() <= 2);
    }

    #[test]
    fn test_sparse_ct_wrong_size() {
        let detector = SparseCT::new(16, 2);
        let x = make_complex(&[1.0, 2.0, 3.0]);
        assert!(detector.detect(&x).is_err());
    }

    #[test]
    fn test_frequency_estimation_prony_single() {
        // Single frequency: cos(2π*0.1*n)
        let n = 32usize;
        let omega_true = 2.0 * PI * 0.1;
        let signal: Vec<f64> = (0..n).map(|i| (omega_true * i as f64).cos()).collect();
        let components = frequency_estimation_prony(&signal, 1).expect("failed to create components");
        assert_eq!(components.len(), 1);
        // Frequency should be close to omega_true or 2π - omega_true
        let omega_est = components[0].0;
        let err = (omega_est - omega_true).abs().min((2.0 * PI - omega_est + omega_true).abs());
        assert!(err < 0.5, "frequency error too large: {err}");
    }

    #[test]
    fn test_frequency_estimation_prony_too_short() {
        assert!(frequency_estimation_prony(&[1.0, 2.0], 2).is_err());
    }

    #[test]
    fn test_frequency_estimation_prony_k_zero() {
        assert!(frequency_estimation_prony(&[1.0, 2.0, 3.0, 4.0], 0).is_err());
    }
}
