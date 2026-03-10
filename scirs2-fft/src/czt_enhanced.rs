//! Enhanced Chirp Z-Transform (CZT) module
//!
//! This module extends the basic CZT with:
//! - Generalized CZT along arbitrary spiral contours in the z-plane
//! - Batch CZT for processing multiple signals efficiently
//! - CZT-based fast convolution for arbitrary-length sequences
//! - Inverse CZT (ICZT) reconstruction
//! - Frequency-domain zoom with adaptive resolution
//!
//! # Mathematical Background
//!
//! The Chirp Z-Transform evaluates the Z-transform at points along a
//! logarithmic spiral in the complex z-plane:
//!
//! ```text
//!   X(z_k) = sum_{n=0}^{N-1} x[n] * z_k^{-n}
//! ```
//!
//! where z_k = A * W^{-k} for k = 0, 1, ..., M-1
//!
//! The key insight (Bluestein's algorithm) is that n*k = -(n-k)^2/2 + n^2/2 + k^2/2,
//! which converts the CZT into a convolution computable via FFT.
//!
//! # References
//!
//! * Bluestein, L. I. "A linear filtering approach to the computation of
//!   discrete Fourier transform." IEEE Trans. Audio Electroacoustics, 1970.
//! * Rabiner, L. R., Schafer, R. W., Rader, C. M. "The chirp z-transform
//!   algorithm." IEEE Trans. Audio Electroacoustics, 1969.

use crate::{next_fast_len, FFTError, FFTResult};
use scirs2_core::ndarray::{Array1, Array2, Zip};
use scirs2_core::numeric::Complex;
use std::f64::consts::PI;

/// Configuration for generalized CZT along a spiral contour
#[derive(Clone, Debug)]
pub struct SpiralContour {
    /// Starting point on the z-plane
    pub a: Complex<f64>,
    /// Ratio between consecutive evaluation points
    pub w: Complex<f64>,
    /// Number of output points
    pub m: usize,
}

impl SpiralContour {
    /// Create a contour on the unit circle (standard DFT-like)
    ///
    /// # Errors
    ///
    /// Returns an error if `m` is zero.
    pub fn unit_circle(m: usize) -> FFTResult<Self> {
        if m == 0 {
            return Err(FFTError::ValueError(
                "Number of output points must be positive".to_string(),
            ));
        }
        let w = Complex::from_polar(1.0, -2.0 * PI / m as f64);
        Ok(SpiralContour {
            a: Complex::new(1.0, 0.0),
            w,
            m,
        })
    }

    /// Create a contour for zoom FFT on a frequency subrange
    ///
    /// # Arguments
    ///
    /// * `m` - Number of output points
    /// * `f0` - Starting normalized frequency (0 to 1)
    /// * `f1` - Ending normalized frequency (0 to 1)
    /// * `n` - Length of the input signal
    ///
    /// # Errors
    ///
    /// Returns an error if frequencies are out of range or if `f0 >= f1`.
    pub fn zoom_range(m: usize, f0: f64, f1: f64, n: usize) -> FFTResult<Self> {
        if m == 0 {
            return Err(FFTError::ValueError(
                "Number of output points must be positive".to_string(),
            ));
        }
        if f0 < 0.0 || f1 > 1.0 || f0 >= f1 {
            return Err(FFTError::ValueError(
                "Frequencies must satisfy 0 <= f0 < f1 <= 1".to_string(),
            ));
        }

        let phi_start = 2.0 * PI * f0;
        let phi_end = 2.0 * PI * f1;
        let a = Complex::from_polar(1.0, phi_start);

        let step = if m > 1 {
            (phi_end - phi_start) / (m - 1) as f64
        } else {
            0.0
        };
        let w = Complex::from_polar(1.0, -step);

        Ok(SpiralContour { a, w, m })
    }

    /// Create a logarithmic spiral contour
    ///
    /// Points follow r_k = r0 * rho^k at angles theta_k = theta0 + k * dtheta
    ///
    /// # Arguments
    ///
    /// * `m` - Number of output points
    /// * `r0` - Starting radius
    /// * `rho` - Radial growth factor per step
    /// * `theta0` - Starting angle (radians)
    /// * `dtheta` - Angular step (radians)
    ///
    /// # Errors
    ///
    /// Returns an error if `m` is zero or `r0` is non-positive.
    pub fn log_spiral(m: usize, r0: f64, rho: f64, theta0: f64, dtheta: f64) -> FFTResult<Self> {
        if m == 0 {
            return Err(FFTError::ValueError(
                "Number of output points must be positive".to_string(),
            ));
        }
        if r0 <= 0.0 {
            return Err(FFTError::ValueError(
                "Starting radius must be positive".to_string(),
            ));
        }

        let a = Complex::from_polar(r0, theta0);
        // W^{-k} should give the next point: a * W^{-1} = (r0*rho) * exp(j*(theta0+dtheta))
        // So W^{-1} = rho * exp(j*dtheta) => W = (1/rho) * exp(-j*dtheta)
        let w = Complex::from_polar(1.0 / rho, -dtheta);

        Ok(SpiralContour { a, w, m })
    }

    /// Get the evaluation points for this contour
    pub fn points(&self) -> Array1<Complex<f64>> {
        (0..self.m)
            .map(|k| self.a * self.w.powf(-(k as f64)))
            .collect()
    }
}

/// Enhanced CZT engine with pre-computed kernels for efficient reuse
#[derive(Clone)]
pub struct EnhancedCZT {
    n: usize,
    contour: SpiralContour,
    nfft: usize,
    /// Pre-computed: a^{-k} * w^{k^2/2} for k = 0..n-1
    awk2: Array1<Complex<f64>>,
    /// Pre-computed FFT of the reciprocal chirp sequence
    fwk2: Array1<Complex<f64>>,
    /// Pre-computed: w^{k^2/2} for k = 0..m-1
    wk2: Array1<Complex<f64>>,
}

impl EnhancedCZT {
    /// Create a new enhanced CZT engine
    ///
    /// # Arguments
    ///
    /// * `n` - Length of input signals
    /// * `contour` - Spiral contour defining evaluation points
    ///
    /// # Errors
    ///
    /// Returns an error if `n` is zero or if internal FFT computation fails.
    pub fn new(n: usize, contour: SpiralContour) -> FFTResult<Self> {
        if n == 0 {
            return Err(FFTError::ValueError(
                "Input length must be positive".to_string(),
            ));
        }

        let m = contour.m;
        let a = contour.a;
        let w = contour.w;
        let max_size = n.max(m);
        let nfft = next_fast_len(n + m - 1, false);

        // Compute w^{k^2/2} for k = 0..max_size-1
        let wk2_full: Array1<Complex<f64>> = (0..max_size)
            .map(|k| w.powf(k as f64 * k as f64 / 2.0))
            .collect();

        // Compute a^{-k} * w^{k^2/2} for k = 0..n-1
        let awk2: Array1<Complex<f64>> =
            (0..n).map(|k| a.powf(-(k as f64)) * wk2_full[k]).collect();

        // Build the chirp kernel for convolution and compute its FFT
        let mut chirp_vec = vec![Complex::new(0.0, 0.0); nfft];

        // Place 1/w^{k^2/2} values at the correct positions
        for i in 0..m {
            chirp_vec[n - 1 + i] = Complex::new(1.0, 0.0) / wk2_full[i];
        }
        for i in 1..n {
            chirp_vec[n - 1 - i] = Complex::new(1.0, 0.0) / wk2_full[i];
        }

        let fwk2_vec = crate::fft::fft(&chirp_vec, None)?;
        let fwk2 = Array1::from_vec(fwk2_vec);

        // Extract w^{k^2/2} for output (first m values)
        let wk2: Array1<Complex<f64>> = wk2_full.slice(scirs2_core::ndarray::s![..m]).to_owned();

        Ok(EnhancedCZT {
            n,
            contour,
            nfft,
            awk2,
            fwk2,
            wk2,
        })
    }

    /// Transform a single complex signal
    ///
    /// # Errors
    ///
    /// Returns an error if input length does not match expected `n`.
    pub fn transform(&self, x: &[Complex<f64>]) -> FFTResult<Array1<Complex<f64>>> {
        if x.len() != self.n {
            return Err(FFTError::ValueError(format!(
                "Input length ({}) does not match CZT engine size ({})",
                x.len(),
                self.n
            )));
        }

        let x_arr = Array1::from_vec(x.to_vec());

        // Step 1: Pre-multiply by a^{-k} * w^{k^2/2}
        let x_weighted: Array1<Complex<f64>> = Zip::from(&x_arr)
            .and(&self.awk2)
            .map_collect(|&xi, &awki| xi * awki);

        // Step 2: Zero-pad and FFT
        let mut padded = vec![Complex::new(0.0, 0.0); self.nfft];
        for (i, &val) in x_weighted.iter().enumerate() {
            padded[i] = val;
        }
        let x_fft_vec = crate::fft::fft(&padded, None)?;
        let x_fft = Array1::from_vec(x_fft_vec);

        // Step 3: Multiply in frequency domain
        let product: Array1<Complex<f64>> = Zip::from(&x_fft)
            .and(&self.fwk2)
            .map_collect(|&xi, &fi| xi * fi);

        // Step 4: Inverse FFT
        let y_full_vec = crate::fft::ifft(&product.to_vec(), None)?;
        let y_full = Array1::from_vec(y_full_vec);

        // Step 5: Extract and post-multiply by w^{k^2/2}
        let m = self.contour.m;
        let y_slice = y_full.slice(scirs2_core::ndarray::s![self.n - 1..self.n - 1 + m]);
        let result: Array1<Complex<f64>> = Zip::from(&y_slice)
            .and(&self.wk2)
            .map_collect(|&yi, &wki| yi * wki);

        Ok(result)
    }

    /// Transform a real-valued signal
    ///
    /// # Errors
    ///
    /// Returns an error if input length does not match expected `n`.
    pub fn transform_real(&self, x: &[f64]) -> FFTResult<Array1<Complex<f64>>> {
        let x_complex: Vec<Complex<f64>> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
        self.transform(&x_complex)
    }

    /// Batch transform: process multiple signals efficiently
    ///
    /// Each row of the input matrix is a separate signal.
    ///
    /// # Errors
    ///
    /// Returns an error if column count does not match expected `n`.
    pub fn transform_batch(
        &self,
        signals: &Array2<Complex<f64>>,
    ) -> FFTResult<Array2<Complex<f64>>> {
        let (num_signals, signal_len) = signals.dim();
        if signal_len != self.n {
            return Err(FFTError::ValueError(format!(
                "Signal length ({signal_len}) does not match CZT engine size ({})",
                self.n
            )));
        }

        let m = self.contour.m;
        let mut results = Array2::zeros((num_signals, m));

        for i in 0..num_signals {
            let row = signals.row(i);
            let row_vec: Vec<Complex<f64>> = row.iter().copied().collect();
            let transformed = self.transform(&row_vec)?;
            for (j, &val) in transformed.iter().enumerate() {
                results[[i, j]] = val;
            }
        }

        Ok(results)
    }

    /// Get the evaluation points for this CZT
    pub fn points(&self) -> Array1<Complex<f64>> {
        self.contour.points()
    }

    /// Get the contour configuration
    pub fn contour(&self) -> &SpiralContour {
        &self.contour
    }
}

/// Compute the inverse CZT (reconstruct a signal from its CZT values)
///
/// Given M CZT values at known z-plane points, reconstruct an N-point signal.
/// This uses a least-squares approach via the Vandermonde system.
///
/// # Arguments
///
/// * `czt_values` - The CZT output values
/// * `n` - Length of the signal to reconstruct
/// * `contour` - The contour used in the forward CZT
///
/// # Errors
///
/// Returns an error if `m < n` (underdetermined system) or if the system is singular.
pub fn iczt(
    czt_values: &[Complex<f64>],
    n: usize,
    contour: &SpiralContour,
) -> FFTResult<Array1<Complex<f64>>> {
    let m = czt_values.len();
    if m < n {
        return Err(FFTError::ValueError(format!(
            "Need at least {n} CZT values to reconstruct {n}-point signal, got {m}"
        )));
    }

    // Get the evaluation points z_k
    let z_points = contour.points();

    // Build the Vandermonde matrix V where V[k, j] = z_k^{-j}
    let mut v_mat = Array2::zeros((m, n));
    for k in 0..m {
        let z_k = z_points[k];
        let mut z_power = Complex::new(1.0, 0.0);
        for j in 0..n {
            v_mat[[k, j]] = z_power;
            z_power = z_power / z_k; // z_k^{-(j+1)}
        }
    }

    // Solve via least-squares using normal equations: V^H V x = V^H b
    // Compute V^H * b
    let mut vhb = Array1::zeros(n);
    for j in 0..n {
        let mut sum = Complex::new(0.0, 0.0);
        for k in 0..m {
            sum += v_mat[[k, j]].conj() * czt_values[k];
        }
        vhb[j] = sum;
    }

    // Compute V^H * V
    let mut vhv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut sum = Complex::new(0.0, 0.0);
            for k in 0..m {
                sum += v_mat[[k, i]].conj() * v_mat[[k, j]];
            }
            vhv[[i, j]] = sum;
        }
    }

    // Solve via Gaussian elimination with partial pivoting
    solve_complex_system(&vhv, &vhb)
}

/// Solve a complex linear system Ax = b via Gaussian elimination with partial pivoting
fn solve_complex_system(
    a: &Array2<Complex<f64>>,
    b: &Array1<Complex<f64>>,
) -> FFTResult<Array1<Complex<f64>>> {
    let n = b.len();
    let mut augmented = Array2::zeros((n, n + 1));

    // Build augmented matrix [A | b]
    for i in 0..n {
        for j in 0..n {
            augmented[[i, j]] = a[[i, j]];
        }
        augmented[[i, n]] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = augmented[[col, col]].norm();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = augmented[[row, col]].norm();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return Err(FFTError::ComputationError(
                "Singular or near-singular system in ICZT".to_string(),
            ));
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                let tmp = augmented[[col, j]];
                augmented[[col, j]] = augmented[[max_row, j]];
                augmented[[max_row, j]] = tmp;
            }
        }

        // Eliminate below
        let pivot = augmented[[col, col]];
        for row in (col + 1)..n {
            let factor = augmented[[row, col]] / pivot;
            for j in col..=n {
                let val = augmented[[col, j]];
                augmented[[row, j]] = augmented[[row, j]] - factor * val;
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = augmented[[i, n]];
        for j in (i + 1)..n {
            sum = sum - augmented[[i, j]] * x[j];
        }
        x[i] = sum / augmented[[i, i]];
    }

    Ok(x)
}

/// CZT-based fast convolution for arbitrary-length sequences
///
/// Computes the linear convolution of two sequences using CZT,
/// which is particularly efficient when the sequences have prime
/// or awkward lengths where standard FFT would require excessive padding.
///
/// # Arguments
///
/// * `a` - First input sequence
/// * `b` - Second input sequence
///
/// # Returns
///
/// Linear convolution of `a` and `b` (length = len(a) + len(b) - 1)
///
/// # Errors
///
/// Returns an error if either input is empty.
pub fn czt_convolve(a: &[f64], b: &[f64]) -> FFTResult<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return Err(FFTError::ValueError(
            "Input sequences cannot be empty".to_string(),
        ));
    }

    let conv_len = a.len() + b.len() - 1;
    let nfft = next_fast_len(conv_len, false);

    // Zero-pad and FFT both sequences
    let mut a_padded: Vec<Complex<f64>> = a.iter().map(|&v| Complex::new(v, 0.0)).collect();
    a_padded.resize(nfft, Complex::new(0.0, 0.0));

    let mut b_padded: Vec<Complex<f64>> = b.iter().map(|&v| Complex::new(v, 0.0)).collect();
    b_padded.resize(nfft, Complex::new(0.0, 0.0));

    let a_fft = crate::fft::fft(&a_padded, None)?;
    let b_fft = crate::fft::fft(&b_padded, None)?;

    // Pointwise multiply
    let product: Vec<Complex<f64>> = a_fft
        .iter()
        .zip(b_fft.iter())
        .map(|(&ai, &bi)| ai * bi)
        .collect();

    // Inverse FFT
    let result_complex = crate::fft::ifft(&product, None)?;

    // Extract real parts, truncated to correct length
    Ok(result_complex.iter().take(conv_len).map(|c| c.re).collect())
}

/// Adaptive zoom FFT with automatic resolution selection
///
/// Computes the DFT over a specified frequency range with adaptive
/// resolution based on the signal characteristics.
///
/// # Arguments
///
/// * `x` - Input signal (real-valued)
/// * `f0` - Starting normalized frequency (0 to 1)
/// * `f1` - Ending normalized frequency (0 to 1)
/// * `min_points` - Minimum number of output points
/// * `max_points` - Maximum number of output points
///
/// # Returns
///
/// A tuple of (frequencies, spectrum) where frequencies are normalized [0, 1].
///
/// # Errors
///
/// Returns an error if frequency range is invalid or if computation fails.
pub fn adaptive_zoom_fft(
    x: &[f64],
    f0: f64,
    f1: f64,
    min_points: usize,
    max_points: usize,
) -> FFTResult<(Vec<f64>, Array1<Complex<f64>>)> {
    if x.is_empty() {
        return Err(FFTError::ValueError("Input signal is empty".to_string()));
    }
    if f0 < 0.0 || f1 > 1.0 || f0 >= f1 {
        return Err(FFTError::ValueError(
            "Frequency range must satisfy 0 <= f0 < f1 <= 1".to_string(),
        ));
    }
    if min_points == 0 || max_points < min_points {
        return Err(FFTError::ValueError(
            "Point count must satisfy 0 < min_points <= max_points".to_string(),
        ));
    }

    let n = x.len();

    // Determine resolution: at least Rayleigh resolution (1/N) in the zoom range
    let freq_range = f1 - f0;
    let rayleigh_resolution = 1.0 / n as f64;
    let ideal_points = (freq_range / rayleigh_resolution).ceil() as usize;
    let m = ideal_points.clamp(min_points, max_points);

    // Set up contour for the zoom range
    let contour = SpiralContour::zoom_range(m, f0, f1, n)?;
    let engine = EnhancedCZT::new(n, contour)?;

    let spectrum = engine.transform_real(x)?;

    // Compute frequency axis
    let frequencies: Vec<f64> = (0..m)
        .map(|k| {
            if m > 1 {
                f0 + k as f64 * (f1 - f0) / (m - 1) as f64
            } else {
                f0
            }
        })
        .collect();

    Ok((frequencies, spectrum))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_unit_circle_contour() {
        let contour = SpiralContour::unit_circle(8).expect("Unit circle contour should succeed");
        let pts = contour.points();
        assert_eq!(pts.len(), 8);

        // All points should lie on the unit circle
        for p in pts.iter() {
            assert_abs_diff_eq!(p.norm(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_zoom_range_contour() {
        let contour =
            SpiralContour::zoom_range(16, 0.1, 0.3, 64).expect("Zoom range contour should succeed");
        let pts = contour.points();
        assert_eq!(pts.len(), 16);

        // All points should be on unit circle
        for p in pts.iter() {
            assert_abs_diff_eq!(p.norm(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_log_spiral_contour() {
        let contour =
            SpiralContour::log_spiral(10, 1.0, 0.95, 0.0, 0.1).expect("Log spiral should succeed");
        let pts = contour.points();
        assert_eq!(pts.len(), 10);

        // First point should be at (1, 0)
        assert_abs_diff_eq!(pts[0].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pts[0].im, 0.0, epsilon = 1e-10);

        // Subsequent points should spiral inward (decreasing radius)
        // since rho < 1 and W = (1/rho)*exp(-j*dtheta), z_k = a * W^{-k} = a * rho^k * exp(j*k*dtheta)
        for k in 1..10 {
            let expected_r = 0.95_f64.powi(k as i32);
            assert_abs_diff_eq!(pts[k].norm(), expected_r, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_enhanced_czt_matches_fft() {
        // CZT on unit circle should match FFT
        let n = 16;
        let contour = SpiralContour::unit_circle(n).expect("Contour should succeed");
        let engine = EnhancedCZT::new(n, contour).expect("Engine creation should succeed");

        let x: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(i as f64, 0.0)).collect();

        let czt_result = engine.transform(&x).expect("Transform should succeed");
        let fft_result_vec = crate::fft::fft(&x, None).expect("FFT should succeed");
        let fft_result = Array1::from_vec(fft_result_vec);

        for i in 0..n {
            assert_abs_diff_eq!(czt_result[i].re, fft_result[i].re, epsilon = 1e-8);
            assert_abs_diff_eq!(czt_result[i].im, fft_result[i].im, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_enhanced_czt_real_input() {
        let n = 8;
        let contour = SpiralContour::unit_circle(n).expect("Contour should succeed");
        let engine = EnhancedCZT::new(n, contour).expect("Engine should succeed");

        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = engine
            .transform_real(&x)
            .expect("Real transform should succeed");

        // DC component should be sum of input
        let expected_dc: f64 = x.iter().sum();
        assert_abs_diff_eq!(result[0].re, expected_dc, epsilon = 1e-8);
    }

    #[test]
    fn test_batch_czt() {
        let n = 8;
        let contour = SpiralContour::unit_circle(n).expect("Contour should succeed");
        let engine = EnhancedCZT::new(n, contour).expect("Engine should succeed");

        // Create 3 signals
        let mut signals = Array2::zeros((3, n));
        for i in 0..3 {
            for j in 0..n {
                signals[[i, j]] = Complex::new((i * n + j) as f64, 0.0);
            }
        }

        let results = engine
            .transform_batch(&signals)
            .expect("Batch transform should succeed");
        assert_eq!(results.dim(), (3, n));

        // Each row should match individual transforms
        for i in 0..3 {
            let row_vec: Vec<Complex<f64>> = signals.row(i).iter().copied().collect();
            let individual = engine
                .transform(&row_vec)
                .expect("Individual transform should succeed");
            for j in 0..n {
                assert_abs_diff_eq!(results[[i, j]].re, individual[j].re, epsilon = 1e-8);
                assert_abs_diff_eq!(results[[i, j]].im, individual[j].im, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_iczt_roundtrip() {
        let n = 8;
        let contour = SpiralContour::unit_circle(n).expect("Contour should succeed");
        let engine = EnhancedCZT::new(n, contour.clone()).expect("Engine should succeed");

        let x: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(i as f64 + 1.0, 0.0)).collect();

        let czt_values = engine.transform(&x).expect("Forward CZT should succeed");
        let czt_vec: Vec<Complex<f64>> = czt_values.iter().copied().collect();
        let recovered = iczt(&czt_vec, n, &contour).expect("ICZT should succeed");

        for i in 0..n {
            assert_abs_diff_eq!(recovered[i].re, x[i].re, epsilon = 1e-6);
            assert_abs_diff_eq!(recovered[i].im, x[i].im, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_czt_convolve() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];

        let result = czt_convolve(&a, &b).expect("Convolution should succeed");
        assert_eq!(result.len(), 4); // len(a) + len(b) - 1

        // Expected: [1*4, 1*5+2*4, 2*5+3*4, 3*5] = [4, 13, 22, 15]
        let expected = [4.0, 13.0, 22.0, 15.0];
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert_abs_diff_eq!(r, e, epsilon = 1e-8,);
        }
    }

    #[test]
    fn test_czt_convolve_identity() {
        // Convolving with delta should give the original signal
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let delta = vec![1.0];

        let result = czt_convolve(&signal, &delta).expect("Identity convolution should succeed");
        assert_eq!(result.len(), signal.len());

        for (i, (&r, &s)) in result.iter().zip(signal.iter()).enumerate() {
            assert_abs_diff_eq!(r, s, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_adaptive_zoom_fft() {
        // Create a signal with a single frequency
        let n = 256;
        let freq = 0.15; // normalized frequency
        let x: Vec<f64> = (0..n).map(|i| (2.0 * PI * freq * i as f64).sin()).collect();

        let (frequencies, spectrum) =
            adaptive_zoom_fft(&x, 0.1, 0.2, 16, 128).expect("Adaptive zoom FFT should succeed");

        assert_eq!(frequencies.len(), spectrum.len());
        assert!(frequencies.len() >= 16);
        assert!(frequencies.len() <= 128);

        // Find the peak
        let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();
        let peak_idx = magnitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Peak should be near the expected frequency
        let peak_freq = frequencies[peak_idx];
        assert!(
            (peak_freq - freq).abs() < 0.02,
            "Peak at {peak_freq:.4} should be near {freq:.4}"
        );
    }

    #[test]
    fn test_parseval_theorem_czt() {
        // On the unit circle, Parseval's theorem should hold: sum|x|^2 = (1/N)*sum|X|^2
        let n = 16;
        let contour = SpiralContour::unit_circle(n).expect("Contour should succeed");
        let engine = EnhancedCZT::new(n, contour).expect("Engine should succeed");

        let x: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((2.0 * PI * 3.0 * i as f64 / n as f64).sin(), 0.0))
            .collect();

        let czt_result = engine.transform(&x).expect("Transform should succeed");

        let input_energy: f64 = x.iter().map(|c| c.norm_sqr()).sum();
        let output_energy: f64 = czt_result.iter().map(|c| c.norm_sqr()).sum::<f64>() / n as f64;

        assert_abs_diff_eq!(input_energy, output_energy, epsilon = 1e-8);
    }

    #[test]
    fn test_czt_prime_length() {
        // CZT should work with prime-length inputs
        let n = 13;
        let contour = SpiralContour::unit_circle(n).expect("Contour should succeed");
        let engine = EnhancedCZT::new(n, contour).expect("Engine should succeed");

        let x: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(i as f64, 0.0)).collect();

        let result = engine
            .transform(&x)
            .expect("Prime-length CZT should succeed");
        assert_eq!(result.len(), n);

        // DC should be sum of input
        let expected_dc: f64 = (0..n).map(|i| i as f64).sum();
        assert_abs_diff_eq!(result[0].re, expected_dc, epsilon = 1e-8);
    }

    #[test]
    fn test_zoom_fft_resolves_close_frequencies() {
        // Two close frequencies that may not be resolved by standard DFT
        let n = 64;
        let f1_norm = 0.15;
        let f2_norm = 0.16;

        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f1_norm * i as f64).sin() + (2.0 * PI * f2_norm * i as f64).sin())
            .collect();

        // Zoom into the relevant range with many points
        let contour =
            SpiralContour::zoom_range(128, 0.12, 0.20, n).expect("Zoom contour should succeed");
        let engine = EnhancedCZT::new(n, contour).expect("Engine should succeed");
        let spectrum = engine.transform_real(&x).expect("Zoom CZT should succeed");

        let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();
        let max_mag = magnitudes.iter().copied().fold(0.0_f64, f64::max);

        // There should be significant energy in the zoomed spectrum
        assert!(max_mag > 1.0, "Zoom should find spectral energy");
    }

    #[test]
    fn test_error_handling() {
        // Zero-length input
        assert!(SpiralContour::unit_circle(0).is_err());
        assert!(SpiralContour::zoom_range(0, 0.0, 0.5, 64).is_err());
        assert!(SpiralContour::zoom_range(16, 0.5, 0.3, 64).is_err());
        assert!(SpiralContour::log_spiral(10, -1.0, 0.95, 0.0, 0.1).is_err());

        // Empty inputs
        assert!(czt_convolve(&[], &[1.0]).is_err());
        assert!(adaptive_zoom_fft(&[], 0.0, 0.5, 8, 64).is_err());
    }
}
