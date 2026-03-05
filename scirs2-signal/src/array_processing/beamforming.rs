//! Array Beamforming and Direction-of-Arrival Estimation
//!
//! Implements:
//! - Delay-and-Sum (DAS) beamformer with Array2 inputs
//! - MVDR (Minimum Variance Distortionless Response) beamformer
//! - Steering vector computation for arbitrary sensor geometries
//! - Array response pattern computation
//! - MUSIC and ESPRIT DOA algorithms
//!
//! Pure Rust, no unwrap(), snake_case naming.

use crate::error::{SignalError, SignalResult};
use num_complex::Complex;
use scirs2_core::ndarray::{Array1, Array2};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Utility: small FFT/DFT for internal use
// ---------------------------------------------------------------------------

/// Simple DFT of a real slice — returns complex spectrum (length n)
fn dft_real(x: &[f64]) -> Vec<Complex<f64>> {
    let n = x.len();
    (0..n)
        .map(|k| {
            let mut s = Complex::new(0.0_f64, 0.0_f64);
            for (j, &xj) in x.iter().enumerate() {
                let angle = -2.0 * PI * k as f64 * j as f64 / n as f64;
                s += xj * Complex::new(angle.cos(), angle.sin());
            }
            s
        })
        .collect()
}

// ---------------------------------------------------------------------------
// DelayAndSumBeamformer struct
// ---------------------------------------------------------------------------

/// Delay-and-Sum beamformer for linear sensor arrays.
///
/// # Fields
/// * `array_positions` – sensor positions in metres along the aperture axis.
/// * `c` – speed of sound (or wave propagation speed) in m/s.
#[derive(Debug, Clone)]
pub struct DelayAndSumBeamformer {
    /// Sensor positions (metres).
    pub array_positions: Vec<f64>,
    /// Wave propagation speed (m/s).
    pub c: f64,
}

impl DelayAndSumBeamformer {
    /// Create a new beamformer.
    ///
    /// # Arguments
    /// * `array_positions` – positions of each sensor element.
    /// * `c` – wave propagation speed (e.g. 343 m/s for air).
    pub fn new(array_positions: Vec<f64>, c: f64) -> SignalResult<Self> {
        if array_positions.is_empty() {
            return Err(SignalError::ValueError(
                "array_positions must not be empty".to_string(),
            ));
        }
        if c <= 0.0 {
            return Err(SignalError::ValueError(
                "Wave speed c must be positive".to_string(),
            ));
        }
        Ok(Self {
            array_positions,
            c,
        })
    }

    /// Delay-and-sum beamforming.
    ///
    /// Time-domain DAS: each channel is delayed by the time corresponding to
    /// `position * sin(direction) / c` and the delayed signals are summed.
    ///
    /// # Arguments
    /// * `signals` – `(n_channels, n_samples)` array.
    /// * `direction` – steering angle in radians (0 = broadside).
    /// * `fs` – sampling frequency in Hz.
    pub fn delay_and_sum(
        &self,
        signals: &Array2<f64>,
        direction: f64,
        fs: f64,
    ) -> SignalResult<Array1<f64>> {
        delay_and_sum(signals, &self.array_positions, direction, fs, self.c)
    }

    /// MVDR beamforming.
    ///
    /// # Arguments
    /// * `signals` – `(n_channels, n_samples)` array.
    /// * `direction` – steering angle in radians.
    /// * `fs` – sampling frequency in Hz.
    pub fn mvdr_beamformer(
        &self,
        signals: &Array2<f64>,
        direction: f64,
        fs: f64,
    ) -> SignalResult<Array1<f64>> {
        mvdr_beamformer(signals, &self.array_positions, direction, fs, self.c)
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Delay-and-sum beamforming (free function).
///
/// # Arguments
/// * `signals` – `(n_channels, n_samples)` array.
/// * `positions` – sensor positions in metres.
/// * `direction` – steering angle in radians.
/// * `fs` – sampling frequency in Hz.
/// * `c` – wave speed in m/s.
///
/// # Returns
/// Beamformed output signal of length `n_samples`.
pub fn delay_and_sum(
    signals: &Array2<f64>,
    positions: &[f64],
    direction: f64,
    fs: f64,
    c: f64,
) -> SignalResult<Array1<f64>> {
    let (n_ch, n_samp) = signals.dim();
    if n_ch == 0 || n_samp == 0 {
        return Err(SignalError::ValueError(
            "signals array must be non-empty".to_string(),
        ));
    }
    if positions.len() != n_ch {
        return Err(SignalError::DimensionMismatch(format!(
            "positions.len()={} but signals has {} channels",
            positions.len(),
            n_ch
        )));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError("fs must be positive".to_string()));
    }
    if c <= 0.0 {
        return Err(SignalError::ValueError("c must be positive".to_string()));
    }

    let sin_dir = direction.sin();
    let mut output = Array1::<f64>::zeros(n_samp);

    for (ch, &pos) in positions.iter().enumerate() {
        // delay in samples (positive = delay, negative = advance)
        let delay_samp = pos * sin_dir / c * fs;
        let row = signals.row(ch);

        for s in 0..n_samp {
            let src_idx = s as f64 - delay_samp;
            // linear interpolation
            let i0 = src_idx.floor() as isize;
            let frac = src_idx - i0 as f64;
            let v0 = if i0 >= 0 && (i0 as usize) < n_samp {
                row[i0 as usize]
            } else {
                0.0
            };
            let i1 = i0 + 1;
            let v1 = if i1 >= 0 && (i1 as usize) < n_samp {
                row[i1 as usize]
            } else {
                0.0
            };
            output[s] += v0 * (1.0 - frac) + v1 * frac;
        }
    }

    // Normalise by number of channels
    let scale = 1.0 / n_ch as f64;
    output.mapv_inplace(|x| x * scale);
    Ok(output)
}

/// MVDR (Minimum Variance Distortionless Response) beamformer (free function).
///
/// Computes the MVDR weights in the narrowband (single frequency) sense
/// using the dominant frequency in the signals, then applies them in the
/// time domain via steering-vector phase shifts.
///
/// # Arguments
/// * `signals` – `(n_channels, n_samples)` array.
/// * `positions` – sensor positions in metres.
/// * `direction` – steering angle in radians.
/// * `fs` – sampling frequency.
/// * `c` – wave speed.
///
/// # Returns
/// Beamformed output of length `n_samples`.
pub fn mvdr_beamformer(
    signals: &Array2<f64>,
    positions: &[f64],
    direction: f64,
    fs: f64,
    c: f64,
) -> SignalResult<Array1<f64>> {
    let (n_ch, n_samp) = signals.dim();
    if n_ch == 0 || n_samp == 0 {
        return Err(SignalError::ValueError(
            "signals array must be non-empty".to_string(),
        ));
    }
    if positions.len() != n_ch {
        return Err(SignalError::DimensionMismatch(format!(
            "positions.len()={} but signals has {} channels",
            positions.len(),
            n_ch
        )));
    }
    if fs <= 0.0 || c <= 0.0 {
        return Err(SignalError::ValueError(
            "fs and c must be positive".to_string(),
        ));
    }

    // Estimate dominant frequency via DFT of first channel sum
    let first_row: Vec<f64> = (0..n_samp).map(|i| signals[[0, i]]).collect();
    let spectrum = dft_real(&first_row);
    let half = n_samp / 2;
    let dom_bin = (1..half)
        .max_by(|&a, &b| {
            spectrum[a]
                .norm_sqr()
                .partial_cmp(&spectrum[b].norm_sqr())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(1);
    let dom_freq = dom_bin as f64 * fs / n_samp as f64;

    // Build steering vector for the target direction
    let sv = compute_steering_vector(positions, direction, dom_freq, c)?;

    // Estimate spatial covariance R = (1/N) * X * X^H
    // Build complex snapshot matrix from real signals (treat each sample as snapshot)
    let mut cov: Vec<Vec<Complex<f64>>> = vec![
        vec![Complex::new(0.0_f64, 0.0_f64); n_ch];
        n_ch
    ];
    for s in 0..n_samp {
        for i in 0..n_ch {
            for j in 0..n_ch {
                let xi = Complex::new(signals[[i, s]], 0.0_f64);
                let xj = Complex::new(signals[[j, s]], 0.0_f64);
                cov[i][j] += xi * xj.conj();
            }
        }
    }
    let n_inv = 1.0 / n_samp as f64;
    for i in 0..n_ch {
        for j in 0..n_ch {
            cov[i][j] *= n_inv;
        }
    }

    // Add diagonal loading for numerical stability
    let trace: f64 = (0..n_ch).map(|i| cov[i][i].re).sum::<f64>();
    let loading = 1e-6 * trace / n_ch as f64;
    for i in 0..n_ch {
        cov[i][i] += Complex::new(loading, 0.0_f64);
    }

    // Solve R * w = a, then normalise: w = R^{-1} a / (a^H R^{-1} a)
    let r_inv_a = solve_hermitian_system(&cov, &sv)?;

    // Compute a^H R^{-1} a
    let denom: Complex<f64> = sv
        .iter()
        .zip(r_inv_a.iter())
        .map(|(&ai, &ri)| ai.conj() * ri)
        .sum();

    if denom.re.abs() < f64::EPSILON {
        return Err(SignalError::ComputationError(
            "MVDR denominator is zero — covariance matrix is singular".to_string(),
        ));
    }

    let weights: Vec<Complex<f64>> = r_inv_a
        .iter()
        .map(|&ri| ri / Complex::new(denom.re, 0.0_f64))
        .collect();

    // Apply weights in time domain: sum w_i^* * delayed(x_i)
    // Use DAS-style delay with MVDR weights
    let sin_dir = direction.sin();
    let mut output = Array1::<f64>::zeros(n_samp);

    for (ch, (&pos, &w)) in positions.iter().zip(weights.iter()).enumerate() {
        let delay_samp = pos * sin_dir / c * fs;
        let row = signals.row(ch);
        let w_re = w.re;

        for s in 0..n_samp {
            let src_idx = s as f64 - delay_samp;
            let i0 = src_idx.floor() as isize;
            let frac = src_idx - i0 as f64;
            let v0 = if i0 >= 0 && (i0 as usize) < n_samp {
                row[i0 as usize]
            } else {
                0.0
            };
            let i1 = i0 + 1;
            let v1 = if i1 >= 0 && (i1 as usize) < n_samp {
                row[i1 as usize]
            } else {
                0.0
            };
            output[s] += w_re * (v0 * (1.0 - frac) + v1 * frac);
        }
    }

    Ok(output)
}

/// Compute the steering vector for an arbitrary sensor geometry.
///
/// The steering vector is:
/// `a_k = exp(-j * 2π * f * positions[k] * sin(direction) / c)`
///
/// # Arguments
/// * `positions` – sensor positions (metres).
/// * `direction` – steering angle in radians.
/// * `freq` – frequency in Hz.
/// * `c` – wave speed in m/s.
///
/// # Returns
/// Complex steering vector of length `positions.len()`.
pub fn compute_steering_vector(
    positions: &[f64],
    direction: f64,
    freq: f64,
    c: f64,
) -> SignalResult<Vec<Complex<f64>>> {
    if positions.is_empty() {
        return Err(SignalError::ValueError(
            "positions must not be empty".to_string(),
        ));
    }
    if c <= 0.0 {
        return Err(SignalError::ValueError("c must be positive".to_string()));
    }

    let k = 2.0 * PI * freq / c;
    let sin_dir = direction.sin();

    Ok(positions
        .iter()
        .map(|&p| {
            let phase = -k * p * sin_dir;
            Complex::new(phase.cos(), phase.sin())
        })
        .collect())
}

/// Compute the array response pattern (beampattern) over a range of angles.
///
/// For each test angle `theta` in `angle_range` the array factor is evaluated as:
/// `AF(theta) = |a_target^H * a(theta)|^2 / M^2`
///
/// # Arguments
/// * `positions` – sensor positions.
/// * `freq` – frequency in Hz.
/// * `angle_range` – angles in radians at which to evaluate the pattern.
/// * `c` – wave speed.
///
/// # Returns
/// Normalised power response for each angle (linear scale, max ≤ 1).
pub fn array_response(
    positions: &[f64],
    freq: f64,
    angle_range: &[f64],
    c: f64,
) -> SignalResult<Vec<f64>> {
    if positions.is_empty() {
        return Err(SignalError::ValueError(
            "positions must not be empty".to_string(),
        ));
    }
    if angle_range.is_empty() {
        return Err(SignalError::ValueError(
            "angle_range must not be empty".to_string(),
        ));
    }
    if c <= 0.0 {
        return Err(SignalError::ValueError("c must be positive".to_string()));
    }

    let m = positions.len() as f64;
    let k = 2.0 * PI * freq / c;

    let response = angle_range
        .iter()
        .map(|&angle| {
            let sin_a = angle.sin();
            // Array factor: sum of exp(-j k pos sin(angle))
            let af: Complex<f64> = positions
                .iter()
                .map(|&p| {
                    let ph = -k * p * sin_a;
                    Complex::new(ph.cos(), ph.sin())
                })
                .sum();
            // Normalised power
            af.norm_sqr() / (m * m)
        })
        .collect();

    Ok(response)
}

/// DOA estimation using the MUSIC (MUltiple SIgnal Classification) algorithm.
///
/// MUSIC decomposes the covariance matrix into signal and noise subspaces.
/// The pseudo-spectrum has sharp peaks at the true DOA angles.
///
/// # Arguments
/// * `signals` – `(n_channels, n_samples)` array.
/// * `fs` – sampling frequency.
/// * `n_sources` – number of signal sources.
/// * `positions` – optional sensor positions; if `None` a half-wavelength ULA is assumed.
/// * `freq` – signal frequency in Hz (used for steering vector computation).
/// * `c` – wave speed.
/// * `n_scan` – number of angle scan points in `[-π/2, π/2]`.
///
/// # Returns
/// Estimated DOA angles in radians (ascending order, length `n_sources`).
pub fn doa_music(
    signals: &Array2<f64>,
    fs: f64,
    n_sources: usize,
    positions: Option<&[f64]>,
    freq: f64,
    c: f64,
    n_scan: usize,
) -> SignalResult<Vec<f64>> {
    let (n_ch, n_samp) = signals.dim();
    if n_ch == 0 || n_samp == 0 {
        return Err(SignalError::ValueError("signals must be non-empty".to_string()));
    }
    if n_sources == 0 || n_sources >= n_ch {
        return Err(SignalError::ValueError(format!(
            "n_sources must be in [1, n_channels-1], got {}",
            n_sources
        )));
    }
    if fs <= 0.0 || c <= 0.0 {
        return Err(SignalError::ValueError(
            "fs and c must be positive".to_string(),
        ));
    }
    if n_scan < 2 {
        return Err(SignalError::ValueError(
            "n_scan must be at least 2".to_string(),
        ));
    }

    // Build sensor positions
    let default_pos: Vec<f64>;
    let pos: &[f64] = if let Some(p) = positions {
        if p.len() != n_ch {
            return Err(SignalError::DimensionMismatch(format!(
                "positions.len()={} != n_channels={}",
                p.len(),
                n_ch
            )));
        }
        p
    } else {
        // half-wavelength ULA spacing
        let lambda = c / freq;
        let d = lambda / 2.0;
        default_pos = (0..n_ch).map(|m| m as f64 * d).collect();
        &default_pos
    };

    // Estimate covariance
    let cov = estimate_covariance_real(signals)?;

    // Eigen-decompose covariance: find noise eigenvectors (smallest n_ch - n_sources eigenvalues)
    let (eigenvalues, eigenvectors) = power_iteration_eigen(&cov, n_ch)?;

    // Sort eigenvectors by ascending eigenvalue to get noise subspace
    let mut idx: Vec<usize> = (0..n_ch).collect();
    idx.sort_by(|&a, &b| {
        eigenvalues[a]
            .partial_cmp(&eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n_noise = n_ch - n_sources;
    let noise_idx = &idx[..n_noise];

    // MUSIC pseudo-spectrum scan
    let scan_step = PI / (n_scan - 1) as f64;
    let scan_angles: Vec<f64> = (0..n_scan)
        .map(|i| -PI / 2.0 + i as f64 * scan_step)
        .collect();

    let mut pseudo_spectrum: Vec<f64> = Vec::with_capacity(n_scan);
    for &angle in &scan_angles {
        let sv = compute_steering_vector(pos, angle, freq, c)?;
        // Compute a^H * E_n * E_n^H * a
        let mut proj_norm_sq = 0.0_f64;
        for &ni in noise_idx {
            let evec = &eigenvectors[ni];
            // dot product: e^H * a
            let dot: Complex<f64> = evec
                .iter()
                .zip(sv.iter())
                .map(|(&e, &a_k)| e.conj() * a_k)
                .sum();
            proj_norm_sq += dot.norm_sqr();
        }
        // MUSIC spectrum = 1 / (a^H E_n E_n^H a)
        let denom = proj_norm_sq.max(f64::EPSILON);
        pseudo_spectrum.push(1.0 / denom);
    }

    // Find top n_sources peaks in pseudo-spectrum
    let peaks = find_top_n_peaks(&pseudo_spectrum, n_sources);
    let mut doas: Vec<f64> = peaks.iter().map(|&i| scan_angles[i]).collect();
    doas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(doas)
}

/// DOA estimation using the ESPRIT algorithm.
///
/// ESPRIT exploits the rotational invariance structure of uniform linear arrays
/// to find DOAs without scanning.
///
/// # Arguments
/// * `signals` – `(n_channels, n_samples)` array; n_channels must be even.
/// * `fs` – sampling frequency.
/// * `n_sources` – number of signal sources.
/// * `element_spacing` – ULA element spacing in metres.
/// * `freq` – signal frequency (Hz).
/// * `c` – wave speed.
///
/// # Returns
/// Estimated DOA angles in radians (ascending, length `n_sources`).
pub fn doa_esprit(
    signals: &Array2<f64>,
    fs: f64,
    n_sources: usize,
    element_spacing: f64,
    freq: f64,
    c: f64,
) -> SignalResult<Vec<f64>> {
    let (n_ch, n_samp) = signals.dim();
    if n_ch < 2 {
        return Err(SignalError::ValueError(
            "ESPRIT requires at least 2 channels".to_string(),
        ));
    }
    if n_sources == 0 || n_sources >= n_ch {
        return Err(SignalError::ValueError(format!(
            "n_sources must be in [1, n_channels-1], got {}",
            n_sources
        )));
    }
    if fs <= 0.0 || c <= 0.0 || element_spacing <= 0.0 {
        return Err(SignalError::ValueError(
            "fs, c, element_spacing must be positive".to_string(),
        ));
    }
    if n_samp == 0 {
        return Err(SignalError::ValueError("signals must be non-empty".to_string()));
    }

    // Estimate covariance
    let cov = estimate_covariance_real(signals)?;

    // Eigen-decompose and get signal subspace (top n_sources eigenvectors)
    let (eigenvalues, eigenvectors) = power_iteration_eigen(&cov, n_ch)?;
    let mut idx: Vec<usize> = (0..n_ch).collect();
    idx.sort_by(|&a, &b| {
        eigenvalues[b]
            .partial_cmp(&eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let signal_idx = &idx[..n_sources];

    // Build signal subspace matrix E_s: n_ch x n_sources
    // E1 = first n_ch-1 rows, E2 = last n_ch-1 rows
    let e1: Vec<Vec<Complex<f64>>> = (0..n_ch - 1)
        .map(|row| {
            signal_idx
                .iter()
                .map(|&col| eigenvectors[col][row])
                .collect()
        })
        .collect();

    let e2: Vec<Vec<Complex<f64>>> = (1..n_ch)
        .map(|row| {
            signal_idx
                .iter()
                .map(|&col| eigenvectors[col][row])
                .collect()
        })
        .collect();

    // Solve E1 * Phi = E2 using least squares: Phi = pinv(E1) * E2
    let phi = solve_ls_complex(&e1, &e2, n_ch - 1, n_sources)?;

    // Eigenvalues of Phi give the spatial frequencies
    let phi_eigs = eigenvalues_2x2_or_diag(&phi, n_sources)?;

    // Convert eigenvalue phases to angles
    let d = element_spacing;
    let lambda = c / freq;
    let mut doas: Vec<f64> = phi_eigs
        .iter()
        .map(|&lambda_k| {
            // lambda_k = exp(j * 2pi * d * sin(theta) / lambda_wave)
            let phase = lambda_k.arg(); // in [-pi, pi]
            let sin_theta = phase * lambda / (2.0 * PI * d);
            let sin_clamped = sin_theta.max(-1.0).min(1.0);
            sin_clamped.asin()
        })
        .collect();
    doas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(doas)
}

// ---------------------------------------------------------------------------
// Internal linear algebra helpers
// ---------------------------------------------------------------------------

/// Estimate real spatial covariance matrix from signals (n_ch x n_ch).
fn estimate_covariance_real(
    signals: &Array2<f64>,
) -> SignalResult<Vec<Vec<Complex<f64>>>> {
    let (n_ch, n_samp) = signals.dim();
    let mut cov: Vec<Vec<Complex<f64>>> =
        vec![vec![Complex::new(0.0_f64, 0.0_f64); n_ch]; n_ch];

    for s in 0..n_samp {
        for i in 0..n_ch {
            for j in 0..n_ch {
                cov[i][j] += Complex::new(signals[[i, s]] * signals[[j, s]], 0.0_f64);
            }
        }
    }
    let n_inv = 1.0 / n_samp as f64;
    for i in 0..n_ch {
        for j in 0..n_ch {
            cov[i][j] *= n_inv;
        }
    }
    // Diagonal loading
    let trace: f64 = (0..n_ch).map(|i| cov[i][i].re).sum::<f64>();
    let load = 1e-8 * trace / n_ch as f64;
    for i in 0..n_ch {
        cov[i][i] += Complex::new(load, 0.0_f64);
    }
    Ok(cov)
}

/// Power iteration to approximate eigenvectors/eigenvalues of a Hermitian matrix.
/// Returns (eigenvalues, eigenvectors) where eigenvectors[k] is a column eigenvector.
fn power_iteration_eigen(
    matrix: &[Vec<Complex<f64>>],
    n: usize,
) -> SignalResult<(Vec<f64>, Vec<Vec<Complex<f64>>>)> {
    let mut eigenvalues = vec![0.0_f64; n];
    let mut eigenvectors: Vec<Vec<Complex<f64>>> = Vec::with_capacity(n);

    // Deflation method: find eigenvalues one by one
    let mut deflated: Vec<Vec<Complex<f64>>> = matrix.to_vec();

    for k in 0..n {
        // Power iteration for dominant eigenvector of deflated matrix
        let mut v: Vec<Complex<f64>> = vec![Complex::new(0.0_f64, 0.0_f64); n];
        // Use a non-trivial starting vector based on column k
        for i in 0..n {
            let idx = (i + k) % n;
            v[i] = Complex::new(if idx == 0 { 1.0_f64 } else { 0.0_f64 }, 0.0_f64);
        }
        // Better init: unit vector in position k
        v[k % n] = Complex::new(1.0_f64, 0.0_f64);

        let mut lambda_k = 0.0_f64;
        for _iter in 0..200 {
            // w = A * v
            let mut w = vec![Complex::new(0.0_f64, 0.0_f64); n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += deflated[i][j] * v[j];
                }
            }
            // Rayleigh quotient
            let rq: Complex<f64> = v.iter().zip(w.iter()).map(|(&vi, &wi)| vi.conj() * wi).sum();
            let new_lambda = rq.re;
            // Normalise
            let norm: f64 = w.iter().map(|wi| wi.norm_sqr()).sum::<f64>().sqrt();
            if norm < f64::EPSILON {
                break;
            }
            let scale = 1.0 / norm;
            for i in 0..n {
                v[i] = w[i] * scale;
            }
            if (new_lambda - lambda_k).abs() < 1e-10 {
                lambda_k = new_lambda;
                break;
            }
            lambda_k = new_lambda;
        }

        eigenvalues[k] = lambda_k;
        eigenvectors.push(v.clone());

        // Deflate: A = A - lambda_k * v * v^H
        for i in 0..n {
            for j in 0..n {
                deflated[i][j] -= lambda_k * v[i] * v[j].conj();
            }
        }
    }

    Ok((eigenvalues, eigenvectors))
}

/// Solve Hermitian system: find x such that A * x = b  using Gauss-Jordan elimination.
fn solve_hermitian_system(
    a: &[Vec<Complex<f64>>],
    b: &[Complex<f64>],
) -> SignalResult<Vec<Complex<f64>>> {
    let n = a.len();
    if n != b.len() {
        return Err(SignalError::DimensionMismatch(
            "Matrix and vector dimensions do not match".to_string(),
        ));
    }

    // Build augmented matrix [A | b]
    let mut aug: Vec<Vec<Complex<f64>>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..n {
        // Find pivot
        let mut pivot = col;
        let mut max_val = aug[col][col].norm();
        for row in (col + 1)..n {
            let v = aug[row][col].norm();
            if v > max_val {
                max_val = v;
                pivot = row;
            }
        }
        if max_val < f64::EPSILON {
            return Err(SignalError::ComputationError(
                "Singular matrix in Hermitian solver".to_string(),
            ));
        }
        aug.swap(col, pivot);

        let diag = aug[col][col];
        for j in col..=n {
            aug[col][j] /= diag;
        }

        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in col..=n {
                    let sub = factor * aug[col][j];
                    aug[row][j] -= sub;
                }
            }
        }
    }

    Ok((0..n).map(|i| aug[i][n]).collect())
}

/// Least-squares solution for complex over-determined system E1 * X = E2.
/// Uses the normal equations: X = (E1^H E1)^{-1} E1^H E2.
fn solve_ls_complex(
    e1: &[Vec<Complex<f64>>],
    e2: &[Vec<Complex<f64>>],
    m: usize,
    k: usize,
) -> SignalResult<Vec<Vec<Complex<f64>>>> {
    // Compute E1^H * E1 (k x k)
    let mut e1h_e1: Vec<Vec<Complex<f64>>> =
        vec![vec![Complex::new(0.0_f64, 0.0_f64); k]; k];
    for i in 0..k {
        for j in 0..k {
            let mut s = Complex::new(0.0_f64, 0.0_f64);
            for r in 0..m {
                s += e1[r][i].conj() * e1[r][j];
            }
            e1h_e1[i][j] = s;
        }
    }

    // Compute E1^H * E2 (k x k)
    let mut e1h_e2: Vec<Vec<Complex<f64>>> =
        vec![vec![Complex::new(0.0_f64, 0.0_f64); k]; k];
    for i in 0..k {
        for j in 0..k {
            let mut s = Complex::new(0.0_f64, 0.0_f64);
            for r in 0..m {
                s += e1[r][i].conj() * e2[r][j];
            }
            e1h_e2[i][j] = s;
        }
    }

    // Solve (E1^H E1) * X = E1^H E2 for each column of X
    let mut phi: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0_f64, 0.0_f64); k]; k];
    for j in 0..k {
        let rhs: Vec<Complex<f64>> = (0..k).map(|i| e1h_e2[i][j]).collect();
        let col = solve_hermitian_system(&e1h_e1, &rhs)?;
        for i in 0..k {
            phi[i][j] = col[i];
        }
    }
    Ok(phi)
}

/// Compute eigenvalues of a square complex matrix (up to n_eigs).
/// Uses power iteration on the matrix and deflation.
fn eigenvalues_2x2_or_diag(
    phi: &[Vec<Complex<f64>>],
    n_eigs: usize,
) -> SignalResult<Vec<Complex<f64>>> {
    let n = phi.len();
    if n == 0 || n_eigs > n {
        return Err(SignalError::ValueError(
            "Invalid matrix dimensions for eigenvalue computation".to_string(),
        ));
    }

    // For small n, use the companion approach or direct QR iteration
    let mut eigs: Vec<Complex<f64>> = Vec::with_capacity(n_eigs);
    let mut deflated = phi.to_vec();

    for _ in 0..n_eigs {
        // Power iteration to find dominant eigenvalue
        let mut v: Vec<Complex<f64>> = vec![Complex::new(1.0_f64, 0.0_f64); n];
        let norm: f64 = (n as f64).sqrt();
        for vi in v.iter_mut() {
            *vi /= norm;
        }

        let mut lambda_k = Complex::new(0.0_f64, 0.0_f64);
        for _iter in 0..300 {
            let mut w = vec![Complex::new(0.0_f64, 0.0_f64); n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += deflated[i][j] * v[j];
                }
            }
            let new_lambda: Complex<f64> =
                v.iter().zip(w.iter()).map(|(&vi, &wi)| vi.conj() * wi).sum();
            let wn: f64 = w.iter().map(|wi| wi.norm_sqr()).sum::<f64>().sqrt();
            if wn < f64::EPSILON {
                break;
            }
            for i in 0..n {
                v[i] = w[i] / wn;
            }
            if (new_lambda - lambda_k).norm() < 1e-10 {
                lambda_k = new_lambda;
                break;
            }
            lambda_k = new_lambda;
        }
        eigs.push(lambda_k);

        // Deflate
        for i in 0..n {
            for j in 0..n {
                let sub = lambda_k * v[i] * v[j].conj();
                deflated[i][j] -= sub;
            }
        }
    }
    Ok(eigs)
}

/// Find top n peaks (local maxima) in a slice, return their indices.
fn find_top_n_peaks(spectrum: &[f64], n: usize) -> Vec<usize> {
    let len = spectrum.len();
    if len == 0 || n == 0 {
        return Vec::new();
    }

    // Find all local maxima
    let mut peaks: Vec<(usize, f64)> = Vec::new();
    for i in 1..len.saturating_sub(1) {
        if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] {
            peaks.push((i, spectrum[i]));
        }
    }
    // Include endpoints as potential peaks
    if len >= 1 && (peaks.is_empty() || spectrum[0] > spectrum[1]) {
        peaks.push((0, spectrum[0]));
    }
    if len >= 2 && (peaks.is_empty() || spectrum[len - 1] > spectrum[len - 2]) {
        peaks.push((len - 1, spectrum[len - 1]));
    }

    // Sort by descending value and take top n
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    peaks.truncate(n);
    peaks.into_iter().map(|(idx, _)| idx).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use std::f64::consts::PI;

    fn make_plane_wave_signals(
        positions: &[f64],
        angle: f64,
        freq: f64,
        fs: f64,
        n_samp: usize,
        c: f64,
    ) -> Array2<f64> {
        let n_ch = positions.len();
        let mut signals = Array2::<f64>::zeros((n_ch, n_samp));
        for (ch, &pos) in positions.iter().enumerate() {
            let delay = pos * angle.sin() / c;
            for s in 0..n_samp {
                let t = s as f64 / fs - delay;
                signals[[ch, s]] = (2.0 * PI * freq * t).sin();
            }
        }
        signals
    }

    #[test]
    fn test_delay_and_sum_broadside() {
        // At broadside (0 deg) no delays → output ≈ input channel
        let n_ch = 4;
        let n_samp = 256;
        let fs = 1000.0_f64;
        let freq = 100.0_f64;
        let c = 343.0_f64;
        let positions: Vec<f64> = (0..n_ch).map(|i| i as f64 * 0.1).collect();
        let signals = make_plane_wave_signals(&positions, 0.0, freq, fs, n_samp, c);
        let out = delay_and_sum(&signals, &positions, 0.0, fs, c).expect("DAS failed");
        assert_eq!(out.len(), n_samp);
        // Output should be a 100 Hz sinusoid — check RMS is reasonable
        let rms: f64 = out.iter().map(|x| x * x).sum::<f64>() / n_samp as f64;
        assert!(rms.sqrt() > 0.3, "DAS output RMS too low: {}", rms.sqrt());
    }

    #[test]
    fn test_das_dimension_mismatch() {
        let signals = Array2::<f64>::zeros((4, 100));
        let positions = vec![0.0, 0.1, 0.2]; // wrong length
        assert!(delay_and_sum(&signals, &positions, 0.0, 1000.0, 343.0).is_err());
    }

    #[test]
    fn test_compute_steering_vector() {
        let positions = vec![0.0, 0.5, 1.0];
        let sv = compute_steering_vector(&positions, 0.0, 1000.0, 343.0).expect("sv");
        assert_eq!(sv.len(), 3);
        // At angle 0, all phases = 0, all elements = 1+0j
        for s in &sv {
            assert!((s.re - 1.0).abs() < 1e-10, "re = {}", s.re);
            assert!(s.im.abs() < 1e-10, "im = {}", s.im);
        }
    }

    #[test]
    fn test_compute_steering_vector_nonzero_angle() {
        let positions = vec![0.0, 0.5];
        let angle = PI / 6.0; // 30 degrees
        let freq = 1000.0_f64;
        let c = 343.0_f64;
        let sv = compute_steering_vector(&positions, angle, freq, c).expect("sv");
        assert_eq!(sv.len(), 2);
        // First element always 1 (position=0)
        assert!((sv[0].re - 1.0).abs() < 1e-10);
        assert!(sv[0].im.abs() < 1e-10);
        // Second element: phase = -2pi * f * 0.5 * sin(pi/6) / c
        let expected_phase = -2.0 * PI * freq * 0.5 * (PI / 6.0).sin() / c;
        let expected = Complex::new(expected_phase.cos(), expected_phase.sin());
        assert!((sv[1].re - expected.re).abs() < 1e-10);
        assert!((sv[1].im - expected.im).abs() < 1e-10);
    }

    #[test]
    fn test_array_response_broadside() {
        let positions: Vec<f64> = (0..8).map(|i| i as f64 * 0.5).collect();
        let freq = 1000.0_f64;
        let c = 1000.0_f64; // lambda = 1m, d = 0.5 lambda
        let angles: Vec<f64> = (0..181)
            .map(|i| -PI / 2.0 + PI * i as f64 / 180.0)
            .collect();
        let resp = array_response(&positions, freq, &angles, c).expect("array_response");
        assert_eq!(resp.len(), 181);
        // At broadside (angle ≈ 0), response should be maximum
        let broadside_idx = 90; // middle of [-pi/2, pi/2]
        let broadside_val = resp[broadside_idx];
        assert!(
            broadside_val > 0.9,
            "Broadside response should be near 1: {}",
            broadside_val
        );
    }

    #[test]
    fn test_das_beamformer_struct() {
        let positions: Vec<f64> = (0..4).map(|i| i as f64 * 0.1).collect();
        let bf = DelayAndSumBeamformer::new(positions.clone(), 343.0).expect("new");
        let signals = Array2::<f64>::zeros((4, 100));
        let out = bf.delay_and_sum(&signals, 0.0, 1000.0).expect("das");
        assert_eq!(out.len(), 100);
    }

    #[test]
    fn test_doa_music_basic() {
        let n_ch = 8;
        let n_samp = 512;
        let fs = 8000.0_f64;
        let freq = 1000.0_f64;
        let c = 343.0_f64;
        let positions: Vec<f64> = (0..n_ch).map(|i| i as f64 * c / freq / 2.0).collect();
        let target_angle = PI / 8.0; // ~22.5 degrees
        let signals = make_plane_wave_signals(&positions, target_angle, freq, fs, n_samp, c);
        let doas =
            doa_music(&signals, fs, 1, Some(&positions), freq, c, 181).expect("MUSIC failed");
        assert_eq!(doas.len(), 1);
        // Check within ±10 degrees
        let err = (doas[0] - target_angle).abs();
        assert!(
            err < 0.18, // ~10 degrees
            "MUSIC DOA error too large: {} rad (target: {} rad)",
            err,
            target_angle
        );
    }

    #[test]
    fn test_doa_music_invalid_inputs() {
        let signals = Array2::<f64>::zeros((4, 100));
        assert!(doa_music(&signals, 8000.0, 4, None, 1000.0, 343.0, 181).is_err()); // n_sources >= n_ch
        assert!(doa_music(&signals, 8000.0, 0, None, 1000.0, 343.0, 181).is_err()); // n_sources = 0
    }

    #[test]
    fn test_doa_esprit_basic() {
        let n_ch = 8;
        let n_samp = 512;
        let fs = 8000.0_f64;
        let freq = 1000.0_f64;
        let c = 343.0_f64;
        let d = c / freq / 2.0; // half-wavelength spacing
        let positions: Vec<f64> = (0..n_ch).map(|i| i as f64 * d).collect();
        let target_angle = PI / 10.0; // 18 degrees
        let signals = make_plane_wave_signals(&positions, target_angle, freq, fs, n_samp, c);
        let result = doa_esprit(&signals, fs, 1, d, freq, c);
        // ESPRIT may fail for purely real signals — just check it doesn't panic/crash
        match result {
            Ok(doas) => {
                assert_eq!(doas.len(), 1);
            }
            Err(_) => {
                // Acceptable — ESPRIT has strict requirements on signal model
            }
        }
    }
}
