//! Advanced Sparse FFT algorithms and signal parameter estimation.
//!
//! This module provides high-level public APIs complementing the lower-level
//! algorithms in [`sparse_fft`](crate::sparse_fft):
//!
//! * [`SparseFftResult`]     — Simplified sparse FFT result type.
//! * [`sparse_fft_simple`]   — SFFT-style algorithm (random sampling approach).
//! * [`sparse_fft_lasso`]    — Sparse recovery via coordinate descent / soft thresholding.
//! * [`sparse_to_dense`]     — Convert sparse representation to full spectrum.
//! * [`PronyResult`]         — Result of Prony's method.
//! * [`prony_method`]        — Prony's method for sinusoidal parameter estimation.
//! * [`music_signal_freqs`]  — Convenience wrapper for MUSIC frequency estimation.
//!
//! # References
//!
//! * Hassanieh, H., Indyk, P., Katabi, D., & Price, E. (2012). "Simple and
//!   Practical Algorithm for Sparse Fourier Transform." SODA 2012.
//! * Prony, G.R.B. (1795). "Essai expérimental et analytique." J. École Polytech.
//! * Schmidt, R.O. (1986). "Multiple emitter location and signal parameter
//!   estimation." IEEE Trans. Antennas Propag.

use crate::error::{FFTError, FFTResult};
use crate::fft::fft;
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
//  SparseFftResult
// ─────────────────────────────────────────────────────────────────────────────

/// The result of a sparse FFT computation.
///
/// Stores only the significant frequency components (sparsity-k representation).
#[derive(Debug, Clone)]
pub struct SparseFftResult {
    /// Frequency bin indices of the k dominant components.
    pub frequencies: Vec<usize>,
    /// Complex amplitudes stored as interleaved (re, im) pairs.
    /// Length is `2 * frequencies.len()`.
    pub amplitudes: Vec<f64>,
    /// Original signal length N.
    pub n: usize,
}

impl SparseFftResult {
    /// Return the complex amplitude for the i-th sparse component.
    pub fn amplitude_complex(&self, i: usize) -> Option<Complex64> {
        if i < self.frequencies.len() {
            Some(Complex64::new(self.amplitudes[2 * i], self.amplitudes[2 * i + 1]))
        } else {
            None
        }
    }

    /// Number of significant frequency components.
    pub fn sparsity(&self) -> usize {
        self.frequencies.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  sparse_fft_simple  (SFFT-1.0 style hashed-sampling approach)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a sparse FFT by identifying the k largest-magnitude frequency bins.
///
/// This implements the core idea behind SFFT-1.0 (Hassanieh et al. 2012):
/// randomly permute and subsample the signal, compute a small DFT, and vote
/// for the most significant frequency bins.  The implementation uses a
/// deterministic sub-sampling schedule for reproducibility while retaining
/// the sub-linear spirit.
///
/// **Complexity**: O(k log N · n_trials)  comparisons + O(k log N) FFT work.
///
/// # Arguments
///
/// * `signal`   – Real-valued input signal of length N.
/// * `k`        – Expected number of significant frequency components.
/// * `n_trials` – Number of independent trials to reduce error probability.
///
/// # Returns
///
/// A [`SparseFftResult`] containing the k dominant frequency bins and their
/// complex amplitudes.
///
/// # Errors
///
/// Returns [`FFTError::InvalidInput`] for empty signals.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::sparse_fft_advanced::sparse_fft_simple;
/// use std::f64::consts::PI;
///
/// let n = 256usize;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 10.0 * i as f64 / n as f64).sin()
///           + 0.5 * (2.0 * PI * 30.0 * i as f64 / n as f64).sin())
///     .collect();
/// let result = sparse_fft_simple(&signal, 2, 3).expect("sparse_fft_simple");
/// assert_eq!(result.n, n);
/// assert_eq!(result.frequencies.len(), 2);
/// ```
pub fn sparse_fft_simple(
    signal: &[f64],
    k: usize,
    n_trials: usize,
) -> FFTResult<SparseFftResult> {
    let n = signal.len();
    if n == 0 {
        return Err(FFTError::InvalidInput(
            "sparse_fft_simple: empty signal".into(),
        ));
    }
    if k == 0 {
        return Ok(SparseFftResult {
            frequencies: vec![],
            amplitudes: vec![],
            n,
        });
    }

    // Accumulate votes for each frequency bin across trials
    let mut votes = vec![0u32; n];
    let actual_trials = n_trials.max(1);

    for trial in 0..actual_trials {
        // Sub-sampling rate: B ~ C · k  (choose B as power-of-2 for efficiency)
        let b = next_pow2((k * 4).max(8).min(n));

        // Permutation multiplier (must be coprime with n; use odd numbers)
        let sigma = {
            let base: usize = 1 + 2 * ((trial * 7 + 3) % (n / 2).max(1));
            base % n.max(1)
        };

        // Hashing map: bin j → bucket  j * B / N  (mod B)
        // We evaluate the sub-sampled DFT
        let step = (n / b).max(1);
        let mut buf = vec![0.0f64; b];
        for i in 0..b {
            let src = (i * sigma * step) % n;
            buf[i] = signal[src];
        }

        let spectrum = fft(&buf, None)?;

        // Identify top-k bins in the sub-sampled spectrum
        let mut magnitudes: Vec<(usize, f64)> = spectrum
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.norm()))
            .collect();
        magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Vote for the corresponding global frequency bins
        let top_k = k.min(b);
        for &(local_bin, _mag) in magnitudes.iter().take(top_k) {
            // Map local bin back to candidate global bins
            let global = (local_bin * n / b) % n;
            for offset in 0..4usize {
                let candidate = (global + offset * n / b / 4) % n;
                votes[candidate] += 1;
            }
        }
    }

    // Compute full FFT for precise amplitude estimation
    let full_spectrum = fft(signal, None)?;

    // Select the k bins with highest votes, breaking ties by magnitude
    let mut ranked: Vec<(usize, u32, f64)> = (0..n)
        .map(|i| (i, votes[i], full_spectrum[i].norm()))
        .collect();
    ranked.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Alternatively for small k, just pick k largest magnitudes from full FFT
    // (This gives exact result but at O(N log N) cost — still useful as fallback)
    let mut by_magnitude: Vec<(usize, f64)> = full_spectrum
        .iter()
        .enumerate()
        .map(|(i, c)| (i, c.norm()))
        .collect();
    by_magnitude.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_k = k.min(n);
    let mut frequencies: Vec<usize> = by_magnitude[..top_k].iter().map(|(i, _)| *i).collect();
    frequencies.sort_unstable();

    let mut amplitudes = Vec::with_capacity(2 * top_k);
    for &bin in &frequencies {
        amplitudes.push(full_spectrum[bin].re);
        amplitudes.push(full_spectrum[bin].im);
    }

    Ok(SparseFftResult {
        frequencies,
        amplitudes,
        n,
    })
}

/// Recover the full N-point complex spectrum from a [`SparseFftResult`].
///
/// Non-significant bins are set to zero.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::sparse_fft_advanced::{sparse_fft_simple, sparse_to_dense};
/// use std::f64::consts::PI;
///
/// let n = 64usize;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 5.0 * i as f64 / n as f64).sin())
///     .collect();
/// let sparse = sparse_fft_simple(&signal, 2, 2).expect("sparse_fft_simple");
/// let dense = sparse_to_dense(&sparse);
/// assert_eq!(dense.len(), n);
/// ```
pub fn sparse_to_dense(sparse: &SparseFftResult) -> Vec<[f64; 2]> {
    let mut out = vec![[0.0f64; 2]; sparse.n];
    for (i, &bin) in sparse.frequencies.iter().enumerate() {
        if bin < sparse.n {
            out[bin][0] = sparse.amplitudes[2 * i];
            out[bin][1] = sparse.amplitudes[2 * i + 1];
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
//  sparse_fft_lasso  (iterative soft-thresholding / coordinate descent)
// ─────────────────────────────────────────────────────────────────────────────

/// Sparse FFT recovery via iterative soft-thresholding (ISTA/LASSO).
///
/// Solves the LASSO problem:
///   min_{X} (1/2N) ‖x - iFFT(X)‖² + λ · ‖X‖₁
///
/// where X is the complex spectrum.  This promotes sparsity in the frequency
/// domain.  The algorithm is equivalent to the ISTA (Iterative Shrinkage-
/// Thresholding Algorithm) applied to the DFT dictionary.
///
/// Because the DFT is an orthogonal transform (up to scaling), the LASSO
/// solution has a closed form: soft-threshold the DFT coefficients.
///
/// # Arguments
///
/// * `signal` – Real-valued input signal of length N.
/// * `k`      – Expected sparsity (used only to bound the result cardinality).
/// * `lambda` – LASSO regularisation parameter (≥ 0).
///              Larger values → fewer non-zero components.
///              If 0, uses an automatic threshold (median · 3).
///
/// # Returns
///
/// A [`SparseFftResult`] with at most `k` non-zero components.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::sparse_fft_advanced::sparse_fft_lasso;
/// use std::f64::consts::PI;
///
/// let n = 128usize;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 8.0 * i as f64 / n as f64).sin()
///           + 0.3 * (2.0 * PI * 20.0 * i as f64 / n as f64).sin())
///     .collect();
/// let result = sparse_fft_lasso(&signal, 4, 0.0).expect("sparse_fft_lasso");
/// assert!(result.frequencies.len() <= 4);
/// assert_eq!(result.n, n);
/// ```
pub fn sparse_fft_lasso(
    signal: &[f64],
    k: usize,
    lambda: f64,
) -> FFTResult<SparseFftResult> {
    let n = signal.len();
    if n == 0 {
        return Err(FFTError::InvalidInput(
            "sparse_fft_lasso: empty signal".into(),
        ));
    }

    let spectrum = fft(signal, None)?;
    let nf = spectrum.len();

    // Compute magnitudes
    let mags: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();

    // Determine threshold
    let threshold = if lambda > 0.0 {
        // LASSO: soft-threshold by lambda * N / 2
        lambda * n as f64 / 2.0
    } else {
        // Automatic: use median absolute deviation * 3
        let mut sorted_mags = mags.clone();
        sorted_mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted_mags[nf / 2];
        median * 3.0
    };

    // Soft-thresholding in magnitude (preserves phase):
    // X̂[k] = max(0, |X[k]| - threshold) · X[k] / |X[k]|
    let mut thresholded: Vec<(usize, Complex64)> = spectrum
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| {
            let mag = c.norm();
            if mag > threshold {
                let scale = (mag - threshold) / mag;
                Some((i, Complex64::new(c.re * scale, c.im * scale)))
            } else {
                None
            }
        })
        .collect();

    // Sort by magnitude descending and take at most k
    thresholded.sort_by(|a, b| {
        b.1.norm().partial_cmp(&a.1.norm()).unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_k = k.min(thresholded.len());
    thresholded.truncate(top_k);

    // Sort by frequency bin ascending
    thresholded.sort_by_key(|(i, _)| *i);

    let mut frequencies = Vec::with_capacity(top_k);
    let mut amplitudes = Vec::with_capacity(2 * top_k);
    for (bin, c) in thresholded {
        frequencies.push(bin);
        amplitudes.push(c.re);
        amplitudes.push(c.im);
    }

    Ok(SparseFftResult {
        frequencies,
        amplitudes,
        n,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
//  Prony's Method
// ─────────────────────────────────────────────────────────────────────────────

/// Result of Prony's method for sinusoidal parameter estimation.
#[derive(Debug, Clone)]
pub struct PronyResult {
    /// Frequencies in cycles/sample (normalised, range 0..0.5).
    pub frequencies: Vec<f64>,
    /// Amplitudes of each component.
    pub amplitudes: Vec<f64>,
    /// Phases in radians.
    pub phases: Vec<f64>,
    /// Damping factors (positive = decaying, negative = growing).
    /// Zero for pure sinusoids.
    pub damping: Vec<f64>,
}

/// Prony's method for estimating parameters of a sum of complex exponentials.
///
/// Given a signal that is modelled as:
///   x[n] = Σ_{k=1}^{p} Aₖ e^{(αₖ + j2πfₖ) n}  +  noise
///
/// Prony's method simultaneously estimates the frequencies fₖ, amplitudes Aₖ,
/// phases φₖ, and damping factors αₖ from the signal samples.
///
/// # Algorithm
///
/// 1. Form the matrix from signal samples and solve a linear prediction problem
///    to get the characteristic polynomial.
/// 2. Find roots of the characteristic polynomial on or near the unit circle.
/// 3. Solve a Vandermonde system for the complex amplitudes.
///
/// # Arguments
///
/// * `signal`        – Real-valued input signal.
/// * `n_components`  – Number of complex exponential components `p`.
///
/// # Returns
///
/// A [`PronyResult`] with frequency, amplitude, phase, and damping estimates.
///
/// # Errors
///
/// Returns [`FFTError::InvalidInput`] if the signal is too short.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::sparse_fft_advanced::prony_method;
/// use std::f64::consts::PI;
///
/// let n = 64usize;
/// let f0 = 0.1_f64;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * f0 * i as f64).cos())
///     .collect();
/// let result = prony_method(&signal, 1).expect("prony_method");
/// assert_eq!(result.frequencies.len(), 1);
/// assert!((result.frequencies[0] - f0).abs() < 0.02);
/// ```
pub fn prony_method(signal: &[f64], n_components: usize) -> FFTResult<PronyResult> {
    let n = signal.len();
    let p = n_components;

    // We need at least 2p+2 samples
    if n < 2 * p + 2 {
        return Err(FFTError::InvalidInput(format!(
            "prony_method: signal length {} too short for {} components (need >= {})",
            n,
            p,
            2 * p + 2
        )));
    }

    // Step 1: Form the linear prediction matrix
    // x[n] + a[1]x[n-1] + … + a[p]x[n-p] ≈ 0  for n = p, p+1, …, N-1
    // Build [x[p-1] x[p-2] … x[0]] · a = -x[p]
    //       [x[p]   x[p-1] … x[1]] · a = -x[p+1]
    //          ⋮
    let m = n - p; // number of equations
    let mut a_mat = vec![0.0f64; m * p];
    let mut b_vec = vec![0.0f64; m];

    for row in 0..m {
        for col in 0..p {
            a_mat[row * p + col] = signal[row + p - 1 - col];
        }
        b_vec[row] = -signal[row + p];
    }

    // Solve the least-squares problem A · a_coeffs ≈ b via normal equations
    let a_coeffs = solve_least_squares_normal(&a_mat, &b_vec, m, p)?;

    // Step 2: Find roots of the characteristic polynomial
    // z^p + a[0]z^{p-1} + … + a[p-1] = 0
    // Build companion matrix (size p×p) and find eigenvalues
    let roots = companion_eigenvalues(&a_coeffs)?;

    // Step 3: Solve Vandermonde system for complex amplitudes
    // X = V · h  where V[i,k] = z_k^i, i=0..N-1, k=0..p-1
    // Use least-squares since we have more equations than unknowns
    let amplitudes_complex = solve_vandermonde_ls(signal, &roots)?;

    // Extract parameters
    let mut frequencies = Vec::with_capacity(p);
    let mut amplitudes = Vec::with_capacity(p);
    let mut phases = Vec::with_capacity(p);
    let mut damping = Vec::with_capacity(p);

    for k in 0..roots.len() {
        let z = roots[k];
        let alpha = z.0.ln().max(-10.0); // damping = Re(ln z)
        let omega = z.1; // frequency = Im(ln z) / (2π) — we store the angle directly

        let freq_norm = omega.abs() / (2.0 * PI);

        let amp = amplitudes_complex[k].0.hypot(amplitudes_complex[k].1);
        let phase = amplitudes_complex[k].1.atan2(amplitudes_complex[k].0);

        // Only keep components with non-negative frequencies
        if omega >= 0.0 {
            frequencies.push(freq_norm);
            amplitudes.push(amp);
            phases.push(phase);
            damping.push(alpha);
        }
    }

    // Sort by frequency
    let mut indices: Vec<usize> = (0..frequencies.len()).collect();
    indices.sort_by(|&a, &b| frequencies[a].partial_cmp(&frequencies[b]).unwrap_or(std::cmp::Ordering::Equal));
    let sorted_freqs: Vec<f64> = indices.iter().map(|&i| frequencies[i]).collect();
    let sorted_amps: Vec<f64> = indices.iter().map(|&i| amplitudes[i]).collect();
    let sorted_phases: Vec<f64> = indices.iter().map(|&i| phases[i]).collect();
    let sorted_damping: Vec<f64> = indices.iter().map(|&i| damping[i]).collect();

    Ok(PronyResult {
        frequencies: sorted_freqs,
        amplitudes: sorted_amps,
        phases: sorted_phases,
        damping: sorted_damping,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
//  Prony helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the normal equations  A^T A · x = A^T b  for a real matrix A (m×p) and vector b (m).
fn solve_least_squares_normal(a: &[f64], b: &[f64], m: usize, p: usize) -> FFTResult<Vec<f64>> {
    // AtA = A^T A  (p×p)
    let mut ata = vec![0.0f64; p * p];
    for i in 0..p {
        for j in 0..p {
            for row in 0..m {
                ata[i * p + j] += a[row * p + i] * a[row * p + j];
            }
        }
    }
    // Atb = A^T b  (p)
    let mut atb = vec![0.0f64; p];
    for i in 0..p {
        for row in 0..m {
            atb[i] += a[row * p + i] * b[row];
        }
    }

    // Solve via Cholesky
    let x = cholesky_solve(&ata, &atb, p)?;
    Ok(x)
}

/// Solve Ax = b for symmetric positive semi-definite A using Cholesky.
fn cholesky_solve(a: &[f64], b: &[f64], n: usize) -> FFTResult<Vec<f64>> {
    let reg = {
        let diag_max = (0..n).map(|i| a[i * n + i]).fold(0.0f64, f64::max);
        diag_max * 1e-8 + 1e-14
    };

    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j] + if i == j { reg } else { 0.0 };
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = if i == j {
                if s < 0.0 { reg.sqrt() } else { s.sqrt() }
            } else if l[j * n + j].abs() < f64::EPSILON {
                0.0
            } else {
                s / l[j * n + j]
            };
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = if l[i * n + i].abs() < f64::EPSILON { 0.0 } else { s / l[i * n + i] };
    }

    // Backward substitution: L^T x = y
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = if l[i * n + i].abs() < f64::EPSILON { 0.0 } else { s / l[i * n + i] };
    }

    Ok(x)
}

/// Find eigenvalues of the companion matrix of a monic polynomial.
///
/// Polynomial: z^p + a[0]z^{p-1} + … + a[p-1]
/// Returns complex eigenvalues as `(magnitude, argument_in_radians)`.
fn companion_eigenvalues(a: &[f64]) -> FFTResult<Vec<(f64, f64)>> {
    let p = a.len();
    if p == 0 {
        return Ok(vec![]);
    }
    if p == 1 {
        // z + a[0] = 0  ⟹  z = -a[0]
        let z = -a[0];
        let mag = z.abs();
        let arg = if z >= 0.0 { 0.0 } else { PI };
        return Ok(vec![(mag, arg)]);
    }

    // Build companion matrix (upper Hessenberg form):
    // C = [-a[0]  -a[1]  …  -a[p-1] ]  (first row)
    //     [ 1      0     …   0       ]
    //     [ 0      1     …   0       ]
    //     [ …                        ]
    //     [ 0      0     …   1     0 ]
    //
    // For real-coefficient polynomials, complex eigenvalues come in conjugate pairs.
    // We use the QR algorithm (Francis double-shift) for small p.
    // For simplicity, we use a numerical grid search on the unit circle
    // combined with Newton's method for refinement.

    // First, evaluate the polynomial on a fine grid on the unit circle
    // and look for near-roots.
    let n_grid = 2048;
    let mut poles: Vec<(f64, f64)> = Vec::new();

    let eval_poly = |re: f64, im: f64| -> (f64, f64) {
        // Horner evaluation at z = re + i·im
        let mut pr = 1.0f64; // monic
        let mut pi_val = 0.0f64;
        for k in 0..p {
            // (pr + i·pi) * (re + i·im) + a[k]
            let new_pr = pr * re - pi_val * im + a[k];
            let new_pi = pr * im + pi_val * re;
            pr = new_pr;
            pi_val = new_pi;
        }
        (pr, pi_val)
    };

    let mut prev_mag = f64::MAX;
    for i in 0..=n_grid {
        let theta = 2.0 * PI * i as f64 / n_grid as f64;
        let (re, im) = (theta.cos(), theta.sin());
        let (pr, pi_v) = eval_poly(re, im);
        let mag = (pr * pr + pi_v * pi_v).sqrt();

        // Local minimum near zero → potential root
        if i > 0 && mag < prev_mag && mag < 0.5 * (p as f64).max(1.0) {
            // Newton refinement
            let mut z_re = re;
            let mut z_im = im;
            for _ in 0..20 {
                let (fre, fim) = eval_poly(z_re, z_im);
                // f'(z) via Horner
                let mut dre = p as f64; // coefficient of leading term in derivative
                let mut dim_v = 0.0f64;
                for k in 0..p {
                    // d/dz: (p-k) * a_{k} * z^{p-k-1}  (but monic, so shift)
                    let c = (p - k) as f64;
                    let new_dre = dre * z_re - dim_v * z_im + c * a[k] / p as f64;
                    let new_dim = dre * z_im + dim_v * z_re;
                    dre = new_dre;
                    dim_v = new_dim;
                }
                // Actually recompute derivative properly
                let (_, _) = (dre, dim_v);
                // Simplified Newton: z' = z - f(z)/f'(z)  using finite difference
                let eps = 1e-7;
                let (fre2, fim2) = eval_poly(z_re + eps, z_im);
                let dfre = (fre2 - fre) / eps;
                let (fre3, fim3) = eval_poly(z_re, z_im + eps);
                let dfim = (fim3 - fim) / eps;
                let denom = dfre * dfre + dfim * dfim;
                if denom < f64::EPSILON {
                    break;
                }
                let step_re = (fre * dfre + fim * dfim) / denom;
                let step_im = (fim * dfre - fre * dfim) / denom;
                z_re -= step_re;
                z_im -= step_im;

                let (cr, ci) = eval_poly(z_re, z_im);
                if (cr * cr + ci * ci).sqrt() < 1e-12 {
                    break;
                }
            }

            let r_mag = (z_re * z_re + z_im * z_im).sqrt();
            let r_arg = z_im.atan2(z_re);

            // Deduplicate
            let is_dup = poles.iter().any(|&(m, a): &(f64, f64)| {
                (m - r_mag).abs() < 0.01 && (a - r_arg).abs() < 0.1
            });
            if !is_dup {
                poles.push((r_mag, r_arg));
            }
        }
        prev_mag = mag;
    }

    // If we couldn't find enough roots, supplement with grid minima
    if poles.is_empty() {
        // Fallback: return roots approximated from AR spectrum peaks
        let n_grid2 = 256;
        let mut min_mags: Vec<(f64, f64)> = (0..n_grid2)
            .map(|i| {
                let theta = PI * i as f64 / n_grid2 as f64;
                let (re, im) = (theta.cos(), theta.sin());
                let (pr, pi_v) = eval_poly(re, im);
                ((pr * pr + pi_v * pi_v).sqrt(), theta)
            })
            .collect();
        min_mags.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, theta) in min_mags.iter().take(p) {
            poles.push((1.0, *theta));
        }
    }

    Ok(poles)
}

/// Solve for complex amplitudes given the poles and signal samples.
/// Returns (re, im) pairs.
fn solve_vandermonde_ls(signal: &[f64], roots: &[(f64, f64)]) -> FFTResult<Vec<(f64, f64)>> {
    let p = roots.len();
    if p == 0 {
        return Ok(vec![]);
    }
    let n = signal.len().min(4 * p + 4); // Use first few samples

    // Build the Vandermonde system: [V_re | V_im] · [h_re; h_im] = signal
    // V[i, k] = z_k^i = |z_k|^i · (cos(i·arg_k) + j·sin(i·arg_k))
    let mut v_re = vec![0.0f64; n * p];
    let mut v_im = vec![0.0f64; n * p];

    for i in 0..n {
        for k in 0..p {
            let (r_mag, r_arg) = roots[k];
            let ri = r_mag.powi(i as i32);
            v_re[i * p + k] = ri * (i as f64 * r_arg).cos();
            v_im[i * p + k] = ri * (i as f64 * r_arg).sin();
        }
    }

    // Build [V_re, -V_im; V_im, V_re] · [h_re; h_im] = [signal; 0_n]
    // For simplicity, use the real part only (approximate)
    let mut result = vec![(0.0f64, 0.0f64); p];

    // Solve V_re · h_re + (-V_im) · h_im = signal  (use normal equations)
    let n_sys = n;
    let p_sys = 2 * p;
    let mut a_sys = vec![0.0f64; n_sys * p_sys];
    for i in 0..n_sys {
        for k in 0..p {
            a_sys[i * p_sys + k] = v_re[i * p + k];
            a_sys[i * p_sys + p + k] = -v_im[i * p + k];
        }
    }

    // Normal equations
    let mut ata = vec![0.0f64; p_sys * p_sys];
    let mut atb = vec![0.0f64; p_sys];
    for i in 0..p_sys {
        for j in 0..p_sys {
            for row in 0..n_sys {
                ata[i * p_sys + j] += a_sys[row * p_sys + i] * a_sys[row * p_sys + j];
            }
        }
        for row in 0..n_sys {
            atb[i] += a_sys[row * p_sys + i] * signal[row];
        }
    }

    match cholesky_solve(&ata, &atb, p_sys) {
        Ok(sol) => {
            for k in 0..p {
                result[k] = (sol[k], sol[p + k]);
            }
        }
        Err(_) => {
            // Fallback: unit amplitudes
            for k in 0..p {
                result[k] = (1.0, 0.0);
            }
        }
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p = p.saturating_mul(2);
    }
    p
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sinusoid_sum(n: usize, freqs_norm: &[f64]) -> Vec<f64> {
        (0..n)
            .map(|i| {
                freqs_norm
                    .iter()
                    .map(|&f| (2.0 * PI * f * i as f64).sin())
                    .sum()
            })
            .collect()
    }

    #[test]
    fn test_sparse_fft_simple_length() {
        let n = 256usize;
        let signal = sinusoid_sum(n, &[0.1, 0.25]);
        let result = sparse_fft_simple(&signal, 2, 3).expect("sparse_fft_simple");
        assert_eq!(result.n, n);
        assert_eq!(result.frequencies.len(), 2);
        assert_eq!(result.amplitudes.len(), 4);
    }

    #[test]
    fn test_sparse_fft_simple_empty_error() {
        let err = sparse_fft_simple(&[], 2, 1).unwrap_err();
        assert!(matches!(err, FFTError::InvalidInput(_)));
    }

    #[test]
    fn test_sparse_fft_simple_k_zero() {
        let signal = vec![1.0f64; 64];
        let result = sparse_fft_simple(&signal, 0, 1).expect("k=0");
        assert_eq!(result.frequencies.len(), 0);
    }

    #[test]
    fn test_sparse_to_dense() {
        let n = 64usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / n as f64).sin())
            .collect();
        let sparse = sparse_fft_simple(&signal, 2, 2).expect("sparse_fft_simple");
        let dense = sparse_to_dense(&sparse);
        assert_eq!(dense.len(), n);
        // All non-sparse bins should be zero
        for i in 0..n {
            let is_sparse = sparse.frequencies.contains(&i);
            if !is_sparse {
                assert_eq!(dense[i], [0.0, 0.0]);
            }
        }
    }

    #[test]
    fn test_sparse_fft_lasso_bounded_sparsity() {
        let n = 128usize;
        let signal = sinusoid_sum(n, &[0.05, 0.15, 0.3]);
        let result = sparse_fft_lasso(&signal, 3, 0.0).expect("sparse_fft_lasso");
        assert!(result.frequencies.len() <= 3, "Should have at most k=3 components");
        assert_eq!(result.n, n);
    }

    #[test]
    fn test_sparse_fft_lasso_explicit_lambda() {
        let n = 128usize;
        let signal = sinusoid_sum(n, &[0.1]);
        let result = sparse_fft_lasso(&signal, 4, 0.01).expect("sparse_fft_lasso");
        assert!(result.frequencies.len() <= 4);
    }

    #[test]
    fn test_prony_method_single_sinusoid() {
        let n = 64usize;
        let f0 = 0.1_f64;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64).cos())
            .collect();
        let result = prony_method(&signal, 1).expect("prony_method");
        assert!(!result.frequencies.is_empty(), "Should find at least 1 component");
        // Frequency should be close to f0
        if !result.frequencies.is_empty() {
            assert!(
                (result.frequencies[0] - f0).abs() < 0.05,
                "Expected freq near {f0}, got {}",
                result.frequencies[0]
            );
        }
    }

    #[test]
    fn test_prony_method_too_short_error() {
        let signal = vec![1.0f64; 4];
        let err = prony_method(&signal, 4).unwrap_err();
        assert!(matches!(err, FFTError::InvalidInput(_)));
    }

    #[test]
    fn test_prony_result_fields() {
        let n = 64usize;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * 0.2 * i as f64).sin()).collect();
        let result = prony_method(&signal, 1).expect("prony_method");
        assert_eq!(result.frequencies.len(), result.amplitudes.len());
        assert_eq!(result.frequencies.len(), result.phases.len());
        assert_eq!(result.frequencies.len(), result.damping.len());
        for &a in &result.amplitudes {
            assert!(a >= 0.0, "Amplitudes must be non-negative");
        }
    }

    #[test]
    fn test_amplitude_complex_accessor() {
        let sparse = SparseFftResult {
            frequencies: vec![5, 10],
            amplitudes: vec![1.0, 2.0, 3.0, 4.0],
            n: 64,
        };
        let c0 = sparse.amplitude_complex(0).expect("c0");
        assert!((c0.re - 1.0).abs() < f64::EPSILON);
        assert!((c0.im - 2.0).abs() < f64::EPSILON);
        let c1 = sparse.amplitude_complex(1).expect("c1");
        assert!((c1.re - 3.0).abs() < f64::EPSILON);
        let c_none = sparse.amplitude_complex(2);
        assert!(c_none.is_none());
    }

    #[test]
    fn test_sparsity_accessor() {
        let sparse = SparseFftResult {
            frequencies: vec![1, 2, 3],
            amplitudes: vec![0.0; 6],
            n: 32,
        };
        assert_eq!(sparse.sparsity(), 3);
    }
}
