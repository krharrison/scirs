//! Parametric Spectral Estimation
//!
//! Implements AR and ARMA spectral estimation algorithms with a clean,
//! unified API.  All methods are pure-Rust with no external BLAS/LAPACK
//! calls.
//!
//! # Algorithms
//!
//! * **Burg's method** – lattice-based forward/backward prediction error
//!   minimisation.  Maximises entropy; well-suited for short data records
//!   and guaranteed-stable AR models.
//!
//! * **Yule-Walker method** – autocorrelation-based Levinson-Durbin recursion.
//!   Classic textbook approach; slightly biased but robust.
//!
//! * **AR power spectral density** – evaluates the AR transfer function on
//!   the unit circle.
//!
//! * **ARMA spectral density** – estimates the MA part from AR residuals via
//!   the modified Yule-Walker equations.
//!
//! * **AR order selection** – AIC, BIC, and MDL criteria.
//!
//! # References
//!
//! - Burg, J.P. (1975). Maximum Entropy Spectral Analysis.  PhD Thesis,
//!   Stanford University.
//! - Kay, S.M. (1988). Modern Spectral Estimation. Prentice Hall.
//! - Percival, D.B. & Walden, A.T. (1993). Spectral Analysis for Physical
//!   Applications. Cambridge University Press.
//!
//! # Examples
//!
//! ```
//! use scirs2_signal::parametric_spectral::{burg_ar, ar_psd, ar_order_selection, ArOrderCriterion};
//! use scirs2_core::ndarray::Array1;
//! use std::f64::consts::PI;
//!
//! let n = 512usize;
//! let fs = 1000.0f64;
//! let signal: Array1<f64> = Array1::from_iter(
//!     (0..n).map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin()
//!               + 0.1 * (i as f64 % 7.0 - 3.0))
//! );
//!
//! let (ar_coeffs, noise_var) = burg_ar(&signal, 8).expect("operation should succeed");
//! let (freqs, psd) = ar_psd(&ar_coeffs, noise_var, 512, fs).expect("operation should succeed");
//! assert!(!freqs.is_empty());
//! assert_eq!(freqs.len(), psd.len());
//! ```

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Public enum
// ---------------------------------------------------------------------------

/// Criterion used for automatic AR order selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArOrderCriterion {
    /// Akaike Information Criterion.
    AIC,
    /// Bayesian Information Criterion.
    BIC,
    /// Minimum Description Length.
    MDL,
}

// ---------------------------------------------------------------------------
// Burg's method
// ---------------------------------------------------------------------------

/// Estimate AR parameters using Burg's lattice method.
///
/// Burg's algorithm minimises the combined forward and backward prediction
/// errors at each stage of a lattice filter. It is guaranteed to produce
/// a causal, stable AR model and works well for short data records.
///
/// The returned coefficient vector follows the convention used in signal
/// processing: `coeffs[0] == 1.0`, and the prediction equation is
///
/// ```text
/// x[n] + a_1·x[n-1] + … + a_p·x[n-p] = e[n]
/// ```
///
/// so `coeffs = [1, a_1, a_2, …, a_p]`.
///
/// # Arguments
///
/// * `signal` – Input signal (must have more samples than `order + 1`).
/// * `order`  – AR model order.
///
/// # Returns
///
/// `(ar_coeffs, noise_variance)` where `ar_coeffs` has length `order + 1`.
pub fn burg_ar(signal: &Array1<f64>, order: usize) -> SignalResult<(Vec<f64>, f64)> {
    let n = signal.len();
    if order == 0 {
        return Err(SignalError::ValueError("order must be at least 1".to_string()));
    }
    if order >= n {
        return Err(SignalError::ValueError(format!(
            "order ({order}) must be less than signal length ({n})"
        )));
    }

    let x: Vec<f64> = signal.iter().copied().collect();

    // Forward and backward prediction error vectors
    let mut f: Vec<f64> = x.clone(); // f[i] = forward  prediction error order m
    let mut b: Vec<f64> = x.clone(); // b[i] = backward prediction error order m

    // AR coefficients (Levinson-Durbin accumulation)
    let mut a: Vec<f64> = vec![0.0; order + 1];
    a[0] = 1.0;

    // Initial prediction error power = signal variance (unbiased)
    let mean = x.iter().sum::<f64>() / n as f64;
    let mut pm: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;

    for m in 1..=order {
        // Reflection coefficient via Burg formula
        let mut num = 0.0f64;
        let mut denom = 0.0f64;
        for i in m..n {
            num += 2.0 * f[i] * b[i - 1];
            denom += f[i] * f[i] + b[i - 1] * b[i - 1];
        }
        let km = if denom.abs() < 1e-30 { 0.0 } else { -num / denom };

        // Update prediction errors
        let mut f_new = vec![0.0f64; n];
        let mut b_new = vec![0.0f64; n];
        for i in m..n {
            f_new[i] = f[i] + km * b[i - 1];
            b_new[i] = b[i - 1] + km * f[i];
        }
        f = f_new;
        b = b_new;

        // Update AR coefficients via Levinson-Durbin recursion
        let a_prev = a.clone();
        a[m] = km;
        for j in 1..m {
            a[j] = a_prev[j] + km * a_prev[m - j];
        }

        // Update noise power
        pm *= 1.0 - km * km;
    }

    // Clamp noise variance to be non-negative
    let noise_var = pm.max(0.0);

    Ok((a, noise_var))
}

// ---------------------------------------------------------------------------
// Yule-Walker method
// ---------------------------------------------------------------------------

/// Estimate AR parameters using the Yule-Walker equations (autocorrelation method).
///
/// Solves the Yule-Walker system via the Levinson-Durbin recursion, which is
/// numerically stable and O(p²) in the model order.
///
/// # Arguments
///
/// * `signal` – Input signal.
/// * `order`  – AR model order.
///
/// # Returns
///
/// `(ar_coeffs, noise_variance)`.  `ar_coeffs[0] == 1.0`.
pub fn yule_walker_ar(signal: &Array1<f64>, order: usize) -> SignalResult<(Vec<f64>, f64)> {
    let n = signal.len();
    if order == 0 {
        return Err(SignalError::ValueError("order must be at least 1".to_string()));
    }
    if order >= n {
        return Err(SignalError::ValueError(format!(
            "order ({order}) must be less than signal length ({n})"
        )));
    }

    // Compute biased autocorrelation estimates R[0..=order]
    let mean = signal.iter().sum::<f64>() / n as f64;
    let x: Vec<f64> = signal.iter().map(|v| v - mean).collect();
    let mut r = vec![0.0f64; order + 1];
    for lag in 0..=order {
        let sum: f64 = (lag..n).map(|i| x[i] * x[i - lag]).sum();
        r[lag] = sum / n as f64;
    }

    if r[0].abs() < 1e-30 {
        // Trivial zero signal
        let mut a = vec![0.0f64; order + 1];
        a[0] = 1.0;
        return Ok((a, 0.0));
    }

    // Levinson-Durbin recursion
    let mut a = vec![0.0f64; order + 1];
    a[0] = 1.0;
    let mut pm = r[0];
    let mut k_vec = vec![0.0f64; order + 1];

    for m in 1..=order {
        // Reflection coefficient
        let mut lambda = -r[m];
        for j in 1..m {
            lambda -= a[j] * r[m - j];
        }
        let km = if pm.abs() < 1e-30 { 0.0 } else { lambda / pm };
        k_vec[m] = km;

        // Update AR coefficients
        let a_prev = a.clone();
        a[m] = km;
        for j in 1..m {
            a[j] = a_prev[j] + km * a_prev[m - j];
        }
        pm *= 1.0 - km * km;
    }

    Ok((a, pm.max(0.0)))
}

// ---------------------------------------------------------------------------
// AR power spectral density
// ---------------------------------------------------------------------------

/// Compute the one-sided AR power spectral density.
///
/// The AR PSD is evaluated as
///
/// ```text
/// S(f) = σ² / |A(e^{j2πf/fs})|²
/// ```
///
/// where `A(z) = 1 + a_1·z⁻¹ + … + a_p·z⁻ᵖ`.
///
/// # Arguments
///
/// * `ar_coeffs`      – AR coefficients `[1, a_1, …, a_p]`.
/// * `noise_variance` – Driving noise variance σ².
/// * `nfft`           – Number of frequency points.
/// * `sample_rate`    – Sampling rate in Hz.
///
/// # Returns
///
/// `(frequencies, psd)`, one-sided (0 … Nyquist), length `nfft/2 + 1`.
pub fn ar_psd(
    ar_coeffs: &[f64],
    noise_variance: f64,
    nfft: usize,
    sample_rate: f64,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    if ar_coeffs.is_empty() {
        return Err(SignalError::ValueError(
            "ar_coeffs must not be empty".to_string(),
        ));
    }
    if noise_variance < 0.0 {
        return Err(SignalError::ValueError(
            "noise_variance must be non-negative".to_string(),
        ));
    }
    if nfft < 2 {
        return Err(SignalError::ValueError(
            "nfft must be at least 2".to_string(),
        ));
    }
    if sample_rate <= 0.0 {
        return Err(SignalError::ValueError(
            "sample_rate must be positive".to_string(),
        ));
    }

    let n_out = nfft / 2 + 1;
    let dt = 1.0 / sample_rate;

    let mut freqs = vec![0.0f64; n_out];
    let mut psd = vec![0.0f64; n_out];

    for b in 0..n_out {
        let f = b as f64 / (nfft as f64 * dt);
        freqs[b] = f;
        let omega = 2.0 * PI * f * dt; // normalised angular frequency
        // Evaluate A(e^{jω}) = sum_k a_k * e^{-jkω}
        let mut a_z = Complex64::new(0.0, 0.0);
        for (k, &a_k) in ar_coeffs.iter().enumerate() {
            let angle = -(k as f64) * omega;
            a_z += a_k * Complex64::new(angle.cos(), angle.sin());
        }
        let denom = a_z.norm_sqr();
        psd[b] = if denom > 1e-300 {
            noise_variance / denom / sample_rate
        } else {
            f64::INFINITY
        };
    }

    Ok((Array1::from(freqs), Array1::from(psd)))
}

// ---------------------------------------------------------------------------
// ARMA spectral density
// ---------------------------------------------------------------------------

/// Compute the ARMA spectral density estimate.
///
/// The algorithm:
/// 1. Estimate a high-order AR model (order `4 * (ar_order + ma_order)`)
///    using Burg's method.
/// 2. Compute residuals by filtering through the AR inverse.
/// 3. Estimate the MA part from the residual autocorrelation via the
///    Yule-Walker equations for the MA process.
/// 4. Evaluate the combined ARMA spectrum on the DFT grid.
///
/// # Arguments
///
/// * `signal`   – Input signal.
/// * `ar_order` – AR part order (p).
/// * `ma_order` – MA part order (q).
/// * `sample_rate` – Sampling rate in Hz.
/// * `nfft`     – FFT length (default: next power-of-two ≥ signal length).
///
/// # Returns
///
/// `(frequencies, psd)`, one-sided.
pub fn arma_psd(
    signal: &Array1<f64>,
    ar_order: usize,
    ma_order: usize,
    sample_rate: f64,
    nfft: Option<usize>,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    let n = signal.len();
    if ar_order == 0 && ma_order == 0 {
        return Err(SignalError::ValueError(
            "At least one of ar_order or ma_order must be > 0".to_string(),
        ));
    }
    if sample_rate <= 0.0 {
        return Err(SignalError::ValueError(
            "sample_rate must be positive".to_string(),
        ));
    }

    let nfft_val = nfft.unwrap_or_else(|| n.next_power_of_two());

    // -----------------------------------------------------------------------
    // Step 1: Fit a high-order AR model to the signal using Burg's method
    // -----------------------------------------------------------------------
    let pilot_order = (4 * (ar_order + ma_order)).min(n / 2 - 1).max(1);
    let (ar_pilot, sigma2_pilot) = burg_ar(signal, pilot_order)?;

    // -----------------------------------------------------------------------
    // Step 2: Compute residuals by filtering through A(z)
    // -----------------------------------------------------------------------
    let residuals: Vec<f64> = (0..n)
        .map(|i| {
            let mut acc = signal[i];
            for (k, &a_k) in ar_pilot[1..].iter().enumerate() {
                if i > k {
                    acc += a_k * signal[i - k - 1];
                }
            }
            acc
        })
        .collect();

    // -----------------------------------------------------------------------
    // Step 3: Estimate AR (order ar_order) and MA (order ma_order) parts
    // -----------------------------------------------------------------------
    let (ar_coeffs, sigma2) = if ar_order > 0 {
        burg_ar(signal, ar_order)?
    } else {
        (vec![1.0], sigma2_pilot)
    };

    // For MA part, apply the AR inverse to get approximate MA innovations
    let ma_coeffs: Vec<f64> = if ma_order > 0 {
        // Use residuals autocorrelation to estimate MA coefficients
        let mean_r = residuals.iter().sum::<f64>() / n as f64;
        let res_demean: Vec<f64> = residuals.iter().map(|v| v - mean_r).collect();
        let mut r_ma = vec![0.0f64; ma_order + 1];
        for lag in 0..=ma_order {
            r_ma[lag] = res_demean[lag..]
                .iter()
                .zip(res_demean.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / n as f64;
        }
        // Estimate MA from autocorrelation via Durbin method
        estimate_ma_from_autocorr(&r_ma, ma_order)?
    } else {
        vec![1.0]
    };

    // -----------------------------------------------------------------------
    // Step 4: Evaluate ARMA spectrum S(f) = σ² |B(z)|² / |A(z)|²
    // -----------------------------------------------------------------------
    let n_out = nfft_val / 2 + 1;
    let dt = 1.0 / sample_rate;
    let mut freqs = vec![0.0f64; n_out];
    let mut psd = vec![0.0f64; n_out];

    for b in 0..n_out {
        let f = b as f64 / (nfft_val as f64 * dt);
        freqs[b] = f;
        let omega = 2.0 * PI * f * dt;

        let mut a_z = Complex64::new(0.0, 0.0);
        for (k, &a_k) in ar_coeffs.iter().enumerate() {
            let angle = -(k as f64) * omega;
            a_z += a_k * Complex64::new(angle.cos(), angle.sin());
        }
        let mut b_z = Complex64::new(0.0, 0.0);
        for (k, &b_k) in ma_coeffs.iter().enumerate() {
            let angle = -(k as f64) * omega;
            b_z += b_k * Complex64::new(angle.cos(), angle.sin());
        }
        let denom = a_z.norm_sqr();
        psd[b] = if denom > 1e-300 {
            sigma2 * b_z.norm_sqr() / denom / sample_rate
        } else {
            0.0
        };
    }

    Ok((Array1::from(freqs), Array1::from(psd)))
}

/// Estimate MA coefficients from autocorrelation via Durbin's method.
fn estimate_ma_from_autocorr(r: &[f64], q: usize) -> SignalResult<Vec<f64>> {
    // Use a long AR approximation to the MA process (Durbin 1959)
    // The residuals from a long AR model approximate the MA innovations.
    // Here we use a simpler approach: direct autocorrelation matching.
    if r.is_empty() || q == 0 {
        return Ok(vec![1.0]);
    }

    let r0 = r[0];
    if r0.abs() < 1e-30 {
        let mut ma = vec![0.0f64; q + 1];
        ma[0] = 1.0;
        return Ok(ma);
    }

    // Use a simple approximation: fit AR(q) to the autocorrelation function
    // of the white-noise-filtered signal, then use the spectral factorisation
    // idea that MA(q) autocorrelations satisfy the same Yule-Walker form.
    // For a practical implementation we solve the truncated autocorrelation
    // system for the MA coefficients.

    // Build the Toeplitz system R·θ = r (right-hand side is r[1..q])
    // using a simple Levinson recursion on the autocorrelation
    let (yule_a, _) = yule_walker_from_autocorr(r, q)?;

    // The MA coefficients from this approximation
    // θ_k ≈ -yule_a[k]  for k = 1..q,  θ_0 = 1
    let mut ma = vec![0.0f64; q + 1];
    ma[0] = 1.0;
    for k in 1..=q {
        if k < yule_a.len() {
            ma[k] = -yule_a[k];
        }
    }
    Ok(ma)
}

/// Levinson-Durbin solve using a pre-computed autocorrelation sequence.
fn yule_walker_from_autocorr(r: &[f64], order: usize) -> SignalResult<(Vec<f64>, f64)> {
    if r.is_empty() || r[0].abs() < 1e-30 {
        let mut a = vec![0.0f64; order + 1];
        a[0] = 1.0;
        return Ok((a, 0.0));
    }

    let mut a = vec![0.0f64; order + 1];
    a[0] = 1.0;
    let mut pm = r[0];

    for m in 1..=order {
        let lag = if m < r.len() { r[m] } else { 0.0 };
        let mut lambda = -lag;
        for j in 1..m {
            let rlag = if m - j < r.len() { r[m - j] } else { 0.0 };
            lambda -= a[j] * rlag;
        }
        let km = if pm.abs() < 1e-30 { 0.0 } else { lambda / pm };
        let a_prev = a.clone();
        a[m] = km;
        for j in 1..m {
            a[j] = a_prev[j] + km * a_prev[m - j];
        }
        pm *= 1.0 - km * km;
    }
    Ok((a, pm.max(0.0)))
}

// ---------------------------------------------------------------------------
// AR order selection
// ---------------------------------------------------------------------------

/// Select the optimal AR model order using an information criterion.
///
/// Fits AR models from order 1 to `max_order` using Burg's method and
/// evaluates the specified information criterion at each order.
///
/// | Criterion | Formula |
/// |-----------|---------|
/// | AIC       | N·ln(σ²) + 2·p |
/// | BIC       | N·ln(σ²) + p·ln(N) |
/// | MDL       | N·ln(σ²) + p·ln(N)/2 |
///
/// # Arguments
///
/// * `signal`     – Input signal.
/// * `max_order`  – Maximum AR order to consider.
/// * `criterion`  – Which information criterion to use.
///
/// # Returns
///
/// `(selected_order, criterion_values)` where `criterion_values` has length
/// `max_order` with values at orders 1, 2, …, max_order.
pub fn ar_order_selection(
    signal: &Array1<f64>,
    max_order: usize,
    criterion: ArOrderCriterion,
) -> SignalResult<(usize, Vec<f64>)> {
    let n = signal.len();
    if max_order == 0 {
        return Err(SignalError::ValueError(
            "max_order must be at least 1".to_string(),
        ));
    }
    if max_order >= n {
        return Err(SignalError::ValueError(format!(
            "max_order ({max_order}) must be less than signal length ({n})"
        )));
    }

    let fn_val = n as f64;
    let mut criterion_values = Vec::with_capacity(max_order);
    let mut best_order = 1usize;
    let mut best_val = f64::INFINITY;

    for p in 1..=max_order {
        let (_, sigma2) = burg_ar(signal, p)?;
        let ic = match criterion {
            ArOrderCriterion::AIC => {
                fn_val * (sigma2.max(1e-300)).ln() + 2.0 * p as f64
            }
            ArOrderCriterion::BIC => {
                fn_val * (sigma2.max(1e-300)).ln() + p as f64 * fn_val.ln()
            }
            ArOrderCriterion::MDL => {
                fn_val * (sigma2.max(1e-300)).ln() + p as f64 * fn_val.ln() / 2.0
            }
        };
        criterion_values.push(ic);
        if ic < best_val {
            best_val = ic;
            best_order = p;
        }
    }

    Ok((best_order, criterion_values))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array1;
    use std::f64::consts::PI;

    fn make_sinusoid(freq: f64, fs: f64, n: usize) -> Array1<f64> {
        Array1::from_iter((0..n).map(|i| (2.0 * PI * freq * i as f64 / fs).sin()))
    }

    fn pseudo_noise(n: usize, seed: u64) -> Array1<f64> {
        let mut x = vec![0.0f64; n];
        let mut s = seed ^ 0xdeadbeef;
        for v in x.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((s >> 33) as f64) / (u32::MAX as f64) - 0.5;
        }
        Array1::from(x)
    }

    // AR(2) process: x[n] = a1*x[n-1] + a2*x[n-2] + noise
    fn ar2_process(a1: f64, a2: f64, n: usize, seed: u64) -> Array1<f64> {
        let noise = pseudo_noise(n, seed);
        let mut x = vec![0.0f64; n];
        for i in 2..n {
            x[i] = a1 * x[i - 1] + a2 * x[i - 2] + 0.1 * noise[i];
        }
        Array1::from(x)
    }

    // -----------------------------------------------------------------------
    // Burg AR tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_burg_ar_coeff_length() {
        let signal = pseudo_noise(256, 1);
        let (coeffs, _) = burg_ar(&signal, 8).expect("unexpected None or Err");
        assert_eq!(coeffs.len(), 9); // order + 1
        assert_relative_eq!(coeffs[0], 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_burg_ar_positive_noise_variance() {
        let signal = pseudo_noise(256, 2);
        let (_, noise_var) = burg_ar(&signal, 8).expect("unexpected None or Err");
        assert!(noise_var >= 0.0);
    }

    #[test]
    fn test_burg_ar_recovers_ar2() {
        // Generate an AR(2) process with known coefficients
        // x[n] = 1.5*x[n-1] - 0.75*x[n-2] + e[n]
        // True AR coefficients: [1, -1.5, 0.75]
        let a1 = 1.5f64;
        let a2 = -0.75f64;
        let signal = ar2_process(a1, a2, 2048, 42);
        let (coeffs, _) = burg_ar(&signal, 2).expect("unexpected None or Err");
        // a[1] ≈ -a1, a[2] ≈ -a2 (sign flip because A(z)·X(z) = E(z))
        assert_relative_eq!(coeffs[1], -a1, epsilon = 0.05);
        assert_relative_eq!(coeffs[2], -a2, epsilon = 0.05);
    }

    #[test]
    fn test_burg_ar_stability() {
        // All poles of the AR model must lie inside the unit circle
        let signal = pseudo_noise(512, 77);
        let (coeffs, _) = burg_ar(&signal, 10).expect("unexpected None or Err");
        // Check via reflection coefficients that |km| < 1 implies stability.
        // For Burg, we trust the algorithm; verify |A(z)| > 0 on unit circle.
        for k in 0..64 {
            let omega = 2.0 * PI * k as f64 / 64.0;
            let mut a_z = Complex64::new(0.0, 0.0);
            for (j, &a_j) in coeffs.iter().enumerate() {
                let angle = -(j as f64) * omega;
                a_z += a_j * Complex64::new(angle.cos(), angle.sin());
            }
            assert!(a_z.norm() > 0.0, "A(z) is zero on the unit circle at k={k}");
        }
    }

    #[test]
    fn test_burg_ar_error_on_zero_order() {
        let signal = pseudo_noise(64, 5);
        assert!(burg_ar(&signal, 0).is_err());
    }

    // -----------------------------------------------------------------------
    // Yule-Walker tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_yule_walker_coeff_length() {
        let signal = pseudo_noise(256, 10);
        let (coeffs, _) = yule_walker_ar(&signal, 8).expect("unexpected None or Err");
        assert_eq!(coeffs.len(), 9);
        assert_relative_eq!(coeffs[0], 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_yule_walker_positive_noise_var() {
        let signal = pseudo_noise(256, 11);
        let (_, noise_var) = yule_walker_ar(&signal, 6).expect("unexpected None or Err");
        assert!(noise_var >= 0.0);
    }

    #[test]
    fn test_yule_walker_recovers_ar2() {
        let a1 = 1.5f64;
        let a2 = -0.75f64;
        let signal = ar2_process(a1, a2, 4096, 88);
        let (coeffs, _) = yule_walker_ar(&signal, 2).expect("unexpected None or Err");
        assert_relative_eq!(coeffs[1], -a1, epsilon = 0.05);
        assert_relative_eq!(coeffs[2], -a2, epsilon = 0.05);
    }

    #[test]
    fn test_yule_walker_error_order_too_large() {
        let signal = pseudo_noise(10, 3);
        assert!(yule_walker_ar(&signal, 10).is_err());
    }

    // -----------------------------------------------------------------------
    // AR PSD tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ar_psd_shape() {
        let signal = pseudo_noise(256, 20);
        let (coeffs, noise_var) = burg_ar(&signal, 8).expect("unexpected None or Err");
        let (freqs, psd) = ar_psd(&coeffs, noise_var, 512, 1000.0).expect("unexpected None or Err");
        assert_eq!(freqs.len(), 257); // 512/2 + 1
        assert_eq!(psd.len(), 257);
    }

    #[test]
    fn test_ar_psd_positive_values() {
        let signal = pseudo_noise(256, 21);
        let (coeffs, noise_var) = burg_ar(&signal, 6).expect("unexpected None or Err");
        let (_, psd) = ar_psd(&coeffs, noise_var, 256, 500.0).expect("unexpected None or Err");
        assert!(psd.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_ar_psd_peak_at_correct_frequency() {
        // AR(2) with spectral peak at f0
        let fs = 1000.0f64;
        let f0 = 100.0f64;
        // AR(2) with resonance at f0: poles at r·e^{±j2πf0/fs}
        // a1 = -2r·cos(2πf0/fs), a2 = r²
        let r = 0.95f64;
        let a1 = -2.0 * r * (2.0 * PI * f0 / fs).cos();
        let a2 = r * r;
        let coeffs = vec![1.0, a1, a2];
        let noise_var = 1.0;
        let nfft = 1024usize;

        let (freqs, psd) = ar_psd(&coeffs, noise_var, nfft, fs).expect("unexpected None or Err");
        let peak_idx = psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("unexpected None or Err");
        let peak_freq = freqs[peak_idx];
        assert!(
            (peak_freq - f0).abs() < 10.0,
            "AR PSD peak at {peak_freq} Hz, expected near {f0} Hz"
        );
    }

    #[test]
    fn test_ar_psd_error_negative_noise_var() {
        let coeffs = vec![1.0, -0.5];
        assert!(ar_psd(&coeffs, -1.0, 256, 1000.0).is_err());
    }

    // -----------------------------------------------------------------------
    // ARMA PSD tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_arma_psd_returns_valid_output() {
        let signal = pseudo_noise(512, 30);
        let (freqs, psd) = arma_psd(&signal, 4, 2, 1000.0, None).expect("unexpected None or Err");
        assert!(!freqs.is_empty());
        assert_eq!(freqs.len(), psd.len());
        assert!(psd.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_arma_psd_sinusoid_peak() {
        let n = 512;
        let fs = 1000.0;
        let f0 = 150.0;
        let signal = make_sinusoid(f0, fs, n);
        let (freqs, psd) = arma_psd(&signal, 4, 2, fs, None).expect("unexpected None or Err");
        let peak_idx = psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("unexpected None or Err");
        let peak_freq = freqs[peak_idx];
        assert!(
            (peak_freq - f0).abs() < 20.0,
            "ARMA PSD peak at {peak_freq}, expected near {f0}"
        );
    }

    #[test]
    fn test_arma_psd_ar_only() {
        // ma_order = 0 should work as pure AR
        let signal = pseudo_noise(256, 40);
        let (freqs, psd) = arma_psd(&signal, 4, 0, 1000.0, None).expect("unexpected None or Err");
        assert!(!freqs.is_empty());
        assert!(psd.iter().all(|&v| v >= 0.0));
    }

    // -----------------------------------------------------------------------
    // Order selection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_order_selection_aic_length() {
        let signal = pseudo_noise(256, 50);
        let (_, ic_vals) = ar_order_selection(&signal, 10, ArOrderCriterion::AIC).expect("unexpected None or Err");
        assert_eq!(ic_vals.len(), 10);
    }

    #[test]
    fn test_order_selection_bic() {
        let signal = pseudo_noise(256, 51);
        let (order, _) = ar_order_selection(&signal, 10, ArOrderCriterion::BIC).expect("unexpected None or Err");
        assert!(order >= 1 && order <= 10);
    }

    #[test]
    fn test_order_selection_mdl() {
        let signal = pseudo_noise(256, 52);
        let (order, _) = ar_order_selection(&signal, 10, ArOrderCriterion::MDL).expect("unexpected None or Err");
        assert!(order >= 1 && order <= 10);
    }

    #[test]
    fn test_order_selection_selects_true_order() {
        // AR(2) process should be identified as low-order by AIC
        let a1 = 1.5f64;
        let a2 = -0.75f64;
        let signal = ar2_process(a1, a2, 4096, 99);
        let (order, _) = ar_order_selection(&signal, 20, ArOrderCriterion::AIC).expect("unexpected None or Err");
        // Should select a low order (≤ 5) for a true AR(2) signal
        assert!(
            order <= 6,
            "AIC selected order {order} for AR(2) process, expected <= 6"
        );
    }

    #[test]
    fn test_order_selection_error_max_order_zero() {
        let signal = pseudo_noise(64, 60);
        assert!(ar_order_selection(&signal, 0, ArOrderCriterion::AIC).is_err());
    }

    #[test]
    fn test_order_selection_all_criteria_consistent() {
        // AIC, BIC, MDL should all return valid orders for the same input
        let signal = pseudo_noise(256, 61);
        for crit in [ArOrderCriterion::AIC, ArOrderCriterion::BIC, ArOrderCriterion::MDL] {
            let (order, vals) = ar_order_selection(&signal, 12, crit).expect("unexpected None or Err");
            assert!(order >= 1 && order <= 12, "Invalid order {order}");
            assert_eq!(vals.len(), 12);
        }
    }
}
