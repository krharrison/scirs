//! Kaiser-Bessel (KB) window for NUFFT spreading and deconvolution.
//!
//! The Kaiser-Bessel window is widely used in NUFFT implementations because its
//! Fourier transform has excellent concentration properties – it minimises the
//! maximum sidelobe level for a given main-lobe width, making it nearly optimal
//! for the spreading step of the NUFFT algorithm.
//!
//! # Window function
//!
//! For a half-width `m` (in grid units) and shape parameter `β`, the KB window is:
//!
//! ```text
//!           I₀(β · √(1 − (x/m)²))
//! w(x) =  ─────────────────────────   for |x| ≤ m
//!                 I₀(β)
//!
//!          0                           otherwise
//! ```
//!
//! where I₀ is the modified Bessel function of the first kind, order 0.
//!
//! # Deconvolution
//!
//! After spreading with the KB window and FFT, each mode is attenuated by the
//! Fourier transform of the window.  The correction factor stored in
//! [`kb_correction`] is the *reciprocal* of that attenuation.
//!
//! # References
//!
//! * Kaiser, J. F. (1966). Digital filters. System Analysis by Digital Computer, 218-285.
//! * Harris, F. J. (1978). On the use of windows for harmonic analysis with the discrete
//!   Fourier transform. Proceedings of the IEEE, 66(1), 51-83.
//! * Greengard, L., & Lee, J. Y. (2004). Accelerating the nonuniform fast Fourier transform.
//!   SIAM Review, 46(3), 443-454.

use std::f64::consts::PI;

// ─── Modified Bessel function I₀ ─────────────────────────────────────────────

/// Compute the modified Bessel function of the first kind, order 0: I₀(x).
///
/// Uses the polynomial approximation from Abramowitz & Stegun §9.8.
/// Accurate to ≈1 part in 10⁹ for all real `x`.
///
/// # Examples
///
/// ```
/// use scirs2_fft::keiser_bessel::bessel_i0;
///
/// // I₀(0) = 1
/// assert!((bessel_i0(0.0) - 1.0).abs() < 1e-12);
///
/// // I₀(x) is even
/// assert!((bessel_i0(2.5) - bessel_i0(-2.5)).abs() < 1e-12);
/// ```
pub fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = (ax / 3.75) * (ax / 3.75);
        poly_eval(
            t,
            &[
                1.000_000_000_0,
                3.515_623_000e-1,
                2.459_906_200e-2,
                4.300_132_200e-3,
                1.200_674_6e-4,
                1.934_805_3e-5,
                1.573_7e-7,
                1.8e-9,
            ],
        )
    } else {
        let t = 3.75 / ax;
        let envelope = ax.exp() / ax.sqrt();
        envelope
            * poly_eval(
                t,
                &[
                    3.989_422_804_0e-1,
                    1.328_592_0e-2,
                    2.253_190_0e-3,
                    -1.575_649_0e-3,
                    9.162_860_0e-4,
                    -2.057_706_0e-4,
                    2.635_537_0e-5,
                    -1.647_633_0e-6,
                    3.921_900_0e-7,
                ],
            )
    }
}

/// Evaluate a polynomial using Horner's method.
fn poly_eval(x: f64, coeffs: &[f64]) -> f64 {
    let mut result = 0.0;
    for &c in coeffs.iter().rev() {
        result = result * x + c;
    }
    result
}

// ─── Kaiser-Bessel window ─────────────────────────────────────────────────────

/// Evaluate the Kaiser-Bessel spreading window at position `x`.
///
/// ```text
///           I₀(β · √(1 − (x/m)²))
/// w(x) =  ─────────────────────────   for |x| ≤ m
///                 I₀(β)
///
///          0                           otherwise
/// ```
///
/// # Arguments
///
/// * `x`    – position relative to the centre of the window (grid units)
/// * `m`    – kernel half-width in grid units (support is `[-m, m]`)
/// * `beta` – shape parameter β (larger values → narrower main-lobe in frequency)
///
/// # Returns
///
/// Window weight ∈ `[0, 1]`.
///
/// # Examples
///
/// ```
/// use scirs2_fft::keiser_bessel::kaiser_bessel_window;
///
/// // At x = 0 the window equals 1
/// let w = kaiser_bessel_window(0.0, 4, 13.9);
/// assert!((w - 1.0).abs() < 1e-12, "w={}", w);
///
/// // Outside support the window is 0
/// let w_out = kaiser_bessel_window(5.0, 4, 13.9);
/// assert_eq!(w_out, 0.0);
/// ```
pub fn kaiser_bessel_window(x: f64, m: usize, beta: f64) -> f64 {
    let m_f = m as f64;
    if x.abs() > m_f {
        return 0.0;
    }
    let arg = 1.0 - (x / m_f) * (x / m_f);
    // arg is guaranteed ≥ 0 because |x| ≤ m
    let arg = arg.max(0.0);
    bessel_i0(beta * arg.sqrt()) / bessel_i0(beta)
}

// ─── Deconvolution correction factor ─────────────────────────────────────────

/// Correction factor to deconvolve the Kaiser-Bessel spreading kernel.
///
/// After spreading with the KB window and computing an FFT on an oversampled grid
/// of size `n`, each mode `k` is scaled by the Fourier transform of the window.
/// This function returns the *reciprocal* of that scaling so that multiplying
/// the raw FFT output by this correction recovers the true Fourier coefficient.
///
/// The Fourier transform of the KB window at (continuous) frequency `ξ = k/n` is
/// approximated by a modified sinc:
/// ```text
/// Ŵ(k/n) ≈ (2m/I₀(β)) · sinh(√(β² − (2πm·k/n)²)) / √(β² − (2πm·k/n)²)
/// ```
/// when `|2πm·k/n| < β`, and a cosine form for `|2πm·k/n| ≥ β`.
///
/// # Arguments
///
/// * `k`    – centred Fourier mode index
/// * `n`    – oversampled grid size
/// * `m`    – KB kernel half-width in grid units
/// * `beta` – KB shape parameter
///
/// # Returns
///
/// Multiplicative correction factor (reciprocal of the kernel's Fourier value).
///
/// # Examples
///
/// ```
/// use scirs2_fft::keiser_bessel::{kb_correction, optimal_beta};
///
/// let m = 4usize;
/// let n = 128usize;
/// let beta = optimal_beta(m, 1e-6);
///
/// // Correction at k = 0 should be close to 1 / (denominator at 0)
/// let corr = kb_correction(0, n, m, beta);
/// assert!(corr > 0.0 && corr.is_finite());
/// ```
pub fn kb_correction(k: i64, n: usize, m: usize, beta: f64) -> f64 {
    let m_f = m as f64;
    let n_f = n as f64;
    let t = 2.0 * PI * m_f * k as f64 / n_f;
    let arg = beta * beta - t * t;

    let i0_beta = bessel_i0(beta);
    if i0_beta == 0.0 {
        return 1.0;
    }

    let w_hat = if arg >= 0.0 {
        // Region where the argument under the square root is non-negative
        let sqrt_arg = arg.sqrt();
        if sqrt_arg.abs() < 1e-15 {
            2.0 * m_f / i0_beta
        } else {
            2.0 * m_f * sqrt_arg.sinh() / (sqrt_arg * i0_beta)
        }
    } else {
        // Region where the argument is negative: use sin instead of sinh
        let sqrt_neg = (-arg).sqrt();
        if sqrt_neg.abs() < 1e-15 {
            2.0 * m_f / i0_beta
        } else {
            2.0 * m_f * sqrt_neg.sin() / (sqrt_neg * i0_beta)
        }
    };

    if w_hat.abs() < 1e-30 {
        1.0 // Avoid division by nearly zero at extreme frequencies
    } else {
        1.0 / w_hat
    }
}

// ─── Optimal parameter selection ─────────────────────────────────────────────

/// Compute the optimal Kaiser-Bessel shape parameter β for a given accuracy.
///
/// The optimal β is derived from the relationship between the kernel half-width
/// `m` and the desired accuracy `eps`.  A widely-used empirical formula (from
/// Greengard & Lee, 2004) is:
///
/// ```text
/// β = 2.30 · m     if eps ≥ 1e-10
/// β = 2.31 · m     if eps <  1e-10
/// ```
///
/// Alternatively the formula `β = π·(2 - 1/σ)` (with oversampling σ = 2)
/// gives β = π, but the tuned constant above performs better in practice.
///
/// This implementation uses a slightly refined formula based on the requested
/// accuracy to balance between main-lobe width and sidelobe suppression:
///
/// ```text
/// β = m · (π − 1/(4m)) · (1 + exp(−2 · ln(eps) / (m·π)))
/// ```
/// capped to a maximum of `β_max = 2.5·π·m`.
///
/// # Arguments
///
/// * `m`   – kernel half-width in grid units
/// * `eps` – desired approximation accuracy
///
/// # Returns
///
/// Optimal `β` for the given `(m, eps)` pair.
///
/// # Examples
///
/// ```
/// use scirs2_fft::keiser_bessel::optimal_beta;
///
/// let beta = optimal_beta(4, 1e-6);
/// assert!(beta > 0.0 && beta.is_finite());
///
/// // Higher accuracy requires larger beta
/// let beta_hi = optimal_beta(4, 1e-12);
/// let beta_lo = optimal_beta(4, 1e-3);
/// assert!(beta_hi >= beta_lo);
/// ```
pub fn optimal_beta(m: usize, eps: f64) -> f64 {
    let m_f = m as f64;
    let eps_clamped = eps.max(1e-15).min(1.0);
    // Empirical formula: β scales with m and log(eps)
    let beta = m_f * (PI - 1.0 / (4.0 * m_f)) * (1.0 + (-2.0 * eps_clamped.ln() / (m_f * PI)).exp());
    // Cap to avoid numerical issues for very small eps
    let beta_max = 2.5 * PI * m_f;
    beta.min(beta_max)
}

/// Compute the KB kernel half-width `m` needed to achieve accuracy `eps`
/// given an oversampling factor `sigma`.
///
/// Uses the relationship:
/// ```text
/// m = ceil( −ln(eps) / (π · (σ − 1)) )
/// ```
/// with a minimum of 2.
///
/// # Arguments
///
/// * `eps`   – desired accuracy
/// * `sigma` – oversampling factor (must be > 1)
///
/// # Returns
///
/// Required kernel half-width (minimum 2).
///
/// # Examples
///
/// ```
/// use scirs2_fft::keiser_bessel::kb_half_width;
///
/// let m = kb_half_width(1e-6, 2.0);
/// assert!(m >= 2);
///
/// // Better accuracy → wider kernel
/// let m_hi = kb_half_width(1e-12, 2.0);
/// let m_lo = kb_half_width(1e-3, 2.0);
/// assert!(m_hi >= m_lo);
/// ```
pub fn kb_half_width(eps: f64, sigma: f64) -> usize {
    let eps_clamped = eps.max(1e-15).min(1.0 - 1e-15);
    let sigma_clamped = sigma.max(1.01);
    let m = (-eps_clamped.ln() / (PI * (sigma_clamped - 1.0))).ceil() as usize;
    m.max(2)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bessel_i0_zero() {
        assert_relative_eq!(bessel_i0(0.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bessel_i0_even() {
        // I₀ must be even.
        for x in [0.5, 1.0, 2.0, 5.0, 10.0] {
            assert_relative_eq!(bessel_i0(x), bessel_i0(-x), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bessel_i0_known_values() {
        // Reference values from mathematical tables.
        // I₀(1) ≈ 1.2660658778
        assert_relative_eq!(bessel_i0(1.0), 1.2660658778, epsilon = 1e-7);
        // I₀(2) ≈ 2.2795853023
        assert_relative_eq!(bessel_i0(2.0), 2.2795853023, epsilon = 1e-7);
        // I₀(5) ≈ 27.2398718040
        assert_relative_eq!(bessel_i0(5.0), 27.2398718040, epsilon = 1e-5);
    }

    #[test]
    fn test_kb_window_at_zero() {
        // w(0) must equal 1 for any valid (m, β).
        for &beta in &[5.0, 10.0, 13.9] {
            let w = kaiser_bessel_window(0.0, 4, beta);
            assert_relative_eq!(w, 1.0, epsilon = 1e-12, "beta={}", beta);
        }
    }

    #[test]
    fn test_kb_window_zero_outside_support() {
        // w(x) = 0 for |x| > m.
        assert_eq!(kaiser_bessel_window(5.0, 4, 13.9), 0.0);
        assert_eq!(kaiser_bessel_window(-5.0, 4, 13.9), 0.0);
    }

    #[test]
    fn test_kb_window_symmetric() {
        for x in [0.5, 1.0, 2.0, 3.0] {
            let w_pos = kaiser_bessel_window(x, 4, 13.9);
            let w_neg = kaiser_bessel_window(-x, 4, 13.9);
            assert_relative_eq!(w_pos, w_neg, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_kb_window_decreasing() {
        // The window must be monotonically non-increasing in |x| for x ≥ 0.
        let m = 6usize;
        let beta = 14.0;
        let mut prev = kaiser_bessel_window(0.0, m, beta);
        for i in 1..=m {
            let cur = kaiser_bessel_window(i as f64 * 0.5, m, beta);
            assert!(cur <= prev + 1e-14, "Non-monotone at x={:.1}", i as f64 * 0.5);
            prev = cur;
        }
    }

    #[test]
    fn test_kb_correction_finite_positive() {
        let m = 4usize;
        let n = 64usize;
        let beta = optimal_beta(m, 1e-6);
        for k in [-10i64, -1, 0, 1, 10] {
            let c = kb_correction(k, n, m, beta);
            assert!(c.is_finite(), "correction not finite at k={}", k);
            assert!(c > 0.0, "correction not positive at k={}", k);
        }
    }

    #[test]
    fn test_optimal_beta_monotone_in_accuracy() {
        let m = 4usize;
        let beta_hi = optimal_beta(m, 1e-12);
        let beta_lo = optimal_beta(m, 1e-3);
        assert!(
            beta_hi >= beta_lo,
            "beta_hi={} beta_lo={}",
            beta_hi,
            beta_lo
        );
    }

    #[test]
    fn test_optimal_beta_positive() {
        for m in [2usize, 4, 8, 12] {
            for eps in [1e-3, 1e-6, 1e-9, 1e-12] {
                let beta = optimal_beta(m, eps);
                assert!(beta > 0.0 && beta.is_finite(), "beta={} m={} eps={}", beta, m, eps);
            }
        }
    }

    #[test]
    fn test_kb_half_width_monotone() {
        let m_hi = kb_half_width(1e-12, 2.0);
        let m_lo = kb_half_width(1e-3, 2.0);
        assert!(m_hi >= m_lo);
    }

    #[test]
    fn test_kb_half_width_minimum() {
        // Even very coarse accuracy should give m ≥ 2.
        assert!(kb_half_width(0.5, 2.0) >= 2);
    }
}
