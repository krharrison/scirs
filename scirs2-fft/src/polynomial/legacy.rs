//! # Polynomial Arithmetic with FFT Acceleration
//!
//! This module provides fast polynomial operations using FFT-based algorithms.
//! Key algorithms include:
//!
//! - **FFT multiplication**: O(n log n) instead of naive O(n²)
//! - **Root finding**: via companion matrix eigenvalues (QR iteration)
//! - **Multi-point evaluation**: via divide-and-conquer subproduct tree
//! - **Polynomial interpolation**: via Newton's divided-difference formula
//! - **Chebyshev expansion**: via DCT
//! - **Special polynomials**: Chebyshev T, Legendre P, Hermite H
//!
//! ## Example
//!
//! ```rust
//! use scirs2_fft::polynomial::{Polynomial, poly_multiply, poly_add};
//!
//! let p = Polynomial::new(vec![1.0, 2.0, 1.0]); // 1 + 2x + x²
//! let q = Polynomial::new(vec![1.0, 1.0]);       // 1 + x
//! let product = poly_multiply(&p, &q).expect("multiplication failed");
//! // product = 1 + 3x + 3x² + x³
//! assert!((product.coeffs[0] - 1.0).abs() < 1e-10);
//! assert!((product.coeffs[1] - 3.0).abs() < 1e-10);
//! ```

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Dense polynomial type
// ---------------------------------------------------------------------------

/// Dense polynomial represented as a coefficient vector.
///
/// `P(x) = coeffs[0] + coeffs[1]*x + ... + coeffs[n-1]*x^(n-1)`
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    /// Coefficients in ascending power order.
    pub coeffs: Array1<f64>,
}

impl Polynomial {
    /// Create a polynomial from a coefficient vector.
    ///
    /// # Arguments
    /// * `coeffs` – Coefficients in ascending power order `[a0, a1, ..., an]`.
    ///
    /// # Panics
    /// Panics if `coeffs` is empty; use [`Polynomial::zero`] for the zero polynomial.
    pub fn new(coeffs: Vec<f64>) -> Self {
        assert!(!coeffs.is_empty(), "coefficient vector must be non-empty");
        Self {
            coeffs: Array1::from_vec(coeffs),
        }
    }

    /// The zero polynomial (degree −∞, represented as `[0.0]`).
    pub fn zero() -> Self {
        Self {
            coeffs: Array1::from_vec(vec![0.0]),
        }
    }

    /// The constant polynomial `1`.
    pub fn one() -> Self {
        Self {
            coeffs: Array1::from_vec(vec![1.0]),
        }
    }

    /// The degree of the polynomial (0 for constant, including zero poly).
    pub fn degree(&self) -> usize {
        let n = self.coeffs.len();
        if n == 0 {
            return 0;
        }
        // The leading nonzero coefficient index gives the degree.
        for i in (0..n).rev() {
            if self.coeffs[i].abs() > 0.0 {
                return i;
            }
        }
        0
    }

    /// Evaluate the polynomial at `x` using Horner's method.
    pub fn evaluate(&self, x: f64) -> f64 {
        let n = self.coeffs.len();
        if n == 0 {
            return 0.0;
        }
        let mut result = self.coeffs[n - 1];
        for i in (0..n - 1).rev() {
            result = result * x + self.coeffs[i];
        }
        result
    }

    /// Evaluate the polynomial at a complex point `z` using Horner's method.
    pub fn evaluate_complex(&self, z: Complex64) -> Complex64 {
        let n = self.coeffs.len();
        if n == 0 {
            return Complex64::new(0.0, 0.0);
        }
        let mut result = Complex64::new(self.coeffs[n - 1], 0.0);
        for i in (0..n - 1).rev() {
            result = result * z + Complex64::new(self.coeffs[i], 0.0);
        }
        result
    }

    /// Compute the formal derivative polynomial.
    pub fn derivative(&self) -> Self {
        let n = self.coeffs.len();
        if n <= 1 {
            return Polynomial::zero();
        }
        let deriv: Vec<f64> = (1..n).map(|i| (i as f64) * self.coeffs[i]).collect();
        Self {
            coeffs: Array1::from_vec(deriv),
        }
    }

    /// Compute the antiderivative (indefinite integral), adding constant `c`.
    pub fn antiderivative(&self, c: f64) -> Self {
        let n = self.coeffs.len();
        let mut anti = vec![0.0_f64; n + 1];
        anti[0] = c;
        for i in 0..n {
            anti[i + 1] = self.coeffs[i] / ((i + 1) as f64);
        }
        Self {
            coeffs: Array1::from_vec(anti),
        }
    }

    /// Compute the definite integral from `a` to `b`.
    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        let anti = self.antiderivative(0.0);
        anti.evaluate(b) - anti.evaluate(a)
    }

    /// Trim trailing (leading-in-power) near-zero coefficients.
    pub fn trim(mut self) -> Self {
        let tol = 1e-14_f64;
        let n = self.coeffs.len();
        let mut end = n;
        while end > 1 && self.coeffs[end - 1].abs() <= tol {
            end -= 1;
        }
        if end < n {
            let new_coeffs: Vec<f64> = self.coeffs.iter().take(end).copied().collect();
            self.coeffs = Array1::from_vec(new_coeffs);
        }
        self
    }

    /// Build a polynomial from its real roots: `(x - r1)(x - r2)...(x - rn)`.
    pub fn from_roots(roots: &[f64]) -> Self {
        let mut p = Polynomial::one();
        for &r in roots {
            // Multiply by (x - r)
            let factor = Polynomial::new(vec![-r, 1.0]);
            // Use naive multiplication for factors (each is degree 1)
            p = poly_multiply_naive(&p, &factor);
        }
        p
    }

    /// Find all roots by building the companion matrix and performing
    /// Francis double-shift QR iteration to extract eigenvalues.
    pub fn roots(&self) -> FFTResult<Vec<Complex64>> {
        let trimmed = self.clone().trim();
        let n = trimmed.degree();
        if n == 0 {
            return Ok(vec![]);
        }
        if n == 1 {
            // Linear: a0 + a1*x = 0  =>  x = -a0/a1
            let a0 = trimmed.coeffs[0];
            let a1 = trimmed.coeffs[1];
            if a1.abs() < 1e-300 {
                return Err(FFTError::ComputationError(
                    "leading coefficient is zero".to_string(),
                ));
            }
            return Ok(vec![Complex64::new(-a0 / a1, 0.0)]);
        }
        if n == 2 {
            return roots_quadratic(&trimmed);
        }

        // Build companion matrix (n×n) for monic polynomial
        // The companion matrix for p(x) = x^n + c_{n-1}x^{n-1} + ... + c_0 is:
        // Last column: [-c0/cn, -c1/cn, ..., -c_{n-1}/cn]
        // Sub-diagonal: [1, 1, ..., 1]
        let cn = trimmed.coeffs[n];
        if cn.abs() < 1e-300 {
            return Err(FFTError::ComputationError(
                "leading coefficient is zero after trim".to_string(),
            ));
        }

        // Use Hessenberg QR iteration on companion matrix
        companion_eigenvalues(&trimmed, n, cn)
    }
}

// ---------------------------------------------------------------------------
// Quadratic root helper
// ---------------------------------------------------------------------------

fn roots_quadratic(p: &Polynomial) -> FFTResult<Vec<Complex64>> {
    // p(x) = a + b*x + c*x²
    let a = p.coeffs[0];
    let b = p.coeffs[1];
    let c = p.coeffs[2];
    if c.abs() < 1e-300 {
        return Err(FFTError::ComputationError(
            "leading coefficient is zero".to_string(),
        ));
    }
    let disc = b * b - 4.0 * a * c;
    if disc >= 0.0 {
        let sqrt_d = disc.sqrt();
        Ok(vec![
            Complex64::new((-b + sqrt_d) / (2.0 * c), 0.0),
            Complex64::new((-b - sqrt_d) / (2.0 * c), 0.0),
        ])
    } else {
        let sqrt_d = (-disc).sqrt();
        Ok(vec![
            Complex64::new(-b / (2.0 * c), sqrt_d / (2.0 * c)),
            Complex64::new(-b / (2.0 * c), -sqrt_d / (2.0 * c)),
        ])
    }
}

// ---------------------------------------------------------------------------
// Companion matrix QR iteration for root finding
// ---------------------------------------------------------------------------

/// Compute eigenvalues of the companion matrix via Francis QR iteration.
fn companion_eigenvalues(p: &Polynomial, n: usize, cn: f64) -> FFTResult<Vec<Complex64>> {
    // Build upper Hessenberg companion matrix
    let mut h = vec![vec![0.0_f64; n]; n];
    // Sub-diagonal ones
    for i in 1..n {
        h[i][i - 1] = 1.0;
    }
    // Last column: -c_k / c_n (for k = 0..n-1)
    for i in 0..n {
        h[i][n - 1] = -p.coeffs[i] / cn;
    }

    // Francis double-shift QR on H (stored as real upper Hessenberg).
    // We maintain a complex Schur form via repeated single/double shift steps.
    // For simplicity, use the real Francis double-shift algorithm.
    let max_iter = 30 * n;
    let eps = f64::EPSILON * 10.0;

    let mut q_shift_count = 0usize;
    let mut eigenvalues: Vec<Complex64> = Vec::with_capacity(n);

    // Work on sub-matrix H[lo..=hi, lo..=hi]
    let mut active_hi = n - 1;

    'outer: while active_hi > 0 {
        // Check for small sub-diagonal entries
        let mut found_deflation = false;
        for i in (1..=active_hi).rev() {
            if h[i][i - 1].abs() <= eps * (h[i - 1][i - 1].abs() + h[i][i].abs()) {
                h[i][i - 1] = 0.0;
                if i == active_hi {
                    // Eigenvalue found
                    eigenvalues.push(Complex64::new(h[active_hi][active_hi], 0.0));
                    active_hi -= 1;
                    found_deflation = true;
                    break;
                }
                // 2×2 block at bottom
                if i == active_hi - 1 {
                    let (e1, e2) = eig2x2(
                        h[active_hi - 1][active_hi - 1],
                        h[active_hi - 1][active_hi],
                        h[active_hi][active_hi - 1],
                        h[active_hi][active_hi],
                    );
                    eigenvalues.push(e1);
                    eigenvalues.push(e2);
                    if active_hi < 2 {
                        break 'outer;
                    }
                    active_hi -= 2;
                    found_deflation = true;
                    break;
                }
            }
        }
        if found_deflation {
            q_shift_count = 0;
            continue;
        }

        q_shift_count += 1;
        if q_shift_count > max_iter {
            // Push remaining diagonal as approximate eigenvalues
            for i in 0..=active_hi {
                eigenvalues.push(Complex64::new(h[i][i], 0.0));
            }
            break;
        }

        // Francis double-shift: compute shifts from 2×2 trailing sub-matrix
        let m = active_hi;
        let s = h[m][m] + h[m - 1][m - 1];
        let t = h[m][m] * h[m - 1][m - 1] - h[m][m - 1] * h[m - 1][m];

        // Compute first column of (H² - sH + tI)
        let lo = if active_hi >= 2 { active_hi - 2 } else { 0 };
        let x = h[lo][lo] * h[lo][lo] + h[lo][lo + 1] * h[lo + 1][lo] - s * h[lo][lo] + t;
        let y = h[lo + 1][lo] * (h[lo][lo] + h[lo + 1][lo + 1] - s);
        let z = if lo + 2 <= active_hi {
            h[lo + 2][lo + 1] * h[lo + 1][lo]
        } else {
            0.0
        };

        // Apply Householder reflectors to chase the bulge
        francis_double_shift_step(&mut h, active_hi, lo, x, y, z);
    }

    // Collect any remaining 1×1 block
    if active_hi == 0 {
        eigenvalues.push(Complex64::new(h[0][0], 0.0));
    }

    Ok(eigenvalues)
}

/// 2×2 eigenvalues.
fn eig2x2(a00: f64, a01: f64, a10: f64, a11: f64) -> (Complex64, Complex64) {
    let tr = a00 + a11;
    let det = a00 * a11 - a01 * a10;
    let disc = tr * tr - 4.0 * det;
    if disc >= 0.0 {
        let s = disc.sqrt();
        (
            Complex64::new((tr + s) / 2.0, 0.0),
            Complex64::new((tr - s) / 2.0, 0.0),
        )
    } else {
        let s = (-disc).sqrt();
        (
            Complex64::new(tr / 2.0, s / 2.0),
            Complex64::new(tr / 2.0, -s / 2.0),
        )
    }
}

/// Apply one Francis double-shift QR step.
fn francis_double_shift_step(
    h: &mut Vec<Vec<f64>>,
    hi: usize,
    lo: usize,
    x: f64,
    y: f64,
    z: f64,
) {
    let n = h.len();
    let mut v = [x, y, z];

    // Normalize
    let norm_v = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm_v < f64::EPSILON {
        return;
    }
    v[0] /= norm_v;
    v[1] /= norm_v;
    v[2] /= norm_v;

    // Chase the bulge from column lo to hi-1
    let mut k = lo;
    loop {
        let r = k.min(hi);
        let c_len = 3.min(hi - k + 1);

        // Apply Householder from the left
        let row_end = (k + c_len + 1).min(n);
        for j in k..row_end {
            let mut dot = 0.0;
            for i in 0..c_len {
                if k + i < n {
                    dot += v[i] * h[k + i][j];
                }
            }
            dot *= 2.0;
            for i in 0..c_len {
                if k + i < n {
                    h[k + i][j] -= dot * v[i];
                }
            }
        }

        // Apply Householder from the right
        let row_start = if k >= 1 { k - 1 } else { 0 };
        let row_stop = (k + c_len + 1).min(n);
        for i in row_start..row_stop {
            let mut dot = 0.0;
            for j in 0..c_len {
                if k + j < n {
                    dot += v[j] * h[i][k + j];
                }
            }
            dot *= 2.0;
            for j in 0..c_len {
                if k + j < n {
                    h[i][k + j] -= dot * v[j];
                }
            }
        }

        // Determine next bulge position
        if k + 1 >= r || c_len < 2 {
            break;
        }

        // Compute new Householder vector for next step
        let x2 = h[k + 1][k];
        let y2 = if k + 2 <= hi { h[k + 2][k] } else { 0.0 };
        let z2 = if k + 3 <= hi { h[k + 3][k] } else { 0.0 };
        let norm2 = (x2 * x2 + y2 * y2 + z2 * z2).sqrt();
        if norm2 < f64::EPSILON {
            break;
        }
        v[0] = x2 / norm2;
        v[1] = y2 / norm2;
        v[2] = z2 / norm2;
        k += 1;
    }
}

// ---------------------------------------------------------------------------
// Naive polynomial multiplication (used for small polynomials)
// ---------------------------------------------------------------------------

fn poly_multiply_naive(p: &Polynomial, q: &Polynomial) -> Polynomial {
    let np = p.coeffs.len();
    let nq = q.coeffs.len();
    let result_len = np + nq - 1;
    let mut result = vec![0.0_f64; result_len];
    for i in 0..np {
        for j in 0..nq {
            result[i + j] += p.coeffs[i] * q.coeffs[j];
        }
    }
    Polynomial {
        coeffs: Array1::from_vec(result),
    }
}

// ---------------------------------------------------------------------------
// FFT-based polynomial multiplication
// ---------------------------------------------------------------------------

/// Multiply two polynomials using FFT in O(n log n).
///
/// # Example
/// ```rust
/// use scirs2_fft::polynomial::{Polynomial, poly_multiply};
///
/// let p = Polynomial::new(vec![1.0, 1.0]); // 1 + x
/// let q = Polynomial::new(vec![1.0, 1.0]); // 1 + x
/// let r = poly_multiply(&p, &q).expect("multiply failed");
/// // r = (1+x)² = 1 + 2x + x²
/// assert!((r.coeffs[0] - 1.0).abs() < 1e-10);
/// assert!((r.coeffs[1] - 2.0).abs() < 1e-10);
/// assert!((r.coeffs[2] - 1.0).abs() < 1e-10);
/// ```
pub fn poly_multiply(p: &Polynomial, q: &Polynomial) -> FFTResult<Polynomial> {
    let np = p.coeffs.len();
    let nq = q.coeffs.len();

    // For very small polynomials, naive is faster
    if np <= 32 || nq <= 32 {
        return Ok(poly_multiply_naive(p, q));
    }

    let result_len = np + nq - 1;
    // Find next power of two >= result_len
    let fft_size = result_len.next_power_of_two();

    // Zero-pad coefficient vectors
    let mut p_padded: Vec<f64> = p.coeffs.iter().copied().collect();
    let mut q_padded: Vec<f64> = q.coeffs.iter().copied().collect();
    p_padded.resize(fft_size, 0.0);
    q_padded.resize(fft_size, 0.0);

    // FFT of each
    let p_freq = fft(&p_padded, Some(fft_size))?;
    let q_freq = fft(&q_padded, Some(fft_size))?;

    // Pointwise multiply in frequency domain
    let product_freq: Vec<Complex64> = p_freq
        .iter()
        .zip(q_freq.iter())
        .map(|(a, b)| Complex64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re))
        .collect();

    // IFFT and extract real part
    let product_time = ifft(&product_freq, Some(fft_size))?;

    // Take only the first result_len coefficients and round small imaginaries
    let result_coeffs: Vec<f64> = product_time
        .iter()
        .take(result_len)
        .map(|c| c.re)
        .collect();

    Ok(Polynomial {
        coeffs: Array1::from_vec(result_coeffs),
    })
}

// ---------------------------------------------------------------------------
// Polynomial addition / subtraction
// ---------------------------------------------------------------------------

/// Add two polynomials.
///
/// # Example
/// ```rust
/// use scirs2_fft::polynomial::{Polynomial, poly_add};
///
/// let p = Polynomial::new(vec![1.0, 2.0]);
/// let q = Polynomial::new(vec![3.0, 4.0, 5.0]);
/// let r = poly_add(&p, &q);
/// assert!((r.coeffs[0] - 4.0).abs() < 1e-14);
/// assert!((r.coeffs[1] - 6.0).abs() < 1e-14);
/// assert!((r.coeffs[2] - 5.0).abs() < 1e-14);
/// ```
pub fn poly_add(p: &Polynomial, q: &Polynomial) -> Polynomial {
    let np = p.coeffs.len();
    let nq = q.coeffs.len();
    let result_len = np.max(nq);
    let mut result = vec![0.0_f64; result_len];
    for i in 0..np {
        result[i] += p.coeffs[i];
    }
    for i in 0..nq {
        result[i] += q.coeffs[i];
    }
    Polynomial {
        coeffs: Array1::from_vec(result),
    }
}

/// Subtract polynomial `q` from `p`.
///
/// # Example
/// ```rust
/// use scirs2_fft::polynomial::{Polynomial, poly_sub};
///
/// let p = Polynomial::new(vec![5.0, 6.0, 7.0]);
/// let q = Polynomial::new(vec![1.0, 2.0]);
/// let r = poly_sub(&p, &q);
/// assert!((r.coeffs[0] - 4.0).abs() < 1e-14);
/// assert!((r.coeffs[1] - 4.0).abs() < 1e-14);
/// assert!((r.coeffs[2] - 7.0).abs() < 1e-14);
/// ```
pub fn poly_sub(p: &Polynomial, q: &Polynomial) -> Polynomial {
    let np = p.coeffs.len();
    let nq = q.coeffs.len();
    let result_len = np.max(nq);
    let mut result = vec![0.0_f64; result_len];
    for i in 0..np {
        result[i] += p.coeffs[i];
    }
    for i in 0..nq {
        result[i] -= q.coeffs[i];
    }
    Polynomial {
        coeffs: Array1::from_vec(result),
    }
}

// ---------------------------------------------------------------------------
// Polynomial long division
// ---------------------------------------------------------------------------

/// Polynomial long division: returns `(quotient, remainder)` such that
/// `dividend = quotient * divisor + remainder` and
/// `deg(remainder) < deg(divisor)`.
///
/// # Errors
/// Returns an error if `divisor` is the zero polynomial.
///
/// # Example
/// ```rust
/// use scirs2_fft::polynomial::{Polynomial, poly_divmod};
///
/// // x² - 1 = (x-1)(x+1) + 0
/// let dividend = Polynomial::new(vec![-1.0, 0.0, 1.0]); // x² - 1
/// let divisor  = Polynomial::new(vec![-1.0, 1.0]);       // x - 1
/// let (q, r) = poly_divmod(&dividend, &divisor).expect("divmod failed");
/// // quotient = x + 1
/// assert!((q.coeffs[0] - 1.0).abs() < 1e-10);
/// assert!((q.coeffs[1] - 1.0).abs() < 1e-10);
/// ```
pub fn poly_divmod(
    dividend: &Polynomial,
    divisor: &Polynomial,
) -> FFTResult<(Polynomial, Polynomial)> {
    let divisor_trimmed = divisor.clone().trim();
    let dividend_trimmed = dividend.clone().trim();

    let d_deg = divisor_trimmed.degree();
    let n_deg = dividend_trimmed.degree();

    // Check for zero divisor
    if divisor_trimmed
        .coeffs
        .iter()
        .all(|&c| c.abs() < f64::EPSILON)
    {
        return Err(FFTError::ValueError(
            "division by zero polynomial".to_string(),
        ));
    }

    if n_deg < d_deg {
        // Quotient is zero, remainder is the dividend
        return Ok((Polynomial::zero(), dividend_trimmed));
    }

    let mut remainder: Vec<f64> = dividend_trimmed.coeffs.iter().copied().collect();
    let mut quotient_coeffs = vec![0.0_f64; n_deg - d_deg + 1];

    let leading_d = divisor_trimmed.coeffs[d_deg];

    // Classical polynomial long division
    for i in (0..=n_deg - d_deg).rev() {
        let coeff = remainder[i + d_deg] / leading_d;
        quotient_coeffs[i] = coeff;
        for j in 0..=d_deg {
            remainder[i + j] -= coeff * divisor_trimmed.coeffs[j];
        }
    }

    let quotient = Polynomial {
        coeffs: Array1::from_vec(quotient_coeffs),
    }
    .trim();
    let rem = Polynomial {
        coeffs: Array1::from_vec(remainder[..d_deg].to_vec()),
    }
    .trim();

    Ok((quotient, rem))
}

// ---------------------------------------------------------------------------
// Polynomial GCD
// ---------------------------------------------------------------------------

/// Compute GCD of two polynomials via the Euclidean algorithm.
///
/// # Example
/// ```rust
/// use scirs2_fft::polynomial::{Polynomial, poly_gcd};
///
/// // gcd(x² - 1, x - 1) = x - 1
/// let p = Polynomial::new(vec![-1.0, 0.0, 1.0]);
/// let q = Polynomial::new(vec![-1.0, 1.0]);
/// let g = poly_gcd(&p, &q).expect("gcd failed");
/// // Monic GCD: x - 1
/// let leading = g.coeffs[g.degree()];
/// assert!(leading.abs() > 1e-12);
/// ```
pub fn poly_gcd(p: &Polynomial, q: &Polynomial) -> FFTResult<Polynomial> {
    let mut a = p.clone().trim();
    let mut b = q.clone().trim();

    // Euclidean algorithm
    loop {
        if b.coeffs.iter().all(|&c| c.abs() < 1e-12) {
            break;
        }
        let (_, r) = poly_divmod(&a, &b)?;
        a = b;
        b = r.trim();
    }

    // Normalise to monic
    let deg = a.degree();
    let leading = a.coeffs[deg];
    if leading.abs() > 1e-14 {
        let new_coeffs: Vec<f64> = a.coeffs.iter().map(|&c| c / leading).collect();
        a.coeffs = Array1::from_vec(new_coeffs);
    }

    Ok(a)
}

// ---------------------------------------------------------------------------
// Polynomial modular exponentiation
// ---------------------------------------------------------------------------

/// Compute `p^n mod m` using fast exponentiation.
///
/// # Example
/// ```rust
/// use scirs2_fft::polynomial::{Polynomial, poly_powmod};
///
/// // (x)^3 mod (x^2 - 1)
/// let p = Polynomial::new(vec![0.0, 1.0]);         // x
/// let m = Polynomial::new(vec![-1.0, 0.0, 1.0]);   // x² - 1
/// let result = poly_powmod(&p, 3, &m).expect("powmod failed");
/// // x³ mod (x²-1) = x*(x²-1) + x = x, so result ≈ [0,1]
/// ```
pub fn poly_powmod(p: &Polynomial, n: usize, m: &Polynomial) -> FFTResult<Polynomial> {
    if n == 0 {
        return Ok(Polynomial::one());
    }

    let mut result = Polynomial::one();
    let mut base = {
        let (_, r) = poly_divmod(p, m)?;
        r
    };

    let mut exp = n;
    while exp > 0 {
        if exp & 1 == 1 {
            let prod = poly_multiply(&result, &base)?;
            let (_, r) = poly_divmod(&prod, m)?;
            result = r;
        }
        exp >>= 1;
        if exp > 0 {
            let sq = poly_multiply(&base, &base)?;
            let (_, r) = poly_divmod(&sq, m)?;
            base = r;
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Multi-point evaluation
// ---------------------------------------------------------------------------

/// Evaluate a polynomial at multiple points efficiently.
///
/// Uses a subproduct tree for O(n log² n) when n ≥ 64, falling back to
/// direct Horner evaluation for smaller inputs.
///
/// # Example
/// ```rust
/// use scirs2_fft::polynomial::{Polynomial, poly_multipoint_eval};
///
/// let p = Polynomial::new(vec![1.0, 2.0, 1.0]); // 1 + 2x + x²
/// let pts = vec![0.0, 1.0, 2.0, -1.0];
/// let vals = poly_multipoint_eval(&p, &pts).expect("eval failed");
/// assert!((vals[0] - 1.0).abs() < 1e-10);  // p(0) = 1
/// assert!((vals[1] - 4.0).abs() < 1e-10);  // p(1) = 4
/// assert!((vals[2] - 9.0).abs() < 1e-10);  // p(2) = 9
/// assert!((vals[3] - 0.0).abs() < 1e-10);  // p(-1) = 0
/// ```
pub fn poly_multipoint_eval(p: &Polynomial, points: &[f64]) -> FFTResult<Vec<f64>> {
    if points.is_empty() {
        return Ok(vec![]);
    }
    let n = points.len();

    // For small inputs use direct Horner evaluation
    if n < 64 || p.coeffs.len() < 64 {
        return Ok(points.iter().map(|&x| p.evaluate(x)).collect());
    }

    // Subproduct tree approach
    multipoint_eval_subproduct(p, points)
}

/// Recursive subproduct tree multi-point evaluation.
fn multipoint_eval_subproduct(p: &Polynomial, points: &[f64]) -> FFTResult<Vec<f64>> {
    let n = points.len();
    if n == 1 {
        return Ok(vec![p.evaluate(points[0])]);
    }
    if n <= 8 {
        return Ok(points.iter().map(|&x| p.evaluate(x)).collect());
    }

    let mid = n / 2;
    let left_pts = &points[..mid];
    let right_pts = &points[mid..];

    // Build subproducts
    let m0 = build_subproduct(left_pts)?;
    let m1 = build_subproduct(right_pts)?;

    // Reduce: p mod m0 and p mod m1
    let (_, r0) = poly_divmod(p, &m0)?;
    let (_, r1) = poly_divmod(p, &m1)?;

    // Recurse
    let mut left_vals = multipoint_eval_subproduct(&r0, left_pts)?;
    let right_vals = multipoint_eval_subproduct(&r1, right_pts)?;

    left_vals.extend(right_vals);
    Ok(left_vals)
}

/// Build the product polynomial (x - x0)(x - x1)...(x - xk).
fn build_subproduct(points: &[f64]) -> FFTResult<Polynomial> {
    if points.is_empty() {
        return Ok(Polynomial::one());
    }
    if points.len() == 1 {
        return Ok(Polynomial::new(vec![-points[0], 1.0]));
    }

    let mid = points.len() / 2;
    let left = build_subproduct(&points[..mid])?;
    let right = build_subproduct(&points[mid..])?;
    poly_multiply(&left, &right)
}

// ---------------------------------------------------------------------------
// Polynomial interpolation
// ---------------------------------------------------------------------------

/// Interpolate a polynomial from `(x, y)` value pairs using Newton's
/// divided-difference formula, accelerated by FFT multiplication for the
/// subproduct tree.
///
/// # Errors
/// Returns an error if `x` and `y` have different lengths, or if `x` contains
/// duplicate values.
///
/// # Example
/// ```rust
/// use scirs2_fft::polynomial::{Polynomial, poly_interpolate};
///
/// // Interpolate x² at points 0, 1, 2
/// let x = vec![0.0, 1.0, 2.0];
/// let y = vec![0.0, 1.0, 4.0];
/// let p = poly_interpolate(&x, &y).expect("interpolation failed");
/// assert!((p.evaluate(0.5) - 0.25).abs() < 1e-9);
/// assert!((p.evaluate(1.5) - 2.25).abs() < 1e-9);
/// ```
pub fn poly_interpolate(x: &[f64], y: &[f64]) -> FFTResult<Polynomial> {
    if x.len() != y.len() {
        return Err(FFTError::DimensionError(
            "x and y must have the same length".to_string(),
        ));
    }
    let n = x.len();
    if n == 0 {
        return Ok(Polynomial::zero());
    }
    if n == 1 {
        return Ok(Polynomial::new(vec![y[0]]));
    }

    // Verify all x are distinct
    for i in 0..n {
        for j in i + 1..n {
            if (x[i] - x[j]).abs() < 1e-14 {
                return Err(FFTError::ValueError(
                    "interpolation points must be distinct".to_string(),
                ));
            }
        }
    }

    // Newton divided differences
    let mut dd = y.to_vec();
    for i in 1..n {
        for j in (i..n).rev() {
            dd[j] = (dd[j] - dd[j - 1]) / (x[j] - x[j - i]);
        }
    }

    // Build Newton basis polynomials and accumulate
    // P(x) = dd[0] + dd[1]*(x-x0) + dd[2]*(x-x0)*(x-x1) + ...
    let mut result = Polynomial::new(vec![dd[0]]);
    let mut basis = Polynomial::one();

    for i in 1..n {
        // basis *= (x - x[i-1])
        let factor = Polynomial::new(vec![-x[i - 1], 1.0]);
        basis = poly_multiply(&basis, &factor)?;
        // term = dd[i] * basis
        let term_coeffs: Vec<f64> = basis.coeffs.iter().map(|&c| c * dd[i]).collect();
        let term = Polynomial {
            coeffs: Array1::from_vec(term_coeffs),
        };
        result = poly_add(&result, &term);
    }

    Ok(result.trim())
}

// ---------------------------------------------------------------------------
// Polynomial exponentiation
// ---------------------------------------------------------------------------

/// Compute `p^n` via repeated squaring using FFT multiplication.
///
/// # Example
/// ```rust
/// use scirs2_fft::polynomial::{Polynomial, poly_pow};
///
/// let p = Polynomial::new(vec![1.0, 1.0]); // 1 + x
/// let p3 = poly_pow(&p, 3).expect("pow failed");
/// // (1+x)³ = 1 + 3x + 3x² + x³
/// assert!((p3.coeffs[0] - 1.0).abs() < 1e-10);
/// assert!((p3.coeffs[1] - 3.0).abs() < 1e-10);
/// assert!((p3.coeffs[2] - 3.0).abs() < 1e-10);
/// assert!((p3.coeffs[3] - 1.0).abs() < 1e-10);
/// ```
pub fn poly_pow(p: &Polynomial, n: usize) -> FFTResult<Polynomial> {
    if n == 0 {
        return Ok(Polynomial::one());
    }
    if n == 1 {
        return Ok(p.clone());
    }

    let mut result = Polynomial::one();
    let mut base = p.clone();
    let mut exp = n;

    while exp > 0 {
        if exp & 1 == 1 {
            result = poly_multiply(&result, &base)?;
        }
        exp >>= 1;
        if exp > 0 {
            base = poly_multiply(&base, &base)?;
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Polynomial composition
// ---------------------------------------------------------------------------

/// Compute the composition `p(q(x))`.
///
/// # Example
/// ```rust
/// use scirs2_fft::polynomial::{Polynomial, poly_compose};
///
/// let p = Polynomial::new(vec![0.0, 0.0, 1.0]); // x²
/// let q = Polynomial::new(vec![1.0, 1.0]);       // 1 + x
/// let r = poly_compose(&p, &q).expect("compose failed");
/// // (1+x)² = 1 + 2x + x²
/// assert!((r.coeffs[0] - 1.0).abs() < 1e-10);
/// assert!((r.coeffs[1] - 2.0).abs() < 1e-10);
/// assert!((r.coeffs[2] - 1.0).abs() < 1e-10);
/// ```
pub fn poly_compose(p: &Polynomial, q: &Polynomial) -> FFTResult<Polynomial> {
    let n = p.coeffs.len();
    if n == 0 {
        return Ok(Polynomial::zero());
    }

    // Horner's method for composition:
    // p(q(x)) = (...((p[n-1]*q + p[n-2])*q + p[n-3])*q + ...) + p[0]
    let mut result = Polynomial::new(vec![p.coeffs[n - 1]]);
    for i in (0..n - 1).rev() {
        result = poly_multiply(&result, q)?;
        result = poly_add(&result, &Polynomial::new(vec![p.coeffs[i]]));
    }

    Ok(result.trim())
}

// ---------------------------------------------------------------------------
// Special polynomial families
// ---------------------------------------------------------------------------

/// Compute the Chebyshev polynomial of the first kind `T_n(x)`.
///
/// Uses the three-term recurrence: T₀ = 1, T₁ = x, Tₙ = 2x·Tₙ₋₁ - Tₙ₋₂.
///
/// # Example
/// ```rust
/// use scirs2_fft::polynomial::chebyshev_t;
///
/// let t2 = chebyshev_t(2); // T₂(x) = 2x² - 1
/// assert!((t2.evaluate(0.0) - (-1.0)).abs() < 1e-14);
/// assert!((t2.evaluate(1.0) - 1.0).abs() < 1e-14);
/// ```
pub fn chebyshev_t(n: usize) -> Polynomial {
    if n == 0 {
        return Polynomial::one();
    }
    if n == 1 {
        return Polynomial::new(vec![0.0, 1.0]);
    }

    let mut t_prev = Polynomial::one();
    let mut t_curr = Polynomial::new(vec![0.0, 1.0]);

    // x polynomial
    let x_poly = Polynomial::new(vec![0.0, 1.0]);

    for _ in 2..=n {
        // T_n = 2x T_{n-1} - T_{n-2}
        let two_x_tcurr = poly_multiply_naive(
            &Polynomial::new(vec![0.0, 2.0]),
            &poly_multiply_naive(&x_poly, &t_curr),
        );
        // Actually: 2*x*T_curr
        let two_x_tcurr = {
            let tmp = poly_multiply_naive(&x_poly, &t_curr);
            // Multiply by 2
            let coeffs: Vec<f64> = tmp.coeffs.iter().map(|&c| 2.0 * c).collect();
            Polynomial {
                coeffs: Array1::from_vec(coeffs),
            }
        };
        let t_next = poly_sub(&two_x_tcurr, &t_prev);
        t_prev = t_curr;
        t_curr = t_next;
    }

    t_curr
}

/// Compute the Legendre polynomial `P_n(x)`.
///
/// Uses the three-term recurrence:
/// P₀ = 1, P₁ = x, (n+1)Pₙ₊₁ = (2n+1)x·Pₙ - n·Pₙ₋₁
///
/// # Example
/// ```rust
/// use scirs2_fft::polynomial::legendre_p;
///
/// let p2 = legendre_p(2); // P₂(x) = (3x² - 1) / 2
/// assert!((p2.evaluate(0.0) - (-0.5)).abs() < 1e-14);
/// assert!((p2.evaluate(1.0) - 1.0).abs() < 1e-14);
/// ```
pub fn legendre_p(n: usize) -> Polynomial {
    if n == 0 {
        return Polynomial::one();
    }
    if n == 1 {
        return Polynomial::new(vec![0.0, 1.0]);
    }

    let mut p_prev = Polynomial::one();
    let mut p_curr = Polynomial::new(vec![0.0, 1.0]);
    let x_poly = Polynomial::new(vec![0.0, 1.0]);

    for k in 1..n {
        // (k+1) P_{k+1} = (2k+1) x P_k - k P_{k-1}
        let factor1 = (2 * k + 1) as f64;
        let factor2 = k as f64;
        let denom = (k + 1) as f64;

        let term1 = {
            let tmp = poly_multiply_naive(&x_poly, &p_curr);
            let coeffs: Vec<f64> = tmp.coeffs.iter().map(|&c| c * factor1 / denom).collect();
            Polynomial {
                coeffs: Array1::from_vec(coeffs),
            }
        };
        let term2 = {
            let coeffs: Vec<f64> = p_prev.coeffs.iter().map(|&c| c * factor2 / denom).collect();
            Polynomial {
                coeffs: Array1::from_vec(coeffs),
            }
        };
        let p_next = poly_sub(&term1, &term2);
        p_prev = p_curr;
        p_curr = p_next;
    }

    p_curr
}

/// Compute the probabilist's Hermite polynomial `He_n(x)`.
///
/// Uses the recurrence: He₀ = 1, He₁ = x, Heₙ = x·Heₙ₋₁ - (n-1)·Heₙ₋₂
///
/// # Example
/// ```rust
/// use scirs2_fft::polynomial::hermite_prob;
///
/// let h3 = hermite_prob(3); // He₃(x) = x³ - 3x
/// assert!((h3.evaluate(0.0) - 0.0).abs() < 1e-14);
/// assert!((h3.evaluate(1.0) - (-2.0)).abs() < 1e-14);
/// ```
pub fn hermite_prob(n: usize) -> Polynomial {
    if n == 0 {
        return Polynomial::one();
    }
    if n == 1 {
        return Polynomial::new(vec![0.0, 1.0]);
    }

    let mut h_prev = Polynomial::one();
    let mut h_curr = Polynomial::new(vec![0.0, 1.0]);
    let x_poly = Polynomial::new(vec![0.0, 1.0]);

    for k in 2..=n {
        // He_k = x * He_{k-1} - (k-1) * He_{k-2}
        let term1 = poly_multiply_naive(&x_poly, &h_curr);
        let factor = (k - 1) as f64;
        let term2 = {
            let coeffs: Vec<f64> = h_prev.coeffs.iter().map(|&c| c * factor).collect();
            Polynomial {
                coeffs: Array1::from_vec(coeffs),
            }
        };
        let h_next = poly_sub(&term1, &term2);
        h_prev = h_curr;
        h_curr = h_next;
    }

    h_curr
}

// ---------------------------------------------------------------------------
// Chebyshev expansion
// ---------------------------------------------------------------------------

/// Compute Chebyshev expansion coefficients from function values at Chebyshev nodes.
///
/// Given values `f_values[k] = f(cos(πk/N))` for `k = 0..=N`, compute the
/// Chebyshev expansion coefficients `c_k` such that
/// `f(x) ≈ Σ c_k T_k(x)` (with c_0 and c_N halved in the sum convention).
///
/// This uses a DCT-I style computation via FFT.
///
/// # Example
/// ```rust
/// use scirs2_fft::polynomial::{chebyshev_expansion, chebyshev_eval, chebyshev_nodes};
///
/// // f(x) = x² on [-1, 1]
/// let n = 16;
/// let nodes = chebyshev_nodes(n, -1.0, 1.0);
/// let f_vals: Vec<f64> = nodes.iter().map(|&x| x * x).collect();
/// let coeffs = chebyshev_expansion(&f_vals);
/// // Evaluate at an arbitrary point and compare to x²
/// let x = 0.5;
/// let approx = chebyshev_eval(&coeffs, x);
/// assert!((approx - x * x).abs() < 1e-10);
/// ```
pub fn chebyshev_expansion(f_values: &[f64]) -> Vec<f64> {
    let n = f_values.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![f_values[0]];
    }

    // Build the symmetric extension for DCT-I: [f0, f1, ..., fn-1, fn-2, ..., f1]
    // This gives us a real-valued DFT of length 2*(n-1) whose real part is the DCT-I.
    let m = n - 1;
    if m == 0 {
        return vec![f_values[0]];
    }

    let ext_len = 2 * m;
    let mut ext: Vec<f64> = Vec::with_capacity(ext_len);
    ext.extend_from_slice(f_values);
    // Mirror: f[m-1], f[m-2], ..., f[1]
    for i in (1..m).rev() {
        ext.push(f_values[i]);
    }

    // FFT of the extended signal
    let spectrum = match fft(&ext, Some(ext_len)) {
        Ok(s) => s,
        Err(_) => {
            // Fallback: direct DCT-I
            return dct1_direct(f_values);
        }
    };

    // Extract DCT-I coefficients: c_k = Re(FFT[k]) / m
    let mut coeffs: Vec<f64> = (0..n).map(|k| spectrum[k].re / (m as f64)).collect();

    // Scale: c_0 and c_{n-1} are halved in the standard convention
    coeffs[0] /= 2.0;
    if n > 1 {
        coeffs[n - 1] /= 2.0;
    }

    coeffs
}

/// Direct DCT-I computation (fallback).
fn dct1_direct(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    if n == 0 {
        return vec![];
    }
    let m = (n - 1) as f64;
    let mut c = vec![0.0_f64; n];
    for k in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            let angle = PI * (k as f64) * (j as f64) / m;
            sum += f[j] * angle.cos();
        }
        c[k] = sum / m;
    }
    if n > 1 {
        c[0] /= 2.0;
        c[n - 1] /= 2.0;
    }
    c
}

/// Evaluate a Chebyshev expansion at a point `x ∈ [-1, 1]`.
///
/// Uses Clenshaw's recurrence for numerical stability.
///
/// # Example
/// ```rust
/// use scirs2_fft::polynomial::chebyshev_eval;
///
/// // T₂(x) = 2x² - 1 → coeffs = [0, 0, 1] in Chebyshev basis
/// let cheb_coeffs = vec![0.0, 0.0, 1.0];
/// let val = chebyshev_eval(&cheb_coeffs, 0.5);
/// // T₂(0.5) = 2*0.25 - 1 = -0.5
/// assert!((val - (-0.5)).abs() < 1e-14);
/// ```
pub fn chebyshev_eval(cheb_coeffs: &[f64], x: f64) -> f64 {
    let n = cheb_coeffs.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return cheb_coeffs[0];
    }

    // Clenshaw recurrence for sum c_k T_k(x)
    let mut b_next = 0.0_f64;
    let mut b_curr = 0.0_f64;

    for k in (1..n).rev() {
        let b_prev = cheb_coeffs[k] + 2.0 * x * b_curr - b_next;
        b_next = b_curr;
        b_curr = b_prev;
    }

    cheb_coeffs[0] + x * b_curr - b_next
}

/// Generate `n+1` Chebyshev nodes on the interval `[a, b]`.
///
/// The nodes are `x_k = ((a+b) + (b-a) * cos(π k / n)) / 2` for `k = 0..=n`.
///
/// # Example
/// ```rust
/// use scirs2_fft::polynomial::chebyshev_nodes;
///
/// let nodes = chebyshev_nodes(4, -1.0, 1.0);
/// assert_eq!(nodes.len(), 5);
/// // Endpoints should be ±1
/// assert!((nodes[0] - 1.0).abs() < 1e-14);
/// assert!((nodes[4] - (-1.0)).abs() < 1e-14);
/// ```
pub fn chebyshev_nodes(n: usize, a: f64, b: f64) -> Vec<f64> {
    (0..=n)
        .map(|k| {
            let cos_val = (PI * k as f64 / n as f64).cos();
            (a + b) / 2.0 + (b - a) / 2.0 * cos_val
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Complex polynomial
// ---------------------------------------------------------------------------

/// Dense polynomial with complex coefficients.
///
/// `P(z) = coeffs[0] + coeffs[1]*z + ... + coeffs[n-1]*z^(n-1)`
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexPolynomial {
    /// Coefficients in ascending power order.
    pub coeffs: Array1<Complex64>,
}

impl ComplexPolynomial {
    /// Create a complex polynomial from a coefficient vector.
    pub fn new(coeffs: Vec<Complex64>) -> Self {
        assert!(!coeffs.is_empty(), "coefficient vector must be non-empty");
        Self {
            coeffs: Array1::from_vec(coeffs),
        }
    }

    /// Evaluate the polynomial at complex point `z` using Horner's method.
    pub fn evaluate(&self, z: Complex64) -> Complex64 {
        let n = self.coeffs.len();
        if n == 0 {
            return Complex64::new(0.0, 0.0);
        }
        let mut result = self.coeffs[n - 1];
        for i in (0..n - 1).rev() {
            result = result * z + self.coeffs[i];
        }
        result
    }

    /// Find roots via companion matrix eigenvalues.
    pub fn roots_companion(&self) -> FFTResult<Vec<Complex64>> {
        // Build companion matrix using the real parts and imaginary parts separately
        let n = self.coeffs.len() - 1;
        if n == 0 {
            return Ok(vec![]);
        }

        let cn = self.coeffs[n];
        let cn_abs = (cn.re * cn.re + cn.im * cn.im).sqrt();
        if cn_abs < 1e-300 {
            return Err(FFTError::ComputationError(
                "leading coefficient is zero".to_string(),
            ));
        }

        // For complex polynomial root finding, convert to real poly (if imaginary parts are small)
        // and use the real companion matrix; otherwise use power iteration
        let all_real = self.coeffs.iter().all(|c| c.im.abs() < 1e-12);
        if all_real {
            let real_poly = Polynomial {
                coeffs: Array1::from_vec(self.coeffs.iter().map(|c| c.re).collect()),
            };
            return real_poly.roots();
        }

        // Power iteration with random starting vectors for each root
        complex_polynomial_roots(self, n, cn)
    }
}

/// Find roots of a general complex polynomial via Weierstrass-Durand-Kerner iteration.
fn complex_polynomial_roots(
    p: &ComplexPolynomial,
    n: usize,
    cn: Complex64,
) -> FFTResult<Vec<Complex64>> {
    let cn_abs = (cn.re * cn.re + cn.im * cn.im).sqrt();

    // Normalize to monic
    let monic_coeffs: Vec<Complex64> = p.coeffs.iter().map(|&c| c / cn_abs).collect();
    let monic = ComplexPolynomial {
        coeffs: Array1::from_vec(monic_coeffs),
    };

    // Initial guesses: uniformly spaced on a circle of appropriate radius
    let mut roots: Vec<Complex64> = (0..n)
        .map(|k| {
            let angle = 2.0 * PI * k as f64 / n as f64;
            Complex64::new(angle.cos(), angle.sin())
        })
        .collect();

    // Weierstrass (DK) iteration
    let max_iter = 200;
    for _ in 0..max_iter {
        let mut max_correction = 0.0_f64;
        let old_roots = roots.clone();

        for i in 0..n {
            let f_val = monic.evaluate(old_roots[i]);
            let mut denom = Complex64::new(1.0, 0.0);
            for j in 0..n {
                if j != i {
                    let diff = old_roots[i] - old_roots[j];
                    denom = denom * diff;
                }
            }
            let denom_abs = (denom.re * denom.re + denom.im * denom.im).sqrt();
            if denom_abs < 1e-300 {
                continue;
            }
            let correction = f_val / denom;
            roots[i] = old_roots[i] - correction;
            let corr_abs = (correction.re * correction.re + correction.im * correction.im).sqrt();
            max_correction = max_correction.max(corr_abs);
        }

        if max_correction < 1e-12 {
            break;
        }
    }

    Ok(roots)
}

/// Multiply two complex polynomials using FFT.
///
/// # Example
/// ```rust
/// use scirs2_fft::polynomial::{ComplexPolynomial, complex_poly_multiply};
/// use scirs2_core::numeric::Complex64;
///
/// let p = ComplexPolynomial::new(vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)]);
/// let q = ComplexPolynomial::new(vec![Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)]);
/// let r = complex_poly_multiply(&p, &q).expect("multiply failed");
/// // (1 + z)(1 - z) = 1 - z²
/// assert!((r.coeffs[0].re - 1.0).abs() < 1e-10);
/// assert!(r.coeffs[1].re.abs() < 1e-10);
/// assert!((r.coeffs[2].re - (-1.0)).abs() < 1e-10);
/// ```
pub fn complex_poly_multiply(
    p: &ComplexPolynomial,
    q: &ComplexPolynomial,
) -> FFTResult<ComplexPolynomial> {
    let np = p.coeffs.len();
    let nq = q.coeffs.len();
    let result_len = np + nq - 1;
    let fft_size = result_len.next_power_of_two();

    // Zero-pad and convert to complex slices for FFT
    let mut p_padded: Vec<Complex64> = p.coeffs.iter().copied().collect();
    let mut q_padded: Vec<Complex64> = q.coeffs.iter().copied().collect();
    p_padded.resize(fft_size, Complex64::new(0.0, 0.0));
    q_padded.resize(fft_size, Complex64::new(0.0, 0.0));

    // Convert to f64 slices for fft() — pack real and imag alternately via split trick:
    // Use two real FFTs and recombine
    let p_real: Vec<f64> = p_padded.iter().map(|c| c.re).collect();
    let p_imag: Vec<f64> = p_padded.iter().map(|c| c.im).collect();
    let q_real: Vec<f64> = q_padded.iter().map(|c| c.re).collect();
    let q_imag: Vec<f64> = q_padded.iter().map(|c| c.im).collect();

    let pr_freq = fft(&p_real, Some(fft_size))?;
    let pi_freq = fft(&p_imag, Some(fft_size))?;
    let qr_freq = fft(&q_real, Some(fft_size))?;
    let qi_freq = fft(&q_imag, Some(fft_size))?;

    // P_freq = P_real_freq + i * P_imag_freq
    // Q_freq = Q_real_freq + i * Q_imag_freq
    // Product_freq = P_freq * Q_freq
    let product_freq: Vec<Complex64> = (0..fft_size)
        .map(|k| {
            let pr = pr_freq[k];
            let pi = pi_freq[k];
            let qr = qr_freq[k];
            let qi = qi_freq[k];
            // (pr + i*pi)(qr + i*qi) = (pr*qr - pi*qi) + i*(pr*qi + pi*qr)
            let p_c = Complex64::new(pr.re - pi.im, pr.im + pi.re);
            let q_c = Complex64::new(qr.re - qi.im, qr.im + qi.re);
            Complex64::new(
                p_c.re * q_c.re - p_c.im * q_c.im,
                p_c.re * q_c.im + p_c.im * q_c.re,
            )
        })
        .collect();

    let product_time = ifft(&product_freq, Some(fft_size))?;

    let result_coeffs: Vec<Complex64> = product_time.iter().take(result_len).copied().collect();

    Ok(ComplexPolynomial {
        coeffs: Array1::from_vec(result_coeffs),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    // --- Basic construction ---

    #[test]
    fn test_polynomial_new_and_degree() {
        let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(p.degree(), 2);
        let q = Polynomial::new(vec![0.0, 0.0, 1.0]);
        assert_eq!(q.degree(), 2);
        let z = Polynomial::zero();
        assert_eq!(z.degree(), 0);
    }

    #[test]
    fn test_polynomial_evaluate_horner() {
        // p(x) = 1 + 2x + x²  →  p(3) = 1 + 6 + 9 = 16
        let p = Polynomial::new(vec![1.0, 2.0, 1.0]);
        assert!((p.evaluate(3.0) - 16.0).abs() < TOL);
        assert!((p.evaluate(0.0) - 1.0).abs() < TOL);
        assert!((p.evaluate(-1.0) - 0.0).abs() < TOL);
    }

    #[test]
    fn test_polynomial_evaluate_complex() {
        // p(x) = x², p(i) = -1
        let p = Polynomial::new(vec![0.0, 0.0, 1.0]);
        let val = p.evaluate_complex(Complex64::new(0.0, 1.0));
        assert!((val.re - (-1.0)).abs() < TOL);
        assert!(val.im.abs() < TOL);
    }

    #[test]
    fn test_derivative() {
        // d/dx (1 + 2x + x²) = 2 + 2x
        let p = Polynomial::new(vec![1.0, 2.0, 1.0]);
        let d = p.derivative();
        assert_eq!(d.coeffs.len(), 2);
        assert!((d.coeffs[0] - 2.0).abs() < TOL);
        assert!((d.coeffs[1] - 2.0).abs() < TOL);
    }

    #[test]
    fn test_antiderivative_and_integrate() {
        // ∫ x dx from 0 to 2 = 2
        let p = Polynomial::new(vec![0.0, 1.0]);
        assert!((p.integrate(0.0, 2.0) - 2.0).abs() < TOL);

        // antiderivative of 1 + x is x + x²/2 + C
        let a = p.antiderivative(5.0);
        assert!((a.coeffs[0] - 5.0).abs() < TOL); // C = 5
        assert!((a.coeffs[1] - 0.0).abs() < TOL); // coeff of x from 0
        assert!((a.coeffs[2] - 0.5).abs() < TOL); // coeff of x²
    }

    // --- Arithmetic ---

    #[test]
    fn test_poly_add() {
        let p = Polynomial::new(vec![1.0, 2.0]);
        let q = Polynomial::new(vec![3.0, 4.0, 5.0]);
        let r = poly_add(&p, &q);
        assert!((r.coeffs[0] - 4.0).abs() < TOL);
        assert!((r.coeffs[1] - 6.0).abs() < TOL);
        assert!((r.coeffs[2] - 5.0).abs() < TOL);
    }

    #[test]
    fn test_poly_sub() {
        let p = Polynomial::new(vec![5.0, 6.0]);
        let q = Polynomial::new(vec![1.0, 2.0]);
        let r = poly_sub(&p, &q);
        assert!((r.coeffs[0] - 4.0).abs() < TOL);
        assert!((r.coeffs[1] - 4.0).abs() < TOL);
    }

    #[test]
    fn test_fft_multiply_matches_naive() {
        // (1 + x)(1 + x) = 1 + 2x + x²
        let p = Polynomial::new(vec![1.0; 33]); // large enough to trigger FFT path
        let q = Polynomial::new(vec![1.0; 33]);
        let naive = poly_multiply_naive(&p, &q);
        let fft_result = poly_multiply(&p, &q).expect("FFT multiply failed");
        assert_eq!(naive.coeffs.len(), fft_result.coeffs.len());
        for (a, b) in naive.coeffs.iter().zip(fft_result.coeffs.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_poly_multiply_small() {
        // (1 + x)(1 + x) = 1 + 2x + x²
        let p = Polynomial::new(vec![1.0, 1.0]);
        let q = Polynomial::new(vec![1.0, 1.0]);
        let r = poly_multiply(&p, &q).expect("multiply failed");
        assert!((r.coeffs[0] - 1.0).abs() < TOL);
        assert!((r.coeffs[1] - 2.0).abs() < TOL);
        assert!((r.coeffs[2] - 1.0).abs() < TOL);
    }

    #[test]
    fn test_poly_multiply_degree_correctness() {
        // x * x = x²
        let x = Polynomial::new(vec![0.0, 1.0]);
        let x2 = poly_multiply(&x, &x).expect("multiply failed");
        assert!((x2.coeffs[0]).abs() < TOL);
        assert!((x2.coeffs[1]).abs() < TOL);
        assert!((x2.coeffs[2] - 1.0).abs() < TOL);
    }

    // --- Division ---

    #[test]
    fn test_poly_divmod() {
        // x² - 1 = (x - 1)(x + 1)
        let p = Polynomial::new(vec![-1.0, 0.0, 1.0]);
        let d = Polynomial::new(vec![-1.0, 1.0]);
        let (q, r) = poly_divmod(&p, &d).expect("divmod failed");
        // quotient should be x + 1
        assert!((q.coeffs[0] - 1.0).abs() < TOL);
        assert!((q.coeffs[1] - 1.0).abs() < TOL);
        // remainder should be 0
        assert!(r.coeffs.iter().all(|&c| c.abs() < TOL));
    }

    #[test]
    fn test_poly_divmod_with_remainder() {
        // x² divided by x-1: quotient = x+1, remainder = 1
        let p = Polynomial::new(vec![0.0, 0.0, 1.0]); // x²
        let d = Polynomial::new(vec![-1.0, 1.0]); // x - 1
        let (q, r) = poly_divmod(&p, &d).expect("divmod failed");
        // Verify: q * d + r = p
        let reconstructed =
            poly_add(&poly_multiply(&q, &d).expect("multiply failed"), &r);
        for i in 0..3 {
            let p_val = if i < p.coeffs.len() { p.coeffs[i] } else { 0.0 };
            let r_val = if i < reconstructed.coeffs.len() {
                reconstructed.coeffs[i]
            } else {
                0.0
            };
            assert!((p_val - r_val).abs() < TOL, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_poly_divmod_zero_divisor() {
        let p = Polynomial::new(vec![1.0, 1.0]);
        let zero = Polynomial::zero();
        assert!(poly_divmod(&p, &zero).is_err());
    }

    // --- GCD ---

    #[test]
    fn test_poly_gcd() {
        // gcd(x² - 1, x - 1) = x - 1 (monic)
        let p = Polynomial::new(vec![-1.0, 0.0, 1.0]);
        let q = Polynomial::new(vec![-1.0, 1.0]);
        let g = poly_gcd(&p, &q).expect("gcd failed");
        // Check g divides both p and q
        let (_, rp) = poly_divmod(&p, &g).expect("divmod failed");
        let (_, rq) = poly_divmod(&q, &g).expect("divmod failed");
        assert!(rp.coeffs.iter().all(|&c| c.abs() < 1e-8));
        assert!(rq.coeffs.iter().all(|&c| c.abs() < 1e-8));
    }

    #[test]
    fn test_poly_gcd_coprime() {
        // gcd(x, x+1) = 1
        let p = Polynomial::new(vec![0.0, 1.0]);
        let q = Polynomial::new(vec![1.0, 1.0]);
        let g = poly_gcd(&p, &q).expect("gcd failed");
        // GCD should be monic degree-0 (constant 1)
        assert_eq!(g.degree(), 0);
    }

    // --- Root finding ---

    #[test]
    fn test_roots_linear() {
        // 2x - 4 = 0  =>  x = 2
        let p = Polynomial::new(vec![-4.0, 2.0]);
        let roots = p.roots().expect("roots failed");
        assert_eq!(roots.len(), 1);
        assert!((roots[0].re - 2.0).abs() < TOL);
        assert!(roots[0].im.abs() < TOL);
    }

    #[test]
    fn test_roots_quadratic_real() {
        // x² - 3x + 2 = (x-1)(x-2)
        let p = Polynomial::new(vec![2.0, -3.0, 1.0]);
        let roots = p.roots().expect("roots failed");
        assert_eq!(roots.len(), 2);
        let mut re_roots: Vec<f64> = roots.iter().map(|r| r.re).collect();
        re_roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!((re_roots[0] - 1.0).abs() < TOL);
        assert!((re_roots[1] - 2.0).abs() < TOL);
    }

    #[test]
    fn test_roots_quadratic_complex() {
        // x² + 1 = 0  =>  roots = ±i
        let p = Polynomial::new(vec![1.0, 0.0, 1.0]);
        let roots = p.roots().expect("roots failed");
        assert_eq!(roots.len(), 2);
        for r in &roots {
            assert!(r.re.abs() < TOL);
            assert!((r.im.abs() - 1.0).abs() < TOL);
        }
    }

    #[test]
    fn test_roots_from_roots_roundtrip() {
        // Build polynomial from known roots, then verify root recovery
        let known_roots = vec![1.0, -1.0, 2.0];
        let p = Polynomial::from_roots(&known_roots);
        let found = p.roots().expect("roots failed");
        assert_eq!(found.len(), 3);
        let mut found_re: Vec<f64> = found.iter().map(|r| r.re).collect();
        found_re.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut known_sorted = known_roots.clone();
        known_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (a, b) in found_re.iter().zip(known_sorted.iter()) {
            assert!((a - b).abs() < 1e-6, "root mismatch: {a} vs {b}");
        }
    }

    // --- Multi-point evaluation ---

    #[test]
    fn test_multipoint_eval_small() {
        let p = Polynomial::new(vec![1.0, 2.0, 1.0]); // 1 + 2x + x²
        let pts = vec![0.0, 1.0, -1.0, 2.0];
        let vals = poly_multipoint_eval(&p, &pts).expect("eval failed");
        assert!((vals[0] - p.evaluate(0.0)).abs() < TOL);
        assert!((vals[1] - p.evaluate(1.0)).abs() < TOL);
        assert!((vals[2] - p.evaluate(-1.0)).abs() < TOL);
        assert!((vals[3] - p.evaluate(2.0)).abs() < TOL);
    }

    #[test]
    fn test_multipoint_eval_large() {
        // Test subproduct tree path
        let coeffs: Vec<f64> = (0..65).map(|i| i as f64).collect();
        let p = Polynomial::new(coeffs);
        let pts: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let vals_multi = poly_multipoint_eval(&p, &pts).expect("multipoint eval failed");
        for (x, v) in pts.iter().zip(vals_multi.iter()) {
            let direct = p.evaluate(*x);
            assert!((v - direct).abs() < 1e-6, "mismatch at x={x}");
        }
    }

    // --- Interpolation ---

    #[test]
    fn test_interpolation_quadratic() {
        // Interpolate x² at 0, 1, 2
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];
        let p = poly_interpolate(&x, &y).expect("interpolation failed");
        // Should recover x²
        assert!((p.evaluate(0.5) - 0.25).abs() < 1e-9);
        assert!((p.evaluate(1.5) - 2.25).abs() < 1e-9);
    }

    #[test]
    fn test_interpolation_linear() {
        // y = 2x + 1
        let x = vec![0.0, 1.0];
        let y = vec![1.0, 3.0];
        let p = poly_interpolate(&x, &y).expect("interpolation failed");
        assert!((p.evaluate(0.5) - 2.0).abs() < 1e-9);
        assert!((p.evaluate(2.0) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_interpolation_duplicate_x() {
        let x = vec![1.0, 1.0];
        let y = vec![2.0, 3.0];
        assert!(poly_interpolate(&x, &y).is_err());
    }

    // --- Polynomial power ---

    #[test]
    fn test_poly_pow() {
        // (1 + x)^3 = 1 + 3x + 3x² + x³
        let p = Polynomial::new(vec![1.0, 1.0]);
        let p3 = poly_pow(&p, 3).expect("pow failed");
        assert!((p3.coeffs[0] - 1.0).abs() < TOL);
        assert!((p3.coeffs[1] - 3.0).abs() < TOL);
        assert!((p3.coeffs[2] - 3.0).abs() < TOL);
        assert!((p3.coeffs[3] - 1.0).abs() < TOL);
    }

    #[test]
    fn test_poly_pow_zero() {
        let p = Polynomial::new(vec![2.0, 3.0]);
        let p0 = poly_pow(&p, 0).expect("pow 0 failed");
        // p^0 = 1
        assert_eq!(p0.coeffs.len(), 1);
        assert!((p0.coeffs[0] - 1.0).abs() < TOL);
    }

    // --- Composition ---

    #[test]
    fn test_poly_compose() {
        // (x²)(1 + x) = (1+x)² = 1 + 2x + x²
        let p = Polynomial::new(vec![0.0, 0.0, 1.0]); // x²
        let q = Polynomial::new(vec![1.0, 1.0]); // 1 + x
        let r = poly_compose(&p, &q).expect("compose failed");
        assert!((r.coeffs[0] - 1.0).abs() < TOL);
        assert!((r.coeffs[1] - 2.0).abs() < TOL);
        assert!((r.coeffs[2] - 1.0).abs() < TOL);
    }

    #[test]
    fn test_poly_compose_linear() {
        // p(q(x)) where p = 2x+1, q = x+3 → 2(x+3)+1 = 2x+7
        let p = Polynomial::new(vec![1.0, 2.0]);
        let q = Polynomial::new(vec![3.0, 1.0]);
        let r = poly_compose(&p, &q).expect("compose failed");
        assert!((r.evaluate(0.0) - 7.0).abs() < TOL);
        assert!((r.evaluate(1.0) - 9.0).abs() < TOL);
    }

    // --- Special polynomials ---

    #[test]
    fn test_chebyshev_t() {
        let t0 = chebyshev_t(0);
        assert!((t0.evaluate(0.5) - 1.0).abs() < TOL);

        let t1 = chebyshev_t(1);
        assert!((t1.evaluate(0.5) - 0.5).abs() < TOL);

        // T₂(x) = 2x² - 1
        let t2 = chebyshev_t(2);
        assert!((t2.evaluate(0.0) - (-1.0)).abs() < TOL);
        assert!((t2.evaluate(1.0) - 1.0).abs() < TOL);
        assert!((t2.evaluate(0.5) - (-0.5)).abs() < TOL);

        // T₃(x) = 4x³ - 3x
        let t3 = chebyshev_t(3);
        assert!((t3.evaluate(1.0) - 1.0).abs() < TOL);
        assert!((t3.evaluate(0.0) - 0.0).abs() < TOL);
    }

    #[test]
    fn test_legendre_p() {
        let p0 = legendre_p(0);
        assert!((p0.evaluate(0.5) - 1.0).abs() < TOL);

        let p1 = legendre_p(1);
        assert!((p1.evaluate(0.5) - 0.5).abs() < TOL);

        // P₂(x) = (3x² - 1)/2
        let p2 = legendre_p(2);
        assert!((p2.evaluate(0.0) - (-0.5)).abs() < TOL);
        assert!((p2.evaluate(1.0) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_hermite_prob() {
        let h0 = hermite_prob(0);
        assert!((h0.evaluate(2.0) - 1.0).abs() < TOL);

        let h1 = hermite_prob(1);
        assert!((h1.evaluate(2.0) - 2.0).abs() < TOL);

        // He₂(x) = x² - 1
        let h2 = hermite_prob(2);
        assert!((h2.evaluate(0.0) - (-1.0)).abs() < TOL);
        assert!((h2.evaluate(1.0) - 0.0).abs() < TOL);

        // He₃(x) = x³ - 3x
        let h3 = hermite_prob(3);
        assert!((h3.evaluate(1.0) - (-2.0)).abs() < TOL);
    }

    // --- Chebyshev expansion ---

    #[test]
    fn test_chebyshev_nodes() {
        let nodes = chebyshev_nodes(4, -1.0, 1.0);
        assert_eq!(nodes.len(), 5);
        assert!((nodes[0] - 1.0).abs() < TOL);
        assert!((nodes[4] - (-1.0)).abs() < TOL);
    }

    #[test]
    fn test_chebyshev_eval_basis() {
        // T₂ expressed as Chebyshev coefficients: c₂ = 1, rest = 0
        let cheb_coeffs = vec![0.0, 0.0, 1.0];
        // T₂(0.5) = 2*(0.5)² - 1 = -0.5
        let val = chebyshev_eval(&cheb_coeffs, 0.5);
        assert!((val - (-0.5)).abs() < TOL);
    }

    #[test]
    fn test_chebyshev_expansion_and_eval() {
        // f(x) = x² on [-1,1]; expand and recover
        let n = 8;
        let nodes = chebyshev_nodes(n, -1.0, 1.0);
        let f_vals: Vec<f64> = nodes.iter().map(|&x| x * x).collect();
        let coeffs = chebyshev_expansion(&f_vals);
        // Evaluate at a few test points
        for &x in &[-0.9, -0.5, 0.0, 0.3, 0.7] {
            let approx = chebyshev_eval(&coeffs, x);
            let expected = x * x;
            assert!(
                (approx - expected).abs() < 1e-8,
                "x={x}: approx={approx} expected={expected}"
            );
        }
    }

    // --- Complex polynomial ---

    #[test]
    fn test_complex_poly_evaluate() {
        // p(z) = z² + 1,  p(i) = i² + 1 = -1 + 1 = 0
        let p = ComplexPolynomial::new(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ]);
        let val = p.evaluate(Complex64::new(0.0, 1.0));
        assert!(val.re.abs() < TOL);
        assert!(val.im.abs() < TOL);
    }

    #[test]
    fn test_complex_poly_multiply() {
        // (1 + z)(1 - z) = 1 - z²
        let p = ComplexPolynomial::new(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ]);
        let q = ComplexPolynomial::new(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ]);
        let r = complex_poly_multiply(&p, &q).expect("complex multiply failed");
        assert!((r.coeffs[0].re - 1.0).abs() < TOL);
        assert!(r.coeffs[1].re.abs() < TOL);
        assert!((r.coeffs[2].re - (-1.0)).abs() < TOL);
    }

    #[test]
    fn test_complex_poly_roots() {
        // z² - 1 = (z-1)(z+1), roots are ±1
        let p = ComplexPolynomial::new(vec![
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ]);
        let roots = p.roots_companion().expect("roots failed");
        assert_eq!(roots.len(), 2);
    }

    // --- Poly powmod ---

    #[test]
    fn test_poly_powmod() {
        // x^3 mod (x^2 - 1)
        // x^3 = x*(x^2-1) + x, so result = x = [0, 1]
        let p = Polynomial::new(vec![0.0, 1.0]);
        let m = Polynomial::new(vec![-1.0, 0.0, 1.0]);
        let result = poly_powmod(&p, 3, &m).expect("powmod failed");
        // result should be approximately x
        let trimmed = result.trim();
        // evaluate at 2: should give 2 (since x mod (x²-1) = x when deg < 2)
        let val = trimmed.evaluate(2.0);
        assert!((val - 2.0).abs() < 1e-8);
    }

    // --- from_roots ---

    #[test]
    fn test_from_roots() {
        // (x-1)(x-2) = x² - 3x + 2
        let p = Polynomial::from_roots(&[1.0, 2.0]);
        assert!((p.evaluate(1.0)).abs() < TOL);
        assert!((p.evaluate(2.0)).abs() < TOL);
        assert!((p.evaluate(0.0) - 2.0).abs() < TOL);
    }

    // --- Trim ---

    #[test]
    fn test_trim() {
        let p = Polynomial::new(vec![1.0, 2.0, 0.0, 0.0]);
        let t = p.trim();
        assert_eq!(t.coeffs.len(), 2);
        assert!((t.coeffs[0] - 1.0).abs() < TOL);
        assert!((t.coeffs[1] - 2.0).abs() < TOL);
    }
}
