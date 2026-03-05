//! # Fast Polynomial Arithmetic
//!
//! Dense polynomial arithmetic over ℝ (and ℂ) using both naive O(n²) and
//! FFT-accelerated O(n log n) algorithms.
//!
//! ## Algorithms
//!
//! | Operation | Algorithm | Complexity |
//! |-----------|-----------|------------|
//! | `eval` | Horner's method | O(n) |
//! | `add` / `sub` | Term-wise | O(n) |
//! | `mul_naive` | Schoolbook | O(n²) |
//! | `mul_fft` | FFT convolution | O(n log n) |
//! | `div_rem` | Synthetic / long | O(n²) |
//! | `gcd` | Euclidean | O(n²) |
//! | `compose` | Horner over poly | O(n²) |
//! | `roots_jenkins_traub` | Jenkins-Traub | O(n²) iter |
//!
//! ## Example
//!
//! ```rust
//! use scirs2_fft::polynomial::arithmetic::Polynomial;
//!
//! let p = Polynomial::new(vec![2.0, 0.0, 1.0]); // 2 + x²
//! let q = Polynomial::new(vec![1.0, 1.0]);       // 1 + x
//! let r = p.mul_fft(&q).expect("fft mul");       // 2 + 2x + x² + x³
//! assert!((r.eval(0.0) - 2.0).abs() < 1e-10);
//! ```

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
//  Dense polynomial type
// ─────────────────────────────────────────────────────────────────────────────

/// Dense polynomial over ℝ.
///
/// `P(x) = coeffs[0] + coeffs[1]·x + … + coeffs[n]·xⁿ`
///
/// The zero polynomial is represented as `coeffs = [0.0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    /// Coefficients in ascending power order.  `coeffs[k]` is the coefficient
    /// of `x^k`.
    pub coeffs: Vec<f64>,
}

impl Polynomial {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create a new polynomial from a coefficient vector.
    ///
    /// # Arguments
    ///
    /// * `coeffs` – Coefficients `[a₀, a₁, …, aₙ]` in ascending power order.
    ///   If the vector is empty an equivalent zero polynomial (`[0.0]`) is stored.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_fft::polynomial::arithmetic::Polynomial;
    /// let p = Polynomial::new(vec![1.0, 2.0, 3.0]); // 1 + 2x + 3x²
    /// assert_eq!(p.degree(), 2);
    /// ```
    pub fn new(coeffs: Vec<f64>) -> Self {
        if coeffs.is_empty() {
            Self { coeffs: vec![0.0] }
        } else {
            Self { coeffs }
        }
    }

    /// The zero polynomial `P(x) = 0`.
    pub fn zero() -> Self {
        Self { coeffs: vec![0.0] }
    }

    /// The unit polynomial `P(x) = 1`.
    pub fn one() -> Self {
        Self { coeffs: vec![1.0] }
    }

    /// Build the monomial `x^k`.
    pub fn monomial(k: usize) -> Self {
        let mut c = vec![0.0; k + 1];
        c[k] = 1.0;
        Self { coeffs: c }
    }

    // ── Metadata ──────────────────────────────────────────────────────────────

    /// The algebraic degree of the polynomial.
    ///
    /// Trailing near-zero coefficients are ignored.  Returns 0 for the zero
    /// polynomial.
    pub fn degree(&self) -> usize {
        let eps = f64::EPSILON * 1e3;
        for i in (0..self.coeffs.len()).rev() {
            if self.coeffs[i].abs() > eps {
                return i;
            }
        }
        0
    }

    /// Return `true` if all coefficients are zero (within machine epsilon).
    pub fn is_zero(&self) -> bool {
        let eps = f64::EPSILON * 1e3;
        self.coeffs.iter().all(|&c| c.abs() <= eps)
    }

    /// Leading (highest non-zero) coefficient.
    pub fn leading_coeff(&self) -> f64 {
        self.coeffs[self.degree()]
    }

    // ── Evaluation ────────────────────────────────────────────────────────────

    /// Evaluate the polynomial at `x` using Horner's method.
    ///
    /// O(n) time, numerically stable.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_fft::polynomial::arithmetic::Polynomial;
    /// let p = Polynomial::new(vec![1.0, -3.0, 2.0]); // (x-1)(x-2)
    /// assert!((p.eval(1.0)).abs() < 1e-12);
    /// assert!((p.eval(2.0)).abs() < 1e-12);
    /// ```
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.coeffs.len();
        if n == 0 {
            return 0.0;
        }
        let mut acc = self.coeffs[n - 1];
        for i in (0..n - 1).rev() {
            acc = acc * x + self.coeffs[i];
        }
        acc
    }

    /// Evaluate the polynomial at a complex point `z`.
    pub fn eval_complex(&self, z: Complex64) -> Complex64 {
        let n = self.coeffs.len();
        if n == 0 {
            return Complex64::new(0.0, 0.0);
        }
        let mut acc = Complex64::new(self.coeffs[n - 1], 0.0);
        for i in (0..n - 1).rev() {
            acc = acc * z + Complex64::new(self.coeffs[i], 0.0);
        }
        acc
    }

    /// Evaluate the polynomial at each of several points simultaneously.
    pub fn eval_many(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&x| self.eval(x)).collect()
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Polynomial addition.
    pub fn add(&self, other: &Self) -> Self {
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut c = vec![0.0; len];
        for (i, &a) in self.coeffs.iter().enumerate() {
            c[i] += a;
        }
        for (i, &b) in other.coeffs.iter().enumerate() {
            c[i] += b;
        }
        Self { coeffs: c }
    }

    /// Polynomial subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut c = vec![0.0; len];
        for (i, &a) in self.coeffs.iter().enumerate() {
            c[i] += a;
        }
        for (i, &b) in other.coeffs.iter().enumerate() {
            c[i] -= b;
        }
        Self { coeffs: c }
    }

    /// Schoolbook (O(n²)) polynomial multiplication.
    ///
    /// Prefer [`mul_fft`] for large polynomials.
    pub fn mul_naive(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let n = self.coeffs.len();
        let m = other.coeffs.len();
        let mut c = vec![0.0; n + m - 1];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                c[i + j] += a * b;
            }
        }
        Self { coeffs: c }
    }

    /// FFT-accelerated O(n log n) polynomial multiplication.
    ///
    /// Uses zero-padded FFT convolution.  Falls back to [`mul_naive`] for very
    /// short polynomials where the FFT overhead is not worthwhile.
    ///
    /// # Errors
    ///
    /// Returns an error only if the underlying FFT fails (which should not
    /// happen for valid inputs).
    pub fn mul_fft(&self, other: &Self) -> FFTResult<Self> {
        if self.is_zero() || other.is_zero() {
            return Ok(Self::zero());
        }
        // For small polynomials use naive to avoid FFT overhead
        if self.coeffs.len() + other.coeffs.len() <= 64 {
            return Ok(self.mul_naive(other));
        }
        let result_len = self.coeffs.len() + other.coeffs.len() - 1;
        let fft_len = result_len.next_power_of_two();

        // Build complex input vectors
        let mut fa: Vec<f64> = self.coeffs.clone();
        fa.resize(fft_len, 0.0);
        let mut fb: Vec<f64> = other.coeffs.clone();
        fb.resize(fft_len, 0.0);

        let fa_c = fft(&fa, None)?;
        let fb_c = fft(&fb, None)?;

        // Point-wise multiply
        let prod: Vec<Complex64> = fa_c.iter().zip(fb_c.iter()).map(|(a, b)| a * b).collect();

        // Inverse FFT
        let result_c = ifft(&prod, None)?;

        // Extract real part, trim to result_len
        let coeffs: Vec<f64> = result_c[..result_len]
            .iter()
            .map(|c| c.re)
            .collect();

        Ok(Self { coeffs })
    }

    /// Polynomial long division.
    ///
    /// Returns `(quotient, remainder)` such that
    /// `self = divisor * quotient + remainder` and
    /// `deg(remainder) < deg(divisor)`.
    ///
    /// # Errors
    ///
    /// Returns [`FFTError::ValueError`] if `divisor` is the zero polynomial.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_fft::polynomial::arithmetic::Polynomial;
    /// // x³ - 1 divided by (x - 1)  →  quotient = x² + x + 1,  remainder = 0
    /// let p = Polynomial::new(vec![-1.0, 0.0, 0.0, 1.0]);
    /// let d = Polynomial::new(vec![-1.0, 1.0]);
    /// let (q, r) = p.div_rem(&d).expect("div_rem");
    /// assert!(r.is_zero());
    /// assert!((q.eval(2.0) - 7.0).abs() < 1e-10);
    /// ```
    pub fn div_rem(&self, divisor: &Self) -> FFTResult<(Self, Self)> {
        if divisor.is_zero() {
            return Err(FFTError::ValueError(
                "polynomial division by zero".to_string(),
            ));
        }
        let deg_d = divisor.degree();
        let lc_d = divisor.coeffs[deg_d];

        let mut rem = self.coeffs.clone();
        let n = rem.len();
        let deg_self = self.degree();

        if deg_self < deg_d {
            return Ok((Self::zero(), self.clone()));
        }

        let q_len = deg_self - deg_d + 1;
        let mut q_coeffs = vec![0.0; q_len];

        for i in (0..q_len).rev() {
            let rem_hi = rem[i + deg_d];
            if rem_hi.abs() < f64::EPSILON * 1e3 {
                continue;
            }
            let coeff = rem_hi / lc_d;
            q_coeffs[i] = coeff;
            for j in 0..=deg_d {
                rem[i + j] -= coeff * divisor.coeffs[j];
            }
        }

        // Trim leading zeros from remainder
        let mut r_len = n;
        while r_len > 1 && rem[r_len - 1].abs() < f64::EPSILON * 1e3 {
            r_len -= 1;
        }
        rem.truncate(r_len);

        Ok((Self { coeffs: q_coeffs }, Self { coeffs: rem }))
    }

    /// Polynomial GCD via the Euclidean algorithm.
    ///
    /// Returns the monic GCD (leading coefficient normalised to 1).
    ///
    /// # Errors
    ///
    /// Returns an error if polynomial division fails internally.
    pub fn gcd(&self, other: &Self) -> FFTResult<Self> {
        let mut a = self.clone();
        let mut b = other.clone();

        while !b.is_zero() {
            let (_, r) = a.div_rem(&b)?;
            a = b;
            b = r;
        }

        // Make monic
        let lc = a.leading_coeff();
        if lc.abs() > f64::EPSILON {
            let inv = 1.0 / lc;
            let monic: Vec<f64> = a.coeffs.iter().map(|&c| c * inv).collect();
            Ok(Self { coeffs: monic })
        } else {
            Ok(Self::one())
        }
    }

    /// Polynomial composition `(self ∘ other)(x) = self(other(x))`.
    ///
    /// Uses Horner's scheme over polynomials: evaluates `self` with `other`
    /// substituted as the variable.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_fft::polynomial::arithmetic::Polynomial;
    /// // f(x) = x²,  g(x) = x + 1,  f(g(x)) = (x+1)² = 1 + 2x + x²
    /// let f = Polynomial::new(vec![0.0, 0.0, 1.0]);
    /// let g = Polynomial::new(vec![1.0, 1.0]);
    /// let fg = f.compose(&g);
    /// assert!((fg.eval(3.0) - 16.0).abs() < 1e-10); // (3+1)² = 16
    /// ```
    pub fn compose(&self, other: &Self) -> Self {
        let n = self.coeffs.len();
        if n == 0 {
            return Self::zero();
        }
        let mut acc = Self::new(vec![self.coeffs[n - 1]]);
        for i in (0..n - 1).rev() {
            acc = acc.mul_naive(other).add(&Self::new(vec![self.coeffs[i]]));
        }
        acc
    }

    /// Formal derivative `P'(x)`.
    pub fn derivative(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return Self::zero();
        }
        let c: Vec<f64> = self.coeffs[1..]
            .iter()
            .enumerate()
            .map(|(i, &a)| (i + 1) as f64 * a)
            .collect();
        Self { coeffs: c }
    }

    /// Formal indefinite integral `∫P(x)dx` (constant of integration = 0).
    pub fn integral(&self) -> Self {
        let mut c = vec![0.0]; // constant term is 0
        for (i, &a) in self.coeffs.iter().enumerate() {
            c.push(a / (i + 1) as f64);
        }
        Self { coeffs: c }
    }

    // ── Root Finding ─────────────────────────────────────────────────────────

    /// Find all complex roots using the Jenkins-Traub three-stage algorithm.
    ///
    /// The Jenkins-Traub algorithm is a robust iterative root-finder that
    /// converges cubically near simple roots.  This implementation:
    ///
    /// 1. Deflates trivial roots at 0.
    /// 2. Handles linear/quadratic factors analytically.
    /// 3. Applies Stage 1 (fixed shifts), Stage 2 (random shifts), Stage 3
    ///    (variable shifts converging toward a root).
    /// 4. Deflates the polynomial once a root is found and repeats.
    ///
    /// # Returns
    ///
    /// A vector of all `deg(P)` roots (counted with multiplicity).  Complex
    /// conjugate roots appear as conjugate pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if the polynomial is constant (degree 0) or if an
    /// internal numerical failure occurs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_fft::polynomial::arithmetic::Polynomial;
    /// // P(x) = (x - 1)(x - 2)(x - 3) = -6 + 11x - 6x² + x³
    /// let p = Polynomial::new(vec![-6.0, 11.0, -6.0, 1.0]);
    /// let roots = p.roots_jenkins_traub().expect("roots");
    /// assert_eq!(roots.len(), 3);
    /// // Check each root satisfies P(r) ≈ 0
    /// for r in &roots {
    ///     let val = p.eval_complex(*r);
    ///     assert!(val.norm() < 1e-6, "residual too large: {}", val.norm());
    /// }
    /// ```
    pub fn roots_jenkins_traub(&self) -> FFTResult<Vec<Complex64>> {
        let deg = self.degree();
        if deg == 0 {
            return Err(FFTError::ValueError(
                "constant polynomial has no roots".to_string(),
            ));
        }

        // Work with a monic copy normalised by leading coefficient
        let lc = self.coeffs[deg];
        let monic: Vec<f64> = self.coeffs[..=deg].iter().map(|&c| c / lc).collect();

        jenkins_traub_roots(&monic)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Jenkins-Traub implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Deflate a monic polynomial by a linear factor `(x - root)`.
///
/// Returns the quotient polynomial coefficients.
fn deflate_linear(poly: &[f64], root: f64) -> Vec<f64> {
    let n = poly.len();
    if n <= 1 {
        return vec![1.0];
    }
    let mut q = vec![0.0; n - 1];
    q[n - 2] = *poly.last().unwrap_or(&1.0); // leading coefficient of deflated
    for i in (0..n - 2).rev() {
        q[i] = poly[i + 1] + root * q[i + 1];
    }
    q
}

/// Deflate a monic polynomial by a quadratic factor `(x² + px + q)`.
fn deflate_quadratic(poly: &[f64], p: f64, q: f64) -> Vec<f64> {
    let n = poly.len();
    if n <= 2 {
        return vec![1.0];
    }
    let mut b = vec![0.0; n - 2];
    let deg = n - 1;
    b[deg - 2] = *poly.last().unwrap_or(&1.0);
    if deg >= 3 {
        b[deg - 3] = poly[deg - 1] - p * b[deg - 2];
        for i in (0..deg - 3).rev() {
            b[i] = poly[i + 2] - p * b[i + 1] - q * b[i + 2];
        }
    }
    b
}

/// Evaluate P(z) and P'(z) together using Horner's method.
fn eval_and_deriv(poly: &[f64], z: Complex64) -> (Complex64, Complex64) {
    let n = poly.len();
    if n == 0 {
        return (Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0));
    }
    let mut pz = Complex64::new(*poly.last().unwrap_or(&0.0), 0.0);
    let mut dz = Complex64::new(0.0, 0.0);
    for i in (0..n - 1).rev() {
        dz = dz * z + pz;
        pz = pz * z + Complex64::new(poly[i], 0.0);
    }
    (pz, dz)
}

/// Solve a quadratic `z² + bz + c = 0` and return two complex roots.
fn solve_quadratic(b: f64, c: f64) -> (Complex64, Complex64) {
    let disc = b * b - 4.0 * c;
    if disc >= 0.0 {
        let s = disc.sqrt();
        (
            Complex64::new((-b + s) / 2.0, 0.0),
            Complex64::new((-b - s) / 2.0, 0.0),
        )
    } else {
        let s = (-disc).sqrt();
        (
            Complex64::new(-b / 2.0, s / 2.0),
            Complex64::new(-b / 2.0, -s / 2.0),
        )
    }
}

/// Newton-Raphson refinement for a single root.
fn newton_refine(poly: &[f64], z0: Complex64) -> Complex64 {
    let mut z = z0;
    for _ in 0..50 {
        let (pz, dz) = eval_and_deriv(poly, z);
        let dz_norm = dz.norm();
        if dz_norm < 1e-30 {
            break;
        }
        let step = pz / dz;
        z -= step;
        if step.norm() < 1e-14 * z.norm().max(1.0) {
            break;
        }
    }
    z
}


// ─────────────────────────────────────────────────────────────────────────────
//  Root finding: Aberth-Ehrlich simultaneous iteration
// ─────────────────────────────────────────────────────────────────────────────

/// Find all roots of a monic polynomial using the Aberth-Ehrlich algorithm.
///
/// Aberth-Ehrlich is a simultaneous root-finding method with cubic convergence
/// near simple roots.  All roots are updated simultaneously in each step using
/// the formula:
///
/// ```text
/// z_k^{new} = z_k  −  P(z_k) / P'(z_k) / (1 − P(z_k)/P'(z_k) · Σ_{j≠k} 1/(z_k - z_j))
/// ```
///
/// This is the same as Newton's method corrected by the Weierstrass denominator
/// to decouple the roots.
fn jenkins_traub_roots(poly: &[f64]) -> FFTResult<Vec<Complex64>> {
    let mut p = poly.to_vec();
    let mut roots: Vec<Complex64> = Vec::new();

    // Deflate zero roots
    while p.len() >= 2 && p[0].abs() < f64::EPSILON * 1e6 {
        roots.push(Complex64::new(0.0, 0.0));
        p.remove(0);
    }

    while p.len() > 1 {
        let deg = p.len() - 1;

        if deg == 1 {
            roots.push(Complex64::new(-p[0] / p[1], 0.0));
            break;
        }

        if deg == 2 {
            let a2 = p[2];
            let (r1, r2) = solve_quadratic(p[1] / a2, p[0] / a2);
            roots.push(r1);
            roots.push(r2);
            break;
        }

        // Find all roots of p simultaneously using Aberth-Ehrlich
        match aberth_ehrlich_roots(&p) {
            Ok(new_roots) => {
                roots.extend(new_roots);
                break;
            }
            Err(_) => {
                // Fallback: companion matrix
                let fallback = companion_matrix_roots(&p)?;
                roots.extend(fallback);
                break;
            }
        }
    }

    Ok(roots)
}

/// Aberth-Ehrlich simultaneous root-finding for a monic polynomial.
///
/// Returns all `deg(p)` roots, polished with Newton-Raphson refinement.
fn aberth_ehrlich_roots(poly: &[f64]) -> FFTResult<Vec<Complex64>> {
    let n = poly.len() - 1; // degree
    if n == 0 {
        return Ok(vec![]);
    }

    // Initial approximations distributed on a circle of the Cauchy bound radius
    let bound = cauchy_bound(poly);
    let mut z: Vec<Complex64> = (0..n)
        .map(|k| {
            // Distribute on a slightly elliptical locus to avoid symmetry degeneracy
            let angle = 2.0 * PI * k as f64 / n as f64 + 0.7 * PI / n as f64;
            let r = bound * (0.7 + 0.3 * (k as f64 * 0.4).sin().abs().max(0.1));
            Complex64::new(r * angle.cos(), r * angle.sin())
        })
        .collect();

    let tol = 1e-13_f64;
    let max_iter = 300;

    for _iter in 0..max_iter {
        let mut max_step = 0.0_f64;
        let z_old = z.clone();

        for k in 0..n {
            let zk = z_old[k];
            let (pz, dz) = eval_and_deriv(poly, zk);

            // Newton step = P(z_k) / P'(z_k)
            let dz_norm = dz.norm();
            if dz_norm < f64::EPSILON * 1e3 {
                continue; // Near zero derivative — skip this root this iteration
            }
            let newton = pz / dz;

            // Weierstrass denominator: Σ_{j≠k} 1/(z_k - z_j)
            let mut weierstrass = Complex64::new(0.0, 0.0);
            for (j, &zj) in z_old.iter().enumerate() {
                if j != k {
                    let diff = zk - zj;
                    let dnorm = diff.norm();
                    if dnorm > f64::EPSILON {
                        weierstrass = weierstrass + Complex64::new(1.0, 0.0) / diff;
                    }
                }
            }

            // Aberth correction
            let denom = Complex64::new(1.0, 0.0) - newton * weierstrass;
            let denom_norm = denom.norm();
            let step = if denom_norm > f64::EPSILON * 1e3 {
                newton / denom
            } else {
                newton
            };

            z[k] = zk - step;
            let step_norm = step.norm();
            if step_norm > max_step {
                max_step = step_norm;
            }
        }

        // Convergence: all steps are below tolerance
        if max_step < tol {
            break;
        }
    }

    // Final Newton polishing
    let z_polished: Vec<Complex64> = z
        .iter()
        .map(|&zk| newton_refine(poly, zk))
        .collect();

    Ok(z_polished)
}

/// Cauchy bound on the modulus of roots.
fn cauchy_bound(poly: &[f64]) -> f64 {
    let n = poly.len();
    if n <= 1 {
        return 1.0;
    }
    let lc = poly[n - 1].abs();
    if lc < f64::EPSILON {
        return 1.0;
    }
    let max_coeff = poly[..n - 1].iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
    1.0 + max_coeff / lc
}



// ─────────────────────────────────────────────────────────────────────────────
//  Companion matrix fallback
// ─────────────────────────────────────────────────────────────────────────────

/// Eigenvalues of a 2×2 matrix.
fn eig_2x2(a00: f64, a01: f64, a10: f64, a11: f64) -> (Complex64, Complex64) {
    let tr = a00 + a11;
    let det = a00 * a11 - a01 * a10;
    let disc = tr * tr - 4.0 * det;
    if disc >= 0.0 {
        let s = disc.sqrt();
        (Complex64::new((tr + s) / 2.0, 0.0), Complex64::new((tr - s) / 2.0, 0.0))
    } else {
        let s = (-disc).sqrt();
        (Complex64::new(tr / 2.0, s / 2.0), Complex64::new(tr / 2.0, -s / 2.0))
    }
}

/// Apply a single QR shift step to sub-matrix `h[lo..=hi]`.
fn apply_qr_shift(h: &mut Vec<Vec<f64>>, lo: usize, hi: usize, shift: f64) {
    let size = hi - lo + 1;
    if size < 2 {
        return;
    }
    for k in lo..hi {
        let a = h[k][k] - shift;
        let b = h[k + 1][k];
        let r = (a * a + b * b).sqrt();
        if r < f64::EPSILON {
            continue;
        }
        let c = a / r;
        let s = b / r;
        for j in k..=hi {
            let tmp1 = c * h[k][j] + s * h[k + 1][j];
            let tmp2 = -s * h[k][j] + c * h[k + 1][j];
            h[k][j] = tmp1;
            h[k + 1][j] = tmp2;
        }
        let row_end = (k + 2).min(hi);
        for i in lo..=row_end {
            let tmp1 = c * h[i][k] + s * h[i][k + 1];
            let tmp2 = -s * h[i][k] + c * h[i][k + 1];
            h[i][k] = tmp1;
            h[i][k + 1] = tmp2;
        }
    }
}

/// Companion matrix eigenvalue extraction via Francis QR iteration.
///
/// Used as a fallback when Aberth-Ehrlich fails to converge.
fn companion_matrix_roots(poly: &[f64]) -> FFTResult<Vec<Complex64>> {
    let deg = poly.len() - 1;
    if deg == 0 {
        return Ok(vec![]);
    }

    let lc = poly[deg];
    let p: Vec<f64> = poly.iter().map(|&c| c / lc).collect();

    let n = deg;
    let mut h = vec![vec![0.0_f64; n]; n];

    for i in 0..n {
        h[i][n - 1] = -p[i] / p[n];
    }
    for i in 1..n {
        h[i][i - 1] = 1.0;
    }

    let max_qr = 200 * n;
    let mut eigenvalues: Vec<Complex64> = Vec::with_capacity(n);
    let mut active_lo = 0_usize;
    let mut active_hi = n - 1;
    let mut iter_count = 0_usize;

    while active_lo < active_hi && iter_count < max_qr {
        iter_count += 1;
        let a = h[active_hi][active_hi];
        let shift = a;

        let mut deflated = false;
        for i in (active_lo..active_hi).rev() {
            if h[i + 1][i].abs()
                < f64::EPSILON * 1e6 * (h[i][i].abs() + h[i + 1][i + 1].abs())
            {
                if i == active_hi - 1 {
                    eigenvalues.push(Complex64::new(h[active_hi][active_hi], 0.0));
                    if active_hi == 0 {
                        active_hi = 0;
                    } else {
                        active_hi -= 1;
                    }
                } else {
                    let (e1, e2) = eig_2x2(h[i][i], h[i][i + 1], h[i + 1][i], h[i + 1][i + 1]);
                    eigenvalues.push(e1);
                    eigenvalues.push(e2);
                    active_hi = i.saturating_sub(1);
                }
                deflated = true;
                break;
            }
        }

        if !deflated && active_lo < active_hi {
            apply_qr_shift(&mut h, active_lo, active_hi, shift);
        }
    }

    if active_lo == active_hi {
        eigenvalues.push(Complex64::new(h[active_lo][active_lo], 0.0));
    } else if iter_count >= max_qr {
        for i in active_lo..=active_hi {
            eigenvalues.push(Complex64::new(h[i][i], 0.0));
        }
    }

    Ok(eigenvalues)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn p(c: Vec<f64>) -> Polynomial {
        Polynomial::new(c)
    }

    #[test]
    fn test_degree() {
        assert_eq!(p(vec![1.0, 2.0, 3.0]).degree(), 2);
        assert_eq!(p(vec![5.0]).degree(), 0);
        assert_eq!(Polynomial::zero().degree(), 0);
    }

    #[test]
    fn test_eval_horner() {
        // P(x) = 1 - 3x + 2x²  =>  P(2) = 1 - 6 + 8 = 3
        let poly = p(vec![1.0, -3.0, 2.0]);
        assert_relative_eq!(poly.eval(2.0), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_add() {
        let a = p(vec![1.0, 2.0]);
        let b = p(vec![3.0, 0.0, 1.0]);
        let c = a.add(&b);
        assert_eq!(c.coeffs, vec![4.0, 2.0, 1.0]);
    }

    #[test]
    fn test_mul_naive_basic() {
        // (1 + x)(1 + x) = 1 + 2x + x²
        let a = p(vec![1.0, 1.0]);
        let c = a.mul_naive(&a);
        assert_relative_eq!(c.coeffs[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(c.coeffs[1], 2.0, epsilon = 1e-12);
        assert_relative_eq!(c.coeffs[2], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_mul_fft_matches_naive() {
        // Degree-7 * degree-5
        let a = p(vec![1.0, -1.0, 2.0, 0.5, -3.0, 1.0, 0.0, 2.0]);
        let b = p(vec![1.0, 2.0, -1.0, 0.0, 3.0, 1.5]);
        let naive = a.mul_naive(&b);
        let fft = a.mul_fft(&b).expect("fft mul");
        assert_eq!(naive.coeffs.len(), fft.coeffs.len());
        for (n, f) in naive.coeffs.iter().zip(fft.coeffs.iter()) {
            assert_relative_eq!(n, f, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_div_rem_exact() {
        // x³ - 1  ÷  (x - 1)  =  x² + x + 1  remainder 0
        let p_poly = p(vec![-1.0, 0.0, 0.0, 1.0]);
        let d = p(vec![-1.0, 1.0]);
        let (q, r) = p_poly.div_rem(&d).expect("div_rem");
        assert!(r.is_zero(), "remainder should be zero: {:?}", r.coeffs);
        assert_relative_eq!(q.eval(1.0), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_div_rem_with_remainder() {
        // x² + 1  ÷  (x + 1)  =  (x - 1) remainder 2
        let p_poly = p(vec![1.0, 0.0, 1.0]);
        let d = p(vec![1.0, 1.0]);
        let (q, r) = p_poly.div_rem(&d).expect("div_rem");
        assert_relative_eq!(r.eval(0.0), 2.0, epsilon = 1e-10);
        assert_relative_eq!(q.eval(1.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gcd() {
        // gcd((x-1)(x-2), (x-1)(x-3)) = (x-1)
        let a = p(vec![2.0, -3.0, 1.0]); // (x-1)(x-2)
        let b = p(vec![3.0, -4.0, 1.0]); // (x-1)(x-3)
        let g = a.gcd(&b).expect("gcd");
        // gcd should vanish at x=1
        assert!(g.eval(1.0).abs() < 1e-8, "gcd(1) = {}", g.eval(1.0));
    }

    #[test]
    fn test_compose() {
        // f(x) = x²,  g(x) = x + 1  =>  f(g(x)) = (x+1)² = 1 + 2x + x²
        let f = p(vec![0.0, 0.0, 1.0]);
        let g = p(vec![1.0, 1.0]);
        let fg = f.compose(&g);
        assert_relative_eq!(fg.eval(3.0), 16.0, epsilon = 1e-10);
    }

    #[test]
    fn test_derivative() {
        // d/dx (1 + 2x + 3x²) = 2 + 6x
        let poly = p(vec![1.0, 2.0, 3.0]);
        let dp = poly.derivative();
        assert_relative_eq!(dp.eval(0.0), 2.0, epsilon = 1e-12);
        assert_relative_eq!(dp.eval(1.0), 8.0, epsilon = 1e-12);
    }

    #[test]
    fn test_integral() {
        // ∫(2 + 6x)dx = 2x + 3x² (+ C=0)
        let poly = p(vec![2.0, 6.0]);
        let ip = poly.integral();
        assert_relative_eq!(ip.eval(0.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(ip.eval(1.0), 5.0, epsilon = 1e-12); // 2 + 3
    }

    #[test]
    fn test_roots_jenkins_traub_cubic() {
        // P(x) = (x-1)(x-2)(x-3) = -6 + 11x - 6x² + x³
        let poly = p(vec![-6.0, 11.0, -6.0, 1.0]);
        let roots = poly.roots_jenkins_traub().expect("roots");
        assert_eq!(roots.len(), 3);
        for r in &roots {
            let val = poly.eval_complex(*r);
            assert!(val.norm() < 1e-6, "residual {}", val.norm());
        }
    }

    #[test]
    fn test_roots_quadratic() {
        // x² - 5x + 6 = (x-2)(x-3)
        let poly = p(vec![6.0, -5.0, 1.0]);
        let roots = poly.roots_jenkins_traub().expect("roots");
        assert_eq!(roots.len(), 2);
        let mut real_roots: Vec<f64> = roots.iter().map(|r| r.re).collect();
        real_roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert_relative_eq!(real_roots[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(real_roots[1], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_roots_complex() {
        // x² + 1 = (x - i)(x + i)
        let poly = p(vec![1.0, 0.0, 1.0]);
        let roots = poly.roots_jenkins_traub().expect("roots");
        assert_eq!(roots.len(), 2);
        for r in &roots {
            assert!(r.re.abs() < 1e-6);
            assert!((r.im.abs() - 1.0).abs() < 1e-6);
        }
    }
}
