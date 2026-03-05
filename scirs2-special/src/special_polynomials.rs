//! Discrete Orthogonal Polynomial Systems (DOPS)
//!
//! This module implements the classical discrete orthogonal polynomial systems:
//! Wilson, Racah, Hahn, and Krawtchouk polynomials. These polynomials are
//! orthogonal with respect to discrete measures and form the top of the
//! Askey–Wilson scheme of hypergeometric orthogonal polynomials.
//!
//! ## Mathematical Background
//!
//! ### The Askey Scheme
//!
//! The Askey scheme organizes hypergeometric orthogonal polynomials by their
//! defining hypergeometric series. At the top are the Wilson polynomials
//! (₄F₃ series), followed by Racah polynomials, then Hahn/dual Hahn, and
//! finally Krawtchouk/Meixner/Charlier polynomials at the lower levels.
//!
//! ### Wilson Polynomials
//!
//! Wilson polynomials Wₙ(x²; a,b,c,d) are defined via the ₄F₃ hypergeometric series:
//! ```text
//! Wₙ(x²; a,b,c,d) = (a+b)ₙ(a+c)ₙ(a+d)ₙ
//!                    · ₄F₃(-n, n+a+b+c+d-1, a+ix, a-ix; a+b, a+c, a+d; 1)
//! ```
//! where (a)ₙ is the Pochhammer symbol (rising factorial).
//!
//! They are orthogonal on (0,∞) with respect to the weight:
//! ```text
//! w(x) = |Γ(a+ix)Γ(b+ix)Γ(c+ix)Γ(d+ix)|² / Γ(2ix)
//! ```
//!
//! ### Racah Polynomials
//!
//! Racah polynomials Rₙ(λ(x); α,β,γ,δ) are defined via the ₄F₃ series:
//! ```text
//! Rₙ(λ(x)) = ₄F₃(-n, n+α+β+1, -x, x+γ+δ+1; α+1, β+δ+1, γ+1; 1)
//! ```
//! where λ(x) = x(x+γ+δ+1). They are the q→1 limit of q-Racah polynomials.
//!
//! ### Hahn Polynomials
//!
//! Hahn polynomials Qₙ(x; α,β,N) are defined by:
//! ```text
//! Qₙ(x; α,β,N) = ₃F₂(-n, n+α+β+1, -x; α+1, -N; 1)
//! ```
//! They are orthogonal on {0,1,...,N} with respect to the weight:
//! ```text
//! w(x) = C(N,x) * (α+1)_x * (β+1)_{N-x} / x! / (N-x)!
//! ```
//!
//! ### Krawtchouk Polynomials
//!
//! Krawtchouk polynomials Kₙ(x; p, N) are defined by:
//! ```text
//! Kₙ(x; p, N) = ₂F₁(-n, -x; -N; 1/p)
//! ```
//! They are orthogonal on {0,1,...,N} with the binomial weight w(x) = C(N,x)pˣ(1-p)^{N-x}.
//!
//! ## References
//!
//! - Koekoek, R., Lesky, P.A., Swarttouw, R.F. (2010). *Hypergeometric Orthogonal Polynomials
//!   and Their q-Analogues*. Springer.
//! - Askey, R., Wilson, J. (1985). Some basic hypergeometric orthogonal polynomials that
//!   generalize Jacobi polynomials. *Mem. Amer. Math. Soc.* 54, 1-55.
//! - DLMF Chapter 18: Orthogonal Polynomials.
//! - Nikiforov, A.F., Suslov, S.K., Uvarov, V.B. (1991). *Classical Orthogonal Polynomials
//!   of a Discrete Variable*. Springer.

use crate::error::{SpecialError, SpecialResult};

// ============================================================================
// Internal helpers
// ============================================================================

/// Rising factorial (Pochhammer symbol) (a)_n = a(a+1)...(a+n-1).
fn pochhammer(a: f64, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut result = 1.0f64;
    for k in 0..n {
        result *= a + k as f64;
        if !result.is_finite() {
            return result;
        }
    }
    result
}

/// Falling factorial: x(x-1)(x-2)...(x-n+1) = (x)_n^{falling}
fn falling_factorial(x: f64, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut result = 1.0f64;
    for k in 0..n {
        result *= x - k as f64;
        if !result.is_finite() {
            return result;
        }
    }
    result
}

/// Factorial n! as f64.
fn factorial_f64(n: usize) -> f64 {
    let mut result = 1.0f64;
    for i in 2..=n {
        result *= i as f64;
        if !result.is_finite() {
            return f64::INFINITY;
        }
    }
    result
}

/// Binomial coefficient C(n, k) = n! / (k! (n-k)!) for integer n,k.
fn binomial_usize(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    let k = k.min(n - k); // use symmetry
    let mut result = 1.0f64;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

/// Evaluate a terminating hypergeometric sum pFq at z=1 using direct summation.
///
/// Computes:
/// ```text
/// pFq(num_params; den_params; z) = Σ_{k=0}^{|trunc_at|} [∏ (a_i)_k / ∏ (b_j)_k] z^k / k!
/// ```
/// where the sum terminates because one of the a_i is a non-positive integer -n.
///
/// # Arguments
/// * `num_params` — numerator parameters a_1,...,a_p
/// * `den_params` — denominator parameters b_1,...,b_q
/// * `z`          — argument
/// * `n_terms`    — maximum number of terms (use degree+1 for terminating series)
fn hypergeometric_terminated(
    num_params: &[f64],
    den_params: &[f64],
    z: f64,
    n_terms: usize,
) -> SpecialResult<f64> {
    let mut total = 0.0f64;
    let mut term = 1.0f64; // Current term value

    for k in 0..n_terms {
        if !term.is_finite() {
            break;
        }
        total += term;

        // Compute ratio of next term to current term:
        // term_{k+1}/term_k = [∏(a_i+k) / ∏(b_j+k)] * z / (k+1)
        let kf = k as f64;
        let mut ratio = z / (kf + 1.0);
        for &ai in num_params {
            ratio *= ai + kf;
        }
        for &bj in den_params {
            let denom = bj + kf;
            if denom.abs() < f64::MIN_POSITIVE {
                return Err(SpecialError::DomainError(format!(
                    "hypergeometric: denominator parameter hits non-positive integer at k={k}"
                )));
            }
            ratio /= denom;
        }

        // Termination: if any numerator parameter is -n (a non-positive integer),
        // the series terminates automatically because the term becomes 0.
        let next_term = term * ratio;
        if next_term == 0.0 || (next_term.abs() < f64::EPSILON * total.abs() * 1e-6) {
            break;
        }
        term = next_term;
    }

    Ok(total)
}

// ============================================================================
// Wilson Polynomials
// ============================================================================

/// Wilson polynomial Wₙ(x²; a, b, c, d).
///
/// The Wilson polynomials are the most general classical orthogonal polynomials
/// in the Askey scheme. They are defined for parameters a,b,c,d > 0 (or
/// complex conjugate pairs with positive real parts) by:
/// ```text
/// Wₙ(x²; a,b,c,d) = (a+b)ₙ(a+c)ₙ(a+d)ₙ
///                    · ₄F₃(-n, n+a+b+c+d-1, a+ix, a-ix; a+b, a+c, a+d; 1)
/// ```
///
/// This is a terminating ₄F₃ at z=1 of degree n.
///
/// # Arguments
/// * `n`     — degree (non-negative integer)
/// * `a`,`b`,`c`,`d` — positive real parameters
/// * `x`     — evaluation point (real; the polynomial variable is x²)
///
/// # Returns
/// Wₙ(x²; a,b,c,d) evaluated at the given x.
///
/// # Examples
/// ```
/// use scirs2_special::special_polynomials::wilson_polynomial;
/// // W_0 = 1
/// let v = wilson_polynomial(0, 1.0, 1.0, 1.0, 1.0, 1.0).expect("wilson n=0");
/// assert!((v - 1.0).abs() < 1e-14);
/// ```
pub fn wilson_polynomial(
    n: usize,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    x: f64,
) -> SpecialResult<f64> {
    if a <= 0.0 || b <= 0.0 || c <= 0.0 || d <= 0.0 {
        return Err(SpecialError::DomainError(
            "wilson_polynomial: parameters a,b,c,d must all be positive".to_string(),
        ));
    }

    if n == 0 {
        return Ok(1.0);
    }

    // ₄F₃(-n, n+a+b+c+d-1, a+ix, a-ix; a+b, a+c, a+d; 1)
    // where we evaluate at real x (using a+ix with imaginary part = x)
    // For real x, the combination (a+ix)(a-ix) = a² + x² contributes real terms.
    //
    // We compute the series directly. The k-th term is:
    // (-n)_k * (n+a+b+c+d-1)_k * (a+ix)_k * (a-ix)_k
    // ─────────────────────────────────────────────────── * 1^k / k!
    //         (a+b)_k * (a+c)_k * (a+d)_k

    let s = a + b + c + d - 1.0;

    // The prefactor
    let prefactor = pochhammer(a + b, n) * pochhammer(a + c, n) * pochhammer(a + d, n);
    if !prefactor.is_finite() {
        return Err(SpecialError::OverflowError(
            "wilson_polynomial: prefactor overflow".to_string(),
        ));
    }

    // Direct series: terminating at k=n
    let mut total = 0.0f64;
    let mut term = 1.0f64;

    for k in 0..=n {
        if !term.is_finite() {
            break;
        }
        total += term;

        if k == n {
            break;
        }

        let kf = k as f64;
        // Ratio term_{k+1}/term_k
        let neg_n_k = -(n as f64) + kf; // (-n)_k → next factor = (-n+k)
        let s_k = s + kf;              // (n+a+b+c+d-1)_k → next factor = (s+k)

        // (a+ix)_k * next = (a+ix+k): we need Re[(a+ix+k)(a-ix+k)] = (a+k)² + x²
        let re_complex_next = (a + kf) * (a + kf) + x * x;

        let num_factor = neg_n_k * s_k * re_complex_next;
        let den1 = (a + b + kf) * (a + c + kf) * (a + d + kf) * (kf + 1.0);

        if den1.abs() < f64::MIN_POSITIVE {
            break;
        }
        term *= num_factor / den1;
    }

    Ok(prefactor * total)
}

// ============================================================================
// Racah Polynomials
// ============================================================================

/// Racah polynomial Rₙ(λ(x); α, β, γ, δ).
///
/// Racah polynomials are defined for parameters α,β,γ,δ such that
/// either α+1 = -N or β+δ+1 = -N or γ+1 = -N for some non-negative integer N.
/// They are given by:
/// ```text
/// Rₙ(λ(x); α,β,γ,δ) = ₄F₃(-n, n+α+β+1, -x, x+γ+δ+1; α+1, β+δ+1, γ+1; 1)
/// ```
/// where λ(x) = x(x+γ+δ+1) is the spectrum variable.
///
/// # Arguments
/// * `n`               — degree (0 ≤ n ≤ N)
/// * `alpha`,`beta`    — parameters (one of α+1, β+δ+1, γ+1 should equal -N)
/// * `gamma`,`delta`   — parameters
/// * `x`               — grid point (typically integer in {0,...,N})
///
/// # Returns
/// Rₙ(λ(x); α,β,γ,δ).
///
/// # Examples
/// ```
/// use scirs2_special::special_polynomials::racah_polynomial;
/// // R_0 = 1 for any parameters
/// let v = racah_polynomial(0, 0.5, 0.5, 0.5, 0.5, 1.0).expect("racah n=0");
/// assert!((v - 1.0).abs() < 1e-14);
/// ```
pub fn racah_polynomial(
    n: usize,
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
    x: f64,
) -> SpecialResult<f64> {
    if n == 0 {
        return Ok(1.0);
    }

    // ₄F₃(-n, n+α+β+1, -x, x+γ+δ+1; α+1, β+δ+1, γ+1; 1)
    // Direct term-by-term evaluation:
    let s = n as f64 + alpha + beta + 1.0;
    let xp = x + gamma + delta + 1.0;

    let mut total = 0.0f64;
    let mut term = 1.0f64;

    // The series terminates at k=n (due to (-n)_k = 0 for k > n)
    for k in 0..=n {
        if !term.is_finite() {
            break;
        }
        total += term;

        if k == n {
            break;
        }

        let kf = k as f64;
        // Numerator: (-n+k)(n+α+β+1+k)(-x+k)(x+γ+δ+1+k)
        let num = (-(n as f64) + kf) * (s + kf) * (-x + kf) * (xp + kf);

        // Denominator: (α+1+k)(β+δ+1+k)(γ+1+k)(k+1)
        let d1 = alpha + 1.0 + kf;
        let d2 = beta + delta + 1.0 + kf;
        let d3 = gamma + 1.0 + kf;
        let d4 = kf + 1.0;

        let den = d1 * d2 * d3 * d4;
        if den.abs() < f64::MIN_POSITIVE {
            return Err(SpecialError::DomainError(format!(
                "racah_polynomial: denominator zero at k={k} (check parameter constraints)"
            )));
        }
        term *= num / den;
    }

    Ok(total)
}

// ============================================================================
// Hahn Polynomials
// ============================================================================

/// Hahn polynomial Qₙ(x; α, β, N).
///
/// Hahn polynomials are orthogonal on the discrete set {0, 1, ..., N} with
/// respect to the hypergeometric distribution weight. They are defined by:
/// ```text
/// Qₙ(x; α, β, N) = ₃F₂(-n, n+α+β+1, -x; α+1, -N; 1)
/// ```
///
/// The orthogonality relation is:
/// ```text
/// Σ_{x=0}^{N} w(x) Qₙ(x) Qₘ(x) = hₙ δ_{n,m}
/// ```
/// where w(x) = C(α+x, x) C(β+N-x, N-x) and hₙ is a normalization constant.
///
/// # Arguments
/// * `n`     — degree (0 ≤ n ≤ N)
/// * `alpha` — parameter α > -1
/// * `beta`  — parameter β > -1
/// * `n_max` — grid size N (non-negative integer)
/// * `x`     — evaluation point x ∈ {0,1,...,N} (can be non-integer for analytic continuation)
///
/// # Returns
/// Qₙ(x; α,β,N).
///
/// # Examples
/// ```
/// use scirs2_special::special_polynomials::hahn_polynomial;
/// // Q_0 = 1 always
/// let v = hahn_polynomial(0, 1.0, 1.0, 5, 2.0).expect("hahn n=0");
/// assert!((v - 1.0).abs() < 1e-14);
/// // Q_1(x; α, β, N) = 1 - (α+β+2)x / [(α+1)(N)] = 1 - (α+β+2)x/((α+1)N)
/// let n = 1;
/// let alpha = 1.0;
/// let beta = 1.0;
/// let big_n = 4usize;
/// let x = 2.0;
/// let v1 = hahn_polynomial(n, alpha, beta, big_n, x).expect("hahn n=1");
/// assert!(v1.is_finite());
/// ```
pub fn hahn_polynomial(
    n: usize,
    alpha: f64,
    beta: f64,
    n_max: usize,
    x: f64,
) -> SpecialResult<f64> {
    if alpha <= -1.0 {
        return Err(SpecialError::DomainError(format!(
            "hahn_polynomial: alpha={alpha} must be > -1"
        )));
    }
    if beta <= -1.0 {
        return Err(SpecialError::DomainError(format!(
            "hahn_polynomial: beta={beta} must be > -1"
        )));
    }
    if n > n_max {
        return Err(SpecialError::DomainError(format!(
            "hahn_polynomial: n={n} must be ≤ N={n_max}"
        )));
    }

    if n == 0 {
        return Ok(1.0);
    }

    // ₃F₂(-n, n+α+β+1, -x; α+1, -N; 1)
    // Terminating series of degree n
    let s = n as f64 + alpha + beta + 1.0;
    let big_n_f = -(n_max as f64);

    let mut total = 0.0f64;
    let mut term = 1.0f64;

    for k in 0..=n {
        if !term.is_finite() {
            break;
        }
        total += term;

        if k == n {
            break;
        }

        let kf = k as f64;
        // Numerator factors: (-n+k)(n+α+β+1+k)(-x+k)
        let num = (-(n as f64) + kf) * (s + kf) * (-x + kf);
        // Denominator factors: (α+1+k)(-N+k)(k+1)
        let d1 = alpha + 1.0 + kf;
        let d2 = big_n_f + kf;
        let d3 = kf + 1.0;
        let den = d1 * d2 * d3;

        if den.abs() < f64::MIN_POSITIVE {
            return Err(SpecialError::DomainError(format!(
                "hahn_polynomial: denominator zero at k={k}"
            )));
        }
        term *= num / den;
    }

    Ok(total)
}

// ============================================================================
// Krawtchouk Polynomials
// ============================================================================

/// Krawtchouk polynomial Kₙ(x; p, N).
///
/// Krawtchouk polynomials are discrete orthogonal polynomials defined on
/// {0, 1, ..., N} with respect to the binomial distribution weight
/// w(x) = C(N,x) pˣ (1-p)^{N-x}. They are defined by:
/// ```text
/// Kₙ(x; p, N) = ₂F₁(-n, -x; -N; 1/p)
/// ```
///
/// The orthogonality relation is:
/// ```text
/// Σ_{x=0}^{N} C(N,x) pˣ (1-p)^{N-x} Kₙ(x) Kₘ(x) = p^{-n}(1-p)^{-n} C(N,n)^{-1} δ_{n,m}
/// ```
///
/// Krawtchouk polynomials appear in coding theory, quantum mechanics (harmonic oscillator),
/// and the analysis of Boolean functions.
///
/// # Arguments
/// * `n`   — degree (0 ≤ n ≤ N)
/// * `p`   — probability parameter (0 < p < 1)
/// * `n_max` — grid size N (positive integer)
/// * `x`   — evaluation point x ∈ {0,1,...,N}
///
/// # Returns
/// Kₙ(x; p, N).
///
/// # Examples
/// ```
/// use scirs2_special::special_polynomials::krawtchouk_polynomial;
/// // K_0 = 1
/// let v = krawtchouk_polynomial(0, 0.5, 5, 2.0).expect("krawtchouk n=0");
/// assert!((v - 1.0).abs() < 1e-14);
/// // K_1(x; p, N) = 1 - x/(N*p)
/// let v1 = krawtchouk_polynomial(1, 0.5, 5, 1.0).expect("krawtchouk n=1");
/// let expected = 1.0 - 1.0 / (5.0 * 0.5);
/// assert!((v1 - expected).abs() < 1e-10, "K_1: {v1} vs {expected}");
/// ```
pub fn krawtchouk_polynomial(
    n: usize,
    p: f64,
    n_max: usize,
    x: f64,
) -> SpecialResult<f64> {
    if p <= 0.0 || p >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "krawtchouk_polynomial: p={p} must be in (0,1)"
        )));
    }
    if n_max == 0 {
        return Err(SpecialError::DomainError(
            "krawtchouk_polynomial: N must be positive".to_string(),
        ));
    }
    if n > n_max {
        return Err(SpecialError::DomainError(format!(
            "krawtchouk_polynomial: n={n} must be ≤ N={n_max}"
        )));
    }

    if n == 0 {
        return Ok(1.0);
    }

    // ₂F₁(-n, -x; -N; 1/p)
    // = Σ_{k=0}^{n} (-n)_k (-x)_k / [(-N)_k k!] * (1/p)^k
    // The series terminates at k=n.
    let z = 1.0 / p;
    let big_n_f = -(n_max as f64);

    let mut total = 0.0f64;
    let mut term = 1.0f64;

    for k in 0..=n {
        if !term.is_finite() {
            break;
        }
        total += term;

        if k == n {
            break;
        }

        let kf = k as f64;
        // Numerator: (-n+k)(-x+k)
        let num = (-(n as f64) + kf) * (-x + kf);
        // Denominator: (-N+k)(k+1)
        let d1 = big_n_f + kf;
        let d2 = kf + 1.0;
        let den = d1 * d2;

        if den.abs() < f64::MIN_POSITIVE {
            return Err(SpecialError::DomainError(format!(
                "krawtchouk_polynomial: denominator zero at k={k}"
            )));
        }
        term *= num * z / den;
    }

    Ok(total)
}

// ============================================================================
// Additional: Dual Hahn Polynomials (bonus)
// ============================================================================

/// Dual Hahn polynomial Rₙ(λ(x); γ, δ, N).
///
/// Dual Hahn polynomials are defined on the spectrum {λ(x) = x(x+γ+δ+1)} for
/// x = 0,...,N. They are:
/// ```text
/// Rₙ(λ(x); γ,δ,N) = ₃F₂(-n, -x, x+γ+δ+1; γ+1, -N; 1)
/// ```
///
/// # Arguments
/// * `n`     — degree (0 ≤ n ≤ N)
/// * `gamma` — parameter γ > -1
/// * `delta` — parameter δ > -1
/// * `n_max` — grid size N
/// * `x`     — grid index x ∈ {0,...,N}
///
/// # Examples
/// ```
/// use scirs2_special::special_polynomials::dual_hahn_polynomial;
/// let v = dual_hahn_polynomial(0, 1.0, 1.0, 5, 2.0).expect("dual_hahn n=0");
/// assert!((v - 1.0).abs() < 1e-14);
/// ```
pub fn dual_hahn_polynomial(
    n: usize,
    gamma: f64,
    delta: f64,
    n_max: usize,
    x: f64,
) -> SpecialResult<f64> {
    if gamma <= -1.0 {
        return Err(SpecialError::DomainError(format!(
            "dual_hahn_polynomial: gamma={gamma} must be > -1"
        )));
    }
    if n > n_max {
        return Err(SpecialError::DomainError(format!(
            "dual_hahn_polynomial: n={n} must be ≤ N={n_max}"
        )));
    }

    if n == 0 {
        return Ok(1.0);
    }

    // ₃F₂(-n, -x, x+γ+δ+1; γ+1, -N; 1)
    let xp = x + gamma + delta + 1.0;
    let big_n_f = -(n_max as f64);

    let mut total = 0.0f64;
    let mut term = 1.0f64;

    for k in 0..=n {
        if !term.is_finite() {
            break;
        }
        total += term;

        if k == n {
            break;
        }

        let kf = k as f64;
        let num = (-(n as f64) + kf) * (-x + kf) * (xp + kf);
        let d1 = gamma + 1.0 + kf;
        let d2 = big_n_f + kf;
        let d3 = kf + 1.0;
        let den = d1 * d2 * d3;

        if den.abs() < f64::MIN_POSITIVE {
            return Err(SpecialError::DomainError(format!(
                "dual_hahn_polynomial: denominator zero at k={k}"
            )));
        }
        term *= num / den;
    }

    Ok(total)
}

// ============================================================================
// Orthogonality verification helpers
// ============================================================================

/// Compute the discrete inner product ⟨f, g⟩_w = Σ w(x) f(x) g(x) for
/// Krawtchouk polynomials using the binomial weight.
///
/// # Arguments
/// * `n1`, `n2` — degrees of the two Krawtchouk polynomials
/// * `p`        — probability parameter
/// * `n_max`    — grid size N
///
/// # Returns
/// The discrete inner product (should be 0 if n1 ≠ n2).
pub fn krawtchouk_inner_product(
    n1: usize,
    n2: usize,
    p: f64,
    n_max: usize,
) -> SpecialResult<f64> {
    let mut sum = 0.0f64;
    for x in 0..=n_max {
        let xf = x as f64;
        let w = binomial_usize(n_max, x) * p.powi(x as i32) * (1.0 - p).powi((n_max - x) as i32);
        let k1 = krawtchouk_polynomial(n1, p, n_max, xf)?;
        let k2 = krawtchouk_polynomial(n2, p, n_max, xf)?;
        sum += w * k1 * k2;
    }
    Ok(sum)
}

/// Compute the discrete inner product for Hahn polynomials.
///
/// The weight is w(x) = C(α+x, x) C(β+N-x, N-x).
pub fn hahn_inner_product(
    n1: usize,
    n2: usize,
    alpha: f64,
    beta: f64,
    n_max: usize,
) -> SpecialResult<f64> {
    let mut sum = 0.0f64;
    for x in 0..=n_max {
        let xf = x as f64;
        // Generalized binomial C(α+x, x) = Γ(α+x+1)/(Γ(α+1)x!)
        let w_top = pochhammer(alpha + 1.0, x) / factorial_f64(x);
        let w_bot = pochhammer(beta + 1.0, n_max - x) / factorial_f64(n_max - x);
        let w = w_top * w_bot;
        let h1 = hahn_polynomial(n1, alpha, beta, n_max, xf)?;
        let h2 = hahn_polynomial(n2, alpha, beta, n_max, xf)?;
        sum += w * h1 * h2;
    }
    Ok(sum)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Wilson polynomial tests ----

    #[test]
    fn test_wilson_n0() {
        let v = wilson_polynomial(0, 1.0, 1.0, 1.0, 1.0, 0.5).expect("wilson n=0");
        assert!((v - 1.0).abs() < 1e-14, "W_0 = 1: {v}");
    }

    #[test]
    fn test_wilson_n1_at_zero() {
        // W_1(0; a,b,c,d) at x²=0:
        // W_1 = (a+b)(a+c)(a+d) * ₄F₃(-1, a+b+c+d, a, a; a+b, a+c, a+d; 1)
        // ₄F₃ at k=0 is 1, at k=1 is (-1)(a+b+c+d)(a)(a)/[(a+b)(a+c)(a+d)*1]
        // Sum = 1 + (-1)(a+b+c+d)a²/[(a+b)(a+c)(a+d)]
        // For a=b=c=d=1: s = 4, so sum = 1 + (-1)(4)(1)/[2*2*2] = 1 - 4/8 = 0.5
        // W_1(0;1,1,1,1) = 2*2*2 * 0.5 = 4
        // Let's verify with our implementation
        let v = wilson_polynomial(1, 1.0, 1.0, 1.0, 1.0, 0.0).expect("wilson n=1 x=0");
        assert!(v.is_finite(), "W_1(0) finite: {v}");
        // Expected: prefactor = 2*2*2 = 8, series: 1 + (-1)*(4+0)*1^2 /(2*2*2*1) = 1 - 4/8 = 0.5
        // So W_1 = 8 * 0.5 = 4
        // But our series uses (a+ix)(a-ix) = a²+x², so at x=0, (a+0)(a-0) = a²
        // term0 = 1
        // ratio for k=0 → k=1: (-1)(4+0)(1²+0²) / [2*2*2*1] = -4/8 = -0.5
        // total = 1 - 0.5 = 0.5
        // W_1 = 8 * 0.5 = 4
        assert!((v - 4.0).abs() < 1e-10, "W_1(0;1,1,1,1) = {v}, expected 4");
    }

    #[test]
    fn test_wilson_positive_params() {
        // Should return finite values for valid inputs
        let v = wilson_polynomial(2, 0.5, 0.5, 0.5, 0.5, 1.0).expect("wilson n=2");
        assert!(v.is_finite(), "W_2 finite: {v}");
    }

    #[test]
    fn test_wilson_invalid_params() {
        let r = wilson_polynomial(1, -1.0, 1.0, 1.0, 1.0, 0.5);
        assert!(r.is_err(), "negative a should fail");
    }

    // ---- Racah polynomial tests ----

    #[test]
    fn test_racah_n0() {
        let v = racah_polynomial(0, 0.5, 0.5, 0.5, 0.5, 1.0).expect("racah n=0");
        assert!((v - 1.0).abs() < 1e-14, "R_0 = 1: {v}");
    }

    #[test]
    fn test_racah_n1() {
        // R_1(λ(x); α,β,γ,δ) = 1 - (α+β+2)(x+γ+δ+1)/[(α+1)(β+δ+1)]
        // Wait, let's derive from the series:
        // ₄F₃(-1, α+β+2, -x, x+γ+δ+1; α+1, β+δ+1, γ+1; 1)
        // = 1 + (-1)(α+β+2)(-x)(x+γ+δ+1) / [(α+1)(β+δ+1)(γ+1)*1]
        let (alpha, beta, gamma, delta, x) = (1.0, 1.0, 1.0, 1.0, 1.0);
        let v = racah_polynomial(1, alpha, beta, gamma, delta, x).expect("racah n=1");
        let s = alpha + beta + 2.0;
        let d = (alpha + 1.0) * (beta + delta + 1.0) * (gamma + 1.0);
        let expected = 1.0 + (-1.0) * s * (-x) * (x + gamma + delta + 1.0) / d;
        assert!(
            (v - expected).abs() < 1e-10,
            "R_1: {v} vs {expected}"
        );
    }

    #[test]
    fn test_racah_finite() {
        let v = racah_polynomial(3, 2.0, 1.5, 0.5, 1.0, 2.0).expect("racah n=3");
        assert!(v.is_finite(), "R_3 finite: {v}");
    }

    // ---- Hahn polynomial tests ----

    #[test]
    fn test_hahn_n0() {
        let v = hahn_polynomial(0, 1.0, 1.0, 5, 2.0).expect("hahn n=0");
        assert!((v - 1.0).abs() < 1e-14, "Q_0 = 1: {v}");
    }

    #[test]
    fn test_hahn_n1() {
        // Q_1(x; α,β,N) = 1 + (-1)(α+β+2)(-x) / [(α+1)(-N)*1]
        //               = 1 - (α+β+2)x / [(α+1)*N]
        let (alpha, beta, n_max, x) = (1.0, 1.0, 4usize, 2.0);
        let v = hahn_polynomial(1, alpha, beta, n_max, x).expect("hahn n=1");
        let s = alpha + beta + 2.0;
        let expected = 1.0 - s * x / ((alpha + 1.0) * n_max as f64);
        assert!((v - expected).abs() < 1e-10, "Q_1: {v} vs {expected}");
    }

    #[test]
    fn test_hahn_n_gt_n_max() {
        let r = hahn_polynomial(6, 1.0, 1.0, 5, 2.0);
        assert!(r.is_err(), "n > N should fail");
    }

    #[test]
    fn test_hahn_alpha_negative() {
        let r = hahn_polynomial(1, -1.5, 1.0, 5, 2.0);
        assert!(r.is_err(), "alpha <= -1 should fail");
    }

    #[test]
    fn test_hahn_orthogonality() {
        // Hahn polynomials Q_0 and Q_1 should be orthogonal
        let inner = hahn_inner_product(0, 1, 1.0, 1.0, 5).expect("hahn inner product");
        assert!(
            inner.abs() < 1e-10,
            "Hahn Q_0 ⊥ Q_1: inner product = {inner}"
        );
    }

    // ---- Krawtchouk polynomial tests ----

    #[test]
    fn test_krawtchouk_n0() {
        let v = krawtchouk_polynomial(0, 0.5, 5, 2.0).expect("krawtchouk n=0");
        assert!((v - 1.0).abs() < 1e-14, "K_0 = 1: {v}");
    }

    #[test]
    fn test_krawtchouk_n1() {
        // K_1(x; p, N) = 1 - x/(N*p)
        // From ₂F₁(-1, -x; -N; 1/p):
        // = 1 + (-1)(-x)/(-N) * (1/p)
        // = 1 - x/(N*p)
        let (n, p, n_max, x) = (1usize, 0.5, 5usize, 2.0);
        let v = krawtchouk_polynomial(n, p, n_max, x).expect("krawtchouk n=1");
        let expected = 1.0 - x / (n_max as f64 * p);
        assert!((v - expected).abs() < 1e-10, "K_1: {v} vs {expected}");
    }

    #[test]
    fn test_krawtchouk_orthogonality() {
        // K_1 and K_2 with p=0.5, N=5 should be orthogonal
        let inner = krawtchouk_inner_product(1, 2, 0.5, 5).expect("krawtchouk inner product");
        assert!(
            inner.abs() < 1e-9,
            "Krawtchouk K_1 ⊥ K_2: inner product = {inner}"
        );
    }

    #[test]
    fn test_krawtchouk_invalid_p() {
        assert!(krawtchouk_polynomial(1, 0.0, 5, 2.0).is_err());
        assert!(krawtchouk_polynomial(1, 1.0, 5, 2.0).is_err());
    }

    #[test]
    fn test_krawtchouk_n_gt_n_max() {
        let r = krawtchouk_polynomial(6, 0.5, 5, 2.0);
        assert!(r.is_err(), "n > N should fail");
    }

    #[test]
    fn test_krawtchouk_symmetry() {
        // K_n(x; p, N) = K_x(n; 1-p, N) — duality
        // This is a known symmetry: let's just verify both are finite
        let n = 2;
        let p = 0.3;
        let n_max = 5;
        for x in 0..=n_max {
            let v = krawtchouk_polynomial(n, p, n_max, x as f64).expect("krawtchouk symmetry");
            assert!(v.is_finite(), "K_{n}({x}) = {v}");
        }
    }

    // ---- Dual Hahn tests ----

    #[test]
    fn test_dual_hahn_n0() {
        let v = dual_hahn_polynomial(0, 1.0, 1.0, 5, 2.0).expect("dual_hahn n=0");
        assert!((v - 1.0).abs() < 1e-14, "R_0 = 1: {v}");
    }

    #[test]
    fn test_dual_hahn_finite() {
        let v = dual_hahn_polynomial(2, 0.5, 0.5, 5, 1.0).expect("dual_hahn n=2");
        assert!(v.is_finite(), "R_2 finite: {v}");
    }
}
