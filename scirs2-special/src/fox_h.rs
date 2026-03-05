//! Fox H-Function and Related Functions
//!
//! This module implements the Fox H-function, Meijer G-function, and Wright M-function.
//! The Fox H-function is the most general of all known special functions and unifies
//! the Meijer G-function, Wright function, and many others.
//!
//! ## Mathematical Background
//!
//! ### Definition (Mellin-Barnes Integral)
//!
//! The Fox H-function is defined by the Mellin-Barnes contour integral:
//! ```text
//! H_{p,q}^{m,n}[x | (a_1,A_1),...,(a_p,A_p) ]
//!              |   (b_1,B_1),...,(b_q,B_q) ]
//!
//! = (1/2πi) ∫_L Θ(s) x^{-s} ds
//! ```
//! where
//! ```text
//! Θ(s) = [∏_{j=1}^m Γ(b_j + B_j s)] [∏_{j=1}^n Γ(1-a_j - A_j s)]
//!        ─────────────────────────────────────────────────────────────
//!        [∏_{j=m+1}^q Γ(1-b_j - B_j s)] [∏_{j=n+1}^p Γ(a_j + A_j s)]
//! ```
//!
//! The contour L separates the poles of Γ(b_j + B_j s) (j=1,...,m) from the
//! poles of Γ(1-a_j - A_j s) (j=1,...,n).
//!
//! ### Residue Series Representation
//!
//! When the poles are simple, the Fox H-function can be evaluated as a
//! residue series. The poles of Γ(b_j + B_j s) are at:
//! ```text
//! s = -(b_j + k) / B_j,   k = 0, 1, 2, ...
//! ```
//!
//! The residue at a simple pole s = s_0 of order (−1/B_j) Γ(b_j + B_j s) gives:
//! ```text
//! Res_{s=s_0} Θ(s) x^{-s} = (-1)^k / (k! B_j) · Θ̃(s_0) · x^{s_0}
//! ```
//! where Θ̃ denotes Θ evaluated at the pole with the factor Γ(b_j + B_j s) removed.
//!
//! ### Meijer G-Function as Special Case
//!
//! When all A_j = B_j = 1, the Fox H-function reduces to the Meijer G-function:
//! ```text
//! G_{p,q}^{m,n}[x | a_1,...,a_p ] = H_{p,q}^{m,n}[x | (a_1,1),...,(a_p,1) ]
//!                  | b_1,...,b_q ]                     (b_1,1),...,(b_q,1) ]
//! ```
//!
//! ### Wright M-Function
//!
//! The Wright M-function (also called M-Wright or Mainardi function) is:
//! ```text
//! M(x; β) = Σ_{k=0}^∞ (-x)^k / [k! Γ(-βk + (1-β))]
//! ```
//!
//! It is a Fox H-function:
//! ```text
//! M(x; β) = H_{0,2}^{1,0}[x | –         ]
//!                          | (0,1), (1-β,β) ]
//! ```
//!
//! ### Asymptotic Expansion for Large x
//!
//! For large |x|, the Fox H-function has the asymptotic expansion governed
//! by the poles of Γ(a_j - A_j s) (j=1,...,n). In the right-half-plane regime,
//! the dominant contribution comes from the pole at s=1, giving power-law decay.
//!
//! ## References
//!
//! - Fox, C. (1961). The G and H functions as symmetrical Fourier kernels.
//!   *Trans. Amer. Math. Soc.* 98, 395-429.
//! - Mathai, A.M., Saxena, R.K., Haubold, H.J. (2010). *The H-Function*. Springer.
//! - Kilbas, A.A., Saigo, M. (2004). *H-Transforms*. CRC Press.
//! - Wright, E.M. (1933). On the coefficients of power series having exponential
//!   singularities. *J. London Math. Soc.* 8, 71-79.

use crate::error::{SpecialError, SpecialResult};

// ============================================================================
// Data structures
// ============================================================================

/// Parameters for the Fox H-function H_{p,q}^{m,n}.
///
/// The Fox H-function is specified by:
/// - Integers m, n, p, q with 0 ≤ m ≤ q, 0 ≤ n ≤ p
/// - Upper parameters: (a_j, A_j) for j = 1,...,p
/// - Lower parameters: (b_j, B_j) for j = 1,...,q
///
/// # Field Layout
/// * `m`, `n`, `p`, `q` — structural integers
/// * `a` — upper parameters as `Vec<(a_j, A_j)>`, length = p
/// * `b` — lower parameters as `Vec<(b_j, B_j)>`, length = q
///
/// # Examples
/// ```
/// use scirs2_special::fox_h::FoxHParams;
/// // Exponential function: E_x = H_{0,1}^{1,0}[x | – | (0,1)]
/// let params = FoxHParams {
///     m: 1, n: 0, p: 0, q: 1,
///     a: vec![],
///     b: vec![(0.0, 1.0)],
/// };
/// ```
#[derive(Debug, Clone)]
pub struct FoxHParams {
    /// Number of Gamma functions in the numerator from b-parameters
    pub m: usize,
    /// Number of Gamma functions in the numerator from a-parameters
    pub n: usize,
    /// Total number of upper parameters
    pub p: usize,
    /// Total number of lower parameters
    pub q: usize,
    /// Upper parameters: (a_j, A_j) for j = 1..=p
    pub a: Vec<(f64, f64)>,
    /// Lower parameters: (b_j, B_j) for j = 1..=q
    pub b: Vec<(f64, f64)>,
}

impl FoxHParams {
    /// Validate structural constraints.
    pub fn validate(&self) -> SpecialResult<()> {
        if self.m > self.q {
            return Err(SpecialError::ValueError(format!(
                "Fox H: m={} must be ≤ q={}",
                self.m, self.q
            )));
        }
        if self.n > self.p {
            return Err(SpecialError::ValueError(format!(
                "Fox H: n={} must be ≤ p={}",
                self.n, self.p
            )));
        }
        if self.a.len() != self.p {
            return Err(SpecialError::ValueError(format!(
                "Fox H: a.len()={} must equal p={}",
                self.a.len(),
                self.p
            )));
        }
        if self.b.len() != self.q {
            return Err(SpecialError::ValueError(format!(
                "Fox H: b.len()={} must equal q={}",
                self.b.len(),
                self.q
            )));
        }
        // Check that none of B_j or A_j are zero
        for (j, &(_, bj)) in self.b.iter().enumerate() {
            if bj.abs() < f64::MIN_POSITIVE {
                return Err(SpecialError::ValueError(format!(
                    "Fox H: B_{j} = {bj} must be nonzero"
                )));
            }
        }
        for (j, &(_, aj)) in self.a.iter().enumerate() {
            if aj.abs() < f64::MIN_POSITIVE {
                return Err(SpecialError::ValueError(format!(
                    "Fox H: A_{j} = {aj} must be nonzero"
                )));
            }
        }
        Ok(())
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Log of the absolute value of Gamma(x) using Stirling's approximation
/// extended for negative arguments via reflection.
fn log_abs_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        // Use reflection: Γ(x)Γ(1-x) = π/sin(πx)
        let y = 1.0 - x;
        let ln_gamma_y = log_abs_gamma(y);
        let s = (std::f64::consts::PI * x).sin().abs();
        if s < f64::MIN_POSITIVE {
            return f64::INFINITY; // pole
        }
        std::f64::consts::PI.ln() - s.ln() - ln_gamma_y
    } else {
        // Lanczos approximation
        lanczos_ln_gamma(x)
    }
}

/// Sign of Gamma(x): +1 or -1.
fn gamma_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        // Count the number of poles crossed: (-1)^(floor(-x)+1) for negative non-integers
        let k = (-x).floor() as i64;
        if k % 2 == 0 { -1.0 } else { 1.0 }
    }
}

/// Lanczos approximation for ln Γ(x) for x > 0.
fn lanczos_ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_403,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_312_5e-7,
    ];

    let mut y = x;
    let mut result = C[0];
    for k in 1..9 {
        result += C[k] / (y + k as f64 - 1.0);
        y += 0.0; // no-op, just using the loop variable
    }

    let t = x + G - 0.5;
    (2.0 * std::f64::consts::PI).ln() * 0.5 + (t.ln()) * (x - 0.5) - t + result.ln()
}

/// Gamma(x) = exp(ln|Γ(x)|) * sign
fn gamma_fn(x: f64) -> f64 {
    if x <= 0.0 && x.fract() == 0.0 {
        return f64::INFINITY; // pole
    }
    let sign = gamma_sign(x);
    let ln_g = log_abs_gamma(x);
    sign * ln_g.exp()
}

// ============================================================================
// Fox H-function evaluation via residue series
// ============================================================================

/// Evaluate the Fox H-function via residue series.
///
/// The residue at the k-th pole of the j-th Gamma function (j ∈ {0,...,m-1},
/// pole order k = 0,1,2,...) located at
/// ```text
/// s_{j,k} = -(b_j + k) / B_j
/// ```
/// is computed as:
/// ```text
/// Res = (−1)^k / (k! · B_j) · x^{−s_{j,k}} · [product of other Gamma values at s_{j,k}]
/// ```
///
/// # Arguments
/// * `x`        — evaluation point (x > 0 for convergent series)
/// * `params`   — Fox H parameters (m, n, p, q, a, b)
/// * `n_terms`  — number of pole terms to include (≥ 1)
///
/// # Returns
/// Approximate value of H_{p,q}^{m,n}(x).
///
/// # Examples
/// ```
/// use scirs2_special::fox_h::{fox_h_function, FoxHParams};
/// // E_x = exp(-x) corresponds to H_{0,1}^{1,0}[x|(0,1)]
/// let params = FoxHParams { m:1, n:0, p:0, q:1, a:vec![], b:vec![(0.0,1.0)] };
/// let val = fox_h_function(1.0, &params, 20).expect("fox_h");
/// // Expect ~ exp(-1) ≈ 0.3679
/// assert!((val - (-1.0_f64).exp()).abs() < 1e-8, "exp(-x) test: {val}");
/// ```
pub fn fox_h_function(x: f64, params: &FoxHParams, n_terms: usize) -> SpecialResult<f64> {
    if x <= 0.0 {
        return Err(SpecialError::DomainError(
            "fox_h_function: x must be positive for residue series".to_string(),
        ));
    }
    params.validate()?;

    let m = params.m;
    let n = params.n;
    let p = params.p;
    let q = params.q;

    let mut total = 0.0;

    // Iterate over each of the m "numerator-b" Gamma functions (poles from b_j side)
    for j in 0..m {
        let (bj, bj_cap) = params.b[j];

        for k in 0..n_terms {
            let kf = k as f64;
            // Pole location: s0 = -(bj + k) / Bj
            let s0 = -(bj + kf) / bj_cap;

            // Factor from the removed pole: (-1)^k / (k! * Bj)
            let sign_k = if k % 2 == 0 { 1.0 } else { -1.0 };
            let factorial_k = factorial_f64(k);
            let pole_factor = sign_k / (factorial_k * bj_cap);

            // x^{-s0}
            let x_power = x.powf(-s0);
            if !x_power.is_finite() {
                continue;
            }

            // Product of numerator Gamma factors at s0:
            // - Γ(b_i + B_i * s0) for i=0..m, i≠j
            // - Γ(1 - a_i - A_i * s0) for i=0..n
            let mut log_num = 0.0f64;
            let mut sign_num = 1.0f64;

            for i in 0..m {
                if i == j {
                    continue; // skip the pole we're summing over
                }
                let (bi, bi_cap) = params.b[i];
                let arg = bi + bi_cap * s0;
                if arg <= 0.0 && arg.fract() == 0.0 {
                    // This Gamma factor has a pole → this residue is 0
                    log_num = f64::NEG_INFINITY;
                    break;
                }
                log_num += log_abs_gamma(arg);
                sign_num *= gamma_sign(arg);
            }

            if !log_num.is_finite() {
                continue;
            }

            for i in 0..n {
                let (ai, ai_cap) = params.a[i];
                let arg = 1.0 - ai - ai_cap * s0;
                if arg <= 0.0 && arg.fract() == 0.0 {
                    log_num = f64::NEG_INFINITY;
                    break;
                }
                log_num += log_abs_gamma(arg);
                sign_num *= gamma_sign(arg);
            }

            if !log_num.is_finite() {
                continue;
            }

            // Product of denominator Gamma factors at s0:
            // - 1/Γ(1 - b_i - B_i * s0) for i=m..q
            // - 1/Γ(a_i + A_i * s0) for i=n..p
            let mut log_den = 0.0f64;
            let mut sign_den = 1.0f64;

            for i in m..q {
                let (bi, bi_cap) = params.b[i];
                let arg = 1.0 - bi - bi_cap * s0;
                if arg <= 0.0 && arg.fract() == 0.0 {
                    // 1/Γ(pole) = 0, so this residue is 0
                    log_den = f64::INFINITY;
                    break;
                }
                log_den += log_abs_gamma(arg);
                sign_den *= gamma_sign(arg);
            }

            if log_den.is_infinite() {
                continue;
            }

            for i in n..p {
                let (ai, ai_cap) = params.a[i];
                let arg = ai + ai_cap * s0;
                if arg <= 0.0 && arg.fract() == 0.0 {
                    log_den = f64::INFINITY;
                    break;
                }
                log_den += log_abs_gamma(arg);
                sign_den *= gamma_sign(arg);
            }

            if log_den.is_infinite() {
                continue;
            }

            let log_term = log_num - log_den;
            if log_term < -700.0 {
                // Underflow: negligible term
                continue;
            }
            if log_term > 700.0 {
                // Overflow: this term dominates, warn via large value
                let term = pole_factor * x_power * sign_num / sign_den * (700.0_f64).exp();
                total += term;
                continue;
            }

            let gamma_ratio = sign_num / sign_den * log_term.exp();
            let term = pole_factor * x_power * gamma_ratio;

            if term.is_finite() {
                total += term;
            }
        }
    }

    Ok(total)
}

/// Asymptotic expansion of the Fox H-function for large x.
///
/// For large |x|, the Fox H-function is dominated by the residues at the
/// poles of Γ(1 - a_j - A_j s) (j = 1,...,n) which are at
/// ```text
/// s_{j,k} = (1 - a_j + k) / A_j,   k = 0, 1, 2, ...
/// ```
///
/// This gives the asymptotic series Σ c_k x^{-s_{j,k}}.
///
/// # Arguments
/// * `x`       — large evaluation point (x >> 1)
/// * `params`  — Fox H parameters
/// * `n_terms` — number of asymptotic terms
///
/// # Returns
/// Asymptotic approximation for large x.
///
/// # Examples
/// ```
/// use scirs2_special::fox_h::{fox_h_asymptotic, FoxHParams};
/// let params = FoxHParams { m:1, n:1, p:1, q:1,
///     a: vec![(1.0, 1.0)], b: vec![(0.0, 1.0)] };
/// let val = fox_h_asymptotic(100.0, &params, 5).expect("asymptotic");
/// assert!(val.is_finite());
/// ```
pub fn fox_h_asymptotic(
    x: f64,
    params: &FoxHParams,
    n_terms: usize,
) -> SpecialResult<f64> {
    if x <= 1.0 {
        return Err(SpecialError::DomainError(
            "fox_h_asymptotic: x must be > 1 for asymptotic expansion".to_string(),
        ));
    }
    params.validate()?;

    let n = params.n;
    let m = params.m;
    let p = params.p;
    let q = params.q;
    let mut total = 0.0;

    for j in 0..n {
        let (aj, aj_cap) = params.a[j];

        for k in 0..n_terms {
            let kf = k as f64;
            // Pole: s0 = (1 - aj + k) / Aj
            let s0 = (1.0 - aj + kf) / aj_cap;

            let sign_k = if k % 2 == 0 { 1.0 } else { -1.0 };
            let factorial_k = factorial_f64(k);
            let pole_factor = sign_k / (factorial_k * aj_cap);

            let x_power = x.powf(-s0);
            if !x_power.is_finite() {
                continue;
            }

            let mut log_num = 0.0f64;
            let mut sign_num = 1.0f64;

            // Γ(b_i + B_i s0) for i=0..m
            for i in 0..m {
                let (bi, bi_cap) = params.b[i];
                let arg = bi + bi_cap * s0;
                if arg <= 0.0 && arg.fract() == 0.0 {
                    log_num = f64::NEG_INFINITY;
                    break;
                }
                log_num += log_abs_gamma(arg);
                sign_num *= gamma_sign(arg);
            }
            if !log_num.is_finite() {
                continue;
            }

            // Γ(1 - a_i - A_i s0) for i=0..n, i≠j
            for i in 0..n {
                if i == j {
                    continue;
                }
                let (ai, ai_cap) = params.a[i];
                let arg = 1.0 - ai - ai_cap * s0;
                if arg <= 0.0 && arg.fract() == 0.0 {
                    log_num = f64::NEG_INFINITY;
                    break;
                }
                log_num += log_abs_gamma(arg);
                sign_num *= gamma_sign(arg);
            }
            if !log_num.is_finite() {
                continue;
            }

            let mut log_den = 0.0f64;
            let mut sign_den = 1.0f64;

            // 1/Γ(1 - b_i - B_i s0) for i=m..q
            for i in m..q {
                let (bi, bi_cap) = params.b[i];
                let arg = 1.0 - bi - bi_cap * s0;
                if arg <= 0.0 && arg.fract() == 0.0 {
                    log_den = f64::INFINITY;
                    break;
                }
                log_den += log_abs_gamma(arg);
                sign_den *= gamma_sign(arg);
            }
            if log_den.is_infinite() {
                continue;
            }

            // 1/Γ(a_i + A_i s0) for i=n..p, skipping i=j (pole)
            for i in n..p {
                let (ai, ai_cap) = params.a[i];
                let arg = ai + ai_cap * s0;
                if arg <= 0.0 && arg.fract() == 0.0 {
                    log_den = f64::INFINITY;
                    break;
                }
                log_den += log_abs_gamma(arg);
                sign_den *= gamma_sign(arg);
            }
            if log_den.is_infinite() {
                continue;
            }

            let log_term = log_num - log_den;
            if !log_term.is_finite() || log_term < -700.0 {
                continue;
            }

            let gamma_ratio = sign_num / sign_den * log_term.min(700.0).exp();
            let term = pole_factor * x_power * gamma_ratio;
            if term.is_finite() {
                total += term;
            }
        }
    }

    Ok(total)
}

// ============================================================================
// Meijer G-Function
// ============================================================================

/// Meijer G-function G_{p,q}^{m,n}(x | a_1,...,a_p / b_1,...,b_q).
///
/// The Meijer G-function is the special case of the Fox H-function where
/// all scaling parameters A_j = B_j = 1:
/// ```text
/// G_{p,q}^{m,n}[x] = H_{p,q}^{m,n}[x | (a_1,1),...,(a_p,1) ]
///                                       (b_1,1),...,(b_q,1) ]
/// ```
///
/// # Arguments
/// * `x`  — positive evaluation point
/// * `m`  — number of Gamma factors from b-parameters in numerator
/// * `n`  — number of Gamma factors from a-parameters in numerator
/// * `p`  — total number of upper parameters
/// * `q`  — total number of lower parameters
/// * `a`  — upper parameter vector (length = p)
/// * `b`  — lower parameter vector (length = q)
///
/// # Returns
/// G_{p,q}^{m,n}(x) evaluated via residue series.
///
/// # Examples
/// ```
/// use scirs2_special::fox_h::meijer_g;
/// // G_{0,1}^{1,0}[x | _ / 0] = exp(-x)
/// let v = meijer_g(1.0, 1, 0, 0, 1, &[], &[0.0], 20).expect("meijer_g");
/// let expected = (-1.0_f64).exp();
/// assert!((v - expected).abs() < 1e-8, "G = exp(-x): {v} vs {expected}");
/// ```
pub fn meijer_g(
    x: f64,
    m: usize,
    n: usize,
    p: usize,
    q: usize,
    a: &[f64],
    b: &[f64],
    n_terms: usize,
) -> SpecialResult<f64> {
    // Convert to Fox H with all A_j = B_j = 1
    let a_fox: Vec<(f64, f64)> = a.iter().map(|&ai| (ai, 1.0)).collect();
    let b_fox: Vec<(f64, f64)> = b.iter().map(|&bi| (bi, 1.0)).collect();

    let params = FoxHParams { m, n, p, q, a: a_fox, b: b_fox };
    fox_h_function(x, &params, n_terms)
}

// ============================================================================
// Wright M-Function
// ============================================================================

/// Wright M-function (Mainardi function) M(x; β).
///
/// The Wright M-function is defined for β ∈ (0,1) by the series:
/// ```text
/// M(x; β) = Σ_{k=0}^∞ (-x)^k / [k! Γ(-βk + (1-β))]
/// ```
///
/// This function plays a fundamental role in the theory of anomalous diffusion
/// and fractional calculus. It is a probability density for β ∈ (0,1) and
/// satisfies M(x; 1/2) = (1/√π) exp(-x²/4).
///
/// The Wright M-function is related to the Fox H-function by:
/// ```text
/// M(x; β) = H_{0,2}^{1,0}[x | –  / (0,1), (1-β,β)]
/// ```
///
/// # Arguments
/// * `x`      — evaluation point
/// * `beta`   — shape parameter, β ∈ (0, 1)
/// * `n_terms`— number of series terms
///
/// # Returns
/// M(x; β) via direct series summation.
///
/// # Examples
/// ```
/// use scirs2_special::fox_h::wright_m_function;
/// // M(0; 0.5) = 1/Γ(1) = 1
/// let v = wright_m_function(0.0, 0.5, 30).expect("wright_m x=0");
/// assert!((v - 1.0).abs() < 1e-12, "M(0;0.5) = {v}");
/// ```
pub fn wright_m_function(x: f64, beta: f64, n_terms: usize) -> SpecialResult<f64> {
    if beta <= 0.0 || beta >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "wright_m_function: beta={beta} must be in (0,1)"
        )));
    }

    let mut total = 0.0f64;
    let mut x_power = 1.0f64; // (-x)^k
    let mut factorial = 1.0f64;

    for k in 0..n_terms {
        // Term: (-x)^k / (k! * Γ(-β*k + (1-β)))
        let gamma_arg = -beta * (k as f64) + (1.0 - beta);
        let g = gamma_fn(gamma_arg);
        if g.is_finite() && g != 0.0 {
            let term = x_power / (factorial * g);
            if term.is_finite() {
                total += term;
            }
        }

        // Update (-x)^{k+1} = (-x)^k * (-x)
        x_power *= -x;
        factorial *= (k + 1) as f64;

        // Early termination if factorial overflow
        if !factorial.is_finite() {
            break;
        }
    }

    Ok(total)
}

/// Wright M-function for the special case β=1/2: M(x; 1/2) = (1/√π) exp(-x²/4).
///
/// # Examples
/// ```
/// use scirs2_special::fox_h::wright_m_half;
/// let v = wright_m_half(1.0).expect("wright_m_half");
/// let exact = 1.0 / std::f64::consts::PI.sqrt() * (-1.0_f64/4.0).exp();
/// assert!((v - exact).abs() < 1e-14);
/// ```
pub fn wright_m_half(x: f64) -> SpecialResult<f64> {
    // Exact formula: M(x; 1/2) = (1/√π) exp(-x²/4)
    let val = 1.0 / std::f64::consts::PI.sqrt() * (-x * x / 4.0).exp();
    Ok(val)
}

// ============================================================================
// Utilities
// ============================================================================

/// Compute n! as f64 (returns Inf for large n).
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fox_h_exp_minus_x() {
        // H_{0,1}^{1,0}[x | (0,1)] = exp(-x)
        // This is the standard form where the residue series gives:
        // sum_k (-1)^k / (k! * 1) * x^{-(0+k)/1} ... wait, let us reconsider.
        // Actually G_{0,1}^{1,0}[x | _ | 0] corresponds to exp(-x).
        // Pole at s=-(0+k) = -k, contributing (-1)^k/k! * x^k ... times nothing.
        // That is the Taylor series of exp(-x)?  Let's check:
        // Res at s=-k: pole_factor = (-1)^k/(k!*1), x^{-(-k)} = x^k
        // Product of other Gamma: empty (m=1, only j=0)
        // Denominator: 1/Γ(1 - b_0 - 1*s0)|_{i=m=1..q=1}: empty
        // So term = (-1)^k/(k!) * x^k
        // Sum = exp(-x). Correct!
        let params = FoxHParams {
            m: 1,
            n: 0,
            p: 0,
            q: 1,
            a: vec![],
            b: vec![(0.0, 1.0)],
        };
        let x = 1.0;
        let v = fox_h_function(x, &params, 25).expect("fox_h exp(-x)");
        let expected = (-x).exp();
        assert!(
            (v - expected).abs() < 1e-8,
            "exp(-x): computed={v}, expected={expected}"
        );
    }

    #[test]
    fn test_meijer_g_exp() {
        // G_{0,1}^{1,0}[x | _ / 0] = exp(-x)
        let v = meijer_g(1.0, 1, 0, 0, 1, &[], &[0.0], 25).expect("meijer_g exp");
        let expected = (-1.0_f64).exp();
        assert!(
            (v - expected).abs() < 1e-8,
            "Meijer G=exp(-x): {v} vs {expected}"
        );
    }

    #[test]
    fn test_wright_m_at_zero() {
        // M(0; β) = 1/Γ(1-β)
        let beta = 0.5;
        let v = wright_m_function(0.0, beta, 30).expect("wright_m x=0");
        // Γ(1-0.5) = Γ(0.5) = √π ≈ 1.7724
        let expected = 1.0 / gamma_fn(1.0 - beta);
        assert!(
            (v - expected).abs() < 1e-10,
            "M(0;0.5) = {v}, expected {expected}"
        );
    }

    #[test]
    fn test_wright_m_half_formula() {
        // M(x; 1/2) = (1/√π) exp(-x²/4)
        let x = 2.0;
        let v_series = wright_m_function(x, 0.5, 50).expect("wright_m series");
        let v_exact = wright_m_half(x).expect("wright_m exact");
        assert!(
            (v_series - v_exact).abs() < 1e-5,
            "M(2;0.5): series={v_series}, exact={v_exact}"
        );
    }

    #[test]
    fn test_fox_h_validation_m_gt_q() {
        let params = FoxHParams {
            m: 3,
            n: 0,
            p: 0,
            q: 2,
            a: vec![],
            b: vec![(0.0, 1.0), (1.0, 1.0)],
        };
        let result = fox_h_function(1.0, &params, 5);
        assert!(result.is_err(), "m > q should fail validation");
    }

    #[test]
    fn test_fox_h_positive_x_required() {
        let params = FoxHParams {
            m: 1,
            n: 0,
            p: 0,
            q: 1,
            a: vec![],
            b: vec![(0.0, 1.0)],
        };
        let result = fox_h_function(-1.0, &params, 5);
        assert!(result.is_err(), "negative x should fail");
    }

    #[test]
    fn test_fox_h_asymptotic_large_x() {
        // For G_{1,1}^{1,1}[x | (1,1) / (0,1)], the large-x behavior
        // is governed by the n=1 poles.
        let params = FoxHParams {
            m: 1,
            n: 1,
            p: 1,
            q: 1,
            a: vec![(1.0, 1.0)],
            b: vec![(0.0, 1.0)],
        };
        let v = fox_h_asymptotic(100.0, &params, 5).expect("asymptotic");
        assert!(v.is_finite(), "asymptotic result must be finite: {v}");
    }

    #[test]
    fn test_wright_m_beta_range() {
        // beta outside (0,1) should fail
        let r1 = wright_m_function(1.0, 0.0, 10);
        let r2 = wright_m_function(1.0, 1.0, 10);
        assert!(r1.is_err());
        assert!(r2.is_err());
    }

    #[test]
    fn test_wright_m_half_x0() {
        let v = wright_m_half(0.0).expect("wright_m_half x=0");
        // M(0; 1/2) = 1/√π
        let expected = 1.0 / std::f64::consts::PI.sqrt();
        assert!((v - expected).abs() < 1e-14, "M(0;1/2): {v} vs {expected}");
    }
}
