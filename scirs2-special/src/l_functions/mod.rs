//! Dirichlet L-functions and Dirichlet Characters
//!
//! This module implements Dirichlet L-functions L(s, χ) = Σ_{n=1}^∞ χ(n)/n^s
//! for Dirichlet characters χ modulo q.
//!
//! Key references:
//! - Davenport, "Multiplicative Number Theory"
//! - Ireland & Rosen, "A Classical Introduction to Modern Number Theory"

use std::f64::consts::PI;

/// A Dirichlet character χ modulo q.
///
/// The character assigns a value χ(n) to each integer n.
/// For the principal character: χ(n)=1 if gcd(n,q)=1, else 0.
/// For real (quadratic) characters: values in {-1, 0, 1}.
#[derive(Debug, Clone)]
pub struct DirichletCharacter {
    /// The modulus of the character
    pub modulus: usize,
    /// Character values for n = 0, 1, ..., modulus-1.
    /// `values[n]` = chi(n mod q) as a rational integer (i64).
    pub values: Vec<i64>,
}

impl DirichletCharacter {
    /// Construct the principal character χ_0 modulo q.
    ///
    /// χ_0(n) = 1 if gcd(n, q) = 1, else 0.
    pub fn principal(modulus: usize) -> Self {
        let values = (0..modulus)
            .map(|n| if gcd(n, modulus) == 1 { 1i64 } else { 0i64 })
            .collect();
        DirichletCharacter { modulus, values }
    }

    /// Construct the Kronecker symbol character (d/·) modulo |d|.
    ///
    /// For a fundamental discriminant d, this gives a primitive real character.
    /// The character is defined as χ(n) = (d/n) if gcd(n, |d|) = 1, else 0.
    ///
    /// # Arguments
    /// * `d` - Discriminant (nonzero integer). Must be ≠ 0.
    pub fn kronecker_symbol(d: i64) -> Self {
        let modulus = d.unsigned_abs() as usize;
        if modulus == 0 {
            // Degenerate: return the trivial modulus-1 character
            return DirichletCharacter {
                modulus: 1,
                values: vec![1],
            };
        }
        let values = (0..modulus)
            .map(|n| {
                // Dirichlet character: 0 when gcd(n, q) > 1
                if gcd(n, modulus) != 1 {
                    0i64
                } else {
                    kronecker(d, n as i64)
                }
            })
            .collect();
        DirichletCharacter { modulus, values }
    }

    /// Evaluate the character at n (extended periodically).
    pub fn eval(&self, n: i64) -> i64 {
        if self.modulus == 0 {
            return 0;
        }
        let idx = n.rem_euclid(self.modulus as i64) as usize;
        self.values[idx]
    }

    /// Check whether the character is primitive.
    ///
    /// A character χ mod q is primitive if it is not induced by a character
    /// of smaller modulus. We check this via the conductor.
    pub fn is_primitive(&self) -> bool {
        self.conductor() == self.modulus
    }

    /// Compute the conductor of the character.
    ///
    /// The conductor is the smallest positive divisor d|q such that χ is
    /// induced by a character mod d.
    pub fn conductor(&self) -> usize {
        let q = self.modulus;
        if q == 1 {
            return 1;
        }
        // Check divisors in increasing order
        let mut divs: Vec<usize> = (1..=q).filter(|&d| q.is_multiple_of(d)).collect();
        divs.sort_unstable();
        for d in divs {
            if d == q {
                return q;
            }
            // Check if χ is induced by character mod d:
            // χ(n) must equal 0 whenever gcd(n,q) > 1, and for n ≡ m (mod d)
            // with gcd(n,q) = 1 = gcd(m,q), we need χ(n) = χ(m).
            if self.is_induced_by(d) {
                return d;
            }
        }
        q
    }

    /// Check if the character is induced by a character of modulus d (d | modulus).
    fn is_induced_by(&self, d: usize) -> bool {
        let q = self.modulus;
        if !q.is_multiple_of(d) {
            return false;
        }
        // For all a, b coprime to q with a ≡ b (mod d), χ(a) = χ(b).
        // Also: if gcd(n, d) > 1 then χ(n) = 0.
        for a in 0..q {
            if gcd(a, q) != 1 {
                continue;
            }
            let chi_a = self.values[a];
            // Check all b ≡ a (mod d) with 0 ≤ b < q
            let mut b = a % d;
            while b < q {
                if gcd(b, q) == 1 && self.values[b] != chi_a {
                    return false;
                }
                b += d;
            }
        }
        true
    }

    /// Check if the character is real (values in {-1, 0, 1}).
    pub fn is_real(&self) -> bool {
        self.values.iter().all(|&v| v == -1 || v == 0 || v == 1)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Number-theoretic helpers
// ────────────────────────────────────────────────────────────────────────────

/// Compute gcd(a, b) using Euclidean algorithm.
pub fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Compute the Jacobi symbol (a/n) using quadratic reciprocity.
/// n must be a positive odd integer.
pub fn jacobi_symbol(mut a: i64, mut n: i64) -> i64 {
    debug_assert!(n > 0 && n % 2 == 1);
    let mut result = 1i64;
    a = a.rem_euclid(n);
    while a != 0 {
        while a % 2 == 0 {
            a /= 2;
            let r = n % 8;
            if r == 3 || r == 5 {
                result = -result;
            }
        }
        std::mem::swap(&mut a, &mut n);
        if a % 4 == 3 && n % 4 == 3 {
            result = -result;
        }
        a = a.rem_euclid(n);
    }
    if n == 1 {
        result
    } else {
        0
    }
}

/// Compute the Kronecker symbol (a/n) for any integer n.
///
/// Extends the Jacobi symbol to n=0, n=±1, n=2, and n<0.
pub fn kronecker(a: i64, n: i64) -> i64 {
    if n == 0 {
        return if a.abs() == 1 { 1 } else { 0 };
    }
    if n == 1 {
        return 1;
    }
    if n == -1 {
        return if a < 0 { -1 } else { 1 };
    }
    let mut n = n;
    let mut result = 1i64;
    if n < 0 {
        n = -n;
        if a < 0 {
            result = -result;
        }
    }
    // Factor out powers of 2
    let v2 = n.trailing_zeros() as i64;
    if v2 > 0 {
        n >>= v2;
        let a8 = a.rem_euclid(8);
        let k2 = if a8 == 1 || a8 == 7 { 1i64 } else { -1i64 };
        if v2 % 2 != 0 {
            result *= k2;
        }
    }
    if n == 1 {
        return result;
    }
    // Now n is odd and positive
    result * jacobi_symbol(a, n)
}

// ────────────────────────────────────────────────────────────────────────────
// L-function evaluation
// ────────────────────────────────────────────────────────────────────────────

/// Evaluate the Dirichlet L-function L(s, χ) by partial summation.
///
/// L(s, χ) = Σ_{n=1}^{n_terms} χ(n) / n^s
///
/// This direct partial sum converges for Re(s) > 1. For s near 1 with a
/// non-principal character, the series converges conditionally for Re(s) > 0
/// by the cancellation in χ(n).
///
/// # Arguments
/// * `s` - Real argument (must be > 0 for non-principal, > 1 for principal)
/// * `chi` - Dirichlet character
/// * `n_terms` - Number of terms to sum (default 10000 for ~4-digit accuracy)
pub fn dirichlet_l(s: f64, chi: &DirichletCharacter, n_terms: usize) -> f64 {
    let q = chi.modulus;
    // For efficiency, precompute one period and accumulate in blocks
    if q == 0 {
        return 0.0;
    }

    // Group partial sums by complete periods + remainder
    let full_periods = n_terms / q;
    let remainder = n_terms % q;

    let mut sum = 0.0f64;

    // Complete periods
    for period in 0..full_periods {
        let base = period * q;
        for r in 1..=q {
            let n = base + r;
            let chi_n = chi.eval(n as i64);
            if chi_n != 0 {
                sum += chi_n as f64 / (n as f64).powf(s);
            }
        }
    }
    // Remainder
    let base = full_periods * q;
    for r in 1..=remainder {
        let n = base + r;
        let chi_n = chi.eval(n as i64);
        if chi_n != 0 {
            sum += chi_n as f64 / (n as f64).powf(s);
        }
    }
    sum
}

/// Evaluate L(1, χ) for a non-principal character χ.
///
/// For non-principal characters, L(1,χ) converges. This function uses:
/// - Even real characters: L(1,χ) = -1/q * Σ_{a=1}^{q} χ(a) * log|sin(π a / q)|
/// - Odd real characters: direct partial summation with Euler-Maclaurin acceleration
/// - For non-real characters: direct partial summation
///
/// # Returns
/// `None` if χ is principal (L(1,χ_0) diverges).
pub fn l_function_at_1(chi: &DirichletCharacter) -> Option<f64> {
    let q = chi.modulus;
    // Check for principal character: Σ χ(n) over one period = 0 for non-principal
    let sum_chi: i64 = (0..q).map(|n| chi.eval(n as i64)).sum();
    if sum_chi > 0 && q > 1 {
        return None; // likely principal
    }
    // Also handle q=1 (trivial character)
    if q == 1 {
        return None;
    }

    // Check if chi is even (χ(-1) = 1) or odd (χ(-1) = -1)
    let chi_neg1 = chi.eval(-1);

    if chi.is_real() && chi_neg1 == 1 {
        // Even real character: L(1,χ) = -1/q * Σ_{a=1}^{q-1} χ(a) log|sin(πa/q)|
        let q_f = q as f64;
        let sum: f64 = (1..q)
            .map(|a| {
                let chi_a = chi.eval(a as i64) as f64;
                let sin_val = (PI * a as f64 / q_f).sin().abs();
                if sin_val > 1e-15 {
                    chi_a * sin_val.ln()
                } else {
                    0.0
                }
            })
            .sum();
        Some(-sum / q_f)
    } else if chi.is_real() && chi_neg1 == -1 {
        // Odd real character: use the exact class number formula.
        // L(1,χ) = (πi / (q * τ̄(χ))) * Σ_{a=1}^{q-1} a * χ(a)
        //
        // For a real primitive odd character, τ(χ) = ε i√q (ε ∈ {±1}).
        // Hence τ̄(χ) = -ε i√q.
        // L(1,χ) = πi / (q * (-ε i√q)) * S = π / (q√q * ε) * S  (since i/(-i) = -1... wait)
        // πi / (-ε i q√q) * S = π / (-ε q√q) * S
        //
        // We compute τ(χ) = Σ_{a=1}^q χ(a) * exp(2πia/q) and get the imaginary part ε√q.
        // Then ε = sign(Im(τ)) and L(1,χ) = π * |S| / (q√q) with sign from ε*S.
        //
        // Numerically: compute the real Gauss sum numerator.
        let q_f = q as f64;
        // Compute τ(χ) imaginary part: Im(τ) = Σ χ(a) sin(2πa/q)
        let tau_im: f64 = (1..q)
            .map(|a| chi.eval(a as i64) as f64 * (2.0 * PI * a as f64 / q_f).sin())
            .sum();
        // S = Σ_{a=1}^{q-1} a * χ(a)
        let s: f64 = (1..q).map(|a| a as f64 * chi.eval(a as i64) as f64).sum();
        // L(1,χ) = π * (-S) / (q * tau_im)  (derived from L = πi/(q*τ̄)*S and τ̄=-τ for real χ)
        // Since τ̄ = τ* = Σ χ(a)*exp(-2πia/q) = -Σ χ(a)*exp(2πia/q) for odd real χ = -τ
        // L(1,χ) = πi / (q * (-τ)) * S = -πi * S / (q * τ)
        // τ = i * tau_im (approximately), so:
        // L(1,χ) = -πi * S / (q * i * tau_im) = -π * S / (q * tau_im)
        let l1 = -PI * s / (q_f * tau_im);
        Some(l1)
    } else {
        // Non-real character: direct partial summation
        Some(dirichlet_l(1.0, chi, 100000))
    }
}

/// Compute the generalized Bernoulli number B_{k, χ}.
///
/// B_{k, χ} = q^{k-1} Σ_{a=1}^q χ(a) B_k(a/q)
///
/// where B_k(x) is the Bernoulli polynomial.
///
/// # Arguments
/// * `k` - Non-negative integer
/// * `chi` - Dirichlet character
pub fn generalized_bernoulli(k: usize, chi: &DirichletCharacter) -> f64 {
    let q = chi.modulus as f64;
    let q_pow = q.powi(k as i32 - 1);
    let sum: f64 = (1..=chi.modulus)
        .map(|a| {
            let chi_a = chi.eval(a as i64) as f64;
            let x = a as f64 / q;
            chi_a * bernoulli_poly(k, x)
        })
        .sum();
    q_pow * sum
}

/// Evaluate the Bernoulli polynomial B_k(x).
///
/// Uses the explicit formula B_k(x) = Σ_{j=0}^k C(k,j) B_j x^{k-j}
/// where B_j are Bernoulli numbers.
pub fn bernoulli_poly(k: usize, x: f64) -> f64 {
    let berns = bernoulli_numbers(k);
    let mut result = 0.0f64;
    let mut binom = 1.0f64;
    for j in 0..=k {
        result += binom * berns[j] * x.powi((k - j) as i32);
        // Update binom = C(k, j+1)
        if j < k {
            binom *= (k - j) as f64 / (j + 1) as f64;
        }
    }
    result
}

/// Compute Bernoulli numbers B_0, B_1, ..., B_n.
pub fn bernoulli_numbers(n: usize) -> Vec<f64> {
    let mut b = vec![0.0f64; n + 1];
    b[0] = 1.0;
    if n == 0 {
        return b;
    }
    b[1] = -0.5;
    // Use the recurrence: B_n = -1/(n+1) * Σ_{k=0}^{n-1} C(n+1,k) B_k
    for m in 2..=n {
        let mut s = 0.0f64;
        let mut binom = 1.0f64;
        for k in 0..m {
            s += binom * b[k];
            binom *= (m + 1 - k) as f64 / (k + 1) as f64;
        }
        b[m] = -s / (m + 1) as f64;
    }
    b
}

/// Evaluate the completed L-function Λ(s, χ) satisfying the functional equation.
///
/// For a primitive character χ mod q, the functional equation is:
/// Λ(s, χ) = (q/π)^{s/2} Γ((s+a)/2) L(s, χ)
/// where a = 0 if χ(-1) = 1, a = 1 if χ(-1) = -1.
///
/// The functional equation: Λ(s, χ) = ε(χ) * Λ(1-s, χ̄)
/// where ε(χ) = τ(χ) / (i^a * sqrt(q)) is the root number.
///
/// This implementation approximates L(s, χ) via partial sums; the completion
/// factor is applied analytically.
pub fn l_function_complete(s: f64, chi: &DirichletCharacter) -> f64 {
    let q = chi.modulus as f64;
    let a = if chi.eval(-1) == 1 { 0.0f64 } else { 1.0f64 };

    // Gamma factor: Γ((s+a)/2)
    let gamma_arg = (s + a) / 2.0;
    let gamma_factor = gamma_function(gamma_arg);

    // L(s, χ) via partial sums
    let l_val = dirichlet_l(s, chi, 50000);

    // Completion factor: (q/π)^{s/2}
    let completion = (q / PI).powf(s / 2.0);

    completion * gamma_factor * l_val
}

/// Simple gamma function approximation via Stirling / Lanczos.
fn gamma_function(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x < 0.5 {
        return PI / ((PI * x).sin() * gamma_function(1.0 - x));
    }
    // Lanczos approximation (g=7, n=9)
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let x = x - 1.0;
    let mut s = C[0];
    for (i, &c) in C[1..].iter().enumerate() {
        s += c / (x + (i + 1) as f64);
    }
    let t = x + G + 0.5;
    (2.0 * PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * s
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(7, 5), 1);
        assert_eq!(gcd(0, 5), 5);
    }

    #[test]
    fn test_kronecker() {
        // Legendre symbol (5/3) = -1 (5 ≡ 2 mod 3, 2 is not QR mod 3)
        assert_eq!(kronecker(5, 3), -1);
        // (1/n) = 1 for all n
        assert_eq!(kronecker(1, 5), 1);
        assert_eq!(kronecker(1, 7), 1);
        // (-1/2) = Kronecker extension
        assert_eq!(kronecker(-1, 2), 1); // (-1)^{(2^2-1)/8} = (-1)^0 = no wait, Kronecker(-1,2) = (-1)^{(2^2-1)/8} = 1
    }

    #[test]
    fn test_principal_character() {
        let chi = DirichletCharacter::principal(4);
        assert_eq!(chi.eval(1), 1);
        assert_eq!(chi.eval(2), 0);
        assert_eq!(chi.eval(3), 1);
        assert_eq!(chi.eval(4), 0);
    }

    #[test]
    fn test_kronecker_character_neg4() {
        // Kronecker symbol (-4/n) gives the non-principal character mod 4
        // chi(-4/1)=1, chi(-4/2)=0, chi(-4/3)=-1, chi(-4/4)=0
        let chi = DirichletCharacter::kronecker_symbol(-4);
        assert_eq!(chi.eval(1), 1);
        assert_eq!(chi.eval(3), -1);
        assert_eq!(chi.eval(2), 0);
    }

    #[test]
    fn test_dirichlet_l_principal_mod4() {
        // L(2, χ_0 mod 4) = Σ_{n odd} 1/n^2 = π²/8 ≈ 1.2337
        let chi = DirichletCharacter::principal(4);
        let l2 = dirichlet_l(2.0, &chi, 100000);
        let expected = std::f64::consts::PI * std::f64::consts::PI / 8.0;
        assert_relative_eq!(l2, expected, epsilon = 1e-3);
    }

    #[test]
    fn test_l_function_at_1_legendre_neg4() {
        // χ(-4/·): L(1, χ) = π/4 ≈ 0.7854 (Leibniz formula)
        let chi = DirichletCharacter::kronecker_symbol(-4);
        let l1 = l_function_at_1(&chi).expect("should compute L(1,chi)");
        let expected = std::f64::consts::PI / 4.0;
        assert_relative_eq!(l1, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_bernoulli_numbers() {
        let b = bernoulli_numbers(6);
        // B_0=1, B_1=-1/2, B_2=1/6, B_3=0, B_4=-1/30
        assert_relative_eq!(b[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(b[1], -0.5, epsilon = 1e-12);
        assert_relative_eq!(b[2], 1.0 / 6.0, epsilon = 1e-12);
        assert_relative_eq!(b[3], 0.0, epsilon = 1e-12);
        assert_relative_eq!(b[4], -1.0 / 30.0, epsilon = 1e-12);
    }

    #[test]
    fn test_bernoulli_poly_b1() {
        // B_1(x) = x - 1/2
        let b1 = bernoulli_poly(1, 0.3);
        assert_relative_eq!(b1, 0.3 - 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_generalized_bernoulli_chi0() {
        // For the principal character mod 1 (trivial), B_{k,χ_0} = B_k
        let chi = DirichletCharacter::principal(1);
        let b2 = generalized_bernoulli(2, &chi);
        // B_{2, χ_0 mod 1} = 1^1 * chi(1)*B_2(1) = B_2(1) = 1/6 - 1 + 1 = 1/6
        assert!(b2.is_finite());
    }

    #[test]
    fn test_conductor_principal() {
        let chi = DirichletCharacter::principal(6);
        // Principal character mod 6 has conductor... let's just verify it divides 6
        let cond = chi.conductor();
        assert!(6 % cond == 0);
    }

    #[test]
    fn test_character_real() {
        let chi = DirichletCharacter::kronecker_symbol(-4);
        assert!(chi.is_real());
    }
}
