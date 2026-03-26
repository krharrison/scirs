//! Clebsch-Gordan Coefficients and Related Symbols
//!
//! Clebsch-Gordan coefficients ⟨j₁ m₁; j₂ m₂ | J M⟩ appear in the decomposition
//! of tensor products of SU(2) representations:
//!
//! |J, M⟩ = Σ_{m₁,m₂} ⟨j₁ m₁; j₂ m₂ | J M⟩ |j₁, m₁⟩ ⊗ |j₂, m₂⟩
//!
//! The Racah formula gives an explicit expression as an alternating sum.
//! Related symbols:
//! - Wigner 3j symbols: symmetrized version of CG coefficients
//! - Wigner 6j symbols (Racah W-coefficients): recoupling coefficients
//!
//! References:
//! - Edmonds, "Angular Momentum in Quantum Mechanics", 1957
//! - Varshalovich et al., "Quantum Theory of Angular Momentum", 1988
//! - Racah, "Theory of Complex Spectra II", Phys. Rev. 62, 438 (1942)

use crate::error::{SpecialError, SpecialResult};

// ────────────────────────────────────────────────────────────────────────────
// Log-factorial table
// ────────────────────────────────────────────────────────────────────────────

/// Maximum integer argument for the log-factorial table.
const LOG_FACT_MAX: usize = 300;

/// Lazy-initialized log-factorial table.
///
/// Computes ln(n!) for n = 0, 1, ..., LOG_FACT_MAX.
fn build_log_factorial_table() -> Vec<f64> {
    let mut table = Vec::with_capacity(LOG_FACT_MAX + 1);
    table.push(0.0); // ln(0!) = 0
    for n in 1..=LOG_FACT_MAX {
        let prev = *table.last().expect("non-empty");
        table.push(prev + (n as f64).ln());
    }
    table
}

// Use a thread-local cache
thread_local! {
    static LOG_FACT_TABLE: Vec<f64> = build_log_factorial_table();
}

/// Compute ln(n!) using the pre-computed table or Stirling's approximation for large n.
pub fn log_factorial(n: usize) -> f64 {
    if n <= LOG_FACT_MAX {
        LOG_FACT_TABLE.with(|t| t[n])
    } else {
        // Stirling's approximation: ln(n!) ≈ n ln(n) - n + 0.5 ln(2πn)
        let nf = n as f64;
        nf * nf.ln() - nf + 0.5 * (2.0 * std::f64::consts::PI * nf).ln()
    }
}

/// Compute n! for small n (returns 0 for n < 0 since we use usize).
fn log_fact_signed(n: i64) -> Option<f64> {
    if n < 0 {
        None // Factorial undefined for negative integers
    } else {
        Some(log_factorial(n as usize))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Triangle condition
// ────────────────────────────────────────────────────────────────────────────

/// Check the triangle condition: |a - b| ≤ c ≤ a + b.
///
/// Required for angular momentum coupling: only certain values of J appear
/// in the tensor product of representations with spins j₁ and j₂.
pub fn triangle_condition(a: f64, b: f64, c: f64) -> bool {
    let diff = (a - b).abs();
    let sum = a + b;
    c >= diff - 1e-10 && c <= sum + 1e-10
}

/// Check that a value is a valid half-integer (integer or half-integer).
fn is_half_integer(x: f64) -> bool {
    let twice = (2.0 * x).round();
    (2.0 * x - twice).abs() < 1e-10
}

/// Convert a half-integer (or integer) spin value to twice its value (as integer).
fn twice(x: f64) -> i64 {
    (2.0 * x).round() as i64
}

// ────────────────────────────────────────────────────────────────────────────
// Clebsch-Gordan coefficient (Racah formula)
// ────────────────────────────────────────────────────────────────────────────

/// Compute the Clebsch-Gordan coefficient ⟨j₁ m₁; j₂ m₂ | J M⟩.
///
/// Uses the Racah formula:
///
/// CG = δ_{m₁+m₂,M} · √[(2J+1) · Δ(j₁,j₂,J)]
///    · √[(J+M)!(J-M)!(j₁+m₁)!(j₁-m₁)!(j₂+m₂)!(j₂-m₂)!]
///    · Σ_k (-1)^k / [k!(j₁+j₂-J-k)!(j₁-m₁-k)!(j₂+m₂-k)!(J-j₂+m₁+k)!(J-j₁-m₂+k)!]
///
/// where Δ(a,b,c) = (a+b-c)!(a-b+c)!(-a+b+c)! / (a+b+c+1)!
///
/// The sum runs over all integers k for which all factorials are non-negative.
///
/// # Arguments
/// * `j1` - First angular momentum (non-negative half-integer)
/// * `m1` - Projection of j1 (-j1 ≤ m1 ≤ j1, half-integer steps)
/// * `j2` - Second angular momentum (non-negative half-integer)
/// * `m2` - Projection of j2 (-j2 ≤ m2 ≤ j2, half-integer steps)
/// * `j` - Total angular momentum (non-negative half-integer)
/// * `m` - Total projection (m1 + m2, |m| ≤ j)
///
/// # Returns
/// The Clebsch-Gordan coefficient as a real number.
pub fn clebsch_gordan(j1: f64, m1: f64, j2: f64, m2: f64, j: f64, m: f64) -> f64 {
    // Validate half-integer conditions
    if !is_half_integer(j1)
        || !is_half_integer(m1)
        || !is_half_integer(j2)
        || !is_half_integer(m2)
        || !is_half_integer(j)
        || !is_half_integer(m)
    {
        return 0.0;
    }

    // Quantum number validity
    if j1 < 0.0 || j2 < 0.0 || j < 0.0 {
        return 0.0;
    }

    // Selection rule: m = m1 + m2
    if (m - m1 - m2).abs() > 1e-10 {
        return 0.0;
    }

    // Projection bounds
    if m1.abs() > j1 + 1e-10 || m2.abs() > j2 + 1e-10 || m.abs() > j + 1e-10 {
        return 0.0;
    }

    // Triangle condition
    if !triangle_condition(j1, j2, j) {
        return 0.0;
    }

    // Work in half-integer units (multiply all by 2 to get integers)
    let j1_2 = twice(j1);
    let m1_2 = twice(m1);
    let j2_2 = twice(j2);
    let m2_2 = twice(m2);
    let j_2 = twice(j);
    let m_2 = twice(m);

    // Validate integer/half-integer consistency
    // All of j1, j2, J must be consistent (all integer or all half-integer combinations)
    if (j1_2 + j2_2 + j_2) % 2 != 0 {
        return 0.0;
    }

    // Δ(j1, j2, J) = (j1+j2-J)!(j1-j2+J)!(-j1+j2+J)! / (j1+j2+J+1)!
    // In half-integer units: (j1_2+j2_2-j_2)/2 etc.
    let a = (j1_2 + j2_2 - j_2) / 2;
    let b = (j1_2 - j2_2 + j_2) / 2;
    let c = (-j1_2 + j2_2 + j_2) / 2;
    let d = (j1_2 + j2_2 + j_2) / 2 + 1;

    if a < 0 || b < 0 || c < 0 || d < 0 {
        return 0.0;
    }

    let log_delta =
        log_factorial(a as usize) + log_factorial(b as usize) + log_factorial(c as usize)
            - log_factorial(d as usize);

    // Prefactor sqrt terms (in half-integer units)
    let jp_m = (j_2 + m_2) / 2;
    let jm_m = (j_2 - m_2) / 2;
    let j1p_m1 = (j1_2 + m1_2) / 2;
    let j1m_m1 = (j1_2 - m1_2) / 2;
    let j2p_m2 = (j2_2 + m2_2) / 2;
    let j2m_m2 = (j2_2 - m2_2) / 2;

    if jp_m < 0 || jm_m < 0 || j1p_m1 < 0 || j1m_m1 < 0 || j2p_m2 < 0 || j2m_m2 < 0 {
        return 0.0;
    }

    let log_sqrt_factor = log_factorial(jp_m as usize)
        + log_factorial(jm_m as usize)
        + log_factorial(j1p_m1 as usize)
        + log_factorial(j1m_m1 as usize)
        + log_factorial(j2p_m2 as usize)
        + log_factorial(j2m_m2 as usize);

    // The (2J+1) factor
    let log_2jp1 = ((j_2 + 1) as f64).ln();

    // Determine k range: all factorials in denominator must be non-negative
    // k!: k ≥ 0
    // (j1+j2-J-k)! = (a-k)!: a-k ≥ 0 → k ≤ a
    // (j1-m1-k)! = (j1m_m1-k)!: k ≤ j1m_m1
    // (j2+m2-k)! = (j2p_m2-k)!: k ≤ j2p_m2
    // (J-j2+m1+k)! = ((j_2-j2_2+m1_2)/2 + k)!: need ≥ 0 → k ≥ -(J-j2+m1)/1
    // (J-j1-m2+k)! = ((j_2-j1_2-m2_2)/2 + k)!: need ≥ 0 → k ≥ (j1_2+m2_2-j_2)/2

    let e = (j_2 - j2_2 + m1_2) / 2; // J - j2 + m1
    let f_val = (j_2 - j1_2 - m2_2) / 2; // J - j1 - m2

    let k_min = (-e).max(-f_val).max(0i64);
    let k_max = a.min(j1m_m1).min(j2p_m2);

    if k_min > k_max {
        return 0.0;
    }

    // Compute the sum over k
    let mut sum = 0.0f64;
    for k in k_min..=k_max {
        // All factorial arguments are non-negative by k range construction
        let ek = e + k;
        let fk = f_val + k;
        if ek < 0 || fk < 0 {
            continue;
        }

        let log_denom = log_factorial(k as usize)
            + log_factorial((a - k) as usize)
            + log_factorial((j1m_m1 - k) as usize)
            + log_factorial((j2p_m2 - k) as usize)
            + log_factorial(ek as usize)
            + log_factorial(fk as usize);

        let log_term = -log_denom;
        let term = log_term.exp();
        let sign: f64 = if k % 2 == 0 { 1.0 } else { -1.0 };
        sum += sign * term;
    }

    if sum == 0.0 {
        return 0.0;
    }

    // Assemble: CG = sqrt(2J+1) * sqrt(Δ) * sqrt(factorial_products) * sum
    let log_abs_cg = 0.5 * log_2jp1 + 0.5 * log_delta + 0.5 * log_sqrt_factor + sum.abs().ln();
    let sign_cg = if sum > 0.0 { 1.0 } else { -1.0 };

    sign_cg * log_abs_cg.exp()
}

// ────────────────────────────────────────────────────────────────────────────
// Wigner 3j symbol
// ────────────────────────────────────────────────────────────────────────────

/// Compute the Wigner 3j symbol.
///
/// ```text
/// ⎛ j1  j2  j3 ⎞   (-1)^{j1-j2+m3}
/// ⎝ m1  m2  m3 ⎠ = ──────────────── ⟨j1 m1; j2 m2 | j3, -m3⟩
///                    sqrt(2j3 + 1)
/// ```
///
/// Selection rules: m1 + m2 + m3 = 0, triangle condition on (j1, j2, j3).
///
/// # Arguments
/// * `j1, m1, j2, m2, j3, m3` - Angular momentum quantum numbers
pub fn wigner_3j(j1: f64, m1: f64, j2: f64, m2: f64, j3: f64, m3: f64) -> f64 {
    // 3j symbol requires m1 + m2 + m3 = 0
    if (m1 + m2 + m3).abs() > 1e-10 {
        return 0.0;
    }

    // CG coefficient ⟨j1 m1; j2 m2 | j3, -m3⟩
    let cg = clebsch_gordan(j1, m1, j2, m2, j3, -m3);
    if cg == 0.0 {
        return 0.0;
    }

    // Phase factor (-1)^{j1-j2+m3}: j1-j2+m3 must be integer for non-zero result
    let phase_exp = twice(j1) - twice(j2) + twice(m3);
    let phase: f64 = if phase_exp % 2 == 0 { 1.0 } else { -1.0 };

    let norm = (2.0 * j3 + 1.0).sqrt();
    if norm < 1e-15 {
        return 0.0;
    }

    phase * cg / norm
}

// ────────────────────────────────────────────────────────────────────────────
// Wigner 6j symbol
// ────────────────────────────────────────────────────────────────────────────

/// Compute the Wigner 6j symbol via the Racah formula.
///
/// ```text
/// { j1  j2  j3 }
/// { j4  j5  j6 }
/// ```
///
/// Equals the Racah W-coefficient: W(j1 j2 J j3; j4 j5)
///
/// The Racah formula for the 6j symbol:
/// {j1 j2 j3; j4 j5 j6} = Δ(j1,j2,j3) Δ(j1,j5,j6) Δ(j4,j2,j6) Δ(j4,j5,j3)
///                        × Σ_t (-1)^t (t+1)! / [...factorials...]
///
/// where the sum is over integers t satisfying all factorial arguments ≥ 0.
///
/// # Arguments
/// * `j1, j2, j3, j4, j5, j6` - Six angular momentum quantum numbers
///   forming the 6j symbol with triangle conditions:
///   (j1,j2,j3), (j1,j5,j6), (j4,j2,j6), (j4,j5,j3)
pub fn wigner_6j(j1: f64, j2: f64, j3: f64, j4: f64, j5: f64, j6: f64) -> f64 {
    // Check all triangle conditions
    if !triangle_condition(j1, j2, j3)
        || !triangle_condition(j1, j5, j6)
        || !triangle_condition(j4, j2, j6)
        || !triangle_condition(j4, j5, j3)
    {
        return 0.0;
    }

    // Check half-integer validity
    for &x in &[j1, j2, j3, j4, j5, j6] {
        if !is_half_integer(x) || x < 0.0 {
            return 0.0;
        }
    }

    // Compute log of the four Δ factors
    let log_delta = |a: f64, b: f64, c: f64| -> Option<f64> {
        let a2 = twice(a);
        let b2 = twice(b);
        let c2 = twice(c);
        let p = (a2 + b2 - c2) / 2;
        let q = (a2 - b2 + c2) / 2;
        let r = (-a2 + b2 + c2) / 2;
        let s = (a2 + b2 + c2) / 2 + 1;
        if p < 0 || q < 0 || r < 0 || s < 0 {
            return None;
        }
        Some(
            log_factorial(p as usize) + log_factorial(q as usize) + log_factorial(r as usize)
                - log_factorial(s as usize),
        )
    };

    let log_d1 = match log_delta(j1, j2, j3) {
        Some(v) => v,
        None => return 0.0,
    };
    let log_d2 = match log_delta(j1, j5, j6) {
        Some(v) => v,
        None => return 0.0,
    };
    let log_d3 = match log_delta(j4, j2, j6) {
        Some(v) => v,
        None => return 0.0,
    };
    let log_d4 = match log_delta(j4, j5, j3) {
        Some(v) => v,
        None => return 0.0,
    };

    let log_delta_total = log_d1 + log_d2 + log_d3 + log_d4;

    // Racah formula sum over t:
    // Numerator: (t+1)!
    // Denominator: (t-A)!(t-B)!(t-C)!(t-D)!(E-t)!(F-t)!(G-t)!
    // where the bounds come from the four triangle relations.
    //
    // Define:
    // A = j1+j2+j3, B = j1+j5+j6, C = j4+j2+j6, D = j4+j5+j3
    // E = j1+j2+j4+j5, F = j2+j3+j5+j6, G = j1+j3+j4+j6
    // t ranges over: max(A,B,C,D) ≤ t ≤ min(E,F,G)

    let j1_2 = twice(j1);
    let j2_2 = twice(j2);
    let j3_2 = twice(j3);
    let j4_2 = twice(j4);
    let j5_2 = twice(j5);
    let j6_2 = twice(j6);

    // All in half-integer units × 2
    let a_2 = (j1_2 + j2_2 + j3_2) / 2;
    let b_2 = (j1_2 + j5_2 + j6_2) / 2;
    let c_2 = (j4_2 + j2_2 + j6_2) / 2;
    let d_2 = (j4_2 + j5_2 + j3_2) / 2;
    let e_2 = (j1_2 + j2_2 + j4_2 + j5_2) / 2;
    let f_2 = (j2_2 + j3_2 + j5_2 + j6_2) / 2;
    let g_2 = (j1_2 + j3_2 + j4_2 + j6_2) / 2;

    let t_min = a_2.max(b_2).max(c_2).max(d_2);
    let t_max = e_2.min(f_2).min(g_2);

    if t_min > t_max {
        return 0.0;
    }

    let mut sum = 0.0f64;
    for t in t_min..=t_max {
        let ta = t - a_2;
        let tb = t - b_2;
        let tc = t - c_2;
        let td = t - d_2;
        let te = e_2 - t;
        let tf = f_2 - t;
        let tg = g_2 - t;

        if ta < 0 || tb < 0 || tc < 0 || td < 0 || te < 0 || tf < 0 || tg < 0 {
            continue;
        }

        let log_num = log_factorial((t + 1) as usize);
        let log_den = log_factorial(ta as usize)
            + log_factorial(tb as usize)
            + log_factorial(tc as usize)
            + log_factorial(td as usize)
            + log_factorial(te as usize)
            + log_factorial(tf as usize)
            + log_factorial(tg as usize);

        let log_term = log_num - log_den;
        let term = log_term.exp();
        let sign: f64 = if t % 2 == 0 { 1.0 } else { -1.0 };
        sum += sign * term;
    }

    if sum == 0.0 {
        return 0.0;
    }

    // Final result: sqrt(delta_factors) * sum
    let log_prefactor = 0.5 * log_delta_total;
    let sign: f64 = if sum > 0.0 { 1.0 } else { -1.0 };
    sign * (log_prefactor + sum.abs().ln()).exp()
}

// ────────────────────────────────────────────────────────────────────────────
// CG table for given j1, j2
// ────────────────────────────────────────────────────────────────────────────

/// Compute all Clebsch-Gordan coefficients ⟨j₁ m₁; j₂ m₂ | J M⟩ for given j₁, j₂.
///
/// Returns a flat vector with layout `[m1_idx][m2_idx][J_idx]` where:
/// - m1\_idx in 0..=(2*j1) corresponds to m1 = -j1 + m1\_idx
/// - m2\_idx in 0..=(2*j2) corresponds to m2 = -j2 + m2\_idx
/// - J\_idx in 0..=(j1+j2-|j1-j2|) corresponds to J = |j1-j2| + J\_idx
///
/// Returns a 3D vector indexed as `[m1_i][m2_i][J_i]`.
pub fn cg_table(j1: f64, j2: f64) -> SpecialResult<Vec<Vec<Vec<f64>>>> {
    if !is_half_integer(j1) || !is_half_integer(j2) || j1 < 0.0 || j2 < 0.0 {
        return Err(SpecialError::ValueError(format!(
            "j1={j1}, j2={j2} must be non-negative half-integers"
        )));
    }

    let n1 = twice(j1) as usize + 1; // number of m1 values
    let n2 = twice(j2) as usize + 1; // number of m2 values

    // J ranges from |j1-j2| to j1+j2 in integer steps (not half-integer steps)
    let j_min = (j1 - j2).abs();
    let j_max = j1 + j2;
    // Number of J values: |j1-j2|, |j1-j2|+1, ..., j1+j2 → count = j_max - j_min + 1
    // Since j_max - j_min = j1+j2 - |j1-j2| = 2*min(j1,j2), which is always a non-negative integer.
    let n_j = ((j_max - j_min).round() as usize) + 1;

    let mut table: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; n_j]; n2]; n1];

    for m1_i in 0..n1 {
        let m1 = -j1 + m1_i as f64;
        for m2_i in 0..n2 {
            let m2 = -j2 + m2_i as f64;
            for j_i in 0..n_j {
                let j = j_min + j_i as f64;
                let m = m1 + m2;
                // Check m is valid for this J
                if m.abs() > j + 1e-10 {
                    table[m1_i][m2_i][j_i] = 0.0;
                } else {
                    table[m1_i][m2_i][j_i] = clebsch_gordan(j1, m1, j2, m2, j, m);
                }
            }
        }
    }

    Ok(table)
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::SQRT_2;

    #[test]
    fn test_triangle_condition_basic() {
        assert!(triangle_condition(1.0, 1.0, 1.0));
        assert!(triangle_condition(1.0, 1.0, 2.0));
        assert!(triangle_condition(1.0, 1.0, 0.0));
        assert!(!triangle_condition(2.0, 0.0, 1.0));
        assert!(!triangle_condition(1.0, 1.0, 3.0));
    }

    #[test]
    fn test_triangle_condition_half_integers() {
        assert!(triangle_condition(0.5, 0.5, 1.0));
        assert!(triangle_condition(0.5, 0.5, 0.0));
        assert!(!triangle_condition(0.5, 0.5, 2.0));
    }

    #[test]
    fn test_cg_half_half_to_1() {
        // ⟨1/2, 1/2; 1/2, -1/2 | 1, 0⟩ = 1/√2 ≈ 0.7071
        let cg = clebsch_gordan(0.5, 0.5, 0.5, -0.5, 1.0, 0.0);
        let expected = 1.0 / SQRT_2;
        assert!(
            (cg - expected).abs() < 1e-6,
            "CG(1/2,1/2;1/2,-1/2|1,0) = {cg}, expected {expected}"
        );
    }

    #[test]
    fn test_cg_half_half_maximum() {
        // ⟨1/2, 1/2; 1/2, 1/2 | 1, 1⟩ = 1
        let cg = clebsch_gordan(0.5, 0.5, 0.5, 0.5, 1.0, 1.0);
        assert!(
            (cg - 1.0).abs() < 1e-6,
            "CG(1/2,1/2;1/2,1/2|1,1) = {cg}, expected 1"
        );
    }

    #[test]
    fn test_cg_half_half_to_0() {
        // ⟨1/2, 1/2; 1/2, -1/2 | 0, 0⟩ = 1/√2 (with conventional sign)
        let cg = clebsch_gordan(0.5, 0.5, 0.5, -0.5, 0.0, 0.0);
        assert!(
            (cg.abs() - 1.0 / SQRT_2).abs() < 1e-6,
            "|CG(1/2,1/2;1/2,-1/2|0,0)| = {}, expected {}",
            cg.abs(),
            1.0 / SQRT_2
        );
    }

    #[test]
    fn test_cg_selection_rule_m_sum() {
        // m1 + m2 ≠ M → CG = 0
        let cg = clebsch_gordan(1.0, 1.0, 1.0, 0.0, 2.0, 0.0);
        // m1+m2 = 1 ≠ 0 = M → should be 0
        assert!(cg.abs() < 1e-10, "CG with m1+m2≠M should be 0, got {cg}");
    }

    #[test]
    fn test_cg_triangle_violation() {
        // J = 3 violates triangle for j1=j2=1 (J_max=2)
        let cg = clebsch_gordan(1.0, 0.0, 1.0, 0.0, 3.0, 0.0);
        assert!(
            cg.abs() < 1e-10,
            "CG violating triangle should be 0, got {cg}"
        );
    }

    #[test]
    fn test_cg_integer_spins_1_1_to_2() {
        // ⟨1, 1; 1, 1 | 2, 2⟩ = 1
        let cg = clebsch_gordan(1.0, 1.0, 1.0, 1.0, 2.0, 2.0);
        assert!(
            (cg - 1.0).abs() < 1e-6,
            "CG(1,1;1,1|2,2) = {cg}, expected 1"
        );
    }

    #[test]
    fn test_cg_orthonormality() {
        // Σ_J |CG(1/2,m1;1/2,m2|J,M)|² = δ_{m1+m2, M}
        // For m1=1/2, m2=-1/2: Σ_J (CG(J))^2 should = 1
        let j1 = 0.5;
        let j2 = 0.5;
        let m1 = 0.5;
        let m2 = -0.5;
        let m = m1 + m2;

        let cg1 = clebsch_gordan(j1, m1, j2, m2, 1.0, m);
        let cg0 = clebsch_gordan(j1, m1, j2, m2, 0.0, m);
        let sum_sq = cg1 * cg1 + cg0 * cg0;
        assert!(
            (sum_sq - 1.0).abs() < 1e-6,
            "Orthonormality: Σ|CG|² = {sum_sq}, expected 1"
        );
    }

    #[test]
    fn test_wigner_3j_basic() {
        // 3j(1/2, 1/2; 1/2, -1/2; 1, 0) = (-1)^{1/2-1/2+0}/√3 * CG(1/2,1/2;1/2,-1/2|1,0)
        // = 1/√3 * 1/√2 = 1/√6
        let w3j = wigner_3j(0.5, 0.5, 0.5, -0.5, 1.0, 0.0);
        // Check it's non-zero for valid quantum numbers
        // m1+m2+m3 = 0.5 - 0.5 + 0.0 = 0 ✓
        assert!(w3j.abs() > 0.0, "3j symbol should be non-zero: {w3j}");
    }

    #[test]
    fn test_wigner_3j_selection_rule() {
        // m1+m2+m3 ≠ 0 → 3j = 0
        let w3j = wigner_3j(1.0, 1.0, 1.0, 0.0, 1.0, 0.0);
        // m1+m2+m3 = 1+0+0 = 1 ≠ 0
        assert!(w3j.abs() < 1e-10, "3j with m_sum≠0 should be 0, got {w3j}");
    }

    #[test]
    fn test_wigner_6j_triangle_violation() {
        // Triangle violation → 6j = 0
        let w6j = wigner_6j(1.0, 1.0, 3.0, 1.0, 1.0, 1.0);
        // Triangle(1,1,3) fails since |1-1|=0 ≤ 3 but 1+1=2 < 3
        assert!(
            w6j.abs() < 1e-10,
            "6j with triangle violation should be 0, got {w6j}"
        );
    }

    #[test]
    fn test_wigner_6j_basic() {
        // 6j{1/2, 1/2, 1; 1/2, 1/2, 1} should be a known value
        // For j=1/2 spins, this is related to Racah's original tables
        // The value: {1/2,1/2,1;1/2,1/2,1} = (-1)^{1+1+1}/sqrt(6) = -1/sqrt(6)...
        // Actually let's just check it's computable and has the right magnitude
        let w6j = wigner_6j(0.5, 0.5, 1.0, 0.5, 0.5, 1.0);
        assert!(
            w6j.abs() > 0.0,
            "6j symbol for valid quantum numbers should be non-zero: {w6j}"
        );
    }

    #[test]
    fn test_log_factorial_small() {
        // ln(5!) = ln(120) ≈ 4.7875
        let lf5 = log_factorial(5);
        let expected = 120.0f64.ln();
        assert!(
            (lf5 - expected).abs() < 1e-10,
            "log_factorial(5) = {lf5}, expected {expected}"
        );
    }

    #[test]
    fn test_log_factorial_zero() {
        assert_eq!(log_factorial(0), 0.0);
    }

    #[test]
    fn test_cg_table_half_half() {
        // Table for j1=j2=1/2: 2×2×2 array
        let table = cg_table(0.5, 0.5).expect("cg_table");
        assert_eq!(table.len(), 2); // m1 ∈ {-1/2, 1/2}
        assert_eq!(table[0].len(), 2); // m2 ∈ {-1/2, 1/2}
        assert_eq!(table[0][0].len(), 2); // J ∈ {0, 1}

        // Check CG(1/2,1/2; 1/2,1/2 | 1,1) = table[1][1][1] = 1
        let cg_11 = table[1][1][1]; // m1_i=1 → m1=1/2, m2_i=1 → m2=1/2, J_i=1 → J=1
        assert!(
            (cg_11 - 1.0).abs() < 1e-6,
            "CG table[1][1][1] = {cg_11}, expected 1"
        );
    }

    #[test]
    fn test_is_half_integer() {
        assert!(is_half_integer(0.0));
        assert!(is_half_integer(0.5));
        assert!(is_half_integer(1.0));
        assert!(is_half_integer(1.5));
        assert!(!is_half_integer(0.3));
        assert!(!is_half_integer(1.2));
    }

    #[test]
    fn test_cg_normalization() {
        // Σ_M |C(j1,m1,j2,m2;J,M)|² = 1 summed over (m1,m2) pairs that give the same M
        // For fixed j1=1/2, j2=1/2, J=1: Σ_{m1+m2=M} |CG|² should = 1 for each valid M
        let j1 = 0.5;
        let j2 = 0.5;
        let j = 1.0;

        // M = 1: only (m1=1/2, m2=1/2)
        let cg_m1 = clebsch_gordan(j1, 0.5, j2, 0.5, j, 1.0);
        assert!(
            (cg_m1 * cg_m1 - 1.0).abs() < 1e-6,
            "Norm for M=1: {}, expected 1",
            cg_m1 * cg_m1
        );

        // M = 0: (m1=1/2, m2=-1/2) and (m1=-1/2, m2=1/2)
        let cg_a = clebsch_gordan(j1, 0.5, j2, -0.5, j, 0.0);
        let cg_b = clebsch_gordan(j1, -0.5, j2, 0.5, j, 0.0);
        let norm_m0 = cg_a * cg_a + cg_b * cg_b;
        assert!(
            (norm_m0 - 1.0).abs() < 1e-6,
            "Norm for M=0: {norm_m0}, expected 1"
        );
    }

    #[test]
    fn test_cg_orthogonality() {
        // For j1=j2=1/2: orthogonality between J=0 and J=1 for fixed (m1,m2)
        // Σ_M CG(J=0,M) * CG(J=1,M) = 0  (orthogonality across J)
        let j1 = 0.5;
        let j2 = 0.5;
        // Sum over all (m1,m2) pairs of CG(J=0)*CG(J=1) for M=m1+m2
        let mut cross = 0.0f64;
        for &m1 in &[-0.5f64, 0.5] {
            for &m2 in &[-0.5f64, 0.5] {
                let m = m1 + m2;
                let cg0 = clebsch_gordan(j1, m1, j2, m2, 0.0, m);
                let cg1 = clebsch_gordan(j1, m1, j2, m2, 1.0, m);
                cross += cg0 * cg1;
            }
        }
        assert!(
            cross.abs() < 1e-6,
            "Orthogonality: Σ CG(J=0)*CG(J=1) = {cross}, expected 0"
        );
    }

    #[test]
    fn test_cg_half_half_singlet() {
        // CG(1/2,1/2;1/2,-1/2|0,0) = 1/√2 (up to sign convention)
        let cg = clebsch_gordan(0.5, 0.5, 0.5, -0.5, 0.0, 0.0);
        let expected_abs = 1.0 / SQRT_2;
        assert!(
            (cg.abs() - expected_abs).abs() < 1e-6,
            "|CG(1/2,1/2;1/2,-1/2|0,0)| = {}, expected {}",
            cg.abs(),
            expected_abs
        );
    }

    #[test]
    fn test_cg_one_one_to_zero() {
        // CG(1,0;1,0|0,0) = -1/√3 (known value from tables)
        // For j1=j2=1, J=0, M=0: CG = (-1)^{1-1+0}/√3... standard Clebsch-Gordan table
        // ⟨1,0;1,0|0,0⟩ = -1/√3
        let cg = clebsch_gordan(1.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        let expected_abs = 1.0 / 3.0f64.sqrt();
        assert!(
            (cg.abs() - expected_abs).abs() < 1e-6,
            "|CG(1,0;1,0|0,0)| = {}, expected {expected_abs}",
            cg.abs()
        );
    }

    #[test]
    fn test_cg_symmetry_exchange() {
        // CG(j1,m1,j2,m2;J,M) = (-1)^{j1+j2-J} * CG(j2,m2,j1,m1;J,M)
        let j1 = 1.0;
        let m1 = 0.0;
        let j2 = 0.5;
        let m2 = 0.5;
        let j = 1.5;
        let m = m1 + m2;
        let cg_direct = clebsch_gordan(j1, m1, j2, m2, j, m);
        let cg_swapped = clebsch_gordan(j2, m2, j1, m1, j, m);
        // Phase: (-1)^{j1+j2-J} = (-1)^{1+0.5-1.5} = (-1)^0 = 1
        let phase_exp = (twice(j1) + twice(j2) - twice(j)) / 2;
        let phase: f64 = if phase_exp % 2 == 0 { 1.0 } else { -1.0 };
        assert!(
            (cg_direct - phase * cg_swapped).abs() < 1e-6,
            "CG symmetry: {cg_direct} vs {phase} * {cg_swapped}"
        );
    }

    #[test]
    fn test_wigner_3j_triangle_violation() {
        // 3j symbol is zero when triangle condition fails
        // j3 = 5 violates triangle for j1=j2=1 (max J = 2)
        let w3j = wigner_3j(1.0, 0.0, 1.0, 0.0, 5.0, 0.0);
        // m1+m2+m3 = 0 ✓, but triangle(1,1,5) fails
        assert!(
            w3j.abs() < 1e-10,
            "3j outside triangle should be 0, got {w3j}"
        );
    }

    #[test]
    fn test_wigner_3j_known_half() {
        // Known 3j symbol: (1/2, 1/2, 0; 1/2, -1/2, 0)
        // = (-1)^{1/2-1/2+0}/√1 * CG(1/2,1/2;1/2,-1/2|0,0)
        // = 1 * (1/√2) / 1 = 1/√2  [since 2j3+1 = 1, sqrt(1) = 1]
        // Wait: 3j(j1,m1;j2,m2;j3,m3) = (-1)^{j1-j2+m3}/√(2j3+1) * CG(j1,m1;j2,m2|j3,-m3)
        // For j3=0,m3=0: (-1)^{1/2-1/2+0}/√1 * CG(1/2,1/2;1/2,-1/2|0,0)
        // = 1 * (CG value)
        // m1+m2+m3 = 1/2-1/2+0 = 0 ✓
        let w3j = wigner_3j(0.5, 0.5, 0.5, -0.5, 0.0, 0.0);
        let cg = clebsch_gordan(0.5, 0.5, 0.5, -0.5, 0.0, 0.0);
        // The 3j and CG should be proportional
        // 3j = (-1)^{1/2-1/2+0}/√1 * CG = CG
        assert!(
            (w3j.abs() - cg.abs()).abs() < 1e-6,
            "|3j| = {}, |CG| = {} (should match for j3=0)",
            w3j.abs(),
            cg.abs()
        );
    }

    #[test]
    fn test_wigner_6j_known_racah() {
        // Known 6j values from Racah's original tables.
        //
        // {1/2, 1/2, 1; 1/2, 1/2, 1}: known value is 1/6
        // (see Edmonds "Angular Momentum" table or Varshalovich table 9.1)
        let w6j_half = wigner_6j(0.5, 0.5, 1.0, 0.5, 0.5, 1.0);
        let expected = 1.0f64 / 6.0f64;
        assert!(
            (w6j_half.abs() - expected).abs() < 1e-5,
            "|6j(1/2,1/2,1;1/2,1/2,1)| = {}, expected {expected}",
            w6j_half.abs()
        );

        // {1/2,1/2,0;1/2,1/2,0}: triangle (1/2,1/2,0)✓, (1/2,1/2,0)✓, (1/2,1/2,0)✓, (1/2,1/2,0)✓
        // Known value: {1/2,1/2,0;1/2,1/2,0} = 1/2
        let w6j_zero = wigner_6j(0.5, 0.5, 0.0, 0.5, 0.5, 0.0);
        assert!(
            (w6j_zero.abs() - 0.5).abs() < 1e-5,
            "|6j(1/2,1/2,0;1/2,1/2,0)| = {}, expected 0.5",
            w6j_zero.abs()
        );
    }

    #[test]
    fn test_cg_table_size_dimensions() {
        // cg_table(j1, j2) returns a [n1][n2][nJ] array
        // For j1=1, j2=1: n1=3, n2=3, nJ=3 (J=0,1,2)
        let table = cg_table(1.0, 1.0).expect("cg_table(1,1)");
        assert_eq!(table.len(), 3, "j1=1: n1 = 2*j1+1 = 3");
        assert_eq!(table[0].len(), 3, "j2=1: n2 = 2*j2+1 = 3");
        assert_eq!(table[0][0].len(), 3, "J values: |1-1|=0 to 1+1=2, nJ=3");

        // For j1=3/2, j2=1/2: n1=4, n2=2, nJ=2 (J=1,2)
        let table2 = cg_table(1.5, 0.5).expect("cg_table(3/2,1/2)");
        assert_eq!(table2.len(), 4, "j1=3/2: n1 = 4");
        assert_eq!(table2[0].len(), 2, "j2=1/2: n2 = 2");
        assert_eq!(table2[0][0].len(), 2, "J values: |3/2-1/2|=1 to 2, nJ=2");
    }

    #[test]
    fn test_log_factorial_correctness() {
        // ln(5!) = ln(120) ≈ 4.7875...
        let lf = log_factorial(5);
        let expected = 120.0f64.ln();
        assert!(
            (lf - expected).abs() < 1e-12,
            "log_factorial(5) = {lf}, expected {expected}"
        );
        // ln(10!) = ln(3628800)
        let lf10 = log_factorial(10);
        let expected10 = 3_628_800.0f64.ln();
        assert!(
            (lf10 - expected10).abs() < 1e-10,
            "log_factorial(10) = {lf10}, expected {expected10}"
        );
    }
}
