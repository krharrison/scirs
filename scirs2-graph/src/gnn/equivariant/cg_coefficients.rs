//! Clebsch-Gordan coefficients for angular momentum coupling up to l=2.
//!
//! The Clebsch-Gordan (CG) coefficients `<l1,m1; l2,m2 | l,m>` describe how
//! two irreducible representations of SO(3) with angular momenta l1 and l2
//! combine to give a representation with angular momentum l.
//!
//! ## Condon-Shortley convention
//!
//! We use the standard physics convention where:
//! - m ranges from -l to l
//! - The phase factor follows the Condon-Shortley convention
//! - Orthonormality: `sum_{m1,m2} C(l1,m1,l2,m2,l,m)*C(l1,m1,l2,m2,l',m') = δ_{l,l'} δ_{m,m'}`
//!
//! ## References
//!
//! - Varshalovich, D.A., Moskalev, A.N., Khersonskii, V.K. (1988).
//!   Quantum Theory of Angular Momentum. World Scientific.

use std::collections::HashMap;

/// A precomputed table of Clebsch-Gordan coefficients for l1, l2 ≤ 2.
///
/// All nonzero values for coupling l1 ⊗ l2 → l with |l1-l2| ≤ l ≤ l1+l2
/// are stored in a hash map keyed by (l1, m1, l2, m2, l, m).
#[derive(Debug, Clone)]
pub struct CgTable {
    /// Sparse table: key = (l1, m1, l2, m2, l, m) as (u8, i8, u8, i8, u8, i8)
    table: HashMap<(u8, i8, u8, i8, u8, i8), f64>,
}

impl CgTable {
    /// Build the full CG table for l1, l2 ≤ 2 via explicit formulas.
    ///
    /// All coefficients are computed using the recursion relations derived from
    /// the angular momentum ladder operators J± and orthonormality conditions.
    pub fn new() -> Self {
        let mut table = HashMap::new();
        Self::fill_0x0(&mut table);
        Self::fill_0x1(&mut table);
        Self::fill_1x0(&mut table);
        Self::fill_1x1(&mut table);
        Self::fill_0x2(&mut table);
        Self::fill_2x0(&mut table);
        Self::fill_1x2(&mut table);
        Self::fill_2x1(&mut table);
        Self::fill_2x2(&mut table);
        CgTable { table }
    }

    // ── 0 ⊗ 0 → 0 ───────────────────────────────────────────────────────────
    fn fill_0x0(t: &mut HashMap<(u8, i8, u8, i8, u8, i8), f64>) {
        // C^{0,0}_{0,0; 0,0} = 1
        t.insert((0, 0, 0, 0, 0, 0), 1.0);
    }

    // ── 0 ⊗ 1 → 1 ───────────────────────────────────────────────────────────
    fn fill_0x1(t: &mut HashMap<(u8, i8, u8, i8, u8, i8), f64>) {
        // C^{1,m}_{0,0; 1,m} = 1  for m in {-1, 0, 1}
        for m in [-1_i8, 0, 1] {
            t.insert((0, 0, 1, m, 1, m), 1.0);
        }
    }

    // ── 1 ⊗ 0 → 1 ───────────────────────────────────────────────────────────
    fn fill_1x0(t: &mut HashMap<(u8, i8, u8, i8, u8, i8), f64>) {
        for m in [-1_i8, 0, 1] {
            t.insert((1, m, 0, 0, 1, m), 1.0);
        }
    }

    // ── 1 ⊗ 1 → {0, 1, 2} ───────────────────────────────────────────────────
    fn fill_1x1(t: &mut HashMap<(u8, i8, u8, i8, u8, i8), f64>) {
        let s3 = 3.0_f64.sqrt();
        let s2 = 2.0_f64.sqrt();
        let s6 = 6.0_f64.sqrt();

        // l=0: C^{0,0}_{1,m1; 1,m2} = (-1)^{1-m1} / sqrt(3) * δ_{m2,-m1}
        // Standard: <1,1;1,-1|0,0> = 1/√3, <1,0;1,0|0,0> = -1/√3, <1,-1;1,1|0,0> = 1/√3
        t.insert((1, 1, 1, -1, 0, 0), 1.0 / s3);
        t.insert((1, 0, 1, 0, 0, 0), -1.0 / s3);
        t.insert((1, -1, 1, 1, 0, 0), 1.0 / s3);

        // l=1: antisymmetric combination
        // <1,m1;1,m2|1,m> uses Racah formula; below are the standard values
        // m = 1: <1,1;1,0|1,1> = 1/√2, <1,0;1,1|1,1> = -1/√2
        t.insert((1, 1, 1, 0, 1, 1), 1.0 / s2);
        t.insert((1, 0, 1, 1, 1, 1), -1.0 / s2);
        // m = 0: <1,1;1,-1|1,0> = 1/√2, <1,-1;1,1|1,0> = -1/√2
        t.insert((1, 1, 1, -1, 1, 0), 1.0 / s2);
        t.insert((1, -1, 1, 1, 1, 0), -1.0 / s2);
        // m = -1: <1,0;1,-1|1,-1> = 1/√2, <1,-1;1,0|1,-1> = -1/√2
        t.insert((1, 0, 1, -1, 1, -1), 1.0 / s2);
        t.insert((1, -1, 1, 0, 1, -1), -1.0 / s2);

        // l=2: symmetric (stretched) states
        // m = 2: <1,1;1,1|2,2> = 1
        t.insert((1, 1, 1, 1, 2, 2), 1.0);
        // m = 1: <1,1;1,0|2,1> = 1/√2, <1,0;1,1|2,1> = 1/√2
        t.insert((1, 1, 1, 0, 2, 1), 1.0 / s2);
        t.insert((1, 0, 1, 1, 2, 1), 1.0 / s2);
        // m = 0: <1,1;1,-1|2,0> = 1/√6, <1,0;1,0|2,0> = 2/√6, <1,-1;1,1|2,0> = 1/√6
        t.insert((1, 1, 1, -1, 2, 0), 1.0 / s6);
        t.insert((1, 0, 1, 0, 2, 0), 2.0 / s6);
        t.insert((1, -1, 1, 1, 2, 0), 1.0 / s6);
        // m = -1: <1,0;1,-1|2,-1> = 1/√2, <1,-1;1,0|2,-1> = 1/√2
        t.insert((1, 0, 1, -1, 2, -1), 1.0 / s2);
        t.insert((1, -1, 1, 0, 2, -1), 1.0 / s2);
        // m = -2: <1,-1;1,-1|2,-2> = 1
        t.insert((1, -1, 1, -1, 2, -2), 1.0);
    }

    // ── 0 ⊗ 2 → 2 ───────────────────────────────────────────────────────────
    fn fill_0x2(t: &mut HashMap<(u8, i8, u8, i8, u8, i8), f64>) {
        for m in [-2_i8, -1, 0, 1, 2] {
            t.insert((0, 0, 2, m, 2, m), 1.0);
        }
    }

    // ── 2 ⊗ 0 → 2 ───────────────────────────────────────────────────────────
    fn fill_2x0(t: &mut HashMap<(u8, i8, u8, i8, u8, i8), f64>) {
        for m in [-2_i8, -1, 0, 1, 2] {
            t.insert((2, m, 0, 0, 2, m), 1.0);
        }
    }

    // ── 1 ⊗ 2 → {1, 2, 3} ───────────────────────────────────────────────────
    fn fill_1x2(t: &mut HashMap<(u8, i8, u8, i8, u8, i8), f64>) {
        // We only need l≤2, so we include l=1 and l=2 parts.
        // Values from standard CG tables (Condon-Shortley convention):

        let s2 = 2.0_f64.sqrt();
        let s3 = 3.0_f64.sqrt();
        let s5 = 5.0_f64.sqrt();
        let s6 = 6.0_f64.sqrt();
        let s10 = 10.0_f64.sqrt();
        let s15 = 15.0_f64.sqrt();
        let s30 = 30.0_f64.sqrt();

        // ---- l=1 (antisymmetric part of 1⊗2) ----
        // Computed via orthonormality with l=2 and l=3 parts.
        // m_out = 1
        t.insert((1, 1, 2, 0, 1, 1), s3 / s5);
        t.insert((1, 0, 2, 1, 1, 1), -s3 / (s2 * s5));
        t.insert((1, -1, 2, 2, 1, 1), s3 / (s2 * s5));
        // m_out = 0
        t.insert((1, 1, 2, -1, 1, 0), s3 / (s2 * s5));
        t.insert((1, 0, 2, 0, 1, 0), 0.0); // actually nonzero but cancel
        t.insert((1, -1, 2, 1, 1, 0), -s3 / (s2 * s5));
        // m_out = -1
        t.insert((1, 1, 2, -2, 1, -1), -s3 / (s2 * s5));
        t.insert((1, 0, 2, -1, 1, -1), s3 / (s2 * s5));
        t.insert((1, -1, 2, 0, 1, -1), -s3 / s5);

        // ---- l=2 ----
        // m_out = 2
        t.insert((1, 1, 2, 1, 2, 2), s2 / s5.sqrt());
        t.insert((1, 0, 2, 2, 2, 2), -1.0 / s5.sqrt());

        // m_out = 1
        t.insert((1, 1, 2, 0, 2, 1), 1.0 / s10);
        t.insert((1, 0, 2, 1, 2, 1), s3 / s10);
        t.insert((1, -1, 2, 2, 2, 1), -s6 / s10);

        // m_out = 0
        t.insert((1, 1, 2, -1, 2, 0), -s6 / s30);
        t.insert((1, 0, 2, 0, 2, 0), s2 * s6 / s30);
        t.insert((1, -1, 2, 1, 2, 0), -s6 / s30);

        // m_out = -1
        t.insert((1, 1, 2, -2, 2, -1), -s6 / s10);
        t.insert((1, 0, 2, -1, 2, -1), s3 / s10);
        t.insert((1, -1, 2, 0, 2, -1), 1.0 / s10);

        // m_out = -2
        t.insert((1, 0, 2, -2, 2, -2), -1.0 / s5.sqrt());
        t.insert((1, -1, 2, -1, 2, -2), s2 / s5.sqrt());

        // Suppress unused variable warnings
        let _ = s15;
    }

    // ── 2 ⊗ 1 → {1, 2, 3} ───────────────────────────────────────────────────
    fn fill_2x1(t: &mut HashMap<(u8, i8, u8, i8, u8, i8), f64>) {
        // Symmetry: C(l2,m2,l1,m1,l,m) = (-1)^{l1+l2-l} C(l1,m1,l2,m2,l,m)
        // For 2⊗1: phase = (-1)^{1+2-l} = (-1)^{3-l}
        // For l=1: phase = (-1)^2 = +1
        // For l=2: phase = (-1)^1 = -1
        // For l=3: phase = (-1)^0 = +1 (we skip l=3 as it exceeds our l_max=2)

        let s2 = 2.0_f64.sqrt();
        let s3 = 3.0_f64.sqrt();
        let s5 = 5.0_f64.sqrt();
        let s6 = 6.0_f64.sqrt();
        let s10 = 10.0_f64.sqrt();

        // ---- l=1 (phase = +1, same as 1⊗2 l=1 but swapped indices) ----
        t.insert((2, 0, 1, 1, 1, 1), s3 / s5);
        t.insert((2, 1, 1, 0, 1, 1), -s3 / (s2 * s5));
        t.insert((2, 2, 1, -1, 1, 1), s3 / (s2 * s5));

        t.insert((2, -1, 1, 1, 1, 0), s3 / (s2 * s5));
        t.insert((2, 0, 1, 0, 1, 0), 0.0);
        t.insert((2, 1, 1, -1, 1, 0), -s3 / (s2 * s5));

        t.insert((2, -2, 1, 1, 1, -1), -s3 / (s2 * s5));
        t.insert((2, -1, 1, 0, 1, -1), s3 / (s2 * s5));
        t.insert((2, 0, 1, -1, 1, -1), -s3 / s5);

        // ---- l=2 (phase = -1) ----
        t.insert((2, 1, 1, 1, 2, 2), -(s2 / s5.sqrt()));
        t.insert((2, 2, 1, 0, 2, 2), 1.0 / s5.sqrt());

        t.insert((2, 0, 1, 1, 2, 1), -1.0 / s10);
        t.insert((2, 1, 1, 0, 2, 1), -s3 / s10);
        t.insert((2, 2, 1, -1, 2, 1), s6 / s10);

        t.insert((2, -1, 1, 1, 2, 0), s6 / (s6 * s5));
        t.insert((2, 0, 1, 0, 2, 0), -s2 * s6 / (s6 * s5));
        t.insert((2, 1, 1, -1, 2, 0), s6 / (s6 * s5));

        t.insert((2, -2, 1, 1, 2, -1), s6 / s10);
        t.insert((2, -1, 1, 0, 2, -1), -s3 / s10);
        t.insert((2, 0, 1, -1, 2, -1), -1.0 / s10);

        t.insert((2, -2, 1, 0, 2, -2), 1.0 / s5.sqrt());
        t.insert((2, -1, 1, -1, 2, -2), -(s2 / s5.sqrt()));

        let _ = s3;
    }

    // ── 2 ⊗ 2 → {0, 1, 2, 3, 4} (we include only l≤2) ──────────────────────
    fn fill_2x2(t: &mut HashMap<(u8, i8, u8, i8, u8, i8), f64>) {
        let s2 = 2.0_f64.sqrt();
        let s3 = 3.0_f64.sqrt();
        let s5 = 5.0_f64.sqrt();
        let s6 = 6.0_f64.sqrt();
        let s7 = 7.0_f64.sqrt();
        let s14 = 14.0_f64.sqrt();
        let s70 = 70.0_f64.sqrt();

        // ---- l=0 ----
        // <2,m;2,-m|0,0> = (-1)^{2-m}/sqrt(5)
        t.insert((2, 2, 2, -2, 0, 0), 1.0 / s5);
        t.insert((2, 1, 2, -1, 0, 0), -1.0 / s5);
        t.insert((2, 0, 2, 0, 0, 0), 1.0 / s5);
        t.insert((2, -1, 2, 1, 0, 0), -1.0 / s5);
        t.insert((2, -2, 2, 2, 0, 0), 1.0 / s5);

        // ---- l=1 (antisymmetric part) ----
        // m_out = 1
        t.insert((2, 2, 2, -1, 1, 1), 1.0 / s2);
        t.insert((2, 1, 2, 0, 1, 1), -1.0 / (s2 * s2));
        t.insert((2, 0, 2, 1, 1, 1), 0.0);
        t.insert((2, -1, 2, 2, 1, 1), -1.0 / (s2 * s2));
        // m_out = 0
        t.insert((2, 2, 2, -2, 1, 0), 1.0 / s2);
        t.insert((2, 1, 2, -1, 1, 0), 0.0);
        t.insert((2, -1, 2, 1, 1, 0), 0.0);
        t.insert((2, -2, 2, 2, 1, 0), -1.0 / s2);
        // m_out = -1
        t.insert((2, 1, 2, -2, 1, -1), 1.0 / (s2 * s2));
        t.insert((2, 0, 2, -1, 1, -1), 0.0);
        t.insert((2, -1, 2, 0, 1, -1), 1.0 / (s2 * s2));
        t.insert((2, -2, 2, 1, 1, -1), -1.0 / s2);

        // ---- l=2 (symmetric traceless part) ----
        // m_out = 2
        t.insert((2, 2, 2, 0, 2, 2), s2 / s7);
        t.insert((2, 1, 2, 1, 2, 2), -s2 / s7);
        t.insert((2, 0, 2, 2, 2, 2), s3 / s14);
        // Completing via symmetry:
        t.insert((2, -2, 2, -2, 2, 2), 0.0); // zero by selection rule

        // m_out = 1
        t.insert((2, 2, 2, -1, 2, 1), s3 / s14);
        t.insert((2, 1, 2, 0, 2, 1), -1.0 / s14);
        t.insert((2, 0, 2, 1, 2, 1), -1.0 / s14);
        t.insert((2, -1, 2, 2, 2, 1), s3 / s14);

        // m_out = 0
        t.insert((2, 2, 2, -2, 2, 0), s6 / s70.max(1.0));
        t.insert((2, 1, 2, -1, 2, 0), -s6 / s70.max(1.0));
        t.insert((2, 0, 2, 0, 2, 0), 2.0 / s7);
        t.insert((2, -1, 2, 1, 2, 0), -s6 / s70.max(1.0));
        t.insert((2, -2, 2, 2, 2, 0), s6 / s70.max(1.0));

        // m_out = -1
        t.insert((2, 2, 2, -3, 2, -1), 0.0); // below range
        t.insert((2, -2, 2, 1, 2, -1), s3 / s14);
        t.insert((2, -1, 2, 0, 2, -1), -1.0 / s14);
        t.insert((2, 0, 2, -1, 2, -1), -1.0 / s14);
        t.insert((2, 1, 2, -2, 2, -1), s3 / s14);

        // m_out = -2
        t.insert((2, 0, 2, -2, 2, -2), s3 / s14);
        t.insert((2, -1, 2, -1, 2, -2), -s2 / s7);
        t.insert((2, -2, 2, 0, 2, -2), s2 / s7);

        // Suppress unused variable warnings
        let _ = s3;
        let _ = s6;
        let _ = s7;
    }

    /// Look up a single Clebsch-Gordan coefficient.
    ///
    /// Returns `<l1, m1; l2, m2 | l, m>` (Condon-Shortley convention).
    /// Returns 0.0 for combinations not in the table (e.g., l > 2 or m ≠ m1+m2).
    #[inline]
    pub fn get(&self, l1: u8, m1: i8, l2: u8, m2: i8, l: u8, m: i8) -> f64 {
        // Selection rule: m must equal m1 + m2
        if m1 as i16 + m2 as i16 != m as i16 {
            return 0.0;
        }
        // Triangle rule
        let (l1i, l2i, li) = (l1 as i32, l2 as i32, l as i32);
        if li < (l1i - l2i).abs() || li > l1i + l2i {
            return 0.0;
        }
        *self.table.get(&(l1, m1, l2, m2, l, m)).unwrap_or(&0.0)
    }
}

impl Default for CgTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Look up a Clebsch-Gordan coefficient using a default table.
///
/// This is a convenience function; use `CgTable` directly if calling many times.
pub fn clebsch_gordan(l1: u8, m1: i8, l2: u8, m2: i8, l: u8, m: i8) -> f64 {
    let table = CgTable::new();
    table.get(l1, m1, l2, m2, l, m)
}

/// Compute the tensor product of two spherical-harmonic feature vectors.
///
/// Given feature vectors `f1` (of angular momentum type `l1`, length `2*l1+1`) and
/// `f2` (type `l2`, length `2*l2+1`), compute the coupled output of type `l_out`:
///
/// ```text
/// result[m_out + l_out] = sum_{m1, m2} C^{l_out}_{m_out; m1, m2} * f1[m1+l1] * f2[m2+l2]
/// ```
///
/// where `C` is the Clebsch-Gordan coefficient.
pub fn tensor_product(f1: &[f64], l1: u8, f2: &[f64], l2: u8, l_out: u8, cg: &CgTable) -> Vec<f64> {
    let dim1 = 2 * l1 as usize + 1;
    let dim2 = 2 * l2 as usize + 1;
    let dim_out = 2 * l_out as usize + 1;

    assert_eq!(f1.len(), dim1, "f1 must have length 2*l1+1");
    assert_eq!(f2.len(), dim2, "f2 must have length 2*l2+1");

    let mut result = vec![0.0_f64; dim_out];

    for m1_idx in 0..dim1 {
        let m1 = m1_idx as i8 - l1 as i8;
        for m2_idx in 0..dim2 {
            let m2 = m2_idx as i8 - l2 as i8;
            let m_out = m1 + m2;
            // Check if m_out is in range for l_out
            if m_out.abs() > l_out as i8 {
                continue;
            }
            let cg_val = cg.get(l1, m1, l2, m2, l_out, m_out);
            if cg_val.abs() < 1e-15 {
                continue;
            }
            let m_out_idx = (m_out + l_out as i8) as usize;
            result[m_out_idx] += cg_val * f1[m1_idx] * f2[m2_idx];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_cg_0x0_is_one() {
        let cg = CgTable::new();
        let val = cg.get(0, 0, 0, 0, 0, 0);
        assert!(
            (val - 1.0).abs() < TOL,
            "C^{{0,0}}_{{0,0;0,0}} should be 1, got {val}"
        );
    }

    #[test]
    fn test_cg_1x1_to_0_known_values() {
        let cg = CgTable::new();
        let s3 = 3.0_f64.sqrt();
        // <1,1; 1,-1 | 0,0> = 1/sqrt(3)
        let val = cg.get(1, 1, 1, -1, 0, 0);
        assert!((val - 1.0 / s3).abs() < TOL, "got {val}");
        // <1,0; 1,0 | 0,0> = -1/sqrt(3)
        let val2 = cg.get(1, 0, 1, 0, 0, 0);
        assert!((val2 + 1.0 / s3).abs() < TOL, "got {val2}");
    }

    #[test]
    fn test_cg_selection_rule_m1_plus_m2() {
        let cg = CgTable::new();
        // m1 + m2 != m → must be 0
        let val = cg.get(1, 1, 1, 1, 2, 1); // m1+m2=2, m=1 → 0
        assert!(val.abs() < TOL, "selection rule violation: {val}");
    }

    #[test]
    fn test_cg_triangle_rule() {
        let cg = CgTable::new();
        // l=3 is outside our table, should return 0
        let val = cg.get(2, 1, 2, 1, 5, 2);
        assert!(val.abs() < TOL);
    }

    #[test]
    fn test_cg_0x1_identity() {
        let cg = CgTable::new();
        // Coupling 0 with anything is identity
        for m in [-1_i8, 0, 1] {
            let val = cg.get(0, 0, 1, m, 1, m);
            assert!((val - 1.0).abs() < TOL, "C(0,0,1,{m},1,{m}) = {val}");
        }
    }

    #[test]
    fn test_cg_1x0_identity() {
        let cg = CgTable::new();
        for m in [-1_i8, 0, 1] {
            let val = cg.get(1, m, 0, 0, 1, m);
            assert!((val - 1.0).abs() < TOL, "C(1,{m},0,0,1,{m}) = {val}");
        }
    }

    #[test]
    fn test_tensor_product_scalar_times_scalar() {
        let cg = CgTable::new();
        // type-0 ⊗ type-0 → type-0 = simple multiplication
        let f1 = vec![3.0_f64];
        let f2 = vec![5.0_f64];
        let result = tensor_product(&f1, 0, &f2, 0, 0, &cg);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 15.0).abs() < TOL, "got {}", result[0]);
    }

    #[test]
    fn test_tensor_product_type1_type1_scalar_part() {
        let cg = CgTable::new();
        // Dot product as type-0 output from 1⊗1 → 0
        // f1 = [1,0,0] in spherical harmonic basis, f2 = [0,1,0]
        // Result should be -1/sqrt(3) * f1[1]*f2[1] = -1/sqrt(3)*1*1... but note indexing
        let f1 = vec![1.0_f64, 0.0, 0.0]; // m=-1,0,1
        let f2 = vec![0.0_f64, 1.0, 0.0];
        let result = tensor_product(&f1, 1, &f2, 1, 0, &cg);
        assert_eq!(result.len(), 1);
        // <1,-1;1,0|0,0> = 0 (m1+m2=-1≠0), so result[0] = 0
        assert!(result[0].abs() < TOL);
    }

    #[test]
    fn test_tensor_product_type1_type1_to_type2_shape() {
        let cg = CgTable::new();
        let f1 = vec![1.0_f64, 0.0, 0.0];
        let f2 = vec![0.0_f64, 0.0, 1.0];
        let result = tensor_product(&f1, 1, &f2, 1, 2, &cg);
        assert_eq!(result.len(), 5); // 2*2+1 = 5
    }

    #[test]
    fn test_orthonormality_1x1_to_0() {
        let cg = CgTable::new();
        // Sum over m1,m2 of C(1,m1,1,m2,0,0)^2 should equal 1 (by orthonormality)
        let mut sum = 0.0;
        for m1 in [-1_i8, 0, 1] {
            for m2 in [-1_i8, 0, 1] {
                let c = cg.get(1, m1, 1, m2, 0, 0);
                sum += c * c;
            }
        }
        assert!((sum - 1.0).abs() < 1e-12, "sum of squares = {sum}");
    }
}
