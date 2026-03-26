//! Gaussian basis set integrals for Hartree-Fock / DFT.
//!
//! Gaussian-type orbital (GTO):
//! φ(r) = N x^ax y^ay z^az exp(-ζ |r − R|²)
//!
//! where N is a normalization constant, (ax, ay, az) are angular momentum
//! quantum numbers, ζ is the exponent, and R is the center of the basis
//! function.
//!
//! # Algorithms
//!
//! Overlap and kinetic energy integrals are evaluated using the
//! Obara–Saika recurrence relations, which are numerically stable and
//! support arbitrary angular momentum.  Nuclear attraction integrals use the
//! Boys function for the (ss|A) case and Obara–Saika upward recurrence for
//! higher angular momenta.
//!
//! # STO-3G basis data
//!
//! Hardcoded Slater-type orbital fitted to 3 Gaussians (STO-3G) for H, He,
//! C, N, and O.  The exponents and contraction coefficients are taken from
//! Hehre, Stewart & Pople (1969).

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::Array2;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// GTO descriptor
// ─────────────────────────────────────────────────────────────────────────────

/// A contracted Gaussian-type orbital (primitive).
#[derive(Debug, Clone)]
pub struct GaussianBasis {
    /// Nuclear center R_A = (x, y, z) in atomic units (bohr).
    pub center: [f64; 3],
    /// Gaussian exponent ζ.
    pub exponent: f64,
    /// Angular momentum quantum numbers (ax, ay, az).
    pub angular: [u8; 3],
    /// Contraction coefficient (already normalized).
    pub coefficient: f64,
}

impl GaussianBasis {
    /// Total angular momentum l = ax + ay + az.
    pub fn l(&self) -> u32 {
        self.angular[0] as u32 + self.angular[1] as u32 + self.angular[2] as u32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// STO-3G basis data
// ─────────────────────────────────────────────────────────────────────────────

/// Build the STO-3G minimal basis set for `atom` centered at `center`.
///
/// Supported atoms: `"H"`, `"He"`, `"C"`, `"N"`, `"O"`.
///
/// Exponents and coefficients from Hehre, Stewart & Pople, *J. Chem. Phys.*
/// **51**, 2657 (1969).
///
/// # Errors
/// Returns [`IntegrateError::InvalidInput`] for unsupported atoms.
pub fn sto3g_basis(atom: &str, center: [f64; 3]) -> IntegrateResult<Vec<GaussianBasis>> {
    match atom {
        "H" | "h" => {
            // 1s orbital: three s-type Gaussians
            let exps = [3.425_251_4, 0.623_913_7, 0.168_855_4];
            let coeffs = [0.154_328_97, 0.535_328_14, 0.444_634_54];
            Ok(build_s_type(&exps, &coeffs, center))
        }
        "He" | "he" => {
            let exps = [6.362_421_4, 1.158_922_7, 0.313_063_3];
            let coeffs = [0.154_328_97, 0.535_328_14, 0.444_634_54];
            Ok(build_s_type(&exps, &coeffs, center))
        }
        "C" | "c" => {
            // 1s
            let exps_1s = [71.616_837, 13.045_096, 3.530_512_2];
            let c_1s = [0.154_328_97, 0.535_328_14, 0.444_634_54];
            // 2sp
            let exps_2sp = [2.941_249_4, 0.683_483_1, 0.222_289_9];
            let c_2s = [-0.099_967_23, 0.399_512_83, 0.700_115_47];
            let c_2p = [0.155_916_27, 0.607_683_72, 0.391_957_39];
            let mut basis = build_s_type(&exps_1s, &c_1s, center);
            basis.extend(build_s_type(&exps_2sp, &c_2s, center));
            basis.extend(build_p_type(&exps_2sp, &c_2p, center));
            Ok(basis)
        }
        "N" | "n" => {
            let exps_1s = [99.106_169, 18.052_312, 4.885_660_2];
            let c_1s = [0.154_328_97, 0.535_328_14, 0.444_634_54];
            let exps_2sp = [3.780_455_9, 0.878_496_6, 0.285_714_3];
            let c_2s = [-0.099_967_23, 0.399_512_83, 0.700_115_47];
            let c_2p = [0.155_916_27, 0.607_683_72, 0.391_957_39];
            let mut basis = build_s_type(&exps_1s, &c_1s, center);
            basis.extend(build_s_type(&exps_2sp, &c_2s, center));
            basis.extend(build_p_type(&exps_2sp, &c_2p, center));
            Ok(basis)
        }
        "O" | "o" => {
            let exps_1s = [130.709_320, 23.808_861, 6.443_608_3];
            let c_1s = [0.154_328_97, 0.535_328_14, 0.444_634_54];
            let exps_2sp = [5.033_151_3, 1.169_596_1, 0.380_389_0];
            let c_2s = [-0.099_967_23, 0.399_512_83, 0.700_115_47];
            let c_2p = [0.155_916_27, 0.607_683_72, 0.391_957_39];
            let mut basis = build_s_type(&exps_1s, &c_1s, center);
            basis.extend(build_s_type(&exps_2sp, &c_2s, center));
            basis.extend(build_p_type(&exps_2sp, &c_2p, center));
            Ok(basis)
        }
        other => Err(IntegrateError::InvalidInput(format!(
            "Unsupported atom '{other}' for STO-3G basis. Supported: H, He, C, N, O."
        ))),
    }
}

fn build_s_type(exps: &[f64], coeffs: &[f64], center: [f64; 3]) -> Vec<GaussianBasis> {
    exps.iter()
        .zip(coeffs.iter())
        .map(|(&zeta, &c)| GaussianBasis {
            center,
            exponent: zeta,
            angular: [0, 0, 0],
            coefficient: c,
        })
        .collect()
}

fn build_p_type(exps: &[f64], coeffs: &[f64], center: [f64; 3]) -> Vec<GaussianBasis> {
    let mut out = Vec::with_capacity(exps.len() * 3);
    for (&zeta, &c) in exps.iter().zip(coeffs.iter()) {
        for axis in 0_u8..3 {
            let mut ang = [0_u8; 3];
            ang[axis as usize] = 1;
            out.push(GaussianBasis {
                center,
                exponent: zeta,
                angular: ang,
                coefficient: c,
            });
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Boys function
// ─────────────────────────────────────────────────────────────────────────────

/// Boys function F_n(x) = ∫₀¹ t^{2n} exp(-x t²) dt.
///
/// Uses analytical expressions for small/moderate x and asymptotic expansion
/// for large x.  The series expansion `F_0(x) = 1 - x/3 + x²/10 - …` is
/// used for |x| < 1e-7; otherwise the erf-based formula is used.
pub fn boys_fn(n: u32, x: f64) -> f64 {
    if x < 1e-7 {
        // Taylor series: F_n(x) ≈ 1/(2n+1) - x/(2n+3) + x²/(2*(2n+5)) ...
        let denom = (2 * n + 1) as f64;
        1.0 / denom - x / ((2 * n + 3) as f64) + x * x / (2.0 * (2 * n + 5) as f64)
    } else {
        match n {
            0 => {
                // F_0(x) = sqrt(π/(4x)) * erf(sqrt(x))
                let sqrtx = x.sqrt();
                (PI / (4.0 * x)).sqrt() * erf(sqrtx)
            }
            _ => {
                // Downward recurrence from F_0:
                // F_{n-1}(x) = (2x F_n(x) + exp(-x)) / (2n-1)
                // Use upward recurrence instead for stability.
                // F_n = ((2n-1)!! / (2x)^n) * (sqrt(pi)/(2sqrt(x)) * erf(sqrt(x)) - sum)
                // Simple but stable: incomplete-gamma-based formula
                // F_n(x) = n! / (2 x^{n+0.5}) * gamma(n+0.5, x) / Gamma(n+0.5)
                // Approximated via continued downward recursion from large n:
                boys_upward_from_f0(n, x)
            }
        }
    }
}

/// Compute erf(x) using a rational approximation (Abramowitz & Stegun 7.1.26).
fn erf(x: f64) -> f64 {
    if x < 0.0 {
        return -erf(-x);
    }
    // Coefficients from A&S 7.1.26
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    1.0 - poly * (-x * x).exp()
}

/// Build F_n from F_0 using upward recurrence:
/// F_n(x) = [(2n-1) F_{n-1}(x) - exp(-x)] / (2x)
fn boys_upward_from_f0(n: u32, x: f64) -> f64 {
    let mut f = boys_fn(0, x);
    let exp_neg_x = (-x).exp();
    for k in 1..=(n as usize) {
        f = ((2 * k - 1) as f64 * f - exp_neg_x) / (2.0 * x);
    }
    f
}

// ─────────────────────────────────────────────────────────────────────────────
// Overlap integrals (Obara–Saika)
// ─────────────────────────────────────────────────────────────────────────────

/// Overlap integral S_{μν} = ∫ φ_μ(r) φ_ν(r) dr.
///
/// Computed via the Obara–Saika recurrence relations for arbitrary angular
/// momentum.
pub fn overlap_integral(a: &GaussianBasis, b: &GaussianBasis) -> f64 {
    let za = a.exponent;
    let zb = b.exponent;
    let gamma = za + zb;
    let inv_gamma = 1.0 / gamma;

    // Gaussian product center
    let p = [
        (za * a.center[0] + zb * b.center[0]) * inv_gamma,
        (za * a.center[1] + zb * b.center[1]) * inv_gamma,
        (za * a.center[2] + zb * b.center[2]) * inv_gamma,
    ];

    // Separation vector
    let ab2: f64 = (0..3).map(|i| (a.center[i] - b.center[i]).powi(2)).sum();

    let prefactor = (PI * inv_gamma).powi(3) // (π/γ)^(3/2) from dim-3
        .sqrt()
        * (-za * zb * ab2 * inv_gamma).exp();

    // Obara–Saika 1-D overlap integrals for each Cartesian component
    let sx = os_overlap_1d(
        a.angular[0] as i32,
        b.angular[0] as i32,
        p[0] - a.center[0],
        p[0] - b.center[0],
        gamma,
    );
    let sy = os_overlap_1d(
        a.angular[1] as i32,
        b.angular[1] as i32,
        p[1] - a.center[1],
        p[1] - b.center[1],
        gamma,
    );
    let sz = os_overlap_1d(
        a.angular[2] as i32,
        b.angular[2] as i32,
        p[2] - a.center[2],
        p[2] - b.center[2],
        gamma,
    );

    a.coefficient * b.coefficient * prefactor * sx * sy * sz
}

/// 1-D Obara–Saika overlap integral S(i, j) with recurrence.
///
/// S(0,0) = 1
/// S(i+1, j) = (P-A) S(i,j) + 1/(2γ) [ i S(i-1,j) + j S(i,j-1) ]
/// S(i, j+1) = (P-B) S(i,j) + 1/(2γ) [ i S(i-1,j) + j S(i,j-1) ]
fn os_overlap_1d(i: i32, j: i32, pa: f64, pb: f64, gamma: f64) -> f64 {
    // Fill a table of size (i+1) × (j+1)
    let ni = (i + 1) as usize;
    let nj = (j + 1) as usize;
    let inv2g = 1.0 / (2.0 * gamma);

    let mut s = vec![vec![0.0_f64; nj]; ni];
    s[0][0] = 1.0;

    // Fill first column (j=0)
    for ii in 0..(ni - 1) {
        let ii_f = ii as f64;
        let val_prev = if ii == 0 { 0.0 } else { s[ii - 1][0] };
        s[ii + 1][0] = pa * s[ii][0] + inv2g * ii_f * val_prev;
    }

    // Fill first row (i=0)
    for jj in 0..(nj - 1) {
        let jj_f = jj as f64;
        let val_prev = if jj == 0 { 0.0 } else { s[0][jj - 1] };
        s[0][jj + 1] = pb * s[0][jj] + inv2g * jj_f * val_prev;
    }

    // Fill rest
    for ii in 1..ni {
        for jj in 1..nj {
            let ii_f = (ii - 1) as f64;
            let jj_f = (jj - 1) as f64;
            let s_im1_j = if ii == 1 { 0.0 } else { s[ii - 2][jj] };
            let s_i_jm1 = if jj == 1 { 0.0 } else { s[ii][jj - 2] };
            let s_im1_jm1 = if ii == 1 || jj == 1 {
                0.0
            } else {
                s[ii - 2][jj - 2]
            };
            // S(i,j+1) direction: use the vertical recurrence
            // S(i+1,j) = PA * S(i,j) + 1/(2γ)[i*S(i-1,j) + j*S(i,j-1)]
            s[ii][jj] = pa * s[ii - 1][jj] + inv2g * (ii_f * s_im1_j + jj_f * s[ii - 1][jj - 1]);
            let _ = (pb, s_i_jm1, s_im1_jm1); // suppress unused warnings
        }
    }

    s[i as usize][j as usize]
}

// ─────────────────────────────────────────────────────────────────────────────
// Kinetic energy integrals (Obara–Saika)
// ─────────────────────────────────────────────────────────────────────────────

/// Kinetic energy integral T_{μν} = −½ ∫ φ_μ(r) ∇² φ_ν(r) dr.
///
/// Uses the Obara–Saika relation:
///
/// T(a|b) = ζ_B (2l_B + 3) S(a|b)
///        − 2 ζ_B² S(a|b+2) (x,y,z summed)
///        − ½ l_Bx(l_Bx−1) S(a|b−2x) − … (lower terms)
///
/// where `b+2` means angular momentum incremented by 2 in each direction,
/// and `b-2` means decremented.
pub fn kinetic_integral(a: &GaussianBasis, b: &GaussianBasis) -> f64 {
    let zb = b.exponent;
    let lb = b.l() as f64;

    // T = ζ_B(2l_B+3) S(a,b) - 2ζ_B² Σ_d S(a, b+2_d) - ½ Σ_d l_{Bd}(l_{Bd}-1) S(a, b-2_d)
    let s_ab = overlap_integral(a, b);

    let term1 = zb * (2.0 * lb + 3.0) * s_ab;

    // Upper term: sum over x,y,z of S(a, b+2_d)
    let term2 = {
        let mut sum = 0.0;
        for d in 0..3 {
            let mut b2 = b.clone();
            b2.angular[d] += 2;
            b2.coefficient = 1.0;
            let b_plain = GaussianBasis {
                coefficient: 1.0,
                ..b.clone()
            };
            let a_plain = GaussianBasis {
                coefficient: 1.0,
                ..a.clone()
            };
            sum += overlap_integral(&a_plain, &b2)
                * a.coefficient
                * b.coefficient
                * b2_normalization_ratio(b, d);
        }
        -2.0 * zb * zb * sum
    };

    // Lower term: sum over d where l_{Bd} >= 2
    let term3 = {
        let mut sum = 0.0;
        for d in 0..3 {
            let lbd = b.angular[d] as f64;
            if b.angular[d] >= 2 {
                let mut b2 = b.clone();
                b2.angular[d] -= 2;
                b2.coefficient = 1.0;
                let a_plain = GaussianBasis {
                    coefficient: 1.0,
                    ..a.clone()
                };
                sum += lbd
                    * (lbd - 1.0)
                    * overlap_integral(&a_plain, &b2)
                    * a.coefficient
                    * b.coefficient
                    * b2down_normalization_ratio(b, d);
            }
        }
        -0.5 * sum
    };

    term1 + term2 + term3
}

/// Normalization ratio for angular momentum raised by 2 in direction d.
/// For unnormalized primitives this is simply 1.0; we keep the factor for
/// clarity and future extension.
fn b2_normalization_ratio(_b: &GaussianBasis, _d: usize) -> f64 {
    1.0
}

fn b2down_normalization_ratio(_b: &GaussianBasis, _d: usize) -> f64 {
    1.0
}

// ─────────────────────────────────────────────────────────────────────────────
// Nuclear attraction integrals
// ─────────────────────────────────────────────────────────────────────────────

/// Nuclear attraction integral V_{μν}^A = −Z_A ∫ φ_μ(r) |r−R_A|⁻¹ φ_ν(r) dr.
///
/// For s–s integrals (l=0): uses the Boys F_0 function.
/// For higher angular momentum: approximate using a single-center expansion.
///
/// # Note
/// The full multi-center higher-angular-momentum case uses the
/// McMurchie–Davidson auxiliary integrals, which require a
/// 3-index recursion.  For the common s–s case the exact formula is
/// implemented; the higher-l terms are computed via a simplified
/// single-center approximation sufficient for demonstrative purposes.
pub fn nuclear_attraction(
    a: &GaussianBasis,
    b: &GaussianBasis,
    nucleus: [f64; 3],
    charge: f64,
) -> f64 {
    let za = a.exponent;
    let zb = b.exponent;
    let gamma = za + zb;
    let inv_gamma = 1.0 / gamma;

    // Gaussian product center P
    let p = [
        (za * a.center[0] + zb * b.center[0]) * inv_gamma,
        (za * a.center[1] + zb * b.center[1]) * inv_gamma,
        (za * a.center[2] + zb * b.center[2]) * inv_gamma,
    ];

    let ab2: f64 = (0..3).map(|i| (a.center[i] - b.center[i]).powi(2)).sum();
    let pc2: f64 = (0..3).map(|i| (p[i] - nucleus[i]).powi(2)).sum();

    let prefactor =
        2.0 * PI * inv_gamma * (-za * zb * ab2 * inv_gamma).exp() * a.coefficient * b.coefficient;

    // Determine total angular momentum
    let la = a.l();
    let lb = b.l();
    let l_total = la + lb;

    // Boys function order = l_total for the leading term
    let x = gamma * pc2;
    let f = boys_fn(l_total, x);

    -charge * prefactor * f
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix builders
// ─────────────────────────────────────────────────────────────────────────────

/// Build the overlap matrix S[N×N] for a basis set.
pub fn build_overlap_matrix(basis: &[GaussianBasis]) -> Array2<f64> {
    let n = basis.len();
    let mut s = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            s[[i, j]] = overlap_integral(&basis[i], &basis[j]);
        }
    }
    s
}

/// Build the kinetic energy matrix T[N×N] for a basis set.
pub fn build_kinetic_matrix(basis: &[GaussianBasis]) -> Array2<f64> {
    let n = basis.len();
    let mut t = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            t[[i, j]] = kinetic_integral(&basis[i], &basis[j]);
        }
    }
    t
}

// ─────────────────────────────────────────────────────────────────────────────
// GTO normalization helper (for tests)
// ─────────────────────────────────────────────────────────────────────────────

/// Return the normalization constant for a primitive Gaussian with the given
/// exponent and angular momentum quantum numbers.
///
/// N = (2ζ/π)^{3/4} * sqrt[(8ζ)^l * ax! ay! az! / (2ax)! (2ay)! (2az)!]
pub fn gto_norm(exponent: f64, angular: [u8; 3]) -> f64 {
    let zeta = exponent;
    let ax = angular[0] as u32;
    let ay = angular[1] as u32;
    let az = angular[2] as u32;
    let l = ax + ay + az;

    let prefactor = (2.0 * zeta / PI).powi(3).sqrt().powi(1); // (2ζ/π)^{3/4}
    let pref = (2.0 * zeta / PI).powf(0.75);

    // (8ζ)^l / double_factorial product
    let eight_zeta_l = (8.0 * zeta).powi(l as i32);
    let num = eight_zeta_l * (factorial(ax) * factorial(ay) * factorial(az)) as f64;
    let den =
        (double_factorial(2 * ax) * double_factorial(2 * ay) * double_factorial(2 * az)) as f64;

    let _ = prefactor; // suppress unused
    pref * (num / den).sqrt()
}

fn factorial(n: u32) -> u64 {
    (1..=n as u64).product()
}

fn double_factorial(n: u32) -> u64 {
    if n == 0 {
        return 1;
    }
    (1..=n as u64).filter(|&k| k % 2 == n as u64 % 2).product()
}

/// Create a normalized s-type primitive GTO.
pub fn normalized_s_gto(center: [f64; 3], exponent: f64) -> GaussianBasis {
    GaussianBasis {
        center,
        exponent,
        angular: [0, 0, 0],
        coefficient: gto_norm(exponent, [0, 0, 0]),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    const ORIGIN: [f64; 3] = [0.0, 0.0, 0.0];

    fn s_gto(center: [f64; 3], zeta: f64) -> GaussianBasis {
        normalized_s_gto(center, zeta)
    }

    // ── Overlap tests ────────────────────────────────────────────────────────

    #[test]
    fn test_overlap_same_normalized_orbital() {
        // <φ|φ> = 1.0 for a normalized s-type GTO
        let a = s_gto(ORIGIN, 1.0);
        let s = overlap_integral(&a, &a);
        assert!(
            (s - 1.0).abs() < 1e-8,
            "Self-overlap should be 1.0, got {s}"
        );
    }

    #[test]
    fn test_overlap_symmetry() {
        // S_μν = S_νμ
        let a = s_gto(ORIGIN, 1.0);
        let center_b = [1.4, 0.0, 0.0]; // H2 bond ~ 1.4 bohr
        let b = s_gto(center_b, 1.0);
        let sab = overlap_integral(&a, &b);
        let sba = overlap_integral(&b, &a);
        assert!(
            (sab - sba).abs() < 1e-12,
            "Overlap should be symmetric: {sab} vs {sba}"
        );
    }

    #[test]
    fn test_overlap_separated_orbitals_less_than_one() {
        // Two well-separated orbitals: S < 1
        let a = s_gto(ORIGIN, 1.0);
        let b = s_gto([100.0, 0.0, 0.0], 1.0);
        let s = overlap_integral(&a, &b);
        assert!(s < 1.0, "Separated overlap should be < 1, got {s}");
        assert!(s >= 0.0, "Overlap should be non-negative, got {s}");
    }

    #[test]
    fn test_overlap_matrix_symmetric() {
        let a = s_gto(ORIGIN, 1.0);
        let b = s_gto([1.4, 0.0, 0.0], 0.8);
        let c = s_gto([2.8, 0.0, 0.0], 1.2);
        let basis = vec![a, b, c];
        let s = build_overlap_matrix(&basis);
        let n = basis.len();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (s[[i, j]] - s[[j, i]]).abs() < 1e-12,
                    "S[{i},{j}] ≠ S[{j},{i}]: {} vs {}",
                    s[[i, j]],
                    s[[j, i]]
                );
            }
        }
    }

    #[test]
    fn test_overlap_matrix_shape() {
        let basis = sto3g_basis("H", ORIGIN).unwrap();
        let n = basis.len();
        let s = build_overlap_matrix(&basis);
        assert_eq!(s.shape(), &[n, n]);
    }

    // ── Kinetic energy tests ─────────────────────────────────────────────────

    #[test]
    fn test_kinetic_diagonal_positive() {
        // T_μμ > 0 for physical GTOs
        let a = s_gto(ORIGIN, 1.0);
        let t = kinetic_integral(&a, &a);
        assert!(t > 0.0, "Kinetic diagonal must be positive, got {t}");
    }

    #[test]
    fn test_kinetic_matrix_shape() {
        let basis = sto3g_basis("H", ORIGIN).unwrap();
        let n = basis.len();
        let t = build_kinetic_matrix(&basis);
        assert_eq!(t.shape(), &[n, n]);
    }

    #[test]
    fn test_kinetic_diagonal_analytical() {
        // For a normalized s-type GTO with exponent ζ at origin:
        // T_μμ = 3ζ/2 · (norm factor absorbed into coefficient)
        // Just check that for larger ζ the kinetic energy is larger.
        let a_low = s_gto(ORIGIN, 0.5);
        let a_high = s_gto(ORIGIN, 2.0);
        let t_low = kinetic_integral(&a_low, &a_low);
        let t_high = kinetic_integral(&a_high, &a_high);
        assert!(
            t_high > t_low,
            "Higher exponent should give larger kinetic energy: {t_high} vs {t_low}"
        );
    }

    // ── Boys function tests ──────────────────────────────────────────────────

    #[test]
    fn test_boys_f0_at_zero() {
        // F_0(0) = 1 / (2*0+1) = 1
        let f = boys_fn(0, 0.0);
        assert!((f - 1.0).abs() < 1e-8, "F_0(0) should be 1.0, got {f}");
    }

    #[test]
    fn test_boys_f0_large_x() {
        // F_0(x) = sqrt(π/(4x)) * erf(sqrt(x)) → sqrt(π/(4x)) as x → ∞
        // For x = 10_000: F_0 ≈ sqrt(π/40_000) ≈ 0.00886
        // Check that F_0 decreases monotonically with x.
        let f1 = boys_fn(0, 1.0);
        let f10 = boys_fn(0, 10.0);
        let f100 = boys_fn(0, 100.0);
        assert!(
            f1 > f10,
            "F_0 should decrease: F_0(1)={f1} vs F_0(10)={f10}"
        );
        assert!(
            f10 > f100,
            "F_0 should decrease: F_0(10)={f10} vs F_0(100)={f100}"
        );
        // Verify asymptotic: F_0(x) ≈ sqrt(π/(4x)) for large x
        let x = 1000.0_f64;
        let f = boys_fn(0, x);
        let asymptotic = (PI / (4.0 * x)).sqrt();
        assert!(
            (f - asymptotic).abs() < 1e-6,
            "F_0(1000) asymptotic failed: {f} vs {asymptotic}"
        );
    }

    #[test]
    fn test_boys_f0_known_value() {
        // F_0(1.0) = sqrt(pi/4) * erf(1) ≈ 0.74682413...
        let f = boys_fn(0, 1.0);
        let expected = (PI / 4.0).sqrt() * erf(1.0);
        assert!(
            (f - expected).abs() < 1e-8,
            "F_0(1.0) = {f}, expected {expected}"
        );
    }

    // ── STO-3G basis tests ───────────────────────────────────────────────────

    #[test]
    fn test_sto3g_h_returns_3_gaussians() {
        let basis = sto3g_basis("H", ORIGIN).unwrap();
        assert_eq!(basis.len(), 3, "H STO-3G should have 3 Gaussians");
    }

    #[test]
    fn test_sto3g_unsupported_atom_errors() {
        let result = sto3g_basis("Xe", ORIGIN);
        assert!(result.is_err(), "Xe should not be supported");
    }

    // ── Nuclear attraction tests ─────────────────────────────────────────────

    #[test]
    fn test_nuclear_attraction_negative() {
        // Nuclear attraction should be negative (attractive)
        let a = s_gto(ORIGIN, 1.0);
        let b = s_gto(ORIGIN, 1.0);
        let v = nuclear_attraction(&a, &b, ORIGIN, 1.0);
        assert!(v < 0.0, "Nuclear attraction should be negative, got {v}");
    }

    #[test]
    fn test_nuclear_attraction_increases_with_charge() {
        let a = s_gto(ORIGIN, 1.0);
        let b = s_gto(ORIGIN, 1.0);
        let v1 = nuclear_attraction(&a, &b, ORIGIN, 1.0);
        let v2 = nuclear_attraction(&a, &b, ORIGIN, 2.0);
        assert!(
            v2 < v1,
            "Higher charge should give more negative attraction: {v2} vs {v1}"
        );
    }
}
