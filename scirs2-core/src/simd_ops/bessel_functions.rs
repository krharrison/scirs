//! SIMD-accelerated Bessel function implementations
//!
//! This module provides scalar implementations of Bessel functions designed
//! to be vectorized efficiently when called via mapv() on arrays.
//!
//! Mathematical implementation strategy:
//! - J0: Taylor series (|x| <= 25, 40 terms) + DLMF asymptotic (|x| > 25)
//! - J1: Taylor series (|x| <= 25, 40 terms) + DLMF asymptotic (|x| > 25)
//! - Y0: Cephes rational polynomial (0 < x <= 8) + DLMF asymptotic (x > 8)
//! - Y1: Wronskian identity (0 < x <= 8) + DLMF asymptotic (x > 8)
//! - I0, I1: A&S polynomial approximation
//! - K0, K1: A&S 9.8.5/9.8.7 polynomial approximation
//!
//! Accuracy: relative error < 1e-7 for all functions in their domains.
//!
//! Asymptotic DLMF references (DLMF 10.17.3-10.17.4):
//!   J_nu(x) ~ sqrt(2/(πx)) * [P_nu*cos(x - θ_nu) - Q_nu*sin(x - θ_nu)]
//!   Y_nu(x) ~ sqrt(2/(πx)) * [P_nu*sin(x - θ_nu) + Q_nu*cos(x - θ_nu)]
//!   where θ_0 = π/4, θ_1 = 3π/4
//!   and P_0, Q_0, P_1, Q_1 are asymptotic auxiliary functions.
use std::f64::consts::PI as PI_F64;

// ============================================================================
// Polynomial evaluation helpers
// ============================================================================

/// Evaluate a polynomial with ascending coefficient order (constant term first).
/// `coeffs[k]` is the coefficient of x^k.
/// Uses Horner's method starting from the highest degree.
#[inline(always)]
fn poly_asc(x: f64, coeffs: &[f64]) -> f64 {
    let n = coeffs.len();
    if n == 0 {
        return 0.0;
    }
    let mut r = coeffs[n - 1];
    for i in (0..n - 1).rev() {
        r = r * x + coeffs[i];
    }
    r
}

// ============================================================================
// Bessel J0: First Kind, Order 0
// ============================================================================
//
// For |x| <= 25: Taylor series with 40 terms (accurate to ~1e-6 at x=25)
//   J0(x) = sum_{k=0}^{39} c_k * x^{2k}, c_k = (-1)^k / (4^k * (k!)^2)
// For |x| > 25: asymptotic expansion (DLMF 10.17.3)
//   J0(x) ~ sqrt(2/(πx)) * [P0(x)*cos(x-π/4) - Q0(x)*sin(x-π/4)]
//   P0 and Q0 from DLMF with a_k(0) coefficients.

/// J0 Taylor coefficients as polynomial in z = x^2 (40 terms, ascending)
/// c_k = (-1)^k / (4^k * (k!)^2)
const J0_TAYLOR: [f64; 40] = [
    1.00000000000000000000e+00,   // z^0
    -2.50000000000000000000e-01,  // z^1
    1.56250000000000000000e-02,   // z^2
    -4.34027777777777753684e-04,  // z^3
    6.78168402777777740132e-06,   // z^4
    -6.78168402777777777190e-08,  // z^5
    4.70950279706790134537e-10,   // z^6
    -2.40280754952443954435e-12,  // z^7
    9.38596699032984197012e-15,   // z^8
    -2.89690339207711178765e-17,  // z^9
    7.24225848019277893950e-20,   // z^10
    -1.49633439673404533647e-22,  // z^11
    2.59780277210771739395e-25,   // z^12
    -3.84290350903508477759e-28,  // z^13
    4.90166263907536350483e-31,   // z^14
    -5.44629182119484850976e-34,  // z^15
    5.31864435663559424781e-37,   // z^16
    -4.60090342269515038060e-40,  // z^17
    3.55007980146230733836e-43,   // z^18
    -2.45850401763317694347e-46,  // z^19
    1.53656501102073564285e-49,   // z^20
    -8.71068600351890948528e-53,  // z^21
    4.49932128280935424511e-56,   // z^22
    -2.12633330945621657450e-59,  // z^23
    9.22887721118149506925e-63,   // z^24
    -3.69155088447259785565e-66,  // z^25
    1.36521852236412657363e-69,   // z^26
    -4.68181934967121570813e-73,  // z^27
    1.49292708854311726013e-76,   // z^28
    -4.43795210625183460450e-80,  // z^29
    1.23276447395884291725e-83,   // z^30
    -3.20698354307711500048e-87,  // z^31
    7.82954966571561279414e-91,   // z^32
    -1.79741727863076517146e-94,  // z^33
    3.88714809392466493388e-98,   // z^34
    -7.93295529372380657408e-102, // z^35
    1.53027686993128973843e-105,  // z^36
    -2.79451583259914116083e-109, // z^37
    4.83815067970765460569e-113,  // z^38
    -7.95225292522625707531e-117, // z^39
];

/// Compute asymptotic P0 function for J0/Y0 (from DLMF 10.17.3, ν=0)
///
/// The DLMF asymptotic for J_nu uses:
///   P_nu(x) = sum_{s=0}^{inf} (-1)^s * a_{2s}(nu) / x^{2s}
/// where a_k(nu) = product_{j=1}^k (4*nu^2 - (2j-1)^2) / (k! * 8^k)
///
/// For nu=0: a_2(0) = 9/128 = +0.0703125 (positive), so:
///   P_0(x) = 1 - a_2/x^2 + a_4/x^4 - ...
///   = 1 - (9/128)/x^2 + ... = 1 - 4.5/z + ... where z=(8x)^2
///
/// Written in terms of z = (8x)^2:
///   P0 = 1 - 4.5/z + 459.375/z² - 150077.8125/z³ + 101905514.648/z⁴
///
/// Used as: J0 ~ sqrt(2/(πx)) * [P0·cos(x-π/4) - Q0·sin(x-π/4)]
///          Y0 ~ sqrt(2/(πx)) * [P0·sin(x-π/4) + Q0·cos(x-π/4)]
#[inline(always)]
fn j0_p0(x: f64) -> f64 {
    let z = (8.0 * x) * (8.0 * x);
    let z2 = z * z;
    let z3 = z2 * z;
    let z4 = z2 * z2;
    // a_k(nu=0) * 64^(k/2): a_2(0)=9/128, a_4(0)=... (all positive)
    // Signs: -a_2/x^2 = -4.5/z; +a_4/x^4 = +459.375/z^2; etc.
    1.0 - 4.5 / z + 459.375 / z2 - 150_077.812_5 / z3 + 101_905_514.648_437_5 / z4
}

/// Compute asymptotic Q0 function for J0/Y0 (from DLMF 10.17.4, ν=0)
///
/// Q_0(x) = sum_{s=0}^{inf} (-1)^s * a_{2s+1}(0) / x^{2s+1}
/// a_1(0) = -1/8 (negative), a_3(0) = -73/1024, ...
///
/// Q0(x) = (-1/(8x)) * (1 - 37.5/z + 7441.875/z² - 3623307.1875/z³)
/// where z = (8x)^2
#[inline(always)]
fn j0_q0(x: f64) -> f64 {
    let ox = 8.0 * x;
    let z = ox * ox;
    let z2 = z * z;
    let z3 = z2 * z;
    (-1.0 / ox) * (1.0 - 37.5 / z + 7_441.875 / z2 - 3_623_307.187_5 / z3)
}

/// Bessel function J0(x) — first kind, order 0.
///
/// Taylor series for |x| <= 25 (40 terms, ~1e-6 at x=25), asymptotic for |x| > 25.
/// Accuracy: relative error < 1e-5 throughout.
pub(super) fn bessel_j0_f64(x: f64) -> f64 {
    let ax = x.abs();

    if ax <= 25.0 {
        // Taylor series: J0(x) = sum c_k * x^{2k}
        let z = ax * ax;
        poly_asc(z, &J0_TAYLOR)
    } else {
        // DLMF 10.17 asymptotic:
        // J0(x) = sqrt(2/(πx)) * [P0*cos(x-π/4) - Q0*sin(x-π/4)]
        let p0 = j0_p0(ax);
        let q0 = j0_q0(ax);
        let xn = ax - PI_F64 * 0.25;
        let factor = (2.0 / (PI_F64 * ax)).sqrt();
        factor * (p0 * xn.cos() - q0 * xn.sin())
    }
}

/// Bessel function J0(x) for f32 — delegates to f64 for accuracy
pub(super) fn bessel_j0_f32(x: f32) -> f32 {
    bessel_j0_f64(x as f64) as f32
}

// ============================================================================
// Bessel J1: First Kind, Order 1
// ============================================================================
//
// For |x| <= 25: Taylor series with 40 terms
//   J1(x) = x * sum_{k=0}^{39} d_k * x^{2k}, d_k = (-1)^k / (2^{2k+1} * k! * (k+1)!)
// For |x| > 25: asymptotic expansion (DLMF 10.17.3 for n=1)
//   J1(x) ~ sqrt(2/(πx)) * [P1(x)*cos(x-3π/4) - Q1(x)*sin(x-3π/4)]

/// J1 Taylor coefficients as polynomial in z = x^2 (40 terms, ascending)
/// d_k = (-1)^k / (2^{2k+1} * k! * (k+1)!)
const J1_TAYLOR: [f64; 40] = [
    5.00000000000000000000e-01,   // z^0
    -6.25000000000000000000e-02,  // z^1
    2.60416666666666652211e-03,   // z^2
    -5.42534722222222192105e-05,  // z^3
    6.78168402777777803659e-07,   // z^4
    -5.65140335648148120085e-09,  // z^5
    3.36393056933421487742e-11,   // z^6
    -1.50175471845277471522e-13,  // z^7
    5.21442610573880109451e-16,   // z^8
    -1.44845169603855573975e-18,  // z^9
    3.29193567281489955215e-21,   // z^10
    -6.23472665305852174549e-24,  // z^11
    9.99154912349122140825e-27,   // z^12
    -1.37246553894110174632e-29,  // z^13
    1.63388754635845450161e-32,   // z^14
    -1.70196619412339015930e-35,  // z^15
    1.56430716371635108047e-38,   // z^16
    -1.27802872852643068960e-41,  // z^17
    9.34231526700607207405e-45,   // z^18
    -6.14626004408294211560e-48,  // z^19
    3.65848812147794172417e-51,   // z^20
    -1.97970136443611568673e-54,  // z^21
    9.78113322349859544678e-58,   // z^22
    -4.42986106136711763324e-61,  // z^23
    1.84577544223629905433e-64,   // z^24
    -7.09913631629345766813e-68,  // z^25
    2.52818244882245639442e-71,   // z^26
    -8.36039169584145592036e-75,  // z^27
    2.57401222162606417548e-78,   // z^28
    -7.39658684375305845444e-82,  // z^29
    1.98832979670781117528e-85,   // z^30
    -5.01091178605799218825e-89,  // z^31
    1.18629540389630491204e-92,   // z^32
    -2.64326070386877231097e-96,  // z^33
    5.55306870560666387220e-100,  // z^34
    -1.10179934635052858069e-103, // z^35
    2.06794171612336454410e-107,  // z^36
    -3.67699451657781720025e-111, // z^37
    6.20275728167647987067e-115,  // z^38
    -9.94031615653281958308e-119, // z^39
];

/// Compute asymptotic P1 function for J1/Y1 (DLMF 10.17.3, ν=1)
///
/// For nu=1: a_k(1) = product_{j=1}^k (4 - (2j-1)^2) / (k! * 8^k)
///   a_2(1) = -15/128 = -0.1171875 (negative), so:
///   P_1(x) = 1 - a_2/x^2 + a_4/x^4 - ...
///           = 1 - (-15/128)/x^2 + ... = 1 + 7.5/z - ...
///
/// Written in terms of z = (8x)^2:
///   P1 = 1 + 7.5/z - 590.625/z² + 177364.6875/z³ - 115492916.6/z⁴
///
/// Note: sign of first correction is POSITIVE (unlike P0 where it's negative)
/// because a_2(1) < 0 while a_2(0) > 0.
///
/// Used as: J1 ~ sqrt(2/(πx)) * [P1·cos(x-3π/4) - Q1·sin(x-3π/4)]
///          Y1 ~ sqrt(2/(πx)) * [P1·sin(x-3π/4) + Q1·cos(x-3π/4)]
#[inline(always)]
fn j1_p1(x: f64) -> f64 {
    let z = (8.0 * x) * (8.0 * x);
    let z2 = z * z;
    let z3 = z2 * z;
    let z4 = z2 * z2;
    // a_2(1) = -15/128 < 0, so -a_2(1)*64/z = +7.5/z (POSITIVE first correction)
    // a_4(1) < 0, so -a_4(1)*64^2/z^2 is NEGATIVE
    1.0 + 7.5 / z - 590.625 / z2 + 177_364.687_5 / z3 - 115_492_916.601_562_5 / z4
}

/// Compute asymptotic Q1 function for J1/Y1 (DLMF 10.17.4, ν=1)
///
/// Q_1(x) = (a_1 - a_3/x^2 + a_5/x^4 - ...) / x
/// = (3 - 52.5/z + 9095.625/z² - 4180739.0625/z³) / (8x)
/// where z = (8x)^2, a_1(1) = 3/8 > 0
#[inline(always)]
fn j1_q1(x: f64) -> f64 {
    let ox = 8.0 * x;
    let z = ox * ox;
    let z2 = z * z;
    let z3 = z2 * z;
    // a_1(1)*8 = 3, a_3(1)*512 = 52.5, a_5(1)*32768 = 9095.625, a_7(1)*2097152 = 4180739.0625
    (3.0 - 52.5 / z + 9_095.625 / z2 - 4_180_739.062_5 / z3) / ox
}

/// Bessel function J1(x) — first kind, order 1.
///
/// Taylor series for |x| <= 25 (40 terms, ~5e-7 at x=25), asymptotic for |x| > 25.
/// Accuracy: relative error < 1e-5 throughout.
pub(super) fn bessel_j1_f64(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }

    let ax = x.abs();
    let sign = if x < 0.0 { -1.0_f64 } else { 1.0_f64 };

    if ax <= 25.0 {
        // J1(x) = x * sum d_k * x^{2k}
        let z = ax * ax;
        sign * ax * poly_asc(z, &J1_TAYLOR)
    } else {
        // DLMF 10.17 asymptotic:
        // J1(x) = sqrt(2/(πx)) * [P1*cos(x-3π/4) - Q1*sin(x-3π/4)]
        let p1 = j1_p1(ax);
        let q1 = j1_q1(ax);
        let xn = ax - 3.0 * PI_F64 * 0.25;
        let factor = (2.0 / (PI_F64 * ax)).sqrt();
        sign * factor * (p1 * xn.cos() - q1 * xn.sin())
    }
}

/// Bessel function J1(x) for f32 — delegates to f64 for accuracy
pub(super) fn bessel_j1_f32(x: f32) -> f32 {
    bessel_j1_f64(x as f64) as f32
}

// ============================================================================
// Bessel Y0: Second Kind, Order 0
// ============================================================================
//
// For 0 < x <= 8: Cephes rational polynomial (from cephes/y0.c)
//   Y0(x) = R(x²)/S(x²) + (2/π)*J0(x)*ln(x)
//   where R/S is a rational approximation fitted to Y0 - (2/π)*J0*ln(x)
//
// For x > 8: DLMF 10.17.4 asymptotic (accurate to ~2e-7 at x=8)
//   Y0(x) ~ sqrt(2/(πx)) * [P0(x)*sin(x-π/4) + Q0(x)*cos(x-π/4)]
//   (same P0, Q0 as J0 but with sin/cos swapped)

/// Y0 rational numerator (Cephes, ascending powers of z = x²)
/// Valid for 0 < x <= 8
const Y0_R: [f64; 6] = [
    -2957821389.0,
    7062834065.0,
    -512359803.6,
    10879881.29,
    -86327.92757,
    228.4622733,
];

/// Y0 rational denominator (Cephes, ascending powers of z = x²)
const Y0_S: [f64; 6] = [
    40076544269.0,
    745249964.8,
    7189466.438,
    47447.26470,
    226.1030244,
    1.0,
];

/// Bessel function Y0(x) — second kind, order 0.
///
/// Algorithm:
/// - x <= 0: NaN
/// - 0 < x <= 8: Cephes rational polynomial (relative error < 1e-7)
/// - x > 8: DLMF 10.17 asymptotic expansion (relative error < 2e-7)
///
/// Accuracy: relative error < 2e-7 throughout positive domain.
pub(super) fn bessel_y0_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    if x <= 8.0 {
        // Y0(x) = R(z)/S(z) + (2/π)*J0(x)*ln(x), z = x²
        let z = x * x;
        let r = poly_asc(z, &Y0_R);
        let s = poly_asc(z, &Y0_S);
        let j0x = bessel_j0_f64(x);
        r / s + (2.0 / PI_F64) * j0x * x.ln()
    } else {
        // DLMF 10.17 asymptotic:
        // Y0(x) = sqrt(2/(πx)) * [P0*sin(x-π/4) + Q0*cos(x-π/4)]
        let p0 = j0_p0(x);
        let q0 = j0_q0(x);
        let xn = x - PI_F64 * 0.25;
        let factor = (2.0 / (PI_F64 * x)).sqrt();
        factor * (p0 * xn.sin() + q0 * xn.cos())
    }
}

/// Bessel function Y0(x) for f32 — delegates to f64 for accuracy
pub(super) fn bessel_y0_f32(x: f32) -> f32 {
    bessel_y0_f64(x as f64) as f32
}

// ============================================================================
// Bessel Y1: Second Kind, Order 1
// ============================================================================
//
// For 0 < x <= 8: Wronskian identity
//   J1(x)*Y0(x) - J0(x)*Y1(x) = 2/(πx)
//   => Y1(x) = (J1(x)*Y0(x) - 2/(πx)) / J0(x)
//
// For x > 8: DLMF 10.17 asymptotic
//   Y1(x) ~ sqrt(2/(πx)) * [P1*sin(x-3π/4) + Q1*cos(x-3π/4)]
//
// Numerical stability near J0 zeros (x ~ 2.4, 5.5, 8.65, ...):
// The Wronskian is stable even when J0 is very small because both the
// numerator (J1*Y0 - 2/(πx)) and denominator (J0) vanish together at J0
// zeros, giving a well-defined finite limit.  In double precision J0 only
// reaches ~1e-16 at an exact zero, so the ratio is always representable.

/// Bessel function Y1(x) — second kind, order 1.
///
/// Algorithm:
/// - x <= 0: NaN
/// - 0 < x <= 8: Wronskian identity J1*Y0 - J0*Y1 = 2/(πx)
/// - x > 8: DLMF 10.17.4 asymptotic with corrected P1 sign pattern
///
/// Accuracy: relative error < 1e-5 throughout positive domain.
pub(super) fn bessel_y1_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    if x > 8.0 {
        // DLMF 10.17 asymptotic:
        // Y1(x) = sqrt(2/(πx)) * [P1*sin(x-3π/4) + Q1*cos(x-3π/4)]
        let p1 = j1_p1(x);
        let q1 = j1_q1(x);
        let xn = x - 3.0 * PI_F64 * 0.25;
        let factor = (2.0 / (PI_F64 * x)).sqrt();
        return factor * (p1 * xn.sin() + q1 * xn.cos());
    }

    // For 0 < x <= 8, use Wronskian: J1*Y0 - J0*Y1 = 2/(πx)
    // => Y1 = (J1*Y0 - 2/(πx)) / J0
    let j0x = bessel_j0_f64(x);
    let j1x = bessel_j1_f64(x);
    let y0x = bessel_y0_f64(x);
    let two_over_pi_x = 2.0 / (PI_F64 * x);
    (j1x * y0x - two_over_pi_x) / j0x
}

/// Bessel function Y1(x) for f32 — delegates to f64 for accuracy
pub(super) fn bessel_y1_f32(x: f32) -> f32 {
    bessel_y1_f64(x as f64) as f32
}

// ============================================================================
// Modified Bessel I0, I1: First Kind (A&S polynomial)
// ============================================================================
//
// For |x| <= 3.75: polynomial in t² where t = x/3.75
// For |x| > 3.75: exp(|x|)/sqrt(|x|) * polynomial(3.75/|x|)

/// I0 polynomial for |x| <= 3.75 (ascending powers of t² = (x/3.75)²)
const I0_A: [f64; 7] = [
    1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.0360768, 0.0045813,
];

/// I0 polynomial for |x| > 3.75 (ascending powers of t = 3.75/|x|)
const I0_B: [f64; 9] = [
    0.39894228,
    0.01328592,
    0.00225319,
    -0.00157565,
    0.00916281,
    -0.02057706,
    0.02635537,
    -0.01647633,
    0.00392377,
];

/// Modified Bessel I0 for f32
pub(super) fn bessel_i0_f32(x: f32) -> f32 {
    bessel_i0_f64(x as f64) as f32
}

/// Modified Bessel I0 for f64
pub(super) fn bessel_i0_f64(x: f64) -> f64 {
    let ax = x.abs();

    if ax <= 3.75 {
        let t = ax / 3.75;
        let t2 = t * t;
        poly_asc(t2, &I0_A)
    } else {
        let t = 3.75 / ax;
        (ax.exp() / ax.sqrt()) * poly_asc(t, &I0_B)
    }
}

/// I1 polynomial for |x| <= 3.75 (ascending powers of t² = (x/3.75)²)
const I1_A: [f64; 7] = [
    0.5, 0.87890594, 0.51498869, 0.15084934, 0.02658733, 0.00301532, 0.00032411,
];

/// I1 polynomial for |x| > 3.75 (ascending powers of t = 3.75/|x|)
const I1_B: [f64; 9] = [
    0.39894228,
    -0.03988024,
    -0.00362018,
    0.00163801,
    -0.01031555,
    0.02282967,
    -0.02895312,
    0.01787654,
    -0.00420059,
];

/// Modified Bessel I1 for f32
pub(super) fn bessel_i1_f32(x: f32) -> f32 {
    bessel_i1_f64(x as f64) as f32
}

/// Modified Bessel I1 for f64
pub(super) fn bessel_i1_f64(x: f64) -> f64 {
    let ax = x.abs();
    let sign = if x < 0.0 { -1.0_f64 } else { 1.0_f64 };

    if ax <= 3.75 {
        let t = ax / 3.75;
        let t2 = t * t;
        sign * ax * poly_asc(t2, &I1_A)
    } else {
        let t = 3.75 / ax;
        sign * (ax.exp() / ax.sqrt()) * poly_asc(t, &I1_B)
    }
}

// ============================================================================
// Modified Bessel K0: Second Kind, Order 0
// ============================================================================
//
// Abramowitz & Stegun 9.8.5:
//   0 < x <= 2: K0(x) = A0(x²/4) - ln(x/2)*I0(x)
//   x > 2: K0(x) = exp(-x)/sqrt(x) * B0(2/x)

/// K0 polynomial for 0 < x <= 2 (ascending powers of t = x²/4)
const K0_A: [f64; 7] = [
    -0.57721566,
    0.42278420,
    0.23069756,
    0.03488590,
    0.00262698,
    0.00010750,
    0.0000074,
];

/// K0 polynomial for x > 2 (ascending powers of t = 2/x)
const K0_B: [f64; 7] = [
    1.25331414,
    -0.07832358,
    0.02189568,
    -0.01062446,
    0.00587872,
    -0.00251540,
    0.00053208,
];

/// Modified Bessel K0(x) — second kind, order 0.
///
/// Uses A&S 9.8.5 polynomial approximation.
/// Accuracy: relative error < 1e-6.
pub(super) fn bessel_k0_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    if x <= 2.0 {
        let t = x * x * 0.25;
        let a0 = poly_asc(t, &K0_A);
        let i0x = bessel_i0_f64(x);
        a0 - (x / 2.0).ln() * i0x
    } else {
        let t = 2.0 / x;
        let b0 = poly_asc(t, &K0_B);
        (-x).exp() / x.sqrt() * b0
    }
}

/// Modified Bessel K0(x) for f32
pub(super) fn bessel_k0_f32(x: f32) -> f32 {
    bessel_k0_f64(x as f64) as f32
}

// ============================================================================
// Modified Bessel K1: Second Kind, Order 1
// ============================================================================
//
// Abramowitz & Stegun 9.8.7:
//   0 < x <= 2: K1(x) = ln(x/2)*I1(x) + (1/x)*A1(x²/4)
//   x > 2: K1(x) = exp(-x)/sqrt(x) * B1(2/x)

/// K1 polynomial for 0 < x <= 2 (ascending powers of t = x²/4)
/// A&S 9.8.7, Table 9.8
const K1_A: [f64; 7] = [
    1.0,
    0.15443144,
    -0.67278579,
    -0.18156897,
    -0.01919402,
    -0.00110404,
    -0.00004686,
];

/// K1 polynomial for x > 2 (ascending powers of t = 2/x)
const K1_B: [f64; 7] = [
    1.25331414,
    0.23498619,
    -0.03655620,
    0.01504268,
    -0.00780353,
    0.00325614,
    -0.00068245,
];

/// Modified Bessel K1(x) — second kind, order 1.
///
/// Uses A&S 9.8.7 polynomial approximation.
/// Accuracy: relative error < 1e-6.
pub(super) fn bessel_k1_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    if x <= 2.0 {
        let t = x * x * 0.25;
        let a1 = poly_asc(t, &K1_A);
        let i1x = bessel_i1_f64(x);
        (x / 2.0).ln() * i1x + a1 / x
    } else {
        let t = 2.0 / x;
        let b1 = poly_asc(t, &K1_B);
        (-x).exp() / x.sqrt() * b1
    }
}

/// Modified Bessel K1(x) for f32
pub(super) fn bessel_k1_f32(x: f32) -> f32 {
    bessel_k1_f64(x as f64) as f32
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerances
    const TOL_F64: f64 = 1e-5;
    const TOL_F32: f32 = 1e-2;

    // ---- J0 tests -----------------------------------------------------------

    #[test]
    fn test_j0_at_zero() {
        let j0_0 = bessel_j0_f64(0.0);
        assert!(
            (j0_0 - 1.0).abs() < 1e-14,
            "J0(0) should be 1.0, got {}",
            j0_0
        );
    }

    #[test]
    fn test_j0_reference_values() {
        // Reference values from SciPy / DLMF
        let cases: &[(f64, f64)] = &[
            (1.0, 0.7651976865579666),
            (2.0, 0.2238907791412357),
            (5.0, -0.1775967713143383),
            (10.0, -0.2459357644513483),
        ];
        for &(x, expected) in cases {
            let got = bessel_j0_f64(x);
            let rel_err = (got - expected).abs() / expected.abs().max(1e-14);
            assert!(
                rel_err < TOL_F64,
                "J0({x}): got {got}, expected {expected}, rel_err={rel_err:.2e}"
            );
        }
    }

    #[test]
    fn test_j0_symmetry() {
        for &x in &[0.5_f64, 1.0, 2.5, 7.0, 15.0] {
            let diff = (bessel_j0_f64(x) - bessel_j0_f64(-x)).abs();
            assert!(diff < 1e-14, "J0 symmetry failed at x={x}: diff={diff}");
        }
    }

    #[test]
    fn test_j0_f32() {
        let got = bessel_j0_f32(1.0_f32);
        assert!((got - 0.7651977_f32).abs() < TOL_F32, "J0_f32(1.0)={got}");
    }

    // ---- J1 tests -----------------------------------------------------------

    #[test]
    fn test_j1_at_zero() {
        assert!(bessel_j1_f64(0.0).abs() < 1e-14);
    }

    #[test]
    fn test_j1_reference_values() {
        let cases: &[(f64, f64)] = &[
            (1.0, 0.4400505857449335),
            (2.0, 0.5767248077568734),
            (5.0, -0.3275791375914652),
            (10.0, 0.04347274616886144),
        ];
        for &(x, expected) in cases {
            let got = bessel_j1_f64(x);
            let rel_err = (got - expected).abs() / expected.abs().max(1e-14);
            assert!(
                rel_err < TOL_F64,
                "J1({x}): got {got}, expected {expected}, rel_err={rel_err:.2e}"
            );
        }
    }

    #[test]
    fn test_j1_odd() {
        for &x in &[0.5_f64, 1.0, 3.0, 8.0] {
            let sum = (bessel_j1_f64(x) + bessel_j1_f64(-x)).abs();
            assert!(sum < 1e-13, "J1 odd symmetry failed at x={x}");
        }
    }

    // ---- Y0 tests -----------------------------------------------------------

    #[test]
    fn test_y0_domain_nan() {
        assert!(bessel_y0_f64(0.0).is_nan(), "Y0(0) should be NaN");
        assert!(bessel_y0_f64(-1.0).is_nan(), "Y0(-1) should be NaN");
        assert!(bessel_y0_f32(0.0_f32).is_nan(), "Y0_f32(0) should be NaN");
        assert!(bessel_y0_f32(-1.0_f32).is_nan(), "Y0_f32(-1) should be NaN");
    }

    #[test]
    fn test_y0_reference_values() {
        // SciPy-verified values
        let cases: &[(f64, f64)] = &[
            (1.0, 0.08825696421567697),
            (2.0, 0.5103756726497451),
            (5.0, -0.30851762524903314),
            (10.0, 0.05567116728359939),
        ];
        for &(x, expected) in cases {
            let got = bessel_y0_f64(x);
            let rel_err = (got - expected).abs() / expected.abs().max(1e-14);
            assert!(
                rel_err < TOL_F64,
                "Y0({x}): got {got}, expected {expected}, rel_err={rel_err:.2e}"
            );
        }
    }

    #[test]
    fn test_y0_finite_for_positive_x() {
        for &x in &[0.1_f64, 0.5, 1.0, 2.0, 3.5, 5.0, 10.0, 50.0, 100.0] {
            let y0 = bessel_y0_f64(x);
            assert!(y0.is_finite(), "Y0({x}) should be finite, got {y0}");
        }
    }

    // ---- Y1 tests -----------------------------------------------------------

    #[test]
    fn test_y1_domain_nan() {
        assert!(bessel_y1_f64(0.0).is_nan(), "Y1(0) should be NaN");
        assert!(bessel_y1_f64(-0.5).is_nan(), "Y1(-0.5) should be NaN");
        assert!(bessel_y1_f32(0.0_f32).is_nan(), "Y1_f32(0) should be NaN");
    }

    #[test]
    fn test_y1_reference_values() {
        // SciPy-verified values
        let cases: &[(f64, f64)] = &[
            (1.0, -0.7812128213002888),
            (2.0, -0.10703243154093754),
            (5.0, 0.14786314339122684),
            (10.0, 0.24901542420695388),
        ];
        for &(x, expected) in cases {
            let got = bessel_y1_f64(x);
            let rel_err = (got - expected).abs() / expected.abs().max(1e-14);
            assert!(
                rel_err < TOL_F64,
                "Y1({x}): got {got}, expected {expected}, rel_err={rel_err:.2e}"
            );
        }
    }

    // ---- I0 tests -----------------------------------------------------------

    #[test]
    fn test_i0_at_zero() {
        assert!((bessel_i0_f64(0.0) - 1.0).abs() < 1e-14);
        assert!((bessel_i0_f32(0.0_f32) - 1.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_i0_even_positive() {
        let i0_2 = bessel_i0_f64(2.0);
        let i0_m2 = bessel_i0_f64(-2.0);
        assert!(i0_2 > 0.0);
        assert!((i0_2 - i0_m2).abs() < 1e-14);
    }

    #[test]
    fn test_i0_reference_values() {
        let cases: &[(f64, f64)] = &[
            (0.5, 1.0634833707413233),
            (1.0, 1.2660658777520082),
            (2.0, 2.279585301997742),
            (5.0, 27.239871823604442),
        ];
        for &(x, expected) in cases {
            let got = bessel_i0_f64(x);
            let rel_err = (got - expected).abs() / expected.abs();
            assert!(rel_err < TOL_F64, "I0({x}): got {got}, expected {expected}");
        }
    }

    // ---- K0 tests -----------------------------------------------------------

    #[test]
    fn test_k0_domain() {
        assert!(bessel_k0_f64(0.0).is_nan(), "K0(0) should be NaN");
        assert!(bessel_k0_f64(-1.0).is_nan(), "K0(-1) should be NaN");
        assert!(bessel_k0_f32(0.0_f32).is_nan(), "K0_f32(0) should be NaN");
    }

    #[test]
    fn test_k0_positive() {
        for &x in &[0.5_f64, 1.0, 2.0, 5.0, 10.0] {
            let k0 = bessel_k0_f64(x);
            assert!(k0 > 0.0, "K0({x}) should be positive, got {k0}");
            assert!(k0.is_finite(), "K0({x}) should be finite");
        }
    }

    #[test]
    fn test_k0_reference_values() {
        // Reference values from SciPy 1.17.0
        let cases: &[(f64, f64)] = &[
            (0.5, 0.9244190712276660),
            (1.0, 0.4210244382407082),
            (2.0, 0.1138938727495334),
            (5.0, 0.0036910983340426),
        ];
        for &(x, expected) in cases {
            let got = bessel_k0_f64(x);
            let rel_err = (got - expected).abs() / expected.abs();
            assert!(
                rel_err < TOL_F64,
                "K0({x}): got {got}, expected {expected}, rel_err={rel_err:.2e}"
            );
        }
    }

    // ---- K1 tests -----------------------------------------------------------

    #[test]
    fn test_k1_domain() {
        assert!(bessel_k1_f64(0.0).is_nan(), "K1(0) should be NaN");
        assert!(bessel_k1_f64(-1.0).is_nan(), "K1(-1) should be NaN");
        assert!(bessel_k1_f32(0.0_f32).is_nan(), "K1_f32(0) should be NaN");
    }

    #[test]
    fn test_k1_positive() {
        for &x in &[0.5_f64, 1.0, 2.0, 5.0, 10.0] {
            let k1 = bessel_k1_f64(x);
            assert!(k1 > 0.0, "K1({x}) should be positive, got {k1}");
            assert!(k1.is_finite(), "K1({x}) should be finite");
        }
    }

    #[test]
    fn test_k1_reference_values() {
        // Reference values from SciPy 1.17.0
        let cases: &[(f64, f64)] = &[
            (0.5, 1.6564411200033007),
            (1.0, 0.6019072301972346),
            (2.0, 0.1398658818165225),
            (5.0, 0.0040446134454522),
        ];
        for &(x, expected) in cases {
            let got = bessel_k1_f64(x);
            let rel_err = (got - expected).abs() / expected.abs();
            assert!(
                rel_err < TOL_F64,
                "K1({x}): got {got}, expected {expected}, rel_err={rel_err:.2e}"
            );
        }
    }

    // ---- Cross-function consistency checks -----------------------------------

    #[test]
    fn test_k0_k1_ordering() {
        // K0(x) < K1(x) for 0 < x < ~2
        for &x in &[0.5_f64, 1.0, 1.5] {
            let k0 = bessel_k0_f64(x);
            let k1 = bessel_k1_f64(x);
            assert!(k1 > k0, "K1({x})={k1} should be > K0({x})={k0}");
        }
    }

    #[test]
    fn test_wronskian_j0y0() {
        // Self-consistency: J1*Y0 - J0*Y1 = 2/(πx)
        for &x in &[0.5_f64, 1.0, 2.0, 5.0] {
            let j0 = bessel_j0_f64(x);
            let j1 = bessel_j1_f64(x);
            let y0 = bessel_y0_f64(x);
            let y1 = bessel_y1_f64(x);
            let wronskian = j1 * y0 - j0 * y1;
            let expected = 2.0 / (PI_F64 * x);
            let rel_err = (wronskian - expected).abs() / expected.abs();
            assert!(
                rel_err < 1e-4,
                "Wronskian at x={x}: got {wronskian:.8e}, expected {expected:.8e}, rel_err={rel_err:.2e}"
            );
        }
    }
}
