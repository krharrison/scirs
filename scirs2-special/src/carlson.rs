//! Carlson elliptic integrals
//!
//! This module implements the Carlson symmetric standard forms of elliptic integrals.
//! These are the modern canonical forms recommended for numerical computation.
//!
//! ## Mathematical Theory
//!
//! The Carlson elliptic integrals are defined as:
//!
//! ### RC(x, y) - Degenerate case
//! ```text
//! RC(x, y) = (1/2) integral_0^inf dt / [sqrt(t+x) * (t+y)]
//! ```
//!
//! ### RD(x, y, z) - Symmetric elliptic integral of the second kind
//! ```text
//! RD(x, y, z) = (3/2) integral_0^inf dt / [sqrt(t+x) * sqrt(t+y) * (t+z)^{3/2}]
//! ```
//!
//! ### RF(x, y, z) - Symmetric elliptic integral of the first kind
//! ```text
//! RF(x, y, z) = (1/2) integral_0^inf dt / [sqrt(t+x) * sqrt(t+y) * sqrt(t+z)]
//! ```
//!
//! ### RG(x, y, z) - Symmetric elliptic integral
//! ```text
//! RG(x, y, z) = (1/4) integral_0^inf t * dt / [sqrt(t+x) * sqrt(t+y) * sqrt(t+z)]
//! ```
//!
//! ### RJ(x, y, z, p) - Symmetric elliptic integral of the third kind
//! ```text
//! RJ(x, y, z, p) = (3/2) integral_0^inf dt / [sqrt(t+x) * sqrt(t+y) * sqrt(t+z) * (t+p)]
//! ```
//!
//! ## Properties
//! 1. **Symmetry**: The integrals are symmetric in their first arguments
//! 2. **Homogeneity**: RF(lx, ly, lz) = RF(x, y, z) / sqrt(l)
//! 3. **Reduction**: Classical elliptic integrals can be expressed in terms of Carlson forms
//! 4. **Numerical stability**: Duplication algorithm provides stable computation
//!
//! ## References
//!
//! - Carlson, B. C. (1995). "Numerical computation of real or complex elliptic integrals."
//!   *Numerical Algorithms*, 10(1), 13--26.
//! - DLMF Chapter 19: Elliptic Integrals.

use crate::{SpecialError, SpecialResult};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::validation::check_finite;
use std::fmt::{Debug, Display};

/// Maximum number of iterations for convergence
const MAX_ITERATIONS: usize = 100;

/// Convergence tolerance
const TOLERANCE: f64 = 1e-15;

/// Carlson elliptic integral RC(x, y)
///
/// Computes the degenerate Carlson elliptic integral RC(x, y)
/// using the duplication algorithm.
///
/// # Arguments
/// * `x` - First parameter (x >= 0)
/// * `y` - Second parameter (y != 0)
///
/// # Returns
/// Value of RC(x, y)
///
/// # Examples
/// ```
/// use scirs2_special::elliprc;
///
/// // RC(0, 1) = pi/2
/// let result = elliprc(0.0, 1.0).expect("Test/example failed");
/// assert!((result - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
///
/// // RC(1, 1) = 1
/// let result = elliprc(1.0, 1.0).expect("Test/example failed");
/// assert!((result - 1.0f64).abs() < 1e-10);
/// ```
pub fn elliprc<T>(x: T, y: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    check_finite(x, "x value")?;
    check_finite(y, "y value")?;

    let zero = T::zero();
    let one = T::one();
    let two = one + one;
    let three = two + one;
    let four = two + two;

    if x < zero {
        return Err(SpecialError::DomainError(
            "x must be non-negative".to_string(),
        ));
    }

    if y == zero {
        return Err(SpecialError::DomainError("y cannot be zero".to_string()));
    }

    // Handle special case x == 0
    if x == zero {
        if y > zero {
            let pi_half = T::from_f64(std::f64::consts::FRAC_PI_2).unwrap_or(one);
            return Ok(pi_half / y.sqrt());
        } else {
            let pi_half = T::from_f64(std::f64::consts::FRAC_PI_2).unwrap_or(one);
            return Ok(pi_half / (-y).sqrt());
        }
    }

    if x == y {
        return Ok(one / x.sqrt());
    }

    // Duplication algorithm for RC(x, y)
    // RC(x, y) = RF(x, y, y) but we use the specialized duplication:
    //   lambda = 2*sqrt(x)*sqrt(y) + y
    //   x <- (x + lambda) / 4, y <- (y + lambda) / 4
    //   Converge to A = (x + 2y) / 3, then series expansion.
    let tol = T::from_f64(TOLERANCE).unwrap_or_else(|| T::epsilon());
    let mut xt = x;
    let mut yt = y;

    for _ in 0..MAX_ITERATIONS {
        let lambda = two * xt.sqrt() * yt.sqrt() + yt;
        xt = (xt + lambda) / four;
        yt = (yt + lambda) / four;

        let a = (xt + two * yt) / three;
        let s = (yt - a) / a;

        if s.abs() < tol {
            // Series expansion for RC:
            // RC = (1/sqrt(a)) * (1 + s^2*(3/10 + s*(1/7 + s*(3/8 + s*9/22))))
            let c1 = T::from_f64(3.0 / 10.0).unwrap_or(zero);
            let c2 = T::from_f64(1.0 / 7.0).unwrap_or(zero);
            let c3 = T::from_f64(3.0 / 8.0).unwrap_or(zero);
            let c4 = T::from_f64(9.0 / 22.0).unwrap_or(zero);

            let series = one + s * s * (c1 + s * (c2 + s * (c3 + s * c4)));
            return Ok(series / a.sqrt());
        }
    }

    // If we reached max iterations, compute best estimate
    let a = (xt + two * yt) / three;
    Ok(one / a.sqrt())
}

/// Carlson elliptic integral RF(x, y, z)
///
/// Computes the symmetric elliptic integral of the first kind RF(x, y, z)
/// using the duplication algorithm.
///
/// # Arguments
/// * `x` - First parameter (x >= 0)
/// * `y` - Second parameter (y >= 0)
/// * `z` - Third parameter (z >= 0)
///
/// # Returns
/// Value of RF(x, y, z)
///
/// # Examples
/// ```
/// use scirs2_special::elliprf;
///
/// // RF(0, 1, 1) = pi/2
/// let result = elliprf(0.0, 1.0, 1.0).expect("Test/example failed");
/// assert!((result - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
///
/// // Symmetry test
/// let x = 2.0f64;
/// let y = 3.0f64;
/// let z = 4.0f64;
/// let rf1 = elliprf(x, y, z).expect("Test/example failed");
/// let rf2 = elliprf(y, z, x).expect("Test/example failed");
/// assert!((rf1 - rf2).abs() < 1e-12);
/// ```
pub fn elliprf<T>(x: T, y: T, z: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    check_finite(x, "x value")?;
    check_finite(y, "y value")?;
    check_finite(z, "z value")?;

    let zero = T::zero();
    let one = T::one();
    let three = one + one + one;
    let four = three + one;
    let tol = T::from_f64(TOLERANCE).unwrap_or_else(|| T::epsilon());

    if x < zero || y < zero || z < zero {
        return Err(SpecialError::DomainError(
            "All arguments must be non-negative".to_string(),
        ));
    }

    // At most one argument can be zero
    let zero_count = (if x == zero { 1 } else { 0 })
        + (if y == zero { 1 } else { 0 })
        + (if z == zero { 1 } else { 0 });

    if zero_count > 1 {
        return Err(SpecialError::DomainError(
            "At most one argument can be zero".to_string(),
        ));
    }

    // Duplication algorithm (Carlson 1995)
    let mut xt = x;
    let mut yt = y;
    let mut zt = z;

    for _ in 0..MAX_ITERATIONS {
        let sqrt_x = xt.sqrt();
        let sqrt_y = yt.sqrt();
        let sqrt_z = zt.sqrt();

        let lambda = sqrt_x * sqrt_y + sqrt_y * sqrt_z + sqrt_z * sqrt_x;

        xt = (xt + lambda) / four;
        yt = (yt + lambda) / four;
        zt = (zt + lambda) / four;

        let a = (xt + yt + zt) / three;
        let dx = (one - xt / a).abs();
        let dy = (one - yt / a).abs();
        let dz = (one - zt / a).abs();

        let max_diff = dx.max(dy).max(dz);
        if max_diff < tol {
            break;
        }
    }

    let a = (xt + yt + zt) / three;
    let x_dev = (a - xt) / a;
    let y_dev = (a - yt) / a;
    let z_dev = (a - zt) / a;

    let e2 = x_dev * y_dev - z_dev * z_dev;
    let e3 = x_dev * y_dev * z_dev;

    // Series expansion coefficients (Carlson 1995, Table 1)
    let c1 = T::from_f64(-1.0 / 10.0).unwrap_or(zero);
    let c2 = T::from_f64(1.0 / 14.0).unwrap_or(zero);
    let c3 = T::from_f64(1.0 / 24.0).unwrap_or(zero);
    let c4 = T::from_f64(-3.0 / 44.0).unwrap_or(zero);

    let series = one + c1 * e2 + c2 * e3 + c3 * e2 * e2 + c4 * e2 * e3;

    Ok(series / a.sqrt())
}

/// Carlson elliptic integral RD(x, y, z)
///
/// Computes the symmetric elliptic integral of the second kind RD(x, y, z)
/// using the duplication algorithm.
///
/// # Arguments
/// * `x` - First parameter (x >= 0)
/// * `y` - Second parameter (y >= 0)
/// * `z` - Third parameter (z > 0)
///
/// # Examples
/// ```
/// use scirs2_special::elliprd;
///
/// // RD(0, 2, 1) ~ 1.7972 (SciPy reference)
/// let result = elliprd(0.0_f64, 2.0, 1.0).expect("Test/example failed");
/// assert!((result - 1.7972103521033884_f64).abs() < 1e-10);
/// ```
pub fn elliprd<T>(x: T, y: T, z: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    check_finite(x, "x value")?;
    check_finite(y, "y value")?;
    check_finite(z, "z value")?;

    let zero = T::zero();
    let one = T::one();
    let three = one + one + one;
    let four = three + one;
    let five = four + one;
    let tol = T::from_f64(TOLERANCE).unwrap_or_else(|| T::epsilon());

    if x < zero || y < zero || z <= zero {
        return Err(SpecialError::DomainError(
            "x, y must be non-negative and z must be positive".to_string(),
        ));
    }

    // At most one of x, y can be zero
    if x == zero && y == zero {
        return Err(SpecialError::DomainError(
            "At most one of x, y can be zero".to_string(),
        ));
    }

    // Duplication algorithm (Carlson 1995)
    let mut xt = x;
    let mut yt = y;
    let mut zt = z;
    let mut sum = zero;
    let mut power4 = one;

    for _ in 0..MAX_ITERATIONS {
        let sqrt_x = xt.sqrt();
        let sqrt_y = yt.sqrt();
        let sqrt_z = zt.sqrt();

        let lambda = sqrt_x * sqrt_y + sqrt_y * sqrt_z + sqrt_z * sqrt_x;

        sum = sum + power4 / (sqrt_z * (zt + lambda));

        xt = (xt + lambda) / four;
        yt = (yt + lambda) / four;
        zt = (zt + lambda) / four;
        power4 = power4 / four;

        let a = (xt + yt + three * zt) / five;
        let dx = (one - xt / a).abs();
        let dy = (one - yt / a).abs();
        let dz = (one - zt / a).abs();

        let max_diff = dx.max(dy).max(dz);
        if max_diff < tol {
            break;
        }
    }

    let a = (xt + yt + three * zt) / five;
    let x_dev = (a - xt) / a;
    let y_dev = (a - yt) / a;
    let z_dev = (a - zt) / a;

    // Series expansion for RD (Carlson 1979, 1995)
    // Using the standard E2/E3 formulation from Carlson
    let six = three + three;
    let e2 = x_dev * y_dev + six * z_dev * z_dev - three * z_dev * (x_dev + y_dev);
    let e3 = x_dev * y_dev * z_dev;

    let c1 = T::from_f64(-3.0 / 14.0).unwrap_or(zero);
    let c2 = T::from_f64(1.0 / 6.0).unwrap_or(zero);
    let c3 = T::from_f64(9.0 / 88.0).unwrap_or(zero);
    let c4 = T::from_f64(-3.0 / 22.0).unwrap_or(zero);

    let series = one + c1 * e2 + c2 * e3 + c3 * e2 * e2 + c4 * e2 * e3;

    Ok(three * sum + power4 * series / (a * a.sqrt()))
}

/// Carlson elliptic integral RG(x, y, z)
///
/// Computes the symmetric elliptic integral RG(x, y, z).
///
/// # Arguments
/// * `x` - First parameter (x >= 0)
/// * `y` - Second parameter (y >= 0)
/// * `z` - Third parameter (z >= 0)
///
/// # Mathematical Definition
/// RG(x,y,z) = [RF(x,y,z)*(x+y+z) - RD(x,y,z)*z - RD(y,z,x)*x - RD(z,x,y)*y] / 4
///
/// # Examples
/// ```
/// use scirs2_special::elliprg;
///
/// // Test basic functionality
/// let result = elliprg(1.0, 2.0, 3.0).expect("Test/example failed");
/// assert!(result > 0.0);
/// ```
pub fn elliprg<T>(x: T, y: T, z: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    check_finite(x, "x value")?;
    check_finite(y, "y value")?;
    check_finite(z, "z value")?;

    let zero = T::zero();
    let two = T::one() + T::one();
    let four = two + two;

    if x < zero || y < zero || z < zero {
        return Err(SpecialError::DomainError(
            "All arguments must be non-negative".to_string(),
        ));
    }

    // Handle special cases where arguments are zero
    let zero_count = (if x == zero { 1 } else { 0 })
        + (if y == zero { 1 } else { 0 })
        + (if z == zero { 1 } else { 0 });

    if zero_count >= 2 {
        return Ok(zero);
    }

    if zero_count == 1 {
        // One argument is zero - use simplified formula
        if x == zero {
            return Ok((y * z).sqrt() * elliprf(zero, y, z)? / two);
        } else if y == zero {
            return Ok((x * z).sqrt() * elliprf(x, zero, z)? / two);
        } else {
            return Ok((x * y).sqrt() * elliprf(x, y, zero)? / two);
        }
    }

    // General case: RG(x,y,z) = [RF(x,y,z)*(x+y+z) - RD(x,y,z)*z - RD(y,z,x)*x - RD(z,x,y)*y] / 4
    let rf_val = elliprf(x, y, z)?;
    let rd_xyz = elliprd(x, y, z)?;
    let rd_yzx = elliprd(y, z, x)?;
    let rd_zxy = elliprd(z, x, y)?;

    let sum = x + y + z;
    let rg = (rf_val * sum - rd_xyz * z - rd_yzx * x - rd_zxy * y) / four;

    Ok(rg)
}

/// Carlson elliptic integral RJ(x, y, z, p)
///
/// Computes the symmetric elliptic integral of the third kind RJ(x, y, z, p)
/// using the duplication algorithm.
///
/// # Arguments
/// * `x` - First parameter (x >= 0)
/// * `y` - Second parameter (y >= 0)
/// * `z` - Third parameter (z >= 0)
/// * `p` - Fourth parameter (p > 0)
///
/// # Examples
/// ```
/// use scirs2_special::elliprj;
///
/// // Test basic functionality
/// let result = elliprj(1.0, 2.0, 3.0, 4.0).expect("Test/example failed");
/// assert!(result > 0.0);
/// ```
pub fn elliprj<T>(x: T, y: T, z: T, p: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    check_finite(x, "x value")?;
    check_finite(y, "y value")?;
    check_finite(z, "z value")?;
    check_finite(p, "p value")?;

    let zero = T::zero();
    let one = T::one();
    let two = one + one;
    let three = two + one;
    let four = two + two;
    let five = four + one;
    let tol = T::from_f64(TOLERANCE).unwrap_or_else(|| T::epsilon());

    if x < zero || y < zero || z < zero {
        return Err(SpecialError::DomainError(
            "x, y, z must be non-negative".to_string(),
        ));
    }

    if p == zero {
        return Err(SpecialError::DomainError("p cannot be zero".to_string()));
    }

    // At most one of x, y, z can be zero
    let zero_count = (if x == zero { 1 } else { 0 })
        + (if y == zero { 1 } else { 0 })
        + (if z == zero { 1 } else { 0 });

    if zero_count > 1 {
        return Err(SpecialError::DomainError(
            "At most one of x, y, z can be zero".to_string(),
        ));
    }

    // Duplication algorithm following Boost/Carlson 1995
    // Reference: Boost ellint_rj_imp_final
    let six = three + three;

    let mut xn = x;
    let mut yn = y;
    let mut zn = z;
    let mut pn = p;
    let a0 = (x + y + z + two * p) / five;
    let mut an = a0;
    let mut delta = (p - x) * (p - y) * (p - z);
    let mut fmn = one; // 4^{-n}
    let mut rc_sum = zero;

    for _ in 0..MAX_ITERATIONS {
        let rx = xn.sqrt();
        let ry = yn.sqrt();
        let rz = zn.sqrt();
        let rp = pn.sqrt();

        let dn = (rp + rx) * (rp + ry) * (rp + rz);
        let en = delta / (dn * dn);

        // RC(1, 1+En) contribution
        // When En is small, RC(1, 1+En) can be computed as elliprc(1, 1+en)
        let one_plus_en = one + en;
        if one_plus_en > zero {
            rc_sum = rc_sum + fmn / dn * elliprc(one, one_plus_en)?;
        } else {
            // Fallback for En near -1: use the substitution formula
            // 1+En = 2*rp*(pn + rx*(ry+rz) + ry*rz) / Dn
            let b = two * rp * (pn + rx * (ry + rz) + ry * rz) / dn;
            rc_sum = rc_sum + fmn / dn * elliprc(one, b)?;
        }

        let lambda = rx * ry + rx * rz + ry * rz;

        // Move to n+1
        an = (an + lambda) / four;
        fmn = fmn / four;

        let sixty_four = T::from_f64(64.0).unwrap_or_else(|| four * four * four);

        xn = (xn + lambda) / four;
        yn = (yn + lambda) / four;
        zn = (zn + lambda) / four;
        pn = (pn + lambda) / four;
        delta = delta / sixty_four;

        // Convergence check: all variables close to An
        let dx = (one - xn / an).abs();
        let dy = (one - yn / an).abs();
        let dz = (one - zn / an).abs();
        let dp = (one - pn / an).abs();
        if dx.max(dy).max(dz).max(dp) < tol {
            break;
        }
    }

    // Final series expansion (Boost formulation)
    // X = fmn * (A0 - x) / An, etc.
    let xd = fmn * (a0 - x) / an;
    let yd = fmn * (a0 - y) / an;
    let zd = fmn * (a0 - z) / an;
    let pd = (zero - xd - yd - zd) / two;

    let e2 = xd * yd + xd * zd + yd * zd - three * pd * pd;
    let e3 = xd * yd * zd + two * e2 * pd + four * pd * pd * pd;
    let e4 = (two * xd * yd * zd + e2 * pd + three * pd * pd * pd) * pd;
    let e5 = xd * yd * zd * pd * pd;

    let c_3_14 = T::from_f64(3.0 / 14.0).unwrap_or(zero);
    let c_1_6 = T::from_f64(1.0 / 6.0).unwrap_or(zero);
    let c_9_88 = T::from_f64(9.0 / 88.0).unwrap_or(zero);
    let c_3_22 = T::from_f64(3.0 / 22.0).unwrap_or(zero);
    let c_9_52 = T::from_f64(9.0 / 52.0).unwrap_or(zero);
    let c_3_26 = T::from_f64(3.0 / 26.0).unwrap_or(zero);
    let c_1_16 = T::from_f64(1.0 / 16.0).unwrap_or(zero);
    let c_3_40 = T::from_f64(3.0 / 40.0).unwrap_or(zero);
    let c_3_20 = T::from_f64(3.0 / 20.0).unwrap_or(zero);
    let c_45_272 = T::from_f64(45.0 / 272.0).unwrap_or(zero);
    let c_9_68 = T::from_f64(9.0 / 68.0).unwrap_or(zero);

    let series = one - c_3_14 * e2 + c_1_6 * e3 + c_9_88 * e2 * e2 - c_3_22 * e4 - c_9_52 * e2 * e3
        + c_3_26 * e5
        - c_1_16 * e2 * e2 * e2
        + c_3_40 * e3 * e3
        + c_3_20 * e2 * e4
        + c_45_272 * e2 * e2 * e3
        - c_9_68 * (e3 * e4 + e2 * e5);

    let result = fmn * series / (an * an.sqrt());
    Ok(result + six * rc_sum)
}

/// Array versions of Carlson elliptic integrals
pub fn elliprf_array<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    z: &ArrayView1<T>,
) -> SpecialResult<Array1<T>>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    if x.len() != y.len() || y.len() != z.len() {
        return Err(SpecialError::DomainError(
            "Arrays must have same length".to_string(),
        ));
    }

    let mut result = Array1::zeros(x.len());
    for (i, ((&xi, &yi), &zi)) in x.iter().zip(y.iter()).zip(z.iter()).enumerate() {
        result[i] = elliprf(xi, yi, zi)?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::{FRAC_PI_2, PI, SQRT_2};

    // ===== RC tests =====

    #[test]
    fn test_elliprc_special_cases() {
        // RC(0, 1) = pi/2
        let result = elliprc(0.0, 1.0).expect("RC(0,1) failed");
        assert_relative_eq!(result, FRAC_PI_2, epsilon = 1e-12);

        // RC(1, 1) = 1
        let result = elliprc(1.0, 1.0).expect("RC(1,1) failed");
        assert_relative_eq!(result, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_elliprc_general_values() {
        // RC(x,y) for y > x > 0: RC(x,y) = arctan(sqrt((y-x)/x)) / sqrt(y-x)
        // RC(1, 2) = arctan(1) / 1 = pi/4
        let result = elliprc(1.0, 2.0).expect("RC(1,2) failed");
        assert_relative_eq!(result, std::f64::consts::FRAC_PI_4, epsilon = 1e-10);

        // RC(2, 1): for x > y > 0, RC(x,y) = atanh(sqrt((x-y)/x)) / sqrt(x-y)
        // RC(2, 1) = atanh(1/sqrt(2)) / 1
        let expected = (1.0 / SQRT_2).atanh();
        let result = elliprc(2.0, 1.0).expect("RC(2,1) failed");
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_elliprc_homogeneity() {
        // RC(lx, ly) = RC(x, y) / sqrt(l)
        let x = 2.0_f64;
        let y = 3.0_f64;
        let lambda = 4.0_f64;
        let rc1 = elliprc(x, y).expect("failed");
        let rc2 = elliprc(lambda * x, lambda * y).expect("failed");
        assert_relative_eq!(rc2, rc1 / lambda.sqrt(), epsilon = 1e-10);
    }

    // ===== RF tests =====

    #[test]
    fn test_elliprf_special_cases() {
        // RF(0, 1, 1) = pi/2
        let result = elliprf(0.0, 1.0, 1.0).expect("RF(0,1,1) failed");
        assert_relative_eq!(result, FRAC_PI_2, epsilon = 1e-12);

        // RF(1, 1, 1) = 1
        let result = elliprf(1.0, 1.0, 1.0).expect("RF(1,1,1) failed");
        assert_relative_eq!(result, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_elliprf_symmetry() {
        let rf1 = elliprf(1.0, 2.0, 3.0).expect("failed");
        let rf2 = elliprf(2.0, 3.0, 1.0).expect("failed");
        let rf3 = elliprf(3.0, 1.0, 2.0).expect("failed");
        assert_relative_eq!(rf1, rf2, epsilon = 1e-12);
        assert_relative_eq!(rf2, rf3, epsilon = 1e-12);
    }

    #[test]
    fn test_elliprf_homogeneity() {
        // RF(lx, ly, lz) = RF(x, y, z) / sqrt(l)
        let x = 1.0_f64;
        let y = 2.0_f64;
        let z = 3.0_f64;
        let lambda = 9.0_f64;
        let rf1 = elliprf(x, y, z).expect("failed");
        let rf2 = elliprf(lambda * x, lambda * y, lambda * z).expect("failed");
        assert_relative_eq!(rf2, rf1 / lambda.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_elliprf_complete_k() {
        // K(m) = RF(0, 1-m, 1)
        // K(0) = pi/2
        let k0 = elliprf(0.0, 1.0, 1.0).expect("K(0) failed");
        assert_relative_eq!(k0, FRAC_PI_2, epsilon = 1e-12);

        // K(0.5) ~ 1.854074677301372
        let k_half = elliprf(0.0, 0.5, 1.0).expect("K(0.5) failed");
        assert_relative_eq!(k_half, 1.854_074_677_301_37, epsilon = 1e-10);
    }

    // ===== RD tests =====

    #[test]
    fn test_elliprd_special_cases() {
        // RD(0, 2, 1) ~ 1.7972103521033884 (SciPy reference value)
        let result = elliprd(0.0, 2.0, 1.0).expect("RD(0,2,1) failed");
        assert_relative_eq!(result, 1.7972103521033884, epsilon = 1e-10);

        // RD(1, 1, 1) = 1 (trivial case: all arguments equal)
        let result = elliprd(1.0, 1.0, 1.0).expect("RD(1,1,1) failed");
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_elliprd_symmetry_xy() {
        // RD is symmetric in first two arguments
        let rd1 = elliprd(1.0, 2.0, 3.0).expect("failed");
        let rd2 = elliprd(2.0, 1.0, 3.0).expect("failed");
        assert_relative_eq!(rd1, rd2, epsilon = 1e-10);
    }

    #[test]
    fn test_elliprd_complete_e_relation() {
        // E(m) = RF(0, 1-m, 1) - (m/3) * RD(0, 1-m, 1)
        let m = 0.5_f64;
        let rf_val = elliprf(0.0, 1.0 - m, 1.0).expect("failed");
        let rd_val = elliprd(0.0, 1.0 - m, 1.0).expect("failed");
        let e_val = rf_val - (m / 3.0) * rd_val;
        // E(0.5) ~ 1.350643881047675
        assert_relative_eq!(e_val, 1.350_643_881_047_67, epsilon = 1e-8);
    }

    // ===== RG tests =====

    #[test]
    fn test_elliprg_symmetry() {
        let rg1 = elliprg(1.0, 2.0, 3.0).expect("failed");
        let rg2 = elliprg(2.0, 3.0, 1.0).expect("failed");
        let rg3 = elliprg(3.0, 1.0, 2.0).expect("failed");
        assert_relative_eq!(rg1, rg2, epsilon = 1e-8);
        assert_relative_eq!(rg2, rg3, epsilon = 1e-8);
    }

    #[test]
    fn test_elliprg_basic() {
        let rg = elliprg(1.0, 2.0, 3.0).expect("RG(1,2,3) failed");
        assert!(rg > 0.0 && rg.is_finite());
    }

    // ===== RJ tests =====

    #[test]
    fn test_elliprj_basic() {
        let rj = elliprj(1.0, 2.0, 3.0, 4.0).expect("RJ(1,2,3,4) failed");
        assert!(rj > 0.0 && rj.is_finite());
    }

    #[test]
    fn test_elliprj_symmetry() {
        // RJ is symmetric in first three arguments
        let rj1 = elliprj(1.0, 2.0, 3.0, 4.0).expect("failed");
        let rj2 = elliprj(2.0, 3.0, 1.0, 4.0).expect("failed");
        let rj3 = elliprj(3.0, 1.0, 2.0, 4.0).expect("failed");
        assert_relative_eq!(rj1, rj2, epsilon = 1e-8);
        assert_relative_eq!(rj2, rj3, epsilon = 1e-8);
    }

    #[test]
    fn test_elliprj_rd_relation() {
        // RD(x, y, z) = RJ(x, y, z, z)
        let x = 1.0_f64;
        let y = 2.0_f64;
        let z = 3.0_f64;
        let rd = elliprd(x, y, z).expect("RD failed");
        let rj = elliprj(x, y, z, z).expect("RJ failed");
        assert_relative_eq!(rd, rj, epsilon = 1e-8);
    }

    // ===== Error condition tests =====

    #[test]
    fn test_carlson_error_conditions() {
        // Negative arguments
        assert!(elliprf(-1.0, 1.0, 1.0).is_err());
        assert!(elliprd(-1.0, 1.0, 1.0).is_err());
        assert!(elliprg(-1.0, 1.0, 1.0).is_err());

        // Zero conditions
        assert!(elliprc(1.0, 0.0).is_err());
        assert!(elliprf(0.0, 0.0, 1.0).is_err());
        assert!(elliprd(0.0, 0.0, 1.0).is_err());
        assert!(elliprj(1.0, 2.0, 3.0, 0.0).is_err());
    }

    #[test]
    fn test_carlson_basic_functionality() {
        let rc = elliprc(1.0, 2.0).expect("RC failed");
        assert!(rc > 0.0 && rc.is_finite());

        let rf = elliprf(1.0, 2.0, 3.0).expect("RF failed");
        assert!(rf > 0.0 && rf.is_finite());

        let rd = elliprd(1.0, 2.0, 3.0).expect("RD failed");
        assert!(rd > 0.0 && rd.is_finite());

        let rg = elliprg(1.0, 2.0, 3.0).expect("RG failed");
        assert!(rg > 0.0 && rg.is_finite());

        let rj = elliprj(1.0, 2.0, 3.0, 4.0).expect("RJ failed");
        assert!(rj > 0.0 && rj.is_finite());
    }
}
