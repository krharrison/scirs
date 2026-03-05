//! Extended Airy function interface
//!
//! Provides ergonomic f64-specialised wrappers matching the API requested by
//! the task specification:
//!
//! * `airy_ai(x)` / `airy_bi(x)` – function values
//! * `airy_ai_prime(x)` / `airy_bi_prime(x)` – derivatives
//! * `airy_ai_zeros(n)` / `airy_bi_zeros(n)` – first n zeros as `Vec<f64>`
//!
//! All functions use the high-quality implementations already present in
//! `crate::airy`.

use crate::error::{SpecialError, SpecialResult};

/// Airy function of the first kind, Ai(x).
///
/// Ai(x) is the unique solution to y'' = x·y that decays to zero as x → +∞.
///
/// # Arguments
///
/// * `x` – Real argument
///
/// # Returns
///
/// * `f64` – value of Ai(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::airy_ai;
///
/// // Ai(0) ≈ 0.35502805388781724
/// let val = airy_ai(0.0);
/// assert!((val - 0.3550280538878172).abs() < 1e-10);
///
/// // Ai(1) ≈ 0.13529241631288141
/// let val1 = airy_ai(1.0);
/// assert!((val1 - 0.1352924163128814).abs() < 1e-10);
/// ```
#[inline]
pub fn airy_ai(x: f64) -> f64 {
    crate::airy::ai(x)
}

/// Airy function of the second kind, Bi(x).
///
/// Bi(x) is the unique (up to normalisation) solution to y'' = x·y that
/// grows as x → +∞.
///
/// # Arguments
///
/// * `x` – Real argument
///
/// # Returns
///
/// * `f64` – value of Bi(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::airy_bi;
///
/// // Bi(0) ≈ 0.6149266274460007
/// let val = airy_bi(0.0);
/// assert!((val - 0.6149266274460007).abs() < 1e-10);
/// ```
#[inline]
pub fn airy_bi(x: f64) -> f64 {
    crate::airy::bi(x)
}

/// Derivative of the Airy function of the first kind, Ai'(x).
///
/// # Arguments
///
/// * `x` – Real argument
///
/// # Returns
///
/// * `f64` – value of Ai'(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::airy_ai_prime;
///
/// // Ai'(0) ≈ -0.25881940379280678
/// let val = airy_ai_prime(0.0);
/// assert!((val - (-0.25881940379280678)).abs() < 1e-10);
/// ```
#[inline]
pub fn airy_ai_prime(x: f64) -> f64 {
    crate::airy::aip(x)
}

/// Derivative of the Airy function of the second kind, Bi'(x).
///
/// # Arguments
///
/// * `x` – Real argument
///
/// # Returns
///
/// * `f64` – value of Bi'(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::airy_bi_prime;
///
/// // Bi'(0) ≈ 0.4482883573538264
/// let val = airy_bi_prime(0.0);
/// assert!((val - 0.4482883573538264).abs() < 1e-10);
/// ```
#[inline]
pub fn airy_bi_prime(x: f64) -> f64 {
    crate::airy::bip(x)
}

/// First `n` zeros of the Airy function Ai(x), returned as a `Vec<f64>`.
///
/// All zeros are negative real numbers (Ai oscillates on the negative axis).
/// The zeros are sorted in ascending order (most negative first, i.e. the
/// largest-magnitude zero first).
///
/// The implementation delegates to `crate::airy::ai_zeros` for each index.
///
/// # Arguments
///
/// * `n` – Number of zeros to compute (n ≥ 1)
///
/// # Returns
///
/// * `Ok(Vec<f64>)` with `n` entries
/// * `Err(SpecialError::ValueError)` if n = 0
///
/// # Examples
///
/// ```
/// use scirs2_special::airy_ai_zeros;
///
/// let zeros = airy_ai_zeros(3).expect("airy_ai_zeros");
/// assert_eq!(zeros.len(), 3);
/// // First zero: a_1 ≈ -2.3381074
/// assert!((zeros[0] - (-2.338107410459767)).abs() < 1e-6);
/// // Zeros are negative and decreasing
/// assert!(zeros[0] < 0.0 && zeros[1] < zeros[0]);
/// ```
pub fn airy_ai_zeros(n: usize) -> SpecialResult<Vec<f64>> {
    if n == 0 {
        return Err(SpecialError::ValueError(
            "airy_ai_zeros: n must be >= 1".to_string(),
        ));
    }
    let mut zeros = Vec::with_capacity(n);
    for k in 1..=n {
        let z: f64 = crate::airy::ai_zeros(k)?;
        zeros.push(z);
    }
    Ok(zeros)
}

/// First `n` zeros of the Airy function Bi(x), returned as a `Vec<f64>`.
///
/// All zeros are negative real numbers, sorted in ascending order (most
/// negative first).
///
/// # Arguments
///
/// * `n` – Number of zeros to compute (n ≥ 1)
///
/// # Returns
///
/// * `Ok(Vec<f64>)` with `n` entries
/// * `Err(SpecialError::ValueError)` if n = 0
///
/// # Examples
///
/// ```
/// use scirs2_special::airy_bi_zeros;
///
/// let zeros = airy_bi_zeros(3).expect("airy_bi_zeros");
/// assert_eq!(zeros.len(), 3);
/// // First zero: b_1 ≈ -1.1737132
/// assert!((zeros[0] - (-1.173713222709128)).abs() < 1e-6);
/// assert!(zeros[0] < 0.0 && zeros[1] < zeros[0]);
/// ```
pub fn airy_bi_zeros(n: usize) -> SpecialResult<Vec<f64>> {
    if n == 0 {
        return Err(SpecialError::ValueError(
            "airy_bi_zeros: n must be >= 1".to_string(),
        ));
    }
    let mut zeros = Vec::with_capacity(n);
    for k in 1..=n {
        let z: f64 = crate::airy::bi_zeros(k)?;
        zeros.push(z);
    }
    Ok(zeros)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_airy_ai_at_zero() {
        assert_relative_eq!(airy_ai(0.0), 0.3550280538878172, epsilon = 1e-10);
    }

    #[test]
    fn test_airy_ai_positive() {
        assert_relative_eq!(airy_ai(1.0), 0.1352924163128814, epsilon = 1e-10);
        assert_relative_eq!(airy_ai(2.0), 0.03492413042327235, epsilon = 1e-10);
    }

    #[test]
    fn test_airy_ai_negative() {
        assert_relative_eq!(airy_ai(-1.0), 0.5355608832923521, epsilon = 1e-10);
        assert_relative_eq!(airy_ai(-2.0), 0.22740742820168557, epsilon = 1e-10);
    }

    #[test]
    fn test_airy_bi_at_zero() {
        assert_relative_eq!(airy_bi(0.0), 0.6149266274460007, epsilon = 1e-10);
    }

    #[test]
    fn test_airy_bi_positive() {
        assert_relative_eq!(airy_bi(1.0), 1.2074235949528715, epsilon = 1e-10);
    }

    #[test]
    fn test_airy_bi_negative() {
        assert_relative_eq!(airy_bi(-1.0), 0.103_997_389_496_944_6, epsilon = 1e-10);
    }

    #[test]
    fn test_airy_ai_prime_at_zero() {
        assert_relative_eq!(airy_ai_prime(0.0), -0.25881940379280678, epsilon = 1e-10);
    }

    #[test]
    fn test_airy_ai_prime_values() {
        assert_relative_eq!(airy_ai_prime(1.0), -0.16049975743698353, epsilon = 1e-10);
        assert_relative_eq!(airy_ai_prime(-1.0), -0.3271928185544436, epsilon = 1e-10);
    }

    #[test]
    fn test_airy_bi_prime_at_zero() {
        assert_relative_eq!(airy_bi_prime(0.0), 0.4482883573538264, epsilon = 1e-10);
    }

    #[test]
    fn test_airy_ai_zeros_count() {
        let zeros = airy_ai_zeros(5).expect("airy_ai_zeros(5)");
        assert_eq!(zeros.len(), 5);
        // All should be negative
        for z in &zeros {
            assert!(*z < 0.0, "zero should be negative, got {z}");
        }
    }

    #[test]
    fn test_airy_ai_zeros_values() {
        let zeros = airy_ai_zeros(3).expect("airy_ai_zeros(3)");
        // Known values from DLMF 9.9.1
        assert_relative_eq!(zeros[0], -2.338107410459767, epsilon = 1e-8);
        assert_relative_eq!(zeros[1], -4.087949444130970, epsilon = 1e-6);
        assert_relative_eq!(zeros[2], -5.520559828095551, epsilon = 1e-6);
    }

    #[test]
    fn test_airy_bi_zeros_count() {
        let zeros = airy_bi_zeros(5).expect("airy_bi_zeros(5)");
        assert_eq!(zeros.len(), 5);
        for z in &zeros {
            assert!(*z < 0.0, "zero should be negative, got {z}");
        }
    }

    #[test]
    fn test_airy_bi_zeros_values() {
        let zeros = airy_bi_zeros(3).expect("airy_bi_zeros(3)");
        assert_relative_eq!(zeros[0], -1.173713222709128, epsilon = 1e-8);
        assert_relative_eq!(zeros[1], -3.271093302836353, epsilon = 1e-6);
        assert_relative_eq!(zeros[2], -4.830737841662016, epsilon = 1e-6);
    }

    #[test]
    fn test_airy_zeros_n_zero_error() {
        assert!(airy_ai_zeros(0).is_err());
        assert!(airy_bi_zeros(0).is_err());
    }

    #[test]
    fn test_airy_wronskian() {
        // W[Ai, Bi] = Ai(x)Bi'(x) - Ai'(x)Bi(x) = 1/π
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            let ai_val = airy_ai(x);
            let bi_val = airy_bi(x);
            let aip_val = airy_ai_prime(x);
            let bip_val = airy_bi_prime(x);
            let wronskian = ai_val * bip_val - aip_val * bi_val;
            let expected = 1.0 / std::f64::consts::PI;
            let diff = (wronskian - expected).abs();
            assert!(diff < 1e-9, "Wronskian at x={x}: got {wronskian}, expected {expected}, diff={diff}");
        }
    }
}
