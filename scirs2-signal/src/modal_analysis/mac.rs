//! Modal Assurance Criterion (MAC) for comparing mode shapes.
//!
//! The MAC quantifies the correlation between two mode shape vectors:
//!
//! ```text
//! MAC(φ₁, φ₂) = |φ₁ᴴ φ₂|² / (φ₁ᴴ φ₁ · φ₂ᴴ φ₂)
//! ```
//!
//! A value of 1 indicates identical spatial distributions; 0 means orthogonal
//! shapes.  MAC is used to:
//! - Cross-validate results from two different measurement set-ups.
//! - Pair modes from SSI and FDD results.
//! - Assess convergence of iterative identification.
//!
//! ## References
//! - Allemang, R. J. & Brown, D. L. (1982). "A Correlation Coefficient for
//!   Modal Vector Analysis." Proc. IMAC 1.
//! - Pastor, M. et al. (2012). "Modal Assurance Criterion." Procedia Eng. 48.

use super::types::{MacMatrix, OmaResult};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Core MAC value
// ---------------------------------------------------------------------------

/// Compute the Modal Assurance Criterion between two mode shape vectors.
///
/// # Arguments
/// * `phi1` — first mode shape vector (any length)
/// * `phi2` — second mode shape vector (must have the same length as `phi1`)
///
/// # Returns
/// MAC value in `[0, 1]`.  Returns `0.0` if either vector has zero norm or
/// the lengths differ.
///
/// # Examples
/// ```
/// use scirs2_signal::modal_analysis::mac_value;
/// use scirs2_core::ndarray::array;
///
/// let phi1 = array![1.0_f64, 0.0, 0.0];
/// let phi2 = array![1.0_f64, 0.0, 0.0];
/// let m = mac_value(&phi1, &phi2);
/// assert!((m - 1.0).abs() < 1e-12);
/// ```
pub fn mac_value(phi1: &Array1<f64>, phi2: &Array1<f64>) -> f64 {
    if phi1.len() != phi2.len() || phi1.is_empty() {
        return 0.0;
    }
    let dot12: f64 = phi1.iter().zip(phi2.iter()).map(|(a, b)| a * b).sum();
    let norm1_sq: f64 = phi1.iter().map(|v| v * v).sum();
    let norm2_sq: f64 = phi2.iter().map(|v| v * v).sum();
    let denom = norm1_sq * norm2_sq;
    if denom < f64::EPSILON * f64::EPSILON {
        return 0.0;
    }
    (dot12 * dot12 / denom).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// MAC matrix
// ---------------------------------------------------------------------------

/// Build the full MAC matrix between two sets of mode shapes.
///
/// # Arguments
/// * `modes1` — slice of mode shape vectors (set 1, `n` vectors)
/// * `modes2` — slice of mode shape vectors (set 2, `m` vectors)
///
/// # Returns
/// [`MacMatrix`] of shape `n × m` where `matrix[i, j] = MAC(modes1[i], modes2[j])`.
///
/// # Errors
/// Returns an error if any pair of vectors has mismatched lengths.
///
/// # Examples
/// ```
/// use scirs2_signal::modal_analysis::mac_matrix;
/// use scirs2_core::ndarray::array;
///
/// let m1 = vec![array![1.0_f64, 0.0], array![0.0_f64, 1.0]];
/// let m2 = vec![array![1.0_f64, 0.0], array![0.0_f64, 1.0]];
/// let mac = mac_matrix(&m1, &m2).expect("mac_matrix should succeed");
/// assert!((mac.get(0, 0) - 1.0).abs() < 1e-12);
/// assert!(mac.get(0, 1) < 1e-12);
/// ```
pub fn mac_matrix(modes1: &[Array1<f64>], modes2: &[Array1<f64>]) -> SignalResult<MacMatrix> {
    let n = modes1.len();
    let m = modes2.len();
    if n == 0 || m == 0 {
        return Err(SignalError::InvalidArgument(
            "mode shape lists must be non-empty".to_string(),
        ));
    }
    let mut matrix = Array2::<f64>::zeros((n, m));
    for (i, phi1) in modes1.iter().enumerate() {
        for (j, phi2) in modes2.iter().enumerate() {
            if phi1.len() != phi2.len() {
                return Err(SignalError::DimensionMismatch(format!(
                    "mode shape lengths differ: {} vs {}",
                    phi1.len(),
                    phi2.len()
                )));
            }
            matrix[[i, j]] = mac_value(phi1, phi2);
        }
    }
    Ok(MacMatrix::from_array(matrix))
}

// ---------------------------------------------------------------------------
// Mode pairing
// ---------------------------------------------------------------------------

/// A single mode-pairing record: indices from the two results and MAC value.
#[derive(Debug, Clone)]
pub struct ModePair {
    /// Index of the mode in `result1`.
    pub idx1: usize,
    /// Index of the best-matching mode in `result2`.
    pub idx2: usize,
    /// MAC value for this pairing.
    pub mac: f64,
}

/// Pair modes from `result1` to modes in `result2` by maximum MAC.
///
/// For each mode in `result1`, the function finds the mode in `result2` with
/// the highest MAC value.  The returned vector has one entry per mode in
/// `result1`, sorted by `result1` index.
///
/// # Errors
/// Returns an error if the mode shape lengths are inconsistent.
///
/// # Examples
/// ```
/// use scirs2_signal::modal_analysis::{pair_modes, OmaResult, OmaMethod, ModalMode};
/// use scirs2_core::ndarray::array;
///
/// let m1 = vec![
///     ModalMode::new(5.0, 0.02, array![1.0_f64, 0.0]),
///     ModalMode::new(15.0, 0.03, array![0.0_f64, 1.0]),
/// ];
/// let m2 = vec![
///     ModalMode::new(5.1, 0.025, array![0.95_f64, 0.05]),
///     ModalMode::new(14.8, 0.028, array![0.05_f64, 0.95]),
/// ];
/// let r1 = OmaResult::new(m1, OmaMethod::Fdd, 0.25);
/// let r2 = OmaResult::new(m2, OmaMethod::SsiCov, 0.0);
/// let pairs = pair_modes(&r1, &r2).expect("pair_modes should succeed");
/// assert_eq!(pairs.len(), 2);
/// // Mode 0 of r1 should pair with mode 0 of r2 (both near 5 Hz with similar shapes)
/// assert_eq!(pairs[0].idx1, 0);
/// ```
pub fn pair_modes(result1: &OmaResult, result2: &OmaResult) -> SignalResult<Vec<ModePair>> {
    if result1.modes.is_empty() || result2.modes.is_empty() {
        return Ok(Vec::new());
    }
    let shapes1: Vec<Array1<f64>> = result1.modes.iter().map(|m| m.mode_shape.clone()).collect();
    let shapes2: Vec<Array1<f64>> = result2.modes.iter().map(|m| m.mode_shape.clone()).collect();

    // Check lengths
    let len1 = shapes1[0].len();
    let len2 = shapes2[0].len();
    if len1 != len2 {
        return Err(SignalError::DimensionMismatch(format!(
            "mode shape lengths differ between results: {len1} vs {len2}"
        )));
    }

    let mac = mac_matrix(&shapes1, &shapes2)?;
    let mut pairs = Vec::with_capacity(shapes1.len());
    for i in 0..shapes1.len() {
        let mut best_j = 0;
        let mut best_mac = mac.get(i, 0);
        for j in 1..shapes2.len() {
            let v = mac.get(i, j);
            if v > best_mac {
                best_mac = v;
                best_j = j;
            }
        }
        pairs.push(ModePair {
            idx1: i,
            idx2: best_j,
            mac: best_mac,
        });
    }
    Ok(pairs)
}

// ---------------------------------------------------------------------------
// Diagnostics helpers
// ---------------------------------------------------------------------------

/// Compute the average off-diagonal MAC value of a matrix (indicator of
/// mode shape independence; should be close to 0 for well-separated modes).
pub fn average_off_diagonal_mac(mac: &MacMatrix) -> f64 {
    let n = mac.n_rows().min(mac.n_cols());
    if n <= 1 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0_usize;
    for i in 0..mac.n_rows() {
        for j in 0..mac.n_cols() {
            if i != j {
                sum += mac.get(i, j);
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

/// Check whether all diagonal MAC values exceed a threshold (used to verify
/// that auto-MAC is near-identity for a set of orthogonal mode shapes).
pub fn diagonal_mac_all_above(mac: &MacMatrix, threshold: f64) -> bool {
    let n = mac.n_rows().min(mac.n_cols());
    (0..n).all(|i| mac.get(i, i) >= threshold)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // --- mac_value ---

    #[test]
    fn test_mac_identical_vectors() {
        let phi = array![1.0_f64, 2.0, 3.0];
        let m = mac_value(&phi, &phi);
        assert!(
            (m - 1.0).abs() < 1e-12,
            "MAC of identical vectors should be 1, got {m}"
        );
    }

    #[test]
    fn test_mac_orthogonal_vectors() {
        let phi1 = array![1.0_f64, 0.0, 0.0];
        let phi2 = array![0.0_f64, 1.0, 0.0];
        let m = mac_value(&phi1, &phi2);
        assert!(m < 1e-12, "MAC of orthogonal vectors should be ~0, got {m}");
    }

    #[test]
    fn test_mac_scaled_vector() {
        let phi1 = array![1.0_f64, 2.0, 3.0];
        let phi2 = array![2.0_f64, 4.0, 6.0];
        let m = mac_value(&phi1, &phi2);
        assert!(
            (m - 1.0).abs() < 1e-12,
            "MAC of scaled vectors should be 1, got {m}"
        );
    }

    #[test]
    fn test_mac_zero_vector() {
        let phi1 = array![0.0_f64, 0.0, 0.0];
        let phi2 = array![1.0_f64, 0.0, 0.0];
        let m = mac_value(&phi1, &phi2);
        assert_eq!(m, 0.0, "MAC with zero vector should be 0");
    }

    #[test]
    fn test_mac_length_mismatch() {
        let phi1 = array![1.0_f64, 0.0];
        let phi2 = array![1.0_f64, 0.0, 0.0];
        let m = mac_value(&phi1, &phi2);
        assert_eq!(m, 0.0, "MAC with mismatched lengths should be 0");
    }

    // --- mac_matrix ---

    #[test]
    fn test_mac_matrix_identity_diagonal() {
        let modes = vec![
            array![1.0_f64, 0.0, 0.0],
            array![0.0_f64, 1.0, 0.0],
            array![0.0_f64, 0.0, 1.0],
        ];
        let mac = mac_matrix(&modes, &modes).expect("mac_matrix should succeed");
        assert_eq!(mac.n_rows(), 3);
        assert_eq!(mac.n_cols(), 3);
        for i in 0..3 {
            assert!(
                (mac.get(i, i) - 1.0).abs() < 1e-12,
                "diagonal[{i}] should be 1"
            );
        }
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(
                        mac.get(i, j) < 1e-12,
                        "off-diagonal[{i},{j}] should be ~0, got {}",
                        mac.get(i, j)
                    );
                }
            }
        }
    }

    #[test]
    fn test_mac_matrix_empty_error() {
        let empty: Vec<Array1<f64>> = vec![];
        let modes = vec![array![1.0_f64, 0.0]];
        assert!(mac_matrix(&empty, &modes).is_err());
        assert!(mac_matrix(&modes, &empty).is_err());
    }

    // --- pair_modes ---

    #[test]
    fn test_pair_modes_perfect_match() {
        use super::super::types::{ModalMode, OmaMethod, OmaResult};
        let m1 = vec![
            ModalMode::new(5.0, 0.02, array![1.0_f64, 0.0]),
            ModalMode::new(15.0, 0.03, array![0.0_f64, 1.0]),
        ];
        let m2 = vec![
            ModalMode::new(5.0, 0.02, array![1.0_f64, 0.0]),
            ModalMode::new(15.0, 0.03, array![0.0_f64, 1.0]),
        ];
        let r1 = OmaResult::new(m1, OmaMethod::Fdd, 0.25);
        let r2 = OmaResult::new(m2, OmaMethod::SsiCov, 0.0);
        let pairs = pair_modes(&r1, &r2).expect("pair_modes should succeed");
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].idx2, 0);
        assert!((pairs[0].mac - 1.0).abs() < 1e-10);
        assert_eq!(pairs[1].idx2, 1);
        assert!((pairs[1].mac - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pair_modes_empty_results() {
        use super::super::types::{OmaMethod, OmaResult};
        let r1 = OmaResult::new(vec![], OmaMethod::Fdd, 0.25);
        let r2 = OmaResult::new(vec![], OmaMethod::SsiCov, 0.0);
        let pairs = pair_modes(&r1, &r2).expect("empty results should return empty pairs");
        assert!(pairs.is_empty());
    }

    // --- diagnostics ---

    #[test]
    fn test_average_off_diagonal_zero() {
        let modes = vec![array![1.0_f64, 0.0], array![0.0_f64, 1.0]];
        let mac = mac_matrix(&modes, &modes).expect("mac_matrix should succeed");
        let avg = average_off_diagonal_mac(&mac);
        assert!(avg < 1e-12, "off-diagonal avg should be ~0, got {avg}");
    }

    #[test]
    fn test_diagonal_mac_all_above_threshold() {
        let modes = vec![array![1.0_f64, 0.0], array![0.0_f64, 1.0]];
        let mac = mac_matrix(&modes, &modes).expect("mac_matrix should succeed");
        assert!(diagonal_mac_all_above(&mac, 0.99));
        assert!(!diagonal_mac_all_above(&mac, 1.01));
    }
}
