//! Modal Assurance Criterion (MAC) for mode shape comparison
//!
//! The MAC is the most widely used metric for comparing mode shapes in
//! structural dynamics and modal analysis.  It quantifies the degree of
//! consistency (linearity) between two mode shape vectors, producing a
//! scalar in `[0, 1]` where 1 indicates identical (or linearly dependent)
//! mode shapes and 0 indicates completely orthogonal shapes.
//!
//! # Algorithms
//!
//! * **MAC value** (scalar) between two vectors.
//! * **MAC matrix** for comparing two full sets of mode shapes.
//! * **Auto-MAC** to check orthogonality within a single set.
//! * **Cross-MAC** for comparing experimental vs. analytical modes.
//! * **CoMAC** (Coordinate MAC) for evaluating individual DOF contributions,
//!   useful for sensor placement optimisation.
//!
//! # References
//! - Allemang, R.J. & Brown, D.L. (1982). "A Correlation Coefficient for
//!   Modal Vector Analysis." *Proc. 1st IMAC*, pp. 110–116.
//! - Allemang, R.J. (2003). "The Modal Assurance Criterion – Twenty Years
//!   of Use and Abuse." *Sound and Vibration*, 37(8), 14–21.

use crate::error::{SignalError, SignalResult};

// ---------------------------------------------------------------------------
// Single MAC value
// ---------------------------------------------------------------------------

/// Compute the Modal Assurance Criterion between two mode shape vectors.
///
/// ```text
/// MAC = |phi_1^H * phi_2|^2 / (phi_1^H * phi_1  *  phi_2^H * phi_2)
/// ```
///
/// For real-valued mode shapes the Hermitian transpose reduces to the
/// ordinary transpose, and the formula becomes:
///
/// ```text
/// MAC = (phi_1 . phi_2)^2 / (||phi_1||^2  *  ||phi_2||^2)
/// ```
///
/// # Errors
/// Returns an error if the vectors have different lengths or are empty.
pub fn mac_value(phi1: &[f64], phi2: &[f64]) -> SignalResult<f64> {
    validate_pair(phi1, phi2)?;
    Ok(mac_inner(phi1, phi2))
}

/// Compute MAC between two complex mode shape vectors.
///
/// Each mode shape is given as a pair of slices `(real, imag)`.
///
/// # Errors
/// Returns an error if lengths are inconsistent or zero.
pub fn mac_value_complex(
    phi1_re: &[f64],
    phi1_im: &[f64],
    phi2_re: &[f64],
    phi2_im: &[f64],
) -> SignalResult<f64> {
    let n = phi1_re.len();
    if phi1_im.len() != n || phi2_re.len() != n || phi2_im.len() != n {
        return Err(SignalError::DimensionMismatch(
            "All real/imaginary parts must have the same length".to_string(),
        ));
    }
    if n == 0 {
        return Err(SignalError::InvalidInput(
            "Mode shape vectors must not be empty".to_string(),
        ));
    }

    // phi_1^H * phi_2 = sum_k (conj(phi1_k) * phi2_k)
    let mut cross_re = 0.0f64;
    let mut cross_im = 0.0f64;
    for k in 0..n {
        // conj(a+bi) * (c+di) = (ac+bd) + (ad-bc)i  ... wait, that's wrong
        // conj(a+bi) = a - bi
        // (a - bi)(c + di) = ac + adi - bci - bdi^2 = (ac+bd) + (ad-bc)i
        let a = phi1_re[k];
        let b = phi1_im[k];
        let c = phi2_re[k];
        let d = phi2_im[k];
        cross_re += a * c + b * d;
        cross_im += a * d - b * c;
    }

    let cross_abs_sq = cross_re * cross_re + cross_im * cross_im;

    // phi_1^H * phi_1 = sum |phi1_k|^2
    let norm1_sq: f64 = phi1_re
        .iter()
        .zip(phi1_im.iter())
        .map(|(r, i)| r * r + i * i)
        .sum();
    let norm2_sq: f64 = phi2_re
        .iter()
        .zip(phi2_im.iter())
        .map(|(r, i)| r * r + i * i)
        .sum();

    let denom = norm1_sq * norm2_sq;
    if denom < 1e-30 {
        return Ok(0.0);
    }
    Ok(cross_abs_sq / denom)
}

// ---------------------------------------------------------------------------
// MAC matrix
// ---------------------------------------------------------------------------

/// Compute the MAC matrix between two sets of mode shapes.
///
/// Returns an `(m1 x m2)` matrix where entry `[i][j]` is the MAC value
/// between the i-th mode of set A and the j-th mode of set B.
///
/// Each inner `Vec<f64>` represents one mode shape vector.
///
/// # Errors
/// Returns an error if any pair of mode shapes has a length mismatch, or if
/// sets are empty.
pub fn mac_matrix(set_a: &[Vec<f64>], set_b: &[Vec<f64>]) -> SignalResult<Vec<Vec<f64>>> {
    if set_a.is_empty() || set_b.is_empty() {
        return Err(SignalError::InvalidInput(
            "Mode shape sets must not be empty".to_string(),
        ));
    }
    let n_dof = set_a[0].len();
    // Validate lengths
    for (i, phi) in set_a.iter().enumerate() {
        if phi.len() != n_dof {
            return Err(SignalError::DimensionMismatch(format!(
                "set_a[{i}] has length {} but set_a[0] has length {n_dof}",
                phi.len()
            )));
        }
    }
    for (j, phi) in set_b.iter().enumerate() {
        if phi.len() != n_dof {
            return Err(SignalError::DimensionMismatch(format!(
                "set_b[{j}] has length {} but expected {n_dof}",
                phi.len()
            )));
        }
    }

    let m1 = set_a.len();
    let m2 = set_b.len();
    let mut mat = vec![vec![0.0f64; m2]; m1];
    for i in 0..m1 {
        for j in 0..m2 {
            mat[i][j] = mac_inner(&set_a[i], &set_b[j]);
        }
    }
    Ok(mat)
}

// ---------------------------------------------------------------------------
// Auto-MAC
// ---------------------------------------------------------------------------

/// Compute the Auto-MAC matrix for a single set of mode shapes.
///
/// The Auto-MAC is the MAC matrix of a set compared with itself.  Ideally the
/// diagonal entries should be 1.0 and off-diagonal entries should be close to
/// 0.0, indicating well-separated (orthogonal) modes.
///
/// Returns an `(m x m)` symmetric matrix.
pub fn auto_mac(modes: &[Vec<f64>]) -> SignalResult<Vec<Vec<f64>>> {
    if modes.is_empty() {
        return Err(SignalError::InvalidInput(
            "Mode shape set must not be empty".to_string(),
        ));
    }
    let n_dof = modes[0].len();
    for (i, phi) in modes.iter().enumerate() {
        if phi.len() != n_dof {
            return Err(SignalError::DimensionMismatch(format!(
                "modes[{i}] has length {} but modes[0] has length {n_dof}",
                phi.len()
            )));
        }
    }

    let m = modes.len();
    let mut mat = vec![vec![0.0f64; m]; m];
    for i in 0..m {
        mat[i][i] = 1.0; // MAC of a vector with itself is always 1
        for j in i + 1..m {
            let val = mac_inner(&modes[i], &modes[j]);
            mat[i][j] = val;
            mat[j][i] = val;
        }
    }
    Ok(mat)
}

// ---------------------------------------------------------------------------
// Cross-MAC
// ---------------------------------------------------------------------------

/// Compute the Cross-MAC matrix between experimental and analytical mode shapes.
///
/// This is functionally identical to [`mac_matrix`] but semantically indicates
/// that set A is from an experiment and set B is from an analytical/FE model.
/// The function additionally returns a vector of best-match pairs with their
/// MAC values, sorted by descending MAC.
///
/// # Returns
/// `(mac_matrix, matched_pairs)` where each matched pair is
/// `(experimental_index, analytical_index, mac_value)`.
pub fn cross_mac(
    experimental: &[Vec<f64>],
    analytical: &[Vec<f64>],
) -> SignalResult<(Vec<Vec<f64>>, Vec<(usize, usize, f64)>)> {
    let mat = mac_matrix(experimental, analytical)?;
    let m1 = experimental.len();
    let m2 = analytical.len();

    // Greedy best-match pairing (Hungarian-like greedy assignment)
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    let mut used_exp = vec![false; m1];
    let mut used_ana = vec![false; m2];
    let n_pairs = m1.min(m2);

    for _ in 0..n_pairs {
        // Find the maximum unassigned MAC entry
        let mut best_val = -1.0f64;
        let mut best_i = 0;
        let mut best_j = 0;
        for i in 0..m1 {
            if used_exp[i] {
                continue;
            }
            for j in 0..m2 {
                if used_ana[j] {
                    continue;
                }
                if mat[i][j] > best_val {
                    best_val = mat[i][j];
                    best_i = i;
                    best_j = j;
                }
            }
        }
        if best_val < 0.0 {
            break;
        }
        used_exp[best_i] = true;
        used_ana[best_j] = true;
        pairs.push((best_i, best_j, best_val));
    }

    // Sort by descending MAC value
    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    Ok((mat, pairs))
}

// ---------------------------------------------------------------------------
// CoMAC (Coordinate MAC)
// ---------------------------------------------------------------------------

/// Compute the Coordinate Modal Assurance Criterion (CoMAC).
///
/// CoMAC evaluates how well each DOF (coordinate/sensor location) contributes
/// to mode shape correlation between two sets.  It is defined per DOF as:
///
/// ```text
/// CoMAC(k) = [ sum_r |phi_A_r(k) * phi_B_r(k)| ]^2
///            / [ sum_r |phi_A_r(k)|^2  *  sum_r |phi_B_r(k)|^2 ]
/// ```
///
/// where the sum runs over all mode pairs `r`.
///
/// A low CoMAC value at DOF `k` indicates that sensor `k` is poorly
/// correlated, suggesting a possible modelling error at that DOF or that the
/// sensor placement there is suboptimal.
///
/// # Arguments
/// * `set_a` – First set of mode shapes (e.g. experimental).
/// * `set_b` – Second set of mode shapes (e.g. analytical), paired 1:1 with `set_a`.
///
/// # Returns
/// A vector of length `n_dof` containing CoMAC values in `[0, 1]`.
///
/// # Errors
/// Returns an error if the two sets have different numbers of modes, if mode
/// shape vectors differ in length, or if either set is empty.
pub fn comac(set_a: &[Vec<f64>], set_b: &[Vec<f64>]) -> SignalResult<Vec<f64>> {
    if set_a.is_empty() || set_b.is_empty() {
        return Err(SignalError::InvalidInput(
            "Mode shape sets must not be empty".to_string(),
        ));
    }
    if set_a.len() != set_b.len() {
        return Err(SignalError::DimensionMismatch(format!(
            "set_a has {} modes but set_b has {} — they must be paired 1:1",
            set_a.len(),
            set_b.len()
        )));
    }
    let n_dof = set_a[0].len();
    for (i, phi) in set_a.iter().enumerate() {
        if phi.len() != n_dof {
            return Err(SignalError::DimensionMismatch(format!(
                "set_a[{i}] has length {} but expected {n_dof}",
                phi.len()
            )));
        }
    }
    for (i, phi) in set_b.iter().enumerate() {
        if phi.len() != n_dof {
            return Err(SignalError::DimensionMismatch(format!(
                "set_b[{i}] has length {} but expected {n_dof}",
                phi.len()
            )));
        }
    }

    let n_modes = set_a.len();
    let mut comac_vals = vec![0.0f64; n_dof];

    for k in 0..n_dof {
        let mut numerator_sum = 0.0f64;
        let mut denom_a_sum = 0.0f64;
        let mut denom_b_sum = 0.0f64;

        for r in 0..n_modes {
            let a_k = set_a[r][k];
            let b_k = set_b[r][k];
            numerator_sum += (a_k * b_k).abs();
            denom_a_sum += a_k * a_k;
            denom_b_sum += b_k * b_k;
        }

        let denom = denom_a_sum * denom_b_sum;
        comac_vals[k] = if denom < 1e-30 {
            0.0
        } else {
            (numerator_sum * numerator_sum / denom).clamp(0.0, 1.0)
        };
    }

    Ok(comac_vals)
}

// ---------------------------------------------------------------------------
// Partial MAC (PMAC)
// ---------------------------------------------------------------------------

/// Compute the Partial MAC between two mode shapes using only a subset of DOFs.
///
/// This is useful when comparing mode shapes measured at different sensor
/// layouts, where only a common subset of DOFs is available.
///
/// # Arguments
/// * `phi1` – First mode shape (full DOF set).
/// * `phi2` – Second mode shape (full DOF set, same length as `phi1`).
/// * `dof_indices` – Indices of the DOFs to include in the comparison.
///
/// # Errors
/// Returns an error if vectors differ in length or any index is out of range.
pub fn partial_mac(phi1: &[f64], phi2: &[f64], dof_indices: &[usize]) -> SignalResult<f64> {
    if phi1.len() != phi2.len() {
        return Err(SignalError::DimensionMismatch(format!(
            "Mode shape vectors have different lengths: {} vs {}",
            phi1.len(),
            phi2.len()
        )));
    }
    if dof_indices.is_empty() {
        return Err(SignalError::InvalidInput(
            "DOF index set must not be empty".to_string(),
        ));
    }
    let n = phi1.len();
    for &idx in dof_indices {
        if idx >= n {
            return Err(SignalError::InvalidInput(format!(
                "DOF index {idx} is out of range (n_dof = {n})"
            )));
        }
    }

    // Extract subset
    let sub1: Vec<f64> = dof_indices.iter().map(|&i| phi1[i]).collect();
    let sub2: Vec<f64> = dof_indices.iter().map(|&i| phi2[i]).collect();

    Ok(mac_inner(&sub1, &sub2))
}

// ---------------------------------------------------------------------------
// Weighted MAC
// ---------------------------------------------------------------------------

/// Compute a mass-weighted MAC value.
///
/// ```text
/// MAC_w = |phi_1^T W phi_2|^2 / ((phi_1^T W phi_1) * (phi_2^T W phi_2))
/// ```
///
/// where `W = diag(weights)`.  When `weights` is the diagonal of the mass
/// matrix, this gives the mass-weighted MAC which accounts for the physical
/// significance of each DOF.
///
/// # Errors
/// Returns an error if vectors or weights have inconsistent lengths.
pub fn weighted_mac(phi1: &[f64], phi2: &[f64], weights: &[f64]) -> SignalResult<f64> {
    let n = phi1.len();
    if phi2.len() != n || weights.len() != n {
        return Err(SignalError::DimensionMismatch(format!(
            "phi1 ({}), phi2 ({}), and weights ({}) must all have the same length",
            n,
            phi2.len(),
            weights.len()
        )));
    }
    if n == 0 {
        return Err(SignalError::InvalidInput(
            "Mode shape vectors must not be empty".to_string(),
        ));
    }

    let cross: f64 = (0..n).map(|k| phi1[k] * weights[k] * phi2[k]).sum();
    let norm1: f64 = (0..n).map(|k| phi1[k] * weights[k] * phi1[k]).sum();
    let norm2: f64 = (0..n).map(|k| phi2[k] * weights[k] * phi2[k]).sum();

    let denom = norm1 * norm2;
    if denom < 1e-30 {
        return Ok(0.0);
    }
    Ok((cross * cross / denom).clamp(0.0, 1.0))
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

/// MAC computation for real-valued vectors (no error checking).
fn mac_inner(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum();
    let norm_b: f64 = b.iter().map(|x| x * x).sum();
    let denom = norm_a * norm_b;
    if denom < 1e-30 {
        return 0.0;
    }
    (dot * dot) / denom
}

/// Validate that two mode shape slices are non-empty and of equal length.
fn validate_pair(phi1: &[f64], phi2: &[f64]) -> SignalResult<()> {
    if phi1.len() != phi2.len() {
        return Err(SignalError::DimensionMismatch(format!(
            "Mode shape vectors have different lengths: {} vs {}",
            phi1.len(),
            phi2.len()
        )));
    }
    if phi1.is_empty() {
        return Err(SignalError::InvalidInput(
            "Mode shape vectors must not be empty".to_string(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mac_identical_vectors() {
        let phi = vec![1.0, 0.5, -0.3, 0.8];
        let mac = mac_value(&phi, &phi).expect("MAC should succeed");
        assert!(
            (mac - 1.0).abs() < 1e-10,
            "MAC of identical vectors should be 1.0, got {mac}"
        );
    }

    #[test]
    fn test_mac_orthogonal_vectors() {
        let phi1 = vec![1.0, 0.0, 0.0];
        let phi2 = vec![0.0, 1.0, 0.0];
        let mac = mac_value(&phi1, &phi2).expect("MAC should succeed");
        assert!(
            mac.abs() < 1e-10,
            "MAC of orthogonal vectors should be 0.0, got {mac}"
        );
    }

    #[test]
    fn test_mac_scaled_vectors() {
        let phi1 = vec![1.0, 2.0, 3.0];
        let phi2 = vec![2.0, 4.0, 6.0]; // 2 * phi1
        let mac = mac_value(&phi1, &phi2).expect("MAC should succeed");
        assert!(
            (mac - 1.0).abs() < 1e-10,
            "MAC of scaled vectors should be 1.0, got {mac}"
        );
    }

    #[test]
    fn test_mac_negated_vectors() {
        let phi1 = vec![1.0, -0.5, 0.3];
        let phi2 = vec![-1.0, 0.5, -0.3]; // -phi1
        let mac = mac_value(&phi1, &phi2).expect("MAC should succeed");
        assert!(
            (mac - 1.0).abs() < 1e-10,
            "MAC of negated vectors should be 1.0, got {mac}"
        );
    }

    #[test]
    fn test_mac_dimension_mismatch() {
        let phi1 = vec![1.0, 2.0];
        let phi2 = vec![1.0, 2.0, 3.0];
        let result = mac_value(&phi1, &phi2);
        assert!(result.is_err(), "Mismatched lengths should fail");
    }

    #[test]
    fn test_mac_empty_vectors() {
        let phi1: Vec<f64> = vec![];
        let phi2: Vec<f64> = vec![];
        let result = mac_value(&phi1, &phi2);
        assert!(result.is_err(), "Empty vectors should fail");
    }

    #[test]
    fn test_mac_complex_identical() {
        let re = vec![1.0, 0.5, -0.3];
        let im = vec![0.2, -0.1, 0.4];
        let mac = mac_value_complex(&re, &im, &re, &im).expect("Complex MAC should succeed");
        assert!(
            (mac - 1.0).abs() < 1e-10,
            "Complex MAC of identical vectors should be 1.0, got {mac}"
        );
    }

    #[test]
    fn test_mac_complex_orthogonal() {
        // Two complex vectors that are orthogonal in the Hermitian sense
        let re1 = vec![1.0, 0.0];
        let im1 = vec![0.0, 0.0];
        let re2 = vec![0.0, 1.0];
        let im2 = vec![0.0, 0.0];
        let mac = mac_value_complex(&re1, &im1, &re2, &im2).expect("Complex MAC should succeed");
        assert!(
            mac.abs() < 1e-10,
            "Complex MAC of orthogonal vectors should be 0.0, got {mac}"
        );
    }

    #[test]
    fn test_mac_matrix_basic() {
        let modes_a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let modes_b = modes_a.clone();
        let mat = mac_matrix(&modes_a, &modes_b).expect("MAC matrix should succeed");
        assert_eq!(mat.len(), 3);
        assert_eq!(mat[0].len(), 3);

        // Diagonal should be 1.0
        for i in 0..3 {
            assert!(
                (mat[i][i] - 1.0).abs() < 1e-10,
                "Diagonal MAC[{i}][{i}] should be 1.0, got {}",
                mat[i][i]
            );
        }
        // Off-diagonal should be 0.0
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(
                        mat[i][j].abs() < 1e-10,
                        "Off-diagonal MAC[{i}][{j}] should be 0.0, got {}",
                        mat[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn test_auto_mac_orthogonal_modes() {
        let modes = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let mat = auto_mac(&modes).expect("Auto-MAC should succeed");
        // Symmetry check
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (mat[i][j] - mat[j][i]).abs() < 1e-14,
                    "Auto-MAC must be symmetric"
                );
            }
        }
        // Diagonal = 1
        for i in 0..3 {
            assert!((mat[i][i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_auto_mac_correlated_modes() {
        // Two nearly-identical modes should have high off-diagonal MAC
        let modes = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.01, 2.01, 3.01], // nearly identical
        ];
        let mat = auto_mac(&modes).expect("Auto-MAC should succeed");
        assert!(
            mat[0][1] > 0.99,
            "Nearly identical modes should have MAC close to 1, got {}",
            mat[0][1]
        );
    }

    #[test]
    fn test_cross_mac_perfect_match() {
        let exp = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let ana = vec![vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 0.0]];
        let (mat, pairs) = cross_mac(&exp, &ana).expect("Cross-MAC should succeed");
        assert_eq!(pairs.len(), 2);
        // Each experimental mode should match exactly one analytical mode with MAC = 1
        for &(_, _, mac_val) in &pairs {
            assert!(
                (mac_val - 1.0).abs() < 1e-10,
                "Perfect match expected, got MAC = {mac_val}"
            );
        }
        // Check matrix shape
        assert_eq!(mat.len(), 2);
        assert_eq!(mat[0].len(), 2);
    }

    #[test]
    fn test_cross_mac_unequal_sets() {
        let exp = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let ana = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let (mat, pairs) = cross_mac(&exp, &ana).expect("Cross-MAC should succeed");
        assert_eq!(mat.len(), 3);
        assert_eq!(mat[0].len(), 2);
        // Only min(3, 2) = 2 pairs
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn test_comac_identical_sets() {
        let modes = vec![vec![1.0, 2.0, 3.0], vec![0.5, -1.0, 0.8]];
        let comac_vals = comac(&modes, &modes).expect("CoMAC should succeed");
        assert_eq!(comac_vals.len(), 3);
        // With identical sets, CoMAC should be 1.0 at every DOF
        for (k, &v) in comac_vals.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-10,
                "CoMAC[{k}] should be 1.0 for identical sets, got {v}"
            );
        }
    }

    #[test]
    fn test_comac_mismatched_dof() {
        let set_a = vec![vec![1.0, 2.0]];
        let set_b = vec![vec![1.0, 2.0, 3.0]];
        let result = comac(&set_a, &set_b);
        assert!(result.is_err(), "Mismatched DOF count should fail");
    }

    #[test]
    fn test_comac_mismatched_mode_count() {
        let set_a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let set_b = vec![vec![1.0, 2.0]];
        let result = comac(&set_a, &set_b);
        assert!(result.is_err(), "Mismatched mode count should fail");
    }

    #[test]
    fn test_partial_mac() {
        let phi1 = vec![1.0, 0.0, 0.5, 0.0];
        let phi2 = vec![1.0, 0.0, 0.5, 0.0];
        // Full MAC should be 1.0
        let full = mac_value(&phi1, &phi2).expect("full MAC");
        assert!((full - 1.0).abs() < 1e-10);
        // Partial MAC using DOFs 0 and 2
        let partial = partial_mac(&phi1, &phi2, &[0, 2]).expect("partial MAC");
        assert!(
            (partial - 1.0).abs() < 1e-10,
            "Partial MAC should be 1.0 for identical subsets"
        );
    }

    #[test]
    fn test_partial_mac_out_of_range() {
        let phi = vec![1.0, 2.0];
        let result = partial_mac(&phi, &phi, &[0, 5]);
        assert!(result.is_err(), "Out-of-range DOF index should fail");
    }

    #[test]
    fn test_weighted_mac_uniform() {
        let phi1 = vec![1.0, 2.0, 3.0];
        let phi2 = vec![2.0, 4.0, 6.0];
        let weights = vec![1.0, 1.0, 1.0];
        let wmac = weighted_mac(&phi1, &phi2, &weights).expect("Weighted MAC should succeed");
        assert!(
            (wmac - 1.0).abs() < 1e-10,
            "Weighted MAC with uniform weights on scaled vectors should be 1.0"
        );
    }

    #[test]
    fn test_weighted_mac_non_uniform() {
        // With non-uniform weights, MAC of orthogonal vectors should still be 0
        let phi1 = vec![1.0, 0.0];
        let phi2 = vec![0.0, 1.0];
        let weights = vec![2.0, 3.0];
        let wmac = weighted_mac(&phi1, &phi2, &weights).expect("Weighted MAC should succeed");
        assert!(
            wmac.abs() < 1e-10,
            "Weighted MAC of orthogonal vectors should be 0.0"
        );
    }
}
