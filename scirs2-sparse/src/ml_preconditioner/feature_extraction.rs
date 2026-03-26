//! Feature extraction from raw CSR sparse matrix data.
//!
//! All functions operate on raw CSR arrays (`values`, `row_ptr`, `col_idx`)
//! to avoid coupling to specific sparse matrix wrapper types.

use crate::error::{SparseError, SparseResult};

use super::types::MatrixFeatures;

/// Extract numerical features from a sparse matrix in CSR format.
///
/// # Arguments
///
/// * `values`  – non-zero values array
/// * `row_ptr` – row pointer array of length `n + 1`
/// * `col_idx` – column index array (same length as `values`)
/// * `n`       – matrix dimension (assumes square n×n)
///
/// # Errors
///
/// Returns [`SparseError::InconsistentData`] if the input arrays are
/// inconsistent (e.g. `row_ptr` length ≠ `n + 1`, or `col_idx` / `values`
/// lengths don't match `row_ptr[n]`).
pub fn extract_features(
    values: &[f64],
    row_ptr: &[usize],
    col_idx: &[usize],
    n: usize,
) -> SparseResult<MatrixFeatures> {
    // ---- Validate inputs ----
    if row_ptr.len() != n + 1 {
        return Err(SparseError::InconsistentData {
            reason: format!(
                "row_ptr length {} does not equal n+1 = {}",
                row_ptr.len(),
                n + 1
            ),
        });
    }
    let nnz = row_ptr.get(n).copied().unwrap_or(0);
    if values.len() != nnz || col_idx.len() != nnz {
        return Err(SparseError::InconsistentData {
            reason: format!(
                "values/col_idx length ({}/{}) does not match nnz = {}",
                values.len(),
                col_idx.len(),
                nnz
            ),
        });
    }

    // Handle degenerate empty matrix
    if n == 0 {
        return Ok(MatrixFeatures {
            n: 0,
            nnz: 0,
            density: 0.0,
            max_row_nnz: 0,
            mean_row_nnz: 0.0,
            bandwidth: 0,
            bandwidth_ratio: 0.0,
            cond_estimate: 1.0,
            spectral_radius: 0.0,
            diag_dominance: 0.0,
            symmetry_measure: 0.0,
            has_positive_diagonal: true,
        });
    }

    // ---- Row-nnz statistics ----
    let mut max_row_nnz: usize = 0;
    for i in 0..n {
        let row_len = row_ptr[i + 1] - row_ptr[i];
        if row_len > max_row_nnz {
            max_row_nnz = row_len;
        }
    }
    let mean_row_nnz = nnz as f64 / n as f64;

    // ---- Density ----
    let density = nnz as f64 / (n as f64 * n as f64);

    // ---- Bandwidth ----
    let mut bandwidth: usize = 0;
    for i in 0..n {
        for idx in row_ptr[i]..row_ptr[i + 1] {
            let j = col_idx[idx];
            let diff = j.abs_diff(i);
            if diff > bandwidth {
                bandwidth = diff;
            }
        }
    }
    let bandwidth_ratio = if n > 1 {
        bandwidth as f64 / (n - 1) as f64
    } else {
        0.0
    };

    // ---- Diagonal extraction ----
    let mut diag = vec![0.0_f64; n];
    for i in 0..n {
        for idx in row_ptr[i]..row_ptr[i + 1] {
            if col_idx[idx] == i {
                diag[i] = values[idx];
            }
        }
    }

    // ---- Condition estimate from diagonal ----
    let mut abs_diag_max: f64 = 0.0;
    let mut abs_diag_min: f64 = f64::INFINITY;
    let mut has_positive_diagonal = true;
    for &d in &diag {
        let ad = d.abs();
        if ad > abs_diag_max {
            abs_diag_max = ad;
        }
        if ad < abs_diag_min {
            abs_diag_min = ad;
        }
        if d <= 0.0 {
            has_positive_diagonal = false;
        }
    }
    let cond_estimate = if abs_diag_min > 1e-15 {
        abs_diag_max / abs_diag_min
    } else {
        1e15
    };

    // ---- Diagonal dominance ----
    // min_i  |a_ii| / sum_{j != i} |a_ij|
    let mut diag_dominance = f64::INFINITY;
    for i in 0..n {
        let mut off_diag_sum = 0.0_f64;
        for idx in row_ptr[i]..row_ptr[i + 1] {
            if col_idx[idx] != i {
                off_diag_sum += values[idx].abs();
            }
        }
        let ratio = if off_diag_sum > 1e-30 {
            diag[i].abs() / off_diag_sum
        } else if diag[i].abs() > 1e-30 {
            // Row has no off-diagonal entries: trivially dominant.
            f64::INFINITY
        } else {
            0.0
        };
        if ratio < diag_dominance {
            diag_dominance = ratio;
        }
    }
    // Clamp infinity for serialization friendliness
    if diag_dominance.is_infinite() {
        diag_dominance = 1e12;
    }

    // ---- Spectral radius estimate (Gershgorin) ----
    let mut spectral_radius: f64 = 0.0;
    for i in 0..n {
        let mut row_abs_sum = 0.0_f64;
        for idx in row_ptr[i]..row_ptr[i + 1] {
            row_abs_sum += values[idx].abs();
        }
        if row_abs_sum > spectral_radius {
            spectral_radius = row_abs_sum;
        }
    }

    // ---- Symmetry measure ----
    // Fraction of nonzeros (i,j) for which (j,i) also exists structurally.
    let symmetry_measure = compute_symmetry_measure(row_ptr, col_idx, n);

    Ok(MatrixFeatures {
        n,
        nnz,
        density,
        max_row_nnz,
        mean_row_nnz,
        bandwidth,
        bandwidth_ratio,
        cond_estimate,
        spectral_radius,
        diag_dominance,
        symmetry_measure,
        has_positive_diagonal,
    })
}

/// Compute the structural symmetry measure of a CSR matrix.
///
/// Returns the fraction of entries (i, j) for which (j, i) also exists,
/// using a binary-search approach on sorted column indices.
fn compute_symmetry_measure(row_ptr: &[usize], col_idx: &[usize], n: usize) -> f64 {
    let nnz = row_ptr.get(n).copied().unwrap_or(0);
    if nnz == 0 {
        return 1.0; // vacuously symmetric
    }

    let mut symmetric_count: usize = 0;
    for i in 0..n {
        for idx in row_ptr[i]..row_ptr[i + 1] {
            let j = col_idx[idx];
            if j >= n {
                continue;
            }
            // Check whether (j, i) exists by searching row j
            let row_j_start = row_ptr[j];
            let row_j_end = row_ptr[j + 1];
            let row_j_cols = &col_idx[row_j_start..row_j_end];
            if row_j_cols.binary_search(&i).is_ok() {
                symmetric_count += 1;
            }
        }
    }
    symmetric_count as f64 / nnz as f64
}

/// Normalize matrix features into a fixed-length feature vector.
///
/// Large-scale features (n, nnz, bandwidth, spectral_radius, cond_estimate)
/// are log-transformed for better classifier behaviour.
pub fn normalize_features(features: &MatrixFeatures) -> Vec<f64> {
    vec![
        (features.n as f64 + 1.0).ln(),           // 0: log(n)
        (features.nnz as f64 + 1.0).ln(),         // 1: log(nnz)
        features.density,                         // 2: density
        (features.max_row_nnz as f64 + 1.0).ln(), // 3: log(max_row_nnz)
        features.mean_row_nnz,                    // 4: mean_row_nnz
        (features.bandwidth as f64 + 1.0).ln(),   // 5: log(bandwidth)
        features.bandwidth_ratio,                 // 6: bandwidth_ratio
        (features.cond_estimate + 1.0).ln(),      // 7: log(cond_estimate)
        (features.spectral_radius + 1.0).ln(),    // 8: log(spectral_radius)
        features.diag_dominance.min(100.0),       // 9: diag_dominance (clamped)
        features.symmetry_measure,                // 10: symmetry_measure
        if features.has_positive_diagonal {
            1.0
        } else {
            0.0
        }, // 11: has_positive_diagonal
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a simple 3×3 diagonally-dominant SPD matrix in CSR.
    ///
    ///  [ 4 -1  0 ]
    ///  [-1  4 -1 ]
    ///  [ 0 -1  4 ]
    fn tridiag_3x3() -> (Vec<f64>, Vec<usize>, Vec<usize>, usize) {
        let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptr = vec![0, 2, 5, 7];
        (values, row_ptr, col_idx, 3)
    }

    #[test]
    fn test_extract_features_basic() {
        let (vals, rp, ci, n) = tridiag_3x3();
        let f = extract_features(&vals, &rp, &ci, n).expect("extract");
        assert_eq!(f.n, 3);
        assert_eq!(f.nnz, 7);
        assert!(f.density > 0.7); // 7/9
        assert_eq!(f.max_row_nnz, 3);
        assert!((f.mean_row_nnz - 7.0 / 3.0).abs() < 1e-10);
        assert_eq!(f.bandwidth, 1);
        assert!(f.has_positive_diagonal);
        assert!(f.diag_dominance >= 1.0); // strictly diag dominant
    }

    #[test]
    fn test_extract_features_identity() {
        // 3×3 identity
        let values = vec![1.0, 1.0, 1.0];
        let col_idx = vec![0, 1, 2];
        let row_ptr = vec![0, 1, 2, 3];
        let f = extract_features(&values, &row_ptr, &col_idx, 3).expect("extract");
        assert_eq!(f.bandwidth, 0);
        assert!((f.density - 1.0 / 3.0).abs() < 1e-10);
        assert!((f.symmetry_measure - 1.0).abs() < 1e-10);
        assert!((f.cond_estimate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_features_empty() {
        let f = extract_features(&[], &[0], &[], 0).expect("extract");
        assert_eq!(f.n, 0);
        assert_eq!(f.nnz, 0);
    }

    #[test]
    fn test_symmetry_measure() {
        let (vals, rp, ci, n) = tridiag_3x3();
        let f = extract_features(&vals, &rp, &ci, n).expect("extract");
        // The tridiag is structurally symmetric
        assert!((f.symmetry_measure - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_features_length() {
        let (vals, rp, ci, n) = tridiag_3x3();
        let f = extract_features(&vals, &rp, &ci, n).expect("extract");
        let nf = normalize_features(&f);
        assert_eq!(nf.len(), 12);
        // All values should be finite
        for v in &nf {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_inconsistent_data_error() {
        // row_ptr too short
        let result = extract_features(&[1.0], &[0, 1], &[0], 3);
        assert!(result.is_err());
    }
}
