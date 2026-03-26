//! Gaussian field precision matrix construction for INLA latent models
//!
//! This module provides types and functions for building precision matrices
//! of common latent Gaussian field types used in INLA, including:
//!
//! - IID (independent identically distributed)
//! - RW1/RW2 (random walks of order 1 and 2)
//! - AR1 (autoregressive order 1)
//! - ICAR (intrinsic conditional autoregressive / spatial)
//! - Matern (Matern covariance field on 1D grid via SPDE)
//!
//! The precision matrices encode the GMRF (Gaussian Markov Random Field)
//! structure that INLA exploits for fast inference.

use scirs2_core::ndarray::Array2;

use crate::error::{StatsError, StatsResult};

/// Type of latent Gaussian field, determining the structure of the precision matrix
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum LatentFieldType {
    /// Independent identically distributed: Q = scale * I_n
    IID,
    /// First-order random walk: Q encodes first differences
    RW1,
    /// Second-order random walk: Q encodes second differences
    RW2,
    /// Autoregressive order 1 with correlation parameter phi in (-1, 1)
    AR1 {
        /// Autoregressive correlation parameter (must satisfy |phi| < 1)
        phi: f64,
    },
    /// Intrinsic Conditional Autoregressive model (spatial Laplacian)
    ICAR {
        /// Edge list: pairs of (node_i, node_j) representing adjacency
        adjacency: Vec<(usize, usize)>,
    },
    /// Matern covariance field on a regular 1D grid (SPDE discretization)
    Matern {
        /// Spatial range parameter (must be positive)
        range: f64,
        /// Smoothness parameter nu (must be positive)
        smoothness: f64,
    },
}

/// Validate parameters for a latent field type
///
/// # Errors
///
/// Returns `StatsError::InvalidArgument` if:
/// - AR1: |phi| >= 1
/// - ICAR: adjacency contains self-loops or out-of-bound indices (when n is known)
/// - Matern: range or smoothness is non-positive
pub fn validate_field_params(field_type: &LatentFieldType) -> StatsResult<()> {
    match field_type {
        LatentFieldType::IID | LatentFieldType::RW1 | LatentFieldType::RW2 => Ok(()),
        LatentFieldType::AR1 { phi } => {
            if phi.abs() >= 1.0 {
                return Err(StatsError::InvalidArgument(format!(
                    "AR1 parameter phi must satisfy |phi| < 1, got phi = {phi}"
                )));
            }
            if phi.is_nan() {
                return Err(StatsError::InvalidArgument(
                    "AR1 parameter phi must not be NaN".to_string(),
                ));
            }
            Ok(())
        }
        LatentFieldType::ICAR { adjacency } => {
            for &(i, j) in adjacency {
                if i == j {
                    return Err(StatsError::InvalidArgument(format!(
                        "ICAR adjacency contains self-loop at node {i}"
                    )));
                }
            }
            Ok(())
        }
        LatentFieldType::Matern { range, smoothness } => {
            if *range <= 0.0 || range.is_nan() {
                return Err(StatsError::InvalidArgument(format!(
                    "Matern range must be positive, got {range}"
                )));
            }
            if *smoothness <= 0.0 || smoothness.is_nan() {
                return Err(StatsError::InvalidArgument(format!(
                    "Matern smoothness must be positive, got {smoothness}"
                )));
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Build the precision matrix Q for a given latent field type
///
/// The precision matrix Q is the inverse covariance matrix of the GMRF.
/// It is scaled by `scale` (typically the precision hyperparameter tau).
///
/// # Arguments
/// * `field_type` - The type of latent field
/// * `n` - Dimension of the field (number of nodes/time points)
/// * `scale` - Scale parameter (precision hyperparameter, must be positive)
///
/// # Returns
/// An n x n symmetric precision matrix
///
/// # Errors
/// - If n == 0
/// - If scale is non-positive
/// - If field parameters are invalid (see [`validate_field_params`])
/// - If ICAR adjacency references indices >= n
/// - If RW2 requires n < 3
pub fn build_precision_matrix(
    field_type: &LatentFieldType,
    n: usize,
    scale: f64,
) -> StatsResult<Array2<f64>> {
    if n == 0 {
        return Err(StatsError::InvalidArgument(
            "Field dimension n must be at least 1".to_string(),
        ));
    }
    if scale <= 0.0 || scale.is_nan() {
        return Err(StatsError::InvalidArgument(format!(
            "Scale parameter must be positive, got {scale}"
        )));
    }
    validate_field_params(field_type)?;

    match field_type {
        LatentFieldType::IID => build_iid_precision(n, scale),
        LatentFieldType::RW1 => build_rw1_precision(n, scale),
        LatentFieldType::RW2 => build_rw2_precision(n, scale),
        LatentFieldType::AR1 { phi } => build_ar1_precision(n, scale, *phi),
        LatentFieldType::ICAR { adjacency } => build_icar_precision(n, scale, adjacency),
        LatentFieldType::Matern { range, smoothness } => {
            build_matern_precision(n, scale, *range, *smoothness)
        }
        _ => Err(StatsError::NotImplementedError(
            "Unknown latent field type".to_string(),
        )),
    }
}

/// IID: Q = scale * I_n
fn build_iid_precision(n: usize, scale: f64) -> StatsResult<Array2<f64>> {
    let mut q = Array2::zeros((n, n));
    for i in 0..n {
        q[[i, i]] = scale;
    }
    Ok(q)
}

/// RW1: First-order random walk precision
///
/// Structure matrix R has entries:
///   R[i,i] = 2 for interior, 1 for boundary
///   R[i,i+1] = R[i+1,i] = -1
///
/// For n=1, returns scale * [[1]].
fn build_rw1_precision(n: usize, scale: f64) -> StatsResult<Array2<f64>> {
    let mut q = Array2::zeros((n, n));
    if n == 1 {
        q[[0, 0]] = scale;
        return Ok(q);
    }
    for i in 0..n {
        if i == 0 || i == n - 1 {
            q[[i, i]] = scale;
        } else {
            q[[i, i]] = 2.0 * scale;
        }
        if i + 1 < n {
            q[[i, i + 1]] = -scale;
            q[[i + 1, i]] = -scale;
        }
    }
    Ok(q)
}

/// RW2: Second-order random walk precision
///
/// Encodes second differences. The structure matrix has pentadiagonal pattern:
///   Band 0: [1, 5, 6, ..., 6, 5, 1]
///   Band 1: [-2, -4, ..., -4, -2]
///   Band 2: [1, 1, ..., 1]
///
/// Requires n >= 3 for a well-defined second-order structure.
fn build_rw2_precision(n: usize, scale: f64) -> StatsResult<Array2<f64>> {
    if n < 3 {
        return Err(StatsError::InvalidArgument(format!(
            "RW2 requires n >= 3, got n = {n}"
        )));
    }
    // Build RW2 structure matrix: Q = D2^T D2 where D2 is the second difference operator
    // D2 is (n-2) x n with rows [1, -2, 1]
    let mut q = Array2::zeros((n, n));
    for row in 0..(n - 2) {
        // Each row of D2 contributes D2[row,:]^T D2[row,:] to Q
        // D2[row, row] = 1, D2[row, row+1] = -2, D2[row, row+2] = 1
        let indices = [row, row + 1, row + 2];
        let coeffs = [1.0, -2.0, 1.0];
        for (a, &ia) in indices.iter().enumerate() {
            for (b, &ib) in indices.iter().enumerate() {
                q[[ia, ib]] += scale * coeffs[a] * coeffs[b];
            }
        }
    }
    Ok(q)
}

/// AR1: Autoregressive order 1 precision
///
/// For stationary AR(1) with parameter phi, the precision matrix is:
///   Q[0,0] = Q[n-1,n-1] = 1
///   Q[i,i] = 1 + phi^2  for 0 < i < n-1
///   Q[i,i+1] = Q[i+1,i] = -phi
///
/// Scaled by scale / (1 - phi^2) to normalize the marginal variance.
fn build_ar1_precision(n: usize, scale: f64, phi: f64) -> StatsResult<Array2<f64>> {
    let mut q = Array2::zeros((n, n));
    if n == 1 {
        q[[0, 0]] = scale;
        return Ok(q);
    }

    let marginal_scale = scale / (1.0 - phi * phi);
    for i in 0..n {
        if i == 0 || i == n - 1 {
            q[[i, i]] = marginal_scale;
        } else {
            q[[i, i]] = marginal_scale * (1.0 + phi * phi);
        }
        if i + 1 < n {
            q[[i, i + 1]] = -marginal_scale * phi;
            q[[i + 1, i]] = -marginal_scale * phi;
        }
    }
    Ok(q)
}

/// ICAR: Intrinsic conditional autoregressive precision (graph Laplacian)
///
/// Q = scale * (D - W) where D is the degree matrix and W is the adjacency matrix.
/// The adjacency list defines undirected edges.
fn build_icar_precision(
    n: usize,
    scale: f64,
    adjacency: &[(usize, usize)],
) -> StatsResult<Array2<f64>> {
    // Validate indices
    for &(i, j) in adjacency {
        if i >= n || j >= n {
            return Err(StatsError::InvalidArgument(format!(
                "ICAR adjacency index out of bounds: ({i}, {j}) for n = {n}"
            )));
        }
    }

    let mut q = Array2::zeros((n, n));
    for &(i, j) in adjacency {
        // Off-diagonal: W entries
        q[[i, j]] -= scale;
        q[[j, i]] -= scale;
        // Diagonal: degree contribution
        q[[i, i]] += scale;
        q[[j, j]] += scale;
    }
    Ok(q)
}

/// Matern: SPDE-based precision on a regular 1D grid
///
/// Uses the stochastic partial differential equation (SPDE) approach:
///   (kappa^2 - Delta)^{alpha/2} x = W
///
/// For smoothness nu, alpha = nu + 1/2 (in 1D, d=1, so alpha = nu + d/2 = nu + 0.5).
/// kappa = sqrt(8 * nu) / range.
///
/// The discretized operator on a regular grid produces a banded precision matrix.
/// For alpha=1 (nu=0.5): tridiagonal. For alpha=2 (nu=1.5): pentadiagonal, etc.
/// We implement up to alpha=2 (nu=1.5) explicitly and use the alpha=1 form otherwise.
fn build_matern_precision(
    n: usize,
    scale: f64,
    range: f64,
    smoothness: f64,
) -> StatsResult<Array2<f64>> {
    let kappa = (8.0 * smoothness).sqrt() / range;
    let kappa2 = kappa * kappa;
    let alpha = smoothness + 0.5;

    if alpha <= 1.0 + 1e-10 {
        // alpha ~ 1: (kappa^2 - Delta) discretized as tridiagonal
        // Diagonal: kappa^2 + 2/h^2 (interior), kappa^2 + 1/h^2 (boundary)
        // Off-diagonal: -1/h^2
        // On a unit grid, h = 1
        let h = 1.0;
        let h2_inv = 1.0 / (h * h);
        let mut q = Array2::zeros((n, n));
        for i in 0..n {
            if i == 0 || i == n - 1 {
                q[[i, i]] = scale * (kappa2 + h2_inv);
            } else {
                q[[i, i]] = scale * (kappa2 + 2.0 * h2_inv);
            }
            if i + 1 < n {
                q[[i, i + 1]] = -scale * h2_inv;
                q[[i + 1, i]] = -scale * h2_inv;
            }
        }
        Ok(q)
    } else {
        // alpha ~ 2: (kappa^2 - Delta)^2 = kappa^4 - 2*kappa^2*Delta + Delta^2
        // We compose: first build C = kappa^2 * I + L (where L is the 1D Laplacian),
        // then Q = scale * C^T * C
        let h = 1.0;
        let h2_inv = 1.0 / (h * h);

        // Build C = kappa^2 * I + L (tridiagonal)
        let mut c = Array2::zeros((n, n));
        for i in 0..n {
            if i == 0 || i == n - 1 {
                c[[i, i]] = kappa2 + h2_inv;
            } else {
                c[[i, i]] = kappa2 + 2.0 * h2_inv;
            }
            if i + 1 < n {
                c[[i, i + 1]] = -h2_inv;
                c[[i + 1, i]] = -h2_inv;
            }
        }

        // Q = scale * C^T * C
        let ct = c.t();
        let q = ct.dot(&c) * scale;
        Ok(q)
    }
}

/// Compute the Kronecker product of two matrices
///
/// For space-time models, the precision matrix is often Q_time (x) Q_space
/// (Kronecker product). This function computes A (x) B.
///
/// # Arguments
/// * `q1` - First matrix (m x m)
/// * `q2` - Second matrix (n x n)
///
/// # Returns
/// (m*n) x (m*n) Kronecker product matrix
pub fn kronecker_precision(q1: &Array2<f64>, q2: &Array2<f64>) -> Array2<f64> {
    let (m1, n1) = (q1.nrows(), q1.ncols());
    let (m2, n2) = (q2.nrows(), q2.ncols());
    let mut result = Array2::zeros((m1 * m2, n1 * n2));

    for i1 in 0..m1 {
        for j1 in 0..n1 {
            let val = q1[[i1, j1]];
            if val.abs() < 1e-30 {
                continue;
            }
            for i2 in 0..m2 {
                for j2 in 0..n2 {
                    result[[i1 * m2 + i2, j1 * n2 + j2]] = val * q2[[i2, j2]];
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_iid_precision() {
        let q = build_precision_matrix(&LatentFieldType::IID, 4, 2.0)
            .expect("IID precision should succeed");
        assert_eq!(q.nrows(), 4);
        assert_eq!(q.ncols(), 4);
        for i in 0..4 {
            assert!((q[[i, i]] - 2.0).abs() < 1e-12);
            for j in 0..4 {
                if i != j {
                    assert!(q[[i, j]].abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_rw1_precision_n5() {
        let q = build_precision_matrix(&LatentFieldType::RW1, 5, 1.0)
            .expect("RW1 precision should succeed");
        assert_eq!(q.nrows(), 5);
        // Boundary: q[0,0] = 1, q[4,4] = 1
        assert!((q[[0, 0]] - 1.0).abs() < 1e-12, "q[0,0] = {}", q[[0, 0]]);
        assert!((q[[4, 4]] - 1.0).abs() < 1e-12, "q[4,4] = {}", q[[4, 4]]);
        // Interior: q[1,1] = q[2,2] = q[3,3] = 2
        assert!((q[[1, 1]] - 2.0).abs() < 1e-12);
        assert!((q[[2, 2]] - 2.0).abs() < 1e-12);
        assert!((q[[3, 3]] - 2.0).abs() < 1e-12);
        // Off-diagonal
        assert!((q[[0, 1]] - (-1.0)).abs() < 1e-12);
        assert!((q[[1, 0]] - (-1.0)).abs() < 1e-12);
        assert!((q[[3, 4]] - (-1.0)).abs() < 1e-12);
        // Non-adjacent should be 0
        assert!(q[[0, 2]].abs() < 1e-12);
    }

    #[test]
    fn test_rw2_precision_n6() {
        let q = build_precision_matrix(&LatentFieldType::RW2, 6, 1.0)
            .expect("RW2 precision should succeed");
        assert_eq!(q.nrows(), 6);
        // RW2 structure = D2^T D2 where D2 is second-difference operator
        // Check symmetry
        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (q[[i, j]] - q[[j, i]]).abs() < 1e-12,
                    "Not symmetric at ({i},{j})"
                );
            }
        }
        // Check pentadiagonal: entries at distance > 2 should be zero
        for i in 0..6 {
            for j in 0..6 {
                if (i as isize - j as isize).unsigned_abs() > 2 {
                    assert!(
                        q[[i, j]].abs() < 1e-12,
                        "Non-zero at ({i},{j}): {}",
                        q[[i, j]]
                    );
                }
            }
        }
        // Known values for D2^T D2 with n=6:
        // Diagonal: [1, 5, 6, 6, 5, 1]
        assert!((q[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((q[[1, 1]] - 5.0).abs() < 1e-12);
        assert!((q[[2, 2]] - 6.0).abs() < 1e-12);
        assert!((q[[3, 3]] - 6.0).abs() < 1e-12);
        assert!((q[[4, 4]] - 5.0).abs() < 1e-12);
        assert!((q[[5, 5]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_ar1_phi_zero_equals_iid() {
        let n = 5;
        let scale = 3.0;
        let q_ar1 = build_precision_matrix(&LatentFieldType::AR1 { phi: 0.0 }, n, scale)
            .expect("AR1(0) should succeed");
        let q_iid =
            build_precision_matrix(&LatentFieldType::IID, n, scale).expect("IID should succeed");
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (q_ar1[[i, j]] - q_iid[[i, j]]).abs() < 1e-10,
                    "Mismatch at ({i},{j}): AR1={}, IID={}",
                    q_ar1[[i, j]],
                    q_iid[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_ar1_banding_phi09() {
        let n = 5;
        let q = build_precision_matrix(&LatentFieldType::AR1 { phi: 0.9 }, n, 1.0)
            .expect("AR1(0.9) should succeed");
        let phi = 0.9;
        let s = 1.0 / (1.0 - phi * phi);
        // Interior diagonal: s * (1 + phi^2)
        assert!((q[[2, 2]] - s * (1.0 + phi * phi)).abs() < 1e-10);
        // Boundary diagonal: s
        assert!((q[[0, 0]] - s).abs() < 1e-10);
        // Off-diagonal: -s * phi
        assert!((q[[0, 1]] - (-s * phi)).abs() < 1e-10);
        // Non-adjacent zeros
        assert!(q[[0, 2]].abs() < 1e-12);
    }

    #[test]
    fn test_icar_line_graph_4() {
        // Line graph: 0-1-2-3
        let adj = vec![(0, 1), (1, 2), (2, 3)];
        let q = build_precision_matrix(&LatentFieldType::ICAR { adjacency: adj }, 4, 1.0)
            .expect("ICAR line graph should succeed");
        // Graph Laplacian of a path on 4 nodes:
        // Degrees: [1, 2, 2, 1]
        assert!((q[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((q[[1, 1]] - 2.0).abs() < 1e-12);
        assert!((q[[2, 2]] - 2.0).abs() < 1e-12);
        assert!((q[[3, 3]] - 1.0).abs() < 1e-12);
        // Off-diag
        assert!((q[[0, 1]] - (-1.0)).abs() < 1e-12);
        assert!((q[[1, 2]] - (-1.0)).abs() < 1e-12);
        assert!(q[[0, 2]].abs() < 1e-12);
    }

    #[test]
    fn test_icar_triangle_graph() {
        // Triangle: 0-1, 1-2, 0-2
        let adj = vec![(0, 1), (1, 2), (0, 2)];
        let q = build_precision_matrix(&LatentFieldType::ICAR { adjacency: adj }, 3, 1.0)
            .expect("ICAR triangle should succeed");
        // All degrees = 2
        for i in 0..3 {
            assert!((q[[i, i]] - 2.0).abs() < 1e-12);
        }
        // All off-diagonal = -1
        assert!((q[[0, 1]] - (-1.0)).abs() < 1e-12);
        assert!((q[[0, 2]] - (-1.0)).abs() < 1e-12);
        assert!((q[[1, 2]] - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_matern_positive_semidefinite() {
        let q = build_precision_matrix(
            &LatentFieldType::Matern {
                range: 2.0,
                smoothness: 1.0,
            },
            8,
            1.0,
        )
        .expect("Matern should succeed");
        // Check symmetry
        for i in 0..8 {
            for j in 0..8 {
                assert!(
                    (q[[i, j]] - q[[j, i]]).abs() < 1e-12,
                    "Matern not symmetric at ({i},{j})"
                );
            }
        }
        // Check positive semi-definiteness via Gershgorin: each diagonal >= sum of |off-diagonal|
        for i in 0..8 {
            let off_sum: f64 = (0..8).filter(|&j| j != i).map(|j| q[[i, j]].abs()).sum();
            assert!(
                q[[i, i]] >= off_sum - 1e-10,
                "Gershgorin violated at row {i}: diag={}, off_sum={}",
                q[[i, i]],
                off_sum
            );
        }
    }

    #[test]
    fn test_invalid_ar1_phi() {
        let result = build_precision_matrix(&LatentFieldType::AR1 { phi: 1.0 }, 5, 1.0);
        assert!(result.is_err());
        let result2 = build_precision_matrix(&LatentFieldType::AR1 { phi: -1.5 }, 5, 1.0);
        assert!(result2.is_err());
    }

    #[test]
    fn test_empty_adjacency_icar() {
        let q = build_precision_matrix(&LatentFieldType::ICAR { adjacency: vec![] }, 3, 1.0)
            .expect("Empty ICAR should produce zero matrix");
        // With no edges, the Laplacian is the zero matrix
        for i in 0..3 {
            for j in 0..3 {
                assert!(q[[i, j]].abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_kronecker_product() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let k = kronecker_precision(&a, &b);
        assert_eq!(k.nrows(), 4);
        assert_eq!(k.ncols(), 4);
        // k[0,0] = a[0,0]*b[0,0] = 5
        assert!((k[[0, 0]] - 5.0).abs() < 1e-12);
        // k[0,1] = a[0,0]*b[0,1] = 6
        assert!((k[[0, 1]] - 6.0).abs() < 1e-12);
        // k[0,2] = a[0,1]*b[0,0] = 10
        assert!((k[[0, 2]] - 10.0).abs() < 1e-12);
        // k[3,3] = a[1,1]*b[1,1] = 32
        assert!((k[[3, 3]] - 32.0).abs() < 1e-12);
    }

    #[test]
    fn test_precision_matrix_symmetric() {
        let types: Vec<LatentFieldType> = vec![
            LatentFieldType::IID,
            LatentFieldType::RW1,
            LatentFieldType::AR1 { phi: 0.5 },
            LatentFieldType::ICAR {
                adjacency: vec![(0, 1), (1, 2), (2, 3)],
            },
            LatentFieldType::Matern {
                range: 1.0,
                smoothness: 0.5,
            },
        ];
        for ft in &types {
            let q = build_precision_matrix(ft, 5, 1.0)
                .unwrap_or_else(|_| panic!("Failed for {:?}", ft));
            for i in 0..5 {
                for j in 0..5 {
                    assert!(
                        (q[[i, j]] - q[[j, i]]).abs() < 1e-12,
                        "Not symmetric for {:?} at ({i},{j})",
                        ft
                    );
                }
            }
        }
    }

    #[test]
    fn test_scale_applied_correctly() {
        let scale = 3.5;
        let q1 = build_precision_matrix(&LatentFieldType::RW1, 4, 1.0).expect("RW1 scale=1");
        let q_s = build_precision_matrix(&LatentFieldType::RW1, 4, scale).expect("RW1 scaled");
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (q_s[[i, j]] - scale * q1[[i, j]]).abs() < 1e-12,
                    "Scale mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_zero_dimension_rejected() {
        let result = build_precision_matrix(&LatentFieldType::IID, 0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_negative_scale_rejected() {
        let result = build_precision_matrix(&LatentFieldType::IID, 3, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_icar_out_of_bounds_rejected() {
        let result = build_precision_matrix(
            &LatentFieldType::ICAR {
                adjacency: vec![(0, 5)],
            },
            3,
            1.0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_rw2_too_small_rejected() {
        let result = build_precision_matrix(&LatentFieldType::RW2, 2, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_matern_invalid_range() {
        let result = build_precision_matrix(
            &LatentFieldType::Matern {
                range: -1.0,
                smoothness: 1.0,
            },
            5,
            1.0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_icar_self_loop() {
        let result = validate_field_params(&LatentFieldType::ICAR {
            adjacency: vec![(1, 1)],
        });
        assert!(result.is_err());
    }
}
