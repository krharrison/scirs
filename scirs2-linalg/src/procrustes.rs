//! Procrustes analysis: finding optimal transformations between point sets.
//!
//! - **Orthogonal Procrustes**: find rotation R minimizing ||A - B R||_F, R^T R = I
//! - **Extended Procrustes**: rotation + scaling + translation
//! - **Oblique Procrustes**: find T minimizing ||A - B T||_F with diag(T^T T) = I
//! - **Generalized Procrustes analysis**: align multiple matrices simultaneously
//! - **Procrustes distance**: distance metric after optimal alignment

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Trait alias
// ---------------------------------------------------------------------------

/// Trait alias for float bounds in this module.
pub trait ProcFloat: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}
impl<T> ProcFloat for T where T: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}

// ---------------------------------------------------------------------------
// Helper: Frobenius norm
// ---------------------------------------------------------------------------

fn frob_norm<F: ProcFloat>(m: &Array2<F>) -> F {
    let mut acc = F::zero();
    for &v in m.iter() {
        acc += v * v;
    }
    acc.sqrt()
}

/// Column-wise mean, returning a 1-D array of length ncols.
fn col_mean<F: ProcFloat>(a: &ArrayView2<F>) -> Array1<F> {
    let nrows = a.shape()[0];
    let ncols = a.shape()[1];
    let n_f = F::from(nrows).unwrap_or(F::one());
    let mut mean = Array1::<F>::zeros(ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            mean[j] += a[[i, j]];
        }
    }
    for j in 0..ncols {
        mean[j] /= n_f;
    }
    mean
}

/// Center a matrix by subtracting column means; returns (centered, mean).
fn center_matrix<F: ProcFloat>(a: &ArrayView2<F>) -> (Array2<F>, Array1<F>) {
    let mean = col_mean(a);
    let nrows = a.shape()[0];
    let ncols = a.shape()[1];
    let mut centered = a.to_owned();
    for i in 0..nrows {
        for j in 0..ncols {
            centered[[i, j]] -= mean[j];
        }
    }
    (centered, mean)
}

// ===================================================================
// Orthogonal Procrustes
// ===================================================================

/// Result of the orthogonal Procrustes problem.
#[derive(Debug, Clone)]
pub struct OrthogonalProcrustesResult<F> {
    /// Orthogonal matrix R such that A ~ B R.
    pub rotation: Array2<F>,
    /// Frobenius norm of the residual ||A - B R||_F.
    pub residual: F,
}

/// Solve the orthogonal Procrustes problem: find R with R^T R = I
/// minimizing ||A - B R||_F.
///
/// Uses SVD of B^T A: if B^T A = U S V^T, then R = U V^T.
///
/// # Arguments
/// * `a` - Target matrix (n x p)
/// * `b` - Source matrix (n x p)
///
/// # Returns
/// Orthogonal matrix R (p x p) and the residual norm.
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::procrustes::orthogonal_procrustes;
///
/// let a = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
/// let b = array![[0.0, 1.0], [-1.0, 0.0], [-1.0, 1.0]]; // rotated version
/// let res = orthogonal_procrustes(&a.view(), &b.view()).expect("procrustes failed");
/// // R should be close to a 90-degree rotation
/// assert!(res.rotation.dot(&res.rotation.t())[[0, 0]] - 1.0 < 1e-8);
/// ```
pub fn orthogonal_procrustes<F: ProcFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
) -> LinalgResult<OrthogonalProcrustesResult<F>> {
    if a.shape() != b.shape() {
        return Err(LinalgError::ShapeError(
            "Matrices A and B must have the same shape".into(),
        ));
    }
    let p = a.shape()[1];

    // M = B^T A
    let m = b.t().dot(a);

    // SVD of M
    let (u, _s, vt) = crate::decomposition::svd(&m.view(), true, None)?;

    // R = U V^T
    let r = u.dot(&vt);

    // Ensure det(R) = +1 (proper rotation); if det < 0, flip last column of U
    let det_r = det_sign(&r);
    let rotation = if det_r < F::zero() {
        let mut u_fixed = u;
        for i in 0..p {
            u_fixed[[i, p - 1]] = -u_fixed[[i, p - 1]];
        }
        u_fixed.dot(&vt)
    } else {
        r
    };

    let residual_mat = a.to_owned() - b.dot(&rotation);
    let residual = frob_norm(&residual_mat);

    Ok(OrthogonalProcrustesResult { rotation, residual })
}

/// Quick determinant sign for a square matrix using LU.
fn det_sign<F: ProcFloat>(m: &Array2<F>) -> F {
    match crate::basic::det(&m.view(), None) {
        Ok(d) => d,
        Err(_) => F::zero(),
    }
}

// ===================================================================
// Extended Procrustes: rotation + scaling + translation
// ===================================================================

/// Result of the extended Procrustes problem.
#[derive(Debug, Clone)]
pub struct ExtendedProcrustesResult<F> {
    /// Orthogonal rotation matrix R (p x p).
    pub rotation: Array2<F>,
    /// Scaling factor s.
    pub scale: F,
    /// Translation vector t (length p).
    pub translation: Array1<F>,
    /// Frobenius residual after transformation.
    pub residual: F,
}

/// Solve the extended Procrustes problem: find R, s, t minimizing
/// ||A - (s B R + 1 t^T)||_F where R^T R = I.
///
/// Algorithm:
/// 1. Center both matrices
/// 2. Solve orthogonal Procrustes on centered matrices
/// 3. Compute optimal scale s = trace(A_c^T B_c R) / ||B_c||^2
/// 4. Compute translation t = mean(A) - s * R^T mean(B)
///
/// # Arguments
/// * `a` - Target matrix (n x p)
/// * `b` - Source matrix (n x p)
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::procrustes::extended_procrustes;
///
/// let a = array![[2.0_f64, 0.0], [0.0, 2.0], [2.0, 2.0]];
/// let b = array![[1.0_f64, 0.0], [0.0, 1.0], [1.0, 1.0]];
/// let res = extended_procrustes(&a.view(), &b.view()).expect("failed");
/// // scale should be approximately 2.0
/// assert!((res.scale - 2.0_f64).abs() < 0.5);
/// ```
pub fn extended_procrustes<F: ProcFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
) -> LinalgResult<ExtendedProcrustesResult<F>> {
    if a.shape() != b.shape() {
        return Err(LinalgError::ShapeError(
            "Matrices A and B must have the same shape".into(),
        ));
    }

    let (a_c, a_mean) = center_matrix(a);
    let (b_c, b_mean) = center_matrix(b);

    // Orthogonal Procrustes on centered data
    let proc_res = orthogonal_procrustes(&a_c.view(), &b_c.view())?;
    let r = &proc_res.rotation;

    // Optimal scale: s = trace(A_c^T B_c R) / ||B_c||_F^2
    let b_c_r = b_c.dot(r);
    let mut trace_val = F::zero();
    for i in 0..a_c.shape()[0] {
        for j in 0..a_c.shape()[1] {
            trace_val += a_c[[i, j]] * b_c_r[[i, j]];
        }
    }
    let b_c_norm_sq = {
        let mut acc = F::zero();
        for &v in b_c.iter() {
            acc += v * v;
        }
        acc
    };

    let scale = if b_c_norm_sq > F::epsilon() {
        trace_val / b_c_norm_sq
    } else {
        F::one()
    };

    // Translation: t = mean(A) - s * R^T mean(B)
    // Actually: A ~ s * B * R + ones * t^T
    // => t = mean_A - s * mean_B * R  (since mean of ones * t^T is t^T along each row)
    // Wait: need to be careful. The model is: each row a_i ~ s * b_i * R + t
    // mean_a = s * mean_b * R + t  =>  t = mean_a - s * mean_b * R
    let mean_b_r = {
        let p = b.shape()[1];
        let mut result = Array1::<F>::zeros(p);
        for j in 0..p {
            for k in 0..p {
                result[j] += b_mean[k] * r[[k, j]];
            }
        }
        result
    };
    let translation = &a_mean - &(&mean_b_r * scale);

    // Compute residual
    let n = a.shape()[0];
    let p = a.shape()[1];
    let mut transformed = Array2::<F>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            let mut val = translation[j];
            for k in 0..p {
                val += scale * b[[i, k]] * r[[k, j]];
            }
            transformed[[i, j]] = val;
        }
    }
    let residual_mat = a.to_owned() - &transformed;
    let residual = frob_norm(&residual_mat);

    Ok(ExtendedProcrustesResult {
        rotation: r.clone(),
        scale,
        translation,
        residual,
    })
}

// ===================================================================
// Oblique Procrustes
// ===================================================================

/// Result of the oblique Procrustes problem.
#[derive(Debug, Clone)]
pub struct ObliqueProcrustesResult<F> {
    /// Transformation matrix T (p x p) with unit-length columns.
    pub transform: Array2<F>,
    /// Frobenius residual ||A - B T||_F.
    pub residual: F,
    /// Number of iterations used.
    pub iterations: usize,
}

/// Solve the oblique Procrustes problem: find T minimizing ||A - B T||_F
/// subject to diag(T^T T) = I (columns of T have unit length).
///
/// Uses an alternating projection / normalized gradient descent.
///
/// # Arguments
/// * `a` - Target matrix (n x p)
/// * `b` - Source matrix (n x p)
/// * `max_iter` - Maximum iterations (default 200)
/// * `tol`      - Convergence tolerance (default 1e-10)
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::procrustes::oblique_procrustes;
///
/// let a = array![[1.0, 0.0], [0.0, 1.0]];
/// let b = array![[1.0, 0.0], [0.0, 1.0]];
/// let res = oblique_procrustes(&a.view(), &b.view(), None, None).expect("failed");
/// assert!(res.residual < 1e-6);
/// ```
pub fn oblique_procrustes<F: ProcFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    max_iter: Option<usize>,
    tol: Option<F>,
) -> LinalgResult<ObliqueProcrustesResult<F>> {
    if a.shape() != b.shape() {
        return Err(LinalgError::ShapeError(
            "Matrices A and B must have the same shape".into(),
        ));
    }
    let p = a.shape()[1];
    let max_it = max_iter.unwrap_or(200);
    let eps = tol.unwrap_or_else(|| F::from(1e-10).unwrap_or(F::epsilon()));

    // Initialize T as the least-squares solution projected onto oblique constraint
    // T_ls = (B^T B)^{-1} B^T A, then normalize columns
    let btb = b.t().dot(b);
    let bta = b.t().dot(a);
    let btb_inv = crate::inv(&btb.view(), None)?;
    let mut t_mat = btb_inv.dot(&bta);

    // Normalize columns to unit length
    normalize_columns(&mut t_mat);

    for iter in 0..max_it {
        let t_old = t_mat.clone();

        // Gradient step: T <- T - alpha * grad
        // grad = -2 B^T (A - B T)
        let residual_mat = a.to_owned() - b.dot(&t_mat);
        let grad = b.t().dot(&residual_mat) * F::from(-2.0).unwrap_or(-F::one() - F::one());

        // Step size via Barzilai-Borwein-like heuristic
        let grad_norm = frob_norm(&grad);
        let alpha = if grad_norm > F::epsilon() {
            F::from(0.01).unwrap_or(F::epsilon()) / grad_norm
        } else {
            F::zero()
        };

        t_mat = &t_mat - &(&grad * alpha);

        // Project: normalize columns
        normalize_columns(&mut t_mat);

        let diff = frob_norm(&(&t_mat - &t_old));
        if diff < eps {
            let res_mat = a.to_owned() - b.dot(&t_mat);
            return Ok(ObliqueProcrustesResult {
                transform: t_mat,
                residual: frob_norm(&res_mat),
                iterations: iter + 1,
            });
        }
    }

    let res_mat = a.to_owned() - b.dot(&t_mat);
    Ok(ObliqueProcrustesResult {
        transform: t_mat,
        residual: frob_norm(&res_mat),
        iterations: max_it,
    })
}

/// Normalize each column of a matrix to unit length.
fn normalize_columns<F: ProcFloat>(m: &mut Array2<F>) {
    let p = m.shape()[1];
    let nrows = m.shape()[0];
    for j in 0..p {
        let mut col_norm_sq = F::zero();
        for i in 0..nrows {
            col_norm_sq += m[[i, j]] * m[[i, j]];
        }
        let col_norm = col_norm_sq.sqrt();
        if col_norm > F::epsilon() {
            for i in 0..nrows {
                m[[i, j]] /= col_norm;
            }
        }
    }
}

// ===================================================================
// Generalized Procrustes Analysis (GPA)
// ===================================================================

/// Result of generalized Procrustes analysis.
#[derive(Debug, Clone)]
pub struct GeneralizedProcrustesResult<F> {
    /// Aligned matrices (one per input).
    pub aligned: Vec<Array2<F>>,
    /// Consensus (mean) shape.
    pub consensus: Array2<F>,
    /// Rotation matrices applied to each input.
    pub rotations: Vec<Array2<F>>,
    /// Disparity: sum of squared distances from consensus.
    pub disparity: F,
    /// Number of GPA iterations.
    pub iterations: usize,
}

/// Generalized Procrustes Analysis: simultaneously align k matrices.
///
/// Iteratively:
/// 1. Compute consensus shape (mean of aligned matrices)
/// 2. Align each matrix to the consensus via orthogonal Procrustes
/// 3. Repeat until convergence
///
/// # Arguments
/// * `matrices` - Slice of matrices, all with the same shape (n x p)
/// * `max_iter` - Maximum GPA iterations (default 100)
/// * `tol`      - Convergence tolerance on disparity change (default 1e-10)
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::procrustes::generalized_procrustes;
///
/// let m1 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// let m2 = array![[0.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]; // rotated
/// let m3 = array![[0.1, 0.0], [1.1, 0.0], [0.1, 1.0]]; // slight translation
/// let res = generalized_procrustes(&[m1.view(), m2.view(), m3.view()], None, None)
///     .expect("GPA failed");
/// assert!(res.disparity < 1.0); // should be small after alignment
/// ```
pub fn generalized_procrustes<F: ProcFloat>(
    matrices: &[ArrayView2<F>],
    max_iter: Option<usize>,
    tol: Option<F>,
) -> LinalgResult<GeneralizedProcrustesResult<F>> {
    if matrices.is_empty() {
        return Err(LinalgError::ValueError(
            "At least one matrix required".into(),
        ));
    }

    let k = matrices.len();
    let shape = matrices[0].shape();
    let n = shape[0];
    let p = shape[1];

    for (idx, m) in matrices.iter().enumerate() {
        if m.shape() != [n, p] {
            return Err(LinalgError::ShapeError(format!(
                "Matrix {idx} has shape {:?}, expected [{n}, {p}]",
                m.shape()
            )));
        }
    }

    let max_it = max_iter.unwrap_or(100);
    let eps = tol.unwrap_or_else(|| F::from(1e-10).unwrap_or(F::epsilon()));
    let k_f = F::from(k).unwrap_or(F::one());

    // Center each matrix
    let mut aligned: Vec<Array2<F>> = matrices.iter().map(|m| center_matrix(m).0).collect();

    // Scale to unit Frobenius norm
    for mat in &mut aligned {
        let nrm = frob_norm(mat);
        if nrm > F::epsilon() {
            *mat = &*mat * (F::one() / nrm);
        }
    }

    let mut rotations: Vec<Array2<F>> = (0..k).map(|_| Array2::<F>::eye(p)).collect();
    let mut prev_disparity = F::infinity();

    for iter in 0..max_it {
        // Compute consensus (mean of aligned matrices)
        let mut consensus = Array2::<F>::zeros((n, p));
        for mat in &aligned {
            consensus += mat;
        }
        consensus *= F::one() / k_f;

        // Align each matrix to consensus
        for i in 0..k {
            let proc_res = orthogonal_procrustes(&consensus.view(), &aligned[i].view())?;
            aligned[i] = aligned[i].dot(&proc_res.rotation);
            rotations[i] = rotations[i].dot(&proc_res.rotation);
        }

        // Compute disparity
        let mut disparity = F::zero();
        let mut consensus_new = Array2::<F>::zeros((n, p));
        for mat in &aligned {
            consensus_new += mat;
        }
        consensus_new *= F::one() / k_f;
        for mat in &aligned {
            let diff = mat - &consensus_new;
            let mut ss = F::zero();
            for &v in diff.iter() {
                ss += v * v;
            }
            disparity += ss;
        }

        let change = (prev_disparity - disparity).abs();
        prev_disparity = disparity;

        if change < eps {
            return Ok(GeneralizedProcrustesResult {
                aligned,
                consensus: consensus_new,
                rotations,
                disparity,
                iterations: iter + 1,
            });
        }
    }

    // Final consensus
    let mut consensus = Array2::<F>::zeros((n, p));
    for mat in &aligned {
        consensus += mat;
    }
    consensus *= F::one() / k_f;

    Ok(GeneralizedProcrustesResult {
        aligned,
        consensus,
        rotations,
        disparity: prev_disparity,
        iterations: max_it,
    })
}

// ===================================================================
// Procrustes distance
// ===================================================================

/// Compute the Procrustes distance between two matrices.
///
/// This is the Frobenius norm of the residual after optimal orthogonal
/// alignment: d(A, B) = min_{R: R^T R = I} ||A - B R||_F.
///
/// Both matrices are centered and scaled to unit norm before alignment.
///
/// # Arguments
/// * `a` - First matrix (n x p)
/// * `b` - Second matrix (n x p)
///
/// # Returns
/// The Procrustes distance (non-negative scalar).
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::procrustes::procrustes_distance;
///
/// let a = array![[1.0, 0.0], [0.0, 1.0]];
/// let b = array![[1.0, 0.0], [0.0, 1.0]];
/// let d = procrustes_distance(&a.view(), &b.view()).expect("distance failed");
/// assert!(d < 1e-10);
/// ```
pub fn procrustes_distance<F: ProcFloat>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<F> {
    if a.shape() != b.shape() {
        return Err(LinalgError::ShapeError(
            "Matrices must have the same shape".into(),
        ));
    }

    // Center
    let (a_c, _) = center_matrix(a);
    let (b_c, _) = center_matrix(b);

    // Scale to unit norm
    let norm_a = frob_norm(&a_c);
    let norm_b = frob_norm(&b_c);

    if norm_a < F::epsilon() || norm_b < F::epsilon() {
        return Ok(F::zero());
    }

    let a_scaled = &a_c * (F::one() / norm_a);
    let b_scaled = &b_c * (F::one() / norm_b);

    let res = orthogonal_procrustes(&a_scaled.view(), &b_scaled.view())?;
    Ok(res.residual)
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_orthogonal_procrustes_identity() {
        let a = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let b = a.clone();
        let res = orthogonal_procrustes(&a.view(), &b.view()).expect("failed");
        // R should be identity (or close)
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(res.rotation[[i, j]], expected, epsilon = 1e-6);
            }
        }
        assert!(res.residual < 1e-8);
    }

    #[test]
    fn test_orthogonal_procrustes_rotation() {
        // 90-degree rotation
        let a = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let r90 = array![[0.0, -1.0], [1.0, 0.0]];
        let b = a.dot(&r90);
        let res = orthogonal_procrustes(&a.view(), &b.view()).expect("failed");
        // R should be r90^T
        let reconstructed = b.dot(&res.rotation);
        for i in 0..3 {
            for j in 0..2 {
                assert_abs_diff_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_orthogonal_procrustes_orthogonality() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let b = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]];
        let res = orthogonal_procrustes(&a.view(), &b.view()).expect("failed");
        // R^T R = I
        let rtr = res.rotation.t().dot(&res.rotation);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(rtr[[i, j]], expected, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_extended_procrustes_scale() {
        let a = array![[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]];
        let b = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let res = extended_procrustes(&a.view(), &b.view()).expect("failed");
        assert!((res.scale - 2.0).abs() < 0.3);
        assert!(res.residual < 0.5);
    }

    #[test]
    fn test_extended_procrustes_identity() {
        let a = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let b = a.clone();
        let res = extended_procrustes(&a.view(), &b.view()).expect("failed");
        assert!((res.scale - 1.0).abs() < 0.1);
        assert!(res.residual < 1e-6);
    }

    #[test]
    fn test_oblique_procrustes_identity() {
        let a = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let b = a.clone();
        let res = oblique_procrustes(&a.view(), &b.view(), None, None).expect("failed");
        // Columns of T should have unit norm
        let p = 2;
        for j in 0..p {
            let mut col_norm_sq = 0.0;
            for i in 0..p {
                col_norm_sq += res.transform[[i, j]] * res.transform[[i, j]];
            }
            assert_abs_diff_eq!(col_norm_sq.sqrt(), 1.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_generalized_procrustes_identical() {
        let m = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let res =
            generalized_procrustes(&[m.view(), m.view(), m.view()], None, None).expect("failed");
        assert!(res.disparity < 1e-6);
        assert!(res.iterations <= 5);
    }

    #[test]
    fn test_generalized_procrustes_rotated() {
        let m1 = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let r90 = array![[0.0, -1.0], [1.0, 0.0]];
        let m2 = m1.dot(&r90);
        let res = generalized_procrustes(&[m1.view(), m2.view()], None, None).expect("failed");
        assert!(res.disparity < 1e-6);
    }

    #[test]
    fn test_procrustes_distance_identical() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let d = procrustes_distance(&a.view(), &a.view()).expect("failed");
        assert!(d < 1e-8);
    }

    #[test]
    fn test_procrustes_distance_symmetric() {
        // Use shapes that differ only by rotation for symmetric distance
        let a = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let r90 = array![[0.0, -1.0], [1.0, 0.0]];
        let b = a.dot(&r90);
        let d_ab = procrustes_distance(&a.view(), &b.view()).expect("failed");
        let d_ba = procrustes_distance(&b.view(), &a.view()).expect("failed");
        assert_abs_diff_eq!(d_ab, d_ba, epsilon = 1e-6);
    }

    #[test]
    fn test_procrustes_distance_nonnegative() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let d = procrustes_distance(&a.view(), &b.view()).expect("failed");
        assert!(d >= 0.0);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(orthogonal_procrustes(&a.view(), &b.view()).is_err());
    }
}
