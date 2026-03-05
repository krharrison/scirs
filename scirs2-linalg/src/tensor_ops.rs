//! Tensor operations for N-way arrays.
//!
//! - **CP/PARAFAC decomposition** via Alternating Least Squares (ALS)
//! - **Tucker decomposition** (HOSVD)
//! - **Tensor-matrix products** (n-mode product)
//! - **Tensor unfolding / matricization**
//! - **Khatri-Rao product** (column-wise Kronecker)
//! - **Tensor norms**
//!
//! Tensors are represented as a flat `Vec<F>` with explicit shape (dimensions).

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Trait alias
// ---------------------------------------------------------------------------

/// Float bounds used in this module.
pub trait TensorFloat: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}
impl<T> TensorFloat for T where T: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}

// ===================================================================
// Tensor type
// ===================================================================

/// A dense N-way tensor stored in row-major order.
#[derive(Debug, Clone)]
pub struct Tensor<F> {
    /// Flat data in row-major (C) order.
    pub data: Vec<F>,
    /// Dimensions (shape) of the tensor.
    pub shape: Vec<usize>,
}

impl<F: TensorFloat> Tensor<F> {
    /// Create a tensor from shape and data.
    pub fn new(shape: Vec<usize>, data: Vec<F>) -> LinalgResult<Self> {
        let total: usize = shape.iter().product();
        if data.len() != total {
            return Err(LinalgError::ShapeError(format!(
                "Data length {} does not match shape {:?} (total {})",
                data.len(),
                shape,
                total
            )));
        }
        Ok(Self { data, shape })
    }

    /// Create a zero tensor with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        Self {
            data: vec![F::zero(); total],
            shape,
        }
    }

    /// Number of dimensions (order) of the tensor.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Get element by multi-index.
    pub fn get(&self, idx: &[usize]) -> LinalgResult<F> {
        let flat = self.flat_index(idx)?;
        Ok(self.data[flat])
    }

    /// Set element by multi-index.
    pub fn set(&mut self, idx: &[usize], value: F) -> LinalgResult<()> {
        let flat = self.flat_index(idx)?;
        self.data[flat] = value;
        Ok(())
    }

    /// Convert multi-index to flat index (row-major).
    fn flat_index(&self, idx: &[usize]) -> LinalgResult<usize> {
        if idx.len() != self.shape.len() {
            return Err(LinalgError::IndexError(format!(
                "Index has {} dimensions but tensor has {}",
                idx.len(),
                self.shape.len()
            )));
        }
        let mut flat = 0usize;
        let mut stride = 1usize;
        for d in (0..self.shape.len()).rev() {
            if idx[d] >= self.shape[d] {
                return Err(LinalgError::IndexError(format!(
                    "Index {} out of bounds for dimension {} (size {})",
                    idx[d], d, self.shape[d]
                )));
            }
            flat += idx[d] * stride;
            stride *= self.shape[d];
        }
        Ok(flat)
    }

    /// Convert flat index to multi-index (row-major).
    fn multi_index(&self, mut flat: usize) -> Vec<usize> {
        let mut idx = vec![0usize; self.shape.len()];
        for d in (0..self.shape.len()).rev() {
            idx[d] = flat % self.shape[d];
            flat /= self.shape[d];
        }
        idx
    }
}

// ===================================================================
// Tensor unfolding (matricization)
// ===================================================================

/// Unfold (matricize) a tensor along mode `mode`.
///
/// The mode-n unfolding maps the tensor to a matrix where:
/// - Rows correspond to the n-th dimension
/// - Columns correspond to all other dimensions (in order)
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `mode`   - Mode to unfold along (0-indexed)
///
/// # Returns
/// Matrix of shape (I_mode, product of other dimensions)
///
/// # Example
/// ```
/// use scirs2_linalg::tensor_ops::{Tensor, unfold};
///
/// // 2x3x2 tensor
/// let t = Tensor::new(vec![2, 3, 2], (0..12).map(|x| x as f64).collect()).expect("valid input");
/// let m = unfold(&t, 0).expect("valid input");
/// assert_eq!(m.shape(), &[2, 6]);
/// ```
pub fn unfold<F: TensorFloat>(tensor: &Tensor<F>, mode: usize) -> LinalgResult<Array2<F>> {
    let ndim = tensor.ndim();
    if mode >= ndim {
        return Err(LinalgError::IndexError(format!(
            "Mode {mode} out of range for {ndim}-D tensor"
        )));
    }

    let i_mode = tensor.shape[mode];
    let other_size: usize = tensor
        .shape
        .iter()
        .enumerate()
        .filter(|&(d, _)| d != mode)
        .map(|(_, &s)| s)
        .product();

    let mut mat = Array2::<F>::zeros((i_mode, other_size));

    // For each element in the tensor, compute its position in the unfolded matrix
    for flat in 0..tensor.numel() {
        let idx = tensor.multi_index(flat);
        let row = idx[mode];

        // Column index: combine all other indices
        let mut col = 0usize;
        let mut col_stride = 1usize;
        for d in (0..ndim).rev() {
            if d == mode {
                continue;
            }
            col += idx[d] * col_stride;
            col_stride *= tensor.shape[d];
        }
        mat[[row, col]] = tensor.data[flat];
    }

    Ok(mat)
}

/// Fold a matrix back into a tensor (inverse of unfolding).
///
/// # Arguments
/// * `mat`   - Unfolded matrix (I_mode x product_of_others)
/// * `mode`  - Mode that was unfolded
/// * `shape` - Target tensor shape
pub fn fold<F: TensorFloat>(
    mat: &ArrayView2<F>,
    mode: usize,
    shape: &[usize],
) -> LinalgResult<Tensor<F>> {
    let ndim = shape.len();
    if mode >= ndim {
        return Err(LinalgError::IndexError(format!(
            "Mode {mode} out of range for {ndim}-D tensor"
        )));
    }
    if mat.shape()[0] != shape[mode] {
        return Err(LinalgError::ShapeError(
            "Matrix rows must equal shape[mode]".into(),
        ));
    }

    let total: usize = shape.iter().product();
    let mut data = vec![F::zero(); total];

    let mut tensor = Tensor {
        data: data.clone(),
        shape: shape.to_vec(),
    };

    for (flat, data_elem) in data.iter_mut().enumerate().take(total) {
        let idx = tensor.multi_index(flat);
        let row = idx[mode];
        let mut col = 0usize;
        let mut col_stride = 1usize;
        for d in (0..ndim).rev() {
            if d == mode {
                continue;
            }
            col += idx[d] * col_stride;
            col_stride *= shape[d];
        }
        *data_elem = mat[[row, col]];
    }
    tensor.data = data;
    Ok(tensor)
}

// ===================================================================
// N-mode product
// ===================================================================

/// Compute the n-mode product of a tensor with a matrix.
///
/// The n-mode product of tensor T (I_0 x ... x I_{N-1}) with matrix
/// M (J x I_n) results in a tensor of shape (I_0 x ... x J x ... x I_{N-1}).
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `matrix` - Matrix M (J x I_n)
/// * `mode`   - Mode along which to multiply
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::tensor_ops::{Tensor, n_mode_product};
///
/// let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid input");
/// let m = array![[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
/// let result = n_mode_product(&t, &m.view(), 1).expect("valid input");
/// assert_eq!(result.shape, vec![2, 2]);
/// ```
pub fn n_mode_product<F: TensorFloat>(
    tensor: &Tensor<F>,
    matrix: &ArrayView2<F>,
    mode: usize,
) -> LinalgResult<Tensor<F>> {
    let ndim = tensor.ndim();
    if mode >= ndim {
        return Err(LinalgError::IndexError(format!(
            "Mode {mode} out of range for {ndim}-D tensor"
        )));
    }
    if matrix.shape()[1] != tensor.shape[mode] {
        return Err(LinalgError::ShapeError(format!(
            "Matrix columns ({}) must equal tensor dimension {} (size {})",
            matrix.shape()[1],
            mode,
            tensor.shape[mode]
        )));
    }

    // Unfold along mode, multiply, fold back
    let unfolded = unfold(tensor, mode)?;
    let product = matrix.dot(&unfolded);

    let mut new_shape = tensor.shape.clone();
    new_shape[mode] = matrix.shape()[0];

    fold(&product.view(), mode, &new_shape)
}

// ===================================================================
// Khatri-Rao product
// ===================================================================

/// Compute the Khatri-Rao product (column-wise Kronecker) of two matrices.
///
/// Given A (I x R) and B (J x R), the result is a (I*J x R) matrix
/// where column r is the Kronecker product of `A[:,r]` and `B[:,r]`.
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::tensor_ops::khatri_rao;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let b = array![[5.0, 6.0], [7.0, 8.0]];
/// let kr = khatri_rao(&a.view(), &b.view()).expect("valid input");
/// assert_eq!(kr.shape(), &[4, 2]);
/// // Column 0: kron([1,3], [5,7]) = [5, 7, 15, 21]
/// assert!((kr[[0, 0]] - 5.0).abs() < 1e-10);
/// assert!((kr[[1, 0]] - 7.0).abs() < 1e-10);
/// assert!((kr[[2, 0]] - 15.0).abs() < 1e-10);
/// assert!((kr[[3, 0]] - 21.0).abs() < 1e-10);
/// ```
pub fn khatri_rao<F: TensorFloat>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
    let ra = a.shape()[1];
    let rb = b.shape()[1];
    if ra != rb {
        return Err(LinalgError::ShapeError(
            "Matrices must have the same number of columns for Khatri-Rao product".into(),
        ));
    }
    let ia = a.shape()[0];
    let jb = b.shape()[0];
    let r = ra;

    let mut result = Array2::<F>::zeros((ia * jb, r));
    for col in 0..r {
        for i in 0..ia {
            for j in 0..jb {
                result[[i * jb + j, col]] = a[[i, col]] * b[[j, col]];
            }
        }
    }
    Ok(result)
}

// ===================================================================
// Tensor norms
// ===================================================================

/// Compute the Frobenius norm of a tensor: sqrt(sum of squares).
///
/// # Example
/// ```
/// use scirs2_linalg::tensor_ops::{Tensor, tensor_norm_frobenius};
///
/// let t = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("valid input");
/// let nrm = tensor_norm_frobenius(&t);
/// assert!((nrm - (30.0_f64).sqrt()).abs() < 1e-10);
/// ```
pub fn tensor_norm_frobenius<F: TensorFloat>(tensor: &Tensor<F>) -> F {
    let mut acc = F::zero();
    for &v in &tensor.data {
        acc += v * v;
    }
    acc.sqrt()
}

/// Compute the max-abs norm of a tensor: max |x_i|.
pub fn tensor_norm_max<F: TensorFloat>(tensor: &Tensor<F>) -> F {
    let mut mx = F::zero();
    for &v in &tensor.data {
        let av = v.abs();
        if av > mx {
            mx = av;
        }
    }
    mx
}

/// Compute the L1 norm of a tensor: sum |x_i|.
pub fn tensor_norm_l1<F: TensorFloat>(tensor: &Tensor<F>) -> F {
    let mut acc = F::zero();
    for &v in &tensor.data {
        acc += v.abs();
    }
    acc
}

// ===================================================================
// CP/PARAFAC decomposition via ALS
// ===================================================================

/// Result of CP/PARAFAC decomposition.
#[derive(Debug, Clone)]
pub struct CpResult<F> {
    /// Factor matrices, one per mode. `factors[n]` has shape (I_n x R).
    pub factors: Vec<Array2<F>>,
    /// Weights (lambda) of each component (length R).
    pub weights: Array1<F>,
    /// Reconstruction error (Frobenius norm of residual).
    pub reconstruction_error: F,
    /// Number of ALS iterations.
    pub iterations: usize,
}

/// Compute the CP/PARAFAC decomposition of a tensor via ALS.
///
/// Decomposes a tensor T into a sum of R rank-1 tensors:
///   T ~ sum_{r=1}^{R} lambda_r * a^{(1)}_r o a^{(2)}_r o ... o a^{(N)}_r
///
/// # Arguments
/// * `tensor`  - Input tensor
/// * `rank`    - Number of components R
/// * `max_iter`- Maximum ALS iterations (default 200)
/// * `tol`     - Convergence tolerance (default 1e-8)
///
/// # Example
/// ```
/// use scirs2_linalg::tensor_ops::{Tensor, cp_als};
///
/// // Construct a simple rank-1 tensor
/// let data = vec![1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0];
/// let t = Tensor::new(vec![2, 2, 2], data).expect("valid input");
/// let res = cp_als(&t, 1, None, None).expect("valid input");
/// assert!(res.reconstruction_error < 1e-4);
/// ```
pub fn cp_als<F: TensorFloat>(
    tensor: &Tensor<F>,
    rank: usize,
    max_iter: Option<usize>,
    tol: Option<F>,
) -> LinalgResult<CpResult<F>> {
    let ndim = tensor.ndim();
    if ndim < 2 {
        return Err(LinalgError::ValueError(
            "Tensor must have at least 2 dimensions".into(),
        ));
    }
    if rank == 0 {
        return Err(LinalgError::ValueError("Rank must be > 0".into()));
    }

    let max_it = max_iter.unwrap_or(200);
    let eps = tol.unwrap_or_else(|| F::from(1e-8).unwrap_or(F::epsilon()));

    // Initialize factor matrices with sequential values (deterministic)
    let mut factors: Vec<Array2<F>> = Vec::with_capacity(ndim);
    for d in 0..ndim {
        let rows = tensor.shape[d];
        let mut mat = Array2::<F>::zeros((rows, rank));
        for i in 0..rows {
            for r in 0..rank {
                // Deterministic initialization
                let val = F::from((i + r + d + 1) as f64).unwrap_or(F::one())
                    / F::from((rows + rank) as f64).unwrap_or(F::one());
                mat[[i, r]] = val;
            }
        }
        factors.push(mat);
    }

    let tensor_norm = tensor_norm_frobenius(tensor);

    for iter in 0..max_it {
        for mode in 0..ndim {
            // Compute the Khatri-Rao product of all factors except current mode
            let kr = khatri_rao_all_except(&factors, mode)?;

            // Unfolding of tensor along this mode
            let x_unf = unfold(tensor, mode)?;

            // V = hadamard product of (A_j^T A_j) for j != mode
            let mut v = Array2::<F>::ones((rank, rank));
            for (j, factor_j) in factors.iter().enumerate().take(ndim) {
                if j == mode {
                    continue;
                }
                let ftf = factor_j.t().dot(factor_j);
                for r1 in 0..rank {
                    for r2 in 0..rank {
                        v[[r1, r2]] *= ftf[[r1, r2]];
                    }
                }
            }

            // Update: factors[mode] = X_(mode) * KR * V^{-1}
            let rhs = x_unf.dot(&kr);
            match crate::inv(&v.view(), None) {
                Ok(v_inv) => {
                    factors[mode] = rhs.dot(&v_inv);
                }
                Err(_) => {
                    // V is singular; add small regularization
                    let reg = F::from(1e-12).unwrap_or(F::epsilon());
                    let mut v_reg = v.clone();
                    for r in 0..rank {
                        v_reg[[r, r]] += reg;
                    }
                    let v_inv = crate::inv(&v_reg.view(), None)?;
                    factors[mode] = rhs.dot(&v_inv);
                }
            }
        }

        // Compute reconstruction error
        let reconstructed = reconstruct_cp(&factors, tensor)?;
        let mut err_sq = F::zero();
        for i in 0..tensor.numel() {
            let diff = tensor.data[i] - reconstructed.data[i];
            err_sq += diff * diff;
        }
        let err = err_sq.sqrt();

        if tensor_norm > F::epsilon() {
            let rel_err = err / tensor_norm;
            if rel_err < eps {
                let (normed_factors, weights) = normalize_cp_factors(&factors);
                return Ok(CpResult {
                    factors: normed_factors,
                    weights,
                    reconstruction_error: err,
                    iterations: iter + 1,
                });
            }
        }
    }

    let reconstructed = reconstruct_cp(&factors, tensor)?;
    let mut err_sq = F::zero();
    for i in 0..tensor.numel() {
        let diff = tensor.data[i] - reconstructed.data[i];
        err_sq += diff * diff;
    }

    let (normed_factors, weights) = normalize_cp_factors(&factors);
    Ok(CpResult {
        factors: normed_factors,
        weights,
        reconstruction_error: err_sq.sqrt(),
        iterations: max_it,
    })
}

/// Khatri-Rao product of all factor matrices except one mode.
fn khatri_rao_all_except<F: TensorFloat>(
    factors: &[Array2<F>],
    skip: usize,
) -> LinalgResult<Array2<F>> {
    let ndim = factors.len();
    let mut result: Option<Array2<F>> = None;
    for (d, factor) in factors.iter().enumerate().take(ndim) {
        if d == skip {
            continue;
        }
        result = Some(match result {
            None => factor.clone(),
            Some(prev) => khatri_rao(&prev.view(), &factor.view())?,
        });
    }
    result.ok_or_else(|| LinalgError::ValueError("No factors to combine".into()))
}

/// Reconstruct a tensor from CP factors (without weights, they are absorbed).
fn reconstruct_cp<F: TensorFloat>(
    factors: &[Array2<F>],
    reference: &Tensor<F>,
) -> LinalgResult<Tensor<F>> {
    let ndim = factors.len();
    let rank = factors[0].shape()[1];
    let total = reference.numel();
    let shape = &reference.shape;

    let mut data = vec![F::zero(); total];

    for (flat, item) in data.iter_mut().enumerate().take(total) {
        let idx = reference.multi_index(flat);
        let mut val = F::zero();
        for r in 0..rank {
            let mut prod = F::one();
            for d in 0..ndim {
                prod *= factors[d][[idx[d], r]];
            }
            val += prod;
        }
        *item = val;
    }

    Ok(Tensor {
        data,
        shape: shape.clone(),
    })
}

/// Normalize CP factors: extract column norms as weights.
fn normalize_cp_factors<F: TensorFloat>(factors: &[Array2<F>]) -> (Vec<Array2<F>>, Array1<F>) {
    let rank = factors[0].shape()[1];
    let mut weights = Array1::<F>::ones(rank);
    let mut normed: Vec<Array2<F>> = factors.to_vec();

    for factor in &mut normed {
        let rows = factor.shape()[0];
        for r in 0..rank {
            let mut col_norm_sq = F::zero();
            for i in 0..rows {
                col_norm_sq += factor[[i, r]] * factor[[i, r]];
            }
            let col_norm = col_norm_sq.sqrt();
            if col_norm > F::epsilon() {
                weights[r] *= col_norm;
                for i in 0..rows {
                    factor[[i, r]] /= col_norm;
                }
            }
        }
    }

    (normed, weights)
}

// ===================================================================
// Tucker decomposition (HOSVD)
// ===================================================================

/// Result of the Tucker decomposition.
#[derive(Debug, Clone)]
pub struct TuckerResult<F> {
    /// Core tensor G.
    pub core: Tensor<F>,
    /// Factor matrices, one per mode. `factors[n]` has shape (I_n x R_n).
    pub factors: Vec<Array2<F>>,
    /// Reconstruction error (Frobenius norm).
    pub reconstruction_error: F,
}

/// Compute the Tucker decomposition via HOSVD (Higher-Order SVD).
///
/// Decomposes tensor T into a core tensor G and factor matrices U_n:
///   T ~ G x_1 U_1 x_2 U_2 x_3 ... x_N U_N
///
/// The HOSVD computes each U_n from the SVD of the mode-n unfolding.
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `ranks`  - Target rank for each mode (length must equal ndim)
///
/// # Example
/// ```
/// use scirs2_linalg::tensor_ops::{Tensor, tucker_hosvd};
///
/// let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
/// let t = Tensor::new(vec![2, 3, 4], data).expect("valid input");
/// let res = tucker_hosvd(&t, &[2, 2, 2]).expect("valid input");
/// assert_eq!(res.core.shape, vec![2, 2, 2]);
/// assert_eq!(res.factors[0].shape(), &[2, 2]);
/// assert_eq!(res.factors[1].shape(), &[3, 2]);
/// assert_eq!(res.factors[2].shape(), &[4, 2]);
/// ```
pub fn tucker_hosvd<F: TensorFloat>(
    tensor: &Tensor<F>,
    ranks: &[usize],
) -> LinalgResult<TuckerResult<F>> {
    let ndim = tensor.ndim();
    if ranks.len() != ndim {
        return Err(LinalgError::ShapeError(format!(
            "ranks length {} must equal tensor ndim {}",
            ranks.len(),
            ndim
        )));
    }

    // Compute factor matrices via truncated SVD of each mode-n unfolding
    let mut factors: Vec<Array2<F>> = Vec::with_capacity(ndim);
    for (mode, &rank) in ranks.iter().enumerate().take(ndim) {
        let x_n = unfold(tensor, mode)?;
        let (u, _s, _vt) = crate::decomposition::svd(&x_n.view(), true, None)?;

        // Truncate to ranks[mode] columns
        let r = rank.min(u.shape()[1]);
        let u_trunc = u.slice(scirs2_core::ndarray::s![.., ..r]).to_owned();
        factors.push(u_trunc);
    }

    // Core tensor: G = T x_1 U_1^T x_2 U_2^T x_3 ... x_N U_N^T
    let mut core_tensor = tensor.clone();
    for (mode, factor) in factors.iter().enumerate().take(ndim) {
        let u_t = factor.t().to_owned();
        core_tensor = n_mode_product(&core_tensor, &u_t.view(), mode)?;
    }

    // Reconstruction error
    let reconstructed = reconstruct_tucker(&core_tensor, &factors)?;
    let mut err_sq = F::zero();
    for i in 0..tensor.numel() {
        let diff = tensor.data[i] - reconstructed.data[i];
        err_sq += diff * diff;
    }

    Ok(TuckerResult {
        core: core_tensor,
        factors,
        reconstruction_error: err_sq.sqrt(),
    })
}

/// Reconstruct tensor from Tucker decomposition.
fn reconstruct_tucker<F: TensorFloat>(
    core: &Tensor<F>,
    factors: &[Array2<F>],
) -> LinalgResult<Tensor<F>> {
    let mut result = core.clone();
    for (mode, factor) in factors.iter().enumerate() {
        result = n_mode_product(&result, &factor.view(), mode)?;
    }
    Ok(result)
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    // --- Tensor basics ---

    #[test]
    fn test_tensor_create() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(t.is_ok());
        let t = t.expect("create ok");
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn test_tensor_get_set() {
        let mut t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("ok");
        assert_abs_diff_eq!(t.get(&[0, 0]).expect("ok"), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(t.get(&[1, 2]).expect("ok"), 6.0, epsilon = 1e-10);
        t.set(&[0, 1], 99.0).expect("ok");
        assert_abs_diff_eq!(t.get(&[0, 1]).expect("ok"), 99.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tensor_shape_mismatch() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0]);
        assert!(t.is_err());
    }

    // --- Unfolding ---

    #[test]
    fn test_unfold_2d() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("ok");
        let m0 = unfold(&t, 0).expect("ok");
        assert_eq!(m0.shape(), &[2, 3]);
        // Mode-0 unfolding of a matrix is the matrix itself
        assert_abs_diff_eq!(m0[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(m0[[1, 2]], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_unfold_3d() {
        // 2x3x2 tensor
        let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let t = Tensor::new(vec![2, 3, 2], data).expect("ok");
        let m0 = unfold(&t, 0).expect("ok");
        assert_eq!(m0.shape(), &[2, 6]);
        let m1 = unfold(&t, 1).expect("ok");
        assert_eq!(m1.shape(), &[3, 4]);
        let m2 = unfold(&t, 2).expect("ok");
        assert_eq!(m2.shape(), &[2, 6]);
    }

    #[test]
    fn test_fold_unfold_roundtrip() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let t = Tensor::new(vec![2, 3, 4], data.clone()).expect("ok");
        for mode in 0..3 {
            let mat = unfold(&t, mode).expect("ok");
            let t2 = fold(&mat.view(), mode, &[2, 3, 4]).expect("ok");
            for i in 0..24 {
                assert_abs_diff_eq!(t.data[i], t2.data[i], epsilon = 1e-10);
            }
        }
    }

    // --- N-mode product ---

    #[test]
    fn test_n_mode_product_2d() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("ok");
        let m = array![[1.0, 0.0], [0.0, 1.0]]; // 2x2 identity
        let res = n_mode_product(&t, &m.view(), 0).expect("ok");
        assert_eq!(res.shape, vec![2, 3]);
        for i in 0..6 {
            assert_abs_diff_eq!(res.data[i], t.data[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_n_mode_product_reduces() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("ok");
        // Sum rows: [1x2] * [2x3] = [1x3]
        let m = array![[1.0, 1.0]]; // 1x2
        let res = n_mode_product(&t, &m.view(), 0).expect("ok");
        assert_eq!(res.shape, vec![1, 3]);
        assert_abs_diff_eq!(res.data[0], 5.0, epsilon = 1e-10); // 1+4
        assert_abs_diff_eq!(res.data[1], 7.0, epsilon = 1e-10); // 2+5
        assert_abs_diff_eq!(res.data[2], 9.0, epsilon = 1e-10); // 3+6
    }

    // --- Khatri-Rao product ---

    #[test]
    fn test_khatri_rao_basic() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let kr = khatri_rao(&a.view(), &b.view()).expect("ok");
        assert_eq!(kr.shape(), &[4, 2]);
        // Col 0: kron([1,3], [5,7]) = [5, 7, 15, 21]
        assert_abs_diff_eq!(kr[[0, 0]], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(kr[[1, 0]], 7.0, epsilon = 1e-10);
        assert_abs_diff_eq!(kr[[2, 0]], 15.0, epsilon = 1e-10);
        assert_abs_diff_eq!(kr[[3, 0]], 21.0, epsilon = 1e-10);
    }

    #[test]
    fn test_khatri_rao_mismatch() {
        let a = array![[1.0, 2.0]]; // 1x2
        let b = array![[1.0, 2.0, 3.0]]; // 1x3
        assert!(khatri_rao(&a.view(), &b.view()).is_err());
    }

    // --- Tensor norms ---

    #[test]
    fn test_tensor_norm_frobenius() {
        let t = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("ok");
        let nrm = tensor_norm_frobenius(&t);
        assert_abs_diff_eq!(nrm, (30.0_f64).sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_tensor_norm_max() {
        let t = Tensor::new(vec![2, 2], vec![1.0, -5.0, 3.0, 4.0]).expect("ok");
        assert_abs_diff_eq!(tensor_norm_max(&t), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tensor_norm_l1() {
        let t = Tensor::new(vec![2, 2], vec![1.0, -2.0, 3.0, -4.0]).expect("ok");
        assert_abs_diff_eq!(tensor_norm_l1(&t), 10.0, epsilon = 1e-10);
    }

    // --- CP/PARAFAC ---

    #[test]
    fn test_cp_als_rank1() {
        // Rank-1 tensor: outer product of [1,2] and [3,4,5]
        let data = vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0];
        let t = Tensor::new(vec![2, 3], data).expect("ok");
        let res = cp_als(&t, 1, Some(100), Some(1e-6)).expect("ok");
        assert!(res.reconstruction_error < 1.0);
    }

    #[test]
    fn test_cp_als_3d() {
        // Rank-1 3D tensor: outer product of [1,2], [1,1], [1,1]
        let data = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let t = Tensor::new(vec![2, 2, 2], data).expect("ok");
        let res = cp_als(&t, 1, None, None).expect("ok");
        assert!(res.reconstruction_error < 0.5);
    }

    // --- Tucker/HOSVD ---

    #[test]
    fn test_tucker_hosvd_basic() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let t = Tensor::new(vec![2, 3, 4], data).expect("ok");
        let res = tucker_hosvd(&t, &[2, 2, 2]).expect("ok");
        assert_eq!(res.core.shape, vec![2, 2, 2]);
        assert_eq!(res.factors.len(), 3);
    }

    #[test]
    fn test_tucker_hosvd_full_rank() {
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let t = Tensor::new(vec![2, 3, 2], data).expect("ok");
        let res = tucker_hosvd(&t, &[2, 3, 2]).expect("ok");
        // Full rank should give near-zero reconstruction error
        assert!(res.reconstruction_error < 1e-6);
    }

    #[test]
    fn test_tucker_rank_mismatch() {
        let t = Tensor::new(vec![2, 3], vec![1.0; 6]).expect("ok");
        assert!(tucker_hosvd(&t, &[2]).is_err()); // wrong number of ranks
    }
}
