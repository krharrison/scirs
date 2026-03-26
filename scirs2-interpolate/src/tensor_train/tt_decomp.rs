//! TT-SVD decomposition for tensor-train format.
//!
//! Implements the TT-SVD algorithm from Oseledets 2011:
//! "Tensor-Train Decomposition", SIAM J. Sci. Comput., 33(5), 2295-2317.

use crate::error::InterpolateError;
use scirs2_core::ndarray::{Array2, Array3, ArrayD, IxDyn};

/// Tensor-train representation of a d-dimensional array.
///
/// `A[i1,...,id] ≈ G1[i1] * G2[i2] * ... * Gd[id]`
///
/// where each core `Gk` has shape `[r_{k-1}, n_k, r_k]` and the boundary ranks
/// satisfy `r_0 = r_d = 1`.
#[derive(Debug, Clone)]
pub struct TensorTrain {
    /// Cores: `cores[k]` has shape `[r_{k-1}, n_k, r_k]`.
    pub cores: Vec<Array3<f64>>,
    /// Mode sizes `n_1, ..., n_d`.
    pub shape: Vec<usize>,
    /// Ranks `r_0=1, r_1, ..., r_{d-1}, r_d=1`.
    pub ranks: Vec<usize>,
}

impl TensorTrain {
    /// Create a TT from pre-built cores.
    ///
    /// Validates that shapes are consistent.
    pub fn new(cores: Vec<Array3<f64>>) -> Result<Self, InterpolateError> {
        if cores.is_empty() {
            return Err(InterpolateError::InvalidInput {
                message: "TensorTrain requires at least one core".into(),
            });
        }
        let d = cores.len();
        let mut shape = Vec::with_capacity(d);
        let mut ranks = Vec::with_capacity(d + 1);
        ranks.push(cores[0].shape()[0]);
        for (k, core) in cores.iter().enumerate() {
            let s = core.shape();
            if s.len() != 3 {
                return Err(InterpolateError::InvalidInput {
                    message: format!("Core {k} must be a 3-D array, got {}D", s.len()),
                });
            }
            let prev_rank = *ranks.last().ok_or_else(|| InterpolateError::InvalidInput {
                message: "Internal rank mismatch".into(),
            })?;
            if k > 0 && s[0] != prev_rank {
                return Err(InterpolateError::InvalidInput {
                    message: format!(
                        "Left rank of core {k} ({}) does not match right rank of core {} ({})",
                        s[0],
                        k - 1,
                        prev_rank,
                    ),
                });
            }
            shape.push(s[1]);
            ranks.push(s[2]);
        }
        if ranks[0] != 1
            || *ranks.last().ok_or_else(|| InterpolateError::InvalidInput {
                message: "Empty ranks".into(),
            })? != 1
        {
            return Err(InterpolateError::InvalidInput {
                message: format!(
                    "Boundary ranks must be 1, got r_0={} and r_d={}",
                    ranks[0], ranks[d]
                ),
            });
        }
        Ok(Self {
            cores,
            shape,
            ranks,
        })
    }

    /// Evaluate the TT at a multi-index `idx = [i1, ..., id]`.
    ///
    /// Returns the scalar `G1[:,i1,:] * G2[:,i2,:] * ... * Gd[:,id,:]` contracted
    /// to a single `f64`.
    pub fn eval(&self, idx: &[usize]) -> Result<f64, InterpolateError> {
        let d = self.cores.len();
        if idx.len() != d {
            return Err(InterpolateError::DimensionMismatch(format!(
                "idx has length {}, expected {d}",
                idx.len()
            )));
        }
        for (k, (&ik, &nk)) in idx.iter().zip(self.shape.iter()).enumerate() {
            if ik >= nk {
                return Err(InterpolateError::OutOfBounds(format!(
                    "Index {ik} out of range [0, {nk}) in dimension {k}"
                )));
            }
        }

        // Carry a row-vector of shape [1, r_k]; start with [1.0] (r_0=1).
        let mut v = vec![1.0f64];
        for (k, &ik) in idx.iter().enumerate() {
            let core = &self.cores[k];
            let (r_left, _n, r_right) = (core.shape()[0], core.shape()[1], core.shape()[2]);
            // Slice: G_k[:, ik, :] => shape [r_left, r_right]
            let mut new_v = vec![0.0f64; r_right];
            for j in 0..r_right {
                let mut s = 0.0f64;
                for i in 0..r_left {
                    s += v[i] * core[[i, ik, j]];
                }
                new_v[j] = s;
            }
            v = new_v;
        }
        // v should have length 1 (r_d = 1)
        Ok(v[0])
    }

    /// Reconstruct the full dense tensor.
    ///
    /// **Warning**: exponential cost in `d` — only practical for small tensors.
    pub fn to_dense(&self) -> Result<ArrayD<f64>, InterpolateError> {
        let d = self.cores.len();
        let total: usize = self.shape.iter().product();
        let mut data = vec![0.0f64; total];

        // Iterate over all multi-indices
        let mut idx = vec![0usize; d];
        loop {
            let flat = row_major_index(&idx, &self.shape);
            data[flat] = self.eval(&idx)?;

            // Increment multi-index (last dimension varies fastest)
            let mut carry = true;
            for k in (0..d).rev() {
                if carry {
                    idx[k] += 1;
                    if idx[k] >= self.shape[k] {
                        idx[k] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry {
                break; // all combinations exhausted
            }
        }

        ArrayD::from_shape_vec(IxDyn(&self.shape), data)
            .map_err(|e| InterpolateError::ComputationError(format!("to_dense shape error: {e}")))
    }

    /// Compute the Frobenius norm of the TT tensor via a transfer-matrix approach.
    pub fn norm(&self) -> f64 {
        let d = self.cores.len();
        if d == 0 {
            return 0.0;
        }
        // Build Gram matrix G of shape [1,1] = [[1]], then transfer left to right.
        let mut gram = Array2::<f64>::eye(1);

        for core in &self.cores {
            let (r_left, n, r_right) = (core.shape()[0], core.shape()[1], core.shape()[2]);
            let mut new_gram = Array2::<f64>::zeros((r_right, r_right));
            for ik in 0..n {
                for beta1 in 0..r_right {
                    for beta2 in 0..r_right {
                        let mut s = 0.0f64;
                        for alpha1 in 0..r_left {
                            for alpha2 in 0..r_left {
                                s += core[[alpha1, ik, beta1]]
                                    * gram[[alpha1, alpha2]]
                                    * core[[alpha2, ik, beta2]];
                            }
                        }
                        new_gram[[beta1, beta2]] += s;
                    }
                }
            }
            gram = new_gram;
        }
        // gram is now 1x1
        gram[[0, 0]].max(0.0).sqrt()
    }

    /// Number of parameters (stored f64 elements) in the TT format.
    pub fn n_params(&self) -> usize {
        self.cores.iter().map(|c| c.len()).sum()
    }

    /// Convenience: build from a dense tensor with TT-SVD.
    pub fn from_dense(
        tensor: &ArrayD<f64>,
        max_rank: usize,
        tol: f64,
    ) -> Result<Self, InterpolateError> {
        tt_svd(tensor, max_rank, tol)
    }
}

// ─── Row-major flat index ─────────────────────────────────────────────────────

fn row_major_index(idx: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for k in (0..idx.len()).rev() {
        flat += idx[k] * stride;
        stride *= shape[k];
    }
    flat
}

// ─── Truncated SVD via deflation + power iteration ────────────────────────────

/// Compute a rank-`r` truncated SVD of an `m x n` matrix.
///
/// Uses sequential deflation with power-iteration to extract singular triplets.
/// Returns `(U, s, Vt)` where `U` is `m x r`, `s` is `r`, `Vt` is `r x n`.
/// All singular values are non-negative; only those >= `tol * sigma_max` are kept
/// (and at most `max_rank`).
pub fn truncated_svd(
    a: &Array2<f64>,
    max_rank: usize,
    tol: f64,
) -> Result<(Array2<f64>, Vec<f64>, Array2<f64>), InterpolateError> {
    let m = a.nrows();
    let n = a.ncols();
    if m == 0 || n == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "truncated_svd: matrix must have positive dimensions".into(),
        });
    }
    let r_max = max_rank.min(m).min(n);
    let mat_data: Vec<f64> = a.iter().copied().collect();
    let (u_data, s_vals, vt_data) = svd_deflation(&mat_data, m, n, r_max, tol);

    let r = s_vals.len();
    let u_arr = Array2::from_shape_vec((m, r), u_data)
        .map_err(|e| InterpolateError::ComputationError(format!("SVD U shape error: {e}")))?;
    let vt_arr = Array2::from_shape_vec((r, n), vt_data)
        .map_err(|e| InterpolateError::ComputationError(format!("SVD Vt shape error: {e}")))?;
    Ok((u_arr, s_vals, vt_arr))
}

/// Extract singular triplets one by one via power-iteration + deflation.
///
/// Guarantees non-negative singular values and consistent U/Vt.
fn svd_deflation(
    a: &[f64],
    m: usize,
    n: usize,
    max_rank: usize,
    tol: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Compute Frobenius norm to determine threshold
    let frob_sq: f64 = a.iter().map(|x| x * x).sum();
    let sigma_ref = frob_sq.sqrt();
    if sigma_ref < 1e-300 {
        let mut u0 = vec![0.0f64; m];
        let mut v0 = vec![0.0f64; n];
        if m > 0 {
            u0[0] = 1.0;
        }
        if n > 0 {
            v0[0] = 1.0;
        }
        return (u0, vec![0.0], v0);
    }
    let threshold = tol * sigma_ref;

    let mut residual = a.to_vec();
    let mut u_cols: Vec<Vec<f64>> = Vec::new();
    let mut s_vals: Vec<f64> = Vec::new();
    let mut v_rows: Vec<Vec<f64>> = Vec::new();

    for _rank in 0..max_rank {
        // Initialise v as the row of residual with largest norm
        let mut vk: Vec<f64> = vec![0.0f64; n];
        let mut best_norm = 0.0f64;
        for i in 0..m {
            let row_norm: f64 = (0..n)
                .map(|j| residual[i * n + j] * residual[i * n + j])
                .sum::<f64>()
                .sqrt();
            if row_norm > best_norm {
                best_norm = row_norm;
                for j in 0..n {
                    vk[j] = residual[i * n + j];
                }
            }
        }
        let vnorm: f64 = vk.iter().map(|x| x * x).sum::<f64>().sqrt();
        if vnorm < 1e-300 {
            break;
        }
        for x in vk.iter_mut() {
            *x /= vnorm;
        }

        // Power iteration: alternate between A*v and A^T*u
        let mut uk = vec![0.0f64; m];
        for _iter in 0..20 {
            // uk = A * vk
            for i in 0..m {
                let mut s = 0.0f64;
                for j in 0..n {
                    s += residual[i * n + j] * vk[j];
                }
                uk[i] = s;
            }
            let unorm: f64 = uk.iter().map(|x| x * x).sum::<f64>().sqrt();
            if unorm < 1e-300 {
                break;
            }
            for x in uk.iter_mut() {
                *x /= unorm;
            }

            // vk = A^T * uk
            let mut new_vk = vec![0.0f64; n];
            for j in 0..n {
                let mut s = 0.0f64;
                for i in 0..m {
                    s += residual[i * n + j] * uk[i];
                }
                new_vk[j] = s;
            }
            let new_vnorm: f64 = new_vk.iter().map(|x| x * x).sum::<f64>().sqrt();
            if new_vnorm < 1e-300 {
                break;
            }
            // Check convergence
            let diff: f64 = new_vk
                .iter()
                .zip(vk.iter())
                .map(|(a, b)| (a / new_vnorm - b).powi(2))
                .sum::<f64>()
                .sqrt();
            for j in 0..n {
                vk[j] = new_vk[j] / new_vnorm;
            }
            if diff < 1e-12 {
                break;
            }
        }

        // Compute sigma = ||A * vk||
        let mut uk_final = vec![0.0f64; m];
        for i in 0..m {
            let mut s = 0.0f64;
            for j in 0..n {
                s += residual[i * n + j] * vk[j];
            }
            uk_final[i] = s;
        }
        let sigma: f64 = uk_final.iter().map(|x| x * x).sum::<f64>().sqrt();
        if sigma < threshold {
            break;
        }
        for x in uk_final.iter_mut() {
            *x /= sigma;
        }

        // Deflate: residual -= sigma * uk * vk^T
        for i in 0..m {
            for j in 0..n {
                residual[i * n + j] -= sigma * uk_final[i] * vk[j];
            }
        }

        u_cols.push(uk_final);
        s_vals.push(sigma);
        v_rows.push(vk);
    }

    if s_vals.is_empty() {
        let mut u0 = vec![0.0f64; m];
        let mut v0 = vec![0.0f64; n];
        if m > 0 {
            u0[0] = 1.0;
        }
        if n > 0 {
            v0[0] = 1.0;
        }
        return (u0, vec![threshold.max(1e-300)], v0);
    }

    let r = s_vals.len();
    // Pack U in row-major: u_data[i*r + k] = u_cols[k][i]
    let mut u_data = vec![0.0f64; m * r];
    for i in 0..m {
        for k in 0..r {
            u_data[i * r + k] = u_cols[k][i];
        }
    }
    let vt_data: Vec<f64> = v_rows.into_iter().flatten().collect();
    (u_data, s_vals, vt_data)
}

// ─── TT-SVD ──────────────────────────────────────────────────────────────────

/// TT-SVD algorithm (Oseledets 2011).
///
/// Sequentially unfolds the tensor into matrices and performs truncated SVD
/// to obtain the TT cores.
///
/// # Parameters
/// - `tensor`:   dense input tensor of any shape
/// - `max_rank`: maximum TT rank per bond
/// - `tol`:      relative truncation tolerance (`tol * sigma_max` threshold)
///
/// # Returns
/// A [`TensorTrain`] approximating `tensor`.
pub fn tt_svd(
    tensor: &ArrayD<f64>,
    max_rank: usize,
    tol: f64,
) -> Result<TensorTrain, InterpolateError> {
    let shape: Vec<usize> = tensor.shape().to_vec();
    let d = shape.len();

    if d == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "tt_svd: tensor must have at least one dimension".into(),
        });
    }
    if max_rank == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "tt_svd: max_rank must be >= 1".into(),
        });
    }

    let mut cores = Vec::with_capacity(d);
    let mut r_left = 1usize;

    // Start with the flat data copy
    let mut remainder: Vec<f64> = tensor.iter().copied().collect();

    for k in 0..d {
        let n_k = shape[k];
        // Remaining dimensions product
        let n_right: usize = shape[k + 1..].iter().product::<usize>().max(1);

        let rows = r_left * n_k;
        let cols = n_right;

        // Build matrix of shape [r_left * n_k, n_right]
        let mat = Array2::from_shape_vec((rows, cols), remainder.clone()).map_err(|e| {
            InterpolateError::ComputationError(format!("tt_svd reshape error at k={k}: {e}"))
        })?;

        if k < d - 1 {
            let (u, s, vt) = truncated_svd(&mat, max_rank, tol)?;
            let r_right = s.len();

            // Core k has shape [r_left, n_k, r_right]
            // U has shape [r_left * n_k, r_right]; reshape to [r_left, n_k, r_right]
            let u_flat: Vec<f64> = u.iter().copied().collect();
            let core = Array3::from_shape_vec((r_left, n_k, r_right), u_flat).map_err(|e| {
                InterpolateError::ComputationError(format!("tt_svd core shape error k={k}: {e}"))
            })?;
            cores.push(core);

            // New remainder = diag(s) * Vt  (shape [r_right, n_right])
            let mut new_rem = vec![0.0f64; r_right * cols];
            for i in 0..r_right {
                let si = s[i];
                for j in 0..cols {
                    new_rem[i * cols + j] = si * vt[[i, j]];
                }
            }
            remainder = new_rem;
            r_left = r_right;
        } else {
            // Last core: the remainder is already [r_left * n_d, 1]
            // => shape [r_left, n_d, 1]
            let mat_flat: Vec<f64> = mat.iter().copied().collect();
            let core = Array3::from_shape_vec((r_left, n_k, 1), mat_flat).map_err(|e| {
                InterpolateError::ComputationError(format!(
                    "tt_svd last core shape error k={k}: {e}"
                ))
            })?;
            cores.push(core);
        }
    }

    TensorTrain::new(cores)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::IxDyn;

    fn make_rank1_tt(shape: &[usize]) -> TensorTrain {
        // Build a TT of all ones: each core is all-1s of shape [1, n_k, 1]
        let cores: Vec<Array3<f64>> = shape.iter().map(|&n| Array3::ones((1, n, 1))).collect();
        TensorTrain::new(cores).expect("rank-1 TT valid")
    }

    #[test]
    fn test_tt_eval_correct() {
        // 2D: TT where core0[0, i, 0] = i+1  and core1[0, j, 0] = j+1
        // => TT[i,j] = (i+1)*(j+1)
        let core0 = Array3::from_shape_fn((1, 3, 1), |(_, i, _)| (i + 1) as f64);
        let core1 = Array3::from_shape_fn((1, 4, 1), |(_, j, _)| (j + 1) as f64);
        let tt = TensorTrain::new(vec![core0, core1]).expect("valid TT");

        for i in 0..3 {
            for j in 0..4 {
                let val = tt.eval(&[i, j]).expect("eval ok");
                let expected = ((i + 1) * (j + 1)) as f64;
                assert!(
                    (val - expected).abs() < 1e-12,
                    "TT[{i},{j}] expected {expected} got {val}"
                );
            }
        }
    }

    #[test]
    fn test_tt_norm() {
        // Rank-1 TT of shape [2,2]: each entry = 1, Frobenius norm = 2
        let tt = make_rank1_tt(&[2, 2]);
        let norm = tt.norm();
        assert!((norm - 2.0).abs() < 1e-10, "norm={norm}");
    }

    #[test]
    fn test_tt_n_params() {
        // shape [3, 4], ranks [1,1,1]: params = 1*3*1 + 1*4*1 = 7
        let tt = make_rank1_tt(&[3, 4]);
        assert_eq!(tt.n_params(), 7);
    }

    #[test]
    fn test_tt_svd_2d() {
        // TT-SVD of a rank-1 2D tensor: outer product a x b
        let a = [1.0, 2.0, 3.0f64];
        let b = [1.0, -1.0, 2.0, -2.0f64];
        let data: Vec<f64> = a
            .iter()
            .flat_map(|&ai| b.iter().map(move |&bj| ai * bj))
            .collect();
        let tensor = ArrayD::from_shape_vec(IxDyn(&[3, 4]), data).expect("valid");

        let tt = tt_svd(&tensor, 4, 1e-10).expect("TT-SVD ok");
        assert_eq!(tt.shape, vec![3, 4]);
        // Recover values — should match original tensor
        for i in 0..3 {
            for j in 0..4 {
                let val = tt.eval(&[i, j]).expect("eval ok");
                let expected = a[i] * b[j];
                assert!(
                    (val - expected).abs() < 1e-7,
                    "TT-SVD[{i},{j}] expected {expected:.6} got {val:.6}"
                );
            }
        }
    }

    #[test]
    fn test_tt_from_dense_rank_compression() {
        // A 2x2x2 tensor of all 1s has TT rank 1
        let tensor = ArrayD::ones(IxDyn(&[2, 2, 2]));
        let tt = TensorTrain::from_dense(&tensor, 4, 1e-8).expect("from_dense ok");
        // Rank-1 TT of shape [2,2,2] has 2+2+2=6 params; max allowed 8
        assert!(tt.n_params() <= 8, "n_params={}", tt.n_params());
        // Values should be recoverable
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let val = tt.eval(&[i, j, k]).expect("eval ok");
                    assert!((val - 1.0).abs() < 1e-6, "val={val}");
                }
            }
        }
    }

    #[test]
    fn test_tt_to_dense() {
        let core0 = Array3::from_shape_fn((1, 2, 1), |(_, i, _)| (i + 1) as f64);
        let core1 = Array3::from_shape_fn((1, 2, 1), |(_, j, _)| (j + 1) as f64);
        let tt = TensorTrain::new(vec![core0, core1]).expect("valid");
        let dense = tt.to_dense().expect("to_dense ok");
        assert_eq!(dense.shape(), &[2, 2]);
        // [1*1, 1*2; 2*1, 2*2] = [[1,2],[2,4]]
        assert!((dense[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((dense[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((dense[[1, 0]] - 2.0).abs() < 1e-12);
        assert!((dense[[1, 1]] - 4.0).abs() < 1e-12);
    }
}
