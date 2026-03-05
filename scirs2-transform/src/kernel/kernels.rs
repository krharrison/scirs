//! Kernel Functions Library
//!
//! Provides a comprehensive set of kernel functions for use in kernel methods
//! such as Kernel PCA, Kernel Ridge Regression, and Support Vector Machines.
//!
//! ## Available Kernels
//!
//! - **Linear**: `k(x, y) = x^T y`
//! - **Polynomial**: `k(x, y) = (gamma * x^T y + coef0)^degree`
//! - **RBF/Gaussian**: `k(x, y) = exp(-gamma * ||x - y||^2)`
//! - **Laplacian**: `k(x, y) = exp(-gamma * ||x - y||_1)`
//! - **Sigmoid/Tanh**: `k(x, y) = tanh(gamma * x^T y + coef0)`
//!
//! ## Gram Matrix and Centering
//!
//! The module also provides utilities for computing Gram matrices (kernel matrices)
//! and centering them in feature space.

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use scirs2_core::numeric::{Float, NumCast};

use crate::error::{Result, TransformError};

/// Kernel function type
#[derive(Debug, Clone, PartialEq)]
pub enum KernelType {
    /// Linear kernel: `k(x, y) = x^T y`
    Linear,
    /// Polynomial kernel: `k(x, y) = (gamma * x^T y + coef0)^degree`
    Polynomial {
        /// Scaling factor for the dot product
        gamma: f64,
        /// Independent term in the polynomial
        coef0: f64,
        /// Degree of the polynomial
        degree: u32,
    },
    /// RBF (Gaussian) kernel: `k(x, y) = exp(-gamma * ||x - y||^2)`
    RBF {
        /// Width parameter (inverse of twice the squared bandwidth)
        gamma: f64,
    },
    /// Laplacian kernel: `k(x, y) = exp(-gamma * ||x - y||_1)`
    Laplacian {
        /// Width parameter
        gamma: f64,
    },
    /// Sigmoid (tanh) kernel: `k(x, y) = tanh(gamma * x^T y + coef0)`
    Sigmoid {
        /// Scaling factor
        gamma: f64,
        /// Independent term
        coef0: f64,
    },
}

impl KernelType {
    /// Create an RBF kernel with automatic gamma selection based on data
    ///
    /// Uses the median heuristic: gamma = 1 / (2 * median_distance^2)
    pub fn rbf_auto<S>(x: &ArrayBase<S, Ix2>) -> Result<Self>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let gamma = estimate_rbf_gamma(x)?;
        Ok(KernelType::RBF { gamma })
    }

    /// Create a polynomial kernel with default parameters
    pub fn polynomial_default() -> Self {
        KernelType::Polynomial {
            gamma: 1.0,
            coef0: 1.0,
            degree: 3,
        }
    }

    /// Create an RBF kernel with the given gamma
    pub fn rbf(gamma: f64) -> Self {
        KernelType::RBF { gamma }
    }

    /// Create a Laplacian kernel with the given gamma
    pub fn laplacian(gamma: f64) -> Self {
        KernelType::Laplacian { gamma }
    }

    /// Create a sigmoid kernel with default parameters
    pub fn sigmoid_default() -> Self {
        KernelType::Sigmoid {
            gamma: 1.0,
            coef0: 0.0,
        }
    }
}

/// Evaluate a kernel function between two vectors
///
/// # Arguments
/// * `x` - First input vector
/// * `y` - Second input vector
/// * `kernel` - The kernel function type
///
/// # Returns
/// * `Result<f64>` - The kernel evaluation k(x, y)
pub fn kernel_eval<S1, S2>(
    x: &ArrayBase<S1, Ix1>,
    y: &ArrayBase<S2, Ix1>,
    kernel: &KernelType,
) -> Result<f64>
where
    S1: Data,
    S2: Data,
    S1::Elem: Float + NumCast,
    S2::Elem: Float + NumCast,
{
    if x.len() != y.len() {
        return Err(TransformError::InvalidInput(format!(
            "Vector dimensions must match: {} vs {}",
            x.len(),
            y.len()
        )));
    }

    let n = x.len();
    match kernel {
        KernelType::Linear => {
            let mut dot = 0.0;
            for i in 0..n {
                let xi: f64 = NumCast::from(x[i]).unwrap_or(0.0);
                let yi: f64 = NumCast::from(y[i]).unwrap_or(0.0);
                dot += xi * yi;
            }
            Ok(dot)
        }
        KernelType::Polynomial {
            gamma,
            coef0,
            degree,
        } => {
            let mut dot = 0.0;
            for i in 0..n {
                let xi: f64 = NumCast::from(x[i]).unwrap_or(0.0);
                let yi: f64 = NumCast::from(y[i]).unwrap_or(0.0);
                dot += xi * yi;
            }
            Ok((gamma * dot + coef0).powi(*degree as i32))
        }
        KernelType::RBF { gamma } => {
            let mut dist_sq = 0.0;
            for i in 0..n {
                let xi: f64 = NumCast::from(x[i]).unwrap_or(0.0);
                let yi: f64 = NumCast::from(y[i]).unwrap_or(0.0);
                let diff = xi - yi;
                dist_sq += diff * diff;
            }
            Ok((-gamma * dist_sq).exp())
        }
        KernelType::Laplacian { gamma } => {
            let mut l1_dist = 0.0;
            for i in 0..n {
                let xi: f64 = NumCast::from(x[i]).unwrap_or(0.0);
                let yi: f64 = NumCast::from(y[i]).unwrap_or(0.0);
                l1_dist += (xi - yi).abs();
            }
            Ok((-gamma * l1_dist).exp())
        }
        KernelType::Sigmoid { gamma, coef0 } => {
            let mut dot = 0.0;
            for i in 0..n {
                let xi: f64 = NumCast::from(x[i]).unwrap_or(0.0);
                let yi: f64 = NumCast::from(y[i]).unwrap_or(0.0);
                dot += xi * yi;
            }
            Ok((gamma * dot + coef0).tanh())
        }
    }
}

/// Compute the Gram matrix (kernel matrix) for a dataset
///
/// The Gram matrix K has entries K\[i,j\] = k(x_i, x_j).
///
/// # Arguments
/// * `x` - Input data matrix, shape (n_samples, n_features)
/// * `kernel` - The kernel function type
///
/// # Returns
/// * `Result<Array2<f64>>` - The Gram matrix, shape (n_samples, n_samples)
pub fn gram_matrix<S>(x: &ArrayBase<S, Ix2>, kernel: &KernelType) -> Result<Array2<f64>>
where
    S: Data,
    S::Elem: Float + NumCast,
{
    let n_samples = x.nrows();
    let mut k = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        for j in i..n_samples {
            let val = kernel_eval(&x.row(i), &x.row(j), kernel)?;
            k[[i, j]] = val;
            k[[j, i]] = val;
        }
    }

    Ok(k)
}

/// Compute the Gram matrix between two datasets
///
/// K\[i,j\] = k(x_i, y_j)
///
/// # Arguments
/// * `x` - First input data, shape (n_x, n_features)
/// * `y` - Second input data, shape (n_y, n_features)
/// * `kernel` - The kernel function type
///
/// # Returns
/// * `Result<Array2<f64>>` - The cross-kernel matrix, shape (n_x, n_y)
pub fn cross_gram_matrix<S1, S2>(
    x: &ArrayBase<S1, Ix2>,
    y: &ArrayBase<S2, Ix2>,
    kernel: &KernelType,
) -> Result<Array2<f64>>
where
    S1: Data,
    S2: Data,
    S1::Elem: Float + NumCast,
    S2::Elem: Float + NumCast,
{
    if x.ncols() != y.ncols() {
        return Err(TransformError::InvalidInput(format!(
            "Feature dimensions must match: {} vs {}",
            x.ncols(),
            y.ncols()
        )));
    }

    let n_x = x.nrows();
    let n_y = y.nrows();
    let mut k = Array2::zeros((n_x, n_y));

    for i in 0..n_x {
        for j in 0..n_y {
            k[[i, j]] = kernel_eval(&x.row(i), &y.row(j), kernel)?;
        }
    }

    Ok(k)
}

/// Center a kernel matrix in feature space
///
/// The centered kernel matrix is:
/// K_c = K - 1_n K - K 1_n + 1_n K 1_n
///
/// where 1_n is the n x n matrix with all entries 1/n.
///
/// This corresponds to centering the data in the feature space
/// without explicitly computing the feature map.
///
/// # Arguments
/// * `k` - The kernel (Gram) matrix, shape (n, n)
///
/// # Returns
/// * `Result<Array2<f64>>` - The centered kernel matrix
pub fn center_kernel_matrix(k: &Array2<f64>) -> Result<Array2<f64>> {
    let n = k.nrows();
    if n != k.ncols() {
        return Err(TransformError::InvalidInput(
            "Kernel matrix must be square".to_string(),
        ));
    }
    if n == 0 {
        return Err(TransformError::InvalidInput(
            "Kernel matrix must be non-empty".to_string(),
        ));
    }

    let n_f64 = n as f64;

    // Compute row means and column means (they should be equal for symmetric K)
    let row_means = k.mean_axis(Axis(0)).ok_or_else(|| {
        TransformError::ComputationError("Failed to compute row means".to_string())
    })?;
    let col_means = k.mean_axis(Axis(1)).ok_or_else(|| {
        TransformError::ComputationError("Failed to compute column means".to_string())
    })?;
    let grand_mean = row_means.sum() / n_f64;

    let mut k_centered = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            k_centered[[i, j]] = k[[i, j]] - row_means[j] - col_means[i] + grand_mean;
        }
    }

    Ok(k_centered)
}

/// Center a test kernel matrix using the training kernel matrix statistics
///
/// For out-of-sample data, the centering must use the training data statistics:
/// K_test_c = K_test - 1'_m K_train - K_test 1_n + 1'_m K_train 1_n
///
/// # Arguments
/// * `k_test` - Test kernel matrix, shape (m, n) where m = test samples, n = training samples
/// * `k_train` - Training kernel matrix, shape (n, n)
///
/// # Returns
/// * `Result<Array2<f64>>` - The centered test kernel matrix, shape (m, n)
pub fn center_kernel_matrix_test(
    k_test: &Array2<f64>,
    k_train: &Array2<f64>,
) -> Result<Array2<f64>> {
    let n_train = k_train.nrows();
    let n_test = k_test.nrows();

    if k_train.nrows() != k_train.ncols() {
        return Err(TransformError::InvalidInput(
            "Training kernel matrix must be square".to_string(),
        ));
    }
    if k_test.ncols() != n_train {
        return Err(TransformError::InvalidInput(format!(
            "Test kernel matrix columns ({}) must match training samples ({})",
            k_test.ncols(),
            n_train
        )));
    }

    let n_f64 = n_train as f64;

    // Mean of each column of K_train
    let train_col_means = k_train.mean_axis(Axis(0)).ok_or_else(|| {
        TransformError::ComputationError("Failed to compute train column means".to_string())
    })?;

    // Mean of each row of K_test (mean over training samples for each test point)
    let test_row_means = k_test.mean_axis(Axis(1)).ok_or_else(|| {
        TransformError::ComputationError("Failed to compute test row means".to_string())
    })?;

    // Grand mean of K_train
    let train_grand_mean = train_col_means.sum() / n_f64;

    let mut k_centered = Array2::zeros((n_test, n_train));
    for i in 0..n_test {
        for j in 0..n_train {
            k_centered[[i, j]] =
                k_test[[i, j]] - test_row_means[i] - train_col_means[j] + train_grand_mean;
        }
    }

    Ok(k_centered)
}

/// Estimate the RBF gamma parameter using the median heuristic
///
/// gamma = 1 / (2 * median_distance^2)
///
/// This is a common automatic bandwidth selection method for the RBF kernel.
///
/// # Arguments
/// * `x` - Input data, shape (n_samples, n_features)
///
/// # Returns
/// * `Result<f64>` - The estimated gamma parameter
pub fn estimate_rbf_gamma<S>(x: &ArrayBase<S, Ix2>) -> Result<f64>
where
    S: Data,
    S::Elem: Float + NumCast,
{
    let n = x.nrows();
    if n < 2 {
        return Err(TransformError::InvalidInput(
            "Need at least 2 samples to estimate gamma".to_string(),
        ));
    }

    // Compute all pairwise squared distances
    let mut distances: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dist_sq = 0.0;
            for k in 0..x.ncols() {
                let xi: f64 = NumCast::from(x[[i, k]]).unwrap_or(0.0);
                let xj: f64 = NumCast::from(x[[j, k]]).unwrap_or(0.0);
                let diff = xi - xj;
                dist_sq += diff * diff;
            }
            distances.push(dist_sq);
        }
    }

    // Sort distances
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Get median squared distance
    let median_sq = if distances.len() % 2 == 0 {
        let mid = distances.len() / 2;
        (distances[mid - 1] + distances[mid]) / 2.0
    } else {
        distances[distances.len() / 2]
    };

    if median_sq < 1e-15 {
        // Data points are very close together, use a default
        Ok(1.0)
    } else {
        Ok(1.0 / (2.0 * median_sq))
    }
}

/// Compute the diagonal of a kernel matrix (self-similarities)
///
/// # Arguments
/// * `x` - Input data, shape (n_samples, n_features)
/// * `kernel` - The kernel function type
///
/// # Returns
/// * `Result<Array1<f64>>` - Diagonal entries k(x_i, x_i)
pub fn kernel_diagonal<S>(x: &ArrayBase<S, Ix2>, kernel: &KernelType) -> Result<Array1<f64>>
where
    S: Data,
    S::Elem: Float + NumCast,
{
    let n = x.nrows();
    let mut diag = Array1::zeros(n);

    for i in 0..n {
        diag[i] = kernel_eval(&x.row(i), &x.row(i), kernel)?;
    }

    Ok(diag)
}

/// Check if a kernel matrix is positive semi-definite
///
/// A kernel matrix should be positive semi-definite (PSD). This function
/// checks by verifying that all eigenvalues are non-negative (within tolerance).
///
/// # Arguments
/// * `k` - The kernel matrix to check
/// * `tol` - Tolerance for negative eigenvalues (default: -1e-10)
///
/// # Returns
/// * `Result<bool>` - True if the matrix is PSD within the given tolerance
pub fn is_positive_semidefinite(k: &Array2<f64>, tol: f64) -> Result<bool> {
    if k.nrows() != k.ncols() {
        return Err(TransformError::InvalidInput(
            "Matrix must be square".to_string(),
        ));
    }

    let (eigenvalues, _) =
        scirs2_linalg::eigh(&k.view(), None).map_err(TransformError::LinalgError)?;

    let min_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);

    Ok(min_eigenvalue >= tol)
}

/// Kernel alignment between two kernel matrices
///
/// The alignment A(K1, K2) = <K1, K2>_F / (||K1||_F * ||K2||_F)
/// where <.,.>_F is the Frobenius inner product.
///
/// This measures how similar two kernel matrices are.
///
/// # Arguments
/// * `k1` - First kernel matrix
/// * `k2` - Second kernel matrix
///
/// # Returns
/// * `Result<f64>` - The alignment score in [0, 1]
pub fn kernel_alignment(k1: &Array2<f64>, k2: &Array2<f64>) -> Result<f64> {
    if k1.dim() != k2.dim() {
        return Err(TransformError::InvalidInput(
            "Kernel matrices must have the same dimensions".to_string(),
        ));
    }

    let frobenius_inner: f64 = k1.iter().zip(k2.iter()).map(|(&a, &b)| a * b).sum();
    let norm1: f64 = k1.iter().map(|&a| a * a).sum::<f64>().sqrt();
    let norm2: f64 = k2.iter().map(|&a| a * a).sum::<f64>().sqrt();

    let denom = norm1 * norm2;
    if denom < 1e-15 {
        Ok(0.0)
    } else {
        Ok((frobenius_inner / denom).clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    fn sample_data() -> Array2<f64> {
        Array::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            ],
        )
        .expect("Failed to create sample data")
    }

    #[test]
    fn test_linear_kernel() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array::from_vec(vec![4.0, 5.0, 6.0]);
        let result =
            kernel_eval(&x.view(), &y.view(), &KernelType::Linear).expect("kernel eval failed");
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((result - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_kernel() {
        let x = Array::from_vec(vec![1.0, 2.0]);
        let y = Array::from_vec(vec![3.0, 4.0]);
        let kernel = KernelType::Polynomial {
            gamma: 1.0,
            coef0: 1.0,
            degree: 2,
        };
        let result = kernel_eval(&x.view(), &y.view(), &kernel).expect("kernel eval failed");
        // (1*3 + 2*4 + 1)^2 = (3 + 8 + 1)^2 = 12^2 = 144
        assert!((result - 144.0).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_kernel() {
        let x = Array::from_vec(vec![1.0, 0.0]);
        let y = Array::from_vec(vec![0.0, 1.0]);
        let kernel = KernelType::RBF { gamma: 0.5 };
        let result = kernel_eval(&x.view(), &y.view(), &kernel).expect("kernel eval failed");
        // exp(-0.5 * (1 + 1)) = exp(-1) ~ 0.3679
        assert!((result - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_kernel_self() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let kernel = KernelType::RBF { gamma: 1.0 };
        let result = kernel_eval(&x.view(), &x.view(), &kernel).expect("kernel eval failed");
        // k(x, x) = exp(0) = 1
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_kernel() {
        let x = Array::from_vec(vec![1.0, 2.0]);
        let y = Array::from_vec(vec![3.0, 4.0]);
        let kernel = KernelType::Laplacian { gamma: 0.5 };
        let result = kernel_eval(&x.view(), &y.view(), &kernel).expect("kernel eval failed");
        // exp(-0.5 * (|1-3| + |2-4|)) = exp(-0.5 * 4) = exp(-2)
        assert!((result - (-2.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_kernel() {
        let x = Array::from_vec(vec![1.0, 0.0]);
        let y = Array::from_vec(vec![0.0, 1.0]);
        let kernel = KernelType::Sigmoid {
            gamma: 1.0,
            coef0: 0.0,
        };
        let result = kernel_eval(&x.view(), &y.view(), &kernel).expect("kernel eval failed");
        // tanh(1 * 0 + 0) = tanh(0) = 0
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gram_matrix_symmetry() {
        let data = sample_data();
        let kernel = KernelType::RBF { gamma: 0.1 };
        let k = gram_matrix(&data.view(), &kernel).expect("gram matrix failed");

        assert_eq!(k.shape(), &[5, 5]);
        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    (k[[i, j]] - k[[j, i]]).abs() < 1e-10,
                    "Gram matrix not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_gram_matrix_diagonal() {
        let data = sample_data();
        let kernel = KernelType::RBF { gamma: 0.1 };
        let k = gram_matrix(&data.view(), &kernel).expect("gram matrix failed");

        // RBF diagonal should be 1.0
        for i in 0..5 {
            assert!(
                (k[[i, i]] - 1.0).abs() < 1e-10,
                "RBF diagonal should be 1.0"
            );
        }
    }

    #[test]
    fn test_cross_gram_matrix() {
        let x = Array::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("Failed");
        let y = Array::from_shape_vec((2, 2), vec![1.5, 2.5, 3.5, 4.5]).expect("Failed");
        let kernel = KernelType::Linear;
        let k = cross_gram_matrix(&x.view(), &y.view(), &kernel).expect("cross gram matrix failed");

        assert_eq!(k.shape(), &[3, 2]);
        // k[0,0] = 1*1.5 + 2*2.5 = 1.5 + 5.0 = 6.5
        assert!((k[[0, 0]] - 6.5).abs() < 1e-10);
    }

    #[test]
    fn test_center_kernel_matrix() {
        let data = sample_data();
        let kernel = KernelType::RBF { gamma: 0.01 };
        let k = gram_matrix(&data.view(), &kernel).expect("gram matrix failed");
        let k_centered = center_kernel_matrix(&k).expect("centering failed");

        // Centered kernel matrix should have zero column means
        let col_means = k_centered
            .mean_axis(Axis(0))
            .expect("Failed to compute means");
        for i in 0..col_means.len() {
            assert!(
                col_means[i].abs() < 1e-10,
                "Centered kernel column mean should be ~0, got {}",
                col_means[i]
            );
        }

        // And zero row means
        let row_means = k_centered
            .mean_axis(Axis(1))
            .expect("Failed to compute means");
        for i in 0..row_means.len() {
            assert!(
                row_means[i].abs() < 1e-10,
                "Centered kernel row mean should be ~0, got {}",
                row_means[i]
            );
        }
    }

    #[test]
    fn test_center_kernel_matrix_test() {
        let x_train = sample_data();
        let x_test =
            Array::from_shape_vec((2, 3), vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5]).expect("Failed");
        let kernel = KernelType::RBF { gamma: 0.01 };

        let k_train = gram_matrix(&x_train.view(), &kernel).expect("gram failed");
        let k_test =
            cross_gram_matrix(&x_test.view(), &x_train.view(), &kernel).expect("cross gram failed");

        let k_test_centered =
            center_kernel_matrix_test(&k_test, &k_train).expect("test centering failed");
        assert_eq!(k_test_centered.shape(), &[2, 5]);

        // Values should be finite
        for val in k_test_centered.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_estimate_rbf_gamma() {
        let data = sample_data();
        let gamma = estimate_rbf_gamma(&data.view()).expect("gamma estimation failed");
        assert!(gamma > 0.0);
        assert!(gamma.is_finite());
    }

    #[test]
    fn test_kernel_diagonal() {
        let data = sample_data();
        let kernel = KernelType::Linear;
        let diag = kernel_diagonal(&data.view(), &kernel).expect("diagonal failed");

        assert_eq!(diag.len(), 5);
        // Linear kernel diagonal: k(x, x) = x^T x
        // First row: 1^2 + 2^2 + 3^2 = 14
        assert!((diag[0] - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_gram_psd() {
        let data = sample_data();
        let kernel = KernelType::RBF { gamma: 0.1 };
        let k = gram_matrix(&data.view(), &kernel).expect("gram matrix failed");
        let psd = is_positive_semidefinite(&k, -1e-10).expect("PSD check failed");
        assert!(psd, "RBF Gram matrix should be PSD");
    }

    #[test]
    fn test_kernel_alignment() {
        let data = sample_data();
        let k1 = gram_matrix(&data.view(), &KernelType::RBF { gamma: 0.1 }).expect("gram failed");
        let k2 = gram_matrix(&data.view(), &KernelType::RBF { gamma: 0.1 }).expect("gram failed");

        let alignment = kernel_alignment(&k1, &k2).expect("alignment failed");
        // Same kernel should have alignment 1.0
        assert!(
            (alignment - 1.0).abs() < 1e-10,
            "Self-alignment should be 1.0, got {}",
            alignment
        );
    }

    #[test]
    fn test_kernel_alignment_different() {
        let data = sample_data();
        let k1 = gram_matrix(&data.view(), &KernelType::RBF { gamma: 0.01 }).expect("gram failed");
        let k2 = gram_matrix(&data.view(), &KernelType::Linear).expect("gram failed");

        let alignment = kernel_alignment(&k1, &k2).expect("alignment failed");
        assert!(alignment >= 0.0 && alignment <= 1.0);
    }

    #[test]
    fn test_rbf_auto() {
        let data = sample_data();
        let kernel = KernelType::rbf_auto(&data.view()).expect("auto rbf failed");
        match kernel {
            KernelType::RBF { gamma } => {
                assert!(gamma > 0.0);
                assert!(gamma.is_finite());
            }
            _ => panic!("Expected RBF kernel type"),
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let x = Array::from_vec(vec![1.0, 2.0]);
        let y = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let result = kernel_eval(&x.view(), &y.view(), &KernelType::Linear);
        assert!(result.is_err());
    }
}
