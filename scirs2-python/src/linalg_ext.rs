//! Extended Python bindings for scirs2-linalg
//!
//! Provides bindings for v0.3.0 features:
//! - Tensor decompositions (CP/Tucker/randomized SVD)
//! - Structured matrices (Toeplitz/circulant/tridiagonal solvers)
//! - Randomized methods (randomized SVD, PCA)
//! - Procrustes analysis

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use scirs2_core::ndarray::{Array1, Array2, Array3};

// =============================================================================
// Tensor Decompositions
// =============================================================================

/// CP (CANDECOMP/PARAFAC) decomposition of a 3D tensor using ALS.
///
/// Decomposes a tensor T ≈ sum_r lambda_r * a_r ⊗ b_r ⊗ c_r.
///
/// Parameters:
///     tensor: 3D array with shape (I, J, K) as list[list[list[float]]]
///     rank: Number of components
///     max_iter: Maximum ALS iterations (default: 200)
///     tol: Convergence tolerance (default: 1e-6)
///
/// Returns:
///     Dict with:
///     - 'factors': list of 3 factor matrices (A, B, C) as numpy arrays
///     - 'lambdas': component weights (1D numpy array)
///     - 'converged': bool indicating convergence
///     - 'n_iter': number of iterations performed
#[pyfunction]
#[pyo3(signature = (tensor, rank, max_iter=200, tol=1e-6))]
pub fn cp_decomp_py(
    py: Python,
    tensor: Vec<Vec<Vec<f64>>>,
    rank: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<PyAny>> {
    if tensor.is_empty() {
        return Err(PyValueError::new_err("tensor must not be empty"));
    }
    if rank == 0 {
        return Err(PyValueError::new_err("rank must be > 0"));
    }

    let i_dim = tensor.len();
    let j_dim = tensor[0].len();
    let k_dim = tensor[0].first().map(|r| r.len()).unwrap_or(0);

    if j_dim == 0 || k_dim == 0 {
        return Err(PyValueError::new_err("tensor dimensions must all be > 0"));
    }

    let mut flat: Vec<f64> = Vec::with_capacity(i_dim * j_dim * k_dim);
    for mat in &tensor {
        if mat.len() != j_dim {
            return Err(PyValueError::new_err("tensor must have uniform dimensions"));
        }
        for row in mat {
            if row.len() != k_dim {
                return Err(PyValueError::new_err("tensor must have uniform dimensions"));
            }
            flat.extend_from_slice(row);
        }
    }

    let tensor_arr = Array3::from_shape_vec((i_dim, j_dim, k_dim), flat)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tensor: {}", e)))?;

    let (decomp, diag) = scirs2_linalg::cp_decomp(&tensor_arr, rank, max_iter, tol)
        .map_err(|e| PyRuntimeError::new_err(format!("CP decomposition failed: {}", e)))?;

    let factors_list = pyo3::types::PyList::new(
        py,
        decomp.factors.iter().map(|f| f.clone().into_pyarray(py)),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create factors list: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("factors", factors_list)?;
    dict.set_item("lambdas", decomp.lambdas.into_pyarray(py))?;
    dict.set_item("converged", diag.converged)?;
    dict.set_item("n_iter", diag.n_iter)?;

    Ok(dict.into())
}

/// Reconstruct a tensor from CP decomposition factors.
///
/// Parameters:
///     factors: List of 3 factor matrices (each as list[list[float]])
///     lambdas: Component weights (list of floats)
///
/// Returns:
///     Reconstructed 3D tensor as numpy array
#[pyfunction]
pub fn cp_reconstruct_py(
    py: Python,
    factors: Vec<Vec<Vec<f64>>>,
    lambdas: Vec<f64>,
) -> PyResult<Py<PyArray3<f64>>> {
    if factors.len() != 3 {
        return Err(PyValueError::new_err("factors must have exactly 3 elements"));
    }
    let rank = lambdas.len();
    if rank == 0 {
        return Err(PyValueError::new_err("lambdas must not be empty"));
    }

    let factor_arrays: Vec<Array2<f64>> = factors
        .iter()
        .enumerate()
        .map(|(idx, f)| {
            let nrows = f.len();
            let ncols = f.first().map(|r| r.len()).unwrap_or(0);
            if nrows == 0 || ncols == 0 {
                return Err(format!("factor[{}] must be non-empty", idx));
            }
            let mut flat = Vec::with_capacity(nrows * ncols);
            for row in f {
                if row.len() != ncols {
                    return Err(format!("factor[{}] has inconsistent column sizes", idx));
                }
                flat.extend_from_slice(row);
            }
            Array2::from_shape_vec((nrows, ncols), flat)
                .map_err(|e| format!("Factor[{}] shape error: {}", idx, e))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PyRuntimeError::new_err(e))?;

    let lambdas_arr = Array1::from_vec(lambdas);

    let decomp = scirs2_linalg::CpDecomp {
        factors: [
            factor_arrays[0].clone(),
            factor_arrays[1].clone(),
            factor_arrays[2].clone(),
        ],
        lambdas: lambdas_arr,
    };

    let reconstructed = scirs2_linalg::cp_reconstruct(&decomp);
    Ok(reconstructed.into_pyarray(py).unbind())
}

/// Tucker decomposition of a 3D tensor via HOSVD.
///
/// Decomposes tensor T ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃ where G is the core tensor.
///
/// Parameters:
///     tensor: 3D array with shape (I, J, K) as list[list[list[float]]]
///     ranks: List of 3 target ranks [r1, r2, r3]
///
/// Returns:
///     Dict with:
///     - 'core': core tensor as 3D numpy array
///     - 'factors': list of 3 factor matrices as numpy arrays
#[pyfunction]
pub fn tucker_hosvd_py(
    py: Python,
    tensor: Vec<Vec<Vec<f64>>>,
    ranks: [usize; 3],
) -> PyResult<Py<PyAny>> {
    if tensor.is_empty() {
        return Err(PyValueError::new_err("tensor must not be empty"));
    }
    for &r in &ranks {
        if r == 0 {
            return Err(PyValueError::new_err("all ranks must be > 0"));
        }
    }

    let i_dim = tensor.len();
    let j_dim = tensor[0].len();
    let k_dim = tensor[0].first().map(|r| r.len()).unwrap_or(0);

    if j_dim == 0 || k_dim == 0 {
        return Err(PyValueError::new_err("tensor dimensions must all be > 0"));
    }

    let mut flat: Vec<f64> = Vec::with_capacity(i_dim * j_dim * k_dim);
    for mat in &tensor {
        for row in mat {
            flat.extend_from_slice(row);
        }
    }

    let tensor_arr = Array3::from_shape_vec((i_dim, j_dim, k_dim), flat)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tensor: {}", e)))?;

    let decomp = scirs2_linalg::tucker_hosvd_3d(&tensor_arr.view(), ranks)
        .map_err(|e| PyRuntimeError::new_err(format!("Tucker HOSVD failed: {}", e)))?;

    let factors_list = pyo3::types::PyList::new(
        py,
        decomp.factors.iter().map(|f| f.clone().into_pyarray(py)),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create factors list: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("core", decomp.core.into_pyarray(py))?;
    dict.set_item("factors", factors_list)?;

    Ok(dict.into())
}

// =============================================================================
// Randomized Methods
// =============================================================================

/// Randomized SVD for large matrices.
///
/// Computes an approximate SVD using random projections, which is
/// efficient for large matrices when only a few singular values are needed.
///
/// Parameters:
///     a: Input 2D matrix (m × n)
///     rank: Target rank (number of singular values/vectors)
///     oversampling: Oversampling parameter for accuracy (default: 10)
///     power_iter: Number of power iterations for accuracy (default: 2)
///     seed: Random seed for reproducibility (default: None)
///
/// Returns:
///     Tuple (U, S, Vt) where U is (m, rank), S is (rank,), Vt is (rank, n)
#[pyfunction]
#[pyo3(signature = (a, rank, oversampling=10, power_iter=2, seed=None))]
pub fn randomized_svd_py(
    py: Python,
    a: &scirs2_numpy::PyArray2<f64>,
    rank: usize,
    oversampling: usize,
    power_iter: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    if rank == 0 {
        return Err(PyValueError::new_err("rank must be > 0"));
    }
    let binding = a.readonly();
    let a_arr = binding.as_array();

    let mut config = scirs2_linalg::RandomizedConfig::new(rank)
        .with_oversampling(oversampling)
        .with_power_iterations(power_iter);
    if let Some(s) = seed {
        config = config.with_seed(s);
    }

    let (u, s, vt) = scirs2_linalg::randomized_svd_hmt(&a_arr, &config)
        .map_err(|e| PyRuntimeError::new_err(format!("Randomized SVD failed: {}", e)))?;

    let tup = pyo3::types::PyTuple::new(
        py,
        [
            u.into_pyarray(py).as_any(),
            s.into_pyarray(py).as_any(),
            vt.into_pyarray(py).as_any(),
        ],
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tuple: {}", e)))?;

    Ok(tup.into())
}

/// Randomized PCA for dimensionality reduction.
///
/// Efficient PCA via random projections, suitable for large matrices.
///
/// Parameters:
///     a: Input 2D matrix (samples × features)
///     n_components: Number of principal components
///     oversampling: Oversampling parameter (default: 10)
///     power_iter: Power iterations for accuracy (default: 2)
///     seed: Random seed (default: None)
///
/// Returns:
///     Dict with:
///     - 'components': Principal components (n_components × features)
///     - 'singular_values': Singular values (n_components,)
///     - 'explained_variance': Variance for each component
///     - 'explained_variance_ratio': Fraction of variance explained
///     - 'mean': Feature means used for centering
#[pyfunction]
#[pyo3(signature = (a, n_components, oversampling=10, power_iter=2, seed=None))]
pub fn randomized_pca_py(
    py: Python,
    a: &scirs2_numpy::PyArray2<f64>,
    n_components: usize,
    oversampling: usize,
    power_iter: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    if n_components == 0 {
        return Err(PyValueError::new_err("n_components must be > 0"));
    }
    let binding = a.readonly();
    let a_arr = binding.as_array();

    let mut config = scirs2_linalg::RandomizedConfig::new(n_components)
        .with_oversampling(oversampling)
        .with_power_iterations(power_iter);
    if let Some(s) = seed {
        config = config.with_seed(s);
    }

    let result = scirs2_linalg::randomized_pca(&a_arr, &config)
        .map_err(|e| PyRuntimeError::new_err(format!("Randomized PCA failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("components", result.components.into_pyarray(py))?;
    dict.set_item("singular_values", result.singular_values.into_pyarray(py))?;
    dict.set_item("explained_variance", result.explained_variance.into_pyarray(py))?;
    dict.set_item(
        "explained_variance_ratio",
        result.explained_variance_ratio.into_pyarray(py),
    )?;
    dict.set_item("mean", result.mean.into_pyarray(py))?;

    Ok(dict.into())
}

// =============================================================================
// Structured Matrix Solvers
// =============================================================================

/// Solve a Toeplitz system T @ x = b.
///
/// Exploits the Toeplitz structure for O(n²) instead of O(n³) complexity.
///
/// Parameters:
///     r: First row of the Toeplitz matrix (length n)
///     c: First column of the Toeplitz matrix (length n)
///     b: Right-hand side vector (length n)
///
/// Returns:
///     Solution vector x as numpy array (length n)
#[pyfunction]
pub fn solve_toeplitz_py(
    py: Python,
    r: Vec<f64>,
    c: Vec<f64>,
    b: Vec<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    if r.is_empty() {
        return Err(PyValueError::new_err("r must not be empty"));
    }
    if r.len() != c.len() || r.len() != b.len() {
        return Err(PyValueError::new_err("r, c, and b must have the same length"));
    }

    let r_arr = Array1::from_vec(r);
    let c_arr = Array1::from_vec(c);
    let b_arr = Array1::from_vec(b);

    let x = scirs2_linalg::solve_toeplitz(&r_arr.view(), &c_arr.view(), &b_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Toeplitz solve failed: {}", e)))?;

    Ok(x.into_pyarray(py).unbind())
}

/// Solve a circulant system C @ x = b using FFT.
///
/// Parameters:
///     c: First row of the circulant matrix (length n)
///     b: Right-hand side vector (length n)
///
/// Returns:
///     Solution vector x as numpy array (length n)
#[pyfunction]
pub fn solve_circulant_py(
    py: Python,
    c: Vec<f64>,
    b: Vec<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    if c.is_empty() {
        return Err(PyValueError::new_err("c must not be empty"));
    }
    if c.len() != b.len() {
        return Err(PyValueError::new_err("c and b must have the same length"));
    }

    let c_arr = Array1::from_vec(c);
    let b_arr = Array1::from_vec(b);

    let x = scirs2_linalg::solve_circulant(&c_arr.view(), &b_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Circulant solve failed: {}", e)))?;

    Ok(x.into_pyarray(py).unbind())
}

/// Solve a tridiagonal system using the Thomas algorithm (O(n)).
///
/// Parameters:
///     lower: Lower diagonal (length n-1)
///     main: Main diagonal (length n)
///     upper: Upper diagonal (length n-1)
///     b: Right-hand side vector (length n)
///
/// Returns:
///     Solution vector x as numpy array (length n)
#[pyfunction]
pub fn solve_tridiagonal_py(
    py: Python,
    lower: Vec<f64>,
    main: Vec<f64>,
    upper: Vec<f64>,
    b: Vec<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let n = main.len();
    if n == 0 {
        return Err(PyValueError::new_err("main diagonal must not be empty"));
    }
    if lower.len() != n - 1 || upper.len() != n - 1 || b.len() != n {
        return Err(PyValueError::new_err(
            "lower and upper must have length n-1, b must have length n",
        ));
    }

    let lower_arr = Array1::from_vec(lower);
    let main_arr = Array1::from_vec(main);
    let upper_arr = Array1::from_vec(upper);
    let b_arr = Array1::from_vec(b);

    let x = scirs2_linalg::tridiagonal_solve(
        &lower_arr.view(),
        &main_arr.view(),
        &upper_arr.view(),
        &b_arr.view(),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Tridiagonal solve failed: {}", e)))?;

    Ok(x.into_pyarray(py).unbind())
}

// =============================================================================
// Procrustes Analysis
// =============================================================================

/// Orthogonal Procrustes analysis.
///
/// Find the orthogonal matrix R that best aligns matrix A to matrix B:
/// minimize ||A - B @ R||_F subject to R^T @ R = I.
///
/// Parameters:
///     a: Target matrix (m × n)
///     b: Source matrix (m × n)
///
/// Returns:
///     Dict with:
///     - 'rotation': Orthogonal rotation matrix R (n × n)
///     - 'residual': Frobenius distance ||A - B @ R||_F after alignment
#[pyfunction]
pub fn orthogonal_procrustes_py(
    py: Python,
    a: &scirs2_numpy::PyArray2<f64>,
    b: &scirs2_numpy::PyArray2<f64>,
) -> PyResult<Py<PyAny>> {
    let a_binding = a.readonly();
    let a_arr = a_binding.as_array();
    let b_binding = b.readonly();
    let b_arr = b_binding.as_array();

    let result = scirs2_linalg::orthogonal_procrustes(&a_arr, &b_arr)
        .map_err(|e| PyRuntimeError::new_err(format!("Orthogonal Procrustes failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("rotation", result.rotation.into_pyarray(py))?;
    dict.set_item("residual", result.residual)?;

    Ok(dict.into())
}

/// Python module registration for linalg extensions
pub fn register_linalg_ext_module(m: &Bound<'_, pyo3::PyModule>) -> pyo3::PyResult<()> {
    // Tensor decompositions
    m.add_function(wrap_pyfunction!(cp_decomp_py, m)?)?;
    m.add_function(wrap_pyfunction!(cp_reconstruct_py, m)?)?;
    m.add_function(wrap_pyfunction!(tucker_hosvd_py, m)?)?;

    // Randomized methods
    m.add_function(wrap_pyfunction!(randomized_svd_py, m)?)?;
    m.add_function(wrap_pyfunction!(randomized_pca_py, m)?)?;

    // Structured matrix solvers
    m.add_function(wrap_pyfunction!(solve_toeplitz_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_circulant_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_tridiagonal_py, m)?)?;

    // Procrustes
    m.add_function(wrap_pyfunction!(orthogonal_procrustes_py, m)?)?;

    Ok(())
}
