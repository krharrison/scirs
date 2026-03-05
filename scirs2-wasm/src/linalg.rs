//! Linear algebra operations for WASM

use crate::array::WasmArray;
use crate::error::WasmError;
use scirs2_core::ndarray::Array2;
use wasm_bindgen::prelude::*;

/// Compute the determinant of a square matrix
#[wasm_bindgen]
pub fn det(arr: &WasmArray) -> Result<f64, JsValue> {
    if arr.ndim() != 2 {
        return Err(
            WasmError::InvalidDimensions("Determinant requires a 2D matrix".to_string()).into(),
        );
    }

    let matrix = arr
        .data()
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;

    if matrix.nrows() != matrix.ncols() {
        return Err(WasmError::InvalidDimensions(
            "Determinant requires a square matrix".to_string(),
        )
        .into());
    }

    // Simple implementation for small matrices
    match matrix.nrows() {
        1 => Ok(matrix[[0, 0]]),
        2 => Ok(matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]]),
        _ => {
            // For larger matrices, use a more sophisticated algorithm
            // This is a simplified LU decomposition-based approach
            Ok(compute_det_lu(&matrix))
        }
    }
}

/// Compute matrix inverse
#[wasm_bindgen]
pub fn inv(arr: &WasmArray) -> Result<WasmArray, JsValue> {
    if arr.ndim() != 2 {
        return Err(
            WasmError::InvalidDimensions("Inverse requires a 2D matrix".to_string()).into(),
        );
    }

    let matrix = arr
        .data()
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;

    if matrix.nrows() != matrix.ncols() {
        return Err(
            WasmError::InvalidDimensions("Inverse requires a square matrix".to_string()).into(),
        );
    }

    // Check for singularity
    let det_val = det(arr)?;
    if det_val.abs() < 1e-10 {
        return Err(
            WasmError::ComputationError("Matrix is singular or near-singular".to_string()).into(),
        );
    }

    // Use Gauss-Jordan elimination for matrix inverse
    let inv_matrix = compute_inverse_gauss_jordan(&matrix).map_err(WasmError::ComputationError)?;

    Ok(WasmArray::from_array(inv_matrix.into_dyn()))
}

/// Compute the trace of a matrix (sum of diagonal elements)
#[wasm_bindgen]
pub fn trace(arr: &WasmArray) -> Result<f64, JsValue> {
    if arr.ndim() != 2 {
        return Err(WasmError::InvalidDimensions("Trace requires a 2D matrix".to_string()).into());
    }

    let matrix = arr
        .data()
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;

    if matrix.nrows() != matrix.ncols() {
        return Err(
            WasmError::InvalidDimensions("Trace requires a square matrix".to_string()).into(),
        );
    }

    let mut sum = 0.0;
    for i in 0..matrix.nrows() {
        sum += matrix[[i, i]];
    }

    Ok(sum)
}

/// Compute the matrix rank
#[wasm_bindgen]
pub fn rank(arr: &WasmArray, tolerance: Option<f64>) -> Result<usize, JsValue> {
    if arr.ndim() != 2 {
        return Err(WasmError::InvalidDimensions("Rank requires a 2D matrix".to_string()).into());
    }

    let matrix = arr
        .data()
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;

    let tol = tolerance.unwrap_or(1e-10);
    let rank = compute_rank(&matrix, tol);

    Ok(rank)
}

/// Compute the Frobenius norm of a matrix
#[wasm_bindgen]
pub fn norm_frobenius(arr: &WasmArray) -> f64 {
    arr.data().iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Solve a system of linear equations Ax = b
#[wasm_bindgen]
pub fn solve(a: &WasmArray, b: &WasmArray) -> Result<WasmArray, JsValue> {
    if a.ndim() != 2 || b.ndim() != 1 {
        return Err(WasmError::InvalidDimensions(
            "solve requires A to be 2D and b to be 1D".to_string(),
        )
        .into());
    }

    let matrix_a = a
        .data()
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;
    let vector_b = b
        .data()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|e: ndarray::ShapeError| WasmError::ComputationError(e.to_string()))?;

    if matrix_a.nrows() != matrix_a.ncols() {
        return Err(WasmError::InvalidDimensions("A must be a square matrix".to_string()).into());
    }

    if matrix_a.nrows() != vector_b.len() {
        return Err(WasmError::ShapeMismatch {
            expected: vec![matrix_a.nrows()],
            actual: vec![vector_b.len()],
        }
        .into());
    }

    // Use Gaussian elimination with partial pivoting
    let solution =
        solve_linear_system(&matrix_a, &vector_b).map_err(WasmError::ComputationError)?;

    Ok(WasmArray::from_array(solution.into_dyn()))
}

// Helper functions

fn compute_det_lu(matrix: &Array2<f64>) -> f64 {
    let n = matrix.nrows();
    let mut lu = matrix.clone();
    let mut sign = 1.0;

    // LU decomposition with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_val = lu[[k, k]].abs();
        let mut max_row = k;

        for i in (k + 1)..n {
            let val = lu[[i, k]].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        if max_val < 1e-15 {
            return 0.0; // Matrix is singular
        }

        // Swap rows if needed
        if max_row != k {
            for j in 0..n {
                let temp = lu[[k, j]];
                lu[[k, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = temp;
            }
            sign = -sign;
        }

        // Eliminate
        for i in (k + 1)..n {
            let factor = lu[[i, k]] / lu[[k, k]];
            for j in k..n {
                lu[[i, j]] -= factor * lu[[k, j]];
            }
        }
    }

    // Compute determinant as product of diagonal elements
    let mut det = sign;
    for i in 0..n {
        det *= lu[[i, i]];
    }

    det
}

fn compute_inverse_gauss_jordan(matrix: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = matrix.nrows();
    let mut aug = Array2::zeros((n, 2 * n));

    // Create augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = matrix[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut max_val = aug[[i, i]].abs();
        let mut max_row = i;

        for k in (i + 1)..n {
            let val = aug[[k, i]].abs();
            if val > max_val {
                max_val = val;
                max_row = k;
            }
        }

        if max_val < 1e-15 {
            return Err("Matrix is singular".to_string());
        }

        // Swap rows
        if max_row != i {
            for j in 0..(2 * n) {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Scale pivot row
        let pivot = aug[[i, i]];
        for j in 0..(2 * n) {
            aug[[i, j]] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[[k, i]];
                for j in 0..(2 * n) {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }
    }

    // Extract inverse matrix from right half
    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }

    Ok(inv)
}

fn compute_rank(matrix: &Array2<f64>, tolerance: f64) -> usize {
    let mut temp = matrix.clone();
    let (m, n) = temp.dim();
    let mut rank = 0;

    for col in 0..n.min(m) {
        // Find pivot
        let mut max_val = 0.0;
        let mut max_row = col;

        for row in col..m {
            let val = temp[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < tolerance {
            continue;
        }

        // Swap rows
        if max_row != col {
            for j in 0..n {
                let temp_val = temp[[col, j]];
                temp[[col, j]] = temp[[max_row, j]];
                temp[[max_row, j]] = temp_val;
            }
        }

        rank += 1;

        // Eliminate
        for row in (col + 1)..m {
            let factor = temp[[row, col]] / temp[[col, col]];
            for j in col..n {
                temp[[row, j]] -= factor * temp[[col, j]];
            }
        }
    }

    rank
}

fn solve_linear_system(
    a: &Array2<f64>,
    b: &ndarray::Array1<f64>,
) -> Result<ndarray::Array1<f64>, String> {
    let n = a.nrows();
    let mut aug = Array2::zeros((n, n + 1));

    // Create augmented matrix [A | b]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_val = aug[[i, i]].abs();
        let mut max_row = i;

        for k in (i + 1)..n {
            let val = aug[[k, i]].abs();
            if val > max_val {
                max_val = val;
                max_row = k;
            }
        }

        if max_val < 1e-15 {
            return Err("System is singular or ill-conditioned".to_string());
        }

        // Swap rows
        if max_row != i {
            for j in 0..=n {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Eliminate
        for k in (i + 1)..n {
            let factor = aug[[k, i]] / aug[[i, i]];
            for j in i..=n {
                aug[[k, j]] -= factor * aug[[i, j]];
            }
        }
    }

    // Back substitution
    let mut x = ndarray::Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}
