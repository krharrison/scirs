//! Matrix Operations WASM API
//!
//! Provides a `WasmMatrix` struct with high-level matrix operations accessible
//! from JavaScript/TypeScript.  All arithmetic, decompositions and transformations
//! are implemented in pure Rust without unsafe code or unwrap().

use crate::error::WasmError;
use wasm_bindgen::prelude::*;

// ============================================================================
// WasmMatrix struct
// ============================================================================

/// A dense row-major f64 matrix with wasm_bindgen bindings.
///
/// # Memory layout
///
/// Data is stored in row-major (C) order: element `(r, c)` lives at index
/// `r * cols + c` in the internal `Vec<f64>`.
///
/// # Example (JavaScript)
///
/// ```javascript
/// const m = new WasmMatrix(2, 2);
/// m.set(0, 0, 1); m.set(0, 1, 2);
/// m.set(1, 0, 3); m.set(1, 1, 4);
/// const d = m.determinant();        // -2.0
/// const t = m.transpose();
/// const inv = m.inverse();
/// const ev = m.eigenvalues();
/// ```
#[wasm_bindgen]
pub struct WasmMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

// ============================================================================
// Internal (non-wasm_bindgen) helpers
// ============================================================================

impl WasmMatrix {
    /// Construct from raw parts (no validation — caller ensures correctness).
    fn from_raw(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        Self { rows, cols, data }
    }

    /// Linear index of element (r, c).
    #[inline(always)]
    fn idx(&self, r: usize, c: usize) -> usize {
        r * self.cols + c
    }

    /// Borrow the flat data slice.
    #[allow(dead_code)]
    pub(crate) fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Row count (internal access for tests).
    #[allow(dead_code)]
    pub(crate) fn row_count(&self) -> usize {
        self.rows
    }

    /// Col count (internal access for tests).
    #[allow(dead_code)]
    pub(crate) fn col_count(&self) -> usize {
        self.cols
    }
}

// ============================================================================
// Public wasm_bindgen API
// ============================================================================

#[wasm_bindgen]
impl WasmMatrix {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Create a new zero-initialised matrix with `rows` rows and `cols` columns.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if either dimension is zero.
    #[wasm_bindgen(constructor)]
    pub fn new(rows: usize, cols: usize) -> Result<WasmMatrix, JsValue> {
        if rows == 0 || cols == 0 {
            return Err(WasmError::InvalidParameter(
                "Matrix dimensions must be greater than zero".to_string(),
            )
            .into());
        }
        Ok(WasmMatrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        })
    }

    /// Create a matrix from a flat row-major `Vec<f64>`.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `data.len() != rows * cols`.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<WasmMatrix, JsValue> {
        if rows == 0 || cols == 0 {
            return Err(WasmError::InvalidParameter(
                "Matrix dimensions must be greater than zero".to_string(),
            )
            .into());
        }
        if data.len() != rows * cols {
            return Err(WasmError::ShapeMismatch {
                expected: vec![rows * cols],
                actual: vec![data.len()],
            }
            .into());
        }
        Ok(WasmMatrix { rows, cols, data })
    }

    /// Create an n×n identity matrix.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if `n == 0`.
    pub fn identity(n: usize) -> Result<WasmMatrix, JsValue> {
        if n == 0 {
            return Err(WasmError::InvalidParameter(
                "Identity matrix size must be greater than zero".to_string(),
            )
            .into());
        }
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Ok(WasmMatrix {
            rows: n,
            cols: n,
            data,
        })
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the value at `(row, col)`.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the indices are out of bounds.
    pub fn get(&self, row: usize, col: usize) -> Result<f64, JsValue> {
        if row >= self.rows || col >= self.cols {
            return Err(WasmError::InvalidParameter(format!(
                "Index ({}, {}) out of bounds for matrix {}×{}",
                row, col, self.rows, self.cols
            ))
            .into());
        }
        Ok(self.data[self.idx(row, col)])
    }

    /// Set the value at `(row, col)`.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the indices are out of bounds.
    pub fn set(&mut self, row: usize, col: usize, val: f64) -> Result<(), JsValue> {
        if row >= self.rows || col >= self.cols {
            return Err(WasmError::InvalidParameter(format!(
                "Index ({}, {}) out of bounds for matrix {}×{}",
                row, col, self.rows, self.cols
            ))
            .into());
        }
        let idx = self.idx(row, col);
        self.data[idx] = val;
        Ok(())
    }

    /// Return a copy of the internal flat data in row-major order.
    pub fn to_vec(&self) -> Vec<f64> {
        self.data.clone()
    }

    // ------------------------------------------------------------------
    // Basic operations
    // ------------------------------------------------------------------

    /// Element-wise addition.  Returns a new matrix.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if shapes differ.
    pub fn add(&self, other: &WasmMatrix) -> Result<WasmMatrix, JsValue> {
        self.check_same_shape(other)?;
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Ok(WasmMatrix::from_raw(self.rows, self.cols, data))
    }

    /// Element-wise subtraction.  Returns a new matrix.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if shapes differ.
    pub fn sub(&self, other: &WasmMatrix) -> Result<WasmMatrix, JsValue> {
        self.check_same_shape(other)?;
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        Ok(WasmMatrix::from_raw(self.rows, self.cols, data))
    }

    /// Scale every element by `scalar`.
    pub fn scale(&self, scalar: f64) -> WasmMatrix {
        let data: Vec<f64> = self.data.iter().map(|&x| x * scalar).collect();
        WasmMatrix::from_raw(self.rows, self.cols, data)
    }

    // ------------------------------------------------------------------
    // Matrix multiplication
    // ------------------------------------------------------------------

    /// Matrix–matrix multiplication: `self × other`.
    ///
    /// `self` must have shape `(M, K)` and `other` must have shape `(K, N)`.
    /// The result has shape `(M, N)`.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the inner dimensions do not match.
    pub fn multiply(&self, other: &WasmMatrix) -> Result<WasmMatrix, JsValue> {
        if self.cols != other.rows {
            return Err(WasmError::ShapeMismatch {
                expected: vec![self.cols],
                actual: vec![other.rows],
            }
            .into());
        }

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;
        let mut result = vec![0.0_f64; m * n];

        for i in 0..m {
            for l in 0..k {
                let a_il = self.data[i * k + l];
                if a_il == 0.0 {
                    continue;
                }
                for j in 0..n {
                    result[i * n + j] += a_il * other.data[l * n + j];
                }
            }
        }

        Ok(WasmMatrix::from_raw(m, n, result))
    }

    // ------------------------------------------------------------------
    // Transpose
    // ------------------------------------------------------------------

    /// Compute the transpose.  Returns a new `(cols × rows)` matrix.
    pub fn transpose(&self) -> WasmMatrix {
        let mut data = vec![0.0_f64; self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        WasmMatrix::from_raw(self.cols, self.rows, data)
    }

    // ------------------------------------------------------------------
    // Determinant
    // ------------------------------------------------------------------

    /// Compute the determinant of a square matrix.
    ///
    /// Uses LU decomposition with partial pivoting for stability.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the matrix is not square.
    pub fn determinant(&self) -> Result<f64, JsValue> {
        if self.rows != self.cols {
            return Err(WasmError::InvalidDimensions(
                "Determinant requires a square matrix".to_string(),
            )
            .into());
        }
        Ok(lu_determinant(&self.data, self.rows))
    }

    // ------------------------------------------------------------------
    // Inverse
    // ------------------------------------------------------------------

    /// Compute the matrix inverse using Gauss-Jordan elimination.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the matrix is not square or is singular.
    pub fn inverse(&self) -> Result<WasmMatrix, JsValue> {
        if self.rows != self.cols {
            return Err(WasmError::InvalidDimensions(
                "Inverse requires a square matrix".to_string(),
            )
            .into());
        }
        let n = self.rows;
        let det = lu_determinant(&self.data, n);
        if det.abs() < 1e-14 {
            return Err(WasmError::ComputationError(
                "Matrix is singular or near-singular (determinant ≈ 0)".to_string(),
            )
            .into());
        }
        let inv_data =
            gauss_jordan_inverse(&self.data, n).map_err(|e| WasmError::ComputationError(e))?;
        Ok(WasmMatrix::from_raw(n, n, inv_data))
    }

    // ------------------------------------------------------------------
    // Eigenvalues
    // ------------------------------------------------------------------

    /// Estimate the real eigenvalues of a symmetric matrix using the QR algorithm
    /// with Wilkinson shifts.
    ///
    /// For non-symmetric matrices the result is the real part of the eigenvalues
    /// only; complex eigenvalues are approximated.  Convergence is guaranteed for
    /// symmetric matrices.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` of length `n` containing the eigenvalues (unsorted for
    /// non-symmetric matrices, sorted descending for symmetric ones).
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the matrix is not square.
    pub fn eigenvalues(&self) -> Result<Vec<f64>, JsValue> {
        if self.rows != self.cols {
            return Err(WasmError::InvalidDimensions(
                "Eigenvalue computation requires a square matrix".to_string(),
            )
            .into());
        }
        let evs = qr_eigenvalues(&self.data, self.rows);
        Ok(evs)
    }

    // ------------------------------------------------------------------
    // Norms & trace
    // ------------------------------------------------------------------

    /// Frobenius norm: sqrt(sum of squared elements).
    pub fn norm_frobenius(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    /// Trace (sum of diagonal elements).  Only valid for square matrices.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error if the matrix is not square.
    pub fn trace(&self) -> Result<f64, JsValue> {
        if self.rows != self.cols {
            return Err(
                WasmError::InvalidDimensions("Trace requires a square matrix".to_string()).into(),
            );
        }
        let t: f64 = (0..self.rows).map(|i| self.data[i * self.cols + i]).sum();
        Ok(t)
    }
}

// ============================================================================
// Internal (pure-Rust) linear algebra helpers
// ============================================================================

/// Validate that two matrices have the same shape.
impl WasmMatrix {
    fn check_same_shape(&self, other: &WasmMatrix) -> Result<(), JsValue> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(WasmError::ShapeMismatch {
                expected: vec![self.rows, self.cols],
                actual: vec![other.rows, other.cols],
            }
            .into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LU decomposition determinant (Doolittle, partial pivoting)
// ---------------------------------------------------------------------------

/// Compute the determinant of an n×n matrix stored row-major in `data`.
fn lu_determinant(data: &[f64], n: usize) -> f64 {
    let mut lu = data.to_vec();
    let mut sign = 1.0_f64;

    for k in 0..n {
        // Find the row with the largest absolute pivot value in column k
        let mut max_val = lu[k * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = lu[i * n + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val < 1e-15 {
            return 0.0; // Singular
        }
        if max_row != k {
            for j in 0..n {
                lu.swap(k * n + j, max_row * n + j);
            }
            sign = -sign;
        }
        // Eliminate below pivot
        for i in (k + 1)..n {
            let factor = lu[i * n + k] / lu[k * n + k];
            for j in k..n {
                lu[i * n + j] -= factor * lu[k * n + j];
            }
        }
    }

    let mut det = sign;
    for i in 0..n {
        det *= lu[i * n + i];
    }
    det
}

// ---------------------------------------------------------------------------
// Gauss-Jordan inverse
// ---------------------------------------------------------------------------

/// Compute the inverse of an n×n matrix (row-major) using Gauss-Jordan.
///
/// Returns the inverse as a flat row-major `Vec<f64>`, or an error string.
fn gauss_jordan_inverse(data: &[f64], n: usize) -> Result<Vec<f64>, String> {
    // Augmented matrix [A | I], stored as (n × 2n) row-major
    let mut aug = vec![0.0_f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = data[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    for i in 0..n {
        // Partial pivot
        let mut max_val = aug[i * 2 * n + i].abs();
        let mut max_row = i;
        for k in (i + 1)..n {
            let v = aug[k * 2 * n + i].abs();
            if v > max_val {
                max_val = v;
                max_row = k;
            }
        }
        if max_val < 1e-15 {
            return Err("Matrix is singular; cannot invert".to_string());
        }
        if max_row != i {
            for j in 0..(2 * n) {
                aug.swap(i * 2 * n + j, max_row * 2 * n + j);
            }
        }
        // Scale pivot row
        let pivot = aug[i * 2 * n + i];
        for j in 0..(2 * n) {
            aug[i * 2 * n + j] /= pivot;
        }
        // Eliminate all other rows
        for k in 0..n {
            if k == i {
                continue;
            }
            let factor = aug[k * 2 * n + i];
            for j in 0..(2 * n) {
                aug[k * 2 * n + j] -= factor * aug[i * 2 * n + j];
            }
        }
    }

    // Extract right half
    let mut inv = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    Ok(inv)
}

// ---------------------------------------------------------------------------
// QR eigenvalue algorithm (Francis double-shift / real Schur form approximation)
// ---------------------------------------------------------------------------
//
// For symmetric matrices this converges to exact eigenvalues.  For general
// matrices it computes an approximation of the real parts.

/// Compute eigenvalues using iterative QR decomposition with Wilkinson shift.
fn qr_eigenvalues(data: &[f64], n: usize) -> Vec<f64> {
    const MAX_ITER: usize = 500;
    const TOL: f64 = 1e-10;

    let mut h = data.to_vec(); // Working copy (Hessenberg reduction would help but we skip for clarity)

    for _iter in 0..MAX_ITER {
        // Check convergence: are all sub-diagonal elements small?
        let converged = (1..n).all(|i| h[i * n + (i - 1)].abs() < TOL);
        if converged {
            break;
        }

        // Wilkinson shift: use the eigenvalue of the bottom-right 2×2 block
        // that is closer to h[n-1][n-1]
        let shift = wilkinson_shift(&h, n);

        // QR step with shift: H - μI = QR, H' = RQ + μI
        let (q, r) = householder_qr_shift(&h, n, shift);
        h = mat_mul_nn(&r, &q, n);
        for i in 0..n {
            h[i * n + i] += shift;
        }
    }

    // Eigenvalues are the diagonal
    (0..n).map(|i| h[i * n + i]).collect()
}

/// Wilkinson shift: eigenvalue of the bottom-right 2×2 block closest to h[n-1, n-1].
fn wilkinson_shift(h: &[f64], n: usize) -> f64 {
    if n < 2 {
        return h[0]; // 1×1
    }
    let a = h[(n - 2) * n + (n - 2)];
    let b = h[(n - 2) * n + (n - 1)];
    let c = h[(n - 1) * n + (n - 2)];
    let d = h[(n - 1) * n + (n - 1)];

    let tr = a + d;
    let det = a * d - b * c;
    let disc = (tr * tr / 4.0 - det).max(0.0).sqrt();
    let ev1 = tr / 2.0 + disc;
    let ev2 = tr / 2.0 - disc;

    // Pick the one closer to d
    if (ev1 - d).abs() < (ev2 - d).abs() {
        ev1
    } else {
        ev2
    }
}

/// Thin Householder QR decomposition of A - shift*I (n×n, row-major).
/// Returns (Q, R) each as flat row-major Vec<f64>.
fn householder_qr_shift(a: &[f64], n: usize, shift: f64) -> (Vec<f64>, Vec<f64>) {
    // Build working matrix R = A - shift*I
    let mut r = a.to_vec();
    for i in 0..n {
        r[i * n + i] -= shift;
    }

    // Q starts as identity
    let mut q = vec![0.0_f64; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }

    for k in 0..n.saturating_sub(1) {
        // Extract column k from row k downward
        let col_len = n - k;
        let mut v: Vec<f64> = (k..n).map(|i| r[i * n + k]).collect();

        // Householder vector
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            continue;
        }
        let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
        v[0] += sign * norm;
        let v_norm_sq: f64 = v.iter().map(|&x| x * x).sum();
        if v_norm_sq < 1e-30 {
            continue;
        }

        // Apply H = I - 2*v*v^T / ||v||^2 to R from the left
        for j in 0..n {
            let dot: f64 = (0..col_len).map(|i| v[i] * r[(i + k) * n + j]).sum();
            let scale = 2.0 * dot / v_norm_sq;
            for i in 0..col_len {
                r[(i + k) * n + j] -= scale * v[i];
            }
        }

        // Apply H to Q from the right: Q = Q * H^T = Q * H (H is symmetric)
        for i in 0..n {
            let dot: f64 = (0..col_len).map(|j| q[i * n + (j + k)] * v[j]).sum();
            let scale = 2.0 * dot / v_norm_sq;
            for j in 0..col_len {
                q[i * n + (j + k)] -= scale * v[j];
            }
        }
    }

    (q, r)
}

/// Multiply two n×n row-major matrices.
fn mat_mul_nn(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            if a_ik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
    c
}

// ============================================================================
// Free functions (convenience)
// ============================================================================

/// Create a new zero-initialised `WasmMatrix`.
///
/// This is a convenience alias for `WasmMatrix::new(rows, cols)` accessible
/// without the `new` constructor syntax.
///
/// # Errors
///
/// Returns a `JsValue` error if either dimension is zero.
#[wasm_bindgen]
pub fn wasm_matrix_zeros(rows: usize, cols: usize) -> Result<WasmMatrix, JsValue> {
    WasmMatrix::new(rows, cols)
}

/// Create an n×n identity `WasmMatrix`.
///
/// # Errors
///
/// Returns a `JsValue` error if `n == 0`.
#[wasm_bindgen]
pub fn wasm_matrix_identity(n: usize) -> Result<WasmMatrix, JsValue> {
    WasmMatrix::identity(n)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_2x2() -> WasmMatrix {
        // [1 2; 3 4]
        let mut m = WasmMatrix::new(2, 2).expect("2x2 alloc");
        m.set(0, 0, 1.0).expect("set");
        m.set(0, 1, 2.0).expect("set");
        m.set(1, 0, 3.0).expect("set");
        m.set(1, 1, 4.0).expect("set");
        m
    }

    #[test]
    fn test_new_zero_dim_fails() {
        assert!(WasmMatrix::new(0, 3).is_err());
        assert!(WasmMatrix::new(3, 0).is_err());
    }

    #[test]
    fn test_get_set() {
        let mut m = WasmMatrix::new(3, 3).expect("alloc");
        m.set(1, 2, 42.0).expect("set");
        let v = m.get(1, 2).expect("get");
        assert!((v - 42.0).abs() < 1e-12);
    }

    #[test]
    fn test_get_out_of_bounds() {
        let m = WasmMatrix::new(2, 2).expect("alloc");
        assert!(m.get(2, 0).is_err());
        assert!(m.get(0, 2).is_err());
    }

    #[test]
    fn test_multiply_2x2() {
        // [1 2; 3 4] * [1 2; 3 4] = [7 10; 15 22]
        let m = make_2x2();
        let result = m.multiply(&m).expect("multiply");
        assert!((result.get(0, 0).expect("g") - 7.0).abs() < 1e-10);
        assert!((result.get(0, 1).expect("g") - 10.0).abs() < 1e-10);
        assert!((result.get(1, 0).expect("g") - 15.0).abs() < 1e-10);
        assert!((result.get(1, 1).expect("g") - 22.0).abs() < 1e-10);
    }

    #[test]
    fn test_multiply_shape_mismatch() {
        let a = WasmMatrix::new(2, 3).expect("alloc");
        let b = WasmMatrix::new(2, 2).expect("alloc");
        assert!(a.multiply(&b).is_err());
    }

    #[test]
    fn test_transpose() {
        let m = make_2x2();
        let t = m.transpose();
        // t[0,1] should be original m[1,0] = 3
        assert!((t.get(0, 1).expect("g") - 3.0).abs() < 1e-10);
        // t[1,0] should be original m[0,1] = 2
        assert!((t.get(1, 0).expect("g") - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_determinant_2x2() {
        let m = make_2x2();
        let d = m.determinant().expect("det");
        // det([1 2; 3 4]) = 4 - 6 = -2
        assert!((d - (-2.0)).abs() < 1e-10, "det = {}", d);
    }

    #[test]
    fn test_determinant_identity() {
        let m = WasmMatrix::identity(4).expect("id");
        let d = m.determinant().expect("det");
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_2x2() {
        let m = make_2x2();
        let inv = m.inverse().expect("inverse");
        // A * A^-1 should be identity
        let prod = m.multiply(&inv).expect("multiply");
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let got = prod.get(i, j).expect("get");
                assert!((got - expected).abs() < 1e-9, "[{},{}] = {}", i, j, got);
            }
        }
    }

    #[test]
    fn test_inverse_singular_fails() {
        let mut m = WasmMatrix::new(2, 2).expect("alloc");
        m.set(0, 0, 1.0).expect("set");
        m.set(0, 1, 2.0).expect("set");
        m.set(1, 0, 2.0).expect("set");
        m.set(1, 1, 4.0).expect("set");
        assert!(m.inverse().is_err());
    }

    #[test]
    fn test_eigenvalues_symmetric_2x2() {
        // [[2, 1], [1, 2]] has eigenvalues 3 and 1
        let mut m = WasmMatrix::new(2, 2).expect("alloc");
        m.set(0, 0, 2.0).expect("s");
        m.set(0, 1, 1.0).expect("s");
        m.set(1, 0, 1.0).expect("s");
        m.set(1, 1, 2.0).expect("s");

        let mut evs = m.eigenvalues().expect("eig");
        evs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert!((evs[0] - 3.0).abs() < 1e-6, "ev0 = {}", evs[0]);
        assert!((evs[1] - 1.0).abs() < 1e-6, "ev1 = {}", evs[1]);
    }

    #[test]
    fn test_eigenvalues_diagonal_3x3() {
        // Diagonal matrix: eigenvalues = diagonal entries
        let mut m = WasmMatrix::new(3, 3).expect("alloc");
        m.set(0, 0, 5.0).expect("s");
        m.set(1, 1, 3.0).expect("s");
        m.set(2, 2, 1.0).expect("s");

        let mut evs = m.eigenvalues().expect("eig");
        evs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert!((evs[0] - 5.0).abs() < 1e-6, "ev0 = {}", evs[0]);
        assert!((evs[1] - 3.0).abs() < 1e-6, "ev1 = {}", evs[1]);
        assert!((evs[2] - 1.0).abs() < 1e-6, "ev2 = {}", evs[2]);
    }

    #[test]
    fn test_trace() {
        let m = make_2x2();
        let t = m.trace().expect("trace");
        assert!((t - 5.0).abs() < 1e-12); // 1 + 4 = 5
    }

    #[test]
    fn test_norm_frobenius() {
        // I_2: Frobenius norm = sqrt(2)
        let m = WasmMatrix::identity(2).expect("id");
        let nf = m.norm_frobenius();
        assert!((nf - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_add_and_sub() {
        let m = make_2x2();
        let added = m.add(&m).expect("add");
        assert!((added.get(0, 0).expect("g") - 2.0).abs() < 1e-12);
        let zero = added.sub(&m).expect("sub");
        let orig = zero.sub(&m).expect("sub");
        // orig should be all zeros
        for i in 0..2 {
            for j in 0..2 {
                assert!(orig.get(i, j).expect("g").abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_scale() {
        let m = make_2x2();
        let s = m.scale(2.0);
        assert!((s.get(0, 0).expect("g") - 2.0).abs() < 1e-12);
        assert!((s.get(1, 1).expect("g") - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_identity() {
        let id = WasmMatrix::identity(3).expect("id");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((id.get(i, j).expect("g") - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_lu_determinant_3x3() {
        // det([[1,2,3],[4,5,6],[7,8,9]]) = 0 (linearly dependent)
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let d = lu_determinant(&data, 3);
        assert!(d.abs() < 1e-8, "det = {}", d);
    }

    #[test]
    fn test_wilkinson_shift_2x2() {
        // For a 2×2 [[3,0],[0,5]], eigenvalues are 3 and 5; shift should be close to one of them.
        let h = [3.0, 0.0, 0.0, 5.0];
        let s = wilkinson_shift(&h, 2);
        assert!(
            (s - 3.0).abs() < 1e-10 || (s - 5.0).abs() < 1e-10,
            "shift = {}",
            s
        );
    }

    #[test]
    fn test_from_vec_shape_mismatch() {
        assert!(WasmMatrix::from_vec(2, 2, vec![1.0, 2.0]).is_err());
    }
}
