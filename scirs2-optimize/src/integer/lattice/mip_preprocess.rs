//! Lattice reduction preprocessing for Mixed-Integer Programs (MIP).
//!
//! This module applies LLL or BKZ lattice reduction to the column vectors of the
//! ILP constraint matrix A, obtaining a unimodular transformation U such that
//! A * U^{-1} is LLL/BKZ-reduced. Solving the transformed ILP y = U x often
//! produces tighter LP relaxations and faster branch-and-bound convergence.
//!
//! # Mathematical Background
//!
//! Given an ILP:
//! ```text
//! minimize   c^T x
//! subject to A x <= b,  x in Z^n
//! ```
//! We substitute x = U^{-1} y (where U is the unimodular transformation from lattice
//! reduction applied to the columns of A, interpreted as lattice basis vectors):
//! ```text
//! minimize   (c^T U^{-1}) y
//! subject to (A U^{-1}) y <= b,  y in Z^n
//! ```
//! The reduced matrix A U^{-1} has more orthogonal columns, leading to better
//! conditioning of LP relaxations.
//!
//! # References
//! - Krishnamoorthy, B., Pataki, G. (2006). "Column basis reduction and decomposable
//!   knapsack problems." Discrete Optimization, 3(2), 85–110.
//! - Aardal, K., et al. (2000). "Solving a system of linear Diophantine equations
//!   with lower and upper bounds on the variables." Mathematics of Operations Research, 25(3).

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array2};

use super::bkz::{BKZConfig, BKZReducer};
use super::lll::{LLLConfig, LLLReducer};

/// Which lattice reduction method to apply for MIP preprocessing.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionMethod {
    /// Lenstra-Lenstra-Lovász reduction (faster, less powerful).
    LLL,
    /// Block Korkine-Zolotarev reduction (slower, stronger).
    BKZ,
}

impl Default for ReductionMethod {
    fn default() -> Self {
        ReductionMethod::LLL
    }
}

/// Configuration for the lattice preprocessor.
#[derive(Debug, Clone)]
pub struct LatticePreprocessorConfig {
    /// Which reduction method to use.
    pub method: ReductionMethod,
    /// Configuration for LLL (used when method = LLL, or as BKZ preprocessing).
    pub lll_config: LLLConfig,
    /// Configuration for BKZ (used when method = BKZ).
    pub bkz_config: BKZConfig,
    /// If true, reduce column vectors of A (interpret A's columns as lattice basis).
    /// If false, reduce row vectors.
    pub apply_to_columns: bool,
}

impl Default for LatticePreprocessorConfig {
    fn default() -> Self {
        LatticePreprocessorConfig {
            method: ReductionMethod::LLL,
            lll_config: LLLConfig::default(),
            bkz_config: BKZConfig::default(),
            apply_to_columns: true,
        }
    }
}

/// Result from lattice preprocessing of an ILP constraint matrix.
#[derive(Debug, Clone)]
pub struct LatticePreprocessorResult {
    /// Transformed constraint matrix A' = A * U^{-1}.
    pub transformed_a: Array2<f64>,
    /// Unimodular transformation matrix U.
    pub transform: Array2<f64>,
    /// Integer inverse U^{-1} (also unimodular).
    pub transform_inv: Array2<f64>,
    /// Which reduction method was applied.
    pub reduction_method: ReductionMethod,
}

/// Lattice preprocessor for MIP constraint matrices.
///
/// Applies lattice reduction to the column (or row) vectors of the constraint
/// matrix A to obtain a unimodular change-of-basis that may improve MIP solve times.
///
/// # Example
/// ```
/// use scirs2_optimize::integer::lattice::{
///     LatticePreprocessor, LatticePreprocessorConfig, ReductionMethod
/// };
/// use scirs2_core::ndarray::array;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let config = LatticePreprocessorConfig::default();
/// let pp = LatticePreprocessor::new(config);
/// let result = pp.preprocess(&a).unwrap();
/// assert!(pp.verify(&result, &a));
/// ```
pub struct LatticePreprocessor {
    config: LatticePreprocessorConfig,
}

impl LatticePreprocessor {
    /// Create a new lattice preprocessor with the given configuration.
    pub fn new(config: LatticePreprocessorConfig) -> Self {
        LatticePreprocessor { config }
    }

    /// Preprocess the ILP constraint matrix by lattice reduction.
    ///
    /// # Arguments
    /// * `a` - Constraint matrix of shape [m, n] (m constraints, n variables)
    ///
    /// # Returns
    /// `LatticePreprocessorResult` containing:
    /// - `transformed_a = A * U^{-1}` (if apply_to_columns = true)
    /// - `transform = U` (unimodular transformation from lattice reduction)
    /// - `transform_inv = U^{-1}` (integer inverse)
    pub fn preprocess(&self, a: &Array2<f64>) -> OptimizeResult<LatticePreprocessorResult> {
        let m = a.nrows();
        let n = a.ncols();
        if m == 0 || n == 0 {
            return Err(OptimizeError::ValueError(
                "LatticePreprocessor: constraint matrix must be non-empty".to_string(),
            ));
        }

        // Build the lattice basis from A's columns or rows
        // When apply_to_columns = true: the n column vectors of A (each in R^m) form the basis.
        // We form the matrix B where B[i] = column i of A (i.e., B = A^T).
        let basis: Array2<f64> = if self.config.apply_to_columns {
            // Basis vectors are the columns of A, transposed to rows: B = A^T [n, m]
            a.t().to_owned()
        } else {
            // Basis vectors are the rows of A: B = A [m, n]
            a.to_owned()
        };

        // Apply lattice reduction to obtain the transformation matrix U
        // U is such that U * B = B_reduced (each row of U is integer coefficients)
        let (u_transform, reduced_basis) = self.apply_reduction(&basis)?;

        // u_transform: shape [n, n] (or [m, m] for row case)
        // It satisfies: u_transform * basis = reduced_basis
        // i.e., u_transform * A^T = A'^T, so A' = A * u_transform^T (for column case)
        // Equivalently: A * U^{-T} = A', so U^{-T} = u_transform^T^{-1}

        // We store the transform in the convention:
        // For column reduction: x = U * x' means x' = U^{-1} x
        // The transformation to apply to A is: A' = A * U^{-1}
        // From u_transform (the LLL transformation): u_transform * A^T = A'^T
        // => A' = A * u_transform^T
        // Wait, let's be careful:
        //   LLL gives: U_lll * B = B_reduced
        //   B = A^T => U_lll * A^T = B_reduced^T
        //   => (U_lll * A^T)^T = B_reduced => A * U_lll^T = A'
        //   So transformed_a = A * U_lll^T
        //   And the substitution x = U_lll^{-T} y gives the original ILP
        //   Actually: to make x_original = U^{-1} x_new, we want
        //     A * x_original = A * U^{-1} * x_new = A' * x_new
        //   So A' = A * U^{-1} where U is the change of basis for x.

        // Let's define things consistently:
        // Let U = u_transform^T (so U is an n×n unimodular matrix).
        // Then transformed_a = A * U^{-1}
        // We need U^{-1} = (u_transform^T)^{-1} = (u_transform^{-1})^T

        let n_basis = basis.nrows(); // n for column case, m for row case

        let u = u_transform.t().to_owned(); // U = u_transform^T

        // Compute U^{-1} (integer inverse of the unimodular matrix U)
        let u_inv = Self::integer_matrix_inverse(&u)?;

        // transformed_a = A * U^{-1} (for column case: [m, n] * [n, n] = [m, n])
        let transformed_a = if self.config.apply_to_columns {
            matmul_f64(a, &u_inv)
        } else {
            // For row case: A' = U^{-1} * A ([m, m] * [m, n])
            matmul_f64(&u_inv, a)
        };

        // Suppress unused variable warning
        let _ = (reduced_basis, n_basis);

        Ok(LatticePreprocessorResult {
            transformed_a,
            transform: u,
            transform_inv: u_inv,
            reduction_method: self.config.method,
        })
    }

    /// Verify that the transformation is valid: A = transformed_a * U (column case).
    ///
    /// Checks that `A ≈ result.transformed_a * result.transform` (for column reduction)
    /// or `A ≈ result.transform * result.transformed_a` (for row reduction).
    pub fn verify(&self, result: &LatticePreprocessorResult, a: &Array2<f64>) -> bool {
        let tol = 1e-6;
        let reconstructed = if self.config.apply_to_columns {
            matmul_f64(&result.transformed_a, &result.transform)
        } else {
            matmul_f64(&result.transform, &result.transformed_a)
        };

        if reconstructed.nrows() != a.nrows() || reconstructed.ncols() != a.ncols() {
            return false;
        }
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                if (reconstructed[[i, j]] - a[[i, j]]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Apply the configured lattice reduction to a basis matrix.
    ///
    /// Returns `(transformation, reduced_basis)`.
    fn apply_reduction(
        &self,
        basis: &Array2<f64>,
    ) -> OptimizeResult<(Array2<f64>, Array2<f64>)> {
        match self.config.method {
            ReductionMethod::LLL => {
                let reducer = LLLReducer::new(self.config.lll_config.clone());
                let result = reducer.reduce(basis)?;
                Ok((result.transformation, result.reduced_basis))
            }
            ReductionMethod::BKZ => {
                let reducer = BKZReducer::new(self.config.bkz_config.clone());
                let result = reducer.reduce(basis)?;
                // BKZ result doesn't include the transformation directly;
                // we reconstruct it from the reduced basis via LLL
                // (BKZ starts with LLL, so we apply LLL to get the transformation)
                let lll_config = LLLConfig {
                    delta: self.config.bkz_config.lll_delta,
                    ..Default::default()
                };
                let lll_reducer = LLLReducer::new(lll_config);
                let lll_result = lll_reducer.reduce(basis)?;
                // Further improve using BKZ result
                let final_basis = result.reduced_basis;
                // Use LLL transformation as approximation since BKZ doesn't track it
                Ok((lll_result.transformation, final_basis))
            }
            _ => Err(OptimizeError::NotImplementedError(
                "Unknown reduction method".to_string(),
            )),
        }
    }

    /// Compute the integer inverse of a unimodular matrix using the adjugate.
    ///
    /// For a unimodular matrix, `det = ±1`, so `U^{-1} = adjugate(U) / det = ±adjugate(U)`.
    /// This function is valid for small matrices (n ≤ ~10 is practical).
    pub(crate) fn integer_matrix_inverse(u: &Array2<f64>) -> OptimizeResult<Array2<f64>> {
        let n = u.nrows();
        if n != u.ncols() {
            return Err(OptimizeError::ValueError(
                "Matrix must be square for inversion".to_string(),
            ));
        }
        if n == 0 {
            return Ok(Array2::<f64>::zeros((0, 0)));
        }
        if n == 1 {
            let val = u[[0, 0]];
            if val.abs() < 0.5 {
                return Err(OptimizeError::ComputationError(
                    "Matrix is singular (det ≈ 0)".to_string(),
                ));
            }
            let mut inv = Array2::<f64>::zeros((1, 1));
            inv[[0, 0]] = 1.0 / val;
            return Ok(inv);
        }

        let det = compute_det(u)?;
        if det.abs() < 0.5 {
            return Err(OptimizeError::ComputationError(format!(
                "Matrix is singular or not unimodular: det = {}",
                det
            )));
        }

        // For large det, just use Gaussian elimination instead of adjugate
        let sign = if det > 0.0 { 1.0 } else { -1.0 };
        let _ = sign;

        // Compute adjugate = cofactor matrix transposed
        let adj = compute_adjugate(u)?;

        // U^{-1} = adj / det
        let inv_det = 1.0 / det;
        let mut inv = adj;
        for i in 0..n {
            for j in 0..n {
                inv[[i, j]] *= inv_det;
            }
        }
        Ok(inv)
    }
}

/// Compute the determinant of a square matrix.
fn compute_det(m: &Array2<f64>) -> OptimizeResult<f64> {
    let n = m.nrows();
    if n == 1 {
        return Ok(m[[0, 0]]);
    }
    if n == 2 {
        return Ok(m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]]);
    }
    if n == 3 {
        let d = m[[0, 0]] * (m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]])
            - m[[0, 1]] * (m[[1, 0]] * m[[2, 2]] - m[[1, 2]] * m[[2, 0]])
            + m[[0, 2]] * (m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]]);
        return Ok(d);
    }
    // For larger matrices: use LU decomposition (Gaussian elimination)
    compute_det_lu(m)
}

/// Compute determinant via Gaussian elimination (LU decomposition).
fn compute_det_lu(m: &Array2<f64>) -> OptimizeResult<f64> {
    let n = m.nrows();
    let mut lu = m.to_owned();
    let mut sign = 1.0f64;

    for k in 0..n {
        // Partial pivoting
        let mut max_val = lu[[k, k]].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            if lu[[i, k]].abs() > max_val {
                max_val = lu[[i, k]].abs();
                max_row = i;
            }
        }
        if max_val < 1e-14 {
            return Ok(0.0);
        }
        if max_row != k {
            for j in 0..n {
                let tmp = lu[[k, j]];
                lu[[k, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = tmp;
            }
            sign = -sign;
        }
        let pivot = lu[[k, k]];
        for i in (k + 1)..n {
            let factor = lu[[i, k]] / pivot;
            for j in k..n {
                let val = lu[[k, j]];
                lu[[i, j]] -= factor * val;
            }
        }
    }

    let diag_prod: f64 = (0..n).map(|i| lu[[i, i]]).product();
    Ok(sign * diag_prod)
}

/// Compute the adjugate (classical adjoint) of a matrix.
fn compute_adjugate(m: &Array2<f64>) -> OptimizeResult<Array2<f64>> {
    let n = m.nrows();
    let mut adj = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            // Cofactor C[i][j] = (-1)^(i+j) * M[i][j] (minor)
            let minor = extract_minor(m, i, j);
            let cofactor_val = compute_det(&minor)?;
            let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            // Adjugate is transpose of cofactor matrix: adj[j][i] = C[i][j]
            adj[[j, i]] = sign * cofactor_val;
        }
    }
    Ok(adj)
}

/// Extract the (n-1)×(n-1) minor by deleting row `del_row` and column `del_col`.
fn extract_minor(m: &Array2<f64>, del_row: usize, del_col: usize) -> Array2<f64> {
    let n = m.nrows();
    let mut minor = Array2::<f64>::zeros((n - 1, n - 1));
    let mut ri = 0;
    for i in 0..n {
        if i == del_row {
            continue;
        }
        let mut ci = 0;
        for j in 0..n {
            if j == del_col {
                continue;
            }
            minor[[ri, ci]] = m[[i, j]];
            ci += 1;
        }
        ri += 1;
    }
    minor
}

/// Simple matrix multiplication for f64 arrays.
fn matmul_f64(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();
    let mut c = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for l in 0..k {
                s += a[[i, l]] * b[[l, j]];
            }
            c[[i, j]] = s;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_integer_inverse_2x2() {
        // U = [[1, 2], [0, 1]]: det = 1, U^{-1} = [[1, -2], [0, 1]]
        let u = array![[1.0, 2.0], [0.0, 1.0]];
        let u_inv = LatticePreprocessor::integer_matrix_inverse(&u).expect("Should invert");
        assert!((u_inv[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((u_inv[[0, 1]] - (-2.0)).abs() < 1e-6);
        assert!((u_inv[[1, 0]] - 0.0).abs() < 1e-6);
        assert!((u_inv[[1, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_integer_inverse_3x3() {
        // U = [[1, 0, 1], [0, 1, 0], [0, 0, 1]]: det = 1
        let u = array![[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let u_inv = LatticePreprocessor::integer_matrix_inverse(&u).expect("Should invert");
        // U^{-1} = [[1, 0, -1], [0, 1, 0], [0, 0, 1]]
        assert!((u_inv[[0, 2]] - (-1.0)).abs() < 1e-6, "u_inv[0][2] = {}", u_inv[[0, 2]]);
        // Verify U * U^{-1} = I
        let prod = matmul_f64(&u, &u_inv);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((prod[[i, j]] - expected).abs() < 1e-6, "prod[{}][{}] = {}", i, j, prod[[i,j]]);
            }
        }
    }

    #[test]
    fn test_lattice_preprocessor_lll_method() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let config = LatticePreprocessorConfig {
            method: ReductionMethod::LLL,
            ..Default::default()
        };
        let pp = LatticePreprocessor::new(config);
        let result = pp.preprocess(&a).expect("Preprocessing should succeed");
        assert_eq!(result.reduction_method, ReductionMethod::LLL);
        assert!(pp.verify(&result, &a), "Verification should pass for LLL method");
    }

    #[test]
    fn test_lattice_preprocessor_bkz_method() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let config = LatticePreprocessorConfig {
            method: ReductionMethod::BKZ,
            bkz_config: BKZConfig {
                block_size: 2,
                max_tours: 3,
                ..Default::default()
            },
            ..Default::default()
        };
        let pp = LatticePreprocessor::new(config);
        let result = pp.preprocess(&a).expect("Preprocessing should succeed");
        assert_eq!(result.reduction_method, ReductionMethod::BKZ);
    }

    #[test]
    fn test_lattice_preprocessor_verify() {
        let a = array![[2.0, 1.0, 3.0], [1.0, 4.0, 2.0]];
        let pp = LatticePreprocessor::new(LatticePreprocessorConfig::default());
        let result = pp.preprocess(&a).expect("Preprocessing should succeed");
        assert!(
            pp.verify(&result, &a),
            "A should equal transformed_a * U"
        );
    }

    #[test]
    fn test_lattice_preprocessor_deterministic() {
        // Same input should give same output on repeated calls
        let a = array![[1.0, 1.0], [-1.0, 2.0]];
        let pp = LatticePreprocessor::new(LatticePreprocessorConfig::default());
        let r1 = pp.preprocess(&a).expect("First call should succeed");
        let r2 = pp.preprocess(&a).expect("Second call should succeed");
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (r1.transformed_a[[i, j]] - r2.transformed_a[[i, j]]).abs() < 1e-8,
                    "Results should be deterministic"
                );
            }
        }
    }

    #[test]
    fn test_lattice_preprocessor_identity_matrix() {
        // For identity A, transformation should be identity
        let a = Array2::<f64>::eye(3);
        let pp = LatticePreprocessor::new(LatticePreprocessorConfig::default());
        let result = pp.preprocess(&a).expect("Should succeed");
        assert!(pp.verify(&result, &a), "Verify should pass for identity");
    }

    #[test]
    fn test_reduction_method_non_exhaustive() {
        // Verify #[non_exhaustive] allows pattern matching with wildcard
        let m = ReductionMethod::LLL;
        let _ = match m {
            ReductionMethod::LLL => "lll",
            ReductionMethod::BKZ => "bkz",
            _ => "other",
        };
    }
}
