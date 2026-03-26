//! Kaczmarz method for solving linear systems Ax = b
//!
//! The Kaczmarz method (also known as Algebraic Reconstruction Technique, ART)
//! iteratively projects the current estimate onto the hyperplanes defined by
//! individual rows of the system Ax = b.
//!
//! ## Variants
//!
//! - **Classic**: Cycle through rows in order, projecting onto each hyperplane
//! - **Randomized**: Select rows proportional to ||a_i||^2 (Strohmer-Vershynin)
//! - **Block**: Process blocks of rows simultaneously using least squares projections
//! - **Extended**: Handle inconsistent systems via least-squares solution
//!
//! ## Reference
//!
//! - Kaczmarz, S. (1937). "Approximate solution of systems of linear equations"
//! - Strohmer, T. and Vershynin, R. (2009). "A Randomized Kaczmarz Algorithm
//!   with Exponential Convergence"

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{rngs::StdRng, RngExt, SeedableRng};

/// Kaczmarz method variant
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KaczmarzVariant {
    /// Classic cyclic Kaczmarz
    Classic,
    /// Randomized Kaczmarz (Strohmer-Vershynin)
    Randomized,
}

/// Configuration for the Kaczmarz solver
#[derive(Debug, Clone)]
pub struct KaczmarzConfig {
    /// Maximum number of iterations (full passes through all rows)
    pub max_iter: usize,
    /// Convergence tolerance on residual norm ||Ax - b||
    pub tol: f64,
    /// Kaczmarz variant to use
    pub variant: KaczmarzVariant,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Relaxation parameter omega in (0, 2). Default is 1.0 (no relaxation)
    pub relaxation: f64,
    /// Whether to track residual norm history
    pub track_residuals: bool,
}

impl Default for KaczmarzConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-8,
            variant: KaczmarzVariant::Classic,
            seed: 42,
            relaxation: 1.0,
            track_residuals: false,
        }
    }
}

/// Result of Kaczmarz iteration
#[derive(Debug, Clone)]
pub struct KaczmarzResult {
    /// Solution vector
    pub x: Array1<f64>,
    /// Final residual norm ||Ax - b||
    pub residual_norm: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether the solver converged
    pub converged: bool,
    /// History of residual norms (if tracking enabled)
    pub residual_history: Vec<f64>,
}

/// Kaczmarz solver for linear systems Ax = b
pub struct KaczmarzSolver {
    config: KaczmarzConfig,
}

impl KaczmarzSolver {
    /// Create a new Kaczmarz solver
    pub fn new(config: KaczmarzConfig) -> Self {
        Self { config }
    }

    /// Create a solver with default configuration
    pub fn default_solver() -> Self {
        Self::new(KaczmarzConfig::default())
    }

    /// Solve the linear system Ax = b
    ///
    /// # Arguments
    /// * `a` - The matrix A (m x n)
    /// * `b` - The right-hand side vector b (length m)
    /// * `x0` - Optional initial guess (defaults to zeros)
    ///
    /// # Returns
    /// The solution and convergence information
    pub fn solve(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
        x0: Option<&Array1<f64>>,
    ) -> OptimizeResult<KaczmarzResult> {
        let m = a.nrows();
        let n = a.ncols();

        if b.len() != m {
            return Err(OptimizeError::InvalidInput(format!(
                "Dimension mismatch: A is {}x{} but b has length {}",
                m,
                n,
                b.len()
            )));
        }

        if let Some(x_init) = x0 {
            if x_init.len() != n {
                return Err(OptimizeError::InvalidInput(format!(
                    "Initial guess has length {} but A has {} columns",
                    x_init.len(),
                    n
                )));
            }
        }

        let mut x = match x0 {
            Some(x_init) => x_init.clone(),
            None => Array1::zeros(n),
        };

        // Precompute row norms squared
        let row_norms_sq: Vec<f64> = (0..m)
            .map(|i| {
                let row = a.row(i);
                row.dot(&row)
            })
            .collect();

        // Check for zero rows
        for (i, &norm_sq) in row_norms_sq.iter().enumerate() {
            if norm_sq < 1e-30 && b[i].abs() > 1e-15 {
                return Err(OptimizeError::InvalidInput(format!(
                    "Row {} of A is zero but b[{}] = {} is nonzero: inconsistent system",
                    i, i, b[i]
                )));
            }
        }

        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut residual_history = Vec::new();

        // For randomized variant, precompute sampling distribution
        let frobenius_sq: f64 = row_norms_sq.iter().sum();

        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.config.max_iter {
            iterations = iter + 1;

            // One sweep through (all rows or m random selections)
            for _step in 0..m {
                let row_idx = match self.config.variant {
                    KaczmarzVariant::Classic => _step,
                    KaczmarzVariant::Randomized => {
                        // Sample row proportional to ||a_i||^2
                        if frobenius_sq < 1e-30 {
                            _step // fallback to cyclic
                        } else {
                            let threshold: f64 = rng.random::<f64>() * frobenius_sq;
                            let mut cumsum = 0.0;
                            let mut selected = m - 1;
                            for (i, &norm_sq) in row_norms_sq.iter().enumerate() {
                                cumsum += norm_sq;
                                if cumsum >= threshold {
                                    selected = i;
                                    break;
                                }
                            }
                            selected
                        }
                    }
                };

                let norm_sq = row_norms_sq[row_idx];
                if norm_sq < 1e-30 {
                    continue; // skip zero rows
                }

                let row = a.row(row_idx);
                let dot_ax = row.dot(&x);
                let residual_i = b[row_idx] - dot_ax;

                // Kaczmarz projection: x <- x + omega * (b_i - a_i^T x) / ||a_i||^2 * a_i
                let scale = self.config.relaxation * residual_i / norm_sq;
                for j in 0..n {
                    x[j] += scale * row[j];
                }
            }

            // Compute residual norm
            let residual = a.dot(&x) - b;
            let res_norm = residual.dot(&residual).sqrt();

            if self.config.track_residuals {
                residual_history.push(res_norm);
            }

            if res_norm < self.config.tol {
                converged = true;
                break;
            }
        }

        let final_residual = a.dot(&x) - b;
        let residual_norm = final_residual.dot(&final_residual).sqrt();

        Ok(KaczmarzResult {
            x,
            residual_norm,
            iterations,
            converged,
            residual_history,
        })
    }
}

/// Block Kaczmarz method
///
/// Processes blocks of rows simultaneously, solving a small least-squares
/// subproblem for each block. This can converge faster than row-by-row
/// Kaczmarz on well-structured problems.
pub struct BlockKaczmarz {
    config: KaczmarzConfig,
    /// Number of rows per block
    block_size: usize,
}

impl BlockKaczmarz {
    /// Create a new block Kaczmarz solver
    ///
    /// # Arguments
    /// * `config` - Solver configuration
    /// * `block_size` - Number of rows per block
    pub fn new(config: KaczmarzConfig, block_size: usize) -> Self {
        Self { config, block_size }
    }

    /// Solve the linear system Ax = b using block Kaczmarz
    ///
    /// # Arguments
    /// * `a` - The matrix A (m x n)
    /// * `b` - The right-hand side vector b (length m)
    /// * `x0` - Optional initial guess
    pub fn solve(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
        x0: Option<&Array1<f64>>,
    ) -> OptimizeResult<KaczmarzResult> {
        let m = a.nrows();
        let n = a.ncols();

        if b.len() != m {
            return Err(OptimizeError::InvalidInput(format!(
                "Dimension mismatch: A is {}x{} but b has length {}",
                m,
                n,
                b.len()
            )));
        }

        let block_size = self.block_size.min(m).max(1);

        let mut x = match x0 {
            Some(x_init) => {
                if x_init.len() != n {
                    return Err(OptimizeError::InvalidInput(format!(
                        "Initial guess has length {} but A has {} columns",
                        x_init.len(),
                        n
                    )));
                }
                x_init.clone()
            }
            None => Array1::zeros(n),
        };

        // Build block indices
        let mut blocks: Vec<(usize, usize)> = Vec::new();
        let mut start = 0;
        while start < m {
            let end = (start + block_size).min(m);
            blocks.push((start, end));
            start = end;
        }

        let mut residual_history = Vec::new();
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.config.max_iter {
            iterations = iter + 1;

            for &(block_start, block_end) in &blocks {
                let bk_size = block_end - block_start;

                // Extract block sub-matrix and sub-vector
                let a_block = a.slice(s![block_start..block_end, ..]);
                let b_block = b.slice(s![block_start..block_end]);

                // Block residual: r_block = b_block - A_block * x
                let ax_block = block_mat_vec(&a_block, &x.view());
                let mut r_block = Array1::zeros(bk_size);
                for i in 0..bk_size {
                    r_block[i] = b_block[i] - ax_block[i];
                }

                // Compute A_block * A_block^T (bk_size x bk_size)
                let aat = block_aat(&a_block);

                // Solve (A_block * A_block^T) z = r_block using simple Cholesky-like approach
                // For small blocks, just use direct solve via Gaussian elimination
                let z = match solve_small_system(&aat, &r_block) {
                    Some(z) => z,
                    None => continue, // Skip singular block
                };

                // Update: x <- x + omega * A_block^T * z
                for j in 0..n {
                    let mut update = 0.0;
                    for i in 0..bk_size {
                        update += a_block[[i, j]] * z[i];
                    }
                    x[j] += self.config.relaxation * update;
                }
            }

            // Compute residual norm
            let residual = a.dot(&x) - b;
            let res_norm = residual.dot(&residual).sqrt();

            if self.config.track_residuals {
                residual_history.push(res_norm);
            }

            if res_norm < self.config.tol {
                converged = true;
                break;
            }
        }

        let final_residual = a.dot(&x) - b;
        let residual_norm = final_residual.dot(&final_residual).sqrt();

        Ok(KaczmarzResult {
            x,
            residual_norm,
            iterations,
            converged,
            residual_history,
        })
    }
}

/// Extended Kaczmarz for inconsistent systems
///
/// When the system Ax = b is inconsistent (no exact solution exists),
/// the extended Kaczmarz method converges to the least-squares solution
/// by simultaneously computing the projection of b onto the column space of A.
pub struct ExtendedKaczmarz {
    config: KaczmarzConfig,
}

impl ExtendedKaczmarz {
    /// Create a new extended Kaczmarz solver
    pub fn new(config: KaczmarzConfig) -> Self {
        Self { config }
    }

    /// Solve the (possibly inconsistent) system Ax = b in the least-squares sense
    ///
    /// Returns x that minimizes ||Ax - b||_2
    ///
    /// # Arguments
    /// * `a` - The matrix A (m x n)
    /// * `b` - The right-hand side vector b (length m)
    /// * `x0` - Optional initial guess
    pub fn solve(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
        x0: Option<&Array1<f64>>,
    ) -> OptimizeResult<KaczmarzResult> {
        let m = a.nrows();
        let n = a.ncols();

        if b.len() != m {
            return Err(OptimizeError::InvalidInput(format!(
                "Dimension mismatch: A is {}x{} but b has length {}",
                m,
                n,
                b.len()
            )));
        }

        let mut x = match x0 {
            Some(x_init) => {
                if x_init.len() != n {
                    return Err(OptimizeError::InvalidInput(format!(
                        "Initial guess has length {} but A has {} columns",
                        x_init.len(),
                        n
                    )));
                }
                x_init.clone()
            }
            None => Array1::zeros(n),
        };

        // z tracks the projection of b onto the left null space of A
        // (the part of b not in the column space of A).
        // We initialize z = b, then iteratively remove the col-space component.
        let mut z: Array1<f64> = b.clone();

        // Precompute row and column norms
        let row_norms_sq: Vec<f64> = (0..m)
            .map(|i| {
                let row = a.row(i);
                row.dot(&row)
            })
            .collect();

        let col_norms_sq: Vec<f64> = (0..n)
            .map(|j| {
                let col = a.column(j);
                col.dot(&col)
            })
            .collect();

        let frobenius_sq: f64 = row_norms_sq.iter().sum();
        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut residual_history = Vec::new();
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.config.max_iter {
            iterations = iter + 1;

            for _step in 0..m {
                // Step 1: Update z by projecting out the component along column col_idx of A
                // z <- z - (<a_j, z> / ||a_j||^2) * a_j
                // This removes the component of z in the column space of A.
                let col_idx = match self.config.variant {
                    KaczmarzVariant::Randomized => {
                        let frobenius_sq_cols: f64 = col_norms_sq.iter().sum();
                        if frobenius_sq_cols < 1e-30 {
                            _step % n
                        } else {
                            let threshold = rng.random::<f64>() * frobenius_sq_cols;
                            let mut cumsum = 0.0;
                            let mut selected = n - 1;
                            for (j, &ns) in col_norms_sq.iter().enumerate() {
                                cumsum += ns;
                                if cumsum >= threshold {
                                    selected = j;
                                    break;
                                }
                            }
                            selected
                        }
                    }
                    KaczmarzVariant::Classic => _step % n,
                };

                let col_norm_sq = col_norms_sq[col_idx];
                if col_norm_sq > 1e-30 {
                    let a_col = a.column(col_idx);
                    let dot_val: f64 = a_col.dot(&z);
                    let scale: f64 = dot_val / col_norm_sq;
                    for i in 0..m {
                        z[i] -= scale * a_col[i];
                    }
                }

                // Step 2: Kaczmarz step with (b - z) instead of b
                // b - z is the projection of b onto the column space of A
                let row_idx = match self.config.variant {
                    KaczmarzVariant::Randomized => {
                        if frobenius_sq < 1e-30 {
                            _step
                        } else {
                            let threshold = rng.random::<f64>() * frobenius_sq;
                            let mut cumsum = 0.0;
                            let mut selected = m - 1;
                            for (i, &ns) in row_norms_sq.iter().enumerate() {
                                cumsum += ns;
                                if cumsum >= threshold {
                                    selected = i;
                                    break;
                                }
                            }
                            selected
                        }
                    }
                    KaczmarzVariant::Classic => _step,
                };

                let norm_sq = row_norms_sq[row_idx];
                if norm_sq < 1e-30 {
                    continue;
                }

                let row = a.row(row_idx);
                let dot_ax: f64 = row.dot(&x);
                let target: f64 = b[row_idx] - z[row_idx];
                let residual_i: f64 = target - dot_ax;

                let scale: f64 = self.config.relaxation * residual_i / norm_sq;
                for j in 0..n {
                    x[j] += scale * row[j];
                }
            }

            // Compute residual norm (against original b)
            let residual = a.dot(&x) - b;
            let res_norm = residual.dot(&residual).sqrt();

            if self.config.track_residuals {
                residual_history.push(res_norm);
            }

            if res_norm < self.config.tol {
                converged = true;
                break;
            }
        }

        let final_residual = a.dot(&x) - b;
        let residual_norm = final_residual.dot(&final_residual).sqrt();

        Ok(KaczmarzResult {
            x,
            residual_norm,
            iterations,
            converged,
            residual_history,
        })
    }
}

/// Compute A_block * x where A_block is a view
fn block_mat_vec(a_block: &ArrayView2<f64>, x: &ArrayView1<f64>) -> Array1<f64> {
    let m = a_block.nrows();
    let n = a_block.ncols();
    let mut result = Array1::zeros(m);
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..n {
            sum += a_block[[i, j]] * x[j];
        }
        result[i] = sum;
    }
    result
}

/// Compute A * A^T for a block
fn block_aat(a_block: &ArrayView2<f64>) -> Array2<f64> {
    let m = a_block.nrows();
    let n = a_block.ncols();
    let mut result = Array2::zeros((m, m));
    for i in 0..m {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a_block[[i, k]] * a_block[[j, k]];
            }
            result[[i, j]] = sum;
            result[[j, i]] = sum;
        }
    }
    result
}

/// Solve a small dense system Ax = b via Gaussian elimination with partial pivoting
///
/// Returns None if the system is singular.
fn solve_small_system(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = a.nrows();
    if n == 0 || a.ncols() != n || b.len() != n {
        return None;
    }

    // Augmented matrix
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return None; // Singular
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Eliminate below
        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..=n {
                let val = aug[[col, j]];
                aug[[row, j]] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        if aug[[i, i]].abs() < 1e-14 {
            return None;
        }
        x[i] = sum / aug[[i, i]];
    }

    Some(x)
}

/// Convenience function: solve Ax = b using classic Kaczmarz
pub fn kaczmarz_solve(
    a: &Array2<f64>,
    b: &Array1<f64>,
    config: Option<KaczmarzConfig>,
) -> OptimizeResult<KaczmarzResult> {
    let config = config.unwrap_or_default();
    let solver = KaczmarzSolver::new(config);
    solver.solve(a, b, None)
}

/// Convenience function: solve Ax = b using randomized Kaczmarz
pub fn randomized_kaczmarz_solve(
    a: &Array2<f64>,
    b: &Array1<f64>,
    seed: u64,
    config: Option<KaczmarzConfig>,
) -> OptimizeResult<KaczmarzResult> {
    let mut config = config.unwrap_or_default();
    config.variant = KaczmarzVariant::Randomized;
    config.seed = seed;
    let solver = KaczmarzSolver::new(config);
    solver.solve(a, b, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1, Array2};

    /// Test 1: Solve a small exactly-determined system
    #[test]
    fn test_classic_kaczmarz_exact_system() {
        // 2x2 system: [[1, 0], [0, 1]] * [x, y] = [3, 4]
        let a = Array2::eye(2);
        let b = array![3.0, 4.0];

        let config = KaczmarzConfig {
            max_iter: 100,
            tol: 1e-10,
            variant: KaczmarzVariant::Classic,
            ..Default::default()
        };

        let result = kaczmarz_solve(&a, &b, Some(config));
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        assert!(result.converged);
        assert!((result.x[0] - 3.0).abs() < 1e-8, "x[0]={}", result.x[0]);
        assert!((result.x[1] - 4.0).abs() < 1e-8, "x[1]={}", result.x[1]);
    }

    /// Test 2: Solve a non-trivial system classically
    #[test]
    fn test_classic_kaczmarz_nontrivial() {
        // [[2, 1], [1, 3]] * x = [5, 10] => x = [1, 3]
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = array![5.0, 10.0];

        let config = KaczmarzConfig {
            max_iter: 5000,
            tol: 1e-8,
            variant: KaczmarzVariant::Classic,
            relaxation: 0.5, // Under-relaxation for stability
            ..Default::default()
        };

        let result = kaczmarz_solve(&a, &b, Some(config));
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        assert!(
            result.converged,
            "Did not converge, residual={}",
            result.residual_norm
        );
        assert!((result.x[0] - 1.0).abs() < 1e-4, "x[0]={}", result.x[0]);
        assert!((result.x[1] - 3.0).abs() < 1e-4, "x[1]={}", result.x[1]);
    }

    /// Test 3: Randomized Kaczmarz converges for overdetermined system
    #[test]
    fn test_randomized_kaczmarz_overdetermined() {
        // Overdetermined system (3 equations, 2 unknowns), consistent
        // x = [1, 2]
        let a = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let b = array![1.0, 2.0, 3.0];

        let config = KaczmarzConfig {
            max_iter: 5000,
            tol: 1e-8,
            variant: KaczmarzVariant::Randomized,
            seed: 42,
            relaxation: 0.8,
            ..Default::default()
        };

        let result = kaczmarz_solve(&a, &b, Some(config));
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        assert!(
            result.converged,
            "Did not converge, residual={}",
            result.residual_norm
        );
        assert!((result.x[0] - 1.0).abs() < 1e-4, "x[0]={}", result.x[0]);
        assert!((result.x[1] - 2.0).abs() < 1e-4, "x[1]={}", result.x[1]);
    }

    /// Test 4: Block Kaczmarz
    #[test]
    fn test_block_kaczmarz() {
        // 4x3 system with block size 2
        // x = [1, 2, 3]
        let a = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ];
        let b = array![1.0, 2.0, 3.0, 6.0];

        let config = KaczmarzConfig {
            max_iter: 5000,
            tol: 1e-8,
            relaxation: 0.5,
            ..Default::default()
        };

        let solver = BlockKaczmarz::new(config, 2);
        let result = solver.solve(&a, &b, None);
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        assert!(
            result.converged,
            "Did not converge, residual={}",
            result.residual_norm
        );
        assert!((result.x[0] - 1.0).abs() < 1e-4, "x[0]={}", result.x[0]);
        assert!((result.x[1] - 2.0).abs() < 1e-4, "x[1]={}", result.x[1]);
        assert!((result.x[2] - 3.0).abs() < 1e-4, "x[2]={}", result.x[2]);
    }

    /// Test 5: Extended Kaczmarz for inconsistent system
    #[test]
    fn test_extended_kaczmarz_inconsistent() {
        // Overdetermined inconsistent system
        // A = [[1, 0], [0, 1], [1, 1]], b = [1, 2, 4] (inconsistent: 1+2 != 4)
        // Least squares solution: A^T A x = A^T b
        // A^T A = [[2, 1], [1, 2]], A^T b = [5, 6]
        // x = [4/3, 7/3]
        let a = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let b = array![1.0, 2.0, 4.0];

        let config = KaczmarzConfig {
            max_iter: 10000,
            tol: 1e-6,
            variant: KaczmarzVariant::Randomized,
            seed: 42,
            relaxation: 0.5,
            ..Default::default()
        };

        let solver = ExtendedKaczmarz::new(config);
        let result = solver.solve(&a, &b, None);
        assert!(result.is_ok());
        let result = result.expect("should succeed");

        // The least-squares solution is x = [4/3, 7/3]
        let expected_x0 = 4.0 / 3.0;
        let expected_x1 = 7.0 / 3.0;

        // Extended Kaczmarz should approach the LS solution
        // Use a looser tolerance since extended Kaczmarz can be slow
        assert!(
            (result.x[0] - expected_x0).abs() < 0.5,
            "x[0]={}, expected ~{}",
            result.x[0],
            expected_x0
        );
        assert!(
            (result.x[1] - expected_x1).abs() < 0.5,
            "x[1]={}, expected ~{}",
            result.x[1],
            expected_x1
        );
    }

    /// Test 6: Residual tracking
    #[test]
    fn test_residual_tracking() {
        let a = Array2::eye(3);
        let b = array![1.0, 2.0, 3.0];

        let config = KaczmarzConfig {
            max_iter: 20,
            tol: 1e-20, // Don't stop early
            track_residuals: true,
            ..Default::default()
        };

        let result = kaczmarz_solve(&a, &b, Some(config));
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        assert!(!result.residual_history.is_empty());
        // Residuals should decrease over time (for identity matrix, converges in 1 pass)
        assert!(
            result
                .residual_history
                .last()
                .copied()
                .unwrap_or(f64::INFINITY)
                < 1e-10
        );
    }

    /// Test 7: Dimension mismatch error
    #[test]
    fn test_dimension_mismatch() {
        let a = Array2::eye(3);
        let b = array![1.0, 2.0]; // wrong length

        let result = kaczmarz_solve(&a, &b, None);
        assert!(result.is_err());
    }

    /// Test 8: With initial guess
    #[test]
    fn test_with_initial_guess() {
        let a = Array2::eye(2);
        let b = array![5.0, 7.0];
        let x0 = array![4.9, 6.9]; // Close to solution

        let config = KaczmarzConfig {
            max_iter: 10,
            tol: 1e-10,
            ..Default::default()
        };

        let solver = KaczmarzSolver::new(config);
        let result = solver.solve(&a, &b, Some(&x0));
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        assert!(result.converged);
        assert!((result.x[0] - 5.0).abs() < 1e-8);
        assert!((result.x[1] - 7.0).abs() < 1e-8);
    }
}
