//! Randomized preconditioning for overdetermined least-squares systems
//!
//! This module implements randomized preconditioning techniques that dramatically
//! accelerate the solution of overdetermined least-squares problems by constructing
//! well-conditioned preconditioned systems via random sketching:
//!
//! - **SparseSign**: Sparse ±1 random matrix with fast application
//! - **SubsampledRandomizedHadamard (SRHT)**: Fast structured random projection
//! - **Blendenpik**: Randomized preconditioning for overdetermined LS (Avron et al. 2010)
//! - **LSRN**: Least-Squares via Random Normal preconditioning (Meng et al. 2014)
//! - **IterativeRefinementLS**: Mixed-precision iterative refinement for LS
//!
//! ## Mathematical Foundation
//!
//! For an overdetermined system A x ≈ b (m × n, m >> n), a right preconditioner R
//! is constructed so that A R⁻¹ has a small condition number. The preconditioned
//! system (A R⁻¹)(R x) = b is then solved by an iterative method such as LSQR.
//!
//! Blendenpik constructs R via QR of a random sketch S A, where S is a SRHT or
//! sparse sign matrix with O(n log n) application cost.
//!
//! ## References
//!
//! - Avron, Maymounkov, Toledo (2010). "Blendenpik: Supercharging LAPACK's least-squares solver"
//! - Meng, Saunders, Mahoney (2014). "LSRN: A parallel iterative solver for strongly
//!   over- or under-determined systems"
//! - Rokhlin, Tygert (2008). "A fast randomized algorithm for overdetermined linear
//!   least-squares regression"

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign, One, Zero};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution, Normal, Uniform};
use std::fmt::Debug;
use std::iter::Sum;

use crate::decomposition::qr;
use crate::error::{LinalgError, LinalgResult};

// ============================================================================
// SparseSign: Sparse ±1 random matrix
// ============================================================================

/// Sparse sign random matrix: each column has exactly `s` non-zeros, each ±1.
///
/// This provides an efficient O(s·n) matrix-vector product compared to O(m·n)
/// for dense random matrices. Typically s = O(log(m)) suffices for good
/// embedding properties.
#[derive(Debug, Clone)]
pub struct SparseSign {
    /// Number of rows (sketch dimension)
    pub rows: usize,
    /// Number of columns (original dimension)
    pub cols: usize,
    /// Sparsity: number of non-zeros per column
    pub sparsity: usize,
    /// Row indices of non-zeros (length cols × sparsity)
    row_indices: Vec<usize>,
    /// Signs (±1) for each non-zero (length cols × sparsity)
    signs: Vec<f64>,
    /// Scaling factor 1/sqrt(sparsity)
    scale: f64,
}

impl SparseSign {
    /// Construct a sparse sign matrix with given dimensions and sparsity.
    ///
    /// # Arguments
    ///
    /// * `rows` - Sketch dimension (number of rows)
    /// * `cols` - Input dimension (number of columns)
    /// * `sparsity` - Non-zeros per column (default: max(1, ceil(log2(rows))))
    /// * `seed` - Optional RNG seed for reproducibility
    pub fn new(
        rows: usize,
        cols: usize,
        sparsity: Option<usize>,
        seed: Option<u64>,
    ) -> LinalgResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(LinalgError::ShapeError(
                "SparseSign: rows and cols must be positive".to_string(),
            ));
        }

        let s = sparsity.unwrap_or_else(|| {
            let log2r = (rows as f64).log2().ceil() as usize;
            log2r.max(1).min(rows)
        });

        if s > rows {
            return Err(LinalgError::ShapeError(format!(
                "SparseSign: sparsity {s} cannot exceed rows {rows}"
            )));
        }

        let mut rng = {
            let s = seed.unwrap_or_else(|| {
                let mut tr = scirs2_core::random::rng();
                scirs2_core::random::Rng::random::<u64>(&mut tr)
            });
            scirs2_core::random::seeded_rng(s)
        };

        let mut row_indices = vec![0usize; cols * s];
        let mut signs = vec![0.0f64; cols * s];

        let uniform_sign = Uniform::new(0u8, 2).map_err(|e| {
            LinalgError::ComputationError(format!("Failed to create uniform distribution: {e}"))
        })?;

        for col in 0..cols {
            // Sample s distinct row indices without replacement (Fisher-Yates)
            let mut pool: Vec<usize> = (0..rows).collect();
            for k in 0..s {
                let j = (Uniform::new(k, rows)
                    .map_err(|e| {
                        LinalgError::ComputationError(format!("Failed to sample index: {e}"))
                    })?
                    .sample(&mut rng)) as usize;
                pool.swap(k, j);
                let row = pool[k];
                row_indices[col * s + k] = row;
                let sign_bit = uniform_sign.sample(&mut rng);
                signs[col * s + k] = if sign_bit == 0 { 1.0 } else { -1.0 };
            }
        }

        let scale = 1.0 / (s as f64).sqrt();

        Ok(Self {
            rows,
            cols,
            sparsity: s,
            row_indices,
            signs,
            scale,
        })
    }

    /// Apply the sparse sign matrix: y = S * x, result is `rows`-vector.
    ///
    /// # Arguments
    ///
    /// * `x` - Input vector of length `self.cols`
    pub fn apply(&self, x: &ArrayView1<f64>) -> LinalgResult<Array1<f64>> {
        if x.len() != self.cols {
            return Err(LinalgError::DimensionError(format!(
                "SparseSign::apply: x has length {} but cols={}",
                x.len(),
                self.cols
            )));
        }

        let mut y = Array1::zeros(self.rows);
        let s = self.sparsity;

        for col in 0..self.cols {
            let xval = x[col];
            for k in 0..s {
                let row = self.row_indices[col * s + k];
                let sign = self.signs[col * s + k];
                y[row] += self.scale * sign * xval;
            }
        }

        Ok(y)
    }

    /// Apply S to each column of matrix A (result: rows × n).
    ///
    /// Computes S * A for a `cols × n` matrix A.
    pub fn apply_matrix(&self, a: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
        let (nrows_a, ncols_a) = a.dim();
        if nrows_a != self.cols {
            return Err(LinalgError::DimensionError(format!(
                "SparseSign::apply_matrix: A has {} rows but cols={}",
                nrows_a, self.cols
            )));
        }

        let mut y = Array2::zeros((self.rows, ncols_a));
        let s = self.sparsity;

        for col in 0..self.cols {
            for k in 0..s {
                let row = self.row_indices[col * s + k];
                let sign = self.signs[col * s + k];
                let coeff = self.scale * sign;
                for j in 0..ncols_a {
                    y[[row, j]] += coeff * a[[col, j]];
                }
            }
        }

        Ok(y)
    }

    /// Apply S^T (transpose): y = S^T * x, result is `cols`-vector.
    pub fn apply_transpose(&self, x: &ArrayView1<f64>) -> LinalgResult<Array1<f64>> {
        if x.len() != self.rows {
            return Err(LinalgError::DimensionError(format!(
                "SparseSign::apply_transpose: x has length {} but rows={}",
                x.len(),
                self.rows
            )));
        }

        let mut y = Array1::zeros(self.cols);
        let s = self.sparsity;

        for col in 0..self.cols {
            let mut val = 0.0f64;
            for k in 0..s {
                let row = self.row_indices[col * s + k];
                let sign = self.signs[col * s + k];
                val += sign * x[row];
            }
            y[col] = self.scale * val;
        }

        Ok(y)
    }
}

// ============================================================================
// SubsampledRandomizedHadamard (SRHT)
// ============================================================================

/// Subsampled Randomized Hadamard Transform (SRHT) for fast preconditioning.
///
/// The SRHT maps an n-vector to a k-vector via:
///   y = sqrt(n/k) * R * H * D * x
/// where:
///   - D is a random diagonal ±1 matrix
///   - H is the normalized Walsh-Hadamard transform (n must be a power of 2)
///   - R is a random row-sampling operator selecting k rows uniformly
///
/// Cost: O(n log n) per matrix-vector product.
#[derive(Debug, Clone)]
pub struct SubsampledRandomizedHadamard {
    /// Input dimension (must be power of 2)
    pub n: usize,
    /// Sketch dimension k ≤ n
    pub k: usize,
    /// Random diagonal signs (length n)
    diag_signs: Vec<f64>,
    /// Sampled row indices (length k)
    sampled_rows: Vec<usize>,
    /// Scaling factor sqrt(n / k)
    scale: f64,
}

impl SubsampledRandomizedHadamard {
    /// Construct an SRHT with sketch dimension k.
    ///
    /// # Arguments
    ///
    /// * `n` - Input dimension (must be a power of 2, or will be padded)
    /// * `k` - Sketch dimension (k ≤ n)
    /// * `seed` - Optional RNG seed
    pub fn new(n: usize, k: usize, seed: Option<u64>) -> LinalgResult<Self> {
        if n == 0 {
            return Err(LinalgError::ShapeError(
                "SRHT: n must be positive".to_string(),
            ));
        }
        if k == 0 || k > n {
            return Err(LinalgError::ShapeError(format!(
                "SRHT: k must satisfy 1 ≤ k ≤ n={n}, got k={k}"
            )));
        }

        // Round n up to next power of 2
        let n_padded = n.next_power_of_two();

        let mut rng = {
            let s = seed.unwrap_or_else(|| {
                let mut tr = scirs2_core::random::rng();
                scirs2_core::random::Rng::random::<u64>(&mut tr)
            });
            scirs2_core::random::seeded_rng(s)
        };

        // Random diagonal signs ±1
        let uniform_bit = Uniform::new(0u8, 2).map_err(|e| {
            LinalgError::ComputationError(format!("Failed to create uniform distribution: {e}"))
        })?;
        let diag_signs: Vec<f64> = (0..n_padded)
            .map(|_| {
                if uniform_bit.sample(&mut rng) == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();

        // Sample k distinct rows uniformly from {0..n_padded}
        let mut pool: Vec<usize> = (0..n_padded).collect();
        let k_actual = k.min(n_padded);
        for i in 0..k_actual {
            let j_range = Uniform::new(i, n_padded).map_err(|e| {
                LinalgError::ComputationError(format!("Failed to sample row index: {e}"))
            })?;
            let j = j_range.sample(&mut rng);
            pool.swap(i, j);
        }
        let sampled_rows: Vec<usize> = pool[..k_actual].to_vec();

        let scale = (n_padded as f64 / k_actual as f64).sqrt();

        Ok(Self {
            n: n_padded,
            k: k_actual,
            diag_signs,
            sampled_rows,
            scale,
        })
    }

    /// Apply the SRHT: y = scale * R * H * D * x
    ///
    /// Returns a k-vector.
    pub fn apply(&self, x: &ArrayView1<f64>) -> LinalgResult<Array1<f64>> {
        if x.len() > self.n {
            return Err(LinalgError::DimensionError(format!(
                "SRHT::apply: x has length {} but n={}",
                x.len(),
                self.n
            )));
        }

        // Pad x to length n with zeros
        let mut buf = vec![0.0f64; self.n];
        for i in 0..x.len() {
            buf[i] = self.diag_signs[i] * x[i];
        }

        // In-place Walsh-Hadamard transform
        Self::fwht(&mut buf);

        // Sample k rows and scale
        let scale = self.scale / (self.n as f64).sqrt();
        let y: Array1<f64> = self.sampled_rows.iter().map(|&r| scale * buf[r]).collect();

        Ok(y)
    }

    /// Apply SRHT to each column of matrix A (applies SRHT row-wise to A).
    ///
    /// For A of shape (n, p), computes S * A of shape (k, p).
    pub fn apply_matrix(&self, a: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
        let (nrows, ncols) = a.dim();
        if nrows > self.n {
            return Err(LinalgError::DimensionError(format!(
                "SRHT::apply_matrix: A has {} rows but n={}",
                nrows, self.n
            )));
        }

        let scale = self.scale / (self.n as f64).sqrt();
        let mut result = Array2::zeros((self.k, ncols));

        for j in 0..ncols {
            // Apply D to column j (pad to n)
            let mut buf = vec![0.0f64; self.n];
            for i in 0..nrows {
                buf[i] = self.diag_signs[i] * a[[i, j]];
            }
            // FWHT
            Self::fwht(&mut buf);
            // Sample rows
            for (ki, &row) in self.sampled_rows.iter().enumerate() {
                result[[ki, j]] = scale * buf[row];
            }
        }

        Ok(result)
    }

    /// Fast Walsh-Hadamard Transform (in-place, unnormalized).
    ///
    /// Requires `data.len()` to be a power of 2.
    fn fwht(data: &mut [f64]) {
        let n = data.len();
        let mut h = 1;
        while h < n {
            let mut i = 0;
            while i < n {
                for j in i..(i + h) {
                    let u = data[j];
                    let v = data[j + h];
                    data[j] = u + v;
                    data[j + h] = u - v;
                }
                i += 2 * h;
            }
            h *= 2;
        }
    }
}

// ============================================================================
// Result type
// ============================================================================

/// Result of a randomized preconditioned solve
#[derive(Debug, Clone)]
pub struct RandomizedPrecondResult {
    /// Computed solution vector x
    pub solution: Array1<f64>,
    /// Relative residual ‖Ax - b‖₂ / ‖b‖₂
    pub relative_residual: f64,
    /// Number of iterations used
    pub iterations: usize,
    /// Whether the solver converged
    pub converged: bool,
    /// Condition number estimate of preconditioned system
    pub preconditioned_cond_estimate: Option<f64>,
}

// ============================================================================
// Blendenpik Preconditioner
// ============================================================================

/// Blendenpik randomized preconditioner for overdetermined least squares.
///
/// Constructs an upper triangular preconditioner R such that A R⁻¹ is
/// well-conditioned, then solves min ‖A x - b‖ by iterating on
/// min ‖(A R⁻¹) y - b‖ with y = R x.
///
/// Algorithm:
/// 1. Sketch: B = S A, where S is a SRHT or sparse sign matrix, B is (k × n)
/// 2. QR: B = Q R (thin QR of k × n matrix)
/// 3. Solve the preconditioned system (A R⁻¹) y ≈ b using LSQR / CG-normal
/// 4. Recover x = R⁻¹ y
#[derive(Debug, Clone)]
pub struct BlendenpikPreconditioner {
    /// Upper triangular preconditioner R (n × n)
    pub r_factor: Array2<f64>,
    /// Size n (number of columns of A)
    pub n: usize,
    /// Sketch dimension used
    pub sketch_dim: usize,
    /// Condition number estimate
    pub cond_estimate: f64,
}

impl BlendenpikPreconditioner {
    /// Build the Blendenpik preconditioner from matrix A.
    ///
    /// # Arguments
    ///
    /// * `a` - Overdetermined matrix (m × n, m ≥ n)
    /// * `oversampling` - Sketch oversampling factor (default: 2, sketch_dim = oversampling * n)
    /// * `use_srht` - Use SRHT if true, sparse sign otherwise
    /// * `seed` - Optional RNG seed
    pub fn new(
        a: &ArrayView2<f64>,
        oversampling: Option<f64>,
        use_srht: bool,
        seed: Option<u64>,
    ) -> LinalgResult<Self> {
        let (m, n) = a.dim();
        if m < n {
            return Err(LinalgError::ShapeError(format!(
                "Blendenpik requires m ≥ n, got {}×{}",
                m, n
            )));
        }

        let alpha = oversampling.unwrap_or(2.0).max(1.0);
        let sketch_dim = ((alpha * n as f64) as usize).max(n + 1).min(m);

        // Sketch A
        let sketch_a = if use_srht {
            let srht = SubsampledRandomizedHadamard::new(m, sketch_dim, seed)?;
            srht.apply_matrix(a)?
        } else {
            let sparse = SparseSign::new(sketch_dim, m, None, seed)?;
            sparse.apply_matrix(a)?
        };

        // Thin QR of sketch: sketch_a = Q R
        let (_, r_factor) = qr(&sketch_a.view(), None)?;

        // Truncate R to n × n
        let r_n = r_factor.nrows().min(n);
        let r_square = r_factor
            .slice(scirs2_core::ndarray::s![..r_n, ..n])
            .to_owned();

        // Estimate condition number via ratio of largest/smallest |diagonal|
        let diag_abs: Vec<f64> = (0..r_n).map(|i| r_square[[i, i]].abs()).collect();
        let r_max = diag_abs.iter().cloned().fold(0.0_f64, f64::max);
        let r_min = diag_abs
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            .max(1e-300);
        let cond_estimate = if r_max > 0.0 { r_max / r_min } else { 1e16 };

        Ok(Self {
            r_factor: r_square,
            n: r_n,
            sketch_dim,
            cond_estimate,
        })
    }

    /// Apply R⁻¹ to a vector: solve R x = v via back substitution.
    pub fn apply_r_inverse(&self, v: &ArrayView1<f64>) -> LinalgResult<Array1<f64>> {
        let n = self.n;
        if v.len() != n {
            return Err(LinalgError::DimensionError(format!(
                "BlendenpikPreconditioner::apply_r_inverse: v has {} elements but n={}",
                v.len(),
                n
            )));
        }

        // Back substitution: R x = v
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = v[i];
            for j in (i + 1)..n {
                sum -= self.r_factor[[i, j]] * x[j];
            }
            let diag = self.r_factor[[i, i]];
            if diag.abs() < 1e-300 {
                return Err(LinalgError::SingularMatrixError(format!(
                    "Blendenpik R is singular at diagonal index {i}"
                )));
            }
            x[i] = sum / diag;
        }

        Ok(x)
    }

    /// Apply R (forward multiplication): y = R * x
    pub fn apply_r(&self, x: &ArrayView1<f64>) -> LinalgResult<Array1<f64>> {
        let n = self.n;
        if x.len() != n {
            return Err(LinalgError::DimensionError(format!(
                "BlendenpikPreconditioner::apply_r: x has {} elements but n={}",
                x.len(),
                n
            )));
        }

        let mut y = Array1::zeros(n);
        for i in 0..n {
            let mut val = 0.0f64;
            for j in i..n {
                val += self.r_factor[[i, j]] * x[j];
            }
            y[i] = val;
        }

        Ok(y)
    }

    /// Solve the preconditioned normal equations using LSQR-style CG on normal equations.
    ///
    /// Minimizes ‖A x - b‖ using R as right preconditioner.
    ///
    /// # Arguments
    ///
    /// * `a` - Matrix A (m × n)
    /// * `b` - Right-hand side (m-vector)
    /// * `max_iter` - Maximum iterations
    /// * `tol` - Convergence tolerance
    pub fn solve(
        &self,
        a: &ArrayView2<f64>,
        b: &ArrayView1<f64>,
        max_iter: Option<usize>,
        tol: Option<f64>,
    ) -> LinalgResult<RandomizedPrecondResult> {
        let (m, n_a) = a.dim();
        if m != b.len() {
            return Err(LinalgError::DimensionError(format!(
                "BlendenpikPreconditioner::solve: A has {} rows but b has {} elements",
                m,
                b.len()
            )));
        }

        let max_it = max_iter.unwrap_or(200);
        let tolerance = tol.unwrap_or(1e-10);

        // Solve preconditioned normal equations via LSQR on (A R⁻¹) y = b
        // Using CG on normal equations: (R⁻ᵀ Aᵀ A R⁻¹) y = R⁻ᵀ Aᵀ b
        let n = self.n.min(n_a);

        // Compute Aᵀb
        let mut atb = Array1::zeros(n);
        for i in 0..n {
            let mut val = 0.0f64;
            for k in 0..m {
                val += a[[k, i]] * b[k];
            }
            atb[i] = val;
        }

        // Initial solution: y = 0, x = R⁻¹ y = 0
        let mut y = Array1::zeros(n);

        // Compute R⁻ᵀ Aᵀ b = gradient at y=0
        // R⁻ᵀ z means solve Rᵀ g = z
        let mut g = self.apply_r_transpose_inverse(&atb.view())?;

        // Preconditioned residual: r = R⁻ᵀ Aᵀ b - R⁻ᵀ Aᵀ A R⁻¹ y = g (since y=0)
        let mut p = g.clone();
        let mut rr = dot_product(&g.view(), &g.view());

        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..max_it {
            iterations = iter + 1;

            // Compute A R⁻¹ p
            let rp = self.apply_r_inverse(&p.view())?;
            // A * rp
            let mut arp = Array1::zeros(m);
            for i in 0..m {
                let mut val = 0.0f64;
                for j in 0..n.min(n_a) {
                    val += a[[i, j]] * rp[j];
                }
                arp[i] = val;
            }
            // Aᵀ * arp
            let mut aarp = Array1::zeros(n);
            for i in 0..n {
                let mut val = 0.0f64;
                for k in 0..m {
                    val += a[[k, i]] * arp[k];
                }
                aarp[i] = val;
            }
            // R⁻ᵀ * aarp
            let q = self.apply_r_transpose_inverse(&aarp.view())?;

            let pq = dot_product(&p.view(), &q.view());
            if pq.abs() < 1e-300 {
                break;
            }

            let alpha = rr / pq;
            y = y + alpha * &p;
            g = g - alpha * &q;

            let rr_new = dot_product(&g.view(), &g.view());
            let beta = rr_new / rr.max(1e-300);
            rr = rr_new;

            if rr.sqrt() < tolerance {
                converged = true;
                break;
            }

            p = &g + beta * &p;
        }

        // Recover x = R⁻¹ y
        let solution = self.apply_r_inverse(&y.view())?;

        // Compute residual
        let mut ax = Array1::zeros(m);
        for i in 0..m {
            let mut val = 0.0f64;
            for j in 0..n.min(n_a) {
                val += a[[i, j]] * solution[j];
            }
            ax[i] = val;
        }
        let residual = (&ax - b).mapv(|v: f64| v * v).sum().sqrt();
        let b_norm = b.mapv(|v: f64| v * v).sum().sqrt().max(1e-300);
        let relative_residual = residual / b_norm;

        Ok(RandomizedPrecondResult {
            solution,
            relative_residual,
            iterations,
            converged,
            preconditioned_cond_estimate: Some(self.cond_estimate),
        })
    }

    /// Apply R⁻ᵀ (solve Rᵀ x = v via forward substitution).
    fn apply_r_transpose_inverse(&self, v: &ArrayView1<f64>) -> LinalgResult<Array1<f64>> {
        let n = self.n;
        if v.len() != n {
            return Err(LinalgError::DimensionError(format!(
                "apply_r_transpose_inverse: v has {} elements but n={}",
                v.len(),
                n
            )));
        }

        let mut x = Array1::zeros(n);
        // Rᵀ is lower triangular; forward substitution
        for i in 0..n {
            let mut sum = v[i];
            for j in 0..i {
                sum -= self.r_factor[[j, i]] * x[j];
            }
            let diag = self.r_factor[[i, i]];
            if diag.abs() < 1e-300 {
                return Err(LinalgError::SingularMatrixError(format!(
                    "Blendenpik R^T is singular at index {i}"
                )));
            }
            x[i] = sum / diag;
        }

        Ok(x)
    }
}

// ============================================================================
// LSRN: Least Squares via Random Normal preconditioning
// ============================================================================

/// LSRN: Least-Squares via Random Normal preconditioning.
///
/// Uses a standard Gaussian sketch S (k × m, k = oversampling * n) to construct
/// a preconditioner via SVD of S*A, then solves the preconditioned system
/// iteratively. Particularly suited for highly overdetermined systems.
///
/// Algorithm:
/// 1. Sample S ~ N(0, 1/k) of shape (k × m)
/// 2. Compute B = S A (k × n)
/// 3. SVD: B = U Σ Vᵀ
/// 4. Preconditioner: N = V Σ⁻¹ (n × k, effective right preconditioner)
/// 5. Solve preconditioned system (A N) y = b via LSQR
/// 6. Recover x = N y
#[derive(Debug, Clone)]
pub struct LSRNPreconditioner {
    /// Right preconditioner factor V (n × rank)
    pub v_factor: Array2<f64>,
    /// Inverse singular values (length rank)
    pub sigma_inv: Array1<f64>,
    /// Number of columns of A
    pub n: usize,
    /// Effective rank used
    pub rank: usize,
    /// Sketch dimension k
    pub sketch_dim: usize,
}

impl LSRNPreconditioner {
    /// Build the LSRN preconditioner from matrix A.
    ///
    /// # Arguments
    ///
    /// * `a` - Overdetermined matrix (m × n, m ≥ n)
    /// * `oversampling` - Sketch oversampling factor (default: 2)
    /// * `rcond` - Relative threshold for truncated SVD (default: 1e-12)
    /// * `seed` - Optional RNG seed
    pub fn new(
        a: &ArrayView2<f64>,
        oversampling: Option<f64>,
        rcond: Option<f64>,
        seed: Option<u64>,
    ) -> LinalgResult<Self> {
        let (m, n) = a.dim();
        if m < n {
            return Err(LinalgError::ShapeError(format!(
                "LSRN requires m ≥ n, got {}×{}",
                m, n
            )));
        }

        let alpha = oversampling.unwrap_or(2.0).max(1.0);
        let k = ((alpha * n as f64) as usize).max(n + 1).min(m);
        let threshold = rcond.unwrap_or(1e-12);

        // Generate Gaussian sketch S of shape (k × m)
        let s_matrix = gaussian_random_matrix(k, m, seed)?;

        // Compute B = S * A (k × n)
        let mut b_sketch = Array2::zeros((k, n));
        let scale = 1.0 / (k as f64).sqrt();
        for i in 0..k {
            for j in 0..n {
                let mut val = 0.0f64;
                for l in 0..m {
                    val += s_matrix[[i, l]] * a[[l, j]];
                }
                b_sketch[[i, j]] = scale * val;
            }
        }

        // Thin SVD of B
        let (_, sigma, vt) = crate::decomposition::svd(&b_sketch.view(), true, None)?;

        let sigma_max = sigma[0].max(1e-300);
        let tol = threshold * sigma_max;

        // Determine effective rank
        let rank = sigma.iter().filter(|&&s| s > tol).count().max(1);
        let rank = rank.min(n);

        // Build preconditioner: N = V * Sigma^{-1}
        // Vᵀ has shape (rank × n); V has shape (n × rank)
        let v_factor = vt
            .slice(scirs2_core::ndarray::s![..rank, ..])
            .t()
            .to_owned();
        let sigma_inv: Array1<f64> =
            sigma
                .slice(scirs2_core::ndarray::s![..rank])
                .mapv(|s| if s > tol { 1.0 / s } else { 0.0 });

        Ok(Self {
            v_factor,
            sigma_inv,
            n,
            rank,
            sketch_dim: k,
        })
    }

    /// Apply the preconditioner N = V Σ⁻¹: y = N * x (n-vector output from rank-vector input).
    pub fn apply_n(&self, x: &ArrayView1<f64>) -> LinalgResult<Array1<f64>> {
        if x.len() != self.rank {
            return Err(LinalgError::DimensionError(format!(
                "LSRN::apply_n: x has {} elements but rank={}",
                x.len(),
                self.rank
            )));
        }

        // N = V * Sigma^{-1}; apply: first scale by sigma_inv, then multiply by V
        let mut scaled = Array1::zeros(self.rank);
        for i in 0..self.rank {
            scaled[i] = self.sigma_inv[i] * x[i];
        }

        let mut result = Array1::zeros(self.n);
        for i in 0..self.n {
            let mut val = 0.0f64;
            for j in 0..self.rank {
                val += self.v_factor[[i, j]] * scaled[j];
            }
            result[i] = val;
        }

        Ok(result)
    }

    /// Apply Nᵀ = Σ⁻¹ Vᵀ: y = Nᵀ * x (rank-vector output from n-vector input).
    pub fn apply_n_transpose(&self, x: &ArrayView1<f64>) -> LinalgResult<Array1<f64>> {
        if x.len() != self.n {
            return Err(LinalgError::DimensionError(format!(
                "LSRN::apply_n_transpose: x has {} elements but n={}",
                x.len(),
                self.n
            )));
        }

        // Nᵀ x = Σ⁻¹ Vᵀ x
        let mut vt_x = Array1::zeros(self.rank);
        for i in 0..self.rank {
            let mut val = 0.0f64;
            for j in 0..self.n {
                val += self.v_factor[[j, i]] * x[j];
            }
            vt_x[i] = self.sigma_inv[i] * val;
        }

        Ok(vt_x)
    }

    /// Solve the least-squares problem min ‖A x - b‖ using LSRN preconditioning.
    ///
    /// Uses LSQR on the preconditioned system min ‖(A N) y - b‖, then x = N y.
    pub fn solve(
        &self,
        a: &ArrayView2<f64>,
        b: &ArrayView1<f64>,
        max_iter: Option<usize>,
        tol: Option<f64>,
    ) -> LinalgResult<RandomizedPrecondResult> {
        let (m, n) = a.dim();
        if b.len() != m {
            return Err(LinalgError::DimensionError(format!(
                "LSRN::solve: A has {} rows but b has {} elements",
                m,
                b.len()
            )));
        }

        let max_it = max_iter.unwrap_or(300);
        let tolerance = tol.unwrap_or(1e-10);
        let rank = self.rank;

        // Compute Aᵀb
        let mut atb = Array1::zeros(n);
        for i in 0..n {
            let mut val = 0.0f64;
            for k in 0..m {
                val += a[[k, i]] * b[k];
            }
            atb[i] = val;
        }

        // Transform: Nᵀ Aᵀ b
        let nt_atb = self.apply_n_transpose(&atb.view())?;

        // CG on normal equations in y-space: (Nᵀ Aᵀ A N) y = Nᵀ Aᵀ b
        let mut y = Array1::zeros(rank);
        let mut g = nt_atb.clone();
        let mut p = g.clone();
        let mut rr = dot_product(&g.view(), &g.view());

        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..max_it {
            iterations = iter + 1;

            // q = Nᵀ Aᵀ A N p
            let np = self.apply_n(&p.view())?;
            // A * np
            let mut anp = Array1::zeros(m);
            for i in 0..m {
                let mut val = 0.0f64;
                for j in 0..n.min(self.n) {
                    val += a[[i, j]] * np[j];
                }
                anp[i] = val;
            }
            // Aᵀ * anp
            let mut at_anp = Array1::zeros(n);
            for i in 0..n {
                let mut val = 0.0f64;
                for k in 0..m {
                    val += a[[k, i]] * anp[k];
                }
                at_anp[i] = val;
            }
            // Nᵀ * at_anp
            let q = self.apply_n_transpose(&at_anp.view())?;

            let pq = dot_product(&p.view(), &q.view());
            if pq.abs() < 1e-300 {
                break;
            }

            let alpha = rr / pq;
            y = y + alpha * &p;
            g = g - alpha * &q;

            let rr_new = dot_product(&g.view(), &g.view());
            let beta = rr_new / rr.max(1e-300);
            rr = rr_new;

            if rr.sqrt() < tolerance {
                converged = true;
                break;
            }

            p = &g + beta * &p;
        }

        // Recover x = N y
        let solution = self.apply_n(&y.view())?;

        // Compute residual
        let mut ax = Array1::zeros(m);
        for i in 0..m {
            let mut val = 0.0f64;
            for j in 0..n.min(self.n) {
                val += a[[i, j]] * solution[j];
            }
            ax[i] = val;
        }
        let residual = (&ax - b).mapv(|v: f64| v * v).sum().sqrt();
        let b_norm = b.mapv(|v: f64| v * v).sum().sqrt().max(1e-300);
        let relative_residual = residual / b_norm;

        Ok(RandomizedPrecondResult {
            solution,
            relative_residual,
            iterations,
            converged,
            preconditioned_cond_estimate: None,
        })
    }
}

// ============================================================================
// IterativeRefinementLS
// ============================================================================

/// Iterative refinement for least-squares with a random preconditioner.
///
/// Combines Blendenpik/LSRN preconditioning with iterative refinement steps
/// to achieve high-accuracy solutions even for ill-conditioned problems.
///
/// Algorithm:
/// 1. Compute initial solution x₀ using Blendenpik/LSRN
/// 2. Compute residual r = b - A x₀
/// 3. Solve the correction problem: x₁ = x₀ + δ where δ minimizes ‖A δ - r‖
/// 4. Repeat until residual tolerance is met
#[derive(Debug, Clone)]
pub struct IterativeRefinementLS {
    /// Maximum number of refinement steps
    pub max_refinement_steps: usize,
    /// Convergence tolerance for each inner solve
    pub inner_tol: f64,
    /// Overall convergence tolerance
    pub outer_tol: f64,
    /// Use Blendenpik (true) or LSRN (false) as inner preconditioner
    pub use_blendenpik: bool,
    /// Oversampling factor for the preconditioner
    pub oversampling: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl IterativeRefinementLS {
    /// Create with default parameters.
    pub fn new() -> Self {
        Self {
            max_refinement_steps: 3,
            inner_tol: 1e-8,
            outer_tol: 1e-12,
            use_blendenpik: true,
            oversampling: 2.0,
            seed: None,
        }
    }

    /// Set the maximum number of refinement steps.
    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_refinement_steps = steps;
        self
    }

    /// Set inner solver tolerance.
    pub fn with_inner_tol(mut self, tol: f64) -> Self {
        self.inner_tol = tol;
        self
    }

    /// Set outer convergence tolerance.
    pub fn with_outer_tol(mut self, tol: f64) -> Self {
        self.outer_tol = tol;
        self
    }

    /// Use LSRN instead of Blendenpik.
    pub fn with_lsrn(mut self) -> Self {
        self.use_blendenpik = false;
        self
    }

    /// Set oversampling factor.
    pub fn with_oversampling(mut self, alpha: f64) -> Self {
        self.oversampling = alpha;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Solve min ‖A x - b‖ with iterative refinement.
    pub fn solve(
        &self,
        a: &ArrayView2<f64>,
        b: &ArrayView1<f64>,
    ) -> LinalgResult<RandomizedPrecondResult> {
        let (m, n) = a.dim();
        if b.len() != m {
            return Err(LinalgError::DimensionError(format!(
                "IterativeRefinementLS: A has {} rows but b has {} elements",
                m,
                b.len()
            )));
        }

        // Compute initial solution
        let mut result = if self.use_blendenpik {
            let precond =
                BlendenpikPreconditioner::new(a, Some(self.oversampling), true, self.seed)?;
            precond.solve(a, b, Some(200), Some(self.inner_tol))?
        } else {
            let precond = LSRNPreconditioner::new(a, Some(self.oversampling), None, self.seed)?;
            precond.solve(a, b, Some(300), Some(self.inner_tol))?
        };

        let mut x = result.solution.clone();
        let mut total_iters = result.iterations;

        // Iterative refinement loop
        for step in 0..self.max_refinement_steps {
            let rel_res = result.relative_residual;
            if rel_res < self.outer_tol {
                result.converged = true;
                break;
            }

            // Compute residual r = b - A x
            let mut r = Array1::zeros(m);
            for i in 0..m {
                let mut ax_i = 0.0f64;
                for j in 0..n {
                    ax_i += a[[i, j]] * x[j];
                }
                r[i] = b[i] - ax_i;
            }

            // Solve correction: min ‖A δ - r‖
            let correction_result = if self.use_blendenpik {
                // Reuse the preconditioner with a different seed
                let new_seed = self.seed.map(|s| s + step as u64 + 1);
                let precond =
                    BlendenpikPreconditioner::new(a, Some(self.oversampling), true, new_seed)?;
                precond.solve(a, &r.view(), Some(100), Some(self.inner_tol * 0.1))?
            } else {
                let new_seed = self.seed.map(|s| s + step as u64 + 1);
                let precond = LSRNPreconditioner::new(a, Some(self.oversampling), None, new_seed)?;
                precond.solve(a, &r.view(), Some(150), Some(self.inner_tol * 0.1))?
            };

            x += &correction_result.solution;
            total_iters += correction_result.iterations;

            // Recompute relative residual
            let mut ax = Array1::zeros(m);
            for i in 0..m {
                let mut val = 0.0f64;
                for j in 0..n {
                    val += a[[i, j]] * x[j];
                }
                ax[i] = val;
            }
            let res_norm = (&ax - b).mapv(|v: f64| v * v).sum().sqrt();
            let b_norm = b.mapv(|v: f64| v * v).sum().sqrt().max(1e-300);
            result.relative_residual = res_norm / b_norm;
        }

        result.solution = x;
        result.iterations = total_iters;

        Ok(result)
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Dot product of two vectors
fn dot_product(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Generate a standard Gaussian random matrix (rows × cols)
fn gaussian_random_matrix(
    rows: usize,
    cols: usize,
    seed: Option<u64>,
) -> LinalgResult<Array2<f64>> {
    let mut rng = {
        let seed_val = seed.unwrap_or_else(|| {
            let mut tr = scirs2_core::random::rng();
            scirs2_core::random::Rng::random::<u64>(&mut tr)
        });
        scirs2_core::random::seeded_rng(seed_val)
    };

    let normal = Normal::new(0.0, 1.0).map_err(|e| {
        LinalgError::ComputationError(format!("Failed to create Normal distribution: {e}"))
    })?;

    let mut matrix = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            matrix[[i, j]] = normal.sample(&mut rng);
        }
    }

    Ok(matrix)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_sparse_sign_dimensions() {
        let ss = SparseSign::new(20, 10, Some(3), Some(42)).expect("SparseSign::new failed");
        assert_eq!(ss.rows, 20);
        assert_eq!(ss.cols, 10);
        assert_eq!(ss.sparsity, 3);

        let x = Array1::ones(10);
        let y = ss.apply(&x.view()).expect("apply failed");
        assert_eq!(y.len(), 20);
    }

    #[test]
    fn test_sparse_sign_linearity() {
        let ss = SparseSign::new(30, 15, Some(4), Some(7)).expect("SparseSign::new failed");
        let x1 = Array1::from_vec((0..15).map(|i| i as f64).collect());
        let x2 = Array1::from_vec((0..15).map(|i| (i + 1) as f64).collect());

        let y1 = ss.apply(&x1.view()).expect("apply x1 failed");
        let y2 = ss.apply(&x2.view()).expect("apply x2 failed");
        let y12 = ss.apply(&(&x1 + &x2).view()).expect("apply x1+x2 failed");

        for i in 0..30 {
            assert!((y12[i] - y1[i] - y2[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_srht_dimensions() {
        let srht = SubsampledRandomizedHadamard::new(64, 16, Some(99)).expect("SRHT::new failed");
        let x = Array1::ones(64);
        let y = srht.apply(&x.view()).expect("SRHT apply failed");
        assert_eq!(y.len(), 16);
    }

    #[test]
    fn test_blendenpik_overdetermined() {
        // Build a well-conditioned overdetermined system
        let m = 20;
        let n = 5;
        let mut a = Array2::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                a[[i, j]] = ((i + j + 1) as f64).cos() + if i == j { 2.0 } else { 0.0 };
            }
        }
        let x_true = array![1.0, -1.0, 2.0, 0.5, -0.5];
        let mut b = Array1::zeros(m);
        for i in 0..m {
            for j in 0..n {
                b[i] += a[[i, j]] * x_true[j];
            }
        }

        let precond = BlendenpikPreconditioner::new(&a.view(), Some(2.0), false, Some(1))
            .expect("Blendenpik::new failed");
        let result = precond
            .solve(&a.view(), &b.view(), Some(100), Some(1e-8))
            .expect("Blendenpik::solve failed");

        assert!(
            result.relative_residual < 1e-4,
            "Blendenpik residual too large: {}",
            result.relative_residual
        );
    }

    #[test]
    fn test_lsrn_overdetermined() {
        let m = 30;
        let n = 6;
        let mut a = Array2::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                a[[i, j]] = 0.5 * ((i * n + j) as f64 / (m * n) as f64);
                if i == j {
                    a[[i, j]] += 3.0;
                }
            }
        }
        let x_true = array![1.0, 2.0, -1.0, 0.5, -2.0, 0.0];
        let mut b = Array1::zeros(m);
        for i in 0..m {
            for j in 0..n {
                b[i] += a[[i, j]] * x_true[j];
            }
        }

        let precond = LSRNPreconditioner::new(&a.view(), Some(2.0), Some(1e-12), Some(42))
            .expect("LSRN::new failed");
        let result = precond
            .solve(&a.view(), &b.view(), Some(200), Some(1e-8))
            .expect("LSRN::solve failed");

        assert!(
            result.relative_residual < 1e-3,
            "LSRN residual too large: {}",
            result.relative_residual
        );
    }

    #[test]
    fn test_iterative_refinement_ls() {
        let m = 25;
        let n = 5;
        let mut a = Array2::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                a[[i, j]] = (i as f64 * 0.1 + j as f64 * 0.3 + 1.0).sin();
                if i == j && j < n {
                    a[[i, j]] += 2.0;
                }
            }
        }
        let x_true = Array1::from_vec(vec![1.0, -1.0, 0.5, -0.5, 2.0]);
        let mut b = Array1::zeros(m);
        for i in 0..m {
            for j in 0..n {
                b[i] += a[[i, j]] * x_true[j];
            }
        }

        let solver = IterativeRefinementLS::new()
            .with_max_steps(2)
            .with_outer_tol(1e-6)
            .with_seed(123);

        let result = solver
            .solve(&a.view(), &b.view())
            .expect("IterativeRefinementLS::solve failed");

        assert!(
            result.relative_residual < 1e-3,
            "IterativeRefinementLS residual too large: {}",
            result.relative_residual
        );
    }
}
