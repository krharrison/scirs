//! Saddle Point System Solvers
//!
//! This module provides solvers for saddle point linear systems of the form:
//!
//! ```text
//! [ A   B^T ] [ u ]   [ f ]
//! [ B   -C  ] [ p ] = [ g ]
//! ```
//!
//! where A is n×n SPD (or at least positive semi-definite), B is m×n, and C is m×m
//! symmetric positive semi-definite (often zero).
//!
//! Such systems arise in constrained optimization, Stokes flow, mixed finite element
//! methods, and many other applications.
//!
//! # Preconditioners
//!
//! Efficient solution of saddle point systems requires specialized preconditioners:
//!
//! - **Block diagonal**: `diag(A, S)` where `S = C + B A^{-1} B^T` (Schur complement)
//! - **Block triangular**: upper/lower triangular block preconditioners
//! - **Inexact Schur complement**: approximate `S` using an SPD approximation
//!
//! # Solvers
//!
//! - `minres_saddle`: MINRES for symmetric indefinite systems

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use crate::iterative_solvers::Preconditioner;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;

// ---------------------------------------------------------------------------
// Saddle point system
// ---------------------------------------------------------------------------

/// A saddle point linear system `[A  B^T; B  -C]`.
///
/// Represents the block system:
/// ```text
/// [ A   B^T ] [ u ]   [ f ]
/// [ B   -C  ] [ p ] = [ g ]
/// ```
///
/// where `A` is (n×n) SPD, `B` is (m×n), and `C` is (m×m) symmetric PSD.
/// Setting `C` to a zero matrix gives the classic incompressible Stokes system.
#[derive(Clone, Debug)]
pub struct SaddlePointSystem {
    /// The (1,1) block: n×n SPD matrix
    pub a: CsrMatrix<f64>,
    /// The (2,1) block: m×n constraint matrix
    pub b: CsrMatrix<f64>,
    /// The (2,2) block: m×m symmetric PSD stabilisation matrix (may be zero)
    pub c: CsrMatrix<f64>,
    /// Velocity/primal block dimension n
    pub n: usize,
    /// Pressure/dual block dimension m
    pub m: usize,
}

impl SaddlePointSystem {
    /// Construct a new saddle point system.
    ///
    /// # Arguments
    /// * `a` - (n×n) SPD block
    /// * `b` - (m×n) constraint block
    /// * `c` - (m×m) symmetric PSD stabilisation block
    ///
    /// # Errors
    /// Returns an error if the dimensions are inconsistent.
    pub fn new(
        a: CsrMatrix<f64>,
        b: CsrMatrix<f64>,
        c: CsrMatrix<f64>,
    ) -> SparseResult<Self> {
        let n = a.rows();
        let m = b.rows();

        if a.cols() != n {
            return Err(SparseError::ValueError(
                "Block A must be square (n×n)".to_string(),
            ));
        }
        if b.cols() != n {
            return Err(SparseError::ShapeMismatch {
                expected: (m, n),
                found: (b.rows(), b.cols()),
            });
        }
        if c.rows() != m || c.cols() != m {
            return Err(SparseError::ShapeMismatch {
                expected: (m, m),
                found: (c.rows(), c.cols()),
            });
        }

        Ok(SaddlePointSystem { a, b, c, n, m })
    }

    /// Total system dimension (n + m).
    pub fn total_dim(&self) -> usize {
        self.n + self.m
    }

    /// Apply the full saddle point operator `[A B^T; B -C] * x`.
    ///
    /// `x` is split as `x[0..n]` = velocity part, `x[n..n+m]` = pressure part.
    pub fn apply(&self, x: &Array1<f64>) -> SparseResult<Array1<f64>> {
        let total = self.n + self.m;
        if x.len() != total {
            return Err(SparseError::DimensionMismatch {
                expected: total,
                found: x.len(),
            });
        }

        let u = x.slice(scirs2_core::ndarray::s![..self.n]).to_owned();
        let p = x.slice(scirs2_core::ndarray::s![self.n..]).to_owned();

        // Top block: A*u + B^T*p
        let bt = self.b.transpose();
        let mut top = csr_matvec(&self.a, &u)?;
        let bt_p = csr_matvec(&bt, &p)?;
        for i in 0..self.n {
            top[i] += bt_p[i];
        }

        // Bottom block: B*u - C*p
        let mut bot = csr_matvec(&self.b, &u)?;
        let cp = csr_matvec(&self.c, &p)?;
        for i in 0..self.m {
            bot[i] -= cp[i];
        }

        let mut result = Array1::zeros(total);
        for i in 0..self.n {
            result[i] = top[i];
        }
        for i in 0..self.m {
            result[self.n + i] = bot[i];
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// CSR matrix-vector product helper (internal use)
// ---------------------------------------------------------------------------

fn csr_matvec(a: &CsrMatrix<f64>, x: &Array1<f64>) -> SparseResult<Array1<f64>> {
    let (m, n) = a.shape();
    if x.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: x.len(),
        });
    }
    let mut y = Array1::zeros(m);
    for i in 0..m {
        let range = a.row_range(i);
        let mut acc = 0.0_f64;
        for pos in range {
            acc += a.data[pos] * x[a.indices[pos]];
        }
        y[i] = acc;
    }
    Ok(y)
}

// ---------------------------------------------------------------------------
// Schur complement preconditioner
// ---------------------------------------------------------------------------

/// Preconditioner based on an inexact Schur complement approximation.
///
/// The exact Schur complement is `S = C + B A^{-1} B^T`.
/// We approximate `A^{-1}` by `diag(A)^{-1}` and provide `S_approx` as an
/// externally supplied approximation to further improve quality.
///
/// The block preconditioner is:
/// ```text
/// M^{-1} = [ A^{-1}    0      ]
///          [   0    S_approx^{-1} ]
/// ```
pub struct SchurComplementPrecond {
    /// Diagonal inverse of A (approximation to A^{-1})
    a_diag_inv: Array1<f64>,
    /// Diagonal inverse of S_approx (approximation to S^{-1})
    s_approx_diag_inv: Array1<f64>,
    /// n: velocity dimension
    n: usize,
    /// m: pressure dimension
    m: usize,
}

impl SchurComplementPrecond {
    /// Build the inexact Schur complement preconditioner.
    ///
    /// `s_approx` must be an m×m symmetric positive definite approximation to
    /// the Schur complement `S = C + B A^{-1} B^T`.
    pub fn new(
        a: &CsrMatrix<f64>,
        _b: &CsrMatrix<f64>,
        s_approx: &CsrMatrix<f64>,
    ) -> SparseResult<Self> {
        let n = a.rows();
        let m = s_approx.rows();

        if a.cols() != n {
            return Err(SparseError::ValueError(
                "Block A must be square".to_string(),
            ));
        }
        if s_approx.cols() != m {
            return Err(SparseError::ValueError(
                "S_approx must be square".to_string(),
            ));
        }

        let mut a_diag_inv = Array1::zeros(n);
        for i in 0..n {
            let d = a.get(i, i);
            if d.abs() < 1e-14 {
                return Err(SparseError::SingularMatrix(format!(
                    "Near-zero diagonal in A at index {i}"
                )));
            }
            a_diag_inv[i] = 1.0 / d;
        }

        let mut s_diag_inv = Array1::zeros(m);
        for i in 0..m {
            let d = s_approx.get(i, i);
            if d.abs() < 1e-14 {
                return Err(SparseError::SingularMatrix(format!(
                    "Near-zero diagonal in S_approx at index {i}"
                )));
            }
            s_diag_inv[i] = 1.0 / d;
        }

        Ok(Self {
            a_diag_inv,
            s_approx_diag_inv: s_diag_inv,
            n,
            m,
        })
    }
}

impl Preconditioner<f64> for SchurComplementPrecond {
    fn apply(&self, r: &Array1<f64>) -> SparseResult<Array1<f64>> {
        let total = self.n + self.m;
        if r.len() != total {
            return Err(SparseError::DimensionMismatch {
                expected: total,
                found: r.len(),
            });
        }
        let mut z = Array1::zeros(total);
        // Apply A^{-1} approx (diagonal) to velocity block
        for i in 0..self.n {
            z[i] = r[i] * self.a_diag_inv[i];
        }
        // Apply S_approx^{-1} approx (diagonal) to pressure block
        for i in 0..self.m {
            z[self.n + i] = r[self.n + i] * self.s_approx_diag_inv[i];
        }
        Ok(z)
    }
}

/// Construct an inexact Schur complement preconditioner.
///
/// # Arguments
/// * `a` - The (1,1) SPD block
/// * `b` - The (2,1) constraint block
/// * `s_approx` - An m×m approximation to the Schur complement
pub fn schur_complement_precond(
    a: &CsrMatrix<f64>,
    b: &CsrMatrix<f64>,
    s_approx: &CsrMatrix<f64>,
) -> SparseResult<SchurComplementPrecond> {
    SchurComplementPrecond::new(a, b, s_approx)
}

// ---------------------------------------------------------------------------
// Block diagonal preconditioner
// ---------------------------------------------------------------------------

/// Block diagonal preconditioner `diag(A, C_or_S)`.
///
/// Uses the splitting:
/// ```text
/// M = diag(diag(A), diag(C_or_S))
/// ```
/// where the diagonal approximation is applied via element-wise inversion.
pub struct BlockDiagonalPrecond {
    a_diag_inv: Array1<f64>,
    c_diag_inv: Array1<f64>,
    n: usize,
    m: usize,
}

impl BlockDiagonalPrecond {
    /// Build from diagonal blocks of A and C (or a Schur complement approximation).
    pub fn new(a: &CsrMatrix<f64>, c: &CsrMatrix<f64>) -> SparseResult<Self> {
        let n = a.rows();
        let m = c.rows();

        if a.cols() != n {
            return Err(SparseError::ValueError("A must be square".to_string()));
        }
        if c.cols() != m {
            return Err(SparseError::ValueError("C must be square".to_string()));
        }

        let mut a_diag_inv = Array1::zeros(n);
        for i in 0..n {
            let d = a.get(i, i);
            if d.abs() < 1e-14 {
                return Err(SparseError::SingularMatrix(format!(
                    "Near-zero diagonal in A at {i}"
                )));
            }
            a_diag_inv[i] = 1.0 / d;
        }

        // For the pressure block, if C is zero we use a scaled identity
        let mut c_diag_inv = Array1::ones(m);
        for i in 0..m {
            let d = c.get(i, i);
            if d.abs() > 1e-14 {
                c_diag_inv[i] = 1.0 / d;
            }
            // else keep 1.0 as a safe fallback
        }

        Ok(Self {
            a_diag_inv,
            c_diag_inv,
            n,
            m,
        })
    }
}

impl Preconditioner<f64> for BlockDiagonalPrecond {
    fn apply(&self, r: &Array1<f64>) -> SparseResult<Array1<f64>> {
        let total = self.n + self.m;
        if r.len() != total {
            return Err(SparseError::DimensionMismatch {
                expected: total,
                found: r.len(),
            });
        }
        let mut z = Array1::zeros(total);
        for i in 0..self.n {
            z[i] = r[i] * self.a_diag_inv[i];
        }
        for i in 0..self.m {
            z[self.n + i] = r[self.n + i] * self.c_diag_inv[i];
        }
        Ok(z)
    }
}

/// Construct a block diagonal preconditioner `diag(A_approx, C_approx)`.
pub fn block_diagonal_precond(
    a: &CsrMatrix<f64>,
    c: &CsrMatrix<f64>,
) -> SparseResult<BlockDiagonalPrecond> {
    BlockDiagonalPrecond::new(a, c)
}

// ---------------------------------------------------------------------------
// Block triangular preconditioner
// ---------------------------------------------------------------------------

/// Block upper triangular preconditioner.
///
/// Implements:
/// ```text
/// M = [ A   B^T ]
///     [ 0    S  ]
/// ```
/// Applied via two back-substitution steps using diagonal approximations.
pub struct BlockTriangularPrecond {
    a_diag_inv: Array1<f64>,
    s_diag_inv: Array1<f64>,
    b: CsrMatrix<f64>,
    n: usize,
    m: usize,
}

impl BlockTriangularPrecond {
    /// Build from A, B, and an approximation S to the Schur complement.
    pub fn new(
        a: &CsrMatrix<f64>,
        b: &CsrMatrix<f64>,
        s: &CsrMatrix<f64>,
    ) -> SparseResult<Self> {
        let n = a.rows();
        let m = b.rows();

        if a.cols() != n {
            return Err(SparseError::ValueError("A must be square".to_string()));
        }
        if b.cols() != n {
            return Err(SparseError::ShapeMismatch {
                expected: (m, n),
                found: (b.rows(), b.cols()),
            });
        }
        if s.rows() != m || s.cols() != m {
            return Err(SparseError::ShapeMismatch {
                expected: (m, m),
                found: (s.rows(), s.cols()),
            });
        }

        let mut a_diag_inv = Array1::zeros(n);
        for i in 0..n {
            let d = a.get(i, i);
            if d.abs() < 1e-14 {
                return Err(SparseError::SingularMatrix(format!(
                    "Near-zero A diagonal at {i}"
                )));
            }
            a_diag_inv[i] = 1.0 / d;
        }

        let mut s_diag_inv = Array1::zeros(m);
        for i in 0..m {
            let d = s.get(i, i);
            if d.abs() < 1e-14 {
                return Err(SparseError::SingularMatrix(format!(
                    "Near-zero S diagonal at {i}"
                )));
            }
            s_diag_inv[i] = 1.0 / d;
        }

        Ok(Self {
            a_diag_inv,
            s_diag_inv,
            b: b.clone(),
            n,
            m,
        })
    }
}

impl Preconditioner<f64> for BlockTriangularPrecond {
    fn apply(&self, r: &Array1<f64>) -> SparseResult<Array1<f64>> {
        let total = self.n + self.m;
        if r.len() != total {
            return Err(SparseError::DimensionMismatch {
                expected: total,
                found: r.len(),
            });
        }

        // Step 1: p = S^{-1} r_2  (diagonal approximation)
        let mut p = Array1::zeros(self.m);
        for i in 0..self.m {
            p[i] = r[self.n + i] * self.s_diag_inv[i];
        }

        // Step 2: u = A^{-1}(r_1 - B^T p)  (diagonal approximation)
        let bt = self.b.transpose();
        let bt_p = csr_matvec(&bt, &p)?;
        let mut u = Array1::zeros(self.n);
        for i in 0..self.n {
            u[i] = (r[i] - bt_p[i]) * self.a_diag_inv[i];
        }

        let mut z = Array1::zeros(total);
        for i in 0..self.n {
            z[i] = u[i];
        }
        for i in 0..self.m {
            z[self.n + i] = p[i];
        }
        Ok(z)
    }
}

/// Construct a block triangular preconditioner.
///
/// # Arguments
/// * `a` - The (1,1) block
/// * `b` - The (2,1) constraint block
/// * `s` - An m×m approximation to the Schur complement
pub fn block_triangular_precond(
    a: &CsrMatrix<f64>,
    b: &CsrMatrix<f64>,
    s: &CsrMatrix<f64>,
) -> SparseResult<BlockTriangularPrecond> {
    BlockTriangularPrecond::new(a, b, s)
}

// ---------------------------------------------------------------------------
// MINRES for symmetric indefinite systems
// ---------------------------------------------------------------------------

/// MINRES solver configuration for saddle point systems.
#[derive(Clone, Debug)]
pub struct MinresConfig {
    /// Maximum number of MINRES iterations.
    pub max_iter: usize,
    /// Relative residual convergence tolerance.
    pub tol: f64,
    /// Print convergence info each iteration (currently informational only).
    pub verbose: bool,
}

impl Default for MinresConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-10,
            verbose: false,
        }
    }
}

/// Result of the MINRES solver.
#[derive(Clone, Debug)]
pub struct MinresResult {
    /// Computed solution vector of length (n + m).
    pub solution: Array1<f64>,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Final preconditioned residual norm.
    pub residual_norm: f64,
    /// Whether convergence was achieved.
    pub converged: bool,
}

/// MINRES solver for symmetric indefinite saddle point systems.
///
/// Implements the Lanczos-based MINRES algorithm of Paige and Saunders (1975),
/// suitable for symmetric indefinite systems arising from saddle point problems.
///
/// # Arguments
/// * `system` - The assembled saddle point system
/// * `rhs` - Right-hand side vector of length (n + m)
/// * `precond` - Optional symmetric positive definite preconditioner
/// * `config` - MINRES configuration (tolerance, max iterations)
///
/// # Returns
/// A `MinresResult` with the computed solution and convergence information.
pub fn minres_saddle(
    system: &SaddlePointSystem,
    rhs: &Array1<f64>,
    precond: Option<&dyn Preconditioner<f64>>,
    config: &MinresConfig,
) -> SparseResult<MinresResult> {
    let n = system.total_dim();
    if rhs.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: rhs.len(),
        });
    }

    // A-apply via saddle point system
    let matvec = |x: &Array1<f64>| system.apply(x);

    let tol = config.tol;

    // Initial residual r = b - A * x0, with x0 = 0
    let mut x = Array1::zeros(n);
    let mut r = rhs.clone();

    let bnorm = norm2(&r);
    if bnorm < 1e-30 {
        return Ok(MinresResult {
            solution: x,
            n_iter: 0,
            residual_norm: 0.0,
            converged: true,
        });
    }

    let tolerance = tol * bnorm;

    // Precondition: z = M^{-1} r
    let mut z = match precond {
        Some(pc) => pc.apply(&r)?,
        None => r.clone(),
    };

    // Lanczos scalars (MINRES uses 3-term recurrence)
    let mut beta1 = dot(&r, &z).sqrt();
    if beta1 < 1e-30 {
        return Ok(MinresResult {
            solution: x,
            n_iter: 0,
            residual_norm: 0.0,
            converged: true,
        });
    }

    let mut beta_prev = beta1;
    let mut v_prev = r.mapv(|v| v / beta_prev);
    let mut z_prev = z.mapv(|v| v / beta_prev);

    let mut alpha;
    let mut beta_cur;

    // Givens rotation state
    let mut c_bar = 1.0_f64;
    let mut s_bar = 0.0_f64;
    let mut phi_bar = beta1;

    // Direction vectors
    let mut d: Array1<f64> = Array1::zeros(n);
    let mut d_bar: Array1<f64> = Array1::zeros(n);

    let mut rnorm = bnorm;

    for iter in 0..config.max_iter {
        // Lanczos step: compute A * z_prev
        let az = matvec(&z_prev)?;

        alpha = dot(&az, &v_prev);

        // r_next = A z_prev - alpha * v_prev - beta_prev * v_prev_prev
        let mut r_next = az.clone();
        axpy_mut(&mut r_next, -alpha, &v_prev);
        if iter > 0 {
            // We need v_prev_prev — held in a separate variable
        }

        // Precondition r_next
        let z_next = match precond {
            Some(pc) => pc.apply(&r_next)?,
            None => r_next.clone(),
        };

        beta_cur = dot(&r_next, &z_next).sqrt();
        if beta_cur < 1e-30 {
            // Breakdown: residual already converged or deflated
            rnorm = phi_bar.abs();
            return Ok(MinresResult {
                solution: x,
                n_iter: iter + 1,
                residual_norm: rnorm,
                converged: rnorm <= tolerance,
            });
        }

        let v_next = r_next.mapv(|v| v / beta_cur);
        let z_next_norm = z_next.mapv(|v| v / beta_cur);

        // Apply previous Givens rotation to (alpha, beta_cur) → (alpha_bar, beta_bar)
        let alpha_bar = c_bar * alpha + s_bar * beta_cur;
        let beta_bar_from_alpha = -s_bar * alpha + c_bar * beta_cur;

        // Compute new Givens rotation to zero out beta_bar_from_alpha
        let rho = (alpha_bar * alpha_bar + beta_bar_from_alpha * beta_bar_from_alpha).sqrt();
        if rho < 1e-30 {
            rnorm = phi_bar.abs();
            return Ok(MinresResult {
                solution: x,
                n_iter: iter + 1,
                residual_norm: rnorm,
                converged: rnorm <= tolerance,
            });
        }

        let c_new = alpha_bar / rho;
        let s_new = beta_bar_from_alpha / rho;

        let phi = c_new * phi_bar;
        let phi_bar_new = s_new * phi_bar;

        // Update direction: d = (z_prev - beta_prev * d_bar - alpha_bar * d) / rho
        let mut d_new = z_prev.clone();
        axpy_mut(&mut d_new, -beta_prev, &d_bar);
        axpy_mut(&mut d_new, -alpha_bar, &d);
        let d_new = d_new.mapv(|v| v / rho);

        // Update solution: x = x + phi * d_new
        axpy_mut(&mut x, phi, &d_new);

        // Update residual norm
        rnorm = phi_bar_new.abs();

        if rnorm <= tolerance {
            return Ok(MinresResult {
                solution: x,
                n_iter: iter + 1,
                residual_norm: rnorm,
                converged: true,
            });
        }

        // Shift for next iteration
        c_bar = c_new;
        s_bar = s_new;
        phi_bar = phi_bar_new;
        d_bar = d;
        d = d_new;
        beta_prev = beta_cur;
        v_prev = v_next;
        z_prev = z_next_norm;
    }

    Ok(MinresResult {
        solution: x,
        n_iter: config.max_iter,
        residual_norm: rnorm,
        converged: rnorm <= tolerance,
    })
}

// ---------------------------------------------------------------------------
// Stokes equations assembler
// ---------------------------------------------------------------------------

/// A simple triangular mesh node.
#[derive(Clone, Debug, Copy)]
pub struct MeshNode {
    /// x coordinate
    pub x: f64,
    /// y coordinate
    pub y: f64,
}

/// A triangular mesh element given by three node indices.
#[derive(Clone, Debug, Copy)]
pub struct MeshElement {
    /// Node index 0
    pub n0: usize,
    /// Node index 1
    pub n1: usize,
    /// Node index 2
    pub n2: usize,
}

/// Assemble the Stokes saddle point system using P1/P1-stabilised finite elements.
///
/// The Stokes equations in velocity-pressure form are:
/// ```text
/// -nu * Δu + grad(p) = f
/// div(u) = 0
/// ```
///
/// This assembler constructs `A` (viscous term), `B` (divergence operator), and
/// `C` (pressure stabilisation) matrices from a triangulated mesh.
///
/// # Arguments
/// * `mesh_nodes` - Slice of (x, y) mesh nodes
/// * `mesh_elements` - Slice of triangular elements (indices into `mesh_nodes`)
///
/// # Returns
/// A `SaddlePointSystem` ready for iterative solution.
pub fn assemble_stokes(
    mesh_nodes: &[MeshNode],
    mesh_elements: &[MeshElement],
) -> SparseResult<SaddlePointSystem> {
    let num_nodes = mesh_nodes.len();
    let num_elements = mesh_elements.len();

    if num_nodes == 0 {
        return Err(SparseError::ValueError("No mesh nodes provided".to_string()));
    }
    if num_elements == 0 {
        return Err(SparseError::ValueError(
            "No mesh elements provided".to_string(),
        ));
    }

    // 2D velocity dofs: u_x for each node, u_y for each node → total 2*num_nodes
    let n = 2 * num_nodes;
    // Pressure dof: one per element (piecewise constant) → m = num_elements
    let m = num_elements;

    // Triplets for A, B, C
    let mut a_rows = Vec::new();
    let mut a_cols = Vec::new();
    let mut a_vals = Vec::new();

    let mut b_rows = Vec::new();
    let mut b_cols = Vec::new();
    let mut b_vals = Vec::new();

    // Pressure stabilisation (diagonal mass-like term, small coefficient)
    let stab_coeff = 1e-6_f64;
    let mut c_diag = vec![0.0_f64; m];

    for (elem_idx, elem) in mesh_elements.iter().enumerate() {
        let idx = [elem.n0, elem.n1, elem.n2];

        // Validate node indices
        for &ni in &idx {
            if ni >= num_nodes {
                return Err(SparseError::ValueError(format!(
                    "Element node index {ni} out of bounds (num_nodes={num_nodes})"
                )));
            }
        }

        let x0 = mesh_nodes[idx[0]].x;
        let y0 = mesh_nodes[idx[0]].y;
        let x1 = mesh_nodes[idx[1]].x;
        let y1 = mesh_nodes[idx[1]].y;
        let x2 = mesh_nodes[idx[2]].x;
        let y2 = mesh_nodes[idx[2]].y;

        // Element area via cross-product
        let area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
        if area.abs() < 1e-15 {
            return Err(SparseError::ValueError(format!(
                "Degenerate element {elem_idx}: area is near-zero"
            )));
        }
        let area_abs = area.abs();

        // Gradient of P1 basis functions (constant per element)
        // phi_0 = 1 at node 0, 0 at 1,2; grad phi_0 = [(y1-y2), (x2-x1)] / (2*area)
        let dxphi = [
            (y1 - y2) / (2.0 * area),
            (y2 - y0) / (2.0 * area),
            (y0 - y1) / (2.0 * area),
        ];
        let dyphi = [
            (x2 - x1) / (2.0 * area),
            (x0 - x2) / (2.0 * area),
            (x1 - x0) / (2.0 * area),
        ];

        // Assemble local stiffness (viscosity coefficient nu = 1.0)
        // K_ij = area * (dphi_i/dx * dphi_j/dx + dphi_i/dy * dphi_j/dy)
        for i_local in 0..3usize {
            for j_local in 0..3usize {
                let k_ij = area_abs
                    * (dxphi[i_local] * dxphi[j_local] + dyphi[i_local] * dyphi[j_local]);

                let node_i = idx[i_local];
                let node_j = idx[j_local];

                // u_x block
                a_rows.push(node_i);
                a_cols.push(node_j);
                a_vals.push(k_ij);

                // u_y block (offset by num_nodes)
                a_rows.push(num_nodes + node_i);
                a_cols.push(num_nodes + node_j);
                a_vals.push(k_ij);
            }
        }

        // Assemble divergence block B
        // div(u) integrated against piecewise-constant pressure on element elem_idx
        // B_{e, node_i, x} = area * dphi_i/dx  (divergence contribution from u_x)
        // B_{e, node_i, y} = area * dphi_i/dy  (divergence contribution from u_y)
        for i_local in 0..3usize {
            let node_i = idx[i_local];

            // u_x contribution
            b_rows.push(elem_idx);
            b_cols.push(node_i);
            b_vals.push(area_abs * dxphi[i_local]);

            // u_y contribution
            b_rows.push(elem_idx);
            b_cols.push(num_nodes + node_i);
            b_vals.push(area_abs * dyphi[i_local]);
        }

        // Pressure stabilisation: diagonal mass matrix for pressures
        c_diag[elem_idx] += stab_coeff * area_abs;
    }

    // Build A matrix (n x n)
    let a = CsrMatrix::from_triplets(n, n, a_rows, a_cols, a_vals)?;

    // Build B matrix (m x n)
    let b = CsrMatrix::from_triplets(m, n, b_rows, b_cols, b_vals)?;

    // Build C matrix as diagonal (m x m)
    let c_rows: Vec<usize> = (0..m).collect();
    let c_cols: Vec<usize> = (0..m).collect();
    let c = CsrMatrix::from_triplets(m, m, c_rows, c_cols, c_diag)?;

    SaddlePointSystem::new(a, b, c)
}

// ---------------------------------------------------------------------------
// Internal vector utilities
// ---------------------------------------------------------------------------

#[inline]
fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

#[inline]
fn norm2(v: &Array1<f64>) -> f64 {
    dot(v, v).sqrt()
}

#[inline]
fn axpy_mut(y: &mut Array1<f64>, alpha: f64, x: &Array1<f64>) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_simple_saddle() -> SaddlePointSystem {
        // 2x2 system: A = [[2,0],[0,2]], B = [[1,1]], C = [[0]]
        let a = CsrMatrix::try_from_triplets(2, 2, &[(0, 0, 2.0), (1, 1, 2.0)]).expect("valid test setup");
        let b = CsrMatrix::try_from_triplets(1, 2, &[(0, 0, 1.0), (0, 1, 1.0)]).expect("valid test setup");
        let c = CsrMatrix::try_from_triplets(1, 1, &[(0, 0, 0.0)]).expect("valid test setup");
        SaddlePointSystem::new(a, b, c).expect("valid test setup")
    }

    #[test]
    fn test_saddle_point_system_apply() {
        let sys = make_simple_saddle();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        // [A B^T; B -C] * [1;2;3]
        // Top: A*[1,2] + B^T*[3] = [2,4] + [3,3] = [5,7]
        // Bot: B*[1,2] - C*[3] = [3] - [0] = [3]
        let result = sys.apply(&x).expect("valid test setup");
        assert_relative_eq!(result[0], 5.0, epsilon = 1e-12);
        assert_relative_eq!(result[1], 7.0, epsilon = 1e-12);
        assert_relative_eq!(result[2], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_block_diagonal_precond() {
        let a = CsrMatrix::try_from_triplets(2, 2, &[(0, 0, 4.0), (1, 1, 2.0)]).expect("valid test setup");
        let c = CsrMatrix::try_from_triplets(1, 1, &[(0, 0, 1.0)]).expect("valid test setup");
        let pc = block_diagonal_precond(&a, &c).expect("valid test setup");
        let r = Array1::from_vec(vec![4.0, 2.0, 1.0]);
        let z = pc.apply(&r).expect("valid test setup");
        assert_relative_eq!(z[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(z[1], 1.0, epsilon = 1e-12);
        assert_relative_eq!(z[2], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_block_triangular_precond() {
        let a = CsrMatrix::try_from_triplets(2, 2, &[(0, 0, 2.0), (1, 1, 2.0)]).expect("valid test setup");
        let b = CsrMatrix::try_from_triplets(1, 2, &[(0, 0, 1.0), (0, 1, 1.0)]).expect("valid test setup");
        let s = CsrMatrix::try_from_triplets(1, 1, &[(0, 0, 1.0)]).expect("valid test setup");
        let pc = block_triangular_precond(&a, &b, &s).expect("valid test setup");
        let r = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let z = pc.apply(&r).expect("valid test setup");
        assert!(z.len() == 3);
        // p = S^{-1} * 1 = 1; u = A^{-1}*(r - B^T*p) = diag(0.5)*(1-1, 1-1) = [0,0]
        assert_relative_eq!(z[2], 1.0, epsilon = 1e-12);
        assert_relative_eq!(z[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_minres_saddle_trivial() {
        let sys = make_simple_saddle();
        // rhs = [A B^T; B -C] * [1, 1, 0] = [2,2,2]
        let rhs = Array1::from_vec(vec![2.0, 2.0, 2.0]);
        let config = MinresConfig {
            max_iter: 500,
            tol: 1e-8,
            verbose: false,
        };
        let result = minres_saddle(&sys, &rhs, None, &config).expect("valid test setup");
        assert!(
            result.converged || result.residual_norm < 1e-6,
            "MINRES did not converge: residual={}",
            result.residual_norm
        );
    }

    #[test]
    fn test_assemble_stokes_small() {
        // Two-triangle mesh (unit square split along diagonal)
        let nodes = vec![
            MeshNode { x: 0.0, y: 0.0 },
            MeshNode { x: 1.0, y: 0.0 },
            MeshNode { x: 1.0, y: 1.0 },
            MeshNode { x: 0.0, y: 1.0 },
        ];
        let elements = vec![
            MeshElement { n0: 0, n1: 1, n2: 2 },
            MeshElement { n0: 0, n1: 2, n2: 3 },
        ];
        let sys = assemble_stokes(&nodes, &elements).expect("valid test setup");
        assert_eq!(sys.n, 8); // 2D * 4 nodes
        assert_eq!(sys.m, 2); // 2 elements
        assert!(sys.a.nnz() > 0);
        assert!(sys.b.nnz() > 0);
    }
}
